"""S09b：DROID 评测任务读训练检查点（`evaluate.load_selected`，10 §4.2/§8.1 的放宽读法）和 `eval_job` 接线
（10 §8.1、§8.9，11；13 §8.3 的 `read_eval_view`）。

训练目录用 S09a 的合成训练根（test_droid_train）真训 U = 20 步的 matched_transformer；评测用 S11b 的合成测试根
（test_droid_evaluate_extra：两个测试集、附加特征、全帧率位姿）。两边都是假编码器，来源字典的数据字段相同。
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

from po_jepa import ac_baseline
from po_jepa.droid import checkpoints as ck
from po_jepa.droid import conditions as cd
from po_jepa.droid import eval_job as ej
from po_jepa.droid import evaluate as ev
from po_jepa.droid import train as tr
from po_jepa.droid.actions import droid_control_spec
from po_jepa.runtime import cache
from test_droid_evaluate import SCALE
import test_droid_evaluate_extra as tx
import test_droid_train as tt

spec = importlib.util.spec_from_file_location("droid_summary", Path(cache.ROOT)/"tools"/"droid_summary.py")
if "droid_summary" in sys.modules:
    ds = sys.modules["droid_summary"]
else:
    ds = importlib.util.module_from_spec(spec)
    sys.modules["droid_summary"] = ds          # dataclass 要在 sys.modules 里找到模块
    spec.loader.exec_module(ds)

CONDITION = "matched_transformer"


def eval_config():
    """训练和评测共用一个配置（报告里的配置 SHA 要对上）：U = 20、规划干扰项 2（测试根只有 4 条回合）。"""
    c = tt.small(updates=20)
    e = dataclasses.replace(c.evaluation, planning_candidates=2)
    return tt.Small(**{f.name: getattr(c, f.name) for f in dataclasses.fields(c) if f.name != "evaluation"}, evaluation=e)


CONFIG = eval_config()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ac_baseline, "OFFICIAL", tt.REDUCED_OFFICIAL)
        train_root = tt.build_root(tmp_path_factory.mktemp("s09b_train"))
        test_root = tx.build_root(tmp_path_factory.mktemp("s09b_test"))
        train_dir = tmp_path_factory.mktemp("s09b_runs")/"train"
        assert tt.go(train_root, train_dir, CONDITION, CONFIG) == 0
        full_dir = train_dir.parent/"full"                # 有 λ 的条件（读检查点时核损失配置快照）
        assert tt.go(train_root, full_dir, "po_jepa_full", CONFIG) == 0
    torch.set_num_threads(threads)
    return dict(train=train_root, test=test_root, train_dir=train_dir, full_dir=full_dir)


@pytest.fixture(autouse=True)
def reduced_official(monkeypatch):
    monkeypatch.setattr(ac_baseline, "OFFICIAL", tt.REDUCED_OFFICIAL)


def built(condition=CONDITION, weights=tt.WEIGHTS, seed=1703, ac=None):
    cal = tt.calib(condition, weights)
    return cd.build(condition, config=CONFIG, seed=seed, sigma_z=cal.sigma_z, device="cpu", weights=cal.weights,
                    ac_checkpoint=ac)


def provenance(world, **changes):
    return dict(world["test"]["provenance"], **changes)


def load(train_dir, world, *, condition=CONDITION, weights=tt.WEIGHTS, config=CONFIG, prov=None, seed=1703):
    model, loss, _ = built(condition, weights, seed=seed)
    got = ev.load_selected(train_dir, condition=condition, model=model, loss_config=loss, config=config,
                           provenance=prov or provenance(world))
    return model, got


def copy(world, tmp_path, name="t") -> Path:
    out = tmp_path/name
    shutil.copytree(world["train_dir"], out)
    return out


def rewrite(path: Path, **changes) -> None:
    doc = json.loads(path.read_text())
    doc.update(changes)
    path.write_text(json.dumps(doc))


# ---------------------------------------------------------------- 读检查点

def test_load_selected_reads_the_selected_checkpoint(world):
    """读 selection.json 选的那一步：返回的路径/SHA/步数和 selection 一致，权重等于检查点里的 model_state，模型在 eval 模式；
    来源记了 report/selection 的 SHA、数据字段和源码 SHA。"""
    root = world["train_dir"]
    selection = json.loads((root/"selection.json").read_text())
    best = selection["best"]
    model, got = load(root, world, seed=7)          # 另一个种子建的模型：权重相同只能是读进来的
    c = got["checkpoint"]
    assert c["step"] == best["step"] and c["sha256"] == best["checkpoint_sha256"] == cache.file_sha(root/c["path"])
    assert c["path"].startswith("checkpoints/") and c["mean"] == best["mean"]
    state = torch.load(io.BytesIO((root/c["path"]).read_bytes()), map_location="cpu", weights_only=True)["model_state"]
    fresh = built(seed=7)[0].state_dict()
    assert any(not torch.equal(v, state[k]) for k, v in fresh.items())
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items()) and not model.training
    s = got["source"]
    assert s["report_sha256"] == cache.file_sha(root/"report.json")
    assert s["selection_sha256"] == cache.file_sha(root/"selection.json")
    assert set(s["data_fields"]) == set(ev.DATA_FIELDS) and s["source_files_sha256"]
    assert best["step"] in {r["step"] for r in tr.validation_records(root, CONFIG.training.updates)}


def test_load_selected_accepts_other_source_files(world):
    """源码 SHA 只记录不核（评测代码可以在训练之后才修好，10 §8.1）；数据字段不同就拒绝。"""
    _, got = load(world["train_dir"], world, prov=provenance(world, source_files_sha256={"x": "0"*64}))
    assert got["source"]["source_files_sha256"] != {"x": "0"*64}
    with pytest.raises(ValueError, match="data provenance"):
        load(world["train_dir"], world, prov=provenance(world, upstream_commit="0"*40))
    with pytest.raises(ValueError, match="data provenance"):
        load(world["train_dir"], world, prov=provenance(world, split="train"))


def test_load_selected_rejects_incomplete_or_tampered_training(world, tmp_path):
    out = copy(world, tmp_path, "failed")
    (out/"FAILED.json").write_text("{}")
    with pytest.raises(ValueError, match="FAILED"):
        load(out, world)

    out = copy(world, tmp_path, "nocomplete")
    (out/"COMPLETE.json").unlink()
    with pytest.raises(ValueError, match="not complete"):
        load(out, world)

    out = copy(world, tmp_path, "report")             # 报告改了：COMPLETE.json 绑定的 SHA 对不上
    rewrite(out/"report.json", note="edited")
    with pytest.raises(ValueError, match="not complete"):
        load(out, world)

    with pytest.raises(ValueError, match="not a finished"):   # 换了 U：配置 SHA 和完成步数都不对
        load(world["train_dir"], world, config=dataclasses.replace(CONFIG, training=dataclasses.replace(
            CONFIG.training, updates=30)))
    with pytest.raises(ValueError, match="not a finished"):   # 别的条件
        load(world["train_dir"], world, condition="po_jepa_no_consistency")


def test_load_selected_rejects_another_selection(world, tmp_path):
    """selection.json 选的不是按验证记录重算的最优（或和报告不一致）；检查点文件被改；索引里不是这一步的验证点。"""
    root = world["train_dir"]
    selection = json.loads((root/"selection.json").read_text())
    records = tr.validation_records(root, CONFIG.training.updates)
    other = next(r for r in records if r["step"] != selection["best"]["step"])

    out = copy(world, tmp_path, "selection")
    rewrite(out/"selection.json", best=dict(selection["best"], step=other["step"], mean=other["mean"]))
    with pytest.raises(ValueError, match="disagree on the best step"):
        load(out, world)

    out = copy(world, tmp_path, "both")              # selection 和报告一起改到另一步（COMPLETE 重新绑定）：按验证记录重算才抓得到
    picked = dict(selection["best"], step=other["step"], mean=other["mean"],
                  checkpoint=f"checkpoints/{ck.file_name('validation', other['step'])}",
                  checkpoint_sha256=next(e["sha256"] for e in ck.read_index(out/"checkpoints") if e["step"] == other["step"]))
    rewrite(out/"selection.json", best=picked)
    rewrite(out/"report.json", best=picked)
    rewrite(out/"COMPLETE.json", report_sha256=cache.file_sha(out/"report.json"))
    with pytest.raises(ValueError, match="disagree on the best step"):
        load(out, world)

    out = copy(world, tmp_path, "file")
    path = out/selection["best"]["checkpoint"]
    raw = bytearray(path.read_bytes())
    raw[-10] ^= 1
    path.write_bytes(bytes(raw))
    assert cache.file_sha(path) != selection["best"]["checkpoint_sha256"]
    with pytest.raises(ValueError, match="SHA mismatch"):
        load(out, world)

    out = copy(world, tmp_path, "index")
    entries = ck.read_index(out/"checkpoints")
    name = Path(selection["best"]["checkpoint"]).name
    index = json.loads((out/"checkpoints"/"index.json").read_text())
    for e in index["checkpoints"]:
        if e["path"] == name:
            e["sha256"] = "0"*64
    (out/"checkpoints"/"index.json").write_text(json.dumps(index))
    assert ck.read_index(out/"checkpoints") != entries
    with pytest.raises(ValueError, match="indexed validation checkpoint"):
        load(out, world)


def test_load_selected_rejects_another_lambda_or_model(world):
    """有 λ 的条件：同一 λ 读得进，λ 不同（损失配置快照）拒绝；没有 λ 的条件不看 λ；模型结构不同（参数形状）拒绝。"""
    load(world["full_dir"], world, condition="po_jepa_full")
    with pytest.raises(ValueError):
        load(world["full_dir"], world, condition="po_jepa_full", weights={"time_weight": 0.25, "space_weight": 0.5})
    load(world["train_dir"], world, weights={"time_weight": 0.25, "space_weight": 0.5})
    model, loss, _ = built("po_jepa_no_consistency")
    with pytest.raises(ValueError):
        ev.load_selected(world["train_dir"], condition=CONDITION, model=model, loss_config=loss, config=CONFIG,
                         provenance=provenance(world))


# ---------------------------------------------------------------- 评测任务

def eval_data(world) -> ej.EvalData:
    r = world["test"]
    return ej.EvalData(splits=r["docs"], prepared=r["prepared"], outcomes=r["outcomes"], reader=r["reader"],
                       extra_reader=r["extra"], control_spec=droid_control_spec(SCALE, source="realized", config=CONFIG),
                       provenance=r["provenance"], feature_manifest_sha256="f"*64,
                       data_sha256={"action_scale.json": "a"*64, "splits.json": "s"*64,
                                    "windows/id_test.json": "i"*64, "windows/ood_test.json": "o"*64})


def run_eval(world, out, condition, seed):
    return ej.run(out, condition=condition, seed=seed, config=CONFIG, data=eval_data(world),
                  calibration=tt.calib(condition), device="cpu",
                  train_dir=world["train_dir"] if seed is not None else None,
                  ac_checkpoint=world["train"][3] if condition in cd.OFFICIAL_AC else None)


def test_parse_job_and_dirs(tmp_path):
    assert ej.parse_job("eval/matched_transformer/seed1703", CONFIG) == ("matched_transformer", 1703)
    assert ej.parse_job("eval/identity", CONFIG) == ("identity", None)
    assert ej.parse_job("eval/vjepa2_ac_zeroshot", CONFIG) == ("vjepa2_ac_zeroshot", None)
    for bad in ("eval/matched_transformer/seed1", "eval/identity/seed1703", "train/koopman/seed1703", "eval/x"):
        with pytest.raises(ValueError):
            ej.parse_job(bad, CONFIG)
    assert ej.eval_dir(tmp_path, CONFIG, "identity", None) == tmp_path/CONFIG.name/"eval"/"identity"
    assert ej.construction_seed(CONFIG) == CONFIG.training.seeds[0]


def test_eval_job_trained_condition_end_to_end(world, tmp_path):
    """训练过的条件：读选出的检查点，两个测试集写 rows、frame_rate、resolution、planning；job.json 记检查点和输入身份，
    source.json 的 SHA 在 job.json 里；重跑返回同一份报告；source.json 不同（换了训练）就拒绝；汇总能读。"""
    out = tmp_path/"eval"
    report = run_eval(world, out, CONDITION, 1703)
    assert report["status"] == "complete" or (out/"COMPLETE.json").exists()
    assert report["axes"] == {"windows": True, "frame_rate": True, "resolution": True, "planning": True}
    selection = json.loads((world["train_dir"]/"selection.json").read_text())
    job = json.loads((out/"job.json").read_text())
    assert job["checkpoint"]["sha256"] == selection["best"]["checkpoint_sha256"]
    assert job["checkpoint"]["step"] == selection["best"]["step"]
    assert job["source_sha256"] == ej._sha(json.loads((out/"source.json").read_text()))
    assert job["construction_seed"] is None and job["lambda_sha256"] is None and job["sigma_z"]["sha256"] == "s"*64
    for split in ev.TEST_SPLITS:
        assert (out/split/"planning").is_dir() and any((out/split/"planning").iterdir())
        assert "planning" in report["splits"][split]
    assert run_eval(world, out, CONDITION, 1703) == report
    view = ds.read_eval_view(out)
    assert view.condition == CONDITION and view.seed == 1703
    assert view.checkpoint_sha256 == selection["best"]["checkpoint_sha256"] and view.construction_seed is None
    assert dict(view.data_sha256) == eval_data(world).data_sha256
    source = json.loads((out/"source.json").read_text())
    (out/"source.json").write_text(json.dumps(dict(source, report_sha256="0"*64)))
    with pytest.raises(ValueError, match="source.json differs"):
        ds.read_eval_view(out)
    with pytest.raises(ValueError, match="differs from this command"):
        run_eval(world, out, CONDITION, 1703)


@pytest.mark.parametrize("condition", ["identity", "vjepa2_ac_zeroshot"])
def test_eval_job_untrained_controls(world, tmp_path, condition):
    """不训练的对照：不读检查点；构造种子 = 第一个种子；vjepa2_ac_zeroshot 的初始权重哈希等于同种子 vjepa2_ac 的（09 §2.4），
    并记官方权重 SHA；AC 走 bind/小步前向。"""
    out = tmp_path/condition
    report = run_eval(world, out, condition, None)
    job = json.loads((out/"job.json").read_text())
    assert job["checkpoint"] is None and job["construction_seed"] == 1703
    assert report["axes"]["planning"] is True
    view = ds.read_eval_view(out)
    assert view.checkpoint_sha256 is None and view.construction_seed == 1703
    if condition == "vjepa2_ac_zeroshot":
        _, _, info = built("vjepa2_ac", ac=world["train"][3])
        assert job["initial_model_sha256"] == info["initial_model_sha256"] == view.initial_model_sha256
        assert job["ac_checkpoint_sha256"] == cache.file_sha(world["train"][3])
    with pytest.raises(ValueError):
        ej.run(out, condition=condition, seed=1703, config=CONFIG, data=eval_data(world),
               calibration=tt.calib(condition), device="cpu")


def test_eval_job_needs_its_training_directory(world, tmp_path):
    with pytest.raises(ValueError, match="training directory"):
        ej.run(tmp_path/"x", condition=CONDITION, seed=1703, config=CONFIG, data=eval_data(world),
               calibration=tt.calib(CONDITION), device="cpu")
