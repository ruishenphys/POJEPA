"""DROID 上的 AC 无状态对照 vjepa2_ac_noprop、vjepa2_ac_small_noprop（docs/design/09-droid-training.md §15，2026-11-26）。

Jerry 决策卡定：主表 AC 照官方吃真实状态（`ac_state: true_state`），另补一组不吃状态的 AC（learned_constant，v7 做法）
作附表，和主配置共用数据、特征、σ_z、check/env。核对：

- 条件表、参数量（缩小版 248,896、mlp_ratio 6.0）、优先级、损失标签；
- 在 true_state 的主配置下，无状态对照的构造、训练、验证和“把整个配置换成 learned_constant 时的 vjepa2_ac(_small)”
  逐位相同（训练 20 步后模型、优化器、日志、验证值都相同）——即和另起一个 learned_constant 配置完全等价；
- 训练步不读状态（把状态换成随机数结果逐位不变）、验证不绑定起点状态；
- 检查点记自己的条件名，和吃状态的 AC 的检查点互相读不进；评测能读回所选检查点；
- 任务图：完整任务图里每个种子各一条训练、一条评测，依赖主配置同一个 check/env 和测试附加；`--conditions` 只排这两个，
  数据和特征任务与完整任务图逐键相同（不重编码），不排 λ。
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from po_jepa import ac_baseline
from po_jepa.droid import batches as bt
from po_jepa.droid import check as chk
from po_jepa.droid import checkpoints as ck
from po_jepa.droid import conditions as cd
from po_jepa.droid import evaluate as ev
from po_jepa.droid import train as tr
from po_jepa.droid.config import CONDITIONS
from po_jepa.losses import jepa_distance
from po_jepa.runtime import ac_training
from test_droid_train import (REDUCED_OFFICIAL, SIGMA_Z, Small, go, log_values, make_data, payload,  # noqa: F401
                              root, same_tree, small, validations)

from test_droid_jobs import expected, graph_of  # noqa: E402（先导入它，tools/ 才在 sys.path 上）

import droid_jobs  # noqa: E402

NOPROP = ("vjepa2_ac_noprop", "vjepa2_ac_small_noprop")
BASE = dict(vjepa2_ac_noprop="vjepa2_ac", vjepa2_ac_small_noprop="vjepa2_ac_small")
SEEDS = (1703, 1704, 1705)


@pytest.fixture(autouse=True)
def one_thread_and_reduced_official(monkeypatch):
    """同 test_droid_train：1 线程（逐位比较）、测试专用的缩小“官方”规格（假权重按它的格式存）。"""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(ac_baseline, "OFFICIAL", REDUCED_OFFICIAL)
    yield
    torch.set_num_threads(threads)


def _learned_constant(config):
    lc = dataclasses.replace(config, model=dataclasses.replace(config.model, ac_state="learned_constant"))
    return Small(**{f.name: getattr(lc, f.name) for f in dataclasses.fields(lc)})


def _build(c, config, root, seed=1703):
    return cd.build(c, config=config, seed=seed, sigma_z=SIGMA_Z, device="cpu",
                    ac_checkpoint=root[3] if c in cd.OFFICIAL_AC else None)


def _batch(data, config, step=0):
    dataset = bt.StepBatches("ac", data.data_root, data.manifests, data.train_set, data.episodes, seed=1703,
                             start=step, stop=step + 1, config=config)
    batches = bt.training_batches(dataset, data.control_spec, data.provenance, torch.device("cpu"), workers=0)
    _, batch = next(batches)
    batches.close()
    return batch


# ---------------------------------------------------------------- 条件表

def test_condition_table():
    assert CONDITIONS[-2:] == NOPROP and cd.NOPROP_AC == BASE
    assert set(NOPROP) <= set(cd.AC_CONDITIONS) and set(NOPROP) <= set(ev.AC_CONDITIONS)
    assert not set(NOPROP) & set(cd.STATE_AC) and not set(NOPROP) & set(ev.STATE_AC_CONDITIONS)
    assert "vjepa2_ac_noprop" in cd.OFFICIAL_AC and "vjepa2_ac_small_noprop" in cd.SMALL_AC
    assert "vjepa2_ac_noprop" in tr.RETAIN_TWO and "vjepa2_ac_small_noprop" not in tr.RETAIN_TWO
    assert not set(NOPROP) & set(cd.LAMBDA_CONDITIONS) and not set(NOPROP) & set(cd.REFERENCE)
    assert all(cd.LOSS_LABELS[c] == "no_consistency" for c in NOPROP)
    assert droid_jobs.TRAIN_PRIORITY["vjepa2_ac_noprop"] == droid_jobs.TRAIN_PRIORITY["vjepa2_ac"] == 1
    assert droid_jobs.TRAIN_PRIORITY["vjepa2_ac_small_noprop"] == droid_jobs.TRAIN_PRIORITY["vjepa2_ac_small"] == 2
    config = small()
    assert config.model.ac_state == "true_state"
    for c in NOPROP:
        assert cd.state_input(c, config) == cd.state_input(c, _learned_constant(config)) == "learned_constant"
        assert not cd.binds_state(c, config)
    assert cd.state_input("vjepa2_ac", config) == "true_state" and cd.binds_state("vjepa2_ac_small", config)
    assert chk.expected_parameters("vjepa2_ac_small_noprop", config) == 248_896      # 09 §2.3 的 learned_constant 数
    assert chk.expected_parameters("vjepa2_ac_small", config) == 249_344
    with pytest.raises(ValueError, match="not a vjepa2_ac condition"):
        cd.state_input("dino_wm_small", config)


# ---------------------------------------------------------------- 构造

@pytest.mark.parametrize("condition", NOPROP)
def test_build_equals_the_base_under_learned_constant(root, condition):
    """true_state 主配置下的无状态对照 = learned_constant 配置下的同构造 AC：规格、参数量、匹配、初始权重哈希、官方权重载入
    记录、ac_condition 都相同；和 true_state 下的同构造 AC 规格不同（没有状态编码器，多一个常向量）。"""
    config = small()
    model, loss, info = _build(condition, config, root)
    ref_model, ref_loss, ref = _build(BASE[condition], _learned_constant(config), root)
    state_model, _, state = _build(BASE[condition], config, root)
    assert model.spec == ref_model.spec and model.spec.state_input == "learned_constant" and not model.true_state
    assert state_model.true_state and info["spec"] != state["spec"]
    assert hasattr(model, "state_token") and not hasattr(model, "state_encoder")
    assert info["initial_model_sha256"] == ref["initial_model_sha256"] != state["initial_model_sha256"]
    assert info["parameters"] == ref["parameters"] and info["matching"] == ref["matching"]
    assert info["ac"] == ref["ac"] and model.ac_info == dict(ref_model.ac_info, condition=condition)
    assert model.ac_condition == ref_model.ac_condition == BASE[condition]
    assert loss == ref_loss and (loss.time_weight, loss.space_weight) == (0., 0.)
    assert [g["lr"] for g in cd.make_optimizer(condition, model).param_groups] == \
        [g["lr"] for g in ac_training.make_optimizer(ref_model, BASE[condition]).param_groups]
    if condition == "vjepa2_ac_small_noprop":
        assert info["parameters"] == 248_896 and info["matching"]["selected"]["mlp_ratio"] == 6.0
        assert info["ac"]["init"] == "scratch"
        with pytest.raises(ValueError, match="no official weights"):
            cd.build(condition, config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu", ac_checkpoint=root[3])
    else:
        assert info["ac"]["init"] == "official_checkpoint" and "state_encoder" in str(info["ac"]["official_load"])
        with pytest.raises(ValueError, match="official weights"):
            cd.build(condition, config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu")
    other = _build(condition, config, root, seed=1704)[2]
    assert (other["initial_model_sha256"] != info["initial_model_sha256"]) == (condition == "vjepa2_ac_small_noprop")


# ---------------------------------------------------------------- 训练步、验证

@pytest.mark.parametrize("condition", NOPROP)
def test_train_step_ignores_states(root, condition):
    """`conditions.train_step`（ac_state: true_state）不传状态：把片段的状态换成随机数结果逐位不变，且和直接调用
    `ac_training.train_step(states=None)` 逐位相同。"""
    config = small()
    batch = _batch(make_data(root, config), config)
    noisy = dataclasses.replace(batch, states=torch.randn_like(batch.states))
    out = []
    for how, b in (("droid", batch), ("noisy", noisy), ("direct", batch)):
        model, loss, _ = _build(condition, config, root)
        optimizer = cd.make_optimizer(condition, model)
        if how == "direct":
            result, lr = ac_training.train_step(BASE[condition], model, optimizer, b.features, b.actions, loss, 0,
                                                states=None)
        else:
            result, lr = cd.train_step(condition, model, optimizer, b, loss, 0, ac_state=config.model.ac_state)
        out.append((result.loss_values, lr, {k: v.clone() for k, v in model.state_dict().items()}))
    assert out[0][0] == out[1][0] == out[2][0] and out[0][1] == out[2][1]
    assert same_tree(out[0][2], out[1][2]) and same_tree(out[0][2], out[2][2])
    model, loss, _ = _build(condition, config, root)
    with pytest.raises(ValueError, match="takes no states"):          # 模型本身就不收状态
        ac_training.train_step(BASE[condition], model, cd.make_optimizer(condition, model), batch.features,
                               batch.actions, loss, 0, states=batch.states)


@pytest.mark.parametrize("condition", NOPROP)
def test_validation_does_not_bind_states(root, condition):
    config = small()
    data = make_data(root, config)
    model, loss, _ = _build(condition, config, root)
    got = tr.validate(model, condition=condition, data=data, config=config, loss_config=loss, device="cpu")
    batch, _ = bt.make_eval_batch(data.val_windows, data.reader, data.val_episodes, data.control_spec,
                                  data.provenance, "cpu")
    model.eval()
    with torch.no_grad():
        want = jepa_distance(model(batch.raw_z, batch.controls, batch.duration, batch.transfer.source), batch.raw_target,
                             config=loss).per_sample.double()
    assert got["values"] == want.tolist() and all(torch.isfinite(want))


# ---------------------------------------------------------------- 整段训练：和另起 learned_constant 配置等价

@pytest.mark.parametrize("condition", NOPROP)
def test_training_equals_a_learned_constant_config(root, tmp_path, condition):
    """主配置（true_state）下训练无状态对照 20 步，和把配置换成 learned_constant 后训练同构造的 AC 逐位相同：最终模型、
    优化器、每步日志的数值、每个验证点。登记和检查点记自己的条件名；两者的检查点互相读不进。"""
    config = small(updates=20)
    lc = _learned_constant(config)
    a, b = tmp_path/"noprop", tmp_path/"base_lc"
    assert go(root, a, condition, config) == 0
    assert go(root, b, BASE[condition], lc) == 0
    last = "checkpoints/step-000020.pt"
    pa, pb = payload(a/last), payload(b/last)
    assert same_tree(pa["model_state"], pb["model_state"]) and same_tree(pa["optimizer_state"], pb["optimizer_state"])
    assert pa["condition"] == condition and pb["condition"] == BASE[condition] and pa["spec"] == pb["spec"]
    assert log_values(a) == log_values(b) and validations(a) == validations(b)
    ra, rb = (json.loads((x/"report.json").read_text()) for x in (a, b))
    assert ra["validations"] == rb["validations"] and ra["parameters"] == rb["parameters"]
    reg = json.loads((a/"registration.json").read_text())
    assert reg["condition"] == condition and reg["config_sha256"] == config.config_sha256
    first = tr._read_lines(a/tr.LOG_FILE)[0]["steps"][0]
    assert first["p_out_calls"] is None and {"ac_teacher_forcing", "ac_rollout"} <= set(first["losses"])
    entry = ck.latest(ck.read_index(a/"checkpoints"))
    state_model, _, _ = _build(BASE[condition], config, root)
    with pytest.raises(ValueError, match="mismatch"):                 # 吃状态的 AC 读不进无状态对照的检查点
        ac_training.load_checkpoint(BASE[condition], a/"checkpoints"/entry["path"], entry["sha256"], state_model,
                                    cd.make_optimizer(BASE[condition], state_model), pa["provenance"])
    model, _, _ = _build(condition, config, root)
    with pytest.raises(ValueError, match="mismatch"):                 # 换成同构造 AC 的名字也读不进
        ac_training.load_checkpoint(BASE[condition], a/"checkpoints"/entry["path"], entry["sha256"], model,
                                    cd.make_optimizer(condition, model), pa["provenance"])


def test_evaluation_reads_the_selected_checkpoint(root, tmp_path):
    """评测的 `load_selected` 读回无状态对照所选的检查点（检查点里的条件名是自己的），权重等于检查点里的。"""
    config = small(updates=20)
    condition = "vjepa2_ac_small_noprop"
    out = tmp_path/"train"
    assert go(root, out, condition, config) == 0
    data = make_data(root, config)
    model, loss, _ = _build(condition, config, root)
    got = ev.load_selected(out, condition=condition, seed=1703, model=model, loss_config=loss, config=config,
                           provenance=data.provenance)
    best = json.loads((out/"selection.json").read_text())["best"]
    assert got["checkpoint"]["sha256"] == best["checkpoint_sha256"] and got["checkpoint"]["step"] == best["step"]
    saved = payload(out/best["checkpoint"])
    assert saved["condition"] == condition and same_tree(dict(model.state_dict()), saved["model_state"])
    assert not model.training


# ---------------------------------------------------------------- 任务图

def test_full_graph_has_six_trainings_sharing_the_main_data():
    full = graph_of("full")
    by_id = {j["id"]: j for j in full["jobs"]}
    for c in NOPROP:
        for s in SEEDS:
            train, ev_id = by_id[f"train/{c}/seed{s}"], by_id[f"eval/{c}/seed{s}"]
            assert train["deps"] == by_id[f"train/{BASE[c]}/seed{s}"]["deps"] == ["check/env"]
            assert ev_id["deps"] == ["encode/test-extra", train["id"]]
            assert train["complete"].endswith(f"/full/train/{c}/seed{s}/COMPLETE.json")
            assert "--config" in train["command"] and train["command"][train["command"].index("--config") + 1] \
                == by_id["encode/train-000"]["command"][by_id["encode/train-000"]["command"].index("--config") + 1]
    assert sum(j["stage"] == "train" for j in full["jobs"]) == len(CONDITIONS)*len(SEEDS) == 36
    assert set(by_id["summary"]["deps"]) >= {f"eval/{c}/seed{s}" for c in NOPROP for s in SEEDS}


def test_conditions_filter_schedules_only_the_controls():
    full = graph_of("full")
    got = graph_of("full", conditions=droid_jobs.parse_conditions(",".join(NOPROP)))
    ids = {j["id"] for j in got["jobs"]}
    want = {k for k in expected("full", 20, list(SEEDS))
            if not k.startswith(("train/", "eval/", "calibrate/lambda-")) and k != "summary"}
    want |= {f"{st}/{c}/seed{s}" for st in ("train", "eval") for c in NOPROP for s in SEEDS}
    assert ids == want and got["conditions"] == list(NOPROP)
    assert not any(i.startswith("calibrate/lambda-") for i in ids)
    by_id = {j["id"]: j for j in full["jobs"]}
    for j in got["jobs"]:                       # 数据、编码、σ_z、check/env 与完整任务图逐键相同：同一完成标志，不重编码
        assert j == by_id[j["id"]]
    assert {j["complete"] for j in got["jobs"] if j["stage"] == "encode"} == \
        {j["complete"] for j in full["jobs"] if j["stage"] == "encode"}
    assert sum(j["stage"] == "train" for j in got["jobs"]) == 6


@pytest.mark.parametrize("condition", NOPROP)
def test_train_entry_passes_the_official_weights(tmp_path, monkeypatch, condition):
    """`droid.train.main`：官方规模的无状态对照从 `<数据根>/checkpoints/vjepa2-ac-vitg.pt` 载入官方权重，缩小版不传。"""
    from test_droid_index import CONFIGS
    calls = {}
    monkeypatch.setattr(tr, "load_data", lambda config, root: type("D", (), dict(
        feature_manifest_sha256="f"*64, action_scale_sha256="a"*64, train_set=[]))())
    monkeypatch.setattr(tr, "load_calibration", lambda *a, **k: None)
    monkeypatch.setattr(tr, "run", lambda out, **kw: calls.update(kw, out=out) or 0)
    assert tr.main(["--config", str(CONFIGS/"pilot.yaml"), "--data-root", str(tmp_path/"data"), "--run-root",
                    str(tmp_path/"runs"), "--job", f"train/{condition}/seed1703", "--device", "cpu"]) == 0
    want = tmp_path/"data"/"checkpoints"/tr.OFFICIAL_AC_FILE if condition == "vjepa2_ac_noprop" else None
    assert calls["condition"] == condition and calls["ac_checkpoint"] == want
    assert calls["out"] == tmp_path/"runs"/"pilot"/"train"/condition/"seed1703"


def test_summary_labels_state_inputs_and_supplementary_controls():
    """Codex ACNP1 P3：汇总写出 AC 一族的状态输入，并把两个无状态对照标成附表。"""
    import droid_summary as ds
    config = small()
    got = ds.ac_state_inputs(config, (*CONDITIONS, "identity", "vjepa2_ac_zeroshot"))
    assert got == dict(vjepa2_ac="true_state", vjepa2_ac_small="true_state", vjepa2_ac_zeroshot="true_state",
                       vjepa2_ac_noprop="learned_constant", vjepa2_ac_small_noprop="learned_constant",
                       dino_wm_small="none (no-prop)", jepa_wm_small="none (no-prop)")
    assert ds.ac_state_inputs(_learned_constant(config), NOPROP) == dict.fromkeys(NOPROP, "learned_constant")
