"""S09b：DROID 的 σ_z / λ 标定（docs/design/09-droid-training.md §3）和环境检查 `check/env`（§7），以及训练任务的
汇总读取函数（13 §8.1，`tools/droid_summary.read_train`、`read_env`）。

数据根同 S09a 的测试（`test_droid_train`：训练片 4 条回合、验证片 2 条，随机 float16 特征，内存里的准备文件，缩小的
“官方”AC 规格）。σ_z 和 v7 `workflow.calibrate_sigma_z` 直接对照（换掉它的数据访问，喂同样的行）；λ 和按 v7
`calibrate_weights` 的步骤手写的参照逐位对照（同样的批给出同样的梯度范数和 λ）。逐位比较都在 1 线程下做（14 §1）。
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from po_jepa.droid import batches as bt
from po_jepa.droid import calibrate as cb
from po_jepa.droid import check as chk
from po_jepa.droid import conditions as cd
from po_jepa.droid import features as ft
from po_jepa.droid import prepare as pr
from po_jepa.droid import train as tr
from po_jepa.droid import windows as wn
from po_jepa.droid.config import CONDITIONS
from po_jepa.droid.spectral_gap import read_sigma_z
from po_jepa.runtime import cache, training, workflow
from test_droid_index import CONFIGS
from test_droid_train import (PILOT, ROWS, SHARDS, SIGMA_Z, Small, calib, go, make_data,  # noqa: F401
                              one_thread_and_reduced_official, root, small, uid)

tools = importlib.util.spec_from_file_location("droid_summary", Path(cache.ROOT)/"tools"/"droid_summary.py")
ds = sys.modules.get("droid_summary") or importlib.util.module_from_spec(tools)
if "droid_summary" not in sys.modules:
    sys.modules["droid_summary"] = ds          # dataclass 要在 sys.modules 里找到模块
    tools.loader.exec_module(ds)


def with_calibration(config, **changes):
    cal = dataclasses.replace(config.calibration, **changes)
    return Small(**{f.name: getattr(config, f.name) for f in dataclasses.fields(config) if f.name != "calibration"},
                 calibration=cal)


def scale_doc(pool, config=PILOT):
    return {"schema": pr.SCALE_SCHEMA, "pool": {"scale_episodes": config.action.scale_episodes},
            "episodes": {"realized": list(pool), "commanded": list(pool[:1])}}


# ---------------------------------------------------------------- §3.1 σ_z

def test_sigma_z_pool_and_picks():
    """回合池 = 动作尺度的池（action_scale.json 的 episodes[source]）；取第 ⌊i·n/8⌋ 条；池不对、不够 8 条报错。"""
    pool = [f"u{i:02d}" for i in range(13)]
    assert cb.sigma_z_pool(scale_doc(pool), PILOT) == pool
    assert cb.sigma_z_picks(pool, 8) == [pool[i*13//8] for i in range(8)] == \
        ["u00", "u01", "u03", "u04", "u06", "u08", "u09", "u11"]
    assert cb.sigma_z_picks(pool[:8], 8) == pool[:8]
    with pytest.raises(ValueError, match="fewer than"):
        cb.sigma_z_picks(pool[:7], 8)
    with pytest.raises(ValueError, match="action_scale"):
        cb.sigma_z_pool(dict(scale_doc(pool), schema="x"), PILOT)
    with pytest.raises(ValueError, match="scale_episodes"):
        cb.sigma_z_pool(dict(scale_doc(pool), pool={"scale_episodes": 7}), PILOT)
    with pytest.raises(ValueError, match="repeats"):
        cb.sigma_z_pool(scale_doc(pool + pool[:1]), PILOT)


def test_sigma_z_is_the_v7_rule_on_the_same_rows(root, monkeypatch):
    """§8：和直接调用 v7 `workflow.calibrate_sigma_z` 的同一算法（数据访问换成同样的 DROID 行）逐位相同。"""
    data, manifests, _, _ = root
    reader = ft.FeatureReader(data, manifests)
    pool = [uid(k) for k in "ABCDEF"]
    config = with_calibration(small(), sigma_z_episodes=4)
    got = cb.calibrate_sigma_z(reader, pool, config)
    assert got["episodes"] == [pool[i*6//4] for i in range(4)] and got["rows"] == [0, 8, 16]
    assert got["pairs"] == 4*3*256*255 and (got["q_seed"], got["q_dim"]) == (1703, 16)
    assert got["row_sha256"][pool[0]] == [reader.record(pool[0])["row_sha256"][r] for r in (0, 8, 16)]

    class Window:
        def __init__(self, z):
            self.inputs = SimpleNamespace(raw_z=z)

    def read_window(path, *, start_tick, duration_cycles, expected_split, device):
        return Window(reader.rows(pool[int(Path(path).name)], [start_tick]))
    monkeypatch.setattr(workflow, "data", SimpleNamespace(datasets=SimpleNamespace(active=lambda: SimpleNamespace(train=6)),
                                                          cache=SimpleNamespace(read_window=read_window)))
    v7 = workflow.calibrate_sigma_z("features", {"calibration": {"sigma_z_ids": 4, "sigma_z_ticks": [0, 8, 16]},
                                                 "model": {"graph": {"q_seed": 1703, "q_dim": 16}}})
    assert v7["sigma_z_ids"] == [pool.index(u) for u in got["episodes"]]
    assert got["sigma_z"] == v7["sigma_z"] and got["featsq_quantiles_5_50_95"] == v7["featsq_quantiles_5_50_95"]
    # 偶数个值取较小的中间值（torch.median），不是两者的平均
    values = torch.tensor([1., 4., 9., 16.], dtype=torch.float64)
    assert float(values.median()) == 4.


def test_run_sigma_z_writes_once(root, tmp_path, monkeypatch):
    """`calibrate/sigma-z`：读 action_scale.json 的池和特征清单，写 sigma_z.json（训练任务的 `read_sigma_z` 读得出）；
    已存在不覆盖，命令行直接退出 0。"""
    data, manifests, _, _ = root
    monkeypatch.setattr(ft, "feature_manifests", lambda d, c: (manifests, "f"*64))
    scale = pr.prepared_dir(data)/pr.ACTION_SCALE_FILE
    scale.parent.mkdir(parents=True, exist_ok=True)
    scale.unlink(missing_ok=True)
    pool = [uid(k) for k in "ABCD"] + [uid(k) for k in "EF"]
    cache._write_json(scale, scale_doc(pool))
    config = with_calibration(small(), sigma_z_episodes=3)
    doc = cb.run_sigma_z(config, data, tmp_path/"runs")
    path = tmp_path/"runs"/config.name/"calibration"/"sigma_z.json"
    assert read_sigma_z(path, config) == (doc["sigma_z"], cache.file_sha(path))
    assert doc["config_sha256"] == config.config_sha256 and doc["feature_manifest_sha256"] == "f"*64
    assert doc["action_scale_sha256"] == cache.file_sha(scale) and doc["pool_size"] == 6
    assert doc["sigma_z"] == cb.calibrate_sigma_z(ft.FeatureReader(data, manifests), pool, config)["sigma_z"]
    with pytest.raises(FileExistsError):
        cb.run_sigma_z(config, data, tmp_path/"runs")
    assert cb.main(["--config", str(CONFIGS/"pilot.yaml"), "--data-root", str(data), "--run-root", str(tmp_path/"runs"),
                    "--job", "calibrate/sigma-z"]) == 0           # 已标定：不重算
    assert cb.parse_job("calibrate/lambda-fixed_fourier") == "fixed_fourier"
    for bad in ("calibrate/lambda-koopman", "calibrate/sigma", "check/env"):
        with pytest.raises(ValueError):
            cb.parse_job(bad)


# ---------------------------------------------------------------- §3.2 λ

def lambda_config():
    """ρ 取 0.75（不等于 1，乘没乘 ρ 才看得出来）。"""
    c = with_calibration(small(updates=20), pretrain_updates=3, median_batches=2)
    return Small(**{f.name: getattr(c, f.name) for f in dataclasses.fields(c) if f.name != "model"},
                 model=dataclasses.replace(c.model, rho=0.75))


def reference_lambda(root, config):
    """按 v7 `workflow.calibrate_weights` 的步骤手写：no_consistency 模型、初始状态量前 2 批、前 3 批只做 JEPA 训练、
    再量第 3、4 批；批用 06 §2 的窗口函数和 `make_training_batch` 逐步现组。"""
    data = make_data(root, config)
    model, loss, _ = cd.build("po_jepa_no_consistency", config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu")
    model.train()
    rows = {u: data.episodes[u].rows for u in data.train_set}

    def batch(step):
        w = wn.training_windows(data.train_set, rows, seed=1703, step=step, config=config)
        return bt.make_training_batch(w, data.reader, data.episodes, data.control_spec, data.provenance, "cpu")
    initial = [training.consistency_gradient_norms(model, batch(k), loss_config=loss) for k in range(2)]
    optimizer = training.make_optimizer(model)
    for k in range(3):
        training.train_step(model, optimizer, batch(k), loss_config=loss, completed_steps=k, condition="no_consistency")
    after = [training.consistency_gradient_norms(model, batch(k), loss_config=loss) for k in (3, 4)]
    median = {key: statistics.median(r[key] for r in after) for key in after[0]}
    return initial, after, median


def test_lambda_follows_the_v7_steps(root):
    """po_jepa_full 的 λ：同样的批给出同样的梯度范数和 λ（逐位）；λ = ρ·median|∇L_JEPA| / median|∇L_c|。"""
    config = lambda_config()
    doc = cb.calibrate_lambda("po_jepa_full", config=config, data=make_data(root, config), sigma_z=SIGMA_Z,
                              sigma_z_sha256="s"*64, device="cpu", workers=0)
    initial, after, median = reference_lambda(root, config)
    assert doc["initial_state"]["per_batch"] == initial and doc["per_batch"] == after and doc["median"] == median
    assert config.model.rho == doc["rho"] == 0.75
    assert doc["time_weight"] == config.model.rho*median["grad_jepa"]/median["grad_time"]
    assert doc["space_weight"] == config.model.rho*median["grad_jepa"]/median["grad_space"]
    assert doc["jepa_only_condition"] == "po_jepa_no_consistency" and doc["pretrain_updates"] == 3
    assert [h["step"] for h in doc["pretrain_history"]] == [1, 2, 3] and doc["batches"] == 2
    data = make_data(root, config)
    rows = {u: data.episodes[u].rows for u in data.train_set}
    assert doc["step_window_ids_sha256"] == [wn.list_sha256([w["id"] for w in wn.training_windows(
        data.train_set, rows, seed=1703, step=k, config=config)]) for k in range(5)]
    assert doc["initial_model_sha256"] == cd.build("po_jepa_no_consistency", config=config, seed=1703, sigma_z=SIGMA_Z,
                                                   device="cpu")[2]["initial_model_sha256"]
    assert doc["graph"] == dict(sigma_x=0.125, sigma_z=SIGMA_Z, max_modes=16)
    assert {"rho", "batches", "statistic", "method", "pretrain_updates", "pretrain_history", "per_batch", "median",
            "mean", "initial_state", "time_weight", "space_weight", "graph", "config_sha256", "sigma_z_sha256",
            "initial_model_sha256", "step_window_ids_sha256"} <= set(doc)


def test_lambda_other_conditions_and_zero_denominator(root, monkeypatch):
    """fixed_fourier：两个权重为 0 的 fixed_fourier；transformer_consistency：matched_transformer 和通用梯度范数
    （初始状态 p_out 零初始化、一致性梯度为 0，初始 λ 记 None）。量到的分母为 0 时报错，不除出 inf。"""
    config = lambda_config()
    model, loss, _ = cb.jepa_only("fixed_fourier", config, SIGMA_Z, "cpu")
    assert (loss.time_weight, loss.space_weight) == (0., 0.) and type(model.spectrum).__name__ == "FourierSpectrum"
    model, loss, info = cb.jepa_only("transformer_consistency", config, SIGMA_Z, "cpu")
    assert info["condition"] == "matched_transformer" and (loss.time_weight, loss.space_weight) == (0., 0.)
    with pytest.raises(ValueError, match="no lambda"):
        cb.jepa_only("koopman", config, SIGMA_Z, "cpu")
    for c in ("fixed_fourier", "transformer_consistency"):
        doc = cb.calibrate_lambda(c, config=config, data=make_data(root, config), sigma_z=SIGMA_Z, sigma_z_sha256="s"*64,
                                  device="cpu", workers=0)
        assert all(math.isfinite(doc[k]) and doc[k] > 0 for k in ("time_weight", "space_weight"))
        assert doc["jepa_only_condition"] == cb.JEPA_ONLY[c]
    assert doc["initial_state"]["time_weight"] is None and doc["initial_state"]["median"]["grad_time"] == 0.
    real = cb.gradient_norms
    monkeypatch.setattr(cb, "gradient_norms", lambda *a: dict(real(*a), grad_space=0.))
    with pytest.raises(ValueError, match="denominator"):
        cb.calibrate_lambda("po_jepa_full", config=config, data=make_data(root, config), sigma_z=SIGMA_Z,
                            sigma_z_sha256="s"*64, device="cpu", workers=0)


def test_run_lambda_file_is_read_by_the_training_task(root, tmp_path, monkeypatch):
    """`calibrate/lambda-<条件>` 写的文件，训练任务的 `read_lambda` 读得出（σ_z 文件 SHA、配置 SHA 对得上）；不覆盖。"""
    config = lambda_config()
    monkeypatch.setattr(tr, "load_data", lambda c, d: make_data(root, c))
    cal = tmp_path/"runs"/config.name/"calibration"
    cal.mkdir(parents=True)
    cache._write_json(cal/"sigma_z.json", dict(sigma_z=SIGMA_Z, config_sha256=config.config_sha256))
    doc = cb.run_lambda("po_jepa_full", config, "unused", tmp_path/"runs", "cpu", workers=0)
    weights, sha = tr.read_lambda(cal/"lambda-po_jepa_full.json", config, cache.file_sha(cal/"sigma_z.json"))
    assert weights == {k: doc[k] for k in ("time_weight", "space_weight")} and sha == cache.file_sha(cal/"lambda-po_jepa_full.json")
    assert tr.load_calibration(tmp_path/"runs", config, "po_jepa_full").weights == weights
    with pytest.raises(FileExistsError):
        cb.run_lambda("po_jepa_full", config, "unused", tmp_path/"runs", "cpu", workers=0)


# ---------------------------------------------------------------- §7 check/env

def upstream_ok():
    return dict(upstream_commit="204698b", encoder_checkpoint_sha256=cache.CHECKPOINT_SHA), []


def run_env(root, config, **kw):
    kw.setdefault("conditions", CONDITIONS)
    return chk.run_check(config, data=make_data(root, config), sigma_z=SIGMA_Z, sigma_z_sha256="s"*64, device="cpu",
                         ac_checkpoint=root[3], upstream=kw.pop("upstream", upstream_ok), steps=kw.pop("steps", 7),
                         loader_steps=4, disk_rows=12, workers=0, **kw)


def test_check_env_passes_on_every_condition(root, tmp_path):
    """8 个可训练条件：参数量和 §2.3 的表一致、§2.4 的关系（含 zeroshot）、每步损失和梯度有限且每个参数都有梯度、
    p_out 调用次数、第 6 步起的单步中位数、验证计时和总时长推算；求谱后端一致（CPU 上没有 cuda_float64）；两种加载器
    都比最快的条件快。写 env.json，汇总的 `read_env` 读得出。"""
    config = small(updates=20)
    doc = run_env(root, config)
    assert doc["status"] == "passed", doc["problems"]
    recs = doc["conditions"]
    assert {c: recs[c]["parameters"] for c in chk.PARAMETER_TABLE} == chk.PARAMETER_TABLE
    assert recs["vjepa2_ac_small"]["parameters"] == chk.AC_SMALL_TABLE["true_state"]
    assert all(r["same"] for r in doc["relations"]) and len(doc["relations"]) == 4
    for c, r in recs.items():
        assert [s["step"] for s in r["steps"]] == list(range(1, 8)) and r["timed_steps"] == [6, 7]
        assert r["median_step_seconds"] == statistics.median(s["seconds"] for s in r["steps"][5:])
        if c in cd.AC_CONDITIONS:
            assert all(s["p_out_calls"] is None for s in r["steps"])
        else:
            assert all(s["p_out_calls"] == cd.expected_readouts(c, cd.build(
                c, config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu",
                weights=dict(cb.PLACEHOLDER_WEIGHTS) if c in cd.LAMBDA_CONDITIONS else None)[0]) for s in r["steps"])
        e = r["estimate"]
        assert e["validations"] == 3 and e["seconds"] == 20*r["median_step_seconds"] + 3*r["validation_seconds"]
        assert r["validation_windows"] == 8
    b = doc["spectral_backends"]
    assert b["agree"] == {"cpu_batched": True, "cpu_threads": True} and b["unavailable"] == ["cuda_float64"]
    assert b["batches"] == 8 and b["fastest_agreeing"] in ("cpu_batched", "cpu_threads")
    assert set(doc["loaders"]) == {"train", "ac"} and all(v["ok"] and v["steps"] == 4 for v in doc["loaders"].values())
    assert doc["loaders"]["ac"]["fastest_condition"] in cd.AC_CONDITIONS
    assert doc["disk"]["rows"] == 12 and doc["upstream"]["ac_checkpoint_sha256"] == cache.file_sha(root[3])
    assert chk.write_result(tmp_path, config, doc) == 0
    path = tmp_path/config.name/"check"/"env.json"
    assert ds.read_env(path) == json.loads(path.read_text())
    with pytest.raises(ValueError):
        ds.read_env(_failed_copy(path))


def _failed_copy(path: Path) -> Path:
    other = path.with_name("failed.json")
    other.write_text(json.dumps(dict(json.loads(path.read_text()), status="failed")))
    return other


def test_check_env_failures_write_failed_json(root, tmp_path, monkeypatch):
    """任一项不通过：status failed、problems 列出，写 FAILED.json、不写 env.json、退出 1。逐项：参数量对不上、上游提交
    不对、一致性后端不一致、加载器太慢、损失非有限、参数没有梯度。"""
    config = small(updates=20)
    monkeypatch.setitem(chk.PARAMETER_TABLE, "matched_transformer", 1)
    doc = run_env(root, config, conditions=("matched_transformer",), steps=2)
    assert doc["status"] == "failed" and any("09 §2.3" in p for p in doc["problems"])
    assert chk.write_result(tmp_path, config, doc) == 1
    assert (tmp_path/config.name/"check"/"FAILED.json").exists() and not (tmp_path/config.name/"check"/"env.json").exists()
    monkeypatch.setitem(chk.PARAMETER_TABLE, "matched_transformer", 249_024)
    doc = run_env(root, config, conditions=("matched_transformer",), steps=2,
                  upstream=lambda: ({}, ["upstream commit x != y"]))
    assert doc["problems"] == ["upstream commit x != y"]
    real = chk.spectral_backends.agreement
    monkeypatch.setattr(chk.spectral_backends, "agreement", lambda a, b, k: dict(real(a, b, k), agree=False))
    threads = dataclasses.replace(config.model, spectral_backend="cpu_threads")
    cfg = Small(**{f.name: getattr(config, f.name) for f in dataclasses.fields(config) if f.name != "model"}, model=threads)
    doc = run_env(root, cfg, conditions=("matched_transformer",), steps=2)
    assert any("does not agree" in p for p in doc["problems"]) and doc["spectral_backends"]["fastest_agreeing"] == "cpu_batched"
    monkeypatch.setattr(chk.spectral_backends, "agreement", real)
    monkeypatch.setattr(chk, "loader_speed", lambda kind, *a, **k: dict(kind=kind, steps=4, seconds=1e9, steps_per_second=4e-9))
    doc = run_env(root, config, conditions=("matched_transformer",), steps=2)
    assert any("too slow" in p for p in doc["problems"]) and not doc["loaders"]["train"]["ok"]
    monkeypatch.undo()
    step = cd.train_step

    def broken(condition, model, *a, **k):
        result, lr = step(condition, model, *a, **k)
        next(model.parameters()).grad = None
        return dataclasses.replace(result, loss_values=dict(result.loss_values, jepa=float("nan"))), lr
    monkeypatch.setattr(chk.cd, "train_step", broken)
    doc = run_env(root, config, conditions=("matched_transformer",), steps=2)
    assert any("nonfinite loss" in p for p in doc["problems"]) and any("no gradient" in p for p in doc["problems"])


def test_upstream_identity(tmp_path):
    """12 §1：上游代码的 git 提交不是 204698b、编码器权重不在或 SHA 不对，都算不通过。"""
    up = tmp_path/"upstream"/"vjepa2"
    up.mkdir(parents=True)
    git = ["git", "-C", str(up), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q", str(up)], check=True)
    (up/"f").write_text("x")
    subprocess.run(git + ["add", "f"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "x"], check=True)
    ident, problems = chk.upstream_identity(tmp_path, PILOT)
    assert ident["upstream_commit"] and any("upstream commit" in p for p in problems)
    assert any("encoder weights" in p for p in problems) and ident["encoder_checkpoint_sha256"] is None
    ident, problems = chk.upstream_identity(tmp_path/"nowhere", PILOT)
    assert ident["upstream_commit"] is None and any("cannot read" in p for p in problems)


# ---------------------------------------------------------------- 汇总的训练读取函数（13 §8.1）

def test_read_train_on_a_real_training_directory(root, tmp_path):
    """`read_train` 读 S09a 训练任务的真实目录：登记的数据 SHA、σ_z、λ、初始哈希、日志覆盖、选点最优、每步用时、
    续跑次数、求谱后端、收敛曲线；CPU 运行的 GPU 小时和显存记 None。"""
    config = small(updates=20)
    out = tmp_path/"train"
    assert go(root, out, "po_jepa_full", config) == 0
    view = ds.read_train(out)
    reg = json.loads((out/"registration.json").read_text())
    report = json.loads((out/"report.json").read_text())
    selection = json.loads((out/"selection.json").read_text())
    assert (view.condition, view.seed, view.updates, view.completed) == ("po_jepa_full", 1703, 20, 20)
    assert view.data_sha256 == {"splits.json": None, "windows/val.json": "b"*64, "action_scale.json": "a"*64}
    assert view.sigma_z == SIGMA_Z and view.lambdas == {"time_weight": 0.5, "space_weight": 0.5}
    assert view.initial_model_sha256 == reg["model"]["initial_model_sha256"] and view.train_set_sha256 == reg["train_set"]["sha256"]
    rows = tr._read_lines(out/tr.LOG_FILE)
    assert view.log_ranges == tuple((r["first_step"], r["last_step"]) for r in rows) and view.log_ranges[0][0] == 1
    assert view.best_checkpoint_sha256 == selection["best"]["checkpoint_sha256"] == report["best"]["checkpoint_sha256"]
    assert len(view.step_seconds) == 20 and view.resumes == 0 and view.spectral_backend == "cpu_batched"
    assert view.peak_memory_bytes is None and view.gpu_hours is None
    assert view.curves["validation"] == [(float(v["step"]), v["mean"]) for v in report["validations"]]
    assert view.curves["train_jepa"][0] == (float(rows[0]["last_step"]),
                                            math.fsum(s["losses"]["jepa"] for s in rows[0]["steps"])/len(rows[0]["steps"]))
    tampered = json.loads((out/"registration.json").read_text())
    tampered["seed"] = 1
    (out/"registration.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="another registration"):
        ds.read_train(out)


def test_registration_records_the_splits_sha(root, tmp_path):
    """13 §8.3“划分和窗口”要比 splits.json：训练登记记 `splits_sha256`（S09b 加）。"""
    config = small(updates=20)
    data = dataclasses.replace(make_data(root, config), splits_sha256="9"*64)
    current = tr.registration(condition="koopman", seed=1703, config=config, data=data, calibration=calib("koopman"),
                              info=cd.build("koopman", config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu")[2],
                              reference=None, loss_config=config.loss_config(time_weight=0., space_weight=0.),
                              optimizer=cd.make_optimizer("koopman", cd.build("koopman", config=config, seed=1703,
                                                                              sigma_z=SIGMA_Z, device="cpu")[0]))
    assert current["splits_sha256"] == "9"*64
