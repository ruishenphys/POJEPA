"""S26a：N1 基线、N2 消融（仿真补跑的 5 个条件）做成 v8 任务表的可选阶段。

- 补跑标定的基础标定可以是本目录 run.py calibrate 写的（--base-calibration local），默认仍是导入的 v7 标定；
- make_v8_jobs.py --stage addon / addon-pilot，variants、strength、long、timing 的 --conditions 收这 5 个条件；
- summarize_v7.py 把补跑条件并进同一张表，按 08 §1.3 核初始权重，补跑条件全缺时不算缺失。
"""

import importlib.util
import math
import json
import shlex
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml

from po_jepa import datasets
from po_jepa.runtime import cache, checks
from po_jepa.runtime import workflow as w
from po_jepa.runtime.addon import calibrate, calibration, conditions, entry
from test_addon_conditions import po, transformer
from test_training import e06_case, e06_config

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
V7_HEADER = "# v7 标定（run.py calibrate 自动写入，勿手改）："


@pytest.fixture
def local(monkeypatch, tmp_path):
    """v7 协议、push:2000（v7 没有的场景）；基础标定由 workflow.write_v7_calibration 写在临时目录，没有导入记录。"""
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    old = datasets._set_for_tests(datasets.DatasetSpec("push", 2000))
    folder = tmp_path/"configs/calibration"
    folder.mkdir(parents=True)
    path = folder/"push-2000-u10000.yaml"
    monkeypatch.setattr(checks, "v7_calibration_path", lambda: path)
    monkeypatch.setattr(calibration, "addon_calibration_path", lambda: folder/"addon"/path.name)
    w.write_v7_calibration(dict(sigma_z=1.25, time_weight=.03, space_weight=4.0, rho=1.0), "test")
    yield path
    datasets._set_for_tests(old)


@pytest.fixture
def small(local, monkeypatch):
    """和 test_addon_entry.small 相同：协议的 pretrain_updates/batches 调成 3/2，小模型，e06 合成批。"""
    import copy
    full = checks.load_protocol

    def protocol():
        value = copy.deepcopy(full())
        value["calibration"].update(pretrain_updates=3, batches=2)
        return value
    monkeypatch.setattr(checks, "load_protocol", protocol)
    zero = e06_config(time_weight=0., space_weight=0.)
    models = dict(fixed_fourier=lambda: po("fourier"), transformer_consistency=transformer)
    monkeypatch.setattr(calibrate, "_jepa_only", lambda condition, device: (models[condition](), zero))
    monkeypatch.setattr(calibrate.data, "load_training_batch", lambda features, queries, device: e06_case()[1])
    return local


def _imported(path):
    (path.parent/"IMPORTED.json").write_bytes(cache._json_bytes(dict(schema="addon_v7_calibration_import_v1", files={
        path.name: dict(source="fictional", sha256=cache.file_sha(path), imported_at="2026-11-13T00:00:00+00:00",
                        v7_commit=calibrate.V7_COMMIT)})))


# ---------- 基础标定的来源 ----------

def test_local_calibration_accepts_the_file_run_py_calibrate_writes(local):
    assert local.read_text(encoding="utf-8").startswith(f"{V7_HEADER}push-2000，10000 步\n")
    assert calibrate.require_local_calibration() == cache.file_sha(local)
    assert calibrate.require_base_calibration("local") == cache.file_sha(local)
    with pytest.raises(ValueError, match="IMPORTED.json：v7 标定必须用 tools/import_v7_calibration.py 导入"):
        calibrate.require_base_calibration("v7-import")       # 默认模式不变：没有导入记录就拒绝
    with pytest.raises(ValueError, match="^base calibration must be one of"):
        calibrate.require_base_calibration("v8")


def test_local_calibration_refuses_an_imported_file(local):
    _imported(local)
    assert calibrate.require_base_calibration("v7-import") == cache.file_sha(local)
    with pytest.raises(ValueError, match="是从 v7 导入的"):
        calibrate.require_local_calibration()
    (local.parent/"IMPORTED.json").write_bytes(cache._json_bytes(dict(schema="addon_v7_calibration_import_v1", files={
        "other-2000-u10000.yaml": dict(sha256="0"*64)})))
    assert calibrate.require_local_calibration() == cache.file_sha(local)   # 只看这个文件名的记录


def test_local_calibration_checks_header_and_values(local, monkeypatch):
    good = local.read_text(encoding="utf-8")
    local.write_text(good.replace("push-2000", "push-200", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="首行不是 run.py calibrate 写的表头"):
        calibrate.require_local_calibration()
    local.write_text(good.replace("10000 步", "2000 步", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="首行不是 run.py calibrate 写的表头"):
        calibrate.require_local_calibration()
    local.write_text(good.split("\n", 1)[1], encoding="utf-8")            # 去掉表头（手写的文件）
    with pytest.raises(ValueError, match="首行不是"):
        calibrate.require_local_calibration()
    for key, bad in (("sigma_z", None), ("time_weight", 0), ("space_weight", -1.), ("sigma_z", float("inf")), ("time_weight", True)):
        value = yaml.safe_load(good)
        value[key] = bad
        local.write_text(good.split("\n", 1)[0]+"\n"+yaml.safe_dump(value), encoding="utf-8")
        with pytest.raises(ValueError, match=f"{key} 必须是有限正数"):
            calibrate.require_local_calibration()
    local.write_text(good.split("\n", 1)[0]+"\n- 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不是 YAML 映射"):
        calibrate.require_local_calibration()
    local.unlink()
    with pytest.raises(ValueError, match="^缺少基础标定"):
        calibrate.require_local_calibration()


def test_uncalibrated_placeholder_is_refused(local):
    """write_v7_calibration 只写了 σ_z（λ 还是 None）时，local 模式不收。"""
    local.unlink()
    w.write_v7_calibration(dict(sigma_z=1.25), "only sigma")
    with pytest.raises(ValueError, match="time_weight 必须是有限正数"):
        calibrate.require_local_calibration()


# ---------- calibrate-addon ----------

def test_calibrate_addon_with_local_base(small, tmp_path):
    output = tmp_path/"calibrate-addon-push"/"run"
    output.mkdir(parents=True)
    payload = calibrate.calibrate_addon(tmp_path/"features", output, CPU, base="local")
    assert payload["v7_calibration_sha256"] == cache.file_sha(small) and payload["sigma_z_used"] == 1.25
    for condition in calibration.CALIBRATED_CONDITIONS:
        record = json.loads((output/f"calibration-{condition}.json").read_text(encoding="utf-8"))
        assert record["base_calibration"] == "local" and record["v7_calibration_sha256"] == cache.file_sha(small)
        weights = calibration.load_addon_weights(condition)   # 训练任务读补跑标定时的核对和来源无关，照样通过
        assert weights == dict(time_weight=payload[condition]["time_weight"], space_weight=payload[condition]["space_weight"])
    with pytest.raises(ValueError, match="补跑标定文件已存在"):
        calibrate.calibrate_addon(tmp_path/"features", output, CPU, base="local")


def test_calibrate_addon_local_refuses_import_and_change(small, tmp_path, monkeypatch):
    out = lambda name: (tmp_path/"c"/name).mkdir(parents=True) or tmp_path/"c"/name
    with pytest.raises(ValueError, match="IMPORTED.json：v7 标定必须用"):
        calibrate.calibrate_addon(tmp_path/"features", out("a"), CPU)            # 默认 base 是 v7-import
    _imported(small)
    with pytest.raises(ValueError, match="是从 v7 导入的"):
        calibrate.calibrate_addon(tmp_path/"features", out("b"), CPU, base="local")
    (small.parent/"IMPORTED.json").unlink()
    original = calibrate.calibrate_condition

    def edit(condition, features, device):
        result = original(condition, features, device)
        if condition == "transformer_consistency":
            small.write_text(small.read_text(encoding="utf-8")+"# edited\n", encoding="utf-8")
        return result
    monkeypatch.setattr(calibrate, "calibrate_condition", edit)
    with pytest.raises(ValueError, match="^v7 calibration file changed during calibrate-addon$"):
        calibrate.calibrate_addon(tmp_path/"features", out("c"), CPU, base="local")
    assert not calibration.addon_calibration_path().exists()


# ---------- 入口和 run.py ----------

def test_entry_passes_base_calibration(local, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(w, "verify_bundle", lambda: "fictional-bundle")
    monkeypatch.setattr(w.environment, "verify_versions", lambda: dict(fictional=True))

    def fake_calibrate(features, output, device, base):
        calls.append(base)
        path = calibration.addon_calibration_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fictional: 1\n", encoding="utf-8")
    monkeypatch.setattr(calibrate, "calibrate_addon", fake_calibrate)
    args = Namespace(command="calibrate-addon", features_root=tmp_path/"f", device="cpu", output=tmp_path/"o/run", seed=1703,
                     base_calibration="local")
    assert entry.execute(args) == 0 and calls == ["local"]
    assert json.loads((tmp_path/"o/run/COMPLETE.json").read_text(encoding="utf-8"))["base_calibration"] == "local"
    with pytest.raises(ValueError, match="IMPORTED.json"):                       # 同一份文件按 v7-import 核：没有导入记录
        entry.execute(Namespace(**{**vars(args), "output": tmp_path/"p/run", "base_calibration": "v7-import"}))
    assert json.loads((tmp_path/"p/run/FAILED.json").read_text(encoding="utf-8"))["stage"] == "calibration" and calls == ["local"]
    seen = []
    monkeypatch.setattr(entry, "execute", lambda a: seen.append(a) or 0)
    base = ["--features-root", str(tmp_path), "--output", str(tmp_path/"q")]
    assert entry.main(["calibrate-addon", *base]) == 0 and seen[-1].base_calibration == "v7-import"
    assert entry.main(["check-addon", *base, "--base-calibration", "local"]) == 0 and seen[-1].base_calibration == "local"
    with pytest.raises(SystemExit):
        entry.main(["check-addon", *base, "--base-calibration", "v8"])


def _run_py():
    spec = importlib.util.spec_from_file_location("s26a_run_py", ROOT/"run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_py_forwards_base_calibration(tmp_path, monkeypatch):
    run = _run_py()
    seen = []
    monkeypatch.setattr(run.subprocess, "run", lambda command, **kw: seen.append(command) or Namespace(returncode=0))
    features = tmp_path/"features"
    features.mkdir()
    base = ["--features-root", str(features), "--system", "push", "--trajectories", "2000", "--updates", "10000"]
    for command in ("calibrate-addon", "check-addon"):
        seen.clear()
        assert run.main([command, *base, "--output", str(tmp_path/command), "--base-calibration", "local"]) == 0
        child = seen[0][seen[0].index("--")+1:]
        assert child[child.index("--base-calibration")+1] == "local" and child[5:7] == ["po_jepa.runtime.addon.entry", command]
        seen.clear()
        assert run.main([command, *base, "--output", str(tmp_path/f"{command}-default")]) == 0
        assert "--base-calibration" not in seen[0]                                # 不给就不传，入口用默认 v7-import
    extra = dict(run=["--condition", "koopman"], calibrate=[], check=[], transfer=["--condition", "identity", "--test-variant", "gain=2"],
                 d4=["--condition", "koopman", "--training-run", str(tmp_path)])
    for command, more in extra.items():
        argv = [command, *base, "--output", str(tmp_path/f"y-{command}"), *more]
        seen.clear()
        assert run.main(argv) == 0                                              # 不带 --base-calibration 时照常通过
        with pytest.raises(SystemExit) as error:
            run.main([*argv, "--base-calibration", "local"])
        assert error.value.code == 2


# ---------- 任务表 ----------

def _clean_env():
    import os
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")}
    env.update(PYTHONPATH=str(ROOT/"src"))
    return env


def _jobs(*args, ok=True):
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/make_v8_jobs.py"), *args], env=_clean_env(), capture_output=True, text=True)
    assert (r.returncode == 0) == ok, r.stderr
    return r.stdout.splitlines() if ok else r.stderr


def _tool(name):
    spec = importlib.util.spec_from_file_location(f"s26a_{name}", ROOT/"tools"/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_addon_condition_lists_agree():
    assert _tool("make_v8_jobs").ADDON_CONDITIONS == conditions.ADDON_CONDITIONS
    summarize = _tool("summarize_v7")
    assert summarize.ADDON_CONDITIONS == conditions.ADDON_CONDITIONS
    assert summarize.CONDITIONS == (*_tool("make_v8_jobs").CONDITIONS, *conditions.ADDON_CONDITIONS)
    assert set(summarize.INITIAL_SAME_AS) == set(conditions.ADDON_CONDITIONS)-{"koopman"}


def test_make_v8_jobs_addon_stage():
    from po_jepa import protocols
    lines = _jobs("--stage", "addon")
    commands = [l for l in lines if not l.startswith("#")]
    cal = [l for l in commands if " calibrate-addon " in l]
    check = [l for l in commands if " check-addon " in l]
    runs = [l for l in commands if l.startswith("python run.py run ")]
    assert len(cal) == len(protocols.SCENES) == 7 and len(check) == 1 and len(runs) == 7*3*5
    assert len(commands) == len(cal)+len(check)+len(runs)+1 and commands[-1].startswith("python tools/summarize_v7.py --results ../results-v8/full ")
    assert all("--base-calibration local" in l and "--trajectories 2000 --updates 10000" in l for l in cal+check)
    assert not any("--base-calibration" in l for l in runs)
    expected = {f"../results-v8/full/{scene}/seed{seed}/{c}"
                for scene in protocols.SCENES for seed in (1703, 1704, 1705) for c in conditions.ADDON_CONDITIONS}
    outputs = {shlex.split(l)[shlex.split(l).index("--output")+1] for l in runs}
    assert outputs == expected
    assert {shlex.split(l)[shlex.split(l).index("--condition")+1] for l in runs} == set(conditions.ADDON_CONDITIONS)
    # 评测阶段的 --job 路径和训练输出一致：long 阶段选补跑条件时读的就是上面这些目录
    long = [l for l in _jobs("--stage", "long", "--conditions", ",".join(conditions.ADDON_CONDITIONS), "--shards", "1")
            if l.startswith("python tools/long_rollout.py")]
    assert {shlex.split(l)[shlex.split(l).index("--job")+1] for l in long} == expected
    pilot = [l for l in _jobs("--stage", "addon-pilot") if not l.startswith("#")]
    assert len([l for l in pilot if l.startswith("python run.py run ")]) == 7*1*5
    assert all("--trajectories 200 --updates 2000" in l and "../results-v8/pilot/" in l for l in pilot if l.startswith("python run.py"))
    second = _jobs("--stage", "addon", "--scenes", "pendulum")
    assert second[0].startswith("# 注意：二阶场景 pendulum") and len([l for l in second if " run.py run " in f" {l}"]) == 3*5


def test_make_v8_jobs_conditions_option():
    for stage in ("variants", "strength", "long", "timing"):
        lines = _jobs("--stage", stage, "--scenes", "push", "--conditions", "fixed_fourier,koopman", "--shards", "1")
        evals = [l for l in lines if l.startswith(("python run.py transfer", "python tools/long_rollout.py", "python tools/timing_eval.py",
                                                  "python tools/plan_v7.py"))]
        assert evals and all(("fixed_fourier" in l) != ("koopman" in l) for l in evals)
        assert not any("--ac-checkpoint" in l for l in evals) and all("--no-baselines" in l for l in evals if "plan_v7" in l)
    space = [l for l in _jobs("--stage", "space", "--scenes", "push", "--conditions", "fixed_fourier,transformer_consistency", "--shards", "1")
             if l.startswith("python tools/space_eval.py")]
    assert space and all(("fixed_fourier" in l) != ("transformer_consistency" in l) for l in space)
    assert "koopman 还不能评空间采样" in _jobs("--stage", "space", "--conditions", "koopman", ok=False)
    assert "没有这些条件" in _jobs("--stage", "long", "--conditions", "fictional", ok=False)
    for stage in ("full", "pilot", "addon"):
        assert "--conditions 只用于" in _jobs("--stage", stage, "--conditions", "koopman", ok=False)
    for stage in ("pilot", "full", "variants", "strength", "long", "timing", "space"):      # 默认任务表不排补跑条件
        assert not any(c in l for l in _jobs("--stage", stage) for c in conditions.ADDON_CONDITIONS)


def test_every_addon_command_passes_run_py(tmp_path, monkeypatch):
    lines = _jobs("--stage", "addon", "--scenes", "push,landscape~shape=multi_well")
    run = _run_py()
    monkeypatch.setattr(run.subprocess, "run", lambda command, **kw: Namespace(returncode=0))
    monkeypatch.chdir(tmp_path)
    count = 0
    for line in lines:
        if not line.startswith("python run.py"):
            continue
        argv = shlex.split(line)
        Path(argv[argv.index("--features-root")+1]).mkdir(parents=True, exist_ok=True)
        assert run.main(argv[2:]) == 0, line
        count += 1
    assert count == 2+1+2*3*5


# ---------- 汇总 ----------

def _job(root, condition, seed, initial, dataset="push-2000", value=.5):
    folder = root/dataset/f"seed{seed}"/condition/"run"/"job"
    rows = folder/"evaluation"/condition/"rows"
    rows.mkdir(parents=True)
    (folder/"job.json").write_text(json.dumps(dict(condition=condition, dataset=dataset, seed=seed, updates=10000)))
    (folder/"COMPLETE.json").write_text("{}")
    (folder/"report.json").write_text(json.dumps(dict(registration_sha256="r", initial_model_sha256=initial, best=dict(step=1),
                                                      training=dict(consistency_monitor=None))))
    query = dict(kind="point", trajectory_id=1800, start_tick=0, duration_cycles=1, grid_size=16, stratum="id")
    (rows/"0.json").write_text(json.dumps(dict(status="ok", query=query, values=dict(future_jepa_distance=value, delta_act=0.,
                                                                                          tcd_normalized=0.))))


def _summarize(results, out, monkeypatch):
    s = _tool("summarize_v7")
    monkeypatch.setattr(sys, "argv", ["summarize_v7", "--results", str(results), "--output", str(out)])
    s.main()
    return json.loads((out/"summary.json").read_text(encoding="utf-8"))


V7 = dict(po_jepa_full="P", po_jepa_no_consistency="P", matched_transformer="T", identity=None, vjepa2_ac_small="A", vjepa2_ac="B")


def test_summarize_without_addon_conditions_is_unchanged(tmp_path, monkeypatch):
    for c, h in V7.items():
        _job(tmp_path/"r", c, 1703, h)
    summary = _summarize(tmp_path/"r", tmp_path/"o", monkeypatch)
    assert summary["missing"] == [] and summary["problems"] == []
    assert {r["condition"] for r in summary["table"]} == set(V7)


def test_summarize_with_addon_conditions(tmp_path, monkeypatch):
    for c, h in V7.items():
        _job(tmp_path/"r", c, 1703, h)
    addon = dict(po_jepa_time_only="P", po_jepa_space_only="P", fixed_fourier="P", transformer_consistency="T")
    for c, h in addon.items():
        _job(tmp_path/"r", c, 1703, h, value=.25)
    summary = _summarize(tmp_path/"r", tmp_path/"o", monkeypatch)
    assert summary["problems"] == []
    assert summary["missing"] == [dict(dataset="push-2000", updates=10000, seed=1703, condition="koopman")]
    assert {r["condition"] for r in summary["table"]} == set(V7)|set(addon)
    refs = {c["reference"]: c["future"]["mean"] for c in summary["comparisons"]}
    assert refs["fixed_fourier"] == pytest.approx(.25) and refs["matched_transformer"] == pytest.approx(0.)
    assert any(r.get("condition") == "fixed_fourier" for r in summary["across_seeds"])
    assert any(r.get("comparison") == "po_jepa_full − koopman" for r in summary["across_seeds"]) is False
    assert "fixed_fourier" in (tmp_path/"o/SUMMARY.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("condition,reference", [("po_jepa_time_only", "po_jepa_full"), ("po_jepa_space_only", "po_jepa_full"),
                                                 ("fixed_fourier", "po_jepa_full"), ("transformer_consistency", "matched_transformer")])
def test_summarize_checks_initial_weights(tmp_path, monkeypatch, condition, reference):
    for c, h in V7.items():
        _job(tmp_path/"r", c, 1703, h)
    _job(tmp_path/"r", condition, 1703, "different")
    _job(tmp_path/"r", "koopman", 1703, "K")
    summary = _summarize(tmp_path/"r", tmp_path/"o", monkeypatch)
    assert [p["problem"] for p in summary["problems"]] == [f"{condition} 与 {reference} 初始权重不同（08 §1.3）"]


# ---------- 补跑条件的真实结构走长时程/时间采样评测 ----------

from test_s25d_timing_eval import JIT, SW, caches, real  # noqa: E402,F401  （模块级 fixture：假编码器编的 double_well:40）


@pytest.mark.parametrize("spec,name", [(JIT, "jit"), (SW, "sw")])
def test_addon_models_run_through_timing_eval(caches, real, spec, name, monkeypatch, tmp_path):  # noqa: F811
    """未训练的真实结构 koopman、fixed_fourier、transformer_consistency（= matched_transformer 结构）在抖动（非整数时长）和
    半 tick 切换（tick 内换动作）的窗口上能跑完，输出有限，且不改模型状态。"""
    from po_jepa.fourier import FourierSpectrum, build_fixed_fourier
    from po_jepa.koopman import build_koopman
    from po_jepa.runtime import comparison, timing_eval
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    monkeypatch.setenv(checks.UPDATES_ENV, "10000")
    monkeypatch.setenv(checks.SEED_ENV, "1703")
    old = datasets._set_for_tests(spec)
    try:
        path = tmp_path/"cal.yaml"
        monkeypatch.setattr(checks, "v7_calibration_path", lambda: path)
        w.write_v7_calibration(dict(sigma_z=1.25, time_weight=.03, space_weight=4.0, rho=1.0), "test")
        config = checks.load_protocol()
        models = dict(koopman=build_koopman(conditions.koopman_config(config), checks.model_seed(), CPU),
                      fixed_fourier=build_fixed_fourier(config["model"], checks.model_seed(), CPU),
                      transformer_consistency=comparison.build_condition("matched_transformer", CPU)[0])
        assert type(models["fixed_fourier"].spectrum) is FourierSpectrum and models["koopman"].config.modes == 103
        for c, model in models.items():
            before = {k: v.clone() for k, v in model.state_dict().items()}
            rows, means = timing_eval.evaluate(model, caches[name], e06_config(), CPU)
            assert rows and all(math.isfinite(v) for r in rows for v in r["values"].values()), c
            assert all(torch.equal(before[k], v) for k, v in model.state_dict().items()), c
            if name == "jit":
                assert any(r["duration_ticks"] != int(r["duration_ticks"]) for r in rows)
    finally:
        datasets._set_for_tests(old)
