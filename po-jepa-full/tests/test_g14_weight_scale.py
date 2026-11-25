"""G14（2026-11-25）：一致性权重缩放（--time-weight-scale / --space-weight-scale）和“少动”分离实验的前提。

- 缩放乘在同一变体标定出的 λ 上，默认 1 时协议、变体 dict、标定文件名逐键不变；
- 缩放进 job.json 的 variant（评测工具 plan_v7._job_env 设回），汇总分组带 ~lt/~ls 后缀，标定文件名不带；
- calibrate / calibrate-addon / check-addon 和非基础标定条件拒收缩放；
- 分离实验的前提：两个权重都是 0（no_consistency）时，切分不改变任何一项损失和梯度的含义，只差 float32 舍入。
"""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from po_jepa import datasets
from po_jepa.runtime import checks, training, workflow

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def v7(monkeypatch):
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    monkeypatch.setenv(checks.UPDATES_ENV, "2000")
    for env in (checks.PARTITION_ENV, checks.READOUT_INIT_ENV, checks.TIME_SCALE_ENV, checks.SPACE_SCALE_ENV):
        monkeypatch.delenv(env, raising=False)
    old = datasets._set_for_tests(datasets.DatasetSpec("double_well", 200))
    yield
    datasets._set_for_tests(old)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_scale_changes_nothing(v7, monkeypatch):
    base = checks.load_protocol()
    variant, path = checks.v7_variant(), checks.v7_calibration_path()
    monkeypatch.setenv(checks.TIME_SCALE_ENV, "1")
    monkeypatch.setenv(checks.SPACE_SCALE_ENV, "1")
    assert checks.load_protocol() == base and checks.v7_variant() == variant == checks.V7_VARIANT_DEFAULT
    assert checks.v7_calibration_path() == path and path.name == "double_well-200-u2000.yaml"


@pytest.mark.parametrize("t,s", [("0.5", "1"), ("1", "0.25"), ("0.25", "0.5")])
def test_scale_multiplies_the_calibrated_weights_only(v7, monkeypatch, t, s):
    import yaml
    cal = yaml.safe_load(checks.v7_calibration_path().read_text(encoding="utf-8"))
    base = checks.load_protocol()
    monkeypatch.setenv(checks.TIME_SCALE_ENV, t)
    monkeypatch.setenv(checks.SPACE_SCALE_ENV, s)
    scaled = checks.load_protocol()
    assert scaled["loss"]["time_weight"] == cal["time_weight"]*float(t)
    assert scaled["loss"]["space_weight"] == cal["space_weight"]*float(s)
    # 只改这两个数：其他键（σ_z、模型、切分）不变；标定文件名和不缩放时相同
    strip = lambda c: {**c, "loss": {k: v for k, v in c["loss"].items() if k not in ("time_weight", "space_weight", "temporal_partition")}}
    assert strip(scaled) == strip(base) and scaled["loss"].get("temporal_partition", "bisection") == "bisection"
    assert checks.v7_calibration_path().name == "double_well-200-u2000.yaml"
    variant = checks.v7_variant()
    assert {k: variant[k] for k in variant if k.endswith("_scale")} == {
        k: v for k, v in (("time_weight_scale", t), ("space_weight_scale", s)) if v != "1"}
    assert checks.variant_suffix(variant) == "" and checks.weight_scale_suffix(variant) == \
        (f"~lt{t}" if t != "1" else "")+(f"~ls{s}" if s != "1" else "")


def test_scale_follows_the_variant_calibration(v7, monkeypatch, tmp_path):
    """缩放乘在本变体（这里 ~dyadic~zeroinit）的标定值上，不是默认变体的。"""
    monkeypatch.setenv(checks.PARTITION_ENV, "dyadic")
    monkeypatch.setenv(checks.READOUT_INIT_ENV, "zero")
    monkeypatch.setattr(checks, "ROOT", tmp_path)
    for name in ("configs/train/default.yaml", "configs/train/v4.yaml", "configs/train/v7.yaml"):
        (tmp_path/name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path/name).write_bytes((ROOT/name).read_bytes())
    monkeypatch.setattr(checks.data.cache, "verify_fixed_protocol", lambda: None)
    path = checks.v7_calibration_path()
    assert path.name == "double_well-200-u2000~dyadic~zeroinit.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("sigma_z: 0.7\ntime_weight: 3.0\nspace_weight: 5.0\nsource: test\n", encoding="utf-8")
    monkeypatch.setenv(checks.TIME_SCALE_ENV, "0.25")
    config = checks.load_protocol()
    assert (config["loss"]["time_weight"], config["loss"]["space_weight"]) == (0.75, 5.0)
    assert config["loss"]["temporal_partition"] == "dyadic" and checks.v7_calibration_path() == path


def test_bad_scale_and_v6_refused(v7, monkeypatch):
    monkeypatch.setenv(checks.TIME_SCALE_ENV, "0.3")
    with pytest.raises(ValueError, match=checks.TIME_SCALE_ENV):
        checks.v7_variant()
    monkeypatch.setenv(checks.TIME_SCALE_ENV, "0.5")
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v6")
    with pytest.raises(ValueError, match="v7"):
        checks.v7_variant()


def test_calibration_write_refused_under_scale(v7, monkeypatch, tmp_path):
    # 写到临时目录：守卫被改坏时也不能覆写 git 里的标定文件
    monkeypatch.setattr(checks, "v7_calibration_path", lambda: tmp_path/"cal.yaml")
    monkeypatch.setattr(workflow, "require_calibrated_weights", lambda **k: None, raising=False)
    monkeypatch.setenv(checks.SPACE_SCALE_ENV, "0.5")
    with pytest.raises(ValueError, match="缩放"):
        workflow.write_v7_calibration(dict(time_weight=1.), "test")


def test_registration_and_job_record_carry_the_scale(v7, monkeypatch, tmp_path):
    from po_jepa.runtime import v7jobs
    monkeypatch.setenv(checks.TIME_SCALE_ENV, "0.5")
    variant = checks.v7_variant()
    assert variant == dict(checks.V7_VARIANT_DEFAULT, time_weight_scale="0.5")
    assert v7jobs.with_variant(dict(condition="po_jepa_full"))["variant"] == variant


def test_job_env_restores_scale(tmp_path, monkeypatch):
    module = _load(ROOT/"tools/plan_v7.py", "plan_v7_g14")
    monkeypatch.setattr(os, "environ", dict(os.environ, PO_JEPA_TIME_WEIGHT_SCALE="0.25", PO_JEPA_SPACE_WEIGHT_SCALE="0.5"))
    job = dict(seed=1703, updates=2000, dataset="double_well-200", condition="po_jepa_full")
    (tmp_path/"a").mkdir()
    (tmp_path/"a/job.json").write_text(json.dumps(job))
    module._job_env(tmp_path/"a")
    assert (os.environ["PO_JEPA_TIME_WEIGHT_SCALE"], os.environ["PO_JEPA_SPACE_WEIGHT_SCALE"]) == ("1", "1")
    (tmp_path/"b").mkdir()
    (tmp_path/"b/job.json").write_text(json.dumps(dict(job, variant=dict(temporal_partition="dyadic", readout_init="xavier",
                                                                         time_weight_scale="0.5"))))
    module._job_env(tmp_path/"b")
    assert (os.environ["PO_JEPA_TIME_WEIGHT_SCALE"], os.environ["PO_JEPA_SPACE_WEIGHT_SCALE"]) == ("0.5", "1")
    assert os.environ["PO_JEPA_TEMPORAL_PARTITION"] == "dyadic"


def test_summary_groups_scales_apart():
    module = _load(ROOT/"tools/summarize_v7.py", "summarize_v7_g14")
    for variant in (dict(temporal_partition="dyadic", readout_init="zero", time_weight_scale="0.5"),
                    dict(temporal_partition="dyadic", readout_init="xavier", time_weight_scale="0.25", space_weight_scale="0.5"),
                    dict(temporal_partition="bisection", readout_init="xavier", space_weight_scale="0.25")):
        assert module._variant_suffix(dict(variant=variant)) == checks.variant_suffix(variant)+checks.weight_scale_suffix(variant)
    assert module._variant_suffix(dict(variant=dict(temporal_partition="dyadic", readout_init="zero", time_weight_scale="0.5"))) \
        == "~dyadic~zeroinit~lt0.5"


def _run_py(monkeypatch, tmp_path, *extra, env=None):
    module = _load(ROOT/"run.py", "run_g14")
    seen = []
    monkeypatch.setattr(module.subprocess, "run", lambda cmd, check, env: seen.append(env) or subprocess.CompletedProcess(cmd, 0))
    if env:
        for k, v in env.items():
            monkeypatch.setenv(k, v)
    (tmp_path/"f").mkdir(exist_ok=True)
    argv = ["--system", "double_well", "--trajectories", "200", "--updates", "2000", "--features-root", str(tmp_path/"f"),
            "--output", str(tmp_path/f"o{len(list(tmp_path.iterdir()))}"), *extra]
    return module, module.main(argv), seen


def test_run_py_passes_scale_flags(monkeypatch, tmp_path):
    _, code, seen = _run_py(monkeypatch, tmp_path, "run", "--condition", "po_jepa_full", "--time-weight-scale", "0.5")
    assert code == 0 and seen[-1]["PO_JEPA_TIME_WEIGHT_SCALE"] == "0.5" and "PO_JEPA_SPACE_WEIGHT_SCALE" not in seen[-1]
    _, code, seen = _run_py(monkeypatch, tmp_path, "run", "--condition", "po_jepa_time_only", "--time-weight-scale", "0.25",
                            "--space-weight-scale", "0.5")
    assert (seen[-1]["PO_JEPA_TIME_WEIGHT_SCALE"], seen[-1]["PO_JEPA_SPACE_WEIGHT_SCALE"]) == ("0.25", "0.5")
    with pytest.raises(SystemExit):   # 和已设的环境变量不同
        _run_py(monkeypatch, tmp_path, "run", "--condition", "po_jepa_full", "--time-weight-scale", "0.5",
                env=dict(PO_JEPA_TIME_WEIGHT_SCALE="0.25"))


@pytest.mark.parametrize("extra", [("calibrate",), ("calibrate-addon",), ("check-addon",), ("check",),
                                   ("equal-step", "--condition", "po_jepa_no_consistency", "--training-run", "a", "--limit-run", "b"),
                                   ("run", "--condition", "po_jepa_no_consistency"), ("run", "--condition", "transformer_consistency"),
                                   ("run", "--condition", "po_jepa_rollout_sup"), ("run", "--condition", "fixed_fourier")])
def test_run_py_refuses_scale_where_it_has_no_meaning(monkeypatch, tmp_path, extra):
    with pytest.raises(SystemExit):
        _run_py(monkeypatch, tmp_path, *extra, "--time-weight-scale", "0.5")
    # G14-fix1（Codex G14 P1）：不给参数、只继承环境变量也拦
    with pytest.raises(SystemExit):
        _run_py(monkeypatch, tmp_path, *extra, env=dict(PO_JEPA_SPACE_WEIGHT_SCALE="0.25"))
    monkeypatch.delenv("PO_JEPA_SPACE_WEIGHT_SCALE")
    _, code, _ = _run_py(monkeypatch, tmp_path, *extra, "--time-weight-scale", "1")   # 默认值照常放行
    assert code == 0


# ---------- 分离实验的前提：no_consistency 下切分不改变训练（只差 float32 舍入）----------

def _first_gradient(partition, condition, init, weights):
    """一次 train_step 后（clip 之前的）全部参数梯度拼成一个向量；clip_gradients 换成只记录。"""
    from test_s28b_dyadic_zero_init import make_model, make_batch, config
    model, batch = make_model(readout_init=init), make_batch([1., 2., 4., 8.])
    model.train()
    optimizer = training.make_optimizer(model)
    cfg = replace(config(partition), time_weight=weights[0], space_weight=weights[1])
    result = training.train_step(model, optimizer, batch, loss_config=cfg, completed_steps=0, condition=condition)
    return torch.cat([p.grad.detach().double().flatten() for p in model.parameters()]), result.loss_values


def _relative(a, b):
    return ((a-b).norm()/a.norm()).item()


@pytest.mark.parametrize("init", ["xavier", "zero"])
def test_no_consistency_partition_only_rounds(init):
    """权重都为 0 时，二等分和多段切分的 JEPA 损失逐位相同（同一条长支），整体梯度只差 float32 舍入（零权重时一致性分支仍在
    计算图里，两种切分的图不同，浮点梯度累加的过程不同；精确的内部来源没有追）：相对差 < 1e-4（实测 0～2e-5，随线程数变）。AdamW 会把梯度几乎为 0 的个别参数上的舍入放大，所以两边训练不逐位相同；
    G14 里 no_consistency 不分切分，dyadic-zero 那一条只当数值噪声（同设定的另一次实现）参照。"""
    (gb, lb), (gd, ld) = (_first_gradient(p, "no_consistency", init, (0., 0.)) for p in ("bisection", "dyadic"))
    assert lb["jepa"] == ld["jepa"] and lb["total"] == ld["total"]
    assert _relative(gb, gd) < 1e-4


def test_time_only_partition_changes_the_gradient():
    """对照：时间权重为正时，切分真的改变梯度（上面的检查不是恒真的）。"""
    (gb, _), (gd, _) = (_first_gradient(p, "time_only", "xavier", (.5, 0.)) for p in ("bisection", "dyadic"))
    assert _relative(gb, gd) > 1e-3


# ---------- G14-fix1（Codex G14 P1、P2）：子进程里只靠环境变量也拦住 ----------

@pytest.mark.parametrize("condition", ["po_jepa_no_consistency", "identity", "transformer_consistency", "fixed_fourier",
                                       "po_jepa_rollout_sup", "koopman", "matched_transformer"])
def test_build_condition_refuses_scale_for_other_conditions(v7, monkeypatch, condition):
    from po_jepa.runtime import comparison
    monkeypatch.setenv(checks.SPACE_SCALE_ENV, "0.5")
    with pytest.raises(ValueError, match="缩放"):
        comparison.build_condition(condition, torch.device("cpu"))


def test_build_condition_scales_the_allowed_conditions(v7, monkeypatch):
    from po_jepa.runtime import comparison
    base = {c: comparison.build_condition(c, torch.device("cpu"))[1] for c in checks.WEIGHT_SCALE_CONDITIONS}
    monkeypatch.setenv(checks.TIME_SCALE_ENV, "0.25")
    for c, loss in base.items():
        scaled = comparison.build_condition(c, torch.device("cpu"))[1]
        assert scaled.time_weight == loss.time_weight*0.25 and scaled.space_weight == loss.space_weight


def test_addon_entry_and_calibration_refuse_scale(v7, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from po_jepa.runtime.addon import calibrate, entry
    monkeypatch.setenv(checks.TIME_SCALE_ENV, "0.5")
    for command in ("calibrate-addon", "check-addon"):
        with pytest.raises(ValueError, match="缩放"):
            entry.execute(SimpleNamespace(command=command, device="cpu", output=str(tmp_path/command), features_root=str(tmp_path),
                                          base_calibration="local", seed=1703))
        assert not (tmp_path/command).exists()
    with pytest.raises(ValueError, match="缩放"):
        calibrate.calibrate_addon(tmp_path, tmp_path/"out", torch.device("cpu"), base="local")


def test_equal_step_refuses_scale(v7, monkeypatch, tmp_path):
    from po_jepa.runtime import v7jobs
    monkeypatch.setenv(checks.TIME_SCALE_ENV, "0.5")
    with pytest.raises(ValueError, match="equal-step"):
        v7jobs.equal_step_job(tmp_path, tmp_path/"out", torch.device("cpu"), "po_jepa_no_consistency",
                              training_run=tmp_path, limit_run=tmp_path)
    assert not (tmp_path/"out").exists()
