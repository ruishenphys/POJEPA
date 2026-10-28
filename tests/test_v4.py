"""v4 整改（导师确认的 P1–P7）的 CPU 单元测试：编码器特征建图、固定 K、残差读出、LN 空间一致性、预热与标定。"""
from dataclasses import replace
import math

import pytest
import torch
import yaml

from po_jepa.actions import ControlSpec, from_physical_segments
from po_jepa.dynamics import DynamicsConfig, IntegratorConfig, ModalDynamics
from po_jepa.geometry import build_spatial_transfer, regular_grid
from po_jepa.losses import (LossConfig, spatial_consistency_loss, spatial_predictions, temporal_consistency_loss,
                            temporal_predictions, weighted_total_loss, jepa_distance)
from po_jepa.model import POJEPA, ProjectionConfig
from po_jepa.runtime import checks, training
from po_jepa.spectral import GraphConfig, GraphSpectrum

E = 32
SPEC = ControlSpec(2, .25, ("u_x", "u_y"), ("p/s", "p/s"), (-.5, -.5), (.5, .5), (1., 1.), (0., 0.), "test")


def _model(*, readout="residual", source="encoder", k=8, fixed=True, sigma_z=4.):
    torch.manual_seed(1703)
    cfg = checks.load_protocol()["model"]
    graph = GraphConfig(16, 1703, .125, sigma_z, k, 1e-10, 1e-8, affinity_source=source,
                        feature_dim=E if source == "encoder" else 64, fixed_modes=fixed)
    return POJEPA(ProjectionConfig(E, 64, readout=readout), GraphSpectrum(graph),
                  ModalDynamics(DynamicsConfig(**cfg["dynamics"])), IntegratorConfig(**cfg["integrator"]))


def _controls(cycles, u):
    b = len(cycles)
    end = torch.tensor(cycles, dtype=torch.float64)*.25
    return from_physical_segments(starts_seconds=torch.zeros(b, 1, dtype=torch.float64), ends_seconds=end[:, None],
        u_values=torch.tensor(u, dtype=torch.float32)[:, None, :], n_segments=torch.ones(b, dtype=torch.int64),
        segment_valid=torch.ones(b, 1, dtype=torch.bool), origin_seconds=torch.zeros(b, dtype=torch.float64),
        control_spec=SPEC, source_ids=tuple(f"s{i}" for i in range(b)))


def _z(b=2, h=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 1, E, generator=g)*2+torch.randn(b, h*h, E, generator=g)).float()


def test_overlay_merges_onto_unchanged_default():
    """v4 覆盖层只改模型/损失字段；default.yaml 的数据协议保持原值，旧缓存 SHA 不受影响。"""
    config = checks.load_protocol()
    base = yaml.safe_load((checks.ROOT/"configs/train/default.yaml").read_text(encoding="utf-8"))
    assert config["model"]["graph"]["affinity_source"] == "encoder" and config["model"]["graph"]["fixed_modes"] is True
    assert config["model"]["projection"]["readout"] == "residual" and config["loss"]["consistency_metric"] == "input_scale"
    assert config["data"] == base["data"] and config["encoder"] == base["encoder"] and config["grids"] == base["grids"]
    assert config["model"]["dynamics"] == base["model"]["dynamics"]
    assert GraphConfig(**config["model"]["graph"]).feature_dim == config["encoder"]["encoder_dim"]


def test_encoder_affinity_ignores_trainable_projection():
    """P1：编码器模式建图只依赖 Z，P_in 改变不影响 W/谱；Q 为 [16,E] 正交行。"""
    model = _model()
    assert model.spectrum.q.shape == (16, E)
    assert torch.allclose(model.spectrum.q.double()@model.spectrum.q.double().T, torch.eye(16, dtype=torch.float64), atol=1e-5)
    z, grid = _z(), regular_grid(8, 8)
    before = model.prepare_query(z, grid).spectrum.basis_cpu
    with torch.no_grad():
        model.p_in.weight.mul_(3.)
    after = model.prepare_query(z, grid).spectrum.basis_cpu
    assert torch.equal(before, after)


def test_projected_mode_keeps_first_round_q():
    """projected 模式与一轮完全相同（同种子同 Q）。"""
    old = GraphSpectrum(GraphConfig(16, 1703, .25, 1., 128, 1e-10, 1e-8))
    new = GraphSpectrum(GraphConfig(16, 1703, .25, 1., 128, 1e-10, 1e-8, affinity_source="projected", feature_dim=64))
    assert torch.equal(old.q, new.q) and old.q.shape == (16, 64)


def test_fixed_modes_same_k_on_every_grid_and_rejects_small_grid():
    """P2：所有网格 K 相同；N<K 报错而不是静默截成 N。"""
    model = _model(k=16)
    for h in (8, 6, 4):
        z = _z(h=h)
        if h*h >= 16:
            assert model.prepare_query(z, regular_grid(h, h)).spectrum.basis.shape[-1] == 16
        else:
            with pytest.raises(ValueError, match="fixed_modes"):
                model.prepare_query(z, regular_grid(h, h))


def test_residual_readout_is_exact_identity_at_zero_duration():
    """P6：零时长输出逐位等于输入 Z；direct 读出不满足。"""
    z, grid = _z(), regular_grid(8, 8)
    controls, zero = _controls([1., 1.], [[.1, -.2], [0., .3]]), torch.zeros(2, dtype=torch.float64)
    assert torch.equal(_model()(z, controls, zero, grid), z)
    assert not torch.allclose(_model(readout="direct")(z, controls, zero, grid), z)


def test_residual_readout_keeps_all_parameter_gradients_finite():
    """残差读出下 p_out.bias 梯度为 0 但存在且有限，train_step 的完整梯度检查不受影响。"""
    model = _model()
    z, grid = _z(), regular_grid(8, 8)
    out = model(z, _controls([2., 2.], [[.1, -.2], [0., .3]]), torch.tensor([2., 2.], dtype=torch.float64), grid)
    out.square().mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert torch.count_nonzero(model.p_out.bias.grad) == 0


def _consistency(model, config):
    z, grid = _z(seed=3), regular_grid(8, 8)
    controls = _controls([2., 4.], [[.1, -.2], [0., .3]])
    tau = torch.tensor([1., 2.], dtype=torch.float64)
    t = temporal_consistency_loss(temporal_predictions(model, z, controls, tau, tau.clone(), grid), config=config)
    transfer = build_spatial_transfer(grid, target_height=4, target_width=4)
    s = spatial_consistency_loss(spatial_predictions(model, z, controls, tau*2, transfer), config=config)
    return t, s


def test_ln_mean_consistency_is_output_scale_invariant_and_small():
    """P3：LN 空间均值版不能靠缩小 P_out 输出尺度降低（堵住尺度捷径），且量级为 O(1) 而非 raw 平方和。"""
    base = LossConfig(1., 1e-5, 1e-8, 1e-8, 1., 1.)
    ln = replace(base, consistency_metric="ln_mean")
    model = _model(readout="direct", k=8)
    t_ln, s_ln = _consistency(model, ln)
    t_raw, _ = _consistency(model, base)
    with torch.no_grad():
        model.p_out.weight.mul_(.1); model.p_out.bias.mul_(.1)
    t_ln2, s_ln2 = _consistency(model, ln)
    t_raw2, _ = _consistency(model, base)
    # 空间两支都直接作用于未缩放输入，LN 后严格不变；时间第二段以缩放后的中间预测为输入，只要求不再按平方缩小。
    assert torch.allclose(s_ln.per_sample, s_ln2.per_sample, rtol=1e-2)  # 仅 LN eps 带来的微小差
    assert float(t_raw2.mean) < .05*float(t_raw.mean) and float(t_ln2.mean) > .5*float(t_ln.mean)
    assert float(t_ln.mean) < 100 and float(s_ln.mean) < 100


def test_ln_mean_matches_explicit_formula():
    """ln_mean = mean_{N,E}(LN(a)-LN(b))^2/((tau sigma)^2+eps)。"""
    ln = LossConfig(1., 1e-5, 1e-8, 1e-8, 1., 1., consistency_metric="ln_mean")
    model = _model()
    z, grid = _z(seed=4), regular_grid(8, 8)
    tau = torch.tensor([.5, 1.], dtype=torch.float64)
    p = temporal_predictions(model, z, _controls([1., 2.], [[.1, 0.], [0., .2]]), tau, tau.clone(), grid)
    n = torch.nn.functional.layer_norm
    expected = ((n(p.direct, (E,), eps=1e-5)-n(p.composed, (E,), eps=1e-5)).double().square().mean((1, 2)))/((tau*tau)**2+1e-8)
    assert torch.allclose(temporal_consistency_loss(p, config=ln).per_sample, expected, rtol=1e-6)


def test_warmup_scale_and_weighted_total():
    """P7：w(s)=min(1,s/W)；s=0 时一致性加权为 0 且不报“正权重下溢”。"""
    cfg = LossConfig(1., 1e-5, 1e-8, 1e-8, .07, .3, consistency_metric="ln_mean", consistency_warmup_updates=200)
    assert [training.consistency_warmup_scale(cfg, s) for s in (0, 100, 200, 500)] == [0., .5, 1., 1.]
    assert training.consistency_warmup_scale(replace(cfg, consistency_warmup_updates=0), 0) == 1.
    from po_jepa.losses import ReducedLoss
    one = lambda v, d: ReducedLoss(torch.tensor([v], dtype=d), torch.tensor(v, dtype=d))
    total = weighted_total_loss(one(1., torch.float32), one(2., torch.float64), one(3., torch.float64), config=cfg, consistency_scale=0.)
    assert float(total.total) == 1.
    total = weighted_total_loss(one(1., torch.float32), one(2., torch.float64), one(3., torch.float64), config=cfg, consistency_scale=.5)
    assert math.isclose(float(total.total), 1+.5*(.07*2+.3*3))
    with pytest.raises(ValueError):
        weighted_total_loss(one(1., torch.float32), one(2., torch.float64), one(3., torch.float64), config=cfg, consistency_scale=1.5)


def test_loss_config_rejects_unknown_metric_and_negative_warmup():
    with pytest.raises(ValueError):
        LossConfig(1., 1e-5, 1e-8, 1e-8, 1., 1., consistency_metric="l1")
    with pytest.raises(ValueError):
        LossConfig(1., 1e-5, 1e-8, 1e-8, 1., 1., consistency_warmup_updates=-1)


def _production_batch():
    """生产维度 1408、16×16→8×8 的虚构批次；复用 E06 夹具的控制与时长，只替换 Z 与网格。"""
    from test_training import e06_case
    from po_jepa.runtime import comparison
    _, batch = e06_case(encoder_dim=1408)
    g = torch.Generator().manual_seed(0)
    raw = torch.randn(2, 256, 1408, generator=g)*3
    return replace(batch, raw_z=raw, raw_target=torch.roll(raw, 3, 1),
                   transfer=build_spatial_transfer(regular_grid(16, 16), target_height=8, target_width=8))


def test_preflight_diagnostic_step_accepts_residual_readout_on_cpu():
    """check 的 diagnostic_step 用生产 v4 模型：5 次 Phi、残差读出 10 次 P_out，K 固定且三项损失都反传。"""
    from po_jepa.runtime import comparison
    model, loss = comparison.build_condition("po_jepa_full", torch.device("cpu"))
    model.train()
    assert model.readout == "residual" and checks.readouts_per_phi(model) == 2
    report = checks.diagnostic_step(model, comparison.make_optimizer(model), _production_batch(), loss)
    assert report["phi_calls"] == 5 and len(report["raw_predictions"]) == 10
    assert [q["basis"]["shape"][-1] for q in report["query_records"]] == [model.spectrum.config.max_modes]*5


@pytest.mark.parametrize("condition", ["po_jepa_full", "po_jepa_no_consistency"])
def test_training_loop_observed_update_accepts_residual_readout_on_cpu(condition):
    """正式训练循环的计数检查用真实 v4 条件更新一步，不再按直接读出的 5 次 P_out 拒绝。"""
    from po_jepa.runtime import comparison
    model, loss = comparison.build_condition(condition, torch.device("cpu"))
    model.train()
    result, scales = comparison._observed_update(condition, model, comparison.make_optimizer(model), _production_batch(), loss, 0)
    assert result.completed_steps == 1 and len(scales) == 10


# ---- v4 第一次 GPU 运行后的纯代码修复（C2/C3/C5/C7/E02/E03/E04） ----

def test_calibration_uses_median_of_per_batch_gradient_norms(monkeypatch, tmp_path):
    """C2：一个离群 batch 不再把 lambda 拉偏；结果里保留逐 batch 数值、中位数和均值。"""
    from po_jepa.runtime import workflow as w
    rows = [dict(jepa=1., time=1., space=1., grad_jepa=g, grad_time=2., grad_space=4.) for g in (10., 10., 10000.)]
    feed = iter(rows)
    monkeypatch.setattr(w.comparison, "build_condition", lambda *a: (torch.nn.Linear(1, 1), None))
    monkeypatch.setattr(w.comparison.learning, "make_query_sequence", lambda c: ([[None]]*3, None))
    monkeypatch.setattr(w.data, "load_training_batch", lambda *a, **k: None)
    monkeypatch.setattr(w.training, "consistency_gradient_norms", lambda *a, **k: next(feed))
    real = w.checks.load_protocol
    monkeypatch.setattr(w.checks, "load_protocol", lambda: {**real(), "calibration": dict(rho=.05, batches=3)})
    result = w.calibrate_weights(None, tmp_path/"cal", torch.device("cpu"))
    assert result["statistic"] == "median" and len(result["per_batch"]) == 3
    assert result["time_weight"] == pytest.approx(.05*10/2) and result["space_weight"] == pytest.approx(.05*10/4)
    assert result["mean"]["grad_jepa"] > 3000


def test_v4_overlay_enables_clipping_and_32_calibration_batches():
    """C2/C3：覆盖层默认 32 个标定 batch、全局梯度范数裁剪 1.0；default.yaml 的 SHA 不受影响。"""
    config = checks.load_protocol()
    assert config["calibration"]["batches"] == 32 and config["loss"]["grad_clip_norm"] == 1.0
    with pytest.raises(ValueError):
        LossConfig(1., 1e-5, 1., 1., 0., 0., grad_clip_norm=-1.)


@pytest.mark.parametrize("clip", [0., 1e-3])
def test_train_step_clips_global_gradient_norm_and_logs_pre_clip_norm(clip):
    """C3：裁剪发生在 step 前，返回值记录裁剪前范数；clip=0 时不改梯度。"""
    from po_jepa.runtime import comparison
    model, loss = comparison.build_condition("po_jepa_full", torch.device("cpu"))
    model.train()
    loss = replace(loss, grad_clip_norm=clip)
    optimizer = comparison.make_optimizer(model)
    result = training.train_step(model, optimizer, _production_batch(), loss_config=loss, completed_steps=0)
    after = math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters() if p.grad is not None))
    assert result.grad_norm > 0 and math.isfinite(result.grad_norm)
    if clip:
        assert after == pytest.approx(clip, rel=1e-3) and result.grad_norm > clip
    else:
        assert after == pytest.approx(result.grad_norm, rel=1e-6)


def test_checkpoint_model_config_records_residual_readout():
    """C5：checkpoint 的 model_config 记录实际读出方式，不再一律写 direct。"""
    assert training._model_config(_model(readout="residual"))["projection"]["readout"] == "residual"
    assert training._model_config(_model(readout="direct"))["projection"]["readout"] == "direct"


def test_run_entry_resolves_relative_features_root(monkeypatch, tmp_path):
    """C7：子进程在代码根目录运行，--features-root 先按调用者当前目录解析成绝对路径；不存在时直接报错。"""
    import importlib.util, subprocess
    from po_jepa.runtime import workflow as w
    spec = importlib.util.spec_from_file_location("v4_run_entry", w.ROOT/"run.py")
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    (tmp_path/"feats").mkdir(); monkeypatch.chdir(tmp_path); seen = []
    monkeypatch.setattr(subprocess, "run", lambda command, check: seen.append(command) or subprocess.CompletedProcess(command, 0))
    assert entry.main(["check", "--features-root", "feats", "--device", "cpu", "--output", "out"]) == 0
    child = seen[0]
    assert child[child.index("--features-root")+1] == str((tmp_path/"feats").resolve())
    with pytest.raises(SystemExit):
        entry.main(["check", "--features-root", "missing", "--output", "out2"])


def test_evaluation_runtime_comparison_reports_differences(tmp_path):
    """E02：评价记录当前运行签名并与训练 runtime.json 对照，只报告差异项。"""
    from po_jepa.runtime import environment, workflow as w
    assert w.runtime_comparison(tmp_path, torch.device("cpu"))["matches"] is None
    runtime = environment.runtime_signature(torch.device("cpu"))
    w.cache._write_json(tmp_path/"runtime.json", runtime)
    assert w.runtime_comparison(tmp_path, torch.device("cpu"))["matches"] is True
    (tmp_path/"other").mkdir()
    w.cache._write_json(tmp_path/"other"/"runtime.json", {**runtime, "cpu_threads": runtime["cpu_threads"]+1})
    result = w.runtime_comparison(tmp_path/"other", torch.device("cpu"))
    assert result["matches"] is False and result["differences"] == ["cpu_threads"]


def test_numeric_health_summary_flags_spikes_and_trend():
    """E03：100 次试跑的 JEPA 走势、梯度尖峰次数与读出放大比例，只报告不阻断。"""
    from po_jepa.runtime import resources
    trace = [dict(jepa=1.-i/200, grad_max_abs=1., grad_norm=1.) for i in range(100)]
    trace[50]["grad_norm"] = 1e4
    health = resources.numeric_health(trace, [0.5])
    assert health["jepa_decreased"] is True and health["gradient_spikes"] == 1 and health["report_only"]
    assert resources.numeric_health([dict(jepa=None, grad_max_abs=None, grad_norm=None)], None)["jepa_decreased"] is None


def test_readout_ratio_measures_change_of_direct_prediction():
    """E03：初始化时残差读出的变化幅度有限且非负。"""
    from po_jepa.runtime import comparison, resources
    model, _ = comparison.build_condition("po_jepa_full", torch.device("cpu"))
    ratio = resources.readout_ratio(model, _production_batch())
    assert len(ratio) == 2 and all(math.isfinite(r) and r >= 0 for r in ratio)


def test_verified_cache_is_reused_until_files_change(monkeypatch, tmp_path):
    """E04：同一轨迹只完整核验一次；文件改动（mtime/size 变化）后重新核验。"""
    import os
    from po_jepa.runtime import cache
    for name in ("COMPLETE.json", "manifest.json", "raw_z.npy"):
        (tmp_path/name).write_bytes(b"x")
    calls = []
    monkeypatch.setattr(cache, "_VERIFIED_CACHE", {})
    monkeypatch.setattr(cache, "_read_complete_cache_uncached", lambda p: calls.append(p) or ("raw", {}, "sha"))
    assert cache._read_complete_cache(tmp_path) == cache._read_complete_cache(tmp_path) == ("raw", {}, "sha")
    assert len(calls) == 1
    os.utime(tmp_path/"raw_z.npy", ns=(1, 1))
    cache._read_complete_cache(tmp_path)
    assert len(calls) == 2
    (tmp_path/"failure.json").write_text("{}")
    with pytest.raises(ValueError, match="failure"):
        cache._read_complete_cache(tmp_path)


# ---- v5：撤回逐路径 LN，一致性按输入尺度归一化 ----

def _shortcut_predictions(c):
    """构造塌缩时的形态：两条路径共有一个放大 c 倍的分量 G，真正的差别 a-b 固定。"""
    from po_jepa.losses import TemporalPredictions
    g = torch.Generator().manual_seed(7)
    z, common = torch.randn(2, 16, E, generator=g)*3, torch.randn(2, 16, E, generator=g)*3
    a, b = torch.randn(2, 16, E, generator=g), torch.randn(2, 16, E, generator=g)
    first, second = z+c*common+a, z+c*common+b
    tau = torch.ones(2, dtype=torch.float64)
    from po_jepa.losses import input_scale
    return TemporalPredictions(first, first, second, first-second, tau, tau.clone(), input_scale(z))


def test_input_scale_metric_closes_amplification_shortcut():
    """v5：LN 逐路径归一化下，共同分量放大 c 倍会让缺陷趋于 0（v4 塌缩机制）；按输入尺度归一化时缺陷不随 c 变化。"""
    ln = LossConfig(1., 1e-5, 1e-8, 1e-8, 1., 1., consistency_metric="ln_mean")
    inp = replace(ln, consistency_metric="input_scale")
    ln_values = [float(temporal_consistency_loss(_shortcut_predictions(c), config=ln).mean) for c in (1., 10., 100.)]
    inp_values = [float(temporal_consistency_loss(_shortcut_predictions(c), config=inp).mean) for c in (1., 10., 100.)]
    assert ln_values[2] < .01*ln_values[0]
    assert inp_values[0] == pytest.approx(inp_values[1]) == pytest.approx(inp_values[2])


def test_input_scale_metric_matches_formula_and_is_scale_equivariant():
    """v5：delta = mean_{N,E}(A-B)^2 / var(Z_t) / ((tau sigma)^2+eps)；Z 与输出同乘 k 时缺陷不变，只放大差时按 k^2 增大。"""
    inp = LossConfig(1., 1e-5, 1e-8, 1e-8, 1., 1., consistency_metric="input_scale")
    model = _model()
    z, grid = _z(seed=4), regular_grid(8, 8)
    tau = torch.tensor([.5, 1.], dtype=torch.float64)
    p = temporal_predictions(model, z, _controls([1., 2.], [[.1, 0.], [0., .2]]), tau, tau.clone(), grid)
    var = z.double().var(dim=(1, 2), unbiased=False)
    expected = (p.difference.double().square().mean((1, 2))/var)/((tau*tau)**2+1e-8)
    assert torch.allclose(temporal_consistency_loss(p, config=inp).per_sample, expected, rtol=1e-10)
    k = 5.
    scaled = replace(p, direct=p.direct*k, composed=p.composed*k, difference=p.difference*k, input_scale=p.input_scale*k*k)
    assert torch.allclose(temporal_consistency_loss(scaled, config=inp).per_sample, expected, rtol=1e-6)
    doubled = replace(p, difference=p.difference*2)
    assert torch.allclose(temporal_consistency_loss(doubled, config=inp).per_sample, 4*expected, rtol=1e-6)


def test_input_scale_metric_requires_scale_and_overlay_uses_it():
    """v5：覆盖层默认用 input_scale；缺少输入尺度时拒绝，而不是悄悄退回别的度量。"""
    assert checks.load_protocol()["loss"]["consistency_metric"] == "input_scale"
    inp = LossConfig(1., 1e-5, 1e-8, 1e-8, 1., 1., consistency_metric="input_scale")
    with pytest.raises(ValueError, match="input scale"):
        temporal_consistency_loss(replace(_shortcut_predictions(1.), input_scale=None), config=inp)


def test_spatial_input_scale_uses_fine_grid_input():
    """v5：空间项两侧除以同一个细网格输入尺度。"""
    inp = LossConfig(1., 1e-5, 1e-8, 1e-8, 1., 1., consistency_metric="input_scale")
    model = _model()
    z = _z(seed=5)
    transfer = build_spatial_transfer(regular_grid(8, 8), target_height=4, target_width=4)
    duration = torch.tensor([1., 2.], dtype=torch.float64)
    s = spatial_predictions(model, z, _controls([1., 2.], [[.1, 0.], [0., .2]]), duration, transfer)
    var = z.double().var(dim=(1, 2), unbiased=False)
    assert torch.allclose(s.input_scale, var)
    expected = (s.difference.double().square().mean((1, 2))/var)/(duration**2+1e-8)
    assert torch.allclose(spatial_consistency_loss(s, config=inp).per_sample, expected, rtol=1e-10)
