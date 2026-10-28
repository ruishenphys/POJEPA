"""E07：仅用虚构特征/动作核验评价原语，不是正式数据或科学结论。"""

from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
import importlib
import importlib.util
import inspect
import math

import numpy as np
import pytest
import torch

from po_jepa.geometry import build_spatial_transfer, regular_grid
from po_jepa.losses import (SpatialPredictions, TemporalPredictions, jepa_distance,
    spatial_consistency_loss, spatial_predictions, temporal_consistency_loss,
    temporal_predictions)
from test_losses import configuration, e04_real_case
from test_model import d12_controls


def api():
    """RED在测试体内报告缺少模块，不能把收集失败误作公式失败。"""
    assert importlib.util.find_spec("po_jepa.metrics") is not None, "E07评价原语尚未实现"
    return importlib.import_module("po_jepa.metrics")


def temporal_case(difference=None, tau=None, sigma=None):
    """虚构3/4及5/12直角范数，所有raw字段遵守同形同设备合同。"""
    if difference is None:
        difference = torch.tensor([[[3., 4.]], [[5., 12.]]])
    batch = difference.shape[0]
    tau = torch.arange(1, batch+1, dtype=torch.float64) if tau is None else tau
    sigma = torch.full((batch,), 2., dtype=torch.float64) if sigma is None else sigma
    zero = torch.zeros_like(difference)
    return TemporalPredictions(difference, zero, zero, difference, tau, sigma)


def spatial_case(difference=None, duration=None):
    """目标网格1x1；N_target不是细网格N，保存的difference不被重算。"""
    temporal = temporal_case(difference)
    duration = temporal.tau if duration is None else duration
    return SpatialPredictions(temporal.direct, temporal.composed, temporal.difference,
                              duration, regular_grid(1, temporal.difference.shape[1]))


def test_e07_fields_signatures_and_field_only_holders():
    """A06字段/签名精确；两个冻结结构只存字段，不隐含detach或统计。"""
    m = api()
    assert [f.name for f in fields(m.SampleMetric)] == ["values", "name", "unit"]
    assert [f.name for f in fields(m.MetricSummary)] == ["count", "mean", "quantile_levels", "quantile_values"]
    value = torch.tensor([2.], requires_grad=True)
    holder = m.SampleMetric(value, "fixture", "fictional")
    assert holder.values is value
    summary = m.MetricSummary(7, -1., (.25,), (3.,))
    assert summary.count == 7 and summary.mean == -1.
    for obj, name in ((holder, "name"), (summary, "mean")):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, name, None)
    signatures = {
        "future_error": ["model", "raw_z", "controls", "duration", "grid", "raw_target", "loss_config"],
        "action_intervention_margin": ["model", "raw_z", "executed", "alternative", "duration", "grid", "raw_target", "loss_config"],
        "summarize_metric": ["metric", "quantile_levels"],
    }
    for name in ("temporal_composition_defect", "spatial_commutation_defect",
                 "temporal_composition_rms_defect", "spatial_commutation_rms_defect"):
        signatures[name] = ["predictions"]
    for name, parameters in signatures.items():
        assert list(inspect.signature(getattr(m, name)).parameters) == parameters
    for name, parameter in (("future_error", "loss_config"), ("action_intervention_margin", "loss_config"),
                            ("summarize_metric", "quantile_levels")):
        assert inspect.signature(getattr(m, name)).parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY


def test_e07_hand_reference():
    """独立NumPy double手算Frobenius/RMS/均值/秒制，区分带epsilon训练loss。"""
    m, temporal, spatial = api(), temporal_case(), spatial_case()
    d = np.array([[[3., 4.]], [[5., 12.]]], dtype=np.float64)
    norm = np.sqrt((d*d).sum((1, 2)))
    time_expected, space_expected = norm/np.array([2., 4.]), norm/np.array([1., 2.])
    metric_cases = [
        (m.temporal_composition_defect(temporal), time_expected, "tcd_frobenius", "control_cycle^-2"),
        (m.spatial_commutation_defect(spatial), space_expected, "scd_frobenius", "control_cycle^-1"),
        (m.temporal_composition_rms_defect(temporal), time_expected/np.sqrt(2), "tcd_rms", "control_cycle^-2"),
        (m.spatial_commutation_rms_defect(spatial), space_expected/np.sqrt(2), "scd_rms", "control_cycle^-1"),
    ]
    for metric, expected, name, unit in metric_cases:
        assert metric.name == name and metric.unit == unit and metric.values.shape == (2,)
        assert metric.values.dtype == torch.float64 and metric.values.device.type == "cpu"
        assert not metric.values.requires_grad and metric.values.grad_fn is None
        np.testing.assert_allclose(metric.values.numpy(), expected, atol=1e-10, rtol=1e-8)
    config = configuration(time_epsilon=3., space_epsilon=2.)
    time_loss = temporal_consistency_loss(temporal, config=config)
    space_loss = spatial_consistency_loss(spatial, config=config)
    assert not np.allclose(np.sqrt(time_loss.per_sample.numpy()), time_expected)
    assert not np.allclose(np.sqrt(space_loss.per_sample.numpy()), space_expected)
    assert not math.isclose(np.sqrt(np.mean(time_expected**2)), np.mean(time_expected))
    p = .05  # 虚构秒/控制周期；正式周期来源仍由F01/F06核验。
    np.testing.assert_allclose(time_expected/p**2, [1000., 1300.], atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(space_expected/p, [100., 130.], atol=1e-10, rtol=1e-8)
    print("E07 independent hand reference", {"tcd": time_expected.tolist(), "scd": space_expected.tolist(),
        "tcd_rms": (time_expected/np.sqrt(2)).tolist(), "control_period_seconds_fixture": p})


@pytest.mark.parametrize("value", [2e20, 1e-30, torch.finfo(torch.float32).smallest_normal/2, 0.])
def test_e07_double_before_square_and_legal_zero(value):
    """先转double防float32平方溢出/下溢；正分母下真保存零必须为零。"""
    m = api()
    d = torch.full((2, 3, 2), value, dtype=torch.float32).transpose(1, 2)
    expected = np.sqrt((d.numpy().astype(np.float64)**2).sum((1, 2)))
    temporal = temporal_case(d, torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))
    for function, case, factor in (
        (m.temporal_composition_defect, temporal, 1.),
        (m.temporal_composition_rms_defect, temporal, np.sqrt(6)),
        (m.spatial_commutation_defect, spatial_case(d, torch.ones(2, dtype=torch.float64)), 1.),
        (m.spatial_commutation_rms_defect, spatial_case(d, torch.ones(2, dtype=torch.float64)), np.sqrt(6))):
        np.testing.assert_allclose(function(case).values.numpy(), expected/factor, atol=0, rtol=1e-8)


@pytest.mark.parametrize("field", ["tau", "sigma", "duration"])
@pytest.mark.parametrize("bad", ["zero", "negative", "nan", "inf", "float32", "shape", "grad", "object"])
def test_e07_rejects_invalid_time_contracts(field, bad):
    """无epsilon缺陷只接严格正CPU64[B]；0/0也不可补零。"""
    m = api()
    t = torch.ones(2, dtype=torch.float64)
    if bad in ("zero", "negative", "nan", "inf"):
        t[0] = {"zero": 0., "negative": -1., "nan": float("nan"), "inf": float("inf")}[bad]
    elif bad == "float32": t = t.float()
    elif bad == "shape": t = t[:, None]
    elif bad == "grad": t.requires_grad_()
    else: t = [1., 1.]
    case = spatial_case(duration=t) if field == "duration" else replace(temporal_case(torch.zeros(2, 1, 2)), **{field: t})
    functions = (m.spatial_commutation_defect, m.spatial_commutation_rms_defect) if field == "duration" else (m.temporal_composition_defect, m.temporal_composition_rms_defect)
    for function in functions:
        with pytest.raises((TypeError, ValueError)):
            function(case)


@pytest.mark.parametrize("kind", ["temporal", "spatial"])
@pytest.mark.parametrize("bad", ["object", "raw_dtype", "raw_rank", "raw_shape", "raw_nan", "raw_inf", "raw_inference", "grid"])
def test_e07_branch_contracts(kind, bad):
    """只消费既有分支；全部raw同形设备，空间N匹配合法目标网格。"""
    m = api()
    case = temporal_case() if kind == "temporal" else spatial_case()
    field = "intermediate" if kind == "temporal" else "coarse_prediction"
    raw = getattr(case, field)
    if bad == "object": case = object()
    elif bad == "raw_dtype": case = replace(case, **{field: raw.double()})
    elif bad == "raw_rank": case = replace(case, **{field: raw[..., 0]})
    elif bad == "raw_shape": case = replace(case, **{field: raw.repeat(1, 2, 1)})
    elif bad in ("raw_nan", "raw_inf"):
        raw = raw.clone(); raw[0, 0, 0] = float("nan") if bad == "raw_nan" else float("inf")
        case = replace(case, **{field: raw})
    elif bad == "raw_inference":
        with torch.inference_mode(): raw = raw.clone()
        case = replace(case, **{field: raw})
    elif kind == "spatial": case = replace(case, target_grid=regular_grid(2, 1))
    else: case = replace(case, sigma=torch.ones(3, dtype=torch.float64))
    functions = (m.temporal_composition_defect, m.temporal_composition_rms_defect) if kind == "temporal" else (m.spatial_commutation_defect, m.spatial_commutation_rms_defect)
    for function in functions:
        with pytest.raises((TypeError, ValueError)) as error:
            function(case)
        if bad == "raw_shape":
            assert "expected shape=(2, 1, 2)" in str(error.value)
            assert "actual shape=(2, 2, 2)" in str(error.value)
            assert "dtype=torch.float32" in str(error.value) and "device=cpu" in str(error.value)


@pytest.mark.parametrize("tau,sigma,value", [
    (1e-200, 1e-200, 0.), (1e200, 1e200, 0.),
    (1e-160, 1e-160, 1.), (1e150, 1e150, 1e-30)])
def test_e07_temporal_rejects_product_and_ratio_precision(tau, sigma, value):
    """正乘积的0/Inf、除法Inf、非零差除成伪零分别报告精度边界。"""
    m = api()
    case = temporal_case(torch.full((1, 1, 1), value), torch.tensor([tau], dtype=torch.float64), torch.tensor([sigma], dtype=torch.float64))
    for function in (m.temporal_composition_defect, m.temporal_composition_rms_defect):
        with pytest.raises(ValueError): function(case)


@pytest.mark.parametrize("duration,value", [(1e-320, 1.), (1e300, 1e-30)])
def test_e07_spatial_rejects_ratio_precision(duration, value):
    """SCD除法非有限或非零差变伪零均不能报告严格零缺陷。"""
    m = api()
    case = spatial_case(torch.full((1, 1, 1), value), torch.tensor([duration], dtype=torch.float64))
    for function in (m.spatial_commutation_defect, m.spatial_commutation_rms_defect):
        with pytest.raises(ValueError): function(case)


def test_e07_representable_tiny_time_and_existing_graph_backward():
    """合法小时间不加阈值；no_grad不原地detach输入，原图仍可一次反向。"""
    m = api()
    leaf = torch.tensor([[[1e-30, 2e-30]]], requires_grad=True)
    difference = leaf*2
    temporal = temporal_case(difference, torch.tensor([1e-150], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64))
    spatial = spatial_case(difference, torch.tensor([1e-150], dtype=torch.float64))
    leaf.grad = torch.ones_like(leaf)*3
    original_grad = leaf.grad.clone()
    version, grad_fn = difference._version, difference.grad_fn
    for function, case in ((m.temporal_composition_defect, temporal), (m.temporal_composition_rms_defect, temporal),
                           (m.spatial_commutation_defect, spatial), (m.spatial_commutation_rms_defect, spatial)):
        actual = function(case)
        assert actual.values.item() > 0 and not actual.values.requires_grad
    assert difference._version == version and difference.grad_fn is grad_fn
    assert torch.equal(leaf.grad, original_grad)
    difference.sum().backward()
    assert torch.equal(leaf.grad, original_grad+2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_e07_real_phi_1408_chain_no_grad_and_state(monkeypatch, device):
    """真实完整E1408链与手动Phi+d独立核对，记录调用数并检查状态/标签隔离。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("E07 CUDA unavailable; CPU real chain mandatory")
    m = api()
    model, raw, controls, tau, sigma, grid = e04_real_case(device=device, encoder_dim=1408)
    model.eval()
    duration = tau+sigma
    alternative = replace(controls, source_ids=tuple("other-"+s for s in controls.source_ids),
        source_origin_seconds=controls.source_origin_seconds+10., u_values=-controls.u_values)
    target = raw.flip(-1).detach().clone().requires_grad_()
    with torch.no_grad():
        prediction = model(raw, controls, duration, grid)
        manual = jepa_distance(prediction, target, config=configuration()).per_sample
        manual_alt = jepa_distance(model(raw, alternative, duration, grid), target, config=configuration()).per_sample
        temporal = temporal_predictions(model, raw, controls, tau, sigma, grid)
        transfer = build_spatial_transfer(grid, target_height=1, target_width=2)
        spatial = spatial_predictions(model, raw, controls, duration, transfer)
    calls = []
    hook = model.register_forward_hook(lambda owner, args, result: calls.append((args, torch.is_grad_enabled())))
    model.p_in.weight.grad = torch.full_like(model.p_in.weight, .125)
    raw.grad, target.grad = torch.full_like(raw, .5), torch.full_like(target, .75)
    params = {name: (p.detach().clone(), p._version, p.requires_grad, None if p.grad is None else p.grad.clone()) for name, p in model.named_parameters()}
    buffers = {name: b.clone() for name, b in model.named_buffers()}
    tensor_versions = [(x, x._version) for x in (raw, target, grid.coords, controls.u_values, controls.starts)]
    rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    default_dtype, default_device = torch.get_default_dtype(), torch.get_default_device()
    future = m.future_error(model, raw, controls, duration, grid, target, loss_config=configuration())
    assert len(calls) == 1 and calls[0][0] == (raw, controls, duration, grid) and not calls[0][1]
    delta = m.action_intervention_margin(model, raw, controls, alternative, duration, grid, target, loss_config=configuration())
    assert len(calls) == 3 and all(not enabled for args, enabled in calls)
    defects = [m.temporal_composition_defect(temporal), m.temporal_composition_rms_defect(temporal),
               m.spatial_commutation_defect(spatial), m.spatial_commutation_rms_defect(spatial)]
    assert len(calls) == 3
    hook.remove()
    for actual, expected in ((future.values, manual), (delta.values, manual_alt-manual)):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        assert actual.dtype == torch.float32 and actual.device == raw.device and actual.shape == (2,)
        assert actual.grad_fn is None and not actual.requires_grad
    assert future.name == "future_jepa_distance" and delta.name == "delta_act"
    assert future.unit == delta.unit == "dimensionless"
    for metric, branch, denominator, factor in (
        (defects[0], temporal, tau*sigma, 1.), (defects[1], temporal, tau*sigma, math.sqrt(6*1408)),
        (defects[2], spatial, duration, 1.), (defects[3], spatial, duration, math.sqrt(2*1408))):
        d = branch.difference.detach().cpu().numpy().astype(np.float64)
        expected = np.sqrt((d*d).sum((1, 2)))/denominator.numpy()/factor
        np.testing.assert_allclose(metric.values.cpu().numpy(), expected, atol=1e-10, rtol=1e-8)
        assert metric.values.device == raw.device and metric.values.grad_fn is None
    assert model.training is False and torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng))
    assert torch.get_default_dtype() == default_dtype and torch.get_default_device() == default_device
    assert all(x._version == version for x, version in tensor_versions)
    for name, p in model.named_parameters():
        before, version, requires_grad, grad = params[name]
        assert torch.equal(p, before) and p._version == version and p.requires_grad == requires_grad
        assert p.grad is None if grad is None else torch.equal(p.grad, grad)
    assert all(torch.equal(b, buffers[name]) for name, b in model.named_buffers())
    assert torch.equal(raw.grad, torch.full_like(raw, .5)) and torch.equal(target.grad, torch.full_like(target, .75))
    print("E07 real chain", {"device": str(raw.device), "shape": list(raw.shape), "future": future.values.tolist(),
        "delta": delta.values.tolist(), "production_phi_calls": len(calls), "tcd": defects[0].values.tolist(), "scd": defects[2].values.tolist()})


def test_e07_delta_positive_negative_zero_and_zero_time_diagnostic():
    """真实模型预测各作标签制造可辨正负Delta；零时长仍走Phi读出而非恒等。"""
    m = api()
    model, raw, controls, tau, sigma, grid = e04_real_case()
    model.eval()
    with torch.no_grad():
        # 测试选择：放大真实B线性控制项，让符号远离初始化带来的近零区。
        model.dynamics.b_head.weight.mul_(100.)
    duration = tau+sigma
    alternative = replace(controls, source_ids=("other-0", "other-1"), u_values=-controls.u_values,
                          source_origin_seconds=controls.source_origin_seconds+123.)
    with torch.no_grad():
        executed_prediction = model(raw, controls, duration, grid)
        alternative_prediction = model(raw, alternative, duration, grid)
    positive = m.action_intervention_margin(model, raw, controls, alternative, duration, grid, executed_prediction, loss_config=configuration())
    negative = m.action_intervention_margin(model, raw, controls, alternative, duration, grid, alternative_prediction, loss_config=configuration())
    assert torch.all(positive.values > .001) and torch.all(negative.values < -.001)
    torch.testing.assert_close(positive.values, -negative.values, atol=1e-5, rtol=1e-4)
    same_commands = replace(alternative, u_values=controls.u_values)
    zero = m.action_intervention_margin(model, raw, controls, same_commands, duration, grid, raw, loss_config=configuration())
    assert torch.count_nonzero(zero.values) == 0
    zero_duration = torch.zeros(2, dtype=torch.float64)
    diagnostic = m.future_error(model, raw, controls, zero_duration, grid, raw, loss_config=configuration())
    assert torch.all(diagnostic.values > .001)
    zero_delta = m.action_intervention_margin(model, raw, controls, alternative, zero_duration, grid, raw, loss_config=configuration())
    assert torch.count_nonzero(zero_delta.values) == 0
    print("E07 Delta signs", {"positive": positive.values.tolist(), "negative": negative.values.tolist(), "zero": zero.values.tolist(), "phi_zero_distance": diagnostic.values.tolist()})


@pytest.mark.parametrize("bad", ["same_source", "period", "unit", "scale", "offset", "range", "definition", "batch", "coverage", "gap", "overlap", "precision", "action", "train"])
def test_e07_delta_rejects_invalid_source_alignment_and_controls(bad):
    """拒绝元数据同源和实质时钟/动作错误；宽松B04检查不能绕开D11。"""
    m = api()
    model, raw, controls, tau, sigma, grid = e04_real_case()
    model.eval()
    duration = tau+sigma
    alternative = replace(controls, source_ids=("other-0", "other-1"))
    spec_changes = {"period": {"control_period_seconds": .1}, "unit": {"channel_units": ("other", "unit")},
        "scale": {"scale": (2., 1.)}, "offset": {"offset": (.2, 0.)}, "range": {"lower_bounds": (-99., -100.)},
        "definition": {"definition_source": "different known source"}}
    if bad == "same_source": alternative = replace(alternative, source_ids=(controls.source_ids[0], "other-1"), u_values=-controls.u_values)
    elif bad in spec_changes: alternative = replace(alternative, control_spec=replace(controls.control_spec, **spec_changes[bad]))
    elif bad == "batch": alternative = d12_controls([[(0., 1., (0., 0.))]])
    elif bad in ("coverage", "precision"):
        amount = .125 if bad == "coverage" else math.ulp(.75)
        rows = [[(0., float(duration[0])-amount, (.2, .1))], [(0., float(duration[1])-amount, (.2, .1))]]
        alternative = replace(d12_controls(rows), source_ids=("other-0", "other-1"))
    elif bad in ("gap", "overlap"):
        starts = alternative.starts.clone(); starts[0, 1] += .01 if bad == "gap" else -.01
        alternative = replace(alternative, starts=starts)
    elif bad == "action":
        values = alternative.u_values.detach().clone(); values[0, 0, 0] = 101.
        alternative = replace(alternative, u_values=values)
    elif bad == "train": model.train()
    with pytest.raises((TypeError, ValueError)):
        m.action_intervention_margin(model, raw, controls, alternative, duration, grid, raw, loss_config=configuration())
    assert model.training == (bad == "train")


def test_e07_alternative_relative_alignment_allows_distinct_segments_padding_and_tail():
    """不同来源绝对原点/段数/切换点/尾部合法；padding不是实际控制。"""
    m = api()
    model, raw, controls, tau, sigma, grid = e04_real_case()
    model.eval()
    alternative = replace(d12_controls([[(0., 1., (.2, -.1))],
        [(0., .25, (.8, .1)), (.25, 1.75, (-.1, .2))]]), source_ids=("other-0", "other-1"),
        source_origin_seconds=torch.tensor([100., 200.], dtype=torch.float64))
    actual = m.action_intervention_margin(model, raw, controls, alternative, tau+sigma, grid, raw, loss_config=configuration())
    assert torch.isfinite(actual.values).all()


@pytest.mark.parametrize("kind", ["future", "delta"])
def test_e07_eval_required_and_target_contract_is_not_adapted(kind):
    """调用者显式eval；目标错精度/轴不隐式转换，不将标签用于Phi输入。"""
    m = api()
    model, raw, controls, tau, sigma, grid = e04_real_case()
    alternative = replace(controls, source_ids=("other-0", "other-1"))
    def invoke(target):
        args = (model, raw, controls, tau+sigma, grid, target) if kind == "future" else (model, raw, controls, alternative, tau+sigma, grid, target)
        return getattr(m, "future_error" if kind == "future" else "action_intervention_margin")(*args, loss_config=configuration())
    with pytest.raises(ValueError): invoke(raw)
    assert model.training
    model.eval()
    for target in (raw.double(), raw[:, :-1], torch.full_like(raw, float("nan"))):
        with pytest.raises((TypeError, ValueError)): invoke(target)


@pytest.mark.parametrize("values", [[-3., 1., 7., 1e12], [4.]])
@pytest.mark.parametrize("device,dtype", [("cpu", torch.float32), ("cpu", torch.float64), ("cuda", torch.float64)])
def test_e07_summary_numpy_linear_reference_and_grad(values, device, dtype):
    """分位水平只是fixture；独立NumPy linear，样本数与分位数长度分开。"""
    m = api()
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("E07 CUDA summary unavailable; CPU float32/64 mandatory")
    leaf = torch.tensor(values, dtype=dtype, device=device, requires_grad=True)
    value = leaf*2
    value.retain_grad(); value.grad = torch.ones_like(value)*7
    before, grad_fn, version, grad = value.clone(), value.grad_fn, value._version, value.grad.clone()
    levels = (0., Fraction(1, 4), np.float64(.5), 1.)
    metric = m.SampleMetric(value, "delta_act", "dimensionless")
    actual = m.summarize_metric(metric, quantile_levels=levels)
    expected = leaf.detach().cpu().numpy().astype(np.float64)*2
    np.testing.assert_allclose(actual.mean, np.mean(expected), atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(actual.quantile_values, np.quantile(expected, [0., .25, .5, 1.], method="linear"), atol=1e-10, rtol=1e-8)
    assert actual.count == len(values) and actual.quantile_levels == (0., .25, .5, 1.)
    assert type(actual.mean) is float and all(type(x) is float for x in actual.quantile_levels+actual.quantile_values)
    assert value.grad_fn is grad_fn and value._version == version and torch.equal(value, before) and torch.equal(value.grad, grad)
    value.sum().backward()
    assert torch.equal(leaf.grad, torch.full_like(leaf, 2.))


@pytest.mark.parametrize("levels", [(), [], (True,), (float("nan"),), (float("inf"),), (-.1,), (1.1,),
    (.5, .5), (.8, .2), (".5",), (torch.tensor(.5),), (10**400,),
    (Fraction(1, 2), Fraction(1, 2)+Fraction(1, 2**200))])
def test_e07_summary_rejects_bad_quantiles(levels):
    """拒绝空/排序/类型/范围错误以及转float64后坍缩的两个不同水平。"""
    m = api()
    with pytest.raises((TypeError, ValueError)):
        m.summarize_metric(m.SampleMetric(torch.tensor([1., 2.]), "fixture", "unit"), quantile_levels=levels)


@pytest.mark.parametrize("level", [Fraction(-1, 10**400), Fraction(10**400+1, 10**400)],
                         ids=["negative-rounded-to-zero", "above-one-rounded-to-one"])
def test_e07_summary_rejects_original_domain_before_float_rounding(level):
    """精确Real原值须在[0,1]；不能用转换到-0.0/1.0的舍入掩盖越界。"""
    m = api()
    metric = m.SampleMetric(torch.tensor([1., 2.]), "fixture", "dimensionless")
    assert level < 0 or level > 1
    with pytest.raises(ValueError):
        m.summarize_metric(metric, quantile_levels=(level,))


@pytest.mark.parametrize("bad", ["object", "empty", "rank", "int", "half", "nan", "inf", "name", "unit", "overflow"])
def test_e07_summary_rejects_invalid_values(bad):
    """不丢NaN/Inf、不挑样本或补0；统计结果自身也必须有限。"""
    m = api()
    values = {"empty": torch.empty(0), "rank": torch.ones(2, 1), "int": torch.ones(2, dtype=torch.int64),
        "half": torch.ones(2, dtype=torch.float16), "nan": torch.tensor([float("nan")]),
        "inf": torch.tensor([float("inf")]), "overflow": torch.tensor([1e308, 1e308], dtype=torch.float64)}.get(bad, torch.ones(2))
    metric = m.SampleMetric(values, " " if bad == "name" else "fixture", "" if bad == "unit" else "unit")
    if bad == "object": metric = object()
    with pytest.raises((TypeError, ValueError)):
        m.summarize_metric(metric, quantile_levels=(0., 1.))


def test_e07_window_distribution_differs_from_batch_and_trajectory_macro():
    """虚构不等长轨迹保留原值；总体窗口/简单batch平均/轨迹宏平均不可混用。"""
    m = api()
    values = torch.tensor([0., 2., 4., 20.])
    ids = ("fictional-A", "fictional-A", "fictional-A", "fictional-B")
    def report(v): return m.summarize_metric(m.SampleMetric(v, "fixture", "unit"), quantile_levels=(0., .5, 1.))
    window = report(values)
    batches = (report(values[:3]), report(values[3:]))
    trajectories = [report(values[torch.tensor([x == key for x in ids])]) for key in dict.fromkeys(ids)]
    macro = np.mean([item.mean for item in trajectories])
    assert window.count == 4 and window.mean == 6.5
    assert macro == 11. and macro != window.mean
    assert np.mean([batch.mean for batch in batches]) != window.mean
    assert report(torch.tensor([item.mean for item in trajectories])).quantile_values != window.quantile_values
    assert torch.equal(values, torch.tensor([0., 2., 4., 20.]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="E07 mixed CPU/CUDA boundary unavailable")
def test_e07_mixed_device_and_time_boundary():
    """相同shape不能掩盖实际设备错配；时间合同仍只接受CPU64。"""
    m = api()
    for function, case in (
        (m.temporal_composition_defect, replace(temporal_case(), intermediate=temporal_case().intermediate.cuda())),
        (m.spatial_commutation_defect, replace(spatial_case(), coarse_prediction=spatial_case().coarse_prediction.cuda())),
        (m.temporal_composition_defect, replace(temporal_case(), tau=torch.ones(2, dtype=torch.float64, device="cuda"))),
        (m.spatial_commutation_defect, spatial_case(duration=torch.ones(2, dtype=torch.float64, device="cuda")))):
        with pytest.raises(ValueError): function(case)
    model, raw, controls, tau, sigma, grid = e04_real_case(device="cuda")
    model.eval()
    alternative = replace(controls, source_ids=("other-0", "other-1"), u_values=controls.u_values.cpu())
    with pytest.raises(ValueError):
        m.action_intervention_margin(model, raw, controls, alternative, tau+sigma, grid, raw, loss_config=configuration())
