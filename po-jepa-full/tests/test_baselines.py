"""F05：虚构 CPU 数据检验普通 Transformer/identity，不形成效果排名。"""
from dataclasses import replace
import importlib
import inspect

import pytest
import torch


@pytest.fixture
def baseline():
    """缺实现必须以明确的接口 RED 暴露。"""
    import importlib.util
    assert importlib.util.find_spec("po_jepa.baselines"), "F05 baseline 尚未实现"
    return importlib.import_module("po_jepa.baselines")


@pytest.fixture
def case():
    """复用已验 E06 虚构输入，包含非相同持续时间和真实动作段 mask。"""
    from test_training import e06_case
    return e06_case()[1]


def test_transformer_all_parameters_receive_finite_gradients(baseline, case):
    """所有参与预测的参数均须有有限梯度，不能用从不前向的占位层凑数。"""
    model = baseline.TokenTransformer(baseline.TransformerSpec(encoder_dim=3))
    raw = case.raw_z.clone().requires_grad_()
    result = model(raw, case.controls, case.duration, case.transfer.source)
    assert result.shape == raw.shape and result.dtype == torch.float32
    result.square().mean().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert case.controls.u_values.grad is not None and torch.isfinite(case.controls.u_values.grad).all()
    assert "target" not in inspect.signature(model.forward).parameters


def test_masked_nonfinite_padding_is_invisible(baseline, case):
    """无效槽的NaN/Inf在投影之前排除；原始控制保持原样。"""
    model = baseline.TokenTransformer(baseline.TransformerSpec(encoder_dim=3)).eval()
    controls = case.controls
    mask = controls.segment_valid
    padded = replace(controls,
        starts=torch.cat([controls.starts, torch.full((2, 2), float("nan"), dtype=torch.float64)], 1),
        ends=torch.cat([controls.ends, torch.full((2, 2), float("inf"), dtype=torch.float64)], 1),
        u_values=torch.cat([controls.u_values, torch.full((2, 2, 2), float("nan"))], 1),
        segment_valid=torch.cat([mask, torch.zeros(2, 2, dtype=torch.bool)], 1))
    with torch.no_grad():
        expected = model(case.raw_z, controls, case.duration, case.transfer.source)
        actual = model(case.raw_z, padded, case.duration, case.transfer.source)
    assert torch.equal(actual, expected) and torch.isnan(padded.u_values[:, -2:]).all()


def test_batch_slices_and_controls_time_information(baseline, case):
    """无batch耦合；逐段值/局部时间确实进入token，不压成平均动作。"""
    model = baseline.TokenTransformer(baseline.TransformerSpec(encoder_dim=3)).eval()
    tokens, mask = model.action_tokens(case.controls, case.duration)
    assert tokens.shape[-1] == 4 and mask.dtype == torch.bool
    from po_jepa.actions import clip_action_segments
    prefix = clip_action_segments(case.controls, start=torch.zeros_like(case.duration), end=case.duration)
    assert torch.equal(tokens[mask][:, :2], prefix.u_values[prefix.segment_valid])
    assert torch.equal(tokens[mask][:, 2], prefix.starts[prefix.segment_valid].float())
    assert torch.equal(tokens[mask][:, 3], prefix.ends[prefix.segment_valid].float())
    with torch.no_grad():
        whole = model(case.raw_z, case.controls, case.duration, case.transfer.source)
        for index in range(2):
            fields = dict(starts=case.controls.starts[index:index+1], ends=case.controls.ends[index:index+1],
                u_values=case.controls.u_values[index:index+1], n_segments=case.controls.n_segments[index:index+1],
                segment_valid=case.controls.segment_valid[index:index+1], source_origin_seconds=case.controls.source_origin_seconds[index:index+1],
                duration=case.controls.duration[index:index+1], source_ids=case.controls.source_ids[index:index+1])
            single = model(case.raw_z[index:index+1], replace(case.controls, **fields),
                case.duration[index:index+1], case.transfer.source)
            torch.testing.assert_close(single, whole[index:index+1], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("size", [8, 12, 16])
def test_actual_grid_coordinates_and_variable_resolution(baseline, case, size):
    """用实际coords支持既定三种网格；没有固定16×16位置查表。"""
    from po_jepa.geometry import regular_grid
    grid = regular_grid(size, size)
    model = baseline.TokenTransformer(baseline.TransformerSpec(encoder_dim=3)).eval()
    raw = torch.linspace(-1, 1, 2*size*size*3).reshape(2, size*size, 3)
    with torch.no_grad():
        out = model(raw, case.controls, case.duration, grid)
        changed = model(raw, case.controls, case.duration, replace(grid, coords=1-grid.coords))
    assert out.shape == raw.shape and torch.isfinite(out).all() and not torch.equal(out, changed)


@pytest.mark.parametrize("bad", ["dtype", "shape", "nan", "time_dtype", "control_nan"])
def test_invalid_inputs_are_rejected(baseline, case, bad):
    """有效输入中的非法值必须报错；padding非有限值另由mask测试覆盖。"""
    model = baseline.TokenTransformer(baseline.TransformerSpec(encoder_dim=3))
    raw, controls, duration = case.raw_z, case.controls, case.duration
    if bad == "dtype": raw = raw.double()
    elif bad == "shape": raw = raw[:, :-1]
    elif bad == "nan": raw = raw.clone(); raw[0, 0, 0] = float("nan")
    elif bad == "time_dtype": duration = duration.float()
    else:
        values = controls.u_values.detach().clone(); values[0, 0, 0] = float("nan")
        controls = replace(controls, u_values=values)
    with pytest.raises((ValueError, TypeError)):
        model(raw, controls, duration, case.transfer.source)


def test_identity_returns_current_raw_and_has_no_parameters(baseline, case):
    """identity只是当前特征，零一致性不代表未来准确；不分配优化参数。"""
    model = baseline.IdentityPredictor(3)
    assert model(case.raw_z, case.controls, case.duration, case.transfer.source) is case.raw_z
    assert list(model.parameters()) == []


def test_positive_control_interval_cannot_collapse_in_token_encoding(baseline,case):
    """CPU64合法正段若转FP32后端点相同应报错，不静默丢掉整段控制时间。"""
    model=baseline.TokenTransformer(baseline.TransformerSpec(encoder_dim=3))
    starts,ends=case.controls.starts.clone(),case.controls.ends.clone()
    ends[0,0]=1e-50;starts[0,1]=1e-50
    with pytest.raises(ValueError,match="representable"):
        model.action_tokens(replace(case.controls,starts=starts,ends=ends),case.duration)
