"""S08a：ln_metrics 的 RMS_LN 和输入尺度 RMS（10 §3.1）：手算公式、度量性质、与 v7 归一化缺陷一致、报错。"""
import numpy as np
import pytest
import torch

from po_jepa import ln_metrics, losses, metrics
from po_jepa.geometry import build_spatial_transfer, regular_grid

from test_v4 import _controls as v4_controls, _model as v4_model, _z as v4_z

SHAPE_MESSAGE = "a and b must be dense tensors with the same shape [positive B, positive N, positive E]"
DEVICE_MESSAGE = "a and b must both be float32 on the same real device"
RAW_MESSAGE = "raw_z must be a dense float32 [B, positive N0, E] tensor with a's B and E on a's device"


def small_pair(seed=0, b=2, n=3, e=5):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(b, n, e, generator=g)*1.5+.3).float()
    c = (torch.randn(b, n, e, generator=g)*.7-.2).float()      # 和 a 独立生成，量级不同
    z = (torch.randn(b, 7, e, generator=g)*.8-.1).float()
    return a, c, z


def hand_ln(x):
    x = x.astype(np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x-mean)**2).mean(axis=-1, keepdims=True)
    return (x-mean)/np.sqrt(var+1e-5)


def assert_metric_output(actual, a):
    assert actual.dtype == torch.float64 and actual.shape == (2,) and actual.device == a.device
    assert not actual.requires_grad


def test_rms_ln_matches_hand_formula():
    """RMS_LN = sqrt(mean_(N,E) (LN(a)-LN(b))^2)，numpy float64 手算，相对误差 1e-12。"""
    a, b, _ = small_pair()
    expected = np.sqrt(((hand_ln(a.numpy())-hand_ln(b.numpy()))**2).mean(axis=(1, 2)))
    actual = ln_metrics.rms_ln(a, b)
    assert_metric_output(actual, a)
    assert np.allclose(actual.numpy(), expected, rtol=1e-12, atol=0)


def test_rms_input_scale_matches_hand_formula():
    """RMS_in = sqrt(mean((a-b)^2)) / sqrt(mean((z-mean(z))^2))，a、b 先各自转 float64，相对误差 1e-12。"""
    a, b, z = small_pair()
    a64, b64, z64 = (x.numpy().astype(np.float64) for x in (a, b, z))
    numerator = np.sqrt(((a64-b64)**2).mean(axis=(1, 2)))
    denominator = np.sqrt(((z64-z64.mean(axis=(1, 2), keepdims=True))**2).mean(axis=(1, 2)))
    actual = ln_metrics.rms_input_scale(a, b, z)
    assert_metric_output(actual, a)
    assert np.allclose(actual.numpy(), numerator/denominator, rtol=1e-12, atol=0)


def test_metric_properties():
    """f(a, a) = 0、对称、三角不等式；输入带梯度时结果也不带梯度。"""
    a, b, z = small_pair(1)
    c = small_pair(2)[0]
    functions = (ln_metrics.rms_ln, lambda x, y: ln_metrics.rms_input_scale(x, y, z))
    for f in functions:
        assert torch.equal(f(a, a.clone()), torch.zeros(2, dtype=torch.float64))
        assert torch.equal(f(a, b), f(b, a))
        assert torch.all(f(a, c) <= f(a, b)+f(b, c)+1e-12)
    a.requires_grad_()
    for f in functions:
        assert not f(a, b).requires_grad


def test_input_scale_version_agrees_with_v7_normalized_defects():
    """rms_input_scale/(τσ)、/T 和 v7 的 tcd_rms/√s²、scd_rms/√s² 相对差 ≤ 1e-7（v7 先在 float32 里相减，不逐位）。"""
    model = v4_model().eval()
    raw = v4_z()
    controls = v4_controls([4., 3.], [[.1, -.2], [0., .3]])
    worst = 0.
    with torch.no_grad():
        for duration in ([1., 2.], [4., 3.]):
            d = torch.tensor(duration, dtype=torch.float64)
            temporal = losses.temporal_predictions(model, raw, controls, d/2, d/2, regular_grid(8, 8))
            v7 = metrics.temporal_composition_rms_defect(temporal).values/losses.input_scale(raw).sqrt()
            new = ln_metrics.rms_input_scale(temporal.direct, temporal.composed, raw)/(temporal.tau*temporal.sigma)
            worst = max(worst, ((new-v7).abs()/v7).max().item())
            transfer = build_spatial_transfer(regular_grid(8, 8), target_height=4, target_width=4)
            spatial = losses.spatial_predictions(model, raw, controls, d, transfer)
            v7 = metrics.spatial_commutation_rms_defect(spatial).values/losses.input_scale(raw).sqrt()
            new = ln_metrics.rms_input_scale(spatial.restricted_prediction, spatial.coarse_prediction, raw)/d
            worst = max(worst, ((new-v7).abs()/v7).max().item())
    print("rms_input_scale vs v7 normalized defects, max relative difference:", worst)
    assert worst <= 1e-7


@pytest.mark.parametrize("case", ["a_list", "b_list", "2d", "shape", "empty", "float64", "meta", "nan"])
def test_pair_errors(case):
    """a、b 的合同：两个函数报同一句（类型和文字精确相等）。"""
    a, b, z = small_pair()
    if case == "a_list":
        a, kind, message = a.tolist(), TypeError, "a must be a tensor"
    elif case == "b_list":
        b, kind, message = b.tolist(), TypeError, "b must be a tensor"
    elif case == "2d":
        a, b, kind, message = a[0], b[0], ValueError, SHAPE_MESSAGE
    elif case == "shape":
        b, kind, message = b[:, :2], ValueError, SHAPE_MESSAGE
    elif case == "empty":
        a, b, kind, message = a[:, :0], b[:, :0], ValueError, SHAPE_MESSAGE
    elif case == "float64":
        b, kind, message = b.double(), ValueError, DEVICE_MESSAGE
    elif case == "meta":
        a, b, kind, message = a.to("meta"), b.to("meta"), ValueError, DEVICE_MESSAGE
    else:
        b = b.clone()
        b[0, 0, 0] = float("nan")
        kind, message = ValueError, "a and b must contain finite values"
    for call in (lambda: ln_metrics.rms_ln(a, b), lambda: ln_metrics.rms_input_scale(a, b, z)):
        with pytest.raises(kind) as error:
            call()
        assert str(error.value) == message


@pytest.mark.parametrize("case", ["list", "batch", "width", "float64", "inf", "constant"])
def test_scale_source_errors(case):
    """rms_input_scale 的 raw_z 合同；常数输入报 losses.input_scale 自己的错。"""
    a, b, z = small_pair()
    if case == "list":
        z, kind, message = z.tolist(), TypeError, "raw_z must be a tensor"
    elif case == "batch":
        z, kind, message = z[:1], ValueError, RAW_MESSAGE
    elif case == "width":
        z, kind, message = z[..., :4], ValueError, RAW_MESSAGE
    elif case == "float64":
        z, kind, message = z.double(), ValueError, RAW_MESSAGE
    elif case == "inf":
        z = z.clone()
        z[1, 0, 0] = float("inf")
        kind, message = ValueError, "raw_z must contain finite values"
    else:
        z, kind, message = torch.full_like(z, .5), ValueError, "input scale must be finite and strictly positive"
    with pytest.raises(kind) as error:
        ln_metrics.rms_input_scale(a, b, z)
    assert str(error.value) == message
