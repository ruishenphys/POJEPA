"""B05：用独立小矩阵核对空间 R；不是空间预测效果实验。"""

from dataclasses import replace
import importlib
import importlib.util

import pytest
import torch


def geometry():
    """缺功能时明确失败，避免导入错误掩盖首次 RED 的原因。"""
    assert importlib.util.find_spec("po_jepa.geometry") is not None, "B05 geometry 尚未实现"
    return importlib.import_module("po_jepa.geometry")


def reference_bilinear():
    """独立手算 2×3→3×2：半像素坐标、边缘延拓，x 列最快。"""
    ry = torch.tensor([[1., 0.], [.5, .5], [0., 1.]], dtype=torch.float64)
    rx = torch.tensor([[.75, .25, 0.], [0., .25, .75]], dtype=torch.float64)
    return torch.kron(ry, rx)


def test_row_major_coordinates_and_channelwise_transfer():
    g = geometry()
    source = g.regular_grid(2, 3)
    expected = torch.tensor([[1/6, 1/4], [1/2, 1/4], [5/6, 1/4],
                             [1/6, 3/4], [1/2, 3/4], [5/6, 3/4]], dtype=torch.float64)
    torch.testing.assert_close(source.coords, expected, atol=1e-10, rtol=1e-8)
    assert source.coords.device.type == "cpu" and not source.coords.requires_grad
    assert source.coordinate_source
    transfer = g.build_spatial_transfer(source, target_height=3, target_width=2)
    assert transfer.method == "bilinear" and transfer.align_corners is False
    assert transfer.boundary_mode == "edge"
    r = reference_bilinear()
    torch.testing.assert_close(transfer.target.coords, r @ expected, atol=1e-10, rtol=1e-8)
    # 显式时间、batch、通道各带不同数值，且最后两轴故意不连续。
    values = torch.arange(2*3*4*6, dtype=torch.float64).reshape(2, 3, 4, 6).transpose(-1, -2)
    assert not values.is_contiguous()
    before = values.clone()
    actual = g.apply_spatial_transfer(values, transfer)
    torch.testing.assert_close(actual, torch.einsum("ji,btic->btjc", r, values), atol=1e-10, rtol=1e-8)
    assert actual.shape == (2, 3, 6, 4) and torch.equal(values, before)


@pytest.mark.parametrize("target", [(2, 3), (3, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transfer_preserves_constants_linearity_and_input_gradient(target, dtype):
    atol, rtol = (1e-10, 1e-8) if dtype == torch.float64 else (1e-5, 1e-4)
    g = geometry()
    # 4×6→2×3 为池化；4×6→3×2 的一轴不整除，为双线性。
    source = g.regular_grid(4, 6)
    transfer = g.build_spatial_transfer(source, target_height=target[0], target_width=target[1])
    eye = torch.eye(24, dtype=dtype).unsqueeze(0)
    r = g.apply_spatial_transfer(eye, transfer)[0]
    assert torch.all(r >= 0)
    torch.testing.assert_close(r.sum(-1), torch.ones(r.size(0), dtype=dtype), atol=atol, rtol=rtol)
    x = torch.arange(48, dtype=dtype).reshape(1, 24, 2).requires_grad_()
    y = torch.flip(x.detach(), (1,))
    before = x.detach().clone()
    actual = g.apply_spatial_transfer(x, transfer)
    torch.testing.assert_close(g.apply_spatial_transfer(2*x-3*y, transfer),
                               2*actual-3*g.apply_spatial_transfer(y, transfer), atol=atol, rtol=rtol)
    torch.testing.assert_close(g.apply_spatial_transfer(torch.ones_like(x), transfer),
                               torch.ones_like(actual), atol=atol, rtol=rtol)
    grad = torch.arange(actual.numel(), dtype=dtype).reshape_as(actual)
    (actual*grad).sum().backward()
    torch.testing.assert_close(x.grad, torch.einsum("ji,bjc->bic", r, grad), atol=atol, rtol=rtol)
    assert actual.dtype == dtype and actual.device == x.device and torch.equal(x.detach(), before)
    assert torch.equal(g.apply_spatial_transfer(x, transfer), actual)


def test_bilinear_matches_independent_matrix_values_coordinates_and_transposed_gradient():
    g = geometry()
    transfer = g.build_spatial_transfer(g.regular_grid(2, 3), target_height=3, target_width=2)
    x = torch.arange(12, dtype=torch.float64).reshape(1, 6, 2).requires_grad_()
    r = reference_bilinear()
    actual = g.apply_spatial_transfer(x, transfer)
    torch.testing.assert_close(actual[0], r @ x[0], atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(g.apply_spatial_transfer(torch.eye(6, dtype=torch.float64)[None], transfer)[0],
                               r, atol=1e-10, rtol=1e-8)
    grad = torch.arange(12, dtype=torch.float64).reshape_as(actual)
    (actual*grad).sum().backward()
    torch.testing.assert_close(x.grad[0], r.T @ grad[0], atol=1e-10, rtol=1e-8)


def test_average_pool_matches_hand_blocks_and_identity():
    g = geometry()
    source = g.regular_grid(2, 4)
    transfer = g.build_spatial_transfer(source, target_height=1, target_width=2)
    assert transfer.method == "average_pool"
    x = torch.arange(8, dtype=torch.float64).reshape(1, 8, 1)
    assert torch.equal(g.apply_spatial_transfer(x, transfer), torch.tensor([[[2.5], [4.5]]], dtype=torch.float64))
    assert torch.equal(transfer.target.coords, torch.tensor([[.25, .5], [.75, .5]], dtype=torch.float64))
    identity = g.build_spatial_transfer(source, target_height=2, target_width=4)
    assert identity.method == "average_pool"
    assert torch.equal(g.apply_spatial_transfer(x, identity), x)


def test_upsampling_preserves_transferred_edge_coordinates_instead_of_new_centers():
    g = geometry()
    source = g.regular_grid(1, 2)
    transfer = g.build_spatial_transfer(source, target_height=1, target_width=4)
    expected = torch.tensor([[.25, .5], [.375, .5], [.625, .5], [.75, .5]], dtype=torch.float64)
    assert torch.equal(transfer.target.coords, expected)
    assert not torch.equal(transfer.target.coords, g.regular_grid(1, 4).coords)


@pytest.mark.parametrize("size", [(0, 2), (-1, 2), (True, 2), (2, 1.5)])
def test_invalid_grid_sizes_are_rejected(size):
    with pytest.raises((ValueError, TypeError)):
        geometry().regular_grid(*size)


def test_invalid_coordinates_transfer_metadata_and_values_are_rejected():
    g = geometry()
    source = g.regular_grid(2, 3)
    for coords in (torch.full((6, 2), float("nan"), dtype=torch.float64),
                   source.coords.float(), source.coords.clone().requires_grad_(), source.coords[:5]):
        with pytest.raises((ValueError, TypeError)):
            replace(source, coords=coords)
    transfer = g.build_spatial_transfer(source, target_height=3, target_width=2)
    for change in ({"method": "nearest"}, {"align_corners": True}, {"boundary_mode": "reflect"},
                   {"target": g.regular_grid(3, 2)}):
        with pytest.raises((ValueError, TypeError)):
            replace(transfer, **change)
    for x in (torch.zeros(1, 5, 2), torch.zeros(6, 2), torch.zeros(1, 6, 2, dtype=torch.int64),
              torch.full((1, 6, 2), float("inf")), torch.zeros(0, 6, 2)):
        with pytest.raises((ValueError, TypeError)):
            g.apply_spatial_transfer(x, transfer)
