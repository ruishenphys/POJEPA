"""S10a：固定 Fourier 谱模块（08 §4）——模态表、特征值、正交归一、池化关系、spectrum_for 合同、初始权重。"""

import dataclasses
import math

import pytest
import torch

from po_jepa import fourier
from po_jepa.fourier import (FourierConfig, FourierSpectrum, build_fixed_fourier, dct_basis, fourier_eigenvalues,
                             mode_order)
from po_jepa.geometry import GridSpec, apply_spatial_transfer, build_spatial_transfer, regular_grid
from po_jepa.runtime import checks, training, v7jobs
from po_jepa.spectral import GraphConfig, GraphSpectrum, SpectralResult

# 08 §4.3 的表（σ_x = 0.125），第 17 个是不用的 (4,0)
TABLE = [((0, 0), 0.), ((0, 1), .037819), ((1, 0), .037819), ((1, 1), .074209), ((0, 2), .142910), ((2, 0), .142910),
         ((1, 2), .175325), ((2, 1), .175325), ((2, 2), .265397), ((0, 3), .293179), ((3, 0), .293179),
         ((1, 3), .319911), ((3, 1), .319911), ((2, 3), .394191), ((3, 2), .394191), ((0, 4), .460359),
         ((4, 0), .460359)]
SX2 = torch.tensor(.125, dtype=torch.float32).to(torch.float64).square()   # 和 _working_buffers 的 sx2 相同：0.015625
CPU = torch.device("cpu")


def protocol_graph():
    return checks.load_protocol()["model"]["graph"]


def grids():
    """仿真里实际出现的三种网格：16×16 规则格点、8×8 平均池化目标、12×12 双线性目标。"""
    source = regular_grid(16, 16)
    out = {16: source}
    for size in (8, 12):
        out[size] = build_spatial_transfer(source, target_height=size, target_width=size).target
    return out


def source_on(grid, batch=2, seed=0, width=1408):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, grid.height*grid.width, width, generator=g)


def test_mode_order_matches_table():
    assert mode_order(17) == tuple(mode for mode, _ in TABLE)
    assert mode_order(1) == ((0, 0),)
    assert mode_order(16)[-1] == (0, 4) and mode_order(17)[-1] == (4, 0)
    assert len(set(mode_order(76))) == 76 and mode_order(76)[:17] == mode_order(17)
    # Koopman 的 K = 76 切在 p²+q² = 85 那组里：靠 max(p,q) 排序取 (6,7)，不取 (2,9)、(9,2)、(7,6)
    assert mode_order(77)[68:] == ((7, 5), (4, 8), (8, 4), (0, 9), (9, 0), (1, 9), (9, 1), (6, 7), (7, 6))


def test_eigenvalues_match_table_and_formula():
    values = fourier_eigenvalues(mode_order(17), SX2)
    assert values.dtype == torch.float64 and values.device.type == "cpu" and values.shape == (17,)
    for (mode, expected), value in zip(TABLE, values.tolist()):
        assert abs(value-expected) < 1e-6, mode
        p, q = mode
        assert value == pytest.approx(1-math.exp(-.125**2*math.pi**2*(p*p+q*q)/4), rel=1e-14, abs=0)
    assert values[15].item() == values[16].item()   # K = 16 切开简并对 (0,4)/(4,0)


@pytest.mark.parametrize("size", [8, 12, 16, 20])
def test_basis_orthonormal(size):
    basis = dct_basis(size, size, mode_order(16))
    assert basis.dtype == torch.float64 and basis.device.type == "cpu" and basis.shape == (size*size, 16)
    assert torch.linalg.matrix_norm(basis.T@basis-torch.eye(16, dtype=torch.float64)).item() < 1e-12


def test_basis_hand_values_fix_axes_and_token_order():
    """非方形 4×6（H=4，W=6）上手算：p 配 x、q 配 y，token n = y·W + x。"""
    basis = dct_basis(4, 6, [(0, 0), (2, 1), (1, 3)])
    assert basis.shape == (24, 3)
    assert basis[:, 0].tolist() == pytest.approx([1/math.sqrt(24)]*24, rel=1e-15)
    x, y = 1, 3
    expected = math.sqrt(2/6)*math.sqrt(2/4)*math.cos(math.pi*2*(x+.5)/6)*math.cos(math.pi*1*(y+.5)/4)
    assert basis[y*6+x, 1].item() == pytest.approx(expected, rel=1e-14)
    x, y = 4, 2
    expected = math.sqrt(2/6)*math.sqrt(2/4)*math.cos(math.pi*1*(x+.5)/6)*math.cos(math.pi*3*(y+.5)/4)
    assert basis[y*6+x, 2].item() == pytest.approx(expected, rel=1e-14)


def test_pooling_relation():
    """08 §4.3：2×2 平均池化把 u_pq^16 映成 ½·cos(πp/32)·cos(πq/32)·u_pq^8（p, q < 8）。"""
    modes = [(p, q) for p in range(8) for q in range(8)]
    transfer = build_spatial_transfer(regular_grid(16, 16), target_height=8, target_width=8)
    pooled = apply_spatial_transfer(dct_basis(16, 16, modes)[None], transfer)[0]
    factor = torch.tensor([.5*math.cos(math.pi*p/32)*math.cos(math.pi*q/32) for p, q in modes], dtype=torch.float64)
    assert (pooled-dct_basis(8, 8, modes)*factor).abs().max().item() < 1e-12


@pytest.mark.parametrize("size", [16, 8, 12])
def test_spectrum_for_matches_graph_spectrum_contract(size):
    grid = grids()[size]
    z = source_on(grid)
    ours = FourierSpectrum(FourierConfig(**protocol_graph())).spectrum_for(z, grid, k=16, model_device=CPU)
    theirs = GraphSpectrum(GraphConfig(**protocol_graph())).spectrum_for(z, grid, k=16, model_device=CPU)
    assert type(ours) is SpectralResult and type(ours.diagnostics) is type(theirs.diagnostics)
    pairs = [(f.name, getattr(ours, f.name), getattr(theirs, f.name)) for f in dataclasses.fields(ours) if f.name != "diagnostics"]
    pairs += [(f.name, getattr(ours.diagnostics, f.name), getattr(theirs.diagnostics, f.name))
              for f in dataclasses.fields(ours.diagnostics)]
    for name, a, b in pairs:
        assert (a.dtype, a.shape, a.device, a.requires_grad) == (b.dtype, b.shape, b.device, b.requires_grad), name
    basis = dct_basis(size, size, mode_order(16))
    assert torch.equal(ours.basis_cpu, basis[None].repeat(2, 1, 1))
    assert torch.equal(ours.eigenvalues_cpu, fourier_eigenvalues(mode_order(16), SX2)[None].repeat(2, 1))
    assert torch.equal(ours.basis, ours.basis_cpu.float()) and torch.equal(ours.eigenvalues, ours.eigenvalues_cpu.float())
    d = ours.diagnostics
    assert torch.equal(d.residual_norms, torch.zeros(2, 16, dtype=torch.float64))
    assert torch.equal(d.cutoff_gap, torch.zeros(2, dtype=torch.float64))
    assert (d.orthogonality_error < 1e-12).all()
    assert torch.equal(d.consecutive_gaps, ours.eigenvalues_cpu[:, 1:]-ours.eigenvalues_cpu[:, :-1])
    assert torch.equal(d.sign_pivot_indices, ours.basis_cpu.abs().argmax(dim=1))


def test_spectrum_for_ignores_source_values_and_full_k_has_no_cutoff():
    spectrum = FourierSpectrum(FourierConfig(**protocol_graph()))
    grid = regular_grid(16, 16)
    a = spectrum.spectrum_for(source_on(grid, seed=1), grid, k=16, model_device=CPU)
    b = spectrum.spectrum_for(torch.zeros(2, 256, 1408), grid, k=16, model_device=CPU)
    assert torch.equal(a.basis_cpu, b.basis_cpu) and torch.equal(a.eigenvalues_cpu, b.eigenvalues_cpu)
    small = regular_grid(2, 2)
    full = spectrum.spectrum_for(source_on(small, batch=1), small, k=4, model_device=CPU)
    assert full.diagnostics.cutoff_gap is None and full.diagnostics.consecutive_gaps.shape == (1, 3)
    assert full.basis_cpu.shape == (1, 4, 4)


def test_spectrum_errors(monkeypatch):
    spectrum = FourierSpectrum(FourierConfig(**protocol_graph()))
    grid = regular_grid(16, 16)
    z = source_on(grid, batch=1)
    cases = [
        (TypeError, "^x must be a tensor and grid must be a GridSpec$", dict(source=z.tolist())),
        (ValueError, "^x must have shape \\[positive B, grid height\\*width, 1408\\]$", dict(source=z[:, :255])),
        (ValueError, "^x must be a finite float32 tensor on a real model device$", dict(source=z.double())),
        (ValueError, "^k must be an integer in \\[1,N\\]$", dict(k=0)),
        (TypeError, "^model_device must be a torch.device$", dict(model_device="cpu")),
        (ValueError, "^first-round model_device must be CPU or available CUDA$", dict(model_device=torch.device("meta"))),
        (ValueError, "^fixed Fourier basis requires regular cell-center coordinates$",
         dict(grid=build_spatial_transfer(grid, target_height=20, target_width=20).target, source=source_on(regular_grid(20, 20), batch=1))),
        (ValueError, "^grid does not resolve the first k\\+1 Fourier modes$",
         dict(grid=regular_grid(4, 4), source=source_on(regular_grid(4, 4), batch=1), k=15)),
    ]
    if not torch.cuda.is_available():
        cases.append((ValueError, "^requested CUDA is unavailable$", dict(model_device=torch.device("cuda", 0))))
    with monkeypatch.context() as patch:      # 照抄 lowest_eigenpairs 的设备号检查：CPU 机器上假装有 1 张卡
        patch.setattr(torch.cuda, "is_available", lambda: True)
        patch.setattr(torch.cuda, "device_count", lambda: 1)
        with pytest.raises(ValueError, match="^requested CUDA index does not exist$"):
            spectrum.spectrum_for(z, grid, k=16, model_device=torch.device("cuda", 1))
    for error, message, change in cases:
        kwargs = dict(source=z, grid=grid, k=16, model_device=CPU) | change
        with pytest.raises(error, match=message):
            spectrum.spectrum_for(kwargs["source"], kwargs["grid"], k=kwargs["k"], model_device=kwargs["model_device"])
    monkeypatch.setattr(fourier, "dct_basis", lambda *a: torch.full((256, 16), float("nan"), dtype=torch.float64))
    with pytest.raises(ValueError, match="^spectral diagnostics must contain finite values$"):
        spectrum.spectrum_for(z, grid, k=16, model_device=CPU)


def test_helper_and_config_errors():
    with pytest.raises(ValueError, match="^FourierConfig.basis must be 'dct2'$"):
        FourierConfig(**protocol_graph(), basis="dct3")
    with pytest.raises(ValueError, match="^q_dim must be the fixed first-round width 16$"):
        FourierConfig(**(protocol_graph() | dict(q_dim=8)))
    with pytest.raises(TypeError, match="^config must be a FourierConfig$"):
        FourierSpectrum(GraphConfig(**protocol_graph()))
    for bad in (0, -1, True, 2.):
        with pytest.raises(ValueError, match="^mode count must be a positive integer$"):
            mode_order(bad)
    for height, width in ((0, 4), (4, True)):
        with pytest.raises(ValueError, match="^grid height/width must be positive integers$"):
            dct_basis(height, width, [(0, 0)])
    for modes in ([], [(0,)], [(0, -1)], [[0, 1]], [(0, 1.)]):
        with pytest.raises(ValueError, match="^modes must be a nonempty sequence of \\(p, q\\) nonnegative integer pairs$"):
            dct_basis(4, 4, modes)
    with pytest.raises(ValueError, match="^grid does not resolve the requested Fourier modes$"):
        dct_basis(4, 6, [(0, 4)])
    with pytest.raises(TypeError, match="^seed must be an int$"):
        build_fixed_fourier(checks.load_protocol()["model"], "1703", CPU)


def test_model_config_snapshot_differs_only_by_basis():
    config = checks.load_protocol()
    full = training._model_config(checks.build_full_model(config, CPU))
    ours = training._model_config(build_fixed_fourier(config["model"], checks.model_seed(), CPU))
    assert {k: v for k, v in ours.items() if k != "graph"} == {k: v for k, v in full.items() if k != "graph"}
    assert ours["graph"] == dict(full["graph"], basis="dct2") and list(ours["graph"])[-1] == "basis"


@pytest.mark.parametrize("seed", [None, "1704"])
def test_initial_weights_equal_full(monkeypatch, seed):
    """08 §1.3：同一种子下 fixed_fourier 的 _model_hash 等于 po_jepa_full 的；构造不动全局随机状态。"""
    if seed is not None:
        monkeypatch.setenv(checks.SEED_ENV, seed)
    config = checks.load_protocol()
    before = torch.get_rng_state().clone()
    full = checks.build_full_model(config, CPU)
    ours = build_fixed_fourier(config["model"], checks.model_seed(), CPU)
    assert torch.equal(before, torch.get_rng_state())
    assert type(ours.spectrum) is FourierSpectrum and type(full.spectrum) is GraphSpectrum
    assert list(ours.state_dict()) == list(full.state_dict())
    assert all(torch.equal(v, full.state_dict()[k]) for k, v in ours.state_dict().items())
    assert v7jobs._model_hash(ours) == v7jobs._model_hash(full)
    if seed is not None:
        monkeypatch.delenv(checks.SEED_ENV)
        assert v7jobs._model_hash(ours) != v7jobs._model_hash(checks.build_full_model(checks.load_protocol(), CPU))


def test_protocol_model_forward_on_three_grids():
    """经 POJEPA.prepare_query 调到 FourierSpectrum.spectrum_for；T = 0 时残差读出逐位恒等。"""
    from test_v4 import _controls
    config = checks.load_protocol()
    model = build_fixed_fourier(config["model"], checks.model_seed(), CPU).eval()
    z = source_on(regular_grid(16, 16), batch=1, seed=3)
    controls = _controls([2.], [[.3, -.2]])
    for size, grid in grids().items():
        raw = z if size == 16 else apply_spatial_transfer(z, build_spatial_transfer(regular_grid(16, 16), target_height=size, target_width=size))
        with torch.no_grad():
            query = model.prepare_query(raw, grid)
            assert torch.equal(query.spectrum.basis_cpu[0], dct_basis(size, size, mode_order(16)))
            out = model(raw, controls, torch.tensor([2.], dtype=torch.float64), grid)
            assert out.shape == raw.shape and torch.isfinite(out).all()
            assert torch.equal(model(raw, controls, torch.zeros(1, dtype=torch.float64), grid), raw)
