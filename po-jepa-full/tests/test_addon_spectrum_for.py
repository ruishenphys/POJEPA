"""S08a：GraphSpectrum.spectrum_for 与依次调用三步逐位相同，POJEPA.prepare_query 恰好委托它一次（07 §2.1、§2.2）。"""
import dataclasses

import pytest
import torch

from po_jepa.geometry import GridSpec, regular_grid
from po_jepa.model import QueryContext
from po_jepa.runtime import checks
from po_jepa.spectral import GraphConfig, GraphSpectrum, SpectralResult, modal_analysis

from test_losses import e05_real_case
from test_model import d12_record
from test_v4 import _model as v4_model, _z as v4_z


@pytest.fixture
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def assert_same(a, b, path="value"):
    """数据类逐字段、张量逐位（dtype、shape、device、requires_grad、字节）比较；GridSpec 要求是同一个对象。"""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), path
        assert (a.dtype, a.shape, a.device, a.requires_grad) == (b.dtype, b.shape, b.device, b.requires_grad), path
        assert torch.equal(a, b), path
        return
    if isinstance(a, GridSpec):
        assert a is b, path
        return
    if dataclasses.is_dataclass(a):
        assert type(a) is type(b), path
        for field in dataclasses.fields(a):
            assert_same(getattr(a, field.name), getattr(b, field.name), f"{path}.{field.name}")
        return
    assert a == b, path


def protocol_z(batch, height, width, seed):
    g = torch.Generator().manual_seed(seed)
    e = 1408
    return (torch.randn(1, 1, e, generator=g)*2+torch.randn(batch, height*width, e, generator=g)).float()


def v7_prepare_query(model, raw_z, grid):
    """911bc09 的 POJEPA.prepare_query 原文（参照；输入合法，省掉开头的类型检查）。"""
    model._validate_components()
    x = model.project_in(raw_z)
    graph = model.spectrum.config
    affinity = model.spectrum.affinity(raw_z if graph.affinity_source == "encoder" else x, grid)
    laplacian = model.spectrum.normalized_laplacian(affinity).laplacian
    if graph.fixed_modes and x.shape[1] < graph.max_modes:
        raise ValueError("fixed_modes requires every grid to have at least K tokens")
    modes = graph.max_modes if graph.fixed_modes else min(graph.max_modes, x.shape[1])
    spectrum = model.spectrum.lowest_eigenpairs(laplacian, k=modes, model_device=x.device)
    initial = modal_analysis(x, spectrum.basis)
    context = model.dynamics.prepare_context(x, spectrum.eigenvalues, spectrum.basis, grid.coords)
    return QueryContext(x, grid, spectrum, initial, context)


def protocol_spectrum():
    """协议的图配置：encoder 模式、feature_dim 1408、K=16、fixed_modes。"""
    return GraphSpectrum(GraphConfig(**checks.load_protocol()["model"]["graph"]))


@pytest.mark.parametrize("size", [16, 8, 12])
def test_spectrum_for_matches_three_calls_bitwise(one_thread, size):
    """spectrum_for 与手调 affinity → normalized_laplacian → lowest_eigenpairs 的结果逐位相同。"""
    spectrum = protocol_spectrum()
    grid = regular_grid(size, size)
    z = protocol_z(2, size, size, 1703+size)
    cpu = torch.device("cpu")
    actual = spectrum.spectrum_for(z, grid, k=16, model_device=cpu)
    affinity = spectrum.affinity(z, grid)
    laplacian = spectrum.normalized_laplacian(affinity).laplacian
    expected = spectrum.lowest_eigenpairs(laplacian, k=16, model_device=cpu)
    assert type(actual) is SpectralResult
    assert_same(actual, expected)
    assert actual.basis_cpu.shape == (2, size*size, 16)
    assert actual.diagnostics.cutoff_gap.shape == (2,)


def test_spectrum_for_calls_the_three_steps_once_in_order(monkeypatch):
    """三步各调用一次、顺序不变；affinity 收到同一个 z 和 grid，lowest_eigenpairs 的关键字原样。"""
    spectrum = protocol_spectrum()
    grid = regular_grid(8, 8)
    z = protocol_z(1, 8, 8, 1711)
    calls = []

    def wrap(name):
        original = getattr(spectrum, name)

        def record(*args, **kwargs):
            calls.append((name, args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(spectrum, name, record)

    for name in ("affinity", "normalized_laplacian", "lowest_eigenpairs"):
        wrap(name)
    spectrum.spectrum_for(z, grid, k=16, model_device=torch.device("cpu"))
    assert [name for name, _, _ in calls] == ["affinity", "normalized_laplacian", "lowest_eigenpairs"]
    _, args, kwargs = calls[0]
    assert len(args) == 2 and args[0] is z and args[1] is grid and kwargs == {}
    assert calls[2][2] == {"k": 16, "model_device": torch.device("cpu")}


def test_spectrum_for_forwards_arguments_unchanged(monkeypatch):
    """记录桩替换 lowest_eigenpairs：laplacian、k、model_device（cuda:3，只构造设备对象）原样转交，返回值原样返回。"""
    spectrum = protocol_spectrum()
    grid = regular_grid(8, 8)
    z = protocol_z(2, 8, 8, 1719)
    sentinel, calls = object(), []

    def stub(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(spectrum, "lowest_eigenpairs", stub)
    result = spectrum.spectrum_for(z, grid, k=5, model_device=torch.device("cuda", 3))
    assert result is sentinel
    assert len(calls) == 1
    args, kwargs = calls[0]
    expected = spectrum.normalized_laplacian(spectrum.affinity(z, grid)).laplacian
    assert len(args) == 1 and torch.equal(args[0], expected)
    assert kwargs == {"k": 5, "model_device": torch.device("cuda", 3)}


@pytest.mark.parametrize("case", ["protocol16", "protocol8", "protocol12", "projected"])
def test_prepare_query_matches_v7_reference_and_delegates_once(monkeypatch, one_thread, case):
    """prepare_query 与 911bc09 原文逐位相同，且恰好调用 spectrum_for、lowest_eigenpairs 各一次。"""
    if case == "projected":
        model, raw, *_ = e05_real_case()
        raw = raw.detach()
        grid, modes = regular_grid(4, 6), 20
    else:
        size = int(case.removeprefix("protocol"))
        model = checks.build_full_model(checks.load_protocol(), torch.device("cpu"))
        raw, grid, modes = protocol_z(2, size, size, size), regular_grid(size, size), 16
    expected = v7_prepare_query(model, raw, grid)
    spectra = d12_record(monkeypatch, model.spectrum, "spectrum_for")
    eigenpairs = d12_record(monkeypatch, model.spectrum, "lowest_eigenpairs")
    query = model.prepare_query(raw, grid)
    assert_same(query, expected)
    assert len(spectra) == 1 and len(eigenpairs) == 1
    args, kwargs, _ = spectra[0]
    source = query.projected if case == "projected" else raw
    assert len(args) == 2 and args[0] is source and args[1] is grid
    assert kwargs == {"k": modes, "model_device": raw.device}
    assert eigenpairs[0][1] == {"k": modes, "model_device": raw.device}


def test_fixed_modes_rejected_before_any_spectral_work(monkeypatch):
    """fixed_modes 且 N<K：报 v7 原句，且在建图之前（spectrum_for、affinity 都没被调用）。"""
    model = v4_model(k=16)
    spectra = d12_record(monkeypatch, model.spectrum, "spectrum_for")
    affinities = d12_record(monkeypatch, model.spectrum, "affinity")
    with pytest.raises(ValueError, match="^fixed_modes requires every grid to have at least K tokens$"):
        model.prepare_query(v4_z(h=3), regular_grid(3, 3))
    assert spectra == [] and affinities == []
