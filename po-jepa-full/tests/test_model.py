"""C01：用虚构特征核对投影数学；不提供假的谱、动力学或积分结果。"""

import importlib
import importlib.util
import inspect
import math
import random
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace

import numpy as np
import pytest
import torch
from torch import nn


def api():
    """首次RED必须明确指向尚未编写的C01模块。"""
    assert importlib.util.find_spec("po_jepa.model") is not None, "C01 model 尚未实现"
    return importlib.import_module("po_jepa.model")


class PassiveReference(nn.Module):
    """仅检查引用和副作用的无行为哨兵，绝不产生物理计算结果。"""

    def __init__(self):
        super().__init__()
        self.existing = nn.Parameter(torch.tensor([7.]), requires_grad=False)
        self.register_buffer("existing_buffer", torch.tensor([11.]))
        self.eval()

    def forward(self, *args, **kwargs):
        """调用即失败，防止无行为哨兵被用作假的物理计算。"""
        raise AssertionError("C01 must not call an injected reference sentinel")


def make_model(encoder_dim=3):
    """真实两Linear配无行为引用；小E只用于明确可手算的虚构单测。"""
    m = api()
    return m.POJEPA(m.ProjectionConfig(encoder_dim, 64), PassiveReference(),
                    PassiveReference(), object())


def hand_model(device="cpu"):
    """仅在测试覆盖权重：原稿的生产初始化仍为Xavier uniform和零bias。"""
    model = make_model().to(device)
    with torch.no_grad():
        for layer in (model.p_in, model.p_out):
            layer.weight.zero_()
            layer.bias.zero_()
        model.p_in.weight[0, :2] = torch.tensor([2., -1.], device=device)
        model.p_in.weight[1, 2] = 1.
        model.p_in.bias[:2] = torch.tensor([.5, -1.], device=device)
        model.p_out.weight[:, :2] = torch.tensor([[1., 2.], [-1., 0.], [0., 1.]], device=device)
        model.p_out.bias[:] = torch.tensor([1., 0., -1.], device=device)
    return model


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_hand_output_and_full_parameter_gradients(device):
    """冻结z不冻结投影；检查整块梯度，包括应为零的元素。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available; CPU acceptance remains mandatory")
    model = hand_model(device)
    z = torch.tensor([[[1., 2., 3.]]], device=device)
    x = model.project_in(z)
    expected_x = torch.zeros(1, 1, 64, device=device)
    expected_x[..., :2] = torch.tensor([.5, 2.], device=device)
    assert torch.equal(x, expected_x)
    out = model.project_out(x)
    assert torch.equal(out, torch.tensor([[[5.5, -.5, 1.]]], device=device))
    error = (out - z).double()
    assert error.abs().max().item() == 4.5
    assert error.square().sum().item() == 30.5
    out.sum().backward()
    assert z.grad is None
    wi = torch.zeros_like(model.p_in.weight)
    wi[1] = torch.tensor([3., 6., 9.], device=device)
    bi = torch.zeros_like(model.p_in.bias)
    bi[1] = 3.
    wo = torch.zeros_like(model.p_out.weight)
    wo[:, :2] = torch.tensor([.5, 2.], device=device)
    for actual, expected in ((model.p_in.weight.grad, wi), (model.p_in.bias.grad, bi),
                             (model.p_out.weight.grad, wo),
                             (model.p_out.bias.grad, torch.ones(3, device=device))):
        assert torch.equal(actual, expected)


def test_axes_input_synthesized_and_feedback_gradients():
    """解析式独立于Linear；batch/token不混合，预测反馈不额外detach。"""
    model = hand_model()
    z = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3).requires_grad_()
    before = z.detach().clone()
    x = model.project_in(z)
    expected = torch.zeros(2, 3, 64)
    expected[..., 0] = 2*z[..., 0] - z[..., 1] + .5
    expected[..., 1] = z[..., 2] - 1
    assert torch.equal(x, expected)
    out = model.project_out(x)
    expected_out = torch.stack((expected[..., 0] + 2*expected[..., 1] + 1,
                                -expected[..., 0], expected[..., 1] - 1), -1)
    assert torch.equal(out, expected_out)
    out.sum().backward()
    assert torch.equal(z.grad, torch.tensor([0., 0., 3.]).expand_as(z))
    assert torch.equal(z, before)
    synthesized = torch.zeros(2, 3, 64, requires_grad=True)
    model.project_out(synthesized).sum().backward()
    expected_grad = torch.zeros_like(synthesized)
    expected_grad[..., 1] = 3.
    assert torch.equal(synthesized.grad, expected_grad)
    z2 = torch.tensor([[[1., 2., 3.]]], requires_grad=True)
    first = model.project_out(model.project_in(z2))
    first.retain_grad()
    model.project_in(first).sum().backward()
    assert torch.equal(first.grad, torch.tensor([[[2., -1., 1.]]]))
    assert torch.equal(z2.grad, torch.tensor([[[6., -3., 5.]]]))


def test_general_float32_outputs_and_gradients_against_float64_sum_reference():
    """参考仅用float64逐项求和和矩阵导数，不调用另一层Linear。"""
    torch.manual_seed(1703)
    model = make_model(5)
    z = torch.linspace(-.7, .9, 30).reshape(2, 3, 5).requires_grad_()
    wi, bi = model.p_in.weight.detach().double(), model.p_in.bias.detach().double()
    wo, bo = model.p_out.weight.detach().double(), model.p_out.bias.detach().double()
    xr = (z.detach().double().unsqueeze(-2) * wi).sum(-1) + bi
    yr = (xr.unsqueeze(-2) * wo).sum(-1) + bo
    # 独立float64累加与矩阵乘法应在已锁定的双精度容差内一致。
    torch.testing.assert_close(xr, z.detach().double() @ wi.T + bi, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(yr, xr @ wo.T + bo, atol=1e-10, rtol=1e-8)
    out = model.project_out(model.project_in(z))
    torch.testing.assert_close(out.double(), yr, atol=1e-5, rtol=1e-4)
    coeff = torch.linspace(-.3, .8, out.numel()).reshape_as(out)
    (out*coeff).sum().backward()
    cr = coeff.double()
    dx = (cr.unsqueeze(-1) * wo).sum(-2)
    dz = (dx.unsqueeze(-1) * wi).sum(-2)
    expected = (dz, (dx.unsqueeze(-1)*z.detach().double().unsqueeze(-2)).sum((0, 1)),
                dx.sum((0, 1)), (cr.unsqueeze(-1)*xr.unsqueeze(-2)).sum((0, 1)), cr.sum((0, 1)))
    actual = (z.grad, model.p_in.weight.grad, model.p_in.bias.grad,
              model.p_out.weight.grad, model.p_out.bias.grad)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.double(), e, atol=1e-5, rtol=1e-4)


def test_parameter_reuse_no_input_rng_or_mode_side_effects():
    """改变token数仍复用同组参数，投影调用无RNG和模式副作用。"""
    model = make_model()
    params = {n: (id(p), p.detach().clone()) for n, p in model.named_parameters()}
    modes = {n: m.training for n, m in model.named_modules()}
    rng = torch.get_rng_state().clone()
    for n in (1, 7, 3):
        z = torch.ones(2, n, 3)
        before = z.clone()
        assert model.project_out(model.project_in(z)).shape == (2, n, 3)
        assert torch.equal(z, before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert modes == {n: m.training for n, m in model.named_modules()}
    for name, p in model.named_parameters():
        assert id(p) == params[name][0] and torch.equal(p, params[name][1])


def test_init_config_references_and_only_four_new_parameters(monkeypatch):
    """注入参数、buffer和eval状态原样保存；不将其计为新增投影参数。"""
    m = api()
    spectrum, dynamics, integrator = PassiveReference(), PassiveReference(), object()
    refs = [(x, x.existing, x.existing.clone(), x.existing_buffer,
             x.existing_buffer.clone(), x.training) for x in (spectrum, dynamics)]
    calls = []
    initialize = nn.init.xavier_uniform_

    def record(tensor, gain=1., **kwargs):
        """记录实际Xavier调用，仍执行真实初始化，不伪造权重。"""
        calls.append((tuple(tensor.shape), gain))
        return initialize(tensor, gain=gain, **kwargs)

    monkeypatch.setattr(nn.init, "xavier_uniform_", record)
    config = m.ProjectionConfig(1408, 64)
    with pytest.raises(FrozenInstanceError):
        config.dynamic_dim = 128
    torch.manual_seed(1703)
    rng = torch.get_rng_state().clone()
    py_rng, np_rng = random.getstate(), np.random.get_state()
    old_dtype, old_device = torch.get_default_dtype(), torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float64)
        torch.set_default_device("meta")
        model = m.POJEPA(config, spectrum, dynamics, integrator)
        assert torch.get_default_dtype() == torch.float64
        assert torch.get_default_device() == torch.device("meta")
    finally:
        torch.set_default_dtype(old_dtype)
        torch.set_default_device(old_device)
    assert calls == [((64, 1408), 1.), ((1408, 64), 1.)]
    assert not torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == py_rng
    current_np = np.random.get_state()
    assert current_np[0] == np_rng[0] and np.array_equal(current_np[1], np_rng[1])
    assert current_np[2:] == np_rng[2:]
    assert model.encoder_dim == 1408 and model.dynamic_dim == 64
    assert model.spectrum is spectrum and model.dynamics is dynamics and model.integrator is integrator
    for obj, param, value, buffer, buffer_value, mode in refs:
        assert obj.existing is param and torch.equal(param, value) and not param.requires_grad
        assert obj.existing_buffer is buffer and torch.equal(buffer, buffer_value)
        assert obj.training == mode
    new = {n: p for n, p in model.named_parameters() if n.startswith(("p_in.", "p_out."))}
    assert list(new) == ["p_in.weight", "p_in.bias", "p_out.weight", "p_out.bias"]
    assert sum(p.numel() for p in new.values()) == 181696
    assert set(dict(model.named_parameters())) == set(new) | {"spectrum.existing", "dynamics.existing"}
    assert set(dict(model.named_modules())) == {"", "p_in", "p_out", "spectrum", "dynamics"}
    for layer in (model.p_in, model.p_out):
        assert type(layer) is nn.Linear and layer.weight.dtype == torch.float32
        assert layer.weight.device.type == "cpu" and layer.bias.device.type == "cpu"
        assert layer.bias.dtype == torch.float32 and torch.count_nonzero(layer.bias) == 0
        bound = math.sqrt(6/(layer.in_features + layer.out_features))
        assert layer.weight.abs().max().item() <= bound
    with pytest.raises(TypeError):
        model(torch.ones(1, 1, 1408))


def test_seed_is_callers_choice_not_reset_in_constructor(monkeypatch):
    """相同调用者种子可复现；连续构造不同，禁止内部重新设种子。"""
    torch.manual_seed(1703)
    a = make_model()
    b = make_model()
    assert not torch.equal(a.p_in.weight, b.p_in.weight)
    torch.manual_seed(1703)

    def forbidden(*args, **kwargs):
        """若构造器试图重设全局torch种子，立即暴露副作用。"""
        raise AssertionError("constructor must not reset a caller's seed")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    c = make_model()
    for key, val in a.state_dict().items():
        assert torch.equal(val, c.state_dict()[key])


def test_constructor_requires_config_and_all_dependency_references():
    """拒绝缺参和非约定配置，不用默认假对象掩盖尚未完成的依赖组装。"""
    m = api()
    with pytest.raises(TypeError, match="ProjectionConfig"):
        m.POJEPA(None, object(), object(), object())
    with pytest.raises(TypeError):
        m.POJEPA(m.ProjectionConfig(3, 64))


@pytest.mark.parametrize("encoder,dynamic", [(0, 64), (-1, 64), (True, 64), (3., 64),
                                             (3, 2), (3, 128), (3, True), (3, 64.)])
def test_bad_config(encoder, dynamic):
    """明确拒绝非法宽度，不把bool当int或放行未经本轮选择的128。"""
    with pytest.raises((TypeError, ValueError)):
        m = api()
        m.POJEPA(m.ProjectionConfig(encoder, dynamic), object(), object(), object())


@pytest.mark.parametrize("method,width", [("project_in", 3), ("project_out", 64)])
@pytest.mark.parametrize("bad", ["object", "rank", "width", "batch", "tokens", "dtype", "device", "nan", "inf", "overflow"])
def test_bad_projection_inputs_and_nonfinite_output(method, width, bad):
    """覆盖两个入口的边界错误与有限输入溢出，不以夹值掩盖异常。"""
    model = make_model()
    x = torch.ones(1, 2, width)
    if bad == "object":
        x = [1., 2., 3.]
    elif bad == "rank":
        x = x.unsqueeze(0)
    elif bad == "width":
        x = x[..., :-1]
    elif bad in ("batch", "tokens"):
        x = torch.empty(0, 2, width) if bad == "batch" else torch.empty(1, 0, width)
    elif bad == "dtype":
        x = x.double()
    elif bad == "device":
        x = x.to("meta")
    elif bad in ("nan", "inf"):
        x[0, 0, 0] = float(bad)
    else:
        with torch.no_grad():
            getattr(model, "p_in" if method == "project_in" else "p_out").weight.fill_(torch.finfo(torch.float32).max)
        x.fill_(2.)
    with pytest.raises(TypeError if bad == "object" else ValueError):
        getattr(model, method)(x)


def d12_model(*, device="cpu", max_modes=4, fixed_rates=False):
    """D12虚构小E真实链；固定率仅为可解析诊断，仍执行全部生产网络。"""
    from po_jepa.dynamics import DynamicsConfig, IntegratorConfig, ModalDynamics
    from po_jepa.spectral import GraphConfig, GraphSpectrum

    torch.manual_seed(1712)
    spectrum = GraphSpectrum(GraphConfig(16, 1703, .25, 1., max_modes, 1e-10, 1e-8))
    dynamics = ModalDynamics(DynamicsConfig(2, 64, 64, 32, 16, 32, 32, 64, .1, 1e-3))
    model = api().POJEPA(api().ProjectionConfig(3, 64), spectrum, dynamics,
        IntegratorConfig("exponential_euler", .25, 1e-12, 1e-4, 6)).to(device)
    if fixed_rates:
        with torch.no_grad():
            for parameter in dynamics.parameters():
                parameter.zero_()
            dynamics.gamma_head.bias.fill_(.2+math.log(-math.expm1(-.2)))
    return model


def d12_controls(rows, *, device="cpu"):
    """保留原始端点的虚构两通道命令；padding不代表动作，来源只用于诊断。"""
    from po_jepa.actions import ActionSegments, ControlSpec

    batch, width = len(rows), max(1, max(len(row) for row in rows))
    starts = torch.full((batch, width), float("nan"), dtype=torch.float64)
    ends = torch.full_like(starts, float("nan"))
    values = torch.full((batch, width, 2), float("nan"), dtype=torch.float32, device=device)
    for b, row in enumerate(rows):
        for j, (start, end, command) in enumerate(row):
            starts[b, j], ends[b, j] = start, end
            values[b, j] = torch.tensor(command, dtype=torch.float32, device=device)
    counts = torch.tensor([len(row) for row in rows], dtype=torch.int64)
    spec = ControlSpec(2, .05, ("fictional_u", "fictional_v"), ("unit", "unit"),
        (-100., -100.), (100., 100.), (1., 1.), (0., 0.), "D12 fictional known commands")
    return ActionSegments(starts, ends, values, counts, torch.arange(width)[None] < counts[:, None],
        torch.arange(batch, dtype=torch.float64)+2.,
        torch.tensor([row[-1][1] if row else 0. for row in rows], dtype=torch.float64),
        spec, tuple(f"d12-fictional-{b}" for b in range(batch)))


def test_d12_b04_decimal_split_is_exactly_covered_by_real_d11():
    """非二进制小数经真实B04切分仍须覆盖原tau/sigma；不放宽D11严格时钟。"""
    from po_jepa.actions import split_action_segments
    from po_jepa.geometry import regular_grid
    from po_jepa.spectral import modal_analysis

    model = d12_model()
    z = torch.linspace(-.7, .9, 12).reshape(1, 4, 3)
    grid = regular_grid(2, 2)
    x = model.project_in(z)
    w = model.spectrum.affinity(x, grid)
    laplacian = model.spectrum.normalized_laplacian(w).laplacian
    spectrum = model.spectrum.lowest_eigenpairs(laplacian, k=4, model_device=x.device)
    c0 = modal_analysis(x, spectrum.basis)
    context = model.dynamics.prepare_context(x, spectrum.eigenvalues, spectrum.basis, grid.coords)
    tau, sigma = torch.tensor([.17], dtype=torch.float64), torch.tensor([.29], dtype=torch.float64)
    controls = d12_controls([[(0., .17, (.2, -.4)), (.17, 1., (-.7, .3))]])
    pair = split_action_segments(controls, tau=tau, sigma=sigma)
    print("D12 decimal split", {"tau": tau.tolist(), "sigma": sigma.tolist(),
        "first_duration": pair.first.duration.tolist(), "second_duration": pair.second.duration.tolist()})
    first = model.dynamics.integrate_segments(c0, pair.first, context,
        duration=tau, config=model.integrator)
    second = model.dynamics.integrate_segments(first.final, pair.second, context,
        duration=sigma, config=model.integrator)
    assert torch.isfinite(second.final).all()
    assert torch.equal(pair.first.duration, tau) and torch.equal(pair.second.duration, sigma)


def d12_record(monkeypatch, owner, name):
    """只包裹真实方法记录输入/输出，绝不返回替代计算或修改生产接口。"""
    original, calls = getattr(owner, name), []

    def record(*args, **kwargs):
        """真实计算恰运行一次，诊断保留同次结果便于核对来源和调用数。"""
        result = original(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(owner, name, record)
    return calls


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_d12_real_shared_kernel_contract_and_coordinate_boundary(monkeypatch, device):
    """真实Z→W/L/谱→完整N/D11→raw读出，核对A06字段与唯一坐标转换边界。"""
    from po_jepa.dynamics import DynamicsContext, IntegrationResult
    from po_jepa.geometry import regular_grid
    from po_jepa.spectral import SpectralResult

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available; CPU acceptance remains mandatory")
    model = d12_model(device=device, max_modes=3)
    assert hasattr(model, "_predict_raw_with_trace"), "D12 shared real kernel 尚未实现"
    z = torch.linspace(-.7, .9, 24, device=device).reshape(2, 4, 3).requires_grad_()
    grid = regular_grid(2, 2)
    controls = d12_controls([[(0., .2, (.2, -.4)), (.2, .6, (-.7, .3))],
                            [(0., .6, (.8, .1))]], device=device)
    duration = torch.tensor([.4, .3], dtype=torch.float64)
    spectra = d12_record(monkeypatch, model.spectrum, "lowest_eigenpairs")
    contexts = d12_record(monkeypatch, model.dynamics, "prepare_context")
    steps = d12_record(monkeypatch, model.dynamics, "exponential_euler_step")
    trace = model._predict_raw_with_trace(z, controls, duration, grid, collect_trace=True)
    assert isinstance(trace, api().PredictionTrace)
    assert tuple(f.name for f in fields(trace)) == ("raw_prediction", "query", "integration", "synthesized")
    query = trace.query
    assert isinstance(query, api().QueryContext)
    assert tuple(f.name for f in fields(query)) == ("projected", "grid", "spectrum", "initial_coefficients", "dynamics")
    assert isinstance(query.spectrum, SpectralResult) and isinstance(query.dynamics, DynamicsContext)
    assert isinstance(trace.integration, IntegrationResult) and query.grid is grid
    assert query.projected.shape == (2, 4, 64) and query.initial_coefficients.shape == (2, 3, 64)
    assert trace.synthesized.shape == (2, 4, 64) and trace.raw_prediction.shape == z.shape
    for value in (query.projected, query.initial_coefficients, query.dynamics.pooled,
                  query.dynamics.coefficients.a_blocks, query.dynamics.coefficients.b_matrix,
                  query.dynamics.spectral_bias, query.dynamics.spatial_bias,
                  trace.integration.final, trace.synthesized, trace.raw_prediction):
        assert value.dtype == torch.float32 and value.device == z.device
        assert value.requires_grad and torch.isfinite(value).all()
    for name in ("eigenvalues", "basis"):
        value, native = getattr(query.spectrum, name), getattr(query.spectrum, name+"_cpu")
        assert value.dtype == torch.float32 and value.device == z.device and not value.requires_grad
        assert native.dtype == torch.float64 and native.device.type == "cpu" and not native.requires_grad
    assert len(spectra) == len(contexts) == 1
    assert spectra[0][1] == {"k": 3, "model_device": z.device}
    assert contexts[0][0][0] is query.projected and contexts[0][0][3] is grid.coords
    assert query.dynamics.coords.dtype == torch.float32 and query.dynamics.coords.device == z.device
    torch.testing.assert_close(query.dynamics.coords, grid.coords.to(z.device, torch.float32), atol=1e-5, rtol=1e-4)
    psi = query.spectrum.basis
    torch.testing.assert_close(query.initial_coefficients, psi.transpose(1, 2) @ query.projected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(trace.synthesized, psi @ trace.integration.final, atol=1e-5, rtol=1e-4)
    expected_raw = trace.synthesized @ model.p_out.weight.T+model.p_out.bias
    torch.testing.assert_close(trace.raw_prediction, expected_raw, atol=1e-5, rtol=1e-4)
    assert len(steps) == len(trace.integration.steps) > 0
    assert len(list(model.dynamics.parameters())) == 23 and len(list(model.parameters())) == 27
    assert set(dict(model.spectrum.named_buffers())) == {"q", "sigma_x", "sigma_z"}
    with pytest.raises(FrozenInstanceError):
        query.grid = regular_grid(1, 4)
    with pytest.raises(FrozenInstanceError):
        trace.raw_prediction = z
    print("D12 real kernel", {"device": device, "modes": 3,
        "n_steps": trace.integration.n_steps.tolist(), "step_calls": len(steps)})


def d12_controlled_projection(model):
    """虚构不互逆投影：提供非退化的可解释重启差异，不修改生产初始化。"""
    device = model.p_in.weight.device
    with torch.no_grad():
        for layer in (model.p_in, model.p_out):
            layer.weight.zero_()
            layer.bias.zero_()
        model.p_in.weight[:3] = torch.tensor([[1.2, .4, 0.], [0., -.7, .3], [.2, 0., .8]], device=device)
        model.p_in.bias[:3] = torch.tensor([.1, -.2, .15], device=device)
        model.p_out.weight[:, :3] = torch.tensor([[.3, .1, -.2], [-.1, .4, .2], [.2, -.3, .5]], device=device)
        model.p_out.bias[:] = torch.tensor([.25, -.1, .3], device=device)


def test_d12_internal_continuation_reuses_query_but_raw_restart_rebuilds(monkeypatch):
    """内部传当前C和旧上下文；式15第二段只传预测raw Z并完整重建。"""
    from po_jepa.actions import split_action_segments
    from po_jepa.geometry import regular_grid

    model = d12_model(max_modes=3)
    d12_controlled_projection(model)
    z = torch.tensor([[[-.7, .2, .8], [.1, -.6, .4], [.9, .3, -.2], [-.2, .7, -.5]]])
    grid = regular_grid(2, 2)
    tau, sigma = torch.tensor([.25], dtype=torch.float64), torch.tensor([.375], dtype=torch.float64)
    controls = d12_controls([[(0., .125, (.2, -.4)), (.125, .5, (-.7, .3)), (.5, 1., (.6, .8))]])
    pair = split_action_segments(controls, tau=tau, sigma=sigma)
    assert torch.equal(pair.first.duration, tau) and torch.equal(pair.second.duration, sigma)
    assert pair.first.source_ids == pair.second.source_ids == controls.source_ids
    assert torch.equal(pair.second.source_origin_seconds, controls.source_origin_seconds+.05*tau)
    spies = {(owner, name): d12_record(monkeypatch, getattr(model, owner) if owner else model, name)
        for owner, name in (("", "project_in"), ("", "project_out"), ("spectrum", "affinity"),
            ("spectrum", "normalized_laplacian"), ("spectrum", "lowest_eigenpairs"),
            ("dynamics", "prepare_context"), ("dynamics", "pool_context"),
            ("dynamics", "condition"), ("dynamics", "integrate_segments"))}
    query0 = model.prepare_query(z, grid)
    internal1 = model.integrate_internal_continuation(query0, query0.initial_coefficients, pair.first, tau)
    internal2 = model.integrate_internal_continuation(query0, internal1.final, pair.second, sigma)
    for key, calls in spies.items():
        assert len(calls) == (0 if key == ("", "project_out") else 2 if key == ("dynamics", "integrate_segments") else 1)
    integrations = spies[("dynamics", "integrate_segments")]
    assert integrations[0][0][0] is query0.initial_coefficients
    assert integrations[1][0][0] is internal1.final
    assert integrations[0][0][2] is integrations[1][0][2] is query0.dynamics
    first = model._predict_raw_with_trace(z, pair.first, tau, grid, collect_trace=True)
    second = model._predict_raw_with_trace(first.raw_prediction, pair.second, sigma, grid, collect_trace=True)
    for key, calls in spies.items():
        assert len(calls) == (2 if key == ("", "project_out") else 4 if key == ("dynamics", "integrate_segments") else 3)
    assert spies[("", "project_in")][2][0][0] is first.raw_prediction
    assert spies[("spectrum", "affinity")][2][0][0] is second.query.projected
    assert spies[("dynamics", "prepare_context")][2][0][0] is second.query.projected
    assert second.query is not first.query and second.query.dynamics is not first.query.dynamics
    torch.testing.assert_close(second.query.projected,
        first.raw_prediction @ model.p_in.weight.T+model.p_in.bias, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(second.query.dynamics.pooled, second.query.projected.mean(1), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(internal1.final, first.integration.final, atol=1e-5, rtol=1e-4)
    continuation_raw = model.project_out(query0.spectrum.basis @ internal2.final)
    gap = (continuation_raw-second.raw_prediction).abs().max().item()
    assert gap > 1e-3
    assert (first.query.dynamics.pooled-second.query.dynamics.pooled).abs().max().item() > 1e-3
    for name, trace in (("first", first), ("restart", second)):
        context = trace.query.dynamics
        print("D12 "+name, {"projected": trace.query.projected[0, 0, :3].tolist(),
            "lambda": trace.query.spectrum.eigenvalues[0].tolist(),
            "C0": trace.query.initial_coefficients[0, 0, :3].tolist(),
            "g": context.pooled[0, :3].tolist(),
            "A": context.coefficients.a_blocks[0, 0, 0].tolist(),
            "B": context.coefficients.b_matrix[0, 0, :3].tolist(),
            "spectral_bias": context.spectral_bias[0, 0].tolist(),
            "spatial_bias": context.spatial_bias[0, 0].tolist(),
            "C_final": trace.integration.final[0, 0, :3].tolist(),
            "raw": trace.raw_prediction[0, 0].tolist()})
    print("D12 continuation", {"first_C": internal1.final[0, 0, :3].tolist(),
        "second_start_C": integrations[1][0][0][0, 0, :3].tolist(),
        "final_C": internal2.final[0, 0, :3].tolist(), "raw": continuation_raw[0, 0].tolist(),
        "restart_gap": gap, "scope": "same first integration; final difference includes readout/rebuild"})


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_d12_feedback_gradient_matches_fixed_spectrum_analytic_reference(device):
    """只以第二段raw损失反传一次；解析导数固定两次实际lambda/Psi，不重求谱。"""
    from po_jepa.actions import split_action_segments
    from po_jepa.geometry import regular_grid

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available; CPU acceptance remains mandatory")
    model = d12_model(device=device, max_modes=3, fixed_rates=True)
    d12_controlled_projection(model)
    z = torch.tensor([[[-.7, .2, .8], [.1, -.6, .4], [.9, .3, -.2], [-.2, .7, -.5]]],
        device=device, requires_grad=True)
    grid = regular_grid(2, 2)
    tau, sigma = torch.tensor([.25], dtype=torch.float64), torch.tensor([.375], dtype=torch.float64)
    controls = d12_controls([[(0., 1., (.2, -.4))]], device=device)
    pair = split_action_segments(controls, tau=tau, sigma=sigma)
    first = model._predict_raw_with_trace(z, pair.first, tau, grid)
    first.raw_prediction.retain_grad()
    second = model._predict_raw_with_trace(first.raw_prediction, pair.second, sigma, grid)
    weight = torch.linspace(-.4, .7, second.raw_prediction.numel(), device=device).reshape_as(second.raw_prediction)
    wi, wo = model.p_in.weight.detach().double(), model.p_out.weight.detach().double()
    psi1, psi2 = first.query.spectrum.basis.double(), second.query.spectrum.basis.double()
    # 固定率各段为exp(-gamma*t)I；只在测试中用独立double矩阵链计算伴随。
    gamma1 = first.query.dynamics.coefficients.gamma.detach().double()
    gamma2 = second.query.dynamics.coefficients.gamma.detach().double()
    scale1 = torch.exp(-gamma1.repeat_interleave(2, -1)*tau.to(device)[:, None, None])
    scale2 = torch.exp(-gamma2.repeat_interleave(2, -1)*sigma.to(device)[:, None, None])
    expected_first_grad = (psi2 @ ((psi2.transpose(1, 2) @ (weight.double() @ wo))*scale2)) @ wi
    expected_z_grad = (psi1 @ ((psi1.transpose(1, 2) @ (expected_first_grad @ wo))*scale1)) @ wi
    (second.raw_prediction*weight).sum().backward()
    torch.testing.assert_close(first.raw_prediction.grad.double(), expected_first_grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(z.grad.double(), expected_z_grad, atol=1e-5, rtol=1e-4)
    assert first.raw_prediction.grad.abs().max().item() > 1e-4 and z.grad.abs().max().item() > 1e-4
    assert model.p_in.weight.grad.abs().max().item() > 0 and model.p_out.weight.grad.abs().max().item() > 0
    print("D12 fixed-rate feedback", {"device": device,
        "first_gradient_error": (first.raw_prediction.grad.double()-expected_first_grad).abs().max().item(),
        "source_gradient_error": (z.grad.double()-expected_z_grad).abs().max().item()})


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_d12_nonzero_real_network_feedback_has_all_key_paths(device):
    """全非零真实网络的新图一次backward；不要求softmax平移等合法零方向非零。"""
    from po_jepa.actions import split_action_segments
    from po_jepa.geometry import regular_grid

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available; CPU acceptance remains mandatory")
    model = d12_model(device=device, max_modes=3)
    z = torch.tensor([[[-.7, .2, .8], [.1, -.6, .4], [.9, .3, -.2], [-.2, .7, -.5]]],
        device=device, requires_grad=True)
    controls = d12_controls([[(0., .375, (.2, -.4)), (.375, 1., (-.7, .3))]], device=device)
    controls.u_values.requires_grad_()
    tau, sigma = torch.tensor([.25], dtype=torch.float64), torch.tensor([.375], dtype=torch.float64)
    pair = split_action_segments(controls, tau=tau, sigma=sigma)
    first = model._predict_raw_with_trace(z, pair.first, tau, regular_grid(2, 2))
    first.raw_prediction.retain_grad()
    second = model._predict_raw_with_trace(first.raw_prediction, pair.second, sigma, regular_grid(2, 2))
    watched = {"first_X": first.query.projected, "first_C0": first.query.initial_coefficients,
        "first_g": first.query.dynamics.pooled, "first_gamma": first.query.dynamics.coefficients.gamma,
        "first_omega": first.query.dynamics.coefficients.omega,
        "first_B": first.query.dynamics.coefficients.b_matrix, "first_final": first.integration.final,
        "second_X": second.query.projected, "second_C0": second.query.initial_coefficients,
        "second_g": second.query.dynamics.pooled, "second_final": second.integration.final}
    for value in watched.values():
        value.retain_grad()
    weight = torch.linspace(-.4, .7, second.raw_prediction.numel(), device=device).reshape_as(second.raw_prediction)
    (second.raw_prediction.square()*(weight+1.)).sum().backward()
    norms = {}
    for name, value in {"first_raw": first.raw_prediction, "source_Z": z, "actions": controls.u_values,
                        **watched}.items():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
        norms[name] = value.grad.abs().max().item()
        assert norms[name] > 0, name
    parameters = dict(model.named_parameters())
    for name in ("p_in.weight", "p_out.weight", "dynamics.condition_trunk.0.weight",
                 "dynamics.gamma_head.weight", "dynamics.omega_head.weight", "dynamics.b_head.weight",
                 "dynamics.q_proj.weight", "dynamics.k_proj.weight", "dynamics.v_proj.weight",
                 "dynamics.spectral_bias_net.0.weight", "dynamics.spectral_bias_net.2.weight",
                 "dynamics.spatial_bias_net.0.weight", "dynamics.spatial_bias_net.2.weight",
                 "dynamics.action_residual_net.0.weight", "dynamics.action_residual_net.2.weight"):
        gradient = parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        norms[name] = gradient.abs().max().item()
        assert norms[name] > 0, name
    assert not first.query.spectrum.basis.requires_grad and not second.query.spectrum.eigenvalues.requires_grad
    print("D12 real feedback", {"device": device, "max_abs_gradients": norms,
        "backward_calls": 1, "first_raw_retain_calls": 1})


@pytest.mark.parametrize("max_modes", [2, 4])
def test_d12_zero_and_mixed_duration_use_real_projected_readout(monkeypatch, max_modes):
    """Phi0始终经谱投影/读出；全基只重建X，截断与不互逆读出均不伪装Z恒等。"""
    from po_jepa.geometry import regular_grid

    model = d12_model(max_modes=max_modes)
    d12_controlled_projection(model)
    z = torch.tensor([[[-.7, .2, .8], [.1, -.6, .4], [.9, .3, -.2], [-.2, .7, -.5]],
                      [[.4, -.2, .1], [-.8, .5, .3], [.6, -.1, -.7], [.2, .8, -.4]]])
    grid = regular_grid(2, 2)
    steps = d12_record(monkeypatch, model.dynamics, "exponential_euler_step")
    zero = model._predict_raw_with_trace(z, d12_controls([[], []]),
        torch.zeros(2, dtype=torch.float64), grid, collect_trace=True)
    assert not steps and zero.integration.steps == ()
    assert torch.equal(zero.integration.n_steps, torch.zeros(2, dtype=torch.int64))
    assert torch.equal(zero.integration.final, zero.query.initial_coefficients)
    # 参考使用独立double矩阵表达式；生产输出必须由上述同一内核实际算出。
    x = z.double() @ model.p_in.weight.detach().double().T+model.p_in.bias.detach().double()
    psi = zero.query.spectrum.basis.double()
    reconstructed = psi @ (psi.transpose(1, 2) @ x)
    expected = reconstructed @ model.p_out.weight.detach().double().T+model.p_out.bias.detach().double()
    torch.testing.assert_close(zero.raw_prediction.double(), expected, atol=1e-5, rtol=1e-4)
    gap = (zero.raw_prediction-z).abs().max().item()
    truncation = (reconstructed-x).abs().max().item()
    assert gap > 1e-2
    if max_modes == 4:
        torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-4)
    else:
        assert truncation > 1e-3
    controls = d12_controls([[], [(0., .5, (.6, -.3))]])
    mixed = model._predict_raw_with_trace(z, controls,
        torch.tensor([0., .375], dtype=torch.float64), grid, collect_trace=True)
    torch.testing.assert_close(mixed.raw_prediction[0], zero.raw_prediction[0], atol=1e-5, rtol=1e-4)
    assert torch.equal(mixed.integration.n_steps, torch.tensor([0, 2]))
    assert all(torch.equal(step.batch_index, torch.tensor([1])) for step in mixed.integration.steps)
    single = model._predict_raw_with_trace(z[1:], d12_controls([[(0., .5, (.6, -.3))]]),
        torch.tensor([.375], dtype=torch.float64), grid)
    torch.testing.assert_close(mixed.raw_prediction[1:], single.raw_prediction, atol=1e-5, rtol=1e-4)
    print("D12 zero readout", {"modes": max_modes, "nonidentity_gap": gap,
        "truncation_gap": truncation, "mixed_steps": mixed.integration.n_steps.tolist()})


def test_d12_collect_trace_changes_only_retention_not_network_calls(monkeypatch):
    """trace来自同次真实单步，开启诊断不重新构图、条件化或运行任何网络。"""
    from po_jepa.geometry import regular_grid

    model = d12_model(max_modes=3)
    z = torch.linspace(-.7, .9, 12).reshape(1, 4, 3)
    controls = d12_controls([[(0., .2, (.2, -.4)), (.2, .8, (-.7, .3))]])
    duration, grid = torch.tensor([.6], dtype=torch.float64), regular_grid(2, 2)
    spies = {(owner, name): d12_record(monkeypatch, getattr(model, owner) if owner else model, name)
        for owner, name in (("", "project_in"), ("", "project_out"), ("spectrum", "affinity"),
            ("spectrum", "normalized_laplacian"), ("spectrum", "lowest_eigenpairs"),
            ("dynamics", "pool_context"), ("dynamics", "condition"),
            ("dynamics", "spectral_bias"), ("dynamics", "spatial_bias"),
            ("dynamics", "nonlinear"), ("dynamics", "exponential_euler_step"))}
    plain = model._predict_raw_with_trace(z, controls, duration, grid)
    counts = {key: len(calls) for key, calls in spies.items()}
    traced = model._predict_raw_with_trace(z, controls, duration, grid, collect_trace=True)
    assert all(len(calls) == 2*counts[key] for key, calls in spies.items())
    assert plain.integration.steps == () and len(traced.integration.steps) == counts[("dynamics", "exponential_euler_step")]
    for name in ("raw_prediction", "synthesized"):
        assert torch.equal(getattr(plain, name), getattr(traced, name))
    assert torch.equal(plain.integration.final, traced.integration.final)
    assert torch.equal(plain.integration.n_steps, traced.integration.n_steps)
    actual_steps = spies[("dynamics", "exponential_euler_step")][counts[("dynamics", "exponential_euler_step")]:]
    assert all(step.result is actual[2] for step, actual in zip(traced.integration.steps, actual_steps))


def d12_tensor_leaves(value):
    """递归枚举只读数据合同里的张量，用原值快照核对新增路径无原地改写。"""
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from d12_tensor_leaves(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from d12_tensor_leaves(item)


def test_d12_no_input_query_parameter_member_mode_or_rng_mutation():
    """共享模型跨段/跨网格复用，不缓存查询或改变输入、参数、模式与任何RNG。"""
    from po_jepa.geometry import regular_grid

    model = d12_model(max_modes=3)
    model.spectrum.eval()
    model.p_out.eval()
    z = torch.linspace(-.7, .9, 12).reshape(1, 4, 3)
    grid = regular_grid(2, 2)
    controls = d12_controls([[(0., .2, (.2, -.4)), (.2, .8, (-.7, .3))]])
    duration = torch.tensor([.375], dtype=torch.float64)
    query = model.prepare_query(z, grid)
    tensors = list(d12_tensor_leaves((z, grid, controls, duration, query)))
    before = [(value, value.detach().clone()) for value in tensors]
    state = {name: (id(value), value.detach().clone()) for name, value in
             (*model.named_parameters(), *model.named_buffers())}
    members = {name: set(vars(module)) for name, module in model.named_modules()}
    modes = {name: module.training for name, module in model.named_modules()}
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    defaults = torch.get_default_dtype(), torch.get_default_device()
    model.integrate_internal_continuation(query, query.initial_coefficients, controls, duration)
    first = model._predict_raw_with_trace(z, controls, duration, grid)
    model._predict_raw_with_trace(first.raw_prediction, controls, duration, grid, collect_trace=True)
    other = torch.linspace(-.3, .8, 9).reshape(1, 3, 3)
    assert model._predict_raw_with_trace(other, controls, duration, regular_grid(1, 3)).raw_prediction.shape == other.shape
    for value, expected in before:
        torch.testing.assert_close(value, expected, atol=0, rtol=0, equal_nan=True)
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        assert id(value) == state[name][0] and torch.equal(value, state[name][1])
        assert value.grad is None
    assert members == {name: set(vars(module)) for name, module in model.named_modules()}
    assert modes == {name: module.training for name, module in model.named_modules()}
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())
    assert torch.equal(torch.get_rng_state(), rng) and random.getstate() == py_rng
    current_np = np.random.get_state()
    assert current_np[0] == np_rng[0] and np.array_equal(current_np[1], np_rng[1]) and current_np[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))


def test_d12_noncontiguous_raw_and_current_coefficients_are_preserved():
    """合法非连续视图与连续副本等价；内部续算实际消费当前C而非偷偷重置C0。"""
    from po_jepa.geometry import regular_grid

    model = d12_model(max_modes=3)
    raw = torch.linspace(-.7, .9, 24).reshape(1, 4, 6)[..., ::2]
    assert not raw.is_contiguous()
    controls = d12_controls([[(0., .5, (.2, -.4))]])
    duration, grid = torch.tensor([.375], dtype=torch.float64), regular_grid(2, 2)
    actual = model._predict_raw_with_trace(raw, controls, duration, grid)
    contiguous = model._predict_raw_with_trace(raw.contiguous(), controls, duration, grid)
    torch.testing.assert_close(actual.raw_prediction, contiguous.raw_prediction, atol=1e-5, rtol=1e-4)
    query = actual.query
    changed = 1.3*query.initial_coefficients+.1
    current = torch.stack((changed, torch.zeros_like(changed)), -1)[..., 0]
    assert not current.is_contiguous()
    before = current.detach().clone()
    result = model.integrate_internal_continuation(query, current, controls, duration)
    expected = model.dynamics.integrate_segments(current.contiguous(), controls, query.dynamics,
        duration=duration, config=model.integrator)
    reset = model.dynamics.integrate_segments(query.initial_coefficients, controls, query.dynamics,
        duration=duration, config=model.integrator)
    torch.testing.assert_close(result.final, expected.final, atol=1e-5, rtol=1e-4)
    assert (result.final-reset.final).abs().max().item() > 1e-3 and torch.equal(current, before)
    zero = model.integrate_internal_continuation(query, current, d12_controls([[]]),
        torch.zeros(1, dtype=torch.float64))
    assert torch.equal(zero.final, current) and zero.steps == ()


def test_d12_production_signatures_accept_only_known_start_controls_and_grid():
    """API不接target/真实中间观察/隐藏物理状态/query复用开关或预测callback。"""
    expected = {
        "prepare_query": ("self", "raw_z", "grid"),
        "integrate_internal_continuation": ("self", "query", "current_coefficients", "controls", "duration"),
        "_predict_raw_with_trace": ("self", "raw_z", "controls", "duration", "grid", "collect_trace"),
    }
    for name, names in expected.items():
        signature = inspect.signature(getattr(api().POJEPA, name))
        assert tuple(signature.parameters) == names
        assert all(parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                   for parameter in signature.parameters.values())
    trace_argument = inspect.signature(api().POJEPA._predict_raw_with_trace).parameters["collect_trace"]
    assert trace_argument.kind is inspect.Parameter.KEYWORD_ONLY and trace_argument.default is False


@pytest.mark.parametrize("component", ["spectrum", "dynamics", "integrator"])
@pytest.mark.parametrize("entry", ["prepare_query", "_predict_raw_with_trace", "integrate_internal_continuation"])
def test_d12_production_entries_reject_c01_reference_sentinels(component, entry):
    """C01仍可保管哨兵引用；实际计算入口必须拒绝假谱/假动力学/假积分配置。"""
    from po_jepa.geometry import regular_grid

    model = d12_model()
    z, grid = torch.ones(1, 4, 3), regular_grid(2, 2)
    controls = d12_controls([[(0., .5, (.2, -.4))]])
    duration = torch.tensor([.25], dtype=torch.float64)
    query = model.prepare_query(z, grid)
    setattr(model, component, object() if component == "integrator" else PassiveReference())
    args = (z, grid) if entry == "prepare_query" else (z, controls, duration, grid) if entry == "_predict_raw_with_trace" else (
        query, query.initial_coefficients, controls, duration)
    with pytest.raises(TypeError, match={"spectrum": "GraphSpectrum", "dynamics": "ModalDynamics", "integrator": "IntegratorConfig"}[component]):
        getattr(model, entry)(*args)


@pytest.mark.parametrize("bad", ["raw_object", "raw_rank", "raw_width", "raw_dtype", "raw_device",
    "raw_nan", "grid_object", "grid_shape", "grid_corrupted_coords", "parameter_dtype", "parameter_device"])
def test_d12_prepare_query_rejects_illegal_raw_grid_and_model_contracts(bad):
    """生产起点拒绝错轴/设备/精度/网格，不隐式转换、不传播坏输入或改既有投影。"""
    from po_jepa.geometry import regular_grid

    model, grid = d12_model(), regular_grid(2, 2)
    raw = torch.ones(1, 4, 3)
    if bad == "raw_object":
        raw = object()
    elif bad == "raw_rank":
        raw = raw.unsqueeze(0)
    elif bad == "raw_width":
        raw = raw[..., :2]
    elif bad == "raw_dtype":
        raw = raw.double()
    elif bad == "raw_device":
        raw = raw.to("meta")
    elif bad == "raw_nan":
        raw[0, 0, 0] = float("nan")
    elif bad == "grid_object":
        grid = object()
    elif bad == "grid_shape":
        grid = regular_grid(1, 3)
    elif bad == "grid_corrupted_coords":
        grid.coords[0, 0] = float("nan")
    elif bad == "parameter_dtype":
        model.dynamics.double()
    else:
        model.dynamics.to("meta")
    with pytest.raises(TypeError if bad in ("raw_object", "grid_object") else ValueError):
        model.prepare_query(raw, grid)


@pytest.mark.parametrize("bad", ["controls_object", "controls_dtype", "controls_device", "controls_nan",
    "controls_action_width", "controls_gap", "duration_object", "duration_dtype", "duration_shape",
    "duration_device", "duration_negative", "duration_nan", "duration_gradient", "strict_undercoverage", "trace_flag"])
def test_d12_shared_kernel_rejects_invalid_control_time_and_trace_contracts(bad):
    """D12沿用D11严格覆盖/时间/已知动作边界，不能以默认值或模型补尾放行。"""
    from po_jepa.actions import ControlSpec
    from po_jepa.geometry import regular_grid

    model, grid = d12_model(), regular_grid(2, 2)
    raw = torch.ones(1, 4, 3)
    controls = d12_controls([[(0., .25, (.2, -.4)), (.25, .5, (-.7, .3))]])
    duration, collect_trace = torch.tensor([.375], dtype=torch.float64), False
    if bad == "controls_object":
        controls = object()
    elif bad == "controls_dtype":
        controls = replace(controls, u_values=controls.u_values.double())
    elif bad == "controls_device":
        controls = replace(controls, u_values=controls.u_values.to("meta"))
    elif bad == "controls_nan":
        controls.u_values[0, 0, 0] = float("nan")
    elif bad == "controls_action_width":
        spec = ControlSpec(1, .05, ("fictional_u",), ("unit",), (-100.,), (100.,), (1.,), (0.,),
            "D12 wrong action dimension diagnostic")
        controls = replace(controls, control_spec=spec, u_values=controls.u_values[..., :1])
    elif bad == "controls_gap":
        controls.starts[0, 1] = torch.nextafter(torch.tensor(.25, dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
    elif bad == "duration_object":
        duration = [.375]
    elif bad == "duration_dtype":
        duration = duration.float()
    elif bad == "duration_shape":
        duration = duration.expand(2)
    elif bad == "duration_device":
        duration = duration.to("meta")
    elif bad == "duration_negative":
        duration[0] = -.1
    elif bad == "duration_nan":
        duration[0] = float("nan")
    elif bad == "duration_gradient":
        duration.requires_grad_()
    elif bad == "strict_undercoverage":
        duration[0] = torch.nextafter(torch.tensor(.5, dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
    else:
        collect_trace = 1
    with pytest.raises(TypeError if bad in ("controls_object", "duration_object", "trace_flag") else ValueError):
        model._predict_raw_with_trace(raw, controls, duration, grid, collect_trace=collect_trace)


@pytest.mark.parametrize("bad", ["query_object", "context_object", "current_object", "current_dtype",
    "current_device", "current_batch", "current_modes", "current_width", "current_nan"])
def test_d12_internal_continuation_rejects_invalid_query_and_current_coefficients(bad):
    """内部诊断不因复用上下文而跳过当前C或DynamicsContext的生产输入核对。"""
    from po_jepa.geometry import regular_grid

    model = d12_model(max_modes=3)
    query = model.prepare_query(torch.ones(1, 4, 3), regular_grid(2, 2))
    current = query.initial_coefficients
    if bad == "query_object":
        query = object()
    elif bad == "context_object":
        query = replace(query, dynamics=object())
    elif bad == "current_object":
        current = object()
    elif bad == "current_dtype":
        current = current.double()
    elif bad == "current_device":
        current = current.to("meta")
    elif bad == "current_batch":
        current = current.expand(2, -1, -1)
    elif bad == "current_modes":
        current = current[:, :2]
    elif bad == "current_width":
        current = current[..., :63]
    else:
        current = current.clone()
        current[0, 0, 0] = float("nan")
    with pytest.raises(TypeError if bad in ("query_object", "context_object", "current_object") else ValueError):
        model.integrate_internal_continuation(query, current,
            d12_controls([[(0., .5, (.2, -.4))]]), torch.tensor([.25], dtype=torch.float64))


def test_e01_public_entries_execute_the_shared_kernel_once(monkeypatch):
    """虚构Z验收公开Phi：每个入口仅执行一次真实内核，trace仅改变保留方式。"""
    from po_jepa.geometry import regular_grid

    model = d12_model(max_modes=3)
    raw = torch.linspace(-.7, .9, 12).reshape(1, 4, 3)
    controls = d12_controls([[(0., .5, (.2, -.4))]])
    duration, grid = torch.tensor([.375], dtype=torch.float64), regular_grid(2, 2)
    calls = d12_record(monkeypatch, model, "_predict_raw_with_trace")
    plain = model(raw, controls, duration, grid)
    traced = model.trace_forward(raw, controls, duration, grid)
    assert len(calls) == 2
    assert calls[0][1] == {"collect_trace": False} and calls[1][1] == {"collect_trace": True}
    assert isinstance(plain, torch.Tensor) and isinstance(traced, api().PredictionTrace)
    assert plain is calls[0][2].raw_prediction and traced is calls[1][2]
    assert calls[0][2].integration.steps == () and traced.integration.steps
    assert torch.equal(plain, traced.raw_prediction)
    assert torch.equal(calls[0][2].integration.final, traced.integration.final)


def test_e01_public_signatures_and_complete_reference_sentinel_rejection():
    """公开签名只接受已知起点/控制/时间/网格；C01哨兵完整调用仍明确拒绝。"""
    from po_jepa.geometry import regular_grid

    for entry, result in (("forward", "Tensor"), ("trace_forward", "PredictionTrace")):
        signature = inspect.signature(getattr(api().POJEPA, entry))
        assert tuple(signature.parameters) == ("self", "raw_z", "controls", "duration", "grid")
        assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default is inspect.Parameter.empty
                   for p in signature.parameters.values())
        assert signature.return_annotation == result
        for component in ("spectrum", "dynamics", "integrator"):
            model = d12_model()
            setattr(model, component, object() if component == "integrator" else PassiveReference())
            with pytest.raises(TypeError, match=component):
                getattr(model, entry)(torch.ones(1, 4, 3), d12_controls([[(0., .5, (.2, -.4))]]),
                    torch.tensor([.375], dtype=torch.float64), regular_grid(2, 2))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_e01_full_width_real_sparse_chain_trace_and_joint_gradient(monkeypatch, device):
    """正式E=1408配虚构特征/控制：真实稀疏生产链、异步切换和新图一次联合反向。"""
    from po_jepa.geometry import regular_grid

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available; CPU acceptance remains mandatory")
    base = d12_model(device=device, max_modes=20)
    model = api().POJEPA(api().ProjectionConfig(1408, 64), base.spectrum,
        base.dynamics, base.integrator).to(device)
    index = torch.arange(2*24*1408, dtype=torch.float32, device=device).reshape(2, 24, 1408)
    raw = (.2*torch.sin(index/37)+.1*torch.cos(index/11)).requires_grad_()
    grid = regular_grid(4, 6)
    controls = d12_controls([[(0., .375, (.2, -.4)), (.375, .75, (-.7, .3))],
                            [(0., .125, (.8, .1)), (.125, .5, (-.3, .6))]], device=device)
    controls.u_values.requires_grad_()
    duration = torch.tensor([.625, .375], dtype=torch.float64)
    keys = (("", "_predict_raw_with_trace"), ("", "prepare_query"), ("", "project_in"),
        ("", "project_out"), ("spectrum", "affinity"), ("spectrum", "normalized_laplacian"),
        ("spectrum", "lowest_eigenpairs"), ("dynamics", "prepare_context"),
        ("dynamics", "pool_context"), ("dynamics", "condition"), ("dynamics", "spectral_bias"),
        ("dynamics", "spatial_bias"), ("dynamics", "integrate_segments"),
        ("dynamics", "nonlinear"), ("dynamics", "exponential_euler_step"))
    calls = {key: d12_record(monkeypatch, getattr(model, key[0]) if key[0] else model, key[1])
             for key in keys}
    prediction = model(raw, controls, duration, grid)
    counts = {key: len(value) for key, value in calls.items()}
    trace = model.trace_forward(raw, controls, duration, grid)
    assert all(len(value) == 2*counts[key] for key, value in calls.items())
    assert all(counts[key] == (3 if key[1] in ("nonlinear", "exponential_euler_step") else 1) for key in keys)
    plain = calls[("", "_predict_raw_with_trace")][0][2]
    assert plain.integration.steps == () and prediction is plain.raw_prediction
    assert trace is calls[("", "_predict_raw_with_trace")][1][2]
    assert torch.equal(prediction, trace.raw_prediction) and torch.equal(plain.integration.final, trace.integration.final)
    # 目标只在测试外部检查轴、顺序、精度和设备，不传给预测器，也不实现E03距离。
    target = torch.cos(index/19)
    assert prediction.shape == target.shape == (2, 24, 1408)
    assert prediction.dtype == target.dtype == torch.float32 and prediction.device == target.device
    query, psi = trace.query, trace.query.spectrum.basis
    assert query.grid is grid and query.projected.shape == (2, 24, 64)
    assert query.initial_coefficients.shape == trace.integration.final.shape == (2, 20, 64)
    torch.testing.assert_close(query.initial_coefficients, psi.transpose(1, 2) @ query.projected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(trace.synthesized, psi @ trace.integration.final, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(prediction, trace.synthesized @ model.p_out.weight.T+model.p_out.bias, atol=1e-5, rtol=1e-4)
    for name in ("basis", "eigenvalues"):
        native, value = getattr(query.spectrum, name+"_cpu"), getattr(query.spectrum, name)
        assert native.device.type == "cpu" and native.dtype == torch.float64 and not native.requires_grad
        assert value.device == raw.device and value.dtype == torch.float32 and not value.requires_grad
    for name in ("affinity", "normalized_laplacian"):
        value = calls[("spectrum", name)][1][2]
        value = value.laplacian if name == "normalized_laplacian" else value
        assert value.dtype == torch.float64 and value.device.type == "cpu" and not value.requires_grad
        assert torch.isfinite(value).all()
    context_call = calls[("dynamics", "prepare_context")][1]
    assert context_call[0][0] is query.projected and context_call[0][3] is grid.coords
    assert query.dynamics.coords.dtype == torch.float32 and query.dynamics.coords.device == raw.device
    torch.testing.assert_close(query.dynamics.coords, grid.coords.to(raw.device, torch.float32), atol=1e-5, rtol=1e-4)
    expected = (([0, 1], [0., 0.], [.25, .125], [0, 0]),
                ([0, 1], [.25, .125], [.125, .25], [0, 1]), ([0], [.375], [.25], [1]))
    actual_steps = calls[("dynamics", "exponential_euler_step")][3:]
    assert torch.equal(trace.integration.n_steps, torch.tensor([3, 2]))
    assert len(trace.integration.steps) == len(actual_steps) == 3
    for position, (step, actual, (batch, start, delta, segment)) in enumerate(zip(trace.integration.steps, actual_steps, expected)):
        assert step.result is actual[2] and step.before is actual[0][0]
        assert step.delta is actual[1]["delta"]
        for value, values, dtype in ((step.batch_index, batch, torch.int64),
                (step.segment_index, segment, torch.int64), (step.start_time, start, torch.float64),
                (step.delta, delta, torch.float64)):
            assert value.device.type == "cpu" and value.dtype == dtype
            assert torch.equal(value, torch.tensor(values, dtype=dtype))
        rhs, active = step.result.rhs, actual[0][2]
        interaction = rhs.interaction
        assert interaction is calls[("dynamics", "nonlinear")][3+position][2]
        torch.testing.assert_close(rhs.free, (active.coefficients.a_blocks @ step.before.reshape(len(batch), 20, 32, 2, 1)).reshape_as(step.before), atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(rhs.action_linear, torch.einsum("bkcm,bm->bkc", active.coefficients.b_matrix, actual[0][1]), atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(interaction.total_scores, interaction.content.scores+interaction.spectral_scores+interaction.spatial_scores, atol=1e-5, rtol=1e-4)
        assert torch.equal(interaction.spectral_scores, active.spectral_bias)
        assert torch.equal(interaction.spatial_scores, active.spatial_bias)
        assert interaction.attention.neighbor_index.shape == (len(batch), 20, 16)
        torch.testing.assert_close(interaction.attention.selected_scores,
            interaction.total_scores.gather(-1, interaction.attention.neighbor_index), atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(interaction.attention.weights.sum(-1), torch.ones(len(batch), 20, device=device), atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(interaction.nonlinear, interaction.attention.aggregate+interaction.action_residual, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(rhs.derivative, rhs.free+rhs.action_linear+interaction.nonlinear, atol=1e-5, rtol=1e-4)
        for value in d12_tensor_leaves(step.result):
            assert torch.isfinite(value).all()
    assert len(list(model.parameters())) == 27 and len(list(model.dynamics.parameters())) == 23
    assert sum(p.numel() for p in model.parameters()) == 215778
    watched = {"Z": raw, "action": controls.u_values, "X": query.projected,
        "C0": query.initial_coefficients, "g": query.dynamics.pooled,
        "gamma": query.dynamics.coefficients.gamma, "omega": query.dynamics.coefficients.omega,
        "B": query.dynamics.coefficients.b_matrix, "C_final": trace.integration.final}
    for value in watched.values():
        if not value.is_leaf:
            value.retain_grad()
    # 测试标量仅探测连接；新图只做一次backward，不累计非叶保留梯度。
    weight = torch.linspace(.3, 1.1, prediction.numel(), device=device).reshape_as(prediction)
    (trace.raw_prediction.square()*weight).sum().backward()
    gradients = {}
    for name, value in {**watched, **dict(model.named_parameters())}.items():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
        gradients[name] = value.grad.abs().max().item()
        if name in watched or name.endswith("weight"):
            assert gradients[name] > 0, name
    # 两类末层标量bias的共同平移可合法为零；连接存在与有效权重非零分开检查。
    print("E01 real public sparse chain", {"device": device, "shape": list(prediction.shape),
        "K": 20, "r_eff": 16, "n_steps": trace.integration.n_steps.tolist(),
        "parameter_tensors": 27, "parameter_scalars": 215778, "backward_calls": 1,
        "max_abs_gradients": gradients})


def test_e01_public_raw_restart_rebuilds_and_keeps_first_prediction_gradient(monkeypatch):
    """式15公开组合仅接预测raw Z；重新起点/图谱/Pool，单次反向到第一段合法路径。"""
    from po_jepa.actions import split_action_segments
    from po_jepa.geometry import regular_grid

    model, grid = d12_model(max_modes=3), regular_grid(2, 2)
    raw = torch.tensor([[[-.7, .2, .8], [.1, -.6, .4], [.9, .3, -.2], [-.2, .7, -.5]]], requires_grad=True)
    controls = d12_controls([[(0., .375, (.2, -.4)), (.375, 1., (-.7, .3))]])
    controls.u_values.requires_grad_()
    tau, sigma = torch.tensor([.25], dtype=torch.float64), torch.tensor([.375], dtype=torch.float64)
    pair = split_action_segments(controls, tau=tau, sigma=sigma)
    calls = {name: d12_record(monkeypatch, owner, name) for owner, name in
        ((model, "prepare_query"), (model.spectrum, "affinity"), (model.spectrum, "lowest_eigenpairs"),
         (model.dynamics, "pool_context"), (model.dynamics, "condition"))}
    first = model(raw, pair.first, tau, grid)
    first.retain_grad()
    second = model(first, pair.second, sigma, grid)
    assert all(len(value) == 2 for value in calls.values())
    query1, query2 = calls["prepare_query"][0][2], calls["prepare_query"][1][2]
    assert calls["prepare_query"][1][0][0] is first and calls["prepare_query"][1][0][1] is grid
    assert query1 is not query2 and query1.dynamics is not query2.dynamics
    assert calls["affinity"][1][0][0] is query2.projected
    assert calls["pool_context"][1][0][0] is query2.projected
    watched = {"first_raw": first, "first_X": query1.projected, "first_C0": query1.initial_coefficients,
        "first_g": query1.dynamics.pooled, "first_gamma": query1.dynamics.coefficients.gamma,
        "first_B": query1.dynamics.coefficients.b_matrix}
    for name, value in watched.items():
        if name != "first_raw":
            value.retain_grad()
    (second.square()*torch.linspace(.3, 1.1, second.numel()).reshape_as(second)).sum().backward()
    for name, value in {**watched, "source_Z": raw, "source_actions": controls.u_values}.items():
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().max() > 0, name
    assert not query1.spectrum.basis.requires_grad and not query2.spectrum.eigenvalues.requires_grad
    print("E01 public restart", {"rebuilds": 2, "first_raw_max_abs_grad": first.grad.abs().max().item(),
        "first_X_max_abs_grad": query1.projected.grad.abs().max().item(), "backward_calls": 1})


def test_e01_same_parameters_cross_grid_rebuild_from_actual_rz(monkeypatch):
    """同一模型的细/粗公开Phi分别用Z/RZ重建；不剪细模态，不断言未训练空间恒等。"""
    from po_jepa.geometry import apply_spatial_transfer, build_spatial_transfer, regular_grid

    model, fine_grid = d12_model(max_modes=20), regular_grid(4, 6)
    raw = torch.sin(torch.arange(144, dtype=torch.float32).reshape(2, 24, 3)/7)
    transfer = build_spatial_transfer(fine_grid, target_height=2, target_width=3)
    coarse_raw = apply_spatial_transfer(raw, transfer)
    controls = d12_controls([[(0., .5, (.2, -.4))], [(0., .5, (-.7, .3))]])
    duration = torch.tensor([.375, .25], dtype=torch.float64)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    ids = tuple(id(p) for p in model.parameters())
    config = model.spectrum.config, model.dynamics.config, model.integrator
    spectra = d12_record(monkeypatch, model.spectrum, "lowest_eigenpairs")
    contexts = d12_record(monkeypatch, model.dynamics, "prepare_context")
    spaces = d12_record(monkeypatch, model.dynamics, "spatial_descriptors")
    fine = model.trace_forward(raw, controls, duration, fine_grid)
    coarse = model.trace_forward(coarse_raw, controls, duration, transfer.target)
    assert [call[1]["k"] for call in spectra] == [20, 6]
    for trace, value, grid, modes, context, spatial in zip((fine, coarse), (raw, coarse_raw),
            (fine_grid, transfer.target), (20, 6), contexts, spaces):
        assert trace.query.grid is grid and trace.raw_prediction.shape == value.shape
        assert trace.query.spectrum.basis.shape == (2, value.shape[1], modes)
        assert context[0][0] is trace.query.projected and context[0][3] is grid.coords
        assert spatial[0][0] is trace.query.spectrum.basis
        assert spatial[0][1] is trace.query.dynamics.coords
        assert spatial[2].pair_features.shape == (2, modes, modes, 16)
        assert torch.isfinite(trace.raw_prediction).all()
    assert tuple(id(p) for p in model.parameters()) == ids and len(ids) == 27
    assert config == (model.spectrum.config, model.dynamics.config, model.integrator)
    assert state.keys() == model.state_dict().keys()
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    print("E01 same-model RZ", {"fine": [24, 20], "coarse": [6, 6], "parameter_tensors": len(ids)})


@pytest.mark.parametrize("entry", ["forward", "trace_forward"])
@pytest.mark.parametrize("bad", ["raw_object", "raw_nan", "grid_shape", "action_width",
    "time_object", "time_dtype", "time_shape", "strict_undercoverage", "tiny_uncovered_tail"])
def test_e01_public_entries_preserve_real_boundary_errors(entry, bad):
    """公开包装不转换或补尾；tiny正时长的真实欠覆盖仍在D11精确拒绝。"""
    from po_jepa.actions import ControlSpec
    from po_jepa.geometry import regular_grid

    model, raw, grid = d12_model(), torch.ones(1, 4, 3), regular_grid(2, 2)
    controls = d12_controls([[(0., .5, (.2, -.4))]])
    duration = torch.tensor([.375], dtype=torch.float64)
    if bad == "raw_object":
        raw = object()
    elif bad == "raw_nan":
        raw[0, 0, 0] = float("nan")
    elif bad == "grid_shape":
        grid = regular_grid(1, 3)
    elif bad == "action_width":
        spec = ControlSpec(1, .05, ("fictional_u",), ("unit",), (-100.,), (100.,), (1.,), (0.,),
            "E01 fictional wrong-width diagnostic")
        controls = replace(controls, control_spec=spec, u_values=controls.u_values[..., :1])
    elif bad == "time_object":
        duration = [.375]
    elif bad == "time_dtype":
        duration = duration.float()
    elif bad == "time_shape":
        duration = duration[:, None]
    elif bad == "strict_undercoverage":
        duration[0] = torch.nextafter(torch.tensor(.5, dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
    else:
        controls = d12_controls([[(0., 1e-21, (.2, -.4))]])
        duration[0] = 1e-20
    with pytest.raises(TypeError if bad in ("raw_object", "time_object") else ValueError):
        getattr(model, entry)(raw, controls, duration, grid)


def test_e01_public_zero_mixed_noncontiguous_and_representable_tiny_time(monkeypatch):
    """虚构Phi0非恒等、混合零/正查询和非连续Z；可表示tiny正时长仍执行真实正步。"""
    from po_jepa.geometry import regular_grid

    model, grid = d12_model(max_modes=3), regular_grid(2, 2)
    d12_controlled_projection(model)
    raw = torch.linspace(-.7, .9, 48).reshape(2, 4, 6)[..., ::2]
    assert not raw.is_contiguous()
    steps = d12_record(monkeypatch, model.dynamics, "exponential_euler_step")
    zero_controls, zero_duration = d12_controls([[], []]), torch.zeros(2, dtype=torch.float64)
    zero = model.trace_forward(raw, zero_controls, zero_duration, grid)
    plain_zero = model(raw, zero_controls, zero_duration, grid)
    assert not steps and zero.integration.steps == () and torch.equal(plain_zero, zero.raw_prediction)
    assert torch.equal(zero.integration.n_steps, torch.zeros(2, dtype=torch.int64))
    assert torch.equal(zero.integration.final, zero.query.initial_coefficients)
    x = raw.double() @ model.p_in.weight.detach().double().T+model.p_in.bias.detach().double()
    psi = zero.query.spectrum.basis.double()
    expected = (psi @ (psi.transpose(1, 2) @ x)) @ model.p_out.weight.detach().double().T+model.p_out.bias.detach().double()
    torch.testing.assert_close(zero.raw_prediction.double(), expected, atol=1e-5, rtol=1e-4)
    assert (zero.raw_prediction-raw).abs().max() > 1e-2
    controls, duration = d12_controls([[], [(0., .5, (.6, -.3))]]), torch.tensor([0., .375], dtype=torch.float64)
    mixed = model.trace_forward(raw, controls, duration, grid)
    plain = model(raw.contiguous(), controls, duration, grid)
    torch.testing.assert_close(plain, mixed.raw_prediction, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(mixed.raw_prediction[0], zero.raw_prediction[0], atol=1e-5, rtol=1e-4)
    assert torch.equal(mixed.integration.n_steps, torch.tensor([0, 2]))
    assert all(torch.equal(step.batch_index, torch.tensor([1])) for step in mixed.integration.steps)
    tiny_duration = torch.tensor([1e-21], dtype=torch.float64)
    tiny = model.trace_forward(raw[:1], d12_controls([[(0., 1e-21, (.2, -.4))]]), tiny_duration, grid)
    assert tiny.integration.n_steps.item() == len(tiny.integration.steps) == 1
    assert torch.equal(tiny.integration.steps[0].delta, tiny_duration)
    assert torch.isfinite(tiny.raw_prediction).all()
    print("E01 zero/mixed/tiny", {"zero_gap": (zero.raw_prediction-raw).abs().max().item(),
        "mixed_steps": mixed.integration.n_steps.tolist(), "tiny_delta": tiny.integration.steps[0].delta.tolist()})


@pytest.mark.parametrize("training", [True, False])
def test_e01_queries_preserve_inputs_state_members_modes_rng_and_external_no_grad(training):
    """公开查询无输入/状态/RNG副作用；调用者no_grad正常生效，模型不自行切模式。"""
    from po_jepa.geometry import regular_grid

    model = d12_model(max_modes=3)
    model.train(training)
    model.spectrum.eval()
    model.p_out.eval()
    raw = torch.linspace(-.7, .9, 12).reshape(1, 4, 3).requires_grad_()
    controls = d12_controls([[(0., .5, (.2, -.4))]])
    grid, duration = regular_grid(2, 2), torch.tensor([.375], dtype=torch.float64)
    before = [(value, value.detach().clone()) for value in d12_tensor_leaves((raw, controls, duration, grid))]
    state = {name: (id(value), value.clone()) for name, value in
             (*model.named_parameters(), *model.named_buffers())}
    members = {name: {key: id(value) for key, value in vars(module).items()}
               for name, module in model.named_modules()}
    modes = {name: module.training for name, module in model.named_modules()}
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    defaults = torch.get_default_dtype(), torch.get_default_device()
    assert model(raw, controls, duration, grid).requires_grad
    with torch.no_grad():
        plain = model(raw, controls, duration, grid)
        trace = model.trace_forward(raw, controls, duration, grid)
        assert not plain.requires_grad and not any(value.requires_grad for value in d12_tensor_leaves(trace))
        torch.testing.assert_close(plain, trace.raw_prediction, atol=1e-5, rtol=1e-4)
    for value, expected in before:
        torch.testing.assert_close(value, expected, atol=0, rtol=0, equal_nan=True)
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        assert id(value) == state[name][0] and torch.equal(value, state[name][1]) and value.grad is None
    assert raw.grad is None
    assert members == {name: {key: id(value) for key, value in vars(module).items()}
                       for name, module in model.named_modules()}
    assert modes == {name: module.training for name, module in model.named_modules()}
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())
    assert torch.equal(torch.get_rng_state(), rng) and random.getstate() == py_rng
    current_np = np.random.get_state()
    assert current_np[0] == np_rng[0] and np.array_equal(current_np[1], np_rng[1]) and current_np[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
