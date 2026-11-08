"""C06—D11：虚构条件、完整N/RHS、指数Euler和分段推进；m非正式定义。"""

import importlib
import importlib.util
import json
import math
import random
import warnings
from dataclasses import FrozenInstanceError, MISSING, fields, replace

import numpy as np
import pytest
import torch
from torch import nn

from po_jepa.geometry import regular_grid
from po_jepa.model import POJEPA, ProjectionConfig
from po_jepa.spectral import GraphConfig, GraphSpectrum, modal_analysis


def api():
    """首次RED明确报告缺少C06模块，而非在收集时产生无关导入错误。"""
    assert importlib.util.find_spec("po_jepa.dynamics") is not None, "C06 dynamics 尚未实现"
    return importlib.import_module("po_jepa.dynamics")


def config(**changes):
    """十项显式配置；m=2是虚构单元值，未确定正式动作含义/单位/周期。"""
    values = dict(action_dim=2, dynamic_dim=64, condition_width=64, query_dim=32,
                  top_r=16, spectral_bias_width=32, spatial_bias_width=32,
                  action_hidden_width=64, initial_damping=.1, head_weight_std=1e-3)
    values.update(changes)
    return api().DynamicsConfig(**values)


def hand_action_residual_dynamics(device="cpu", m=2):
    """虚构全宽非零权重便于独立解析核验；不改变生产初始化或动作单位。"""
    dynamics = api().ModalDynamics(config(action_dim=m)).to(device)
    assert hasattr(dynamics, "action_residual"), "D08 action_residual 尚未实现"
    with torch.no_grad():
        hidden, output = dynamics.action_residual_net[0], dynamics.action_residual_net[2]
        hidden.weight.copy_(torch.sin(torch.arange(64*(64+m), device=device).float()*.037).reshape(64, 64+m)*.09)
        hidden.bias.copy_(torch.linspace(-.1, .15, 64, device=device))
        output.weight.copy_(torch.cos(torch.arange(4096, device=device).float()*.043).reshape(64, 64)*.03)
        output.bias.copy_(torch.linspace(-.13, .11, 64, device=device))
    return dynamics


def action_residual_reference(coefficients, action, dynamics, upstream=None):
    """独立NumPy double两层仿射/SiLU及六路解析反向，动作梯度沿K求和。"""
    c, a = numpy64(coefficients), numpy64(action)
    w1, b1, w2, b2 = (numpy64(p) for p in dynamics.action_residual_net.parameters())
    z = np.concatenate((c, np.repeat(a[:, None, :], c.shape[1], axis=1)), axis=-1)
    p = z@w1.T+b1
    sigmoid = 1/(1+np.exp(-p))
    h = p*sigmoid
    y = h@w2.T+b2
    if upstream is None:
        return y, {}
    g = numpy64(upstream)
    dp = (g@w2)*(sigmoid+p*sigmoid*(1-sigmoid))
    dz = dp@w1
    return y, dict(c=dz[..., :64], action=dz[..., 64:].sum(axis=1),
        hidden_weight=dp.reshape(-1, 64).T@z.reshape(-1, z.shape[-1]), hidden_bias=dp.sum(axis=(0, 1)),
        output_weight=g.reshape(-1, 64).T@h.reshape(-1, 64), output_bias=g.sum(axis=(0, 1)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_action_residual_matches_manual_mlp_and_broadcast_gradient(device):
    """完整[C,a]次序、两层SiLU及C/a/四参数解析导数；新图一次反向。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_action_residual_dynamics(device)
    c = torch.sin(torch.arange(384, device=device).float()*.071).reshape(2, 3, 64).requires_grad_()
    action = torch.tensor([[-.4, .7], [.9, -.2]], device=device, requires_grad=True)
    upstream = torch.cos(torch.arange(384, device=device).float()*.13).reshape_as(c)+.2
    expected, gradients = action_residual_reference(c, action, dynamics, upstream)
    actual = dynamics.action_residual(c, action)
    errors = dict(forward=close(actual, expected))
    assert actual.shape == c.shape and actual.dtype == torch.float32 and actual.device == c.device
    assert actual.min() < 0 < actual.max() and actual.requires_grad
    (actual*upstream).sum().backward()
    observed = dict(c=c.grad, action=action.grad,
        hidden_weight=dynamics.action_residual_net[0].weight.grad, hidden_bias=dynamics.action_residual_net[0].bias.grad,
        output_weight=dynamics.action_residual_net[2].weight.grad, output_bias=dynamics.action_residual_net[2].bias.grad)
    for name, gradient in observed.items():
        assert np.linalg.norm(gradients[name]) > .001
        errors[name] = close(gradient, gradients[name])
    assert all(p.grad is None for n, p in dynamics.named_parameters() if not n.startswith("action_residual_net."))
    print(json.dumps(dict(probe="D08-manual-MLP-and-broadcast-gradient", device=device, errors=errors)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_action_residual_real_projection_spectrum_chain_single_backward(device):
    """真实P_in→C02/3/4→C05→M_a；固定本次Psi核X/P_in/raw Z，未使用支路无梯度。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(1703)
    dynamics = hand_action_residual_dynamics(device)
    graph = GraphSpectrum(GraphConfig(16, 1703, .25, 1., 128, 1e-10, 1e-8)).to(device)
    model = POJEPA(ProjectionConfig(5, 64), graph, dynamics, object()).to(device)
    raw_z = torch.linspace(-.8, .9, 60, device=device).reshape(2, 6, 5).requires_grad_()
    x = model.project_in(raw_z)
    x.retain_grad()
    affinity = graph.affinity(x, regular_grid(2, 3))
    spectrum = graph.lowest_eigenpairs(graph.normalized_laplacian(affinity).laplacian,
                                      k=3, model_device=torch.device(device))
    c = modal_analysis(x, spectrum.basis)
    c.retain_grad()
    action = torch.tensor([[-.3, .6], [.8, -.4]], device=device, requires_grad=True)
    # 虚构诊断放大损失信号，避免把合法的小动作梯度误报为断图；容差不变。
    upstream = 32*(torch.sin(torch.arange(384, device=device).float()*.17).reshape_as(c)+.1)
    expected, gradients = action_residual_reference(c, action, dynamics, upstream)
    dx = numpy64(spectrum.basis)@gradients["c"]
    gradients.update(x=dx, raw_z=dx@numpy64(model.p_in.weight),
        p_in_weight=dx.reshape(-1, 64).T@numpy64(raw_z).reshape(-1, 5), p_in_bias=dx.sum(axis=(0, 1)))
    actual = dynamics.action_residual(c, action)
    errors = dict(forward=close(actual, expected))
    (actual*upstream).sum().backward()
    observed = dict(c=c.grad, action=action.grad, x=x.grad, raw_z=raw_z.grad,
        p_in_weight=model.p_in.weight.grad, p_in_bias=model.p_in.bias.grad,
        hidden_weight=dynamics.action_residual_net[0].weight.grad, hidden_bias=dynamics.action_residual_net[0].bias.grad,
        output_weight=dynamics.action_residual_net[2].weight.grad, output_bias=dynamics.action_residual_net[2].bias.grad)
    for name, gradient in observed.items():
        assert np.linalg.norm(gradients[name]) > .001
        errors[name] = close(gradient, gradients[name])
    assert not spectrum.eigenvalues.requires_grad and spectrum.eigenvalues.grad is None
    assert not spectrum.basis.requires_grad and spectrum.basis.grad is None
    assert model.p_out.weight.grad is None and model.p_out.bias.grad is None
    assert all(p.grad is None for n, p in dynamics.named_parameters() if not n.startswith("action_residual_net."))
    print(json.dumps(dict(probe="D08-real-C01-C05-chain", device=device, errors=errors)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("m", [1, 2, 9])
def test_action_residual_responses_mode_permutation_and_shared_sizes(device, m):
    """同一参数复用不同B/K；状态与带符号动作独立响应，模态排列同步，不缓存调用。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_action_residual_dynamics(device, m)
    parameter_ids = tuple(id(p) for p in dynamics.parameters())
    for batch, modes in ((1, 1), (2, 3), (3, 5)):
        c = torch.linspace(-.7, .8, batch*modes*64, device=device).reshape(batch, modes, 64)
        action = torch.linspace(-.6, .9, batch*m, device=device).reshape(batch, m)
        result = dynamics.action_residual(c, action)
        close(result, action_residual_reference(c, action, dynamics)[0])
        assert (dynamics.action_residual(c, action+.4)-result).abs().max() > 1e-4
        assert (dynamics.action_residual(c+.3, action)-result).abs().max() > 1e-4
        permutation = torch.arange(modes-1, -1, -1, device=device)
        close(dynamics.action_residual(c[:, permutation], action), numpy64(result[:, permutation]))
    assert tuple(id(p) for p in dynamics.parameters()) == parameter_ids
    with torch.no_grad():
        dynamics.action_residual_net[2].bias.add_(.25)
    close(dynamics.action_residual(c, action), numpy64(result)+.25)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_action_residual_zero_action_can_be_nonzero_and_zero_origin(device):
    """零偏置手算C0→SiLU→正负两通道；零动作非零合法，零C/a初始化输出为零。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = api().ModalDynamics(config()).to(device)
    c, action = torch.zeros(1, 1, 64, device=device), torch.zeros(1, 2, device=device)
    assert torch.count_nonzero(dynamics.action_residual(c, action)) == 0
    with torch.no_grad():
        for p in dynamics.action_residual_net.parameters():
            p.zero_()
        dynamics.action_residual_net[0].weight[0, 0] = 1.
        dynamics.action_residual_net[2].weight[0, 0] = 2.
        dynamics.action_residual_net[2].weight[1, 0] = -3.
    c[..., 0] = 2.
    expected = np.zeros((1, 1, 64))
    expected[..., :2] = np.array([4., -6.])/(1+math.exp(-2.))
    actual = dynamics.action_residual(c, action)
    close(actual, expected)
    assert actual[0, 0, 0] > 1. and actual[0, 0, 1] < -1.


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_action_residual_noncontiguous_inputs_and_forward_state(device):
    """非连续C/a合法；前向保持已有grad/值/版本、全部成员/模式、RNG及默认值。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_action_residual_dynamics(device).eval()
    c = torch.linspace(-.5, .7, 768, device=device).reshape(2, 3, 128)[..., ::2].requires_grad_()
    action = torch.tensor([[-.3, 9., .8, 9.], [.7, 9., -.6, 9.]], device=device)[:, ::2].requires_grad_()
    assert not c.is_contiguous() and not action.is_contiguous()
    inputs = (c, action)
    for x in inputs:
        x.grad = torch.full_like(x, .3)
    for p in dynamics.parameters():
        p.grad = torch.full_like(p, .2)
    saved = [(x.detach().clone(), x._version, id(x.grad), x.grad.clone(), x.grad._version) for x in inputs]
    parameters = {n: (id(p), p.detach().clone(), p._version, id(p.grad), p.grad.clone(), p.grad._version)
                  for n, p in dynamics.named_parameters()}
    members = {n: set(vars(layer)) for n, layer in dynamics.named_modules()}
    modes = {n: layer.training for n, layer in dynamics.named_modules()}
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    defaults = torch.get_default_dtype(), torch.get_default_device()
    expected = action_residual_reference(c, action, dynamics)[0]
    assert dynamics.action_residual(c, action).requires_grad
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            result = dynamics.action_residual(c, action)
            close(result, expected)
            assert not result.requires_grad and not torch.is_grad_enabled()
    for x, (value, version, grad_id, grad, grad_version) in zip(inputs, saved):
        assert torch.equal(x, value) and x._version == version
        assert id(x.grad) == grad_id and torch.equal(x.grad, grad) and x.grad._version == grad_version
    assert tuple(parameters) == tuple(dict(dynamics.named_parameters()))
    for n, p in dynamics.named_parameters():
        old_id, value, version, grad_id, grad, grad_version = parameters[n]
        assert id(p) == old_id and torch.equal(p, value) and p._version == version
        assert id(p.grad) == grad_id and torch.equal(p.grad, grad) and p.grad._version == grad_version
    assert members == {n: set(vars(layer)) for n, layer in dynamics.named_modules()}
    assert modes == {n: layer.training for n, layer in dynamics.named_modules()}
    assert list(dynamics.named_buffers()) == [] and torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())


@pytest.mark.parametrize("target", ["coefficients", "action"])
@pytest.mark.parametrize("kind", ["none", "list", "double", "half", "bfloat16", "int", "bool", "complex", "sparse", "nested", "meta", "nan", "inf"])
def test_action_residual_rejects_invalid_tensor_contract(target, kind):
    """模型输入拒绝非稠密/精度/设备/有限性错误，不转换、不广播batch。"""
    inputs = dict(coefficients=torch.zeros(2, 3, 64), action=torch.zeros(2, 2))
    conversions = dict(none=lambda x: None, list=lambda x: [], double=lambda x: x.double(),
        half=lambda x: x.half(), bfloat16=lambda x: x.bfloat16(), int=lambda x: x.long(),
        bool=lambda x: x.bool(), complex=lambda x: x.to(torch.complex64), sparse=lambda x: x.to_sparse(),
        nested=lambda x: torch.nested.nested_tensor([x[0], x[1]]), meta=lambda x: x.to("meta"),
        nan=lambda x: x*math.nan, inf=lambda x: x*math.inf)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")
        inputs[target] = conversions[kind](inputs[target])
    with pytest.raises(TypeError if kind in ("none", "list") else ValueError):
        api().ModalDynamics(config()).action_residual(**inputs)


@pytest.mark.parametrize("target,shape", [("coefficients", (64,)), ("coefficients", (2, 64)),
    ("coefficients", (2, 3, 1, 64)), ("coefficients", (0, 3, 64)), ("coefficients", (2, 0, 64)),
    ("coefficients", (2, 3, 63)), ("action", (2,)), ("action", (2, 1, 2)), ("action", (1, 2)),
    ("action", (0, 2)), ("action", (2, 1)), ("action", (2, 3))])
def test_action_residual_rejects_invalid_shapes(target, shape):
    """B/K正、64通道及配置m严格匹配，禁止样本轴静默广播。"""
    inputs = dict(coefficients=torch.zeros(2, 3, 64), action=torch.zeros(2, 2))
    inputs[target] = torch.zeros(shape)
    with pytest.raises(ValueError):
        api().ModalDynamics(config()).action_residual(**inputs)


@pytest.mark.parametrize("layer_index,parameter_name", [(0, "weight"), (0, "bias"), (2, "weight"), (2, "bias")])
@pytest.mark.parametrize("kind", ["double", "meta", "sparse", "nan", "inf"])
def test_action_residual_rejects_invalid_parameters(layer_index, parameter_name, kind):
    """逐个损坏四参数，保持公共有限float32与真实设备边界。"""
    dynamics = api().ModalDynamics(config())
    layer = dynamics.action_residual_net[layer_index]
    value = getattr(layer, parameter_name).detach()
    conversions = dict(double=lambda x: x.double(), meta=lambda x: x.to("meta"),
                       sparse=lambda x: x.to_sparse(), nan=lambda x: x*math.nan, inf=lambda x: x*math.inf)
    setattr(layer, parameter_name, nn.Parameter(conversions[kind](value)))
    with pytest.raises(ValueError):
        dynamics.action_residual(torch.zeros(2, 3, 64), torch.zeros(2, 2))


@pytest.mark.parametrize("target", ["coefficients", "action", "hidden", "output", "model"])
def test_action_residual_rejects_device_mismatch(target):
    """CUDA存在时逐输入/层与整个网络核对实际设备，不靠隐式复制掩盖错配。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = api().ModalDynamics(config())
    inputs = dict(coefficients=torch.zeros(2, 3, 64), action=torch.zeros(2, 2))
    if target in inputs:
        inputs[target] = inputs[target].cuda()
    elif target == "model":
        dynamics.cuda()
    else:
        dynamics.action_residual_net[0 if target == "hidden" else 2].cuda()
    with pytest.raises(ValueError):
        dynamics.action_residual(**inputs)


@pytest.mark.parametrize("stage", ["hidden", "output"])
def test_action_residual_rejects_finite_input_overflow(stage):
    """有限输入可使隐藏或输出仿射溢出；明确失败，不裁剪或返回非有限导数。"""
    dynamics = api().ModalDynamics(config())
    with torch.no_grad():
        for p in dynamics.action_residual_net.parameters():
            p.zero_()
        if stage == "hidden":
            dynamics.action_residual_net[0].weight.fill_(torch.finfo(torch.float32).max)
            dynamics.action_residual_net[2].weight.fill_(1.)
        else:
            dynamics.action_residual_net[0].bias.fill_(1.)
            dynamics.action_residual_net[2].weight.fill_(torch.finfo(torch.float32).max)
    with pytest.raises(ValueError):
        dynamics.action_residual(torch.ones(1, 1, 64), torch.ones(1, 2))


@pytest.mark.parametrize("m", [1, 2, 9])
def test_action_residual_complete_inventory_and_initialization(monkeypatch, m):
    """严格完整23项/四新参数/8320+64m增量；初始化按权重身份核对。"""
    calls, active = [], False
    xavier, normal = nn.init.xavier_uniform_, nn.init.normal_
    def record(function, tensor, **kwargs):
        """只记录最外层初始化调用，兼容PyTorch设备分派重入。"""
        nonlocal active
        if active:
            return function(tensor, **kwargs)
        calls.append((function, tensor, dict(kwargs)))
        active = True
        try:
            return function(tensor, **kwargs)
        finally:
            active = False
    monkeypatch.setattr(nn.init, "xavier_uniform_", lambda tensor, **kw: record(xavier, tensor, **kw))
    monkeypatch.setattr(nn.init, "normal_", lambda tensor, **kw: record(normal, tensor, **kw))
    dynamics = api().ModalDynamics(config(action_dim=m))
    names = ("condition_trunk.0.weight", "condition_trunk.0.bias", "gamma_head.weight", "gamma_head.bias",
        "omega_head.weight", "omega_head.bias", "b_head.weight", "b_head.bias", "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "spectral_bias_net.0.weight", "spectral_bias_net.0.bias", "spectral_bias_net.2.weight", "spectral_bias_net.2.bias",
        "spatial_bias_net.0.weight", "spatial_bias_net.0.bias", "spatial_bias_net.2.weight", "spatial_bias_net.2.bias",
        "action_residual_net.0.weight", "action_residual_net.0.bias", "action_residual_net.2.weight", "action_residual_net.2.bias")
    shapes = ((64, 65), (64,), (32, 64), (32,), (32, 64), (32,), (64*m, 64), (64*m,),
              (32, 64), (32, 64), (64, 64), (32, 3), (32,), (1, 32), (1,), (32, 16), (32,), (1, 32), (1,),
              (64, 64+m), (64,), (64, 64), (64,))
    assert tuple((n, tuple(p.shape)) for n, p in dynamics.named_parameters()) == tuple(zip(names, shapes))
    total = sum(p.numel() for p in dynamics.parameters())
    increment = sum(p.numel() for p in dynamics.action_residual_net.parameters())
    assert total == 25634+4224*m and increment == 8320+64*m
    assert tuple(dict(dynamics.named_children())) == ("condition_trunk", "gamma_head", "omega_head", "b_head", "q_proj", "k_proj", "v_proj", "spectral_bias_net", "spatial_bias_net", "action_residual_net")
    assert list(dynamics.named_buffers()) == []
    assert all(p.requires_grad and p.dtype == torch.float32 and p.device.type == "cpu" for p in dynamics.parameters())
    hidden, output = dynamics.action_residual_net[0], dynamics.action_residual_net[2]
    assert type(hidden) is nn.Linear and type(output) is nn.Linear
    assert len(dynamics.action_residual_net) == 3 and type(dynamics.action_residual_net[1]) is nn.SiLU
    assert [(fn, kw) for fn, tensor, kw in calls if tensor is hidden.weight] == [(xavier, dict(gain=1.))]
    assert [(fn, kw) for fn, tensor, kw in calls if tensor is output.weight] == [(normal, dict(mean=0., std=.001))]
    assert hidden.bias.count_nonzero() == 0 and output.bias.count_nonzero() == 0
    assert 0 < hidden.weight.abs().max() <= math.sqrt(6/(128+m)) and output.weight.count_nonzero() == 4096
    for name in ("context",):
        assert not hasattr(dynamics, name)
    print(json.dumps(dict(probe="D08-inventory-initialization", m=m, total=total, increment=increment,
        inventory=[dict(name=n, shape=list(p.shape)) for n, p in dynamics.named_parameters()])))


def sparse_example(device="cpu"):
    """虚构两batch四模态；含完全/边界并列、自身高低分及不同来源值。"""
    scores = torch.tensor([[[4., 4., -1., 2.], [-2., -3., 3., 1.],
                            [2., -1., -2., 2.], [0., 0., 0., 0.]],
                           [[.5, 2., -.4, 1.], [3., -2., 1., 0.],
                            [-4., .2, .9, .1], [1., 3., 3., -2.]]], device=device)
    value = torch.cos(torch.arange(512, device=device).reshape(2, 4, 64)*.13)
    return scores, value


def sparse_reference(scores, value, r, upstream=None):
    """独立Python(-分数,列号)排序、NumPy double前向和固定选择区解析反向。"""
    s, v = numpy64(scores), numpy64(value)
    batch, modes, _ = s.shape
    width = min(r, modes)
    index = np.empty((batch, modes, width), dtype=np.int64)
    selected, weights = np.empty(index.shape), np.empty(index.shape)
    aggregate = np.empty(v.shape)
    ds, dv = np.zeros(s.shape), np.zeros(v.shape)
    g = None if upstream is None else numpy64(upstream)
    for b in range(batch):
        for k in range(modes):
            js = sorted(range(modes), key=lambda j: (-s[b, k, j], j))[:width]
            index[b, k] = js
            selected[b, k] = s[b, k, js]
            exp = np.exp(selected[b, k]-max(selected[b, k]))
            weights[b, k] = exp/sum(exp)
            aggregate[b, k] = sum(a*v[b, j] for a, j in zip(weights[b, k], js))
            if g is not None:
                for a, j in zip(weights[b, k], js):
                    ds[b, k, j] = a*np.dot(g[b, k], v[b, j]-aggregate[b, k])
                    dv[b, j] += a*g[b, k]
    return index, selected, weights, aggregate, ds, dv


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_top_r_small_k_ties_self_candidates_and_sparse_gradients(device):
    """T-D6：原分数保图、稳定并列/自身候选、稀疏归一和S/V解析导数。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = api().ModalDynamics(config(top_r=2)).to(device)
    assert hasattr(dynamics, "sparse_attention"), "D07 sparse_attention 尚未实现"
    scores, value = (x.requires_grad_() for x in sparse_example(device))
    upstream = torch.sin(torch.arange(512, device=device).reshape(2, 4, 64)*.17)+.2
    expected = sparse_reference(scores, value, 2, upstream)
    result = dynamics.sparse_attention(scores, value)
    assert type(result) is api().SparseAttentionResult
    assert tuple(f.name for f in fields(result)) == ("neighbor_index", "selected_scores", "weights", "aggregate")
    with pytest.raises(FrozenInstanceError):
        result.weights = result.weights
    assert result.neighbor_index.dtype == torch.int64 and result.neighbor_index.device == scores.device
    assert not result.neighbor_index.requires_grad
    assert np.array_equal(result.neighbor_index.cpu().numpy(), expected[0])
    assert result.neighbor_index[0, 0].tolist() == [0, 1]
    assert result.neighbor_index[0, 1].tolist() == [2, 3]
    assert result.neighbor_index[0, 2].tolist() == [0, 3]
    assert result.neighbor_index[0, 3].tolist() == [0, 1]
    errors = {name: close(actual, ref) for name, actual, ref in zip(
        ("selected_scores", "weights", "aggregate"),
        (result.selected_scores, result.weights, result.aggregate), expected[1:4])}
    assert torch.equal(result.selected_scores, scores.gather(-1, result.neighbor_index))
    assert all(x.requires_grad and torch.isfinite(x).all() and x.dtype == torch.float32
               for x in (result.selected_scores, result.weights, result.aggregate))
    assert (result.weights >= 0).all()
    close(result.weights.sum(-1), np.ones((2, 4)))
    dense = torch.zeros_like(scores).scatter(-1, result.neighbor_index, result.weights)
    selected = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, result.neighbor_index, True)
    assert torch.count_nonzero(dense[~selected]) == 0
    (result.aggregate*upstream).sum().backward()
    errors.update(score_gradient=close(scores.grad, expected[4]), value_gradient=close(value.grad, expected[5]))
    assert torch.count_nonzero(scores.grad[~selected]) == 0
    assert scores.grad.abs().max() > .01 and value.grad.abs().max() > .01
    assert all(p.grad is None for p in dynamics.parameters())
    print(json.dumps(dict(probe="D07-top-r", device=device, errors=errors,
        score_gradient_max=float(scores.grad.abs().max()), value_gradient_max=float(value.grad.abs().max()))))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("modes,r", [(1, 1), (1, 16), (4, 1), (4, 4), (4, 9)])
def test_sparse_attention_small_k_full_softmax_and_single_slot_gradient(device, modes, r):
    """K=1/r=1/r=K/r>K；单槽权重恒1，聚合对分数的零梯度合法。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = api().ModalDynamics(config(top_r=r)).to(device)
    scores, value = (x[:, :modes, :modes] if i == 0 else x[:, :modes]
                     for i, x in enumerate(sparse_example(device)))
    scores, value = scores.detach().requires_grad_(), value.detach().requires_grad_()
    expected = sparse_reference(scores, value, r, torch.ones_like(value))
    result = dynamics.sparse_attention(scores, value)
    assert result.neighbor_index.shape == (2, modes, min(r, modes))
    assert np.array_equal(result.neighbor_index.cpu().numpy(), expected[0])
    for actual, ref in zip((result.selected_scores, result.weights, result.aggregate), expected[1:4]):
        close(actual, ref)
    if r >= modes:
        close(result.aggregate, numpy64(torch.softmax(scores, dim=-1)@value))
    result.aggregate.sum().backward()
    close(scores.grad, expected[4])
    close(value.grad, expected[5])
    if min(r, modes) == 1:
        assert torch.equal(result.weights, torch.ones_like(result.weights))
        assert torch.count_nonzero(scores.grad) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("single_row", [False, True])
def test_sparse_attention_shared_source_accumulates_and_unselected_row_is_zero(device, single_row):
    """多接收行选同一来源时dV求和；仅一行损失才能断言其未选来源为零。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = api().ModalDynamics(config(top_r=2)).to(device)
    scores = torch.tensor([[[-.4, .3, 1.2, -.9]]*4]*2, device=device, requires_grad=True)
    value = sparse_example(device)[1].requires_grad_()
    upstream = torch.cos(torch.arange(512, device=device).reshape(2, 4, 64)*.07)+.1
    if single_row:
        upstream[:, 1:] = 0
    expected = sparse_reference(scores, value, 2, upstream)
    result = dynamics.sparse_attention(scores, value)
    (upstream*result.aggregate).sum().backward()
    close(scores.grad, expected[4])
    error = close(value.grad, expected[5])
    assert torch.count_nonzero(value.grad[:, (0, 3)]) == 0
    if not single_row:
        row_only = numpy64(result.weights[:, 0, 0, None])*numpy64(upstream[:, 0])
        assert np.max(abs(expected[5][:, 2]-row_only)) > .1
    print(json.dumps(dict(probe="D07-source-gradient", device=device, single_row=single_row,
                          value_gradient_error=error)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_attention_non_tied_permutation_and_tie_index_rule(device):
    """非并列时映射新旧索引；并列只核对当前较小列，近似相等不归为并列。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = api().ModalDynamics(config(top_r=2)).to(device)
    scores = torch.tensor([[[.2, 3., -1., 2.], [4., -2., 1., .5],
                            [-3., 1., 2., 5.], [.8, -.5, 2., 1.]]], device=device)
    value = sparse_example(device)[1][:1]
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    original = dynamics.sparse_attention(scores, value)
    changed = dynamics.sparse_attention(scores[:, permutation][:, :, permutation], value[:, permutation])
    close(changed.aggregate, numpy64(original.aggregate[:, permutation]))
    assert torch.equal(permutation[changed.neighbor_index], original.neighbor_index[:, permutation])
    equal = dynamics.sparse_attention(torch.ones_like(scores), value[:, permutation])
    assert equal.neighbor_index.tolist() == [[[0, 1]]*4]
    near = torch.ones(1, 2, 2, device=device)
    near[..., 1] = torch.nextafter(near[..., 1], torch.full_like(near[..., 1], math.inf))
    one = api().ModalDynamics(config(top_r=1)).to(device).sparse_attention(near, value[:, :2])
    assert one.neighbor_index.tolist() == [[[1], [1]]]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_attention_large_shifted_scores_and_underflow(device):
    """适度大logit和逐行常数平移保权重；返回原分数，极小指数可自然下溢。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = api().ModalDynamics(config(top_r=2)).to(device)
    scores = torch.tensor([[[.25, 1., -.5, 2.], [3., -.25, 1., 0.],
                            [-2., 1.5, 4., 0.], [2., -1., 0., .75]]], device=device)
    value = sparse_example(device)[1][:1]
    original = dynamics.sparse_attention(scores, value)
    shifted = scores+torch.tensor([128., -128., 256., -256.], device=device)[None, :, None]
    for current in (shifted, scores+10000.):
        result = dynamics.sparse_attention(current, value)
        expected = sparse_reference(current, value, 2)
        assert torch.equal(result.neighbor_index, original.neighbor_index)
        close(result.weights, numpy64(original.weights))
        close(result.aggregate, expected[3])
        assert torch.equal(result.selected_scores, current.gather(-1, result.neighbor_index))
    extreme = torch.tensor([[[0., -200.], [-torch.finfo(torch.float32).max,
                                            torch.finfo(torch.float32).max]]], device=device)
    result = dynamics.sparse_attention(extreme, value[:, :2])
    assert result.weights.tolist() == [[[1., 0.], [1., 0.]]]
    assert torch.isfinite(result.aggregate).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_attention_noncontiguous_inputs_and_forward_state(device):
    """纯选择/收缩不读取网络；保持输入、已有grad、参数、成员、模式、RNG及默认值。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = api().ModalDynamics(config(top_r=2)).to(device).eval()
    inputs = tuple(x.transpose(1, 2).contiguous().transpose(1, 2).requires_grad_() for x in sparse_example(device))
    assert all(not x.is_contiguous() for x in inputs)
    for x in inputs:
        x.grad = torch.full_like(x, .3)
    for p in dynamics.parameters():
        p.grad = torch.full_like(p, .2)
    saved = [(x.detach().clone(), x.grad.clone()) for x in inputs]
    parameters = {name: (id(p), p.detach().clone(), p.grad.clone()) for name, p in dynamics.named_parameters()}
    members = {name: set(vars(m)) for name, m in dynamics.named_modules()}
    modes = {name: m.training for name, m in dynamics.named_modules()}
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    defaults = torch.get_default_dtype(), torch.get_default_device()
    expected = sparse_reference(*inputs, 2)[3]
    assert dynamics.sparse_attention(*inputs).aggregate.requires_grad
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            result = dynamics.sparse_attention(*inputs)
            close(result.aggregate, expected)
            assert not result.aggregate.requires_grad and not torch.is_grad_enabled()
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert all(torch.equal(x, value) and torch.equal(x.grad, grad) for x, (value, grad) in zip(inputs, saved))
    assert members == {name: set(vars(m)) for name, m in dynamics.named_modules()}
    assert modes == {name: m.training for name, m in dynamics.named_modules()}
    assert tuple(parameters) == tuple(dict(dynamics.named_parameters()))
    assert all(id(p) == parameters[n][0] and torch.equal(p, parameters[n][1]) and torch.equal(p.grad, parameters[n][2])
               for n, p in dynamics.named_parameters())
    names = ("condition_trunk.0.weight", "condition_trunk.0.bias", "gamma_head.weight", "gamma_head.bias",
        "omega_head.weight", "omega_head.bias", "b_head.weight", "b_head.bias", "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "spectral_bias_net.0.weight", "spectral_bias_net.0.bias", "spectral_bias_net.2.weight", "spectral_bias_net.2.bias",
        "spatial_bias_net.0.weight", "spatial_bias_net.0.bias", "spatial_bias_net.2.weight", "spatial_bias_net.2.bias")
    assert len(names) == 19 and sum(dict(dynamics.named_parameters())[n].numel() for n in names) == 17314+4160*2
    assert list(dynamics.named_buffers()) == [] and torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())
    for target in ("double", "meta"):
        network = dynamics.double() if target == "double" else dynamics.to("meta")
        close(network.sparse_attention(*inputs).aggregate, expected)


@pytest.mark.parametrize("target", ["scores", "value"])
@pytest.mark.parametrize("kind", ["none", "list", "double", "half", "bfloat16", "int", "bool", "complex", "sparse", "nested", "meta", "nan", "inf"])
def test_sparse_attention_rejects_invalid_tensor_contract(target, kind):
    """边界拒绝错误存储/精度/有限性，nested必须在读取shape前拒绝。"""
    scores, value = sparse_example()
    inputs = dict(scores=scores, value=value)
    conversions = dict(none=lambda x: None, list=lambda x: [], double=lambda x: x.double(),
        half=lambda x: x.half(), bfloat16=lambda x: x.bfloat16(), int=lambda x: x.long(),
        bool=lambda x: x.bool(), complex=lambda x: x.to(torch.complex64), sparse=lambda x: x.to_sparse(),
        nested=lambda x: torch.nested.nested_tensor([x[0], x[1]]), meta=lambda x: x.to("meta"),
        nan=lambda x: x*math.nan, inf=lambda x: x*math.inf)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")
        inputs[target] = conversions[kind](inputs[target])
    with pytest.raises(TypeError if kind in ("none", "list") else ValueError):
        api().ModalDynamics(config()).sparse_attention(inputs["scores"], inputs["value"])


@pytest.mark.parametrize("target,shape", [("scores", (4,)), ("scores", (2, 4)), ("scores", (2, 4, 4, 1)),
    ("scores", (0, 4, 4)), ("scores", (2, 0, 0)), ("scores", (2, 4, 3)),
    ("value", (2, 4)), ("value", (2, 4, 1, 64)), ("value", (1, 4, 64)),
    ("value", (2, 1, 64)), ("value", (2, 4, 63)), ("value", (0, 4, 64))])
def test_sparse_attention_rejects_wrong_shapes_without_broadcast(target, shape):
    """B/K正、分数方阵、值通道64，拒绝batch/模态的隐式广播。"""
    scores, value = sparse_example()
    inputs = dict(scores=scores, value=value)
    inputs[target] = torch.ones(shape)
    with pytest.raises(ValueError):
        api().ModalDynamics(config()).sparse_attention(inputs["scores"], inputs["value"])


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1., "2", None])
def test_sparse_attention_rejects_invalid_top_r_configuration(bad):
    """沿用配置严格正整数约定，不新增方法级覆盖或静默转换。"""
    with pytest.raises(ValueError):
        config(top_r=bad)


def test_sparse_attention_actual_cuda_device_boundary():
    """真实CPU/CUDA混用须拒绝；网络设备无关，两输入在同一CUDA即可。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = api().ModalDynamics(config())
    scores, value = sparse_example()
    for inputs in ((scores.cuda(), value), (scores, value.cuda())):
        with pytest.raises(ValueError):
            dynamics.sparse_attention(*inputs)
    close(dynamics.sparse_attention(scores.cuda(), value.cuda()).aggregate, sparse_reference(scores, value, 16)[3])


def spatial_example(device="cpu"):
    """A03-D08虚构三点原例，列是欧氏正交单位模态，坐标顺序为[x,y]。"""
    basis = torch.tensor([[1/math.sqrt(3), 1/math.sqrt(2)], [1/math.sqrt(3), 0.],
                          [1/math.sqrt(3), -1/math.sqrt(2)]], device=device).unsqueeze(0)
    return basis, torch.tensor([[0., 0.], [.5, 0.], [1., 0.]], device=device)


def hand_spatial_dynamics(device="cpu"):
    """虚构全宽非零网络，四项参数均有非退化解析梯度；不修改生产初始化。"""
    dynamics = api().ModalDynamics(config()).to(device)
    assert hasattr(dynamics, "spatial_descriptors"), "D06空间描述尚未实现"
    assert hasattr(dynamics, "spatial_bias"), "D06空间偏置尚未实现"
    with torch.no_grad():
        hidden, output = dynamics.spatial_bias_net[0], dynamics.spatial_bias_net[2]
        hidden.weight.copy_(torch.sin(torch.arange(512, device=device).reshape(32, 16)*.13)*.12)
        hidden.bias.copy_(torch.linspace(-.2, .15, 32, device=device))
        output.weight.copy_(torch.linspace(-.17, .24, 32, device=device).reshape(1, 32))
        output.bias.fill_(-.4)
    return dynamics


def spatial_reference(basis, coords):
    """逐b/k/j/i的NumPy double独立求和，验证五矩、绝对重叠及完整16维。"""
    psi, xy = numpy64(basis), numpy64(coords)
    batch, _, modes = psi.shape
    moments = np.zeros((batch, modes, 5))
    overlap = np.zeros((batch, modes, modes))
    features = np.empty((batch, modes, modes, 16))
    for b in range(batch):
        for k in range(modes):
            for i, (x, y) in enumerate(xy):
                moments[b, k] += psi[b, i, k]**2*np.array([x, y, x*x, y*y, x*y])
            for j in range(modes):
                overlap[b, k, j] = sum(abs(psi[b, i, k]*psi[b, i, j]) for i in range(len(xy)))
        for k in range(modes):
            for j in range(modes):
                features[b, k, j] = np.concatenate((moments[b, k], moments[b, j],
                    abs(moments[b, k]-moments[b, j]), [overlap[b, k, j]]))
    return moments, overlap, features


def spatial_mlp_reference(features, dynamics, upstream):
    """固定F的double仿射/SiLU链式参考，不借助autograd求四项参数导数。"""
    f, g = numpy64(features), numpy64(upstream)
    hidden, output = dynamics.spatial_bias_net[0], dynamics.spatial_bias_net[2]
    w1, b1, w2, b2 = [numpy64(p) for p in (hidden.weight, hidden.bias, output.weight, output.bias)]
    z = f@w1.T+b1
    sigmoid = 1/(1+np.exp(-z))
    h = z*sigmoid
    dz = (g[..., None]*w2)*(sigmoid+z*sigmoid*(1-sigmoid))
    gradients = (dz.reshape(-1, 32).T@f.reshape(-1, 16), dz.sum(axis=(0, 1, 2)),
                 g.reshape(1, -1)@h.reshape(-1, 32), np.array([g.sum()]))
    return (h@w2.T+b2)[..., 0], gradients


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_spectral_and_spatial_biases_respond_to_descriptors(device):
    """T-D5：原三点反例和新支撑的完整16维手算、受控非零网络响应。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_spatial_dynamics(device)
    basis, coords = spatial_example(device)
    with torch.no_grad():
        hidden, output = dynamics.spatial_bias_net[0], dynamics.spatial_bias_net[2]
        hidden.weight.zero_()
        hidden.bias.zero_()
        hidden.weight[0, 2], hidden.weight[1, 7], hidden.weight[2, 15] = 1., 1., 1.
        output.weight.zero_()
        output.weight[0, :3] = torch.tensor([.7, -.4, .3], device=device)
    descriptors = dynamics.spatial_descriptors(basis, coords)
    assert type(descriptors) is api().SpatialDescriptors
    assert tuple(field.name for field in fields(descriptors)) == ("moments", "overlap", "pair_features")
    with pytest.raises(FrozenInstanceError):
        descriptors.moments = descriptors.moments
    mu = np.array([[[.5, 0., 5/12, 0., 0.], [.5, 0., .5, 0., 0.]]])
    rho = np.array([[[1., 2/math.sqrt(6)], [2/math.sqrt(6), 1.]]])
    expected_pair = np.concatenate((mu[0, 0], mu[0, 1], abs(mu[0, 0]-mu[0, 1]), [2/math.sqrt(6)]))
    errors = dict(moments=close(descriptors.moments, mu), overlap=close(descriptors.overlap, rho),
                  pair12=close(descriptors.pair_features[0, 0, 1], expected_pair))
    for actual, expected in zip((descriptors.moments, descriptors.overlap, descriptors.pair_features), spatial_reference(basis, coords)):
        close(actual, expected)
        assert not actual.requires_grad and actual.grad is None
    close(basis.transpose(1, 2)@basis, np.eye(2)[None])
    assert abs(float((basis[0, :, 0]*basis[0, :, 1]).sum())) < 1e-5
    changed = basis.clone()
    changed[0, :, 1] = torch.tensor([1., -2., 1.], device=device)/math.sqrt(6)
    other = dynamics.spatial_descriptors(changed, coords)
    close(changed.transpose(1, 2)@changed, np.eye(2)[None])
    close(other.moments[0, 1], np.array([.5, 0., 1/3, 0., 0.]))
    close(other.overlap[0, 0, 1], np.array(4/math.sqrt(18)))
    for actual, expected in zip((other.moments, other.overlap, other.pair_features), spatial_reference(changed, coords)):
        close(actual, expected)
    bias = dynamics.spatial_bias(descriptors)
    expected, _ = spatial_mlp_reference(descriptors.pair_features, dynamics, torch.ones_like(bias))
    errors["scores"] = close(bias, expected)
    response = float((dynamics.spatial_bias(other)-bias).detach().abs().max())
    assert response > .01 and bias.min() < 0 and not torch.equal(bias, bias.transpose(1, 2))
    assert torch.count_nonzero(bias.diagonal(dim1=1, dim2=2)) == 2
    spectral_dynamics = hand_spectral_dynamics(device)
    eigenvalues = torch.tensor([[.1, .9]], device=device)
    spectral_response = float((spectral_dynamics.spectral_bias(eigenvalues+.4)-spectral_dynamics.spectral_bias(eigenvalues)).detach().abs().max())
    assert spectral_response > .001
    print(json.dumps(dict(probe="D06-three-point", device=device, errors=errors, rho12=float(descriptors.overlap[0, 0, 1]),
        old_second_moment=float(descriptors.moments[0, 1, 2]), new_second_moment=float(other.moments[0, 1, 2]),
        new_rho12=float(other.overlap[0, 0, 1]), spatial_response=response, spectral_response=spectral_response)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("real_spectrum", [False, True])
def test_spatial_bias_four_analytic_gradients_and_eig_boundary(device, real_spectrum):
    """新图一次backward；四项MLP解析导数，真实C04基底与coords不向P_in回传。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_spatial_dynamics(device)
    if real_spectrum:
        graph = GraphSpectrum(GraphConfig(16, 1703, .25, 1., 128, 1e-10, 1e-8)).to(device)
        model = POJEPA(ProjectionConfig(5, 64), graph, dynamics, object()).to(device)
        raw_z = torch.linspace(-.8, .9, 60, device=device).reshape(2, 6, 5).requires_grad_()
        x = model.project_in(raw_z)
        x.retain_grad()
        grid = regular_grid(2, 3)
        spectrum = graph.lowest_eigenpairs(graph.normalized_laplacian(graph.affinity(x, grid)).laplacian,
                                         k=3, model_device=torch.device(device))
        basis = spectrum.basis
        coords = grid.coords.to(device=device, dtype=torch.float32)
    else:
        basis, coords = spatial_example(device)
        basis = basis.expand(2, -1, -1)
    descriptors = dynamics.spatial_descriptors(basis, coords)
    for value, reference in zip((descriptors.moments, descriptors.overlap, descriptors.pair_features), spatial_reference(basis, coords)):
        close(value, reference)
        assert value.dtype == torch.float32 and value.device == basis.device and not value.requires_grad
    close(basis.transpose(1, 2)@basis, np.broadcast_to(np.eye(basis.shape[2]), (basis.shape[0], basis.shape[2], basis.shape[2])))
    for value in (descriptors.moments, descriptors.overlap):
        assert value.min() >= -1e-5 and value.max() <= 1+1e-5
    close(descriptors.overlap.diagonal(dim1=1, dim2=2), np.ones(basis.shape[::2]))
    upstream = torch.linspace(-.6, .9, basis.shape[0]*basis.shape[2]**2, device=device).reshape(basis.shape[0], basis.shape[2], -1)
    expected, gradients = spatial_mlp_reference(descriptors.pair_features, dynamics, upstream)
    actual = dynamics.spatial_bias(descriptors)
    errors = dict(scores=close(actual, expected))
    (actual*upstream).sum().backward()
    for (name, parameter), gradient in zip(dynamics.spatial_bias_net.named_parameters(), gradients):
        assert np.linalg.norm(gradient) > .001
        errors[name] = close(parameter.grad, gradient)
    assert all(p.grad is None for n, p in dynamics.named_parameters() if not n.startswith("spatial_bias_net."))
    assert not basis.requires_grad and basis.grad is None and not coords.requires_grad and coords.grad is None
    assert all(value.grad is None for value in (descriptors.moments, descriptors.overlap, descriptors.pair_features))
    if real_spectrum:
        assert x.grad is None and raw_z.grad is None and all(p.grad is None for p in model.p_in.parameters())
    print(json.dumps(dict(probe="D06-analytic-gradients", device=device, real_spectrum=real_spectrum, errors=errors)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_spatial_descriptors_symmetries_shared_grids_and_forward_state(device):
    """符号/模态/一致token排列、支撑、两个网格、单点及无前向副作用。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_spatial_dynamics(device).eval()
    for parameter in dynamics.parameters():
        parameter.grad = torch.full_like(parameter, .3)
    parameters = {n: (id(p), p.detach().clone(), p._version, id(p.grad), p.grad.clone(), p.grad._version)
                  for n, p in dynamics.named_parameters()}
    members = {n: set(vars(layer)) for n, layer in dynamics.named_modules()}
    modes = {n: layer.training for n, layer in dynamics.named_modules()}
    defaults = torch.get_default_dtype(), torch.get_default_device()
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    basis, coords = spatial_example(device)
    basis = torch.stack((basis, basis), dim=-1)[..., 0].expand(2, -1, -1)
    coords = torch.stack((coords, coords), dim=-1)[..., 0]
    assert not basis.is_contiguous() and not coords.is_contiguous()
    saved_inputs = [(value, value.clone(), value._version) for value in (basis, coords)]
    descriptors = dynamics.spatial_descriptors(basis, coords)
    saved_inputs += [(value, value.clone(), value._version) for value in (descriptors.moments, descriptors.overlap, descriptors.pair_features)]
    bias = dynamics.spatial_bias(descriptors)
    assert bias.shape == (2, 2, 2) and bias.requires_grad
    noncontiguous = api().SpatialDescriptors(*(torch.stack((value, value), dim=-1)[..., 0]
        for value in (descriptors.moments, descriptors.overlap, descriptors.pair_features)))
    assert all(not getattr(noncontiguous, name).is_contiguous() for name in ("moments", "overlap", "pair_features"))
    close(dynamics.spatial_bias(noncontiguous), numpy64(bias))
    flipped = dynamics.spatial_descriptors(basis*torch.tensor([-1., 1.], device=device), coords)
    token_order = torch.tensor([2, 0, 1], device=device)
    reordered = dynamics.spatial_descriptors(basis[:, token_order], coords[token_order])
    for transformed in (flipped, reordered):
        for name in ("moments", "overlap", "pair_features"):
            close(getattr(transformed, name), numpy64(getattr(descriptors, name)))
        close(dynamics.spatial_bias(transformed), numpy64(bias))
    permutation = torch.tensor([1, 0], device=device)
    permuted = dynamics.spatial_descriptors(basis[:, :, permutation], coords)
    close(permuted.moments, numpy64(descriptors.moments[:, permutation]))
    for name in ("overlap", "pair_features"):
        close(getattr(permuted, name), numpy64(getattr(descriptors, name)[:, permutation][:, :, permutation]))
    close(dynamics.spatial_bias(permuted), numpy64(bias[:, permutation][:, :, permutation]))
    shifted = dynamics.spatial_descriptors(basis[:, token_order], coords)
    assert not torch.equal(shifted.moments, descriptors.moments)
    assert float((dynamics.spatial_bias(shifted)-bias).detach().abs().max()) > .001
    for grid in (regular_grid(1, 1), regular_grid(2, 2)):
        xy = grid.coords.to(device=device, dtype=torch.float32)
        psi = torch.eye(len(xy), device=device).unsqueeze(0)
        result = dynamics.spatial_bias(dynamics.spatial_descriptors(psi, xy))
        assert result.shape == (1, len(xy), len(xy)) and result.requires_grad
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                inferred = dynamics.spatial_bias(dynamics.spatial_descriptors(psi, xy))
                assert not inferred.requires_grad and not torch.is_grad_enabled()
                close(inferred, numpy64(result))
    for n, p in dynamics.named_parameters():
        old_id, value, version, grad_id, grad, grad_version = parameters[n]
        assert id(p) == old_id and torch.equal(p, value)
        assert p._version == version
        assert id(p.grad) == grad_id and torch.equal(p.grad, grad) and p.grad._version == grad_version
    assert tuple(parameters) == tuple(dict(dynamics.named_parameters())) and list(dynamics.named_buffers()) == []
    assert members == {n: set(vars(layer)) for n, layer in dynamics.named_modules()}
    assert modes == {n: layer.training for n, layer in dynamics.named_modules()}
    for value, saved, version in saved_inputs:
        assert torch.equal(value, saved) and value._version == version and value.grad is None
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())


def test_spatial_bias_updates_after_parameter_change():
    """独立诊断对象改输出偏置后重算，不能复用旧的网络输出缓存。"""
    dynamics = hand_spatial_dynamics()
    descriptors = dynamics.spatial_descriptors(*spatial_example())
    before = dynamics.spatial_bias(descriptors)
    with torch.no_grad():
        dynamics.spatial_bias_net[2].bias.add_(.25)
    close(dynamics.spatial_bias(descriptors), numpy64(before)+.25)


@pytest.mark.parametrize("m", [1, 2, 9])
def test_spatial_bias_complete_inventory_and_initialization(monkeypatch, m):
    """D06固定十九项子集/577增量，身份归属初始化；旧十五项同外部种子逐值保留。"""
    calls, active = [], False
    xavier, normal = nn.init.xavier_uniform_, nn.init.normal_
    def record(function, tensor, **kwargs):
        """仅登记最外层真实初始化，兼容设备分派重入。"""
        nonlocal active
        if active:
            return function(tensor, **kwargs)
        calls.append((function, tensor, dict(kwargs)))
        active = True
        try:
            return function(tensor, **kwargs)
        finally:
            active = False
    monkeypatch.setattr(nn.init, "xavier_uniform_", lambda tensor, **kw: record(xavier, tensor, **kw))
    monkeypatch.setattr(nn.init, "normal_", lambda tensor, **kw: record(normal, tensor, **kw))
    torch.manual_seed(1703)
    dynamics = api().ModalDynamics(config(action_dim=m))
    assert hasattr(dynamics, "spatial_bias_net"), "D06空间偏置网络尚未实现"
    names = ("condition_trunk.0.weight", "condition_trunk.0.bias", "gamma_head.weight", "gamma_head.bias",
        "omega_head.weight", "omega_head.bias", "b_head.weight", "b_head.bias", "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "spectral_bias_net.0.weight", "spectral_bias_net.0.bias", "spectral_bias_net.2.weight", "spectral_bias_net.2.bias",
        "spatial_bias_net.0.weight", "spatial_bias_net.0.bias", "spatial_bias_net.2.weight", "spatial_bias_net.2.bias")
    shapes = ((64, 65), (64,), (32, 64), (32,), (32, 64), (32,), (64*m, 64), (64*m,),
              (32, 64), (32, 64), (64, 64), (32, 3), (32,), (1, 32), (1,), (32, 16), (32,), (1, 32), (1,))
    parameters = dict(dynamics.named_parameters())
    assert tuple((n, tuple(parameters[n].shape)) for n in names) == tuple(zip(names, shapes))
    total = sum(parameters[n].numel() for n in names)
    assert total == 17314+4160*m and sum(p.numel() for p in dynamics.spatial_bias_net.parameters()) == 577
    assert list(dynamics.named_buffers()) == []
    children = ("condition_trunk", "gamma_head", "omega_head", "b_head", "q_proj", "k_proj", "v_proj", "spectral_bias_net", "spatial_bias_net")
    assert tuple(n for n, layer in dynamics.named_children() if n in children) == children
    assert all(p.requires_grad and p.dtype == torch.float32 and p.device.type == "cpu" for p in dynamics.parameters())
    hidden, output = dynamics.spatial_bias_net[0], dynamics.spatial_bias_net[2]
    assert type(dynamics.spatial_bias_net[1]) is nn.SiLU and len(dynamics.spatial_bias_net) == 3
    assert [(fn, kw) for fn, tensor, kw in calls if tensor is hidden.weight] == [(xavier, dict(gain=1.))]
    assert [(fn, kw) for fn, tensor, kw in calls if tensor is output.weight] == [(normal, dict(mean=0., std=.001))]
    assert hidden.bias.count_nonzero() == 0 and output.bias.count_nonzero() == 0
    assert 0 < hidden.weight.abs().max() <= math.sqrt(6/48) and output.weight.count_nonzero() == 32
    torch.manual_seed(1703)
    trunk = nn.Linear(65, 64, bias=True, device="cpu", dtype=torch.float32)
    nn.init.xavier_uniform_(trunk.weight, gain=1.)
    nn.init.zeros_(trunk.bias)
    gamma = nn.Linear(64, 32, bias=True, device="cpu", dtype=torch.float32)
    omega = nn.Linear(64, 32, bias=True, device="cpu", dtype=torch.float32)
    nn.init.normal_(gamma.weight, mean=0., std=.001)
    nn.init.normal_(omega.weight, mean=0., std=.001)
    nn.init.constant_(gamma.bias, .1+math.log(-math.expm1(-.1)))
    nn.init.zeros_(omega.bias)
    b = nn.Linear(64, 64*m, bias=True, device="cpu", dtype=torch.float32)
    nn.init.normal_(b.weight, mean=0., std=.001)
    nn.init.zeros_(b.bias)
    projections = [nn.Linear(64, width, bias=False, device="cpu", dtype=torch.float32) for width in (32, 32, 64)]
    for layer in projections:
        nn.init.xavier_uniform_(layer.weight, gain=1.)
    spectral = nn.Sequential(nn.Linear(3, 32, device="cpu", dtype=torch.float32), nn.SiLU(), nn.Linear(32, 1, device="cpu", dtype=torch.float32))
    nn.init.xavier_uniform_(spectral[0].weight, gain=1.)
    nn.init.zeros_(spectral[0].bias)
    nn.init.normal_(spectral[2].weight, mean=0., std=.001)
    nn.init.zeros_(spectral[2].bias)
    expected = (trunk.weight, trunk.bias, gamma.weight, gamma.bias, omega.weight, omega.bias, b.weight, b.bias,
                *(layer.weight for layer in projections), *spectral.parameters())
    assert all(torch.equal(dict(dynamics.named_parameters())[name], ref) for name, ref in zip(names[:15], expected))
    print(json.dumps(dict(probe="D06-inventory-initialization", m=m, total=total, increment=577,
                         inventory=[dict(name=n, shape=list(parameters[n].shape)) for n in names])))


@pytest.mark.parametrize("field", ["basis", "coords"])
@pytest.mark.parametrize("kind", ["none", "rank", "empty", "mismatch", "double", "half", "bfloat16", "int", "bool", "complex", "sparse", "nested", "meta", "nan", "inf", "requires_grad"])
def test_spatial_descriptors_reject_invalid_inputs(field, kind):
    """两输入严格稠密有限float32/无梯度/正尺寸；错误坐标或精度不隐式修正。"""
    dynamics = hand_spatial_dynamics()
    basis, coords = spatial_example()
    value = basis if field == "basis" else coords
    conversions = dict(none=lambda t: None, rank=lambda t: t.unsqueeze(0), empty=lambda t: t[:0],
        mismatch=lambda t: torch.ones(1, 2, 3) if field == "basis" else torch.ones(3, 3),
        double=lambda t: t.double(), half=lambda t: t.half(), bfloat16=lambda t: t.bfloat16(),
        int=lambda t: t.long(), bool=lambda t: t.bool(), complex=lambda t: t.to(torch.complex64),
        sparse=lambda t: t.to_sparse(), nested=lambda t: torch.nested.nested_tensor([torch.ones(2), torch.ones(3)]),
        meta=lambda t: t.to("meta"), nan=lambda t: t*math.nan, inf=lambda t: t*math.inf,
        requires_grad=lambda t: t.requires_grad_())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = conversions[kind](value)
    with pytest.raises(TypeError if kind == "none" else ValueError):
        dynamics.spatial_descriptors(value if field == "basis" else basis, value if field == "coords" else coords)


@pytest.mark.parametrize("kind", ["empty_n", "empty_k", "negative", "above_one", "coordinate_count", "overflow"])
def test_spatial_descriptors_reject_coordinate_and_overflow_boundaries(kind):
    """N/K边界、coords行数与[0,1]及有限Psi平方产生的溢出明确失败。"""
    dynamics = hand_spatial_dynamics()
    basis, coords = spatial_example()
    if kind == "empty_n":
        basis, coords = basis[:, :0], coords[:0]
    elif kind == "empty_k":
        basis = basis[:, :, :0]
    elif kind == "coordinate_count":
        coords = coords[:2]
    elif kind == "overflow":
        basis.fill_(torch.finfo(torch.float32).max)
    else:
        coords[0, 0] = -.001 if kind == "negative" else 1.001
    with pytest.raises(ValueError):
        dynamics.spatial_descriptors(basis, coords)


@pytest.mark.parametrize("field", ["moments", "overlap", "pair_features"])
@pytest.mark.parametrize("kind", ["none", "shape", "double", "complex", "sparse", "nested", "meta", "nan", "inf", "requires_grad"])
def test_spatial_bias_rejects_invalid_descriptor_fields(field, kind):
    """三个字段均校验类型/形状/精度/真实设备/有限性/无梯度，即使只消费F。"""
    dynamics = hand_spatial_dynamics()
    descriptors = dynamics.spatial_descriptors(*spatial_example())
    values = {name: getattr(descriptors, name) for name in ("moments", "overlap", "pair_features")}
    conversions = dict(none=lambda t: None, shape=lambda t: t[..., :1], double=lambda t: t.double(),
        complex=lambda t: t.to(torch.complex64), sparse=lambda t: t.to_sparse(),
        nested=lambda t: torch.nested.nested_tensor([torch.ones(2), torch.ones(3)]), meta=lambda t: t.to("meta"),
        nan=lambda t: t*math.nan, inf=lambda t: t*math.inf, requires_grad=lambda t: t.requires_grad_())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values[field] = conversions[kind](values[field])
    with pytest.raises(TypeError if kind == "none" else ValueError):
        dynamics.spatial_bias(api().SpatialDescriptors(**values))


@pytest.mark.parametrize("kind", ["none", "dict", "empty_b", "empty_k"])
def test_spatial_bias_rejects_wrong_descriptor_or_empty_axes(kind):
    """拒绝伪类型及一致但空的B/K，不能只依赖linear能否运行。"""
    dynamics = hand_spatial_dynamics()
    descriptors = None if kind == "none" else {} if kind == "dict" else api().SpatialDescriptors(
        torch.empty(0 if kind == "empty_b" else 1, 0 if kind == "empty_k" else 2, 5),
        torch.empty(0 if kind == "empty_b" else 1, 0 if kind == "empty_k" else 2, 0 if kind == "empty_k" else 2),
        torch.empty(0 if kind == "empty_b" else 1, 0 if kind == "empty_k" else 2, 0 if kind == "empty_k" else 2, 16))
    with pytest.raises(TypeError if kind in ("none", "dict") else ValueError):
        dynamics.spatial_bias(descriptors)


@pytest.mark.parametrize("layer_index", [0, 2])
@pytest.mark.parametrize("parameter_name", ["weight", "bias"])
@pytest.mark.parametrize("kind", ["double", "complex", "meta", "sparse", "nan", "inf"])
def test_spatial_bias_rejects_invalid_parameters(layer_index, parameter_name, kind):
    """空间MLP两层四参数的精度、布局、真实设备与非有限边界。"""
    dynamics = hand_spatial_dynamics()
    descriptors = dynamics.spatial_descriptors(*spatial_example())
    layer = dynamics.spatial_bias_net[layer_index]
    value = getattr(layer, parameter_name).detach().clone()
    if kind in ("double", "complex", "meta", "sparse"):
        value = value.to_sparse() if kind == "sparse" else value.to("meta") if kind == "meta" else value.to(torch.float64 if kind == "double" else torch.complex64)
    else:
        value.flatten()[0] = math.nan if kind == "nan" else math.inf
    setattr(layer, parameter_name, nn.Parameter(value))
    with pytest.raises(ValueError):
        dynamics.spatial_bias(descriptors)


@pytest.mark.parametrize("layer_index", [0, 2])
def test_spatial_bias_rejects_output_overflow(layer_index):
    """有限但极大网络参数导致隐藏层或输出溢出时不静默接受。"""
    dynamics = hand_spatial_dynamics()
    descriptors = dynamics.spatial_descriptors(*spatial_example())
    with torch.no_grad():
        dynamics.spatial_bias_net[layer_index].weight.fill_(torch.finfo(torch.float32).max)
        if layer_index == 2:
            dynamics.spatial_bias_net[0].bias.fill_(3.)
    with pytest.raises(ValueError):
        dynamics.spatial_bias(descriptors)


def test_spatial_bias_actual_device_mismatch():
    """实际CPU/CUDA错配覆盖Psi、coords、三个字段和两层四参数。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_spatial_dynamics()
    basis, coords = spatial_example()
    for b, xy in ((basis.cuda(), coords), (basis, coords.cuda()), (basis.cuda(), coords.cuda())):
        with pytest.raises(ValueError):
            dynamics.spatial_descriptors(b, xy)
    descriptors = dynamics.spatial_descriptors(basis, coords)
    for field in ("moments", "overlap", "pair_features"):
        values = {name: getattr(descriptors, name) for name in ("moments", "overlap", "pair_features")}
        values[field] = values[field].cuda()
        with pytest.raises(ValueError):
            dynamics.spatial_bias(api().SpatialDescriptors(**values))
    for index in (0, 2):
        for name in ("weight", "bias"):
            dynamics = hand_spatial_dynamics()
            layer = dynamics.spatial_bias_net[index]
            setattr(layer, name, nn.Parameter(getattr(layer, name).detach().cuda()))
            with pytest.raises(ValueError):
                dynamics.spatial_bias(descriptors)


def hand_spectral_dynamics(device="cpu"):
    """D05虚构全宽非零权重与偏置，让四项参数都能收到解析学习信号。"""
    dynamics = api().ModalDynamics(config()).to(device)
    assert hasattr(dynamics, "spectral_bias"), "D05谱偏置尚未实现"
    with torch.no_grad():
        hidden, output = dynamics.spectral_bias_net[0], dynamics.spectral_bias_net[2]
        hidden.weight.copy_(torch.sin(torch.arange(96, device=device).reshape(32, 3)*.23)*.2)
        hidden.bias.copy_(torch.linspace(-.15, .2, 32, device=device))
        output.weight.copy_(torch.linspace(-.17, .24, 32, device=device).reshape(1, 32))
        output.bias.fill_(-.4)
    return dynamics


def spectral_bias_reference(eigenvalues, dynamics, upstream):
    """NumPy double逐batch/k/j建特征，独立算仿射、SiLU及四项解析梯度。"""
    values, g = numpy64(eigenvalues), numpy64(upstream)
    batch, modes = values.shape
    features = np.empty((batch, modes, modes, 3))
    for b in range(batch):
        for k in range(modes):
            for j in range(modes):
                features[b, k, j] = values[b, k], values[b, j], abs(values[b, k]-values[b, j])
    hidden, output = dynamics.spectral_bias_net[0], dynamics.spectral_bias_net[2]
    w1, b1, w2, b2 = [numpy64(p) for p in (hidden.weight, hidden.bias, output.weight, output.bias)]
    z = features@w1.T+b1
    sigmoid = 1/(1+np.exp(-z))
    h = z*sigmoid
    scores = (h@w2.T+b2)[..., 0]
    dz = (g[..., None]*w2)*(sigmoid+z*sigmoid*(1-sigmoid))
    gradients = (dz.reshape(-1, 32).T@features.reshape(-1, 3), dz.sum(axis=(0, 1, 2)),
                 g.reshape(1, -1)@h.reshape(-1, 32), np.array([g.sum()]))
    return features, scores, gradients


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_spectral_bias_uses_ordered_eigenvalue_pair_features(device):
    """T-D5谱部分：有序三特征、完整MLP手算，保留负数/非对称/非零对角。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_spectral_dynamics(device)
    eigenvalues = torch.tensor([[.2, 1.1, .2], [1.7, -.0000001, .6]], device=device)
    observed_features = []
    hook = dynamics.spectral_bias_net[0].register_forward_pre_hook(
        lambda layer, args: observed_features.append(args[0].detach().clone()))
    try:
        actual = dynamics.spectral_bias(eigenvalues)
    finally:
        hook.remove()
    features, expected, _ = spectral_bias_reference(eigenvalues, dynamics, torch.ones_like(actual))
    errors = dict(features=close(observed_features[0], features), scores=close(actual, expected))
    assert actual.shape == (2, 3, 3) and actual.dtype == torch.float32 and actual.device == eigenvalues.device
    assert actual.min() < 0 and not torch.equal(actual, actual.transpose(1, 2))
    assert torch.count_nonzero(actual.diagonal(dim1=1, dim2=2)) == 6
    assert torch.equal(actual[0, 0], actual[0, 2]) and torch.equal(actual[0, :, 0], actual[0, :, 2])
    c = torch.linspace(-.2, .5, 384, device=device).reshape(2, 3, 64)
    content = dynamics.content_terms(c).scores
    changed = dynamics.spectral_bias(eigenvalues+.4)
    response = float((changed-actual).detach().abs().max())
    assert response > .001 and torch.equal(content, dynamics.content_terms(c).scores)
    print(json.dumps(dict(probe="D05-ordered-pairs", device=device, errors=errors, lambda_response=response)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("real_spectrum", [False, True])
def test_spectral_bias_four_analytic_gradients_and_eig_boundary(device, real_spectrum):
    """新图仅一次backward：核四项解析梯度；真实C04谱不向P_in/X/raw Z反传。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_spectral_dynamics(device)
    if real_spectrum:
        graph = GraphSpectrum(GraphConfig(16, 1703, .25, 1., 128, 1e-10, 1e-8)).to(device)
        model = POJEPA(ProjectionConfig(5, 64), graph, dynamics, object()).to(device)
        raw_z = torch.linspace(-.8, .9, 60, device=device).reshape(2, 6, 5).requires_grad_()
        x = model.project_in(raw_z)
        x.retain_grad()
        spectrum = graph.lowest_eigenpairs(graph.normalized_laplacian(
            graph.affinity(x, regular_grid(2, 3))).laplacian, k=3, model_device=torch.device(device))
        eigenvalues = spectrum.eigenvalues
    else:
        eigenvalues = torch.tensor([[.2, 1.1, .2], [1.7, -.0000001, .6]], device=device)
    upstream = torch.linspace(-.6, .9, 18, device=device).reshape(2, 3, 3)
    _, expected, gradients = spectral_bias_reference(eigenvalues, dynamics, upstream)
    actual = dynamics.spectral_bias(eigenvalues)
    errors = dict(scores=close(actual, expected))
    (actual*upstream).sum().backward()
    for (name, parameter), gradient in zip(dynamics.spectral_bias_net.named_parameters(), gradients):
        assert np.linalg.norm(gradient) > .001
        errors[name] = close(parameter.grad, gradient)
    assert not eigenvalues.requires_grad and eigenvalues.grad is None
    assert all(p.grad is None for n, p in dynamics.named_parameters() if not n.startswith("spectral_bias_net."))
    if real_spectrum:
        assert x.grad is None and raw_z.grad is None
        assert all(p.grad is None for p in model.p_in.parameters())
        assert not spectrum.basis.requires_grad and spectrum.basis.grad is None
    print(json.dumps(dict(probe="D05-analytic-gradients", device=device, real_spectrum=real_spectrum, errors=errors)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_spectral_bias_permutation_recomputation_and_forward_state(device):
    """变B/K、双轴排列、非连续和推理；前向不改输入/梯度/成员/RNG/默认值。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_spectral_dynamics(device).eval()
    for parameter in dynamics.parameters():
        parameter.grad = torch.full_like(parameter, .3)
    parameters = {n: (id(p), p.detach().clone(), p._version, id(p.grad), p.grad.clone(), p.grad._version)
                  for n, p in dynamics.named_parameters()}
    members = {n: set(vars(layer)) for n, layer in dynamics.named_modules()}
    modes = {n: layer.training for n, layer in dynamics.named_modules()}
    defaults = torch.get_default_dtype(), torch.get_default_device()
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    for batch, modes_count in ((2, 3), (1, 1), (3, 5)):
        eigenvalues = torch.linspace(-1e-7, 1.7, batch*modes_count*2, device=device).reshape(batch, -1)[:, ::2]
        assert eigenvalues.numel() == 1 or not eigenvalues.is_contiguous()
        saved, version = eigenvalues.clone(), eigenvalues._version
        first = dynamics.spectral_bias(eigenvalues)
        assert first.shape == (batch, modes_count, modes_count) and first.requires_grad
        assert not torch.equal(first, dynamics.spectral_bias(eigenvalues+.4))
        assert torch.equal(first, dynamics.spectral_bias(eigenvalues))
        permutation = torch.arange(modes_count-1, -1, -1, device=device)
        close(dynamics.spectral_bias(eigenvalues[:, permutation]), numpy64(first[:, permutation][:, :, permutation]))
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                inferred = dynamics.spectral_bias(eigenvalues)
                assert not inferred.requires_grad and not torch.is_grad_enabled()
                close(inferred, numpy64(first))
        assert torch.equal(eigenvalues, saved) and eigenvalues._version == version and eigenvalues.grad is None
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert members == {n: set(vars(layer)) for n, layer in dynamics.named_modules()}
    assert modes == {n: layer.training for n, layer in dynamics.named_modules()}
    assert tuple(parameters) == tuple(dict(dynamics.named_parameters())) and list(dynamics.named_buffers()) == []
    for n, p in dynamics.named_parameters():
        old_id, value, version, grad_id, grad, grad_version = parameters[n]
        assert id(p) == old_id and torch.equal(p, value) and p._version == version
        assert id(p.grad) == grad_id and torch.equal(p.grad, grad) and p.grad._version == grad_version
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())


def test_spectral_bias_updates_after_parameter_change():
    """独立诊断对象主动改输出偏置，随后重算，不能用网络输出缓存。"""
    dynamics = hand_spectral_dynamics()
    eigenvalues = torch.tensor([[.2, .7]])
    before = dynamics.spectral_bias(eigenvalues)
    with torch.no_grad():
        dynamics.spectral_bias_net[2].bias.add_(.25)
    close(dynamics.spectral_bias(eigenvalues), numpy64(before)+.25)


@pytest.mark.parametrize("m", [1, 2, 9])
def test_spectral_bias_complete_inventory_and_initialization(monkeypatch, m):
    """D05固定十五项子集/161增量；按真实权重身份核对初始化调用。"""
    calls, active = [], False
    xavier, normal = nn.init.xavier_uniform_, nn.init.normal_
    def record(function, tensor, **kwargs):
        """保留设备分派重入保护，只登记最外层真实初始化。"""
        nonlocal active
        if active:
            return function(tensor, **kwargs)
        calls.append((function, tensor, dict(kwargs)))
        active = True
        try:
            return function(tensor, **kwargs)
        finally:
            active = False
    monkeypatch.setattr(nn.init, "xavier_uniform_", lambda tensor, **kw: record(xavier, tensor, **kw))
    monkeypatch.setattr(nn.init, "normal_", lambda tensor, **kw: record(normal, tensor, **kw))
    dynamics = api().ModalDynamics(config(action_dim=m))
    names = ("condition_trunk.0.weight", "condition_trunk.0.bias", "gamma_head.weight", "gamma_head.bias",
        "omega_head.weight", "omega_head.bias", "b_head.weight", "b_head.bias", "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "spectral_bias_net.0.weight", "spectral_bias_net.0.bias", "spectral_bias_net.2.weight", "spectral_bias_net.2.bias")
    shapes = ((64, 65), (64,), (32, 64), (32,), (32, 64), (32,), (64*m, 64), (64*m,),
              (32, 64), (32, 64), (64, 64), (32, 3), (32,), (1, 32), (1,))
    parameters = dict(dynamics.named_parameters())
    assert tuple((n, tuple(parameters[n].shape)) for n in names) == tuple(zip(names, shapes))
    total = sum(parameters[n].numel() for n in names)
    assert total == 16737+4160*m and sum(p.numel() for p in dynamics.spectral_bias_net.parameters()) == 161
    assert list(dynamics.named_buffers()) == []
    children = ("condition_trunk", "gamma_head", "omega_head", "b_head", "q_proj", "k_proj", "v_proj", "spectral_bias_net")
    assert tuple(n for n, _ in dynamics.named_children() if n in children) == children
    assert all(p.requires_grad and p.dtype == torch.float32 and p.device.type == "cpu" for p in dynamics.parameters())
    hidden, output = dynamics.spectral_bias_net[0], dynamics.spectral_bias_net[2]
    assert type(dynamics.spectral_bias_net[1]) is nn.SiLU and len(dynamics.spectral_bias_net) == 3
    hidden_calls = [(fn, kw) for fn, tensor, kw in calls if tensor is hidden.weight]
    output_calls = [(fn, kw) for fn, tensor, kw in calls if tensor is output.weight]
    assert hidden_calls == [(xavier, dict(gain=1.))]
    assert output_calls == [(normal, dict(mean=0., std=dynamics.config.head_weight_std))]
    assert hidden.bias.count_nonzero() == 0 and output.bias.count_nonzero() == 0
    assert 0 < hidden.weight.abs().max() <= math.sqrt(6/35) and output.weight.count_nonzero() == 32
    print(json.dumps(dict(probe="D05-inventory-initialization", m=m, total=total, increment=161,
        inventory=[dict(name=n, shape=list(parameters[n].shape)) for n in names],
        hidden_initialization="Xavier uniform gain=1", output_initialization=dict(mean=0., std=dynamics.config.head_weight_std))))


@pytest.mark.parametrize("kind", ["none", "list", "rank1", "rank3", "empty_b", "empty_k", "double", "half", "bfloat16", "int", "bool", "complex", "sparse", "nested", "meta", "nan", "inf", "requires_grad"])
def test_spectral_bias_rejects_invalid_input(kind):
    """只接收正B/K、稠密有限float32、无梯度的原始谱，不隐式转换或detach。"""
    eigenvalues = torch.ones(2, 3)
    conversions = dict(none=lambda t: None, list=lambda t: t.tolist(), rank1=lambda t: t[0], rank3=lambda t: t.unsqueeze(-1),
        empty_b=lambda t: t[:0], empty_k=lambda t: t[:, :0], double=lambda t: t.double(), half=lambda t: t.half(),
        bfloat16=lambda t: t.bfloat16(), int=lambda t: t.long(), bool=lambda t: t.bool(), complex=lambda t: t.to(torch.complex64),
        sparse=lambda t: t.to_sparse(), nested=lambda t: torch.nested.nested_tensor([t[0], t[1, :2]]),
        meta=lambda t: t.to("meta"), nan=lambda t: t*math.nan, inf=lambda t: t*math.inf, requires_grad=lambda t: t.requires_grad_())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eigenvalues = conversions[kind](eigenvalues)
    with pytest.raises(TypeError if kind in ("none", "list") else ValueError):
        hand_spectral_dynamics().spectral_bias(eigenvalues)


@pytest.mark.parametrize("layer_index", [0, 2])
@pytest.mark.parametrize("parameter_name", ["weight", "bias"])
@pytest.mark.parametrize("kind", ["double", "complex", "meta", "sparse", "nan", "inf"])
def test_spectral_bias_rejects_invalid_parameters(layer_index, parameter_name, kind):
    """逐一核对两层权重/偏置的真实设备、float32及有限性；不静默修补。"""
    dynamics = hand_spectral_dynamics()
    layer = dynamics.spectral_bias_net[layer_index]
    value = getattr(layer, parameter_name).detach().clone()
    if kind in ("double", "complex", "meta", "sparse"):
        value = value.to_sparse() if kind == "sparse" else value.to("meta") if kind == "meta" else value.to(torch.float64 if kind == "double" else torch.complex64)
    else:
        value.flatten()[0] = math.nan if kind == "nan" else math.inf
    setattr(layer, parameter_name, nn.Parameter(value))
    with pytest.raises(ValueError):
        dynamics.spectral_bias(torch.ones(2, 3))


@pytest.mark.parametrize("layer_index", [0, 2])
def test_spectral_bias_rejects_output_overflow(layer_index):
    """即使输入和参数有限，仿射/输出溢出也须明确失败。"""
    dynamics = hand_spectral_dynamics()
    with torch.no_grad():
        dynamics.spectral_bias_net[layer_index].weight.fill_(torch.finfo(torch.float32).max)
    with pytest.raises(ValueError):
        dynamics.spectral_bias(torch.full((2, 3), 3.))


def test_spectral_bias_actual_device_mismatch():
    """实际CUDA检查输入及两层每项参数错配；meta测试不替代设备实测。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with pytest.raises(ValueError):
        hand_spectral_dynamics().spectral_bias(torch.ones(1, 1, device="cuda"))
    for index in (0, 2):
        for name in ("weight", "bias"):
            dynamics = hand_spectral_dynamics()
            layer = dynamics.spectral_bias_net[index]
            setattr(layer, name, nn.Parameter(getattr(layer, name).detach().to("cuda")))
            with pytest.raises(ValueError):
                dynamics.spectral_bias(torch.ones(1, 1))


def hand_content_dynamics(device="cpu"):
    """D04虚构全宽非均匀权重：三投影独立，防止漏通道和错误共享被掩盖。"""
    dynamics = api().ModalDynamics(config()).to(device)
    assert hasattr(dynamics, "content_terms"), "D04当前C内容分数尚未实现"
    with torch.no_grad():
        for index, layer in enumerate((dynamics.q_proj, dynamics.k_proj, dynamics.v_proj)):
            values = torch.arange(layer.weight.numel(), device=device).float().reshape_as(layer.weight)
            layer.weight.copy_(torch.sin(values*(.17+.03*index)+index)*.025+.001*index)
    return dynamics


def content_reference(coefficients, dynamics, score_upstream, value_upstream):
    """NumPy double独立投影和一次标量损失的Q/K/V全路径解析导数。"""
    c, g, t = numpy64(coefficients), numpy64(score_upstream), numpy64(value_upstream)
    wq, wk, wv = [numpy64(layer.weight) for layer in (dynamics.q_proj, dynamics.k_proj, dynamics.v_proj)]
    q, key, value = c@wq.T, c@wk.T, c@wv.T
    scores = q@key.transpose(0, 2, 1)/math.sqrt(32)
    dq, dk = g@key/math.sqrt(32), g.transpose(0, 2, 1)@q/math.sqrt(32)
    gradients = dict(c=dq@wq+dk@wk+t@wv,
        q_weight=dq.reshape(-1, 32).T@c.reshape(-1, 64),
        k_weight=dk.reshape(-1, 32).T@c.reshape(-1, 64),
        v_weight=t.reshape(-1, 64).T@c.reshape(-1, 64))
    return (q, key, value, scores), gradients


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_content_score_uses_query_width_scaling(device):
    """T-D4：K=3、两batch独立手算；负数、非零对角和非对称均须原样保留。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_content_dynamics(device)
    with torch.no_grad():
        for layer in (dynamics.q_proj, dynamics.k_proj, dynamics.v_proj):
            layer.weight.zero_()
        dynamics.q_proj.weight[:2, :2].copy_(torch.eye(2, device=device))
        dynamics.k_proj.weight[:2, :2].copy_(torch.tensor([[1., 1.], [-1., 0.]], device=device))
        dynamics.v_proj.weight[0, :2] = torch.tensor([2., -3.], device=device)
        dynamics.v_proj.weight[7, :2] = torch.tensor([-.5, 4.], device=device)
    c = torch.zeros(2, 3, 64, device=device)
    c[..., :2] = torch.tensor([[[1., 2.], [3., -1.], [-2., .5]],
                               [[-.5, 3.], [2., 4.], [1., -3.]]], device=device)
    q, key, value = np.zeros((2, 3, 32)), np.zeros((2, 3, 32)), np.zeros((2, 3, 64))
    cn = numpy64(c)
    q[..., :2] = cn[..., :2]
    key[..., 0], key[..., 1] = cn[..., 0]+cn[..., 1], -cn[..., 0]
    value[..., 0], value[..., 7] = 2*cn[..., 0]-3*cn[..., 1], -.5*cn[..., 0]+4*cn[..., 1]
    scores = np.zeros((2, 3, 3))
    for b in range(2):
        for k in range(3):
            for j in range(3):
                scores[b, k, j] = sum(q[b, k, d]*key[b, j, d] for d in range(32))/math.sqrt(32)
    actual = dynamics.content_terms(c)
    assert type(actual) is api().ContentTerms
    assert tuple(f.name for f in fields(actual)) == ("query", "key", "value", "scores")
    with pytest.raises(FrozenInstanceError):
        actual.scores = c
    for field, expected in zip(fields(actual), (q, key, value, scores)):
        output = getattr(actual, field.name)
        assert output.shape == expected.shape and output.dtype == torch.float32 and output.device == c.device
        close(output, expected)
    close(actual.scores[0, :2, :2], np.array([[1., -4.], [10., 9.]])/math.sqrt(32))
    assert actual.scores.min() < 0 and not torch.equal(actual.scores, actual.scores.transpose(1, 2))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_content_full_width_forward_and_all_analytic_gradients(device):
    """全宽独立前向和Q/K/V/C解析反向；一张新图只做一次backward。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_content_dynamics(device)
    c = (torch.sin(torch.arange(384, device=device).float()*.071)+.1).reshape(2, 3, 64).requires_grad_()
    g = torch.linspace(-.8, 1.1, 18, device=device).reshape(2, 3, 3)
    t = torch.cos(torch.arange(384, device=device).float()*.13).reshape_as(c)
    expected, gradients = content_reference(c, dynamics, g, t)
    actual = dynamics.content_terms(c)
    errors = {f.name: close(getattr(actual, f.name), ref) for f, ref in zip(fields(actual), expected)}
    ((actual.scores*g).sum()+(actual.value*t).sum()).backward()
    observed = dict(c=c.grad, q_weight=dynamics.q_proj.weight.grad,
                    k_weight=dynamics.k_proj.weight.grad, v_weight=dynamics.v_proj.weight.grad)
    for name, gradient in observed.items():
        assert np.linalg.norm(gradients[name]) > .01
        errors[name] = close(gradient, gradients[name])
    assert all(p.grad is None for n, p in dynamics.named_parameters() if not n.startswith(("q_proj.", "k_proj.", "v_proj.")))
    print(json.dumps(dict(probe="D04-full-width-analytic", device=device, errors=errors)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_content_real_projection_spectrum_chain_single_backward(device):
    """真实P_in→C02/3/4→C05→内容；固定本次Psi解析核X、P_in与raw Z。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_content_dynamics(device)
    graph = GraphSpectrum(GraphConfig(16, 1703, .25, 1., 128, 1e-10, 1e-8)).to(device)
    model = POJEPA(ProjectionConfig(5, 64), graph, dynamics, object()).to(device)
    raw_z = torch.linspace(-.8, .9, 60, device=device).reshape(2, 6, 5).requires_grad_()
    x = model.project_in(raw_z)
    x.retain_grad()
    affinity = graph.affinity(x, regular_grid(2, 3))
    spectrum = graph.lowest_eigenpairs(graph.normalized_laplacian(affinity).laplacian,
                                      k=3, model_device=torch.device(device))
    c = modal_analysis(x, spectrum.basis)
    c.retain_grad()
    g = torch.linspace(-.6, .9, 18, device=device).reshape(2, 3, 3)
    t = torch.sin(torch.arange(384, device=device).float()*.17).reshape_as(c)
    expected, gradients = content_reference(c, dynamics, g, t)
    dx = numpy64(spectrum.basis)@gradients["c"]
    gradients.update(x=dx, raw_z=dx@numpy64(model.p_in.weight),
        p_in_weight=dx.reshape(-1, 64).T@numpy64(raw_z).reshape(-1, 5), p_in_bias=dx.sum(axis=(0, 1)))
    actual = dynamics.content_terms(c)
    errors = {f.name: close(getattr(actual, f.name), ref) for f, ref in zip(fields(actual), expected)}
    ((actual.scores*g).sum()+(actual.value*t).sum()).backward()
    observed = dict(c=c.grad, x=x.grad, raw_z=raw_z.grad, p_in_weight=model.p_in.weight.grad,
        p_in_bias=model.p_in.bias.grad, q_weight=dynamics.q_proj.weight.grad,
        k_weight=dynamics.k_proj.weight.grad, v_weight=dynamics.v_proj.weight.grad)
    for name, gradient in observed.items():
        assert np.linalg.norm(gradients[name]) > .001
        errors[name] = close(gradient, gradients[name])
    assert not spectrum.eigenvalues.requires_grad and spectrum.eigenvalues.grad is None
    assert not spectrum.basis.requires_grad and spectrum.basis.grad is None
    assert model.p_out.weight.grad is None and model.p_out.bias.grad is None
    assert all(p.grad is None for n, p in dynamics.named_parameters() if not n.startswith(("q_proj.", "k_proj.", "v_proj.")))
    print(json.dumps(dict(probe="D04-real-C01-C05-chain", device=device, errors=errors)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_content_permutation_current_state_and_no_forward_side_effects(device):
    """双轴排列/C1→C2→C1、变B/K、非连续及推理；整实例状态和既有梯度不变。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_content_dynamics(device).eval()
    for p in dynamics.parameters():
        p.grad = torch.full_like(p, .3)
    parameters = {n: (id(p), p.detach().clone(), p._version, id(p.grad), p.grad.clone(), p.grad._version)
                  for n, p in dynamics.named_parameters()}
    members = {n: set(vars(layer)) for n, layer in dynamics.named_modules()}
    modes = {n: layer.training for n, layer in dynamics.named_modules()}
    defaults = torch.get_default_dtype(), torch.get_default_device()
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    for batch, k in ((2, 3), (1, 1), (3, 5)):
        c = torch.linspace(-.7, .9, batch*64*k, device=device).reshape(batch, 64, k).transpose(1, 2).requires_grad_()
        assert k == 1 or not c.is_contiguous()
        saved, version = c.detach().clone(), c._version
        first = dynamics.content_terms(c)
        other = dynamics.content_terms(c*.7+.1)
        again = dynamics.content_terms(c)
        assert not torch.equal(first.value, other.value)
        permutation = torch.arange(k-1, -1, -1, device=device)
        permuted = dynamics.content_terms(c[:, permutation])
        for f in fields(first):
            result = getattr(first, f.name)
            assert torch.equal(result, getattr(again, f.name)) and result.requires_grad
            expected = result[:, permutation][:, :, permutation] if f.name == "scores" else result[:, permutation]
            close(getattr(permuted, f.name), numpy64(expected))
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                inferred = dynamics.content_terms(c)
                assert not torch.is_grad_enabled()
                assert all(not getattr(inferred, f.name).requires_grad for f in fields(inferred))
        assert torch.equal(c, saved) and c._version == version and c.grad is None
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert members == {n: set(vars(layer)) for n, layer in dynamics.named_modules()}
    assert modes == {n: layer.training for n, layer in dynamics.named_modules()}
    assert tuple(parameters) == tuple(dict(dynamics.named_parameters())) and list(dynamics.named_buffers()) == []
    for n, p in dynamics.named_parameters():
        old_id, value, version, grad_id, grad, grad_version = parameters[n]
        assert id(p) == old_id and torch.equal(p, value) and p._version == version
        assert id(p.grad) == grad_id and torch.equal(p.grad, grad) and p.grad._version == grad_version
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())


@pytest.mark.parametrize("m", [1, 2, 9])
def test_content_complete_inventory_and_exact_old_initialization(m):
    """D04固定十一项子集；同种子独立重放旧八项和三项Xavier初始化。"""
    torch.manual_seed(1703)
    dynamics = api().ModalDynamics(config(action_dim=m))
    names = ("condition_trunk.0.weight", "condition_trunk.0.bias", "gamma_head.weight", "gamma_head.bias",
        "omega_head.weight", "omega_head.bias", "b_head.weight", "b_head.bias",
        "q_proj.weight", "k_proj.weight", "v_proj.weight")
    shapes = ((64, 65), (64,), (32, 64), (32,), (32, 64), (32,), (64*m, 64), (64*m,),
              (32, 64), (32, 64), (64, 64))
    parameters = dict(dynamics.named_parameters())
    old_parameters = tuple(parameters[name] for name in names)
    assert tuple(tuple(p.shape) for p in old_parameters) == shapes
    assert sum(p.numel() for p in old_parameters) == 16576+4160*m
    assert list(dynamics.named_buffers()) == []
    assert all(p.requires_grad and p.dtype == torch.float32 and p.device.type == "cpu" for p in dynamics.parameters())
    torch.manual_seed(1703)
    trunk = nn.Linear(65, 64, bias=True, device="cpu", dtype=torch.float32)
    nn.init.xavier_uniform_(trunk.weight, gain=1.)
    nn.init.zeros_(trunk.bias)
    gamma = nn.Linear(64, 32, bias=True, device="cpu", dtype=torch.float32)
    omega = nn.Linear(64, 32, bias=True, device="cpu", dtype=torch.float32)
    nn.init.normal_(gamma.weight, mean=0., std=.001)
    nn.init.normal_(omega.weight, mean=0., std=.001)
    nn.init.constant_(gamma.bias, .1+math.log(-math.expm1(-.1)))
    nn.init.zeros_(omega.bias)
    b = nn.Linear(64, 64*m, bias=True, device="cpu", dtype=torch.float32)
    nn.init.normal_(b.weight, mean=0., std=.001)
    nn.init.zeros_(b.bias)
    projections = [nn.Linear(64, width, bias=False, device="cpu", dtype=torch.float32) for width in (32, 32, 64)]
    for layer in projections:
        nn.init.xavier_uniform_(layer.weight, gain=1.)
    expected = (trunk.weight, trunk.bias, gamma.weight, gamma.bias, omega.weight, omega.bias, b.weight, b.bias,
                *(layer.weight for layer in projections))
    assert all(torch.equal(actual, ref) for actual, ref in zip(old_parameters, expected))
    for layer in (dynamics.q_proj, dynamics.k_proj, dynamics.v_proj):
        assert type(layer) is nn.Linear and layer.bias is None
        assert 0 < layer.weight.abs().max() <= math.sqrt(6/(64+layer.out_features))


@pytest.mark.parametrize("bad", [None, [], 3, "coefficients"])
def test_content_rejects_non_tensor(bad):
    """公开边界拒绝非Tensor，不尝试转换。"""
    with pytest.raises(TypeError):
        hand_content_dynamics().content_terms(bad)


@pytest.mark.parametrize("shape", [(64,), (2, 64), (2, 3, 1, 64), (0, 3, 64), (2, 0, 64), (2, 3, 63)])
def test_content_rejects_invalid_rank_empty_axes_and_width(shape):
    """B/K必须为正，通道64且恰有三个轴。"""
    with pytest.raises(ValueError):
        hand_content_dynamics().content_terms(torch.ones(shape))


@pytest.mark.parametrize("kind", ["double", "half", "bfloat16", "int", "bool", "complex", "sparse", "nested", "meta", "nan", "inf"])
def test_content_rejects_invalid_tensor_storage_dtype_and_finiteness(kind):
    """只接收真实稠密有限float32，nested检查先于shape读取。"""
    c = torch.ones(2, 3, 64)
    if kind in ("double", "half", "bfloat16", "int", "bool", "complex"):
        c = c.to({"double": torch.float64, "half": torch.float16, "bfloat16": torch.bfloat16,
                  "int": torch.int64, "bool": torch.bool, "complex": torch.complex64}[kind])
    elif kind == "sparse":
        c = c.to_sparse()
    elif kind == "nested":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = torch.nested.nested_tensor([torch.ones(3, 64), torch.ones(2, 64)])
    elif kind == "meta":
        c = c.to("meta")
    else:
        c[0, 0, 0] = math.nan if kind == "nan" else math.inf
    with pytest.raises(ValueError):
        hand_content_dynamics().content_terms(c)


@pytest.mark.parametrize("layer_name", ["q_proj", "k_proj", "v_proj"])
@pytest.mark.parametrize("kind", ["double", "complex", "meta", "sparse", "shape", "nan", "inf"])
def test_content_rejects_corrupted_projection_weight(layer_name, kind):
    """三权重逐一检查shape、精度、真实设备、稠密性和有限性，不能悄悄转换。"""
    dynamics = hand_content_dynamics()
    layer = getattr(dynamics, layer_name)
    weight = layer.weight.detach().clone()
    if kind in ("double", "complex", "meta", "sparse"):
        weight = weight.to_sparse() if kind == "sparse" else weight.to("meta") if kind == "meta" else weight.to(torch.float64 if kind == "double" else torch.complex64)
    elif kind == "shape":
        weight = weight[:, :-1]
    else:
        weight[0, 0] = math.nan if kind == "nan" else math.inf
    layer.weight = nn.Parameter(weight)
    with pytest.raises(ValueError):
        dynamics.content_terms(torch.ones(2, 3, 64))


@pytest.mark.parametrize("kind", ["q_projection", "k_projection", "v_projection", "dot_product"])
def test_content_rejects_finite_projection_or_dot_product_overflow(kind):
    """有限输入也可能在投影或点积溢出，必须失败而非传播inf。"""
    dynamics = hand_content_dynamics()
    with torch.no_grad():
        if kind == "dot_product":
            dynamics.q_proj.weight.fill_(1e18)
            dynamics.k_proj.weight.fill_(1e18)
        else:
            getattr(dynamics, kind[0]+"_proj").weight.fill_(torch.finfo(torch.float32).max)
    with pytest.raises(ValueError):
        dynamics.content_terms(torch.ones(2, 3, 64))


def test_content_actual_cuda_device_boundaries():
    """可用CUDA实测输入错配及任一投影权重错配，不用meta冒充实际设备。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_content_dynamics()
    with pytest.raises(ValueError):
        dynamics.content_terms(torch.ones(1, 1, 64, device="cuda"))
    for name in ("q_proj", "k_proj", "v_proj"):
        dynamics = hand_content_dynamics()
        getattr(dynamics, name).to("cuda")
        with pytest.raises(ValueError):
            dynamics.content_terms(torch.ones(1, 1, 64))


def numpy64(value):
    """按实际存储的float32值复制为独立NumPy double参考，不用于生产计算。"""
    return value.detach().cpu().numpy().astype(np.float64)


def stage_parameters(dynamics, *, include_b=False):
    """旧阶段固定责任子集：干路+A为8384，D03再包含B头的4160m。"""
    names = ("condition_trunk", "gamma_head", "omega_head")+(('b_head',) if include_b else ())
    return tuple(p for name in names for p in getattr(dynamics, name).parameters())


def close(actual, expected):
    """保持A03锁定的float32容差，返回最大绝对误差便于保存原始诊断。"""
    array = numpy64(actual)
    np.testing.assert_allclose(array, expected, atol=1e-5, rtol=1e-4)
    return float(np.max(np.abs(array - expected)))


def hand_dynamics(device="cpu"):
    """仅单测覆盖为明确非零权重与偏置，生产初始化仍按Xavier/零偏置。"""
    dynamics = api().ModalDynamics(config()).to(device)
    weight = (((np.arange(64*65).reshape(64, 65) % 23)-11)*.006+.0003).astype(np.float32)
    bias = np.linspace(-.2, .2, 64, dtype=np.float32)
    with torch.no_grad():
        dynamics.condition_trunk[0].weight.copy_(torch.from_numpy(weight).to(device))
        dynamics.condition_trunk[0].bias.copy_(torch.from_numpy(bias).to(device))
    return dynamics


def reference(x, eigenvalues, weight, bias, upstream):
    """独立公式：mean、[lambda,g]、p*sigmoid(p)，以及SiLU全矩阵链式导数。"""
    g = x.sum(axis=1)/x.shape[1]
    v = np.concatenate((eigenvalues[..., None], np.repeat(g[:, None, :], eigenvalues.shape[1], axis=1)), -1)
    p = np.einsum("bki,ci->bkc", v, weight)+bias
    sigmoid = 1/(1+np.exp(-p))
    h = p*sigmoid
    d = upstream*(sigmoid+p*sigmoid*(1-sigmoid))
    dv = np.einsum("bkc,ci->bki", d, weight)
    dg = dv[..., 1:].sum(axis=1)
    dx = np.repeat((dg/x.shape[1])[:, None, :], x.shape[1], axis=1)
    dw = np.einsum("bkc,bki->ci", d, v)
    db = d.sum(axis=(0, 1))
    return g, v, h, dg, dx, dw, db


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_context_conditioning_permutation_and_pool_gradient(device):
    """T-D1：多样本非均匀tokens；独立前向、解析全梯度与排列等变。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; CPU acceptance is mandatory")
    dynamics = hand_dynamics(device)
    x = (torch.sin(torch.arange(384, device=device).float())*.4+
         torch.arange(384, device=device).float()/1000).reshape(2, 3, 64).requires_grad_()
    eigenvalues = torch.tensor([[1.7, .1, .1, -.2], [.3, 1.2, -.4, .6]], device=device)
    upstream = torch.linspace(-.4, .8, 512, device=device).reshape(2, 4, 64)
    layer = dynamics.condition_trunk[0]
    g_ref, v_ref, h_ref, dg, dx, dw, db = reference(
        numpy64(x), numpy64(eigenvalues), numpy64(layer.weight), numpy64(layer.bias), numpy64(upstream))
    g = dynamics.pool_context(x)
    g.retain_grad()
    captured = []
    handle = layer.register_forward_pre_hook(lambda module, args: captured.append(numpy64(args[0])))
    h = dynamics._condition_features(eigenvalues, g)
    handle.remove()
    errors = dict(pool=close(g, g_ref), concat=float(np.max(np.abs(captured[0]-v_ref))),
                  silu=close(h, h_ref))
    np.testing.assert_allclose(captured[0], v_ref, atol=1e-5, rtol=1e-4)
    (h*upstream).sum().backward()
    for name, actual, expected in (("g_grad", g.grad, dg), ("x_grad", x.grad, dx),
                                   ("weight_grad", layer.weight.grad, dw), ("bias_grad", layer.bias.grad, db)):
        errors[name] = close(actual, expected)
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    permuted = dynamics._condition_features(eigenvalues[:, permutation], g)
    errors["permutation"] = close(permuted, numpy64(h[:, permutation]))
    assert torch.equal(h[0, 1], h[0, 2])
    altered_lambda = dynamics._condition_features(eigenvalues+.2, g)
    altered_g = dynamics._condition_features(eigenvalues, g+.3)
    responses = dict(lambda_response=float((altered_lambda-h).abs().max()),
                     g_response=float((altered_g-h).abs().max()))
    assert responses["lambda_response"] > 1e-4 and responses["g_response"] > 1e-4
    assert eigenvalues.grad is None
    print(json.dumps(dict(probe="T-D1", device=device, fictional_action_dim=2, errors=errors, **responses)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pool_context_axis_and_independent_analytic_gradient(device):
    """独立池化导数J/N，避免把条件干路的梯度误当作mean的验证。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = api().ModalDynamics(config()).to(device)
    x = (torch.arange(640, device=device).float().reshape(2, 5, 64)/19-3).requires_grad_()
    j = torch.linspace(-.9, 1.1, 128, device=device).reshape(2, 64)
    pooled = dynamics.pool_context(x)
    error = close(pooled, numpy64(x).sum(axis=1)/5)
    (pooled*j).sum().backward()
    gradient_error = close(x.grad, np.repeat((numpy64(j)/5)[:, None, :], 5, axis=1))
    print(json.dumps(dict(probe="pool-only", device=device, forward_error=error, gradient_error=gradient_error)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_project_in_and_c04_constants_chain_gradients(device):
    """真实C01/C04输入接C06；核对raw反馈Z、P_in全部梯度，不通过eig求导。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(1703)
    dynamics = hand_dynamics(device)
    graph = GraphSpectrum(GraphConfig(16, 1703, .25, 1., 128, 1e-10, 1e-8)).to(device)
    before = {name: (id(p), p.detach().clone()) for name, p in dynamics.named_parameters()}
    model = POJEPA(ProjectionConfig(3, 64), graph, dynamics, object()).to(device)
    for name, p in dynamics.named_parameters():
        assert id(p) == before[name][0] and torch.equal(p, before[name][1])
    raw_z = (torch.arange(36, device=device).float().reshape(2, 6, 3)/20-.7).requires_grad_()
    x = model.project_in(raw_z)
    x.retain_grad()
    affinity = graph.affinity(x, regular_grid(2, 3))
    spectrum = graph.lowest_eigenpairs(graph.normalized_laplacian(affinity).laplacian,
                                      k=3, model_device=torch.device(device))
    g = dynamics.pool_context(x)
    g.retain_grad()
    h = dynamics._condition_features(spectrum.eigenvalues, g)
    upstream = torch.linspace(-.5, .7, h.numel(), device=device).reshape_as(h)
    layer = dynamics.condition_trunk[0]
    _, _, h_ref, dg, dx, dw, db = reference(numpy64(x), numpy64(spectrum.eigenvalues),
        numpy64(layer.weight), numpy64(layer.bias), numpy64(upstream))
    errors = dict(forward=close(h, h_ref))
    (h*upstream).sum().backward()
    dz = np.einsum("bnc,ce->bne", dx, numpy64(model.p_in.weight))
    dwi = np.einsum("bnc,bne->ce", dx, numpy64(raw_z))
    dbi = dx.sum(axis=(0, 1))
    for name, actual, expected in (("g", g.grad, dg), ("x", x.grad, dx),
            ("p_in_weight", model.p_in.weight.grad, dwi), ("p_in_bias", model.p_in.bias.grad, dbi),
            ("raw_z", raw_z.grad, dz), ("trunk_weight", layer.weight.grad, dw), ("trunk_bias", layer.bias.grad, db)):
        errors[name] = close(actual, expected)
    assert not spectrum.eigenvalues.requires_grad and spectrum.eigenvalues.grad is None
    assert not spectrum.basis.requires_grad and model.p_out.weight.grad is None
    print(json.dumps(dict(probe="real-C01-C04-C06-chain", device=device, errors=errors)))


def test_config_exact_explicit_fields_and_frozen_state():
    """十字段均必填且冻结；诊断r不按尚未给定的K截断。"""
    cls = api().DynamicsConfig
    names = ("action_dim", "dynamic_dim", "condition_width", "query_dim", "top_r",
             "spectral_bias_width", "spatial_bias_width", "action_hidden_width", "initial_damping", "head_weight_std")
    assert tuple(f.name for f in fields(cls)) == names
    assert all(f.default is MISSING and f.default_factory is MISSING for f in fields(cls))
    with pytest.raises(TypeError):
        cls()
    with pytest.raises(TypeError):
        cls(**{f.name: getattr(config(), f.name) for f in fields(cls) if f.name != "action_dim"})
    with pytest.raises(FrozenInstanceError):
        config().action_dim = 3
    assert config(top_r=1).top_r == 1
    assert config(top_r=999).top_r == 999
    assert config(action_dim=7).action_dim == 7  # 同为虚构边界诊断，不是正式动作定义。
    assert config(initial_damping=1, head_weight_std=2).initial_damping == 1


@pytest.mark.parametrize("name", ["action_dim", "dynamic_dim", "condition_width", "query_dim", "top_r",
                                   "spectral_bias_width", "spatial_bias_width", "action_hidden_width"])
@pytest.mark.parametrize("bad", [True, False, 0, -1, 2.0, "64", None, np.int64(64)])
def test_config_rejects_invalid_integer_fields(name, bad):
    """严格整数拒绝bool、零、负数和隐式类型转换。"""
    with pytest.raises(ValueError):
        config(**{name: bad})


@pytest.mark.parametrize("name", ["dynamic_dim", "condition_width", "query_dim", "spectral_bias_width",
                                  "spatial_bias_width", "action_hidden_width"])
def test_config_rejects_unregistered_positive_width(name):
    """正整数也不能改变首轮已经锁定的通道宽度。"""
    with pytest.raises(ValueError):
        config(**{name: 63})


@pytest.mark.parametrize("name", ["initial_damping", "head_weight_std"])
@pytest.mark.parametrize("bad", [True, False, 0., -1., math.nan, math.inf, -math.inf, 1j, "0.1", None])
def test_config_rejects_invalid_positive_reals(name, bad):
    """两个未来初始化量完整保存，但非法值应在分配网络前失败。"""
    with pytest.raises(ValueError):
        config(**{name: bad})


@pytest.mark.parametrize("bad", [None, {}, object(), 2])
def test_constructor_rejects_non_config(bad):
    """配置对象类型错误与数值错误分开报告。"""
    with pytest.raises(TypeError):
        api().ModalDynamics(bad)


@pytest.mark.parametrize("name", ["initial_damping", "head_weight_std"])
def test_config_real_overflow_is_value_error(name):
    """超出浮点有限检查可表示范围的显式实数也要以合同ValueError拒绝。"""
    with pytest.raises(ValueError):
        config(**{name: 10**1000})


def test_shared_trunk_and_free_heads_initialization_and_external_seed():
    """D03子集8384+4160m，干路仍4224；外部1703复现且不内部设种子。"""
    torch.manual_seed(1703)
    before = torch.get_rng_state().clone()
    dynamics = api().ModalDynamics(config())
    after = torch.get_rng_state().clone()
    assert not torch.equal(before, after)
    assert dynamics.config == config()
    assert tuple(n for n, p in dynamics.named_parameters() if n.startswith(("condition_trunk.", "gamma_head.", "omega_head.", "b_head."))) == ("condition_trunk.0.weight", "condition_trunk.0.bias",
        "gamma_head.weight", "gamma_head.bias", "omega_head.weight", "omega_head.bias", "b_head.weight", "b_head.bias")
    assert sum(p.numel() for p in dynamics.condition_trunk.parameters()) == 4224
    assert sum(p.numel() for p in stage_parameters(dynamics, include_b=True)) == 8384+4160*dynamics.config.action_dim
    assert all(p.requires_grad and p.dtype == torch.float32 and p.device.type == "cpu" for p in dynamics.parameters())
    assert list(dynamics.named_buffers()) == []
    assert tuple(n for n, m in dynamics.named_children() if n in ("condition_trunk", "gamma_head", "omega_head", "b_head")) == ("condition_trunk", "gamma_head", "omega_head", "b_head")
    assert len(dynamics.condition_trunk) == 2
    layer, activation = dynamics.condition_trunk
    assert type(layer) is nn.Linear and type(activation) is nn.SiLU
    assert layer.weight.shape == (64, 65) and layer.bias.shape == (64,)
    assert layer.weight.abs().max() <= math.sqrt(6/129)
    assert torch.count_nonzero(layer.weight) > 0 and torch.count_nonzero(layer.bias) == 0
    assert "forward" not in type(dynamics).__dict__
    torch.manual_seed(1703)
    again = api().ModalDynamics(config())
    assert all(torch.equal(a, b) for a, b in zip(dynamics.parameters(), again.parameters()))
    assert torch.equal(after, torch.get_rng_state())
    third = api().ModalDynamics(config())
    assert not torch.equal(layer.weight, third.condition_trunk[0].weight)


def hand_conditioned_dynamics(device="cpu"):
    """D03虚构m=2诊断：保留旧受控A，给每个(c,d)不同的非零B仿射权重。"""
    dynamics = hand_free_dynamics(device)
    assert hasattr(dynamics, "condition") and hasattr(dynamics, "action_rhs"), "D03公共接口尚未实现"
    weight = (((np.arange(128*64).reshape(128, 64) % 29)-14)*.005+.0007).astype(np.float32)
    with torch.no_grad():
        dynamics.b_head.weight.copy_(torch.from_numpy(weight).to(device))
        dynamics.b_head.bias.copy_(torch.linspace(-.3, .4, 128, device=device))
    return dynamics


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_linear_action_port_matches_manual_contraction_and_gradient(device, m):
    """T-D3：独立叶子B的全通道手算及a/B解析导数；不冒充端到端网络梯度。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = api().ModalDynamics(config(action_dim=m))
    assert hasattr(dynamics, "action_rhs"), "D03动作端口尚未实现"
    b = (torch.arange(2*3*64*m, device=device).float().reshape(2, 3, 64, m)%31-15)/7
    a = (torch.arange(2*m, device=device).float().reshape(2, m)+1)/3
    if m == 2:
        a[0] = torch.tensor([2., -1.], device=device)
        b[0, 0, :2] = torch.tensor([[1., 3.], [-2., .5]], device=device)
    a.requires_grad_()
    b.requires_grad_()
    upstream = torch.cos(torch.arange(384, device=device).float()*.17).reshape(2, 3, 64)+.2
    an, bn, gn = numpy64(a), numpy64(b), numpy64(upstream)
    expected = np.zeros((2, 3, 64))
    da, db = np.zeros_like(an), np.zeros_like(bn)
    for batch in range(2):
        for mode in range(3):
            for channel in range(64):
                for d in range(m):
                    expected[batch, mode, channel] += bn[batch, mode, channel, d]*an[batch, d]
                    da[batch, d] += gn[batch, mode, channel]*bn[batch, mode, channel, d]
                    db[batch, mode, channel, d] = gn[batch, mode, channel]*an[batch, d]
    result = dynamics.action_rhs(a, b)
    assert result.shape == (2, 3, 64) and result.dtype == torch.float32 and result.device == a.device
    errors = dict(output=close(result, expected))
    if m == 2:
        assert result[0, 0, 0] == -1 and result[0, 0, 1] == -4.5
    (result*upstream).sum().backward()
    errors.update(action=close(a.grad, da), b_matrix=close(b.grad, db))
    assert all(p.grad is None for p in dynamics.parameters())
    print(json.dumps(dict(probe="D03-manual-contraction", device=device, fictional_action_dim=m, errors=errors)))


def test_action_fixed_b_linearity_and_zero_action_gradient():
    """同一B下核零、负号、线性叠加；零动作时dB=0而da通常不为零。"""
    dynamics = hand_conditioned_dynamics()
    b = torch.linspace(-.2, .9, 2*3*64*2).reshape(2, 3, 64, 2).requires_grad_()
    a1, a2 = torch.tensor([[.4, -.7], [.9, .3]]), torch.tensor([[-.5, .2], [.6, -.8]])
    port = dynamics.action_rhs
    close(port(-a1, b), -numpy64(port(a1, b)))
    close(port(.7*a1-1.3*a2, b), .7*numpy64(port(a1, b))-1.3*numpy64(port(a2, b)))
    zero = torch.zeros(2, 2, requires_grad=True)
    out = port(zero, b)
    assert torch.count_nonzero(out) == 0
    out.sum().backward()
    assert torch.count_nonzero(b.grad) == 0
    close(zero.grad, numpy64(b).sum(axis=(1, 2)))
    assert torch.count_nonzero(zero.grad) > 0


def test_condition_five_fields_shared_features_affine_layout_and_private_a_only():
    """五字段真实返回，同一H给三头；B_flat[c*m+d]并按模态排列，不混动作轴。"""
    dynamics = hand_conditioned_dynamics()
    x = torch.linspace(-.4, .8, 2*5*64).reshape(2, 5, 64)
    eigenvalues = torch.tensor([[.1, .4, 1.2], [.3, .7, 1.7]])
    g = dynamics.pool_context(x).requires_grad_()
    expected_a, _, h = free_reference(dynamics, x, eigenvalues, torch.zeros(2, 3, 32, 2, 2))
    flat = h@numpy64(dynamics.b_head.weight).T+numpy64(dynamics.b_head.bias)
    expected_b = np.empty((2, 3, 64, 2))
    for c in range(64):
        for d in range(2):
            expected_b[:, :, c, d] = flat[:, :, c*2+d]
    trunk_outputs, head_inputs = [], []
    handles = [dynamics.condition_trunk.register_forward_hook(lambda module, args, output: trunk_outputs.append(output))]
    handles += [head.register_forward_pre_hook(lambda module, args: head_inputs.append(args[0]))
                for head in (dynamics.gamma_head, dynamics.omega_head, dynamics.b_head)]
    try:
        actual = dynamics.condition(eigenvalues, g)
    finally:
        for handle in handles:
            handle.remove()
    assert len(trunk_outputs) == 1 and len(head_inputs) == 3
    assert all(value is trunk_outputs[0] for value in head_inputs)
    assert type(actual) is api().ConditionedCoefficients
    assert tuple(f.name for f in fields(actual)) == ("raw_gamma", "gamma", "omega", "a_blocks", "b_matrix")
    assert all(f.default is MISSING and f.default_factory is MISSING for f in fields(actual))
    with pytest.raises(FrozenInstanceError):
        actual.b_matrix = actual.a_blocks
    expected = dict(expected_a, b_matrix=expected_b)
    for name, value in expected.items():
        field = getattr(actual, name)
        close(field, value)
        assert tuple(field.shape) == value.shape and field.requires_grad
        assert field.dtype == torch.float32 and field.device == g.device and torch.isfinite(field).all()
    calls = []
    handle = dynamics.b_head.register_forward_pre_hook(lambda module, args: calls.append(args))
    try:
        private = dynamics._condition_free(eigenvalues, g)
    finally:
        handle.remove()
    assert calls == []
    for name in expected_a:
        assert torch.equal(getattr(private, name), getattr(actual, name))
    permutation = torch.tensor([2, 0, 1])
    permuted = dynamics.condition(eigenvalues[:, permutation], g)
    for name in expected:
        close(getattr(permuted, name), numpy64(getattr(actual, name)[:, permutation]))
    assert (dynamics.condition(eigenvalues+.4, g).b_matrix-actual.b_matrix).abs().max() > 1e-4
    assert (dynamics.condition(eigenvalues, g+.3).b_matrix-actual.b_matrix).abs().max() > 1e-4


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("joint", [False, True])
def test_condition_action_real_c04_chain_and_shared_branch_analytic_gradients(device, joint):
    """真实C04常量与raw Z链：独立核B/W_B/b_B/H/g/X/P_in/Z及A+Ba梯度相加。

    B来自本次condition原图，不detach后重造叶子。joint诊断的<A,G_A>+
    <Ba,G_Y>使三头参与；单B分支的两A头梯度None是正确的未使用状态。
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_conditioned_dynamics(device)
    graph = GraphSpectrum(GraphConfig(16, 1703, .25, 1., 128, 1e-10, 1e-8)).to(device)
    torch.manual_seed(1703)
    model = POJEPA(ProjectionConfig(3, 64), graph, dynamics, object()).to(device)
    raw_z = torch.linspace(-.4, .9, 36, device=device).reshape(2, 6, 3).requires_grad_()
    x = model.project_in(raw_z)
    x.retain_grad()
    spectrum = graph.lowest_eigenpairs(graph.normalized_laplacian(graph.affinity(x, regular_grid(2, 3))).laplacian,
                                      k=3, model_device=torch.device(device))
    g = dynamics.pool_context(x)
    g.retain_grad()
    action = torch.tensor([[.4, -.7], [.8, .3]], device=device, requires_grad=True)
    gy = torch.sin(torch.arange(384, device=device).float()*.19).reshape(2, 3, 64)+.2
    ga = torch.cos(torch.arange(2*3*32*4, device=device).float()*.23).reshape(2, 3, 32, 2, 2)
    if not joint:
        ga.zero_()
    _, a_gradients, h = free_reference(dynamics, x, spectrum.eigenvalues, ga)
    wb, bb, an, gyn = numpy64(dynamics.b_head.weight), numpy64(dynamics.b_head.bias), numpy64(action), numpy64(gy)
    b_ref = (h@wb.T+bb).reshape(2, 3, 64, 2)
    db = gyn[..., None]*an[:, None, None, :]
    dflat = db.reshape(2, 3, 128)
    dh_b = dflat@wb
    dh = a_gradients["h"]+dh_b
    _, _, _, dg, dx, dw, d_bias = reference(numpy64(x), numpy64(spectrum.eigenvalues),
        numpy64(dynamics.condition_trunk[0].weight), numpy64(dynamics.condition_trunk[0].bias), dh)
    gradients = dict(b_matrix=db, action=np.einsum("bkc,bkcd->bd", gyn, b_ref),
        b_weight=dflat.reshape(-1, 128).T@h.reshape(-1, 64), b_bias=dflat.sum(axis=(0, 1)),
        h=dh, g=dg, x=dx, trunk_weight=dw, trunk_bias=d_bias,
        raw_z=np.einsum("bnc,ce->bne", dx, numpy64(model.p_in.weight)),
        p_in_weight=np.einsum("bnc,bne->ce", dx, numpy64(raw_z)), p_in_bias=dx.sum(axis=(0, 1)))
    captured = []
    def retain_features(module, inputs, output):
        """只为单测读取本次H实际梯度，不给生产模型增加缓存。"""
        output.retain_grad()
        captured.append(output)
    handle = dynamics.condition_trunk.register_forward_hook(retain_features)
    actual = dynamics.condition(spectrum.eigenvalues, g)
    handle.remove()
    actual.b_matrix.retain_grad()
    y = dynamics.action_rhs(action, actual.b_matrix)
    loss = (y*gy).sum()
    if joint:
        loss = loss+(actual.a_blocks*ga).sum()
    loss.backward()
    layer = dynamics.condition_trunk[0]
    actual_gradients = dict(b_matrix=actual.b_matrix.grad, action=action.grad,
        b_weight=dynamics.b_head.weight.grad, b_bias=dynamics.b_head.bias.grad,
        h=captured[0].grad, g=g.grad, x=x.grad, trunk_weight=layer.weight.grad, trunk_bias=layer.bias.grad,
        raw_z=raw_z.grad, p_in_weight=model.p_in.weight.grad, p_in_bias=model.p_in.bias.grad)
    errors = dict(b_forward=close(actual.b_matrix, b_ref), output=close(y, np.einsum("bkcd,bd->bkc", b_ref, an)))
    for name, value in actual_gradients.items():
        errors[name] = close(value, gradients[name])
        assert torch.count_nonzero(value) > 0
    for head_name in ("gamma", "omega"):
        head = getattr(dynamics, head_name+"_head")
        for suffix, value in (("weight", head.weight.grad), ("bias", head.bias.grad)):
            if joint:
                errors[head_name+"_"+suffix] = close(value, a_gradients[head_name+"_"+suffix])
                assert torch.count_nonzero(value) > 0
            else:
                assert value is None
    assert len(captured) == 1 and not spectrum.eigenvalues.requires_grad and spectrum.eigenvalues.grad is None
    assert not spectrum.basis.requires_grad and model.p_out.weight.grad is None
    print(json.dumps(dict(probe="D03-real-chain", device=device, joint=joint, errors=errors)))


def test_b_initialization_preserves_exact_old_six_parameter_rng_prefix():
    """按D02原顺序在相同外部seed构造六参数；追加B不能改变任何旧初值。"""
    torch.manual_seed(1703)
    trunk = nn.Linear(65, 64, bias=True, device="cpu", dtype=torch.float32)
    nn.init.xavier_uniform_(trunk.weight, gain=1.)
    nn.init.zeros_(trunk.bias)
    gamma, omega = (nn.Linear(64, 32, bias=True, device="cpu", dtype=torch.float32) for _ in range(2))
    nn.init.normal_(gamma.weight, mean=0., std=1e-3)
    nn.init.normal_(omega.weight, mean=0., std=1e-3)
    nn.init.constant_(gamma.bias, .1+math.log(-math.expm1(-.1)))
    nn.init.zeros_(omega.bias)
    torch.manual_seed(1703)
    dynamics = api().ModalDynamics(config())
    assert hasattr(dynamics, "b_head"), "D03 B头尚未实现"
    for old, current in zip((trunk, gamma, omega), (dynamics.condition_trunk[0], dynamics.gamma_head, dynamics.omega_head)):
        assert torch.equal(old.weight, current.weight) and torch.equal(old.bias, current.bias)
    assert sum(p.numel() for p in stage_parameters(dynamics, include_b=True)) == 8384+4160*2


@pytest.mark.parametrize("m,batch,k", [(1, 1, 1), (3, 2, 4)])
def test_condition_b_layout_other_action_dimensions_and_minimal_axes(m, batch, k):
    """另用虚构m=1/3核c*m+d顺序及最小B=K=m=1；正式动作维数仍待F01。"""
    dynamics = api().ModalDynamics(config(action_dim=m))
    with torch.no_grad():
        dynamics.b_head.weight.copy_(torch.arange(64*m*64).reshape(64*m, 64)%17*.002-.015)
        dynamics.b_head.bias.copy_(torch.linspace(-.4, .7, 64*m))
    lam = torch.linspace(.1, 1.8, batch*k).reshape(batch, k)
    g = torch.linspace(-.4, .8, batch*64).reshape(batch, 64)
    _, _, h, *_ = reference(numpy64(g)[:, None], numpy64(lam), numpy64(dynamics.condition_trunk[0].weight),
        numpy64(dynamics.condition_trunk[0].bias), np.zeros((batch, k, 64)))
    flat = h@numpy64(dynamics.b_head.weight).T+numpy64(dynamics.b_head.bias)
    expected = np.empty((batch, k, 64, m))
    for c in range(64):
        for d in range(m):
            expected[:, :, c, d] = flat[:, :, c*m+d]
    actual = dynamics.condition(lam, g)
    close(actual.b_matrix, expected)
    action = torch.linspace(-.5, .9, batch*m).reshape(batch, m)
    close(dynamics.action_rhs(action, actual.b_matrix), np.sum(expected*numpy64(action)[:, None, None], axis=-1))
    assert sum(p.numel() for p in stage_parameters(dynamics, include_b=True)) == 8384+4160*m


def test_public_condition_noncontiguous_lambda_g_preserves_real_gradients():
    """非连续lambda/g经公共B真实反向，不以shape或requires_grad替代解析导数。"""
    dynamics = hand_conditioned_dynamics()
    lam = torch.linspace(.1, 1.8, 12).reshape(6, 2).T
    g = torch.linspace(-.4, .8, 128).reshape(64, 2).T.requires_grad_()
    assert not lam.is_contiguous() and not g.is_contiguous()
    _, _, h, *_ = reference(numpy64(g)[:, None], numpy64(lam), numpy64(dynamics.condition_trunk[0].weight),
        numpy64(dynamics.condition_trunk[0].bias), np.zeros((2, 6, 64)))
    action = torch.tensor([[.7, -.4], [.3, .9]])
    db = np.broadcast_to(numpy64(action)[:, None, None], (2, 6, 64, 2))
    dh = db.reshape(2, 6, 128)@numpy64(dynamics.b_head.weight)
    _, _, _, dg, *_ = reference(numpy64(g)[:, None], numpy64(lam), numpy64(dynamics.condition_trunk[0].weight),
        numpy64(dynamics.condition_trunk[0].bias), dh)
    result = dynamics.condition(lam, g)
    close(result.b_matrix, (h@numpy64(dynamics.b_head.weight).T+numpy64(dynamics.b_head.bias)).reshape(2, 6, 64, 2))
    dynamics.action_rhs(action, result.b_matrix).sum().backward()
    close(g.grad, dg)
    assert torch.count_nonzero(g.grad) > 0


def test_b_initialization_rejects_its_own_nonfinite_parameters(monkeypatch):
    """A两头正常而仅B初始化损坏也必须拒绝，不靠旧两头检查碰巧失败。"""
    normal = nn.init.normal_
    def corrupt_b(tensor, **kwargs):
        """只注入B头故障；不改生产初值或全局数值合同。"""
        if tensor.shape == (128, 64):
            with torch.no_grad():
                return tensor.fill_(math.inf)
        return normal(tensor, **kwargs)
    monkeypatch.setattr(nn.init, "normal_", corrupt_b)
    with pytest.raises(ValueError):
        api().ModalDynamics(config())


def test_condition_rejects_unexpected_b_output_precision(monkeypatch):
    """B参数正确但运算输出精度意外改变也明确失败，不隐式转回float32。"""
    dynamics = hand_conditioned_dynamics()
    forward = dynamics.b_head.forward
    monkeypatch.setattr(dynamics.b_head, "forward", lambda value: forward(value).to(torch.bfloat16))
    with pytest.raises(ValueError):
        dynamics.condition(torch.ones(1, 1), torch.ones(1, 64))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_condition_action_noncontiguous_inputs_and_forward_state(device):
    """非连续输入保图；变化B/N/K不建参数，输入/模式/RNG/默认设置和成员不变。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_conditioned_dynamics(device).eval()
    parameters = {n: (id(p), p.detach().clone()) for n, p in dynamics.named_parameters()}
    members = {n: set(vars(m)) for n, m in dynamics.named_modules()}
    modes = {n: m.training for n, m in dynamics.named_modules()}
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    defaults = torch.get_default_dtype(), torch.get_default_device()
    for batch, n, k in ((2, 5, 3), (1, 1, 1), (3, 4, 2)):
        x = torch.linspace(-.3, .7, batch*64*n, device=device).reshape(batch, 64, n).transpose(1, 2)
        lam = torch.linspace(0., 1.8, batch*k*2, device=device).reshape(batch, k*2)[:, ::2]
        g = dynamics.pool_context(x).T.contiguous().T
        action = torch.linspace(-.8, .5, batch*4, device=device).reshape(batch, 4)[:, ::2]
        saved = [v.clone() for v in (x, lam, g, action)]
        actual = dynamics.condition(lam, g)
        b = actual.b_matrix.transpose(1, 2).contiguous().transpose(1, 2)
        assert batch == 1 or not action.is_contiguous()
        if batch > 1:
            assert not g.is_contiguous() and not b.is_contiguous()
        close(dynamics.action_rhs(action, b), np.einsum("bkcd,bd->bkc", numpy64(b), numpy64(action)))
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                out = dynamics.condition(lam, g)
                y = dynamics.action_rhs(action, out.b_matrix)
                assert not torch.is_grad_enabled() and not y.requires_grad
        assert all(torch.equal(v, copy) for v, copy in zip((x, lam, g, action), saved))
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert members == {n: set(vars(m)) for n, m in dynamics.named_modules()}
    assert modes == {n: m.training for n, m in dynamics.named_modules()}
    assert tuple(parameters) == tuple(dict(dynamics.named_parameters())) and list(dynamics.named_buffers()) == []
    assert all(id(p) == parameters[n][0] and torch.equal(p, parameters[n][1]) and p.grad is None for n, p in dynamics.named_parameters())
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())


@pytest.mark.parametrize("ashape,bshape", [((1, 2), (2, 3, 64, 2)), ((2, 2), (1, 3, 64, 2)),
    ((2,), (2, 3, 64, 2)), ((2, 1, 2), (2, 3, 64, 2)), ((2, 3), (2, 3, 64, 2)),
    ((2, 2), (2, 3, 63, 2)), ((2, 2), (2, 3, 64, 3)), ((2, 2), (2, 64, 2)),
    ((0, 2), (0, 3, 64, 2)), ((2, 2), (2, 0, 64, 2))])
def test_action_port_rejects_bad_axes_and_batch_broadcast(ashape, bshape):
    """rank、B、K、64和m严格检查；形状可广播也不能跨batch悄悄共享动作。"""
    dynamics = hand_conditioned_dynamics()
    with pytest.raises(ValueError):
        dynamics.action_rhs(torch.zeros(ashape), torch.zeros(bshape))


@pytest.mark.parametrize("name", ["action", "b_matrix"])
@pytest.mark.parametrize("kind", ["none", "double", "int", "complex", "nan", "inf", "meta", "sparse", "nested"])
def test_action_port_rejects_non_dense_nonfinite_and_non_float32(name, kind):
    """显式边界拒绝非法张量；D02的double解析例外不扩展到动作端口。"""
    dynamics = hand_conditioned_dynamics()
    args = dict(action=torch.ones(1, 2), b_matrix=torch.ones(1, 1, 64, 2))
    value = args[name]
    if kind == "none":
        value = None
    elif kind in ("double", "int", "complex"):
        value = value.to(dtype={"double": torch.float64, "int": torch.int64, "complex": torch.complex64}[kind])
    elif kind in ("nan", "inf"):
        value.fill_(float(kind))
    elif kind == "meta":
        value = value.to("meta")
    elif kind == "sparse":
        value = value.to_sparse()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = torch.nested.nested_tensor([value[0]])
    args[name] = value
    with pytest.raises(TypeError if kind == "none" else ValueError):
        dynamics.action_rhs(**args)


@pytest.mark.parametrize("head", ["gamma_head", "omega_head", "b_head"])
@pytest.mark.parametrize("kind", ["double", "meta", "bias_double", "nonfinite"])
def test_public_condition_checks_all_heads(head, kind):
    """公共condition逐头验证实际设备/精度和输出；不静默转型或修补非有限结果。"""
    dynamics = hand_conditioned_dynamics()
    layer = getattr(dynamics, head)
    if kind == "double":
        layer.double()
    elif kind == "meta":
        layer.to("meta")
    elif kind == "bias_double":
        layer.bias = nn.Parameter(layer.bias.double())
    else:
        with torch.no_grad():
            layer.bias.fill_(math.inf)
    with pytest.raises(ValueError):
        dynamics.condition(torch.ones(1, 1), torch.ones(1, 64))


@pytest.mark.parametrize("which", ["lambda_grad", "lambda_shape", "g_shape", "empty_batch", "empty_k", "dtype", "nan"])
def test_public_condition_preserves_existing_input_boundary(which):
    """公共condition沿用C06真实输入验证，lambda不能求导，B/K正且batch相同。"""
    dynamics = hand_conditioned_dynamics()
    lam, g = torch.zeros(1, 1), torch.ones(1, 64)
    if which == "lambda_grad":
        lam.requires_grad_()
    elif which == "lambda_shape":
        lam = torch.ones(1)
    elif which == "g_shape":
        g = torch.ones(2, 64)
    elif which == "empty_batch":
        lam, g = torch.ones(0, 1), torch.ones(0, 64)
    elif which == "empty_k":
        lam = torch.ones(1, 0)
    elif which == "dtype":
        g = g.double()
    else:
        lam.fill_(math.nan)
    with pytest.raises(ValueError):
        dynamics.condition(lam, g)


def test_action_port_noncontiguous_backward_and_finite_product_overflow():
    """非连续a/B保留解析导数；有限输入的乘积溢出必须显式失败。"""
    dynamics = hand_conditioned_dynamics()
    a = torch.linspace(-.4, .9, 8).reshape(2, 4)[:, ::2].detach().requires_grad_()
    b = torch.linspace(-.2, .7, 2*64*3*2).reshape(2, 64, 3, 2).transpose(1, 2).detach().requires_grad_()
    assert not a.is_contiguous() and not b.is_contiguous()
    dynamics.action_rhs(a, b).sum().backward()
    close(a.grad, numpy64(b).sum(axis=(1, 2)))
    close(b.grad, np.broadcast_to(numpy64(a)[:, None, None], b.shape))
    with pytest.raises(ValueError):
        dynamics.action_rhs(torch.full((1, 2), torch.finfo(torch.float32).max), torch.full((1, 1, 64, 2), 2.))
    for target in ("double", "meta"):
        altered = dynamics.double() if target == "double" else dynamics.to("meta")
        close(altered.action_rhs(torch.ones(1, 2), torch.ones(1, 1, 64, 2)), np.full((1, 1, 64), 2.))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_action_condition_reject_mixed_real_devices_and_allow_parameter_independent_contraction():
    """网络参数约束condition；显式动作收缩只约束a/B，不依赖网络参数设备。"""
    dynamics = hand_conditioned_dynamics("cuda")
    with pytest.raises(ValueError):
        dynamics.condition(torch.ones(1, 1), torch.ones(1, 64, device="cuda"))
    with pytest.raises(ValueError):
        dynamics.condition(torch.ones(1, 1, device="cuda"), torch.ones(1, 64))
    for a, b in ((torch.ones(1, 2), torch.ones(1, 1, 64, 2, device="cuda")),
                 (torch.ones(1, 2, device="cuda"), torch.ones(1, 1, 64, 2))):
        with pytest.raises(ValueError):
            dynamics.action_rhs(a, b)
    close(dynamics.action_rhs(torch.ones(1, 2), torch.ones(1, 1, 64, 2)), np.full((1, 1, 64), 2.))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_parameters_no_hidden_cache_and_no_call_side_effects(device):
    """连续变化B/N/K、g1/g2/g1及排列/复制复用同一参数；不声称重采样一致。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_dynamics(device).eval()
    parameters = {name: (id(p), p.detach().clone()) for name, p in dynamics.named_parameters()}
    members = set(vars(dynamics))
    modes = {name: module.training for name, module in dynamics.named_modules()}
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    defaults = torch.get_default_dtype(), torch.get_default_device()
    for b, n, k in ((2, 5, 4), (1, 1, 1), (3, 7, 2), (2, 3, 6)):
        x = torch.linspace(-.5, 1.3, b*n*64, device=device).reshape(b, n, 64)
        eigenvalues = torch.linspace(-.4, 2.4, b*k, device=device).reshape(b, k)
        saved = x.clone(), eigenvalues.clone()
        g1 = dynamics.pool_context(x)
        g2 = dynamics.pool_context(x+1)
        h1 = dynamics._condition_features(eigenvalues, g1)
        h2 = dynamics._condition_features(eigenvalues, g2)
        close(dynamics._condition_features(eigenvalues, g1), numpy64(h1))
        assert not torch.allclose(h1, h2, atol=1e-5, rtol=1e-4)
        close(dynamics.pool_context(x.flip(1)), numpy64(g1))
        close(dynamics.pool_context(x.repeat(1, 2, 1)), numpy64(g1))
        assert h1.shape == (b, k, 64)
        assert torch.equal(x, saved[0]) and torch.equal(eigenvalues, saved[1])
    assert members == set(vars(dynamics))
    assert tuple(parameters) == tuple(dict(dynamics.named_parameters()))
    for name, p in dynamics.named_parameters():
        assert id(p) == parameters[name][0] and torch.equal(p, parameters[name][1])
    assert {name: m.training for name, m in dynamics.named_modules()} == modes
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())
    assert list(dynamics.named_buffers()) == []


def test_constructor_preserves_non_torch_state_and_explicit_cpu_precision():
    """调用者默认meta/double也不改变本步CPU float32分配；不影响其他模型和RNG。"""
    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    other = nn.Linear(2, 2).eval()
    before = {n: p.detach().clone() for n, p in other.named_parameters()}
    py_rng, np_rng = random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        torch.set_default_dtype(torch.float64)
        torch.set_default_device("meta")
        dynamics = api().ModalDynamics(config())
        assert torch.get_default_dtype() == torch.float64 and torch.get_default_device().type == "meta"
        assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in dynamics.parameters())
    finally:
        torch.set_default_dtype(dtype)
        torch.set_default_device(device)
    assert not other.training and all(torch.equal(p, before[n]) for n, p in other.named_parameters())
    assert py_rng == random.getstate()
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))


def test_noncontiguous_inputs_and_no_grad_inference_preserve_mode():
    """非连续X/lambda/g合法；上下文不要求输入有梯度，不改变调用者grad mode。"""
    dynamics = hand_dynamics()
    x = torch.linspace(-1., 1., 2*64*5).reshape(2, 64, 5).transpose(1, 2).requires_grad_()
    eigenvalues = torch.arange(12).float().reshape(6, 2).T/8
    pooled = torch.linspace(-.4, .8, 128).reshape(64, 2).T.requires_grad_()
    assert not x.is_contiguous() and not eigenvalues.is_contiguous() and not pooled.is_contiguous()
    close(dynamics.pool_context(x), numpy64(x).sum(axis=1)/5)
    h = dynamics._condition_features(eigenvalues, pooled)
    assert h.shape == (2, 6, 64) and h.requires_grad
    h.sum().backward()
    assert pooled.grad is not None
    with torch.no_grad():
        assert not torch.is_grad_enabled()
        g = dynamics.pool_context(x)
        assert not g.requires_grad and not dynamics._condition_features(eigenvalues, g).requires_grad
        assert not torch.is_grad_enabled()
    with torch.inference_mode():
        assert torch.is_inference_mode_enabled()
        assert not dynamics._condition_features(eigenvalues, dynamics.pool_context(x)).requires_grad
        assert torch.is_inference_mode_enabled()
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()


@pytest.mark.parametrize("target", ["x", "eigenvalues", "pooled"])
@pytest.mark.parametrize("bad", [None, [[1.]], 1.])
def test_rejects_non_tensor_inputs(target, bad):
    """类型错误统一TypeError，不悄悄包装调用者数据。"""
    dynamics = hand_dynamics()
    with pytest.raises(TypeError):
        if target == "x":
            dynamics.pool_context(bad)
        else:
            args = dict(eigenvalues=torch.ones(2, 3), pooled=torch.ones(2, 64))
            args[target] = bad
            dynamics._condition_features(**args)


@pytest.mark.parametrize("shape", [(0, 2, 64), (2, 0, 64), (2, 3, 63), (2, 64), (2, 1, 3, 64)])
def test_pool_rejects_invalid_shapes(shape):
    """B/N正且特征宽度恰为64，不能猜测/合并轴。"""
    with pytest.raises(ValueError):
        hand_dynamics().pool_context(torch.zeros(shape))


@pytest.mark.parametrize("lshape,gshape", [((0, 3), (0, 64)), ((2, 0), (2, 64)), ((2,), (2, 64)),
            ((2, 3, 1), (2, 64)), ((2, 3), (1, 64)), ((2, 3), (2, 63)), ((2, 3), (2, 1, 64))])
def test_condition_rejects_invalid_shapes_and_batch_broadcast(lshape, gshape):
    """只广播K轴；不同样本batch绝不自动广播。"""
    with pytest.raises(ValueError):
        hand_dynamics()._condition_features(torch.zeros(lshape), torch.zeros(gshape))


@pytest.mark.parametrize("target", ["x", "eigenvalues", "pooled"])
@pytest.mark.parametrize("kind", ["double", "half", "integer", "complex", "sparse", "meta", "nan", "inf"])
def test_rejects_invalid_tensor_contract(target, kind):
    """稠密实数有限float32且非meta；不做dtype/device补救。"""
    values = dict(x=torch.ones(2, 3, 64), eigenvalues=torch.ones(2, 3), pooled=torch.ones(2, 64))
    value = values[target]
    conversions = dict(double=lambda t: t.double(), half=lambda t: t.half(), integer=lambda t: t.long(),
                       complex=lambda t: t.to(torch.complex64), sparse=lambda t: t.to_sparse(),
                       meta=lambda t: t.to("meta"), nan=lambda t: t*math.nan, inf=lambda t: t*math.inf)
    values[target] = conversions[kind](value)
    dynamics = hand_dynamics()
    with pytest.raises(ValueError):
        if target == "x":
            dynamics.pool_context(values["x"])
        else:
            dynamics._condition_features(values["eigenvalues"], values["pooled"])


@pytest.mark.parametrize("target", ["x", "eigenvalues", "pooled"])
def test_rejects_nested_tensors_as_non_dense(target):
    """NestedTensor会报告strided但不满足稠密矩形合同，应明确ValueError。"""
    values = dict(x=torch.ones(2, 3, 64), eigenvalues=torch.ones(2, 3), pooled=torch.ones(2, 64))
    items = dict(x=[torch.ones(2, 64), torch.ones(3, 64)],
                 eigenvalues=[torch.ones(2), torch.ones(3)], pooled=[torch.ones(64), torch.ones(64)])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")
        values[target] = torch.nested.nested_tensor(items[target], dtype=torch.float32)
    dynamics = hand_dynamics()
    with pytest.raises(ValueError):
        if target == "x":
            dynamics.pool_context(values["x"])
        else:
            dynamics._condition_features(values["eigenvalues"], values["pooled"])


def test_lambda_grad_is_an_error_and_finite_unordered_values_are_not_repaired():
    """C04负责谱合法性；本步拒绝可训练lambda，不排序/裁剪或悄detach。"""
    dynamics = hand_dynamics()
    eigenvalues = torch.tensor([[2.2, -.1, .4]], requires_grad=True)
    with pytest.raises(ValueError):
        dynamics._condition_features(eigenvalues, torch.ones(1, 64))
    assert eigenvalues.requires_grad and eigenvalues.grad is None
    constant = eigenvalues.detach()
    h = dynamics._condition_features(constant, torch.ones(1, 64))
    expected = reference(np.ones((1, 1, 64)), numpy64(constant), numpy64(dynamics.condition_trunk[0].weight),
                         numpy64(dynamics.condition_trunk[0].bias), np.ones((1, 3, 64)))[2]
    close(h, expected)
    assert torch.equal(constant, torch.tensor([[2.2, -.1, .4]]))


@pytest.mark.parametrize("parameter_kind", ["double", "meta", "bias_double"])
def test_model_parameter_precision_and_device_are_checked(parameter_kind):
    """即使pool不乘权重，也按实际干路精度/设备核对输入合同。"""
    dynamics = hand_dynamics()
    if parameter_kind == "bias_double":
        dynamics.condition_trunk[0].bias = nn.Parameter(dynamics.condition_trunk[0].bias.double())
    else:
        dynamics = dynamics.double() if parameter_kind == "double" else dynamics.to("meta")
    with pytest.raises(ValueError):
        dynamics.pool_context(torch.ones(1, 1, 64))
    with pytest.raises(ValueError):
        dynamics._condition_features(torch.ones(1, 1), torch.ones(1, 64))


def test_cuda_device_mismatch_fails_without_copy():
    """有CUDA时实际验证输入/参数设备错配，正确设备路径已独立反向验收。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cpu, cuda = hand_dynamics(), hand_dynamics("cuda")
    with pytest.raises(ValueError):
        cpu.pool_context(torch.ones(1, 1, 64, device="cuda"))
    with pytest.raises(ValueError):
        cuda.pool_context(torch.ones(1, 1, 64))
    with pytest.raises(ValueError):
        cuda._condition_features(torch.ones(1, 1), torch.ones(1, 64, device="cuda"))
    with pytest.raises(ValueError):
        cuda._condition_features(torch.ones(1, 1, device="cuda"), torch.ones(1, 64))


def test_finite_inputs_overflow_is_reported_not_clipped():
    """输入有限不保证求和/仿射有限，溢出须明确失败而非剪裁。"""
    dynamics = hand_dynamics()
    maximum = torch.finfo(torch.float32).max
    with pytest.raises(ValueError):
        dynamics.pool_context(torch.full((1, 2, 64), maximum))
    with torch.no_grad():
        dynamics.condition_trunk[0].weight.fill_(1.)
    with pytest.raises(ValueError):
        dynamics._condition_features(torch.ones(1, 1), torch.full((1, 64), maximum))


def test_external_autocast_wrong_output_precision_is_rejected():
    """默认不开AMP；外部autocast导致输出非float32时明确拒绝。"""
    dynamics = hand_dynamics()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError):
            dynamics._condition_features(torch.ones(1, 2), torch.ones(1, 64))


def hand_free_dynamics(device="cpu"):
    """虚构非零双头夹具；32对采用不同权重/偏置，避免轴或符号错误被常量掩盖。"""
    dynamics = hand_dynamics(device)
    assert hasattr(dynamics, "_condition_free"), "D01 A 分支尚未实现"
    with torch.no_grad():
        for index, head in enumerate((dynamics.gamma_head, dynamics.omega_head)):
            weight = (((np.arange(32*64).reshape(32, 64) % (17+index*2))-8)*.008+.001).astype(np.float32)
            head.weight.copy_(torch.from_numpy(weight).to(device))
            head.bias.copy_(torch.linspace(-1.3, .4, 32, device=device) if index == 0 else
                            torch.linspace(-.5, .5, 32, device=device))
    return dynamics


def free_reference(dynamics, x, eigenvalues, upstream):
    """独立double前向及<A,G>解析反向；调试标量不是论文训练损失。

    G给出每块四个位置的导数；先算E_gamma/E_omega，再经线性头、SiLU、
    [lambda,g]和mean链式回传。只读取实际float32存储值，不调用被测前向。
    """
    layer = dynamics.condition_trunk[0]
    w, bias = numpy64(layer.weight), numpy64(layer.bias)
    x, eigenvalues, upstream = numpy64(x), numpy64(eigenvalues), numpy64(upstream)
    _, _, h, *_ = reference(x, eigenvalues, w, bias, np.zeros((*eigenvalues.shape, 64)))
    wg, bg = numpy64(dynamics.gamma_head.weight), numpy64(dynamics.gamma_head.bias)
    wo, bo = numpy64(dynamics.omega_head.weight), numpy64(dynamics.omega_head.bias)
    raw = np.einsum("bkh,rh->bkr", h, wg)+bg
    gamma = np.logaddexp(0., raw)
    omega = np.einsum("bkh,rh->bkr", h, wo)+bo
    a = np.empty((*raw.shape, 2, 2))
    a[..., 0, 0], a[..., 0, 1] = -gamma, -omega
    a[..., 1, 0], a[..., 1, 1] = omega, -gamma
    eg = -(upstream[..., 0, 0]+upstream[..., 1, 1])/(1+np.exp(-raw))
    eo = -upstream[..., 0, 1]+upstream[..., 1, 0]
    dh = np.einsum("bkr,rh->bkh", eg, wg)+np.einsum("bkr,rh->bkh", eo, wo)
    _, _, _, dg, dx, dw, db = reference(x, eigenvalues, w, bias, dh)
    gradients = dict(g=dg, x=dx, h=dh, trunk_weight=dw, trunk_bias=db,
        gamma_weight=eg.reshape(-1, 32).T@h.reshape(-1, 64), gamma_bias=eg.sum(axis=(0, 1)),
        omega_weight=eo.reshape(-1, 32).T@h.reshape(-1, 64), omega_bias=eo.sum(axis=(0, 1)))
    return dict(raw_gamma=raw, gamma=gamma, omega=omega, a_blocks=a), gradients, h


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_free_blocks_independent_formula_and_real_projection_analytic_gradients(device):
    """式9/10：真实C01/C04常量接A；核四位置、32对、两头至raw Z全部解析导数。

    m=2仅虚构诊断；lambda不求eig梯度，率按内部周期^-1解释，不声称物理频率。
    CPU必验；有CUDA时实际组块和反向，float32容差保持1e-5/1e-4。
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_free_dynamics(device)
    graph = GraphSpectrum(GraphConfig(16, 1703, .25, 1., 128, 1e-10, 1e-8)).to(device)
    torch.manual_seed(1703)
    model = POJEPA(ProjectionConfig(3, 64), graph, dynamics, object()).to(device)
    raw_z = (torch.cos(torch.arange(36, device=device).float())*.3+
             torch.arange(36, device=device).float()/50).reshape(2, 6, 3).requires_grad_()
    x = model.project_in(raw_z)
    x.retain_grad()
    affinity = graph.affinity(x, regular_grid(2, 3))
    spectrum = graph.lowest_eigenpairs(graph.normalized_laplacian(affinity).laplacian,
                                      k=3, model_device=torch.device(device))
    eigenvalues = spectrum.eigenvalues
    g = dynamics.pool_context(x)
    g.retain_grad()
    upstream = (torch.sin(torch.arange(2*3*32*4, device=device).float()*.27)*.7+
                .1).reshape(2, 3, 32, 2, 2)
    expected, gradients, h_ref = free_reference(dynamics, x, eigenvalues, upstream)
    captured = []
    def retain_features(module, args, output):
        """只在调试钩子保留本次H导数，不在模型成员缓存。"""
        output.retain_grad()
        captured.append(output)
    handle = dynamics.condition_trunk.register_forward_hook(retain_features)
    actual = dynamics._condition_free(eigenvalues, g)
    handle.remove()
    assert tuple(f.name for f in fields(actual)) == ("raw_gamma", "gamma", "omega", "a_blocks")
    with pytest.raises(FrozenInstanceError):
        actual.gamma = actual.raw_gamma
    errors = {name: close(getattr(actual, name), value) for name, value in expected.items()}
    errors["h"] = close(captured[0], h_ref)
    assert actual.a_blocks.shape == (2, 3, 32, 2, 2)
    assert all(getattr(actual, name).dtype == torch.float32 and getattr(actual, name).device == x.device
               and getattr(actual, name).requires_grad for name in expected)
    assert (actual.gamma >= 0).all() and (actual.omega > 0).any() and (actual.omega < 0).any()
    assert not torch.equal(actual.a_blocks[..., 0, :, :], actual.a_blocks[..., 31, :, :])
    symmetric = actual.a_blocks+actual.a_blocks.transpose(-1, -2)
    errors["symmetric_part"] = close(symmetric, -2*expected["gamma"][..., None, None]*np.eye(2))
    c = np.linspace(-.7, 1.2, 2*3*64).reshape(2, 3, 32, 2)
    product = torch.einsum("bkrij,bkrj->bkri", actual.a_blocks,
                           torch.from_numpy(c.copy()).to(device=device, dtype=torch.float32))
    local = np.stack((-expected["gamma"]*c[..., 0]-expected["omega"]*c[..., 1],
                       expected["omega"]*c[..., 0]-expected["gamma"]*c[..., 1]), -1)
    errors["local_Ac"] = close(product, local)
    (actual.a_blocks*upstream).sum().backward()
    layer = dynamics.condition_trunk[0]
    actual_gradients = dict(g=g.grad, x=x.grad, h=captured[0].grad,
        trunk_weight=layer.weight.grad, trunk_bias=layer.bias.grad,
        gamma_weight=dynamics.gamma_head.weight.grad, gamma_bias=dynamics.gamma_head.bias.grad,
        omega_weight=dynamics.omega_head.weight.grad, omega_bias=dynamics.omega_head.bias.grad)
    gradients.update(raw_z=np.einsum("bnc,ce->bne", gradients["x"], numpy64(model.p_in.weight)),
        p_in_weight=np.einsum("bnc,bne->ce", gradients["x"], numpy64(raw_z)),
        p_in_bias=gradients["x"].sum(axis=(0, 1)))
    actual_gradients.update(raw_z=raw_z.grad, p_in_weight=model.p_in.weight.grad,
                            p_in_bias=model.p_in.bias.grad)
    for name, value in actual_gradients.items():
        errors[name+"_grad"] = close(value, gradients[name])
        assert torch.count_nonzero(value) > 0
    assert not eigenvalues.requires_grad and eigenvalues.grad is None
    assert not spectrum.basis.requires_grad and model.p_out.weight.grad is None
    permutation = torch.tensor([2, 0, 1], device=device)
    permuted = dynamics._condition_free(eigenvalues[:, permutation], g)
    for name in expected:
        errors[name+"_permutation"] = close(getattr(permuted, name), numpy64(getattr(actual, name)[:, permutation]))
    repeated = dynamics._condition_free(eigenvalues[:, :1].expand(-1, 2), g)
    for name in expected:
        assert torch.equal(getattr(repeated, name)[:, 0], getattr(repeated, name)[:, 1])
    responses = dict(lambda_response=float((dynamics._condition_free(eigenvalues+.4, g).a_blocks-actual.a_blocks).abs().max()),
                     g_response=float((dynamics._condition_free(eigenvalues, g+.3).a_blocks-actual.a_blocks).abs().max()))
    assert min(responses.values()) > 1e-4
    print(json.dumps(dict(probe="D01-real-chain", device=device, fictional_action_dim=2, errors=errors, **responses)))


def test_free_heads_initialization_calls_and_unchanged_trunk_prefix(monkeypatch):
    """核实Normal实际调用、非零权重、零H阻尼和C06同种子干路；样本std不强求相等。"""
    normal, calls, active = nn.init.normal_, [], False
    def record_normal(tensor, mean=0., std=1., generator=None):
        """记录真实调用；默认device的__torch_function__分派重入不重复计数。"""
        nonlocal active
        if active:
            return normal(tensor, mean=mean, std=std, generator=generator)
        calls.append((tensor, mean, std))
        active = True
        try:
            return normal(tensor, mean=mean, std=std, generator=generator)
        finally:
            active = False
    monkeypatch.setattr(nn.init, "normal_", record_normal)
    torch.manual_seed(1703)
    trunk = nn.Sequential(nn.Linear(65, 64, bias=True, device="cpu", dtype=torch.float32), nn.SiLU())
    nn.init.xavier_uniform_(trunk[0].weight, gain=1.)
    nn.init.zeros_(trunk[0].bias)
    torch.manual_seed(1703)
    dynamics = api().ModalDynamics(config())
    assert torch.equal(trunk[0].weight, dynamics.condition_trunk[0].weight)
    assert torch.equal(trunk[0].bias, dynamics.condition_trunk[0].bias)
    head_weights = (dynamics.gamma_head.weight, dynamics.omega_head.weight, dynamics.b_head.weight)
    head_calls = tuple(call for call in calls if any(call[0] is weight for weight in head_weights))
    assert len(head_calls) == 3
    for (tensor, mean, std), head in zip(head_calls, (dynamics.gamma_head, dynamics.omega_head, dynamics.b_head)):
        assert tensor is head.weight and mean == 0. and std == config().head_weight_std
        width = 64*dynamics.config.action_dim if head is dynamics.b_head else 32
        assert head.weight.shape == (width, 64) and head.bias.shape == (width,)
        assert torch.count_nonzero(head.weight) > 0 and torch.isfinite(head.weight).all()
    assert torch.count_nonzero(dynamics.omega_head.bias) == 0
    assert torch.count_nonzero(dynamics.b_head.bias) == 0
    close(torch.nn.functional.softplus(dynamics.gamma_head.bias), np.full(32, .1))
    zero = dynamics._condition_free(torch.zeros(1, 1), torch.zeros(1, 64))
    close(zero.gamma, np.full((1, 1, 32), .1))
    assert torch.count_nonzero(zero.omega) == 0
    assert sum(p.numel() for p in dynamics.gamma_head.parameters()) == 2080
    assert sum(p.numel() for p in dynamics.omega_head.parameters()) == 2080
    other_m = api().ModalDynamics(config(action_dim=9))
    assert [(n, p.shape) for n, p in other_m.named_parameters() if n.startswith(("condition_trunk.", "gamma_head.", "omega_head."))] == [
        (n, p.shape) for n, p in dynamics.named_parameters() if n.startswith(("condition_trunk.", "gamma_head.", "omega_head."))]
    assert other_m.b_head.weight.shape == (64*9, 64) and other_m.b_head.bias.shape == (64*9,)
    assert sum(p.numel() for p in stage_parameters(other_m)) == 8384
    print(json.dumps(dict(probe="D01-initialization", stage_parameters=8384, trunk=4224, heads=4160,
        normal_calls=[dict(mean=mean, std=std, shape=list(tensor.shape)) for tensor, mean, std in head_calls])))


@pytest.mark.parametrize("damping,std", [(1e-30, 2e-4), (1., .02), (1000., .003)])
def test_free_initialization_uses_nondefault_finite_positive_values(damping, std):
    """消费端允许可表示的非默认值；大阻尼用同一逆softplus公式稳定初始化。"""
    dynamics = api().ModalDynamics(config(initial_damping=damping, head_weight_std=std))
    actual = dynamics._condition_free(torch.zeros(1, 1), torch.zeros(1, 64))
    torch.testing.assert_close(actual.gamma, torch.full((1, 1, 32), damping), atol=0., rtol=1e-4)
    assert all(torch.isfinite(p).all() for p in dynamics.parameters())


@pytest.mark.parametrize("name", ["initial_damping", "head_weight_std"])
@pytest.mark.parametrize("value", [1e-100, 1e100])
def test_free_initialization_rejects_values_unrepresentable_as_positive_float32(name, value):
    """配置原校验保持，实际消费时拒绝float32零/无穷；不裁剪或悄改目标。"""
    cfg = config(**{name: value})
    assert getattr(cfg, name) == value
    with pytest.raises(ValueError):
        api().ModalDynamics(cfg)


def test_free_initialization_rejects_nonfinite_random_parameters(monkeypatch):
    """实际初始化若非有限必须失败，不能只验证输入std。"""
    def overflow(tensor, **kwargs):
        """局部故障注入模拟初始化溢出，不改变正式数值目标。"""
        with torch.no_grad():
            return tensor.fill_(math.inf)
    monkeypatch.setattr(nn.init, "normal_", overflow)
    with pytest.raises(ValueError):
        api().ModalDynamics(config())


def test_free_stable_softplus_signed_frequency_and_no_damping_floor():
    """负/零/正/极端raw用logaddexp独立核验；极负下溢0合法，不设下限。"""
    dynamics = hand_free_dynamics()
    raw = torch.tensor([-1000., -100., -3., 0., 3., 100., 1000., -30.]).repeat(4)
    omega = torch.linspace(-2., 2., 32)
    with torch.no_grad():
        dynamics.gamma_head.weight.zero_()
        dynamics.omega_head.weight.zero_()
        dynamics.gamma_head.bias.copy_(raw)
        dynamics.omega_head.bias.copy_(omega)
    actual = dynamics._condition_free(torch.zeros(1, 1), torch.zeros(1, 64))
    close(actual.raw_gamma, numpy64(raw)[None, None])
    close(actual.gamma, np.logaddexp(0., numpy64(raw))[None, None])
    close(actual.omega, numpy64(omega)[None, None])
    assert actual.gamma[0, 0, 0] == 0 and actual.gamma[0, 0, 1] > 0
    assert actual.gamma[0, 0, 7] < 1e-12 and (actual.gamma >= 0).all()
    assert all(torch.isfinite(getattr(actual, f.name)).all() for f in fields(actual))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_free_reuses_parameters_without_caches_or_forward_side_effects(device):
    """变化B/N/K及g1/g2/g1复用六参数；核输入/模式/RNG/精度/设备/成员不变。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_free_dynamics(device).eval()
    parameters = {name: (id(p), p.detach().clone()) for name, p in dynamics.named_parameters()}
    members = {name: set(vars(module)) for name, module in dynamics.named_modules()}
    modes = {name: module.training for name, module in dynamics.named_modules()}
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    defaults = torch.get_default_dtype(), torch.get_default_device()
    for b, n, k in ((2, 5, 4), (1, 1, 1), (3, 7, 2), (2, 3, 6)):
        x = torch.linspace(-.5, 1.3, b*n*64, device=device).reshape(b, n, 64)
        eigenvalues = torch.linspace(-.4, 2.4, b*k, device=device).reshape(b, k)
        saved = x.clone(), eigenvalues.clone()
        g1, g2 = dynamics.pool_context(x), dynamics.pool_context(x+1)
        a1, a2 = dynamics._condition_free(eigenvalues, g1), dynamics._condition_free(eigenvalues, g2)
        repeated = dynamics._condition_free(eigenvalues, g1)
        for f in fields(a1):
            close(getattr(repeated, f.name), numpy64(getattr(a1, f.name)))
        assert not torch.allclose(a1.a_blocks, a2.a_blocks, atol=1e-5, rtol=1e-4)
        assert a1.a_blocks.shape == (b, k, 32, 2, 2)
        assert torch.equal(x, saved[0]) and torch.equal(eigenvalues, saved[1])
        assert torch.equal(g1, saved[0].mean(1))
    assert members == {name: set(vars(module)) for name, module in dynamics.named_modules()}
    assert tuple(parameters) == tuple(dict(dynamics.named_parameters()))
    for name, p in dynamics.named_parameters():
        assert id(p) == parameters[name][0] and torch.equal(p, parameters[name][1]) and p.grad is None
    assert {name: m.training for name, m in dynamics.named_modules()} == modes
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())
    assert list(dynamics.named_buffers()) == []


def test_free_noncontiguous_inputs_and_inference_preserve_grad_modes():
    """非连续lambda/g与B=K=1合法；no_grad/inference保持调用者模式。"""
    dynamics = hand_free_dynamics()
    eigenvalues = torch.arange(12).float().reshape(6, 2).T/8
    pooled = torch.linspace(-.4, .8, 128).reshape(64, 2).T.requires_grad_()
    assert not eigenvalues.is_contiguous() and not pooled.is_contiguous()
    actual = dynamics._condition_free(eigenvalues, pooled)
    assert actual.a_blocks.shape == (2, 6, 32, 2, 2)
    actual.a_blocks.sum().backward()
    assert torch.count_nonzero(pooled.grad) > 0
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            out = dynamics._condition_free(eigenvalues, pooled)
            assert not torch.is_grad_enabled() and not out.a_blocks.requires_grad
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()


@pytest.mark.parametrize("lshape,gshape", [((0, 3), (0, 64)), ((2, 0), (2, 64)), ((2,), (2, 64)),
            ((2, 3, 1), (2, 64)), ((2, 3), (1, 64)), ((2, 3), (2, 63)), ((2, 3), (2, 1, 64))])
def test_free_reuses_shape_validation_without_batch_broadcast(lshape, gshape):
    """A分支仍拒绝非法shape与batch广播，不修复lambda。"""
    with pytest.raises(ValueError):
        hand_free_dynamics()._condition_free(torch.zeros(lshape), torch.zeros(gshape))


@pytest.mark.parametrize("target", ["eigenvalues", "pooled"])
@pytest.mark.parametrize("kind", ["none", "double", "half", "integer", "complex", "sparse", "meta", "nan", "inf", "nested"])
def test_free_reuses_invalid_input_validation(target, kind):
    """A边界复用C06实际验证，拒绝错误类型/精度/设备/有限性/非稠密数据。"""
    args = dict(eigenvalues=torch.ones(2, 3), pooled=torch.ones(2, 64))
    value = args[target]
    conversions = dict(none=lambda t: None, double=lambda t: t.double(), half=lambda t: t.half(),
        integer=lambda t: t.long(), complex=lambda t: t.to(torch.complex64), sparse=lambda t: t.to_sparse(),
        meta=lambda t: t.to("meta"), nan=lambda t: t*math.nan, inf=lambda t: t*math.inf,
        nested=lambda t: torch.nested.nested_tensor([t[0], t[1]], dtype=torch.float32))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")
        args[target] = conversions[kind](value)
    with pytest.raises(TypeError if kind == "none" else ValueError):
        hand_free_dynamics()._condition_free(**args)


def test_free_rejects_trainable_lambda_and_external_autocast():
    """lambda梯度属于合同错误；外部AMP不能暗改本步float32。"""
    dynamics = hand_free_dynamics()
    with pytest.raises(ValueError):
        dynamics._condition_free(torch.ones(1, 1, requires_grad=True), torch.ones(1, 64))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError):
            dynamics._condition_free(torch.ones(1, 1), torch.ones(1, 64))


@pytest.mark.parametrize("head_name", ["gamma_head", "omega_head"])
@pytest.mark.parametrize("kind", ["double", "meta", "bias_double", "bias_meta", "nonfinite", "overflow"])
def test_free_checks_each_head_precision_device_and_output(head_name, kind):
    """分别核双头权重/偏置精度和实际设备；有限输入产生非有限输出也明确失败。"""
    dynamics = hand_free_dynamics()
    head = getattr(dynamics, head_name)
    if kind in ("double", "meta"):
        setattr(dynamics, head_name, head.double() if kind == "double" else head.to("meta"))
    elif kind in ("bias_double", "bias_meta"):
        head.bias = nn.Parameter(head.bias.double() if kind == "bias_double" else head.bias.to("meta"))
    else:
        with torch.no_grad():
            head.weight.fill_(math.inf if kind == "nonfinite" else torch.finfo(torch.float32).max)
    with pytest.raises(ValueError):
        dynamics._condition_free(torch.ones(1, 1), torch.ones(1, 64))


def test_free_rejects_trunk_mismatch_and_actual_cuda_head_device_mismatch():
    """核干路错精度；可用CUDA时实际检查双头与干路跨设备，绝不隐式拷贝。"""
    dynamics = hand_free_dynamics()
    dynamics.condition_trunk.double()
    with pytest.raises(ValueError):
        dynamics._condition_free(torch.ones(1, 1), torch.ones(1, 64))
    if not torch.cuda.is_available():
        pytest.skip("CUDA head device mismatch unavailable; trunk check ran")
    for name in ("gamma_head", "omega_head"):
        dynamics = hand_free_dynamics()
        getattr(dynamics, name).cuda()
        with pytest.raises(ValueError):
            dynamics._condition_free(torch.ones(1, 1), torch.ones(1, 64))


def propagation_inputs(dtype=torch.float64, device="cpu", b=3, k=2):
    """虚构独立C/率；不同batch、模态、32通道对和有符号omega暴露轴混淆。"""
    c = torch.linspace(-.8, 1.1, b*k*64, dtype=dtype, device=device).reshape(b, k, 64)
    gamma = torch.linspace(0., .7, b*k*32, dtype=dtype, device=device).reshape(b, k, 32)
    omega = torch.sin(torch.arange(b*k*32, dtype=dtype, device=device)*.37).reshape(b, k, 32)
    duration = torch.linspace(0., 1.2, b, dtype=torch.float64, device="cpu")
    return dict(coefficients=c, gamma=gamma, omega=omega, duration=duration)


def propagation_reference(coefficients, gamma, omega, duration, upstream):
    """独立NumPy double闭式和四项解析导数；不读取生产传播输出。"""
    c = numpy64(coefficients).reshape(*gamma.shape, 2)
    g, w = numpy64(gamma), numpy64(omega)
    t = numpy64(duration)[:, None, None]
    weight = numpy64(upstream).reshape(*gamma.shape, 2)
    decay, cosine, sine = np.exp(-g*t), np.cos(w*t), np.sin(w*t)
    y0 = decay*(cosine*c[..., 0]-sine*c[..., 1])
    y1 = decay*(sine*c[..., 0]+cosine*c[..., 1])
    dc0 = decay*(cosine*weight[..., 0]+sine*weight[..., 1])
    dc1 = decay*(-sine*weight[..., 0]+cosine*weight[..., 1])
    return (np.stack((y0, y1), -1).reshape(coefficients.shape),
            np.stack((dc0, dc1), -1).reshape(coefficients.shape),
            -t*(weight[..., 0]*y0+weight[..., 1]*y1),
            t*(-weight[..., 0]*y1+weight[..., 1]*y0))


def close64(actual, expected):
    """沿用A03解析double容差1e-10/1e-8，输出最大误差供原始日志记录。"""
    expected = torch.as_tensor(expected, dtype=torch.float64, device="cpu")
    actual = actual.detach().cpu()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    return float((actual-expected).abs().max())


def test_damped_rotation_matches_matrix_exp_and_fixed_semigroup():
    """T-D2式11/12与附录：J²=-I、J转置=-J及固定A可交换；不是完整Phi。"""
    dynamics = hand_free_dynamics()
    assert hasattr(dynamics, "free_propagate"), "D02 free_propagate 尚未实现"
    args = propagation_inputs()
    j = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    assert torch.equal(j@j, -identity) and torch.equal(j.T, -j)
    damping = -args["gamma"][..., None, None]*identity
    rotation = args["omega"][..., None, None]*j
    assert torch.equal(damping@rotation, rotation@damping)
    a = damping+rotation
    exponential = torch.matrix_exp(a*args["duration"][:, None, None, None, None])
    expected = (exponential@args["coefficients"].reshape(3, 2, 32, 2, 1)).reshape(3, 2, 64)
    actual = dynamics.free_propagate(**args)
    errors = dict(matrix_exp=close64(actual, expected))
    sigma = torch.tensor([.4, 0., .6], dtype=torch.float64)
    composed = dynamics.free_propagate(actual, gamma=args["gamma"], omega=args["omega"], duration=sigma)
    direct = dynamics.free_propagate(**dict(args, duration=args["duration"]+sigma))
    errors["fixed_semigroup"] = close64(composed, direct)
    assert actual.dtype == torch.float64 and actual.device.type == "cpu"
    print(json.dumps(dict(probe="T-D2", errors=errors, fixed_coefficients=True)))


def test_propagation_recovers_blocks_and_nonexpansive_spectral_norm():
    """式13：用生产方法恢复两列，SVD核每块与整块最大值，状态欧氏范数不增。"""
    dynamics = hand_free_dynamics()
    args = propagation_inputs()
    c0, c1 = torch.zeros_like(args["coefficients"]), torch.zeros_like(args["coefficients"])
    c0[..., 0::2], c1[..., 1::2] = 1., 1.
    col0 = dynamics.free_propagate(**dict(args, coefficients=c0)).reshape(3, 2, 32, 2)
    col1 = dynamics.free_propagate(**dict(args, coefficients=c1)).reshape(3, 2, 32, 2)
    blocks = torch.stack((col0, col1), -1)
    gamma, omega, t = numpy64(args["gamma"]), numpy64(args["omega"]), numpy64(args["duration"])[:, None, None]
    decay, angle = np.exp(-gamma*t), omega*t
    reference_blocks = np.empty((3, 2, 32, 2, 2))
    reference_blocks[..., 0, 0] = reference_blocks[..., 1, 1] = decay*np.cos(angle)
    reference_blocks[..., 0, 1], reference_blocks[..., 1, 0] = -decay*np.sin(angle), decay*np.sin(angle)
    errors = dict(blocks=close64(blocks, reference_blocks),
                  block_norm=close64(torch.linalg.svdvals(blocks)[..., 0], decay))
    full_norm = torch.stack([torch.linalg.svdvals(torch.block_diag(*blocks[b].reshape(-1, 2, 2)))[0] for b in range(3)])
    errors["full_max_norm"] = close64(full_norm, decay.reshape(3, -1).max(axis=1))
    assert (full_norm <= 1.+1e-10).all()
    actual = dynamics.free_propagate(**args)
    assert (torch.linalg.vector_norm(actual, dim=(1, 2)) <=
            torch.linalg.vector_norm(args["coefficients"], dim=(1, 2))+1e-10).all()
    print(json.dumps(dict(probe="D02-spectral-norm", errors=errors)))


@pytest.mark.parametrize("dtype,device", [(torch.float64, "cpu"), (torch.float32, "cpu"), (torch.float32, "cuda")])
def test_propagation_independent_analytic_gradients(dtype, device):
    """式11对C/gamma/omega独立解析反向；F64工作转换保图，CPU和可用CUDA实测。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_free_dynamics()
    args = propagation_inputs(dtype, device)
    for name in ("coefficients", "gamma", "omega"):
        args[name].requires_grad_()
    upstream = torch.cos(torch.arange(3*2*64, device=device, dtype=dtype)*.13).reshape(3, 2, 64)
    expected = propagation_reference(**args, upstream=upstream)
    actual = dynamics.free_propagate(**args)
    (actual*upstream).sum().backward()
    compare = close64 if dtype == torch.float64 else close
    errors = {name: compare(value, ref) for name, value, ref in zip(
        ("forward", "c_grad", "gamma_grad", "omega_grad"),
        (actual, args["coefficients"].grad, args["gamma"].grad, args["omega"].grad), expected)}
    assert torch.equal(args["coefficients"].grad[0], upstream[0])
    assert torch.count_nonzero(args["gamma"].grad[0]) == torch.count_nonzero(args["omega"].grad[0]) == 0
    assert torch.count_nonzero(args["gamma"].grad[1:]) and torch.count_nonzero(args["omega"].grad[1:])
    assert all(p.grad is None for p in dynamics.parameters())
    assert actual.dtype == dtype and actual.device == args["coefficients"].device
    print(json.dumps(dict(probe="D02-analytic-gradient", dtype=str(dtype), device=device, errors=errors)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_propagation_real_condition_chain_reaches_heads_pool_project_in_and_raw_z(device):
    """独立率导数接D01链式参考，真实C01→Pool→两头→传播到raw Z；C独立不混路径。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_free_dynamics(device)
    graph = GraphSpectrum(GraphConfig(16, 1703, .25, 1., 128, 1e-10, 1e-8)).to(device)
    torch.manual_seed(1703)
    model = POJEPA(ProjectionConfig(3, 64), graph, dynamics, object()).to(device)
    raw_z = torch.linspace(-.4, .9, 36, device=device).reshape(2, 6, 3).requires_grad_()
    x = model.project_in(raw_z)
    x.retain_grad()
    spectrum = graph.lowest_eigenpairs(graph.normalized_laplacian(graph.affinity(x, regular_grid(2, 3))).laplacian,
                                      k=3, model_device=torch.device(device))
    g = dynamics.pool_context(x)
    g.retain_grad()
    captured = []
    def retain_features(module, inputs, output):
        """单测保存H以核解析梯度，模型本身不缓存。"""
        output.retain_grad()
        captured.append(output)
    handle = dynamics.condition_trunk.register_forward_hook(retain_features)
    rates = dynamics._condition_free(spectrum.eigenvalues, g)
    handle.remove()
    rates.gamma.retain_grad()
    rates.omega.retain_grad()
    args = propagation_inputs(torch.float32, device, b=2, k=3)
    args["duration"] = torch.tensor([.4, 1.1], dtype=torch.float64)
    args.update(gamma=rates.gamma, omega=rates.omega)
    upstream = torch.sin(torch.arange(384, device=device).float()*.19).reshape(2, 3, 64)+.2
    rate_values, _, _ = free_reference(dynamics, x, spectrum.eigenvalues, torch.zeros(2, 3, 32, 2, 2, device=device))
    expected, _, dgamma, domega = propagation_reference(**dict(args,
        gamma=torch.from_numpy(rate_values["gamma"]), omega=torch.from_numpy(rate_values["omega"])), upstream=upstream)
    a_upstream = np.zeros((2, 3, 32, 2, 2))
    a_upstream[..., 0, 0], a_upstream[..., 1, 0] = -dgamma, domega
    _, gradients, _ = free_reference(dynamics, x, spectrum.eigenvalues, torch.from_numpy(a_upstream))
    gradients.update(raw_z=np.einsum("bnc,ce->bne", gradients["x"], numpy64(model.p_in.weight)),
        p_in_weight=np.einsum("bnc,bne->ce", gradients["x"], numpy64(raw_z)), p_in_bias=gradients["x"].sum(axis=(0, 1)))
    actual = dynamics.free_propagate(**args)
    (actual*upstream).sum().backward()
    layer = dynamics.condition_trunk[0]
    actual_gradients = dict(g=g.grad, x=x.grad, h=captured[0].grad, trunk_weight=layer.weight.grad,
        trunk_bias=layer.bias.grad, gamma_weight=dynamics.gamma_head.weight.grad, gamma_bias=dynamics.gamma_head.bias.grad,
        omega_weight=dynamics.omega_head.weight.grad, omega_bias=dynamics.omega_head.bias.grad,
        raw_z=raw_z.grad, p_in_weight=model.p_in.weight.grad, p_in_bias=model.p_in.bias.grad)
    errors = dict(forward=close(actual, expected), gamma_grad=close(rates.gamma.grad, dgamma), omega_grad=close(rates.omega.grad, domega))
    for name, value in actual_gradients.items():
        errors[name] = close(value, gradients[name])
        assert torch.count_nonzero(value) > 0
    assert not spectrum.eigenvalues.requires_grad and spectrum.eigenvalues.grad is None
    assert not spectrum.basis.requires_grad and model.p_out.weight.grad is None
    print(json.dumps(dict(probe="D02-real-condition-chain", device=device, errors=errors)))


@pytest.mark.parametrize("zero", ["gamma", "omega", "both", "duration", "underflow"])
def test_propagation_zero_limits_and_legal_decay_underflow(zero):
    """纯旋转、纯衰减、全零与零时长均走原闭式；有限gamma*t下合法exp下溢不设下限。"""
    args = propagation_inputs(b=1, k=1)
    args["duration"].fill_(.8)
    if zero in ("gamma", "both"):
        args["gamma"].zero_()
    if zero in ("omega", "both"):
        args["omega"].zero_()
    if zero == "duration":
        args["duration"].zero_()
    if zero == "underflow":
        args["gamma"].fill_(10000.)
    expected = propagation_reference(**args, upstream=torch.ones_like(args["coefficients"]))[0]
    actual = hand_free_dynamics().free_propagate(**args)
    close64(actual, expected)
    if zero in ("both", "duration"):
        assert torch.equal(actual, args["coefficients"])
    if zero == "underflow":
        assert torch.count_nonzero(actual) == 0


def test_propagation_noncontiguous_inputs_and_no_call_side_effects():
    """显式输入与网络无关；非连续合法，保持输入/参数/成员/模式/RNG/default和grad模式。"""
    dynamics = hand_free_dynamics().eval()
    args = propagation_inputs()
    for name in ("coefficients", "gamma", "omega"):
        value = args[name]
        args[name] = value.transpose(1, 2).contiguous().transpose(1, 2).requires_grad_()
        assert not args[name].is_contiguous()
    args["duration"] = torch.tensor([0., -9., .6, -9., 1.2, -9.], dtype=torch.float64)[::2]
    assert not args["duration"].is_contiguous()
    saved = {name: value.clone() for name, value in args.items()}
    parameters = {name: (id(p), p.detach().clone()) for name, p in dynamics.named_parameters()}
    members = {name: set(vars(m)) for name, m in dynamics.named_modules()}
    modes = {name: m.training for name, m in dynamics.named_modules()}
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    defaults = torch.get_default_dtype(), torch.get_default_device()
    expected = propagation_reference(**args, upstream=torch.ones_like(args["coefficients"]))[0]
    close64(dynamics.free_propagate(**args), expected)
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            actual = dynamics.free_propagate(**args)
            assert not actual.requires_grad and not torch.is_grad_enabled()
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert all(torch.equal(args[name], value) for name, value in saved.items())
    assert members == {name: set(vars(m)) for name, m in dynamics.named_modules()}
    assert modes == {name: m.training for name, m in dynamics.named_modules()}
    assert tuple(parameters) == tuple(dict(dynamics.named_parameters())) and sum(p.numel() for p in stage_parameters(dynamics)) == 8384
    assert all(id(p) == parameters[name][0] and torch.equal(p, parameters[name][1]) for name, p in dynamics.named_parameters())
    assert list(dynamics.named_buffers()) == [] and torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())
    for target in ("double", "meta"):
        network = dynamics.double() if target == "double" else dynamics.to("meta")
        close64(network.free_propagate(**args), expected)


@pytest.mark.parametrize("target", ["coefficients", "gamma", "omega", "duration"])
@pytest.mark.parametrize("kind", ["none", "wrong_precision", "half", "integer", "complex", "sparse", "meta", "nan", "inf", "nested"])
def test_propagation_rejects_invalid_tensor_contract(target, kind):
    """D02自己的稠密实数边界，CPU double解析例外不放宽现有网络验证器。"""
    args = propagation_inputs(torch.float32)
    value = args[target]
    conversions = dict(none=lambda t: None,
        wrong_precision=lambda t: t.float() if target == "duration" else t.double(), half=lambda t: t.half(),
        integer=lambda t: t.long(), complex=lambda t: t.to(torch.complex64), sparse=lambda t: t.to_sparse(),
        meta=lambda t: t.to("meta"), nan=lambda t: t*math.nan, inf=lambda t: t*math.inf,
        nested=lambda t: torch.nested.nested_tensor([t[0], t[1]], dtype=t.dtype))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")
        args[target] = conversions[kind](value)
    with pytest.raises(TypeError if kind == "none" else ValueError):
        hand_free_dynamics().free_propagate(**args)


@pytest.mark.parametrize("target,shape", [("coefficients", (0, 2, 64)), ("coefficients", (3, 0, 64)),
    ("coefficients", (3, 2, 63)), ("coefficients", (3, 64)), ("coefficients", (3, 1, 2, 64)),
    ("gamma", (1, 2, 32)), ("gamma", (3, 1, 32)), ("gamma", (3, 2, 31)), ("gamma", (3, 2, 1, 32)),
    ("omega", (3, 1, 32)), ("omega", (3, 2, 33)), ("omega", (3, 2)),
    ("duration", ()), ("duration", (3, 1)), ("duration", (1,)), ("duration", (0,))])
def test_propagation_rejects_wrong_shapes_without_implicit_broadcast(target, shape):
    """只按严格[B]时间广播；batch/模态/32通道对不能静默广播或合并。"""
    args = propagation_inputs()
    args[target] = torch.ones(shape, dtype=torch.float64)
    with pytest.raises(ValueError):
        hand_free_dynamics().free_propagate(**args)


@pytest.mark.parametrize("bad", ["negative_gamma", "negative_duration", "duration_gradient", "gamma_work_overflow", "omega_work_overflow", "output_overflow"])
def test_propagation_rejects_negative_metadata_and_finite_work_or_output_overflow(bad):
    """非负无容差放行；有限输入的工作乘积/输出溢出须拒绝，不能裁剪掩盖。"""
    args = propagation_inputs(b=1, k=1)
    args["duration"].fill_(2.)
    if bad == "negative_gamma":
        args["gamma"][0, 0, 0] = -torch.finfo(torch.float64).tiny
    elif bad == "negative_duration":
        args["duration"].fill_(-torch.finfo(torch.float64).tiny)
    elif bad == "duration_gradient":
        args["duration"].requires_grad_()
    elif bad in ("gamma_work_overflow", "omega_work_overflow"):
        args["gamma" if bad.startswith("gamma") else "omega"].fill_(torch.finfo(torch.float64).max)
    else:
        args["coefficients"].fill_(torch.finfo(torch.float64).max)
        args["gamma"].zero_()
        args["omega"].fill_(math.pi/8)
    with pytest.raises(ValueError):
        hand_free_dynamics().free_propagate(**args)


def test_propagation_actual_cuda_device_boundary_and_cpu_double_exception():
    """CUDA可用时实际拒绝混设备、设备时间和GPU double；CPU double无需搬网络。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_free_dynamics("cuda")
    args = propagation_inputs()
    close64(dynamics.free_propagate(**args), propagation_reference(**args, upstream=torch.ones_like(args["coefficients"]))[0])
    for target in args:
        mixed = dict(args, **{target: args[target].cuda()})
        with pytest.raises(ValueError):
            dynamics.free_propagate(**mixed)
    with pytest.raises(ValueError):
        dynamics.free_propagate(**propagation_inputs(torch.float64, "cuda"))


def hand_complete_dynamics(device="cpu", r=2):
    """D09虚构非退化全分支权重；复用各模块已验收的受控参数，不改生产初值。"""
    dynamics = api().ModalDynamics(config(top_r=r)).to(device)
    dynamics.load_state_dict(hand_conditioned_dynamics(device).state_dict())
    for source, names in ((hand_content_dynamics(device), ("q_proj", "k_proj", "v_proj")),
            (hand_spectral_dynamics(device), ("spectral_bias_net",)),
            (hand_spatial_dynamics(device), ("spatial_bias_net",)),
            (hand_action_residual_dynamics(device), ("action_residual_net",))):
        for name in names:
            getattr(dynamics, name).load_state_dict(getattr(source, name).state_dict())
    for name in ("prepare_context", "free_rhs", "nonlinear", "rhs"):
        assert hasattr(dynamics, name), f"D09 {name} 尚未实现"
    return dynamics


def complete_inputs(device="cpu", batch=2, modes=3):
    """虚构六点DCT正交列、CPU双精度网格和当前C/a，均非正式实验数据。"""
    x = torch.cos(torch.arange(batch*6*64, device=device).float()*.031).reshape(batch, 6, 64)*.5
    eigenvalues = torch.linspace(.1, 1.4, batch*modes, device=device).reshape(batch, modes)
    i, k = np.arange(6)[:, None], np.arange(modes)[None, :]
    psi = np.cos(math.pi*(i+.5)*k/6)*math.sqrt(2/6)
    psi[:, 0] /= math.sqrt(2)
    basis = torch.tensor(np.repeat(psi[None], batch, axis=0), dtype=torch.float32, device=device)
    c = torch.sin(torch.arange(batch*modes*64, device=device).float()*.071).reshape(batch, modes, 64)
    action = torch.linspace(-.6, .8, batch*2, device=device).reshape(batch, 2)
    return x, eigenvalues, basis, regular_grid(2, 3).coords, c, action


def complete_reference(dynamics, x, eigenvalues, basis, coords, c, action, upstream=None):
    """组合已有NumPy独立公式及解析pullback；本次lambda/Psi固定，不重求谱。"""
    cn, an = numpy64(c), numpy64(action)
    gy = np.zeros_like(cn) if upstream is None else numpy64(upstream)
    paired, paired_gy = cn.reshape(*cn.shape[:2], 32, 2), gy.reshape(*cn.shape[:2], 32, 2)
    da = paired_gy[..., :, None]*paired[..., None, :]
    rates, free_grad, h = free_reference(dynamics, x, eigenvalues, torch.from_numpy(da))
    wb, bb = numpy64(dynamics.b_head.weight), numpy64(dynamics.b_head.bias)
    b_matrix = (h@wb.T+bb).reshape(*cn.shape[:2], 64, 2)
    free = np.einsum("bkrpq,bkrq->bkrp", rates["a_blocks"], paired).reshape(cn.shape)
    linear = np.einsum("bkcm,bm->bkc", b_matrix, an)
    zeros = torch.zeros(*cn.shape[:2], cn.shape[1])
    content, _ = content_reference(c, dynamics, zeros, torch.zeros(cn.shape))
    _, spectral, _ = spectral_bias_reference(eigenvalues, dynamics, zeros)
    descriptors = spatial_reference(basis, coords)[2]
    spatial, _ = spatial_mlp_reference(torch.from_numpy(descriptors), dynamics, zeros)
    total = content[3]+spectral+spatial
    selected, scores, weights, aggregate, ds, dv = sparse_reference(torch.from_numpy(total),
        torch.from_numpy(content[2]), dynamics.config.top_r, torch.from_numpy(gy))
    residual, residual_grad = action_residual_reference(c, action, dynamics, torch.from_numpy(gy))
    expected = dict(pooled=numpy64(x).mean(axis=1), b_matrix=b_matrix, free=free,
        action_linear=linear, query=content[0], key=content[1], value=content[2], content_scores=content[3],
        spectral_scores=spectral, spatial_scores=spatial, total_scores=total, neighbor_index=selected,
        selected_scores=scores, weights=weights, aggregate=aggregate, action_residual=residual,
        nonlinear=aggregate+residual, derivative=free+linear+aggregate+residual, **rates)
    if upstream is None:
        return expected, {}
    _, content_grad = content_reference(c, dynamics, torch.from_numpy(ds), torch.from_numpy(dv))
    _, _, spectral_grad = spectral_bias_reference(eigenvalues, dynamics, torch.from_numpy(ds))
    _, spatial_grad = spatial_mlp_reference(torch.from_numpy(descriptors), dynamics, torch.from_numpy(ds))
    db = gy[..., None]*an[:, None, None, :]
    dflat = db.reshape(*cn.shape[:2], 128)
    dh = free_grad["h"]+dflat@wb
    _, _, _, dg, dx, dw, d_bias = reference(numpy64(x), numpy64(eigenvalues),
        numpy64(dynamics.condition_trunk[0].weight), numpy64(dynamics.condition_trunk[0].bias), dh)
    dc_free = np.einsum("bkrp,bkrpq->bkrq", paired_gy, rates["a_blocks"]).reshape(cn.shape)
    gradients = dict(c=dc_free+content_grad["c"]+residual_grad["c"],
        action=np.einsum("bkc,bkcm->bm", gy, b_matrix)+residual_grad["action"],
        x=dx, pooled=dg, a_blocks=da, b_matrix=db, spectral_bias=ds, spatial_bias=ds)
    parameters = dict(zip(("condition_trunk.0.weight", "condition_trunk.0.bias", "gamma_head.weight",
        "gamma_head.bias", "omega_head.weight", "omega_head.bias", "b_head.weight", "b_head.bias"),
        (dw, d_bias, free_grad["gamma_weight"], free_grad["gamma_bias"], free_grad["omega_weight"],
         free_grad["omega_bias"], dflat.reshape(-1, 128).T@h.reshape(-1, 64), dflat.sum(axis=(0, 1)))))
    parameters.update({name+"_proj.weight": content_grad[name+"_weight"] for name in ("q", "k", "v")})
    for prefix, values in (("spectral_bias_net", spectral_grad), ("spatial_bias_net", spatial_grad),
            ("action_residual_net", [residual_grad[key] for key in
                ("hidden_weight", "hidden_bias", "output_weight", "output_bias")])):
        parameters.update({prefix+suffix: value for suffix, value in
            zip((".0.weight", ".0.bias", ".2.weight", ".2.bias"), values)})
    gradients["parameters"] = parameters
    return expected, gradients


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_rhs_keeps_three_scores_ba_and_action_residual(device):
    """T-D7完整独立参考：g/A/B、三分数先相加、真实top-r、M_a及Ac+Ba+N。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device)
    x, eigenvalues, basis, coords, c, action = complete_inputs(device)
    expected, _ = complete_reference(dynamics, x, eigenvalues, basis, coords, c, action)
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    result = dynamics.rhs(c, action, context)
    interaction = result.interaction
    observed = dict(pooled=context.pooled, b_matrix=context.coefficients.b_matrix,
        free=result.free, action_linear=result.action_linear, derivative=result.derivative,
        spectral_scores=interaction.spectral_scores, spatial_scores=interaction.spatial_scores,
        total_scores=interaction.total_scores, action_residual=interaction.action_residual,
        nonlinear=interaction.nonlinear, content_scores=interaction.content.scores)
    observed.update({name: getattr(context.coefficients, name) for name in
        ("raw_gamma", "gamma", "omega", "a_blocks")})
    observed.update({name: getattr(interaction.content, name) for name in ("query", "key", "value")})
    observed.update({name: getattr(interaction.attention, name) for name in
        ("selected_scores", "weights", "aggregate")})
    errors = {name: close(value, expected[name]) for name, value in observed.items()}
    assert np.array_equal(numpy64(interaction.attention.neighbor_index), expected["neighbor_index"])
    assert all(np.linalg.norm(expected[name]) > .001 for name in
        ("free", "action_linear", "spectral_scores", "spatial_scores", "aggregate", "action_residual"))
    assert interaction.spectral_scores is context.spectral_bias and interaction.spatial_scores is context.spatial_bias
    assert not np.allclose(expected["derivative"], expected["nonlinear"])
    print(json.dumps(dict(probe="D09-complete-independent-RHS", device=device, errors=errors)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_nonlinear_sums_three_scores_before_selecting_and_keeps_each_branch(device):
    """受控当前内容0>1>2、谱提2、空间提1；先求和top2成为[1,2]而非内容[0,1]。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device)
    x, eigenvalues, basis, coords, c, action = complete_inputs(device, batch=1)
    with torch.no_grad():
        for layer in (dynamics.q_proj, dynamics.k_proj):
            layer.weight.zero_()
            layer.weight[0, 0] = 1.
    c.zero_()
    c[0, :, 0] = torch.tensor([3., 2., 1.], device=device)
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    spectral, spatial = torch.zeros(1, 3, 3, device=device), torch.zeros(1, 3, 3, device=device)
    spectral[..., 2], spatial[..., 1] = 2., 3.
    isolated = replace(context, spectral_bias=spectral, spatial_bias=spatial)
    result = dynamics.rhs(c, action, isolated)
    interaction = result.interaction
    content, _ = content_reference(c, dynamics, torch.zeros(1, 3, 3), torch.zeros_like(c))
    expected = sparse_reference(torch.from_numpy(content[3]+numpy64(spectral)+numpy64(spatial)),
        torch.from_numpy(content[2]), 2)
    assert np.array_equal(expected[0], np.array([[[1, 2], [1, 2], [1, 2]]]))
    assert np.array_equal(numpy64(interaction.attention.neighbor_index), expected[0])
    close(interaction.attention.aggregate, expected[3])
    close(interaction.nonlinear, expected[3]+action_residual_reference(c, action, dynamics)[0])
    for field in ("spectral_bias", "spatial_bias"):
        changed = dynamics.rhs(c, action, replace(isolated, **{field: torch.zeros_like(spectral)}))
        assert not torch.equal(changed.interaction.attention.neighbor_index, interaction.attention.neighbor_index)
        assert (changed.interaction.nonlinear-interaction.nonlinear).abs().max() > 1e-4
    changed_content = dynamics.rhs(-c, action, isolated)
    assert not torch.equal(changed_content.interaction.content.value, interaction.content.value)
    for name, value in (("a_blocks", context.coefficients.a_blocks+.1),
                        ("b_matrix", context.coefficients.b_matrix+.2)):
        changed = dynamics.rhs(c, action, replace(isolated,
            coefficients=replace(context.coefficients, **{name: value})))
        assert (changed.derivative-result.derivative).abs().max() > 1e-4
    with torch.no_grad():
        dynamics.action_residual_net[2].bias.add_(.25)
    changed = dynamics.rhs(c, action, isolated)
    close(changed.derivative, numpy64(result.derivative)+.25)
    zero = dynamics.rhs(c, torch.zeros_like(action), isolated)
    assert zero.action_linear.count_nonzero() == 0
    assert zero.interaction.attention.aggregate.count_nonzero() and zero.interaction.action_residual.count_nonzero()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_context_lifecycle_uses_current_c_and_action_without_reconditioning(device, monkeypatch):
    """同一查询保持g/A/B/两bias，每次C/a重算；调用计数与独立数值共同核验。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device)
    x, eigenvalues, basis, coords, c, action = complete_inputs(device)
    calls = {}
    for name in ("pool_context", "condition", "spectral_bias", "spatial_descriptors", "spatial_bias",
                 "content_terms", "sparse_attention", "action_residual"):
        original = getattr(dynamics, name)
        def counted(*args, _name=name, _original=original, **kwargs):
            """仅测试统计真实调用，原函数完整执行。"""
            calls[_name] = calls.get(_name, 0)+1
            return _original(*args, **kwargs)
        monkeypatch.setattr(dynamics, name, counted)
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    saved = [v.detach().clone() for v in (context.pooled, context.coefficients.a_blocks,
        context.coefficients.b_matrix, context.spectral_bias, context.spatial_bias)]
    results = []
    for current_c, current_a in ((c, action), (30*c.flip(1)+.2, action), (c, action+.7)):
        result = dynamics.rhs(current_c, current_a, context)
        expected, _ = complete_reference(dynamics, x, eigenvalues, basis, coords, current_c, current_a)
        close(result.derivative, expected["derivative"])
        close(result.interaction.action_residual, expected["action_residual"])
        results.append(result)
    assert all(calls[name] == 1 for name in
        ("pool_context", "condition", "spectral_bias", "spatial_descriptors", "spatial_bias"))
    assert all(calls[name] == 3 for name in ("content_terms", "sparse_attention", "action_residual"))
    assert all(torch.equal(v, old) for v, old in zip((context.pooled, context.coefficients.a_blocks,
        context.coefficients.b_matrix, context.spectral_bias, context.spatial_bias), saved))
    assert not torch.equal(results[0].interaction.content.scores, results[1].interaction.content.scores)
    assert not torch.equal(results[0].interaction.attention.neighbor_index, results[1].interaction.attention.neighbor_index)
    assert torch.equal(results[0].interaction.content.scores, results[2].interaction.content.scores)
    assert not torch.equal(results[0].interaction.action_residual, results[2].interaction.action_residual)
    fresh = dynamics.prepare_context(x+.3, eigenvalues, basis, coords)
    assert calls["pool_context"] == calls["condition"] == 2
    assert not torch.equal(fresh.pooled, context.pooled) and not torch.equal(fresh.coefficients.a_blocks, context.coefficients.a_blocks)
    assert "context" not in vars(dynamics)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("real_chain", [False, True])
def test_complete_rhs_independent_gradients_and_real_c04_coordinate_boundary(device, real_chain, monkeypatch):
    """RHS唯一损失、一次反向核全部解析路径；真实C01→C04→C05时固定lambda/Psi。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device)
    x, eigenvalues, basis, coords, c, action = complete_inputs(device)
    model, raw_z = None, None
    if real_chain:
        torch.manual_seed(1703)
        graph = GraphSpectrum(GraphConfig(16, 1703, .25, 1., 128, 1e-10, 1e-8)).to(device)
        model = POJEPA(ProjectionConfig(5, 64), graph, dynamics, object()).to(device)
        raw_z = torch.linspace(-.8, .9, 60, device=device).reshape(2, 6, 5).requires_grad_()
        x = model.project_in(raw_z)
        spectrum = graph.lowest_eigenpairs(graph.normalized_laplacian(graph.affinity(x, regular_grid(2, 3))).laplacian,
            k=3, model_device=torch.device(device))
        eigenvalues, basis = spectrum.eigenvalues, spectrum.basis
        assert spectrum.eigenvalues_cpu.dtype == spectrum.basis_cpu.dtype == torch.float64
        c = modal_analysis(x, basis)
    else:
        x.requires_grad_()
        c.requires_grad_()
    x.retain_grad()
    c.retain_grad()
    action.requires_grad_()
    upstream = 16*(torch.cos(torch.arange(c.numel(), device=device).float()*.113).reshape_as(c)+.2)
    expected, gradients = complete_reference(dynamics, x, eigenvalues, basis, coords, c, action, upstream)
    saved_coords, coords_version = coords.clone(), coords._version
    captured_coords = []
    original = dynamics.spatial_descriptors
    def capture(actual_basis, actual_coords):
        """读取真实D06入口的坐标，仍执行实际空间描述。"""
        assert actual_basis is basis
        captured_coords.append(actual_coords)
        return original(actual_basis, actual_coords)
    monkeypatch.setattr(dynamics, "spatial_descriptors", capture)
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    assert context.eigenvalues is eigenvalues and context.basis is basis
    assert context.coords is captured_coords[0] and len(captured_coords) == 1
    assert context.coords.dtype == torch.float32 and context.coords.device == x.device
    assert torch.equal(context.coords, coords.to(device=x.device, dtype=torch.float32)) and torch.equal(coords, saved_coords)
    assert coords._version == coords_version and coords.dtype == torch.float64 and coords.device.type == "cpu"
    tracked = dict(pooled=context.pooled, a_blocks=context.coefficients.a_blocks,
        b_matrix=context.coefficients.b_matrix, spectral_bias=context.spectral_bias, spatial_bias=context.spatial_bias)
    for value in tracked.values():
        assert value.requires_grad and value.grad_fn is not None
        value.retain_grad()
    result = dynamics.rhs(c, action, context)
    close(result.derivative, expected["derivative"])
    (result.derivative*upstream).sum().backward()
    errors = {name: close(value.grad, gradients[name]) for name, value in tracked.items()}
    errors.update(c=close(c.grad, gradients["c"]), action=close(action.grad, gradients["action"]))
    dx = gradients["x"]+(numpy64(basis)@gradients["c"] if real_chain else 0)
    errors["x"] = close(x.grad, dx)
    zero_biases = {"spectral_bias_net.2.bias", "spatial_bias_net.2.bias"}
    assert len(gradients["parameters"]) == len(tuple(dynamics.parameters())) == 23
    for name, parameter in dynamics.named_parameters():
        assert parameter.grad is not None, name
        errors[name] = close(parameter.grad, gradients["parameters"][name])
        if name in zero_biases:
            close(parameter.grad, np.zeros(parameter.shape))
        else:
            assert np.linalg.norm(gradients["parameters"][name]) > 1e-7, name
            assert parameter.grad.count_nonzero() > 0, name
    if real_chain:
        errors.update(raw_z=close(raw_z.grad, dx@numpy64(model.p_in.weight)),
            p_in_weight=close(model.p_in.weight.grad, dx.reshape(-1, 64).T@numpy64(raw_z).reshape(-1, 5)),
            p_in_bias=close(model.p_in.bias.grad, dx.sum(axis=(0, 1))))
        assert raw_z.grad.count_nonzero() and model.p_in.weight.grad.count_nonzero()
        assert model.p_out.weight.grad is None and model.p_out.bias.grad is None
    assert all(not value.requires_grad and value.grad is None for value in (eigenvalues, basis, coords, context.coords))
    print(json.dumps(dict(probe="D09-joint-analytic-gradient", device=device, real_chain=real_chain, errors=errors,
        zero_bias_actual_max={name: float(dict(dynamics.named_parameters())[name].grad.abs().max()) for name in sorted(zero_biases)},
        scores_weight_grad_norms={name: float(getattr(dynamics, name)[2].weight.grad.norm()) for name in
            ("spectral_bias_net", "spatial_bias_net")})))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("branch", ["spectral_bias_net", "spatial_bias_net"])
def test_complete_rhs_bias_scalar_shift_is_invariant_but_relative_weights_are_effective(device, branch):
    """重新准备上下文后scalar bias平移分数但不改N/RHS；相对分数权重实际改变聚合。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device)
    x, eigenvalues, basis, coords, c, action = complete_inputs(device)
    before = dynamics.rhs(c, action, dynamics.prepare_context(x, eigenvalues, basis, coords))
    with torch.no_grad():
        getattr(dynamics, branch)[2].bias.add_(.625)
    shifted = dynamics.rhs(c, action, dynamics.prepare_context(x, eigenvalues, basis, coords))
    score_field = "spectral_scores" if branch == "spectral_bias_net" else "spatial_scores"
    close(getattr(shifted.interaction, score_field), numpy64(getattr(before.interaction, score_field))+.625)
    close(shifted.interaction.total_scores, numpy64(before.interaction.total_scores)+.625)
    assert torch.equal(shifted.interaction.attention.neighbor_index, before.interaction.attention.neighbor_index)
    for field in ("weights", "aggregate"):
        close(getattr(shifted.interaction.attention, field), numpy64(getattr(before.interaction.attention, field)))
    close(shifted.interaction.nonlinear, numpy64(before.interaction.nonlinear))
    close(shifted.derivative, numpy64(before.derivative))
    with torch.no_grad():
        getattr(dynamics, branch)[2].weight.add_(torch.linspace(-.4, .3, 32, device=device)[None])
    changed = dynamics.rhs(c, action, dynamics.prepare_context(x, eigenvalues, basis, coords))
    relative = getattr(changed.interaction, score_field)-getattr(shifted.interaction, score_field)
    assert (relative-relative[..., :1]).abs().max() > 1e-4
    assert (changed.interaction.attention.aggregate-shifted.interaction.attention.aggregate).abs().max() > 1e-5
    expected, _ = complete_reference(dynamics, x, eigenvalues, basis, coords, c, action)
    close(changed.derivative, expected["derivative"])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch,modes", [(1, 1), (2, 2), (3, 6)])
def test_complete_rhs_small_k_different_batch_and_noncontiguous_inputs(device, batch, modes):
    """同一网络支持小K、不同B、非连续合法输入；不借batch广播或连续布局过验收。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device, r=16)
    inputs = complete_inputs(device, batch, modes)
    x, eigenvalues, basis, coords, c, action = (torch.stack((v, v), -1)[..., 0] for v in inputs)
    assert not x.is_contiguous() and not basis.is_contiguous() and not coords.is_contiguous()
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    result = dynamics.rhs(c, action, context)
    expected, _ = complete_reference(dynamics, x, eigenvalues, basis, coords, c, action)
    close(result.derivative, expected["derivative"])
    assert result.derivative.shape == (batch, modes, 64)
    assert result.interaction.attention.weights.shape == (batch, modes, modes)
    assert len(tuple(dynamics.parameters())) == 23
    assert sum(p.numel() for p in dynamics.parameters()) == 25634+4224*2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_rhs_frozen_public_fields_and_no_forward_side_effects(device):
    """前向保留值/版本/已有梯度、成员/模式/RNG/default；无梯度推理无需强制网络输出有梯度。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device).eval()
    x, eigenvalues, basis, coords, c, action = complete_inputs(device)
    for value in (x, c, action):
        value.requires_grad_()
        value.grad = torch.full_like(value, .3)
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    assert tuple(f.name for f in fields(context)) == ("eigenvalues", "basis", "coords", "pooled", "coefficients", "spectral_bias", "spatial_bias")
    with pytest.raises(FrozenInstanceError):
        context.pooled = x.mean(1)
    values = [x, eigenvalues, basis, coords, c, action, context.coords, context.pooled,
        context.spectral_bias, context.spatial_bias]+[getattr(context.coefficients, f.name) for f in fields(context.coefficients)]
    for value in values:
        if value.requires_grad and not value.is_leaf:
            value.retain_grad()
            value.grad = torch.full_like(value, .4)
    for parameter in dynamics.parameters():
        parameter.grad = torch.full_like(parameter, .2)
    tensors = values+list(dynamics.parameters())
    saved = [(id(v), v.detach().clone(), v._version, None if v.grad is None else
        (id(v.grad), v.grad.clone(), v.grad._version)) for v in tensors]
    members = {name: set(vars(layer)) for name, layer in dynamics.named_modules()}
    modes = {name: layer.training for name, layer in dynamics.named_modules()}
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    defaults = torch.get_default_dtype(), torch.get_default_device()
    expected, _ = complete_reference(dynamics, x, eigenvalues, basis, coords, c, action)
    result = dynamics.rhs(c, action, context)
    assert tuple(f.name for f in fields(result)) == ("free", "action_linear", "interaction", "derivative")
    assert tuple(f.name for f in fields(result.interaction)) == ("content", "spectral_scores", "spatial_scores", "total_scores", "attention", "action_residual", "nonlinear")
    for obj, field in ((result, "derivative"), (result.interaction, "nonlinear")):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, torch.zeros_like(c))
    close(result.derivative, expected["derivative"])
    assert result.derivative.requires_grad
    for scope in (torch.no_grad, torch.inference_mode):
        with scope():
            fresh = dynamics.prepare_context(x, eigenvalues, basis, coords)
            actual = dynamics.rhs(c, action, fresh)
            assert not actual.derivative.requires_grad and not torch.is_grad_enabled()
            close(actual.derivative, expected["derivative"])
    for value, (identity, prior, version, grad) in zip(tensors, saved):
        assert id(value) == identity and torch.equal(value, prior) and value._version == version
        if grad is None:
            assert value.grad is None
        else:
            assert id(value.grad) == grad[0] and torch.equal(value.grad, grad[1]) and value.grad._version == grad[2]
    assert members == {name: set(vars(layer)) for name, layer in dynamics.named_modules()}
    assert modes == {name: layer.training for name, layer in dynamics.named_modules()}
    assert list(dynamics.named_buffers()) == [] and len(tuple(dynamics.parameters())) == 23
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_free_rhs_channel_pair_contraction_and_independent_gradients(device):
    """任意非对称2×2显式块逐模态作用，核A非转置、无传播及C/A独立解析梯度。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device)
    c = torch.sin(torch.arange(768, device=device).float()*.071).reshape(2, 3, 128)[..., ::2].requires_grad_()
    a = torch.cos(torch.arange(1536, device=device).float()*.093).reshape(2, 3, 32, 2, 4)[..., ::2].requires_grad_()
    upstream = torch.sin(torch.arange(384, device=device).float()*.17).reshape_as(c)+.1
    cn, an, gy = numpy64(c).reshape(2, 3, 32, 2), numpy64(a), numpy64(upstream).reshape(2, 3, 32, 2)
    expected = np.einsum("bkrpq,bkrq->bkrp", an, cn).reshape(2, 3, 64)
    actual = dynamics.free_rhs(c, a)
    close(actual, expected)
    (actual*upstream).sum().backward()
    close(c.grad, np.einsum("bkrp,bkrpq->bkrq", gy, an).reshape(2, 3, 64))
    close(a.grad, gy[..., :, None]*cn[..., None, :])
    assert all(p.grad is None for p in dynamics.parameters())
    close(dynamics.to("meta").free_rhs(c, a), expected)


@pytest.mark.parametrize("field,kind", [(field, kind) for field in ("x", "eigenvalues", "basis", "coords")
    for kind in ("none", "double_or_float", "meta", "nan", "shape")]+[
    ("eigenvalues", "grad"), ("basis", "grad"), ("coords", "grad"),
    ("coords", "negative"), ("coords", "above_one"), ("coords", "sparse")])
def test_complete_prepare_context_rejects_invalid_original_inputs(field, kind):
    """唯一坐标转换入口拒绝错误原始协议；lambda/Psi的CPU double副本不能冒充模型字段。"""
    dynamics = hand_complete_dynamics()
    x, eigenvalues, basis, coords, _, _ = complete_inputs()
    inputs = dict(x=x, eigenvalues=eigenvalues, basis=basis, coords=coords)
    value = inputs[field]
    if kind == "none":
        bad = None
    elif kind == "double_or_float":
        bad = value.float() if field == "coords" else value.double()
    elif kind == "shape":
        bad = value[..., :1] if field == "coords" else value[:1] if field in ("eigenvalues", "basis") else value[..., :63]
    elif kind == "negative":
        bad = value.clone()
        bad[0, 0] = -torch.finfo(torch.float64).tiny
    elif kind == "above_one":
        bad = value+1.
    else:
        bad = dict(meta=lambda: value.to("meta"), nan=lambda: value*math.nan,
            grad=lambda: value.clone().requires_grad_(), sparse=lambda: value.to_sparse())[kind]()
    inputs[field] = bad
    with pytest.raises(TypeError if kind == "none" else ValueError):
        dynamics.prepare_context(**inputs)
    with pytest.raises(TypeError):
        dynamics.prepare_context(x, eigenvalues, basis)


@pytest.mark.parametrize("method", ["rhs", "nonlinear"])
@pytest.mark.parametrize("field,kind", [(field, kind) for field in
    ("eigenvalues", "basis", "coords", "pooled", "spectral_bias", "spatial_bias")
    for kind in ("none", "double", "meta", "nan", "shape")]+[
    ("eigenvalues", "grad"), ("basis", "grad"), ("coords", "grad"),
    ("spectral_bias", "column_broadcast"), ("spatial_bias", "column_broadcast"),
    ("coords", "negative"), ("coords", "above_one")])
def test_complete_rhs_rejects_invalid_context_fields_without_broadcast(method, field, kind):
    """完整context七字段均验证，偏置严格[B,K,K]；推理输出无梯度仍合法。"""
    dynamics = hand_complete_dynamics()
    x, eigenvalues, basis, coords, c, action = complete_inputs()
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    value = getattr(context, field)
    if kind == "none":
        bad = None
    elif kind == "shape":
        bad = value[..., :1] if field == "coords" else value[:1]
    elif kind == "column_broadcast":
        bad = value[..., :1]
    elif kind == "negative":
        bad = value.clone()
        bad[0, 0] = -torch.finfo(torch.float32).tiny
    elif kind == "above_one":
        bad = value+1.
    else:
        bad = dict(double=lambda: value.double(), meta=lambda: value.to("meta"), nan=lambda: value*math.nan,
            grad=lambda: value.clone().requires_grad_())[kind]()
    with pytest.raises(TypeError if kind == "none" else ValueError):
        getattr(dynamics, method)(c, action, replace(context, **{field: bad}))


@pytest.mark.parametrize("method", ["rhs", "nonlinear"])
@pytest.mark.parametrize("field,kind", [(field, kind) for field in
    ("raw_gamma", "gamma", "omega", "a_blocks", "b_matrix") for kind in ("none", "double", "nan", "shape")])
def test_complete_rhs_rejects_invalid_nested_conditioned_fields(method, field, kind):
    """条件系数完整五字段不能借未使用诊断字段绕过shape/精度/有限性合同。"""
    dynamics = hand_complete_dynamics()
    x, eigenvalues, basis, coords, c, action = complete_inputs()
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    value = getattr(context.coefficients, field)
    bad = dict(none=lambda: None, double=lambda: value.double(), nan=lambda: value*math.nan,
        shape=lambda: value[:1])[kind]()
    corrupted = replace(context, coefficients=replace(context.coefficients, **{field: bad}))
    with pytest.raises(TypeError if kind == "none" else ValueError):
        getattr(dynamics, method)(c, action, corrupted)


@pytest.mark.parametrize("method", ["rhs", "nonlinear"])
@pytest.mark.parametrize("bad", ["none", "dict", "condition_none", "condition_dict", "c_batch", "c_modes", "c_width", "a_batch", "a_width", "c_double", "a_double", "c_nan", "a_inf"])
def test_complete_rhs_rejects_missing_context_and_current_input_mismatch(method, bad):
    """唯一公共类型与当前C/a严格匹配；禁止默认上下文、batch或模态广播。"""
    dynamics = hand_complete_dynamics()
    x, eigenvalues, basis, coords, c, action = complete_inputs()
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    if bad in ("none", "dict"):
        context = None if bad == "none" else vars(context)
    elif bad.startswith("condition_"):
        context = replace(context, coefficients=None if bad.endswith("none") else vars(context.coefficients))
    elif bad.startswith("c_"):
        c = dict(c_batch=lambda: c[:1], c_modes=lambda: c[:, :1], c_width=lambda: c[..., :63],
            c_double=lambda: c.double(), c_nan=lambda: c*math.nan)[bad]()
    else:
        action = dict(a_batch=lambda: action[:1], a_width=lambda: action[:, :1],
            a_double=lambda: action.double(), a_inf=lambda: action*math.inf)[bad]()
    with pytest.raises(TypeError if bad in ("none", "dict", "condition_none", "condition_dict") else ValueError):
        getattr(dynamics, method)(c, action, context)


@pytest.mark.parametrize("stage", ["score_sum", "nonlinear_sum", "rhs_sum"])
def test_complete_rhs_rejects_finite_inputs_that_overflow_combined_outputs(stage):
    """单分支有限不代表完整求和有限；总分、N、最终RHS溢出均失败且不裁剪。"""
    dynamics = hand_complete_dynamics()
    x, eigenvalues, basis, coords, c, action = complete_inputs(batch=1, modes=1)
    c, action = torch.ones_like(c), torch.ones_like(action)
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    huge = torch.finfo(torch.float32).max*.6
    if stage == "score_sum":
        context = replace(context, spectral_bias=torch.full_like(context.spectral_bias, huge),
            spatial_bias=torch.full_like(context.spatial_bias, huge))
    elif stage == "nonlinear_sum":
        with torch.no_grad():
            dynamics.v_proj.weight.zero_()
            dynamics.v_proj.weight[:, 0] = huge
            dynamics.action_residual_net[2].weight.zero_()
            dynamics.action_residual_net[2].bias.fill_(huge)
    else:
        blocks = torch.zeros_like(context.coefficients.a_blocks)
        blocks[..., 0, 0] = blocks[..., 1, 1] = huge
        matrix = torch.zeros_like(context.coefficients.b_matrix)
        matrix[..., 0] = huge
        context = replace(context, coefficients=replace(context.coefficients, a_blocks=blocks, b_matrix=matrix))
    with pytest.raises(ValueError):
        dynamics.rhs(c, action, context)


@pytest.mark.parametrize("target", ["x", "eigenvalues", "basis", "coords", "c", "action", "context", "model"])
def test_complete_rhs_actual_cuda_device_mismatch(target):
    """实际CUDA核查仅coords可从CPU协议转换，其余模型输入/上下文不隐式搬设备。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics()
    x, eigenvalues, basis, coords, c, action = complete_inputs()
    if target in ("x", "eigenvalues", "basis", "coords"):
        inputs = dict(x=x, eigenvalues=eigenvalues, basis=basis, coords=coords)
        inputs[target] = inputs[target].cuda()
        with pytest.raises(ValueError):
            dynamics.prepare_context(**inputs)
        return
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    if target == "c":
        c = c.cuda()
    elif target == "action":
        action = action.cuda()
    elif target == "context":
        context = replace(context, spectral_bias=context.spectral_bias.cuda())
    else:
        dynamics.cuda()
    with pytest.raises(ValueError):
        dynamics.rhs(c, action, context)


@pytest.mark.parametrize("field,kind", [(field, kind) for field in ("c", "a")
    for kind in ("none", "double", "meta", "nan", "sparse", "batch", "modes", "width")])
def test_complete_free_rhs_rejects_invalid_contraction_inputs(field, kind):
    """纯Ac入口自行拒绝非模型精度/存储与轴错配，不能借其他完整入口代验。"""
    dynamics = hand_complete_dynamics()
    _, _, _, _, c, _ = complete_inputs()
    a = torch.eye(2).expand(2, 3, 32, 2, 2).clone()
    value = c if field == "c" else a
    bad = dict(none=lambda: None, double=lambda: value.double(), meta=lambda: value.to("meta"),
        nan=lambda: value*math.nan, sparse=lambda: value.to_sparse(), batch=lambda: value[:1],
        modes=lambda: value[:, :1], width=lambda: value[..., :1])[kind]()
    with pytest.raises(TypeError if kind == "none" else ValueError):
        dynamics.free_rhs(bad if field == "c" else c, bad if field == "a" else a)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_rhs_single_slot_has_connected_legal_zero_score_gradients(device):
    """r_eff=1时所有评分路径合法零；单次完整RHS损失仍使其grad存在，其他支路参与。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device, r=1)
    x, eigenvalues, basis, coords, c, action = complete_inputs(device)
    x.requires_grad_()
    c.requires_grad_()
    action.requires_grad_()
    result = dynamics.rhs(c, action, dynamics.prepare_context(x, eigenvalues, basis, coords))
    upstream = torch.cos(torch.arange(c.numel(), device=device).float()*.113).reshape_as(c)+.2
    (result.derivative*upstream).sum().backward()
    zero_prefixes = ("q_proj.", "k_proj.", "spectral_bias_net.", "spatial_bias_net.")
    for name, parameter in dynamics.named_parameters():
        assert parameter.grad is not None, name
        if name.startswith(zero_prefixes):
            close(parameter.grad, np.zeros(parameter.shape))
        else:
            assert parameter.grad.count_nonzero() > 0, name
    assert x.grad.count_nonzero() and c.grad.count_nonzero() and action.grad.count_nonzero()


def integration_config(**changes):
    """D10五个显式积分配置；时间单位为虚构的内部控制周期。"""
    assert hasattr(api(), "IntegratorConfig"), "D10 IntegratorConfig 尚未实现"
    values = dict(method="exponential_euler", max_step=.25, endpoint_rtol=1e-12,
                  phi1_threshold=1e-4, phi1_order=6)
    values.update(changes)
    return api().IntegratorConfig(**values)


def test_d10_zero_rates_have_identity_and_duration_integral():
    """最小RED：零A必须返回E=I、H=delta I，不直接计算0/0。"""
    dynamics = api().ModalDynamics(config())
    assert hasattr(dynamics, "integration_coefficients"), "D10 integration_coefficients 尚未实现"
    rates = torch.zeros(2, 3, 32, dtype=torch.float64)
    delta = torch.tensor([.125, .5], dtype=torch.float64)
    result = dynamics.integration_coefficients(gamma=rates, omega=rates, delta=delta,
                                             config=integration_config())
    identity = torch.eye(2, dtype=torch.float64).expand(2, 3, 32, 2, 2)
    close64(result.exponential, identity)
    close64(result.forcing_integral, identity*delta[:, None, None, None, None])


def test_d10_single_step_exposes_actual_left_rhs_and_propagator():
    """最小RED：更新结果携带同次左端RHS及实际使用的E/H，不提前分段。"""
    dynamics = hand_complete_dynamics()
    assert hasattr(dynamics, "exponential_euler_step"), "D10 exponential_euler_step 尚未实现"
    x, eigenvalues, basis, coords, c, action = complete_inputs()
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    result = dynamics.exponential_euler_step(c, action, context,
        delta=torch.tensor([.125, .5], dtype=torch.float64), config=integration_config())
    assert tuple(f.name for f in fields(result)) == ("coefficients", "rhs", "propagator")
    assert result.coefficients.shape == c.shape and result.coefficients.dtype == c.dtype
    assert result.rhs.derivative.shape == c.shape
    assert result.propagator.exponential.shape == (2, 3, 32, 2, 2)


def integration_reference(gamma, omega, delta):
    """独立CPU double增广4x4指数的E/H；仅参考临时矩阵contiguous，梯度保留。

    gamma/omega为[B,K,32]率，delta为CPU double [B]周期；右上块为
    单位阵，所以exp(delta*[[A,I],[0,0]])右上恰是积分H，不用生产phi1。
    """
    g, w = gamma.to(device="cpu", dtype=torch.float64), omega.to(device="cpu", dtype=torch.float64)
    a = torch.stack((-g, -w, w, -g), -1).reshape(*g.shape, 2, 2)
    identity = torch.eye(2, dtype=torch.float64).expand_as(a)
    augmented = torch.cat((torch.cat((a, identity), -1), torch.zeros(*a.shape[:-2], 2, 4, dtype=torch.float64)), -2)
    exponential = torch.matrix_exp((augmented*delta[:, None, None, None, None]).contiguous())
    return exponential[..., :2, :2], exponential[..., :2, 2:]


def integration_inputs(dtype=torch.float64, device="cpu"):
    """虚构零率/纯衰减/旋转/混合及phi1阈值内外；不同B/K/通道不可混轴。"""
    scale = torch.tensor([0., 1e-12, .9999e-4, 1e-4, 1.0001e-4, .03, .8, 2.], dtype=dtype, device=device)
    gamma = scale.repeat(24).reshape(2, 3, 32)
    omega = torch.roll(scale, 3).repeat(24).reshape(2, 3, 32)
    gamma[..., :8] = 0.
    omega[..., 8:16] = 0.
    delta = torch.tensor([.125, 1.], dtype=torch.float64)
    return dict(gamma=gamma, omega=omega, delta=delta, config=integration_config())


@pytest.mark.parametrize("dtype,device", [(torch.float64, "cpu"), (torch.float32, "cpu"), (torch.float32, "cuda")])
def test_d10_eh_matches_augmented_matrix_exp_values_and_rate_gradients(dtype, device):
    """独立增广指数值与率导数；CPU double和模型float32分别按已锁定容差。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    args = integration_inputs(dtype, device)
    # 生产输入保留非连续步幅，参考求解器才可整理临时4x4矩阵。
    for name in ("gamma", "omega", "delta"):
        args[name] = torch.stack((args[name], args[name]), -1)[..., 0]
        assert not args[name].is_contiguous()
    gamma, omega, delta = (args[n] for n in ("gamma", "omega", "delta"))
    gamma.requires_grad_()
    omega.requires_grad_()
    result = api().ModalDynamics(config()).integration_coefficients(**args)
    reference_gamma = gamma.detach().clone().requires_grad_()
    reference_omega = omega.detach().clone().requires_grad_()
    expected_e, expected_h = integration_reference(reference_gamma, reference_omega, delta)
    check = close64 if dtype == torch.float64 else close
    expected = [expected_e.detach(), expected_h.detach()]
    errors = {"exponential": check(result.exponential, numpy64(expected[0])),
              "forcing_integral": check(result.forcing_integral, numpy64(expected[1]))}
    weight_e = torch.sin(torch.arange(result.exponential.numel(), dtype=dtype, device=device)*.13).reshape_as(result.exponential)
    weight_h = torch.cos(torch.arange(result.exponential.numel(), dtype=dtype, device=device)*.17).reshape_as(result.exponential)
    (result.exponential*weight_e+result.forcing_integral*weight_h).sum().backward()
    (expected_e*weight_e.cpu().double()+expected_h*weight_h.cpu().double()).sum().backward()
    errors.update(gamma=check(gamma.grad, numpy64(reference_gamma.grad)), omega=check(omega.grad, numpy64(reference_omega.grad)))
    for value in (result.exponential, result.forcing_integral):
        assert value.shape == (2, 3, 32, 2, 2) and value.dtype == dtype and value.device == gamma.device
    assert all(torch.isfinite(v.grad).all() for v in (gamma, omega))
    print(json.dumps(dict(probe="D10-augmented-exponential-value-gradient", dtype=str(dtype), device=device, errors=errors)))


@pytest.mark.parametrize("dtype,device", [(torch.float64, "cpu"), (torch.float32, "cpu"), (torch.float32, "cuda")])
def test_d10_zero_rate_derivatives_match_integral_definition(dtype, device):
    """零点独立加权解析导数：E率导数为delta*(-I,J)，H为delta²/2*(-I,J)。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    gamma = torch.zeros(2, 2, 32, dtype=dtype, device=device, requires_grad=True)
    omega = torch.zeros_like(gamma, requires_grad=True)
    delta = torch.tensor([.125, .7], dtype=torch.float64)
    result = api().ModalDynamics(config()).integration_coefficients(gamma=gamma, omega=omega, delta=delta, config=integration_config())
    weight_e = torch.linspace(-.8, .9, result.exponential.numel(), dtype=dtype, device=device).reshape_as(result.exponential)
    weight_h = torch.sin(torch.arange(result.exponential.numel(), dtype=dtype, device=device)*.137).reshape_as(result.exponential)
    (result.exponential*weight_e+result.forcing_integral*weight_h).sum().backward()
    t = delta[:, None, None]
    we, wh = numpy64(weight_e), numpy64(weight_h)
    dg = -numpy64(t)*(we[..., 0, 0]+we[..., 1, 1])-numpy64(t)**2/2*(wh[..., 0, 0]+wh[..., 1, 1])
    dw = numpy64(t)*(-we[..., 0, 1]+we[..., 1, 0])+numpy64(t)**2/2*(-wh[..., 0, 1]+wh[..., 1, 0])
    check = close64 if dtype == torch.float64 else close
    identity = np.broadcast_to(np.eye(2), result.exponential.shape).copy()
    errors = dict(exponential=check(result.exponential, identity), forcing_integral=check(result.forcing_integral,
        identity*numpy64(delta)[:, None, None, None, None]), gamma=check(gamma.grad, dg), omega=check(omega.grad, dw))
    assert torch.isfinite(gamma.grad).all() and torch.isfinite(omega.grad).all()
    print(json.dumps(dict(probe="D10-zero-rate-analytic-gradient", dtype=str(dtype), device=device, errors=errors)))


def controlled_step_context(dynamics, device="cpu", *, zero_rates=False, forcing=0.):
    """合法D09上下文替换为一致解析率/A及常值M_a；未更改生产初始化。"""
    x, eigenvalues, basis, coords, c, action = complete_inputs(device)
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    gamma = torch.zeros_like(context.coefficients.gamma) if zero_rates else torch.full_like(context.coefficients.gamma, .3)
    omega = torch.zeros_like(gamma) if zero_rates else torch.full_like(gamma, -.7)
    a_blocks = torch.stack((-gamma, -omega, omega, -gamma), -1).reshape(2, 3, 32, 2, 2)
    conditioned = replace(context.coefficients, gamma=gamma, omega=omega, a_blocks=a_blocks,
                          b_matrix=torch.zeros_like(context.coefficients.b_matrix))
    with torch.no_grad():
        dynamics.v_proj.weight.zero_()
        for parameter in dynamics.action_residual_net.parameters():
            parameter.zero_()
        dynamics.action_residual_net[2].bias.copy_(torch.linspace(-forcing, forcing, 64, device=device))
    return replace(context, coefficients=conditioned), c, action


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("zero_rates,forcing", [(False, 0.), (False, .4), (True, .4)])
def test_d10_exponential_euler_forcing_limits_and_batched_boundaries(device, zero_rates, forcing, monkeypatch):
    """纯自由Ac非零可抓双计；常forcing与A0极限，左端RHS一次且诊断同对象。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device)
    context, c, action = controlled_step_context(dynamics, device, zero_rates=zero_rates, forcing=forcing)
    delta = torch.tensor([.125, .6], dtype=torch.float64)
    calls, original = [], dynamics.rhs
    propagators, original_propagator = [], dynamics.integration_coefficients
    def capture(left_c, left_action, left_context):
        """记录真实左端求值；保留完整D09 RHS而不模拟其返回值。"""
        assert left_c is c and left_action is action and left_context is context
        result = original(left_c, left_action, left_context)
        calls.append(result)
        return result
    monkeypatch.setattr(dynamics, "rhs", capture)
    def capture_propagator(**kwargs):
        """记录本步真实E/H结构，返回对象仍走实际更新而不重算诊断。"""
        value = original_propagator(**kwargs)
        propagators.append(value)
        return value
    monkeypatch.setattr(dynamics, "integration_coefficients", capture_propagator)
    result = dynamics.exponential_euler_step(c, action, context, delta=delta, config=integration_config())
    assert len(calls) == 1 and result.rhs is calls[0]
    assert len(propagators) == 1 and result.propagator is propagators[0]
    e, h = integration_reference(context.coefficients.gamma, context.coefficients.omega, delta)
    f = result.rhs.action_linear+result.rhs.interaction.nonlinear
    expected = (e@c.cpu().double().reshape(2, 3, 32, 2, 1)+h@f.cpu().double().reshape(2, 3, 32, 2, 1)).reshape_as(c)
    errors = dict(step=close(result.coefficients, numpy64(expected)), exponential=close(result.propagator.exponential, numpy64(e)),
                  forcing_integral=close(result.propagator.forcing_integral, numpy64(h)))
    if not zero_rates:
        assert result.rhs.free.abs().max() > .1
        wrong = expected+(h@result.rhs.free.cpu().double().reshape(2, 3, 32, 2, 1)).reshape_as(c)
        assert (expected-wrong).abs().max() > 1e-3
    if forcing == 0.:
        assert f.count_nonzero() == 0
        close(result.coefficients, numpy64(dynamics.free_propagate(c, gamma=context.coefficients.gamma,
            omega=context.coefficients.omega, duration=delta)))
    if zero_rates:
        close(result.coefficients, numpy64(c)+numpy64(delta)[:, None, None]*numpy64(f))
    print(json.dumps(dict(probe="D10-forcing-limits-left-RHS", device=device, zero_rates=zero_rates, forcing=forcing, errors=errors)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_d10_state_dependent_forcing_recomputed_and_first_order_halving(device, monkeypatch):
    """测试内固定步序列仅核单步：N(C)=C给出y'=y，减半步长误差应约减半。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device)
    context, c, action = controlled_step_context(dynamics, device, zero_rates=True)
    original, observed = dynamics.rhs, []
    def linear_residual(left_c, left_action, left_context):
        """受控可解析残差只替换N；完整RHS仍用于合法结构与当前C/a接线。"""
        base = original(left_c, left_action, left_context)
        observed.append(left_c.detach().clone())
        return replace(base, interaction=replace(base.interaction, nonlinear=left_c), derivative=base.free+base.action_linear+left_c)
    monkeypatch.setattr(dynamics, "rhs", linear_residual)
    errors = []
    with torch.no_grad():
        for count in (4, 8, 16):
            state = c.clone()
            before = len(observed)
            for _ in range(count):
                state = dynamics.exponential_euler_step(state, action, context,
                    delta=torch.full((2,), 1/count, dtype=torch.float64), config=integration_config()).coefficients
            assert len(observed)-before == count
            close(state, numpy64(c)*(1+1/count)**count)
            assert not torch.equal(observed[before], observed[before+1])
            errors.append(float(np.max(np.abs(numpy64(state)-numpy64(c)*math.e))))
    ratios = [errors[i]/errors[i+1] for i in range(2)]
    assert errors[0] > errors[1] > errors[2] and all(1.7 < ratio < 2.1 for ratio in ratios)
    print(json.dumps(dict(probe="D10-state-residual-first-order-halving", device=device, errors=errors, ratios=ratios)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("r", [1, 2])
def test_d10_real_projection_d09_chain_matches_independent_eh_backward(device, r):
    """实际C01—D09链与独立E/H比较单步梯度；复用已验收D09，不重复其算法。

    两张新图各一次backward，lambda/Psi保持同次常量。scalar bias和r=1评分
    路径允许连接的零梯度；非零条件率经E/H和完整forcing均需参与。
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(1703)
    dynamics = hand_complete_dynamics(device, r=r)
    graph = GraphSpectrum(GraphConfig(16, 1703, .25, 1., 128, 1e-10, 1e-8)).to(device)
    model = POJEPA(ProjectionConfig(5, 64), graph, dynamics, object()).to(device)
    raw_z = torch.linspace(-.8, .9, 60, device=device).reshape(2, 6, 5).requires_grad_()
    action = torch.tensor([[-.3, .6], [.8, -.4]], device=device, requires_grad=True)
    x = model.project_in(raw_z)
    spectrum = graph.lowest_eigenpairs(graph.normalized_laplacian(graph.affinity(x, regular_grid(2, 3))).laplacian,
        k=3, model_device=torch.device(device))
    x.retain_grad()
    c = modal_analysis(x, spectrum.basis)
    c.retain_grad()
    coords, delta = regular_grid(2, 3).coords, torch.tensor([.125, .6], dtype=torch.float64)
    context = dynamics.prepare_context(x, spectrum.eigenvalues, spectrum.basis, coords)
    for value in (context.coefficients.gamma, context.coefficients.omega, context.coefficients.b_matrix,
                  context.spectral_bias, context.spatial_bias):
        value.retain_grad()
    actual = dynamics.exponential_euler_step(c, action, context, delta=delta, config=integration_config())
    upstream = 16*(torch.cos(torch.arange(c.numel(), device=device).float()*.113).reshape_as(c)+.2)
    (actual.coefficients*upstream).sum().backward()
    other = hand_complete_dynamics(device, r=r)
    other.load_state_dict(dynamics.state_dict())
    reference_x = x.detach().clone().requires_grad_()
    reference_c = c.detach().clone().requires_grad_()
    reference_action = action.detach().clone().requires_grad_()
    reference_context = other.prepare_context(reference_x, spectrum.eigenvalues, spectrum.basis, coords)
    rhs = other.rhs(reference_c, reference_action, reference_context)
    e, h = integration_reference(reference_context.coefficients.gamma, reference_context.coefficients.omega, delta)
    e, h = e.to(device=device, dtype=torch.float32), h.to(device=device, dtype=torch.float32)
    forcing = rhs.action_linear+rhs.interaction.nonlinear
    expected = (e@reference_c.reshape(2, 3, 32, 2, 1)+h@forcing.reshape(2, 3, 32, 2, 1)).reshape_as(c)
    (expected*upstream).sum().backward()
    expected_x = reference_x.grad+spectrum.basis@reference_c.grad
    errors = dict(forward=close(actual.coefficients, numpy64(expected)), c=close(c.grad, numpy64(reference_c.grad)),
        action=close(action.grad, numpy64(reference_action.grad)), x=close(x.grad, numpy64(expected_x)),
        raw_z=close(raw_z.grad, numpy64(expected_x)@numpy64(model.p_in.weight)),
        p_in_weight=close(model.p_in.weight.grad, numpy64(expected_x).reshape(-1, 64).T@numpy64(raw_z).reshape(-1, 5)),
        p_in_bias=close(model.p_in.bias.grad, numpy64(expected_x).sum(axis=(0, 1))))
    zero_names = {"spectral_bias_net.2.bias", "spatial_bias_net.2.bias"}
    for name, parameter in dynamics.named_parameters():
        reference_gradient = dict(other.named_parameters())[name].grad
        assert parameter.grad is not None and reference_gradient is not None, name
        errors[name] = close(parameter.grad, numpy64(reference_gradient))
        if name in zero_names or (r == 1 and name.startswith(("q_proj.", "k_proj.", "spectral_bias_net.", "spatial_bias_net."))):
            close(parameter.grad, np.zeros(parameter.shape))
        else:
            assert parameter.grad.count_nonzero() > 0, name
    for value in (context.coefficients.gamma, context.coefficients.omega, context.coefficients.b_matrix):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.count_nonzero()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in (context.spectral_bias, context.spatial_bias))
    assert model.p_out.weight.grad is None and model.p_out.bias.grad is None
    assert not spectrum.eigenvalues.requires_grad and not spectrum.basis.requires_grad
    print(json.dumps(dict(probe="D10-real-chain-independent-EH-backward", device=device, top_r=r, errors=errors)))


@pytest.mark.parametrize("dtype,device", [(torch.float64, "cpu"), (torch.float32, "cpu"), (torch.float32, "cuda")])
def test_d10_safe_inactive_branches_for_zero_and_large_z(dtype, device):
    """零点未选ratio不做0/0，大z未选六阶多项式不溢出；合法指数下溢可为零。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    large = 1e150 if dtype == torch.float64 else 1e30
    gamma = torch.cat((torch.tensor([0., .2], dtype=dtype, device=device).repeat(16),
                       torch.full((32,), large, dtype=dtype, device=device))).reshape(2, 1, 32).requires_grad_()
    omega = torch.cat((torch.tensor([0., -.7], dtype=dtype, device=device).repeat(16),
                       torch.full((32,), large, dtype=dtype, device=device))).reshape(2, 1, 32).requires_grad_()
    result = api().ModalDynamics(config()).integration_coefficients(gamma=gamma, omega=omega,
        delta=torch.tensor([1., 1e30], dtype=torch.float64), config=integration_config())
    (result.exponential.sum()+result.forcing_integral.sum()).backward()
    assert all(torch.isfinite(value).all() for value in (result.exponential, result.forcing_integral, gamma.grad, omega.grad))
    check = close64 if dtype == torch.float64 else close
    check(result.exponential[0, 0, 0], np.eye(2))
    check(result.forcing_integral[0, 0, 0], np.eye(2))
    assert result.exponential[1].count_nonzero() == 0
    assert result.forcing_integral[1].abs().max() > 0
    print(json.dumps(dict(probe="D10-safe-inactive-branches", dtype=str(dtype), device=device,
        gamma_gradient_max=float(gamma.grad.abs().max()), omega_gradient_max=float(omega.grad.abs().max()),
        tiny_h_max=float(result.forcing_integral[1].abs().max()))))


def test_d10_configuration_public_frozen_fields_and_required_values():
    """唯一五字段且无隐式默认；冻结配置/返回结构不冻结Tensor autograd。"""
    configuration = integration_config()
    assert tuple(f.name for f in fields(configuration)) == ("method", "max_step", "endpoint_rtol", "phi1_threshold", "phi1_order")
    assert all(f.default is MISSING and f.default_factory is MISSING for f in fields(configuration))
    with pytest.raises(FrozenInstanceError):
        configuration.max_step = .5
    args = integration_inputs()
    result = api().ModalDynamics(config()).integration_coefficients(**args)
    assert tuple(f.name for f in fields(result)) == ("exponential", "forcing_integral")
    with pytest.raises(FrozenInstanceError):
        result.exponential = result.forcing_integral
    dynamics = hand_complete_dynamics()
    x, eigenvalues, basis, coords, c, action = complete_inputs()
    step = dynamics.exponential_euler_step(c, action, dynamics.prepare_context(x, eigenvalues, basis, coords),
        delta=args["delta"], config=configuration)
    with pytest.raises(FrozenInstanceError):
        step.coefficients = c
    assert step.coefficients.requires_grad


@pytest.mark.parametrize("name,bad", [("method", None), ("method", True), ("method", "euler"),
    pytest.param("method", np.array(["exponential_euler"]), id="method-numpy-array"),
    *[(name, value) for name in ("max_step", "endpoint_rtol") for value in
      (True, False, 0., -1., math.nan, math.inf, -math.inf, "0.25", None, 1j, 10**1000)],
    *[("phi1_threshold", value) for value in (True, False, 0., 1e-5, math.nan, math.inf, None, "1e-4")],
    *[("phi1_order", value) for value in (True, False, 0, 5, 7, 6., None, "6")]])
def test_d10_rejects_invalid_integrator_configuration(name, bad):
    """未登记算法/phi1方案与非正非有限实数均拒绝，不能悄改数值方法。"""
    with pytest.raises(ValueError):
        integration_config(**{name: bad})


@pytest.mark.parametrize("bad", [None, {}, object(), 2])
def test_d10_rejects_non_config_objects_at_both_entrypoints(bad):
    """两公共入口拒绝假配置；不允许字段相似对象隐式替代冻结配置。"""
    dynamics = hand_complete_dynamics()
    args = integration_inputs()
    with pytest.raises(TypeError):
        dynamics.integration_coefficients(**dict(args, config=bad))
    x, eigenvalues, basis, coords, c, action = complete_inputs()
    with pytest.raises(TypeError):
        dynamics.exponential_euler_step(c, action, dynamics.prepare_context(x, eigenvalues, basis, coords),
            delta=args["delta"], config=bad)


@pytest.mark.parametrize("target", ["gamma", "omega", "delta"])
@pytest.mark.parametrize("kind", ["none", "list", "half", "bfloat16", "int", "bool", "complex", "sparse", "nested", "meta", "nan", "inf"])
def test_d10_rejects_invalid_eh_tensor_contract(target, kind):
    """率及时间只接受真实稠密有限约定精度；非Tensor与值合同区分失败。"""
    args = integration_inputs()
    conversions = dict(none=lambda x: None, list=lambda x: [], half=lambda x: x.half(),
        bfloat16=lambda x: x.bfloat16(), int=lambda x: x.long(), bool=lambda x: x.bool(),
        complex=lambda x: x.to(torch.complex128), sparse=lambda x: x.to_sparse(),
        nested=lambda x: torch.nested.nested_tensor([x[0:1], x[1:2]]), meta=lambda x: x.to("meta"),
        nan=lambda x: x*math.nan, inf=lambda x: x*math.inf)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")
        args[target] = conversions[kind](args[target])
    with pytest.raises(TypeError if kind in ("none", "list") else ValueError):
        api().ModalDynamics(config()).integration_coefficients(**args)


@pytest.mark.parametrize("target,bad", [("gamma", torch.zeros(2, 3, 31, dtype=torch.float64)),
    ("omega", torch.zeros(2, 2, 32, dtype=torch.float64)), ("gamma", torch.zeros(0, 3, 32, dtype=torch.float64)),
    ("gamma", torch.zeros(2, 0, 32, dtype=torch.float64)), ("delta", torch.ones(2, 1, dtype=torch.float64)),
    ("delta", torch.ones(1, dtype=torch.float64)), ("delta", torch.ones(2, dtype=torch.float32)),
    ("delta", torch.tensor([0., .1], dtype=torch.float64)), ("delta", torch.tensor([-.1, .1], dtype=torch.float64)),
    ("delta", torch.ones(2, dtype=torch.float64, requires_grad=True)), ("gamma", -torch.ones(2, 3, 32, dtype=torch.float64)),
    ("gamma", torch.ones(2, 3, 32, dtype=torch.float32))])
def test_d10_rejects_eh_shape_sign_time_and_mixed_precision(target, bad):
    """禁止batch广播/负阻尼/非正步长/可训练时间以及混精度，不替调用者修复。"""
    args = integration_inputs()
    args[target] = bad
    with pytest.raises(ValueError):
        api().ModalDynamics(config()).integration_coefficients(**args)


def test_d10_eh_rejects_work_product_and_cast_output_overflow():
    """有限输入仍可能工作乘积或模型H溢出，必须报告且不裁剪。"""
    dynamics = api().ModalDynamics(config())
    with pytest.raises(ValueError):
        dynamics.integration_coefficients(gamma=torch.full((1, 1, 32), 1e308, dtype=torch.float64),
            omega=torch.zeros(1, 1, 32, dtype=torch.float64), delta=torch.tensor([2.], dtype=torch.float64), config=integration_config())
    with pytest.raises(ValueError):
        dynamics.integration_coefficients(gamma=torch.zeros(1, 1, 32), omega=torch.zeros(1, 1, 32),
            delta=torch.tensor([1e40], dtype=torch.float64), config=integration_config())


def test_d10_single_step_rejects_finite_inputs_with_update_overflow():
    """D09 RHS有限不保证EC+Hf有限；更新溢出明确失败，不返回或裁剪inf。"""
    dynamics = hand_complete_dynamics()
    context, c, action = controlled_step_context(dynamics, zero_rates=True)
    with torch.no_grad():
        dynamics.q_proj.weight.zero_()
        dynamics.k_proj.weight.zero_()
        dynamics.action_residual_net[2].bias.fill_(torch.finfo(torch.float32).max)
    large_c = torch.full_like(c, torch.finfo(torch.float32).max)
    rhs = dynamics.rhs(large_c, action, context)
    assert all(torch.isfinite(value).all() for value in
        (rhs.free, rhs.action_linear, rhs.interaction.nonlinear, rhs.derivative))
    with pytest.raises(ValueError, match="updated coefficients"):
        dynamics.exponential_euler_step(large_c, action, context,
            delta=torch.ones(2, dtype=torch.float64), config=integration_config())


@pytest.mark.parametrize("target,kind", [("coefficients", "double"), ("action", "double"),
    ("coefficients", "nan"), ("action", "nan"), ("coefficients", "shape"), ("action", "shape"),
    ("coefficients", "none"), ("action", "none"), ("context", "none"), ("context", "pooled"),
    ("context", "rates"), ("delta", "zero"), ("delta", "dtype")])
def test_d10_single_step_rejects_invalid_model_context_and_time(target, kind):
    """单步保留D09完整float32模型合同，也拒绝损坏上下文/非法步长。"""
    dynamics = hand_complete_dynamics()
    x, eigenvalues, basis, coords, c, action = complete_inputs()
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    args = dict(coefficients=c, action=action, context=context, delta=torch.ones(2, dtype=torch.float64), config=integration_config())
    if kind == "none":
        args[target] = None
    elif kind == "double":
        args[target] = args[target].double()
    elif kind == "nan":
        args[target] = args[target]*math.nan
    elif kind == "shape":
        args[target] = args[target][:1]
    elif kind == "pooled":
        args[target] = replace(context, pooled=context.pooled.double())
    elif kind == "rates":
        args[target] = replace(context, coefficients=replace(context.coefficients, gamma=-context.coefficients.gamma))
    elif kind == "zero":
        args[target] = torch.zeros(2, dtype=torch.float64)
    else:
        args[target] = args[target].float()
    with pytest.raises(TypeError if kind == "none" else ValueError):
        dynamics.exponential_euler_step(**args)


def test_d10_actual_cuda_eh_device_contract():
    """实际CUDA模型率/CPU时间合法；率跨设备、CUDA时间和CUDA double均拒绝。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = api().ModalDynamics(config())
    args = integration_inputs(torch.float32, "cuda")
    for name in ("gamma", "omega"):
        with pytest.raises(ValueError):
            dynamics.integration_coefficients(**dict(args, **{name: args[name].cpu()}))
    with pytest.raises(ValueError):
        dynamics.integration_coefficients(**dict(args, delta=args["delta"].cuda()))
    with pytest.raises(ValueError):
        dynamics.integration_coefficients(**dict(args, gamma=args["gamma"].double(), omega=args["omega"].double()))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_d10_step_noncontiguous_inputs_preserve_inventory_initialization_and_state(device):
    """非连续C/a/率/时间合法；单步不改成员/参数/输入/既有grad/模式/RNG/default。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(1703)
    dynamics = api().ModalDynamics(config()).to(device).eval()
    parameters = {n: (id(p), p.detach().clone(), p._version) for n, p in dynamics.named_parameters()}
    assert len(parameters) == 23 and sum(p.numel() for p in dynamics.parameters()) == 25634+4224*2
    x, eigenvalues, basis, coords, c, action = complete_inputs(device)
    c, action = (torch.stack((v, v), -1)[..., 0].requires_grad_() for v in (c, action))
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    rates = context.coefficients
    context = replace(context, coefficients=replace(rates,
        gamma=torch.stack((rates.gamma, rates.gamma), -1)[..., 0],
        omega=torch.stack((rates.omega, rates.omega), -1)[..., 0]))
    delta = torch.tensor([.125, 9., .5, 9.], dtype=torch.float64)[::2]
    values = [c, action, delta, context.pooled, context.coefficients.gamma, context.coefficients.omega,
        context.coefficients.a_blocks, context.coefficients.b_matrix, context.spectral_bias, context.spatial_bias]
    for value in (c, action):
        value.grad = torch.full_like(value, .3)
    for p in dynamics.parameters():
        p.grad = torch.full_like(p, .2)
    saved = [(v.detach().clone(), v._version) for v in values]
    grads = [(v, id(v.grad), v.grad.clone(), v.grad._version) for v in (c, action, *dynamics.parameters())]
    members = {n: set(vars(layer)) for n, layer in dynamics.named_modules()}
    modes = {n: layer.training for n, layer in dynamics.named_modules()}
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    defaults = torch.get_default_dtype(), torch.get_default_device()
    result = dynamics.exponential_euler_step(c, action, context, delta=delta, config=integration_config())
    assert result.coefficients.requires_grad and result.coefficients is not c
    for grad_context in (torch.no_grad, torch.inference_mode):
        with grad_context():
            inferred = dynamics.exponential_euler_step(c, action, context, delta=delta, config=integration_config())
            close(inferred.coefficients, numpy64(result.coefficients))
            assert not inferred.coefficients.requires_grad
    for value, (old, version) in zip(values, saved):
        assert torch.equal(value, old) and value._version == version
    for value, grad_id, old_grad, version in grads:
        assert id(value.grad) == grad_id and torch.equal(value.grad, old_grad) and value.grad._version == version
    for name, p in dynamics.named_parameters():
        identifier, old, version = parameters[name]
        assert id(p) == identifier and torch.equal(p, old) and p._version == version
    assert members == {n: set(vars(layer)) for n, layer in dynamics.named_modules()}
    assert modes == {n: layer.training for n, layer in dynamics.named_modules()}
    assert list(dynamics.named_buffers()) == [] and torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())


def d11_controls(rows, *, device="cpu", values=None):
    """直接建立虚构已知命令路径，保留原始端点；不借B04构造器对齐测试输入。"""
    from po_jepa.actions import ActionSegments, ControlSpec
    batch, width = len(rows), max(1, max(len(row) for row in rows))
    starts = torch.full((batch, width), float("nan"), dtype=torch.float64)
    ends = torch.full_like(starts, float("nan"))
    commands = torch.full((batch, width, 2), float("nan"), device=device)
    for b, row in enumerate(rows):
        for j, (start, end, command) in enumerate(row):
            starts[b, j], ends[b, j] = start, end
            commands[b, j] = torch.tensor(command, dtype=torch.float32, device=device)
    counts = torch.tensor([len(row) for row in rows], dtype=torch.int64)
    spec = ControlSpec(2, 1., ("fictional_u", "fictional_v"), ("unit", "unit"),
        (-100., -100.), (100., 100.), (1., 1.), (0., 0.), "D11 fictional commands")
    return ActionSegments(starts, ends, commands if values is None else values, counts,
        torch.arange(width)[None, :] < counts[:, None], torch.full((batch,), 1e12, dtype=torch.float64),
        torch.tensor([row[-1][1] if row else 0. for row in rows], dtype=torch.float64),
        spec, tuple(f"d11-fictional-{b}" for b in range(batch)))


def test_d11_zero_duration_identity_has_no_step_or_action(monkeypatch):
    """D11最小RED：零查询逐样本保持initial与恒等梯度，不取动作或执行D10。"""
    dynamics = hand_complete_dynamics()
    assert hasattr(dynamics, "integrate_segments"), "D11 integrate_segments 尚未实现"
    x, eigenvalues, basis, coords, c, _ = complete_inputs()
    c.requires_grad_()
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    def forbidden(*args, **kwargs):
        """零时长没有物理步，不能通过调用后丢弃结果伪装。"""
        pytest.fail("zero duration executed a step/action query")
    monkeypatch.setattr(dynamics, "exponential_euler_step", forbidden)
    monkeypatch.setattr(api(), "action_at", forbidden)
    result = dynamics.integrate_segments(c, d11_controls([[], []]), context,
        duration=torch.zeros(2, dtype=torch.float64), config=integration_config(), collect_trace=True)
    assert torch.equal(result.final, c) and result.steps == ()
    assert result.n_steps.dtype == torch.int64 and result.n_steps.device.type == "cpu"
    assert torch.equal(result.n_steps, torch.zeros(2, dtype=torch.int64))
    result.final.sum().backward()
    assert torch.equal(c.grad, torch.ones_like(c))


def d11_context_rows(context, indices):
    """测试侧按已知原样本取上下文；五系数全保留图，coords没有batch轴。"""
    index = torch.tensor(indices, dtype=torch.int64, device=context.pooled.device)
    coefficients = replace(context.coefficients, **{f.name: getattr(context.coefficients, f.name).index_select(0, index)
        for f in fields(context.coefficients)})
    return replace(context, coefficients=coefficients, **{name: getattr(context, name).index_select(0, index)
        for name in ("eigenvalues", "basis", "pooled", "spectral_bias", "spatial_bias")})


def d11_asynchronous_paths(device="cpu"):
    """独立手算案例：不同切换/段数、部分末段、零查询及完全未使用的末段。"""
    rows = [[(0., .3, (2., -.5)), (.3, 1., (-1., .7))],
        [(0., .125, (.4, 1.5)), (.125, .5, (1.2, -.3)), (.5, .9, (3., -2.))],
        [], [(0., .2, (-.7, .6)), (.2, .6, (.8, 1.1)), (.6, 1., (-1.3, .2))]]
    duration = torch.tensor([.8, .45, 0., .64], dtype=torch.float64)
    # 直接根据min(.25,查询剩余,当前段剩余)手算，禁止用待测schedule构造预期。
    plan = [([0, 1, 3], [0., 0., 0.], [.25, .125, .2], [0, 0, 0]),
        ([0, 1, 3], [.25, .125, .2], [.05, .25, .25], [0, 1, 1]),
        ([0, 1, 3], [.3, .375, .45], [.25, .075, .15], [1, 1, 1]),
        ([0, 3], [.55, .6], [.25, .04], [1, 2])]
    return rows, d11_controls(rows, device=device), duration, plan


def d11_constant_context(dynamics, *, batch=4, device="cpu", zero_rates=False):
    """测试内置固定A/B、N=0系统；通道/模态/B各异，解析forcing为Ba。"""
    x, eigenvalues, basis, coords, c, _ = complete_inputs(device, batch=batch)
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    shape = context.coefficients.gamma.shape
    gamma = torch.zeros(shape, device=device) if zero_rates else torch.linspace(.1, .6, math.prod(shape), device=device).reshape(shape)
    omega = torch.zeros(shape, device=device) if zero_rates else torch.linspace(-.8, .9, math.prod(shape), device=device).reshape(shape)
    a = torch.stack((-gamma, -omega, omega, -gamma), -1).reshape(batch, 3, 32, 2, 2)
    b = torch.sin(torch.arange(batch*3*64*2, device=device).float()*.037).reshape(batch, 3, 64, 2)*.3
    conditioned = replace(context.coefficients, gamma=gamma, omega=omega, a_blocks=a, b_matrix=b)
    with torch.no_grad():
        dynamics.v_proj.weight.zero_()
        for parameter in dynamics.action_residual_net.parameters():
            parameter.zero_()
    return c, replace(context, coefficients=conditioned)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("zero_rates", [False, True])
def test_d11_piecewise_constant_analytic_solution_and_hand_trace(device, zero_rates, monkeypatch):
    """按原段解析组合独立终点；逐轮手算trace/active上下文/右动作与同次StepResult。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device)
    c, context = d11_constant_context(dynamics, device=device, zero_rates=zero_rates)
    c.requires_grad_()
    rows, controls, duration, plan = d11_asynchronous_paths(device)
    recorded, original_step = [], dynamics.exponential_euler_step
    def capture(left_c, action, active_context, *, delta, config):
        """记录实际D10输入及对象，核五系数/B字段/共享coords，不代替真实步计算。"""
        indices, _, _, segments = plan[len(recorded)]
        expected_context = d11_context_rows(context, indices)
        assert active_context.coords is context.coords
        for name in ("eigenvalues", "basis", "pooled", "spectral_bias", "spatial_bias"):
            assert torch.equal(getattr(active_context, name), getattr(expected_context, name)), name
        for f in fields(active_context.coefficients):
            assert torch.equal(getattr(active_context.coefficients, f.name), getattr(expected_context.coefficients, f.name)), f.name
        expected_action = torch.stack([controls.u_values[b, j] for b, j in zip(indices, segments)])
        assert torch.equal(action, expected_action)
        result = original_step(left_c, action, active_context, delta=delta, config=config)
        recorded.append((left_c, result))
        return result
    monkeypatch.setattr(dynamics, "exponential_euler_step", capture)
    result = dynamics.integrate_segments(c, controls, context, duration=duration,
        config=integration_config(max_step=np.float64(.25)), collect_trace=True)
    assert tuple(f.name for f in fields(result)) == ("final", "n_steps", "steps")
    assert tuple(f.name for f in fields(result.steps[0])) == ("batch_index", "start_time", "delta", "segment_index", "before", "result")
    with pytest.raises(FrozenInstanceError):
        result.final = c
    with pytest.raises(FrozenInstanceError):
        result.steps[0].before = c
    assert all(f.default is MISSING and f.default_factory is MISSING for cls in (api().IntegrationResult, api().IntegrationStep) for f in fields(cls))
    assert result.final.dtype == torch.float32 and result.final.device == c.device and result.final.requires_grad
    assert result.n_steps.dtype == torch.int64 and result.n_steps.device.type == "cpu"
    assert torch.equal(result.n_steps, torch.tensor([4, 3, 0, 4])) and len(result.steps) == 4
    last = {b: c[b:b+1] for b in range(4)}
    for step, (indices, start, delta, segments), (before, actual) in zip(result.steps, plan, recorded):
        assert torch.equal(step.batch_index, torch.tensor(indices)) and step.batch_index.device.type == "cpu"
        assert step.batch_index.dtype == torch.int64 and step.segment_index.dtype == torch.int64
        assert torch.equal(step.segment_index, torch.tensor(segments)) and step.segment_index.device.type == "cpu"
        for tensor, expected in ((step.start_time, start), (step.delta, delta)):
            assert tensor.dtype == torch.float64 and tensor.device.type == "cpu" and not tensor.requires_grad
            torch.testing.assert_close(tensor, torch.tensor(expected, dtype=torch.float64), atol=1e-16, rtol=1e-14)
        assert (step.delta > 0).all() and ((step.start_time+step.delta) > step.start_time).all()
        assert step.before is before and step.result is actual and step.before.requires_grad
        close(step.before, numpy64(torch.cat([last[b] for b in indices])))
        for i, b in enumerate(indices):
            last[b] = actual.coefficients[i:i+1]
    expected = []
    for b, row in enumerate(rows):
        state = c[b:b+1].cpu().double()
        for start, end, command in row:
            length = min(end, float(duration[b]))-start
            if length <= 0:
                break
            gamma, omega = (getattr(context.coefficients, name)[b:b+1] for name in ("gamma", "omega"))
            e, h = integration_reference(gamma, omega, torch.tensor([length], dtype=torch.float64))
            forcing = torch.einsum("bkcm,m->bkc", context.coefficients.b_matrix[b:b+1].cpu().double(), torch.tensor(command, dtype=torch.float64))
            state = (e@state.reshape(1, 3, 32, 2, 1)+h@forcing.reshape(1, 3, 32, 2, 1)).reshape_as(state)
        expected.append(state)
    errors = dict(analytic=close(result.final, numpy64(torch.cat(expected))))
    if zero_rates:
        additive = numpy64(c).copy()
        for b, row in enumerate(rows):
            for start, end, command in row:
                length = max(0., min(end, float(duration[b]))-start)
                additive[b] += length*np.einsum("kcm,m->kc", numpy64(context.coefficients.b_matrix[b]), command)
        errors["zero_A"] = close(result.final, additive)
    monkeypatch.setattr(dynamics, "exponential_euler_step", original_step)
    for b, row in enumerate(rows):
        solo = dynamics.integrate_segments(c[b:b+1], d11_controls([row], device=device),
            d11_context_rows(context, [b]), duration=duration[b:b+1], config=integration_config())
        close(solo.final, numpy64(result.final[b:b+1]))
        assert solo.steps == () and solo.n_steps.item() == result.n_steps[b].item()
    print(json.dumps(dict(probe="D11-piecewise-analytic-hand-trace", device=device, zero_rates=zero_rates,
        errors=errors, n_steps=result.n_steps.tolist(), rounds=len(result.steps))))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("duration,max_step,expected_steps", [(1e-21, .25, 1), (1e-13, .25, 1),
    (1., np.nextafter(1., 0.), 2)])
def test_d11_positive_tiny_intervals_and_last_representable_remainder(device, duration, max_step, expected_steps):
    """小于端点容差仍执行正步；边界前一可表示点仍有独立正末步，F32差可为0。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device)
    c, context = d11_constant_context(dynamics, batch=1, device=device)
    controls = d11_controls([[(0., duration, (1., -.5))]], device=device)
    result = dynamics.integrate_segments(c, controls, context, duration=controls.duration,
        config=integration_config(max_step=max_step), collect_trace=True)
    assert result.n_steps.item() == expected_steps and len(result.steps) == expected_steps
    end = 0.
    for step in result.steps:
        assert step.start_time.item() == end and step.delta.item() > 0
        end = step.start_time.item()+step.delta.item()
        assert end > step.start_time.item()
    assert end == duration
    if expected_steps == 2:
        assert result.steps[-1].delta.item() == 1.-np.nextafter(1., 0.)
    print(json.dumps(dict(probe="D11-positive-time-progress", device=device, duration=duration,
        n_steps=result.n_steps.item(), final_time=end, deltas=[s.delta.item() for s in result.steps])))


@pytest.mark.parametrize("kind", ["tiny-gap", "tiny-overlap", "start-gap", "end-gap", "tiny-uncovered-tail", "uncovered-before-clip"])
def test_d11_strict_original_coverage_rejects_subtolerance_defects(kind, monkeypatch):
    """原始端点小于1e-12的真实缺口/重叠仍拒绝，不能先裁剪或大物理原点吞掉。"""
    dynamics = hand_complete_dynamics()
    c, context = d11_constant_context(dynamics, batch=1)
    controls = d11_controls([[(0., .5, (1., 0.)), (.5, 1., (0., 1.))]])
    query = torch.tensor([.25], dtype=torch.float64)
    if kind in ("tiny-gap", "tiny-overlap"):
        controls.starts[0, 1] = np.nextafter(.5, 1. if kind == "tiny-gap" else 0.)
    elif kind == "start-gap":
        controls.starts[0, 0] = 1e-20
    elif kind == "end-gap":
        controls.ends[0, 1] = np.nextafter(1., 0.)
    elif kind == "tiny-uncovered-tail":
        controls = d11_controls([[(0., 1e-21, (1., 0.))]])
        query = torch.tensor([1e-20], dtype=torch.float64)
    else:
        controls.starts[0, 1] = .6
    def forbidden(*args, **kwargs):
        """覆盖错误必须在物理积分前拒绝。"""
        pytest.fail("invalid original coverage reached D10")
    monkeypatch.setattr(dynamics, "exponential_euler_step", forbidden)
    with pytest.raises((TypeError, ValueError)):
        dynamics.integrate_segments(c, controls, context, duration=query, config=integration_config())


@pytest.mark.parametrize("kind", ["inactive", "nonadvance", "cross-boundary", "rewind"])
def test_d11_broken_schedule_cannot_fake_completion_or_spin(kind, monkeypatch):
    """调度器若异常，入口拒绝未完成inactive、不可表示推进和越界，不能静默到终点。"""
    from po_jepa import actions
    dynamics = hand_complete_dynamics()
    c, context = d11_constant_context(dynamics, batch=1)
    controls = d11_controls([[(0., .5, (1., 0.)), (.5, 1., (0., 1.))]])
    original = actions.next_step_schedule
    calls = []
    def broken(*args, **kwargs):
        """注入helper故障以检查正推进合同；不是正常时钟的预期生成器。"""
        value = original(*args, **kwargs)
        calls.append(value)
        if kind == "inactive":
            return replace(value, active=torch.zeros_like(value.active), delta=torch.zeros_like(value.delta))
        if kind == "nonadvance":
            return replace(value, delta=torch.full_like(value.delta, torch.finfo(torch.float64).tiny)) if len(calls) > 1 else value
        if kind == "cross-boundary":
            return replace(value, delta=torch.ones_like(value.delta))
        return replace(value, time=value.time-.01)
    monkeypatch.setattr(actions, "next_step_schedule", broken)
    if hasattr(api(), "next_step_schedule"):
        monkeypatch.setattr(api(), "next_step_schedule", broken)
    with pytest.raises((ValueError, RuntimeError)):
        dynamics.integrate_segments(c, controls, context, duration=controls.duration, config=integration_config())
    assert len(calls) <= 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("r", [1, 2])
@pytest.mark.parametrize("trace", [False, True])
def test_d11_real_d09_chain_multistep_gradients_match_manual_reference(device, r, trace):
    """真实P_in/C04/D09和跨时间反传；手列子步+独立E/H，新图各一次联合backward。

    参考固定实际C04的lambda/Psi，用已验收D09 RHS和增广matrix_exp，避免把
    被测调度、D10传播或trace当成期望。r1评分及标量bias的合法零梯度保留。
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(1703)
    dynamics = hand_complete_dynamics(device, r=r)
    graph = GraphSpectrum(GraphConfig(16, 1703, .25, 1., 128, 1e-10, 1e-8)).to(device)
    model = POJEPA(ProjectionConfig(5, 64), graph, dynamics, object()).to(device)
    raw_z = torch.linspace(-.8, .9, 90, device=device).reshape(3, 6, 5).requires_grad_()
    x = model.project_in(raw_z)
    spectrum = graph.lowest_eigenpairs(graph.normalized_laplacian(graph.affinity(x, regular_grid(2, 3))).laplacian,
        k=3, model_device=torch.device(device))
    c = modal_analysis(x, spectrum.basis)
    x.retain_grad(); c.retain_grad()
    rows = [[(0., .2, (-.3, .6)), (.2, .6, (.8, -.4)), (.6, 1., (2., 1.))],
        [(0., .9, (.5, -.7))], [(0., 1., (1.1, .3))]]
    controls = d11_controls(rows, device=device)
    values = controls.u_values.detach().clone().requires_grad_()
    controls = replace(controls, u_values=values)
    duration = torch.tensor([.55, .125, 0.], dtype=torch.float64)
    context = dynamics.prepare_context(x, spectrum.eigenvalues, spectrum.basis, regular_grid(2, 3).coords)
    retained = dict(pooled=context.pooled, raw_gamma=context.coefficients.raw_gamma,
        gamma=context.coefficients.gamma, omega=context.coefficients.omega,
        b_matrix=context.coefficients.b_matrix, spectral_bias=context.spectral_bias, spatial_bias=context.spatial_bias)
    for value in retained.values():
        value.retain_grad()
    actual = dynamics.integrate_segments(c, controls, context, duration=duration,
        config=integration_config(), collect_trace=trace)
    upstream = 8*(torch.cos(torch.arange(c.numel(), device=device).float()*.113).reshape_as(c)+.2)
    (actual.final*upstream).sum().backward()
    other = hand_complete_dynamics(device, r=r)
    other.load_state_dict(dynamics.state_dict())
    other_model = POJEPA(ProjectionConfig(5, 64), graph, other, object()).to(device)
    other_model.p_in.load_state_dict(model.p_in.state_dict())
    ref_z = raw_z.detach().clone().requires_grad_()
    ref_x = other_model.project_in(ref_z)
    ref_c = modal_analysis(ref_x, spectrum.basis)
    ref_x.retain_grad(); ref_c.retain_grad()
    ref_values = values.detach().clone().requires_grad_()
    ref_context = other.prepare_context(ref_x, spectrum.eigenvalues, spectrum.basis, regular_grid(2, 3).coords)
    ref_retained = dict(pooled=ref_context.pooled, raw_gamma=ref_context.coefficients.raw_gamma,
        gamma=ref_context.coefficients.gamma,
        omega=ref_context.coefficients.omega, b_matrix=ref_context.coefficients.b_matrix,
        spectral_bias=ref_context.spectral_bias, spatial_bias=ref_context.spatial_bias)
    for value in ref_retained.values():
        value.retain_grad()
    manual_steps = [[(0, .2), (1, .25), (1, .1)], [(0, .125)], []]
    finals = []
    for b, steps in enumerate(manual_steps):
        state = ref_c[b:b+1]
        local_context = d11_context_rows(ref_context, [b])
        for segment, delta in steps:
            rhs = other.rhs(state, ref_values[b:b+1, segment], local_context)
            e, h = integration_reference(local_context.coefficients.gamma, local_context.coefficients.omega,
                torch.tensor([delta], dtype=torch.float64))
            e, h = e.to(device=device, dtype=torch.float32), h.to(device=device, dtype=torch.float32)
            forcing = rhs.action_linear+rhs.interaction.nonlinear
            state = (e@state.reshape(1, 3, 32, 2, 1)+h@forcing.reshape(1, 3, 32, 2, 1)).reshape_as(state)
        finals.append(state)
    expected = torch.cat(finals)
    (expected*upstream).sum().backward()
    errors = dict(forward=close(actual.final, numpy64(expected)), c=close(c.grad, numpy64(ref_c.grad)),
        x=close(x.grad, numpy64(ref_x.grad)), raw_z=close(raw_z.grad, numpy64(ref_z.grad)),
        action=close(values.grad, numpy64(ref_values.grad)))
    assert torch.equal(actual.n_steps, torch.tensor([3, 1, 0]))
    assert len(actual.steps) == (3 if trace else 0)
    close(c.grad[2], numpy64(upstream[2]))
    assert values.grad[0, :2].count_nonzero() and values.grad[1, 0].count_nonzero()
    assert values.grad[0, 2].count_nonzero() == 0 and values.grad[2].count_nonzero() == 0
    assert values.grad[1, 1:].count_nonzero() == 0 and torch.isfinite(values.grad).all()
    for name, value in retained.items():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
        errors[name] = close(value.grad, numpy64(ref_retained[name].grad))
        assert value.grad[2].count_nonzero() == 0, name
        if name not in ("spectral_bias", "spatial_bias"):
            assert value.grad[:2].count_nonzero(), name
    for name, value in model.p_in.named_parameters():
        errors["p_in."+name] = close(value.grad, numpy64(dict(other_model.p_in.named_parameters())[name].grad))
    zero_names = {"spectral_bias_net.2.bias", "spatial_bias_net.2.bias"}
    for name, parameter in dynamics.named_parameters():
        expected_gradient = dict(other.named_parameters())[name].grad
        assert parameter.grad is not None and expected_gradient is not None, name
        errors[name] = close(parameter.grad, numpy64(expected_gradient))
        if name in zero_names or (r == 1 and name.startswith(("q_proj.", "k_proj.", "spectral_bias_net.", "spatial_bias_net."))):
            close(parameter.grad, np.zeros(parameter.shape))
        else:
            assert parameter.grad.count_nonzero(), name
    assert model.p_out.weight.grad is None and model.p_out.bias.grad is None
    assert not spectrum.eigenvalues.requires_grad and not spectrum.basis.requires_grad
    print(json.dumps(dict(probe="D11-real-chain-manual-multistep-gradient", device=device, top_r=r,
        trace=trace, n_steps=actual.n_steps.tolist(), errors=errors)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_d11_trace_switch_preserves_results_counts_and_gradients(device):
    """trace只是同次结果记录；两张相同新图各一次联合反向，全部梯度一致。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    outputs = []
    for trace in (False, True):
        dynamics = hand_complete_dynamics(device)
        if outputs:
            dynamics.load_state_dict(outputs[0][0])
        x, eigenvalues, basis, coords, c, _ = complete_inputs(device)
        x.requires_grad_(); c.requires_grad_()
        controls = d11_controls([[(0., .2, (.5, -.3)), (.2, 1., (.8, .4))],
            [(0., 1., (-.6, .2))]], device=device)
        values = controls.u_values.detach().clone().requires_grad_()
        context = dynamics.prepare_context(x, eigenvalues, basis, coords)
        result = dynamics.integrate_segments(c, replace(controls, u_values=values), context,
            duration=torch.tensor([.65, .125], dtype=torch.float64), config=integration_config(), collect_trace=trace)
        result.final.square().sum().backward()
        gradients = {"x": x.grad, "c": c.grad, "action": values.grad,
            **{name: p.grad for name, p in dynamics.named_parameters()}}
        outputs.append((dynamics.state_dict(), result, gradients))
    assert outputs[0][1].steps == () and len(outputs[1][1].steps) == 3
    assert torch.equal(outputs[0][1].n_steps, outputs[1][1].n_steps)
    assert torch.equal(outputs[0][1].final, outputs[1][1].final)
    for name, gradient in outputs[0][2].items():
        assert gradient is not None and torch.equal(gradient, outputs[1][2][name]), name


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_d11_query_lifecycle_and_shared_prefix_recompute_current_n_only(device, monkeypatch):
    """一次上下文后不重做Pool/A/B/bias；每真实步重算内容、选择、M_a，前缀无尾时长依赖。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device)
    x, eigenvalues, basis, coords, c, _ = complete_inputs(device)
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    controls = d11_controls([[(0., .3, (.4, -.2)), (.3, 1., (-.1, .7))],
        [(0., .125, (.5, -.3)), (.125, 1., (.8, .4))]], device=device)
    calls = {name: [] for name in ("content_terms", "sparse_attention", "action_residual")}
    def forbidden(*args, **kwargs):
        """查询起点量已存在，循环中不可重建。"""
        pytest.fail("integration recomputed query context")
    for name in ("pool_context", "condition", "spectral_bias", "spatial_descriptors", "spatial_bias", "prepare_context"):
        monkeypatch.setattr(dynamics, name, forbidden)
    for name in calls:
        original = getattr(dynamics, name)
        def record(*args, _name=name, _original=original, **kwargs):
            """计数并保留实际D09计算，捕获当前输入但不额外执行网络。"""
            calls[_name].append(args[0])
            return _original(*args, **kwargs)
        monkeypatch.setattr(dynamics, name, record)
    results = []
    for duration in ([.8, .125], [.8, .55], [.64, .45]):
        for values in calls.values():
            values.clear()
        result = dynamics.integrate_segments(c, controls, context,
            duration=torch.tensor(duration, dtype=torch.float64), config=integration_config(), collect_trace=True)
        assert all(len(values) == len(result.steps) for values in calls.values())
        assert all(torch.equal(actual, step.before) for actual, step in zip(calls["content_terms"], result.steps))
        assert all(torch.equal(actual, step.before) for actual, step in zip(calls["action_residual"], result.steps))
        assert not torch.equal(result.steps[0].before[0], result.steps[1].before[0])
        results.append(result)
    def sample_prefix(result, b):
        """抽取某原样本的实际步；不从schedule反算期望。"""
        return [(step, int(torch.nonzero(step.batch_index == b).item()))
            for step in result.steps if (step.batch_index == b).any()]
    a, b, short = (sample_prefix(result, 0) for result in results)
    for (left, li), (right, ri) in zip(a, b):
        assert left.start_time[li] == right.start_time[ri] and left.delta[li] == right.delta[ri]
        # active批大小改变GEMM舍入；沿用固定F32容差，不要求跨批形状bitwise相等。
        close(left.before[li], numpy64(right.before[ri]))
        close(left.result.coefficients[li], numpy64(right.result.coefficients[ri]))
    # 第四步被短查询截断，只比较共有的前三个完整步，不伪称半群。
    for (left, li), (right, ri) in zip(a[:3], short[:3]):
        assert left.start_time[li] == right.start_time[ri] and left.delta[li] == right.delta[ri]
        close(left.result.coefficients[li], numpy64(right.result.coefficients[ri]))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_d11_noncontiguous_inputs_preserve_values_gradients_members_modes_and_rng(device):
    """查询支持非连续输入并保持状态；已有grad只读，推理模式不建图或缓存trace。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics(device).eval()
    x, eigenvalues, basis, coords, c, _ = complete_inputs(device)
    c = torch.stack((c, c), -1)[..., 0].requires_grad_()
    controls = d11_controls([[(0., .2, (.5, -.3)), (.2, .9, (.8, .4))],
        [(0., 1., (-.6, .2))]], device=device)
    values = torch.stack((controls.u_values, controls.u_values), -1)[..., 0].requires_grad_()
    controls = replace(controls, u_values=values)
    duration = torch.tensor([.65, 9., .125, 9.], dtype=torch.float64)[::2]
    assert not c.is_contiguous() and not values.is_contiguous() and not duration.is_contiguous()
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    tensors = [c, values, duration, controls.starts, controls.ends, controls.duration,
        controls.n_segments, controls.segment_valid, controls.source_origin_seconds,
        context.eigenvalues, context.basis, context.coords, context.pooled, context.spectral_bias,
        context.spatial_bias, *(getattr(context.coefficients, f.name) for f in fields(context.coefficients))]
    saved = [(value.detach().clone(), value._version) for value in tensors]
    for value in (c, values):
        value.grad = torch.full_like(value, .3)
    for parameter in dynamics.parameters():
        parameter.grad = torch.full_like(parameter, .2)
    grads = [(value, id(value.grad), value.grad.clone(), value.grad._version)
        for value in (c, values, *dynamics.parameters())]
    parameters = {name: (id(value), value.detach().clone(), value._version) for name, value in dynamics.named_parameters()}
    members = {name: set(vars(layer)) for name, layer in dynamics.named_modules()}
    modes = {name: layer.training for name, layer in dynamics.named_modules()}
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    defaults = torch.get_default_dtype(), torch.get_default_device()
    result = dynamics.integrate_segments(c, controls, context, duration=duration,
        config=integration_config(), collect_trace=True)
    assert result.final.requires_grad and result.final is not c and result.steps
    for grad_context in (torch.no_grad, torch.inference_mode):
        with grad_context():
            inferred = dynamics.integrate_segments(c, controls, context, duration=duration,
                config=integration_config(), collect_trace=False)
            close(inferred.final, numpy64(result.final))
            assert not inferred.final.requires_grad and inferred.steps == ()
    for value, (old, version) in zip(tensors, saved):
        torch.testing.assert_close(value, old, atol=0., rtol=0., equal_nan=True)
        assert value._version == version
    for value, identifier, old, version in grads:
        assert id(value.grad) == identifier and torch.equal(value.grad, old) and value.grad._version == version
    for name, parameter in dynamics.named_parameters():
        identifier, old, version = parameters[name]
        assert id(parameter) == identifier and torch.equal(parameter, old) and parameter._version == version
    assert len(parameters) == 23 and sum(p.numel() for p in dynamics.parameters()) == 25634+4224*2
    assert members == {name: set(vars(layer)) for name, layer in dynamics.named_modules()}
    assert modes == {name: layer.training for name, layer in dynamics.named_modules()}
    assert list(dynamics.named_buffers()) == [] and not hasattr(dynamics, "context")
    assert "forward" not in type(dynamics).__dict__
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert defaults == (torch.get_default_dtype(), torch.get_default_device())


@pytest.mark.parametrize("target,kind", [("initial", kind) for kind in
    ("not-tensor", "double", "sparse", "meta", "nan", "inf", "batch", "modes", "width", "rank", "empty")]
    + [("duration", kind) for kind in ("not-tensor", "float32", "grad", "nan", "inf", "negative", "batch", "rank", "meta")]
    + [("path-duration", kind) for kind in ("not-tensor", "float32", "grad", "nan", "inf", "negative", "batch", "rank", "meta")])
def test_d11_rejects_invalid_initial_and_time_contract(target, kind):
    """初态与查询/路径时间入口拒绝错误合同，覆盖比较之前也不接受错误元数据。"""
    dynamics = hand_complete_dynamics()
    x, eigenvalues, basis, coords, c, _ = complete_inputs()
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    controls = d11_controls([[(0., 1., (.4, -.2))], [(0., 1., (.6, .1))]])
    duration = torch.tensor([.5, .25], dtype=torch.float64)
    value = c if target == "initial" else duration
    if kind == "not-tensor":
        value = [1.]
    elif kind in ("double", "float32"):
        value = value.to(dtype=torch.float64 if kind == "double" else torch.float32)
    elif kind == "sparse":
        value = value.to_sparse()
    elif kind == "meta":
        value = value.to(device="meta")
    elif kind == "grad":
        value = value.clone().requires_grad_()
    elif kind in ("nan", "inf", "negative"):
        value = value.clone(); value.flatten()[0] = dict(nan=float("nan"), inf=float("inf"), negative=-.1)[kind]
    elif kind == "batch":
        value = value[:1]
    elif kind == "rank":
        value = value.unsqueeze(0)
    elif kind == "modes":
        value = value[:, :2]
    elif kind == "width":
        value = value[..., :63]
    else:
        value = value[:0]
    if target == "initial":
        c = value
    elif target == "duration":
        duration = value
    else:
        controls = replace(controls, duration=value)
    with pytest.raises((TypeError, ValueError)):
        dynamics.integrate_segments(c, controls, context, duration=duration, config=integration_config())


@pytest.mark.parametrize("target,kind", [("context", "object"), ("context", "batch"), ("context", "double"),
    ("controls", "object"), ("controls", "batch"), ("controls", "action-dim"), ("controls", "not-tensor"),
    ("controls", "double"), ("controls", "sparse"), ("controls", "meta"), ("controls", "nan"),
    ("controls", "inf"), ("controls", "valid-mask"), ("controls", "zero-length"),
    ("config", "object"), ("config", "endpoint"), ("trace", "integer"), ("trace", "none"), ("trace", "tensor")])
def test_d11_rejects_invalid_controls_context_configuration_and_trace(target, kind):
    """控制/上下文维数、精度、有效值与配置/strict bool合同均明确失败。"""
    dynamics = hand_complete_dynamics()
    x, eigenvalues, basis, coords, c, _ = complete_inputs()
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    controls = d11_controls([[(0., 1., (.4, -.2))], [(0., 1., (.6, .1))]])
    configuration, trace = integration_config(), False
    if target == "context":
        if kind == "object":
            context = object()
        elif kind == "batch":
            context = d11_context_rows(context, [0])
        else:
            context = replace(context, pooled=context.pooled.double())
    elif target == "controls":
        if kind == "object":
            controls = object()
        elif kind == "batch":
            controls = d11_controls([[(0., 1., (.4, -.2))]])
        elif kind == "action-dim":
            spec = replace(controls.control_spec, action_dim=1, channel_names=("fictional",), channel_units=("unit",),
                lower_bounds=(-100.,), upper_bounds=(100.,), scale=(1.,), offset=(0.,))
            controls = replace(controls, u_values=controls.u_values[..., :1], control_spec=spec)
        elif kind == "valid-mask":
            controls = replace(controls, segment_valid=~controls.segment_valid)
        elif kind == "zero-length":
            controls.ends[0, 0] = 0.
        else:
            values = controls.u_values
            if kind == "not-tensor":
                values = []
            elif kind == "double":
                values = values.double()
            elif kind == "sparse":
                values = values.to_sparse()
            elif kind == "meta":
                values = values.to(device="meta")
            else:
                values = values.clone(); values[0, 0, 0] = float(kind)
            controls = replace(controls, u_values=values)
    elif target == "config":
        configuration = object() if kind == "object" else integration_config(endpoint_rtol=1e-10)
    else:
        trace = dict(integer=1, none=None, tensor=torch.tensor(True))[kind]
    with pytest.raises((TypeError, ValueError)):
        dynamics.integrate_segments(c, controls, context, duration=torch.tensor([.5, .25], dtype=torch.float64),
            config=configuration, collect_trace=trace)


@pytest.mark.parametrize("target", ["initial", "controls", "duration", "path-duration", "context", "model"])
def test_d11_actual_cuda_device_mismatch_rejected(target):
    """真实CPU/CUDA边界不隐式搬初态、控制或元数据；只允许模型侧显式索引复制。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dynamics = hand_complete_dynamics("cuda")
    x, eigenvalues, basis, coords, c, _ = complete_inputs("cuda")
    context = dynamics.prepare_context(x, eigenvalues, basis, coords)
    controls = d11_controls([[(0., 1., (.4, -.2))], [(0., 1., (.6, .1))]], device="cuda")
    duration = torch.tensor([.5, .25], dtype=torch.float64)
    if target == "initial":
        c = c.cpu()
    elif target == "controls":
        controls = replace(controls, u_values=controls.u_values.cpu())
    elif target == "duration":
        duration = duration.cuda()
    elif target == "path-duration":
        controls = replace(controls, duration=controls.duration.cuda())
    elif target == "context":
        context = replace(context, pooled=context.pooled.cpu())
    else:
        dynamics = dynamics.cpu()
    with pytest.raises((TypeError, ValueError)):
        dynamics.integrate_segments(c, controls, context, duration=duration, config=integration_config())
