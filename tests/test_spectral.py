"""C02—C05：虚构双核、L、最低K谱及欧氏模态公式验收，不代表完整Phi。"""

from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib
import importlib.util
import math
import random

import numpy as np
import pytest
import torch

from po_jepa.geometry import GridSpec, regular_grid
from po_jepa.model import POJEPA, ProjectionConfig


def api():
    """首次RED明确指出C02模块尚未编写，不伪造未来谱接口。"""
    assert importlib.util.find_spec("po_jepa.spectral") is not None, "C02 spectral 尚未实现"
    return importlib.import_module("po_jepa.spectral")


def config(**changes):
    """正式七项固定选择；非单位带宽只用于明确标记的公式诊断。"""
    return api().GraphConfig(**(dict(q_dim=16, q_seed=1703, sigma_x=.25, sigma_z=1.,
                                   max_modes=128, diagnostic_atol=1e-10, diagnostic_rtol=1e-8) | changes))


def controlled_graph():
    """仅在测试实例中设Q=[I16|0]、sigma_z=2，让两个平方分母都可手算。"""
    graph = api().GraphSpectrum(config(sigma_z=2.))
    with torch.no_grad():
        graph.q.copy_(torch.eye(16, 64))
    return graph


def test_dual_affinity_matches_hand_calculation_and_fixed_q():
    """式5独立标量exp(-5)：排除漏平方、额外1/2、加核和全64维距离。"""
    graph = controlled_graph()
    grid = regular_grid(1, 2)
    x = torch.zeros(2, 2, 64)
    x[0, 1, 0], x[1, 1, 0] = 2., 4.
    w = graph.affinity(x, grid)
    expected = torch.tensor([[[1., math.exp(-5)], [math.exp(-5), 1.]],
                             [[1., math.exp(-8)], [math.exp(-8), 1.]]], dtype=torch.float64)
    torch.testing.assert_close(w, expected, atol=1e-10, rtol=1e-8)
    no_feature = graph.affinity(torch.zeros_like(x), grid)
    torch.testing.assert_close(no_feature[:, 0, 1], torch.full((2,), math.exp(-4), dtype=torch.float64), atol=1e-10, rtol=1e-8)
    nullspace = torch.zeros_like(x)
    nullspace[:, 1, 63] = 20.
    assert torch.equal(graph.affinity(nullspace, grid), no_feature)
    changed_grid = GridSpec(1, 2, torch.tensor([[.375, .5], [.625, .5]], dtype=torch.float64),
                            "fictional C02 spatial perturbation only")
    changed = graph.affinity(x, changed_grid)
    torch.testing.assert_close(changed[:, 0, 1], torch.tensor([math.exp(-2), math.exp(-5)], dtype=torch.float64), atol=1e-10, rtol=1e-8)
    assert list(graph.named_parameters()) == []
    assert not w.requires_grad and w.grad_fn is None


def scalar_reference(x, coords, q, sigma_x, sigma_z):
    """逐对逐通道Python标量求和，不复用生产距离计算或矩阵乘法。"""
    batch, tokens, _ = x.shape
    values = []
    for b in range(batch):
        projected = [[sum(float(q[a, c])*float(x[b, i, c]) for c in range(64))
                      for a in range(16)] for i in range(tokens)]
        rows = []
        for i in range(tokens):
            row = []
            for j in range(tokens):
                ds = sum((float(coords[i, c])-float(coords[j, c]))**2 for c in range(2))
                dz = sum((projected[i][a]-projected[j][a])**2 for a in range(16))
                row.append(math.exp(-ds/float(sigma_x)**2)*math.exp(-dz/float(sigma_z)**2))
            rows.append(row)
        values.append(rows)
    return torch.tensor(values, dtype=torch.float64)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_formal_seed_scalar_reference_recompute_and_input_immutability(device):
    """真实seed Q配非方形网格；CUDA只测小输入→CPU构图，不代表完整训练。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; mandatory CPU test remains")
    graph = api().GraphSpectrum(config()).to(device)
    grid = regular_grid(2, 3)
    x = (torch.arange(2*64*6, dtype=torch.float32).reshape(2, 64, 6) % 17 / 13).transpose(1, 2).to(device)
    x.requires_grad_()
    assert not x.is_contiguous()
    before, coords = x.detach().clone(), grid.coords.clone()
    buffers = {n: (id(v), v.clone()) for n, v in graph.named_buffers()}
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    w = graph.affinity(x, grid)
    expected = scalar_reference(x.detach().cpu(), coords, graph.q.cpu(), graph.sigma_x.cpu(), graph.sigma_z.cpu())
    torch.testing.assert_close(w, expected, atol=1e-10, rtol=1e-8)
    assert w.dtype == torch.float64 and w.device.type == "cpu" and not w.requires_grad
    assert w.shape == (2, 6, 6) and torch.equal(w, w.transpose(-1, -2))
    assert torch.equal(w.diagonal(dim1=-2, dim2=-1), torch.ones(2, 6, dtype=torch.float64))
    assert torch.isfinite(w).all() and torch.all((w >= 0) & (w <= 1))
    assert torch.equal(graph.affinity(x, grid), w)
    perturbed = x.detach().clone()
    perturbed[:, 1, 0] += 2.
    torch.testing.assert_close(graph.affinity(perturbed, grid),
                               scalar_reference(perturbed.cpu(), coords, graph.q.cpu(), graph.sigma_x.cpu(), graph.sigma_z.cpu()), atol=1e-10, rtol=1e-8)
    assert not torch.equal(graph.affinity(perturbed, grid), w)
    assert torch.equal(x, before) and x.requires_grad and torch.equal(grid.coords, coords)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    for name, value in graph.named_buffers():
        assert id(value) == buffers[name][0] and torch.equal(value, buffers[name][1])


def test_q_generation_precision_rng_fixed_buffers_and_persistence(monkeypatch, tmp_path):
    """记录原生double QR与float32存储舍入；保存的是实际Q，不重新QR。"""
    s = api()
    qr = torch.linalg.qr
    captured = []

    def capture(matrix, mode="reduced"):
        """执行真实QR，只记录初始化数值供独立检查。"""
        u, r = qr(matrix, mode=mode)
        captured.append((matrix.clone(), u.clone(), r.clone(), mode))
        return u, r

    monkeypatch.setattr(torch.linalg, "qr", capture)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    py_rng, np_rng = random.getstate(), np.random.get_state()
    old_dtype, old_device = torch.get_default_dtype(), torch.get_default_device()
    cfg = config()
    try:
        torch.set_default_dtype(torch.float64)
        torch.set_default_device("meta")
        graph = s.GraphSpectrum(cfg)
        assert torch.get_default_dtype() == torch.float64 and torch.get_default_device().type == "meta"
    finally:
        torch.set_default_dtype(old_dtype)
        torch.set_default_device(old_device)
    assert len(captured) == 1
    matrix, u, r, mode = captured[0]
    assert matrix.shape == (64, 16) and matrix.device.type == "cpu" and matrix.dtype == torch.float64
    assert mode == "reduced"
    expected_g = torch.randn(64, 16, dtype=torch.float64, device="cpu", generator=torch.Generator(device="cpu").manual_seed(1703))
    assert torch.equal(matrix, expected_g)
    signs = torch.where(r.diagonal() < 0, -torch.ones(16, dtype=torch.float64), torch.ones(16, dtype=torch.float64))
    q64 = (u*signs).T
    torch.testing.assert_close(q64 @ q64.T, torch.eye(16, dtype=torch.float64), atol=1e-10, rtol=1e-8)
    assert torch.equal(graph.q, q64.float())
    torch.testing.assert_close(graph.q.double() @ graph.q.double().T, torch.eye(16, dtype=torch.float64), atol=1e-5, rtol=1e-4)
    assert graph.config is cfg and list(graph.named_parameters()) == []
    assert set(dict(graph.named_buffers())) == {"q", "sigma_x", "sigma_z"}
    assert set(graph.state_dict()) == {"q", "sigma_x", "sigma_z"}
    for v in graph.buffers():
        assert v.dtype == torch.float32 and v.device.type == "cpu" and not v.requires_grad
    assert graph.sigma_x.shape == graph.sigma_z.shape == ()
    assert graph.sigma_x.item() == .25 and graph.sigma_z.item() == 1.
    assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == py_rng
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]
    q_hash = hashlib.sha256(graph.q.contiguous().numpy().tobytes()).hexdigest()
    repeat = s.GraphSpectrum(cfg)
    assert torch.equal(graph.q, repeat.q)
    assert hashlib.sha256(repeat.q.contiguous().numpy().tobytes()).hexdigest() == q_hash
    assert not torch.equal(graph.q, s.GraphSpectrum(replace(cfg, q_seed=1704)).q)
    with pytest.raises(FrozenInstanceError):
        cfg.q_seed = 1704
    file = tmp_path/"fictional_fixed_buffers.pt"
    torch.save(graph.state_dict(), file)
    repeat.q.zero_()
    repeat.load_state_dict(torch.load(file, weights_only=True))
    for name, value in graph.state_dict().items():
        assert torch.equal(value, repeat.state_dict()[name])
    calls = len(captured)
    repeat.affinity(torch.zeros(1, 1, 64), regular_grid(1, 1))
    assert len(captured) == calls


def test_single_token_underflow_and_c01_gradient_boundary():
    """合法下溢不是缺边错误；只分离构图副本，不切断原投影训练路径。"""
    graph = controlled_graph()
    assert torch.equal(graph.affinity(torch.ones(3, 1, 64), regular_grid(1, 1)), torch.ones(3, 1, 1, dtype=torch.float64))
    far = torch.zeros(1, 2, 64)
    far[0, 1, 0] = torch.finfo(torch.float32).max
    assert torch.equal(graph.affinity(far, regular_grid(1, 2)), torch.eye(2, dtype=torch.float64).unsqueeze(0))
    model = POJEPA(ProjectionConfig(3, 64), graph, object(), object())
    raw = torch.tensor([[[1., 2., 3.], [2., -1., 4.]]], requires_grad=True)
    x = model.project_in(raw)
    x.retain_grad()
    before = x.detach().clone()
    w = graph.affinity(x, regular_grid(1, 2))
    assert not w.requires_grad and w.grad_fn is None and torch.equal(x, before) and x.requires_grad
    x.sum().backward()  # 仅用于检查梯度的调试标量，不是新增训练损失。
    expected_weight_grad = raw.detach().sum((0, 1)).expand(64, 3)
    assert torch.equal(model.p_in.weight.grad, expected_weight_grad)
    assert torch.equal(model.p_in.bias.grad, torch.full((64,), 2.))
    assert torch.equal(x.grad, torch.ones_like(x)) and raw.grad is not None
    assert all(v.grad is None for v in graph.buffers())


@pytest.mark.parametrize("changes", [{"q_dim":8}, {"q_dim":True}, {"q_seed":True}, {"q_seed":1.5},
    {"q_seed":2**64}, {"q_seed":-(2**63)-1}, {"max_modes":0}, {"max_modes":True},
    {"sigma_x":0.}, {"sigma_z":-1.}, {"sigma_x":math.inf}, {"sigma_z":math.nan},
    {"sigma_z":1e-300}, {"sigma_x":1e300}, {"diagnostic_atol":-1.}, {"diagnostic_rtol":math.inf}])
def test_bad_graph_config(changes):
    """拒绝非法配置及float32保存后失去有限正值的带宽，不加epsilon。"""
    with pytest.raises((TypeError, ValueError)):
        api().GraphSpectrum(config(**changes))


def test_constructor_requires_explicit_graph_config():
    """不接受未声明的配置对象，不暗建默认Q或默认配置。"""
    with pytest.raises(TypeError, match="GraphConfig"):
        api().GraphSpectrum(None)


@pytest.mark.parametrize("bad", ["object", "rank", "width", "batch", "dtype", "meta", "nan", "inf", "grid_type", "grid_tokens", "mutated_coords"])
def test_bad_affinity_inputs(bad):
    """检查输入/坐标信任边界，损坏的冻结dataclass内张量仍会被发现。"""
    graph = api().GraphSpectrum(config())
    x, grid = torch.zeros(1, 2, 64), regular_grid(1, 2)
    if bad == "object": x = None
    elif bad == "rank": x = x.unsqueeze(0)
    elif bad == "width": x = x[..., :-1]
    elif bad == "batch": x = torch.zeros(0, 2, 64)
    elif bad == "dtype": x = x.double()
    elif bad == "meta": x = x.to("meta")
    elif bad in ("nan", "inf"): x[0, 0, 0] = float(bad)
    elif bad == "grid_type": grid = None
    elif bad == "grid_tokens": grid = regular_grid(2, 2)
    else: grid.coords[0, 0] = math.nan
    with pytest.raises(TypeError if bad in ("object", "grid_type") else ValueError):
        graph.affinity(x, grid)


@pytest.mark.parametrize("bad", ["q_nan", "q_shape", "q_dtype", "q_gradient", "q_scale", "sigma_zero", "sigma_inf", "sigma_shape", "sigma_gradient"])
def test_corrupt_fixed_buffers_rejected(bad):
    """状态损坏要明确失败，不重新QR或替换带宽掩盖错误。"""
    graph = api().GraphSpectrum(config())
    if bad == "q_nan": graph.q[0, 0] = math.nan
    elif bad == "q_shape": graph.q = torch.zeros(15, 64)
    elif bad == "q_dtype": graph.q = graph.q.double()
    elif bad == "q_gradient": graph.q.requires_grad_()
    elif bad == "q_scale": graph.q.mul_(2.)
    elif bad == "sigma_zero": graph.sigma_z.zero_()
    elif bad == "sigma_inf": graph.sigma_x.fill_(math.inf)
    elif bad == "sigma_shape": graph.sigma_z = torch.ones(1)
    else: graph.sigma_x.requires_grad_()
    with pytest.raises(ValueError):
        graph.affinity(torch.zeros(1, 2, 64), regular_grid(1, 2))


def unequal_degree_fixture():
    """仅作式6代数验收的虚构W；手写预期，不依赖待测归一化函数。"""
    w = torch.tensor([[1., .5, 0.], [.5, 1., .25], [0., .25, 1.]], dtype=torch.float64)
    expected = torch.tensor([[1/3, -math.sqrt(2/21), 0.],
                             [-math.sqrt(2/21), 3/7, -1/math.sqrt(35)],
                             [0., -1/math.sqrt(35), 1/5]], dtype=torch.float64)
    return w, expected


def test_normalized_laplacian_uses_both_degree_axes():
    """非均匀度抓出I-D^-1W；第二个batch用不同W排除批次混度。"""
    graph = api().GraphSpectrum(config())
    w, expected = unequal_degree_fixture()
    inputs = torch.stack((w, torch.ones(3, 3, dtype=torch.float64)))
    before = inputs.clone()
    result = graph.normalized_laplacian(inputs)
    assert isinstance(result, api().LaplacianResult)
    assert torch.equal(result.degree, torch.tensor([[1.5, 1.75, 1.25], [3., 3., 3.]], dtype=torch.float64))
    torch.testing.assert_close(result.laplacian[0], expected, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(result.laplacian[1], torch.eye(3, dtype=torch.float64)-torch.ones(3, 3, dtype=torch.float64)/3, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(result.laplacian @ result.degree.sqrt().unsqueeze(-1), torch.zeros(2, 3, 1, dtype=torch.float64), atol=1e-10, rtol=1e-8)
    assert (result.laplacian[0] @ torch.ones(3, dtype=torch.float64)).abs().max() > .01
    assert torch.equal(inputs, before)
    for tensor in (result.degree, result.laplacian):
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float64 and not tensor.requires_grad
    with pytest.raises(FrozenInstanceError):
        result.degree = torch.ones_like(result.degree)


@pytest.mark.parametrize("kind", ["single", "identity", "complete", "pair"])
def test_normalized_laplacian_analytic_boundaries(kind):
    """孤立/不连通图合法；解析零模态及二节点谱不靠数值修补。"""
    graph = api().GraphSpectrum(config())
    if kind == "single":
        w = torch.ones(1, 1, 1, dtype=torch.float64)
        expected = torch.zeros_like(w)
        degree = torch.ones(1, 1, dtype=torch.float64)
        spectrum = torch.zeros(1, 1, dtype=torch.float64)
    elif kind == "identity":
        w = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        expected = torch.zeros_like(w)
        degree = torch.ones(1, 4, dtype=torch.float64)
        spectrum = torch.zeros(1, 4, dtype=torch.float64)
    elif kind == "complete":
        w = torch.ones(1, 4, 4, dtype=torch.float64)
        expected = torch.eye(4, dtype=torch.float64).unsqueeze(0)-w/4
        degree = torch.full((1, 4), 4., dtype=torch.float64)
        spectrum = torch.tensor([[0., 1., 1., 1.]], dtype=torch.float64)
    else:
        a = .5
        w = torch.tensor([[[1., a], [a, 1.]]], dtype=torch.float64)
        expected = torch.tensor([[[1., -1.], [-1., 1.]]], dtype=torch.float64)*a/(1+a)
        degree = torch.full((1, 2), 1+a, dtype=torch.float64)
        spectrum = torch.tensor([[0., 2*a/(1+a)]], dtype=torch.float64)
    result = graph.normalized_laplacian(w)
    assert torch.equal(result.degree, degree)
    torch.testing.assert_close(result.laplacian, expected, atol=1e-10, rtol=1e-8)
    if kind in ("single", "identity"):
        assert torch.equal(result.laplacian, expected)
    torch.testing.assert_close(torch.linalg.eigvalsh(result.laplacian), spectrum, atol=1e-10, rtol=1e-8)


def test_normalized_laplacian_independent_matrix_spectrum_and_no_production_eigh(monkeypatch):
    """独立矩阵乘法+仅测试求谱；生产路径不得提前调用eigh/eigvalsh。"""
    graph = api().GraphSpectrum(config())
    generator = torch.Generator(device="cpu").manual_seed(1803)
    w = torch.rand(2, 7, 7, dtype=torch.float64, generator=generator)
    w = (w+w.transpose(-1, -2))/2  # 只构造测试矩阵，生产函数不能这样修补输入。
    w.diagonal(dim1=-2, dim2=-1).fill_(1.)

    def forbidden(*args, **kwargs):
        """明确暴露生产C03提前求谱；离开patch作用域后再独立诊断。"""
        raise AssertionError("C03 production must not solve an eigenproblem")

    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg, "eigh", forbidden)
        patch.setattr(torch.linalg, "eigvalsh", forbidden)
        result = graph.normalized_laplacian(w)
    reference = []
    for matrix in w.numpy():
        d = np.sum(matrix, axis=1)
        scale = np.diag(1/np.sqrt(d))
        reference.append(np.eye(7)-scale @ matrix @ scale)
    torch.testing.assert_close(result.laplacian, torch.from_numpy(np.stack(reference)), atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(result.laplacian, result.laplacian.transpose(-1, -2), atol=1e-10, rtol=1e-8)
    eigenvalues = torch.linalg.eigvalsh(result.laplacian)
    assert eigenvalues.min() >= -1e-10 and eigenvalues.max() <= 2+1e-10+2*1e-8


def test_normalized_laplacian_accepts_tolerance_without_repair():
    """容差仅判断：细小不对称原样参与公式，不能平均、填对角或改W。"""
    graph = api().GraphSpectrum(config())
    w = torch.tensor([[[1., .5], [.5+1e-12, 1.]]], dtype=torch.float64)
    before = w.clone()
    result = graph.normalized_laplacian(w)
    expected_degree = torch.tensor([[1.5, 1.5+1e-12]], dtype=torch.float64)
    scale = np.diag(1/np.sqrt(expected_degree[0].numpy()))
    reference = torch.from_numpy(np.eye(2)-scale @ w[0].numpy() @ scale)
    assert torch.equal(w, before)
    torch.testing.assert_close(result.degree, expected_degree, atol=0., rtol=0.)
    torch.testing.assert_close(result.laplacian[0], reference, atol=1e-10, rtol=1e-8)
    assert result.laplacian[0, 0, 1] != result.laplacian[0, 1, 0]


def test_normalized_laplacian_c02_connection_and_no_side_effects():
    """真实C01→C02→C03局部连通；调试标量仍可经原X反传，不冒称完整Phi。"""
    graph = api().GraphSpectrum(config())
    graph.eval()
    model = POJEPA(ProjectionConfig(3, 64), graph, object(), object())
    z = torch.tensor([[[1., 2., 3.], [-1., 1., 4.]]], requires_grad=True)
    x = model.project_in(z)
    before_x = x.detach().clone()
    w = graph.affinity(x, regular_grid(1, 2))
    before_w = w.clone()
    buffers = {n: (id(t), t.clone()) for n, t in graph.named_buffers()}
    rng = torch.get_rng_state().clone()
    result = graph.normalized_laplacian(w)
    assert torch.equal(w, before_w) and torch.equal(x, before_x) and x.requires_grad
    assert not result.degree.requires_grad and not result.laplacian.requires_grad
    assert list(graph.named_parameters()) == [] and set(graph.state_dict()) == set(buffers)
    assert not graph.training and torch.equal(rng, torch.get_rng_state())
    for name, value in graph.named_buffers():
        assert id(value) == buffers[name][0] and torch.equal(value, buffers[name][1])
    x.sum().backward()  # 只是梯度诊断，不新增训练目标或优化器。
    assert torch.equal(model.p_in.weight.grad, z.detach().sum((0, 1)).expand(64, 3))
    assert torch.equal(model.p_in.bias.grad, torch.full((64,), 2.))


@pytest.mark.parametrize("bad", ["object", "sparse", "complex", "rank", "nonsquare", "batch", "nodes", "dtype", "device", "gradient", "nan", "inf", "asymmetric", "negative", "above_one", "diagonal", "zero", "diagonal_ulp"])
def test_normalized_laplacian_rejects_invalid_affinity(bad):
    """严格拒绝非法式5输出；不依赖不可达的度错误分支或放宽范围。"""
    graph = api().GraphSpectrum(config())
    w = torch.tensor([[[1., .5], [.5, 1.]]], dtype=torch.float64)
    if bad == "object": w = None
    elif bad == "sparse": w = w.to_sparse()
    elif bad == "complex": w = w.to(torch.complex128)
    elif bad == "rank": w = w[0]
    elif bad == "nonsquare": w = w[:, :1, :]
    elif bad == "batch": w = torch.zeros(0, 2, 2, dtype=torch.float64)
    elif bad == "nodes": w = torch.zeros(1, 0, 0, dtype=torch.float64)
    elif bad == "dtype": w = w.float()
    elif bad == "device": w = w.to("meta")
    elif bad == "gradient": w.requires_grad_()
    elif bad in ("nan", "inf"): w[0, 0, 1] = float(bad)
    elif bad == "asymmetric": w[0, 0, 1] = .2
    elif bad == "negative": w[0, 0, 1] = w[0, 1, 0] = -1e-12
    elif bad == "above_one": w[0, 0, 1] = w[0, 1, 0] = 1+1e-12
    elif bad == "diagonal": w[0, 0, 0] = .5
    elif bad == "zero": w.zero_()
    else: w[0, 0, 0] = torch.nextafter(torch.tensor(1., dtype=torch.float64), torch.tensor(2., dtype=torch.float64))
    with pytest.raises(TypeError if bad == "object" else ValueError):
        graph.normalized_laplacian(w)


def analytic_eigen_fixture(values):
    """独立解析U：±1/2精确正交；虚构合法谱仅用于求解算法验收。"""
    u = torch.tensor([[1., 1., 1., 1.], [1., 1., -1., -1.],
                      [1., -1., 1., -1.], [1., -1., -1., 1.]], dtype=torch.float64)/2
    eigenvalues = torch.tensor(values, dtype=torch.float64)
    return u @ torch.diag(eigenvalues) @ u.T, u, eigenvalues


def assert_eigen_diagnostics(result, matrix, k):
    """核对原生double诊断，模型float32副本用另一套既定精度标准。"""
    batch, nodes, _ = matrix.shape
    assert isinstance(result, api().SpectralResult)
    assert isinstance(result.diagnostics, api().SpectralDiagnostics)
    assert result.eigenvalues_cpu.shape == (batch, k) and result.basis_cpu.shape == (batch, nodes, k)
    d = result.diagnostics
    assert d.residual_norms.shape == (batch, k) and d.orthogonality_error.shape == (batch,)
    assert d.consecutive_gaps.shape == (batch, max(k-1, 0))
    assert d.sign_pivot_indices.shape == (batch, k) and d.sign_pivot_indices.dtype == torch.int64
    assert (d.cutoff_gap is None) == (k == nodes)
    if k < nodes:
        assert d.cutoff_gap.shape == (batch,)
    fields = [result.eigenvalues_cpu, result.basis_cpu, result.eigenvalues, result.basis,
              d.residual_norms, d.orthogonality_error, d.consecutive_gaps, d.sign_pivot_indices]
    if d.cutoff_gap is not None: fields.append(d.cutoff_gap)
    assert all(not t.requires_grad and t.grad_fn is None and torch.isfinite(t).all() for t in fields)
    for t in [result.eigenvalues_cpu, result.basis_cpu, d.residual_norms, d.orthogonality_error, d.consecutive_gaps]:
        assert t.dtype == torch.float64 and t.device.type == "cpu"
    assert result.eigenvalues.dtype == result.basis.dtype == torch.float32
    assert torch.equal(result.eigenvalues.cpu(), result.eigenvalues_cpu.float())
    assert torch.equal(result.basis.cpu(), result.basis_cpu.float())
    assert torch.all(d.residual_norms <= 1e-10+1e-8*result.eigenvalues_cpu.abs())
    assert torch.all(d.orthogonality_error <= 1e-10+1e-8*math.sqrt(k))
    assert torch.all(result.eigenvalues_cpu[:, 1:] >= result.eigenvalues_cpu[:, :-1])
    pivots = result.basis_cpu.abs().argmax(dim=1)
    assert torch.equal(d.sign_pivot_indices, pivots)
    assert torch.all(result.basis_cpu.gather(1, pivots[:, None, :]) >= 0)
    basis32, values32 = result.basis.cpu(), result.eigenvalues.cpu()
    residual32 = torch.linalg.vector_norm(matrix.detach().float() @ basis32 - basis32*values32[:, None, :], dim=1)
    assert torch.all(residual32 <= 1e-5+1e-4*values32.abs())
    torch.testing.assert_close(basis32.transpose(-1, -2) @ basis32, torch.eye(k).expand(batch, k, k), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("k", [1, 3, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_lowest_eigenpairs_residual_order_degeneracy_and_stop_gradient(k, device):
    """真实求解两批不同分离谱；投影比较消除求解器列符号的不唯一性。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; mandatory native CPU solve still tested")
    graph = api().GraphSpectrum(config())
    a, u, va = analytic_eigen_fixture([0., .25, .75, 1.])
    b, _, vb = analytic_eigen_fixture([0., .5, 1., 1.5])
    matrix = torch.stack((a, b)).requires_grad_()
    before = matrix.detach().clone()
    result = graph.lowest_eigenpairs(matrix, k=k, model_device=torch.device(device))
    assert_eigen_diagnostics(result, matrix, k)
    torch.testing.assert_close(result.eigenvalues_cpu, torch.stack((va[:k], vb[:k])), atol=1e-10, rtol=1e-8)
    for column in range(k):
        actual = result.basis_cpu[:, :, column:column+1]
        target = u[:, column:column+1] @ u[:, column:column+1].T
        torch.testing.assert_close(actual @ actual.transpose(-1, -2), target.expand(2, 4, 4), atol=1e-10, rtol=1e-8)
    if k < 4:
        torch.testing.assert_close(result.diagnostics.cutoff_gap, torch.stack((va[k]-va[k-1], vb[k]-vb[k-1])), atol=1e-10, rtol=1e-8)
    assert result.basis.device.type == device
    if device == "cuda": assert result.basis.device.index == torch.cuda.current_device()
    assert torch.equal(matrix, before) and matrix.requires_grad and matrix.grad is None
    matrix.sum().backward()  # 原L的其他调试路线仍可求梯度，eig输出没有图。
    assert torch.equal(matrix.grad, torch.ones_like(matrix))


def test_lowest_eigenpairs_exact_sign_ties_with_controlled_valid_solver(monkeypatch):
    """仅此测试注入合法解析特征对；±1/2确实并列，验证最小token索引。"""
    graph = api().GraphSpectrum(config())
    matrix, u, values = analytic_eigen_fixture([0., .25, .75, 1.])
    matrix = matrix.unsqueeze(0).requires_grad_()
    calls = []

    def known_solver(argument, UPLO="L"):
        """返回数学上真实的解析特征对，只人为翻转列符号以隔离符号逻辑。"""
        assert argument.dtype == torch.float64 and argument.device.type == "cpu" and not argument.requires_grad
        assert UPLO == "L"
        calls.append(UPLO)
        return values.unsqueeze(0), (u*torch.tensor([-1., 1., -1., 1.])).unsqueeze(0)

    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg, "eigh", known_solver)
        result = graph.lowest_eigenpairs(matrix, k=4, model_device=torch.device("cpu:0"))
    assert calls == ["L"]
    assert torch.equal(result.basis_cpu, u.unsqueeze(0))
    assert torch.equal(result.diagnostics.sign_pivot_indices, torch.zeros(1, 4, dtype=torch.int64))
    assert result.basis.device == torch.device("cpu")
    assert_eigen_diagnostics(result, matrix, 4)


def test_lowest_eigenpairs_repeated_subspace_cutoff_and_native_small_gaps():
    """重根整簇投影可比较；切入重根的部分子空间不唯一，间隙不改零。"""
    graph = api().GraphSpectrum(config())
    matrix = (torch.eye(4, dtype=torch.float64)-torch.ones(4, 4, dtype=torch.float64)/4).unsqueeze(0)
    complete = graph.lowest_eigenpairs(matrix, k=4, model_device=torch.device("cpu"))
    assert_eigen_diagnostics(complete, matrix, 4)
    repeated = complete.basis_cpu[:, :, 1:]
    torch.testing.assert_close(repeated @ repeated.transpose(-1, -2), matrix, atol=1e-10, rtol=1e-8)
    cut = graph.lowest_eigenpairs(matrix, k=2, model_device=torch.device("cpu"))
    assert_eigen_diagnostics(cut, matrix, 2)
    torch.testing.assert_close(matrix @ cut.basis_cpu[:, :, 1:], cut.basis_cpu[:, :, 1:], atol=1e-10, rtol=1e-8)
    assert torch.all(cut.diagnostics.cutoff_gap <= 1e-10+1e-8)
    near, _, _ = analytic_eigen_fixture([0., 1.-1e-9, 1., 1.])
    near = near.unsqueeze(0)
    result = graph.lowest_eigenpairs(near, k=4, model_device=torch.device("cpu"))
    assert_eigen_diagnostics(result, near, 4)
    gap = result.diagnostics.consecutive_gaps[0, 1]
    assert 0 < gap < 2e-9
    assert result.eigenvalues[0, 2]-result.eigenvalues[0, 1] == 0
    truncated = graph.lowest_eigenpairs(near, k=2, model_device=torch.device("cpu"))
    assert 0 < truncated.diagnostics.cutoff_gap[0] < 2e-9


def test_lowest_eigenpairs_single_token_and_no_clipping_or_asymmetry_repair():
    """N1精确边界、小负值原样保留；容差内不对称的诊断针对原L。"""
    graph = api().GraphSpectrum(config())
    single = graph.lowest_eigenpairs(torch.zeros(2, 1, 1, dtype=torch.float64), k=1, model_device=torch.device("cpu"))
    assert torch.equal(single.eigenvalues_cpu, torch.zeros(2, 1, dtype=torch.float64))
    assert torch.equal(single.basis_cpu, torch.ones(2, 1, 1, dtype=torch.float64))
    assert single.diagnostics.cutoff_gap is None and single.diagnostics.consecutive_gaps.shape == (2, 0)
    tiny = torch.diag(torch.tensor([-5e-11, .5, 2.+1e-8], dtype=torch.float64)).unsqueeze(0)
    result = graph.lowest_eigenpairs(tiny, k=3, model_device=torch.device("cpu"))
    assert torch.equal(result.eigenvalues_cpu, tiny.diagonal(dim1=-2, dim2=-1))
    asym = torch.tensor([[[.5, -.5+1e-9], [-.5, .5]]], dtype=torch.float64)
    before = asym.clone()
    result = graph.lowest_eigenpairs(asym, k=2, model_device=torch.device("cpu"))
    assert torch.equal(asym, before)
    assert result.diagnostics.residual_norms[0, 0] > 5e-10
    expected = math.sqrt(.5)*1e-9
    torch.testing.assert_close(result.diagnostics.residual_norms[0, 0], torch.tensor(expected, dtype=torch.float64), atol=1e-10, rtol=1e-8)


def test_lowest_eigenpairs_c02_c03_state_rng_and_compact_storage():
    """实际链路新算谱且不缓存；截断结果不占着不必要的完整特征向量存储。"""
    graph = api().GraphSpectrum(config())
    graph.eval()
    x = torch.arange(2*6*64, dtype=torch.float32).reshape(2, 6, 64) % 13 / 11
    matrix = graph.normalized_laplacian(graph.affinity(x, regular_grid(2, 3))).laplacian
    before = matrix.clone()
    state = {n: (id(v), v.clone()) for n, v in graph.named_buffers()}
    attributes = set(graph.__dict__)
    rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    for _ in range(2):
        result = graph.lowest_eigenpairs(matrix, k=2, model_device=torch.device("cpu"))
        assert_eigen_diagnostics(result, matrix, 2)
        assert result.basis_cpu.untyped_storage().nbytes() == result.basis_cpu.numel()*8
    assert not graph.training and torch.equal(matrix, before)
    assert set(graph.__dict__) == attributes and list(graph.named_parameters()) == [] and set(graph.state_dict()) == set(state)
    for n, value in graph.named_buffers(): assert id(value) == state[n][0] and torch.equal(value, state[n][1])
    assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == py_rng
    now = np.random.get_state()
    assert now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:]


@pytest.mark.parametrize("bad", ["object", "sparse", "complex", "rank", "nonsquare", "batch", "nodes", "dtype", "device", "nan", "inf", "asymmetric", "low_spectrum", "hidden_high_spectrum", "k_zero", "k_big", "k_bool", "k_float", "target_string", "target_meta", "target_cuda_index", "target_cuda_unavailable"])
def test_lowest_eigenpairs_rejects_invalid_inputs(bad, monkeypatch):
    """拒绝非法K/目标设备和整个谱越界，不静默截断或回退CPU。"""
    graph = api().GraphSpectrum(config())
    matrix = torch.diag(torch.tensor([0., .5, 1.], dtype=torch.float64)).unsqueeze(0)
    k, device = 1, torch.device("cpu")
    if bad == "object": matrix = None
    elif bad == "sparse": matrix = matrix.to_sparse()
    elif bad == "complex": matrix = matrix.to(torch.complex128)
    elif bad == "rank": matrix = matrix[0]
    elif bad == "nonsquare": matrix = matrix[:, :2, :]
    elif bad == "batch": matrix = torch.zeros(0, 3, 3, dtype=torch.float64)
    elif bad == "nodes": matrix = torch.zeros(1, 0, 0, dtype=torch.float64)
    elif bad == "dtype": matrix = matrix.float()
    elif bad == "device": matrix = matrix.to("meta")
    elif bad in ("nan", "inf"): matrix[0, 0, 0] = float(bad)
    elif bad == "asymmetric": matrix[0, 0, 1] = .1
    elif bad == "low_spectrum": matrix[0, 0, 0] = -1.
    elif bad == "hidden_high_spectrum": matrix[0, 2, 2] = 3.
    elif bad == "k_zero": k = 0
    elif bad == "k_big": k = 4
    elif bad == "k_bool": k = True
    elif bad == "k_float": k = 1.
    elif bad == "target_string": device = "cpu"
    elif bad == "target_meta": device = torch.device("meta")
    elif bad == "target_cuda_index": device = torch.device("cuda", torch.cuda.device_count())
    else:
        device = torch.device("cuda")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(TypeError if bad in ("object", "target_string") else ValueError):
        graph.lowest_eigenpairs(matrix, k=k, model_device=device)


def test_lowest_eigenpairs_preserves_solver_failure_reason(monkeypatch):
    """求解器失败保持原始原因，不重试加扰动或返回假特征对。"""
    graph = api().GraphSpectrum(config())
    failure = RuntimeError("fictional solver failure for C04 boundary test")
    calls = []

    def fail(*args, **kwargs):
        """只在本测试模拟求解失败，并核对调用次数。"""
        calls.append(1)
        raise failure

    monkeypatch.setattr(torch.linalg, "eigh", fail)
    with pytest.raises(RuntimeError) as caught:
        graph.lowest_eigenpairs(torch.zeros(1, 1, 1, dtype=torch.float64), k=1, model_device=torch.device("cpu"))
    assert caught.value is failure or caught.value.__cause__ is failure
    assert len(calls) == 1


def modal_fixture(device):
    """虚构四节点正交基；两批用不同列顺序、场值和64个通道尺度。"""
    u = torch.tensor([[1.,1.,1.,1.],[1.,1.,-1.,-1.],[1.,-1.,1.,-1.],[1.,-1.,-1.,1.]], device=device)/2
    basis = torch.stack((u, u[:, [0,2,1,3]]))
    scales = torch.stack(((torch.arange(64,device=device)%7+1)/8, (torch.arange(64,device=device)%5+3)/8))
    x = torch.tensor([[1.,2.,3.,4.],[4.,3.,2.,1.]],device=device)[:,:,None]*scales[:,None,:]
    c = torch.tensor([[5.,-2.,-1.,0.],[5.,1.,2.,0.]],device=device)[:,:,None]*scales[:,None,:]
    return x, basis, c, scales


def modal_numpy(t):
    """测试参考读取实际存储的float32数后独立用NumPy double计算。"""
    return t.detach().cpu().numpy().astype(np.float64)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("k", [1,2,4])
def test_modal_analysis_synthesis_projection_and_x_gradient(device, k):
    """T-S4：独立手算分析/合成、全基与截断，以及三条固定Psi解析反向。"""
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    x, full, hand_c, scales = modal_fixture(device)
    basis = full[:,:,:k]
    x.requires_grad_()
    c = api().modal_analysis(x,basis)
    torch.testing.assert_close(c,hand_c[:,:k],atol=1e-5,rtol=1e-4)
    independent_c = hand_c[:,:k].clone().requires_grad_()
    y = api().modal_synthesis(independent_c,basis)
    spatial = {1:[[2.5]*4,[2.5]*4],2:[[1.5,1.5,3.5,3.5],[3.,2.,3.,2.]],4:[[1.,2.,3.,4.],[4.,3.,2.,1.]]}
    expected = torch.tensor(spatial[k],device=device)[:,:,None]*scales[:,None,:]
    combined = api().modal_synthesis(c,basis)
    torch.testing.assert_close(y,expected,atol=1e-5,rtol=1e-4)
    torch.testing.assert_close(combined,expected,atol=1e-5,rtol=1e-4)
    residual = x-combined
    np.testing.assert_allclose(np.einsum('bnk,bnc->bkc',modal_numpy(basis),modal_numpy(residual)),0.,atol=1e-5,rtol=1e-4)
    if k==4: torch.testing.assert_close(combined,x,atol=1e-5,rtol=1e-4)
    else: assert residual.square().sum().item()>0
    # 线性调试标量核验反向，不是论文训练目标；固定同一次Psi计算解析参考。
    g = ((torch.arange(2*k*64,device=device).reshape(2,k,64)%11)-5).float()/8
    h = ((torch.arange(512,device=device).reshape(2,4,64)%13)-6).float()/8
    p = modal_numpy(basis)
    references = [np.einsum('bnk,bkc->bnc',p,modal_numpy(g)),np.einsum('bnk,bnc->bkc',p,modal_numpy(h)),
                  np.einsum('bnk,bmk,bmc->bnc',p,p,modal_numpy(h))]
    (c*g).sum().backward(retain_graph=True)
    torch.testing.assert_close(x.grad.cpu().double(),torch.from_numpy(references[0]),atol=1e-5,rtol=1e-4)
    x.grad=None
    (y*h).sum().backward()
    torch.testing.assert_close(independent_c.grad.cpu().double(),torch.from_numpy(references[1]),atol=1e-5,rtol=1e-4)
    (combined*h).sum().backward()
    torch.testing.assert_close(x.grad.cpu().double(),torch.from_numpy(references[2]),atol=1e-5,rtol=1e-4)
    assert basis.grad is None and not basis.requires_grad


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_modal_noncontiguous_single_token_subspace_and_inference(device):
    """合法非连续输入与无梯度推理；已在保留子空间的场可被截断基底无损重建。"""
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    x,basis,_,_ = modal_fixture(device)
    def strided_copy(t):
        """只在测试中以隔列存储制造非连续视图，数值不变。"""
        storage=torch.empty(*t.shape[:-1],t.shape[-1]*2,device=device)
        storage[...,::2]=t
        return storage[...,::2]
    x,basis=strided_copy(x),strided_copy(basis)
    before_x,before_basis=x.clone(),basis.clone()
    assert not x.is_contiguous() and not basis.is_contiguous()
    c=strided_copy(api().modal_analysis(x,basis))
    assert not c.is_contiguous()
    torch.testing.assert_close(api().modal_synthesis(c,basis),x,atol=1e-5,rtol=1e-4)
    assert torch.equal(x,before_x) and torch.equal(basis,before_basis)
    field=basis[:,:,:2]@torch.ones(2,2,64,device=device)
    torch.testing.assert_close(api().modal_synthesis(api().modal_analysis(field,basis[:,:,:2]),basis[:,:,:2]),field,atol=1e-5,rtol=1e-4)
    single=torch.arange(128,device=device,dtype=torch.float32).reshape(2,1,64)/8
    unit=torch.ones(2,1,1,device=device)
    assert torch.equal(api().modal_analysis(single,unit),single)
    assert torch.equal(api().modal_synthesis(single,unit),single)
    with torch.no_grad():
        fixed=api().modal_analysis(x.requires_grad_(),basis)
        output=api().modal_synthesis(fixed,basis)
        assert not fixed.requires_grad and not output.requires_grad
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_modal_real_projection_graph_chain_parameter_gradients_and_state(device):
    """真实C01→C04基底接C05；独立核对X、P_in和raw Z的解析梯度及状态。"""
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    graph=api().GraphSpectrum(config()).to(device).eval()
    model=POJEPA(ProjectionConfig(3,64),graph,object(),object()).to(device).eval()
    z=(torch.arange(36,device=device).reshape(2,6,3).float()/16-1).requires_grad_()
    x=model.project_in(z);x.retain_grad()
    w=graph.affinity(x,regular_grid(2,3))
    spectrum=graph.lowest_eigenpairs(graph.normalized_laplacian(w).laplacian,k=3,model_device=torch.device(device))
    before={n:(id(t),t.detach().clone()) for n,t in model.state_dict(keep_vars=True).items()}
    modes={n:m.training for n,m in model.named_modules()}
    rng,py_rng,np_rng=torch.get_rng_state().clone(),random.getstate(),np.random.get_state()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    defaults=torch.get_default_dtype(),torch.get_default_device()
    values_before=[t.detach().clone() for t in (z,x,spectrum.basis)]
    c=api().modal_analysis(x,spectrum.basis)
    rebuilt=api().modal_synthesis(c,spectrum.basis)
    g=((torch.arange(384,device=device).reshape(2,3,64)%17)-8).float()/16
    (c*g).sum().backward()  # 仅解析反向诊断，无优化器或训练。
    dx=np.einsum('bnk,bkc->bnc',modal_numpy(spectrum.basis),modal_numpy(g))
    expected=[(x.grad,dx),(model.p_in.weight.grad,np.einsum('bnc,bne->ce',dx,modal_numpy(z))),
              (model.p_in.bias.grad,dx.sum((0,1))),(z.grad,np.einsum('bnc,ce->bne',dx,modal_numpy(model.p_in.weight)))]
    for actual,target in expected:
        torch.testing.assert_close(actual.cpu().double(),torch.from_numpy(target),atol=1e-5,rtol=1e-4)
    assert rebuilt.requires_grad and not spectrum.basis.requires_grad and spectrum.basis.grad is None
    for original,current in zip(values_before,(z,x,spectrum.basis)): assert torch.equal(original,current)
    assert {n:m.training for n,m in model.named_modules()}==modes
    for n,t in model.state_dict(keep_vars=True).items(): assert id(t)==before[n][0] and torch.equal(t,before[n][1])
    assert torch.equal(rng,torch.get_rng_state()) and py_rng==random.getstate()
    now=np.random.get_state()
    assert now[0]==np_rng[0] and np.array_equal(now[1],np_rng[1]) and now[2:]==np_rng[2:]
    if cuda_rng: assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    assert defaults==(torch.get_default_dtype(),torch.get_default_device())
    assert set(dict(model.named_parameters()))=={'p_in.weight','p_in.bias','p_out.weight','p_out.bias'}
    assert sum(t.numel() for t in model.parameters())==451


def test_modal_zero_evolution_readout_combination_reports_two_bias_sources():
    """零演化数学组合分开显示非互逆读出及谱截断偏差，不声称完整Phi(0)。"""
    model=POJEPA(ProjectionConfig(3,64),api().GraphSpectrum(config()),object(),object())
    with torch.no_grad():
        model.p_in.weight.zero_();model.p_in.bias.zero_();model.p_out.weight.zero_();model.p_out.bias.zero_()
        model.p_in.weight[:3,:3].copy_(torch.eye(3))
        model.p_out.weight[:3,:3].copy_(torch.diag(torch.tensor([2.,1.,1.])))
    z=torch.arange(1,5,dtype=torch.float32).reshape(1,4,1)*torch.tensor([1.,2.,-1.])
    _,full,_,_=modal_fixture('cpu');basis=full[:1];x=model.project_in(z)
    y=model.project_out(api().modal_synthesis(api().modal_analysis(x,basis),basis))
    torch.testing.assert_close(y,z*torch.tensor([2.,1.,1.]),atol=1e-5,rtol=1e-4)
    assert (y-z).square().sum().item()==30.
    with torch.no_grad(): model.p_out.weight[0,0]=1.
    cut=basis[:,:,:2]
    y=model.project_out(api().modal_synthesis(api().modal_analysis(x,cut),cut))
    expected=torch.tensor([1.5,1.5,3.5,3.5]).reshape(1,4,1)*torch.tensor([1.,2.,-1.])
    torch.testing.assert_close(y,expected,atol=1e-5,rtol=1e-4)
    assert (y-z).square().sum().item()==6.
    with pytest.raises(TypeError): model(z)


@pytest.mark.parametrize('operation',['analysis','synthesis'])
@pytest.mark.parametrize('bad',['data_object','basis_object','data_rank','basis_rank','batch_mismatch','axis_mismatch',
    'empty_batch','empty_nodes','empty_modes','k_above_n','channels','data_double','basis_double','data_int',
    'basis_complex','data_sparse','basis_sparse','data_meta','basis_meta','device_mismatch','data_nan','basis_inf','basis_gradient','overflow'])
def test_modal_rejects_invalid_inputs(operation,bad):
    """两接口都拒绝非法合同与有限乘积溢出，不广播/转换/剪裁或偷偷detach。"""
    x,basis,c,_=modal_fixture('cpu');values=x if operation=='analysis' else c
    if bad=='data_object': values=[]
    elif bad=='basis_object': basis=None
    elif bad=='data_rank': values=values[0]
    elif bad=='basis_rank': basis=basis[0]
    elif bad=='batch_mismatch': basis=basis[:1]
    elif bad=='axis_mismatch': values=values[:,:3]
    elif bad=='empty_batch': values,basis=values[:0],basis[:0]
    elif bad=='empty_nodes': basis=torch.empty(2,0,1);values=torch.empty(2,0 if operation=='analysis' else 1,64)
    elif bad=='empty_modes':
        basis=torch.empty(2,4,0)
        if operation=='synthesis': values=torch.empty(2,0,64)
    elif bad=='k_above_n':
        basis=torch.ones(2,4,5)
        if operation=='synthesis': values=torch.ones(2,5,64)
    elif bad=='channels': values=values[:,:,:63]
    elif bad=='data_double': values=values.double()
    elif bad=='basis_double': basis=basis.double()
    elif bad=='data_int': values=values.long()
    elif bad=='basis_complex': basis=basis.to(torch.complex64)
    elif bad=='data_sparse': values=values.to_sparse()
    elif bad=='basis_sparse': basis=basis.to_sparse()
    elif bad=='data_meta': values=values.to('meta')
    elif bad=='basis_meta': basis=basis.to('meta')
    elif bad=='device_mismatch':
        if not torch.cuda.is_available(): pytest.skip('CUDA unavailable for physical mismatch')
        basis=basis.cuda()
    elif bad=='data_nan': values[0,0,0]=math.nan
    elif bad=='basis_inf': basis[0,0,0]=math.inf
    elif bad=='basis_gradient': basis.requires_grad_()
    elif bad=='overflow': values.fill_(torch.finfo(torch.float32).max)
    fn=api().modal_analysis if operation=='analysis' else api().modal_synthesis
    with pytest.raises(TypeError if bad.endswith('object') else ValueError): fn(values,basis)
