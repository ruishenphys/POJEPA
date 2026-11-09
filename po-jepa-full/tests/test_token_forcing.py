"""forcing_space="token"（导师 2026-11-08）：图谱特征向量的符号/重根旋转不改变预测。

modal 原式在 Psi->Psi Q 下预测会变（Ba 与模态注意力不随 Q 协变）；token 式 Ψ^T[B(X,a)+F(ΨC,X,a)] 严格不变。
"""
import math

import pytest
import torch

from po_jepa.actions import ControlSpec, from_physical_segments
from po_jepa.dynamics import DynamicsConfig, IntegratorConfig, ModalDynamics

SPEC = ControlSpec(2, .25, ("u_x", "u_y"), ("p/s", "p/s"), (-.5, -.5), (.5, .5), (1., 1.), (0., 0.), "surrogate")
INTEGRATOR = IntegratorConfig("exponential_euler", .25, 1e-12, 1e-4, 6)


def make(space, seed=0, head_std=.05):
    torch.manual_seed(seed)
    # 头部初始化放大到 0.05，让 B/N 在测试里有明显贡献（正式配置为 1e-3）
    return ModalDynamics(DynamicsConfig(2, 64, 64, 32, 16, 32, 32, 64, .1, head_std, forcing_space=space))


def inputs(batch=2, side=6, modes=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    n = side*side
    ys, xs = torch.meshgrid(torch.linspace(0, 1, side, dtype=torch.float64), torch.linspace(0, 1, side, dtype=torch.float64), indexing="ij")
    coords = torch.stack((xs.flatten(), ys.flatten()), -1)
    basis = torch.linalg.qr(torch.randn(batch, n, modes, generator=g, dtype=torch.float64))[0].float()
    eigenvalues = torch.sort(torch.rand(batch, modes, generator=g), dim=-1).values
    eigenvalues[:, 3] = eigenvalues[:, 2]       # 一对重根：允许在该二维子空间内任意旋转
    x = torch.randn(batch, n, 64, generator=g)
    ctl = from_physical_segments(
        starts_seconds=torch.tensor([[0., .5]]*batch, dtype=torch.float64), ends_seconds=torch.tensor([[.5, 1.5]]*batch, dtype=torch.float64),
        u_values=(torch.rand(batch, 2, 2, generator=g)-.5), n_segments=torch.full((batch,), 2, dtype=torch.int64),
        segment_valid=torch.ones(batch, 2, dtype=torch.bool), origin_seconds=torch.zeros(batch, dtype=torch.float64),
        control_spec=SPEC, source_ids=tuple(f"t{i}" for i in range(batch)))
    return x, eigenvalues, basis, coords, ctl


def gauge(modes, seed=2):
    """随机符号翻转 × 重根子空间 (2,3) 内的随机旋转。"""
    g = torch.Generator().manual_seed(seed)
    q = torch.diag(torch.where(torch.rand(modes, generator=g) < .5, -1., 1.))
    angle = float(torch.rand((), generator=g))*2*math.pi
    rot = torch.eye(modes)
    rot[2:4, 2:4] = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return q@rot


def predict(dyn, x, eigenvalues, basis, coords, ctl):
    context = dyn.prepare_context(x, eigenvalues, basis, coords)
    c0 = basis.transpose(1, 2)@x
    duration = torch.full((x.shape[0],), 6., dtype=torch.float64)
    final = dyn.integrate_segments(c0, ctl, context, duration=duration, config=INTEGRATOR).final
    return basis@final


def relative_gauge_change(space):
    dyn = make(space)
    x, lam, basis, coords, ctl = inputs()
    q = gauge(basis.shape[-1])
    with torch.no_grad():
        a = predict(dyn, x, lam, basis, coords, ctl)
        b = predict(dyn, x, lam, basis@q, coords, ctl)
    return float((a-b).norm()/a.norm())


def test_token_prediction_is_gauge_invariant():
    assert relative_gauge_change("token") < 1e-5


def test_modal_prediction_depends_on_gauge():
    # 原式的预测随特征向量的符号/旋转选择而变（导师指出的问题），作为对照
    assert relative_gauge_change("modal") > 1e-3


def test_token_rhs_is_equivariant():
    dyn = make("token")
    x, lam, basis, coords, _ = inputs()
    q = gauge(basis.shape[-1])
    c = basis.transpose(1, 2)@x
    a = torch.tensor([[.3, -.2], [-.1, .4]])
    with torch.no_grad():
        r1 = dyn.rhs(c, a, dyn.prepare_context(x, lam, basis, coords)).derivative
        r2 = dyn.rhs(q.T@c, a, dyn.prepare_context(x, lam, basis@q, coords)).derivative
    assert torch.allclose(q.T@r1, r2, atol=1e-5)


def test_token_mode_parameters_and_gradients():
    dyn = make("token")
    names = {n.split(".")[0] for n, _ in dyn.named_parameters()}
    assert names == {"condition_trunk", "gamma_head", "omega_head", "token_b_net", "token_q", "token_k", "token_v",
                     "token_bias_net", "token_residual_net"}
    x, lam, basis, coords, ctl = inputs()
    x.requires_grad_()
    predict(dyn, x, lam, basis, coords, ctl).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in dyn.parameters())


def test_modal_default_unchanged():
    """默认 forcing_space 仍是 v7 原式：同种子下参数逐位相同。"""
    torch.manual_seed(0)
    a = ModalDynamics(DynamicsConfig(2, 64, 64, 32, 16, 32, 32, 64, .1, 1e-3))
    b = make("modal", head_std=1e-3)
    assert [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
