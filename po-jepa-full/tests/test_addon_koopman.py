"""S10a：Koopman 对照（08 §5）——参数量和 K 规则、初始化、T=0 恒等、分段积分的手算值、φ1 两个分支、网格和报错。"""

import cmath
import copy
import hashlib
import math

import pytest
import torch

from po_jepa import koopman
from po_jepa.actions import concatenate_action_segments, split_action_segments
from po_jepa.fourier import dct_basis, mode_order
from po_jepa.geometry import GridSpec, apply_spatial_transfer, build_spatial_transfer, regular_grid
from po_jepa.koopman import KoopmanConfig, KoopmanPredictor, build_koopman, choose_modes, parameter_count, phi1
from po_jepa.runtime import checks, comparison
from test_model import d12_controls
from test_v4 import _controls

CPU = torch.device("cpu")
E = 3


@pytest.fixture
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small(modes=76, seed=1703, perturb=True):
    """E=3 的 Koopman；perturb 时把 γ、ω、B、p_out 换成量级明显的随机值（否则初始化下动力学太弱，测不出公式错误）。"""
    model = build_koopman(KoopmanConfig(modes=modes, action_dim=2, encoder_dim=E), seed, CPU)
    if perturb:
        g = torch.Generator().manual_seed(seed+1)
        with torch.no_grad():
            model.raw_gamma.copy_(torch.randn(model.raw_gamma.shape, generator=g)*.5-1)
            model.omega.copy_(torch.randn(model.omega.shape, generator=g)*.7)
            model.action_gain.copy_(torch.randn(model.action_gain.shape, generator=g)*.3)
            model.p_out.weight.copy_(torch.randn(model.p_out.weight.shape, generator=g)*.1)
    return model


def raw16(batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 256, E, generator=g)


def coarse(raw, size):
    transfer = build_spatial_transfer(regular_grid(16, 16), target_height=size, target_width=size)
    return apply_spatial_transfer(raw, transfer), transfer.target


def test_parameter_count_and_mode_rule():
    assert parameter_count(KoopmanConfig(modes=76, action_dim=2)) == 215_744
    assert parameter_count(KoopmanConfig(modes=72, action_dim=7)) == 236_992
    for k in (1, 16, 76, 200):
        for m in (1, 2, 7):
            assert parameter_count(KoopmanConfig(modes=k, action_dim=m)) == 181_696+k*(320+64*m)
    assert parameter_count(KoopmanConfig(modes=2, action_dim=2, encoder_dim=3)) == 2*3*64+64+3+2*(256+64+128)
    po = comparison.parameter_counts(checks.build_full_model(checks.load_protocol(), CPU))["total"]
    assert po == 215_778
    assert choose_modes(po, action_dim=2) == 76 and choose_modes(236_898, action_dim=7) == 72
    # 平局取较小的 K：m=2 时 K=10、11 的参数量是 186,176、186,624，中点 186,400
    assert choose_modes(186_400, action_dim=2) == 10 and choose_modes(186_401, action_dim=2) == 11
    assert choose_modes(1, action_dim=2) == 1
    assert choose_modes(451+448*5, action_dim=2, encoder_dim=3) == 5
    for bad in (0, -5, 1.5, True):
        with pytest.raises(ValueError, match="^po_parameters must be a positive integer$"):
            choose_modes(bad, action_dim=2)


@pytest.mark.parametrize("m, k, total", [(2, 76, 215_744), (7, 72, 236_992)])
def test_built_model_counts_and_layout(m, k, total):
    model = build_koopman(KoopmanConfig(modes=k, action_dim=m), 1703, CPU)
    counts = comparison.parameter_counts(model)
    assert counts["total"] == total and dict(model.named_buffers()) == {}
    assert counts["modules"] == dict(basis=256*k, raw_gamma=32*k, omega=32*k, action_gain=64*m*k, p_in=1408*64+64, p_out=64*1408+1408)
    assert [n for n, _ in model.named_parameters()] == ["basis", "raw_gamma", "omega", "action_gain",
                                                         "p_in.weight", "p_in.bias", "p_out.weight", "p_out.bias"]
    assert model.readout == "residual" and model.encoder_dim == 1408 and checks.readouts_per_phi(model) == 2


def test_initialization():
    before = torch.get_rng_state().clone()
    model = build_koopman(KoopmanConfig(modes=76, action_dim=2), 1703, CPU)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(model.basis, dct_basis(16, 16, mode_order(76)).to(torch.float32))
    basis = model.basis.detach().double()
    assert torch.linalg.matrix_norm(basis.T@basis-torch.eye(76, dtype=torch.float64)).item() < 1e-5
    softplus_inverse = math.log(math.expm1(.1))
    assert softplus_inverse == pytest.approx(-2.252168, abs=1e-6)
    assert torch.equal(model.raw_gamma, torch.full((76, 32), softplus_inverse, dtype=torch.float32))
    assert torch.nn.functional.softplus(model.raw_gamma.double()).sub(.1).abs().max().item() < 1e-8
    assert torch.equal(model.omega, torch.zeros(76, 32))
    assert torch.equal(model.p_in.bias, torch.zeros(64)) and torch.equal(model.p_out.bias, torch.zeros(1408))
    bound = math.sqrt(6/(1408+64))
    assert model.p_in.weight.abs().max().item() <= bound and model.p_out.weight.abs().max().item() <= bound
    assert .0009 < model.action_gain.std().item() < .0011 and model.action_gain.mean().abs().item() < 1e-4
    # 按 08 §5.2 表的顺序建（p_in、p_out、action_gain）：种子 1703 下三个随机张量的字节固定（torch 2.11 CPU）
    digests = {k: hashlib.sha256(model.state_dict()[k].contiguous().numpy().tobytes()).hexdigest()
               for k in ("p_in.weight", "p_out.weight", "action_gain")}
    assert digests == {"p_in.weight": "803c58285587507e831337f1406738159fe47c7ae96b1224140e44d333b96db4",
                       "p_out.weight": "7605ce43f405124a259072c91108404954dad998ad4cf9f7cfd4e38393f7996a",
                       "action_gain": "240ba9c7a122cca6d44efb576839e70f5e7c6b8ed765d10c2acfc6579bc03bcf"}
    again = build_koopman(KoopmanConfig(modes=76, action_dim=2), 1703, CPU)
    assert all(torch.equal(v, again.state_dict()[k]) for k, v in model.state_dict().items())
    other = build_koopman(KoopmanConfig(modes=76, action_dim=2), 1704, CPU).state_dict()
    assert {k for k, v in model.state_dict().items() if not torch.equal(v, other[k])} == {"p_in.weight", "p_out.weight", "action_gain"}


def test_zero_duration_is_bitwise_identity():
    model = small()
    raw = raw16()
    controls = _controls([4., 2.], [[.3, -.2], [-.1, .4]])
    zero = torch.zeros(2, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(model(raw, controls, zero, regular_grid(16, 16)), raw)
        x = model.p_in(raw)
        c0 = model.analysis(x, model.basis)
        assert torch.equal(model.integrate(c0, controls, zero), c0)
        z8, g8 = coarse(raw, 8)
        assert torch.equal(model(z8, controls, zero, g8), z8)
        mixed = model(raw, controls, torch.tensor([0., 2.], dtype=torch.float64), regular_grid(16, 16))
        assert torch.equal(mixed[0], raw[0]) and not torch.equal(mixed[1], raw[1])


def test_split_constant_segment_matches_whole():
    model = small()
    raw = raw16()
    controls = _controls([4., 2.], [[.3, -.2], [-.1, .4]])
    duration = controls.duration.clone()
    pair = split_action_segments(controls, tau=duration/2, sigma=duration/2)
    halves = concatenate_action_segments(pair.first, pair.second)
    assert halves.n_segments.tolist() == [2, 2]
    with torch.no_grad():
        c0 = model.analysis(model.p_in(raw), model.basis)
        whole, split = model.integrate(c0, controls, duration), model.integrate(c0, halves, duration)
        assert ((split-whole).norm()/whole.norm()).item() < 1e-6
        a = model(raw, controls, duration, regular_grid(16, 16))
        b = model(raw, halves, duration, regular_grid(16, 16))
        assert ((a-b).norm()/(a-raw).norm()).item() < 1e-6


def hand_model():
    """K=2、E=3 的手算模型：只有两对通道有动力学和动作项。"""
    model = build_koopman(KoopmanConfig(modes=2, action_dim=2, encoder_dim=E), 1703, CPU)
    with torch.no_grad():
        model.raw_gamma.zero_(); model.raw_gamma[0, 0] = math.log(math.expm1(.3))
        model.omega.zero_(); model.omega[0, 0] = 1.25; model.omega[1, 5] = -.5
        model.action_gain.zero_()
        model.action_gain[0, 0, 0] = .5; model.action_gain[0, 1, 1] = -.25; model.action_gain[1, 11, 0] = 2.
    return model


def closed_form(z, rate, segments):
    """z ← e^{λΔ}z + (e^{λΔ} − 1)/λ · φ，逐段（complex128，和被测代码无关）。"""
    for delta, phi in segments:
        z = cmath.exp(rate*delta)*z+(cmath.exp(rate*delta)-1)/rate*phi
    return z


def test_integration_matches_hand_computed_values():
    """第 0 个模态第 0 对：λ = −0.3 + 1.25i，φ = 0.5·u_x + i·(−0.25·u_y)；第 1 个模态第 5 对（通道 10、11）：
    λ = −log 2 − 0.5i，φ = i·2·u_x。第 1 行 3 段、第 0 行 2 段，padding 是 NaN，不能被读到。"""
    model = hand_model()
    initial = torch.zeros(2, 2, 64)
    for row in (0, 1):
        initial[row, 0, 0], initial[row, 0, 1], initial[row, 1, 10], initial[row, 1, 11] = 1.-row, -2., .75, .5+row
    controls = d12_controls([[(0., .5, (.8, .4)), (.5, 1.25, (-.6, .2))],
                             [(0., .25, (.1, -.9)), (.25, 1., (.7, .3)), (1., 2., (-.4, -.5))]])
    duration = torch.tensor([1.25, 1.5], dtype=torch.float64)
    with torch.no_grad():
        final = model.integrate(initial, controls, duration)
    rows = {0: [(.5, (.8, .4)), (.75, (-.6, .2))], 1: [(.25, (.1, -.9)), (.75, (.7, .3)), (.5, (-.4, -.5))]}
    for row, segments in rows.items():
        z0 = closed_form(complex(1.-row, -2.), complex(-.3, 1.25), [(d, complex(.5*u[0], -.25*u[1])) for d, u in segments])
        z1 = closed_form(complex(.75, .5+row), complex(-math.log(2), -.5), [(d, complex(0., 2*u[0])) for d, u in segments])
        assert final[row, 0, 0].item() == pytest.approx(z0.real, abs=1e-6) and final[row, 0, 1].item() == pytest.approx(z0.imag, abs=1e-6)
        assert final[row, 1, 10].item() == pytest.approx(z1.real, abs=1e-6) and final[row, 1, 11].item() == pytest.approx(z1.imag, abs=1e-6)
        others = final[row].clone(); others[0, :2] = 0; others[1, 10:12] = 0
        assert torch.equal(others, torch.zeros(2, 64))
    assert torch.isfinite(final).all()


def test_phi1_branches_agree_across_threshold():
    for factor in (1-1e-9, 1+1e-9):
        for angle in (0., .7, math.pi/2, 2., math.pi):
            w = torch.tensor([1e-4*factor*cmath.exp(1j*angle)], dtype=torch.complex128)
            series, ratio = 1+w/2+w*w/6, torch.expm1(w)/w
            assert torch.equal(phi1(w, 1e-4), series if factor < 1 else ratio)
            assert ((series-ratio).abs()/ratio.abs()).item() < 1e-10
    w = torch.zeros(1, dtype=torch.complex128, requires_grad=True)
    value = phi1(w, 1e-4)
    assert value.item() == 1
    value.real.sum().backward()
    assert torch.isfinite(torch.view_as_real(w.grad)).all()
    big = torch.tensor([3.-2j], dtype=torch.complex128)
    assert phi1(big, 1e-4).item() == pytest.approx((cmath.exp(3.-2j)-1)/(3.-2j), rel=1e-14)


def test_analysis_and_grid_basis(one_thread):
    model = small()
    raw = raw16()
    x = model.p_in(raw)
    assert model.grid_basis(regular_grid(16, 16)) is model.basis
    assert torch.equal(model.analysis(x, model.basis), torch.matmul(model.basis.transpose(0, 1), x))
    z8, g8 = coarse(raw, 8)
    transfer = build_spatial_transfer(regular_grid(16, 16), target_height=8, target_width=8)
    w8 = model.grid_basis(g8)
    assert torch.equal(w8, apply_spatial_transfer(model.basis[None], transfer)[0]) and w8.shape == (64, 76)
    x8 = model.p_in(z8)
    assert torch.equal(model.analysis(x8, w8), torch.matmul(w8.transpose(0, 1), x8)*4.)


def test_grid_coordinates_must_match():
    model = small()
    raw = raw16()
    controls = _controls([2., 2.], [[.3, -.2], [-.1, .4]])
    duration = controls.duration.clone()
    z12, g12 = coarse(raw, 12)
    assert torch.isfinite(model(z12, controls, duration, g12)).all()
    shifted = GridSpec(16, 16, regular_grid(16, 16).coords.flip(0), "fictional reversed order")
    message = "^grid coordinates must equal the 16x16 basis grid or its spatial transfer$"
    for z, grid in ((z12, regular_grid(12, 12)), (raw, shifted), (torch.randn(2, 400, E), regular_grid(20, 20))):
        with pytest.raises(ValueError, match=message):
            model(z, controls, duration, grid)


def test_coarse_grid_with_more_modes_than_tokens_and_all_gradients():
    model = small(perturb=False)
    raw = raw16()
    controls = _controls([4., 2.], [[.3, -.2], [-.1, .4]])
    duration = controls.duration.clone()
    z8, g8 = coarse(raw, 8)
    assert model.config.modes == 76 > z8.shape[1] == 64
    out8 = model(z8, controls, duration, g8)
    out16 = model(raw, controls, duration, regular_grid(16, 16))
    assert out8.shape == (2, 64, E) and torch.isfinite(out8).all()
    (out8.square().mean()+out16.square().mean()).backward()
    for name, p in model.named_parameters():
        if name == "p_out.bias":   # 残差读出里两次 p_out 的偏置相消（和 POJEPA 的残差读出相同），梯度恰好是 0
            assert torch.equal(p.grad, torch.zeros_like(p))
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.square().sum().item() > 0, name


def test_forward_does_not_consume_rng():
    model = small()
    raw = raw16()
    controls = _controls([4., 2.], [[.3, -.2], [-.1, .4]])
    before = torch.get_rng_state().clone()
    model(raw, controls, controls.duration.clone(), regular_grid(16, 16))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("change, message", [
    (dict(modes=0), "^modes must be a positive integer$"),
    (dict(modes=True), "^modes must be a positive integer$"),
    (dict(action_dim=0), "^action_dim must be a positive integer$"),
    (dict(encoder_dim=-1), "^encoder_dim must be a positive integer$"),
    (dict(dynamic_dim=32), "^dynamic_dim must be 64 \\(32 complex channel pairs\\)$"),
    (dict(basis_grid=[16, 16]), "^basis_grid must be the tuple \\(16, 16\\)$"),
    (dict(basis_grid=(8, 8)), "^basis_grid must be the tuple \\(16, 16\\)$"),
    (dict(basis_init="random"), "^Koopman uses the dct2 basis initialization and the residual readout$"),
    (dict(readout="direct"), "^Koopman uses the dct2 basis initialization and the residual readout$"),
    (dict(gamma_init=0.), "^gamma_init must be a finite positive float$"),
    (dict(gamma_init=1), "^gamma_init must be a finite positive float$"),
    (dict(action_gain_std=float("inf")), "^action_gain_std must be a finite positive float$"),
    (dict(phi1_series_threshold=1e-5), "^phi1_series_threshold must be 1e-4$"),
])
def test_config_errors(change, message):
    with pytest.raises(ValueError, match=message):
        KoopmanConfig(**(dict(modes=76, action_dim=2) | change))


def test_type_errors():
    with pytest.raises(TypeError, match="^config must be a KoopmanConfig$"):
        KoopmanPredictor(dict(modes=76, action_dim=2))
    with pytest.raises(TypeError, match="^config must be a KoopmanConfig$"):
        parameter_count(dict(modes=76, action_dim=2))
    with pytest.raises(TypeError, match="^seed must be an int$"):
        build_koopman(KoopmanConfig(modes=76, action_dim=2), 1703., CPU)
    with pytest.raises(ValueError, match="^grid does not resolve the requested Fourier modes$"):
        KoopmanPredictor(KoopmanConfig(modes=300, action_dim=2, encoder_dim=E))


def test_input_errors(monkeypatch):
    model = small(modes=4)
    raw = raw16()
    controls = _controls([2., 2.], [[.3, -.2], [-.1, .4]])
    duration = controls.duration.clone()
    grid = regular_grid(16, 16)
    from po_jepa.actions import ControlSpec
    three = copy.copy(controls)
    object.__setattr__(three, "u_values", torch.zeros(2, 1, 3))
    object.__setattr__(three, "control_spec", ControlSpec(3, .25, ("a", "b", "c"), ("u", "u", "u"), (-1.,)*3, (1.,)*3, (1.,)*3, (0.,)*3, "test"))
    cases = [
        (ValueError, "^raw must be finite model-device float32 \\[B,N,encoder_dim\\]$", (raw[..., :2], controls, duration, grid)),
        (ValueError, "^raw must be finite model-device float32 \\[B,N,encoder_dim\\]$", (raw.double(), controls, duration, grid)),
        (TypeError, "^grid must be GridSpec$", (raw, controls, duration, "16x16")),
        (ValueError, "^raw N must match actual grid$", (raw[:, :64], controls, duration, grid)),
        (ValueError, "^control B/device must match raw$", (raw[:1], controls, duration, grid)),
        (ValueError, "^control channels must equal the Koopman action_dim$", (raw, three, duration, grid)),
    ]
    for error, message, args in cases:
        with pytest.raises(error, match=message):
            model(*args)
    real_clip = koopman.clip_action_segments
    def gap(controls, *, start, end):
        clipped = real_clip(controls, start=start, end=end)
        object.__setattr__(clipped, "ends", clipped.ends*.5)
        return clipped
    monkeypatch.setattr(koopman, "clip_action_segments", gap)
    with pytest.raises(ValueError, match="^effective segment lengths must sum to the query duration$"):
        model(raw, controls, duration, grid)
    for excess, fails in ((1.5e-12, True), (5e-13, False)):   # T = 2：绝对容差 1e-12（不是相对 T）
        def shifted(controls, *, start, end, excess=excess):
            clipped = real_clip(controls, start=start, end=end)
            object.__setattr__(clipped, "ends", torch.where(clipped.segment_valid & (clipped.ends == end[:, None]),
                                                            clipped.ends+excess, clipped.ends))
            return clipped
        monkeypatch.setattr(koopman, "clip_action_segments", shifted)
        if fails:
            with pytest.raises(ValueError, match="^effective segment lengths must sum to the query duration$"):
                model(raw, controls, duration, grid)
        else:
            assert torch.isfinite(model(raw, controls, duration, grid)).all()
    monkeypatch.undo()
    with torch.no_grad():
        model.p_in.weight.fill_(3e38)   # c_0 溢出成 ±inf，读出差里出现 inf-inf
    with pytest.raises(ValueError, match="^nonfinite Koopman raw prediction$"):
        model(raw, controls, duration, grid)
