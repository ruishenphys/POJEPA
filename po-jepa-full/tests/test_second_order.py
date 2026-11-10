"""S23a 二阶场景（second_order.py）：随机流、34 帧约定与续算、物理解析解和守恒量、数值收敛、画面包含全部位置。"""
import math

import numpy as np
import pytest

from po_jepa import datasets, second_order as so
from po_jepa.systems import COLORS, _texture
from po_jepa.variants import Variant

PE, CP, SC, BB, WV = (so.SYSTEMS[n] for n in ("pendulum", "cart_pole", "spring_chain", "bouncing_balls", "wave"))
NAMES = tuple(so.SYSTEMS)


def controls(seed, gain=1.):
    """v6 规则（1/4/8 段、20% 零动作、[-0.5,0.5]²）的执行动作 [32,2]，只供测试。"""
    g = np.random.default_rng(seed); u = np.zeros((32, 2)); t = 0
    while t < 32:
        n = int(g.choice([1, 4, 8])); u[t:t+n] = 0. if g.random() < .2 else g.uniform(-.5, .5, 2); t += n
    return gain*u


@pytest.fixture(scope="module")
def runs():
    """每个场景一条默认参数的轨迹（状态 [34,D]、动作）。"""
    return {name: (so.SYSTEMS[name].trajectory(3, controls(3), {}), controls(3)) for name in NAMES}


def centroid(frame, color, tol=40., near=None, radius=20.):
    """颜色接近 color 的像素的质心（行、列）；near 给出时只看它附近 radius 像素内（别的光点边缘的混色不算进来）。"""
    mask = np.abs(frame.astype(float)-np.asarray(color)).max(-1) < tol
    if near is not None:
        row, col = np.meshgrid(np.arange(256)+.5, np.arange(256)+.5, indexing="ij")
        mask &= np.hypot(row-near[0], col-near[1]) < radius
    assert mask.sum() > 3
    r, c = np.nonzero(mask)
    return np.array([r.mean()+.5, c.mean()+.5])


# —————————————————————— 共同约定 ——————————————————————

def test_random_streams_follow_the_scheme_after_registration():
    for order, name in enumerate(NAMES):
        expected = datasets.SYSTEMS.index(name) if name in datasets.SYSTEMS else 7+order
        assert so.SYSTEM_INDEX[name] == expected
    assert len(datasets.SYSTEMS) == 7 or datasets.SYSTEMS[7:12] == NAMES     # 登记前后都成立：追加在 push 之后
    ref = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1703, 1007, 0, 0]))).random(4)
    assert np.array_equal(so.rng("pendulum", 0, 0).random(4), ref)
    firsts = {name: so.rng(name, 0, 0).random() for name in NAMES}
    assert len(set(firsts.values())) == len(NAMES)


def test_params_defaults_ranges_and_types():
    assert so.param("pendulum", {}, "g_over_l") == 9.81 and so.param("bouncing_balls", {}, "n") == 3
    for system, name, bad in (("pendulum", "g_over_l", 3.), ("wave", "speed", .31), ("cart_pole", "length", float("nan")),
                              ("spring_chain", "n", 7), ("spring_chain", "n", 4.), ("bouncing_balls", "restitution", 1.01),
                              ("wave", "load", -.1)):
        with pytest.raises(ValueError):
            so.param(system, {name: bad}, name)
    with pytest.raises(ValueError):
        PE.trajectory(0, np.zeros((31, 2)), {})
    with pytest.raises(ValueError):
        PE.trajectory(0, np.full((32, 2), np.nan), {})


@pytest.mark.parametrize("name", NAMES)
def test_shapes_preroll_determinism_and_restart(name, runs):
    """34 帧：下标 0 = t=−1 的初值，下标 1 = 零动作预跑一个 tick；整条确定；从任意 tick 续算与整条轨迹逐位相同。"""
    S = so.SYSTEMS[name]; states, u = runs[name]
    assert states.shape[0] == 34 and states.dtype == np.float64 and np.isfinite(states).all()
    assert np.array_equal(states[0], S.initial(3, {}))
    assert np.array_equal(states[1], S.tick(states[0], np.zeros(2), {}))
    assert np.array_equal(states, S.trajectory(3, u, {}))
    for start in (0, 7, 31):
        assert np.array_equal(S.advance(states[start+1], u, {}, start, 32-start), states[start+1:])
    assert np.abs(states[-1]-states[1]).max() > 1e-3                                    # 在动
    with pytest.raises(ValueError):
        S.advance(states[1], u, {}, 30, 3)


def test_preroll_uses_zero_action_and_actions_start_at_tick_zero():
    """预跑一个 tick 不用动作：改 u[0] 不改下标 0、1，改下标 2；改 u[5] 只从下标 7 起变。"""
    for S in (PE, CP, SC, BB):
        u = controls(5); a = S.trajectory(1, u, {})
        u2 = u.copy(); u2[0] += .3; b = S.trajectory(1, u2, {})
        assert np.array_equal(a[:2], b[:2]) and not np.array_equal(a[2], b[2])
        u3 = u.copy(); u3[5] += .3; c = S.trajectory(1, u3, {})
        assert np.array_equal(a[:7], c[:7]) and not np.array_equal(a[7], c[7])


@pytest.mark.parametrize("name", NAMES)
def test_frames_partial_redraw_and_observation_knobs(name, runs):
    """frames() = (隐藏状态, draw)；只画几帧（含 t=−1）时，带像素噪声、遮挡块、会动的干扰光点也与整条逐位相同；
    观察调节项不改状态；frame_ticks 的检查。"""
    S = so.SYSTEMS[name]; states, u = runs[name]
    spots = name not in ("wave",)
    v = Variant(pixel_noise=8., occluders=2, movers=2 if spots else 0, distractors=1 if spots else 0,
                background="checker", color="white")
    hidden, full = S.frames(3, u, {}, v)
    assert full.shape == (34, 256, 256, 3) and full.dtype == np.uint8
    assert np.array_equal(hidden, S.hidden(states, {}))
    picks = np.array([0, 1, 20, 33])
    assert np.array_equal(S.draw(states[picks], 3, {}, v, ticks=picks-1), full[picks])
    with pytest.raises(ValueError):
        S.draw(states[picks], 3, {}, v)
    with pytest.raises(ValueError):
        S.draw(states[picks], 3, {}, v, ticks=picks)          # 只有 34 帧时 tick 33 越界
    plain = S.draw(states, 3, {}, Variant())
    assert not np.array_equal(plain, full)
    assert np.array_equal(S.frames(3, u, {}, Variant())[1], plain)


def test_pixel_noise_frame_f_uses_draw_f_of_stream_4():
    """像素噪声：一次抽 [34,256,256,3]（流 4），下标 f 的帧（tick f−1）加第 f 份；t = −1 的帧用第 0 份。"""
    q = PE.initial(6, {})[None].repeat(3, 0)
    ticks = np.array([-1, 0, 32])
    clean = PE.draw(q, 6, {}, Variant(background="plain"), ticks=ticks).astype(float)
    noisy = PE.draw(q, 6, {}, Variant(background="plain", pixel_noise=6.), ticks=ticks).astype(float)
    noise = so.rng("pendulum", 6, 4).normal(0., 6., (34, 256, 256, 3))[ticks+1]
    inside = (clean > 30) & (clean < 220)                       # 不碰 0/255 截断的像素
    assert np.abs(noisy-clean-noise)[inside].max() <= 1.


def test_variant_checks():
    u = controls(0)
    for S, bad in ((BB, Variant(sigma=4.)), (WV, Variant(sigma=4.)), (WV, Variant(distractors=1)), (WV, Variant(movers=1))):
        with pytest.raises(ValueError):
            S.frames(0, u, {}, bad)
    for S in (PE, CP, SC):
        S.check_variant(Variant(sigma=4., distractors=2, movers=1))
    BB.check_variant(Variant(distractors=2, movers=1))


def test_distractors_and_movers_use_streams_3_and_8():
    """静止干扰光点在随机流 3 的 uniform(24, 232) 位置（与 systems.compose 同一规则），会动的干扰光点按流 8 的
    mover_centers 在 tick −1 也有定义；两者都画在场景之下（这里取空白处的像素核对）。"""
    from po_jepa.systems import DISTRACTOR_COLOR, mover_centers
    image = so._layers("pendulum", 4, Variant(distractors=2, background="plain"), 1, np.array([5]), 6.)[0]
    for r, c in so.rng("pendulum", 4, 3).uniform(24., 232., (2, 2)):
        assert np.abs(image[int(r), int(c)]-DISTRACTOR_COLOR).max() < 8
    ticks = np.array([-1, 12])
    image = so._layers("pendulum", 4, Variant(movers=1, background="plain"), 2, ticks, 6.)
    for k, (r, c) in enumerate(mover_centers(so.rng("pendulum", 4, 8), 1, ticks)[:, 0]):
        assert np.abs(image[k, int(r), int(c)]-DISTRACTOR_COLOR).max() < 8


@pytest.mark.parametrize("name", ("pendulum", "cart_pole", "spring_chain", "bouncing_balls"))
def test_single_frame_shows_positions_not_velocities(name, runs):
    """画面只由位置决定（同位置、不同速度 → 同一画面），速度只能从相邻两帧得到；位置变了画面一定变。"""
    S = so.SYSTEMS[name]; states, _ = runs[name]
    q = states[10].copy(); q2 = q.copy()
    vel = {"pendulum": [1], "cart_pole": [1, 3]}.get(name)
    if vel is None:
        n = len(q)//4; vel = list(range(2*n, 4*n))
    q2[vel] += .7
    a, b = S.draw(np.stack((q, q2)), 3, {}, Variant(), ticks=np.array([9, 9]))
    assert np.array_equal(a, b)
    q3 = q.copy(); q3[0] += .05
    c = S.draw(q3[None], 3, {}, Variant(), ticks=np.array([9]))[0]
    assert not np.array_equal(a, c)


# —————————————————————— pendulum ——————————————————————

def test_pendulum_energy_conserved_without_damping_or_force():
    q = np.array([2.5, 1.])
    E = lambda q: .5*q[1]**2-9.81*math.cos(q[0])
    st = PE.advance(q, np.zeros((32, 2)), {"damping": 0.}, 0, 32)
    assert max(abs(E(s)-E(q)) for s in st)/abs(E(q)) < 1e-8


def test_pendulum_linear_damped_step_response():
    """小角度：θ̈ + γθ̇ + ω0²θ = K·u_x（θ≈0 时切向 = x 方向）。用 θ ~ 1e−4 的小动作与解析解比较（γ = 0.2、ω0² = 9.81）。"""
    gamma, w2, a = .2, 9.81, 4.*1e-4
    u = np.zeros((32, 2)); u[:, 0] = 1e-4
    st = PE.advance(np.zeros(2), u, {}, 0, 32)
    t = .25*np.arange(33); wd = math.sqrt(w2-gamma**2/4)
    ref = a/w2*(1-np.exp(-gamma*t/2)*(np.cos(wd*t)+gamma/(2*wd)*np.sin(wd*t)))
    assert np.abs(st[:, 0]-ref).max() < 1e-6*np.abs(ref).max()


def test_pendulum_force_acts_tangentially():
    """θ = π/2（摆锤在右边水平处）：沿摆杆的力 (1,0) 不产生角加速度；向上的力 (0,1) 给 +K。θ = 0 时 (1,0) 给 +K。"""
    assert PE.field(np.array([math.pi/2, 0.]), np.array([1., 0.]), 9.81, .2)[1] == pytest.approx(-9.81)
    assert PE.field(np.array([math.pi/2, 0.]), np.array([0., 1.]), 9.81, .2)[1] == pytest.approx(-9.81+4.)
    assert PE.field(np.array([0., 0.]), np.array([1., 0.]), 9.81, .2)[1] == pytest.approx(4.)


def test_pendulum_substeps_converge_and_stay_below_nyquist():
    u = controls(7)
    a = PE.trajectory(7, u, {}); b = PE.trajectory(7, u, {}, substeps=200)
    assert np.abs(a-b).max() < 1e-7
    worst = 0.
    for params in ({}, {"g_over_l": 16., "damping": 0.}):
        for tid in range(30):
            st = PE.trajectory(tid, controls(100+tid), params)
            worst = max(worst, np.abs(np.diff(st[:, 0])).max())
    assert worst < .8*math.pi       # 随机动作、增益 1 下相邻两帧转角 < π；不是全范围的保证（构造的动作能超过 π，见类说明）


def test_pendulum_bob_drawn_at_angle():
    for theta, rc in ((0., (208., 128.)), (math.pi/2, (128., 208.)), (math.pi, (48., 128.))):
        f = PE.draw(np.array([[theta, 0.]]), 0, {}, Variant(), ticks=np.array([0]))[0]
        assert np.abs(centroid(f, COLORS["orange"])-rc).max() < 1.


# —————————————————————— cart_pole ——————————————————————

@pytest.fixture
def no_pole_damping(monkeypatch):
    monkeypatch.setattr(CP, "POLE_DAMPING", 0.)


def test_cart_pole_energy_conserved_without_friction(no_pole_damping):
    q = np.array([0., .1, 2.2, .5])                             # 摆锤在下半圈摆动，小车不碰缓冲
    st = CP.advance(q, np.zeros((32, 2)), {"friction": 0.}, 0, 32)
    assert np.abs(st[:, 0]).max() < CP.TRACK
    E = [CP.energy(s, {}) for s in st]
    assert np.ptp(E)/abs(E[0]) < 1e-9


def test_cart_pole_horizontal_momentum_conserved():
    """无推力、无小车摩擦、不碰缓冲时，水平动量 (M+m)ẋ + m·l·θ̇·cos θ 守恒（摆的转动阻尼是内力矩，不改变它）。"""
    q = np.array([0., -.2, 1., 2.])
    for params in ({"friction": 0.}, {"friction": 0., "mass_ratio": 1., "length": .6}):
        st = CP.advance(q, np.zeros((32, 2)), params, 0, 32)
        assert np.abs(st[:, 0]).max() < CP.TRACK
        l, m = so.param("cart_pole", params, "length"), so.param("cart_pole", params, "mass_ratio")
        p = (1+m)*st[:, 1]+m*l*st[:, 3]*np.cos(st[:, 2])
        assert np.ptp(p) < 1e-7*np.abs(p).max()                  # 动量不是状态的线性函数，RK4 只近似守恒
        fine = CP.advance(q, np.zeros((32, 2)), params, 0, 32, substeps=400)
        pf = (1+m)*fine[:, 1]+m*l*fine[:, 3]*np.cos(fine[:, 2])
        assert np.ptp(pf) < np.ptp(p)/50                          # 步长减到 1/4，误差按四阶降


def test_cart_pole_small_oscillation_frequency(no_pole_damping):
    """挂在下方（θ = π + ε）的小振动：ε̈ = −(g/l)·(M+m)/M·ε。"""
    for l, ratio in ((1., .3), (.6, 1.)):
        params = {"friction": 0., "length": l, "mass_ratio": ratio}
        eps0 = 1e-4
        st = CP.advance(np.array([0., 0., math.pi+eps0, 0.]), np.zeros((32, 2)), params, 0, 32)
        w = math.sqrt(9.81/l*(1+ratio))
        ref = eps0*np.cos(w*.25*np.arange(33))
        assert np.abs(st[:, 2]-math.pi-ref).max() < 1e-3*eps0


def test_cart_pole_pole_damping_rate():
    """摆的转动阻尼 c = m·l²·0.1：挂着的小振动线性化后 ε̈ = −Ω²ε − 0.1·(M+m)/M·ε̇（小车无摩擦），按阻尼振子衰减。"""
    l, ratio, gp = .8, .5, .1
    params = {"friction": 0., "length": l, "mass_ratio": ratio}
    eps0 = 1e-4
    st = CP.advance(np.array([0., 0., math.pi+eps0, 0.]), np.zeros((32, 2)), params, 0, 32)
    w2 = 9.81/l*(1+ratio); g = gp*(1+ratio); wd = math.sqrt(w2-g*g/4); t = .25*np.arange(33)
    ref = eps0*np.exp(-g*t/2)*(np.cos(wd*t)+g/(2*wd)*np.sin(wd*t))
    assert np.abs(st[:, 2]-math.pi-ref).max() < 1e-3*eps0


def test_cart_pole_push_signs():
    """静止挂着时向右推：小车向右加速 K·u/M，摆锤落在小车后面（θ 往大于 π 走 → 摆锤偏左）。u_y 不起作用。"""
    f = CP.field(np.array([0., 0., math.pi, 0.]), np.array([.5, 0.]), 1., .3, .5)
    assert f[1] == pytest.approx(1.5*.5) and f[3] == pytest.approx(1.5*.5/1.)
    assert np.array_equal(CP.field(np.array([.3, .2, 1., -1.]), np.array([.5, .4]), 1., .3, .5),
                          CP.field(np.array([.3, .2, 1., -1.]), np.array([.5, -.4]), 1., .3, .5))


def test_cart_pole_substeps_converge_and_bumpers_hold():
    u = controls(11, 2.)
    a = CP.trajectory(11, u, {}); b = CP.trajectory(11, u, {}, substeps=400)
    assert np.abs(a-b).max() < 1e-6
    hit = 0
    for tid in range(40):
        st = CP.trajectory(tid, controls(200+tid, 2.), {"length": .6, "mass_ratio": .1, "friction": 0.})
        hit += np.abs(st[:, 0]).max() > CP.TRACK
        assert np.abs(st[:, 0]).max() < CP.TRACK+.2 and np.abs(np.diff(st[:, 2])).max() < .85*math.pi
    assert hit > 0                                             # 缓冲确实被碰到过（上面的上限有意义）


def test_cart_pole_drawn_where_the_state_says():
    q = np.array([[.8, 0., .6, 0.]])
    f = CP.draw(q, 0, {}, Variant(), ticks=np.array([0]))[0]
    bob = np.array([128-36*math.cos(.6), 128+36*(.8+math.sin(.6))])
    assert np.abs(centroid(f, COLORS["orange"])-bob).max() < 1.
    cart = np.abs(f.astype(float)-so.BODY_COLOR).max(-1) < 3
    r, c = np.nonzero(cart)
    assert abs(c.mean()+.5-(128+36*.8)) < 1. and abs(r.mean()+.5-128) < 1.


# —————————————————————— spring_chain ——————————————————————

@pytest.mark.parametrize("params", ({}, {"n": 2, "stiffness": 9., "damping": 0.}, {"n": 6, "stiffness": 25., "damping": .5}))
def test_spring_chain_tick_matrix_has_analytic_modes(params):
    """一个 tick 的传播矩阵 Φ = exp(A·0.25) 的特征值 = exp((−γ/2 ± iω_j,d)·0.25)，ω_j = 2√k·sin(jπ/(2(n+1)))；
    数值用说明里的公式重写一遍（不读类的 modes()），也核 modes()。"""
    n = params.get("n", 4); k = params.get("stiffness", 16.); g = params.get("damping", .1)
    w = 2*math.sqrt(k)*np.sin(np.arange(1, n+1)*math.pi/(2*(n+1)))
    wd = np.sqrt(w**2-g**2/4)
    Phi, _ = SC.matrices(params)
    ev = np.linalg.eigvals(Phi)
    assert np.allclose(np.abs(ev), math.exp(-g/2*.25), atol=1e-12)
    assert np.allclose(np.sort(np.abs(np.angle(ev))), np.sort(np.concatenate((wd, wd))*.25), atol=1e-10)
    mw, mwd, decay = SC.modes(params)
    assert np.allclose(mw, w) and np.allclose(mwd, wd) and decay == g/2


def test_spring_chain_matches_fine_rk4_and_pushes_mass_zero():
    """精确推进 = 小步长 RK4（d̈ = −k·L·d − γḋ + e0·K·u，墙固定）；u_x 不影响 y。"""
    params = {"n": 3}; n = 3; k, g = 16., .1
    q = SC.initial(4, params); u = np.array([.4, 0.])
    lap = 2*np.eye(n)-np.eye(n, k=1)-np.eye(n, k=-1)
    rest = SC.rest(n)
    d = q[:2*n].reshape(n, 2)-np.stack((rest, np.zeros(n)), -1); v = q[2*n:].reshape(n, 2)
    F = np.zeros((n, 2)); F[0] = 1.5*u

    def f(z):
        d, v = z[:n], z[n:]
        return np.concatenate((v, -k*lap@d-g*v+F))
    z = np.concatenate((d, v))
    for _ in range(2500):
        h = .25/2500
        k1 = f(z); k2 = f(z+h*k1/2); k3 = f(z+h*k2/2); k4 = f(z+h*k3); z = z+h*(k1+2*k2+2*k3+k4)/6
    q1 = SC.tick(q, u, params)
    assert np.abs(q1[:2*n].reshape(n, 2)-(z[:n]+np.stack((rest, np.zeros(n)), -1))).max() < 1e-11
    assert np.abs(q1[2*n:].reshape(n, 2)-z[n:]).max() < 1e-11


def test_spring_chain_static_deflection_is_a_fixed_point():
    """常力 F 推第 0 个质点时，静平衡 d = (k·L)⁻¹·e0·F（速度为 0）是 tick 的不动点。"""
    n, k = 4, 16.
    lap = 2*np.eye(n)-np.eye(n, k=1)-np.eye(n, k=-1)
    u = np.array([.3, -.2]); e0 = np.zeros(n); e0[0] = 1.
    d = np.linalg.solve(k*lap, e0)[:, None]*(1.5*u)[None, :]
    q = np.concatenate(((d+np.stack((SC.rest(n), np.zeros(n)), -1)).ravel(), np.zeros(2*n)))
    assert np.abs(SC.tick(q, u, {})-q).max() < 1e-12


def test_spring_chain_rest_positions_are_evenly_spaced_between_walls():
    """静止位置在墙 x = ±2 之间等距：n = 4 → −1.2、−0.4、0.4、1.2；n = 2 → ±2/3。初值位移 ≤ 0.15。"""
    assert np.allclose(SC.rest(4), [-1.2, -.4, .4, 1.2]) and np.allclose(SC.rest(2), [-2/3, 2/3])
    for tid in range(5):
        q = SC.initial(tid, {})
        assert np.abs(q[:8].reshape(4, 2)-np.array([[-1.2, 0.], [-.4, 0.], [.4, 0.], [1.2, 0.]])).max() <= .15


def test_spring_chain_stays_ordered_and_drawn_in_place():
    for params in ({}, {"n": 6, "stiffness": 9., "damping": 0.}):
        n = params.get("n", 4)
        for tid in range(20):
            st = SC.trajectory(tid, controls(300+tid), params)
            x = st[:, :2*n].reshape(34, n, 2)[..., 0]
            assert np.diff(x, axis=1).min() > .05                 # 相邻质点不交叉
    q = SC.initial(0, {})
    f = SC.draw(q[None], 0, {}, Variant(), ticks=np.array([-1]))[0]
    rc = np.array([128-48*q[1], 128+48*q[0]])
    assert np.abs(centroid(f, COLORS["orange"], near=rc)-rc).max() < 1.


# —————————————————————— bouncing_balls ——————————————————————

def one_ball(x, v, params, u=(0., 0.), ticks=1, substeps=None):
    params = dict({"n": 1}, **params)
    q = np.concatenate((np.asarray(x, float), np.asarray(v, float)))
    for _ in range(ticks):
        q = BB.tick(q, np.asarray(u, float), params, substeps=substeps)
    return q


def test_balls_free_flight_matches_closed_form():
    """不碰撞时：v(t) = v∞ + (v0 − v∞)e^{−βt}，v∞ = K·u/β；x(t) 是它的积分；β = 0 时匀加速。"""
    x0, v0, u = np.array([-.5, .2]), np.array([.4, -.3]), np.array([.3, .1])
    for beta in (.2, 0.):
        q = one_ball(x0, v0, {"drag": beta}, u)
        a = 1.5*u; t = .25
        if beta:
            vinf = a/beta; vt = vinf+(v0-vinf)*math.exp(-beta*t); xt = x0+vinf*t+(v0-vinf)*(1-math.exp(-beta*t))/beta
        else:
            vt = v0+a*t; xt = x0+v0*t+a*t*t/2
        assert np.abs(q-np.concatenate((xt, vt))).max() < 1e-13


def test_balls_only_ball_zero_is_pushed():
    """两个相距很远、静止的球：动作只加速第 0 个，第 1 个不动。"""
    q = np.array([-1., 0., 1., 0., 0., 0., 0., 0.])
    q1 = BB.tick(q, np.array([.4, .2]), {"n": 2})
    assert np.abs(q1[4:6]).min() > 0. and np.array_equal(q1[2:4], q[2:4]) and not q1[6:].any()


def test_balls_slow_collision_without_push_still_bounces():
    """接近速度 0.02 < V_REST 但没有动作往里顶：照常按 e 反弹（不当成静止接触）。"""
    lim = 2.-.2
    q = one_ball([lim-.001, 0.], [.02, 0.], {"drag": 0., "restitution": 1.})
    assert q[2] == pytest.approx(-.02, abs=1e-15) and q[0] == pytest.approx(lim-.02*(.25-.05), abs=1e-12)


def test_balls_wall_bounce_is_exact():
    """撞 +x 墙：接触时刻 t_c = (W−R−x0)/v，之后 v ← −e·v；β = 0、无动作时位置和速度有闭式解。"""
    lim = 2.-.2
    for e in (1., .7):
        x0, v = 1.5, 2.
        q = one_ball([x0, .1], [v, 0.], {"drag": 0., "restitution": e})
        tc = (lim-x0)/v
        assert q[0] == pytest.approx(lim-e*v*(.25-tc), abs=1e-12) and q[2] == pytest.approx(-e*v, abs=1e-12)
        assert q[1] == pytest.approx(.1) and q[3] == 0.


def test_balls_head_on_and_oblique_collisions():
    """等质量正碰：e = 1 交换速度；一般 e：法向相对速度变成 −e 倍，切向不变，动量守恒，动能损失 (1−e²)/4·v_rel,n²。"""
    params = {"n": 2, "drag": 0., "restitution": 1.}
    q = np.array([-.5, 0., .5, 0., 1., 0., -1., 0.])
    q1 = BB.tick(q, np.zeros(2), params)
    # 接触时刻 t_c = (1 − 0.4)/2 = 0.3 s > 0.25：这个 tick 内还没碰；再推一个 tick
    q2 = BB.tick(q1, np.zeros(2), params)
    assert np.allclose(q2[4:], [-1., 0., 1., 0.], atol=1e-12)
    assert np.allclose(q2[:4], [-.5+.3-.2, 0., .5-.3+.2, 0.], atol=1e-11)
    for e in (.5, .8):
        params = {"n": 2, "drag": 0., "restitution": e}
        q = np.array([-.3, -.1, .3, .05, 1.2, .4, -.6, -.2])
        p0 = q[4:6]+q[6:8]
        q1 = BB.tick(q, np.zeros(2), params)
        assert np.allclose(q1[4:6]+q1[6:8], p0, atol=1e-12)
        dv = q1[4:6]-q[4:6]                                       # 无阻力、无动作：速度变化只来自冲量，沿 −n
        nrm = -dv/np.linalg.norm(dv)
        tc = (q1[0:2]-q[0:2]-q1[4:6]*.25)/(q[4:6]-q1[4:6])       # 由位置反推碰撞时刻（两个分量应给出同一个 t_c）
        assert tc[0] == pytest.approx(tc[1], abs=1e-9)
        pi, pj = q[0:2]+q[4:6]*tc[0], q[2:4]+q[6:8]*tc[0]
        assert np.linalg.norm(pj-pi) == pytest.approx(.4, abs=1e-9)   # 碰撞时刻两球心相距 2R
        assert np.allclose(nrm, (pj-pi)/.4, atol=1e-9)                 # 冲量沿连心线
        rel0 = q[6:8]-q[4:6]; rel1 = q1[6:8]-q1[4:6]
        vn0 = rel0@nrm
        assert vn0 < 0 and rel1@nrm == pytest.approx(-e*vn0, abs=1e-9)
        tang = np.array([-nrm[1], nrm[0]])
        assert rel1@tang == pytest.approx(rel0@tang, abs=1e-9)
        ke = lambda z: .5*(z[4:8]**2).sum()
        assert ke(q)-ke(q1) == pytest.approx((1-e*e)/4*vn0**2, abs=1e-9)


def test_balls_elastic_energy_conserved_and_no_penetration():
    """e = 1、β = 0、无动作：多次碰撞后动能不变；任何参数下球都不穿墙，球间重叠 < 1e−5。"""
    params = {"n": 5, "drag": 0., "restitution": 1.}
    for tid in range(3):
        st = BB.trajectory(tid, np.zeros((32, 2)), params)
        ke = .5*(st[:, 10:]**2).sum(1)
        assert np.ptp(ke)/ke[0] < 1e-10
    for params in ({}, {"n": 5, "restitution": .5, "drag": 1.}, {"n": 5, "restitution": .5, "drag": 0.}):
        for tid in range(15):
            st = BB.trajectory(tid, controls(400+tid, 2.), params)
            n = params.get("n", 3)
            for q in st:
                gaps, _ = BB._gaps(q[:2*n].reshape(n, 2))
                assert gaps[:4*n].min() >= -1e-12 and gaps.min() > -1e-5


def test_balls_pressed_against_wall_come_to_rest():
    """动作一直把球往墙上顶（e = 0.5）：球贴在墙上不动、不穿墙、速度法向分量为 0（静止接触，不会无穷次碰撞）。"""
    q = one_ball([1.6, 0.], [.3, 0.], {"restitution": .5}, u=(.5, 0.), ticks=12)
    assert q[0] == pytest.approx(1.8, abs=1e-12) and abs(q[2]) < 1e-12


def test_balls_collisions_converge_in_substeps():
    """事件驱动：子步数从 32 加到 128，有碰撞的轨迹几乎不变（只差接触时刻二分的精度和极少的擦边碰撞）。"""
    params = {"n": 4}
    u = controls(9)
    a = BB.trajectory(9, u, params); b = BB.trajectory(9, u, params, substeps=128)
    assert np.abs(a-b).max() < 1e-8


def test_balls_resting_contact_does_not_swallow_other_collisions():
    """S23a-fix1（Codex P1 反例）：第 0 个球贴右墙、被动作一直往墙上顶（静止接触）；第 1 个球 0.001 s 后撞左墙，
    必须照常按 e = 1 反弹（原实现把剩余子步自由推进再全局投影，第 1 个球停在墙上、速度清零）。"""
    h = .25/32
    pos = np.array([[1.8, 0.], [-1.799, 0.]]); vel = np.array([[0., 0.], [-1., 0.]]); acc = np.array([[.75, 0.], [0., 0.]])
    BB.substep(pos, vel, acc, 0., 1., h)
    assert pos[1, 0] == pytest.approx(-1.8+(h-.001), abs=1e-12) and vel[1, 0] == 1.     # −1.7931875，速度 +1
    assert pos[0, 0] == pytest.approx(1.8, abs=1e-15) and vel[0, 0] == 0.


def test_balls_resting_ball_hit_toward_the_wall_bounces():
    """S23a-fix1：第 0 个球贴墙静止接触时，第 1 个球（e = 1、速度 +1）把它撞向墙：牛顿摆，第 0 个撞墙反弹、再把速度
    还给第 1 个，第 1 个以 −1 离开（不能被投影吞掉）。第 1 个球不受动作，速度精确是 −1。"""
    h = .25/32
    pos = np.array([[1.8, 0.], [1.8-.4-.001, 0.]]); vel = np.array([[0., 0.], [1., 0.]]); acc = np.array([[.75, 0.], [0., 0.]])
    BB.substep(pos, vel, acc, 0., 1., h)
    assert vel[1] == pytest.approx([-1., 0.], abs=1e-12)
    assert pos[1, 0] == pytest.approx(1.4-(h-.001), abs=1e-6)          # 只差静止接触压进去的那一点（.75·t²/2 量级）
    assert pos[0, 0] <= 1.8 and abs(vel[0, 0]) < 1e-2


def test_balls_grazing_collision_inside_a_substep_is_found():
    """S23a-fix1（Codex P2 反例）：两球在子步内部擦边相碰、子步两端都不重叠。无阻力无动作时是直线运动：
    接触时刻解 |d₀ + Δv·t| = 2R 的二次方程，冲量沿连心线（e = 1 交换法向速度），之后再直线运动——与闭式解比。"""
    q = np.array([-.18046875, 0., .18046875, .3999, 1.4, 0., -1.4, 0.])
    q1 = BB.tick(q, np.zeros(2), {"n": 2, "drag": 0., "restitution": 1.})
    p, v = q[:4].reshape(2, 2), q[4:].reshape(2, 2)
    d0, dv = p[1]-p[0], v[1]-v[0]
    a, b, c = dv @ dv, 2*d0 @ dv, d0 @ d0-.16
    tc = (-b-math.sqrt(b*b-4*a*c))/(2*a)
    assert 0. < tc < .25
    pc = p+v*tc; nrm = (pc[1]-pc[0])/.4
    vn = (v[1]-v[0]) @ nrm
    assert vn < -.05                                                    # 真碰撞（法向接近速度约 0.0626），不是静止接触
    v1 = v.copy(); v1[0] += vn*nrm; v1[1] -= vn*nrm
    expect = np.concatenate(((pc+v1*(.25-tc)).ravel(), v1.ravel()))
    assert np.abs(q1-expect).max() < 1e-10


def test_balls_first_contact_matches_exact_roots():
    """连续碰撞检测与精确根对照：β = 0 时球对的 |d(t)|² − 4R² 是 t 的四次多项式（第 0 个球有加速度）、墙间隙是二次式；
    随机的近距离状态（一部分子步两端都不重叠、中间擦边），_first_contact 的时刻 = 最早的实根（差 < 1e−10）。"""
    g = np.random.default_rng(0); h = .25/32; checked = graze = 0
    for k in range(400):
        ang = g.uniform(0., 2*math.pi); nrm = np.array([math.cos(ang), math.sin(ang)]); center = g.uniform(-1.2, 1.2, 2)
        acc = np.zeros((2, 2)); acc[0] = g.uniform(-3., 3., 2)
        if k % 2:                                                    # 随机近距离
            off = .5*(.4+g.uniform(0., .02))*nrm
            vel = g.uniform(-3., 3., (2, 2))
        else:                                                        # 构造擦边：子步中点最近距离略小于 2R，两端都不重叠
            rel = g.uniform(4., 6.)*np.array([-nrm[1], nrm[0]])
            off = .5*(g.uniform(.3999, .4)*nrm-rel*h/2)
            base = g.uniform(-1., 1., 2); vel = np.array([base-rel/2, base+rel/2])
            acc[0] *= .1
        pos = np.array([center-off, center+off])
        assert BB._sgaps(pos).min() > 0.                             # 起点不重叠
        c, t = BB._first_contact(pos, vel, acc, 0., h, set())
        d0, dv, da = pos[1]-pos[0], vel[1]-vel[0], acc[1]-acc[0]
        poly = np.polynomial.polynomial.Polynomial
        d = [poly([d0[k], dv[k], da[k]/2]) for k in range(2)]
        roots = (d[0]**2+d[1]**2-.16).roots()
        roots = sorted(r.real for r in roots if abs(r.imag) < 1e-9 and 0. < r.real < h)
        walls = []
        for k in range(2):
            for i in range(2):
                for side in (1., -1.):
                    w = poly([1.8-side*pos[i, k], -side*vel[i, k], -side*acc[i, k]/2]).roots()
                    walls += [r.real for r in w if abs(r.imag) < 1e-9 and 0. < r.real < h]
        first = min(roots[:1]+walls, default=None)
        if first is None:
            assert c is None
            continue
        assert c is not None and t == pytest.approx(first, abs=1e-10)       # 阈值 −EPS：擦边时差 EPS/|S'|
        checked += 1
        end = BB._sgaps(BB.flow(pos, vel, acc, 0., h)[0])
        graze += roots and first == roots[0] and end[-1] > 0.
    assert checked > 100 and graze > 5


def test_balls_curvature_bound_holds():
    """_curvature 的 M 是 [0, T] 上 S'' 的上界（分支定界的前提）：随机状态（含阻力、动作）下用细网格二阶差分核对。"""
    g = np.random.default_rng(1); h = .25/32
    ts = np.linspace(0., h, 401); dt = ts[1]-ts[0]
    for _ in range(60):
        n = int(g.integers(2, 6)); beta = float(g.choice([0., .3, 1.]))
        pos = g.uniform(-1.7, 1.7, (n, 2)); vel = g.uniform(-4., 4., (n, 2)); acc = np.zeros((n, 2)); acc[0] = g.uniform(-3., 3., 2)
        M = BB._curvature(pos, vel, acc, beta, h)
        S = np.array([BB._sgaps(BB.flow(pos, vel, acc, beta, t)[0]) for t in ts])
        d2 = (S[2:]-2*S[1:-1]+S[:-2])/dt**2
        assert (d2.max(0) <= M*(1+1e-6)+1e-4).all()                   # 墙的界是紧的（t = 0 处取等），留差分噪声的余量
        assert (M > 0.).any()


def test_balls_flow_small_drag_is_stable():
    """S23a-fix1（Codex P2 反例）：极小正阻力 β 时 p(τ) = p + v·E + a·F 不能相近数相减。参照用 700 位十进制独立计算
    E = (1 − e^{−βτ})/β、F = (τ − E)/β；β → 0 连续地趋近匀加速（原实现 β = 1e−16 时一个 tick 位移少约 3%）。"""
    from decimal import Decimal, getcontext
    getcontext().prec = 700                                      # β = 1e−300 也要算准 1 − e^{−βτ}
    p0, v0, a = np.array([[.1, -.2]]), np.array([[.3, .5]]), np.array([[.75, -.4]])
    for beta in (1e-300, 1e-16, 1e-12, 1e-8, 1e-4, .2, 1., 3.):
        for tau in (.25/32, .001, .25):
            p, v = BB.flow(p0, v0, a, beta, tau)
            B, T = Decimal(beta), Decimal(tau)
            E = (1-(-B*T).exp())/B; F = (T-E)/B
            for k in range(2):
                pr = Decimal(p0[0, k])+Decimal(v0[0, k])*E+Decimal(a[0, k])*F
                vr = Decimal(v0[0, k])*(-B*T).exp()+Decimal(a[0, k])*E
                assert abs(float(pr)-p[0, k]) < 1e-15 and abs(float(vr)-v[0, k]) < 1e-15
    q0 = one_ball([-.5, .2], [.4, -.3], {"drag": 0.}, (.3, .1))
    for beta in (5e-324, 1e-320, 1e-300, 1e-16, 1e-14, 1e-10):      # 含次正规数（βτ 下溢）
        q = one_ball([-.5, .2], [.4, -.3], {"drag": beta}, (.3, .1))
        assert np.abs(q-q0).max() < 1e-13+beta                         # 物理上差 O(β)


def test_balls_resting_contact_does_not_hit_the_event_cap():
    """静止接触与别的碰撞并存时不会在同一时刻来回触发事件（到 MAX_EVENTS 走投影兜底）：球被顶在墙上、另一个球追上来；
    第 0 个球把另一个球推到墙上一直顶着（墙对第 1 个球也算静止接触，它被别的静止接触顶着）。每个子步的事件数远小于上限。"""
    counts = []
    orig = type(BB).substep

    def counting(self, *a):
        k = orig(self, *a); counts.append(k); return k
    try:
        type(BB).substep = counting
        params = {"n": 2, "restitution": .5}
        a = np.array([1.6, 0., 1.0, 0., .3, 0., .2, 0.])          # 第 0 个被顶在墙上，第 1 个追上来靠着它
        b = np.array([1.0, 0., 1.6, 0., .3, 0., .2, 0.])          # 第 0 个把第 1 个推到墙上、一直顶着
        for _ in range(12):
            a = BB.tick(a, np.array([.5, 0.]), params); b = BB.tick(b, np.array([.5, 0.]), params)
    finally:
        type(BB).substep = orig
    assert max(counts) < 8                                           # 实测最多 5（几次真碰撞的反弹串），上限是 20
    assert a[0] == pytest.approx(1.8, abs=1e-9) and a[2] < 1.4+1e-5     # 第 1 个按 e 弹开或靠着，不穿过第 0 个
    assert b[2] == pytest.approx(1.8, abs=1e-9) and b[0] == pytest.approx(1.4, abs=1e-4)


def test_balls_drawn_as_discs_at_positions():
    q = BB.initial(2, {})
    f = BB.draw(q[None], 2, {}, Variant(), ticks=np.array([-1]))[0]
    for p, color in zip(q[:6].reshape(3, 2), (COLORS["orange"], so.OTHER_COLORS[0], so.OTHER_COLORS[1])):
        rc = np.array([128-48*p[1], 128+48*p[0]])
        assert np.abs(centroid(f, color, 2.)-rc).max() < .5
        r, c = int(rc[0]), int(rc[1])
        assert np.abs(f[r, c].astype(float)-color).max() <= 1
    # 墙的内沿就是 W：墙线中心在 128 ± 48·W ± 2 像素
    row = f[128].astype(float)
    assert np.abs(row[int(128+48*2.+2)]-so.BODY_COLOR).max() < 3


# —————————————————————— wave ——————————————————————

def modes(m, axis):
    x = np.arange(WV.N)
    mode = np.cos(2*np.pi*m*(x+.5)/WV.N)
    return mode[:, None]*np.ones(WV.N) if axis == 0 else np.ones(WV.N)[:, None]*mode


def test_wave_free_modes_follow_damped_oscillator():
    """无源（load = 0）时余弦模 a·cos(2πm·x) 按阻尼振子演化：ω = c·2π·m，衰减 γ/2（数值用说明里的公式重写）。"""
    for m, axis, c, g in ((3, 0, .15, .1), (7, 1, .3, 0.), (1, 0, .08, .5)):
        params = {"speed": c, "damping": g, "load": 0.}
        u0 = .3*modes(m, axis)
        q = WV.pack(u0, np.zeros_like(u0), np.array([.5, .5]))
        st = WV.advance(q, np.zeros((32, 2)), params, 0, 8)
        w = c*2*np.pi*m; wd = math.sqrt(w*w-g*g/4)
        for k, s in enumerate(st):
            t = .25*k
            amp = math.exp(-g*t/2)*(math.cos(wd*t)+g/(2*wd)*math.sin(wd*t))
            assert np.abs(WV.unpack(s)[0]-.3*amp*modes(m, axis)).max() < 1e-12


def test_wave_uniform_mode_drifts_with_damping():
    """均匀模（k = 0）不受力：ü₀ = −γu̇₀，u₀(t) = u₀ + u̇₀·(1 − e^{−γt})/γ；γ = 0 时匀速。"""
    for g in (.3, 0.):
        N = WV.N; u = np.full((N, N), .1); w = np.full((N, N), .2)
        q1 = WV.unpack(WV.tick(WV.pack(u, w, np.array([.5, .5])), np.zeros(2), {"damping": g, "load": 0.}))
        drift = .2*(1-math.exp(-g*.25))/g if g else .2*.25
        assert np.abs(q1[0]-.1-drift).max() < 1e-14 and np.abs(q1[1]-.2*math.exp(-g*.25)).max() < 1e-14


def test_wave_energy_conserved_without_damping_and_source():
    params = {"damping": 0., "load": 0., "speed": .2}
    q = WV.initial(5, {"load": 0.})
    st = WV.advance(q, np.zeros((32, 2)), params, 0, 32)

    def energy(s):
        u, w, _ = WV.unpack(s)
        uh = np.fft.fft2(u)
        grad2 = (np.abs(uh)**2*(WV.kr[:, None]**2+WV.kr[None, :]**2)).sum()/WV.N**2
        return .5*(w**2).sum()+.5*.2**2*grad2
    E = np.array([energy(s) for s in st])
    assert np.ptp(E)/E[0] < 1e-10


def test_wave_static_dent_is_fixed_point_and_mean_stays_zero():
    """静止的激振源：u = −load·(B(x−p) − B̄)、u_t = 0 是 tick 的不动点；B 是 σ = 0.03 的周期高斯（在格点上直接算）。"""
    p = np.array([.3, .8]); load = .25
    c = (np.arange(WV.N)+.5)/WV.N
    dx = (c-p[0]+.5) % 1.-.5; dy = (c-(1-p[1])+.5) % 1.-.5
    B = np.exp(-(dy[:, None]**2+dx[None, :]**2)/(2*.03**2))
    u = -load*(B-B.mean())
    q = WV.pack(u, np.zeros_like(u), p)
    q1 = WV.tick(q, np.zeros(2), {})
    assert np.abs(q1-q).max() < 1e-12
    st = WV.trajectory(1, controls(1), {})
    assert np.abs(st[:, :WV.N**2].mean(1)).max() < 1e-13


def test_wave_moving_source_is_second_order_in_substeps():
    q = WV.initial(2, {}); u = np.full((32, 2), .4)
    run = lambda m: WV.tick(q, u[0], {}, substeps=m)[:WV.N**2]
    a, b, c = run(4), run(8), run(16)
    e1, e2 = np.abs(a-b).max(), np.abs(b-c).max()
    assert e2 < 1e-3 and 3. < e1/e2 < 5.
    assert np.abs(run(16)-WV.tick(q, u[0], {})[:WV.N**2]).max() == 0.            # 默认 16 个子步


def test_wave_periodic_translation_equivariance():
    q = WV.initial(3, {}); N = WV.N
    u, w, p = WV.unpack(q); act = np.array([.3, -.2])
    a = WV.unpack(WV.tick(q, act, {}))
    # 右移 5 列 ↔ x + 5/N；下移 3 行 ↔ y − 3/N（y 向上）
    q2 = WV.pack(np.roll(u, (3, 5), (0, 1)), np.roll(w, (3, 5), (0, 1)), (p+np.array([5/N, -3/N])) % 1.)
    b = WV.unpack(WV.tick(q2, act, {}))
    assert np.abs(np.roll(a[0], (3, 5), (0, 1))-b[0]).max() < 1e-12
    assert np.abs(np.roll(a[1], (3, 5), (0, 1))-b[1]).max() < 1e-12


def test_wave_source_follows_controls_and_is_drawn(runs):
    st, u = runs["wave"]
    p = st[:, 2*WV.N**2:]
    expected = (p[1]+np.concatenate((np.zeros((1, 2)), np.cumsum(.4*u*.25, 0)))) % 1.
    assert np.abs((p[1:]-expected+.5) % 1.-.5).max() < 1e-12 and np.array_equal(p[0], p[1])   # 预跑时源不动
    f = WV.draw(st[20:21], 3, {}, Variant(), ticks=np.array([19]))[0]
    r = 256*(1-p[20, 1]); c = 256*p[20, 0]
    assert np.abs(f[int(r) % 256, int(c+5) % 256].astype(float)-so.MARK_COLOR).max() < 60


def test_wave_render_is_linear_invertible_and_unsaturated(runs):
    st, _ = runs["wave"]
    s = st[12]; u, _, p = WV.unpack(s)
    f = WV.draw(s[None], 3, {}, Variant(background="plain"), ticks=np.array([11]))[0].astype(float)
    bg = _texture("plain")[0, 0]
    pos = np.array(COLORS["orange"])-bg; neg = WV.NEG_COLOR-bg
    up = np.repeat(np.repeat(u, 2, 0), 2, 1)
    ring = WV.draw(WV.pack(np.zeros_like(u), np.zeros_like(u), p)[None], 3, {}, Variant(background="plain"),
                   ticks=np.array([11]))[0]
    free = np.abs(ring.astype(float)-bg).max(-1) < 1
    rec = np.where(((f-bg)@pos) >= ((f-bg)@neg), ((f-bg)@pos)/(pos@pos), -((f-bg)@neg)/(neg@neg))
    assert np.abs(rec-up)[free].max() < .01
    assert np.abs(u).max() < 1.
    for color in COLORS:
        for background in ("grid", "plain", "checker", "noise"):
            assert WV.draw(s[None], 3, {}, Variant(color=color, background=background), ticks=np.array([11])).max() < 255


def test_wave_amplitude_stays_in_display_range():
    """默认参数下 |u| < 1（画面不截断）；最极端的角（c = 0.08、无阻尼、load = 0.4，激振源超音速）截断的像素 < 0.1%。"""
    for params, frac in (({}, 0.), ({"speed": .08, "damping": 0., "load": .4}, 1e-3)):
        for tid in range(6):
            u = WV.trajectory(tid, controls(500+tid), params)[:, :WV.N**2]
            assert (np.abs(u) > 1).mean() <= frac


def test_wave_low_modes_underdamped_over_the_range_and_hidden_pools(monkeypatch):
    """参数范围保证最低非零模欠阻尼（γ/2 < 2π·c）；放宽范围后越界的组合会报错；隐藏状态是 8×8 池化 + p。"""
    (_, (c_lo, _)), (_, (_, g_hi)) = so.PARAMS["wave"]["speed"], so.PARAMS["wave"]["damping"]
    assert g_hi/2 < 2*np.pi*c_lo
    monkeypatch.setitem(so.PARAMS, "wave", dict(so.PARAMS["wave"], damping=(.1, (0., 2.))))
    with pytest.raises(ValueError):
        WV.tick(WV.initial(0, {}), np.zeros(2), {"speed": .08, "damping": 1.1})
    st = WV.trajectory(0, controls(0), {})
    h = WV.hidden(st)
    N = WV.N; u = st[5, :N*N].reshape(N, N); w = st[5, N*N:2*N*N].reshape(N, N)
    assert h.shape == (34, 514) and h[5, 0] == pytest.approx(u[:8, :8].mean()) and h[5, 256+17] == pytest.approx(w[8:16, 8:16].mean())
    assert np.array_equal(h[:, 512:], st[:, 2*N*N:])
