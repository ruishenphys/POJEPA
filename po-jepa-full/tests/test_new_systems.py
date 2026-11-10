"""S20a 新增场景（new_systems.py）：随机流、确定性、物理解析解、数值收敛、画面包含全部状态。"""
import math

import numpy as np
import pytest

from po_jepa import datasets, new_systems
from po_jepa.systems import COLORS, _texture
from po_jepa.variants import Variant

GS = new_systems.SYSTEMS["gray_scott"]
PU = new_systems.SYSTEMS["push"]


def controls(seed):
    """v6 规则（1/4/8 段、20% 零动作、[-0.5,0.5]²）的执行动作 [32,2]，只供测试。"""
    g = np.random.default_rng(seed); u = np.zeros((32, 2)); t = 0
    while t < 32:
        n = int(g.choice([1, 4, 8])); u[t:t+n] = 0. if g.random() < .2 else g.uniform(-.5, .5, 2); t += n
    return u


@pytest.fixture(scope="module")
def gs_fields():
    u = controls(0)
    return u, GS.fields(3, u, {})


@pytest.fixture(scope="module")
def gs_run(gs_fields):
    u, (U, V, p) = gs_fields
    return u, (GS.states(U, V, p), GS.draw(U, V, p, 3, Variant()))


# —— 随机流与参数 ——

def test_random_streams_follow_the_v7_scheme_after_registration():
    for order, name in enumerate(("gray_scott", "push")):
        expected = datasets.SYSTEMS.index(name) if name in datasets.SYSTEMS else len(datasets.SYSTEMS)+order
        assert new_systems.SYSTEM_INDEX[name] == expected
    a = new_systems.rng("gray_scott", 0, 0).random(4); b = new_systems.rng("push", 0, 0).random(4)
    ref = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1703, 1005, 0, 0]))).random(4)
    assert np.array_equal(a, ref) and not np.array_equal(a, b)


def test_params_defaults_and_ranges():
    assert new_systems.param("push", {}, "mu_s") == .3
    for system, name, bad in (("gray_scott", "feed", .03), ("gray_scott", "kill", .07), ("push", "slope", .7),
                              ("push", "size", float("nan"))):
        with pytest.raises(ValueError):
            new_systems.param(system, {name: bad}, name)
    with pytest.raises(ValueError):
        GS.fields(0, np.zeros((31, 2)), {})


# —— gray_scott ——

def test_gray_scott_shapes_determinism_and_change(gs_run):
    u, (s, f) = gs_run
    assert s.shape == (33, 514) and f.shape == (33, 256, 256, 3) and f.dtype == np.uint8
    s2, f2 = GS.frames(3, u, {}, Variant())
    assert np.array_equal(s, s2) and np.array_equal(f, f2)
    assert np.abs(s[-1, :512]-s[16, :512]).mean() > 1e-3            # 场在变（不是均匀稳态）


def test_gray_scott_homogeneous_fixed_point():
    N = GS.N; U, V = np.ones((N, N)), np.zeros((N, N))
    U2, V2, _ = GS._run(U, V, np.array([.3, .7]), np.array([1e-3, 0.]), 20., .5, .04, .06, 0.)
    assert np.abs(U2-1).max() < 1e-12 and np.abs(V2).max() < 1e-12


def test_gray_scott_strang_second_order():
    U, V, p = GS.initial(1)
    vel = np.array([2e-3, -1e-3])
    run = lambda dt: GS._run(U, V, p.copy(), vel, 20., dt, .04, .06, .05)[1]
    a, b, c = run(.5), run(.25), run(.125)
    e1, e2 = np.abs(a-b).max(), np.abs(b-c).max()
    assert e2 < 1e-3 and 3. < e1/e2 < 5.                           # 二阶：步长减半误差约 1/4


def test_gray_scott_periodic_translation_equivariance():
    U, V, p = GS.initial(2); N = GS.N; vel = np.array([1e-3, 2e-3])
    U1, V1, _ = GS._run(U, V, p.copy(), vel, 10., .5, .04, .06, .1)
    # 右移 5 列 ↔ x + 5/N；下移 3 行 ↔ y − 3/N（y 向上）
    Us, Vs = np.roll(U, (3, 5), (0, 1)), np.roll(V, (3, 5), (0, 1))
    U2, V2, _ = GS._run(Us, Vs, (p+np.array([5/N, -3/N])) % 1., vel, 10., .5, .04, .06, .1)
    assert np.abs(np.roll(V1, (3, 5), (0, 1))-V2).max() < 1e-9 and np.abs(np.roll(U1, (3, 5), (0, 1))-U2).max() < 1e-9


def test_gray_scott_injection_point_follows_controls(gs_run):
    u, (s, f) = gs_run
    p = s[:, 512:]
    expected = (p[0]+np.concatenate((np.zeros((1, 2)), np.cumsum(GS.SPEED*u*.25, 0)))) % 1.
    d = (p-expected+.5) % 1.-.5
    assert np.abs(d).max() < 1e-10
    # 画面里能看到注入点：圆环上的像素接近品红
    t = 20; r = 256*(1-p[t, 1]); c = 256*p[t, 0]
    pix = f[t, int(r) % 256, int(c+5) % 256].astype(float)
    assert np.abs(pix-GS.COLOR_MARK).max() < 60


def test_gray_scott_fields_stay_physical(gs_fields):
    _, (U, V, _) = gs_fields                                           # 原始场，不池化
    assert np.isfinite(U).all() and np.isfinite(V).all()
    assert U.min() > -1e-4 and U.max() < 1+1e-4 and V.min() > -1e-4 and V.max() < 1+1e-4   # 谱法下冲约 1e−5
    assert V.max() > .4 and U.min() < .3                               # 高 V、低 U 确实出现（渲染测试要覆盖到）


def test_gray_scott_diffusion_matches_fourier_decay():
    """无反应（F=k=a=0）时 U−1 与 V 的余弦模按 exp(−D·(2πm/N)²·t) 衰减，D 用说明里的数值 0.16、0.08（不读类常量）。"""
    N = GS.N; x = np.arange(N); t = 20.
    for m, axis in ((3, 0), (7, 1)):
        mode = np.cos(2*np.pi*m*x/N); mode = mode[:, None]*np.ones(N) if axis == 0 else np.ones(N)[:, None]*mode
        U0 = 1.+.1*mode
        U, V, _ = GS._run(U0, np.zeros((N, N)), np.array([.5, .5]), np.zeros(2), t, .5, 0., 0., 0.)
        assert np.abs(U-1.-.1*mode*np.exp(-.16*(2*np.pi*m/N)**2*t)).max() < 1e-12
        U, V, _ = GS._run(np.zeros((N, N)), .1*(1+mode), np.array([.5, .5]), np.zeros(2), t, .5, 0., 0., 0.)
        assert np.abs(V-.1*(1+mode*np.exp(-.08*(2*np.pi*m/N)**2*t))).max() < 1e-12


def test_gray_scott_render_is_invertible_and_unsaturated(gs_fields):
    """线性渲染：(U, V) 能从 RGB 解回；大幅改变 V 或 U 一定改变画面；任何颜色/背景都不触到 255。"""
    _, (U, V, p) = gs_fields
    t = 0
    f = GS.draw(U[t:t+1], V[t:t+1], p[t:t+1], 3, Variant(background="plain"), ticks=[t]).astype(float)[0]   # S24-fix1：部分帧要给 ticks
    bg = _texture("plain")
    A = np.stack((GS.MIX_V*(np.array(COLORS["orange"])-bg[0, 0]), GS.MIX_U*(GS.COLOR_U-bg[0, 0])), 1)   # [3,2]
    w, *_ = np.linalg.lstsq(A, (f-bg).reshape(-1, 3).T, rcond=None)
    rep = 256//GS.N; up = lambda X: np.repeat(np.repeat(X, rep, 0), rep, 1).ravel()
    ring = GS.draw(np.ones((1, GS.N, GS.N)), np.zeros((1, GS.N, GS.N)), p[t:t+1], 3, Variant(background="plain"), ticks=[t])[0]
    free = (np.abs(ring.astype(float)-bg).max(-1) < 1).ravel()        # 去掉注入点圆环盖住的像素
    assert np.abs(w[0]-up(V[t]))[free].max() < .02 and np.abs(w[1]-(1-up(U[t])))[free].max() < .02
    V2 = V[t].copy(); r, c = np.unravel_index(np.argmax(V2), V2.shape); V2[r, c] += .1
    U2 = U[t].copy(); r2, c2 = np.unravel_index(np.argmin(U2), U2.shape); U2[r2, c2] -= .1
    for color in COLORS:
        for background in ("grid", "plain", "checker", "noise"):
            v = Variant(color=color, background=background)
            f0 = GS.draw(U[t:t+1], V[t:t+1], p[t:t+1], 3, v, ticks=[t])
            assert f0.max() < 255
            assert not np.array_equal(f0, GS.draw(U[t:t+1], V2[None], p[t:t+1], 3, v, ticks=[t]))
            assert not np.array_equal(f0, GS.draw(U2[None], V[t:t+1], p[t:t+1], 3, v, ticks=[t]))


@pytest.mark.parametrize("params", [{"feed": .035, "kill": .063, "inject": 0.}, {"feed": .05, "kill": .0595, "inject": 0.}])
def test_gray_scott_range_corners_keep_patterns(params):
    """允许范围的两个危险角（低 F 高 k 会熄灭、高 F 低 k 会变均匀），无注入时斑图仍在且在变。"""
    U, V, _ = GS.fields(11, controls(11), params)
    assert V[0].std() > .03 and V[-1].std() > .03 and np.abs(V[-1]-V[16]).mean() > 1e-3


def test_gray_scott_observation_knobs_keep_states_and_reject_spots(gs_run):
    u, (s, f) = gs_run
    s2, f2 = GS.frames(3, u, {}, Variant(color="white", background="checker", pixel_noise=5.))
    assert np.array_equal(s, s2) and not np.array_equal(f, f2)
    for v in (Variant(sigma=4.), Variant(distractors=1)):
        with pytest.raises(ValueError):
            GS.frames(3, u, {}, v)


# —— push ——

def _facing(size=.8, gap=0.):
    """方块在原点、θ=0；推杆在左边，刚好接触（gap>0 留空隙）。"""
    return np.array([-size/2-PU.RP-gap, 0., 0., 0., 0.])


def test_push_static_friction_holds_block():
    u = np.tile([.1, 0.], (32, 1))                                   # 推力 SPEED·0.1 = 0.2 < μ_s·W = 0.3
    s = PU.trajectory(0, u, {}, state=_facing())
    assert np.abs(s[:, 2:]).max() < 1e-12
    delta = PU.contact(s[-1], .8)[2]
    assert abs(delta-PU.SPEED*.1/PU.STIFF) < 1e-6                    # 推杆被顶住：力平衡 STIFF·δ = SPEED·u


def test_push_centered_push_matches_series_solution():
    u = np.tile([.4, 0.], (32, 1))
    s = PU.trajectory(0, u, {}, state=_facing())
    v_block = (s[8, 2]-s[4, 2])/(4*.25)                          # 8 个 tick 内还没到墙（1.6）
    assert abs(v_block-(PU.SPEED*.4-.3)/2) < 1e-6                   # 稳态 v = (SPEED·u − μ_s·W)/2
    assert np.abs(s[:, 3:]).max() < 1e-12                            # 正推：不偏、不转
    assert abs((s[8, 0]-s[4, 0])/(4*.25)-v_block) < 1e-6


def test_push_slope_slides_without_action():
    far = np.array([-2., -2., 0., 0., 0.])
    s = PU.trajectory(0, np.zeros((32, 2)), {"slope": .5, "slope_dir": 90.}, state=far)
    v = (s[8, 2:4]-s[0, 2:4])/2.
    assert abs(v[0]) < 1e-12 and abs(v[1]-(.5-.3*math.sqrt(1-.25))) < 1e-9
    s = PU.trajectory(0, np.zeros((32, 2)), {"slope": .2}, state=far)   # 0.2 < μ_s·√(1−0.04)：不动
    assert np.abs(s[:, 2:]).max() < 1e-12


@pytest.mark.parametrize("size", [.5, 1.])
def test_push_offcenter_field_matches_hand_computation(size):
    """非中心接触：推杆从左面 y0 处压入 δ0，θ=0、方块在原点。手算 F、τ、ρ、s 后核平移和角速度幅值。"""
    d0, y0 = .004, .3*size
    q = np.array([-size/2-PU.RP+d0, y0, 0., 0., 0.])
    f = PU.STIFF*d0                                                  # 0.6
    tau = -y0*f                                                      # r=(−size/2, y0) × (f, 0)
    rho = .383*size
    s = 1-.3/math.sqrt(f**2+(tau/rho)**2)
    dq = PU.field(q, np.zeros(2), {"size": size})
    assert np.allclose(dq, [-f, 0., s*f, 0., s*tau/rho**2], rtol=0, atol=1e-12)


def test_push_release_relaxes_then_stops():
    """无惯性：稳态正推后动作归零，方块第一个 tick 只滑 <1e−3（这个默认案例的结果，不是普遍界），之后完全静止。"""
    u = np.zeros((32, 2)); u[:6] = [.4, 0.]
    s = PU.trajectory(0, u, {}, state=_facing())
    assert 0. < s[7, 2]-s[6, 2] < 1e-3 and np.abs(s[7:, 2:]-s[7, 2:]).max() == 0.     # 第二个释放 tick 起完全静止（S20b 审核）


def test_push_offcenter_push_rotates_clockwise():
    q = _facing(); q[1] = .25                                        # 在中线上方推 +x
    s = PU.trajectory(0, np.tile([.5, 0.], (32, 1)), {"mu_s": .1}, state=q)
    assert s[-1, 4] < -.05


def test_push_step_convergence_and_contact_bounds():
    u = controls(4)
    a = PU.trajectory(4, u, {}); b = PU.trajectory(4, u, {}, steps_per_tick=512)
    assert np.abs(a-b).max() < 1e-5
    for i in range(6):
        s = PU.trajectory(i, controls(i), {"mu_s": .1})
        assert max(PU.contact(q, .8)[2] for q in s) < .02 and np.abs(s[:, :4]).max() < PU.BOUND


def test_push_initial_state_never_overlaps_and_is_paired():
    for i in range(50):
        for size in (.4, 1.2):
            assert PU.contact(PU.initial(i, {"size": size}), size)[2] == 0.
    assert np.array_equal(PU.initial(7, {})[2:], PU.initial(7, {"size": 1.2})[2:])


def test_push_frames_show_orientation_and_knobs():
    q = np.array([[1.5, 1.5, 0., 0., .3], [1.5, 1.5, 0., 0., .3+math.pi]])
    f = PU.draw(q, 0, {}, Variant(), ticks=[0, 1])      # S24-fix1：部分帧要给 ticks
    assert f.shape == (2, 256, 256, 3) and not np.array_equal(f[0], f[1])   # 方块转 180° 画面不同
    u = controls(1)
    s, f1 = PU.frames(1, u, {}, Variant())
    s2, f2 = PU.frames(1, u, {}, Variant(color="green", distractors=2, sigma=4., pixel_noise=3.))
    assert np.array_equal(s, s2) and not np.array_equal(f1, f2)
    assert np.array_equal(f1, PU.frames(1, u, {}, Variant())[1])
