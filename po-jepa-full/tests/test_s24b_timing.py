"""S24b 时间采样：调节项 fps、jitter、exposure、switch_offset（docs/segments/S24b.md）。

核的是：解析与覆盖表；帧时刻（帧数、帧间隔、抖动的范围和随机流）；整 tick 的帧与 fps=4 的数据集逐位相同（帧率 2、8、16 是
同一条轨迹的子采样或加密，12 个场景、带像素噪声和别的调节项）；同一时刻的帧与取了哪些别的时刻无关；tick 内部的状态与独立的
细步积分一致；switch_offset 的下达动作、执行动作和物理（与独立的细步积分对照）；曝光是因果窗口里取样画面的平均；
tick 内部的帧的像素噪声；二阶场景的前一帧；编码、探针、规划拒绝时间采样调节项。
写死的数值（帧数、取样时刻、段边界）不调被测函数推出来（错误清单 2026-11-09 S23a 那条）。
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import expm

from po_jepa import datasets, simulation, systems, timing, variants
from po_jepa.variants import Variant

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"src"
ALL = ("double_well", "van_der_pol", "multi_particle", "advection_diffusion", "landscape", "gray_scott", "push",
       "pendulum", "bouncing_balls", "wave", "cart_pole", "spring_chain")
SPOTS = ("double_well", "van_der_pol", "multi_particle", "landscape", "push", "pendulum", "cart_pole", "spring_chain",
         "bouncing_balls")
ODE = ("double_well", "van_der_pol", "multi_particle", "landscape")


@pytest.fixture
def use():
    old = datasets.active()
    def set_(system, knobs="", count=200):
        text = ("shape=ring," if system == "landscape" and "shape=" not in knobs else "") + knobs
        text = text.rstrip(",")
        spec = datasets.DatasetSpec(system, count, variants.format(system, variants.parse(system, text)))
        datasets._set_for_tests(spec)
        return spec
    yield set_
    datasets._set_for_tests(old)


def join(*parts):
    return ",".join(p for p in parts if p)


def frames(system=None):
    return systems.get(system).frames


def u4_of(tid):
    v = datasets.active().knobs
    return systems.applied_subticks(tid, v, systems.commanded_controls(tid))


# —— 解析、格式、覆盖表 ——

def test_parse_format_and_validate():
    v = variants.parse("double_well", "fps=8,jitter=0.2,exposure=0.5,switch_offset=0.25")
    assert (v.fps, v.jitter, v.exposure, v.switch_offset) == (8, .2, .5, .25) and type(v.fps) is int
    assert variants.format("double_well", v) == "exposure=0.5,fps=8,jitter=0.2,switch_offset=0.25"
    assert v.changes_timing() and not Variant().changes_timing() and not variants.parse("push", "horizon=128").changes_timing()
    # 默认值不写进场景串
    assert variants.format("double_well", variants.parse("double_well", "fps=4,jitter=0,exposure=0,switch_offset=0")) == ""
    assert variants.FPS == (2, 4, 8, 16) and variants.SWITCH_OFFSETS == (0., .25, .5, .75) and variants.JITTER_MAX == .4
    for bad in ("fps=3", "fps=32", "fps=8.0", "fps=1", "jitter=0.5", "jitter=-0.1", "jitter=nan", "exposure=1.5",
                "exposure=-1", "exposure=inf", "switch_offset=0.3", "switch_offset=1", "switch_offset=-0.25"):
        with pytest.raises(ValueError):
            variants.parse("double_well", bad)
    with pytest.raises(ValueError):
        variants.validate("double_well", Variant(fps=8.))
    for system in ALL:
        text = ("shape=ring," if system == "landscape" else "") + "fps=16,jitter=0.4,exposure=1,switch_offset=0.75"
        assert variants.parse(system, text).fps == 16


def test_coverage_lists_time_knobs_for_every_scene():
    cov = variants.coverage()
    assert variants.KNOB_GROUPS["时间"] == ("horizon", "fps", "jitter", "exposure", "switch_offset")
    text = (ROOT/"DATASET.md").read_text(encoding="utf-8")
    for name in ("fps", "jitter", "exposure", "switch_offset"):
        assert cov[name] == datasets.SYSTEMS
        assert f"| 时间 | `{name}` | " + " | ".join(["✓"]*12) + " | 12 |" in text


# —— 帧时刻 ——

@pytest.mark.parametrize("fps,horizon,count,step", [(2, 32, 17, 2.), (4, 32, 33, 1.), (8, 32, 65, .5), (16, 32, 129, .25),
                                                     (2, 128, 65, 2.), (16, 64, 257, .25)])
def test_frame_times_grid(use, fps, horizon, count, step):
    use("van_der_pol", f"fps={fps},horizon={horizon}")
    t = timing.frame_times(5)
    assert t.shape == (count,) and t.dtype == np.float64
    assert np.array_equal(t, step*np.arange(count)) and t[-1] == horizon


def test_frame_times_jitter(use):
    use("pendulum", "fps=8,jitter=0.4")
    t = timing.frame_times(7)
    assert t.shape == (65,) and t[0] == 0. and t[-1] == 32.
    dev = t-.5*np.arange(65)
    assert np.all(np.abs(dev) <= .4*.5+1e-15) and np.all(np.diff(t) >= .5*.2-1e-12) and np.abs(dev[1:-1]).min() > 0.
    # 流 10，独立构造熵：二阶场景的系统序号是 datasets.SYSTEMS 的位置（pendulum = 7）
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1703, 1007, 7, 10])))
    xi = g.uniform(-1., 1., 65)
    assert np.array_equal(t[1:-1], .5*np.arange(1, 64)+.4*.5*xi[1:-1])
    # 长时程：前面各帧（最后一帧除外）与 horizon=32 相同
    use("pendulum", "fps=8,jitter=0.4,horizon=64")
    t64 = timing.frame_times(7)
    assert t64.shape == (129,) and np.array_equal(t64[:64], t[:64]) and t64[64] != 32.
    # 不同轨迹抖动不同
    use("pendulum", "fps=8,jitter=0.4")
    assert not np.array_equal(timing.frame_times(8), t)


# —— 整 tick 的帧与 fps=4 逐位相同 ——

CASES = [("double_well", "pixel_noise=6,noise=0.1,delay=1", 16), ("van_der_pol", "movers=2,pixel_noise=4", 8),
         ("multi_particle", "process_noise=0.1,occluders=1", 8), ("landscape", "shape=obstacles,gain=2,distractors=1", 2),
         ("advection_diffusion", "pixel_noise=5,background=checker", 8), ("gray_scott", "pixel_noise=3", 8),
         ("push", "movers=1,pixel_noise=5,policy=goal", 8), ("pendulum", "pixel_noise=5,movers=1", 16),
         ("bouncing_balls", "pixel_noise=5,occluders=1", 8), ("wave", "pixel_noise=5", 2),
         ("cart_pole", "pixel_noise=5,saturation=0.3", 8), ("spring_chain", "movers=2,noise=0.1", 16)]


@pytest.mark.parametrize("system,knobs,fps", CASES)
def test_tick_frames_equal_the_4fps_dataset(use, system, knobs, fps):
    use(system, knobs)
    tid = 11 if system != "gray_scott" else 3
    s4, f4 = frames()(tid)
    use(system, join(knobs, f"fps={fps}"))
    s, f = frames()(tid)
    every = fps//4 if fps >= 4 else None
    if every:
        assert f.shape[0] == 32*every+1
        assert np.array_equal(f[::every], f4) and np.array_equal(s[::every], s4)
    else:      # fps=2：每 2 个 tick 一帧
        assert f.shape[0] == 17 and np.array_equal(f, f4[::2]) and np.array_equal(s, s4[::2])
    if system in variants.SECOND_ORDER and fps >= 4:
        # frames_with_prev：fps=4 时就是原来的 34 帧；更高帧率的前一帧在 −Δ，其余同 frames()
        sp, fp = systems.get().frames_with_prev(tid)
        assert np.array_equal(fp[1:], f) and np.array_equal(sp[1:], s) and fp.shape[0] == f.shape[0]+1


def test_default_timing_knobs_do_not_route_through_timing(use, monkeypatch):
    """默认时间采样调节项（fps=4 等）时各场景的 frames() 不经过 timing（原来的实现，逐位不变）。"""
    def boom(*a, **k):
        raise AssertionError("timing.frames 不该被调用")
    monkeypatch.setattr(timing, "frames", boom)
    for system in ("double_well", "advection_diffusion", "push", "pendulum"):
        use(system, "pixel_noise=2")
        systems.get().frames(1)
    use("spring_chain", "")
    systems.get().frames_with_prev(1)


def test_raw_images_context_restores_flag():
    assert systems._RAW[0] is False
    with pytest.raises(RuntimeError):
        with systems.raw_images():
            assert systems._RAW[0] is True
            raise RuntimeError
    assert systems._RAW[0] is False


# —— 同一时刻的帧与别的时刻无关；tick 内部的状态 ——

@pytest.mark.parametrize("system", ["double_well", "multi_particle", "advection_diffusion", "push", "pendulum",
                                    "bouncing_balls", "wave", "cart_pole", "spring_chain"])
def test_same_time_same_frame_across_fps(use, system):
    knobs = "movers=1" if system in SPOTS and system != "bouncing_balls" else ""
    use(system, join(knobs, "fps=8"))
    s8, f8 = frames()(4)
    use(system, join(knobs, "fps=16"))
    s16, f16 = frames()(4)
    assert np.array_equal(f16[::2], f8) and np.array_equal(s16[::2], s8)
    # 半个 tick 的帧确实不是相邻整 tick 帧的复制（取第一个前后两个整 tick 状态不同的半 tick；push 静止时就是不动）
    i = next(i for i in range(1, 64, 2) if not np.array_equal(s8[i-1], s8[i+1]))
    assert not np.array_equal(s8[i], s8[i-1]) and not np.array_equal(s8[i], s8[i+1])


def _fine_rk4(field, q, u, seconds, n=4096):
    h = seconds/n
    for _ in range(n):
        k1 = field(q, u); k2 = field(q+h*k1/2, u); k3 = field(q+h*k2/2, u); k4 = field(q+h*k3, u)
        q = q+h*(k1+2*k2+2*k3+k4)/6
    return q


def test_intra_tick_state_double_well_against_fine_integration(use):
    """jitter 的帧时刻不在子步网格上：状态与独立的 4096 步 RK4（simulation.vector_field，v6 原方程）一致。"""
    use("double_well", "fps=8,jitter=0.3")
    tid = 9
    t = timing.frame_times(tid)
    s, _ = frames()(tid)
    c = simulation.sample_controls(tid)
    u = np.zeros((32, 2))
    for a, b, val in zip(c.starts_ticks, c.ends_ticks, c.values):
        u[a:b] = val
    ref = simulation.reference_trajectory(tid)
    for i in (1, 7, 30, 63):
        k = int(math.floor(t[i])); phi = t[i]-k
        assert 0 < phi < 1 and phi*128 != round(phi*128)
        q = _fine_rk4(simulation.vector_field, ref[k], u[k], phi*.25)
        assert np.abs(s[i]-q).max() < 1e-10


def test_intra_tick_state_pendulum_and_spring_against_independent_solutions(use):
    """pendulum：fps=16 的 1/4 tick 是 12.5 个子步（最后一截是短步）→ 与独立细步 RK4 一致；
    spring_chain：精确传播 → 与独立写的线性 ODE 细步 RK4 一致。"""
    use("pendulum", "fps=16")
    s, _ = frames()(2)
    u = u4_of(2)[::4]
    gl, K = 9.81, 4.
    f = lambda q, uu: np.array([q[1], -gl*math.sin(q[0])-.2*q[1]+K*(uu[0]*math.cos(q[0])+uu[1]*math.sin(q[0]))])
    for i in (1, 2, 3, 41, 127):
        k, j = divmod(i, 4)
        q = _fine_rk4(f, s[4*k], u[k], j*.25*.25)
        assert np.abs(s[i]-q).max() < 1e-8       # 生产积分器（每 tick 50 步 RK4）本身的截断误差约 1e-10–1e-9
    use("spring_chain", "fps=8")
    s, _ = frames()(2)
    u = u4_of(2)[::4]
    n, kk, gamma, K = 4, 16., .1, 1.5
    rest = np.stack((-2.+4.*(np.arange(n)+1.)/(n+1), np.zeros(n)), -1).ravel()
    lap = 2*np.eye(n)-np.eye(n, k=1)-np.eye(n, k=-1)
    def g(q, uu):
        pos = (q[:2*n]-rest).reshape(n, 2); vel = q[2*n:].reshape(n, 2)
        acc = -kk*(lap@pos)-gamma*vel; acc[0] += K*uu
        return np.concatenate((vel.ravel(), acc.ravel()))
    for i in (1, 13, 63):
        k = i//2
        assert np.abs(s[i]-_fine_rk4(g, s[2*k], u[k], .125)).max() < 1e-10


@pytest.mark.parametrize("system", ["van_der_pol", "push", "bouncing_balls", "wave", "cart_pole", "gray_scott"])
def test_partial_step_is_continuous_with_the_substep_grid(use, system):
    """不在子步网格上的时刻：离网格点很近（1e−6 个子步）时状态趋于网格点（两侧都核），说明短步与整子步是同一个格式、同一个起点。"""
    use(system, "fps=8,process_noise=0.1" if system == "van_der_pol" else "fps=8")     # 过程噪声：短步的噪声也要按比例缩
    tid = 6
    v = datasets.active().knobs
    sc = timing._scene(systems.get(), tid, v, 32)
    S = sc.S
    k = 3 if system != "gray_scott" else 1
    on = [k+j/S for j in (2, 3, S//2)]
    eps = 1e-6/S
    st = timing.simulate(sc, u4_of(tid), on+[on[0]+eps, on[1]-eps, on[2]+eps])
    flat = [np.concatenate([np.ravel(z) for z in x]) if isinstance(x, tuple) else np.ravel(x) for x in st]
    for a, b in ((0, 3), (1, 4), (2, 5)):
        d = np.abs(flat[a]-flat[b]).max()
        assert 0. < d < 1e-4, (a, d)


@pytest.mark.parametrize("system", ALL)
def test_two_half_pieces_equal_the_whole_tick(use, system):
    """switch_offset 把一个 tick 分成几段时，每段按子步网格（或精确传播子）推进。同一个动作分成 [0, 1/2)、[1/2, 1) 两段，
    结果与生产的整 tick 相同（RK4、分裂步逐位；谱方法、矩阵指数差在舍入量级）。第二段从 tick 中间出发，核起点位置的处理
    （wave 的源位置、精确传播子的时长）。"""
    use(system, "fps=8")
    tid = 4
    v = datasets.active().knobs
    sc = timing._scene(systems.get(), tid, v, 32)
    u = u4_of(tid)
    k = next(k for k in range(32) if np.abs(u[4*k]).max() > .1)
    q = timing.simulate(sc, u, [float(k)])[0]
    whole, _ = timing._advance(sc, q, u[4*k], k, 0., 1.)
    half, _ = timing._advance(sc, q, u[4*k], k, 0., .5)
    end, _ = timing._advance(sc, half, u[4*k], k, .5, 1.)
    flat = lambda x: np.concatenate([np.ravel(np.abs(z)) for z in x]) if isinstance(x, tuple) else np.ravel(x)
    a, b, c = flat(whole), flat(end), flat(q)
    assert np.abs(a-b).max() <= 1e-12*max(1., np.abs(a).max())
    assert np.abs(a-c).max() > 1e-6          # 这个 tick 确实动了


def test_advection_half_tick_is_exact(use):
    """advection_diffusion：半个 tick 的场 = 起点谱乘以半个 tick 的精确传播子（独立写的表达式）。"""
    use("advection_diffusion", "fps=8")
    tid = 3
    v = datasets.active().knobs
    sc = timing._scene(systems.get(), tid, v, 32)
    u = u4_of(tid)
    q5, q55 = timing.simulate(sc, u, [5., 5.5])
    k = np.fft.fftfreq(256, d=1/256)*2*np.pi
    ky, kx = np.meshgrid(k, k, indexing="ij")
    wx, wy = .4*u[20]
    expect = q5*np.exp((-2e-4*(kx**2+ky**2)-1j*(kx*wx-ky*wy))*.125)
    assert np.abs(q55-expect).max() < 1e-9*np.abs(q5).max()


# —— switch_offset ——

def test_switch_offset_commanded_controls(use):
    use("van_der_pol", "segments=148")
    base = systems.commanded_controls(12)
    use("van_der_pol", "switch_offset=0.5")
    c = systems.commanded_controls(12)
    assert isinstance(c, systems.SubtickControls) and not isinstance(c, simulation.TrajectoryControls)
    assert np.array_equal(c.values, base.values) and c.horizon == 32
    assert c.starts_subticks[0] == 0 and c.ends_subticks[-1] == 128
    assert np.array_equal(c.ends_subticks[:-1], 4*base.ends_ticks[:-1]+2)
    assert np.array_equal(c.starts_subticks[1:], 4*base.starts_ticks[1:]+2)
    use("van_der_pol", "switch_offset=0.25,horizon=64")
    assert systems.commanded_controls(12).ends_subticks[-1] == 256
    # 下达动作检查
    good = dict(trajectory_id=1, starts_subticks=np.array([0, 6], dtype=np.int64), ends_subticks=np.array([6, 256], dtype=np.int64),
                values=np.zeros((2, 2)))
    systems.SubtickControls(**good)
    systems.SubtickControls(**{**good, "ends_subticks": np.array([6, 128], dtype=np.int64)})     # 覆盖 [0,128) = horizon 32，也合法
    for change in (dict(ends_subticks=np.array([6, 130], dtype=np.int64)), dict(ends_subticks=np.array([6, 192], dtype=np.int64)), dict(starts_subticks=np.array([1, 6], dtype=np.int64)),
                   dict(starts_subticks=np.array([0, 7], dtype=np.int64)), dict(values=np.full((2, 2), .6)),
                   dict(ends_subticks=np.array([0, 256], dtype=np.int64), starts_subticks=np.array([0, 0], dtype=np.int64)),
                   dict(starts_subticks=np.array([0., 6.]))):
        with pytest.raises(ValueError):
            systems.SubtickControls(**{**good, **change})
    # 逐 tick 的接口拒收
    with pytest.raises(ValueError, match="SubtickControls"):
        systems.applied_ticks(12, datasets.active().knobs, systems.commanded_controls(12))
    with pytest.raises(ValueError):
        systems.get().variant_trajectory(12, datasets.active().knobs)


@pytest.mark.parametrize("knobs", ["", "delay=2,gain=1.5", "noise=0.2,saturation=0.3", "delay=1,noise=0.1,horizon=64"])
def test_applied_subticks_repeat_applied_ticks(use, knobs):
    use("multi_particle", knobs)
    v = datasets.active().knobs
    c = systems.commanded_controls(5)
    a = systems.applied_ticks(5, v, c); b = systems.applied_subticks(5, v, c)
    assert b.shape == (4*v.horizon, 2) and np.array_equal(b, np.repeat(a, 4, axis=0))
    # 挪了边界：同一 tick 的执行噪声相同，延迟按整 tick 挪
    use("multi_particle", join(knobs, "switch_offset=0.75"))
    v2 = datasets.active().knobs
    b2 = systems.applied_subticks(5, v2, systems.commanded_controls(5))
    if not v.noise and not v.delay and v.saturation is None:
        expect = np.repeat(per_tick_values(c, v.horizon), 4, axis=0)
        shifted = np.concatenate((expect[:1].repeat(3, 0), expect[:-3]))
        assert np.array_equal(b2, v.gain*shifted)


def per_tick_values(c, T):
    u = np.full((T, 2), np.nan)
    for a, b, val in zip(c.starts_ticks, c.ends_ticks, c.values):
        u[a:b] = val
    return u


def _fine_subtick(field, q, u4, k, n_per_quarter=1024):
    """独立细步：tick k 的 4 个 1/4 tick 各用 u4 的那一行，RK4 各 n 步。"""
    for j in range(4):
        q = _fine_rk4(field, q, u4[4*k+j], .0625, n_per_quarter)
    return q


@pytest.mark.parametrize("system,offset", [("double_well", .5), ("pendulum", .25), ("cart_pole", .75)])
def test_switch_offset_physics_against_fine_integration(use, system, offset):
    use(system, f"switch_offset={offset}")
    tid = 21
    u4 = u4_of(tid)
    assert any(not np.array_equal(u4[4*k], u4[4*k+3]) for k in range(32)), "这条轨迹应当有 tick 内部的切换"
    s, _ = frames()(tid)
    if system == "double_well":
        f = simulation.vector_field
    elif system == "pendulum":
        f = lambda q, uu: np.array([q[1], -9.81*math.sin(q[0])-.2*q[1]+4.*(uu[0]*math.cos(q[0])+uu[1]*math.sin(q[0]))])
    else:
        from po_jepa import second_order
        cp = second_order.SYSTEMS["cart_pole"]
        f = lambda q, uu: cp.field(q, uu, 1., .3, .5)
    for k in (0, 5, 17, 31):
        # 门槛 1e-8：生产积分器本身（pendulum 每 tick 50 步、cart_pole 100 步 RK4）与细步差约 1e-10–1e-9；切换时刻错 1/4 tick 差约 1e-2
        assert np.abs(s[k+1]-_fine_subtick(f, s[k], u4, k, 256)).max() < 1e-8
    # 与不挪边界的轨迹确实不同（同一组取值，切换时刻不同）
    use(system, "pixel_noise=0.5")
    s0, _ = frames()(tid)
    assert np.abs(s-s0).max() > 1e-3 and np.array_equal(s[0], s0[0])


# —— 曝光 ——

def test_exposure_sample_times():
    v = variants.parse("double_well", "fps=4,exposure=0.5")
    t = np.array([0., 1., 2.5])
    got = timing.exposure_times(t, v, 0.)
    assert np.array_equal(got, [[0., 0., 0., 0., 0.], [1., .875, .75, .625, .5], [2.5, 2.375, 2.25, 2.125, 2.]])
    v = variants.parse("double_well", "fps=2,exposure=1")       # E = 2 tick → 17 个时刻，间隔 1/8
    got = timing.exposure_times(np.array([4.]), v, -1.)
    assert got.shape == (1, 17) and got[0, 0] == 4. and got[0, -1] == 2. and np.allclose(np.diff(got[0]), -.125)
    v = variants.parse("double_well", "fps=16,exposure=0.3")    # E = 0.075 tick → M = 1+⌈0.6⌉ = 2
    got = timing.exposure_times(np.array([1.]), v, 0.)
    assert got.shape == (1, 2) and got[0, 1] == 1.-.075
    assert timing.exposure_times(t, variants.parse("double_well", "fps=8"), 0.).shape == (3, 1)


@pytest.mark.parametrize("system,knobs", [("van_der_pol", "movers=1"), ("pendulum", ""), ("advection_diffusion", "gain=2")])
def test_exposure_is_the_mean_of_sharp_frames(use, system, knobs):
    """fps=4、exposure=0.5：第 k 帧 = t = k, k−1/8, …, k−1/2 五个时刻原始画面的平均（取整前）。其中 k、k−1/4、k−1/2 三个时刻
    的清晰画面独立地来自 fps=16 的数据集。"""
    use(system, join(knobs, "fps=16"))
    s16, f16 = frames()(8)
    use(system, join(knobs, "exposure=0.5"))
    s, f = frames()(8)
    v = datasets.active().knobs
    sc = timing._scene(systems.get(), 8, v, 32)
    for k in (3, 20):
        ts = np.array([k, k-.125, k-.25, k-.375, k-.5])
        st = timing.simulate(sc, u4_of(8), ts)
        with systems.raw_images():
            raw = sc.draw(st, ts)
        for j, m in ((0, 0), (2, 1), (4, 2)):
            assert np.array_equal(np.rint(np.clip(raw[j], 0, 255)).astype(np.uint8), f16[4*k-m])
        assert np.array_equal(f[k], np.rint(np.clip(raw.mean(0), 0, 255)).astype(np.uint8))
        assert not np.array_equal(f[k], f16[4*k])        # 确实模糊了
        assert np.array_equal(s[k], s16[4*k])            # 隐藏状态是帧时刻（曝光窗口的末端）的状态
    # 一阶场景第 0 帧：窗口早于起点的部分取起点，没有会动的干扰光点时等于清晰的第 0 帧
    if system != "van_der_pol" and system not in variants.SECOND_ORDER:
        assert np.array_equal(f[0], f16[0])


def test_exposure_is_causal(use):
    """画面只用 t_k 及以前：把 t_k 以后的执行动作全换掉，第 0…k 帧不变。"""
    use("double_well", "exposure=1,fps=8")
    v = datasets.active().knobs
    sc = timing._scene(systems.get(), 4, v, 32)
    times = timing.frame_times(4)
    samples = timing.exposure_times(times, v, 0.)
    assert np.all(samples <= times[:, None]) and np.all(samples >= 0.)
    u = u4_of(4)
    u2 = u.copy(); u2[4*10:] = -u2[4*10:]
    a = timing.simulate(sc, u, samples[:21].ravel()); b = timing.simulate(sc, u2, samples[:21].ravel())
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# —— 像素噪声、二阶场景的前一帧 ——

def test_offgrid_pixel_noise(use):
    """tick 内部的帧：流 11 按帧号另起（独立构造熵）；整 tick 的帧用流 4（上面的逐位相同已核）。"""
    use("spring_chain", "fps=8,pixel_noise=8")
    s, f = frames()(3)
    use("spring_chain", "fps=8")
    _, clean = frames()(3)
    sc = timing._scene(systems.get(), 3, datasets.active().knobs, 32)
    st = timing.simulate(sc, u4_of(3), [2.5])
    with systems.raw_images():
        raw = sc.draw(st, np.array([2.5]))[0]
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1703, 1011, 3, 11, 6])))   # 帧号 5 → 键 6
    assert np.array_equal(f[5], np.rint(np.clip(raw+g.normal(0., 8., (256, 256, 3)), 0, 255)).astype(np.uint8))
    assert not np.array_equal(f[5], clean[5])
    # 前一帧（帧号 −1，键 0）与 frames() 各帧的噪声互不相同、frames_with_prev()[1:] 与 frames() 相同
    use("spring_chain", "fps=8,pixel_noise=8")
    sp, fp = systems.get().frames_with_prev(3)
    assert np.array_equal(fp[1:], f) and np.array_equal(sp[1:], s)


def test_noise_blocks_equal_one_big_draw():
    g1 = np.random.Generator(np.random.PCG64(5)); g2 = np.random.Generator(np.random.PCG64(5))
    big = g1.normal(0., 3., (6, 256, 256, 3))
    blocks = timing._NoiseBlocks(g2, 3.)
    for i in (0, 2, 3, 5):
        assert np.array_equal(blocks.block(i), big[i])
    with pytest.raises(ValueError):
        blocks.block(1)


def test_second_order_previous_frame(use):
    # fps=4 加抖动：前一帧仍在 t = −1，就是原来 frames_with_prev 的第 0 帧（流 4 的第 0 份噪声）
    use("cart_pole", "pixel_noise=4")
    sp, fp = systems.get().frames_with_prev(5)
    use("cart_pole", "pixel_noise=4,jitter=0.2")
    sj, fj = systems.get().frames_with_prev(5)
    assert np.array_equal(fj[0], fp[0]) and np.array_equal(sj[0], sp[0]) and fj.shape == fp.shape
    # fps=16：前一帧在 t = −1/4，是预跑（零动作）那个 tick 的 3/4 处：与独立细步 RK4 一致
    use("pendulum", "fps=16")
    sp, fp = systems.get().frames_with_prev(5)
    from po_jepa import second_order
    q_prev, _ = second_order.SYSTEMS["pendulum"].start(5, {})
    f = lambda q, uu: np.array([q[1], -9.81*math.sin(q[0])-.2*q[1]])
    assert np.abs(sp[0]-_fine_rk4(f, q_prev, np.zeros(2), .75*.25)).max() < 1e-8
    assert fp.shape == (130, 256, 256, 3)
    use("pendulum", "fps=2")
    systems.get().frames(5)
    with pytest.raises(ValueError, match="预跑"):
        systems.get().frames_with_prev(5)


def test_horizon_prefix_with_time_sampling(use):
    use("push", "fps=8,exposure=0.5,movers=1,pixel_noise=3")
    s32, f32 = frames()(9)
    use("push", "fps=8,exposure=0.5,movers=1,pixel_noise=3,horizon=64")
    s64, f64 = frames()(9)
    assert f64.shape[0] == 129 and np.array_equal(f64[:65], f32) and np.array_equal(s64[:65], s32)


# —— 不接入正式流程 ——

def _clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")}
    env["PYTHONPATH"] = str(SRC)
    return env


def test_encode_probe_and_plan_refuse_time_sampling(tmp_path):
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/encode_v7.py"), "encode", "--system", "double_well", "--variant", "fps=8",
                        "--trajectories", "40", "--fake-encoder", "--device", "cpu", "--output", str(tmp_path/"enc")],
                       env=_clean_env(), capture_output=True, text=True)
    assert r.returncode != 0 and "时间采样" in r.stderr and not (tmp_path/"enc").exists()
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/probe_actions.py"), "--system", "push", "--variant", "switch_offset=0.5",
                        "--trajectories", "40", "--fake-encoder", "--device", "cpu", "--output", str(tmp_path/"p.json")],
                       env=_clean_env(), capture_output=True, text=True)
    assert r.returncode != 0 and "时间采样" in r.stderr and not (tmp_path/"p.json").exists()
    job = tmp_path/"job"; job.mkdir()
    (job/"job.json").write_text(json.dumps(dict(seed=0, updates=1, dataset="double_well~exposure=0.5-40", condition="identity")))
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/plan_v7.py"), "--job", str(job), "--fake-encoder",
                        "--device", "cpu", "--output", str(tmp_path/"plan.json")], env=_clean_env(), capture_output=True, text=True)
    assert r.returncode != 0 and "时间采样" in r.stderr and not (tmp_path/"plan.json").exists()
