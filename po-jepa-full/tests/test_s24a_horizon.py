"""S24a 长时程：调节项 horizon ∈ {32, 64, 128}（docs/segments/S24a.md）。

核的是：解析与覆盖表；长时程的下达动作（LongControls）和它的检查；前缀一致（horizon=H 的前 32 个 tick 的动作、
执行动作、隐藏状态和前 33 帧画面与 horizon=32、其余调节项相同的数据集逐位相同，12 个场景都核，含 v7 场景的基础场景）；
续算和部分帧重画到 tick 32 以后与整条轨迹逐位相同；32 tick 的动作和长时程数据集不能混用；编码、探针、规划拒绝长时程。
写死的数值（帧数、tick 数、段长）不调被测函数推出来（错误清单 2026-11-09 S23a 那条）。
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from po_jepa import datasets, second_order, simulation, systems, variants
from po_jepa.variants import Variant

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"src"
ALL = ("double_well", "van_der_pol", "multi_particle", "advection_diffusion", "landscape", "gray_scott", "push",
       "pendulum", "bouncing_balls", "wave", "cart_pole", "spring_chain")


@pytest.fixture
def use():
    old = datasets.active()
    def set_(system, knobs="", count=200):
        spec = datasets.DatasetSpec(system, count, variants.format(system, variants.parse(system, knobs)))
        datasets._set_for_tests(spec)
        return spec
    yield set_
    datasets._set_for_tests(old)


def per_tick(controls, T):
    """分段动作 → 逐 tick [T,2]（独立于 systems.applied_ticks 的展开）。"""
    u = np.full((T, 2), np.nan)
    for a, b, value in zip(controls.starts_ticks, controls.ends_ticks, controls.values):
        u[a:b] = value
    assert np.isfinite(u).all()
    return u


def join(*parts):
    return ",".join(p for p in parts if p)


# —— 解析、格式、覆盖表 ——

def test_parse_format_and_validate():
    v = variants.parse("double_well", "horizon=128")
    assert v.horizon == 128 and type(v.horizon) is int and variants.format("double_well", v) == "horizon=128"
    assert Variant().horizon == 32 and variants.HORIZONS == (32, 64, 128)
    # 默认值不写进场景串：已有数据集的名字、标签一个字不变
    assert variants.format("double_well", variants.parse("double_well", "horizon=32")) == ""
    assert datasets.DatasetSpec("push", 2000, "horizon=128").tag == "push~horizon=128-2000"
    assert datasets.parse("van_der_pol~gain=2,horizon=64:2000").knobs.horizon == 64
    for bad in ("horizon=48", "horizon=0", "horizon=256", "horizon=33", "horizon=1e2", "horizon=abc"):
        with pytest.raises(ValueError):
            variants.parse("double_well", bad)
    with pytest.raises(ValueError):
        variants.validate("double_well", Variant(horizon=128.))        # 必须是整数
    for system in ALL:
        text = ("shape=ring," if system == "landscape" else "") + "horizon=128"
        assert variants.parse(system, text).horizon == 128


def test_coverage_lists_horizon_for_every_scene():
    assert variants.coverage()["horizon"] == datasets.SYSTEMS
    assert variants.KNOB_GROUPS["时间"] == ("horizon",)
    text = (ROOT/"DATASET.md").read_text(encoding="utf-8")
    assert "| 时间 | `horizon` | " + " | ".join(["✓"]*12) + " | 12 |" in text


# —— 下达动作 ——

@pytest.mark.parametrize("system,knobs", [("double_well", ""), ("van_der_pol", "segments=2"), ("multi_particle", "zero=0.5,axes=x"),
                                          ("landscape", "shape=ring,policy=goal"), ("push", "policy=mixed"),
                                          ("pendulum", "segments=8"), ("gray_scott", "")])
def test_commanded_controls_extend_the_32_tick_controls(use, system, knobs):
    use(system, knobs)
    short = systems.commanded_controls(5)
    if knobs in ("", "segments=8") or system == "gray_scott":
        assert type(short) is simulation.TrajectoryControls
    for H in (64, 128):
        use(system, join(knobs, f"horizon={H}"))
        long = systems.commanded_controls(5)
        assert type(long) is systems.LongControls and long.horizon == H and long.ends_ticks[-1] == H
        assert not isinstance(long, simulation.TrajectoryControls)
        # 逐 tick 的前 32 个相同；跨过 tick 32 之前的段（起止和值）完全相同
        assert np.array_equal(per_tick(long, H)[:32], per_tick(short, 32))
        k = int(np.searchsorted(short.ends_ticks, 32))            # short 的最后一段（被截在 32）
        assert np.array_equal(long.starts_ticks[:k+1], short.starts_ticks)
        assert np.array_equal(long.ends_ticks[:k], short.ends_ticks[:k]) and np.array_equal(long.values[:k+1], short.values)
        d = np.diff(np.concatenate(([0], long.ends_ticks)))
        assert np.isin(d[:-1], [1, 2, 4, 8]).all() and 1 <= d[-1] <= 8


def test_default_knobs_reuse_v6_controls_stream(use):
    """只设 horizon 时走通用循环：前 32 个 tick 必须和 v6 的 simulation.sample_controls 一样（基础场景的动作）。"""
    use("double_well", "horizon=128", count=2000)
    for tid in (0, 1, 1800, 1999):
        long = systems.commanded_controls(tid)
        use("double_well", "", count=2000)
        assert np.array_equal(per_tick(long, 128)[:32], per_tick(simulation.sample_controls(tid), 32))
        use("double_well", "horizon=128", count=2000)


def test_goal_policy_dead_reckoning_runs_past_tick_32(use):
    """policy=goal 的开环推算跨过 tick 32 不截断：独立重算（流 9：目标 2 个数，每个非零段 coin、探索噪声 2 个数）。"""
    use("double_well", "policy=goal,horizon=128")
    c = systems.commanded_controls(12)
    g9 = simulation._rng(12, 9)
    estimate, target = simulation.initial_state(12)[:2].copy(), g9.uniform(-1.5, 1.5, 2)
    goal_segments = 0
    for a, b, value in zip(c.starts_ticks, c.ends_ticks, c.values):
        if value.any():
            g9.random(); explore = g9.normal(0., .1, 2)
            assert np.array_equal(value, np.clip(target-estimate+explore, -.5, .5))
            goal_segments += a >= 32
        estimate = estimate+value*(b-a)*.25
    assert c.ends_ticks[-1] == 128 and goal_segments > 3


def test_long_controls_checks(use):
    use("double_well", "horizon=128")
    c = systems.commanded_controls(2)
    ok = dict(trajectory_id=2, starts_ticks=c.starts_ticks, ends_ticks=c.ends_ticks, values=c.values)
    systems.LongControls(**ok)
    s, e, v = c.starts_ticks, c.ends_ticks, c.values
    bad = [dict(ok, ends_ticks=np.concatenate((e[:-1], [e[-1]-1]))),                               # 不到 128
           dict(ok, starts_ticks=s.astype(np.int32)),
           dict(ok, values=np.concatenate((v[:-1], [[.6, 0.]]))),
           dict(ok, values=np.concatenate((v[:-1], [[np.nan, 0.]]))),
           dict(ok, values=v.astype(np.float32)),
           dict(ok, trajectory_id=200),
           dict(ok, starts_ticks=s+np.r_[0, np.ones(len(s)-1, dtype=np.int64)])]                  # 段间有缝
    for kw in bad:
        with pytest.raises(ValueError):
            systems.LongControls(**kw)
    # 32 tick 的动作不能写成 LongControls；不在 HORIZONS 里的长度也不行；段长只能 1/2/4/8（最后一段 ≤ 8）
    with pytest.raises(ValueError):
        systems.LongControls(2, np.arange(0, 32, 8, dtype=np.int64), np.arange(8, 40, 8, dtype=np.int64), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        systems.LongControls(2, np.arange(0, 96, 8, dtype=np.int64), np.arange(8, 104, 8, dtype=np.int64), np.zeros((12, 2)))
    with pytest.raises(ValueError):
        systems.LongControls(2, np.arange(0, 128, 16, dtype=np.int64), np.arange(16, 144, 16, dtype=np.int64), np.zeros((8, 2)))
    systems.LongControls(2, np.arange(0, 64, 8, dtype=np.int64), np.arange(8, 72, 8, dtype=np.int64), np.zeros((8, 2)))
    # v7 的 TrajectoryControls 不收 128 tick（simulation.py 没改）
    with pytest.raises(ValueError):
        simulation.TrajectoryControls(2, c.starts_ticks, c.ends_ticks, c.values)


def test_applied_ticks_prefix_and_no_mixing(use):
    knobs = "gain=2,noise=0.2,delay=2,saturation=0.7"
    use("van_der_pol", knobs)
    v32 = datasets.active().knobs; c32 = systems.commanded_controls(9); u32 = systems.applied_ticks(9, v32, c32)
    use("van_der_pol", join(knobs, "horizon=128"))
    v128 = datasets.active().knobs; c128 = systems.commanded_controls(9); u128 = systems.applied_ticks(9, v128, c128)
    assert u32.shape == (32, 2) and u128.shape == (128, 2) and np.array_equal(u128[:32], u32)
    # 独立重算：下达动作逐 tick 展开 → 延迟 2 tick（先补零）→ ×2 → 加执行噪声（流 5，[128,2] 一次抽）→ 截断到 ±0.7
    shifted = np.concatenate((np.zeros((2, 2)), per_tick(c128, 128)[:-2]))
    expected = np.clip(2*shifted+simulation._rng(9, 5).normal(0., .2, (128, 2)), -.7, .7)
    assert np.array_equal(u128, expected) and np.abs(u128[32:]).max() > 0
    with pytest.raises(ValueError):
        systems.applied_ticks(9, v128, c32)
    with pytest.raises(ValueError):
        systems.applied_ticks(9, v32, c128)
    with pytest.raises(ValueError):
        systems.applied_ticks(9, v128, object())


# —— 前缀一致：12 个场景 ——

# 每个场景一组会用到各条随机流（执行噪声、像素噪声、会动的干扰光点、遮挡块、过程噪声）的调节项
KNOBS = {"double_well": "noise=0.1,pixel_noise=8,movers=2,occluders=1,process_noise=0.1",
         "van_der_pol": "noise=0.1,pixel_noise=8,movers=1",
         "multi_particle": "process_noise=0.05,pixel_noise=8,distractors=2",
         "advection_diffusion": "noise=0.1,pixel_noise=8,occluders=1",
         "landscape": "shape=periodic,noise=0.1,movers=2,pixel_noise=8",
         "gray_scott": "noise=0.1,pixel_noise=8",
         "push": "noise=0.1,movers=1,pixel_noise=8,occluders=1",
         "pendulum": "noise=0.1,movers=1,pixel_noise=8",
         "bouncing_balls": "noise=0.1,pixel_noise=8,occluders=1",
         "wave": "noise=0.1,pixel_noise=8",
         "cart_pole": "noise=0.1,movers=1,pixel_noise=8",
         "spring_chain": "noise=0.1,pixel_noise=8,distractors=1"}


@pytest.mark.parametrize("system", ALL)
def test_first_33_frames_equal_the_32_tick_dataset(use, system):
    tid = 1801 if system != "gray_scott" else 1800
    use(system, KNOBS[system], count=2000)
    s, f = systems.get().frames(tid)
    use(system, join(KNOBS[system], "horizon=64"), count=2000)
    sl, fl = systems.get().frames(tid)
    assert fl.shape == (65, 256, 256, 3) and fl.dtype == np.uint8 and sl.shape[0] == 65
    assert np.array_equal(fl[:33], f) and np.array_equal(sl[:33], s)
    assert not np.array_equal(fl[33], fl[32]) or system == "gray_scott"      # 后面的帧确实接着动
    if system in variants.SECOND_ORDER:
        sp, fp = systems.get().frames_with_prev(tid)
        assert fp.shape == (66, 256, 256, 3) and np.array_equal(fp[1:], fl) and np.array_equal(sp[1:], sl)


@pytest.mark.parametrize("system", ("double_well", "van_der_pol", "multi_particle", "advection_diffusion"))
def test_base_scene_prefix_against_the_v6_reference_path(use, system):
    """基础场景（调节项为空）走 v6/v7 原实现，horizon=128 走可调实现：前 33 帧画面逐位相同。
    隐藏状态：粒子类逐位相同；advection_diffusion 两条路径的浮点运算顺序不同，差 ≤ 1e-12（画面逐位相同）。"""
    use(system, "", count=2000)
    s, f = systems.get().frames(1803)
    use(system, "horizon=128", count=2000)
    sl, fl = systems.get().frames(1803)
    assert fl.shape[0] == 129 and np.array_equal(fl[:33], f)
    if system == "advection_diffusion":
        assert np.abs(sl[:33]-s).max() <= 1e-12
    else:
        assert np.array_equal(sl[:33], s)


# —— 续算、部分帧重画 ——

@pytest.mark.parametrize("system,knobs", [("double_well", "process_noise=0.1"), ("push", "noise=0.1"), ("pendulum", ""),
                                          ("advection_diffusion", "noise=0.1")])
def test_restart_after_tick_32_matches_the_full_trajectory(use, system, knobs):
    use(system, join(knobs, "horizon=64"))
    s = systems.get(); v = datasets.active().knobs
    if system == "advection_diffusion":
        full = s.variant_fields(4, v)
        part = s.variant_fields(4, v, start=40, field=full[40], ticks=20)
    else:
        full = s.variant_trajectory(4, v)
        part = s.variant_trajectory(4, v, start=40, state=full[40], ticks=20)
    if system == "advection_diffusion":
        # 续算从实数场重新做 FFT（S21b 的接口），与整条轨迹里一直留在谱空间的推进差一次往返舍入
        assert full.shape[0] == 65 and np.abs(part-full[40:61]).max() <= 1e-12
    else:
        assert full.shape[0] == 65 and np.array_equal(part, full[40:61])
    if system in ("push", "pendulum"):
        with pytest.raises(ValueError):
            s.variant_trajectory(4, v, start=40, state=full[40], ticks=25)        # 超过 64


@pytest.mark.parametrize("system,knobs", [("double_well", "movers=2,pixel_noise=8"), ("push", "movers=1,pixel_noise=8"),
                                          ("cart_pole", "movers=1,pixel_noise=8"), ("advection_diffusion", "pixel_noise=8")])
def test_partial_redraw_after_tick_32(use, system, knobs):
    use(system, join(knobs, "horizon=64"))
    s = systems.get(); v = datasets.active().knobs
    states, frames = s.frames(6)
    ticks = np.array([0, 33, 50, 64])
    if system == "advection_diffusion":
        fields = s.variant_fields(6, v)
        part = s.draw_fields(fields[ticks], 6, v, ticks=ticks)
    else:
        full = s.variant_trajectory(6, v)
        part = s.draw(full[ticks], 6, v, ticks=ticks)
        assert np.array_equal(s.draw(full, 6, v), frames)          # 整条 65 帧不给 ticks 时按 0…64 画
    assert np.array_equal(part, frames[ticks])
    # 部分帧不给 ticks、tick 超过 horizon 都报错
    with pytest.raises(ValueError):
        (s.draw_fields(fields[:33], 6, v) if system == "advection_diffusion" else s.draw(full[:33], 6, v))
    with pytest.raises(ValueError):
        (s.draw_fields(fields[:1], 6, v, ticks=np.array([65])) if system == "advection_diffusion"
         else s.draw(full[:1], 6, v, ticks=np.array([65])))


def test_pure_modules_accept_long_controls_only_from_horizons():
    """new_systems、second_order 收 [H,2]（H ∈ HORIZONS）的执行动作，别的长度报错；推进 H 个 tick。"""
    from po_jepa import new_systems
    for H in (32, 64, 128):
        assert new_systems._check_controls(np.zeros((H, 2))).shape == (H, 2)
        assert second_order._check_controls(np.zeros((H, 2))).shape == (H, 2)
    for bad in (np.zeros((33, 2)), np.zeros((100, 2)), np.zeros((64, 3)), np.zeros(64), np.full((64, 2), np.nan)):
        with pytest.raises(ValueError):
            new_systems._check_controls(bad)
        with pytest.raises(ValueError):
            second_order._check_controls(bad)
    u = np.zeros((64, 2)); u[:, 0] = .3
    q = second_order.SYSTEMS["spring_chain"].trajectory(3, u, {})
    assert q.shape[0] == 66
    q = new_systems.SYSTEMS["push"].trajectory(3, u, {})
    assert q.shape[0] == 65
    assert second_order.frame_ticks(66, None, 64).tolist() == list(range(-1, 65))
    with pytest.raises(ValueError):
        second_order.frame_ticks(34, None, 64)


# —— 入口拒绝长时程 ——

def _clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")}
    env["PYTHONPATH"] = str(SRC)
    return env


def test_encode_probe_and_plan_refuse_long_horizon(tmp_path):
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/encode_v7.py"), "encode", "--system", "double_well", "--variant", "horizon=128",
                        "--trajectories", "40", "--fake-encoder", "--device", "cpu", "--output", str(tmp_path/"enc")],
                       env=_clean_env(), capture_output=True, text=True)
    assert r.returncode != 0 and "长时程" in r.stderr and not (tmp_path/"enc").exists()
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/probe_actions.py"), "--system", "push", "--variant", "horizon=64",
                        "--trajectories", "40", "--fake-encoder", "--device", "cpu", "--output", str(tmp_path/"p.json")],
                       env=_clean_env(), capture_output=True, text=True)
    assert r.returncode != 0 and "长时程" in r.stderr and not (tmp_path/"p.json").exists()
    job = tmp_path/"job"; job.mkdir()
    (job/"job.json").write_text(json.dumps(dict(seed=0, updates=1, dataset="double_well~horizon=128-40", condition="identity")))
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/plan_v7.py"), "--job", str(job), "--fake-encoder",
                        "--device", "cpu", "--output", str(tmp_path/"plan.json")], env=_clean_env(), capture_output=True, text=True)
    assert r.returncode != 0 and "长时程" in r.stderr and not (tmp_path/"plan.json").exists()
