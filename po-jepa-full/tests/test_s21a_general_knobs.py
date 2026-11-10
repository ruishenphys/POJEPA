"""S21a 通用调节项：process_noise、saturation、occluders、movers、policy，以及 DATASET.md 的覆盖表。

基础场景与已有调节项逐位不变由跨树对照核（docs/segments/S21a.md §4.2）；这里核新调节项本身的数学和接线。
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from po_jepa import datasets, new_systems, simulation, systems, variants
from po_jepa.variants import Variant

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def use():
    old = datasets.active()
    def set_(system, knobs="", count=200):
        spec = datasets.DatasetSpec(system, count, variants.format(system, variants.parse(system, knobs)))
        datasets._set_for_tests(spec)
        return spec
    yield set_
    datasets._set_for_tests(old)


def ticks_of(controls):
    u = np.zeros((32, 2))
    for a, b, value in zip(controls.starts_ticks, controls.ends_ticks, controls.values):
        u[a:b] = value
    return u


# —— 解析、范围、适用场景、覆盖表 ——

def test_defaults_parse_format_and_ranges():
    v = Variant()
    assert (v.process_noise, v.saturation, v.occluders, v.movers, v.policy) == (0., None, 0, 0, "random")
    assert v.is_default and not (v.changes_controls() or v.changes_execution() or v.changes_observation())
    v = variants.parse("double_well", "process_noise=0.2,saturation=0.30,occluders=2,movers=3,policy=mixed")
    assert variants.format("double_well", v) == "movers=3,occluders=2,policy=mixed,process_noise=0.2,saturation=0.3"
    assert variants.parse("double_well", variants.format("double_well", v)) == v
    assert v.changes_controls() and v.changes_execution() and v.changes_observation()
    assert variants.parse("double_well", "policy=goal").changes_controls()
    assert variants.parse("double_well", "saturation=2").changes_execution()
    for bad in ("process_noise=0.31", "process_noise=-0.1", "saturation=0.04", "saturation=2.1", "saturation=nan",
                "occluders=4", "movers=5", "policy=expert", "process_noise=nan"):
        with pytest.raises(ValueError):
            variants.parse("double_well", bad)
    # 名字不与任何场景的物理参数重名（否则 parse 会当成物理参数）
    new = {"process_noise", "saturation", "occluders", "movers", "policy"}
    assert not new & {n for p in variants.PARAMS.values() for n in p}


def test_where_each_new_knob_applies():
    cov = variants.coverage()
    every = datasets.SYSTEMS
    assert cov["saturation"] == every and cov["occluders"] == every
    # S23b 的二阶场景：光点类（pendulum、bouncing_balls、cart_pole、spring_chain）接受干扰光点，wave 是场；policy、
    # process_noise 都不接受（tests/test_s23b_register.py 核原因）
    assert cov["movers"] == ("double_well", "van_der_pol", "multi_particle", "landscape", "push",
                             "pendulum", "bouncing_balls", "cart_pole", "spring_chain") == cov["distractors"]
    assert cov["policy"] == ("double_well", "van_der_pol", "multi_particle", "landscape", "push")
    assert cov["process_noise"] == ("double_well", "van_der_pol", "multi_particle", "landscape")
    for system, knob in (("advection_diffusion", "movers=1"), ("gray_scott", "movers=1"), ("gray_scott", "policy=goal"),
                         ("advection_diffusion", "policy=mixed"), ("push", "process_noise=0.1"),
                         ("gray_scott", "process_noise=0.1"), ("advection_diffusion", "process_noise=0.1")):
        with pytest.raises(ValueError):
            variants.parse(system, knob)
    # 每个 Variant 字段（除物理参数和 landscape 的 shape）都在覆盖表里，分组不重不漏
    fields = set(Variant.__dataclass_fields__) - {"params", "shape"}
    grouped = [n for names in variants.KNOB_GROUPS.values() for n in names]
    assert set(variants.GENERAL_SAMPLES) == fields == set(grouped) and len(grouped) == len(set(grouped))


def test_dataset_md_coverage_table_is_generated_from_code():
    spec = importlib.util.spec_from_file_location("knob_coverage", ROOT/"tools"/"knob_coverage.py")
    tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    text = (ROOT/"DATASET.md").read_text(encoding="utf-8")
    assert tool.render(text) == text, "运行 python tools/knob_coverage.py --write"
    assert "| 观察 | `movers` | ✓ | ✓ | ✓ | — | ✓ | — | ✓ | ✓ | ✓ | — | ✓ | ✓ | 9 |" in text


# —— saturation ——

def test_saturation_clips_executed_actions_after_gain_and_noise(use):
    use("double_well", "gain=4,noise=0.3,saturation=0.6")
    v = datasets.active().knobs
    c = systems.commanded_controls(11)
    u = systems.applied_ticks(11, v, c)
    raw = systems.applied_ticks(11, variants.parse("double_well", "gain=4,noise=0.3"), c)
    assert np.array_equal(u, np.clip(raw, -.6, .6)) and np.abs(raw).max() > .6 and np.abs(u).max() == .6
    # 下达动作（模型看到的）不变
    assert np.array_equal(ticks_of(c), ticks_of(simulation.sample_controls(11)))
    # 新场景也经 applied_ticks
    use("push", "saturation=0.1")
    impl, u, _ = systems.get()._impl(3)
    assert np.abs(u).max() <= .1 and np.array_equal(u, np.clip(ticks_of(simulation.sample_controls(3)), -.1, .1))


# —— process_noise ——

def test_process_noise_is_additive_euler_maruyama_on_rk4_drift(use, monkeypatch):
    """漂移为 0 时，终点 = 初值 + σ√h·Σξ（ξ 来自随机流 6，按绝对 tick 取）。"""
    use("double_well", "process_noise=0.3")
    v = datasets.active().knobs
    sysm = systems.get()
    monkeypatch.setattr(type(sysm), "vfield", lambda self, q, u, v: np.zeros_like(q))
    monkeypatch.setattr(type(sysm), "WALL", None)
    states = sysm.variant_trajectory(5, v)
    xi = simulation._rng(5, 6).standard_normal((32, 128, 2))
    h = .25/128
    increments = .3*math.sqrt(h)*xi.sum(1)                       # [32,2] 每个 tick 的增量
    assert np.allclose(np.diff(states, axis=0), increments, rtol=0, atol=1e-12)
    assert np.allclose(states[-1]-states[0], increments.sum(0), rtol=0, atol=1e-12)
    # 方差：每 tick 增量的标准差 = σ√0.25（64 个独立样本，宽松的统计界）
    assert abs(increments.std()/(.3*.5)-1) < .3


def test_process_noise_counterfactual_reuses_the_same_noise(use):
    use("van_der_pol", "process_noise=0.2")
    v = datasets.active().knobs; sysm = systems.get()
    states = sysm.variant_trajectory(9, v)
    for start, k in ((0, 32), (5, 8), (20, 12)):
        again = sysm.variant_trajectory(9, v, controls=systems.commanded_controls(9), start=start, state=states[start], ticks=k)
        assert np.array_equal(again, states[start:start+k+1])
    use("van_der_pol")
    clean = systems.get().variant_trajectory(9, datasets.active().knobs)
    d = np.abs(states-clean).max()
    assert 1e-3 < d < 2., d
    # 初值不变（配对），噪声不影响下达动作
    assert np.array_equal(states[0], clean[0])


def test_process_noise_keeps_multi_particle_dimension(use):
    use("multi_particle", "n=5,process_noise=0.3")
    s, f = systems.get().frames(4)
    assert s.shape == (33, 10) and np.isfinite(s).all() and f.shape == (33, 256, 256, 3)


# —— occluders ——

def _occluder_mask(trajectory_id, count, rng):
    mask = np.zeros((256, 256), bool)
    for r, c in rng.integers(16, 256-16-48+1, (count, 2)):
        mask[r:r+48, c:c+48] = True
    return mask


# 基线也带一个观察调节项：advection_diffusion 的基础场景走 v6 原实现，和可调实现差浮点舍入（S21a 之前就如此）
@pytest.mark.parametrize("system,base", [("double_well", "color=white,"), ("advection_diffusion", "color=white,"),
                                         ("landscape", "shape=ring,"), ("push", "")])
def test_occluders_only_cover_their_squares(use, system, base):
    use(system, base.rstrip(","))
    s0, f0 = systems.get().frames(6)
    use(system, base+"occluders=3")
    s1, f1 = systems.get().frames(6)
    g = new_systems.rng(system, 6, 7) if system in new_systems.SYSTEM_INDEX else simulation._rng(6, 7)
    mask = _occluder_mask(6, 3, g)
    assert np.array_equal(s0, s1)
    assert np.array_equal(f1[:, ~mask], f0[:, ~mask])
    assert (f1[:, mask] == systems.OCCLUDER_COLOR.astype(np.uint8)).all() and mask.sum() >= 48*48


def test_occluders_are_under_pixel_noise(use):
    """遮挡块是场景里的物体，相机的像素噪声也加在它上面（先画遮挡块，再加像素噪声）。"""
    use("double_well", "occluders=1,pixel_noise=10")
    f = systems.get().frames(6)[1]
    mask = _occluder_mask(6, 1, simulation._rng(6, 7))
    assert f[:, mask].std() > 5


def test_occluders_on_gray_scott_go_through_finish():
    """gray_scott 的渲染很慢：直接用小场核 _finish 的遮挡（与 push 同一个函数）。"""
    v = variants.parse("gray_scott", "occluders=2")
    image = np.full((2, 256, 256, 3), 30.)
    out = new_systems._finish(image, 4, "gray_scott", v, ticks=[0, 1])   # S24-fix1：部分帧要给 ticks
    mask = _occluder_mask(4, 2, new_systems.rng("gray_scott", 4, 7))
    assert (out[:, mask] == systems.OCCLUDER_COLOR.astype(np.uint8)).all() and (out[:, ~mask] == 30).all()


# —— movers ——

def test_mover_motion_is_a_fixed_rotation_about_the_center():
    g = np.random.default_rng(0)
    c = systems.mover_centers(g, 3, np.arange(33))
    z = (c[..., 1]-128)+1j*(128-c[..., 0])                       # 复平面，y 向上
    assert np.allclose(z[1:]/z[:-1], np.exp(1j*.6*.25))            # 协议值 0.6 rad/s、0.25 s 一帧（不用被测常量）
    r = np.abs(z)
    assert np.allclose(r, r[:1]) and (r >= 40).all() and (r <= 110).all()


@pytest.mark.parametrize("system,base", [("double_well", "background=plain,"), ("push", "background=plain,")])
def test_movers_are_drawn_and_partial_frames_need_ticks(use, system, base):
    use(system, base.rstrip(","))
    s0, f0 = systems.get().frames(2)
    use(system, base+"movers=2")
    sysm = systems.get(); v = datasets.active().knobs
    s1, f1 = sysm.frames(2)
    assert np.array_equal(s0, s1) and not np.array_equal(f0, f1)
    g = new_systems.rng(system, 2, 8) if system == "push" else simulation._rng(2, 8)
    centers = systems.mover_centers(g, 2, np.arange(33))
    # 光点中心像素被拉向灰色（离受控物体远的帧）
    changed = [np.abs(f1[t, int(r), int(c)].astype(int)-f0[t, int(r), int(c)]).sum() > 30
               for t in range(33) for r, c in centers[t] if 8 <= r < 248 and 8 <= c < 248]
    assert np.mean(changed) > .8
    # 单帧重画（反事实、规划用）要给 tick，给了就和整条轨迹的那一帧逐位相同
    if system == "push":
        impl = new_systems.SYSTEMS["push"]
        one = lambda k, t: impl.draw(s1[k:k+1], 2, dict(v.params), v, ticks=t)
    else:
        one = lambda k, t: sysm.draw(s1[k:k+1], 2, v, ticks=t)
    for k in (0, 7, 32):
        assert np.array_equal(one(k, [k])[0], f1[k])
    with pytest.raises(ValueError):
        one(7, None)


# —— policy ——

def test_policy_keeps_segments_and_zero_segments(use):
    for system, base in (("double_well", ""), ("landscape", "shape=ring,"), ("push", "")):
        use(system, base+"zero=0.4")
        r = systems.commanded_controls(13)
        for policy in ("goal", "mixed"):
            use(system, base+f"zero=0.4,policy={policy}")
            p = systems.commanded_controls(13)
            assert np.array_equal(p.starts_ticks, r.starts_ticks) and np.array_equal(p.ends_ticks, r.ends_ticks)
            zero = ~r.values.any(1)
            assert zero.any() and not p.values[zero].any() and np.abs(p.values).max() <= .5


def test_goal_policy_matches_open_loop_dead_reckoning(use, monkeypatch):
    """独立重算：p̂ 从初值出发，非零段 u = clip((目标 − p̂)/K + ε)，每段后 p̂ += K·u·段长·0.25；不跑仿真。"""
    def no_sim(*a, **k): raise AssertionError("采集策略不应跑仿真")
    for system, base in (("multi_particle", "n=4,axes=x,"), ("push", ""), ("van_der_pol", "")):
        use(system, base+"policy=goal")
        v = datasets.active().knobs
        monkeypatch.setattr(systems.System, "variant_trajectory", no_sim)
        monkeypatch.setattr(new_systems.Push, "trajectory", no_sim)
        c = systems.commanded_controls(21)
        monkeypatch.undo()
        r = simulation.sample_controls(21) if base == "" else None
        g9 = simulation._rng(21, 9)
        if system == "push":
            q0 = new_systems.SYSTEMS["push"].initial(21, {})
            p, d = q0[:2].copy(), q0[2:4]-q0[:2]
            target, K = q0[2:4]+.6*d/np.linalg.norm(d), 2.
        else:
            p = systems.get().vinitial(21, v)[:2].copy(); target, K = g9.uniform(-1.5, 1.5, 2), 1.
        for a, b, value in zip(c.starts_ticks, c.ends_ticks, c.values):
            if value.any():
                g9.random(); eps = g9.normal(0., .1, 2)
                want = np.clip(target-p+eps, -.5, .5) if K == 1. else np.clip((target-p)/K+eps, -.5, .5)
                if "axes=x" in base: want[1] = 0.
                assert np.allclose(value, want, atol=1e-12), system
            p = p+K*value*(b-a)*.25
        if r is not None:
            assert np.array_equal(c.starts_ticks, r.starts_ticks)


def test_mixed_policy_uses_both_kinds_of_segments(use):
    use("double_well", "policy=mixed,segments=1,zero=0")
    m = systems.commanded_controls(3).values
    use("double_well", "segments=1,zero=0")
    r = systems.commanded_controls(3).values
    same = (m == r).all(1)
    assert 5 < same.sum() < 27            # 32 段里约一半保留随机值


def test_goal_policy_makes_push_move_the_block(use):
    moved = {}
    for policy in ("random", "goal"):
        use("push", "" if policy == "random" else "policy=goal")
        sysm = systems.get()
        moved[policy] = sum(np.linalg.norm(np.diff(sysm.reference_trajectory(t)[[0, -1], 2:4], axis=0)) > .05
                            for t in range(10))
    assert moved["goal"] >= 9 and moved["goal"] > moved["random"], moved
