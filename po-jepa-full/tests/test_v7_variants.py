"""v7 可调数据集：调节项解析、基础场景逐位不变、各调节项确实起作用、极端值不出画面。"""
import numpy as np
import pytest

from po_jepa import datasets, simulation, systems, variants


@pytest.fixture
def use():
    old = datasets.active()
    def set_(system, knobs="", count=200):
        spec = datasets.DatasetSpec(system, count, variants.format(system, variants.parse(system, knobs)))
        datasets._set_for_tests(spec)
        return spec
    yield set_
    datasets._set_for_tests(old)


def test_parse_format_roundtrip_and_errors():
    v = variants.parse("double_well", "gain=2,axes=x,coupling=0.4")
    assert variants.format("double_well", v) == "axes=x,coupling=0.4,gain=2"
    assert variants.parse("double_well", variants.format("double_well", v)) == v
    assert variants.parse("double_well", "") == variants.Variant() and variants.Variant().is_default
    for bad in ("gain=9", "mu=1", "axes=z", "coupling", "gain=2,gain=3", "shape=ring"):
        with pytest.raises(ValueError):
            variants.parse("double_well", bad)
    with pytest.raises(ValueError):
        variants.parse("landscape", "")
    assert variants.parse("multi_particle", "n=5").param("multi_particle", "n") == 5


def test_dataset_spec_tags():
    assert datasets.parse("double_well:200").tag == "double_well-200" and datasets.parse("double_well:200").is_v6_default
    s = datasets.parse("double_well~gain=2.0,axes=x:2000")
    assert s.tag == "double_well~axes=x,gain=2-2000" and not s.is_v6_default
    with pytest.raises(ValueError):
        datasets.DatasetSpec("double_well", 200, "gain=2.0")     # 非规范写法


def test_observation_only_variant_keeps_states(use):
    """只改观察：隐藏状态与基础场景逐位相同（可调积分器在默认参数下与原实现一致），画面不同。"""
    for system in ("double_well", "van_der_pol", "multi_particle"):
        use(system)
        base_states, base_frames = systems.get(system).frames(3)
        use(system, "color=white,background=plain")
        states, frames = systems.get(system).frames(3)
        assert np.array_equal(states, base_states), system
        assert not np.array_equal(frames, base_frames)


def test_control_knobs(use):
    use("double_well", "axes=x")
    c = systems.commanded_controls(5)
    assert np.all(c.values[:, 1] == 0.)
    use("double_well", "segments=2,zero=0")
    c = systems.commanded_controls(5)
    assert np.all(c.ends_ticks-c.starts_ticks == 2) and np.all(np.abs(c.values).sum(1) > 0)
    use("double_well")
    base = systems.get().frames(7)[0]
    use("double_well", "gain=2")
    assert not np.allclose(systems.get().frames(7)[0], base)
    use("double_well", "delay=2")
    v = datasets.active().knobs
    u = systems.applied_ticks(7, v, systems.commanded_controls(7))
    u0 = systems.applied_ticks(7, variants.Variant(), simulation.sample_controls(7))
    assert np.array_equal(u[2:], u0[:-2]) and np.all(u[:2] == 0)


def test_physics_params_change_dynamics(use):
    use("van_der_pol"); base = systems.get().frames(4)[0]
    use("van_der_pol", "mu=1.5"); assert not np.allclose(systems.get().frames(4)[0], base)
    use("multi_particle", "n=5"); s, f = systems.get().frames(4)
    assert s.shape == (33, 10) and f.shape == (33, 256, 256, 3)


@pytest.mark.parametrize("shape", variants.SHAPES)
def test_landscape_shapes(use, shape):
    use("landscape", f"shape={shape},gain=4")
    sysm = systems.get()
    for tid in range(0, 200, 25):
        states, frames = sysm.frames(tid)
        assert states.shape == (33, 2) and np.abs(states).max() < sysm.BOUND
        assert frames.dtype == np.uint8 and frames.shape == (33, 256, 256, 3)


def test_advection_variant_and_noise(use):
    use("advection_diffusion", "blobs=1,diffusion=0.001,pixel_noise=10")
    s, f = systems.get().frames(2)
    assert s.shape == (33, 256) and f.std() > 0
    use("double_well"); base = systems.get().frames(2)
    use("double_well", "distractors=3"); s, f = systems.get().frames(2)
    assert np.array_equal(s, base[0]) and not np.array_equal(f, base[1])
