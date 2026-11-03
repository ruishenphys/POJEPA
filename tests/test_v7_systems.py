"""v7 新物理系统：确定性、画面域、双势阱与 v6 逐位相同、平流扩散精确性。"""
import numpy as np
import pytest

from po_jepa import datasets, simulation, systems


@pytest.fixture(params=datasets.SYSTEMS)
def system(request):
    old = datasets._set_for_tests(datasets.DatasetSpec(request.param, 200))
    yield systems.get()
    datasets._set_for_tests(old)


def test_frames_shape_and_determinism(system):
    s1, f1 = system.frames(3)
    s2, f2 = system.frames(3)
    assert f1.shape == (33, 256, 256, 3) and f1.dtype == np.uint8
    assert s1.shape == (33, system.state_dim) and np.array_equal(s1, s2) and np.array_equal(f1, f2)
    assert not np.array_equal(f1[0], f1[-1])          # 画面确实在变


def test_double_well_is_v6():
    old = datasets._set_for_tests(datasets.DatasetSpec("double_well", 200))
    try:
        states, frames = systems.get().frames(5)
        ref = simulation.reference_trajectory(5)
        assert np.array_equal(states, ref) and np.array_equal(frames, simulation.render_positions(ref))
    finally:
        datasets._set_for_tests(old)


def test_systems_use_distinct_random_streams():
    values = []
    for name in datasets.SYSTEMS:
        old = datasets._set_for_tests(datasets.DatasetSpec(name, 200))
        try:
            values.append(simulation.sample_controls(0).values.ravel().tolist())
        finally:
            datasets._set_for_tests(old)
    assert len({tuple(v) for v in values}) == len(values)


def test_advection_diffusion_conserves_mass_and_is_exact():
    old = datasets._set_for_tests(datasets.DatasetSpec("advection_diffusion", 200))
    try:
        s = systems.get()
        f = s.fields(11)
        assert np.allclose(f.sum((1, 2)), f[0].sum(), rtol=1e-9)
        # 零控制且 D=0 时应精确不变：用一个局部子类验证傅里叶推进
        class Frozen(systems.AdvectionDiffusion): D = 0.
        g = Frozen("x", 256, "").fields(11)
        c = simulation.sample_controls(11)
        if np.all(c.values[0] == 0):
            assert np.allclose(g[1], g[0], atol=1e-10)
    finally:
        datasets._set_for_tests(old)
