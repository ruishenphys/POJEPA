"""S20b 登记 gray_scott、push：登记顺序与随机流、调节项解析、System 子类把调节项和执行动作接到 new_systems。

物理本身在 tests/test_new_systems.py 里核；这里只核“接线”：同一条轨迹经 systems.get().frames() 和
直接调用 new_systems（手工给同样的执行动作和参数）逐位相同，每类调节项都传到了该传的地方。
"""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from po_jepa import datasets, new_systems, simulation, systems, variants
from po_jepa.variants import Variant

V7_SYSTEMS = ("double_well", "van_der_pol", "multi_particle", "advection_diffusion", "landscape")
SRC = Path(__file__).resolve().parents[1]/"src"


@pytest.fixture
def use():
    old = datasets.active()
    def set_(system, knobs="", count=200):
        spec = datasets.DatasetSpec(system, count, variants.format(system, variants.parse(system, knobs)))
        datasets._set_for_tests(spec)
        return spec
    yield set_
    datasets._set_for_tests(old)


def ticks(controls):
    u = np.zeros((32, 2))
    for a, b, value in zip(controls.starts_ticks, controls.ends_ticks, controls.values):
        u[a:b] = value
    return u


# —— 登记顺序、随机流 ——

def test_new_systems_are_appended_after_the_v7_systems(use):
    assert datasets.SYSTEMS == V7_SYSTEMS+("gray_scott", "push")
    assert set(systems.REGISTRY) == set(datasets.SYSTEMS) == set(variants.PARAMS)
    for name, index in new_systems.SYSTEM_INDEX.items():
        assert datasets.SYSTEMS.index(name) == index
        assert systems.REGISTRY[name].state_dim == new_systems.SYSTEMS[name].state_dim
    # v7 场景的随机流熵与登记前相同（写死，不从 SYSTEMS 推）
    expected = {"double_well": [1703, 7, 2], "van_der_pol": [1703, 1001, 7, 2], "multi_particle": [1703, 1002, 7, 2],
                "advection_diffusion": [1703, 1003, 7, 2], "landscape": [1703, 1004, 7, 2]}
    for system, entropy in expected.items():
        use(system, "shape=ring" if system == "landscape" else "")
        assert simulation.seed_entropy(7, 2) == entropy
    # 新场景：simulation 的随机流（动作、执行噪声）和 new_systems.rng（初值、干扰光点、像素噪声）是同一套熵
    for name in ("gray_scott", "push"):
        use(name)
        assert simulation.seed_entropy(7, 2) == [1703, 1000+new_systems.SYSTEM_INDEX[name], 7, 2]
        assert np.array_equal(simulation._rng(7, 0).random(3), new_systems.rng(name, 7, 0).random(3))


# —— 调节项 ——

def test_params_have_one_source_and_parse():
    for name in ("gray_scott", "push"):
        assert new_systems.PARAMS[name] is variants.PARAMS[name]
    assert set(variants.PARAMS["gray_scott"]) == {"feed", "kill", "inject"}
    assert set(variants.PARAMS["push"]) == {"mu_s", "slope", "slope_dir", "size"}
    v = variants.parse("gray_scott", "feed=0.045,kill=0.062,color=white,gain=2")
    assert v.param("gray_scott", "feed") == .045 and v.param("gray_scott", "inject") == .05
    assert variants.format("gray_scott", v) == "color=white,feed=0.045,gain=2,kill=0.062"
    v = variants.parse("push", "slope_dir=-90,size=1.2,sigma=4,distractors=2")
    assert v.param("push", "slope_dir") == -90. and type(v.param("push", "slope_dir")) is float
    assert variants.parse("push", variants.format("push", v)) == v
    assert variants.parse("gray_scott", "") == Variant() == variants.parse("push", "")
    for system, bad in (("gray_scott", "feed=0.03"), ("gray_scott", "kill=0.07"), ("gray_scott", "inject=0.3"),
                        ("gray_scott", "sigma=4"), ("gray_scott", "distractors=1"), ("gray_scott", "mu_s=0.5"),
                        ("gray_scott", "shape=ring"), ("push", "slope=0.7"), ("push", "size=0.3"),
                        ("push", "slope_dir=200"), ("push", "mu_s=nan"), ("push", "feed=0.04"), ("push", "shape=ring")):
        with pytest.raises(ValueError):
            variants.parse(system, bad)
    with pytest.raises(ValueError):
        variants.validate("gray_scott", Variant(sigma=4.))
    with pytest.raises(ValueError):
        variants.validate("gray_scott", Variant(distractors=1))


def test_dataset_specs_for_new_systems():
    s = datasets.parse("push~size=1.2,mu_s=0.5:2000")
    assert s.system == "push" and s.tag == "push~mu_s=0.5,size=1.2-2000" and not s.is_v6_default
    assert s.knobs.param("push", "size") == 1.2
    assert datasets.parse("gray_scott:40").knobs == Variant()
    with pytest.raises(ValueError):
        datasets.parse("gray_scott~sigma=4:40")


@pytest.mark.parametrize("name", ("gray_scott", "push"))
def test_environment_variable_and_import_order(name):
    """子进程：PO_JEPA_DATASET 在导入时就要能解析新场景；先导入 new_systems 也不会循环导入失败；
    只导入 systems（v7 场景）不会加载 new_systems。"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")}
    env.update(PYTHONPATH=str(SRC), PO_JEPA_DATASET=f"{name}~gain=2:40")
    code = ("import po_jepa.new_systems as n, sys; from po_jepa import datasets, systems; "
            f"assert datasets.active().system == {name!r} and systems.get().name == {name!r}; "
            "assert systems.get().state_dim == n.SYSTEMS[datasets.active().system].state_dim")
    subprocess.run([sys.executable, "-B", "-c", code], env=env, check=True)
    env["PO_JEPA_DATASET"] = "van_der_pol:40"
    code = ("import sys; from po_jepa import datasets, systems, variants; systems.get().frames(0); "
            "assert 'po_jepa.new_systems' not in sys.modules")
    subprocess.run([sys.executable, "-B", "-c", code], env=env, check=True)


# —— 接线：frames() = new_systems(执行动作, 物理参数, 观察项) ——

def test_push_frames_match_direct_call(use):
    use("push")
    u = ticks(simulation.sample_controls(3))
    states, frames = systems.get().frames(3)
    ref_states, ref_frames = new_systems.SYSTEMS["push"].frames(3, u, {}, Variant())
    assert np.array_equal(states, ref_states) and np.array_equal(frames, ref_frames)
    assert states.shape == (33, 5) and frames.shape == (33, 256, 256, 3) and frames.dtype == np.uint8
    assert np.array_equal(systems.get().reference_trajectory(3), states)
    assert np.array_equal(ticks(systems.commanded_controls(3)), u)


def test_gray_scott_frames_match_direct_call(use):
    use("gray_scott")
    u = ticks(simulation.sample_controls(2))
    states, frames = systems.get().frames(2)
    ref_states, ref_frames = new_systems.SYSTEMS["gray_scott"].frames(2, u, {}, Variant())
    assert np.array_equal(states, ref_states) and np.array_equal(frames, ref_frames)
    assert states.shape == (33, 514) and frames.dtype == np.uint8


@pytest.fixture
def captured(monkeypatch):
    """把 new_systems 的 frames 换成记录参数的桩（不跑物理），核调节项传到哪里。"""
    calls = []
    for name, impl in new_systems.SYSTEMS.items():
        monkeypatch.setattr(impl, "frames", lambda tid, u, params, v, _n=name: calls.append((_n, tid, u, params, v)) or (None, None))
    return calls


@pytest.mark.parametrize("name", ("gray_scott", "push"))
def test_knobs_reach_the_right_place(use, captured, name):
    phys = "feed=0.045,kill=0.062" if name == "gray_scott" else "mu_s=0.5,slope=0.2,slope_dir=45"
    params = {"feed": .045, "kill": .062} if name == "gray_scott" else {"mu_s": .5, "slope": .2, "slope_dir": 45.}
    use(name); systems.get().frames(5)
    base = ticks(simulation.sample_controls(5))
    _, tid, u, p, v = captured[-1]
    assert tid == 5 and p == {} and v == Variant() and np.array_equal(u, base)
    # 物理参数 → params；观察项 → v，执行动作不变
    use(name, phys+",color=white,background=checker,pixel_noise=5"); systems.get().frames(5)
    _, _, u, p, v = captured[-1]
    assert p == params and np.array_equal(u, base)
    assert (v.color, v.background, v.pixel_noise) == ("white", "checker", 5.)
    # 执行类：增益、延迟、执行噪声只改执行动作（模型看到的下达动作不变）
    use(name, "gain=2"); systems.get().frames(5)
    assert np.array_equal(captured[-1][2], 2*base) and np.array_equal(ticks(systems.commanded_controls(5)), base)
    use(name, "delay=2"); systems.get().frames(5)
    assert np.array_equal(captured[-1][2], np.concatenate((np.zeros((2, 2)), base[:-2])))
    use(name, "noise=0.1"); systems.get().frames(5)
    expected = base+simulation._rng(5, 5).normal(0., .1, (32, 2))
    assert np.array_equal(captured[-1][2], expected)
    # 改动作抽样：下达动作和执行动作一起变
    use(name, "axes=x,segments=2"); systems.get().frames(5)
    c = systems.commanded_controls(5)
    assert np.all(c.values[:, 1] == 0.) and np.all(c.ends_ticks-c.starts_ticks == 2)
    assert np.array_equal(captured[-1][2], ticks(c))


def test_push_knobs_end_to_end(use):
    """真跑一遍：物理参数和增益改了状态，只改观察的调节项状态逐位不变、画面变了。"""
    use("push"); base_states, base_frames = systems.get().frames(4)
    u = ticks(simulation.sample_controls(4)); impl = new_systems.SYSTEMS["push"]
    use("push", "gain=2,mu_s=0.1"); states, _ = systems.get().frames(4)
    assert np.array_equal(states, impl.trajectory(4, 2*u, {"mu_s": .1})) and not np.array_equal(states, base_states)
    assert np.array_equal(systems.get().reference_trajectory(4), states)
    use("push", "color=white,distractors=2,sigma=4"); states, frames = systems.get().frames(4)
    assert np.array_equal(states, base_states) and not np.array_equal(frames, base_frames)


def test_guards(use):
    use("double_well")
    for name in ("gray_scott", "push"):
        with pytest.raises(ValueError):
            systems.REGISTRY[name].frames(0)        # 当前数据集不是这个场景
    use("push")
    with pytest.raises(ValueError):
        systems.get().render(np.zeros((33, 5)))
    with pytest.raises(ValueError):          # S21b 接上了续算接口；调节项与当前数据集不一致时报错
        systems.get().variant_trajectory(0, Variant(gain=2.))
