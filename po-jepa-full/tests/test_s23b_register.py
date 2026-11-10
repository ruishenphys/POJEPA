"""S23b 登记二阶场景：pendulum、bouncing_balls、wave、cart_pole、spring_chain（物理与渲染在 second_order，S23a）。

物理本身在 tests/test_second_order.py 里核；这里只核“接线”（做法同 tests/test_new_systems_registration.py）：
登记顺序与随机流、参数单一来源、validate 的适用性、frames()/frames_with_prev() 与直接调用 second_order 逐位相同、
每类调节项传到该传的地方、续算和部分帧重画与整条轨迹逐位相同（探针和规划对二阶场景报错那一条 S25a 起不再成立，见 test_s25a）。
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from po_jepa import datasets, protocols, second_order, simulation, systems, variants
from po_jepa.variants import Variant

NAMES = ("pendulum", "bouncing_balls", "wave", "cart_pole", "spring_chain")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"src"


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

def test_appended_after_push_with_s23a_stream_indices(use):
    # 写死，不从 second_order.SYSTEM_INDEX 推
    assert datasets.SYSTEMS == ("double_well", "van_der_pol", "multi_particle", "advection_diffusion", "landscape",
                                "gray_scott", "push", "pendulum", "bouncing_balls", "wave", "cart_pole", "spring_chain")
    assert variants.SECOND_ORDER == NAMES
    assert set(systems.REGISTRY) == set(datasets.SYSTEMS) == set(variants.PARAMS)
    for name in NAMES:
        assert datasets.SYSTEMS.index(name) == second_order.SYSTEM_INDEX[name]
        assert isinstance(systems.REGISTRY[name], systems.SecondOrderSystem)
    # 已有 7 个场景的熵不变（写死）
    expected = {"double_well": [1703, 7, 2], "van_der_pol": [1703, 1001, 7, 2], "landscape": [1703, 1004, 7, 2],
                "gray_scott": [1703, 1005, 7, 2], "push": [1703, 1006, 7, 2]}
    for system, entropy in expected.items():
        use(system, "shape=ring" if system == "landscape" else "")
        assert simulation.seed_entropy(7, 2) == entropy
    # 二阶场景：simulation 的随机流（动作 1、执行噪声 5）和 second_order.rng（初值 0、干扰光点、像素噪声）是同一套熵
    for i, name in enumerate(NAMES):
        use(name)
        assert simulation.seed_entropy(7, 2) == [1703, 1007+i, 7, 2]
        assert np.array_equal(simulation._rng(7, 0).random(3), second_order.rng(name, 7, 0).random(3))


def test_state_dim_is_the_default_hidden_dim(use):
    for name, dim in zip(NAMES, (2, 12, 514, 4, 16)):
        use(name)
        assert systems.get().state_dim == dim
        assert systems.get().reference_trajectory(1).shape == (33, dim)


# —— 调节项 ——

def test_params_have_one_source_and_parse():
    for name in NAMES:
        assert second_order.PARAMS[name] is variants.PARAMS[name]
    assert variants.PARAMS["pendulum"] == {"g_over_l": (9.81, (4., 16.)), "damping": (.2, (0., 1.))}
    assert variants.PARAMS["spring_chain"]["n"] == (4, (2, 6)) and variants.PARAMS["bouncing_balls"]["n"] == (3, (1, 5))
    v = variants.parse("spring_chain", "n=6,stiffness=9,gain=2")
    assert v.param("spring_chain", "n") == 6 and type(v.param("spring_chain", "n")) is int
    assert v.param("spring_chain", "damping") == .1
    assert variants.format("spring_chain", v) == "gain=2,n=6,stiffness=9"
    assert variants.parse("spring_chain", variants.format("spring_chain", v)) == v
    assert datasets.parse("wave~load=0.4,speed=0.3:2000").tag == "wave~load=0.4,speed=0.3-2000"
    for system, bad in (("pendulum", "g_over_l=3"), ("pendulum", "damping=nan"), ("bouncing_balls", "n=6"),
                        ("bouncing_balls", "restitution=0.4"), ("wave", "speed=0.35"), ("wave", "load=0.5"),
                        ("cart_pole", "length=1.5"), ("cart_pole", "mu=1"), ("spring_chain", "stiffness=8"),
                        ("spring_chain", "n=7"), ("pendulum", "shape=ring")):
        with pytest.raises(ValueError):
            variants.parse(system, bad)


ALLOWED = {   # 调节项 → 接受它的二阶场景（写死）
    "sigma=4": ("pendulum", "cart_pole", "spring_chain"),
    "distractors=2": ("pendulum", "bouncing_balls", "cart_pole", "spring_chain"),
    "movers=2": ("pendulum", "bouncing_balls", "cart_pole", "spring_chain"),
    "process_noise=0.1": (), "policy=goal": (), "policy=mixed": (),
    "axes=y": ("pendulum", "bouncing_balls", "wave", "spring_chain"),
    "axes=x": NAMES, "saturation=0.3": NAMES, "gain=2": NAMES, "delay=1": NAMES, "noise=0.1": NAMES,
    "segments=2": NAMES, "zero=0.5": NAMES, "color=white": NAMES, "background=noise": NAMES,
    "occluders=2": NAMES, "pixel_noise=10": NAMES,
}


@pytest.mark.parametrize("knob", sorted(ALLOWED))
def test_validate_applicability(knob):
    for name in NAMES:
        if name in ALLOWED[knob]:
            variants.parse(name, knob)
        else:
            with pytest.raises(ValueError):
                variants.parse(name, knob)


def test_validate_agrees_with_second_order_check_variant():
    """validate 接受的观察调节项，second_order 画得出来；second_order 拒绝的，validate 也拒绝（两处的规则不打架）。"""
    for name in NAMES:
        impl = second_order.SYSTEMS[name]
        for knob in ("sigma=4", "distractors=2", "movers=2"):
            v = variants._convert({knob.split("=")[0]: knob.split("=")[1]})
            v = Variant(**v)
            try:
                variants.validate(name, v)
                accepted = True
            except ValueError:
                accepted = False
            if accepted:
                impl.check_variant(v)
            else:
                with pytest.raises(ValueError):
                    impl.check_variant(v)


# —— 接线：frames() = second_order(执行动作, 物理参数, 观察项) 的下标 1…33 ——

@pytest.mark.parametrize("name", NAMES)
def test_frames_match_direct_call(use, name):
    use(name)
    S = systems.get(); impl = second_order.SYSTEMS[name]
    u = ticks(simulation.sample_controls(3))
    ref_states, ref_frames = impl.frames(3, u, {}, Variant())
    states, frames = S.frames(3)
    prev_states, prev_frames = S.frames_with_prev(3)
    assert frames.shape == (33, 256, 256, 3) and frames.dtype == np.uint8 and prev_frames.shape == (34, 256, 256, 3)
    assert np.array_equal(prev_states, ref_states) and np.array_equal(prev_frames, ref_frames)
    assert np.array_equal(states, ref_states[1:]) and np.array_equal(frames, ref_frames[1:])
    assert np.array_equal(S.reference_trajectory(3), states)
    # 下标 0 是 t = −1 的初值（随机流 0），不是 tick 0
    full = impl.trajectory(3, u, {})
    assert np.array_equal(full[0], impl.initial(3, {})) and np.array_equal(prev_states[0], impl.hidden(full[:1], {})[0])
    assert np.array_equal(ticks(systems.commanded_controls(3)), u)


@pytest.fixture
def captured(monkeypatch):
    """把 second_order 的 frames 换成记录参数的桩（不跑物理），核调节项传到哪里。"""
    calls = []
    for name, impl in second_order.SYSTEMS.items():
        monkeypatch.setattr(impl, "frames", lambda tid, u, params, v, _n=name: calls.append((_n, tid, u, params, v))
                            or (np.zeros((34, 1)), np.zeros((34, 1, 1, 3), np.uint8)))
    return calls


PHYS = {"pendulum": ("g_over_l=12,damping=0.5", {"g_over_l": 12., "damping": .5}),
        "bouncing_balls": ("n=5,restitution=0.7", {"n": 5, "restitution": .7}),
        "wave": ("speed=0.2,load=0.1", {"speed": .2, "load": .1}),
        "cart_pole": ("length=0.8,friction=0", {"length": .8, "friction": 0.}),
        "spring_chain": ("n=3,stiffness=20", {"n": 3, "stiffness": 20.})}


@pytest.mark.parametrize("name", NAMES)
def test_knobs_reach_the_right_place(use, captured, name):
    use(name); systems.get().frames(5)
    base = ticks(simulation.sample_controls(5))
    n, tid, u, p, v = captured[-1]
    assert (n, tid, p, v) == (name, 5, {}, Variant()) and np.array_equal(u, base)
    # 物理参数 → params；观察项 → v，执行动作不变
    use(name, PHYS[name][0]+",color=white,background=checker,pixel_noise=5,occluders=1"); systems.get().frames(5)
    _, _, u, p, v = captured[-1]
    assert p == PHYS[name][1] and np.array_equal(u, base)
    assert (v.color, v.background, v.pixel_noise, v.occluders) == ("white", "checker", 5., 1)
    # 执行类：增益、延迟、执行噪声、饱和只改执行动作（模型看到的下达动作不变）
    use(name, "gain=2"); systems.get().frames(5)
    assert np.array_equal(captured[-1][2], 2*base) and np.array_equal(ticks(systems.commanded_controls(5)), base)
    use(name, "delay=2"); systems.get().frames(5)
    assert np.array_equal(captured[-1][2], np.concatenate((np.zeros((2, 2)), base[:-2])))
    use(name, "noise=0.1"); systems.get().frames(5)
    assert np.array_equal(captured[-1][2], base+simulation._rng(5, 5).normal(0., .1, (32, 2)))
    use(name, "saturation=0.1"); systems.get().frames(5)
    assert np.array_equal(captured[-1][2], np.clip(base, -.1, .1))
    # 改动作抽样：下达动作和执行动作一起变
    # S23a-fix1（Codex S23b P3-1）：非默认 zero 真的改了零动作段的比例——按 v6 规则（流 1、段长 1/4/8）独立重抽
    use(name, "zero=0.5"); changed = 0
    for tid in (5, 6, 7, 8):
        systems.get().frames(tid)
        g = simulation._rng(tid, 1); expect = np.zeros((32, 2)); t = 0
        while t < 32:
            n_ = int(g.choice([1, 4, 8])); expect[t:t+n_] = 0. if g.random() < .5 else g.uniform(-.5, .5, 2); t += n_
        assert np.array_equal(ticks(systems.commanded_controls(tid)), expect) and np.array_equal(captured[-1][2], expect)
        changed += not np.array_equal(expect, ticks(simulation.sample_controls(tid)))
    assert changed >= 2
    use(name, "axes=x,segments=2"); systems.get().frames(5)
    c = systems.commanded_controls(5)
    assert np.all(c.values[:, 1] == 0.) and np.all(c.ends_ticks-c.starts_ticks == 2)
    assert np.array_equal(captured[-1][2], ticks(c))


def test_pendulum_and_cart_pole_end_to_end(use):
    """真跑：物理参数和增益改了状态；只改观察的调节项状态逐位不变、画面变了。"""
    impl = second_order.SYSTEMS["pendulum"]
    use("pendulum"); base_states, base_frames = systems.get().frames(4)
    u = ticks(simulation.sample_controls(4))
    use("pendulum", "gain=2,damping=0.5"); states, _ = systems.get().frames(4)
    assert np.array_equal(states, impl.trajectory(4, 2*u, {"damping": .5})[1:]) and not np.array_equal(states, base_states)
    use("pendulum", "color=white,movers=2,sigma=4"); states, frames = systems.get().frames(4)
    assert np.array_equal(states, base_states) and not np.array_equal(frames, base_frames)
    # cart_pole：u_y 不起作用，axes=x 的下达动作 u_y = 0，物理与“只给 u_x”的执行动作相同
    use("cart_pole", "axes=x"); states, _ = systems.get().frames(2)
    c = ticks(systems.commanded_controls(2)); assert np.all(c[:, 1] == 0.)
    assert np.array_equal(states, second_order.SYSTEMS["cart_pole"].trajectory(2, c, {})[1:])


# —— 续算与部分帧重画（反事实探针、规划要用的接口；工具本身还没接）——

@pytest.mark.parametrize("name", NAMES)
def test_restart_and_partial_redraw_match_the_trajectory(use, name):
    knobs = "pixel_noise=10,occluders=1" + ("" if name == "wave" else ",movers=1")
    use(name, knobs); S = systems.get(); v = datasets.active().knobs
    hidden, frames = S.frames(6)
    full = S.variant_trajectory(6, v)
    assert full.shape[0] == 33 and np.array_equal(S.hidden(full), hidden)
    # 整条 33 帧不给 ticks：画成 tick 0…32（不是 second_order 默认的 −1…32，那样会错位一帧）
    assert np.array_equal(S.draw(full, 6, v), frames)
    for start in (0, 7, 31):
        rest = S.variant_trajectory(6, v, start=start, state=full[start])
        assert np.array_equal(rest, full[start:])
        assert np.array_equal(S.draw(full[[start]], 6, v, ticks=[start]), frames[[start]])
    # 换成别的轨迹的动作：起点相同、之后不同
    other = S.variant_trajectory(6, v, controls=systems.commanded_controls(7), start=7, state=full[7], ticks=4)
    assert other.shape[0] == 5 and np.array_equal(other[0], full[7]) and not np.array_equal(other[-1], full[11])
    with pytest.raises(ValueError):
        S.draw(full[:5], 6, v)                                   # 部分帧不给 ticks
    with pytest.raises(ValueError):
        S.variant_trajectory(6, v, start=3)                      # tick>0 要给状态
    with pytest.raises(ValueError):
        S.variant_trajectory(6, v, start=30, state=full[30], ticks=5)


def test_reference_trajectory_steps_per_tick(use):
    use("pendulum"); S = systems.get()
    states = S.frames(1)[0]
    assert np.array_equal(S.reference_trajectory(1, steps_per_tick=50), states)
    fine = S.reference_trajectory(1, steps_per_tick=200)
    assert not np.array_equal(fine, states) and np.abs(fine-states).max() < 1e-6
    for bad in (0, 2.5):
        with pytest.raises(ValueError):
            S.reference_trajectory(1, steps_per_tick=bad)
    use("spring_chain")
    with pytest.raises(ValueError):          # 矩阵指数精确推进，没有子步：不静默忽略
        systems.get().reference_trajectory(1, steps_per_tick=10)


def test_guards(use):
    use("double_well")
    for name in NAMES:
        with pytest.raises(ValueError):
            systems.REGISTRY[name].frames(0)                     # 当前数据集不是这个场景
    use("pendulum"); S = systems.get()
    with pytest.raises(ValueError):
        S.render(np.zeros((33, 2)))
    with pytest.raises(ValueError):
        S.variant_trajectory(0, Variant(gain=2.))                # 调节项与当前数据集不一致
    with pytest.raises(ValueError):
        S.draw(np.zeros((33, 2)), 0, Variant(gain=2.))
    with pytest.raises(ValueError):
        S.policy_anchor(0, Variant(), simulation._rng(0, 9))


# —— 入口：环境变量、导入、run.py、encode_v7、探针和规划、协议 ——

def test_environment_variable_and_import_order():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")}
    env.update(PYTHONPATH=str(SRC), PO_JEPA_DATASET="spring_chain~gain=2,n=3:40")
    code = ("import po_jepa.second_order as s; from po_jepa import datasets, systems; "
            "assert systems.get().name == 'spring_chain' and datasets.active().knobs.param('spring_chain', 'n') == 3; "
            "assert systems.get().frames(0)[0].shape == (33, 12)")
    subprocess.run([sys.executable, "-B", "-c", code], env=env, check=True)
    env["PO_JEPA_DATASET"] = "push:40"           # 已有场景不加载 second_order
    code = ("import sys; from po_jepa import datasets, systems; systems.get().frames(0); "
            "assert 'po_jepa.second_order' not in sys.modules")
    subprocess.run([sys.executable, "-B", "-c", code], env=env, check=True)


def test_run_py_and_encode_v7_know_the_new_scenes():
    text = (ROOT/"run.py").read_text(encoding="utf-8")
    choices = re.search(r"--system',required=True,choices=\(([^)]*)\)", text).group(1)
    assert tuple(x.strip("'") for x in choices.split(",")) == datasets.SYSTEMS
    assert '"src/po_jepa/second_order.py"' in (ROOT/"tools/encode_v7.py").read_text(encoding="utf-8")


def _clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")}
    env["PYTHONPATH"] = str(SRC)
    return env


# S25a：探针和规划接上了二阶场景（以前这里核“直接报错”），改由 tests/test_s25a_second_order_tasks.py 核。


def test_protocols_do_not_pick_up_second_order_scenes():
    """协议的场景表（SCENES，v8 任务表默认）不变：二阶场景不在里面。S25a 起二阶场景有 P1 档位，P13 给它们组合档
    （tests/test_s25a_second_order_tasks.py 核），它们在 protocols.SECOND_ORDER_SCENES。"""
    assert not set(NAMES) & {protocols.system_of(s) for s in protocols.SCENES}
    assert protocols.SECOND_ORDER_SCENES == NAMES
    assert sum(len(protocols.tests(s)) for s in protocols.SCENES) == 82


def test_original_protocol_entries_are_frozen():
    """S23a-fix1（Codex S23b P3-2）：只数 82 格挡不住改档位取值或换顺序。原 7 个场景三档的完整条目
    (场景, 协议, 档位, 变体) 按顺序写成紧凑 JSON 的 SHA-256，在 S23b 之前的 a15737c 上算出（e61df67 上相同）。"""
    import hashlib, json
    rows = [[s, pid, lv, var] for s in protocols.SCENES for (pid, lv, var) in protocols.tests(s, levels=("轻", "标准", "重"))]
    assert len(rows) == 234 and sum(r[2] == "标准" for r in rows) == 82
    blob = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == "cb5b07c09788ce203321ed18aab44ad605d3b89c1021ae5dd13ba19b5734a230"
