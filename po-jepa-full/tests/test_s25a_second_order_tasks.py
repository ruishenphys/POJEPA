"""S25a：五个二阶场景接反事实探针（tools/probe_actions.py）、规划（tools/plan_v7.py）、评测协议和 v8 任务表。

说明和验收单 docs/segments/S25a.md。参照值尽量写死或从渲染器的坐标独立算（错误清单：测试不拿被测函数当参照）。
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from po_jepa import configuration, datasets, protocols, second_order, simulation, systems, variants

ROOT = Path(__file__).resolve().parents[1]
SECOND = ("pendulum", "bouncing_balls", "wave", "cart_pole", "spring_chain")


@pytest.fixture
def use():
    old = datasets.active()
    def set_(system, knobs="", count=200):
        spec = datasets.DatasetSpec(system, count, variants.format(system, variants.parse(system, knobs)))
        datasets._set_for_tests(spec)
        return spec
    yield set_
    datasets._set_for_tests(old)


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"tools"/f"{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


# —— 构形 ——

def test_configuration_values_are_written_out():
    assert configuration.PX == {"pendulum": 80., "cart_pole": 36., "spring_chain": 48., "bouncing_balls": 48., "wave": None}
    c = configuration.configuration("pendulum", [math.pi/2, 7.], {})
    assert np.allclose(c, [1., 0.], atol=1e-15)                                   # θ = π/2：摆锤在支点右边
    assert np.allclose(configuration.configuration("pendulum", [0., -3.], {}), [0., -1.])   # θ = 0 竖直向下
    c = configuration.configuration("cart_pole", [.5, 9., math.pi/6, 9.], {})
    assert np.allclose(c, [.5, 1., math.sqrt(3)/2])                               # l = 1：摆锤 (x + 0.5, cos 30°)
    c = configuration.configuration("cart_pole", [.5, 9., math.pi/6, 9.], {"length": 1.4})
    assert np.allclose(c, [.5, 1.2, 1.4*math.sqrt(3)/2])
    h = np.arange(16.)
    assert np.array_equal(configuration.configuration("spring_chain", h, {}), np.arange(8.))      # n = 4：位置在前 8 维
    assert np.array_equal(configuration.controlled("spring_chain", h, {}), [0., 1.])
    h = np.arange(24.)
    assert np.array_equal(configuration.configuration("spring_chain", h, {"n": 6}), np.arange(12.))
    assert np.array_equal(configuration.configuration("bouncing_balls", np.arange(12.), {}), np.arange(6.))
    assert np.array_equal(configuration.controlled("bouncing_balls", np.arange(12.), {}), [0., 1.])
    h = np.arange(514.)
    assert np.array_equal(configuration.configuration("wave", h, {}), np.arange(256.))           # u 池化，不含 u_t、p
    assert np.array_equal(configuration.controlled("wave", h, {}), np.arange(256.))
    assert np.array_equal(configuration.controlled("pendulum", [0., 1.], {}), [0., -1.])
    assert configuration.motion("wave", np.zeros(514), np.r_[np.zeros(256), np.ones(258)], {}) == 0.
    b = np.zeros(514); b[7] = 3.
    assert configuration.motion("wave", np.zeros(514), b, {}) == 3.


def test_configuration_rejects_wrong_shapes_and_scenes():
    for name, h, params in (("pendulum", np.zeros(4), {}), ("spring_chain", np.zeros(16), {"n": 5}),
                            ("bouncing_balls", np.zeros(13), {}), ("wave", np.zeros(32770), {}), ("cart_pole", np.zeros((1, 4)), {})):
        with pytest.raises(ValueError):
            configuration.configuration(name, h, params)
    with pytest.raises(ValueError):
        configuration.configuration("double_well", np.zeros(2), {})


def test_motion_is_the_pixel_distance_of_the_drawn_positions():
    """*_px 和渲染器画的位置一致：用 second_order 自己的像素坐标函数独立算。"""
    P = second_order.SYSTEMS["pendulum"]
    a, b = np.array([.3, 0.]), np.array([2.1, 5.])
    assert configuration.motion("pendulum", a, b, {}) == pytest.approx(np.linalg.norm(P.bob_rc(a[0])-P.bob_rc(b[0])))
    C = second_order.SYSTEMS["cart_pole"]
    a, b = np.array([.2, 0., .4, 0.]), np.array([-.7, 0., 2.5, 0.])
    for params in ({}, {"length": .6}):
        l = params.get("length", 1.)
        px = lambda q: np.concatenate((C.to_rc(q[0], 0.), C.to_rc(q[0]+l*math.sin(q[2]), l*math.cos(q[2]))))
        assert configuration.motion("cart_pole", a, b, params) == pytest.approx(np.linalg.norm(px(a)-px(b)))
    for name in ("spring_chain", "bouncing_balls"):
        S = second_order.SYSTEMS[name]; n = 3
        a, b = np.random.default_rng(1).normal(size=(2, 4*n))
        px = lambda q: np.concatenate([S.to_rc(p) for p in q[:2*n].reshape(n, 2)])
        assert configuration.motion(name, a, b, {"n": n}) == pytest.approx(np.linalg.norm(px(a)-px(b)))


def test_probe_motion_dispatches_second_order_and_needs_params():
    probe = _tool("probe_actions")
    a, b = np.array([0., 0.]), np.array([math.pi/2, 0.])
    assert probe.motion("pendulum", a, b, 123., {}) == pytest.approx(80*math.sqrt(2))       # scale 参数不用
    with pytest.raises(ValueError):
        probe.motion("pendulum", a, b, 1.)
    assert probe.motion("double_well", a, b, 48.) == pytest.approx(48*math.pi/2)            # 已有场景不变


# —— 反事实探针 ——

@pytest.mark.parametrize("name,knobs", [("pendulum", "pixel_noise=5,movers=1"), ("bouncing_balls", "gain=2,occluders=1"),
                                        ("wave", "pixel_noise=5,speed=0.25"), ("cart_pole", "delay=1,distractors=2"),
                                        ("spring_chain", "n=3,saturation=0.3")])
def test_probe_counterfactual_end_on_second_order(use, name, knobs):
    use(name, knobs); probe = _tool("probe_actions"); S = systems.get(); v = datasets.active().knobs
    probe._FULL.clear()
    states, frames = S.frames(181)
    # 返回的两个状态都是 frames() 那种隐藏状态（wave 是池化的 514 维）
    # 换成自己的动作 = 数据集本身：终点状态和画面都逐位相同
    true_end, cf, frame = probe.counterfactual_end(S, simulation, 181, 181, 5, 8)
    assert np.array_equal(true_end, states[13]) and np.array_equal(cf, states[13])
    assert np.array_equal(frame, frames[13])
    # 换成 donor 的动作：与直接续算相同，且和真实终点不同
    true_end, cf, frame = probe.counterfactual_end(S, simulation, 181, 182, 5, 8)
    full = S.variant_trajectory(181, v)
    # 参照（S25a 修改）：窗口前是 181 自己的下达动作，窗口 [5,13) 是 182 的；逐 tick 独立展开（不调 splice_window）
    own, other = _per_tick(systems.commanded_controls(181)), _per_tick(systems.commanded_controls(182))
    mixed = np.concatenate((own[:5], other[5:13], own[13:]))
    expect = S.variant_trajectory(181, v, controls=simulation.TrajectoryControls(181, np.arange(32), np.arange(1, 33), mixed),
                                  start=5, state=full[5], ticks=8)[-1]
    assert np.array_equal(cf, S.hidden(expect[None])[0]) and np.array_equal(frame, S.draw(expect[None], 181, v, ticks=[13])[0])
    assert np.array_equal(true_end, states[13]) and not np.array_equal(cf, true_end)
    assert probe.motion(name, cf, true_end, 1., dict(v.params)) > 0


def _same(a, b, field):
    """场（advection_diffusion）续算要从实数场重新做 FFT，与整条仿真差 1e-16 量级；其他场景逐位相同。"""
    return np.allclose(a, b, rtol=0, atol=1e-12) if field else np.array_equal(a, b)


def _per_tick(c):
    """下达动作逐 tick 展开 [32,2]（测试自己的写法：按段长 repeat）。"""
    return np.repeat(c.values, (c.ends_ticks-c.starts_ticks).astype(int), axis=0)


@pytest.mark.parametrize("name,knobs", [("pendulum", "delay=1"), ("cart_pole", "delay=2"), ("double_well", "delay=1"),
                                        ("advection_diffusion", "delay=1"), ("push", "delay=1")])
def test_counterfactual_only_swaps_the_window_under_delay(use, monkeypatch, name, knobs):
    """S25a 修改（Codex 审核问题 1）：donor 与 recipient 窗口内的下达动作相同、窗口前不同 → 反事实终点必须等于真实终点。
    原来把 donor 的整条动作传进续算，有延迟时窗口开头执行的是 donor 窗口前的命令（Codex 反例：单摆差 4.7 像素）。"""
    use(name, knobs); probe = _tool("probe_actions"); S = systems.get(); probe._FULL.clear()
    real = systems.commanded_controls
    own = _per_tick(real(181))
    fake = own.copy(); fake[:5] = np.where(np.abs(own[:5]) < .2, .45, -own[:5])    # 窗口前每个 tick 都不同，窗口内及以后相同
    assert (fake[:5] != own[:5]).all()
    monkeypatch.setattr(systems, "commanded_controls",
                        lambda tid: simulation.TrajectoryControls(182, np.arange(32), np.arange(1, 33), fake) if tid == 182 else real(tid))
    for start, tau in ((5, 1), (5, 4)):
        true_end, cf, frame = probe.counterfactual_end(S, simulation, 181, 182, start, tau)
        assert _same(cf, true_end, name == "advection_diffusion"), (name, start, tau)
    # 窗口内不同的对照：反事实终点确实会变（不是因为根本没用 donor）
    fake2 = own.copy(); fake2[5] = np.where(np.abs(own[5]) < .2, .45, -own[5])
    monkeypatch.setattr(systems, "commanded_controls",
                        lambda tid: simulation.TrajectoryControls(182, np.arange(32), np.arange(1, 33), fake2) if tid == 182 else real(tid))
    true_end, cf, frame = probe.counterfactual_end(S, simulation, 181, 182, 5, 4)
    assert not np.allclose(cf, true_end, rtol=0, atol=1e-6)


def test_splice_window_values():
    c1 = simulation.TrajectoryControls(3, np.array([0, 8, 16, 24]), np.array([8, 16, 24, 32]),
                                       np.array([[.1, .2], [.3, .4], [-.1, -.2], [0., .5]]))
    c2 = simulation.TrajectoryControls(4, np.array([0, 4, 12, 20, 28]), np.array([4, 12, 20, 28, 32]),
                                       np.array([[-.5, .5], [.2, 0.], [0., 0.], [.4, -.4], [.1, .1]]))
    out = systems.splice_window(c1, c2, 6, 8)
    want = np.concatenate((np.array([[.1, .2]]*6), np.array([[.2, 0.]]*6), np.array([[0., 0.]]*2),
                           np.array([[.3, .4]]*2), np.array([[-.1, -.2]]*8), np.array([[0., .5]]*8)))
    assert out.trajectory_id == 3 and np.array_equal(out.values, want)
    assert np.array_equal(out.starts_ticks, np.arange(32)) and np.array_equal(out.ends_ticks, np.arange(1, 33))
    for bad in ((-1, 2), (30, 3), (4, 0)):
        with pytest.raises(ValueError):
            systems.splice_window(c1, c2, *bad)


def _probe(tmp_path, system, knobs=""):
    out = tmp_path/f"probe-{system}.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")} | dict(OMP_NUM_THREADS="2")
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/probe_actions.py"), "--system", system, "--trajectories", "200",
                        *(("--variant", knobs) if knobs else ()), "--fake-encoder", "--device", "cpu", "--max-trajectories", "1",
                        "--output", str(out)], env=env, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize("system,knobs", [("pendulum", ""), ("spring_chain", "n=2")])
def test_probe_runs_on_second_order_with_the_fake_encoder(tmp_path, use, system, knobs):
    rep = _probe(tmp_path, system, knobs)
    assert rep["tubelet"] == "dup" and rep["overall"]["windows"] == len(rep["rows"]) > 0
    use(system, knobs); S = systems.get(); params = dict(datasets.active().knobs.params)
    states, _ = S.frames(180)                                   # 测试段第一条
    for r in rep["rows"]:
        assert r["trajectory_id"] == 180 and r["donor_id"] == datasets.donor_id(180)
        want = configuration.motion(system, states[r["start_tick"]+r["duration_cycles"]], states[r["start_tick"]], params)
        assert r["motion_px"] == pytest.approx(want) and np.isfinite(r["action_swap_px"])
        assert np.isfinite(r["identity"]) and np.isfinite(r["oracle_delta_act"])


# —— 规划 ——

def test_plan_summaries_on_second_order(use):
    plan = _tool("plan_v7")
    use("pendulum"); S = systems.get(); v = datasets.active().knobs
    assert np.allclose(plan.summary("pendulum", S, v, np.array([math.pi/2, 4.])), [1., 0.], atol=1e-15)
    assert np.array_equal(plan.oracle_target("pendulum", S, v, np.array([.3, 4.])), plan.summary("pendulum", S, v, np.array([.3, 4.])))
    use("cart_pole", "length=1.4"); S = systems.get(); v = datasets.active().knobs
    assert np.allclose(plan.summary("cart_pole", S, v, np.array([.5, 0., math.pi/6, 0.])), [.5, 1.2, 1.4*math.sqrt(3)/2])
    use("spring_chain", "n=3"); S = systems.get(); v = datasets.active().knobs
    assert np.array_equal(plan.summary("spring_chain", S, v, np.arange(12.)), [0., 1.])
    use("bouncing_balls"); S = systems.get(); v = datasets.active().knobs
    assert np.array_equal(plan.summary("bouncing_balls", S, v, np.arange(12.)), [0., 1.])
    use("wave"); S = systems.get(); v = datasets.active().knobs
    full = S.variant_trajectory(3, v)
    states, _ = S.frames(3)
    assert full.shape == (33, 2*128*128+2)
    assert np.array_equal(plan.summary("wave", S, v, full[7]), states[7][:256])      # 整场先池化，只取 u
    # 目标动作：与粒子类场景相同的均匀随机（每 2 tick 一段）
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    assert np.array_equal(plan.goal_controls("wave", full[0], 7, r1), np.repeat(r2.uniform(-.5, .5, (4, 2)), 2, 0)[:7])


@pytest.mark.parametrize("name,knobs", [("pendulum", "delay=1"), ("cart_pole", "delay=2"), ("spring_chain", "delay=1"),
                                        ("double_well", "delay=1"), ("advection_diffusion", "delay=2"), ("gray_scott", "delay=1")])
def test_plan_rollout_keeps_queued_commands_under_delay(use, name, knobs):
    """S25a 修改（Codex 审核问题 2）：带执行延迟时，规划的续算要带上已下达、还没执行的命令。
    (1) 从起点用这条轨迹自己的下达动作续算 = 数据集本身（逐位）；(2) 分两段执行（更新已下达命令）= 一次执行（逐位）；
    (3) 起点前的命令丢掉（原来的全零命令表）时终点不同（说明 (1) 确实在核延迟队列）。"""
    use(name, knobs); plan = _tool("plan_v7"); S = systems.get(); v = datasets.active().knobs
    field = name == "advection_diffusion"
    tid, k = 181, 8
    full = S.variant_fields(tid, v) if field else S.variant_trajectory(tid, v)
    own = _per_tick(systems.commanded_controls(tid))
    start = next(t for t in range(4, 24) if np.abs(own[t-v.delay:t]).max() > .05)   # 起点前排队的命令不是零
    history = own.copy(); history[start:] = 0.
    one = plan.rollout(S, v, field, tid, full[start], start, own[start:start+k], history)
    assert _same(one, full[start+k], field)
    acts = np.random.default_rng(0).uniform(-.5, .5, (k, 2))
    whole = plan.rollout(S, v, field, tid, full[start], start, acts, history)
    h = history.copy()
    mid = plan.rollout(S, v, field, tid, full[start], start, acts[:3], h); h[start:start+3] = acts[:3]
    assert _same(plan.rollout(S, v, field, tid, mid, start+3, acts[3:], h), whole, field)
    assert not np.allclose(plan.rollout(S, v, field, tid, full[start], start, own[start:start+k], np.zeros((32, 2))), full[start+k],
                           rtol=0, atol=1e-6)


def test_plan_episode_commands():
    plan = _tool("plan_v7")
    hist = np.arange(64.).reshape(32, 2)/100
    out = plan.episode_commands(hist, 5, np.array([[.7, -.1], [.2, -.9]]))
    assert np.array_equal(out[:5], hist[:5]) and np.array_equal(out[5:7], [[.5, -.1], [.2, -.5]]) and not out[7:].any()


def _plan(tmp_path, system, *extra):
    job = tmp_path/f"job-{system}"; job.mkdir(exist_ok=True)
    (job/"job.json").write_text(json.dumps(dict(seed=1703, updates=2000, dataset=f"{system}-200", condition="identity")))
    out = tmp_path/f"{system}.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")} | dict(OMP_NUM_THREADS="2")
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/plan_v7.py"), "--job", str(job), "--fake-encoder", "--device", "cpu",
                        "--episodes", "1", "--output", str(out), *extra], env=env, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize("system,summary,px", [("pendulum", "摆锤", 80.), ("cart_pole", "小车和摆锤", 36.),
                                               ("bouncing_balls", "受控球", 48.), ("spring_chain", "受控质点", 48.)])
def test_plan_runs_on_second_order_particles(tmp_path, system, summary, px):
    rep = _plan(tmp_path, system, "--horizon", "8", "--replan", "4", "--samples", "8", "--iterations", "1", "--oracle-samples", "4")
    assert rep["state_summary"].startswith(summary) and rep["tubelet"] == "dup" and rep["dataset"] == f"{system}-200"
    ep = rep["episodes"][0]
    assert all(np.isfinite(ep[k]) for k in ("zero", "model", "oracle", "random"))
    assert ep["goal_valid"]
    assert ep["zero"] >= (.03 if system == "spring_chain" else .15)
    assert rep["model"]["mean_error_px"] == pytest.approx(rep["model"]["mean_error"]*px)


def test_plan_runs_on_wave_with_the_relative_gap(tmp_path):
    rep = _plan(tmp_path, "wave", "--horizon", "8", "--replan", "8", "--samples", "4", "--iterations", "1", "--no-baselines")
    assert rep["state_summary"].startswith("u 的 16×16") and rep["model"]["mean_error_px"] is None


def test_goal_gap_threshold_per_scene():
    plan = _tool("plan_v7")
    assert plan.GOAL_GAP == {"spring_chain": .03}
    z = np.zeros(2)
    assert plan.goal_far_enough(np.array([.03, 0.]), z, False, .03) and not plan.goal_far_enough(np.array([.029, 0.]), z, False, .03)
    assert not plan.goal_far_enough(np.array([.1, 0.]), z, False)          # 默认仍是 0.15


# —— 评测协议 ——

def test_second_order_protocol_levels_and_applicability():
    assert protocols.SCENES == ("double_well", "van_der_pol", "multi_particle", "advection_diffusion",
                                "landscape~shape=multi_well", "gray_scott", "push")             # 默认任务表的场景不变
    assert protocols.SECOND_ORDER_SCENES == SECOND and protocols.ALL_SCENES == protocols.SCENES+SECOND
    assert protocols.BY_ID["P1"].levels["pendulum"] == {"轻": "g_over_l=12", "标准": "g_over_l=14", "重": "g_over_l=16"}
    assert protocols.BY_ID["P2"].levels["bouncing_balls"] == {"轻": "restitution=0.8", "标准": "restitution=0.7", "重": "restitution=0.5"}
    for name in SECOND:                       # 档位都在参数范围里（parse 会核），且三档单调远离默认值
        for pid in ("P1", "P2"):
            lv = protocols.BY_ID[pid].levels[name]
            vals = [dict(variants.parse(name, lv[l]).params) for l in protocols.LEVELS]
            (key,) = vals[0]
            default = variants.PARAMS[name][key][0]
            gaps = [abs(v[key]-default) for v in vals]
            assert gaps[0] > 0 and gaps[0] < gaps[1] < gaps[2]
    missing = {s: {p.id for p in protocols.PROTOCOLS if not protocols.tests(s, protocols.LEVELS, [p.id])} for s in SECOND}
    # 二阶场景都不接受过程噪声、采集策略；wave 是场，没有光点
    assert missing == {"pendulum": {"P3", "P7"}, "bouncing_balls": {"P3", "P7"}, "cart_pole": {"P3", "P7"},
                       "spring_chain": {"P3", "P7"}, "wave": {"P3", "P7", "P9", "P10"}}
    assert [len(protocols.tests(s)) for s in SECOND] == [11, 11, 9, 11, 11]
    assert sum(len(protocols.tests(s)) for s in protocols.SCENES) == 82                         # S21b 的 82 格不变
    for s in SECOND:
        for level in protocols.LEVELS:
            parts = [protocols.BY_ID[i].levels[s].get(level) for i in ("P1", "P4", "P8")]
            assert protocols.BY_ID["P13"].levels[s][level] == ",".join(parts)
    assert protocols.BY_ID["P13"].levels["wave"]["标准"] == "speed=0.25,gain=2,background=checker,color=white"


def test_every_second_order_protocol_cell_generates(use):
    """每个适用的格（三档）都能生成：第一条测试轨迹不发散、不出画面（second_order.check 会报错）。完整扫描（每格 20 条）见
    项目文件夹 po-jepa-s25a/scan.py。"""
    for s in SECOND:
        for pid, level, t in protocols.tests(s, protocols.LEVELS):
            use(s, protocols.merged(s, t))
            states, frames = systems.get().frames(180)
            assert np.isfinite(states).all() and frames.shape == (33, 256, 256, 3), (s, pid, level)


# —— v8 任务表 ——

def _jobs(*args):
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/make_v8_jobs.py"), *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_v8_jobs_default_is_unchanged_and_second_order_is_opt_in():
    default = _jobs("--stage", "full")
    assert not any(f"--system {s} " in l for l in default for s in SECOND) and not any("二阶" in l for l in default[:1])
    lines = _jobs("--stage", "full", "--scenes", ",".join(SECOND))
    assert lines[0].startswith("# 注意：二阶场景") and "dup" in lines[0]
    cmds = [l for l in lines if l and not l.startswith("#")]
    assert sum("probe_actions.py" in l for l in cmds) == 5 and sum("run.py run" in l for l in cmds) == 5*3*6
    var = [l for l in _jobs("--stage", "variants", "--scenes", ",".join(SECOND)) if l and not l.startswith("#")]
    plan = [l for l in var if "plan_v7.py" in l]
    assert len(plan) == 5*6 and sum("run.py transfer" in l for l in var) == 6*sum(len(protocols.tests(s)) for s in SECOND)
    mk = _tool("make_v8_jobs"); plan_v7 = _tool("plan_v7")
    for l in plan:
        if "--no-baselines" in l:
            continue
        system = l.split("--job ")[1].split("/")[3]
        assert ("--oracle-samples 32" in l) == (system == "wave") and "--oracle-samples" not in l.replace("--oracle-samples 32", "")
        argv = l.split()[2:]; old = sys.argv; sys.argv = ["plan_v7.py", *argv]
        try:
            plan_v7._parse()
        finally:
            sys.argv = old
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/make_v8_jobs.py"), "--stage", "full", "--scenes", "pendulum~n=2"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
