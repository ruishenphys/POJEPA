"""S21b：gray_scott、push 接反事实探针和规划（续算接口）、单帧重画的像素噪声、评测协议、v8 任务表。

基础场景与已有调节项的数据逐位不变由跨树对照核（docs/segments/S21b.md §5.2）；这里核新接口本身和接线。
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from po_jepa import datasets, new_systems, protocols, simulation, systems, variants

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


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"tools"/f"{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _controls(tid, values, start):
    starts = np.arange(32, dtype=np.int64); vals = np.zeros((32, 2)); vals[start:start+len(values)] = values
    return simulation.TrajectoryControls(tid, starts, starts+1, vals)


# —— 续算接口 ——

@pytest.mark.parametrize("knobs", ["", "gain=2,delay=1,noise=0.1,mu_s=0.5", "pixel_noise=10,movers=2,occluders=1,saturation=0.3"])
def test_push_restart_is_bitwise_and_redraw_matches(use, knobs):
    use("push", knobs); S = systems.get(); v = datasets.active().knobs
    states, frames = S.frames(3)
    full = S.variant_trajectory(3, v)
    assert np.array_equal(full, states) and np.array_equal(S.hidden(full), states)
    assert np.array_equal(S.draw(full, 3, v), frames)
    for start, k in ((0, 4), (5, 8), (20, 12), (31, 1), (32, 0)):
        again = S.variant_trajectory(3, v, start=start, state=full[start], ticks=k)
        assert np.array_equal(again, full[start:start+k+1])
        assert np.array_equal(S.draw(again[-1:], 3, v, ticks=[start+k])[0], frames[start+k])


def test_push_counterfactual_uses_the_new_actions(use):
    use("push"); S = systems.get(); v = datasets.active().knobs
    full = S.variant_trajectory(3, v)
    d = full[0, 2:4]-full[0, :2]; u = np.tile(.5*d/np.linalg.norm(d), (16, 1))
    q = S.variant_trajectory(3, v, controls=_controls(3, u, 0), start=0, state=full[0], ticks=16)
    impl = new_systems.SYSTEMS["push"]
    assert np.array_equal(q, impl.trajectory(3, np.concatenate((u, np.zeros((16, 2)))), {}, state=full[0], ticks=16))
    assert np.linalg.norm(q[-1, 2:4]-full[0, 2:4]) > .05          # 朝方块推 4 秒，方块被推动


def test_gray_scott_restart_is_bitwise_and_redraw_matches(use):
    use("gray_scott", "pixel_noise=10,occluders=1,gain=2"); S = systems.get(); v = datasets.active().knobs
    states, frames = S.frames(2)
    full = S.variant_trajectory(2, v)
    assert full.shape == (33, 2*128*128+2) and np.array_equal(S.hidden(full), states)
    impl = new_systems.SYSTEMS["gray_scott"]
    U, V, p = impl.unpack(full)
    assert np.array_equal(impl.pack(U, V, p), full) and np.array_equal(impl.states(U, V, p), states)
    for start, k in ((0, 2), (17, 3)):
        again = S.variant_trajectory(2, v, start=start, state=full[start], ticks=k)
        assert np.array_equal(again, full[start:start+k+1])
        assert np.array_equal(S.draw(again[-1:], 2, v, ticks=[start+k])[0], frames[start+k])
    # 换动作：注入点按新动作走（dp/dt = 0.4·u，周期回绕）
    q = S.variant_trajectory(2, v, controls=_controls(2, np.full((2, 2), .5), 10), start=10, state=full[10], ticks=2)
    p_end = impl.unpack(q[-1:])[2][0]
    assert np.allclose(p_end, (p[10]+2*.4*2*.5*.25) % 1., atol=1e-12)


def test_new_system_restart_guards(use):
    use("push"); S = systems.get(); v = datasets.active().knobs
    full = S.variant_trajectory(1, v)
    with pytest.raises(ValueError):
        S.variant_trajectory(1, variants.parse("push", "gain=2"))      # 调节项与当前数据集不一致
    with pytest.raises(ValueError):
        S.variant_trajectory(1, v, start=5, ticks=2)                    # tick>0 不给状态
    for start, k in ((30, 3), (-1, 2), (5, -1)):
        with pytest.raises(ValueError):
            S.variant_trajectory(1, v, start=start, state=full[0], ticks=k)
    with pytest.raises(ValueError):
        systems.REGISTRY["gray_scott"].variant_trajectory(1, v)          # 当前数据集不是 gray_scott
    with pytest.raises(ValueError):
        new_systems.SYSTEMS["gray_scott"].unpack(np.zeros((1, 514)))


# —— 单帧重画的像素噪声 ——

def test_pixel_noise_full_draw_is_the_old_single_draw_and_partial_takes_the_same_tick():
    g = lambda: simulation._rng(4, 4)
    shape = (33, 8, 8, 3)
    assert np.array_equal(systems.pixel_noise(g(), 10., shape), g().normal(0., 10., shape))
    full = g().normal(0., 10., shape)
    assert np.array_equal(systems.pixel_noise(g(), 10., (2, 8, 8, 3), [7, 30]), full[[7, 30]])
    with pytest.raises(ValueError):
        systems.pixel_noise(g(), 10., (1, 8, 8, 3))


@pytest.mark.parametrize("system,knobs", [("double_well", "pixel_noise=10"), ("advection_diffusion", "pixel_noise=10,occluders=1"),
                                          ("landscape", "shape=ring,pixel_noise=5")])
def test_v7_scene_single_frame_redraw_has_that_ticks_noise(use, system, knobs):
    use(system, knobs); S = systems.get(); v = datasets.active().knobs
    states, frames = S.frames(6)
    if system == "advection_diffusion":
        f = S.variant_fields(6, v)
        assert np.array_equal(S.draw_fields(f[9:10], 6, v, ticks=[9])[0], frames[9])
        with pytest.raises(ValueError):
            S.draw_fields(f[9:10], 6, v)
    else:
        assert np.array_equal(S.draw(states[9:10], 6, v, ticks=[9])[0], frames[9])
        with pytest.raises(ValueError):
            S.draw(states[9:10], 6, v)


# —— 反事实探针 ——

def test_probe_counterfactual_end_on_push(use):
    use("push", "pixel_noise=5", 200); probe = _tool("probe_actions"); S = systems.get()
    states, frames = S.frames(181)
    true_end, cf, frame = probe.counterfactual_end(S, simulation, 181, 182, 5, 8)
    assert np.array_equal(true_end, states[13])
    v = datasets.active().knobs
    expect = S.variant_trajectory(181, v, controls=systems.commanded_controls(182), start=5, state=states[5], ticks=8)[-1]
    assert np.array_equal(cf, expect) and np.array_equal(frame, S.draw(expect[None], 181, v, ticks=[13])[0])
    assert not np.array_equal(cf, true_end)
    assert probe.motion("push", cf, true_end, 1.) == pytest.approx(48*np.linalg.norm(cf[:4]-true_end[:4]))


def test_probe_motion_definitions():
    probe = _tool("probe_actions")
    a = np.zeros(514); b = np.zeros(514); b[512:] = .9; b[3] = 2.
    assert probe.motion("gray_scott", a, b, 1.) == 2.                   # 池化场的 L2，注入点不计入
    a = np.zeros(5); b = np.array([0., 0., .3, .4, 1.])
    assert probe.motion("push", a, b, 1.) == pytest.approx(24.)          # 位置 L2 × 48，朝向角不计入
    assert probe.motion("double_well", a[:2], b[2:4], 48.) == pytest.approx(24.)


# —— 规划 ——

def test_plan_summaries_and_goal_actions(use):
    plan = _tool("plan_v7")
    use("push", "size=1"); S = systems.get(); v = datasets.active().knobs
    q = np.array([.1, .2, .3, .4, .5])
    assert np.array_equal(plan.summary("push", S, v, q), [.3, .4, .25])               # 方块位姿，θ 乘 size/2
    assert np.array_equal(plan.oracle_target("push", S, v, q), [.1, .2, .3, .4, .25])
    rng = np.random.default_rng(0)
    u = plan.goal_controls("push", np.array([0., 0., 1., 1., 0.]), 7, rng)
    assert u.shape == (7, 2) and np.array_equal(u[0], u[1]) and np.array_equal(u[4], u[5])
    norms = np.linalg.norm(u, axis=1); angles = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
    assert (norms >= .25-1e-12).all() and (norms <= .5+1e-12).all() and (np.abs(angles-45) <= 30+1e-9).all()
    # 其他场景：和 S21b 之前完全相同的抽法（每 2 tick 一段的均匀随机）
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    assert np.array_equal(plan.goal_controls("double_well", q, 7, r1), np.repeat(r2.uniform(-.5, .5, (4, 2)), 2, 0)[:7])
    use("gray_scott"); S = systems.get(); v = datasets.active().knobs
    impl = new_systems.SYSTEMS["gray_scott"]
    U, V, p = impl.initial(0)
    full = impl.pack(U[None], V[None], p[None])[0]
    assert np.array_equal(plan.summary("gray_scott", S, v, full), impl.states(U[None], V[None], p[None])[0][:512])
    assert np.array_equal(plan.oracle_target("gray_scott", S, v, full), plan.summary("gray_scott", S, v, full))
    assert np.array_equal(plan.summary("van_der_pol", None, None, q), q[:2])


# —— 规划：用 identity 条件和假编码器走通一个回合 ——

def _plan(tmp_path, system, *extra, ok=True):
    job = tmp_path/f"job-{system}"; job.mkdir(exist_ok=True)
    (job/"job.json").write_text(json.dumps(dict(seed=1703, updates=2000, dataset=f"{system}-200", condition="identity")))
    out = tmp_path/f"{system}.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")} | dict(OMP_NUM_THREADS="2")
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/plan_v7.py"), "--job", str(job), "--fake-encoder", "--device", "cpu",
                        "--episodes", "1", "--output", str(out), *extra], env=env, capture_output=True, text=True, cwd=ROOT)
    if not ok:
        return r
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


def test_plan_runs_on_push_and_scores_the_block(tmp_path):
    rep = _plan(tmp_path, "push", "--horizon", "8", "--replan", "4", "--samples", "16", "--iterations", "2",
                "--oracle-samples", "16")
    assert rep["state_summary"].startswith("方块位姿") and rep["dataset"] == "push-200"
    ep = rep["episodes"][0]
    assert ep["goal_valid"] and ep["zero"] >= .15 and rep["goals"] == dict(valid=1, invalid=0, rule=rep["goals"]["rule"])
    assert all(np.isfinite(ep[k]) for k in ("zero", "model", "oracle", "random"))
    assert rep["model"]["mean_error_px"] is not None


def test_plan_refuses_to_score_goals_that_never_pass_the_gap(tmp_path):
    """Codex 审核 P2：push 时域 2 时 20 次重抽的目标都没推动方块（gap = 0），不能拿它评分。"""
    r = _plan(tmp_path, "push", "--horizon", "2", "--replan", "2", "--samples", "4", "--iterations", "1", "--no-baselines", ok=False)
    assert r.returncode != 0 and "目标都没达到门槛" in r.stderr


def test_goal_gap_rule():
    plan = _tool("plan_v7")
    z = np.zeros(3)
    assert plan.goal_far_enough(np.array([.15, 0., 0.]), z, False) and not plan.goal_far_enough(np.array([.149, 0., 0.]), z, False)
    g = np.full(400, 1.); zf = g.copy(); zf[0] -= .05*20*1.01
    zf2 = g.copy(); zf2[0] -= .5                                  # 差 0.5 = 2.5% 范数，不够
    assert plan.goal_far_enough(g, zf, True) and not plan.goal_far_enough(g, zf2, True)


def test_plan_runs_on_gray_scott(tmp_path):
    rep = _plan(tmp_path, "gray_scott", "--horizon", "8", "--replan", "8", "--samples", "4", "--iterations", "1",
                "--no-baselines")
    assert rep["state_summary"].startswith("U、V") and rep["model"]["mean_error_px"] is None
    assert rep["episodes"][0]["goal_valid"] and np.isfinite(rep["episodes"][0]["model"])


# —— 评测协议 ——

def test_v7_test_variants_are_the_standard_level():
    v7 = _tool("make_v7_jobs")
    assert set(v7.BASE_SCENES) <= set(protocols.SCENES)
    for scene in v7.BASE_SCENES:
        standard = [t for _, _, t in protocols.tests(scene)]
        assert set(v7.TEST_VARIANTS[protocols.system_of(scene)]) <= set(standard)
        for t in v7.TEST_VARIANTS[protocols.system_of(scene)]:     # 目录名（规范字符串）也相同，v7 编码的变体测试集能复用
            assert protocols.merged(scene, t) == v7.merged(scene, t)


def test_protocols_parse_and_applicability_follow_validate():
    for p in protocols.PROTOCOLS:
        assert set(p.levels) <= set(datasets.SYSTEMS)
        for scene in protocols.SCENES:
            for level in protocols.LEVELS:
                text = p.levels.get(protocols.system_of(scene), {}).get(level)
                ok = protocols.applicable(p, scene, level)
                if text is None:
                    assert not ok
                    continue
                try:
                    protocols.merged(scene, text); valid = True
                except ValueError:
                    valid = False
                assert ok == valid
                if ok:
                    datasets.DatasetSpec(protocols.system_of(scene), 2000, protocols.merged(scene, text))
    # 每个场景只缺 validate 不允许的协议
    missing = {s: {p.id for p in protocols.PROTOCOLS if not protocols.tests(s, protocols.LEVELS, [p.id])} for s in protocols.SCENES}
    assert missing["double_well"] == missing["landscape~shape=multi_well"] == set()
    assert missing["advection_diffusion"] == missing["gray_scott"] == {"P3", "P7", "P9", "P10"}
    assert missing["push"] == {"P3"}
    # P13 = P1 + P4 + P8 的同一档；每个通用协议只动一个类别
    for s in protocols.SCENES:
        system = protocols.system_of(s)
        for level in protocols.LEVELS:
            parts = [protocols.BY_ID[i].levels[system].get(level) for i in ("P1", "P4", "P8")]
            if None not in parts:
                assert protocols.BY_ID["P13"].levels[system][level] == ",".join(parts)
    counts = [len(protocols.tests(s)) for s in protocols.SCENES]
    assert counts == [13, 13, 13, 9, 13, 9, 12]


def test_protocol_table_in_dataset_md_is_generated():
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/protocol_table.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert protocols.table() in (ROOT/"DATASET.md").read_text(encoding="utf-8")


# —— v8 任务表 ——

def _jobs(*args):
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/make_v8_jobs.py"), *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return [l for l in r.stdout.splitlines() if l and not l.startswith("#")]


def test_v8_variants_stage_covers_every_standard_test():
    lines = _jobs("--stage", "variants")
    v1 = [l for l in lines if "encode_v7.py encode" in l]; v3 = [l for l in lines if "run.py transfer" in l]
    plan = [l for l in lines if "plan_v7.py" in l]
    total = sum(len(protocols.tests(s)) for s in protocols.SCENES)
    assert len(v1) == 4*total and len(v3) == 6*total and len(plan) == 7*6
    for s in protocols.SCENES:
        system = protocols.system_of(s)
        for _, _, t in protocols.tests(s):
            knobs = protocols.merged(s, t); folder = f"../features-v8/{system}~{knobs}-2000"
            assert sum(f"--variant {knobs} " in l and f"--output {folder} " in l for l in v1) == 4
            assert sum(f"--system {system} " in l and f"--test-variant {t} " in l and f"--features-root {folder} " in l for l in v3) == 6
    oracle = [l for l in plan if "--no-baselines" not in l]
    assert len(oracle) == 7 and all("identity" in l for l in oracle)
    assert sum("--oracle-samples 16" in l for l in oracle) == 1 and sum("--oracle-samples 32" in l for l in oracle) == 1
    plan_v7 = _tool("plan_v7")
    for l in plan:                                   # 每条规划命令过 plan_v7 的参数解析
        argv = l.split()[2:]
        old = sys.argv; sys.argv = ["plan_v7.py", *argv]
        try:
            plan_v7._parse()
        finally:
            sys.argv = old


def test_v8_strength_stage_is_light_and_heavy_only():
    lines = _jobs("--stage", "strength")
    v1 = [l for l in lines if "encode_v7.py encode" in l]
    want = sum(len(protocols.tests(s, ("轻", "重"))) for s in protocols.SCENES)
    assert len(v1) == 4*want and not any("plan_v7.py" in l for l in lines)
    got = {(l.split("--system ")[1].split()[0], l.split("--variant ")[1].split()[0]) for l in v1}
    assert got == {(protocols.system_of(s), protocols.merged(s, t)) for s in protocols.SCENES
                   for _, _, t in protocols.tests(s, ("轻", "重"))}
    standard = {(protocols.system_of(s), protocols.merged(s, t)) for s in protocols.SCENES for _, _, t in protocols.tests(s)}
    assert not got & standard                                  # 标准档在 variants 阶段已经评过
    subset = _jobs("--stage", "strength", "--protocols", "P4", "--scenes", "push", "--conditions", "identity")
    assert [l.split("--test-variant ")[1].split()[0] for l in subset if "run.py transfer" in l] == ["gain=1.5", "gain=3"]


def test_v8_full_stage_has_all_scenes_and_probes():
    lines = _jobs("--stage", "full")
    probes = [l for l in lines if "probe_actions.py" in l]
    assert len(probes) == 7 and any("--system landscape --variant shape=multi_well" in l for l in probes)
    runs = [l for l in lines if "run.py run" in l]
    assert len(runs) == 7*3*6 and sum("--system push " in l for l in runs) == 18
    pilot = _jobs("--stage", "pilot")
    assert not any("probe_actions.py" in l for l in pilot) and sum("run.py run" in l for l in pilot) == 7*6


def test_v7_job_table_is_untouched():
    """v7 正式实验和仿真补跑都读 make_v7_jobs：S21b 不改它。核文件的 SHA-256（42f6d1f、87d09d9 上的同一份），不只核数量。"""
    import hashlib
    v7 = _tool("make_v7_jobs")
    assert hashlib.sha256((ROOT/"tools/make_v7_jobs.py").read_bytes()).hexdigest() == \
        "e9a7ac907d8e2785634c5dd8b1adf812d2cf55fe17fa622494ec01abdee4e5d7"
    assert len(v7.BASE_SCENES) == 5 and "push" not in v7.TEST_VARIANTS and sum(map(len, v7.TEST_VARIANTS.values())) == 30
