"""S25b5（Jerry 2026-11-25 定，新口径）：二阶场景的规划成功判据并入速度、读出任务加上速度。

规划：成功 ⇔ max(e_q, τ·e_v) ≤ --success × e_q0，τ = 1 tick = 0.25 s；oracle 代价 √(e_q² + τ²·e_v²)；一阶场景、push 不变。
读出：位置探针不变，另拟合速度探针（velocity/ 子目录），滚动和反事实多出 vel_ 开头的指标。
期望值都按公式独立写出（摆锤速度 θ̇·(cos θ, sin θ)、小车倒立摆的雅可比、扰动差分），不从被测函数取。
"""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from po_jepa import configuration, datasets, systems, variants
from po_jepa.baselines import IdentityPredictor
from po_jepa.runtime import long_rollout, state_readout, v7jobs
from test_s25c_long_horizon import ActionToy, _clean_env, active, real, toy_windows  # noqa: F401
from test_training import e06_config

ROOT = Path(__file__).resolve().parents[1]
SECOND = ("pendulum", "cart_pole", "spring_chain", "bouncing_balls", "wave")
PAIR = datasets.parse("pendulum~tubelet=pair:40")
PAIR_LONG = datasets.parse("pendulum~horizon=64,tubelet=pair:40")
VAL_IDS, TEST_IDS = tuple(range(32, 36)), tuple(range(36, 40))


def _tool(name):
    spec = importlib.util.spec_from_file_location(f"{name}_s25b5", ROOT/f"tools/{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _pend_v(h):
    """摆锤 (sin θ, −cos θ) 的时间导数，手写。"""
    return h[1]*np.array([math.cos(h[0]), math.sin(h[0])])


def _cart_v(h, l):
    x, xd, th, om = h
    return np.array([xd, xd+l*om*math.cos(th), -l*om*math.sin(th)])


# —— 速度的定义 ——

def test_second_order_set_and_constants():
    assert set(variants.SECOND_ORDER) == set(SECOND)
    assert {n for n in SECOND if state_readout.has_velocity(n)} == set(SECOND)
    for n in ("double_well", "van_der_pol", "multi_particle", "advection_diffusion", "landscape", "gray_scott", "push"):
        assert not state_readout.has_velocity(n)
    plan = _tool("plan_v7")
    assert plan.VELOCITY_SECONDS == .25 == systems.PERIOD             # τ = 1 tick（两帧编码的间隔）
    assert state_readout.velocity_units("pendulum") == "px_per_s" and state_readout.velocity_units("wave") == "field_l2_per_s"


@pytest.mark.parametrize("name,params,dim", [("pendulum", {}, 2), ("cart_pole", {"length": 1.4}, 4),
                                             ("spring_chain", {"n": 3}, 12), ("bouncing_balls", {}, 12)])
def test_velocity_is_the_time_derivative_of_the_configuration(name, params, dim):
    """扰动差分：沿速度方向移动位置坐标 ±ε（速度坐标不动），构形的中心差分 / 2ε = velocity。"""
    g = np.random.default_rng(7)
    for _ in range(5):
        h = g.uniform(-1.5, 1.5, dim)
        dh = np.zeros(dim)
        if name == "pendulum":
            dh[0] = h[1]
        elif name == "cart_pole":
            dh[0], dh[2] = h[1], h[3]
        else:
            dh[:dim//2] = h[dim//2:]
        eps = 1e-6
        num = (configuration.configuration(name, h+eps*dh, params)-configuration.configuration(name, h-eps*dh, params))/(2*eps)
        np.testing.assert_allclose(configuration.velocity(name, h, params), num, rtol=0, atol=1e-7)
    h = np.array([.4, -2., .9, 3.][:dim] if dim <= 4 else np.arange(dim, dtype=float))
    if name == "pendulum":
        np.testing.assert_allclose(configuration.velocity(name, h, params), _pend_v(h), rtol=0, atol=1e-15)
    if name == "cart_pole":
        np.testing.assert_allclose(configuration.velocity(name, h, params), _cart_v(h, 1.4), rtol=0, atol=1e-15)
        assert not np.allclose(configuration.velocity(name, h, {}), _cart_v(h, 1.4))      # 摆长跟参数走
    if name in ("spring_chain", "bouncing_balls"):
        assert np.array_equal(configuration.velocity(name, h, params), np.arange(6., 12.))
        assert np.array_equal(configuration.controlled_velocity(name, h, params), [6., 7.])     # 受控的第 0 个


def test_wave_velocity_is_the_pooled_u_t(active):
    active(datasets.parse("wave:200"))
    S = systems.get(); v = datasets.active().knobs
    full = S.variant_trajectory(3, v)[9]
    n = 128
    ut = full[n*n:2*n*n].reshape(16, 8, 16, 8).mean((1, 3)).reshape(-1)        # 整场 u_t 直接池化
    h = S.hidden(full[None])[0]
    np.testing.assert_allclose(configuration.velocity("wave", h, {}), ut, rtol=0, atol=1e-12)
    assert np.array_equal(configuration.controlled_velocity("wave", h, {}), configuration.velocity("wave", h, {}))
    assert not np.allclose(configuration.velocity("wave", h, {}), configuration.configuration("wave", h, {}))


def test_velocity_target_scaling():
    h = np.array([.4, -2.])
    np.testing.assert_allclose(state_readout.velocity_target("pendulum", h, {}), 80.*_pend_v(h), rtol=0, atol=1e-13)
    hc = np.array([.2, .5, 1., -1.])
    np.testing.assert_allclose(state_readout.velocity_target("cart_pole", hc, {}), 36.*_cart_v(hc, 1.), rtol=0, atol=1e-13)
    assert np.array_equal(state_readout.velocity_target("bouncing_balls", np.arange(12.), {}), 48.*np.arange(6., 12.))
    hw = np.r_[np.zeros(256), np.arange(256.), 0., 0.]
    assert np.array_equal(state_readout.velocity_target("wave", hw, {}), np.arange(256.))       # 场：不乘
    with pytest.raises(ValueError, match="没有速度"):
        state_readout.velocity_target("double_well", np.zeros(2), {})
    np.testing.assert_allclose(state_readout.velocity_targets("pendulum", np.array([h, h+1]), {}),
                               80.*np.stack([_pend_v(h), _pend_v(h+1)]), rtol=0, atol=1e-13)


# —— 规划判据 ——

def test_success_rule_cases():
    plan = _tool("plan_v7")
    s = lambda q, v, q0: bool(plan.success(np.array([q]), None if v is None else np.array([v]), np.array([q0]), .5)[0])
    assert s(.4, None, 1.) and not s(.6, None, 1.) and s(.5, None, 1.)            # 没有速度：旧判据（边界算成功）
    assert s(.4, 1.6, 1.)                         # τ·e_v = 0.4 ≤ 0.5
    assert not s(.4, 2.4, 1.)                     # 到了位置、速度不对：失败
    assert not s(.6, 0., 1.)                      # 速度对、位置不对：失败
    assert s(.5, 2., 1.)                          # 两条都在边界上
    assert not s(.4, 2.4, 1.) and s(.4, 2.4, 1.3)  # 速度容差跟“不动”的位置误差走，不跟“不动”的速度误差走
    assert not s(1e-9, 0., 0.) and s(5e-10, 0., 0.)   # e_q0 下限 1e-9（与以前相同）：容差 5e-10
    out = plan.success(np.array([.1, .1, .9]), np.array([.1, 3., .1]), np.ones(3), .5)
    assert out.tolist() == [True, False, False]


def _hidden_state(name, S, v, state):
    return S.hidden(np.asarray(state)[None])[0]


@pytest.mark.parametrize("name,knobs", [("pendulum", ""), ("cart_pole", "length=1.4"), ("spring_chain", "n=3"),
                                        ("bouncing_balls", ""), ("wave", "")])
def test_oracle_cost_is_the_phase_space_distance(active, name, knobs):
    plan = _tool("plan_v7")
    active(datasets.parse(f"{name}{'~'+knobs if knobs else ''}:200"))
    S = systems.get(); v = datasets.active().knobs; p = dict(v.params)
    traj = S.variant_trajectory(5, v)
    a, b = traj[6], traj[13]
    ha, hb = _hidden_state(name, S, v, a), _hidden_state(name, S, v, b)
    dq = configuration.controlled(name, ha, p)-configuration.controlled(name, hb, p)
    dv = configuration.controlled_velocity(name, ha, p)-configuration.controlled_velocity(name, hb, p)
    if name == "pendulum":                        # 独立写一遍速度
        dv_hand = _pend_v(ha)-_pend_v(hb)
        np.testing.assert_allclose(dv, dv_hand, rtol=0, atol=1e-12)
    d = np.linalg.norm(plan.oracle_target(name, S, v, a)-plan.oracle_target(name, S, v, b))
    assert d == pytest.approx(math.sqrt(dq@dq+.0625*(dv@dv)), rel=1e-12)
    assert np.linalg.norm(dv) > 0 and d > np.linalg.norm(dq)
    np.testing.assert_array_equal(plan.summary_velocity(name, S, v, a), configuration.controlled_velocity(name, ha, p))


def test_first_order_and_push_oracle_unchanged(active):
    plan = _tool("plan_v7")
    active(datasets.parse("double_well:200"))
    S = systems.get(); v = datasets.active().knobs
    st = np.array([.3, -.2])
    assert np.array_equal(plan.oracle_target("double_well", S, v, st), st) and plan.summary_velocity("double_well", S, v, st) is None
    active(datasets.parse("push:200"))
    S = systems.get(); v = datasets.active().knobs
    st = np.array([.1, .2, .3, .4, .5])
    np.testing.assert_allclose(plan.oracle_target("push", S, v, st), [.1, .2, .3, .4, .5*v.param("push", "size")*.5])


def _plan(tmp_path, dataset, *extra):
    job = tmp_path/f"job-{dataset}"; job.mkdir(exist_ok=True)
    (job/"job.json").write_text(json.dumps(dict(seed=1703, updates=2000, dataset=dataset, condition="identity")))
    out = tmp_path/f"{dataset}.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")} | dict(OMP_NUM_THREADS="2")
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/plan_v7.py"), "--job", str(job), "--fake-encoder", "--device", "cpu",
                        "--output", str(out), *extra], env=env, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


def test_plan_report_on_pendulum(tmp_path, active):
    rep = _plan(tmp_path, "pendulum-200", "--episodes", "3", "--horizon", "8", "--replan", "4", "--samples", "8",
                "--iterations", "1", "--oracle-samples", "4")
    assert "速度" in rep["success_rule"] and rep["success_metric"]["velocity_seconds"] == .25
    rows = [r for r in rep["episodes"] if r["goal_valid"]]
    assert rows
    for r in rep["episodes"]:
        assert set(r["velocity"]) == {"model", "oracle", "random", "zero"}
    for name in ("model", "oracle", "random"):
        e = np.array([r[name] for r in rows]); ev = np.array([r["velocity"][name] for r in rows]); z = np.array([r["zero"] for r in rows])
        assert rep[name]["success_rate"] == pytest.approx(float((np.maximum(e, .25*ev) <= .5*z).mean()))
        assert rep[name]["position_only_success_rate"] == pytest.approx(float((e <= .5*z).mean()))
        assert rep[name]["mean_velocity_error"] == pytest.approx(float(ev.mean()))
        assert rep[name]["mean_velocity_error_px_per_s"] == pytest.approx(80.*float(ev.mean()))
    assert rep["zero"]["mean_velocity_error"] == pytest.approx(float(np.mean([r["velocity"]["zero"] for r in rows])))
    # 第一个回合的“不动”和目标独立重做一遍：目标随机流 1704、同一套续算，速度按手写公式
    plan = _tool("plan_v7")
    active(datasets.parse("pendulum:200"))
    S = systems.get(); v = datasets.active().knobs; spec = datasets.active()
    tid = list(spec.split_range("test"))[0]; start = int(datasets.START_TICKS[0])
    assert (tid, start) == (rep["episodes"][0]["trajectory_id"], rep["episodes"][0]["start_tick"])
    s0 = S.variant_trajectory(tid, v)[start]
    hist = systems.command_ticks(systems.commanded_controls(tid)); hist[start:] = 0.
    zero = plan.rollout(S, v, False, tid, s0, start, np.zeros((8, 2)), hist)
    goal_rng = np.random.default_rng(1704)
    for _ in range(20):
        goal = plan.rollout(S, v, False, tid, s0, start, plan.goal_controls("pendulum", s0, 8, goal_rng), hist)
        if plan.goal_far_enough(plan.summary("pendulum", S, v, goal), plan.summary("pendulum", S, v, zero), False):
            break
    ep = rep["episodes"][0]
    assert ep["zero"] == pytest.approx(float(np.linalg.norm(plan.summary("pendulum", S, v, zero)-plan.summary("pendulum", S, v, goal))))
    assert ep["velocity"]["zero"] == pytest.approx(float(np.linalg.norm(_pend_v(zero)-_pend_v(goal))), rel=1e-12)


def test_plan_report_first_order_unchanged(tmp_path):
    rep = _plan(tmp_path, "double_well-200", "--episodes", "1", "--horizon", "8", "--replan", "8", "--samples", "8",
                "--iterations", "1", "--oracle-samples", "4")
    assert rep["success_rule"] == "误差 ≤ 0.5 × 不动时的误差" and "success_metric" not in rep
    assert "velocity" not in rep["episodes"][0] and "position_only_success_rate" not in rep["model"]
    assert set(rep["zero"]) == {"mean_error"}


# —— 读出：公式 ——

def _probe(o, seed, var=24.):
    g = torch.Generator().manual_seed(seed)
    return state_readout.Probe("mean", torch.randn(1408, o, generator=g), torch.zeros(1408), np.arange(o, dtype=np.float64), var, {})


def test_rollout_values_with_velocity(active):
    active(datasets.parse("double_well~horizon=64:40"))
    windows = toy_windows(64)
    p, vp = _probe(2, 1), _probe(3, 2, var=7.)
    times = long_rollout.curve_times(64)
    y = {t: np.array([1., 2.])*t for t in times}
    yv = {t: np.array([3., -1., .5])*t/8 for t in times}
    only = state_readout.rollout_values(ActionToy(), p, windows, y, 64)
    both = state_readout.rollout_values(ActionToy(), p, windows, y, 64, vp, yv)
    assert tuple(both) == state_readout.rollout_names(64, True) == state_readout.rollout_names(64)+tuple(
        "vel_"+k for k in state_readout.rollout_names(64))
    assert {k: both[k] for k in only} == only                       # 位置指标不受影响
    alone = state_readout.rollout_values(ActionToy(), vp, windows, yv, 64)    # 速度探针当成唯一的探针 = vel_ 那一半
    assert {k: both["vel_"+k] for k in alone} == alone
    # 再按定义核一个点（identity：读出 = R(z_0)）
    vi = state_readout.rollout_values(IdentityPredictor(), p, windows, y, 64, vp, yv)
    r0 = vp(windows[64.].inputs.raw_z)[0]
    assert vi["vel_readout:T=24"] == pytest.approx(float(np.linalg.norm(r0-yv[24.])), rel=1e-6)
    assert vi["vel_nmse:T=24"] == pytest.approx(vi["vel_readout:T=24"]**2/7., rel=1e-6)
    with pytest.raises(ValueError, match="一起给"):
        state_readout.rollout_values(ActionToy(), p, windows, y, 64, vp, None)


class Shift(torch.nn.Module):
    def forward(self, raw, controls, duration, grid):
        n = int(controls.n_segments[0])
        integral = float(((controls.ends[0, :n]-controls.starts[0, :n]).double()*controls.u_values[0, :n, 0].double().cpu()).sum())
        out = raw.clone(); out[0, :, 0] += integral
        return out


def test_counterfactual_values_and_summary_with_velocity(active):
    active(datasets.parse("double_well~horizon=64:40"))
    windows = toy_windows(64)
    w, donor = windows[16.], windows[8.]
    p, vp = _probe(2, 1), _probe(3, 5)
    ya, yb, va, vb = np.array([1., 2.]), np.array([4., 6.]), np.array([0., 1., 2.]), np.array([2., 1., 0.])
    only = state_readout.counterfactual_values(Shift(), p, w, donor, ya, yb)
    both = state_readout.counterfactual_values(Shift(), p, w, donor, ya, yb, vp, va, vb)
    assert {k: both[k] for k in only} == only
    alone = state_readout.counterfactual_values(Shift(), vp, w, donor, va, vb)
    assert {k: both["vel_"+k] for k in alone} == alone and len(both) == 2*len(only)
    assert both["vel_cf_true"] == pytest.approx(math.sqrt(8.))
    with pytest.raises(ValueError, match="一起给"):
        state_readout.counterfactual_values(Shift(), p, w, donor, ya, yb, vp, va, None)
    pos = [dict(cf_true=5., cf_pred=5., cf_error=0., cf_inner=25., actual_readout=1., actual_floor=0.),
           dict(cf_true=1., cf_pred=0., cf_error=1., cf_inner=0., actual_readout=3., actual_floor=2.)]
    vel = [dict(cf_true=2., cf_pred=1., cf_error=1., cf_inner=2., actual_readout=4., actual_floor=1.),
           dict(cf_true=3., cf_pred=3., cf_error=6., cf_inner=-9., actual_readout=2., actual_floor=3.)]
    rows = [dict(identical_controls=False, values={**a, **{"vel_"+k: x for k, x in b.items()}}) for a, b in zip(pos, vel)]
    big = dict(cf_true=0., cf_pred=9., cf_error=9., cf_inner=0., actual_readout=3., actual_floor=1.)
    rows.append(dict(identical_controls=True, values={**big, **{"vel_"+k: (100.*x if k != "cf_true" else 0.) for k, x in big.items()}}))
    s = state_readout.counterfactual_summary(rows)
    assert s["explained"] == pytest.approx(1-1/26) and s["gain"] == pytest.approx(25/26)        # 位置汇总不变
    sv = s["velocity"]
    assert sv["explained"] == pytest.approx(1-37/13) and sv["gain"] == pytest.approx(-7/13)
    assert sv["cos_mean"] == pytest.approx((1.-1.)/2) and sv["windows"] == 3 and sv["windows_with_different_actions"] == 2
    assert sv["actual_readout"] == pytest.approx((4.+2.+300.)/3)          # 均值取全部窗口；cos、explained、gain 不含动作相同的
    assert "velocity" not in state_readout.counterfactual_summary([dict(identical_controls=False, values=pos[0])])


def test_probe_load_checks_quantity(tmp_path):
    p = _probe(2, 1)
    p = state_readout.Probe(p.kind, p.weight, p.mean_x, p.mean_y, p.variance, dict(kind="mean", outputs=2, variance=24.))
    state_readout.save(p, tmp_path/"pos", dict(dataset="d", system="pendulum", quantity="position"))
    state_readout.save(p, tmp_path/"old", dict(dataset="d", system="pendulum"))            # S25b5 之前：没有 quantity
    state_readout.save(p, tmp_path/"vel", dict(dataset="d", system="pendulum", quantity="velocity"))
    state_readout.load(tmp_path/"pos", dataset_tag="d", system="pendulum")
    state_readout.load(tmp_path/"old", dataset_tag="d", system="pendulum")
    state_readout.load(tmp_path/"vel", dataset_tag="d", system="pendulum", quantity="velocity")
    with pytest.raises(ValueError, match="reads velocity"):
        state_readout.load(tmp_path/"vel", dataset_tag="d", system="pendulum")
    with pytest.raises(ValueError, match="reads position"):
        state_readout.load(tmp_path/"old", dataset_tag="d", system="pendulum", quantity="velocity")
    with pytest.raises(ValueError, match="fit"):
        state_readout.load(tmp_path/"missing", dataset_tag="d", system="pendulum", quantity="velocity")


# —— 读出：工具（pendulum~tubelet=pair，假编码器的缓存当成真实编码器读） ——

def _encode(out, variant, splits):
    args = ["--system", "pendulum", "--trajectories", "40", "--output", str(out), "--variant", variant]
    for mode, extra in (("encode", ["--fake-encoder", "--device", "cpu", "--splits", splits]), ("finalize", ["--splits", splits])):
        r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/encode_v7.py"), mode, *args, *extra],
                           env=_clean_env(), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]


@pytest.fixture(scope="module")
def pair_caches(tmp_path_factory):
    root = tmp_path_factory.mktemp("s25b5")
    _encode(root/"base", "tubelet=pair", "validation,test")
    _encode(root/"long", "horizon=64,tubelet=pair", "test")
    return root/"base", root/"long"


@pytest.fixture
def pair_readout(pair_caches, real, active, tmp_path, monkeypatch):
    base, long = pair_caches
    tool = _tool("state_readout")
    active(PAIR)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    args = ["--system", "pendulum", "--variant", "tubelet=pair", "--trajectories", "40", "--features", str(base)]
    tool.main(["fit", *args, "--kind", "mean", "--output", str(tmp_path/"probe"), "--device", "cpu"])
    tool.main(["truth", *args, "--long-features", str(long), "--horizon", "64", "--output", str(tmp_path/"truth.json")])
    active(PAIR)
    return tmp_path/"probe", tmp_path/"truth.json"


def _job(root, dataset=PAIR.tag):
    job = root/"job"/"run"/"job"; job.mkdir(parents=True)
    (job/"job.json").write_text(json.dumps(dict(seed=1703, updates=10000, dataset=dataset, condition="identity")))
    return root/"job"


def _fake_load(model=None):
    return lambda condition, training_run, device: ((IdentityPredictor() if model is None else model), e06_config(), None, dict(fake=True))


def test_fit_and_truth_write_velocity(pair_readout, active):
    probe, truth = pair_readout
    pm = json.loads((probe/"probe.json").read_text()); vm = json.loads((probe/"velocity"/"probe.json").read_text())
    assert (pm["quantity"], pm["units"], pm["velocity_probe"]) == ("position", "px", "velocity")
    assert (vm["quantity"], vm["units"], vm["tubelet"], vm["trajectories"]) == ("velocity", "px_per_s", "pair", list(VAL_IDS))
    assert vm["fit"]["samples"] == pm["fit"]["samples"] == 4*33 and vm["fit"]["outputs"] == 2
    assert vm["cache_identity"] == pm["cache_identity"] and vm["mean_x_sha256"] == pm["mean_x_sha256"]
    assert vm["position_probe"] == dict(weight_sha256=pm["weight_sha256"], mean_x_sha256=pm["mean_x_sha256"])
    active(PAIR)
    S = systems.get()
    vp, _ = state_readout.load(probe/"velocity", dataset_tag=PAIR.tag, system="pendulum", quantity="velocity")
    assert vp.mean_y.shape == (2,)
    t = json.loads(truth.read_text())
    assert t["velocity"] is True and t["velocity_units"] == "px_per_s"
    r = next(r for r in t["counterfactual"] if (r["trajectory_id"], r["start_tick"], r["duration_cycles"]) == (37, 10, 4.))
    states, _ = S.frames(37)
    np.testing.assert_allclose(r["v_actual"], 80.*_pend_v(states[14]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(r["y_actual"], 80.*np.array([math.sin(states[14][0]), -math.cos(states[14][0])]), rtol=0, atol=1e-12)
    assert r["v_counterfactual"] != r["v_actual"]
    # 反事实终点（S25f 已核过的续算路径 probe_actions.counterfactual_end）：速度和位置真值来自同一个状态，比例按手写公式
    sys.path.insert(0, str(ROOT/"tools"))
    import probe_actions
    from po_jepa import simulation
    _, cf_state, _ = probe_actions.counterfactual_end(S, simulation, 37, r["donor_id"], 10, 4)
    np.testing.assert_allclose(r["v_counterfactual"], 80.*_pend_v(cf_state), rtol=0, atol=1e-12)
    np.testing.assert_allclose(r["y_counterfactual"], 80.*np.array([math.sin(cf_state[0]), -math.cos(cf_state[0])]), rtol=0, atol=1e-12)
    active(PAIR_LONG)
    long_states, _ = systems.get().frames(39)
    active(PAIR)
    row = next(r for r in t["long"] if r["trajectory_id"] == 39)
    assert list(row["v"]) == list(row["y"]) == [f"{x}" for x in range(8, 65, 8)]
    np.testing.assert_allclose(row["v"]["56"], 80.*_pend_v(long_states[56]), rtol=0, atol=1e-12)


def test_position_probe_identical_to_position_only_fit(pair_caches, real, active, tmp_path, monkeypatch):
    """速度探针单独选 λ：位置探针和只拟合位置时逐位相同（把 has_velocity 关掉再拟合一次对照）。"""
    base, _ = pair_caches
    tool = _tool("state_readout")
    active(PAIR)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    args = ["--system", "pendulum", "--variant", "tubelet=pair", "--trajectories", "40", "--features", str(base), "--kind", "mean",
            "--device", "cpu"]
    tool.main(["fit", *args, "--output", str(tmp_path/"with")])
    monkeypatch.setattr(state_readout, "has_velocity", lambda name: False)
    active(PAIR)
    tool.main(["fit", *args, "--output", str(tmp_path/"without")])
    assert not (tmp_path/"without"/"velocity").exists()
    for f in ("weight.npy", "mean_x.npy"):
        assert (tmp_path/"with"/f).read_bytes() == (tmp_path/"without"/f).read_bytes()
    assert json.loads((tmp_path/"with"/"probe.json").read_text())["fit"] == json.loads((tmp_path/"without"/"probe.json").read_text())["fit"]
    assert not np.array_equal(np.load(tmp_path/"with"/"velocity"/"weight.npy"), np.load(tmp_path/"with"/"weight.npy"))


def test_rollout_and_counterfactual_tools_report_velocity(pair_caches, pair_readout, active, tmp_path, monkeypatch):
    base, long = pair_caches
    probe, truth = pair_readout
    tool = _tool("state_readout")
    monkeypatch.setattr(v7jobs, "load_for_evaluation", _fake_load())
    out = tmp_path/"roll.json"
    tool.main(["rollout", "--job", str(_job(tmp_path/"r")), "--probe", str(probe), "--truth", str(truth), "--features", str(long),
               "--horizon", "64", "--device", "cpu", "--output", str(out)])
    r = json.loads(out.read_text())
    assert set(r["means"]) == set(state_readout.rollout_names(64, True)) and r["velocity_units"] == "px_per_s"
    assert r["velocity_probe"]["path"].endswith("velocity")
    vp, _ = state_readout.load(probe/"velocity", dataset_tag=PAIR.tag, system="pendulum", quantity="velocity")
    z = torch.from_numpy(np.load(long/"0038"/"raw_z.npy").astype(np.float32))
    tr = json.loads(truth.read_text()); yv = next(x for x in tr["long"] if x["trajectory_id"] == 38)["v"]
    row = next(x for x in r["rows"] if x["trajectory_id"] == 38)["values"]
    assert row["vel_readout:T=40"] == pytest.approx(float(np.linalg.norm(vp(z[:1])[0]-np.array(yv["40"]))), rel=1e-6)
    assert row["vel_floor:T=40"] == pytest.approx(float(np.linalg.norm(vp(z[40:41])[0]-np.array(yv["40"]))), rel=1e-6)
    for name, model in (("identity", None), ("shift", Shift())):
        monkeypatch.setattr(v7jobs, "load_for_evaluation", _fake_load(model))
        o = tmp_path/f"cf-{name}.json"
        tool.main(["counterfactual", "--job", str(_job(tmp_path/name)), "--probe", str(probe), "--truth", str(truth),
                   "--features", str(base), "--device", "cpu", "--output", str(o)])
        c = json.loads(o.read_text())
        assert "velocity" in c["overall"] and all("velocity" in c["by_duration"][k] for k in ("1", "2", "4", "8"))
        assert all("vel_cf_error" in x["values"] for x in c["rows"])
        if name == "identity":
            assert c["overall"]["velocity"]["explained"] == 0. and c["overall"]["velocity"]["gain"] == 0.
            # identity：ẑ_A = ẑ_B，预测的 Δ 为 0，vel_cf_error = ‖v(s_B) − v(s_A)‖，两个终点按手写公式独立算
            row = next(x for x in c["rows"] if (x["trajectory_id"], x["start_tick"], x["duration_cycles"]) == (37, 10, 4.))
            active(PAIR)
            sys.path.insert(0, str(ROOT/"tools"))
            import probe_actions
            from po_jepa import simulation
            actual, cf_state, _ = probe_actions.counterfactual_end(systems.get(), simulation, 37, row["donor_id"], 10, 4)
            want = float(np.linalg.norm(80.*(_pend_v(cf_state)-_pend_v(actual))))
            assert want > 0 and row["values"]["vel_cf_error"] == pytest.approx(want, rel=1e-9)
            assert row["values"]["vel_cf_true"] == pytest.approx(want, rel=1e-9) and row["values"]["vel_cf_pred"] == 0.
        else:
            assert c["overall"]["velocity"]["cf_pred"] > 0
    active(PAIR)


def test_tools_refuse_second_order_inputs_without_velocity(pair_caches, pair_readout, active, tmp_path, monkeypatch):
    base, long = pair_caches
    probe, truth = pair_readout
    tool = _tool("state_readout")
    monkeypatch.setattr(v7jobs, "load_for_evaluation", _fake_load())
    run = lambda p, t, name: tool.main(["counterfactual", "--job", str(_job(tmp_path/name)), "--probe", str(p), "--truth", str(t),
                                         "--features", str(base), "--device", "cpu", "--output", str(tmp_path/f"{name}.json")])
    old = json.loads(truth.read_text()); old.pop("velocity"); old.pop("velocity_units")
    (tmp_path/"old_truth.json").write_text(json.dumps(old))
    with pytest.raises(SystemExit, match="速度真值"):
        run(probe, tmp_path/"old_truth.json", "a")
    import shutil
    shutil.copytree(probe, tmp_path/"noprobe"); shutil.rmtree(tmp_path/"noprobe"/"velocity")
    with pytest.raises(SystemExit, match="速度探针"):
        run(tmp_path/"noprobe", truth, "b")
    shutil.copytree(probe, tmp_path/"mixed")
    m = json.loads((tmp_path/"mixed"/"velocity"/"probe.json").read_text()); m["trajectories"] = [32, 33, 34]
    (tmp_path/"mixed"/"velocity"/"probe.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="同一份缓存"):
        run(tmp_path/"mixed", truth, "c")
    # S25b5-fix1（Codex 审核 F1）：两边都缺来源字段、特征均值 SHA 不同、配对的位置探针不是这一个，都拒绝
    def tampered(name, edit_pos, edit_vel):
        shutil.copytree(probe, tmp_path/name)
        for sub, edit in (("", edit_pos), ("velocity", edit_vel)):
            f = tmp_path/name/sub/"probe.json"; m = json.loads(f.read_text()); edit(m); f.write_text(json.dumps(m))
        return tmp_path/name
    drop = lambda m: [m.pop(k) for k in ("cache_identity", "trajectories", "tubelet")]
    keep = lambda m: None
    with pytest.raises(SystemExit, match="缺来源字段"):
        run(tampered("bare", drop, drop), truth, "d")
    with pytest.raises(SystemExit, match="缺来源字段"):
        run(tampered("bare_v", keep, lambda m: m.pop("cache_identity")), truth, "e")
    mean_dir = tampered("mean", keep, keep)                      # 速度探针换一份特征均值（数组和它的 SHA 一起换，单探针加载照样通过）
    mx = np.load(mean_dir/"velocity"/"mean_x.npy"); np.save(mean_dir/"velocity"/"mean_x.npy", mx+np.float32(1.))
    vf = mean_dir/"velocity"/"probe.json"; m = json.loads(vf.read_text())
    m["mean_x_sha256"] = state_readout._sha(mean_dir/"velocity"/"mean_x.npy"); vf.write_text(json.dumps(m))
    state_readout.load(mean_dir/"velocity", dataset_tag=PAIR.tag, system="pendulum", quantity="velocity")
    with pytest.raises(SystemExit, match="特征均值"):
        run(mean_dir, truth, "f")
    with pytest.raises(SystemExit, match="配对位置探针"):
        run(tampered("pair", keep, lambda m: m["position_probe"].update(weight_sha256="0"*64)), truth, "g")
    with pytest.raises(SystemExit, match="配对位置探针"):
        run(tampered("nopair", keep, lambda m: m.pop("position_probe")), truth, "h")
    assert not any((tmp_path/f"{n}.json").exists() for n in "abcdefgh")
    active(PAIR)


def test_plan_velocity_error_is_the_executed_end_against_the_goal(tmp_path, monkeypatch, active):
    """模型规划的速度误差 = 实际执行的终点和目标终点的速度之差（不是和起点、不是只看终点）。--no-baselines、--replan = 时域：
    续算的次序是 [不动, 目标 × n, 执行]，最后一次是执行、倒数第二次是选中的目标。"""
    from test_s25b2_online_pair import _plan_in_process
    rep, _, paths, _ = _plan_in_process(tmp_path, monkeypatch, active, "tubelet=pair", "--replan", "3")
    ep, = rep["episodes"]
    assert ep["goal_valid"] and set(ep["velocity"]) == {"model", "zero"}
    end, goal, zero = paths[-1]["path"][-1], paths[-2]["path"][-1], paths[0]["path"][-1]
    assert not paths[0]["actions"].any() and len(paths[-1]["actions"]) == 3
    assert ep["velocity"]["model"] == pytest.approx(float(np.linalg.norm(_pend_v(end)-_pend_v(goal))), rel=1e-12)
    assert ep["velocity"]["zero"] == pytest.approx(float(np.linalg.norm(_pend_v(zero)-_pend_v(goal))), rel=1e-12)
    sin_cos = lambda h: np.array([math.sin(h[0]), -math.cos(h[0])])
    assert ep["model"] == pytest.approx(float(np.linalg.norm(sin_cos(end)-sin_cos(goal))), rel=1e-12)
    ok = max(ep["model"], .25*ep["velocity"]["model"]) <= .5*ep["zero"]
    assert rep["model"]["success_rate"] == float(ok) and rep["model"]["position_only_success_rate"] == float(ep["model"] <= .5*ep["zero"])


def test_plan_baseline_velocity_errors_are_the_executed_ends(tmp_path, monkeypatch, active):
    """S25b5-fix1（Codex 审核 F2）：oracle、random 的速度误差也从实际执行的终点独立算（手写摆锤公式）。--replan = 时域 3、
    CEM 1 轮、oracle 4 个样本：续算次序是 [不动, 目标 × n, 模型执行, oracle 候选 × 4, oracle 执行, 随机]。"""
    from test_s25b2_online_pair import _job as job_dir, _load, _spec
    spec = _spec("pendulum", "tubelet=pair"); active(spec)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    plan = _load("plan_v7")
    paths, original = [], plan.rollout_path
    def spy(*args):
        path = original(*args); paths.append(dict(actions=np.array(args[6]), start=args[5], end=path[-1].copy())); return path
    monkeypatch.setattr(plan, "rollout_path", spy)
    out = tmp_path/"plan.json"
    monkeypatch.setattr(sys, "argv", ["plan_v7.py", "--job", str(job_dir(tmp_path, spec.tag)), "--device", "cpu", "--output", str(out),
                                      "--fake-encoder", "--episodes", "1", "--horizon", "3", "--replan", "3", "--samples", "8",
                                      "--iterations", "1", "--oracle-samples", "4"])
    plan.main()
    rep = json.loads(out.read_text(encoding="utf-8"))
    ep, = rep["episodes"]
    n_goal = len(paths)-8
    assert n_goal >= 1 and not paths[0]["actions"].any()
    zero, goal, model, oracle, random = (paths[i]["end"] for i in (0, n_goal, n_goal+1, -2, -1))
    assert all(p["start"] == ep["start_tick"] and len(p["actions"]) == 3 for p in paths)
    sin_cos = lambda h: np.array([math.sin(h[0]), -math.cos(h[0])])
    for name, end in (("zero", zero), ("model", model), ("oracle", oracle), ("random", random)):
        assert ep["velocity"][name] == pytest.approx(float(np.linalg.norm(_pend_v(end)-_pend_v(goal))), rel=1e-12), name
        assert ep[name] == pytest.approx(float(np.linalg.norm(sin_cos(end)-sin_cos(goal))), rel=1e-12), name
    for name in ("model", "oracle", "random"):
        ok = max(ep[name], .25*ep["velocity"][name]) <= .5*ep["zero"]
        assert rep[name]["success_rate"] == float(ok) and rep[name]["mean_velocity_error_px_per_s"] == pytest.approx(80.*ep["velocity"][name])
