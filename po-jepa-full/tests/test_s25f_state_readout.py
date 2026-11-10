"""S25f：物理状态读出、长时程 VPT、反事实预测、模态频率对照（docs/segments/S25f.md）。

假特征用 tools/encode_v7.py --fake-encoder（子进程，同 S25c 的测试），读取时把 synthetic 标记换成 real（S25c 的 real 夹具）。
数据集 double_well:40（验证段 32–35、测试段 36–39）；长时程 horizon=64。
"""

import copy
import hashlib
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
from torch.nn import functional as F

from po_jepa import configuration, datasets, protocols, systems, tubelet_probe, variants
from po_jepa.baselines import IdentityPredictor
from po_jepa.runtime import cache, comparison, evaluation, long_cache, long_rollout, modal_spectrum, state_readout, v7jobs
from po_jepa.runtime.addon import d4
from test_s25c_long_horizon import ActionToy, _clean_env, _encode, active, real, toy_windows  # noqa: F401
from test_training import e06_config

ROOT = Path(__file__).resolve().parents[1]
BASE = datasets.parse("double_well:40")
LONG = datasets.parse("double_well~horizon=64:40")
VAL_IDS = tuple(range(32, 36))
TEST_IDS = tuple(range(36, 40))


def _tool(name):
    spec = importlib.util.spec_from_file_location(f"{name}_tool", ROOT/f"tools/{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    root = tmp_path_factory.mktemp("s25f")
    for name, variant, splits in (("base", "", "validation,test"), ("long", "horizon=64", "test")):
        r = _encode(root/name, variant, splits=splits)
        assert r.returncode == 0, r.stderr
    return root/"base", root/"long"


# —— 读出目标 ——

def test_target_constants():
    """目标的倍数写死（不调被测函数）：粒子类 × 48、van_der_pol × 27、push 前 4 维 × 48、场原样。"""
    np.testing.assert_array_equal(state_readout.target("double_well", [1., -2.], {}), [48., -96.])
    np.testing.assert_array_equal(state_readout.target("van_der_pol", [1., .5], {}), [27., 13.5])
    np.testing.assert_array_equal(state_readout.target("landscape", [.5, 1.], {}), [24., 48.])
    np.testing.assert_array_equal(state_readout.target("push", np.arange(7.), {}), [0., 48., 96., 144.])
    h = np.arange(600.)
    np.testing.assert_array_equal(state_readout.target("gray_scott", h, {}), h[:512])
    np.testing.assert_array_equal(state_readout.target("advection_diffusion", h[:256], {}), h[:256])
    np.testing.assert_allclose(state_readout.target("pendulum", [math.pi/2, 3.], {}), [80., 0.], atol=1e-12)
    sc = np.concatenate((np.arange(8.), np.full(8, 9.)))              # spring_chain n=4：位置 8 维 + 速度 8 维
    np.testing.assert_array_equal(state_readout.target("spring_chain", sc, {}), np.arange(8.)*48.)
    assert [state_readout.units(s) for s in ("double_well", "wave", "gray_scott", "advection_diffusion", "push")] == \
        ["px", "field_l2", "field_l2", "field_l2", "px"]
    with pytest.raises(ValueError):
        state_readout.target("double_well", np.zeros((2, 2)), {})


@pytest.mark.parametrize("scene", [s for s in protocols.ALL_SCENES if not s.startswith("gray_scott")])
def test_target_distance_equals_probe_actions_motion(scene, active):
    """‖target(a) − target(b)‖ = probe_actions.motion(a, b)：读出误差和动作探针的 *_px 列同一个单位。"""
    probe_actions = _tool("probe_actions")
    active(datasets.parse(f"{scene}:40"))
    system = datasets.active().system
    sys_ = systems.get()
    states, _ = sys_.frames(0)
    params = dict(datasets.active().knobs.params)
    scale = 48. if system == "double_well" else getattr(sys_, "SCALE", 1.)
    for a, b in ((states[0], states[9]), (states[3], states[32])):
        d = float(np.linalg.norm(state_readout.target(system, a, params)-state_readout.target(system, b, params)))
        assert d == pytest.approx(probe_actions.motion(system, a, b, scale, params), rel=1e-12, abs=1e-12)


def test_target_gray_scott_and_push_match_motion():
    probe_actions = _tool("probe_actions")
    g = np.random.default_rng(0)
    for system, dim in (("gray_scott", 514), ("push", 7)):
        a, b = g.normal(size=dim), g.normal(size=dim)
        d = float(np.linalg.norm(state_readout.target(system, a, {})-state_readout.target(system, b, {})))
        assert d == pytest.approx(probe_actions.motion(system, a, b, 1., {}), rel=1e-12)


# —— 探针 ——

def test_features_are_layer_normed_readouts():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(3, 256, 1408, generator=g)*3+1
    ln = (z-z.mean(-1, keepdim=True))/torch.sqrt(z.var(-1, unbiased=False, keepdim=True)+1e-5)
    for kind in ("mean", "pool4", "full"):
        assert torch.allclose(state_readout.features(z, kind), tubelet_probe.readout(ln, kind), atol=1e-5)
    scale = torch.rand(3, 256, 1, generator=g)+.5; shift = torch.randn(3, 256, 1, generator=g)
    assert torch.allclose(state_readout.features(z*scale+shift, "full"), state_readout.features(z, "full"), atol=1e-4)
    with pytest.raises(ValueError):
        state_readout.features(z, "max")


def _synthetic(n_traj=8, rows=6, seed=0):
    """低秩特征：z = 隐变量 [n,4] × 基 + 小噪声，目标是隐变量的线性函数（少样本时岭回归也能泛化）。"""
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(n_traj*rows, 4, generator=g)
    basis = torch.randn(4, 256, 1408, generator=g)
    z = torch.einsum("nl,ltd->ntd", latent, basis)+.01*torch.randn(n_traj*rows, 256, 1408, generator=g)+3.
    y = (latent.double()@torch.randn(4, 3, generator=g, dtype=torch.float64)).numpy()
    groups = np.repeat(np.arange(100, 100+n_traj), rows)
    return z, y, groups


def test_fit_primal_weights_equal_the_dual_prediction():
    """存的原始形式 W = X̃ᵀα 在新样本上给出和对偶形式（核 + α）相同的预测。"""
    z, y, groups = _synthetic()
    x = state_readout.features(z, "mean")
    probe, record = state_readout.fit(x, y, groups, "mean")
    new = torch.randn(5, 256, 1408, generator=torch.Generator().manual_seed(9))
    xn = state_readout.features(new, "mean").double().numpy()
    xa = x.double().numpy()
    mu = xa.mean(0)
    k = (xa-mu)@(xa-mu).T
    lam = record["lambda_value"]
    alpha = np.linalg.solve(k+lam*np.eye(len(k)), y-y.mean(0))
    expected = (xn-mu)@(xa-mu).T@alpha+y.mean(0)
    np.testing.assert_allclose(probe(new), expected, rtol=1e-4, atol=1e-6)
    assert record["samples"] == 48 and record["trajectories"] == 8 and record["outputs"] == 3 and record["folds"] == 5
    assert record["variance"] == pytest.approx(float(((y-y.mean(0))**2).sum(1).mean()))
    assert record["cv_r2"] == max(record["cv_r2_by_lambda"].values()) and record["cv_r2"] > .5
    assert record["lambda_value"] == pytest.approx(float(np.trace(k))/len(k)*10.**record["lambda_exponent"], rel=1e-9)


def test_select_lambda_splits_by_trajectory(monkeypatch):
    """交叉验证按轨迹分折：同一条轨迹的样本不会同时出现在训练和测试；分数是各折 SSE 相加后的合并 R²。"""
    z, y, groups = _synthetic()
    x = state_readout.features(z, "mean").double().numpy()
    k = (x-x.mean(0))@(x-x.mean(0)).T
    seen = []
    fit = tubelet_probe._ridge_fit
    monkeypatch.setattr(tubelet_probe, "_ridge_fit", lambda k_, y_, rows: (seen.append(rows), fit(k_, y_, rows))[1])
    choice = state_readout.select_lambda(k, y, groups)
    assert len(seen) == 5
    for rows in seen:
        test = set(groups[np.setdiff1d(np.arange(len(y)), rows)].tolist())
        assert test and not test & set(groups[rows].tolist()) and len(test) in (1, 2)
    # 独立重算第一个 λ 的合并 R²
    fold = groups % 5; sse = 0.
    for f in range(5):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        kc = k[np.ix_(tr, tr)]; m = kc.mean(1)
        kt = k[np.ix_(tr, tr)]-k[np.ix_(tr, tr)].mean(0)[None]-k[np.ix_(tr, tr)].mean(1)[:, None]+k[np.ix_(tr, tr)].mean()
        ks = k[np.ix_(te, tr)]-k[np.ix_(te, tr)].mean(1)[:, None]-k[np.ix_(tr, tr)].mean(1)[None]+k[np.ix_(tr, tr)].mean()
        lam = np.trace(kt)/len(tr)*10.**tubelet_probe.LAMBDA_EXPONENTS[0]
        a = np.linalg.solve(kt+lam*np.eye(len(tr)), y[tr]-y[tr].mean(0))
        sse += ((y[te]-ks@a-y[tr].mean(0))**2).sum()
    assert choice["cv_r2"][0] == pytest.approx(1-sse/((y-y.mean(0))**2).sum(), rel=1e-6)
    with pytest.raises(ValueError, match="2 条轨迹"):
        state_readout.select_lambda(k[:6, :6], y[:6], groups[:6])


def test_probe_save_load_checks(tmp_path):
    z, y, groups = _synthetic()
    probe, _ = state_readout.fit(state_readout.features(z, "mean"), y, groups, "mean")
    meta = state_readout.save(probe, tmp_path/"p", dict(dataset="double_well-40", system="double_well"))
    loaded, meta2 = state_readout.load(tmp_path/"p", dataset_tag="double_well-40", system="double_well")
    assert meta == meta2 and loaded.kind == "mean" and loaded.variance == probe.variance
    np.testing.assert_array_equal(loaded(z[:4]), probe(z[:4]))
    with pytest.raises(FileExistsError):
        state_readout.save(probe, tmp_path/"p", dict(dataset="double_well-40", system="double_well"))
    with pytest.raises(ValueError, match="fitted on"):
        state_readout.load(tmp_path/"p", dataset_tag="double_well-200", system="double_well")
    w = np.load(tmp_path/"p"/"weight.npy"); w[0, 0] += 1.; np.save(tmp_path/"p"/"weight.npy", w)
    with pytest.raises(ValueError, match="SHA"):
        state_readout.load(tmp_path/"p", dataset_tag="double_well-40", system="double_well")


def test_vpt_definition():
    t = (8., 16., 24., 32.)
    assert state_readout.vpt(t, [.01, .02, .2, .01], .1) == 16.
    assert state_readout.vpt(t, [.2, .01, .01, .01], .1) == 0.
    assert state_readout.vpt(t, [.1, .1, .1, .1], .1) == 32.
    assert state_readout.vpt(t, [float("nan"), 0., 0., 0.], .1) == 0.
    assert state_readout.VPT_THRESHOLDS == (.05, .1, .25)


# —— 长时程读出 ——

def _mean_probe(o=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(1408, o, generator=g)
    return state_readout.Probe("mean", w, torch.zeros(1408), np.arange(o, dtype=np.float64), 24., {})


def test_rollout_predictions_are_the_s25c_rollout(active):
    """读出用的 8 步滚动与 long_rollout.trajectory_values 的 rollout:T 是同一个预测（JEPA 距离逐位相同），调用次序相同。"""
    active(LONG)
    windows = toy_windows(64)
    a, b = ActionToy(), ActionToy()
    preds = state_readout.rollout_predictions(a, windows[64.], 64)
    values = long_rollout.trajectory_values(b, windows, e06_config(), 64)
    assert list(preds) == list(long_rollout.curve_times(64)) and a.calls == [8.]*8
    for t, p in preds.items():
        assert d4._jepa(p, windows[t].target.raw_z, e06_config()) == values[f"rollout:T={t:g}"]


def test_rollout_values_formulas(active):
    """identity：每个 t 的读出 = R(z_0)；floor = R(z*_t)；nmse = 误差²/V；vpt 按定义。"""
    active(LONG)
    windows = toy_windows(64)
    probe = _mean_probe()
    truth = {t: probe(windows[t].target.raw_z)[0]+np.array([3., 4.])*(t/64.) for t in long_rollout.curve_times(64)}
    v = state_readout.rollout_values(IdentityPredictor(), probe, windows, truth, 64)
    assert tuple(v) == state_readout.rollout_names(64)
    r0 = probe(windows[64.].inputs.raw_z)[0]
    for t in long_rollout.curve_times(64):
        assert v[f"readout:T={t:g}"] == pytest.approx(float(np.linalg.norm(r0-truth[t])), rel=1e-6)
        assert v[f"floor:T={t:g}"] == pytest.approx(5.*t/64., abs=1e-4)
        assert v[f"nmse:T={t:g}"] == pytest.approx(v[f"readout:T={t:g}"]**2/24., rel=1e-6)
        assert v[f"floor_nmse:T={t:g}"] == pytest.approx((5.*t/64.)**2/24., abs=1e-5)
    # floor 的 nmse = (5t/64)²/2.5：≤ 0.1 ⇔ t ≤ 32·… 写死：t=8 → 0.0244、16 → 0.0977、24 → 0.2197
    assert v["floor_vpt@0.05"] == 8. and v["floor_vpt@0.1"] == 16. and v["floor_vpt@0.25"] == 24.
    # 模型的 vpt 按 nmse 独立数：最长的前缀
    times = long_rollout.curve_times(64)
    for eps in (.05, .1, .25):
        k = 0
        while k < len(times) and v[f"nmse:T={times[k]:g}"] <= eps:
            k += 1
        assert v[f"vpt@{eps:g}"] == (times[k-1] if k else 0.)
    assert any(v[f"vpt@{e:g}"] != v[f"floor_vpt@{e:g}"] for e in (.05, .1, .25))


# —— 反事实 ——

class Shift(torch.nn.Module):
    """Φ(z, u) = z + c·(∫u_x)·e：只读动作，第 0 个 token 的前两维动。"""
    def forward(self, raw, controls, duration, grid):
        n = int(controls.n_segments[0])
        integral = float(((controls.ends[0, :n]-controls.starts[0, :n]).double()*controls.u_values[0, :n, 0].double().cpu()).sum())
        out = raw.clone(); out[0, :, 0] += integral
        return out


def test_counterfactual_values_formulas(active):
    active(LONG)
    windows = toy_windows(64)
    w = windows[16.]
    donor = windows[8.]               # 动作只到 8 个 tick：积分和 16 的窗口不同（Shift 不看时长是否一致）
    probe = _mean_probe()
    ya, yb = np.array([1., 2.]), np.array([4., 6.])
    v = state_readout.counterfactual_values(Shift(), probe, w, donor, ya, yb)
    za = Shift()(w.inputs.raw_z, w.inputs.controls, w.inputs.duration, w.inputs.grid)
    zb = Shift()(w.inputs.raw_z, donor.inputs.controls, w.inputs.duration, w.inputs.grid)
    pred = probe(zb)[0]-probe(za)[0]
    assert np.linalg.norm(pred) > .1 and abs(float(pred@np.array([3., 4.]))) > .1
    assert v["cf_true"] == 5. and v["cf_pred"] == pytest.approx(float(np.linalg.norm(pred)), rel=1e-4)
    assert v["cf_error"] == pytest.approx(float(np.linalg.norm(pred-np.array([3., 4.]))), rel=1e-4)
    assert v["cf_inner"] == pytest.approx(float(pred@np.array([3., 4.])), rel=1e-4)
    assert v["actual_readout"] == pytest.approx(float(np.linalg.norm(probe(za)[0]-ya)), rel=1e-4)
    assert v["actual_floor"] == pytest.approx(float(np.linalg.norm(probe(w.target.raw_z)[0]-ya)), rel=1e-4)
    vi = state_readout.counterfactual_values(IdentityPredictor(), probe, w, toy_windows(64, tid=37)[16.], ya, yb)
    assert vi["cf_pred"] == 0. and vi["cf_inner"] == 0. and vi["cf_error"] == 5.


def test_counterfactual_summary():
    rows = [dict(identical_controls=False, values=dict(cf_true=5., cf_pred=5., cf_error=0., cf_inner=25., actual_readout=1., actual_floor=0.)),
            dict(identical_controls=False, values=dict(cf_true=1., cf_pred=0., cf_error=1., cf_inner=0., actual_readout=3., actual_floor=2.)),
            dict(identical_controls=True, values=dict(cf_true=0., cf_pred=2., cf_error=2., cf_inner=0., actual_readout=2., actual_floor=1.))]
    s = state_readout.counterfactual_summary(rows)
    assert s["windows"] == 3 and s["windows_with_different_actions"] == 2
    assert s["explained"] == pytest.approx(1-1/26) and s["gain"] == pytest.approx(25/26)
    assert s["cos_mean"] == 1. and s["cos_windows"] == 1
    assert s["cf_true"] == 2. and s["actual_readout"] == 2. and s["actual_floor"] == 1. and "cf_inner" not in s
    assert state_readout.counterfactual_summary([]) == {}
    assert state_readout.counterfactual_summary(rows[2:])["explained"] is None


# —— 工具 ——

def _job(root, condition="identity", dataset=BASE.tag):
    job = root/"job"/"run"/"job"; job.mkdir(parents=True)
    (job/"job.json").write_text(json.dumps(dict(seed=1703, updates=10000, dataset=dataset, condition=condition)))
    return root/"job"


def test_fit_and_truth_tools(caches, real, active, tmp_path, monkeypatch):
    base, long = caches
    tool = _tool("state_readout")
    active(BASE)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    args = ["--system", "double_well", "--trajectories", "40", "--features", str(base)]
    tool.main(["fit", *args, "--output", str(tmp_path/"probe"), "--device", "cpu"])
    meta = json.loads((tmp_path/"probe"/"probe.json").read_text())
    assert (meta["dataset"], meta["split"], meta["trajectories"], meta["units"]) == (BASE.tag, "validation", list(VAL_IDS), "px")
    assert meta["fit"]["samples"] == 4*33 and meta["fit"]["folds"] == 4 and meta["fit"]["kind"] == "full"
    assert np.load(tmp_path/"probe"/"weight.npy").shape == (256*1408, 2)
    # 探针在真实特征上的读出：用 frames() 的状态独立核一条验证轨迹的拟合残差不大于总方差
    probe, _ = state_readout.load(tmp_path/"probe", dataset_tag=BASE.tag, system="double_well")
    states, _ = systems.get().frames(33)
    z = torch.from_numpy(np.load(base/"0033"/"raw_z.npy").astype(np.float32))
    err = ((probe(z)-states*48.)**2).sum(1).mean()
    assert err < probe.variance
    with pytest.raises(SystemExit, match="已存在"):
        tool.main(["fit", *args, "--output", str(tmp_path/"probe"), "--device", "cpu"])
    tool.main(["truth", *args, "--long-features", str(long), "--horizon", "64", "--output", str(tmp_path/"truth.json")])
    t = json.loads((tmp_path/"truth.json").read_text())
    assert (t["dataset"], t["long_dataset"], t["horizon"], t["units"]) == (BASE.tag, LONG.tag, 64, "px")
    assert [(r["trajectory_id"], r["start_tick"], r["duration_cycles"]) for r in t["counterfactual"]] == \
        [(i, s, d) for i in TEST_IDS for s in (0, 5, 10, 15, 20) for d in (1., 2., 4., 8.)]
    r = next(r for r in t["counterfactual"] if (r["trajectory_id"], r["start_tick"], r["duration_cycles"]) == (37, 10, 4.))
    states, _ = systems.get().frames(37)
    np.testing.assert_array_equal(r["y_actual"], states[14]*48.)
    assert r["donor_id"] == 38 and r["y_counterfactual"] != r["y_actual"]
    active(LONG)
    long_states, _ = systems.get().frames(39)
    active(BASE)
    row = next(r for r in t["long"] if r["trajectory_id"] == 39)
    assert list(row["y"]) == [f"{x}" for x in range(8, 65, 8)]
    np.testing.assert_array_equal(row["y"]["56"], long_states[56]*48.)
    assert datasets.active() == BASE
    # 拿错数据集的缓存：缓存登记的调节项或画面哈希对不上
    with pytest.raises((SystemExit, ValueError)):
        tool.main(["truth", "--system", "double_well", "--variant", "sigma=4", "--trajectories", "40", "--features", str(base),
                   "--long-features", str(long), "--horizon", "64", "--output", str(tmp_path/"bad.json")])
    active(BASE)
    with pytest.raises(SystemExit, match="基础场景"):
        tool.main(["fit", "--system", "double_well", "--variant", "horizon=64", "--trajectories", "40", "--features", str(base),
                   "--output", str(tmp_path/"p2"), "--device", "cpu"])
    active(BASE)


def test_checked_states_compares_image_hashes(active):
    tool = _tool("state_readout")
    active(BASE)
    _, frames = systems.get().frames(36)
    m = dict(frames=[dict(tick=t, image_sha256=hashlib.sha256(f.tobytes()).hexdigest()) for t, f in enumerate(frames)])
    states = tool._checked_states(systems.get(), 36, m, [0, 5, 32])
    assert states.shape == (33, 2)
    bad = copy.deepcopy(m); bad["frames"][5]["image_sha256"] = "0"*64
    tool._checked_states(systems.get(), 36, bad, [0, 32])            # 没要求核的帧不管
    with pytest.raises(SystemExit, match="第 5 帧"):
        tool._checked_states(systems.get(), 36, bad, [0, 5])
    with pytest.raises(SystemExit, match="帧"):
        tool._checked_states(systems.get(), 36, dict(frames=m["frames"][:32]), [0])


@pytest.fixture
def readout_inputs(caches, real, active, tmp_path, monkeypatch):
    """每个用到的测试各自拟合一次（mean 读出，快），返回 (probe 目录, truth 路径)。"""
    base, long = caches
    tool = _tool("state_readout")
    active(BASE)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    args = ["--system", "double_well", "--trajectories", "40", "--features", str(base)]
    tool.main(["fit", *args, "--kind", "mean", "--output", str(tmp_path/"probe"), "--device", "cpu"])
    tool.main(["truth", *args, "--long-features", str(long), "--horizon", "64", "--output", str(tmp_path/"truth.json")])
    active(BASE)
    return tmp_path/"probe", tmp_path/"truth.json"


def _fake_load(seen, model=None):
    def load(condition, training_run, device):
        seen.update(condition=condition, training_run=training_run, dataset=datasets.active().tag)
        return (IdentityPredictor() if model is None else model), e06_config(), None, dict(fake=True)
    return load


def test_rollout_tool(caches, readout_inputs, active, tmp_path, monkeypatch):
    _, long = caches
    probe, truth = readout_inputs
    seen = {}
    monkeypatch.setattr(v7jobs, "load_for_evaluation", _fake_load(seen))
    tool = _tool("state_readout")
    out = tmp_path/"out"/"identity.json"
    args = ["rollout", "--job", str(_job(tmp_path)), "--probe", str(probe), "--truth", str(truth), "--features", str(long),
            "--horizon", "64", "--device", "cpu"]
    tool.main([*args, "--output", str(out)])
    assert seen == dict(condition="identity", training_run=None, dataset=BASE.tag) and datasets.active() == BASE
    r = json.loads(out.read_text())
    assert (r["kind"], r["dataset"], r["train_dataset"], r["horizon"], r["units"]) == ("state_readout_rollout", LONG.tag, BASE.tag, 64, "px")
    assert [x["trajectory_id"] for x in r["rows"]] == list(TEST_IDS) and set(r["means"]) == set(state_readout.rollout_names(64))
    # identity：每个 t 的读出 = R(z_0)，独立重算一条
    p, _ = state_readout.load(probe, dataset_tag=BASE.tag, system="double_well")
    z = torch.from_numpy(np.load(long/"0038"/"raw_z.npy").astype(np.float32))
    tr = json.loads(truth.read_text()); y = next(x for x in tr["long"] if x["trajectory_id"] == 38)["y"]
    row = next(x for x in r["rows"] if x["trajectory_id"] == 38)["values"]
    assert row["readout:T=40"] == pytest.approx(float(np.linalg.norm(p(z[:1])[0]-np.array(y["40"]))), rel=1e-6)
    assert row["floor:T=40"] == pytest.approx(float(np.linalg.norm(p(z[40:41])[0]-np.array(y["40"]))), rel=1e-6)
    assert r["means"]["vpt@0.1"] == pytest.approx(np.mean([x["values"]["vpt@0.1"] for x in r["rows"]]))
    with pytest.raises(SystemExit, match="已存在"):
        tool.main([*args, "--output", str(out)])
    with pytest.raises(SystemExit, match="horizon=64"):
        tool.main([*args[:-4], "--horizon", "128", "--device", "cpu", "--output", str(tmp_path/"x.json")])
    bad = json.loads(truth.read_text()); bad["long"][0]["manifest_sha256"] = "0"*64
    (tmp_path/"bad.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="manifest"):
        tool.main([*args[:6], str(tmp_path/"bad.json"), *args[7:], "--output", str(tmp_path/"y.json")])
    assert datasets.active() == BASE and not (tmp_path/"x.json").exists() and not (tmp_path/"y.json").exists()


def test_counterfactual_tool(caches, readout_inputs, active, tmp_path, monkeypatch):
    base, _ = caches
    probe, truth = readout_inputs
    tool = _tool("state_readout")
    for name, model in (("identity", None), ("shift", Shift())):
        seen = {}
        monkeypatch.setattr(v7jobs, "load_for_evaluation", _fake_load(seen, model))
        out = tmp_path/f"{name}.json"
        tool.main(["counterfactual", "--job", str(_job(tmp_path/name)), "--probe", str(probe), "--truth", str(truth),
                   "--features", str(base), "--device", "cpu", "--output", str(out)])
        r = json.loads(out.read_text())
        assert r["kind"] == "state_readout_counterfactual" and r["dataset"] == BASE.tag and len(r["rows"]) == 80
        assert set(r["by_duration"]) == {"1", "2", "4", "8"} and r["overall"]["windows"] == 80
        if name == "identity":
            assert r["overall"]["explained"] == 0. and r["overall"]["gain"] == 0. and r["overall"]["cos_windows"] == 0
        else:
            assert r["overall"]["cf_pred"] > 0 and r["overall"]["cos_windows"] > 0
    # 真值的窗口和登记的测试窗口对不上
    bad = json.loads(truth.read_text()); bad["counterfactual"] = bad["counterfactual"][1:]
    (tmp_path/"bad.json").write_text(json.dumps(bad))
    monkeypatch.setattr(v7jobs, "load_for_evaluation", _fake_load({}))
    with pytest.raises(SystemExit, match="登记"):
        tool.main(["counterfactual", "--job", str(_job(tmp_path/"bad")), "--probe", str(probe), "--truth", str(tmp_path/"bad.json"),
                   "--features", str(base), "--device", "cpu", "--output", str(tmp_path/"z.json")])
    # 探针是别的数据集的
    active(datasets.parse("double_well:200"))
    with pytest.raises(ValueError, match="fitted on"):
        tool.main(["counterfactual", "--job", str(_job(tmp_path/"other", dataset="double_well-200")), "--probe", str(probe),
                   "--truth", str(truth), "--features", str(base), "--device", "cpu", "--output", str(tmp_path/"w.json")])
    assert datasets.active().tag in (BASE.tag, "double_well-200")


def test_rollout_tool_refuses_sampled_training_dataset(tmp_path, active, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    tool = _tool("state_readout")
    for dataset in ("double_well~horizon=64-40", "double_well~fps=8-40", "double_well~zoom=0.8-40"):
        active(datasets.parse(dataset.rpartition("-")[0]+":40"))
        job = tmp_path/dataset; job.mkdir()
        (job/"job.json").write_text(json.dumps(dict(seed=1703, updates=1, dataset=dataset, condition="identity")))
        for command in ("rollout", "counterfactual"):
            with pytest.raises(SystemExit, match="基础场景"):
                tool.main([command, "--job", str(job), "--probe", str(tmp_path), "--truth", str(tmp_path), "--features", str(tmp_path),
                           "--output", str(tmp_path/"o.json"), "--device", "cpu"])


# —— 模态频率 ——

def test_true_modes_constants():
    sc = modal_spectrum.true_modes("spring_chain", datasets.parse("spring_chain:40").knobs)
    assert [m["multiplicity"] for m in sc] == [2]*4 and all(m["decay"] == .05 for m in sc)
    w1 = 2*4*math.sin(math.pi/10)                                        # k = 16、n = 4
    assert sc[0]["frequency"] == pytest.approx(math.sqrt(w1*w1-.0025), rel=1e-12)
    assert sc[3]["frequency"] == pytest.approx(math.sqrt((8*math.sin(4*math.pi/10))**2-.0025), rel=1e-12)
    wv = modal_spectrum.true_modes("wave", datasets.parse("wave:40").knobs)
    assert wv[0]["label"] == "m²=1" and wv[0]["multiplicity"] == 4 and wv[0]["decay"] == .05
    assert wv[0]["frequency"] == pytest.approx(math.sqrt((.15*2*math.pi)**2-.0025), rel=1e-12)
    assert max(m["frequency"] for m in wv) <= 4*math.pi and wv[1]["label"] == "m²=2" and wv[1]["multiplicity"] == 4
    assert len(wv) == 41     # c = 0.15：c·2π·√m² ≤ 4π ⇔ m² ≤ 177.8，|m_x|,|m_y| ≤ 8 时 m² 的不同取值有 41 个
    ad = modal_spectrum.true_modes("advection_diffusion", datasets.parse("advection_diffusion~diffusion=0.001:40").knobs)
    assert ad[0]["frequency"] == 0. and ad[0]["decay"] == pytest.approx(1e-3*4*math.pi**2, rel=1e-12)
    assert max(m["decay"] for m in ad) == pytest.approx(1e-3*4*math.pi**2*128, rel=1e-12)
    with pytest.raises(ValueError):
        modal_spectrum.true_modes("double_well", datasets.parse("double_well:40").knobs)


def test_learned_blocks_read_the_damped_rotation_blocks(active):
    active(datasets.parse("advection_diffusion:40"))
    torch.manual_seed(0)
    model, _ = comparison.build_condition("po_jepa_full", torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        for p in model.dynamics.omega_head.parameters():
            p.normal_(0., .3)
    z = torch.randn(1, 256, 1408, generator=torch.Generator().manual_seed(4))
    s = modal_spectrum.learned_blocks(model, z, cache.regular_grid(16, 16))
    with torch.no_grad():
        q = model.prepare_query(z, cache.regular_grid(16, 16))
    gamma, omega = q.dynamics.coefficients.gamma[0], q.dynamics.coefficients.omega[0]
    c = q.initial_coefficients[0]
    e = c[:, 0::2]**2+c[:, 1::2]**2
    np.testing.assert_allclose(s["decay"], (gamma*4).reshape(-1).double().numpy(), rtol=1e-6)
    np.testing.assert_allclose(s["frequency"], (omega.abs()*4).reshape(-1).double().numpy(), rtol=1e-6)
    np.testing.assert_allclose(s["weight"], (e/e.sum()).reshape(-1).double().numpy(), rtol=1e-5)
    assert s["weight"].sum() == pytest.approx(1.) and len(s["weight"]) == q.initial_coefficients.shape[1]*32


def test_compare_statistics():
    knobs = datasets.parse("spring_chain:40").knobs
    f = [m["frequency"] for m in modal_spectrum.true_modes("spring_chain", knobs)]
    # 一个样本：4 个块（数组顺序故意不按能量）：0.15 在 11、0.5 落在 f_1 上（衰减 0.1 = 真值 2 倍）、0.05 在 f_4 附近 7.7
    # （能量小，不在主导块里）、0.3 在 f_2 的 +5%
    s = dict(frequency=np.array([11., f[0], 7.7, f[1]*1.05]), decay=np.array([.3, .1, .2, .05]), weight=np.array([.15, .5, .05, .3]))
    c = modal_spectrum.compare("spring_chain", knobs, [s])
    assert c["near_true"] == pytest.approx(.85) and c["learned"]["energy_above_nyquist"] == 0.
    assert c["per_mode"][0]["nearest_rel_error"] == 0. and c["per_mode"][0]["decay_ratio"] == pytest.approx(2.)
    assert c["per_mode"][1]["nearest_rel_error"] == pytest.approx(.05) and c["per_mode"][1]["decay_ratio"] == pytest.approx(1.)
    assert c["per_mode"][2]["samples_with_near_block"] == 0 and c["per_mode"][2]["decay_ratio"] is None
    assert c["per_mode"][3]["samples_with_near_block"] == 1 and c["per_mode"][3]["decay_ratio"] == pytest.approx(4.)
    # 主导块（按能量累计到 90%）= 2.47、4.94、11；7.7 不在里面，所以 f_4 = 7.61 最近的是 4.94
    assert c["per_mode"][3]["nearest_rel_error"] == pytest.approx((f[3]-f[1]*1.05)/f[3])
    assert c["per_mode"][2]["nearest_rel_error"] == pytest.approx((f[2]-f[1]*1.05)/f[2])
    union = (.2*f[0]+.2*f[1]+1.1*f[3]-.9*f[2])/(4*math.pi)          # f_3、f_4 的 ±10% 区间重叠（7.12 > 6.85），前两个各自独立
    assert c["null_near_true"] == pytest.approx(union)
    assert c["learned"]["frequency_quantiles"] == [f[0], f[0], 11.]
    ad = modal_spectrum.compare("advection_diffusion", datasets.parse("advection_diffusion:40").knobs,
                                [dict(frequency=np.array([0., .5, 2.]), decay=np.array([.001, .5, 2.]), weight=np.array([.2, .3, .5]))])
    assert ad["non_oscillatory"] == pytest.approx(.5) and ad["decay_within_true"] == pytest.approx(.5)
    assert ad["null_non_oscillatory"] == pytest.approx(1/16)


def test_modal_tool(tmp_path_factory, real, active, tmp_path, monkeypatch):
    root = tmp_path_factory.mktemp("s25f_modal")
    args = ["--system", "advection_diffusion", "--trajectories", "40", "--output", str(root/"ad")]
    for mode, extra in (("encode", ["--fake-encoder", "--device", "cpu", "--splits", "test"]), ("finalize", ["--splits", "test"])):
        r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/encode_v7.py"), mode, *args, *extra],
                           env=_clean_env(), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    spec = datasets.parse("advection_diffusion:40")
    active(spec)
    monkeypatch.setattr(os, "environ", dict(os.environ))

    def load(condition, training_run, device):
        torch.manual_seed(0)
        model, loss = comparison.build_condition(condition, device)
        return model, loss, None, dict(fake=True)
    monkeypatch.setattr(v7jobs, "load_for_evaluation", load)
    tool = _tool("modal_spectrum")
    out = tmp_path/"m.json"
    tool.main(["--job", str(_job(tmp_path/"a", "po_jepa_full", spec.tag)), "--features", str(root/"ad"), "--output", str(out), "--device", "cpu"])
    r = json.loads(out.read_text())
    c = r["comparison"]
    assert r["kind"] == "modal_spectrum" and r["exploratory"] and c["samples"] == 20 and len(r["starts"]) == 20
    assert c["learned"]["decay_quantiles"][1] == pytest.approx(.4, rel=1e-2)     # 初始化：γ = 0.1/周期 = 0.4 s⁻¹
    assert r["init_reference"]["samples"] == 20
    with pytest.raises(SystemExit, match="PO-JEPA"):
        tool.main(["--job", str(_job(tmp_path/"b", "matched_transformer", spec.tag)), "--features", str(root/"ad"),
                   "--output", str(tmp_path/"n.json"), "--device", "cpu"])
    active(BASE)
    with pytest.raises(SystemExit, match="线性"):
        tool.main(["--job", str(_job(tmp_path/"c", "po_jepa_full", BASE.tag)), "--features", str(root/"ad"),
                   "--output", str(tmp_path/"n.json"), "--device", "cpu"])


# —— 任务表 ——

def test_make_v8_jobs_readout_stage():
    def run(*args):
        r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/make_v8_jobs.py"), *args], env=_clean_env(), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout.splitlines()
    lines = run("--stage", "readout")
    commands = [l for l in lines if l.startswith("python")]
    scenes = len(protocols.SCENES)
    fit = [c for c in commands if " fit " in c]; truth = [c for c in commands if " truth " in c]
    roll = [c for c in commands if "state_readout.py rollout" in c]; cf = [c for c in commands if "state_readout.py counterfactual" in c]
    modal = [c for c in commands if "modal_spectrum.py" in c]
    assert len(fit) == len(truth) == scenes and len(roll) == len(cf) == scenes*3*6
    assert len(modal) == 1*3*2          # 默认场景里线性的只有 advection_diffusion；PO-JEPA 条件两个
    assert "--features ../features-v8/double_well-2000 --output ../results-v8/full/readout/probe/double_well" in fit[0]
    assert "--long-features ../features-v8/double_well~horizon=128-2000 --horizon 128" in truth[0]
    assert all("horizon=128" in c.split("--features ")[1].split()[0] for c in roll)
    assert not any("horizon=128" in c for c in cf)
    assert all(("--ac-checkpoint" in c) == ("/vjepa2_ac " in c+" " or c.split("--job ")[1].split()[0].endswith("/vjepa2_ac")) for c in roll+cf)
    assert all("/po_jepa_" in c for c in modal) and all("advection_diffusion" in c for c in modal)
    second = run("--stage", "readout", "--scenes", "spring_chain,wave,pendulum", "--conditions", "po_jepa_full,identity")
    assert second[0].startswith("# 注意：二阶场景")
    assert len([l for l in second if l.startswith("python tools/modal_spectrum.py")]) == 2*3*1
    assert [l for l in lines if l.startswith("# =====")][0].startswith("# ===== R1")
    # 其他阶段没有读出命令
    assert not any("state_readout.py" in l or "modal_spectrum.py" in l for l in run("--stage", "long"))
