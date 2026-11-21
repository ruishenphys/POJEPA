"""C-2 diagnostics: tools/shrinkage_check.py (prediction shrinkage, evaluation only).

Fixtures and toy models come from the S25f state-readout tests (double_well:40, fake encoder, mean probe).
"""

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from po_jepa import datasets
from po_jepa.runtime import state_readout, v7jobs
from test_s25c_long_horizon import active, real  # noqa: F401
from test_s25f_state_readout import BASE, Shift, _fake_load, _job, _tool, caches, readout_inputs  # noqa: F401


def test_readout_summary_formulas():
    tool = _tool("shrinkage_check")
    rng = np.random.default_rng(0)
    y = rng.normal(size=(50, 3))*np.array([1., 2., 3.]); r_0 = y+rng.normal(size=y.shape)
    half = y.mean(axis=0)+.5*(y-y.mean(axis=0))
    s = tool.readout_summary(y, half, r_0, 2.)
    assert s["slope"] == pytest.approx(.5) and s["var_ratio"] == pytest.approx(.25) and s["corr"] == pytest.approx(1.)
    perfect = tool.readout_summary(y, y.copy(), r_0, 2.)
    assert perfect["nmse"] == 0. and perfect["motion_gain"] == pytest.approx(1.) and perfect["motion_ratio"] == pytest.approx(1.)
    still = tool.readout_summary(y, r_0.copy(), r_0, 2.)
    assert still["motion_gain"] == 0. and still["motion_ratio"] == 0.
    assert still["nmse"] == pytest.approx(float(((r_0-y)**2).sum(axis=1).mean())/2.)
    # pooled multi-output regression slope, written out
    yc, rc = y-y.mean(axis=0), r_0-r_0.mean(axis=0)
    assert still["slope"] == pytest.approx(float((rc*yc).sum()/(yc*yc).sum()))
    # constant truth: no spread to compare with
    flat = tool.readout_summary(np.ones((4, 2)), rng.normal(size=(4, 2)), np.zeros((4, 2)), 1.)
    assert flat["slope"] is None and flat["var_ratio"] is None and flat["corr"] is None


def test_accumulator_ln_statistics_match_direct_computation():
    tool = _tool("shrinkage_check")
    g = torch.Generator().manual_seed(3)
    acc = tool.Accumulator()
    xs, hs, ss = [], [], []
    for _ in range(6):
        x, h, s = (torch.randn(1, 4, 8, generator=g)*3+1 for _ in range(3))
        acc.add(x=x, z_hat=h, z_star=s, y=np.zeros(2), r_hat=np.zeros(2), r_star=np.zeros(2), r_0=np.zeros(2), fjd=.5, ln_eps=1e-5)
        xs.append(x); hs.append(h); ss.append(s)
    out = acc.summary(1.)
    ln = lambda t: F.layer_norm(t.double(), (8,), eps=1e-5).reshape(-1).numpy()
    lx, lh, ls = (np.stack([ln(t) for t in v]) for v in (xs, hs, ss))
    assert out["windows"] == 6 and out["fjd"] == .5
    assert out["ln_var_ratio"] == pytest.approx(lh.var(axis=0).sum()/ls.var(axis=0).sum(), rel=1e-9)
    dh, ds = lh-lx, ls-lx
    assert out["ln_motion_gain"] == pytest.approx((dh*ds).sum()/(ds*ds).sum(), rel=1e-9)
    assert out["ln_motion_ratio"] == pytest.approx(((dh*dh).sum()/(ds*ds).sum())**.5, rel=1e-9)
    cos = [(a@b)/np.linalg.norm(a)/np.linalg.norm(b) for a, b in zip(dh, ds)]
    assert out["ln_motion_cos"] == pytest.approx(np.mean(cos), rel=1e-9)
    norm = [float(h.double().norm()/s.double().norm()) for h, s in zip(hs, ss)]
    assert out["raw_norm_ratio"] == pytest.approx(np.mean(norm), rel=1e-12)
    std = [float((h.double().std(dim=-1)/s.double().std(dim=-1)).mean()) for h, s in zip(hs, ss)]
    assert out["raw_token_std_ratio"] == pytest.approx(np.mean(std), rel=1e-12)
    # a per-token shift and positive scale of the prediction changes neither LN statistic
    acc2 = tool.Accumulator()
    for x, h, s in zip(xs, hs, ss):
        acc2.add(x=x, z_hat=h*2.5+7., z_star=s, y=np.zeros(2), r_hat=np.zeros(2), r_star=np.zeros(2), r_0=np.zeros(2),
                 fjd=.5, ln_eps=0.)
    acc3 = tool.Accumulator()
    for x, h, s in zip(xs, hs, ss):
        acc3.add(x=x, z_hat=h, z_star=s, y=np.zeros(2), r_hat=np.zeros(2), r_star=np.zeros(2), r_0=np.zeros(2),
                 fjd=.5, ln_eps=0.)
    a2, a3 = acc2.summary(1.), acc3.summary(1.)
    for k in ("ln_var_ratio", "ln_motion_gain", "ln_motion_ratio", "ln_motion_cos"):
        assert a2[k] == pytest.approx(a3[k], rel=1e-6)          # float32 rounding of h*2.5+7
    assert a2["raw_norm_ratio"] != pytest.approx(a3["raw_norm_ratio"])


def test_shrinkage_tool(caches, readout_inputs, active, tmp_path, monkeypatch):
    base, _ = caches
    probe_dir, truth = readout_inputs
    tool = _tool("shrinkage_check")
    p, _ = state_readout.load(probe_dir, dataset_tag=BASE.tag, system="double_well")
    tr = json.loads(truth.read_text())
    reports = {}
    for name, model in (("identity", None), ("shift", Shift())):
        seen = {}
        monkeypatch.setattr(v7jobs, "load_for_evaluation", _fake_load(seen, model))
        out = tmp_path/f"{name}.json"
        tool.main(["--job", str(_job(tmp_path/name)), "--probe", str(probe_dir), "--truth", str(truth),
                   "--features", str(base), "--device", "cpu", "--output", str(out)])
        assert seen["dataset"] == BASE.tag and datasets.active() == BASE
        r = reports[name] = json.loads(out.read_text())
        assert r["kind"] == "shrinkage_check" and r["dataset"] == BASE.tag and r["grid_size"] == 16
        assert set(r["by_duration"]) == {"1", "2", "4", "8"}
        assert all(r["by_duration"][k]["windows"] == 20 for k in r["by_duration"])
        assert r["scientific_conclusions_verified"] is False
    for k, s in reports["identity"]["by_duration"].items():
        # identity predicts the input frame: no motion at all, and r_hat = r_0
        assert s["motion_gain"] == 0. and s["motion_ratio"] == 0.
        assert s["ln_motion_gain"] == 0. and s["ln_motion_ratio"] == 0. and s["ln_motion_cos_windows"] == 0
        # nmse equals the counterfactual tool's actual_readout for identity, recomputed from one window
        rows = [w for w in tr["counterfactual"] if w["duration_cycles"] == float(k)]
        z = [torch.from_numpy(np.load(base/f"{w['trajectory_id']:04d}"/"raw_z.npy")[w["start_tick"]:w["start_tick"]+1]
                              .astype(np.float32)) for w in rows]
        r0 = p(torch.cat(z)); y = np.array([w["y_actual"] for w in rows])
        assert s["nmse"] == pytest.approx(float(((r0-y)**2).sum(axis=1).mean())/p.variance, rel=1e-6)
        # *_star uses the real future frame: recompute motion_gain_star and the identity's spread independently
        zs = [torch.from_numpy(np.load(base/f"{w['trajectory_id']:04d}"/"raw_z.npy")
                               [w["start_tick"]+int(w["duration_cycles"]):w["start_tick"]+int(w["duration_cycles"])+1]
                               .astype(np.float32)) for w in rows]
        rs = p(torch.cat(zs))
        assert s["motion_gain_star"] == pytest.approx(float(((rs-r0)*(y-r0)).sum()/((y-r0)**2).sum()), rel=1e-6)
        yc, rc = y-y.mean(axis=0), r0-r0.mean(axis=0)
        assert s["var_ratio"] == pytest.approx(float((rc*rc).sum()/(yc*yc).sum()), rel=1e-6)
    for s in reports["shift"]["by_duration"].values():
        assert s["ln_motion_ratio"] > 0
    with pytest.raises(SystemExit, match="exist|已存在"):
        monkeypatch.setattr(v7jobs, "load_for_evaluation", _fake_load({}))
        tool.main(["--job", str(_job(tmp_path/"again")), "--probe", str(probe_dir), "--truth", str(truth),
                   "--features", str(base), "--device", "cpu", "--output", str(tmp_path/"identity.json")])
    bad = json.loads(truth.read_text()); bad["counterfactual"] = bad["counterfactual"][1:]
    (tmp_path/"bad.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="registered"):
        tool.main(["--job", str(_job(tmp_path/"bad")), "--probe", str(probe_dir), "--truth", str(tmp_path/"bad.json"),
                   "--features", str(base), "--device", "cpu", "--output", str(tmp_path/"z.json")])
    assert not (tmp_path/"z.json").exists()
