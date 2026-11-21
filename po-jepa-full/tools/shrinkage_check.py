"""C-2 diagnostics (evaluation only, no training): does a trained predictor shrink its predictions?

Question (C-2 analysis §6, 2026-11-21): on double_well the full model's future JEPA distance (FJD) is about equal to
po_jepa_no_consistency, yet the physical readout error is ~40% worse. One hypothesis is that the prediction is pulled
toward the mean / toward "nothing moves". This tool measures that directly on the registered test windows that
tools/state_readout.py counterfactual already uses (every test trajectory x registered start x duration 1, 2, 4, 8 cycles,
16x16 grid), with the same frozen probe R and the same truth file, for one trained job.

For every window: x = input frame (raw z at the start tick), z_hat = Phi(x, u, T) (the prediction, same call as
state_readout.counterfactual_values), z_star = the real future frame, y = target(s_A) (truth file, y_actual).
Readouts r_hat = R(z_hat), r_star = R(z_star), r_0 = R(x). Per duration T, summed over the n windows of that T:

Readout level (physical units, the probe's target):
  var_ratio      = sum||r_hat - mean(r_hat)||^2 / sum||y - mean(y)||^2        (< 1: predictions spread less than truth)
  slope          = sum<r_hat - mean(r_hat), y - mean(y)> / sum||y - mean(y)||^2 (regression of prediction on truth;
                                                                                  < 1: shrunk toward the mean)
  corr           = slope / sqrt(var_ratio)                                   (centred correlation, multi-output pooled)
  nmse           = mean ||r_hat - y||^2 / V   (V = probe variance, as in state_readout; cross-check with counterfactual)
  motion_gain    = sum<r_hat - r_0, y - r_0> / sum||y - r_0||^2  (share of the true displacement predicted; identity = 0)
  motion_ratio   = sqrt(sum||r_hat - r_0||^2 / sum||y - r_0||^2)
  *_star         the same with r_star in place of r_hat (probe sanity: ~1 where the probe floor is small)
LN feature level (per-token LayerNorm without affine, eps = state_readout.LN_EPS, flattened 256 x 1408):
  ln_var_ratio   = sum_d Var_n(LN z_hat)_d / sum_d Var_n(LN z_star)_d   (spread across windows of the same T)
  ln_motion_gain = sum<LN z_hat - LN x, LN z_star - LN x> / sum||LN z_star - LN x||^2
  ln_motion_ratio= sqrt(sum||LN z_hat - LN x||^2 / sum||LN z_star - LN x||^2)
  ln_motion_cos  = mean over windows of cos(LN z_hat - LN x, LN z_star - LN x)
Raw level (before LN; FJD and the probe are both invariant to a per-token shift and scale, so this is reported, not used):
  raw_norm_ratio = mean ||z_hat||_F / ||z_star||_F,  raw_token_std_ratio = mean over windows of mean_token std(z_hat)/std(z_star)
  fjd            = mean jepa_distance(z_hat, z_star) (losses.jepa_distance; should match the evaluation report's
                   grid-16 future_jepa_distance for T = 1, 2, 4, 8 up to the different control packing)

Usage (same --job/--probe/--truth/--features as `state_readout.py counterfactual`; GPU):
  python tools/shrinkage_check.py --job <results>/c2/dyadic-zero/double_well/seed1703/po_jepa_full \\
      --probe <results>/readout/probe/double_well --truth <results>/readout/truth/double_well.json \\
      --features ../features-v7/double_well-2000 --output <out>/double_well/po_jepa_full.json --device cuda:0
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DURATIONS = (1., 2., 4., 8.)


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", type=Path, required=True, help="run.py run output directory of the job (base scene)")
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True, help="main-experiment cache of the base scene (33 frames)")
    p.add_argument("--output", type=Path, required=True, help="result JSON (must not exist)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ac-checkpoint", type=Path, help="needed when the condition is vjepa2_ac")
    p.add_argument("--cross-version", action="store_true",
                   help="job was trained by another code version (v7.1 runs); read as in state_readout.py counterfactual")
    a = p.parse_args(argv)
    a.command = "counterfactual"              # state_readout._load_job / _load_cross_version read the main cache
    return a


class Accumulator:
    """Streaming sums for one duration. Readouts are kept (n x o, small); LN features only as running sums."""

    def __init__(self):
        self.y, self.r_hat, self.r_star, self.r_0 = [], [], [], []
        self.n = 0
        self.ln_sum_hat = self.ln_sq_hat = self.ln_sum_star = self.ln_sq_star = None
        self.ln_inner = self.ln_true = self.ln_pred = 0.
        self.ln_cos, self.raw_norm, self.raw_std, self.fjd = [], [], [], []

    def add(self, *, x, z_hat, z_star, y, r_hat, r_star, r_0, fjd, ln_eps):
        import torch
        from torch.nn import functional as F
        self.y.append(y); self.r_hat.append(r_hat); self.r_star.append(r_star); self.r_0.append(r_0)
        self.fjd.append(float(fjd))
        ln = [F.layer_norm(t.double(), (t.shape[-1],), eps=ln_eps).reshape(-1) for t in (x, z_hat, z_star)]
        lx, lh, ls = ln
        if self.ln_sum_hat is None:
            zeros = torch.zeros_like(lh)
            self.ln_sum_hat, self.ln_sq_hat, self.ln_sum_star, self.ln_sq_star = (zeros.clone() for _ in range(4))
        self.ln_sum_hat += lh; self.ln_sq_hat += lh*lh; self.ln_sum_star += ls; self.ln_sq_star += ls*ls
        dh, ds = lh-lx, ls-lx
        self.ln_inner += float(dh@ds); self.ln_true += float(ds@ds); self.ln_pred += float(dh@dh)
        denominator = float(dh.norm()*ds.norm())
        self.ln_cos.append(float(dh@ds)/denominator if denominator > 0 else float("nan"))
        self.raw_norm.append(float(z_hat.double().norm()/z_star.double().norm()))
        self.raw_std.append(float((z_hat.double().std(dim=-1)/z_star.double().std(dim=-1)).mean()))
        self.n += 1

    def summary(self, variance):
        import numpy as np
        y, r_hat, r_star, r_0 = (np.asarray(v, dtype=np.float64) for v in (self.y, self.r_hat, self.r_star, self.r_0))
        out = dict(windows=self.n)
        out.update(readout_summary(y, r_hat, r_0, variance))
        out.update({f"{k}_star": v for k, v in readout_summary(y, r_star, r_0, variance).items()})
        n = self.n
        var_hat = float((self.ln_sq_hat-self.ln_sum_hat**2/n).sum())/n
        var_star = float((self.ln_sq_star-self.ln_sum_star**2/n).sum())/n
        cos = [c for c in self.ln_cos if c == c]
        out.update(ln_var_ratio=var_hat/var_star if var_star > 0 else None,
                   ln_motion_gain=self.ln_inner/self.ln_true if self.ln_true > 0 else None,
                   ln_motion_ratio=(self.ln_pred/self.ln_true)**.5 if self.ln_true > 0 else None,
                   ln_motion_cos=float(np.mean(cos)) if cos else None, ln_motion_cos_windows=len(cos),
                   raw_norm_ratio=float(np.mean(self.raw_norm)), raw_token_std_ratio=float(np.mean(self.raw_std)),
                   fjd=float(np.mean(self.fjd)))
        return out


def readout_summary(y, r, r_0, variance):
    """y, r, r_0: [n, o] float64. Formulas in the module docstring."""
    import numpy as np
    yc, rc = y-y.mean(axis=0), r-r.mean(axis=0)
    sst = float((yc*yc).sum())
    dt, dp = y-r_0, r-r_0
    true_motion = float((dt*dt).sum())
    out = dict(var_ratio=float((rc*rc).sum())/sst if sst > 0 else None,
               slope=float((rc*yc).sum())/sst if sst > 0 else None,
               nmse=float(((r-y)**2).sum(axis=1).mean())/variance,
               motion_gain=float((dp*dt).sum())/true_motion if true_motion > 0 else None,
               motion_ratio=(float((dp*dp).sum())/true_motion)**.5 if true_motion > 0 else None)
    out["corr"] = (out["slope"]/out["var_ratio"]**.5 if out["var_ratio"] else None) if sst > 0 else None
    return out


def run(a, state_readout_tool):
    import numpy as np
    import torch
    from po_jepa import datasets, losses
    from po_jepa.runtime import data, evaluation, state_readout
    job_dir, job, base, probe, probe_meta, truth, model, loss, selected, registration, device = state_readout_tool._load_job(a)
    started = time.perf_counter()
    identity = data.approved_cache_metadata(a.features, splits=("test",))
    if identity["encode_report_sha256"] != truth["cache_identity"]["encode_report_sha256"]:
        raise SystemExit("--features is not the main cache that --truth was checked against")
    expected = [(tid, s, t) for tid in base.split_range("test") for s in datasets.test_start_ticks(tid)
                for t in state_readout.CF_DURATIONS]
    if [(r["trajectory_id"], r["start_tick"], r["duration_cycles"]) for r in truth["counterfactual"]] != expected:
        raise SystemExit("--truth windows do not match the registered test windows")
    if tuple(state_readout.CF_DURATIONS) != DURATIONS:
        raise SystemExit("state_readout.CF_DURATIONS changed")
    acc = {t: Accumulator() for t in DURATIONS}
    modes, state = state_readout_tool._eval_mode(model)
    try:
        with torch.no_grad():
            for r in truth["counterfactual"]:
                q = evaluation.EvaluationQuery(r["trajectory_id"], r["start_tick"], r["duration_cycles"], 16, "point")
                w = evaluation.load_test_window(a.features, q, device)
                if w.provenance["actual_manifest_sha256"] != r["manifest_sha256"]:
                    raise SystemExit(f"trajectory {q.trajectory_id}: manifest in --truth differs from the cache")
                x, z_star = w.inputs.raw_z, w.target.raw_z
                z_hat = model(x, w.inputs.controls, w.inputs.duration, w.inputs.grid)
                r_hat, r_star, r_0 = probe(torch.cat((z_hat, z_star, x)))
                y = np.asarray(r["y_actual"], dtype=np.float64)
                if y.shape != r_hat.shape:
                    raise ValueError("truth and probe outputs differ in shape")
                fjd = losses.jepa_distance(z_hat, z_star, config=loss).per_sample
                acc[float(q.duration_cycles)].add(x=x, z_hat=z_hat, z_star=z_star, y=y, r_hat=r_hat, r_star=r_star, r_0=r_0,
                                                  fjd=float(fjd.reshape(-1)[0]), ln_eps=state_readout.LN_EPS)
    finally:
        state_readout_tool._restore(model, modes, state)
    by_duration = {f"{t:g}": acc[t].summary(probe.variance) for t in DURATIONS}
    report = state_readout_tool._report("shrinkage_check", a, job_dir, job, base, probe_meta, truth, selected, registration, dict(
        dataset=base.tag, durations=list(DURATIONS), grid_size=16, by_duration=by_duration, device=str(device),
        seconds=round(time.perf_counter()-started, 1)))
    state_readout_tool._write(a.output, report)
    s = by_duration["8"]
    print(f"{base.tag} {job['condition']}: T=8 slope {s['slope']:.3g}, var_ratio {s['var_ratio']:.3g}, "
          f"motion_gain {s['motion_gain']:.3g}, ln_var_ratio {s['ln_var_ratio']:.3g} -> {a.output}")
    return report


def main(argv=None):
    a = _parse(argv)
    sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"tools"))
    import importlib.util
    import plan_v7                                        # does not import po_jepa at top level
    # the dataset environment must be set before po_jepa is first imported (G4 R3)
    a.job_dir, a.job_record = plan_v7._job_env(a.job)
    from po_jepa import datasets
    if datasets.active().tag != datasets.parse(os.environ["PO_JEPA_DATASET"]).tag:
        datasets._set_for_tests(datasets.parse(os.environ["PO_JEPA_DATASET"]))
    spec = importlib.util.spec_from_file_location("state_readout_tool", ROOT/"tools/state_readout.py")
    tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    run(a, tool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
