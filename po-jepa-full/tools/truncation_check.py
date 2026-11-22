"""G10 (evaluation only, no training): is the spatial-mode truncation what limits PO-JEPA's direct prediction?

Background (direct-prediction diagnosis, 2026-11-22): with the residual readout and a linear P_out the PO-JEPA change is
  Z_hat - Z = Psi_K (C_tau - C_0) W_out^T          (biases cancel, model.py _predict_raw_with_trace)
so whatever the dynamics learn, the predicted change lies in span(Psi_K) (x) R^E, Psi_K = the K = 16 lowest graph-Laplacian
modes of the start frame (model.spectrum.spectrum_for on the same view grid as the model). This tool checks that claim on the
C-2 test windows: every test trajectory x registered start x T in {1, 2, 4, 8} cycles x grid in {16, 12, 8} (the "point"
queries of evaluation.build_index with those T), coarse grids through evaluation.spatial_view (same transfer R as the
evaluation). The direct prediction is the one evaluation.point_metrics scores as future_jepa_distance (controls split at T/2
and concatenated again, same call). d(P, Y) = losses.jepa_distance (per-token LN without affine, mean |.| over N x E), the
evaluation metric. Per (grid, T) everything is a window mean over the n windows, and
  gain(x) = 1 - mean_w d(x) / mean_w d(Z),   Z = identity prediction (the "better than identity" share of the C-2 tables).

Three subcommands, each one JSON (refuses to overwrite; written via .partial at the end):

ceiling  --job <a PO-JEPA job: its spectrum module gives Psi, its P_out gives W_out>
  Oracle ceilings that use the true future Z* (no predictor can reach them; they only compare output structures). For each
  subspace S, minimise d(Z + D, Z*) over D in S by Adam (lr --lr, cosine to 0 over --steps, start from the L2 projection
  of Delta = Z* - Z onto S; the reported distance is the smallest one met at any step or at D = 0 (the identity, which
  lies in every S), i.e. an achievable point of S, so the gain is >= 0 and a lower bound of the true ceiling):
    modal_K16        D = Psi_16 A,            A in R^{16 x E}     (contains every PO-JEPA change, any W_out)
    modal_K16_pout   D = Psi_16 A Q^T,        A in R^{16 x 64}    (Q = orthonormal basis of col-span(W_out) of this job:
                                                                    exactly the changes the trained PO-JEPA readout can make)
    modal_K64        D = Psi_64 A,            A in R^{64 x E}     (option A; on 8x8, K = N = 64 and S is everything)
    token_U64        D = A U^T,               A in R^{N x 64}     (per-token free, 64-dim channel space = the residual
                                                                    Transformer's form; U = top-64 principal directions of
                                                                    the view-grid frame changes Z_{s+T} - Z_s on the
                                                                    validation split, starts 0, 5, 10, 15, 20, T in {1,2,4,8})
  l2_<S> is the distance at the L2 start point. Same construction as the 2026-11-22 probe subspace_ceiling_probe.py
  (which had 1000-1500 steps and 4 validation trajectories), now on the test windows of the cache and on all three grids.

project  --job <any trained job, normally transformer_consistency> --spectrum-job <po_jepa_full of the same scene/seed>
  Delta = Phi(Z) - Z of --job; Pi_K = Psi_K Psi_K^T (Psi from --spectrum-job's spectrum module on the start frame, view grid).
    d_identity, d_model = d(Z + Delta, Z*), d_proj_K = d(Z + Pi_K Delta, Z*) and d_perp_16 = d(Z + (I - Pi_16) Delta, Z*)
    energy_in_span_K = sum_w ||Pi_K Delta||^2 / sum_w ||Delta||^2
  and the same for --spectrum-job's own prediction, where it is a check: its change must lie in span(Psi_16)
  (offspan_16 = max_w ||(I - Pi_16) Delta|| / ||Delta|| ~ float32 rounding; it sees only the directions the prediction
  uses, so it does not prove the two subspaces are equal). How to read it (G10 review P2-2): Pi_K Delta is the Euclidean
  projection of the change, not the best point of span(Psi_K) under the LN-L1 metric. A drop of the Transformer's gain
  after Pi_16 shows that what it learned depends on the removed directions and supports the truncation explanation; it
  does not by itself prove that the truncation explains the whole gap, and it is not a capacity bound (that is the
  ceiling, whose optimised value is itself only a lower bound of the true optimum). When <job>/evaluation/<condition>/rows exist, d_model is
  compared window by window with the stored future_jepa_distance (evaluation_check: max |difference|, not an error).

dynamics --job <a PO-JEPA job with forcing_space: token>
  On the same windows (direct prediction, trace_forward), at every actual integration step the right-hand side is
  recomputed from the traced state C and that step's action (checked equal to the traced derivative), and per (grid, T)
  over all steps of all windows it reports: the mean Frobenius norms ||C||, ||A C||, ||Psi^T B(X)a||, ||Psi^T F||,
  ||Psi^T attention||, ||Psi^T M|| (each [K, 64]) and of the token-space B(X)a, attention aggregate, residual M and F = agg + M
  ([N, 64]); the shares of token-space energy that survive Psi^T (sum ||Psi^T v||^2 / sum ||v||^2, v = B(X)a and F); the
  top-r attention: mean max weight, mean entropy (nats) and entropy / log r, std of the content scores and of the total
  scores (content + relative-position bias), all over tokens of every step.

Usage (GPU; every C-2 line needs --cross-version, see GPU_TASKS.md section 8d):
  python tools/truncation_check.py ceiling  --job <C2>/double_well/seed1703/po_jepa_full --cross-version \\
      --features ../features-v7/double_well-2000 --output <out>/ceiling/double_well.json --device cuda:0
  python tools/truncation_check.py project  --job <C2>/double_well/seed1703/transformer_consistency \\
      --spectrum-job <C2>/double_well/seed1703/po_jepa_full --cross-version --features ... --output ... --device cuda:0
  python tools/truncation_check.py dynamics --job <C2>/double_well/seed1703/po_jepa_full --cross-version --features ... \\
      --output ... --device cuda:0
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GRIDS = (16, 12, 8)
DURATIONS = (1., 2., 4., 8.)
FIT_STARTS = (0, 5, 10, 15, 20)
CEILINGS = ("modal_K16", "modal_K16_pout", "modal_K64", "token_U64")
PROJECT_KS = (16, 64)
K = 16


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("ceiling", "project", "dynamics"):
        s = sub.add_parser(name)
        s.add_argument("--job", type=Path, required=True, help="run.py run output directory (base scene)")
        s.add_argument("--features", type=Path, required=True, help="main-experiment cache of the base scene (33 frames)")
        s.add_argument("--output", type=Path, required=True, help="result JSON (must not exist)")
        s.add_argument("--device", default="cuda:0")
        s.add_argument("--cross-version", action="store_true",
                       help="jobs were trained by another code version (C-2: d5a8fc1); read as state_readout.py --cross-version")
        s.add_argument("--grids", default="16,12,8", help="subset of 16,12,8 (tests)")
    sub.choices["project"].add_argument("--spectrum-job", type=Path, required=True,
                                        help="PO-JEPA job of the same scene, seed and variant: source of Psi")
    c = sub.choices["ceiling"]
    c.add_argument("--steps", type=int, default=6000)
    c.add_argument("--lr", type=float, default=.5)
    c.add_argument("--batch", type=int, default=100, help="windows optimised together (independent; memory only)")
    a = p.parse_args(argv)
    a.grids = tuple(int(g) for g in a.grids.split(","))
    if not a.grids or any(g not in GRIDS for g in a.grids) or len(set(a.grids)) != len(a.grids):
        p.error("--grids must be a subset of 16,12,8")
    if a.command == "ceiling" and (a.steps < 1 or a.batch < 1 or not a.lr > 0):
        p.error("--steps, --batch and --lr must be positive")
    return a


# —— loading ——

def _find_job(path):
    """Same search as plan_v7._job_env, without touching the environment."""
    for sub in ("run/job", "job", "run", "."):
        if (path/sub/"job.json").exists():
            d = (path/sub).resolve()
            return d, json.loads((d/"job.json").read_text())
    raise SystemExit(f"{path}: no job.json")


def _load(job_dir, job, a, device):
    """(model, loss, selected, registration) as state_readout.py rollout/counterfactual load a job."""
    from po_jepa import datasets
    from po_jepa.runtime import comparison, v7jobs
    base = datasets.active()
    scene, _, count = job["dataset"].rpartition("-")
    if base.tag != datasets.parse(f"{scene}:{count}").tag:
        raise SystemExit(f"active dataset {base.tag} differs from the job's {job['dataset']}")
    if job.get("kind"):
        raise SystemExit(f"{job_dir} is a {job['kind']} job; a run.py run directory is needed")
    if not a.cross_version:
        model, loss, selected, registration = v7jobs.load_for_evaluation(
            job["condition"], None if job["condition"] == "identity" else job_dir, device)
        return model, loss, selected, dict(registration_sha256=_sha(registration))
    from po_jepa.runtime.addon import v7load
    model, loss = comparison.build_condition(job["condition"], device)
    try:
        record = v7load.load_v7_selected(job_dir, job["condition"], model, loss_config=loss, features=a.features, device=device)
    except ValueError as error:
        raise SystemExit(f"{job_dir}: cross-version checkpoint read failed ({error})")
    best = record["best"]
    selected = None if best is None else dict(step=best["step"], checkpoint_sha256=best["checkpoint_sha256"])
    return model, loss, selected, dict(registration_sha256=None, cross_version=record)


def _sha(value):
    from po_jepa.runtime import cache
    return hashlib.sha256(cache._json_bytes(value)).hexdigest()


def _po_model(model, job, what):
    from po_jepa.model import POJEPA
    if not isinstance(model, POJEPA):
        raise SystemExit(f"{what} must be a PO-JEPA job (condition {job['condition']} has no graph spectrum)")
    if model.dynamics.complement_mode:
        # C3: the decomposition here (projection onto Psi_16 / Psi_64, modal right-hand side) assumes the truncated model
        raise SystemExit(f"{what} keeps the spectral complement (C3 scheme B); this tool is for the truncated model only")
    return model


def _eval_mode(model):
    from po_jepa.runtime import evaluation
    modes = {m: m.training for m in model.modules()}
    model.eval()
    return modes, evaluation._state_identity(model)


def _restore(model, modes, identity):
    from po_jepa.runtime import evaluation
    for module, flag in modes.items():
        module.training = flag
    if evaluation._state_identity(model) != identity:
        raise RuntimeError("truncation_check changed parameters, gradients or buffers")


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".partial")
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def _job_fields(job_dir, job, selected, registration):
    return dict(job_dir=str(job_dir), condition=job["condition"], seed=job["seed"], updates=job["updates"],
                variant=job.get("variant"),
                best=None if selected is None else dict(step=selected["step"], checkpoint_sha256=selected.get("checkpoint_sha256")),
                **registration)


# —— windows ——

def window_keys():
    """(trajectory, start, T) of the registered test windows, T in DURATIONS; all are point queries of build_index."""
    from po_jepa import datasets
    from po_jepa.runtime import evaluation
    ds = datasets.active()
    keys = [(tid, s, t) for tid in ds.split_range("test") for s in datasets.test_start_ticks(tid) for t in DURATIONS]
    index = {(q.trajectory_id, q.start_tick, float(q.duration_cycles), q.grid_size) for q in evaluation.build_index()
             if q.kind == "point"}
    if any((tid, s, t, g) not in index for tid, s, t in keys for g in GRIDS):
        raise SystemExit("a window is not a registered evaluation point")
    return keys


def load_window(features, key, device):
    from po_jepa.runtime import evaluation
    q = evaluation.EvaluationQuery(key[0], key[1], key[2], 16, "point")
    w = evaluation.load_test_window(features, q, device)
    evaluation._recipient(w, q)
    return w


def direct_controls(w):
    """The controls evaluation.point_metrics passes for future_jepa_distance (split at T/2, concatenated again)."""
    from po_jepa.runtime import comparison
    controls, duration = w.inputs.controls, w.inputs.duration
    tau = duration*.5
    pair = comparison.split_action_segments(controls, tau=tau, sigma=tau)
    return comparison.concatenate_action_segments(pair.first, pair.second), duration


def basis(model, raw, grid, k, device):
    """Psi_k [B, N, k] of the start frame on the view grid, the model's own spectrum module (model float32 copy)."""
    graph = model.spectrum.config
    source = raw if graph.affinity_source == "encoder" else model.project_in(raw)
    return model.spectrum.spectrum_for(source, grid, k=k, model_device=device).basis


def project(psi, delta):
    """Pi delta = Psi (Psi^T delta), Psi with orthonormal columns: [B, N, k] x [B, N, E]."""
    import torch
    return torch.bmm(psi, torch.bmm(psi.transpose(1, 2), delta))


def _stored_evaluation(job_dir, condition):
    folder = Path(job_dir)/"evaluation"/condition/"rows"
    if not folder.is_dir():
        return None
    out = {}
    for path in sorted(folder.glob("*.json")):
        row = json.loads(path.read_text())
        q = row["query"]
        if q["kind"] == "point" and row["status"] == "ok":
            out[(q["trajectory_id"], q["start_tick"], float(q["duration_cycles"]), q["grid_size"])] = \
                row["values"]["future_jepa_distance"]
    return out


def gain(d, d_identity):
    return 1.-d/d_identity if d_identity > 0 else None


# —— ceiling ——

def channel_basis(features, grid16, grid_size, device, k=64):
    """U [E, k]: top-k eigenvectors of sum Delta^T Delta over the validation split (view grid; float64), explained share."""
    import numpy as np
    import torch
    from po_jepa import datasets
    from po_jepa.runtime import cache, comparison, data
    ds = datasets.active()
    approved = data.approved_cache_metadata(features, splits=("validation",))
    cov, transfer = None, None
    for tid in ds.split_range("validation"):
        raw, m, sha = cache._read_complete_cache(features/f"{tid:04d}")
        if m["trajectory_id"] != tid or m["split"] != "validation" or sha != approved["manifest_sha256_by_id"][tid]:
            raise SystemExit(f"trajectory {tid}: not the validation cache of this dataset")
        z = torch.from_numpy(np.array(raw, dtype=np.float32)).to(device)
        if grid_size != 16:
            if transfer is None:
                transfer = comparison.build_spatial_transfer(grid16, target_height=grid_size, target_width=grid_size)
            z = comparison.apply_spatial_transfer(z, transfer)
        z = z.double()
        for s in FIT_STARTS:
            for t in DURATIONS:
                if s+int(t) < z.shape[0]:
                    d = z[s+int(t)]-z[s]
                    cov = d.T@d if cov is None else cov+d.T@d
    values, vectors = torch.linalg.eigh(cov)
    order = torch.argsort(values, descending=True)
    u = vectors[:, order[:k]].contiguous()
    return u.float(), float(values[order[:k]].sum()/values.sum())


def optimise(z, target, init, left, right, loss, steps, lr):
    """min over A of d(Z + left A right^T, Z*) per window (left/right None = no restriction on that side).
    Returns (distance at the start point, smallest distance met), float64 [B]."""
    import torch
    from po_jepa import losses

    def predict(a):
        u = a if left is None else torch.bmm(left, a)
        return z+(u if right is None else u@right.T)
    with torch.no_grad():
        start = losses.jepa_distance(predict(init), target, config=loss).per_sample.double()
    a = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best = start.clone()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        d = losses.jepa_distance(predict(a), target, config=loss).per_sample
        d.sum().backward()
        best = torch.minimum(best, d.detach().double())
        opt.step(); schedule.step()
    with torch.no_grad():
        best = torch.minimum(best, losses.jepa_distance(predict(a), target, config=loss).per_sample.double())
    return start, best


def ceiling_batch(z, target, psi16, psi64, q_out, u, loss, steps, lr, d_identity):
    """All ceilings for one batch of windows; returns {name: (l2 distance, ceiling distance)}. D = 0 is in every subspace,
    so the ceiling distance is never above d_identity (the L2 start can be: L2 is not the evaluation metric)."""
    import torch
    delta = target-z
    out = {}
    for name in CEILINGS:
        if name == "modal_K16":
            left, right, init = psi16, None, torch_bmm_t(psi16, delta)
        elif name == "modal_K16_pout":
            if q_out.shape[1] == 0:                  # W_out = 0: the trained readout can make no change at all
                out[name] = d_identity.clone(), d_identity.clone()
                continue
            left, right, init = psi16, q_out, torch_bmm_t(psi16, delta)@q_out
        elif name == "modal_K64":
            left, right, init = psi64, None, torch_bmm_t(psi64, delta)
        else:
            left, right, init = None, u, delta@u
        start, best = optimise(z, target, init, left, right, loss, steps, lr)
        out[name] = start, torch.minimum(best, d_identity)
    return out


def readout_basis(weight):
    """Orthonormal basis [E, r] of col-span(W_out), r = numerical rank (G10 review P2-1: a thin QR of a rank-deficient
    W returns 64 columns, most of them outside the span). Left singular vectors with s > max(E, 64) * eps * s_max, eps of
    the stored dtype (float32: torch.linalg.matrix_rank's default tolerance on the stored weight), SVD in float64,
    returned as float32."""
    import torch
    eps = torch.finfo(weight.dtype).eps
    w = weight.detach().double().cpu()
    u, singular, _ = torch.linalg.svd(w, full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) == 0.:
        return torch.zeros(w.shape[0], 0), 0
    keep = singular > max(w.shape)*eps*singular[0]
    return u[:, keep].contiguous().float(), int(keep.sum())


def torch_bmm_t(psi, delta):
    import torch
    return torch.bmm(psi.transpose(1, 2), delta)


def cmd_ceiling(a, job_dir, job, model, loss, selected, registration, device):
    import torch
    from po_jepa import losses
    from po_jepa.runtime import evaluation
    model = _po_model(model, job, "--job")
    keys = window_keys()
    started = time.perf_counter()
    q_out, rank = readout_basis(model.p_out.weight)                                 # [E, rank], col-span(W_out)
    q_out = q_out.to(device)
    results, channel = {}, {}
    cpu = torch.device("cpu")
    grid16 = load_window(a.features, keys[0], cpu).inputs.grid
    for g in a.grids:
        u, explained = channel_basis(a.features, grid16, g, device)
        channel[str(g)] = dict(explained_share_of_validation_changes=explained)
        by_t = {}
        for t in DURATIONS:
            sel = [load_window(a.features, key, cpu) for key in keys if key[2] == t]   # one duration on the CPU at a time
            sums = {"identity": 0., **{f"l2_{n}": 0. for n in CEILINGS}, **{n: 0. for n in CEILINGS}}
            for i in range(0, len(sel), a.batch):
                views = [evaluation.spatial_view(w, g) for w in sel[i:i+a.batch]]
                z = torch.cat([v[0] for v in views]).to(device); target = torch.cat([v[1] for v in views]).to(device)
                grid = views[0][2]
                n_tokens = z.shape[1]
                psi16 = basis(model, z, grid, K, device)
                psi64 = basis(model, z, grid, min(64, n_tokens), device)
                with torch.no_grad():
                    d0 = losses.jepa_distance(z, target, config=loss).per_sample.double()
                sums["identity"] += float(d0.sum())
                for name, (start, best) in ceiling_batch(z, target, psi16, psi64, q_out, u, loss, a.steps, a.lr, d0).items():
                    sums[f"l2_{name}"] += float(start.sum()); sums[name] += float(best.sum())
            n = len(sel)
            mean = {k: v/n for k, v in sums.items()}
            by_t[f"{t:g}"] = dict(windows=n, tokens=n_tokens, distance=mean,
                                  gain={k: gain(v, mean["identity"]) for k, v in mean.items() if k != "identity"},
                                  modal_K64_unrestricted=min(64, n_tokens) >= n_tokens)
        results[str(g)] = by_t
        print(f"grid {g}: " + "  ".join(f"T={t}: K16 {100*r['gain']['modal_K16']:.1f}% K64 {100*r['gain']['modal_K64']:.1f}% "
                                        f"U64 {100*r['gain']['token_U64']:.1f}%" for t, r in by_t.items()), flush=True)
    return dict(kind="truncation_ceiling", p_out_rank=rank, steps=a.steps, lr_cosine=a.lr, batch=a.batch,
                channel_basis=channel, results=results, seconds=round(time.perf_counter()-started, 1))


# —— project ——

def cmd_project(a, job_dir, job, model, loss, selected, registration, device, spectrum):
    import torch
    from po_jepa import losses
    from po_jepa.runtime import evaluation
    s_dir, s_job, s_model, s_loss, s_selected, s_registration = spectrum
    s_model = _po_model(s_model, s_job, "--spectrum-job")
    if s_loss.layer_norm_epsilon != loss.layer_norm_epsilon:
        raise SystemExit("--job and --spectrum-job use different LN epsilons")
    keys = window_keys()
    started = time.perf_counter()
    stored = {"model": _stored_evaluation(job_dir, job["condition"]), "spectrum_model": _stored_evaluation(s_dir, s_job["condition"])}
    results = {}
    checks = {name: dict(compared=0, max_abs_difference=None) for name in stored}
    offspan_max = 0.
    names = ("identity", "model", *(f"model_proj_K{k}" for k in PROJECT_KS), "model_perp_K16",
             "spectrum_model", *(f"spectrum_model_proj_K{k}" for k in PROJECT_KS))
    with torch.no_grad():
        for g in a.grids:
            sums = {t: dict(n=0, **{n: 0. for n in names}, **{f"{m}_energy_K{k}": 0. for m in ("model", "spectrum_model")
                                                                  for k in (*PROJECT_KS, 0)}) for t in DURATIONS}
            for key in keys:
                w = load_window(a.features, key, device)
                z, target, grid = evaluation.spatial_view(w, g)
                controls, duration = direct_controls(w)
                psi = {k: basis(s_model, z, grid, min(k, z.shape[1]), device) for k in PROJECT_KS}
                d = lambda p: float(losses.jepa_distance(p, target, config=loss).per_sample[0])
                s = sums[key[2]]
                s["n"] += 1
                s["identity"] += d(z)
                for name, m in (("model", model), ("spectrum_model", s_model)):
                    pred = m(z, controls, duration, grid)
                    value = d(pred)
                    s[name] += value
                    if stored[name] is not None:
                        ref = stored[name].get((key[0], key[1], key[2], g))
                        if ref is not None:
                            c = checks[name]; c["compared"] += 1
                            c["max_abs_difference"] = max(c["max_abs_difference"] or 0., abs(value-ref))
                    delta = pred-z
                    energy = float(delta.double().square().sum())
                    s[f"{name}_energy_K0"] += energy
                    for k in PROJECT_KS:
                        pk = project(psi[k], delta)
                        s[f"{name}_proj_K{k}"] += d(z+pk)
                        s[f"{name}_energy_K{k}"] += float(pk.double().square().sum())
                        if k == K:
                            perp = delta-pk
                            if name == "model":
                                s["model_perp_K16"] += d(z+perp)
                            elif energy > 0:
                                offspan_max = max(offspan_max, float(perp.double().norm())/energy**.5)
            by_t = {}
            for t in DURATIONS:
                s = sums[t]; n = s["n"]
                mean = {k: s[k]/n for k in names}
                by_t[f"{t:g}"] = dict(
                    windows=n, distance=mean, gain={k: gain(v, mean["identity"]) for k, v in mean.items() if k != "identity"},
                    energy_in_span={f"{m}_K{k}": (s[f"{m}_energy_K{k}"]/s[f"{m}_energy_K0"] if s[f"{m}_energy_K0"] > 0 else None)
                                    for m in ("model", "spectrum_model") for k in PROJECT_KS})
            results[str(g)] = by_t
            print(f"grid {g}: " + "  ".join(f"T={t}: model {100*r['gain']['model']:.1f}% ->Psi16 "
                                            f"{100*r['gain']['model_proj_K16']:.1f}% (PO {100*r['gain']['spectrum_model']:.1f}%)"
                                            for t, r in by_t.items()), flush=True)
    evaluation_check = {name: (None if stored[name] is None else checks[name]) for name in stored}
    return dict(kind="truncation_project", spectrum_job=_job_fields(s_dir, s_job, s_selected, s_registration),
                ks=list(PROJECT_KS), spectrum_model_offspan_16_max=offspan_max, evaluation_check=evaluation_check,
                results=results, seconds=round(time.perf_counter()-started, 1))


# —— dynamics ——

class DynamicsSums:
    """Sums over all integration steps of all windows of one (grid, T)."""
    NORMS = ("C", "AC", "PsiT_Ba", "PsiT_F", "PsiT_attention", "PsiT_M", "Ba_token", "attention_token", "M_token", "F_token")

    def __init__(self):
        self.windows = self.steps = self.tokens = 0
        self.norm = {k: 0. for k in self.NORMS}
        self.energy = {k: 0. for k in ("Ba_kept", "Ba_all", "F_kept", "F_all")}
        self.max_weight = self.entropy = self.entropy_ratio = 0.
        self.content_sq = self.content_sum = self.total_sq = self.total_sum = 0.
        self.content_count = 0

    def add_step(self, *, c, rhs, basis, drive):
        import torch
        nl = rhs.interaction
        fro = lambda x: float(x.double().norm())
        token_f = nl.attention.aggregate+nl.action_residual
        self.steps += 1
        for name, value in (("C", c), ("AC", rhs.free), ("PsiT_Ba", rhs.action_linear), ("PsiT_F", nl.nonlinear),
                            ("PsiT_attention", torch.bmm(basis.transpose(1, 2), nl.attention.aggregate)),
                            ("PsiT_M", torch.bmm(basis.transpose(1, 2), nl.action_residual)),
                            ("Ba_token", drive), ("attention_token", nl.attention.aggregate), ("M_token", nl.action_residual),
                            ("F_token", token_f)):
            self.norm[name] += fro(value)
        self.energy["Ba_kept"] += float(rhs.action_linear.double().square().sum())
        self.energy["Ba_all"] += float(drive.double().square().sum())
        self.energy["F_kept"] += float(nl.nonlinear.double().square().sum())
        self.energy["F_all"] += float(token_f.double().square().sum())
        w = nl.attention.weights.double()                                 # [1, N, r]
        r = w.shape[-1]
        entropy = -torch.special.xlogy(w, w).sum(-1)
        self.tokens += w.shape[1]
        self.max_weight += float(w.max(-1).values.sum())
        self.entropy += float(entropy.sum())
        self.entropy_ratio += float(entropy.sum())/math.log(r) if r > 1 else 0.
        content, total = nl.content.scores.double(), nl.total_scores.double()
        self.content_count += content.numel()
        self.content_sum += float(content.sum()); self.content_sq += float(content.square().sum())
        self.total_sum += float(total.sum()); self.total_sq += float(total.square().sum())
        self.r = r

    def summary(self):
        s, n = self.steps, self.content_count
        std = lambda total, sq: math.sqrt(max(sq/n-(total/n)**2, 0.))
        return dict(windows=self.windows, steps=s, top_r=getattr(self, "r", None),
                    mean_norm={k: v/s for k, v in self.norm.items()},
                    kept_share_after_PsiT=dict(Ba=self.energy["Ba_kept"]/self.energy["Ba_all"] if self.energy["Ba_all"] > 0 else None,
                                               F=self.energy["F_kept"]/self.energy["F_all"] if self.energy["F_all"] > 0 else None),
                    attention=dict(mean_max_weight=self.max_weight/self.tokens, mean_entropy=self.entropy/self.tokens,
                                   mean_entropy_over_log_r=self.entropy_ratio/self.tokens,
                                   content_score_std=std(self.content_sum, self.content_sq),
                                   total_score_std=std(self.total_sum, self.total_sq)))


def step_action(clipped, step):
    """The action integrate_segments used at this step: segment segment_index of the clipped controls."""
    rows = step.batch_index.to(clipped.u_values.device)
    return clipped.u_values[rows, step.segment_index.to(clipped.u_values.device)]


def cmd_dynamics(a, job_dir, job, model, loss, selected, registration, device):
    import torch
    from po_jepa.actions import clip_action_segments
    from po_jepa.runtime import evaluation
    model = _po_model(model, job, "--job")
    if not model.dynamics.token_mode:
        raise SystemExit("dynamics needs forcing_space: token")
    keys = window_keys()
    started = time.perf_counter()
    results = {}
    with torch.no_grad():
        for g in a.grids:
            acc = {t: DynamicsSums() for t in DURATIONS}
            for key in keys:
                w = load_window(a.features, key, device)
                z, target, grid = evaluation.spatial_view(w, g)
                controls, duration = direct_controls(w)
                trace = model.trace_forward(z, controls, duration, grid)
                ctx = trace.query.dynamics
                clipped = clip_action_segments(controls, start=torch.zeros_like(duration), end=duration)
                acc[key[2]].windows += 1
                if not trace.integration.steps:
                    raise SystemExit("no integration step traced")
                for step in trace.integration.steps:
                    if not torch.equal(step.batch_index, torch.zeros(1, dtype=torch.int64)):
                        raise SystemExit("one window per forward expected")
                    action = step_action(clipped, step)
                    rhs = model.dynamics.rhs(step.before, action, ctx)
                    if not torch.allclose(rhs.derivative, step.result.rhs.derivative, rtol=1e-5, atol=1e-6):
                        raise SystemExit("recomputed right-hand side differs from the traced one (wrong action?)")
                    drive = torch.einsum("bncd,bd->bnc", ctx.b_field, action)
                    acc[key[2]].add_step(c=step.before, rhs=rhs, basis=ctx.basis, drive=drive)
            results[str(g)] = {f"{t:g}": acc[t].summary() for t in DURATIONS}
            r = results[str(g)]["8"]
            print(f"grid {g} T=8: max weight {r['attention']['mean_max_weight']:.3g}, entropy/log r "
                  f"{r['attention']['mean_entropy_over_log_r']:.3g}, |AC| {r['mean_norm']['AC']:.3g}, "
                  f"|Psi^T F| {r['mean_norm']['PsiT_F']:.3g}, F kept {r['kept_share_after_PsiT']['F']:.3g}", flush=True)
    return dict(kind="truncation_dynamics", results=results, seconds=round(time.perf_counter()-started, 1))


def run(a):
    import torch
    from po_jepa import datasets
    if a.output.exists():
        raise SystemExit(f"{a.output} exists; not overwriting")
    device = torch.device(a.device)
    job_dir, job = a.job_dir, a.job_record
    loaded = _load(job_dir, job, a, device)
    model, loss, selected, registration = loaded
    extra = ()
    modes = [_eval_mode(model)]
    if a.command == "project":
        s_dir, s_job = _find_job(a.spectrum_job)
        for k in ("dataset", "seed", "updates"):
            if s_job.get(k) != job.get(k):
                raise SystemExit(f"--spectrum-job {k} {s_job.get(k)} differs from --job {job.get(k)}")
        if (s_job.get("variant") or {}) != (job.get("variant") or {}):
            raise SystemExit("--spectrum-job was trained with another variant than --job")
        s_loaded = _load(s_dir, s_job, a, device)
        modes.append(_eval_mode(s_loaded[0]))
        extra = ((s_dir, s_job, *s_loaded),)
    try:
        body = {"ceiling": cmd_ceiling, "project": cmd_project, "dynamics": cmd_dynamics}[a.command](
            a, job_dir, job, model, loss, selected, registration, device, *extra)
    finally:
        _restore(model, *modes[0])
        if extra:
            _restore(extra[0][2], *modes[1])
    report = dict(body, dataset=datasets.active().tag, grids=list(a.grids), durations=list(DURATIONS),
                  **_job_fields(job_dir, job, selected, registration), features=str(a.features), device=str(device),
                  scientific_conclusions_verified=False)
    _write(a.output, report)
    print(f"{report['kind']} {job['condition']} -> {a.output}")
    return report


def main(argv=None):
    a = _parse(argv)
    sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"tools"))
    import plan_v7                                        # does not import po_jepa at top level
    # the dataset environment must be set before po_jepa is first imported (G4 R3)
    a.job_dir, a.job_record = plan_v7._job_env(a.job)
    from po_jepa import datasets
    if datasets.active().tag != datasets.parse(os.environ["PO_JEPA_DATASET"]).tag:
        datasets._set_for_tests(datasets.parse(os.environ["PO_JEPA_DATASET"]))
    run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
