"""Extra evaluation of already-trained v7 runs (no training, analysis only).

For every finished job of po_jepa_full / po_jepa_no_consistency / matched_transformer / identity, load the checkpoint
that validation selected (same file and SHA the registered evaluation used) and measure, on the 16x16 test windows,
the JEPA distance to the true future frame of several ways of reaching the same horizon T:

  id      : the current tokens Z_t (identity reference)
  direct  : Phi_T(Z_t)                                    (one call, what the registered evaluation reports)
  half    : Phi_{T/2}(Phi_{T/2}(Z_t))                     (the composed path of the temporal consistency loss)
  unit    : T steps of 1 cycle, each from the previous prediction (v7.1 unit rollout)
  p2_4 / p4_2 : 6 = 2+4 and 6 = 4+2                       (unequal splits, only for T = 6)
  dc_half : JEPA distance between direct and half         (temporal defect in the prediction metric)

Durations 1, 2, 3, 4, 6, 8, 12, 16 cycles; 6 and 16 are outside the registered evaluation (never trained, never
evaluated) and are reported separately. Windows whose end frame does not exist are skipped. Nothing here feeds back
into training, checkpoint selection or the registered tables.

Usage (inside the po-jepa directory; one process per job is spawned automatically):
  python tools/extra_eval_v7.py run --results-root ../results-v7/pilot --features-base ../features-v7 \
      --output ../extra-eval-v7/pilot --device cuda:0
(each job's own --features-root is read from <job>/console/command.json; --features-base is only the fallback)
  python tools/extra_eval_v7.py summarize --output ../extra-eval-v7/pilot
"""
import argparse
import glob
import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity")
DURATIONS = (1., 2., 3., 4., 6., 8., 12., 16.)
REGISTERED = (1., 2., 3., 4., 8., 12.)
EXTRA_SPLITS = {6.: {"p2_4": (2., 4.), "p4_2": (4., 2.)}}


def dataset_env(tag: str) -> str:
    """job.json stores the dataset tag '<scene>-<count>'; run.py's environment variable is '<scene>:<count>'."""
    scene, sep, count = tag.rpartition("-")
    if not sep or not count.isdigit():
        raise ValueError(f"unexpected dataset tag {tag!r}")
    return f"{scene}:{count}"


def find_jobs(results_root: Path):
    """All finished training jobs (job.json without 'kind', i.e. not transfer/equal-step) of the four conditions."""
    jobs = []
    for path in sorted(glob.glob(str(results_root/"**"/"run"/"job"/"job.json"), recursive=True)):
        job = json.loads(Path(path).read_text(encoding="utf-8"))
        if job.get("kind") or job.get("condition") not in CONDITIONS:
            continue
        if not (Path(path).parent/"COMPLETE.json").exists():
            continue
        jobs.append((Path(path).parent, job))
    return jobs


def features_for(job_dir: Path, job: dict, features_base):
    """The feature cache the job was trained on: --features-root recorded in <output>/console/command.json
    (job_dir is <output>/run/job); falls back to <features-base>/<dataset tag>."""
    record = job_dir.parent.parent/"console"/"command.json"
    if record.exists():
        argv = json.loads(record.read_text(encoding="utf-8"))["argv"]
        if "--features-root" in argv:
            path = Path(argv[argv.index("--features-root")+1])
            if path.is_dir():
                return path
    if features_base is None:
        raise ValueError(f"{job_dir}: no recorded feature root found; pass --features-base")
    return Path(features_base).resolve()/job["dataset"]


# ---------------------------------------------------------------- numerics (importable without a dataset)

def partition_rollout(model, raw, controls, parts, grid, split):
    """Phi_{parts[-1]}(...Phi_{parts[0]}(raw)): each piece starts from the previous prediction and sees only its own
    actions. split(controls, tau, sigma) must cut a control window of duration tau+sigma into (first tau, rest)."""
    import torch
    if not parts or any(p <= 0 for p in parts):
        raise ValueError("partition pieces must be positive")
    state, rest, remaining = raw, controls, float(sum(parts))
    for piece in parts[:-1]:
        remaining -= piece
        tau = torch.tensor([float(piece)], dtype=torch.float64)
        sigma = torch.tensor([remaining], dtype=torch.float64)
        pair = split(rest, tau=tau, sigma=sigma)
        state, rest = model(state, pair.first, tau, grid), pair.second
    return model(state, rest, torch.tensor([float(parts[-1])], dtype=torch.float64), grid)


def window_metrics(model, raw, target, controls, T, grid, distance, split):
    """All distances for one window; distance(pred, target) -> float."""
    import torch
    out = dict(id=distance(raw, target))
    direct = model(raw, controls, torch.tensor([T], dtype=torch.float64), grid)
    half = partition_rollout(model, raw, controls, (T/2, T/2), grid, split)
    out.update(direct=distance(direct, target), half=distance(half, target), dc_half=distance(direct, half))
    if float(T).is_integer():
        out["unit"] = distance(partition_rollout(model, raw, controls, (1.,)*int(T), grid, split), target)
    for name, parts in EXTRA_SPLITS.get(T, {}).items():
        out[name] = distance(partition_rollout(model, raw, controls, parts, grid, split), target)
    return out


# ---------------------------------------------------------------- one job (dataset set through the environment)

def load_selected(condition, job_dir, device):
    """Model of `condition` with the checkpoint validation selected: SHA, step, architecture and strict state check.
    Optimizer/RNG/runtime are not restored (evaluation only), so a different GPU index or driver does not matter."""
    import torch
    from dataclasses import asdict
    from po_jepa.runtime import comparison, training
    model, loss = comparison.build_condition(condition, device)
    if condition == "identity":
        return model.eval(), loss, None
    job = json.loads((job_dir/"job.json").read_text(encoding="utf-8"))
    root = job_dir/"training"/condition
    if not (root/"COMPLETE.json").exists() and job.get("training_run"):
        old = Path(job["training_run"])
        for sub in ("run/job", "job", "run", ""):
            if (old/sub/"job.json").exists():
                old = old/sub
                break
        root = old/"training"/condition
    if not (root/"COMPLETE.json").exists() or (root/"FAILED.json").exists():
        raise ValueError(f"{root}: training not complete")
    selected = json.loads((root/"report.json").read_text(encoding="utf-8"))["best"]
    path = (root/selected["checkpoint"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("selected checkpoint escapes its run")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != selected["checkpoint_sha256"]:
        raise ValueError(f"{path}: checkpoint SHA mismatch")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if payload["completed_steps"] != selected["step"]:
        raise ValueError("checkpoint step differs from the selected validation step")
    if condition == "matched_transformer":
        if payload["config"] != asdict(model.spec):
            raise ValueError("Transformer architecture differs from the checkpoint")
    else:
        expected = "full" if condition == "po_jepa_full" else "no_consistency"
        if payload["condition"] != expected:
            raise ValueError("checkpoint condition mismatch")
        if not training._same_native(payload["model_config"], training._model_config(model)):
            raise ValueError("PO-JEPA model configuration differs from the checkpoint")
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval(), loss, dict(step=selected["step"], sha256=selected["checkpoint_sha256"],
                                    path=str(path.relative_to(root.resolve())), validation_mean=selected["mean"])


def read_test_window(features, approved, trajectory_id, start, T, device):
    """Same files and checks as the registered test reader, without its whitelist of durations."""
    import numpy as np
    import torch
    from po_jepa.runtime import cache
    raw, meta, manifest_sha = cache._read_complete_cache(Path(features)/f"{trajectory_id:04d}")
    if meta["split"] != "test" or manifest_sha != approved["manifest_sha256_by_id"][trajectory_id]:
        raise ValueError(f"trajectory {trajectory_id}: not an approved test cache")
    end = int(start+T)
    if end != start+T or end >= raw.shape[0]:
        return None
    z = torch.from_numpy(np.array(raw[start:start+1], dtype=np.float32, copy=True)).to(device)
    y = torch.from_numpy(np.array(raw[end:end+1], dtype=np.float32, copy=True)).to(device)
    return z, y, cache._window_controls(meta, start, float(T), str(device))


def run_one(job_dir, features, output, device_name, max_test=None):
    import torch
    from po_jepa import datasets
    from po_jepa.runtime import cache, comparison, data
    job = json.loads((job_dir/"job.json").read_text(encoding="utf-8"))
    device = torch.device(device_name)
    condition = job["condition"]
    model, loss, checkpoint = load_selected(condition, job_dir, device)
    approved = data.approved_cache_metadata(Path(features), splits=("test",))
    grid = cache.regular_grid(16, 16)

    def distance(pred, target):
        return float(comparison.jepa_distance(pred, target, config=loss).per_sample[0])

    spec = datasets.active()
    rows, t0 = [], time.time()
    with torch.no_grad():
        test_ids = list(spec.split_range("test"))
        for tid in (test_ids if max_test is None else test_ids[:max_test]):
            for start in datasets.test_start_ticks(tid):
                for T in DURATIONS:
                    window = read_test_window(features, approved, tid, start, T, device)
                    if window is None:
                        continue
                    z, y, controls = window
                    values = window_metrics(model, z, y, controls, T, grid, distance, comparison.split_action_segments)
                    rows.append(dict(trajectory_id=tid, start_tick=start, duration_cycles=T, **values))
    result = dict(condition=condition, dataset=job["dataset"], seed=job["seed"], updates=job["updates"],
                  job=str(job_dir), features=str(features), checkpoint=checkpoint, device=device_name,
                  max_test_trajectories=max_test,
                  seconds=round(time.time()-t0, 1), registered_durations=list(REGISTERED), rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"{job['dataset']} seed{job['seed']} {condition}: {len(rows)} windows, {result['seconds']} s -> {output}", flush=True)


# ---------------------------------------------------------------- batch and summary

def run_all(args):
    jobs = find_jobs(Path(args.results_root).resolve())
    if not jobs:
        raise SystemExit(f"no finished jobs under {args.results_root}")
    out_root = Path(args.output).resolve()
    failures = 0
    for job_dir, job in jobs:
        target = out_root/job["dataset"]/f"seed{job['seed']}"/f"{job['condition']}.json"
        if target.exists():
            print(f"skip (exists): {target}")
            continue
        features = features_for(job_dir, job, args.features_base)
        env = dict(os.environ, PO_JEPA_PROTOCOL="v7", PO_JEPA_DATASET=dataset_env(job["dataset"]),
                   PO_JEPA_UPDATES=str(job["updates"]), PO_JEPA_SEED=str(job["seed"]),
                   PYTHONPATH=os.pathsep.join([str(ROOT/"src"), *filter(None, [os.environ.get("PYTHONPATH")])]))
        cmd = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "one", "--job", str(job_dir),
               "--features-root", str(features), "--output-file", str(target), "--device", args.device]
        if args.max_test_trajectories:
            cmd += ["--max-test-trajectories", str(args.max_test_trajectories)]
        code = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        if code:
            failures += 1
            print(f"FAILED ({code}): {job_dir}", flush=True)
    print(f"done; {failures} failed")
    return 1 if failures else 0


def summarize(args):
    """Per dataset/seed/duration: mean of each metric per condition, and full vs no-consistency
    (relative difference of means, and how many test trajectories full wins on)."""
    out_root = Path(args.output).resolve()
    files = sorted(out_root.glob("*/seed*/*.json"))
    table = {}
    for path in files:
        if path.name == "summary.json":
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (result["dataset"], result["seed"])
        table.setdefault(key, {})[result["condition"]] = result["rows"]
    summary, lines = [], []
    for (dataset, seed), by_condition in sorted(table.items()):
        durations = sorted({r["duration_cycles"] for rows in by_condition.values() for r in rows})
        for T in durations:
            metrics = [m for m in ("id", "direct", "half", "unit", "p2_4", "p4_2", "dc_half")
                       if any(m in r for rows in by_condition.values() for r in rows if r["duration_cycles"] == T)]
            entry = dict(dataset=dataset, seed=seed, duration_cycles=T, registered=T in REGISTERED, means={}, full_vs_nocons={})
            for condition, rows in by_condition.items():
                sel = [r for r in rows if r["duration_cycles"] == T]
                entry["means"][condition] = {m: statistics.fmean(r[m] for r in sel) for m in metrics if all(m in r for r in sel)}
                entry["means"][condition]["windows"] = len(sel)
            full, nocons = by_condition.get("po_jepa_full"), by_condition.get("po_jepa_no_consistency")
            if full and nocons:
                f = {(r["trajectory_id"], r["start_tick"]): r for r in full if r["duration_cycles"] == T}
                n = {(r["trajectory_id"], r["start_tick"]): r for r in nocons if r["duration_cycles"] == T}
                keys = sorted(f.keys() & n.keys())
                for m in metrics:
                    if m in ("id",) or not keys or not all(m in f[k] and m in n[k] for k in keys):
                        continue
                    mf, mn = statistics.fmean(f[k][m] for k in keys), statistics.fmean(n[k][m] for k in keys)
                    tids = sorted({k[0] for k in keys})
                    wins = sum(statistics.fmean(f[k][m]-n[k][m] for k in keys if k[0] == t) < 0 for t in tids)
                    entry["full_vs_nocons"][m] = dict(relative=(mf-mn)/mn if mn else None, full_better_trajectories=wins,
                                                      trajectories=len(tids))
            summary.append(entry)
            cells = "  ".join(f"{m}:{v['relative']*100:+.2f}%({v['full_better_trajectories']}/{v['trajectories']})"
                              for m, v in entry["full_vs_nocons"].items() if v["relative"] is not None)
            lines.append(f"{dataset:32s} seed{seed} T={T:<4g}{'' if T in REGISTERED else ' (extra)':8s} full-vs-nocons  {cells}")
    (out_root/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten {out_root/'summary.json'}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="evaluate every finished job under --results-root")
    r.add_argument("--results-root", required=True)
    r.add_argument("--features-base", help="fallback when a job has no recorded --features-root: directory holding "
                   "<dataset tag>/ feature caches, e.g. ../features-v7")
    r.add_argument("--output", required=True)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--max-test-trajectories", type=int, help="only the first N test trajectories (for the 2000-trajectory runs)")
    o = sub.add_parser("one", help="(internal) one job; the dataset is set through the environment by 'run'")
    o.add_argument("--job", required=True)
    o.add_argument("--features-root", required=True)
    o.add_argument("--output-file", required=True)
    o.add_argument("--device", default="cuda:0")
    o.add_argument("--max-test-trajectories", type=int)
    s = sub.add_parser("summarize", help="tables from the per-job files under --output")
    s.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_all(args)
    if args.command == "one":
        run_one(Path(args.job).resolve(), Path(args.features_root).resolve(), Path(args.output_file).resolve(), args.device,
                args.max_test_trajectories)
        return 0
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
