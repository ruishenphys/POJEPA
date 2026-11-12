"""Extra evaluation of already-trained v7 runs (no training, analysis only).

For every finished job of po_jepa_full / po_jepa_no_consistency / matched_transformer / identity, load the checkpoint
the job's registered evaluation used and measure, on the 16x16 test windows, the JEPA distance to the true future frame
of several ways of reaching the same horizon T:

  id      : the current tokens Z_t (identity reference)
  direct  : Phi_T(Z_t) with the controls cut at T/2 and concatenated again, exactly as the registered evaluation
            (runtime/evaluation.point_metrics) feeds them; equals its future_jepa_distance
  half    : Phi_{T/2}(Phi_{T/2}(Z_t))                     (the composed path of the temporal consistency loss)
  unit    : T steps of 1 cycle, each from the previous prediction (runtime/evaluation.unit_rollout, the v7.1 rollout)
  p2_4 / p4_2 : 6 = 2+4 and 6 = 4+2                       (unequal splits, only for T = 6)
  dc_half : JEPA distance between direct and half         (temporal defect in the prediction metric)

Durations 1, 2, 3, 4, 6, 8, 12, 16 cycles; 6 and 16 are outside the registered evaluation (never trained, never
evaluated) and are reported separately. Windows whose end frame does not exist are skipped. Nothing here feeds back
into training, checkpoint selection or the registered tables.

Identity (round-3 review, 2026-11-11): the checkpoint must be the one the job's completion flag, top-level report and
evaluation registration all name (training_run / resume_from resolved as runtime/v7jobs.run_job does); the feature
cache must have the same identity as the job's preflight receipt (the cache its registered evaluation read); every
output file carries an identity block and is reused only when that block matches, and is written atomically.

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
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity")
DURATIONS = (1., 2., 3., 4., 6., 8., 12., 16.)
REGISTERED = (1., 2., 3., 4., 8., 12.)
EXTRA_SPLITS = {6.: {"p2_4": (2., 4.), "p4_2": (4., 2.)}}
METRICS = ("id", "direct", "half", "unit", "p2_4", "p4_2", "dc_half")
# Bump whenever a number in the output changes meaning. 1 = a305bfe (Transformer direct without the T/2 cut, no
# output identity); its files are refused by 'run' and 'summarize'.
METRIC_VERSION = 2


def dataset_env(tag: str) -> str:
    """job.json stores the dataset tag '<scene>-<count>'; run.py's environment variable is '<scene>:<count>'."""
    scene, sep, count = tag.rpartition("-")
    if not sep or not count.isdigit():
        raise ValueError(f"unexpected dataset tag {tag!r}")
    return f"{scene}:{count}"


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _normal(value):
    """JSON round trip, so in-memory dicts (int keys) compare equal to what the job wrote to disk."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


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


def duplicate_keys(jobs):
    """(dataset, seed, condition) claimed by more than one finished job: their outputs would share one file name."""
    counts = Counter((job["dataset"], job["seed"], job["condition"]) for _, job in jobs)
    return sorted(key for key, n in counts.items() if n > 1)


def features_for(job_dir: Path, job: dict, features_base):
    """The feature cache the job was trained on: --features-root recorded in <output>/console/command.json
    (job_dir is <output>/run/job); falls back to <features-base>/<dataset tag>. Which root is picked does not decide
    identity: run_one checks it against the job's preflight receipt."""
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


def window_metrics(model, raw, target, controls, T, grid, distance):
    """All distances for one window; distance(pred, target) -> float. direct/half follow evaluation.point_metrics
    line by line (controls cut at T/2; direct sees the two halves concatenated) and unit is evaluation.unit_rollout."""
    import torch
    from po_jepa.runtime import comparison, evaluation
    split = comparison.split_action_segments
    duration = torch.tensor([float(T)], dtype=torch.float64)
    tau = duration*.5
    pair = split(controls, tau=tau, sigma=tau)
    direct = model(raw, comparison.concatenate_action_segments(pair.first, pair.second), duration, grid)
    half = model(model(raw, pair.first, tau, grid), pair.second, tau, grid)
    out = dict(id=distance(raw, target), direct=distance(direct, target), half=distance(half, target),
               dc_half=distance(direct, half))
    if float(T).is_integer():
        out["unit"] = distance(evaluation.unit_rollout(model, raw, controls, duration, grid), target)
    for name, parts in EXTRA_SPLITS.get(float(T), {}).items():
        out[name] = distance(partition_rollout(model, raw, controls, parts, grid, split), target)
    return out


# ---------------------------------------------------------------- identity of the registered evaluation

def formal_selection(job_dir: Path, condition: str):
    """What the job's registered evaluation used: completion flag -> top-level report -> evaluation registration ->
    training run (resolved like v7jobs.run_job: local training/<condition>, else training_run / resume_from) whose own
    completion flag and report name the same selected checkpoint. Returns the identity pieces; reads no weights."""
    from po_jepa.runtime import v7jobs
    job = _read(job_dir/"job.json")
    if job.get("condition") != condition:
        raise ValueError(f"{job_dir}: job.json is for {job.get('condition')!r}, not {condition!r}")
    complete = _read(job_dir/"COMPLETE.json")
    report_sha = _sha(job_dir/"report.json")
    if complete.get("report_sha256") != report_sha:
        raise ValueError(f"{job_dir}: report.json is not the one COMPLETE.json was written for")
    report = _read(job_dir/"report.json")
    if report.get("job") != job:
        raise ValueError(f"{job_dir}: report.json describes another job than job.json")
    evaluation_dir = job_dir/"evaluation"/condition
    if not (evaluation_dir/"COMPLETE.json").exists() or (evaluation_dir/"FAILED.json").exists():
        raise ValueError(f"{evaluation_dir}: registered evaluation not complete")
    registration = _read(evaluation_dir/"registration.json")
    if registration.get("condition") != condition or registration.get("checkpoint") != report.get("best"):
        raise ValueError(f"{job_dir}: evaluation registration and report name different checkpoints")
    receipt = (report.get("preflight") or {}).get("cache_identity")
    if not receipt or not receipt.get("manifest_sha256_by_id"):
        raise ValueError(f"{job_dir}: report has no preflight cache identity")
    out = dict(job=job, job_report_sha256=report_sha, cache_identity=receipt,
               evaluation_registration_sha256=_sha(evaluation_dir/"registration.json"), best=None, root=None)
    if condition == "identity":
        if report.get("best") is not None:
            raise ValueError(f"{job_dir}: identity job names a checkpoint")
        return out
    best = report.get("best")
    if not best or not best.get("checkpoint") or not best.get("checkpoint_sha256"):
        raise ValueError(f"{job_dir}: report names no selected checkpoint")
    local = job_dir/"training"/condition
    if local.exists():
        root = local
    else:
        old = v7jobs._job_dir(job.get("training_run") or job.get("resume_from"))
        if old is None:
            raise ValueError(f"{job_dir}: no training/{condition} and no training_run/resume_from")
        root = old/"training"/condition
    if not (root/"COMPLETE.json").exists() or (root/"FAILED.json").exists():
        raise ValueError(f"{root}: training not complete")
    if _read(root/"COMPLETE.json").get("report_sha256") != _sha(root/"report.json"):
        raise ValueError(f"{root}: training report.json is not the one its COMPLETE.json was written for")
    if _read(root/"report.json").get("best") != best:
        raise ValueError(f"{root}: training report selects another checkpoint than the registered evaluation")
    out.update(best=best, root=root)
    return out


def verify_registered_run(job_dir: Path, selection: dict):
    """R4-2-P1-1: facts that a consistent rewrite of job.json / report.json / registration.json (with rebound
    COMPLETE flags) cannot fake. Needs the job's protocol, dataset and seed active (run_one checks the environment).
    - every record of the registered evaluation (trajectories/*.json) names the selected checkpoint and the job's seed;
    - the training's selection is recomputed from its raw validation values (workflow.verify_selection: the same
      400 windows, the fixed future-only rule), so `best` cannot be re-pointed to another stored checkpoint;
    - the seed-dependent query sequence (window seed shifted with the model seed) recomputed for the job's seed equals
      the one in the training preregistration and in the selected checkpoint's provenance (load_selected compares
      that provenance with the checkpoint bytes), so a run trained with another seed cannot be relabelled.
    Rewriting the checkpoint, the raw validation values and every evaluation row together is out of scope."""
    from po_jepa.runtime import checks, comparison, learning, workflow
    job, best = selection["job"], selection["best"]
    if type(job.get("seed")) is not int:
        raise ValueError(f"{job_dir}: job seed {job.get('seed')!r} is not an integer")
    registered = _read(job_dir/"evaluation"/job["condition"]/"registration.json")["checkpoint"]
    # evaluation.run_prepared_evaluation writes every row to rows/ and again, grouped, to trajectories/<id>.json at
    # the end; the trajectory files are what result packs keep, so they are read here
    files = sorted((job_dir/"evaluation"/job["condition"]/"trajectories").glob("*.json"))
    records = [(path, row) for path in files for row in _read(path).get("records") or []]
    if not files or not records:
        raise ValueError(f"{job_dir}: the registered evaluation has no trajectory records")
    for path, row in records:
        if row.get("checkpoint") != registered or type(row.get("seed")) is not int or row["seed"] != job["seed"]:
            raise ValueError(f"{path}: registered evaluation row is for checkpoint step "
                             f"{(row.get('checkpoint') or {}).get('step')!r} / seed {row.get('seed')!r}, not the "
                             f"selected step {(best or {}).get('step')!r} / seed {job['seed']}")
    if best is None:
        return
    root = selection["root"]
    train_report = _read(root/"report.json")
    try:
        workflow.verify_selection(root, train_report)
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(f"{root}: selection does not follow from the raw validation values ({error})") from None
    sequence, _ = learning.make_query_sequence(checks.load_protocol())
    expected = comparison.sequence_sha(sequence)
    preregistration = _read(root/"registration.json").get("preregistration") or {}
    found = dict(preregistration=(preregistration.get("model_seed"), preregistration.get("query_sequence_sha256")),
                 checkpoint=(job["seed"], (best.get("checkpoint_provenance") or {}).get("f04_query_sequence_sha256")))
    if any(v != (job["seed"], expected) for v in found.values()):
        raise ValueError(f"{root}: training was not run with seed {job['seed']} (query sequence {expected[:12]}…; "
                         f"found {found})")


def load_selected(condition, job_dir, device, selection=None):
    """Model of `condition` with the checkpoint the registered evaluation used: identity chain (formal_selection), file
    SHA, step, training provenance, condition/architecture and strict state. Optimizer/RNG/runtime are not restored
    (evaluation only), so a different GPU index or driver does not matter and the global RNG is left alone."""
    import torch
    from dataclasses import asdict
    from po_jepa.runtime import comparison, training
    selection = formal_selection(job_dir, condition) if selection is None else selection
    model, loss = comparison.build_condition(condition, device)
    if condition == "identity":
        return model.eval(), loss, None
    selected, root = selection["best"], selection["root"]
    path = (root/selected["checkpoint"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("selected checkpoint escapes its run")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != selected["checkpoint_sha256"]:
        raise ValueError(f"{path}: checkpoint SHA mismatch")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if payload["completed_steps"] != selected["step"]:
        raise ValueError("checkpoint step differs from the selected validation step")
    if not training._same_native(payload["provenance"], selected.get("checkpoint_provenance")):
        raise ValueError("checkpoint training provenance differs from the selected validation record")
    if condition == "matched_transformer":
        if payload.get("schema") != "F05_transformer_checkpoint_v1" or payload["config"] != asdict(model.spec):
            raise ValueError("Transformer architecture differs from the checkpoint")
    else:
        expected = "full" if condition == "po_jepa_full" else "no_consistency"
        if payload["condition"] != expected:
            raise ValueError("checkpoint condition mismatch")
        if not training._same_native(payload["model_config"], training._model_config(model)):
            raise ValueError("PO-JEPA model configuration differs from the checkpoint")
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval(), loss, dict(step=selected["step"], sha256=selected["checkpoint_sha256"],
                                    path=str(path.relative_to(root.resolve())), training_root=str(root),
                                    validation_mean=selected["mean"])


def features_identity(features, receipt):
    """The cache at `features` must be the one the job's preflight receipt (and so its registered evaluation) saw."""
    from po_jepa.runtime import data
    current = _normal(data.approved_cache_metadata(Path(features), splits=("test",)))
    if current != _normal(receipt):
        raise ValueError(f"{features}: feature cache differs from the one the job's registered evaluation used")
    return dict(encode_report_sha256=current.get("encode_report_sha256"),
                manifests_sha256=hashlib.sha256(json.dumps(current["manifest_sha256_by_id"], sort_keys=True).encode())
                .hexdigest())


def output_identity(selection, features_digest, max_test):
    job = selection["job"]
    best = selection["best"]
    return dict(metric_version=METRIC_VERSION, dataset=job["dataset"], seed=job["seed"], condition=job["condition"],
                updates=job["updates"], job_report_sha256=selection["job_report_sha256"],
                evaluation_registration_sha256=selection["evaluation_registration_sha256"],
                checkpoint=None if best is None else dict(step=best["step"], sha256=best["checkpoint_sha256"]),
                features=features_digest, max_test_trajectories=max_test, durations=list(DURATIONS))


def reusable(output: Path, identity: dict) -> bool:
    """True when `output` already holds a result with exactly this identity; a parseable file with another identity is
    refused (never silently reused or overwritten); an unreadable leftover is recomputed."""
    if not output.exists():
        return False
    try:
        old = _read(output)
    except (ValueError, UnicodeDecodeError):
        print(f"recomputing unreadable {output}", flush=True)
        return False
    if old.get("identity") == _normal(identity):
        return True
    raise ValueError(f"{output} holds a result of another run/checkpoint/cache/budget/version; "
                     "move it away or use a new --output")


def _write_atomic(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.partial-{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


# ---------------------------------------------------------------- one job (dataset set through the environment)

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
    from po_jepa.runtime import cache, checks, comparison, data
    if max_test is not None and (type(max_test) is not int or max_test < 1):
        raise ValueError("max_test must be a positive integer or None")
    job = _read(job_dir/"job.json")
    condition = job["condition"]
    if (datasets.active().tag, checks.model_seed(), checks.v7_updates()) != (job["dataset"], job["seed"], job["updates"]):
        raise ValueError(f"{job_dir}: dataset/seed/updates in the environment differ from job.json")
    selection = formal_selection(job_dir, condition)
    verify_registered_run(job_dir, selection)
    digest = features_identity(features, selection["cache_identity"])
    identity = output_identity(selection, digest, max_test)
    if reusable(output, identity):
        print(f"skip (same identity): {output}", flush=True)
        return
    device = torch.device(device_name)
    model, loss, checkpoint = load_selected(condition, job_dir, device, selection)
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
                    values = window_metrics(model, z, y, controls, T, grid, distance)
                    rows.append(dict(trajectory_id=tid, start_tick=start, duration_cycles=T, **values))
    result = dict(condition=condition, dataset=job["dataset"], seed=job["seed"], updates=job["updates"],
                  identity=identity, job=str(job_dir), features=str(features), checkpoint=checkpoint, device=device_name,
                  max_test_trajectories=max_test,
                  seconds=round(time.time()-t0, 1), registered_durations=list(REGISTERED), rows=rows)
    _write_atomic(output, _normal(result))
    print(f"{job['dataset']} seed{job['seed']} {condition}: {len(rows)} windows, {result['seconds']} s -> {output}", flush=True)


# ---------------------------------------------------------------- batch and summary

def run_all(args):
    jobs = find_jobs(Path(args.results_root).resolve())
    if not jobs:
        raise SystemExit(f"no finished jobs under {args.results_root}")
    duplicates = duplicate_keys(jobs)
    if duplicates:
        listing = "\n".join(f"  {d} seed{s} {c}: " + ", ".join(str(j) for j, job in jobs
                            if (job["dataset"], job["seed"], job["condition"]) == (d, s, c)) for d, s, c in duplicates)
        raise SystemExit(f"several finished jobs for the same dataset/seed/condition; narrow --results-root:\n{listing}")
    out_root = Path(args.output).resolve()
    failures = 0
    for job_dir, job in jobs:
        target = out_root/job["dataset"]/f"seed{job['seed']}"/f"{job['condition']}.json"
        features = features_for(job_dir, job, args.features_base)
        env = dict(os.environ, PO_JEPA_PROTOCOL="v7", PO_JEPA_DATASET=dataset_env(job["dataset"]),
                   PO_JEPA_UPDATES=str(job["updates"]), PO_JEPA_SEED=str(job["seed"]),
                   PYTHONPATH=os.pathsep.join([str(ROOT/"src"), *filter(None, [os.environ.get("PYTHONPATH")])]))
        cmd = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "one", "--job", str(job_dir),
               "--features-root", str(features), "--output-file", str(target), "--device", args.device]
        if args.max_test_trajectories is not None:
            cmd += ["--max-test-trajectories", str(args.max_test_trajectories)]
        code = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        if code:
            failures += 1
            print(f"FAILED ({code}): {job_dir}", flush=True)
    print(f"done; {failures} failed")
    return 1 if failures else 0


def _load_results(out_root: Path):
    """Per (dataset, seed): {condition: result}. Refuses files without the current identity block, files whose
    content does not match their slot, and groups whose conditions saw different caches, budgets or test subsets."""
    table = {}
    for path in sorted(out_root.glob("*/seed*/*.json")):
        result = _read(path)
        identity = result.get("identity") or {}
        if identity.get("metric_version") != METRIC_VERSION:
            raise SystemExit(f"{path}: written by an older extra_eval_v7 (metric version "
                             f"{identity.get('metric_version', 1)}); rerun into a new --output")
        slot = (path.parent.parent.name, path.parent.name, path.stem)
        if (slot != (result["dataset"], f"seed{result['seed']}", result["condition"])
                or (identity["dataset"], identity["seed"], identity["condition"])
                != (result["dataset"], result["seed"], result["condition"])):
            raise SystemExit(f"{path}: content belongs to another dataset/seed/condition")
        table.setdefault((result["dataset"], result["seed"]), {})[result["condition"]] = result
    for (dataset, seed), by_condition in table.items():
        shared = {json.dumps([r["identity"][k] for k in ("updates", "features", "max_test_trajectories", "durations")],
                             sort_keys=True) for r in by_condition.values()}
        if len(shared) > 1:
            raise SystemExit(f"{dataset} seed{seed}: conditions were evaluated on different caches/budgets/test subsets")
    return table


def summarize(args):
    """Per dataset/seed/duration: mean of each metric per condition over its own windows (None and windows=0 when a
    condition has none), and full vs no-consistency on the paired windows only, with paired and unpaired counts."""
    out_root = Path(args.output).resolve()
    table = _load_results(out_root)
    summary, lines = [], []
    for (dataset, seed), by_condition in sorted(table.items()):
        rows_of = {c: r["rows"] for c, r in by_condition.items()}
        durations = sorted({r["duration_cycles"] for rows in rows_of.values() for r in rows})
        for T in durations:
            metrics = [m for m in METRICS
                       if any(m in r for rows in rows_of.values() for r in rows if r["duration_cycles"] == T)]
            entry = dict(dataset=dataset, seed=seed, duration_cycles=T, registered=T in REGISTERED, means={},
                         full_vs_nocons={}, pairing=None)
            for condition in CONDITIONS:
                if condition not in rows_of:
                    continue
                sel = [r for r in rows_of[condition] if r["duration_cycles"] == T]
                means = {m: (statistics.fmean(r[m] for r in sel) if sel and all(m in r for r in sel) else None)
                         for m in metrics}
                entry["means"][condition] = dict(means, windows=len(sel))
            full, nocons = rows_of.get("po_jepa_full"), rows_of.get("po_jepa_no_consistency")
            if full is not None and nocons is not None:
                f = {(r["trajectory_id"], r["start_tick"]): r for r in full if r["duration_cycles"] == T}
                n = {(r["trajectory_id"], r["start_tick"]): r for r in nocons if r["duration_cycles"] == T}
                keys = sorted(f.keys() & n.keys())
                entry["pairing"] = dict(paired_windows=len(keys), full_only=len(f.keys()-n.keys()),
                                        nocons_only=len(n.keys()-f.keys()))
                for m in metrics:
                    if m == "id" or not keys or not all(m in f[k] and m in n[k] for k in keys):
                        continue
                    mf, mn = statistics.fmean(f[k][m] for k in keys), statistics.fmean(n[k][m] for k in keys)
                    tids = sorted({k[0] for k in keys})
                    wins = sum(statistics.fmean(f[k][m]-n[k][m] for k in keys if k[0] == t) < 0 for t in tids)
                    entry["full_vs_nocons"][m] = dict(relative=(mf-mn)/mn if mn else None, full_mean=mf, nocons_mean=mn,
                                                      full_better_trajectories=wins, trajectories=len(tids))
            summary.append(entry)
            cells = "  ".join(f"{m}:{v['relative']*100:+.2f}%({v['full_better_trajectories']}/{v['trajectories']})"
                              for m, v in entry["full_vs_nocons"].items() if v["relative"] is not None)
            pairing = entry["pairing"]
            note = "" if not pairing or not (pairing["full_only"] or pairing["nocons_only"]) else \
                f"  [unpaired: full {pairing['full_only']}, nocons {pairing['nocons_only']}]"
            lines.append(f"{dataset:32s} seed{seed} T={T:<4g}{'' if T in REGISTERED else ' (extra)':8s} "
                         f"full-vs-nocons  {cells}{note}")
    _write_atomic(out_root/"summary.json", summary)
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
    r.add_argument("--max-test-trajectories", type=positive_int,
                   help="only the first N test trajectories (for the 2000-trajectory runs)")
    o = sub.add_parser("one", help="(internal) one job; the dataset is set through the environment by 'run'")
    o.add_argument("--job", required=True)
    o.add_argument("--features-root", required=True)
    o.add_argument("--output-file", required=True)
    o.add_argument("--device", default="cuda:0")
    o.add_argument("--max-test-trajectories", type=positive_int)
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
