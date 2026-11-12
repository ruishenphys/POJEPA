"""tools/extra_eval_v7.py: partitions, metrics against the registered point_metrics, identity of the registered
evaluation (checkpoint, cache, output), one job end to end, batch guards and summary (CPU, fictional data).
Self-contained: the v7 calibration file is written to tmp_path, nothing outside the repository is needed."""
import hashlib
import importlib.util
import io
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("extra_eval_v7", ROOT/"tools/extra_eval_v7.py")
xe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xe)

from po_jepa import datasets  # noqa: E402
from po_jepa.runtime import cache, checks, comparison, data, evaluation, training  # noqa: E402
from test_training import e06_config  # noqa: E402

META = dict(trajectory_id=180, controls=dict(starts_ticks=[0, 8, 16, 24], ends_ticks=[8, 16, 24, 32],
            values=[[.1, 0.], [.2, 0.], [.3, 0.], [.4, 0.]], units="simulation_position_per_second"))
PROV = dict(evidence_role="fictional", split="train", feature_manifest_sha256="f"*64)
RECEIPT = dict(encode_report_sha256="enc-A", manifest_sha256_by_id={180: "m180", 181: "m181"})


class Scale(torch.nn.Module):
    """Fictional predictor: output = input * (1 + duration/10); records (duration, actions seen)."""
    def __init__(self):
        super().__init__(); self.w = torch.nn.Parameter(torch.ones(())); self.calls = []
    def forward(self, raw, controls, duration, grid):
        self.calls.append((float(duration[0]), controls.u_values[0, controls.segment_valid[0], 0].tolist()))
        return raw*(1+duration.to(raw.dtype).reshape(-1, 1, 1)/10)*self.w


def dist(pred, target):
    return float(comparison.jepa_distance(pred, target, config=e06_config()).per_sample[0])


def test_dataset_env():
    assert xe.dataset_env("van_der_pol-200") == "van_der_pol:200"
    assert xe.dataset_env("landscape~shape=multi_well-2000") == "landscape~shape=multi_well:2000"
    with pytest.raises(ValueError):
        xe.dataset_env("van_der_pol")


def test_partition_rollout_pieces_see_their_own_actions():
    controls = cache._window_controls(META, 6, 6., "cpu")
    raw, model = torch.ones(1, 256, 3), Scale()
    out = xe.partition_rollout(model, raw, controls, (2., 4.), cache.regular_grid(16, 16), comparison.split_action_segments)
    assert [c[0] for c in model.calls] == [2., 4.]
    assert model.calls[0][1] == pytest.approx([.1])            # ticks 6-8
    assert model.calls[1][1] == pytest.approx([.2])            # ticks 8-12
    assert torch.allclose(out, raw*1.2*1.4)
    model.calls.clear()
    xe.partition_rollout(model, raw, controls, (4., 2.), cache.regular_grid(16, 16), comparison.split_action_segments)
    assert model.calls[0][1] == pytest.approx([.1, .2]) and model.calls[1][1] == pytest.approx([.2])
    model.calls.clear()
    out = xe.partition_rollout(model, raw, controls, (1., 2., 3.), cache.regular_grid(16, 16), comparison.split_action_segments)
    assert [c[0] for c in model.calls] == [1., 2., 3.]
    assert [c[1] for c in model.calls] == [pytest.approx([.1]), pytest.approx([.1, .2]), pytest.approx([.2])]  # 6-7, 7-9, 9-12
    assert torch.allclose(out, raw*1.1*1.2*1.3)
    with pytest.raises(ValueError):
        xe.partition_rollout(model, raw, controls, (0., 6.), cache.regular_grid(16, 16), comparison.split_action_segments)


def test_read_test_window_frames_and_checks(monkeypatch):
    frames = torch.arange(20, dtype=torch.float32).reshape(20, 1, 1).expand(20, 256, 4).numpy()
    meta = dict(META, split="test")
    monkeypatch.setattr(cache, "_read_complete_cache", lambda path: (frames, meta, "sha-180"))
    approved = dict(manifest_sha256_by_id={180: "sha-180"})
    z, y, controls = xe.read_test_window(Path("."), approved, 180, 5, 6., "cpu")
    assert z.shape == (1, 256, 4) and float(z[0, 0, 0]) == 5. and float(y[0, 0, 0]) == 11.
    assert controls.u_values[0, controls.segment_valid[0], 0].tolist() == pytest.approx([.1, .2])
    assert xe.read_test_window(Path("."), approved, 180, 5, 16., "cpu") is None      # frame 21 does not exist
    assert xe.read_test_window(Path("."), approved, 180, 5, 15., "cpu") is None      # frame 20 does not exist either
    assert xe.read_test_window(Path("."), approved, 180, 5, 14., "cpu") is not None  # frame 19 is the last one
    with pytest.raises(ValueError):
        xe.read_test_window(Path("."), dict(manifest_sha256_by_id={180: "other"}), 180, 5, 6., "cpu")
    monkeypatch.setattr(cache, "_read_complete_cache", lambda path: (frames, dict(meta, split="train"), "sha-180"))
    with pytest.raises(ValueError):
        xe.read_test_window(Path("."), approved, 180, 5, 6., "cpu")


class ScaleShift(torch.nn.Module):
    """Fictional predictor that LayerNorm does not cancel: x*(1+d/10) + d*c with a fixed per-channel c."""
    def __init__(self, channels):
        super().__init__(); self.c = torch.linspace(-1, 1, channels)
    def forward(self, raw, controls, duration, grid):
        d = float(duration[0])
        return raw*(1+d/10)+d*self.c


def apply(x, parts, c):
    for d in parts:
        x = x*(1+d/10)+d*c
    return x


def test_window_metrics_values():
    torch.manual_seed(0)
    raw, target = torch.randn(1, 256, 8), torch.randn(1, 256, 8)
    model = ScaleShift(8)
    controls = cache._window_controls(META, 2, 6., "cpu")
    got = xe.window_metrics(model, raw, target, controls, 6., cache.regular_grid(16, 16), dist)
    assert set(got) == {"id", "direct", "half", "dc_half", "unit", "p2_4", "p4_2"}
    c = model.c
    assert got["id"] == pytest.approx(dist(raw, target))
    assert got["direct"] == pytest.approx(dist(apply(raw, (6.,), c), target))
    assert got["half"] == pytest.approx(dist(apply(raw, (3., 3.), c), target))
    assert got["unit"] == pytest.approx(dist(apply(raw, (1.,)*6, c), target))
    assert got["p2_4"] == pytest.approx(dist(apply(raw, (2., 4.), c), target))
    assert got["p4_2"] == pytest.approx(dist(apply(raw, (4., 2.), c), target))
    assert got["dc_half"] == pytest.approx(dist(apply(raw, (6.,), c), apply(raw, (3., 3.), c)))
    assert got["dc_half"] > 1e-3 and got["unit"] != pytest.approx(got["direct"])
    odd = xe.window_metrics(model, raw, target, cache._window_controls(META, 0, 3., "cpu"), 3., cache.regular_grid(16, 16), dist)
    assert odd["half"] == pytest.approx(dist(apply(raw, (1.5, 1.5), c), target)) and "p2_4" not in odd
    assert odd["unit"] == pytest.approx(dist(apply(raw, (1., 1., 1.), c), target))


def _window(raw, target, tid, start, T, action=None):
    """CacheWindow as the registered reader returns it (fictional source)."""
    meta = dict(META, trajectory_id=tid)
    if action is not None:
        meta = dict(meta, controls=dict(META["controls"], values=[[action, 0.]]*4))
    x = cache.WindowInput(raw, cache._window_controls(meta, start, T, "cpu"), torch.tensor([T], dtype=torch.float64),
                          cache.regular_grid(16, 16), torch.tensor([start*.25], dtype=torch.float64), "input", tid, "test")
    y = cache.WindowTarget(target, torch.tensor([(start+T)*.25], dtype=torch.float64), "target")
    return cache.CacheWindow(x, y, dict(actual_manifest_sha256=f"m{tid}", generation=dict(evidence_role="real_frozen_encoder"),
                                        observation_variant="canonical", fictional=True))


def _perturb(model, seed=3, scale=.1):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p)*scale)
    return model.eval()


def _registered(model, raw, target, T, config, start=5):
    """future_jepa_distance and rollout_future_jepa_distance of the registered evaluation for one 16x16 point."""
    w, donor = _window(raw, target, 180, start, T), _window(raw, target, 181, start, T, action=-.2)
    values, _ = evaluation.point_metrics(model, w, donor, evaluation.EvaluationQuery(180, start, T, 16, "point"), config)
    return values, w


@pytest.mark.parametrize("kind", ["transformer", "po"])
def test_window_metrics_match_registered_point_metrics(monkeypatch, kind):
    """V7-P1-1: direct and unit equal the registered evaluation's numbers, including a T/2 cut inside an action segment
    (start 5: segment 0-8 is cut at 5.5, 6.5 and 9), where the uncut controls give the Transformer another output."""
    from po_jepa.baselines import TokenTransformer, TransformerSpec
    from test_training import e06_case
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    model = _perturb(TokenTransformer(TransformerSpec(encoder_dim=3, residual=True)) if kind == "transformer" else e06_case()[0])
    torch.manual_seed(1)
    raw, target = torch.randn(1, 256, 3), torch.randn(1, 256, 3)
    config = e06_config()
    for T in (1., 3., 8.):
        values, w = _registered(model, raw, target, T, config)
        with torch.no_grad():
            got = xe.window_metrics(model, raw, target, w.inputs.controls, T, cache.regular_grid(16, 16), dist)
        assert got["direct"] == pytest.approx(values["future_jepa_distance"], rel=1e-6, abs=0)
        assert got["unit"] == pytest.approx(values["rollout_future_jepa_distance"], rel=1e-6, abs=0)
        if kind == "transformer":
            with torch.no_grad():
                uncut = dist(model(raw, w.inputs.controls, w.inputs.duration, cache.regular_grid(16, 16)), target)
            assert uncut != pytest.approx(got["direct"], rel=1e-6, abs=0)   # the a305bfe input differs here


# ---------------------------------------------------------------- formal chain fixtures

@pytest.fixture
def v7_double_well(monkeypatch, tmp_path):
    """v7 protocol on double_well-200 with a calibration file written here (no file outside the repository)."""
    calibration = tmp_path/"calibration.yaml"
    calibration.write_text("sigma_z: 1.0\ntime_weight: 0.25\nspace_weight: 0.5\n")
    monkeypatch.setattr(checks, "v7_calibration_path", lambda: calibration)
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    monkeypatch.setenv(checks.UPDATES_ENV, "2000")
    monkeypatch.delenv(checks.SEED_ENV, raising=False)
    old = datasets._set_for_tests(datasets.DatasetSpec("double_well", 200))
    yield
    datasets._set_for_tests(old)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _bind(folder):
    """COMPLETE.json bound to report.json, as run_job / run_prepared_condition write it."""
    _json(folder/"COMPLETE.json", dict(report_sha256=hashlib.sha256((folder/"report.json").read_bytes()).hexdigest()))


def _checkpoint(root, payload, step, prov=PROV):
    buffer = io.BytesIO(); torch.save(payload, buffer); raw = buffer.getvalue()
    path = root/"checkpoints"/f"step-{step:04d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    return dict(step=step, mean=.5+step/1000, windows=400, metric="future_jepa_distance",
                checkpoint=f"checkpoints/step-{step:04d}.pt", checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                checkpoint_provenance=prov)


def fake_job(base, condition, payloads=None, selected=10, layout="local", seed=1703):
    """A finished v7 job with the whole registered chain: job.json, report.json (+COMPLETE), evaluation registration,
    and the training run (local, or an older job referenced by training_run / resume_from)."""
    job_dir = base/"run"/"job"
    job = dict(condition=condition, dataset="double_well-200", seed=seed, updates=2000, resume_from=None, training_run=None)
    best, root = None, None
    if condition != "identity":
        train_job = job_dir
        if layout != "local":
            train_job = base/"old"/"run"/"job"
            _json(train_job/"job.json", dict(job))
            job[layout] = str(base/"old")
        root = train_job/"training"/condition
        rows = {step: _checkpoint(root, payload, step) for step, payload in (payloads or {10: {"x": 1}}).items()}
        best = rows[selected]
        _json(root/"report.json", dict(best=best, validations=list(rows.values()))); _bind(root)
    _json(job_dir/"job.json", job)
    _json(job_dir/"report.json", dict(job=job, preflight=dict(cache_identity=RECEIPT), best=best)); _bind(job_dir)
    evaluation_dir = job_dir/"evaluation"/condition
    _json(evaluation_dir/"registration.json", dict(condition=condition, checkpoint=best))
    _json(evaluation_dir/"COMPLETE.json", dict(status="completed_real_evaluation"))
    return job_dir, root


def sequence_sha():
    """Query-sequence SHA of the active protocol/dataset/seed, as comparison.preregister computes it."""
    from po_jepa.runtime import learning
    return comparison.sequence_sha(learning.make_query_sequence(checks.load_protocol())[0])


def registered_job(base, condition, payloads, selected, seed=1703, means=None):
    """A finished v7 job whose registered run is complete enough for verify_registered_run (needs v7_double_well):
    the full validation schedule with the 400 raw validation values per step (selected step strictly best), the
    training registration with the preregistered seed and query sequence, checkpoints whose provenance carries that
    sequence, and registered evaluation rows naming the checkpoint and seed. `payloads` must be built with PROV_SEQ()."""
    job_dir = base/"run"/"job"
    job = dict(condition=condition, dataset=datasets.active().tag, seed=seed, updates=checks.v7_updates(),
               resume_from=None, training_run=None)
    best = None
    if condition != "identity":
        root = job_dir/"training"/condition
        interval = checks.load_protocol()["validation"]["interval_updates"]
        steps = list(range(0, job["updates"]+1, interval))
        means = means or {s: (.25 if s == selected else .5) for s in steps}
        queries = [dict(trajectory_id=t, start_tick=st, duration_cycles=d) for t in datasets.validation_ids()
                   for st in (0, 5, 10, 15, 20) for d in (1., 2., 4., 8.)]
        prov = dict(PROV, f04_query_sequence_sha256=sequence_sha())
        rows = []
        for step in steps:
            row = _checkpoint(root, payloads[step], step, prov) if step in payloads else dict(
                step=step, windows=400, metric="future_jepa_distance", checkpoint=f"checkpoints/step-{step:04d}.pt",
                checkpoint_sha256="0"*64, checkpoint_provenance=prov)
            row["mean"] = means[step]
            rows.append(row)
            for offset in range(0, 400, 4):
                _json(root/"validations"/f"{step:04d}"/f"{offset:04d}.json",
                      dict(queries=queries[offset:offset+4], values=dict(future_jepa_distance=[means[step]]*4)))
        best = next(r for r in rows if r["step"] == selected)
        _json(root/"report.json", dict(best=best, validations=rows, completed_updates=job["updates"])); _bind(root)
        _json(root/"registration.json", dict(condition=condition, preregistration=dict(
            model_seed=seed, query_sequence_sha256=prov["f04_query_sequence_sha256"])))
    _json(job_dir/"job.json", job)
    _json(job_dir/"report.json", dict(job=job, preflight=dict(cache_identity=RECEIPT), best=best)); _bind(job_dir)
    evaluation_dir = job_dir/"evaluation"/condition
    _json(evaluation_dir/"registration.json", dict(condition=condition, checkpoint=best))
    _json(evaluation_dir/"COMPLETE.json", dict(status="completed_real_evaluation"))
    for tid in (180, 181):
        _json(evaluation_dir/"trajectories"/f"{tid:04d}.json", dict(records=[
            dict(condition=condition, seed=seed, checkpoint=best) for _ in range(2)], summary={}))
    return job_dir


def PROV_SEQ():
    return dict(PROV, f04_query_sequence_sha256=sequence_sha())


def _edit(path, change, bind=False):
    value = json.loads(path.read_text()); change(value); path.write_text(json.dumps(value))
    if bind:
        _bind(path.parent)


def test_formal_selection_follows_the_registered_chain(tmp_path):
    """V7-P1-2 / V7-P2-1: the checkpoint is the one completion flag, report and evaluation registration name."""
    job, root = fake_job(tmp_path/"a", "po_jepa_full", {10: {"x": 1}, 20: {"x": 2}})
    got = xe.formal_selection(job, "po_jepa_full")
    assert got["best"]["step"] == 10 and got["root"] == root and got["cache_identity"]["encode_report_sha256"] == "enc-A"
    with pytest.raises(ValueError, match="job.json is for"):
        xe.formal_selection(job, "po_jepa_no_consistency")
    # reviewer's case: training report re-pointed to another valid step-20 checkpoint (even with its flag rebound)
    _edit(root/"report.json", lambda r: r.update(best=r["validations"][1]), bind=True)
    with pytest.raises(ValueError, match="training report selects another checkpoint"):
        xe.formal_selection(job, "po_jepa_full")
    # training report no longer the one its COMPLETE.json names
    job, root = fake_job(tmp_path/"b", "po_jepa_full")
    _edit(root/"report.json", lambda r: r.update(note=1))
    with pytest.raises(ValueError, match="training report.json is not the one"):
        xe.formal_selection(job, "po_jepa_full")
    # top-level report edited without its flag; then rebound but the evaluation registration still names step 10
    job, root = fake_job(tmp_path/"c", "po_jepa_full", {10: {"x": 1}, 20: {"x": 2}})
    _edit(job/"report.json", lambda r: r.update(best=json.loads((root/"report.json").read_text())["validations"][1]))
    with pytest.raises(ValueError, match="not the one COMPLETE.json"):
        xe.formal_selection(job, "po_jepa_full")
    _bind(job)
    with pytest.raises(ValueError, match="evaluation registration and report"):
        xe.formal_selection(job, "po_jepa_full")
    # evaluation failed / missing
    job, _ = fake_job(tmp_path/"d", "po_jepa_full")
    _json(job/"evaluation"/"po_jepa_full"/"FAILED.json", {})
    with pytest.raises(ValueError, match="registered evaluation not complete"):
        xe.formal_selection(job, "po_jepa_full")
    # job.json relabelled to another seed: report.json still describes seed 1703
    job, _ = fake_job(tmp_path/"e", "po_jepa_full")
    _edit(job/"job.json", lambda j: j.update(seed=1704))
    with pytest.raises(ValueError, match="another job"):
        xe.formal_selection(job, "po_jepa_full")
    # identity has no checkpoint
    job, _ = fake_job(tmp_path/"f", "identity")
    assert xe.formal_selection(job, "identity")["best"] is None


@pytest.mark.parametrize("layout", ["resume_from", "training_run"])
def test_formal_selection_reused_training(tmp_path, layout):
    """V7-P2-1: a job that only evaluated an already finished training (no local training/ directory)."""
    job, root = fake_job(tmp_path, "matched_transformer", layout=layout)
    assert not (job/"training").exists()
    got = xe.formal_selection(job, "matched_transformer")
    assert got["root"] == root and root.is_relative_to(tmp_path/"old") and got["best"]["step"] == 10


def _relabel_best(job, root, condition, step, rows_too):
    """Re-point the training report, the top-level report and the evaluation registration (and optionally the
    registered rows) consistently to another stored checkpoint, rebinding every COMPLETE flag."""
    target = next(r for r in json.loads((root/"report.json").read_text())["validations"] if r["step"] == step)
    _edit(root/"report.json", lambda r: r.update(best=target), bind=True)
    _edit(job/"report.json", lambda r: r.update(best=target), bind=True)
    _edit(job/"evaluation"/condition/"registration.json", lambda r: r.update(checkpoint=target))
    if rows_too:
        for path in (job/"evaluation"/condition/"trajectories").glob("*.json"):
            _edit(path, lambda t: [r.update(checkpoint=target) for r in t["records"]])
    return target


def test_verify_registered_run_refuses_consistent_rewrites(tmp_path, v7_double_well, monkeypatch):
    """R4-2-P1-1: rewriting best in all three records (flags rebound) or relabelling the seed is caught by what a
    metadata rewrite cannot change: registered rows, the raw validation values, the seed-dependent query sequence."""
    model, _ = comparison.build_condition("matched_transformer", "cpu")
    def payload(step):
        return dict(schema="F05_transformer_checkpoint_v1", config=asdict(model.spec), provenance=PROV_SEQ(),
                    completed_steps=step, model_state=model.state_dict())
    def make(name, seed=1703):
        job = registered_job(tmp_path/name, "matched_transformer", {200: payload(200), 400: payload(400)}, 200, seed=seed)
        return job, job/"training"/"matched_transformer"
    job, root = make("ok")
    sel = xe.formal_selection(job, "matched_transformer")
    xe.verify_registered_run(job, sel)
    assert xe.load_selected("matched_transformer", job, torch.device("cpu"), sel)[2]["step"] == 200
    # reviewer's case: best re-pointed to the stored step-400 checkpoint in all three records, rows untouched
    job, root = make("repoint")
    _relabel_best(job, root, "matched_transformer", 400, rows_too=False)
    sel = xe.formal_selection(job, "matched_transformer")                     # the metadata chain alone accepts it
    with pytest.raises(ValueError, match="registered evaluation row is for checkpoint step 200"):
        xe.verify_registered_run(job, sel)
    # ... and with the rows rewritten too: the selection no longer follows from the raw validation values
    job, root = make("repoint-rows")
    _relabel_best(job, root, "matched_transformer", 400, rows_too=True)
    with pytest.raises(ValueError, match="does not follow from the raw validation values"):
        xe.verify_registered_run(job, xe.formal_selection(job, "matched_transformer"))
    # seed relabel: job.json / report.job -> 1704 with the environment on 1704 (as run_all would set it)
    job, root = make("seed")
    monkeypatch.setenv(checks.SEED_ENV, "1704")
    for path in (job/"job.json",):
        _edit(path, lambda j: j.update(seed=1704))
    _edit(job/"report.json", lambda r: r["job"].update(seed=1704), bind=True)
    sel = xe.formal_selection(job, "matched_transformer")
    with pytest.raises(ValueError, match="seed 1703, not the selected step 200 / seed 1704"):
        xe.verify_registered_run(job, sel)
    for path in (job/"evaluation"/"matched_transformer"/"trajectories").glob("*.json"):  # records relabelled too
        _edit(path, lambda t: [r.update(seed=1704) for r in t["records"]])
    with pytest.raises(ValueError, match="was not run with seed 1704"):
        xe.verify_registered_run(job, xe.formal_selection(job, "matched_transformer"))
    # R5-v7 P3: each of the two sequence guards on its own. (a) training preregistration rewritten to seed 1704 and
    # its sequence, only the selected checkpoint's provenance still carries seed 1703's sequence
    seq_now = sequence_sha()
    _edit(root/"registration.json", lambda r: r["preregistration"].update(model_seed=1704, query_sequence_sha256=seq_now))
    with pytest.raises(ValueError, match="was not run with seed 1704.*'checkpoint': \\(1704, '(?!" + seq_now + ")"):
        xe.verify_registered_run(job, xe.formal_selection(job, "matched_transformer"))
    monkeypatch.delenv(checks.SEED_ENV)
    # (b) a seed-1703 job whose checkpoint provenance is right but whose training preregistration names another seed
    job_b, root_b = make("prereg-only")
    _edit(root_b/"registration.json", lambda r: r["preregistration"].update(model_seed=1704))
    with pytest.raises(ValueError, match="was not run with seed 1703"):
        xe.verify_registered_run(job_b, xe.formal_selection(job_b, "matched_transformer"))
    monkeypatch.setenv(checks.SEED_ENV, "1704")
    # a run genuinely trained with seed 1704 passes under seed 1704 (sequence differs from seed 1703's)
    seq_1704 = sequence_sha()
    monkeypatch.delenv(checks.SEED_ENV)
    assert sequence_sha() != seq_1704
    monkeypatch.setenv(checks.SEED_ENV, "1704")
    job, root = make("genuine-1704", seed=1704)
    xe.verify_registered_run(job, xe.formal_selection(job, "matched_transformer"))
    # identity: rows only
    monkeypatch.delenv(checks.SEED_ENV)
    job = registered_job(tmp_path/"identity", "identity", {}, None)
    xe.verify_registered_run(job, xe.formal_selection(job, "identity"))
    _edit(sorted((job/"evaluation"/"identity"/"trajectories").glob("*.json"))[1], lambda t: t["records"][1].update(seed=1704))
    with pytest.raises(ValueError, match="registered evaluation row"):
        xe.verify_registered_run(job, xe.formal_selection(job, "identity"))
    for path in (job/"evaluation"/"identity"/"trajectories").glob("*.json"):            # no records at all
        _edit(path, lambda t: t.update(records=[]))
    with pytest.raises(ValueError, match="no trajectory records"):
        xe.verify_registered_run(job, xe.formal_selection(job, "identity"))


def _po_payload(model, step=10, condition="full"):
    state = {k: v+.01 if v.is_floating_point() else v for k, v in model.state_dict().items()}
    return dict(completed_steps=step, condition=condition, model_config=training._model_config(model), model_state=state,
                provenance=PROV), state


def test_load_selected_po_checkpoint(tmp_path, v7_double_well):
    model, _ = comparison.build_condition("po_jepa_full", "cpu")
    payload, state = _po_payload(model)
    job, root = fake_job(tmp_path/"ok", "po_jepa_full", {10: payload})
    loaded, _, info = xe.load_selected("po_jepa_full", job, torch.device("cpu"))
    assert info["step"] == 10 and info["training_root"] == str(root) and not loaded.training
    for k, v in loaded.state_dict().items():
        assert torch.equal(v, state[k]), k
    refused = dict(condition=dict(payload, condition="no_consistency"), step=dict(payload, completed_steps=11),
                   provenance=dict(payload, provenance=dict(PROV, feature_manifest_sha256="e"*64)))
    for name, bad in refused.items():
        job, _ = fake_job(tmp_path/name, "po_jepa_full", {10: bad})
        with pytest.raises(ValueError):
            xe.load_selected("po_jepa_full", job, torch.device("cpu"))
    job, root = fake_job(tmp_path/"bytes", "po_jepa_full", {10: payload})
    with open(root/"checkpoints"/"step-0010.pt", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="SHA"):
        xe.load_selected("po_jepa_full", job, torch.device("cpu"))


def test_load_selected_transformer_checkpoint(tmp_path, v7_double_well):
    model, _ = comparison.build_condition("matched_transformer", "cpu")
    payload = dict(schema="F05_transformer_checkpoint_v1", config=asdict(model.spec), provenance=PROV, completed_steps=10,
                   model_state=_perturb(model).state_dict())
    job, _ = fake_job(tmp_path/"ok", "matched_transformer", {10: payload}, layout="resume_from")
    loaded, _, info = xe.load_selected("matched_transformer", job, torch.device("cpu"))
    assert info["step"] == 10 and all(torch.equal(v, payload["model_state"][k]) for k, v in loaded.state_dict().items())
    job, _ = fake_job(tmp_path/"arch", "matched_transformer", {10: dict(payload, config=dict(payload["config"], width=32))})
    with pytest.raises(ValueError, match="architecture"):
        xe.load_selected("matched_transformer", job, torch.device("cpu"))


def test_features_identity(monkeypatch):
    """V7-P1-3: another cache with the same dataset tag is refused; the digest goes into the output identity."""
    monkeypatch.setattr(data, "approved_cache_metadata", lambda root, splits: dict(RECEIPT))
    receipt = json.loads(json.dumps(RECEIPT))                      # as read back from report.json (string keys)
    digest = xe.features_identity(Path("A"), receipt)
    assert digest["encode_report_sha256"] == "enc-A" and len(digest["manifests_sha256"]) == 64
    monkeypatch.setattr(data, "approved_cache_metadata",
                        lambda root, splits: dict(RECEIPT, encode_report_sha256="enc-B"))
    with pytest.raises(ValueError, match="differs from the one"):
        xe.features_identity(Path("B"), receipt)
    monkeypatch.setattr(data, "approved_cache_metadata",
                        lambda root, splits: dict(RECEIPT, manifest_sha256_by_id={180: "m180", 181: "other"}))
    with pytest.raises(ValueError):
        xe.features_identity(Path("B"), receipt)


def test_run_one_end_to_end(tmp_path, v7_double_well, monkeypatch, capsys):
    """Real matched Transformer with a non-zero readout: rows equal the registered point_metrics; output identity,
    reuse only on an equal identity, refusal otherwise, atomic write, environment/job consistency."""
    model, loss = comparison.build_condition("matched_transformer", "cpu")
    weights = _perturb(model).state_dict()
    payload = dict(schema="F05_transformer_checkpoint_v1", config=asdict(model.spec), provenance=PROV_SEQ(),
                   completed_steps=200, model_state=weights)
    job = registered_job(tmp_path, "matched_transformer", {200: payload}, 200)
    torch.manual_seed(1)
    frames = torch.randn(20, 1, 256, 1408)

    def reader(features, approved, tid, start, T, device):
        end = start+int(T)
        if end >= frames.shape[0]:
            return None
        return frames[start], frames[end], cache._window_controls(dict(META, trajectory_id=tid), start, float(T), str(device))

    monkeypatch.setattr(xe, "read_test_window", reader)
    monkeypatch.setattr(xe, "DURATIONS", (1., 3., 6., 16.))
    monkeypatch.setattr(datasets, "test_start_ticks", lambda tid, spec=None: (5,))
    monkeypatch.setattr(data, "approved_cache_metadata", lambda root, splits: dict(RECEIPT))
    out = tmp_path/"out"/"double_well-200"/"seed1703"/"matched_transformer.json"
    xe.run_one(job, tmp_path, out, "cpu", max_test=1)
    result = json.loads(out.read_text())
    rows = {r["duration_cycles"]: r for r in result["rows"]}
    assert set(rows) == {1., 3., 6.} and all(r["trajectory_id"] == 180 and r["start_tick"] == 5 for r in rows.values())
    identity = result["identity"]
    assert identity["metric_version"] == xe.METRIC_VERSION and identity["checkpoint"]["step"] == 200
    assert identity["features"]["encode_report_sha256"] == "enc-A" and identity["max_test_trajectories"] == 1
    reference, _ = comparison.build_condition("matched_transformer", "cpu")
    reference.load_state_dict(weights); reference.eval()
    for T in (1., 3.):
        values, _ = _registered(reference, frames[5], frames[5+int(T)], T, loss)
        assert rows[T]["direct"] == pytest.approx(values["future_jepa_distance"], rel=1e-6, abs=0)
        assert rows[T]["unit"] == pytest.approx(values["rollout_future_jepa_distance"], rel=1e-6, abs=0)
    assert [p.name for p in out.parent.iterdir()] == [out.name]              # no partial file left behind
    # same identity: reused, file untouched
    before = out.read_bytes(); capsys.readouterr()
    xe.run_one(job, tmp_path, out, "cpu", max_test=1)
    assert out.read_bytes() == before and "skip (same identity)" in capsys.readouterr().out
    # another budget / subset / cache: refused, not reused and not overwritten
    with pytest.raises(ValueError, match="another run"):
        xe.run_one(job, tmp_path, out, "cpu", max_test=2)
    monkeypatch.setattr(data, "approved_cache_metadata", lambda root, splits: dict(RECEIPT, encode_report_sha256="enc-B"))
    with pytest.raises(ValueError, match="feature cache differs"):
        xe.run_one(job, tmp_path, out, "cpu", max_test=1)
    monkeypatch.setattr(data, "approved_cache_metadata", lambda root, splits: dict(RECEIPT))
    assert out.read_bytes() == before
    # a truncated leftover (from a305bfe's non-atomic write) is recomputed
    out.write_text(before.decode()[:100])
    xe.run_one(job, tmp_path, out, "cpu", max_test=1)
    assert json.loads(out.read_text())["identity"] == identity
    with pytest.raises(ValueError):
        xe.run_one(job, tmp_path, out, "cpu", max_test=0)
    # environment and job.json must agree (run sets the dataset/seed/updates from job.json)
    _edit(job/"job.json", lambda j: j.update(seed=1704))
    with pytest.raises(ValueError, match="environment differ"):
        xe.run_one(job, tmp_path, out, "cpu", max_test=1)


def _finished(results, scene, condition, seed=1703):
    job = results/scene/f"seed{seed}"/condition/"run"/"job"; job.mkdir(parents=True)
    _json(job/"job.json", dict(condition=condition, dataset=f"{scene}-200", seed=seed, updates=2000))
    _json(job/"COMPLETE.json", {})
    return job


def test_run_all_refuses_duplicates_and_bad_limits(tmp_path):
    results = tmp_path/"results"
    _finished(results, "van_der_pol", "po_jepa_full")
    _finished(results/"again", "van_der_pol", "po_jepa_full")
    with pytest.raises(SystemExit, match="several finished jobs"):
        xe.main(["run", "--results-root", str(results), "--output", str(tmp_path/"out"), "--features-base", "x"])
    for bad in ("0", "-1"):
        with pytest.raises(SystemExit):
            xe.main(["run", "--results-root", str(results), "--output", str(tmp_path/"out"), "--max-test-trajectories", bad])


def _result(out, condition, rows, dataset="van_der_pol-200", seed=1703, features="enc-A", identity=True, slot=None):
    path = out/(slot or dataset)/f"seed{seed}"/f"{condition}.json"
    value = dict(condition=condition, dataset=dataset, seed=seed, rows=rows)
    if identity:
        value["identity"] = dict(metric_version=xe.METRIC_VERSION, dataset=dataset, seed=seed, condition=condition,
                                 updates=2000, features=dict(encode_report_sha256=features), max_test_trajectories=None,
                                 durations=list(xe.DURATIONS))
    _json(path, value)


def _rows(values, T=2.):
    return [dict(trajectory_id=t, start_tick=0, duration_cycles=T, id=1., direct=v, half=v+.1, unit=v+.2, dc_half=.1)
            for t, v in values.items()]


def test_summarize(tmp_path):
    _result(tmp_path, "po_jepa_full", _rows({180: .5, 181: .55}))
    _result(tmp_path, "po_jepa_no_consistency", _rows({180: .6, 181: .6}))
    xe.summarize(type("A", (), dict(output=str(tmp_path)))())
    entry = json.loads((tmp_path/"summary.json").read_text())[0]
    assert entry["means"]["po_jepa_full"]["direct"] == pytest.approx(.525)
    assert entry["full_vs_nocons"]["direct"]["relative"] == pytest.approx((.525-.6)/.6)
    assert entry["full_vs_nocons"]["unit"]["relative"] == pytest.approx((.725-.8)/.8)
    assert entry["full_vs_nocons"]["unit"]["full_better_trajectories"] == 2
    assert "id" not in entry["full_vs_nocons"] and entry["means"]["po_jepa_full"]["windows"] == 2
    assert entry["pairing"] == dict(paired_windows=2, full_only=0, nocons_only=0)


def test_summarize_unpaired_and_missing_durations(tmp_path):
    """V7-P2-2: paired statistics use the common windows only and say how many were left out; a condition with no
    window at some T gives None means instead of crashing."""
    _result(tmp_path, "po_jepa_full", _rows({180: 1., 181: 9.}))
    _result(tmp_path, "po_jepa_no_consistency", _rows({180: 2.}) + _rows({180: 4.}, T=3.))
    xe.summarize(type("A", (), dict(output=str(tmp_path)))())
    two, three = json.loads((tmp_path/"summary.json").read_text())
    assert two["means"]["po_jepa_full"]["direct"] == pytest.approx(5.) and two["means"]["po_jepa_full"]["windows"] == 2
    assert two["pairing"] == dict(paired_windows=1, full_only=1, nocons_only=0)
    assert two["full_vs_nocons"]["direct"]["full_mean"] == pytest.approx(1.)
    assert two["full_vs_nocons"]["direct"]["relative"] == pytest.approx(-.5)
    assert three["means"]["po_jepa_full"]["windows"] == 0 and three["means"]["po_jepa_full"]["direct"] is None
    assert three["pairing"] == dict(paired_windows=0, full_only=0, nocons_only=1) and three["full_vs_nocons"] == {}


@pytest.mark.parametrize("case", ["old_format", "wrong_slot", "other_cache"])
def test_summarize_refuses_unidentified_or_mixed_results(tmp_path, case):
    _result(tmp_path, "po_jepa_full", _rows({180: .5}))
    if case == "old_format":
        _result(tmp_path, "po_jepa_no_consistency", _rows({180: .6}), identity=False)
    elif case == "wrong_slot":
        _result(tmp_path, "po_jepa_no_consistency", _rows({180: .6}), dataset="double_well-200", slot="van_der_pol-200")
    else:
        _result(tmp_path, "po_jepa_no_consistency", _rows({180: .6}), features="enc-B")
    with pytest.raises(SystemExit):
        xe.summarize(type("A", (), dict(output=str(tmp_path)))())
    assert not (tmp_path/"summary.json").exists()


def test_find_jobs_and_features_for(tmp_path):
    out = tmp_path/"pilot"/"van_der_pol"/"seed1703"/"po_jepa_full"
    job = out/"run"/"job"; job.mkdir(parents=True)
    (job/"job.json").write_text(json.dumps(dict(condition="po_jepa_full", dataset="van_der_pol-200", seed=1703, updates=2000)))
    (job/"COMPLETE.json").write_text("{}")
    other = tmp_path/"pilot"/"van_der_pol"/"seed1703"/"vjepa2_ac"/"run"/"job"; other.mkdir(parents=True)
    (other/"job.json").write_text(json.dumps(dict(condition="vjepa2_ac", dataset="van_der_pol-200", seed=1703, updates=2000)))
    (other/"COMPLETE.json").write_text("{}")
    unfinished = tmp_path/"pilot"/"van_der_pol"/"seed1703"/"identity"/"run"/"job"; unfinished.mkdir(parents=True)
    (unfinished/"job.json").write_text(json.dumps(dict(condition="identity", dataset="van_der_pol-200", seed=1703, updates=2000)))
    jobs = xe.find_jobs(tmp_path/"pilot")
    assert [j[0] for j in jobs] == [job] and xe.duplicate_keys(jobs) == []
    recorded = tmp_path/"features"/"vdp"; recorded.mkdir(parents=True)
    (out/"console").mkdir()
    (out/"console"/"command.json").write_text(json.dumps(dict(argv=["python", "run", "--features-root", str(recorded)])))
    assert xe.features_for(job, jobs[0][1], None) == recorded
    (out/"console"/"command.json").unlink()
    assert xe.features_for(job, jobs[0][1], tmp_path/"base") == (tmp_path/"base").resolve()/"van_der_pol-200"
    with pytest.raises(ValueError):
        xe.features_for(job, jobs[0][1], None)
