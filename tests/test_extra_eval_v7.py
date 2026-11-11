"""tools/extra_eval_v7.py: partitions, metrics, checkpoint loading, one job end to end, summary (CPU, fictional data)."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("extra_eval_v7", ROOT/"tools/extra_eval_v7.py")
xe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xe)

from po_jepa import datasets  # noqa: E402
from po_jepa.runtime import cache, checks, comparison, training  # noqa: E402
from test_training import e06_config  # noqa: E402

META = dict(trajectory_id=180, controls=dict(starts_ticks=[0, 8, 16, 24], ends_ticks=[8, 16, 24, 32],
            values=[[.1, 0.], [.2, 0.], [.3, 0.], [.4, 0.]], units="simulation_position_per_second"))


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
    got = xe.window_metrics(model, raw, target, controls, 6., cache.regular_grid(16, 16), dist, comparison.split_action_segments)
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
    odd = xe.window_metrics(model, raw, target, cache._window_controls(META, 0, 3., "cpu"), 3., cache.regular_grid(16, 16),
                            dist, comparison.split_action_segments)
    assert odd["half"] == pytest.approx(dist(apply(raw, (1.5, 1.5), c), target)) and "p2_4" not in odd
    assert odd["unit"] == pytest.approx(dist(apply(raw, (1., 1., 1.), c), target))


@pytest.fixture
def v7_double_well(monkeypatch):
    if not (ROOT/"configs/calibration/double_well-200-u2000.yaml").exists():
        pytest.skip("needs configs/calibration/double_well-200-u2000.yaml (written by run.py calibrate)")
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    monkeypatch.setenv(checks.UPDATES_ENV, "2000")
    old = datasets._set_for_tests(datasets.DatasetSpec("double_well", 200))
    yield
    datasets._set_for_tests(old)


def fake_job(tmp_path, condition, payload, step=10):
    job = tmp_path/"run"/"job"
    root = job/"training"/condition
    (root/"checkpoints").mkdir(parents=True)
    (job/"job.json").write_text(json.dumps(dict(condition=condition, dataset="double_well-200", seed=1703, updates=2000,
                                                 training_run=None, resume_from=None)))
    (job/"COMPLETE.json").write_text("{}"); (root/"COMPLETE.json").write_text("{}")
    buffer = io.BytesIO(); torch.save(payload, buffer); raw = buffer.getvalue()
    (root/"checkpoints"/f"step-{step}.pt").write_bytes(raw)
    best = dict(checkpoint=f"checkpoints/step-{step}.pt", checkpoint_sha256=hashlib.sha256(raw).hexdigest(), step=step, mean=.5)
    (root/"report.json").write_text(json.dumps(dict(best=best)))
    return job, root


def test_load_selected_po_checkpoint(tmp_path, v7_double_well):
    model, _ = comparison.build_condition("po_jepa_full", "cpu")
    state = {k: v+.01 if v.is_floating_point() else v for k, v in model.state_dict().items()}
    payload = dict(completed_steps=10, condition="full", model_config=training._model_config(model), model_state=state)
    job, root = fake_job(tmp_path, "po_jepa_full", payload)
    loaded, _, info = xe.load_selected("po_jepa_full", job, torch.device("cpu"))
    assert info["step"] == 10 and not loaded.training
    for k, v in loaded.state_dict().items():
        assert torch.equal(v, state[k]), k
    # tampering is refused: other condition, wrong step, changed bytes
    with pytest.raises(ValueError):
        xe.load_selected("po_jepa_no_consistency", *_relabel(tmp_path/"b", payload, "po_jepa_no_consistency"))
    report = json.loads((root/"report.json").read_text()); report["best"]["step"] = 11
    (root/"report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        xe.load_selected("po_jepa_full", job, torch.device("cpu"))
    report["best"]["step"] = 10; (root/"report.json").write_text(json.dumps(report))
    with open(root/"checkpoints"/"step-10.pt", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError):
        xe.load_selected("po_jepa_full", job, torch.device("cpu"))


def _relabel(path, payload, condition):
    path.mkdir()
    job, _ = fake_job(path, condition, payload)       # payload still says condition="full"
    return job, torch.device("cpu")


def test_run_one_end_to_end(tmp_path, v7_double_well, monkeypatch):
    model, _ = comparison.build_condition("po_jepa_full", "cpu")
    payload = dict(completed_steps=10, condition="full", model_config=training._model_config(model), model_state=model.state_dict())
    job, _ = fake_job(tmp_path, "po_jepa_full", payload)
    torch.manual_seed(1)
    frames = torch.randn(18, 1, 256, 1408)
    reads = []

    def reader(features, approved, tid, start, T, device):
        reads.append((tid, start, T))
        end = start+int(T)
        if end >= frames.shape[0]:
            return None
        meta = dict(META, trajectory_id=tid)
        return frames[start], frames[end], cache._window_controls(meta, start, float(T), str(device))

    monkeypatch.setattr(xe, "read_test_window", reader)
    monkeypatch.setattr(xe, "DURATIONS", (1., 6., 16.))
    monkeypatch.setattr(datasets, "test_start_ticks", lambda tid, spec=None: (0, 5))
    from po_jepa.runtime import data
    monkeypatch.setattr(data, "approved_cache_metadata", lambda root, splits: {})
    out = tmp_path/"out"/"r.json"
    xe.run_one(job, tmp_path, out, "cpu", max_test=1)
    result = json.loads(out.read_text())
    rows = result["rows"]
    assert result["checkpoint"]["step"] == 10 and result["max_test_trajectories"] == 1
    assert {(r["start_tick"], r["duration_cycles"]) for r in rows} == {(0, 1.), (0, 6.), (0, 16.), (5, 1.), (5, 6.)}
    assert all(r["trajectory_id"] == 180 for r in rows)
    six = next(r for r in rows if r["duration_cycles"] == 6. and r["start_tick"] == 0)
    loaded, loss, _ = xe.load_selected("po_jepa_full", job, torch.device("cpu"))
    with torch.no_grad():
        direct = loaded(frames[0], cache._window_controls(META, 0, 6., "cpu"), torch.tensor([6.], dtype=torch.float64),
                        cache.regular_grid(16, 16))
    assert six["direct"] == pytest.approx(float(comparison.jepa_distance(direct, frames[6], config=loss).per_sample[0]), rel=1e-5)
    assert six["half"] != pytest.approx(six["direct"])        # a real model is not an exact semigroup


def test_summarize(tmp_path):
    def write(condition, values):
        rows = [dict(trajectory_id=t, start_tick=0, duration_cycles=2., id=1., direct=v, half=v+.1, unit=v+.2, dc_half=.1)
                for t, v in values.items()]
        path = tmp_path/"van_der_pol-200"/"seed1703"/f"{condition}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(condition=condition, dataset="van_der_pol-200", seed=1703, rows=rows)))
    write("po_jepa_full", {180: .5, 181: .55})
    write("po_jepa_no_consistency", {180: .6, 181: .6})
    xe.summarize(type("A", (), dict(output=str(tmp_path)))())
    entry = json.loads((tmp_path/"summary.json").read_text())[0]
    assert entry["means"]["po_jepa_full"]["direct"] == pytest.approx(.525)
    assert entry["full_vs_nocons"]["direct"]["relative"] == pytest.approx((.525-.6)/.6)
    assert entry["full_vs_nocons"]["unit"]["relative"] == pytest.approx((.725-.8)/.8)
    assert entry["full_vs_nocons"]["unit"]["full_better_trajectories"] == 2
    assert "id" not in entry["full_vs_nocons"] and entry["means"]["po_jepa_full"]["windows"] == 2


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
    assert [j[0] for j in jobs] == [job]
    recorded = tmp_path/"features"/"vdp"; recorded.mkdir(parents=True)
    (out/"console").mkdir()
    (out/"console"/"command.json").write_text(json.dumps(dict(argv=["python", "run", "--features-root", str(recorded)])))
    assert xe.features_for(job, jobs[0][1], None) == recorded
    (out/"console"/"command.json").unlink()
    assert xe.features_for(job, jobs[0][1], tmp_path/"base") == (tmp_path/"base").resolve()/"van_der_pol-200"
    with pytest.raises(ValueError):
        xe.features_for(job, jobs[0][1], None)
