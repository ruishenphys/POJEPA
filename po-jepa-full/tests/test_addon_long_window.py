"""S18a：长时程窗口（10 §3.7）。read_long_window 除时长和起点的检查外和 cache.read_window 逐字段相同；
d4.load_long_test_window 的核法和 evaluation.load_test_window 相同。缓存用内存里的假缓存（monkeypatch
cache._read_complete_cache），33 帧、float16，和真实仿真缓存的形状一致。"""

from pathlib import Path

import numpy as np
import pytest
import torch

from po_jepa import datasets
from po_jepa.runtime import cache, comparison, data
from po_jepa.runtime.addon import d4, long_window

FRAMES = 33
CACHE_SHA = "3cc53a9e1826074e7255ad334179cf94233e581e35b9d6ec5c01493285c1c03c"


def metadata(tid=36, split="test", role="real_frozen_encoder", variant="canonical"):
    return dict(split=split, trajectory_id=tid, frames=[dict(image_sha256=f"img-{k:02d}") for k in range(FRAMES)],
                generation=dict(evidence_role=role), preprocess_sha256="preprocess", observation_variant=variant,
                controls=dict(starts_ticks=list(range(32)), ends_ticks=list(range(1, 33)),
                              values=[[.01*(k+1), -.005*k] for k in range(32)], units="simulation_position_per_second"))


@pytest.fixture
def fake_cache(monkeypatch):
    raw = ((np.arange(FRAMES*256*1408, dtype=np.int64).reshape(FRAMES, 256, 1408) % 251)/251.-.5).astype(np.float16)
    state = dict(m=metadata(), sha="manifest-36")
    monkeypatch.setattr(cache, "_read_complete_cache", lambda path: (raw, state["m"], state["sha"]))
    return raw, state


def same_window(a, b):
    x, y = a.inputs, b.inputs
    for p, q in ((x.raw_z, y.raw_z), (a.target.raw_z, b.target.raw_z), (x.duration, y.duration),
                 (x.observed_time_seconds, y.observed_time_seconds), (a.target.target_time_seconds, b.target.target_time_seconds),
                 (x.grid.coords, y.grid.coords)):
        assert p.dtype == q.dtype and p.device == q.device and p.shape == q.shape and torch.equal(p, q)
    for field in ("starts", "ends", "u_values", "n_segments", "segment_valid", "source_origin_seconds", "duration"):
        p, q = getattr(x.controls, field), getattr(y.controls, field)
        assert p.dtype == q.dtype and torch.equal(p, q), field
    assert x.controls.source_ids == y.controls.source_ids and x.controls.control_spec == y.controls.control_spec
    assert (x.image_sha256, x.trajectory_id, x.split, a.target.image_sha256) == (y.image_sha256, y.trajectory_id, y.split, b.target.image_sha256)
    assert a.provenance == b.provenance


@pytest.mark.parametrize("start,t", [(0, 1.), (5, 4.), (3, 8.), (20, 12.), (0, 12.)])
def test_body_equals_read_window(fake_cache, monkeypatch, start, t):
    """把 12 等短时长临时放进 LONG_TIMES：同一 (起点, 时长) 上两个读法逐字段逐位相同。"""
    monkeypatch.setattr(long_window, "LONG_TIMES", (1., 4., 8., 12., 16., 24., 32.))
    a = long_window.read_long_window(Path("x"), start_tick=start, duration_cycles=t, expected_split="test", device="cpu")
    b = cache.read_window(Path("x"), start_tick=start, duration_cycles=t, expected_split="test", device="cpu")
    same_window(a, b)


@pytest.mark.parametrize("t", [16., 24., 32.])
def test_long_window_fields(fake_cache, t):
    raw, _ = fake_cache
    w = long_window.read_long_window(Path("x"), start_tick=0, duration_cycles=t, expected_split="test", device="cpu")
    assert w.inputs.raw_z.dtype == torch.float32 and w.inputs.raw_z.shape == (1, 256, 1408)
    assert torch.equal(w.inputs.raw_z, torch.from_numpy(raw[0:1].astype(np.float32)))
    assert torch.equal(w.target.raw_z, torch.from_numpy(raw[int(t):int(t)+1].astype(np.float32)))
    assert float(w.inputs.duration[0]) == t and float(w.inputs.observed_time_seconds[0]) == 0.
    assert float(w.target.target_time_seconds[0]) == t*.25 and w.target.image_sha256 == f"img-{int(t):02d}"
    assert float(w.inputs.controls.duration[0]) == t and int(w.inputs.controls.n_segments[0]) == int(t)
    assert w.inputs.controls.u_values[0, int(t)-1, 0].item() == pytest.approx(.01*t)
    data._window_boundary(w, w)


SPLICES = {16.: ((0, 12.), (12, 4.)), 24.: ((0, 12.), (12, 12.)), 32.: ((0, 12.), (12, 8.), (20, 12.))}


@pytest.mark.parametrize("t", [16., 24., 32.])
def test_long_window_matches_spliced_read_window(fake_cache, t):
    """14 §3 S18：起点帧等于第一段 read_window 的起点帧，终点帧等于最后一段的终点帧，动作等于各段 read_window 的
    动作依次 concatenate_action_segments（read_window 只收起点 0–20、T ≤ 12，所以按 SPLICES 分段）。"""
    long = long_window.read_long_window(Path("x"), start_tick=0, duration_cycles=t, expected_split="test", device="cpu")
    parts = [cache.read_window(Path("x"), start_tick=s, duration_cycles=d, expected_split="test", device="cpu") for s, d in SPLICES[t]]
    assert sum(d for _, d in SPLICES[t]) == t
    first, last = parts[0], parts[-1]
    assert torch.equal(long.inputs.raw_z, first.inputs.raw_z) and long.inputs.image_sha256 == first.inputs.image_sha256
    assert torch.equal(long.inputs.observed_time_seconds, first.inputs.observed_time_seconds)
    assert torch.equal(long.target.raw_z, last.target.raw_z) and long.target.image_sha256 == last.target.image_sha256
    assert torch.equal(long.target.target_time_seconds, last.target.target_time_seconds)
    controls = first.inputs.controls
    for part in parts[1:]:
        controls = comparison.concatenate_action_segments(controls, part.inputs.controls)
    for field in ("starts", "ends", "u_values", "n_segments", "segment_valid", "source_origin_seconds", "duration"):
        p, q = getattr(long.inputs.controls, field), getattr(controls, field)
        assert p.dtype == q.dtype and torch.equal(p, q), field
    assert long.inputs.controls.source_ids == controls.source_ids and long.inputs.controls.control_spec == controls.control_spec


def test_long_window_refusals(fake_cache):
    _, state = fake_cache
    read = long_window.read_long_window
    for t in (12., 8., 40., 16.5, "16"):
        with pytest.raises(ValueError, match="16, 24 or 32"):
            read(Path("x"), start_tick=0, duration_cycles=t, expected_split="test", device="cpu")
    for start in (-1, 1., True):
        with pytest.raises(ValueError, match="nonnegative integer"):
            read(Path("x"), start_tick=start, duration_cycles=16., expected_split="test", device="cpu")
    assert read(Path("x"), start_tick=16, duration_cycles=16., expected_split="test", device="cpu").target.image_sha256 == "img-32"
    with pytest.raises(ValueError, match="past the last cached frame"):
        read(Path("x"), start_tick=1, duration_cycles=32., expected_split="test", device="cpu")
    with pytest.raises(ValueError, match="past the last cached frame"):
        read(Path("x"), start_tick=9, duration_cycles=24., expected_split="test", device="cpu")
    with pytest.raises(ValueError, match="cross-split"):
        read(Path("x"), start_tick=0, duration_cycles=16., expected_split="validation", device="cpu")


@pytest.fixture
def van_der_pol(monkeypatch):
    old = datasets._set_for_tests(datasets.DatasetSpec("van_der_pol", 40))
    monkeypatch.setattr(data, "approved_cache_metadata", lambda root, splits: dict(manifest_sha256_by_id={36: "manifest-36", 37: "manifest-37"}))
    yield
    datasets._set_for_tests(old)


def test_load_long_test_window_checks_identity(fake_cache, van_der_pol):
    _, state = fake_cache
    w = d4.load_long_test_window(Path("features"), 36, 32., "cpu")
    assert w.target.image_sha256 == "img-32" and w.inputs.trajectory_id == 36
    for tid in (35, 40, 36.):                                    # 验证轨迹、越界、非 int
        with pytest.raises(ValueError, match="test trajectory"):
            d4.load_long_test_window(Path("features"), tid, 16., "cpu")
    for change in (dict(sha="manifest-other"), dict(m=metadata(role="fictional_encoder")), dict(m=metadata(variant="shifted")),
                   dict(m=metadata(tid=37))):
        saved = dict(state)
        state.update(change)
        with pytest.raises(ValueError, match="identity/source/origin/endpoint"):
            d4.load_long_test_window(Path("features"), 36, 16., "cpu")
        state.update(saved)


def test_cache_py_unchanged():
    assert cache.file_sha(Path(cache.__file__)) == CACHE_SHA
