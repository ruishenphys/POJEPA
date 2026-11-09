"""S25d：时间采样测试集（调节项 fps、jitter、exposure、switch_offset，S24b）的缓存读取和按帧取窗口。

缓存由 tools/encode_v7.py --splits test 写出：帧数 L = 32·fps/4 + 1（17/33/65/129），manifest 的 frames 记每帧的实际时刻
time_ticks（= timing.frame_times，带抖动时不是 Δ 的整数倍），switch_offset > 0 时下达动作是 systems.SubtickControls
（controls 记 starts_subticks/ends_subticks，1/4 tick 为单位）。

cache.py 一个字节不改（S18a 钉了它的 SHA）。cache._read_v7_cache 写死“33 帧、第 k 帧在 tick k、TrajectoryControls”，
时间采样的缓存三条都不满足，所以这里是它的时间采样版（照 S25c 的 long_cache 的写法）：

- read_complete(path)：核法逐条照抄 cache._read_complete_cache_uncached 和 cache._read_v7_cache；不同的只有
  帧数（L）、逐帧来源（frame = k、time_ticks = frame_times(tid)[k]、time_seconds = time_ticks·0.25）、
  下达动作（TrajectoryControls 或 SubtickControls，与 systems.commanded_controls 逐位相同）、多一个 timing 记录。
  当前数据集必须是带时间采样调节项、horizon = 32、不带空间采样的数据集。
- full_controls(m, device)：整条下达动作转成 ActionSegments（cache._window_controls 的写法；1/4 tick 的边界是二进有理数，
  换算成秒、再换回周期都是精确的）。
- read_window(path, start_frame=i, frames=n, …)：起点第 i 帧、目标第 i+n 帧，时长 T = t_{i+n} − t_i（tick，float64），
  动作裁到 [t_i, t_{i+n})。返回 cache.CacheWindow（字段、精度、设备和 cache.read_window 相同；起点物理秒 = t_i·0.25，
  目标物理秒 = 起点 + T·0.25，与 data._window_boundary 的精确等式一致）。

时间采样的缓存只用来评测（runtime/timing_eval.py），训练和普通窗口（cache.read_window）都不收它。
"""

from pathlib import Path

import json

import numpy as np

import torch

from po_jepa import datasets, vjepa
from po_jepa.actions import ControlSpec, from_physical_segments, clip_action_segments
from . import cache

_VERIFIED = {}
SUBTICKS = 4


def _timing_spec():
    from po_jepa import systems
    spec = datasets.active()
    if not spec.knobs.changes_timing():
        raise ValueError(f"{spec.tag} has no time-sampling knob (fps/jitter/exposure/switch_offset): use cache.read_window")
    if spec.knobs.horizon != systems.TICKS or spec.knobs.changes_space():
        raise ValueError(f"{spec.tag}: time-sampling caches are wired only for horizon=32 without space sampling")
    return spec


def frame_count(spec=None) -> int:
    spec = datasets.active() if spec is None else spec
    return spec.knobs.horizon*spec.knobs.fps//4+1


def read_complete(path):
    """→ (raw [L,256,1408] memmap, manifest, manifest_sha)。"""
    spec = _timing_spec()
    path = Path(path).resolve(strict=True)
    if (path/"failure.json").exists():
        raise ValueError("cache has retained failure marker")
    key = (str(path), spec.tag, cache.PAPER_SHA, cache.PROTOCOL_SHA, cache.CONFIG_SHA, cache.CHECKPOINT_SHA,
           *((s.st_mtime_ns, s.st_size) for s in ((path/n).stat() for n in ("COMPLETE.json", "manifest.json", "raw_z.npy"))))
    if key not in _VERIFIED:
        _VERIFIED[key] = _read_uncached(path, spec)
    return _VERIFIED[key]


def _read_uncached(path, spec):
    for name in ("COMPLETE.json", "manifest.json", "raw_z.npy"):
        if (path/name).resolve().parent != path:
            raise ValueError("cache child path escapes trajectory directory")
    if (path/"failure.json").exists():
        raise ValueError("cache has retained failure marker")
    complete = json.loads((path/"COMPLETE.json").read_bytes())
    manifest_sha = cache.file_sha(path/"manifest.json")
    if complete.get("manifest_sha256") != manifest_sha:
        raise ValueError("manifest SHA mismatch")
    m = json.loads((path/"manifest.json").read_bytes())
    cache.verify_fixed_protocol()
    if m.get("schema_version") != cache.V7_SCHEMA:
        raise ValueError("time-sampling cache must be a v7 (encode_v7.py) cache")
    return check_manifest(path, m, manifest_sha, spec)


def _controls(m):
    """manifest 的下达动作 → systems.TrajectoryControls 或 SubtickControls（构造时做各自的检查）。"""
    from po_jepa import simulation, systems
    c = m["controls"]
    values = np.array(c["values"], dtype=np.float64)
    if "starts_subticks" in c or "ends_subticks" in c:
        if "starts_ticks" in c or "ends_ticks" in c:
            raise ValueError("controls record mixes tick and subtick boundaries")
        return systems.SubtickControls(m["trajectory_id"], np.array(c["starts_subticks"], dtype=np.int64),
                                       np.array(c["ends_subticks"], dtype=np.int64), values)
    out = simulation.TrajectoryControls(m["trajectory_id"], np.array(c["starts_ticks"], dtype=np.int64),
                                        np.array(c["ends_ticks"], dtype=np.int64), values)
    out.__post_init__()
    return out


def check_manifest(path, m, manifest_sha, spec):
    """cache._read_v7_cache 的时间采样版：帧数 L、逐帧实际时刻、两种下达动作。"""
    from po_jepa import systems, timing
    frames = frame_count(spec)
    if (m.get("status") != "complete" or m.get("feature_file") != "raw_z.npy"
            or m.get("system") != spec.system or m.get("dataset_count") != spec.count or m.get("variant", "") != spec.variant
            or m.get("shape") != [frames, 256, 1408] or m.get("dtype") not in cache.V7_DTYPES
            or m.get("grid") != [16, 16] or m.get("token_order") != "row_major"
            or m.get("split") != spec.split_for(m["trajectory_id"]) or m.get("observation_variant") != "canonical"
            or m.get("preprocess_sha256") != cache.v7_preprocess_sha256()):
        raise ValueError("time-sampling cache schema/system/variant/count/split mismatch (check --system/--variant/--trajectories)")
    v = spec.knobs
    if m.get("timing") != dict(fps=v.fps, jitter=v.jitter, exposure=v.exposure, switch_offset=v.switch_offset):
        raise ValueError("time-sampling cache timing record differs from the active dataset's knobs")
    g = m["generation"]
    if (g.get("evidence_role") != "real_frozen_encoder" or g.get("encoder_checkpoint_sha256") != cache.CHECKPOINT_SHA
            or g.get("upstream_commit") != vjepa.UPSTREAM_COMMIT or g.get("preprocess") != cache._v7_preprocess_record()):
        raise ValueError("time-sampling cache encoder/preprocess provenance mismatch")
    report = g["encoder_load_report"]
    if (report["sha256"] != cache.CHECKPOINT_SHA or report["encoder_key"] != "target_encoder" or report["strict"] is not True
            or report["missing_keys"] or report["unexpected_keys"]):
        raise ValueError("invalid real encoder provenance")
    if cache.file_sha(path/"raw_z.npy") != m["feature_sha256"]:
        raise ValueError("feature SHA mismatch")
    raw = np.load(path/"raw_z.npy", allow_pickle=False, mmap_mode="r")
    if raw.dtype != cache.V7_DTYPES[m["dtype"]] or raw.shape != (frames, 256, 1408) or not np.isfinite(raw).all():
        raise ValueError("invalid actual raw Z array")
    expected = systems.commanded_controls(m["trajectory_id"]); stored = _controls(m)
    if type(stored) is not type(expected) or any(
            not np.array_equal(getattr(stored, k), getattr(expected, k)) for k in vars(expected) if k != "trajectory_id"):
        raise ValueError("time-sampling cache controls differ from the registered random stream of this system")
    if m["controls"].get("units") != "simulation_position_per_second":
        raise ValueError("unknown control units")
    ticks = timing.frame_times(m["trajectory_id"])
    if len(m["frames"]) != frames or ticks.shape != (frames,):
        raise ValueError("incomplete endpoint provenance")
    for k, f in enumerate(m["frames"]):
        if (f.get("frame") != k or type(f.get("time_ticks")) is not float or f["time_ticks"] != float(ticks[k])
                or f.get("time_seconds") != f["time_ticks"]*.25 or type(f.get("image_sha256")) is not str
                or len(f["image_sha256"]) != 64):
            raise ValueError("endpoint source/time mismatch")
    return raw, m, manifest_sha


def frame_ticks(m) -> np.ndarray:
    """manifest 里每帧的实际时刻（tick，float64；已由 check_manifest 核过 = timing.frame_times）。"""
    return np.array([f["time_ticks"] for f in m["frames"]], dtype=np.float64)


def full_controls(m, device):
    """整条下达动作 → ActionSegments（周期 = tick，原点 0）。SubtickControls 的边界 = subticks/4（精确）。"""
    from po_jepa import systems
    c = _controls(m)
    if isinstance(c, systems.SubtickControls):
        starts, ends = c.starts_subticks/SUBTICKS, c.ends_subticks/SUBTICKS
    else:
        starts, ends = c.starts_ticks.astype(np.float64), c.ends_ticks.astype(np.float64)
    if ends[-1] != datasets.active().knobs.horizon:
        raise ValueError("controls do not cover the active dataset's horizon")
    spec = ControlSpec(2, .25, ("u_x", "u_y"), ("simulation_position_per_second",)*2,
                       (-.5, -.5), (.5, .5), (1., 1.), (0., 0.), "reference/protocol.md")
    n = len(starts)
    return from_physical_segments(starts_seconds=torch.tensor(starts[None]*.25, dtype=torch.float64, device="cpu"),
        ends_seconds=torch.tensor(ends[None]*.25, dtype=torch.float64, device="cpu"),
        u_values=torch.tensor(c.values[None], dtype=torch.float32, device=device),
        n_segments=torch.tensor([n], dtype=torch.int64, device="cpu"), segment_valid=torch.ones((1, n), dtype=torch.bool, device="cpu"),
        origin_seconds=torch.zeros(1, dtype=torch.float64, device="cpu"), control_spec=spec,
        source_ids=(f"trajectory-{m['trajectory_id']:04d}",))


def read_window(path, *, start_frame, frames, expected_split, device):
    """起点第 start_frame 帧、目标第 start_frame+frames 帧的窗口（cache.CacheWindow）。"""
    if type(start_frame) is not int or start_frame < 0:
        raise ValueError("time-sampling window start must be a nonnegative integer frame index")
    if type(frames) is not int or frames < 1:
        raise ValueError("time-sampling window must span a positive integer number of frames")
    raw, m, manifest_sha = read_complete(Path(path))
    end = start_frame+frames
    if end > raw.shape[0]-1:
        raise ValueError("time-sampling window runs past the last cached frame")
    if m["split"] != expected_split:
        raise ValueError("cross-split cache/window request")
    ticks = frame_ticks(m)
    t0, t1 = float(ticks[start_frame]), float(ticks[end])
    duration_ticks = t1-t0
    if not duration_ticks > 0:
        raise ValueError("frame times must increase")
    with torch.inference_mode(False), torch.no_grad():
        z = torch.from_numpy(np.array(raw[start_frame:start_frame+1], dtype=np.float32, copy=True)).to(device).clone()
        target = torch.from_numpy(np.array(raw[end:end+1], dtype=np.float32, copy=True)).to(device).clone()
        duration = torch.tensor([duration_ticks], dtype=torch.float64, device="cpu")
        observed = torch.tensor([t0*.25], dtype=torch.float64, device="cpu")
        controls = clip_action_segments(full_controls(m, device), start=torch.tensor([t0], dtype=torch.float64),
                                        end=torch.tensor([t1], dtype=torch.float64))
        if not torch.equal(controls.duration, duration):
            raise ValueError("clipped controls do not span the window duration exactly")
        inputs = cache.WindowInput(z, controls, duration, cache.regular_grid(16, 16), observed,
                                   m["frames"][start_frame]["image_sha256"], m["trajectory_id"], m["split"])
        label = cache.WindowTarget(target, observed+duration*.25, m["frames"][end]["image_sha256"])
    provenance = dict(actual_manifest_sha256=manifest_sha, generation=m["generation"], preprocess_sha256=m["preprocess_sha256"],
                      observation_variant=m["observation_variant"], start_frame=start_frame, end_frame=end,
                      start_tick=t0, end_tick=t1)
    return cache.CacheWindow(inputs, label, provenance)
