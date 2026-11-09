"""S25c：长时程测试集（horizon = 64、128，tools/encode_v7.py --splits test 写出的 65、129 帧缓存）的读取。

cache.py 一个字节不改（S18a 钉了它的 SHA：33 帧缓存的读法和 v7 逐字相同，D4 的等价核对靠它）。cache._read_v7_cache
写死 33 帧、cache._window_controls 写死 TrajectoryControls（覆盖 [0, 32)），所以这里是两者的长时程版：

- read_complete(path)：核法逐条照抄 cache._read_complete_cache_uncached 和 cache._read_v7_cache，只有帧数换成
  当前数据集的 horizon + 1（manifest 的 shape、实际数组、逐帧来源三处）；当前数据集必须是长时程数据集。
  同一进程内按 (路径, 数据集, 固定来源 SHA, 三个文件的 mtime/大小) 复用已核过的结果（cache 的键里没有数据集）。
- window_controls(m, start, T, device)：照抄 cache._window_controls，下达动作用 systems.LongControls 核（覆盖 [0, horizon)）。

长时程缓存只用来评测（runtime/long_rollout.py），训练和普通窗口（cache.read_window）都不收它：cache 的核法见到 65/129
帧直接报错。
"""

from pathlib import Path

import json

import numpy as np

import torch

from po_jepa import datasets, vjepa
from po_jepa.actions import ControlSpec, from_physical_segments, clip_action_segments
from . import cache

_VERIFIED = {}


def _long_spec():
    from po_jepa import systems
    spec = datasets.active()
    if spec.knobs.horizon == systems.TICKS:
        raise ValueError(f"{spec.tag} is not a long-horizon dataset (horizon=32): use cache.read_window")
    return spec


def read_complete(path):
    """→ (raw [H+1,256,1408] memmap, manifest, manifest_sha)。"""
    spec = _long_spec()
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
        raise ValueError("long-horizon cache must be a v7 (encode_v7.py) cache")
    return check_manifest(path, m, manifest_sha, spec)


def check_manifest(path, m, manifest_sha, spec):
    """cache._read_v7_cache 的长时程版：帧数 = horizon + 1。"""
    frames = spec.knobs.horizon+1
    if (m.get("status") != "complete" or m.get("feature_file") != "raw_z.npy"
            or m.get("system") != spec.system or m.get("dataset_count") != spec.count or m.get("variant", "") != spec.variant
            or m.get("shape") != [frames, 256, 1408] or m.get("dtype") not in cache.V7_DTYPES
            or m.get("grid") != [16, 16] or m.get("token_order") != "row_major"
            or m.get("split") != spec.split_for(m["trajectory_id"]) or m.get("observation_variant") != "canonical"
            or m.get("preprocess_sha256") != cache.v7_preprocess_sha256()):
        raise ValueError("long-horizon cache schema/system/variant/count/split mismatch (check --system/--variant/--trajectories)")
    g = m["generation"]
    if (g.get("evidence_role") != "real_frozen_encoder" or g.get("encoder_checkpoint_sha256") != cache.CHECKPOINT_SHA
            or g.get("upstream_commit") != vjepa.UPSTREAM_COMMIT or g.get("preprocess") != cache._v7_preprocess_record()):
        raise ValueError("long-horizon cache encoder/preprocess provenance mismatch")
    report = g["encoder_load_report"]
    if (report["sha256"] != cache.CHECKPOINT_SHA or report["encoder_key"] != "target_encoder" or report["strict"] is not True
            or report["missing_keys"] or report["unexpected_keys"]):
        raise ValueError("invalid real encoder provenance")
    if cache.file_sha(path/"raw_z.npy") != m["feature_sha256"]:
        raise ValueError("feature SHA mismatch")
    raw = np.load(path/"raw_z.npy", allow_pickle=False, mmap_mode="r")
    if raw.dtype != cache.V7_DTYPES[m["dtype"]] or raw.shape != (frames, 256, 1408) or not np.isfinite(raw).all():
        raise ValueError("invalid actual raw Z array")
    from po_jepa import systems
    expected = systems.commanded_controls(m["trajectory_id"]); c = m["controls"]
    if any(not np.array_equal(np.asarray(c[key]), value) for key, value in
           (("starts_ticks", expected.starts_ticks), ("ends_ticks", expected.ends_ticks), ("values", expected.values))):
        raise ValueError("long-horizon cache controls differ from the registered random stream of this system")
    if len(m["frames"]) != frames:
        raise ValueError("incomplete endpoint provenance")
    for tick, f in enumerate(m["frames"]):
        if (f.get("tick") != tick or f.get("time_seconds") != tick*.25 or type(f.get("image_sha256")) is not str
                or len(f["image_sha256"]) != 64):
            raise ValueError("endpoint source mismatch")
    return raw, m, manifest_sha


def window_controls(m, start_tick, total, device):
    """cache._window_controls 的长时程版：下达动作按 systems.LongControls 核，再转换、裁到 [start, start+total)。"""
    from po_jepa import systems
    data = m["controls"]
    c = systems.LongControls(m["trajectory_id"], np.array(data["starts_ticks"], dtype=np.int64),
                             np.array(data["ends_ticks"], dtype=np.int64), np.array(data["values"], dtype=np.float64))
    if c.horizon != datasets.active().knobs.horizon:
        raise ValueError("long-horizon controls do not cover the active dataset's horizon")
    if data["units"] != "simulation_position_per_second":
        raise ValueError("unknown control units")
    spec = ControlSpec(2, .25, ("u_x", "u_y"), ("simulation_position_per_second",)*2,
                       (-.5, -.5), (.5, .5), (1., 1.), (0., 0.), "reference/protocol.md")
    n = len(c.starts_ticks)
    full = from_physical_segments(starts_seconds=torch.tensor(c.starts_ticks[None]*.25, dtype=torch.float64, device="cpu"),
        ends_seconds=torch.tensor(c.ends_ticks[None]*.25, dtype=torch.float64, device="cpu"),
        u_values=torch.tensor(c.values[None], dtype=torch.float32, device=device),
        n_segments=torch.tensor([n], dtype=torch.int64, device="cpu"), segment_valid=torch.ones((1, n), dtype=torch.bool, device="cpu"),
        origin_seconds=torch.zeros(1, dtype=torch.float64, device="cpu"), control_spec=spec,
        source_ids=(f"trajectory-{m['trajectory_id']:04d}",))
    return clip_action_segments(full, start=torch.tensor([float(start_tick)], dtype=torch.float64, device="cpu"),
                                end=torch.tensor([start_tick+total], dtype=torch.float64, device="cpu"))
