"""S25e：空间采样测试集（调节项 resolution、zoom、pan、drift，S24c）的缓存读取和取窗口。

缓存由 tools/encode_v7.py --splits test 写出：每条 33 帧（每 tick 一帧，同主实验），画面是相机里原生渲染的 R×R
（R = resolution），按原生分辨率编码（po_jepa.native_encoder），特征 [33, G², 1408]，G = R/16（12/16/20/24），
token 行优先。manifest 比 v7 多一个 space 记录；预处理记录是原生分辨率版（crop_size = R，native_resolution = R），
所以 preprocess_sha256 与 v7 的不同，cache.read_window（核 v7_preprocess_sha256）不收空间采样缓存，zoom/pan/drift 这些
仍是 256、16×16 的缓存也不收。

cache.py 一个字节不改（S18a 钉了它的 SHA）。这里是 cache._read_complete_cache_uncached + cache._read_v7_cache 的空间采样版
（照 S25c long_cache、S25d timing_cache 的写法）：

- read_complete(path)：核法逐条照抄；不同的只有特征形状 [33, G², 1408]、grid [G, G]、预处理记录和摘要（原生分辨率版）、
  多一个 space 记录。下达动作、逐帧来源（tick、time_seconds、image_sha256）与 v7 相同（空间采样不改物理和动作）。
  当前数据集必须带空间采样调节项、horizon = 32、不带时间采样。
- read_window(path, start_tick=s, duration_cycles=T, …)：与 cache.read_window 同一组起点（0…20 的整数 tick）和时长
  （1、2、3、4、8、12），动作用 cache._window_controls（同一个函数），grid = regular_grid(G, G)。
- window_boundary(window)：data._window_boundary 的 G×G 版（那个写死 16×16）。

空间采样的缓存只用来评测（runtime/space_eval.py），训练、cache.read_window、run.py transfer 都不收它。
"""

import hashlib
from pathlib import Path

import json

import numpy as np

import torch

from po_jepa import datasets, native_encoder, vjepa
from . import cache

_VERIFIED = {}
FRAMES = 33
START_TICKS = range(0, 21)
DURATIONS = (1., 2., 3., 4., 8., 12.)
PREPROCESS_RULE = "reset_same_trajectory_generator_state_before_each_single_frame"


def _space_spec():
    from po_jepa import systems
    spec = datasets.active()
    if not spec.knobs.changes_space():
        raise ValueError(f"{spec.tag} has no space-sampling knob (resolution/zoom/pan/drift): use cache.read_window")
    if spec.knobs.horizon != systems.TICKS or spec.knobs.changes_timing():
        raise ValueError(f"{spec.tag}: space-sampling caches are wired only for horizon=32 without time sampling")
    return spec


def side(spec=None) -> int:
    """特征网格边长 G = resolution/16。"""
    spec = datasets.active() if spec is None else spec
    return native_encoder.grid_side(spec.knobs.resolution)


def space_record(spec) -> dict:
    v = spec.knobs
    return dict(resolution=v.resolution, zoom=v.zoom, pan=v.pan, drift=v.drift)


def preprocess_record(resolution: int) -> dict:
    """写进空间采样 manifest 的预处理描述：v7 的固定 DROID 变换，输出边长换成原生分辨率 R（native_encoder.preprocess_native）。"""
    native_encoder.grid_side(resolution)
    return dict(cache._v7_preprocess_record(), crop_size=resolution, native_resolution=resolution)


def preprocess_sha256(resolution: int) -> str:
    """v7_preprocess_sha256 的原生分辨率版（记录里多 native_resolution，所以即使 R = 256 也和 v7 的摘要不同）。"""
    return hashlib.sha256(cache._json_bytes(dict(spec=preprocess_record(resolution), rule=PREPROCESS_RULE))).hexdigest()


def read_complete(path):
    """→ (raw [33,G²,1408] memmap, manifest, manifest_sha)。"""
    spec = _space_spec()
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
        raise ValueError("space-sampling cache must be a v7 (encode_v7.py) cache")
    return check_manifest(path, m, manifest_sha, spec)


def check_manifest(path, m, manifest_sha, spec):
    """cache._read_v7_cache 的空间采样版：特征形状 [33,G²,1408]、grid [G,G]、原生分辨率预处理、space 记录。"""
    from po_jepa import systems
    g_side = side(spec)
    tokens = g_side*g_side
    if (m.get("status") != "complete" or m.get("feature_file") != "raw_z.npy"
            or m.get("system") != spec.system or m.get("dataset_count") != spec.count or m.get("variant", "") != spec.variant
            or m.get("shape") != [FRAMES, tokens, 1408] or m.get("dtype") not in cache.V7_DTYPES
            or m.get("grid") != [g_side, g_side] or m.get("token_order") != "row_major"
            or m.get("split") != spec.split_for(m["trajectory_id"]) or m.get("observation_variant") != "canonical"
            or m.get("preprocess_sha256") != preprocess_sha256(spec.knobs.resolution)):
        raise ValueError("space-sampling cache schema/system/variant/count/split/grid mismatch (check --system/--variant/--trajectories)")
    if m.get("space") != space_record(spec):
        raise ValueError("space-sampling cache space record differs from the active dataset's knobs")
    g = m["generation"]
    if (g.get("evidence_role") != "real_frozen_encoder" or g.get("encoder_checkpoint_sha256") != cache.CHECKPOINT_SHA
            or g.get("upstream_commit") != vjepa.UPSTREAM_COMMIT or g.get("preprocess") != preprocess_record(spec.knobs.resolution)):
        raise ValueError("space-sampling cache encoder/preprocess provenance mismatch")
    report = g["encoder_load_report"]
    if (report["sha256"] != cache.CHECKPOINT_SHA or report["encoder_key"] != "target_encoder" or report["strict"] is not True
            or report["missing_keys"] or report["unexpected_keys"]):
        raise ValueError("invalid real encoder provenance")
    if cache.file_sha(path/"raw_z.npy") != m["feature_sha256"]:
        raise ValueError("feature SHA mismatch")
    raw = np.load(path/"raw_z.npy", allow_pickle=False, mmap_mode="r")
    if raw.dtype != cache.V7_DTYPES[m["dtype"]] or raw.shape != (FRAMES, tokens, 1408) or not np.isfinite(raw).all():
        raise ValueError("invalid actual raw Z array")
    expected = systems.commanded_controls(m["trajectory_id"]); c = m["controls"]
    if any(not np.array_equal(np.asarray(c[key]), value) for key, value in
           (("starts_ticks", expected.starts_ticks), ("ends_ticks", expected.ends_ticks), ("values", expected.values))):
        raise ValueError("space-sampling cache controls differ from the registered random stream of this system")
    if len(m["frames"]) != FRAMES:
        raise ValueError("incomplete endpoint provenance")
    for tick, f in enumerate(m["frames"]):
        if (f.get("tick") != tick or f.get("time_seconds") != tick*.25 or type(f.get("image_sha256")) is not str
                or len(f["image_sha256"]) != 64):
            raise ValueError("endpoint source mismatch")
    return raw, m, manifest_sha


def read_window(path, *, start_tick, duration_cycles, expected_split, device):
    """cache.read_window 的空间采样版：起点 tick s（0…20 的整数）、时长 T ∈ {1,2,3,4,8,12}，grid = G×G。"""
    if type(start_tick) is not int or start_tick not in START_TICKS:
        raise ValueError("F01 start must be integer tick in [0,20]")
    if type(duration_cycles) not in (int, float) or duration_cycles not in DURATIONS:
        raise ValueError("supervised endpoint must use an allowed integer duration")
    raw, m, manifest_sha = read_complete(Path(path))
    end = int(start_tick+duration_cycles)
    if m["split"] != expected_split:
        raise ValueError("cross-split cache/window request")
    g_side = m["grid"][0]
    with torch.inference_mode(False), torch.no_grad():
        z = torch.from_numpy(np.array(raw[start_tick:start_tick+1], dtype=np.float32, copy=True)).to(device).clone()
        target = torch.from_numpy(np.array(raw[end:end+1], dtype=np.float32, copy=True)).to(device).clone()
        duration = torch.tensor([float(duration_cycles)], dtype=torch.float64, device="cpu")
        observed = torch.tensor([start_tick*.25], dtype=torch.float64, device="cpu")
        controls = cache._window_controls(m, start_tick, float(duration_cycles), device)
        inputs = cache.WindowInput(z, controls, duration, cache.regular_grid(g_side, g_side), observed,
                                   m["frames"][start_tick]["image_sha256"], m["trajectory_id"], m["split"])
        label = cache.WindowTarget(target, torch.tensor([end*.25], dtype=torch.float64, device="cpu"), m["frames"][end]["image_sha256"])
    provenance = dict(actual_manifest_sha256=manifest_sha, generation=m["generation"], preprocess_sha256=m["preprocess_sha256"],
                      observation_variant=m["observation_variant"], space=m["space"])
    return cache.CacheWindow(inputs, label, provenance)


def window_boundary(window) -> None:
    """data._window_boundary 的 G×G 版：网格、形状、精度、有限性、时刻精确相等。"""
    x, y = window.inputs, window.target
    x.grid.__post_init__()
    g_side = side()
    tokens = g_side*g_side
    if (x.grid.height, x.grid.width) != (g_side, g_side) or not torch.equal(x.grid.coords, cache.regular_grid(g_side, g_side).coords):
        raise ValueError("space-sampling rows require the regular G×G grid of the active resolution")
    for value in (x.raw_z, y.raw_z):
        if (value.shape != (1, tokens, 1408) or value.dtype != torch.float32 or value.device != x.raw_z.device
                or torch.is_inference(value) or value.requires_grad or value.grad_fn is not None or not torch.isfinite(value).all()):
            raise ValueError("frozen raw rows must be ordinary finite model float32[1,G²,1408]")
    for value in (x.duration, x.observed_time_seconds, y.target_time_seconds):
        if (value.shape != (1,) or value.device.type != "cpu" or value.dtype != torch.float64
                or value.requires_grad or torch.is_inference(value) or not torch.isfinite(value).all()):
            raise ValueError("query times must be ordinary finite CPU float64[1]")
    expected = x.observed_time_seconds+x.duration*x.controls.control_spec.control_period_seconds
    if not torch.equal(expected, y.target_time_seconds) or not torch.equal(x.observed_time_seconds, x.controls.source_origin_seconds):
        raise ValueError("physical endpoint/origin must exactly match query cycles")
