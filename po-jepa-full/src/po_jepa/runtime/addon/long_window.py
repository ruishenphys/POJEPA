"""10 §3.7：D4 长时程窗口（T = 16、24、32）的读取。

和 cache.read_window 相同，只有时长和起点的检查不同：时长只收 16、24、32 个周期，起点是非负整数 tick，
并要求 start_tick + T 不超过缓存的最后一帧（仿真轨迹 33 帧，最后一帧是第 32 个 tick）。读缓存、转 float32、
裁动作、组窗口的写法逐行照抄 cache.read_window，复用 cache._read_complete_cache 和 cache._window_controls；
cache.read_window 本身不改。

S25c：长时程测试集（horizon=64/128 的缓存，65、129 帧）上的滚动评测（runtime/long_rollout.py）用同一个读法：
时长换成 allowed（不给时仍是 D4 的 LONG_TIMES）；当前数据集是长时程数据集时，读缓存和裁动作换成 runtime/long_cache.py
的 read_complete、window_controls（帧数 horizon + 1、动作是 LongControls；cache.py 不改）。horizon = 32 时 D4 的行为不变。
"""

from pathlib import Path

import numpy as np

import torch

from po_jepa import datasets
from .. import cache, long_cache

LONG_TIMES = (16., 24., 32.)


def read_long_window(path, *, start_tick, duration_cycles, expected_split, device, allowed=None):
    """读明确 ID 缓存的起点帧和第 start_tick+T 帧，返回 cache.CacheWindow（字段、精度、设备和 read_window 相同）。
    allowed：允许的时长（默认 LONG_TIMES，调用时再取，D4 不变）。"""
    times = LONG_TIMES if allowed is None else tuple(allowed)
    if type(start_tick) is not int or start_tick < 0:
        raise ValueError("long-horizon start must be a nonnegative integer tick")
    if type(duration_cycles) not in (int, float) or duration_cycles not in times:
        raise ValueError("long-horizon duration must be 16, 24 or 32 cycles" if allowed is None
                         else f"long-horizon duration must be one of {times} cycles")
    long = datasets.active().knobs.horizon != 32                         # S25c：长时程测试集的缓存
    raw, m, manifest_sha = (long_cache.read_complete if long else cache._read_complete_cache)(Path(path))
    end = int(start_tick+duration_cycles)
    if end > raw.shape[0]-1:
        raise ValueError("long-horizon window runs past the last cached frame")
    if m["split"] != expected_split:
        raise ValueError("cross-split cache/window request")
    with torch.inference_mode(False), torch.no_grad():
        z = torch.from_numpy(np.array(raw[start_tick:start_tick+1], dtype=np.float32, copy=True)).to(device).clone()
        target = torch.from_numpy(np.array(raw[end:end+1], dtype=np.float32, copy=True)).to(device).clone()
        duration = torch.tensor([float(duration_cycles)], dtype=torch.float64, device="cpu")
        observed = torch.tensor([start_tick*.25], dtype=torch.float64, device="cpu")
        controls = (long_cache.window_controls if long else cache._window_controls)(m, start_tick, float(duration_cycles), device)
        inputs = cache.WindowInput(z, controls, duration, cache.regular_grid(16, 16), observed,
            m["frames"][start_tick]["image_sha256"], m["trajectory_id"], m["split"])
        label = cache.WindowTarget(target, torch.tensor([end*.25], dtype=torch.float64, device="cpu"), m["frames"][end]["image_sha256"])
    provenance = dict(actual_manifest_sha256=manifest_sha, generation=m["generation"], preprocess_sha256=m["preprocess_sha256"],
        observation_variant=m["observation_variant"])
    return cache.CacheWindow(inputs, label, provenance)
