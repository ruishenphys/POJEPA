"""F03 的虚构 CPU 接口测试；唯一真实单卡验收显式 opt-in，跳过不等于通过。"""

from dataclasses import replace

import hashlib

import copy

import importlib

import json

import os

from pathlib import Path

from types import SimpleNamespace

import pytest

import torch

def unit_windows(data):
    """虚构四条路径，E/N与真实接口一致；generation明确synthetic，不能过真实入口。"""
    from po_jepa.actions import ActionSegments, ControlSpec
    from po_jepa.geometry import regular_grid
    spec = ControlSpec(2, .25, ("u_x", "u_y"), ("position/s", "position/s"),
                       (-.5, -.5), (.5, .5), (1., 1.), (0., 0.), "F03 fictional unit fixture")
    windows = []
    for tid, (start, duration) in enumerate(zip((0, 3, 5, 20), (1, 2, 4, 8))):
        duration_t = torch.tensor([duration], dtype=torch.float64)
        origin = torch.tensor([start * .25], dtype=torch.float64)
        starts = torch.arange(duration, dtype=torch.float64)[None, :]
        controls = ActionSegments(starts, starts + 1, torch.full((1, duration, 2), .1),
            torch.tensor([duration]), torch.ones((1, duration), dtype=torch.bool), origin,
            duration_t, spec, (f"fictional-control-{tid}",))
        z = torch.full((1, 256, 1408), float(tid + 1))
        sha = hashlib.sha256(f"fictional-only-{tid}".encode()).hexdigest()
        x = data.cache.WindowInput(z, controls, duration_t, regular_grid(16, 16), origin, sha, tid, "train")
        y = data.cache.WindowTarget(z + .1, origin + duration_t * .25, sha)
        windows.append(data.cache.CacheWindow(x, y, dict(actual_manifest_sha256=sha,
            preprocess_sha256=sha, observation_variant="canonical",
            generation={"evidence_role": "synthetic_development_only", "note": "虚构测试替身"})))
    return windows

