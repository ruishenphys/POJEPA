"""DROID 7 维动作和 ControlSpec（docs/design/01-droid-config.md §5、§6）。

`poses_to_diffs` 的函数体逐字取自 vjepa2@204698b `app/vjepa_droid/droid.py`（上游是 `DROIDVideoDataset` 的方法，
这里去掉 self 写成模块函数，函数体按 AST 和上游相同，见 tests/test_droid_actions.py）；上游 MIT 许可证在
docs/design/droid-golden/LICENSE-vjepa2。模型里的动作值 = 物理值 / 尺度（v7 `from_physical_segments` 的规则）。
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.spatial.transform import Rotation

from po_jepa.actions import ControlSpec
from po_jepa.runtime import cache

from .config import ACTION_SOURCES, DroidConfig

CHANNEL_NAMES = ("dx", "dy", "dz", "rx", "ry", "rz", "grip")
CHANNEL_UNITS = ("m", "m", "m", "rad", "rad", "rad", "1")
SCALE_GROUPS = ("position", "rotation", "gripper")   # action_scale.json 里每个动作源一个 {组: 尺度}（01 §5.2）
GROUP_WIDTHS = (3, 3, 1)


def poses_to_diffs(poses):            # 逐字取自 vjepa2@204698b app/vjepa_droid/droid.py
    xyz = poses[:, :3]
    thetas = poses[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    xyz_diff = xyz[1:] - xyz[:-1]
    angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    angle_diff = np.stack([d for d in angle_diff], axis=0)
    closedness = poses[:, -1:]
    closedness_delta = closedness[1:] - closedness[:-1]
    return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)


def droid_control_spec(scale: dict, *, source: str, config: DroidConfig) -> ControlSpec:
    """01 §5.3。`scale` 是 `prepared/action_scale.json` 的内容，`scale[source]` 给三组尺度；界取自 `config.action`。"""
    if source not in ACTION_SOURCES:
        raise ValueError(f"source must be one of {ACTION_SOURCES}")
    if not isinstance(scale, dict) or not isinstance(scale.get(source), dict):
        raise ValueError(f"action scale must contain a {source!r} mapping")
    groups = scale[source]
    if set(groups) != set(SCALE_GROUPS):
        raise ValueError(f"action scale {source!r} must give exactly {SCALE_GROUPS}")
    values = []
    for group in SCALE_GROUPS:
        value = groups[group]
        if type(value) is not float or not math.isfinite(value) or value <= 0:
            raise ValueError(f"action scale {source}.{group} must be a positive finite float")
        values.append(value)
    a = config.action
    bounds = (a.position_bound_m, a.rotation_bound_rad, a.gripper_bound)
    sha = hashlib.sha256(cache._json_bytes(scale)).hexdigest()   # 等于 _write_json 写出的 action_scale.json 文件的 SHA
    return ControlSpec(
        action_dim=7,
        control_period_seconds=config.time.nominal_interval_seconds,
        channel_names=CHANNEL_NAMES,
        channel_units=CHANNEL_UNITS,
        lower_bounds=_per_channel(tuple(-b for b in bounds)),
        upper_bounds=_per_channel(bounds),
        scale=_per_channel(tuple(values)),
        offset=(0.0,) * 7,
        definition_source=(f"DROID {config.data.version} {source} pose diffs, "
                           f"vjepa2@{config.encode.upstream_commit[:7]} poses_to_diffs, "
                           f"{config.time.frames_per_interval} frames per interval, action_scale.json sha256={sha}"))


def _per_channel(group_values: tuple[float, float, float]) -> tuple[float, ...]:
    return tuple(float(v) for v, width in zip(group_values, GROUP_WIDTHS) for _ in range(width))
