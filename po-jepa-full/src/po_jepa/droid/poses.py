"""DROID 状态积分 `compute_new_pose`（docs/design/09-droid-training.md §6.2；S06 核对）。

函数体逐字取自 vjepa2@204698b `notebooks/utils/mpc_utils.py`（副本 docs/design/droid-golden/official_mpc_utils_204698b.py，
上游 MIT 许可证在同目录 LICENSE-vjepa2）：位置相加，旋转 R_new = R(Δ)·R，夹爪相加后截到 [0, 1]。
输入输出是 torch 张量 [B, 1, 7]（物理单位）；AC 的 `true_state` 模式在 CPU float64 上调用它（S08b）。
tests/test_droid_poses.py 核函数体的 AST 和上游相同，并在 5 条金标准回合上和官方输出逐位相同。
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def compute_new_pose(pose, action):   # 逐字取自 vjepa2@204698b notebooks/utils/mpc_utils.py
    """
    :param pose: [B, T=1, 7]
    :param action: [B, T=1, 7]
    :returns: [B, T=1, 7]
    """
    device, dtype = pose.device, pose.dtype
    pose = pose[:, 0].cpu().numpy()
    action = action[:, 0].cpu().numpy()
    # -- compute delta xyz
    new_xyz = pose[:, :3] + action[:, :3]
    # -- compute delta theta
    thetas = pose[:, 3:6]
    delta_thetas = action[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas]
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    new_angle = np.stack([d for d in angle_diff], axis=0)  # [B, 7]
    # -- compute delta gripper
    new_closedness = pose[:, -1:] + action[:, -1:]
    new_closedness = np.clip(new_closedness, 0, 1)
    # -- new pose
    new_pose = np.concatenate([new_xyz, new_angle, new_closedness], axis=-1)
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]
