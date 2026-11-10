"""S06a：`compute_new_pose`（docs/design/09-droid-training.md §6.2；验收 14 §3 S06）和官方函数逐位相同（同一环境执行上游原函数），
和存档金标准在跨平台容差内相同（S06a-fix1）。"""

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from po_jepa.droid import poses

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT/"docs"/"design"/"droid-golden"
OFFICIAL = GOLDEN/"official_mpc_utils_204698b.py"
# 上游 git 里 notebooks/utils/mpc_utils.py 的原始字节（LF），raw.githubusercontent.com/…/204698b…
OFFICIAL_LF_SHA = "79f4b753e0faf4ea5a7531dd1f2f1b2e160d8d6907c34b4c1914aeef09f28341"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1
    return found[0]


def test_vendored_upstream_file_is_pinned():
    assert hashlib.sha256(OFFICIAL.read_bytes()).hexdigest() == OFFICIAL_LF_SHA


def test_compute_new_pose_is_verbatim_upstream():
    upstream = _function(ast.parse(OFFICIAL.read_text(encoding="utf-8")), "compute_new_pose")
    ours = _function(ast.parse(inspect.getsource(poses)), "compute_new_pose")
    dump = lambda f: [ast.dump(s, include_attributes=False) for s in f.body]
    assert dump(ours) == dump(upstream)
    assert ast.dump(ours.args) == ast.dump(upstream.args)


def _upstream_compute_new_pose():
    """从上游副本里取出 compute_new_pose 原函数（AST 原样编译执行，不导入上游的 tqdm/logging 依赖）。"""
    from scipy.spatial.transform import Rotation
    fn = _function(ast.parse(OFFICIAL.read_text(encoding="utf-8")), "compute_new_pose")
    namespace = dict(np=np, torch=torch, Rotation=Rotation)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(OFFICIAL), "exec"), namespace)
    return namespace["compute_new_pose"]


# 存档金标准（docs/design/droid-golden，在生成它的平台上算出）的跨平台容差（S06a-fix1，Codex S06a-P3-1）：
# scipy 的欧拉角↔矩阵在不同 CPU/数学库上末位可以不同（macOS ARM64 实测最大 4.44e-16），逐位相同只在同一环境里要求，
# 由上面执行上游原函数来核。1e-12 与 README 里“积分回到下一真实状态差小于 1e-12”同一量级，远小于任何物理意义的位姿差。
GOLDEN_ATOL = 1e-12


@pytest.mark.parametrize("k", range(5))
def test_compute_new_pose_matches_upstream_and_golden(k):
    ep = json.loads((GOLDEN/"droid_samples_golden.json").read_text(encoding="utf-8"))["episodes"][k]
    st = np.asarray(ep["stride15_states"], dtype=np.float64)
    acts = np.asarray(ep["official_actions"], dtype=np.float64)
    pose_in, act_in = torch.from_numpy(st[:-1, None]), torch.from_numpy(acts[:, None])
    out = poses.compute_new_pose(pose_in, act_in)
    assert out.dtype == torch.float64 and tuple(out.shape) == (len(st) - 1, 1, 7)
    # 同一环境里和上游原函数逐位相同
    assert out.numpy().tobytes() == _upstream_compute_new_pose()(pose_in.clone(), act_in.clone()).numpy().tobytes()
    # 和存档金标准在容差内相同（跨平台）
    want = np.asarray(ep["official_new_pose"], dtype=np.float64)
    assert want.shape == out[:, 0].shape
    assert np.abs(out[:, 0].numpy() - want).max() < GOLDEN_ATOL
    # 积分回到下一个真实状态（官方参照的性质，README：差小于 1e-12）
    assert np.abs(out[:, 0].numpy() - st[1:]).max() < 1e-12


def test_compute_new_pose_hand_values():
    pose = torch.tensor([[[0.1, -0.2, 0.3, 0.0, 0.0, 0.5, 0.9]]], dtype=torch.float64)
    action = torch.tensor([[[0.01, 0.02, -0.03, 0.0, 0.0, 0.25, 0.3]]], dtype=torch.float64)
    out = poses.compute_new_pose(pose, action)[0, 0].numpy()
    assert np.allclose(out[:3], [0.11, -0.18, 0.27], atol=1e-15)
    assert np.allclose(out[3:6], [0.0, 0.0, 0.75], atol=1e-12)     # 绕 z 的转角相加
    assert out[6] == 1.0                                             # 夹爪截到 [0, 1]
    # 非对易：R_new = R(Δ)·R，不是 R·R(Δ)
    p = torch.tensor([[[0, 0, 0, 0.4, 0.0, 0.0, 0.5]]], dtype=torch.float64)
    a = torch.tensor([[[0, 0, 0, 0.0, 0.3, 0.0, -0.7]]], dtype=torch.float64)
    from scipy.spatial.transform import Rotation as R
    want = (R.from_euler("xyz", [0, 0.3, 0]).as_matrix() @ R.from_euler("xyz", [0.4, 0, 0]).as_matrix())
    got = poses.compute_new_pose(p, a)[0, 0].numpy()
    assert np.allclose(R.from_euler("xyz", got[3:6]).as_matrix(), want, atol=1e-12)
    assert got[6] == 0.0
    # float32 进 float32 出
    assert poses.compute_new_pose(p.float(), a.float()).dtype == torch.float32
