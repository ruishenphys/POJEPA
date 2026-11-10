"""S05：DROID 准备（docs/design/04-droid-prepare.md；验收 14 §3 S05）。

- 金标准回合（需要真实 h5 和 MP4，设 `PO_JEPA_DROID_GOLDEN_RAW` 指向放着 `<uuid>/{trajectory.h5, <序列号>.mp4}`
  的目录，否则跳过）：frame_index、state、command、两种动作、capture_ms、每帧 SHA 和参照逐位相同。
- 合成 h5（h5py 现写）和合成视频（单元测试换掉解码函数；整片测试用 OpenCV 现写的 MP4）覆盖 04 §4 每一项检查。
- 整片和统计任务走 S02 的假存储桶建索引、S04 下载，再准备和统计。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from unittest import mock

import cv2
import h5py
import numpy as np
import pytest

from po_jepa.droid import download as dl
from po_jepa.droid import index as ix
from po_jepa.droid import prepare as pr
from po_jepa.droid import splits as sp
from po_jepa.droid import video
from po_jepa.droid import config as config_module
from po_jepa.droid import windows as wn
from po_jepa.droid.actions import droid_control_spec, poses_to_diffs
from po_jepa.droid.config import load_config
from po_jepa.runtime import cache
from test_droid_download import raw  # noqa: F401 — fixture（本机假存储桶）
from test_droid_index import CONFIGS, _config, meta

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT/"docs"/"design"/"droid-golden"/"droid_samples_golden.json"
GOLDEN_RAW = os.environ.get("PO_JEPA_DROID_GOLDEN_RAW")
SERIAL = "22008760"
CFG = load_config(CONFIGS/"pilot.yaml")
SOURCE = {"h5_sha256": "1"*64, "mp4_sha256": "2"*64}


# ---------------------------------------------------------------- 合成回合

def trajectory(T: int, seed: int = 0) -> dict:
    """T 步的平滑合成轨迹：位置每步 1 mm、欧拉角小幅变化、夹爪在 [0, 1]；指令 = 状态 + 小偏移；相机每步 70 ms
    （15 帧 = 1.05 秒，和真实数据一样）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    pos = np.stack([0.4 + 1e-3*t, -0.05 + 5e-4*t, 0.3 + 2e-4*t], axis=1) + rng.normal(0, 1e-4, (T, 3))
    euler = np.stack([3.1 - 1e-3*t, 0.05 + 2e-3*np.sin(t/7), 0.01*np.cos(t/11)], axis=1)
    grip = 0.5 + 0.4*np.sin(t/13)
    return {"state_position": np.concatenate([pos, euler], axis=1), "state_gripper": grip,
            "command_position": np.concatenate([pos + 2e-3, euler + 1e-3], axis=1),
            "command_gripper": np.clip(grip + 0.05, 0, 1),
            "capture": 1_681_341_667_628 + 70*np.arange(T, dtype=np.int64),
            "skip_action": (np.arange(T) % 4 == 0)}


H5_NAMES = {key: name for key, (name, _, _) in pr.h5_datasets(SERIAL).items()}


def write_h5(path: Path, data: dict, *, drop=(), extra=()) -> Path:
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            if key not in drop:
                f.create_dataset(H5_NAMES[key], data=value)
        for name in extra:
            f.create_dataset(name, data=np.zeros(3))
    return path


def frames_of(V: int, shape=(720, 1280, 3), bad=None):
    """V 帧合成 RGB（每帧内容不同）；bad = {帧号: 形状} 换成别的尺寸。"""
    out = []
    for i in range(V):
        s = (bad or {}).get(i, shape)
        frame = np.zeros(s, dtype=np.uint8)
        frame.flat[i % frame.size] = 1 + i % 250
        out.append(frame)
    return out


@pytest.fixture
def fake_video(monkeypatch):
    """换掉解码：`fake_video[path] = 帧列表`；没登记的路径当打不开（0 帧）。"""
    table: dict[str, list] = {}
    monkeypatch.setattr(video, "decode_rgb", lambda path: iter(enumerate(table.get(str(path), []))))
    monkeypatch.setattr(video, "header", lambda path: ({"width": 1280, "height": 720, "fps": 60.0}
                                                       if str(path) in table else {"width": None, "height": None, "fps": None}))
    return table


def episode(T: int, **kw) -> dict:
    return dict(uuid="AUTOLab+5d05c5aa+2023-07-07-09h-42m-23s", lab="AUTOLab", scene_id=7, outcome="success",
                steps=T, left_serial=SERIAL, **kw)


def prepare(tmp_path, fake_video, data=None, *, T=76, V=None, frames=None, ep=None, s5=False, **h5kw):
    data = trajectory(T) if data is None else data
    h5 = write_h5(tmp_path/"trajectory.h5", data, **h5kw)
    mp4 = tmp_path/f"{SERIAL}.mp4"
    fake_video[str(mp4)] = frames if frames is not None else frames_of(T - 1 if V is None else V)
    return pr.prepare_episode(h5, mp4, ep or episode(T), CFG, shard="train-000", s5=s5, source=SOURCE)


# ---------------------------------------------------------------- §1、§3：对齐和数组

def test_arrays_alignment_and_values(tmp_path, fake_video):
    T = 76
    data = trajectory(T)
    arrays, rec = prepare(tmp_path, fake_video, data)
    V = T - 1
    F = (V - 1)//15 + 1
    idx = np.arange(0, V, 15)
    assert F == 5 and arrays["frame_index"].tolist() == [0, 15, 30, 45, 60] and arrays["frame_index"].dtype == np.int64
    state_all = np.concatenate([data["state_position"], data["state_gripper"][:, None]], axis=1)
    command_all = np.concatenate([data["command_position"], data["command_gripper"][:, None]], axis=1)
    assert arrays["state_all"].tobytes() == state_all.tobytes() and arrays["command_all"].tobytes() == command_all.tobytes()
    assert arrays["state"].tobytes() == state_all[idx].tobytes() and arrays["command"].tobytes() == command_all[idx].tobytes()
    assert arrays["action_realized"].tobytes() == poses_to_diffs(state_all[idx]).tobytes()
    assert arrays["action_commanded"].tobytes() == poses_to_diffs(command_all[idx]).tobytes()
    assert arrays["capture_ms"].tolist() == data["capture"].tolist() and arrays["capture_ms"].dtype == np.int64
    assert arrays["interval_seconds"].tolist() == [1.05]*4   # 15 × 70 ms
    for name, (dtype, shape) in {"state": ("float64", (F, 7)), "action_realized": ("float64", (F - 1, 7)),
                                 "state_all": ("float64", (T, 7)), "interval_seconds": ("float64", (F - 1,))}.items():
        assert str(arrays[name].dtype) == dtype and arrays[name].shape == shape
    assert list(arrays) == list(pr.ARRAY_NAMES)
    assert rec["rows"] == F and rec["steps"] == T and rec["video_frames"] == V
    assert rec["passed_realized"] and rec["passed_commanded"] and rec["reasons"] == []
    assert all(rec["checks"][c]["passed"] is True for c in pr.CHECKS)
    assert rec["skip_action_count"] == int(data["skip_action"].sum()) == 19
    assert rec["arrays_sha256"] == {n: {"sha256": hashlib.sha256(np.ascontiguousarray(arrays[n]).tobytes()).hexdigest(),
                                        "dtype": str(arrays[n].dtype), "shape": list(arrays[n].shape)}
                                    for n in pr.ARRAY_NAMES}
    frames = fake_video[str(tmp_path/f"{SERIAL}.mp4")]
    assert rec["frame_sha256"] == [hashlib.sha256(frames[i].tobytes()).hexdigest() for i in idx]
    assert "frame_sha256_s5" not in rec
    assert rec["source"]["h5_sha256"] == SOURCE["h5_sha256"] and rec["source"]["mp4_sha256"] == SOURCE["mp4_sha256"]
    assert set(rec["source"]) == {"h5_sha256", "mp4_sha256", "code_commit", "h5py", "opencv", "numpy", "scipy"}
    assert rec["source"]["opencv"] == cv2.__version__ and rec["source"]["h5py"] == h5py.__version__
    assert rec["video"] == {"width": 1280, "height": 720, "fps": 60.0}
    assert {k: rec[k] for k in ("uuid", "shard", "lab", "scene_id", "outcome")} == {
        "uuid": episode(T)["uuid"], "shard": "train-000", "lab": "AUTOLab", "scene_id": 7, "outcome": "success"}


def test_frame_count_formula_when_v_minus_1_is_a_multiple_of_15(tmp_path, fake_video):
    arrays, rec = prepare(tmp_path, fake_video, T=62)   # V = 61 → 帧 0…60，F = 5（最后一帧正好是第 60 帧）
    assert arrays["frame_index"].tolist() == [0, 15, 30, 45, 60] and rec["rows"] == 5 == (61 - 1)//15 + 1


def test_s5_frame_shas(tmp_path, fake_video):
    T = 76
    _, rec = prepare(tmp_path, fake_video, T=T, s5=True)
    frames = fake_video[str(tmp_path/f"{SERIAL}.mp4")]
    V = T - 1
    assert len(rec["frame_sha256_s5"]) == (V - 1)//5 + 1 == 15
    assert rec["frame_sha256_s5"] == [hashlib.sha256(frames[i].tobytes()).hexdigest() for i in range(0, V, 5)]


# ---------------------------------------------------------------- §4：每一项检查

def _fails(rec, *names):
    assert rec["reasons"] == [c for c in pr.CHECKS if c in names], rec["reasons"]
    for name in names:
        assert rec["checks"][name]["passed"] is False


@pytest.mark.parametrize("key", list(H5_NAMES))
def test_h5_structure_missing_dataset(tmp_path, fake_video, key):
    arrays, rec = prepare(tmp_path, fake_video, drop=(key,))
    _fails(rec, "h5_structure")
    assert arrays is None and rec["arrays_sha256"] is None and rec["rows"] is None and rec["frame_sha256"] is None
    assert rec["checks"]["h5_structure"]["problems"] == [f"{H5_NAMES[key]}: missing"]
    for c in ("finite", "timestamps", "action_bounds_realized", "action_bounds_commanded"):
        assert rec["checks"][c] == {"passed": None}
    assert not rec["passed_realized"] and not rec["passed_commanded"]


def test_h5_structure_length_must_equal_metadata_steps(tmp_path, fake_video):
    T = 76
    arrays, rec = prepare(tmp_path, fake_video, trajectory(T), ep=episode(T + 1), V=T)
    _fails(rec, "h5_structure")
    assert len(rec["checks"]["h5_structure"]["problems"]) == 6 and rec["checks"]["h5_structure"]["steps"] == T + 1


@pytest.mark.parametrize("key,value,problem", [
    ("state_position", np.zeros((76, 5)), "shape [76, 5] != [76, 6]"),
    ("command_gripper", np.zeros((76, 1)), "shape [76, 1] != [76]"),
    ("capture", np.zeros(76), "dtype float64 is not of kind 'i'"),
    ("state_gripper", np.zeros(76, dtype=np.int64), "dtype int64 is not of kind 'f'"),
    ("skip_action", np.zeros(76, dtype=np.float32), "dtype float32 is not of kind 'biu'"),
])
def test_h5_structure_shape_and_dtype(tmp_path, fake_video, key, value, problem):
    data = dict(trajectory(76), **{key: value})
    arrays, rec = prepare(tmp_path, fake_video, data)
    _fails(rec, "h5_structure")
    assert rec["checks"]["h5_structure"]["problems"] == [f"{H5_NAMES[key]}: {problem}"]


def test_h5_structure_unreadable_file(tmp_path, fake_video):
    (tmp_path/"trajectory.h5").write_bytes(b"not hdf5")
    mp4 = tmp_path/f"{SERIAL}.mp4"
    fake_video[str(mp4)] = frames_of(75)
    arrays, rec = pr.prepare_episode(tmp_path/"trajectory.h5", mp4, episode(76), CFG, shard="x", s5=False, source=SOURCE)
    _fails(rec, "h5_structure")
    assert rec["checks"]["h5_structure"]["problems"] == ["cannot open the file as HDF5"] and arrays is None


def test_h5_extra_datasets_are_fine(tmp_path, fake_video):
    _, rec = prepare(tmp_path, fake_video, extra=("action/cartesian_velocity",))
    assert rec["reasons"] == []


@pytest.mark.parametrize("V,built", [(76, True), (74, True), (77, False)])
def test_video_frames_must_be_steps_minus_one(tmp_path, fake_video, V, built):
    arrays, rec = prepare(tmp_path, fake_video, T=76, V=V)
    _fails(rec, "video_frames")
    assert rec["checks"]["video_frames"] == {"passed": False, "video_frames": V, "expected": 75}
    assert (arrays is not None) == built   # 帧数超过 h5 步数时取帧位置越界，数组无法构造
    if built:
        assert rec["rows"] == (V - 1)//15 + 1
    else:
        assert rec["checks"]["timestamps"] == {"passed": None} and rec["checks"]["finite"]["passed"] is True


def test_unopenable_video(tmp_path, fake_video):
    h5 = write_h5(tmp_path/"trajectory.h5", trajectory(76))
    arrays, rec = pr.prepare_episode(h5, tmp_path/"missing.mp4", episode(76), CFG, shard="x", s5=False, source=SOURCE)
    _fails(rec, "video_frames", "video_size")
    assert arrays is None and rec["video_frames"] == 0 and rec["video"] == {"width": None, "height": None, "fps": None}


def test_video_size_every_frame(tmp_path, fake_video):
    arrays, rec = prepare(tmp_path, fake_video, frames=frames_of(75, bad={7: (720, 1279, 3)}))
    _fails(rec, "video_size")
    assert rec["checks"]["video_size"]["bad_frames"] == 1
    assert rec["checks"]["video_size"]["first_bad"] == {"index": 7, "shape": [720, 1279, 3], "dtype": "uint8"}
    assert arrays is not None   # 帧号 7 不在取帧位置上，也要判失败


def test_video_size_dtype(tmp_path, fake_video):
    frames = frames_of(75)
    frames[3] = frames[3].astype(np.uint16)
    _, rec = prepare(tmp_path, fake_video, frames=frames)
    _fails(rec, "video_size")


@pytest.mark.parametrize("key,row,col", [("command_position", 3, 2), ("state_position", 4, 0),
                                         ("command_gripper", 2, None), ("state_gripper", 1, None)])
def test_finite_anywhere_in_all_steps(tmp_path, fake_video, key, row, col):
    data = trajectory(76)
    if col is None:
        data[key][row] = np.inf
    else:
        data[key][row, col] = np.nan
    arrays, rec = prepare(tmp_path, fake_video, data)
    _fails(rec, "finite")   # 行 1–4 不在取帧位置上：动作不受影响，只有 finite 失败
    assert arrays is not None and not rec["passed_realized"] and not rec["passed_commanded"]


def test_nonfinite_on_a_sampled_row_breaks_that_action(tmp_path, fake_video):
    data = trajectory(76)
    data["state_position"][30, 4] = np.nan
    arrays, rec = prepare(tmp_path, fake_video, data)
    _fails(rec, "finite", "action_bounds_realized")
    assert np.isnan(arrays["action_realized"]).all() and np.isfinite(arrays["action_commanded"]).all()
    assert rec["checks"]["action_bounds_realized"]["rotation"]["nonfinite"] == 12
    json.dumps(rec, allow_nan=False)   # 记录里没有 NaN


@pytest.mark.parametrize("value", [1.2, -0.01])
def test_finite_state_gripper_in_unit_interval(tmp_path, fake_video, value):
    data = trajectory(76)
    data["state_gripper"][5] = value
    _, rec = prepare(tmp_path, fake_video, data)
    _fails(rec, "finite")
    assert rec["checks"]["finite"]["gripper_outside_0_1"] == 1


def test_finite_gripper_bounds_are_inclusive_and_command_gripper_is_not_bounded(tmp_path, fake_video):
    data = trajectory(76)
    data["state_gripper"][3], data["state_gripper"][4] = 0.0, 1.0
    data["command_gripper"][5] = 1.3
    _, rec = prepare(tmp_path, fake_video, data)
    assert rec["reasons"] == []


def test_timestamps_strictly_increasing(tmp_path, fake_video):
    data = trajectory(76)
    data["capture"][8] = data["capture"][7]
    _, rec = prepare(tmp_path, fake_video, data)
    _fails(rec, "timestamps")
    assert rec["checks"]["timestamps"]["non_increasing_steps"] == 1


def _capture_with_intervals(T, intervals_ms):
    """取帧位置之间的间隔按 intervals_ms（毫秒），位置内部均匀（整数毫秒、严格递增）。"""
    capture = np.zeros(T, dtype=np.int64)
    t = 1_681_341_667_628
    for k, total in enumerate(intervals_ms):
        step = np.linspace(0, total, 16).round().astype(np.int64)
        capture[15*k:15*k + 16] = t + step
        t += total
    capture[15*len(intervals_ms) + 1:] = t + 70*np.arange(1, T - 15*len(intervals_ms))
    assert (np.diff(capture) > 0).all()
    return capture


@pytest.mark.parametrize("ms,ok", [([800, 1600, 1050, 1050], True), ([799, 1050, 1050, 1050], False),
                                   ([1050, 1050, 1050, 1601], False), ([500, 1050, 1050, 1050], False)])
def test_timestamps_interval_range_inclusive(tmp_path, fake_video, ms, ok):
    data = trajectory(76)
    data["capture"] = _capture_with_intervals(76, ms)
    arrays, rec = prepare(tmp_path, fake_video, data)
    assert arrays["interval_seconds"].tolist() == [m/1000 for m in ms]
    if ok:
        assert rec["reasons"] == []
    else:
        _fails(rec, "timestamps")
        assert rec["checks"]["timestamps"]["intervals_out_of_range"] == 1


def test_action_bounds_realized_only(tmp_path, fake_video):
    data = trajectory(76)
    data["state_position"][45:, 1] += 1.2   # 第 30 → 45 帧之间位置跳 1.2 m（实际位姿），指令没跳
    _, rec = prepare(tmp_path, fake_video, data)
    _fails(rec, "action_bounds_realized")
    assert not rec["passed_realized"] and rec["passed_commanded"]
    assert rec["checks"]["action_bounds_realized"]["position"]["inside"] is False
    assert rec["checks"]["action_bounds_realized"]["position"]["max_abs"] > 1.2


def test_action_bounds_commanded_only(tmp_path, fake_video):
    data = trajectory(76)
    data["command_gripper"][15] = 2.5   # 指令夹爪跳 > 1.0；状态没变
    _, rec = prepare(tmp_path, fake_video, data)
    _fails(rec, "action_bounds_commanded")
    assert rec["passed_realized"] and not rec["passed_commanded"]
    assert rec["checks"]["action_bounds_commanded"]["gripper"]["inside"] is False


def test_action_position_bound_is_inclusive(tmp_path, fake_video):
    data = trajectory(76)
    data["state_position"][:, 0] = 0.0
    data["state_position"][15:, 0] = 1.0       # 正好 1.0 m：在界内
    data["command_position"][:, 0] = 0.0
    data["command_position"][15:, 0] = 1.0 + 1e-9
    _, rec = prepare(tmp_path, fake_video, data)
    _fails(rec, "action_bounds_commanded")
    assert rec["checks"]["action_bounds_realized"]["position"]["max_abs"] == 1.0


@pytest.mark.parametrize("group,col,value,bound", [("position", 1, 1.01, 1.0), ("rotation", 4, 3.21, 3.2),
                                                   ("rotation", 5, -3.21, 3.2), ("gripper", 6, -1.01, 1.0)])
def test_run_checks_each_group_bound(group, col, value, bound):
    """周期动作的旋转来自 as_euler，总在 [−π, π] 内，碰不到 3.2；这里直接给 run_checks 一份动作核每组的界。"""
    F = 5
    action = np.zeros((F - 1, 7))
    arrays = {"frame_index": np.arange(0, 75, 15), "state": np.zeros((F, 7)), "command": np.zeros((F, 7)),
              "action_realized": action.copy(), "action_commanded": action.copy(), "state_all": np.zeros((76, 7)),
              "command_all": np.zeros((76, 7)), "capture_ms": 70*np.arange(76), "interval_seconds": np.full(F - 1, 1.05)}
    arrays["action_commanded"][2, col] = value
    arrays["action_realized"][2, col] = math.copysign(bound, value)
    scan = {"frames": 75, "bad": 0, "first_bad": None}
    checks = pr.run_checks({"passed": True}, None, arrays, scan, 76, CFG)
    assert checks["action_bounds_realized"]["passed"] is True
    assert checks["action_bounds_commanded"]["passed"] is False
    assert [g for g in ("position", "rotation", "gripper") if not checks["action_bounds_commanded"][g]["inside"]] == [group]


def test_several_failures_at_once_keep_check_order(tmp_path, fake_video):
    data = trajectory(76)
    data["capture"][8] = data["capture"][7]
    data["command_gripper"][15] = 2.5
    _, rec = prepare(tmp_path, fake_video, data, frames=frames_of(74))
    assert rec["reasons"] == ["video_frames", "timestamps", "action_bounds_commanded"]


def test_verdicts_none_is_not_a_pass():
    checks = {c: {"passed": True} for c in pr.CHECKS}
    assert pr.verdicts(checks) == {"passed_realized": True, "passed_commanded": True, "reasons": []}
    checks["timestamps"] = {"passed": None}
    assert pr.verdicts(checks) == {"passed_realized": False, "passed_commanded": False, "reasons": []}


# ---------------------------------------------------------------- 写和读

def _write(tmp_path, fake_video, sub, **kw):
    work = tmp_path/sub
    work.mkdir()
    arrays, rec = prepare(work, fake_video, **kw)
    json_path, npz_path = work/"out"/f"{rec['uuid']}.json", work/"out"/f"{rec['uuid']}.npz"
    sha = pr.write_prepared(json_path, npz_path, arrays, rec)
    return json_path, npz_path, sha, rec


def test_json_written_twice_is_byte_identical(tmp_path, fake_video):
    a, _, sha_a, rec = _write(tmp_path, fake_video, "a")
    b, _, sha_b, _ = _write(tmp_path, fake_video, "b")
    assert a.read_bytes() == b.read_bytes() and sha_a == sha_b == cache.file_sha(a)
    assert a.read_bytes() == cache._json_bytes(rec)
    loaded_a, loaded_b = pr.load_prepared(a), pr.load_prepared(b)
    assert all(np.array_equal(loaded_a.arrays[n], loaded_b.arrays[n]) for n in pr.ARRAY_NAMES)


def test_load_prepared_returns_the_arrays(tmp_path, fake_video):
    json_path, npz_path, sha, rec = _write(tmp_path, fake_video, "a")
    for path in (json_path, npz_path, json_path.with_suffix("")):
        ep = pr.load_prepared(path)
        assert ep.info == rec and ep.json_sha256 == sha and ep.uuid == rec["uuid"] and ep.rows == 5
        assert ep.passed("realized") and list(ep.arrays) == list(pr.ARRAY_NAMES)
    with pytest.raises(ValueError, match="source must be"):
        ep.passed("other")


def _rewrite_npz(npz_path, change):
    with np.load(npz_path) as npz:
        arrays = {n: npz[n].copy() for n in npz.files}
    change(arrays)
    with npz_path.open("wb") as f:
        np.savez(f, **arrays)


@pytest.mark.parametrize("change,message", [
    (lambda a: a["state"].__setitem__((2, 3), a["state"][2, 3] + 1e-12), "array state does not match"),
    (lambda a: a.__setitem__("capture_ms", a["capture_ms"].astype(np.int32)), "array capture_ms does not match"),
    (lambda a: a.__setitem__("interval_seconds", a["interval_seconds"][:-1]), "array interval_seconds does not match"),
    (lambda a: a.pop("command"), "arrays .* != "),
    (lambda a: a.__setitem__("extra", np.zeros(2)), "arrays .* != "),
])
def test_load_prepared_detects_changed_arrays(tmp_path, fake_video, change, message):
    json_path, npz_path, _, _ = _write(tmp_path, fake_video, "a")
    _rewrite_npz(npz_path, change)
    with pytest.raises(ValueError, match=message):
        pr.load_prepared(json_path)


def test_load_prepared_checks_the_json(tmp_path, fake_video):
    json_path, npz_path, _, rec = _write(tmp_path, fake_video, "a")
    other = json_path.with_name("other.json")
    shutil.copy(json_path, other)
    with pytest.raises(ValueError, match="does not match the file name"):
        pr.load_prepared(other)
    for broken, message in ((dict(rec, schema="x"), "schema"),
                            (dict(rec, arrays_sha256={k: v for k, v in rec["arrays_sha256"].items() if k != "state"}),
                             "must list exactly"),
                            (dict(rec, rows=4), "rows 4")):
        json_path.write_bytes(cache._json_bytes(broken))
        with pytest.raises(ValueError, match=message):
            pr.load_prepared(json_path)
    json_path.write_bytes(cache._json_bytes(rec))
    npz_path.unlink()
    with pytest.raises(FileNotFoundError):
        pr.load_prepared(json_path)


def test_failed_episode_without_arrays_writes_json_only(tmp_path, fake_video):
    json_path, npz_path, _, rec = _write(tmp_path, fake_video, "a", drop=("capture",))
    assert json_path.is_file() and not npz_path.exists()
    # 同一路径上原来有数组（上次的输出），这次数组无法构造：旧 npz 要删掉
    _, _, _, good = _write(tmp_path, fake_video, "b")
    arrays = {n: np.zeros(1) for n in pr.ARRAY_NAMES}
    with npz_path.open("wb") as f:
        np.savez(f, **arrays)
    pr.write_prepared(json_path, npz_path, None, rec)
    assert not npz_path.exists() and json_path.read_bytes() == cache._json_bytes(rec)
    ep = pr.load_prepared(json_path)
    assert ep.arrays == {} and not ep.passed("realized") and ep.info["reasons"] == ["h5_structure"]


# ---------------------------------------------------------------- 金标准回合（14 §3 S05 第一条）

def _golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))["episodes"]


@pytest.mark.skipif(not GOLDEN_RAW or not Path(GOLDEN_RAW or ".").is_dir(),
                    reason="set PO_JEPA_DROID_GOLDEN_RAW to a directory with <uuid>/{trajectory.h5,<serial>.mp4} "
                           "for the five golden episodes")
@pytest.mark.parametrize("k", range(5))
def test_golden_episode_bit_identical(k):
    ref = _golden()[k]
    d = Path(GOLDEN_RAW)/ref["uuid"]
    serial = ref["left_mp4"][:-4]
    ep = dict(uuid=ref["uuid"], lab=ref["uuid"].split("+")[0], scene_id=-1, outcome="success", steps=ref["h5_steps"],
              left_serial=serial)
    arrays, rec = pr.prepare_episode(d/"trajectory.h5", d/ref["left_mp4"], ep, CFG, shard="x", s5=True, source=SOURCE)
    idx = np.asarray(ref["stride15_indices"])
    assert arrays["frame_index"].tolist() == ref["stride15_indices"]
    assert arrays["state"].tobytes() == np.asarray(ref["stride15_states"], dtype=np.float64).tobytes()
    assert arrays["action_realized"].tobytes() == np.asarray(ref["official_actions"], dtype=np.float64).tobytes()
    # 参照里没有指令位姿和相机时间戳：用 h5py 直接读（官方读法 f[...][()]）作参照
    with h5py.File(d/"trajectory.h5", "r") as f:
        command = np.concatenate([f["action/cartesian_position"][()], f["action/gripper_position"][()][:, None]], axis=1)
        capture = f[f"observation/timestamp/cameras/{serial}_estimated_capture"][()]
    assert arrays["command"].tobytes() == command[idx].tobytes()
    assert arrays["action_commanded"].tobytes() == poses_to_diffs(command[idx]).tobytes()
    assert arrays["capture_ms"].tobytes() == capture.astype(np.int64).tobytes()
    assert float(np.median(np.diff(arrays["capture_ms"]))) == ref["capture_dt_ms_median"]
    assert rec["steps"] == ref["h5_steps"] and rec["video_frames"] == ref["video_frames"]
    assert rec["skip_action_count"] == ref["skip_action_count"] and rec["video"]["fps"] == ref["container_fps"]
    shas = {int(i): s for i, s in ref["decoded_frame_sha256"].items()}
    for i, s in shas.items():
        if i % 15 == 0:
            assert rec["frame_sha256"][i//15] == s
        if i % 5 == 0:
            assert rec["frame_sha256_s5"][i//5] == s
    last = ref["video_frames"] - 1
    assert video.frame_sha256(next(f for j, f in video.decode_rgb(d/ref["left_mp4"]) if j == last)) == shas[last]
    assert rec["passed_realized"] and rec["passed_commanded"] and rec["reasons"] == []


# ---------------------------------------------------------------- 整片任务和统计（04 §5、§6）

# 5 个同分布场景各 2 条（第 2 个场景一条失败回合）、2 条留出实验室回合；T = 91 → V = 90、F = 6。
# 测试、验证各 1 个场景（各 2 条），训练顺序 6 条，N = 4、片大小 3。
T_FIX = 91
LABS = ("AUTOLab", "IPRL", "TRI", "CLVR", "IRIS")
SPECS = [dict(lab=lab, stamp=f"2023-07-07-09h-{10*i + j:02d}m-3{j}s", scene_id=1000 + i, user=f"5d05c5a{j}",
              outcome="failure" if (i, j) == (1, 1) else "success", steps=T_FIX)
         for i, lab in enumerate(LABS) for j in range(2)]
SPECS += [dict(lab="PennPAL", stamp=f"2023-07-15-17h-1{j}m-49s", user=f"acda9df{j}", scene_id=-1, steps=T_FIX)
          for j in range(2)]
SMALL = dict(id_test_scenes=1, val_scenes=1, episodes_per_scene=2, ood_quota=(2, 0, 0), train_episodes=4,
             shard_size=3, min_steps=60)


@pytest.fixture(scope="module")
def mp4_bytes(tmp_path_factory):
    """一份真实编码的 90 帧 720p MP4（OpenCV mp4v），所有回合共用。"""
    path = tmp_path_factory.mktemp("mp4")/"v.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 60, (1280, 720))
    base = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    for i in range(T_FIX - 1):
        writer.write(np.roll(base, 7*i, axis=1))
    writer.release()
    return path.read_bytes()


def _h5_bytes(tmp_path, data, name):
    return write_h5(tmp_path/f"{name}.h5", data).read_bytes()


def build_root(fake, root: Path, mp4_bytes: bytes, *, edit=None, **overrides):
    """假存储桶里放元数据和真实的 h5、MP4，建索引、划分，下载全部片。`edit(uuid, data)` 可以改某条回合的轨迹。
    返回 (配置, 入选回合)。"""
    work = root/"_payload"
    work.mkdir(parents=True)
    rows = []
    for spec in SPECS:
        name, value = meta(**spec)
        fake.put(name, value)
        rows.append(dict(value, _name=name))
    cfg = _config(fake, count=len(SPECS), **SMALL)
    cfg = replace(cfg, **overrides) if overrides else cfg
    for k, e in enumerate(sorted((ix.normalize(r, cfg.data) for r in rows), key=lambda e: e["uuid"])):
        data = trajectory(T_FIX, seed=k)
        data["capture"] = data["capture"] + 3*np.arange(T_FIX)   # 每条回合的间隔略有不同
        data = {key: v.copy() for key, v in data.items()}
        if edit:
            edit(e["uuid"], data)
        local = dict(dl.episode_files(e, cfg.data))
        fake.put(local["trajectory.h5"], _h5_bytes(work, data, str(k)))
        fake.put(local[f"{e['left_serial']}.mp4"], mp4_bytes)
    ix.run_index(cfg, root, sleep=lambda s: None, threads=2)
    episodes = {e["uuid"]: e for e in sp.read_episodes(ix.index_dir(root)) if e["selected"]}
    assert len(episodes) == len(SPECS)
    for shard in sp.shard_plan(sp.load_splits(root, cfg), cfg.data):
        dl.run_download(cfg, root, shard, sleep=lambda s: None)
    return cfg, episodes


def replace(cfg, **kw):
    """dataclasses.replace，留出实验室条数按这份小配置的 ood_quota 校验（同 test_droid_index._config）。"""
    with mock.patch.object(config_module, "OOD_TEST_EPISODES", sum(cfg.data.ood_quota)):
        return dataclasses.replace(cfg, **kw)


def small(cfg, **kw):
    """小规模的统计参数：尺度回合池 3 条，附加回合 2 条，规划候选 2 个（1 个干扰项）。"""
    action = dataclasses.replace(cfg.action, scale_episodes=kw.get("scale_episodes", 3))
    encode = dataclasses.replace(cfg.encode, test_extra_episodes=2)
    evaluation = dataclasses.replace(cfg.evaluation, planning_candidates=2)
    return replace(cfg, action=action, encode=encode, evaluation=evaluation)


def prepare_all(cfg, root, workers=1):
    """本配置的每一片：下载（别的配置下好的回合跳过）再准备。"""
    plan = sp.shard_plan(sp.load_splits(root, cfg), cfg.data)
    for shard in plan:
        dl.run_download(cfg, root, shard, sleep=lambda s: None)
    return {shard: pr.run_prepare(cfg, root, shard, workers=workers) for shard in plan}


def test_shard_task_outputs_and_complete_record(raw, mp4_bytes, tmp_path):
    cfg, episodes = build_root(raw, tmp_path, mp4_bytes)
    cfg = small(cfg)
    plan = sp.shard_plan(sp.load_splits(tmp_path, cfg), cfg.data)
    assert list(plan) == ["train-000", "train-001", "val", "id_test", "ood_test"]
    results = prepare_all(cfg, tmp_path)
    for shard, uuids in plan.items():
        done = tmp_path/"prepared"/shard/"_complete"/f"{cfg.name}.json"
        record = json.loads(done.read_text(encoding="utf-8"))
        shas = [cache.file_sha(tmp_path/"prepared"/shard/f"{u}.json") for u in uuids]
        failed = 0   # 失败回合（outcome）照样能通过准备检查
        assert record == {"schema": "droid_prepare_complete_v1", "shard": shard, "config": cfg.name,
                          "config_sha256": cfg.config_sha256,
                          "splits_sha256": cache.file_sha(tmp_path/"index"/"splits.json"), "uuids": uuids,
                          "prepared_sha256": shas,
                          "counts": {"episodes": len(uuids), "passed_realized": len(uuids) - failed, "failed_realized": failed,
                                     "passed_commanded": len(uuids), "failed_commanded": 0},
                          "code_commit": ix.code_commit()}
        assert done.read_bytes() == cache._json_bytes(record)
        assert results[shard]["statuses"] == {"prepared": len(uuids), "skipped": 0}
        for u in uuids:
            ep = pr.load_prepared(tmp_path/"prepared"/shard/f"{u}.json")
            downloaded = json.loads((tmp_path/"raw"/shard/u/"DOWNLOADED.json").read_text(encoding="utf-8"))
            assert ep.info["source"]["h5_sha256"] == downloaded["files"]["trajectory.h5"]["sha256"]
            assert ep.info["source"]["mp4_sha256"] == downloaded["files"][f"{episodes[u]['left_serial']}.mp4"]["sha256"]
            assert ep.rows == 6 and ep.info["reasons"] == [] and ep.info["shard"] == shard
            # 每帧 SHA 是真实 MP4 解码后的
            assert ep.info["frame_sha256"][1] == video.frame_sha256(
                next(f for j, f in video.decode_rgb(tmp_path/"raw"/shard/u/f"{episodes[u]['left_serial']}.mp4") if j == 15))
            # s5 只给三片测试的前 test_extra_episodes 条
            assert ("frame_sha256_s5" in ep.info) == (shard in sp.TEST_SHARDS and uuids.index(u) < 2)
    # 完成记录已在：什么都不做
    assert pr.run_prepare(cfg, tmp_path, "val") == json.loads(
        (tmp_path/"prepared"/"val"/"_complete"/f"{cfg.name}.json").read_text(encoding="utf-8"))


def test_rerun_skips_current_outputs_and_redoes_stale_ones(raw, mp4_bytes, tmp_path):
    cfg, episodes = build_root(raw, tmp_path, mp4_bytes)
    shard = "train-000"
    pr.run_prepare(cfg, tmp_path, shard)
    uuids = sp.shard_plan(sp.load_splits(tmp_path, cfg), cfg.data)[shard]
    before = {u: (tmp_path/"prepared"/shard/f"{u}.json").read_bytes() for u in uuids}
    done = tmp_path/"prepared"/shard/"_complete"/f"{cfg.name}.json"
    done.unlink()
    assert pr.run_prepare(cfg, tmp_path, shard)["statuses"] == {"prepared": 0, "skipped": 3}
    done.unlink()
    # 第 1、2 条：JSON 记的 h5 / MP4 的 SHA 和 DOWNLOADED.json 对不上 → 重做；第 3 条：npz 没了 → 重做
    for u, key in ((uuids[0], "h5_sha256"), (uuids[1], "mp4_sha256")):
        stale = json.loads(before[u])
        stale["source"][key] = "0"*64
        (tmp_path/"prepared"/shard/f"{u}.json").write_bytes(cache._json_bytes(stale))
    (tmp_path/"prepared"/shard/f"{uuids[2]}.npz").unlink()
    result = pr.run_prepare(cfg, tmp_path, shard)
    assert result["statuses"] == {"prepared": 3, "skipped": 0}
    after = {u: (tmp_path/"prepared"/shard/f"{u}.json").read_bytes() for u in uuids}
    assert after == before
    assert not list((tmp_path/"prepared").rglob("*.partial"))


def test_rerun_with_deleted_raw(raw, mp4_bytes, tmp_path):
    cfg, episodes = build_root(raw, tmp_path, mp4_bytes)
    shard = "train-001"
    pr.run_prepare(cfg, tmp_path, shard)
    (u,) = sp.shard_plan(sp.load_splits(tmp_path, cfg), cfg.data)[shard]
    d = tmp_path/"raw"/shard/u
    for f in ("trajectory.h5", f"{episodes[u]['left_serial']}.mp4"):
        (d/f).unlink()
    (d/"RAW_DELETED.json").write_text("{}")
    (tmp_path/"prepared"/shard/"_complete"/f"{cfg.name}.json").unlink()
    assert pr.run_prepare(cfg, tmp_path, shard)["statuses"] == {"prepared": 0, "skipped": 1}
    (tmp_path/"prepared"/shard/f"{u}.json").unlink()
    (tmp_path/"prepared"/shard/"_complete"/f"{cfg.name}.json").unlink()
    with pytest.raises(ValueError, match="raw files were deleted"):
        pr.run_prepare(cfg, tmp_path, shard)


def test_limits_mismatch_is_refused(raw, mp4_bytes, tmp_path):
    cfg, _ = build_root(raw, tmp_path, mp4_bytes)
    pr.run_prepare(cfg, tmp_path, "val")
    other = replace(cfg, time=dataclasses.replace(cfg.time, max_interval_seconds=1.7))
    (tmp_path/"prepared"/"val"/"_complete"/f"{cfg.name}.json").unlink()
    with pytest.raises(ValueError, match="one data root cannot mix them"):
        pr.run_prepare(other, tmp_path, "val")


def test_shard_task_preconditions(raw, mp4_bytes, tmp_path):
    cfg, _ = build_root(raw, tmp_path, mp4_bytes)
    with pytest.raises(ValueError, match="is not used by config"):
        pr.run_prepare(cfg, tmp_path, "train-002")
    (tmp_path/"raw"/"val"/"_complete"/f"{cfg.name}.json").unlink()
    with pytest.raises(ValueError, match="download/val has not completed"):
        pr.run_prepare(cfg, tmp_path, "val")


def test_process_pool_gives_the_same_bytes(raw, mp4_bytes, tmp_path):
    cfg, _ = build_root(raw, tmp_path, mp4_bytes)
    pr.run_prepare(cfg, tmp_path, "train-000", workers=1)
    first = {p.name: p.read_bytes() for p in (tmp_path/"prepared"/"train-000").glob("*.json")}
    shutil.rmtree(tmp_path/"prepared")
    pr.run_prepare(cfg, tmp_path, "train-000", workers=2)
    assert {p.name: p.read_bytes() for p in (tmp_path/"prepared"/"train-000").glob("*.json")} == first


def test_failed_episode_is_written_and_counted(raw, mp4_bytes, tmp_path):
    def edit(uuid, data):
        if uuid.startswith("PennPAL+acda9df0"):
            data["command_gripper"][15] = 2.5   # 只有 commanded 越界
    cfg, _ = build_root(raw, tmp_path, mp4_bytes, edit=edit)
    result = pr.run_prepare(cfg, tmp_path, "ood_test")
    assert result["counts"] == {"episodes": 2, "passed_realized": 2, "failed_realized": 0,
                                "passed_commanded": 1, "failed_commanded": 1}


def test_stats_scale_windows_and_completion(raw, mp4_bytes, tmp_path):
    def edit(uuid, data):
        if uuid == pool[1] or uuid.startswith("PennPAL+acda9df0"):
            data["command_gripper"][30] = 2.5   # 回合池第 2 条和一条留出实验室测试回合只在 commanded 下没通过
    pool = []
    cfg0, _ = build_root(raw, tmp_path/"probe", mp4_bytes)   # 先算出训练顺序，找回合池
    pool[:] = sp.load_splits(tmp_path/"probe", cfg0)["train_order"][:3]
    raw.objects.clear()
    cfg, episodes = build_root(raw, tmp_path/"r", mp4_bytes, edit=edit)
    cfg = small(cfg)
    root = tmp_path/"r"
    prepare_all(cfg, root)
    record = pr.run_stats(cfg, root)
    splits = sp.load_splits(root, cfg)
    assert splits["train_order"][:3] == pool
    assert any(episodes[u]["outcome"] == "failure" for u in splits["train_order"][:6])
    # 动作尺度：按 04 §6 的回合池，手算分位数
    scale = json.loads((root/"prepared"/"action_scale.json").read_text(encoding="utf-8"))
    shard_of = {u: s for s, us in sp.shard_plan(splits, cfg.data).items() for u in us}
    assert scale["episodes"] == {"realized": pool, "commanded": [pool[0], pool[2]]}
    assert scale["excluded"] == {"realized": [], "commanded": [pool[1]]}
    for source in ("realized", "commanded"):
        acts = np.concatenate([pr.load_prepared(root/"prepared"/shard_of[u]/f"{u}.json").arrays[f"action_{source}"]
                               for u in scale["episodes"][source]])
        for group, cols in (("position", [0, 1, 2]), ("rotation", [3, 4, 5]), ("gripper", [6])):
            assert scale[source][group] == _hand_quantile(np.abs(acts[:, cols]).ravel(), 0.99)
            assert type(scale[source][group]) is float
    spec = droid_control_spec(scale, source="realized", config=cfg)
    assert spec.scale == (scale["realized"]["position"],)*3 + (scale["realized"]["rotation"],)*3 + (scale["realized"]["gripper"],)
    assert f"action_scale.json sha256={cache.file_sha(root/'prepared'/'action_scale.json')}" in spec.definition_source
    # 冻结窗口：三个文件，val 只有 windows 和 counts
    for split in wn.SPLITS:
        content = json.loads((root/"prepared"/"windows"/f"{split}.json").read_text(encoding="utf-8"))
        assert content["splits_sha256"] == cache.file_sha(root/"index"/"splits.json")
        assert sorted(content["prepared_sha256"]) == sorted(splits[split])
        assert content["prepared_sha256"] == {u: cache.file_sha(root/"prepared"/split/f"{u}.json") for u in splits[split]}
    # 窗口只看 passed_realized：commanded 没通过的留出实验室回合照样有窗口
    ood = json.loads((root/"prepared"/"windows"/"ood_test.json").read_text(encoding="utf-8"))
    bad = next(u for u in splits["ood_test"] if u.startswith("PennPAL+acda9df0"))
    assert ood["counts"]["passed"] == 2 and any(w["uuid"] == bad for w in ood["windows"])
    val = json.loads((root/"prepared"/"windows"/"val.json").read_text(encoding="utf-8"))
    assert set(val) == {"schema", "split", "salt", "splits_sha256", "prepared_sha256", "windows", "counts"}
    assert val["counts"]["windows"] == 2*3   # F = 6：d = 8 没有候选起点（F − 1 < d）
    assert val["counts"]["windows_by_duration"] == {"1": 2, "2": 2, "4": 2, "8": 0}
    # 统计和完成记录
    stats = json.loads((root/"prepared"/"stats"/f"{cfg.name}.json").read_text(encoding="utf-8"))
    assert stats["over_limit"] == [] and stats["shards"]["train-000"]["episodes"] == 3
    assert stats["shards"]["train-000"]["rows"] == 18
    assert sum(v["reasons"].get("action_bounds_commanded", 0) for v in stats["shards"].values()) == 2
    q = stats["interval_seconds_quantiles"]
    assert list(q) == ["0.01", "0.5", "0.99"] and 1.05 <= q["0.01"] <= q["0.99"] <= 1.05 + 15*3/1000
    assert record == {"schema": "droid_prepare_stats_complete_v1", "config": cfg.name, "config_sha256": cfg.config_sha256,
                      "splits_sha256": cache.file_sha(root/"index"/"splits.json"),
                      "action_scale_sha256": cache.file_sha(root/"prepared"/"action_scale.json"),
                      "windows_sha256": {s: cache.file_sha(root/"prepared"/"windows"/f"{s}.json") for s in wn.SPLITS},
                      "stats_sha256": cache.file_sha(root/"prepared"/"stats"/f"{cfg.name}.json"),
                      "code_commit": ix.code_commit()}
    assert (root/"prepared"/"_complete"/f"stats-{cfg.name}.json").read_bytes() == cache._json_bytes(record)


def _hand_quantile(values, q):
    """numpy 'linear' 分位数的手算：排序后在位置 q·(n − 1) 处线性插值。"""
    v = sorted(float(x) for x in values)
    pos = q*(len(v) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (pos - lo)*(v[hi] - v[lo])


def test_hand_quantile_matches_numpy_linear():
    x = np.random.default_rng(3).random(1001)
    assert _hand_quantile(x, 0.99) == pytest.approx(float(np.quantile(x, 0.99, method="linear")), abs=1e-15)


def test_stats_second_config_reuses_shared_files_and_other_inputs_are_refused(raw, mp4_bytes, tmp_path):
    cfg, _ = build_root(raw, tmp_path, mp4_bytes)
    cfg = small(cfg)
    prepare_all(cfg, tmp_path)
    pr.run_stats(cfg, tmp_path)
    shared = {p: p.read_bytes() for p in [tmp_path/"prepared"/"action_scale.json",
                                          *(tmp_path/"prepared"/"windows").glob("*.json")]}
    # 另一个配置（同一数据根，同样的 N）：重算后逐字节相同，照常完成
    other = replace(cfg, name="pilot2")
    prepare_all(other, tmp_path)
    pr.run_stats(other, tmp_path)
    assert {p: p.read_bytes() for p in shared} == shared
    # 共用文件内容不同 → 报错
    (tmp_path/"prepared"/"_complete"/"stats-pilot2.json").unlink()
    scale = json.loads(shared[tmp_path/"prepared"/"action_scale.json"])
    scale["realized"]["position"] *= 2
    (tmp_path/"prepared"/"action_scale.json").write_bytes(cache._json_bytes(scale))
    with pytest.raises(ValueError, match="already exists with different content"):
        pr.run_stats(other, tmp_path)


def test_stats_use_failures_false_reuses_main_files(raw, mp4_bytes, tmp_path):
    cfg, _ = build_root(raw, tmp_path, mp4_bytes)
    cfg = small(cfg)
    nofail = replace(cfg, name="nofail", data=dataclasses.replace(cfg.data, use_failures=False))
    prepare_all(nofail, tmp_path)
    with pytest.raises(ValueError, match="reuses the files written by the main configuration"):
        pr.run_stats(nofail, tmp_path)
    prepare_all(cfg, tmp_path)
    pr.run_stats(cfg, tmp_path)
    before = (tmp_path/"prepared"/"action_scale.json").read_bytes()
    record = pr.run_stats(nofail, tmp_path)
    assert (tmp_path/"prepared"/"action_scale.json").read_bytes() == before
    assert record["action_scale_sha256"] == cache.file_sha(tmp_path/"prepared"/"action_scale.json")


def test_stats_needs_every_shard_and_unchanged_outputs(raw, mp4_bytes, tmp_path):
    cfg, _ = build_root(raw, tmp_path, mp4_bytes)
    cfg = small(cfg)
    plan = prepare_all(cfg, tmp_path)
    (tmp_path/"prepared"/"ood_test"/"_complete"/f"{cfg.name}.json").rename(tmp_path/"kept.json")
    with pytest.raises(ValueError, match="prepare/ood_test has not completed"):
        pr.run_stats(cfg, tmp_path)
    (tmp_path/"kept.json").rename(tmp_path/"prepared"/"ood_test"/"_complete"/f"{cfg.name}.json")
    u = plan["val"]["uuids"][0]
    path = tmp_path/"prepared"/"val"/f"{u}.json"
    path.write_bytes(path.read_bytes().replace(b'"lab"', b'"lab" ', 1))
    with pytest.raises(ValueError, match="changed after prepare/val completed"):
        pr.run_stats(cfg, tmp_path)


def test_stats_pool_must_be_prepared(raw, mp4_bytes, tmp_path):
    cfg, _ = build_root(raw, tmp_path, mp4_bytes)
    cfg = small(cfg, scale_episodes=5)   # N = 4 < 5：回合池第 5 条没准备
    prepare_all(cfg, tmp_path)
    with pytest.raises(ValueError, match="are not prepared"):
        pr.run_stats(cfg, tmp_path)


def test_stats_fails_above_three_percent_and_writes_nothing_shared(raw, mp4_bytes, tmp_path):
    def edit(uuid, data):
        if uuid.startswith("PennPAL+acda9df1"):
            data["state_gripper"][3] = 1.5   # finite 失败
    cfg, _ = build_root(raw, tmp_path, mp4_bytes, edit=edit)
    cfg = small(cfg)
    prepare_all(cfg, tmp_path)
    with pytest.raises(RuntimeError, match=r"shards \['ood_test'\] have more than 3%"):
        pr.run_stats(cfg, tmp_path)
    stats = json.loads((tmp_path/"prepared"/"stats"/f"{cfg.name}.json").read_text(encoding="utf-8"))
    assert stats["over_limit"] == ["ood_test"] and stats["shards"]["ood_test"]["reasons"] == {"finite": 1}
    assert stats["shards"]["ood_test"]["failed_ratio"] == 0.5
    assert not (tmp_path/"prepared"/"action_scale.json").exists() and not (tmp_path/"prepared"/"windows").exists()
    assert not (tmp_path/"prepared"/"_complete").exists()


def _records(n, failed):
    return {f"u{i}": {"passed_realized": i >= failed, "passed_commanded": True, "rows": 20,
                      "reasons": [] if i >= failed else ["timestamps"]} for i in range(n)}


@pytest.mark.parametrize("n,failed,over", [(100, 3, False), (100, 4, True), (200, 6, False), (199, 6, True),
                                           (1000, 30, False), (1000, 31, True), (5, 0, False)])
def test_three_percent_threshold(n, failed, over):
    records = _records(n, failed)
    shards, excluded = pr.shard_stats({"train-000": list(records)}, records, CFG)
    assert shards["train-000"]["failed"] == failed and excluded == [f"u{i}" for i in range(failed)]
    assert pr.over_limit(shards) == (["train-000"] if over else [])
    assert shards["train-000"]["rows_passed"] == 20*(n - failed)


def test_action_scale_on_synthetic_episodes():
    rng = np.random.default_rng(5)

    def ep(uuid, passed_c=True):
        acts = {s: rng.normal(0, 0.05, (30, 7)) for s in ("realized", "commanded")}
        info = {"uuid": uuid, "rows": 31, "passed_realized": True, "passed_commanded": passed_c}
        return pr.PreparedEpisode(info, hashlib.sha256(uuid.encode()).hexdigest(),
                                  {f"action_{s}": a for s, a in acts.items()})

    eps = {u: ep(u, passed_c=(u != "b")) for u in "abcd"}
    scale = pr.action_scale(list("abcd"), eps, CFG, "s"*64)
    for source, used in (("realized", "abcd"), ("commanded", "acd")):
        acts = np.concatenate([eps[u].arrays[f"action_{source}"] for u in used])
        assert scale[source] == {"position": _hand_quantile(np.abs(acts[:, :3]).ravel(), 0.99),
                                 "rotation": _hand_quantile(np.abs(acts[:, 3:6]).ravel(), 0.99),
                                 "gripper": _hand_quantile(np.abs(acts[:, 6]).ravel(), 0.99)}
        assert scale["values"][source] == {"position": 90*len(used), "rotation": 90*len(used), "gripper": 30*len(used)}
    assert scale["episodes"]["commanded"] == ["a", "c", "d"] and scale["excluded"]["commanded"] == ["b"]
    assert scale["pool"]["prepared_sha256"] == {u: eps[u].json_sha256 for u in "abcd"}
    assert scale["quantile"] == 0.99 and scale["schema"] == "droid_action_scale_v1"
    # 不减均值：全体加常数会改变尺度
    zero = {u: pr.PreparedEpisode(e.info, e.json_sha256, {k: np.zeros_like(v) for k, v in e.arrays.items()})
            for u, e in eps.items()}
    with pytest.raises(ValueError, match="is not a positive finite number"):
        pr.action_scale(list("abcd"), zero, CFG, "s"*64)
