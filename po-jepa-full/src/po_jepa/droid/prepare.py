"""DROID 1.0.1 准备（docs/design/04-droid-prepare.md）。

阶段 3：任务 `prepare/<片名>` 从下载的 h5 和左相机 MP4 里取出每条回合要用的全部数值、做 §4 的检查，写
`prepared/<片名>/<uuid>.npz` 和字节固定的 `<uuid>.json`（§3）；全片完成后写 `prepared/<片名>/_complete/<配置名>.json`
（§5）。任务 `prepare/stats`（§6）算动作尺度 `prepared/action_scale.json`、冻结窗口 `prepared/windows/<split>.json`
（06 §3，`windows.frozen_windows`）和统计。

帧和步的对齐沿用官方（§1）：MP4 第 i 帧对 h5 第 i 步，取帧位置 `frame_index = [0, 15, …, 15(F−1)]`，
F = ⌊(V − 1)/15⌋ + 1。7 维动作是 `actions.poses_to_diffs`（01 §5.1，逐字取自上游）。
检查失败的回合照样写输出（h5 结构不对、或视频帧数超过 h5 步数时数组无法构造，只写 JSON），下游一律跳过。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy

from po_jepa.droid import download as dl
from po_jepa.droid import index as ix
from po_jepa.droid import splits as sp
from po_jepa.droid import video
from po_jepa.droid import windows as wn
from po_jepa.droid.actions import GROUP_WIDTHS, SCALE_GROUPS, poses_to_diffs
from po_jepa.droid.config import ACTION_SOURCES, DroidConfig, load_config
from po_jepa.runtime import cache

PREPARED_DIR = "prepared"
COMPLETE_DIR = "_complete"
WINDOWS_DIR = "windows"
STATS_DIR = "stats"                     # prepared/stats/<配置名>.json（S05 定：统计和配置有关，见 docs/segments/S05.md §4）
ACTION_SCALE_FILE = "action_scale.json"
PARTIAL_SUFFIX = ".partial"
SCHEMA = "droid_prepared_v1"
COMPLETE_SCHEMA = "droid_prepare_complete_v1"
SCALE_SCHEMA = "droid_action_scale_v1"
STATS_SCHEMA = "droid_prepare_stats_v1"
STATS_COMPLETE_SCHEMA = "droid_prepare_stats_complete_v1"
WORKERS = 8                             # §5：8 个进程并行
MAX_FAILED_RATIO = 0.01                 # §6 第 3 条：任一片失败比例超过 1% 时任务失败
INTERVAL_QUANTILES = (0.01, 0.5, 0.99)  # §6 第 3 条

# §3 的数组（顺序即 npz 和 arrays_sha256 里的顺序）
ARRAY_NAMES = ("frame_index", "state", "command", "action_realized", "action_commanded",
               "state_all", "command_all", "capture_ms", "interval_seconds")
# §4 的检查（顺序即 reasons 的顺序）；前五项是两种动作源共用的
CHECKS = ("h5_structure", "video_frames", "video_size", "finite", "timestamps",
          "action_bounds_realized", "action_bounds_commanded")
BASE_CHECKS = CHECKS[:5]
FLOAT_KINDS, INT_KINDS, FLAG_KINDS = "f", "i", "biu"


def h5_datasets(serial: str) -> dict[str, tuple[str, int | None, str]]:
    """§3 用到的 h5 数据集：键 → (路径, 第二维（None 表示一维）, 允许的 dtype kind)。"""
    return {
        "state_position": ("observation/robot_state/cartesian_position", 6, FLOAT_KINDS),
        "state_gripper": ("observation/robot_state/gripper_position", None, FLOAT_KINDS),
        "command_position": ("action/cartesian_position", 6, FLOAT_KINDS),
        "command_gripper": ("action/gripper_position", None, FLOAT_KINDS),
        "capture": (f"observation/timestamp/cameras/{serial}_estimated_capture", None, INT_KINDS),
        "skip_action": ("observation/timestamp/skip_action", None, FLAG_KINDS),
    }


# ---------------------------------------------------------------- 读 h5（§4 h5_structure）

def read_h5(path: Path | str, serial: str, steps: int) -> tuple[dict | None, dict]:
    """返回 (数据或 None, h5_structure 检查)。每个数据集都在、第一维都是 `steps`（元数据的 trajectory_length）、
    位姿 [T, 6]、夹爪和时间戳 [T]、dtype 是浮点（位姿、夹爪）、有符号整数（时间戳）、bool 或整数（skip_action）。"""
    problems = []
    try:
        file = h5py.File(path, "r")
    except OSError:
        return None, {"passed": False, "steps": steps, "problems": ["cannot open the file as HDF5"]}
    data = {}
    with file:
        for key, (name, width, kinds) in h5_datasets(serial).items():
            obj = file.get(name)
            if not isinstance(obj, h5py.Dataset):
                problems.append(f"{name}: missing")
                continue
            shape = (steps,) if width is None else (steps, width)
            if obj.shape != shape:
                problems.append(f"{name}: shape {list(obj.shape)} != {list(shape)}")
            elif obj.dtype.kind not in kinds:
                problems.append(f"{name}: dtype {obj.dtype} is not of kind {kinds!r}")
            else:
                data[key] = obj[()]
    passed = not problems
    return (data if passed else None), {"passed": passed, "steps": steps, "problems": problems}


# ---------------------------------------------------------------- 解码（§2，§4 video_frames / video_size）

def scan_video(path: Path | str, frames_per_interval: int, s5_stride: int | None) -> dict:
    """顺序解码全部帧：帧数 V、尺寸不对的帧、每第 `frames_per_interval` 帧和（需要时）每第 `s5_stride` 帧的 RGB SHA。"""
    sha15, sha5, bad, first_bad, count = [], [], 0, None, 0
    for index, frame in video.decode_rgb(path):
        count = index + 1
        if frame.shape != video.FRAME_SHAPE or frame.dtype != np.uint8:
            bad += 1
            if first_bad is None:
                first_bad = {"index": index, "shape": list(frame.shape), "dtype": str(frame.dtype)}
        if index % frames_per_interval == 0 or (s5_stride is not None and index % s5_stride == 0):
            digest = video.frame_sha256(frame)
            if index % frames_per_interval == 0:
                sha15.append(digest)
            if s5_stride is not None and index % s5_stride == 0:
                sha5.append(digest)
    return {"frames": count, "bad": bad, "first_bad": first_bad, "sha15": sha15,
            "sha5": sha5 if s5_stride is not None else None, "header": video.header(path)}


# ---------------------------------------------------------------- 数组和检查（§1、§3、§4）

def _num(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def periodic_actions(poses: np.ndarray) -> np.ndarray:
    """[F, 7] 位姿 → [F−1, 7] 周期动作（`poses_to_diffs`）。F < 2 时为空；位姿有非有限值时全为 NaN（action_bounds 判失败）。"""
    rows = poses.shape[0]
    if rows < 2:
        return np.zeros((0, 7), dtype=np.float64)
    if not np.isfinite(poses).all():
        return np.full((rows - 1, 7), np.nan, dtype=np.float64)
    return np.asarray(poses_to_diffs(poses), dtype=np.float64)


def full_poses(data: dict, kind: str) -> np.ndarray:
    """全部步的 7 维位姿 float64 [T, 7]：`kind` 是 state（observation/robot_state）或 command（action）。"""
    return np.concatenate([np.asarray(data[f"{kind}_position"], dtype=np.float64),
                           np.asarray(data[f"{kind}_gripper"], dtype=np.float64)[:, None]], axis=1)


def build_arrays(data: dict, video_frames: int, frames_per_interval: int) -> dict[str, np.ndarray] | None:
    """§3 的九个数组；视频没有帧，或帧数超过 h5 步数（取帧位置越界）时返回 None。"""
    steps = data["state_position"].shape[0]
    if video_frames < 1 or video_frames > steps:
        return None
    state_all, command_all = full_poses(data, "state"), full_poses(data, "command")
    capture_ms = np.asarray(data["capture"], dtype=np.int64)
    frame_index = np.arange(0, video_frames, frames_per_interval, dtype=np.int64)
    state, command = state_all[frame_index], command_all[frame_index]
    arrays = {
        "frame_index": frame_index,
        "state": state,
        "command": command,
        "action_realized": periodic_actions(state),
        "action_commanded": periodic_actions(command),
        "state_all": state_all,
        "command_all": command_all,
        "capture_ms": capture_ms,
        "interval_seconds": np.diff(capture_ms[frame_index]) / 1000,
    }
    check_arrays(arrays)
    return arrays


def check_arrays(arrays: dict[str, np.ndarray]) -> None:
    """九个数组的名字、dtype 和形状关系（§3 的表）。"""
    if sorted(arrays) != sorted(ARRAY_NAMES):
        raise ValueError(f"prepared arrays are {sorted(arrays)}, expected {sorted(ARRAY_NAMES)}")
    T, F = arrays["state_all"].shape[0], arrays["frame_index"].shape[0]
    expected = {"frame_index": ("int64", (F,)), "state": ("float64", (F, 7)), "command": ("float64", (F, 7)),
                "action_realized": ("float64", (max(F - 1, 0), 7)), "action_commanded": ("float64", (max(F - 1, 0), 7)),
                "state_all": ("float64", (T, 7)), "command_all": ("float64", (T, 7)), "capture_ms": ("int64", (T,)),
                "interval_seconds": ("float64", (max(F - 1, 0),))}
    for name, (dtype, shape) in expected.items():
        a = arrays[name]
        if str(a.dtype) != dtype or a.shape != shape:
            raise ValueError(f"prepared array {name}: {a.dtype} {list(a.shape)}, expected {dtype} {list(shape)}")


def run_checks(h5_check: dict, data: dict | None, arrays: dict | None, scan: dict, steps: int,
               config: DroidConfig) -> dict:
    """§4 的七项检查。前提没满足、算不了的检查记 `passed: None`（不进 reasons，但两种动作源都判为没通过）。"""
    t, a = config.time, config.action
    checks = {"h5_structure": h5_check,
              "video_frames": {"passed": scan["frames"] == steps - 1, "video_frames": scan["frames"],
                               "expected": steps - 1},
              "video_size": {"passed": scan["frames"] > 0 and scan["bad"] == 0, "bad_frames": scan["bad"],
                             "first_bad": scan["first_bad"], "expected": list(video.FRAME_SHAPE)}}
    if data is None:
        checks["finite"] = {"passed": None}
    else:
        state_all, command_all = full_poses(data, "state"), full_poses(data, "command")
        grip = state_all[:, 6]
        finite_grip = grip[np.isfinite(grip)]
        nonfinite_state = int((~np.isfinite(state_all)).sum())
        nonfinite_command = int((~np.isfinite(command_all)).sum())
        outside = int((~((grip >= 0.0) & (grip <= 1.0))).sum())
        checks["finite"] = {"passed": nonfinite_state == 0 and nonfinite_command == 0 and outside == 0,
                            "nonfinite_state": nonfinite_state, "nonfinite_command": nonfinite_command,
                            "gripper_outside_0_1": outside,
                            "gripper_min": _num(finite_grip.min()) if finite_grip.size else None,
                            "gripper_max": _num(finite_grip.max()) if finite_grip.size else None}
    if arrays is None:
        checks["timestamps"] = {"passed": None}
        for source in ACTION_SOURCES:
            checks[f"action_bounds_{source}"] = {"passed": None}
        return checks
    intervals = arrays["interval_seconds"]
    non_increasing = int((np.diff(arrays["capture_ms"]) <= 0).sum())
    out_of_range = int((~((intervals >= t.min_interval_seconds) & (intervals <= t.max_interval_seconds))).sum())
    checks["timestamps"] = {"passed": non_increasing == 0 and out_of_range == 0,
                            "non_increasing_steps": non_increasing, "intervals_out_of_range": out_of_range,
                            "interval_min": _num(intervals.min()) if intervals.size else None,
                            "interval_max": _num(intervals.max()) if intervals.size else None}
    bounds = (a.position_bound_m, a.rotation_bound_rad, a.gripper_bound)
    for source in ACTION_SOURCES:
        action = arrays[f"action_{source}"]
        start, groups, ok = 0, {}, True
        for group, width, bound in zip(SCALE_GROUPS, GROUP_WIDTHS, bounds):
            block = np.abs(action[:, start:start + width])
            start += width
            inside = bool((block <= bound).all())   # NaN 不满足 ≤，判为越界
            ok = ok and inside
            finite = block[np.isfinite(block)]
            groups[group] = {"max_abs": _num(finite.max()) if finite.size else None,
                             "nonfinite": int((~np.isfinite(block)).sum()), "bound": bound, "inside": inside}
        checks[f"action_bounds_{source}"] = {"passed": ok, **groups}
    return checks


def verdicts(checks: dict) -> dict:
    """`passed_<源>` = 前五项都通过且该源的 action_bounds 通过；`reasons` = 判为没通过（False）的检查名，按 CHECKS 顺序。"""
    base = all(checks[c]["passed"] is True for c in BASE_CHECKS)
    out = {f"passed_{s}": base and checks[f"action_bounds_{s}"]["passed"] is True for s in ACTION_SOURCES}
    out["reasons"] = [c for c in CHECKS if checks[c]["passed"] is False]
    return out


def array_sha256(a: np.ndarray) -> str:
    """§3：sha256(np.ascontiguousarray(a).tobytes())。"""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def limits(config: DroidConfig) -> dict:
    """检查和取帧用到的配置值（记进 JSON；同一数据根下必须一致，否则报错）。"""
    t, a = config.time, config.action
    return {"frames_per_interval": t.frames_per_interval, "min_interval_seconds": t.min_interval_seconds,
            "max_interval_seconds": t.max_interval_seconds, "position_bound_m": a.position_bound_m,
            "rotation_bound_rad": a.rotation_bound_rad, "gripper_bound": a.gripper_bound,
            "frame_shape": list(video.FRAME_SHAPE), "s5_stride_frames": config.encode.frame_rate_stride_frames}


def versions() -> dict:
    return {"h5py": h5py.__version__, "opencv": video.opencv_version(), "numpy": np.__version__,
            "scipy": scipy.__version__}


def prepare_episode(h5_path: Path | str, mp4_path: Path | str, episode: dict, config: DroidConfig, *,
                    shard: str, s5: bool, source: dict) -> tuple[dict[str, np.ndarray] | None, dict]:
    """一条回合：返回 (§3 的数组或 None, 准备 JSON 的内容)。`source` 是 {h5_sha256, mp4_sha256}（取自 DOWNLOADED.json）。"""
    per = config.time.frames_per_interval
    steps = episode["steps"]
    data, h5_check = read_h5(h5_path, episode["left_serial"], steps)
    scan = scan_video(mp4_path, per, config.encode.frame_rate_stride_frames if s5 else None)
    arrays = build_arrays(data, scan["frames"], per) if data is not None else None
    checks = run_checks(h5_check, data, arrays, scan, steps, config)
    record = {
        "schema": SCHEMA,
        "uuid": episode["uuid"], "shard": shard, "lab": episode["lab"], "scene_id": episode["scene_id"],
        "outcome": episode["outcome"], "steps": steps, "video_frames": scan["frames"],
        "rows": int(arrays["frame_index"].shape[0]) if arrays is not None else None,
        "video": {k: (_num(v) if k == "fps" and v is not None else v) for k, v in scan["header"].items()},
        "skip_action_count": int(np.sum(data["skip_action"])) if data is not None else None,
        "arrays_sha256": ({name: {"sha256": array_sha256(arrays[name]), "dtype": str(arrays[name].dtype),
                                  "shape": list(arrays[name].shape)} for name in ARRAY_NAMES}
                          if arrays is not None else None),
        "frame_sha256": scan["sha15"] if arrays is not None else None,
        "checks": checks,
        **verdicts(checks),
        "limits": limits(config),
        "source": {"h5_sha256": source["h5_sha256"], "mp4_sha256": source["mp4_sha256"],
                   "code_commit": ix.code_commit(), **versions()},
    }
    if s5:
        record["frame_sha256_s5"] = scan["sha5"]
    return arrays, record


# ---------------------------------------------------------------- 写和读（§3）

def prepared_dir(data_root: Path | str) -> Path:
    return Path(data_root)/PREPARED_DIR


def episode_paths(data_root: Path | str, shard: str, uuid: str) -> tuple[Path, Path]:
    base = prepared_dir(data_root)/shard
    return base/f"{uuid}.json", base/f"{uuid}.npz"


def write_prepared(json_path: Path, npz_path: Path, arrays: dict | None, record: dict) -> str:
    """先删旧的 JSON（没有 JSON 就算没做完），再写 npz（先写 .partial 再换名），最后写 JSON（`cache._write_json`）。
    返回 JSON 的 SHA。数组无法构造时不写 npz（并删掉旧的）。"""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (json_path, json_path.with_name(json_path.name + PARTIAL_SUFFIX),
                 npz_path.with_name(npz_path.name + PARTIAL_SUFFIX)):
        path.unlink(missing_ok=True)
    if arrays is None:
        npz_path.unlink(missing_ok=True)
    else:
        temp = npz_path.with_name(npz_path.name + PARTIAL_SUFFIX)
        with temp.open("wb") as file:
            np.savez(file, **{name: arrays[name] for name in ARRAY_NAMES})
        os.replace(temp, npz_path)
    cache._write_json(json_path, record)
    return cache.file_sha(json_path)


@dataclass(frozen=True)
class PreparedEpisode:
    """`load_prepared` 的结果：准备 JSON 的内容、它的 SHA 和九个数组（数组无法构造的回合为空字典）。"""

    info: dict
    json_sha256: str
    arrays: dict

    @property
    def uuid(self) -> str:
        return self.info["uuid"]

    @property
    def rows(self) -> int | None:
        return self.info["rows"]

    def passed(self, source: str) -> bool:
        if source not in ACTION_SOURCES:
            raise ValueError(f"source must be one of {ACTION_SOURCES}")
        return self.info[f"passed_{source}"]


def _read_json(path: Path) -> dict:
    value = ix._loads(path.read_bytes(), str(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def read_record(json_path: Path) -> dict:
    record = _read_json(json_path)
    if record.get("schema") != SCHEMA:
        raise ValueError(f"{json_path}: schema is not {SCHEMA!r}")
    if record.get("uuid") != json_path.stem:
        raise ValueError(f"{json_path}: uuid {record.get('uuid')!r} does not match the file name")
    return record


def load_prepared(path: Path | str) -> PreparedEpisode:
    """读一条回合的准备输出（`path` 是 `<uuid>.json`、`<uuid>.npz` 或去掉扩展名的路径），先核 JSON 里的 `arrays_sha256`
    （每个数组的 SHA、dtype、形状，名字集合必须正好是九个），对不上就报错。"""
    path = Path(path)
    base = path.with_suffix("") if path.suffix in (".json", ".npz") else path
    json_path, npz_path = base.with_name(base.name + ".json"), base.with_name(base.name + ".npz")
    record = read_record(json_path)
    json_sha = cache.file_sha(json_path)
    expected = record.get("arrays_sha256")
    if expected is None:
        return PreparedEpisode(record, json_sha, {})
    if not isinstance(expected, dict) or sorted(expected) != sorted(ARRAY_NAMES):
        raise ValueError(f"{json_path}: arrays_sha256 must list exactly {list(ARRAY_NAMES)}")
    with np.load(npz_path, allow_pickle=False) as npz:
        if sorted(npz.files) != sorted(ARRAY_NAMES):
            raise ValueError(f"{npz_path}: arrays {sorted(npz.files)} != {sorted(ARRAY_NAMES)}")
        arrays = {name: npz[name] for name in ARRAY_NAMES}
    for name in ARRAY_NAMES:
        a, want = arrays[name], expected[name]
        got = {"sha256": array_sha256(a), "dtype": str(a.dtype), "shape": list(a.shape)}
        if got != want:
            raise ValueError(f"{npz_path}: array {name} does not match {json_path.name} ({got} != {want})")
    check_arrays(arrays)
    if record.get("rows") != arrays["frame_index"].shape[0]:
        raise ValueError(f"{json_path}: rows {record.get('rows')} != len(frame_index)")
    return PreparedEpisode(record, json_sha, arrays)


# ---------------------------------------------------------------- 一条回合的任务（§5）

def _source_shas(downloaded: dict, episode: dict) -> dict:
    files = downloaded["files"]
    return {"h5_sha256": files[dl.H5_FILE]["sha256"], "mp4_sha256": files[f"{episode['left_serial']}.mp4"]["sha256"]}


def _current(record: dict, npz_path: Path, source: dict, shard: str, s5: bool, config: DroidConfig) -> bool:
    """已有输出是否可以直接用：源文件 SHA 和 DOWNLOADED.json 一致、片名一致、有数组时 npz 在、该有 s5 帧 SHA 时有。
    检查用的配置值（limits）不同就报错：同一数据根不能混用两套检查口径。"""
    if record.get("limits") != limits(config):
        raise ValueError(f"prepared {record.get('uuid')}: written with limits {record.get('limits')}, "
                         f"this config has {limits(config)}; one data root cannot mix them")
    src = record.get("source") or {}
    return (src.get("h5_sha256") == source["h5_sha256"] and src.get("mp4_sha256") == source["mp4_sha256"]
            and record.get("shard") == shard
            and (record.get("arrays_sha256") is None or npz_path.is_file())
            and (not s5 or "frame_sha256_s5" in record))


def prepare_one(config: DroidConfig, data_root: Path | str, shard: str, episode: dict, s5: bool) -> dict:
    """一条回合：返回 {uuid, status: prepared|skipped, prepared_sha256, passed_realized, passed_commanded}。"""
    uuid = episode["uuid"]
    directory = dl.episode_dir(data_root, shard, uuid)
    source = _source_shas(dl.read_downloaded(directory, episode, config.data), episode)
    json_path, npz_path = episode_paths(data_root, shard, uuid)
    if json_path.is_file():
        record = read_record(json_path)
        if _current(record, npz_path, source, shard, s5, config):
            return {"uuid": uuid, "status": "skipped", "prepared_sha256": cache.file_sha(json_path),
                    "passed_realized": record["passed_realized"], "passed_commanded": record["passed_commanded"]}
    if (directory/dl.RAW_DELETED_FILE).exists():
        raise ValueError(f"{directory}: raw files were deleted but the prepared output is missing or stale")
    arrays, record = prepare_episode(directory/dl.H5_FILE, directory/f"{episode['left_serial']}.mp4", episode, config,
                                     shard=shard, s5=s5, source=source)
    sha = write_prepared(json_path, npz_path, arrays, record)
    return {"uuid": uuid, "status": "prepared", "prepared_sha256": sha,
            "passed_realized": record["passed_realized"], "passed_commanded": record["passed_commanded"]}


def _prepare_star(args: tuple) -> dict:
    return prepare_one(*args)


def complete_path(data_root: Path | str, shard: str, config: DroidConfig) -> Path:
    return prepared_dir(data_root)/shard/COMPLETE_DIR/f"{config.name}.json"


def run_prepare(config: DroidConfig, data_root: Path | str, shard: str, *, workers: int = WORKERS) -> dict:
    """任务 `prepare/<片名>`。片完成记录已存在就什么都不做，返回它（12 §3）。

    要求 `download/<片名>` 在本配置下已完成（片完成记录的 uuid 列表等于片计划）。`val`、`id_test`、`ood_test`
    三片的前 `test_extra_episodes` 条另记 `frame_sha256_s5`。`workers` > 1 时用进程池（每个进程 OpenCV 单线程）。
    """
    done = complete_path(data_root, shard, config)
    if done.exists():
        return _read_json(done)
    splits = sp.load_splits(data_root, config)
    plan = sp.shard_plan(splits, config.data)
    if shard not in plan:
        raise ValueError(f"shard {shard!r} is not used by config {config.name!r}; its shards are {list(plan)}")
    uuids = plan[shard]
    downloaded = dl.complete_path(data_root, shard, config)
    if not downloaded.is_file():
        raise ValueError(f"download/{shard} has not completed for config {config.name!r} ({downloaded} is missing)")
    if _read_json(downloaded).get("uuids") != uuids:
        raise ValueError(f"{downloaded}: its uuid list is not this config's plan for {shard}")
    index = ix.index_dir(data_root)
    by_uuid = {e["uuid"]: e for e in sp.read_episodes(index) if e["selected"]}
    extra = set(uuids[:config.encode.test_extra_episodes]) if shard in sp.TEST_SHARDS else set()
    tasks = [(config, str(data_root), shard, by_uuid[u], u in extra) for u in uuids]
    if workers > 1:
        # spawn：不在多线程的父进程里 fork（OpenCV、HTTP 线程），子进程从头导入
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=video.single_thread) as pool:
            results = list(pool.map(_prepare_star, tasks))
    else:
        video.single_thread()
        results = [_prepare_star(task) for task in tasks]
    counts = {"episodes": len(results)}
    for source in ACTION_SOURCES:
        passed = sum(r[f"passed_{source}"] for r in results)
        counts |= {f"passed_{source}": passed, f"failed_{source}": len(results) - passed}
    record = {
        "schema": COMPLETE_SCHEMA,
        "shard": shard,
        "config": config.name,
        "config_sha256": config.config_sha256,
        "splits_sha256": cache.file_sha(index/ix.SPLITS_FILE),
        "uuids": uuids,
        "prepared_sha256": [r["prepared_sha256"] for r in results],
        "counts": counts,
        "code_commit": ix.code_commit(),
    }
    done.parent.mkdir(parents=True, exist_ok=True)
    done.with_name(done.name + PARTIAL_SUFFIX).unlink(missing_ok=True)
    cache._write_json(done, record)
    statuses = {status: sum(r["status"] == status for r in results) for status in ("prepared", "skipped")}
    return dict(record, statuses=statuses)


# ---------------------------------------------------------------- 统计任务（§6）

def action_scale(pool: list[str], episodes: dict[str, PreparedEpisode], config: DroidConfig,
                 splits_sha256: str) -> dict:
    """§6 第 1 条 / 01 §5.2：每种动作源只用 `pool` 里 `passed_<源>` 为真的回合，按组把通道的绝对值拼在一起取
    `numpy.quantile(…, q, method="linear")`（float64）。三个尺度都必须是有限正数。"""
    q = config.action.scale_quantile
    out = {}
    record = {"episodes": {}, "excluded": {}, "values": {}}
    for source in ACTION_SOURCES:
        used = [u for u in pool if episodes[u].passed(source)]
        if not used:
            raise ValueError(f"no episode in the action-scale pool passed the {source} checks")
        actions = np.concatenate([episodes[u].arrays[f"action_{source}"] for u in used], axis=0)
        scale, start, counts = {}, 0, {}
        for group, width in zip(SCALE_GROUPS, GROUP_WIDTHS):
            values = np.abs(actions[:, start:start + width]).astype(np.float64).ravel()
            start += width
            value = float(np.quantile(values, q, method="linear"))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"action scale {source}.{group} = {value} is not a positive finite number")
            scale[group], counts[group] = value, int(values.size)
        out[source] = scale
        record["episodes"][source] = used
        record["excluded"][source] = [u for u in pool if u not in set(used)]
        record["values"][source] = counts
    return {
        **out,
        "schema": SCALE_SCHEMA,
        "pool": {"rule": "splits.json train_order[:scale_episodes] (full training order, failures included)",
                 "scale_episodes": config.action.scale_episodes, "splits_sha256": splits_sha256,
                 "prepared_sha256": {u: episodes[u].json_sha256 for u in pool}},
        "quantile": q,
        "algorithm": ("numpy.quantile(values, quantile, method='linear') in float64; values = |action_<source>| of "
                      "every interval of the used episodes, channels pooled per group (position 0-2, rotation 3-5, "
                      "gripper 6); scale only, no mean subtraction"),
        **record,
    }


def shard_stats(plan: dict[str, list[str]], records: dict[str, dict], config: DroidConfig) -> tuple[dict, list[str]]:
    """§6 第 3 条：每片通过和失败的条数（失败按本配置的动作源）、失败原因计数、rows 总数；interval_seconds 的分位数。"""
    source = config.action.source
    shards, train_excluded = {}, []
    for shard, uuids in plan.items():
        reasons: dict[str, int] = {}
        failed = [u for u in uuids if not records[u][f"passed_{source}"]]
        for u in uuids:
            for reason in records[u]["reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
        if shard.startswith(sp.SHARD_PREFIX):
            train_excluded.extend(failed)
        shards[shard] = {
            "episodes": len(uuids),
            **{f"passed_{s}": sum(bool(records[u][f"passed_{s}"]) for u in uuids) for s in ACTION_SOURCES},
            "failed": len(failed),
            "failed_ratio": len(failed)/len(uuids) if uuids else 0.0,
            "reasons": {r: reasons[r] for r in CHECKS if r in reasons},
            "rows": sum(records[u]["rows"] or 0 for u in uuids),
            "rows_passed": sum(records[u]["rows"] or 0 for u in uuids if u not in set(failed)),
        }
    return shards, train_excluded


def over_limit(shards: dict) -> list[str]:
    """失败比例超过 MAX_FAILED_RATIO（1%）的片；恰好 1% 不算超过。"""
    return [shard for shard, v in shards.items() if v["failed_ratio"] > MAX_FAILED_RATIO]


def run_stats(config: DroidConfig, data_root: Path | str) -> dict:
    """任务 `prepare/stats`（§6）。完成记录 `prepared/_complete/stats-<配置名>.json` 已存在就什么都不做，返回它。

    依赖本配置的全部 `prepare/<片名>`。动作尺度和冻结窗口和配置无关（同一数据根共用）：文件不存在就写，已存在就
    重算并要求逐字节相同。`use_failures: false` 的配置不重算这两样，要求它们已由主配置写好。统计写
    `prepared/stats/<配置名>.json`（各片条数和配置有关）；任一片失败比例超过 1% 时写完统计后任务失败。
    """
    root = prepared_dir(data_root)
    done = root/COMPLETE_DIR/f"stats-{config.name}.json"
    if done.exists():
        return _read_json(done)
    splits = sp.load_splits(data_root, config)
    splits_sha = cache.file_sha(ix.index_dir(data_root)/ix.SPLITS_FILE)
    plan = sp.shard_plan(splits, config.data)
    records, json_sha, shard_of = {}, {}, {}
    for shard, uuids in plan.items():
        path = complete_path(data_root, shard, config)
        if not path.is_file():
            raise ValueError(f"prepare/{shard} has not completed for config {config.name!r} ({path} is missing)")
        complete = _read_json(path)
        if complete.get("uuids") != uuids:
            raise ValueError(f"{path}: its uuid list is not this config's plan for {shard}")
        for uuid, sha in zip(uuids, complete["prepared_sha256"]):
            json_path, _ = episode_paths(data_root, shard, uuid)
            if cache.file_sha(json_path) != sha:
                raise ValueError(f"{json_path} changed after prepare/{shard} completed")
            records[uuid], json_sha[uuid], shard_of[uuid] = read_record(json_path), sha, shard

    shards, train_excluded = shard_stats(plan, records, config)
    intervals = []
    for uuid, record in records.items():
        if record["arrays_sha256"] is not None:
            episode = load_prepared(episode_paths(data_root, shard_of[uuid], uuid)[0])
            values = episode.arrays["interval_seconds"]
            intervals.append(values[np.isfinite(values)])
    pooled = np.concatenate(intervals) if intervals else np.zeros(0)
    over = over_limit(shards)
    stats = {
        "schema": STATS_SCHEMA,
        "config": config.name,
        "config_sha256": config.config_sha256,
        "splits_sha256": splits_sha,
        "source": config.action.source,
        "shards": shards,
        "train_excluded": {"count": len(train_excluded), "uuids": train_excluded},
        "interval_seconds_quantiles": ({str(q): float(v) for q, v in
                                        zip(INTERVAL_QUANTILES, np.quantile(pooled, INTERVAL_QUANTILES, method="linear"))}
                                       if pooled.size else None),
        "max_failed_ratio": MAX_FAILED_RATIO,
        "over_limit": over,
    }
    stats_path = root/STATS_DIR/f"{config.name}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    temp = stats_path.with_name(stats_path.name + PARTIAL_SUFFIX)
    temp.write_bytes(cache._json_bytes(stats))
    os.replace(temp, stats_path)   # 统计是可重算的报告，重跑时覆盖
    if over:
        raise RuntimeError(f"prepare/stats: shards {over} have more than {MAX_FAILED_RATIO:.0%} failed episodes "
                           f"(see {stats_path}); a systematic problem needs a look before going on")

    scale_path = root/ACTION_SCALE_FILE
    window_paths = {split: root/WINDOWS_DIR/f"{split}.json" for split in wn.SPLITS}
    if config.data.use_failures:
        pool = splits["train_order"][:config.action.scale_episodes]
        if len(pool) < config.action.scale_episodes:
            raise ValueError(f"training order has {len(pool)} episodes, action.scale_episodes needs "
                             f"{config.action.scale_episodes}")
        missing = [u for u in pool if u not in records]
        if missing:
            raise ValueError(f"action-scale pool episodes {missing[:3]} are not prepared under config {config.name!r} "
                             f"(data.train_episodes must be at least action.scale_episodes)")
        pool_episodes = {u: load_prepared(episode_paths(data_root, shard_of[u], u)[0]) for u in pool}
        for u in pool:
            if pool_episodes[u].json_sha256 != json_sha[u]:
                raise ValueError(f"prepared {u} changed while prepare/stats was running")
        _write_or_same(scale_path, cache._json_bytes(action_scale(pool, pool_episodes, config, splits_sha)))
        for split in wn.SPLITS:
            uuids = splits[split]
            content = wn.frozen_windows(split, uuids, rows={u: records[u]["rows"] for u in uuids},
                                        passed={u: records[u]["passed_realized"] for u in uuids},
                                        prepared_sha256={u: json_sha[u] for u in uuids},
                                        splits_sha256=splits_sha, config=config)
            _write_or_same(window_paths[split], cache._json_bytes(content))
    else:
        for path in (scale_path, *window_paths.values()):
            if not path.is_file():
                raise ValueError(f"{path} is missing: with data.use_failures false, prepare/stats reuses the files "
                                 f"written by the main configuration's prepare/stats; run that first")
        scale = _read_json(scale_path)
        if scale.get("schema") != SCALE_SCHEMA or scale.get("pool", {}).get("splits_sha256") != splits_sha:
            raise ValueError(f"{scale_path} was not written from this data root's splits.json")
        for split, path in window_paths.items():
            content = _read_json(path)
            if content.get("schema") != wn.SCHEMA or content.get("splits_sha256") != splits_sha:
                raise ValueError(f"{path} was not written from this data root's splits.json")
            for u, sha in content["prepared_sha256"].items():
                if json_sha.get(u, sha) != sha:
                    raise ValueError(f"{path}: prepared {u} changed after the windows were frozen")
    record = {
        "schema": STATS_COMPLETE_SCHEMA,
        "config": config.name,
        "config_sha256": config.config_sha256,
        "splits_sha256": splits_sha,
        "action_scale_sha256": cache.file_sha(scale_path),
        "windows_sha256": {split: cache.file_sha(path) for split, path in window_paths.items()},
        "stats_sha256": cache.file_sha(stats_path),
        "code_commit": ix.code_commit(),
    }
    done.parent.mkdir(parents=True, exist_ok=True)
    done.with_name(done.name + PARTIAL_SUFFIX).unlink(missing_ok=True)
    cache._write_json(done, record)
    return record


def _write_or_same(path: Path, body: bytes) -> None:
    """数据根共用的文件：不存在就写（先写 .partial 再换名）；已存在就必须逐字节相同，否则报错。"""
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f"{path} already exists with different content; the data root was prepared from "
                             f"other inputs or other code (move it away only after finding out why)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + PARTIAL_SUFFIX)
    temp.write_bytes(body)
    os.replace(temp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DROID 1.0.1 preparation (docs/design/04): one shard or stats")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--shard", required=True, help="train-000 … , val, id_test, ood_test (02 §6) or stats (§6)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.shard == "stats":
        result = run_stats(config, args.data_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result = run_prepare(config, args.data_root, args.shard, workers=args.workers)
    print(json.dumps({"shard": result["shard"], "config": result["config"], "counts": result["counts"],
                      "statuses": result.get("statuses", "already complete")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
