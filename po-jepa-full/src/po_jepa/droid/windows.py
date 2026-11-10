"""DROID 冻结的验证和测试窗口（docs/design/06-droid-windows-and-batches.md §3）。

由 `prepare/stats`（04 §6，`prepare.run_stats`）生成 `prepared/windows/{val,id_test,ood_test}.json`，开训前登记、之后不改。
只含 `passed_realized` 为真的回合。排序哈希 `hw(*parts) = sha256("|".join([window_salt, *map(str, parts)]))`，
哈希相同（不会发生）时按窗口编号再排，结果仍确定。

训练集（06 §1 `train_episodes`）和每一步的训练窗口（06 §2 `step_rng`、`training_windows`、`ac_clips`）是 S07 加的：
只由（种子，步数，流）决定，不依赖数据加载器的状态。把窗口组装成批在 `batches.py`。
"""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np

from po_jepa.droid import splits as sp
from po_jepa.droid.config import ACTION_SOURCES, DroidConfig

SCHEMA = "droid_windows_v1"
SPLITS = ("val", "id_test", "ood_test")
VAL_SPLIT = "val"


def hw(salt: str, *parts) -> str:
    """06 §3 的排序哈希：sha256("|".join([window_salt, *map(str, parts)])) 的十六进制串。"""
    return hashlib.sha256("|".join([salt, *map(str, parts)]).encode("utf-8")).hexdigest()


TRAIN_STREAM = 0   # 06 §2：PO-JEPA 类、Transformer 类和 Koopman（同一种子下同一步看到同样的窗口）
AC_STREAM = 1      # 06 §2：AC 两个条件的 9 帧片段


def window_id(uuid: str, duration: int, start: int) -> str:
    return f"{uuid}|d{duration}|s{start}"


def clip_id(uuid: str, start: int) -> str:
    return f"{uuid}|clip|s{start}"


def frame_rate_id(uuid: str, start_frame: int) -> str:
    return f"{uuid}|fr|f{start_frame}"


def evaluation_durations(config: DroidConfig) -> list[int]:
    """测试时长 {1, 2, 3, 4, 8, 12, 16, 24, 32}：评测配置三组时长的并集，升序。"""
    e = config.evaluation
    durations = sorted({*e.durations_id, *e.durations_heldout, *e.durations_long})
    if len(durations) != len(e.durations_id) + len(e.durations_heldout) + len(e.durations_long):
        raise ValueError("evaluation durations overlap between durations_id, durations_heldout and durations_long")
    return durations


def frame_rate_duration(config: DroidConfig) -> int:
    """原始帧率评测用的窗口时长：frame_rate_total_frames / frames_per_interval（60 / 15 = 4 个周期，10 §8.4）。"""
    total, per = config.evaluation.frame_rate_total_frames, config.time.frames_per_interval
    if total % per:
        raise ValueError(f"frame_rate_total_frames {total} is not a multiple of frames_per_interval {per}")
    return total // per


def _pick(salt: str, split: str, uuid: str, rows: int, duration: int, count: int) -> list[int]:
    """候选起点 s ∈ [0, rows − 1 − d] 按 hw(split, uuid, d, s) 升序取前 count 个（rows − 1 < d 时没有）。"""
    starts = range(rows - duration)
    ranked = sorted(starts, key=lambda s: (hw(salt, split, uuid, duration, s), s))
    return ranked[:count]


def _windows(salt: str, split: str, uuids: Sequence[str], rows: Mapping[str, int],
             durations: Sequence[int], count: int) -> list[dict]:
    """窗口按（回合在划分列表里的顺序，时长，哈希顺序）排列。"""
    out = []
    for uuid in uuids:
        for d in durations:
            for s in _pick(salt, split, uuid, rows[uuid], d, count):
                out.append({"id": window_id(uuid, d, s), "uuid": uuid, "duration": d, "start": s})
    return out


def _check_inputs(split: str, uuids: Sequence[str], rows: Mapping[str, int], passed: Mapping[str, bool],
                  prepared_sha256: Mapping[str, str]) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if len(set(uuids)) != len(uuids):
        raise ValueError(f"{split}: the split list has duplicate uuids")
    for name, table in (("rows", rows), ("passed", passed), ("prepared_sha256", prepared_sha256)):
        missing = [u for u in uuids if u not in table]
        if missing:
            raise ValueError(f"{split}: no {name} for {missing[:3]}")
    for u in uuids:
        if passed[u] and (type(rows[u]) is not int or rows[u] < 1):
            raise ValueError(f"{split}: episode {u} passed but has rows {rows[u]!r}")


def _donors(salt: str, split: str, windows: list[dict]) -> dict[str, str]:
    """每个时长 d：该时长的全部窗口按 hw("donor", split, id) 排成环；窗口 w 的替身是环上 w 之后第一个来自别的回合的窗口。"""
    donors = {}
    for d in sorted({w["duration"] for w in windows}):
        ring = sorted((w for w in windows if w["duration"] == d), key=lambda w: (hw(salt, "donor", split, w["id"]), w["id"]))
        n = len(ring)
        for i, w in enumerate(ring):
            for step in range(1, n):
                other = ring[(i + step) % n]
                if other["uuid"] != w["uuid"]:
                    donors[w["id"]] = other["id"]
                    break
            else:
                raise ValueError(f"{split}: duration {d} has windows from one episode only, window {w['id']} has no donor")
    return {w["id"]: donors[w["id"]] for w in windows}   # 按窗口顺序


def frozen_windows(split: str, uuids: Sequence[str], *, rows: Mapping[str, int], passed: Mapping[str, bool],
                   prepared_sha256: Mapping[str, str], splits_sha256: str, config: DroidConfig) -> dict:
    """06 §3：一个划分的冻结窗口文件内容。

    `uuids` 是划分列表（顺序即 `index/splits.json` 里的顺序）；`rows`、`passed`（= 准备 JSON 的 `passed_realized`）、
    `prepared_sha256`（准备 JSON 的文件 SHA）覆盖列表里的每条回合（含没通过的，SHA 照记，便于追溯）。
    """
    _check_inputs(split, uuids, rows, passed, prepared_sha256)
    salt = config.data.window_salt
    kept = [u for u in uuids if passed[u]]
    head = {"schema": SCHEMA, "split": split, "salt": salt, "splits_sha256": splits_sha256,
            "prepared_sha256": {u: prepared_sha256[u] for u in uuids}}
    counts = {"episodes": len(uuids), "passed": len(kept), "failed": len(uuids) - len(kept)}
    if split == VAL_SPLIT:
        durations = list(config.training.durations)
        windows = _windows(salt, split, kept, rows, durations, 1)
        counts |= {"windows": len(windows), "windows_by_duration": _by_duration(windows, durations)}
        return head | {"windows": windows, "counts": counts}

    durations = evaluation_durations(config)
    windows = _windows(salt, split, kept, rows, durations, config.evaluation.windows_per_duration)
    extra = [u for u in uuids[:config.encode.test_extra_episodes] if passed[u]]   # 跳过没通过的，不补位
    extra_set = set(extra)
    donors = _donors(salt, split, windows)

    distractors = config.evaluation.planning_candidates - 1   # 64 个候选 = 自己 + 63 个干扰项
    rank, cem = {}, []
    planning = [w for w in windows if w["uuid"] in extra_set and w["duration"] in config.evaluation.planning_durations]
    for d in config.evaluation.planning_durations:
        if d not in durations:
            raise ValueError(f"planning duration {d} is not an evaluation duration {durations}")
    for w in planning:
        others = [o for o in planning if o["duration"] == w["duration"] and o["uuid"] != w["uuid"]]
        others.sort(key=lambda o: (hw(salt, "distractor", split, w["id"], o["id"]), o["id"]))
        if len(others) < distractors:
            raise ValueError(f"{split}: window {w['id']} has only {len(others)} planning windows from other episodes, "
                             f"needs {distractors}")
        rank[w["id"]] = [o["id"] for o in others[:distractors]]
    for u in extra:
        for d in config.evaluation.planning_durations:
            first = next((w for w in planning if w["uuid"] == u and w["duration"] == d), None)   # 哈希顺序第 1 个
            if first is not None:
                cem.append(first["id"])

    fr_d, per = frame_rate_duration(config), config.time.frames_per_interval
    if fr_d not in durations:
        raise ValueError(f"frame-rate duration {fr_d} is not an evaluation duration {durations}")
    frame_rate = [{"id": frame_rate_id(w["uuid"], per*w["start"]), "uuid": w["uuid"], "start_frame": per*w["start"]}
                  for w in windows if w["uuid"] in extra_set and w["duration"] == fr_d]

    counts |= {"windows": len(windows), "windows_by_duration": _by_duration(windows, durations),
               "extra_episodes": len(extra), "donors": len(donors), "planning_rank": len(rank), "cem": len(cem),
               "frame_rate": len(frame_rate)}
    return head | {"extra_episodes": extra, "windows": windows, "donors": donors,
                   "planning": {"rank": rank, "cem": cem}, "frame_rate": frame_rate, "counts": counts}


def _by_duration(windows: list[dict], durations: Sequence[int]) -> dict[str, int]:
    return {str(d): sum(w["duration"] == d for w in windows) for d in durations}


# ---------------------------------------------------------------- 训练集和每一步的训练窗口（06 §1、§2，S07）

def train_episodes(splits: dict, prepared: Mapping, config: DroidConfig) -> list[str]:
    """06 §1：`splits.train_order(splits, config.data)` 的前 `train_episodes` 条，再去掉 `passed_<config.action.source>`
    不为真的回合，保持原顺序。`prepared` 是 {uuid: `prepare.PreparedEpisode`}（至少覆盖前 `train_episodes` 条）。"""
    source = config.action.source
    if source not in ACTION_SOURCES:
        raise ValueError(f"action.source must be one of {ACTION_SOURCES}")
    head = sp.train_order(splits, config.data)[:config.data.train_episodes]
    if len(head) < config.data.train_episodes:
        raise ValueError(f"data.train_episodes = {config.data.train_episodes} but the training order has only {len(head)}")
    missing = [u for u in head if u not in prepared]
    if missing:
        raise ValueError(f"no prepared episode for {missing[:3]}")
    return [u for u in head if prepared[u].passed(source) is True]


def list_sha256(ids: Sequence[str]) -> str:
    """一串编号（回合 uuid、一步的窗口编号）用 "\n" 连接后的 SHA-256：训练登记里训练集的 SHA（06 §1）、
    日志里每一步窗口编号列表的 SHA（06 §5）。"""
    if any(type(i) is not str or not i or "\n" in i for i in ids):
        raise ValueError("ids must be non-empty strings without newlines")
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def step_rng(seed: int, step: int, stream: int) -> np.random.Generator:
    """06 §2：第 `step` 步（从 0 计）、流 `stream` 的随机数发生器，只由这三个整数决定。"""
    for name, value in (("seed", seed), ("step", step), ("stream", stream)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, step, stream])))


def _episode_rows(episodes: Sequence[str], rows: Mapping[str, int], need: int) -> None:
    if not episodes:
        raise ValueError("no training episodes")
    for u in episodes:
        r = rows.get(u)
        if type(r) is not int or r < need + 1:
            raise ValueError(f"training episode {u} has rows {r!r}; needs at least {need + 1}")


def training_windows(episodes: Sequence[str], rows: Mapping[str, int], *, seed: int, step: int,
                     config: DroidConfig) -> list[dict]:
    """06 §2（stream 0）：第 `step` 步的 `training.batch` 行，逐行依次抽回合 e、时长 d ∈ `training.durations`、
    起点 s ∈ [0, F_e − 1 − d]。`rows[uuid]` 是这条回合的特征行数 F（准备 JSON 的 `rows`）。"""
    durations = list(config.training.durations)
    _episode_rows(episodes, rows, max(durations))
    rng = step_rng(seed, step, TRAIN_STREAM)
    out = []
    for _ in range(config.training.batch):
        uuid = episodes[int(rng.integers(len(episodes)))]
        d = durations[int(rng.integers(len(durations)))]
        s = int(rng.integers(rows[uuid] - d))
        out.append({"id": window_id(uuid, d, s), "uuid": uuid, "duration": d, "start": s})
    return out


def ac_clips(episodes: Sequence[str], rows: Mapping[str, int], *, seed: int, step: int,
             config: DroidConfig) -> list[dict]:
    """06 §2（stream 1）：第 `step` 步的 `training.ac_batch_clips` 个 `ac_clip_frames`（9）帧片段，逐个抽回合 e 和起点
    s ∈ [0, F_e − ac_clip_frames]（片段是第 s 到 s + 8 行）。"""
    frames = config.training.ac_clip_frames
    _episode_rows(episodes, rows, frames - 1)
    rng = step_rng(seed, step, AC_STREAM)
    out = []
    for _ in range(config.training.ac_batch_clips):
        uuid = episodes[int(rng.integers(len(episodes)))]
        s = int(rng.integers(rows[uuid] - (frames - 1)))
        out.append({"id": clip_id(uuid, s), "uuid": uuid, "start": s, "frames": frames})
    return out
