"""DROID 的批（docs/design/06-droid-windows-and-batches.md §4–§6，S07）。

把窗口（`windows.training_windows`、冻结的验证/测试窗口）组装成 v7 的 `training.TrainingBatch`，把 9 帧片段
（`windows.ac_clips`）组装成 AC 的片段批，并给训练任务的来源字典（§5）和按步号产出批的数据加载器（§6）。

分两半做：`load_windows` / `load_ac_clips` 读特征行（每行核 SHA）、取动作和状态、转 float32，全在 CPU 上，数据加载器
的工作进程里做；`assemble_training_batch` / `ac_clips_to` 只把张量搬到模型设备并构造 v7 的控制表示，在主进程做。
批的内容只由（种子，步号）决定，和工作进程数无关。

动作：第 j 段（j < d）是物理动作 `action_<source>[s + j]`（准备文件，01 §5.1），起止秒 `(s + j)·P`、`(s + j + 1)·P`，
P = `control_spec.control_period_seconds`（名义周期 1.0 秒）；本批最大时长 J 以外的 padding 段时间填 0、动作值填 NaN
（v7 `from_physical_segments` 只读有效段）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch

from po_jepa.actions import ControlSpec, from_physical_segments
from po_jepa.droid import encode as en
from po_jepa.droid import features as ft
from po_jepa.droid import windows as wn
from po_jepa.droid.config import ACTION_SOURCES, DroidConfig
from po_jepa.geometry import build_spatial_transfer, regular_grid
from po_jepa.runtime import cache, training

GRID = 16                      # 特征网格 16×16（06 §4 transfer）
CONSISTENCY_GRID = 8           # 空间一致性的目标网格 8×8（和 v7 训练相同）
EVAL_BATCH = 64                # 06 §4：冻结窗口按文件顺序每 64 个一批
WORKERS = 2                    # 06 §6：2 个工作进程，预取 4 步
PREFETCH = 4
TRAINING_ROLE = "droid_training"
SYNTHETIC_SPLIT = "development_smoke"
PAPER_SHA256 = "d851778ecf4a1f160da9c681a831b3bc9333dae2b60eaeefa6090c0af44aaff1"
V7_SOURCES = tuple(f"src/po_jepa/{n}.py" for n in ("actions", "dynamics", "geometry", "losses", "model", "spectral")) \
    + ("src/po_jepa/runtime/training.py",)


# ---------------------------------------------------------------- 每条回合要用的数组

@dataclass(frozen=True)
class EpisodeArrays:
    """一条回合在批里要用的数：特征行数 F、准备 JSON 的 SHA、动作源、物理动作 float64 [F−1, 7]、状态 float64 [F, 7]。"""

    uuid: str
    rows: int
    prepared_sha256: str
    source: str
    actions: np.ndarray
    states: np.ndarray


def episode_arrays(prepared: Mapping, uuids: Sequence[str], source: str) -> dict[str, EpisodeArrays]:
    """{uuid: `prepare.PreparedEpisode`} → {uuid: EpisodeArrays}，只取 `uuids`（训练集或冻结窗口的回合）。
    要求这些回合在 `source` 下通过检查（训练集已按 06 §1 筛过，冻结窗口只含 `passed_realized`）。"""
    if source not in ACTION_SOURCES:
        raise ValueError(f"source must be one of {ACTION_SOURCES}")
    out = {}
    for u in uuids:
        if u in out:
            continue
        p = prepared.get(u)
        if p is None:
            raise ValueError(f"no prepared episode for {u}")
        if p.passed(source) is not True:
            raise ValueError(f"episode {u} did not pass the prepare checks for action source {source!r}")
        rows = p.rows
        actions, states = p.arrays[f"action_{source}"], p.arrays["state"]
        if (type(rows) is not int or actions.dtype != np.float64 or states.dtype != np.float64
                or actions.shape != (rows - 1, 7) or states.shape != (rows, 7)):
            raise ValueError(f"episode {u}: prepared arrays do not match rows {rows}")
        out[u] = EpisodeArrays(u, rows, p.json_sha256, source, actions, states)
    return out


def _episode(uuid: str, reader: ft.FeatureReader, episodes: Mapping[str, EpisodeArrays]) -> EpisodeArrays:
    """特征和动作出自同一份准备文件：特征 JSON 记的准备 JSON SHA 和行数等于这里的。"""
    ep = episodes.get(uuid)
    if ep is None:
        raise ValueError(f"no prepared arrays for {uuid}")
    record = reader.record(uuid)
    if record.get("prepared_sha256") != ep.prepared_sha256 or record.get("rows") != ep.rows:
        raise ValueError(f"episode {uuid}: the features were encoded from another prepared file (SHA or rows differ)")
    return ep


def _int(value, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an int, got {value!r}")
    return value


# ---------------------------------------------------------------- 窗口 → TrainingBatch（§4）

@dataclass(frozen=True)
class LoadedWindows:
    """`load_windows` 的结果（CPU）：窗口编号、起点 s 和时长 d（int64 [B]）、起点行和终点行的特征 float32 [B, 256, 1408]、
    物理动作 float32 [B, J, 7]（padding 为 NaN）、起点行的真实状态 float64 [B, 7]（物理单位）。"""

    sample_ids: tuple[str, ...]
    starts: torch.Tensor
    durations: torch.Tensor
    raw_z: torch.Tensor
    raw_target: torch.Tensor
    actions: torch.Tensor
    start_states: torch.Tensor


def load_windows(windows: Sequence[Mapping], reader: ft.FeatureReader,
                 episodes: Mapping[str, EpisodeArrays]) -> LoadedWindows:
    """读每个窗口 {id, uuid, duration, start} 的第 s 行和第 s + d 行（逐行核 SHA）、动作 `action[s : s + d]`、状态 `state[s]`。"""
    if not windows:
        raise ValueError("no windows")
    sources = {ep.source for ep in episodes.values()}
    if len(sources) != 1:
        raise ValueError(f"episode arrays mix action sources {sorted(sources)}")
    width = max(_int(w["duration"], "duration") for w in windows)
    ids, starts, durations, rows, acts, states = [], [], [], [], [], []
    for w in windows:
        uuid, d, s = w["uuid"], _int(w["duration"], "duration"), _int(w["start"], "start")
        if w["id"] != wn.window_id(uuid, d, s):
            raise ValueError(f"window id {w['id']!r} does not match its uuid, duration and start")
        ep = _episode(uuid, reader, episodes)
        if d < 1 or s < 0 or s + d > ep.rows - 1:
            raise ValueError(f"window {w['id']}: needs rows {s} and {s + d} of an episode with {ep.rows} rows")
        rows.append(reader.rows(uuid, [s, s + d]))
        a = np.full((width, 7), np.nan, dtype=np.float32)
        a[:d] = ep.actions[s:s + d].astype(np.float32)
        acts.append(a)
        states.append(ep.states[s])
        ids.append(w["id"]); starts.append(s); durations.append(d)
    pair = torch.stack(rows)                                   # [B, 2, N, E]
    return LoadedWindows(tuple(ids), torch.tensor(starts, dtype=torch.int64), torch.tensor(durations, dtype=torch.int64),
                         pair[:, 0].contiguous(), pair[:, 1].contiguous(), torch.from_numpy(np.stack(acts)),
                         torch.from_numpy(np.stack(states).astype(np.float64)))


def spatial_transfer():
    """和 v7 训练相同：16×16 规则网格到 8×8 的固定 R。"""
    return build_spatial_transfer(regular_grid(GRID, GRID), target_height=CONSISTENCY_GRID, target_width=CONSISTENCY_GRID)


def window_controls(starts: torch.Tensor, durations: torch.Tensor, actions: torch.Tensor, control_spec: ControlSpec,
                    sample_ids: tuple[str, ...], device):
    """06 §4 的控制列：第 j 段（j < d）起止秒 (s + j)·P、(s + j + 1)·P，物理动作 `actions[:, j]`（float32 [B, J, m]，
    padding 段不读），`origin_seconds = s·P`，用 v7 `from_physical_segments` 构造。返回 (controls, 起点秒, 终点秒)，
    时间都是 CPU float64；终点秒 (s + d)·P 必须逐位等于 s·P + d·P。离线规划（11 §1）的候选动作也走这里。"""
    if control_spec.action_dim != actions.shape[-1]:
        raise ValueError(f"control_spec has {control_spec.action_dim} channels, the actions have {actions.shape[-1]}")
    period = control_spec.control_period_seconds
    width = actions.shape[1]
    s = starts.to(torch.float64)
    d = durations.to(torch.float64)
    j = torch.arange(width, dtype=torch.float64)
    valid = torch.arange(width)[None, :] < durations[:, None]
    starts_seconds = torch.where(valid, (s[:, None] + j[None, :])*period, torch.zeros((), dtype=torch.float64))
    ends_seconds = torch.where(valid, (s[:, None] + j[None, :] + 1)*period, torch.zeros((), dtype=torch.float64))
    origin = s*period
    controls = from_physical_segments(
        starts_seconds=starts_seconds, ends_seconds=ends_seconds,
        u_values=actions.to(device), n_segments=durations.clone(), segment_valid=valid,
        origin_seconds=origin, control_spec=control_spec, source_ids=sample_ids)
    target = (starts + durations).to(torch.float64)*period
    if not torch.equal(target, origin + d*period):
        raise ValueError(f"window endpoints are not exactly representable with period {period}")
    return controls, origin, target


def assemble_training_batch(loaded: LoadedWindows, control_spec: ControlSpec, provenance: dict,
                            device) -> training.TrainingBatch:
    """06 §4 的表：特征搬到 `device`，控制用 v7 `from_physical_segments` 构造（`window_controls`），时间都是 CPU float64。"""
    controls, origin, target = window_controls(loaded.starts, loaded.durations, loaded.actions, control_spec,
                                               loaded.sample_ids, device)
    d = loaded.durations.to(torch.float64)
    return training.TrainingBatch(
        raw_z=loaded.raw_z.to(device), raw_target=loaded.raw_target.to(device), controls=controls,
        duration=d, tau=d/2, sigma=d/2, transfer=spatial_transfer(),
        observed_time_seconds=origin.clone(), target_time_seconds=target,
        sample_ids=loaded.sample_ids, provenance=provenance)


def make_training_batch(windows: Sequence[Mapping], reader: ft.FeatureReader, episodes: Mapping[str, EpisodeArrays],
                        control_spec: ControlSpec, provenance: dict, device) -> training.TrainingBatch:
    """06 §4：`assemble_training_batch(load_windows(…))`。"""
    return assemble_training_batch(load_windows(windows, reader, episodes), control_spec, provenance, device)


def make_eval_batch(windows: Sequence[Mapping], reader: ft.FeatureReader, episodes: Mapping[str, EpisodeArrays],
                    control_spec: ControlSpec, provenance: dict, device) -> tuple[training.TrainingBatch, torch.Tensor]:
    """验证和测试批：组装同训练批，另返回每个窗口起点行的真实状态 `state[s]`（CPU float64 [B, 7]，物理单位，只给 AC 的
    官方输入模式用，09 §6.2）。"""
    loaded = load_windows(windows, reader, episodes)
    return assemble_training_batch(loaded, control_spec, provenance, device), loaded.start_states


def chunks(windows: Sequence[Mapping], size: int = EVAL_BATCH) -> list[list[Mapping]]:
    """冻结窗口按文件顺序每 `size` 个一批（最后一批可以更少）。"""
    if type(size) is not int or size < 1:
        raise ValueError("size must be a positive int")
    return [list(windows[i:i + size]) for i in range(0, len(windows), size)]


# ---------------------------------------------------------------- AC 片段（§4）

@dataclass(frozen=True)
class ACClipBatch:
    """AC 的片段批：特征 float32 [C, 9, 256, 1408]、物理动作 float32 [C, 8, 7]（第 k 个是 `action[s + k]`）、
    真实状态 float32 [C, 9, 7]（`state[s : s + 9]`，`ac_state: true_state` 时用）。"""

    sample_ids: tuple[str, ...]
    features: torch.Tensor
    actions: torch.Tensor
    states: torch.Tensor


def load_ac_clips(clips: Sequence[Mapping], reader: ft.FeatureReader,
                  episodes: Mapping[str, EpisodeArrays]) -> ACClipBatch:
    """读每个片段 {id, uuid, start, frames} 的第 s … s + frames − 1 行（逐行核 SHA）、动作和状态（CPU）。"""
    if not clips:
        raise ValueError("no clips")
    ids, feats, acts, states = [], [], [], []
    frames = {_int(c["frames"], "frames") for c in clips}
    if len(frames) != 1 or min(frames) < 2:
        raise ValueError(f"clips must share one length >= 2, got {sorted(frames)}")
    n = frames.pop()
    for c in clips:
        uuid, s = c["uuid"], _int(c["start"], "start")
        if c["id"] != wn.clip_id(uuid, s):
            raise ValueError(f"clip id {c['id']!r} does not match its uuid and start")
        ep = _episode(uuid, reader, episodes)
        if s < 0 or s + n > ep.rows:
            raise ValueError(f"clip {c['id']}: rows {s}…{s + n - 1} outside an episode with {ep.rows} rows")
        feats.append(reader.rows(uuid, list(range(s, s + n))))
        acts.append(ep.actions[s:s + n - 1].astype(np.float32))
        states.append(ep.states[s:s + n].astype(np.float32))
        ids.append(c["id"])
    return ACClipBatch(tuple(ids), torch.stack(feats), torch.from_numpy(np.stack(acts)), torch.from_numpy(np.stack(states)))


def ac_clips_to(batch: ACClipBatch, device) -> ACClipBatch:
    return ACClipBatch(batch.sample_ids, batch.features.to(device), batch.actions.to(device), batch.states.to(device))


def make_ac_clip_batch(clips: Sequence[Mapping], reader: ft.FeatureReader, episodes: Mapping[str, EpisodeArrays],
                       device) -> ACClipBatch:
    """06 §4：AC 片段批，放到 `device`。动作源由 `episodes`（`episode_arrays(…, source)`）决定。"""
    return ac_clips_to(load_ac_clips(clips, reader, episodes), device)


# ---------------------------------------------------------------- 来源（§5）

def droid_sources() -> list[str]:
    """`src/po_jepa/droid/` 下的全部 .py（按名字排序），都记进来源。"""
    base = cache.ROOT/"src"/"po_jepa"/"droid"
    return [f"src/po_jepa/droid/{p.name}" for p in sorted(base.glob("*.py"))]


def training_provenance(config: DroidConfig, manifests: Mapping[str, dict], feature_manifest_sha256: str) -> dict:
    """06 §5 的来源字典（每个训练任务开始时算一次，之后每批共用）。`manifests` 是训练用到的片清单
    （`features.feature_manifests` 的第一项），它们的编码器身份必须相同；假编码器的特征只能跑冒烟（split =
    development_smoke，四个 SHA 为 None）。"""
    idents = [m.get("encoder") for m in manifests.values()]
    if not idents or any(i != idents[0] for i in idents):
        raise ValueError("feature manifests must share one encoder identity")
    ident = idents[0]
    sources = {name: cache.file_sha(cache.ROOT/name) for name in (*V7_SOURCES, *droid_sources())}
    head = {"paper_sha256": PAPER_SHA256, "upstream_commit": config.encode.upstream_commit,
            "source_files_sha256": sources}
    if ident.get("evidence_role") == en.SYNTHETIC_ROLE:
        value = {"evidence_role": en.SYNTHETIC_ROLE, "split": SYNTHETIC_SPLIT, **head,
                 "encoder_checkpoint_sha256": None, "preprocess_sha256": None,
                 "feature_manifest_sha256": None, "protocol_sha256": None}
    elif ident.get("evidence_role") == en.REAL_ROLE:
        if ident.get("preprocess_sha256") != cache.v7_preprocess_sha256():
            raise ValueError("the main features were not made with the v7 preprocessing")
        value = {"evidence_role": TRAINING_ROLE, "split": "train", **head,
                 "encoder_checkpoint_sha256": ident.get("encoder_checkpoint_sha256"),
                 "preprocess_sha256": cache.v7_preprocess_sha256(),
                 "feature_manifest_sha256": feature_manifest_sha256, "protocol_sha256": config.config_sha256}
    else:
        raise ValueError(f"unknown encoder evidence role {ident.get('evidence_role')!r}")
    return training._provenance(value)


# ---------------------------------------------------------------- 按步号产出批的数据加载器（§6）

class StepBatches(torch.utils.data.IterableDataset):
    """第 `start` … `stop − 1` 步的批（`kind` = "train"：stream 0 的训练窗口，`LoadedWindows`；"ac"：stream 1 的片段，
    `ACClipBatch`），产出 (步号, 批)。每个工作进程自己开一个 `FeatureReader`，第 w 个（共 n 个）只产出步号 ≡ start + w
    (mod n) 的步，`DataLoader` 按工作进程轮流取，所以产出顺序就是步号顺序，内容和工作进程数无关。"""

    def __init__(self, kind: str, data_root: Path | str, manifests: Mapping[str, dict], train_set: Sequence[str],
                 episodes: Mapping[str, EpisodeArrays], *, seed: int, start: int, stop: int, config: DroidConfig) -> None:
        super().__init__()
        if kind not in ("train", "ac"):
            raise ValueError("kind must be 'train' or 'ac'")
        if type(start) is not int or type(stop) is not int or not 0 <= start <= stop:
            raise ValueError("need 0 <= start <= stop")
        missing = [u for u in train_set if u not in episodes]
        if missing:
            raise ValueError(f"no episode arrays for {missing[:3]}")
        self.kind, self.data_root, self.manifests = kind, Path(data_root), dict(manifests)
        self.train_set, self.episodes = list(train_set), dict(episodes)
        self.rows = {u: self.episodes[u].rows for u in self.train_set}
        self.seed, self.start, self.stop, self.config = seed, start, stop, config

    def batch(self, step: int, reader: ft.FeatureReader):
        if self.kind == "train":
            w = wn.training_windows(self.train_set, self.rows, seed=self.seed, step=step, config=self.config)
            return load_windows(w, reader, self.episodes)
        c = wn.ac_clips(self.train_set, self.rows, seed=self.seed, step=step, config=self.config)
        return load_ac_clips(c, reader, self.episodes)

    def __iter__(self) -> Iterator:
        info = torch.utils.data.get_worker_info()
        worker, count = (0, 1) if info is None else (info.id, info.num_workers)
        reader = ft.FeatureReader(self.data_root, self.manifests)
        for step in range(self.start + worker, self.stop, count):
            yield step, self.batch(step, reader)


def loader(dataset: StepBatches, *, workers: int = WORKERS, prefetch: int = PREFETCH) -> torch.utils.data.DataLoader:
    """06 §6：`workers` 个工作进程、每个预取 `prefetch` 步；不自动拼批（每一项已经是一整步）。"""
    if type(workers) is not int or workers < 0:
        raise ValueError("workers must be a non-negative int")
    extra = {"prefetch_factor": prefetch} if workers else {}
    return torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers, **extra)


def training_batches(dataset: StepBatches, control_spec: ControlSpec, provenance: dict, device, *,
                     workers: int = WORKERS, prefetch: int = PREFETCH) -> Iterator[tuple[int, object]]:
    """主进程这一侧：按步号顺序产出 (步号, 放到 `device` 的批)，训练窗口是 `TrainingBatch`，AC 是 `ACClipBatch`；
    核对加载器给的步号正好是 start、start + 1、……（09 §4.3）。"""
    expected = dataset.start
    for step, item in loader(dataset, workers=workers, prefetch=prefetch):
        step = int(step)
        if step != expected:
            raise RuntimeError(f"data loader produced step {step}, expected {expected}")
        expected += 1
        if dataset.kind == "train":
            yield step, assemble_training_batch(item, control_spec, provenance, device)
        else:
            yield step, ac_clips_to(item, device)
    if expected != dataset.stop:
        raise RuntimeError(f"data loader stopped at step {expected}, expected {dataset.stop}")


def batch_sha256(item) -> str:
    """一批（`LoadedWindows`、`ACClipBatch`）的全部张量字节和编号的 SHA-256，测试和 `check/env` 比较批用。"""
    h = hashlib.sha256()
    h.update("\n".join(item.sample_ids).encode("utf-8"))
    for name in sorted(vars(item)):
        value = getattr(item, name)
        if isinstance(value, torch.Tensor):
            h.update(name.encode("ascii")); h.update(str(value.dtype).encode("ascii"))
            h.update(str(list(value.shape)).encode("ascii")); h.update(value.contiguous().cpu().numpy().tobytes())
    return h.hexdigest()
