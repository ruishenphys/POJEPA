"""冻结特征窗口组批，不改变模型、动作或损失公式。

raw Z/target 为 FP32 [B,256,1408]，控制为模型设备 FP32；duration/
tau/sigma 为 CPU float64 周期，观测/目标端点为物理秒。显式读取才
访问缓存；真实读取入口拒绝 synthetic 替身。
"""

from dataclasses import dataclass

from pathlib import Path

import hashlib

import json

import torch

import torch.nn.functional as F

from . import cache as cache
from . import training as training
from po_jepa.actions import ActionSegments, validate_action_segments

from po_jepa.simulation import split_for

ROOT = Path(__file__).resolve().parents[3]

def consumer_sources() -> dict:
    """当前消费者身份，与 generation 内的历史生成源码分别保存。"""
    names = tuple("src/po_jepa/"+name+".py" for name in
                  ("actions", "dynamics", "geometry", "losses", "model", "spectral", "vjepa", "simulation", "metrics"))
    names += tuple("src/po_jepa/runtime/"+name for name in
                   ("training.py", "cache.py", "data.py", "checks.py"))
    return cache._source_hashes(names)

def approved_cache_metadata(root: Path) -> dict:
    """核已封存的整套冻结缓存摘要，不重新编码或复跑数值门槛。"""
    record_path = ROOT/"reference/cache_asset.json"
    accepted = json.loads(record_path.read_bytes())
    if not accepted["approved"]:
        raise ValueError("F02 must be accepted before F03 cache use")
    report_path = Path(root)/"report.json"
    expected = accepted["encode_report_sha256"]
    if cache.file_sha(report_path) != expected:
        raise ValueError("approved F02 encode report SHA mismatch")
    complete = json.loads((Path(root)/"COMPLETE.json").read_bytes())
    report = json.loads(report_path.read_bytes())
    if complete["report_sha256"] != expected or report["status"] != "passed" or report["mode"] != "encode":
        raise ValueError("approved F02 root completion mismatch")
    records = report["result"]["records"]
    manifests = {row["trajectory_id"]: row["manifest_sha256"] for row in records}
    if len(records) != 200 or set(manifests) != set(range(200)):
        raise ValueError("approved F02 trajectory manifest inventory incomplete")
    return dict(accepted_record_sha256=cache.file_sha(record_path), encode_report_sha256=expected,
                manifest_sha256_by_id=manifests)

@dataclass(frozen=True)
class WindowQuery:
    """显式轨迹/起点/时长；时长是控制周期，不是秒或隐藏系统状态。"""
    trajectory_id: int
    start_tick: int
    duration_cycles: float

    def __post_init__(self):
        """仅允许 train/validation 固定整数查询，不扩展到留出时长。"""
        if type(self.trajectory_id) is not int or not 0 <= self.trajectory_id < 200:
            raise ValueError("trajectory ID must be a native integer in [0,199]")
        if type(self.start_tick) is not int or not 0 <= self.start_tick <= 20:
            raise ValueError("start tick must be a native integer in [0,20]")
        if type(self.duration_cycles) not in (int, float) or self.duration_cycles not in (1., 2., 4., 8.):
            raise ValueError("F03 query duration must be 1/2/4/8 control cycles")

def stack_controls(rows: list[ActionSegments]) -> ActionSegments:
    """沿 B 轴补齐变长控制，零 padding 不参与有效控制。

    每行原为 [1,J,2]，仅复制有效段，保留幅值梯度、路径 ID 和物理原点。
    变长组装保留循环，不拼接时间或重新缩放幅值。
    """
    if not rows:
        raise ValueError("control batch cannot be empty")
    for row in rows:
        validate_action_segments(row, query_duration=row.duration, require_exact_end=True)
        if row.duration.shape != (1,) or row.control_spec != rows[0].control_spec:
            raise ValueError("batch rows require one path and identical ControlSpec")
        if row.u_values.device != rows[0].u_values.device or row.u_values.dtype != torch.float32:
            raise ValueError("control rows must share model device and float32 dtype")
    counts = torch.cat([r.n_segments for r in rows])
    width = max(1, int(counts.max()))
    lengths = [int(r.n_segments[0]) for r in rows]
    starts = torch.cat([F.pad(r.starts[:, :n], (0, width-n)) for r, n in zip(rows, lengths)])
    ends = torch.cat([F.pad(r.ends[:, :n], (0, width-n)) for r, n in zip(rows, lengths)])
    values = torch.cat([F.pad(r.u_values[:, :n], (0, 0, 0, width-n)) for r, n in zip(rows, lengths)])
    result = ActionSegments(starts, ends, values, counts,
        torch.arange(width)[None, :] < counts[:, None], torch.cat([r.source_origin_seconds for r in rows]),
        torch.cat([r.duration for r in rows]), rows[0].control_spec, tuple(r.source_ids[0] for r in rows))
    validate_action_segments(result, query_duration=result.duration, require_exact_end=True)
    return result

def _window_boundary(window, first) -> None:
    """在训练前拒绝错网格/非有限/上游图/非精确时间，不自动detach或修端点。"""
    x, y = window.inputs, window.target
    x.grid.__post_init__()
    if (x.grid.height, x.grid.width) != (16, 16) or not torch.equal(x.grid.coords, first.inputs.grid.coords):
        raise ValueError("F03 rows require the identical canonical 16x16 grid")
    for value in (x.raw_z, y.raw_z):
        if (value.shape != (1, 256, 1408) or value.dtype != torch.float32
            or value.device != first.inputs.raw_z.device or torch.is_inference(value)
            or value.requires_grad or value.grad_fn is not None or not torch.isfinite(value).all()):
            raise ValueError("frozen raw rows must be ordinary finite model float32[1,256,1408]")
    for value in (x.duration, x.observed_time_seconds, y.target_time_seconds):
        if (value.shape != (1,) or value.device.type != "cpu" or value.dtype != torch.float64
            or value.requires_grad or torch.is_inference(value) or not torch.isfinite(value).all()):
            raise ValueError("query times must be ordinary finite CPU float64[1]")
    expected = x.observed_time_seconds + x.duration*x.controls.control_spec.control_period_seconds
    if not torch.equal(expected, y.target_time_seconds) or not torch.equal(x.observed_time_seconds, x.controls.source_origin_seconds):
        raise ValueError("physical endpoint/origin must exactly match query cycles")

def _row_record(window) -> dict:
    """保留逐行真实来源；generation 是历史生成记录，不改写为当前训练源码。"""
    x, y = window.inputs, window.target
    period = x.controls.control_spec.control_period_seconds
    tick = float(x.observed_time_seconds[0])/period
    if not tick.is_integer() or not 0 <= tick <= 20:
        raise ValueError("window observation must be an allowed integer tick")
    return dict(trajectory_id=x.trajectory_id, split=x.split, start_tick=int(tick),
        target_tick=int(tick+float(x.duration[0])), duration_cycles=float(x.duration[0]),
        observed_time_seconds=float(x.observed_time_seconds[0]), target_time_seconds=float(y.target_time_seconds[0]),
        control_source_ids=list(x.controls.source_ids), source_origin_seconds=float(x.controls.source_origin_seconds[0]),
        cache_manifest_sha256=window.provenance["actual_manifest_sha256"],
        preprocess_sha256=window.provenance["preprocess_sha256"],
        start_image_sha256=x.image_sha256, target_image_sha256=y.image_sha256,
        generation=window.provenance["generation"])

def _batch_provenance(windows, fields) -> dict:
    """集合哈希引用每行，而非拿第一行冒充整批；synthetic仍不声称真实工件。"""
    provenance = dict(fields[0]["provenance"])
    common = ("evidence_role", "split", "paper_sha256", "upstream_commit", "encoder_checkpoint_sha256",
              "protocol_sha256", "source_files_sha256")
    if any(any(f["provenance"][key] != provenance[key] for key in common) for f in fields):
        raise ValueError("batch common source identities/roles must agree")
    rows = [_row_record(w) for w in windows]
    collections = dict(feature_windows=dict(schema="F03_ordered_window_collection_v1", rows=rows),
        preprocess=dict(schema="F03_ordered_preprocess_collection_v1", rows=[
            {k: r[k] for k in ("trajectory_id", "start_tick", "target_tick", "preprocess_sha256",
                              "start_image_sha256", "target_image_sha256")} for r in rows]))
    training._plain(collections)
    digests = {key: hashlib.sha256(cache._json_bytes(value)).hexdigest() for key, value in collections.items()}
    provenance.update(f03_collections=collections, f03_collection_sha256=digests)
    if provenance["evidence_role"] == "real_frozen_features":
        provenance.update(preprocess_sha256=digests["preprocess"], feature_manifest_sha256=digests["feature_windows"])
    provenance["source_files_sha256"] = consumer_sources()
    training._provenance(provenance)
    return provenance

def collate_windows(windows) -> training.TrainingBatch:
    """独立窗口→TrainingBatch；只沿B轴拼接，T=tau+sigma逐位保持。

    此纯组装接口允许明确synthetic的CPU测试替身；真实入口另拒绝synthetic。
    raw、控制、物理端点和逐行来源均保留，目标仅进入标签字段。
    """
    if not windows:
        raise ValueError("window batch cannot be empty")
    fields = []
    for window in windows:
        _window_boundary(window, windows[0])
        fields.append(cache.training_batch_fields(window))
    merged = {key: torch.cat([f[key] for f in fields]) for key in ("raw_z", "raw_target", "duration", "tau", "sigma",
                                                                "observed_time_seconds", "target_time_seconds")}
    return training.TrainingBatch(**merged, controls=stack_controls([f["controls"] for f in fields]),
        transfer=fields[0]["transfer"], sample_ids=tuple(s for f in fields for s in f["sample_ids"]),
        provenance=_batch_provenance(windows, fields))

def load_windows(root: Path, queries, *, split: str, device: str):
    """只从真实canonical完整缓存读显式窗口；先拒绝跨split，不访问未来作输入。"""
    if split not in ("train", "validation") or not queries:
        raise ValueError("F03 accepts nonempty train/validation queries only")
    if any(not isinstance(q, WindowQuery) or split_for(q.trajectory_id) != split for q in queries):
        raise ValueError("cross-split query refused before cache reads")
    approved = approved_cache_metadata(root)
    windows = [cache.read_window(Path(root)/f"{q.trajectory_id:04d}", start_tick=q.start_tick,
        duration_cycles=q.duration_cycles, expected_split=split, device=device) for q in queries]
    for q, w in zip(queries, windows):
        if (w.provenance["generation"]["evidence_role"] != "real_frozen_encoder"
            or w.provenance["observation_variant"] != "canonical"):
            raise ValueError("real canonical frozen features required; synthetic refused")
        if (w.inputs.trajectory_id != q.trajectory_id or w.inputs.split != split
            or w.provenance["actual_manifest_sha256"] != approved["manifest_sha256_by_id"][q.trajectory_id]
            or float(w.inputs.duration[0]) != q.duration_cycles
            or float(w.inputs.observed_time_seconds[0]) != q.start_tick*.25
            or float(w.target.target_time_seconds[0]) != (q.start_tick+q.duration_cycles)*.25):
            raise ValueError("window identity/endpoint does not match approved F02 query")
    return windows

def load_training_batch(root: Path, queries, *, device: str) -> training.TrainingBatch:
    """先核完整冻结缓存来源再组批，不接受 synthetic 冒充真实编码。"""
    return collate_windows(load_windows(root, queries, split="train", device=device))

def write_collections(batch: training.TrainingBatch, directory: Path) -> dict:
    """写确定JSON原字节并核摘要；同名已存在须完全一致，不能覆盖损坏证据。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, record in batch.provenance["f03_collections"].items():
        expected = batch.provenance["f03_collection_sha256"][kind]
        if hashlib.sha256(cache._json_bytes(record)).hexdigest() != expected:
            raise ValueError("collection content/hash mismatch")
        path = directory/f"{kind}-{expected}.json"
        if not path.exists():
            cache._write_json(path, record)
        if cache.file_sha(path) != expected:
            raise ValueError("existing collection differs; original file preserved")
        paths[kind] = path
    return paths

