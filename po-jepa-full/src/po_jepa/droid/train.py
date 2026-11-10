"""DROID 训练任务 `train/<条件>/seed<种子>`（docs/design/09-droid-training.md §4、§5，S09a）。

目录 `<run-root>/<name>/train/<条件>/seed<种子>/`：`registration.json`（§4.2）、`checkpoints/`（`index.json`、
`step-<6 位>.pt` 验证点、`resume-<6 位>.pt` 续跑）、`validations/step-<6 位>.json`、`selection.json`、`train-log.jsonl`
（§5）、`starts.jsonl`（每次启动一行，算续跑次数，S09a 加）、`early-check.json`、`report.json`、`COMPLETE.json`、
出错时 `FAILED.json`。

主循环照 §4.3：完成了就退出 0；FAILED.json 在而没有 `--retry` 退出 1；登记对不上退出 2；从 `index.json` 里步数最大的
检查点接着跑（同一步取验证点的），没有就在第 0 步验证并存 step-000000；删孤立文件、去掉日志里末步大于检查点步数的行；
每一步：取批（核步号）、单步、一致性占比（每 U/400 步）、写日志（每 100 步一行）、提前检查（§4.6，不过退出 3）、
每 V = U/20 步或最后一步验证并存验证点，否则距上次存档 ≥ checkpoint_minutes、收到 SIGTERM/SIGUSR1 或离时长上限不到
stop_margin_minutes 时存续跑检查点；收到信号或快到上限就存档后退出 75。最后选点、写 report.json 和 COMPLETE.json。
批只由（种子，步号）决定，检查点恢复随机状态，所以在 CPU 上续跑和不中断的结果逐位相同（§8）。

  python -m po_jepa.droid.train --config … --data-root … --run-root … --job train/<条件>/seed<种子> \\
      [--time-limit-minutes N] [--retry] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

from po_jepa.actions import ControlSpec
from po_jepa.droid import ac as droid_ac
from po_jepa.droid import batches as bt
from po_jepa.droid import checkpoints as ck
from po_jepa.droid import conditions as cd
from po_jepa.droid import evaluate as ev
from po_jepa.droid import features as ft
from po_jepa.droid import index as ix
from po_jepa.droid import prepare as pr
from po_jepa.droid import splits as sp
from po_jepa.droid import windows as wn
from po_jepa.droid.actions import droid_control_spec
from po_jepa.droid.config import CONDITIONS, DroidConfig, load_config
from po_jepa.droid.spectral_gap import read_sigma_z
from po_jepa.losses import jepa_distance
from po_jepa.runtime import cache, training

REGISTRATION_SCHEMA = "droid_train_registration_v1"
VALIDATION_SCHEMA = "droid_train_validation_v1"
SELECTION_SCHEMA = "droid_train_selection_v1"
REPORT_SCHEMA = "droid_train_report_v1"
LOG_FILE = "train-log.jsonl"
STARTS_FILE = "starts.jsonl"
EXIT_DONE, EXIT_FAILED, EXIT_REGISTRATION, EXIT_WEAK, EXIT_REQUEUE = 0, 1, 2, 3, 75   # 09 §4.7
OFFICIAL_AC_FILE = "vjepa2-ac-vitg.pt"            # 12 §1：<数据根>/checkpoints/
RETAIN_TWO = ("vjepa2_ac",)                       # 09 §4.4：只留最优和最近一个验证点（约 3.6 GB 一个）


class ConsistencyTooWeak(RuntimeError):
    """09 §4.6：提前检查时一致性占比中位数低于 min_share。"""


class RegistrationMismatch(RuntimeError):
    """09 §4.2：已有 registration.json 和按当前代码、配置重算的不同。"""


# ---------------------------------------------------------------- 输入

@dataclass(frozen=True)
class TrainData:
    """训练任务读的数据（`load_data` 从数据根读；测试直接构造）。`reader` 读验证窗口，训练批由 `StepBatches` 在
    工作进程里各开一个 `FeatureReader`。"""

    data_root: Path
    manifests: dict
    feature_manifest_sha256: str
    train_set: list
    episodes: dict
    val_windows: list
    val_episodes: dict
    val_skipped: list
    control_spec: ControlSpec
    provenance: dict
    action_scale_sha256: str
    val_windows_sha256: str
    reader: ft.FeatureReader


@dataclass(frozen=True)
class Calibration:
    """σ_z（`sigma_z.json`）和 λ（`lambda-<条件>.json`，只有三个带一致性的条件有）的值和文件 SHA。"""

    sigma_z: float
    sigma_z_sha256: str
    weights: dict | None
    lambda_sha256: str | None


def read_lambda(path: Path, config: DroidConfig, sigma_z_sha256: str) -> tuple[dict, str]:
    """`lambda-<条件>.json`（09 §3.2）的 time_weight、space_weight（有限正数）和文件 SHA；记了 config_sha256、
    sigma_z_sha256 时必须等于本配置的和本次读到的 σ_z 文件的。"""
    path = Path(path)
    doc = ix._loads(path.read_bytes(), str(path))
    if not isinstance(doc, dict):
        raise ValueError(f"{path} is not a JSON object")
    weights = {k: doc.get(k) for k in ("time_weight", "space_weight")}
    for k, v in weights.items():
        if type(v) is not float or not math.isfinite(v) or v <= 0:
            raise ValueError(f"{path}: {k} must be a positive finite float, got {v!r}")
    if "config_sha256" in doc and doc["config_sha256"] != config.config_sha256:
        raise ValueError(f"{path} was calibrated under another config")
    if "sigma_z_sha256" in doc and doc["sigma_z_sha256"] != sigma_z_sha256:
        raise ValueError(f"{path} was calibrated with another sigma_z.json")
    return weights, cache.file_sha(path)


def load_calibration(run_root: Path | str, config: DroidConfig, condition: str) -> Calibration:
    directory = Path(run_root)/config.name/"calibration"
    sigma_z, sigma_sha = read_sigma_z(directory/"sigma_z.json", config)
    weights, lam_sha = None, None
    if condition in cd.LAMBDA_CONDITIONS:
        weights, lam_sha = read_lambda(directory/f"lambda-{condition}.json", config, sigma_sha)
    return Calibration(sigma_z, sigma_sha, weights, lam_sha)


def load_data(config: DroidConfig, data_root: Path | str) -> TrainData:
    """从数据根读训练集（06 §1）、准备数组、动作尺度、验证窗口（`commanded` 配置跳过没通过的回合，同 10 §8.1）、
    特征清单（划分的全部片）和来源字典（06 §5）。"""
    data_root = Path(data_root)
    splits = sp.load_splits(data_root, config)
    plan = sp.shard_plan(splits, config.data)
    source = config.action.source
    prepared = {}
    for shard, uuids in plan.items():
        if shard in sp.TEST_SHARDS and shard != "val":
            continue
        for u in uuids:
            prepared[u] = pr.load_prepared(pr.episode_paths(data_root, shard, u)[0])
    train_set = wn.train_episodes(splits, prepared, config)
    episodes = bt.episode_arrays(prepared, train_set, source)
    root = pr.prepared_dir(data_root)
    scale_path, val_path = root/pr.ACTION_SCALE_FILE, root/pr.WINDOWS_DIR/f"{wn.VAL_SPLIT}.json"
    scale = ix._loads(scale_path.read_bytes(), str(scale_path))
    control_spec = droid_control_spec(scale, source=source, config=config)
    doc = ix._loads(val_path.read_bytes(), str(val_path))
    ev._windows_doc(doc, wn.VAL_SPLIT)
    if doc.get("salt") != config.data.window_salt:
        raise ValueError(f"{val_path} was frozen with another window salt")
    val_windows, kept, skipped = ev._selected(doc, prepared, source)
    if not val_windows:
        raise ValueError("no validation windows left")
    val_episodes = bt.episode_arrays(prepared, kept, source)
    manifests, manifest_sha = ft.feature_manifests(data_root, config)
    provenance = bt.training_provenance(config, manifests, manifest_sha)
    return TrainData(data_root, manifests, manifest_sha, train_set, episodes, val_windows, val_episodes, sorted(skipped),
                     control_spec, provenance, cache.file_sha(scale_path), cache.file_sha(val_path),
                     ft.FeatureReader(data_root, manifests))


# ---------------------------------------------------------------- 小工具

def _json(value) -> object:
    """规范化成写盘后读回的样子（tuple → list 等），登记的比较用它。"""
    return json.loads(cache._json_bytes(value))


def _read_json(path: Path) -> dict:
    return ix._loads(Path(path).read_bytes(), str(path))


def _append_line(path: Path, value: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_lines(path: Path) -> list[dict]:
    """jsonl；最后一行写到一半（中断）时丢掉那一行。"""
    if not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i < len(lines) - 2:          # 只有最后一行（后面没有换行）允许是半行
                raise ValueError(f"{path}: line {i + 1} is not JSON")
    return out


def _rewrite_lines(path: Path, rows: list[dict]) -> None:
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows)
    temp = Path(path).with_name(Path(path).name + ".partial")
    temp.write_text(body, encoding="utf-8")
    os.replace(temp, path)


def schedule(config: DroidConfig) -> dict:
    """01 §2 由 U 推出的量（登记里记一份）。"""
    return dict(updates=config.training.updates, validation_interval=config.validation_interval,
                consistency_interval=config.consistency_interval, early_check_updates=config.early_check_updates,
                consistency_warmup_updates=config.consistency_warmup_updates, lr_warmup_updates=config.lr_warmup_updates,
                log_every=config.training.log_every, checkpoint_minutes=config.training.checkpoint_minutes,
                stop_margin_minutes=config.training.stop_margin_minutes, min_share=config.monitoring.min_share)


def versions() -> dict:
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=str(np.__version__))


def registration(*, condition: str, seed: int, config: DroidConfig, data: TrainData, calibration: Calibration,
                 info: dict, reference: dict | None, loss_config, optimizer) -> dict:
    """09 §4.2 的登记内容（规范化后的 JSON）。"""
    return _json(dict(
        schema=REGISTRATION_SCHEMA, condition=condition, seed=seed, config_name=config.name,
        config_sha256=config.config_sha256,
        train_set=dict(count=len(data.train_set), sha256=wn.list_sha256(data.train_set)),
        action_scale_sha256=data.action_scale_sha256, val_windows_sha256=data.val_windows_sha256,
        val_windows=dict(count=len(data.val_windows), skipped_episodes=list(data.val_skipped)),
        feature_manifest_sha256=data.feature_manifest_sha256,
        sigma_z=dict(value=calibration.sigma_z, sha256=calibration.sigma_z_sha256),
        lambda_weights=None if calibration.weights is None else dict(calibration.weights, sha256=calibration.lambda_sha256),
        model=dict(type=info["model_type"], spec=info["spec"], parameters=info["parameters"],
                   po_parameters=info["po_parameters"], matching=info["matching"],
                   initial_model_sha256=info["initial_model_sha256"]),
        reference=reference, spectral_backend=info["spectral_backend"],
        loss_config=training._loss_config(loss_config, cd.LOSS_LABELS[condition]),
        optimizer=training._optimizer_config(optimizer), schedule=schedule(config),
        provenance=data.provenance, ac=info["ac"], versions=versions()))


def check_registration(path: Path, current: dict) -> None:
    """第一次启动写；之后每次启动都和按当前代码、配置重算的比，不同就 RegistrationMismatch（退出 2）。"""
    if not path.exists():
        cache._write_json(path, current)
        return
    old = _read_json(path)
    if old != current:
        keys = sorted(k for k in set(old) | set(current) if old.get(k) != current.get(k))
        raise RegistrationMismatch(f"{path} differs from this run in {keys}: code, config or upstream artifacts "
                                   f"changed; use a new run root or delete the directory to start over")


class StopFlag:
    """SIGTERM、SIGUSR1 只设标志，当前这一步做完再存档退出（09 §4.7）。"""

    SIGNALS = tuple(s for s in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGUSR1", None)) if s is not None)

    def __init__(self) -> None:
        self.raised: int | None = None
        self._old = {}

    def __enter__(self):
        for s in self.SIGNALS:
            self._old[s] = signal.signal(s, self._handle)
        return self

    def _handle(self, signum, frame) -> None:
        self.raised = signum

    def __exit__(self, *error) -> None:
        for s, handler in self._old.items():
            signal.signal(s, handler)


# ---------------------------------------------------------------- 验证（§4.4）

def validate(model, *, condition: str, data: TrainData, config: DroidConfig, loss_config, device) -> dict:
    """1,200 个验证窗口按文件顺序每 64 个一批，eval 模式、no_grad，每个窗口一次直接前向的未来误差
    `jepa_distance(prediction, raw_target).per_sample`；AC 官方输入用绑定起点状态的前向（09 §6.2）。"""
    bind = condition in cd.AC_CONDITIONS and config.model.ac_state == "true_state"
    modes = {m: m.training for m in model.modules()}
    started = time.perf_counter()
    values = []
    try:
        model.eval()
        with torch.no_grad():
            for chunk in bt.chunks(data.val_windows):
                batch, start_states = bt.make_eval_batch(chunk, data.reader, data.val_episodes, data.control_spec,
                                                         data.provenance, device)
                phi = droid_ac.bind(model, start_states, batch.controls) if bind else model
                prediction = phi(batch.raw_z, batch.controls, batch.duration, batch.transfer.source)
                values.append(jepa_distance(prediction, batch.raw_target, config=loss_config).per_sample.detach()
                              .double().cpu())
    finally:
        for module, flag in modes.items():
            module.training = flag
    if device_is_cuda(device):
        torch.cuda.synchronize(device)
    per_window = torch.cat(values)
    if not torch.isfinite(per_window).all() or torch.any(per_window < 0):
        raise ValueError("validation future distances must be finite and non-negative")
    return dict(ids=[w["id"] for w in data.val_windows], values=per_window.tolist(),
                mean=float(per_window.mean()), seconds=time.perf_counter() - started)


def device_is_cuda(device) -> bool:
    return torch.device(device).type == "cuda"


def _validation_path(out: Path, step: int) -> Path:
    return out/"validations"/f"step-{step:06d}.json"


def validation_records(out: Path, completed: int) -> list[dict]:
    """validations/ 下步数 ≤ completed 的记录（按步数）。"""
    rows = []
    for path in sorted((out/"validations").glob("step-*.json")):
        doc = _read_json(path)
        if doc.get("schema") != VALIDATION_SCHEMA or path.name != f"step-{doc.get('step', -1):06d}.json":
            raise ValueError(f"{path} is not a validation record of this run")
        if doc["step"] <= completed:
            rows.append(doc)
    return sorted(rows, key=lambda r: r["step"])


def best_of(records: list[dict]) -> dict:
    """均值严格更小才换最优，平局留更早的；第 0 步也参加（v7 相同）。"""
    best = None
    for r in records:
        if best is None or r["mean"] < best["mean"]:
            best = r
    return best


def selection(records: list[dict], entries: list[dict]) -> dict:
    best = best_of(records)
    entry = next((e for e in entries if e["kind"] == "validation" and e["step"] == best["step"]), None)
    if entry is None:
        raise ValueError(f"the best validation step {best['step']} has no checkpoint in the index")
    return dict(schema=SELECTION_SCHEMA, validations=[dict(step=r["step"], mean=r["mean"]) for r in records],
                best=dict(step=best["step"], mean=best["mean"], checkpoint=f"checkpoints/{entry['path']}",
                          checkpoint_sha256=entry["sha256"]))


# ---------------------------------------------------------------- 一致性监控（§4.6）

def monitor_summary(measurements: list[dict], terms: tuple[str, ...], config: DroidConfig) -> dict:
    """v7 `_checked_terms_monitor_summary` 的字段：一致性预热结束后（步数 ≥ W_c）各次测量的中位数，全都 ≥ min_share 才算参与。"""
    threshold, rho = config.monitoring.min_share, config.model.rho
    rows = [m for m in measurements if m["step"] >= config.consistency_warmup_updates]
    if not rows:
        return dict(measurements=0, rho=rho, active=False, median_share=None, threshold=threshold, checked_terms=list(terms))
    median = {k: statistics.median(m["share"][k] for m in rows) for k in terms}
    return dict(measurements=len(rows), rho=rho, threshold=threshold, median_share=median,
                active=all(v >= threshold for v in median.values()), checked_terms=list(terms))


# ---------------------------------------------------------------- 日志（§5）

class Log:
    """train-log.jsonl：每 log_every 步一行；存档前把不满的部分也写成一行。"""

    def __init__(self, path: Path, device) -> None:
        self.path, self.device = path, torch.device(device)
        self.steps: list[dict] = []
        self.consistency: list[dict] = []

    def add(self, record: dict, share: dict | None) -> None:
        self.steps.append(record)
        if share is not None:
            self.consistency.append(share)

    def flush(self) -> None:
        if not self.steps:
            return
        gpu = None
        if self.device.type == "cuda":
            gpu = dict(name=torch.cuda.get_device_name(self.device),
                       peak_memory_bytes=int(torch.cuda.max_memory_allocated(self.device)))
            torch.cuda.reset_peak_memory_stats(self.device)
        _append_line(self.path, dict(first_step=self.steps[0]["step"], last_step=self.steps[-1]["step"],
                                     steps=self.steps, consistency=self.consistency, gpu=gpu))
        self.steps, self.consistency = [], []


def check_log_coverage(rows: list[dict], updates: int) -> None:
    """日志各行按 first_step 排起来正好覆盖 1…U，不重不漏（§5）。"""
    expected = 1
    for r in sorted(rows, key=lambda r: r["first_step"]):
        steps = [s["step"] for s in r["steps"]]
        if r["first_step"] != expected or steps != list(range(r["first_step"], r["last_step"] + 1)):
            raise ValueError(f"train-log.jsonl does not cover the steps contiguously at step {expected}")
        expected = r["last_step"] + 1
    if expected != updates + 1:
        raise ValueError(f"train-log.jsonl covers steps 1…{expected - 1}, not 1…{updates}")


# ---------------------------------------------------------------- 一个训练任务

def run(output: Path | str, *, condition: str, seed: int, config: DroidConfig, data: TrainData,
        calibration: Calibration, device, time_limit_minutes: float | None = None, retry: bool = False,
        ac_checkpoint: Path | str | None = None, workers: int = bt.WORKERS, started: float | None = None,
        clock: Callable[[], float] = time.monotonic, after_step: Callable[[int], None] | None = None) -> int:
    """09 §4.3。返回退出码（§4.7）。`clock` 和 `after_step` 只给测试用（模拟时长上限和信号）。"""
    if condition not in CONDITIONS:
        raise ValueError(f"{condition} is not a trained DROID condition")
    if seed not in config.training.seeds:
        raise ValueError(f"seed {seed} is not one of this config's seeds {list(config.training.seeds)}")
    started = clock() if started is None else started
    out = Path(output)
    if (out/"COMPLETE.json").exists():
        report = out/"report.json"
        if _read_json(out/"COMPLETE.json").get("report_sha256") != cache.file_sha(report):
            raise ValueError(f"{out}: COMPLETE.json does not match report.json")
        print(f"{condition}/seed{seed}: already complete")
        return EXIT_DONE
    if (out/"FAILED.json").exists():
        if not retry:
            print(f"{out/'FAILED.json'} exists: look at it, then rerun with --retry to continue from the latest checkpoint",
                  file=sys.stderr)
            return EXIT_FAILED
        (out/"FAILED.json").unlink()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    try:
        with StopFlag() as stop:
            return _run(out, condition=condition, seed=seed, config=config, data=data, calibration=calibration,
                        device=device, time_limit_minutes=time_limit_minutes, ac_checkpoint=ac_checkpoint,
                        workers=workers, started=started, clock=clock, after_step=after_step, stop=stop)
    except RegistrationMismatch as error:
        print(str(error), file=sys.stderr)
        return EXIT_REGISTRATION
    except ConsistencyTooWeak as error:
        cache._write_json(out/"FAILED.json", dict(condition=condition, seed=seed, reason="consistency_too_weak",
                                                  error=str(error)))
        print(str(error), file=sys.stderr)
        return EXIT_WEAK
    except Exception as error:   # noqa: BLE001 —— 09 §4.7：其他错误写 FAILED.json、退出 1
        cache._write_json(out/"FAILED.json", dict(condition=condition, seed=seed, reason="error",
                                                  error_type=type(error).__name__, error=str(error)))
        print(f"{condition}/seed{seed} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_FAILED


def _run(out: Path, *, condition, seed, config, data, calibration, device, time_limit_minutes, ac_checkpoint, workers,
         started, clock, after_step, stop: StopFlag) -> int:
    t = config.training
    updates, interval = t.updates, config.validation_interval
    deadline = None if time_limit_minutes is None else started + 60*float(time_limit_minutes)
    margin = 60*t.stop_margin_minutes

    # 构造（§2）、§2.4 初始权重关系、登记（§4.2）
    model, loss_config, info = cd.build(condition, config=config, seed=seed, sigma_z=calibration.sigma_z, device=device,
                                        weights=calibration.weights, ac_checkpoint=ac_checkpoint)
    reference = None
    if condition in cd.REFERENCE:
        ref_name = cd.REFERENCE[condition]
        ref_hash = cd.reference_hash(ref_name, config=config, seed=seed, sigma_z=calibration.sigma_z)
        cd.check_reference(condition, info, dict(condition=ref_name, seed=seed, initial_model_sha256=ref_hash))
        reference = dict(condition=ref_name, initial_model_sha256=ref_hash)
    optimizer = cd.make_optimizer(condition, model)
    current = registration(condition=condition, seed=seed, config=config, data=data, calibration=calibration, info=info,
                           reference=reference, loss_config=loss_config, optimizer=optimizer)
    check_registration(out/"registration.json", current)
    registration_sha = cache.file_sha(out/"registration.json")

    # 从最新检查点接着跑（§4.3、§4.5）
    ckdir = out/"checkpoints"
    ckdir.mkdir(exist_ok=True)
    (out/"validations").mkdir(exist_ok=True)
    entries = ck.read_index(ckdir)
    ck.clean_orphans(ckdir, entries)
    head = ck.latest(entries)
    completed = 0
    if head is not None:
        completed = ck.load(condition, ckdir/head["path"], head["sha256"], model=model, optimizer=optimizer,
                            loss_config=loss_config, provenance=data.provenance, updates=updates)
        if completed != head["step"]:
            raise ValueError(f"checkpoint {head['path']} holds step {completed}, the index says {head['step']}")
    log_path = out/LOG_FILE
    kept = [r for r in _read_lines(log_path) if r["last_step"] <= completed]
    _rewrite_lines(log_path, kept)
    for path in (out/"validations").glob("step-*.json"):
        if int(path.stem.split("-")[1]) > completed:
            path.unlink()
    _append_line(out/STARTS_FILE, dict(completed=completed, resumed_from=None if head is None else head["path"],
                                       time_limit_minutes=time_limit_minutes, device=str(device)))
    measurements = [m for r in kept for m in r["consistency"]]
    terms = cd.monitored_terms(condition, loss_config)
    log = Log(log_path, device)
    is_ac = condition in cd.AC_CONDITIONS

    def save(kind: str, step: int) -> dict:
        nonlocal entries
        log.flush()
        name = ck.file_name(kind, step)
        sha = ck.save(condition, ckdir/name, model=model, optimizer=optimizer, loss_config=loss_config,
                      completed=step, provenance=data.provenance)
        entry = dict(path=name, step=step, kind=kind, sha256=sha)
        old = list(entries)
        entries = [e for e in entries if not (e["kind"] == "resume" and e["step"] < step)] + [entry]
        if kind == "validation" and condition in RETAIN_TWO:
            best = best_of(validation_records(out, step))["step"]
            entries = [e for e in entries if e["kind"] == "resume" or e["step"] in (best, step)]
        ck.write_index(ckdir, entries)
        for e in old:
            if e not in entries:
                (ckdir/e["path"]).unlink(missing_ok=True)
        return entry

    def validate_and_save(step: int) -> None:
        result = validate(model, condition=condition, data=data, config=config, loss_config=loss_config, device=device)
        ck.write_json_replace(_validation_path(out, step), dict(schema=VALIDATION_SCHEMA, step=step, mean=result["mean"],
                              count=len(result["values"]), seconds=result["seconds"],
                              windows=[dict(id=i, value=v) for i, v in zip(result["ids"], result["values"])]))
        save("validation", step)
        ck.write_json_replace(out/"selection.json", selection(validation_records(out, step), entries))

    if head is None:
        validate_and_save(0)
    else:
        ck.write_json_replace(out/"selection.json", selection(validation_records(out, completed), entries))
    last_save = clock()

    if completed < updates:
        dataset = bt.StepBatches("ac" if is_ac else "train", data.data_root, data.manifests, data.train_set,
                                 data.episodes, seed=seed, start=completed, stop=updates, config=config)
        batches = bt.training_batches(dataset, data.control_spec, data.provenance, device, workers=workers)
        readouts = cd.expected_readouts(condition, model)
        try:
            for step in range(completed, updates):
                waited = time.perf_counter()
                got, batch = next(batches)
                waited = time.perf_counter() - waited
                if got != step:
                    raise RuntimeError(f"data loader gave step {got}, expected {step}")
                calls = []
                hook = None if readouts is None else model.p_out.register_forward_hook(lambda *a: calls.append(1))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                began = time.perf_counter()
                try:
                    result, lr = cd.train_step(condition, model, optimizer, batch, loss_config, step,
                                               ac_state=config.model.ac_state)
                finally:
                    if hook is not None:
                        hook.remove()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - began
                completed = step + 1
                if result.completed_steps != completed:
                    raise RuntimeError("the update did not advance exactly one step")
                if readouts is not None and len(calls) != readouts:
                    raise RuntimeError(f"p_out was called {len(calls)} times, expected {readouts} (07 §2.4)")
                if (not math.isfinite(result.grad_max_abs) or not math.isfinite(result.grad_norm)
                        or any(v is not None and not math.isfinite(v) for v in result.loss_values.values())):
                    raise RuntimeError("nonfinite training diagnostics")
                share = None
                if terms and completed % config.consistency_interval == 0:
                    share = dict(step=completed, **cd.consistency_share(condition, model, batch, loss_config, completed))
                    measurements.append(share)
                log.add(dict(step=completed, losses=result.loss_values, grad_norm=result.grad_norm,
                             grad_max_abs=result.grad_max_abs, lr=lr, seconds=seconds, wait_seconds=waited,
                             window_ids_sha256=wn.list_sha256(batch.sample_ids),
                             p_out_calls=None if readouts is None else len(calls)), share)
                del batch
                if completed % t.log_every == 0:
                    log.flush()
                if terms and completed == config.early_check_updates:
                    summary = monitor_summary(measurements, terms, config)
                    ck.write_json_replace(out/"early-check.json", dict(completed_updates=completed, **summary))
                    if not summary["active"]:
                        log.flush()
                        raise ConsistencyTooWeak(f"{condition}/seed{seed}: median consistency share "
                                                 f"{summary['median_share']} below {summary['threshold']} at step {completed}")
                if after_step is not None:
                    after_step(completed)
                halt = stop.raised is not None or (deadline is not None and clock() >= deadline - margin)
                if completed % interval == 0 or completed == updates:
                    validate_and_save(completed)
                    last_save = clock()
                elif halt or clock() - last_save >= 60*t.checkpoint_minutes:
                    save("resume", completed)
                    last_save = clock()
                if halt and completed < updates:
                    why = "signal" if stop.raised is not None else "time limit"
                    print(f"{condition}/seed{seed}: saved at step {completed} ({why}); rerun the same command",
                          file=sys.stderr)
                    return EXIT_REQUEUE
        finally:
            batches.close()

    # 选点、报告（§4.4、§5）
    log.flush()
    rows = _read_lines(log_path)
    check_log_coverage(rows, updates)
    records = validation_records(out, updates)
    sel = selection(records, entries)
    ck.write_json_replace(out/"selection.json", sel)
    starts = _read_lines(out/STARTS_FILE)
    total = (sum(s["seconds"] + s["wait_seconds"] for r in rows for s in r["steps"])
             + sum(r["seconds"] for r in records))
    report = dict(schema=REPORT_SCHEMA, condition=condition, seed=seed, config_name=config.name,
                  config_sha256=config.config_sha256, completed_steps=updates, best=sel["best"],
                  validations=sel["validations"],
                  consistency_monitor=monitor_summary(measurements, terms, config) if terms else None,
                  total_seconds=total, resumes=len(starts) - 1, parameters=info["parameters"],
                  initial_model_sha256=info["initial_model_sha256"], registration_sha256=registration_sha,
                  scientific_conclusions_verified=False)
    (out/"report.json").unlink(missing_ok=True)
    cache._write_json(out/"report.json", report)
    cache._write_json(out/"COMPLETE.json", dict(report_sha256=cache.file_sha(out/"report.json")))
    return EXIT_DONE


# ---------------------------------------------------------------- 命令行

def parse_job(job: str, config: DroidConfig) -> tuple[str, int]:
    """`train/<条件>/seed<种子>` → (条件, 种子)。"""
    parts = job.split("/")
    if len(parts) != 3 or parts[0] != "train" or not parts[2].startswith("seed") or not parts[2][4:].isdigit():
        raise ValueError(f"not a training job: {job!r}")
    condition, seed = parts[1], int(parts[2][4:])
    if condition not in CONDITIONS or seed not in config.training.seeds:
        raise ValueError(f"{job}: condition or seed is not in this config")
    return condition, seed


def job_dir(run_root: Path | str, config: DroidConfig, condition: str, seed: int) -> Path:
    return Path(run_root)/config.name/"train"/condition/f"seed{seed}"


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--job", required=True)
    p.add_argument("--time-limit-minutes", type=float)
    p.add_argument("--retry", action="store_true", help="删掉 FAILED.json，从最新检查点接着跑（09 §4.3）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=bt.WORKERS)
    a = p.parse_args(argv)
    config = load_config(a.config)
    condition, seed = parse_job(a.job, config)
    out = job_dir(a.run_root, config, condition, seed)
    if (out/"COMPLETE.json").exists():
        return run(out, condition=condition, seed=seed, config=config, data=None, calibration=None, device=a.device)
    calibration = load_calibration(a.run_root, config, condition)
    data = load_data(config, a.data_root)
    ac_checkpoint = Path(a.data_root)/"checkpoints"/OFFICIAL_AC_FILE if condition == "vjepa2_ac" else None
    return run(out, condition=condition, seed=seed, config=config, data=data, calibration=calibration, device=a.device,
               time_limit_minutes=a.time_limit_minutes, retry=a.retry, ac_checkpoint=ac_checkpoint, workers=a.workers,
               started=started)


if __name__ == "__main__":
    raise SystemExit(main())
