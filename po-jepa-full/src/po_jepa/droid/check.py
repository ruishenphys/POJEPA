"""DROID 环境检查 `check/env`（docs/design/09-droid-training.md §7，S09b）。

在一张训练卡上，用种子 `calibration.seed`（1703）的真实批（`step_rng(1703, step, 0)` 的训练窗口，AC 用 stream 1 的片段）：

1. 环境：GPU 型号和驱动、torch、CUDA、CPU 核数、特征所在磁盘的读速度（训练集里随机 500 行计时）；上游代码的 git 提交
   （必须是 `vjepa.UPSTREAM_COMMIT` = 204698b）和编码器权重 SHA（必须等于 `cache.CHECKPOINT_SHA`），AC 官方权重的 SHA（12 §1）。
2. 每个可训练条件：构造、参数量和 §2.3 的表一致、§2.4 初始权重哈希的关系（含 vjepa2_ac_zeroshot = vjepa2_ac）、20 步训练
   （每步损失和梯度有限、每个可训练参数都有梯度、p_out 调用次数符合 07 §2.4）、第 6–20 步单步用时的中位数、峰值显存，
   再验证一次计时。
3. 推算每个训练任务的总时长：U × 单步中位数 + (U/V + 1) 次验证（第 0 步也验证，09 §4.3）。
4. 求谱后端（§6.3）：8 个真实训练批的图拉普拉斯上比较三个后端（`spectral_backends.agreement`，参照 cpu_batched），记测速和
   一致的后端里最快的那个；配置选用的后端不一致算不通过。CPU 上没有 cuda_float64，记为不可用。
5. 数据加载器：不跑模型，训练窗口和 AC 片段各连续产出 200 步，记每秒步数；要求比用这种批的条件里最快那个的单步快。

带一致性的三个条件用占位权重 λ = 1（`calibrate.PLACEHOLDER_WEIGHTS`，check/env 不依赖 λ 标定，12 §2；只测用时和有限性）。
全部通过才写 `<run-root>/<name>/check/env.json`（完成标志）；任一项不通过写 `check/FAILED.json`（同样的内容加 problems），
退出 1，训练任务不开始。

  python -m po_jepa.droid.check --config … --data-root … --run-root … [--device cuda] [--workers 2]
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from po_jepa import spectral_backends, vjepa
from po_jepa.droid import batches as bt
from po_jepa.droid import calibrate as cb
from po_jepa.droid import conditions as cd
from po_jepa.droid import features as ft
from po_jepa.droid import train as tr
from po_jepa.droid.config import CONDITIONS, DroidConfig, load_config
from po_jepa.droid.spectral_gap import read_sigma_z
from po_jepa.model import POJEPA
from po_jepa.runtime import cache
from po_jepa.spectral import GraphSpectrum

SCHEMA = "droid_check_env_v1"
STEPS = 20                       # §7 第 2 条
TIMED_FROM = 6                   # 第 6–20 步（从 1 计）的单步用时取中位数
DISK_ROWS = 500                  # §7 第 1 条
BACKEND_BATCHES = 8              # §6.3：8 个真实训练批
LOADER_STEPS = 200               # §7 第 5 条
# 09 §2.3 的表（PO-JEPA 类、Transformer 类、AC 缩小版按 ac_state、Koopman）；官方规模 AC 不在表里
PARAMETER_TABLE = dict(po_jepa_full=248_737, po_jepa_no_consistency=248_737, fixed_fourier=248_737,
                       matched_transformer=249_024, transformer_consistency=249_024, koopman=248_512)
AC_SMALL_TABLE = dict(true_state=249_344, learned_constant=248_896)


def expected_parameters(condition: str, config: DroidConfig) -> int | None:
    if condition == "vjepa2_ac_small":
        return AC_SMALL_TABLE[config.model.ac_state]
    return PARAMETER_TABLE.get(condition)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def batch_kind(condition: str) -> str:
    return "ac" if condition in cd.AC_CONDITIONS else "train"


# ---------------------------------------------------------------- 1. 环境

def _nvidia_driver() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out.splitlines()[0].strip() if out else None


def environment(device: torch.device) -> dict:
    gpu = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu = dict(name=torch.cuda.get_device_name(device), total_memory_bytes=int(props.total_memory),
                   capability=list(torch.cuda.get_device_capability(device)), driver=_nvidia_driver())
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return dict(device=str(device), gpu=gpu, torch=str(torch.__version__), cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                python=platform.python_version(), numpy=str(np.__version__), cpu_count=os.cpu_count(),
                cpu_affinity=affinity, torch_threads=torch.get_num_threads(), platform=platform.platform())


def disk_read(data: tr.TrainData, *, rows: int = DISK_ROWS, seed: int) -> dict:
    """训练集里随机 `rows` 行（回合均匀、行均匀，`numpy.random.default_rng(seed)`），新开一个 `FeatureReader` 计时读
    （含每条回合第一次打开时的 JSON 核对和每行的 SHA 核对，和训练时的读法相同）。"""
    rng = np.random.default_rng(seed)
    picks = []
    for _ in range(rows):
        u = data.train_set[int(rng.integers(len(data.train_set)))]
        picks.append((u, int(rng.integers(data.episodes[u].rows))))
    reader = ft.FeatureReader(data.data_root, data.manifests)
    began = time.perf_counter()
    total = 0
    for u, r in picks:
        total += reader.rows(u, [r]).shape[1]*ft.WIDTH*2        # float16 的字节数
    seconds = time.perf_counter() - began
    return dict(rows=rows, seconds=seconds, rows_per_second=rows/seconds if seconds > 0 else None,
                megabytes_per_second=total/1e6/seconds if seconds > 0 else None)


def upstream_identity(data_root: Path | str, config: DroidConfig) -> tuple[dict, list[str]]:
    """12 §1：上游代码的 git 提交、编码器权重的 SHA（文件不在、不对都算不通过）。"""
    problems = []
    root = Path(data_root)/"upstream"/"vjepa2"
    commit = None
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=60, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        problems.append(f"cannot read the upstream commit of {root}: {error}")
    if commit is not None and commit != vjepa.UPSTREAM_COMMIT:
        problems.append(f"upstream commit {commit} != {vjepa.UPSTREAM_COMMIT}")
    if config.encode.upstream_commit != vjepa.UPSTREAM_COMMIT:
        problems.append(f"encode.upstream_commit {config.encode.upstream_commit} != {vjepa.UPSTREAM_COMMIT}")
    weights = Path(data_root)/"checkpoints"/config.encode.checkpoint_file
    digest = cache.file_sha(weights) if weights.is_file() else None
    if digest != cache.CHECKPOINT_SHA:
        problems.append(f"encoder weights {weights}: SHA {digest} != {cache.CHECKPOINT_SHA}")
    return dict(upstream_root=str(root), upstream_commit=commit, encoder_checkpoint=str(weights),
                encoder_checkpoint_sha256=digest), problems


# ---------------------------------------------------------------- 2. 每个可训练条件

def _batches(kind: str, data: tr.TrainData, config: DroidConfig, steps: int, device, workers: int):
    dataset = bt.StepBatches(kind, data.data_root, data.manifests, data.train_set, data.episodes,
                             seed=config.calibration.seed, start=0, stop=steps, config=config)
    return bt.training_batches(dataset, data.control_spec, data.provenance, device, workers=workers)


def _finite_losses(values: Mapping) -> bool:
    return all(v is None or math.isfinite(v) for v in values.values())


def train_steps(condition: str, model, loss, data: tr.TrainData, config: DroidConfig, device: torch.device, *,
                steps: int = STEPS, workers: int = bt.WORKERS) -> dict:
    """§7 第 2 条的训练部分：`steps` 步，每步核损失和梯度有限、每个可训练参数都有梯度、p_out 调用次数。"""
    optimizer = cd.make_optimizer(condition, model)
    readouts = cd.expected_readouts(condition, model)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    problems, rows = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for step, batch in _batches(batch_kind(condition), data, config, steps, device, workers):
        calls = []
        hook = None if readouts is None else model.p_out.register_forward_hook(lambda *a: calls.append(1))
        _sync(device)
        began = time.perf_counter()
        try:
            result, lr = cd.train_step(condition, model, optimizer, batch, loss, step, ac_state=config.model.ac_state)
        finally:
            if hook is not None:
                hook.remove()
        _sync(device)
        seconds = time.perf_counter() - began
        missing = [n for n, p in named if p.grad is None]
        nonfinite = [n for n, p in named if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        row = dict(step=step + 1, seconds=seconds, losses=result.loss_values, grad_norm=result.grad_norm,
                   grad_max_abs=result.grad_max_abs, lr=lr, p_out_calls=None if readouts is None else len(calls))
        rows.append(row)
        if not _finite_losses(result.loss_values) or not math.isfinite(result.grad_norm or 0.) \
                or not math.isfinite(result.grad_max_abs):
            problems.append(f"{condition} step {step + 1}: nonfinite loss or gradient")
        if missing:
            problems.append(f"{condition} step {step + 1}: {len(missing)} parameters got no gradient ({missing[:3]})")
        if nonfinite:
            problems.append(f"{condition} step {step + 1}: nonfinite gradients in {nonfinite[:3]}")
        if readouts is not None and len(calls) != readouts:
            problems.append(f"{condition} step {step + 1}: p_out called {len(calls)} times, expected {readouts} (07 §2.4)")
        del batch
    if len(rows) != steps:
        problems.append(f"{condition}: ran {len(rows)} steps, expected {steps}")
    timed = [r["seconds"] for r in rows[TIMED_FROM - 1:]] or [r["seconds"] for r in rows]
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    return dict(steps=rows, median_step_seconds=statistics.median(timed) if timed else None,
                timed_steps=[TIMED_FROM, steps], peak_memory_bytes=peak, problems=problems)


def estimate(config: DroidConfig, step_seconds: float | None, validation_seconds: float | None) -> dict:
    """§7 第 3 条：U 步加 (U/V + 1) 次验证（第 0 步也验证）。"""
    updates = config.training.updates
    validations = updates//config.validation_interval + 1
    total = None if step_seconds is None or validation_seconds is None else updates*step_seconds + validations*validation_seconds
    return dict(updates=updates, validations=validations, seconds=total, hours=None if total is None else total/3600)


def check_condition(condition: str, *, config: DroidConfig, data: tr.TrainData, sigma_z: float, device: torch.device,
                    ac_checkpoint: Path | None, steps: int = STEPS, workers: int = bt.WORKERS) -> dict:
    weights = dict(cb.PLACEHOLDER_WEIGHTS) if condition in cd.LAMBDA_CONDITIONS else None
    model, loss, info = cd.build(condition, config=config, seed=config.calibration.seed, sigma_z=sigma_z, device=device,
                                 weights=weights, ac_checkpoint=ac_checkpoint if condition in cd.OFFICIAL_AC else None)
    problems = []
    want = expected_parameters(condition, config)
    if want is not None and info["parameters"] != want:
        problems.append(f"{condition}: {info['parameters']} parameters, 09 §2.3 says {want}")
    record = dict(condition=condition, parameters=info["parameters"], expected_parameters=want,
                  initial_model_sha256=info["initial_model_sha256"], matching=info["matching"],
                  spectral_backend=info["spectral_backend"], ac=info["ac"], placeholder_weights=weights)
    trained = train_steps(condition, model, loss, data, config, device, steps=steps, workers=workers)
    problems += trained.pop("problems")
    _sync(device)
    validation = tr.validate(model, condition=condition, data=data, config=config, loss_config=loss, device=device)
    record.update(trained, validation_seconds=validation["seconds"], validation_windows=len(validation["values"]),
                  estimate=estimate(config, trained["median_step_seconds"], validation["seconds"]), problems=problems)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def relations(records: Mapping[str, dict], zeroshot_hash: str | None) -> tuple[list[dict], list[str]]:
    """§2.4：同一种子下的初始权重哈希关系（含 vjepa2_ac_zeroshot = vjepa2_ac）。"""
    rows, problems = [], []
    pairs = [(c, ref) for c, ref in cd.REFERENCE.items() if c != "vjepa2_ac_zeroshot"]
    for c, ref in pairs:
        if c in records and ref in records:
            same = records[c]["initial_model_sha256"] == records[ref]["initial_model_sha256"]
            rows.append(dict(condition=c, reference=ref, same=same))
            if not same:
                problems.append(f"{c} and {ref} start from different weights (09 §2.4)")
    if "vjepa2_ac" in records:
        same = zeroshot_hash == records["vjepa2_ac"]["initial_model_sha256"]
        rows.append(dict(condition="vjepa2_ac_zeroshot", reference="vjepa2_ac", same=same))
        if not same:
            problems.append("vjepa2_ac_zeroshot and vjepa2_ac start from different weights (09 §2.4)")
    return rows, problems


# ---------------------------------------------------------------- 4. 求谱后端

def backend_comparison(config: DroidConfig, data: tr.TrainData, sigma_z: float, device: torch.device, *,
                       batches: int = BACKEND_BATCHES, workers: int = bt.WORKERS) -> tuple[dict, list[str]]:
    """§6.3：po_jepa_full 的图（σ_z 标定值）在前 `batches` 个训练批上建拉普拉斯，三个后端各解一次 eigh。"""
    model, _, _ = cd.build("po_jepa_full", config=config, seed=config.calibration.seed, sigma_z=sigma_z, device=device,
                           weights=dict(cb.PLACEHOLDER_WEIGHTS))
    if not isinstance(model, POJEPA) or type(model.spectrum) is not GraphSpectrum:
        raise TypeError("po_jepa_full must use GraphSpectrum")
    k = model.spectrum.config.max_modes
    threads = config.resources.train.cpus
    names = [b for b in spectral_backends.BACKENDS if b != "cuda_float64" or device.type == "cuda"]
    seconds = {b: 0. for b in names}
    per_batch = []
    with torch.no_grad():
        for step, batch in _batches("train", data, config, batches, device, workers):
            affinity = model.spectrum.affinity(batch.raw_z, batch.transfer.source)
            laplacian = model.spectrum.normalized_laplacian(affinity).laplacian.detach()
            results = {}
            for b in names:
                _sync(device)
                began = time.perf_counter()
                results[b] = spectral_backends.eigh(laplacian, b, threads=threads, device=device)
                _sync(device)
                seconds[b] += time.perf_counter() - began
            per_batch.append({b: spectral_backends.agreement(results["cpu_batched"], results[b], k)
                              for b in names if b != "cpu_batched"})
            del batch
    agree = {b: b == "cpu_batched" or all(row[b]["agree"] for row in per_batch) for b in names}
    worst = {b: dict(max_eigenvalue_diff=max(row[b]["max_eigenvalue_diff"] for row in per_batch),
                     max_projector_diff=max(row[b]["max_projector_diff"] for row in per_batch))
             for b in names if b != "cpu_batched"}
    fastest = min((b for b in names if agree[b]), key=lambda b: (seconds[b], b))
    chosen = config.model.spectral_backend
    problems = []
    if chosen not in names:
        problems.append(f"model.spectral_backend {chosen} cannot run on {device}")
    elif not agree[chosen]:
        problems.append(f"model.spectral_backend {chosen} does not agree with cpu_batched (09 §6.3)")
    out = dict(batches=len(per_batch), k=k, threads=threads, seconds=seconds, agree=agree, worst=worst,
               per_batch=per_batch, unavailable=[b for b in spectral_backends.BACKENDS if b not in names],
               fastest_agreeing=fastest, configured=chosen,
               tolerances=dict(eigenvalue=spectral_backends.EIGENVALUE_ATOL, projector=spectral_backends.PROJECTOR_ATOL,
                               gap_floor=spectral_backends.GAP_FLOOR))
    del model
    return out, problems


# ---------------------------------------------------------------- 5. 数据加载器

def loader_speed(kind: str, data: tr.TrainData, config: DroidConfig, *, steps: int = LOADER_STEPS,
                 workers: int = bt.WORKERS) -> dict:
    """不跑模型，按步号连续产出 `steps` 步的批（CPU 上组装，和训练的加载器相同），记每秒步数。"""
    began = time.perf_counter()
    count = 0
    for _ in _batches(kind, data, config, steps, "cpu", workers):
        count += 1
    seconds = time.perf_counter() - began
    return dict(kind=kind, steps=count, seconds=seconds, steps_per_second=count/seconds if seconds > 0 else None,
                workers=workers)


# ---------------------------------------------------------------- 整个任务

def run_check(config: DroidConfig, *, data: tr.TrainData, sigma_z: float, sigma_z_sha256: str, device,
              ac_checkpoint: Path | None, upstream: Callable[[], tuple[dict, list[str]]],
              conditions: Sequence[str] = CONDITIONS, steps: int = STEPS, loader_steps: int = LOADER_STEPS,
              disk_rows: int = DISK_ROWS, workers: int = bt.WORKERS) -> dict:
    """§7 的五项，返回 env.json 的内容（`status` = passed / failed，`problems` 列出不通过的项）。"""
    device = torch.device(device)
    problems: list[str] = []
    ident, bad = upstream()
    problems += bad
    env = environment(device)
    disk = disk_read(data, rows=disk_rows, seed=config.calibration.seed)
    records = {}
    for c in conditions:
        if c not in CONDITIONS:
            raise ValueError(f"{c} is not a trained DROID condition")
        records[c] = check_condition(c, config=config, data=data, sigma_z=sigma_z, device=device,
                                     ac_checkpoint=ac_checkpoint, steps=steps, workers=workers)
        problems += records[c]["problems"]
    zeroshot = None
    if "vjepa2_ac" in records:
        model, _, info = cd.build("vjepa2_ac_zeroshot", config=config, seed=config.calibration.seed, sigma_z=sigma_z,
                                  device="cpu", ac_checkpoint=ac_checkpoint)
        zeroshot = info["initial_model_sha256"]
        ident["ac_checkpoint_sha256"] = info["ac"]["official_checkpoint_sha256"]
        if records["vjepa2_ac"]["ac"]["official_checkpoint_sha256"] != ident["ac_checkpoint_sha256"]:
            problems.append("vjepa2_ac and vjepa2_ac_zeroshot read different official weights")
        del model
    rel, bad = relations(records, zeroshot)
    problems += bad
    backends, bad = backend_comparison(config, data, sigma_z, device, workers=workers)
    problems += bad
    loaders = {}
    for kind in ("train", "ac"):
        users = [c for c in records if batch_kind(c) == kind and records[c]["median_step_seconds"]]
        if not users:
            continue
        speed = loader_speed(kind, data, config, steps=loader_steps, workers=workers)
        fastest = min(users, key=lambda c: records[c]["median_step_seconds"])
        need = 1/records[fastest]["median_step_seconds"]
        speed.update(fastest_condition=fastest, required_steps_per_second=need,
                     ok=speed["steps_per_second"] is not None and speed["steps_per_second"] > need)
        if not speed["ok"]:
            problems.append(f"the {kind} data loader gives {speed['steps_per_second']} steps/s, slower than one "
                            f"{fastest} step ({need:.3g}/s): the feature storage is too slow (09 §7)")
        loaders[kind] = speed
    return dict(schema=SCHEMA, config_name=config.name, config_sha256=config.config_sha256,
                sigma_z=dict(value=sigma_z, sha256=sigma_z_sha256), seed=config.calibration.seed,
                feature_manifest_sha256=data.feature_manifest_sha256, environment=env, disk=disk, upstream=ident,
                conditions=records, relations=rel, spectral_backends=backends, loaders=loaders,
                estimates={c: r["estimate"] for c, r in records.items()},
                status="failed" if problems else "passed", problems=problems, scientific_conclusions_verified=False)


def check_dir(run_root: Path | str, config: DroidConfig) -> Path:
    return Path(run_root)/config.name/"check"


def write_result(run_root: Path | str, config: DroidConfig, doc: dict) -> int:
    """通过：写 env.json（完成标志），退出 0；不通过：写（覆盖）FAILED.json，退出 1。"""
    out = check_dir(run_root, config)
    out.mkdir(parents=True, exist_ok=True)
    if doc["status"] == "passed":
        (out/"FAILED.json").unlink(missing_ok=True)
        cache._write_json(out/"env.json", doc)
        return 0
    from po_jepa.droid import checkpoints as ck
    ck.write_json_replace(out/"FAILED.json", doc)
    for p in doc["problems"]:
        print(f"check/env: {p}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=bt.WORKERS)
    a = p.parse_args(argv)
    config = load_config(a.config)
    if (check_dir(a.run_root, config)/"env.json").exists():
        print("check/env: already complete")
        return 0
    sigma_z, sigma_sha = read_sigma_z(cb.calibration_dir(a.run_root, config)/cb.SIGMA_FILE, config)
    data = tr.load_data(config, a.data_root)
    doc = run_check(config, data=data, sigma_z=sigma_z, sigma_z_sha256=sigma_sha, device=a.device,
                    ac_checkpoint=Path(a.data_root)/"checkpoints"/tr.OFFICIAL_AC_FILE,
                    upstream=lambda: upstream_identity(a.data_root, config), workers=a.workers)
    return write_result(a.run_root, config, doc)


if __name__ == "__main__":
    sys.exit(main())
