"""DROID 训练的检查点（docs/design/09-droid-training.md §4.1、§4.5，S09a）。

- 格式完全用 v7 的保存函数（`comparison.save_checkpoint` 按条件分发）：PO-JEPA 类 `training.save_training_checkpoint`，
  matched_transformer `comparison` 自己的 Transformer 格式，transformer_consistency、fixed_fourier、koopman 走 S10 的保存
  函数，AC `ac_training.save_checkpoint`（规格快照 `spec_snapshot`）。存进去的来源是 `comparison._provenance(来源)`。
- `sampling_state = {"kind": "explicit_prepared_batches", "next_batch_index": completed, "query_generator_state": None}`：
  DROID 的批由（种子，步号）算出，没有要保存的生成器状态。
- 载入：PO-JEPA 类 `training.load_training_checkpoint`；matched_transformer 用本模块的 `load_matched_transformer`（检查项和
  v7 `comparison.load_transformer_checkpoint` 逐条相同，只是期望的损失配置换成 DROID 的）；transformer_consistency、koopman
  用 S10 的读法；AC 用 `ac_training.load_checkpoint`，另核步数和游标。都先核 `index.json` 里记的 SHA。
- `checkpoints/index.json`：现存检查点的列表（路径、步数、种类、SHA），每次存档后先写临时文件再改名。v7 的保存函数拒绝
  覆盖，所以每次存档用新文件名；`index.json` 之外的文件是存档中途中断留下的，续跑时删掉。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from po_jepa.droid import conditions as cd
from po_jepa.runtime import ac_training, cache, comparison, training
from po_jepa.runtime.addon import conditions as addon_conditions

INDEX = "index.json"
INDEX_SCHEMA = "droid_train_checkpoints_v1"
KINDS = ("validation", "resume")
PREFIX = dict(validation="step", resume="resume")


def sampling_state(completed: int) -> dict:
    return {"kind": "explicit_prepared_batches", "next_batch_index": completed, "query_generator_state": None}


def file_name(kind: str, step: int) -> str:
    """验证点 `step-<6 位>.pt`，续跑 `resume-<6 位>.pt`（09 §4.1）。"""
    if kind not in KINDS or type(step) is not int or step < 0:
        raise ValueError(f"bad checkpoint kind/step {kind!r}/{step!r}")
    return f"{PREFIX[kind]}-{step:06d}.pt"


def write_json_replace(path: Path, value: dict) -> None:
    """可以覆盖的原子写（`index.json`、`selection.json` 等会重写的文件）：同目录临时文件写完、fsync，再 os.replace。"""
    temp = path.with_name(path.name + ".partial")
    with temp.open("wb") as handle:
        handle.write(cache._json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


# ---------------------------------------------------------------- index.json

def read_index(directory: Path) -> list[dict]:
    path = Path(directory)/INDEX
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != INDEX_SCHEMA or not isinstance(doc.get("checkpoints"), list):
        raise ValueError(f"{path} is not a DROID checkpoint index")
    entries = doc["checkpoints"]
    for e in entries:
        if (not isinstance(e, dict) or set(e) != {"path", "step", "kind", "sha256"} or e["kind"] not in KINDS
                or e["path"] != file_name(e["kind"], e["step"]) or not training._sha(e["sha256"])):
            raise ValueError(f"{path}: malformed entry {e!r}")
    if len({e["path"] for e in entries}) != len(entries):
        raise ValueError(f"{path}: duplicate entries")
    return entries


def write_index(directory: Path, entries: list[dict]) -> None:
    ordered = sorted(entries, key=lambda e: (e["step"], e["kind"]))
    write_json_replace(Path(directory)/INDEX, dict(schema=INDEX_SCHEMA, checkpoints=ordered))


def latest(entries: list[dict]) -> dict | None:
    """步数最大的检查点；同一步两种都有时取验证点的（09 §4.3）。"""
    if not entries:
        return None
    return max(entries, key=lambda e: (e["step"], e["kind"] == "validation"))


def clean_orphans(directory: Path, entries: list[dict]) -> list[str]:
    """删掉 checkpoints/ 下不在 index.json 里的文件（上次存档时中断留下的，含临时文件）；返回删掉的文件名。"""
    keep = {e["path"] for e in entries} | {INDEX}
    removed = []
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.name not in keep:
            path.unlink()
            removed.append(path.name)
    return removed


# ---------------------------------------------------------------- 保存和载入

def save(condition: str, path: Path, *, model, optimizer, loss_config, completed: int, provenance: dict) -> str:
    """v7 的保存函数（`comparison.save_checkpoint` 按条件分发）；返回文件 SHA。目标已存在就报错（不覆盖）。"""
    if condition not in cd.CONDITIONS:
        raise ValueError(f"{condition} is not a trained DROID condition")
    return comparison.save_checkpoint(condition, Path(path), model, optimizer, loss_config, completed,
                                      sampling_state(completed), provenance)


def load_matched_transformer(path: Path, expected_sha: str, model, optimizer, loss_config, expected_provenance):
    """v7 `comparison.load_transformer_checkpoint` 的检查逐条照搬，期望的损失配置换成 DROID 的 `loss_config`
    （"no_consistency" 快照）；v7 那个读法拿 v7 协议的损失配置去比，DROID 不能用（09 §4.5）。"""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("checkpoint SHA mismatch before loading")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    parameters = comparison._tf_optimizer(model, optimizer)
    device = parameters[0][1].device
    if (payload["schema"] != "F05_transformer_checkpoint_v1" or payload["config"] != asdict(model.spec)
            or not training._same_native(payload["provenance"], comparison._provenance(expected_provenance))
            or payload["optimizer_param_names"] != [n for n, _ in parameters]
            or payload["runtime"] != training._runtime(device) or type(payload["model_training"]) is not bool):
        raise ValueError("Transformer checkpoint configuration/source/runtime mismatch")
    if payload["loss_config"] != training._loss_config(loss_config, "no_consistency"):
        raise ValueError("Transformer checkpoint loss configuration mismatch")
    training._count(payload["completed_steps"], "completed_steps")
    training._check_model_state(payload["model_state"], model)
    training._check_optimizer_state(payload["optimizer_state"], optimizer, parameters)
    sampling = training._sampling(payload["sampling_state"])
    if sampling["next_batch_index"] != payload["completed_steps"]:
        raise ValueError("checkpoint cursor mismatch")
    legacy = training._check_rng(payload["rng_state"], device)
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    model.train(payload["model_training"])
    random.setstate(payload["rng_state"]["python"])
    np.random.set_state(legacy)
    torch.set_rng_state(payload["rng_state"]["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(payload["rng_state"]["cuda"]["state"], device=device)
    return training.CheckpointProgress(payload["completed_steps"], sampling)


def load(condition: str, path: Path, expected_sha: str, *, model, optimizer, loss_config, provenance: dict,
         updates: int) -> int:
    """按条件读回检查点（先核 SHA），恢复模型、优化器和随机状态；返回完成步数。要求步数等于游标、不大于 U（09 §4.5）。"""
    if not training._sha(expected_sha):
        raise ValueError("expected_sha must be a lowercase SHA-256")
    path = Path(path)
    if condition in ("po_jepa_full", "po_jepa_no_consistency", "fixed_fourier"):
        if condition == "fixed_fourier":
            addon_conditions.check_model(condition, model)
        progress = training.load_training_checkpoint(path, expected_sha256=expected_sha, model=model, optimizer=optimizer,
                                                     loss_config=loss_config, condition=cd.LOSS_LABELS[condition],
                                                     expected_provenance=comparison._provenance(provenance))
    elif condition == "matched_transformer":
        progress = load_matched_transformer(path, expected_sha, model, optimizer, loss_config, provenance)
    elif condition == "transformer_consistency":
        progress = addon_conditions.load_transformer_consistency_checkpoint(path, expected_sha, model, optimizer,
                                                                           loss_config, provenance)
    elif condition == "koopman":
        progress = addon_conditions.load_koopman_checkpoint(path, expected_sha, model, optimizer, loss_config, provenance)
    elif condition in cd.AC_CONDITIONS:
        progress = ac_training.load_checkpoint(condition, path,
                                               expected_sha, model, optimizer, comparison._provenance(provenance))
    else:
        raise ValueError(f"{condition} is not a trained DROID condition")
    completed = progress.completed_steps
    training._count(completed, "completed_steps")
    if progress.sampling_state["next_batch_index"] != completed:
        raise ValueError("checkpoint cursor differs from its completed steps")
    if completed > updates:
        raise ValueError(f"checkpoint has {completed} steps, more than U = {updates}")
    return completed
