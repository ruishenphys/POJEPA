"""16 §5：run.py check-addon——6 个新条件在真实特征上各做一次检查，写 <输出>/check-addon.json。

每个条件：comparison.build_condition 构造；初始权重哈希（v7jobs._model_hash）和 08 §1.3 的关系（time_only、space_only、
fixed_fourier、po_jepa_rollout_sup 等于 po_jepa_full，transformer_consistency 等于 matched_transformer，koopman 只记录）；参数量和预登记
（comparison._validate_registration，koopman 另记 K）；v7 check 的同一批 4 个窗口上的一次真实更新
（comparison._observed_update：读出次数等于 addon.conditions.expected_readouts），之后每个参数的梯度都存在且有限；
再用新建的模型做 TIMED_UPDATES 次正常更新（v7 check 的同一个 100×4 查询序列）并计时。依赖和设备由 entry 先核。
"""

import math

import platform

import sys

import time

import numpy as np

import torch

from .. import cache, checks as check, comparison, data, resources, v7jobs
from . import conditions

TIMED_UPDATES = 100
REFERENCE = dict(po_jepa_time_only="po_jepa_full", po_jepa_space_only="po_jepa_full", fixed_fourier="po_jepa_full",
                 transformer_consistency="matched_transformer", koopman=None, po_jepa_rollout_sup="po_jepa_full")


def initial_relations(hashes: dict) -> dict:
    """08 §1.3：hashes 含 8 个名字（6 个新条件和两个参照）的初始哈希。返回每个新条件的参照名（koopman 为 None）；
    不相等就报错。"""
    for condition, reference in REFERENCE.items():
        if reference is not None and hashes[condition] != hashes[reference]:
            raise ValueError(f"{condition} 的初始权重和 {reference} 不同（08 §1.3）")
    return dict(REFERENCE)


def _diagnostic(condition, features, device, queries, registration, sequence):
    """一次真实更新：读出次数、每个参数的梯度和更新量、参数量和预登记。返回 (初始哈希, 记录)。"""
    model, loss = comparison.build_condition(condition, device)
    initial = v7jobs._model_hash(model)
    comparison._validate_registration(condition, model, sequence, registration)
    optimizer = comparison.make_optimizer(model)
    batch = data.load_training_batch(features, queries, device=str(device))
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result, scales = comparison._observed_update(condition, model, optimizer, batch, loss, 0)
    resources.synchronize(device)
    memory = check._memory(device) if device.type == "cuda" else None
    parameters = {}
    for name, p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise ValueError(f"{condition}: parameter {name} has no finite gradient")
        parameters[name] = dict(gradient=check.tensor_stats(p.grad), update=check.tensor_stats(p.detach().cpu()-before[name]))
    koopman = condition == "koopman"
    record = dict(loss_label=conditions.LOSS_LABELS[condition], time_weight=loss.time_weight, space_weight=loss.space_weight,
                  parameters=comparison.parameter_counts(model), koopman_modes=model.config.modes if koopman else None,
                  initial_model_sha256=initial, readouts=dict(observed=len(scales), expected=conditions.expected_readouts(condition, model)),
                  losses=result.loss_values, grad_max_abs=result.grad_max_abs, grad_norm=result.grad_norm,
                  gradients=parameters, memory=memory)
    return initial, record


def _timed(condition, features, device):
    """TIMED_UPDATES 次正常更新（不挂钩子、不做诊断），v7 check 的 100×4 查询序列；返回耗时和每步损失。"""
    model, loss = comparison.build_condition(condition, device)
    optimizer = comparison.make_optimizer(model)
    sequence = check.make_query_sequence()[:TIMED_UPDATES]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    resources.synchronize(device)
    started, losses = time.perf_counter(), []
    for completed, rows in enumerate(sequence):
        batch = data.load_training_batch(features, rows, device=str(device))
        result = comparison.condition_step(condition, model, optimizer, batch, loss, completed)
        if result.completed_steps != completed+1 or any(
                value is not None and not math.isfinite(value) for value in result.loss_values.values()):
            raise ValueError(f"{condition}: timed update count or finite losses failed")
        losses.append(result.loss_values["jepa"])
        del batch
    resources.synchronize(device)
    seconds = time.perf_counter()-started
    if not (math.isfinite(seconds) and seconds > 0):
        raise ValueError("nonfinite update timing")
    return dict(updates=len(sequence), seconds=seconds, seconds_per_update=seconds/len(sequence), jepa=losses,
                memory=check._memory(device) if device.type == "cuda" else None)


def check_addon(features, output, device) -> dict:
    """output 是已建好的目录；写 output/check-addon.json 并返回其内容。"""
    with resources.isolated_randomness(device):
        hardware = check.device_metadata(device, output) if device.type == "cuda" else dict(device="cpu")
        assets = data.approved_cache_metadata(features)
        registration, sequence, _ = comparison.preregister()
        queries = [data.WindowQuery(i, 5, d) for i, d in enumerate((1., 2., 4., 8.))]   # 和 v7 check 的诊断批相同
        hashes = {name: v7jobs._model_hash(comparison.build_condition(name, device)[0])
                  for name in ("po_jepa_full", "matched_transformer")}
        records = {}
        for condition in conditions.ADDON_CONDITIONS:
            print(f"check-addon {condition}", flush=True)
            hashes[condition], records[condition] = _diagnostic(condition, features, device, queries, registration, sequence)
            records[condition]["timed"] = _timed(condition, features, device)
        relations = initial_relations(hashes)
        for condition, reference in relations.items():
            records[condition]["initial_reference"] = reference
    report = dict(hardware=hardware, cache_identity=assets, conditions=records,
                  reference_initial_model_sha256={name: hashes[name] for name in ("po_jepa_full", "matched_transformer")},
                  parameter_match=registration["parameter_match"], timed_updates=TIMED_UPDATES,
                  python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                  platform=platform.platform(), cpu_threads=torch.get_num_threads(), check_completed=True)
    cache._write_json(output/"check-addon.json", report)
    return report
