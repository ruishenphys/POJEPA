"""07 §1、§2.4、§2.5，08：5 个新条件的构造、优化器、单步、读出计数、一致性梯度占比、登记核对、检查点和读回。

comparison、evaluation 的共享函数开头各有一行“是新条件就交给本模块”，v7 条件不经过这里。comparison 在模块顶层
导入本模块，所以本模块用到 comparison 时在函数里导入。
"""

import hashlib

import io

import math

import os

import random

import tempfile

from dataclasses import asdict, fields, replace

from pathlib import Path

import numpy as np

import torch

from .. import checks as check
from .. import training
from . import calibration
from po_jepa.baselines import TokenTransformer, TransformerSpec
from po_jepa.fourier import FourierSpectrum, build_fixed_fourier
from po_jepa.generic_consistency import spatial_predictions, temporal_predictions
from po_jepa.koopman import KoopmanConfig, KoopmanPredictor, build_koopman, choose_modes, parameter_count
from po_jepa.losses import (LossConfig, jepa_distance, spatial_consistency_loss, temporal_consistency_loss,
                            weighted_total_loss)
from po_jepa.model import POJEPA
from po_jepa.spectral import GraphSpectrum

ADDON_CONDITIONS = ("po_jepa_time_only", "po_jepa_space_only", "transformer_consistency", "fixed_fourier", "koopman")

# 08 §1.1 的损失标签（training._loss_config 的第二个参数，检查点里存的 condition / loss_config 快照用它）
LOSS_LABELS = dict(po_jepa_time_only="time_only", po_jepa_space_only="space_only", transformer_consistency="full",
                   fixed_fourier="full", koopman="no_consistency")

KOOPMAN_FIELDS = frozenset({"schema_version", "completed_steps", "condition", "model_state", "optimizer_state",
    "optimizer_param_names", "model_config", "loss_config", "optimizer_config", "provenance", "rng_state",
    "sampling_state", "model_training", "scheduler_state", "scaler_state", "runtime"})


def _require(condition) -> None:
    if condition not in ADDON_CONDITIONS:
        raise ValueError("not an add-on condition")


def expected_type(condition, model):
    """07 §2.5：run_prepared_evaluation 的 expected_type。PO 类另核谱模块：fixed_fourier 要 FourierSpectrum，
    两个单项消融要 GraphSpectrum（type is，不收子类），不符报和 v7 相同的 TypeError。"""
    _require(condition)
    if condition == "transformer_consistency":
        return TokenTransformer
    if condition == "koopman":
        return KoopmanPredictor
    spectrum = FourierSpectrum if condition == "fixed_fourier" else GraphSpectrum
    if isinstance(model, POJEPA) and type(model.spectrum) is not spectrum:
        raise TypeError("condition/model type mismatch")
    return POJEPA


def check_model(condition, model) -> None:
    """新条件的模型类型（expected_type 加 isinstance），单步、保存、读回前都核。"""
    if not isinstance(model, expected_type(condition, model)):
        raise TypeError("condition/model type mismatch")


def _registered_transformer(device):
    """和 comparison.build_condition 的 matched_transformer 分支相同：登记规格、model_seed，初始权重因此逐位相同。"""
    from .. import comparison
    registration = comparison.choose_architecture()
    if not registration["matched"]:
        raise ValueError("parameter matching failed; comparison blocked without relaxing 1%")
    return comparison._make_transformer(TransformerSpec(**registration["selected"]), device)


def koopman_config(config: dict) -> KoopmanConfig:
    """仿真的 KoopmanConfig：m 和 E 取协议的 model 块，K 按 08 §5.6 的规则对 PO-JEPA 的实际参数量取（仿真：v7 协议 103，v6 协议 76）。"""
    from .. import comparison
    po = comparison.parameter_counts(check.build_full_model(config, torch.device("cpu")))["total"]
    action_dim = config["model"]["dynamics"]["action_dim"]
    encoder_dim = config["model"]["projection"]["encoder_dim"]
    return KoopmanConfig(modes=choose_modes(po, action_dim=action_dim, encoder_dim=encoder_dim),
                         action_dim=action_dim, encoder_dim=encoder_dim)


def build(condition, device):
    """07 §2.4 build_condition：返回 (model, loss)。权重按 08 §1.1：time_only 只留协议的 λ_t，space_only 只留 λ_s，
    transformer_consistency、fixed_fourier 用补跑标定，koopman 两个都为 0。"""
    _require(condition)
    config = check.load_protocol()
    loss = LossConfig(**{f.name: config["loss"][f.name] for f in fields(LossConfig)})
    if condition == "po_jepa_time_only":
        return check.build_full_model(config, device), replace(loss, space_weight=0.)
    if condition == "po_jepa_space_only":
        return check.build_full_model(config, device), replace(loss, time_weight=0.)
    if condition == "koopman":
        return build_koopman(koopman_config(config), check.model_seed(), device), replace(loss, time_weight=0., space_weight=0.)
    weights = calibration.load_addon_weights(condition)
    if condition == "fixed_fourier":
        model = build_fixed_fourier(config["model"], check.model_seed(), device)
    else:
        model = _registered_transformer(device)
    return model, replace(loss, time_weight=weights["time_weight"], space_weight=weights["space_weight"])


def _koopman_parameters(model) -> list:
    """Koopman 的全部参数：有限、可训练、模型设备 float32，不筛掉任何一个。"""
    if type(model) is not KoopmanPredictor:
        raise ValueError("expected a KoopmanPredictor")
    parameters = list(model.named_parameters())
    device = model.p_in.weight.device
    if any(not p.requires_grad or p.dtype != torch.float32 or p.device != device
           or not torch.isfinite(p).all() for _, p in parameters):
        raise ValueError("all Koopman parameters must be finite trainable model-device float32")
    return parameters


def make_koopman_optimizer(model) -> torch.optim.AdamW:
    """08 §5.4：AdamW(lr=1e-3, betas=(.9,.999), eps=1e-8, weight_decay=0)，全部参数恰好一次（和 Transformer 相同）。"""
    parameters = _koopman_parameters(model)
    return torch.optim.AdamW([p for _, p in parameters], lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0.)


def _koopman_optimizer(model, optimizer) -> list:
    """和 comparison._tf_optimizer 相同的覆盖、顺序、超参核对。"""
    parameters = _koopman_parameters(model)
    if (type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1
        or [id(p) for p in optimizer.param_groups[0]["params"]] != [id(p) for _, p in parameters]):
        raise ValueError("Koopman AdamW coverage/order differs")
    expected = make_koopman_optimizer(model).state_dict()["param_groups"]
    if not training._same_native(optimizer.state_dict()["param_groups"], expected):
        raise ValueError("Koopman optimizer settings differ from fixed AdamW")
    return parameters


def _train_mode(model) -> None:
    if not all(m.training for m in model.modules()) or not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("explicit train mode and enabled gradients required")


def _finish(parameters, optimizer, loss_config, completed_steps, losses, name):
    """反传之后的共同部分（和 matched_transformer 的单步相同）：梯度齐全有限、裁剪、调度步、更新后有限。"""
    if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in parameters):
        raise ValueError(f"required {name} gradient missing or nonfinite")
    maximum = max(p.grad.detach().abs().max().item() for _, p in parameters)
    grad_norm = training.clip_gradients([p for _, p in parameters], loss_config.grad_clip_norm)
    training.scheduled_step(optimizer, loss_config, completed_steps)
    if any(not torch.isfinite(p).all() for _, p in parameters):
        raise ValueError(f"updated {name} parameter nonfinite")
    training._optimizer_finite(optimizer)
    return training.TrainingStepResult(completed_steps+1, losses, maximum, grad_norm)


def transformer_consistency_step(model, optimizer, batch, loss_config, completed_steps):
    """08 §3：JEPA 项另做一次和 matched_transformer 相同的前向（原动作、原时长），加 3 次时间、2 次空间前向。"""
    from .. import comparison
    training._count(completed_steps, "completed_steps")
    training._loss_config(loss_config, "full")
    parameters = comparison._tf_optimizer(model, optimizer)
    comparison._tf_batch(batch, model)
    _train_mode(model)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch.raw_z, batch.controls, batch.duration, batch.transfer.source)
    jepa = jepa_distance(prediction, batch.raw_target, config=loss_config)
    temporal = temporal_predictions(model, batch.raw_z, batch.controls, batch.tau, batch.sigma, batch.transfer.source)
    time = temporal_consistency_loss(temporal, config=loss_config)
    spatial = spatial_predictions(model, batch.raw_z, batch.controls, batch.duration, batch.transfer)
    space = spatial_consistency_loss(spatial, config=loss_config)
    total = weighted_total_loss(jepa, time, space, config=loss_config,
                                consistency_scale=training.consistency_warmup_scale(loss_config, completed_steps))
    losses = dict(jepa=jepa.mean, time=time.mean, space=space.mean, weighted_time=total.weighted_time,
                  weighted_space=total.weighted_space, total=total.total)
    if any(not torch.isfinite(value) for value in losses.values()):
        raise ValueError("all six training losses must be finite before backward")
    total.total.backward()
    return _finish(parameters, optimizer, loss_config, completed_steps,
                   {k: value.detach().item() for k, value in losses.items()}, "Transformer")


def koopman_step(model, optimizer, batch, loss_config, completed_steps):
    """08 §5.4：和 matched_transformer 的单步结构相同，一次 16×16 前向，只有 JEPA 损失。"""
    from .. import comparison
    training._count(completed_steps, "completed_steps")
    training._loss_config(loss_config, "no_consistency")
    parameters = _koopman_optimizer(model, optimizer)
    comparison._tf_batch(batch, model)
    _train_mode(model)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch.raw_z, batch.controls, batch.duration, batch.transfer.source)
    loss = jepa_distance(prediction, batch.raw_target, config=loss_config).mean
    if not torch.isfinite(loss):
        raise ValueError("nonfinite JEPA loss")
    loss.backward()
    value = loss.detach().item()
    return _finish(parameters, optimizer, loss_config, completed_steps,
                   dict(jepa=value, time=None, space=None, weighted_time=0., weighted_space=0., total=value), "Koopman")


def condition_step(condition, model, optimizer, batch, loss_config, completed_steps):
    """07 §2.4 condition_step：PO 类调 training.train_step（标签见 LOSS_LABELS），另两个调本模块的单步。"""
    _require(condition)
    check_model(condition, model)
    if condition == "transformer_consistency":
        return transformer_consistency_step(model, optimizer, batch, loss_config, completed_steps)
    if condition == "koopman":
        return koopman_step(model, optimizer, batch, loss_config, completed_steps)
    return training.train_step(model, optimizer, batch, loss_config=loss_config,
                               completed_steps=completed_steps, condition=LOSS_LABELS[condition])


def expected_readouts(condition, model) -> int:
    """07 §2.4：每次更新 p_out 被调用的次数。PO 类 5 × readouts_per_phi（残差读出 10），transformer_consistency 6，koopman 2。"""
    _require(condition)
    if condition == "transformer_consistency":
        return 6
    if condition == "koopman":
        return 2
    return 5*check.readouts_per_phi(model)


def observed_update(condition, model, optimizer, batch, config, completed):
    """comparison._observed_update 的新条件版：只有读出次数换成 expected_readouts；单步经 comparison.condition_step。"""
    from .. import comparison, learning
    _require(condition)
    scales = []
    handle = model.p_out.register_forward_hook(lambda owner, args, value: scales.append(check.tensor_stats(value)))
    try:
        result = comparison.condition_step(condition, model, optimizer, batch, config, completed)
    finally:
        handle.remove()
    if (result.completed_steps != completed+1 or len(scales) != expected_readouts(condition, model)
        or set(result.loss_values) != learning.LOSS_NAMES
        or not math.isfinite(result.grad_max_abs) or result.grad_max_abs < 0
        or any(value is not None and not math.isfinite(value) for value in result.loss_values.values())):
        raise ValueError("condition update count/readouts/finite diagnostics mismatch")
    return result, scales


def consistency_share(condition, model, batch, loss_config, completed):
    """07 §2.4 consistency_share：间隔、warmup_scale、share 的公式和键和 comparison.consistency_share 相同；
    PO 类用 training.consistency_gradient_norms，transformer_consistency 用 calibration.generic_gradient_norms，koopman 返回 None。"""
    _require(condition)
    interval = check.load_protocol()["monitoring"]["consistency_interval"]
    if condition == "koopman" or completed % interval:
        return None
    if condition == "transformer_consistency":
        norms = calibration.generic_gradient_norms(model, batch, loss_config=loss_config)
    else:
        norms = training.consistency_gradient_norms(model, batch, loss_config=loss_config)
    scale = training.consistency_warmup_scale(loss_config, completed)
    weights = dict(time=loss_config.time_weight, space=loss_config.space_weight)
    return dict(**norms, warmup_scale=scale,
        share={k: w*scale*norms["grad_"+k]/norms["grad_jepa"] if norms["grad_jepa"] > 0 else None for k, w in weights.items()})


def validate_registration(condition, model, sequence, registration):
    """07 §2.4 _validate_registration：PO 类核 po_parameters，transformer_consistency 核 transformer_parameters，
    koopman 核 K 等于按 08 §5.6 规则对 po_parameters 取的值、实际参数量等于公式；其余核对和 v7 相同。"""
    from .. import comparison
    _require(condition)
    if not sequence or any(not rows for rows in sequence):
        raise ValueError("registered trainable condition and nonempty sequence required")
    match = registration["parameter_match"]
    if not comparison.parameter_match(match["po_parameters"], match["transformer_parameters"])["matched"]:
        raise ValueError("capacity matching failed; fixed 1% unchanged")
    if condition == "koopman":
        check_model(condition, model)
        rule = choose_modes(match["po_parameters"], action_dim=model.config.action_dim, encoder_dim=model.config.encoder_dim)
        if model.config.modes != rule:
            raise ValueError("Koopman modes differ from the parameter-matching rule")
        expected = parameter_count(model.config)
    else:
        expected = match["transformer_parameters" if condition == "transformer_consistency" else "po_parameters"]
    if comparison.parameter_counts(model)["total"] != expected:
        raise ValueError("actual condition model count differs from preregistration")
    if (comparison.sequence_sha(sequence) != registration["query_sequence_sha256"]
        or registration["consumer_sha256"] != comparison.consumer_sources()
        or registration["comparison_config_sha256"] != comparison.data.cache.file_sha(comparison.ROOT/"configs/train/comparison.yaml")):
        raise ValueError("shared queries/config/consumer source differs")


def koopman_model_config(model) -> dict:
    """08 §5.7：检查点 model_config 是 KoopmanConfig 的原生快照。"""
    model.config.__post_init__()
    return training._native_config(asdict(model.config))


def _publish(path: Path, payload: dict) -> str:
    """同目录临时文件写完、fsync，再硬链接发布（目标已存在就失败，不覆盖）；返回文件 SHA256。"""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix="."+path.name+".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _save_transformer(path, model, optimizer, config, completed, sampling, provenance) -> str:
    """和 comparison.save_checkpoint 的 matched_transformer 分支相同的格式，只是 loss_config 存 "full" 标签的快照（08 §3）。"""
    from .. import comparison
    parameters = comparison._tf_optimizer(model, optimizer)
    training._count(completed, "completed_steps")
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    payload = dict(schema="F05_transformer_checkpoint_v1", config=asdict(model.spec),
        model_state=training._cpu_tree(dict(model.state_dict())), optimizer_state=training._cpu_tree(optimizer.state_dict()),
        optimizer_param_names=[n for n, _ in parameters], provenance=provenance,
        completed_steps=completed, sampling_state=training._sampling(sampling), model_training=model.training,
        loss_config=training._loss_config(config, "full"), runtime=training._runtime(parameters[0][1].device),
        rng_state=training._rng_state(parameters[0][1].device))
    if payload["sampling_state"]["next_batch_index"] != completed:
        raise ValueError("checkpoint cursor must equal completed updates")
    training._check_model_state(payload["model_state"], model)
    training._check_optimizer_state(payload["optimizer_state"], optimizer, parameters)
    return _publish(path, payload)


def _save_koopman(path, model, optimizer, config, completed, sampling, provenance) -> str:
    """08 §5.7：schema_version 1 的 16 个字段（和 training.save_training_checkpoint 相同），condition 为 "koopman"，
    model_config 为 KoopmanConfig 快照，loss_config 为 "no_consistency" 标签的快照。"""
    parameters = _koopman_optimizer(model, optimizer)
    training._count(completed, "completed_steps")
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    device = parameters[0][1].device
    payload = dict(schema_version=1, completed_steps=completed, condition="koopman",
        model_state=training._cpu_tree(dict(model.state_dict())), optimizer_state=training._cpu_tree(optimizer.state_dict()),
        optimizer_param_names=[[n for n, _ in parameters]], model_config=koopman_model_config(model),
        loss_config=training._loss_config(config, "no_consistency"), optimizer_config=training._optimizer_config(optimizer),
        provenance=training._plain(training._provenance(provenance)), rng_state=training._rng_state(device),
        sampling_state=training._sampling(sampling), model_training=model.training, scheduler_state=None,
        scaler_state=None, runtime=training._runtime(device))
    if payload["sampling_state"]["next_batch_index"] != completed:
        raise ValueError("checkpoint cursor must equal completed updates")
    training._check_model_state(payload["model_state"], model)
    training._check_optimizer_state(payload["optimizer_state"], optimizer, parameters)
    training._check_rng(payload["rng_state"], device)
    return _publish(path, payload)


def save_checkpoint(condition, path, model, optimizer, config, completed, sampling, provenance) -> str:
    """07 §2.4 save_checkpoint：PO 类走 training.save_training_checkpoint（标签见 LOSS_LABELS）；
    transformer_consistency 走 Transformer 格式、loss_config 用 "full"；koopman 走 08 §5.7 的格式。返回文件 SHA256。"""
    from .. import comparison
    _require(condition)
    check_model(condition, model)
    provenance = comparison._provenance(provenance)
    if condition == "transformer_consistency":
        return _save_transformer(path, model, optimizer, config, completed, sampling, provenance)
    if condition == "koopman":
        return _save_koopman(path, model, optimizer, config, completed, sampling, provenance)
    return training.save_training_checkpoint(path, model=model, optimizer=optimizer, loss_config=config,
        condition=LOSS_LABELS[condition], completed_steps=completed, sampling_state=sampling, provenance=provenance)


def _restore(payload, model, optimizer, device):
    """预检通过之后恢复模型、AdamW、模式和 Python/NumPy/CPU/CUDA 随机状态（和 v7 两个读法的最后几行相同）。"""
    legacy = training._check_rng(payload["rng_state"], device)
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    model.train(payload["model_training"])
    random.setstate(payload["rng_state"]["python"])
    np.random.set_state(legacy)
    torch.set_rng_state(payload["rng_state"]["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(payload["rng_state"]["cuda"]["state"], device=device)


def load_transformer_consistency_checkpoint(path, expected_sha, model, optimizer, config, expected_provenance):
    """comparison.load_transformer_checkpoint 的 transformer_consistency 版：loss_config 要等于 config 的 "full" 快照。"""
    from .. import comparison
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
    if payload["loss_config"] != training._loss_config(config, "full"):
        raise ValueError("Transformer checkpoint loss configuration mismatch")
    training._count(payload["completed_steps"], "completed_steps")
    training._check_model_state(payload["model_state"], model)
    training._check_optimizer_state(payload["optimizer_state"], optimizer, parameters)
    sampling = training._sampling(payload["sampling_state"])
    if sampling["next_batch_index"] != payload["completed_steps"]:
        raise ValueError("checkpoint cursor mismatch")
    _restore(payload, model, optimizer, device)
    return training.CheckpointProgress(payload["completed_steps"], sampling)


def load_koopman_checkpoint(path, expected_sha, model, optimizer, config, expected_provenance):
    """08 §5.7：核 SHA、16 个字段、condition、loss/model/optimizer 配置快照、来源、参数名、运行环境、模式，
    model_state 的名字形状精度，AdamW 状态，游标和随机状态，再 load_state_dict(strict=True)。"""
    from .. import comparison
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("checkpoint SHA mismatch before loading")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    parameters = _koopman_optimizer(model, optimizer)
    device = parameters[0][1].device
    if (type(payload) is not dict or set(payload) != KOOPMAN_FIELDS
        or type(payload["schema_version"]) is not int or payload["schema_version"] != 1):
        raise ValueError("unsupported Koopman checkpoint schema/fields")
    training._count(payload["completed_steps"], "completed_steps")
    if (payload["condition"] != "koopman"
        or not training._same_native(payload["loss_config"], training._loss_config(config, "no_consistency"))
        or not training._same_native(payload["model_config"], koopman_model_config(model))
        or not training._same_native(payload["optimizer_config"], training._optimizer_config(optimizer))
        or not training._same_native(payload["provenance"], training._provenance(comparison._provenance(expected_provenance)))
        or not training._same_native(payload["optimizer_param_names"], [[n for n, _ in parameters]])
        or not training._same_native(payload["runtime"], training._runtime(device))
        or type(payload["model_training"]) is not bool or payload["scheduler_state"] is not None
        or payload["scaler_state"] is not None):
        raise ValueError("Koopman checkpoint condition/configuration/source/order/runtime/mode mismatch")
    training._check_model_state(payload["model_state"], model)
    training._check_optimizer_state(payload["optimizer_state"], optimizer, parameters)
    sampling = training._sampling(payload["sampling_state"])
    if sampling["next_batch_index"] != payload["completed_steps"]:
        raise ValueError("checkpoint cursor mismatch")
    _restore(payload, model, optimizer, device)
    return training.CheckpointProgress(payload["completed_steps"], sampling)


def load_selected(condition, selected, *, model, optimizer, config, expected_provenance, root):
    """07 §2.5 load_selected_checkpoint 的新条件版：前面的核对（步数、均值、来源、路径不越界、SHA、步数一致）和
    v7 分支逐行相同，然后按条件读，最后 model.eval()。"""
    _require(condition)
    if type(selected.get("step")) is not int or selected["step"] < 0:
        raise ValueError("trained condition and explicit successful validation step required")
    if type(selected["mean"]) not in (int, float) or not math.isfinite(selected["mean"]) or selected["mean"] < 0 or selected["checkpoint_provenance"] != expected_provenance:
        raise ValueError("invalid validation future mean or original checkpoint provenance")
    path = (Path(root)/selected["checkpoint"]).resolve()
    if not path.is_relative_to(Path(root).resolve()): raise ValueError("selected checkpoint escaped its run")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != selected["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA mismatch before loading")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if payload["completed_steps"] != selected["step"]:
        raise ValueError("selected successful step differs before any state restore")
    check_model(condition, model)
    if condition == "transformer_consistency":
        progress = load_transformer_consistency_checkpoint(path, selected["checkpoint_sha256"], model, optimizer, config, expected_provenance)
    elif condition == "koopman":
        progress = load_koopman_checkpoint(path, selected["checkpoint_sha256"], model, optimizer, config, expected_provenance)
    else:
        progress = training.load_training_checkpoint(path, expected_sha256=selected["checkpoint_sha256"], model=model,
            optimizer=optimizer, loss_config=config, condition=LOSS_LABELS[condition], expected_provenance=expected_provenance)
    if progress.completed_steps != selected["step"]: raise ValueError("selected successful step differs from checkpoint")
    model.eval()
    return progress
