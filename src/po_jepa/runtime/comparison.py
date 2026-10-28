"""三条件公平对照的查询、更新、验证选点与保存。

完整 PO/双零一致性消融/普通 Transformer 使用共用查询和固定预算；
公式、损失和指标沿共同实现。真实运行由统一工作流先完成资源门槛。
"""

import argparse

from dataclasses import asdict, fields, replace

import hashlib

import io

import json

import math

import os

from pathlib import Path

import random

import tempfile

import time

import numpy as np

import torch

import yaml

from . import checks as check
from . import learning as learning
from . import data as data
from . import training as training
from po_jepa.actions import split_action_segments, concatenate_action_segments

from po_jepa.baselines import TokenTransformer, TransformerSpec, IdentityPredictor

from po_jepa.geometry import apply_spatial_transfer, build_spatial_transfer

from po_jepa.losses import (LossConfig, input_scale, TemporalPredictions, SpatialPredictions, jepa_distance,
                            temporal_predictions, spatial_predictions, _temporal_time_contract,
                            _temporal_raw_contract)

from po_jepa.metrics import SampleMetric, temporal_composition_defect, spatial_commutation_defect

from po_jepa.model import POJEPA

ROOT = Path(__file__).resolve().parents[3]

CONDITIONS = ("matched_transformer", "po_jepa_no_consistency", "po_jepa_full")

def consumer_sources():
    """当前消费者的.py摘要，与冻结编码生成来源及独立YAML摘要分开。"""
    sources = data.consumer_sources()
    for name in ("src/po_jepa/runtime/learning.py", "src/po_jepa/baselines.py",
                 "src/po_jepa/runtime/comparison.py"):
        sources[name] = data.cache.file_sha(ROOT/name)
    return sources

def sequence_sha(sequence) -> str:
    """对实际查询列表做规范JSON摘要；相同seed不能代替这个实际序列身份。"""
    return hashlib.sha256(data.cache._json_bytes([[asdict(q) for q in rows] for rows in sequence])).hexdigest()

def load_comparison_config() -> dict:
    """读取固定三条件、1%参数匹配及有限架构候选预登记，不改研究协议。"""
    config = yaml.safe_load((ROOT/"configs/train/comparison.yaml").read_text(encoding="utf-8"))
    if (config["schema_version"] != 1 or config["conditions"] != list(CONDITIONS)
        or config["parameter_relative_tolerance"] != .01 or config["untrained"] != "identity"
        or config["selection"] != "minimum_absolute_parameter_difference_then_lexicographic_spec"):
        raise ValueError("comparison preregistration differs from approved conditions/matching rule")
    actual = config["transformer"]
    expected = dict(activation="gelu", norm_first=False, grid_embedding="linear_xy_without_bias",
        duration_embedding="linear_cycles_without_bias", action_embedding="linear_ux_uy_start_end_with_bias",
        output="spatial_tokens_linear_to_raw_Z")
    if any(actual[k] != value for k,value in expected.items()):
        raise ValueError("configuration describes a different baseline implementation")
    return config

def parameter_match(po_count: int, transformer_count: int) -> dict:
    """整数交叉相乘核固定1%，分母为完整 PO 可训练预测器。"""
    if any(type(n) is not int or n <= 0 for n in (po_count, transformer_count)):
        raise ValueError("positive actual parameter counts required")
    difference = abs(transformer_count-po_count)
    return dict(po_parameters=po_count, transformer_parameters=transformer_count,
        absolute_difference=difference, relative_difference=difference/po_count,
        denominator="complete_PO_trainable_predictor", tolerance=.01, matched=difference*100 <= po_count)

def parameter_counts(model) -> dict:
    """按实际注册参数去重计数，固定buffer不计；逐顶层模块分解供审查。"""
    groups = {}
    for name, value in model.named_parameters():
        if value.requires_grad:
            key = name.split(".")[0]
            groups[key] = groups.get(key, 0)+value.numel()
    return dict(total=sum(groups.values()), modules=groups)

def _make_transformer(spec, device):
    """CPU局部seed1703构造后转指定设备；不消耗全局初始化/采样RNG。"""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1703)
        model = TokenTransformer(spec)
    return model.to(device)

def choose_architecture() -> dict:
    """仅按预登记3个FFN候选的实际参数差选择，不读取任何数据/效果指标。"""
    config = load_comparison_config()["transformer"]
    po = check.build_full_model(check.load_protocol(), torch.device("cpu"))
    po_counts = parameter_counts(po)
    rows = []
    for width in config["feedforward_candidates"]:
        spec = TransformerSpec(encoder_dim=config["encoder_dim"], action_dim=config["action_dim"],
            width=config["width"], heads=config["heads"], layers=config["layers"], feedforward=width,
            dropout=config["dropout"], layer_norm_epsilon=config["layer_norm_epsilon"])
        counts = parameter_counts(_make_transformer(spec, torch.device("cpu")))
        rows.append(dict(spec=asdict(spec), counts=counts, **parameter_match(po_counts["total"], counts["total"])))
    if not rows:
        raise ValueError("empty preregistered candidate set")
    winner = min(rows, key=lambda r:(r["absolute_difference"], tuple(r["spec"].values())))
    return dict(**parameter_match(po_counts["total"], winner["counts"]["total"]),
        po_modules=po_counts["modules"], transformer_modules=winner["counts"]["modules"],
        selected=winner["spec"], candidates=rows, selection=load_comparison_config()["selection"])

def preregister():
    """只构造CPU模型计数并生成共用查询，不编码、训练或评价；返回登记/序列/末状态。"""
    matching = choose_architecture()
    sequence, state = learning.make_query_sequence(check.load_protocol())
    return dict(schema="F05_code_preregistration_v1", conditions=list(CONDITIONS), identity_trained=False,
        parameter_match=matching, query_sequence_sha256=sequence_sha(sequence),
        consumer_sha256=consumer_sources(),
        generator_state_stage="after_complete_sequence_pre_generation",
        comparison_config_sha256=data.cache.file_sha(ROOT/"configs/train/comparison.yaml"),
        true_effect_comparison_run=False), sequence, state

def build_condition(condition: str, device):
    """共用固定模型工厂与初始化，仅按条件设置批准的损失权重。"""
    config = check.load_protocol()
    loss = LossConfig(**{f.name:config["loss"][f.name] for f in fields(LossConfig)})
    if condition == "identity":
        return IdentityPredictor().to(device), loss
    if condition not in CONDITIONS:
        raise ValueError("unapproved condition")
    if condition == "matched_transformer":
        registration = choose_architecture()
        if not registration["matched"]:
            raise ValueError("parameter matching failed; comparison blocked without relaxing 1%")
        model = _make_transformer(TransformerSpec(**registration["selected"]), device)
    else:
        model = check.build_full_model(config, device)
    if condition != "po_jepa_full":
        loss = replace(loss, time_weight=0., space_weight=0.)
    return model, loss

def _tf_parameters(model) -> list:
    """独立核TF的应训练参数，不绕过PO类型检查或筛掉冻结/额外参数。"""
    if type(model) is not TokenTransformer:
        raise ValueError("expected the registered ordinary Transformer")
    parameters = list(model.named_parameters())
    device = model.p_in.weight.device
    if any(not p.requires_grad or p.dtype != torch.float32 or p.device != device
           or not torch.isfinite(p).all() for _, p in parameters):
        raise ValueError("all Transformer parameters must be finite trainable model-device float32")
    return parameters

def make_optimizer(model):
    """全部预测器参数恰一次进入原AdamW设置；identity没有优化器。"""
    if isinstance(model, POJEPA):
        return training.make_optimizer(model)
    parameters = _tf_parameters(model)
    return torch.optim.AdamW([p for _, p in parameters], lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0.)

def _tf_optimizer(model, optimizer):
    """先核精确覆盖/顺序/超参；不能因为某次梯度为零就移走一部分参数。"""
    parameters = _tf_parameters(model)
    if (type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1
        or [id(p) for p in optimizer.param_groups[0]["params"]] != [id(p) for _, p in parameters]):
        raise ValueError("Transformer AdamW coverage/order differs")
    expected = make_optimizer(model).state_dict()["param_groups"]
    if not training._same_native(optimizer.state_dict()["param_groups"], expected):
        raise ValueError("Transformer optimizer settings differ from fixed AdamW")
    return parameters

def _tf_batch(batch, model):
    """独立TF训练边界：冻结特征/精确端点/来源；不调用要求PO模型的更新或假装PO。"""
    if not isinstance(batch, training.TrainingBatch):
        raise TypeError("explicit TrainingBatch required")
    from po_jepa.baselines import validate_inputs
    validate_inputs(batch.raw_z, batch.controls, batch.duration, batch.transfer.source,
                    model.encoder_dim, model.p_in.weight.device)
    _temporal_raw_contract(batch.raw_target, "raw_target")
    batch.transfer.__post_init__()
    if batch.raw_target.shape != batch.raw_z.shape or batch.raw_target.device != batch.raw_z.device:
        raise ValueError("raw target must match input shape/device")
    if any(x.requires_grad or x.grad_fn is not None for x in (batch.raw_z, batch.raw_target)):
        raise ValueError("prepared raw inputs/targets must be frozen")
    for name in ("duration", "tau", "sigma"):
        _temporal_time_contract(getattr(batch, name), name, batch.raw_z.shape[0])
    if not torch.equal(batch.duration, batch.tau+batch.sigma):
        raise ValueError("duration must equal actual tau+sigma")
    for value in (batch.observed_time_seconds, batch.target_time_seconds):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or value.device.type != "cpu"
            or value.shape != batch.duration.shape or value.requires_grad or torch.is_inference(value)
            or not torch.isfinite(value).all()):
            raise ValueError("physical endpoint seconds must be finite graph-free CPU float64[B]")
    expected = batch.observed_time_seconds+batch.duration*batch.controls.control_spec.control_period_seconds
    if (not torch.equal(batch.observed_time_seconds, batch.controls.source_origin_seconds)
        or not torch.equal(expected, batch.target_time_seconds) or torch.any(batch.target_time_seconds <= batch.observed_time_seconds)):
        raise ValueError("actual endpoint/control origin mismatch")
    if (type(batch.sample_ids) is not tuple or len(batch.sample_ids) != batch.raw_z.shape[0]
        or any(type(s) is not str or not s.strip() for s in batch.sample_ids)):
        raise ValueError("one explicit sample identity per row required")
    training._provenance(batch.provenance)

def condition_step(condition, model, optimizer, batch, loss_config, completed_steps):
    """成功才计步：PO 五次 Phi 联合更新，Transformer 仅 JEPA 一次反传。"""
    if condition in ("po_jepa_full", "po_jepa_no_consistency"):
        label = "full" if condition == "po_jepa_full" else "no_consistency"
        return training.train_step(model, optimizer, batch, loss_config=loss_config,
                                   completed_steps=completed_steps, condition=label)
    if condition != "matched_transformer":
        raise ValueError("identity and unapproved conditions cannot train")
    training._count(completed_steps, "completed_steps")
    training._loss_config(loss_config, "no_consistency")
    parameters = _tf_optimizer(model, optimizer)
    _tf_batch(batch, model)
    if not all(m.training for m in model.modules()) or not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("explicit train mode and enabled gradients required")
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch.raw_z, batch.controls, batch.duration, batch.transfer.source)
    loss = jepa_distance(prediction, batch.raw_target, config=loss_config).mean
    if not torch.isfinite(loss):
        raise ValueError("nonfinite JEPA loss")
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in parameters):
        raise ValueError("required Transformer gradient missing or nonfinite")
    maximum = max(p.grad.detach().abs().max().item() for _, p in parameters)
    grad_norm = training.clip_gradients([p for _, p in parameters], loss_config.grad_clip_norm)
    optimizer.step()
    if any(not torch.isfinite(p).all() for _, p in parameters):
        raise ValueError("updated Transformer parameter nonfinite")
    training._optimizer_finite(optimizer)
    return training.TrainingStepResult(completed_steps+1,
        dict(jepa=loss.detach().item(), time=None, space=None, weighted_time=0., weighted_space=0., total=loss.detach().item()), maximum, grad_norm)

@torch.no_grad()
def evaluate_batch(model, raw, controls, duration, transfer, target, loss_config):
    """同一输入计算未来距离/TCD/SCD，沿原范数，不额外训练或归一化。"""
    if model.training:
        raise ValueError("evaluation requires explicit eval mode")
    tau, sigma = duration/2, duration/2
    if isinstance(model, POJEPA):
        temporal = temporal_predictions(model, raw, controls, tau, sigma, transfer.source)
        spatial = spatial_predictions(model, raw, controls, duration, transfer)
    else:
        if not isinstance(model, (TokenTransformer, IdentityPredictor)):
            raise TypeError("unregistered evaluation predictor")
        pair = split_action_segments(controls, tau=tau, sigma=sigma)
        direct = model(raw, concatenate_action_segments(pair.first, pair.second), duration, transfer.source)
        first = model(raw, pair.first, tau, transfer.source)
        composed = model(first, pair.second, sigma, transfer.source)
        temporal = TemporalPredictions(direct, first, composed, direct-composed, tau.clone(), sigma.clone())
        fine = apply_spatial_transfer(model(raw, controls, duration, transfer.source), transfer)
        coarse = model(apply_spatial_transfer(raw, transfer), controls, duration, transfer.target)
        spatial = SpatialPredictions(fine, coarse, fine-coarse, duration.clone(), transfer.target)
    future = SampleMetric(jepa_distance(temporal.direct, target, config=loss_config).per_sample,
                          "future_jepa_distance", "dimensionless")
    tcd, scd = temporal_composition_defect(temporal), spatial_commutation_defect(spatial)
    return {m.name:m for m in (future, tcd, scd)}

def _provenance(value):
    """分别记录消费者源码与独立配置 SHA，不把 YAML 混入仅允许.py的源码字段。"""
    result = dict(value)
    result["source_files_sha256"] = dict(value["source_files_sha256"])
    for name in ("src/po_jepa/baselines.py", "src/po_jepa/runtime/comparison.py"):
        result["source_files_sha256"][name] = data.cache.file_sha(ROOT/name)
    result["f05_comparison_config_sha256"] = data.cache.file_sha(ROOT/"configs/train/comparison.yaml")
    return training._provenance(result)

def save_checkpoint(condition, path, model, optimizer, config, completed, sampling, provenance):
    """PO 沿共同保存器；Transformer 使用自身原生 schema，原子发布并拒覆盖。"""
    provenance = _provenance(provenance)
    if condition in ("po_jepa_full", "po_jepa_no_consistency"):
        return training.save_training_checkpoint(path, model=model, optimizer=optimizer, loss_config=config,
            condition="full" if condition == "po_jepa_full" else "no_consistency", completed_steps=completed,
            sampling_state=sampling, provenance=provenance)
    if condition != "matched_transformer":
        raise ValueError("only trainable registered conditions can checkpoint")
    parameters = _tf_optimizer(model, optimizer)
    training._count(completed, "completed_steps")
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    payload = dict(schema="F05_transformer_checkpoint_v1", config=asdict(model.spec),
        model_state=training._cpu_tree(dict(model.state_dict())), optimizer_state=training._cpu_tree(optimizer.state_dict()),
        optimizer_param_names=[n for n,_ in parameters], provenance=provenance,
        completed_steps=completed, sampling_state=training._sampling(sampling), model_training=model.training,
        loss_config=training._loss_config(config, "no_consistency"), runtime=training._runtime(parameters[0][1].device),
        rng_state=training._rng_state(parameters[0][1].device))
    if payload["sampling_state"]["next_batch_index"] != completed:
        raise ValueError("checkpoint cursor must equal completed updates")
    training._check_model_state(payload["model_state"], model)
    training._check_optimizer_state(payload["optimizer_state"], optimizer, parameters)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="."+path.name, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name); torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        return data.cache.file_sha(path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)

def load_transformer_checkpoint(path, expected_sha, model, optimizer, expected_provenance):
    """指定路径/摘要 weights_only 加载，先核原生状态合同再恢复。"""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("checkpoint SHA mismatch before loading")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    parameters = _tf_optimizer(model, optimizer)
    device = parameters[0][1].device
    if (payload["schema"] != "F05_transformer_checkpoint_v1" or payload["config"] != asdict(model.spec)
        or not training._same_native(payload["provenance"], _provenance(expected_provenance))
        or payload["optimizer_param_names"] != [n for n,_ in parameters]
        or payload["runtime"] != training._runtime(device) or type(payload["model_training"]) is not bool):
        raise ValueError("Transformer checkpoint configuration/source/runtime mismatch")
    fixed = check.load_protocol()["loss"]
    expected_loss = replace(LossConfig(**{f.name:fixed[f.name] for f in fields(LossConfig)}), time_weight=0., space_weight=0.)
    if payload["loss_config"] != training._loss_config(expected_loss, "no_consistency"):
        raise ValueError("Transformer checkpoint loss configuration mismatch")
    training._count(payload["completed_steps"], "completed_steps")
    training._check_model_state(payload["model_state"], model)
    training._check_optimizer_state(payload["optimizer_state"], optimizer, parameters)
    sampling = training._sampling(payload["sampling_state"])
    if sampling["next_batch_index"] != payload["completed_steps"]:
        raise ValueError("checkpoint cursor mismatch")
    legacy = training._check_rng(payload["rng_state"], device)
    model.load_state_dict(payload["model_state"], strict=True); optimizer.load_state_dict(payload["optimizer_state"])
    model.train(payload["model_training"])
    random.setstate(payload["rng_state"]["python"]); np.random.set_state(legacy)
    torch.set_rng_state(payload["rng_state"]["torch_cpu"])
    if device.type == "cuda": torch.cuda.set_rng_state(payload["rng_state"]["cuda"]["state"], device=device)
    return training.CheckpointProgress(payload["completed_steps"], sampling)

def validate_condition(model, features, output, device, config):
    """固定20×5×4 validation，三指标同窗；仅future均值用于选点，不读test。

    no-grad/eval下逐批释放图，复用范数，保存逐窗口值/来源与耗时。退出包括
    异常都恢复每个子模块原模式；已有参数/梯度身份检查不被TF适配削弱。
    """
    output.mkdir()
    queries = [data.WindowQuery(tid, start, duration) for tid in range(160,180)
               for start in (0,5,10,15,20) for duration in (1.,2.,4.,8.)]
    modes = {m:m.training for m in model.modules()}
    identity = check._parameter_identity(model)
    totals = dict(future_jepa_distance=0., tcd_frobenius=0., scd_frobenius=0.)
    count = 0
    if device.type == "cuda": torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    try:
        model.eval()
        with torch.no_grad():
            for offset in range(0,400,4):
                rows = queries[offset:offset+4]
                windows = data.load_windows(features, rows, split="validation", device=str(device))
                for window in windows: data._window_boundary(window, windows[0])
                grid = windows[0].inputs.grid
                transfer = build_spatial_transfer(grid, target_height=8, target_width=8)
                metrics = evaluate_batch(model, torch.cat([w.inputs.raw_z for w in windows]),
                    data.stack_controls([w.inputs.controls for w in windows]),
                    torch.cat([w.inputs.duration for w in windows]), transfer,
                    torch.cat([w.target.raw_z for w in windows]), config)
                values = {name:m.values.detach().cpu().double() for name,m in metrics.items()}
                if any(v.shape != (4,) or not torch.isfinite(v).all() for v in values.values()):
                    raise ValueError("four finite values per validation metric required")
                for name in totals: totals[name] += values[name].sum().item()
                count += 4
                data.cache._write_json(output/f"{offset:04d}.json", dict(queries=[asdict(q) for q in rows],
                    source_rows=[data._row_record(w) for w in windows], values={k:v.tolist() for k,v in values.items()}))
    finally:
        for module, mode in modes.items(): module.training = mode
    if device.type == "cuda": torch.cuda.synchronize(device)
    if count != 400 or check._parameter_identity(model) != identity:
        raise ValueError("validation count or unchanged parameter/gradient contract failed")
    means = {name:value/count for name,value in totals.items()}
    return dict(metric="future_jepa_distance", mean=means["future_jepa_distance"], means=means,
        windows=count, seconds=time.perf_counter()-start_time, modes_restored=True,
        selection_uses_only="future_jepa_distance mean", includes_loading_and_metric_logging=True)

def _validate_registration(condition, model, sequence, registration):
    """先核实际模型计数、封存查询/源码/配置；容量不匹配或身份过时则不创建运行目录。"""
    if condition not in CONDITIONS or not sequence or any(not rows for rows in sequence):
        raise ValueError("registered trainable condition and nonempty sequence required")
    match = registration["parameter_match"]
    if not parameter_match(match["po_parameters"], match["transformer_parameters"])["matched"]:
        raise ValueError("capacity matching failed; fixed 1% unchanged")
    expected = match["transformer_parameters" if condition == "matched_transformer" else "po_parameters"]
    if parameter_counts(model)["total"] != expected:
        raise ValueError("actual condition model count differs from preregistration")
    if (sequence_sha(sequence) != registration["query_sequence_sha256"]
        or registration["consumer_sha256"] != consumer_sources()
        or registration["comparison_config_sha256"] != data.cache.file_sha(ROOT/"configs/train/comparison.yaml")):
        raise ValueError("shared queries/config/consumer source differs")

def _selection(condition, model, optimizer, features, output, device, config, step, state, provenance, best):
    """每条件独立选checkpoint；严格更小才换最优，time/space指标不能干预选点。"""
    metric = validate_condition(model, features, output/"validations"/f"{step:04d}", device, config)
    if not math.isfinite(metric["mean"]) or metric["mean"] < 0:
        raise ValueError("nonfinite/negative validation future distance")
    checkpoint = output/"checkpoints"/f"step-{step:04d}.pt"
    digest = save_checkpoint(condition, checkpoint, model, optimizer, config, step,
        dict(kind="explicit_prepared_batches", next_batch_index=step, query_generator_state=state), provenance)
    row = dict(step=step, **metric, checkpoint=checkpoint.relative_to(output).as_posix(),
               checkpoint_sha256=digest, checkpoint_provenance=provenance)
    data.cache._write_json(output/"validations"/f"selection-{step:04d}.json", row)
    return row, row if best is None or row["mean"] < best["mean"] else best

def _observed_update(condition, model, optimizer, batch, config, completed):
    """记录实际raw读出（TF 1次；PO-JEPA 5次Phi，残差读出时每次2次），不加额外前向/backward；TF未算的缺陷为None。"""
    scales = []
    handle = model.p_out.register_forward_hook(lambda owner,args,value:scales.append(check.tensor_stats(value)))
    try:
        result = condition_step(condition, model, optimizer, batch, config, completed)
    finally:
        handle.remove()
    if (result.completed_steps != completed+1 or len(scales) != (1 if condition == "matched_transformer" else 5*check.readouts_per_phi(model))
        or set(result.loss_values) != learning.LOSS_NAMES
        or not math.isfinite(result.grad_max_abs) or result.grad_max_abs < 0
        or any(value is not None and not math.isfinite(value) for value in result.loss_values.values())):
        raise ValueError("condition update count/readouts/finite diagnostics mismatch")
    return result, scales

def run_prepared_condition(condition, model, optimizer, *, sequence, generator_state, registration,
                           features, output, device, loss_config, validation_interval, resume_from=None):
    """一次持有一个条件模型，连续计步、验证和保存；调用者须先核资源门槛。

    三条件传同一实际查询序列；旧来源不得重标后复用。各步同步计时，
    仅实际 CUDA 报 GPU 显存，CPU 不伪写为0。
    """
    _validate_registration(condition, model, sequence, registration)
    if type(validation_interval) is not int or validation_interval <= 0:
        raise ValueError("positive validation interval required")
    torch.Generator(device="cpu").set_state(generator_state)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for name in ("updates","checkpoints","validations"): (output/name).mkdir()
    data.cache._write_json(output/"query-sequence.json",dict(rows=[[asdict(q) for q in rows] for rows in sequence],
        query_sequence_sha256=registration["query_sequence_sha256"], generator_state=generator_state.tolist(),
        generator_state_stage="after_complete_sequence_pre_generation"))
    completed, attempted, best, validations = 0,0,None,[]
    try:
        if resume_from is not None:
            from .resume import prepare_condition
            completed,validations,best,role=prepare_condition(resume_from,output,condition,model,optimizer,
                sequence,generator_state,registration,features,device,loss_config,validation_interval)
        model.train()
        for index in range(completed,len(sequence)):
            rows=sequence[index]
            attempted = index+1
            batch = learning._learning_batch(features, rows, device, output, registration["query_sequence_sha256"])
            provenance = _provenance(batch.provenance)
            batch = replace(batch, provenance=provenance)
            if index == 0 and resume_from is None:
                role = provenance["evidence_role"]
                data.cache._write_json(output/"registration.json",dict(condition=condition,evidence_role=role,
                    preregistration=registration, provenance=provenance, actual_predictor_parameters=parameter_counts(model)))
                row,best = _selection(condition,model,optimizer,features,output,device,loss_config,0,generator_state,provenance,best)
                validations.append(row)
            if device.type == "cuda": torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            result,scales = _observed_update(condition,model,optimizer,batch,loss_config,completed)
            completed = result.completed_steps
            if device.type == "cuda": torch.cuda.synchronize(device)
            data.cache._write_json(output/"updates"/f"{completed:04d}.json",dict(condition=condition,
                completed_updates=completed,next_batch_index=completed,queries=[asdict(q) for q in rows],
                losses=result.loss_values,grad_max_abs=result.grad_max_abs,grad_norm_before_clip=result.grad_norm,
                grad_clip_norm=loss_config.grad_clip_norm,raw_predictions=scales,
                raw_target=check.tensor_stats(batch.raw_target),phi_calls=len(scales)//(1 if condition == "matched_transformer" else check.readouts_per_phi(model)),provenance=provenance,
                update_and_observer_seconds=time.perf_counter()-started,
                gpu_memory=check._memory(device) if device.type == "cuda" else None))
            del batch
            if completed % validation_interval == 0 or completed == len(sequence):
                row,best = _selection(condition,model,optimizer,features,output,device,loss_config,
                    completed,generator_state,provenance,best)
                validations.append(row)
        report = dict(status="completed_prepared_condition_only",condition=condition,evidence_role=role,
            completed_updates=completed,query_sequence_sha256=registration["query_sequence_sha256"],
            preregistration_sha256=hashlib.sha256(data.cache._json_bytes(registration)).hexdigest(),
            best=best,validations=validations,future_error_decreased=best["step"]>0,
            scientific_reproduction_completed=False)
        data.cache._write_json(output/"report.json",report)
        data.cache._write_json(output/"COMPLETE.json",dict(report_sha256=data.cache.file_sha(output/"report.json")))
        return report
    except Exception as error:
        data.cache._write_json(output/"FAILED.json",dict(condition=condition,completed_updates=completed,
            attempted_update=attempted,query_sequence_sha256=registration["query_sequence_sha256"],best=best,
            error_type=type(error).__name__,error=str(error),retry_performed=False,failed_step_may_have_modified_optimizer=True))
        raise

def comparison_table(reports, identity, registration):
    """只整理各自验证集选中点，不用TCD/SCD选模型，不将表格自动解释成因果收益。"""
    match = registration["parameter_match"]
    if set(reports) != set(CONDITIONS) or not parameter_match(match["po_parameters"],match["transformer_parameters"])["matched"]:
        raise ValueError("all three registered reports and capacity matching required")
    digest = hashlib.sha256(data.cache._json_bytes(registration)).hexdigest()
    budgets = {r["completed_updates"] for r in reports.values()}
    if (len(budgets) != 1 or min(budgets) <= 0
        or len({r["evidence_role"] for r in reports.values()}) != 1
        or any(r["condition"] != name or r["query_sequence_sha256"] != registration["query_sequence_sha256"]
               or r["preregistration_sha256"] != digest for name,r in reports.items())
        or identity["preregistration_sha256"] != digest):
        raise ValueError("actual budgets, roles, condition or shared query/preregistration identities differ")
    return dict(rows={name:r["best"] for name,r in reports.items()}, identity=identity,
        parameter_match=registration["parameter_match"],heldout_test_evaluation_run=False,
        interpretation="joint consistency and whole-scheme comparison only; no separate spectral/time/space attribution")

def identity_reference(features, output, device, loss_config, registration):
    """同一validation入口记录未训练identity；不建优化器、不分配更新预算。"""
    if registration["consumer_sha256"] != consumer_sources():
        raise ValueError("identity consumer registration differs")
    model = IdentityPredictor().to(device)
    result = validate_condition(model,features,output,device,loss_config)
    return dict(validation=result,completed_updates=0,trainable_parameters=0,
        preregistration_sha256=hashlib.sha256(data.cache._json_bytes(registration)).hexdigest())

