"""冻结特征上的五次 Phi/三项损失、单步 AdamW 与原生 checkpoint 保存恢复。

式14—17/算法1的特征与控制由调用者提供；导入不构造模型、读盘或设种子。
"""

from dataclasses import asdict, dataclass

import hashlib

import math

from numbers import Real

from pathlib import Path, PurePosixPath

import io

import os

import platform

import random

import tempfile

import numpy as np

import torch

from torch import Tensor

from po_jepa.actions import ActionSegments, split_action_segments, validate_action_segments

from po_jepa.geometry import SpatialTransfer

from po_jepa.losses import (LossConfig, _temporal_raw_contract, _temporal_time_contract,
    jepa_distance, temporal_predictions, temporal_consistency_loss,
    spatial_predictions, spatial_consistency_loss, weighted_total_loss)

from po_jepa.model import POJEPA

@dataclass(frozen=True)
class TrainingBatch:
    """冻结起点/标签与已知控制，构造仅存字段。

    raw_z/raw_target 为模型 float32 [B,N,E] 无量纲且无上游图。duration/
    tau/sigma 为 CPU float64 [B] 周期，duration 逐位等于原 tau+sigma。
    观测/目标时间为 CPU float64 [B] 秒，仅核端点，不入 Phi；sample_ids/
    provenance 记录逐批来源，真实来源由缓存读取编排核验。
    """
    raw_z: Tensor
    raw_target: Tensor
    controls: ActionSegments
    duration: Tensor
    tau: Tensor
    sigma: Tensor
    transfer: SpatialTransfer
    observed_time_seconds: Tensor
    target_time_seconds: Tensor
    sample_ids: tuple[str, ...]
    provenance: dict

@dataclass(frozen=True)
class TrainingStepResult:
    """成功后步数、六项detached Python浮点日志及梯度最大绝对值，不保存图。"""
    completed_steps: int
    loss_values: dict[str, float]
    grad_max_abs: float
    # v4：裁剪前的全局梯度 L2 范数（grad_max_abs 同样记裁剪前的值）；None 表示未记录。
    grad_norm: float | None = None

@dataclass(frozen=True)
class CheckpointProgress:
    """恢复完成的步数和显式batch进度；独立generator由调用者按返回状态恢复。"""
    completed_steps: int
    sampling_state: dict

def _count(value: int, name: str) -> None:
    """步数和batch索引须非负原生int，不把bool/浮点数当进度。"""
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative native int")

def _plain(value):
    """固定来源只容纳有限原生值/普通容器，拒绝回调、模型和隐藏句柄。"""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (tuple, list):
        return type(value)(_plain(item) for item in value)
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _plain(item) for key, item in value.items()}
    raise ValueError("metadata must contain only finite native values and ordinary containers")

def _sha(value, length=64) -> bool:
    """只核哈希格式；一串SHA本身不能证明真实编码或合法实验来源。"""
    return type(value) is str and len(value) == length and all(c in "0123456789abcdef" for c in value)

def _same_native(actual, expected) -> bool:
    """配置/来源快照比较同时核类型，避免True==1.0或Tensor比较绕过原合同。"""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(_same_native(actual[k], expected[k]) for k in expected)
    if type(expected) in (tuple, list):
        return len(actual) == len(expected) and all(_same_native(a, b) for a, b in zip(actual, expected))
    return actual == expected

def _provenance(value: dict) -> dict:
    """核给定来源合同与实际源码字节，synthetic 未知工件必须为 None。

    只读取小型自有源码；真实轨迹/划分/encoder 来源由缓存编排保证。
    """
    if type(value) is not dict:
        raise TypeError("provenance must be an ordinary dict")
    _plain(value)
    required = {"evidence_role", "split", "paper_sha256", "upstream_commit", "source_files_sha256",
                "encoder_checkpoint_sha256", "preprocess_sha256", "feature_manifest_sha256", "protocol_sha256"}
    if not required <= value.keys():
        raise ValueError("provenance is missing fixed source fields")
    if any(type(value[n]) is not str or not value[n].strip() for n in ("evidence_role", "split")):
        raise ValueError("evidence role and split must be explicit")
    if (value["paper_sha256"] != "d851778ecf4a1f160da9c681a831b3bc9333dae2b60eaeefa6090c0af44aaff1"
        or value["upstream_commit"] != "204698b45b3712590f06245fbfba32d3be539812"):
        raise ValueError("fixed paper/upstream provenance mismatch")
    unknown = ("encoder_checkpoint_sha256", "preprocess_sha256", "feature_manifest_sha256", "protocol_sha256")
    if value["evidence_role"] == "synthetic_development_only":
        if value["split"] != "development_smoke" or any(value[n] is not None for n in unknown):
            raise ValueError("synthetic development must not claim real feature provenance")
    elif value["split"] != "train" or any(not _sha(value[n]) for n in unknown):
        raise ValueError("real prepared features require explicit train split and artifact hashes")
    sources = value["source_files_sha256"]
    names = {f"src/po_jepa/{n}.py" for n in ("actions", "dynamics", "geometry", "losses", "model", "spectral")}
    names.add("src/po_jepa/runtime/training.py")
    if type(sources) is not dict or not names <= sources.keys():
        raise ValueError("source_files_sha256 must record all participating own sources")
    root = Path(__file__).resolve().parents[3]
    for name, expected in sources.items():
        path = PurePosixPath(name)
        if (not _sha(expected) or path.is_absolute() or ".." in path.parts or "\\" in name
            or not name.startswith(("src/po_jepa/", "src/po_jepa/runtime/")) or path.suffix != ".py"):
            raise ValueError("source SHA keys must be portable own-source relative paths")
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"actual source SHA mismatch: {name}")
    return value

def _loss_config(config: LossConfig, condition: str) -> dict:
    """消费合法Real为原生float64快照；训练full双正、批准对照双零，不新增消融。"""
    if not isinstance(config, LossConfig):
        raise TypeError("loss_config must be LossConfig")
    config.__post_init__()
    native = {name: value if isinstance(value, str) else float(value) for name, value in asdict(config).items()}
    for name, scalar in native.items():
        if isinstance(scalar, str):
            continue
        if not math.isfinite(scalar) or (getattr(config, name) > 0 and scalar == 0):
            raise ValueError(f"{name} is not representable at consumption precision")
    weights = (native["time_weight"], native["space_weight"])
    if (condition == "full" and all(w > 0 for w in weights)) or (condition == "no_consistency" and weights == (0., 0.)):
        return native
    raise ValueError("condition must be full with two positive weights or no_consistency with two zero weights")

def _parameters(model: POJEPA) -> list:
    """锁定当前完整结构的27参数与顺序；未训练分支不能靠requires_grad过滤消失。"""
    if not isinstance(model, POJEPA):
        raise TypeError("model must be POJEPA")
    model._validate_components()
    modules = ("dynamics.condition_trunk.0", "dynamics.gamma_head",
        "dynamics.omega_head", "dynamics.b_head", "dynamics.q_proj", "dynamics.k_proj", "dynamics.v_proj",
        "dynamics.spectral_bias_net.0", "dynamics.spectral_bias_net.2", "dynamics.spatial_bias_net.0",
        "dynamics.spatial_bias_net.2", "dynamics.action_residual_net.0", "dynamics.action_residual_net.2", "p_in", "p_out")
    expected = [name+"."+field for name in modules for field in (("weight",) if name.endswith(("q_proj", "k_proj", "v_proj")) else ("weight", "bias"))]
    parameters = list(model.named_parameters())
    if [n for n, _ in parameters] != expected or len({id(p) for _, p in parameters}) != len(expected):
        raise ValueError("complete predictor parameter names/order must remain unchanged and unique")
    device = parameters[0][1].device
    for name, parameter in parameters:
        if (not parameter.requires_grad or parameter.dtype != torch.float32 or parameter.device != device
            or parameter.layout != torch.strided or parameter.is_meta or torch.is_inference(parameter)
            or not torch.isfinite(parameter).all()):
            raise ValueError(f"{name} must be an unfrozen finite model float32 parameter")
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("model must use CPU or available CUDA")
    for buffer in model.buffers():
        if buffer.device != device or buffer.requires_grad or not torch.isfinite(buffer).all():
            raise ValueError("fixed buffers must stay finite, fixed and on model device")
    model.spectrum._working_buffers()
    return parameters

def make_optimizer(model: POJEPA) -> torch.optim.AdamW:
    """全部完整预测参数使用固定 AdamW，不包含 encoder、buffer 或 raw 特征。"""
    parameters = _parameters(model)
    return torch.optim.AdamW([p for _, p in parameters], lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0)

def _optimizer(model: POJEPA, optimizer: torch.optim.AdamW) -> list:
    """更新/保存恢复共用覆盖核验：固定标准选项、同一对象顺序和有限已有状态。"""
    parameters = _parameters(model)
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError("optimizer must be the single-group AdamW from make_optimizer")
    actual = optimizer.param_groups[0]["params"]
    if [id(p) for p in actual] != [id(p) for _, p in parameters]:
        raise ValueError("optimizer must cover all predictor parameters once in original order")
    fixed = dict(lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False,
                 maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)
    if any(optimizer.defaults.get(k) != v or optimizer.param_groups[0].get(k) != v for k, v in fixed.items()):
        raise ValueError("AdamW configuration differs from fixed first-round defaults")
    _optimizer_finite(optimizer)
    return parameters

def _optimizer_finite(optimizer: torch.optim.AdamW) -> None:
    """有限参数/grad不足以保证AdamW二阶矩有限；每次step后显式核查实际state。"""
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor) and not torch.isfinite(value).all():
                raise ValueError("AdamW state contains nonfinite values; stop without rollback")

def _batch(batch: TrainingBatch, model: POJEPA) -> None:
    """在zero_grad/Phi前核冻结raw、精确监督查询、端点与固定来源，绝不静默detach。"""
    if not isinstance(batch, TrainingBatch):
        raise TypeError("batch must be TrainingBatch")
    if not isinstance(batch.transfer, SpatialTransfer) or not isinstance(batch.controls, ActionSegments):
        raise TypeError("batch requires SpatialTransfer and ActionSegments")
    batch.transfer.__post_init__()
    device = model.p_in.weight.device
    for name in ("raw_z", "raw_target"):
        raw = getattr(batch, name)
        _temporal_raw_contract(raw, name)
        if raw.requires_grad or raw.grad_fn is not None:
            raise ValueError("prepared raw features must be frozen without upstream graph")
        if raw.device != device or raw.shape != batch.raw_z.shape:
            raise ValueError("raw input/target must share exact shape and model device")
    b, n, e = batch.raw_z.shape
    if n != batch.transfer.source.height*batch.transfer.source.width or e != model.encoder_dim:
        raise ValueError("raw N/E must match source grid/model")
    for name in ("duration", "tau", "sigma"):
        _temporal_time_contract(getattr(batch, name), name, b)
    if not torch.equal(batch.duration, batch.tau+batch.sigma):
        raise ValueError("supervised duration must equal original float64 tau+sigma")
    values = batch.controls.u_values
    if (not isinstance(values, Tensor) or values.device != device or values.is_nested
        or values.layout != torch.strided or torch.is_inference(values)):
        raise ValueError("controls must use ordinary model-device values")
    validate_action_segments(batch.controls, query_duration=batch.duration)
    batch.controls.control_spec.__post_init__()
    split_action_segments(batch.controls, tau=batch.tau, sigma=batch.sigma)
    for name in ("observed_time_seconds", "target_time_seconds"):
        value = getattr(batch, name)
        if (not isinstance(value, Tensor) or value.is_nested or value.layout != torch.strided
            or value.dtype != torch.float64 or value.device.type != "cpu" or value.shape != (b,)
            or value.requires_grad or torch.is_inference(value) or not torch.isfinite(value).all()):
            raise ValueError("observed/target physical seconds must be finite CPU float64[B] without graph")
    endpoint = batch.observed_time_seconds+batch.duration*batch.controls.control_spec.control_period_seconds
    if (not torch.equal(batch.observed_time_seconds, batch.controls.source_origin_seconds)
        or not torch.equal(batch.target_time_seconds, endpoint)
        or torch.any(batch.target_time_seconds <= batch.observed_time_seconds)):
        raise ValueError("prepared endpoint pairing is not exactly representable/consistent")
    if (type(batch.sample_ids) is not tuple or len(batch.sample_ids) != b
        or any(type(s) is not str or not s.strip() for s in batch.sample_ids)):
        raise ValueError("batch needs one nonempty sample ID per row")
    _provenance(batch.provenance)

def consistency_warmup_scale(config: LossConfig, completed_steps: int) -> float:
    """v4 式17 的 w(s)=min(1,s/W)，s 为本次更新前已完成的步数；W=0 时恒为 1。"""
    warmup = config.consistency_warmup_updates
    return 1.0 if warmup == 0 else min(1.0, completed_steps/warmup)

def train_step(model: POJEPA, optimizer: torch.optim.AdamW, batch: TrainingBatch,
               *, loss_config: LossConfig, completed_steps: int,
               condition: str = "full") -> TrainingStepResult:
    """式14—17/算法1：同模型五次完整 Phi，一次联合 backward 与 AdamW 更新。

    调用者先 train() 并启用梯度；核 batch、双正/双零条件及参数覆盖后
    zero_grad(set_to_none=True)。JEPA 复用时间长支 direct，标签仅入停止
    标签梯度的距离；时空两侧/中间反馈保留图，无跨步缓存。
    loss 或必需梯度非有限/缺失则不 step；step 后参数/AdamW 状态非有限
    立即失败，保留异常状态，不回滚或重试。成功才+1并返回六项 Python 日志。
    不切模式、设种子或训练 encoder；合法零梯度不要求参数必变。
    """
    _count(completed_steps, "completed_steps")
    _loss_config(loss_config, condition)
    parameters = _optimizer(model, optimizer)
    if not all(part.training for part in model.modules()) or not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("caller must explicitly model.train() with gradient mode enabled outside inference")
    _batch(batch, model)
    optimizer.zero_grad(set_to_none=True)
    temporal = temporal_predictions(model, batch.raw_z, batch.controls, batch.tau, batch.sigma, batch.transfer.source)
    jepa = jepa_distance(temporal.direct, batch.raw_target, config=loss_config)
    time = temporal_consistency_loss(temporal, config=loss_config)
    spatial = spatial_predictions(model, batch.raw_z, batch.controls, batch.duration, batch.transfer)
    space = spatial_consistency_loss(spatial, config=loss_config)
    total = weighted_total_loss(jepa, time, space, config=loss_config,
                                consistency_scale=consistency_warmup_scale(loss_config, completed_steps))
    losses = dict(jepa=jepa.mean, time=time.mean, space=space.mean, weighted_time=total.weighted_time,
                  weighted_space=total.weighted_space, total=total.total)
    if any(not torch.isfinite(value) for value in losses.values()):
        raise ValueError("all six training losses must be finite before backward")
    total.total.backward()
    grad_max_abs = 0.
    for name, parameter in parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f"required predictor gradient missing or nonfinite: {name}")
        grad_max_abs = max(grad_max_abs, parameter.grad.detach().abs().max().item())
    grad_norm = clip_gradients([p for _, p in parameters], loss_config.grad_clip_norm)
    scheduled_step(optimizer, loss_config, completed_steps)
    if any(not torch.isfinite(p).all() for _, p in parameters):
        raise ValueError("updated parameter is nonfinite; stop without rollback")
    _optimizer_finite(optimizer)
    return TrainingStepResult(completed_steps+1, {k: value.detach().item() for k, value in losses.items()}, grad_max_abs, grad_norm)

BASE_LR = 1e-3

def scheduled_lr(config: LossConfig, completed_steps: int) -> float:
    """v5：第 completed_steps 次更新（从 0 计）使用的学习率；lr_decay_updates=0 时恒为 1e-3。"""
    total = config.lr_decay_updates
    if total == 0:
        return BASE_LR
    progress = min(completed_steps, total)/total
    return config.lr_min+(BASE_LR-config.lr_min)*.5*(1+math.cos(math.pi*progress))

def scheduled_step(optimizer: torch.optim.AdamW, config: LossConfig, completed_steps: int) -> None:
    """v5：只在这一次 step 内把 lr 设为调度值，之后恢复登记的 1e-3。

    AdamW 的 lr 不进入动量状态，所以调度是"步数的函数"、无额外状态：checkpoint、续训和
    固定超参核验都不受影响，续训从 completed_steps 自动接上同一条曲线。
    """
    group = optimizer.param_groups[0]
    registered = group["lr"]
    group["lr"] = scheduled_lr(config, completed_steps)
    try:
        optimizer.step()
    finally:
        group["lr"] = registered

def clip_gradients(parameters: list, max_norm: float) -> float:
    """v4：返回裁剪前的全局梯度 L2 范数；max_norm>0 时按该阈值整体缩放梯度，0 时只计算不裁剪。"""
    if max_norm > 0:
        return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True))
    return math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in parameters))

def consistency_gradient_norms(model: POJEPA, batch: TrainingBatch, *, loss_config: LossConfig) -> dict:
    """v4 式 lambda 标定：同一批次、与 train_step 相同的五次 Phi，分别求三项对全部参数的梯度范数。

    不更新参数、不写 optimizer；仅用于训练前在初始化状态下标定 lambda_t/lambda_s。
    """
    parameters = [p for _, p in _parameters(model)]
    _batch(batch, model)
    temporal = temporal_predictions(model, batch.raw_z, batch.controls, batch.tau, batch.sigma, batch.transfer.source)
    jepa = jepa_distance(temporal.direct, batch.raw_target, config=loss_config).mean
    time = temporal_consistency_loss(temporal, config=loss_config).mean
    spatial = spatial_predictions(model, batch.raw_z, batch.controls, batch.duration, batch.transfer)
    space = spatial_consistency_loss(spatial, config=loss_config).mean
    result = dict(jepa=jepa.item(), time=time.item(), space=space.item())
    for name, loss in (("jepa", jepa), ("time", time), ("space", space)):
        grads = torch.autograd.grad(loss, parameters, retain_graph=name != "space", allow_unused=True)
        result["grad_"+name] = math.sqrt(sum(float(g.double().square().sum()) for g in grads if g is not None))
    return result

def _native_config(value):
    """配置持久化为原生快照；合法Real按实际消费float精度转换，不pickle配置类。"""
    if type(value) is dict:
        return {k: _native_config(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_native_config(v) for v in value)
    if isinstance(value, Real) and not isinstance(value, bool) and type(value) is not int:
        converted = float(value)
        if not math.isfinite(converted) or (value > 0 and converted == 0):
            raise ValueError("configuration value is not representable at consumption precision")
        return converted
    return _plain(value)

def _model_config(model: POJEPA) -> dict:
    """保存当前projection/graph/dynamics/integrator原生配置，固定buffer另逐值核查。"""
    from po_jepa.model import ProjectionConfig
    projection = ProjectionConfig(model.encoder_dim, model.dynamic_dim, readout=model.readout)
    for config in (model.spectrum.config, model.dynamics.config, model.integrator):
        config.__post_init__()
    return _native_config(dict(projection=asdict(projection), graph=asdict(model.spectrum.config),
        dynamics=asdict(model.dynamics.config), integrator=asdict(model.integrator)))

def _optimizer_config(optimizer: torch.optim.AdamW) -> dict:
    """记录实际defaults和组配置；真实参数顺序另存名称，不以数量替代归属。"""
    return _native_config(dict(defaults=optimizer.defaults,
        param_groups=[{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups]))

def _cpu_tree(value):
    """只在显式checkpoint保存时取无图CPU快照，不给每个训练步复制整套状态。"""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if type(value) is dict:
        return {k: _cpu_tree(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return type(value)(_cpu_tree(v) for v in value)
    return _plain(value)

def _byte_state(state: Tensor, name: str) -> None:
    """RNG状态必须普通CPU ByteTensor，实际长度/内容由隔离generator验证。"""
    if (not isinstance(state, Tensor) or state.is_nested or state.layout != torch.strided
        or state.device.type != "cpu" or state.dtype != torch.uint8 or state.ndim != 1
        or state.numel() == 0 or state.requires_grad or torch.is_inference(state)):
        raise ValueError(f"{name} must be an ordinary CPU ByteTensor vector")

def _sampling(value: dict) -> dict:
    """记录显式 prepared-batch 索引，独立 generator 由调用者按返回状态恢复。"""
    if type(value) is not dict or set(value) != {"kind", "next_batch_index", "query_generator_state"}:
        raise ValueError("sampling_state requires exactly explicit batch fields")
    if value["kind"] != "explicit_prepared_batches":
        raise ValueError("hidden data-loader/sampler state is outside E06")
    _count(value["next_batch_index"], "next_batch_index")
    state = value["query_generator_state"]
    if state is not None:
        _byte_state(state, "query_generator_state")
        torch.Generator(device="cpu").set_state(state)
    return dict(kind=value["kind"], next_batch_index=value["next_batch_index"],
                query_generator_state=None if state is None else state.clone())

def _runtime(device: torch.device) -> dict:
    """记录当前版本与实际单设备恢复环境，不保证跨后端数值一致。"""
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=str(np.__version__), device=str(device))

def _rng_state(device: torch.device) -> dict:
    """保存Python/NumPy legacy/Torch CPU与实际单CUDA状态，CPU任务不初始化GPU。"""
    algorithm, keys, position, has_gauss, cached = np.random.get_state(legacy=True)
    cuda = None
    if device.type == "cuda":
        cuda = dict(device_index=device.index, state=torch.cuda.get_rng_state(device).cpu())
    return dict(python=random.getstate(),
        numpy=dict(algorithm=str(algorithm), keys=[int(n) for n in keys], position=int(position),
                   has_gauss=int(has_gauss), cached_gaussian=float(cached)),
        torch_cpu=torch.get_rng_state(), cuda=cuda)

def _check_rng(rng: dict, device: torch.device) -> tuple:
    """用局部RNG对象预检，不改变全局状态；返回重建的NumPy legacy五元组。"""
    if type(rng) is not dict or set(rng) != {"python", "numpy", "torch_cpu", "cuda"}:
        raise ValueError("invalid RNG fields")
    python = rng["python"]
    if (type(python) is not tuple or len(python) != 3 or type(python[0]) is not int or python[0] != 3
        or type(python[1]) is not tuple or len(python[1]) != 625
        or any(type(n) is not int or not 0 <= n < 2**32 for n in python[1][:-1])
        or type(python[1][-1]) is not int or not 0 <= python[1][-1] <= 624
        or (python[2] is not None and (type(python[2]) is not float or not math.isfinite(python[2])))):
        raise ValueError("invalid Python native RNG state/cache")
    random.Random(0).setstate(python)
    numpy = rng["numpy"]
    if type(numpy) is not dict or set(numpy) != {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}:
        raise ValueError("invalid NumPy legacy RNG fields")
    if (numpy["algorithm"] != "MT19937" or type(numpy["keys"]) is not list or len(numpy["keys"]) != 624
        or any(type(n) is not int or not 0 <= n < 2**32 for n in numpy["keys"])
        or type(numpy["position"]) is not int or not 0 <= numpy["position"] <= 624
        or type(numpy["has_gauss"]) is not int or numpy["has_gauss"] not in (0, 1)
        or type(numpy["cached_gaussian"]) is not float or not math.isfinite(numpy["cached_gaussian"])):
        raise ValueError("invalid NumPy legacy RNG values/uint32 keys")
    legacy = (numpy["algorithm"], np.asarray(numpy["keys"], dtype=np.uint32), numpy["position"],
              numpy["has_gauss"], numpy["cached_gaussian"])
    np.random.RandomState(0).set_state(legacy)
    _byte_state(rng["torch_cpu"], "torch_cpu")
    torch.Generator(device="cpu").set_state(rng["torch_cpu"])
    cuda = rng["cuda"]
    if device.type == "cpu":
        if cuda is not None:
            raise ValueError("CPU checkpoint must not claim CUDA RNG state")
    else:
        if type(cuda) is not dict or set(cuda) != {"device_index", "state"} or type(cuda["device_index"]) is not int or cuda["device_index"] != device.index:
            raise ValueError("CUDA RNG must name the actual single training device")
        _byte_state(cuda["state"], "cuda RNG")
        torch.Generator(device=device).set_state(cuda["state"])
    return legacy

def _check_model_state(state: dict, model: POJEPA) -> None:
    """加载前严格核全部键/形状/dtype/有限值，固定Q/带宽须与显式构建者逐值一致。"""
    current, buffers = model.state_dict(), dict(model.named_buffers())
    if type(state) is not dict or state.keys() != current.keys():
        raise ValueError("model state keys mismatch")
    for name, value in state.items():
        expected = current[name]
        if (not isinstance(value, Tensor) or value.is_nested or value.layout != torch.strided
            or value.device.type != "cpu" or value.requires_grad or torch.is_inference(value)
            or value.shape != expected.shape or value.dtype != expected.dtype or not torch.isfinite(value).all()):
            raise ValueError(f"model state shape/dtype/finite mismatch: {name}")
        if name in buffers and not torch.equal(value, buffers[name].detach().cpu()):
            raise ValueError(f"fixed buffer mismatch: {name}")

def _check_optimizer_state(state: dict, optimizer: torch.optim.AdamW, parameters: list) -> None:
    """预检AdamW实际索引归属和moments；非capturable step仍按原生CPU规则恢复。"""
    if type(state) is not dict or set(state) != {"state", "param_groups"}:
        raise ValueError("invalid AdamW state structure")
    current = optimizer.state_dict()
    if not _same_native(state["param_groups"], current["param_groups"]) or type(state["state"]) is not dict:
        raise ValueError("AdamW parameter IDs/order/group configuration mismatch")
    indices = current["param_groups"][0]["params"]
    states = state["state"]
    if states and (set(states) != set(indices) or any(type(k) is not int for k in states)):
        raise ValueError("AdamW state must cover the actual parameter indices")
    for index, (_, parameter) in zip(indices, parameters):
        if not states:
            break
        item = states[index]
        if type(item) is not dict or set(item) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("AdamW state fields mismatch")
        for name, value in item.items():
            if (not isinstance(value, Tensor) or value.is_nested or value.layout != torch.strided
                or value.device.type != "cpu" or value.requires_grad or torch.is_inference(value)
                or not torch.isfinite(value).all()):
                raise ValueError("AdamW state must be finite ordinary CPU tensors in checkpoint")
            if name == "step":
                if value.shape != () or value.dtype not in (torch.float32, torch.float64) or value.item() < 0 or not value.item().is_integer():
                    raise ValueError("AdamW step must be a finite nonnegative integer-valued scalar")
            elif value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError("AdamW moment shape/dtype mismatch")
            if name == "exp_avg_sq" and torch.any(value < 0):
                raise ValueError("AdamW second moment must be nonnegative")

def save_training_checkpoint(path: Path, *, model: POJEPA, optimizer: torch.optim.AdamW,
    loss_config: LossConfig, condition: str, completed_steps: int, sampling_state: dict, provenance: dict) -> str:
    """schema1 原生保存：指定路径，同目录独占临时写完后发布，拒覆盖。

    保存参数/buffer、配置快照、AdamW 归属/状态、条件/来源、成功进度和
    Python/NumPy/CPU/实际单 CUDA RNG。NumPy uint32 键编码整数列表，
    未启用 scheduler/scaler 为 None；不 pickle 配置或模型。返回实际 SHA。
    失败仅清自己的临时文件，不扫描 latest、删除旧目标或回滚训练状态。
    """
    if not isinstance(path, Path):
        raise TypeError("path must be explicit Path")
    if path.exists():
        raise FileExistsError(path)
    _count(completed_steps, "completed_steps")
    parameters = _optimizer(model, optimizer)
    device = parameters[0][1].device
    payload = dict(schema_version=1, completed_steps=completed_steps, condition=condition,
        model_state=_cpu_tree(dict(model.state_dict())), optimizer_state=_cpu_tree(optimizer.state_dict()),
        optimizer_param_names=[[n for n, _ in parameters]], model_config=_model_config(model),
        loss_config=_loss_config(loss_config, condition), optimizer_config=_optimizer_config(optimizer),
        provenance=_plain(_provenance(provenance)), rng_state=_rng_state(device), sampling_state=_sampling(sampling_state),
        model_training=model.training, scheduler_state=None, scaler_state=None, runtime=_runtime(device))
    _check_model_state(payload["model_state"], model)
    _check_optimizer_state(payload["optimizer_state"], optimizer, parameters)
    _check_rng(payload["rng_state"], device)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix="."+path.name+".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # 同目录硬链接只在目标不存在时发布完整字节；竞争目标存在则原生失败，不覆盖。
        os.link(temporary, path)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def load_training_checkpoint(path: Path, *, expected_sha256: str, model: POJEPA,
    optimizer: torch.optim.AdamW, loss_config: LossConfig, condition: str,
    expected_provenance: dict) -> CheckpointProgress:
    """指定 SHA 的 schema1 文件，预检合同后恢复模型、优化器与 RNG。

    摘要绑定同份读入字节；map_location=cpu、weights_only=True，不放宽
    pickle。先核配置/来源、键/shape/dtype/buffer、AdamW 名称归属/状态、
    步数/模式/RNG/采样，再 strict 加载并保持参数身份。最后恢复模式和
    Python/NumPy/CPU/实际单 CUDA RNG，返回进度，不重设 seed 或自动采样。
    预检 RNG 局部隔离；设备异常不承诺事务回滚，同环境也不保证跨硬件逐位一致。
    """
    if not isinstance(path, Path):
        raise TypeError("path must be explicit Path")
    if not _sha(expected_sha256):
        raise ValueError("expected_sha256 must be explicit lowercase SHA256")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    parameters = _optimizer(model, optimizer)
    device = parameters[0][1].device
    required = {"schema_version", "completed_steps", "condition", "model_state", "optimizer_state", "optimizer_param_names",
        "model_config", "loss_config", "optimizer_config", "provenance", "rng_state", "sampling_state", "model_training",
        "scheduler_state", "scaler_state", "runtime"}
    if type(payload) is not dict or not required <= payload.keys() or type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema/fields")
    _count(payload["completed_steps"], "completed_steps")
    if (not _same_native(payload["condition"], condition) or not _same_native(payload["loss_config"], _loss_config(loss_config, condition))
        or not _same_native(payload["model_config"], _model_config(model)) or not _same_native(payload["optimizer_config"], _optimizer_config(optimizer))
        or not _same_native(payload["provenance"], _provenance(expected_provenance))
        or not _same_native(payload["optimizer_param_names"], [[n for n, _ in parameters]]) or not _same_native(payload["runtime"], _runtime(device))
        or type(payload["model_training"]) is not bool or payload["scheduler_state"] is not None or payload["scaler_state"] is not None):
        raise ValueError("checkpoint condition/configuration/source/order/runtime/mode mismatch")
    _plain(payload["provenance"])
    _check_model_state(payload["model_state"], model)
    _check_optimizer_state(payload["optimizer_state"], optimizer, parameters)
    sampling = _sampling(payload["sampling_state"])
    legacy = _check_rng(payload["rng_state"], device)
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    model.train(payload["model_training"])
    random.setstate(payload["rng_state"]["python"])
    np.random.set_state(legacy)
    torch.set_rng_state(payload["rng_state"]["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(payload["rng_state"]["cuda"]["state"], device=device)
    return CheckpointProgress(payload["completed_steps"], sampling)

