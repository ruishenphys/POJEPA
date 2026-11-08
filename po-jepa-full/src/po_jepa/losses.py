"""式14端点距离与式15/16时空一致性。

JEPA 距离局部对两侧 E 轴作无 affine LN，再按固定 p=1 求 MAE；raw 表示
保持不变。时空分支共用完整 Phi 与固定 R，保留两侧及中间预测的梯度。
"""

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor
from torch.nn import functional as F

from .actions import ActionSegments, split_action_segments, concatenate_action_segments
from .geometry import GridSpec, SpatialTransfer, apply_spatial_transfer
from .model import POJEPA


@dataclass(frozen=True)
class LossConfig:
    """固定 p=1、LN eps=1e-5；时间/空间 epsilon 分别为周期^4/周期^2，均正。

    权重非负；训练条件另核双正或双零，JEPA 距离不使用一致性参数。
    """

    jepa_exponent: float
    layer_norm_epsilon: float
    time_epsilon: float
    space_epsilon: float
    time_weight: float
    space_weight: float
    # v4：consistency_metric="ln_mean" 时两条路径先做与 JEPA 目标相同的逐 token LN，
    # 差的平方按 N*E 取均值；"raw_sum" 保持一轮 raw 平方和。
    # v5：consistency_metric="input_scale" 时 delta=mean_(N,E)(A-B)^2 / s^2，s^2 为输入 Z_t 的方差（停梯度），
    # 两条路径除以同一个数，放大输出只会增大缺陷；d=sqrt(delta) 是真正的度量。
    consistency_metric: str = "raw_sum"
    # v4：前 W 次更新内一致性权重按 min(1, s/W) 线性预热；0 表示不预热。
    consistency_warmup_updates: int = 0
    # v4：全局梯度范数裁剪阈值（在 optimizer.step 前对全部预测参数做 clip_grad_norm_）；0 表示不裁剪。
    grad_clip_norm: float = 0.0
    # v5：学习率余弦衰减，第 s 次更新用 lr_min+(1e-3-lr_min)(1+cos(pi*min(s,S)/S))/2，S=lr_decay_updates；
    # S=0 表示恒定 1e-3（一轮/v4 行为）。只在 step 时临时设置，optimizer 的登记 lr 与状态不变，续训无需保存调度器。
    lr_min: float = 1e-3
    lr_decay_updates: int = 0
    # v5：前 W 次更新学习率从 1e-3/W 线性升到 1e-3（第 s 次用 1e-3*(s+1)/W），之后再做余弦衰减；0 表示不预热。
    lr_warmup_updates: int = 0

    def __post_init__(self) -> None:
        """拒绝非有限值、伪标量及偏离固定 p/LN 选择的配置。"""
        if self.consistency_metric not in ("raw_sum", "ln_mean", "input_scale"):
            raise ValueError("consistency_metric must be 'raw_sum', 'ln_mean' or 'input_scale'")
        if type(self.consistency_warmup_updates) is not int or self.consistency_warmup_updates < 0:
            raise ValueError("consistency_warmup_updates must be a nonnegative integer")
        if type(self.lr_decay_updates) is not int or self.lr_decay_updates < 0:
            raise ValueError("lr_decay_updates must be a nonnegative integer")
        if (type(self.lr_warmup_updates) is not int or self.lr_warmup_updates < 0
                or (self.lr_decay_updates and self.lr_warmup_updates >= self.lr_decay_updates)):
            raise ValueError("lr_warmup_updates must be a nonnegative integer smaller than lr_decay_updates")
        if isinstance(self.lr_min, bool) or not isinstance(self.lr_min, Real) or not 0 < self.lr_min <= 1e-3:
            raise ValueError("lr_min must be a real number in (0, 1e-3]")
        for name in ("jepa_exponent", "layer_norm_epsilon", "time_epsilon",
                     "space_epsilon", "time_weight", "space_weight", "grad_clip_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real scalar, not bool/tensor/array")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name} must be finite")
            if name.endswith("weight") or name == "grad_clip_norm":
                if value < 0:
                    raise ValueError(f"{name} must be nonnegative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.jepa_exponent != 1.0:
            raise ValueError("jepa_exponent must be the fixed first-round value 1.0")
        if self.layer_norm_epsilon != 1e-5:
            raise ValueError("layer_norm_epsilon must be the fixed first-round value 1e-5")


@dataclass(frozen=True)
class ReducedLoss:
    """逐样本 per_sample [B] 与零维 mean，不隐含归约或 detach。

    JEPA 为 float32，一致性损失为 float64；同设备并保留预测梯度。
    """

    per_sample: Tensor
    mean: Tensor


def jepa_distance(raw_prediction: Tensor, raw_target: Tensor, *, config: LossConfig) -> ReducedLoss:
    """式14：双侧局部 LN 后的逐样本 MAE。

    P/Y 为同形同设备有限 float32 [B,N,E]，允许非连续输入，特征无量纲。
    仅 Y.detach()；LN=(z-mu)/sqrt(var+eps)，总体方差沿 E，返回
    mean_(N,E)|LN(P)-LN(Y)| 及 batch 均值。固定 p=1，LN 不写回 raw。
    保持预测及共享源梯度，服从 no_grad，不广播、转精度或设备。
    标签时空配对由调用者保证；形状/设备/非有限计算错误直接抛出。
    """
    if not isinstance(config, LossConfig):
        raise TypeError("config must be a LossConfig")
    config.__post_init__()
    for name, value in (("raw_prediction", raw_prediction), ("raw_target", raw_target)):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.is_nested or value.layout != torch.strided or value.is_meta:
            raise ValueError(f"{name} must be a real dense tensor")
        if torch.is_inference(value):
            raise ValueError(f"{name} must not be an inference tensor")
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must be float32; adapt explicitly with E02 first")
        if value.ndim != 3 or any(size <= 0 for size in value.shape):
            raise ValueError(f"{name} must have shape [positive B, positive N, positive E]")
        if value.device.type not in ("cpu", "cuda") or (value.device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError(f"{name} must use a CPU or available CUDA device")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite values")
    if raw_prediction.shape != raw_target.shape:
        raise ValueError("prediction and target must have exactly the same shape; no broadcasting")
    if raw_prediction.device != raw_target.device:
        raise ValueError("prediction and target must share the same actual device")
    # 配置允许合法Real子类；仅将固定epsilon交为原生算子所需float，不转换raw。
    epsilon = float(config.layer_norm_epsilon)
    prediction = F.layer_norm(raw_prediction, (raw_prediction.shape[-1],), eps=epsilon)
    target = F.layer_norm(raw_target.detach(), (raw_target.shape[-1],), eps=epsilon)
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("JEPA LayerNorm must produce finite values")
    per_sample = torch.abs(prediction-target).mean(dim=(1, 2))
    mean = per_sample.mean()
    if (per_sample.dtype != torch.float32 or mean.dtype != torch.float32
        or not torch.isfinite(per_sample).all() or not torch.isfinite(mean)):
        raise ValueError("JEPA reduction must produce finite float32 values")
    return ReducedLoss(per_sample, mean)


def input_scale(raw_z: Tensor) -> Tensor:
    """v5：逐样本输入尺度 s^2=mean_(N,E)(Z-mean(Z))^2，float64 [B]，停梯度；一致性缺陷与 TCD/SCD 的归一化分母。"""
    scale = raw_z.detach().to(dtype=torch.float64).var(dim=(1, 2), unbiased=False)
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("input scale must be finite and strictly positive")
    return scale


def _consistency_difference(first: Tensor, second: Tensor, difference: Tensor, config: LossConfig,
                            scale: Tensor | None = None) -> Tensor:
    """一致性缺陷的 float64 差：raw_sum 直接用保存的 raw 差；ln_mean 用 LN(first)-LN(second)；
    input_scale 用保存的 raw 差除以输入尺度 s（两条路径共用同一个 s）。

    LN 与 JEPA 目标相同（无 affine、eps=layer_norm_epsilon，沿 E），两侧都保留梯度。
    """
    if config.consistency_metric == "raw_sum":
        return difference.to(dtype=torch.float64)
    if config.consistency_metric == "input_scale":
        if scale is None or scale.shape != difference.shape[:1]:
            raise ValueError("input_scale metric requires a per-sample input scale [B]")
        return difference.to(dtype=torch.float64)/scale.to(device=difference.device).sqrt()[:, None, None]
    eps = float(config.layer_norm_epsilon)
    a = F.layer_norm(first, (first.shape[-1],), eps=eps)
    b = F.layer_norm(second, (second.shape[-1],), eps=eps)
    return (a-b).to(dtype=torch.float64)


def _consistency_reduce(d64: Tensor, config: LossConfig) -> Tensor:
    """raw_sum 沿 (N,E) 求平方和；ln_mean / input_scale 沿 (N,E) 取平方均值，使 lambda 跨网格可比。"""
    if config.consistency_metric == "raw_sum":
        return d64.square().sum(dim=(1, 2))
    return d64.square().mean(dim=(1, 2))


@dataclass(frozen=True)
class TotalLoss:
    """式17：保存三项原归约、两项加权值与总和，保留计算图。

    JEPA 为 float32，时空项和三个合计字段为同设备零维 float64。
    """

    jepa: ReducedLoss
    time: ReducedLoss
    space: ReducedLoss
    weighted_time: Tensor
    weighted_space: Tensor
    total: Tensor


def weighted_total_loss(jepa: ReducedLoss, time: ReducedLoss, space: ReducedLoss,
                        *, config: LossConfig, consistency_scale: float = 1.0) -> TotalLoss:
    """式17：JEPA.mean + lambda_t*time.mean + lambda_s*space.mean。

    各 per_sample 非空一维，mean 零维，均有限非负；三项同设备，JEPA
    为 float32、时空为 float64，允许样本数不同，不重算调用者 mean。
    加权值和 total 为 float64，保持全部梯度。正乘积下溢为 0、乘积或
    总和溢出时报精度不足；显式零权重/零损失合法，输入仍必须有限。
    """
    if not isinstance(config, LossConfig):
        raise TypeError("config must be a LossConfig")
    config.__post_init__()
    device = None
    for name, loss, dtype in (("jepa", jepa, torch.float32), ("time", time, torch.float64),
                              ("space", space, torch.float64)):
        if not isinstance(loss, ReducedLoss):
            raise TypeError(f"{name} must be ReducedLoss")
        for field in ("per_sample", "mean"):
            value = getattr(loss, field)
            if not isinstance(value, Tensor):
                raise TypeError(f"{name}.{field} must be Tensor")
            if value.is_nested or value.layout != torch.strided or value.is_meta or torch.is_inference(value):
                raise ValueError(f"{name}.{field} must be ordinary dense tensor")
            if value.dtype != dtype:
                raise TypeError(f"{name}.{field} has incorrect loss role dtype")
            if (value.device.type not in ("cpu", "cuda")
                or (value.device.type == "cuda" and not torch.cuda.is_available())):
                raise ValueError("losses must use CPU or available CUDA")
            if device is None:
                device = value.device
            if value.device != device:
                raise ValueError("all loss fields must share actual device")
            if not torch.isfinite(value).all() or torch.any(value < 0):
                raise ValueError("loss fields must be finite and nonnegative")
        if loss.per_sample.ndim != 1 or loss.per_sample.numel() == 0 or loss.mean.ndim != 0:
            raise ValueError("per_sample must be nonempty 1D and mean scalar")
    scale = float(consistency_scale)
    if isinstance(consistency_scale, bool) or not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("consistency_scale must be a finite number in [0,1]")
    weighted = []
    for weight, mean in ((config.time_weight, time.mean), (config.space_weight, space.mean)):
        scalar = float(weight)*scale
        if not math.isfinite(scalar) or (weight > 0 and scale > 0 and scalar == 0):
            raise ValueError("positive loss weight is not representable in float64")
        value = scalar*mean
        if not torch.isfinite(value) or (scalar > 0 and mean > 0 and value == 0):
            raise ValueError("weighted positive loss is not representable in float64")
        weighted.append(value)
    total = jepa.mean.to(dtype=torch.float64)+weighted[0]+weighted[1]
    if not torch.isfinite(total):
        raise ValueError("total loss must remain finite float64")
    return TotalLoss(jepa, time, space, weighted[0], weighted[1], total)


@dataclass(frozen=True)
class TemporalPredictions:
    """式15的 direct/intermediate/composed/difference，保留全部预测图。

    四 raw 字段为模型 float32 [B,N,E]，difference=direct-composed；
    tau/sigma 为无梯度 CPU float64 [B] 正周期。中间反馈不 LN 或 detach。
    """

    direct: Tensor
    intermediate: Tensor
    composed: Tensor
    difference: Tensor
    tau: Tensor
    sigma: Tensor
    # v5：输入 Z_t 的逐样本尺度 s^2（float64 [B]，停梯度）；input_scale 度量需要，其他度量可为 None。
    input_scale: Tensor | None = None


def _temporal_raw_contract(value: Tensor, name: str) -> None:
    """核有限稠密 float32 [B,N,E] raw 边界，不转换或修改输入。"""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.is_nested or value.layout != torch.strided or value.is_meta:
        raise ValueError(f"{name} must be a real dense tensor")
    if torch.is_inference(value):
        raise ValueError(f"{name} must not be an inference tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be model float32")
    if value.ndim != 3 or any(size <= 0 for size in value.shape):
        raise ValueError(f"{name} must be [positive B, positive N, positive E]")
    if value.device.type not in ("cpu", "cuda") or (value.device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"{name} must use a CPU or available CUDA device")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")


def _temporal_time_contract(value: Tensor, name: str, batch: int) -> None:
    """核无梯度 CPU float64 [B] 正控制周期，不广播。"""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if (value.is_nested or value.layout != torch.strided or value.device.type != "cpu"
        or value.dtype != torch.float64 or value.shape != (batch,) or value.requires_grad):
        raise ValueError(f"{name} must be CPU float64 [B] without gradient")
    if not torch.isfinite(value).all() or torch.any(value <= 0):
        raise ValueError(f"{name} must be finite and strictly positive")


def temporal_predictions(model: POJEPA, raw_z: Tensor, controls: ActionSegments,
                         tau: Tensor, sigma: Tensor, grid: GridSpec) -> TemporalPredictions:
    """式15：同一路径切分后，显式调用同一完整 Phi 三次。

    raw_z 为模型 float32 [B,N,E]，grid/controls 在预测前已知；tau/sigma
    为无梯度 CPU float64 [B] 正周期，切分保留幅值、来源和精确时长。
    direct=Phi(Z,u1⊕u2,tau+sigma)，intermediate=Phi(Z,u1,tau)，
    composed=Phi(intermediate,u2,sigma)。第三次重建投影、图谱、上下文和动力学，
    仅接 raw 中间预测，不继承隐藏状态、detach、LN 或真实中间观察。
    返回 float32 差与原时长 CPU 副本；欠覆盖和时间精度错误保留。
    """
    if not isinstance(model, POJEPA):
        raise TypeError("model must be a POJEPA")
    if not isinstance(controls, ActionSegments):
        raise TypeError("controls must be ActionSegments")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    _temporal_raw_contract(raw_z, "raw_z")
    grid.__post_init__()
    _temporal_time_contract(tau, "tau", raw_z.shape[0])
    _temporal_time_contract(sigma, "sigma", raw_z.shape[0])
    pair = split_action_segments(controls, tau=tau, sigma=sigma)
    long_controls = concatenate_action_segments(pair.first, pair.second)
    total = tau+sigma
    direct = model(raw_z, long_controls, total, grid)
    intermediate = model(raw_z, pair.first, tau, grid)
    composed = model(intermediate, pair.second, sigma, grid)
    difference = direct-composed
    if not torch.isfinite(difference).all():
        raise ValueError("temporal raw float32 difference must remain finite")
    return TemporalPredictions(direct, intermediate, composed, difference, tau.clone(), sigma.clone(),
                               input_scale(raw_z))


def temporal_consistency_loss(predictions: TemporalPredictions, *, config: LossConfig) -> ReducedLoss:
    """式15：sum_(N,E)(difference64²)/((tau*sigma)²+time_epsilon)，再 mean_B。

    四 raw 字段同形同设备 float32 [B,N,E]；时间为无梯度 CPU float64 [B]
    正周期，epsilon 单位周期^4。结果 [B]/零维均为同设备 float64，保留
    两支与反馈梯度。直接使用保存的 difference，其来源由生产器保证。
    不除元素数、开方或 LN；分母须有限正，epsilon 转换下溢及非有限结果报错。
    """
    if not isinstance(predictions, TemporalPredictions):
        raise TypeError("predictions must be TemporalPredictions")
    if not isinstance(config, LossConfig):
        raise TypeError("config must be a LossConfig")
    config.__post_init__()
    for name in ("direct", "intermediate", "composed", "difference"):
        value = getattr(predictions, name)
        _temporal_raw_contract(value, name)
        if value.shape != predictions.direct.shape or value.device != predictions.direct.device:
            raise ValueError("temporal raw fields must share the same shape and actual device")
    difference = predictions.difference
    _temporal_time_contract(predictions.tau, "tau", difference.shape[0])
    _temporal_time_contract(predictions.sigma, "sigma", difference.shape[0])
    epsilon = float(config.time_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("time_epsilon precision must remain finite and strictly positive in float64")
    d64 = _consistency_difference(predictions.direct, predictions.composed, difference, config, predictions.input_scale)
    tau64 = predictions.tau.to(device=d64.device, dtype=torch.float64)
    sigma64 = predictions.sigma.to(device=d64.device, dtype=torch.float64)
    numerator = _consistency_reduce(d64, config)
    denominator = (tau64*sigma64).square()+epsilon
    if not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all() or torch.any(denominator <= 0):
        raise ValueError("temporal numerator and positive denominator must remain finite float64")
    per_sample = numerator/denominator
    mean = per_sample.mean()
    if not torch.isfinite(per_sample).all() or not torch.isfinite(mean):
        raise ValueError("temporal reduction must remain finite float64")
    return ReducedLoss(per_sample, mean)


@dataclass(frozen=True)
class SpatialPredictions:
    """式16：restricted_prediction=R Phi(Z)、coarse_prediction=Phi(RZ) 及其差。

    三 raw 字段为同设备 float32 [B,N_target,E]，保留两支梯度。duration
    为无梯度 CPU float64 [B] 正周期，target_grid 为同一 R 转移后的真实坐标。
    """

    restricted_prediction: Tensor
    coarse_prediction: Tensor
    difference: Tensor
    duration: Tensor
    target_grid: GridSpec
    # v5：细网格输入 Z_t^h 的逐样本尺度 s^2，粗细两侧共用。
    input_scale: Tensor | None = None


def spatial_predictions(model: POJEPA, raw_z: Tensor, controls: ActionSegments,
                        duration: Tensor, transfer: SpatialTransfer) -> SpatialPredictions:
    """式16：比较同一完整 Phi 的 R Phi(Z) 与 Phi(RZ)。

    raw_z 为模型 float32 [B,N_source,E]，允许非连续输入及源梯度；控制
    预测前已知，duration 为无梯度 CPU float64 [B] 正周期，两支原样共用。
    fine=Phi(Z,source)，restricted=R(fine)，coarse=Phi(RZ,target)；
    粗支重建自身图谱、Pool 和完整动力学，不剪细模态或继承 query。
    R 保留梯度，target 坐标直接沿用 transfer.target，不 LN、detach 或重编码。
    """
    if not isinstance(model, POJEPA):
        raise TypeError("model must be a POJEPA")
    if not isinstance(controls, ActionSegments):
        raise TypeError("controls must be ActionSegments")
    if not isinstance(transfer, SpatialTransfer):
        raise TypeError("transfer must be a SpatialTransfer")
    transfer.__post_init__()
    _temporal_raw_contract(raw_z, "raw_z")
    if (raw_z.shape[1] != transfer.source.height*transfer.source.width
        or raw_z.shape[2] != model.encoder_dim):
        raise ValueError("raw_z N/E must match transfer.source and model.encoder_dim")
    _temporal_time_contract(duration, "duration", raw_z.shape[0])
    fine_prediction = model(raw_z, controls, duration, transfer.source)
    restricted_prediction = apply_spatial_transfer(fine_prediction, transfer)
    _temporal_raw_contract(restricted_prediction, "restricted_prediction")
    coarse_z = apply_spatial_transfer(raw_z, transfer)
    _temporal_raw_contract(coarse_z, "coarse_z")
    coarse_prediction = model(coarse_z, controls, duration, transfer.target)
    difference = restricted_prediction-coarse_prediction
    if not torch.isfinite(difference).all():
        raise ValueError("spatial raw float32 difference must remain finite")
    return SpatialPredictions(restricted_prediction, coarse_prediction, difference,
                              duration.clone(), transfer.target, input_scale(raw_z))


def spatial_consistency_loss(predictions: SpatialPredictions, *, config: LossConfig) -> ReducedLoss:
    """式16：sum_(N_target,E)(difference64²)/(duration²+space_epsilon)，再 mean_B。

    三 raw 字段为同形同设备 float32 [B,N_target,E]，N 匹配 target_grid；
    duration 为无梯度 CPU float64 [B] 正周期，epsilon 为有限正周期²。
    结果 [B]/零维均为同设备 float64，保留两支梯度；直接使用保存 difference。
    不除元素数、开方或 LN；epsilon 下溢、分母非正/溢出和非有限结果报错。
    这是训练平方损失，与评价 SCD 的范数定义不同。
    """
    if not isinstance(predictions, SpatialPredictions):
        raise TypeError("predictions must be SpatialPredictions")
    if not isinstance(config, LossConfig):
        raise TypeError("config must be a LossConfig")
    config.__post_init__()
    if not isinstance(predictions.target_grid, GridSpec):
        raise TypeError("target_grid must be a GridSpec")
    predictions.target_grid.__post_init__()
    target_n = predictions.target_grid.height*predictions.target_grid.width
    for name in ("restricted_prediction", "coarse_prediction", "difference"):
        value = getattr(predictions, name)
        _temporal_raw_contract(value, name)
        if (value.shape != predictions.restricted_prediction.shape
            or value.device != predictions.restricted_prediction.device or value.shape[1] != target_n):
            raise ValueError("spatial raw fields must share shape/device and N must match target_grid")
    difference = predictions.difference
    _temporal_time_contract(predictions.duration, "duration", difference.shape[0])
    epsilon = float(config.space_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("space_epsilon precision must remain finite and strictly positive in float64")
    d64 = _consistency_difference(predictions.restricted_prediction, predictions.coarse_prediction, difference, config,
                                  predictions.input_scale)
    t64 = predictions.duration.to(device=d64.device, dtype=torch.float64)
    numerator = _consistency_reduce(d64, config)
    denominator = t64.square()+epsilon
    if not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all() or torch.any(denominator <= 0):
        raise ValueError("spatial numerator and positive denominator must remain finite float64")
    per_sample = numerator/denominator
    mean = per_sample.mean()
    if not torch.isfinite(per_sample).all() or not torch.isfinite(mean):
        raise ValueError("spatial reduction must remain finite float64")
    return ReducedLoss(per_sample, mean)
