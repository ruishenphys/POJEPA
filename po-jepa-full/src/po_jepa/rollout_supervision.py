"""08 §6：po_jepa_rollout_sup 的有真值两段滚动损失。

L_roll = mean_B mean_(N,E) |LN(Φ(Φ(Z, u1, τ), u2, σ)) − LN(Z_target(τ+σ))|

composed 就是 losses.temporal_predictions 已经算好的两段滚动 Φ(Φ(Z,u1,τ),u2,σ)；Z_target(τ+σ) 是 JEPA 项用的同一个
batch.raw_target（训练批里 duration 逐位等于 τ+σ，training._batch 核过）。距离和 JEPA 项完全相同（losses.jepa_distance：
两侧沿 E 做无 affine LN，eps 1e-5，p=1 的 MAE，标签停梯度），不除 (τσ)²、不用一致性度量 consistency_metric。
中间预测 intermediate 不 detach，梯度经两段传回（和时间一致性项相同）。

返回 float64 的 ReducedLoss，放在 weighted_total_loss 的时间那一格（time），权重是 LossConfig.time_weight（即 λ_r）。
本模块不改 losses.py。
"""

import torch
from torch import Tensor

from .losses import LossConfig, ReducedLoss, TemporalPredictions, jepa_distance


def rollout_supervision_loss(predictions: TemporalPredictions, raw_target: Tensor, *, config: LossConfig) -> ReducedLoss:
    """L_roll：composed 对真值 raw_target 的 JEPA 距离（逐样本 [B] 和均值），转成 float64、保留 composed 的梯度。

    raw_target 的形状、设备、精度、有限性和停梯度都由 jepa_distance 核；这里另核它和 direct 同形（同一批的真值），
    并要求 TemporalPredictions 的四个 raw 字段同形。
    """
    if not isinstance(predictions, TemporalPredictions):
        raise TypeError("predictions must be TemporalPredictions")
    if not isinstance(raw_target, Tensor):
        raise TypeError("raw_target must be a tensor")
    for name in ("direct", "intermediate", "composed", "difference"):
        value = getattr(predictions, name)
        if not isinstance(value, Tensor) or value.shape != predictions.direct.shape:
            raise ValueError("temporal raw fields must share the same shape")
    if raw_target.shape != predictions.composed.shape:
        raise ValueError("rollout target must have exactly the composed prediction shape")
    distance = jepa_distance(predictions.composed, raw_target, config=config)
    per_sample = distance.per_sample.to(dtype=torch.float64)
    mean = per_sample.mean()
    if not torch.isfinite(per_sample).all() or not torch.isfinite(mean) or torch.any(per_sample < 0):
        raise ValueError("rollout supervision loss must be finite and nonnegative float64")
    return ReducedLoss(per_sample, mean)
