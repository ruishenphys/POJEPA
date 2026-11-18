"""08 §6：po_jepa_rollout_sup 的有真值滚动损失（二等分两段；S28d 起多段切分变体下是 2^k 段）。

L_roll = mean_B mean_(N,E) |LN(Φ_Π(Z; u)) − LN(Z_target(T))|

- 二等分（默认变体，S27a 定义，逐位不变）：Φ_Π(Z; u) = Φ(Φ(Z, u1, τ), u2, σ)，就是 losses.temporal_predictions 算好的
  composed，T = τ+σ。
- 多段切分（S28d，Jerry 2026-11-17 决策卡）：和完整模型的时间一致性用同一种切分、同一串 k——k_b = losses.dyadic_levels
  (T, sample_ids, step)（同一步、同一批和 po_jepa_full 抽到的 k 逐行相同），Φ_Π 是 losses.partition_predictions 的
  composed：2^k 段、每段长 h = T/2^k、第 i 段动作是原控制 [i·h, (i+1)·h) 的精确交集、每段输入是上一段的 raw 输出
  （引理 5.4 的 Φ_{Π_k}）。k = 1 时和二等分逐位相同。

两种切分下：真值目标只有终点 Z_target(T)，就是 JEPA 项的同一个 batch.raw_target（训练批里 duration 逐位等于 τ+σ，
training._batch 核过）；中间各段的终点不监督（训练批里没有中间时刻的真值帧，和 S27a 相同）。距离和 JEPA 项完全相同
（losses.jepa_distance：两侧沿 E 做无 affine LN，eps 1e-5，p=1 的 MAE，标签停梯度），不除 (τσ)²、不除 c(Π_k)²、
不按 k 加权，也不用一致性度量 consistency_metric。中间各段不 detach，梯度经全部段传回（和时间一致性项相同）。

返回 float64 的 ReducedLoss，放在 weighted_total_loss 的时间那一格（time），权重是 LossConfig.time_weight（即 λ_r）。
本模块不改 losses.py。
"""

import torch
from torch import Tensor

from .actions import ActionSegments
from .geometry import GridSpec
from .losses import (LossConfig, PartitionPredictions, ReducedLoss, TemporalPredictions, dyadic_levels, jepa_distance,
                     partition_predictions, temporal_predictions)


def rollout_predictions(model, raw_z: Tensor, controls: ActionSegments, tau: Tensor, sigma: Tensor, grid: GridSpec,
                        *, config: LossConfig, step: int | None, sample_ids: tuple[str, ...]):
    """S28d：po_jepa_rollout_sup 的滚动支路，按 config.temporal_partition 选切分；返回的 .direct 是 JEPA 项的长支。

    bisection：losses.temporal_predictions（3 次 Φ，和 S27a 逐位相同，step、sample_ids 不用）。
    dyadic：k = losses.dyadic_levels(τ+σ, sample_ids, step)，losses.partition_predictions（1 + Σ_{不同 k} 2^k 次 Φ），
    和 losses.time_consistency 的多段切分分支同一个抽法、同一条组合路径。
    """
    if not isinstance(config, LossConfig):
        raise TypeError("config must be a LossConfig")
    config.__post_init__()
    if config.temporal_partition == "bisection":
        return temporal_predictions(model, raw_z, controls, tau, sigma, grid)
    if step is None:
        raise ValueError("dyadic temporal partition needs the explicit update index step")
    levels = dyadic_levels(tau+sigma, sample_ids, step=step)
    return partition_predictions(model, raw_z, controls, tau, sigma, levels, grid)


def rollout_supervision_loss(predictions: TemporalPredictions | PartitionPredictions, raw_target: Tensor, *,
                             config: LossConfig) -> ReducedLoss:
    """L_roll：composed（两段或 2^k 段滚动）对真值 raw_target 的 JEPA 距离（逐样本 [B] 和均值），转成 float64、保留 composed 的梯度。

    raw_target 的形状、设备、精度、有限性和停梯度都由 jepa_distance 核；这里另核它和 composed 同形（同一批的真值），
    并要求 predictions 的 raw 字段同形（TemporalPredictions 四个，PartitionPredictions 三个）。不读 tau、sigma、duration、
    levels：损失不除任何时长或切分系数。
    """
    if isinstance(predictions, TemporalPredictions):
        names = ("direct", "intermediate", "composed", "difference")
    elif isinstance(predictions, PartitionPredictions):
        names = ("direct", "composed", "difference")
    else:
        raise TypeError("predictions must be TemporalPredictions or PartitionPredictions")
    if not isinstance(raw_target, Tensor):
        raise TypeError("raw_target must be a tensor")
    for name in names:
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
