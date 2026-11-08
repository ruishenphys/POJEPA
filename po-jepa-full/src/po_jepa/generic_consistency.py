"""08 §3：不限模型类型的式15/16一致性预测（transformer_consistency 用）。

两个函数逐行照抄 losses.temporal_predictions / losses.spatial_predictions，只把
isinstance(model, POJEPA) 换成 _require_predictor：model 是 nn.Module，forward 恰好接受
4 个位置参数 (raw, controls, duration, grid)，并有正整数 encoder_dim。返回 losses 里原有的
TemporalPredictions / SpatialPredictions，交给 losses 里原有的 temporal_consistency_loss、
spatial_consistency_loss、weighted_total_loss。本模块不定义损失，不改 losses.py。
"""

import inspect

import torch
from torch import Tensor, nn

from .actions import ActionSegments, split_action_segments, concatenate_action_segments
from .geometry import GridSpec, SpatialTransfer, apply_spatial_transfer
from .losses import (SpatialPredictions, TemporalPredictions, _temporal_raw_contract,
                     _temporal_time_contract, input_scale)


def _require_predictor(model) -> None:
    """代替 isinstance(model, POJEPA)：nn.Module、forward(raw, controls, duration, grid)、正整数 encoder_dim。"""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    parameters = list(inspect.signature(model.forward).parameters.values())
    if (len(parameters) != 4
            or any(p.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                   or p.default is not inspect.Parameter.empty for p in parameters)):
        raise TypeError("model.forward must take exactly four positional arguments (raw, controls, duration, grid)")
    encoder_dim = getattr(model, "encoder_dim", None)
    if type(encoder_dim) is not int or encoder_dim <= 0:
        raise TypeError("model must expose a positive integer encoder_dim")


def temporal_predictions(model: nn.Module, raw_z: Tensor, controls: ActionSegments,
                         tau: Tensor, sigma: Tensor, grid: GridSpec) -> TemporalPredictions:
    """式15，同 losses.temporal_predictions：direct=Phi(Z,u1⊕u2,tau+sigma)，intermediate=Phi(Z,u1,tau)，
    composed=Phi(intermediate,u2,sigma)；三次完整前向，中间反馈不 LN、不 detach。"""
    _require_predictor(model)
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


def spatial_predictions(model: nn.Module, raw_z: Tensor, controls: ActionSegments,
                        duration: Tensor, transfer: SpatialTransfer) -> SpatialPredictions:
    """式16，同 losses.spatial_predictions：restricted=R Phi(Z, source)，coarse=Phi(RZ, target)；两次完整前向。"""
    _require_predictor(model)
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
