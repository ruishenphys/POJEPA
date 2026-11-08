"""正文 TCD/SCD 与附录 Delta_act 的逐样本评价指标。

缺陷使用 raw Frobenius 范数而非训练平方损失，不加时间 epsilon；
RMS 单独命名。真实窗口/控制/时间来源由评价编排核验。
"""

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor

from .actions import ActionSegments, validate_action_segments
from .geometry import GridSpec
from .losses import (LossConfig, SpatialPredictions, TemporalPredictions,
    _temporal_raw_contract, _temporal_time_contract, jepa_distance)
from .model import POJEPA


@dataclass(frozen=True)
class SampleMetric:
    """逐样本 values [B]、名称与单位，不隐含统计或转换。

    future/Delta 为模型 float32 无量纲；TCD/SCD/RMS 为同设备 float64，
    单位分别周期^-2/周期^-1。保留原样本映射，字段本身不证明来源。
    """

    values: Tensor
    name: str
    unit: str


@dataclass(frozen=True)
class MetricSummary:
    """count、mean、分位水平与分位值，单位继承指标。

    count 为样本数；两个分位 tuple 同长，可不等于 count。仅报告端建立
    CPU float64 副本，原始值/轨迹映射仍须保留；经验统计不证明统一理论界。
    """

    count: int
    mean: float
    quantile_levels: tuple[float, ...]
    quantile_values: tuple[float, ...]


def _eval_model(model: POJEPA) -> None:
    """评价复用原公开Phi，调用者必须先eval；不擅自改模式或恢复状态。"""
    if not isinstance(model, POJEPA):
        raise TypeError("model must be a POJEPA")
    if model.training:
        raise ValueError("evaluation requires caller to set model.eval() first")


@torch.no_grad()
def future_error(model: POJEPA, raw_z: Tensor, controls: ActionSegments,
                 duration: Tensor, grid: GridSpec, raw_target: Tensor, *,
                 loss_config: LossConfig) -> SampleMetric:
    """式14：完整 Phi raw 预测与指定未来标签的逐样本 JEPA 距离。

    起点/标签为同设备有限 float32 [B,N,E]；控制/grid 已知，duration
    为 CPU float64 [B] 非负周期，标签仅用于距离。沿 E 双侧 LN(eps=1e-5)
    后 p=1 MAE，返回 float32 [B]、future_jepa_distance/dimensionless。
    调用者须先 eval；no_grad 不改模式，零时长仍沿原 Phi，不伪造恒等。
    """
    _eval_model(model)
    prediction = model(raw_z, controls, duration, grid)
    distance = jepa_distance(prediction, raw_target, config=loss_config)
    return SampleMetric(distance.per_sample, "future_jepa_distance", "dimensionless")


def _temporal_inputs(predictions: TemporalPredictions) -> tuple[Tensor, Tensor]:
    """核 raw/时间合同并保留保存差值，返回同设备 double 时间乘积。"""
    if not isinstance(predictions, TemporalPredictions):
        raise TypeError("predictions must be TemporalPredictions")
    for name in ("direct", "intermediate", "composed", "difference"):
        value = getattr(predictions, name)
        _temporal_raw_contract(value, name)
        if value.shape != predictions.direct.shape or value.device != predictions.direct.device:
            raise ValueError("temporal raw fields must share shape [B,N,E] and actual device; "
                f"expected shape={tuple(predictions.direct.shape)}, dtype={predictions.direct.dtype}, "
                f"device={predictions.direct.device}; {name} actual shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, device={value.device}")
    difference = predictions.difference
    _temporal_time_contract(predictions.tau, "tau", difference.shape[0])
    _temporal_time_contract(predictions.sigma, "sigma", difference.shape[0])
    tau64 = predictions.tau.to(device=difference.device, dtype=torch.float64)
    sigma64 = predictions.sigma.to(device=difference.device, dtype=torch.float64)
    return difference, tau64*sigma64


def _spatial_inputs(predictions: SpatialPredictions) -> tuple[Tensor, Tensor]:
    """核目标网格、raw 和时间合同，不重算 R 或替换保存差值。"""
    if not isinstance(predictions, SpatialPredictions):
        raise TypeError("predictions must be SpatialPredictions")
    if not isinstance(predictions.target_grid, GridSpec):
        raise TypeError("target_grid must be a GridSpec")
    predictions.target_grid.__post_init__()
    target_n = predictions.target_grid.height*predictions.target_grid.width
    for name in ("restricted_prediction", "coarse_prediction", "difference"):
        value = getattr(predictions, name)
        _temporal_raw_contract(value, name)
        if (value.shape != predictions.restricted_prediction.shape
            or value.device != predictions.restricted_prediction.device or value.shape[1] != target_n):
            expected_shape = (predictions.restricted_prediction.shape[0], target_n,
                              predictions.restricted_prediction.shape[2])
            raise ValueError("spatial raw fields must share shape [B,N_target,E]/device and match target_grid; "
                f"expected shape={expected_shape}, dtype={predictions.restricted_prediction.dtype}, "
                f"device={predictions.restricted_prediction.device}; {name} actual shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, device={value.device}")
    difference = predictions.difference
    _temporal_time_contract(predictions.duration, "duration", difference.shape[0])
    return difference, predictions.duration.to(device=difference.device, dtype=torch.float64)


def _norm_ratio(difference: Tensor, denominator: Tensor, *, rms: bool) -> Tensor:
    """只共用已锁定的Frobenius/RMS double归约，拒绝数值伪零，不加epsilon。"""
    if not torch.isfinite(denominator).all() or torch.any(denominator <= 0):
        raise ValueError("defect denominator precision must remain finite and strictly positive float64")
    d64 = difference.to(dtype=torch.float64)
    squared_norm = d64.square().sum(dim=(1, 2))
    if rms:
        squared_norm = squared_norm/(difference.shape[1]*difference.shape[2])
    norm = squared_norm.sqrt()
    if not torch.isfinite(norm).all():
        raise ValueError("defect norm must remain finite float64")
    nonzero = torch.any(difference != 0, dim=(1, 2))
    if torch.any(nonzero & (norm == 0)):
        raise ValueError("defect norm precision underflow: nonzero stored difference became zero")
    values = norm/denominator
    if not torch.isfinite(values).all():
        raise ValueError("defect division must remain finite float64")
    if torch.any(nonzero & (values == 0)):
        raise ValueError("defect division precision underflow: nonzero stored difference became zero")
    return values


@torch.no_grad()
def temporal_composition_defect(predictions: TemporalPredictions) -> SampleMetric:
    """TCD=sqrt(sum_(N,E)D_t64²)/(tau*sigma)，无训练 epsilon。

    raw 字段为同设备有限 float32 [B,N,E]；tau/sigma 为无梯度 CPU
    float64 [B] 正周期。输出同设备 float64 [B]、tcd_frobenius/周期^-2，
    no_grad；直接使用保存差值，不重跑 Phi 或 LN。零/非有限分母和非零差
    产生数值伪零时报错；低经验缺陷不证明统一理论界。
    """
    difference, denominator = _temporal_inputs(predictions)
    return SampleMetric(_norm_ratio(difference, denominator, rms=False), "tcd_frobenius", "control_cycle^-2")


@torch.no_grad()
def spatial_commutation_defect(predictions: SpatialPredictions) -> SampleMetric:
    """SCD=sqrt(sum_(N_target,E)D_s64²)/duration，无训练 epsilon。

    raw 为同形同设备 float32 [B,N_target,E]，N 匹配目标网格；duration
    为无梯度 CPU float64 [B] 正周期。输出 float64 [B]、scd_frobenius/周期^-1。
    no_grad 使用保存差值，不重跑 Phi/R、取 RMS 或改原图；零/非有限分母
    与数值伪零报错。低 SCD 不说明未来准确或理论界成立。
    """
    difference, denominator = _spatial_inputs(predictions)
    return SampleMetric(_norm_ratio(difference, denominator, rms=False), "scd_frobenius", "control_cycle^-1")


@torch.no_grad()
def temporal_composition_rms_defect(predictions: TemporalPredictions) -> SampleMetric:
    """附加 RMS=sqrt(mean_(N,E)D_t64²)/(tau*sigma)，另名 tcd_rms/周期^-2。

    合同沿 TCD，输出同设备 float64 [B]，相当于 Frobenius 除 sqrt(N*E)，
    不是 batch RMS。换秒制须按核实的周期 p 逐样本除 p²。
    """
    difference, denominator = _temporal_inputs(predictions)
    return SampleMetric(_norm_ratio(difference, denominator, rms=True), "tcd_rms", "control_cycle^-2")


@torch.no_grad()
def spatial_commutation_rms_defect(predictions: SpatialPredictions) -> SampleMetric:
    """附加 RMS=sqrt(mean_(N_target,E)D_s64²)/duration，另名 scd_rms/周期^-1。

    合同沿 SCD，输出同设备 float64 [B]；元素数取目标网格 N_target*E，
    不是细网格或 batch。换秒制须按核实的周期 p 逐样本除 p。
    """
    difference, denominator = _spatial_inputs(predictions)
    return SampleMetric(_norm_ratio(difference, denominator, rms=True), "scd_rms", "control_cycle^-1")


def _action_pair(raw_z: Tensor, executed: ActionSegments, alternative: ActionSegments,
                 duration: Tensor) -> None:
    """先核对两条已记录路径的完整定义、相对覆盖与异源元数据，不采样控制。"""
    _temporal_raw_contract(raw_z, "raw_z")
    for name, controls in (("executed", executed), ("alternative", alternative)):
        if not isinstance(controls, ActionSegments):
            raise TypeError(f"{name} must be ActionSegments")
        validate_action_segments(controls, query_duration=duration)
        controls.control_spec.__post_init__()
        if controls.u_values.shape[0] != raw_z.shape[0] or controls.u_values.device != raw_z.device:
            raise ValueError("both controls must match raw_z batch and actual device; "
                f"expected shape=[B={raw_z.shape[0]}, J>=1, m={controls.control_spec.action_dim}], "
                f"dtype=torch.float32, device={raw_z.device}; {name} actual shape={tuple(controls.u_values.shape)}, "
                f"dtype={controls.u_values.dtype}, device={controls.u_values.device}")
    if executed.control_spec != alternative.control_spec:
        raise ValueError("executed and alternative must have the same complete ControlSpec")
    if any(a == b for a, b in zip(executed.source_ids, alternative.source_ids)):
        raise ValueError("each alternative source ID must differ from executed source ID")


@torch.no_grad()
def action_intervention_margin(model: POJEPA, raw_z: Tensor, executed: ActionSegments,
                               alternative: ActionSegments, duration: Tensor,
                               grid: GridSpec, raw_target: Tensor, *,
                               loss_config: LossConfig) -> SampleMetric:
    """Delta_act=d(Phi(Z;alternative),target)-d(Phi(Z;executed),target)。

    起点/标签同设备 float32 [B,N,E]；duration 为 CPU float64 [B] 非负
    周期。两次同一完整 Phi/JEPA 距离仅替换控制；两路径须同 B/m/device、
    完整 ControlSpec、相对区间覆盖且逐样本 source_id 不同，允许切换点和
    绝对原点不同。输出 float32 [B]、delta_act/dimensionless，保留正负零。
    调用者先 eval；no_grad、不采样或加入训练。来源 ID 不证明真实来源，
    正 Delta 也不证明完整因果或物理正确性。
    """
    _eval_model(model)
    _action_pair(raw_z, executed, alternative, duration)
    executed_metric = future_error(model, raw_z, executed, duration, grid, raw_target, loss_config=loss_config)
    alternative_metric = future_error(model, raw_z, alternative, duration, grid, raw_target, loss_config=loss_config)
    values = alternative_metric.values-executed_metric.values
    if not torch.isfinite(values).all():
        raise ValueError("Delta_act must remain finite float32")
    return SampleMetric(values, "delta_act", "dimensionless")


def _quantile_levels(levels: tuple[float, ...]) -> tuple[float, ...]:
    """先核原实数域再转CPU64，拒绝越界、排序/去重和精度坍缩而不修补。"""
    if not isinstance(levels, tuple) or not levels:
        raise ValueError("quantile_levels must be a nonempty tuple")
    if any(isinstance(q, bool) or not isinstance(q, Real) for q in levels):
        raise TypeError("quantile_levels must contain real scalars, not bool/tensor/array")
    if any(not 0 <= q <= 1 for q in levels):
        raise ValueError("original quantile_levels must be within [0,1] before float64 conversion")
    try:
        converted = tuple(float(q) for q in levels)
    except (OverflowError, ValueError) as error:
        raise ValueError("quantile_levels must be representable as finite float64") from error
    if any(not math.isfinite(q) or not 0 <= q <= 1 for q in converted):
        raise ValueError("quantile_levels must be finite and within [0,1]")
    if any(left >= right for left, right in zip(converted, converted[1:])):
        raise ValueError("quantile_levels must remain strictly increasing in float64")
    return converted


def _summary_values(metric: SampleMetric) -> Tensor:
    """只在报告边界建立CPU64 detach副本；不动原values/grad，不筛坏样本。"""
    if not isinstance(metric, SampleMetric):
        raise TypeError("metric must be SampleMetric")
    if any(not isinstance(s, str) or not s.strip() for s in (metric.name, metric.unit)):
        raise ValueError("metric name and unit must be nonempty strings")
    values = metric.values
    if not isinstance(values, Tensor):
        raise TypeError("metric values must be a tensor")
    if values.is_nested or values.layout != torch.strided or values.is_meta:
        raise ValueError("metric values must be a real dense tensor")
    if values.ndim != 1 or values.numel() == 0 or values.dtype not in (torch.float32, torch.float64):
        raise ValueError("metric values must be nonempty one-dimensional float32/float64 [B]; "
            f"expected shape=[positive B], dtype=float32/float64, device=CPU/CUDA; "
            f"actual shape={tuple(values.shape)}, dtype={values.dtype}, device={values.device}")
    if values.device.type not in ("cpu", "cuda") or not torch.isfinite(values).all():
        raise ValueError("metric values must use CPU/CUDA and contain only finite samples")
    return values.detach().to(device="cpu", dtype=torch.float64, copy=True)


def summarize_metric(metric: SampleMetric, *, quantile_levels: tuple[float, ...]) -> MetricSummary:
    """逐样本向量的 CPU float64 均值与 linear 分位，不替代原始分布。

    values 为非空有限 float32/64 [B]；分位水平为非空有限实数 tuple，
    位于 [0,1]，转 double 后严格递增。仅报告边界 detach/复制，count=B。
    不丢失败样本、排序水平或补0；调用者保留轨迹映射，先拼全体窗口再
    统计，窗口加权均值与等轨迹宏平均分别报告。
    """
    levels = _quantile_levels(quantile_levels)
    values = _summary_values(metric)
    mean = values.mean()
    quantiles = torch.quantile(values, torch.tensor(levels, dtype=torch.float64, device="cpu"), interpolation="linear")
    if not torch.isfinite(mean) or not torch.isfinite(quantiles).all():
        raise ValueError("metric summary statistics must remain finite float64")
    return MetricSummary(values.numel(), float(mean), levels, tuple(float(q) for q in quantiles))
