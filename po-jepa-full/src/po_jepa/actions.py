"""已知分段常值控制的转换、截取与调度。

受控组合采用 u1⊕u2；时间由物理秒转换为控制周期，动作幅值单独缩放。
控制命令不是整段位移，不能随切分乘除持续时长。
"""

from dataclasses import dataclass
from fractions import Fraction
import math

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class ControlSpec:
    """预测前已知的控制单位表，六个通道 tuple 均长 m。

    P 为秒/周期；幅值转换 u_model=(u_physical-offset)/scale，不改变持续时长。
    """

    action_dim: int
    control_period_seconds: float
    channel_names: tuple[str, ...]
    channel_units: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    scale: tuple[float, ...]
    offset: tuple[float, ...]
    definition_source: str

    def __post_init__(self) -> None:
        """拒绝缺单位或不可用的缩放，防止不同物理意义的数值悄悄混用。"""
        if type(self.action_dim) is not int or self.action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        fields = (self.channel_names, self.channel_units, self.lower_bounds,
                  self.upper_bounds, self.scale, self.offset)
        if any(not isinstance(x, tuple) or len(x) != self.action_dim for x in fields):
            raise ValueError("all six channel tuples must have length action_dim")
        if any(not isinstance(x, str) or not x.strip() for x in (*self.channel_names, *self.channel_units, self.definition_source)):
            raise ValueError("channel names, units and definition_source must be explicit")
        numbers = (self.control_period_seconds, *self.lower_bounds, *self.upper_bounds, *self.scale, *self.offset)
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in numbers):
            raise ValueError("control metadata must contain finite real numbers")
        if self.control_period_seconds <= 0 or any(x <= 0 for x in self.scale):
            raise ValueError("period and all scales must be positive")
        if any(lo > hi for lo, hi in zip(self.lower_bounds, self.upper_bounds)):
            raise ValueError("physical lower bounds must not exceed upper bounds")


@dataclass(frozen=True)
class ActionSegments:
    """一批已知控制路径；张量只读，调用者不得原地改写。

    starts/ends是CPU float64 [B,J]内部周期，duration是[B]内部周期；
    source_origin_seconds是CPU float64 [B]绝对物理秒，时间均无梯度。
    u_values是调用者模型设备上的float32 [B,J,m]，保留动作值梯度。
    n_segments是CPU int64 [B]，segment_valid是CPU bool [B,J]。
    只有前n段有效；padding不等于零控制，不参与任何查询或梯度。
    source_ids是每条物理路径的稳定ID。完整验证由validate_action_segments负责。
    """

    starts: Tensor
    ends: Tensor
    u_values: Tensor
    n_segments: Tensor
    segment_valid: Tensor
    source_origin_seconds: Tensor
    duration: Tensor
    control_spec: ControlSpec
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class ActionSlicePair:
    """同一路径的先后两段；second相对时间重置为0，但保留物理原点。"""

    first: ActionSegments
    second: ActionSegments


@dataclass(frozen=True)
class StepSchedule:
    """单步时间计划，无状态积分：time是步起点，下一起点为time+delta。

    time/delta是CPU float64 [B]内部周期；active为CPU bool [B]；
    segment_index为CPU int64 [B]。结束样本delta=0、index=-1，不再推进。
    """

    time: Tensor
    delta: Tensor
    active: Tensor
    segment_index: Tensor


def _cpu(x: Tensor, shape: tuple[int, ...], dtype: torch.dtype, name: str, *, finite: bool = True) -> None:
    """只核对元数据；含padding的矩阵将有限值检查留给有效段。"""
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if x.shape != shape or x.dtype != dtype or x.device.type != "cpu" or x.requires_grad:
        raise ValueError(f"{name} must be CPU {dtype}, shape {shape}, without gradient")
    if finite and not torch.isfinite(x).all():
        raise ValueError(f"{name} must be finite")


def _near(x: float, y: float, query: float) -> bool:
    """仅按内部周期判断舍入范围，避免巨大绝对秒原点放宽时间边界。"""
    return abs(x-y) <= 1e-12 * max(1., abs(query), abs(x), abs(y))


def _canonical(controls: ActionSegments, query: Tensor | None = None, *, check_values: bool = True) -> tuple[Tensor, Tensor]:
    """核验完整路径并返回新对齐视图；不修改调用者的端点或动作值。

    容差内邻接以左段end为界，首点为0、末点为duration；严格正段不能
    被对齐成空段。调度专用check_values=False只读形状/时间，不读值设备。
    """
    if not isinstance(controls, ActionSegments):
        raise TypeError("controls must be ActionSegments")
    if query is None:
        query = controls.duration
    spec, values = controls.control_spec, controls.u_values
    if not isinstance(spec, ControlSpec):
        raise TypeError("control_spec must be ControlSpec")
    if not isinstance(values, Tensor):
        raise TypeError("u_values must be a tensor")
    if values.ndim != 3 or values.dtype != torch.float32 or min(values.shape) < 1 or values.shape[-1] != spec.action_dim:
        raise ValueError("u_values must be float32 [B,J>=1,m], B>=1")
    batch, width, _ = values.shape
    for name in ("starts", "ends"):
        _cpu(getattr(controls, name), (batch, width), torch.float64, name, finite=False)
    _cpu(controls.n_segments, (batch,), torch.int64, "n_segments")
    _cpu(controls.segment_valid, (batch, width), torch.bool, "segment_valid")
    for name in ("source_origin_seconds", "duration"):
        _cpu(getattr(controls, name), (batch,), torch.float64, name)
    _cpu(query, (batch,), torch.float64, "query_duration")
    if not isinstance(controls.source_ids, tuple) or len(controls.source_ids) != batch or any(not isinstance(x, str) or not x.strip() for x in controls.source_ids):
        raise ValueError("one nonempty source ID is required per sample")
    counts = controls.n_segments
    if torch.any((counts < 0) | (counts > width)):
        raise ValueError("n_segments outside padded width")
    expected_mask = torch.arange(width)[None, :] < counts[:, None]
    if not torch.equal(controls.segment_valid, expected_mask):
        raise ValueError("segment_valid must be exactly j < n_segments")
    if torch.any(controls.duration < 0) or torch.any(query < 0):
        raise ValueError("duration and query must be nonnegative")
    starts, ends = torch.zeros_like(controls.starts), torch.zeros_like(controls.ends)
    for b, n_tensor in enumerate(counts):
        n = int(n_tensor); duration = float(controls.duration[b]); q = float(query[b])
        if (duration == 0) != (n == 0):
            raise ValueError("only an empty diagnostic path may have duration zero")
        if (q > 0 and duration == 0) or (q > duration and not _near(q, duration, q)):
            raise ValueError("control path does not cover query")
        if not n:
            continue
        s, e = controls.starts[b, :n], controls.ends[b, :n]
        if not torch.isfinite(s).all() or not torch.isfinite(e).all() or torch.any(e <= s):
            raise ValueError("effective segments must have finite strictly positive lengths")
        if not _near(float(s[0]), 0., q) or not _near(float(e[-1]), duration, q):
            raise ValueError("effective path must cover [0,duration)")
        for j in range(1, n):
            if not _near(float(s[j]), float(e[j-1]), q):
                raise ValueError("gap or overlap in effective control path")
        starts[b, 1:n] = e[:-1]
        ends[b, :n] = e
        ends[b, n-1] = duration
        if torch.any(ends[b, :n] <= starts[b, :n]):
            raise ValueError("roundoff alignment would remove a positive segment")
    if check_values:
        valid = values[expected_mask.to(values.device)]
        scale, offset = values.new_tensor(spec.scale), values.new_tensor(spec.offset)
        lower = (values.new_tensor(spec.lower_bounds)-offset)/scale
        upper = (values.new_tensor(spec.upper_bounds)-offset)/scale
        if not torch.isfinite(valid).all() or torch.any((valid < lower) | (valid > upper)):
            raise ValueError("effective model controls must be finite and within transformed bounds")
    return starts, ends


def from_physical_segments(*, starts_seconds: Tensor, ends_seconds: Tensor,
                           u_values: Tensor, n_segments: Tensor,
                           segment_valid: Tensor, origin_seconds: Tensor,
                           control_spec: ControlSpec,
                           source_ids: tuple[str, ...]) -> ActionSegments:
    """物理秒与物理命令转为内部控制表示。

    时间为无梯度 CPU float64 [B,J]/[B]；命令为模型设备 float32 [B,J,m]。
    仅有效段计算 (u-offset)/scale，padding 中的 NaN 不参与计算或梯度。
    """
    if not isinstance(control_spec, ControlSpec) or not isinstance(u_values, Tensor):
        raise TypeError("explicit ControlSpec and tensor u_values are required")
    if u_values.ndim != 3 or u_values.dtype != torch.float32 or min(u_values.shape) < 1 or u_values.shape[-1] != control_spec.action_dim:
        raise ValueError("physical commands must be float32 [B,J,m]")
    batch, width, _ = u_values.shape
    _cpu(starts_seconds, (batch, width), torch.float64, "starts_seconds", finite=False)
    _cpu(ends_seconds, (batch, width), torch.float64, "ends_seconds", finite=False)
    _cpu(origin_seconds, (batch,), torch.float64, "origin_seconds")
    _cpu(n_segments, (batch,), torch.int64, "n_segments")
    _cpu(segment_valid, (batch, width), torch.bool, "segment_valid")
    if torch.any((n_segments < 0) | (n_segments > width)) or not torch.equal(segment_valid, torch.arange(width)[None, :] < n_segments[:, None]):
        raise ValueError("invalid counts or padding mask")
    mask = segment_valid.to(u_values.device)
    valid = u_values[mask]
    if not torch.isfinite(valid).all() or torch.any((valid < u_values.new_tensor(control_spec.lower_bounds)) | (valid > u_values.new_tensor(control_spec.upper_bounds))):
        raise ValueError("effective physical commands violate finite physical bounds")
    scaled = (valid-u_values.new_tensor(control_spec.offset))/u_values.new_tensor(control_spec.scale)
    converted = torch.zeros_like(u_values).masked_scatter(mask[..., None].expand_as(u_values), scaled.flatten())
    origin = origin_seconds[:, None].expand(batch, width)
    starts = torch.zeros_like(starts_seconds).masked_scatter(segment_valid, (starts_seconds[segment_valid]-origin[segment_valid])/control_spec.control_period_seconds)
    ends = torch.zeros_like(ends_seconds).masked_scatter(segment_valid, (ends_seconds[segment_valid]-origin[segment_valid])/control_spec.control_period_seconds)
    duration = torch.stack([ends[b, int(n)-1] if n else ends.new_zeros(()) for b, n in enumerate(n_segments)])
    result = ActionSegments(starts, ends, converted, n_segments.clone(), segment_valid.clone(), origin_seconds.clone(), duration, control_spec, source_ids)
    s, e = _canonical(result, result.duration)
    return ActionSegments(s, e, converted, result.n_segments, result.segment_valid, result.source_origin_seconds, duration, control_spec, source_ids)


def validate_action_segments(controls: ActionSegments, *, query_duration: Tensor,
                             require_exact_end: bool = False) -> None:
    """验证完整表示和每样本查询覆盖，不修改数据，也不替调用者截取。

    query_duration为CPU float64 [B]内部周期；默认可短于路径，exact时
    须对齐末端。零查询只供诊断；空路径绝不能覆盖严格正查询。
    """
    if type(require_exact_end) is not bool:
        raise TypeError("require_exact_end must be bool")
    if query_duration is None:
        raise TypeError("query_duration must be an explicit tensor")
    _canonical(controls, query_duration)
    if require_exact_end and any(not _near(float(q), float(d), float(q)) for q, d in zip(query_duration, controls.duration)):
        raise ValueError("query must equal the complete path duration")


def _bounded(value: float, duration: float) -> float:
    """只将越过末端的舍入差对齐；不向前跳过末端之前的正区间。"""
    if value < 0 or (value > duration and not _near(value, duration, value)):
        raise ValueError("time outside covered interval")
    return min(value, duration)


def _pack(rows: list[tuple[Tensor, Tensor, Tensor]], *, origin: Tensor,
          duration: Tensor, template: ActionSegments) -> ActionSegments:
    """重新补齐批次；stack/pad保持重复动作值的反向累加，空片段也不伪造段。"""
    counts = torch.tensor([s.numel() for s, _, _ in rows], dtype=torch.int64)
    width = max(1, int(counts.max()))
    starts = torch.stack([F.pad(s, (0, width-s.numel())) for s, _, _ in rows])
    ends = torch.stack([F.pad(e, (0, width-e.numel())) for _, e, _ in rows])
    values = torch.stack([F.pad(v, (0, 0, 0, width-v.shape[0])) for _, _, v in rows])
    result = ActionSegments(starts, ends, values, counts, torch.arange(width)[None, :] < counts[:, None],
                            origin, duration, template.control_spec, template.source_ids)
    _canonical(result, duration)
    return result


def clip_action_segments(controls: ActionSegments, *, start: Tensor,
                         end: Tensor) -> ActionSegments:
    """取每样本 [start,end) 周期交集，可切段或取空诊断片段。

    局部时间从 0 开始，物理原点增加 start*P；保持动作幅值、来源和梯度。
    """
    starts, ends = _canonical(controls)
    batch = controls.duration.numel()
    _cpu(start, (batch,), torch.float64, "start")
    _cpu(end, (batch,), torch.float64, "end")
    if torch.any(start < 0) or torch.any(end < start):
        raise ValueError("clip requires 0 <= start <= end")
    bounded_start = torch.tensor([_bounded(float(x), float(d)) for x, d in zip(start, controls.duration)], dtype=torch.float64)
    bounded_end = torch.tensor([_bounded(float(x), float(d)) for x, d in zip(end, controls.duration)], dtype=torch.float64)
    if torch.any((end > start) & (bounded_end <= bounded_start)):
        raise ValueError("positive requested interval cannot be represented")
    duration = bounded_end-bounded_start
    origin = controls.source_origin_seconds + bounded_start*controls.control_spec.control_period_seconds
    for shift, old, new in zip(bounded_start, controls.source_origin_seconds, origin):
        recovered = float(new-old)/controls.control_spec.control_period_seconds
        if not math.isfinite(float(new)) or not _near(recovered, float(shift), float(shift)):
            raise ValueError("physical origin precision cannot represent this slice")
    rows = []
    for b, n_tensor in enumerate(controls.n_segments):
        n = int(n_tensor)
        left = torch.maximum(starts[b, :n], bounded_start[b])
        right = torch.minimum(ends[b, :n], bounded_end[b])
        chosen = right > left
        values = controls.u_values[b, :n][chosen.to(controls.u_values.device)]
        rows.append((left[chosen]-bounded_start[b], right[chosen]-bounded_start[b], values))
    return _pack(rows, origin=origin, duration=duration, template=controls)


def _exact_split_slice(controls: ActionSegments, *, start: Tensor,
                       duration: Tensor) -> ActionSegments:
    """按 float64 原值的精确有理区间取交集，再生成局部时间轴。

    start/duration 为 CPU float64 [B] 周期，源有效端点须严格覆盖且相邻一致。
    内部边界仅转换一次，首 0、末原 duration；正交集转换坍缩时报精度不足。
    index_select 保留动作来源、幅值和梯度，不补尾或丢段。
    """
    origin = controls.source_origin_seconds + start*controls.control_spec.control_period_seconds
    for shift, old, new in zip(start, controls.source_origin_seconds, origin):
        recovered = float(new-old)/controls.control_spec.control_period_seconds
        if not math.isfinite(float(new)) or not _near(recovered, float(shift), float(shift)):
            raise ValueError("physical origin precision cannot represent this slice")
    rows = []
    for b, count in enumerate(controls.n_segments):
        n = int(count)
        starts = [Fraction.from_float(float(v)) for v in controls.starts[b, :n]]
        ends = [Fraction.from_float(float(v)) for v in controls.ends[b, :n]]
        shift = Fraction.from_float(float(start[b]))
        stop = shift+Fraction.from_float(float(duration[b]))
        if (not n or starts[0] != 0
                or ends[-1] != Fraction.from_float(float(controls.duration[b]))
                or any(left != right for left, right in zip(starts[1:], ends[:-1]))):
            raise ValueError("raw source must strictly cover its duration without gap or overlap")
        if ends[-1] < stop:
            raise ValueError("raw source does not exactly cover requested split")
        chosen, boundaries = [], [0.]
        for j, (left, right) in enumerate(zip(starts, ends)):
            left, right = max(left, shift), min(right, stop)
            if right > left:
                boundary = float(right-shift)
                if boundary <= boundaries[-1]:
                    raise ValueError("time precision cannot represent a positive split interval")
                chosen.append(j)
                boundaries.append(boundary)
        indices = torch.tensor(chosen, dtype=torch.int64, device=controls.u_values.device)
        rows.append((torch.tensor(boundaries[:-1], dtype=torch.float64),
                     torch.tensor(boundaries[1:], dtype=torch.float64),
                     controls.u_values[b].index_select(0, indices)))
    return _pack(rows, origin=origin, duration=duration, template=controls)


def split_action_segments(controls: ActionSegments, *, tau: Tensor,
                          sigma: Tensor) -> ActionSlicePair:
    """按组合式生成 [0,tau) 与 [tau,tau+sigma)，保留同一控制路径。

    tau/sigma 为严格正 CPU float64 [B] 周期；第二段局部时间重置为 0。
    按原端点精确相交，不以舍入总时长拉长命令；欠覆盖、正段坍缩或总时长
    无法前进时报错。保留原 tau/sigma、幅值和来源，不读取真实中间状态。
    """
    _canonical(controls)
    batch = controls.duration.numel()
    _cpu(tau, (batch,), torch.float64, "tau")
    _cpu(sigma, (batch,), torch.float64, "sigma")
    if torch.any(tau <= 0) or torch.any(sigma <= 0):
        raise ValueError("tau and sigma must be strictly positive")
    total = tau+sigma
    if not torch.isfinite(total).all() or torch.any(total <= tau):
        raise ValueError("time precision cannot represent the positive second interval")
    return ActionSlicePair(_exact_split_slice(controls, start=torch.zeros_like(tau), duration=tau),
                           _exact_split_slice(controls, start=tau, duration=sigma))


def concatenate_action_segments(first: ActionSegments,
                                second: ActionSegments) -> ActionSegments:
    """按原稿u1⊕u2保留先后顺序；只接受同源且物理时间邻接的两段。

    检查完整单位表、值设备及逐样本source_id，第二段端点平移first.duration。
    可保留相邻同值段，恢复路径不要求段数相同；空诊断段也遵守来源和原点。
    """
    s1, e1 = _canonical(first)
    s2, e2 = _canonical(second)
    if first.duration.shape != second.duration.shape or first.control_spec != second.control_spec or first.source_ids != second.source_ids:
        raise ValueError("concatenation requires matching batch, complete units and source IDs")
    if first.u_values.device != second.u_values.device:
        raise ValueError("control value devices must match; transfer explicitly before concatenating")
    period = first.control_spec.control_period_seconds
    for left, right, duration in zip(first.source_origin_seconds, second.source_origin_seconds, first.duration):
        if not _near(float(right-left)/period, float(duration), float(duration)):
            raise ValueError("second physical origin is not adjacent to first endpoint")
    rows = []
    for b, (n1, n2) in enumerate(zip(first.n_segments, second.n_segments)):
        j, k = int(n1), int(n2)
        rows.append((torch.cat((s1[b, :j], s2[b, :k]+first.duration[b])),
                     torch.cat((e1[b, :j], e2[b, :k]+first.duration[b])),
                     torch.cat((first.u_values[b, :j], second.u_values[b, :k]))))
    return _pack(rows, origin=first.source_origin_seconds.clone(), duration=first.duration+second.duration, template=first)


def _indices(starts: Tensor, ends: Tensor, valid: Tensor, time: Tensor, active: Tensor) -> Tensor:
    """只在active样本按半开区间选段；精确切换点属于右边的新命令。"""
    index = torch.full(time.shape, -1, dtype=torch.int64)
    for b in torch.nonzero(active).flatten():
        matches = torch.nonzero(valid[b] & (starts[b] <= time[b]) & (time[b] < ends[b])).flatten()
        if matches.numel() != 1:
            raise ValueError("active time must select exactly one half-open segment")
        index[b] = matches[0]
    return index


def action_at(controls: ActionSegments, *, time: Tensor,
              active: Tensor) -> Tensor:
    """返回每样本当前命令[B,m] float32，设备与值相同，保留选中动作梯度。

    time是CPU float64 [B]内部周期，active是CPU bool [B]；active在最终
    终点没有命令，必须报错。inactive返回零占位，后续不得推进或统计。
    """
    starts, ends = _canonical(controls)
    batch = controls.duration.numel()
    _cpu(time, (batch,), torch.float64, "time")
    _cpu(active, (batch,), torch.bool, "active")
    index = _indices(starts, ends, controls.segment_valid, time, active)
    rows = torch.nonzero(active).flatten().to(controls.u_values.device)
    chosen = controls.u_values[rows, index[active].to(controls.u_values.device)]
    return controls.u_values.new_zeros((batch, controls.control_spec.action_dim)).index_copy(0, rows, chosen)


def next_step_schedule(controls: ActionSegments, *, time: Tensor,
                       query_duration: Tensor, max_step: float) -> StepSchedule:
    """delta=min(max_step,剩余查询,到下个动作边界)，不跨越控制切换点。

    时间为 CPU float64 [B] 内部周期，仅调度元数据，不积分或重置状态。
    结束样本 delta=0/index=-1，其余样本独立推进，包括仍可表示的正短步。
    """
    if type(max_step) not in (int, float) or not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be finite and positive")
    starts, ends = _canonical(controls, query_duration, check_values=False)
    _cpu(time, controls.duration.shape, torch.float64, "time")
    query = torch.tensor([_bounded(float(q), float(d)) for q, d in zip(query_duration, controls.duration)], dtype=torch.float64)
    current = torch.tensor([_bounded(float(t), float(q)) for t, q in zip(time, query)], dtype=torch.float64)
    active = current < query
    index = _indices(starts, ends, controls.segment_valid, current, active)
    delta = torch.zeros_like(current)
    for b in torch.nonzero(active).flatten():
        delta[b] = min(max_step, float(query[b]-current[b]), float(ends[b, index[b]]-current[b]))
        if delta[b] <= 0 or current[b]+delta[b] <= current[b]:
            raise ValueError("positive integration interval cannot be represented")
    return StepSchedule(current, delta, active, index)
