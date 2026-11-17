"""S28b（A 层）：在给定模型上实测论文 §5 的三个界，左边和右边用同一批量算，只评测、不训练。

记号和论文 §5 / 附录证明相同（编号按方案文档：引理 5.3 空间、引理 5.4 时间切分、定理 5.6 联合时空）：

- 范数 ‖x − y‖ = sqrt(δ_{h''}(x, y)) = (N_{h''}E)^{-1/2} s^{-1} ‖x − y‖_F，s² 取起点 z（canonical 网格）的输入尺度（式 input_scale），
  所有网格、所有段共用同一个 s（论文 §5 开头）。差在 float64 里算。
- 切分 Π = (τ_1, …, τ_n)，t_i = τ_1 + … + τ_i；第 i 段动作 u_i = u_[t_{i−1}, t_i)，用 actions._exact_split_slice 精确切开，
  和训练的 split_action_segments / 多段切分同一切法。Φ_Π(z) 按式 (partition_rollout) 逐段完整 Φ，每段输入是上一段的 raw 输出。

常数按证明里实际用到的那些量取上确界（这批样本、这些段上的最大值），所以“左边 ≤ 右边”是对实现的一致性检查：
组合路径、动作切法、范数和证明里的对象不一致时，界会被打破。

- 引理 5.4（时间）：p_i = Φ_{τ_i}(p_{i−1}; u_i)，f_i = Φ_{t_i}(z; u_[0,t_i))。
  ε_t = max_{i≥2} ‖f_i − Φ_{τ_i}(f_{i−1}; u_i)‖ / (t_{i−1}τ_i)；L = max(0, max_{i≥2, d_{i−1}>0} log(‖Φ_{τ_i}(f_{i−1}) − Φ_{τ_i}(p_{i−1})‖ / d_{i−1}) / τ_i)。
  左边 d_n = ‖Φ_T(z; u_[0,T)) − Φ_Π(z; u)‖；右边 ε_t Σ_{i≥2} t_{i−1}τ_i e^{L(T−t_i)} ≤ ε_t e^{LT}(T² − Σ τ_i²)/2。
- 引理 5.3（空间）：x_i = Φ^h_{τ_i}(x_{i−1}; u_i)，y_i = Φ^{h'}_{τ_i}(y_{i−1}; u_i)，y_0 = Rz。
  ε_s = max_i ‖RΦ^h_{τ_i}(x_{i−1}) − Φ^{h'}_{τ_i}(Rx_{i−1})‖ / τ_i；L 取 ‖Φ^{h'}_{τ_i}(Rx_{i−1}) − Φ^{h'}_{τ_i}(y_{i−1})‖ / ‖Rx_{i−1} − y_{i−1}‖ 的同样上界。
  左边 ‖RΦ^h_Π(z) − Φ^{h'}_Π(Rz)‖；右边 ε_s Σ_i τ_i e^{L(T−t_i)} ≤ ε_s T e^{LT}。
- 定理（联合）：‖RΦ^h_Π(z) − Φ^{h'}_{Π'}(Rz)‖ ≤ 空间项(Π) + 时间项(Π, 网格 h') + 时间项(Π', 网格 h')，三项共用同一组 ε_s、ε_t、L
  （三次引理用到的量合在一起取上确界）。
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .actions import ActionSegments, _exact_split_slice
from .geometry import GridSpec, SpatialTransfer, apply_spatial_transfer
from .losses import input_scale


def scaled_norm(a: Tensor, b: Tensor, scale2: Tensor) -> Tensor:
    """sqrt(δ(a, b)) = sqrt(mean_(N,E)(a − b)² / s²)，float64 [B]；a、b 同形 [B,N,E]，scale2 是 [B] 的 s²。"""
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("a and b must share shape [B,N,E]")
    difference = a.detach().double()-b.detach().double()
    return (difference.square().mean(dim=(1, 2))/scale2.to(device=difference.device)).sqrt()


def partition_times(partition: tuple[float, ...]) -> list[float]:
    """t_0 = 0, t_i = τ_1 + … + τ_i（float64，逐段累加，和切动作时用的端点相同）。"""
    if type(partition) is not tuple or not partition or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in partition):
        raise ValueError("partition must be a nonempty tuple of finite positive durations")
    times = [0.]
    for tau in partition:
        times.append(times[-1]+float(tau))
    return times


def _cycles(value: float, batch: int) -> Tensor:
    return torch.full((batch,), float(value), dtype=torch.float64)


def _slice(controls: ActionSegments, start: float, duration: float) -> ActionSegments:
    batch = controls.duration.numel()
    return _exact_split_slice(controls, start=_cycles(start, batch), duration=_cycles(duration, batch))


def _phi(model, z: Tensor, controls: ActionSegments, start: float, duration: float, grid: GridSpec) -> Tensor:
    """Φ_duration(z; u_[start, start+duration)) 一次完整前向。"""
    return model(z, _slice(controls, start, duration), _cycles(duration, z.shape[0]), grid)


def _log_ratio(numerator: Tensor, denominator: Tensor, tau: float) -> list[float]:
    """log(num/den)/τ，只取 den > 0 的样本（den = 0 时两边输入逐位相同，Φ 输出也相同，不约束 L）。"""
    out = []
    for n, d in zip(numerator.tolist(), denominator.tolist()):
        if d > 0:
            out.append(math.log(n/d)/tau if n > 0 else -math.inf)
    return out


@dataclass(frozen=True)
class TemporalTerms:
    """引理 5.4 在一个网格上的量：lhs [B]；局部缺陷比 eps（标量，max）；Lipschitz 指数候选 lipschitz（list）；
    partition / times；rollout 是 Φ_Π(z) 本身（供和训练的 composed 逐位比对、供定理用）；direct 是 Φ_T(z)。"""

    lhs: Tensor
    eps: float
    lipschitz: list
    partition: tuple
    times: list
    rollout: Tensor
    direct: Tensor


@dataclass(frozen=True)
class SpatialTerms:
    """引理 5.3 的量：lhs [B]（目标网格范数）；eps（ε_s 候选的 max）；lipschitz 候选；rollout_fine = Φ^h_Π(z)。"""

    lhs: Tensor
    eps: float
    lipschitz: list
    partition: tuple
    times: list
    rollout_fine: Tensor


@torch.no_grad()
def temporal_terms(model, start: Tensor, controls: ActionSegments, partition: tuple[float, ...], grid: GridSpec,
                   *, scale2: Tensor) -> TemporalTerms:
    """引理 5.4 证明里的 p_i、f_i 和两类比值；start 是起点（可以是 Rz，在 h' 上），scale2 是起点 z 的 s² [B]。"""
    times = partition_times(partition)
    p, f = start, start
    eps, lipschitz = [], []
    for i, tau in enumerate(partition, start=1):
        f_next = _phi(model, start, controls, 0., times[i], grid)                 # f_i = Φ_{t_i}(z; u_[0,t_i))
        p_next = _phi(model, p, controls, times[i-1], tau, grid)                  # p_i = Φ_{τ_i}(p_{i−1}; u_i)
        if i >= 2:
            bridge = _phi(model, f, controls, times[i-1], tau, grid)              # Φ_{τ_i}(f_{i−1}; u_i)
            eps += (scaled_norm(f_next, bridge, scale2)/(times[i-1]*tau)).tolist()
            lipschitz += _log_ratio(scaled_norm(bridge, p_next, scale2), scaled_norm(f, p, scale2), tau)
        p, f = p_next, f_next
    return TemporalTerms(scaled_norm(f, p, scale2), max(eps, default=0.), lipschitz, partition, times, p, f)


@torch.no_grad()
def spatial_terms(model, raw_z: Tensor, controls: ActionSegments, partition: tuple[float, ...],
                  transfer: SpatialTransfer, *, scale2: Tensor) -> SpatialTerms:
    """引理 5.3 证明里的 x_i、y_i 和两类比值（范数都在目标网格 h' 上）。"""
    times = partition_times(partition)
    x, y = raw_z, apply_spatial_transfer(raw_z, transfer)
    eps, lipschitz = [], []
    for i, tau in enumerate(partition, start=1):
        x_next = _phi(model, x, controls, times[i-1], tau, transfer.source)
        rx = apply_spatial_transfer(x, transfer)
        y_from_rx = _phi(model, rx, controls, times[i-1], tau, transfer.target)   # Φ^{h'}_{τ_i}(Rx_{i−1})
        y_next = _phi(model, y, controls, times[i-1], tau, transfer.target)
        eps += (scaled_norm(apply_spatial_transfer(x_next, transfer), y_from_rx, scale2)/tau).tolist()
        lipschitz += _log_ratio(scaled_norm(y_from_rx, y_next, scale2), scaled_norm(rx, y, scale2), tau)
        x, y = x_next, y_next
    lhs = scaled_norm(apply_spatial_transfer(x, transfer), y, scale2)
    return SpatialTerms(lhs, max(eps), lipschitz, partition, times, x)


def _lipschitz(*candidates: list) -> float:
    values = [v for group in candidates for v in group]
    return max([0.]+values)


def temporal_sum(times: list, eps: float, lipschitz: float) -> float:
    """ε_t Σ_{i≥2} t_{i−1}τ_i e^{L(T−t_i)}（引理 5.4 右边，式 temporal_accumulation_sharp）。"""
    total = times[-1]
    return eps*sum(times[i-1]*(times[i]-times[i-1])*math.exp(lipschitz*(total-times[i])) for i in range(2, len(times)))


def spatial_sum(times: list, eps: float, lipschitz: float) -> float:
    """ε_s Σ_i τ_i e^{L(T−t_i)}（引理 5.3 右边，式 spatial_accumulation 第一式）。"""
    total = times[-1]
    return eps*sum((times[i]-times[i-1])*math.exp(lipschitz*(total-times[i])) for i in range(1, len(times)))


def temporal_partition_bound(model, raw_z: Tensor, controls: ActionSegments, partition: tuple[float, ...],
                             grid: GridSpec) -> dict:
    """引理 5.4：返回 lhs [B]（list）、sharp、coarse、eps_t、L 和 Φ_Π(z)（rollout，float32 张量）。"""
    scale2 = input_scale(raw_z)
    terms = temporal_terms(model, raw_z, controls, partition, grid, scale2=scale2)
    lip = _lipschitz(terms.lipschitz)
    total = terms.times[-1]
    coarse = terms.eps*math.exp(lip*total)*(total**2-sum(t*t for t in partition))/2
    return dict(lhs=terms.lhs.tolist(), sharp=temporal_sum(terms.times, terms.eps, lip), coarse=coarse,
                eps_t=terms.eps, L=lip, rollout=terms.rollout, direct=terms.direct)


def spatial_partition_bound(model, raw_z: Tensor, controls: ActionSegments, partition: tuple[float, ...],
                            transfer: SpatialTransfer) -> dict:
    """引理 5.3：返回 lhs [B]（list）、sharp、coarse = ε_s T e^{LT}、eps_s、L。"""
    terms = spatial_terms(model, raw_z, controls, partition, transfer, scale2=input_scale(raw_z))
    lip = _lipschitz(terms.lipschitz)
    total = terms.times[-1]
    return dict(lhs=terms.lhs.tolist(), sharp=spatial_sum(terms.times, terms.eps, lip),
                coarse=terms.eps*total*math.exp(lip*total), eps_s=terms.eps, L=lip)


def joint_bound(model, raw_z: Tensor, controls: ActionSegments, partition: tuple[float, ...],
                other: tuple[float, ...], transfer: SpatialTransfer) -> dict:
    """定理（联合时空）：‖RΦ^h_Π(z) − Φ^{h'}_{Π'}(Rz)‖ ≤ 空间(Π) + 时间(Π, h') + 时间(Π', h')，三项共用 ε_s、ε_t、L。

    返回 lhs [B]、sharp（式 joint_bound_sharp）、coarse（式 main_bound）、simple（式 main_bound_simple）和三组常数。
    """
    scale2 = input_scale(raw_z)
    times, others = partition_times(partition), partition_times(other)
    if not math.isclose(times[-1], others[-1], rel_tol=0., abs_tol=0.):
        raise ValueError("both partitions must cover the same horizon T exactly")
    space = spatial_terms(model, raw_z, controls, partition, transfer, scale2=scale2)
    coarse_start = apply_spatial_transfer(raw_z, transfer)
    first = temporal_terms(model, coarse_start, controls, partition, transfer.target, scale2=scale2)
    second = temporal_terms(model, coarse_start, controls, other, transfer.target, scale2=scale2)
    lip = _lipschitz(space.lipschitz, first.lipschitz, second.lipschitz)
    eps_t = max(first.eps, second.eps)
    total = times[-1]
    lhs = scaled_norm(apply_spatial_transfer(space.rollout_fine, transfer), second.rollout, scale2)
    sharp = spatial_sum(times, space.eps, lip)+temporal_sum(times, eps_t, lip)+temporal_sum(others, eps_t, lip)
    coarse = math.exp(lip*total)*(space.eps*total+eps_t/2*(2*total**2-sum(t*t for t in partition)-sum(t*t for t in other)))
    simple = math.exp(lip*total)*(space.eps*total+eps_t*total**2)
    return dict(lhs=lhs.tolist(), sharp=sharp, coarse=coarse, simple=simple, eps_s=space.eps, eps_t=eps_t, L=lip)
