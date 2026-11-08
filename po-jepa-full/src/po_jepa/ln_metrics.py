"""10 §3.1：同一网格上两个预测之间的逐样本差异，两种度量，float64 计算。

rms_ln(a, b)           = sqrt( sum_(N,E) (LN(a) - LN(b))^2 / (N*E) )
rms_input_scale(a, b, raw_z) = sqrt( sum_(N,E) (a - b)^2 / (N*E) ) / sqrt(s^2)，s^2 = losses.input_scale(raw_z)

LN 与 JEPA 损失的定义相同（逐 token 沿 E、总体方差、eps=1e-5、无仿射），但在 float64 上算。
a、b 先各自转 float64 再相减。两者都无量纲、返回与输入同设备的 float64 [B]、不带梯度；
不除时间（tcd_ln、scd_ln、分割差异的时间归一化由调用者做，10 §3.2–§3.5）。
"""

import torch
from torch import Tensor
from torch.nn import functional as F

from .losses import input_scale

LN_EPSILON = 1e-5


def _pair(a: Tensor, b: Tensor) -> None:
    """a、b：同形 [B,N,E]（三轴都为正）、都是 float32、同一设备、全部有限。"""
    for name, value in (("a", a), ("b", b)):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a tensor")
    if (a.is_nested or b.is_nested or a.layout != torch.strided or b.layout != torch.strided
            or a.ndim != 3 or any(size <= 0 for size in a.shape) or a.shape != b.shape):
        raise ValueError("a and b must be dense tensors with the same shape [positive B, positive N, positive E]")
    if a.dtype != torch.float32 or b.dtype != torch.float32 or a.device != b.device or a.is_meta:
        raise ValueError("a and b must both be float32 on the same real device")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("a and b must contain finite values")


def _rms(d64: Tensor) -> Tensor:
    """sum_(N,E) d^2/(N*E) 再开方，与 metrics._norm_ratio(rms=True) 的分子同一写法。"""
    return (d64.square().sum(dim=(1, 2))/(d64.shape[1]*d64.shape[2])).sqrt()


@torch.no_grad()
def rms_ln(a: Tensor, b: Tensor) -> Tensor:
    """RMS_LN(a, b)：跨模型（含 AC）比较用。float64 [B]，与输入同设备。"""
    _pair(a, b)
    e = a.shape[-1]
    la = F.layer_norm(a.detach().to(dtype=torch.float64), (e,), eps=LN_EPSILON)
    lb = F.layer_norm(b.detach().to(dtype=torch.float64), (e,), eps=LN_EPSILON)
    return _rms(la-lb)


@torch.no_grad()
def rms_input_scale(a: Tensor, b: Tensor, raw_z: Tensor) -> Tensor:
    """RMS_in(a, b)：对照理论用。raw_z 是 16×16 起点 [B,N0,E]（N0 可以和 a 的 N 不同），s^2 = losses.input_scale(raw_z)。"""
    _pair(a, b)
    if not isinstance(raw_z, Tensor):
        raise TypeError("raw_z must be a tensor")
    if (raw_z.is_nested or raw_z.layout != torch.strided or raw_z.ndim != 3 or raw_z.shape[1] <= 0
            or raw_z.shape[0] != a.shape[0] or raw_z.shape[2] != a.shape[2]
            or raw_z.dtype != torch.float32 or raw_z.device != a.device):
        raise ValueError("raw_z must be a dense float32 [B, positive N0, E] tensor with a's B and E on a's device")
    if not torch.isfinite(raw_z).all():
        raise ValueError("raw_z must contain finite values")
    d64 = a.detach().to(dtype=torch.float64)-b.detach().to(dtype=torch.float64)
    return _rms(d64)/input_scale(raw_z).sqrt()
