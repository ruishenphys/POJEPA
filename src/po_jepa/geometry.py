"""规则网格与固定空间转移 R。

整除降采样采用平均池化，其余采用双线性。坐标为无量纲 [x,y]；
粗网格坐标必须由同一 R 转移，不能重新生成规则中心。
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F


def _positive_size(height: int, width: int) -> None:
    """拒绝非整数/零尺寸，防止将空间格数误当连续坐标或布尔开关。"""
    if any(type(x) is not int or x <= 0 for x in (height, width)):
        raise ValueError("grid height/width must be positive integers")


@dataclass(frozen=True)
class GridSpec:
    """记录空间格形状和 token 坐标；不持有图像、动作或网络。

    coords 为 CPU float64 [height*width,2]、无梯度，列依次是 x/y。
    p=y*width+x：先向右再换行。构图前已知，预测时无需未来标签。
    字段不可重新赋值；调用者也应把其中的坐标张量当只读数据。
    """

    height: int
    width: int
    coords: Tensor
    coordinate_source: str

    def __post_init__(self) -> None:
        """核对几何来源和数值，避免非法坐标进入后续图谱计算。"""
        _positive_size(self.height, self.width)
        c = self.coords
        if not isinstance(c, Tensor) or c.dtype != torch.float64 or c.device.type != "cpu":
            raise ValueError("grid coords must be CPU float64")
        if c.shape != (self.height*self.width, 2) or c.requires_grad:
            raise ValueError("grid coords must be [H*W,2] without gradient")
        if not torch.isfinite(c).all() or torch.any((c < 0) | (c > 1)):
            raise ValueError("grid coords must be finite and within [0,1]")
        if not isinstance(self.coordinate_source, str) or not self.coordinate_source.strip():
            raise ValueError("coordinate_source must be explicit")


def _method(source: GridSpec, height: int, width: int) -> str:
    """整除降采样采用块平均，其余采用普通双线性。"""
    if (height <= source.height and width <= source.width
        and source.height % height == 0 and source.width % width == 0):
        return "average_pool"
    return "bilinear"


def _resize_field(values: Tensor, source: GridSpec, height: int, width: int, method: str) -> Tensor:
    """同一计算核作用于特征或坐标；仅改变 N，不改变前缀轴及 C。

    输入 [...,H*W,C] 暂转通道优先做空间计算，随后恢复原前缀；reshape
    也支持非连续输入。没有 detach/no_grad，因此预测侧仍可求 R 的反向。
    """
    prefix, channels = values.shape[:-2], values.shape[-1]
    field = values.reshape(-1, source.height, source.width, channels).permute(0, 3, 1, 2)
    if method == "average_pool":
        factor = (source.height//height, source.width//width)
        resized = F.avg_pool2d(field, kernel_size=factor, stride=factor)
    else:
        # half-pixel 坐标与 edge 延拓；不擅自增加抗锯齿滤波或 align_corners。
        resized = F.interpolate(field, size=(height, width), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1).reshape(*prefix, height*width, channels)


@dataclass(frozen=True)
class SpatialTransfer:
    """固定规则网格转移 R，无可训练参数。

    target.coords 必须等于 R(source.coords)，包括上采样边缘；按 float64 容差核对。
    """

    source: GridSpec
    target: GridSpec
    method: Literal["average_pool", "bilinear"]
    align_corners: bool
    boundary_mode: Literal["edge"]

    def __post_init__(self) -> None:
        """直接构造这个公开类型时也检查规则，拒绝伪造的坐标/方法。"""
        if not isinstance(self.source, GridSpec) or not isinstance(self.target, GridSpec):
            raise TypeError("source/target must be GridSpec")
        self.source.__post_init__()
        self.target.__post_init__()
        if (self.method != _method(self.source, self.target.height, self.target.width)
            or self.align_corners is not False or self.boundary_mode != "edge"):
            raise ValueError("invalid fixed spatial method/align_corners/boundary_mode")
        expected = _resize_field(self.source.coords[None], self.source,
                                 self.target.height, self.target.width, self.method)[0]
        if not torch.allclose(self.target.coords, expected, atol=1e-10, rtol=1e-8):
            raise ValueError("target coords must be obtained by the same spatial R")


def regular_grid(height: int, width: int) -> GridSpec:
    """创建 CPU float64 [H*W,2] 规则中心，按先 x 后 y 的行优先顺序。

    归一化中心为 ((x+.5)/W,(y+.5)/H)，不采用包含 0/1 的端点网格。
    """
    _positive_size(height, width)
    y, x = torch.meshgrid(torch.arange(height, dtype=torch.float64, device="cpu"),
                          torch.arange(width, dtype=torch.float64, device="cpu"), indexing="ij")
    return GridSpec(height, width, torch.stack(((x+.5)/width, (y+.5)/height), -1).reshape(-1, 2),
                    "regular cell centers; p=y*width+x; normalized [x,y]")


def build_spatial_transfer(source: GridSpec, *, target_height: int, target_width: int) -> SpatialTransfer:
    """创建确定 R 及 R 后坐标，不编码新图像、不建立第二份模型。

    整除且不放大的两轴做不重叠平均；其余请求使用普通双线性。
    同尺寸属于倍率 1 的平均池化。没有建立潜在巨大的稠密矩阵。
    """
    if not isinstance(source, GridSpec):
        raise TypeError("source must be GridSpec")
    source.__post_init__()
    _positive_size(target_height, target_width)
    method = _method(source, target_height, target_width)
    coords = _resize_field(source.coords[None], source, target_height, target_width, method)[0]
    target = GridSpec(target_height, target_width, coords,
                      f"{method} R of {source.height}x{source.width}; {source.coordinate_source}")
    return SpatialTransfer(source, target, method, False, "edge")


def apply_spatial_transfer(values: Tensor, transfer: SpatialTransfer) -> Tensor:
    """对 [B,N,C] 或 [B,T,N,C] 的空间轴施加同一个 R。

    输出 [B,N',C]/[B,T,N',C]，保持浮点 dtype/device 和输入梯度，不原地修改。
    """
    if not isinstance(transfer, SpatialTransfer):
        raise TypeError("transfer must be SpatialTransfer")
    transfer.__post_init__()
    if not isinstance(values, Tensor) or not values.is_floating_point():
        raise TypeError("values must be a floating point tensor")
    if values.ndim not in (3, 4) or any(x <= 0 for x in values.shape):
        raise ValueError("values must have nonempty [B,N,C] or [B,T,N,C] axes")
    if values.shape[-2] != transfer.source.height*transfer.source.width or not torch.isfinite(values).all():
        raise ValueError("spatial token N mismatch or nonfinite values")
    return _resize_field(values, transfer.source, transfer.target.height, transfer.target.width, transfer.method)
