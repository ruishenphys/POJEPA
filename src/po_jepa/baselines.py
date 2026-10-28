"""附录普通 Transformer 与未训练 identity 参照。

Transformer 结构为固定 baseline 实现选择；只用当前 raw Z、已知控制、
查询时长和坐标。raw 输出不额外归一化，JEPA 距离由共同损失定义。
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .actions import ActionSegments, clip_action_segments, validate_action_segments
from .geometry import GridSpec


@dataclass(frozen=True)
class TransformerSpec:
    """完整构型；生产E=1408，小E只用于明确虚构的CPU测试。"""
    encoder_dim: int = 1408
    action_dim: int = 2
    width: int = 64
    heads: int = 4
    layers: int = 1
    feedforward: int = 128
    dropout: float = 0.
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self):
        """分配前拒绝非法整数/精度设置；不提供AMP或隐式结构降配。"""
        if any(type(v) is not int or v <= 0 for v in
               (self.encoder_dim, self.action_dim, self.width, self.heads, self.layers, self.feedforward)):
            raise ValueError("positive integer dimensions required")
        if self.action_dim != 2 or self.width % self.heads or self.dropout != 0. or self.layer_norm_epsilon != 1e-5:
            raise ValueError("F05 requires two actions, divisible heads, zero dropout and fixed LayerNorm epsilon")


def validate_inputs(raw: Tensor, controls: ActionSegments, duration: Tensor,
                    grid: GridSpec, encoder_dim: int, device) -> None:
    """核有限 FP32 [B,N,E]、实际网格与时间/控制，padding 不作有效命令。"""
    if (not isinstance(raw, Tensor) or raw.layout != torch.strided or raw.is_nested or raw.is_meta
        or raw.dtype != torch.float32 or raw.ndim != 3 or min(raw.shape) <= 0
        or raw.shape[-1] != encoder_dim or raw.device != device or not torch.isfinite(raw).all()):
        raise ValueError("raw must be finite model-device float32 [B,N,encoder_dim]")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be GridSpec")
    grid.__post_init__()
    if raw.shape[1] != grid.height*grid.width:
        raise ValueError("raw N must match actual grid")
    validate_action_segments(controls, query_duration=duration)
    if controls.u_values.shape[0] != raw.shape[0] or controls.u_values.device != device:
        raise ValueError("control B/device must match raw")
    if controls.control_spec.action_dim != 2:
        raise ValueError("F05 requires two control channels")


class TokenTransformer(nn.Module):
    """普通单图token Transformer：空间token和逐段动作token共同做标准自注意力。

    空间token=P_in(Z)+Linear(x,y)+Linear(T)，动作token=Linear(u_x,u_y,start,end)。
    这些是已登记的baseline编码选择；不是原稿动力学公式。无dropout、额外历史、
    hidden q或时间压缩。仅空间token经P_out读出raw Z；所有参数真正参与前向。
    """
    def __init__(self, spec: TransformerSpec):
        """调用者控制初始化RNG；生产工厂另以固定1703局部隔离构造。"""
        super().__init__()
        spec.__post_init__()
        self.spec, self.encoder_dim = spec, spec.encoder_dim
        self.p_in = nn.Linear(spec.encoder_dim, spec.width)
        self.grid_embed = nn.Linear(2, spec.width, bias=False)
        self.duration_embed = nn.Linear(1, spec.width, bias=False)
        self.action_embed = nn.Linear(4, spec.width)
        layer = nn.TransformerEncoderLayer(spec.width, spec.heads, spec.feedforward,
            dropout=spec.dropout, activation="gelu", layer_norm_eps=spec.layer_norm_epsilon,
            batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, spec.layers, enable_nested_tensor=False)
        self.p_out = nn.Linear(spec.width, spec.encoder_dim)

    def action_tokens(self, controls: ActionSegments, duration: Tensor) -> tuple[Tensor, Tensor]:
        """只取查询前缀，保留逐段值/起止周期；先where清padding，再投影，绝不NaN乘零。"""
        validate_action_segments(controls, query_duration=duration)
        clipped = clip_action_segments(controls, start=torch.zeros_like(duration), end=duration)
        mask = clipped.segment_valid.to(clipped.u_values.device)
        values = torch.where(mask[..., None], clipped.u_values, torch.zeros_like(clipped.u_values))
        times = torch.stack([torch.where(clipped.segment_valid, t, torch.zeros_like(t))
                             for t in (clipped.starts, clipped.ends)], -1).to(values)
        if not torch.isfinite(times).all() or torch.any(times[mask][:,1] <= times[mask][:,0]):
            raise ValueError("positive control interval not representable as model float32")
        return torch.cat((values, times), -1), mask

    def forward(self, raw: Tensor, controls: ActionSegments, duration: Tensor, grid: GridSpec) -> Tensor:
        """输入[B,N,E]→同形raw；CPU64时长为控制周期，坐标为已知CPU64[N,2]。"""
        validate_inputs(raw, controls, duration, grid, self.encoder_dim, self.p_in.weight.device)
        actions, valid = self.action_tokens(controls, duration)
        time = duration.to(raw)
        if not torch.isfinite(time).all() or torch.any((duration > 0) & (time.detach().cpu().double() == 0)):
            raise ValueError("query time not representable at model precision")
        spatial = self.p_in(raw)+self.grid_embed(grid.coords.to(raw))[None]+self.duration_embed(time[:, None])[:, None]
        tokens = torch.cat((spatial, self.action_embed(actions)), 1)
        padding = torch.cat((torch.zeros(raw.shape[:2], dtype=torch.bool, device=raw.device), ~valid), 1)
        output = self.p_out(self.encoder(tokens, src_key_padding_mask=padding)[:, :raw.shape[1]])
        if not torch.isfinite(output).all():
            raise ValueError("nonfinite Transformer raw prediction")
        return output


class IdentityPredictor(nn.Module):
    """不训练的当前特征参照；忽略合法控制不代表对未来预测正确。"""
    def __init__(self, encoder_dim: int = 1408):
        """不创建任何参数或优化器；E只用于核验输入。"""
        super().__init__()
        self.encoder_dim = encoder_dim

    def forward(self, raw, controls, duration, grid):
        """返回同一个当前raw对象；仍核可见输入合同，不读取任何目标。"""
        if not isinstance(raw, Tensor):
            raise TypeError("raw must be a Tensor")
        validate_inputs(raw, controls, duration, grid, self.encoder_dim, raw.device)
        return raw
