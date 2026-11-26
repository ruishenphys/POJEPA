"""DINO-WM、JEPA-WM 两个参数匹配基线的预测器核心（接在冻结 V-JEPA 2 ViT-g 特征上，ac_baseline.ACPredictor 包装）。

依据：facebookresearch/jepa-wms @ 13cf1d9（CC BY-NC 4.0，拷贝的代码在 third_party/jepa_wms.py），评估报告
baseline-eval-jepa-wm/2026-11-25_jepa-wm-dino-wm-assessment.md §3、§4。两个核心和 V-JEPA 2-AC 的核心接口相同：
`predict(context [B,T,N,E], actions [B,T,m], H, W) → [B,T,N,E]`，第 t 帧位置输出第 t+1 帧的预测（块因果，只看第 0…t 帧）；
都有 predictor_embed（E→宽）和 predictor_proj（宽→E），读出尺度钩子挂在 predictor_proj 上。

- DinoWMCore（pred_type=dino_wm，“特征拼接”）：动作 Linear(m, a) 拼到每个 patch 的通道上，token 维 D = 宽 + a；
  加固定 3 轴 sincos 位置编码（时间 = 窗口内帧号，空间行列按 16/H、16/W 缩放到 16×16 网格，和 AC 的 RoPE 同一缩放），
  再过 DINO-WM 的 Transformer（前置 LN 注意力 + 前馈、块因果掩码、末尾 LN），只取前“宽”维过 predictor_proj。
  和上游的差别：上游没有 predictor_embed/proj（DINOv2 ViT-S 只有 384 维），这里 1408 维要先降维才能参数匹配；
  上游位置编码是可学习的 randn 表（num_frames × 256 个位置），只对 16×16 有定义，8×8、12×12 评测用不了，
  而且 3 × 256 × D 个参数会占掉参数匹配预算的五分之一，所以换成不训练的 sincos；dropout 上游 0.1，这里 0（参数匹配的小模型，
  和其余基线一致，也避免哪条评测路径漏设 eval() 时预测带随机性）；初始化沿用 PyTorch 默认（上游 DINO-WM 也没有自定义初始化）。
- AdaLNCore（pred_type=AdaLN，JEPA-WM 推荐配方）：动作 Linear(m, 宽) 得到每帧的条件向量 z_t，每层 AdaLN-modulation
  （SiLU + Linear(宽, 6·宽)）给出 shift/scale/gate 调制本帧全部 token；注意力是 3 轴 RoPE（ACRoPEAttention，空间按 16/H 缩放），
  块因果掩码、没有条件 token。初始化照上游 VisionTransformerAdaLN：Linear 截断正态 std 0.02、偏置 0，LN 为 1/0，
  第 l 层 attn.proj 与 mlp.fc2 除以 √(2l)，AdaLN 线性层 std = 0.02 × init_scale_factor_adaln（10）、偏置 0。
  没有本体感受（no-prop；仿真没有本体状态，论文 Figure 4a 的 no-prop 版本）。
"""
import math

import torch
from torch import Tensor, nn

from .third_party import jepa_wms, vjepa2_ac

INIT_STD = 0.02
INIT_SCALE_FACTOR_ADALN = 10   # 上游 init_video_model 默认，JEPA-WM 配置未改


def sincos_1d(positions: Tensor, dim: int) -> Tensor:
    """MAE 写法的一维 sincos：ω_i = 1/10000^(i/(dim/2))，输出 [sin(p·ω), cos(p·ω)]，float64 [len, dim]。"""
    if dim % 2:
        raise ValueError("sincos dimension must be even")
    omega = 1.0/10000**(torch.arange(dim//2, dtype=torch.float64)/(dim/2.0))
    angles = positions.to(torch.float64)[:, None]*omega[None]
    return torch.cat([angles.sin(), angles.cos()], 1)


def sincos_3d(frames: int, height: int, width: int, dim: int, grid_size: int = 16) -> Tensor:
    """[frames·height·width, dim] 的固定位置编码（float64），token 顺序 = 帧、行、列。
    三段各 2·⌊⌊dim/3⌋/2⌋ 维：时间 = 帧号，行、列 = 下标 × grid_size/H（W），余下的维为 0（和 RoPE 的分法相同）。"""
    part = 2*((dim//3)//2)
    if part < 2:
        raise ValueError("token dimension too small for 3-axis sincos")
    t = torch.arange(frames, dtype=torch.float64).repeat_interleave(height*width)
    r = (torch.arange(height, dtype=torch.float64)*grid_size/height).repeat_interleave(width).repeat(frames)
    c = (torch.arange(width, dtype=torch.float64)*grid_size/width).repeat(frames*height)
    out = torch.zeros(frames*height*width, dim, dtype=torch.float64)
    out[:, :part] = sincos_1d(t, part)
    out[:, part:2*part] = sincos_1d(r, part)
    out[:, 2*part:3*part] = sincos_1d(c, part)
    return out


def block_causal_mask(frames: int, tokens: int, device) -> Tensor:
    """块因果布尔掩码 [frames·tokens]²，True = 可以看：第 t 帧的 token 看第 0…t 帧的全部 token。"""
    return jepa_wms.generate_sdpa_mask(tokens, frames, device=device)


class DinoWMCore(nn.Module):
    def __init__(self, *, encoder_dim: int, width: int, depth: int, heads: int, head_dim: int, mlp_hidden: int,
                 action_dim: int, action_embed_dim: int):
        super().__init__()
        self.width, self.action_embed_dim = width, action_embed_dim
        self.dim = width+action_embed_dim
        self.predictor_embed = nn.Linear(encoder_dim, width)
        self.action_encoder = nn.Linear(action_dim, action_embed_dim)
        self.transformer = jepa_wms.Transformer(self.dim, depth, heads, head_dim, mlp_hidden, dropout=0.0)
        self.predictor_proj = nn.Linear(width, encoder_dim)

    def predict(self, context: Tensor, actions: Tensor, height: int, width: int) -> Tensor:
        b, t, n, _ = context.shape
        x = self.predictor_embed(context)                                            # [B,T,N,宽]
        a = self.action_encoder(actions)[:, :, None].expand(b, t, n, self.action_embed_dim)
        x = torch.cat([x, a], dim=3)                                                 # 上游 concat_obs_act：[视觉, 动作]
        x = x.flatten(1, 2)+sincos_3d(t, height, width, self.dim).to(x)[None]
        x = self.transformer(x, block_causal_mask(t, n, x.device))
        x = x.view(b, t, n, self.dim)[..., :self.width]                               # 去掉动作通道
        return self.predictor_proj(x)


class AdaLNCore(nn.Module):
    def __init__(self, *, encoder_dim: int, width: int, depth: int, heads: int, mlp_ratio: float, action_dim: int):
        super().__init__()
        norm_layer = lambda d: nn.LayerNorm(d, eps=1e-6)   # noqa: E731（上游 vit_predictor_AdaLN：LayerNorm eps 1e-6）
        # 构造顺序同上游 VisionTransformerAdaLN.__init__：predictor_embed、action_encoder、blocks、norm、proj
        self.predictor_embed = nn.Linear(encoder_dim, width, bias=True)
        self.action_encoder = nn.Linear(action_dim, width, bias=True)
        self.predictor_blocks = nn.ModuleList([
            jepa_wms.FWAdaLNBlock(use_rope=True, grid_size=16, dim=width, num_heads=heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=True, act_layer=nn.GELU, norm_layer=norm_layer)
            for _ in range(depth)])
        self.predictor_norm = norm_layer(width)
        self.predictor_proj = nn.Linear(width, encoder_dim, bias=True)
        self.apply(self._init_weights)
        for layer_id, layer in enumerate(self.predictor_blocks):   # 上游 _rescale_blocks
            layer.attn.proj.weight.data.div_(math.sqrt(2.0*(layer_id+1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0*(layer_id+1)))
        with torch.no_grad():
            for block in self.predictor_blocks:
                linear = block.adaLN_modulation[1]
                vjepa2_ac.trunc_normal_(linear.weight, std=INIT_STD*INIT_SCALE_FACTOR_ADALN)
                nn.init.constant_(linear.bias, 0)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            vjepa2_ac.trunc_normal_(m.weight, std=INIT_STD)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def predict(self, context: Tensor, actions: Tensor, height: int, width: int) -> Tensor:
        b, t, n, _ = context.shape
        x = self.predictor_embed(context).flatten(1, 2)                              # [B,T·N,宽]
        z = self.action_encoder(actions)                                              # [B,T,宽]
        mask = vjepa2_ac.build_action_block_causal_attention_mask(t, height, width, add_tokens=0).to(x.device)
        for block in self.predictor_blocks:
            x = block(x, z, mask=None, attn_mask=mask, T=t, H_patches=height, W_patches=width, cond_tokens=0)
        x = self.predictor_norm(x).view(b, t, n, -1)
        return self.predictor_proj(x)
