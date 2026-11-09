"""v7 外部基线：V-JEPA 2-AC 动作条件预测器（facebookresearch/vjepa2，提交 204698b）。

两个版本，结构都来自官方 VisionTransformerPredictorAC（third_party/vjepa2_ac.py 原样拷贝）：
- vjepa2_ac：官方规模（24 层、宽 1024、16 头，约 3.05 亿参数），从官方权重 vjepa2-ac-vitg.pt 初始化后在本数据上微调。
- vjepa2_ac_small：同一结构缩到与 PO-JEPA 参数量相差 ≤1%（1 层、宽 64、4 头、MLP 224；token 写法 PO 为 227617 参数），从头训练。

与官方相比的适配（都写在这里，不改官方代码）：
- 动作维 7→2：action_encoder 换成新的 Linear(2, 宽)，随机初始化；
- 没有本体感受状态：官方的 state token 换成一个可学习常向量（官方规模用权重里 state_encoder 的偏置初始化）；
  extrinsics_encoder 本来就不用（use_extrinsics=False），删除；
- 每个控制 tick（0.25 s，正好是官方 4 fps 的帧间隔）一个动作 token，取该 tick 内分段常数控制的时间平均；
- 预测 Δt 个控制周期 = 逐帧自回归滚动 Δt 步（块因果注意力，上下文是起点帧 + 之前的预测帧）；
- 官方没有连续时间接口：半整数时长（只出现在时间一致性 TCD 的半程查询里）取相邻两步预测的平均，
  不进入未来误差（future 时长都是整数）；
- 8×8 / 12×12 网格：官方 RoPE 按 grid_size/H 把位置缩放到 16×16 网格上，直接用；和 Transformer 基线一样是 latent 插值。
- 输入输出与官方 normalize_reps 一致：上下文先做无仿射 LayerNorm，预测再做一次 LayerNorm。

训练（runtime/ac_training.py）照官方 app/vjepa_droid/train.py：9 帧片段（8 次转移）teacher forcing + 2 步自回归，L1。
"""
from dataclasses import asdict, dataclass
from functools import partial
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .actions import ActionSegments, validate_action_segments
from .geometry import GridSpec
from .third_party import vjepa2_ac

AC_CONDITIONS = ("vjepa2_ac", "vjepa2_ac_small")
OFFICIAL_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"


@dataclass(frozen=True)
class ACSpec:
    encoder_dim: int = 1408
    width: int = 1024
    depth: int = 24
    heads: int = 16
    mlp_ratio: float = 4.0
    action_dim: int = 2

    def __post_init__(self):
        if self.width % self.heads or (self.width//self.heads)//3//2 < 1:
            raise ValueError("head_dim too small for 3-axis RoPE")
        if self.action_dim != 2:
            raise ValueError("v7 systems use two control channels")


OFFICIAL = ACSpec()
SMALL = ACSpec(width=64, depth=1, heads=4, mlp_ratio=3.5)   # 2026-11-08：MLP 224，对齐 token 写法 PO 参数 227617（原 2.0 对齐模态写法）


def spec_for(condition: str) -> ACSpec:
    return {"vjepa2_ac": OFFICIAL, "vjepa2_ac_small": SMALL}[condition]


class ACPredictor(nn.Module):
    """V-JEPA 2-AC 预测器的 PO-JEPA 接口包装：model(raw[B,N,E], controls, duration[B], grid) → raw[B,N,E]。"""

    def __init__(self, spec: ACSpec):
        super().__init__()
        spec.__post_init__()
        self.spec = spec
        self.encoder_dim = spec.encoder_dim
        # 与官方 vit_ac_predictor 相同的构造参数（qkv_bias、LayerNorm eps=1e-6）；mlp_ratio 由 spec 给出（官方 4）
        core = vjepa2_ac.VisionTransformerPredictorAC(
            img_size=256, patch_size=16, num_frames=2, tubelet_size=2, embed_dim=spec.encoder_dim,
            predictor_embed_dim=spec.width, depth=spec.depth, num_heads=spec.heads, mlp_ratio=spec.mlp_ratio,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            is_frame_causal=False, use_rope=True, action_embed_dim=7, use_silu=False, wide_silu=True)
        del core.state_encoder, core.extrinsics_encoder
        core.action_encoder = nn.Linear(spec.action_dim, spec.width)
        vjepa2_ac.trunc_normal_(core.action_encoder.weight, std=core.init_std)
        nn.init.zeros_(core.action_encoder.bias)
        core.attn_mask = None
        self.core = core
        self.state_token = nn.Parameter(torch.zeros(spec.width))

    # ---- 官方权重 ----
    def load_official(self, state: dict) -> dict:
        """从 vjepa2-ac-vitg.pt 的 predictor 状态载入；除被替换的动作/状态/外参编码器外，键必须一一对应。"""
        if self.spec != OFFICIAL:
            raise ValueError("only the official-scale spec can load official weights")
        cleaned = {}
        for key, value in state.items():
            for prefix in ("module.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            cleaned[key] = value
        replaced = {k for k in cleaned if k.split(".")[0] in ("action_encoder", "state_encoder", "extrinsics_encoder")}
        own = self.core.state_dict()
        missing = sorted(set(own)-set(cleaned)-{"action_encoder.weight", "action_encoder.bias"})
        unexpected = sorted(set(cleaned)-set(own)-replaced)
        if missing or unexpected:
            raise RuntimeError(f"official AC checkpoint key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        bad = [k for k in own if k in cleaned and k not in replaced and own[k].shape != cleaned[k].shape]
        if bad:
            raise RuntimeError(f"official AC checkpoint shape mismatch: {bad[:5]}")
        self.core.load_state_dict({k: v for k, v in cleaned.items() if k not in replaced}, strict=False)
        with torch.no_grad():
            self.state_token.copy_(cleaned["state_encoder.bias"].to(self.state_token))
        return dict(loaded=len(cleaned)-len(replaced), replaced=sorted(replaced),
                    note="action_encoder 7→2 随机初始化；state token 用官方 state_encoder.bias 初始化")

    # ---- 前向 ----
    def predict_frames(self, context: Tensor, actions: Tensor, height: int, width: int) -> Tensor:
        """context [B,T,N,E]（已 LN）、actions [B,T,2] → 每个位置的下一帧预测 [B,T,N,E]（已 LN）。"""
        core = self.core
        b, t, n, e = context.shape
        x = core.predictor_embed(context)
        a = core.action_encoder(actions)[:, :, None]
        s = self.state_token.to(x.dtype).expand(b, t, 1, -1)
        x = torch.cat([a, s, x], dim=2).flatten(1, 2)
        mask = vjepa2_ac.build_action_block_causal_attention_mask(t, height, width, add_tokens=2).to(x.device)
        for block in core.predictor_blocks:
            x = block(x, mask=None, attn_mask=mask, T=t, H=height, W=width, action_tokens=2)
        x = x.view(b, t, 2+height*width, -1)[:, :, 2:]
        x = core.predictor_proj(core.predictor_norm(x))
        return F.layer_norm(x, (e,))

    def rollout(self, z0: Tensor, actions: Tensor, height: int, width: int) -> Tensor:
        """z0 [B,N,E]（已 LN），actions [B,K,2] → K 步自回归预测 [B,K,N,E]。"""
        frames = [z0]
        for k in range(actions.shape[1]):
            out = self.predict_frames(torch.stack(frames, 1), actions[:, :k+1], height, width)
            frames.append(out[:, -1])
        return torch.stack(frames[1:], 1)

    def forward(self, raw: Tensor, controls: ActionSegments, duration: Tensor, grid: GridSpec) -> Tensor:
        if raw.ndim != 3 or raw.shape[1] != grid.height*grid.width or raw.shape[2] != self.encoder_dim:
            raise ValueError("raw must be [B, grid tokens, encoder_dim]")
        validate_action_segments(controls, query_duration=duration)
        if controls.u_values.shape[0] != raw.shape[0] or duration.shape != (raw.shape[0],):
            raise ValueError("controls/duration batch must match raw")
        d = duration.detach().cpu().double()
        if torch.any(d <= 0) or torch.any((d*2) != (d*2).round()):
            raise ValueError("V-JEPA 2-AC rollout needs positive integer or half-integer control cycles")
        steps = int(math.ceil(float(d.max())))
        actions = tick_actions(controls, steps).to(raw)
        z0 = F.layer_norm(raw, (self.encoder_dim,))
        # 官方规模在 GPU 上用 bf16（与官方训练 dtype 一致）；输出转回 float32 交给共同的 JEPA 距离
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=raw.device.type == "cuda" and self.spec == OFFICIAL):
            preds = self.rollout(z0, actions, grid.height, grid.width).float()
        rows = []
        for i, value in enumerate(d.tolist()):
            k = int(math.floor(value))
            if value == k:
                rows.append(preds[i, k-1])
            else:   # 半整数：相邻两步平均（只用于 TCD 的半程查询）
                lower = z0[i] if k == 0 else preds[i, k-1]
                rows.append(.5*(lower+preds[i, k]))
        return torch.stack(rows)


def tick_actions(controls: ActionSegments, steps: int) -> Tensor:
    """每个控制 tick [j, j+1) 内分段常数控制的时间平均 → [B, steps, 2]；超出查询时长的部分记 0。"""
    starts, ends = controls.starts.double(), controls.ends.double()            # CPU [B,J]
    valid = controls.segment_valid
    lo = torch.arange(steps, dtype=torch.float64)[None, :, None]               # [1,K,1]
    end = torch.minimum(ends, controls.duration.double()[:, None])[:, None, :]  # [B,1,J]
    overlap = (torch.minimum(end, lo+1)-torch.maximum(starts[:, None, :], lo)).clamp(min=0)
    overlap = overlap*valid[:, None, :]
    weight = overlap/overlap.sum(-1, keepdim=True).clamp(min=1e-12)            # [B,K,J]
    u = torch.where(valid[..., None].to(controls.u_values.device), controls.u_values,
                    torch.zeros_like(controls.u_values))
    return torch.einsum("bkj,bjm->bkm", weight.to(u), u)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def describe(condition: str) -> dict:
    spec = spec_for(condition)
    return dict(condition=condition, spec=asdict(spec), init="official_checkpoint" if condition == "vjepa2_ac" else "scratch",
                checkpoint_url=OFFICIAL_CHECKPOINT_URL if condition == "vjepa2_ac" else None)
