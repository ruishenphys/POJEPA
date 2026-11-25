# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# PO-JEPA vendored copy: facebookresearch/jepa-wms @ 13cf1d9 (2026-04-11), licensed under
# Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0),
# https://creativecommons.org/licenses/by-nc/4.0/ . Non-commercial academic use with attribution;
# see THIRD_PARTY_NOTICES.md in po-jepa-full/. Paper: Terver et al., "What Drives Success in
# Physical Planning with Joint-Embedding Predictive World Models", arXiv:2512.24497 (TMLR 2026).
#
#   app/plan_common/models/vit.py        Attention, FeedForward, Transformer
#       (the DINO-WM predictor; upstream says: "Taken from https://github.com/gaoyuezhou/dino_wm,
#        itself adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py")
#   app/plan_common/models/AdaLN_vit.py  FWAdaLNBlock, modulate  (the JEPA-WM AdaLN predictor block)
#
# Changes from upstream (everything else is copied verbatim, parameter names unchanged):
#   vit.py:
#     - einops.rearrange replaced by the equivalent view/permute (einops is not a dependency here);
#     - the causal mask is passed to forward() as a boolean [N, N] tensor (True = attend) instead of the
#       module-level NUM_PATCHES/NUM_FRAMES buffer, so that 8x8/12x12/16x16 grids and short contexts work;
#       Transformer.forward passes it to every Attention;
#     - only the manual attention path (upstream use_sdpa=False, "exact reproducibility with DINO-WM")
#       is kept; the masked_fill/softmax arithmetic is unchanged.
#   AdaLN_vit.py:
#     - FWAdaLNBlock repeats the per-frame modulation H_patches*W_patches + cond_tokens times; upstream
#       repeats grid_size**2 + cond_tokens times, which equals H*W only on the 16x16 grid;
#     - only the RoPE + GELU branch is kept (the JEPA-WM recipe: use_rope true, use_SiLU false,
#       drop_path 0); RoPEAttention and MLP are taken from third_party/vjepa2_ac.py (ACRoPEAttention,
#       MLP), whose code is the same as jepa-wms src/models/utils/modules.py RoPEAttention/MLP
#       (only the SDPA kernel context manager differs; the attention arithmetic is identical).

import torch
from torch import nn

from .vjepa2_ac import MLP, ACRoPEAttention


# ---------------------------------------------------------------- DINO-WM predictor (vit.py)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.dropout_p = dropout

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x, mask):
        B, T, C = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # rearrange(t, "b n (h d) -> b h n d", h=self.heads)
        q, k, v = (t.view(B, T, self.heads, -1).transpose(1, 2) for t in qkv)

        # Original manual attention implementation
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # apply causal mask
        dots = dots.masked_fill(~mask[:T, :T], float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        # rearrange(out, "b h n d -> b n (h d)")
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, mask):
        for attn, ff in self.layers:
            x = attn(x, mask) + x
            x = ff(x) + x

        return self.norm(x)


def generate_sdpa_mask(npatch, nwindow, device=None):
    """Generate causal attention mask for SDPA.

    For SDPA with boolean attn_mask:
    - True = attend
    - False = mask out

    Shape: (nwindow * npatch, nwindow * npatch)
    """
    # Build block-causal boolean mask
    total_len = npatch * nwindow
    mask = torch.zeros(total_len, total_len, device=device).bool()
    mask_block = torch.ones(npatch, npatch, device=device).bool()

    for i in range(nwindow):
        for j in range(i + 1):
            # Block (i, j) should attend (i can attend to j where j <= i)
            row_start = i * npatch
            row_end = (i + 1) * npatch
            col_start = j * npatch
            col_end = (j + 1) * npatch
            mask[row_start:row_end, col_start:col_end] = mask_block

    return mask


# ---------------------------------------------------------------- JEPA-WM AdaLN predictor (AdaLN_vit.py)

class FWAdaLNBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        **kwargs,
    ):
        super().__init__()
        assert use_rope and act_layer is nn.GELU and drop_path == 0.0
        self.norm1 = norm_layer(dim)
        self.grid_size = grid_size
        self.attn = ACRoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            use_sdpa=use_sdpa,
            is_causal=is_causal,
            grid_size=grid_size,
            proj_drop=drop,
        )

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x, z, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, cond_tokens=0):
        """
        Input:
            x : B, N, C with N = T*H*W
            z : B, T, D
        Returns:
            B, N, D
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(z).repeat_interleave((H_patches * W_patches + cond_tokens), dim=1).chunk(6, dim=2)
        )
        y = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            mask=mask,
            attn_mask=attn_mask,
            T=T,
            H=H_patches,
            W=W_patches,
            action_tokens=cond_tokens,
        )
        x = x + self.drop_path(y * gate_msa)
        x = x + self.drop_path(gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift
