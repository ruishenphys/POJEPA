"""DINO-WM、JEPA-WM 参数匹配基线（2026-11-25）：参数量、AC 不变、块因果、上下文窗口、网格缩放、上游结构、两套损失、任务入口。"""
from dataclasses import fields
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from po_jepa import ac_baseline, wm_baselines
from po_jepa.geometry import regular_grid
from po_jepa.losses import LossConfig
from po_jepa.runtime import ac_training, checks
from po_jepa.third_party import jepa_wms
from test_model import d12_controls

ROOT = Path(__file__).resolve().parents[1]
WM = ("dino_wm_small", "jepa_wm_small")
PO_PARAMETERS = 227_617


def _loss():
    c = checks.load_protocol()["loss"]
    return LossConfig(**{f.name: c[f.name] for f in fields(LossConfig)})


def _model(condition, seed=0):
    torch.manual_seed(seed)
    return ac_baseline.ACPredictor(ac_baseline.spec_for(condition))


# ---------------------------------------------------------------- 规格与参数量

def test_conditions_and_parameter_match():
    assert ac_baseline.WM_CONDITIONS == WM and set(WM) <= set(ac_baseline.AC_CONDITIONS)
    counts = {c: ac_baseline.parameter_count(_model(c)) for c in WM}
    assert counts == dict(dino_wm_small=227_486, jepa_wm_small=228_064)
    for n in counts.values():
        assert abs(n-PO_PARAMETERS)*100 <= PO_PARAMETERS           # 同 AC-small、TF 的 1% 规则


def test_po_count_is_the_matching_target(monkeypatch):
    monkeypatch.setenv("PO_JEPA_PROTOCOL", "v7")
    po = checks.build_full_model(checks.load_protocol(), torch.device("cpu"))
    assert sum(p.numel() for p in po.parameters()) == PO_PARAMETERS


def test_specs_snapshots_and_descriptions():
    d, j = ac_baseline.DINO_WM_SMALL, ac_baseline.JEPA_WM_SMALL
    assert (d.width, d.depth, d.heads, d.conditioning, d.action_embed_dim, d.head_dim, d.mlp_hidden, d.context_frames) == \
        (64, 1, 4, "feature", 10, 16, 176, 3)
    assert (j.width, j.depth, j.heads, j.mlp_ratio, j.conditioning, j.context_frames) == (64, 1, 4, .5, "adaln", 3)
    assert ac_baseline.spec_snapshot(j) == dict(encoder_dim=1408, width=64, depth=1, heads=4, mlp_ratio=.5, action_dim=2,
                                                conditioning="adaln", context_frames=3)
    for c in WM:
        info = ac_baseline.describe(c)
        assert info["init"] == "scratch" and info["checkpoint_url"] is None and "CC BY-NC 4.0" in info["source"]
        assert not ac_baseline.is_official_scale(ac_baseline.spec_for(c))
    assert ac_baseline.describe("dino_wm_small")["upstream_pred_type"] == "dino_wm"
    assert ac_baseline.describe("jepa_wm_small")["upstream_pred_type"] == "AdaLN"


def test_ac_snapshots_descriptions_unchanged():
    """AC 两个条件的规格快照和描述与加字段之前逐键相同（预登记里的 ac_baselines 不变）。"""
    assert ac_baseline.spec_snapshot(ac_baseline.SMALL) == dict(encoder_dim=1408, width=64, depth=1, heads=4, mlp_ratio=3.5,
                                                                 action_dim=2)
    assert ac_baseline.spec_snapshot(ac_baseline.OFFICIAL) == dict(encoder_dim=1408, width=1024, depth=24, heads=16,
                                                                    mlp_ratio=4.0, action_dim=2)
    assert ac_baseline.describe("vjepa2_ac_small") == dict(condition="vjepa2_ac_small", spec=ac_baseline.spec_snapshot(ac_baseline.SMALL),
                                                           init="scratch", checkpoint_url=None)


def _stats(model):
    return [float(sum(p.double().sum() for p in model.parameters())), float(sum(p.double().abs().sum() for p in model.parameters()))]


def test_ac_small_unchanged(monkeypatch):
    """参照值在 full-model 1d74a80 上算（v7 协议、2 线程 CPU，和本文件同一写法）：AC-small 的构造、前向（含半整数时长）、
    一步训练都和改动之前相同。用相对 1e-6 的容差而不是逐位哈希，免得换 CPU 指令集时误报。"""
    monkeypatch.setenv("PO_JEPA_PROTOCOL", "v7")
    approx = lambda x: pytest.approx(x, rel=1e-6, abs=1e-9)   # noqa: E731
    torch.manual_seed(0)
    m = ac_baseline.ACPredictor(ac_baseline.SMALL).eval()
    assert _stats(m) == approx([196.06571919972427, 3704.181673082444])
    g = torch.Generator().manual_seed(5)
    raw = torch.randn(1, 64, 1408, generator=g)
    with torch.no_grad():
        out = m(raw, d12_controls([[(0., 5., (.3, -.2))]]), torch.tensor([4.5], dtype=torch.float64), regular_grid(8, 8))
    assert float((out.double()*torch.arange(1408, dtype=torch.float64)).sum()) == approx(-81803.48774196918)
    assert float(out[0, 5, 7]) == approx(-0.6842649579048157)
    torch.manual_seed(1)
    m = ac_baseline.ACPredictor(ac_baseline.SMALL)
    o = ac_training.make_optimizer(m, "vjepa2_ac_small")
    clips = torch.randn(2, 9, 256, 1408, generator=g); acts = torch.rand(2, 8, 2, generator=g)-.5
    r, lr = ac_training.train_step("vjepa2_ac_small", m, o, clips, acts, _loss(), 3)
    assert _stats(m) == approx([198.08162289444925, 3694.631139710173])
    assert r.loss_values["total"] == approx(2.255608081817627) and r.loss_values["ac_rollout"] == approx(1.1278172731399536)
    assert "ac_rollout_prefix" not in r.loss_values


@pytest.mark.parametrize("bad", [dict(conditioning="mlp"), dict(conditioning="feature"),
                                 dict(conditioning="feature", action_embed_dim=10, head_dim=16, mlp_hidden=176),
                                 dict(conditioning="adaln"), dict(conditioning="adaln", context_frames=3, state_input="true_state", action_dim=7),
                                 dict(action_embed_dim=10), dict(context_frames=-1)])
def test_spec_rejects_inconsistent_fields(bad):
    with pytest.raises(ValueError):
        ac_baseline.ACSpec(width=64, depth=1, heads=4, **bad)


# ---------------------------------------------------------------- 结构

def test_cores_have_no_state_token_and_expected_modules():
    d, j = _model("dino_wm_small"), _model("jepa_wm_small")
    assert not hasattr(d, "state_token") and not hasattr(j, "state_token")
    assert isinstance(d.core, wm_baselines.DinoWMCore) and isinstance(j.core, wm_baselines.AdaLNCore)
    assert d.core.transformer.layers[0][0].to_qkv.weight.shape == (3*4*16, 74)
    assert d.core.transformer.layers[0][1].net[1].weight.shape == (176, 74)
    assert j.core.predictor_blocks[0].adaLN_modulation[1].weight.shape == (6*64, 64)
    assert j.core.predictor_blocks[0].mlp.fc1.weight.shape == (32, 64)
    assert not list(d.buffers()) and not list(j.buffers())          # sincos、掩码不进 state_dict


def test_adaln_initialisation_follows_upstream():
    torch.manual_seed(7)
    core = _model("jepa_wm_small", 7).core
    block = core.predictor_blocks[0]
    w = block.adaLN_modulation[1].weight
    assert torch.equal(block.adaLN_modulation[1].bias, torch.zeros_like(block.adaLN_modulation[1].bias))
    assert w.abs().max() <= 2.0 and .15 < w.std() < .25                  # 截断正态 std 0.02 × 10
    assert .015 < core.predictor_embed.weight.std() < .025 and torch.equal(core.predictor_embed.bias, torch.zeros(64))
    assert torch.equal(block.norm1.weight, torch.ones(64)) and block.norm1.eps == 1e-6
    big = _model("jepa_wm_small", 7).core.predictor_blocks[0].attn.proj.weight
    assert .015/math.sqrt(2) < big.std() < .025/math.sqrt(2)             # 第 1 层除以 √2


def test_sincos_positions_scale_to_16_grid():
    full = wm_baselines.sincos_3d(1, 16, 16, 74)
    coarse = wm_baselines.sincos_3d(1, 8, 8, 74)
    assert full.shape == (256, 74) and torch.equal(full[:, 72:], torch.zeros(256, 2))
    # 8×8 的第 (r, c) 个 token 与 16×16 的 (2r, 2c) 同位
    for r, c in ((0, 0), (3, 5), (7, 7)):
        assert torch.allclose(coarse[r*8+c], full[2*r*16+2*c])
    # 轴：token 顺序是 (帧, 行, 列)，第二段只随行变、第三段只随列变
    part = 24
    row3col5 = full[3*16+5]
    assert torch.allclose(row3col5[part:2*part], wm_baselines.sincos_1d(torch.tensor([3.]), part)[0])
    assert torch.allclose(row3col5[2*part:3*part], wm_baselines.sincos_1d(torch.tensor([5.]), part)[0])
    assert torch.allclose(row3col5[:part], wm_baselines.sincos_1d(torch.tensor([0.]), part)[0])
    two = wm_baselines.sincos_3d(2, 8, 8, 74)
    assert torch.equal(two[:64], coarse) and not torch.equal(two[64:], coarse)
    assert torch.allclose(wm_baselines.sincos_1d(torch.tensor([0.]), 4), torch.tensor([[0., 0., 1., 1.]], dtype=torch.float64))


def _independent_dino_transformer(tf, x, frames, tokens):
    """独立写的 DINO-WM 层：x + Attn(LN x)（块因果、无 qkv 偏置），+ FF(LN x)，末尾 LN。"""
    mask = torch.zeros(frames*tokens, frames*tokens, dtype=torch.bool)
    for i in range(frames):
        mask[i*tokens:(i+1)*tokens, :(i+1)*tokens] = True
    for attn, ff in tf.layers:
        y = F.layer_norm(x, (x.shape[-1],), attn.norm.weight, attn.norm.bias)
        q, k, v = (y @ attn.to_qkv.weight.T).split(attn.heads*attn.dim_head, -1)
        heads = lambda t: t.reshape(*t.shape[:2], attn.heads, attn.dim_head).permute(0, 2, 1, 3)   # noqa: E731
        q, k, v = heads(q), heads(k), heads(v)
        s = (q @ k.transpose(-1, -2))/math.sqrt(attn.dim_head)
        s = s.masked_fill(~mask, float("-inf")).softmax(-1)
        o = (s @ v).permute(0, 2, 1, 3).reshape(*x.shape[:2], -1)
        x = x+o @ attn.to_out[0].weight.T+attn.to_out[0].bias
        ln, fc1, fc2 = ff.net[0], ff.net[1], ff.net[4]
        x = x+F.linear(F.gelu(F.linear(F.layer_norm(x, (x.shape[-1],), ln.weight, ln.bias), fc1.weight, fc1.bias)), fc2.weight, fc2.bias)
    return F.layer_norm(x, (x.shape[-1],), tf.norm.weight, tf.norm.bias)


def test_dino_wm_core_matches_independent_formula():
    model = _model("dino_wm_small", 3).eval()
    core = model.core
    g = torch.Generator().manual_seed(11)
    context = torch.randn(2, 3, 64, 1408, generator=g); actions = torch.randn(2, 3, 2, generator=g)
    with torch.no_grad():
        got = core.predict(context, actions, 8, 8)
        x = torch.cat([context @ core.predictor_embed.weight.T+core.predictor_embed.bias,
                       (actions @ core.action_encoder.weight.T+core.action_encoder.bias)[:, :, None].expand(2, 3, 64, 10)], -1)
        x = x.reshape(2, 192, 74)+wm_baselines.sincos_3d(3, 8, 8, 74).float()
        y = _independent_dino_transformer(core.transformer, x, 3, 64).reshape(2, 3, 64, 74)[..., :64]
        want = y @ core.predictor_proj.weight.T+core.predictor_proj.bias
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_adaln_block_reduces_to_plain_block_and_identity():
    """调制为 0、gate 为 1 时 FWAdaLNBlock = x + Attn(LN x) + MLP(LN ·)；gate 为 0 时是恒等。用 8×8 网格（上游 grid_size² 只对 16×16 对）。"""
    torch.manual_seed(4)
    block = jepa_wms.FWAdaLNBlock(dim=64, num_heads=4, mlp_ratio=.5, qkv_bias=True, use_rope=True, grid_size=16,
                                  norm_layer=lambda d: torch.nn.LayerNorm(d, eps=1e-6)).eval()
    x = torch.randn(1, 2*64, 64); z = torch.randn(1, 2, 64)
    mask = wm_baselines.block_causal_mask(2, 64, x.device)
    lin = block.adaLN_modulation[1]
    with torch.no_grad():
        lin.weight.zero_(); lin.bias.zero_()
        assert torch.equal(block(x, z, attn_mask=mask, T=2, H_patches=8, W_patches=8), x)
        lin.bias[2*64:3*64] = 1.; lin.bias[5*64:] = 1.                     # gate_msa、gate_mlp
        got = block(x, z, attn_mask=mask, T=2, H_patches=8, W_patches=8)
        y = x+block.attn(block.norm1(x), attn_mask=mask, T=2, H=8, W=8)
        want = y+block.mlp(block.norm2(y))
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("condition", WM)
def test_block_causal_and_actions_only_affect_their_frame_onward(condition):
    model = _model(condition, 5).eval()
    g = torch.Generator().manual_seed(6)
    context = F.layer_norm(torch.randn(1, 3, 144, 1408, generator=g), (1408,)); actions = torch.randn(1, 3, 2, generator=g)
    with torch.no_grad():
        base = model.predict_frames(context, actions, 12, 12)
        later = context.clone(); later[:, 2] += 1.
        moved = model.predict_frames(later, actions, 12, 12)
        assert torch.equal(base[:, :2], moved[:, :2]) and not torch.allclose(base[:, 2], moved[:, 2])
        acted = actions.clone(); acted[:, 1] += 1.
        moved = model.predict_frames(context, acted, 12, 12)
        assert torch.equal(base[:, :1], moved[:, :1]) and (moved[:, 1:]-base[:, 1:]).abs().amax() > 1e-5
        assert torch.allclose(base.pow(2).mean(-1), torch.ones(1, 3, 144), atol=1e-3)   # 输出 LN（和 AC 同一空间）


@pytest.mark.parametrize("condition", WM)
def test_rollout_uses_last_three_frames(condition):
    model = _model(condition, 8).eval()
    g = torch.Generator().manual_seed(9)
    z0 = F.layer_norm(torch.randn(1, 64, 1408, generator=g), (1408,)); actions = torch.randn(1, 5, 2, generator=g)
    with torch.no_grad():
        preds = model.rollout(z0, actions, 8, 8)
        frames = [z0]
        for k in range(5):                                         # 独立写：最近 3 帧和它们的动作
            lo = max(0, k-2)
            frames.append(model.predict_frames(torch.stack(frames[lo:k+1], 1), actions[:, lo:k+1], 8, 8)[:, -1])
        torch.testing.assert_close(preds, torch.stack(frames[1:], 1), rtol=0, atol=0)
        # 第 0 个动作只影响前 3 步（第 3 步起窗口已不含第 0 帧，但第 1、2 帧是它影响过的预测）——改起点帧之后的帧才看得出窗口
        a2 = actions.clone(); a2[:, 0] += 5.
        moved = model.rollout(z0, a2, 8, 8)
        assert not torch.allclose(preds[:, 0], moved[:, 0])
    ac = _model("vjepa2_ac_small").spec
    assert ac.context_frames == 0                                   # AC 照旧看全部历史


@pytest.mark.parametrize("condition", WM)
def test_forward_all_grids_and_half_integer(condition):
    model = _model(condition, 2).eval()
    for size in (8, 12, 16):
        raw = torch.randn(2, size*size, 1408)
        controls = d12_controls([[(0., 4., (.1, .2))], [(0., 4., (-.3, .1))]])
        with torch.no_grad():
            out = model(raw, controls, torch.tensor([4., 3.5], dtype=torch.float64), regular_grid(size, size))
            assert out.shape == raw.shape and torch.isfinite(out).all()
            z0 = F.layer_norm(raw, (1408,))
            steps = model.rollout(z0, ac_baseline.tick_actions(controls, 4), size, size)
            torch.testing.assert_close(out[0], steps[0, 3]); torch.testing.assert_close(out[1], .5*(steps[1, 2]+steps[1, 3]))
    with pytest.raises(ValueError):
        model.predict_frames(torch.zeros(1, 1, 64, 1408), torch.zeros(1, 1, 2), 8, 8, states=torch.zeros(1, 1, 7))
    with pytest.raises(ValueError):
        model.load_official({})


# ---------------------------------------------------------------- 训练

def _clips(seed=12):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 4, 256, 1408, generator=g), torch.rand(2, 3, 2, generator=g)-.5


def _grads(model, loss):
    model.zero_grad(set_to_none=True); loss.backward()
    return [p.grad.clone() for p in model.parameters()]


def test_dino_wm_loss_is_half_teacher_forcing_mse():
    model = _model("dino_wm_small", 13)
    clips, actions = _clips()
    loss, parts = ac_training._wm_losses("dino_wm_small", model, clips, actions, 0)
    h = F.layer_norm(clips, (1408,))
    want = ((model.predict_frames(h[:, :3], actions, 16, 16)-h[:, 1:])**2).mean()
    torch.testing.assert_close(loss, want/2)
    assert set(parts) == {"ac_teacher_forcing"} and parts["ac_teacher_forcing"] == pytest.approx(want.item(), rel=1e-6)


@pytest.mark.parametrize("completed", [0, 1, 2, 3, 4, 5])
def test_jepa_wm_loss_teacher_forcing_plus_stop_gradient_rollout(completed):
    model = _model("jepa_wm_small", 14)
    clips, actions = _clips()
    t = ac_training.rollout_prefix(completed)
    loss, parts = ac_training._wm_losses("jepa_wm_small", model, clips, actions, completed)
    got = _grads(model, loss)
    h = F.layer_norm(clips, (1408,))
    z = model.predict_frames(h[:, :3], actions, 16, 16)
    tf = ((z-h[:, 1:])**2).mean()
    ctx = torch.cat([h[:, :t+1], z[:, t:t+1].detach()], 1)          # t+2 ≤ 3 帧，窗口截不掉
    nxt = model.predict_frames(ctx, actions[:, :t+2], 16, 16)[:, -1]
    roll = ((nxt-h[:, t+2])**2).mean()
    # 上游两级加权：teacher forcing / (rollout_steps+1)，滚动 / rollout_steps，rollout_steps = 2
    want = _grads(model, tf/3+roll/2)
    torch.testing.assert_close(loss.detach(), (tf/3+roll/2).detach())
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    assert parts["ac_rollout_prefix"] == t and parts["ac_rollout"] == pytest.approx(roll.item(), rel=1e-6)
    # 停梯度要紧：不停的话梯度不同
    z2 = model.predict_frames(h[:, :3], actions, 16, 16)
    leak = model.predict_frames(torch.cat([h[:, :t+1], z2[:, t:t+1]], 1), actions[:, :t+2], 16, 16)[:, -1]
    other = _grads(model, ((z2-h[:, 1:])**2).mean()/3+((leak-h[:, t+2])**2).mean()/2)
    assert any(not torch.allclose(a, b, rtol=1e-5, atol=1e-7) for a, b in zip(got, other))


def test_rollout_prefix_deterministic_uniform_and_seeded(monkeypatch):
    values = [ac_training.rollout_prefix(k) for k in range(400)]
    assert set(values) == {0, 1} and 150 < sum(values) < 250
    assert values == [ac_training.rollout_prefix(k) for k in range(400)]
    monkeypatch.setattr(checks, "model_seed", lambda: 1704)
    assert [ac_training.rollout_prefix(k) for k in range(400)] != values
    state = torch.get_rng_state()
    ac_training.rollout_prefix(3)
    assert torch.equal(state, torch.get_rng_state())


def test_optimizer_settings_per_condition():
    for c, lr, wd in (("dino_wm_small", 5e-4, 1e-7), ("jepa_wm_small", 5e-4, 1e-7), ("vjepa2_ac_small", 4.25e-4, .04)):
        m = _model(c)
        o = ac_training.make_optimizer(m, c)
        assert [g["weight_decay"] for g in o.param_groups] == [wd, 0.] and o.param_groups[0]["lr"] == lr
        assert sum(len(g["params"]) for g in o.param_groups) == len(list(m.parameters()))
        assert ac_training.clip_frames(c) == (9 if c.startswith("vjepa2") else 4)


@pytest.mark.parametrize("condition", WM)
def test_train_step_and_checkpoint_roundtrip(condition, tmp_path):
    model = _model(condition, 15)
    optimizer = ac_training.make_optimizer(model, condition)
    clips, actions = _clips()
    before = [p.detach().clone() for p in model.parameters()]
    result, lr = ac_training.train_step(condition, model, optimizer, clips, actions, _loss(), 0)
    assert result.completed_steps == 1 and lr == ac_training._lr(condition, _loss(), 0) and lr > 0
    assert all(not torch.equal(a, b) for a, b in zip(before, model.parameters()))      # 每个参数都参与训练
    keys = {"jepa", "total", "ac_teacher_forcing"} | ({"ac_rollout", "ac_rollout_prefix"} if condition == "jepa_wm_small" else set())
    assert keys <= set(result.loss_values) and ("ac_rollout" in result.loss_values) == (condition == "jepa_wm_small")
    with pytest.raises(ValueError):
        ac_training.train_step(condition, model, optimizer, torch.randn(2, 9, 256, 1408), torch.rand(2, 8, 2), _loss(), 1)
    path = tmp_path/"wm.pt"
    sha = ac_training.save_checkpoint(condition, path, model, optimizer, 1, dict(kind="explicit_prepared_batches", next_batch_index=1,
                                      query_generator_state=None), {"x": 1}, ac_baseline.describe(condition))
    other = _model(condition, 99)
    progress = ac_training.load_checkpoint(condition, path, sha, other, ac_training.make_optimizer(other, condition), {"x": 1})
    assert progress.completed_steps == 1
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), other.state_dict().values()))
    wrong = "jepa_wm_small" if condition == "dino_wm_small" else "dino_wm_small"
    third = _model(wrong)
    with pytest.raises(ValueError, match="mismatch"):
        ac_training.load_checkpoint(wrong, path, sha, third, ac_training.make_optimizer(third, wrong), {"x": 1})


def test_load_clips_takes_first_four_frames_of_the_same_start(monkeypatch, tmp_path):
    from po_jepa.runtime import ac_training as t
    from po_jepa.runtime import data
    raw = np.arange(33*2*3, dtype=np.float32).reshape(33, 2, 3)
    meta = dict(controls=dict(starts_ticks=[0, 10], ends_ticks=[10, 32], values=[[.1, .2], [.3, .4]]))
    monkeypatch.setattr(t.cache, "_read_complete_cache", lambda path: (raw, meta, None))

    rows = [data.WindowQuery(0, 8, 1.)]

    class Spec:
        @staticmethod
        def split_for(i): return "train"
    monkeypatch.setattr(data.datasets, "active", lambda: Spec)
    c9, a9 = t.load_clips(tmp_path, rows, "cpu")
    c4, a4 = t.load_clips(tmp_path, rows, "cpu", 4)
    assert c4.shape == (1, 4, 2, 3) and a4.shape == (1, 3, 2)
    assert torch.equal(c4, c9[:, :4]) and torch.equal(a4, a9[:, :3])
    assert torch.allclose(a4[0], torch.tensor([[.1, .2], [.1, .2], [.3, .4]]))
    with pytest.raises(ValueError):
        t.load_clips(tmp_path, rows, "cpu", 5)


# ---------------------------------------------------------------- 入口、任务表、汇总

def test_job_entry_points_know_the_conditions(monkeypatch):
    from po_jepa.runtime import comparison, evaluation, v7jobs
    assert set(WM) <= set(v7jobs.JOB_CONDITIONS) and set(WM) <= set(evaluation.CONDITIONS)
    assert all(comparison.is_ac(c) for c in WM)
    text = (ROOT/"run.py").read_text(encoding="utf-8")
    assert "'dino_wm_small','jepa_wm_small'" in text
    monkeypatch.setenv("PO_JEPA_PROTOCOL", "v7")
    sources = comparison.consumer_sources()
    assert {"src/po_jepa/wm_baselines.py", "src/po_jepa/third_party/jepa_wms.py"} <= set(sources)


def test_preregistration_records_wm_counts(monkeypatch):
    from po_jepa.runtime import comparison
    monkeypatch.setenv("PO_JEPA_PROTOCOL", "v7")
    counts = comparison._ac_parameter_counts()
    assert counts["dino_wm_small"] == 227_486 and counts["jepa_wm_small"] == 228_064
    assert comparison.parameter_match(PO_PARAMETERS, counts["jepa_wm_small"])["matched"]


def test_jobs_table_and_summary_lists():
    sys.path.insert(0, str(ROOT/"tools"))
    import make_v8_jobs
    import summarize_v7
    assert make_v8_jobs.CONDITIONS[-2:] == WM and summarize_v7.WM_CONDITIONS == WM
    out = subprocess.run([sys.executable, str(ROOT/"tools/make_v8_jobs.py"), "--stage", "full"], capture_output=True, text=True,
                         check=True).stdout
    for c in WM:
        lines = [x for x in out.splitlines() if x.startswith("python run.py run ") and f"--condition {c} " in x]
        assert len(lines) == 12*3 and all("--ac-checkpoint" not in x for x in lines)
    groups = {("d", 10000, 1703): {c: 1 for c in summarize_v7.V7_CONDITIONS}}
    assert summarize_v7.missing_jobs(groups) == []
    groups[("d", 10000, 1704)] = {**{c: 1 for c in summarize_v7.V7_CONDITIONS}, "jepa_wm_small": 1}
    missing = summarize_v7.missing_jobs(groups)
    assert [(m["seed"], m["condition"]) for m in missing] == [(1703, "jepa_wm_small")]


def test_run_jobs_priority():
    sys.path.insert(0, str(ROOT/"tools"))
    import run_jobs
    assert run_jobs.priority("python run.py run --condition jepa_wm_small --x") == 1
    assert run_jobs.priority("python run.py run --condition dino_wm_small --x") == 1


def test_vendored_file_states_license_and_source():
    text = (ROOT/"src/po_jepa/third_party/jepa_wms.py").read_text(encoding="utf-8")
    assert "CC BY-NC 4.0" in text and "13cf1d9" in text
    notice = (ROOT/"THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "jepa-wms" in notice and "CC BY-NC 4.0" in notice


# ---------------------------------------------------------------- 真实缓存格式上的一步（假编码器）

from test_s25c_long_horizon import real  # noqa: E402,F401（读取时把假编码器的缓存当作真实编码器的）


@pytest.fixture(scope="module")
def train_cache(tmp_path_factory):
    from test_s25c_long_horizon import _encode
    root = tmp_path_factory.mktemp("wm")
    r = _encode(root/"base", "", splits="train")
    assert r.returncode == 0, r.stderr
    return root/"base"


@pytest.mark.parametrize("condition", WM)
def test_observed_update_on_a_cache(condition, train_cache, real, monkeypatch):
    """comparison._observed_update_ac：从缓存读 4 帧片段，一步训练，predictor_proj 的读出钩子记 1 次（DINO-WM）或 2 次（JEPA-WM）。"""
    from po_jepa import datasets
    from po_jepa.runtime import comparison, data
    old = datasets.active()
    datasets._set_for_tests(datasets.parse("double_well:40"))
    try:
        monkeypatch.setenv("PO_JEPA_PROTOCOL", "v7")
        model = _model(condition, 21)
        model.ac_condition = condition
        optimizer = comparison.make_optimizer(model)
        rows = [data.WindowQuery(0, 3, 1.), data.WindowQuery(5, 20, 2.)]
        result, scales, lr = comparison._observed_update_ac(condition, model, optimizer, train_cache, rows, torch.device("cpu"),
                                                           _loss(), 0)
        assert result.completed_steps == 1 and lr > 0 and math.isfinite(result.loss_values["total"])
        assert len(scales) == (1 if condition == "dino_wm_small" else 2)
        clips, actions = ac_training.load_clips(train_cache, rows, "cpu", 4)
        full, _ = ac_training.load_clips(train_cache, rows, "cpu")
        assert torch.equal(clips, full[:, :4]) and clips.shape == (2, 4, 256, 1408) and actions.shape == (2, 3, 2)
    finally:
        datasets._set_for_tests(old)


def test_full_stage_can_schedule_only_some_conditions():
    """导师要求按方法分开跑：--stage full --conditions 只排这些条件的 D 段训练，A–C 段照常；补跑条件、重复、未知条件拒绝。"""
    tool = ROOT/"tools/make_v8_jobs.py"
    run = lambda *extra: subprocess.run([sys.executable, str(tool), "--stage", "full", *extra], capture_output=True, text=True)   # noqa: E731
    out = run("--conditions", "jepa_wm_small,dino_wm_small").stdout
    train = [x for x in out.splitlines() if x.startswith("python run.py run ")]
    assert len(train) == 2*12*3 and {x.split("--condition ")[1].split()[0] for x in train} == set(WM)
    assert train[0].split("--condition ")[1].split()[0] == "dino_wm_small"        # 按 CONDITIONS 的顺序
    assert any(x.startswith("python run.py calibrate ") for x in out.splitlines())
    everything = [x for x in run().stdout.splitlines() if x.startswith("python run.py run ")]
    assert len(everything) == 8*12*3
    for bad in ("koopman", "dino_wm_small,dino_wm_small", "nope", ""):
        r = run("--conditions", bad)
        assert r.returncode != 0 and not r.stdout.startswith("python")
