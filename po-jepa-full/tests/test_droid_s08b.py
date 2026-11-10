"""S08b：DROID 7 维模型、AC 官方输入、求谱后端（docs/design/09-droid-training.md §6；验收 14 §3 S08b）。

参照一律独立给出：v7 不变用改动前的树（e651582）上算好的哈希（`/mnt/project-files/po-jepa-s08b/golden.py`）；
AC true_state 的训练步对照一份照官方 app/vjepa_droid/train.py 写的参照（官方 `VisionTransformerPredictorAC.forward`，
`is_frame_causal=True` 的整段掩码）；状态积分对照逐步手调 `compute_new_pose`；物理动作对照手算 u·scale + offset。
"""

from __future__ import annotations

import copy
from dataclasses import asdict, fields
import hashlib
from functools import partial
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from po_jepa import ac_baseline, baselines, spectral, spectral_backends
from po_jepa.actions import ActionSegments, ControlSpec, split_action_segments
from po_jepa.droid import ac as dac
from po_jepa.droid.poses import compute_new_pose
from po_jepa.geometry import regular_grid
from po_jepa.runtime import ac_training, checks
from po_jepa.third_party import vjepa2_ac
from test_model import d12_controls
from test_v7_ac import _loss

# 改动前的树（e651582）上 golden.py 的输出
GOLDEN = {
    'ac_small_state': '0b1bf0e984209bfd0043fe73761ae1b5e68e1b9bbad6cc0d92de2557c2bec1c3',
    'ac_small_forward': 'fcda7916c4f941df18f5d1ae3b5b5a42853efd848be7ca1e60843c8208db830f',
    'tf_state': '13d7a082d5c17e48828c671ad84a0db5fbb044e71c25c2068abcf258c2f807f2',
    'tf_forward': '064aae5016db13470dbb227f696693f78ab2e0a9043a8894854cd3358e6dd34a',
    'tf_residual_state': '65fe6f0d58a6ae093ef78d5bbe4389471ec891c634e5083ae2e46e108b3faaf5',
    'ac_train_step': 'd3adf2a4061f698c012a57b31445197e9ada56f3a1b449958ec21524f111dfbc',
    'ac_train_losses': "[('ac_rollout', 1.1283080577850342), ('ac_teacher_forcing', 1.128455400466919), ('jepa', 1.128455400466919), ('space', None), ('time', None), ('total', 2.256763458251953), ('weighted_space', 0.0), ('weighted_time', 0.0)]",
    'eig_default': '80463d104b627fea1c9a2f192edf3271d97462af799a270c29ca73ab331aff0a',
}
SAMPLING = {"kind": "explicit_prepared_batches", "next_batch_index": 0, "query_generator_state": None}   # 09 §4.5
E = 32                                                   # 小编码维（只用于 CPU 测试）
SCALE = (0.01, 0.02, 0.03, 0.05, 0.04, 0.06, 0.5)
OFFSET = (0.001, -0.002, 0.0, 0.01, 0.0, -0.01, 0.1)     # DROID 的 offset 是 0；这里非零，核“+ offset”
SPEC7 = ControlSpec(7, 1.0, tuple(f"c{i}" for i in range(7)), ("unit",)*7, (-1.,)*7, (1.,)*7, SCALE, OFFSET,
                    "S08b fictional 7-d commands")


def _h(*ts) -> str:
    return hashlib.sha256(b"".join(t.detach().contiguous().numpy().tobytes() for t in ts)).hexdigest()


def _state_hash(m) -> str:
    return _h(*[v for _, v in sorted(m.state_dict().items())])


def controls7(rows, origins=None, ids=None) -> ActionSegments:
    """rows：每样本 [(start, end, 7 维模型单位值)]，周期单位、原点秒默认 10 + b。"""
    batch, width = len(rows), max(len(r) for r in rows)
    starts = torch.zeros(batch, width, dtype=torch.float64); ends = torch.zeros_like(starts)
    values = torch.zeros(batch, width, 7)
    for b, row in enumerate(rows):
        for j, (s, e, v) in enumerate(row):
            starts[b, j], ends[b, j], values[b, j] = s, e, torch.tensor(v, dtype=torch.float32)
    counts = torch.tensor([len(r) for r in rows])
    origin = torch.tensor(origins if origins is not None else [10.+b for b in range(batch)], dtype=torch.float64)
    return ActionSegments(starts, ends, values, counts, torch.arange(width)[None] < counts[:, None], origin,
                          torch.tensor([r[-1][1] for r in rows], dtype=torch.float64), SPEC7,
                          ids or tuple(f"s08b-{b}" for b in range(batch)))


def per_cycle(values) -> list:
    """[[7 维]] → 每周期一段。"""
    return [(float(j), float(j+1), v) for j, v in enumerate(values)]


def random_rows(g, batch, cycles):
    return [per_cycle((torch.rand(cycles, 7, generator=g)*1.6-.8).tolist()) for _ in range(batch)]


def ac7(state_input, *, depth=1, seed=0, encoder_dim=E):
    torch.manual_seed(seed)
    return ac_baseline.ACPredictor(ac_baseline.ACSpec(encoder_dim=encoder_dim, width=64, depth=depth, heads=4,
                                                      mlp_ratio=4.0, action_dim=7, state_input=state_input))


def start_states(g, batch):
    s = torch.rand(batch, 7, generator=g, dtype=torch.float64)
    s[:, :3] = s[:, :3]*.6-.3; s[:, 3:6] = s[:, 3:6]*2-1
    return s


# ------------------------------------------------------------------ v7 不变（09 §6.4、14 §3 S08b 第 1 条）

def test_v7_construction_forward_and_train_step_match_the_pre_s08b_tree():
    torch.set_num_threads(1)
    torch.manual_seed(1703); ac = ac_baseline.ACPredictor(ac_baseline.SMALL)
    assert _state_hash(ac) == GOLDEN["ac_small_state"]
    g = torch.Generator().manual_seed(5)
    raw = torch.randn(2, 64, 1408, generator=g)
    c = d12_controls([[(0., .5, (.1, .2)), (.5, 1.5, (-.3, .4))], [(0., 2., (-.3, .1))]])
    d = torch.tensor([1.5, 2.], dtype=torch.float64)
    with torch.no_grad():
        assert _h(ac.eval()(raw, c, d, regular_grid(8, 8))) == GOLDEN["ac_small_forward"]
    torch.manual_seed(1703); tf = baselines.TokenTransformer(baselines.TransformerSpec(feedforward=128))
    assert _state_hash(tf) == GOLDEN["tf_state"]
    with torch.no_grad():
        assert _h(tf.eval()(raw, c, d, regular_grid(8, 8))) == GOLDEN["tf_forward"]
    torch.manual_seed(1703)
    assert _state_hash(baselines.TokenTransformer(baselines.TransformerSpec(feedforward=128, residual=True))) == GOLDEN["tf_residual_state"]
    torch.manual_seed(1703); ac = ac_baseline.ACPredictor(ac_baseline.SMALL)
    opt = ac_training.make_optimizer(ac, "vjepa2_ac_small")
    clips = torch.randn(2, 9, 256, 1408, generator=g); acts = torch.randn(2, 8, 2, generator=g)*.3
    res, _ = ac_training.train_step("vjepa2_ac_small", ac, opt, clips, acts, _loss(), 0)
    assert _state_hash(ac) == GOLDEN["ac_train_step"]
    assert repr(sorted(res.loss_values.items())) == GOLDEN["ac_train_losses"]


def test_v7_spec_snapshots_are_byte_identical():
    """v7 的规格快照（检查点、describe、登记）和加字段之前的 asdict 逐字节相同；state_input 取默认值时不进快照。"""
    old = ("encoder_dim", "width", "depth", "heads", "mlp_ratio", "action_dim")
    for spec in (ac_baseline.OFFICIAL, ac_baseline.SMALL):
        snap = ac_baseline.spec_snapshot(spec)
        assert list(snap) == list(old) and snap == {k: getattr(spec, k) for k in old}
    assert ac_baseline.spec_snapshot(ac_baseline.OFFICIAL) == dict(encoder_dim=1408, width=1024, depth=24, heads=16,
                                                                   mlp_ratio=4.0, action_dim=2)
    assert ac_baseline.describe("vjepa2_ac_small")["spec"] == dict(encoder_dim=1408, width=64, depth=1, heads=4,
                                                                   mlp_ratio=3.5, action_dim=2)
    droid = ac_baseline.ACSpec(action_dim=7, state_input="true_state")
    assert ac_baseline.spec_snapshot(droid)["state_input"] == "true_state"
    assert "state_input" not in ac_baseline.spec_snapshot(ac_baseline.ACSpec(action_dim=7))
    assert [f.name for f in fields(baselines.TransformerSpec)] == ["encoder_dim", "action_dim", "width", "heads", "layers",
                                                                   "feedforward", "dropout", "layer_norm_epsilon", "residual"]
    assert asdict(baselines.TransformerSpec()) == dict(encoder_dim=1408, action_dim=2, width=64, heads=4, layers=1,
                                                       feedforward=128, dropout=0., layer_norm_epsilon=1e-5, residual=False)


def test_ac_checkpoint_spec_round_trip_uses_the_snapshot(tmp_path):
    """存档写 spec_snapshot；v7 规格的存档里没有 state_input，true_state 的有；换模式读档被拒。"""
    for state_input in ("learned_constant", "true_state"):
        model = ac7(state_input)
        opt = ac_training.make_optimizer(model, "vjepa2_ac_small")
        path = tmp_path/f"{state_input}.pt"
        sha = ac_training.save_checkpoint("vjepa2_ac_small", path, model, opt, 0, SAMPLING, dict(p=1), dict(i=1))
        payload = torch.load(path, weights_only=True)
        assert payload["spec"] == ac_baseline.spec_snapshot(model.spec)
        assert ("state_input" in payload["spec"]) == (state_input == "true_state")
        other = ac7("true_state" if state_input == "learned_constant" else "learned_constant")
        with pytest.raises(ValueError, match="spec"):
            ac_training.load_checkpoint("vjepa2_ac_small", path, sha, other,
                                        ac_training.make_optimizer(other, "vjepa2_ac_small"), dict(p=1))
        ac_training.load_checkpoint("vjepa2_ac_small", path, sha, model, opt, dict(p=1))


def test_spec_validation():
    for bad in (dict(action_dim=3), dict(state_input="x"), dict(state_input="true_state")):
        with pytest.raises(ValueError):
            ac_baseline.ACSpec(**bad)
    with pytest.raises(ValueError):
        baselines.TransformerSpec(action_dim=3)
    with pytest.raises(ValueError):
        baselines.IdentityPredictor(action_dim=3)


# ------------------------------------------------------------------ 7 维 Transformer、identity（09 §6.1）

def test_seven_dim_parameter_counts_match_09_table():
    """09 §2.3 的数：Transformer FFN 384 → 249,024；AC 缩小版 mlp 6.0 → true_state 249,344、learned_constant 248,896。"""
    tf = baselines.TokenTransformer(baselines.TransformerSpec(action_dim=7, width=64, heads=4, layers=1, feedforward=384,
                                                              residual=True))
    assert tf.action_embed.in_features == 9 and sum(p.numel() for p in tf.parameters()) == 249_024
    for state_input, count in (("true_state", 249_344), ("learned_constant", 248_896)):
        m = ac_baseline.ACPredictor(ac_baseline.ACSpec(width=64, depth=1, heads=4, mlp_ratio=6.0, action_dim=7,
                                                       state_input=state_input))
        assert ac_baseline.parameter_count(m) == count
        assert m.core.action_encoder.in_features == 7
        assert hasattr(m.core, "state_encoder") == (state_input == "true_state")
        assert hasattr(m, "state_token") == (state_input == "learned_constant")
        assert not hasattr(m.core, "extrinsics_encoder")


def test_seven_dim_transformer_identity_and_ac_run_forward_and_backward():
    """14 §3 S08b：7 维的 Transformer、identity、两种 AC 前向和反传，每个参数都有有限梯度。"""
    g = torch.Generator().manual_seed(1)
    grid = regular_grid(4, 4)
    raw = torch.randn(3, 16, E, generator=g)
    c = controls7(random_rows(g, 3, 2))
    d = torch.tensor([2., 1., 1.5], dtype=torch.float64)
    torch.manual_seed(0)
    tf = baselines.TokenTransformer(baselines.TransformerSpec(encoder_dim=E, action_dim=7, feedforward=32))
    ident = baselines.IdentityPredictor(encoder_dim=E, action_dim=7)
    assert ident(raw, c, d, grid) is raw
    for model, kw in ((tf, {}), (ac7("learned_constant"), {}), (ac7("true_state"), dict(start_state=start_states(g, 3)))):
        out = model(raw, c, d, grid, **kw)
        assert out.shape == raw.shape and torch.isfinite(out).all()
        out.square().mean().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    two = d12_controls([[(0., 2., (.1, .2))]]*3)
    for model in (tf, ident):
        with pytest.raises(ValueError, match="action_dim = 7"):
            model(raw, two, d, grid)
    with pytest.raises(ValueError, match="two control channels"):
        baselines.IdentityPredictor(encoder_dim=E)(raw, c, d, grid)


# ------------------------------------------------------------------ AC 官方输入（09 §6.2）

def _fake_official(spec: ac_baseline.ACSpec, seed=3) -> dict:
    """官方键格式的小假权重（含 action/state/extrinsics 三个编码器，带 module. 前缀）。"""
    torch.manual_seed(seed)
    core = vjepa2_ac.VisionTransformerPredictorAC(
        img_size=256, patch_size=16, num_frames=2, tubelet_size=2, embed_dim=spec.encoder_dim,
        predictor_embed_dim=spec.width, depth=spec.depth, num_heads=spec.heads, mlp_ratio=spec.mlp_ratio,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), is_frame_causal=False, use_rope=True,
        action_embed_dim=7, use_silu=False, wide_silu=True)
    return {"module."+k: v.clone() for k, v in core.state_dict().items()}


@pytest.mark.parametrize("action_dim,state_input", [(2, "learned_constant"), (7, "learned_constant"), (7, "true_state")])
def test_load_official_loads_and_replaces_the_right_keys(monkeypatch, action_dim, state_input):
    reduced = ac_baseline.ACSpec(encoder_dim=E, width=64, depth=2, heads=4, mlp_ratio=4.0)
    monkeypatch.setattr(ac_baseline, "OFFICIAL", reduced)
    spec = ac_baseline.ACSpec(encoder_dim=E, width=64, depth=2, heads=4, mlp_ratio=4.0, action_dim=action_dim,
                              state_input=state_input)
    weights = _fake_official(spec)
    torch.manual_seed(9)
    model = ac_baseline.ACPredictor(spec)
    before = {k: v.clone() for k, v in model.core.state_dict().items()}
    info = model.load_official(weights)
    expect = {"extrinsics_encoder"} | ({"action_encoder"} if action_dim == 2 else set()) \
        | ({"state_encoder"} if state_input == "learned_constant" else set())
    assert {k.split(".")[0] for k in info["replaced"]} == expect
    assert info["loaded"] == len(weights) - 2*len(expect)
    for k, v in model.core.state_dict().items():
        if k.split(".")[0] in expect:                     # 只可能是 2 维时新建的 Linear(2, 宽)：保持构造时的值
            assert action_dim == 2 and k.startswith("action_encoder") and torch.equal(v, before[k])
        else:
            assert torch.equal(v, weights["module."+k]), k
    if state_input == "learned_constant":
        assert torch.equal(model.state_token.detach(), weights["module.state_encoder.bias"])
    if action_dim == 2:
        assert info["note"] == "action_encoder 7→2 随机初始化；state token 用官方 state_encoder.bias 初始化"
    broken = dict(weights); broken.pop("module.predictor_proj.bias")
    with pytest.raises(RuntimeError, match="key mismatch"):
        ac_baseline.ACPredictor(spec).load_official(broken)
    if state_input == "true_state":                       # 官方输入模式下 state_encoder 是载入的键，缺了也要报
        broken = dict(weights); broken.pop("module.state_encoder.weight")
        with pytest.raises(RuntimeError, match="key mismatch"):
            ac_baseline.ACPredictor(spec).load_official(broken)
    with pytest.raises(ValueError, match="official-scale"):
        ac7(state_input if action_dim == 7 else "learned_constant", encoder_dim=E).load_official(weights) \
            if action_dim == 7 else ac_baseline.ACPredictor(ac_baseline.SMALL).load_official(weights)


def test_forward_uses_physical_actions_and_integrated_states(monkeypatch):
    """7 维前向：动作 token = 每周期 u·scale + offset（物理单位），状态 token 序列 = 起点状态按 compute_new_pose 逐周期积分。"""
    g = torch.Generator().manual_seed(2)
    model = ac7("true_state").eval()
    rows = random_rows(g, 2, 3)
    c = controls7(rows)
    s0 = start_states(g, 2)
    seen = {}
    original = model.predict_frames

    def spy(context, actions, h, w, states=None):
        seen["actions"], seen["states"] = actions.clone(), states.clone()
        return original(context, actions, h, w, states)
    monkeypatch.setattr(model, "predict_frames", spy)
    with torch.no_grad():
        model(torch.randn(2, 16, E, generator=g), c, torch.tensor([3., 3.], dtype=torch.float64), regular_grid(4, 4),
              start_state=s0)
    u = torch.tensor([[v for *_, v in r] for r in rows], dtype=torch.float32).double()
    phys = u*torch.tensor(SCALE, dtype=torch.float64)+torch.tensor(OFFSET, dtype=torch.float64)
    assert torch.equal(seen["actions"], phys.float())
    states = [s0[:, None]]
    for k in range(2):
        states.append(compute_new_pose(states[-1], phys[:, k:k+1]))
    assert torch.equal(seen["states"], torch.cat(states, 1).float())
    for bad in (None, s0.float(), s0[:1], torch.full_like(s0, float("nan"))):
        with pytest.raises(ValueError, match="start_state"):
            model(torch.randn(2, 16, E), c, torch.tensor([3., 3.], dtype=torch.float64), regular_grid(4, 4), start_state=bad)
    with pytest.raises(ValueError, match="no start_state"):
        ac7("learned_constant")(torch.randn(2, 16, E), c, torch.tensor([1., 1.], dtype=torch.float64), regular_grid(4, 4),
                                start_state=s0)


def test_non_integer_durations_interpolate_and_half_integers_equal_the_v7_average():
    g = torch.Generator().manual_seed(3)
    model = ac7("learned_constant").eval()
    raw = torch.randn(1, 16, E, generator=g)
    c = controls7(random_rows(g, 1, 3))
    grid = regular_grid(4, 4)
    run = lambda t: model(raw, c, torch.tensor([t], dtype=torch.float64), grid)
    with torch.no_grad():
        z0 = F.layer_norm(raw, (E,))
        p1, p2, p3 = run(1.), run(2.), run(3.)
        assert torch.equal(run(1.5), .5*(p1+p2))           # v7 半整数的写法
        assert torch.equal(run(2.25), .75*p2+.25*p3)
        assert torch.equal(run(0.3), (1-.3)*z0+.3*p1)
        assert torch.isfinite(run(1/3)).all()
    with pytest.raises(ValueError, match="positive"):
        model(raw, c, torch.tensor([0.], dtype=torch.float64), grid)


def _official_train_reference(model, clips, actions, states, loss_config, condition):
    """照官方 app/vjepa_droid/train.py（204698b）的 forward_target / forward_predictions / loss_fn 写的参照，
    用官方 `VisionTransformerPredictorAC.forward`（is_frame_causal=True 的整段掩码、[a, s, x] 交错），
    权重从 `model.core` 拷过来；优化器按官方 init_opt 的分组（偏置和一维参数不衰减）。"""
    spec = model.spec
    ref = vjepa2_ac.VisionTransformerPredictorAC(
        img_size=256, patch_size=16, num_frames=16, tubelet_size=2, embed_dim=spec.encoder_dim,
        predictor_embed_dim=spec.width, depth=spec.depth, num_heads=spec.heads, mlp_ratio=spec.mlp_ratio,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), is_frame_causal=True, use_rope=True,
        action_embed_dim=7, use_silu=False, wide_silu=True)
    missing, unexpected = ref.load_state_dict(model.core.state_dict(), strict=False)
    assert set(missing) == {"extrinsics_encoder.weight", "extrinsics_encoder.bias"} and not unexpected
    ref.extrinsics_encoder.requires_grad_(False)
    params = [(n, p) for n, p in ref.named_parameters() if not n.startswith("extrinsics_encoder")]
    opt = torch.optim.AdamW([dict(params=[p for n, p in params if "bias" not in n and len(p.shape) != 1], weight_decay=.04),
                             dict(params=[p for n, p in params if "bias" in n or len(p.shape) == 1], weight_decay=0.)],
                            lr=1., betas=(.9, .999), eps=1e-8)
    tpf = 256
    b, t, n, e = clips.shape
    h = F.layer_norm(clips, (e,)).flatten(1, 2)                                   # forward_target
    step = lambda z, a, s: F.layer_norm(ref(z, a, s), (e,))                       # _step_predictor
    z_tf = step(h[:, :-tpf], actions, states[:, :-1])
    z = torch.cat([h[:, :tpf], z_tf[:, :tpf]], 1)
    for k in range(1, 2):                                                         # auto_steps = 2
        z = torch.cat([z, step(z, actions[:, :k+1], states[:, :k+1])[:, -tpf:]], 1)
    z_ar = z[:, tpf:]
    jloss = torch.mean(torch.abs(z_tf - h[:, tpf:])**1.)/1.                       # loss_fn，loss_exp = 1
    sloss = torch.mean(torch.abs(z_ar - h[:, tpf:z_ar.size(1)+tpf])**1.)/1.
    loss = jloss+sloss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)
    for group in opt.param_groups:
        group["lr"] = ac_training._lr(condition, loss_config, 0)
    opt.step()
    return ref, jloss.item(), sloss.item()


def test_true_state_train_step_matches_an_official_train_py_reference():
    """两项损失逐位相同；更新后的参数除 predictor_embed.bias 外逐位相同。predictor_embed.bias 的梯度是对全部 token 求和，
    官方先把各帧摊平成 [B, T·N, E] 再过 Linear，我们按 [B, T, N, E] 过（v7 的写法，改了会动 v7 的数），求和顺序不同，
    梯度差约 1.5e-7（相对），只在这一项。"""
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(4)
    model = ac7("true_state", depth=2, seed=11)
    clips = torch.randn(2, 9, 256, E, generator=g)
    actions = torch.randn(2, 8, 7, generator=g)*.05
    states = start_states(g, 2*9).view(2, 9, 7).float()
    ref_model = copy.deepcopy(model)
    loss = _loss()
    ref, jl, sl = _official_train_reference(ref_model, clips, actions, states, loss, "vjepa2_ac_small")
    opt = ac_training.make_optimizer(model, "vjepa2_ac_small")
    res, _ = ac_training.train_step("vjepa2_ac_small", model, opt, clips, actions, loss, 0, states=states)
    assert res.loss_values["ac_teacher_forcing"] == jl and res.loss_values["ac_rollout"] == sl
    ours = model.core.state_dict()
    compared = 0
    for k, v in ref.state_dict().items():
        if k == "predictor_embed.bias":
            assert torch.allclose(ours[k], v, rtol=0, atol=1e-9)
        elif not k.startswith("extrinsics_encoder"):
            assert torch.equal(ours[k], v), k
            compared += 1
    assert compared == len(ours) - 1
    with pytest.raises(ValueError, match="states"):
        ac_training.train_step("vjepa2_ac_small", model, opt, clips, actions, loss, 1)
    with pytest.raises(ValueError, match="no states"):
        m = ac7("learned_constant")
        ac_training.train_step("vjepa2_ac_small", m, ac_training.make_optimizer(m, "vjepa2_ac_small"), clips, actions,
                               loss, 0, states=states)


# ------------------------------------------------------------------ 绑定起点状态和小步（09 §6.2、10 §8.4）

def _manual_advance(s0, phys, tau):
    """逐样本逐周期手调 compute_new_pose；余下 r 个周期用 r·a。"""
    out = []
    for b in range(s0.shape[0]):
        s = s0[b:b+1, None]
        whole = int(math.floor(tau[b])); r = tau[b]-whole
        for j in range(whole):
            s = compute_new_pose(s, phys[b:b+1, j:j+1])
        if r > 0:
            s = compute_new_pose(s, r*phys[b:b+1, whole:whole+1])
        out.append(s[:, 0])
    return torch.cat(out)


def test_bind_integrates_the_start_state_to_each_calls_origin(monkeypatch):
    g = torch.Generator().manual_seed(6)
    model = ac7("true_state").eval()
    rows = random_rows(g, 3, 4)
    ref = controls7([r[:d] for r, d in zip(rows, (1, 3, 4))], origins=[5., 7., 100.])
    s0 = start_states(g, 3)
    bound = dac.bind(model, s0, ref)
    u = torch.tensor([[v for *_, v in r] for r in rows], dtype=torch.float32).double()
    phys = u*torch.tensor(SCALE, dtype=torch.float64)+torch.tensor(OFFSET, dtype=torch.float64)
    assert torch.equal(bound.state_at(ref), s0)                                  # 窗口起点：τ = 0
    pair = split_action_segments(ref, tau=ref.duration*.5, sigma=ref.duration*.5)
    tau = (ref.duration*.5).tolist()                                             # 0.5、1.5、2（非整数起点）
    assert torch.equal(dac.offsets(ref, pair.second), torch.tensor(tau, dtype=torch.float64))
    assert torch.equal(bound.state_at(pair.second), _manual_advance(s0, phys, tau))
    seen = []
    original = model.forward
    monkeypatch.setattr(model, "forward", lambda *a, **kw: (seen.append(kw["start_state"].clone()), original(*a, **kw))[1])
    raw = torch.randn(3, 16, E, generator=g)
    with torch.no_grad():
        out = bound(raw, pair.second, pair.second.duration, regular_grid(4, 4))
        direct = original(raw, pair.second, pair.second.duration, regular_grid(4, 4), start_state=_manual_advance(s0, phys, tau))
    assert torch.equal(seen[0], _manual_advance(s0, phys, tau)) and torch.equal(out, direct)
    assert torch.equal(bound._start, s0)                                         # 包装不改起点
    late = controls7([r[:1] for r in rows], origins=[5.+1.5, 7., 100.])          # 第 0 个样本越过窗口末端
    with pytest.raises(ValueError, match="outside the reference"):
        bound.state_at(late)
    with pytest.raises(ValueError, match="source ids"):
        bound.state_at(controls7([r[:1] for r in rows], origins=[5., 7., 100.], ids=("a", "b", "c")))
    with pytest.raises(ValueError, match="true_state"):
        dac.bind(ac7("learned_constant"), s0, ref)
    with pytest.raises(ValueError, match="start_state"):
        dac.bind(model, s0.float(), ref)


def test_bind_runs_through_evaluate_style_split_rollouts():
    """评测的分割滚动（10 §3.1）：用绑定包装逐段调用，第 k 段的起点状态是前面各段动作积分的结果；
    和逐段手给积分状态调用模型逐位相同。"""
    from po_jepa.droid import evaluate as ev
    g = torch.Generator().manual_seed(8)
    model = ac7("true_state").eval()
    rows = random_rows(g, 2, 8)
    ref = controls7(rows, origins=[3., 40.])
    s0 = start_states(g, 2)
    raw = torch.randn(2, 16, E, generator=g)
    grid = regular_grid(4, 4)
    u = torch.tensor([[v for *_, v in r] for r in rows], dtype=torch.float32).double()
    phys = u*torch.tensor(SCALE, dtype=torch.float64)+torch.tensor(OFFSET, dtype=torch.float64)
    with torch.no_grad():
        got = ev.rollout(dac.bind(model, s0, ref), raw, ref, (2., 4., 2.), grid)
        x, t = raw, 0.
        for piece, tau in zip(ev.pieces(ref, (2., 4., 2.)), (2., 4., 2.)):
            x = model(x, piece, torch.full((2,), tau, dtype=torch.float64), grid,
                      start_state=_manual_advance(s0, phys, [t, t]))
            t += tau
    assert torch.equal(got, x)


def test_ac_steps_small_step_forward():
    """S11b 的 ac_steps：n 个小步，动作是物理位姿差（不换算），状态按 compute_new_pose 积分；learned_constant 不用状态。"""
    g = torch.Generator().manual_seed(9)
    grid = regular_grid(4, 4)
    raw = torch.randn(2, 16, E, generator=g)
    a = (torch.rand(2, 3, 7, generator=g, dtype=torch.float64)-.5)*.02
    s0 = start_states(g, 2)
    for state_input in ("true_state", "learned_constant"):
        model = ac7(state_input).eval()
        with torch.no_grad():
            got = dac.ac_steps(model, raw, a, s0, grid)
            states = None
            if state_input == "true_state":
                s = [s0[:, None]]
                for k in range(2):
                    s.append(compute_new_pose(s[-1], a[:, k:k+1]))
                states = torch.cat(s, 1).float()
            want = model.rollout(F.layer_norm(raw, (E,)), a.float(), 4, 4, states)[:, -1]
        assert torch.equal(got, want)
    with pytest.raises(ValueError, match="small-step actions"):
        dac.ac_steps(model, raw, a.float(), s0, grid)
    with pytest.raises(ValueError, match="start_state"):
        dac.ac_steps(model, raw, a, s0.float(), grid)


# ------------------------------------------------------------------ 求谱后端（09 §6.3）

def _laplacians(g, batch=6, n=40):
    w = torch.rand(batch, n, n, generator=g, dtype=torch.float64)
    w = (w+w.transpose(1, 2))/2
    w.diagonal(dim1=1, dim2=2).fill_(1.)
    d = w.sum(-1).rsqrt()
    return torch.eye(n, dtype=torch.float64)[None]-d[:, :, None]*w*d[:, None, :]


def test_default_backend_is_the_v7_eigh_bit_for_bit():
    os_env = __import__("os").environ
    old = os_env.get("PO_JEPA_PROTOCOL"); os_env["PO_JEPA_PROTOCOL"] = "v7"
    try:
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(5)
        torch.randn(2, 64, 1408, generator=g); torch.randn(2, 9, 256, 1408, generator=g)   # 照 golden.py 的随机数消耗
        torch.randn(2, 8, 2, generator=g)
        x = torch.randn(3, 64, 1408, generator=g)
        model = checks.build_full_model(checks.load_protocol(), torch.device("cpu"))
    finally:
        if old is None: os_env.pop("PO_JEPA_PROTOCOL")
        else: os_env["PO_JEPA_PROTOCOL"] = old
    spec = model.spectrum
    assert spec.eig_backend == "cpu_batched" and "eig_backend" not in spec.state_dict() and "eig_backend" not in vars(spec)
    lap = spec.normalized_laplacian(spec.affinity(x, regular_grid(8, 8))).laplacian
    r = spec.lowest_eigenpairs(lap, k=16, model_device=torch.device("cpu"))
    dg = r.diagnostics
    assert _h(r.eigenvalues_cpu, r.basis_cpu, dg.residual_norms, dg.orthogonality_error, dg.cutoff_gap,
              dg.sign_pivot_indices) == GOLDEN["eig_default"]


def test_cpu_threads_agrees_with_cpu_batched_and_is_wired_into_lowest_eigenpairs():
    g = torch.Generator().manual_seed(12)
    lap = _laplacians(g)
    base = spectral_backends.eigh(lap, "cpu_batched")
    assert all(torch.equal(x, y) for x, y in zip(base, torch.linalg.eigh(lap, UPLO="L")))
    for threads in (1, 2, 4, 64):
        other = spectral_backends.eigh(lap, "cpu_threads", threads=threads)
        report = spectral_backends.agreement(base, other, 16)
        assert report["agree"] and report["compared_samples"] == 6, report
    spec = spectral.GraphSpectrum(spectral.GraphConfig(**{f.name: getattr(checks_config(), f.name)
                                                          for f in fields(spectral.GraphConfig)}))
    ref = spec.lowest_eigenpairs(lap, k=16, model_device=torch.device("cpu"))
    spec.eig_backend, spec.eig_threads = "cpu_threads", 3
    got = spec.lowest_eigenpairs(lap, k=16, model_device=torch.device("cpu"))
    assert (got.eigenvalues_cpu-ref.eigenvalues_cpu).abs().max() <= 1e-9
    assert torch.equal(got.diagnostics.sign_pivot_indices, ref.diagnostics.sign_pivot_indices)
    p = lambda v: v @ v.transpose(1, 2)
    assert torch.linalg.matrix_norm(p(got.basis_cpu)-p(ref.basis_cpu), dim=(-2, -1)).max() <= 1e-6
    spec.eig_backend = "nope"
    with pytest.raises(ValueError, match="eig_backend"):
        spec.lowest_eigenpairs(lap, k=16, model_device=torch.device("cpu"))
    spec.eig_backend = "cuda_float64"
    with pytest.raises(ValueError, match="CUDA"):
        spec.lowest_eigenpairs(lap, k=16, model_device=torch.device("cpu"))


def test_agreement_flags_a_wrong_eigenvalue_and_a_wrong_subspace():
    g = torch.Generator().manual_seed(13)
    lap = _laplacians(g, batch=2)
    v, w = spectral_backends.eigh(lap)
    bad_v = v.clone(); bad_v[1, 3] += 2e-9
    assert not spectral_backends.agreement((v, w), (bad_v, w), 16)["agree"]
    bad_w = w.clone(); bad_w[0, :, [15, 16]] = bad_w[0, :, [16, 15]]           # 第 16、17 个特征向量对调
    r = spectral_backends.agreement((v, w), (v, bad_w), 16)
    assert not r["agree"] and r["max_projector_diff"] > 1
    assert spectral_backends.agreement((v, w), (v, -w), 16)["agree"]            # 符号不影响投影


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda_float64 后端要 GPU（合并前在 GPU 机器上跑）")
def test_cuda_float64_agrees_with_cpu_batched():
    lap = _laplacians(torch.Generator().manual_seed(14))
    base = spectral_backends.eigh(lap)
    other = spectral_backends.eigh(lap, "cuda_float64", device=torch.device("cuda"))
    assert other[0].device.type == "cpu" and other[0].dtype == torch.float64
    assert spectral_backends.agreement(base, other, 16)["agree"]


def checks_config():
    """v7 协议的 GraphConfig（只为构造一个 GraphSpectrum）。"""
    import os
    old = os.environ.get("PO_JEPA_PROTOCOL"); os.environ["PO_JEPA_PROTOCOL"] = "v7"
    try:
        return checks.build_full_model(checks.load_protocol(), torch.device("cpu")).spectrum.config
    finally:
        if old is None: os.environ.pop("PO_JEPA_PROTOCOL")
        else: os.environ["PO_JEPA_PROTOCOL"] = old
