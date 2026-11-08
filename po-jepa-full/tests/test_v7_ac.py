"""v7 V-JEPA 2-AC 基线：动作 token、前向形状、半整数时长、训练一步、checkpoint 往返、残差 Transformer。"""
from dataclasses import fields

import pytest
import torch

from po_jepa import ac_baseline
from po_jepa.baselines import TokenTransformer, TransformerSpec
from po_jepa.geometry import regular_grid
from po_jepa.losses import LossConfig
from po_jepa.runtime import ac_training, checks
from test_model import d12_controls


def _loss():
    c = checks.load_protocol()["loss"]
    return LossConfig(**{f.name: c[f.name] for f in fields(LossConfig)})


def test_tick_actions_time_average():
    controls = d12_controls([[(0., .5, (.2, -.4)), (.5, 2., (.4, .0))]])
    a = ac_baseline.tick_actions(controls, 3)
    assert torch.allclose(a[0, 0], torch.tensor([.3, -.2]))
    assert torch.allclose(a[0, 1], torch.tensor([.4, 0.]))
    assert torch.equal(a[0, 2], torch.zeros(2))           # 超出查询时长记 0


def test_small_parameter_count_matches_po_jepa():
    model = ac_baseline.ACPredictor(ac_baseline.SMALL)
    assert ac_baseline.parameter_count(model) == 215_552
    po = sum(p.numel() for p in checks.build_full_model(checks.load_protocol(), torch.device("cpu")).parameters())
    assert abs(po-215_552) <= .01*po


def test_forward_shapes_and_half_integer_average():
    torch.manual_seed(0)
    model = ac_baseline.ACPredictor(ac_baseline.SMALL).eval()
    grid = regular_grid(8, 8)
    raw = torch.randn(2, 64, 1408)
    controls = d12_controls([[(0., 2., (.1, .2))], [(0., 2., (-.3, .1))]])
    with torch.no_grad():
        out = model(raw, controls, torch.tensor([2., 2.], dtype=torch.float64), grid)
        assert out.shape == raw.shape and torch.isfinite(out).all()
        half = model(raw[:1], d12_controls([[(0., 1.5, (.1, .2))]]), torch.tensor([1.5], dtype=torch.float64), grid)
        z0 = torch.nn.functional.layer_norm(raw[:1], (1408,))
        steps = model.rollout(z0, ac_baseline.tick_actions(d12_controls([[(0., 1.5, (.1, .2))]]), 2), 8, 8)
        assert torch.allclose(half, .5*(steps[:, 0]+steps[:, 1]), atol=1e-5)
    with pytest.raises(ValueError):
        model(raw[:1], d12_controls([[(0., 1.25, (.1, .2))]]), torch.tensor([1.25], dtype=torch.float64), grid)


def test_actions_change_prediction():
    torch.manual_seed(1)
    model = ac_baseline.ACPredictor(ac_baseline.SMALL).eval()
    raw = torch.randn(1, 256, 1408); grid = regular_grid(16, 16); d = torch.tensor([1.], dtype=torch.float64)
    with torch.no_grad():
        a = model(raw, d12_controls([[(0., 1., (.5, .5))]]), d, grid)
        b = model(raw, d12_controls([[(0., 1., (-.5, -.5))]]), d, grid)
    assert (a-b).abs().max() > 1e-4


def test_train_step_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(2)
    model = ac_baseline.ACPredictor(ac_baseline.SMALL)
    optimizer = ac_training.make_optimizer(model, "vjepa2_ac_small")
    clips = torch.randn(2, ac_training.CLIP_FRAMES, 256, 1408)
    actions = torch.rand(2, ac_training.CLIP_FRAMES-1, 2)-.5
    before = [p.detach().clone() for p in model.parameters()]
    result, lr = ac_training.train_step("vjepa2_ac_small", model, optimizer, clips, actions, _loss(), 0)
    assert result.completed_steps == 1 and lr > 0
    assert set(result.loss_values) >= {"jepa", "total", "ac_teacher_forcing", "ac_rollout"}
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    path = tmp_path/"ac.pt"
    sha = ac_training.save_checkpoint("vjepa2_ac_small", path, model, optimizer, 1, dict(kind="explicit_prepared_batches", next_batch_index=1, query_generator_state=None),
                                      {"x": 1}, ac_baseline.describe("vjepa2_ac_small"))
    other = ac_baseline.ACPredictor(ac_baseline.SMALL)
    progress = ac_training.load_checkpoint("vjepa2_ac_small", path, sha, other,
                                           ac_training.make_optimizer(other, "vjepa2_ac_small"), {"x": 1})
    assert progress.completed_steps == 1
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), other.state_dict().values()))
    with pytest.raises(ValueError, match="SHA"):
        ac_training.load_checkpoint("vjepa2_ac_small", path, "0"*64, other, ac_training.make_optimizer(other, "vjepa2_ac_small"), {"x": 1})


def test_official_spec_refuses_small_weights():
    with pytest.raises(ValueError):
        ac_baseline.ACPredictor(ac_baseline.SMALL).load_official({})


def test_residual_transformer_starts_at_identity_with_same_parameters():
    spec = dict(encoder_dim=1408, width=64, heads=4, layers=1, feedforward=128)
    torch.manual_seed(3); plain = TokenTransformer(TransformerSpec(**spec))
    torch.manual_seed(3); residual = TokenTransformer(TransformerSpec(**spec, residual=True))
    assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in residual.parameters())
    raw = torch.randn(1, 256, 1408); grid = regular_grid(16, 16)
    controls = d12_controls([[(0., 1., (.1, .1))]])
    out = residual(raw, controls, torch.tensor([1.], dtype=torch.float64), grid)
    assert torch.equal(out, raw)
    out.sum().backward()
    assert residual.p_out.weight.grad.abs().sum() > 0
