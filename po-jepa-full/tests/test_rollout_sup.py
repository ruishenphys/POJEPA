"""08 §6：po_jepa_rollout_sup——时间一致性项换成有真值的两段滚动损失。

L_roll = mean_B mean_(N,E) |LN(Φ(Φ(Z,u1,τ),u2,σ)) − LN(Z_target(τ+σ))|，L = L_JEPA(direct) + w(s)·(λ_r·L_roll + λ_s·L_space)。
全部是 CPU 上的小 POJEPA（test_addon_conditions.po）和 e06 合成批。"""

import copy
import math
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from po_jepa.losses import (ReducedLoss, jepa_distance, spatial_consistency_loss, spatial_predictions,
                            temporal_consistency_loss, temporal_predictions)
from po_jepa.rollout_supervision import rollout_supervision_loss
from po_jepa.runtime import comparison, training
from po_jepa.runtime.addon import calibration, conditions
from test_addon_conditions import _load, _save, numpy_state_equal, po, tensors_equal
from test_training import e06_case, e06_config, v4_loss_fields

LR, LS = .375, 2e-3     # 虚构的 λ_r、λ_s


def rollout_case(**changes):
    _, batch = e06_case()
    model = po(max_modes=20)
    model.train()
    return model, batch, e06_config(time_weight=LR, space_weight=LS, **changes)


def hand_rollout(composed, target, eps=1e-5):
    """手写 L_roll：两侧沿 E 无 affine LN，|差| 在 (N,E) 上取均值；逐样本 [B]，转 float64。"""
    a = F.layer_norm(composed, (composed.shape[-1],), eps=eps)
    b = F.layer_norm(target.detach(), (target.shape[-1],), eps=eps)
    return (a-b).abs().mean(dim=(1, 2)).to(torch.float64)


# ---------- 损失本身 ----------

def test_rollout_loss_is_ln_l1_of_composed_against_truth():
    model, batch, config = rollout_case()
    temporal = temporal_predictions(model, batch.raw_z, batch.controls, batch.tau, batch.sigma, batch.transfer.source)
    loss = rollout_supervision_loss(temporal, batch.raw_target, config=config)
    assert isinstance(loss, ReducedLoss) and loss.per_sample.dtype == loss.mean.dtype == torch.float64
    assert loss.per_sample.shape == (2,) and loss.mean.ndim == 0
    expected = hand_rollout(temporal.composed, batch.raw_target)
    assert torch.equal(loss.per_sample, expected) and loss.mean == expected.mean()
    # 用的是两段滚动 composed，不是长支 direct，也不是时间一致性
    assert not torch.equal(loss.per_sample, hand_rollout(temporal.direct, batch.raw_target))
    assert loss.mean != temporal_consistency_loss(temporal, config=config).mean
    # 和 JEPA 项同一个距离（同一个函数、同一个目标）：等于 jepa_distance(composed, target) 的 float64 版本
    assert torch.equal(loss.per_sample, jepa_distance(temporal.composed, batch.raw_target, config=config).per_sample.double())
    # 不除 (τσ)²：换一组 τ、σ 标签（只改记录、不改预测）损失不变
    relabeled = type(temporal)(temporal.direct, temporal.intermediate, temporal.composed, temporal.difference,
                               temporal.tau*2, temporal.sigma*3, temporal.input_scale)
    assert torch.equal(rollout_supervision_loss(relabeled, batch.raw_target, config=config).per_sample, loss.per_sample)


def test_rollout_loss_is_zero_on_its_own_prediction_and_flows_through_both_segments():
    model, batch, config = rollout_case()
    temporal = temporal_predictions(model, batch.raw_z, batch.controls, batch.tau, batch.sigma, batch.transfer.source)
    assert rollout_supervision_loss(temporal, temporal.composed.detach(), config=config).mean == 0.
    temporal.intermediate.retain_grad()
    target = batch.raw_target.clone().requires_grad_()
    loss = rollout_supervision_loss(temporal, target, config=config)
    loss.mean.backward()
    assert target.grad is None                                      # 真值停梯度
    assert temporal.intermediate.grad is not None and temporal.intermediate.grad.abs().sum() > 0   # 梯度经第一段传回
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_rollout_loss_contract_errors():
    model, batch, config = rollout_case()
    temporal = temporal_predictions(model, batch.raw_z, batch.controls, batch.tau, batch.sigma, batch.transfer.source)
    with pytest.raises(TypeError, match="^predictions must be TemporalPredictions$"):
        rollout_supervision_loss(object(), batch.raw_target, config=config)
    with pytest.raises(TypeError, match="^raw_target must be a tensor$"):
        rollout_supervision_loss(temporal, batch.raw_target.numpy(), config=config)
    with pytest.raises(ValueError, match="^rollout target must have exactly the composed prediction shape$"):
        rollout_supervision_loss(temporal, batch.raw_target[:1], config=config)
    with pytest.raises(ValueError, match="finite"):
        rollout_supervision_loss(temporal, batch.raw_target*float("nan"), config=config)


# ---------- 标签 ----------

@pytest.mark.parametrize("weights, ok", [((LR, LS), True), ((0., LS), False), ((LR, 0.), False), ((0., 0.), False)])
def test_loss_label_requires_two_positive_weights(weights, ok):
    config = e06_config(time_weight=weights[0], space_weight=weights[1])
    if ok:
        assert training._loss_config(config, "rollout_sup")["time_weight"] == LR
    else:
        with pytest.raises(ValueError, match="^condition rollout_sup requires a positive rollout"):
            training._loss_config(config, "rollout_sup")


# ---------- 单步 ----------

def test_step_losses_total_and_gradients_match_hand_computation(monkeypatch):
    """一次更新：日志六项等于手算；反传的梯度等于手写总损失 L_JEPA(direct) + λ_r·L_roll + λ_s·L_space 的梯度。"""
    model, batch, config = rollout_case()
    reference = copy.deepcopy(model)
    optimizer = comparison.make_optimizer(model)
    captured = {}
    real_step = training.scheduled_step
    monkeypatch.setattr(training, "scheduled_step",
                        lambda *a: captured.update(grads=[p.grad.clone() for p in model.parameters()]) or real_step(*a))
    result = comparison.condition_step("po_jepa_rollout_sup", model, optimizer, batch, config, 0)
    temporal = temporal_predictions(reference, batch.raw_z, batch.controls, batch.tau, batch.sigma, batch.transfer.source)
    jepa = jepa_distance(temporal.direct, batch.raw_target, config=config).mean
    roll = hand_rollout(temporal.composed, batch.raw_target).mean()
    space = spatial_consistency_loss(spatial_predictions(reference, batch.raw_z, batch.controls, batch.duration, batch.transfer),
                                     config=config).mean
    values = result.loss_values
    assert result.completed_steps == 1
    assert values["jepa"] == jepa.item() and values["time"] == roll.item() and values["space"] == space.item()
    assert values["weighted_time"] == LR*roll.item() and values["weighted_space"] == LS*space.item()
    assert values["total"] == pytest.approx(jepa.item()+LR*roll.item()+LS*space.item(), rel=1e-12)
    total = jepa.double()+LR*roll+LS*space
    total.backward()
    for (name, p), got in zip(reference.named_parameters(), captured["grads"]):
        torch.testing.assert_close(got, p.grad, rtol=1e-6, atol=1e-10, msg=name)


def test_step0_with_warmup_equals_full_and_later_steps_differ():
    """预热系数 0 的第 0 步只剩 JEPA：和 po_jepa_full 的第 0 步逐位相同；第 1 步起滚动项和时间一致性项不同，更新就不同。"""
    _, batch = e06_case()
    base = po(max_modes=20)
    fields = v4_loss_fields()
    config = e06_config(time_weight=LR, space_weight=LS, **fields)
    assert config.consistency_warmup_updates > 0 and training.consistency_warmup_scale(config, 0) == 0.
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    a.train(); b.train()
    oa, ob = comparison.make_optimizer(a), comparison.make_optimizer(b)
    ra = comparison.condition_step("po_jepa_full", a, oa, batch, config, 0)
    rb = comparison.condition_step("po_jepa_rollout_sup", b, ob, batch, config, 0)
    assert tensors_equal(a.state_dict(), b.state_dict())
    torch.testing.assert_close(oa.state_dict(), ob.state_dict(), rtol=0, atol=0)
    assert rb.loss_values["jepa"] == ra.loss_values["jepa"] and rb.loss_values["space"] == ra.loss_values["space"]
    assert rb.loss_values["weighted_time"] == 0. and rb.loss_values["time"] != ra.loss_values["time"]
    rb2 = comparison.condition_step("po_jepa_rollout_sup", b, ob, batch, config, 1)
    ra2 = comparison.condition_step("po_jepa_full", a, oa, batch, config, 1)
    assert rb2.loss_values["weighted_time"] > 0 and not tensors_equal(a.state_dict(), b.state_dict())
    assert ra2.loss_values["jepa"] == rb2.loss_values["jepa"]   # 第 1 步前的状态相同，JEPA 值也相同


def test_step_guards():
    model, batch, config = rollout_case()
    optimizer = comparison.make_optimizer(model)
    model.eval()
    with pytest.raises(ValueError, match="^caller must explicitly model.train"):
        comparison.condition_step("po_jepa_rollout_sup", model, optimizer, batch, config, 0)
    model.train()
    with pytest.raises(ValueError, match="^condition rollout_sup requires"):
        comparison.condition_step("po_jepa_rollout_sup", model, optimizer, batch, e06_config(time_weight=0., space_weight=LS), 0)
    with pytest.raises(TypeError, match="^condition/model type mismatch$"):
        comparison.condition_step("po_jepa_rollout_sup", po("fourier"), optimizer, batch, config, 0)
    bad = type(batch)(**{**batch.__dict__, "duration": batch.duration+1.})
    with pytest.raises(ValueError, match="^supervised duration must equal original float64 tau\\+sigma$"):
        comparison.condition_step("po_jepa_rollout_sup", model, optimizer, bad, config, 0)


# ---------- 梯度范数（标定和监控） ----------

def test_rollout_gradient_norms_hand_values_and_no_side_effects():
    model, batch, config = rollout_case()
    optimizer = comparison.make_optimizer(model)
    comparison.condition_step("po_jepa_rollout_sup", model, optimizer, batch, config, 0)
    state = copy.deepcopy(model.state_dict())
    grads = [p.grad.clone() for p in model.parameters()]
    moments = copy.deepcopy(optimizer.state_dict())
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    norms = calibration.rollout_gradient_norms(model, batch, loss_config=config)
    assert list(norms) == ["jepa", "time", "space", "grad_jepa", "grad_time", "grad_space"]
    assert tensors_equal(state, model.state_dict()) and all(torch.equal(x, p.grad) for x, p in zip(grads, model.parameters()))
    torch.testing.assert_close(moments, optimizer.state_dict(), rtol=0, atol=0)
    assert random.getstate() == rng[0] and numpy_state_equal(np.random.get_state(), rng[1]) and torch.equal(rng[2], torch.get_rng_state())

    def temporal(m):
        return temporal_predictions(m, batch.raw_z, batch.controls, batch.tau, batch.sigma, batch.transfer.source)
    terms = dict(jepa=lambda m: jepa_distance(temporal(m).direct, batch.raw_target, config=config).mean,
                 time=lambda m: hand_rollout(temporal(m).composed, batch.raw_target).mean(),
                 space=lambda m: spatial_consistency_loss(spatial_predictions(m, batch.raw_z, batch.controls, batch.duration,
                                                                              batch.transfer), config=config).mean)
    for name, loss in terms.items():
        other = copy.deepcopy(model)
        other.zero_grad(set_to_none=True)
        value = loss(other)
        value.backward()
        assert norms[name] == value.item()
        expected = math.sqrt(sum(float(p.grad.double().square().sum()) for p in other.parameters() if p.grad is not None))
        assert norms["grad_"+name] == pytest.approx(expected, rel=1e-12)
    # 不是时间一致性的梯度
    assert norms["grad_time"] != training.consistency_gradient_norms(model, batch, loss_config=config)["grad_time"]


def test_monitoring_checks_rollout_and_space():
    config = e06_config(time_weight=LR, space_weight=LS)
    assert comparison.monitored_consistency_terms("po_jepa_rollout_sup", config) == ("time", "space")
    assert conditions.expected_readouts("po_jepa_rollout_sup", po(max_modes=20)) == 10


# ---------- 检查点隔离 ----------

def test_rollout_and_full_checkpoints_do_not_cross(tmp_path):
    _, batch = e06_case()
    config = e06_config(time_weight=LR, space_weight=LS, **v4_loss_fields())
    rollout = _save("po_jepa_rollout_sup", po(), config, tmp_path/"ro.pt", batch)
    full = _save("po_jepa_full", po(), config, tmp_path/"full.pt", batch)
    payload = torch.load(tmp_path/"ro.pt", weights_only=True)
    assert payload["condition"] == "rollout_sup" and payload["loss_config"] == training._loss_config(config, "rollout_sup")
    message = "^checkpoint condition/configuration/source/order/runtime/mode mismatch$"
    with pytest.raises(ValueError, match=message):
        _load("po_jepa_full", po(), config, rollout, tmp_path)
    with pytest.raises(ValueError, match=message):
        _load("po_jepa_rollout_sup", po(), config, full, tmp_path)
    with pytest.raises(TypeError, match="^condition/model type mismatch$"):
        _load("po_jepa_rollout_sup", po("fourier"), config, rollout, tmp_path)
    assert _load("po_jepa_rollout_sup", po(), config, rollout, tmp_path).completed_steps == 0


def test_calibrate_addon_measures_rollout_not_temporal_consistency(monkeypatch):
    """calibrate-addon 给 po_jepa_rollout_sup 量的是 L_roll 的梯度（time 一格），预训练走 no_consistency 的单步（08 §6）。"""
    from po_jepa.runtime.addon import calibrate
    model, batch, config = rollout_case()
    norms = calibrate._gradient_norms("po_jepa_rollout_sup", model, batch, config)
    assert norms == calibration.rollout_gradient_norms(model, batch, loss_config=config)
    assert norms["time"] != training.consistency_gradient_norms(model, batch, loss_config=config)["time"]
    labels = []
    monkeypatch.setattr(training, "train_step", lambda *a, **k: labels.append(k["condition"]))
    calibrate._pretrain_step("po_jepa_rollout_sup", model, None, batch, config, 0)
    assert labels == ["no_consistency"]
    assert calibrate.CALIBRATED_TERMS["po_jepa_rollout_sup"] == (("time_weight", "grad_time"),)
