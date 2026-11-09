"""S10b：5 个新条件的分发（07 §2.4–§2.7，08）——名单、构造和初始权重、单步和读出次数、20 步续训、检查点读回和隔离、
评测分发、监控名单、补跑标定读取、登记核对、来源登记和入口。全部是 CPU 上的虚构小模型和合成批。"""

import copy
import importlib.util
import math
import random
from argparse import Namespace
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from po_jepa.baselines import TokenTransformer, TransformerSpec
from po_jepa.dynamics import DynamicsConfig, IntegratorConfig, ModalDynamics
from po_jepa.fourier import FourierConfig, FourierSpectrum
from po_jepa.geometry import build_spatial_transfer, regular_grid
from po_jepa.koopman import KoopmanConfig, KoopmanPredictor, build_koopman
from po_jepa.model import POJEPA, ProjectionConfig
from po_jepa.spectral import GraphConfig, GraphSpectrum
from po_jepa.runtime import checks, comparison, evaluation, training, v7jobs
from po_jepa.runtime import workflow as w
from po_jepa.runtime.addon import calibration, conditions
from test_evaluation import fictional_window
from test_training import e06_case, e06_config, v4_loss_fields

CPU = torch.device("cpu")
from po_jepa import datasets
ADDON = ("po_jepa_time_only", "po_jepa_space_only", "transformer_consistency", "fixed_fourier", "koopman")
KOOPMAN_MODES = 5   # E=3 时参数量 451 + 448K


def numpy_state_equal(a, b):
    """整份比较 numpy 的 MT19937 状态（键数组只每 624 个字才变，单比 [1] 测不出少恢复一次；S10b 审核 B）。"""
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:])


@pytest.fixture
def v7(monkeypatch, tmp_path):
    """和 test_v7_protocol 的夹具相同：v7 协议、van_der_pol:2000、标定文件写在临时目录。"""
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    old = datasets._set_for_tests(datasets.DatasetSpec("van_der_pol", 2000))
    monkeypatch.setattr(checks, "v7_calibration_path", lambda: tmp_path/"cal.yaml")
    w.write_v7_calibration(dict(sigma_z=1.0, time_weight=.03, space_weight=4.0, rho=1.0), "test")
    yield tmp_path/"cal.yaml"
    datasets._set_for_tests(old)


@pytest.fixture
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def po(spectrum="graph", max_modes=3):
    """和 test_losses.e05_real_case 相同的小 POJEPA（E=3、同一种子），但用协议的残差读出；spectrum 选图谱或固定 Fourier 基底。"""
    graph = (16, 1703, .25, 1., max_modes, 1e-10, 1e-8)
    module = FourierSpectrum(FourierConfig(*graph, basis="dct2")) if spectrum == "fourier" else GraphSpectrum(GraphConfig(*graph))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1705)
        model = POJEPA(ProjectionConfig(3, 64, readout="residual"), module,
            ModalDynamics(DynamicsConfig(2, 64, 64, 32, 16, 32, 32, 64, .1, 1e-3)),
            IntegratorConfig("exponential_euler", .25, 1e-12, 1e-4, 6))
    return model


def transformer():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1703)
        return TokenTransformer(TransformerSpec(encoder_dim=3, residual=True))


def koopman_model():
    return build_koopman(KoopmanConfig(modes=KOOPMAN_MODES, action_dim=2, encoder_dim=3), 1703, CPU)


def batch16(batch):
    """e06 合成批换成 16×16 网格（Koopman 的基底网格）；动作、时长、tau/sigma、来源不变。"""
    index = torch.arange(2*256*3, dtype=torch.float32).reshape(2, 256, 3)
    raw = .2*torch.sin(index/37)+.1*torch.cos(index/11)
    return replace(batch, raw_z=raw, raw_target=torch.roll(raw, 1, -1)*.8+.03,
                   transfer=build_spatial_transfer(regular_grid(16, 16), target_height=8, target_width=8))


def case(condition):
    """(模型, 合成批, 损失配置)：PO 类和 transformer_consistency 用 e06 的 4×6→2×3 批，koopman 用 16×16 批。"""
    _, batch = e06_case()
    model = dict(po_jepa_time_only=lambda: po(max_modes=20), po_jepa_space_only=lambda: po(max_modes=20),
                 fixed_fourier=lambda: po("fourier"), transformer_consistency=transformer, koopman=koopman_model)[condition]()
    weights = dict(po_jepa_time_only=(1e-3, 0.), po_jepa_space_only=(0., 1e-3), koopman=(0., 0.)).get(condition, (1e-3, 1e-3))
    if condition == "koopman":
        batch = batch16(batch)
    return model, batch, e06_config(time_weight=weights[0], space_weight=weights[1])


def tensors_equal(a, b):
    return list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)


# ---------- 名单、标签、入口 ----------

def test_condition_lists_labels_and_entry_points():
    assert conditions.ADDON_CONDITIONS == ADDON and comparison.ADDON_CONDITIONS is conditions.ADDON_CONDITIONS
    assert comparison.ALL_CONDITIONS == (*comparison.V7_CONDITIONS, *ADDON) and "identity" not in comparison.ALL_CONDITIONS
    assert comparison.V7_CONDITIONS == ("matched_transformer", "po_jepa_no_consistency", "po_jepa_full", "vjepa2_ac", "vjepa2_ac_small")
    assert conditions.LOSS_LABELS == dict(po_jepa_time_only="time_only", po_jepa_space_only="space_only",
                                          transformer_consistency="full", fixed_fourier="full", koopman="no_consistency")
    assert evaluation.CONDITIONS == ("matched_transformer", "po_jepa_no_consistency", "po_jepa_full", "identity",
                                     "vjepa2_ac", "vjepa2_ac_small", *ADDON)
    assert v7jobs.JOB_CONDITIONS == ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity",
                                     "vjepa2_ac_small", "vjepa2_ac", *ADDON)
    for name in ADDON:
        evaluation.EvaluationQuery(180, 0, 2., 16, "point").key(name)
    with pytest.raises(ValueError, match="^not an add-on condition$"):
        conditions.build("po_jepa_full", CPU)


@pytest.mark.parametrize("condition", ADDON)
def test_run_py_accepts_new_conditions(tmp_path, monkeypatch, condition):
    spec = importlib.util.spec_from_file_location("addon_run_entry", w.ROOT/"run.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    seen = []
    monkeypatch.setattr(entry.subprocess, "run", lambda command, **kw: seen.append(command) or Namespace(returncode=0))
    features = tmp_path/"features"
    features.mkdir()
    assert entry.main(["run", "--features-root", str(features), "--output", str(tmp_path/"out"), "--system", "double_well",
                       "--trajectories", "200", "--updates", "2000", "--condition", condition]) == 0
    assert seen[0][seen[0].index("--condition")+1] == condition
    with pytest.raises(SystemExit):
        entry.main(["run", "--features-root", str(features), "--output", str(tmp_path/"out2"), "--system", "double_well",
                    "--trajectories", "200", "--updates", "2000", "--condition", "fictional_condition"])


# ---------- 构造和初始权重（14 §3 S10“初始权重”） ----------

@pytest.mark.parametrize("protocol", ["v6", "v7"])
def test_build_condition_weights_and_initial_hashes(monkeypatch, request, protocol):
    monkeypatch.delenv(checks.PROTOCOL_ENV, raising=False)   # v6 一支不受外部环境变量影响
    if protocol == "v7":
        request.getfixturevalue("v7")
    monkeypatch.setattr(calibration, "load_addon_weights",
                        lambda condition: dict(fixed_fourier=dict(time_weight=.25, space_weight=.5),
                                               transformer_consistency=dict(time_weight=.125, space_weight=.0625))[condition])
    config = checks.load_protocol()
    before = torch.get_rng_state().clone()
    built = {name: comparison.build_condition(name, CPU) for name in ("po_jepa_full", "matched_transformer", *ADDON)}
    assert torch.equal(before, torch.get_rng_state())
    full_hash = v7jobs._model_hash(built["po_jepa_full"][0])
    for name in ("po_jepa_time_only", "po_jepa_space_only", "fixed_fourier"):
        assert v7jobs._model_hash(built[name][0]) == full_hash
    assert type(built["fixed_fourier"][0].spectrum) is FourierSpectrum
    assert type(built["po_jepa_time_only"][0].spectrum) is GraphSpectrum
    assert v7jobs._model_hash(built["transformer_consistency"][0]) == v7jobs._model_hash(built["matched_transformer"][0])
    assert built["transformer_consistency"][0].spec == built["matched_transformer"][0].spec
    assert built["transformer_consistency"][0].spec.residual is (protocol == "v7")
    model, loss = built["koopman"]
    modes, total = (103, 227_840) if protocol == "v7" else (76, 215_744)   # v7 是 token 写法（PO 227,617），v6 是模态写法（215,778）
    assert type(model) is KoopmanPredictor and model.config == KoopmanConfig(modes=modes, action_dim=2, encoder_dim=1408)
    assert comparison.parameter_counts(model)["total"] == total
    weights = {name: (loss.time_weight, loss.space_weight) for name, (_, loss) in built.items()}
    lt, ls = config["loss"]["time_weight"], config["loss"]["space_weight"]
    assert lt > 0 and ls > 0 and (protocol == "v6" or (lt, ls) == (.03, 4.0))
    assert weights["po_jepa_time_only"] == (lt, 0.) and weights["po_jepa_space_only"] == (0., ls)
    assert weights["fixed_fourier"] == (.25, .5) and weights["transformer_consistency"] == (.125, .0625)
    assert weights["koopman"] == (0., 0.)
    reference = built["po_jepa_full"][1]
    for name in ADDON:
        assert replace(built[name][1], time_weight=lt, space_weight=ls) == reference
    assert type(comparison.make_optimizer(model)) is torch.optim.AdamW


def test_fixed_fourier_uses_token_forcing_under_v7(monkeypatch, v7):
    """S00t 审核 B M4：v7 协议下 fixed_fourier 和 po_jepa_full 一样是 token 写法，参数名和顺序相同。"""
    monkeypatch.setattr(calibration, "load_addon_weights", lambda condition: dict(time_weight=.25, space_weight=.5))
    full, _ = comparison.build_condition("po_jepa_full", CPU)
    fourier, _ = comparison.build_condition("fixed_fourier", CPU)
    assert full.dynamics.config.forcing_space == "token" and fourier.dynamics.config.forcing_space == "token"
    assert [n for n, _ in fourier.named_parameters()] == [n for n, _ in full.named_parameters()]


def test_choose_architecture_uses_token_candidates(monkeypatch, request):
    """S00t 审核 B M5：v6 协议下选 FFN 128（215,680），v7 协议下按 token 写法的候选选 FFN 224（228,064）。"""
    monkeypatch.delenv(checks.PROTOCOL_ENV, raising=False)
    for protocol, feedforward, total in (("v6", 128, 215_680), ("v7", 224, 228_064)):
        if protocol == "v7":
            request.getfixturevalue("v7")
        registration = comparison.choose_architecture()
        assert registration["matched"] and registration["selected"]["feedforward"] == feedforward
        assert registration["transformer_parameters"] == total


def test_koopman_optimizer_contract():
    model = koopman_model()
    optimizer = comparison.make_optimizer(model)
    assert optimizer.param_groups[0]["params"] == list(model.parameters())
    for name, value in dict(lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0).items():
        assert optimizer.defaults[name] == value
    assert conditions._koopman_optimizer(model, optimizer) == list(model.named_parameters())
    with pytest.raises(ValueError, match="^Koopman AdamW coverage/order differs$"):
        conditions._koopman_optimizer(model, torch.optim.AdamW(list(model.parameters())[1:]))
    with pytest.raises(ValueError, match="^Koopman optimizer settings differ from fixed AdamW$"):
        conditions._koopman_optimizer(model, torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.))
    with pytest.raises(ValueError, match="^expected a KoopmanPredictor$"):
        conditions.make_koopman_optimizer(transformer())
    model.omega.requires_grad_(False)
    with pytest.raises(ValueError, match="^all Koopman parameters must be finite trainable model-device float32$"):
        conditions.make_koopman_optimizer(model)


# ---------- 单步、读出次数（07 §2.4、08 §3、§5.4） ----------

@pytest.mark.parametrize("condition, readouts, phi_calls", [
    ("po_jepa_time_only", 10, 5), ("po_jepa_space_only", 10, 5), ("fixed_fourier", 10, 5),
    ("transformer_consistency", 6, 6), ("koopman", 2, 1)])
def test_one_update_readouts_and_all_gradients(one_thread, condition, readouts, phi_calls):
    model, batch, config = case(condition)
    optimizer = comparison.make_optimizer(model)
    model.train()
    forwards = []
    handle = model.register_forward_hook(lambda *a: forwards.append(1))
    result, scales = comparison._observed_update(condition, model, optimizer, batch, config, 0)
    handle.remove()
    assert len(scales) == readouts == conditions.expected_readouts(condition, model)
    assert phi_calls == len(scales)//(1 if condition == "matched_transformer" else checks.readouts_per_phi(model))
    if condition in ("transformer_consistency", "koopman"):
        assert len(forwards) == dict(transformer_consistency=6, koopman=1)[condition]
    assert result.completed_steps == 1
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    losses = result.loss_values
    if condition == "koopman":
        assert losses["time"] is None and losses["space"] is None and losses["total"] == losses["jepa"]
        assert losses["weighted_time"] == 0. and losses["weighted_space"] == 0.
    else:
        assert all(math.isfinite(losses[k]) for k in ("jepa", "time", "space", "total"))
    if condition == "po_jepa_time_only":
        assert losses["weighted_space"] == 0. and losses["weighted_time"] > 0
    if condition == "po_jepa_space_only":
        assert losses["weighted_time"] == 0. and losses["weighted_space"] > 0


def test_observed_update_rejects_wrong_readout_count(monkeypatch):
    model, batch, config = case("koopman")
    optimizer = comparison.make_optimizer(model)
    model.train()
    monkeypatch.setattr(conditions, "expected_readouts", lambda condition, model: 3)
    with pytest.raises(ValueError, match="^condition update count/readouts/finite diagnostics mismatch$"):
        comparison._observed_update("koopman", model, optimizer, batch, config, 0)


@pytest.mark.parametrize("perturb", [False, True])
def test_transformer_consistency_step0_equals_matched_transformer(one_thread, perturb):
    """08 §3：预热系数 0 的第 0 步和 matched_transformer 的第 0 步逐位相同；第 1 步起一致性项有非零梯度、更新不同。"""
    _, batch = e06_case()
    base = transformer()
    if perturb:   # 登记初始化的 p_out 为 0，一致性项第 0 步本身就是 0；扰动后它们非零，只靠预热系数 0 消掉
        g = torch.Generator().manual_seed(9)
        with torch.no_grad():
            base.p_out.weight.copy_(torch.randn(base.p_out.weight.shape, generator=g)*.05)
            base.p_out.bias.copy_(torch.randn(base.p_out.bias.shape, generator=g)*.05)
    full = e06_config(**v4_loss_fields())
    none = e06_config(time_weight=0., space_weight=0., **v4_loss_fields())
    assert full.consistency_warmup_updates > 0 and training.consistency_warmup_scale(full, 0) == 0.
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    oa, ob = comparison.make_optimizer(a), comparison.make_optimizer(b)
    a.train(); b.train()
    ra = comparison.condition_step("matched_transformer", a, oa, batch, none, 0)
    rb = comparison.condition_step("transformer_consistency", b, ob, batch, full, 0)
    assert tensors_equal(a.state_dict(), b.state_dict())
    assert all(torch.equal(p.grad, q.grad) for p, q in zip(a.parameters(), b.parameters()))
    torch.testing.assert_close(oa.state_dict(), ob.state_dict(), rtol=0, atol=0)
    assert rb.loss_values["jepa"] == ra.loss_values["jepa"] and rb.grad_norm == ra.grad_norm
    assert rb.grad_max_abs == ra.grad_max_abs
    assert rb.loss_values["weighted_time"] == 0. and rb.loss_values["weighted_space"] == 0.
    assert (rb.loss_values["time"] > 0 and rb.loss_values["space"] > 0) is perturb
    norms = calibration.generic_gradient_norms(b, batch, loss_config=full)
    assert norms["grad_time"] > 0 and norms["grad_space"] > 0 and norms["grad_jepa"] > 0
    comparison.condition_step("matched_transformer", a, oa, batch, none, 1)
    comparison.condition_step("transformer_consistency", b, ob, batch, full, 1)
    assert not tensors_equal(a.state_dict(), b.state_dict())


def test_generic_gradient_norms_hand_values_and_no_side_effects(one_thread):
    model, batch, config = case("transformer_consistency")
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        model.p_out.weight.copy_(torch.randn(model.p_out.weight.shape, generator=g)*.05)
    optimizer = comparison.make_optimizer(model)
    model.train()
    comparison.condition_step("transformer_consistency", model, optimizer, batch, config, 0)
    state = copy.deepcopy(model.state_dict())
    grads = [p.grad.clone() for p in model.parameters()]
    moments = copy.deepcopy(optimizer.state_dict())
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    norms = calibration.generic_gradient_norms(model, batch, loss_config=config)
    assert list(norms) == ["jepa", "time", "space", "grad_jepa", "grad_time", "grad_space"]
    assert tensors_equal(state, model.state_dict())
    assert all(torch.equal(x, p.grad) for x, p in zip(grads, model.parameters()))
    torch.testing.assert_close(moments, optimizer.state_dict(), rtol=0, atol=0)
    assert random.getstate() == rng[0] and numpy_state_equal(np.random.get_state(), rng[1])
    assert torch.equal(rng[2], torch.get_rng_state())
    # 手算：同样的前向，各项单独 backward，再算全部参数梯度的 2-范数
    from po_jepa.generic_consistency import spatial_predictions, temporal_predictions
    from po_jepa.losses import jepa_distance, spatial_consistency_loss, temporal_consistency_loss
    terms = dict(
        jepa=lambda m: jepa_distance(m(batch.raw_z, batch.controls, batch.duration, batch.transfer.source), batch.raw_target, config=config).mean,
        time=lambda m: temporal_consistency_loss(temporal_predictions(m, batch.raw_z, batch.controls, batch.tau, batch.sigma, batch.transfer.source), config=config).mean,
        space=lambda m: spatial_consistency_loss(spatial_predictions(m, batch.raw_z, batch.controls, batch.duration, batch.transfer), config=config).mean)
    for name, loss in terms.items():
        other = copy.deepcopy(model)
        other.zero_grad(set_to_none=True)
        value = loss(other)
        value.backward()
        assert norms[name] == value.item()
        expected = math.sqrt(sum(float(p.grad.double().square().sum()) for p in other.parameters() if p.grad is not None))
        assert norms["grad_"+name] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("condition", ADDON)
def test_consistency_share_dispatch(monkeypatch, condition):
    model, batch, config = case(condition)
    monkeypatch.setattr(checks, "load_protocol", lambda: dict(monitoring=dict(consistency_interval=2)))
    seen = []
    fake = dict(jepa=1., time=2., space=3., grad_jepa=4., grad_time=8., grad_space=2.)
    monkeypatch.setattr(training, "consistency_gradient_norms", lambda *a, **k: seen.append("po") or fake)
    monkeypatch.setattr(calibration, "generic_gradient_norms", lambda *a, **k: seen.append("generic") or fake)
    assert comparison.consistency_share(condition, model, batch, config, 3) is None
    result = comparison.consistency_share(condition, model, batch, config, 2)
    if condition == "koopman":
        assert result is None and seen == []
        return
    assert seen == ["generic" if condition == "transformer_consistency" else "po"]
    assert result == dict(**fake, warmup_scale=1.,
                          share=dict(time=config.time_weight*8./4., space=config.space_weight*2./4.))


@pytest.mark.parametrize("condition", ADDON)
def test_step_guards(condition):
    model, batch, config = case(condition)
    optimizer = comparison.make_optimizer(model)
    model.eval()
    if condition in ("transformer_consistency", "koopman"):
        with pytest.raises(ValueError, match="^explicit train mode and enabled gradients required$"):
            comparison.condition_step(condition, model, optimizer, batch, config, 0)
    model.train()
    wrong = transformer() if condition != "transformer_consistency" else koopman_model()
    with pytest.raises(TypeError, match="^condition/model type mismatch$"):
        comparison.condition_step(condition, wrong, optimizer, batch, config, 0)
    with pytest.raises(ValueError):   # 损失标签和权重不符（各条件沿用 training._loss_config 的报错）
        comparison.condition_step(condition, model, optimizer, batch, e06_config(time_weight=0., space_weight=1e-3)
                                  if condition == "po_jepa_time_only" else e06_config(time_weight=1e-3, space_weight=0.), 0)


def test_new_step_errors(monkeypatch):
    model, batch, config = case("koopman")
    optimizer = comparison.make_optimizer(model)
    model.train()
    with torch.no_grad():
        model.p_out.weight.fill_(1e38)
        model.p_in.weight.fill_(1e38)
    with pytest.raises(ValueError, match="^nonfinite Koopman raw prediction$"):
        comparison.condition_step("koopman", model, optimizer, batch, config, 0)
    model, batch, config = case("koopman")
    optimizer = comparison.make_optimizer(model)
    model.train()
    monkeypatch.setattr(conditions, "jepa_distance", lambda *a, **k: Namespace(mean=torch.tensor(float("nan"), requires_grad=True)))
    with pytest.raises(ValueError, match="^nonfinite JEPA loss$"):
        comparison.condition_step("koopman", model, optimizer, batch, config, 0)
    monkeypatch.undo()
    model, batch, config = case("transformer_consistency")
    optimizer = comparison.make_optimizer(model)
    model.train()
    real = conditions.weighted_total_loss
    def broken(*a, **k):   # 三项损失各自有限非负时 weighted_total_loss 才返回；这里让加权和溢出
        value = real(*a, **k)
        return replace(value, total=value.total*float("inf"))
    monkeypatch.setattr(conditions, "weighted_total_loss", broken)
    with pytest.raises(ValueError, match="^all six training losses must be finite before backward$"):
        comparison.condition_step("transformer_consistency", model, optimizer, batch, config, 0)
    monkeypatch.undo()
    # _finish 的三个报错
    model, batch, config = case("koopman")
    optimizer = comparison.make_optimizer(model)
    parameters = list(model.named_parameters())
    with pytest.raises(ValueError, match="^required Koopman gradient missing or nonfinite$"):
        conditions._finish(parameters, optimizer, config, 0, {}, "Koopman")
    for p in model.parameters():
        p.grad = torch.full_like(p, float("nan")) if p is model.omega else torch.zeros_like(p)
    with pytest.raises(ValueError, match="^required Koopman gradient missing or nonfinite$"):
        conditions._finish(parameters, optimizer, config, 0, {}, "Koopman")
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    monkeypatch.setattr(training, "scheduled_step", lambda *a: model.omega.data.fill_(float("inf")))
    with pytest.raises(ValueError, match="^updated Koopman parameter nonfinite$"):
        conditions._finish(parameters, optimizer, config, 0, {}, "Koopman")


# ---------- 20 步、中断续训、检查点读回（14 §3 S10“5 个新条件”） ----------

def _protocol():
    return {"validation": {"interval_updates": 10}, "loss": asdict(e06_config()),
            "monitoring": {"consistency_interval": 5, "early_check_updates": 10**6, "min_share": .005},
            "calibration": {"rho": .05}}


def _registration(condition, model, sequence):
    count = comparison.parameter_counts(model)["total"]
    return dict(parameter_match=comparison.parameter_match(count, count), query_sequence_sha256=comparison.sequence_sha(sequence),
                consumer_sha256=comparison.consumer_sources(),
                comparison_config_sha256=comparison.data.cache.file_sha(comparison.ROOT/"configs/train/comparison.yaml"))


@pytest.mark.parametrize("condition", ADDON)
@pytest.mark.parametrize("stop_at", [10, 15])
def test_twenty_updates_resume_and_readback(one_thread, monkeypatch, tmp_path, condition, stop_at):
    base, batch, config = case(condition)
    sequence = [[comparison.data.WindowQuery(0, index, 1.)] for index in range(20)]
    state = torch.Generator().manual_seed(2703).get_state()
    registration = _registration(condition, base, sequence)
    monkeypatch.setattr(comparison.data, "load_training_batch", lambda *a, **k: batch)
    monkeypatch.setattr(comparison.data, "write_collections", lambda *a, **k: {})
    def validate(model, features, output, *a):
        output.mkdir()
        expected = [dict(trajectory_id=t, start_tick=s, duration_cycles=d) for t in range(160, 180)
                    for s in (0, 5, 10, 15, 20) for d in (1., 2., 4., 8.)]
        for offset in range(0, 400, 4):
            w.cache._write_json(output/f"{offset:04d}.json",
                                dict(queries=expected[offset:offset+4], values={"future_jepa_distance": [1.]*4}))
        return dict(mean=1., windows=400, metric="future_jepa_distance")
    monkeypatch.setattr(comparison, "validate_condition", validate)
    monkeypatch.setattr(checks, "load_protocol", _protocol)
    original = comparison.condition_step
    active, attempted = {"fail": False}, []
    def step(*a, **k):
        completed = a[-1]
        attempted.append(completed)
        random.random(); np.random.random(); torch.rand(1)
        if active["fail"] and completed == stop_at:
            raise RuntimeError("fictional interruption")
        return original(*a, **k)
    monkeypatch.setattr(comparison, "condition_step", step)
    def run(output, source=None):
        model = copy.deepcopy(base)
        optimizer = comparison.make_optimizer(model)
        kwargs = dict(sequence=sequence, generator_state=state, registration=registration, features=tmp_path/"absent",
                      output=output, device=CPU, loss_config=config, validation_interval=10)
        if source:
            kwargs["resume_from"] = source
        report = comparison.run_prepared_condition(condition, model, optimizer, **kwargs)
        return model, optimizer, report
    random.seed(31); np.random.seed(31); torch.manual_seed(31)
    reference, reference_optimizer, report = run(tmp_path/"continuous")
    reference_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    assert report["completed_updates"] == 20 and [r["step"] for r in report["validations"]] == [0, 10, 20]
    assert not tensors_equal(reference.state_dict(), base.state_dict())
    record = w.read(tmp_path/"continuous/updates/0005.json")
    assert record["condition"] == condition
    assert record["phi_calls"] == dict(transformer_consistency=6, koopman=1).get(condition, 5)
    assert len(record["raw_predictions"]) == conditions.expected_readouts(condition, base)
    share = record["consistency_gradient"]
    if condition == "koopman":
        assert share is None and record["losses"]["time"] is None
    else:
        assert share["grad_jepa"] > 0 and share["warmup_scale"] == 1.
        assert share["share"] == {k: getattr(config, k+"_weight")*share["grad_"+k]/share["grad_jepa"] for k in ("time", "space")}
    assert w.read(tmp_path/"continuous/updates/0004.json")["consistency_gradient"] is None
    terms = comparison.monitored_consistency_terms(condition, config)
    assert (report["consistency_monitor"] is None) is (terms is None)
    if terms is not None:
        assert report["consistency_monitor"]["checked_terms"] == list(terms)
    # 中断在第 stop_at 步之后的那一步，续训从最后一个检查点（第 10 步）补到第 20 步
    random.seed(31); np.random.seed(31); torch.manual_seed(31)
    active["fail"] = True
    with pytest.raises(RuntimeError, match="interruption"):
        run(tmp_path/"interrupted")
    failure_sha = w.cache.file_sha(tmp_path/"interrupted/FAILED.json")
    active["fail"] = False
    attempted.clear()
    resumed, optimizer, report = run(tmp_path/"resumed", tmp_path/"interrupted")
    assert attempted == list(range(10, 20)) and report["completed_updates"] == 20
    assert tensors_equal(reference.state_dict(), resumed.state_dict())
    torch.testing.assert_close(reference_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0)
    assert random.getstate() == reference_rng[0] and numpy_state_equal(np.random.get_state(), reference_rng[1])
    assert torch.equal(torch.get_rng_state(), reference_rng[2])
    assert w.cache.file_sha(tmp_path/"interrupted/FAILED.json") == failure_sha
    for step_index in (10, 20):   # 文件字节含临时文件名，不比 SHA；比读出的模型和优化器状态
        a, b = (torch.load(tmp_path/f"{run_name}/checkpoints/step-{step_index:04d}.pt", weights_only=True)
                for run_name in ("continuous", "resumed"))
        assert tensors_equal(a["model_state"], b["model_state"]) and a["completed_steps"] == b["completed_steps"] == step_index
        torch.testing.assert_close(a["optimizer_state"], b["optimizer_state"], rtol=0, atol=0)
    # 第 20 步的检查点读回到新模型：model_state、AdamW 状态逐位相同，读回后是 eval 模式
    selected = w.read(tmp_path/"continuous/validations/selection-0020.json")
    fresh = copy.deepcopy(base)
    fresh_optimizer = comparison.make_optimizer(fresh)
    progress = evaluation.load_selected_checkpoint(condition, selected, model=fresh, optimizer=fresh_optimizer, config=config,
                                                   expected_provenance=selected["checkpoint_provenance"], root=tmp_path/"continuous")
    assert progress.completed_steps == 20 and not fresh.training
    assert tensors_equal(reference.state_dict(), fresh.state_dict())
    torch.testing.assert_close(reference_optimizer.state_dict(), fresh_optimizer.state_dict(), rtol=0, atol=0)
    if condition == "koopman" and stop_at == 10:
        payload = torch.load(tmp_path/"continuous/checkpoints/step-0020.pt", weights_only=True)
        assert set(payload) == conditions.KOOPMAN_FIELDS and payload["condition"] == "koopman"
        assert payload["model_config"] == asdict(base.config) and payload["model_config"]["basis_grid"] == (16, 16)
        assert payload["loss_config"]["time_weight"] == 0. and payload["optimizer_param_names"] == [[n for n, _ in base.named_parameters()]]
        checkpoint = tmp_path/"interrupted/checkpoints/step-0010.pt"
        checkpoint.write_bytes(checkpoint.read_bytes()+b"fictional corruption")
        attempted.clear()
        with pytest.raises(ValueError, match="SHA"):
            run(tmp_path/"rejected", tmp_path/"interrupted")
        assert attempted == []


# ---------- 检查点格式和隔离（14 §3 S10“检查点隔离”） ----------

def _sampling(step):
    return dict(kind="explicit_prepared_batches", next_batch_index=step,
                query_generator_state=torch.Generator().manual_seed(2703).get_state())


def _save(condition, model, config, path, batch):
    optimizer = comparison.make_optimizer(model)
    sha = comparison.save_checkpoint(condition, path, model, optimizer, config, 0, _sampling(0), batch.provenance)
    provenance = comparison._provenance(batch.provenance)
    return dict(step=0, mean=.2, checkpoint=path.name, checkpoint_sha256=sha, checkpoint_provenance=provenance)


def _load(condition, model, config, selected, root):
    return evaluation.load_selected_checkpoint(condition, selected, model=model, optimizer=comparison.make_optimizer(model),
                                               config=config, expected_provenance=selected["checkpoint_provenance"], root=root)


def test_fixed_fourier_and_full_checkpoints_do_not_cross(tmp_path):
    _, batch = e06_case()
    config = e06_config(**v4_loss_fields())
    fourier, graph = po("fourier"), po("graph")
    assert tensors_equal(fourier.state_dict(), graph.state_dict())
    ff = _save("fixed_fourier", fourier, config, tmp_path/"ff.pt", batch)
    full = _save("po_jepa_full", graph, config, tmp_path/"full.pt", batch)
    a = torch.load(tmp_path/"ff.pt", weights_only=True)["model_config"]
    b = torch.load(tmp_path/"full.pt", weights_only=True)["model_config"]
    assert a["graph"] == b["graph"] | dict(basis="dct2") and {k: v for k, v in a.items() if k != "graph"} == {k: v for k, v in b.items() if k != "graph"}
    message = "^checkpoint condition/configuration/source/order/runtime/mode mismatch$"
    with pytest.raises(ValueError, match=message):
        _load("po_jepa_full", po("graph"), config, ff, tmp_path)
    with pytest.raises(ValueError, match=message):
        _load("fixed_fourier", po("fourier"), config, full, tmp_path)
    with pytest.raises(TypeError, match="^condition/model type mismatch$"):
        _load("fixed_fourier", po("graph"), config, ff, tmp_path)
    assert _load("fixed_fourier", po("fourier"), config, ff, tmp_path).completed_steps == 0
    assert _load("po_jepa_full", po("graph"), config, full, tmp_path).completed_steps == 0


def test_transformer_consistency_and_matched_checkpoints_do_not_cross(tmp_path, monkeypatch):
    _, batch = e06_case()
    full = e06_config(**v4_loss_fields())
    none = e06_config(time_weight=0., space_weight=0., **v4_loss_fields())
    monkeypatch.setattr(checks, "load_protocol", lambda: dict(loss=asdict(none)))   # v7 读法拿协议的 loss 核
    tc = _save("transformer_consistency", transformer(), full, tmp_path/"tc.pt", batch)
    mt = _save("matched_transformer", transformer(), none, tmp_path/"mt.pt", batch)
    payload = torch.load(tmp_path/"tc.pt", weights_only=True)
    assert payload["schema"] == "F05_transformer_checkpoint_v1" and payload["loss_config"] == training._loss_config(full, "full")
    with pytest.raises(ValueError, match="^selected transformer JEPA configuration differs$"):
        _load("matched_transformer", transformer(), none, tc, tmp_path)
    model = transformer()
    with pytest.raises(ValueError, match="^Transformer checkpoint loss configuration mismatch$"):
        comparison.load_transformer_checkpoint(tmp_path/"tc.pt", tc["checkpoint_sha256"], model, comparison.make_optimizer(model),
                                               tc["checkpoint_provenance"])
    with pytest.raises(ValueError, match="^Transformer checkpoint loss configuration mismatch$"):
        _load("transformer_consistency", transformer(), full, mt, tmp_path)
    assert _load("transformer_consistency", transformer(), full, tc, tmp_path).completed_steps == 0
    assert _load("matched_transformer", transformer(), none, mt, tmp_path).completed_steps == 0


def test_koopman_checkpoint_errors(tmp_path):
    model, batch, config = case("koopman")
    selected = _save("koopman", model, config, tmp_path/"k.pt", batch)
    optimizer = comparison.make_optimizer(model)
    with pytest.raises(FileExistsError):
        comparison.save_checkpoint("koopman", tmp_path/"k.pt", model, optimizer, config, 0, _sampling(0), batch.provenance)
    with pytest.raises(ValueError, match="^checkpoint cursor must equal completed updates$"):
        comparison.save_checkpoint("koopman", tmp_path/"k2.pt", model, optimizer, config, 1, _sampling(0), batch.provenance)
    payload = torch.load(tmp_path/"k.pt", weights_only=True)
    def write(name, change):
        value = copy.deepcopy(payload)
        change(value)
        torch.save(value, tmp_path/name)
        return selected | dict(checkpoint=name, checkpoint_sha256=w.cache.file_sha(tmp_path/name))
    cases = [
        ("extra.pt", lambda p: p.update(extra=1), "^unsupported Koopman checkpoint schema/fields$"),
        ("schema.pt", lambda p: p.update(schema_version=2), "^unsupported Koopman checkpoint schema/fields$"),
        ("condition.pt", lambda p: p.update(condition="matched_transformer"), "^Koopman checkpoint condition/configuration/source/order/runtime/mode mismatch$"),
        ("modes.pt", lambda p: p["model_config"].update(modes=6), "^Koopman checkpoint condition/configuration/source/order/runtime/mode mismatch$"),
        ("loss.pt", lambda p: p["loss_config"].update(jepa_exponent=2.), "^Koopman checkpoint condition/configuration/source/order/runtime/mode mismatch$"),
        ("names.pt", lambda p: p.update(optimizer_param_names=[["omega"]]), "^Koopman checkpoint condition/configuration/source/order/runtime/mode mismatch$"),
        ("scheduler.pt", lambda p: p.update(scheduler_state={}), "^Koopman checkpoint condition/configuration/source/order/runtime/mode mismatch$"),
        ("cursor.pt", lambda p: p["sampling_state"].update(next_batch_index=1), "^checkpoint cursor mismatch$"),
    ]
    for name, change, message in cases:
        other = koopman_model()
        with pytest.raises(ValueError, match=message):
            _load("koopman", other, config, write(name, change), tmp_path)
        assert tensors_equal(other.state_dict(), koopman_model().state_dict())
    with pytest.raises(ValueError, match="^checkpoint SHA mismatch before loading$"):
        conditions.load_koopman_checkpoint(tmp_path/"k.pt", "0"*64, model, optimizer, config, selected["checkpoint_provenance"])
    with pytest.raises(ValueError, match="^not an add-on condition$"):
        conditions.save_checkpoint("matched_transformer", tmp_path/"x.pt", model, optimizer, config, 0, _sampling(0), batch.provenance)


def test_transformer_consistency_checkpoint_errors(tmp_path, monkeypatch):
    """transformer_consistency 的保存和读法：拒覆盖、游标、SHA、规格/来源/参数名/运行环境/模式、读回游标；
    登记规格不匹配时不构造。每条失败都不改传进来的模型。"""
    _, batch = e06_case()
    config = e06_config(**v4_loss_fields())
    model = transformer()
    selected = _save("transformer_consistency", model, config, tmp_path/"tc.pt", batch)
    optimizer = comparison.make_optimizer(model)
    with pytest.raises(FileExistsError):
        comparison.save_checkpoint("transformer_consistency", tmp_path/"tc.pt", model, optimizer, config, 0, _sampling(0), batch.provenance)
    with pytest.raises(ValueError, match="^checkpoint cursor must equal completed updates$"):
        comparison.save_checkpoint("transformer_consistency", tmp_path/"tc2.pt", model, optimizer, config, 1, _sampling(0), batch.provenance)
    assert not (tmp_path/"tc2.pt").exists()
    with pytest.raises(ValueError, match="^checkpoint SHA mismatch before loading$"):
        conditions.load_transformer_consistency_checkpoint(tmp_path/"tc.pt", "0"*64, model, optimizer, config,
                                                           selected["checkpoint_provenance"])
    payload = torch.load(tmp_path/"tc.pt", weights_only=True)
    def write(name, change):
        value = copy.deepcopy(payload)
        change(value)
        torch.save(value, tmp_path/name)
        return selected | dict(checkpoint=name, checkpoint_sha256=w.cache.file_sha(tmp_path/name))
    mismatch = "^Transformer checkpoint configuration/source/runtime mismatch$"
    cases = [
        ("schema.pt", lambda p: p.update(schema="F05_transformer_checkpoint_v2"), mismatch),
        ("spec.pt", lambda p: p["config"].update(layers=2), mismatch),
        ("provenance.pt", lambda p: p["provenance"].update(evidence_role="fictional"), mismatch),
        ("names.pt", lambda p: p.update(optimizer_param_names=p["optimizer_param_names"][::-1]), mismatch),
        ("runtime.pt", lambda p: p.update(runtime=dict(p["runtime"], fictional=1)), mismatch),
        ("mode.pt", lambda p: p.update(model_training=1), mismatch),
        ("cursor.pt", lambda p: p["sampling_state"].update(next_batch_index=1), "^checkpoint cursor mismatch$"),
    ]
    for name, change, message in cases:
        other = transformer()
        with pytest.raises(ValueError, match=message):
            _load("transformer_consistency", other, config, write(name, change), tmp_path)
        assert tensors_equal(other.state_dict(), transformer().state_dict()) and other.training
    monkeypatch.setattr(calibration, "load_addon_weights", lambda condition: dict(time_weight=1., space_weight=1.))
    monkeypatch.setattr(comparison, "choose_architecture", lambda: dict(matched=False))
    with pytest.raises(ValueError, match="^parameter matching failed; comparison blocked without relaxing 1%$"):
        comparison.build_condition("transformer_consistency", CPU)


def test_load_selected_errors(tmp_path, monkeypatch):
    """07 §2.5：新条件的 load_selected 和 v7 分支核同样几条（步数、均值、来源、路径不越界、SHA、两处步数），
    任何一条不过都不改模型；全过时恢复存的状态并切到 eval。"""
    model, batch, config = case("koopman")
    with torch.no_grad():
        model.omega.add_(.5)
    (tmp_path/"run").mkdir()
    selected = _save("koopman", model, config, tmp_path/"run"/"k.pt", batch)
    provenance = selected["checkpoint_provenance"]
    def load(changes):
        other = koopman_model()
        try:
            return other, evaluation.load_selected_checkpoint("koopman", selected | changes, model=other,
                optimizer=comparison.make_optimizer(other), config=config, expected_provenance=provenance, root=tmp_path/"run")
        except ValueError:
            assert tensors_equal(other.state_dict(), koopman_model().state_dict()) and other.training
            raise
    step = "^trained condition and explicit successful validation step required$"
    mean = "^invalid validation future mean or original checkpoint provenance$"
    cases = [
        (dict(step=-1), step), (dict(step=True), step),
        (dict(mean=float("nan")), mean), (dict(mean=-.1), mean), (dict(mean="0.2"), mean),
        (dict(checkpoint_provenance=dict(provenance, evidence_role="fictional")), mean),
        (dict(checkpoint="../k.pt"), "^selected checkpoint escaped its run$"),
        (dict(checkpoint_sha256="0"*64), "^checkpoint SHA mismatch before loading$"),
        (dict(step=1), "^selected successful step differs before any state restore$"),
    ]
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            load(changes)
    other, progress = load({})
    assert progress.completed_steps == 0 and not other.training and tensors_equal(other.state_dict(), model.state_dict())
    monkeypatch.setattr(conditions, "load_koopman_checkpoint", lambda *a: training.CheckpointProgress(1, {}))
    with pytest.raises(ValueError, match="^selected successful step differs from checkpoint$"):
        load({})


# ---------- 评测分发（07 §2.5） ----------

def _eval_model(condition):
    return dict(po_jepa_time_only=lambda: po(max_modes=20), po_jepa_space_only=lambda: po(max_modes=20),
                fixed_fourier=lambda: po("fourier"), transformer_consistency=transformer, koopman=koopman_model)[condition]()


@pytest.mark.parametrize("condition", ADDON)
def test_run_prepared_evaluation_dispatch(monkeypatch, tmp_path, condition):
    queries = [evaluation.EvaluationQuery(180, 0, 2., 16, "point"), evaluation.EvaluationQuery(180, 0, 2., 12, "pair"),
               evaluation.EvaluationQuery(180, 0, 2., 8, "point")]
    monkeypatch.setattr(evaluation, "load_test_window",
                        lambda root, q, device: fictional_window(evaluation, q.trajectory_id, q.start_tick, q.duration_cycles))
    model = _eval_model(condition)
    report = evaluation.run_prepared_evaluation(model, condition, tmp_path/"none", tmp_path/"run", e06_config(),
                                                queries=queries, checkpoint_identity=dict(step=0))
    assert report["counts"] == dict(expected=3, completed=3, valid=3, failed=0, not_run=0)
    assert report["condition"] == condition and model.training


@pytest.mark.parametrize("condition", ADDON)
def test_validation_metrics_accept_new_models(condition):
    """训练中的验证（comparison.validate_condition → evaluate_batch）：5 个新模型都能算未来距离、TCD、SCD。"""
    model, batch, config = case(condition)
    model.eval()
    metrics = comparison.evaluate_batch(model, batch.raw_z, batch.controls, batch.duration, batch.transfer, batch.raw_target, config)
    assert set(metrics) == {"future_jepa_distance", "tcd_frobenius", "scd_frobenius"}
    assert all(torch.isfinite(m.values).all() and m.values.shape == (2,) for m in metrics.values())


@pytest.mark.parametrize("condition, wrong", [
    ("fixed_fourier", lambda: po("graph")), ("po_jepa_time_only", lambda: po("fourier")),
    ("po_jepa_space_only", transformer), ("transformer_consistency", koopman_model), ("koopman", transformer)])
def test_evaluation_rejects_wrong_model_type(tmp_path, condition, wrong):
    with pytest.raises(TypeError, match="^condition/model type mismatch$"):
        evaluation.run_prepared_evaluation(wrong(), condition, tmp_path/"none", tmp_path/"run", e06_config(),
                                           queries=[evaluation.EvaluationQuery(180, 0, 2., 16, "point")], checkpoint_identity=dict(step=0))
    assert not (tmp_path/"run").exists()


# ---------- 监控名单（S10 的收紧） ----------

def test_monitoring_accepts_only_known_names(tmp_path):
    config = e06_config()
    assert comparison.monitored_consistency_terms("po_jepa_full", config) == ("time", "space")
    assert comparison.monitored_consistency_terms("identity", config) is None
    assert comparison.monitored_consistency_terms("po_jepa_time_only", e06_config(space_weight=0.)) == ("time",)
    (tmp_path/"updates").mkdir()
    for call in (lambda: comparison.monitored_consistency_terms("fictional_condition", config),
                 lambda: comparison.consistency_monitor_summary(tmp_path, "fictional_condition", config),
                 lambda: comparison.early_check(tmp_path, "fictional_condition", config, 1)):
        with pytest.raises(ValueError, match="^unknown condition for consistency monitoring$"):
            call()


# ---------- 补跑标定读取（16 §3.2） ----------

@pytest.fixture
def addon_files(monkeypatch, tmp_path):
    v7 = tmp_path/"v7.yaml"
    v7.write_text("time_weight: 1.0\n", encoding="utf-8")
    path = tmp_path/"addon.yaml"
    monkeypatch.setattr(checks, "v7_calibration_path", lambda: v7)
    monkeypatch.setattr(calibration, "addon_calibration_path", lambda: path)
    monkeypatch.setattr(checks, "load_protocol", lambda: dict(calibration=dict(rho=1.), model=dict(graph=dict(sigma_z=.5))))
    def write(**changes):
        value = dict(fixed_fourier=dict(rho=1., space_weight=.5, time_weight=.25, source="fictional"),
                     transformer_consistency=dict(rho=1., space_weight=2, time_weight=1e-3, source="fictional"),
                     sigma_z_used=.5, v7_calibration_sha256=w.cache.file_sha(v7))
        value.update(changes)
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        return value
    return write, v7, path


def test_addon_calibration_reads_and_validates(addon_files):
    write, v7, path = addon_files
    with pytest.raises(ValueError, match="请先运行 python run.py calibrate-addon"):
        calibration.load_addon_weights("fixed_fourier")
    value = write()
    assert calibration.load_addon_weights("fixed_fourier") == dict(time_weight=.25, space_weight=.5)
    weights = calibration.load_addon_weights("transformer_consistency")
    assert weights == dict(time_weight=1e-3, space_weight=2.) and type(weights["space_weight"]) is float
    calibration.require_addon_calibration()
    entry = value["fixed_fourier"]
    cases = [
        (dict(extra=1), "^add-on calibration must contain exactly fixed_fourier, transformer_consistency, sigma_z_used, v7_calibration_sha256$"),
        (dict(fixed_fourier=entry | dict(extra=1)), "^add-on calibration entry fixed_fourier must contain exactly rho, space_weight, time_weight, source$"),
        (dict(fixed_fourier=entry | dict(time_weight=0.)), "^add-on calibration fixed_fourier.time_weight must be a finite positive number$"),
        (dict(fixed_fourier=entry | dict(space_weight=True)), "^add-on calibration fixed_fourier.space_weight must be a finite positive number$"),
        (dict(fixed_fourier=entry | dict(space_weight=float("inf"))), "^add-on calibration fixed_fourier.space_weight must be a finite positive number$"),
        (dict(fixed_fourier=entry | dict(time_weight="0.25")), "^add-on calibration fixed_fourier.time_weight must be a finite positive number$"),
        (dict(fixed_fourier=entry | dict(rho=.5)), "^add-on calibration rho differs from the protocol calibration.rho$"),
        (dict(fixed_fourier=entry | dict(rho=True)), "^add-on calibration rho differs from the protocol calibration.rho$"),
        (dict(sigma_z_used=1.), "^add-on calibration sigma_z_used differs from the protocol sigma_z$"),
        (dict(v7_calibration_sha256="0"*64), "^add-on calibration was made against a different v7 calibration file$"),
    ]
    for changes, message in cases:
        write(**changes)
        with pytest.raises(ValueError, match=message):
            calibration.load_addon_weights("fixed_fourier")
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="^add-on calibration must contain exactly"):
        calibration.load_addon_weights("fixed_fourier")
    write()
    v7.unlink()
    with pytest.raises(ValueError, match="^add-on calibration was made against a different v7 calibration file$"):
        calibration.load_addon_weights("fixed_fourier")
    with pytest.raises(ValueError, match="^only fixed_fourier and transformer_consistency use the add-on calibration$"):
        calibration.load_addon_weights("koopman")


def test_addon_calibration_path_follows_v7_name(monkeypatch):
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    monkeypatch.setenv(checks.UPDATES_ENV, "10000")
    old = datasets._set_for_tests(datasets.DatasetSpec("van_der_pol", 2000))
    try:
        assert checks.v7_calibration_path() == checks.ROOT/"configs/calibration/van_der_pol-2000-u10000.yaml"
        assert calibration.addon_calibration_path() == checks.ROOT/"configs/calibration/addon/van_der_pol-2000-u10000.yaml"
    finally:
        datasets._set_for_tests(old)


@pytest.mark.parametrize("condition", ADDON)
def test_workflow_requires_addon_calibration(v7, monkeypatch, tmp_path, condition):
    calls = []
    monkeypatch.setattr(w, "verify_bundle", lambda: "fictional")
    monkeypatch.setattr(w.environment, "verify_versions", lambda: {})
    monkeypatch.setattr(w, "require_calibrated_weights", lambda swept=False: calls.append("v7"))
    monkeypatch.setattr(calibration, "require_addon_calibration", lambda: calls.append("addon") or (_ for _ in ()).throw(ValueError("fictional missing add-on calibration")))
    monkeypatch.setattr(v7jobs, "run_job", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fictional job reached")))
    args = Namespace(command="run", device="cuda:0", features_root=tmp_path/"absent", output=tmp_path/"run", condition=condition,
                     resume_from=None, training_run=None)
    needs = condition in ("transformer_consistency", "fixed_fourier")
    with pytest.raises(ValueError if needs else RuntimeError, match="fictional missing add-on calibration" if needs else "fictional job reached"):
        w.execute(args)
    assert calls == (["v7", "addon"] if needs else ["v7"])
    assert w.read(tmp_path/"run/FAILED.json")["stage"] == ("calibration" if needs else "job")


# ---------- 登记核对（07 §2.4 _validate_registration） ----------

@pytest.mark.parametrize("condition", ADDON)
def test_validate_registration(condition):
    model, _, _ = case(condition)
    sequence = [[comparison.data.WindowQuery(0, 0, 1.)]]
    registration = _registration(condition, model, sequence)
    comparison._validate_registration(condition, model, sequence, registration)
    count = comparison.parameter_counts(model)["total"]
    with pytest.raises(ValueError, match="^registered trainable condition and nonempty sequence required$"):
        comparison._validate_registration(condition, model, [], registration)
    with pytest.raises(ValueError, match="^capacity matching failed; fixed 1% unchanged$"):
        comparison._validate_registration(condition, model, sequence, registration | dict(parameter_match=comparison.parameter_match(count, 2*count)))
    with pytest.raises(ValueError, match="^shared queries/config/consumer source differs$"):
        comparison._validate_registration(condition, model, sequence, registration | dict(query_sequence_sha256="0"*64))
    if condition == "koopman":
        bigger = build_koopman(KoopmanConfig(modes=KOOPMAN_MODES+1, action_dim=2, encoder_dim=3), 1703, CPU)
        with pytest.raises(ValueError, match="^Koopman modes differ from the parameter-matching rule$"):
            comparison._validate_registration(condition, bigger, sequence, registration)
        with pytest.raises(TypeError, match="^condition/model type mismatch$"):
            comparison._validate_registration(condition, transformer(), sequence, registration)
    else:
        near = comparison.parameter_match(count+1, count+1)
        with pytest.raises(ValueError, match="^actual condition model count differs from preregistration$"):
            comparison._validate_registration(condition, model, sequence, registration | dict(parameter_match=near))


def test_transformer_consistency_registration_uses_transformer_count():
    model = transformer()
    sequence = [[comparison.data.WindowQuery(0, 0, 1.)]]
    count = comparison.parameter_counts(model)["total"]
    registration = _registration("transformer_consistency", model, sequence)
    registration["parameter_match"] = comparison.parameter_match(count+1, count)
    comparison._validate_registration("transformer_consistency", model, sequence, registration)
    with pytest.raises(ValueError, match="^actual condition model count differs from preregistration$"):
        comparison._validate_registration("po_jepa_time_only", po(max_modes=20), sequence,
                                          registration | dict(parameter_match=comparison.parameter_match(count, count+1)))


# ---------- 来源登记（07 §2.7） ----------

def test_consumer_sources_register_new_files(v7, monkeypatch):
    sources = comparison.consumer_sources()
    added = ["src/po_jepa/fourier.py", "src/po_jepa/koopman.py", "src/po_jepa/generic_consistency.py", "src/po_jepa/ln_metrics.py",
             "src/po_jepa/runtime/addon/__init__.py", "src/po_jepa/runtime/addon/calibration.py", "src/po_jepa/runtime/addon/conditions.py"]
    assert checks.is_v7()
    assert list(sources)[-len(added):] == added
    for name in added:
        assert sources[name] == w.cache.file_sha(comparison.ROOT/name)
        assert name.startswith("src/po_jepa/") and name.endswith(".py")
    training._provenance(dict(e06_case()[1].provenance, source_files_sha256=sources))
    monkeypatch.delenv(checks.PROTOCOL_ENV)
    assert not set(added) & set(comparison.consumer_sources())   # v6 协议下来源登记和原来相同


@pytest.mark.parametrize("condition", ["po_jepa_time_only", "transformer_consistency"])
def test_consistency_share_applies_warmup_scale(monkeypatch, condition):
    """S10b 审核 B M10：share = 权重 × warmup_scale × grad_k / grad_jepa，warmup_scale < 1 时也对。"""
    norms = dict(jepa=1., time=2., space=3., grad_jepa=4., grad_time=2., grad_space=1.)
    monkeypatch.setattr(training, "consistency_gradient_norms", lambda model, batch, loss_config: dict(norms))
    monkeypatch.setattr(calibration, "generic_gradient_norms", lambda model, batch, loss_config: dict(norms))
    monkeypatch.setattr(training, "consistency_warmup_scale", lambda loss_config, completed: .25)
    interval = checks.load_protocol()["monitoring"]["consistency_interval"]
    loss_config = Namespace(time_weight=.5, space_weight=8.)
    result = conditions.consistency_share(condition, None, None, loss_config, interval)
    assert result["warmup_scale"] == .25
    assert result["share"] == dict(time=.5*.25*2./4., space=8.*.25*1./4.)
