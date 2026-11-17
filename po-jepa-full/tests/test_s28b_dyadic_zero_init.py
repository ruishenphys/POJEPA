"""S28b：时间一致性多段切分（M1）、P_out 零初始化（M2）、v7 变体选择和检查点标签。

对照：差距清单 §2 M1/M2、§3 C2；论文引理 5.4 的 c(Π) = Σ_{i≥2} t_{i−1}τ_i，2^k 等分时 = T²(1 − 2^{−k})/2。
"""
from dataclasses import replace
from fractions import Fraction
import hashlib
from pathlib import Path

import pytest
import torch

from po_jepa import datasets
from po_jepa import losses
from po_jepa.actions import ControlSpec, from_physical_segments, select_action_rows, split_action_segments
from po_jepa.dynamics import DynamicsConfig, IntegratorConfig, ModalDynamics
from po_jepa.geometry import build_spatial_transfer, regular_grid
from po_jepa.model import POJEPA, ProjectionConfig
from po_jepa.runtime import checks, training
from po_jepa.spectral import GraphConfig, GraphSpectrum
from po_jepa import theory_bounds

E = 32
SPEC = ControlSpec(2, .25, ("u_x", "u_y"), ("p/s", "p/s"), (-.5, -.5), (.5, .5), (1., 1.), (0., 0.), "S28b test")
CUTS = (.75, 1.5, 2.5, 3.25, 5.5)   # 动作切换点（周期），故意不落在 2^k 等分点上


def make_model(readout_init="xavier", seed=1703):
    """v7 写法的小模型：encoder 建图、K=8 固定、残差读出、token 驱动；E=32 只为快。"""
    torch.manual_seed(seed)
    dynamics = checks.load_protocol()["model"]["dynamics"]
    graph = GraphConfig(16, 1703, .125, 4., 8, 1e-10, 1e-8, affinity_source="encoder", feature_dim=E, fixed_modes=True)
    return POJEPA(ProjectionConfig(E, 64, readout="residual", readout_init=readout_init), GraphSpectrum(graph),
                  ModalDynamics(DynamicsConfig(**{**dynamics, "forcing_space": "token"})),
                  IntegratorConfig(**checks.load_protocol()["model"]["integrator"]))


def make_controls(durations, seed=0):
    """每行 [0, T) 上的分段常值动作，切换点取 CUTS 里落在 (0, T) 的那些。"""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for T in durations:
        edges = [0.]+[c for c in CUTS if c < T]+[float(T)]
        rows.append([(a, b) for a, b in zip(edges[:-1], edges[1:])])
    width = max(len(r) for r in rows)
    starts = torch.full((len(rows), width), float("nan"), dtype=torch.float64)
    ends = torch.full_like(starts, float("nan"))
    values = torch.full((len(rows), width, 2), float("nan"), dtype=torch.float32)
    for b, row in enumerate(rows):
        for j, (a, c) in enumerate(row):
            starts[b, j], ends[b, j] = a*.25, c*.25
            values[b, j] = torch.rand(2, generator=g)-.5
    counts = torch.tensor([len(r) for r in rows], dtype=torch.int64)
    return from_physical_segments(starts_seconds=starts, ends_seconds=ends, u_values=values, n_segments=counts,
        segment_valid=torch.arange(width)[None] < counts[:, None], origin_seconds=torch.zeros(len(rows), dtype=torch.float64),
        control_spec=SPEC, source_ids=tuple(f"s28b-{b}" for b in range(len(rows))))


def make_z(b, h=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 1, E, generator=g)*2+torch.randn(b, h*h, E, generator=g)).float()


def halves(durations):
    total = torch.tensor(durations, dtype=torch.float64)
    return total/2, total/2


def make_batch(durations, *, seed=0):
    from test_training import e06_provenance
    z = make_z(len(durations), seed=seed)
    controls = make_controls(durations, seed=seed)
    tau, sigma = halves(durations)
    duration = tau+sigma
    observed = controls.source_origin_seconds.clone()
    return training.TrainingBatch(z, torch.roll(z, 1, -1)*.8+.03, controls, duration, tau, sigma,
        build_spatial_transfer(regular_grid(8, 8), target_height=4, target_width=4), observed,
        observed+duration*SPEC.control_period_seconds, tuple(f"s28b:{i}:{d}" for i, d in enumerate(durations)), e06_provenance())


def config(partition="bisection", **changes):
    values = dict(jepa_exponent=1., layer_norm_epsilon=1e-5, time_epsilon=1e-8, space_epsilon=1e-8, time_weight=.5,
                  space_weight=.25, consistency_metric="input_scale", temporal_partition=partition)
    values.update(changes)
    return losses.LossConfig(**values)


# ---------- M1：k 的抽法 ----------

def test_level_options_follow_minimum_segment_rule():
    """段长 T/2^k ≥ 0.25 周期才抽（Jerry 2026-11-17）：T=1 只有 k=1,2；T=2,4,8 有 1,2,3；T=0.5 只有 1。"""
    assert losses.dyadic_level_options(1.) == (1, 2)
    assert all(losses.dyadic_level_options(T) == (1, 2, 3) for T in (2., 4., 8.))
    assert losses.dyadic_level_options(.5) == (1,)
    assert losses.dyadic_level_options(.25) == ()
    with pytest.raises(ValueError):
        losses.dyadic_levels(torch.tensor([.25], dtype=torch.float64), ("a",), step=0)


def test_levels_deterministic_uniform_and_leave_rng_alone():
    """同 (step, 行, 样本 ID) 同一个 k；不同步数近似均匀；全程不碰 torch/python 随机状态。"""
    import random
    torch.manual_seed(5)
    random.seed(5)
    before = (torch.get_rng_state().clone(), random.getstate())
    duration = torch.tensor([1., 2., 4., 8.], dtype=torch.float64)
    ids = ("a", "b", "c", "d")
    first = losses.dyadic_levels(duration, ids, step=7)
    assert torch.equal(first, losses.dyadic_levels(duration, ids, step=7)) and first.dtype == torch.int64
    counts = {T: {} for T in (1., 2., 4., 8.)}
    for step in range(3000):
        for T, k in zip(duration.tolist(), losses.dyadic_levels(duration, ids, step=step).tolist()):
            assert k in losses.dyadic_level_options(T)
            counts[T][k] = counts[T].get(k, 0)+1
    for T, seen in counts.items():
        options = losses.dyadic_level_options(T)
        assert set(seen) == set(options)
        assert all(abs(v/3000-1/len(options)) < .04 for v in seen.values()), (T, seen)
    assert torch.equal(before[0], torch.get_rng_state()) and before[1] == random.getstate()
    # 和步数、行号、样本 ID 都有关
    assert any(not torch.equal(losses.dyadic_levels(duration, ids, step=s), first) for s in range(8, 20))
    expected = int.from_bytes(hashlib.sha256(b"po-jepa-dyadic-v1|7|3|d").digest()[:8], "big") % 3
    assert first[3].item() == (1, 2, 3)[expected]


@pytest.mark.parametrize("T", [1, 2, 4, 8])
def test_dyadic_coefficient_is_lemma_sum_and_k1_is_tau_sigma(T):
    """c(Π_k) = Σ_{i≥2} t_{i−1}τ_i = (T² − Σ τ_i²)/2 = T²(1 − 2^{−k})/2（有理数精确）；k=1 和 τσ 逐位相同。"""
    for k in losses.dyadic_level_options(float(T)):
        n, h = 2**k, Fraction(T, 2**k)
        times = [h*i for i in range(n+1)]
        lemma = sum(times[i-1]*(times[i]-times[i-1]) for i in range(2, n+1))
        assert lemma == (Fraction(T)**2-n*h*h)/2 == Fraction(T)**2*(1-Fraction(1, n))/2
        coefficient = losses.dyadic_coefficient(torch.tensor([float(T)], dtype=torch.float64), torch.tensor([k]))
        assert Fraction(coefficient.item()) == lemma
    half = torch.tensor([T/2], dtype=torch.float64)
    assert torch.equal(losses.dyadic_coefficient(half*2, torch.tensor([1])), half*half)


# ---------- M1：组合路径与损失 ----------

def test_k1_partition_is_bitwise_the_bisection_loss():
    """全是 k=1 时，多段切分的 direct/composed/per_sample 和原式15 二等分逐位相同（方案 §5.1：k=1 逐位等于现在的损失）。"""
    model, durations = make_model(), [2., 4., 8.]
    z, controls = make_z(3), make_controls(durations)
    tau, sigma = halves(durations)
    grid = regular_grid(8, 8)
    old = losses.temporal_predictions(model, z, controls, tau, sigma, grid)
    new = losses.partition_predictions(model, z, controls, tau, sigma, torch.ones(3, dtype=torch.int64), grid)
    assert torch.equal(old.direct, new.direct) and torch.equal(old.composed, new.composed)
    cfg = config()
    assert torch.equal(losses.temporal_consistency_loss(old, config=cfg).per_sample,
                       losses.partition_consistency_loss(new, config=cfg).per_sample)


def test_partition_rollout_is_the_lemma_rollout_and_groups_by_level():
    """composed 就是式 (partition_rollout) 的 Φ_Π：和 theory_bounds 逐段算的 p_n 逐位相同；混合 k 时按组算、按行放回。"""
    model, durations = make_model(), [2., 2., 4., 1.]
    z, controls, grid = make_z(4), make_controls(durations), regular_grid(8, 8)
    tau, sigma = halves(durations)
    levels = torch.tensor([3, 2, 3, 1])
    with torch.no_grad():
        mixed = losses.partition_predictions(model, z, controls, tau, sigma, levels, grid)
        for k in (1, 2, 3):
            rows = [b for b in range(4) if levels[b] == k]
            sub = select_action_rows(controls, rows)
            for row_position, b in enumerate(rows):
                T = durations[b]
                single = select_action_rows(sub, [row_position])
                expected = theory_bounds.temporal_terms(model, z[b:b+1], single, (T/2**k,)*2**k, grid,
                                                        scale2=losses.input_scale(z[b:b+1])).rollout
                # 单行和整组算的只差批大小带来的 float32 舍入（同组逐位相同在下面核）
                assert torch.allclose(mixed.composed[b:b+1], expected, rtol=1e-5, atol=1e-4)
        same = losses.partition_predictions(model, z[:2], select_action_rows(controls, [0, 1]), tau[:2], sigma[:2],
                                            torch.tensor([2, 2]), grid)
        lemma = theory_bounds.temporal_terms(model, z[:2], select_action_rows(controls, [0, 1]), (.5, .5, .5, .5), grid,
                                             scale2=losses.input_scale(z[:2]))
        assert torch.equal(same.composed, lemma.rollout)


def test_partition_loss_matches_hand_formula_and_keeps_both_branch_gradients():
    """per_sample = mean_(N,E)((direct − composed)/s)² / (c(Π_k)² + ε)；两支（含中间段输入）都有梯度。"""
    model, durations = make_model(), [2., 8.]
    z, controls, grid = make_z(2), make_controls(durations), regular_grid(8, 8)
    tau, sigma = halves(durations)
    levels = torch.tensor([2, 3])
    pred = losses.partition_predictions(model, z, controls, tau, sigma, levels, grid)
    cfg = config("dyadic")
    loss = losses.partition_consistency_loss(pred, config=cfg)
    s2 = z.double().var(dim=(1, 2), unbiased=False)
    numerator = ((pred.direct.double()-pred.composed.double()).square().mean(dim=(1, 2)))/s2
    c = torch.tensor([2.**2*(1-.25)/2, 8.**2*(1-.125)/2], dtype=torch.float64)
    assert torch.allclose(loss.per_sample, numerator/(c.square()+1e-8), rtol=1e-6, atol=0)
    assert loss.per_sample.dtype == torch.float64 and torch.all(loss.per_sample > 0)
    grads = torch.autograd.grad(loss.mean, [p for _, p in training._parameters(model)], allow_unused=True, retain_graph=True)
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    # composed 依赖中间段的 raw 输出：只对 composed 求导时，p_in（每段重新投影）的梯度也非零
    composed_grad = torch.autograd.grad(pred.composed.square().mean(), model.p_in.weight, retain_graph=True)[0]
    assert composed_grad.abs().sum() > 0
    # 整批同一个 k（不分组的那条路径）也一样保留梯度
    same = losses.partition_predictions(model, z, controls, tau, sigma, torch.tensor([2, 2]), grid)
    assert torch.autograd.grad(same.composed.square().mean(), model.p_in.weight)[0].abs().sum() > 0


def test_partition_contract_errors():
    model, durations = make_model(), [1., 2.]
    z, controls, grid = make_z(2), make_controls(durations), regular_grid(8, 8)
    tau, sigma = halves(durations)
    with pytest.raises(ValueError, match="not allowed"):
        losses.partition_predictions(model, z, controls, tau, sigma, torch.tensor([3, 1]), grid)
    with pytest.raises(ValueError, match="equal-halves"):
        losses.partition_predictions(model, z, controls, tau*.5, sigma*1.5, torch.tensor([1, 1]), grid)
    with pytest.raises(ValueError, match="int64"):
        losses.partition_predictions(model, z, controls, tau, sigma, torch.tensor([1., 1.]), grid)
    with pytest.raises(ValueError, match="temporal_partition"):
        config("trisection")
    with pytest.raises(ValueError, match="step"):
        losses.time_consistency(model, z, controls, tau, sigma, grid, config=config("dyadic"), step=None,
                                sample_ids=("a", "b"))


def test_select_action_rows_keeps_endpoints_and_gradients():
    controls = make_controls([2., 4., 8.])
    controls = replace(controls, u_values=controls.u_values.detach().requires_grad_())
    sub = select_action_rows(controls, [0, 2])
    assert sub.source_ids == (controls.source_ids[0], controls.source_ids[2])
    assert torch.equal(sub.duration, controls.duration[[0, 2]]) and torch.equal(sub.n_segments, controls.n_segments[[0, 2]])
    sub.u_values.sum().backward()
    assert torch.equal(controls.u_values.grad[1], torch.zeros_like(controls.u_values.grad[1]))
    for bad in ([], [1, 0], [0, 0], [3], (0, 1)):
        with pytest.raises(ValueError):
            select_action_rows(controls, bad)
    # 和 split_action_segments 同一切法：k=1 的两段和二等分的两段逐位相同
    pair = split_action_segments(controls, tau=controls.duration/2, sigma=controls.duration/2)
    from po_jepa.actions import _exact_split_slice
    first = _exact_split_slice(controls, start=controls.duration/2*0, duration=controls.duration/2)
    assert torch.equal(first.ends, pair.first.ends) and torch.equal(first.u_values, pair.first.u_values)


# ---------- M1：单步、读出计数、快照标签 ----------

def count_phi(model, monkeypatch):
    calls = []
    original = model.prepare_query
    monkeypatch.setattr(model, "prepare_query", lambda *a, **k: calls.append(1) or original(*a, **k))
    return calls


@pytest.mark.parametrize("partition", ["bisection", "dyadic"])
def test_train_step_phi_count_logs_and_rng(monkeypatch, partition):
    """一次更新的完整 Φ 次数 = temporal_phi_calls + 2（空间）；二等分是 5。k 只由 (步数, 样本) 定，不读写随机状态。"""
    model, batch = make_model(), make_batch([1., 2., 4., 8.])
    model.train()
    optimizer = training.make_optimizer(model)
    cfg = config(partition)
    calls = count_phi(model, monkeypatch)
    torch.manual_seed(11)
    before = torch.get_rng_state().clone()
    result = training.train_step(model, optimizer, batch, loss_config=cfg, completed_steps=37)
    assert torch.equal(before, torch.get_rng_state())
    expected = losses.temporal_phi_calls(cfg, batch.duration, batch.sample_ids, 37)+2
    assert len(calls) == expected and (partition == "dyadic" or expected == 5)
    if partition == "dyadic":
        levels = losses.dyadic_levels(batch.duration, batch.sample_ids, step=37)
        assert expected == 1+sum(2**k for k in set(levels.tolist()))+2
    assert result.completed_steps == 38 and result.loss_values["time"] > 0


def test_gradient_norms_use_the_training_step_levels():
    """consistency_gradient_norms(step=s) 量的 L_time 就是 train_step 第 s 步那串 k 的损失。"""
    model, batch = make_model(), make_batch([2., 4., 8., 1.])
    cfg = config("dyadic")
    norms = training.consistency_gradient_norms(model, batch, loss_config=cfg, step=12)
    _, time = losses.time_consistency(model, batch.raw_z, batch.controls, batch.tau, batch.sigma, batch.transfer.source,
                                      config=cfg, step=12, sample_ids=batch.sample_ids)
    assert norms["time"] == time.mean.item()
    with pytest.raises(ValueError, match="step"):
        training.consistency_gradient_norms(model, batch, loss_config=cfg)


def test_loss_snapshot_labels_partition_and_keeps_legacy_snapshot():
    """二等分不进快照（v7/v7.1 旧检查点照读）；多段切分进快照（检查点标签）；rollout_sup 只收二等分。"""
    legacy = training._loss_config(config(), "full")
    assert "temporal_partition" not in legacy
    dyadic = training._loss_config(config("dyadic"), "full")
    assert dyadic["temporal_partition"] == "dyadic" and {k: v for k, v in dyadic.items() if k != "temporal_partition"} == legacy
    assert not training._same_native(dyadic, legacy)
    with pytest.raises(ValueError, match="rollout_sup"):
        training._loss_config(config("dyadic"), "rollout_sup")
    assert "temporal_partition" not in training._loss_config(config(), "rollout_sup")


def test_checkpoint_from_one_partition_refuses_the_other(tmp_path):
    """检查点标签：多段切分训练的检查点不能当二等分读，反之亦然。"""
    from test_training import e06_provenance
    model, batch = make_model(), make_batch([2., 4.])
    model.train()
    optimizer = training.make_optimizer(model)
    training.train_step(model, optimizer, batch, loss_config=config("dyadic"), completed_steps=0)
    sampling = dict(kind="explicit_prepared_batches", next_batch_index=1, query_generator_state=None)
    path = tmp_path/"dyadic.pt"
    sha = training.save_training_checkpoint(path, model=model, optimizer=optimizer, loss_config=config("dyadic"),
        condition="full", completed_steps=1, sampling_state=sampling, provenance=e06_provenance())
    fresh = make_model()
    with pytest.raises(ValueError, match="mismatch"):
        training.load_training_checkpoint(path, expected_sha256=sha, model=fresh, optimizer=training.make_optimizer(fresh),
            loss_config=config(), condition="full", expected_provenance=e06_provenance())
    progress = training.load_training_checkpoint(path, expected_sha256=sha, model=fresh,
        optimizer=training.make_optimizer(fresh), loss_config=config("dyadic"), condition="full",
        expected_provenance=e06_provenance())
    assert progress.completed_steps == 1


# ---------- M2：零初始化 ----------

def test_zero_init_consumes_same_rng_and_only_zeroes_p_out_weight():
    """先抽 Xavier 再清零：随机数消耗相同，p_out.weight 以外的参数与 Xavier 版逐位相同（2×2 筛选只差这一项）。"""
    xavier = make_model("xavier")
    after_xavier = torch.get_rng_state().clone()
    zero = make_model("zero")
    assert torch.equal(after_xavier, torch.get_rng_state())
    for (name, a), (_, b) in zip(xavier.state_dict().items(), zero.state_dict().items()):
        if name == "p_out.weight":
            assert torch.count_nonzero(b) == 0 and torch.count_nonzero(a) > 0
        else:
            assert torch.equal(a, b), name
    assert zero.p_out.weight.requires_grad


@pytest.mark.parametrize("duration", [0., .25, 1., 3., 8., 12.])
def test_zero_init_starts_exactly_at_identity(duration):
    """零初始化后第 0 步对任意 τ 都有 Φ_τ(Z) = Z 逐位（方案 §5.1 残差读出那一行）。"""
    model, z = make_model("zero"), make_z(2)
    controls = make_controls([max(duration, .25)]*2)
    out = model(z, controls, torch.full((2,), duration, dtype=torch.float64), regular_grid(8, 8))
    assert torch.equal(out, z)


def test_zero_init_first_update_learns_from_p_out():
    """起点一致性损失恰为 0（两支都等于 Z），JEPA 梯度只经 p_out.weight 进来；一步之后 p_out 不再为 0，其余参数也开始有梯度。"""
    model, batch = make_model("zero"), make_batch([2., 4.])
    model.train()
    optimizer = training.make_optimizer(model)
    result = training.train_step(model, optimizer, batch, loss_config=config("dyadic"), completed_steps=0)
    assert result.loss_values["time"] == 0. and result.loss_values["space"] == 0.
    grads = {n: p.grad for n, p in model.named_parameters()}
    assert grads["p_out.weight"].abs().sum() > 0
    assert all(torch.count_nonzero(g) == 0 for n, g in grads.items() if n != "p_out.weight")
    assert torch.count_nonzero(model.p_out.weight) > 0
    training.train_step(model, optimizer, batch, loss_config=config("dyadic"), completed_steps=1)
    assert model.p_in.weight.grad.abs().sum() > 0


def test_zero_init_config_contract_and_snapshot_label():
    with pytest.raises(ValueError, match="residual"):
        ProjectionConfig(E, 64, readout="direct", readout_init="zero")
    with pytest.raises(ValueError, match="readout_init"):
        ProjectionConfig(E, 64, readout="residual", readout_init="normal")
    assert "readout_init" not in training._model_config(make_model("xavier"))["projection"]
    assert training._model_config(make_model("zero"))["projection"]["readout_init"] == "zero"


# ---------- v7 变体选择 ----------

@pytest.fixture
def v7(monkeypatch):
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    monkeypatch.setenv(checks.UPDATES_ENV, "2000")
    monkeypatch.delenv(checks.PARTITION_ENV, raising=False)
    monkeypatch.delenv(checks.READOUT_INIT_ENV, raising=False)
    old = datasets._set_for_tests(datasets.DatasetSpec("double_well", 200))
    yield
    datasets._set_for_tests(old)


def test_default_variant_leaves_protocol_and_calibration_unchanged(v7, monkeypatch):
    """不设两个环境变量：协议、标定文件名和以前完全相同（进 git 的 double_well-200-u2000.yaml 照用）。"""
    config = checks.load_protocol()
    assert checks.v7_calibration_path().name == "double_well-200-u2000.yaml"
    assert config["loss"]["temporal_partition"] == "bisection" and "readout_init" not in config["model"]["projection"]
    assert "v7_uncalibrated" not in config


@pytest.mark.parametrize("partition,init,suffix", [("dyadic", "xavier", "~dyadic"), ("bisection", "zero", "~zeroinit"),
                                                   ("dyadic", "zero", "~dyadic~zeroinit")])
def test_variant_env_sets_protocol_and_own_calibration(v7, monkeypatch, partition, init, suffix):
    """变体写进协议（loss.temporal_partition、model.projection.readout_init），标定文件按变体分开，没标定就拒跑。"""
    from po_jepa.runtime import comparison, workflow
    monkeypatch.setenv(checks.PARTITION_ENV, partition)
    monkeypatch.setenv(checks.READOUT_INIT_ENV, init)
    config = checks.load_protocol()
    assert config["loss"]["temporal_partition"] == partition
    assert config["model"]["projection"].get("readout_init", "xavier") == init
    assert checks.v7_calibration_path().name == f"double_well-200-u2000{suffix}.yaml"
    assert config["v7_uncalibrated"]
    with pytest.raises(ValueError, match="还没标定"):
        workflow.require_calibrated_weights()
    model, loss = comparison.build_condition("po_jepa_full", torch.device("cpu"))
    assert loss.temporal_partition == partition and model.readout_init == init
    if init == "zero":
        assert torch.count_nonzero(model.p_out.weight) == 0
    from po_jepa.runtime import v7jobs
    assert v7jobs.with_variant(dict(condition="po_jepa_full")) == dict(condition="po_jepa_full",
                                                                     variant=dict(temporal_partition=partition, readout_init=init))


def test_variant_env_rejected_outside_v7_and_bad_values(monkeypatch):
    monkeypatch.delenv(checks.PROTOCOL_ENV, raising=False)
    monkeypatch.setenv(checks.PARTITION_ENV, "dyadic")
    with pytest.raises(ValueError, match="v7"):
        checks.load_protocol()
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    monkeypatch.setenv(checks.PARTITION_ENV, "halves")
    with pytest.raises(ValueError, match=checks.PARTITION_ENV):
        checks.v7_variant()
    monkeypatch.setenv(checks.PARTITION_ENV, "bisection")
    monkeypatch.setenv(checks.READOUT_INIT_ENV, "orthogonal")
    with pytest.raises(ValueError, match=checks.READOUT_INIT_ENV):
        checks.v7_variant()


def test_rollout_sup_build_stays_bisection_under_dyadic(v7, monkeypatch):
    """po_jepa_rollout_sup 的时间格是有真值两段滚动（11-12 定义），多段切分变体下也固定二等分。"""
    from po_jepa.runtime.addon import calibration, conditions
    monkeypatch.setenv(checks.PARTITION_ENV, "dyadic")
    monkeypatch.setattr(calibration, "load_addon_weights", lambda c: dict(time_weight=.3, space_weight=.4))
    _, loss = conditions.build("po_jepa_rollout_sup", torch.device("cpu"))
    assert loss.temporal_partition == "bisection"
    _, loss = conditions.build("po_jepa_time_only", torch.device("cpu"))
    assert loss.temporal_partition == "dyadic"


def test_summary_groups_variants_apart():
    import importlib.util
    spec = importlib.util.spec_from_file_location("summarize_v7", Path(checks.ROOT)/"tools/summarize_v7.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    job = dict(dataset="double_well-200", condition="po_jepa_full")
    assert module._label(job) == "double_well-200"
    assert module._label(dict(job, variant=dict(temporal_partition="dyadic", readout_init="zero"))) == "double_well-200~dyadic~zeroinit"
    assert "identity" not in module.REFERENCES and "po_jepa_no_consistency" in module.REFERENCES
