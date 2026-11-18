"""S28d：多段切分变体下 po_jepa_rollout_sup 改成 2^k 段有真值滚动（Jerry 2026-11-17 决策卡）。

定义（docs/segments/S28d.md §2）：k_b = losses.dyadic_levels(T, sample_ids, step)，和 po_jepa_full 的时间一致性同一串 k；
Φ_Π(Z; u) 是 losses.partition_predictions 的 composed（2^k 段、h = T/2^k、动作精确切开、每段接上一段的 raw 输出）；
L_roll = mean_B mean_(N,E) |LN(Φ_Π(Z; u)) − LN(Z_target(T))|，只监督终点，不除 c(Π_k)²、不按 k 加权；二等分变体逐位不变。
全部是 CPU 上的小模型（test_s28b_dyadic_zero_init 的 make_model / make_batch）。
"""
import copy
import math

import pytest
import torch
import yaml
from torch.nn import functional as F

from po_jepa import losses
from po_jepa.actions import _exact_split_slice, select_action_rows
from po_jepa.geometry import regular_grid
from po_jepa.rollout_supervision import rollout_predictions, rollout_supervision_loss
from po_jepa.runtime import checks, comparison, training
from po_jepa.runtime.addon import calibrate, calibration, conditions
from test_s28b_dyadic_zero_init import config, halves, make_batch, make_controls, make_model, make_z, v7  # noqa: F401

DURATIONS = [1., 2., 4., 8., 2., 4.]
LR, LS = .375, .25


def hand_rollout(composed, target, eps=1e-5):
    """手写 L_roll：两侧沿 E 无 affine LN，|差| 在 (N,E) 上取均值；逐样本 [B]，转 float64。"""
    a = F.layer_norm(composed, (composed.shape[-1],), eps=eps)
    b = F.layer_norm(target.detach(), (target.shape[-1],), eps=eps)
    return (a-b).abs().mean(dim=(1, 2)).to(torch.float64)


def hand_partition_rollout(model, z, controls, duration, levels, grid):
    """逐行单独算 2^k 段滚动（不分组）：第 i 段动作是 [i·h, (i+1)·h) 的精确交集，输入是上一段的 raw 输出。"""
    rows = []
    for b, k in enumerate(levels.tolist()):
        state = z[b:b+1]
        path = select_action_rows(controls, [b])
        h = duration[b:b+1]/2**k
        for i in range(2**k):
            state = model(state, _exact_split_slice(path, start=h*i, duration=h), h.clone(), grid)
        rows.append(state)
    return torch.cat(rows)


def mixed_step(batch, *, start=0):
    """找一个这批抽到的 k 不全相同、且含 k = 3 的更新序号（让分组和 8 段都被走到）。"""
    for step in range(start, start+200):
        levels = losses.dyadic_levels(batch.duration, batch.sample_ids, step=step)
        if len(set(levels.tolist())) > 1 and 3 in levels.tolist():
            return step, levels
    raise AssertionError("no mixed step found")


def rollout_args(batch):
    return batch.raw_z, batch.controls, batch.tau, batch.sigma, batch.transfer.source


# ---------- 定义：路径、目标、归一化 ----------

def test_dyadic_rollout_is_the_lemma_partition_against_endpoint_truth():
    """composed 就是引理 5.4 的 Φ_{Π_k}（逐行手算一致）；L_roll 是它对终点真值的 LN-L1，不除 c(Π_k)²。"""
    model, batch = make_model(), make_batch(DURATIONS)
    cfg = config("dyadic", time_weight=LR, space_weight=LS)
    step, levels = mixed_step(batch)
    predictions = rollout_predictions(model, *rollout_args(batch), config=cfg, step=step, sample_ids=batch.sample_ids)
    assert isinstance(predictions, losses.PartitionPredictions) and torch.equal(predictions.levels, levels)
    grid = batch.transfer.source
    hand = hand_partition_rollout(model, batch.raw_z, batch.controls, batch.duration, levels, grid)
    torch.testing.assert_close(predictions.composed, hand, rtol=1e-4, atol=1e-4)   # 分组只差批大小带来的 float32 舍入
    # direct 和二等分的长支逐位相同（JEPA 项不变）
    assert torch.equal(predictions.direct, losses.temporal_predictions(model, *rollout_args(batch)).direct)
    loss = rollout_supervision_loss(predictions, batch.raw_target, config=cfg)
    assert loss.per_sample.dtype == loss.mean.dtype == torch.float64 and loss.per_sample.shape == (len(DURATIONS),)
    assert torch.equal(loss.per_sample, hand_rollout(predictions.composed, batch.raw_target))
    assert torch.equal(loss.per_sample, losses.jepa_distance(predictions.composed, batch.raw_target, config=cfg).per_sample.double())
    assert loss.mean == loss.per_sample.mean()
    # 不除 c(Π_k)²、不按 k 加权：只改记录里的 T 和 k（预测不变），损失逐位不变
    relabeled = losses.PartitionPredictions(predictions.direct, predictions.composed, predictions.difference,
                                            predictions.duration*2, torch.ones_like(predictions.levels), predictions.input_scale)
    assert torch.equal(rollout_supervision_loss(relabeled, batch.raw_target, config=cfg).per_sample, loss.per_sample)
    # 比较对象是真值，不是长支（那是时间一致性）
    assert not torch.equal(loss.per_sample, hand_rollout(predictions.direct, batch.raw_target))


def test_same_levels_and_path_as_po_jepa_full_time_consistency():
    """同一步、同一批：rollout_sup 抽到的 k、组合路径和 po_jepa_full 的多段切分时间一致性逐位相同。"""
    model, batch = make_model(), make_batch(DURATIONS)
    cfg = config("dyadic")
    for step in (0, 5, 37, mixed_step(batch)[0]):
        mine = rollout_predictions(model, *rollout_args(batch), config=cfg, step=step, sample_ids=batch.sample_ids)
        full, _ = losses.time_consistency(model, *rollout_args(batch), config=cfg, step=step, sample_ids=batch.sample_ids)
        assert torch.equal(mine.levels, full.levels) and torch.equal(mine.composed, full.composed)
        assert torch.equal(mine.direct, full.direct)


def test_bisection_variant_is_bitwise_the_s27a_two_segment_rollout():
    """默认变体：rollout_predictions 就是 temporal_predictions（3 次 Φ），step、sample_ids 不起作用。"""
    model, batch = make_model(), make_batch(DURATIONS)
    cfg = config()
    old = losses.temporal_predictions(model, *rollout_args(batch))
    for step in (None, 0, 9):
        new = rollout_predictions(model, *rollout_args(batch), config=cfg, step=step, sample_ids=batch.sample_ids)
        assert isinstance(new, losses.TemporalPredictions)
        assert torch.equal(new.composed, old.composed) and torch.equal(new.direct, old.direct)
        assert torch.equal(rollout_supervision_loss(new, batch.raw_target, config=cfg).per_sample,
                           rollout_supervision_loss(old, batch.raw_target, config=cfg).per_sample)


def test_k1_dyadic_rollout_equals_the_two_segment_rollout():
    """k = 1 的 2^k 段滚动和两段滚动逐位相同，所以 L_roll 也相同（S27a 是多段切分版 k = 1 的特例）。"""
    model, durations = make_model(), [2., 4., 8.]
    z, controls = make_z(3), make_controls(durations)
    tau, sigma = halves(durations)
    grid = regular_grid(8, 8)
    target = torch.roll(z, 1, -1)*.8+.03
    cfg = config("dyadic")
    two = losses.temporal_predictions(model, z, controls, tau, sigma, grid)
    one = losses.partition_predictions(model, z, controls, tau, sigma, torch.ones(3, dtype=torch.int64), grid)
    assert torch.equal(rollout_supervision_loss(two, target, config=cfg).per_sample,
                       rollout_supervision_loss(one, target, config=cfg).per_sample)


def test_gradient_reaches_every_segment_and_target_stays_stopped():
    """梯度经全部 2^k 段传回：第一段的输入动作也有梯度；真值停梯度；k = 3 那行改第一段动作会改 L_roll。"""
    model, batch = make_model(), make_batch([8., 8.])
    cfg = config("dyadic")
    step = next(s for s in range(200) if losses.dyadic_levels(batch.duration, batch.sample_ids, step=s).tolist() == [3, 3])
    controls = batch.controls
    values = controls.u_values.detach().clone().requires_grad_()
    controls = type(controls)(**{**controls.__dict__, "u_values": values})
    target = batch.raw_target.clone().requires_grad_()
    predictions = rollout_predictions(model, batch.raw_z, controls, batch.tau, batch.sigma, batch.transfer.source,
                                      config=cfg, step=step, sample_ids=batch.sample_ids)
    loss = rollout_supervision_loss(predictions, target, config=cfg)
    loss.mean.backward()
    assert target.grad is None
    first = values.grad[:, 0]                       # 第一段 [0, 0.75) 的动作只进第 1 段 Φ
    assert torch.isfinite(values.grad).all() and first.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_contract_errors():
    model, batch = make_model(), make_batch(DURATIONS)
    with pytest.raises(ValueError, match="^dyadic temporal partition needs the explicit update index step$"):
        rollout_predictions(model, *rollout_args(batch), config=config("dyadic"), step=None, sample_ids=batch.sample_ids)
    with pytest.raises(TypeError, match="^config must be a LossConfig$"):
        rollout_predictions(model, *rollout_args(batch), config=None, step=0, sample_ids=batch.sample_ids)
    predictions = rollout_predictions(model, *rollout_args(batch), config=config("dyadic"), step=0, sample_ids=batch.sample_ids)
    with pytest.raises(ValueError, match="^rollout target must have exactly the composed prediction shape$"):
        rollout_supervision_loss(predictions, batch.raw_target[:2], config=config("dyadic"))
    with pytest.raises(TypeError, match="^predictions must be TemporalPredictions or PartitionPredictions$"):
        rollout_supervision_loss(object(), batch.raw_target, config=config("dyadic"))


# ---------- 单步、梯度范数、监控 ----------

def count_phi(model, monkeypatch):
    calls = []
    real = type(model).forward
    monkeypatch.setattr(type(model), "forward", lambda self, *a, **k: calls.append(1) or real(self, *a, **k))
    return calls


@pytest.fixture
def one_thread():
    """混合 k、8 段深的图在 CPU 多线程下反传不逐位可复现（同一函数连算两次，omega_head 的梯度差到 ~1e-3）；
    单线程下逐位可复现，梯度核对才能用紧容差。"""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_step_under_dyadic_matches_hand_total_and_counts(monkeypatch, one_thread):
    """一次更新：日志六项等于手算（k 按 completed_steps 抽）；梯度等于手写总损失的梯度；Φ 次数 = temporal_phi_calls + 2。"""
    model, batch = make_model(), make_batch(DURATIONS)
    cfg = config("dyadic", time_weight=LR, space_weight=LS)
    step, levels = mixed_step(batch)
    reference = copy.deepcopy(model)
    optimizer = comparison.make_optimizer(model)
    captured = {}
    real_step = training.scheduled_step
    monkeypatch.setattr(training, "scheduled_step",
                        lambda *a: captured.update(grads=[p.grad.clone() for p in model.parameters()]) or real_step(*a))
    calls = count_phi(model, monkeypatch)
    result = comparison.condition_step("po_jepa_rollout_sup", model, optimizer, batch, cfg, step)
    phi = 1+sum(2**k for k in set(levels.tolist()))   # direct 一次，每组 2^k 段
    assert len(calls) == losses.temporal_phi_calls(cfg, batch.duration, batch.sample_ids, step)+2 == phi+2
    monkeypatch.undo()
    grid = batch.transfer.source
    partition = losses.partition_predictions(reference, *rollout_args(batch)[:4], levels, grid)
    jepa = losses.jepa_distance(partition.direct, batch.raw_target, config=cfg).mean
    roll = hand_rollout(partition.composed, batch.raw_target).mean()
    space = losses.spatial_consistency_loss(losses.spatial_predictions(reference, batch.raw_z, batch.controls, batch.duration,
                                                                       batch.transfer), config=cfg).mean
    values = result.loss_values
    assert result.completed_steps == step+1
    assert values["jepa"] == jepa.item() and values["time"] == roll.item() and values["space"] == space.item()
    assert values["weighted_time"] == LR*roll.item() and values["weighted_space"] == LS*space.item()
    (jepa.double()+LR*roll+LS*space).backward()
    for (name, p), got in zip(reference.named_parameters(), captured["grads"]):
        torch.testing.assert_close(got, p.grad, rtol=1e-6, atol=1e-10, msg=lambda m, n=name: f"{n}: {m}")


def test_step_levels_follow_completed_steps():
    """同一个起点，用两个抽到不同 k 的步序号更新，L_roll 不同；和 k 按那一步手算的值一致。"""
    batch = make_batch(DURATIONS)
    cfg = config("dyadic", time_weight=LR, space_weight=LS)
    a, la = mixed_step(batch)
    b, lb = next((s, l) for s in range(a+1, a+200)
                 if not torch.equal(l := losses.dyadic_levels(batch.duration, batch.sample_ids, step=s), la))
    base = make_model()
    out = {}
    for step, levels in ((a, la), (b, lb)):
        model = copy.deepcopy(base)
        model.train()
        result = comparison.condition_step("po_jepa_rollout_sup", model, comparison.make_optimizer(model), batch, cfg, step)
        p = losses.partition_predictions(copy.deepcopy(base), *rollout_args(batch)[:4], levels, batch.transfer.source)
        assert result.loss_values["time"] == hand_rollout(p.composed, batch.raw_target).mean().item()
        out[step] = result.loss_values["time"]
    assert out[a] != out[b]


def test_observed_update_readouts_under_dyadic():
    """读出计数核对（observed_update）在多段切分下通过：(temporal_phi_calls + 2) × 2。"""
    model, batch = make_model(), make_batch(DURATIONS)
    cfg = config("dyadic", time_weight=LR, space_weight=LS)
    step, levels = mixed_step(batch)
    expected = conditions.expected_readouts("po_jepa_rollout_sup", model, loss_config=cfg, batch=batch, step=step)
    assert expected == (1+sum(2**k for k in set(levels.tolist()))+2)*checks.readouts_per_phi(model)
    result, scales = conditions.observed_update("po_jepa_rollout_sup", model, comparison.make_optimizer(model), batch, cfg, step)
    assert len(scales) == expected and result.completed_steps == step+1


def test_rollout_gradient_norms_use_the_step_levels(one_thread):
    """rollout_gradient_norms(step=s) 量的 L_roll 是第 s 步那串 k 的；多段切分不给 step 报错。"""
    model, batch = make_model(), make_batch(DURATIONS)
    model.train()
    cfg = config("dyadic", time_weight=LR, space_weight=LS)
    step, levels = mixed_step(batch)
    norms = calibration.rollout_gradient_norms(model, batch, loss_config=cfg, step=step)
    p = losses.partition_predictions(model, *rollout_args(batch)[:4], levels, batch.transfer.source)
    roll = hand_rollout(p.composed, batch.raw_target).mean()
    assert norms["time"] == roll.item()
    grads = torch.autograd.grad(roll, list(model.parameters()), allow_unused=True)
    assert norms["grad_time"] == pytest.approx(math.sqrt(sum(float(g.double().square().sum()) for g in grads if g is not None)),
                                               rel=1e-12)
    with pytest.raises(ValueError, match="explicit update index step"):
        calibration.rollout_gradient_norms(model, batch, loss_config=cfg)


def test_monitoring_and_calibration_pass_the_step(monkeypatch):
    """consistency_share 传 completed − 1（这一步训练用的 k），calibrate-addon 传批在查询序列里的序号。"""
    seen = []
    fake = dict(jepa=1., time=1., space=1., grad_jepa=1., grad_time=1., grad_space=1.)
    monkeypatch.setattr(calibration, "rollout_gradient_norms", lambda m, b, *, loss_config, step=None: seen.append(step) or fake)
    interval = checks.load_protocol()["monitoring"]["consistency_interval"]
    cfg = config("dyadic", time_weight=LR, space_weight=LS)
    assert conditions.consistency_share("po_jepa_rollout_sup", None, None, cfg, 3*interval) is not None
    calibrate._gradient_norms("po_jepa_rollout_sup", None, None, cfg, 302)
    assert seen == [3*interval-1, 302]


# ---------- 标签和标定 ----------

def test_snapshot_labels_the_rollout_partition():
    """rollout_sup 的 loss_config 快照：多段切分写 temporal_partition，二等分不写（S27a 检查点格式不变），两者不等。"""
    dyadic = training._loss_config(config("dyadic"), "rollout_sup")
    bisection = training._loss_config(config(), "rollout_sup")
    assert dyadic["temporal_partition"] == "dyadic" and "temporal_partition" not in bisection
    assert not training._same_native(dyadic, bisection)


def test_build_and_pretrain_follow_variant(v7, monkeypatch):  # noqa: F811
    monkeypatch.setattr(calibration, "load_addon_weights", lambda c: dict(time_weight=.3, space_weight=.4))
    for partition in ("bisection", "dyadic"):
        monkeypatch.setenv(checks.PARTITION_ENV, partition)
        _, loss = conditions.build("po_jepa_rollout_sup", torch.device("cpu"))
        _, base = comparison.build_condition("po_jepa_full", torch.device("cpu"))
        assert loss.temporal_partition == base.temporal_partition == partition and loss.time_weight == .3
        _, pre = calibrate._jepa_only("po_jepa_rollout_sup", torch.device("cpu"))
        assert pre.temporal_partition == partition and pre.time_weight == pre.space_weight == 0.


@pytest.fixture
def addon_files(monkeypatch, tmp_path):
    from po_jepa.runtime import workflow as w
    v7file = tmp_path/"v7.yaml"
    v7file.write_text("time_weight: 1.0\n", encoding="utf-8")
    path = tmp_path/"addon.yaml"
    state = dict(partition="bisection")
    monkeypatch.setattr(checks, "v7_calibration_path", lambda: v7file)
    monkeypatch.setattr(calibration, "addon_calibration_path", lambda: path)
    monkeypatch.setattr(checks, "load_protocol", lambda: dict(calibration=dict(rho=1.), model=dict(graph=dict(sigma_z=.5)),
                                                              loss=dict(space_weight=4., temporal_partition=state["partition"])))

    def write(rollout=None, **changes):
        value = dict(fixed_fourier=dict(rho=1., space_weight=.5, time_weight=.25, source="fictional"),
                     transformer_consistency=dict(rho=1., space_weight=2, time_weight=1e-3, source="fictional"),
                     po_jepa_rollout_sup=dict(rho=1., space_weight=4., time_weight=.75, source="fictional", **(rollout or {})),
                     sigma_z_used=.5, v7_calibration_sha256=w.cache.file_sha(v7file))
        value.update(changes)
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return write, state


def test_dyadic_addon_calibration_needs_the_rollout_partition_marker(addon_files):
    """多段切分变体的补跑标定：rollout_sup 一项必须带 rollout_partition: dyadic；没有（S28d 以前两段滚动标的）就拒绝。
    二等分变体格式不变，带了这个键反而拒绝。别的条件不受影响。"""
    write, state = addon_files
    write()
    assert calibration.load_addon_weights("po_jepa_rollout_sup") == dict(time_weight=.75, space_weight=4.)
    write(rollout=dict(rollout_partition="dyadic"))
    with pytest.raises(ValueError, match="^add-on calibration entry po_jepa_rollout_sup must contain exactly rho, space_weight, time_weight, source$"):
        calibration.load_addon_weights("po_jepa_rollout_sup")
    state["partition"] = "dyadic"
    assert calibration.load_addon_weights("po_jepa_rollout_sup") == dict(time_weight=.75, space_weight=4.)
    assert calibration.load_addon_weights("fixed_fourier") == dict(time_weight=.25, space_weight=.5)
    write()
    with pytest.raises(ValueError, match="S28d 以前的代码"):
        calibration.load_addon_weights("po_jepa_rollout_sup")
    with pytest.raises(ValueError, match="S28d 以前的代码"):
        calibration.require_addon_calibration()
    write(rollout=dict(rollout_partition="bisection"))
    with pytest.raises(ValueError, match="rollout_partition is 'bisection'"):
        calibration.load_addon_weights("po_jepa_rollout_sup")
    write(fixed_fourier=dict(rho=1., space_weight=.5, time_weight=.25, source="fictional", rollout_partition="dyadic"),
          rollout=dict(rollout_partition="dyadic"))
    with pytest.raises(ValueError, match="^add-on calibration entry fixed_fourier must contain exactly rho, space_weight, time_weight, source$"):
        calibration.load_addon_weights("fixed_fourier")


def test_calibrate_addon_writes_the_marker_only_under_dyadic(monkeypatch, tmp_path):
    """calibrate_addon 写文件：多段切分时 rollout_sup 一项加 rollout_partition: dyadic，二等分时和以前逐键相同；写完读回核对通过。"""
    from po_jepa.runtime import workflow as w
    v7file = tmp_path/"v7.yaml"
    v7file.write_text("time_weight: 1.0\n", encoding="utf-8")
    for partition in ("bisection", "dyadic"):
        path = tmp_path/f"addon-{partition}.yaml"
        protocol = dict(calibration=dict(rho=1., pretrain_updates=300, batches=32), model=dict(graph=dict(sigma_z=.5)),
                        loss=dict(space_weight=4., temporal_partition=partition))
        monkeypatch.setattr(checks, "load_protocol", lambda: protocol)
        monkeypatch.setattr(checks, "v7_calibration_path", lambda: v7file)
        monkeypatch.setattr(checks, "v7_updates", lambda: 2000)
        monkeypatch.setattr(calibration, "addon_calibration_path", lambda: path)
        monkeypatch.setattr(calibrate, "require_base_calibration", lambda base: w.cache.file_sha(v7file))
        monkeypatch.setattr(calibrate, "calibrate_condition", lambda c, f, d: dict(
            rho=1., time_weight=dict(fixed_fourier=.25, transformer_consistency=1e-3, po_jepa_rollout_sup=.75)[c],
            space_weight=dict(fixed_fourier=.5, transformer_consistency=2., po_jepa_rollout_sup=4.)[c]))
        monkeypatch.setattr(calibrate.data.datasets, "active", lambda: type("A", (), dict(tag="double_well-200"))())
        monkeypatch.setattr(calibrate, "OVERLAY", v7file)
        output = tmp_path/f"out-{partition}"
        output.mkdir()
        payload = calibrate.calibrate_addon(tmp_path, output, torch.device("cpu"), base="local")
        written = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert written == payload
        entry = written["po_jepa_rollout_sup"]
        if partition == "dyadic":
            assert entry.pop("rollout_partition") == "dyadic"
        assert set(entry) == calibration.ENTRY_KEYS
        assert all(set(written[c]) == calibration.ENTRY_KEYS for c in ("fixed_fourier", "transformer_consistency"))


def test_c2_job_file_note_points_to_the_new_rollout():
    """S28c 的 c2 任务文件：多段切分变体开头的说明改成指向 S28d（不再是“等 Jerry 定”）；二等分变体没有这一行。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(checks.ROOT)/"tools"))
    import make_small_validation_jobs as jobs
    notes = [l for l in jobs.job_lines("c2", partition="dyadic", init="zero") if "po_jepa_rollout_sup" in l and l.startswith("# 注意")]
    assert len(notes) == 1 and "2^k 段有真值滚动" in notes[0] and "S28d" in notes[0] and "等 Jerry 定" not in notes[0]
    assert not [l for l in jobs.job_lines("c2", partition="bisection", init="xavier") if l.startswith("# 注意")]
