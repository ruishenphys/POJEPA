"""S28b A 层：论文式子对代码（方案 §5.1 的表）。

已有、这里不重复的：命题 4.1 的闭式 e^{Aτ}、‖e^{Aτ}‖₂ = e^{−γτ} ≤ 1、半群 e^{A(τ+σ)} = e^{Aσ}e^{Aτ}
（test_dynamics.py::test_damped_rotation_matches_matrix_exp_and_fixed_semigroup、
::test_propagation_recovers_blocks_and_nonexpansive_spectral_norm）；残差读出 τ = 0 时 Ẑ = Z 逐位
（test_v4.py::test_residual_readout_is_exact_identity_at_zero_duration）；零初始化后任意 τ 都 Ẑ = Z、
多段切分归一化 c(Π_k)² 和 k = 1 逐位相同（test_s28b_dyadic_zero_init.py）。

这里补：驱动对动作线性、a = 0 且 F = 0 时退化为自由流、TCD/SCD 评测式、引理 5.3/5.4 和联合时空界的实测。
"""
import math

import pytest
import torch

from po_jepa import losses, metrics, theory_bounds
from po_jepa.geometry import apply_spatial_transfer, build_spatial_transfer, regular_grid
from po_jepa.runtime import comparison

from test_s28b_dyadic_zero_init import make_controls, make_model, make_z, halves

TOLERANCE = 1e-9   # 界里的三角不等式在 float64 范数上成立；只留舍入余量


def small_case(durations=(2., 2.), seed=0):
    model = make_model()
    model.eval()
    return model, make_z(len(durations), seed=seed), make_controls(list(durations), seed=seed), regular_grid(8, 8)


# ---------- 驱动：B 对动作线性；a = 0 且 F = 0 时是自由流 ----------

def test_token_drive_is_linear_in_action():
    """Ċ = AC + Ψᵀ[B(Z, a) + F(ΨC, Z, a)]，B(Z, αa) = αB(Z, a)（附录 interaction：B 在每个 token 上对 a 线性）。"""
    model, z, _, grid = small_case()
    query = model.prepare_query(z, grid)
    c = query.initial_coefficients
    a = torch.tensor([[.3, -.2], [-.1, .4]])
    base = model.dynamics.token_forcing(c, a, query.dynamics).action_linear
    for alpha in (-2., .5, 3.):
        scaled = model.dynamics.token_forcing(c, alpha*a, query.dynamics).action_linear
        assert torch.allclose(scaled, alpha*base, rtol=1e-5, atol=1e-7)
    zero = model.dynamics.token_forcing(c, torch.zeros_like(a), query.dynamics).action_linear
    assert torch.count_nonzero(zero) == 0
    # 两个动作的驱动可叠加
    other = torch.tensor([[-.2, .1], [.25, -.3]])
    both = model.dynamics.token_forcing(c, a+other, query.dynamics).action_linear
    assert torch.allclose(both, base+model.dynamics.token_forcing(c, other, query.dynamics).action_linear, rtol=1e-5, atol=1e-7)


def test_zero_action_and_zero_interaction_reduce_to_free_flow():
    """a = 0、F = 0（交互网络输出清零）时，Φ_τ 的模态系数恰为自由流 C(τ) = e^{Aτ}C(0)（命题 4.1 的闭式，按通道对旋转衰减）。"""
    model, z, _, grid = small_case()
    with torch.no_grad():
        for layer in (model.dynamics.token_v, model.dynamics.token_residual_net[2]):
            layer.weight.zero_()
            if layer.bias is not None:
                layer.bias.zero_()
    controls = make_controls([2., 2.])
    controls = type(controls)(controls.starts, controls.ends, torch.where(controls.segment_valid[..., None],
        torch.zeros_like(controls.u_values), controls.u_values), controls.n_segments, controls.segment_valid,
        controls.source_origin_seconds, controls.duration, controls.control_spec, controls.source_ids)
    tau = 1.75
    query = model.prepare_query(z, grid)
    result = model.integrate_internal_continuation(query, query.initial_coefficients, controls,
                                                   torch.full((2,), tau, dtype=torch.float64))
    free = query.dynamics.coefficients
    gamma, omega = free.gamma.double(), free.omega.double()                  # [B,K,32]
    c0 = query.initial_coefficients.double().reshape(*gamma.shape, 2)
    decay = torch.exp(-gamma*tau)
    cos, sin = torch.cos(omega*tau), torch.sin(omega*tau)
    expected = torch.stack((decay*(cos*c0[..., 0]-sin*c0[..., 1]), decay*(sin*c0[..., 0]+cos*c0[..., 1])), -1)
    assert torch.allclose(result.final.double(), expected.reshape_as(result.final), rtol=1e-5, atol=1e-6)
    # 非扩张：‖C(τ)‖ ≤ ‖C(0)‖
    assert torch.all(result.final.double().norm(dim=-1) <= query.initial_coefficients.double().norm(dim=-1)+1e-6)


# ---------- TCD / SCD 评测式 ----------

def test_tcd_scd_normalized_match_paper_definitions():
    """§6：TCD(τ,σ) = sqrt(δ_h(Φ_{τ+σ}(z), Φ_σ(Φ_τ(z))))/(τσ)，SCD = sqrt(δ_{h'}(RΦ^h_τ(z), Φ^{h'}_τ(Rz)))/τ；
    评测代码的 tcd_normalized = tcd_rms/s、scd_normalized = scd_rms/s（runtime/evaluation.py）。"""
    model, z, controls, grid = small_case((2., 4.))
    tau, sigma = halves([2., 4.])
    transfer = build_spatial_transfer(grid, target_height=4, target_width=4)
    with torch.no_grad():
        temporal = losses.temporal_predictions(model, z, controls, tau, sigma, grid)
        spatial = losses.spatial_predictions(model, z, controls, tau+sigma, transfer)
        s = losses.input_scale(z).sqrt()
        tcd = metrics.temporal_composition_rms_defect(temporal).values/s
        scd = metrics.spatial_commutation_rms_defect(spatial).values/s
        long = model(z, controls, tau+sigma, grid)
        delta_t = theory_bounds.scaled_norm(long, temporal.composed, losses.input_scale(z))
        restricted = apply_spatial_transfer(long, transfer)
        coarse = model(apply_spatial_transfer(z, transfer), controls, tau+sigma, transfer.target)
        delta_s = theory_bounds.scaled_norm(restricted, coarse, losses.input_scale(z))
    assert torch.allclose(tcd, delta_t/(tau*sigma), rtol=1e-5)
    assert torch.allclose(scd, delta_s/(tau+sigma), rtol=1e-5)
    # 原始 Frobenius 版本（TCD_raw）是 sqrt(Σ D²)/(τσ)，与 normalized 只差 sqrt(N·E)·s
    raw = metrics.temporal_composition_defect(temporal).values
    assert torch.allclose(raw, tcd*s*math.sqrt(z.shape[1]*z.shape[2]), rtol=1e-6)
    # evaluation.py 里的写法（单样本批、取 [0]）和这里的定义相同
    one = comparison.input_scale(z[:1])[0]
    assert math.isclose(float(metrics.temporal_composition_rms_defect(temporal).values[0])/math.sqrt(float(one)),
                        float(delta_t[0]/(tau[0]*sigma[0])), rel_tol=1e-5)


# ---------- 引理 5.4、5.3 和联合时空界：实测左边 ≤ 右边 ----------

def hand_temporal(partition, eps, lip):
    """引理 5.4 右边逐项手算：ε_t Σ_{i≥2} t_{i−1}τ_i e^{L(T−t_i)}。"""
    t = [sum(partition[:i]) for i in range(len(partition)+1)]
    return eps*sum(t[i-1]*partition[i-1]*math.exp(lip*(t[-1]-t[i])) for i in range(2, len(t)))


def hand_spatial(partition, eps, lip):
    """引理 5.3 右边逐项手算：ε_s Σ_{i≥1} τ_i e^{L(T−t_i)}。"""
    t = [sum(partition[:i]) for i in range(len(partition)+1)]
    return eps*sum(partition[i-1]*math.exp(lip*(t[-1]-t[i])) for i in range(1, len(t)))


def test_right_hand_sums_match_hand_values():
    """Codex S28b P3-1：非等分、非零 L 下，右端求和逐项和手算相同（不只是不等式方向）。
    Π = (0.25, 0.75, 1)，L = 0.7：时间 = ε_t(0.25·0.75·e^{0.7} + 1·1·e^0)，空间 = ε_s(0.25e^{1.225} + 0.75e^{0.7} + 1)。"""
    times = theory_bounds.partition_times((.25, .75, 1.))
    assert times == [0., .25, 1., 2.]
    assert math.isclose(theory_bounds.temporal_sum(times, 1.3, .7), 1.3*(.25*.75*math.exp(.7)+1.), rel_tol=1e-14)
    assert math.isclose(theory_bounds.spatial_sum(times, 1.3, .7),
                        1.3*(.25*math.exp(.7*1.75)+.75*math.exp(.7)+1.), rel_tol=1e-14)
    assert math.isclose(theory_bounds.temporal_sum(times, 1.3, .7), hand_temporal((.25, .75, 1.), 1.3, .7), rel_tol=1e-14)
    assert theory_bounds.temporal_sum([0., 2.], 5., 1.) == 0.          # 一段：没有 i ≥ 2 的项


@pytest.mark.parametrize("partition", [(1., 1.), (.5, .5, .5, .5), (.25,)*8, (.5, 1.5), (.25, .75, 1.)])
def test_lemma_temporal_partition_bound(partition):
    """引理 5.4：‖Φ_T(z) − Φ_Π(z)‖ ≤ ε_t Σ_{i≥2} t_{i−1}τ_i e^{L(T−t_i)} ≤ ε_t e^{LT}(T² − Σ τ_i²)/2，
    ε_t、L 按证明用到的量在同一批上取上确界。"""
    model, z, controls, grid = small_case()
    bound = theory_bounds.temporal_partition_bound(model, z, controls, partition, grid)
    assert all(math.isfinite(v) for v in (bound["sharp"], bound["coarse"], bound["eps_t"], bound["L"]))
    for lhs in bound["lhs"]:
        assert lhs <= bound["sharp"]*(1+TOLERANCE)
    assert bound["sharp"] <= bound["coarse"]*(1+TOLERANCE)
    # 右端精确值：用返回的 ε_t、L 逐项重算（式 temporal_accumulation_sharp / _coarse）
    T = sum(partition)
    assert math.isclose(bound["sharp"], hand_temporal(partition, bound["eps_t"], bound["L"]), rel_tol=1e-12, abs_tol=0)
    assert math.isclose(bound["coarse"], bound["eps_t"]*math.exp(bound["L"]*T)*(T*T-sum(x*x for x in partition))/2,
                        rel_tol=1e-12, abs_tol=0)
    if len(partition) > 1:
        assert min(bound["lhs"]) > 0 and bound["eps_t"] > 0   # 不是空洞的 0 ≤ 0
    else:
        assert bound["lhs"] == [0., 0.]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_training_composed_path_is_the_lemma_partition(k):
    """训练用的多段切分组合支路就是引理 5.4 里的 Φ_Π（2^k 等分）：逐位相同；它和直接支路的距离就是引理左边。"""
    model, z, controls, grid = small_case()
    tau, sigma = halves([2., 2.])
    with torch.no_grad():
        trained = losses.partition_predictions(model, z, controls, tau, sigma, torch.full((2,), k), grid)
    bound = theory_bounds.temporal_partition_bound(model, z, controls, (2./2**k,)*2**k, grid)
    assert torch.equal(trained.composed, bound["rollout"])
    # 训练的 direct 用 u1⊕u2（在 T/2 处切开再拼），引理用 u_[0,T)：同一条动作路径，数值上一致
    assert torch.allclose(trained.direct, bound["direct"], rtol=1e-5, atol=1e-5)
    lhs = theory_bounds.scaled_norm(trained.direct, trained.composed, losses.input_scale(z))
    assert torch.allclose(lhs, torch.tensor(bound["lhs"], dtype=torch.float64), rtol=1e-4)


@pytest.mark.parametrize("partition", [(2.,), (1., 1.), (.5, .5, .5, .5), (.25, .75, 1.)])
def test_lemma_spatial_accumulation_bound(partition):
    """引理 5.3：‖RΦ^h_Π(z) − Φ^{h'}_Π(Rz)‖ ≤ ε_s Σ_i τ_i e^{L(T−t_i)} ≤ ε_s T e^{LT}（8×8 → 4×4 池化）。"""
    model, z, controls, grid = small_case()
    transfer = build_spatial_transfer(grid, target_height=4, target_width=4)
    bound = theory_bounds.spatial_partition_bound(model, z, controls, partition, transfer)
    assert min(bound["lhs"]) > 0 and bound["eps_s"] > 0
    for lhs in bound["lhs"]:
        assert lhs <= bound["sharp"]*(1+TOLERANCE)
    assert bound["sharp"] <= bound["coarse"]*(1+TOLERANCE)
    T = sum(partition)
    assert math.isclose(bound["sharp"], hand_spatial(partition, bound["eps_s"], bound["L"]), rel_tol=1e-12)
    assert math.isclose(bound["coarse"], bound["eps_s"]*T*math.exp(bound["L"]*T), rel_tol=1e-12)


@pytest.mark.parametrize("partition,other", [((.5, .5, .5, .5), (1., 1.)), ((1., 1.), (.5, 1.5)), ((.25,)*8, (2.,))])
def test_theorem_joint_space_time_bound(partition, other):
    """联合时空界：‖RΦ^h_Π(z) − Φ^{h'}_{Π'}(Rz)‖ ≤ 空间(Π) + 时间(Π) + 时间(Π')（sharp）
    ≤ e^{LT}[ε_s T + ε_t(2T² − Στ² − Στ'²)/2] ≤ e^{LT}(ε_s T + ε_t T²)，三项共用 ε_s、ε_t、L。"""
    model, z, controls, grid = small_case()
    transfer = build_spatial_transfer(grid, target_height=4, target_width=4)
    bound = theory_bounds.joint_bound(model, z, controls, partition, other, transfer)
    assert min(bound["lhs"]) > 0
    for lhs in bound["lhs"]:
        assert lhs <= bound["sharp"]*(1+TOLERANCE)
    assert bound["sharp"] <= bound["coarse"]*(1+TOLERANCE) <= bound["simple"]*(1+TOLERANCE)**2
    # 右端精确值（式 joint_bound_sharp、main_bound、main_bound_simple），三项共用返回的 ε_s、ε_t、L
    es, et, lip, T = bound["eps_s"], bound["eps_t"], bound["L"], sum(partition)
    assert math.isclose(bound["sharp"], hand_spatial(partition, es, lip)+hand_temporal(partition, et, lip)+hand_temporal(other, et, lip),
                        rel_tol=1e-12)
    assert math.isclose(bound["coarse"], math.exp(lip*T)*(es*T+et/2*(2*T*T-sum(x*x for x in partition)-sum(x*x for x in other))),
                        rel_tol=1e-12)
    assert math.isclose(bound["simple"], math.exp(lip*T)*(es*T+et*T*T), rel_tol=1e-12)


def test_bounds_hold_after_training_updates():
    """训练几步之后（参数不再是初始化）三个界照样成立；零初始化起点上三个左边都恰为 0。"""
    from po_jepa.runtime import training
    from test_s28b_dyadic_zero_init import config, make_batch
    model, batch = make_model("zero"), make_batch([2., 2.])
    grid = regular_grid(8, 8)
    transfer = batch.transfer
    with torch.no_grad():
        zero = theory_bounds.joint_bound(model, batch.raw_z, batch.controls, (.5,)*4, (1., 1.), transfer)
    assert zero["lhs"] == [0., 0.]
    model.train()
    optimizer = training.make_optimizer(model)
    for step in range(4):
        training.train_step(model, optimizer, batch, loss_config=config("dyadic"), completed_steps=step)
    model.eval()
    temporal = theory_bounds.temporal_partition_bound(model, batch.raw_z, batch.controls, (.5,)*4, grid)
    spatial = theory_bounds.spatial_partition_bound(model, batch.raw_z, batch.controls, (.5,)*4, transfer)
    joint = theory_bounds.joint_bound(model, batch.raw_z, batch.controls, (.5,)*4, (1., 1.), transfer)
    for bound in (temporal, spatial, joint):
        assert max(bound["lhs"]) > 0
        assert all(lhs <= bound["sharp"]*(1+TOLERANCE) for lhs in bound["lhs"])
