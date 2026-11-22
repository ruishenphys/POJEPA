"""C3：空间基底数 K 消融，全部不丢信息（Jerry 2026-11-22）：K16c/K32c/K64c（前 K 个模态 + 补空间整体保留）和 K 取满。

对照：方案 `small-validation-results/c3-plan/2026-11-22_mode-ablation-plan.md` §1、§3；诊断报告
`small-validation-results/c2/2026-11-22_direct-prediction-diagnosis.md` §5。方案 B：
    Y = Psi C + R,  P_perp = I - Psi Psi^T,  C(0) = Psi^T X,  R(0) = P_perp X,
    dC/dt = A(Lambda,g) C + Psi^T [B(X,a) + F(Y,X,a)],   dR/dt = A(lambda_perp,g) R + P_perp [B(X,a) + F(Y,X,a)],
    Z_hat = Z + P_out(Y_tau) - P_out(Y_0),   lambda_perp = mean(lambda_{K+1..N}).
它等于“K 取满、但把 lambda_{K+1..N} 全换成 lambda_perp”的模型（§1 的 K16 ⊂ B16 ⊂ K 取满）。
"""
import math

import pytest
import torch

from po_jepa import datasets, losses, theory_bounds
from po_jepa.dynamics import DynamicsConfig, IntegratorConfig, ModalDynamics
from po_jepa.geometry import regular_grid
from po_jepa.model import POJEPA, ProjectionConfig
from po_jepa.runtime import checks, training
from po_jepa.spectral import GraphConfig, GraphSpectrum, SpectralResult

from test_s28b_dyadic_zero_init import E, config, make_batch, make_controls, make_z

GRID = regular_grid(8, 8)
N = 64


def make_model(*, modes=8, fixed=True, complement="none", readout_init="xavier", head_std=.05, seed=1703):
    """S28b 的小模型（encoder 建图、残差读出、token 驱动，E=32），头部放大到 0.05 让 B/F 有明显贡献。"""
    torch.manual_seed(seed)
    dynamics = {**checks.load_protocol()["model"]["dynamics"], "forcing_space": "token", "head_weight_std": head_std}
    if complement != "none":
        dynamics["complement"] = complement
    graph = GraphConfig(16, 1703, .125, 4., modes, 1e-10, 1e-8, affinity_source="encoder", feature_dim=E, fixed_modes=fixed)
    return POJEPA(ProjectionConfig(E, 64, readout="residual", readout_init=readout_init), GraphSpectrum(graph),
                  ModalDynamics(DynamicsConfig(**dynamics)), IntegratorConfig(**checks.load_protocol()["model"]["integrator"]))


def durations(*values):
    return torch.tensor(values, dtype=torch.float64)


def patch_spectrum(monkeypatch, model, change):
    """把 spectrum_for 的结果换成 change(result)（同一类型、同一设备精度），用来改特征值或换基底的取法。"""
    original = model.spectrum.spectrum_for
    def wrapped(*args, **kwargs):
        return change(original(*args, **kwargs))
    monkeypatch.setattr(model.spectrum, "spectrum_for", wrapped)


def tail_replaced(result, k):
    """K 取满的谱，把 lambda_{k+1..N} 全换成它们的均值 lambda_perp（float64 先换，再转模型副本）。"""
    values = result.eigenvalues_cpu.clone()
    values[:, k:] = values[:, k:].mean(-1, keepdim=True)
    return SpectralResult(values, result.basis_cpu, values.to(dtype=torch.float32), result.basis, result.diagnostics,
                          result.complement_eigenvalue_cpu, result.complement_eigenvalue)


# ---------- 1. 默认开关逐位不变 ----------

def test_default_complement_is_bitwise_the_existing_model():
    """complement 不写和写 "none"：参数、随机数消耗、预测逐位相同；model_config 快照里没有 complement 键（旧检查点照样能读）。"""
    torch.manual_seed(3)
    a = make_model()
    state_a = torch.get_rng_state().clone()
    b = make_model(complement="none")
    assert torch.equal(state_a, torch.get_rng_state())
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.equal(pa, pb)
    z, ctl = make_z(2), make_controls([2., 3.5])
    assert torch.equal(a(z, ctl, durations(2., 3.5), GRID), b(z, ctl, durations(2., 3.5), GRID))
    assert "complement" not in training._model_config(a)["dynamics"]
    assert training._model_config(make_model(complement="shared"))["dynamics"]["complement"] == "shared"


def test_complement_adds_no_parameters_and_same_rng():
    """A_perp 用同一个条件网络：方案 B 不加参数，构造消耗的随机数相同，初始参数逐位相同。"""
    a, b = make_model(), make_model(complement="shared")
    assert [(n, p.shape) for n, p in a.named_parameters()] == [(n, p.shape) for n, p in b.named_parameters()]
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


# ---------- 2. 方案 B = K 取满、尾部特征值换成 lambda_perp ----------

@pytest.mark.parametrize("k", [8, 16])
def test_complement_equals_full_basis_with_tail_eigenvalues_replaced(monkeypatch, k):
    """§1：B16 与“K 取满、lambda_{K+1..N} 全换成 lambda_perp”的预测相同（核 R 的初值、强迫项分配 P_perp、A_perp 和读出）。"""
    b = make_model(modes=k, complement="shared")
    full = make_model(modes=1024, fixed=False)
    full.load_state_dict(b.state_dict())
    patch_spectrum(monkeypatch, full, lambda r: tail_replaced(r, k))
    z, ctl = make_z(3), make_controls([1., 3.5, 6.])
    tau = durations(1., 3.5, 6.)
    out_b, out_full = b(z, ctl, tau, GRID), full(z, ctl, tau, GRID)
    scale = (out_full-z).abs().max()
    assert scale > 1e-3                                      # 确实动了
    gap = (out_b-out_full).abs().max()
    assert gap <= 1e-5*float(out_full.abs().max())          # float32 累积误差量级（两边求和顺序不同）
    # 截断模型（同一组参数）明显不同：补空间确实进了预测
    trunc = make_model(modes=k)
    trunc.load_state_dict(b.state_dict())
    assert (trunc(z, ctl, tau, GRID)-out_b).abs().max() > 1000*gap


def test_complement_initial_state_and_lambda_perp():
    """C0 = Psi^T X、R0 = X - Psi Psi^T X（Y0 = X），R0 与 Psi 正交；lambda_perp = (tr L - Σ_{k≤K} λ_k)/(N - K)。"""
    model = make_model(complement="shared")
    z = make_z(2)
    q = model.prepare_query(z, GRID)
    basis, x = q.spectrum.basis, q.projected
    assert q.initial_coefficients.shape == (2, 8+N, 64)
    c0, r0 = q.initial_coefficients[:, :8], q.initial_coefficients[:, 8:]
    assert torch.allclose(c0, basis.transpose(1, 2)@x, atol=0, rtol=0)
    assert torch.allclose(basis@c0+r0, x, atol=1e-5)
    assert (basis.transpose(1, 2)@r0).abs().max() < 1e-4
    lap = model.spectrum.normalized_laplacian(model.spectrum.affinity(z, GRID)).laplacian
    trace = lap.diagonal(dim1=-2, dim2=-1).sum(-1)
    expected = (trace-q.spectrum.eigenvalues_cpu.sum(-1))/(N-8)
    assert torch.allclose(q.spectrum.complement_eigenvalue_cpu, expected, atol=1e-12, rtol=1e-10)
    assert q.spectrum.complement_eigenvalue_cpu.dtype == torch.float64 and q.spectrum.complement_eigenvalue.dtype == torch.float32
    assert torch.equal(q.dynamics.complement_eigenvalue, q.spectrum.complement_eigenvalue)
    # A 系数：前 K 行是模态的，后 N 行是同一组 A(lambda_perp, g)
    gamma = q.dynamics.coefficients.gamma
    assert gamma.shape == (2, 8+N, 32) and torch.equal(gamma[:, 8:], gamma[:, 8:9].expand(-1, N, -1))
    perp = model.dynamics._condition_free(q.spectrum.complement_eigenvalue[:, None], q.dynamics.pooled)
    assert torch.equal(gamma[:, 8], perp.gamma[:, 0]) and torch.equal(q.dynamics.coefficients.omega[:, 8], perp.omega[:, 0])
    # 截断模型：没有 R，也没有 lambda_perp
    plain = make_model().prepare_query(z, GRID)
    assert plain.initial_coefficients.shape == (2, 8, 64) and plain.dynamics.complement_eigenvalue is None


# ---------- 3. 基底不变性 ----------

def _signs(result, generator):
    signs = torch.where(torch.rand(result.basis_cpu.shape[-1], generator=generator) < .5, -1., 1.).double()
    basis = result.basis_cpu*signs
    return SpectralResult(result.eigenvalues_cpu, basis, result.eigenvalues, basis.float(), result.diagnostics,
                          result.complement_eigenvalue_cpu, result.complement_eigenvalue)


def _rotate_tail(result, k, generator):
    """尾部（特征值都换成 lambda_perp 的那 N-k 列）是一个重根子空间：在里面任意正交旋转。"""
    result = tail_replaced(result, k)
    q = torch.linalg.qr(torch.randn(N-k, N-k, generator=generator, dtype=torch.float64))[0]
    basis = result.basis_cpu.clone()
    basis[:, :, k:] = basis[:, :, k:]@q
    return SpectralResult(result.eigenvalues_cpu, basis, result.eigenvalues, basis.float(), result.diagnostics,
                          result.complement_eigenvalue_cpu, result.complement_eigenvalue)


@pytest.mark.parametrize("variant", ["complement", "full"])
def test_prediction_invariant_to_eigenvector_signs(monkeypatch, variant):
    """翻特征向量符号，B16 和 K 取满的预测不变（A 只看 lambda 和 g，token 式强迫与 Psi -> Psi Q 协变）。"""
    model = make_model(complement="shared") if variant == "complement" else make_model(modes=1024, fixed=False)
    z, ctl, tau = make_z(2), make_controls([2., 5.]), durations(2., 5.)
    before = model(z, ctl, tau, GRID)
    g = torch.Generator().manual_seed(7)
    patch_spectrum(monkeypatch, model, lambda r: _signs(r, g))
    after = model(z, ctl, tau, GRID)
    assert (after-before).abs().max() <= 1e-5*float(before.abs().max())


def test_full_basis_invariant_to_rotation_inside_a_repeated_eigenspace(monkeypatch):
    """K 取满时 Psi A(Lambda) Psi^T 只取决于 L：重根子空间内任意旋转，预测不变（方案 §2.2 的论证）。"""
    model = make_model(modes=1024, fixed=False)
    z, ctl, tau = make_z(2), make_controls([2., 5.]), durations(2., 5.)
    patch_spectrum(monkeypatch, model, lambda r: tail_replaced(r, 8))
    before = model(z, ctl, tau, GRID)
    g = torch.Generator().manual_seed(9)
    monkeypatch.undo()
    patch_spectrum(monkeypatch, model, lambda r: _rotate_tail(r, 8, g))
    after = model(z, ctl, tau, GRID)
    assert (after-before).abs().max() <= 1e-5*float(before.abs().max())


# ---------- 4. 闭式和零时长 ----------

def _silence_forcing(model):
    """B = 0、F = 0：动作场和残差的输出层、注意力 value 清零。"""
    with torch.no_grad():
        for layer in (model.dynamics.token_b_net[2], model.dynamics.token_residual_net[2]):
            layer.weight.zero_(); layer.bias.zero_()
        model.dynamics.token_v.weight.zero_()


def test_free_complement_follows_closed_form():
    """B = F = 0 时 R(tau) = exp(A_perp tau) R(0)、C(tau) = exp(A tau) C(0)（式 11 的闭式）。"""
    model = make_model(complement="shared")
    _silence_forcing(model)
    z, ctl, tau = make_z(2), make_controls([2.5, 6.]), durations(2.5, 6.)
    q = model.prepare_query(z, GRID)
    final = model.dynamics.integrate_segments(q.initial_coefficients, ctl, q.dynamics, duration=tau, config=model.integrator).final
    co = q.dynamics.coefficients
    exact = model.dynamics.free_propagate(q.initial_coefficients, gamma=co.gamma, omega=co.omega, duration=tau)
    assert torch.allclose(final, exact, atol=1e-5, rtol=1e-4)
    assert (final[:, 8:]-exact[:, 8:]).abs().max() <= 1e-5
    assert (q.spectrum.basis.transpose(1, 2)@final[:, 8:]).abs().max() < 1e-4   # R 留在正交补里


@pytest.mark.parametrize("variant", ["complement", "full", "k32"])
def test_zero_duration_is_identity_and_zero_init_starts_at_identity(variant):
    """tau = 0 时 Z_hat 逐位等于 Z；P_out 零初始化下任意 tau 第 0 步 Z_hat = Z。"""
    kwargs = dict(complement="shared") if variant == "complement" else (dict(modes=1024, fixed=False) if variant == "full"
                                                                        else dict(modes=32))
    z, ctl = make_z(2), make_controls([3., 3.])
    assert torch.equal(make_model(**kwargs)(z, ctl, durations(0., 0.), GRID), z)
    assert torch.equal(make_model(readout_init="zero", **kwargs)(z, ctl, durations(3., 3.), GRID), z)


@pytest.mark.parametrize("modes,fixed", [(64, True), (1024, False)])
def test_complement_is_empty_when_the_basis_is_full(modes, fixed):
    """K = N（K64c 在 8×8 上、或 K 取满）时补空间为空：不建 R，状态 K 行，预测和不加补空间的同 K 模型逐位相同。"""
    z, ctl = make_z(2), make_controls([3., 3.])
    plain, kept = make_model(modes=modes, fixed=fixed), make_model(modes=modes, fixed=fixed, complement="shared")
    kept.load_state_dict(plain.state_dict())
    q = kept.prepare_query(z, GRID)
    assert q.initial_coefficients.shape == (2, N, 64) and q.dynamics.complement_eigenvalue is None
    assert kept.dynamics.state_rows(q.dynamics) == N and not kept.dynamics.has_complement(q.dynamics)
    with torch.no_grad():
        assert torch.equal(kept(z, ctl, durations(3., 3.), GRID), plain(z, ctl, durations(3., 3.), GRID))
    with pytest.raises(ValueError, match="lambda_perp"):
        kept.dynamics.prepare_context(q.projected, q.spectrum.eigenvalues, q.spectrum.basis, GRID.coords,
                                      torch.zeros(2, device=q.projected.device))


def test_complement_on_a_mixed_pair_of_grids():
    """K64c 的 16×16 → 8×8 那种情形（这里 K=16：8×8 有补空间，4×4 的 K = N 没有）：两个网格各自走对的分支。"""
    model = make_model(modes=16, complement="shared")
    fine = model.prepare_query(make_z(1), GRID)
    coarse = model.prepare_query(torch.randn(1, 16, E), regular_grid(4, 4))
    assert fine.initial_coefficients.shape == (1, 16+N, 64) and fine.dynamics.complement_eigenvalue is not None
    assert coarse.initial_coefficients.shape == (1, 16, 64) and coarse.dynamics.complement_eigenvalue is None


def test_complement_with_fourier_or_modal_forcing_is_refused():
    """fixed_fourier 没有 lambda_perp，报错；modal 模式不接补空间。"""
    z = make_z(1)
    with pytest.raises(ValueError, match="forcing_space"):
        DynamicsConfig(**{**checks.load_protocol()["model"]["dynamics"], "forcing_space": "modal", "complement": "shared"})
    with pytest.raises(ValueError, match="complement"):
        DynamicsConfig(**{**checks.load_protocol()["model"]["dynamics"], "forcing_space": "token", "complement": "mean"})
    from po_jepa.fourier import FourierConfig, FourierSpectrum
    model = make_model(complement="shared")
    graph = model.spectrum.config
    model.spectrum = FourierSpectrum(FourierConfig(graph.q_dim, graph.q_seed, graph.sigma_x, graph.sigma_z, graph.max_modes,
        graph.diagnostic_atol, graph.diagnostic_rtol, affinity_source=graph.affinity_source, feature_dim=graph.feature_dim,
        fixed_modes=True))
    with pytest.raises(ValueError, match="lambda_perp"):
        model.prepare_query(z, GRID)


# ---------- 6. 梯度 ----------

def test_spectrum_stays_stopped_and_gradient_reaches_a_perp():
    """谱分支仍停梯度；A_perp 的梯度回到条件网络（只对补空间 R 的读出求导也有梯度）。"""
    model = make_model(complement="shared")
    z = make_z(2).requires_grad_()
    q = model.prepare_query(z, GRID)
    assert not q.spectrum.basis.requires_grad and not q.spectrum.eigenvalues.requires_grad
    assert not q.dynamics.complement_eigenvalue.requires_grad
    final = model.dynamics.integrate_segments(q.initial_coefficients, make_controls([3., 3.]), q.dynamics,
                                              duration=durations(3., 3.), config=model.integrator).final
    final[:, 8:].square().sum().backward()
    trunk = model.dynamics.condition_trunk[0].weight.grad
    assert trunk is not None and trunk.abs().sum() > 0
    assert model.dynamics.gamma_head.weight.grad.abs().sum() > 0 and z.grad.abs().sum() > 0


# ---------- 7. 训练损失、理论界在新变体下能跑通 ----------

# K16c_mixed：K=16 + 补空间，细网格 8×8 有补空间、粗网格 4×4 的 K = N 没有（对应 K64c 的 16×16 → 8×8）
VARIANTS = dict(K16c=dict(complement="shared"), K16c_mixed=dict(modes=16, complement="shared"), K32=dict(modes=32),
                Kfull=dict(modes=1024, fixed=False))


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("partition", ["bisection", "dyadic"])
def test_training_step_and_bounds_are_finite(name, partition):
    """时间一致性（二等分、多段切分）、空间一致性（8×8 → 4×4）、JEPA 一次更新，和引理 5.3/5.4 的界都有限。"""
    kwargs = dict(VARIANTS[name])
    if name == "K32":
        kwargs["modes"] = 8   # 粗网格 4×4 只有 16 个 token；固定 K 族在这个小网格上用 K=8 代表
    model = make_model(**kwargs)
    batch = make_batch([1., 2., 4., 8.])
    model.train()
    optimizer = training.make_optimizer(model)
    result = training.train_step(model, optimizer, batch, loss_config=config(partition), completed_steps=3)
    assert all(math.isfinite(v) for v in result.loss_values.values()) and result.loss_values["space"] > 0
    model.eval()
    with torch.no_grad():
        z, ctl = batch.raw_z[:1], make_controls([4.])
        bound = theory_bounds.temporal_partition_bound(model, z, ctl, (1., 1., 2.), GRID)
        assert all(math.isfinite(v) for v in bound["lhs"]) and math.isfinite(bound["sharp"])
        space = theory_bounds.spatial_partition_bound(model, z, ctl, (2., 2.), batch.transfer)
        assert all(math.isfinite(v) for v in space["lhs"])


def test_rollout_supervision_runs_with_complement():
    from po_jepa import rollout_supervision
    model = make_model(complement="shared")
    batch = make_batch([2., 4.])
    cfg = config("dyadic")
    predictions = rollout_supervision.rollout_predictions(model, batch.raw_z, batch.controls, batch.tau, batch.sigma, GRID,
                                                          config=cfg, step=0, sample_ids=batch.sample_ids)
    assert torch.isfinite(predictions.direct).all()


# ---------- 协议、标定文件名、变体记录 ----------

@pytest.fixture
def v7(monkeypatch):
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    monkeypatch.setenv(checks.UPDATES_ENV, "10000")
    for name in (checks.PARTITION_ENV, checks.READOUT_INIT_ENV, checks.MODES_ENV, checks.COMPLEMENT_ENV):
        monkeypatch.delenv(name, raising=False)
    old = datasets._set_for_tests(datasets.DatasetSpec("double_well", 2000))
    yield
    datasets._set_for_tests(old)


def test_default_protocol_is_k16_fixed_and_unchanged(v7):
    config = checks.load_protocol()
    assert config["model"]["graph"]["max_modes"] == 16 and config["model"]["graph"]["fixed_modes"] is True
    assert "complement" not in config["model"]["dynamics"]
    assert checks.v7_variant() == checks.V7_VARIANT_DEFAULT
    assert checks.v7_calibration_path().name == "double_well-2000-u10000.yaml"


@pytest.mark.parametrize("modes,complement,suffix,max_modes,fixed", [
    ("32", "none", "~K32", 32, True), ("64", "none", "~K64", 64, True), ("full", "none", "~Kfull", checks.FULL_MAX_MODES, False),
    ("16", "shared", "~K16c", 16, True), ("32", "shared", "~K32c", 32, True), ("64", "shared", "~K64c", 64, True)])
def test_mode_variants_set_protocol_calibration_and_job_record(v7, monkeypatch, modes, complement, suffix, max_modes, fixed):
    """变体写进协议（graph.max_modes/fixed_modes、dynamics.complement），标定文件、job.json、预登记按变体分开；参数量不变。"""
    from po_jepa.runtime import comparison, v7jobs, workflow
    default_model, _ = comparison.build_condition("po_jepa_full", torch.device("cpu"))
    monkeypatch.setenv(checks.PARTITION_ENV, "dyadic")
    monkeypatch.setenv(checks.READOUT_INIT_ENV, "zero")
    monkeypatch.setenv(checks.MODES_ENV, modes)
    monkeypatch.setenv(checks.COMPLEMENT_ENV, complement)
    config = checks.load_protocol()
    assert config["model"]["graph"]["max_modes"] == max_modes and config["model"]["graph"]["fixed_modes"] is fixed
    assert config["model"]["dynamics"].get("complement", "none") == complement
    assert checks.v7_calibration_path().name == f"double_well-2000-u10000~dyadic~zeroinit{suffix}.yaml"
    assert config["v7_uncalibrated"]
    with pytest.raises(ValueError, match="还没标定"):
        workflow.require_calibrated_weights()
    model, _ = comparison.build_condition("po_jepa_full", torch.device("cpu"))
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in default_model.parameters()) == 227_617
    assert model.spectrum.config.max_modes == max_modes and model.dynamics.config.complement == complement
    variant = dict(temporal_partition="dyadic", readout_init="zero", **({"modes": modes} if modes != "16" else {}),
                   **({"complement": complement} if complement != "none" else {}))
    assert v7jobs.with_variant(dict(condition="po_jepa_full")) == dict(condition="po_jepa_full", variant=variant)


def test_mode_variant_env_contract(v7, monkeypatch):
    monkeypatch.setenv(checks.MODES_ENV, "48")
    with pytest.raises(ValueError, match=checks.MODES_ENV):
        checks.v7_variant()
    monkeypatch.setenv(checks.MODES_ENV, "full")
    monkeypatch.setenv(checks.COMPLEMENT_ENV, "shared")
    with pytest.raises(ValueError, match="truncated basis"):
        checks.v7_variant()
    monkeypatch.setenv(checks.MODES_ENV, "16")
    monkeypatch.setenv(checks.COMPLEMENT_ENV, "mean")
    with pytest.raises(ValueError, match=checks.COMPLEMENT_ENV):
        checks.v7_variant()
    monkeypatch.delenv(checks.COMPLEMENT_ENV)
    monkeypatch.setenv(checks.MODES_ENV, "32")
    monkeypatch.delenv(checks.PROTOCOL_ENV)
    with pytest.raises(ValueError, match="v7"):
        checks.load_protocol()


def test_variant_suffix_agrees_between_runtime_and_summary():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("summarize_v7_c3", Path(__file__).resolve().parents[1]/"tools/summarize_v7.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for variant in (dict(temporal_partition="dyadic", readout_init="zero"),
                    dict(temporal_partition="dyadic", readout_init="zero", modes="32"),
                    dict(temporal_partition="dyadic", readout_init="zero", modes="full"),
                    dict(temporal_partition="dyadic", readout_init="zero", complement="shared")):
        assert module._variant_suffix(dict(variant=variant)) == checks.variant_suffix(variant)
    assert checks.variant_suffix(dict(temporal_partition="dyadic", readout_init="zero", complement="shared")) == "~dyadic~zeroinit~K16c"


def test_job_env_restores_mode_variant(tmp_path, monkeypatch):
    """评测工具（plan_v7._job_env）按 job.json 的 variant 设回 K 和补空间；默认变体显式设回默认，不受残留环境变量影响。"""
    import importlib.util, json, os
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("plan_v7_c3", Path(__file__).resolve().parents[1]/"tools/plan_v7.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(os, "environ", dict(os.environ, PO_JEPA_MODES="full", PO_JEPA_COMPLEMENT="shared"))
    job = dict(seed=1703, updates=10000, dataset="double_well-2000", condition="po_jepa_full")
    (tmp_path/"a").mkdir()
    (tmp_path/"a/job.json").write_text(json.dumps(job))
    module._job_env(tmp_path/"a")
    assert (os.environ["PO_JEPA_MODES"], os.environ["PO_JEPA_COMPLEMENT"]) == ("16", "none")
    (tmp_path/"b").mkdir()
    (tmp_path/"b/job.json").write_text(json.dumps(dict(job, variant=dict(temporal_partition="dyadic", readout_init="zero",
                                                                         complement="shared"))))
    module._job_env(tmp_path/"b")
    assert (os.environ["PO_JEPA_MODES"], os.environ["PO_JEPA_COMPLEMENT"]) == ("16", "shared")


def test_run_py_passes_mode_flags(monkeypatch, tmp_path):
    """run.py --modes/--complement 转成环境变量；和已设的环境变量不同时报错。"""
    import importlib.util, subprocess
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("run_c3", Path(__file__).resolve().parents[1]/"run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda argv, check, env: seen.append(env) or subprocess.CompletedProcess(argv, 0))
    common = ["calibrate", "--features-root", str(tmp_path), "--system", "double_well", "--trajectories", "2000",
              "--updates", "10000", "--device", "cpu"]
    assert run.main([*common, "--output", str(tmp_path/"o1"), "--modes", "full"]) == 0
    assert seen[-1]["PO_JEPA_MODES"] == "full" and "PO_JEPA_COMPLEMENT" not in seen[-1]
    assert run.main([*common, "--output", str(tmp_path/"o2"), "--complement", "shared"]) == 0
    assert seen[-1]["PO_JEPA_COMPLEMENT"] == "shared"
    monkeypatch.setenv("PO_JEPA_MODES", "32")
    with pytest.raises(SystemExit):
        run.main([*common, "--output", str(tmp_path/"o3"), "--modes", "64"])


# ---------- 任务表（make_small_validation_jobs.py --stage c3）----------

import json
import shlex
import sys
from argparse import Namespace
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]/"tools"
sys.path.insert(0, str(TOOLS))
import make_small_validation_jobs as jobs  # noqa: E402
import run_jobs  # noqa: E402
from test_s28c_small_validation import P, load_module, make_job, readout_file, summary_tool  # noqa: E402

SCENES = ("advection_diffusion", "double_well", "van_der_pol")
C3_COUNTS = {1: dict(C=6, C2=2, D=15, D2=6, R3=24, R4=4, G9=12, E=3), 2: dict(C=6, C2=2, D=6, R3=12, R4=2, E=3)}


def c3_sections(batch, tmp_path):
    path = tmp_path/f"jobs-c3-b{batch}.txt"
    path.write_text("\n".join(jobs.job_lines("c3", batch=batch))+"\n", encoding="utf-8")
    return run_jobs.parse_sections(path)


@pytest.mark.parametrize("batch", [1, 2])
def test_c3_sections_and_counts(batch, tmp_path):
    """方案 §4、§6：第一批标定 6 + 检查 2 + 训练 12 + identity 3 + 同步数 6 + 读出 24（+ 模态 4）+ G9 12；第二批训练 6。"""
    assert {s["name"]: len(s["commands"]) for s in c3_sections(batch, tmp_path)} == C3_COUNTS[batch]


def test_c3_training_lines_follow_the_plan():
    seen = {}
    for batch in (1, 2):
        for line in jobs.job_lines("c3", batch=batch):
            if not line.startswith("python run.py run "):
                continue
            a = shlex.split(line)
            flags = tuple(a[a.index(k)+1] for k in ("--temporal-partition", "--readout-init", "--modes", "--complement"))
            out = a[a.index("--output")+1]
            assert out.startswith("../../results-small-validation/c3/dyadic-zero-")
            name = out.split("/")[4].removeprefix("dyadic-zero-")
            assert flags == ("dyadic", "zero", *jobs.C3_VARIANTS[name]) and a[a.index("--seed")+1] == "1703"
            assert a[a.index("--trajectories")+1] == "2000" and a[a.index("--updates")+1] == "10000"
            seen.setdefault((batch, name), set()).add(a[a.index("--condition")+1])
    assert seen == {(1, "K16c"): {"po_jepa_full", "po_jepa_no_consistency", "identity"},
                    (1, "Kfull"): {"po_jepa_full", "po_jepa_no_consistency"},
                    (2, "K32c"): {"po_jepa_full"}, (2, "K64c"): {"po_jepa_full"}}
    assert all(c == "shared" for name, (_, c) in jobs.C3_VARIANTS.items() if name != "Kfull")   # 四个变体都不丢信息


def test_c3_every_run_py_line_is_accepted_and_sets_the_variant(tmp_path, monkeypatch):
    entry = load_module(Path(__file__).resolve().parents[1]/"run.py", "c3_run_py")
    seen = []
    monkeypatch.setattr(entry.subprocess, "run", lambda command, **kw: seen.append(kw["env"]) or Namespace(returncode=0))
    for name in ("PO_JEPA_TEMPORAL_PARTITION", "PO_JEPA_READOUT_INIT", "PO_JEPA_MODES", "PO_JEPA_COMPLEMENT"):
        monkeypatch.delenv(name, raising=False)
    work = tmp_path/"work"/"full-model"/"po-jepa-full"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    counted = 0
    for batch in (1, 2):
        for line in jobs.job_lines("c3", batch=batch):
            if not line.startswith("python run.py"):
                continue
            argv = shlex.split(line)
            for key in ("--features-root", "--training-run", "--limit-run"):
                if key in argv:
                    Path(argv[argv.index(key)+1]).mkdir(parents=True, exist_ok=True)
            assert entry.main(argv[2:]) == 0, line
            env = seen[-1]
            assert (env["PO_JEPA_MODES"], env["PO_JEPA_COMPLEMENT"]) == (argv[argv.index("--modes")+1], argv[argv.index("--complement")+1])
            assert (env["PO_JEPA_TEMPORAL_PARTITION"], env["PO_JEPA_READOUT_INIT"]) == ("dyadic", "zero")
            counted += 1
    assert counted == (6+2+15+6)+(6+2+6)


def test_c3_tool_lines_parse():
    """shrinkage_check、state_readout、modal_spectrum、summarize 的任务行都被各自的参数解析接受（不跑）。"""
    shrink = load_module(TOOLS/"shrinkage_check.py", "c3_shrinkage_check")
    lines = jobs.job_lines("c3", batch=1)
    g9 = [shlex.split(l) for l in lines if l.startswith("python tools/shrinkage_check.py")]
    assert len(g9) == 12
    for a in g9:
        parsed = shrink._parse(a[2:]) if hasattr(shrink, "_parse") else None
        assert parsed is None or not parsed.cross_version
        assert a[a.index("--job")+1].rsplit("/", 1)[1] in ("po_jepa_full", "po_jepa_no_consistency")
    readout = [shlex.split(l) for l in lines if "state_readout.py" in l or "modal_spectrum.py" in l]
    assert readout and not any("--cross-version" in a for a in readout)
    summary = [shlex.split(l) for l in lines if "summarize_small_validation.py" in l]
    assert summary == [["python", "tools/summarize_small_validation.py", "c3", "--results", "../../results-small-validation/c3",
                        "--c2-results", "../../results-small-validation/c2/dyadic-zero", "--output",
                        "../../results-small-validation/c3/summary"]]


def test_c3_options_are_checked(capsys):
    for argv, message in ((["--stage", "c3"], "--batch"), (["--stage", "c2", "--temporal-partition", "dyadic", "--readout-init", "zero",
                                                            "--batch", "1"], "--batch"),
                          (["--stage", "c3", "--batch", "1", "--readout-init", "zero"], "只用于 --stage c2")):
        with pytest.raises(SystemExit) as error:
            jobs.main(argv)
        assert error.value.code == 2 and message in capsys.readouterr().err


def test_c3_lists_match_the_summary_and_runtime():
    assert set(jobs.C3_VARIANTS) == set(summary_tool.C3_SPACE)
    for name, (modes, complement) in jobs.C3_VARIANTS.items():
        expected = dict(**({"modes": modes} if modes != "16" else {}), **({"complement": complement} if complement != "none" else {}))
        assert summary_tool.C3_SPACE[name] == expected
        assert summary_tool.variant_label(dict(summary_tool.C3_BASE, **expected)) == f"dyadic-zero-{name}"
        assert checks.variant_suffix(dict(summary_tool.C3_BASE, **expected)) == f"~dyadic~zeroinit~{name}"
    assert {v: set(c) for v, c in summary_tool.C3_CONDITIONS.items()} == {
        v: set(jobs.C3_CONDITIONS[b]) for b, names in jobs.C3_BATCHES.items() for v in names}


# ---------- 汇总（summarize_small_validation.py c3）----------

DYZ = dict(temporal_partition="dyadic", readout_init="zero")
IDENTITY = dict(P, f16=2., f12=2., f8=2.)


def c3_tree(tmp_path, values=None, identity=None, batches=(1, 2)):
    """合成 C-3 和 C-2 的结果根。f16 是 16×16 直接预测的距离（identity 是 2.0，所以比 identity 好 = 1 − f16/2）。"""
    root, c2 = tmp_path/"c3", tmp_path/"c2"
    base = {"K16c": dict(P, f16=1.3, r12=2.0), "K16c/no_cons": dict(P, f16=1.31, r12=2.5),
            "Kfull": dict(P, f16=1.34, r12=2.05), "Kfull/no_cons": dict(P, f16=1.35, r12=2.6),
            "K32c": dict(P, f16=1.32), "K64c": dict(P, f16=1.33)}
    ref = dict(po_jepa_full=dict(P, f16=1.45, r12=2.2), po_jepa_no_consistency=dict(P, f16=1.46, r12=2.6),
               transformer_consistency=dict(P, f16=1.31, r12=2.1), identity=IDENTITY)
    values = values or {}
    for s in SCENES:
        for label, p in base.items():
            v, _, nc = label.partition("/")
            if (1 if v in ("K16c", "Kfull") else 2) not in batches:
                continue
            c = "po_jepa_no_consistency" if nc else "po_jepa_full"
            make_job(root/f"dyadic-zero-{v}"/s/"seed1703"/c, c, f"{s}-2000", dict(p, **values.get((s, label), {})), updates=10000,
                     variant=dict(DYZ, **summary_tool.C3_SPACE[v]))
        if 1 in batches:
            make_job(root/"dyadic-zero-K16c"/s/"seed1703"/"identity", "identity", f"{s}-2000", dict(IDENTITY, **(identity or {}).get(s, {})),
                     updates=10000, variant=dict(DYZ, complement="shared"))
        for c, p in ref.items():
            make_job(c2/s/"seed1703"/c, c, f"{s}-2000", dict(p, **values.get((s, f"C-2 {c}"), {})), updates=10000, variant=DYZ)
    return root, c2


def test_c3_summary_constructed_case(tmp_path):
    s = summary_tool.c3_summary(*c3_tree(tmp_path))
    assert not s["problems"] and not s["missing"] and not s["invalid"]
    assert all(x["identity_gate"]["identical"] for x in s["scenes"].values())
    rows = s["scenes"]["double_well"]["rows"]
    assert rows["K16"]["source"] == "C-2" and rows["K16c"]["source"] == "C-3"
    assert rows["K16c"]["direct"]["8"] == pytest.approx(1-1.3/2) and rows["K16"]["direct"]["8"] == pytest.approx(1-1.45/2)
    h = s["hypotheses"]
    assert h["H1"]["pass_"] is True                                   # 0.075 ≥ 0.03
    assert h["H2"]["verdict"] == "B16 更好" and h["H2"]["candidate"] == "K16c"   # 0.02 > 0.01
    assert h["H3"]["variant"] == "K16c" and h["H3"]["pass_"] is True and h["H3"]["inconclusive"]   # 1.30 vs TF 1.31
    assert h["H4"]["K16c"]["one"] is True and h["H4"]["K16c"]["two"] is True   # 2.0 ≤ 0.95·2.5、≤ 2.1
    assert h["H4"]["Kfull"]["one"] is True and h["H4"]["Kfull"]["two"] is True
    assert h["H5"]["pass_"] is True
    assert h["H6"]["pass_"] is False                                  # 1.30 → 1.32 → 1.33 → 1.34：基底越多越差，不单调
    assert h["H6"]["scenes"][0]["curve"] == pytest.approx(dict(K16c=1-1.3/2, K32c=1-1.32/2, K64c=1-1.33/2, Kfull=1-1.34/2))
    md = summary_tool.markdown(s)
    assert "## H1" in md and "dyadic-zero" in md


def test_c3_h6_curve_tolerance(tmp_path):
    """K16c → K32c → K64c 变好；K 取满比 K64c 差 0.5 pp 以内算单调（噪声），差 1 pp 不算。"""
    curve = {(sc, v): dict(f16=f) for sc in SCENES for v, f in (("K16c", 1.42), ("K32c", 1.40), ("K64c", 1.38))}
    s = summary_tool.c3_summary(*c3_tree(tmp_path, values={**curve, **{(sc, "Kfull"): dict(f16=1.388) for sc in SCENES}}))
    assert s["hypotheses"]["H6"]["pass_"] is True                     # gain 差 −0.4 pp
    s = summary_tool.c3_summary(*c3_tree(tmp_path/"b", values={**curve, **{(sc, "Kfull"): dict(f16=1.40) for sc in SCENES}}))
    assert s["hypotheses"]["H6"]["pass_"] is False                    # gain 差 −1 pp


def test_c3_failures(tmp_path):
    """H1 不过（放开截断没变好）、H4① 不过（一致性对滚动没用）、H5 不过（8×8 变差）。"""
    values = {}
    for sc in SCENES:
        values[(sc, "K16c")] = dict(f16=1.43, r12=2.5, f8=1.5)
        values[(sc, "Kfull")] = dict(f16=1.44, r12=2.6)
    h = summary_tool.c3_summary(*c3_tree(tmp_path, values=values))["hypotheses"]
    assert h["H1"]["pass_"] is False and h["H4"]["K16c"]["one"] is False and h["H5"]["pass_"] is False
    assert h["H2"]["verdict"] == "持平" and h["H3"]["pass_"] is False


def test_c3_identity_mismatch_blocks_the_c2_references(tmp_path):
    s = summary_tool.c3_summary(*c3_tree(tmp_path, identity={"double_well": dict(f16=2.0001)}))
    g = s["scenes"]["double_well"]["identity_gate"]
    assert not g["identical"] and "值不同" in g["reason"] and "K16" not in s["scenes"]["double_well"]["rows"]
    assert s["hypotheses"]["H1"]["pass_"] is None                     # 缺 K16，不下判定
    assert s["scenes"]["van_der_pol"]["identity_gate"]["identical"]


def test_c3_only_first_batch_and_wrong_variant(tmp_path):
    root, c2 = c3_tree(tmp_path, batches=(1,))
    s = summary_tool.c3_summary(root, c2)
    assert {(m["variant"], m["condition"]) for m in s["missing"]} == {("K32c", "po_jepa_full"), ("K64c", "po_jepa_full")}
    assert s["hypotheses"]["H6"]["pass_"] is None and s["hypotheses"]["H1"]["pass_"] is True
    # 结果根里放了别的变体的任务：无效，不进判定
    bad = root/"dyadic-zero-Kfull"/"double_well"/"seed1703"/"po_jepa_full"
    import shutil
    shutil.rmtree(bad)
    make_job(bad, "po_jepa_full", "double_well-2000", P, updates=10000, variant=dict(DYZ, modes="64"))
    s = summary_tool.c3_summary(root, c2)
    assert any("Kfull po_jepa_full" in x["condition"] for x in s["invalid"])
    assert "Kfull" not in s["scenes"]["double_well"]["rows"]


def test_c3_readout_report(tmp_path):
    root, c2 = c3_tree(tmp_path, batches=(1,))
    v = dict(DYZ, complement="shared")
    for sc in SCENES:
        readout_file(root/"dyadic-zero-K16c"/"readout", "rollout", sc, 1703, "po_jepa_full", v, nmse=.2, vpt=40)
        readout_file(root/"dyadic-zero-K16c"/"readout", "rollout", sc, 1703, "po_jepa_no_consistency", v, nmse=.3, vpt=30)
    s = summary_tool.c3_summary(root, c2)
    assert not s["problems"]
    items = [e for e in s["readout"] if e["variant"] == "K16c"]
    assert len(items) == 3 and items[0]["po_jepa_full"]["nmse_mean"] == pytest.approx(.2)
    assert "## 读出" in summary_tool.markdown(s)


def test_c3_summary_cli(tmp_path):
    root, c2 = c3_tree(tmp_path)
    assert summary_tool.main(["c3", "--results", str(root), "--c2-results", str(c2), "--output", str(tmp_path/"out")]) == 0
    data = json.loads((tmp_path/"out"/"summary.json").read_text(encoding="utf-8"))
    assert data["stage"] == "c3" and (tmp_path/"out"/"SUMMARY.md").exists()


def test_c3_h1_needs_only_one_lossless_variant(tmp_path):
    """H1 看“K 取满或 B16”：只有 B16 比 K16 高 ≥ 3 pp、K 取满和 K16 一样，也算通过；两个都不到才不过。"""
    values = {(sc, "Kfull"): dict(f16=1.45) for sc in SCENES}
    h = summary_tool.c3_summary(*c3_tree(tmp_path, values=values))["hypotheses"]["H1"]
    assert h["pass_"] is True and all(x["minus_k16"]["Kfull"] == pytest.approx(0.) for x in h["scenes"])
    values.update({(sc, "K16c"): dict(f16=1.40) for sc in SCENES})   # 只高 2.5 pp
    assert summary_tool.c3_summary(*c3_tree(tmp_path/"b", values=values))["hypotheses"]["H1"]["pass_"] is False
