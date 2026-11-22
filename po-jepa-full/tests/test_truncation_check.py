"""G10: tools/truncation_check.py (spatial-mode truncation check, evaluation only).

Fixtures come from the S25f state-readout tests (double_well:40, fake encoder; validation 32-35, test 36-39 with 5 starts).
The models are real: an untrained PO-JEPA (token forcing, v7 protocol) with a random P_out, so that its change is nonzero,
and a toy "Transformer" whose change is a token-wise nonlinear map with components outside span(Psi_16).
Projectors, LN distances and channel bases are recomputed here with numpy / float64, not with the tool's helpers.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from po_jepa import datasets
from po_jepa.runtime import comparison, evaluation, v7jobs
from test_s25c_long_horizon import active, real  # noqa: F401
from test_s25f_state_readout import BASE, _job, _tool, caches  # noqa: F401

ENV = dict(PO_JEPA_PROTOCOL="v7", PO_JEPA_SEED="1703", PO_JEPA_UPDATES="10000", PO_JEPA_TEMPORAL_PARTITION="bisection",
           PO_JEPA_READOUT_INIT="xavier", PO_JEPA_DATASET="double_well:40")
KEYS = [(tid, s, t) for tid in range(36, 40) for s in (0, 5, 10, 15, 20) for t in (1., 2., 4., 8.)]
EPS = 1e-5


@pytest.fixture
def env(monkeypatch, active):
    """_job_env writes these keys into os.environ; setting them first through monkeypatch restores them afterwards."""
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    active(BASE)


def _po():
    model, loss = comparison.build_condition("po_jepa_full", "cpu")
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        model.p_out.weight.copy_(torch.randn(model.p_out.weight.shape, generator=g)*.005)
    return model.eval(), loss


class Leak(torch.nn.Module):
    """Phi(Z) = Z + 0.05 T tanh(Z M): per-token change with a rank-8 channel map, not restricted to span(Psi_16)."""
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(5)
        self.register_buffer("m", torch.randn(1408, 8, generator=g)@torch.randn(8, 1408, generator=g)/1408)

    def forward(self, raw, controls, duration, grid):
        return raw+.05*float(duration[0])*torch.tanh(raw@self.m)


def _loader(models, loss, seen=None):
    def load(condition, training_run, device):
        key = None if training_run is None else str(Path(training_run).resolve())
        if seen is not None:
            seen.append((condition, key))
        return models[key], loss, dict(step=7, checkpoint_sha256="ab"*32), dict(fake=True)
    return load


def _ln(x):
    x = np.asarray(x, dtype=np.float64)
    return (x-x.mean(-1, keepdims=True))/np.sqrt(x.var(-1, keepdims=True)+EPS)


def _d(p, y):
    return float(np.abs(_ln(p)-_ln(y)).mean())


def _view(base, key, g):
    w = evaluation.load_test_window(base, evaluation.EvaluationQuery(key[0], key[1], key[2], 16, "point"), "cpu")
    return w, *evaluation.spatial_view(w, g)


def _projector(model, z, grid, k):
    """Pi_k from numpy eigh of the model's normalised Laplacian (sign- and rotation-free)."""
    lap = model.spectrum.normalized_laplacian(model.spectrum.affinity(z, grid)).laplacian[0].numpy()
    values, vectors = np.linalg.eigh(lap)
    assert k == lap.shape[0] or values[k]-values[k-1] > 1e-6          # the cut is not inside a repeated eigenvalue
    v = vectors[:, :k]
    return v@v.T


def _direct(model, w, z, grid):
    c, d = w.inputs.controls, w.inputs.duration
    with torch.no_grad():
        return model(z, c, d, grid)          # the evaluation's split/concatenate leaves the control path unchanged


def test_window_keys_are_registered_points(env):
    tool = _tool("truncation_check")
    keys = tool.window_keys()
    assert keys == KEYS
    index = {(q.trajectory_id, q.start_tick, float(q.duration_cycles), q.grid_size) for q in evaluation.build_index()
             if q.kind == "point"}
    assert all((*k, g) in index for k in keys for g in (16, 12, 8))


def test_optimise_reaches_a_reachable_target():
    tool = _tool("truncation_check")
    from po_jepa.losses import LossConfig
    loss = comparison.build_condition("identity", "cpu")[1]
    assert isinstance(loss, LossConfig)
    g = torch.Generator().manual_seed(2)
    z, target = torch.randn(3, 4, 8, generator=g), torch.randn(3, 4, 8, generator=g)
    start, best = tool.optimise(z, target, torch.zeros(3, 4, 8), None, None, loss, 400, .05)
    assert torch.all(best <= start) and torch.all(best < .1*start)
    exact_start, exact_best = tool.optimise(z, target, target-z, None, None, loss, 5, .05)
    assert float(exact_start.max()) < 1e-5 and float(exact_best.max()) < 1e-5
    # restricted on both sides: the start point is the given coefficient
    left = torch.linalg.qr(torch.randn(3, 4, 2, generator=g))[0]
    right = torch.linalg.qr(torch.randn(8, 3, generator=g))[0]
    a0 = torch.randn(3, 2, 3, generator=g)
    s, _ = tool.optimise(z, target, a0, left, right, loss, 1, .05)
    p = (z+torch.bmm(left, a0)@right.T).numpy()
    assert [float(x) for x in s] == pytest.approx([_d(p[i], target[i].numpy()) for i in range(3)], rel=1e-5)


def test_optimise_reports_the_smallest_iterate_not_the_last(monkeypatch):
    """The ceiling is the best point met along the optimiser path, not the last one: a scripted optimiser visits the
    exact optimum at the first update and then leaves it."""
    tool = _tool("truncation_check")
    loss = comparison.build_condition("identity", "cpu")[1]
    g = torch.Generator().manual_seed(4)
    z, target = torch.randn(2, 4, 8, generator=g), torch.randn(2, 4, 8, generator=g)
    points = [target-z, torch.full_like(z, 50.)]

    class Scripted(torch.optim.SGD):
        def step(self, closure=None):
            with torch.no_grad():
                self.param_groups[0]["params"][0].copy_(points.pop(0))
    monkeypatch.setattr(torch.optim, "Adam", lambda params, lr: Scripted(params, lr=lr))
    start, best = tool.optimise(z, target, torch.zeros(2, 4, 8), None, None, loss, 2, .1)
    assert float(start.min()) > .1 and float(best.max()) < 1e-6 and not points


def test_project(caches, real, env, tmp_path, monkeypatch):
    base, _ = caches
    tool = _tool("truncation_check")
    po, loss = _po()
    leak = Leak()
    tf_job, po_job = _job(tmp_path/"tf", "transformer_consistency"), _job(tmp_path/"po", "po_jepa_full")
    po_dir = (po_job/"run"/"job").resolve()
    stored = [evaluation.EvaluationQuery(36, 0, 1., 16, "point"), evaluation.EvaluationQuery(37, 5, 2., 8, "point")]
    evaluation.run_prepared_evaluation(po, "po_jepa_full", base, po_dir/"evaluation"/"po_jepa_full", loss, queries=stored,
                                       checkpoint_identity=dict(fake=1), device="cpu")
    models = {str((tf_job/"run"/"job").resolve()): leak, str(po_dir): po}
    monkeypatch.setattr(v7jobs, "load_for_evaluation", _loader(models, loss))
    out = tmp_path/"project.json"
    args = ["project", "--job", str(tf_job), "--spectrum-job", str(po_job), "--features", str(base), "--device", "cpu",
            "--grids", "16,8"]
    tool.main([*args, "--output", str(out)])
    r = json.loads(out.read_text())
    assert (r["kind"], r["dataset"], r["condition"], r["grids"]) == ("truncation_project", BASE.tag, "transformer_consistency", [16, 8])
    assert r["spectrum_job"]["condition"] == "po_jepa_full" and r["scientific_conclusions_verified"] is False
    assert r["spectrum_model_offspan_16_max"] < 1e-4
    assert r["evaluation_check"]["model"] is None
    assert r["evaluation_check"]["spectrum_model"]["compared"] == 2
    assert r["evaluation_check"]["spectrum_model"]["max_abs_difference"] < 1e-6
    for g in ("16", "8"):
        for t in ("1", "2", "4", "8"):
            s = r["results"][g][t]
            assert s["windows"] == 20
            assert s["energy_in_span"]["spectrum_model_K16"] == pytest.approx(1., abs=1e-6)
            assert s["distance"]["spectrum_model_proj_K16"] == pytest.approx(s["distance"]["spectrum_model"], rel=1e-5)
            assert 0 < s["energy_in_span"]["model_K16"] < s["energy_in_span"]["model_K64"] <= 1+1e-6
    assert r["results"]["8"]["1"]["distance"]["model_proj_K64"] == pytest.approx(r["results"]["8"]["1"]["distance"]["model"], rel=1e-5)
    # independent recomputation of two cells
    for g, t in ((16, 2.), (8, 4.)):
        acc = dict(identity=[], model=[], proj=[], perp=[], po=[], e16=0., e=0.)
        for key in [k for k in KEYS if k[2] == t]:
            w, z, y, grid = _view(base, key, g)
            pi = _projector(po, z, grid, 16)
            p = _direct(leak, w, z, grid)[0].double().numpy()
            zz, yy = z[0].double().numpy(), y[0].double().numpy()
            delta = p-zz
            acc["identity"].append(_d(zz, yy)); acc["model"].append(_d(p, yy))
            acc["proj"].append(_d(zz+pi@delta, yy)); acc["perp"].append(_d(zz+delta-pi@delta, yy))
            acc["po"].append(_d(_direct(po, w, z, grid)[0].double().numpy(), yy))
            acc["e16"] += float(((pi@delta)**2).sum()); acc["e"] += float((delta**2).sum())
        s = r["results"][str(g)][f"{t:g}"]
        m = {k: float(np.mean(v)) for k, v in acc.items() if isinstance(v, list)}
        assert s["distance"]["identity"] == pytest.approx(m["identity"], rel=1e-5)
        assert s["distance"]["model"] == pytest.approx(m["model"], rel=1e-5)
        assert s["distance"]["model_proj_K16"] == pytest.approx(m["proj"], rel=1e-5)
        assert s["distance"]["model_perp_K16"] == pytest.approx(m["perp"], rel=1e-5)
        assert s["distance"]["spectrum_model"] == pytest.approx(m["po"], rel=1e-5)
        assert s["gain"]["model_proj_K16"] == pytest.approx(1-m["proj"]/m["identity"], rel=1e-4)
        assert s["energy_in_span"]["model_K16"] == pytest.approx(acc["e16"]/acc["e"], rel=1e-5)
    with pytest.raises(SystemExit, match="exists"):
        tool.main([*args, "--output", str(out)])


def test_project_refuses_mismatched_or_non_po_spectrum_job(caches, real, env, tmp_path, monkeypatch):
    base, _ = caches
    tool = _tool("truncation_check")
    po, loss = _po()
    tf_job, po_job = _job(tmp_path/"tf", "transformer_consistency"), _job(tmp_path/"po", "po_jepa_full")
    path = po_job/"run"/"job"/"job.json"
    record = json.loads(path.read_text())
    path.write_text(json.dumps(dict(record, variant=dict(temporal_partition="dyadic", readout_init="zero"))))
    models = {str((tf_job/"run"/"job").resolve()): Leak(), str((po_job/"run"/"job").resolve()): po}
    monkeypatch.setattr(v7jobs, "load_for_evaluation", _loader(models, loss))
    args = ["project", "--job", str(tf_job), "--features", str(base), "--device", "cpu", "--grids", "8",
            "--output", str(tmp_path/"x.json")]
    with pytest.raises(SystemExit, match="variant"):
        tool.main([*args, "--spectrum-job", str(po_job)])
    path.write_text(json.dumps(dict(record, seed=1704)))
    with pytest.raises(SystemExit, match="seed"):
        tool.main([*args, "--spectrum-job", str(po_job)])
    with pytest.raises(SystemExit, match="PO-JEPA"):
        tool.main([*args, "--spectrum-job", str(tf_job)])
    assert not (tmp_path/"x.json").exists()


def _channel_projector(base, g, k=64):
    from po_jepa.geometry import regular_grid
    cov = 0.
    transfer = None if g == 16 else comparison.build_spatial_transfer(regular_grid(16, 16), target_height=g, target_width=g)
    for tid in range(32, 36):
        z = torch.from_numpy(np.load(base/f"{tid:04d}"/"raw_z.npy").astype(np.float32))
        if transfer is not None:
            z = comparison.apply_spatial_transfer(z, transfer)
        z = z.double().numpy()
        for s in (0, 5, 10, 15, 20):
            for t in (1, 2, 4, 8):
                d = z[s+t]-z[s]; cov = cov+d.T@d
    values, vectors = np.linalg.eigh(cov)
    u = vectors[:, ::-1][:, :k]
    return u@u.T, float(values[::-1][:k].sum()/values.sum())


def test_ceiling(caches, real, env, tmp_path, monkeypatch):
    base, _ = caches
    tool = _tool("truncation_check")
    po, loss = _po()
    job = _job(tmp_path/"po", "po_jepa_full")
    monkeypatch.setattr(v7jobs, "load_for_evaluation", _loader({str((job/"run"/"job").resolve()): po}, loss))
    out = tmp_path/"ceiling.json"
    tool.main(["ceiling", "--job", str(job), "--features", str(base), "--device", "cpu", "--grids", "16,8", "--steps", "12",
               "--batch", "7", "--lr", "1e-3", "--output", str(out)])
    r = json.loads(out.read_text())
    assert (r["kind"], r["p_out_rank"], r["steps"], r["batch"]) == ("truncation_ceiling", 64, 12, 7)
    q, _ = np.linalg.qr(po.p_out.weight.detach().double().numpy())
    qq = q@q.T
    for g, tokens in ((16, 256), (8, 64)):
        cu, explained = _channel_projector(base, g)
        assert r["channel_basis"][str(g)]["explained_share_of_validation_changes"] == pytest.approx(explained, rel=1e-6)
        for t in (1., 2., 4., 8.):
            s = r["results"][str(g)][f"{t:g}"]
            assert s["windows"] == 20 and s["tokens"] == tokens and s["modal_K64_unrestricted"] is (tokens == 64)
            for name in tool.CEILINGS:
                assert s["distance"][name] <= min(s["distance"][f"l2_{name}"], s["distance"]["identity"])+1e-7
                assert s["gain"][name] == pytest.approx(1-s["distance"][name]/s["distance"]["identity"], rel=1e-9)
            if t in (1., 8.):
                acc = dict(identity=[], k16=[], pout=[], u64=[])
                for key in [k for k in KEYS if k[2] == t]:
                    _, z, y, grid = _view(base, key, g)
                    pi = _projector(po, z, grid, 16)
                    zz, yy = z[0].double().numpy(), y[0].double().numpy()
                    delta = yy-zz
                    acc["identity"].append(_d(zz, yy)); acc["k16"].append(_d(zz+pi@delta, yy))
                    acc["pout"].append(_d(zz+pi@delta@qq, yy)); acc["u64"].append(_d(zz+delta@cu, yy))
                m = {k: float(np.mean(v)) for k, v in acc.items()}
                assert s["distance"]["identity"] == pytest.approx(m["identity"], rel=1e-5)
                assert s["distance"]["l2_modal_K16"] == pytest.approx(m["k16"], rel=1e-5)
                assert s["distance"]["l2_modal_K16_pout"] == pytest.approx(m["pout"], rel=1e-5)
                assert s["distance"]["l2_token_U64"] == pytest.approx(m["u64"], rel=1e-4)
    full = r["results"]["8"]["4"]
    assert full["distance"]["l2_modal_K64"] < 1e-5 and full["gain"]["modal_K64"] == pytest.approx(1., abs=1e-4)
    # the optimiser improves on the L2 start under the LN metric, and the identity (D = 0) bounds every ceiling
    for g in ("16", "8"):
        s = r["results"][g]["8"]["distance"]
        assert s["token_U64"] < s["l2_token_U64"] and s["modal_K16_pout"] < s["l2_modal_K16_pout"]
        assert all(s[n] <= s["identity"]+1e-9 for n in tool.CEILINGS)


def _action_at(controls, t):
    n = int(controls.n_segments[0])
    starts, ends = controls.starts[0, :n].double(), controls.ends[0, :n].double()
    j = int(((starts <= t) & (t < ends)).nonzero()[0])
    return controls.u_values[0, j][None]


def test_dynamics(caches, real, env, tmp_path, monkeypatch):
    base, _ = caches
    tool = _tool("truncation_check")
    po, loss = _po()
    job = _job(tmp_path/"po", "po_jepa_full")
    monkeypatch.setattr(v7jobs, "load_for_evaluation", _loader({str((job/"run"/"job").resolve()): po}, loss))
    out = tmp_path/"dynamics.json"
    tool.main(["dynamics", "--job", str(job), "--features", str(base), "--device", "cpu", "--grids", "16,8", "--output", str(out)])
    r = json.loads(out.read_text())
    assert r["kind"] == "truncation_dynamics"
    for g in (16, 8):
        for t in (1., 2., 4., 8.):
            s = r["results"][str(g)][f"{t:g}"]
            a = s["attention"]
            assert s["windows"] == 20 and 0 <= s["kept_share_after_PsiT"]["F"] <= 1 and 0 <= s["kept_share_after_PsiT"]["Ba"] <= 1
            assert 1/s["top_r"]-1e-9 <= a["mean_max_weight"] <= 1 and 0 <= a["mean_entropy_over_log_r"] <= 1+1e-6
    # independent recomputation of one cell from the traced right-hand sides
    g, t = 16, 2.
    steps, norms, kept, entropy, maxw, tokens = 0, dict(C=0., AC=0., PsiT_Ba=0., PsiT_F=0., Ba_token=0.), \
        dict(F=0., Fall=0., Ba=0., Baall=0.), 0., 0., 0
    content = []
    for key in [k for k in KEYS if k[2] == t]:
        w, z, y, grid = _view(base, key, g)
        with torch.no_grad():
            trace = po.trace_forward(z, w.inputs.controls, w.inputs.duration, grid)
        pi = _projector(po, z, grid, 16)
        ctx = trace.query.dynamics
        for step in trace.integration.steps:
            rhs = step.result.rhs; nl = rhs.interaction
            steps += 1
            drive = torch.einsum("bncd,bd->bnc", ctx.b_field, _action_at(w.inputs.controls, float(step.start_time[0])))
            f_token = (nl.attention.aggregate+nl.action_residual)[0].double().numpy()
            ba = drive[0].double().numpy()
            for name, v in (("C", step.before), ("AC", rhs.free), ("PsiT_Ba", rhs.action_linear), ("PsiT_F", nl.nonlinear),
                            ("Ba_token", drive)):
                norms[name] += float(np.linalg.norm(v[0].double().numpy()))
            kept["F"] += float(((pi@f_token)**2).sum()); kept["Fall"] += float((f_token**2).sum())
            kept["Ba"] += float(((pi@ba)**2).sum()); kept["Baall"] += float((ba**2).sum())
            wts = nl.attention.weights[0].double().numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                entropy += float(-np.where(wts > 0, wts*np.log(wts), 0.).sum())
            maxw += float(wts.max(-1).sum()); tokens += wts.shape[0]
            content.append(nl.content.scores[0].double().numpy().ravel())
    s = r["results"]["16"]["2"]
    assert s["steps"] == steps
    for name, v in norms.items():
        assert s["mean_norm"][name] == pytest.approx(v/steps, rel=1e-5)
    assert s["kept_share_after_PsiT"]["F"] == pytest.approx(kept["F"]/kept["Fall"], rel=1e-5)
    assert s["kept_share_after_PsiT"]["Ba"] == pytest.approx(kept["Ba"]/kept["Baall"], rel=1e-5)
    assert s["attention"]["mean_entropy"] == pytest.approx(entropy/tokens, rel=1e-6)
    assert s["attention"]["mean_max_weight"] == pytest.approx(maxw/tokens, rel=1e-6)
    assert s["attention"]["content_score_std"] == pytest.approx(float(np.concatenate(content).std()), rel=1e-5)


def test_ceiling_and_dynamics_refuse_a_non_po_job(caches, real, env, tmp_path, monkeypatch):
    base, _ = caches
    tool = _tool("truncation_check")
    _, loss = _po()
    job = _job(tmp_path/"tf", "transformer_consistency")
    monkeypatch.setattr(v7jobs, "load_for_evaluation", _loader({str((job/"run"/"job").resolve()): Leak()}, loss))
    for command in ("ceiling", "dynamics"):
        with pytest.raises(SystemExit, match="PO-JEPA"):
            tool.main([command, "--job", str(job), "--features", str(base), "--device", "cpu", "--grids", "8",
                       "--output", str(tmp_path/f"{command}.json")])
        assert not (tmp_path/f"{command}.json").exists()


def test_cross_version_flag_uses_the_cross_version_loader(caches, real, env, tmp_path, monkeypatch):
    """C-2 jobs are read with --cross-version (addon.v7load.load_v7_selected, checked against --features)."""
    from po_jepa.runtime.addon import v7load
    base, _ = caches
    seen = {}

    def refuse(*a, **k):
        raise AssertionError("the default loader must not be used with --cross-version")

    def fake(job_dir, condition, model, *, loss_config, features, device):
        seen.update(condition=condition, features=features, dataset=datasets.active().tag)
        return dict(best=dict(step=3, checkpoint_sha256="cd"*32), job_dir=str(job_dir))
    monkeypatch.setattr(v7jobs, "load_for_evaluation", refuse)
    monkeypatch.setattr(v7load, "load_v7_selected", fake)
    out = tmp_path/"x.json"
    _tool("truncation_check").main(["dynamics", "--job", str(_job(tmp_path/"cv", "po_jepa_full")), "--cross-version",
                                    "--features", str(base), "--device", "cpu", "--grids", "8", "--output", str(out)])
    r = json.loads(out.read_text())
    assert seen == dict(condition="po_jepa_full", features=base, dataset=BASE.tag)
    assert r["registration_sha256"] is None and r["cross_version"]["best"]["step"] == 3 and r["best"]["step"] == 3


def test_readout_basis_spans_exactly_the_column_space_of_w_out():
    """G10 review P2-1: a rank-deficient W_out (thin QR would return 64 columns) and W_out = 0."""
    tool = _tool("truncation_check")
    g = torch.Generator().manual_seed(8)
    w = torch.zeros(1408, 64); w[0, 0] = 1.
    q, rank = tool.readout_basis(w)
    assert rank == 1 and q.shape == (1408, 1)
    np.testing.assert_allclose((q@q.T).numpy(), (w@w.T).numpy(), atol=1e-7)          # projector onto e_0
    low = torch.randn(1408, 3, generator=g)@torch.randn(3, 64, generator=g)
    q, rank = tool.readout_basis(low)
    a = low.double().numpy(); u = np.linalg.svd(a, full_matrices=False)[0][:, :3]
    assert rank == 3 and q.shape == (1408, 3)
    np.testing.assert_allclose((q@q.T).double().numpy(), u@u.T, atol=1e-6)
    full = torch.randn(1408, 64, generator=g)
    q, rank = tool.readout_basis(full)
    assert rank == 64 and float((q.T@q-torch.eye(64)).abs().max()) < 1e-5
    q, rank = tool.readout_basis(torch.zeros(1408, 64))
    assert rank == 0 and q.shape == (1408, 0)


def test_ceiling_with_a_zero_readout_is_the_identity(env):
    tool = _tool("truncation_check")
    loss = comparison.build_condition("identity", "cpu")[1]
    g = torch.Generator().manual_seed(9)
    z, target = torch.randn(2, 16, 1408, generator=g), torch.randn(2, 16, 1408, generator=g)
    psi = torch.linalg.qr(torch.randn(2, 16, 16, generator=g))[0]
    u = torch.linalg.qr(torch.randn(1408, 64, generator=g))[0]
    d0 = torch.tensor([_d(z[i].numpy(), target[i].numpy()) for i in range(2)], dtype=torch.float64)
    out = tool.ceiling_batch(z, target, psi, psi, torch.zeros(1408, 0), u, loss, 3, .01, d0)
    start, best = out["modal_K16_pout"]
    assert torch.allclose(start, d0) and torch.allclose(best, d0)
    # rank 1: the start point is the projection onto that one channel direction
    e = torch.zeros(1408, 1); e[5, 0] = 1.
    start, _ = tool.ceiling_batch(z, target, psi, psi, e, u, loss, 1, .01, d0)["modal_K16_pout"]
    p = (z+psi@(psi.transpose(1, 2)@(target-z))@e@e.T).numpy()
    assert [float(x) for x in start] == pytest.approx([_d(p[i], target[i].numpy()) for i in range(2)], rel=1e-5)
