"""S18a：截断间隙（10 §5）。手算的桩谱、真实 GraphSpectrum 上 λ_{K+1} 和完整特征分解对照、分位数和判断规则、任务入口。"""

import json
from types import SimpleNamespace

import pytest
import torch

from po_jepa import datasets
from po_jepa.geometry import regular_grid
from po_jepa.runtime import cache, checks, comparison, evaluation
from po_jepa.runtime.addon import spectral_gap
from po_jepa.spectral import GraphConfig, GraphSpectrum
from test_addon_spectrum_for import one_thread, protocol_z  # noqa: F401  (one_thread 是 fixture)


def stub(eigenvalues, cutoff):
    """批大小 1 的 SpectralResult 替身：只有 gap_record 读的三个字段。"""
    values = torch.tensor([eigenvalues], dtype=torch.float64)
    gaps = values[:, 1:]-values[:, :-1]
    return SimpleNamespace(eigenvalues_cpu=values, diagnostics=SimpleNamespace(
        cutoff_gap=None if cutoff is None else torch.tensor([cutoff], dtype=torch.float64), consecutive_gaps=gaps))


def test_gap_record_hand_values():
    """λ_i = 0.01·(i−1)（i = 1…16），λ_17 = 0.5：λ_16 = 0.15，截断间隙 0.35，15 个相邻间隙都是 0.01，比值 35。"""
    r = spectral_gap.gap_record(stub([.01*i for i in range(16)], .35))
    assert r["lambda_k"] == pytest.approx(.15, abs=1e-15) and r["lambda_k1"] == pytest.approx(.5, abs=1e-15)
    assert r["cutoff_gap"] == .35 and r["median_gap"] == pytest.approx(.01, rel=1e-12)
    assert r["ratio"] == pytest.approx(35., rel=1e-12)
    r = spectral_gap.gap_record(stub([0., .1, .1, .4, .5], .02))       # 间隙 .1、0、.3、.1：中位数 .1（偶数个取中间两个的平均）
    assert r["median_gap"] == pytest.approx(.1) and r["ratio"] == pytest.approx(.2)
    r = spectral_gap.gap_record(stub([.2]*16, .3))
    assert r["median_gap"] == 0. and r["ratio"] is None
    with pytest.raises(ValueError, match="cutoff gap is undefined"):
        spectral_gap.gap_record(stub([.1, .2], None))


@pytest.mark.parametrize("size", [16, 8, 12])
def test_window_gaps_match_full_eigendecomposition(one_thread, size):
    """真实协议图：λ_K、λ_{K+1} 和稠密 eigvalsh 的第 16、17 个特征值相同（1e-10）；8、12 网格先用 R 变换。"""
    spectrum = GraphSpectrum(GraphConfig(**checks.load_protocol()["model"]["graph"]))
    z, grid = protocol_z(1, 16, 16, 1703), regular_grid(16, 16)
    out = spectral_gap.window_gaps(spectrum, z, grid, 16)
    assert set(out) == {"16", "8", "12"}
    if size == 16:
        source, target = z, grid
    else:
        transfer = comparison.build_spatial_transfer(grid, target_height=size, target_width=size)
        source, target = comparison.apply_spatial_transfer(z, transfer), transfer.target
    laplacian = spectrum.normalized_laplacian(spectrum.affinity(source, target)).laplacian
    full = torch.linalg.eigvalsh(laplacian[0].double())
    r = out[str(size)]
    assert r["lambda_k"] == pytest.approx(float(full[15]), abs=1e-10)
    assert r["lambda_k1"] == pytest.approx(float(full[16]), abs=1e-10)
    gaps = sorted(float(full[i+1]-full[i]) for i in range(15))
    assert r["median_gap"] == pytest.approx(gaps[7], abs=1e-10)
    assert r["ratio"] == pytest.approx(r["cutoff_gap"]/r["median_gap"], rel=1e-12)


def windows_with(ratios_by_grid):
    count = len(next(iter(ratios_by_grid.values())))
    out = []
    for i in range(count):
        grids = {}
        for size, ratios in ratios_by_grid.items():
            ratio = ratios[i]
            grids[size] = dict(lambda_k=.1*i, lambda_k1=.1*i+1., cutoff_gap=1., median_gap=None if ratio is None else (1./ratio if ratio else 0.),
                               ratio=ratio)
            if ratio is None:
                grids[size]["median_gap"] = 0.
        out.append(dict(grids=grids))
    return out


def test_summarize_quantiles_and_decision():
    w = windows_with({"16": [1., 2., 3., 4., 5.], "8": [2., None, 4., 6., 8.], "12": [.5, .9, 1., 3., 4.]})
    s = spectral_gap.summarize(w)
    assert s["quantile_levels"] == [.05, .25, .5, .75, .95]
    assert s["grids"]["16"]["ratio"] == pytest.approx([1.2, 2., 3., 4., 4.8])     # linear 插值
    assert s["grids"]["8"]["undefined_ratio"] == 1 and s["grids"]["8"]["ratio"] == pytest.approx([2.3, 3.5, 5., 6.5, 7.7])
    assert s["grids"]["16"]["lambda_k"] == pytest.approx([.02, .1, .2, .3, .38])
    assert s["grids"]["12"]["median_ratio"] == 1. and s["grids"]["12"]["median_ratio_above_1"] is False   # 等于 1 不算大于 1
    assert s["keep_sentence"] is False and s["decision"] == spectral_gap.REWRITE
    s = spectral_gap.summarize(windows_with({"16": [1.5, 2.], "8": [1.01, 3.], "12": [1.2, 1.3]}))
    assert s["keep_sentence"] is True and s["decision"] == spectral_gap.KEEP_SENTENCE
    s = spectral_gap.summarize(windows_with({"16": [None, None], "8": [2., 3.], "12": [2., 3.]}))
    assert s["grids"]["16"]["ratio"] is None and s["grids"]["16"]["median_ratio"] is None and s["keep_sentence"] is False
    assert spectral_gap.REWRITE == "K = 16 is fixed in advance and shared across grids"


def big_window(tid, start):
    z = protocol_z(1, 16, 16, 1000*tid+start)
    controls = cache._window_controls(dict(trajectory_id=tid, controls=dict(starts_ticks=list(range(32)), ends_ticks=list(range(1, 33)),
                                      values=[[0., 0.]]*32, units="simulation_position_per_second")), start, 1., "cpu")
    x = cache.WindowInput(z, controls, torch.tensor([1.], dtype=torch.float64), cache.regular_grid(16, 16),
                          torch.tensor([start*.25], dtype=torch.float64), "input", tid, "test")
    y = cache.WindowTarget(z.clone(), torch.tensor([(start+1)*.25], dtype=torch.float64), "target")
    return cache.CacheWindow(x, y, dict(actual_manifest_sha256=f"manifest-{tid}"))


@pytest.fixture
def gap_job(monkeypatch, tmp_path, one_thread):
    old = datasets._set_for_tests(datasets.DatasetSpec("van_der_pol", 40))
    seen = []
    def load(features, q, device):
        seen.append((q.trajectory_id, q.start_tick, q.duration_cycles, q.grid_size, q.kind, str(device)))
        return big_window(q.trajectory_id, q.start_tick)
    monkeypatch.setattr(evaluation, "load_test_window", load)
    yield tmp_path, seen
    datasets._set_for_tests(old)


def test_run_spectral_gap_layout_and_rerun(gap_job):
    tmp, seen = gap_job
    report = spectral_gap.run_spectral_gap(tmp/"features", tmp/"a", "cpu")
    spectral_gap.run_spectral_gap(tmp/"features", tmp/"b", torch.device("cpu"))
    names = ["COMPLETE.json", "job.json", "report.json"]
    assert sorted(p.name for p in (tmp/"a").iterdir()) == names
    assert all((tmp/"a"/n).read_bytes() == (tmp/"b"/n).read_bytes() for n in names)
    assert json.loads((tmp/"a"/"COMPLETE.json").read_text()) == dict(report_sha256=cache.file_sha(tmp/"a"/"report.json"))
    assert len(report["windows"]) == 20 and seen[0] == (36, seen[0][1], 1., 16, "point", "cpu")
    assert [(w["trajectory_id"], w["start_tick"]) for w in report["windows"]] == [(t, s) for t, s, *_ in seen[:20]]
    graph = checks.load_protocol()["model"]["graph"]
    assert report["graph"] == dict(q_seed=graph["q_seed"], sigma_x=graph["sigma_x"], sigma_z=graph["sigma_z"],
                                   affinity_source=graph["affinity_source"], k=16)
    assert report["job"]["kind"] == "spectral_gap" and report["job"]["k"] == 16 and report["job"]["dataset"] == "van_der_pol-40"
    assert report["fixed_fourier"]["cutoff_gap"] == 0.
    assert report["summary"]["grids"]["16"]["windows"] == 20
    assert cache._json_bytes(report) == (tmp/"a"/"report.json").read_bytes()


def test_run_spectral_gap_refusals(gap_job, monkeypatch):
    tmp, _ = gap_job
    with pytest.raises(ValueError, match="CPU"):
        spectral_gap.run_spectral_gap(tmp/"features", tmp/"gpu", "cuda:0")
    assert not (tmp/"gpu").exists()
    (tmp/"exists").mkdir()
    with pytest.raises(FileExistsError):
        spectral_gap.run_spectral_gap(tmp/"features", tmp/"exists", "cpu")
    config = checks.load_protocol()
    config["model"]["graph"]["fixed_modes"] = False
    monkeypatch.setattr(checks, "load_protocol", lambda: config)
    with pytest.raises(ValueError, match="fixed K"):
        spectral_gap.run_spectral_gap(tmp/"features", tmp/"adaptive", "cpu")
