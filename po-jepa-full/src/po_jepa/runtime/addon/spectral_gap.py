"""10 §5：截断间隙（run.py spectral-gap 的计算部分，CPU，每个场景一次）。

图只由编码器特征、Q（种子 1703）、σ_x 和本场景的 σ_z 决定，和条件、种子无关。对全部测试窗口的起点帧，在
16×16（z）、8×8（R z）、12×12（R z）上用 po_jepa_full 的 GraphSpectrum.spectrum_for 求最低 K 个特征对
（K = 协议的 model.graph.max_modes，仿真 16），每个窗口每个网格记：

- lambda_k = λ_K（eigenvalues_cpu 的最后一个），lambda_k1 = λ_K + cutoff_gap（= λ_{K+1}），
- cutoff_gap = diagnostics.cutoff_gap（λ_{K+1} − λ_K），
- median_gap = diagnostics.consecutive_gaps（前 K 个模态的 K−1 个相邻间隙）的中位数（statistics.median），
- ratio = cutoff_gap / median_gap；median_gap = 0 时为 None（计入 undefined_ratio，不进分位数）。

报告每个网格上这五个量的 5、25、50、75、95 分位数（torch.quantile，float64，linear 插值）。预先定好的规则：
每个网格上 ratio 的中位数（50 分位）都大于 1，论文保留原句；否则改成 “K = 16 is fixed in advance and shared
across grids”。fixed_fourier 的截断间隙由公式给出，恒为 0（08 §4.3），报告里单列一行。
"""

import statistics

from pathlib import Path

import torch

from .. import cache, checks, comparison, data, evaluation
from . import d4

GRIDS = (16, 8, 12)
QUANTILE_LEVELS = (.05, .25, .5, .75, .95)
FIELDS = ("lambda_k", "lambda_k1", "cutoff_gap", "median_gap", "ratio")
KEEP_SENTENCE = "keep the original sentence (median ratio > 1 on every grid)"
REWRITE = "K = 16 is fixed in advance and shared across grids"


def gap_record(result) -> dict:
    """一个 SpectralResult（批大小 1）→ 一条记录。"""
    eigenvalues = result.eigenvalues_cpu[0]
    if result.diagnostics.cutoff_gap is None:
        raise ValueError("K equals the number of tokens; the cutoff gap is undefined")
    cutoff = float(result.diagnostics.cutoff_gap[0])
    gaps = [float(g) for g in result.diagnostics.consecutive_gaps[0]]
    median = statistics.median(gaps)
    lam = float(eigenvalues[-1])
    return dict(lambda_k=lam, lambda_k1=lam+cutoff, cutoff_gap=cutoff, median_gap=median,
                ratio=cutoff/median if median > 0 else None)


def window_gaps(spectrum, raw_z, grid, k) -> dict:
    """一个窗口起点帧在三个网格上的记录：{"16": …, "8": …, "12": …}。"""
    out = {}
    for size in GRIDS:
        if size == 16:
            source, target = raw_z, grid
        else:
            transfer = comparison.build_spatial_transfer(grid, target_height=size, target_width=size)
            source, target = comparison.apply_spatial_transfer(raw_z, transfer), transfer.target
        out[str(size)] = gap_record(spectrum.spectrum_for(source, target, k=k, model_device=torch.device("cpu")))
    return out


def summarize(windows) -> dict:
    """每个网格上五个量的分位数、ratio 中位数是否大于 1，以及判断结果。"""
    grids, keep = {}, True
    for size in GRIDS:
        records = [w["grids"][str(size)] for w in windows]
        entry = dict(windows=len(records), undefined_ratio=sum(r["ratio"] is None for r in records))
        for name in FIELDS:
            values = [r[name] for r in records if r[name] is not None]
            entry[name] = (torch.quantile(torch.tensor(values, dtype=torch.float64),
                                          torch.tensor(QUANTILE_LEVELS, dtype=torch.float64)).tolist() if values else None)
        entry["median_ratio"] = entry["ratio"][2] if entry["ratio"] else None
        entry["median_ratio_above_1"] = entry["median_ratio"] is not None and entry["median_ratio"] > 1
        keep = keep and entry["median_ratio_above_1"]
        grids[str(size)] = entry
    return dict(quantile_levels=list(QUANTILE_LEVELS), grids=grids, keep_sentence=keep,
                decision=KEEP_SENTENCE if keep else REWRITE)


def run_spectral_gap(features, output, device):
    """一个场景的截断间隙：写 output/job.json、output/report.json、output/COMPLETE.json（output 必须是新目录）。
    文件名和 D 任务、D4 任务相同，汇总按 job.json + COMPLETE.json + report.json 找任务（13 §1）。"""
    if torch.device(device).type != "cpu":
        raise ValueError("spectral-gap runs on CPU (10 §5)")
    features, output = Path(features), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = checks.load_protocol()
    graph = config["model"]["graph"]
    if not graph["fixed_modes"]:
        raise ValueError("spectral-gap expects a fixed K shared across grids")
    k = graph["max_modes"]
    job = dict(kind="spectral_gap", dataset=data.datasets.active().tag, updates=checks.v7_updates(), k=k)
    cache._write_json(output/"job.json", job)
    spectrum = checks.build_full_model(config, torch.device("cpu")).spectrum
    windows = []
    with torch.no_grad():
        for tid, start in d4.windows_index():
            w = evaluation.load_test_window(features, evaluation.EvaluationQuery(tid, start, 1., 16, "point"), "cpu")
            windows.append(dict(trajectory_id=tid, start_tick=start, manifest_sha256=w.provenance["actual_manifest_sha256"],
                                grids=window_gaps(spectrum, w.inputs.raw_z, w.inputs.grid, k)))
    report = dict(job=job, graph=dict(q_seed=graph["q_seed"], sigma_x=graph["sigma_x"], sigma_z=graph["sigma_z"],
                                      affinity_source=graph["affinity_source"], k=k),
                  windows=windows, summary=summarize(windows),
                  fixed_fourier=dict(cutoff_gap=0.0, note="08 §4.3: the DCT-II cutoff gap is zero by formula"),
                  scientific_conclusions_verified=False)
    cache._write_json(output/"report.json", report)
    cache._write_json(output/"COMPLETE.json", dict(report_sha256=cache.file_sha(output/"report.json")))
    return report
