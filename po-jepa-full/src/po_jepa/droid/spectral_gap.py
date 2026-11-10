"""DROID 截断间隙（docs/design/10-evaluation.md §8.8，S11b）：每个配置一次的 CPU 任务 `eval/spectral-gap`。

和仿真的 10 §5（`runtime/addon/spectral_gap.py`）同一做法、同一套函数：两个测试集冻结窗口里 d = 1 的全部窗口的起点行
（主特征第 s 行），在 16×16（z）、8×8（R_8 z）、12×12（R_12 z）上用 `GraphSpectrum.spectrum_for` 求最低 K = 16 个特征对，
每个窗口每个网格记 λ_K、λ_{K+1}、`cutoff_gap`、前 K 个模态 K − 1 个相邻间隙的中位数和两者之比（`addon.spectral_gap.
window_gaps`）；报告每个网格上的 5、25、50、75、95 分位数和预先定好的判断规则（`addon.spectral_gap.summarize`：每个网格上
比值的中位数都严格大于 1 才保留原句）。判断用两个测试集合在一起的窗口，每个测试集各自的分位数另记。

图：09 §2.1 的模型配置块里的 `graph` 段（po-jepa-full `configs/train/default.yaml` 深度合并 `v4.yaml`，读之前
`cache.verify_fixed_protocol()`），只把 `sigma_z` 换成 DROID 的标定值（`<run-root>/<name>/calibration/sigma_z.json`，
09 §3.1，S09 写）。K、σ_x 必须等于 DROID 配置的 `model.modes`、`model.sigma_x`。图和条件、种子无关。

输出 `<run-root>/<name>/eval/spectral-gap/`：`job.json`、`report.json`、`COMPLETE.json`（`report.json` 的 SHA）。
已完成且 job（配置、sigma_z、输入 SHA）和窗口都与本次相同时直接返回，不同报错；没完成时整份重算（几千个 256×256 的 eigh，CPU 几分钟）。

  python -m po_jepa.droid.spectral_gap --config configs/droid/<name>.yaml --data-root <数据根> --run-root <运行根>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping

import torch
import yaml

from po_jepa.droid import evaluate as ev
from po_jepa.droid import features as ft
from po_jepa.droid import index as ix
from po_jepa.droid import prepare as pr
from po_jepa.droid.config import DroidConfig, load_config
from po_jepa.geometry import regular_grid
from po_jepa.runtime import cache, checks
from po_jepa.runtime.addon import spectral_gap as sim
from po_jepa.spectral import GraphConfig, GraphSpectrum

SCHEMA_JOB = "droid_spectral_gap_job_v1"
SCHEMA_REPORT = "droid_spectral_gap_report_v1"
JOB_DIR = ("eval", "spectral-gap")
DURATION = 1                                   # 10 §8.8：d = 1 的全部窗口


def graph_block(sigma_z: float) -> dict:
    """09 §2.1 的 `model.graph`（default.yaml ← v4.yaml），`sigma_z` 换成 DROID 的标定值。"""
    if type(sigma_z) is not float or not math.isfinite(sigma_z) or sigma_z <= 0:
        raise ValueError(f"sigma_z must be a positive finite float, got {sigma_z!r}")
    cache.verify_fixed_protocol()
    root = cache.ROOT/"configs"/"train"
    base = yaml.safe_load((root/"default.yaml").read_text(encoding="utf-8"))
    overlay = yaml.safe_load((root/"v4.yaml").read_text(encoding="utf-8"))
    merged = checks._merge_v4(base, {k: v for k, v in overlay.items() if k not in ("schema_version", "protocol_id")})
    return dict(merged["model"]["graph"], sigma_z=sigma_z)


def build_spectrum(config: DroidConfig, sigma_z: float) -> tuple[GraphSpectrum, dict]:
    graph = graph_block(sigma_z)
    if (graph["max_modes"] != config.model.modes or graph["sigma_x"] != config.model.sigma_x
            or graph["fixed_modes"] is not True):
        raise ValueError(f"the PO-JEPA graph (K = {graph['max_modes']}, sigma_x = {graph['sigma_x']}, fixed_modes = "
                         f"{graph['fixed_modes']}) differs from the DROID config (modes {config.model.modes}, sigma_x "
                         f"{config.model.sigma_x})")
    return GraphSpectrum(GraphConfig(**graph)).eval(), graph


def read_sigma_z(path: Path | str, config: DroidConfig) -> tuple[float, str]:
    """`sigma_z.json`（09 §3.1）的 `sigma_z` 和文件 SHA；记了 `config_sha256` 时必须等于本配置的。"""
    path = Path(path)
    doc = ix._loads(path.read_bytes(), str(path))
    value = doc.get("sigma_z") if isinstance(doc, dict) else None
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path}: sigma_z must be a positive finite float, got {value!r}")
    if "config_sha256" in doc and doc["config_sha256"] != config.config_sha256:
        raise ValueError(f"{path} was calibrated under another config")
    return value, cache.file_sha(path)


def gap_windows(docs: Mapping[str, Mapping]) -> list[dict]:
    """两个测试集里 d = 1 的全部窗口（文件顺序，先 id_test 后 ood_test）。"""
    if set(docs) != set(ev.TEST_SPLITS):
        raise ValueError(f"the spectral gap uses both test splits {ev.TEST_SPLITS}")
    out = []
    for split in ev.TEST_SPLITS:
        ev._windows_doc(docs[split], split)
        out += [dict(split=split, id=w["id"], uuid=w["uuid"], start=w["start"])
                for w in docs[split]["windows"] if w["duration"] == DURATION]
    if not out:
        raise ValueError("no d = 1 test windows")
    return out


def run_spectral_gap(output: Path | str, *, docs: Mapping[str, Mapping], reader: ft.FeatureReader, config: DroidConfig,
                     sigma_z: float, provenance: Mapping) -> dict:
    """`<output>/job.json`、`report.json`、`COMPLETE.json`。`provenance` 是写进 job.json 的输入 SHA（窗口文件、特征
    清单、sigma_z.json）。已完成时，本次的 job 和 d = 1 窗口与已存的完全相同才返回已有的报告，否则报错。"""
    out = Path(output)
    spectrum, graph = build_spectrum(config, sigma_z)          # 先核 sigma_z、K、σ_x，再看完成缓存（S11b-fix1）
    k = graph["max_modes"]
    job = dict(schema=SCHEMA_JOB, kind="droid_spectral_gap", config_name=config.name, config_sha256=config.config_sha256,
               duration=DURATION, k=k, sigma_z=sigma_z, inputs=dict(provenance))
    want = ix._loads(cache._json_bytes(job), "job")
    selected = gap_windows(docs)
    job_path = out/"job.json"
    if (out/"COMPLETE.json").exists():                 # 完成缓存只在配置、sigma_z、来源和窗口都相同时复用
        report = ix._loads((out/"report.json").read_bytes(), "report.json")
        if ix._loads((out/"COMPLETE.json").read_bytes(), "COMPLETE.json").get("report_sha256") != cache.file_sha(out/"report.json"):
            raise ValueError(f"{out}: COMPLETE.json does not match report.json")
        stored = [{n: w.get(n) for n in ("split", "id", "uuid", "start")} for w in report.get("windows", [])]
        if (not job_path.exists() or ix._loads(job_path.read_bytes(), "job.json") != want
                or report.get("job") != want or stored != selected):
            raise ValueError(f"{out} is complete for another request (config, sigma_z, inputs or test windows "
                             "changed); use a new output directory")
        return report
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.partial"):
        stale.unlink()
    if job_path.exists() and ix._loads(job_path.read_bytes(), "job.json") != want:
        raise ValueError(f"{job_path} differs from this command (config, sigma_z or inputs changed)")
    if not job_path.exists():
        cache._write_json(job_path, job)
    grid = regular_grid(ev.GRID, ev.GRID)
    windows = []
    with torch.no_grad():
        for w in selected:
            raw = reader.rows(w["uuid"], [w["start"]])                 # [1, 256, 1408]，逐行核 SHA
            windows.append(dict(w, grids=sim.window_gaps(spectrum, raw, grid, k)))
    report = dict(schema=SCHEMA_REPORT, job=job,
                  graph=dict(q_seed=graph["q_seed"], sigma_x=graph["sigma_x"], sigma_z=sigma_z,
                             affinity_source=graph["affinity_source"], k=k),
                  windows=windows, summary=sim.summarize(windows),
                  by_split={s: sim.summarize([w for w in windows if w["split"] == s]) for s in ev.TEST_SPLITS},
                  fixed_fourier=dict(cutoff_gap=0.0, note="08 §4.3: the DCT-II cutoff gap is zero by formula"),
                  scientific_conclusions_verified=False)
    (out/"report.json").unlink(missing_ok=True)
    cache._write_json(out/"report.json", report)
    cache._write_json(out/"COMPLETE.json", dict(report_sha256=cache.file_sha(out/"report.json")))
    return report


def job_dir(run_root: Path | str, config: DroidConfig) -> Path:
    return Path(run_root)/config.name/Path(*JOB_DIR)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--run-root", required=True, type=Path)
    a = p.parse_args(argv)
    config = load_config(a.config)
    windows_dir = pr.prepared_dir(a.data_root)/pr.WINDOWS_DIR
    docs, shas = {}, {}
    for split in ev.TEST_SPLITS:
        path = windows_dir/f"{split}.json"
        docs[split], shas[split] = ix._loads(path.read_bytes(), str(path)), cache.file_sha(path)
    manifests, manifest_sha = ft.feature_manifests(a.data_root, config, shards=list(ev.TEST_SPLITS))
    sigma_path = Path(a.run_root)/config.name/"calibration"/"sigma_z.json"
    sigma_z, sigma_sha = read_sigma_z(sigma_path, config)
    provenance = dict(windows_sha256=shas, feature_manifest_sha256=manifest_sha, sigma_z_sha256=sigma_sha)
    run_spectral_gap(job_dir(a.run_root, config), docs=docs, reader=ft.FeatureReader(a.data_root, manifests),
                     config=config, sigma_z=sigma_z, provenance=provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
