"""13 §1–§7：把 v7 正式运行和仿真补跑的结果合成表和图（只读，CPU）。

用法（在 po-jepa-full 目录里）：
  python tools/summarize_addon.py --v7-results ../../results-v7/full --addon-results ../../results-addon/full \
      --output ../../results-addon/full/summary
输出 <输出>/summary.json、summary.md、figures/*.svg、missing.json、checks.json。先写到同一父目录下的临时目录，全部写完
再整体换名；已有的 <输出> 先改名为 <输出>.previousN（N 从 1 起取第一个不存在的）。

任务：两个结果根下任意层级里同时有 job.json、COMPLETE.json、report.json 的目录（run.py 的输出是 <目录>/run/job）。
同一任务（种类、数据集、训练数据集、种子、步数、条件都相同）有两个完成的目录时报错退出（13 §1），什么都不写。
某个场景的数据一致性核对（13 §4）有一项不通过，这个场景只写 checks.json 里的结果，不出表和图；全部写完后退出码 1。
缺的任务列进 missing.json，不影响退出码。
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from po_jepa.runtime import cache, plots  # noqa: E402
from po_jepa.runtime.addon import d4  # noqa: E402

V7_CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity", "vjepa2_ac_small", "vjepa2_ac")
ADDON_CONDITIONS = ("po_jepa_time_only", "po_jepa_space_only", "transformer_consistency", "fixed_fourier", "koopman")
CONDITIONS = V7_CONDITIONS+ADDON_CONDITIONS
TRAINED = tuple(c for c in CONDITIONS if c != "identity")
REFERENCES = tuple(c for c in CONDITIONS if c != "po_jepa_full")
AC = ("vjepa2_ac_small", "vjepa2_ac")
PO_CLASS = ("po_jepa_full", "po_jepa_no_consistency", "po_jepa_time_only", "po_jepa_space_only", "fixed_fourier")
ABLATION = ("po_jepa_full", "po_jepa_time_only", "po_jepa_space_only", "po_jepa_no_consistency")
TRAIN_TIMES = (1., 2., 4., 8.)
HELDOUT_TIMES = (3., 12.)
MATCH_TOLERANCE = .05                  # 13 §2：|e_a − e_b| ≤ 0.05·min(e_a, e_b)，等号算匹配
BOOTSTRAP_REPS = 2000                  # 13 §3
BOOTSTRAP_SEED = 1703
NOT_COMPARABLE = {"tcd_normalized", "scd_normalized", "pd_in", "xpd_in"}   # 10 §3.1、§3.8：AC 只比 LN 空间的量

# v7 评测记录上的量：(kind, 网格, 时长, 值的键)
ROW_METRICS = {
    "future": ("point", (16,), TRAIN_TIMES, "future_jepa_distance"),
    "future_heldout_time": ("point", (16,), HELDOUT_TIMES, "future_jepa_distance"),
    "future_heldout_grid": ("point", (12,), TRAIN_TIMES, "future_jepa_distance"),
    "future_heldout_both": ("point", (12,), HELDOUT_TIMES, "future_jepa_distance"),
    "tcd_normalized": ("point", (16,), TRAIN_TIMES, "tcd_normalized"),
    "scd_normalized": ("pair", (8, 12), TRAIN_TIMES, "scd_normalized"),
    "delta_act": ("point", (16,), TRAIN_TIMES, "delta_act"),
}
# D4 记录上的量：一组键在每个窗口里各算一个值（窗口键带上量的名字）
D4_GROUPS = {
    "tcd_ln": tuple(f"tcd_ln:T={t:g}" for t in TRAIN_TIMES),
    "scd_ln": tuple(f"scd_ln:h={h}:T={t:g}" for h in (8, 12) for t in TRAIN_TIMES),
    **{f"pd_{m}": tuple(f"pd_{m}:T={t:g}:{d4.partition_label(p)}" for t, p in d4.PARTITIONS) for m in ("ln", "in")},
    **{f"xpd_{m}": tuple(f"xpd_{m}:T={t:g}:n={n}" for t, n in d4.CROSS) for m in ("ln", "in")},
}
# D4 长时程记录上的量（每条轨迹一条，窗口键是 (轨迹, 量的名字)）
LONG_GROUPS = {f"long_{k}": tuple(f"long_{k}:T={t:g}" for t in d4.LONG_TIMES) for k in ("direct", "composed")}
T1_METRICS = ("future", "tcd_ln", "scd_ln", "tcd_normalized", "scd_normalized", "delta_act")
T2_METRICS = ("future_heldout_time", "future_heldout_grid", "future_heldout_both")
# 13 §2：比较缺陷（TCD、SCD、分割差异）和泛化（留出、长时程）的配对表；变体的匹配标记在 T8 的每一行
PAIRED_METRICS = T1_METRICS+T2_METRICS+("pd_ln", "pd_in", "xpd_ln", "xpd_in", "long_direct", "long_composed")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------- 读任务（13 §1） ----------

def find_jobs(*roots):
    """和 v7 summarize_v7.find_jobs 相同：有 job.json、COMPLETE.json、report.json 的目录。"""
    for root in roots:
        for path in sorted(Path(root).rglob("job.json")):
            folder = path.parent
            if (folder/"COMPLETE.json").exists() and (folder/"report.json").exists():
                yield folder


def identity_of(job):
    return (job.get("kind") or "run", job["dataset"], job.get("train_dataset"), job.get("seed"), job.get("updates"), job.get("condition"))


def collect(*roots):
    """{identity: folder}；同一 identity 有两个完成的目录报 ValueError。"""
    found, duplicates = {}, []
    for folder in find_jobs(*roots):
        key = identity_of(read(folder/"job.json"))
        if key in found:
            duplicates.append(dict(job=list(key), folders=[str(found[key]), str(folder)]))
        else:
            found[key] = folder
    if duplicates:
        raise ValueError(f"the same job completed twice: {json.dumps(duplicates, ensure_ascii=False)}")
    return found


def _row_key(q):
    return (q["kind"], q["trajectory_id"], q["start_tick"], float(q["duration_cycles"]), q["grid_size"])


def load_rows(folder, condition):
    rows, failed = {}, 0
    for path in sorted((folder/"evaluation"/condition/"rows").glob("*.json")):
        r = read(path)
        if r["status"] != "ok":
            failed += 1
            continue
        rows[_row_key(r["query"])] = r["values"]
    return rows, failed


def training_folder(folder, job):
    """训练目录：<任务>/training/<条件>，续训或只评测的任务沿 training_run / resume_from 找（v7load.training_dir）。"""
    from po_jepa.runtime.addon import v7load
    try:
        return v7load.training_dir(folder, job, job["condition"])
    except ValueError:
        return None


def update_costs(train):
    seconds, peaks = [], []
    for path in sorted((train/"updates").glob("*.json")):
        u = read(path)
        if u.get("update_and_observer_seconds") is not None:
            seconds.append(float(u["update_and_observer_seconds"]))
        if isinstance(u.get("gpu_memory"), dict):
            peaks.append(int(u["gpu_memory"]["peak_allocated"]))
    return dict(seconds=seconds, peaks=peaks)


def checkpoint_snapshot(train, report):
    """选中检查点里的 model_config / loss_config（σ_z、λ 的核对用，13 §4）。"""
    best = report.get("best") or {}
    path = train/best["checkpoint"] if best.get("checkpoint") else None
    if path is None or not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return dict(model_config=payload.get("model_config"), loss_config=payload.get("loss_config"))


def load_run(folder):
    job, report = read(folder/"job.json"), read(folder/"report.json")
    condition = job["condition"]
    rows, failed = load_rows(folder, condition)
    registration = folder/"evaluation"/condition/"registration.json"
    out = dict(folder=str(folder), job=job, report=report, rows=rows, failed_rows=failed,
               queries=read(registration)["queries"] if registration.exists() else None,
               curves=read(folder/"figures"/"curves.json") if (folder/"figures"/"curves.json").exists() else None,
               cache_identity=(report.get("preflight") or {}).get("cache_identity"),
               initial_model_sha256=report.get("initial_model_sha256"), sequence_sha256=None, cost=None, snapshot=None)
    if condition != "identity" and job.get("kind") is None:
        train = training_folder(folder, job)
        if train is not None:
            out["sequence_sha256"] = read(train/"report.json").get("query_sequence_sha256")
            out["cost"] = update_costs(train)
            if condition in PO_CLASS or condition == "transformer_consistency":
                out["snapshot"] = checkpoint_snapshot(train, read(train/"report.json"))
    return out


def load_d4(folder):
    rows = {(r["trajectory_id"], r["start_tick"]): r for r in map(read, sorted((folder/"rows").glob("*.json")))}
    longs = {r["trajectory_id"]: r for r in map(read, sorted((folder/"long").glob("*.json")))}
    return dict(folder=str(folder), job=read(folder/"job.json"), report=read(folder/"report.json"), rows=rows, long=longs,
                source=read(folder/"source.json") if (folder/"source.json").exists() else None)


# ---------- 统计（13 §3） ----------

def window_values(cell, metric):
    """一个 (场景, 种子, 条件) 上某个量的逐窗口值：{窗口键: 值}，窗口键的第一项是轨迹编号；没有数据时返回 None。"""
    if metric in ROW_METRICS:
        run = cell.get("run")
        if run is None:
            return None
        kind, grids, times, name = ROW_METRICS[metric]
        return {k[1:]: v[name] for k, v in run["rows"].items()
                if k[0] == kind and k[4] in grids and k[3] in times and v.get(name) is not None}
    job = cell.get("d4")
    if job is None:
        return None
    if metric in D4_GROUPS:
        return {(tid, start, name): r["values"][name] for (tid, start), r in job["rows"].items() for name in D4_GROUPS[metric]}
    if metric in LONG_GROUPS:
        return {(tid, name): r["values"][name] for tid, r in job["long"].items() for name in LONG_GROUPS[metric]}
    if metric in d4.value_names():
        return {(tid, start): r["values"][metric] for (tid, start), r in job["rows"].items()}
    if metric in d4.long_value_names():
        return {(tid,): r["values"][metric] for tid, r in job["long"].items()}
    raise KeyError(metric)


def by_trajectory(values):
    groups = defaultdict(list)
    for key, value in values.items():
        groups[key[0]].append(value)
    return {tid: sum(v)/len(v) for tid, v in sorted(groups.items())}


def macro(values):
    """窗口 → 轨迹内平均 → 轨迹间平均（13 §3 的前两层）。"""
    if not values:
        return None
    means = list(by_trajectory(values).values())
    return sum(means)/len(means)


def seed_stats(per_seed):
    """{种子: 数} → 种子平均和种子标准差（statistics.stdev，少于 2 个种子时为 None）。"""
    values = [v for _, v in sorted(per_seed.items()) if v is not None]
    return dict(mean=sum(values)/len(values) if values else None, sd=statistics.stdev(values) if len(values) > 1 else None,
                seeds=len(values), per_seed={str(s): v for s, v in sorted(per_seed.items())})


def paired_seed(full, other):
    """同一窗口键配对 full − 对照，在每条轨迹内平均（v7 summarize_v7.paired 的做法）：返回 {轨迹: 差}。"""
    if full is None or other is None:
        return None
    common = {k: full[k]-other[k] for k in full if k in other}
    return by_trajectory(common) if common else None


def paired_stats(diffs):
    values = list(diffs.values())
    return dict(mean=sum(values)/len(values), se=statistics.stdev(values)/math.sqrt(len(values)) if len(values) > 1 else None,
                trajectories=len(values), full_lower=sum(v < 0 for v in values))


def bootstrap_ci(per_seed):
    """按轨迹整群自助：所有种子共有的轨迹编号上有放回地抽 2000 次（torch.Generator 种子 1703，每次比较都从这个种子重新开始，
    所以同一场景里所有比较、所有种子用同一组抽样），每次算种子平均的配对差，取 2.5、97.5 分位数（linear 插值，float64）。"""
    ids = sorted(set.intersection(*[set(d) for d in per_seed]))
    if not ids:
        return None
    table = torch.tensor([[d[t] for t in ids] for d in per_seed], dtype=torch.float64)
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    draws = torch.randint(len(ids), (BOOTSTRAP_REPS, len(ids)), generator=generator)
    boot = table[:, draws].mean(dim=2).mean(dim=0)
    low, high = torch.quantile(boot, torch.tensor([.025, .975], dtype=torch.float64)).tolist()
    return dict(low=low, high=high, reps=BOOTSTRAP_REPS, trajectories=len(ids), seed=BOOTSTRAP_SEED)


def matched(ea, eb):
    """13 §2 的精度匹配：返回 (是否匹配, 误差相差的百分比 |ea − eb|/min·100)。"""
    if ea is None or eb is None:
        return None, None
    low = min(ea, eb)
    return abs(ea-eb) <= MATCH_TOLERANCE*low, abs(ea-eb)/low*100 if low > 0 else None


# ---------- 数据一致性核对（13 §4） ----------

def _same(values):
    distinct = {json.dumps(v, sort_keys=True) for v in values}
    return len(distinct) <= 1


IMPORT_SCHEMA = "addon_v7_calibration_import_v1"     # S17 §4.5


def imported_check(calibration_root):
    """13 §4 最后一项。IMPORTED.json 是 S17 的 tools/import_v7_calibration.py 写的（S17 §4.5）：
    {"schema": IMPORT_SCHEMA, "files": {文件名: {"source": v7 目录里的绝对路径, "sha256": …, "imported_at": …, "v7_commit": …}}}。
    每一项的 sha256 必须等于 source 的当前 SHA，也等于导入的副本 <calibration_root>/<文件名> 的 SHA。"""
    path = Path(calibration_root)/"IMPORTED.json"
    if not path.exists():
        return dict(item="imported_calibration", ok=False, detail=[f"{path} missing"])
    record = read(path)
    if record.get("schema") != IMPORT_SCHEMA or type(record.get("files")) is not dict or not record["files"]:
        return dict(item="imported_calibration", ok=False, detail=[f"{path} is not an {IMPORT_SCHEMA} record"])
    bad = []
    for name, entry in sorted(record["files"].items()):
        source, copy = Path(entry["source"]), Path(calibration_root)/name
        if not source.is_file() or cache.file_sha(source) != entry["sha256"] or not copy.is_file() or cache.file_sha(copy) != entry["sha256"]:
            bad.append(name)
    return dict(item="imported_calibration", ok=not bad, detail=bad)


def calibrations(calibration_root, scene, updates):
    name = f"{scene}-u{updates}.yaml"
    v7, addon = Path(calibration_root)/name, Path(calibration_root)/"addon"/name
    return (yaml.safe_load(v7.read_text(encoding="utf-8")) if v7.exists() else None,
            yaml.safe_load(addon.read_text(encoding="utf-8")) if addon.exists() else None)


def scene_checks(cells, gap, v7_cal, addon_cal, imported):
    """一个场景的 8 项核对：[dict(item, ok, detail)]。只核已有的任务（缺的进 missing.json）。"""
    runs = {k: c["run"] for k, c in cells.items() if c.get("run")}
    d4s = {k: c["d4"] for k, c in cells.items() if c.get("d4")}
    out = []
    identities = [r["cache_identity"] for r in runs.values()]
    problems = [] if _same(identities) and None not in identities else ["preflight cache_identity differs or is missing"]
    by_id = (identities[0] or {}).get("manifest_sha256_by_id", {}) if identities else {}
    for (seed, condition), job in sorted(d4s.items()):
        for (tid, _), r in job["rows"].items():
            if by_id.get(str(tid)) != r["manifest_sha256"]:
                problems.append(f"d4 {condition} seed {seed} trajectory {tid}")
                break
    if gap is not None and any(by_id.get(str(w["trajectory_id"])) != w["manifest_sha256"] for w in gap["windows"]):
        problems.append("spectral gap windows")
    out.append(dict(item="features", ok=not problems, detail=problems))
    bad = [seed for seed in sorted({s for s, _ in runs}) if not _same(
        [r["sequence_sha256"] for (s, c), r in runs.items() if s == seed and c != "identity"])
        or any(r["sequence_sha256"] is None for (s, c), r in runs.items() if s == seed and c != "identity")]
    out.append(dict(item="training_query_sequence", ok=not bad, detail=bad))
    out.append(dict(item="evaluation_queries", ok=_same([r["queries"] for r in runs.values()]) and
                    all(r["queries"] is not None for r in runs.values()), detail=[]))
    sigma = [f"{c} seed {s}" for (s, c), r in sorted(runs.items()) if c in PO_CLASS and (
        v7_cal is None or r["snapshot"] is None or r["snapshot"]["model_config"]["graph"]["sigma_z"] != v7_cal["sigma_z"])]
    out.append(dict(item="sigma_z", ok=not sigma, detail=sigma))
    expected = {}
    if v7_cal is not None:
        expected.update(po_jepa_full=dict(time_weight=v7_cal["time_weight"], space_weight=v7_cal["space_weight"]),
                        po_jepa_time_only=dict(time_weight=v7_cal["time_weight"]),
                        po_jepa_space_only=dict(space_weight=v7_cal["space_weight"]))
    if addon_cal is not None:
        for c in ("transformer_consistency", "fixed_fourier"):
            expected[c] = dict(time_weight=addon_cal[c]["time_weight"], space_weight=addon_cal[c]["space_weight"])
    lam = []
    for (s, c), r in sorted(runs.items()):
        if c in ("po_jepa_full", "po_jepa_time_only", "po_jepa_space_only", "transformer_consistency", "fixed_fourier"):
            want = expected.get(c)
            got = (r["snapshot"] or {}).get("loss_config") or {}
            if want is None or any(got.get(k) != v for k, v in want.items()):
                lam.append(f"{c} seed {s}")
    out.append(dict(item="lambda", ok=not lam, detail=lam))
    init = []
    for (s, c), r in sorted(runs.items()):
        reference = dict(po_jepa_time_only="po_jepa_full", po_jepa_space_only="po_jepa_full", fixed_fourier="po_jepa_full",
                         transformer_consistency="matched_transformer").get(c)
        if reference and (s, reference) in runs and r["initial_model_sha256"] != runs[(s, reference)]["initial_model_sha256"]:
            init.append(f"{c} seed {s}")
    out.append(dict(item="initial_weights", ok=not init, detail=init))
    eq = [f"{c} seed {s}" for (s, c), j in sorted(d4s.items())
          if j["report"]["status"] != "passed" or not j["report"]["equivalence"]["passed"]]
    out.append(dict(item="equivalence", ok=not eq, detail=eq))
    out.append(imported)
    return out


# ---------- 一个场景的表数据（13 §2、§3、§5） ----------

def condition_table(cells, conditions, metrics):
    table = {}
    for c in conditions:
        entry = {}
        for m in metrics:
            if c in AC and m in NOT_COMPARABLE:
                entry[m] = "not_comparable"
                continue
            per_seed = {s: macro(window_values(cell, m) or {}) for (s, cc), cell in cells.items() if cc == c}
            entry[m] = seed_stats(per_seed)
        table[c] = entry
    return table


def precision(cells):
    return {c: seed_stats({s: macro(window_values(cell, "future") or {}) for (s, cc), cell in cells.items() if cc == c})["mean"]
            for c in CONDITIONS}


def comparisons(cells, errors):
    out = []
    seeds = sorted({s for s, _ in cells})
    for ref in REFERENCES:
        is_match, gap_pct = (None, None) if ref == "identity" else matched(errors.get("po_jepa_full"), errors.get(ref))
        for m in PAIRED_METRICS:
            row = dict(reference=ref, metric=m, matched=is_match, error_difference_percent=gap_pct)
            if ref in AC and m in NOT_COMPARABLE:
                out.append(dict(row, status="not_comparable"))
                continue
            per_seed = {}
            for s in seeds:
                f, o = cells.get((s, "po_jepa_full")), cells.get((s, ref))
                diffs = paired_seed(window_values(f, m) if f else None, window_values(o, m) if o else None)
                if diffs:
                    per_seed[s] = diffs
            if not per_seed:
                out.append(dict(row, status="missing"))
                continue
            stats = {str(s): paired_stats(d) for s, d in per_seed.items()}
            means = [v["mean"] for v in stats.values()]
            out.append(dict(row, status="ok", per_seed=stats, mean=sum(means)/len(means),
                            seed_sd=statistics.stdev(means) if len(means) > 1 else None,
                            seeds_full_lower=sum(v < 0 for v in means), ci=bootstrap_ci(list(per_seed.values()))))
    return out


def epsilon_hat(cell):
    """13 §5 T5 的 ε̂：同一批 D4 窗口上等分 tcd_normalized（T ∈ {1, 2, 4, 8}）的中位数（窗口 × 时长一起取）。
    v7 的 tcd_normalized = RMS_in(Φ_T, Φ_{T/2}∘Φ_{T/2}) / (τσ)，τσ = (T/2)² 正是等分 2 段的 x，所以它已经是每单位 x 的
    输入尺度缺陷，不再除；有效斜率 pd_in / (x·ε̂) 在定理的一阶近似下约为 1。"""
    job = cell.get("d4")
    if job is None:
        return None
    values = [r["equivalence"][f"tcd_normalized:T={t:g}"]["new"] for r in job["rows"].values() for t in TRAIN_TIMES]
    return statistics.median(values) if values else None


def partitions(cells):
    out = {}
    for c in CONDITIONS:
        entry = dict(partitions=[], cross_grid={}, unequal={})
        seeds = {s: cell for (s, cc), cell in cells.items() if cc == c}
        eps = {s: epsilon_hat(cell) for s, cell in seeds.items()}
        entry["epsilon_hat"] = seed_stats(eps)
        for t, parts in d4.PARTITIONS:
            label, x = d4.partition_label(parts), d4.partition_x(t, parts)
            row = dict(T=t, partition=label, x=x)
            for m in ("ln", "in"):
                row[f"pd_{m}"] = seed_stats({s: macro(window_values(cell, f"pd_{m}:T={t:g}:{label}") or {}) for s, cell in seeds.items()})
            slopes = {}
            for s, cell in seeds.items():
                pd = macro(window_values(cell, f"pd_in:T={t:g}:{label}") or {})
                slopes[s] = pd/(x*eps[s]) if pd is not None and eps[s] else None
            row["effective_slope"] = seed_stats(slopes)
            entry["partitions"].append(row)
        for name in d4.value_names():
            if name.startswith("xpd_"):
                entry["cross_grid"][name] = seed_stats({s: macro(window_values(cell, name) or {}) for s, cell in seeds.items()})
            if name.startswith("tcd_unequal_"):
                entry["unequal"][name] = seed_stats({s: macro(window_values(cell, name) or {}) for s, cell in seeds.items()})
        out[c] = entry
    return out


def decomposition(cells):
    out = {}
    for c in CONDITIONS:
        seeds = {s: cell for (s, cc), cell in cells.items() if cc == c}
        names = [n for n in d4.value_names() if n.startswith("heldout_")]
        out[c] = {n: seed_stats({s: macro(window_values(cell, n) or {}) for s, cell in seeds.items()}) for n in names}
        out[c]["triangle_violations"] = sum(cell["d4"]["report"]["triangle"]["violations"] for cell in seeds.values() if cell.get("d4"))
    return out


def long_horizon(cells):
    return {c: {n: seed_stats({s: macro(window_values(cell, n) or {}) for (s, cc), cell in cells.items() if cc == c})
                for n in d4.long_value_names()} for c in CONDITIONS}


def transfer_changes(scene, updates, runs, transfers, errors):
    """T8：变体评测相对基础场景的变化（同一条件、同一种子、同编号窗口配对，13 §5；v7 summarize_v7.transfer_drops 的做法：
    全部单点查询的 future_jepa_distance，变体 − 基础，正数 = 换了调节项后变差）。"""
    out = []
    for (variant, train, seed, steps, condition), job in sorted(transfers.items()):
        if train != scene or steps != updates or (seed, condition) not in runs:
            continue
        base = runs[(seed, condition)]
        full = {k: v["future_jepa_distance"] for k, v in job["rows"].items() if k[0] == "point"}
        other = {k: v["future_jepa_distance"] for k, v in base["rows"].items() if k[0] == "point"}
        diffs = paired_seed({k[1:]: v for k, v in full.items()}, {k[1:]: v for k, v in other.items()})
        if diffs:
            is_match, gap_pct = (None, None) if condition in ("identity", "po_jepa_full") else matched(errors.get("po_jepa_full"), errors.get(condition))
            out.append(dict(variant=variant, seed=seed, condition=condition, matched=is_match, error_difference_percent=gap_pct,
                            **paired_stats(diffs)))
    return out


# 任务目录里放逐步记录的子目录：规划报告不在里面，跳过它们省掉几百万个小文件（正式运行每个 D 任务 2 万多个）
BULK_PARTS = frozenset({"evaluation", "training", "rows", "long", "trajectories", "console"})


def planning(*roots):
    """v7 summarize_v7.planning_reports 的读法：结果根下 kind 为 "planning" 的 JSON（plan_v7.py 的输出，V4 和 D3）。"""
    rows = []
    for root in roots:
        for path in sorted(Path(root).rglob("*.json")):
            if BULK_PARTS & set(path.relative_to(root).parts[:-1]) or path.stat().st_size > 50_000_000:
                continue
            try:
                r = read(path)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(r, dict) and r.get("kind") == "planning":
                rows.append(dict(file=str(path), dataset=r["dataset"], train_dataset=r["train_dataset"], condition=r["condition"],
                                 seed=r["seed"], samples=r["cem"]["samples"], episodes=len(r["episodes"]),
                                 success_rate=r["model"]["success_rate"], mean_error=r["model"]["mean_error"]))
    return rows


# ---------- 汇总 ----------

def summarize(v7_root, addon_root, calibration_root):
    found = collect(v7_root, addon_root)
    runs, d4s, transfers, gaps = {}, {}, {}, {}
    for (kind, dataset, train, seed, updates, condition), folder in found.items():
        if kind == "run":
            runs[(dataset, updates, seed, condition)] = load_run(folder)
        elif kind == "d4":
            d4s[(dataset, updates, seed, condition)] = load_d4(folder)
        elif kind == "transfer":
            job = load_run(folder)
            transfers[(dataset, train, seed, updates, condition)] = job
        elif kind == "spectral_gap":
            gaps[(dataset, updates)] = read(folder/"report.json")
    scenes = sorted({(d, u) for d, u, _, _ in runs} | {(d, u) for d, u, _, _ in d4s} | set(gaps))
    seeds = sorted({s for _, _, s, _ in runs} | {s for _, _, s, _ in d4s})
    imported = imported_check(calibration_root)
    missing = []
    for dataset, updates in scenes:
        for seed in seeds:
            for c in CONDITIONS:
                for kind, pool in (("run", runs), ("d4", d4s)):
                    if (dataset, updates, seed, c) not in pool:
                        missing.append(dict(kind=kind, dataset=dataset, updates=updates, seed=seed, condition=c))
        if (dataset, updates) not in gaps:
            missing.append(dict(kind="spectral_gap", dataset=dataset, updates=updates))
    encoder_path = Path(addon_root)/"encoder-identity.json"
    encoder = read(encoder_path) if encoder_path.exists() else None
    if encoder is None:
        missing.append(dict(kind="encoder_identity", file=str(encoder_path)))
    plans = planning(v7_root, addon_root)
    out = dict(scenes={}, checks={}, missing=missing, encoder_identity=None if encoder is None else
               dict(verdict=encoder["verdict"], encoder_sha256=encoder["encoder"]["sha256"], ac_sha256=encoder["ac_checkpoint"]["sha256"]),
               planning=plans, jobs=len(found), scientific_conclusions_verified=False,
               rules=dict(match_tolerance=MATCH_TOLERANCE, bootstrap_reps=BOOTSTRAP_REPS, bootstrap_seed=BOOTSTRAP_SEED,
                          train_times=list(TRAIN_TIMES), heldout_times=list(HELDOUT_TIMES)))
    for dataset, updates in scenes:
        scene = f"{dataset}-u{updates}"
        cells = defaultdict(dict)
        for (d, u, s, c), job in runs.items():
            if (d, u) == (dataset, updates):
                cells[(s, c)]["run"] = job
        for (d, u, s, c), job in d4s.items():
            if (d, u) == (dataset, updates):
                cells[(s, c)]["d4"] = job
        cells = dict(cells)
        v7_cal, addon_cal = calibrations(calibration_root, dataset, updates)
        checks = scene_checks(cells, gaps.get((dataset, updates)), v7_cal, addon_cal, imported)
        out["checks"][scene] = checks
        if not all(c["ok"] for c in checks):
            out["scenes"][scene] = dict(status="blocked", failed=[c["item"] for c in checks if not c["ok"]])
            continue
        errors = precision(cells)
        gap = gaps.get((dataset, updates))
        eq = [cell["d4"]["report"]["equivalence"]["metrics"] for cell in cells.values() if cell.get("d4")]
        out["scenes"][scene] = dict(
            status="ok", dataset=dataset, updates=updates, precision=errors,
            in_distribution=condition_table(cells, CONDITIONS, T1_METRICS), heldout=condition_table(cells, CONDITIONS, T2_METRICS),
            comparisons=comparisons(cells, errors), partitions=partitions(cells), decomposition=decomposition(cells),
            long_horizon=long_horizon(cells),
            transfer=transfer_changes(dataset, updates, {(s, c): cell["run"] for (s, c), cell in cells.items() if cell.get("run")}, transfers,
                                      errors),
            spectral_gap=None if gap is None else dict(summary=gap["summary"], fixed_fourier_cutoff_gap=gap["fixed_fourier"]["cutoff_gap"]),
            equivalence_max_abs_diff=max((m["max_abs_diff"] for e in eq for m in e.values()), default=None),
            cost={c: _pooled_cost([cell["run"]["cost"] for (s, cc), cell in cells.items() if cc == c and cell.get("run") and cell["run"]["cost"]])
                  for c in TRAINED},
            scatter={c: {str(s): dict(error=macro(window_values(cell, "future") or {}),
                                      tcd_ln=macro(window_values(cell, "tcd_ln") or {}),
                                      scd_ln=macro(window_values(cell, "scd_ln") or {}))
                         for (s, cc), cell in sorted(cells.items()) if cc == c} for c in CONDITIONS})
    keep = [s["spectral_gap"]["summary"]["keep_sentence"] for s in out["scenes"].values() if s.get("spectral_gap")]
    out["spectral_gap_decision"] = None if not keep else spectral_decision(keep)
    return out, curves_by_scene(runs)


def spectral_decision(keep):
    """10 §5：每个场景、每个网格上比值的中位数都大于 1 才保留原句。"""
    from po_jepa.runtime.addon import spectral_gap
    return spectral_gap.KEEP_SENTENCE if all(keep) else spectral_gap.REWRITE


def _pooled_cost(costs):
    """13 §5 T11：一个条件在本场景全部种子的全部更新记录上，update_and_observer_seconds 的中位数和 peak_allocated 的最大值。"""
    seconds = [x for c in costs for x in c["seconds"]]
    peaks = [x for c in costs for x in c["peaks"]]
    return dict(jobs=len(costs), updates=len(seconds), median_seconds=statistics.median(seconds) if seconds else None,
                peak_allocated_bytes=max(peaks) if peaks else None)


def curves_by_scene(runs):
    out = defaultdict(dict)
    for (d, u, s, c), job in runs.items():
        if job["curves"]:
            out[f"{d}-u{u}"][(c, s)] = job["curves"]
    return out


# ---------- 图（13 §6）和表（13 §5） ----------

def figures(summary, curves):
    files = {}
    for scene, s in summary["scenes"].items():
        if s["status"] != "ok":
            continue
        for defect in ("tcd_ln", "scd_ln"):
            series = []
            for c in CONDITIONS:
                points = sorted((p["error"], 0. if c == "identity" else p[defect]) for p in s["scatter"][c].values()
                                if p["error"] is not None and (c == "identity" or p[defect] is not None))
                if points:
                    series.append((c if c != "identity" else "identity（纵轴 0）", [x for x, _ in points], [y for _, y in points]))
            files[f"{scene}-scatter-{defect}.svg"] = plots.line_chart(
                series, f"{scene}：同分布误差—{defect}（每个种子一个点）", "同分布未来误差 e_c", defect)
        series = []
        for c in CONDITIONS:
            rows = s["partitions"][c]["partitions"]
            points = sorted((r["x"], r["pd_in"]["mean"]) for r in rows if r["pd_in"]["mean"] is not None)
            if points:
                series.append((c, [x for x, _ in points], [y for _, y in points]))
            eps = s["partitions"][c]["epsilon_hat"]["mean"]
            if eps is not None and c == "po_jepa_full":
                series.append(("ε̂·x（po_jepa_full）", [0., 28.], [0., 28.*eps]))
        files[f"{scene}-pd-vs-x.svg"] = plots.line_chart(series, f"{scene}：pd_in 对 x(Π)", "x(Π)", "pd_in")
        validation = []
        for (c, seed), cv in sorted(curves.get(scene, {}).items()):
            validation.append((f"{c} {seed}", [r["step"] for r in cv["validation"]], [r["future_jepa_distance"] for r in cv["validation"]]))
        files[f"{scene}-validation.svg"] = plots.line_chart(validation, f"{scene}：验证集未来误差", "更新步数", "future JEPA distance")
        loss = []
        for (c, seed), cv in sorted(curves.get(scene, {}).items()):
            if "jepa" in cv["losses"]:
                window = max(1, min(100, len(cv["steps"])//20 or 1))
                loss.append((f"{c} {seed}", cv["steps"], plots._smooth(cv["losses"]["jepa"], window)))
        files[f"{scene}-train-jepa.svg"] = plots.line_chart(loss, f"{scene}：训练 JEPA 损失（滑动平均）", "更新步数", "JEPA loss", log_y=True)
    return files


def _f(x, digits=4):
    if x is None:
        return "缺"
    if x == "not_comparable":
        return "不可比"
    return f"{x:.{digits}f}" if x == 0 or 1e-3 <= abs(x) < 1e4 else f"{x:.3e}"


def _ms(stats):
    if stats == "not_comparable":
        return "不可比"
    if stats is None or stats["mean"] is None:
        return "缺"
    return _f(stats["mean"]) + ("" if stats["sd"] is None else f" ± {_f(stats['sd'])}")


def _match_label(row):
    if row.get("reference", row.get("condition")) in ("identity", "po_jepa_full"):
        return "不参与"
    return "缺" if row["matched"] is None else "匹配" if row["matched"] else "不匹配"


def markdown(summary):
    lines = ["# 补跑汇总", "", f"任务数：{summary['jobs']}。未经人工核对，不构成科学结论。", ""]
    lines += ["## 数据一致性核对（13 §4）", ""]
    for scene, checks in summary["checks"].items():
        failed = [c["item"] for c in checks if not c["ok"]]
        lines.append(f"- {scene}：{'全部通过' if not failed else '未通过 ' + '、'.join(failed) + '（本场景不出表）'}")
    lines.append("")
    for scene, s in summary["scenes"].items():
        if s["status"] != "ok":
            continue
        lines += [f"## {scene}", "", "### T1 同分布（种子平均 ± 种子标准差）", "",
                  "| 条件 | " + " | ".join(T1_METRICS) + " |", "|---" * (len(T1_METRICS)+1) + "|"]
        lines += [f"| {c} | " + " | ".join(_ms(s["in_distribution"][c][m]) for m in T1_METRICS) + " |" for c in CONDITIONS]
        lines += ["", "### T2 留出", "", "| 条件 | 留出时长（3、12） | 留出网格（12×12） | 两者同时 |", "|---|---|---|---|"]
        lines += [f"| {c} | " + " | ".join(_ms(s["heldout"][c][m]) for m in T2_METRICS) + " |" for c in CONDITIONS]
        lines += ["", "### T3 配对（po_jepa_full − 对照；负数 = full 更低）", "",
                  "| 对照 | 量 | 精度匹配 | 误差相差 % | 种子平均差 | 种子 SD | 95% CI | full 更低的种子 |", "|---|---|---|---|---|---|---|---|"]
        for r in s["comparisons"]:
            match = _match_label(r)
            if r["status"] != "ok":
                lines.append(f"| {r['reference']} | {r['metric']} | {match} | {_f(r['error_difference_percent'], 1)} | "
                             f"{'不可比' if r['status'] == 'not_comparable' else '缺'} | | | |")
                continue
            ci = "缺" if r["ci"] is None else f"[{_f(r['ci']['low'])}, {_f(r['ci']['high'])}]"
            lines.append(f"| {r['reference']} | {r['metric']} | {match} | {_f(r['error_difference_percent'], 1)} | {_f(r['mean'])} | "
                         f"{_f(r['seed_sd'])} | {ci} | {r['seeds_full_lower']}/{len(r['per_seed'])} |")
        lines += ["", "### T4 消融", "", "| 条件 | " + " | ".join(T1_METRICS+T2_METRICS) + " |", "|---" * (len(T1_METRICS+T2_METRICS)+1) + "|"]
        lines += [f"| {c} | " + " | ".join([_ms(s["in_distribution"][c][m]) for m in T1_METRICS] + [_ms(s["heldout"][c][m]) for m in T2_METRICS]) + " |"
                  for c in ABLATION]
        lines += ["", "### T5 分割差异（pd_in；有效斜率 = pd_in / (x·ε̂)）", "", "| 条件 | ε̂ | " +
                  " | ".join(f"T={t:g} {d4.partition_label(p)} (x={d4.partition_x(t, p):g})" for t, p in d4.PARTITIONS) + " |",
                  "|---" * (len(d4.PARTITIONS)+2) + "|"]
        for c in CONDITIONS:
            p = s["partitions"][c]
            lines.append(f"| {c} | {_ms(p['epsilon_hat'])} | " + " | ".join(
                f"{_ms(r['pd_in'])}（{_ms(r['effective_slope'])}）" for r in p["partitions"]) + " |")
        names = [n for n in d4.value_names() if n.startswith(("xpd_", "tcd_unequal_"))]
        lines += ["", "| 条件 | " + " | ".join(names) + " |", "|---" * (len(names)+1) + "|"]
        lines += [f"| {c} | " + " | ".join(_ms({**s['partitions'][c]['cross_grid'], **s['partitions'][c]['unequal']}[n]) for n in names) + " |"
                  for c in CONDITIONS]
        names = [n for n in d4.value_names() if n.startswith("heldout_")]
        lines += ["", "### T6 误差分解", "", "| 条件 | " + " | ".join(names) + " | 三角不等式违反 |", "|---" * (len(names)+2) + "|"]
        lines += [f"| {c} | " + " | ".join(_ms(s["decomposition"][c][n]) for n in names) + f" | {s['decomposition'][c]['triangle_violations']} |"
                  for c in CONDITIONS]
        names = d4.long_value_names()
        lines += ["", "### T7 长时程", "", "| 条件 | " + " | ".join(names) + " |", "|---" * (len(names)+1) + "|"]
        lines += [f"| {c} | " + " | ".join(_ms(s["long_horizon"][c][n]) for n in names) + " |" for c in CONDITIONS]
        lines += ["", "### T8 变体（变体 − 基础场景，同编号窗口配对；正数 = 变差；精度匹配对 po_jepa_full）", "",
                  "| 变体 | 种子 | 条件 | 精度匹配 | 误差相差 % | 变化 | SE | 轨迹 |", "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['variant']} | {r['seed']} | {r['condition']} | {_match_label(r)} | {_f(r['error_difference_percent'], 1)} | "
                  f"{_f(r['mean'])} | {_f(r['se'])} | {r['trajectories']} |" for r in s["transfer"]]
        lines += ["", "### T10 截断间隙", ""]
        if s["spectral_gap"] is None:
            lines.append("缺")
        else:
            g = s["spectral_gap"]["summary"]
            lines += ["| 网格 | 窗口 | ratio 分位数（5/25/50/75/95） | 中位数 > 1 |", "|---|---|---|---|"]
            for size, e in g["grids"].items():
                q = "缺" if e["ratio"] is None else ", ".join(_f(v) for v in e["ratio"])
                lines.append(f"| {size}×{size} | {e['windows']} | {q} | {'是' if e['median_ratio_above_1'] else '否'} |")
            lines += ["| fixed_fourier | — | 截断间隙恒为 0（08 §4.3） | — |", "", f"判断：{g['decision']}"]
        lines += ["", "### T11 核对与成本", "", f"- 等价核对最大绝对差：{_f(s['equivalence_max_abs_diff'])}", "",
                  "| 条件 | 任务 | 每次更新耗时中位数（秒） | 显存峰值（字节） |", "|---|---|---|---|"]
        lines += [f"| {c} | {s['cost'][c]['jobs']} | {_f(s['cost'][c]['median_seconds'])} | {s['cost'][c]['peak_allocated_bytes'] or '缺'} |" for c in TRAINED]
        lines.append("")
    lines += ["## T9 规划（成功率）", "", "| 场景 | 训练数据集 | 条件 | 种子 | 样本数 | 回合 | 成功率 |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['dataset']} | {r['train_dataset']} | {r['condition']} | {r['seed']} | {r['samples']} | {r['episodes']} | {_f(r['success_rate'], 2)} |"
              for r in summary["planning"]]
    e = summary["encoder_identity"]
    lines += ["", f"编码器核对（10 §6）：{'缺' if e is None else e['verdict']}", "",
              f"截断间隙总判断（每个场景、每个网格的比值中位数都大于 1 才保留原句）：{summary['spectral_gap_decision'] or '缺'}", ""]
    if summary["missing"]:
        lines += [f"缺的任务 {len(summary['missing'])} 个，见 missing.json。", ""]
    return "\n".join(lines)


def write_output(output, summary, svgs):
    """先写临时目录再整体换名；已有的输出改名为 <输出>.previousN。"""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    cache._write_json(temporary/"summary.json", summary)
    cache._write_json(temporary/"checks.json", dict(checks=summary["checks"]))
    cache._write_json(temporary/"missing.json", dict(missing=summary["missing"]))
    (temporary/"summary.md").write_text(markdown(summary), encoding="utf-8")
    (temporary/"figures").mkdir()
    for name, svg in sorted(svgs.items()):
        (temporary/"figures"/name).write_text(svg, encoding="utf-8")
    if output.exists():
        n = 1
        while output.with_name(f"{output.name}.previous{n}").exists():
            n += 1
        output.rename(output.with_name(f"{output.name}.previous{n}"))
    temporary.rename(output)
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v7-results", type=Path, required=True)
    p.add_argument("--addon-results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibration-root", type=Path, default=ROOT/"configs"/"calibration", help="导入的 v7 标定、补跑标定和 IMPORTED.json 所在目录")
    a = p.parse_args(argv)
    for name in ("v7_results", "addon_results"):
        if not getattr(a, name).is_dir():
            p.error(f"--{name.replace('_', '-')} 不是目录：{getattr(a, name)}")
    summary, curves = summarize(a.v7_results, a.addon_results, a.calibration_root)
    output = write_output(a.output, summary, figures(summary, curves))
    blocked = [s for s, v in summary["scenes"].items() if v["status"] != "ok"]
    print(f"{summary['jobs']} 个任务；{len(summary['scenes'])} 个场景，核对未通过 {len(blocked)} 个；缺 {len(summary['missing'])} 个 → {output}")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
