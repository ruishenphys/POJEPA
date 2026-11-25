"""v7 汇总：把各自独立提交的任务（系统 × 种子 × 条件）合成一张总表、配对比较和收敛图。

用法：
  python tools/summarize_v7.py --results ../results-v7 --output ../results-v7/summary
--results 下任意层级里每个含 job.json + COMPLETE.json 的目录都算一个完成的任务（run.py 的输出是 <dir>/run/job/）。
只读，不需要 GPU。缺的任务会列在 SUMMARY.md 的“缺失任务”里，不会中止。

核对项（不通过就在报告里标红，不静默合并）：
- 同一 (数据集, 种子) 下所有条件的预登记（源码、配置、查询序列）SHA 必须一致；
- po_jepa_full 与 po_jepa_no_consistency 的初始权重 SHA 必须一致（唯一差别是一致性项）。
- S26a：补跑条件的初始权重按 08 §1.3 核（time_only、space_only、fixed_fourier、rollout_sup 等于 po_jepa_full，transformer_consistency 等于 matched_transformer）。

v7.1 起另外报：按网格（16/12/8）拆开的未来误差和配对差（主表的“未来误差”仍是全部单点的平均，与以前同口径）、
16×16 上每步 1 周期的多步滚动误差（与同一批窗口的直接预测并列）、各条件实际训练步数和是否早停。
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from po_jepa.runtime import plots  # noqa: E402

V7_CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity", "vjepa2_ac_small", "vjepa2_ac")
# S26a：v8 任务表 --stage addon 的补跑条件（N1 基线、N2 消融），和 po_jepa.runtime.addon.conditions.ADDON_CONDITIONS 相同（测试核对）。
# 它们和 v7 条件进同一张表。缺失怎么算（S27a 审核后）：--addon-conditions 写了时，每个 (数据集, 步数, 种子) 都要有 v7 条件加这些
# 补跑条件（任务生成器 make_v8_jobs.py --stage addon 的 AE 行会传本次排的补跑条件）；没写时，只在某组里出现过补跑条件才核，
# 要求的是结果里任何一组出现过的补跑条件（不再要求 6 个全有：只排一部分补跑条件时其余的不算缺失）。
ADDON_CONDITIONS = ("po_jepa_time_only", "po_jepa_space_only", "transformer_consistency", "fixed_fourier", "koopman",
                    "po_jepa_rollout_sup")   # 08 §6：有真值两段滚动的对照（2026-11-12 加）
# 2026-11-25：DINO-WM、JEPA-WM 参数匹配基线（和 po_jepa.ac_baseline.WM_CONDITIONS 相同，测试核对）。v8 任务表默认排，
# 但 v7、小规模验证的结果里没有：只在结果里出现过时才按组核缺失（同补跑条件没写 --addon-conditions 时的规则）。
WM_CONDITIONS = ("dino_wm_small", "jepa_wm_small")
CONDITIONS = (*V7_CONDITIONS, *WM_CONDITIONS, *ADDON_CONDITIONS)
# S28b（C3，Jerry 2026-11-17）：identity 退出对比，不再当配对比较的对照；它照算，只在单独一节报（数值自检、归一化参照）
REFERENCES = tuple(c for c in CONDITIONS[1:] if c != "identity")
SELF_CHECK = ("identity",)
# 08 §1.3：同一数据集、同一种子下补跑条件的初始权重必须和这个 v7 条件相同（koopman 没有对照）
INITIAL_SAME_AS = dict(po_jepa_time_only="po_jepa_full", po_jepa_space_only="po_jepa_full", fixed_fourier="po_jepa_full",
                       transformer_consistency="matched_transformer", po_jepa_rollout_sup="po_jepa_full")
POINT_METRICS = ("future_jepa_distance", "tcd_normalized", "delta_act")
PAIR_METRICS = ("scd_normalized",)


def find_jobs(root):
    for path in sorted(Path(root).rglob("job.json")):
        folder = path.parent
        if (folder/"COMPLETE.json").exists() and (folder/"report.json").exists():
            yield folder


def _key(q):
    return (q["kind"], q["trajectory_id"], q["start_tick"], float(q["duration_cycles"]), q["grid_size"], q.get("stratum"))


def load_job(folder):
    job = json.loads((folder/"job.json").read_text())
    report = json.loads((folder/"report.json").read_text())
    rows = {}
    for path in sorted((folder/"evaluation"/job["condition"]/"rows").glob("*.json")):
        r = json.loads(path.read_text())
        if r["status"] == "ok":
            rows[_key(r["query"])] = r
    curves = folder/"figures"/"curves.json"
    return dict(folder=str(folder), job=job, report=report, rows=rows,
                curves=json.loads(curves.read_text()) if curves.exists() else None)


def _mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs)/len(xs) if xs else None


def condition_means(job):
    out = {}
    points = [r for k, r in job["rows"].items() if k[0] == "point"]
    pairs = [r for k, r in job["rows"].items() if k[0] == "pair"]
    for m in POINT_METRICS:
        out[m] = _mean([r["values"].get(m) for r in points])
    for m in PAIR_METRICS:
        out[m] = _mean([r["values"].get(m) for r in pairs])
    strata = defaultdict(list)
    for r in points:
        strata[r["query"].get("stratum")].append(r["values"]["future_jepa_distance"])
    out["future_by_stratum"] = {str(s): _mean(v) for s, v in sorted(strata.items(), key=lambda kv: str(kv[0]))}
    by_tau = defaultdict(list)
    for r in points:
        by_tau[float(r["query"]["duration_cycles"])].append(r["values"].get("delta_act"))
    out["delta_act_by_duration"] = {str(t): _mean(v) for t, v in sorted(by_tau.items())}
    out["windows"] = dict(point=len(points), pair=len(pairs))
    by_grid = defaultdict(list)
    for r in points:
        by_grid[int(r["query"]["grid_size"])].append(r["values"]["future_jepa_distance"])
    out["future_by_grid"] = {str(g): _mean(v) for g, v in sorted(by_grid.items(), reverse=True)}
    rolled = [r for r in points if "rollout_future_jepa_distance" in r["values"]]
    out["rollout"] = dict(windows=len(rolled), rollout_future=_mean([r["values"]["rollout_future_jepa_distance"] for r in rolled]),
                          direct_future=_mean([r["values"]["future_jepa_distance"] for r in rolled])) if rolled else None
    return out


def paired(full, other, metric="future_jepa_distance", grid=None):
    """逐窗口配对差（full − 对照），先在每条轨迹内平均，再在轨迹间统计。grid：只取这个网格的单点。"""
    per_traj = defaultdict(list)
    for k, r in full["rows"].items():
        if k[0] != "point" or k not in other["rows"] or (grid is not None and k[4] != grid):
            continue
        a, b = r["values"].get(metric), other["rows"][k]["values"].get(metric)
        if a is not None and b is not None:
            per_traj[k[1]].append(a-b)
    diffs = [sum(v)/len(v) for v in per_traj.values()]
    if not diffs:
        return None
    se = statistics.stdev(diffs)/math.sqrt(len(diffs)) if len(diffs) > 1 else None
    return dict(mean=sum(diffs)/len(diffs), se=se, trajectories=len(diffs), full_better=sum(d < 0 for d in diffs))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--addon-conditions", default=None,
                   help="逗号分隔，每组都必须有的补跑条件（不重复）；不写时按结果里出现过的补跑条件核（见文件头）")
    a = p.parse_args()
    expected_addon = None
    if a.addon_conditions is not None:
        expected_addon = [c for c in a.addon_conditions.split(",") if c]
        if not expected_addon or not set(expected_addon) <= set(ADDON_CONDITIONS) or len(set(expected_addon)) != len(expected_addon):
            raise SystemExit(f"--addon-conditions 只能是不重复的补跑条件：{ADDON_CONDITIONS}")
    jobs = [load_job(f) for f in find_jobs(a.results)]
    if not jobs:
        raise SystemExit(f"{a.results} 下没有完成的 v7 任务")
    equal_jobs = [j for j in jobs if j["job"].get("kind") == "equal_step"]
    jobs = [j for j in jobs if j["job"].get("kind") != "equal_step"]
    groups = defaultdict(dict)        # (测试数据集, 步数, 种子) -> condition -> job；分布外评测的测试数据集带调节项
    duplicates = []
    for j in jobs:
        key = (_label(j["job"]), j["job"]["updates"], j["job"]["seed"])
        c = j["job"]["condition"]
        if c in groups[key]:
            duplicates.append(dict(key=list(key), condition=c, kept=groups[key][c]["folder"], ignored=j["folder"]))
            continue
        groups[key][c] = j

    problems, table, comparisons = [], [], []
    for (dataset, updates, seed), by_c in sorted(groups.items()):
        regs = {c: j["report"]["registration_sha256"] for c, j in by_c.items()}
        if len(set(regs.values())) > 1:
            problems.append(dict(dataset=dataset, updates=updates, seed=seed, problem="预登记 SHA 不一致", detail=regs))
        f, n = by_c.get("po_jepa_full"), by_c.get("po_jepa_no_consistency")
        if f and n and f["report"]["initial_model_sha256"] != n["report"]["initial_model_sha256"]:
            problems.append(dict(dataset=dataset, updates=updates, seed=seed, problem="full 与 no-consistency 初始权重不同"))
        for c, ref in INITIAL_SAME_AS.items():
            if c in by_c and ref in by_c and by_c[c]["report"]["initial_model_sha256"] != by_c[ref]["report"]["initial_model_sha256"]:
                problems.append(dict(dataset=dataset, updates=updates, seed=seed, problem=f"{c} 与 {ref} 初始权重不同（08 §1.3）"))
        for c, j in by_c.items():
            table.append(dict(dataset=dataset, updates=updates, seed=seed, condition=c, **condition_means(j),
                              best_step=(j["report"].get("best") or {}).get("step"),
                              completed_updates=j["report"]["training"].get("completed_updates"),
                              stopped_early=(j["report"]["training"].get("early_stopping") or {}).get("stopped_early"),
                              consistency_monitor=j["report"]["training"].get("consistency_monitor")))
        if f:
            for ref in REFERENCES:
                if ref in by_c:
                    comparisons.append(dict(dataset=dataset, updates=updates, seed=seed, reference=ref,
                                            future=paired(f, by_c[ref]), delta_act=paired(f, by_c[ref], "delta_act"),
                                            future_by_grid={str(g): paired(f, by_c[ref], grid=g) for g in (16, 12, 8)},
                                            rollout=paired(f, by_c[ref], "rollout_future_jepa_distance")))
    missing = missing_jobs(groups, expected_addon)

    # 跨种子
    across = []
    for (dataset, updates) in sorted({(d, u) for d, u, _ in groups}):
        for c in CONDITIONS:
            rows = [r for r in table if r["dataset"] == dataset and r["updates"] == updates and r["condition"] == c]
            if not rows:
                continue
            vals = [r["future_jepa_distance"] for r in rows if r["future_jepa_distance"] is not None]
            across.append(dict(dataset=dataset, updates=updates, condition=c, seeds=len(rows),
                               future_mean=_mean(vals), future_seed_std=statistics.stdev(vals) if len(vals) > 1 else None,
                               delta_act_mean=_mean([r["delta_act"] for r in rows]),
                               tcd_normalized_mean=_mean([r["tcd_normalized"] for r in rows]),
                               scd_normalized_mean=_mean([r["scd_normalized"] for r in rows]),
                               future_by_grid={g: _mean([r["future_by_grid"].get(g) for r in rows]) for g in ("16", "12", "8")},
                               rollout_future_mean=_mean([(r["rollout"] or {}).get("rollout_future") for r in rows]),
                               rollout_direct_mean=_mean([(r["rollout"] or {}).get("direct_future") for r in rows]),
                               completed_updates=[r["completed_updates"] for r in rows],
                               stopped_early=sum(bool(r["stopped_early"]) for r in rows)))
        for ref in REFERENCES:
            rows = [x for x in comparisons if x["dataset"] == dataset and x["updates"] == updates and x["reference"] == ref and x["future"]]
            if rows:
                ms = [x["future"]["mean"] for x in rows]
                across.append(dict(dataset=dataset, updates=updates, comparison=f"po_jepa_full − {ref}", seeds=len(rows),
                                   mean=_mean(ms), seed_std=statistics.stdev(ms) if len(ms) > 1 else None,
                                   seeds_full_better=sum(m < 0 for m in ms),
                                   trajectories_full_better=[x["future"]["full_better"] for x in rows],
                                   trajectories=[x["future"]["trajectories"] for x in rows],
                                   by_grid={g: _mean([(x["future_by_grid"].get(g) or {}).get("mean") for x in rows]) for g in ("16", "12", "8")},
                                   rollout=_mean([(x["rollout"] or {}).get("mean") for x in rows])))

    a.output.mkdir(parents=True, exist_ok=True)
    figures = a.output/"figures"; figures.mkdir(exist_ok=True)
    for (dataset, updates) in sorted({(d, u) for d, u, _ in groups}):
        series_v, series_l = [], []
        for c in CONDITIONS:
            curves = [g[c]["curves"] for (d, u, _), g in groups.items() if d == dataset and u == updates and c in g and g[c]["curves"]]
            if not curves:
                continue
            # 早停后各种子的曲线长短不一：取并集，每个步数只平均到达这一步的种子
            steps = sorted(set.union(*[{r["step"] for r in cv["validation"]} for cv in curves]))
            series_v.append((f"{c} ({len(curves)} 种子)", steps,
                             [_mean([r["future_jepa_distance"] for cv in curves for r in cv["validation"] if r["step"] == s]) for s in steps]))
            first = curves[0]
            if "jepa" in first["losses"]:
                window = max(1, min(100, len(first["steps"])//20 or 1))
                series_l.append((c, first["steps"], plots._smooth(first["losses"]["jepa"], window)))
        (figures/f"{dataset}-u{updates}-validation.svg").write_text(plots.line_chart(
            series_v, f"{dataset}（{updates} 步）验证集未来 JEPA 距离，种子平均", "更新步数", "future JEPA distance"))
        (figures/f"{dataset}-u{updates}-train-jepa.svg").write_text(plots.line_chart(
            series_l, f"{dataset}（{updates} 步）训练 JEPA 损失（第一个种子，滑动平均）", "更新步数", "JEPA loss", log_y=True))

    equal = equal_step_rows(groups, equal_jobs, problems)
    summary = dict(jobs=len(jobs), table=table, comparisons=comparisons, across_seeds=across, equal_step=equal,
                   transfer=transfer_drops(groups), planning=planning_reports(a.results), long=long_reports(a.results),
                   timing=timing_reports(a.results), space=space_reports(a.results),
                   problems=problems, missing=missing, duplicates=duplicates, scientific_conclusions_verified=False)
    (a.output/"summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (a.output/"SUMMARY.md").write_text(_markdown(summary), encoding="utf-8")
    print(f"{len(jobs)} 个任务；问题 {len(problems)} 个；缺失 {len(missing)} 个 → {a.output/'SUMMARY.md'}")


def missing_jobs(groups, expected_addon=None):
    """每组（数据集, 步数, 种子）缺的条件。expected_addon 给了：每组都要 v7 条件 + 这些补跑条件；
    没给：结果里出现过的补跑条件，只在本组出现过补跑条件时要求（补跑是可选的）。"""
    seen = {c for by_c in groups.values() for c in by_c if c in ADDON_CONDITIONS}
    wm = [c for c in WM_CONDITIONS if any(c in by_c for by_c in groups.values())]   # 出现过就每组都要
    out = []
    for (d, u, s), by_c in sorted(groups.items()):
        if expected_addon is not None:
            addon = [c for c in ADDON_CONDITIONS if c in expected_addon]
        else:
            addon = [c for c in ADDON_CONDITIONS if c in seen] if set(by_c) & set(ADDON_CONDITIONS) else []
        out += [dict(dataset=d, updates=u, seed=s, condition=c) for c in (*V7_CONDITIONS, *addon, *wm) if c not in by_c]
    return out


def equal_step_rows(groups, equal_jobs, problems):
    """同步数对照（补充）：po_jepa_full − 截到 full 步数的对照，按网格配对；与主表的 full − 对照并列。"""
    rows = []
    for j in equal_jobs:
        key = (_label(j["job"]), j["job"]["updates"], j["job"]["seed"])   # S28b：变体后缀和主表分组同一写法
        full = groups.get(key, {}).get("po_jepa_full")
        c = j["job"]["condition"]
        if full is None:
            problems.append(dict(dataset=key[0], updates=key[1], seed=key[2], problem=f"同步数对照 {c} 找不到 po_jepa_full"))
            continue
        if j["report"]["registration_sha256"] != full["report"]["registration_sha256"]:
            problems.append(dict(dataset=key[0], updates=key[1], seed=key[2], problem=f"同步数对照 {c} 预登记 SHA 不一致"))
            continue
        main = groups[key].get(c)
        rows.append(dict(dataset=key[0], updates=key[1], seed=key[2], condition=c, **j["report"]["equal_step"],
                         equal_step={str(g): paired(full, j, grid=g) for g in (16, 12, 8)},
                         equal_step_rollout=paired(full, j, "rollout_future_jepa_distance"),
                         main={str(g): paired(full, main, grid=g) for g in (16, 12, 8)} if main else None))
    return rows


def _variant_suffix(job):
    """S28b：非默认变体（job.json 的 variant：多段切分、零初始化）加在数据集名后，和标定文件名同一写法；默认变体为空。"""
    variant = job.get("variant") or {}
    suffix = (("~dyadic" if variant.get("temporal_partition") == "dyadic" else "")
              + ("~zeroinit" if variant.get("readout_init") == "zero" else ""))
    # C3：空间部分（K、补空间），和 runtime.checks.variant_suffix 同一写法
    modes, complement = variant.get("modes", "16"), variant.get("complement", "none")
    if modes != "16" or complement != "none":
        suffix += f"~K{modes}"+("c" if complement == "shared" else "")
    # G14：一致性权重缩放，和 runtime.checks.weight_scale_suffix 同一写法
    suffix += "".join(f"~{tag}{variant[key]}" for key, tag in (("time_weight_scale", "lt"), ("space_weight_scale", "ls"))
                      if variant.get(key, "1") != "1")
    return suffix


def _label(job):
    """普通任务：数据集；分布外评测：训练数据集 → 测试数据集。S28b：非默认变体两边都带变体后缀，不和默认变体混成一组。"""
    suffix = _variant_suffix(job)
    return job["dataset"]+suffix if job.get("kind") != "transfer" else f"{job['train_dataset']}{suffix} → {job['dataset']}{suffix}"


def transfer_drops(groups):
    """分布外评测相对基础测试集的变化（同一条件、同一种子、同编号窗口逐条配对）：正数 = 换了调节项后变差。"""
    out = []
    for (label, updates, seed), by_c in sorted(groups.items()):
        if " → " not in label:
            continue
        base_label = label.split(" → ")[0]
        base = groups.get((base_label, updates, seed), {})
        for c, j in by_c.items():
            if c in base:
                d = paired(j, base[c])
                if d:
                    out.append(dict(transfer=label, updates=updates, seed=seed, condition=c, change=d["mean"], se=d["se"],
                                    trajectories=d["trajectories"]))
    return out


def planning_reports(root):
    rows = []
    for path in sorted(Path(root).rglob("*.json")):
        if path.stat().st_size > 50_000_000:
            continue
        try:
            r = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(r, dict) and r.get("kind") == "planning":
            rows.append(dict(file=str(path), dataset=r["dataset"], train_dataset=r["train_dataset"], condition=r["condition"],
                             seed=r["seed"], episodes=len(r["episodes"]), horizon=r["horizon_ticks"],
                             **{f"{k}_success": (r[k] or {}).get("success_rate") for k in ("model", "oracle", "random")},
                             model_error_px=r["model"]["mean_error_px"], zero_error=r["zero"]["mean_error"]))
    return rows


PAIR_REPORT_FIELDS = ("train_dataset", "registration_sha256", "definitions", "cache_identity")


def pairing_problems(mine, other, key_of, row_fields):
    """S25c/S25d 修改（Codex 审核）、S25-fix3 起空间采样也走这里：full 与对照同一种子的两份报告能不能逐行配对。返回 (full 的行, 对照的行, 问题列表)，
    行按 key_of(行) 索引。问题：报告级字段（训练数据集、预登记、评测定义、缓存身份）缺失或不同；某一侧有重复键；
    两侧键集合不同（缺行）；同键的行 manifest_sha256 或 row_fields 不同（不是同一份缓存、同一个窗口）。
    有问题时调用方不出这个比较的平均差（只取交集会让缺行改变甚至反转比较方向）。"""
    problems = []
    for field in PAIR_REPORT_FIELDS:
        if field not in mine or field not in other or mine[field] is None or other[field] is None:
            problems.append(f"报告缺 {field}")
        elif mine[field] != other[field]:
            problems.append(f"{field} 不同")

    def index(report):
        out, dup = {}, 0
        for row in report["rows"]:
            k = key_of(row)
            dup += k in out
            out[k] = row
        return out, dup
    a, da = index(mine); b, db = index(other)
    if da or db:
        problems.append(f"重复的行：full {da} 行、对照 {db} 行")
    only_a, only_b = len(a.keys()-b.keys()), len(b.keys()-a.keys())
    if only_a or only_b:
        problems.append(f"两侧的行不一致：只在 full {only_a} 行、只在对照 {only_b} 行")
    both = a.keys() & b.keys()
    for field in ("manifest_sha256", *row_fields):
        bad = sum(1 for k in both if field not in a[k] or a[k].get(field) != b[k].get(field))
        if bad:
            problems.append(f"{bad} 行的 {field} 缺失或不同")
    return a, b, problems


def _group_steps(group, key):
    """S25-fix1 修改（Codex 复审 P3-1）：一组报告的评测定义（definitions[key]）取第一份有合法定义的报告，
    原来直接索引按文件顺序的第一份，它缺 definitions 时整个汇总 KeyError（第二份缺时才进 incomplete，结果依赖文件顺序）。
    都没有时返回 None，调用方把这组记进 incomplete。缺定义的报告在配对时由 pairing_problems 报出。"""
    for r in group.values():
        d = r.get("definitions")
        if isinstance(d, dict) and isinstance(d.get(key), list) and d[key]:
            return d[key]
    return None


def _no_definitions(dataset, updates, group, key):
    return dict(dataset=dataset, updates=updates, comparison="全部比较",
                per_seed=[dict(seed=seed, problems=[f"报告缺 definitions.{key}"]) for (_, seed) in sorted(group)])


def long_reports(root):
    """S25c tools/long_rollout.py 的结果（kind = long_rollout）：每个 (长时程测试集, 条件) 跨种子平均；
    full − 对照按同编号轨迹配对（先每个种子内逐条相减取平均，再跨种子平均）。配对前用 pairing_problems 核两份报告
    （同训练数据集、预登记、评测定义、缓存身份，轨迹集合相同，每条的起点和缓存 manifest 相同）；任一种子不通过，这个比较
    不出平均差，记进 incomplete。"""
    reports = []
    for path in sorted(Path(root).rglob("*.json")):
        if path.stat().st_size > 50_000_000:
            continue
        try:
            r = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(r, dict) and r.get("kind") == "long_rollout":
            reports.append(dict(r, file=str(path)))
    by = defaultdict(dict)                     # (测试集, 步数) -> (条件, 种子) -> 报告
    duplicates = []
    for r in reports:
        key, sub = (r["dataset"], r["updates"]), (r["condition"], r["seed"])
        if sub in by[key]:
            duplicates.append(r["file"]); continue
        by[key][sub] = r
    rows, comparisons, incomplete = [], [], []
    for (dataset, updates), group in sorted(by.items()):
        times = _group_steps(group, "direct_times")
        if times is None:
            incomplete.append(_no_definitions(dataset, updates, group, "direct_times")); continue
        for c in CONDITIONS:
            seeds = [r for (cc, _), r in sorted(group.items()) if cc == c]
            if seeds:
                rows.append(dict(dataset=dataset, updates=updates, condition=c, seeds=len(seeds), times=times,
                                 direct=[_mean([r["means"][f"long_direct:T={t:g}"] for r in seeds]) for t in times],
                                 composed=[_mean([r["means"][f"long_composed:T={t:g}"] for r in seeds]) for t in times]))
        for ref in REFERENCES:
            diffs, problems = [], []
            for (cc, seed), r in sorted(group.items()):
                other = group.get((ref, seed))
                if cc != "po_jepa_full" or other is None:
                    continue
                mine, theirs, bad = pairing_problems(r, other, lambda row: row["trajectory_id"], ("start_tick",))
                if bad:
                    problems.append(dict(seed=seed, problems=bad)); continue
                t = times[-1]
                d = [mine[k]["values"][f"long_composed:T={t:g}"]-theirs[k]["values"][f"long_composed:T={t:g}"] for k in sorted(mine)]
                if d:
                    diffs.append(dict(seed=seed, mean=sum(d)/len(d), trajectories=len(d), full_better=sum(x < 0 for x in d)))
            if problems:
                incomplete.append(dict(dataset=dataset, updates=updates, comparison=f"po_jepa_full − {ref}", per_seed=problems))
            elif diffs:
                ms = [x["mean"] for x in diffs]
                comparisons.append(dict(dataset=dataset, updates=updates, comparison=f"po_jepa_full − {ref}", T=times[-1],
                                        seeds=len(diffs), mean=_mean(ms), seed_std=statistics.stdev(ms) if len(ms) > 1 else None,
                                        per_seed=diffs))
    return dict(rows=rows, comparisons=comparisons, incomplete=incomplete, duplicates=duplicates)


def timing_reports(root):
    """S25d tools/timing_eval.py 的结果（kind = timing_eval）：每个 (时间采样测试集, 条件) 跨种子平均（按帧数 n）；
    full − 对照按同一窗口（轨迹、起点帧、帧数）配对，比直接预测（先每个种子内逐窗口相减取平均，再跨种子平均）。
    S25d 修改（Codex 审核）：配对前用 pairing_problems 核两份报告（窗口集合相同且不重复、同一缓存 manifest、同起点 tick 和时长
    等），任一种子不通过，这个比较不出平均差，记进 incomplete（原来静默取交集，缺行能让比较方向反转）；
    主表每个指标另给跨种子 SD（*_sd，种子少于 2 个或有种子缺这个键时为 None）。"""
    reports = []
    for path in sorted(Path(root).rglob("*.json")):
        if path.stat().st_size > 50_000_000:
            continue
        try:
            r = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(r, dict) and r.get("kind") == "timing_eval":
            reports.append(dict(r, file=str(path)))
    by = defaultdict(dict)                     # (测试集, 步数) -> (条件, 种子) -> 报告
    duplicates = []
    for r in reports:
        key, sub = (r["dataset"], r["updates"]), (r["condition"], r["seed"])
        if sub in by[key]:
            duplicates.append(r["file"]); continue
        by[key][sub] = r
    rows, comparisons, incomplete = [], [], []

    def mean_of(seeds, key):
        xs = [r["means"][key] for r in seeds if key in r["means"]]
        return _mean(xs) if len(xs) == len(seeds) else None

    def sd_of(seeds, key):
        xs = [r["means"][key] for r in seeds if key in r["means"]]
        return statistics.stdev(xs) if len(xs) == len(seeds) > 1 else None

    for (dataset, updates), group in sorted(by.items()):
        steps = _group_steps(group, "frame_steps")
        if steps is None:
            incomplete.append(_no_definitions(dataset, updates, group, "frame_steps")); continue
        for c in CONDITIONS:
            seeds = [r for (cc, _), r in sorted(group.items()) if cc == c]
            if seeds:
                rows.append(dict(dataset=dataset, updates=updates, condition=c, seeds=len(seeds), frames=steps,
                                 duration=[mean_of(seeds, f"duration_ticks:n={n}") for n in steps],
                                 **{k: [mean_of(seeds, f"{k}:n={n}") for n in steps] for k in ("direct", "composed", "pd_ln", "frame_step")},
                                 **{f"{k}_sd": [sd_of(seeds, f"{k}:n={n}") for n in steps] for k in ("direct", "composed", "frame_step")}))
        key_of = lambda w: (w["trajectory_id"], w["start_frame"], w["frames"])
        for ref in REFERENCES:
            checked, problems = {}, []
            for (cc, seed), r in sorted(group.items()):
                other = group.get((ref, seed))
                if cc != "po_jepa_full" or other is None:
                    continue
                mine, theirs, bad = pairing_problems(r, other, key_of, ("start_tick", "duration_ticks"))
                if bad:
                    problems.append(dict(seed=seed, problems=bad))
                else:
                    checked[seed] = (mine, theirs)
            if problems:
                incomplete.append(dict(dataset=dataset, updates=updates, comparison=f"po_jepa_full − {ref}", per_seed=problems))
                continue
            for n in steps:
                diffs = []
                for seed, (mine, theirs) in sorted(checked.items()):
                    d = [mine[k]["values"][f"direct:n={n}"]-theirs[k]["values"][f"direct:n={n}"] for k in sorted(mine) if k[2] == n]
                    if d:
                        diffs.append(dict(seed=seed, mean=sum(d)/len(d), windows=len(d), full_better=sum(x < 0 for x in d)))
                if diffs:
                    ms = [x["mean"] for x in diffs]
                    comparisons.append(dict(dataset=dataset, updates=updates, comparison=f"po_jepa_full − {ref}", frames=n,
                                            seeds=len(diffs), mean=_mean(ms), seed_std=statistics.stdev(ms) if len(ms) > 1 else None,
                                            per_seed=diffs))
    return dict(rows=rows, comparisons=comparisons, incomplete=incomplete, duplicates=duplicates)


def space_reports(root):
    """S25e tools/space_eval.py 的结果（kind = space_eval）：每个 (空间采样测试集, 条件) 跨种子平均（按时长 T）：
    空间采样测试集上的直接预测、基础测试集同一窗口上的直接预测、两者之差（gap）；
    full − 对照按同一窗口（轨迹、起点、时长）配对，分别比 direct 和 gap（先每个种子内逐窗口相减取平均，再跨种子平均）。
    S25-fix3 修改：配对核对并进 pairing_problems（和长时程、时间采样同一入口）：两份报告同训练数据集、预登记、评测定义、
    缓存身份（空间采样和基础测试集的编码报告），窗口集合相同且不重复，每个窗口的缓存来源（空间采样和基础测试集的 manifest SHA）
    相同；任一种子不通过，这个比较整个不出平均差，记进 incomplete（原来只核窗口和 manifest，记进 mismatched，其余种子照出）。
    定义取组内第一份合法的（_group_steps），都缺时整组记进 incomplete。行里另给跨种子 SD。"""
    reports = []
    for path in sorted(Path(root).rglob("*.json")):
        if path.stat().st_size > 50_000_000:
            continue
        try:
            r = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(r, dict) and r.get("kind") == "space_eval":
            reports.append(dict(r, file=str(path)))
    by = defaultdict(dict)                     # (测试集, 步数) -> (条件, 种子) -> 报告
    duplicates = []
    for r in reports:
        key, sub = (r["dataset"], r["updates"]), (r["condition"], r["seed"])
        if sub in by[key]:
            duplicates.append(r["file"]); continue
        by[key][sub] = r
    rows, comparisons, incomplete = [], [], []

    def mean_of(seeds, key):
        xs = [r["means"][key] for r in seeds if key in r["means"]]
        return _mean(xs) if len(xs) == len(seeds) else None

    def sd_of(seeds, key):
        xs = [r["means"][key] for r in seeds if key in r["means"]]
        return statistics.stdev(xs) if len(xs) == len(seeds) and len(xs) > 1 else None

    key_of = lambda w: (w["trajectory_id"], w["start_tick"], w["duration"])
    for (dataset, updates), group in sorted(by.items()):
        durations = _group_steps(group, "durations")
        if durations is None:
            incomplete.append(_no_definitions(dataset, updates, group, "durations")); continue
        for c in CONDITIONS:
            seeds = [r for (cc, _), r in sorted(group.items()) if cc == c]
            if seeds:
                rows.append(dict(dataset=dataset, updates=updates, condition=c, seeds=len(seeds), durations=durations,
                                 **{k: [mean_of(seeds, f"{k}:T={t}") for t in durations] for k in ("direct", "base_direct", "gap")},
                                 **{f"{k}_sd": [sd_of(seeds, f"{k}:T={t}") for t in durations] for k in ("direct", "gap")}))
        for ref in REFERENCES:
            checked, problems = {}, []
            for (cc, seed), r in sorted(group.items()):
                other = group.get((ref, seed))
                if cc != "po_jepa_full" or other is None:
                    continue
                mine, theirs, bad = pairing_problems(r, other, key_of, ("base_manifest_sha256",))
                if bad:
                    problems.append(dict(seed=seed, problems=bad))
                else:
                    checked[seed] = (mine, theirs)
            if problems:
                incomplete.append(dict(dataset=dataset, updates=updates, comparison=f"po_jepa_full − {ref}", per_seed=problems))
                continue
            for metric in ("direct", "gap"):
                for t in durations:
                    diffs = []
                    for seed, (mine, theirs) in sorted(checked.items()):
                        d = [w["values"][f"{metric}:T={t}"]-theirs[k]["values"][f"{metric}:T={t}"]
                             for k, w in sorted(mine.items()) if w["duration"] == t]
                        if d:
                            diffs.append(dict(seed=seed, mean=sum(d)/len(d), windows=len(d), full_better=sum(x < 0 for x in d)))
                    if diffs:
                        ms = [x["mean"] for x in diffs]
                        comparisons.append(dict(dataset=dataset, updates=updates, comparison=f"po_jepa_full − {ref}", metric=metric,
                                                T=t, seeds=len(diffs), mean=_mean(ms),
                                                seed_std=statistics.stdev(ms) if len(ms) > 1 else None, per_seed=diffs))
    return dict(rows=rows, comparisons=comparisons, incomplete=incomplete, duplicates=duplicates)


def _f(x, digits=5):
    return "—" if x is None else f"{x:.{digits}f}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"


def _incomplete_lines(title, incomplete):
    """配对核对没通过、没出平均差的比较（pairing_problems）。"""
    if not incomplete:
        return []
    return ["", f"⚠ {title}：以下比较的两份报告不能逐行配对，没给平均差：", ""] + [
        f"- {c['dataset']} {c['updates']} 步 {c['comparison']}：" + "；".join(f"种子 {x['seed']}：" + "、".join(x["problems"]) for x in c["per_seed"])
        for c in incomplete]


def _markdown(s):
    lines = ["# v7 汇总", "", f"任务数：{s['jobs']}。未经人工核对，不构成科学结论。", ""]
    if s["problems"]:
        lines += ["## ⚠ 核对未通过", ""] + [f"- {p['dataset']} 种子 {p['seed']}：{p['problem']}" for p in s["problems"]] + [""]
    lines += ["## 各条件（跨种子平均）", "", "| 数据集 | 步数 | 条件 | 种子 | 未来误差 | 种子间 SD | delta_act | TCD_norm | SCD_norm |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in s["across_seeds"]:
        if "condition" in r and r["condition"] not in SELF_CHECK:
            lines.append(f"| {r['dataset']} | {r['updates']} | {r['condition']} | {r['seeds']} | {_f(r['future_mean'])} | "
                         f"{_f(r['future_seed_std'])} | {_f(r['delta_act_mean'])} | {_f(r['tcd_normalized_mean'])} | {_f(r['scd_normalized_mean'])} |")
    lines += ["", "## 按网格拆开（跨种子平均；16×16 有 JEPA 监督，12×12、8×8 没有）", "",
              "| 数据集 | 步数 | 条件 | 16×16 | 12×12 | 8×8 | 16×16 多步滚动 | 同窗口直接预测 | 实际训练步数 | 早停的种子 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["across_seeds"]:
        if "condition" in r and r["condition"] not in SELF_CHECK:
            g = r["future_by_grid"]
            lines.append(f"| {r['dataset']} | {r['updates']} | {r['condition']} | {_f(g['16'])} | {_f(g['12'])} | {_f(g['8'])} | "
                         f"{_f(r['rollout_future_mean'])} | {_f(r['rollout_direct_mean'])} | "
                         f"{', '.join(str(u) for u in r['completed_updates'])} | {r['stopped_early']}/{r['seeds']} |")
    checks = [r for r in s["across_seeds"] if r.get("condition") in SELF_CHECK]
    if checks:   # S28b（C3）：identity 只作数值自检和归一化参照，不进上面两张表和配对比较
        lines += ["", "## 数值自检：identity（不作对比基线，只用来核对数据和归一化）", "",
                  "| 数据集 | 步数 | 种子 | 未来误差 | 16×16 | 12×12 | 8×8 | 16×16 多步滚动 |", "|---|---|---|---|---|---|---|---|"]
        for r in checks:
            g = r["future_by_grid"]
            lines.append(f"| {r['dataset']} | {r['updates']} | {r['seeds']} | {_f(r['future_mean'])} | {_f(g['16'])} | {_f(g['12'])} | "
                         f"{_f(g['8'])} | {_f(r['rollout_future_mean'])} |")
    lines += ["", "## 配对比较按网格（full − 对照，跨种子平均；负数 = full 更好）", "",
              "| 数据集 | 步数 | 比较 | 16×16 | 12×12 | 8×8 | 16×16 多步滚动 |", "|---|---|---|---|---|---|---|"]
    for r in s["across_seeds"]:
        if "comparison" in r:
            g = r["by_grid"]
            lines.append(f"| {r['dataset']} | {r['updates']} | {r['comparison']} | {_f(g['16'])} | {_f(g['12'])} | {_f(g['8'])} | {_f(r['rollout'])} |")
    lines += ["", "## 配对比较（full − 对照，负数 = full 更好）", "",
              "| 数据集 | 步数 | 比较 | 种子 | 平均差 | 种子间 SD | full 更好的种子 | 每个种子里 full 更好的轨迹 |", "|---|---|---|---|---|---|---|---|"]
    for r in s["across_seeds"]:
        if "comparison" in r:
            traj = ", ".join(f"{w}/{n}" for w, n in zip(r["trajectories_full_better"], r["trajectories"]))
            lines.append(f"| {r['dataset']} | {r['updates']} | {r['comparison']} | {r['seeds']} | {_f(r['mean'])} | "
                         f"{_f(r['seed_std'])} | {r['seeds_full_better']}/{r['seeds']} | {traj} |")
    if s.get("equal_step"):
        lines += ["", "## 同步数对照（补充）：对照截到 po_jepa_full 实际训练的步数（full − 对照，负数 = full 更好）", "",
                  "| 数据集 | 步数 | 种子 | 对照 | full 步数 | 对照选中步 | 对照各自早停的选中步 | 16×16 | 12×12 | 8×8 | 16×16 滚动 | 主表 16×16 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in s["equal_step"]:
            e = {g: (r["equal_step"][g] or {}).get("mean") for g in ("16", "12", "8")}
            m = (r["main"] or {}).get("16") or {}
            lines.append(f"| {r['dataset']} | {r['updates']} | {r['seed']} | {r['condition']} | {r['limit_updates']} | {r['selected_step']} | "
                         f"{r['own_best_step']} | {_f(e['16'])} | {_f(e['12'])} | {_f(e['8'])} | "
                         f"{_f((r['equal_step_rollout'] or {}).get('mean'))} | {_f(m.get('mean'))} |")
    if s["transfer"]:
        lines += ["", "## 分布外评测：换调节项后的变化（正数 = 变差；同编号窗口配对）", "",
                  "| 训练 → 测试 | 步数 | 种子 | 条件 | 未来误差变化 | SE | 轨迹数 |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['transfer']} | {r['updates']} | {r['seed']} | {r['condition']} | {_f(r['change'])} | {_f(r['se'])} | {r['trajectories']} |"
                  for r in s["transfer"]]
    if s["planning"]:
        lines += ["", "## 规划：推到目标的成功率（误差 ≤ 0.5 × 不动时的误差）", "",
                  "| 场景 | 训练数据集 | 条件 | 种子 | 回合 | 模型规划 | 真值规划（上界） | 随机 | 模型规划误差（像素） |", "|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['dataset']} | {r['train_dataset']} | {r['condition']} | {r['seed']} | {r['episodes']} | {_f(r['model_success'], 2)} | "
                  f"{_f(r['oracle_success'], 2)} | {_f(r['random_success'], 2)} | {_f(r['model_error_px'], 1)} |" for r in s["planning"]]
    if s["long"]["rows"]:
        lines += ["", "## 长时程滚动（起点 tick 0，跨种子平均；直接 = 一次预测，8 步滚动 = 每 8 个 tick 一段）", ""]
        times = s["long"]["rows"][0]["times"]
        lines += ["| 测试集 | 步数 | 条件 | 种子 | " + " | ".join(f"直接 T={t:g}" for t in times) + " | " +
                  " | ".join(f"8 步滚动 T={t:g}" for t in times) + " |", "|---|---|---|---|" + "---|"*(2*len(times))]
        lines += [f"| {r['dataset']} | {r['updates']} | {r['condition']} | {r['seeds']} | " + " | ".join(_f(x) for x in r["direct"]) +
                  " | " + " | ".join(_f(x) for x in r["composed"]) + " |" for r in s["long"]["rows"] if r["times"] == times]
        if s["long"]["comparisons"]:
            lines += ["", "| 测试集 | 步数 | 比较（8 步滚动，最长 T） | T | 种子 | 平均差 | 种子间 SD | 每个种子里 full 更好的轨迹 |",
                      "|---|---|---|---|---|---|---|---|"]
            lines += [f"| {c['dataset']} | {c['updates']} | {c['comparison']} | {c['T']:g} | {c['seeds']} | {_f(c['mean'])} | {_f(c['seed_std'])} | "
                      + ", ".join(f"{x['full_better']}/{x['trajectories']}" for x in c["per_seed"]) + " |" for c in s["long"]["comparisons"]]
    lines += _incomplete_lines("长时程", s["long"].get("incomplete", []))
    timing = s.get("timing") or dict(rows=[], comparisons=[])
    if timing["rows"]:
        lines += ["", "## 时间采样（按帧评测，跨种子平均；n = 往后预测的帧数，T = 平均时长（tick）；直接 / 逐帧组合 / AC 一帧一步）", ""]
        frames = timing["rows"][0]["frames"]
        lines += ["| 测试集 | 步数 | 条件 | 种子 | " + " | ".join(f"n={n}" for n in frames) + " |", "|---|---|---|---|" + "---|"*len(frames)]
        for r in (r for r in timing["rows"] if r["frames"] == frames):
            cells = []
            for i in range(len(frames)):
                if r["direct"][i] is None:
                    cells.append("—"); continue
                text = f"T={r['duration'][i]:.3g}: {_f(r['direct'][i])} / {_f(r['composed'][i])}"
                if r["frame_step"][i] is not None:
                    text += f" / {_f(r['frame_step'][i])}"
                cells.append(text)
            lines.append(f"| {r['dataset']} | {r['updates']} | {r['condition']} | {r['seeds']} | " + " | ".join(cells) + " |")
        lines += ["", "跨种子 SD（直接 / 逐帧组合 / AC 一帧一步；— = 种子少于 2 个或有种子缺这一项）：", "",
                  "| 测试集 | 步数 | 条件 | 种子 | " + " | ".join(f"n={n}" for n in frames) + " |", "|---|---|---|---|" + "---|"*len(frames)]
        for r in (r for r in timing["rows"] if r["frames"] == frames):
            cells = [" / ".join(_f(r[f"{k}_sd"][i]) for k in ("direct", "composed") + (("frame_step",) if r["frame_step"][i] is not None else ()))
                     for i in range(len(frames))]
            lines.append(f"| {r['dataset']} | {r['updates']} | {r['condition']} | {r['seeds']} | " + " | ".join(cells) + " |")
        if timing["comparisons"]:
            lines += ["", "| 测试集 | 步数 | 比较（直接预测） | 帧数 n | 种子 | 平均差 | 种子间 SD | 每个种子里 full 更好的窗口 |",
                      "|---|---|---|---|---|---|---|---|"]
            lines += [f"| {c['dataset']} | {c['updates']} | {c['comparison']} | {c['frames']} | {c['seeds']} | {_f(c['mean'])} | {_f(c['seed_std'])} | "
                      + ", ".join(f"{x['full_better']}/{x['windows']}" for x in c["per_seed"]) + " |" for c in timing["comparisons"]]
    lines += _incomplete_lines("时间采样", timing.get("incomplete", []))
    space = s.get("space") or dict(rows=[], comparisons=[])
    if space["rows"]:
        lines += ["", "## 空间采样（主实验测试窗口，跨种子平均；每格 = 空间采样测试集上的直接预测 / 基础测试集同一窗口 / 两者之差）", ""]
        durations = space["rows"][0]["durations"]
        lines += ["| 测试集 | 步数 | 条件 | 种子 | " + " | ".join(f"T={t}" for t in durations) + " |", "|---|---|---|---|" + "---|"*len(durations)]
        for r in (r for r in space["rows"] if r["durations"] == durations):
            cells = ["—" if r["direct"][i] is None else f"{_f(r['direct'][i])} / {_f(r['base_direct'][i])} / {_f(r['gap'][i])}"
                     for i in range(len(durations))]
            lines.append(f"| {r['dataset']} | {r['updates']} | {r['condition']} | {r['seeds']} | " + " | ".join(cells) + " |")
        if space["comparisons"]:
            lines += ["", "| 测试集 | 步数 | 比较 | 量 | T | 种子 | 平均差 | 种子间 SD | 每个种子里 full 更好的窗口 |",
                      "|---|---|---|---|---|---|---|---|---|"]
            lines += [f"| {c['dataset']} | {c['updates']} | {c['comparison']} | {c['metric']} | {c['T']} | {c['seeds']} | {_f(c['mean'])} | "
                      f"{_f(c['seed_std'])} | " + ", ".join(f"{x['full_better']}/{x['windows']}" for x in c["per_seed"]) + " |"
                      for c in space["comparisons"]]
    lines += _incomplete_lines("空间采样", space.get("incomplete", []))
    if s["missing"]:
        lines += ["", "## 缺失任务", ""] + [f"- {m['dataset']} {m['updates']} 步 种子 {m['seed']}：{m['condition']}" for m in s["missing"]]
    if s["duplicates"]:
        lines += ["", "## 重复任务（只用了第一个）", ""] + [f"- {d['condition']}：用 {d['kept']}，忽略 {d['ignored']}" for d in s["duplicates"]]
    lines += ["", "收敛图在 figures/ 目录。"]
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    main()
