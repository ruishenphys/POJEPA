"""v7 汇总：把各自独立提交的任务（系统 × 种子 × 条件）合成一张总表、配对比较和收敛图。

用法：
  python tools/summarize_v7.py --results ../results-v7 --output ../results-v7/summary
--results 下任意层级里每个含 job.json + COMPLETE.json 的目录都算一个完成的任务（run.py 的输出是 <dir>/run/job/）。
只读，不需要 GPU。缺的任务会列在 SUMMARY.md 的“缺失任务”里，不会中止。

核对项（不通过就在报告里标红，不静默合并）：
- 同一 (数据集, 种子) 下所有条件的预登记（源码、配置、查询序列）SHA 必须一致；
- po_jepa_full 与 po_jepa_no_consistency 的初始权重 SHA 必须一致（唯一差别是一致性项）。

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

CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity", "vjepa2_ac_small", "vjepa2_ac")
REFERENCES = CONDITIONS[1:]
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
    a = p.parse_args()
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
    missing = [dict(dataset=d, updates=u, seed=s, condition=c) for (d, u, s), by_c in sorted(groups.items())
               for c in CONDITIONS if c not in by_c]

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
                   transfer=transfer_drops(groups), planning=planning_reports(a.results),
                   problems=problems, missing=missing, duplicates=duplicates, scientific_conclusions_verified=False)
    (a.output/"summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (a.output/"SUMMARY.md").write_text(_markdown(summary), encoding="utf-8")
    print(f"{len(jobs)} 个任务；问题 {len(problems)} 个；缺失 {len(missing)} 个 → {a.output/'SUMMARY.md'}")


def equal_step_rows(groups, equal_jobs, problems):
    """同步数对照（补充）：po_jepa_full − 截到 full 步数的对照，按网格配对；与主表的 full − 对照并列。"""
    rows = []
    for j in equal_jobs:
        key = (j["job"]["dataset"], j["job"]["updates"], j["job"]["seed"])
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


def _label(job):
    """普通任务：数据集；分布外评测：训练数据集 → 测试数据集。"""
    return job["dataset"] if job.get("kind") != "transfer" else f"{job['train_dataset']} → {job['dataset']}"


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


def _f(x, digits=5):
    return "—" if x is None else f"{x:.{digits}f}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"


def _markdown(s):
    lines = ["# v7 汇总", "", f"任务数：{s['jobs']}。未经人工核对，不构成科学结论。", ""]
    if s["problems"]:
        lines += ["## ⚠ 核对未通过", ""] + [f"- {p['dataset']} 种子 {p['seed']}：{p['problem']}" for p in s["problems"]] + [""]
    lines += ["## 各条件（跨种子平均）", "", "| 数据集 | 步数 | 条件 | 种子 | 未来误差 | 种子间 SD | delta_act | TCD_norm | SCD_norm |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in s["across_seeds"]:
        if "condition" in r:
            lines.append(f"| {r['dataset']} | {r['updates']} | {r['condition']} | {r['seeds']} | {_f(r['future_mean'])} | "
                         f"{_f(r['future_seed_std'])} | {_f(r['delta_act_mean'])} | {_f(r['tcd_normalized_mean'])} | {_f(r['scd_normalized_mean'])} |")
    lines += ["", "## 按网格拆开（跨种子平均；16×16 有 JEPA 监督，12×12、8×8 没有）", "",
              "| 数据集 | 步数 | 条件 | 16×16 | 12×12 | 8×8 | 16×16 多步滚动 | 同窗口直接预测 | 实际训练步数 | 早停的种子 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["across_seeds"]:
        if "condition" in r:
            g = r["future_by_grid"]
            lines.append(f"| {r['dataset']} | {r['updates']} | {r['condition']} | {_f(g['16'])} | {_f(g['12'])} | {_f(g['8'])} | "
                         f"{_f(r['rollout_future_mean'])} | {_f(r['rollout_direct_mean'])} | "
                         f"{', '.join(str(u) for u in r['completed_updates'])} | {r['stopped_early']}/{r['seeds']} |")
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
    if s["missing"]:
        lines += ["", "## 缺失任务", ""] + [f"- {m['dataset']} {m['updates']} 步 种子 {m['seed']}：{m['condition']}" for m in s["missing"]]
    if s["duplicates"]:
        lines += ["", "## 重复任务（只用了第一个）", ""] + [f"- {d['condition']}：用 {d['kept']}，忽略 {d['ignored']}" for d in s["duplicates"]]
    lines += ["", "收敛图在 figures/ 目录。"]
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    main()
