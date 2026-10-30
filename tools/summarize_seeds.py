"""v6 多种子汇总：读取若干次 `python run.py run` 的输出目录，给出 full / 无一致性 / Transformer / identity 的配对比较。

用法（在代码根目录）：
  python tools/summarize_seeds.py ../../results/run-1703 ../../results/run-1704 ../../results/run-1705
每个参数是 run.py 的 --output 目录（里面有 run/evaluation 与 run/training）。结果打印在屏幕上，并写到
第一个目录旁边的 seeds-summary.json。只读结果文件，不需要 GPU，也不读特征。

统计口径：每条测试轨迹（20 条）先对 80 个单网格点窗口的 future_jepa_distance 取平均，再在轨迹层面配对比较；
多种子时先在种子内配对，再报告每个种子的结果和三个种子平均后的配对结果。
"""
import argparse
import json
import math
from pathlib import Path
import statistics

CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity")
PAIRS = (("po_jepa_full", "identity"), ("po_jepa_no_consistency", "identity"),
         ("po_jepa_full", "po_jepa_no_consistency"), ("po_jepa_full", "matched_transformer"))


def load_rows(run):
    """一个运行目录 → {condition: [row,...]}；缺条件或有失败行直接报错。"""
    root = Path(run)/"run"/"evaluation"
    if not (root/"COMPLETE.json").exists():
        raise SystemExit(f"{run}: 没有完成的评价（缺 run/evaluation/COMPLETE.json）")
    rows = {}
    for condition in CONDITIONS:
        items = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root/condition/"rows").glob("*.json"))]
        if not items or any(r["status"] != "ok" for r in items):
            raise SystemExit(f"{run}: {condition} 的评价行缺失或有失败")
        rows[condition] = items
    return rows


def trajectory_means(items, metric="future_jepa_distance"):
    """单网格点窗口按轨迹取平均。"""
    groups = {}
    for r in items:
        if r["query"]["kind"] == "point":
            groups.setdefault(r["query"]["trajectory_id"], []).append(r["values"][metric])
    return {t: sum(v)/len(v) for t, v in groups.items()}


def paired(a, b):
    """轨迹层面配对差 a−b：均值、标准误、a 更好的轨迹数、双侧符号检验 p 值。"""
    keys = sorted(a)
    if keys != sorted(b):
        raise SystemExit("两条件的测试轨迹不一致")
    d = [a[k]-b[k] for k in keys]
    n = len(d); mean = sum(d)/n
    se = statistics.stdev(d)/math.sqrt(n) if n > 1 else float("nan")
    wins = sum(x < 0 for x in d); ties = sum(x == 0 for x in d); m = n-ties
    tail = sum(math.comb(m, i) for i in range(0, min(wins, m-wins)+1))/2**m if m else 1.
    return dict(mean_difference=mean, standard_error=se, better_trajectories=wins, trajectories=n,
                sign_test_p=min(1., 2*tail))


def medians(items, metric, kind):
    values = [r["values"][metric] for r in items if r["query"]["kind"] == kind and metric in r["values"]]
    return statistics.median(values) if values else None


def monitor(run):
    """训练报告里的一致性梯度占比（v6 新增）；旧版本结果没有这一项时返回 None。"""
    path = Path(run)/"run"/"training"/"po_jepa_full"/"report.json"
    return json.loads(path.read_text(encoding="utf-8")).get("consistency_monitor") if path.exists() else None


def seed_of(items):
    return items[0]["seed"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    per_seed, means = [], []
    for run in args.runs:
        rows = load_rows(run)
        m = {c: trajectory_means(rows[c]) for c in CONDITIONS}
        means.append(m)
        per_seed.append(dict(run=str(run), seed=seed_of(rows["po_jepa_full"]),
            test_mean={c: sum(v.values())/len(v) for c, v in m.items()},
            paired={f"{a} - {b}": paired(m[a], m[b]) for a, b in PAIRS},
            tcd_normalized_median={c: medians(rows[c], "tcd_normalized", "point") for c in CONDITIONS[:3]},
            scd_normalized_median={c: medians(rows[c], "scd_normalized", "pair") for c in CONDITIONS[:3]},
            delta_act_median={c: medians(rows[c], "delta_act", "point") for c in CONDITIONS[:3]},
            consistency_monitor=monitor(run)))
    seeds = [s["seed"] for s in per_seed]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"种子重复：{seeds}")
    average = {c: {t: sum(m[c][t] for m in means)/len(means) for t in means[0][c]} for c in CONDITIONS}
    pooled = {f"{a} - {b}": paired(average[a], average[b]) for a, b in PAIRS}
    full_better_seeds = sum(s["paired"]["po_jepa_full - po_jepa_no_consistency"]["mean_difference"] < 0 for s in per_seed)
    result = dict(seeds=seeds, per_seed=per_seed, seed_averaged_paired=pooled,
                  full_better_than_no_consistency_in_seeds=f"{full_better_seeds}/{len(per_seed)}")
    out = args.runs[0].resolve().parent/"seeds-summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in per_seed:
        print(f"种子 {s['seed']}：" + "，".join(f"{c} {v:.4f}" for c, v in s["test_mean"].items()))
        cm = s["consistency_monitor"]
        if cm:
            print(f"  一致性梯度占比中位数 {cm['median_share']}（rho={cm.get('rho')}，参与训练={cm['active']}）")
    print("三个种子平均后的轨迹配对比较（差值为负表示前者误差更低）：" if len(per_seed) > 1 else "轨迹配对比较：")
    for name, p in pooled.items():
        print(f"  {name}: {p['mean_difference']:+.5f} ± {p['standard_error']:.5f}，"
              f"{p['better_trajectories']}/{p['trajectories']} 条轨迹更好，符号检验 p={p['sign_test_p']:.3g}")
    print(f"full 比无一致性更好的种子数：{result['full_better_than_no_consistency_in_seeds']}")
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
