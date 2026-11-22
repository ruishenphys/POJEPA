"""小规模验证（方案 `full-model-next-trial` §5）的汇总：C-1 的 2×2 选型、C-2 的验收表（方案 §5.4）、B 层读出汇总。只读，不需要 GPU。

  python tools/summarize_small_validation.py c1 --results ../../results-small-validation/c1 --output …/c1/summary
  python tools/summarize_small_validation.py c2 --results ../../results-small-validation/c2/<变体> --v7-results ../../results-v7/full/v7.1 \\
      --readout …/c2/<变体>/readout --v7-readout ../../results-small-validation/b-v7.1 --output …/c2/<变体>/acceptance
  python tools/summarize_small_validation.py readout --results ../../results-small-validation/b-v7.1 \\
      --actions ../../results-small-validation/readout --output …/b-v7.1/summary

输出 summary.json 和 SUMMARY.md。所有判定是 Claude 提的阈值（方案 §5.4 写明 Jerry 可改），代码里集中在 RULES；
判定只读测试集上已登记的评测行（run.py 写的 evaluation/<条件>/rows），不重算模型，不看训练曲线。

准入（S28c-fix1，Codex S28c P1-1/P1-2/P2-4/P2-5）：一个任务进判定之前要“有效”——
- 任务身份对：数据集 <场景>-<条数>、步数预算、种子、条件、变体是这一阶段的（C-1：200 条 × 2000 步；C-2 和 v7.1：2000 条 × 10000 步）；
- 评测行完整：键集合正好等于 evaluation/<条件>/registration.json 登记的查询（非空），没有重复键、没有失败行、值全部有限；
- 判定要用的指标都算得出来。
无效任务列进 invalid（带原因），不进指标表、不参与多数票：C-1 有任何无效格就不给推荐；C-2 对应的条款因缺场景判未通过。
C-2 里一个结果根混了几个变体时不出判定。第 5 条的读出另核来源：训练数据集、horizon、曲线时间点、探针权重 SHA、真值 SHA
在比较的几份读出之间一致，读出记的检查点 SHA 等于对应训练任务选中的检查点；不一致的读出不进比较。

指标（每个任务，只取 status = ok 的单点行和网格对行；“均值”是行的算术平均，和 summarize_v7 的 condition_means 同一口径）：
- direct16：16×16 单点 future_jepa_distance 的均值（6 个时长）；heldout16：16×16、留出时长 T ∈ {3, 12} 的均值（外推时长）；
- grid12、grid8：12×12、8×8 单点的均值（换网格）；
- rollout16@8、rollout16@12：16×16、T = 8、12 的逐步滚动误差 rollout_future_jepa_distance（每步 1 周期，v7.1 起）；
- tcd：全部单点 tcd_normalized 的均值；scd：网格对 scd_normalized 的均值；delta_act：全部单点的均值；
- 相对变化 rel(a, b) = (a − b)/b，“更好”= 误差更低。

C-1 规则（推荐 C-2 用哪个变体；四个变体都只训 po_jepa_full）：
- M1（多段切分）：在零初始化列比 dyadic-zero 对 bisection-zero。dyadic 被采用，当且仅当 rollout16@8 和 rollout16@12 都更低的场景
  ≥ 2/3，且每个场景 direct16 的相对变化 ≤ +0.5%。Xavier 列的同一比较只报告，不进判定（P_out 零初始化是默认要做的，方案 §2 M2）。
- M2（零初始化）：在选定的切分下比 <P>-zero 对 <P>-xavier。zero 被采用，当且仅当 direct16 相对变化 ≤ +0.5% 的场景 ≥ 2/3。
- 缺任何一格就不给推荐。C-2 的任务表要显式写变体（make_small_validation_jobs.py --stage c2），这里只打印推荐的命令。

C-2 验收（方案 §5.4 的第 2–6 条；第 1 条是 A 层测试、第 7 条是工程清单，这里只列 NaN、早停和缺失）：
- 前提：每个场景本次 identity 的全部评测行和 v7.1 种子 1703 的 identity 逐行逐值相同（键集合相同、values 字典完全相等）。
  通过的场景才复用 v7.1 的 matched_transformer、vjepa2_ac_small、vjepa2_ac 和 v7.1 的 po_jepa_full（底数）；不通过的场景这些对照记“不可比”。
- 第 2 条（时间一致性起作用）：full 对 no-cons，rollout16@8、rollout16@12、tcd 都更低，且 direct16 相对变化 ≤ +0.5%；3/3 场景。
- 第 3 条（空间一致性保持）：grid8 full 低于 no-cons，3/3 场景。
- 第 4 条（结构有必要）：OOD 指标 grid8、grid12、rollout16@12、heldout16。对 transformer_consistency“不输”= 相对变化 ≤ +0.5%，
  每个指标都要在 ≥ 2/3 场景不输；对 fixed_fourier，一个场景里 4 个指标至少 3 个更低算这个场景胜，≥ 2/3 场景胜。
- 第 5 条（学到物理，需要 R 段的读出结果）：读出误差 = 长时程读出 nmse 曲线（每 8 tick 一点）的均值，VPT 取 vpt@0.1。
  full 对 no-cons：nmse 更低且 VPT 不低，≥ 2/3 场景；full 对 v7.1 的 matched_transformer（同一探针、同一真值，b-v7.1 的结果）：
  nmse 相对变化 ≤ +0.5%，≥ 2/3 场景；反事实：full 的 cos_mean > 0 且 gain 高于 v7.1 的 po_jepa_full（种子 1703），3/3 场景。
  “明显提高”的幅度没有定量阈值，表里列出数值。
- 第 6 条（零初始化不伤）：(full − TF)/TF 的 direct16 差距不比 v7.1 的 (full − TF)/TF 大，≥ 2/3 场景（TF = v7.1 的 matched_transformer）。
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

SCENES = ("advection_diffusion", "double_well", "van_der_pol")
SEED = 1703
HELDOUT_TIMES = (3., 12.)
ROLLOUT_TIMES = (8., 12.)
PARTITIONS = ("bisection", "dyadic")
INITS = ("xavier", "zero")
DEFAULT_VARIANT = dict(temporal_partition="bisection", readout_init="xavier")
REUSED_FROM_V71 = ("matched_transformer", "vjepa2_ac_small", "vjepa2_ac")
V71_BASELINE = "po_jepa_full"
OOD_METRICS = ("grid8", "grid12", "rollout16@12", "heldout16")
VPT_KEY = "vpt@0.1"
# 判定阈值（方案 §5.4：Claude 提的，Jerry 可改）
RULES = dict(direct_tolerance=.005, not_worse_tolerance=.005, majority=2, all_scenes=3, ff_metric_wins=3)
C1_DATA = dict(n=200, updates=2000)         # 和 make_small_validation_jobs.C1、C2 一致（测试核对）
C2_DATA = dict(n=2000, updates=10000)       # v7.1 正式运行同一规模
LONG_HORIZON = 128
C1_METRICS = ("direct16", "rollout16@8", "rollout16@12")
TRAINED_METRICS = ("direct16", "heldout16", "grid12", "grid8", "rollout16@8", "rollout16@12")


# —— 读任务 ——

def _key(q):
    return (q["kind"], q["trajectory_id"], q["start_tick"], float(q["duration_cycles"]), q["grid_size"], q.get("stratum"))


def job_folders(root):
    """root 下任意层级里每个有 job.json、COMPLETE.json、report.json 的目录（run.py 的输出是 <dir>/run/job/）。"""
    for path in sorted(Path(root).rglob("job.json")):
        folder = path.parent
        if (folder/"COMPLETE.json").exists() and (folder/"report.json").exists():
            yield folder


def load_job(folder):
    """job.json、report.json、登记的查询（evaluation/<条件>/registration.json）和全部评测行。
    status 不是 ok 的行单独计数（不进均值）；同一查询键出现第二次记为重复，保留第一行，不静默覆盖。"""
    folder = Path(folder)
    job = json.loads((folder/"job.json").read_text(encoding="utf-8"))
    report = json.loads((folder/"report.json").read_text(encoding="utf-8"))
    evaluation = folder/"evaluation"/job["condition"]
    expected = None
    if (evaluation/"registration.json").exists():
        queries = json.loads((evaluation/"registration.json").read_text(encoding="utf-8")).get("queries") or []
        expected = [_key(q) for q in queries]
    rows, failed, duplicates, seen = {}, 0, 0, set()
    for path in sorted((evaluation/"rows").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        k = _key(r["query"])
        if k in seen:
            duplicates += 1; continue
        seen.add(k)
        if r["status"] == "ok":
            rows[k] = r
        else:
            failed += 1
    return dict(folder=str(folder), job=job, report=report, rows=rows, failed_rows=failed, duplicate_rows=duplicates,
                expected_keys=expected)


def row_problems(j) -> list:
    """评测行的完整性：登记的查询非空、行键正好覆盖它们、没有重复键、没有失败行、值全部有限。空表示完整。"""
    out = []
    expected = j["expected_keys"]
    if not expected:
        out.append("没有登记的评测查询（evaluation/<条件>/registration.json 缺失或为空）")
    elif len(set(expected)) != len(expected):
        out.append("登记的评测查询有重复")
    elif set(j["rows"]) != set(expected) and not j["failed_rows"]:
        out.append(f"评测行没有覆盖登记的查询：登记 {len(set(expected))}，成功 {len(j['rows'])}，"
                   f"缺 {len(set(expected)-set(j['rows']))}，多 {len(set(j['rows'])-set(expected))}")
    if j["duplicate_rows"]:
        out.append(f"{j['duplicate_rows']} 行的查询键重复")
    if j["failed_rows"]:
        out.append(f"{j['failed_rows']} 行评测失败")
    if not all(_finite(r["values"]) for r in j["rows"].values()):
        out.append("评测值有非有限数")
    return out


def job_identity_problems(job, *, scene, data, condition=None, variant=None) -> list:
    """任务身份：数据集、步数预算、种子、条件、变体是不是这一阶段的。"""
    out = []
    if job.get("dataset") != f"{scene}-{data['n']}":
        out.append(f"数据集 {job.get('dataset')} 不是 {scene}-{data['n']}")
    if job.get("updates") != data["updates"]:
        out.append(f"步数预算 {job.get('updates')} 不是 {data['updates']}")
    if job.get("seed") != SEED:
        out.append(f"种子 {job.get('seed')} 不是 {SEED}")
    if condition is not None and job.get("condition") != condition:
        out.append(f"条件 {job.get('condition')} 不是 {condition}")
    if variant is not None and variant_of(job) != variant:
        out.append(f"变体 {variant_label(variant_of(job))} 不是 {variant_label(variant)}")
    if job.get("kind"):
        out.append(f"是 {job['kind']} 任务")
    return out


def metric_problems(m, keys) -> list:
    missing = [k for k in keys if m.get(k) is None]
    return [f"算不出 {', '.join(missing)}"] if missing else []


def finished_attempt(path):
    """一个任务输出目录（run.py 的 --output）或它的续训目录（-resume1、-resume2…，tools/run_jobs.py 的写法）里完成的那个，
    返回含 job.json 的目录；都没完成返回 None。"""
    path = Path(path)
    attempts, k = [path], 1
    while Path(f"{path}-resume{k}").exists():
        attempts.append(Path(f"{path}-resume{k}")); k += 1
    for p in attempts:
        for sub in ("run/job", "job", "run", ""):
            folder = p/sub if sub else p
            if (folder/"job.json").exists() and (folder/"COMPLETE.json").exists() and (folder/"report.json").exists():
                return folder
    return None


def variant_of(job) -> dict:
    """S28b：job.json 的 variant（只在非默认变体时写）；没有就是默认变体。"""
    return dict(DEFAULT_VARIANT, **(job.get("variant") or {}))


def variant_label(variant) -> str:
    """bisection-xavier 这种写法；C3 的空间变体（K、补空间）再接 -K32c、-Kfull、-K16c（和结果目录名一致）。"""
    label = f"{variant['temporal_partition']}-{variant['readout_init']}"
    modes, complement = variant.get("modes", "16"), variant.get("complement", "none")
    if modes != "16" or complement != "none":
        label += f"-K{modes}"+("c" if complement == "shared" else "")
    return label


def scene_of(job) -> str:
    return job["dataset"].rpartition("-")[0]


# —— 指标 ——

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def metrics(job) -> dict:
    """文件头“指标”一节的量。"""
    points = [(k, r["values"]) for k, r in job["rows"].items() if k[0] == "point"]
    pairs = [r["values"] for k, r in job["rows"].items() if k[0] == "pair"]

    def future(select):
        return _mean([v.get("future_jepa_distance") for k, v in points if select(k)])
    out = dict(direct16=future(lambda k: k[4] == 16),
               heldout16=future(lambda k: k[4] == 16 and k[3] in HELDOUT_TIMES),
               grid12=future(lambda k: k[4] == 12), grid8=future(lambda k: k[4] == 8))
    for t in ROLLOUT_TIMES:
        out[f"rollout16@{t:g}"] = _mean([v.get("rollout_future_jepa_distance") for k, v in points if k[4] == 16 and k[3] == t])
    out.update(tcd=_mean([v.get("tcd_normalized") for _, v in points]), scd=_mean([v.get("scd_normalized") for v in pairs]),
               delta_act=_mean([v.get("delta_act") for _, v in points]),
               windows=dict(point=len(points), pair=len(pairs), failed=job["failed_rows"]),
               finite=all(_finite(r["values"]) for r in job["rows"].values()))
    training = job["report"].get("training") or {}
    out.update(completed_updates=training.get("completed_updates"),
               stopped_early=(training.get("early_stopping") or {}).get("stopped_early"),
               best_step=(job["report"].get("best") or {}).get("step"))
    return out


def rel(a, b):
    """(a − b)/b；有一个缺就是 None。"""
    if a is None or b is None or b == 0:
        return None
    return (a-b)/b


def lower(a, b):
    return None if a is None or b is None else a < b


def count_true(values):
    return sum(v is True for v in values)


def identity_check(new, old) -> dict:
    """逐行逐值核对两份 identity 评测：键集合相同、每行 values 字典完全相等（JSON 读出的浮点逐位比较）。"""
    if new is None or old is None:
        return dict(identical=False, reason="缺 identity 任务")
    for name, j in (("本次", new), ("v7.1", old)):
        bad = row_problems(j)
        if bad:
            return dict(identical=False, reason=f"{name} identity 的评测行不完整：{'；'.join(bad)}")
    if set(new["expected_keys"]) != set(old["expected_keys"]):
        return dict(identical=False, reason="本次和 v7.1 登记的评测查询不同")
    a, b = new["rows"], old["rows"]
    if set(a) != set(b):
        return dict(identical=False, reason=f"评测行集合不同：本次 {len(a)} 行，v7.1 {len(b)} 行，"
                                           f"只在本次 {len(set(a)-set(b))} 行，只在 v7.1 {len(set(b)-set(a))} 行")
    differ = [k for k in sorted(a, key=str) if a[k]["values"] != b[k]["values"]]
    if differ:
        k = differ[0]
        keys = sorted(set(a[k]["values"]) | set(b[k]["values"]))
        first = {n: (a[k]["values"].get(n), b[k]["values"].get(n)) for n in keys if a[k]["values"].get(n) != b[k]["values"].get(n)}
        return dict(identical=False, reason=f"{len(differ)}/{len(a)} 行的值不同；第一行 {list(k)}：{first}")
    return dict(identical=True, rows=len(a))


# —— C-1 ——

def c1_summary(root) -> dict:
    cells, problems, invalid, queries = {}, [], [], {}
    for folder in job_folders(root):
        j = load_job(folder)
        job = j["job"]
        if job.get("kind") or job["condition"] != "po_jepa_full":
            continue
        variant = variant_of(job)
        key = (variant["temporal_partition"], variant["readout_init"], scene_of(job))
        expected = Path(root)/variant_label(variant)
        if not folder.resolve().is_relative_to(expected.resolve()):
            problems.append(f"{folder}：job.json 的变体是 {variant_label(variant)}，目录不在 {expected} 下")
        if key in cells or any(x["key"] == list(key) for x in invalid):
            problems.append(f"{folder}：{key} 重复")
            continue
        m = metrics(j)
        bad = (job_identity_problems(job, scene=key[2], data=C1_DATA, condition="po_jepa_full") + row_problems(j)
               + metric_problems(m, C1_METRICS))
        if bad:
            invalid.append(dict(key=list(key), folder=str(folder), reasons=bad)); continue
        queries.setdefault(key[2], set()).add(frozenset(j["expected_keys"]))
        cells[key] = dict(folder=str(folder), updates=job["updates"], dataset=job["dataset"], **m)
    for s, sets in queries.items():
        if len(sets) > 1:
            problems.append(f"{s}：四个变体登记的评测查询不同")
    invalid_keys = {tuple(x["key"]) for x in invalid}
    missing = [dict(partition=p, init=i, scene=s) for p in PARTITIONS for i in INITS for s in SCENES
               if (p, i, s) not in cells and (p, i, s) not in invalid_keys]
    m1 = {init: m1_compare(cells, init) for init in INITS}
    decision = None
    if not missing and not invalid and all(len(v) == 1 for v in queries.values()):
        partition = "dyadic" if m1["zero"]["adopt"] else "bisection"
        m2 = m2_compare(cells, partition)
        decision = dict(temporal_partition=partition, readout_init="zero" if m2["adopt"] else "xavier", m1=m1["zero"], m2=m2,
                        command=f"python tools/make_small_validation_jobs.py --stage c2 --temporal-partition {partition} "
                                f"--readout-init {'zero' if m2['adopt'] else 'xavier'} > jobs-sv-c2.txt")
    problems += [f"{x['key']}（{x['folder']}）无效，不进判定：{'；'.join(x['reasons'])}" for x in invalid]
    return dict(stage="c1", rules=RULES, cells=[dict(partition=p, init=i, scene=s, **c) for (p, i, s), c in sorted(cells.items())],
                m1_by_init=m1, decision=decision, missing=missing, invalid=invalid, problems=problems,
                scientific_conclusions_verified=False)


def m1_compare(cells, init) -> dict:
    """dyadic 对 bisection（同一初始化）：每个场景的滚动、直接预测相对变化，和 M1 的采用规则。"""
    rows = []
    for s in SCENES:
        d, b = cells.get(("dyadic", init, s)), cells.get(("bisection", init, s))
        if d is None or b is None:
            rows.append(dict(scene=s, complete=False)); continue
        better = [lower(d[f"rollout16@{t:g}"], b[f"rollout16@{t:g}"]) for t in ROLLOUT_TIMES]
        change = rel(d["direct16"], b["direct16"])
        rows.append(dict(scene=s, complete=True, rollout_better=all(x is True for x in better),
                         rollout_rel={f"{t:g}": rel(d[f"rollout16@{t:g}"], b[f"rollout16@{t:g}"]) for t in ROLLOUT_TIMES},
                         direct_rel=change, direct_ok=change is not None and change <= RULES["direct_tolerance"],
                         tcd_rel=rel(d["tcd"], b["tcd"])))
    complete = all(r["complete"] for r in rows)
    adopt = (complete and count_true(r["rollout_better"] for r in rows) >= RULES["majority"]
             and all(r["direct_ok"] for r in rows))
    return dict(init=init, scenes=rows, complete=complete, adopt=adopt)


def m2_compare(cells, partition) -> dict:
    rows = []
    for s in SCENES:
        z, x = cells[(partition, "zero", s)], cells[(partition, "xavier", s)]
        change = rel(z["direct16"], x["direct16"])
        rows.append(dict(scene=s, direct_rel=change, direct_ok=change is not None and change <= RULES["direct_tolerance"],
                         rollout_rel={f"{t:g}": rel(z[f"rollout16@{t:g}"], x[f"rollout16@{t:g}"]) for t in ROLLOUT_TIMES}))
    return dict(partition=partition, scenes=rows, adopt=count_true(r["direct_ok"] for r in rows) >= RULES["majority"])


# —— C-2 ——

C2_NEW_CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "transformer_consistency", "fixed_fourier", "po_jepa_rollout_sup", "identity")


def c2_summary(root, v7_root, readout=None, v7_readout=None) -> dict:
    problems, invalid, found, variants = [], [], defaultdict(dict), set()
    for folder in job_folders(root):
        j = load_job(folder)
        job = j["job"]
        if job.get("kind"):
            continue
        variants.add(variant_label(variant_of(job)))
        s, c = scene_of(job), job["condition"]
        if c in found[s]:
            problems.append(f"{folder}：{s}/{c} 重复，保留 {found[s][c]['folder']}"); continue
        found[s][c] = j
    if len(variants) > 1:
        problems.append(f"{root} 里混了几个变体：{sorted(variants)}（每个变体一个结果根），不出判定")
    new, scenes, checkpoints = defaultdict(dict), {}, {}
    for s in SCENES:
        for c, j in found[s].items():
            m = metrics(j)
            bad = (job_identity_problems(j["job"], scene=s, data=C2_DATA) + row_problems(j)
                   + (metric_problems(m, TRAINED_METRICS) if c != "identity" else []))
            if bad:
                invalid.append(dict(scene=s, condition=c, folder=j["folder"], reasons=bad)); continue
            new[s][c] = j
        old = {}
        for c in (*REUSED_FROM_V71, V71_BASELINE, "identity"):
            folder = finished_attempt(Path(v7_root)/s/f"seed{SEED}"/c)
            if folder is None:
                continue
            j = load_job(folder)
            bad = job_identity_problems(j["job"], scene=s, data=C2_DATA, condition=c, variant=DEFAULT_VARIANT)
            if not bad and c != "identity":
                bad = row_problems(j) + metric_problems(metrics(j), ("direct16",))
            if bad:
                invalid.append(dict(scene=s, condition=f"v7.1 {c}", folder=str(folder), reasons=bad)); continue
            old[c] = j
        gate = identity_check(new[s].get("identity"), old.get("identity"))
        if gate["identical"]:
            sets = {frozenset(j["expected_keys"]) for j in (*new[s].values(), *old.values())}
            if len(sets) > 1:
                gate = dict(identical=False, reason="各条件登记的评测查询不同")
        m = {c: metrics(j) for c, j in new[s].items()}
        if not gate["identical"]:
            sets = {frozenset(j["expected_keys"]) for j in new[s].values()}
            if len(sets) > 1:
                problems.append(f"{s}：本次各条件登记的评测查询不同")
        v71 = {c: metrics(j) for c, j in old.items()} if gate["identical"] else {}
        updates = {c: new[s][c]["job"]["updates"] for c in new[s]}
        if len(set(updates.values())) > 1:
            problems.append(f"{s}：各条件的训练步数预算不同 {updates}")
        checkpoints[s] = dict(new={c: _selected_sha(j) for c, j in new[s].items()},
                              v71={c: _selected_sha(j) for c, j in old.items()} if gate["identical"] else {})
        scenes[s] = dict(identity_gate=gate, new=m, v71=v71)
    problems += [f"{x['scene']}/{x['condition']}（{x['folder']}）无效，不进判定：{'；'.join(x['reasons'])}" for x in invalid]
    criteria = None
    if len(variants) <= 1:
        criteria = acceptance(scenes, readout_rows(readout) if readout else {}, readout_rows(v7_readout) if v7_readout else {},
                              checkpoints, problems)
    invalid_keys = {(x["scene"], x["condition"]) for x in invalid}
    missing = [dict(scene=s, condition=c) for s in SCENES for c in C2_NEW_CONDITIONS
               if c not in found[s] and (s, c) not in invalid_keys]
    return dict(stage="c2", variant=sorted(variants), rules=RULES, scenes=scenes, criteria=criteria, missing=missing,
                invalid=invalid, problems=problems, scientific_conclusions_verified=False)


def _selected_sha(j):
    return (j["report"].get("best") or {}).get("checkpoint_sha256")


def acceptance(scenes, readout, v7_readout, checkpoints=None, problems=None) -> dict:
    out = {}
    rows2, rows3, rows4tc, rows4ff, rows6 = [], [], [], [], []
    for s in SCENES:
        n = scenes[s]["new"]
        full, nc = n.get("po_jepa_full"), n.get("po_jepa_no_consistency")
        if full and nc:
            change = rel(full["direct16"], nc["direct16"])
            rows2.append(dict(scene=s, rollout8=lower(full["rollout16@8"], nc["rollout16@8"]),
                              rollout12=lower(full["rollout16@12"], nc["rollout16@12"]), tcd=lower(full["tcd"], nc["tcd"]),
                              direct_rel=change, direct_ok=change is not None and change <= RULES["direct_tolerance"]))
            rows2[-1]["pass"] = all(rows2[-1][k] is True for k in ("rollout8", "rollout12", "tcd", "direct_ok"))
            rows3.append(dict(scene=s, grid8_rel=rel(full["grid8"], nc["grid8"]), pass_=lower(full["grid8"], nc["grid8"]) is True))
        tc, ff = n.get("transformer_consistency"), n.get("fixed_fourier")
        if full and tc:
            r = {m: rel(full[m], tc[m]) for m in OOD_METRICS}
            rows4tc.append(dict(scene=s, rel=r, not_worse={m: r[m] is not None and r[m] <= RULES["not_worse_tolerance"] for m in OOD_METRICS}))
        if full and ff:
            wins = {m: lower(full[m], ff[m]) is True for m in OOD_METRICS}
            rows4ff.append(dict(scene=s, rel={m: rel(full[m], ff[m]) for m in OOD_METRICS}, wins=wins,
                                pass_=sum(wins.values()) >= RULES["ff_metric_wins"]))
        v = scenes[s]["v71"]
        tf, old = v.get("matched_transformer"), v.get(V71_BASELINE)
        if full and tf and old:
            gap_new, gap_old = rel(full["direct16"], tf["direct16"]), rel(old["direct16"], tf["direct16"])
            rows6.append(dict(scene=s, gap_new=gap_new, gap_v71=gap_old,
                              pass_=gap_new is not None and gap_old is not None and gap_new <= gap_old))
    out["2_time_consistency"] = dict(scenes=rows2, pass_=len(rows2) == 3 and all(r["pass"] for r in rows2))
    out["3_space_consistency"] = dict(scenes=rows3, pass_=len(rows3) == 3 and all(r["pass_"] for r in rows3))
    per_metric = {m: sum(r["not_worse"][m] for r in rows4tc) for m in OOD_METRICS}
    out["4_structure"] = dict(vs_transformer_consistency=dict(scenes=rows4tc, not_worse_scenes=per_metric,
                                                              pass_=len(rows4tc) == 3 and all(v >= RULES["majority"] for v in per_metric.values())),
                              vs_fixed_fourier=dict(scenes=rows4ff, pass_=len(rows4ff) == 3 and
                                                    sum(r["pass_"] for r in rows4ff) >= RULES["majority"]))
    out["4_structure"]["pass_"] = out["4_structure"]["vs_transformer_consistency"]["pass_"] and out["4_structure"]["vs_fixed_fourier"]["pass_"]
    out["5_physics"] = physics(scenes, readout, v7_readout, checkpoints or {}, problems if problems is not None else [])
    out["6_zero_init"] = dict(scenes=rows6, pass_=len(rows6) == 3 and sum(r["pass_"] for r in rows6) >= RULES["majority"],
                              note="TF = v7.1 的 matched_transformer；v7.1 的 full 是 Xavier + 二等分")
    return out


def physics(scenes, readout, v7_readout, checkpoints, problems) -> dict:
    """第 5 条：本次 R 段（readout）和 b-v7.1（v7_readout）的读出、反事实。读出先核来源（文件头“准入”），不一致的不进比较。"""
    vs_nc, vs_tf, cf = [], [], []
    for s in SCENES:
        gate = scenes[s]["identity_gate"]["identical"]
        ck = checkpoints.get(s) or dict(new={}, v71={})
        full = _readout_entry(readout, s, "po_jepa_full", ck["new"], "本次", problems)
        nc = _readout_entry(readout, s, "po_jepa_no_consistency", ck["new"], "本次", problems)
        tf = _readout_entry(v7_readout, s, "matched_transformer", ck["v71"], "v7.1", problems) if gate else None
        old = _readout_entry(v7_readout, s, V71_BASELINE, ck["v71"], "v7.1", problems) if gate else None
        for kind, group in (("rollout", (full, nc, tf)), ("counterfactual", (full, old))):
            present = [g[kind] for g in group if g and g.get(kind)]
            for field in ("probe_weight_sha256", "truth_sha256", "train_dataset") + (("horizon", "curve_times") if kind == "rollout" else ()):
                if len({json.dumps(x.get(field)) for x in present}) > 1:
                    problems.append(f"{s}：比较的 {kind} 读出 {field} 不一致，第 5 条这一项不比")
                    for g in group:
                        if g:
                            g.pop(kind, None)
                    break
        if full and full.get("rollout") and nc and nc.get("rollout"):
            a, b = full["rollout"], nc["rollout"]
            vs_nc.append(dict(scene=s, nmse_rel=rel(a["nmse_mean"], b["nmse_mean"]), vpt=(a[VPT_KEY], b[VPT_KEY]),
                              pass_=lower(a["nmse_mean"], b["nmse_mean"]) is True and a[VPT_KEY] >= b[VPT_KEY]))
        if full and full.get("rollout") and tf and tf.get("rollout"):
            r = rel(full["rollout"]["nmse_mean"], tf["rollout"]["nmse_mean"])
            vs_tf.append(dict(scene=s, nmse_rel=r, pass_=r is not None and r <= RULES["not_worse_tolerance"]))
        if full and full.get("counterfactual") and old and old.get("counterfactual"):
            a, b = full["counterfactual"], old["counterfactual"]
            cf.append(dict(scene=s, cos_mean=a.get("cos_mean"), gain=a.get("gain"), gain_v71=b.get("gain"),
                           pass_=(a.get("cos_mean") or 0) > 0 and a.get("gain") is not None and b.get("gain") is not None
                           and a["gain"] > b["gain"]))
    return dict(vs_no_consistency=dict(scenes=vs_nc, pass_=len(vs_nc) == 3 and sum(r["pass_"] for r in vs_nc) >= RULES["majority"]),
                vs_matched_transformer=dict(scenes=vs_tf, pass_=len(vs_tf) == 3 and sum(r["pass_"] for r in vs_tf) >= RULES["majority"]),
                counterfactual=dict(scenes=cf, pass_=len(cf) == 3 and all(r["pass_"] for r in cf)),
                pass_=(len(vs_nc) == 3 and len(vs_tf) == 3 and len(cf) == 3
                       and sum(r["pass_"] for r in vs_nc) >= RULES["majority"] and sum(r["pass_"] for r in vs_tf) >= RULES["majority"]
                       and all(r["pass_"] for r in cf)))


def _readout_entry(rows, scene, condition, selected, label, problems):
    """取一份读出，核它的训练数据集、horizon 和检查点（等于对应训练任务选中的那个）；不对就记问题并丢掉这一项。"""
    entry = rows.get((scene, SEED, condition))
    if not entry:
        return None
    entry = {k: dict(v) for k, v in entry.items()}
    for kind in ("rollout", "counterfactual"):
        r = entry.get(kind)
        if not r:
            continue
        bad = []
        if r.get("train_dataset") != f"{scene}-{C2_DATA['n']}":
            bad.append(f"训练数据集 {r.get('train_dataset')}")
        if kind == "rollout" and r.get("horizon") != LONG_HORIZON:
            bad.append(f"horizon {r.get('horizon')}")
        if r.get("updates") != C2_DATA["updates"]:
            bad.append(f"步数预算 {r.get('updates')}")
        if condition not in selected:
            bad.append("没有对应的有效训练任务")
        elif selected[condition] is None or r.get("checkpoint_sha256") != selected[condition]:
            bad.append("检查点不是训练任务选中的那个")
        if bad:
            problems.append(f"{scene}/{label} {condition} 的 {kind} 读出来源不对（{'，'.join(bad)}），不进第 5 条")
            entry.pop(kind)
    return entry


# —— B 层读出 ——

def readout_rows(root) -> dict:
    """{(场景, 种子, 条件): dict(rollout=…, counterfactual=…, modal=…)}，读 <root>/{rollout,counterfactual,modal}/<场景>/seed<种子>/<条件>.json。"""
    out = defaultdict(dict)
    root = Path(root)
    for kind in ("rollout", "counterfactual", "modal"):
        for path in sorted((root/kind).glob("*/seed*/*.json")):
            r = json.loads(path.read_text(encoding="utf-8"))
            key = (path.parent.parent.name, int(path.parent.name[4:]), path.stem)
            expected = dict(rollout="state_readout_rollout", counterfactual="state_readout_counterfactual", modal="modal_spectrum")[kind]
            if r.get("kind") != expected or r.get("condition") != key[2] or r.get("seed") != key[1]:
                raise SystemExit(f"{path} 不是 {key} 的 {expected} 结果")
            source = dict(train_dataset=r.get("train_dataset"), updates=r.get("updates"),
                          checkpoint_sha256=(r.get("best") or {}).get("checkpoint_sha256"),
                          probe_weight_sha256=(r.get("probe") or {}).get("weight_sha256"), truth_sha256=r.get("truth_sha256"))
            if kind == "rollout":
                curve = [v for k, v in r["means"].items() if k.startswith("nmse:")]
                floor = [v for k, v in r["means"].items() if k.startswith("floor_nmse:")]
                out[key]["rollout"] = dict({k: v for k, v in r["means"].items() if "@" in k}, nmse_mean=_mean(curve),
                                           floor_nmse_mean=_mean(floor), probe_cv_r2=r["probe"]["cv_r2"],
                                           trajectories=r["counts"]["trajectories"], cross_version="cross_version" in r,
                                           horizon=r.get("horizon"), curve_times=r.get("curve_times"), **source)
            elif kind == "counterfactual":
                out[key]["counterfactual"] = dict(r["overall"], cross_version="cross_version" in r, **source)
            else:
                c = r["comparison"]
                main = "near_true" if "near_true" in c else "non_oscillatory"
                out[key]["modal"] = dict(statistic=main, value=c[main], null=c["null_"+main], init=r["init_reference"][main],
                                         decay_within_true=c.get("decay_within_true"),
                                         frequency_quantiles=c["learned"]["frequency_quantiles"],
                                         decay_quantiles=c["learned"]["decay_quantiles"])
    return dict(out)


def readout_summary(root, actions=None) -> dict:
    rows = readout_rows(root)
    problems = []
    for s in sorted({k[0] for k in rows}):
        for kind, fields in (("rollout", ("probe_weight_sha256", "truth_sha256", "horizon", "curve_times")),
                             ("counterfactual", ("probe_weight_sha256", "truth_sha256"))):
            present = [v[kind] for k, v in rows.items() if k[0] == s and kind in v]
            problems += [f"{s}：{kind} 读出的 {f} 不一致（同一场景应当用同一探针、同一真值）"
                         for f in fields if len({json.dumps(x.get(f)) for x in present}) > 1]
    across = []
    for s, c in sorted({(k[0], k[2]) for k in rows}):
        seeds = sorted(k[1] for k in rows if k[0] == s and k[2] == c)
        entry = dict(scene=s, condition=c, seeds=seeds)
        for kind, keys in (("rollout", ("nmse_mean", "floor_nmse_mean", "vpt@0.05", VPT_KEY, "vpt@0.25", "floor_vpt@0.1")),
                           ("counterfactual", ("cf_true", "cf_pred", "explained", "gain", "cos_mean")),
                           ("modal", ("value", "null", "init"))):
            values = [rows[(s, seed, c)][kind] for seed in seeds if kind in rows[(s, seed, c)]]
            if values:
                entry[kind] = {k: _mean([v.get(k) for v in values]) for k in keys}
                if kind == "modal":
                    entry[kind]["statistic"] = values[0]["statistic"]
                entry[kind]["seeds"] = len(values)
                if len(values) > 1:
                    first = keys[0] if kind != "rollout" else "nmse_mean"
                    xs = [v.get(first) for v in values if v.get(first) is not None]
                    entry[kind][f"{first}_seed_std"] = statistics.stdev(xs) if len(xs) > 1 else None
        across.append(entry)
    probe_actions = {}
    if actions:
        for path in sorted(Path(actions).glob("probe-actions-*.json")):
            r = json.loads(path.read_text(encoding="utf-8"))
            probe_actions[r["dataset"]] = dict(overall=r["overall"], by_duration=r["by_duration"])
    return dict(stage="readout", rows=[dict(scene=k[0], seed=k[1], condition=k[2], **v) for k, v in sorted(rows.items())],
                across_seeds=across, probe_actions=probe_actions, problems=problems, scientific_conclusions_verified=False)


# —— 输出 ——

# —— C-3：空间基底数 K 消融 + 方案 B（方案 small-validation-results/c3-plan/2026-11-22_mode-ablation-plan.md §5）——
#
# 指标（测试集、单种子、和 C-2 同口径）：直接预测比 identity 好多少 = 1 − d(条件)/d(identity)，d 是 future_jepa_distance
# 在 16×16、时长 T 的测试点上的均值，T = 1, 2, 4, 8；换网格 = 8×8、12×12 上全部时长的同一比值；长滚动 = rollout16@12。
# 判据 H1–H6 照方案 §5 的表写死（阈值见 C3_RULES）；方案没写场景数的 H4②，这里和 H4① 一样按 ≥ 2/3 个场景算（Claude 的读法）。
# K16（现模型）、K16 的 no_consistency 和 transformer_consistency 取 C-2 的 dyadic-zero；先核这里第一批的 identity 和 C-2 的逐位相同、
# 各条件登记的评测查询相同，不同就不拿 C-2 比。

C3_BASE = dict(temporal_partition="dyadic", readout_init="zero")
C3_SPACE = dict(K16c=dict(complement="shared"), Kfull=dict(modes="full"),
                K32c=dict(modes="32", complement="shared"), K64c=dict(modes="64", complement="shared"))
C3_CONDITIONS = dict(K16c=("po_jepa_full", "po_jepa_no_consistency"), Kfull=("po_jepa_full", "po_jepa_no_consistency"),
                     K32c=("po_jepa_full",), K64c=("po_jepa_full",))
C3_IDENTITY_VARIANT = "K16c"
C3_LOSSLESS = ("K16c", "Kfull")
# Jerry 2026-11-22：四个变体都不丢信息，只改显式基底个数；H6 看这条曲线（K16 截断版只作 H1、H5 的“丢信息”对照）
C3_CURVE = ("K16c", "K32c", "K64c", "Kfull")
C3_TIMES = (1., 2., 4., 8.)
C3_RULES = dict(h1_gain=.03, h1_scenes=("double_well", "van_der_pol"), h2_tie=.01, h3_tolerance=.01, h3_inconclusive=.02,
                h4_rollout=.05, h5_tolerance=.01, h6_noise=.005, majority=2)
C2_REFERENCE = dict(K16="po_jepa_full", **{"K16/no_cons": "po_jepa_no_consistency"}, **{"TF+cons": "transformer_consistency"})


def _point_mean(job, grid, duration=None):
    values = [r["values"].get("future_jepa_distance") for k, r in job["rows"].items()
              if k[0] == "point" and k[4] == grid and (duration is None or k[3] == duration)]
    return _mean(values)


def _gain(job, identity, grid, duration=None):
    """1 − d(条件)/d(identity)；有一个缺就是 None。"""
    a, b = _point_mean(job, grid, duration), _point_mean(identity, grid, duration)
    return None if a is None or not b else 1-a/b


def c3_label(variant, condition) -> str:
    return variant if condition == "po_jepa_full" else f"{variant}/no_cons"


def c3_summary(root, c2_root, readout=True) -> dict:
    problems, invalid, scenes = [], [], {}
    for s in SCENES:
        jobs = {}
        for v, conditions in C3_CONDITIONS.items():
            vroot = Path(root)/f"{variant_label(C3_BASE)}-{v}"
            for c in (*conditions, *(("identity",) if v == C3_IDENTITY_VARIANT else ())):
                folder = finished_attempt(vroot/s/f"seed{SEED}"/c)
                if folder is None:
                    continue
                j = load_job(folder)
                bad = job_identity_problems(j["job"], scene=s, data=C2_DATA, condition=c, variant=dict(C3_BASE, **C3_SPACE[v]))
                bad += row_problems(j) + (metric_problems(metrics(j), TRAINED_METRICS) if c != "identity" else [])
                if bad:
                    invalid.append(dict(scene=s, condition=f"{v} {c}", folder=str(folder), reasons=bad)); continue
                jobs["identity" if c == "identity" else c3_label(v, c)] = j
        old = {}
        for label, c in (*C2_REFERENCE.items(), ("identity", "identity")):
            folder = finished_attempt(Path(c2_root)/s/f"seed{SEED}"/c)
            if folder is None:
                continue
            j = load_job(folder)
            bad = job_identity_problems(j["job"], scene=s, data=C2_DATA, condition=c, variant=C3_BASE) + row_problems(j)
            if bad:
                invalid.append(dict(scene=s, condition=f"C-2 {c}", folder=str(folder), reasons=bad)); continue
            old[label] = j
        gate = identity_check(jobs.get("identity"), old.get("identity"))
        if gate.get("reason"):
            gate["reason"] = gate["reason"].replace("v7.1", "C-2")
        if gate["identical"] and len({frozenset(j["expected_keys"]) for j in (*jobs.values(), *old.values())}) > 1:
            gate = dict(identical=False, reason="各条件登记的评测查询不同")
        identity = old.get("identity") if gate["identical"] else jobs.get("identity")
        table = {k: j for k, j in jobs.items() if k != "identity"}
        if gate["identical"]:
            table.update({k: j for k, j in old.items() if k != "identity"})
        elif len({frozenset(j["expected_keys"]) for j in jobs.values()}) > 1:
            problems.append(f"{s}：本次各条件登记的评测查询不同")
        rows = {}
        for label, j in table.items():
            m = metrics(j)
            rows[label] = dict(direct={f"{t:g}": _gain(j, identity, 16, t) if identity else None for t in C3_TIMES},
                               direct16=_gain(j, identity, 16) if identity else None,
                               grid12=_gain(j, identity, 12) if identity else None, grid8=_gain(j, identity, 8) if identity else None,
                               rollout12=m["rollout16@12"], finite=m["finite"], completed_updates=m["completed_updates"],
                               stopped_early=m["stopped_early"], checkpoint_sha256=_selected_sha(j),
                               source="C-2" if label in C2_REFERENCE else "C-3")
        scenes[s] = dict(identity_gate=gate, rows=rows)
    problems += [f"{x['scene']}/{x['condition']}（{x['folder']}）无效，不进判定：{'；'.join(x['reasons'])}" for x in invalid]
    invalid_keys = {(x["scene"], x["condition"]) for x in invalid}
    missing = [dict(scene=s, variant=v, condition=c) for s in SCENES for v, cs in C3_CONDITIONS.items() for c in cs
               if c3_label(v, c) not in scenes[s]["rows"] and (s, f"{v} {c}") not in invalid_keys]
    hyp = c3_hypotheses(scenes)
    reads = c3_readout(root, c2_root, scenes, problems) if readout else []
    return dict(stage="c3", base_variant=variant_label(C3_BASE), rules=C3_RULES, scenes=scenes, hypotheses=hyp, readout=reads,
                missing=missing, invalid=invalid, problems=problems, single_seed=True, scientific_conclusions_verified=False)


def _get(scenes, s, label, *path):
    x = scenes[s]["rows"].get(label)
    for key in path:
        if x is None:
            return None
        x = x.get(key)
    return x


def c3_hypotheses(scenes) -> dict:
    r = C3_RULES
    out = {}
    # H1：截断是不是实际瓶颈——两个场景上，K 取满或 B16 的 T=8 直接预测比 K16 至少高 3 个百分点
    h1 = []
    for s in r["h1_scenes"]:
        base = _get(scenes, s, "K16", "direct", "8")
        gains = {v: _get(scenes, s, v, "direct", "8") for v in C3_LOSSLESS}
        diff = {v: None if g is None or base is None else g-base for v, g in gains.items()}
        known = [d for d in diff.values() if d is not None]
        h1.append(dict(scene=s, k16=base, gains=gains, minus_k16=diff,
                       pass_=None if not known else max(known) >= r["h1_gain"]))
    out["H1"] = dict(scenes=h1, pass_=None if any(x["pass_"] is None for x in h1) else all(x["pass_"] for x in h1))
    # H2：B16 − K 取满（不是通过/不通过，决定主候选）
    h2 = []
    for s in SCENES:
        d = {f"{t:g}": None for t in C3_TIMES}
        for t in d:
            a, b = _get(scenes, s, "K16c", "direct", t), _get(scenes, s, "Kfull", "direct", t)
            d[t] = None if a is None or b is None else a-b
        h2.append(dict(scene=s, b16_minus_kfull=d))
    t8 = [x["b16_minus_kfull"]["8"] for x in h2 if x["scene"] in r["h1_scenes"]]
    if any(v is None for v in t8) or not t8:
        verdict, candidate = None, None
    else:
        mean = sum(t8)/len(t8)
        verdict = "持平" if abs(mean) <= r["h2_tie"] else ("B16 更好" if mean > 0 else "K 取满更好")
        candidate = "Kfull" if verdict == "K 取满更好" else "K16c"
    out["H2"] = dict(scenes=h2, mean_t8=None if verdict is None else sum(t8)/len(t8), verdict=verdict, candidate=candidate)
    # H3：最好的不丢信息变体（两个场景 T=8 的均值更高的那个，事先定、不逐格挑）≥ TF − 1 个百分点，两个场景、四个时长
    best = None
    means = {v: [_get(scenes, s, v, "direct", "8") for s in r["h1_scenes"]] for v in C3_LOSSLESS}
    if all(x is not None for xs in means.values() for x in xs):
        best = max(C3_LOSSLESS, key=lambda v: sum(means[v]))
    h3 = []
    for s in r["h1_scenes"]:
        cells = {}
        for t in (f"{t:g}" for t in C3_TIMES):
            a, tf = (_get(scenes, s, best, "direct", t) if best else None), _get(scenes, s, "TF+cons", "direct", t)
            gap = None if a is None or tf is None else a-tf
            cells[t] = dict(gap=gap, ok=None if gap is None else gap >= -r["h3_tolerance"],
                            inconclusive=None if gap is None else abs(gap) <= r["h3_inconclusive"])
        h3.append(dict(scene=s, cells=cells))
    oks = [c["ok"] for x in h3 for c in x["cells"].values()]
    out["H3"] = dict(variant=best, scenes=h3, pass_=None if None in oks or not oks else all(oks),
                     inconclusive=any(c["inconclusive"] for x in h3 for c in x["cells"].values() if c["inconclusive"] is not None),
                     note="差距在 ±2 个百分点以内的格子单种子不下结论（方案 §5：加种子 1704、1705 要 Jerry 点头）")
    # H4：①B16、K 取满上 full 的 12 周期滚动比 no_cons 低至少 5%，≥ 2/3 场景；② full 的滚动不高于 TF+一致性（≥ 2/3 场景）
    h4 = {}
    for v in C3_LOSSLESS:
        rows = []
        for s in SCENES:
            full, nc, tf = (_get(scenes, s, v, "rollout12"), _get(scenes, s, f"{v}/no_cons", "rollout12"),
                            _get(scenes, s, "TF+cons", "rollout12"))
            rows.append(dict(scene=s, full=full, no_cons=nc, tf=tf, rel_no_cons=rel(full, nc),
                             one=None if full is None or nc is None else full <= (1-r["h4_rollout"])*nc,
                             two=None if full is None or tf is None else full <= tf))
        one, two = [x["one"] for x in rows], [x["two"] for x in rows]
        h4[v] = dict(scenes=rows, one=None if None in one else count_true(one) >= r["majority"],
                     two=None if None in two else count_true(two) >= r["majority"])
    out["H4"] = h4
    # H5：advection_diffusion 直接预测、各场景 8×8 和 12×12 不比 K16 差 1 个百分点以上；有限
    h5 = []
    for v in C3_SPACE:
        for s in SCENES:
            checks_ = {}
            for metric in (("direct16",) if s == "advection_diffusion" else ()) + ("grid12", "grid8"):
                a, b = _get(scenes, s, v, metric), _get(scenes, s, "K16", metric)
                checks_[metric] = dict(minus_k16=None if a is None or b is None else a-b,
                                       ok=None if a is None or b is None else a-b >= -r["h5_tolerance"])
            finite = _get(scenes, s, v, "finite")
            if finite is None and not any(c["ok"] is not None for c in checks_.values()):
                continue
            h5.append(dict(variant=v, scene=s, checks=checks_, finite=finite))
    flags = [c["ok"] for x in h5 for c in x["checks"].values()] + [x["finite"] for x in h5]
    out["H5"] = dict(rows=h5, pass_=None if None in flags or not flags else all(flags))
    # H6：K16c → K32c → K64c → K 取满（T=8，都不丢信息，只改显式基底个数）单调不降，允许 0.5 个百分点噪声
    h6 = []
    for s in SCENES:
        curve = [(v, _get(scenes, s, v, "direct", "8")) for v in C3_CURVE]
        known = [g for _, g in curve if g is not None]
        mono = None if len(known) < len(C3_CURVE) else all(b >= a-r["h6_noise"] for a, b in zip(known, known[1:]))
        h6.append(dict(scene=s, curve=dict(curve), monotone=mono))
    out["H6"] = dict(scenes=h6, pass_=None if any(x["monotone"] is None for x in h6) else all(x["monotone"] for x in h6))
    return out


def c3_readout(root, c2_root, scenes, problems) -> list:
    """只报告、不设门槛：每个变体上 full 对 no_cons 的长时程读出和反事实（同一探针、同一真值；检查点要是训练任务选中的那个）。"""
    out = []
    sources = {v: Path(root)/f"{variant_label(C3_BASE)}-{v}"/"readout" for v in C3_SPACE}
    sources["K16"] = Path(c2_root)/"readout"
    for v, path in sources.items():
        if not path.exists():
            continue
        rows = readout_rows(path)
        for s in SCENES:
            labels = {"po_jepa_full": v, "po_jepa_no_consistency": f"{v}/no_cons"}
            selected = {c: _get(scenes, s, label, "checkpoint_sha256") for c, label in labels.items()
                        if label in scenes[s]["rows"]}
            entry = {c: _readout_entry(rows, s, c, selected, f"C-3 {v}", problems) for c in labels if c in selected}
            if not any(entry.values()):
                continue
            item = dict(variant=v, scene=s)
            for c, e in entry.items():
                if e:
                    item[c] = dict(nmse_mean=(e.get("rollout") or {}).get("nmse_mean"), vpt=(e.get("rollout") or {}).get(VPT_KEY),
                                   gain=(e.get("counterfactual") or {}).get("gain"), cos_mean=(e.get("counterfactual") or {}).get("cos_mean"))
            out.append(item)
    return out



def _f(x, digits=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "是" if x else "否"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def _pct(x):
    return "—" if x is None else f"{100*x:+.2f}%"


def _verdict(flag):
    return "**通过**" if flag else "**未通过**"


def markdown(summary) -> str:
    lines = [f"# 小规模验证汇总：{summary['stage']}", "",
             "只读评测行和读出结果；判定阈值是 Claude 提的（方案 §5.4，Jerry 可改，见 RULES）。科学结论未复核。", ""]
    if summary["stage"] == "c1":
        lines += ["## 2×2 单元（po_jepa_full，种子 1703）", "",
                  "| 切分 | 初始化 | 场景 | direct16 | heldout16 | grid12 | grid8 | rollout16@8 | rollout16@12 | tcd | scd | 步数 | 早停 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in summary["cells"]:
            lines.append(f"| {c['partition']} | {c['init']} | {c['scene']} | " + " | ".join(
                _f(c[k]) for k in ("direct16", "heldout16", "grid12", "grid8", "rollout16@8", "rollout16@12", "tcd", "scd",
                                   "completed_updates", "stopped_early")) + " |")
        for init, m1 in summary["m1_by_init"].items():
            lines += ["", f"## M1 多段切分 vs 二等分（初始化 {init}{'，进判定' if init == 'zero' else '，只报告'}）", "",
                      "| 场景 | rollout16@8 相对 | rollout16@12 相对 | 两个滚动都更好 | direct16 相对 | direct16 ≤ +0.5% | tcd 相对 |",
                      "|---|---|---|---|---|---|---|"]
            for r in m1["scenes"]:
                if not r["complete"]:
                    lines.append(f"| {r['scene']} | 缺 | | | | | |"); continue
                lines.append(f"| {r['scene']} | {_pct(r['rollout_rel']['8'])} | {_pct(r['rollout_rel']['12'])} | {_f(r['rollout_better'])} | "
                             f"{_pct(r['direct_rel'])} | {_f(r['direct_ok'])} | {_pct(r['tcd_rel'])} |")
            lines.append(f"\n采用多段切分：{_f(m1['adopt'])}")
        d = summary["decision"]
        lines += ["", "## 推荐的 C-2 变体", ""]
        if d is None:
            lines.append("缺格或有无效格，不给推荐：" + ", ".join(
                [f"{m['partition']}-{m['init']}/{m['scene']} 缺" for m in summary["missing"]]
                + ["-".join(x["key"][:2]) + "/" + x["key"][2] + " 无效" for x in summary.get("invalid", [])]) + "（见下面的问题）")
        else:
            lines += [f"- M2 零初始化（在 {d['m2']['partition']} 下）：" + "；".join(
                          f"{r['scene']} direct16 {_pct(r['direct_rel'])}" for r in d["m2"]["scenes"]) + f" → 采用：{_f(d['m2']['adopt'])}",
                      f"- 推荐：**{d['temporal_partition']}-{d['readout_init']}**", "", f"```\n{d['command']}\n```"]
    elif summary["stage"] == "c2":
        lines += [f"变体：{', '.join(summary['variant']) or '—'}", "", "## 前提：identity 与 v7.1 逐位相同", ""]
        for s, x in summary["scenes"].items():
            g = x["identity_gate"]
            lines.append(f"- {s}：{'相同（' + str(g['rows']) + ' 行），复用 v7.1 的 TF、AC 和 full 底数' if g['identical'] else '不同：' + g['reason'] + '；v7.1 的对照记不可比'}")
        lines += ["", "## 各条件指标（种子 1703；v7.1 的复用条件标 v7.1）", "",
                  "| 场景 | 条件 | direct16 | heldout16 | grid12 | grid8 | rollout16@8 | rollout16@12 | tcd | scd | delta_act | 步数 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s, x in summary["scenes"].items():
            for src, table in (("", x["new"]), ("v7.1 ", x["v71"])):
                for c, m in table.items():
                    lines.append(f"| {s} | {src}{c} | " + " | ".join(_f(m[k]) for k in (
                        "direct16", "heldout16", "grid12", "grid8", "rollout16@8", "rollout16@12", "tcd", "scd", "delta_act",
                        "completed_updates")) + " |")
        c = summary["criteria"]
        if c is None:
            lines += ["", "## 判定", "", "不出判定（见下面的问题）。"]
            c = {}
    if summary["stage"] == "c2" and summary["criteria"] is not None:
        t = c["2_time_consistency"]
        lines += ["", f"## 第 2 条 时间一致性起作用：{_verdict(t['pass_'])}", "",
                  "| 场景 | rollout@8 更好 | rollout@12 更好 | tcd 更低 | direct16 相对 | 通过 |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['scene']} | {_f(r['rollout8'])} | {_f(r['rollout12'])} | {_f(r['tcd'])} | {_pct(r['direct_rel'])} | {_f(r['pass'])} |"
                  for r in t["scenes"]]
        sp = c["3_space_consistency"]
        lines += ["", f"## 第 3 条 空间一致性保持：{_verdict(sp['pass_'])}", ""]
        lines += [f"- {r['scene']}：grid8 full 对 no-cons {_pct(r['grid8_rel'])}" for r in sp["scenes"]]
        st = c["4_structure"]
        lines += ["", f"## 第 4 条 结构有必要：{_verdict(st['pass_'])}", "",
                  f"对 transformer_consistency（不输 = ≤ +0.5%）：{_verdict(st['vs_transformer_consistency']['pass_'])}", "",
                  "| 场景 | " + " | ".join(OOD_METRICS) + " |", "|---|" + "---|"*len(OOD_METRICS)]
        lines += [f"| {r['scene']} | " + " | ".join(_pct(r["rel"][m]) for m in OOD_METRICS) + " |" for r in st["vs_transformer_consistency"]["scenes"]]
        lines += ["", f"对 fixed_fourier（一个场景 ≥ 3/4 个指标更低算胜）：{_verdict(st['vs_fixed_fourier']['pass_'])}", "",
                  "| 场景 | " + " | ".join(OOD_METRICS) + " | 胜 |", "|---|" + "---|"*(len(OOD_METRICS)+1)]
        lines += [f"| {r['scene']} | " + " | ".join(_pct(r["rel"][m]) for m in OOD_METRICS) + f" | {_f(r['pass_'])} |"
                  for r in st["vs_fixed_fourier"]["scenes"]]
        ph = c["5_physics"]
        lines += ["", f"## 第 5 条 学到物理：{_verdict(ph['pass_'])}", ""]
        lines += [f"- 对 no-cons：{r['scene']} nmse {_pct(r['nmse_rel'])}，VPT@0.1 {r['vpt'][0]:g} vs {r['vpt'][1]:g}，{_f(r['pass_'])}"
                  for r in ph["vs_no_consistency"]["scenes"]]
        lines += [f"- 对 v7.1 TF：{r['scene']} nmse {_pct(r['nmse_rel'])}，{_f(r['pass_'])}" for r in ph["vs_matched_transformer"]["scenes"]]
        lines += [f"- 反事实：{r['scene']} cos_mean {_f(r['cos_mean'])}，gain {_f(r['gain'])}（v7.1 full {_f(r['gain_v71'])}），{_f(r['pass_'])}"
                  for r in ph["counterfactual"]["scenes"]]
        z = c["6_zero_init"]
        lines += ["", f"## 第 6 条 零初始化不伤：{_verdict(z['pass_'])}", "", f"（{z['note']}）", ""]
        lines += [f"- {r['scene']}：(full − TF)/TF 本次 {_pct(r['gap_new'])}，v7.1 {_pct(r['gap_v71'])}，{_f(r['pass_'])}" for r in z["scenes"]]
        lines += ["", "第 1 条（理论一致）见 A 层测试；第 7 条（工程）看下面的问题、缺失和各任务的早停记录。"]
    if summary["stage"] == "readout":
        lines += ["## 长时程读出与反事实（种子平均）", "",
                  "| 场景 | 条件 | 种子 | nmse 均值 | 探针下限 | VPT@0.1 | 探针 VPT@0.1 | 反事实 ‖Δ‖ | ‖Δ̂‖ | explained | gain | cos |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in summary["across_seeds"]:
            r, cf = e.get("rollout", {}), e.get("counterfactual", {})
            lines.append(f"| {e['scene']} | {e['condition']} | {len(e['seeds'])} | {_f(r.get('nmse_mean'))} | {_f(r.get('floor_nmse_mean'))} | "
                         f"{_f(r.get(VPT_KEY))} | {_f(r.get('floor_vpt@0.1'))} | {_f(cf.get('cf_true'))} | {_f(cf.get('cf_pred'))} | "
                         f"{_f(cf.get('explained'))} | {_f(cf.get('gain'))} | {_f(cf.get('cos_mean'))} |")
        modal = [e for e in summary["across_seeds"] if "modal" in e]
        if modal:
            lines += ["", "## 模态频率对照（探索性）", "", "| 场景 | 条件 | 统计量 | 学到的 | 均匀分布 | 未训练模型 |", "|---|---|---|---|---|---|"]
            for e in modal:
                lines.append(f"| {e['scene']} | {e['condition']} | {e['modal']['statistic']} | {_f(e['modal']['value'])} | {_f(e['modal']['null'])} | {_f(e['modal']['init'])} |")
        if summary["probe_actions"]:
            lines += ["", "## 动作探针（换动作后真实终点的变化，和模型无关）", "", "| 数据集 | identity | 反事实可分性 | 画面位移 | 换动作位移 | 可分性/identity |",
                      "|---|---|---|---|---|---|"]
            for name, p in summary["probe_actions"].items():
                o = p["overall"]
                lines.append(f"| {name} | {_f(o['identity'])} | {_f(o['oracle_delta_act'])} | {_f(o['motion_px'])} | {_f(o['action_swap_px'])} | "
                             f"{_f(o['oracle_over_identity'])} |")
    if summary["stage"] == "c3":
        lines += c3_markdown(summary)
    if summary.get("missing"):
        lines += ["", "## 缺失", ""] + [f"- {m}" for m in summary["missing"]]
    if summary.get("problems"):
        lines += ["", "## 问题", ""] + [f"- {p}" for p in summary["problems"]]
    return "\n".join(lines)+"\n"


def _pp(x):
    """百分点（gain 已是比例）：+1.23 pp。"""
    return "—" if x is None else f"{100*x:+.2f} pp"


def c3_markdown(summary) -> list:
    lines = [f"底变体：{summary['base_variant']}（C-2 定下的）；单种子 1703，未复核。指标是“比 identity 好多少”（1 − d/d_identity）。", "",
             "## 前提：identity 与 C-2 逐位相同", ""]
    for s, x in summary["scenes"].items():
        g = x["identity_gate"]
        lines.append(f"- {s}：{'相同（' + str(g['rows']) + ' 行），K16、TF+一致性取 C-2' if g['identical'] else '不同：' + g['reason'] + '；C-2 的参照记不可比'}")
    lines += ["", "## 直接预测与长滚动", "",
              "| 场景 | 模型 | 来源 | T=1 | T=2 | T=4 | T=8 | 16×16 全部 | 12×12 | 8×8 | rollout16@12 | 步数 | 早停 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, x in summary["scenes"].items():
        for label, r in x["rows"].items():
            lines.append(f"| {s} | {label} | {r['source']} | " + " | ".join(_pct(r["direct"][f"{t:g}"]) for t in C3_TIMES)
                         + f" | {_pct(r['direct16'])} | {_pct(r['grid12'])} | {_pct(r['grid8'])} | {_f(r['rollout12'])} | "
                           f"{_f(r['completed_updates'])} | {_f(r['stopped_early'])} |")
    h = summary["hypotheses"]
    def v(flag):
        return "**缺数据**" if flag is None else _verdict(flag)
    lines += ["", f"## H1 截断是不是实际瓶颈（T=8 比 K16 高 ≥ 3 pp）：{v(h['H1']['pass_'])}", ""]
    lines += [f"- {x['scene']}：K16 {_pct(x['k16'])}；" + "，".join(f"{k} {_pp(d)}" for k, d in x["minus_k16"].items()) for x in h["H1"]["scenes"]]
    lines += ["", f"## H2 B16 − K 取满：{h['H2']['verdict'] or '缺数据'}（两场景 T=8 均值 {_pp(h['H2']['mean_t8'])}；主候选 {h['H2']['candidate'] or '—'}）", ""]
    lines += [f"- {x['scene']}：" + "，".join(f"T={t} {_pp(d)}" for t, d in x["b16_minus_kfull"].items()) for x in h["H2"]["scenes"]]
    lines += ["", f"## H3 追不追得上 TF+一致性（{h['H3']['variant'] or '—'} ≥ TF − 1 pp）：{v(h['H3']['pass_'])}", "", f"（{h['H3']['note']}）", ""]
    lines += [f"- {x['scene']}：" + "，".join(f"T={t} {_pp(c['gap'])}" + ("（单种子不下结论）" if c["inconclusive"] else "")
                                         for t, c in x["cells"].items()) for x in h["H3"]["scenes"]]
    lines += ["", "## H4 一致性压长程累积误差（① 比 no_cons 低 ≥ 5%，≥ 2/3 场景；② 不高于 TF+一致性，≥ 2/3 场景）", ""]
    for name, x in h["H4"].items():
        lines.append(f"- {name}：① {v(x['one'])}，② {v(x['two'])}；" + "；".join(
            f"{r['scene']} full {_f(r['full'])} / no_cons {_f(r['no_cons'])}（{_pct(r['rel_no_cons'])}）/ TF {_f(r['tf'])}" for r in x["scenes"]))
    lines += ["", f"## H5 别处有没有变差（不比 K16 差 1 pp 以上、有限）：{v(h['H5']['pass_'])}", ""]
    lines += [f"- {x['variant']} {x['scene']}：" + "，".join(f"{m} {_pp(c['minus_k16'])}" for m, c in x["checks"].items())
              + f"，有限 {_f(x['finite'])}" for x in h["H5"]["rows"]]
    lines += ["", f"## H6 基底数曲线 K16c → K32c → K64c → K 取满（T=8，单调不降，允许 0.5 pp）：{v(h['H6']['pass_'])}", ""]
    lines += [f"- {x['scene']}：" + " → ".join(f"{k} {_pct(g)}" for k, g in x["curve"].items()) for x in h["H6"]["scenes"]]
    if summary["readout"]:
        lines += ["", "## 读出（只报告）", "", "| 变体 | 场景 | full nmse | no_cons nmse | full VPT@0.1 | no_cons VPT@0.1 | full gain | no_cons gain |",
                  "|---|---|---|---|---|---|---|---|"]
        for e in summary["readout"]:
            a, b = e.get("po_jepa_full") or {}, e.get("po_jepa_no_consistency") or {}
            lines.append(f"| {e['variant']} | {e['scene']} | {_f(a.get('nmse_mean'))} | {_f(b.get('nmse_mean'))} | {_f(a.get('vpt'))} | "
                         f"{_f(b.get('vpt'))} | {_f(a.get('gain'))} | {_f(b.get('gain'))} |")
        lines.append("\nG9 收缩检查（motion_gain）在 `<变体>/shrinkage/`，这里不汇总。")
    return lines


def write(output, summary):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=list), encoding="utf-8")
    (output/"SUMMARY.md").write_text(markdown(summary), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    c1 = sub.add_parser("c1"); c1.add_argument("--results", type=Path, required=True); c1.add_argument("--output", type=Path, required=True)
    c2 = sub.add_parser("c2")
    for name in ("--results", "--v7-results", "--output"):
        c2.add_argument(name, type=Path, required=True)
    c2.add_argument("--readout", type=Path); c2.add_argument("--v7-readout", type=Path)
    c3 = sub.add_parser("c3")
    for name in ("--results", "--c2-results", "--output"):
        c3.add_argument(name, type=Path, required=True)
    r = sub.add_parser("readout"); r.add_argument("--results", type=Path, required=True); r.add_argument("--output", type=Path, required=True)
    r.add_argument("--actions", type=Path, help="动作探针 probe-actions-<场景>.json 所在目录")
    a = p.parse_args(argv)
    if a.stage == "c1":
        summary = c1_summary(a.results)
    elif a.stage == "c2":
        summary = c2_summary(a.results, a.v7_results, a.readout, a.v7_readout)
    elif a.stage == "c3":
        summary = c3_summary(a.results, a.c2_results)
    else:
        summary = readout_summary(a.results, a.actions)
    write(a.output, summary)
    print(f"{summary['stage']}：问题 {len(summary.get('problems', []))} 个，缺失 {len(summary.get('missing', []))} 个 → {Path(a.output)/'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
