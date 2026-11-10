"""13 §8：DROID 汇总（`summary` 任务，S13；只读，CPU）。

  python tools/droid_summary.py --config configs/droid/<name>.yaml --data-root <数据根> --run-root <运行根>

输出 `<运行根>/<name>/summary/`：summary.json、summary.md（D1–D11）、figures/*.svg、missing.json、checks.json，全部通过时
再加 COMPLETE.json（记 summary.json 的 SHA）。先写到同一父目录下的临时目录，写完整体换名；已有的 `summary/` 先改名为
`summary.previousN`（13 §7）。13 §8.3 有一项核对不通过：只写 summary.json（status: blocked）、checks.json、missing.json，
不出表和图、不写 COMPLETE.json，退出 1。缺的任务列进 missing.json，表里写“缺”，不影响退出码。

统计、精度匹配、表格的数字格式和画图沿用仿真那份 `tools/summarize_addon.py`（13 §8 “同一套统计和表格代码”）：
`seed_stats`、`paired_seed`、`paired_stats`、`matched`、`MATCH_TOLERANCE`、`BOOTSTRAP_*`、`_f`、`_ms`、`_match_label`、
`plots.line_chart`。DROID 只换输入和聚类单位（13 §8.2）：

- 层次：窗口 → 回合内平均 → 回合间平均（一个测试集、一个种子、一个条件一个数）→ 种子平均 ± 种子 SD。窗口键是
  (回合, 窗口编号, 键名)，一个量由几个键名、几个时长合成时在回合内一起平均（13 §5 的做法）。
- 精度匹配：e_c = id_test 上 16×16、T ∈ 同分布时长的未来误差（种子平均），5% 规则；ood_test 沿用 id_test 的结论。
- 配对：po_jepa_full 减对照，按窗口键配对、回合内平均；不训练的对照（identity、vjepa2_ac_zeroshot）和每个种子配对。
- 自助法整群：id_test 按场景（index 里的 (lab, scene_id)），ood_test 按回合；2000 次，`torch.Generator` 种子 1703，每个比较
  重新开始抽。抽到的群里按回合加权平均（每个场景正好 5 条回合时就是场景平均；群是单条回合时和仿真的
  `summarize_addon.bootstrap_ci` 逐位相同）。
- 成败分开：每张表都按回合的 outcome 再算 success、failure 两份（summary.json 的 `by_outcome`），D9 列 D1、D2 的。

训练产物（09 §4–§5：registration.json、selection.json、report.json、train-log.jsonl）、`check/env.json` 和评测任务的
job.json/source.json 里的来源字段，格式由 S09 定；这里只经过 `Readers` 的三个读取函数（`read_train`、`read_eval_view`、
`read_env`，现在抛 NotImplementedError），13 §8.3 的核对和 D11 成本对着 `TrainView`、`EvalView` 写好。S09 补上这三个
函数后，`tools/droid.py` 的 `MAINS` 加 `summary` 一行即可。评测任务自己的产物（rows、frame_rate.json、resolution.json、
planning/、report.json，S11a/S11b/S12）和 eval/spectral-gap、标定文件 sigma_z.json、lambda-<条件>.json 按已定格式直接读。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
from dataclasses import dataclass
import math
from pathlib import Path
import re
import statistics
import sys
import tempfile
from typing import Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_addon as sa  # noqa: E402  （它把 src 放进 sys.path）

import torch  # noqa: E402

from po_jepa.droid import evaluate as ev  # noqa: E402
from po_jepa.droid import index as ix  # noqa: E402
from po_jepa.droid import planning as pl  # noqa: E402
from po_jepa.droid import prepare as pr  # noqa: E402
from po_jepa.droid import spectral_gap as sg  # noqa: E402
from po_jepa.droid.config import DroidConfig, load_config  # noqa: E402
from po_jepa.runtime import cache, plots  # noqa: E402
from po_jepa.runtime.addon import d4  # noqa: E402
from po_jepa.runtime.addon import spectral_gap as sim_gap  # noqa: E402

SCHEMA = "droid_summary_v1"
SPLITS = ev.TEST_SPLITS
OUTCOMES = ix.OUTCOMES
FULL = "po_jepa_full"
AC = ev.AC_CONDITIONS
PO_CLASS = ("po_jepa_full", "po_jepa_no_consistency", "fixed_fourier")          # 13 §8.3 σ_z：用图谱的条件
LAMBDA_CONDITIONS = ("po_jepa_full", "fixed_fourier", "transformer_consistency")  # 09 §3.2
INITIAL_SAME_AS = {"po_jepa_no_consistency": "po_jepa_full", "fixed_fourier": "po_jepa_full",
                   "transformer_consistency": "matched_transformer"}             # 09 §2.4（zeroshot 另核）
# 13 §2、10 §3.8：AC 只比 LN 空间的量；输入尺度的量（含 D3、D8 的 *_in）标“不可比”
NOT_COMPARABLE = frozenset(sa.NOT_COMPARABLE | set(sa.INPUT_SCALE_D4))
DATA_FILES = ("splits.json", "windows/val.json", "windows/id_test.json", "windows/ood_test.json", "action_scale.json")
D1_METRICS = ("future_g16", "future_g8", "tcd_ln", "scd_ln", "tcd_normalized", "scd_normalized", "delta_act")
D2_METRICS = ("future_heldout_time", "future_heldout_grid", "future_heldout_both")
PAIRED_METRICS = D1_METRICS + D2_METRICS + ("pd_ln", "pd_in", "xpd_ln", "xpd_in", "long_direct", "long_composed")
RANK_KEYS = ("mean_rank", "top1", "reciprocal_rank")
CEM_KEYS = ("position_error_m", "gripper_error", "position_relative")
CHECK_ITEMS = ("配置", "特征", "划分和窗口", "训练集", "σ_z", "λ", "初始权重", "训练日志", "评测", "评测窗口")


# ---------------------------------------------------------------- 训练和评测来源的读取接口（格式等 S09）

@dataclass(frozen=True)
class TrainView:
    """一个训练任务 `train/<条件>/seed<种子>/`（09 §4.1）里汇总要用的字段。读法 `read_train` 等 S09。

    - `data_sha256`：{DATA_FILES 里的键: SHA}（登记里的 `splits.json`、`windows/val.json`、`action_scale.json`）
    - `lambdas`：带一致性的三个条件登记的 {"time_weight", "space_weight"}，其余为 None；`sigma_z` 同理（PO-JEPA 类）
    - `log_ranges`：train-log.jsonl 每行的 (first_step, last_step)，按文件顺序
    - `best_checkpoint_sha256`：selection.json 的当前最优；`step_seconds`：每步用时（日志里的单步用时）
    - `curves`：{"validation": [(步, 均值)], "train_jepa": [(步, JEPA 损失)]}（收敛曲线）"""
    condition: str
    seed: int
    config_sha256: str
    feature_manifest_sha256: str
    data_sha256: Mapping[str, str]
    train_set_sha256: str
    sigma_z: float | None
    lambdas: Mapping[str, float] | None
    initial_model_sha256: str
    updates: int
    completed: int
    log_ranges: tuple[tuple[int, int], ...]
    best_checkpoint_sha256: str
    step_seconds: tuple[float, ...]
    peak_memory_bytes: int | None
    gpu_hours: float | None
    resumes: int
    spectral_backend: str | None
    curves: Mapping[str, Sequence[tuple[float, float]]]


@dataclass(frozen=True)
class EvalView:
    """一个评测任务 `eval/<条件>[/seed<种子>]/` 的 job.json/source.json 里汇总要核的来源字段（10 §8.1、§8.9）。读法
    `read_eval_view` 等 S09（读检查点的记录由 `evaluate.load_selected` 给）。

    - `checkpoint_sha256`：读的检查点 SHA（不训练的对照为 None）
    - `data_sha256`：{DATA_FILES 里的键: SHA}（`splits.json`、`windows/id_test.json`、`windows/ood_test.json`、`action_scale.json`）
    - `initial_model_sha256`、`construction_seed`：只有 vjepa2_ac_zeroshot 用（09 §2.4：等于该种子的 vjepa2_ac 初始哈希）"""
    condition: str
    seed: int | None
    feature_manifest_sha256: str
    data_sha256: Mapping[str, str]
    checkpoint_sha256: str | None
    initial_model_sha256: str | None = None
    construction_seed: int | None = None


def read_train(path: Path) -> TrainView:
    raise NotImplementedError("13 §8.1: reading train/<condition>/seed<seed>/ waits for the S09 registration, "
                              "selection.json, report.json and train-log.jsonl format")


def read_eval_view(path: Path) -> EvalView:
    raise NotImplementedError("13 §8.3: the checkpoint and data fields of eval job.json/source.json wait for S09 "
                              "(evaluate.load_selected)")


def read_env(path: Path) -> dict:
    raise NotImplementedError("13 §8.1: the check/env.json format waits for S09 (09 §7)")


@dataclass(frozen=True)
class Readers:
    train: Callable[[Path], TrainView] = read_train
    evaluation: Callable[[Path], EvalView] = read_eval_view
    env: Callable[[Path], dict] = read_env


# ---------------------------------------------------------------- 量的登记：{名字: (来源, [(键名, 时长或 None)])}

def _label(t: float) -> str:
    return f"{t:g}"


def metric_specs(config: DroidConfig) -> dict[str, tuple[str, tuple]]:
    """汇总里每个量从哪个来源、哪些键名、哪些时长的窗口来。rows 的键名照 `evaluate.value_names`。"""
    e = config.evaluation
    ids, held, long = tuple(e.durations_id), tuple(e.durations_heldout), tuple(e.durations_long)
    joint = tuple(ev.JOINT_HELDOUT)
    m: dict[str, tuple[str, tuple]] = {}

    def rows(name, parts):
        m[name] = ("rows", tuple(parts))
    rows("future_g16", [("future:g=16", ids)])
    rows("future_g8", [("future:g=8", ids)])
    rows("tcd_ln", [("tcd_ln", ids)])
    rows("scd_ln", [(f"scd_ln:h={h}", ids) for h in ev.SCD_GRIDS])
    rows("tcd_normalized", [("tcd_normalized", ids)])
    rows("scd_normalized", [(f"scd_normalized:h={h}", ids) for h in ev.SCD_GRIDS])
    rows("delta_act", [("delta_act", ids)])
    rows("future_heldout_time", [("future:g=16", held)])
    rows("future_heldout_grid", [("future:g=12", ids)])
    rows("future_heldout_both", [("future:g=12", joint)])
    for t in ids:
        rows(f"future_g16:T={t}", [("future:g=16", (t,))])
        rows(f"future_g8:T={t}", [("future:g=8", (t,))])
    for t in (*ids, *held):
        rows(f"tcd_ln:T={t}", [("tcd_ln", (t,))])
    for t in held:
        rows(f"future_g16:T={t}", [("future:g=16", (t,))])
        for term in ("direct", "partition", "rollout"):
            rows(f"heldout_{term}:T={t}", [(f"heldout_{term}", (t,))])
    for t in joint:
        rows(f"future_g12:T={t}", [("future:g=12", (t,))])
    for t in long:
        for k in ("direct", "composed", "pd_ln", "pd_in"):
            rows(f"long_{k}:T={t}", [(f"long_{k}", (t,))])
    for k in ("direct", "composed"):
        rows(f"long_{k}", [(f"long_{k}", long)])
    for metric in ("ln", "in"):
        rows(f"pd_{metric}", [(f"pd_{metric}:{ev.partition_label(p)}", (int(t),)) for t, p in ev.PARTITIONS])
        for t, p in ev.PARTITIONS:
            rows(f"pd_{metric}:T={_label(t)}:{ev.partition_label(p)}", [(f"pd_{metric}:{ev.partition_label(p)}", (int(t),))])
        rows(f"xpd_{metric}", [(f"xpd_{metric}:n={n}", (int(t),)) for t, n in ev.CROSS])
        for t, n in ev.CROSS:
            rows(f"xpd_{metric}:T={_label(t)}:n={n}", [(f"xpd_{metric}:n={n}", (int(t),))])
        for d, p in ev.UNEQUAL:
            label = ev.partition_label(p)
            rows(f"tcd_unequal_{metric}:d={_label(d)}:{label}", [(f"tcd_unequal_{metric}:{label}", (int(d),))])
    for name in ev.frame_rate_names(config):
        m[f"frame_rate:{name}"] = ("frame_rate", ((name, None),))
    for name in ev.resolution_names(config):
        m[f"resolution:{name}"] = ("resolution", ((name, ids),))
    for d in e.planning_durations:
        for key in RANK_KEYS:
            m[f"rank:{key}:d={d}"] = ("rank", ((key, (d,)),))
        for key in CEM_KEYS:
            m[f"cem:{key}:d={d}"] = ("cem", ((key, (d,)),))
    return m


def _comparable(condition: str, metric: str) -> bool:
    base = metric.split(":")[0]
    return not (condition in AC and base in NOT_COMPARABLE)


# ---------------------------------------------------------------- 统计（13 §3、§8.2）

def window_values(cell: Mapping | None, spec: tuple, outcome: str | None = None) -> dict | None:
    """一个 (测试集, 条件, 种子) 上某个量的逐窗口值 {(回合, 窗口编号, 键名): 值}；没有这个来源时返回 None。值为 None 的
    （替身被跳过的 delta_act、越界排除的原始帧率起点、真实净位移 < 1 cm 的相对误差）不进。"""
    if cell is None:
        return None
    source, parts = spec
    records = cell.get(source)
    if records is None:
        return None
    out = {}
    for r in records:
        if outcome is not None and r["outcome"] != outcome:
            continue
        for name, durations in parts:
            if durations is not None and r["duration"] not in durations:
                continue
            v = r["values"][name]
            if v is not None:
                out[(r["uuid"], r["id"], name)] = v
    return out


def episode_count(values: Mapping | None) -> int:
    return len(sa.by_trajectory(values)) if values else 0


def cell_stats(per_seed_values: Mapping) -> dict:
    """{种子标签: 逐窗口值} → 种子平均 ± SD（`summarize_addon.seed_stats`），另记每个种子的回合数和窗口数。"""
    stats = sa.seed_stats({s: sa.macro(v or {}) for s, v in per_seed_values.items()})
    stats["episodes"] = {str(s): episode_count(v) for s, v in sorted(per_seed_values.items())}
    stats["windows"] = {str(s): len(v or {}) for s, v in sorted(per_seed_values.items())}
    return stats


def cluster_bootstrap_ci(per_seed: Sequence[Mapping[str, float]], cluster_of: Mapping[str, str]) -> dict | None:
    """按群整群自助（13 §3、§8.2）：所有种子共有的回合 → 它们的群（排好序）；`torch.Generator().manual_seed(1703)` 抽
    2000 × 群数个群下标；每次每个种子在抽到的群（重复的照算）里按回合加权平均配对差，再对种子平均；取 2.5、97.5 分位数
    （linear，float64）。每个群一条回合时和 `summarize_addon.bootstrap_ci` 相同。"""
    if not per_seed:
        return None
    ids = sorted(set.intersection(*[set(d) for d in per_seed]))
    if not ids:
        return None
    clusters = sorted({cluster_of[u] for u in ids})
    position = {c: i for i, c in enumerate(clusters)}
    member = torch.tensor([position[cluster_of[u]] for u in ids])
    counts = torch.zeros(len(clusters), dtype=torch.float64).index_add_(0, member, torch.ones(len(ids), dtype=torch.float64))
    sums = torch.stack([torch.zeros(len(clusters), dtype=torch.float64).index_add_(
        0, member, torch.tensor([d[u] for u in ids], dtype=torch.float64)) for d in per_seed])        # [种子, 群]
    generator = torch.Generator().manual_seed(sa.BOOTSTRAP_SEED)
    draws = torch.randint(len(clusters), (sa.BOOTSTRAP_REPS, len(clusters)), generator=generator)
    boot = (sums[:, draws].sum(dim=2)/counts[draws].sum(dim=1)).mean(dim=0)
    low, high = torch.quantile(boot, torch.tensor([.025, .975], dtype=torch.float64)).tolist()
    return dict(low=low, high=high, reps=sa.BOOTSTRAP_REPS, clusters=len(clusters), episodes=len(ids),
                seed=sa.BOOTSTRAP_SEED)


def _paired_stats(diffs: Mapping[str, float]) -> dict:
    out = sa.paired_stats(diffs)
    out["episodes"] = out.pop("trajectories")
    return out


# ---------------------------------------------------------------- 读输入

def _json(path: Path):
    return ix._loads(Path(path).read_bytes(), str(path))


def episode_info(data_root: Path) -> dict[str, dict]:
    """`index/episodes.jsonl`：{uuid: {outcome, scene}}，scene = "lab/scene_id"（id_test 的整群单位）。"""
    path = ix.index_dir(data_root)/ix.EPISODES_FILE
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = ix._loads(line.encode("utf-8"), str(path))
            out[e["uuid"]] = dict(outcome=e["outcome"], scene=f"{e['lab']}/{e['scene_id']}", scene_id=e["scene_id"])
    return out


def data_reference(data_root: Path) -> dict[str, str | None]:
    """数据根里 DATA_FILES 的当前 SHA（没有的记 None）。"""
    files = {"splits.json": ix.index_dir(data_root)/ix.SPLITS_FILE,
             "action_scale.json": pr.prepared_dir(data_root)/pr.ACTION_SCALE_FILE,
             **{f"windows/{s}.json": pr.prepared_dir(data_root)/pr.WINDOWS_DIR/f"{s}.json" for s in ("val", *SPLITS)}}
    return {k: cache.file_sha(p) if p.exists() else None for k, p in sorted(files.items())}


def eval_jobs(config: DroidConfig) -> list[tuple[str, int | None]]:
    return [(c, s) for c in config.training.conditions for s in config.training.seeds] + \
           [(c, None) for c in config.training.untrained]


def eval_dir(run: Path, condition: str, seed: int | None) -> Path:
    return run/"eval"/condition/(f"seed{seed}" if seed is not None else "")


def _job_name(condition: str, seed: int | None) -> str:
    return f"eval/{condition}" + (f"/seed{seed}" if seed is not None else "")


def _complete(folder: Path, name: str = "report.json") -> dict | None:
    """有 COMPLETE.json 且记的 SHA 等于 report.json 才算完成，返回报告；没有 COMPLETE.json 返回 None。"""
    if not (folder/"COMPLETE.json").exists():
        return None
    if _json(folder/"COMPLETE.json").get("report_sha256") != cache.file_sha(folder/name):
        raise ValueError(f"{folder}: COMPLETE.json does not match {name}")
    return _json(folder/name)


_ROW = re.compile(r"^\d{5}\.json$")


def _rows(folder: Path, split: str) -> list[dict]:
    files = sorted(p for p in (folder/"rows").iterdir() if _ROW.match(p.name))
    out = []
    for i, p in enumerate(files):
        body = _json(p)
        if body.get("schema") != ev.SCHEMA_ROWS or body.get("split") != split or body.get("batch") != i:
            raise ValueError(f"{p} is not batch {i} of the {split} rows")
        out += [dict(r) for r in body["windows"]]
    return out


def load_eval(folder: Path, report: Mapping, info: Mapping[str, dict], missing: list[str], name: str) -> dict:
    """一个完成的评测任务：每个测试集的 rows、frame_rate、resolution、rank、cem 记录（统一成 {uuid, id, duration,
    outcome, values}），以及报告。没做的轴（`axes` 为假、没有 planning/）记进 `missing`。"""
    cells = {}
    for split in SPLITS:
        base = folder/split
        cell = dict(rows=_rows(base, split), report=report["splits"][split])
        if report.get("axes", {}).get("frame_rate"):
            body = _json(base/ev.FRAME_RATE_FILE)
            if body.get("schema") != ev.SCHEMA_FRAME_RATE or body.get("split") != split:
                raise ValueError(f"{base/ev.FRAME_RATE_FILE} is not the {split} frame-rate file")
            cell["frame_rate"] = [dict(r, duration=None) for r in body["entries"]]
        else:
            missing.append(f"{name}:{split}/frame_rate")
        if report.get("axes", {}).get("resolution"):
            body = _json(base/ev.RESOLUTION_FILE)
            if body.get("schema") != ev.SCHEMA_RESOLUTION or body.get("split") != split:
                raise ValueError(f"{base/ev.RESOLUTION_FILE} is not the {split} resolution file")
            cell["resolution"] = [dict(r) for r in body["entries"]]
        else:
            missing.append(f"{name}:{split}/resolution")
        planning = base/"planning"
        if (planning/"report.json").exists():
            rank, cem_doc = _json(planning/"rank.json"), _json(planning/"cem.json")
            cell["planning_report"] = _json(planning/"report.json")
            for doc in (rank, cem_doc, cell["planning_report"]):
                if doc.get("schema") != pl.SCHEMA or doc.get("split") != split:
                    raise ValueError(f"{planning} does not hold the {split} planning files")
            cell["rank"] = [dict(uuid=r["uuid"], id=r["id"], duration=r["duration"], outcome=info[r["uuid"]]["outcome"],
                                 values={k: r[k] for k in RANK_KEYS}) for r in rank["windows"]]
            cell["cem"] = [dict(uuid=r["uuid"], id=r["id"], duration=r["duration"], outcome=info[r["uuid"]]["outcome"],
                                small_displacement=r["small_displacement"],
                                values={"position_error_m": r["position_error_m"], "gripper_error": r["gripper_error"],
                                        "position_relative": None if r["small_displacement"] else r["position_relative"]})
                           for r in cem_doc["windows"]]
        else:
            missing.append(f"{name}:{split}/planning")
        cells[split] = cell
    return cells


def collect(config: DroidConfig, data_root: Path, run_root: Path, readers: Readers) -> dict:
    """读运行根下本配置的全部结果（13 §8.1）。缺的进 `missing`；训练和评测来源经 `readers`。"""
    run = Path(run_root)/config.name
    info = episode_info(Path(data_root))
    missing: list[str] = []
    cells, evals, trains = {}, {}, {}
    for c, s in eval_jobs(config):
        folder, name = eval_dir(run, c, s), _job_name(c, s)
        report = _complete(folder) if folder.exists() else None
        if report is None:
            missing.append(name)
            continue
        loaded = load_eval(folder, report, info, missing, name)
        for split, cell in loaded.items():
            cells[(split, c, s)] = cell
        evals[(c, s)] = dict(report=report, view=readers.evaluation(folder), job=_json(folder/"job.json"))
    for c in config.training.conditions:
        for s in config.training.seeds:
            folder = run/"train"/c/f"seed{s}"
            if not (folder/"COMPLETE.json").exists():
                missing.append(f"train/{c}/seed{s}")
                continue
            trains[(c, s)] = readers.train(folder)
    gap = _complete(run/"eval"/"spectral-gap")
    if gap is None:
        missing.append(f"eval/{sg.JOB_DIR[1]}")
    calibration = {}
    path = run/"calibration"/"sigma_z.json"
    if path.exists():
        calibration["sigma_z"] = sg.read_sigma_z(path, config)[0]
    else:
        missing.append("calibrate/sigma-z")
    for c in LAMBDA_CONDITIONS:
        path = run/"calibration"/f"lambda-{c}.json"
        if path.exists():
            doc = _json(path)
            calibration[f"lambda-{c}"] = dict(time_weight=doc["time_weight"], space_weight=doc["space_weight"])
        else:
            missing.append(f"calibrate/lambda-{c}")
    env = None
    if (run/"check"/"env.json").exists():
        env = readers.env(run/"check"/"env.json")
    else:
        missing.append("check/env")
    return dict(cells=cells, evals=evals, trains=trains, gap=gap, calibration=calibration, env=env, info=info,
                missing=sorted(missing), reference=data_reference(Path(data_root)))


# ---------------------------------------------------------------- 数据一致性核对（13 §8.3）

def _check(item: str, problems: list[str]) -> dict:
    return dict(item=item, ok=not problems, problems=problems)


def _data_shas(views) -> dict[str, set]:
    seen = defaultdict(set)
    for label, shas in views:
        for k, v in shas.items():
            seen[k].add(v)
    return seen


def run_checks(config: DroidConfig, got: Mapping) -> list[dict]:
    """13 §8.3 的 8 项，另加“配置”（所有任务的 config_sha256 = 本配置）和“评测窗口”（各条件评测的窗口、原始帧率起点、
    分辨率窗口、规划窗口编号相同，rows 的 outcome = 索引）。每项 {item, ok, problems}。"""
    evals, trains, gap, cal = got["evals"], got["trains"], got["gap"], got["calibration"]
    out = []

    p = [f"{_job_name(c, s)} config_sha256 differs" for (c, s), e in sorted(evals.items(), key=_order)
         if e["report"]["job"].get("config_sha256") != config.config_sha256]
    p += [f"train/{c}/seed{s} config_sha256 differs" for (c, s), t in sorted(trains.items()) if t.config_sha256 != config.config_sha256]
    if gap is not None and gap["job"].get("config_sha256") != config.config_sha256:
        p.append("eval/spectral-gap config_sha256 differs")
    out.append(_check("配置", p))

    features = {f"train/{c}/seed{s}": t.feature_manifest_sha256 for (c, s), t in trains.items()}
    features |= {_job_name(c, s): e["view"].feature_manifest_sha256 for (c, s), e in evals.items()}
    out.append(_check("特征", [] if len(set(features.values())) <= 1 else
                      [f"feature_manifest_sha256 differs: {_groups(features)}"]))

    p = []
    views = [(f"train/{c}/seed{s}", t.data_sha256) for (c, s), t in sorted(trains.items())]
    views += [(_job_name(c, s), e["view"].data_sha256) for (c, s), e in sorted(evals.items(), key=_order)]
    if gap is not None:
        views.append(("eval/spectral-gap", {f"windows/{k}.json": v for k, v in gap["job"]["inputs"]["windows_sha256"].items()}))
    for label, shas in views:
        unknown = sorted(set(shas) - set(DATA_FILES))
        if unknown:
            p.append(f"{label} records unknown data files {unknown}")
        for k, v in sorted(shas.items()):
            if k in DATA_FILES and v != got["reference"][k]:
                p.append(f"{label} {k} differs from the data root")
    for k, values in sorted(_data_shas(views).items()):
        if len(values) > 1:
            p.append(f"{k}: {len(values)} different SHAs across tasks")
    out.append(_check("划分和窗口", p))

    sets = {f"train/{c}/seed{s}": t.train_set_sha256 for (c, s), t in trains.items()}
    out.append(_check("训练集", [] if len(set(sets.values())) <= 1 else [f"train set SHA differs: {_groups(sets)}"]))

    p = []
    sigma = cal.get("sigma_z")
    for (c, s), t in sorted(trains.items()):
        if c in PO_CLASS and (sigma is None or t.sigma_z != sigma):
            p.append(f"train/{c}/seed{s} sigma_z {t.sigma_z!r} != sigma_z.json {sigma!r}")
    if gap is not None and gap["job"].get("sigma_z") != sigma:
        p.append(f"eval/spectral-gap sigma_z {gap['job'].get('sigma_z')!r} != sigma_z.json {sigma!r}")
    out.append(_check("σ_z", p))

    p = []
    for (c, s), t in sorted(trains.items()):
        if c in LAMBDA_CONDITIONS:
            want = cal.get(f"lambda-{c}")
            if want is None or t.lambdas is None or dict(t.lambdas) != want:
                p.append(f"train/{c}/seed{s} lambdas {None if t.lambdas is None else dict(t.lambdas)} != lambda-{c}.json {want}")
    out.append(_check("λ", p))

    p = []
    for (c, s), t in sorted(trains.items()):
        base = INITIAL_SAME_AS.get(c)
        if base is not None and (base, s) in trains and t.initial_model_sha256 != trains[(base, s)].initial_model_sha256:
            p.append(f"train/{c}/seed{s} initial weights differ from {base}")
    for (c, s), e in sorted(evals.items(), key=_order):
        if c == "vjepa2_ac_zeroshot":
            v = e["view"]
            partner = trains.get(("vjepa2_ac", v.construction_seed))
            if partner is not None and v.initial_model_sha256 != partner.initial_model_sha256:
                p.append(f"eval/vjepa2_ac_zeroshot initial weights differ from vjepa2_ac seed{v.construction_seed}")
    out.append(_check("初始权重", p))

    p = []
    for (c, s), t in sorted(trains.items()):
        expect, ok = 1, True
        for first, last in t.log_ranges:
            if first != expect or last < first:
                ok = False
                break
            expect = last + 1
        if not ok or expect != t.updates + 1 or t.completed != t.updates:
            p.append(f"train/{c}/seed{s} log does not cover 1…{t.updates} exactly once (completed {t.completed})")
    out.append(_check("训练日志", p))

    p = []
    for (c, s), e in sorted(evals.items(), key=_order):
        name = _job_name(c, s)
        if e["report"].get("status") != "passed":
            p.append(f"{name} report status {e['report'].get('status')!r}")
        for split in SPLITS:
            v = e["report"]["splits"][split]["triangle"]["violations"]
            if v:
                p.append(f"{name} {split}: {v} triangle-inequality violations")
        if s is None:
            continue
        t = trains.get((c, s))
        if t is None:
            p.append(f"{name} has no completed training task")
        elif e["view"].checkpoint_sha256 != t.best_checkpoint_sha256:
            p.append(f"{name} read checkpoint {e['view'].checkpoint_sha256} != selection best {t.best_checkpoint_sha256}")
    out.append(_check("评测", p))

    p = []
    info = got["info"]
    for split in SPLITS:
        for source in ("rows", "frame_rate", "resolution", "rank", "cem"):
            ids = {}
            for (sp, c, s), cell in got["cells"].items():
                if sp == split and cell.get(source) is not None:
                    ids[_job_name(c, s)] = tuple(sorted(r["id"] for r in cell[source]))
            if len(set(ids.values())) > 1:
                p.append(f"{split} {source}: the conditions evaluated different windows ({_groups(ids, short=True)})")
        for (sp, c, s), cell in sorted(got["cells"].items(), key=lambda kv: (kv[0][0], *_order((kv[0][1], kv[0][2])))):
            if sp != split:
                continue
            for source in ("rows", "frame_rate", "resolution"):
                bad = [r["id"] for r in cell.get(source) or [] if info.get(r["uuid"], {}).get("outcome") != r["outcome"]]
                if bad:
                    p.append(f"{_job_name(c, s)} {split} {source}: outcome differs from the index for {bad[:3]}")
            if split == "id_test":
                bad = sorted({r["uuid"] for r in cell["rows"] if info.get(r["uuid"], {}).get("scene_id", -1) < 0})
                if bad:
                    p.append(f"{_job_name(c, s)} id_test episodes without a scene: {bad[:3]}")
    out.append(_check("评测窗口", p))
    assert tuple(c["item"] for c in out) == CHECK_ITEMS
    return out


def _order(key):
    c, s = key
    return (c, -1 if s is None else s)


def _groups(mapping: Mapping, short: bool = False) -> str:
    by = defaultdict(list)
    for k, v in sorted(mapping.items()):
        by[v].append(k)
    return "; ".join(f"{len(ks)} tasks" if short else ", ".join(ks) for ks in by.values())


# ---------------------------------------------------------------- 表（13 §8.4）

class Context:
    def __init__(self, config: DroidConfig, got: Mapping):
        self.config, self.got = config, got
        self.specs = metric_specs(config)
        self.conditions = tuple(config.training.conditions) + tuple(config.training.untrained)
        self.seeds = tuple(config.training.seeds)
        self.untrained = set(config.training.untrained)

    def seed_of(self, condition: str, seed: int | None):
        return None if condition in self.untrained else seed

    def per_seed(self, split: str, condition: str, metric: str, outcome: str | None) -> dict:
        """{种子标签: 逐窗口值}（不训练的对照只有一个标签 "untrained"）；没有这个任务的不进。"""
        seeds = (None,) if condition in self.untrained else self.seeds
        out = {}
        for s in seeds:
            cell = self.got["cells"].get((split, condition, s))
            values = window_values(cell, self.specs[metric], outcome)
            if values is not None:
                out["untrained" if s is None else s] = values
        return out

    def stat(self, split, condition, metric, outcome=None):
        if not _comparable(condition, metric):
            return "not_comparable"
        return cell_stats(self.per_seed(split, condition, metric, outcome))

    def table(self, split, metrics, outcome=None, conditions=None):
        return {c: {m: self.stat(split, c, m, outcome) for m in metrics} for c in (conditions or self.conditions)}

    def clusters(self, split: str) -> dict[str, str]:
        info = self.got["info"]
        if split == "id_test":
            return {u: e["scene"] for u, e in info.items()}
        return {u: u for u in info}


def precision(ctx: Context) -> dict:
    """e_c（13 §8.2）：id_test、16×16、同分布时长的未来误差，回合宏平均的种子平均。"""
    return {c: ctx.stat("id_test", c, "future_g16")["mean"] for c in ctx.conditions}


def comparisons(ctx: Context, split: str, errors: Mapping, outcome: str | None = None) -> list[dict]:
    """D7：po_jepa_full − 每个对照，按窗口键配对、回合内平均，每个种子各报一次；整群自助的 95% 区间；精度匹配标记
    （identity 不参与；ood_test 沿用 id_test 的 e_c）。"""
    out = []
    cluster_of = ctx.clusters(split)
    for ref in ctx.conditions:
        if ref == FULL:
            continue
        is_match, gap_pct = (None, None) if ref == "identity" else sa.matched(errors.get(FULL), errors.get(ref))
        for m in PAIRED_METRICS:
            row = dict(reference=ref, metric=m, matched=is_match, error_difference_percent=gap_pct)
            if not _comparable(ref, m):
                out.append(dict(row, status="not_comparable"))
                continue
            per_seed = {}
            for s in ctx.seeds:
                f = window_values(ctx.got["cells"].get((split, FULL, s)), ctx.specs[m], outcome)
                o = window_values(ctx.got["cells"].get((split, ref, ctx.seed_of(ref, s))), ctx.specs[m], outcome)
                diffs = sa.paired_seed(f, o)
                if diffs:
                    per_seed[s] = diffs
            if not per_seed:
                out.append(dict(row, status="missing"))
                continue
            stats = {str(s): _paired_stats(d) for s, d in per_seed.items()}
            means = [v["mean"] for v in stats.values()]
            out.append(dict(row, status="ok", per_seed=stats, mean=sum(means)/len(means),
                            seed_sd=statistics.stdev(means) if len(means) > 1 else None,
                            seeds_full_lower=sum(v < 0 for v in means),
                            ci=cluster_bootstrap_ci(list(per_seed.values()), cluster_of)))
    return out


def epsilon_hat(ctx: Context, split: str, condition: str, seed, outcome=None) -> float | None:
    """D8 的 ε̂：一个 (测试集, 条件, 种子) 上同分布时长全部窗口的 `tcd_normalized` 的中位数（13 §5 T5 的做法；DROID 的
    窗口每个只有一个时长，所以取同分布时长的全部窗口）。"""
    values = window_values(ctx.got["cells"].get((split, condition, seed)), ctx.specs["tcd_normalized"], outcome)
    return statistics.median(values.values()) if values else None


def partitions(ctx: Context, split: str, outcome=None) -> dict:
    """D8：7 个分割的 pd_ln、pd_in 和有效斜率 pd_in/(x·ε̂)，xpd，不等分 TCD，留出分解和三角不等式违反数。"""
    out = {}
    for c in ctx.conditions:
        seeds = (None,) if c in ctx.untrained else ctx.seeds
        eps = {("untrained" if s is None else s): epsilon_hat(ctx, split, c, s, outcome) for s in seeds
               if (split, c, s) in ctx.got["cells"]}
        entry = dict(epsilon_hat="not_comparable" if c in AC else sa.seed_stats(eps), partitions=[], cross_grid={},
                     unequal={}, decomposition={})
        for t, parts in ev.PARTITIONS:
            label, x = ev.partition_label(parts), d4.partition_x(t, parts)
            row = dict(T=t, partition=label, x=x)
            for metric in ("ln", "in"):
                row[f"pd_{metric}"] = ctx.stat(split, c, f"pd_{metric}:T={_label(t)}:{label}", outcome)
            slopes = {}
            for s, values in ctx.per_seed(split, c, f"pd_in:T={_label(t)}:{label}", outcome).items():
                pd = sa.macro(values)
                slopes[s] = pd/(x*eps[s]) if pd is not None and eps.get(s) else None
            row["effective_slope"] = "not_comparable" if c in AC else sa.seed_stats(slopes)
            entry["partitions"].append(row)
        for metric in ("ln", "in"):
            for t, n in ev.CROSS:
                name = f"xpd_{metric}:T={_label(t)}:n={n}"
                entry["cross_grid"][name] = ctx.stat(split, c, name, outcome)
            for d, p in ev.UNEQUAL:
                name = f"tcd_unequal_{metric}:d={_label(d)}:{ev.partition_label(p)}"
                entry["unequal"][name] = ctx.stat(split, c, name, outcome)
        for t in ctx.config.evaluation.durations_heldout:
            for term in ("direct", "partition", "rollout"):
                entry["decomposition"][f"heldout_{term}:T={t}"] = ctx.stat(split, c, f"heldout_{term}:T={t}", outcome)
        entry["triangle_violations"] = sum(cell["report"]["triangle"]["violations"]
                                           for (sp, cc, _), cell in ctx.got["cells"].items() if sp == split and cc == c)
        out[c] = entry
    return out


def frame_rate(ctx: Context, split: str, outcome=None) -> dict:
    names = [f"frame_rate:{n}" for n in ev.frame_rate_names(ctx.config)]
    out = ctx.table(split, names, outcome)
    for c in ctx.conditions:
        cells = [cell for (sp, cc, _), cell in sorted(ctx.got["cells"].items(), key=lambda kv: str(kv[0]))
                 if sp == split and cc == c and cell.get("frame_rate") is not None]
        out[c]["excluded"] = None if not cells else sum(r["excluded"] is not None for r in cells[0]["frame_rate"])
        out[c]["entries"] = None if not cells else len(cells[0]["frame_rate"])
    return out


def planning(ctx: Context, split: str, outcome=None) -> dict:
    """D6：排名三项（附随机水平）、CEM 的净位移误差、夹爪误差、相对误差（种子平均 ± SD；另给每个种子回合均值的中位数
    的种子平均），真实净位移 < 1 cm 的窗口数。"""
    e = ctx.config.evaluation
    out = dict(random_level=pl.random_level(e.planning_candidates), conditions={})
    for c in ctx.conditions:
        entry = {}
        for d in e.planning_durations:
            row = {}
            for key in RANK_KEYS:
                row[key] = ctx.stat(split, c, f"rank:{key}:d={d}", outcome)
            for key in CEM_KEYS:
                metric = f"cem:{key}:d={d}"
                stats = ctx.stat(split, c, metric, outcome)
                medians = {s: statistics.median(sa.by_trajectory(v).values()) if v else None
                           for s, v in ctx.per_seed(split, c, metric, outcome).items()}
                row[key] = dict(stats, median=sa.seed_stats(medians))
            small = [sum(r["small_displacement"] for r in cell["cem"] if r["duration"] == d
                         and (outcome is None or r["outcome"] == outcome))
                     for (sp, cc, _), cell in ctx.got["cells"].items() if sp == split and cc == c and cell.get("cem")]
            row["small_displacement_windows"] = small[0] if small else None
            entry[str(d)] = row
        out["conditions"][c] = entry
    return out


def split_tables(ctx: Context, split: str, errors: Mapping, outcome=None) -> dict:
    e = ctx.config.evaluation
    ids, held, long = e.durations_id, e.durations_heldout, e.durations_long
    by_t = [f"future_g16:T={t}" for t in ids] + [f"future_g8:T={t}" for t in ids] + \
           [f"tcd_ln:T={t}" for t in (*ids, *held)]
    return dict(
        D1=ctx.table(split, D1_METRICS, outcome), D1_by_duration=ctx.table(split, by_t, outcome),
        D2=ctx.table(split, D2_METRICS + tuple(f"future_g16:T={t}" for t in held)
                     + tuple(f"future_g12:T={t}" for t in ev.JOINT_HELDOUT), outcome),
        D3=ctx.table(split, [f"long_{k}:T={t}" for t in long for k in ("direct", "composed", "pd_ln", "pd_in")], outcome),
        D4=frame_rate(ctx, split, outcome),
        D5=ctx.table(split, [f"resolution:{n}" for n in ev.resolution_names(ctx.config)], outcome),
        D6=planning(ctx, split, outcome),
        D7=comparisons(ctx, split, errors, outcome),
        D8=partitions(ctx, split, outcome))


def cost(ctx: Context) -> dict:
    """D11：每个条件全部种子的每步用时中位数、显存峰值的最大值、GPU 小时之和、续跑次数之和、求谱后端（来自训练任务）。"""
    out = {}
    for c in ctx.config.training.conditions:
        views = [t for (cc, _), t in sorted(ctx.got["trains"].items()) if cc == c]
        seconds = [x for t in views for x in t.step_seconds]
        peaks = [t.peak_memory_bytes for t in views if t.peak_memory_bytes is not None]
        hours = [t.gpu_hours for t in views if t.gpu_hours is not None]
        out[c] = dict(jobs=len(views), median_step_seconds=statistics.median(seconds) if seconds else None,
                      peak_memory_bytes=max(peaks) if peaks else None, gpu_hours=math.fsum(hours) if hours else None,
                      resumes=sum(t.resumes for t in views) if views else None,
                      spectral_backend=sorted({t.spectral_backend for t in views if t.spectral_backend}) or None)
    return out


def spectral_gap(got: Mapping) -> dict | None:
    g = got["gap"]
    if g is None:
        return None
    return dict(summary=g["summary"], by_split=g["by_split"], fixed_fourier_cutoff_gap=g["fixed_fourier"]["cutoff_gap"])


def summarize(config: DroidConfig, got: Mapping) -> dict:
    checks = run_checks(config, got)
    failed = [c["item"] for c in checks if not c["ok"]]
    head = dict(schema=SCHEMA, config=dict(name=config.name, config_sha256=config.config_sha256),
                status="blocked" if failed else "ok", failed=failed,
                jobs=dict(eval=len(got["evals"]), train=len(got["trains"]), spectral_gap=got["gap"] is not None),
                missing=got["missing"], checks=checks, scientific_conclusions_verified=False)
    if failed:
        return head
    ctx = Context(config, got)
    errors = precision(ctx)
    tables = {split: {label: split_tables(ctx, split, errors, outcome)
                      for label, outcome in (("all", None), *((o, o) for o in OUTCOMES))} for split in SPLITS}
    scatter = {split: {c: {str(s): dict(error=sa.macro(v["future_g16"] or {}), tcd_ln=sa.macro(v["tcd_ln"] or {}),
                                        scd_ln=sa.macro(v["scd_ln"] or {}))
                           for s, v in _per_seed_many(ctx, split, c, ("future_g16", "tcd_ln", "scd_ln")).items()}
                       for c in ctx.conditions} for split in SPLITS}
    return dict(head, conditions=list(ctx.conditions), seeds=list(ctx.seeds),
                precision=dict(errors=errors, tolerance=sa.MATCH_TOLERANCE, split="id_test",
                               note="ood_test reuses the id_test matching (13 §8.2)"),
                bootstrap=dict(reps=sa.BOOTSTRAP_REPS, seed=sa.BOOTSTRAP_SEED,
                               clusters=dict(id_test="scene (lab/scene_id)", ood_test="episode"),
                               weighting="episode-weighted mean within the drawn clusters"),
                splits={split: dict(by_outcome=tables[split], scatter=scatter[split]) for split in SPLITS},
                spectral_gap=spectral_gap(got), cost=cost(ctx), check_env=got["env"],
                frame_rate_definitions=ev.frame_rate_definitions(config),
                definitions=dict(metrics={k: dict(source=v[0], parts=[[n, None if d is None else list(d)] for n, d in v[1]])
                                          for k, v in metric_specs(config).items()},
                                 not_comparable_for_ac=sorted(NOT_COMPARABLE),
                                 ac_interpolated_tcd=[f"tcd_ln:T={d}" for d in ev.AC_INTERPOLATED_TCD],
                                 epsilon_hat="median of tcd_normalized over all in-distribution windows of the cell"))


def _per_seed_many(ctx, split, c, metrics):
    tables = {m: ctx.per_seed(split, c, m, None) for m in metrics}
    seeds = sorted(set.intersection(*[set(t) for t in tables.values()]), key=str) if tables else []
    return {s: {m: tables[m][s] for m in metrics} for s in seeds}


# ---------------------------------------------------------------- 图（13 §8.4 最后一段）

def figures(summary: Mapping, curves: Mapping) -> dict[str, str]:
    files = {}
    if summary["status"] != "ok":
        return files
    for split in SPLITS:
        s = summary["splits"][split]
        for defect in ("tcd_ln", "scd_ln"):
            series = []
            for c in summary["conditions"]:
                points = sorted((p["error"], 0. if c == "identity" else p[defect]) for p in s["scatter"][c].values()
                                if p["error"] is not None and (c == "identity" or p[defect] is not None))
                if points:
                    series.append((c if c != "identity" else "identity（纵轴 0）", [x for x, _ in points], [y for _, y in points]))
            files[f"{split}-scatter-{defect}.svg"] = plots.line_chart(
                series, f"{split}：同分布误差—{defect}（每个种子一个点）", "同分布未来误差 e_c", defect)
        d8 = s["by_outcome"]["all"]["D8"]
        series = []
        for c in summary["conditions"]:
            if c in AC:
                continue
            points = sorted((r["x"], r["pd_in"]["mean"]) for r in d8[c]["partitions"] if r["pd_in"]["mean"] is not None)
            if points:
                series.append((c, [x for x, _ in points], [y for _, y in points]))
            eps = d8[c]["epsilon_hat"]["mean"]
            if c == FULL and eps is not None:
                series.append(("ε̂·x（po_jepa_full）", [0., 28.], [0., 28.*eps]))
        files[f"{split}-pd-vs-x.svg"] = plots.line_chart(series, f"{split}：pd_in 对 x(Π)", "x(Π)", "pd_in")
        d4_table = s["by_outcome"]["all"]["D4"]
        per = summary["frame_rate_definitions"]["frames_per_interval"]
        series = []
        for c in summary["conditions"]:
            for kind in ("single", "rollout"):
                points = sorted((int(k.split("=")[1])/per, v["mean"]) for k, v in d4_table[c].items()
                                if k.startswith(f"frame_rate:{kind}:") and isinstance(v, dict) and v.get("mean") is not None)
                if points:
                    series.append((f"{c} {kind}", [x for x, _ in points], [y for _, y in points]))
        files[f"{split}-frame-rate.svg"] = plots.line_chart(
            series, f"{split}：原始帧率下未来误差随步长 δ（周期）", "步长 δ（周期）", "未来误差")
    validation, loss = [], []
    for (c, seed), cv in sorted(curves.items()):
        if cv.get("validation"):
            validation.append((f"{c} {seed}", [x for x, _ in cv["validation"]], [y for _, y in cv["validation"]]))
        if cv.get("train_jepa"):
            loss.append((f"{c} {seed}", [x for x, _ in cv["train_jepa"]], [y for _, y in cv["train_jepa"]]))
    files["validation.svg"] = plots.line_chart(validation, "验证集未来误差", "更新步数", "future JEPA distance")
    files["train-jepa.svg"] = plots.line_chart(loss, "训练 JEPA 损失", "更新步数", "JEPA loss", log_y=True)
    return files


# ---------------------------------------------------------------- summary.md

def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _header(names: Sequence[str]) -> list[str]:
    return [_row(names), "|---"*len(names) + "|"]


def _table(title: str, table: Mapping, metrics: Sequence[str], conditions, label=None) -> list[str]:
    lines = ["", f"#### {title}", "", *_header(["条件", *(label(m) if label else m for m in metrics)])]
    lines += [_row([c, *(sa._ms(table[c][m]) for m in metrics)]) for c in conditions]
    return lines


def markdown(summary: Mapping) -> str:
    lines = [f"# DROID 汇总（{summary['config']['name']}）", "",
             f"评测任务 {summary['jobs']['eval']} 个、训练任务 {summary['jobs']['train']} 个；未经人工核对，不构成科学结论。", "",
             "## 数据一致性核对（13 §8.3）", ""]
    lines += [f"- {c['item']}：{'通过' if c['ok'] else '未通过：' + '；'.join(c['problems'][:5])}" for c in summary["checks"]]
    if summary["status"] != "ok":
        lines += ["", "核对未通过，不出表和图。", ""]
        return "\n".join(lines) + "\n"
    conds = summary["conditions"]
    errors = summary["precision"]["errors"]
    lines += ["", "## 精度匹配（id_test，16×16，同分布时长）", "", *_header(["条件", "e_c"])]
    lines += [_row([c, sa._f(errors[c])]) for c in conds]
    for split in SPLITS:
        t = summary["splits"][split]["by_outcome"]["all"]
        lines += ["", f"## {split}"]
        lines += _table("D1 主表（种子平均 ± 种子 SD；AC 的输入尺度量不可比）", t["D1"], D1_METRICS, conds)
        names = list(next(iter(t["D1_by_duration"].values())))
        lines += _table("D1 按时长（AC 的 tcd_ln:T=1、T=3 是插值，10 §8.6）", t["D1_by_duration"], names, conds)
        names = list(next(iter(t["D2"].values())))
        lines += _table("D2 留出", t["D2"], names, conds)
        names = list(next(iter(t["D3"].values())))
        lines += _table("D3 长时程（identity 是参照）", t["D3"], names, conds)
        windows = {m: max(next(iter(t["D3"].values()))[m]["windows"].values(), default=0)
                   if isinstance(next(iter(t["D3"].values()))[m], dict) else None for m in names}
        lines += ["", "用到的窗口数：" + "，".join(f"{m} {w}" for m, w in windows.items())]
        names = [k for k in next(iter(t["D4"].values())) if k.startswith("frame_rate:")]
        lines += _table("D4 原始帧率（single/rollout:f=15 是 δ = 1 的参照）", t["D4"], names, conds)
        lines += ["", "排除的起点（控制越界）：" + "，".join(f"{c} {t['D4'][c]['excluded'] if t['D4'][c]['excluded'] is not None else '缺'}"
                                               f"/{t['D4'][c]['entries'] if t['D4'][c]['entries'] is not None else '缺'}" for c in conds)]
        names = list(next(iter(t["D5"].values())))
        lines += _table("D5 原始分辨率（future:g=16 是同一批窗口的 16×16 对照）", t["D5"], names, conds)
        p = t["D6"]
        r = p["random_level"]
        lines += ["", "#### D6 规划", "", f"随机水平：平均名次 {sa._f(r['mean_rank'])}，第一名 {sa._f(r['top1'])}，"
                  f"倒数名次 {sa._f(r['reciprocal_rank'])}。相对误差只取位置，不含真实净位移 < 1 cm 的窗口。", ""]
        lines += _header(["条件", "d", *RANK_KEYS, *CEM_KEYS, "CEM 中位数（位置/夹爪/相对）", "< 1 cm 窗口"])
        for c in conds:
            for d, row in p["conditions"][c].items():
                med = " / ".join(sa._ms(row[k]["median"]) if isinstance(row[k], dict) else "缺" for k in CEM_KEYS)
                small = row["small_displacement_windows"]
                lines.append(_row([c, d, *(sa._ms(row[k]) for k in RANK_KEYS), *(sa._ms(row[k]) for k in CEM_KEYS), med,
                                   "缺" if small is None else str(small)]))
        lines += ["", "#### D7 配对（po_jepa_full − 对照；负数 = full 更低；ood_test 沿用 id_test 的精度匹配）", "",
                  *_header(["对照", "量", "精度匹配", "误差相差 %", "种子平均差", "种子 SD", "95% CI（整群）", "full 更低的种子"])]
        for row in t["D7"]:
            match = sa._match_label(row)
            if row["status"] != "ok":
                lines.append(_row([row["reference"], row["metric"], match, sa._f(row["error_difference_percent"], 1),
                                   "不可比" if row["status"] == "not_comparable" else "缺", "", "", ""]))
                continue
            ci = "缺" if row["ci"] is None else f"[{sa._f(row['ci']['low'])}, {sa._f(row['ci']['high'])}]"
            lines.append(_row([row["reference"], row["metric"], match, sa._f(row["error_difference_percent"], 1),
                               sa._f(row["mean"]), sa._f(row["seed_sd"]), ci, f"{row['seeds_full_lower']}/{len(row['per_seed'])}"]))
        d8 = t["D8"]
        parts = d8[conds[0]]["partitions"]
        lines += ["", "#### D8 分割差异（pd_in；括号里是有效斜率 pd_in/(x·ε̂)）和误差分解", "",
                  *_header(["条件", "ε̂", *(f"T={_label(r['T'])} {r['partition']} (x={_label(r['x'])})" for r in parts)])]
        lines += [_row([c, sa._ms(d8[c]["epsilon_hat"]), *(f"{sa._ms(r['pd_in'])}（{sa._ms(r['effective_slope'])}）"
                                                          for r in d8[c]["partitions"])]) for c in conds]
        lines += ["", *_header(["条件", *(f"T={_label(r['T'])} {r['partition']} pd_ln" for r in parts)])]
        lines += [_row([c, *(sa._ms(r["pd_ln"]) for r in d8[c]["partitions"])]) for c in conds]
        names = list(d8[conds[0]]["cross_grid"]) + list(d8[conds[0]]["unequal"])
        lines += ["", *_header(["条件", *names])]
        lines += [_row([c, *(sa._ms({**d8[c]["cross_grid"], **d8[c]["unequal"]}[n]) for n in names)]) for c in conds]
        names = list(d8[conds[0]]["decomposition"])
        lines += ["", *_header(["条件", *names, "三角不等式违反"])]
        lines += [_row([c, *(sa._ms(d8[c]["decomposition"][n]) for n in names), str(d8[c]["triangle_violations"])]) for c in conds]
        lines += ["", "#### D9 成功和失败（D1、D2 的主要指标）"]
        for outcome in OUTCOMES:
            o = summary["splits"][split]["by_outcome"][outcome]
            lines += _table(f"{outcome}：D1", o["D1"], D1_METRICS, conds)
            lines += _table(f"{outcome}：D2", o["D2"], D2_METRICS, conds)
    lines += ["", "## D10 截断间隙（10 §8.8）", ""]
    g = summary["spectral_gap"]
    if g is None:
        lines.append("缺")
    else:
        lines += _header(["网格", "窗口", "ratio 分位数（5/25/50/75/95）", "中位数 > 1"])
        for size, e in g["summary"]["grids"].items():
            q = "缺" if e["ratio"] is None else ", ".join(sa._f(v) for v in e["ratio"])
            lines.append(_row([f"{size}×{size}", str(e["windows"]), q, "是" if e["median_ratio_above_1"] else "否"]))
        lines += [_row(["fixed_fourier", "—", f"截断间隙恒为 {sa._f(g['fixed_fourier_cutoff_gap'])}（08 §4.3）", "—"]), "",
                  f"判断（两个测试集合起来）：{g['summary']['decision']}"]
    lines += ["", "## D11 成本（训练日志）", "", *_header(["条件", "任务", "每步用时中位数（秒）", "显存峰值（字节）",
                                                    "GPU 小时", "续跑次数", "求谱后端"])]
    for c, e in summary["cost"].items():
        lines.append(_row([c, str(e["jobs"]), sa._f(e["median_step_seconds"]), str(e["peak_memory_bytes"] or "缺"),
                           sa._f(e["gpu_hours"]), "缺" if e["resumes"] is None else str(e["resumes"]),
                           "、".join(e["spectral_backend"] or ["缺"])]))
    if summary["missing"]:
        lines += ["", f"缺 {len(summary['missing'])} 项，见 missing.json。"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 输出

def write_output(output: Path, summary: Mapping, svgs: Mapping[str, str]) -> Path:
    """13 §7：先写临时目录再整体换名；已有的输出改名为 <输出>.previousN。核对全部通过时 COMPLETE.json 记 summary.json 的 SHA。"""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    cache._write_json(temporary/"summary.json", summary)
    cache._write_json(temporary/"checks.json", dict(checks=summary["checks"]))
    cache._write_json(temporary/"missing.json", dict(missing=summary["missing"]))
    (temporary/"summary.md").write_text(markdown(summary), encoding="utf-8")
    if summary["status"] == "ok":
        (temporary/"figures").mkdir()
        for name, svg in sorted(svgs.items()):
            (temporary/"figures"/name).write_text(svg, encoding="utf-8")
        cache._write_json(temporary/"COMPLETE.json", dict(summary_sha256=cache.file_sha(temporary/"summary.json")))
    if output.exists():
        n = 1
        while output.with_name(f"{output.name}.previous{n}").exists():
            n += 1
        output.rename(output.with_name(f"{output.name}.previous{n}"))
    temporary.rename(output)
    return output


def run(config: DroidConfig, data_root: Path, run_root: Path, readers: Readers = Readers()) -> tuple[dict, Path]:
    got = collect(config, data_root, run_root, readers)
    summary = summarize(config, got)
    curves = {(c, s): t.curves for (c, s), t in got["trains"].items()}
    output = write_output(Path(run_root)/config.name/"summary", summary, figures(summary, curves))
    return summary, output


def main(argv: list[str] | None = None, readers: Readers = Readers()) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--run-root", required=True, type=Path)
    a = p.parse_args(argv)
    config = load_config(a.config)
    summary, output = run(config, a.data_root, a.run_root, readers)
    print(f"{summary['jobs']['eval']} 个评测任务；核对{'全部通过' if summary['status'] == 'ok' else '未通过 ' + '、'.join(summary['failed'])}；"
          f"缺 {len(summary['missing'])} 项 → {output}")
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
