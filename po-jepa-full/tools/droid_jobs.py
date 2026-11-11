"""DROID 任务图（docs/design/12-droid-jobs.md §2–§4、§5.3；S14）。

  python tools/droid_jobs.py --config configs/droid/full.yaml --data-root /data/droid-1.0.1 \\
      --run-root /runs/po-jepa-droid --output jobs-full.json
  python tools/droid_jobs.py --config configs/droid/pilot.yaml --data-root … --run-root … --format list > jobs-pilot.txt

- 任务图是一份 JSON（schema `droid_jobs_v1`），只由配置、配置文件路径、两个根目录决定，不含任何集群相关的东西；
  同样的参数两次输出逐字节相同（`cache._json_bytes`）。
- 片：`use_failures: true` 时训练片是 `train-000` … `train-<⌈N/shard_size⌉−1>`，不用读索引；数据根里已有
  `index/splits.json` 时改用 `splits.shard_plan`（并核对两者一致）。`use_failures: false` 的片号取决于失败回合落在
  哪里，必须先有索引。之后是 `val`、`id_test`、`ood_test`（02 §6）。
- 完成标志的路径取自各阶段模块自己的路径函数（download/prepare/features），运行根下的按 12 §2 的表。
- `--format list` 输出 v7 那种按 “# =====” 分段的纯文本，每段是一个拓扑层（12 §5.3）。

`tools/droid.py`（统一入口）和 `tools/droid_run.py`（本机运行器）用本模块的 `build_graph`、`find_job`。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from po_jepa.droid import download as dl  # noqa: E402
from po_jepa.droid import features as ft  # noqa: E402
from po_jepa.droid import index as ix  # noqa: E402
from po_jepa.droid import prepare as pp  # noqa: E402
from po_jepa.droid import splits as sp  # noqa: E402
from po_jepa.droid.config import CONDITIONS, DroidConfig, load_config  # noqa: E402
from po_jepa.runtime import cache  # noqa: E402

SCHEMA = "droid_jobs_v1"
ENTRY = "tools/droid.py"
REQUEUE_EXIT_CODES = (75,)                      # 09 §4.7：已存档、要重排同一条命令
TEST_EXTRA = "test-extra"                       # encode_extra.JOB
STATS = "stats"
LAMBDA_CONDITIONS = ("po_jepa_full", "fixed_fourier", "transformer_consistency")   # 09 §1、§3.2：带一致性的三个
UNTRAINED = ("identity", "vjepa2_ac_zeroshot")  # 12 §2 不训练的对照（= base.yaml training.untrained）
SPECTRAL_GAP = "spectral-gap"
# 12 §4：训练任务的 priority（越小越先排）。vjepa2_ac 最慢；PO-JEPA 类（08：fixed_fourier 也是）和 AC 缩小版其次；
# Transformer 类和 Koopman 最后。其他任务的 priority 是拓扑层次。
TRAIN_PRIORITY = {"vjepa2_ac": 1,
                  "po_jepa_full": 2, "po_jepa_no_consistency": 2, "fixed_fourier": 2, "vjepa2_ac_small": 2,
                  "matched_transformer": 3, "transformer_consistency": 3, "koopman": 3}
STAGES = ("index", "download", "prepare", "encode", "calibrate", "check", "train", "eval", "summary")
# 完成标志在时仍要调用阶段、由阶段先核本次请求的身份再决定复用的阶段（S11-P1-1 修复）：训练（COMPLETE.json 绑定
# report.json）、评测（job.json/source.json 的请求身份，10 §8.9）、汇总（每次按当前产物重算，内容不变就不换输出）、
# 环境检查（已有 env.json 的配置、特征总清单、σ_z 文件 SHA 等于当前的才算完成，RF-P2-4；只读几个清单，不重跑检查）。
# 其余阶段的完成标志所在的文件只写一次，下游读它时核身份（特征清单 SHA、σ_z/λ 的 config、条件、SHA）。
RECHECKED_STAGES = ("check", "train", "eval", "summary")
RESOURCE_KEYS = ("gpus", "cpus", "mem_gb", "time_limit_minutes")

assert set(TRAIN_PRIORITY) == set(CONDITIONS) and set(LAMBDA_CONDITIONS) <= set(CONDITIONS)


# ---------------------------------------------------------------- 片

def shard_names(config: DroidConfig, data_root: Path | str, splits: dict | None = None) -> list[str]:
    """本配置的片名：训练片按编号，之后 val、id_test、ood_test（02 §6）。

    `splits` 不给时：数据根里有 `index/splits.json` 就读它（`splits.load_splits` 核过清单），否则只在
    `use_failures: true` 时按 ⌈N/shard_size⌉ 推出训练片（`shard_plan` 在这种情况下恰好是这么多片）。"""
    data = config.data
    if splits is None and (ix.index_dir(data_root)/ix.SPLITS_FILE).exists():
        splits = sp.load_splits(data_root, config)
    computed = None
    if data.use_failures:
        count = -(-data.train_episodes//data.shard_size)
        computed = [sp.shard_name(k) for k in range(count)] + list(sp.TEST_SHARDS)
    if splits is None:
        if computed is None:
            raise ValueError("data.use_failures is false: the training shards depend on where the failed episodes "
                             f"are, so {ix.index_dir(data_root)/ix.SPLITS_FILE} must exist (run the index job of "
                             "the main config first)")
        return computed
    planned = list(sp.shard_plan(splits, data))
    if computed is not None and planned != computed:
        raise ValueError(f"splits.shard_plan gives shards {planned}, expected {computed} for use_failures: true")
    return planned


def is_train_shard(shard: str) -> bool:
    return shard.startswith(sp.SHARD_PREFIX)


# ---------------------------------------------------------------- 完成标志和资源

def run_dir(run_root: Path | str, config: DroidConfig) -> Path:
    """运行根下本配置的子目录 `<run-root>/<name>/`（12 §1）。"""
    return Path(run_root)/config.name


def log_path(run_root: Path | str, config_name: str, job_id: str) -> Path:
    """12 §3：`<运行根>/<name>/logs/<任务编号的 / 换成 __>.log`。"""
    return Path(run_root)/config_name/"logs"/(job_id.replace("/", "__") + ".log")


def complete_path(job_id: str, config: DroidConfig, data_root: Path | str, run_root: Path | str) -> Path:
    """12 §2 表的完成标志列。"""
    stage, _, rest = job_id.partition("/")
    run = run_dir(run_root, config)
    if job_id == "index":
        return ix.index_dir(data_root)/ix.MANIFEST_FILE
    if stage == "download":
        return dl.complete_path(data_root, rest, config)
    if job_id == "prepare/stats":
        return pp.prepared_dir(data_root)/pp.COMPLETE_DIR/f"stats-{config.name}.json"   # = prepare.run_stats 的 done
    if stage == "prepare":
        return pp.complete_path(data_root, rest, config)
    if job_id == f"encode/{TEST_EXTRA}":
        return ft.extra_complete_path(data_root, config)
    if stage == "encode" and is_train_shard(rest):   # 带 --delete-raw：删完原始文件才算完成（S14-fix1）
        return ft.raw_cleanup_path(data_root, rest, config)
    if stage == "encode":
        return ft.complete_path(data_root, rest, config)
    if job_id == "calibrate/sigma-z":
        return run/"calibration"/"sigma_z.json"
    if stage == "calibrate" and rest.startswith("lambda-"):
        return run/"calibration"/f"{rest}.json"
    if job_id == "check/env":
        return run/"check"/"env.json"
    if stage in ("train", "eval"):
        return run/stage/Path(rest)/"COMPLETE.json"
    if job_id == "summary":
        return run/"summary"/"COMPLETE.json"
    raise ValueError(f"unknown job {job_id!r}")


def resource_key(job_id: str) -> str:
    """12 §4：资源取 `resources.<阶段>`；sigma-z 和 spectral-gap 用 stats 的（CPU）。"""
    stage = job_id.partition("/")[0]
    if job_id in ("prepare/stats", "calibrate/sigma-z", f"eval/{SPECTRAL_GAP}"):
        return "stats"
    return {"eval": "evaluate"}.get(stage, stage)


def resources(config: DroidConfig, job_id: str) -> dict:
    block = getattr(config.resources, resource_key(job_id))
    return {k: getattr(block, k) for k in RESOURCE_KEYS}


# ---------------------------------------------------------------- 任务图

def job_deps(config: DroidConfig, shards: list[str]) -> dict[str, list[str]]:
    """12 §2 的任务和依赖（插入顺序无关紧要，输出按拓扑层次和编号排）。"""
    t = config.training
    deps: dict[str, list[str]] = {"index": []}
    for s in shards:
        deps[f"download/{s}"] = ["index"]
        deps[f"prepare/{s}"] = [f"download/{s}"]
        deps[f"encode/{s}"] = [f"prepare/{s}"]
    deps["prepare/stats"] = [f"prepare/{s}" for s in shards]
    deps[f"encode/{TEST_EXTRA}"] = ["prepare/id_test", "prepare/ood_test"]
    deps["calibrate/sigma-z"] = ["prepare/stats"] + [f"encode/{s}" for s in shards]
    deps["check/env"] = ["calibrate/sigma-z"]
    for c in LAMBDA_CONDITIONS:
        deps[f"calibrate/lambda-{c}"] = ["calibrate/sigma-z"]
    evals = []
    for c in t.conditions:
        for seed in t.seeds:
            train = f"train/{c}/seed{seed}"
            deps[train] = ["check/env"] + ([f"calibrate/lambda-{c}"] if c in LAMBDA_CONDITIONS else [])
            deps[f"eval/{c}/seed{seed}"] = [train, f"encode/{TEST_EXTRA}"]
            evals.append(f"eval/{c}/seed{seed}")
    if tuple(t.untrained) != UNTRAINED:
        raise ValueError(f"training.untrained must be {list(UNTRAINED)} (12 §2)")
    for c in t.untrained:
        deps[f"eval/{c}"] = ["check/env", f"encode/{TEST_EXTRA}"]
        evals.append(f"eval/{c}")
    deps[f"eval/{SPECTRAL_GAP}"] = ["calibrate/sigma-z"]
    evals.append(f"eval/{SPECTRAL_GAP}")
    deps["summary"] = sorted(evals)
    return {k: sorted(v) for k, v in deps.items()}


def levels(deps: dict[str, list[str]]) -> dict[str, int]:
    """拓扑层次：没有依赖的是 0，否则 = 1 + 依赖的最大层次。依赖不存在或有环就报错。"""
    for job, ds in deps.items():
        missing = [d for d in ds if d not in deps]
        if missing:
            raise ValueError(f"job {job} depends on unknown jobs {missing}")
    level: dict[str, int] = {}
    state: dict[str, int] = {}          # 1 = 正在访问，2 = 已完成

    def visit(job: str, path: list[str]) -> int:
        if state.get(job) == 2:
            return level[job]
        if state.get(job) == 1:
            raise ValueError(f"dependency cycle: {' -> '.join(path[path.index(job):] + [job])}")
        state[job] = 1
        level[job] = 1 + max((visit(d, path + [job]) for d in deps[job]), default=-1)
        state[job] = 2
        return level[job]

    for job in sorted(deps):
        visit(job, [])
    return level


def check_graph(jobs: list[dict], shards: list[str]) -> None:
    """12 §4 生成时核对：编号唯一、每个依赖都存在、没有环、完成标志互不相同、每个片都有下载/准备/编码三个任务、
    jobs 按（拓扑层次，编号）排好。"""
    ids = [j["id"] for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate job ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    level = levels({j["id"]: list(j["deps"]) for j in jobs})
    flags = [j["complete"] for j in jobs]
    if len(set(flags)) != len(flags):
        raise ValueError(f"completion flags are not distinct: {sorted({f for f in flags if flags.count(f) > 1})}")
    for s in shards:
        missing = [f"{stage}/{s}" for stage in ("download", "prepare", "encode") if f"{stage}/{s}" not in level]
        if missing:
            raise ValueError(f"shard {s} lacks jobs {missing}")
    if ids != sorted(ids, key=lambda i: (level[i], i)):
        raise ValueError("jobs are not in (topological level, id) order")


def command(job_id: str, config_path: str, data_root: str, run_root: str, res: dict) -> list[str]:
    """12 §3：`python tools/droid.py <阶段> --config … --data-root … --run-root … --job <编号>`；训练片的编码加
    `--delete-raw`（03 §4）；训练和评测加 `--time-limit-minutes`。"""
    stage, _, rest = job_id.partition("/")
    argv = ["python", ENTRY, stage, "--config", config_path, "--data-root", data_root, "--run-root", run_root,
            "--job", job_id]
    if stage == "encode" and is_train_shard(rest):
        argv.append("--delete-raw")
    if stage in ("train", "eval"):
        argv += ["--time-limit-minutes", str(res["time_limit_minutes"])]
    return argv


def build_graph(config: DroidConfig, config_path: str, data_root: str, run_root: str, *,
                splits: dict | None = None) -> dict:
    """12 §4 的任务图。`config_path`、`data_root`、`run_root` 原样写进命令（命令在 `po-jepa-full/` 下运行）。"""
    shards = shard_names(config, data_root, splits)
    deps = job_deps(config, shards)
    level = levels(deps)
    jobs = []
    for job_id in sorted(deps, key=lambda i: (level[i], i)):
        res = resources(config, job_id)
        stage, _, rest = job_id.partition("/")
        priority = TRAIN_PRIORITY[rest.partition("/")[0]] if stage == "train" else level[job_id]
        jobs.append({"id": job_id, "stage": stage,
                     "command": command(job_id, config_path, data_root, run_root, res),
                     "resources": res, "deps": deps[job_id],
                     "complete": str(complete_path(job_id, config, data_root, run_root)),
                     "requeue_exit_codes": list(REQUEUE_EXIT_CODES), "priority": priority})
    check_graph(jobs, shards)
    return {"schema": SCHEMA, "config": config.name, "config_sha256": config.config_sha256,
            "data_root": data_root, "run_root": run_root, "jobs": jobs}


def find_job(graph: dict, job_id: str) -> dict:
    for job in graph["jobs"]:
        if job["id"] == job_id:
            return job
    raise ValueError(f"job {job_id!r} is not in the task graph of config {graph['config']!r} "
                     f"(jobs: {', '.join(j['id'] for j in graph['jobs'][:6])}, …)")


def graph_levels(graph: dict) -> dict[str, int]:
    return levels({j["id"]: list(j["deps"]) for j in graph["jobs"]})


def to_list(graph: dict) -> str:
    """12 §5.3：按 “# =====” 分段，每段是一个拓扑层，段内命令互不依赖。"""
    level = graph_levels(graph)
    lines = [f"# DROID 任务列表（12 §5.3）：配置 {graph['config']}，config_sha256 {graph['config_sha256']}，"
             f"{len(graph['jobs'])} 个任务。",
             "# 在 po-jepa-full/ 下按段运行：一段全部完成（退出 0）后再跑下一段；退出 75 = 已存档，重跑同一条命令。"]
    for k in range(max(level.values()) + 1):
        layer = [j for j in graph["jobs"] if level[j["id"]] == k]
        stages = sorted({j["stage"] for j in layer}, key=STAGES.index)
        gpu = sum(j["resources"]["gpus"] > 0 for j in layer)
        lines.append(f"# ===== L{k} {'、'.join(stages)}（{len(layer)} 条，GPU {gpu} 条）")
        lines += [shlex.join(j["command"]) for j in layer]
    return "\n".join(lines) + "\n"


def _write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_bytes(body)
    os.replace(temp, path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configs/droid/<name>.yaml（原样写进每条命令）")
    p.add_argument("--data-root", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--format", choices=("json", "list"), default="json")
    p.add_argument("--output", type=Path, help="不给就写到标准输出")
    a = p.parse_args(argv)
    graph = build_graph(load_config(a.config), a.config, a.data_root, a.run_root)
    body = cache._json_bytes(graph) if a.format == "json" else to_list(graph).encode("utf-8")
    if a.output is None:
        sys.stdout.buffer.write(body)
        sys.stdout.flush()
    else:
        _write(a.output, body)
        print(f"{a.output}: {len(graph['jobs'])} jobs ({graph['config']}, config_sha256 {graph['config_sha256'][:12]})",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
