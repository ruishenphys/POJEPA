"""DROID 所有阶段的统一入口（docs/design/12-droid-jobs.md §3；S14）。

  python tools/droid.py <阶段> --config configs/droid/<name>.yaml --data-root <数据根> --run-root <运行根> \\
      --job <任务编号> [--delete-raw] [--time-limit-minutes <上限>] [--retry]

- 任务编号必须在本配置的任务图里（`tools/droid_jobs.py`），阶段必须是编号的第一段。
- 开始时完成标志已存在：index、download、prepare、encode、calibrate、check 什么都不做，退出 0（这些产物只写一次，
  下游读的时候核身份）；train、eval、summary（`droid_jobs.RECHECKED_STAGES`）照常调用阶段，由阶段先核本次请求的身份
  再决定复用：训练核 COMPLETE.json 绑定 report.json，评测核 source.json、job.json 与本次请求相同（不同报错退出 1；
  S11b-fix1、S11-P1-1 修复），汇总按当前产物重算（内容和已有输出相同就不换）。只有完成标志、没有报告的目录不算完成。完成标志由各阶段自己最后写（先写临时文件再改名）；阶段返回后完成标志
  不存在算失败（退出 1）。训练片的编码任务（带 `--delete-raw`）的完成标志是 `features/<片>/_raw_deleted/<name>.json`，
  原始文件删完才写；特征已编完但没删完时重跑会进入编码阶段，跳过编码、把剩下的删掉（S14-fix1）。
- 退出码：0 完成；75 已存档、要重排同一条命令；其他非 0 是失败（09 §4.7）。阶段抛出的异常退出 1。
- 全部阶段都已接上：index（02，S02/S03）、download（03，S04）、prepare（04，S05）、encode（05，S06a/S06b）、calibrate
  （09 §3，S09b，`droid.calibrate.main`）、check（09 §7，S09b，`droid.check.main`）、train（09 §4，S09a，
  `droid.train.main`；`--retry` 删掉 FAILED.json 从最新检查点接着跑）、eval（`eval/spectral-gap` 是 10 §8.8 的
  `droid.spectral_gap.main`，S11b；其余是 10 §8、11 的 `droid.eval_job.main`，S09b）、summary（13 §8，
  `tools/droid_summary.py`，S13/S09b）。
- 编码任务开始时由 `encode.load_frozen` 核上游提交（204698b）和编码器权重 SHA（= v7 `cache.CHECKPOINT_SHA`），
  不对就报错退出 1（12 §1）。`--fake-encoder` 只给 CPU 流程测试用（特征标成 synthetic_development_only）。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import droid_jobs  # noqa: E402
import droid_summary  # noqa: E402
from po_jepa.droid import calibrate as cb  # noqa: E402
from po_jepa.droid import check as ck  # noqa: E402
from po_jepa.droid import download as dl  # noqa: E402
from po_jepa.droid import eval_job as ej  # noqa: E402
from po_jepa.droid import encode as en  # noqa: E402
from po_jepa.droid import index as ix  # noqa: E402
from po_jepa.droid import prepare as pp  # noqa: E402
from po_jepa.droid import spectral_gap as sg  # noqa: E402
from po_jepa.droid import train as tr  # noqa: E402
from po_jepa.droid.config import load_config  # noqa: E402

NOT_IMPLEMENTED = 2
PENDING: dict[str, str] = {}       # 阶段整体还没接上的（S09b 之后没有了）
assert set(PENDING) | {"index", "download", "prepare", "encode", "calibrate", "check", "train", "eval",
                       "summary"} == set(droid_jobs.STAGES)


def stage_argv(stage: str, rest: str, a: argparse.Namespace) -> list[str]:
    """把任务翻译成各阶段模块 `main()` 的参数（S02–S06 的命令行不变，这里只包一层）。"""
    base = ["--config", str(a.config), "--data-root", str(a.data_root)]
    if stage == "index":
        return base
    if stage in ("download", "prepare"):
        return base + ["--shard", rest]
    if stage == "encode":
        argv = base + ["--shard", rest, "--device", a.device]
        if a.delete_raw:
            argv.append("--delete-raw")
        if a.fake_encoder:
            argv.append("--fake-encoder")
        return argv
    run = base + ["--run-root", str(a.run_root)]
    if stage == "calibrate":
        return run + ["--job", a.job, "--device", a.device]
    if stage == "check":
        return run + ["--device", a.device]
    if stage in ("train", "eval"):
        argv = run + ["--job", a.job, "--device", a.device]
        if a.time_limit_minutes is not None:
            argv += ["--time-limit-minutes", str(a.time_limit_minutes)]
        if a.retry:
            argv.append("--retry")
        return argv
    if stage == "summary":
        return run
    raise AssertionError(stage)


MAINS = {"index": lambda argv: ix.main(argv), "download": lambda argv: dl.main(argv),
         "prepare": lambda argv: pp.main(argv), "encode": lambda argv: en.main(argv),
         "calibrate": lambda argv: cb.main(argv), "check": lambda argv: ck.main(argv),
         "train": lambda argv: tr.main(argv), "eval": lambda argv: ej.main(argv),
         "summary": lambda argv: droid_summary.main(argv)}
# 同一阶段里另有入口的任务（S11b：截断间隙只读特征和 sigma_z.json）
JOB_MAINS = {f"eval/{droid_jobs.SPECTRAL_GAP}": lambda argv: sg.main(argv)}
assert all(job.partition("/")[0] in droid_jobs.RECHECKED_STAGES for job in JOB_MAINS)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=droid_jobs.STAGES)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--job", required=True, help="任务编号，如 encode/train-007（12 §2）")
    p.add_argument("--delete-raw", action="store_true", help="训练片的编码任务（03 §4）")
    p.add_argument("--time-limit-minutes", type=int, help="训练和评测任务的时长上限（09 §4.5）")
    p.add_argument("--retry", action="store_true", help="训练任务：删掉 FAILED.json，从最新检查点接着跑（09 §4.3）")
    p.add_argument("--device", default="cuda", help="编码和训练用的设备（默认 cuda；排队系统用 CUDA_VISIBLE_DEVICES 分卡）")
    p.add_argument("--fake-encoder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    stage, _, rest = a.job.partition("/")
    if stage != a.stage:
        p.error(f"job {a.job!r} belongs to stage {stage!r}, not {a.stage!r}")
    config = load_config(a.config)
    if a.job == "index":     # 索引任务不需要片名（use_failures: false 的片名要先有索引）
        job = {"id": "index", "command": [], "complete": str(droid_jobs.complete_path("index", config, a.data_root,
                                                                                    a.run_root))}
    else:
        graph = droid_jobs.build_graph(config, str(a.config), str(a.data_root), str(a.run_root))
        job = droid_jobs.find_job(graph, a.job)
    if a.delete_raw and "--delete-raw" not in job["command"]:
        p.error(f"--delete-raw is only for the encoding jobs of training shards, not {a.job}")
    if a.time_limit_minutes is not None and stage not in ("train", "eval"):
        p.error("--time-limit-minutes is only for train and eval jobs")
    if a.retry and stage != "train":
        p.error("--retry is only for train jobs")
    done = Path(job["complete"])
    if done.exists() and stage not in droid_jobs.RECHECKED_STAGES:
        print(f"{a.job}: already complete ({done})")
        return 0
    if a.job in JOB_MAINS:
        code = JOB_MAINS[a.job](["--config", str(a.config), "--data-root", str(a.data_root), "--run-root", str(a.run_root)])
    elif stage in PENDING:
        print(f"{a.job}: stage {stage!r} is not implemented yet ({PENDING[stage]}); nothing was written",
              file=sys.stderr)
        return NOT_IMPLEMENTED
    else:
        code = MAINS[stage](stage_argv(stage, rest, a))
    if code != 0:
        return code
    if not done.exists():
        print(f"{a.job}: the stage returned 0 but did not write its completion flag {done}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
