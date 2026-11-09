"""run.py d4 和 run.py spectral-gap 的子进程入口（10 §3–§5、§7；16 §5）。

run.py 把这两个命令交给 python -m po_jepa.runtime.addon.eval_cli（其余命令仍交给 po_jepa.runtime.cli），参数和其他命令
相同：--features-root --device --output <run.py 的 --output>/run --seed，d4 另有 --condition --training-run。
和 workflow.execute 的做法相同：先核清单（verify_bundle），新建 --output，写 environment.json（verify_versions），
核 v7 标定（和补跑标定），跑任务到 <--output>/job/，结束时清单没变才写 <--output>/COMPLETE.json；出错写 FAILED.json。
d4 要显式的 cuda:N；spectral-gap 只在 CPU 上跑。
"""
import argparse
import os
from pathlib import Path
import traceback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("d4", "spectral-gap"))
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1703, 1704, 1705), default=1703)
    parser.add_argument("--condition")
    parser.add_argument("--training-run", type=Path)
    args = parser.parse_args(argv)
    if args.command == "d4" and (args.condition is None or args.training_run is None):
        parser.error("d4 需要 --condition 和 --training-run")
    if args.command == "spectral-gap" and (args.condition is not None or args.training_run is not None or args.seed != 1703):
        parser.error("spectral-gap 不接受 --condition、--training-run，种子只用 1703")
    from ..checks import SEED_ENV
    os.environ[SEED_ENV] = str(args.seed)
    return execute(args)


def execute(args):
    import torch
    from .. import cache, environment, workflow
    from . import calibration, d4, spectral_gap
    bundle = workflow.verify_bundle()
    device = torch.device(args.device)
    if args.command == "d4" and (device.type != "cuda" or device.index is None):
        raise ValueError("explicit cuda:N target required")
    if args.command == "spectral-gap" and device.type != "cpu":
        raise ValueError("spectral-gap runs on CPU: --device cpu")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage = "environment"
    try:
        cache._write_json(output/"environment.json", environment.verify_versions())
        stage = "calibration"
        workflow.require_calibrated_weights(swept=True)
        if args.condition in calibration.CALIBRATED_CONDITIONS:
            calibration.require_addon_calibration()
        stage = args.command
        if args.command == "d4":
            d4.run_d4(Path(args.features_root), output/"job", device, args.condition, training_run=args.training_run)
        else:
            spectral_gap.run_spectral_gap(Path(args.features_root), output/"job", device)
        if workflow.verify_bundle() != bundle:
            raise ValueError("source bundle changed during run")
        cache._write_json(output/"COMPLETE.json", dict(command=args.command, condition=args.condition, manifest_sha256=bundle,
                                                       scientific_conclusions_verified=False))
        return 0
    except Exception as error:
        cache._write_json(output/"FAILED.json", dict(stage=stage, error_type=type(error).__name__, message=str(error),
                                                     traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
