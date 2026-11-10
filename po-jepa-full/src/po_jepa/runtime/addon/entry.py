"""run.py 的补跑子命令 calibrate-addon、check-addon（16 §3.2、§5）。v7 的 cli.py 和 workflow.execute 不动，
run.py 把这两个子命令交给本模块（python -m po_jepa.runtime.addon.entry …），参数和 cli.py 的相同。

--output 是 run.py 传来的 <输出>/run，必须不存在：
  calibrate-addon：calibration-fixed_fourier.json、calibration-transformer_consistency.json、COMPLETE.json，
                   另写 configs/calibration/addon/<tag>-u<步数>.yaml（和 v7 的 calibrate 一样不核 CUDA 设备）
  check-addon：environment.json、check-addon.json、gpu-inventory-*、COMPLETE.json（要求 cuda:N，和 v7 的 check 相同）
失败时写 FAILED.json（字段和 workflow.execute 的相同）并以非零退出码结束。
"""
import argparse
import os
from pathlib import Path
import traceback

COMMANDS = ("calibrate-addon", "check-addon")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1703,), default=1703, help="只用种子 1703（三个种子共用一组 λ，和 v7 的 calibrate 相同）")
    parser.add_argument("--base-calibration", choices=("v7-import", "local"), default="v7-import",
                        help="S26a：基础标定的来源。v7-import（默认）：tools/import_v7_calibration.py 导入的 v7 标定；"
                             "local：本目录 run.py calibrate 写的（v8 任务表 --stage addon）")
    args = parser.parse_args(argv)
    from ..checks import SEED_ENV
    os.environ[SEED_ENV] = str(args.seed)
    return execute(args)


def execute(args):
    import torch
    from .. import cache, checks, environment, workflow
    from . import calibrate, calibration
    if not checks.is_v7():
        raise ValueError("补跑子命令只用于 v7 协议（run.py 会设 PO_JEPA_PROTOCOL=v7）")
    bundle = workflow.verify_bundle()
    device = torch.device(args.device)
    if args.command == "check-addon" and (device.type != "cuda" or device.index is None):
        raise ValueError("explicit cuda:N target required")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage = "environment"
    try:
        if args.command == "check-addon":
            cache._write_json(output/"environment.json", environment.verify_versions())
        stage = "calibration"
        workflow.require_calibrated_weights(swept=False)
        calibrate.require_base_calibration(args.base_calibration)
        if args.command == "calibrate-addon":
            stage = "calibrate-addon"
            calibrate.calibrate_addon(Path(args.features_root), output, device, base=args.base_calibration)
            path = calibration.addon_calibration_path()
            extra = dict(addon_calibration_file=calibrate._relative(path), addon_calibration_sha256=cache.file_sha(path))
        else:
            calibration.require_addon_calibration()
            stage = "check-addon"
            from .check_addon import check_addon
            report = check_addon(Path(args.features_root), output, device)
            extra = dict(check_sha256=cache.file_sha(output/"check-addon.json"), real_device=report["hardware"])
        if workflow.verify_bundle() != bundle:
            raise ValueError("source bundle changed during run")
        cache._write_json(output/"COMPLETE.json", dict(command=args.command, manifest_sha256=bundle, base_calibration=args.base_calibration, **extra,
                                                       scientific_conclusions_verified=False))
        return 0
    except Exception as error:
        cache._write_json(output/"FAILED.json", dict(stage=stage, error_type=type(error).__name__, message=str(error),
                                                     traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
