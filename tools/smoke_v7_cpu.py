"""开发用：在 CPU 上用假编码器特征把 v7 全流程（标定 → 训练全部条件 → 评测 → 汇总图）走一遍。

只检查代码能跑通、产物齐全，不产生任何科学结论；步数被缩到几步。正式运行只用 run.py。
用法：
  python tools/encode_v7.py encode --system van_der_pol --trajectories 40 --output /tmp/feat --device cpu --fake-encoder
  python tools/encode_v7.py finalize --system van_der_pol --trajectories 40 --output /tmp/feat
  python tools/smoke_v7_cpu.py --system van_der_pol --features-root /tmp/feat --output /tmp/smoke
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True)
    p.add_argument("--trajectories", type=int, default=40)
    p.add_argument("--features-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=6)
    p.add_argument("--jobs", default="", help="逗号分隔的条件：走 v7 独立任务入口（v7jobs.run_job）而不是旧的一次训练全部条件")
    p.add_argument("--seed", type=int, default=1703)
    p.add_argument("--transfer", nargs=2, metavar=("VARIANT", "FEATURES"), help="再对每个条件做一次分布外评测")
    a = p.parse_args()
    os.environ["PO_JEPA_PROTOCOL"] = "v7"
    os.environ["PO_JEPA_DATASET"] = f"{a.system}:{a.trajectories}"
    os.environ["PO_JEPA_UPDATES"] = "2000"
    os.environ["PO_JEPA_SEED"] = str(a.seed)
    sys.path.insert(0, str(ROOT/"src"))
    import torch
    from po_jepa.runtime import cache, checks, comparison, workflow

    # 假特征是 synthetic：仅在本脚本里把它当作真实特征读取
    original = cache._read_v7_cache
    def as_real(path, m, sha):
        m = copy.deepcopy(m); g = m["generation"]
        g.update(evidence_role="real_frozen_encoder", encoder_checkpoint_sha256=cache.CHECKPOINT_SHA,
                 encoder_load_report=dict(sha256=cache.CHECKPOINT_SHA, encoder_key="target_encoder", strict=True,
                                          missing_keys=[], unexpected_keys=[]))
        return original(path, m, sha)
    cache._read_v7_cache = as_real

    # 把步数缩到几步（所有读取配置的地方共用同一个缩小版，配置一致性检查仍然成立）
    full = checks.load_protocol
    def tiny():
        c = copy.deepcopy(full())
        c["training"]["updates_per_condition"] = a.updates
        c["loss"].update(lr_decay_updates=a.updates, lr_warmup_updates=1, consistency_warmup_updates=1)
        c["validation"]["interval_updates"] = max(1, a.updates//2)
        c["monitoring"].update(early_check_updates=10**9, consistency_interval=2)
        c["calibration"].update(pretrain_updates=2, batches=2)
        return c
    checks.load_protocol = tiny
    workflow.checks.load_protocol = tiny
    workflow.verify_bundle = lambda: "smoke"
    calib = Path(a.output)/"calibration.yaml"
    checks.v7_calibration_path = lambda: calib
    a.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cpu")
    workflow.calibrate_weights(a.features_root, a.output/"calibrate", device)
    workflow.require_calibrated_weights(swept=True)
    if a.jobs:
        from po_jepa.runtime import v7jobs
        for condition in a.jobs.split(","):
            v7jobs.run_job(a.features_root, a.output/"jobs"/condition, device, condition)
            if a.transfer:
                v7jobs.transfer_job(Path(a.transfer[1]), a.output/"jobs"/f"{condition}-transfer", device, condition,
                                    training_run=None if condition == "identity" else a.output/"jobs"/condition,
                                    test_variant=a.transfer[0])
            if condition != "identity":   # 再走一遍“只评测”入口（与 run.py evaluate --training-run 相同）
                v7jobs.run_job(a.features_root, a.output/"jobs"/f"{condition}-evaluate", device, condition,
                               training_run=a.output/"jobs"/condition)
    else:
        workflow.train_conditions(a.features_root, a.output/"training", device)
        workflow.evaluate_conditions(a.features_root, a.output/"training", a.output/"evaluation", device)
    print(json.dumps(dict(ok=True, output=str(a.output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
