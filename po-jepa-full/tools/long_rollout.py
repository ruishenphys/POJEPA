"""S25c 长时程滚动评测：用基础场景上训练好的模型，在长时程测试集（<场景>~horizon=128 的测试段）上从 tick 0 滚到 64 步。

指标的定义见 src/po_jepa/runtime/long_rollout.py 文件头：T = 16、24、32、48、64 时一次直接预测和每 8 步一段的组合预测
各自离真实特征的 JEPA 距离、两者之差，以及 8 步滚动每隔 8 个 tick 的误差曲线（到 128）。只评测，不训练、不用编码器。

用法（GPU；--job 是 run.py run 的输出目录，条件、数据集、种子都从里面读）：
  python tools/encode_v7.py encode --system double_well --variant horizon=128 --trajectories 2000 --splits test \\
      --upstream-root ../vjepa2 --checkpoint ../checkpoints/vjepa2-vitg-fpc64-256.pth \\
      --output ../features-v8/double_well~horizon=128-2000 --device cuda:0 --shard 0/1
  python tools/encode_v7.py finalize --system double_well --variant horizon=128 --trajectories 2000 --splits test \\
      --output ../features-v8/double_well~horizon=128-2000
  python tools/long_rollout.py --job ../results-v8/full/double_well/seed1703/po_jepa_full \\
      --features ../features-v8/double_well~horizon=128-2000 --output ../results-v8/full/long/double_well/seed1703/po_jepa_full.json --device cuda:0
二阶场景现在按单帧复制（dup）编码（特征看不到速度，S25b 接 pair 编码），报告写 tubelet: dup。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", type=Path, required=True, help="基础场景上 run.py run 的输出目录")
    p.add_argument("--features", type=Path, required=True, help="长时程测试集的特征（encode_v7.py --variant horizon=H --splits test）")
    p.add_argument("--horizon", type=int, default=128, choices=(64, 128))
    p.add_argument("--output", type=Path, required=True, help="结果 JSON（必须不存在）")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ac-checkpoint", type=Path, help="条件是 vjepa2_ac 时需要")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse(argv)
    sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"tools"))
    import plan_v7
    job_dir, job = plan_v7._job_env(a.job)
    if a.ac_checkpoint:
        os.environ["PO_JEPA_AC_CHECKPOINT"] = str(a.ac_checkpoint.resolve())
    if a.output.exists():
        raise SystemExit(f"{a.output} 已存在，不覆盖")
    import torch
    from po_jepa import datasets, protocols, systems, variants
    from po_jepa.runtime import cache, data, long_rollout, v7jobs

    base = datasets.active()
    if base.knobs.horizon != systems.TICKS:
        raise SystemExit(f"--job 的数据集 {base.tag} 本身是长时程数据集：模型应在 33 帧的基础场景上训练")
    if base.knobs.changes_timing() or base.knobs.changes_space():
        raise SystemExit(f"--job 的数据集 {base.tag} 带时间或空间采样调节项：长时程滚动只接基础场景（时间、空间采样在 S25d、S25e 接）")
    condition = job["condition"]
    if job.get("kind"):
        raise SystemExit(f"--job 是 {job['kind']} 任务，要 run.py run 的任务目录")
    device = torch.device(a.device)
    started = time.perf_counter()
    model, loss, selected, registration = v7jobs.load_for_evaluation(condition, None if condition == "identity" else job_dir, device)
    scene = base.system if not base.variant else f"{base.system}~{base.variant}"
    long_spec = datasets.DatasetSpec(base.system, base.count, protocols.merged(scene, f"horizon={a.horizon}"))
    old = datasets._set_for_tests(long_spec)
    try:
        identity = data.approved_cache_metadata(a.features, splits=("test",))
        rows, means = long_rollout.evaluate(model, a.features, loss, a.horizon, device)
    finally:
        datasets._set_for_tests(old)
    report = dict(kind="long_rollout", dataset=long_spec.tag, train_dataset=base.tag, condition=condition, seed=job["seed"],
                  updates=job["updates"], horizon=a.horizon, job_dir=str(job_dir),
                  tubelet="dup" if base.system in variants.SECOND_ORDER else None,
                  best=None if selected is None else dict(step=selected["step"], checkpoint_sha256=selected.get("checkpoint_sha256")),
                  registration_sha256=hashlib.sha256(cache._json_bytes(registration)).hexdigest(),
                  cache_identity=dict(encode_report_sha256=identity["encode_report_sha256"]),
                  definitions=long_rollout.definitions(a.horizon), counts=dict(trajectories=len(rows)),
                  means=means, rows=rows, device=str(device), seconds=round(time.perf_counter()-started, 1),
                  scientific_conclusions_verified=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_name(a.output.name+".partial")
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False))
    tmp.replace(a.output)
    t = long_rollout.direct_times(a.horizon)[-1]
    print(f"{long_spec.tag} {condition}：{len(rows)} 条；T={t:g} 直接 {means[f'long_direct:T={t:g}']:.4g}，"
          f"8 步滚动 {means[f'long_composed:T={t:g}']:.4g} → {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
