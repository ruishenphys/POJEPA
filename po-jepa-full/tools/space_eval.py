"""S25e 空间采样评测：用基础场景（256×256、16×16 网格、相机不动）上训练好的模型，在空间采样测试集
（<场景>~resolution=384 等的测试段）上评测，并在基础测试集的同一组窗口上再评一遍配对。

指标的定义见 src/po_jepa/runtime/space_eval.py 文件头：主实验的测试窗口（起点 s、时长 T = 1、2、3、4、8、12 tick），
空间采样测试集上直接预测离真实特征的 JEPA 距离（模型直接吃 (R/16)×(R/16) 的原生特征）、基础测试集上同一窗口的同一个量、
两者之差。只评测，不训练、不用编码器。

用法（GPU；--job 是 run.py run 的输出目录，条件、数据集、种子都从里面读；--test-variant 只能是空间采样调节项；
--base-features 是这个基础场景主实验的特征目录）：
  python tools/encode_v7.py encode --system double_well --variant resolution=384 --trajectories 2000 --splits test \\
      --upstream-root ../vjepa2 --checkpoint ../checkpoints/vjepa2-vitg-fpc64-256.pth \\
      --output ../features-v8/double_well~resolution=384-2000 --device cuda:0 --shard 0/1
  python tools/encode_v7.py finalize --system double_well --variant resolution=384 --trajectories 2000 --splits test \\
      --output ../features-v8/double_well~resolution=384-2000
  python tools/space_eval.py --job ../results-v8/full/double_well/seed1703/po_jepa_full --test-variant resolution=384 \\
      --features ../features-v8/double_well~resolution=384-2000 --base-features ../features-v8/double_well-2000 \\
      --output ../results-v8/full/space/double_well~resolution=384/seed1703/po_jepa_full.json --device cuda:0
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
    p.add_argument("--test-variant", required=True,
                   help="空间采样调节项，例如 resolution=384、zoom=0.75、pan=1,zoom=0.75、drift=0.5,pan=1,zoom=0.75")
    p.add_argument("--features", type=Path, required=True, help="空间采样测试集的特征（encode_v7.py --variant … --splits test）")
    p.add_argument("--base-features", type=Path, required=True, help="基础场景主实验的特征（同一组测试窗口的配对参照）")
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
    from po_jepa.runtime import cache, data, space_eval, v7jobs

    base = datasets.active()
    if base.knobs.horizon != systems.TICKS or base.knobs.changes_timing() or base.knobs.changes_space():
        raise SystemExit(f"--job 的数据集 {base.tag} 带 horizon、时间或空间采样调节项：模型应在 256、相机不动、33 帧的基础场景上训练")
    if job.get("kind"):
        raise SystemExit(f"--job 是 {job['kind']} 任务，要 run.py run 的任务目录")
    items = [i for i in a.test_variant.split(",") if i]
    if not items or any(i.partition("=")[0] not in space_eval.SPACE_KNOBS for i in items):
        raise SystemExit(f"--test-variant 只能是空间采样调节项 {space_eval.SPACE_KNOBS}（其他调节项用 run.py transfer）：{a.test_variant}")
    scene = base.system if not base.variant else f"{base.system}~{base.variant}"
    test_spec = datasets.DatasetSpec(base.system, base.count, protocols.merged(scene, a.test_variant))
    if not test_spec.knobs.changes_space():
        raise SystemExit(f"--test-variant {a.test_variant} 是默认空间采样（256、zoom=1、不平移、不漂移）：这就是基础测试集")
    condition = job["condition"]
    device = torch.device(a.device)
    started = time.perf_counter()
    model, loss, selected, registration = v7jobs.load_for_evaluation(condition, None if condition == "identity" else job_dir, device)
    base_identity = data.approved_cache_metadata(a.base_features, splits=("test",))
    base_rows = space_eval.evaluate(model, a.base_features, loss, device, base=True)
    old = datasets._set_for_tests(test_spec)
    try:
        identity = data.approved_cache_metadata(a.features, splits=("test",))
        space_rows = space_eval.evaluate(model, a.features, loss, device)
        definitions = space_eval.definitions()
    finally:
        datasets._set_for_tests(old)
    rows, means = space_eval.paired(space_rows, base_rows)
    report = dict(kind="space_eval", dataset=test_spec.tag, train_dataset=base.tag, test_variant=test_spec.variant,
                  condition=condition, seed=job["seed"], updates=job["updates"], job_dir=str(job_dir),
                  tubelet="dup" if base.system in variants.SECOND_ORDER else None,
                  best=None if selected is None else dict(step=selected["step"], checkpoint_sha256=selected.get("checkpoint_sha256")),
                  registration_sha256=hashlib.sha256(cache._json_bytes(registration)).hexdigest(),
                  cache_identity=dict(encode_report_sha256=identity["encode_report_sha256"],
                                      base_encode_report_sha256=base_identity["encode_report_sha256"]),
                  definitions=definitions, counts=dict(windows=len(rows), trajectories=len({r["trajectory_id"] for r in rows})),
                  means=means, rows=rows, device=str(device), seconds=round(time.perf_counter()-started, 1),
                  scientific_conclusions_verified=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_name(a.output.name+".partial")
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False))
    tmp.replace(a.output)
    print(f"{test_spec.tag} {condition}：{len(rows)} 个窗口；T=1 直接 {means.get('direct:T=1', float('nan')):.4g}"
          f"（基础 {means.get('base_direct:T=1', float('nan')):.4g}），T=8 差 {means.get('gap:T=8', float('nan')):.4g} → {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
