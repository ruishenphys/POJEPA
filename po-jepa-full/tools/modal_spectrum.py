"""S25f 模态频率对照（探索性）：PO-JEPA 学到的阻尼旋转块（γ、ω）和线性场景的解析模态。

定义见 src/po_jepa/runtime/modal_spectrum.py 文件头。只接 PO-JEPA 条件（po_jepa_full、po_jepa_no_consistency 等
comparison.POJEPA 模型），只接线性场景（advection_diffusion、spring_chain、wave）的基础场景任务目录。只评测，不训练、不用编码器。

用法（--job 是 run.py run 的输出目录；--features 是同一数据集的主实验缓存，用测试段的登记起点）：
  python tools/modal_spectrum.py --job ../results-v8/full/advection_diffusion/seed1703/po_jepa_full \\
      --features ../features-v8/advection_diffusion-2000 --output ../results-v8/full/modal/advection_diffusion/seed1703/po_jepa_full.json \\
      --device cuda:0
二阶场景（spring_chain、wave）现在按单帧复制（dup）编码，特征里看不到速度：对照要等 S25b 的 pair 编码后重跑才有意义。
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="结果 JSON（必须不存在）")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse(argv)
    sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"tools"))
    import plan_v7
    job_dir, job = plan_v7._job_env(a.job)
    if a.output.exists():
        raise SystemExit(f"{a.output} 已存在，不覆盖")
    import numpy as np
    import torch
    from po_jepa import datasets, systems, variants
    from po_jepa.runtime import cache, comparison, data, evaluation, modal_spectrum, v7jobs
    spec = datasets.active()
    if spec.system not in modal_spectrum.LINEAR_SCENES:
        raise SystemExit(f"{spec.system} 没有解析的线性模态：模态频率对照只做 {modal_spectrum.LINEAR_SCENES}")
    if spec.knobs.horizon != systems.TICKS or spec.knobs.changes_timing() or spec.knobs.changes_space():
        raise SystemExit(f"{spec.tag} 带长时程、时间采样或空间采样调节项：要 33 帧的基础场景任务目录")
    if job.get("kind"):
        raise SystemExit(f"--job 是 {job['kind']} 任务，要 run.py run 的任务目录")
    if not job["condition"].startswith("po_jepa"):
        raise SystemExit(f"{job['condition']} 不是 PO-JEPA 条件：没有阻尼旋转块")
    device = torch.device(a.device)
    started = time.perf_counter()
    model, loss, selected, registration = v7jobs.load_for_evaluation(job["condition"], job_dir, device)
    if not isinstance(model, comparison.POJEPA):
        raise SystemExit(f"{job['condition']} 的模型不是 POJEPA")
    approved = data.approved_cache_metadata(a.features, splits=("test",))
    modes = {m: m.training for m in model.modules()}
    identity = evaluation._state_identity(model)
    samples, rows = [], []
    try:
        model.eval()
        for tid in spec.split_range("test"):
            raw, m, sha = cache._read_complete_cache(a.features/f"{tid:04d}")
            if (m["trajectory_id"] != tid or m["split"] != "test" or sha != approved["manifest_sha256_by_id"][tid]
                    or m["generation"]["evidence_role"] != "real_frozen_encoder"):
                raise SystemExit(f"轨迹 {tid}：缓存不是这个数据集测试段的真实编码器缓存")
            for start in datasets.test_start_ticks(tid):
                z = torch.from_numpy(np.array(raw[start:start+1], dtype=np.float32)).to(device)
                s = modal_spectrum.learned_blocks(model, z, cache.regular_grid(16, 16))
                samples.append(s); rows.append(dict(trajectory_id=tid, start_tick=start, manifest_sha256=sha))
    finally:
        for module, flag in modes.items():
            module.training = flag
    if evaluation._state_identity(model) != identity:
        raise RuntimeError("modal spectrum changed parameters, gradients or buffers")
    # 对照用的未训练模型：同一结构、torch 种子 = 任务种子（不一定和训练的初始权重逐位相同）。初始化时 ω ≈ 0、γ = 0.1/周期，
    # 所以 advection_diffusion 的 non_oscillatory 在初始化时就接近 1，要和这里比，不能只和均匀分布比。
    torch.manual_seed(job["seed"])
    fresh, _ = comparison.build_condition(job["condition"], device)
    fresh.eval()
    reference = []
    for tid in spec.split_range("test"):
        raw, _, _ = cache._read_complete_cache(a.features/f"{tid:04d}")
        for start in datasets.test_start_ticks(tid):
            z = torch.from_numpy(np.array(raw[start:start+1], dtype=np.float32)).to(device)
            reference.append(modal_spectrum.learned_blocks(fresh, z, cache.regular_grid(16, 16)))
    report = dict(kind="modal_spectrum", dataset=spec.tag, condition=job["condition"], seed=job["seed"], updates=job["updates"],
                  job_dir=str(job_dir), tubelet="dup" if spec.system in variants.SECOND_ORDER else None,
                  best=None if selected is None else dict(step=selected["step"], checkpoint_sha256=selected.get("checkpoint_sha256")),
                  registration_sha256=hashlib.sha256(cache._json_bytes(registration)).hexdigest(),
                  cache_identity=dict(encode_report_sha256=approved["encode_report_sha256"]), starts=rows,
                  comparison=modal_spectrum.compare(spec.system, spec.knobs, samples),
                  init_reference=modal_spectrum.compare(spec.system, spec.knobs, reference), exploratory=True,
                  device=str(device), seconds=round(time.perf_counter()-started, 1), scientific_conclusions_verified=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_name(a.output.name+".partial")
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False))
    tmp.replace(a.output)
    c = report["comparison"]
    key = "near_true" if spec.system in modal_spectrum.OSCILLATORY else "non_oscillatory"
    print(f"{spec.tag} {job['condition']}：{c['samples']} 个起点；{key} {c[key]:.3g}（均匀分布时 {c['null_'+key]:.3g}，"
          f"未训练模型 {report['init_reference'][key]:.3g}）→ {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
