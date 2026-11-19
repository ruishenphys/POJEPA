"""C7 AC 疑点核查（差距清单 §3 C7，v7.1 分析 §5 第 1 条）：两个子命令，都只评测、不训练、不改任何训练或评测代码。

疑点：v7.1 里官方 AC 在 16×16 上“直接预测 12 周期”（0.154）反而比“逐步滚动 12 周期”（0.094）差，误差也几乎不随时长涨。
AC 的“直接预测 T 周期”是 T 步自回归，第 k 步的上下文是起点帧加前面 k−1 个预测帧（ac_baseline.ACPredictor.rollout），
T = 12 时上下文最长 12 帧、时间位置到 11；而 AC 训练用 9 帧片段（8 次转移，ac_training.CLIP_FRAMES），teacher forcing
的上下文最长 8 帧、时间位置只到 7。逐步滚动（evaluation.unit_rollout）每步只看 1 帧。推断：上下文超出训练长度是原因（未验证）。

  context  在一个已完成任务（AC 条件）的测试窗口上，把 AC 的直接预测按“只保留最近 W 帧”的滑动窗口重算：
           W ∈ {1, 2, 4, 8}，另算完整上下文（逐位等于直接预测）和 v7.1 的逐步滚动（unit_rollout）。
           第 k 步的上下文是第 max(0, k+1−W)…k 帧，动作取同样那几帧的 tick 动作，时间位置从 0 重新数（和训练片段一样）；
           W = 8 正好是训练时的最长上下文。T ≤ W 时滑动窗口不截断，和直接预测是同一串运算，直接记直接预测的值。
           W = 1 和逐步滚动只差每步对预测帧多做的一次 LayerNorm（unit_rollout 每步走 forward，forward 先做 LN），数值上应几乎相同。
           指标和 v7.1 主表同一个：future_jepa_distance（式 14，逐 token LN 后的 L1），只用 16×16 单点（6 个时长）。
  floor    特征噪声下限：同一状态编码两次的距离。重画前 K 条测试轨迹（核画面哈希等于缓存记的，保证是同一状态的同一画面），
           用和 encode_v7.py 完全相同的冻结编码器和预处理重新编码，和缓存比；同时拆开三个来源：
             rerun     同一进程同一批大小编两次（GPU 运行间不确定性）；
             batch     每次 1 帧 vs 缓存的批大小（分批改变浮点结果，encode_v7 文件头 S25-fix3）；
             rounding  新编码的 float32 和它按缓存精度（默认 float16）存一遍的差（存储舍入）；
             vs_cache  新编码和缓存里的特征（上面几项的合计，就是“同一状态编码两次”的下限）。
           另给 adjacent：缓存里相邻两帧（1 个 tick）的距离，作为特征变化的尺度参照。距离都是 future_jepa_distance 同一个式子。
           本仓库的主实验数据集没有执行噪声、过程噪声、像素噪声（variants 默认全 0），所以这个下限只来自数值，不含物理随机性。

用法（在 full-model/po-jepa-full 里运行；--job 是 run.py run 的任务目录，v7.1 的检查点加 --cross-version，读法同 state_readout.py）：
  python tools/ac_context_check.py context --job ../../results-v7/full/v7.1/double_well/seed1703/vjepa2_ac --cross-version \\
      --features ../../features-v7/double_well-2000 --ac-checkpoint ../../checkpoints/vjepa2-ac-vitg.pt \\
      --output ../../results-small-validation/ac-check/context/double_well/seed1703/vjepa2_ac.json --device cuda:0
  python tools/ac_context_check.py floor --system double_well --trajectories 2000 --features ../../features-v7/double_well-2000 \\
      --upstream-root ../../vjepa2 --checkpoint ../../checkpoints/vjepa2-vitg-fpc64-256.pth \\
      --output ../../results-small-validation/ac-check/floor/double_well.json --device cuda:0
--output 必须不存在（先写 .partial 再换名）。context 只收 AC 条件（vjepa2_ac、vjepa2_ac_small）和 2 维动作的仿真数据集。
本文件不在训练的源码身份（runtime/data.consumer_sources、comparison.consumer_sources）里，加它不改任何预登记。
"""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (1, 2, 4, 8)
FLOOR_TRAJECTORIES = 8


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("context")
    c.add_argument("--job", type=Path, required=True, help="AC 条件的 run.py run 任务目录")
    c.add_argument("--features", type=Path, required=True, help="训练数据集的主实验缓存（33 帧，encode_v7.py）")
    c.add_argument("--output", type=Path, required=True, help="结果 JSON（必须不存在）")
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--ac-checkpoint", type=Path, help="条件是 vjepa2_ac 时需要（官方权重，建模型用）")
    c.add_argument("--cross-version", action="store_true",
                   help="--job 是别的代码版本训练的（v7 分支的 v7.1 正式运行），按 addon.v7load 读选中检查点（同 state_readout.py）")
    c.add_argument("--max-trajectories", type=int, default=0, help="只用前 N 条测试轨迹（0 = 全部，正式结果用全部）")
    f = sub.add_parser("floor")
    f.add_argument("--system", required=True)
    f.add_argument("--variant", default="")
    f.add_argument("--trajectories", type=int, required=True)
    f.add_argument("--features", type=Path, required=True, help="主实验缓存（33 帧，encode_v7.py）")
    f.add_argument("--upstream-root", type=Path, required=True)
    f.add_argument("--checkpoint", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--device", default="cuda:0")
    f.add_argument("--count", type=int, default=FLOOR_TRAJECTORIES, help="重编码的测试轨迹条数（取测试段最前面的 N 条）")
    return p.parse_args(argv)


# ---------------------------------------------------------------- context：滑动窗口上下文

def windowed_rollout(model, raw, controls, duration, grid, window):
    """AC 的整数时长直接预测，第 k 步只看最近 window 帧（None = 完整上下文，和 ACPredictor.forward 同一串运算）。

    raw [1,N,E] 未 LN；controls/duration 同 forward；返回第 T 步的预测 [1,N,E] float32（已 LN）。只接 2 维动作（仿真）。"""
    import math
    import torch
    from torch.nn import functional as F
    from po_jepa import ac_baseline
    from po_jepa.actions import validate_action_segments
    if not isinstance(model, ac_baseline.ACPredictor):
        raise ValueError("windowed_rollout 只接 V-JEPA 2-AC 预测器")
    if model.spec.action_dim != 2 or model.true_state:
        raise ValueError("只接仿真的 2 维动作、learned_constant 状态（DROID 的 7 维官方输入没接）")
    if window is not None and (type(window) is not int or window < 1):
        raise ValueError("window 必须是正整数或 None")
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != grid.height*grid.width or raw.shape[2] != model.encoder_dim:
        raise ValueError("raw 必须是一个窗口 [1, grid tokens, encoder_dim]")
    validate_action_segments(controls, query_duration=duration)
    steps = float(duration[0])
    if duration.shape != (1,) or steps != int(steps) or steps < 1:
        raise ValueError("只接一个正整数时长（周期）")
    steps = int(steps)
    actions = ac_baseline.tick_actions(controls, int(math.ceil(steps))).to(raw)
    z0 = F.layer_norm(raw, (model.encoder_dim,))
    with model._autocast(raw):
        frames = [z0]
        for k in range(steps):
            lo = 0 if window is None else max(0, k+1-window)
            out = model.predict_frames(torch.stack(frames[lo:], 1), actions[:, lo:k+1], grid.height, grid.width)
            frames.append(out[:, -1])
        return frames[-1].float()


def context_values(model, raw, target, controls, duration, grid, loss):
    """一个 16×16 测试点的各种上下文的 future_jepa_distance，外加完整上下文和 forward 的最大差（应为 0）。"""
    import torch
    from po_jepa.runtime import comparison, evaluation
    T = int(float(duration[0]))
    def distance(pred):
        return float(comparison.jepa_distance(pred, target, config=loss).per_sample[0])
    with torch.no_grad():
        direct = model(raw, controls, duration, grid)
        full = windowed_rollout(model, raw, controls, duration, grid, None)
        values = dict(direct=distance(direct), full_context=distance(full),
                      unit_rollout=distance(evaluation.unit_rollout(model, raw, controls, duration, grid)))
        for w in WINDOWS:
            values[f"window{w}"] = (values["direct"] if T <= w
                                    else distance(windowed_rollout(model, raw, controls, duration, grid, w)))
    return values, float((full-direct).abs().max())


def method_names():
    return ("direct", "full_context", *(f"window{w}" for w in WINDOWS), "unit_rollout")


def summarize_context(rows):
    """按时长分组的均值；T = 12 上 window8 − direct 的配对差（负 = 截断后更好）和变好的点数。"""
    import numpy as np
    out = {}
    for T in sorted({r["duration_cycles"] for r in rows}):
        group = [r for r in rows if r["duration_cycles"] == T]
        means = {m: float(np.mean([r["values"][m] for r in group])) for m in method_names()}
        diff = np.array([r["values"]["window8"]-r["values"]["direct"] for r in group])
        out[f"{T:g}"] = dict(points=len(group), means=means,
                             window8_minus_direct=dict(mean=float(diff.mean()), better_points=int((diff < 0).sum()),
                                                       worse_points=int((diff > 0).sum())),
                             max_full_context_minus_direct=float(max(r["max_abs_full_minus_direct"] for r in group)))
    return out


def cmd_context(a):
    import torch
    from po_jepa import datasets
    from po_jepa.runtime import comparison, data, evaluation
    import state_readout as sr
    job_dir, job = a.job_dir, a.job_record
    if a.output.exists():
        raise SystemExit(f"{a.output} 已存在，不覆盖")
    if job.get("kind"):
        raise SystemExit(f"--job 是 {job['kind']} 任务，要 run.py run 的任务目录")
    if job["condition"] not in comparison.AC_CONDITIONS:
        raise SystemExit(f"--job 的条件是 {job['condition']}，context 只核 AC 条件 {comparison.AC_CONDITIONS}")
    if a.max_trajectories < 0:
        raise SystemExit("--max-trajectories 不能是负数")
    base = datasets.active()
    scene, _, count = job["dataset"].rpartition("-")
    if base.tag != datasets.parse(f"{scene}:{count}").tag:
        raise SystemExit(f"进程里的数据集是 {base.tag}，--job 的数据集是 {job['dataset']}：任务环境没在导入 po_jepa 之前设好")
    sr._refuse_sampling(base, "AC 上下文核查")
    if a.ac_checkpoint:
        os.environ["PO_JEPA_AC_CHECKPOINT"] = str(a.ac_checkpoint.resolve())
    device = torch.device(a.device)
    if a.cross_version:
        model, loss, selected, registration = sr._load_cross_version(a, job_dir, job, device)
    else:
        from po_jepa.runtime import v7jobs
        model, loss, selected, registration = v7jobs.load_for_evaluation(job["condition"], job_dir, device)
    identity = data.approved_cache_metadata(a.features, splits=("test",))
    test_ids = list(base.split_range("test"))
    if a.max_trajectories:
        test_ids = test_ids[:a.max_trajectories]
    queries = [q for q in evaluation.build_index() if q.kind == "point" and q.grid_size == 16 and q.trajectory_id in test_ids]
    started = time.perf_counter()
    rows = []
    modes, state = sr._eval_mode(model)
    try:
        for q in queries:
            w = evaluation.load_test_window(a.features, q, device)
            raw, target, grid = evaluation.spatial_view(w, 16)
            values, gap = context_values(model, raw, target, w.inputs.controls, w.inputs.duration, grid, loss)
            rows.append(dict(trajectory_id=q.trajectory_id, start_tick=q.start_tick, duration_cycles=q.duration_cycles,
                             manifest_sha256=w.provenance["actual_manifest_sha256"], values=values,
                             max_abs_full_minus_direct=gap))
    finally:
        sr._restore(model, modes, state)
    from po_jepa.runtime import ac_training
    report = dict(kind="ac_context_check", dataset=base.tag, condition=job["condition"], seed=job["seed"], updates=job["updates"],
                  job_dir=str(job_dir), training_context_frames=ac_training.CLIP_FRAMES-1, windows=list(WINDOWS),
                  methods=list(method_names()), metric="future_jepa_distance", grid=16,
                  best=None if selected is None else dict(step=selected["step"], checkpoint_sha256=selected.get("checkpoint_sha256")),
                  **sr._registration_fields(registration), cache_identity=dict(encode_report_sha256=identity["encode_report_sha256"]),
                  counts=dict(trajectories=len(test_ids), points=len(rows), all_test_trajectories=not a.max_trajectories),
                  by_duration=summarize_context(rows), rows=rows, device=str(device),
                  seconds=round(time.perf_counter()-started, 1), scientific_conclusions_verified=False)
    sr._write(a.output, report)
    last = report["by_duration"].get("12")
    if last:
        m = last["means"]
        print(f"{base.tag} {job['condition']}：T=12 直接 {m['direct']:.4g}，截断到 8 帧 {m['window8']:.4g}，"
              f"逐步滚动 {m['unit_rollout']:.4g}（{last['points']} 个点）→ {a.output}")
    return report


# ---------------------------------------------------------------- floor：同一状态编码两次

def floor_values(first, again, single, stored, stored_dtype, loss):
    """一条轨迹各帧的四项距离和相邻帧参照（逐帧均值、最大）。输入都是 numpy [L,N,E]：first/again/single 是新编码
    （float64，同 encode_v7._encode_trajectory 的返回），stored 是缓存里存的数组（stored_dtype）。"""
    import numpy as np
    import torch
    from po_jepa.runtime import comparison
    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    def d(x, y):
        return comparison.jepa_distance(t(x), t(y), config=loss).per_sample.double().numpy()
    fresh = first.astype(np.float32)
    parts = dict(rerun=d(fresh, again), batch=d(fresh, single), rounding=d(fresh, fresh.astype(stored_dtype)),
                 vs_cache=d(fresh, stored), adjacent=d(stored[:-1], stored[1:]))
    return {k: dict(mean=float(v.mean()), max=float(v.max())) for k, v in parts.items()}


def _real_encoder(upstream_root, checkpoint, device):
    """和 tools/encode_v7.py encode（不带空间采样）完全相同的冻结编码器。"""
    from po_jepa import vjepa
    from po_jepa.runtime import cache
    arch = vjepa.EncoderArchitectureSpec(vjepa.UPSTREAM_COMMIT, "configs/train/vitg16/droid-256px-8f.yaml",
        "vit_giant_xformers", 256, 16, 2, 512, True, True, False, True, True, True)
    build = vjepa.build_encoder_architecture(upstream_root=upstream_root, architecture=arch, initialization_seed=0)
    loaded = vjepa.load_local_encoder(build, checkpoint=vjepa.LocalCheckpointSpec(
        checkpoint, cache.CHECKPOINT_SHA, cache.PUBLICATION, "target_encoder", ("module.", "backbone."), True, (), ()))
    return vjepa.FrozenFrameEncoder(loaded, preprocess=cache.preprocess_spec(), encoder_device=device, autocast_dtype=None)


def cmd_floor(a):
    import hashlib
    import numpy as np
    import torch
    from dataclasses import fields
    from po_jepa import datasets, simulation, systems
    from po_jepa.runtime import cache, checks, comparison, data
    import encode_v7
    import state_readout as sr
    spec = datasets.active()
    sr._refuse_sampling(spec, "特征噪声下限")
    if a.output.exists():
        raise SystemExit(f"{a.output} 已存在，不覆盖")
    if a.count < 1:
        raise SystemExit("--count 必须是正整数")
    approved = data.approved_cache_metadata(a.features, splits=("test",))
    ids = list(spec.split_range("test"))[:a.count]
    config = checks.load_protocol()
    loss = comparison.LossConfig(**{f.name: config["loss"][f.name] for f in fields(comparison.LossConfig)})
    device = torch.device(a.device)
    encoder = _real_encoder(a.upstream_root.resolve(), a.checkpoint.resolve(), device)
    system = systems.get()
    started = time.perf_counter()
    rows = []
    for tid in ids:
        raw, m, sha = cache._read_complete_cache(a.features/f"{tid:04d}")
        if m["trajectory_id"] != tid or m["split"] != "test" or sha != approved["manifest_sha256_by_id"][tid]:
            raise SystemExit(f"轨迹 {tid}：主实验缓存身份不符")
        if m.get("schema_version") != cache.V7_SCHEMA:
            raise SystemExit(f"轨迹 {tid}：不是 encode_v7.py 写的缓存（v6 缓存的编码流程不同，floor 不接）")
        _, frames = system.frames(tid)
        if len(frames) != len(m["frames"]) or any(hashlib.sha256(frames[k].tobytes()).hexdigest() != m["frames"][k]["image_sha256"]
                                                  for k in range(len(frames))):
            raise SystemExit(f"轨迹 {tid}：重画的画面和缓存记的哈希不符（不是同一状态的同一画面）")
        batch = int(m["generation"].get("encode_batch", 33))
        stored = np.asarray(raw)
        enc = lambda size: encode_v7._encode_trajectory(encoder, frames, tid, simulation, cache, torch, np, size, False)
        first, again, single = enc(batch), enc(batch), enc(1)
        rows.append(dict(trajectory_id=tid, manifest_sha256=sha, encode_batch=batch, stored_dtype=str(stored.dtype),
                         recorded_relative_rounding_error=m.get("feature_stats", {}).get("relative_rounding_error"),
                         values=floor_values(first, again, single, stored, stored.dtype, loss)))
        print(f"[{spec.tag}] 轨迹 {tid}：vs_cache {rows[-1]['values']['vs_cache']['mean']:.3g}", flush=True)
    names = ("rerun", "batch", "rounding", "vs_cache", "adjacent")
    summary = {k: dict(mean=float(np.mean([r["values"][k]["mean"] for r in rows])),
                       max=float(max(r["values"][k]["max"] for r in rows))) for k in names}
    report = dict(kind="feature_noise_floor", dataset=spec.tag, metric="future_jepa_distance", trajectories=ids,
                  cache_identity=dict(encode_report_sha256=approved["encode_report_sha256"]), summary=summary, rows=rows,
                  encoder=dict(checkpoint_sha256=cache.CHECKPOINT_SHA, device=str(device),
                               device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
                  seconds=round(time.perf_counter()-started, 1), scientific_conclusions_verified=False)
    sr._write(a.output, report)
    print(f"{spec.tag} 特征噪声下限（{len(rows)} 条）：同一状态编码两次 {summary['vs_cache']['mean']:.3g}，"
          f"相邻两帧 {summary['adjacent']['mean']:.3g} → {a.output}")
    return report


def main(argv=None):
    a = _parse(argv)
    sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"tools"))
    # 数据集环境变量必须在第一次导入 po_jepa 之前设好（G4 R3 的教训，见 state_readout.main）
    if a.command == "floor":
        os.environ["PO_JEPA_PROTOCOL"] = "v7"
        os.environ["PO_JEPA_DATASET"] = f"{a.system}{'~'+a.variant if a.variant else ''}:{a.trajectories}"
    else:
        import plan_v7                                       # 顶层不导入 po_jepa
        a.job_dir, a.job_record = plan_v7._job_env(a.job)
    from po_jepa import datasets
    if datasets.active().tag != datasets.parse(os.environ["PO_JEPA_DATASET"]).tag:
        datasets._set_for_tests(datasets.parse(os.environ["PO_JEPA_DATASET"]))
    {"context": cmd_context, "floor": cmd_floor}[a.command](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
