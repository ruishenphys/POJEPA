"""S25f 物理状态读出：冻结线性探针、长时程滚动的物理单位误差和有效预测时长（VPT）、反事实预测。

定义见 src/po_jepa/runtime/state_readout.py 文件头。四个子命令，每个场景（数据集）一份探针和一份真值，和条件、种子无关：

  fit            在主实验缓存的验证段（33 帧，真实编码器）上拟合探针 → <输出目录>/probe.json、weight.npy、mean_x.npy
  truth          算读出目标的真值（CPU）：长时程测试集每条测试轨迹每 8 个 tick 的目标（核长时程缓存的画面哈希），
                 和主实验测试窗口的反事实终点（换 donor 动作从同一真实起点重新仿真，核主实验缓存的画面哈希）→ JSON
  rollout        S25c 的 8 步滚动接探针（每个条件、种子一条；GPU）→ JSON
  counterfactual 主实验测试窗口上的反事实预测（每个条件、种子一条；GPU）→ JSON

用法（--job 是 run.py run 的输出目录；fit、truth 用 --system/--variant/--trajectories 指数据集，和 encode_v7.py 相同）：
  python tools/state_readout.py fit --system double_well --trajectories 2000 --features ../features-v8/double_well-2000 \\
      --output ../results-v8/full/readout/probe/double_well --device cuda:0
  python tools/state_readout.py truth --system double_well --trajectories 2000 --features ../features-v8/double_well-2000 \\
      --long-features ../features-v8/double_well~horizon=128-2000 --output ../results-v8/full/readout/truth/double_well.json
  python tools/state_readout.py rollout --job ../results-v8/full/double_well/seed1703/po_jepa_full \\
      --probe ../results-v8/full/readout/probe/double_well --truth ../results-v8/full/readout/truth/double_well.json \\
      --features ../features-v8/double_well~horizon=128-2000 --output ../results-v8/full/readout/rollout/double_well/seed1703/po_jepa_full.json
  python tools/state_readout.py counterfactual --job ../results-v8/full/double_well/seed1703/po_jepa_full \\
      --probe ../results-v8/full/readout/probe/double_well --truth ../results-v8/full/readout/truth/double_well.json \\
      --features ../features-v8/double_well-2000 --output ../results-v8/full/readout/counterfactual/double_well/seed1703/po_jepa_full.json
二阶场景的编码看训练数据集的调节项：不带 tubelet=pair 是单帧复制（dup，特征看不到速度），带的是两帧编码（S25b），报告的 tubelet 字段照实写。
S25b5（Jerry 2026-11-25 定，新口径）：二阶场景另读速度（定义见 src/po_jepa/runtime/state_readout.py 文件头第 1 条）。命令行不变：
fit 在 --output 下另写 velocity/ 子目录（速度探针），truth 另写 v_actual、v_counterfactual 和长时程的 v，rollout、counterfactual
从 --probe 的 velocity/ 读速度探针，结果多出 vel_ 开头的速度指标（反事实汇总在 overall.velocity、by_duration.<t>.velocity）。
S25b5 之前拟合的二阶场景探针和真值没有速度，rollout、counterfactual 会拒绝，要用当前代码重做 fit、truth。
速度探针记下同一次 fit 写的位置探针的数组 SHA（position_probe），读的时候核来源字段齐全、相同并按它绑定（S25b5-fix1）。
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LONG_HORIZON = 128


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("fit", "truth"):
        s = sub.add_parser(name)
        s.add_argument("--system", required=True)
        s.add_argument("--variant", default="")
        s.add_argument("--trajectories", type=int, required=True)
        s.add_argument("--features", type=Path, required=True, help="主实验缓存（encode_v7.py，33 帧）")
        s.add_argument("--output", type=Path, required=True)
    sub.choices["fit"].add_argument("--kind", default="full", choices=("mean", "pool4", "full"))
    sub.choices["fit"].add_argument("--device", default="cuda:0")
    sub.choices["truth"].add_argument("--long-features", type=Path, required=True, help="长时程测试集缓存（horizon=H）")
    sub.choices["truth"].add_argument("--horizon", type=int, default=LONG_HORIZON, choices=(64, 128))
    for name in ("rollout", "counterfactual"):
        s = sub.add_parser(name)
        s.add_argument("--job", type=Path, required=True, help="基础场景上 run.py run 的输出目录")
        s.add_argument("--probe", type=Path, required=True)
        s.add_argument("--truth", type=Path, required=True)
        s.add_argument("--features", type=Path, required=True)
        s.add_argument("--output", type=Path, required=True, help="结果 JSON（必须不存在）")
        s.add_argument("--device", default="cuda:0")
        s.add_argument("--ac-checkpoint", type=Path, help="条件是 vjepa2_ac 时需要")
        s.add_argument("--cross-version", action="store_true",
                       help="S28c：--job 是别的代码版本训练的（例如 v7 分支的 v7.1 正式运行），按 addon.v7load 读选中检查点"
                            "（核任务和训练目录的完成证据、选点、查询序列、配置快照和 8 个数据字段，不核源码 SHA）")
    sub.choices["rollout"].add_argument("--horizon", type=int, default=LONG_HORIZON, choices=(64, 128))
    sub.choices["rollout"].add_argument("--base-features", type=Path,
                                        help="S28c，只和 --cross-version 一起用：训练数据集的主实验缓存（v7load 核检查点的数据来源要读它）")
    return p.parse_args(argv)


def _write(path, report):
    from po_jepa.runtime import cache
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".partial")
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)
    return hashlib.sha256(cache._json_bytes(report)).hexdigest()


def _sha_image(frame):
    return hashlib.sha256(frame.tobytes()).hexdigest()


def _refuse_sampling(spec, what):
    from po_jepa import systems
    if spec.knobs.horizon != systems.TICKS or spec.knobs.changes_timing() or spec.knobs.changes_space():
        raise SystemExit(f"{spec.tag} 带长时程、时间采样或空间采样调节项：{what}要用 33 帧的基础场景（主实验数据集）")


def _checked_states(system, tid, m, ticks):
    """frames() 的隐藏状态，并核这些 tick 的画面哈希等于缓存 manifest 记的（状态和特征是同一帧）。"""
    states, frames = system.frames(tid)
    if len(frames) != len(m["frames"]):
        raise SystemExit(f"轨迹 {tid}：frames() 有 {len(frames)} 帧，缓存有 {len(m['frames'])} 帧")
    for t in ticks:
        if m["frames"][t]["tick"] != t or _sha_image(frames[t]) != m["frames"][t]["image_sha256"]:
            raise SystemExit(f"轨迹 {tid} 第 {t} 帧的画面哈希与缓存不一致（检查 --system/--variant/--trajectories）")
    return states


def cmd_fit(a):
    import numpy as np
    import torch
    from po_jepa import datasets, systems, variants
    from po_jepa.runtime import cache, data, state_readout
    spec = datasets.active(); system = systems.get()
    _refuse_sampling(spec, "探针")
    if a.output.exists():
        raise SystemExit(f"{a.output} 已存在，不覆盖")
    started = time.perf_counter()
    approved = data.approved_cache_metadata(a.features, splits=("validation",))
    params = dict(spec.knobs.params)
    ids = list(spec.split_range("validation"))
    x, y, yv, groups = None, [], [], []
    velocity = state_readout.has_velocity(spec.system)          # S25b5
    for i, tid in enumerate(ids):
        raw, m, sha = cache._read_complete_cache(a.features/f"{tid:04d}")
        if (m["trajectory_id"] != tid or m["split"] != "validation" or sha != approved["manifest_sha256_by_id"][tid]
                or m["generation"]["evidence_role"] != "real_frozen_encoder"):
            raise SystemExit(f"轨迹 {tid}：缓存不是这个数据集验证段的真实编码器缓存")
        states = _checked_states(system, tid, m, range(raw.shape[0]))
        f = state_readout.features(torch.from_numpy(np.array(raw, dtype=np.float32)), a.kind)
        if x is None:
            x = torch.empty(len(ids)*f.shape[0], f.shape[1], dtype=torch.float32)
        x[i*f.shape[0]:(i+1)*f.shape[0]] = f
        y.append(state_readout.targets(spec.system, states, params)); groups += [tid]*f.shape[0]
        if velocity:
            yv.append(state_readout.velocity_targets(spec.system, states, params))
    probe, record = state_readout.fit(x, np.concatenate(y), np.array(groups), a.kind, device=a.device)
    # S25b5：速度探针用同一份特征另拟合（单独选 λ，位置探针不受影响），两个都拟合完再写盘
    vprobe, vrecord = (state_readout.fit(x, np.concatenate(yv), np.array(groups), a.kind, device=a.device)
                       if velocity else (None, None))
    common = dict(dataset=spec.tag, system=spec.system, variant=spec.variant, split="validation",
                  trajectories=ids, cache_identity=dict(encode_report_sha256=approved["encode_report_sha256"]),
                  tubelet=variants.tubelet_label(spec.system, spec.knobs), seconds=round(time.perf_counter()-started, 1),
                  scientific_conclusions_verified=False)
    meta = state_readout.save(probe, a.output, dict(common, quantity="position", units=state_readout.units(spec.system),
                                                    **({"velocity_probe": state_readout.VELOCITY_DIR} if velocity else {})))
    show = lambda r: (f"{r['samples']} 个样本，交叉验证 R² {r['cv_r2']:.4f}"
                      f"（λ = 10^{r['lambda_exponent']:g}·trace/n{'，在网格边上' if r['lambda_at_grid_edge'] else ''}）")
    print(f"{spec.tag} 探针（{a.kind}）：{show(record)} → {a.output}")
    if velocity:
        # S25b5-fix1（Codex 审核 F1）：速度探针记下配对的位置探针的两个数组 SHA，读的时候按它绑定
        vmeta = state_readout.save(vprobe, a.output/state_readout.VELOCITY_DIR,
                                   dict(common, quantity="velocity", units=state_readout.velocity_units(spec.system),
                                        position_probe=dict(weight_sha256=meta["weight_sha256"], mean_x_sha256=meta["mean_x_sha256"])))
        print(f"{spec.tag} 速度探针（{a.kind}）：{show(vrecord)} → {a.output/state_readout.VELOCITY_DIR}")
        meta = dict(meta, velocity=vmeta)
    return meta


def cmd_truth(a):
    import numpy as np
    from po_jepa import datasets, protocols, simulation, systems, variants
    from po_jepa.runtime import cache, data, evaluation, long_cache, long_rollout, state_readout
    import probe_actions
    base = datasets.active(); system = systems.get()
    _refuse_sampling(base, "真值")
    if a.output.exists():
        raise SystemExit(f"{a.output} 已存在，不覆盖")
    started = time.perf_counter()
    params = dict(base.knobs.params)
    approved = data.approved_cache_metadata(a.features, splits=("test",))
    counterfactual = []
    velocity = state_readout.has_velocity(base.system)          # S25b5：二阶场景另写速度真值
    vel = lambda state: state_readout.velocity_target(base.system, state, params).tolist()
    for tid in base.split_range("test"):
        donor = datasets.donor_id(tid)
        _, m, sha = cache._read_complete_cache(a.features/f"{tid:04d}")
        if m["trajectory_id"] != tid or m["split"] != "test" or sha != approved["manifest_sha256_by_id"][tid]:
            raise SystemExit(f"轨迹 {tid}：主实验缓存身份不符")
        ends = sorted({s+int(t) for s in datasets.test_start_ticks(tid) for t in state_readout.CF_DURATIONS})
        states = _checked_states(system, tid, m, ends)
        for start in datasets.test_start_ticks(tid):
            for t in state_readout.CF_DURATIONS:
                true_end, cf_state, _ = probe_actions.counterfactual_end(system, simulation, tid, donor, start, int(t))
                if not np.allclose(true_end, states[start+int(t)], atol=1e-6):
                    raise SystemExit(f"轨迹 {tid}：真实终点重新仿真不一致")
                counterfactual.append(dict(trajectory_id=tid, donor_id=donor, start_tick=start, duration_cycles=t,
                                           manifest_sha256=sha,
                                           y_actual=state_readout.target(base.system, states[start+int(t)], params).tolist(),
                                           y_counterfactual=state_readout.target(base.system, cf_state, params).tolist(),
                                           **(dict(v_actual=vel(states[start+int(t)]), v_counterfactual=vel(cf_state))
                                              if velocity else {})))
    scene = base.system if not base.variant else f"{base.system}~{base.variant}"
    long_spec = datasets.DatasetSpec(base.system, base.count, protocols.merged(scene, f"horizon={a.horizon}"))
    old = datasets._set_for_tests(long_spec)
    try:
        long_approved = data.approved_cache_metadata(a.long_features, splits=("test",))
        times = long_rollout.curve_times(a.horizon)
        long = []
        for tid in long_spec.split_range("test"):
            _, m, sha = long_cache.read_complete(a.long_features/f"{tid:04d}")
            if m["trajectory_id"] != tid or sha != long_approved["manifest_sha256_by_id"][tid]:
                raise SystemExit(f"轨迹 {tid}：长时程缓存身份不符")
            states = _checked_states(system, tid, m, [int(t) for t in times])
            long.append(dict(trajectory_id=tid, manifest_sha256=sha,
                             y={f"{t:g}": state_readout.target(base.system, states[int(t)], params).tolist() for t in times},
                             **({"v": {f"{t:g}": vel(states[int(t)]) for t in times}} if velocity else {})))
    finally:
        datasets._set_for_tests(old)
    report = dict(kind="state_readout_truth", dataset=base.tag, long_dataset=long_spec.tag, system=base.system, horizon=a.horizon,
                  units=state_readout.units(base.system), cache_identity=dict(encode_report_sha256=approved["encode_report_sha256"]),
                  long_cache_identity=dict(encode_report_sha256=long_approved["encode_report_sha256"]),
                  counterfactual=counterfactual, long=long, seconds=round(time.perf_counter()-started, 1),
                  tubelet=variants.tubelet_label(base.system, base.knobs),
                  **({"velocity": True, "velocity_units": state_readout.velocity_units(base.system)} if velocity else {}))
    _write(a.output, report)
    print(f"{base.tag} 真值：反事实 {len(counterfactual)} 个窗口，长时程 {len(long)} 条 → {a.output}")
    return report


def _load_job(a):
    """rollout、counterfactual 共用：数据集取自任务目录，载入模型（核法同 transfer、long_rollout），读探针和真值。

    任务环境（数据集、种子、变体）由 main() 在导入 po_jepa 之前用 plan_v7._job_env 设好，结果在 a.job_dir、a.job_record。"""
    job_dir, job = a.job_dir, a.job_record
    if a.ac_checkpoint:
        os.environ["PO_JEPA_AC_CHECKPOINT"] = str(a.ac_checkpoint.resolve())
    if a.output.exists():
        raise SystemExit(f"{a.output} 已存在，不覆盖")
    import torch
    from po_jepa import datasets
    from po_jepa.runtime import state_readout, v7jobs
    base = datasets.active()
    scene, _, count = job["dataset"].rpartition("-")
    if base.tag != datasets.parse(f"{scene}:{count}").tag:
        # G4 R3：po_jepa.datasets 在设好任务环境之前就导入了，会停在默认的 double_well-200 上
        raise SystemExit(f"进程里的数据集是 {base.tag}，--job 的数据集是 {job['dataset']}：任务环境没在导入 po_jepa 之前设好")
    _refuse_sampling(base, "读出")
    if job.get("kind"):
        raise SystemExit(f"--job 是 {job['kind']} 任务，要 run.py run 的任务目录")
    probe, probe_meta = state_readout.load(a.probe, dataset_tag=base.tag, system=base.system)
    truth = json.loads(a.truth.read_text())
    if truth.get("kind") != "state_readout_truth" or truth.get("dataset") != base.tag:
        raise SystemExit(f"--truth 不是 {base.tag} 的真值文件")
    vprobe = vmeta = None
    if state_readout.has_velocity(base.system):
        # S25b5：二阶场景要速度探针和速度真值；S25b5 之前做的没有，拒绝（不静默只报位置）
        if truth.get("velocity") is not True:
            raise SystemExit(f"--truth 没有速度真值（S25b5 之前做的？）：二阶场景要用当前代码重新跑 truth")
        try:
            vprobe, vmeta = state_readout.load(a.probe/state_readout.VELOCITY_DIR, dataset_tag=base.tag, system=base.system,
                                               quantity="velocity")
        except ValueError as error:
            raise SystemExit(f"--probe 的速度探针读不进来（{error}）")
        _check_velocity_pair(probe_meta, vmeta)
    a.velocity_probe = None if vprobe is None else vprobe.to(torch.device(a.device))
    a.velocity_meta = vmeta
    device = torch.device(a.device)
    if a.cross_version:
        model, loss, selected, registration = _load_cross_version(a, job_dir, job, device)
    else:
        if getattr(a, "base_features", None) is not None:
            raise SystemExit("--base-features 只和 --cross-version 一起用")
        model, loss, selected, registration = v7jobs.load_for_evaluation(
            job["condition"], None if job["condition"] == "identity" else job_dir, device)
    return job_dir, job, base, probe.to(device), probe_meta, truth, model, loss, selected, registration, device


def _check_velocity_pair(probe_meta, vmeta):
    """S25b5-fix1（Codex 审核 F1）：速度探针必须是和位置探针同一次 fit 写的。两边的来源字段（缓存身份、轨迹、tubelet、
    读出 kind）都要在且有效、彼此相同（原来用 .get()，两边都缺时 None == None 放行）；两边的 mean_x 是同一份特征的列均值，
    SHA 要相同；速度探针记的配对位置探针（position_probe 的 weight、mean_x SHA）要等于这个位置探针的。"""
    for name, m in (("位置", probe_meta), ("速度", vmeta)):
        ci = m.get("cache_identity")
        if not (isinstance(ci, dict) and isinstance(ci.get("encode_report_sha256"), str) and len(ci["encode_report_sha256"]) == 64
                and isinstance(m.get("trajectories"), list) and m["trajectories"] and m.get("tubelet") in ("dup", "pair")
                and isinstance(m.get("mean_x_sha256"), str)):
            raise SystemExit(f"--probe 的{name}探针缺来源字段（缓存身份、轨迹、tubelet、mean_x SHA）：用当前代码重新跑 fit")
    if any(vmeta[k] != probe_meta[k] for k in ("cache_identity", "trajectories", "tubelet", "mean_x_sha256")) \
            or vmeta["fit"]["kind"] != probe_meta["fit"]["kind"]:
        raise SystemExit("--probe 的速度探针和位置探针不是同一份缓存上拟合的（读出 kind、缓存身份、轨迹、tubelet 或特征均值不同）")
    if vmeta.get("position_probe") != dict(weight_sha256=probe_meta["weight_sha256"], mean_x_sha256=probe_meta["mean_x_sha256"]):
        raise SystemExit("--probe 的速度探针记的配对位置探针不是这一个（不是同一次 fit 写的）")


def _load_cross_version(a, job_dir, job, device):
    """S28c：别的代码版本训练的任务（v7 分支的 v7.1 正式运行）。按 addon.v7load.load_v7_selected 读（D4 的读法，10 §4.2）：
    模型按当前代码现建，核配置快照和检查点来源里的数据字段，源码 SHA 只记录。返回值和 v7jobs.load_for_evaluation 同形，
    registration 换成 dict(cross_version=v7load 的记录)：这份任务的预登记是旧代码的，不和当前预登记比。"""
    from po_jepa.runtime import comparison
    from po_jepa.runtime.addon import v7load
    features = a.base_features if a.command == "rollout" else a.features
    if features is None:
        raise SystemExit("rollout --cross-version 要 --base-features（训练数据集的主实验缓存，v7load 核数据来源）")
    model, loss = comparison.build_condition(job["condition"], device)
    try:
        record = v7load.load_v7_selected(job_dir, job["condition"], model, loss_config=loss, features=features, device=device)
    except ValueError as error:
        raise SystemExit(f"{job_dir}：跨版本读检查点没通过（{error}）")
    best = record["best"]
    selected = None if best is None else dict(step=best["step"], checkpoint_sha256=best["checkpoint_sha256"])
    return model, loss, selected, dict(cross_version=record)


def _report(kind, a, job_dir, job, base, probe_meta, truth, selected, registration, extra):
    from po_jepa import variants
    from po_jepa.runtime import state_readout
    return dict(kind=kind, train_dataset=base.tag, condition=job["condition"], seed=job["seed"], updates=job["updates"],
                job_dir=str(job_dir), units=truth["units"],
                tubelet=variants.tubelet_label(base.system, base.knobs),
                best=None if selected is None else dict(step=selected["step"], checkpoint_sha256=selected.get("checkpoint_sha256")),
                **_registration_fields(registration),
                probe=dict(path=str(a.probe), kind=probe_meta["fit"]["kind"], weight_sha256=probe_meta["weight_sha256"],
                           cv_r2=probe_meta["fit"]["cv_r2"], variance=probe_meta["fit"]["variance"]),
                **({} if a.velocity_meta is None else dict(
                    velocity_units=truth["velocity_units"],
                    velocity_probe=dict(path=str(a.probe/state_readout.VELOCITY_DIR), weight_sha256=a.velocity_meta["weight_sha256"],
                                        cv_r2=a.velocity_meta["fit"]["cv_r2"], variance=a.velocity_meta["fit"]["variance"]))),
                truth_sha256=hashlib.sha256(a.truth.read_bytes()).hexdigest(), **extra, scientific_conclusions_verified=False)


def _registration_fields(registration):
    """同版本任务记当前预登记的 SHA（和以前相同）；跨版本（S28c）没有可比的预登记，记 None，另记 v7load 的读取记录。"""
    from po_jepa.runtime import cache
    if "cross_version" in registration:
        return dict(registration_sha256=None, cross_version=registration["cross_version"])
    return dict(registration_sha256=hashlib.sha256(cache._json_bytes(registration)).hexdigest())


def _eval_mode(model):
    from po_jepa.runtime import evaluation
    modes = {m: m.training for m in model.modules()}
    model.eval()
    return modes, evaluation._state_identity(model)


def _restore(model, modes, identity):
    from po_jepa.runtime import evaluation
    for module, flag in modes.items():
        module.training = flag
    if evaluation._state_identity(model) != identity:
        raise RuntimeError("state readout changed parameters, gradients or buffers")


def cmd_rollout(a):
    import numpy as np
    from po_jepa import datasets, protocols
    from po_jepa.runtime import data, long_rollout, state_readout
    job_dir, job, base, probe, probe_meta, truth, model, loss, selected, registration, device = _load_job(a)
    if truth["horizon"] != a.horizon:
        raise SystemExit(f"--truth 是 horizon={truth['horizon']} 的真值，不是 {a.horizon}")
    started = time.perf_counter()
    scene = base.system if not base.variant else f"{base.system}~{base.variant}"
    long_spec = datasets.DatasetSpec(base.system, base.count, protocols.merged(scene, f"horizon={a.horizon}"))
    if truth["long_dataset"] != long_spec.tag:
        raise SystemExit("--truth 的长时程数据集不符")
    by_id = {r["trajectory_id"]: r for r in truth["long"]}
    old = datasets._set_for_tests(long_spec)
    rows = []
    try:
        identity = data.approved_cache_metadata(a.features, splits=("test",))
        if identity["encode_report_sha256"] != truth["long_cache_identity"]["encode_report_sha256"]:
            raise SystemExit("--features 不是 --truth 核过的那份长时程缓存")
        modes, state = _eval_mode(model)
        try:
            for tid in long_spec.split_range("test"):
                windows = {t: long_rollout.load_window(a.features, tid, t, device, a.horizon) for t in long_rollout.window_times(a.horizon)}
                sha = windows[float(a.horizon)].provenance["actual_manifest_sha256"]
                if by_id[tid]["manifest_sha256"] != sha:
                    raise SystemExit(f"轨迹 {tid}：真值记的 manifest 与缓存不符")
                y = {t: np.asarray(by_id[tid]["y"][f"{t:g}"]) for t in long_rollout.curve_times(a.horizon)}
                yv = None if a.velocity_probe is None else {t: np.asarray(by_id[tid]["v"][f"{t:g}"])
                                                            for t in long_rollout.curve_times(a.horizon)}
                rows.append(dict(trajectory_id=tid, values=state_readout.rollout_values(model, probe, windows, y, a.horizon,
                                                                                        a.velocity_probe, yv),
                                 manifest_sha256=sha))
        finally:
            _restore(model, modes, state)
    finally:
        datasets._set_for_tests(old)
    names = state_readout.rollout_names(a.horizon, a.velocity_probe is not None)
    means = {k: float(np.mean([r["values"][k] for r in rows])) for k in names} if rows else {}
    report = _report("state_readout_rollout", a, job_dir, job, base, probe_meta, truth, selected, registration, dict(
        dataset=long_spec.tag, horizon=a.horizon, curve_times=list(long_rollout.curve_times(a.horizon)),
        vpt_thresholds=list(state_readout.VPT_THRESHOLDS), counts=dict(trajectories=len(rows)), means=means, rows=rows,
        device=str(device), seconds=round(time.perf_counter()-started, 1)))
    _write(a.output, report)
    e = state_readout.VPT_THRESHOLDS[1]
    print(f"{long_spec.tag} {job['condition']}：{len(rows)} 条；VPT@{e:g} {means[f'vpt@{e:g}']:.3g} tick"
          f"（探针本身 {means[f'floor_vpt@{e:g}']:.3g}）→ {a.output}")
    return report


def cmd_counterfactual(a):
    from po_jepa import datasets
    from po_jepa.runtime import data, evaluation, state_readout
    job_dir, job, base, probe, probe_meta, truth, model, loss, selected, registration, device = _load_job(a)
    started = time.perf_counter()
    identity = data.approved_cache_metadata(a.features, splits=("test",))
    if identity["encode_report_sha256"] != truth["cache_identity"]["encode_report_sha256"]:
        raise SystemExit("--features 不是 --truth 核过的那份主实验缓存")
    expected = [(tid, s, t) for tid in base.split_range("test") for s in datasets.test_start_ticks(tid) for t in state_readout.CF_DURATIONS]
    if [(r["trajectory_id"], r["start_tick"], r["duration_cycles"]) for r in truth["counterfactual"]] != expected:
        raise SystemExit("--truth 的反事实窗口与登记的测试窗口不符")
    rows = []
    modes, state = _eval_mode(model)
    try:
        for r in truth["counterfactual"]:
            q = evaluation.EvaluationQuery(r["trajectory_id"], r["start_tick"], r["duration_cycles"], 16, "point")
            w = evaluation.load_test_window(a.features, q, device)
            donor = evaluation.load_test_window(a.features, replace(q, trajectory_id=evaluation.donor_id(q.trajectory_id)), device)
            evaluation._aligned_pair(w, donor, q)
            if w.provenance["actual_manifest_sha256"] != r["manifest_sha256"] or donor.inputs.trajectory_id != r["donor_id"]:
                raise SystemExit(f"轨迹 {q.trajectory_id}：真值记的 manifest 或 donor 与缓存不符")
            same = evaluation.identical_controls(w.inputs.controls, donor.inputs.controls, w.inputs.duration)
            rows.append(dict(trajectory_id=q.trajectory_id, donor_id=r["donor_id"], start_tick=q.start_tick,
                             duration_cycles=q.duration_cycles, identical_controls=bool(same),
                             values=state_readout.counterfactual_values(
                                 model, probe, w, donor, r["y_actual"], r["y_counterfactual"], a.velocity_probe,
                                 *((None, None) if a.velocity_probe is None else (r["v_actual"], r["v_counterfactual"])))))
    finally:
        _restore(model, modes, state)
    report = _report("state_readout_counterfactual", a, job_dir, job, base, probe_meta, truth, selected, registration, dict(
        dataset=base.tag, durations=list(state_readout.CF_DURATIONS), overall=state_readout.counterfactual_summary(rows),
        by_duration={f"{t:g}": state_readout.counterfactual_summary([r for r in rows if r["duration_cycles"] == t])
                     for t in state_readout.CF_DURATIONS}, rows=rows, device=str(device),
        seconds=round(time.perf_counter()-started, 1)))
    _write(a.output, report)
    o = report["overall"]
    print(f"{base.tag} {job['condition']}：{o['windows']} 个窗口；explained {o['explained']}，gain {o['gain']} → {a.output}")
    return report


def main(argv=None):
    a = _parse(argv)
    sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"tools"))
    # 数据集环境变量必须在第一次导入 po_jepa 之前设好：po_jepa.datasets 只在导入时读一次 PO_JEPA_DATASET，
    # 之后再设不起作用（G4 R3：rollout、counterfactual 原来在 cmd_* 里先导入、后设任务环境，数据集停在默认的 double_well-200）
    if a.command in ("fit", "truth"):
        os.environ["PO_JEPA_DATASET"] = f"{a.system}{'~'+a.variant if a.variant else ''}:{a.trajectories}"
    else:
        import plan_v7                                       # 顶层不导入 po_jepa
        a.job_dir, a.job_record = plan_v7._job_env(a.job)    # 设 PO_JEPA_DATASET/SEED/UPDATES/PROTOCOL 和变体
    from po_jepa import datasets
    # 同一进程里已经导入过（测试里多次调用 main）时按本次的环境变量重设
    if datasets.active().tag != datasets.parse(os.environ["PO_JEPA_DATASET"]).tag:
        datasets._set_for_tests(datasets.parse(os.environ["PO_JEPA_DATASET"]))
    {"fit": cmd_fit, "truth": cmd_truth, "rollout": cmd_rollout, "counterfactual": cmd_counterfactual}[a.command](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
