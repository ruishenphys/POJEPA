"""10 §3、§4.3、§7：D4 只评测的新指标、等价核对和任务入口（run.py d4 的计算部分）。

记号（10 §1、§3.1）：z 是窗口起点的 16×16 特征，u 是从起点开始的记录动作，z*_T 是第 s+T 个 tick 的特征。

- Φ_T(z)：一次完整预测 model(z, u₁⊕u₂, T, 16×16)，u₁、u₂ 是 split_action_segments 在 T/2 处切开的两段，
  和 v7 evaluation.point_metrics 的 direct 逐行相同（v7 的 future_jepa_distance、tcd_rms 都用它）。D4 里凡是
  “Φ_T(z)”都是这个张量，每个时长只算一次（tcd_ln、pd、xpd 的细网格一侧、tcd_unequal、留出分解、长时程）。
- scd_ln 用 v7 evaluation.pair_metrics 的那一对：R Φ^16_T(z; u) 和 Φ^h_T(R z; u)，u 不切开。
- Φ_Π(z)：按分割 Π = (τ_1, …, τ_n) 逐段滚动，每段一次完整调用，输入是上一段输出的原始预测；第 i 段的动作是
  u 依次用 split_action_segments 切出来的第 i 段（局部时间从 0 开始），时长张量是 CPU float64 [1] 的 τ_i。
- 两种度量（ln_metrics，float64）：RMS_LN(a, b) 和 RMS_in(a, b) = RMS(a − b)/√s²(z)，s² 用 16×16 起点。
- d(a, b)：JEPA 距离 jepa_distance(a, b).per_sample（float32 的 mean|LN(a) − LN(b)|）。

每个窗口的量（键名见 value_names()）：

| 键 | 公式 | 10 § |
|---|---|---|
| tcd_ln:T=t | RMS_LN(Φ_t, Φ_{t/2}∘Φ_{t/2}) / (t/2)² | 3.2 |
| scd_ln:h=g:T=t | RMS_LN(R Φ^16_t, Φ^g_t(R z)) / t | 3.2 |
| pd_ln:T=t:Π、pd_in:T=t:Π | RMS_LN(Φ_t, Φ_Π)、RMS_in(Φ_t, Φ_Π)，不除时间 | 3.3 |
| xpd_ln:T=t:n=k、xpd_in:T=t:n=k | 两种度量下 (R Φ^16_t, Φ^8_{Π_k}(R z))，8×8 上算，s² 仍用 16×16 起点 | 3.4 |
| tcd_unequal_ln:d=t:τ+σ、tcd_unequal_in:d=t:τ+σ | 两种度量下 (Φ_t, Φ_σ∘Φ_τ) / (τσ) | 3.5 |
| heldout_direct:T=t、heldout_partition:T=t、heldout_rollout:T=t | d(Φ_t, z*_t)、d(Φ_t, Φ_Π)、d(Φ_Π, z*_t) | 3.6 |

每条轨迹（起点 0）：long_direct:T=t = d(Φ_t, z*_t)，long_composed:T=t = d(Φ_{8,…,8}, z*_t)，
long_pd_ln:T=t、long_pd_in:T=t 是两者之间的两种度量（10 §3.7）。

等价核对（10 §4.3）：同一窗口上用 v7 的公式重算 future_jepa_distance、tcd_normalized（16×16，T ∈ TIMES）和
scd_normalized（16→8、16→12），和被读任务自己的 evaluation/<条件>/rows/*.json 逐条比较，容差
|新 − 旧| ≤ 1e-6·max(1, |旧|)。
"""

import math

from dataclasses import asdict

from pathlib import Path

import torch

from .. import cache, checks, comparison, data, environment, evaluation
from po_jepa import ln_metrics, metrics
from . import long_window, v7load

TCD_TIMES = (1., 2., 3., 4., 8., 12.)                                     # 10 §3.2，和 evaluation.TIMES 相同
SCD_GRIDS = (8, 12)
PARTITIONS = ((4., (2., 2.)), (4., (1.,)*4), (4., (.5,)*8),               # 10 §3.3 的表，顺序照抄
              (8., (4., 4.)), (8., (2.,)*4), (8., (1.,)*8), (8., (3., 5.)))
CROSS = ((4., 2), (4., 4), (8., 2), (8., 4))                              # 10 §3.4：(T, n)，Π_n 是 n 段等分
UNEQUAL = ((4., (1., 3.)), (4., (3., 1.)), (8., (2., 6.)), (8., (6., 2.)))   # 10 §3.5：(d, (τ, σ))，先 τ 后 σ
HELDOUT = ((3., (1., 2.)), (12., (8., 4.)))                               # 10 §3.6：先 1 后 2，先 8 后 4
LONG_TIMES = long_window.LONG_TIMES                                       # 10 §3.7
LONG_BLOCK = 8.
EQUIVALENCE_METRICS = ("future_jepa_distance", "tcd_normalized", "scd_normalized")
EQUIVALENCE_TOLERANCE = 1e-6
TRIANGLE_TOLERANCE = 1e-6


def _t(value) -> str:
    """时长写进键名：1.0 → "1"、0.5 → "0.5"、12.0 → "12"。"""
    return f"{float(value):g}"


def partition_label(parts) -> str:
    """等分写成 "段数x每段"（2x2、4x1、8x0.5），其余写成 "τ1+τ2+…"（3+5、1+3）。"""
    if len(parts) > 1 and len(set(parts)) == 1:
        return f"{len(parts)}x{_t(parts[0])}"
    return "+".join(_t(p) for p in parts)


def partition_x(total, parts) -> float:
    """x(Π) = (T² − Σ τ_i²)/2（10 §3.3）。"""
    return (total*total-sum(p*p for p in parts))/2


def value_names() -> tuple:
    """每个窗口 values 的全部键，按固定顺序（54 个）。"""
    names = [f"tcd_ln:T={_t(t)}" for t in TCD_TIMES]
    names += [f"scd_ln:h={h}:T={_t(t)}" for t in TCD_TIMES for h in SCD_GRIDS]
    names += [f"pd_{m}:T={_t(t)}:{partition_label(p)}" for t, p in PARTITIONS for m in ("ln", "in")]
    names += [f"xpd_{m}:T={_t(t)}:n={n}" for t, n in CROSS for m in ("ln", "in")]
    names += [f"tcd_unequal_{m}:d={_t(d)}:{partition_label(p)}" for d, p in UNEQUAL for m in ("ln", "in")]
    names += [f"heldout_{term}:T={_t(t)}" for t, _ in HELDOUT for term in ("direct", "partition", "rollout")]
    return tuple(names)


def long_value_names() -> tuple:
    return tuple(f"long_{k}:T={_t(t)}" for t in LONG_TIMES for k in ("direct", "composed", "pd_ln", "pd_in"))


def equivalence_names() -> tuple:
    names = [f"{m}:T={_t(t)}" for t in TCD_TIMES for m in EQUIVALENCE_METRICS[:2]]
    return tuple(names+[f"scd_normalized:h={h}:T={_t(t)}" for t in TCD_TIMES for h in SCD_GRIDS])


def definitions() -> dict:
    """写进 report.json 的定义表：分割、x(Π)、跨网格、不等分、留出分解、长时程。"""
    return dict(
        partitions=[dict(T=t, parts=list(p), label=partition_label(p), x=partition_x(t, p)) for t, p in PARTITIONS],
        cross_grid=[dict(T=t, n=n, coarse_grid=8) for t, n in CROSS],
        unequal=[dict(d=d, tau=p[0], sigma=p[1], label=partition_label(p)) for d, p in UNEQUAL],
        heldout=[dict(T=t, parts=list(p), label=partition_label(p)) for t, p in HELDOUT],
        long=dict(times=list(LONG_TIMES), block=LONG_BLOCK, start_tick=0),
        direct="model(z, u1⊕u2, T) with u split at T/2 (v7 evaluation.point_metrics direct); scd_ln uses the v7 pair_metrics pair",
        equivalence_tolerance=dict(rule="|new-old| <= 1e-6*max(1,|old|)", value=EQUIVALENCE_TOLERANCE),
        triangle_tolerance=dict(rule="direct <= partition + rollout + 1e-6*max(1,direct)", value=TRIANGLE_TOLERANCE))


def _cycles(value) -> torch.Tensor:
    return torch.tensor([float(value)], dtype=torch.float64)


def pieces(controls, parts) -> list:
    """把时长 Σparts 的动作依次精确切成 len(parts) 段：第 i 次 split_action_segments(rest, τ_i, 剩余时长)。"""
    out, rest = [], controls
    for i, tau in enumerate(parts[:-1]):
        pair = comparison.split_action_segments(rest, tau=_cycles(tau), sigma=_cycles(sum(parts[i+1:])))
        out.append(pair.first)
        rest = pair.second
    out.append(rest)
    return out


def rollout(model, raw, controls, parts, grid):
    """Φ_Π(raw)：逐段完整调用，下一段的输入是上一段输出的原始预测（不 LN、不用真实中间帧）。"""
    x = raw
    for piece, tau in zip(pieces(controls, parts), parts):
        x = model(x, piece, _cycles(tau), grid)
    return x


def direct_prediction(model, window):
    """Φ_T(z)：v7 evaluation.point_metrics 的前 3 行原样（在 T/2 处切开再接上）。返回 (direct, pair, tau)。"""
    raw, grid = window.inputs.raw_z, window.inputs.grid
    controls, duration = window.inputs.controls, window.inputs.duration
    tau = duration*.5
    pair = comparison.split_action_segments(controls, tau=tau, sigma=tau)
    direct = model(raw, comparison.concatenate_action_segments(pair.first, pair.second), duration, grid)
    return direct, pair, tau


def direct_pair(model, window):
    """v7 evaluation.point_metrics 的 direct、first、composed 4 行：返回 (direct, first, composed, tau)。"""
    raw, grid = window.inputs.raw_z, window.inputs.grid
    direct, pair, tau = direct_prediction(model, window)
    first = model(raw, pair.first, tau, grid)
    composed = model(first, pair.second, tau, grid)
    return direct, first, composed, tau


def _ln(a, b) -> float:
    return float(ln_metrics.rms_ln(a, b)[0])


def _in(a, b, raw16) -> float:
    return float(ln_metrics.rms_input_scale(a, b, raw16)[0])


def _jepa(a, b, config) -> float:
    return float(comparison.jepa_distance(a, b, config=config).per_sample[0])


def _finite(values: dict) -> dict:
    bad = [k for k, v in values.items() if type(v) is not float or not math.isfinite(v)]
    if bad:
        raise ValueError(f"nonfinite D4 values: {bad[:5]}")
    return values


@torch.no_grad()
def window_values(model, windows, config):
    """一个测试窗口上 10 §3.2–§3.6 的全部量、等价核对的新值和三角不等式结果。

    windows：{T: CacheWindow}，T 取遍 TCD_TIMES，是同一 (轨迹, 起点) 上 evaluation.load_test_window 读出的
    16×16 单点窗口（起点帧相同，动作裁到 [s, s+T)）。返回 (values, equivalence, triangle)：values 的键是
    value_names()，equivalence 的键是 equivalence_names()，triangle 是 {"T=3": bool, "T=12": bool}。
    """
    values, equivalence, direct, half = {}, {}, {}, {}
    for t in TCD_TIMES:                                                    # 10 §3.2 tcd_ln；§4.3 future、tcd_normalized
        w = windows[t]
        raw = w.inputs.raw_z
        d, first, composed, tau = direct_pair(model, w)
        record = comparison.TemporalPredictions(d, first, composed, d-composed, tau.clone(), tau.clone())
        tcd_rms = float(metrics.temporal_composition_rms_defect(record).values[0])
        equivalence[f"future_jepa_distance:T={_t(t)}"] = _jepa(d, w.target.raw_z, config)
        equivalence[f"tcd_normalized:T={_t(t)}"] = tcd_rms/math.sqrt(float(comparison.input_scale(raw)[0]))
        values[f"tcd_ln:T={_t(t)}"] = _ln(d, composed)/((t/2)**2)
        direct[t], half[t] = d, composed
    for t in TCD_TIMES:                                                    # 10 §3.2 scd_ln；§4.3 scd_normalized
        w = windows[t]
        raw, controls, duration = w.inputs.raw_z, w.inputs.controls, w.inputs.duration
        for h in SCD_GRIDS:
            transfer = comparison.build_spatial_transfer(w.inputs.grid, target_height=h, target_width=h)
            fine = comparison.apply_spatial_transfer(model(raw, controls, duration, transfer.source), transfer)
            coarse = model(comparison.apply_spatial_transfer(raw, transfer), controls, duration, transfer.target)
            record = comparison.SpatialPredictions(fine, coarse, fine-coarse, duration.clone(), transfer.target)
            scd_rms = float(metrics.spatial_commutation_rms_defect(record).values[0])
            equivalence[f"scd_normalized:h={h}:T={_t(t)}"] = scd_rms/math.sqrt(float(comparison.input_scale(raw)[0]))
            values[f"scd_ln:h={h}:T={_t(t)}"] = _ln(fine, coarse)/t
    for t, parts in PARTITIONS:                                            # 10 §3.3
        w = windows[t]
        raw = w.inputs.raw_z
        composed = half[t] if parts == (t/2, t/2) else rollout(model, raw, w.inputs.controls, parts, w.inputs.grid)
        label = partition_label(parts)
        values[f"pd_ln:T={_t(t)}:{label}"] = _ln(direct[t], composed)
        values[f"pd_in:T={_t(t)}:{label}"] = _in(direct[t], composed, raw)
    for t, n in CROSS:                                                     # 10 §3.4
        w = windows[t]
        raw = w.inputs.raw_z
        transfer = comparison.build_spatial_transfer(w.inputs.grid, target_height=8, target_width=8)
        fine = comparison.apply_spatial_transfer(direct[t], transfer)
        coarse = rollout(model, comparison.apply_spatial_transfer(raw, transfer), w.inputs.controls, (t/n,)*n, transfer.target)
        values[f"xpd_ln:T={_t(t)}:n={n}"] = _ln(fine, coarse)
        values[f"xpd_in:T={_t(t)}:n={n}"] = _in(fine, coarse, raw)
    for d, parts in UNEQUAL:                                               # 10 §3.5
        w = windows[d]
        raw = w.inputs.raw_z
        composed = rollout(model, raw, w.inputs.controls, parts, w.inputs.grid)
        label = partition_label(parts)
        scale = parts[0]*parts[1]
        values[f"tcd_unequal_ln:d={_t(d)}:{label}"] = _ln(direct[d], composed)/scale
        values[f"tcd_unequal_in:d={_t(d)}:{label}"] = _in(direct[d], composed, raw)/scale
    triangle = {}
    for t, parts in HELDOUT:                                               # 10 §3.6
        w = windows[t]
        composed = rollout(model, w.inputs.raw_z, w.inputs.controls, parts, w.inputs.grid)
        target = w.target.raw_z
        a, b, c = _jepa(direct[t], target, config), _jepa(direct[t], composed, config), _jepa(composed, target, config)
        values[f"heldout_direct:T={_t(t)}"] = a
        values[f"heldout_partition:T={_t(t)}"] = b
        values[f"heldout_rollout:T={_t(t)}"] = c
        triangle[f"T={_t(t)}"] = triangle_holds(a, b, c)
    if tuple(values) != value_names() or tuple(equivalence) != equivalence_names():
        raise RuntimeError("D4 value names differ from the registered order")
    return _finite(values), _finite(equivalence), triangle


@torch.no_grad()
def long_values(model, windows, config):
    """一条测试轨迹（起点 0）上 10 §3.7 的量。windows：{T: CacheWindow}，T 取遍 LONG_TIMES。"""
    values = {}
    for t in LONG_TIMES:
        w = windows[t]
        raw, target = w.inputs.raw_z, w.target.raw_z
        d, _, _ = direct_prediction(model, w)
        composed = rollout(model, raw, w.inputs.controls, (LONG_BLOCK,)*int(t/LONG_BLOCK), w.inputs.grid)
        values[f"long_direct:T={_t(t)}"] = _jepa(d, target, config)
        values[f"long_composed:T={_t(t)}"] = _jepa(composed, target, config)
        values[f"long_pd_ln:T={_t(t)}"] = _ln(d, composed)
        values[f"long_pd_in:T={_t(t)}"] = _in(d, composed, raw)
    if tuple(values) != long_value_names():
        raise RuntimeError("D4 long-horizon names differ from the registered order")
    return _finite(values)


def compare_equivalence(new, old):
    """一个值的等价核对：返回 (在容差内, 逐位相同, 绝对差, 相对差或 None)。"""
    if type(old) not in (int, float) or not math.isfinite(old):
        raise ValueError("reference value must be a finite number")
    old = float(old)
    difference = abs(new-old)
    return (difference <= EQUIVALENCE_TOLERANCE*max(1., abs(old)), new.hex() == old.hex(), difference,
            difference/abs(old) if old != 0 else None)


def triangle_holds(direct, partition, rollout_value) -> bool:
    """10 §3.6 的核对：direct ≤ partition + rollout + 1e-6·max(1, direct)。"""
    return direct <= partition+rollout_value+TRIANGLE_TOLERANCE*max(1., direct)


def windows_index():
    """D4 的测试窗口：全部测试轨迹 × datasets.test_start_ticks（和 evaluation.build_index 的窗口相同，正式运行 400 个）。"""
    ds = data.datasets
    return [(tid, start) for tid in ds.active().split_range("test") for start in ds.test_start_ticks(tid)]


def load_long_test_window(features, trajectory_id, duration, device):
    """evaluation.load_test_window 的长时程版：核法相同，起点固定为 tick 0。"""
    if type(trajectory_id) is not int or trajectory_id not in data.datasets.active().split_range("test"):
        raise ValueError("long-horizon window requires a test trajectory")
    approved = data.approved_cache_metadata(features, splits=("test",))
    window = long_window.read_long_window(Path(features)/f"{trajectory_id:04d}", start_tick=0, duration_cycles=duration,
                                          expected_split="test", device=str(device))
    x, y, p = window.inputs, window.target, window.provenance
    data._window_boundary(window, window)
    if (x.trajectory_id != trajectory_id or x.split != "test"
            or p["actual_manifest_sha256"] != approved["manifest_sha256_by_id"][trajectory_id]
            or p["generation"]["evidence_role"] != "real_frozen_encoder" or p["observation_variant"] != "canonical"
            or float(x.duration[0]) != duration or float(x.observed_time_seconds[0]) != 0.
            or float(y.target_time_seconds[0]) != duration*.25
            or x.controls.source_ids != (f"trajectory-{trajectory_id:04d}",)):
        raise ValueError("long-horizon test identity/source/origin/endpoint mismatch")
    return window


def reference_rows(job_dir, condition):
    """被读任务自己的标准评测记录：{(kind, 轨迹, 起点, 时长, 网格): values}。查询列表必须和当前 build_index 相同。"""
    from ..workflow import read
    folder = Path(job_dir)/"evaluation"/condition
    if (folder/"FAILED.json").exists() or not (folder/"COMPLETE.json").exists():
        raise ValueError(f"reference evaluation {folder} did not complete")
    registration = read(folder/"registration.json")
    if registration["queries"] != [asdict(q) for q in evaluation.build_index()]:
        raise ValueError("reference evaluation queries differ from the current evaluation.build_index()")
    found = {}
    for path in sorted((folder/"rows").glob("*.json")):
        row = read(path)
        q = row["query"]
        key = (q["kind"], q["trajectory_id"], q["start_tick"], float(q["duration_cycles"]), q["grid_size"])
        if row["condition"] != condition or row["seed"] != checks.model_seed() or row["status"] != "ok" or key in found:
            raise ValueError(f"reference row {path.name} has another condition/seed, failed or is duplicated")
        found[key] = row["values"]
    expected = {(q["kind"], q["trajectory_id"], q["start_tick"], float(q["duration_cycles"]), q["grid_size"])
                for q in registration["queries"]}
    if set(found) != expected:   # 10 §4.3：登记的每个查询恰好一行，缺行、多行都失败（Codex 评审 2026-11-09）
        raise ValueError(f"reference evaluation {folder} rows do not cover the registered queries: "
                         f"{len(expected - set(found))} missing, {len(set(found) - expected)} unregistered")
    return found


def _reference_value(reference, tid, start, name):
    metric, *rest = name.split(":")
    fields = dict(item.split("=") for item in rest)
    t = float(fields["T"])
    key = ("pair", tid, start, t, int(fields["h"])) if metric == "scd_normalized" else ("point", tid, start, t, 16)
    if key not in reference or metric not in reference[key]:
        raise ValueError(f"reference row missing for {key} {metric}")
    return reference[key][metric]


def _macro(rows, names):
    """轨迹宏平均：先在每条轨迹内对窗口平均，再在轨迹间平均（13 §3 的前两层）。"""
    by_trajectory = {}
    for row in rows:
        by_trajectory.setdefault(row["trajectory_id"], []).append(row["values"])
    out = {}
    for name in names:
        means = [sum(v[name] for v in group)/len(group) for group in by_trajectory.values()]
        out[name] = sum(means)/len(means) if means else None
    return out


def run_d4(features, output, device, condition, *, training_run):
    """一个 D4 任务：读 training_run 的选中检查点（10 §4.2），在全部测试窗口和测试轨迹上算 §3 的量和 §4.3 的等价核对，
    按 10 §7 写 output（必须是新目录）。全部核对通过才写 COMPLETE.json，否则写 FAILED.json 并报 ValueError。"""
    if condition not in v7load.D4_CONDITIONS:
        raise ValueError(f"--condition 必须是 {v7load.D4_CONDITIONS} 之一")
    features, output = Path(features), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    job_dir = Path(training_run).resolve()
    from ..v7jobs import _job_dir
    job_dir = _job_dir(job_dir)
    job = dict(kind="d4", condition=condition, dataset=data.datasets.active().tag, seed=checks.model_seed(),
               updates=checks.v7_updates(), training_run=str(Path(training_run).resolve()), job_dir=str(job_dir))
    cache._write_json(output/"job.json", job)
    cache._write_json(output/"runtime.json", environment.runtime_signature(device))
    stage = "load"
    try:
        model, loss = comparison.build_condition(condition, device)
        source = v7load.load_v7_selected(job_dir, condition, model, loss_config=loss, features=features, device=device)
        cache._write_json(output/"source.json", source)
        reference = reference_rows(job_dir, condition)
        (output/"rows").mkdir()
        (output/"long").mkdir()
        modes = {m: m.training for m in model.modules()}
        identity = evaluation._state_identity(model)
        rows, longs = [], []
        stats = {m: dict(count=0, bitwise_equal=0, max_abs_diff=0., max_rel_diff=0., outside_tolerance=0) for m in EQUIVALENCE_METRICS}
        violations = []
        try:
            model.eval()
            stage = "windows"
            for index, (tid, start) in enumerate(windows_index()):
                windows = {t: evaluation.load_test_window(features, evaluation.EvaluationQuery(tid, start, t, 16, "point"), device)
                           for t in TCD_TIMES}
                values, new, triangle = window_values(model, windows, loss)
                pairs = {}
                for name, value in new.items():
                    old = _reference_value(reference, tid, start, name)
                    ok, bitwise, difference, relative = compare_equivalence(value, old)
                    entry = stats[name.split(":")[0]]
                    entry["count"] += 1; entry["bitwise_equal"] += bitwise; entry["outside_tolerance"] += not ok
                    entry["max_abs_diff"] = max(entry["max_abs_diff"], difference)
                    entry["max_rel_diff"] = max(entry["max_rel_diff"], relative or 0.)
                    pairs[name] = dict(new=value, old=float(old), within_tolerance=ok)
                for key, ok in triangle.items():
                    if not ok:
                        violations.append(dict(index=index, trajectory_id=tid, start_tick=start, heldout=key))
                row = dict(index=index, trajectory_id=tid, start_tick=start, values=values, equivalence=pairs,
                           triangle_ok=triangle, manifest_sha256=windows[1.].provenance["actual_manifest_sha256"])
                cache._write_json(output/"rows"/f"{index:05d}.json", row)
                rows.append(row)
            stage = "long"
            for tid in data.datasets.active().split_range("test"):
                windows = {t: load_long_test_window(features, tid, t, device) for t in LONG_TIMES}
                row = dict(trajectory_id=tid, start_tick=0, values=long_values(model, windows, loss),
                           manifest_sha256=windows[LONG_TIMES[0]].provenance["actual_manifest_sha256"])
                cache._write_json(output/"long"/f"{tid:04d}.json", row)
                longs.append(row)
        finally:
            for module, flag in modes.items():
                module.training = flag
        if evaluation._state_identity(model) != identity:
            raise RuntimeError("D4 changed parameters, gradients or buffers")
        failed = [m for m, s in stats.items() if s["outside_tolerance"]]
        problems = ([f"equivalence outside tolerance: {failed}"] if failed else []) + \
                   ([f"triangle inequality violated in {len(violations)} cases"] if violations else [])
        long_means = {}
        for name in long_value_names():
            vals = [r["values"][name] for r in longs]
            long_means[name] = sum(vals)/len(vals) if vals else None
        report = dict(job=job, status="failed" if problems else "passed", problems=problems,
                      counts=dict(windows=len(rows), long_trajectories=len(longs), test_trajectories=len({r["trajectory_id"] for r in rows})),
                      equivalence=dict(tolerance=EQUIVALENCE_TOLERANCE, passed=not failed, metrics=stats),
                      triangle=dict(tolerance=TRIANGLE_TOLERANCE, violations=len(violations), cases=violations),
                      trajectory_macro_means=_macro(rows, value_names()), long_means=long_means,
                      definitions=definitions(), device=str(device), scientific_conclusions_verified=False)
        cache._write_json(output/"report.json", report)
        if problems:
            raise ValueError("; ".join(problems))
        cache._write_json(output/"COMPLETE.json", dict(report_sha256=cache.file_sha(output/"report.json")))
        return report
    except Exception as error:
        cache._write_json(output/"FAILED.json", dict(stage=stage, error_type=type(error).__name__, message=str(error)))
        raise
