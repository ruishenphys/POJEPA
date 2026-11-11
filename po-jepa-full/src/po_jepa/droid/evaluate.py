"""DROID 评测轴（docs/design/10-evaluation.md §8，S11a、S11b）。

S11a 做冻结测试窗口上的各轴（10 §8.2 的表、§8.3 动作干预、§8.6 AC 的处理、§8.7 成败分开、§8.9 的 rows/续跑/report）
和 `commanded` 配置的跳过（§8.1 最后一条）。S11b 加原始帧率（§8.4，`evaluate_frame_rate`）、原始分辨率（§8.5，
`evaluate_resolution`），由 `evaluate_job` 的 `extra_reader` 打开；截断间隙（§8.8）是单独的 CPU 任务，在
`droid/spectral_gap.py`；读检查点（§8.1 前三条）是 `load_selected`（S09b），整个评测任务的命令行在 `droid/eval_job.py`。

记号（10 §8.2）：窗口 w =（回合，时长 d，起点行 s），z 是第 s 行特征，z*_T 是第 s+T 行，u 是第 s … s+T−1 个周期的
动作（`batches.make_eval_batch` 组装，和训练批相同）。每个窗口只算 T = d 的量。d(a, b) 是 JEPA 距离
`losses.jepa_distance(a, b).per_sample`（float32 的 mean|LN(a) − LN(b)|）。

前向一律是 `phi(raw, controls, duration, grid)`：没有 `bind` 时就是模型本身；有 `bind`（AC 的 `ac_state: true_state`，
09 §6.2 的 `droid.ac.bind`，S08b 实现）时是 `bind(model, start_state, reference)` 返回的包装，`start_state` 是窗口起点
的真实状态 `state[s]`（CPU float64 [B, 7]），`reference` 是这个窗口完整的 `ActionSegments`；动作干预另建一个以替身
动作为 `reference` 的包装（起点状态仍是本窗口的）。参数名和调用方式与 S12 `planning` 相同。

两种直接预测（10 §3.1）：
- `direct`：`phi(z, u, T)`，u 不切开。未来误差（09 §4.4，和验证相同）、`delta_act`、SCD 的细网格一侧用它。
- `direct_split`：u 在 T/2 处 `split_action_segments` 切开再 `concatenate_action_segments` 接上（v7
  `evaluation.point_metrics` 的写法）。TCD、分割差异 pd/xpd、不等分 TCD、留出分解、长时程用它。T 为偶数时两者的控制段
  相同；T = 1、3 时 T/2 落在一个周期内部，两者的分段不同。

每个窗口的量（键名见 `value_names(d, config)`）：

| 键 | 公式 | 时长 |
|---|---|---|
| future:g=16 | d(direct, z*) | 全部 |
| tcd_ln、tcd_normalized | RMS_LN(direct_split, Φ_{T/2}∘Φ_{T/2}) / (T/2)²；v7 tcd_rms / √s²(z) | 同分布、时长留出 |
| delta_act | d(phi_donor(z; u_donor), z*) − d(phi(z; u), z*)；POJEPA 走 v7 `action_intervention_margin` | 同分布 |
| future:g=h | d(Φ^h(R_h z), R_h z*)，h = 8、12 | 同分布；h = 12 另加 3、12 |
| scd_ln:h、scd_normalized:h | RMS_LN(R_h direct, Φ^h(R_h z)) / T；v7 scd_rms / √s²(z) | 同分布 |
| heldout_direct/partition/rollout | d(direct_split, z*)、d(direct_split, Φ_Π)、d(Φ_Π, z*)，Π = 1+2、8+4、8+8 | 时长留出 |
| long_direct、long_composed、long_pd_ln、long_pd_in | d(direct_split, z*)、d(Φ_{8,…,8}, z*)、两者间两种度量 | 长时程 |
| pd_ln/pd_in:Π | RMS_LN、RMS_in(direct_split, Φ_Π)，不除时间（10 §3.3 的 7 个分割） | 4、8 |
| xpd_ln/xpd_in:n=k | (R_8 direct_split, Φ^8_{Π_k}(R_8 z))，8×8 上算，s² 用 16×16 起点（10 §3.4） | 4、8 |
| tcd_unequal_ln/in:τ+σ | (direct_split, Φ_σ∘Φ_τ) / (τσ)（10 §3.5） | 4、8 |

`delta_act` 的替身来自被跳过的回合（只在 `commanded` 配置下会发生）时记 None，计数写进报告。
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from po_jepa import ln_metrics, metrics
from po_jepa.actions import ActionSegments, concatenate_action_segments, from_physical_segments, split_action_segments
from po_jepa.droid import batches as bt
from po_jepa.droid import index as ix
from po_jepa.droid import windows as wn
from po_jepa.droid.actions import poses_to_diffs
from po_jepa.droid.poses import compute_new_pose
from po_jepa.droid.config import DroidConfig
from po_jepa.geometry import apply_spatial_transfer, build_spatial_transfer, regular_grid
from po_jepa.losses import LossConfig, SpatialPredictions, TemporalPredictions, input_scale, jepa_distance
from po_jepa.model import POJEPA
from po_jepa.runtime import cache, evaluation
from po_jepa.runtime.addon import d4

SCHEMA_ROWS = "droid_eval_rows_v1"
SCHEMA_REPORT = "droid_eval_report_v1"
TEST_SPLITS = ("id_test", "ood_test")
OUTCOMES = ix.OUTCOMES
GRID = bt.GRID
SCD_GRIDS = (8, 12)
HELDOUT_PARTS = {3: (1., 2.), 12: (8., 4.), 16: (8., 8.)}     # 10 §8.2：3 = 1+2、12 = 8+4、16 = 8+8（先前者后后者）
JOINT_HELDOUT = (3, 12)                                       # 10 §8.2：时长和网格同时留出（12×12）
PARTITIONS = d4.PARTITIONS                                    # 10 §3.3 的 7 个分割（T = 4、8）
CROSS = d4.CROSS                                              # 10 §3.4：(T, n)
UNEQUAL = d4.UNEQUAL                                          # 10 §3.5：(d, (τ, σ))
PARTITION_DURATIONS = (4, 8)
TRIANGLE_TOLERANCE = d4.TRIANGLE_TOLERANCE
AC_CONDITIONS = ("vjepa2_ac", "vjepa2_ac_small", "vjepa2_ac_zeroshot")
INPUT_SPACE_METRICS = ("tcd_normalized", "scd_normalized:h=8", "scd_normalized:h=12")   # AC 不可比（10 §3.8、§8.6）
AC_INTERPOLATED_TCD = (1, 3)                                  # 10 §8.6：半程 0.5、1.5 是 AC 的插值
partition_label = d4.partition_label

Bind = Callable[[object, torch.Tensor, ActionSegments], Callable]


# ---------------------------------------------------------------- 键名

def _axes(config: DroidConfig) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    e = config.evaluation
    if tuple(sorted(HELDOUT_PARTS)) != tuple(e.durations_heldout):
        raise ValueError(f"evaluation.durations_heldout {e.durations_heldout} has no registered decomposition "
                         f"(10 §8.2 registers {sorted(HELDOUT_PARTS)})")
    for t in e.durations_long:
        if t % e.composed_step:
            raise ValueError(f"long duration {t} is not a multiple of evaluation.composed_step {e.composed_step}")
    if not set(PARTITION_DURATIONS) <= set(e.durations_id) or not set(JOINT_HELDOUT) <= set(e.durations_heldout):
        raise ValueError("partition durations must be in-distribution and joint held-out durations must be held out")
    return tuple(e.durations_id), tuple(e.durations_heldout), tuple(e.durations_long)


def value_names(duration: int, config: DroidConfig) -> tuple[str, ...]:
    """时长 d 的窗口要算的全部键，按固定顺序（10 §8.2 的表）。"""
    ids, held, long = _axes(config)
    if duration not in (*ids, *held, *long):
        raise ValueError(f"duration {duration} is not an evaluation duration")
    names = ["future:g=16"]
    if duration in ids or duration in held:
        names += ["tcd_ln", "tcd_normalized"]
    if duration in ids:
        names += ["delta_act"]
        for h in SCD_GRIDS:
            names += [f"future:g={h}", f"scd_ln:h={h}", f"scd_normalized:h={h}"]
    if duration in JOINT_HELDOUT:
        names += ["future:g=12"]
    if duration in held:
        names += ["heldout_direct", "heldout_partition", "heldout_rollout"]
    if duration in long:
        names += ["long_direct", "long_composed", "long_pd_ln", "long_pd_in"]
    if duration in PARTITION_DURATIONS:
        for t, parts in PARTITIONS:
            if t == duration:
                names += [f"pd_{m}:{partition_label(parts)}" for m in ("ln", "in")]
        for t, n in CROSS:
            if t == duration:
                names += [f"xpd_{m}:n={n}" for m in ("ln", "in")]
        for d, parts in UNEQUAL:
            if d == duration:
                names += [f"tcd_unequal_{m}:{partition_label(parts)}" for m in ("ln", "in")]
    return tuple(names)


def definitions(config: DroidConfig) -> dict:
    """写进 report.json 的定义表。"""
    ids, held, long = _axes(config)
    return dict(
        durations=dict(in_distribution=list(ids), heldout=list(held), long=list(long),
                       joint_heldout=list(JOINT_HELDOUT), partition=list(PARTITION_DURATIONS)),
        names={str(d): list(value_names(d, config)) for d in sorted({*ids, *held, *long})},
        heldout={str(t): list(p) for t, p in HELDOUT_PARTS.items()},
        long=dict(block=config.evaluation.composed_step),
        partitions=[dict(T=t, parts=list(p), label=partition_label(p), x=d4.partition_x(t, p)) for t, p in PARTITIONS],
        cross_grid=[dict(T=t, n=n, coarse_grid=8) for t, n in CROSS],
        unequal=[dict(d=d, tau=p[0], sigma=p[1], label=partition_label(p)) for d, p in UNEQUAL],
        direct="future error, delta_act and the fine side of SCD use model(z, u, T); TCD, pd, xpd, tcd_unequal, "
               "heldout and long use u split at T/2 and concatenated (v7 evaluation.point_metrics)",
        triangle_tolerance=dict(rule="direct <= partition + rollout + 1e-6*max(1,direct)", value=TRIANGLE_TOLERANCE),
        macro="episode macro mean: mean over each episode's windows, then over episodes (13 §3, §8)")


# ---------------------------------------------------------------- 一组同时长窗口上的计算

def _full(batch: int, value: float) -> torch.Tensor:
    return torch.full((batch,), float(value), dtype=torch.float64)


def pieces(controls: ActionSegments, parts: Sequence[float]) -> list[ActionSegments]:
    """把时长 Σparts 的一批动作依次精确切成 len(parts) 段（第 i 次 split_action_segments(rest, τ_i, 剩余时长)）。"""
    b = controls.duration.numel()
    out, rest = [], controls
    for i, tau in enumerate(parts[:-1]):
        pair = split_action_segments(rest, tau=_full(b, tau), sigma=_full(b, sum(parts[i + 1:])))
        out.append(pair.first)
        rest = pair.second
    out.append(rest)
    return out


def rollout(phi, raw: torch.Tensor, controls: ActionSegments, parts: Sequence[float], grid) -> torch.Tensor:
    """Φ_Π(raw)：逐段完整调用，下一段的输入是上一段输出的原始预测（不 LN、不用真实中间帧，10 §3.1）。"""
    b = raw.shape[0]
    x = raw
    for piece, tau in zip(pieces(controls, parts), parts):
        x = phi(x, piece, _full(b, tau), grid)
    return x


def _predictor(model, bind: Bind | None, start_states: torch.Tensor, reference: ActionSegments):
    if bind is None:
        return model
    return bind(model, start_states.clone(), reference)


def _list(values: torch.Tensor) -> list[float]:
    out = [float(v) for v in values.detach().to("cpu", torch.float64)]
    if any(not math.isfinite(v) for v in out):
        raise ValueError("nonfinite evaluation value")
    return out


@torch.no_grad()
def group_values(model, batch, start_states: torch.Tensor, donor_batch, *, duration: int, config: DroidConfig,
                 loss_config: LossConfig, bind: Bind | None = None) -> tuple[dict[str, list], list[bool]]:
    """一组时长都等于 `duration` 的窗口（`make_eval_batch` 组装的 `batch`）上的全部量。

    `donor_batch` 是同序的替身窗口批（同分布时长才要），某一行为 None 时该行 `delta_act` 记 None：传入时
    `donor_batch = (批, 有效掩码)`，掩码为假的行用的是占位动作，结果丢弃；掩码全假时批可以是 None（不做替身前向）。返回 ({键: [B 个值]}, 三角不等式逐行结果)。
    """
    names = value_names(duration, config)
    ids, held, long = _axes(config)
    raw, target, controls, dur = batch.raw_z, batch.raw_target, batch.controls, batch.duration
    b = raw.shape[0]
    if not torch.equal(dur, _full(b, duration)):
        raise ValueError(f"batch durations are not all {duration}")
    grid16 = regular_grid(GRID, GRID)
    phi = _predictor(model, bind, start_states, controls)
    jepa = lambda a, y: jepa_distance(a, y, config=loss_config).per_sample
    scale = input_scale(raw).sqrt()
    out: dict[str, torch.Tensor | list] = {}
    triangle = [True]*b

    direct = phi(raw, controls, dur, grid16)                                           # 09 §4.4 的未来误差
    out["future:g=16"] = jepa(direct, target)
    direct_split = half = None
    if duration in ids or duration in held or duration in long:
        tau = dur*.5
        pair = split_action_segments(controls, tau=tau, sigma=tau)
        direct_split = phi(raw, concatenate_action_segments(pair.first, pair.second), dur, grid16)
    if duration in ids or duration in held:                                            # 10 §3.2 tcd_ln；v7 tcd_normalized
        first = phi(raw, pair.first, tau, grid16)
        half = phi(first, pair.second, tau, grid16)
        record = TemporalPredictions(direct_split, first, half, direct_split - half, tau.clone(), tau.clone())
        out["tcd_ln"] = ln_metrics.rms_ln(direct_split, half)/((duration/2)**2)
        out["tcd_normalized"] = metrics.temporal_composition_rms_defect(record).values/scale.to(direct.device)
    if duration in ids:
        donor, valid = donor_batch
        if donor is None:                                                              # 整组替身都被跳过
            if any(valid):
                raise ValueError("a donor batch is required when some donor is valid")
            delta = None
        elif isinstance(model, POJEPA) and bind is None:                               # 10 §8.3
            delta = metrics.action_intervention_margin(model, raw, controls, donor.controls, dur, grid16, target,
                                                       loss_config=loss_config).values
        else:
            metrics._action_pair(raw, controls, donor.controls, dur)
            phi_donor = _predictor(model, bind, start_states, donor.controls)
            delta = jepa(phi_donor(raw, donor.controls, dur, grid16), target) - out["future:g=16"]
        out["delta_act"] = [None]*b if delta is None else [v if ok else None for v, ok in zip(_list(delta), valid)]
    for h in SCD_GRIDS:
        if duration in ids or (h == 12 and duration in JOINT_HELDOUT):
            transfer = build_spatial_transfer(grid16, target_height=h, target_width=h)
            coarse_raw = apply_spatial_transfer(raw, transfer)
            coarse = phi(coarse_raw, controls, dur, transfer.target)
            out[f"future:g={h}"] = jepa(coarse, apply_spatial_transfer(target, transfer))
            if duration in ids:                                                        # 10 §3.2 scd_ln；v7 scd_normalized
                fine = apply_spatial_transfer(direct, transfer)
                record = SpatialPredictions(fine, coarse, fine - coarse, dur.clone(), transfer.target)
                out[f"scd_ln:h={h}"] = ln_metrics.rms_ln(fine, coarse)/duration
                out[f"scd_normalized:h={h}"] = metrics.spatial_commutation_rms_defect(record).values/scale.to(direct.device)
    if duration in held:                                                               # 10 §3.6
        composed = rollout(phi, raw, controls, HELDOUT_PARTS[duration], grid16)
        a, p, r = jepa(direct_split, target), jepa(direct_split, composed), jepa(composed, target)
        out["heldout_direct"], out["heldout_partition"], out["heldout_rollout"] = a, p, r
        triangle = [d4.triangle_holds(x, y, z) for x, y, z in zip(_list(a), _list(p), _list(r))]
    if duration in long:                                                               # 10 §3.7
        step = float(config.evaluation.composed_step)
        composed = rollout(phi, raw, controls, (step,)*int(duration//step), grid16)
        out["long_direct"] = jepa(direct_split, target)
        out["long_composed"] = jepa(composed, target)
        out["long_pd_ln"] = ln_metrics.rms_ln(direct_split, composed)
        out["long_pd_in"] = ln_metrics.rms_input_scale(direct_split, composed, raw)
    if duration in PARTITION_DURATIONS:
        for t, parts in PARTITIONS:                                                    # 10 §3.3
            if t != duration:
                continue
            composed = half if tuple(parts) == (t/2, t/2) else rollout(phi, raw, controls, parts, grid16)
            label = partition_label(parts)
            out[f"pd_ln:{label}"] = ln_metrics.rms_ln(direct_split, composed)
            out[f"pd_in:{label}"] = ln_metrics.rms_input_scale(direct_split, composed, raw)
        transfer = batch.transfer                                                      # 16→8，和训练相同
        for t, n in CROSS:                                                             # 10 §3.4
            if t != duration:
                continue
            fine = apply_spatial_transfer(direct_split, transfer)
            coarse = rollout(phi, apply_spatial_transfer(raw, transfer), controls, (t/n,)*n, transfer.target)
            out[f"xpd_ln:n={n}"] = ln_metrics.rms_ln(fine, coarse)
            out[f"xpd_in:n={n}"] = ln_metrics.rms_input_scale(fine, coarse, raw)
        for d, parts in UNEQUAL:                                                       # 10 §3.5
            if d != duration:
                continue
            composed = rollout(phi, raw, controls, parts, grid16)
            label, s = partition_label(parts), parts[0]*parts[1]
            out[f"tcd_unequal_ln:{label}"] = ln_metrics.rms_ln(direct_split, composed)/s
            out[f"tcd_unequal_in:{label}"] = ln_metrics.rms_input_scale(direct_split, composed, raw)/s
    if tuple(out) != names:
        raise RuntimeError(f"computed names {tuple(out)} differ from the registered order {names}")
    return {k: (v if isinstance(v, list) else _list(v)) for k, v in out.items()}, triangle


# ---------------------------------------------------------------- 一个测试集

def episode_outcomes(data_root: Path | str) -> dict[str, str]:
    """`index/episodes.jsonl` 里每条回合的 outcome（success / failure，10 §8.7）。"""
    path = ix.index_dir(data_root)/ix.EPISODES_FILE
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = ix._loads(line.encode("utf-8"), str(path))
            out[e["uuid"]] = e["outcome"]
    return out


def _windows_doc(doc: Mapping, split: str) -> None:
    if doc.get("schema") != "droid_windows_v1" or doc.get("split") != split:
        raise ValueError(f"not the frozen windows file of {split}")


def _selected(doc: Mapping, prepared: Mapping, source: str) -> tuple[list[dict], list[str], set[str]]:
    """`commanded` 配置跳过 `passed_<source>` 为假的回合（10 §8.1）；冻结窗口本来只含 `passed_realized` 的回合。"""
    episodes = list(dict.fromkeys(w["uuid"] for w in doc["windows"]))
    missing = [u for u in episodes if u not in prepared]
    if missing:
        raise ValueError(f"no prepared episode for {missing[:3]}")
    skipped = {u for u in episodes if prepared[u].passed(source) is not True}
    kept = [w for w in doc["windows"] if w["uuid"] not in skipped]
    return kept, [u for u in episodes if u not in skipped], skipped


def _group(windows: Sequence[Mapping]) -> dict[int, list[int]]:
    """一批里按时长分组（保持文件顺序）：{d: [在批里的下标]}。"""
    groups: dict[int, list[int]] = {}
    for i, w in enumerate(windows):
        groups.setdefault(int(w["duration"]), []).append(i)
    return groups


def chunk_rows(chunk: Sequence[Mapping], *, model, doc: Mapping, skipped: set[str], reader, episodes, outcomes,
               control_spec, provenance, loss_config, config, device, bind=None) -> list[dict]:
    """一批冻结窗口（≤ 64 个，文件顺序）的逐窗口记录：编号、回合 outcome、§8.2–§8.3 的全部量、三角不等式。"""
    ids, _, _ = _axes(config)
    records: list[dict | None] = [None]*len(chunk)
    by_id = {w["id"]: w for w in doc["windows"]}
    for duration, index in _group(chunk).items():
        group = [chunk[i] for i in index]
        batch, states = bt.make_eval_batch(group, reader, episodes, control_spec, provenance, device)
        donor = None
        if duration in ids:
            donors, valid = [], []
            for w in group:
                other = by_id[doc["donors"][w["id"]]]
                if other["duration"] != duration or other["uuid"] == w["uuid"]:
                    raise ValueError(f"donor of {w['id']} is not another episode's window of the same duration")
                ok = other["uuid"] not in skipped
                donors.append(other if ok else w)       # 占位：替身回合被跳过时借本窗口的动作，结果丢弃
                valid.append(ok)
            if not any(valid):
                # 整组的替身回合都被跳过（如按 64 切批后这个时长只剩一个窗口）：不做替身前向，delta_act 全记 None（10 §8.2）
                donor = (None, valid)
            elif not all(valid):
                # 占位行借本窗口的动作时 source id 相同，_action_pair 会拒；换成同组里编号不同的另一个可用窗口（结果仍丢弃）
                pool = group + [d for d, ok in zip(donors, valid) if ok]
                for k, (w, ok) in enumerate(zip(group, valid)):
                    if not ok:
                        spare = next((x for x in pool if x["id"] != w["id"]), None)
                        if spare is None:
                            raise ValueError(f"no placeholder donor for {w['id']}")
                        donors[k] = spare
            if donor is None:
                donor_batch, _ = bt.make_eval_batch(donors, reader, episodes, control_spec, provenance, device)
                donor = (donor_batch, valid)
        values, triangle = group_values(model, batch, states, donor, duration=duration, config=config,
                                        loss_config=loss_config, bind=bind)
        for k, i in enumerate(index):
            w = chunk[i]
            records[i] = dict(id=w["id"], uuid=w["uuid"], duration=duration, start=w["start"],
                              outcome=outcomes[w["uuid"]], values={n: v[k] for n, v in values.items()},
                              triangle_ok=triangle[k])
    return records


def _macro(rows: Sequence[Mapping], name: str, duration: int, outcome: str | None) -> dict:
    by_episode: dict[str, list[float]] = {}
    for r in rows:
        if r["duration"] == duration and (outcome is None or r["outcome"] == outcome):
            v = r["values"][name]
            if v is not None:
                by_episode.setdefault(r["uuid"], []).append(v)
    means = [sum(v)/len(v) for v in by_episode.values()]
    return dict(episode_macro_mean=sum(means)/len(means) if means else None, episodes=len(means),
                windows=sum(len(v) for v in by_episode.values()))


def summarize(rows: Sequence[Mapping], config: DroidConfig) -> dict:
    """每个时长、每个键：全部、success、failure 三份回合宏平均（10 §8.7、13 §8）。"""
    out = {}
    for d in sorted({r["duration"] for r in rows}):
        out[str(d)] = {name: {"all": _macro(rows, name, d, None), **{o: _macro(rows, name, d, o) for o in OUTCOMES}}
                       for name in value_names(d, config)}
    return out


_ROW = re.compile(r"^\d{5}\.json$")


def evaluate_split(output: Path | str, *, model, split: str, doc: Mapping, prepared: Mapping, outcomes: Mapping[str, str],
                   reader, control_spec, provenance: dict, loss_config: LossConfig, config: DroidConfig, device,
                   bind: Bind | None = None) -> dict:
    """一个测试集的冻结窗口：`<output>/rows/<序号>.json` 每批一份（中断后重跑跳过已有的批），返回这个测试集的汇总。

    `doc` 是 `prepared/windows/<split>.json` 的内容，`prepared` 是 {uuid: `PreparedEpisode`}（至少含窗口里的回合），
    `outcomes` 是 {uuid: success/failure}。调用者负责 eval 模式和参数核对（`evaluate_job`）。"""
    _windows_doc(doc, split)
    source = config.action.source
    kept, kept_episodes, skipped = _selected(doc, prepared, source)
    if not kept:
        raise ValueError(f"{split}: no windows left after skipping episodes that failed {source}")
    missing = [u for u in kept_episodes if outcomes.get(u) not in OUTCOMES]
    if missing:
        raise ValueError(f"{split}: no success/failure outcome for {missing[:3]}")
    episodes = bt.episode_arrays(prepared, kept_episodes, source)
    rows_dir = Path(output)/"rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    for stale in rows_dir.glob("*.partial"):                   # 上次写到一半（cache._write_json 先写 .partial 再改名）
        stale.unlink()
    parts = bt.chunks(kept)
    extra = [p.name for p in rows_dir.iterdir() if not _ROW.match(p.name) or int(p.name[:5]) >= len(parts)]
    if extra:
        raise ValueError(f"{rows_dir} has files that are not rows of this evaluation: {sorted(extra)[:3]}")
    done = written = 0
    for i, chunk in enumerate(parts):
        path = rows_dir/f"{i:05d}.json"
        if path.exists():
            done += 1
            continue
        records = chunk_rows(chunk, model=model, doc=doc, skipped=skipped, reader=reader, episodes=episodes,
                             outcomes=outcomes, control_spec=control_spec, provenance=provenance,
                             loss_config=loss_config, config=config, device=device, bind=bind)
        cache._write_json(path, dict(schema=SCHEMA_ROWS, split=split, batch=i, windows=records))
        written += 1
    rows = []
    for i, chunk in enumerate(parts):                           # 10 §8.9：写 report 前核每个窗口恰好出现一次
        body = ix._loads((rows_dir/f"{i:05d}.json").read_bytes(), f"rows/{i:05d}.json")
        if (body.get("schema") != SCHEMA_ROWS or body.get("split") != split or body.get("batch") != i
                or [r["id"] for r in body["windows"]] != [w["id"] for w in chunk]):
            raise ValueError(f"{split} rows/{i:05d}.json does not hold batch {i} of the current windows")
        rows += body["windows"]
    if len({r["id"] for r in rows}) != len(rows) or {r["id"] for r in rows} != {w["id"] for w in kept}:
        raise ValueError(f"{split}: rows do not cover every window exactly once")
    violations = [dict(id=r["id"], duration=r["duration"]) for r in rows if not r["triangle_ok"]]
    skipped_windows = len(doc["windows"]) - len(kept)
    donor_skipped = sum(1 for r in rows if "delta_act" in r["values"] and r["values"]["delta_act"] is None)
    return dict(split=split, counts=dict(windows=len(rows), episodes=len(kept_episodes), batches=len(parts),
                                         batches_reused=done, batches_written=written,
                                         skipped_episodes=len(skipped), skipped_windows=skipped_windows,
                                         delta_act_donor_skipped=donor_skipped),
                skipped_episodes=sorted(skipped), triangle=dict(tolerance=TRIANGLE_TOLERANCE, violations=len(violations),
                                                                cases=violations),
                macro=summarize(rows, config))


# ---------------------------------------------------------------- 原始帧率（10 §8.4，S11b）

SCHEMA_FRAME_RATE = "droid_eval_frame_rate_v1"
SCHEMA_RESOLUTION = "droid_eval_resolution_v1"
FRAME_RATE_FILE = "frame_rate.json"
RESOLUTION_FILE = "resolution.json"

AcSteps = Callable[[object, torch.Tensor, torch.Tensor, torch.Tensor, object], torch.Tensor]


def frame_rate_plan(config: DroidConfig) -> dict:
    """10 §8.4 的帧数表：单步 F ∈ frame_rate_single ∪ {一个周期}、滚动步长 F ∈ frame_rate_rollout ∪ {一个周期}（滚
    total/F 步）、一次直接预测 total 帧；δ = F / frames_per_interval 个周期。s5 文件第 j 行是第 stride·j 帧。"""
    per, stride = config.time.frames_per_interval, config.encode.frame_rate_stride_frames
    e = config.evaluation
    total = e.frame_rate_total_frames
    single = tuple(sorted({*e.frame_rate_single, per}))
    roll = tuple(sorted({*e.frame_rate_rollout, per}))
    if any(f <= 0 or f % stride for f in (*single, *roll, total)) or total % per or any(total % f for f in roll):
        raise ValueError(f"frame-rate frames {single}, {roll}, {total} must be positive multiples of the s5 stride "
                         f"{stride}, the total a multiple of {per} and of every rollout step")
    return dict(per=per, stride=stride, total=total, single=single, rollout=roll, duration=wn.frame_rate_duration(config))


def frame_rate_names(config: DroidConfig) -> tuple[str, ...]:
    """原始帧率每个起点的键（固定顺序）：single:f=F、rollout:f=F、direct:f=total（F 是帧数）。"""
    p = frame_rate_plan(config)
    return (*(f"single:f={f}" for f in p["single"]), *(f"rollout:f={f}" for f in p["rollout"]), f"direct:f={p['total']}")


def frame_rate_definitions(config: DroidConfig) -> dict:
    p = frame_rate_plan(config)
    per = p["per"]
    return dict(
        frames_per_interval=per, s5_stride_frames=p["stride"], total_frames=p["total"],
        single=[dict(name=f"single:f={f}", frames=f, delta=f/per, s5_target_offset=f//p["stride"]) for f in p["single"]],
        rollout=[dict(name=f"rollout:f={f}", frames=f, delta=f/per, steps=p["total"]//f,
                      s5_target_offset=p["total"]//p["stride"]) for f in p["rollout"]],
        direct=dict(name=f"direct:f={p['total']}", frames=p["total"], duration=p["total"]//per, segments=p["total"]//per),
        control="u = poses_to_diffs([s_f, s_{f+F}]) / delta (01 §5.5), one segment [0, delta) per call, local origin 0; "
                "direct uses one per-cycle pose diff per cycle",
        ac="AC: one step = stride frames, actions are the physical stride-frame pose diffs (not divided by delta); a call "
           "of delta cycles = F/stride steps; the start state of each call is state_all[f0] integrated with "
           "compute_new_pose over the earlier steps",
        excluded="an entry whose per-cycle control u/delta leaves the ControlSpec physical bounds for any key is "
                 "excluded for every condition (all values None) and counted")


def _poses(prepared, uuid: str, source: str) -> tuple[np.ndarray, np.ndarray]:
    """全部步的位姿（动作源的 state_all 或 command_all）和真实状态 state_all，float64 [T, 7]。"""
    arrays = prepared[uuid].arrays
    poses, states = arrays["command_all" if source == "commanded" else "state_all"], arrays["state_all"]
    for a in (poses, states):
        if a.dtype != np.float64 or a.ndim != 2 or a.shape[1] != 7:
            raise ValueError(f"episode {uuid}: full-rate poses must be float64 [T, 7]")
    return poses, states


def _diff(poses: np.ndarray, frames: Sequence[int]) -> np.ndarray:
    """poses_to_diffs(poses[frames])，float64 [len(frames) − 1, 7]。"""
    return np.asarray(poses_to_diffs(poses[list(frames)]), dtype=np.float64)


def _segments(values: np.ndarray, lengths: Sequence[float], ids: Sequence[str], control_spec, device) -> ActionSegments:
    """物理控制值 float64 [B, J, 7]（每段一个），段长 `lengths`（周期），局部原点 0 → v7 `ActionSegments`。"""
    b, j = values.shape[:2]
    ends = torch.tensor(np.cumsum(np.asarray(lengths, dtype=np.float64)), dtype=torch.float64)
    starts = torch.cat([torch.zeros(1, dtype=torch.float64), ends[:-1]])
    period = control_spec.control_period_seconds
    return from_physical_segments(
        starts_seconds=(starts*period).expand(b, j).clone(), ends_seconds=(ends*period).expand(b, j).clone(),
        u_values=torch.from_numpy(values.astype(np.float32)).to(device), n_segments=torch.full((b,), j, dtype=torch.int64),
        segment_valid=torch.ones(b, j, dtype=torch.bool), origin_seconds=torch.zeros(b, dtype=torch.float64),
        control_spec=control_spec, source_ids=tuple(ids))


def _in_bounds(values: np.ndarray, control_spec) -> bool:
    """和 `from_physical_segments` 相同的判法（float32 比较物理界）。"""
    v = torch.from_numpy(np.asarray(values, dtype=np.float32))
    lo, hi = v.new_tensor(control_spec.lower_bounds), v.new_tensor(control_spec.upper_bounds)
    return bool(torch.isfinite(v).all() and not torch.any((v < lo) | (v > hi)))


def frame_rate_controls(poses: np.ndarray, f0: int, plan: dict) -> dict[str, list[np.ndarray]]:
    """一个起点每个键的物理控制值：{键: [每次调用的 float64 [J, 7]]}（单步 1 次、滚动 total/F 次、直接 1 次 J = total/per）。"""
    per, total = plan["per"], plan["total"]
    out = {}
    for f in plan["single"]:
        out[f"single:f={f}"] = [_diff(poses, (f0, f0 + f))/(f/per)]
    for f in plan["rollout"]:
        out[f"rollout:f={f}"] = [_diff(poses, (f0 + k*f, f0 + (k + 1)*f))/(f/per) for k in range(total//f)]
    out[f"direct:f={total}"] = [_diff(poses, range(f0, f0 + total + 1, per))]
    return out


def ac_frame_rate_actions(poses: np.ndarray, states: np.ndarray, f0: int, plan: dict) -> dict[str, list[tuple]]:
    """AC 的小步（01 §5.4、10 §8.4）：{键: [每次调用的 (物理动作 float64 [F/stride, 7]，起点状态 float64 [7])]}。
    小步动作是相邻 stride 帧的物理位姿差，不除以 δ；第 k 次调用的起点状态 = state_all[f0] 用 compute_new_pose 按前面的
    小步积分（第一次就是 state_all[f0]）。"""
    stride, total, per = plan["stride"], plan["total"], plan["per"]
    small = _diff(poses, range(f0, f0 + total + 1, stride))                     # [total/stride, 7]
    state = torch.from_numpy(states[f0].copy())[None, None]                     # [1, 1, 7]
    integrated = [state[0, 0].numpy().copy()]
    for a in small:
        state = compute_new_pose(state, torch.from_numpy(a.copy())[None, None])
        integrated.append(state[0, 0].numpy().copy())
    out = {}
    for f in plan["single"]:
        out[f"single:f={f}"] = [(small[:f//stride], integrated[0])]
    for f in plan["rollout"]:
        n = f//stride
        out[f"rollout:f={f}"] = [(small[k*n:(k + 1)*n], integrated[k*n]) for k in range(total//f)]
    out[f"direct:f={total}"] = [(small, integrated[0])]
    return out


def _frame_rate_entries(doc: Mapping, skipped: set[str], config: DroidConfig) -> list[dict]:
    """冻结窗口文件里的原始帧率起点（06 §3），核编号、起点帧 = 15·s 且有对应的 d = 4 窗口；跳过 `commanded` 没通过的回合。"""
    plan = frame_rate_plan(config)
    extra = set(doc.get("extra_episodes", []))
    windows = {w["id"] for w in doc["windows"]}
    out = []
    for e in doc.get("frame_rate", []):
        f0 = _int_value(e["start_frame"], "start_frame")
        if (e["id"] != wn.frame_rate_id(e["uuid"], f0) or f0 % plan["per"] or e["uuid"] not in extra
                or wn.window_id(e["uuid"], plan["duration"], f0//plan["per"]) not in windows):
            raise ValueError(f"frame-rate entry {e['id']!r} is not a start frame 15·s of a d = {plan['duration']} "
                             "window of an extra episode")
        if e["uuid"] not in skipped:
            out.append(dict(e))
    return out


def _int_value(value, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an int, got {value!r}")
    return value


def _extra_record(extra_reader, uuid: str, kind: str, episodes, prepared) -> dict:
    """附加特征和准备文件配对（准备 JSON 的 SHA），并返回特征 JSON。"""
    record = extra_reader.record(uuid, kind)
    if record.get("prepared_sha256") != episodes[uuid].prepared_sha256:
        raise ValueError(f"episode {uuid} {kind}: the test-extra features were encoded from another prepared file")
    return record


def _s5_rows(extra_reader, uuid: str, kind: str, rows: Sequence[int], stride: int, episodes, prepared) -> torch.Tensor:
    record = _extra_record(extra_reader, uuid, kind, episodes, prepared)
    index = record.get("frame_index")
    if not isinstance(index, list) or any(r >= len(index) or index[r] != stride*r for r in rows):
        raise ValueError(f"episode {uuid} {kind}: rows {list(rows)} are not frames {[stride*r for r in rows]} "
                         f"(the s5 file has {len(index) if isinstance(index, list) else None} rows)")
    return extra_reader.rows(uuid, kind, rows)


def _macro_values(rows: Sequence[Mapping], name: str, outcome: str | None, *, duration: int | None = None) -> dict:
    by_episode: dict[str, list[float]] = {}
    for r in rows:
        if (duration is None or r["duration"] == duration) and (outcome is None or r["outcome"] == outcome):
            v = r["values"][name]
            if v is not None:
                by_episode.setdefault(r["uuid"], []).append(v)
    means = [sum(v)/len(v) for v in by_episode.values()]
    return dict(episode_macro_mean=sum(means)/len(means) if means else None, episodes=len(means),
                windows=sum(len(v) for v in by_episode.values()))


def _macros(rows, names, **kw) -> dict:
    return {n: {"all": _macro_values(rows, n, None, **kw), **{o: _macro_values(rows, n, o, **kw) for o in OUTCOMES}}
            for n in names}


def _drop_partial(path: Path) -> None:
    """删掉 `cache._write_json` 上次写 `path` 时在改名前中断留下的 `<名>.partial`（RF-P2-2：frame_rate.json、
    resolution.json 在测试集目录里，评测任务根目录的清理碰不到）。只删本函数要写的这一个临时文件；同一评测目录同时只有
    一个进程（`eval_job.run` 持目录锁，RF-P2-3）。"""
    stale = path.with_name(path.name + ".partial")
    if stale.is_file():
        stale.unlink()


def _reuse(path: Path, schema: str, split: str, ids: list[str]) -> dict | None:
    """已写好的 frame_rate.json / resolution.json：schema、测试集和起点（窗口）编号都等于这一次的就复用，否则报错。"""
    if not path.exists():
        return None
    body = ix._loads(path.read_bytes(), str(path))
    if body.get("schema") != schema or body.get("split") != split or [r["id"] for r in body["entries"]] != ids:
        raise ValueError(f"{path} does not hold the current entries of {split}")
    return body


@torch.no_grad()
def frame_rate_group(model, entries: Sequence[Mapping], *, prepared, episodes, extra_reader, control_spec,
                     loss_config: LossConfig, config: DroidConfig, device, ac_steps: AcSteps | None = None) -> list[dict]:
    """一批原始帧率起点的记录（10 §8.4）：起点和目标都从 s5 文件读，每个键的未来误差 d(预测, 目标)。"""
    plan = frame_rate_plan(config)
    source = config.action.source
    kind, stride, per, total = f"s{plan['stride']}", plan["stride"], plan["per"], plan["total"]
    grid16 = regular_grid(GRID, GRID)
    jepa = lambda a, y: jepa_distance(a, y, config=loss_config).per_sample
    names = frame_rate_names(config)
    records, live = [], []
    for e in entries:
        u, f0 = e["uuid"], e["start_frame"]
        poses, states = _poses(prepared, u, source)
        if f0 + total >= poses.shape[0]:
            raise ValueError(f"frame-rate entry {e['id']}: needs frame {f0 + total} of an episode with {poses.shape[0]} steps")
        controls = frame_rate_controls(poses, f0, plan)
        ok = all(_in_bounds(v, control_spec) for calls in controls.values() for v in calls)
        records.append(dict(id=e["id"], uuid=u, start_frame=f0, s5_start_row=f0//stride, outcome=None,
                            excluded=None if ok else "control_out_of_bounds", values={n: None for n in names}))
        if ok:
            live.append((len(records) - 1, e, poses, states, controls))
    if not live:
        return records
    j0 = [e["start_frame"]//stride for _, e, _, _, _ in live]
    raw = torch.cat([_s5_rows(extra_reader, e["uuid"], kind, [j], stride, episodes, prepared)
                     for (_, e, _, _, _), j in zip(live, j0)]).to(device)
    ids = [e["id"] for _, e, _, _, _ in live]

    def target(offset):
        return torch.cat([_s5_rows(extra_reader, e["uuid"], kind, [j + offset], stride, episodes, prepared)
                          for (_, e, _, _, _), j in zip(live, j0)]).to(device)

    def run(name, step_frames, n_calls, offset):
        x = raw
        if ac_steps is None:
            delta = step_frames/per
            for k in range(n_calls):
                values = np.stack([c[name][k] for *_, c in live])
                lengths = [delta] if values.shape[1] == 1 else [1.]*values.shape[1]
                seg = _segments(values, lengths, [f"{i}|{name}|{k}" for i in ids], control_spec, device)
                x = model(x, seg, seg.duration.clone(), grid16)
        else:
            acts = [ac_frame_rate_actions(p, s, e["start_frame"], plan)[name] for _, e, p, s, _ in live]
            for k in range(n_calls):
                a = torch.from_numpy(np.stack([c[k][0] for c in acts]))
                s0 = torch.from_numpy(np.stack([c[k][1] for c in acts]))
                x = ac_steps(model, x, a, s0, grid16)
        return _list(jepa(x, target(offset)))

    values = {}
    for f in plan["single"]:
        values[f"single:f={f}"] = run(f"single:f={f}", f, 1, f//stride)
    for f in plan["rollout"]:
        values[f"rollout:f={f}"] = run(f"rollout:f={f}", f, total//f, total//stride)
    values[f"direct:f={total}"] = run(f"direct:f={total}", per, 1, total//stride)
    for k, (i, *_rest) in enumerate(live):
        records[i]["values"] = {n: values[n][k] for n in names}
    return records


def evaluate_frame_rate(output: Path | str, *, model, split: str, doc: Mapping, prepared: Mapping, outcomes, skipped,
                        episodes, extra_reader, control_spec, loss_config, config, device, ac_steps=None) -> dict:
    """`<output>/frame_rate.json`（每个起点一条，已有且是这一次的起点就复用），返回这个测试集的汇总。"""
    path = Path(output)/FRAME_RATE_FILE
    entries = _frame_rate_entries(doc, skipped, config)
    names = frame_rate_names(config)
    body = _reuse(path, SCHEMA_FRAME_RATE, split, [e["id"] for e in entries])
    if body is None:
        rows = []
        for chunk in bt.chunks(entries) if entries else []:
            rows += frame_rate_group(model, chunk, prepared=prepared, episodes=episodes, extra_reader=extra_reader,
                                     control_spec=control_spec, loss_config=loss_config, config=config, device=device,
                                     ac_steps=ac_steps)
        for r in rows:
            r["outcome"] = outcomes[r["uuid"]]
        body = dict(schema=SCHEMA_FRAME_RATE, split=split, entries=rows)
        Path(output).mkdir(parents=True, exist_ok=True)
        _drop_partial(path)
        cache._write_json(path, body)
    rows = body["entries"]
    return dict(counts=dict(entries=len(rows), episodes=len({r["uuid"] for r in rows}),
                            excluded=sum(r["excluded"] is not None for r in rows)),
                excluded=[r["id"] for r in rows if r["excluded"] is not None],
                macro=_macros(rows, names))


# ---------------------------------------------------------------- 原始分辨率（10 §8.5，S11b）

def resolution_kinds(config: DroidConfig) -> dict[str, int]:
    """{附加特征种类: 网格边长}：r192 → 12、r320 → 20（patch 16）。"""
    out = {}
    for r in config.encode.extra_resolutions:
        if r % 16:
            raise ValueError(f"extra resolution {r} is not a multiple of the patch size 16")
        out[f"r{r}"] = r//16
    return out


def resolution_names(config: DroidConfig) -> tuple[str, ...]:
    return ("future:g=16", *(f"future:{k}" for k in resolution_kinds(config)))


@torch.no_grad()
def resolution_group(model, group: Sequence[Mapping], *, prepared, episodes, reader, extra_reader, control_spec,
                     provenance, loss_config, config, device, bind: Bind | None = None) -> list[dict]:
    """一组同时长的附加回合窗口：主特征 16×16 的未来误差，和 r192/r320 同一行（s、s + d）上原生网格的未来误差。"""
    duration = group[0]["duration"]
    if any(w["duration"] != duration for w in group):
        raise ValueError("a resolution group shares one duration")
    loaded = bt.load_windows(group, reader, episodes)
    jepa = lambda a, y: jepa_distance(a, y, config=loss_config).per_sample
    batch = bt.assemble_training_batch(loaded, control_spec, provenance, device)
    phi = _predictor(model, bind, loaded.start_states, batch.controls)
    values = {"future:g=16": _list(jepa(phi(batch.raw_z, batch.controls, batch.duration, regular_grid(GRID, GRID)),
                                        batch.raw_target))}
    for kind, side in resolution_kinds(config).items():
        pairs = []
        for w in group:
            u, s = w["uuid"], w["start"]
            record = _extra_record(extra_reader, u, kind, episodes, prepared)
            main_index = [int(i) for i in prepared[u].arrays["frame_index"]]
            if record.get("rows") != episodes[u].rows or record.get("frame_index") != main_index:
                raise ValueError(f"episode {u} {kind}: rows are not the main feature frames")
            pairs.append(extra_reader.rows(u, kind, [s, s + duration]))
        pair = torch.stack(pairs)
        if pair.shape[2] != side*side:
            raise ValueError(f"{kind}: {pair.shape[2]} tokens, expected {side}×{side}")
        native = dataclasses.replace(loaded, raw_z=pair[:, 0].contiguous(), raw_target=pair[:, 1].contiguous())
        nb = bt.assemble_training_batch(native, control_spec, provenance, device)
        values[f"future:{kind}"] = _list(jepa(phi(nb.raw_z, nb.controls, nb.duration, regular_grid(side, side)),
                                              nb.raw_target))
    return [dict(id=w["id"], uuid=w["uuid"], duration=duration, start=w["start"], outcome=None,
                 values={n: values[n][k] for n in resolution_names(config)}) for k, w in enumerate(group)]


def evaluate_resolution(output: Path | str, *, model, split: str, doc: Mapping, prepared, outcomes, skipped, episodes,
                        reader, extra_reader, control_spec, provenance, loss_config, config, device, bind=None) -> dict:
    """`<output>/resolution.json`（附加回合里 d ∈ 同分布时长的每个窗口一条），返回这个测试集的汇总。"""
    path = Path(output)/RESOLUTION_FILE
    extra = set(doc.get("extra_episodes", []))
    ids_d = set(config.evaluation.durations_id)
    windows = [w for w in doc["windows"] if w["uuid"] in extra and w["uuid"] not in skipped and w["duration"] in ids_d]
    body = _reuse(path, SCHEMA_RESOLUTION, split, [w["id"] for w in windows])
    if body is None:
        records: list = []
        for chunk in bt.chunks(windows) if windows else []:
            part: list[dict | None] = [None]*len(chunk)
            for d, index in _group(chunk).items():
                for i, r in zip(index, resolution_group(
                        model, [chunk[i] for i in index], prepared=prepared, episodes=episodes, reader=reader,
                        extra_reader=extra_reader, control_spec=control_spec, provenance=provenance,
                        loss_config=loss_config, config=config, device=device, bind=bind)):
                    part[i] = r
            records += part
        for r in records:
            r["outcome"] = outcomes[r["uuid"]]
        body = dict(schema=SCHEMA_RESOLUTION, split=split, entries=records)
        Path(output).mkdir(parents=True, exist_ok=True)
        _drop_partial(path)
        cache._write_json(path, body)
    rows = body["entries"]
    return dict(counts=dict(windows=len(rows), episodes=len({r["uuid"] for r in rows})),
                macro={str(d): _macros(rows, resolution_names(config), duration=d)
                       for d in sorted({r["duration"] for r in rows})})


# ---------------------------------------------------------------- 读检查点（10 §8.1，S09b）

DATA_FIELDS = ("evidence_role", "split", "paper_sha256", "upstream_commit", "encoder_checkpoint_sha256",
               "preprocess_sha256", "feature_manifest_sha256", "protocol_sha256")     # 10 §4.2 第 6 步（不含源码 SHA）
AC_CHECKPOINT_SCHEMA = "v7_vjepa2_ac_checkpoint_v1"


def _read(path: Path) -> dict:
    return ix._loads(Path(path).read_bytes(), str(path))


def _snapshot(condition: str, payload: Mapping, model, loss_config: LossConfig) -> dict:
    """10 §4.2 第 4 步的配置快照核对（DROID 的标签和规格）：PO-JEPA 类、Koopman、Transformer 类照 `addon.v7load`；AC 要求
    schema、条件和规格快照等于现建模型的（DROID 的规格是 7 维官方规模 / 缩小版，不是 v7 的 `spec_for`）。"""
    from po_jepa import ac_baseline
    from po_jepa.runtime.addon import v7load
    if condition in AC_CONDITIONS:
        spec = ac_baseline.spec_snapshot(model.spec)
        if (payload.get("schema") != AC_CHECKPOINT_SCHEMA or payload.get("condition") != condition
                or payload.get("spec") != spec):
            raise ValueError(f"{condition} checkpoint schema/condition/spec differs from the current build")
        return dict(schema=AC_CHECKPOINT_SCHEMA, condition=condition, spec=payload["spec"], info=payload.get("info"))
    try:
        return v7load._payload_family(condition, payload, model, loss_config)
    except AttributeError as e:                       # 传进来的模型不是这个条件的类（没有它的规格属性）
        raise ValueError(f"the model given is not a {condition} model: {e}") from e


def load_selected(train_dir: Path | str, *, condition: str, seed: int, model, loss_config: LossConfig,
                  config: DroidConfig, provenance: dict) -> dict:
    """10 §8.1 的放宽读法（同 §4.2）：读训练任务 `train/<条件>/seed<种子>/`（09 §4.1）`selection.json` 选出的检查点进
    `model`，`model.eval()`，不读优化器、不恢复随机状态。核：COMPLETE.json 绑定 report.json、没有 FAILED.json；报告的
    条件、种子（= 本次请求的 `seed`）、配置 SHA、完成步数 = U；report.json 记的 `registration_sha256` 等于
    registration.json 的文件 SHA，登记的条件、种子也是本次请求的（S09b-P2-1：目录放错时不把别的种子的训练记成这个种子）；`selection.json` 的最优 = 报告的最优 = 按 `validations/` 重算的选点（均值严格更小才换，
    平局留早的）；检查点路径不越出 `checkpoints/`、在 `index.json` 里且是这一步的验证点、文件 SHA 对；`completed_steps`
    等于选中步数；配置快照（`_snapshot`）；`model_state`（`training._check_model_state`）；来源里的数据字段
    （`DATA_FIELDS`）等于本次评测按当前特征算出的 `provenance`。`source_files_sha256` 只记录（评测代码可能在训练之后才修好）。

    返回 {"checkpoint": job.json 记的检查点（相对训练目录的路径、SHA、步数、验证均值），"source": source.json 的内容}。"""
    import io
    from po_jepa.droid import checkpoints as ck
    from po_jepa.droid import train as tr
    from po_jepa.runtime import training
    root = Path(train_dir).resolve()
    if (root/"FAILED.json").exists():
        raise ValueError(f"{root} retains FAILED.json")
    report_path = root/"report.json"
    if not (root/"COMPLETE.json").exists() or _read(root/"COMPLETE.json").get("report_sha256") != cache.file_sha(report_path):
        raise ValueError(f"{root}: the training task is not complete (COMPLETE.json missing or not bound to report.json)")
    report = _read(report_path)
    updates = config.training.updates
    if (report.get("schema") != tr.REPORT_SCHEMA or report.get("condition") != condition
            or report.get("config_sha256") != config.config_sha256 or report.get("completed_steps") != updates):
        raise ValueError(f"{report_path} is not a finished {condition} training under this config")
    if type(report.get("seed")) is not int or report["seed"] != seed:
        raise ValueError(f"{report_path} is the training of seed {report.get('seed')!r}, not of the requested seed {seed}")
    reg_path = root/"registration.json"
    if not reg_path.is_file() or report.get("registration_sha256") != cache.file_sha(reg_path):
        raise ValueError(f"{root}: report.json was not written for this registration.json")
    registration = _read(reg_path)
    if (registration.get("schema") != tr.REGISTRATION_SCHEMA or registration.get("condition") != condition
            or registration.get("seed") != seed):
        raise ValueError(f"{reg_path} registers {registration.get('condition')!r}/seed{registration.get('seed')!r}, "
                         f"not {condition}/seed{seed}")
    if registration.get("config_sha256") != config.config_sha256:
        raise ValueError(f"{reg_path} was registered under another config")
    selection = _read(root/"selection.json")
    records = tr.validation_records(root, updates)
    best = tr.best_of(records)
    if (selection.get("schema") != tr.SELECTION_SCHEMA or selection.get("best") != report.get("best")
            or selection["best"]["step"] != best["step"] or selection["best"]["mean"] != best["mean"]
            or selection.get("validations") != [dict(step=r["step"], mean=r["mean"]) for r in records]):
        raise ValueError(f"{root}: selection.json, report.json and the validation records disagree on the best step")
    chosen = selection["best"]
    path = (root/chosen["checkpoint"]).resolve()
    if not path.is_relative_to(root/"checkpoints"):
        raise ValueError("the selected checkpoint escaped its checkpoints directory")
    entries = ck.read_index(root/"checkpoints")
    entry = next((e for e in entries if e["path"] == path.name), None)
    if (entry is None or entry["kind"] != "validation" or entry["step"] != chosen["step"]
            or entry["sha256"] != chosen["checkpoint_sha256"]):
        raise ValueError(f"{path.name} is not the indexed validation checkpoint of step {chosen['step']}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != chosen["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA mismatch before loading")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if type(payload) is not dict or payload.get("completed_steps") != chosen["step"]:
        raise ValueError("checkpoint completed_steps differs from the selected step")
    stored = (payload.get("provenance") or {}).get(tr.TRAINING_IDENTITY_KEY)
    want = dict(condition=condition, seed=seed, registration_sha256=cache.file_sha(reg_path))
    if stored != want:          # RF-P2-1：检查点自己记的训练身份（登记和报告一起被改写时只有它不变）
        raise ValueError(f"{path.name} was saved by the training {stored!r}, not by {want!r}")
    snapshot = _snapshot(condition, payload, model, loss_config)
    training._check_model_state(payload["model_state"], model)
    found = {k: payload["provenance"].get(k) for k in DATA_FIELDS}
    current = {k: provenance.get(k) for k in DATA_FIELDS}
    if found != current:
        raise ValueError("checkpoint data provenance differs from the current features: "
                         f"{ {k: (found[k], current[k]) for k in DATA_FIELDS if found[k] != current[k]} }")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    checkpoint = dict(path=path.relative_to(root).as_posix(), sha256=digest, step=chosen["step"], mean=chosen["mean"])
    source = dict(train_dir=str(root), report_sha256=cache.file_sha(report_path),
                  selection_sha256=cache.file_sha(root/"selection.json"),
                  registration_sha256=cache.file_sha(reg_path), checkpoint=checkpoint, snapshot=snapshot,
                  data_fields=found, source_files_sha256=payload["provenance"].get("source_files_sha256"),
                  note="10 §8.1: data fields are checked; source_files_sha256 is recorded only")
    return dict(checkpoint=checkpoint, source=source)


# ---------------------------------------------------------------- 一个评测任务

def _sha(value) -> str:
    return hashlib.sha256(cache._json_bytes(value)).hexdigest()


def prepared_arrays_sha256(prepared: Mapping, episodes: Sequence[str]) -> str:
    """窗口里各回合准备数组本身（`PreparedEpisode.arrays` 里每个数组的 `prepare.array_sha256`）的 SHA（S09b 补 S11-fix1 的
    遗留：只核准备 JSON 的 SHA 时，内存里换了数组的 `PreparedEpisode` 也会被当成同一份输入）。"""
    from po_jepa.droid.prepare import array_sha256
    return _sha({u: {name: array_sha256(a) for name, a in sorted(prepared[u].arrays.items())} for u in episodes})


def job_inputs(*, splits: Mapping[str, Mapping], prepared: Mapping, outcomes: Mapping[str, str], provenance: Mapping,
               config: DroidConfig, extra_reader=None, planning: bool = False) -> dict:
    """写进 `job.json` 的输入身份（S11a-fix1，S09b 补两项）：两个测试集冻结窗口内容、窗口里各回合准备文件的 SHA 和准备数组
    本身的 SHA、回合 outcome、批的来源字典（特征清单、编码器、源码 SHA）、测试附加特征清单的 SHA（`extra_reader.manifest_sha256`，
    没有附加特征时为 None）和本次要不要跑原始帧率/分辨率/规划。续跑和复用完成结果都要求它不变。"""
    episodes = sorted({w["uuid"] for split in TEST_SPLITS for w in splits[split]["windows"]})
    missing = [u for u in episodes if u not in prepared]
    if missing:
        raise ValueError(f"no prepared episode for {missing[:3]}")
    extra_sha = None
    if extra_reader is not None:
        extra_sha = getattr(extra_reader, "manifest_sha256", None)
        if type(extra_sha) is not str or len(extra_sha) != 64:
            raise ValueError("the test-extra reader carries no manifest_sha256")
    return dict(windows_sha256={split: _sha(splits[split]) for split in TEST_SPLITS},
                prepared_sha256=_sha({u: prepared[u].json_sha256 for u in episodes}),
                prepared_arrays_sha256=prepared_arrays_sha256(prepared, episodes),
                outcomes_sha256=_sha({u: outcomes.get(u) for u in episodes}),
                provenance_sha256=_sha(dict(provenance)), extra_manifest_sha256=extra_sha,
                action_source=config.action.source,
                axes=dict(frame_rate=extra_reader is not None, resolution=extra_reader is not None, planning=planning))


def evaluate_job(output: Path | str, *, model, condition: str, seed: int | None, job: Mapping, splits: Mapping[str, Mapping],
                 prepared: Mapping, outcomes: Mapping[str, str], reader, control_spec, provenance: dict,
                 loss_config: LossConfig, config: DroidConfig, device, bind: Bind | None = None,
                 extra_reader=None, ac_steps: AcSteps | None = None,
                 planning: Callable[[str], dict] | None = None) -> dict:
    """`eval/<条件>/seed<种子>/`（10 §8.9）：`job.json`（`job` 由调用者给，读检查点的记录由 `load_selected` 给）、
    两个测试集的 rows、`frame_rate.json` 和 `resolution.json`（给了 `extra_reader` 时，S11b）、`report.json`、
    `COMPLETE.json`。`job.json` 记调用者的 `job` 加条件、种子、配置和 `job_inputs` 的输入身份；已完成时这份身份与已存的
    `job.json`、`report.json` 的 `job` 完全相同才直接返回已有的报告，否则报错（S11a-fix1）；三角不等式有违反时写
    `FAILED.json` 并报错。
    模型在 eval 模式、no_grad 下跑，前后核对参数、梯度和 buffer 没变（10 §1）。

    `extra_reader` 是测试附加特征的 `features.ExtraFeatureReader`；不给时报告的 `axes` 记原始帧率和原始分辨率没做。
    `planning(split)` 给了时在同一次 eval 模式里做这个测试集的离线规划（11，`<split>/planning/`），返回规划报告（S09b）。
    AC 的原始帧率要 `ac_steps(model, raw, actions, start_state, grid)`（S08b 的小步前向：`actions` 是物理单位的小步动作
    CPU float64 [B, n, 7]，`start_state` 是这次调用起点的状态 CPU float64 [B, 7]，返回 n 个小步之后的预测）。"""
    out = Path(output)
    if set(splits) != set(TEST_SPLITS):
        raise ValueError(f"an evaluation job covers both test splits {TEST_SPLITS}")
    head = dict(job, condition=condition, seed=seed, config_name=config.name, config_sha256=config.config_sha256,
                inputs=job_inputs(splits=splits, prepared=prepared, outcomes=outcomes, provenance=provenance,
                                  config=config, extra_reader=extra_reader, planning=planning is not None))
    want = ix._loads(cache._json_bytes(head), "job")
    job_path = out/"job.json"
    out.mkdir(parents=True, exist_ok=True)
    if (out/"COMPLETE.json").exists():                 # 完成缓存只在本次请求的身份与已存的完全相同时复用
        report = ix._loads((out/"report.json").read_bytes(), "report.json")
        if ix._loads((out/"COMPLETE.json").read_bytes(), "COMPLETE.json").get("report_sha256") != cache.file_sha(out/"report.json"):
            raise ValueError(f"{out}: COMPLETE.json does not match report.json")
        if (not job_path.exists() or ix._loads(job_path.read_bytes(), "job.json") != want
                or report.get("job") != want):
            raise ValueError(f"{out} is complete for another request (condition, seed, checkpoint, config or inputs "
                             "changed); use a new output directory")
        return report
    if condition in AC_CONDITIONS and config.model.ac_state == "true_state" and bind is None:
        raise ValueError("ac_state: true_state needs the state-binding wrapper (09 §6.2) for AC conditions")
    if condition not in AC_CONDITIONS and bind is not None:
        raise ValueError("only AC conditions take a state-binding wrapper")
    if condition in AC_CONDITIONS and extra_reader is not None and ac_steps is None:
        raise ValueError("the AC frame-rate axis needs the small-step forward ac_steps (10 §8.4)")
    if condition not in AC_CONDITIONS and ac_steps is not None:
        raise ValueError("only AC conditions take ac_steps")
    for stale in out.glob("*.partial"):
        stale.unlink()
    if job_path.exists():
        if ix._loads(job_path.read_bytes(), "job.json") != want:
            raise ValueError(f"{job_path} differs from this command (condition, seed, checkpoint, config or inputs changed)")
    else:
        cache._write_json(job_path, head)
    (out/"FAILED.json").unlink(missing_ok=True)
    modes = {m: m.training for m in model.modules()}
    identity = evaluation._state_identity(model)
    results = {}
    try:
        model.eval()
        for split in TEST_SPLITS:
            results[split] = evaluate_split(out/split, model=model, split=split, doc=splits[split], prepared=prepared,
                                            outcomes=outcomes, reader=reader, control_spec=control_spec,
                                            provenance=provenance, loss_config=loss_config, config=config,
                                            device=device, bind=bind)
            if extra_reader is not None:                       # 10 §8.4、§8.5（S11b）
                _, kept_episodes, skipped = _selected(splits[split], prepared, config.action.source)
                episodes = bt.episode_arrays(prepared, kept_episodes, config.action.source)
                common = dict(model=model, split=split, doc=splits[split], prepared=prepared, outcomes=outcomes,
                              skipped=skipped, episodes=episodes, extra_reader=extra_reader, control_spec=control_spec,
                              loss_config=loss_config, config=config, device=device)
                results[split]["frame_rate"] = evaluate_frame_rate(out/split, ac_steps=ac_steps, **common)
                results[split]["resolution"] = evaluate_resolution(out/split, reader=reader, provenance=provenance,
                                                                   bind=bind, **common)
            if planning is not None:                           # 11：离线规划也在评测任务里做（10 §8.1，S09b 接上）
                results[split]["planning"] = planning(split)
    finally:
        for module, flag in modes.items():
            module.training = flag
    if evaluation._state_identity(model) != identity:
        raise RuntimeError("evaluation changed parameters, gradients or buffers")
    violations = sum(r["triangle"]["violations"] for r in results.values())
    problems = [f"triangle inequality violated in {violations} windows"] if violations else []
    ac = condition in AC_CONDITIONS
    report = dict(schema=SCHEMA_REPORT, job=head, status="failed" if problems else "passed", problems=problems,
                  splits=results, definitions=definitions(config),
                  axes=dict(windows=True, frame_rate=extra_reader is not None, resolution=extra_reader is not None,
                            planning=planning is not None),
                  frame_rate_definitions=frame_rate_definitions(config),
                  resolution_definitions=dict(kinds=resolution_kinds(config), durations=list(config.evaluation.durations_id),
                                              note="same rows s, s + d of the r<R> files on regular_grid(R/16, R/16); "
                                                   "future:g=16 is the 256-input main feature on the same windows"),
                  not_comparable=list(INPUT_SPACE_METRICS) if ac else [],
                  ac_interpolated_tcd=[f"tcd_ln:d={d}" for d in AC_INTERPOLATED_TCD] if ac else [],
                  device=str(device), scientific_conclusions_verified=False)
    (out/"report.json").unlink(missing_ok=True)
    cache._write_json(out/"report.json", report)
    if problems:
        cache._write_json(out/"FAILED.json", dict(stage="report", message="; ".join(problems)))
        raise ValueError("; ".join(problems))
    cache._write_json(out/"COMPLETE.json", dict(report_sha256=cache.file_sha(out/"report.json")))
    return report
