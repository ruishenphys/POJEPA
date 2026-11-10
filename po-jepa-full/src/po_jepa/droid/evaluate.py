"""DROID 评测轴（docs/design/10-evaluation.md §8，S11a）。

本段（S11a）做冻结测试窗口上的各轴（10 §8.2 的表、§8.3 动作干预、§8.6 AC 的处理、§8.7 成败分开、§8.9 的
rows/续跑/report）和 `commanded` 配置的跳过（§8.1 最后一条）。原始帧率（§8.4）、原始分辨率（§8.5）、截断间隙（§8.8）
和读检查点（§8.1 前三条，依赖 S09 的检查点格式）在 S11b。

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

import math
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from po_jepa import ln_metrics, metrics
from po_jepa.actions import ActionSegments, concatenate_action_segments, split_action_segments
from po_jepa.droid import batches as bt
from po_jepa.droid import index as ix
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
    `donor_batch = (批, 有效掩码)`，掩码为假的行用的是占位动作，结果丢弃。返回 ({键: [B 个值]}, 三角不等式逐行结果)。
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
        if isinstance(model, POJEPA) and bind is None:                                 # 10 §8.3
            delta = metrics.action_intervention_margin(model, raw, controls, donor.controls, dur, grid16, target,
                                                       loss_config=loss_config).values
        else:
            metrics._action_pair(raw, controls, donor.controls, dur)
            phi_donor = _predictor(model, bind, start_states, donor.controls)
            delta = jepa(phi_donor(raw, donor.controls, dur, grid16), target) - out["future:g=16"]
        out["delta_act"] = [v if ok else None for v, ok in zip(_list(delta), valid)]
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
            if not all(valid):
                # 占位行借本窗口的动作时 source id 相同，_action_pair 会拒；换成同组里编号不同的另一个可用窗口（结果仍丢弃）
                pool = group + [d for d, ok in zip(donors, valid) if ok]
                for k, (w, ok) in enumerate(zip(group, valid)):
                    if not ok:
                        spare = next((x for x in pool if x["id"] != w["id"]), None)
                        if spare is None:
                            raise ValueError(f"no placeholder donor for {w['id']}")
                        donors[k] = spare
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


# ---------------------------------------------------------------- 一个评测任务

def evaluate_job(output: Path | str, *, model, condition: str, seed: int | None, job: Mapping, splits: Mapping[str, Mapping],
                 prepared: Mapping, outcomes: Mapping[str, str], reader, control_spec, provenance: dict,
                 loss_config: LossConfig, config: DroidConfig, device, bind: Bind | None = None) -> dict:
    """`eval/<条件>/seed<种子>/`（10 §8.9）的窗口轴部分：`job.json`（`job` 由调用者给，读检查点的记录在 S11b）、两个测试集
    的 rows、`report.json`、`COMPLETE.json`。已完成就直接返回已有的报告；三角不等式有违反时写 `FAILED.json` 并报错。
    模型在 eval 模式、no_grad 下跑，前后核对参数、梯度和 buffer 没变（10 §1）。"""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"COMPLETE.json").exists():
        report = ix._loads((out/"report.json").read_bytes(), "report.json")
        if ix._loads((out/"COMPLETE.json").read_bytes(), "COMPLETE.json").get("report_sha256") != cache.file_sha(out/"report.json"):
            raise ValueError(f"{out}: COMPLETE.json does not match report.json")
        return report
    if condition in AC_CONDITIONS and config.model.ac_state == "true_state" and bind is None:
        raise ValueError("ac_state: true_state needs the state-binding wrapper (09 §6.2) for AC conditions")
    if condition not in AC_CONDITIONS and bind is not None:
        raise ValueError("only AC conditions take a state-binding wrapper")
    if set(splits) != set(TEST_SPLITS):
        raise ValueError(f"an evaluation job covers both test splits {TEST_SPLITS}")
    for stale in out.glob("*.partial"):
        stale.unlink()
    head = dict(job, condition=condition, seed=seed, config_name=config.name, config_sha256=config.config_sha256)
    job_path = out/"job.json"
    if job_path.exists():
        if ix._loads(job_path.read_bytes(), "job.json") != ix._loads(cache._json_bytes(head), "job"):
            raise ValueError(f"{job_path} differs from this command (condition, seed, checkpoint or config changed)")
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
