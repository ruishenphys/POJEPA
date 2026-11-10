"""DROID 离线规划（docs/design/11-droid-planning.md，S12）。

DROID 没有模拟器，只做两种离线检验，都在每个评测任务里做（10 §8.1）：给定起点 z 和真实的未来帧 z*_d，
(1) 真实动作在 64 个候选（自己 + 63 个干扰项窗口的动作）里排第几；(2) CEM 从零开始搜出的动作和真实动作的净位移差多少。

- 窗口和候选取自冻结窗口文件的 `planning`（06 §3，S05 生成）：`rank` 给每个窗口的 63 个干扰项，`cem` 给每条附加回合、
  每个 d 排第 1 的窗口。
- 代价 c(a) = d(Φ_d(z; a), z*_d)：v7 `losses.jepa_distance` 的逐样本值（双侧 LN、mean |·|、float32），16×16 网格。
- 候选和搜索都在模型单位（物理值 / 尺度）里做；送进模型前乘回物理值，按 06 §4 组装成每个周期一段的 `ActionSegments`
  （`batches.window_controls`，时间轴用窗口 w 自己的起点 s）。
- 模型接口：`model(raw [B, 256, E] float32, controls, duration CPU float64 [B], grid) -> [B, 256, E]`，和其他评测轴相同。
  AC 的 `true_state` 模式由调用者传 `bind`（09 §6.2 的 `droid.ac.bind(model, start_state, reference)`，S08b）：每批候选
  用窗口 w 的起点状态 `state[s]` 和这批候选自己的控制作 `reference` 绑定一次，状态由包装层按候选动作积分。

输出 `eval/<条件>/seed<种子>/<测试集>/planning/{rank,cem,report}.json`（11 §4）。三份文件先写进 `planning.partial/`，
写完整个目录改名；`planning/` 存在就是完整的，重跑直接读它的 `report.json`。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from po_jepa.actions import ActionSegments, ControlSpec
from po_jepa.droid import batches as bt
from po_jepa.droid import features as ft
from po_jepa.droid import windows as wn
from po_jepa.droid.config import DroidConfig
from po_jepa.geometry import regular_grid
from po_jepa.losses import LossConfig, jepa_distance
from po_jepa.runtime import cache

SCHEMA = "droid_planning_v1"
GRID = 16
CHUNK = 64                       # 每次前向的候选数（排名正好一批；CEM 256 个样本分 4 批）
POSITION = (0, 1, 2)             # 7 维动作里 CEM 搜的通道：位置 3 维 + 夹爪 1 维（11 §3）
ROTATION = (3, 4, 5)             # 旋转 3 维固定为真实执行的值
GRIPPER = 6
SEARCHED = (*POSITION, GRIPPER)
SMALL_DISPLACEMENT_M = 0.01      # 真实净位移小于 1 cm 的窗口不进相对误差的平均（11 §3）
RELATIVE_FLOOR = 1e-9

Bind = Callable[[Callable, torch.Tensor, ActionSegments], Callable]


# ---------------------------------------------------------------- 代价

def costs(model: Callable, raw_z: torch.Tensor, raw_target: torch.Tensor, start: int, actions: np.ndarray,
          control_spec: ControlSpec, ids: Sequence[str], *, loss_config: LossConfig, start_state: torch.Tensor | None = None,
          bind: Bind | None = None, chunk: int = CHUNK) -> np.ndarray:
    """K 组物理动作 `actions` float32 [K, d, 7] 在窗口（起点行 `start`，起点特征 `raw_z` [256, E]、目标 `raw_target` [256, E]，
    都在模型设备上）上的代价，float32 numpy [K]。每 `chunk` 组一次前向；`bind` 给了就用 `start_state`（[7] float64，物理单位）
    绑定这批候选再前向（AC 的 `true_state`，09 §6.2）。"""
    if actions.ndim != 3 or actions.dtype != np.float32 or actions.shape[-1] != control_spec.action_dim:
        raise ValueError("candidate actions must be float32 [K, d, action_dim]")
    k, d = actions.shape[:2]
    if len(ids) != k or d < 1:
        raise ValueError("need one id per candidate and d >= 1")
    if (bind is None) != (start_state is None):
        raise ValueError("bind and start_state go together (AC true_state mode)")
    grid = regular_grid(GRID, GRID)
    out = []
    with torch.no_grad():
        for i in range(0, k, chunk):
            a = torch.from_numpy(actions[i:i + chunk])
            b = a.shape[0]
            controls, _, _ = bt.window_controls(torch.full((b,), start, dtype=torch.int64),
                                                torch.full((b,), d, dtype=torch.int64), a, control_spec,
                                                tuple(ids[i:i + chunk]), raw_z.device)
            duration = torch.full((b,), float(d), dtype=torch.float64)
            raw = raw_z.expand(b, -1, -1).contiguous()
            predictor = model if bind is None else bind(model, start_state.expand(b, -1).clone(), controls)
            pred = predictor(raw, controls, duration, grid)
            if not isinstance(pred, torch.Tensor) or pred.shape != raw.shape:
                raise ValueError(f"model output must be a tensor of shape {tuple(raw.shape)}")
            per = jepa_distance(pred.float(), raw_target.expand(b, -1, -1).contiguous(), config=loss_config).per_sample
            out.append(per.detach().cpu().numpy())
    result = np.concatenate(out)
    if result.dtype != np.float32 or result.shape != (k,):
        raise ValueError("costs must be float32 [K]")
    return result


# ---------------------------------------------------------------- 动作排名（11 §2）

def rank_metrics(cost: np.ndarray, true_index: int = 0) -> dict:
    """真实动作的名次：r_lo = 1 + #{严格更小}，t = #{逐位相等的其他候选}，r_hi = r_lo + t；平局按随机打破取期望。"""
    if cost.dtype != np.float32 or cost.ndim != 1 or not np.isfinite(cost).all():
        raise ValueError("costs must be a finite float32 vector")
    c0 = cost[true_index]
    others = np.delete(cost, true_index)
    r_lo = 1 + int(np.sum(others < c0))
    ties = int(np.sum(others == c0))
    r_hi = r_lo + ties
    return {"r_lo": r_lo, "ties": ties, "mean_rank": (r_lo + r_hi)/2, "top1": 1/(ties + 1) if r_lo == 1 else 0.0,
            "reciprocal_rank": sum(1/r for r in range(r_lo, r_hi + 1))/(ties + 1)}


def random_level(candidates: int) -> dict:
    """随机水平：平均名次 (K + 1)/2、第一名 1/K、倒数名次的期望 H_K/K（K = 64 时 32.5、1/64、≈ 0.0741）。"""
    return {"mean_rank": (candidates + 1)/2, "top1": 1/candidates,
            "reciprocal_rank": sum(1/r for r in range(1, candidates + 1))/candidates}


# ---------------------------------------------------------------- CEM（11 §3）

def cem_seed(salt: str, window_id: str) -> int:
    """seed = int(sha256(f"{window_salt}|cem|{w.id}").hexdigest()[:16], 16)：所有条件在同一窗口上用同一串噪声。"""
    return int(hashlib.sha256(f"{salt}|cem|{window_id}".encode("utf-8")).hexdigest()[:16], 16)


def search_bounds(control_spec: ControlSpec, clip: float) -> tuple[np.ndarray, np.ndarray]:
    """搜索通道（位置 3 + 夹爪 1）在模型单位里的截断区间：[−clip, clip] 与物理界换算到模型单位的区间取交集。
    尺度 ≤ 物理界 / clip 时（正式配置：位置尺度约厘米级、夹爪尺度 ≤ 2/3）交集就是 [−clip, clip]。"""
    scale = np.array([control_spec.scale[c] for c in SEARCHED], dtype=np.float64)
    offset = np.array([control_spec.offset[c] for c in SEARCHED], dtype=np.float64)
    lo = (np.array([control_spec.lower_bounds[c] for c in SEARCHED]) - offset)/scale
    hi = (np.array([control_spec.upper_bounds[c] for c in SEARCHED]) - offset)/scale
    return np.maximum(lo, -clip), np.minimum(hi, clip)


def to_physical(u: np.ndarray, true_actions: np.ndarray, control_spec: ControlSpec) -> np.ndarray:
    """搜索变量 u（模型单位，[S, d, 4]：位置 3 + 夹爪 1）→ 物理动作 float64 [S, d, 7]，旋转 3 维取真实执行的值。"""
    scale = np.asarray(control_spec.scale, dtype=np.float64)
    offset = np.asarray(control_spec.offset, dtype=np.float64)
    out = np.empty((*u.shape[:2], 7), dtype=np.float64)
    out[..., list(SEARCHED)] = u*scale[list(SEARCHED)] + offset[list(SEARCHED)]
    out[..., list(ROTATION)] = true_actions[None, :, list(ROTATION)]
    return out


def cem(cost_fn: Callable[[np.ndarray], np.ndarray], true_actions: np.ndarray, control_spec: ControlSpec, *,
        seed: int, iterations: int, samples: int, elites: int, init_std: float, clip: float) -> dict:
    """11 §3：搜 4d 维（模型单位），μ₀ = 0、σ₀ = init_std；每轮 S 个样本 clip(μ + σ·ε)，ε ~ N(0, 1) 形状 [S, d, 4]，
    `cost_fn(物理动作 float32 [S, d, 7])` 给代价；按 `argsort(kind="stable")` 取前 `elites` 个，μ ← 精英均值，
    σ ← 精英标准差 + 1e-3。返回每轮记录和最后的 μ。"""
    if true_actions.ndim != 2 or true_actions.shape[1] != 7:
        raise ValueError("true actions must be [d, 7]")
    if not (type(elites) is int and type(samples) is int and 1 <= elites <= samples):
        raise ValueError("need 1 <= elites <= samples")
    d = true_actions.shape[0]
    lo, hi = search_bounds(control_spec, clip)
    rng = np.random.default_rng(seed)
    mu, sd = np.zeros((d, len(SEARCHED))), np.full((d, len(SEARCHED)), float(init_std))
    rounds = []
    for _ in range(iterations):
        u = np.clip(mu + sd*rng.standard_normal((samples, d, len(SEARCHED))), lo, hi)
        c = cost_fn(to_physical(u, true_actions, control_spec).astype(np.float32))
        if c.dtype != np.float32 or c.shape != (samples,) or not np.isfinite(c).all():
            raise ValueError("CEM costs must be a finite float32 vector, one per sample")
        order = np.argsort(c, kind="stable")[:elites]
        elite = u[order]
        mu, sd = elite.mean(0), elite.std(0) + 1e-3
        rounds.append({"mu": mu.tolist(), "sigma": sd.tolist(), "elite_costs": c[order].tolist()})
    return {"rounds": rounds, "mu": mu}


def cem_errors(mu: np.ndarray, true_actions: np.ndarray, control_spec: ControlSpec) -> dict:
    """11 §3 的四个量（物理单位）：净位移误差 ‖Σp̂ − Σp‖₂（米）、夹爪误差 |Σĝ − Σg|、不动时的两个量、相对误差。"""
    plan = to_physical(mu[None], true_actions, control_spec)[0]
    p_hat, p = plan[:, list(POSITION)].sum(0), true_actions[:, list(POSITION)].sum(0)
    g_hat, g = plan[:, GRIPPER].sum(), true_actions[:, GRIPPER].sum()
    position, gripper = float(np.linalg.norm(p_hat - p)), float(abs(g_hat - g))
    position_zero, gripper_zero = float(np.linalg.norm(p)), float(abs(g))
    return {"position_error_m": position, "gripper_error": gripper,
            "position_error_zero_m": position_zero, "gripper_error_zero": gripper_zero,
            "position_relative": position/max(position_zero, RELATIVE_FLOOR),
            "gripper_relative": gripper/max(gripper_zero, RELATIVE_FLOOR),
            "small_displacement": position_zero < SMALL_DISPLACEMENT_M}


# ---------------------------------------------------------------- 一个测试集

@dataclass(frozen=True)
class PlanningInputs:
    """一个测试集上做规划要的东西。`episodes` 是 `batches.episode_arrays(prepared, …, source)` 给出的小表，只含本配置
    动作源下通过检查的回合（`commanded` 配置里没通过的回合不在表里，10 §8.1）。"""

    windows: Mapping                 # 冻结窗口文件 prepared/windows/<split>.json 的内容
    reader: ft.FeatureReader
    episodes: Mapping[str, bt.EpisodeArrays]
    control_spec: ControlSpec
    config: DroidConfig
    loss_config: LossConfig


def _by_id(windows: Mapping) -> dict[str, dict]:
    table = {}
    for w in windows["windows"]:
        if w["id"] != wn.window_id(w["uuid"], w["duration"], w["start"]):
            raise ValueError(f"window id {w['id']!r} does not match its uuid, duration and start")
        table[w["id"]] = w
    return table


def _actions(w: Mapping, episodes: Mapping[str, bt.EpisodeArrays]) -> np.ndarray:
    ep = episodes[w["uuid"]]
    s, d = w["start"], w["duration"]
    if s < 0 or s + d > ep.rows - 1:
        raise ValueError(f"window {w['id']}: actions {s}…{s + d - 1} outside an episode with {ep.rows} rows")
    return ep.actions[s:s + d]


def _check_header(inputs: PlanningInputs) -> None:
    w, e = inputs.windows, inputs.config.evaluation
    if w.get("schema") != wn.SCHEMA or w.get("split") not in ("id_test", "ood_test"):
        raise ValueError("planning needs an id_test or ood_test frozen-window file")
    if w.get("salt") != inputs.config.data.window_salt:
        raise ValueError("the window file was made with another window_salt")
    if inputs.control_spec.action_dim != 7:
        raise ValueError("DROID planning needs the 7-channel control spec")
    for wid, others in w["planning"]["rank"].items():
        if len(others) != e.planning_candidates - 1 or len(set(others)) != len(others) or wid in others:
            raise ValueError(f"window {wid} must have {e.planning_candidates - 1} distinct distractors")


def _load(w: Mapping, inputs: PlanningInputs, device):
    loaded = bt.load_windows([w], inputs.reader, inputs.episodes)
    return loaded.raw_z[0].to(device), loaded.raw_target[0].to(device), loaded.start_states[0]


def rank_split(model: Callable, inputs: PlanningInputs, device, *, bind: Bind | None = None) -> tuple[list, dict]:
    """11 §2：每个排名窗口 64 个候选（真实动作在第 0 位，其后是 63 个干扰项窗口的动作，按文件顺序），返回每个窗口的记录和
    跳过数。窗口或任一干扰项的回合不在 `episodes` 里（commanded 配置里没通过）时跳过这个窗口，候选数始终是 64。"""
    _check_header(inputs)
    table = _by_id(inputs.windows)
    records, skipped = [], 0
    for wid, others in inputs.windows["planning"]["rank"].items():
        w = table[wid]
        if w["duration"] not in inputs.config.evaluation.planning_durations:
            raise ValueError(f"rank window {wid} has a duration outside planning_durations")
        cands = [w, *(table[o] for o in others)]
        if any(c["duration"] != w["duration"] or (c is not w and c["uuid"] == w["uuid"]) for c in cands):
            raise ValueError(f"distractors of {wid} must share its duration and come from other episodes")
        if any(c["uuid"] not in inputs.episodes for c in cands):
            skipped += 1
            continue
        z, target, state = _load(w, inputs, device)
        actions = np.stack([_actions(c, inputs.episodes) for c in cands]).astype(np.float32)
        cost = costs(model, z, target, w["start"], actions, inputs.control_spec,
                     [f"{wid}|cand={c['id']}" for c in cands], loss_config=inputs.loss_config,
                     start_state=state if bind else None, bind=bind)
        records.append({"id": wid, "uuid": w["uuid"], "duration": w["duration"], "candidates": [c["id"] for c in cands],
                        "costs": cost.tolist(), **rank_metrics(cost)})
    return records, {"rank_windows_skipped": skipped}


def cem_split(model: Callable, inputs: PlanningInputs, device, *, bind: Bind | None = None) -> tuple[list, dict]:
    """11 §3：`planning.cem` 的每个窗口一次 CEM（共同随机数），返回每个窗口的记录和跳过数。"""
    _check_header(inputs)
    table, c = _by_id(inputs.windows), inputs.config.evaluation.cem
    salt = inputs.config.data.window_salt
    records, skipped = [], 0
    for wid in inputs.windows["planning"]["cem"]:
        w = table[wid]
        if w["duration"] not in inputs.config.evaluation.planning_durations:
            raise ValueError(f"CEM window {wid} has a duration outside planning_durations")
        if w["uuid"] not in inputs.episodes:
            skipped += 1
            continue
        z, target, state = _load(w, inputs, device)
        true_actions = _actions(w, inputs.episodes)
        ids = [f"{wid}|cem|{i}" for i in range(c.samples)]
        seed = cem_seed(salt, wid)
        result = cem(lambda a: costs(model, z, target, w["start"], a, inputs.control_spec, ids,
                                     loss_config=inputs.loss_config, start_state=state if bind else None, bind=bind),
                     true_actions, inputs.control_spec, seed=seed, iterations=c.iterations, samples=c.samples,
                     elites=c.elites, init_std=c.init_std, clip=c.clip)
        final = to_physical(result["mu"][None], true_actions, inputs.control_spec)[0]
        records.append({"id": wid, "uuid": w["uuid"], "duration": w["duration"], "seed": seed,
                        "rounds": result["rounds"], "final_model_units": result["mu"].tolist(),
                        "final_physical": final.tolist(), "true_physical": true_actions.tolist(),
                        **cem_errors(result["mu"], true_actions, inputs.control_spec)})
    return records, {"cem_windows_skipped": skipped}


# ---------------------------------------------------------------- 汇总（11 §2、§3 的报告）

def _macro(records: Sequence[Mapping], key: str) -> tuple[float | None, float | None, int]:
    """回合宏平均：先在回合内平均，再在回合间平均；另给回合均值的中位数和回合数。"""
    per = {}
    for r in records:
        per.setdefault(r["uuid"], []).append(r[key])
    if not per:
        return None, None, 0
    means = [math.fsum(v)/len(v) for v in per.values()]
    return math.fsum(means)/len(means), float(np.median(means)), len(means)


def report(rank: Sequence[Mapping], cem_records: Sequence[Mapping], *, split: str, config: DroidConfig,
           skipped: Mapping) -> dict:
    """每个 d 上：排名三项的回合宏平均和随机水平；CEM 净位移误差、夹爪误差、相对误差（不含真实净位移 < 1 cm 的窗口）
    的回合宏平均和中位数，以及 < 1 cm 的窗口数。"""
    e = config.evaluation
    out = {"schema": SCHEMA, "split": split, "candidates": e.planning_candidates,
           "random_level": random_level(e.planning_candidates), "cem": dataclasses.asdict(e.cem), **skipped, "by_duration": {}}
    for d in e.planning_durations:
        rk = [r for r in rank if r["duration"] == d]
        cm = [r for r in cem_records if r["duration"] == d]
        big = [r for r in cm if not r["small_displacement"]]
        row = {"rank_windows": len(rk), "cem_windows": len(cm), "cem_small_displacement_windows": len(cm) - len(big)}
        for key in ("mean_rank", "top1", "reciprocal_rank"):
            mean, _, n = _macro(rk, key)
            row[key] = mean
        row["rank_episodes"] = _macro(rk, "mean_rank")[2]
        for key, rows in (("position_error_m", cm), ("gripper_error", cm), ("position_relative", big)):
            mean, median, n = _macro(rows, key)
            row[key] = {"mean": mean, "median": median, "episodes": n}
        out["by_duration"][str(d)] = row
    return out


def run(model: Callable, inputs: PlanningInputs, out_dir: Path | str, device, *, bind: Bind | None = None) -> dict:
    """一个测试集的离线规划：写 `out_dir`（= `<测试集>/planning/`）下的 rank.json、cem.json、report.json，返回报告。
    `out_dir` 已存在就是完整的（整个目录最后才改名），直接返回它的报告；残留的 `planning.partial/` 先删掉重做。
    调用者负责 eval 模式和评测前后参数不变的核对（10 §8.1）。"""
    out_dir = Path(out_dir)
    if out_dir.exists():
        return json.loads((out_dir/"report.json").read_text(encoding="utf-8"))
    partial = out_dir.with_name(out_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    split = inputs.windows["split"]
    rank, rank_skip = rank_split(model, inputs, device, bind=bind)
    cem_records, cem_skip = cem_split(model, inputs, device, bind=bind)
    head = {"schema": SCHEMA, "split": split, "salt": inputs.windows["salt"],
            "windows_canonical_sha256": hashlib.sha256(cache._json_bytes(dict(inputs.windows))).hexdigest(),
            "control_scale": list(inputs.control_spec.scale)}
    cache._write_json(partial/"rank.json", head | {"windows": rank})
    cache._write_json(partial/"cem.json", head | {"windows": cem_records})
    summary = report(rank, cem_records, split=split, config=inputs.config, skipped=rank_skip | cem_skip)
    cache._write_json(partial/"report.json", summary)
    partial.rename(out_dir)
    return summary
