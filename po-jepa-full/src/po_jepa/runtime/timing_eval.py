"""S25d：时间采样测试集上的按帧评测（帧率 OOD、帧时间抖动、曝光、控制在两帧之间切换；S24b 的四个调节项）。

模型在基础场景（4 fps：每 tick 一帧，训练时长 1、2、4、8 个 tick）上训练好，这里只评测。测试集是
`<场景>~<时间采样调节项>:<条数>` 的测试段（编号与基础场景测试段一一对应，同初始状态、同下达动作；switch_offset 只把
下达动作的段边界整体往后挪），用 tools/encode_v7.py --splits test 编码（runtime/timing_cache.py 读）。

窗口按帧计（帧率 OOD 的问法：在测试视频里往后预测 n 帧）。记号：第 k 帧的时刻 t_k（tick，timing.frame_times），
特征 z_k；起点第 i 帧，目标第 i+n 帧，时长 T = t_{i+n} − t_i；u 是下达动作裁到 [t_i, t_{i+n})；d 是 JEPA 距离
（comparison.jepa_distance，v7 的 future_jepa_distance 同一个）。

- 起点：主实验的测试起点 tick s（datasets.test_start_ticks；2000 条的数据集每条 2 个）所在的帧 i = ⌊s·fps/4⌋（fps ≥ 4 时 t_i 的名义时刻就是 s；
  fps = 2 时奇数 tick 往前取到偶数 tick；带抖动时 t_i 是第 i 帧抖动后的时刻）。
- 帧数 n ∈ FRAME_STEPS = 1、2、3、4、8、12（同主实验的 6 个时长，按帧数计；3、12 在 4 fps 时是留出时长），只留 i + n ≤ L − 1 的。

| 键 | 公式 |
|---|---|
| direct:n=k | d(Φ_T(z_i), z_{i+n})，Φ_T 一次完整预测（d4.direct_prediction 的写法：u 在 T/2 处切开再接上，同 v7 point_metrics） |
| composed:n=k | d(Φ_{Δ_{n−1}} ∘ … ∘ Φ_{Δ_0}(z_i), z_{i+n})，Δ_m = t_{i+m+1} − t_{i+m}：每帧一次完整调用，动作是那一帧间隔的，输入是上一次的原始预测（不 LN、不用真实中间帧，同 D4 rollout） |
| pd_ln:n=k、pd_in:n=k | 两种度量下 Φ_T 与逐帧组合之差（ln_metrics，同 D4）：测试帧率下的时间离散一致性 |
| frame_step:n=k | 只有 V-JEPA 2-AC：一帧算一步（不知道帧间隔），动作取每个帧间隔的时间平均，n 步自回归后的 d(·, z_{i+n}) |

各模型怎么吃时长和动作（S25d 约定）：PO-JEPA、Transformer 的接口本来就是连续时间的分段常值动作，直接收非整数时长和
tick 内部的切换时刻。V-JEPA 2-AC 是按 tick 的离散步模型（ac_baseline.ACPredictor.forward：动作取每个 tick 的时间平均
tick_actions，只收整数和半整数时长，半整数取相邻两步的平均）；这里推广成“相邻两步线性插值”：
Φ_T = (1 − w)·Φ_k + w·Φ_{k+1}，k = ⌊T⌋，w = T − k，Φ_0 = LN(z)（整数、半整数时长直接调 forward，逐位相同）。
frame_step 是另一种用法，看 AC 不知道帧间隔时差多少；fps = 4 且无抖动时它和 direct 是同一个预测（动作平均的舍入除外）。
"""

import math
from pathlib import Path

import torch
from torch.nn import functional as F

from po_jepa import protocols
from . import data, timing_cache
from .addon import d4

TESTS = protocols.TIMING_TESTS
FRAME_STEPS = (1, 2, 3, 4, 8, 12)
KEYS = ("direct", "composed", "pd_ln", "pd_in")
AC_KEYS = ("frame_step",)
TIMING_KNOBS = ("fps", "jitter", "exposure", "switch_offset")


def is_ac(model) -> bool:
    from po_jepa.ac_baseline import ACPredictor
    return isinstance(model, ACPredictor)


def _spec():
    spec = data.datasets.active()
    timing_cache._timing_spec()
    return spec


def start_frames(trajectory_id) -> tuple:
    """主实验测试起点 tick 所在的帧（⌊s·fps/4⌋）。"""
    fps = _spec().knobs.fps
    return tuple(sorted({s*fps//4 for s in data.datasets.test_start_ticks(trajectory_id)}))


def frame_steps(start_frame) -> tuple:
    last = timing_cache.frame_count()-1
    return tuple(n for n in FRAME_STEPS if start_frame+n <= last)


def value_names(ac=False) -> tuple:
    return tuple(f"{k}:n={n}" for n in FRAME_STEPS for k in KEYS+(AC_KEYS if ac else ()))


# ———————————————————— V-JEPA 2-AC 的两种用法 ————————————————————

def _ac_steps(model, raw, actions, grid):
    """ACPredictor.forward 的 rollout 那几行：LN 起点、官方规模 GPU 上 bf16。→ (z0, preds [B,K,N,E])。"""
    from po_jepa.ac_baseline import OFFICIAL
    z0 = F.layer_norm(raw, (model.encoder_dim,))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=raw.device.type == "cuda" and model.spec == OFFICIAL):
        preds = model.rollout(z0, actions.to(raw), grid.height, grid.width).float()
    return z0, preds


def ac_interpolated(model, raw, controls, duration, grid):
    """AC 的连续时长：整数、半整数直接 forward；其余在相邻两步之间线性插值（forward 半整数规则的推广）。"""
    from po_jepa.ac_baseline import tick_actions
    from po_jepa.actions import clip_action_segments, validate_action_segments
    d = duration.detach().cpu().double()
    if torch.all(d > 0) and torch.all((d*2) == (d*2).round()):
        return model(raw, controls, duration, grid)
    if raw.ndim != 3 or raw.shape[1] != grid.height*grid.width or raw.shape[2] != model.encoder_dim:
        raise ValueError("raw must be [B, grid tokens, encoder_dim]")
    validate_action_segments(controls, query_duration=duration)
    if controls.u_values.shape[0] != raw.shape[0] or duration.shape != (raw.shape[0],) or torch.any(d <= 0):
        raise ValueError("controls/duration batch must match raw; durations must be positive")
    steps = int(math.ceil(float(d.max())))
    # 先把动作裁到 [0, T)：最后一个不满的 tick 只平均 [k, T) 这一截（评测窗口的动作本来就正好覆盖 T，裁了不变）
    clipped = clip_action_segments(controls, start=torch.zeros_like(d), end=d)
    z0, preds = _ac_steps(model, raw, tick_actions(clipped, steps), grid)
    rows = []
    for i, value in enumerate(d.tolist()):
        k = int(math.floor(value)); w = value-k
        if w == 0:
            rows.append(preds[i, k-1])
        else:
            lower = z0[i] if k == 0 else preds[i, k-1]
            rows.append((1-w)*lower+w*preds[i, k])
    return torch.stack(rows)


def interval_actions(pieces) -> torch.Tensor:
    """每个帧间隔内分段常值动作的时间平均 → [B, n, 2]（每段按长度加权）。pieces：n 个 ActionSegments（局部时间从 0 起）。"""
    out = []
    for c in pieces:
        length = (torch.minimum(c.ends, c.duration[:, None])-c.starts).clamp(min=0)*c.segment_valid
        u = torch.where(c.segment_valid[..., None].to(c.u_values.device), c.u_values, torch.zeros_like(c.u_values))
        out.append(torch.einsum("bj,bjm->bm", (length/c.duration[:, None]).to(u), u))
    return torch.stack(out, 1)


def ac_frame_step(model, raw, pieces, grid):
    """一帧算一步：n = len(pieces) 步自回归，第 m 步的动作是第 m 个帧间隔的平均动作。→ 第 n 步的预测 [B,N,E]（已 LN）。"""
    if raw.ndim != 3 or raw.shape[1] != grid.height*grid.width or raw.shape[2] != model.encoder_dim:
        raise ValueError("raw must be [B, grid tokens, encoder_dim]")
    _, preds = _ac_steps(model, raw, interval_actions(pieces), grid)
    return preds[:, -1]


def predictor(model):
    """评测时调用的预测函数：AC 换成 ac_interpolated，其余就是 model 本身。"""
    if is_ac(model):
        return lambda raw, controls, duration, grid: ac_interpolated(model, raw, controls, duration, grid)
    return model


# ———————————————————— 窗口和评测 ————————————————————

def load_window(features, trajectory_id, start_frame, frames, device):
    """一个测试窗口：核根目录登记的 manifest、真实编码器、起点和目标的物理时刻。"""
    spec = _spec()
    if type(trajectory_id) is not int or trajectory_id not in spec.split_range("test"):
        raise ValueError("time-sampling window requires a test trajectory")
    approved = data.approved_cache_metadata(features, splits=("test",))
    window = timing_cache.read_window(Path(features)/f"{trajectory_id:04d}", start_frame=start_frame, frames=frames,
                                      expected_split="test", device=str(device))
    x, y, p = window.inputs, window.target, window.provenance
    data._window_boundary(window, window)
    if (x.trajectory_id != trajectory_id or x.split != "test"
            or p["actual_manifest_sha256"] != approved["manifest_sha256_by_id"][trajectory_id]
            or p["generation"]["evidence_role"] != "real_frozen_encoder" or p["observation_variant"] != "canonical"
            or float(x.observed_time_seconds[0]) != p["start_tick"]*.25
            or abs(float(y.target_time_seconds[0])-p["end_tick"]*.25) > 1e-12*max(1., p["end_tick"])
            or x.controls.source_ids != (f"trajectory-{trajectory_id:04d}",)):
        raise ValueError("time-sampling test identity/source/origin/endpoint mismatch")
    return window


@torch.no_grad()
def start_values(model, windows, singles, config):
    """一条测试轨迹、一个起点上的全部量。windows：{n: 窗口 (i, n)}，n 取遍 frame_steps(i)；singles：[窗口 (i+m, 1)]，
    m = 0…max n − 1（逐帧组合每一步的时长和动作）。→ {n: {键: 值}}。"""
    call = predictor(model)
    ac = is_ac(model)
    steps = sorted(windows)
    raw, grid = windows[steps[0]].inputs.raw_z, windows[steps[0]].inputs.grid
    chain, x = {}, raw
    for m, w in enumerate(singles):
        x = call(x, w.inputs.controls, w.inputs.duration, grid)             # 第 m 帧间隔：输入是上一次的原始预测
        chain[m+1] = x
    out = {}
    for n in steps:
        w = windows[n]
        direct, _, _ = d4.direct_prediction(call, w)
        target = w.target.raw_z
        values = {f"direct:n={n}": d4._jepa(direct, target, config), f"composed:n={n}": d4._jepa(chain[n], target, config),
                  f"pd_ln:n={n}": d4._ln(direct, chain[n]), f"pd_in:n={n}": d4._in(direct, chain[n], raw)}
        if ac:
            values[f"frame_step:n={n}"] = d4._jepa(ac_frame_step(model, raw, [s.inputs.controls for s in singles[:n]], grid),
                                                   target, config)
        out[n] = d4._finite(values)
    return out


def evaluate(model, features, config, device):
    """全部测试轨迹：返回 (rows, means)。rows 每个 (轨迹, 起点帧, 帧数) 一行；means 按键名跨窗口平均（键名含帧数）。
    模型的参数、梯度、缓冲区不许被改。"""
    from . import evaluation
    spec = _spec()
    modes = {m: m.training for m in model.modules()}
    identity = evaluation._state_identity(model)
    rows = []
    try:
        model.eval()
        for tid in spec.split_range("test"):
            for i in start_frames(tid):
                steps = frame_steps(i)
                if not steps:
                    continue
                windows = {n: load_window(features, tid, i, n, device) for n in steps}
                singles = [load_window(features, tid, i+m, 1, device) for m in range(max(steps))]
                for n, values in start_values(model, windows, singles, config).items():
                    p = windows[n].provenance
                    rows.append(dict(trajectory_id=tid, start_frame=i, frames=n, start_tick=p["start_tick"],
                                     duration_ticks=float(windows[n].inputs.duration[0]), values=values,
                                     manifest_sha256=p["actual_manifest_sha256"]))
    finally:
        for module, flag in modes.items():
            module.training = flag
    if evaluation._state_identity(model) != identity:
        raise RuntimeError("time-sampling evaluation changed parameters, gradients or buffers")
    means = {}
    for name in value_names(is_ac(model)):
        xs = [r["values"][name] for r in rows if name in r["values"]]
        if xs:
            means[name] = sum(xs)/len(xs)
    for n in FRAME_STEPS:
        ds = [r["duration_ticks"] for r in rows if r["frames"] == n]
        if ds:
            means[f"duration_ticks:n={n}"] = sum(ds)/len(ds)
            means[f"windows:n={n}"] = len(ds)
    return rows, means


def definitions() -> dict:
    v = data.datasets.active().knobs
    return dict(fps=v.fps, jitter=v.jitter, exposure=v.exposure, switch_offset=v.switch_offset,
                frame_steps=list(FRAME_STEPS), start_frame="floor(s*fps/4) for s in datasets.test_start_ticks",
                direct="model(z_i, u1⊕u2, T) with u split at T/2 (d4.direct_prediction), T = t_{i+n} - t_i",
                composed="one call per frame interval, each input is the previous raw prediction",
                ac="vjepa2_ac*: per-tick average actions (tick_actions); non-half-integer T linearly interpolated "
                   "between adjacent tick steps; frame_step = one AC step per frame with interval-average actions",
                distance="comparison.jepa_distance per_sample; pd_ln/pd_in as in D4 (ln_metrics)")
