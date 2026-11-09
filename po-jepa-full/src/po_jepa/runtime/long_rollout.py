"""S25c：长时程测试集上的滚动评测（论文评测表的“训练不超过 8 步，测试 16–64 步”）。

长时程测试集是 `<场景>~horizon=H:<条数>` 的测试段（S24a；H = 128 时每类 200 条，每条 129 帧），用 tools/encode_v7.py
--splits test 编码成 H+1 帧的缓存。前 32 个 tick 的动作、画面和特征与主实验测试段逐位相同，所以这就是主实验测试段的延长。
模型是在基础场景（horizon = 32）上训练好的，这里只评测。

记号同 D4（addon/d4.py，10 §3.7）：z 是 tick 0 的特征，u 是从 tick 0 起的下达动作，z*_T 是第 T 个 tick 的特征，
d 是 JEPA 距离（comparison.jepa_distance）。每条测试轨迹（起点固定为 tick 0）：

| 键 | 公式 |
|---|---|
| long_direct:T=t | d(Φ_t(z), z*_t)，Φ_t 是一次完整预测（d4.direct_prediction，和 v7 point_metrics 的 direct 相同） |
| long_composed:T=t | d(Φ_{8,…,8}(z), z*_t)，每 8 个 tick 一次完整调用、下一段的输入是上一段的原始预测（d4.rollout） |
| long_pd_ln:T=t、long_pd_in:T=t | 两种度量下 Φ_t 与 Φ_{8,…,8} 之差（ln_metrics，同 D4） |
| rollout:T=t | 同一条 8 步滚动在第 t 个 tick 的 d(·, z*_t)，t = 8, 16, …, H（误差曲线，给 S25f 的有效预测时长用） |

t 取 DIRECT_TIMES = 16、24、32、48、64（不超过 H）。8 步一段是训练用过的最长时长；滚动曲线在 t ∈ DIRECT_TIMES
上和 long_composed 是同一个预测（同一次滚动的前缀，测试核逐位相同）。只做起点 0：起点 > 0 的长窗口和主实验
测试窗口一样，信息不多，成本却成倍。
"""

from pathlib import Path

import torch

from . import data
from .addon import d4, long_window

DIRECT_TIMES = (16., 24., 32., 48., 64.)
BLOCK = 8.


def _horizon(horizon):
    from po_jepa import variants
    if type(horizon) is not int or horizon not in variants.HORIZONS[1:]:
        raise ValueError(f"long-horizon test set needs horizon in {variants.HORIZONS[1:]}")
    return horizon


def direct_times(horizon) -> tuple:
    return tuple(t for t in DIRECT_TIMES if t <= _horizon(horizon))


def curve_times(horizon) -> tuple:
    return tuple(float(t) for t in range(int(BLOCK), _horizon(horizon)+1, int(BLOCK)))


def window_times(horizon) -> tuple:
    """要读的窗口时长：直接预测的时长和曲线的端点（都是 8 的倍数）。"""
    return tuple(sorted(set(direct_times(horizon)) | set(curve_times(horizon))))


def value_names(horizon) -> tuple:
    names = [f"long_{k}:T={d4._t(t)}" for t in direct_times(horizon) for k in ("direct", "composed", "pd_ln", "pd_in")]
    return tuple(names+[f"rollout:T={d4._t(t)}" for t in curve_times(horizon)])


def load_window(features, trajectory_id, duration, device, horizon):
    """d4.load_long_test_window 的长时程测试集版：当前数据集必须是 horizon = H 的长时程测试集，时长是 window_times(H) 之一。"""
    spec = data.datasets.active()
    if spec.knobs.horizon != _horizon(horizon):
        raise ValueError(f"active dataset {spec.tag} is not the horizon={horizon} long-horizon test set")
    if type(trajectory_id) is not int or trajectory_id not in spec.split_range("test"):
        raise ValueError("long-horizon window requires a test trajectory")
    approved = data.approved_cache_metadata(features, splits=("test",))
    window = long_window.read_long_window(Path(features)/f"{trajectory_id:04d}", start_tick=0, duration_cycles=duration,
                                          expected_split="test", device=str(device), allowed=window_times(horizon))
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


@torch.no_grad()
def trajectory_values(model, windows, config, horizon):
    """一条测试轨迹上的全部量。windows：{t: CacheWindow}，t 取遍 window_times(H)，同一条轨迹、起点 0。"""
    full = windows[float(horizon)]
    raw, grid = full.inputs.raw_z, full.inputs.grid
    values, composed = {}, {}
    x = raw
    for piece, t in zip(d4.pieces(full.inputs.controls, (BLOCK,)*int(horizon/BLOCK)), curve_times(horizon)):
        x = model(x, piece, d4._cycles(BLOCK), grid)                      # 8 步一段，输入是上一段的原始预测
        composed[t] = x
    for t in direct_times(horizon):
        w = windows[t]
        d, _, _ = d4.direct_prediction(model, w)
        target = w.target.raw_z
        values[f"long_direct:T={d4._t(t)}"] = d4._jepa(d, target, config)
        values[f"long_composed:T={d4._t(t)}"] = d4._jepa(composed[t], target, config)
        values[f"long_pd_ln:T={d4._t(t)}"] = d4._ln(d, composed[t])
        values[f"long_pd_in:T={d4._t(t)}"] = d4._in(d, composed[t], raw)
    for t in curve_times(horizon):
        values[f"rollout:T={d4._t(t)}"] = d4._jepa(composed[t], windows[t].target.raw_z, config)
    if tuple(values) != value_names(horizon):
        raise RuntimeError("long-horizon value names differ from the registered order")
    return d4._finite(values)


def evaluate(model, features, config, horizon, device):
    """全部测试轨迹：返回 (rows, means)。rows 每条轨迹一行；means 是轨迹平均。模型的参数、梯度、缓冲区不许被改。"""
    from . import evaluation
    spec = data.datasets.active()
    modes = {m: m.training for m in model.modules()}
    identity = evaluation._state_identity(model)
    rows = []
    try:
        model.eval()
        for tid in spec.split_range("test"):
            windows = {t: load_window(features, tid, t, device, horizon) for t in window_times(horizon)}
            rows.append(dict(trajectory_id=tid, start_tick=0, values=trajectory_values(model, windows, config, horizon),
                             manifest_sha256=windows[float(horizon)].provenance["actual_manifest_sha256"]))
    finally:
        for module, flag in modes.items():
            module.training = flag
    if evaluation._state_identity(model) != identity:
        raise RuntimeError("long-horizon evaluation changed parameters, gradients or buffers")
    means = {name: sum(r["values"][name] for r in rows)/len(rows) for name in value_names(horizon)} if rows else {}
    return rows, means


def definitions(horizon) -> dict:
    return dict(start_tick=0, block=BLOCK, direct_times=list(direct_times(horizon)), curve_times=list(curve_times(horizon)),
                direct="model(z, u1⊕u2, T) with u split at T/2 (d4.direct_prediction)",
                composed="8-tick blocks, each block's input is the previous raw prediction (d4.rollout)",
                distance="comparison.jepa_distance per_sample; pd_ln/pd_in as in D4 (ln_metrics)")
