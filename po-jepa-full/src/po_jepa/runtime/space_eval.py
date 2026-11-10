"""S25e：空间采样测试集上的评测（分辨率 OOD、相机缩小、平移、漂移；S24c 的四个调节项）。

模型在基础场景（256×256、16×16 网格、相机不动）上训练好，这里只评测。测试集是 `<场景>~<空间采样调节项>:<条数>` 的测试段
（编号与基础场景测试段一一对应，同初始状态、同物理、同动作，只有画面不同），用 tools/encode_v7.py --splits test 按原生分辨率
编码（po_jepa.native_encoder；runtime/space_cache.py 读），特征网格 G = resolution/16（12/16/20/24）。

窗口与主实验相同：每条测试轨迹的测试起点 tick s（datasets.test_start_ticks；2000 条的数据集每条 2 个），
时长 T ∈ DURATIONS = 1、2、3、4、8、12 tick（3、12 是留出时长；s ≤ 20，所以 s + T ≤ 32 全部有效）。
记号：z_s 是第 s 帧的特征（G×G），u 是下达动作裁到 [s, s+T)，d 是 JEPA 距离（comparison.jepa_distance，v7 的
future_jepa_distance 同一个）。

| 键 | 公式 |
|---|---|
| direct:T=k | 空间采样测试集上 d(Φ_T(z_s), z_{s+T})：模型直接吃 G×G 的原生特征，一次完整预测（d4.direct_prediction 的写法：u 在 T/2 处切开再接上，同 v7 point_metrics） |
| base_direct:T=k | 同一个模型、同一条轨迹、同一个 (s, T)，在基础测试集（主实验的 256、16×16 缓存）上的 direct |
| gap:T=k | direct − base_direct（同一窗口配对）：画面换成这种分辨率/相机以后，误差多了多少 |

各模型怎么吃 G×G（S25e 约定，模型都不改）：PO-JEPA 在 G×G 的特征上建图、求最低 K 个模态（图谱模态本来就跟着网格走）；
Transformer 的位置嵌入是归一化坐标 ((x+0.5)/G, (y+0.5)/G) 的线性函数（在 16×16 训练的坐标之间插值）；V-JEPA 2-AC 用官方
RoPE，位置按 16/G 缩放到 16 格上（ac_baseline 文件头）。时长都是整数 tick，三者都直接调 forward。identity 是 d(z_s, z_{s+T})。
G ≠ 16 时，JEPA 距离是 token 平均，不同网格之间的数值可以直接比，但 V-JEPA 2 的特征本身随分辨率变（不是同一组 token 的重采样），
所以 gap 量的是“模型 + 冻结编码器”整条链在这种画面上的退化，不是模型单独的。
"""

from pathlib import Path

import torch

from po_jepa import protocols
from . import cache, data, space_cache
from .addon import d4

TESTS = protocols.SPACE_TESTS
DURATIONS = (1, 2, 3, 4, 8, 12)
KEYS = ("direct", "base_direct", "gap")
SPACE_KNOBS = ("resolution", "zoom", "pan", "drift")


def value_names() -> tuple:
    return tuple(f"{k}:T={t}" for t in DURATIONS for k in KEYS)


def windows(trajectory_id) -> tuple:
    """一条测试轨迹上的全部 (起点 tick, 时长)，与主实验测试窗口相同。"""
    return tuple((s, t) for s in data.datasets.test_start_ticks(trajectory_id) for t in DURATIONS if s+t <= 32)


def _identity(window, approved, trajectory_id):
    x, y, p = window.inputs, window.target, window.provenance
    if (x.trajectory_id != trajectory_id or x.split != "test"
            or p["actual_manifest_sha256"] != approved["manifest_sha256_by_id"][trajectory_id]
            or p["generation"]["evidence_role"] != "real_frozen_encoder" or p["observation_variant"] != "canonical"
            or x.controls.source_ids != (f"trajectory-{trajectory_id:04d}",)):
        raise ValueError("space-sampling test identity/source/origin/endpoint mismatch")
    return window


def load_window(features, trajectory_id, start_tick, duration, device):
    """空间采样测试集的一个窗口：核根目录登记的 manifest、真实编码器、G×G 网格和端点时刻。"""
    spec = data.datasets.active()
    space_cache._space_spec()
    if type(trajectory_id) is not int or trajectory_id not in spec.split_range("test"):
        raise ValueError("space-sampling window requires a test trajectory")
    approved = data.approved_cache_metadata(features, splits=("test",))
    window = space_cache.read_window(Path(features)/f"{trajectory_id:04d}", start_tick=start_tick, duration_cycles=float(duration),
                                     expected_split="test", device=str(device))
    space_cache.window_boundary(window)
    return _identity(window, approved, trajectory_id)


def load_base_window(features, trajectory_id, start_tick, duration, device):
    """基础测试集（主实验缓存，cache.read_window）的同一个窗口；当前数据集必须是不带空间、时间采样的基础场景。"""
    spec = data.datasets.active()
    if spec.knobs.changes_space() or spec.knobs.changes_timing() or spec.knobs.horizon != 32:
        raise ValueError(f"{spec.tag}: the reference pass reads the base test set (no space/time sampling, horizon 32)")
    if type(trajectory_id) is not int or trajectory_id not in spec.split_range("test"):
        raise ValueError("space-sampling window requires a test trajectory")
    approved = data.approved_cache_metadata(features, splits=("test",))
    window = cache.read_window(Path(features)/f"{trajectory_id:04d}", start_tick=start_tick, duration_cycles=float(duration),
                               expected_split="test", device=str(device))
    data._window_boundary(window, window)
    return _identity(window, approved, trajectory_id)


@torch.no_grad()
def window_value(model, window, config) -> float:
    """d(Φ_T(z_s), z_{s+T})，Φ_T 是 d4.direct_prediction（同 v7 point_metrics 的 direct）。"""
    direct, _, _ = d4.direct_prediction(model, window)
    return d4._finite({"direct": d4._jepa(direct, window.target.raw_z, config)})["direct"]


def evaluate(model, features, config, device, *, base=False):
    """全部测试轨迹、全部窗口：返回 rows，每个 (轨迹, 起点, 时长) 一行 {trajectory_id, start_tick, duration, value, manifest_sha256}。
    base=True 时读基础测试集（load_base_window），否则读空间采样测试集。模型的参数、梯度、缓冲区不许被改。"""
    from . import evaluation
    spec = data.datasets.active()
    load = load_base_window if base else load_window
    if not base:
        space_cache._space_spec()
    modes = {m: m.training for m in model.modules()}
    identity = evaluation._state_identity(model)
    rows = []
    try:
        model.eval()
        for tid in spec.split_range("test"):
            for s, t in windows(tid):
                w = load(features, tid, s, t, device)
                rows.append(dict(trajectory_id=tid, start_tick=s, duration=t, value=window_value(model, w, config),
                                 manifest_sha256=w.provenance["actual_manifest_sha256"]))
    finally:
        for module, flag in modes.items():
            module.training = flag
    if evaluation._state_identity(model) != identity:
        raise RuntimeError("space-sampling evaluation changed parameters, gradients or buffers")
    return rows


def paired(space_rows, base_rows):
    """按 (轨迹, 起点, 时长) 配对：→ (rows, means)。两边的窗口集合必须完全相同。"""
    key = lambda r: (r["trajectory_id"], r["start_tick"], r["duration"])
    base = {key(r): r for r in base_rows}
    if len(base) != len(base_rows) or sorted(base) != sorted(key(r) for r in space_rows) or len(space_rows) != len(base):
        raise ValueError("space-sampling and base passes must cover exactly the same windows")
    rows = []
    for r in space_rows:
        b = base[key(r)]
        t = r["duration"]
        rows.append(dict(trajectory_id=r["trajectory_id"], start_tick=r["start_tick"], duration=t,
                         values={f"direct:T={t}": r["value"], f"base_direct:T={t}": b["value"], f"gap:T={t}": r["value"]-b["value"]},
                         manifest_sha256=r["manifest_sha256"], base_manifest_sha256=b["manifest_sha256"]))
    means = {}
    for name in value_names():
        xs = [r["values"][name] for r in rows if name in r["values"]]
        if xs:
            means[name] = sum(xs)/len(xs)
    for t in DURATIONS:
        n = sum(1 for r in rows if r["duration"] == t)
        if n:
            means[f"windows:T={t}"] = n
    return rows, means


def definitions() -> dict:
    v = data.datasets.active().knobs
    return dict(resolution=v.resolution, zoom=v.zoom, pan=v.pan, drift=v.drift, grid=[space_cache.side()]*2,
                durations=list(DURATIONS), starts="datasets.test_start_ticks (same windows as the main test set)",
                direct="model(z_s, u1⊕u2, T) on the native G×G features (d4.direct_prediction), d = comparison.jepa_distance",
                base_direct="the same model and window on the base 256-pixel 16×16 test cache",
                gap="direct − base_direct, paired by (trajectory, start, duration)",
                encoder="frozen V-JEPA 2 at the native resolution (po_jepa.native_encoder; RoPE positions 0..G-1)")
