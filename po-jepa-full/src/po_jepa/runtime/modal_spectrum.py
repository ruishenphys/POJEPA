"""S25f：模态频率对照（探索性，v8 规划 §1.6）。线性场景的真值模态频率和衰减率能解析写出，拿来和 PO-JEPA 学到的
阻尼旋转块对照。不保证对得上；只报描述性的量，不下结论。

PO-JEPA 的自由部分（dynamics.py 式 9/10）：每个图谱模态 k 有 32 个 2×2 块 A = [[−γ, −ω], [ω, −γ]]，γ、ω 由 (λ_k, g)
条件生成，单位是“每个控制周期”。块的特征值 −γ ± iω，换成物理单位：衰减率 γ/0.25 s⁻¹、角频率 |ω|/0.25 rad/s。
每个测试起点（主实验测试段每条轨迹的登记起点）做一次 prepare_query，取 γ、ω 和起点系数 C0 的能量
e_{k,r} = C0[k,2r]² + C0[k,2r+1]²；每个起点的能量归一化到 1，起点之间等权。模态是逐样本的图谱模态，跨样本没有对应关系，
所以只比分布，不配对。注意 A 只是“零动作、忽略非线性项”的部分：动作项和交互项 N/F 也改变实际的动力学。

真值（零动作的自由模态，loss 的时间分辨率是 4 fps，所以只列角频率 ≤ 奈奎斯特 π/0.25 = 4π rad/s 的）：
| 场景 | 模态 | 衰减率 | 角频率 |
|---|---|---|---|
| spring_chain | j = 1…n，x、y 两个方向各一个（重数 2） | γ/2 | √(ω_j² − γ²/4)，ω_j = 2√k·sin(jπ/(2(n+1)))（second_order.SpringChain.modes） |
| wave | k = 2π(m_x, m_y)，|m_x|,|m_y| ≤ 8（16×16 token 网格能表示的），m ≠ 0，按 m_x²+m_y² 合并 | γ/2 | √(c²|k|² − γ²/4) |
| advection_diffusion | 同上的波数 | D|k|² | 0（流动来自动作：k·w，不属于自由部分） |

对照量（REL_TOL = 0.1）：
- 振荡场景：near_true = 学到的块里角频率落在某个真值 f_j 的 ±10% 以内的能量比例；null_near_true = 这些区间的并在
  [0, 4π] 上占的比例（学到的频率若在带内均匀分布，near_true 的期望就是它）；每个真值模态的 nearest_rel_error：
  每个起点在“主导块”（按能量从大到小累计到 90% 的那些块）里离 f_j 最近的相对误差，取起点间中位数；decay_ratio：
  落在 f_j ±10% 以内的块的能量加权衰减率 ÷ 真值衰减率（起点间平均，没有这种块的起点不计）。
- advection_diffusion：non_oscillatory = 角频率 ≤ FREQ_RESOLUTION = 2π/8 s（一条 33 帧轨迹里不到一整圈）的能量比例，
  null_non_oscillatory = FREQ_RESOLUTION/(4π)；decay_within_true = 衰减率 ≤ 真值最大衰减率 × 1.1 的能量比例。
- 都报学到的角频率、衰减率的能量加权分位数（10%、50%、90%）和角频率的能量加权直方图（[0, 4π] 分 32 格）。
二阶场景现在按单帧复制（dup）编码，特征里看不到速度，单帧特征上的一阶模型原则上表示不出振荡：spring_chain、wave 的
对照要等 S25b 的 pair 编码后重跑才有意义。
"""

import math

import numpy as np
import torch

from po_jepa import second_order

PERIOD = .25
NYQUIST = math.pi/PERIOD
FREQ_RESOLUTION = 2*math.pi/(32*PERIOD)
REL_TOL = .1
DOMINANT = .9
MAX_WAVENUMBER = 8
HIST_BINS = 32
LINEAR_SCENES = ("advection_diffusion", "spring_chain", "wave")
OSCILLATORY = ("spring_chain", "wave")


def _param(knobs, system, name):
    return float(knobs.param(system, name))


def true_modes(system, knobs) -> list:
    """→ [dict(decay s⁻¹, frequency rad/s, multiplicity, label)]，按 (frequency, decay) 排序；只含带内（≤ 4π）的。"""
    out = []
    if system == "spring_chain":
        params = dict(knobs.params)
        _, wd, decay = second_order.SYSTEMS["spring_chain"].modes(params)
        for j, f in enumerate(wd, 1):
            out.append(dict(decay=float(decay), frequency=float(f), multiplicity=2, label=f"j={j}"))
    elif system in ("wave", "advection_diffusion"):
        m = np.arange(-MAX_WAVENUMBER, MAX_WAVENUMBER+1)
        m2 = (m[:, None]**2+m[None, :]**2).ravel()
        for s in sorted(set(m2.tolist())-{0}):
            k = 2*math.pi*math.sqrt(s)
            if system == "wave":
                c, gamma = _param(knobs, "wave", "speed"), _param(knobs, "wave", "damping")
                decay, f = gamma/2, math.sqrt(max(c*c*k*k-gamma*gamma/4, 0.))
            else:
                decay, f = _param(knobs, "advection_diffusion", "diffusion")*k*k, 0.
            out.append(dict(decay=float(decay), frequency=float(f), multiplicity=int((m2 == s).sum()), label=f"m²={s}"))
    else:
        raise ValueError(f"{system} 没有解析的线性模态（只有 {LINEAR_SCENES}）")
    return sorted((r for r in out if r["frequency"] <= NYQUIST), key=lambda r: (r["frequency"], r["decay"]))


@torch.no_grad()
def learned_blocks(model, raw_z, grid) -> dict:
    """一个起点 [1,256,1408] → dict(decay [K·32] s⁻¹, frequency [K·32] rad/s, weight [K·32]，和为 1)，float64 numpy。"""
    q = model.prepare_query(raw_z, grid)
    co = q.dynamics.coefficients
    gamma, omega = co.gamma[0].double().cpu(), co.omega[0].double().cpu()             # [K,32]，每周期
    c0 = q.initial_coefficients[0].double().cpu()                                     # [K,64]
    if gamma.shape != omega.shape or c0.shape != (gamma.shape[0], 64) or gamma.shape[1] != 32:
        raise ValueError("unexpected modal coefficient shapes")
    energy = (c0.reshape(gamma.shape[0], 32, 2)**2).sum(-1)
    total = float(energy.sum())
    if not total > 0:
        raise ValueError("initial modal coefficients are all zero")
    return dict(decay=(gamma/PERIOD).reshape(-1).numpy(), frequency=(omega.abs()/PERIOD).reshape(-1).numpy(),
                weight=(energy/total).reshape(-1).numpy())


def _quantiles(values, weights, qs=(.1, .5, .9)) -> list:
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    c = np.cumsum(w)/w.sum()
    return [float(v[min(np.searchsorted(c, q), len(v)-1)]) for q in qs]


def _union_fraction(freqs) -> float:
    spans = sorted((max(0., f*(1-REL_TOL)), min(NYQUIST, f*(1+REL_TOL))) for f in freqs if f > 0)
    total, end = 0., 0.
    for a, b in spans:
        a = max(a, end)
        if b > a:
            total += b-a; end = b
    return total/NYQUIST


def _dominant(sample) -> np.ndarray:
    order = np.argsort(-sample["weight"], kind="stable")
    c = np.cumsum(sample["weight"][order])
    return order[:int(np.searchsorted(c, DOMINANT-1e-12))+1]


def compare(system, knobs, samples) -> dict:
    """samples：learned_blocks 的列表（每个测试起点一个）→ 对照报告（文件头的量）。"""
    if not samples:
        raise ValueError("no samples")
    modes = true_modes(system, knobs)
    n = len(samples)
    freq = np.concatenate([s["frequency"] for s in samples]); decay = np.concatenate([s["decay"] for s in samples])
    weight = np.concatenate([s["weight"] for s in samples])/n
    edges = np.linspace(0., NYQUIST, HIST_BINS+1)
    hist, _ = np.histogram(np.minimum(freq, NYQUIST), bins=edges, weights=weight)
    out = dict(system=system, samples=n, blocks_per_sample=int(len(samples[0]["weight"])), true_modes=modes,
               rel_tol=REL_TOL, nyquist=NYQUIST,
               learned=dict(frequency_quantiles=_quantiles(freq, weight), decay_quantiles=_quantiles(decay, weight),
                            energy_above_nyquist=float(weight[freq > NYQUIST].sum()),
                            frequency_histogram=dict(edges=edges.tolist(), energy=hist.tolist())))
    if system in OSCILLATORY:
        f = np.array([m["frequency"] for m in modes])
        near = (np.abs(freq[:, None]-f[None, :]) <= REL_TOL*f[None, :]).any(1)
        per_mode = []
        for m in modes:
            errors, ratios = [], []
            for s in samples:
                dom = _dominant(s)
                errors.append(float(np.min(np.abs(s["frequency"][dom]-m["frequency"]))/m["frequency"]))
                close = np.abs(s["frequency"]-m["frequency"]) <= REL_TOL*m["frequency"]
                if close.any() and s["weight"][close].sum() > 0 and m["decay"] > 0:
                    ratios.append(float((s["decay"][close]*s["weight"][close]).sum()/s["weight"][close].sum())/m["decay"])
            per_mode.append(dict(label=m["label"], frequency=m["frequency"], nearest_rel_error=float(np.median(errors)),
                                 decay_ratio=float(np.mean(ratios)) if ratios else None, samples_with_near_block=len(ratios)))
        out.update(near_true=float(weight[near].sum()), null_near_true=_union_fraction(f), per_mode=per_mode)
    else:
        top = max(m["decay"] for m in modes)
        out.update(non_oscillatory=float(weight[freq <= FREQ_RESOLUTION].sum()), null_non_oscillatory=FREQ_RESOLUTION/NYQUIST,
                   decay_within_true=float(weight[decay <= top*(1+REL_TOL)].sum()), true_decay_max=top,
                   true_decay_min=min(m["decay"] for m in modes))
    return out
