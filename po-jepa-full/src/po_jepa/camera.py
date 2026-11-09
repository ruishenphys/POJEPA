"""S24c 空间采样：输出分辨率 resolution、缩小 zoom、相机平移 pan、相机缓慢漂移 drift（docs/segments/S24c.md）。

“画布”是原来的 256×256 画面所在的坐标系（单位：画布像素，行向下、列向右，像素中心在 k + 0.5）；各场景的世界坐标到
画布坐标的映射不变（如中心列 = 128 + 48x）。相机是画布上一个轴对齐的正方形视野：

- 视野边长 W = 256/zoom（画布像素），输出 R×R 像素（R = resolution），输出像素边长 s = W/R（画布像素）。
- 视野中心 = 画布中心 (128, 128) + 偏移 o(t)。缩小留出的余量 S = (W − 256)/2；偏移半径 ρ = pan·S，方向角
  φ(t) = φ0 + drift·t·0.25 s（φ0 ~ U(0, 2π)，随机流 12；逆时针，世界 y 向上）：o_列 = ρ·cos φ，o_行 = −ρ·sin φ。
  |o| = ρ ≤ S，所以视野始终包含整块画布：画面仍能看全状态（与 variants 的约定一致）。
- 输出像素 (i, j) 的中心在画布上的位置：行 = top + (i + 0.5)·s，列 = left + (j + 0.5)·s，top/left = 128 + o − W/2。
  默认值（R = 256、zoom = 1、pan = 0）时就是 k + 0.5，各场景的画法逐位不变（默认值也不经过这里）。

画法（各场景的画图函数在 systems.camera_view(view) 里调用时按下面的规则在视野的像素中心上取值，是原生渲染，不是把
256 的画面缩放过去）：
- 解析的形状（高斯光点、线段、圆盘、方块、方块推杆）在像素中心直接求值；边缘的线性过渡宽度 = max(1, s) 画布像素
  （输出像素比画布像素大时按输出像素抗锯齿；默认 1 个画布像素）。gray_scott、wave 的 1 像素宽圆环标记，s > 1 时
  方差加 (s² − 1)/12、幅值按 σ/σ' 缩小（像素面积平均的高斯近似）。
- 背景纹理（网格线、棋盘、静态噪声块）和遮挡块是分段常数：按输出像素的面积平均（可分离的盒式滤波，精确）。
  网格线 1 个画布像素宽，s > 1 时点采样会整条漏掉，所以必须面积平均。
- 场（advection_diffusion 的 256 格浓度、gray_scott 的 128 格 U/V、wave 的 128 格位移）在格点中心之间周期双线性插值，
  再按原来的颜色映射上色。
- 画布外：整块画布画完后按像素面积和 SURROUND 色混合（canvas_mask）。SURROUND 比所有背景都暗，画布的边在画面里
  看得到，所以相机偏移能从一帧看出来；drift 是绕画布中心匀速转，当前偏移决定下一帧的偏移（仍是一阶）。

相机只改画面，不改物理、动作和隐藏状态（offsets() 给出每帧偏移，只供诊断）。帧的生成走 po_jepa.timing.frames
（主链 + 旁支，整 tick 时刻的状态与原实现逐位相同），所以空间采样能和 horizon、fps、jitter、exposure、switch_offset 组合；
曝光窗口里每个取样时刻用它自己的相机位置。像素噪声按输出分辨率抽：整 tick 的帧用流 4 按块 [R,R,3] 顺序抽，
R = 256 时与原来的一次抽 [horizon+1,256,256,3] 逐位相同。
"""
from dataclasses import dataclass
import math

import numpy as np

from . import datasets, simulation, systems, variants

CANVAS = systems.SIZE                 # 画布边长（画布像素）
SURROUND = np.array([8., 10., 12.])   # 画布外的颜色（比 BACKGROUND [24,28,32] 暗）
STREAM_CAMERA = 12


@dataclass(frozen=True)
class View:
    """一帧的相机：输出 size×size 像素，像素边长 scale（画布像素），视野左上角 (top, left)（画布坐标）。"""
    size: int
    scale: float
    top: float
    left: float

    def rows(self) -> np.ndarray:
        return self.top+(np.arange(self.size)+.5)*self.scale

    def cols(self) -> np.ndarray:
        return self.left+(np.arange(self.size)+.5)*self.scale


def width(v) -> float:
    """视野边长 W = 256/zoom（画布像素）。"""
    return CANVAS/v.zoom


def slack(v) -> float:
    """缩小留出的余量 S = (W − 256)/2：偏移半径不超过它，视野就包含整块画布。"""
    return (width(v)-CANVAS)/2


def offsets(trajectory_id: int, times, v=None) -> np.ndarray:
    """每个时刻（tick）视野中心相对画布中心的偏移 [L,2]（行、列，画布像素）。pan = 0 时全零（不抽流 12）。"""
    v = datasets.active().knobs if v is None else v
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("times 必须是有限的一维时刻（tick）")
    rho = v.pan*slack(v)
    if not rho:
        return np.zeros((len(times), 2))
    datasets.active().check_id(trajectory_id)
    phi0 = simulation._rng(trajectory_id, STREAM_CAMERA).uniform(0., 2*math.pi)
    a = phi0+v.drift*times*systems.PERIOD
    return np.stack((-rho*np.sin(a), rho*np.cos(a)), -1)


def views(trajectory_id: int, times, v=None) -> list:
    """每个时刻的 View。"""
    v = datasets.active().knobs if v is None else v
    W = width(v); s = W/v.resolution
    return [View(v.resolution, s, CANVAS/2+o_r-W/2, CANVAS/2+o_c-W/2) for o_r, o_c in offsets(trajectory_id, times, v)]


# ———————————————————— 面积平均（输出像素的足迹 [x − s/2, x + s/2]）————————————————————

def _box(F, x, s):
    """1 维盒式滤波：(F(x + s/2) − F(x − s/2))/s，F 是被滤函数的原函数。"""
    return (F(x+s/2)-F(x-s/2))/s


def interval_cover(x, s, a, b) -> np.ndarray:
    """足迹 [x − s/2, x + s/2] 落在 [a, b) 里的比例。整个足迹都在里面时精确取 1（不受 (min − max)/s 的舍入影响，
    canvas_mask 和遮挡块靠 cover == 1 判断“完全盖住”）。"""
    lo, hi = x-s/2, x+s/2
    return np.where((lo >= a) & (hi <= b), 1., np.clip(np.minimum(hi, b)-np.maximum(lo, a), 0., None)/s)


def _line_integral(x):
    """网格线（[32k, 32k+1) 处为 1）的原函数。"""
    k = np.floor(x/32)
    return k+np.clip(x-32*k, 0., 1.)


def _checker_integral(x):
    """棋盘的一维符号（floor(x/32) 偶数为 +1、奇数为 −1）的原函数（周期 64 的三角波）。"""
    y = np.mod(x, 64.)
    return 32.-np.abs(y-32.)


def _cell_weights(x, s, cell, count):
    """[len(x), count]：足迹落在第 m 个格子 [cell·m, cell·(m+1))（m 按 count 周期回绕）里的比例。"""
    lo, hi = x-s/2, x+s/2
    m0 = np.floor(lo/cell).astype(np.int64); m1 = np.floor(hi/cell).astype(np.int64)
    W = np.zeros((len(x), count))
    rows = np.arange(len(x))
    for k in range(int((m1-m0).max())+1):
        m = m0+k
        part = np.clip(np.minimum(hi, cell*(m+1))-np.maximum(lo, cell*m), 0., None)/s
        np.add.at(W, (rows, m % count), part)
    return W


def texture(kind: str, view: View) -> np.ndarray:
    """systems._texture 的四种背景在视野上的面积平均 [R,R,3]（画布外按周期延拓，最后由 canvas_mask 盖成 SURROUND）。"""
    r, c, s = view.rows(), view.cols(), view.scale
    bg = systems.BACKGROUND
    if kind == "grid":
        lr, lc = _box(_line_integral, r, s), _box(_line_integral, c, s)
        on = 1.-(1.-lr)[:, None]*(1.-lc)[None, :]
        return bg+(systems.GRID-bg)*on[..., None]
    if kind == "plain":
        return np.broadcast_to(bg, (view.size, view.size, 3)).astype(np.float64)
    if kind == "checker":
        sr, sc = _box(_checker_integral, r, s), _box(_checker_integral, c, s)
        odd = (1.-sr[:, None]*sc[None, :])/2
        return bg+14.*odd[..., None]
    if kind == "noise":
        coarse = systems._noise_coarse()[..., 0]
        Wr, Wc = _cell_weights(r, s, 8., 32), _cell_weights(c, s, 8., 32)
        return bg+(Wr@coarse@Wc.T)[..., None]
    raise ValueError(f"unknown background {kind!r}")


def square_cover(view: View, top: float, left: float, side: float) -> np.ndarray:
    """[R,R]：每个输出像素被画布上的正方形 [top, top+side) × [left, left+side) 盖住的面积比例。"""
    s = view.scale
    return interval_cover(view.rows(), s, top, top+side)[:, None]*interval_cover(view.cols(), s, left, left+side)[None, :]


def canvas_mask(image: np.ndarray, view: View) -> np.ndarray:
    """画布外画成 SURROUND：image [..., R, R, 3]（画完整块画布之后、像素噪声之前）按画布覆盖比例与 SURROUND 混合。
    完全在画布里的像素原样保留（逐位）。"""
    cover = square_cover(view, 0., 0., float(CANVAS))[..., None]
    return np.where(cover == 1., image, SURROUND+(image-SURROUND)*cover)


def sample_field(F: np.ndarray, cell: float) -> np.ndarray:
    """周期场 F [L,n,n]（格子边长 cell 画布像素，格点中心在 (m + 0.5)·cell）在当前视野像素中心上的双线性插值 → [L,R,R]。
    像素中心正好落在格点中心上时（权重 0/1）逐位等于格点值。"""
    rows, cols = systems.pixel_axes()
    n = F.shape[-1]
    if F.ndim != 3 or F.shape[-2] != n:
        raise ValueError("场必须是 [L,n,n]")

    def index(x):
        g = x/cell-.5
        i0 = np.floor(g)
        return i0.astype(np.int64) % n, (i0.astype(np.int64)+1) % n, g-i0
    r0, r1, fr = index(rows); c0, c1, fc = index(cols)
    top, bottom = F[:, r0], F[:, r1]
    upper = top[:, :, c0]*(1.-fc)+top[:, :, c1]*fc
    lower = bottom[:, :, c0]*(1.-fc)+bottom[:, :, c1]*fc
    return upper*(1.-fr)[None, :, None]+lower*fr[None, :, None]


def ring_weight(d: np.ndarray, radius: float) -> np.ndarray:
    """1 像素宽的高斯圆环标记 exp(−(d − radius)²/2)（gray_scott 的注入点、wave 的激振源）在当前视野上的权重：
    输出像素比画布像素大（边缘宽度 e = max(1, s) > 1）时方差加 (e² − 1)/12、幅值乘 1/σ'（像素面积平均的高斯近似），
    这样圆环不会因为点采样整段漏掉。默认（不在相机里）就是原式。"""
    e = systems.edge_width()
    if e == 1.:
        return np.exp(-(d-radius)**2/2.)
    var = 1.+(e*e-1.)/12.
    return np.exp(-(d-radius)**2/(2.*var))/math.sqrt(var)
