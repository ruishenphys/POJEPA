"""v7 物理系统登记：参考真值与 RGB 渲染（256×256×3 uint8）。

所有系统共用 v6 的控制协议：2 维动作 u∈[-0.5,0.5]²，控制周期 0.25 s，32 个控制 tick、33 帧，
分段长度 1/4/8 tick（simulation.sample_controls）。随机流按系统区分（simulation.seed_entropy）。

模型只看当前一帧，所以每个系统的画面都包含决定未来的全部状态（一阶系统）：
- double_well：v6 原系统，过阻尼双势阱，一个光点（直接调用 simulation.py，逐位不变）。
- van_der_pol：Van der Pol 振子的相平面 (x, v)，光点位置就是全状态；有稳定极限环。
- multi_particle：3 个不同颜色的过阻尼粒子，两两软排斥 + 简谐外势；动作只推第 0 个粒子。
- advection_diffusion：周期域上的染料浓度场，动作是均匀流速；按傅里叶模精确求解（分段常数控制下无积分误差）。

隐藏状态只用于生成画面和诊断，不进入模型。
"""
from dataclasses import dataclass
import math

import numpy as np

from . import datasets, simulation

TICKS = simulation.TRAJECTORY_TICKS
PERIOD = simulation.CONTROL_PERIOD_SECONDS
SIZE = 256
BACKGROUND = np.array([24., 28., 32.])
GRID = np.array([36., 40., 44.])


def _background() -> np.ndarray:
    """与 v6 相同的深色背景 + 32 像素网格线。"""
    row, col = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    return np.where(((row % 32 == 0) | (col % 32 == 0))[..., None], GRID, BACKGROUND)


def _spots(centers_rc: np.ndarray, colors: np.ndarray, sigma: float) -> np.ndarray:
    """centers_rc:[L,P,2]（行、列，像素）→ [L,256,256,3] uint8，按粒子顺序依次叠加高斯亮斑。"""
    row, col = np.meshgrid(np.arange(SIZE)+.5, np.arange(SIZE)+.5, indexing="ij")
    image = np.broadcast_to(_background(), (centers_rc.shape[0], SIZE, SIZE, 3)).copy()
    for p in range(centers_rc.shape[1]):
        r = centers_rc[:, p, 0, None, None]; c = centers_rc[:, p, 1, None, None]
        w = np.exp(-((col-c)**2+(row-r)**2)/(2*sigma**2))[..., None]
        image = image+(colors[p]-image)*w
    return np.rint(image).astype(np.uint8)


def _rk4_trajectory(field, q0: np.ndarray, controls, steps_per_tick: int, check) -> np.ndarray:
    """分段常数控制下的 RK4 参考解；返回 33 个 tick 端点 [33,D] float64。"""
    q = q0.copy(); out = [q.copy()]; h = PERIOD/steps_per_tick; seg = 0
    for tick in range(TICKS):
        while controls.ends_ticks[seg] <= tick: seg += 1
        u = controls.values[seg]
        for _ in range(steps_per_tick):
            k1 = field(q, u); k2 = field(q+h*k1/2, u); k3 = field(q+h*k2/2, u); k4 = field(q+h*k3, u)
            q = q+h*(k1+2*k2+2*k3+k4)/6
        check(q)
        out.append(q.copy())
    return np.stack(out)


@dataclass(frozen=True)
class System:
    name: str
    state_dim: int
    description: str

    def initial_state(self, trajectory_id: int) -> np.ndarray: raise NotImplementedError
    def reference_trajectory(self, trajectory_id: int, *, steps_per_tick: int = 128) -> np.ndarray: raise NotImplementedError
    def render(self, states: np.ndarray) -> np.ndarray: raise NotImplementedError

    def frames(self, trajectory_id: int) -> tuple[np.ndarray, np.ndarray]:
        """返回 (隐藏状态 [33,D], 画面 [33,256,256,3] uint8)。"""
        states = self.reference_trajectory(trajectory_id)
        return states, self.render(states)


class DoubleWell(System):
    """v6 原系统，直接调用 simulation.py。"""
    def initial_state(self, trajectory_id): return simulation.initial_state(trajectory_id)
    def reference_trajectory(self, trajectory_id, *, steps_per_tick=128):
        return simulation.reference_trajectory(trajectory_id, steps_per_tick=steps_per_tick)
    def render(self, states): return simulation.render_positions(states)


class VanDerPol(System):
    """dx/dt = v + u_x，dv/dt = μ(1−x²)v − x + u_y，μ = 0.5；极限环振幅约 2，周期约 6.4 s。

    画面：相平面上一个光点，中心列 = 128 + 27x，中心行 = 128 − 27v；状态域 |x|,|v| ≤ 4.2。
    """
    MU = .5
    SCALE = 27.
    BOUND = 4.2

    def initial_state(self, trajectory_id):
        g = simulation._rng(trajectory_id, 0)
        return g.uniform(-2.5, 2.5, 2)

    def field(self, q, u):
        x, v = q
        return np.array([v+u[0], self.MU*(1-x*x)*v-x+u[1]])

    def _check(self, q):
        if not np.isfinite(q).all() or np.abs(q).max() > self.BOUND:
            raise ValueError("van_der_pol state left the rendered domain")

    def reference_trajectory(self, trajectory_id, *, steps_per_tick=128):
        c = simulation.sample_controls(trajectory_id)
        return _rk4_trajectory(self.field, self.initial_state(trajectory_id), c, steps_per_tick, self._check)

    def render(self, states):
        rc = np.stack((128-self.SCALE*states[:, 1], 128+self.SCALE*states[:, 0]), -1)[:, None]
        return _spots(rc, np.array([[230., 150., 50.]]), 8.)


class MultiParticle(System):
    """3 个过阻尼粒子：dq_i/dt = −k q_i + Σ_j a (q_i−q_j) exp(−|q_i−q_j|²/(2s²)) + [i=0]·u。

    k = 0.5，a = 1.5，s = 0.4。动作只直接推第 0 个（橙色）粒子，另两个（青、品红）靠相互作用被带动。
    画面：中心列 = 128 + 48x，中心行 = 128 − 48y（与双势阱同尺度），亮斑 σ = 6 像素；状态域 |x|,|y| ≤ 2.4。
    """
    K, A, S = .5, 1.5, .4
    N = 3
    SCALE = 48.
    BOUND = 2.4
    COLORS = np.array([[230., 150., 50.], [60., 190., 220.], [210., 80., 180.]])

    def initial_state(self, trajectory_id):
        g = simulation._rng(trajectory_id, 0)
        return g.uniform(-1.5, 1.5, 2*self.N)

    def field(self, q, u):
        p = q.reshape(self.N, 2)
        d = p[:, None, :]-p[None, :, :]
        w = np.exp(-(d**2).sum(-1)/(2*self.S**2))
        np.fill_diagonal(w, 0.)
        f = -self.K*p+self.A*(w[..., None]*d).sum(1)
        f[0] += u
        return f.reshape(-1)

    def _check(self, q):
        if not np.isfinite(q).all() or np.abs(q).max() > self.BOUND:
            raise ValueError("multi_particle state left the rendered domain")

    def reference_trajectory(self, trajectory_id, *, steps_per_tick=128):
        c = simulation.sample_controls(trajectory_id)
        return _rk4_trajectory(self.field, self.initial_state(trajectory_id), c, steps_per_tick, self._check)

    def render(self, states):
        p = states.reshape(states.shape[0], self.N, 2)
        rc = np.stack((128-self.SCALE*p[..., 1], 128+self.SCALE*p[..., 0]), -1)
        return _spots(rc, self.COLORS, 6.)


class AdvectionDiffusion(System):
    """周期单位方域上的浓度场：∂c/∂t = D∇²c − w·∇c，w = 0.4·u（域长/秒），D = 2×10⁻⁴。

    初值是 3 个高斯团（中心均匀、σ∈[0.05,0.12]、幅值∈[0.6,1]，周期延拓）。
    分段常数控制下按 256×256 傅里叶模精确推进，没有时间离散误差。
    画面：每个像素就是一个格点，颜色 = 背景 + (橙色 − 背景)·clip(c,0,1)。
    隐藏状态（只供诊断）：16×16 平均池化后的浓度，展平为 256 维。
    """
    D = 2e-4
    FLOW = .4
    BLOBS = 3

    def initial_state(self, trajectory_id):
        g = simulation._rng(trajectory_id, 0)
        centers = g.uniform(0., 1., (self.BLOBS, 2))
        sigmas = g.uniform(.05, .12, self.BLOBS)
        amps = g.uniform(.6, 1., self.BLOBS)
        return np.concatenate((centers.reshape(-1), sigmas, amps))

    def _field0(self, params):
        b = self.BLOBS
        centers = params[:2*b].reshape(b, 2); sigmas = params[2*b:3*b]; amps = params[3*b:]
        x = (np.arange(SIZE)+.5)/SIZE
        c = np.zeros((SIZE, SIZE))
        for (cx, cy), s, a in zip(centers, sigmas, amps):
            dx = np.min(np.abs(x[None, :]-cx+np.array([-1., 0., 1.])[:, None]), 0)
            dy = np.min(np.abs(x[None, :]-cy+np.array([-1., 0., 1.])[:, None]), 0)
            # 行 = y（向下为正，画面坐标），列 = x
            c += a*np.exp(-(dy[:, None]**2+dx[None, :]**2)/(2*s*s))
        return c

    def fields(self, trajectory_id) -> np.ndarray:
        """[33,256,256] float64 浓度场。"""
        c = simulation.sample_controls(trajectory_id)
        k = np.fft.fftfreq(SIZE, d=1./SIZE)*2*np.pi
        ky, kx = np.meshgrid(k, k, indexing="ij")
        spectrum = np.fft.fft2(self._field0(self.initial_state(trajectory_id)))
        out = [np.fft.ifft2(spectrum).real]; seg = 0
        for tick in range(TICKS):
            while c.ends_ticks[seg] <= tick: seg += 1
            wx, wy = self.FLOW*c.values[seg]
            # 画面行向下，物理 y 向上：u_y > 0 让染料向上（行号减小）移动
            factor = np.exp((-self.D*(kx**2+ky**2)-1j*(kx*wx-ky*wy))*PERIOD)
            spectrum = spectrum*factor
            out.append(np.fft.ifft2(spectrum).real)
        return np.stack(out)

    def reference_trajectory(self, trajectory_id, *, steps_per_tick=128):
        f = self.fields(trajectory_id)
        return f.reshape(f.shape[0], 16, 16, 16, 16).mean((2, 4)).reshape(f.shape[0], -1)

    def render(self, states):
        raise ValueError("advection_diffusion renders from full fields; use frames()")

    def render_fields(self, fields):
        w = np.clip(fields, 0., 1.)[..., None]
        bg = _background()[None]
        return np.rint(bg+(np.array([230., 150., 50.])-bg)*w).astype(np.uint8)

    def frames(self, trajectory_id):
        f = self.fields(trajectory_id)
        states = f.reshape(f.shape[0], 16, 16, 16, 16).mean((2, 4)).reshape(f.shape[0], -1)
        return states, self.render_fields(f)


REGISTRY = {
    "double_well": DoubleWell("double_well", 2, "v6 过阻尼双势阱"),
    "van_der_pol": VanDerPol("van_der_pol", 2, "Van der Pol 极限环（相平面）"),
    "multi_particle": MultiParticle("multi_particle", 6, "3 粒子过阻尼软排斥"),
    "advection_diffusion": AdvectionDiffusion("advection_diffusion", 256, "平流–扩散浓度场"),
}
assert set(REGISTRY) == set(datasets.SYSTEMS)


def get(name: str | None = None) -> System:
    return REGISTRY[datasets.active().system if name is None else name]
