"""v7 物理系统登记：参考真值与 RGB 渲染（256×256×3 uint8）。

所有系统共用 v6 的控制协议：2 维动作 u∈[-0.5,0.5]²，控制周期 0.25 s，32 个控制 tick、33 帧，
分段长度 1/4/8 tick（simulation.sample_controls）。随机流按系统区分（simulation.seed_entropy）。

模型只看当前一帧，所以每个系统的画面都包含决定未来的全部状态（一阶系统）：
- double_well：v6 原系统，过阻尼双势阱，一个光点（直接调用 simulation.py，逐位不变）。
- van_der_pol：Van der Pol 振子的相平面 (x, v)，光点位置就是全状态；有稳定极限环。
- multi_particle：3 个不同颜色的过阻尼粒子，两两软排斥 + 简谐外势；动作只推第 0 个粒子。
- advection_diffusion：周期域上的染料浓度场，动作是均匀流速；按傅里叶模精确求解（分段常数控制下无积分误差）。
- landscape：势能地形族（只有可调实现）。
- gray_scott、push（S20b 登记，v8 新场景）：反应扩散、准静态推方块，物理与渲染在 po_jepa.new_systems。

隐藏状态只用于生成画面和诊断，不进入模型。
"""
from dataclasses import dataclass
import math

import numpy as np

from . import datasets, simulation, variants
from .variants import COLORS

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


def commanded_controls(trajectory_id: int) -> simulation.TrajectoryControls:
    """模型看到的（下达的）动作。基础场景 = v6 的 sample_controls；调节项改动作抽样时用同一随机流按新规则抽。"""
    v = datasets.active().knobs
    if not v.changes_controls():
        return simulation.sample_controls(trajectory_id)
    g = simulation._rng(trajectory_id, 1); lengths = variants.SEGMENT_SETS[v.segments]
    starts, ends, values, tick = [], [], [], 0
    while tick < TICKS:
        length = int(g.choice(lengths)) if len(lengths) > 1 else lengths[0]
        value = np.zeros(2) if g.random() < v.zero else g.uniform(-.5, .5, 2)
        if v.axes == "x": value[1] = 0.
        if v.axes == "y": value[0] = 0.
        starts.append(tick); tick = min(tick+length, TICKS); ends.append(tick); values.append(value)
    return simulation.TrajectoryControls(trajectory_id, np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64),
                                         np.array(values, dtype=np.float64))


def applied_ticks(trajectory_id, v, controls) -> np.ndarray:
    """物理里实际执行的动作 [32,2]：下达动作 → 延迟 delay tick → 乘增益 → 加执行噪声（随机流 5，模型看不到）。"""
    u = np.zeros((TICKS, 2))
    for a, b, value in zip(controls.starts_ticks, controls.ends_ticks, controls.values):
        u[a:b] = value
    if v.delay:
        u = np.concatenate((np.zeros((v.delay, 2)), u[:-v.delay]))
    u = v.gain*u
    if v.noise:
        u = u+simulation._rng(trajectory_id, 5).normal(0., v.noise, u.shape)
    return u


def _texture(kind: str) -> np.ndarray:
    """背景：grid = v6 网格；plain = 纯色；checker = 32 像素棋盘；noise = 固定的静态噪声纹理（同一数据集所有帧相同）。"""
    if kind == "grid":
        return _background()
    row, col = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    if kind == "plain":
        return np.broadcast_to(BACKGROUND, (SIZE, SIZE, 3)).astype(np.float64)
    if kind == "checker":
        return np.where((((row//32)+(col//32)) % 2 == 0)[..., None], BACKGROUND, BACKGROUND+14.)
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1703, 77])))
    coarse = g.normal(0., 8., (32, 32, 1))
    return BACKGROUND+np.kron(coarse, np.ones((8, 8, 3)))[:SIZE, :SIZE]


def compose(rc, colors, sigma, v, trajectory_id, backdrop=None):
    """可调渲染：背景 → （场景底图）→ 静止干扰光点 → 场景光点 → 像素噪声。rc:[L,P,2]（行、列）。"""
    L = rc.shape[0]
    base = _texture(v.background).astype(np.float64)
    if backdrop is not None:
        base = base+backdrop[..., None]*np.array([30., 34., 40.])
    if v.distractors:
        g = simulation._rng(trajectory_id, 3)
        fixed = g.uniform(24., SIZE-24., (v.distractors, 2))
        row, col = np.meshgrid(np.arange(SIZE)+.5, np.arange(SIZE)+.5, indexing="ij")
        for r, c in fixed:
            w = np.exp(-((col-c)**2+(row-r)**2)/(2*sigma**2))[..., None]
            base = base+(np.array([150., 150., 150.])-base)*w
    row, col = np.meshgrid(np.arange(SIZE)+.5, np.arange(SIZE)+.5, indexing="ij")
    image = np.broadcast_to(base, (L, SIZE, SIZE, 3)).copy()
    for p in range(rc.shape[1]):
        r = rc[:, p, 0, None, None]; c = rc[:, p, 1, None, None]
        w = np.exp(-((col-c)**2+(row-r)**2)/(2*sigma**2))[..., None]
        image = image+(colors[p]-image)*w
    if v.pixel_noise:
        image = image+simulation._rng(trajectory_id, 4).normal(0., v.pixel_noise, image.shape)
    return np.rint(np.clip(image, 0., 255.)).astype(np.uint8)


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
        """返回 (隐藏状态 [33,D], 画面 [33,256,256,3] uint8)。基础场景走原实现（逐位不变），有调节项时走可调实现。"""
        v = datasets.active().knobs
        if v.is_default:
            states = self.reference_trajectory(trajectory_id)
            return states, self.render(states)
        states = self.variant_trajectory(trajectory_id, v)
        return states, self.draw(states, trajectory_id, v)

    # —— 可调实现（v7 可调数据集）——
    def variant_trajectory(self, trajectory_id, v, *, controls=None, start=0, state=None, ticks=None):
        """分段常数“执行”动作下的 RK4（每 tick 128 步）；可从任意 tick 的给定状态出发（反事实探针用）。"""
        u = applied_ticks(trajectory_id, v, commanded_controls(trajectory_id) if controls is None else controls)
        q = (self.vinitial(trajectory_id, v) if state is None else np.asarray(state, dtype=np.float64)).copy()
        out = [q.copy()]; h = PERIOD/128
        f = lambda x, ut: self.vfield(x, ut, v)+self.wall(x)
        for tick in range(start, TICKS if ticks is None else start+ticks):
            for _ in range(128):
                k1 = f(q, u[tick]); k2 = f(q+h*k1/2, u[tick])
                k3 = f(q+h*k2/2, u[tick]); k4 = f(q+h*k3, u[tick])
                q = q+h*(k1+2*k2+2*k3+k4)/6
            if not np.isfinite(q).all() or np.abs(q).max() > self.BOUND:
                raise ValueError(f"{self.name} state left the rendered domain (trajectory {trajectory_id})")
            out.append(q.copy())
        return np.stack(out)

    WALL = None    # 可调版本的外围软墙位置（每个坐标）；None = 无

    def wall(self, q):
        """软墙：坐标超过 WALL 后受 20·(|q|−WALL) 的回推，大增益/大参数时把小球留在画面里（基础场景不用）。"""
        if self.WALL is None:
            return 0.
        return -20.*np.sign(q)*np.maximum(np.abs(q)-self.WALL, 0.)

    def draw(self, states, trajectory_id, v):
        rc, colors, sigma = self.spots(states, v)
        return compose(rc, colors, v.sigma or sigma, v, trajectory_id, self.backdrop(v))

    def backdrop(self, v):
        return None


class DoubleWell(System):
    """v6 原系统，直接调用 simulation.py。"""
    def initial_state(self, trajectory_id): return simulation.initial_state(trajectory_id)
    def reference_trajectory(self, trajectory_id, *, steps_per_tick=128):
        return simulation.reference_trajectory(trajectory_id, steps_per_tick=steps_per_tick)
    def render(self, states): return simulation.render_positions(states)

    # —— 可调版本（v6 原式：耦合 0.25、无倾斜）——
    SCALE = 48.
    BOUND = 2.6
    WALL = 2.3

    def vfield(self, q, u, v):
        c, tilt = v.param(self.name, "coupling"), v.param(self.name, "tilt")
        x, y = q
        return np.array([x-x**3-c*y+tilt+u[0], -y-c*x+u[1]])

    def vinitial(self, trajectory_id, v):
        return simulation.initial_state(trajectory_id)

    def spots(self, states, v):
        rc = np.stack((128-self.SCALE*states[:, 1], 128+self.SCALE*states[:, 0]), -1)[:, None]
        return rc, np.array([COLORS[v.color]]), 8.


class VanDerPol(System):
    """dx/dt = v + u_x，dv/dt = μ(1−x²)v − x + u_y，μ = 0.5；极限环振幅约 2，周期约 6.4 s。

    画面：相平面上一个光点，中心列 = 128 + 27x，中心行 = 128 − 27v；状态域 |x|,|v| ≤ 4.6。
    """
    MU = .5
    SCALE = 27.
    BOUND = 4.6     # 基础场景实测最大 3.68；可调版本（大增益/大 μ）有软墙，留出余量（128±27·4.6 像素仍在画面内）
    WALL = 3.8

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

    def vfield(self, q, u, v):
        x, w = q
        mu = v.param(self.name, "mu")
        return np.array([w+u[0], mu*(1-x*x)*w-x+u[1]])

    def vinitial(self, trajectory_id, v):
        return self.initial_state(trajectory_id)

    def spots(self, states, v):
        rc = np.stack((128-self.SCALE*states[:, 1], 128+self.SCALE*states[:, 0]), -1)[:, None]
        return rc, np.array([COLORS[v.color]]), 8.


class MultiParticle(System):
    """3 个过阻尼粒子：dq_i/dt = −k q_i + Σ_j a (q_i−q_j) exp(−|q_i−q_j|²/(2s²)) + [i=0]·u。

    k = 0.5，a = 1.5，s = 0.4。动作只直接推第 0 个（橙色）粒子，另两个（青、品红）靠相互作用被带动。
    画面：中心列 = 128 + 48x，中心行 = 128 − 48y（与双势阱同尺度），亮斑 σ = 6 像素；状态域 |x|,|y| ≤ 2.4。
    """
    K, A, S = .5, 1.5, .4
    N = 3
    SCALE = 48.
    BOUND = 2.4
    WALL = 2.1
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

    EXTRA_COLORS = np.array([[240., 220., 60.], [120., 220., 120.]])     # 第 4、5 个粒子：黄、绿

    def vfield(self, q, u, v):
        n = v.param(self.name, "n"); a = v.param(self.name, "repulsion"); k = v.param(self.name, "trap")
        p = q.reshape(n, 2)
        d = p[:, None, :]-p[None, :, :]
        w = np.exp(-(d**2).sum(-1)/(2*self.S**2))
        np.fill_diagonal(w, 0.)
        f = -k*p+a*(w[..., None]*d).sum(1)
        f[0] += u
        return f.reshape(-1)

    def vinitial(self, trajectory_id, v):
        g = simulation._rng(trajectory_id, 0)
        return g.uniform(-1.5, 1.5, 2*v.param(self.name, "n"))

    def spots(self, states, v):
        n = v.param(self.name, "n")
        p = states.reshape(states.shape[0], n, 2)
        rc = np.stack((128-self.SCALE*p[..., 1], 128+self.SCALE*p[..., 0]), -1)
        colors = np.concatenate((self.COLORS, self.EXTRA_COLORS))[:n].copy()
        colors[0] = COLORS[v.color]
        return rc, colors, 6.


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
        v = datasets.active().knobs
        if not v.is_default:
            f = self.variant_fields(trajectory_id, v)
            states = f.reshape(f.shape[0], 16, 16, 16, 16).mean((2, 4)).reshape(f.shape[0], -1)
            return states, self.draw_fields(f, trajectory_id, v)
        f = self.fields(trajectory_id)
        states = f.reshape(f.shape[0], 16, 16, 16, 16).mean((2, 4)).reshape(f.shape[0], -1)
        return states, self.render_fields(f)

    def variant_initial(self, trajectory_id, v):
        b = v.param(self.name, "blobs")
        g = simulation._rng(trajectory_id, 0)
        centers = g.uniform(0., 1., (b, 2)); sigmas = g.uniform(.05, .12, b); amps = g.uniform(.6, 1., b)
        x = (np.arange(SIZE)+.5)/SIZE
        c = np.zeros((SIZE, SIZE))
        for (cx, cy), s_, a in zip(centers, sigmas, amps):
            dx = np.min(np.abs(x[None, :]-cx+np.array([-1., 0., 1.])[:, None]), 0)
            dy = np.min(np.abs(x[None, :]-cy+np.array([-1., 0., 1.])[:, None]), 0)
            c += a*np.exp(-(dy[:, None]**2+dx[None, :]**2)/(2*s_*s_))
        return c

    def variant_fields(self, trajectory_id, v, *, controls=None, start=0, field=None, ticks=None):
        """可调版本：扩散系数、流速比例、团数；执行动作经 applied_ticks（增益、延迟、噪声）。"""
        u = applied_ticks(trajectory_id, v, commanded_controls(trajectory_id) if controls is None else controls)
        D, flow = v.param(self.name, "diffusion"), v.param(self.name, "flow")
        k = np.fft.fftfreq(SIZE, d=1./SIZE)*2*np.pi
        ky, kx = np.meshgrid(k, k, indexing="ij")
        spectrum = np.fft.fft2(self.variant_initial(trajectory_id, v) if field is None else field)
        out = [np.fft.ifft2(spectrum).real]
        for tick in range(start, TICKS if ticks is None else start+ticks):
            wx, wy = flow*u[tick]
            spectrum = spectrum*np.exp((-D*(kx**2+ky**2)-1j*(kx*wx-ky*wy))*PERIOD)
            out.append(np.fft.ifft2(spectrum).real)
        return np.stack(out)

    def draw_fields(self, fields, trajectory_id, v):
        w = np.clip(fields, 0., 1.)[..., None]
        bg = _texture(v.background)[None]
        image = bg+(np.array(COLORS[v.color])-bg)*w
        if v.pixel_noise:
            image = image+simulation._rng(trajectory_id, 4).normal(0., v.pixel_noise, image.shape)
        return np.rint(np.clip(image, 0., 255.)).astype(np.uint8)


class Landscape(System):
    """势能地形族（v7 可调数据集新增）：过阻尼小球 dq/dt = −depth·∇V(q) − ∇V_wall(q) + u。

    shape 选地形（都是一阶系统，画面上能看到小球位置与地形底图）：
    - multi_well：V = (x²−1)² + (y²−1)²，四个势阱在 (±1, ±1)；
    - ring：V = (x²+y²−1)²，半径 1 的环形槽；
    - periodic：V = 0.5(cos πx + cos πy)，周期排列的势阱；
    - obstacles：三个高斯形障碍物 V = Σ 1.5·exp(−|q−c|²/(2·0.3²))，外加弱中心势 0.1|q|²。
    外围软墙：|x| 或 |y| 超过 2 后有 20·(|q|−2) 的回推（System.wall），保证小球留在画面里。
    画面：中心列 = 128 + 48x，中心行 = 128 − 48y；地形画成暗色底图（静态，每条轨迹相同）。
    """
    SCALE = 48.
    BOUND = 2.6
    WALL = 2.
    OBSTACLES = np.array([[.9, .4], [-.7, .8], [-.2, -1.]])

    def reference_trajectory(self, trajectory_id, *, steps_per_tick=128):
        return self.variant_trajectory(trajectory_id, datasets.active().knobs)

    def render(self, states):
        raise ValueError("landscape 只有可调实现：用 frames()")

    def potential_grad(self, q, shape):
        x, y = q
        if shape == "multi_well":
            g = np.array([4*x*(x*x-1), 4*y*(y*y-1)])
        elif shape == "ring":
            g = 4*(x*x+y*y-1)*np.array([x, y])
        elif shape == "periodic":
            g = -.5*math.pi*np.array([math.sin(math.pi*x), math.sin(math.pi*y)])
        else:
            d = q[None, :]-self.OBSTACLES
            w = 1.5*np.exp(-(d**2).sum(-1)/(2*.3**2))
            g = (-(w[:, None]*d)/.3**2).sum(0)+.2*q
        return g

    def potential(self, x, y, shape):
        if shape == "multi_well":
            return (x*x-1)**2+(y*y-1)**2
        if shape == "ring":
            return (x*x+y*y-1)**2
        if shape == "periodic":
            return .5*(np.cos(np.pi*x)+np.cos(np.pi*y))
        return sum(1.5*np.exp(-((x-a)**2+(y-b)**2)/(2*.3**2)) for a, b in self.OBSTACLES)+.1*(x*x+y*y)

    def vfield(self, q, u, v):
        return -v.param(self.name, "depth")*self.potential_grad(q, v.shape)+u

    def vinitial(self, trajectory_id, v):
        return simulation._rng(trajectory_id, 0).uniform(-1.8, 1.8, 2)

    def spots(self, states, v):
        rc = np.stack((128-self.SCALE*states[:, 1], 128+self.SCALE*states[:, 0]), -1)[:, None]
        return rc, np.array([COLORS[v.color]]), 8.

    def backdrop(self, v):
        """地形底图：势能低处亮、高处暗，归一化到 [0,1]（像素 32 网格中心采样）。"""
        c = (np.arange(SIZE)+.5-128)/self.SCALE
        x, y = np.meshgrid(c, -c)
        V = np.minimum(self.potential(x, y, v.shape), 3.)
        return 1.-(V-V.min())/(V.max()-V.min()+1e-12)


class NewSystem(System):
    """S20b：v8 新场景的登记。物理与渲染在 po_jepa.new_systems（S20a），这里只接三样东西：
    当前数据集的调节项（物理参数 dict(v.params) + 观察项 v）、下达动作 commanded_controls、执行动作 applied_ticks
    （增益、延迟、执行噪声）。新场景没有 v6 原实现要保持逐位不变，基础场景和有调节项时走同一条路径。
    new_systems 在方法里才导入：它自己导入本模块，而且 v7 的 5 个场景不需要加载它。
    """

    def _impl(self, trajectory_id):
        spec = datasets.active()
        if spec.system != self.name:
            # 动作、随机流和调节项都按当前数据集取，系统不一致会静默拿到别的场景的随机流和参数
            raise ValueError(f"{self.name} 需要当前数据集是 {self.name}，现在是 {spec.system}")
        from . import new_systems
        v = spec.knobs
        return new_systems.SYSTEMS[self.name], applied_ticks(trajectory_id, v, commanded_controls(trajectory_id)), v

    def frames(self, trajectory_id):
        impl, u, v = self._impl(trajectory_id)
        return impl.frames(trajectory_id, u, dict(v.params), v)

    def render(self, states):
        raise ValueError(f"{self.name} 的画面要用 frames() 生成")

    def variant_trajectory(self, *args, **kwargs):
        # 反事实探针（tools/probe_actions.py）和规划（tools/plan_v7.py）要从任意 tick 出发，第 2 阶段再接
        raise NotImplementedError(f"{self.name} 还没接反事实探针和规划")


class GrayScottSystem(NewSystem):
    """Gray–Scott 反应扩散，动作移动注入点（new_systems.GrayScott）。隐藏状态 514 维：U、V 各 16×16 池化 + 注入点。"""

    def reference_trajectory(self, trajectory_id, *, steps_per_tick=128):
        impl, u, v = self._impl(trajectory_id)
        return impl.states(*impl.fields(trajectory_id, u, dict(v.params)))


class PushSystem(NewSystem):
    """准静态推方块（new_systems.Push）。隐藏状态 (c_x, c_y, b_x, b_y, θ)。"""

    def reference_trajectory(self, trajectory_id, *, steps_per_tick=256):
        impl, u, v = self._impl(trajectory_id)
        return impl.trajectory(trajectory_id, u, dict(v.params), steps_per_tick=steps_per_tick)


REGISTRY = {
    "double_well": DoubleWell("double_well", 2, "v6 过阻尼双势阱"),
    "van_der_pol": VanDerPol("van_der_pol", 2, "Van der Pol 极限环（相平面）"),
    "multi_particle": MultiParticle("multi_particle", 6, "3 粒子过阻尼软排斥"),
    "advection_diffusion": AdvectionDiffusion("advection_diffusion", 256, "平流–扩散浓度场"),
    "landscape": Landscape("landscape", 2, "势能地形族（多势阱、环形槽、周期势、障碍物）"),
    "gray_scott": GrayScottSystem("gray_scott", 514, "Gray–Scott 反应扩散，动作移动注入点"),
    "push": PushSystem("push", 5, "准静态推方块（静摩擦阈值、坡面）"),
}
assert set(REGISTRY) == set(datasets.SYSTEMS)


def get(name: str | None = None) -> System:
    return REGISTRY[datasets.active().system if name is None else name]
