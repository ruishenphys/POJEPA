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
- pendulum、bouncing_balls、wave、cart_pole、spring_chain（S23b 登记，v8 二阶场景）：物理与渲染在 po_jepa.second_order。
  它们不是一阶系统（一帧看不到速度）：frames() 仍给数据集的 33 帧（tick 0…32），frames_with_prev() 多给 t = −1 一帧，
  留给两帧（pair）编码，见 SecondOrderSystem。

S24a 长时程（调节项 horizon ∈ {32, 64, 128}）：轨迹有 horizon 个控制 tick、horizon+1 帧（二阶场景另多 t = −1 一帧）。
下达动作、执行噪声、过程噪声、像素噪声、会动的干扰光点都按 tick 顺序从各自的随机流接着抽，所以前 32 个 tick（前 33 帧）
与 horizon=32、其余调节项相同的数据集逐位相同；horizon > 32 的下达动作是 LongControls（不是 v7 的 TrajectoryControls，
v7 的缓存和模型接口只收 32 tick，长时程接入正式流程在第 6 阶段）。

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


@dataclass(frozen=True)
class LongControls:
    """S24a：horizon > 32 的下达动作。字段和检查与 simulation.TrajectoryControls 相同，只是半开段覆盖 [0, horizon)。

    有意不做 TrajectoryControls 的子类：v7 的缓存、窗口和模型接口（cache.py、actions.py）按 isinstance 收 32 tick 的动作，
    长时程不能被它们静默当成 32 tick 用。horizon = ends_ticks[-1]，必须是 variants.HORIZONS 里大于 32 的值。
    """
    trajectory_id: int
    starts_ticks: np.ndarray
    ends_ticks: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        datasets.active().check_id(self.trajectory_id)
        for value in (self.starts_ticks, self.ends_ticks):
            if not isinstance(value, np.ndarray) or value.dtype != np.int64 or value.ndim != 1 or not value.size:
                raise ValueError("control ticks must be nonempty int64[S]")
        n = self.starts_ticks.size
        if (self.ends_ticks.shape != (n,) or not isinstance(self.values, np.ndarray) or self.values.dtype != np.float64
                or self.values.shape != (n, 2) or not np.isfinite(self.values).all() or np.any(np.abs(self.values) > .5)):
            raise ValueError("commands must be finite float64[S,2] in physical bounds")
        durations = self.ends_ticks-self.starts_ticks
        if (self.starts_ticks[0] != 0 or int(self.ends_ticks[-1]) not in variants.HORIZONS or self.ends_ticks[-1] <= TICKS
                or not np.array_equal(self.ends_ticks[:-1], self.starts_ticks[1:])
                or np.any(durations <= 0) or not np.isin(durations[:-1], [1, 2, 4, 8]).all() or durations[-1] > 8):
            raise ValueError(f"long-horizon controls must exactly cover [0,horizon) with horizon in {variants.HORIZONS[1:]} "
                             "and segment lengths 1/2/4/8")

    @property
    def horizon(self) -> int:
        return int(self.ends_ticks[-1])


def horizon_of(controls) -> int:
    """下达动作覆盖的 tick 数：TrajectoryControls 是 32，LongControls 是它的 horizon。"""
    if isinstance(controls, simulation.TrajectoryControls):
        return TICKS
    if isinstance(controls, LongControls):
        return controls.horizon
    raise ValueError("controls must be simulation.TrajectoryControls or systems.LongControls")


POLICY_HORIZON = 1.      # 秒：goal 策略按“1 秒内走到目标”给速度，u = (目标 − 估计位置)/(K·1 s)
POLICY_EXPLORE = .1      # goal 策略的探索噪声标准差（动作单位，随机流 9）


def commanded_controls(trajectory_id: int) -> simulation.TrajectoryControls:
    """模型看到的（下达的）动作。基础场景 = v6 的 sample_controls；调节项改动作抽样时用同一随机流按新规则抽。

    S21a 采集策略 policy：段长、零动作段和随机值照旧从流 1 抽（三种策略消耗流 1 的方式完全相同，所以段的边界和
    零动作段与 random 一一对应）；goal 把非零段的值换成“朝目标”的值，mixed 每个非零段按流 9 掷硬币二选一。
    “朝目标”是开环的：不跑仿真，用初值的受控点位置 p̂ 做航位推算 p̂ ← p̂ + K·u·段长，K 是下达动作到受控点速度的
    名义倍数（System.policy_anchor）。所以缓存核对动作（cache.py）时不用重跑仿真；执行时的增益、延迟、噪声、饱和
    和场景本身的动力学策略都不知道。

    S24a 长时程（horizon > 32）：同一个循环抽到 horizon 个 tick（流 1、流 9 都是接着抽），返回 LongControls。前 32 个 tick
    的逐 tick 动作与 horizon=32 相同；跨过 tick 32 的那一段在 horizon=32 时被截在 32，这里不截。
    """
    v = datasets.active().knobs
    T = v.horizon
    if not v.changes_controls() and T == TICKS:
        return simulation.sample_controls(trajectory_id)
    g = simulation._rng(trajectory_id, 1); lengths = variants.SEGMENT_SETS[v.segments]
    if v.policy != "random":
        g9 = simulation._rng(trajectory_id, 9)
        estimate, target, kin = get().policy_anchor(trajectory_id, v, g9)
    starts, ends, values, tick = [], [], [], 0
    while tick < T:
        length = int(g.choice(lengths)) if len(lengths) > 1 else lengths[0]
        value = np.zeros(2) if g.random() < v.zero else g.uniform(-.5, .5, 2)
        if v.policy != "random" and value.any():
            coin, explore = g9.random(), g9.normal(0., POLICY_EXPLORE, 2)
            if v.policy == "goal" or coin < .5:
                value = np.clip((target-estimate)/(kin*POLICY_HORIZON)+explore, -.5, .5)
        if v.axes == "x": value[1] = 0.
        if v.axes == "y": value[0] = 0.
        if v.policy != "random":
            estimate = estimate+kin*value*(min(tick+length, T)-tick)*PERIOD
        starts.append(tick); tick = min(tick+length, T); ends.append(tick); values.append(value)
    kind = simulation.TrajectoryControls if T == TICKS else LongControls
    return kind(trajectory_id, np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64),
                np.array(values, dtype=np.float64))


def applied_ticks(trajectory_id, v, controls) -> np.ndarray:
    """物理里实际执行的动作 [horizon,2]：下达动作 → 延迟 delay tick → 乘增益 → 加执行噪声（随机流 5，模型看不到）。
    S24a：长度 = 下达动作覆盖的 tick 数，必须等于 v.horizon（不许把 32 tick 的动作拿到长时程数据集上用，反之亦然）。"""
    T = horizon_of(controls)
    if T != v.horizon:
        raise ValueError(f"下达动作覆盖 {T} 个 tick，调节项的 horizon 是 {v.horizon}")
    u = np.zeros((T, 2))
    for a, b, value in zip(controls.starts_ticks, controls.ends_ticks, controls.values):
        u[a:b] = value
    if v.delay:
        u = np.concatenate((np.zeros((v.delay, 2)), u[:-v.delay]))
    u = v.gain*u
    if v.noise:
        u = u+simulation._rng(trajectory_id, 5).normal(0., v.noise, u.shape)
    if v.saturation is not None:      # S21a：执行器饱和，按分量截断（在增益和执行噪声之后）
        u = np.clip(u, -v.saturation, v.saturation)
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


# —— S21a 观察调节项：遮挡块、会动的干扰光点（new_systems 也用这两个函数，随机流由调用方给）——
OCCLUDER_SIZE = 48                          # 像素（= 1 个世界单位）
OCCLUDER_COLOR = np.array([110., 112., 120.])
DISTRACTOR_COLOR = np.array([150., 150., 150.])
MOVER_OMEGA = .6                            # rad/s，逆时针（y 向上）；所有干扰光点同一个角速度
MOVER_RADIUS = (40., 110.)                  # 像素，绕画面中心


def frame_ticks(L, ticks, horizon=TICKS):
    """每帧对应的 tick：整条轨迹（horizon+1 帧，默认 33）默认 0…horizon；单帧或几帧（反事实、规划）必须给出 ticks。"""
    if ticks is None:
        if L != horizon+1:
            raise ValueError("会动的干扰光点和像素噪声需要知道每帧的 tick：只画部分帧时要传 ticks")
        return np.arange(horizon+1)
    ticks = np.asarray(ticks)
    if ticks.shape != (L,) or ticks.min() < 0 or ticks.max() > horizon:
        raise ValueError(f"ticks 必须是 [L] 个 0…{horizon} 的整数")
    return ticks


def mover_centers(g, count, ticks) -> np.ndarray:
    """会动的干扰光点中心 [L,count,2]（行、列）：半径 r∈[40,110] 像素、初相 φ0 均匀（随机流 8），
    角度 φ0 + 0.6·t（t = tick·0.25 s）。位置只由当前角度决定、转速固定，所以只看一帧就能推出下一帧（仍是一阶）。"""
    r = g.uniform(*MOVER_RADIUS, count); phi0 = g.uniform(0., 2*np.pi, count)
    a = phi0[None, :]+MOVER_OMEGA*np.asarray(ticks, dtype=np.float64)[:, None]*PERIOD
    return np.stack((SIZE/2-r*np.sin(a), SIZE/2+r*np.cos(a)), -1)


def blend_spots(image, centers, color, sigma):
    """image [L,256,256,3] float；centers [L,K,2]（行、列）；按顺序叠加同一颜色的高斯亮斑。"""
    row, col = np.meshgrid(np.arange(SIZE)+.5, np.arange(SIZE)+.5, indexing="ij")
    for k in range(centers.shape[1]):
        r = centers[:, k, 0, None, None]; c = centers[:, k, 1, None, None]
        w = np.exp(-((col-c)**2+(row-r)**2)/(2*sigma**2))[..., None]
        image = image+(color-image)*w
    return image


def pixel_noise(g, sigma, shape, ticks=None, horizon=TICKS):
    """像素噪声（随机流 4）：整条轨迹一次抽 [horizon+1,256,256,3]（默认 33 帧），第 t 帧用第 t 份。
    画整条 33 帧且不给 ticks 时就是原来的一次抽取（逐位不变）；只画部分帧（反事实、规划，S21b）时按 ticks 取同一份噪声，
    所以单帧重画与整条轨迹的那一帧逐位相同。部分帧不给 ticks 报错（不静默地把第 0 帧的噪声用到别的 tick 上）。
    S24a：长时程按顺序多抽，前 33 帧的噪声与 horizon=32 相同。"""
    if ticks is None and shape[0] == horizon+1:
        return g.normal(0., sigma, shape)
    return g.normal(0., sigma, (horizon+1,)+tuple(shape[1:]))[frame_ticks(shape[0], ticks, horizon)]


def occlude(image, g, count):
    """不透明遮挡块：左上角在 [16, 256−16−48] 均匀取整（随机流 7），整条轨迹静止；画在场景之上、像素噪声之前。"""
    image = image.copy()
    for r, c in g.integers(16, SIZE-16-OCCLUDER_SIZE+1, (count, 2)):
        image[:, r:r+OCCLUDER_SIZE, c:c+OCCLUDER_SIZE] = OCCLUDER_COLOR
    return image


def compose(rc, colors, sigma, v, trajectory_id, backdrop=None, ticks=None):
    """可调渲染：背景 → （场景底图）→ 静止干扰光点 → 会动的干扰光点 → 场景光点 → 遮挡块 → 像素噪声。
    rc:[L,P,2]（行、列）。ticks：每帧的 tick，只在有会动的干扰光点时用到（见 frame_ticks）。"""
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
    if v.movers:
        centers = mover_centers(simulation._rng(trajectory_id, 8), v.movers, frame_ticks(L, ticks, v.horizon))
        image = blend_spots(image, centers, DISTRACTOR_COLOR, sigma)
    for p in range(rc.shape[1]):
        r = rc[:, p, 0, None, None]; c = rc[:, p, 1, None, None]
        w = np.exp(-((col-c)**2+(row-r)**2)/(2*sigma**2))[..., None]
        image = image+(colors[p]-image)*w
    if v.occluders:
        image = occlude(image, simulation._rng(trajectory_id, 7), v.occluders)
    if v.pixel_noise:
        image = image+pixel_noise(simulation._rng(trajectory_id, 4), v.pixel_noise, image.shape, ticks, v.horizon)
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
        """返回 (隐藏状态 [33,D], 画面 [33,256,256,3] uint8)。基础场景走原实现（逐位不变），有调节项时走可调实现。
        S24a：horizon > 32 时是 [horizon+1, …]（horizon 也是调节项，走可调实现）。"""
        v = datasets.active().knobs
        if v.is_default:
            states = self.reference_trajectory(trajectory_id)
            return states, self.render(states)
        states = self.variant_trajectory(trajectory_id, v)
        return states, self.draw(states, trajectory_id, v)

    # —— 可调实现（v7 可调数据集）——
    def variant_trajectory(self, trajectory_id, v, *, controls=None, start=0, state=None, ticks=None):
        """分段常数“执行”动作下的 RK4（每 tick 128 步）；可从任意 tick 的给定状态出发（反事实探针用）。

        S21a 过程噪声：dq = f(q,u) dt + σ dW。漂移每个子步用 RK4，噪声在子步末尾加 σ·√h·ξ（加性噪声的
        Euler–Maruyama）。ξ 是随机流 6 一次抽好的 [horizon,128,D] 标准正态（默认 [32,128,D]），按绝对 tick 取，所以从任意
        tick 出发的反事实仿真用的是同一份噪声（同一起点、同一噪声、只换动作）；长时程的前 32 个 tick 与 horizon=32 相同。
        """
        u = applied_ticks(trajectory_id, v, commanded_controls(trajectory_id) if controls is None else controls)
        T = len(u)
        q = (self.vinitial(trajectory_id, v) if state is None else np.asarray(state, dtype=np.float64)).copy()
        out = [q.copy()]; h = PERIOD/128
        f = lambda x, ut: self.vfield(x, ut, v)+self.wall(x)
        if v.process_noise:
            xi = simulation._rng(trajectory_id, 6).standard_normal((T, 128, q.shape[0]))
            scale = v.process_noise*math.sqrt(h)
        for tick in range(start, T if ticks is None else start+ticks):
            for sub in range(128):
                k1 = f(q, u[tick]); k2 = f(q+h*k1/2, u[tick])
                k3 = f(q+h*k2/2, u[tick]); k4 = f(q+h*k3, u[tick])
                q = q+h*(k1+2*k2+2*k3+k4)/6
                if v.process_noise:
                    q = q+scale*xi[tick, sub]
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

    def draw(self, states, trajectory_id, v, ticks=None):
        rc, colors, sigma = self.spots(states, v)
        return compose(rc, colors, v.sigma or sigma, v, trajectory_id, self.backdrop(v), ticks)

    def backdrop(self, v):
        return None

    def hidden(self, states):
        """variant_trajectory 的（可续算）状态 → frames() 返回的隐藏状态。粒子类场景两者相同；gray_scott 要池化（S21b）。"""
        return states

    def policy_anchor(self, trajectory_id, v, g):
        """S21a 采集策略的开环推算起点：(受控点初值 p̂0 [2], 目标 [2], 下达动作到受控点速度的倍数 K)。
        粒子类场景：受控点是状态前两维（多粒子是第 0 个粒子，Van der Pol 是相平面点），dq/dt 里动作的系数是 1；
        目标在 [−1.5,1.5]² 均匀（随机流 9 的前两个数）。"""
        return self.vinitial(trajectory_id, v)[:2].copy(), g.uniform(-1.5, 1.5, 2), 1.


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
        T = len(u)
        D, flow = v.param(self.name, "diffusion"), v.param(self.name, "flow")
        k = np.fft.fftfreq(SIZE, d=1./SIZE)*2*np.pi
        ky, kx = np.meshgrid(k, k, indexing="ij")
        spectrum = np.fft.fft2(self.variant_initial(trajectory_id, v) if field is None else field)
        out = [np.fft.ifft2(spectrum).real]
        for tick in range(start, T if ticks is None else start+ticks):
            wx, wy = flow*u[tick]
            spectrum = spectrum*np.exp((-D*(kx**2+ky**2)-1j*(kx*wx-ky*wy))*PERIOD)
            out.append(np.fft.ifft2(spectrum).real)
        return np.stack(out)

    def draw_fields(self, fields, trajectory_id, v, ticks=None):
        """ticks：每帧的 tick，只画部分帧且有像素噪声时必须给（见 pixel_noise）。"""
        w = np.clip(fields, 0., 1.)[..., None]
        bg = _texture(v.background)[None]
        image = bg+(np.array(COLORS[v.color])-bg)*w
        if v.occluders:
            image = occlude(image, simulation._rng(trajectory_id, 7), v.occluders)
        if v.pixel_noise:
            image = image+pixel_noise(simulation._rng(trajectory_id, 4), v.pixel_noise, image.shape, ticks, v.horizon)
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

    def _impl(self, trajectory_id, v=None, controls=None):
        """→ (new_systems 里的实现, 执行动作 [horizon,2]（默认 [32,2]）, 当前数据集的调节项)。controls：下达动作，None = 这条轨迹自己的
        （反事实探针、规划换成别的动作）。v：调用方以为的调节项，必须就是当前数据集的（S21b：探针和规划传进来的 v）。"""
        spec = datasets.active()
        if spec.system != self.name:
            # 动作、随机流和调节项都按当前数据集取，系统不一致会静默拿到别的场景的随机流和参数
            raise ValueError(f"{self.name} 需要当前数据集是 {self.name}，现在是 {spec.system}")
        if v is not None and v != spec.knobs:
            raise ValueError(f"{self.name}：传入的调节项与当前数据集 {spec.tag} 的不一致")
        v = spec.knobs
        c = commanded_controls(trajectory_id) if controls is None else controls
        return self._module().SYSTEMS[self.name], applied_ticks(trajectory_id, v, c), v

    def _module(self):
        """物理与渲染所在的模块（S23b：SecondOrderSystem 换成 second_order）。"""
        from . import new_systems
        return new_systems

    def frames(self, trajectory_id):
        impl, u, v = self._impl(trajectory_id)
        return impl.frames(trajectory_id, u, dict(v.params), v)

    def render(self, states):
        raise ValueError(f"{self.name} 的画面要用 frames() 生成")

    @staticmethod
    def _window(start, state, ticks, horizon=TICKS):
        """反事实/规划的续算区间检查：start>0 时必须给出 tick=start 的状态。→ 推进的 tick 数。horizon：轨迹的 tick 数（S24a）。"""
        ticks = horizon-start if ticks is None else ticks
        if type(start) is not int or type(ticks) is not int or not (0 <= start and 0 <= ticks and start+ticks <= horizon):
            raise ValueError(f"续算区间必须是整数 0 ≤ start ≤ start+ticks ≤ {horizon}")
        if start > 0 and state is None:
            raise ValueError("从 tick>0 出发要给出那一刻的状态 state")
        return ticks

    def variant_trajectory(self, trajectory_id, v, *, controls=None, start=0, state=None, ticks=None):
        """S21b：反事实探针（tools/probe_actions.py）和规划（tools/plan_v7.py）用的续算接口，参数与 System.variant_trajectory
        相同：从 tick=start 的（可续算）状态 state 出发，用下达动作 controls（经 applied_ticks）推进 ticks 个 tick，
        返回 [ticks+1, …]。state=None、start=0 时从这条轨迹的初值出发，整条结果与 frames() 用的仿真逐位相同。"""
        raise NotImplementedError

    def draw(self, states, trajectory_id, v, ticks=None):
        raise NotImplementedError


class GrayScottSystem(NewSystem):
    """Gray–Scott 反应扩散，动作移动注入点（new_systems.GrayScott）。隐藏状态 514 维：U、V 各 16×16 池化 + 注入点。

    S21b：池化后的隐藏状态不能续算，所以 variant_trajectory 返回的是完整状态（GrayScott.pack：U、V 各 128×128 展平，
    再接注入点 p，共 32770 维）；hidden() 把它池化成 frames() 的 514 维隐藏状态，draw() 从它画画面。"""

    def reference_trajectory(self, trajectory_id, *, steps_per_tick=None):
        """steps_per_tick：每个 tick 的分裂步数，None = 生产步长 DT=0.5（每 tick 40 步）。S20b 审核：以前静默忽略这个参数，
        现在按 dt = UNITS_PER_TICK/steps_per_tick 传下去（烧入用同一个 dt；必须整除，否则 _run 报错）。"""
        impl, u, v = self._impl(trajectory_id)
        dt = None if steps_per_tick is None else impl.UNITS_PER_TICK/steps_per_tick
        return impl.states(*impl.fields(trajectory_id, u, dict(v.params), dt=dt))

    def variant_trajectory(self, trajectory_id, v, *, controls=None, start=0, state=None, ticks=None):
        impl, u, v = self._impl(trajectory_id, v, controls)
        ticks = self._window(start, state, ticks, len(u)); params = dict(v.params)
        if state is None:
            U, V, p = impl.start_state(trajectory_id, params)
        else:
            (U,), (V,), (p,) = impl.unpack(np.asarray(state, dtype=np.float64)[None])
        return impl.pack(*impl.advance(U, V, p, u, params, start, ticks))

    def hidden(self, states):
        from . import new_systems
        impl = new_systems.SYSTEMS[self.name]
        return impl.states(*impl.unpack(states))

    def draw(self, states, trajectory_id, v, ticks=None):
        from . import new_systems
        impl = new_systems.SYSTEMS[self.name]
        if v.sigma is not None or v.distractors or v.movers:
            raise ValueError("gray_scott 没有光点：sigma/distractors/movers 不适用")
        return impl.draw(*impl.unpack(states), trajectory_id, v, ticks)


class PushSystem(NewSystem):
    """准静态推方块（new_systems.Push）。隐藏状态 (c_x, c_y, b_x, b_y, θ)。"""

    def reference_trajectory(self, trajectory_id, *, steps_per_tick=256):
        impl, u, v = self._impl(trajectory_id)
        return impl.trajectory(trajectory_id, u, dict(v.params), steps_per_tick=steps_per_tick)

    def variant_trajectory(self, trajectory_id, v, *, controls=None, start=0, state=None, ticks=None):
        impl, u, v = self._impl(trajectory_id, v, controls)
        ticks = self._window(start, state, ticks, len(u))
        return impl.trajectory(trajectory_id, u, dict(v.params), state=state, start=start, ticks=ticks)

    def draw(self, states, trajectory_id, v, ticks=None):
        from . import new_systems
        return new_systems.SYSTEMS[self.name].draw(states, trajectory_id, dict(v.params), v, ticks)

    PUSH_THROUGH = .6       # goal 策略：推杆的目标点在方块中心再往前 0.6（沿推杆→方块方向），让推杆穿过方块中心把它推走

    def policy_anchor(self, trajectory_id, v, g):
        """S21a：push 的“朝目标”是朝方块推：受控点是推杆，目标 = b0 + 0.6·(b0 − c0)/|b0 − c0|，K = Push.SPEED。
        只用初值（不跑仿真），不消耗随机流 9 的目标数。"""
        from . import new_systems
        impl = new_systems.SYSTEMS[self.name]
        q0 = impl.initial(trajectory_id, dict(v.params))
        c0, b0 = q0[:2], q0[2:4]
        d = b0-c0
        return c0.copy(), b0+self.PUSH_THROUGH*d/np.linalg.norm(d), impl.SPEED


class SecondOrderSystem(NewSystem):
    """S23b：二阶场景的登记（物理与渲染在 po_jepa.second_order，S23a）。接线与 NewSystem 相同（_impl：当前数据集的
    调节项、下达动作 commanded_controls、执行动作 applied_ticks），区别在帧数：

    second_order 每条轨迹仿真、画 34 帧，下标 f = tick f−1（t = −1 抽初值、零动作预跑一个 tick 到 t = 0）。
    - frames()：数据集的 33 帧 = 下标 1…33（tick 0…32），与一阶场景同一约定，encode_v7、缓存、训练照旧；
    - frames_with_prev()：全部 34 帧，给两帧（pair）编码用（tick 0 那一帧的前一帧是下标 0）。编码怎么用它等 S22 探针结果。
    续算接口与 GrayScottSystem 相同：variant_trajectory 返回可续算的完整状态（wave 是整场），hidden() 换成 frames() 的
    隐藏状态，draw() 按 tick（0…32；−1 也可以）画，与整条轨迹的那一帧逐位相同。
    """

    def _module(self):
        from . import second_order
        return second_order

    def frames_with_prev(self, trajectory_id):
        """→ (隐藏状态 [34, d], 画面 [34,256,256,3] uint8)，下标 f 对应 tick f−1（S24a 长时程是 horizon+2 帧）。"""
        impl, u, v = self._impl(trajectory_id)
        return impl.frames(trajectory_id, u, dict(v.params), v)

    def frames(self, trajectory_id):
        """→ (隐藏状态 [33, d], 画面 [33,256,256,3] uint8)，tick 0…32（= frames_with_prev 的下标 1…33；长时程 0…horizon）。"""
        states, frames = self.frames_with_prev(trajectory_id)
        return states[1:], frames[1:]

    def reference_trajectory(self, trajectory_id, *, steps_per_tick=None):
        """隐藏状态 [33, d]（tick 0…32）。steps_per_tick：每个 tick 的子步数，None = 生产值（pendulum 50、cart_pole 100、
        bouncing_balls 32、wave 16，预跑的那个 tick 也用它）；spring_chain 用矩阵指数精确推进、没有子步，给了就报错。"""
        impl, u, v = self._impl(trajectory_id)
        kw = {}
        if steps_per_tick is not None:
            if self.name == "spring_chain":
                raise ValueError("spring_chain 按矩阵指数精确推进，没有子步：steps_per_tick 不适用")
            if type(steps_per_tick) is not int or steps_per_tick < 1:
                raise ValueError("steps_per_tick 必须是正整数")
            kw["substeps"] = steps_per_tick
        params = dict(v.params)
        return impl.hidden(impl.trajectory(trajectory_id, u, params, **kw)[1:], params)

    def variant_trajectory(self, trajectory_id, v, *, controls=None, start=0, state=None, ticks=None):
        impl, u, v = self._impl(trajectory_id, v, controls)
        ticks = self._window(start, state, ticks, len(u)); params = dict(v.params)
        if state is None:
            state = impl.start(trajectory_id, params)[1]      # tick 0：初值 + 零动作预跑，与 trajectory() 相同
            impl.check(state, params)
        return impl.advance(state, u, params, start, ticks)

    def hidden(self, states):
        impl = self._module().SYSTEMS[self.name]
        return impl.hidden(states, dict(datasets.active().knobs.params))

    def draw(self, states, trajectory_id, v, ticks=None):
        """ticks：每帧的 tick。不给时 states 必须是整条 33 帧（tick 0…32）；second_order 的约定是不给 ticks 就当 34 帧
        （tick −1…32），所以这里必须显式换成 0…32，否则 33 帧会被当成错位一帧。"""
        if v != datasets.active().knobs:
            raise ValueError(f"{self.name}：传入的调节项与当前数据集 {datasets.active().tag} 的不一致")
        if ticks is None:
            if len(states) != v.horizon+1:
                raise ValueError("只画部分帧时要传 ticks")
            ticks = np.arange(v.horizon+1)
        return self._module().SYSTEMS[self.name].draw(states, trajectory_id, dict(v.params), v, ticks)

    def policy_anchor(self, trajectory_id, v, g):
        raise ValueError(f"{self.name} 是二阶场景：policy 只能是 random（见 variants.validate）")


REGISTRY = {
    "double_well": DoubleWell("double_well", 2, "v6 过阻尼双势阱"),
    "van_der_pol": VanDerPol("van_der_pol", 2, "Van der Pol 极限环（相平面）"),
    "multi_particle": MultiParticle("multi_particle", 6, "3 粒子过阻尼软排斥"),
    "advection_diffusion": AdvectionDiffusion("advection_diffusion", 256, "平流–扩散浓度场"),
    "landscape": Landscape("landscape", 2, "势能地形族（多势阱、环形槽、周期势、障碍物）"),
    "gray_scott": GrayScottSystem("gray_scott", 514, "Gray–Scott 反应扩散，动作移动注入点"),
    "push": PushSystem("push", 5, "准静态推方块（静摩擦阈值、坡面）"),
    # S23b：二阶场景；state_dim 是默认参数下的（bouncing_balls、spring_chain 随 n 变，与 multi_particle 相同）
    "pendulum": SecondOrderSystem("pendulum", 2, "阻尼单摆，二维力取切向分量"),
    "bouncing_balls": SecondOrderSystem("bouncing_balls", 12, "方盒弹球（碰撞、恢复系数、阻力）"),
    "wave": SecondOrderSystem("wave", 514, "周期域阻尼波动方程，动作移动压膜激振源"),
    "cart_pole": SecondOrderSystem("cart_pole", 4, "小车倒立摆（只推水平方向，欠驱动）"),
    "spring_chain": SecondOrderSystem("spring_chain", 16, "两端固定的线性弹簧链，动作推第 0 个质点"),
}
assert set(REGISTRY) == set(datasets.SYSTEMS)


def get(name: str | None = None) -> System:
    return REGISTRY[datasets.active().system if name is None else name]
