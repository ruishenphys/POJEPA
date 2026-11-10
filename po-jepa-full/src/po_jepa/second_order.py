"""v8 二阶场景（S23a，第 4 阶段前一段）：pendulum、bouncing_balls、wave、cart_pole、spring_chain。纯函数：物理真值 + 渲染。

规划见项目文件夹 po-jepa-v8-plan §1.3、§1.4；说明和验收单 docs/segments/S23a.md。S23b 登记（docs/segments/S23b.md）：
datasets.SYSTEMS 末尾追加五个名字，物理参数并入 variants.PARAMS，systems.SecondOrderSystem 接调节项、下达动作和执行动作；
systems 在方法里才导入本文件（本文件导入 systems）。

共同约定：
- 控制周期 0.25 s，32 个控制 tick；输入 u 是“执行”动作 [32,2] float64（已含增益、延迟、执行噪声、饱和；由 systems.applied_ticks 给出）。
- 两帧输入（v8 规划 §1.4、S22 探针）：二阶系统一帧看不到速度，所以每条轨迹多仿真、多画一帧 t = −1，共 34 帧，
  下标 f 对应 tick f − 1。初值在 t = −1 抽（随机流 0），t = −1 → 0 是零动作的“预跑”一个 tick（和 delay 调节项在 t = 0 之前
  补零的约定一致），之后第 t 个 tick 执行 u[t]。数据集的 33 帧是下标 1…33，和一阶场景一样；第 0 帧只给 pair 编码当前一帧。
  t = −1 和 t = 0 两帧（及之后任意相邻两帧）合起来决定未来（位置在画面上，速度由两帧之差给出）。
- 画面 256×256×3 uint8，世界坐标 y 向上，中心列 = 128 + SCALE·x，中心行 = 128 − SCALE·y。
- 随机流：熵 [1703, 1000+SYSTEM_INDEX[name], 轨迹编号, 流号]。SYSTEM_INDEX 是五个名字按这里的顺序追加到
  datasets.SYSTEMS 末尾（gray_scott 5、push 6 之后）的位置 7…11（S23b 已登记），simulation.seed_entropy 给出同一个熵。
  流 0 = 初值，3 = 静止干扰光点，4 = 像素噪声，7 = 遮挡块，8 = 会动的干扰光点（与 systems.compose 一致）。
  像素噪声一次抽 34 帧（[34,256,256,3]），第 f 帧用第 f 份；只画部分帧时按 tick 取同一份（续算、规划用）。
- 续算：advance(q, u, params, start, ticks) 从 tick=start 的状态推进；trajectory() = 初值 + 预跑 + advance(0, 32)，
  所以从整条轨迹的第 start 个 tick 续算与整条轨迹逐位相同（反事实探针、规划用）。
- 物理参数：名字 → (默认值, 允许范围)，唯一来源是 variants.PARAMS（S23b 并入，和 new_systems 一样）。
- S24a 长时程：u 是 [horizon,2]（horizon = 64、128），trajectory() 给 horizon+2 帧（tick −1…horizon），像素噪声一次抽
  horizon+2 帧；都按 tick 顺序接着抽，前 34 帧与 horizon=32 逐位相同。
"""
import math

import numpy as np
from scipy.linalg import expm

from . import systems, variants
from .systems import COLORS, DISTRACTOR_COLOR, SIZE, _texture, blend_spots, mover_centers, occlude

TICKS = 32
PERIOD = .25
FRAMES = TICKS+2                   # 帧 0…33 = tick −1…32
SYSTEM_INDEX = {"pendulum": 7, "bouncing_balls": 8, "wave": 9, "cart_pole": 10, "spring_chain": 11}

# S23b：物理参数的唯一来源是 variants.PARAMS（同一个对象，数值与 S23a 相同）
PARAMS = {name: variants.PARAMS[name] for name in SYSTEM_INDEX}
# 光点类场景的其他光点颜色（第 0 个受控物体用调节项颜色）：青、品红、黄、绿、紫
OTHER_COLORS = np.array([[60., 190., 220.], [210., 80., 180.], [240., 220., 60.], [120., 220., 120.], [160., 120., 240.]])
ROD_COLOR = np.array([110., 112., 120.])           # 摆杆、弹簧、轨道（与 S22 探针的摆杆同色）
BODY_COLOR = np.array([190., 195., 205.])          # 小车、墙（与 push 的方块同色）
MARK_COLOR = np.array([230., 70., 200.])           # 波动方程的激振源圆环（与 gray_scott 的注入点同色）


def rng(system: str, trajectory_id: int, stream: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [1703, 1000+SYSTEM_INDEX[system], int(trajectory_id), int(stream)])))


def param(system: str, params: dict, name: str):
    default, (lo, hi) = PARAMS[system][name]
    value = params.get(name, default)
    if isinstance(default, int):
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f"{system} 的 {name} 必须是 [{lo},{hi}] 内的整数")
        return value
    if not (math.isfinite(value) and lo <= value <= hi):
        raise ValueError(f"{system} 的 {name} 必须在 [{lo},{hi}]")
    return float(value)


def _check_controls(u):
    """执行动作 [horizon,2]：默认 32 个 tick；S24a 长时程是 variants.HORIZONS 里的 64、128。"""
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[0] not in variants.HORIZONS or u.shape[1] != 2 or not np.isfinite(u).all():
        raise ValueError(f"执行动作必须是有限 float64[horizon,2]，horizon ∈ {variants.HORIZONS}")
    return u


def frame_ticks(L, ticks, horizon=TICKS):
    """每帧对应的 tick：整条轨迹（horizon+2 帧，默认 34）默认 −1…horizon；只画部分帧时必须给出 ticks（−1…horizon）。
    S24b：ticks 可以是 tick 内部的时刻（浮点，时间采样调节项，只给会动的干扰光点用）；像素噪声按 tick 取，那里另核整数。"""
    if ticks is None:
        if L != horizon+2:
            raise ValueError("只画部分帧时要传 ticks（会动的干扰光点、像素噪声要知道每帧的 tick）")
        return np.arange(-1, horizon+1)
    ticks = np.asarray(ticks)
    if (ticks.shape != (L,) or not (np.issubdtype(ticks.dtype, np.integer) or np.issubdtype(ticks.dtype, np.floating))
            or not np.isfinite(ticks).all() or ticks.min() < -1 or ticks.max() > horizon):
        raise ValueError(f"ticks 必须是 [L] 个 −1…{horizon} 的数")
    return ticks


def _layers(system, trajectory_id, v, L, ticks, sigma):
    """背景 → 静止干扰光点（流 3）→ 会动的干扰光点（流 8）→ [L,256,256,3] float64（与 systems.compose 同一顺序）。"""
    base = _texture(v.background).astype(np.float64)
    if v.distractors:
        row, col = systems.pixel_grid()
        for r, c in rng(system, trajectory_id, 3).uniform(24., SIZE-24., (v.distractors, 2)):
            w = np.exp(-((col-c)**2+(row-r)**2)/(2*sigma**2))[..., None]
            base = base+(DISTRACTOR_COLOR-base)*w
    image = np.broadcast_to(base, (L,)+base.shape).copy()
    if v.movers:
        image = blend_spots(image, mover_centers(rng(system, trajectory_id, 8), v.movers, frame_ticks(L, ticks, v.horizon)),
                            DISTRACTOR_COLOR, sigma)
    return image


def _finish(image, system, trajectory_id, v, ticks):
    """遮挡块（流 7）→ 像素噪声（流 4，一次抽 horizon+2 帧（默认 34），第 f 帧用 tick f−1 那份）→ uint8。
    ticks 不管开没开像素噪声都核（Codex 复审 S24a）：不给时必须是整条 horizon+2 帧，给了必须在 −1…horizon 内。"""
    t = frame_ticks(image.shape[0], ticks, v.horizon)
    if v.occluders:
        image = occlude(image, rng(system, trajectory_id, 7), v.occluders)
    if systems._RAW[0]:           # S24b：timing 要像素噪声和取整之前的画面（systems.raw_images）
        return image
    if v.pixel_noise:
        noise = rng(system, trajectory_id, 4).normal(0., v.pixel_noise, (v.horizon+2,)+image.shape[1:])
        if not np.issubdtype(t.dtype, np.integer):
            raise ValueError("像素噪声按整 tick 取：ticks 必须是整数（tick 内部的帧由 po_jepa.timing 加噪声）")
        image = image+noise[t+1]
    return np.rint(np.clip(image, 0., 255.)).astype(np.uint8)


def _grid():
    """像素中心的世界坐标需要的网格（行、列，像素）。S24c：相机里是视野像素中心的画布坐标（systems.pixel_grid）。"""
    return systems.pixel_grid()


def _segment(image, a, b, color, half_width):
    """抗锯齿线段（端点 a、b 为行、列像素）：到线段距离 ≤ half_width 全覆盖，再往外 1 像素线性淡出。image [256,256,3]。
    S24c：淡出宽度 e = systems.edge_width()（相机里输出像素比画布像素大时按输出像素；默认 1，式子与原来逐位相同）。"""
    row, col = _grid()
    d = np.asarray(b, dtype=np.float64)-np.asarray(a, dtype=np.float64)
    s = np.clip(((row-a[0])*d[0]+(col-a[1])*d[1])/max(d @ d, 1e-12), 0., 1.)
    dist = np.hypot(row-(a[0]+s*d[0]), col-(a[1]+s*d[1]))
    e = systems.edge_width()
    w = (np.clip(half_width+.5*e-dist, 0., e)/e)[..., None]
    return image+(color-image)*w


def _spot(image, rc, color, sigma):
    row, col = _grid()
    w = np.exp(-((row-rc[0])**2+(col-rc[1])**2)/(2*sigma**2))[..., None]
    return image+(color-image)*w


def _disc(image, rc, color, radius):
    """抗锯齿实心圆（半径 radius 像素，边缘 1 像素线性过渡）。"""
    row, col = _grid()
    e = systems.edge_width()
    w = (np.clip(radius+.5*e-np.hypot(row-rc[0], col-rc[1]), 0., e)/e)[..., None]
    return image+(color-image)*w


def _box(image, center_rc, half_rc, color):
    """抗锯齿实心矩形（中心、半边长：行、列，像素）。"""
    row, col = _grid()
    er = np.abs(row-center_rc[0])-half_rc[0]; ec = np.abs(col-center_rc[1])-half_rc[1]
    sd = np.hypot(np.maximum(er, 0.), np.maximum(ec, 0.))+np.minimum(np.maximum(er, ec), 0.)
    e = systems.edge_width()
    w = (np.clip(.5*e-sd, 0., e)/e)[..., None]
    return image+(color-image)*w


def _rk4(field, q, h, n):
    for _ in range(n):
        k1 = field(q); k2 = field(q+h*k1/2); k3 = field(q+h*k2/2); k4 = field(q+h*k3)
        q = q+h*(k1+2*k2+2*k3+k4)/6
    return q


class SecondOrder:
    """共同的时间推进：initial（t=−1 的状态）→ 预跑一个零动作 tick → advance。子类实现 initial、tick、draw，
    状态是一维 float64 向量（wave 是打包的完整场）。"""
    name = ""
    SPOT_SIGMA = 6.
    ALLOW_SIGMA = True       # 光点大小 sigma 调节项是否适用（弹球是有物理半径的圆盘、波动方程是场：不适用）
    ALLOW_SPOTS = True       # 干扰光点（distractors、movers）是否适用（波动方程是场：不适用）

    def check_variant(self, v):
        if v.sigma is not None and not self.ALLOW_SIGMA:
            raise ValueError(f"{self.name} 的画面没有可调大小的光点：sigma 不适用")
        if (v.distractors or v.movers) and not self.ALLOW_SPOTS:
            raise ValueError(f"{self.name} 是场：distractors/movers 不适用")

    def start(self, trajectory_id, params, **kw):
        """→ (t=−1 的状态, t=0 的状态)：初值 + 零动作预跑一个 tick。"""
        q = self.initial(trajectory_id, params)
        return q, self.tick(q, np.zeros(2), params, **kw)

    def advance(self, q, u, params, start, ticks, **kw):
        """从 tick=start 的状态 q 推进 ticks 个 tick（执行动作取 u[start:start+ticks]）→ [ticks+1, D]。"""
        u = _check_controls(u)
        if type(start) is not int or type(ticks) is not int or not 0 <= start <= start+ticks <= len(u):
            raise ValueError(f"续算区间必须是整数 0 ≤ start ≤ start+ticks ≤ {len(u)}")
        q = np.asarray(q, dtype=np.float64).copy(); out = [q]
        for t in range(start, start+ticks):
            q = self.tick(q, u[t], params, **kw)
            self.check(q, params)
            out.append(q)
        return np.stack(out)

    def trajectory(self, trajectory_id, u, params, **kw):
        """→ 状态 [34, D]，下标 f 对应 tick f−1（S24a 长时程 [horizon+2, D]）。"""
        u = _check_controls(u)
        q_prev, q0 = self.start(trajectory_id, params, **kw)
        self.check(q0, params)
        return np.concatenate((q_prev[None], self.advance(q0, u, params, 0, len(u), **kw)))

    def check(self, q, params):
        if not np.isfinite(q).all():
            raise ValueError(f"{self.name} 数值发散")

    def hidden(self, states, params):
        """（可续算的）状态 → 隐藏状态（只供诊断）。除 wave 外两者相同。"""
        return states

    def frames(self, trajectory_id, u, params, v):
        """→ (隐藏状态 [34, d], 画面 [34,256,256,3] uint8)，下标 f 对应 tick f−1。"""
        self.check_variant(v)
        states = self.trajectory(trajectory_id, u, params)
        return self.hidden(states, params), self.draw(states, trajectory_id, params, v)


# ———————————————————————————— pendulum ————————————————————————————

class Pendulum(SecondOrder):
    """阻尼单摆：θ̈ = −(g/L)·sin θ − γ·θ̇ + K·(u_x cos θ + u_y sin θ)。

    θ = 0 竖直向下，θ 增大时摆锤往画面右边走；摆锤的世界位置 L·(sin θ, −cos θ)，切向单位向量 (cos θ, sin θ)。
    动作是作用在摆锤上的二维力，只有切向分量起作用（力矩 = L·F·t̂，θ̈ 的贡献 = F·t̂/(mL)），K = 4 rad/s² 每单位动作
    （u = (0.5, 0) 在 θ = 0 时给 2 rad/s²，与 S22 探针单摆的标量力矩相同）。
    物理参数：g_over_l（默认 9.81 s⁻²，小角周期 2.006 s，每周期约 8 帧）、damping γ（默认 0.2 s⁻¹）。
    初值（t = −1，流 0）：θ ∈ U(−π, π)、θ̇ ∈ U(−3, 3) rad/s。每 tick RK4 50 步。
    相邻两帧转角不保证 < π（S23a-fix1，Codex 补审）：动作能往摆里持续做功，g/l = 16、无阻尼时专门构造的动作序列能到 1.43π。
    v6 规则的随机动作下实测（每组 2000 条）：默认参数增益 1/2/4 最大 0.53π/0.58π/0.67π；最极端的角（g/l = 16、无阻尼）
    增益 4 时 0.05% 的轨迹超过 π（方向混叠），增益 ≤ 2 时没有。
    画面：S22 探针的画法（支点在画面中心，灰色抗锯齿摆杆 80 像素，摆锤是调节项颜色的高斯光点 σ = 6）。只由 θ 决定。
    状态 (θ, θ̇)，θ 不回绕。
    """
    name = "pendulum"
    K = 4.
    SUBSTEPS = 50
    ROD_PX = 80.

    def initial(self, trajectory_id, params):
        g = rng(self.name, trajectory_id, 0)
        return np.array([g.uniform(-math.pi, math.pi), g.uniform(-3., 3.)])

    def field(self, q, u, gl, gamma):
        th, om = q
        return np.array([om, -gl*math.sin(th)-gamma*om+self.K*(u[0]*math.cos(th)+u[1]*math.sin(th))])

    def tick(self, q, u, params, substeps=None):
        gl, gamma = param(self.name, params, "g_over_l"), param(self.name, params, "damping")
        n = self.SUBSTEPS if substeps is None else substeps
        return _rk4(lambda x: self.field(x, u, gl, gamma), q, PERIOD/n, n)

    def bob_rc(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        return np.stack((SIZE/2+self.ROD_PX*np.cos(theta), SIZE/2+self.ROD_PX*np.sin(theta)), -1)

    def draw(self, states, trajectory_id, params, v, ticks=None):
        self.check_variant(v)
        sigma = v.sigma or self.SPOT_SIGMA
        image = _layers(self.name, trajectory_id, v, len(states), ticks, sigma)
        pivot = (SIZE/2, SIZE/2)
        for i, rc in enumerate(self.bob_rc(states[:, 0])):
            image[i] = _spot(_segment(image[i], pivot, rc, ROD_COLOR, 1.5), rc, np.array(COLORS[v.color]), sigma)
        return _finish(image, self.name, trajectory_id, v, ticks)


# ———————————————————————————— cart_pole ————————————————————————————

class CartPole(SecondOrder):
    """小车倒立摆（水平轨道上的小车 + 无质量杆末端的点质量摆锤），动作只有水平推力 F = K·u_x（u_y 不起作用，欠驱动）。

    θ = 0 竖直向上，θ 增大时摆锤往右倒；摆锤位置 (x + l sin θ, l cos θ)。拉格朗日方程（M 小车质量 = 1，m = mass_ratio·M）：
      (M+m)·ẍ + m·l·cos θ·θ̈ = F + m·l·θ̇²·sin θ − b·ẋ + F_wall
      m·l·cos θ·ẍ + m·l²·θ̈ = m·g·l·sin θ − c·θ̇，c = m·l²·POLE_DAMPING
    解 2×2 线性方程（行列式 m·l²·(M + m·sin²θ) > 0）。g = 9.81，K = 1.5 N 每单位动作，POLE_DAMPING = 0.1 s⁻¹。
    轨道两端有缓冲：|x| > 1.6 后 F_wall = −400·(|x| − 1.6)·sgn x − 20·ẋ（默认参数下约 40% 的轨迹碰到缓冲，约 2% 的帧在缓冲里）。
    物理参数：length l（默认 1，[0.6, 1.4]）、mass_ratio m/M（默认 0.3）、friction b（小车粘滞摩擦，默认 0.5）。
    初值（t = −1，流 0）：x ∈ U(−1, 1)、ẋ ∈ U(−0.5, 0.5)、θ ∈ U(−π, π)、θ̇ ∈ U(−2, 2)。每 tick RK4 100 步。
    画面（每单位 36 像素）：灰色轨道和两端缓冲块，小车浅灰矩形（0.5 × 0.25），灰色摆杆，摆锤是调节项颜色的光点。
    状态 (x, ẋ, θ, θ̇)，θ 不回绕。
    """
    name = "cart_pole"
    SCALE = 36.
    G = 9.81
    M = 1.
    K = 1.5
    POLE_DAMPING = .1
    TRACK = 1.6
    WALL_K, WALL_C = 400., 20.
    BOUND = 2.
    SUBSTEPS = 100
    CART_HALF = (.125, .25)          # 半高、半宽（世界单位）

    def initial(self, trajectory_id, params):
        g = rng(self.name, trajectory_id, 0)
        return np.array([g.uniform(-1., 1.), g.uniform(-.5, .5), g.uniform(-math.pi, math.pi), g.uniform(-2., 2.)])

    def field(self, q, u, l, m, b):
        x, xd, th, om = q
        s, c = math.sin(th), math.cos(th)
        wall = 0.
        if abs(x) > self.TRACK:
            wall = -self.WALL_K*(abs(x)-self.TRACK)*math.copysign(1., x)-self.WALL_C*xd
        r1 = self.K*u[0]+m*l*om*om*s-b*xd+wall
        r2 = m*self.G*l*s-m*l*l*self.POLE_DAMPING*om
        det = m*l*l*(self.M+m*s*s)
        return np.array([xd, (m*l*l*r1-m*l*c*r2)/det, om, ((self.M+m)*r2-m*l*c*r1)/det])

    def tick(self, q, u, params, substeps=None):
        l, ratio, b = (param(self.name, params, n) for n in ("length", "mass_ratio", "friction"))
        n = self.SUBSTEPS if substeps is None else substeps
        return _rk4(lambda z: self.field(z, u, l, ratio*self.M, b), q, PERIOD/n, n)

    def check(self, q, params):
        if not np.isfinite(q).all() or abs(q[0]) > self.BOUND:
            raise ValueError("cart_pole 小车离开了轨道")

    def energy(self, q, params):
        """无摩擦、无推力、不碰缓冲时守恒的机械能（测试用）。"""
        l, m = param(self.name, params, "length"), param(self.name, params, "mass_ratio")*self.M
        x, xd, th, om = q
        return .5*(self.M+m)*xd**2+m*l*xd*om*math.cos(th)+.5*m*l*l*om**2+m*self.G*l*math.cos(th)

    def to_rc(self, x, y):
        return np.array([SIZE/2-self.SCALE*y, SIZE/2+self.SCALE*x])

    def draw(self, states, trajectory_id, params, v, ticks=None):
        self.check_variant(v)
        sigma = v.sigma or self.SPOT_SIGMA
        l = param(self.name, params, "length")
        image = _layers(self.name, trajectory_id, v, len(states), ticks, sigma)
        end = self.TRACK+self.CART_HALF[1]
        for i, (x, _, th, _) in enumerate(states):
            img = _segment(image[i], self.to_rc(-end, 0.), self.to_rc(end, 0.), ROD_COLOR, 1.)
            for side in (-1., 1.):
                img = _box(img, self.to_rc(side*(end+.05), 0.), (.15*self.SCALE, .05*self.SCALE), ROD_COLOR)
            img = _box(img, self.to_rc(x, 0.), (self.CART_HALF[0]*self.SCALE, self.CART_HALF[1]*self.SCALE), BODY_COLOR)
            bob = self.to_rc(x+l*math.sin(th), l*math.cos(th))
            img = _segment(img, self.to_rc(x, 0.), bob, ROD_COLOR, 1.5)
            image[i] = _spot(img, bob, np.array(COLORS[v.color]), sigma)
        return _finish(image, self.name, trajectory_id, v, ticks)


# ———————————————————————————— spring_chain ————————————————————————————

class SpringChain(SecondOrder):
    """弹簧链：n 个单位质量的质点，静止位置在 x 轴上等距排开（两端固定在 x = ±2 的墙上），相邻质点、端点与墙之间是线性弹簧。
    位移 d_i ∈ R²（两个方向互不耦合、方程相同）：
      d̈_i = −k·(2d_i − d_{i−1} − d_{i+1}) − γ·ḋ_i + [i = 0]·K·u，d_{−1} = d_n = 0（墙）
    动作推第 0 个（最左边）质点，K = 1.5。第 0 个质点在每个模态里都有分量（sin(jπ/(n+1)) ≠ 0），所以每个模态都能被激发。
    模态频率（无阻尼）ω_j = 2√k·sin(jπ/(2(n+1)))，j = 1…n；有阻尼时 ω_j,d = √(ω_j² − γ²/4)，衰减率 γ/2。
    分段常数动作下按矩阵指数精确推进（scipy.linalg.expm，[d, ḋ, 1] 的增广系统），没有时间离散误差。
    物理参数：n（默认 4，2…6）、stiffness k（默认 16 s⁻²，[9, 25]）、damping γ（默认 0.1 s⁻¹，[0, 0.5]）。
    初值（t = −1，流 0）：d ∈ U(−0.15, 0.15)²、ḋ ∈ U(−0.5, 0.5)²。
    画面（每单位 48 像素）：两端浅灰墙条，灰色弹簧细线，质点是光点（第 0 个调节项颜色，其余青、品红、黄、绿、紫）。
    状态：质点世界位置 [n,2] 展平后接速度 [n,2]，共 4n 维。
    """
    name = "spring_chain"
    SCALE = 48.
    ANCHOR = 2.
    K = 1.5
    BOUND = 2.6

    def rest(self, n):
        return -self.ANCHOR+2*self.ANCHOR*(np.arange(n)+1.)/(n+1)

    def initial(self, trajectory_id, params):
        n = param(self.name, params, "n"); g = rng(self.name, trajectory_id, 0)
        d = g.uniform(-.15, .15, (n, 2)); dv = g.uniform(-.5, .5, (n, 2))
        pos = d+np.stack((self.rest(n), np.zeros(n)), -1)
        return np.concatenate((pos.ravel(), dv.ravel()))

    def matrices(self, params):
        """→ (Φ [2n,2n], Γ [2n])：z = [d, ḋ]（单个方向）一个 tick 后 z' = Φ·z + Γ·u（u 是该方向的动作）。"""
        n, k, gamma = (param(self.name, params, x) for x in ("n", "stiffness", "damping"))
        lap = 2*np.eye(n)-np.eye(n, k=1)-np.eye(n, k=-1)
        A = np.zeros((2*n+1, 2*n+1))
        A[:n, n:2*n] = np.eye(n); A[n:2*n, :n] = -k*lap; A[n:2*n, n:2*n] = -gamma*np.eye(n); A[n, 2*n] = self.K
        E = expm(A*PERIOD)
        return E[:2*n, :2*n], E[:2*n, 2*n]

    def tick(self, q, u, params):
        n = param(self.name, params, "n")
        Phi, Gam = self.matrices(params)
        pos = q[:2*n].reshape(n, 2); vel = q[2*n:].reshape(n, 2)
        rest = np.stack((self.rest(n), np.zeros(n)), -1)
        z = np.concatenate((pos-rest, vel))                  # [2n, 2]：两列是 x、y 两个方向
        z = Phi@z+Gam[:, None]*u[None, :]
        return np.concatenate(((z[:n]+rest).ravel(), z[n:].ravel()))

    def check(self, q, params):
        n = param(self.name, params, "n")
        if not np.isfinite(q).all() or np.abs(q[:2*n]).max() > self.BOUND:
            raise ValueError("spring_chain 质点离开了画面")

    def modes(self, params):
        """解析模态：(无阻尼角频率 ω_j [n], 有阻尼角频率 [n], 衰减率 γ/2)。"""
        n, k, gamma = (param(self.name, params, x) for x in ("n", "stiffness", "damping"))
        w = 2*math.sqrt(k)*np.sin(np.arange(1, n+1)*math.pi/(2*(n+1)))
        return w, np.sqrt(w**2-gamma**2/4), gamma/2

    def to_rc(self, p):
        return np.array([SIZE/2-self.SCALE*p[1], SIZE/2+self.SCALE*p[0]])

    def draw(self, states, trajectory_id, params, v, ticks=None):
        self.check_variant(v)
        n = param(self.name, params, "n"); sigma = v.sigma or self.SPOT_SIGMA
        image = _layers(self.name, trajectory_id, v, len(states), ticks, sigma)
        colors = np.concatenate((np.array([COLORS[v.color]]), OTHER_COLORS))[:n]
        for i, q in enumerate(states):
            pos = q[:2*n].reshape(n, 2)
            pts = [np.array([-self.ANCHOR, 0.])]+list(pos)+[np.array([self.ANCHOR, 0.])]
            img = image[i]
            for side in (-1., 1.):
                img = _box(img, self.to_rc((side*(self.ANCHOR+.06), 0.)), (.5*self.SCALE, .06*self.SCALE), BODY_COLOR)
            for a, b in zip(pts[:-1], pts[1:]):
                img = _segment(img, self.to_rc(a), self.to_rc(b), ROD_COLOR, .75)
            for p, color in zip(pos, colors):
                img = _spot(img, self.to_rc(p), color, sigma)
            image[i] = img
        return _finish(image, self.name, trajectory_id, v, ticks)


# ———————————————————————————— bouncing_balls ————————————————————————————

class BouncingBalls(SecondOrder):
    """方盒里的 n 个弹球（俯视、无重力）：单位质量、半径 R = 0.2，线性阻力，球与墙、球与球按恢复系数 e 碰撞；动作推第 0 个球。
      dv_i/dt = −β·v_i + [i = 0]·K·u，K = 1.5；墙在 |x|, |y| = W = 2（球心活动范围 |x|, |y| ≤ W − R）。
    两次碰撞之间按解析解推进（指数衰减到 K·u/β；βτ 小时用级数，β → 0 连续）。碰撞按事件处理：每 tick 分 32 个子步，
    子步内用连续碰撞检测找最早的接触时刻（S23a-fix1：间隙函数二阶导数有上界，分支定界，子步内进入又离开的擦边碰撞也能找到，
    不只看子步末端），推进到那一刻（取接触前一侧），施加冲量：墙 v_n ← −e·v_n；两球（等质量）
    沿连心线 Δ = −(1+e)·v_rel,n/2，v_i −= Δn、v_j += Δn；然后继续推进本子步剩下的时间。
    接近速度 < V_REST = 0.05 且被往里顶（动作给的法向相对加速度指向约束内，或者它的球正被别的静止接触顶着）时按静止接触
    处理：不反弹，法向相对速度置零，本子步不再对这个约束找事件，子步末尾把位置投影到不重叠；避免动作把球顶在墙上时无穷次
    碰撞（芝诺现象）。这是近似，只在球被慢慢顶住时出现；其余约束照常找事件、按 e 反弹（S23a-fix1：原来一出现静止接触就整段
    自由推进再全局投影，别的球的碰撞被吞掉）。没有往里顶的加速度时，再慢的碰撞也照常按 e 反弹（所以 e = 1、无动作、无阻力时
    动能精确守恒）。每个子步最多 MAX_EVENTS 次碰撞，超过时剩下的时间不再找事件，末尾投影（实测扫描里没有触发）。
    物理参数：n（默认 3，1…5）、restitution e（默认 0.9，[0.5, 1]）、drag β（默认 0.2 s⁻¹，[0, 1]）。
    初值（t = −1，流 0）：位置在 [−(W−R−0.05), W−R−0.05]² 里拒绝抽样（两两间距 ≥ 2R + 0.05），速度 ∈ U(−1.5, 1.5)²。
    画面（每单位 48 像素）：浅灰方框墙（内沿就是 W），球是抗锯齿实心圆盘（半径 9.6 像素；第 0 个调节项颜色，其余青、品红、黄、绿）。
    状态：位置 [n,2] 展平后接速度 [n,2]，共 4n 维。
    """
    name = "bouncing_balls"
    ALLOW_SIGMA = False
    SCALE = 48.
    W = 2.
    R = .2
    K = 1.5
    SUBSTEPS = 32
    EPS = 1e-14             # 找事件的阈值容差（墙：长度；球对：长度²）
    MAX_NODES = 100_000     # 连续碰撞检测每个约束最多展开的区间数（实测远小于此，超了直接报错）
    V_REST = .05
    MAX_EVENTS = 20

    def initial(self, trajectory_id, params):
        n = param(self.name, params, "n"); g = rng(self.name, trajectory_id, 0)
        lim = self.W-self.R-.05; pos = []
        while len(pos) < n:
            p = g.uniform(-lim, lim, 2)
            if all(np.hypot(*(p-o)) >= 2*self.R+.05 for o in pos):
                pos.append(p)
        vel = g.uniform(-1.5, 1.5, (n, 2))
        return np.concatenate((np.array(pos).ravel(), vel.ravel()))

    # F(τ)/τ² = Σ_{k≥0} (−z)^k/(k+2)!（z = βτ）的系数，z < 0.5 时 20 项到机器精度
    _F_SERIES = tuple(1./math.factorial(k+2) for k in range(20))

    @classmethod
    def _phi(cls, beta, tau):
        """E(τ) = (1 − e^{−βτ})/β、F(τ) = (τ − E(τ))/β（= ∫₀^τ E）。z = βτ < 0.5 时 F 用级数 τ²·Σ(−z)^k/(k+2)!，
        避免 τ − E 相近数相减（S23a-fix1：β 极小时原式把加速度位移消掉，β = 1e−16 少约 3%）。β = 0 的情形由 flow 单独处理。"""
        z = beta*tau
        e1 = -math.expm1(-z)/beta
        if z < .5:
            s = 0.
            for c in reversed(cls._F_SERIES):
                s = s*(-z)+c
            return e1, tau*tau*s
        return e1, (tau-e1)/beta

    @classmethod
    def flow(cls, pos, vel, acc, beta, tau):
        """常加速度 acc [n,2]、线性阻力 β 下推进 τ 秒（解析解）：p(τ) = p + v·E + a·F，v(τ) = v·e^{−βτ} + a·E。
        β = 0 时退化为匀加速。"""
        if beta == 0.:
            return pos+vel*tau+.5*acc*tau*tau, vel+acc*tau
        e1, f = cls._phi(beta, tau)
        return pos+vel*e1+acc*f, vel*math.exp(-beta*tau)+acc*e1

    def _gaps(self, pos):
        """所有约束的间隙（≥ 0 不重叠）：4n 个墙（x+、x−、y+、y− 顺序）+ n(n−1)/2 个球对。"""
        lim = self.W-self.R
        walls = np.stack((lim-pos[:, 0], lim+pos[:, 0], lim-pos[:, 1], lim+pos[:, 1]), -1).ravel()
        n = len(pos); i, j = np.triu_indices(n, 1)
        pairs = np.hypot(*(pos[j]-pos[i]).T)-2*self.R if n > 1 else np.zeros(0)
        return np.concatenate((walls, pairs)), (i, j)

    def _sgaps(self, pos):
        """找事件用的间隙函数 S：墙同 _gaps（位置的一次式），球对用 |d|² − (2R)²（与 |d| − 2R 同号、同单调，
        但对时间的二阶导数有简单上界，见 _curvature）。"""
        lim = self.W-self.R
        walls = np.stack((lim-pos[:, 0], lim+pos[:, 0], lim-pos[:, 1], lim+pos[:, 1]), -1).ravel()
        n = len(pos); i, j = np.triu_indices(n, 1)
        d = pos[j]-pos[i]
        return np.concatenate((walls, (d*d).sum(-1)-(2*self.R)**2))

    def _curvature(self, pos, vel, acc, beta, T):
        """[0, T] 上每个约束的 S''(t) 的上界 M（自由推进，没有碰撞）。
        每个球 p'(t) = v·e^{−βt} + a·E(t)，p''(t) = (a − β·v)·e^{−βt}：墙 |S''| ≤ |a − β·v|（对应分量）。
        球对 d = p_j − p_i：|d'| ≤ |Δv| + |Δa|·E(T) =: V，|d''| ≤ |Δa − β·Δv| =: A，|d| ≤ |d₀| + V·T，
        S'' = 2|d'|² + 2·d·d'' ≤ 2V² + 2(|d₀| + V·T)·A。"""
        e1 = T if beta == 0. else self._phi(beta, T)[0]
        m = np.abs(acc-beta*vel)
        walls = np.stack((m[:, 0], m[:, 0], m[:, 1], m[:, 1]), -1).ravel()
        n = len(pos); i, j = np.triu_indices(n, 1)
        dv, da = vel[j]-vel[i], acc[j]-acc[i]
        V = np.hypot(*dv.T)+np.hypot(*da.T)*e1 if n > 1 else np.zeros(0)
        A = np.hypot(*(da-beta*dv).T) if n > 1 else np.zeros(0)
        d0 = np.hypot(*(pos[j]-pos[i]).T) if n > 1 else np.zeros(0)
        return np.concatenate((walls, 2*V*V+2*(d0+V*T)*A))

    def _first_contact(self, pos, vel, acc, beta, T, skip):
        """在 [0, T] 上找最早穿过阈值的约束（连续碰撞检测，S23a-fix1）→ (约束号, 时刻) 或 (None, None)。
        约束 c 的阈值 thr = min(S_c(0), 0) − EPS：起点已经有极小重叠（投影的容差）的约束，只在重叠进一步加深时才算事件。
        分支定界：S'' ≤ M 时，区间 [a, b] 上 S ≥ min(S(a), S(b)) − M(b−a)²/8；这个下界 ≥ thr 的区间里一定没有穿过，
        其余对半分，直到区间宽 T·2⁻⁵⁰，返回最早一个右端点 < thr 的叶子的左端点（接触前一侧）。
        所以子步内进入又离开的擦边碰撞也能找到，不只看子步末端。skip 里的约束（本子步已按静止接触处理）不找。"""
        s0 = self._sgaps(pos)
        sT = self._sgaps(self.flow(pos, vel, acc, beta, T)[0])
        thr = np.minimum(s0, 0.)-self.EPS
        M = self._curvature(pos, vel, acc, beta, T)
        lower = np.minimum(s0, sT)-thr-M*T*T/8
        best, best_t = None, None
        for c in np.flatnonzero(lower < 0.):
            if c in skip:
                continue
            t = self._crossing(c, pos, vel, acc, beta, T, thr[c], M[c], s0[c], sT[c], T if best_t is None else best_t)
            if t is not None and (best_t is None or t < best_t):
                best, best_t = int(c), t
        return best, best_t

    def _crossing(self, c, pos, vel, acc, beta, T, thr, M, s0, sT, limit):
        """约束 c 在 [0, min(T, limit)) 里第一次 S < thr 的时刻（叶子左端点），没有则 None。S 用与 _first_contact 同一套式子算。"""
        leaf = T*2.**-50
        stack = [(0., T, s0, sT)]; nodes = 0
        while stack:
            a, b, sa, sb = stack.pop()
            if a >= limit or min(sa, sb)-thr-M*(b-a)*(b-a)/8 >= 0.:
                continue
            if b-a <= leaf:
                if sb < thr:
                    return a
                continue
            nodes += 1
            if nodes > self.MAX_NODES:
                raise RuntimeError("bouncing_balls 连续碰撞检测没有收敛（分支定界节点超限）")
            m = .5*(a+b)
            sm = self._sgaps(self.flow(pos, vel, acc, beta, m)[0])[c]
            stack.append((m, b, sm, sb)); stack.append((a, m, sa, sm))          # 先看左半
        return None

    def _normal(self, c, pos, pairs):
        """约束 c 的法向：墙 → (球号, 指向墙内的单位法向)；球对 → (i, j, 从 i 指向 j 的单位法向)。"""
        n = len(pos)
        if c < 4*n:
            ball, side = divmod(c, 4)
            normal = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])[side]
            return ball, None, normal
        i, j = pairs[0][c-4*n], pairs[1][c-4*n]
        d = pos[j]-pos[i]
        return i, j, d/np.hypot(*d)

    def _balls(self, c, n, pairs):
        if c < 4*n:
            return {c//4}
        return {int(pairs[0][c-4*n]), int(pairs[1][c-4*n])}

    def _approach(self, c, pos, x, pairs):
        """x 取速度时是法向接近速度，取加速度时是法向接近加速度（> 0 表示往重叠的方向）。"""
        i, j, nrm = self._normal(c, pos, pairs)
        if j is None:
            return -x[i] @ nrm
        return -(x[j]-x[i]) @ nrm

    def _impulse(self, c, pos, vel, pairs, e):
        i, j, nrm = self._normal(c, pos, pairs)
        if j is None:
            vn = vel[i] @ nrm
            if vn < 0.:
                vel[i] = vel[i]-(1+e)*vn*nrm
            return
        vn = (vel[j]-vel[i]) @ nrm
        if vn < 0.:
            delta = -(1+e)*vn/2
            vel[i] = vel[i]-delta*nrm; vel[j] = vel[j]+delta*nrm

    def _settle(self, c, pos, vel, pairs):
        """把单个约束 c 的重叠推回边界、去掉法向接近速度（完全非弹性）。"""
        lim = self.W-self.R
        i, j, nrm = self._normal(c, pos, pairs)
        if j is None:
            axis = 0 if c % 4 < 2 else 1
            pos[i, axis] = np.clip(pos[i, axis], -lim, lim)
            vel[i] = vel[i]-min(vel[i] @ nrm, 0.)*nrm
            return
        d = pos[j]-pos[i]; dist = np.hypot(*d)
        if dist < 2*self.R:
            shift = (2*self.R-dist)/2*nrm
            pos[i] = pos[i]-shift; pos[j] = pos[j]+shift
        vn = (vel[j]-vel[i]) @ nrm
        if vn < 0.:
            vel[i] = vel[i]+vn/2*nrm; vel[j] = vel[j]-vn/2*nrm

    def _project(self, pos, vel):
        """静止接触：把重叠的位置推回边界、去掉法向接近速度（最多 50 轮 Gauss–Seidel）。墙是硬约束：最后再把球心
        截回 [−(W−R), W−R]，所以球被另一个球顶在墙上（卡住）时只会留下极小的球间重叠（实测 < 1e−5），不会穿墙。"""
        lim = self.W-self.R
        for _ in range(50):
            gaps, pairs = self._gaps(pos)
            if gaps.min() >= -1e-12:
                return
            for c in np.flatnonzero(gaps < 0.):
                self._settle(c, pos, vel, pairs)
        for axis, side in ((0, 1.), (0, -1.), (1, 1.), (1, -1.)):
            out = side*pos[:, axis] > lim
            pos[out, axis] = side*lim
            vel[out, axis] = np.where(side*vel[out, axis] > 0., 0., vel[out, axis])

    def substep(self, pos, vel, acc, beta, e, h):
        """推进一个子步 h（事件驱动），原地修改 pos、vel；返回本子步的碰撞事件数。
        S23a-fix1：静止接触只约束它自己——记进 resting，本子步不再对它找事件（它的球在剩下的时间里会稍微压进去，
        子步末尾投影），其余约束照常找事件、按 e 反弹（原来一出现静止接触就把整个剩余子步自由推进再全局投影，
        别的球在这段时间里的碰撞被投影吞掉、法向速度清零）。之后若有真正的碰撞（按 e 反弹）涉及 resting 里某个约束的球，
        先把那个约束单独推回、去掉法向接近速度，再从 resting 里拿掉，让它重新参与找事件（例如别的球把贴墙的球撞向墙）。"""
        remaining, events, resting = h, 0, set()
        n = len(pos)
        self._project(pos, vel)
        while remaining > 0.:
            if events >= self.MAX_EVENTS:
                pos[:], vel[:] = self.flow(pos, vel, acc, beta, remaining); self._project(pos, vel)
                return events
            best, best_t = self._first_contact(pos, vel, acc, beta, remaining, resting)
            if best is None:
                pos[:], vel[:] = self.flow(pos, vel, acc, beta, remaining)
                break
            pc, vc = self.flow(pos, vel, acc, beta, best_t)
            pos[:], vel[:] = pc, vc; remaining -= best_t; events += 1
            pairs = np.triu_indices(n, 1)
            balls = self._balls(best, n, pairs)
            held = any(self._balls(r, n, pairs) & balls for r in resting)
            if self._approach(best, pos, vel, pairs) < self.V_REST and (self._approach(best, pos, acc, pairs) > 0. or held):
                # 慢慢顶住（动作往里顶，或者它的球正被别的静止接触顶着，例如第 0 个球把另一个球推在墙上）：
                # 静止接触（不反弹），本子步不再对它找事件
                self._impulse(best, pos, vel, pairs, 0.)
                resting.add(best)
                continue
            for r in sorted(resting):
                if self._balls(r, n, pairs) & balls:
                    resting.discard(r); self._settle(r, pos, vel, pairs)
            self._impulse(best, pos, vel, pairs, e)
        if resting:
            self._project(pos, vel)
        return events

    def tick(self, q, u, params, substeps=None):
        n, e, beta = (param(self.name, params, x) for x in ("n", "restitution", "drag"))
        pos = q[:2*n].reshape(n, 2).copy(); vel = q[2*n:].reshape(n, 2).copy()
        acc = np.zeros((n, 2)); acc[0] = self.K*u
        m = self.SUBSTEPS if substeps is None else substeps
        for _ in range(m):
            self.substep(pos, vel, acc, beta, e, PERIOD/m)
        return np.concatenate((pos.ravel(), vel.ravel()))

    def check(self, q, params):
        n = param(self.name, params, "n")
        if not np.isfinite(q).all() or np.abs(q[:2*n]).max() > self.W-self.R+1e-9:
            raise ValueError("bouncing_balls 球穿出了墙")

    def to_rc(self, p):
        return np.array([SIZE/2-self.SCALE*p[1], SIZE/2+self.SCALE*p[0]])

    def draw(self, states, trajectory_id, params, v, ticks=None):
        self.check_variant(v)
        n = param(self.name, params, "n")
        image = _layers(self.name, trajectory_id, v, len(states), ticks, self.SPOT_SIGMA)
        colors = np.concatenate((np.array([COLORS[v.color]]), OTHER_COLORS))[:n]
        a, b = SIZE/2-self.SCALE*self.W, SIZE/2+self.SCALE*self.W        # 墙内沿（像素）
        for i, q in enumerate(states):
            img = image[i]
            for p0, p1 in (((a-2, a-2), (a-2, b+2)), ((b+2, a-2), (b+2, b+2)), ((a-2, a-2), (b+2, a-2)), ((a-2, b+2), (b+2, b+2))):
                img = _segment(img, p0, p1, BODY_COLOR, 1.5)
            for p, color in zip(q[:2*n].reshape(n, 2), colors):
                img = _disc(img, self.to_rc(p), color, self.R*self.SCALE)
            image[i] = img
        return _finish(image, self.name, trajectory_id, v, ticks)


# ———————————————————————————— wave ————————————————————————————

class Wave(SecondOrder):
    """周期方域 [0,1)² 上的阻尼波动方程，动作移动一个“压膜”激振源（二阶场，线性）：
      u_tt = c²∇²u − γ·u_t + s，  s = load·c²·∇²B(x − p)，  dp/dt = SPEED·u_act（域长/秒，周期回绕；p = (x, y)，y 向上）
    B 是 σ = 0.03 域长、峰值 1 的周期高斯。源取这个形状是因为静止时的稳态正好是 u = −load·(B − B̄)：激振源把膜压出一个
    深 load 的高斯凹陷（和 c 无关），s 的空间平均为 0，所以均匀模不受力。源一动就拖出尾迹，速度超过 c 时出现马赫锥。
    时间推进：傅里叶空间里每个模是阻尼振子 ẍ + γẋ + ω²x = f_k（ω = c|k|，k 为角波数），每 tick 分 SUBSTEPS = 16 个子步，
    子步内源取子步中点的位置（相移 e^{−ik·p}，亚像素平移，二阶精度），对常数 f_k 用阻尼振子的精确传播子推进（k = 0 单独处理）。
    所有非零模都是欠阻尼（γ/2 < c·2π，由参数范围保证）。每个 tick 从实空间的场重新做 FFT，所以续算与整条轨迹逐位相同。
    格点 N = 128（每格 2 像素），空间离散是谱方法（拉普拉斯 = −|k|²）。
    物理参数：speed c（默认 0.15 域长/秒，[0.08, 0.3]）、damping γ（默认 0.1 s⁻¹）、load（默认 0.25，[0, 0.4]）。
    c 大时高波数模每帧转过的相位超过 π（ω·0.25 s > π ⇔ 每域长超过 2/c 个波长），两帧之间会混叠，这是有意测时间离散的设置。
    初值（t = −1，流 0）：3 个高斯团（中心均匀、σ ∈ U(0.04, 0.08)、幅值 ∈ ±U(0.2, 0.4)，符号随机）减去空间平均，
    再叠加激振源在 p0 的稳态凹陷 −load·(B − B̄)；u_t = 0；p0 均匀。
    画面：像素 = 背景 + u⁺·(调节项颜色 − 背景) + u⁻·(蓝 − 背景)（u⁺ = clip(u, 0, 1)，u⁻ = clip(−u, 0, 1)），线性不饱和，
    u 能从 RGB 解回；激振源画成品红细圆环（半径 5 像素）。u_t 由相邻两帧给出。
    状态：完整场打包 [u (N²), u_t (N²), p (2)]；隐藏状态（只供诊断）是 u、u_t 各 8×8 平均池化成 16×16 再接 p，共 514 维。
    """
    name = "wave"
    ALLOW_SIGMA = False
    ALLOW_SPOTS = False
    N = 128
    SPEED = .4
    SOURCE_SIGMA = .03
    SUBSTEPS = 16
    BLOBS = 3
    NEG_COLOR = np.array([70., 140., 230.])

    def __init__(self):
        n = self.N
        self.kr = 2*np.pi*np.fft.fftfreq(n, d=1./n)          # 行方向角波数（域长⁻¹）
        self.kc = 2*np.pi*np.fft.rfftfreq(n, d=1./n)
        self.k2 = self.kr[:, None]**2+self.kc[None, :]**2
        self.fr = np.fft.fftfreq(n)[:, None]; self.fc = np.fft.rfftfreq(n)[None, :]   # 每格的周数（相移用）
        c = (np.arange(n)+.5)/n
        d = (c-c[0]+.5) % 1.-.5                                 # 以第 0 格中心为原点的周期最小像距
        self.B0 = np.fft.rfft2(np.exp(-(d[:, None]**2+d[None, :]**2)/(2*self.SOURCE_SIGMA**2)))

    def source_hat(self, p):
        """B(x − p) 的 rfft2：B0 平移到 p 的格点坐标（列 = p_x·N − 0.5，行 = (1 − p_y)·N − 0.5）。"""
        col, row = p[0]*self.N-.5, (1.-p[1])*self.N-.5
        return self.B0*np.exp(-2j*np.pi*(self.fr*row+self.fc*col))

    def bump(self, p):
        return np.fft.irfft2(self.source_hat(p), s=(self.N, self.N))

    def pack(self, u, w, p):
        return np.concatenate((u.ravel(), w.ravel(), p))

    def unpack(self, q):
        n2 = self.N*self.N
        return q[:n2].reshape(self.N, self.N), q[n2:2*n2].reshape(self.N, self.N), q[2*n2:]

    def initial(self, trajectory_id, params):
        load = param(self.name, params, "load")
        g = rng(self.name, trajectory_id, 0)
        centers = g.uniform(0., 1., (self.BLOBS, 2)); sigmas = g.uniform(.04, .08, self.BLOBS)
        amps = g.uniform(.2, .4, self.BLOBS)*np.where(g.random(self.BLOBS) < .5, -1., 1.)
        p = g.uniform(0., 1., 2)
        c = (np.arange(self.N)+.5)/self.N
        u = np.zeros((self.N, self.N))
        for (cx, cy), s, a in zip(centers, sigmas, amps):
            dx = (c-cx+.5) % 1.-.5; dy = (c-(1.-cy)+.5) % 1.-.5      # 行 = 1 − y
            u += a*np.exp(-(dy[:, None]**2+dx[None, :]**2)/(2*s*s))
        B = self.bump(p)
        u = u-u.mean()-load*(B-B.mean())
        return self.pack(u, np.zeros((self.N, self.N)), p)

    def propagate(self, xh, vh, fh, dt, c, gamma):
        """傅里叶空间里每个模的阻尼振子 ẍ + γẋ + ω²x = f（f 在 dt 内不变）精确推进 dt。"""
        w2 = c*c*self.k2
        safe = np.where(w2 > 0, w2, 1.)
        xp = np.where(w2 > 0, fh/safe, 0.)                      # 特解（k = 0 时 f = 0）
        y = xh-xp
        wd = np.sqrt(np.maximum(w2-gamma*gamma/4, 0.))
        decay = math.exp(-gamma*dt/2)
        cos, sin = np.cos(wd*dt), np.sin(wd*dt)
        wd_safe = np.where(wd > 0, wd, 1.)
        y1 = decay*(y*cos+(vh+gamma*y/2)*np.where(wd > 0, sin/wd_safe, dt))
        v1 = decay*(vh*cos-(w2*y+gamma*vh/2)*np.where(wd > 0, sin/wd_safe, dt))
        # k = 0：ẍ = −γẋ（f_0 = 0）
        e1 = -math.expm1(-gamma*dt)/gamma if gamma > 0 else dt
        y1[0, 0] = y[0, 0]+vh[0, 0]*e1; v1[0, 0] = vh[0, 0]*math.exp(-gamma*dt)
        return y1+xp, v1

    def tick(self, q, u_act, params, substeps=None):
        c, gamma, load = (param(self.name, params, x) for x in ("speed", "damping", "load"))
        if gamma/2 >= c*2*np.pi:
            raise ValueError("wave 的最低非零模必须欠阻尼（γ/2 < 2πc）")
        u, w, p = self.unpack(q)
        m = self.SUBSTEPS if substeps is None else substeps
        dt = PERIOD/m; vel = self.SPEED*np.asarray(u_act, dtype=np.float64)
        xh, vh = np.fft.rfft2(u), np.fft.rfft2(w)
        for k in range(m):
            mid = p+vel*(k+.5)*dt
            fh = -load*c*c*self.k2*self.source_hat(mid)         # load·c²·∇²B 的变换 = −load·c²·|k|²·B̂
            xh, vh = self.propagate(xh, vh, fh, dt, c, gamma)
        p = (p+vel*PERIOD) % 1.
        shape = (self.N, self.N)
        return self.pack(np.fft.irfft2(xh, s=shape), np.fft.irfft2(vh, s=shape), p)

    def hidden(self, states, params=None):
        n = self.N
        L = states.shape[0]
        U = states[:, :n*n].reshape(L, 16, n//16, 16, n//16).mean((2, 4)).reshape(L, -1)
        W = states[:, n*n:2*n*n].reshape(L, 16, n//16, 16, n//16).mean((2, 4)).reshape(L, -1)
        return np.concatenate((U, W, states[:, 2*n*n:]), axis=1)

    def draw(self, states, trajectory_id, params, v, ticks=None):
        self.check_variant(v)
        n = self.N; rep = SIZE//n; L = len(states)
        u = states[:, :n*n].reshape(L, n, n)
        if systems._VIEW[0] is None:
            up = np.repeat(np.repeat(u, rep, axis=1), rep, axis=2)[..., None]
        else:      # S24c：相机里在视野像素中心上周期双线性插值
            from . import camera
            up = camera.sample_field(u, float(rep))[..., None]
        bg = _texture(v.background)[None]
        image = bg+np.clip(up, 0., 1.)*(np.array(COLORS[v.color])-bg)+np.clip(-up, 0., 1.)*(self.NEG_COLOR-bg)
        p = states[:, 2*n*n:]
        xr, xc = systems.pixel_axes()
        dc = (xc[None, :]-SIZE*p[:, 0, None]+SIZE/2) % SIZE-SIZE/2
        dr = (xr[None, :]-SIZE*(1.-p[:, 1, None])+SIZE/2) % SIZE-SIZE/2
        d = np.sqrt(dr[:, :, None]**2+dc[:, None, :]**2)
        if systems._VIEW[0] is None:
            wm = np.exp(-(d-5.)**2/2.)[..., None]
        else:
            from . import camera
            wm = camera.ring_weight(d, 5.)[..., None]
        image = image+(MARK_COLOR-image)*wm
        return _finish(image, self.name, trajectory_id, v, ticks)


SYSTEMS = {"pendulum": Pendulum(), "bouncing_balls": BouncingBalls(), "wave": Wave(), "cart_pole": CartPole(),
           "spring_chain": SpringChain()}
assert list(SYSTEMS) == list(SYSTEM_INDEX)
