"""新增场景（S20a）：gray_scott 反应扩散、push 准静态推方块。纯函数：物理真值 + 渲染。

S20b 登记：datasets.SYSTEMS 末尾追加两个名字，物理参数并入 variants.PARAMS，systems.REGISTRY 的
GrayScottSystem、PushSystem 只是把 datasets.active() 的调节项、systems.commanded_controls 和 applied_ticks
接到这里的函数上（在方法里才导入本文件，v7 的 5 个场景不经过这里）。

共同约定（与 v7 相同）：
- 控制周期 0.25 s，32 个 tick、33 帧；输入 u 是“执行”动作 [32,2] float64（已含增益、延迟、执行噪声）。
- 画面 256×256×3 uint8；世界坐标中心列 = 128 + 48x，中心行 = 128 − 48y（y 向上）。
- 一阶系统：画面包含决定未来的全部状态（模型只看当前一帧）。
- 随机流：熵 [1703, 1000+SYSTEM_INDEX[name], 轨迹编号, 流号]。SYSTEM_INDEX 取 S20b 把两个名字追加到
  datasets.SYSTEMS 末尾后的位置（5、6），所以登记后 simulation.seed_entropy 给出同一个熵，v7 的 5 个场景不变。
  流 0 = 初值，流 3 = 干扰光点位置，流 4 = 像素噪声（与 systems.compose 一致）。
"""
import math

import numpy as np

from . import variants
from .systems import COLORS, SIZE, _texture

TICKS = 32
PERIOD = .25
SYSTEM_INDEX = {"gray_scott": 5, "push": 6}

# 场景物理参数：名字 → (默认值, 允许范围)。S20b 起唯一来源是 variants.PARAMS（调节项解析用同一份），这里只取这两个场景
PARAMS = {name: variants.PARAMS[name] for name in SYSTEM_INDEX}


def rng(system: str, trajectory_id: int, stream: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [1703, 1000+SYSTEM_INDEX[system], int(trajectory_id), int(stream)])))


def param(system: str, params: dict, name: str) -> float:
    default, (lo, hi) = PARAMS[system][name]
    value = params.get(name, default)
    if not (math.isfinite(value) and lo <= value <= hi):
        raise ValueError(f"{system} 的 {name} 必须在 [{lo},{hi}]")
    return float(value)


def _check_controls(u):
    u = np.asarray(u, dtype=np.float64)
    if u.shape != (TICKS, 2) or not np.isfinite(u).all():
        raise ValueError("执行动作必须是有限 float64[32,2]")
    return u


def _finish(image, trajectory_id, system, v):
    if v.pixel_noise:
        image = image+rng(system, trajectory_id, 4).normal(0., v.pixel_noise, image.shape)
    return np.rint(np.clip(image, 0., 255.)).astype(np.uint8)


# ———————————————————————————— gray_scott ————————————————————————————

class GrayScott:
    """周期方域上的 Gray–Scott 反应扩散，动作移动一个注入点（一阶场，非线性）。

    ∂U/∂t = D_u∇²U − UV² + F(1−U)
    ∂V/∂t = D_v∇²V + UV² − (F+k)V + a·G(x−p)·(1−V)
    dp/dt = SPEED·u（域长/秒，周期回绕；p=(x, y)，y 向上）

    无量纲时间：每个控制 tick（0.25 s）推进 UNITS_PER_TICK=20 个时间单位；格点 N=128（每格 2 像素），
    D_u=0.16、D_v=0.08（格点²/时间单位，Pearson 1993 的取值）。G 是以 p 为中心、σ=2 格的周期高斯。
    初值：U=1、V=0，4 个 10×10 格的方块 U=0.5、V=0.25，加 ±0.01 均匀噪声（流 0）；注入点 p0 均匀；
    然后注入点不动烧入 BURN_IN=1000 个时间单位，让斑图成形，烧入后的状态是第 0 帧。
    时间推进：Strang 分裂，扩散半步（傅里叶精确）→ 反应 RK4 一整步（注入点取步中点位置）→ 扩散半步，DT=0.5。
    画面：像素 = 背景 + 0.6·V·(调节项颜色 − 背景) + 0.4·(1−U)·(青色 − 背景)，线性不饱和，(U, V) 可从 RGB 解回；
    注入点画成品红色细圆环（半径 5 像素），这样 U、V、p 都在画面里（圆环下约 60 个像素被盖住）。
    谱方法有约 1e−5 的负浓度下冲（不 clamp 物理场，只在画面上 clip）。
    隐藏状态（只供诊断）：U、V 各 8×8 平均池化成 16×16，展平后接 p，共 514 维。
    """
    name = "gray_scott"
    state_dim = 514
    N = 128
    DU, DV = .16, .08
    DT = .5
    UNITS_PER_TICK = 20.
    BURN_IN = 1000.
    SPEED = .4
    SIGMA_CELLS = 2.
    SEEDS = 4
    COLOR_U = np.array([70., 200., 190.])
    COLOR_MARK = np.array([230., 70., 200.])
    MIX_V, MIX_U = .6, .4

    def initial(self, trajectory_id):
        g = rng(self.name, trajectory_id, 0)
        U = np.ones((self.N, self.N)); V = np.zeros((self.N, self.N))
        for r, c in g.integers(0, self.N, (self.SEEDS, 2)):
            rows = (np.arange(r, r+10) % self.N)[:, None]; cols = (np.arange(c, c+10) % self.N)[None, :]
            U[rows, cols] = .5; V[rows, cols] = .25
        U += g.uniform(-.01, .01, U.shape); V += g.uniform(-.01, .01, V.shape)
        p = g.uniform(0., 1., 2)
        return U, V, p

    def _gauss(self, p):
        """周期最小像距的高斯 [N,N]；p=(x, y 向上)，行 = (1−y)。"""
        x = (np.arange(self.N)+.5)/self.N
        dx = (x-p[0]+.5) % 1.-.5
        dy = (x-(1.-p[1])+.5) % 1.-.5
        d2 = (dy[:, None]**2+dx[None, :]**2)*self.N**2
        return np.exp(-d2/(2*self.SIGMA_CELLS**2))

    def _run(self, U, V, p, velocity, units, dt, F, k, a, out=None, every=None):
        """推进 units 个时间单位，注入点速度 velocity（域长/时间单位）；every 步记录一次。"""
        steps = int(round(units/dt))
        if abs(steps*dt-units) > 1e-9:
            raise ValueError("时间单位必须是 dt 的整数倍")
        kk = 2*np.pi*np.fft.fftfreq(self.N)
        k2 = kk[:, None]**2+(2*np.pi*np.fft.rfftfreq(self.N))[None, :]**2
        eu = np.exp(-self.DU*k2*dt/2); ev = np.exp(-self.DV*k2*dt/2)
        half = lambda X, e: np.fft.irfft2(np.fft.rfft2(X)*e, s=X.shape)

        def react(U, V, G):
            uvv = U*V*V
            return -uvv+F*(1.-U), uvv-(F+k)*V+a*G*(1.-V)

        for _ in range(steps):
            U = half(U, eu); V = half(V, ev)
            G = self._gauss(p+velocity*dt/2)
            k1 = react(U, V, G); k2 = react(U+dt/2*k1[0], V+dt/2*k1[1], G)
            k3 = react(U+dt/2*k2[0], V+dt/2*k2[1], G); k4 = react(U+dt*k3[0], V+dt*k3[1], G)
            U = U+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6; V = V+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
            U = half(U, eu); V = half(V, ev)
            p = (p+velocity*dt) % 1.
        if not (np.isfinite(U).all() and np.isfinite(V).all()):
            raise ValueError("gray_scott 数值发散")
        return U, V, p

    def fields(self, trajectory_id, u, params, *, dt=None, burn_in=None):
        """→ (U [33,N,N], V [33,N,N], p [33,2])。u 是执行动作 [32,2]。"""
        u = _check_controls(u); dt = self.DT if dt is None else dt
        F, k, a = (param(self.name, params, n) for n in ("feed", "kill", "inject"))
        U, V, p = self.initial(trajectory_id)
        U, V, p = self._run(U, V, p, np.zeros(2), self.BURN_IN if burn_in is None else burn_in, dt, F, k, a)
        Us, Vs, ps = [U], [V], [p]
        for tick in range(TICKS):
            velocity = self.SPEED*u[tick]*PERIOD/self.UNITS_PER_TICK    # 域长/时间单位
            U, V, p = self._run(U, V, p, velocity, self.UNITS_PER_TICK, dt, F, k, a)
            Us.append(U); Vs.append(V); ps.append(p)
        return np.stack(Us), np.stack(Vs), np.stack(ps)

    def states(self, U, V, p):
        pool = lambda X: X.reshape(X.shape[0], 16, self.N//16, 16, self.N//16).mean((2, 4)).reshape(X.shape[0], -1)
        return np.concatenate((pool(U), pool(V), p), axis=1)

    def draw(self, U, V, p, trajectory_id, v):
        rep = SIZE//self.N
        up = lambda X: np.repeat(np.repeat(X, rep, axis=1), rep, axis=2)
        # 线性、不饱和：V 和 1−U 在物理域 [0,1] 内各占一个不平行的颜色方向，叠加后最大约 221 < 255，
        # 所以 (U, V) 能从 RGB 解回（只差 8 位量化）。clip 只挡谱方法约 1e−5 的数值下冲。
        wv = np.clip(up(V), 0., 1.)[..., None]; wu = np.clip(1.-up(U), 0., 1.)[..., None]
        bg = _texture(v.background)[None]
        image = bg+self.MIX_V*(np.array(COLORS[v.color])-bg)*wv+self.MIX_U*(self.COLOR_U-bg)*wu
        x = np.arange(SIZE)+.5
        dc = (x[None, :]-SIZE*p[:, 0, None]+SIZE/2) % SIZE-SIZE/2            # [L,256]
        dr = (x[None, :]-SIZE*(1.-p[:, 1, None])+SIZE/2) % SIZE-SIZE/2
        d = np.sqrt(dr[:, :, None]**2+dc[:, None, :]**2)
        wm = np.exp(-(d-5.)**2/2.)[..., None]
        image = image+(self.COLOR_MARK-image)*wm
        return _finish(image, trajectory_id, self.name, v)

    def frames(self, trajectory_id, u, params, v):
        """→ (隐藏状态 [33,514], 画面 [33,256,256,3] uint8)。"""
        if v.sigma is not None or v.distractors:
            raise ValueError("gray_scott 没有光点：sigma/distractors 不适用")
        U, V, p = self.fields(trajectory_id, u, params)
        return self.states(U, V, p), self.draw(U, V, p, trajectory_id, v)


# ———————————————————————————— push ————————————————————————————

class Push:
    """准静态平面推方块（一阶，刚体接触 + 库仑静摩擦阈值）。无惯性：净外力落入静摩擦阈值内方块就静止；
    停止推动后接触力按 1/(STIFF·迁移率) ≈ 3 ms 的时间常数松弛，方块在这段内还会滑约 1e−3，之后静止（坡面自滑除外）。

    状态 q = (c_x, c_y, b_x, b_y, θ)：推杆中心 c、方块中心 b 与朝向 θ（不回绕）。全部过阻尼：
    - 推杆：dc/dt = SPEED·u − f/ETA_P + 墙；推杆是半径 RP 的圆盘，被方块顶住时会停（推力有限）。
    - 接触：推杆圆盘与方块（边长 size 的正方形）的穿透深度 δ>0 时，方块受 f = STIFF·δ·n（n 从推杆指向方块），
      作用点是方块边界上离推杆中心最近的点 r_c，力矩 τ = (r_c − b) × f。
    - 坡面：重力切向分量 W·slope·(cos φ, sin φ)，法向力 N = W·√(1−slope²)，φ = slope_dir（度）。
    - 方块：合力 F = f + 坡面力 + 墙力，力矩 τ；椭圆极限面范数 ‖w‖ = √(F_x²+F_y²+(τ/ρ)²)，
      ρ = 0.383·size（均匀压力正方形的摩擦力矩/摩擦力之比）。s = max(0, 1 − μ_s·N/‖w‖)，
      db/dt = s·F/ETA_B，dθ/dt = s·τ/(ρ²·ETA_B)。‖w‖ ≤ μ_s·N 时方块不动（静摩擦），坡度 > μ_s 时自己滑。
    积分：分段常数动作下 RK4，每 tick 256 步（刚度 × 迁移率 × 步长 < 1，见测试）。
    画面：方块浅灰实心（抗锯齿），方块体坐标 (+0.25·size, 0) 处一个深色标记点（区分朝向），推杆是调节项颜色的光点。
    软墙：推杆 |c| > 2.2、方块 |b| > 1.6 后受 20·(|q|−阈值) 的回推。
    """
    name = "push"
    state_dim = 5
    SCALE = 48.
    BOUND = 2.6
    RP = .15
    STIFF = 150.
    SPEED = 2.
    ETA_P = ETA_B = W = 1.
    STEPS_PER_TICK = 256
    WALL_P, WALL_B = 2.2, 1.6
    BLOCK = np.array([190., 195., 205.])
    MARK = np.array([60., 60., 72.])

    def initial(self, trajectory_id, params):
        g = rng(self.name, trajectory_id, 0)
        b = g.uniform(-1., 1., 2); theta = g.uniform(-np.pi, np.pi)
        psi = g.uniform(-np.pi, np.pi); extra = g.uniform(.05, .6)
        r = param(self.name, params, "size")/2*math.sqrt(2)+self.RP+extra     # 半对角线外，任何朝向都不重叠
        c = b+r*np.array([math.cos(psi), math.sin(psi)])
        return np.array([c[0], c[1], b[0], b[1], theta])

    def contact(self, q, size):
        """→ (作用在方块上的力 f [2], 力矩 τ, 穿透深度 δ)。"""
        c, b, th = q[:2], q[2:4], q[4]
        co, si = math.cos(th), math.sin(th)
        d = np.array([co*(c[0]-b[0])+si*(c[1]-b[1]), -si*(c[0]-b[0])+co*(c[1]-b[1])])   # 体坐标
        h = size/2
        if np.abs(d).max() > h:          # 推杆中心在方块外
            closest = np.clip(d, -h, h); gap = d-closest; dist = math.hypot(*gap)
            if dist >= self.RP:
                return np.zeros(2), 0., 0.
            n_out = gap/dist; delta = self.RP-dist
        else:                            # 推杆中心已进到方块里（只在极端增益下出现）
            i = int(np.argmax(np.abs(d))); n_out = np.zeros(2); n_out[i] = 1. if d[i] >= 0 else -1.
            closest = d.copy(); closest[i] = h*n_out[i]; delta = self.RP+h-abs(d[i])
        f_body = -self.STIFF*delta*n_out
        tau = closest[0]*f_body[1]-closest[1]*f_body[0]
        f = np.array([co*f_body[0]-si*f_body[1], si*f_body[0]+co*f_body[1]])
        return f, tau, delta

    @staticmethod
    def _wall(x, limit):
        return -20.*np.sign(x)*np.maximum(np.abs(x)-limit, 0.)

    def field(self, q, u, params):
        size, mu, slope = (param(self.name, params, n) for n in ("size", "mu_s", "slope"))
        phi = math.radians(param(self.name, params, "slope_dir"))
        f, tau, _ = self.contact(q, size)
        dc = self.SPEED*u-f/self.ETA_P+self._wall(q[:2], self.WALL_P)
        F = f+self.W*slope*np.array([math.cos(phi), math.sin(phi)])+self._wall(q[2:4], self.WALL_B)
        rho = .383*size
        norm = math.sqrt(F[0]**2+F[1]**2+(tau/rho)**2)
        s = max(0., 1.-mu*self.W*math.sqrt(1.-slope**2)/norm) if norm > 0 else 0.
        return np.array([dc[0], dc[1], s*F[0]/self.ETA_B, s*F[1]/self.ETA_B, s*tau/(rho**2*self.ETA_B)])

    def trajectory(self, trajectory_id, u, params, *, steps_per_tick=None, state=None):
        """→ 隐藏状态 [33,5]。"""
        u = _check_controls(u); n = self.STEPS_PER_TICK if steps_per_tick is None else steps_per_tick
        q = (self.initial(trajectory_id, params) if state is None else np.asarray(state, dtype=np.float64)).copy()
        out = [q.copy()]; h = PERIOD/n
        for tick in range(TICKS):
            for _ in range(n):
                k1 = self.field(q, u[tick], params); k2 = self.field(q+h*k1/2, u[tick], params)
                k3 = self.field(q+h*k2/2, u[tick], params); k4 = self.field(q+h*k3, u[tick], params)
                q = q+h*(k1+2*k2+2*k3+k4)/6
            if not np.isfinite(q).all() or np.abs(q[:4]).max() > self.BOUND:
                raise ValueError(f"push state left the rendered domain (trajectory {trajectory_id})")
            out.append(q.copy())
        return np.stack(out)

    def draw(self, states, trajectory_id, params, v):
        size = param(self.name, params, "size")
        sigma = v.sigma or 6.
        x = (np.arange(SIZE)+.5-128)/self.SCALE
        X, Y = np.meshgrid(x, -x)                      # 世界坐标，行向下 = y 减小
        base = _texture(v.background).astype(np.float64)
        row, col = np.meshgrid(np.arange(SIZE)+.5, np.arange(SIZE)+.5, indexing="ij")
        if v.distractors:
            for r, c in rng(self.name, trajectory_id, 3).uniform(24., SIZE-24., (v.distractors, 2)):
                w = np.exp(-((col-c)**2+(row-r)**2)/(2*sigma**2))[..., None]
                base = base+(np.array([150., 150., 150.])-base)*w
        frames = []
        for cx, cy, bx, by, th in states:
            co, si = math.cos(th), math.sin(th)
            dx, dy = X-bx, Y-by
            ex = np.abs(co*dx+si*dy)-size/2; ey = np.abs(-si*dx+co*dy)-size/2
            sd = np.hypot(np.maximum(ex, 0.), np.maximum(ey, 0.))+np.minimum(np.maximum(ex, ey), 0.)
            a = np.clip(.5-sd*self.SCALE, 0., 1.)[..., None]
            image = base+(self.BLOCK-base)*a
            mx, my = bx+.25*size*co, by+.25*size*si
            wm = np.exp(-((X-mx)**2+(Y-my)**2)*self.SCALE**2/(2*(.06*size*self.SCALE)**2))[..., None]
            image = image+(self.MARK-image)*wm
            wp = np.exp(-((X-cx)**2+(Y-cy)**2)*self.SCALE**2/(2*sigma**2))[..., None]
            image = image+(np.array(COLORS[v.color])-image)*wp
            frames.append(image)
        return _finish(np.stack(frames), trajectory_id, self.name, v)

    def frames(self, trajectory_id, u, params, v):
        states = self.trajectory(trajectory_id, u, params)
        return states, self.draw(states, trajectory_id, params, v)


SYSTEMS = {"gray_scott": GrayScott(), "push": Push()}
