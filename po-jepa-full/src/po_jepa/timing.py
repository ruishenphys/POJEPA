"""S24b 时间采样：帧率 fps、帧时间抖动 jitter、曝光 exposure、控制在两帧之间切换 switch_offset（docs/segments/S24b.md）。

v7 的数据是“每个控制 tick（0.25 s）一帧”。这四个调节项要取 tick 内部的状态，做法统一为“主链 + 旁支”：

- 主链：按 tick 推进物理，与原来的实现逐位相同（每个场景的整 tick 就调用它原来的那一步：second_order 的 tick()、
  gray_scott 的 _run，或照抄的同一个循环，测试逐位核）。switch_offset 让执行动作在 tick 内部换值，这时这个 tick 在换值处
  分成几段，每段按场景自己的子步网格推进（见下）。主链只由执行动作决定，和要取哪些时刻无关。
- 旁支：时刻 τ = k + φ（第 k 个 tick 内部）的状态，从主链上 τ 之前最近的点（tick 起点或换值点）出发，沿该场景生产用的
  子步网格走满整子步，最后不足一个子步的部分用同一个格式走一个短步（步长 = 剩下的那一截）。所以：
  · 落在子步网格上的时刻就是生产积分的中间状态（逐位）；
  · 同一时刻的状态与 fps、抖动、曝光取了哪些别的时刻无关（帧率 8 的第 1 帧与帧率 16 的第 2 帧逐位相同）；
  · 整 tick 时刻的画面与 fps=4 的数据集逐位相同（帧率 2、8、16 是同一条轨迹的子采样或加密）。
  精确解的场景（advection_diffusion 的傅里叶传播子、spring_chain 的矩阵指数）直接按时长 φ·0.25 s 推进，没有子步。
  过程噪声（S21a，粒子类 ODE）：整子步用流 6 的同一个增量；短步取该子步布朗增量的线性插值（布朗桥的条件均值）。

帧时刻（单位 tick；秒 = tick × 0.25）：t_k = k·Δ，Δ = 4/fps，k = 0…N，N = horizon·fps/4。jitter > 0 时中间各帧
t_k += jitter·Δ·ξ_k，ξ_k ~ U(−1,1)（流 10，按帧号顺序抽 N+1 个，首尾两帧不动；jitter ≤ 0.4 保证时刻严格递增）。
曝光：E = exposure·Δ。第 k 帧的画面 = [t_k − E, t_k] 内 M = 1 + ⌈8E⌉ 个等距时刻（含两端，间隔 ≤ 1/8 tick）的原始画面的
平均，再加像素噪声、clip、取整。只用 t_k 及以前（因果：画面不含 t_k 以后的信息）；早于轨迹起点（一阶场景 t = 0，二阶
场景 t = −1）的时刻取起点。
像素噪声：整 tick 时刻的帧用流 4 里那个 tick 的那份（与原来逐位相同）；tick 内部的帧用流 11 按帧号另起一个生成器
（熵 = 流 11 的熵 + [帧号+1]，帧号 −1 是二阶场景的前一帧），所以 frames() 与 frames_with_prev() 的同一帧噪声相同。
二阶场景的 frames_with_prev：多一帧 t = −Δ（预跑那个零动作 tick 内部的状态）；fps = 2 时 −Δ = −2 tick 早于预跑，报错。

S24c 空间采样（resolution、zoom、pan、drift，po_jepa.camera）也走这里（Variant.resampled()）：状态照上面取，每个取样时刻在它
自己的相机视野里画（systems.camera_view），画布外按面积混成 camera.SURROUND，再做曝光平均、加像素噪声（输出 R×R，
流 4 按块 [R,R,3] 顺序抽；R = 256 时与原来逐位相同）。

只做数据生成：编码、缓存、探针、规划遇到时间采样、空间采样调节项直接报错，接入在第 6 阶段。
"""
import math

import numpy as np

from . import datasets, simulation, systems, variants

PERIOD = systems.PERIOD
SUBTICKS = systems.SUBTICKS
EXPOSURE_SAMPLES_PER_TICK = 8      # 曝光窗口里取样时刻的最大间隔 = 1/8 tick
STREAM_JITTER = 10
STREAM_OFFGRID_NOISE = 11


def frame_period(v) -> float:
    """帧间隔（tick）：4/fps。"""
    return 4/v.fps


def frame_times(trajectory_id: int, v=None) -> np.ndarray:
    """第 0…N 帧的时刻（tick，float64）。v 默认是当前数据集的调节项。"""
    v = datasets.active().knobs if v is None else v
    datasets.active().check_id(trajectory_id)
    n = v.horizon*v.fps//4
    step = frame_period(v)
    t = np.arange(n+1)*step
    if v.jitter:
        xi = simulation._rng(trajectory_id, STREAM_JITTER).uniform(-1., 1., n+1)
        t[1:n] = t[1:n]+v.jitter*step*xi[1:n]
    return t


def exposure_times(times: np.ndarray, v, t_min: float) -> np.ndarray:
    """每帧曝光窗口里的取样时刻 [L, M]（第 0 列就是帧时刻），早于 t_min 的取 t_min。"""
    E = v.exposure*frame_period(v)
    M = 1+math.ceil(EXPOSURE_SAMPLES_PER_TICK*E)
    if M == 1:
        return times[:, None].copy()
    return np.maximum(times[:, None]-E*np.arange(M)/(M-1), t_min)


# ———————————————————— 各场景的推进（主链一个 tick / 子步网格上的一步 / 精确传播）————————————————————

class _Scene:
    """一个场景在一条轨迹上的推进接口。子类给出：
    t_min：最早的时刻（一阶 0，二阶 −1）；start() → 主链起点状态（t = t_min）；
    tick(q, u, k)：主链整 tick（生产实现，逐位）；
    S：每 tick 的子步数（None = 精确传播，用 exact(q, u, k, a, b)）；
    begin/step/out：子步网格上的游标（step(c, u, k, x0, x1) 推进同一个子步内的 [x0, x1]，单位子步；整子步时就是生产的那一步）；
    check(q)；hidden(states)；draw(states, ticks)（在 systems.raw_images() 里调用，返回原始 float 画面）。"""
    t_min = 0
    S = None
    pre_run = False         # 二阶场景：主链从 t = −1 起，第一个 tick 是零动作预跑

    def begin(self, q, u, k, x):
        return np.array(q, dtype=np.float64, copy=True)

    def out(self, c):
        return c

    def check(self, q):
        pass


def _walk(sc, q, u, k, x0, targets):
    """从子步位置 x0（单位子步，状态 q）沿子步网格走到各个 targets（升序，≥ x0），返回每个 target 的状态。
    整子步都走（游标前进）；target 不在网格上时，最后一截从游标出发另算（游标不动），所以每个 target 的结果与别的 target
    无关。"""
    c = sc.begin(q, u, k, x0); x = x0; res = []
    for t in targets:
        while True:
            nxt = math.floor(x)+1
            if nxt > t:
                break
            c = sc.step(c, u, k, x, float(nxt)); x = float(nxt)
        res.append(sc.out(c) if t == x else sc.out(sc.step(c, u, k, x, t)))
    return res


class _Ode(_Scene):
    """systems.System 的粒子类 ODE（double_well、van_der_pol、multi_particle、landscape）：每 tick 128 个 RK4 子步，
    f = vfield + wall，过程噪声（流 6）加在每个子步末尾。整 tick 照抄 System.variant_trajectory 的循环（测试逐位核）。"""
    S = 128

    def __init__(self, system, trajectory_id, v, T):
        self.sys, self.tid, self.v = system, trajectory_id, v
        self.h = PERIOD/128
        self.f = lambda x, ut: system.vfield(x, ut, v)+system.wall(x)
        if v.process_noise:
            self.xi = simulation._rng(trajectory_id, 6).standard_normal((T, 128, len(system.vinitial(trajectory_id, v))))
            self.scale = v.process_noise*math.sqrt(self.h)

    def start(self):
        return self.sys.vinitial(self.tid, self.v).copy()

    def tick(self, q, u, k):
        h, f = self.h, self.f
        for sub in range(128):
            k1 = f(q, u); k2 = f(q+h*k1/2, u)
            k3 = f(q+h*k2/2, u); k4 = f(q+h*k3, u)
            q = q+h*(k1+2*k2+2*k3+k4)/6
            if self.v.process_noise:
                q = q+self.scale*self.xi[k, sub]
        return q

    def step(self, q, u, k, x0, x1):
        j = math.floor(x0); full = x0 == j and x1 == j+1
        h = self.h if full else self.h*(x1-x0); f = self.f
        k1 = f(q, u); k2 = f(q+h*k1/2, u)
        k3 = f(q+h*k2/2, u); k4 = f(q+h*k3, u)
        q = q+h*(k1+2*k2+2*k3+k4)/6
        if self.v.process_noise:
            q = q+(self.scale if full else self.scale*(x1-x0))*self.xi[k, j]
        return q

    def check(self, q):
        if not np.isfinite(q).all() or np.abs(q).max() > self.sys.BOUND:
            raise ValueError(f"{self.sys.name} state left the rendered domain (trajectory {self.tid})")

    def hidden(self, states):
        return np.stack(states)

    def draw(self, states, ticks):
        return self.sys.draw(np.stack(states), self.tid, self.v, ticks)


class _Advection(_Scene):
    """advection_diffusion：状态是浓度场的 fft2 谱；分段常数流速下按时长精确推进（照抄 AdvectionDiffusion.variant_fields）。"""

    def __init__(self, system, trajectory_id, v, T):
        self.sys, self.tid, self.v = system, trajectory_id, v
        self.D, self.flow = v.param(system.name, "diffusion"), v.param(system.name, "flow")
        k = np.fft.fftfreq(systems.SIZE, d=1./systems.SIZE)*2*np.pi
        self.ky, self.kx = np.meshgrid(k, k, indexing="ij")

    def start(self):
        return np.fft.fft2(self.sys.variant_initial(self.tid, self.v))

    def _propagate(self, spectrum, u, seconds):
        wx, wy = self.flow*u
        kx, ky = self.kx, self.ky
        return spectrum*np.exp((-self.D*(kx**2+ky**2)-1j*(kx*wx-ky*wy))*seconds)

    def tick(self, q, u, k):
        return self._propagate(q, u, PERIOD)

    def exact(self, q, u, k, a, b):
        return self._propagate(q, u, (b-a)*PERIOD)

    def _fields(self, states):
        return np.stack([np.fft.ifft2(s).real for s in states])

    def hidden(self, states):
        f = self._fields(states)
        return f.reshape(f.shape[0], 16, 16, 16, 16).mean((2, 4)).reshape(f.shape[0], -1)

    def draw(self, states, ticks):
        return self.sys.draw_fields(self._fields(states), self.tid, self.v, ticks)


class _GrayScott(_Scene):
    """gray_scott：状态 (U, V, p)；每 tick UNITS_PER_TICK/DT = 40 个 Strang 分裂步（整 tick 调 GrayScott._run 本身）。
    子步网格上的一步照抄 _run 的循环体（短步用 dt' = dt·(x1−x0) 重算扩散因子）。"""

    def __init__(self, system, trajectory_id, v, T):
        from . import new_systems
        self.impl = new_systems.SYSTEMS["gray_scott"]
        self.tid, self.v, self.params = trajectory_id, v, dict(v.params)
        self.F, self.k, self.a = (new_systems.param("gray_scott", self.params, n) for n in ("feed", "kill", "inject"))
        self.S = int(round(self.impl.UNITS_PER_TICK/self.impl.DT))
        n = self.impl.N
        self.k2 = (2*np.pi*np.fft.fftfreq(n))[:, None]**2+(2*np.pi*np.fft.rfftfreq(n))[None, :]**2

    def _velocity(self, u):
        return self.impl.SPEED*u*PERIOD/self.impl.UNITS_PER_TICK

    def start(self):
        return self.impl.start_state(self.tid, self.params)

    def tick(self, q, u, k):
        return self.impl._run(*q, self._velocity(u), self.impl.UNITS_PER_TICK, self.impl.DT, self.F, self.k, self.a)

    def begin(self, q, u, k, x):
        return tuple(np.array(z, copy=True) for z in q)

    def step(self, c, u, k, x0, x1):
        impl = self.impl; U, V, p = c
        dt = impl.DT if (x0 == math.floor(x0) and x1 == x0+1) else impl.DT*(x1-x0)
        velocity = self._velocity(u); F, kk, a = self.F, self.k, self.a
        eu = np.exp(-impl.DU*self.k2*dt/2); ev = np.exp(-impl.DV*self.k2*dt/2)
        half = lambda X, e: np.fft.irfft2(np.fft.rfft2(X)*e, s=X.shape)

        def react(U, V, G):
            uvv = U*V*V
            return -uvv+F*(1.-U), uvv-(F+kk)*V+a*G*(1.-V)

        U = half(U, eu); V = half(V, ev)
        G = impl._gauss(p+velocity*dt/2)
        k1 = react(U, V, G); k2 = react(U+dt/2*k1[0], V+dt/2*k1[1], G)
        k3 = react(U+dt/2*k2[0], V+dt/2*k2[1], G); k4 = react(U+dt*k3[0], V+dt*k3[1], G)
        U = U+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6; V = V+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
        U = half(U, eu); V = half(V, ev)
        p = (p+velocity*dt) % 1.
        return U, V, p

    def check(self, q):
        if not (np.isfinite(q[0]).all() and np.isfinite(q[1]).all()):
            raise ValueError("gray_scott 数值发散")

    def hidden(self, states):
        return self.impl.states(*(np.stack(z) for z in zip(*states)))

    def draw(self, states, ticks):
        U, V, p = (np.stack(z) for z in zip(*states))
        return self.impl.draw(U, V, p, self.tid, self.v, ticks)


class _Push(_Scene):
    """push：每 tick 256 个 RK4 子步（照抄 Push.trajectory 的循环）。"""

    def __init__(self, system, trajectory_id, v, T):
        from . import new_systems
        self.impl = new_systems.SYSTEMS["push"]
        self.tid, self.v, self.params = trajectory_id, v, dict(v.params)
        self.S = self.impl.STEPS_PER_TICK
        self.h = PERIOD/self.S

    def start(self):
        return self.impl.initial(self.tid, self.params).copy()

    def _rk4(self, q, u, h):
        f, params = self.impl.field, self.params
        k1 = f(q, u, params); k2 = f(q+h*k1/2, u, params)
        k3 = f(q+h*k2/2, u, params); k4 = f(q+h*k3, u, params)
        return q+h*(k1+2*k2+2*k3+k4)/6

    def tick(self, q, u, k):
        for _ in range(self.S):
            q = self._rk4(q, u, self.h)
        return q

    def step(self, q, u, k, x0, x1):
        full = x0 == math.floor(x0) and x1 == x0+1
        return self._rk4(q, u, self.h if full else self.h*(x1-x0))

    def check(self, q):
        if not np.isfinite(q).all() or np.abs(q[:4]).max() > self.impl.BOUND:
            raise ValueError(f"push state left the rendered domain (trajectory {self.tid})")

    def hidden(self, states):
        return np.stack(states)

    def draw(self, states, ticks):
        return self.impl.draw(np.stack(states), self.tid, self.params, self.v, ticks)


class _SecondOrder(_Scene):
    """二阶场景：主链从 t = −1 起（second_order 的 start：初值 + 零动作预跑），整 tick 调 second_order 的 tick() 本身。
    子步网格：pendulum 50、cart_pole 100 个 RK4 子步，bouncing_balls 32 个事件驱动子步，wave 16 个子步（谱空间游标），
    spring_chain 按矩阵指数精确推进。"""
    t_min = -1
    pre_run = True

    def __init__(self, system, trajectory_id, v, T):
        from . import second_order
        self.mod = second_order
        self.impl = second_order.SYSTEMS[system.name]
        self.tid, self.v, self.params = trajectory_id, v, dict(v.params)
        name = system.name
        if name in ("pendulum", "cart_pole", "bouncing_balls", "wave"):
            self.S = self.impl.SUBSTEPS
        if name == "pendulum":
            gl, gamma = (second_order.param(name, self.params, n) for n in ("g_over_l", "damping"))
            self.field = lambda u: (lambda x: self.impl.field(x, u, gl, gamma))
        elif name == "cart_pole":
            l, ratio, b = (second_order.param(name, self.params, n) for n in ("length", "mass_ratio", "friction"))
            self.field = lambda u: (lambda z: self.impl.field(z, u, l, ratio*self.impl.M, b))

    def start(self):
        """→ (t = −1 的状态, t = 0 的状态)。"""
        q_prev, q0 = self.impl.start(self.tid, self.params)
        self.impl.check(q0, self.params)
        return q_prev, q0

    def tick(self, q, u, k):
        return self.impl.tick(q, u, self.params)

    # —— 子步网格上的游标 ——
    def begin(self, q, u, k, x):
        name = self.impl.name
        if name == "wave":
            uu, w, p = self.impl.unpack(np.asarray(q, dtype=np.float64))
            return (np.fft.rfft2(uu), np.fft.rfft2(w), p.copy(), x, x)
        return np.array(q, dtype=np.float64, copy=True)

    def step(self, c, u, k, x0, x1):
        name = self.impl.name; m = self.S
        full = x0 == math.floor(x0) and x1 == x0+1
        h = PERIOD/m
        if name in ("pendulum", "cart_pole"):
            return self.mod._rk4(self.field(u), c, h if full else h*(x1-x0), 1)
        if name == "bouncing_balls":
            n, e, beta = (self.mod.param(name, self.params, x) for x in ("n", "restitution", "drag"))
            pos = c[:2*n].reshape(n, 2).copy(); vel = c[2*n:].reshape(n, 2).copy()
            acc = np.zeros((n, 2)); acc[0] = self.impl.K*u
            self.impl.substep(pos, vel, acc, beta, e, h if full else h*(x1-x0))
            return np.concatenate((pos.ravel(), vel.ravel()))
        # wave：谱空间游标 (x̂, v̂, 起点的源位置 p_a, 起点位置 x_a, 当前位置)；子步中点的源位置相对起点算（同 Wave.tick）
        c_, gamma, load = (self.mod.param(name, self.params, x) for x in ("speed", "damping", "load"))
        xh, vh, pa, xa, _ = c
        vel = self.impl.SPEED*np.asarray(u, dtype=np.float64)
        dt = h if full else h*(x1-x0)
        mid = pa+vel*((x0+x1)/2-xa)*h
        fh = -load*c_*c_*self.impl.k2*self.impl.source_hat(mid)
        xh, vh = self.impl.propagate(xh, vh, fh, dt, c_, gamma)
        return (xh, vh, pa, xa, x1)

    def exact(self, q, u, k, a, b):
        """spring_chain：时长 (b−a)·0.25 s 的矩阵指数（同 SpringChain.matrices / tick，只换时长）。"""
        from scipy.linalg import expm
        impl, params = self.impl, self.params
        n, kk, gamma = (self.mod.param(impl.name, params, x) for x in ("n", "stiffness", "damping"))
        lap = 2*np.eye(n)-np.eye(n, k=1)-np.eye(n, k=-1)
        A = np.zeros((2*n+1, 2*n+1))
        A[:n, n:2*n] = np.eye(n); A[n:2*n, :n] = -kk*lap; A[n:2*n, n:2*n] = -gamma*np.eye(n); A[n, 2*n] = impl.K
        E = expm(A*((b-a)*PERIOD))
        Phi, Gam = E[:2*n, :2*n], E[:2*n, 2*n]
        pos = q[:2*n].reshape(n, 2); vel = q[2*n:].reshape(n, 2)
        rest = np.stack((impl.rest(n), np.zeros(n)), -1)
        z = np.concatenate((pos-rest, vel))
        z = Phi@z+Gam[:, None]*np.asarray(u, dtype=np.float64)[None, :]
        return np.concatenate(((z[:n]+rest).ravel(), z[n:].ravel()))

    def check(self, q):
        self.impl.check(q, self.params)

    def hidden(self, states):
        return self.impl.hidden(np.stack(states), self.params)

    def draw(self, states, ticks):
        return self.impl.draw(np.stack(states), self.tid, self.params, self.v, np.asarray(ticks, dtype=np.float64))


def _scene(system, trajectory_id, v, T):
    name = system.name
    if name == "advection_diffusion":
        return _Advection(system, trajectory_id, v, T)
    if name == "gray_scott":
        return _GrayScott(system, trajectory_id, v, T)
    if name == "push":
        return _Push(system, trajectory_id, v, T)
    if name in variants.SECOND_ORDER:
        return _SecondOrder(system, trajectory_id, v, T)
    return _Ode(system, trajectory_id, v, T)


def _wave_walk(sc, q, u, k, x0, targets):
    """wave 的 _walk：游标在谱空间，取状态时做 irfft2，源位置 p = (p_a + vel·(x − x_a)·h) mod 1（同 Wave.tick 末尾）。"""
    c = sc.begin(q, u, k, x0); x = x0; res = []
    impl = sc.impl; h = PERIOD/sc.S
    vel = impl.SPEED*np.asarray(u, dtype=np.float64)
    shape = (impl.N, impl.N)

    def out(c, x):
        xh, vh, pa, xa, _ = c
        return impl.pack(np.fft.irfft2(xh, s=shape), np.fft.irfft2(vh, s=shape), (pa+vel*(x-xa)*h) % 1.)

    for t in targets:
        while True:
            nxt = math.floor(x)+1
            if nxt > t:
                break
            c = sc.step(c, u, k, x, float(nxt)); x = float(nxt)
        res.append(out(c, x) if t == x else out(sc.step(c, u, k, x, t), t))
    return res


def _advance(sc, q, u, k, a, b, targets=(), end=True):
    """从 tick k 的位置 a（tick 的比例）推进到 b，顺带取 targets（升序，a ≤ t < b，tick 的比例）的状态。
    → (b 处的状态（end=False 时 None）, [targets 的状态])。整 tick（a = 0、b = 1）的终点用生产实现 tick()。
    终点和 targets 分开算：终点只由 q、u 决定，与取了哪些 targets 无关。"""
    whole = a == 0. and b == 1.
    if sc.S is None:
        obs = [q if t == a else sc.exact(q, u, k, a, t) for t in targets]
        if not end:
            return None, obs
        return (sc.tick(q, u, k) if whole else sc.exact(q, u, k, a, b)), obs
    walk = _wave_walk if isinstance(sc, _SecondOrder) and sc.impl.name == "wave" else _walk
    S = sc.S
    later = [t for t in targets if t != a]            # 起点本身就是 q（不经过游标，wave 的游标会多一次 FFT 往返）
    obs = [q]*(len(targets)-len(later)) + (walk(sc, q, u, k, a*S, [t*S for t in later]) if later else [])
    if not end:
        return None, obs
    return (sc.tick(q, u, k) if whole else walk(sc, q, u, k, a*S, [b*S])[0]), obs


def _pieces(u4, k):
    """tick k 里执行动作不变的各段 [(a, b, u)]，a、b 是 tick 的比例（1/4 的整数倍）。"""
    row = u4[SUBTICKS*k:SUBTICKS*(k+1)]
    out, start = [], 0
    for i in range(1, SUBTICKS+1):
        if i == SUBTICKS or not np.array_equal(row[i], row[start]):
            out.append((start/SUBTICKS, i/SUBTICKS, row[start]))
            start = i
    return out


def simulate(sc, u4, times):
    """主链 + 旁支：→ 每个 times（tick，可重复、可乱序，t_min ≤ t ≤ horizon）的状态，顺序与 times 相同。"""
    T = len(u4)//SUBTICKS
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or times.min() < sc.t_min or times.max() > T:
        raise ValueError(f"时刻必须在 [{sc.t_min}, {T}] 内")
    order = np.argsort(times, kind="stable")
    result = [None]*len(times)
    pos = 0                                   # order 里下一个要取的
    def take(lo, hi, include_hi=False):
        nonlocal pos
        idx = []
        while pos < len(order) and (times[order[pos]] < hi or (include_hi and times[order[pos]] == hi)):
            assert times[order[pos]] >= lo
            idx.append(order[pos]); pos += 1
        return idx

    if sc.pre_run:
        q_prev, q = sc.start()
        idx = take(-1., 0.)
        if idx:
            fr = sorted({float(times[i])+1. for i in idx})
            _, obs = _advance(sc, q_prev, np.zeros(2), -1, 0., 1., fr, end=False)
            lookup = dict(zip(fr, obs))
            for i in idx:
                result[i] = lookup[float(times[i])+1.]
    else:
        q = sc.start()
    for k in range(T):
        for a, b, u in _pieces(u4, k):
            idx = take(k+a, k+b)
            fr = sorted({float(times[i])-k for i in idx})
            q_next, obs = _advance(sc, q, u, k, a, b, fr)
            lookup = dict(zip(fr, obs))
            for i in idx:
                result[i] = lookup[float(times[i])-k]
            q = q_next
        sc.check(q)
    for i in take(T, T, include_hi=True):
        result[i] = q
    assert pos == len(order)
    for r in result:
        sc.check(r)
    return result


class _NoiseBlocks:
    """流 4 的像素噪声按块（一帧 [256,256,3]）顺序抽：第 i 块与一次抽 [n,256,256,3] 的第 i 份逐位相同（numpy 按 C 顺序逐个填）。
    S24c：块边长 size = 输出分辨率。"""

    def __init__(self, g, sigma, size=systems.SIZE):
        self.g, self.sigma, self.i, self.last, self.size = g, sigma, -1, None, size

    def block(self, i):
        if i < self.i:
            raise ValueError("像素噪声块要按顺序取")
        while self.i < i:
            self.last = self.g.normal(0., self.sigma, (self.size, self.size, 3)); self.i += 1
        return self.last


def _offgrid_noise(trajectory_id, key, sigma, size=systems.SIZE):
    entropy = simulation.seed_entropy(trajectory_id, STREAM_OFFGRID_NOISE)+[int(key)]
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    return g.normal(0., sigma, (size, size, 3))


def _draw_raw(sc, states, samples, trajectory_id, v):
    """一帧的原始画面（曝光平均之后、像素噪声之前）[R,R,3] float64。samples：这一帧曝光窗口里的 M 个取样时刻。
    S24c 空间采样：每个取样时刻在自己的相机视野里画、画布外混成 SURROUND，再平均（drift 时曝光窗口里相机也在动）。"""
    if not v.changes_space():
        with systems.raw_images():
            raw = sc.draw(states, samples)
        return raw[0] if len(samples) == 1 else raw.mean(0)
    from . import camera
    out = []
    for state, t, view in zip(states, samples, camera.views(trajectory_id, samples, v)):
        with systems.raw_images(), systems.camera_view(view):
            raw = sc.draw([state], np.array([t]))[0]
        if raw.shape != (view.size, view.size, 3):
            raise AssertionError(f"{type(sc).__name__} 的画图函数没有按相机视野画：{raw.shape}")
        out.append(camera.canvas_mask(raw, view))
    return out[0] if len(out) == 1 else np.stack(out).mean(0)


def frames(system, trajectory_id: int, *, prev: bool = False):
    """→ (隐藏状态 [L, d], 画面 [L,R,R,3] uint8)，L = N+1（prev=True 时二阶场景多 t = −Δ 一帧在最前），R = resolution（默认 256）。
    各场景的 frames()（及二阶场景的 frames_with_prev()）在时间采样或空间采样调节项起作用时调这里。"""
    spec = datasets.active(); v = spec.knobs
    if spec.system != system.name:
        raise ValueError(f"{system.name} 需要当前数据集是 {system.name}，现在是 {spec.system}")
    spec.check_id(trajectory_id)
    T = v.horizon
    commanded = systems.commanded_controls(trajectory_id)
    u4 = systems.applied_subticks(trajectory_id, v, commanded)
    sc = _scene(system, trajectory_id, v, T)
    times = frame_times(trajectory_id, v)
    keys = np.arange(1, len(times)+1)                 # 帧号 + 1（帧号 −1 = 二阶场景的前一帧 → 0）
    if prev:
        if system.name not in variants.SECOND_ORDER:
            raise ValueError("只有二阶场景有前一帧")
        if frame_period(v) > 1:
            raise ValueError(f"fps={v.fps}：前一帧在 t = −{frame_period(v):g} tick，早于预跑的起点 t = −1，取不到")
        times = np.concatenate(([-frame_period(v)], times)); keys = np.concatenate(([0], keys))
    samples = exposure_times(times, v, sc.t_min)
    states = simulate(sc, u4, samples.ravel())
    L, M = samples.shape
    hidden = sc.hidden([states[i*M] for i in range(L)])
    offset = 1 if sc.pre_run else 0
    R = v.resolution
    noise4 = _NoiseBlocks(simulation._rng(trajectory_id, 4), v.pixel_noise, R) if v.pixel_noise else None
    images = np.empty((L, R, R, 3), dtype=np.uint8)
    for i in range(L):
        image = _draw_raw(sc, states[i*M:(i+1)*M], samples[i], trajectory_id, v)
        if v.pixel_noise:
            t = times[i]
            if t == math.floor(t):
                image = image+noise4.block(int(t)+offset)
            else:
                image = image+_offgrid_noise(trajectory_id, keys[i], v.pixel_noise, R)
        images[i] = np.rint(np.clip(image, 0., 255.)).astype(np.uint8)
    return hidden, images
