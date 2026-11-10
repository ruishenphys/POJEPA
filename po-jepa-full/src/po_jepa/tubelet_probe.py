"""S22 两帧 tubelet 探针（v8 数据集扩展第 3 阶段）：真实相邻两帧编码后，角速度能不能从冻结特征里线性读出。

背景（v8 规划 §1.4）：V-JEPA 2 的 patch 是两帧一组的三维 tubelet（2×16×16）。v3–v7 每个时刻把同一帧复制两份
（vjepa.FrozenFrameEncoder），特征里没有运动信息；二阶场景（第 4 阶段）只看一帧，速度观测不到。推荐方案是让
tubelet 吃真实的第 t−1、t 帧，PO-JEPA、损失和缓存格式都不变。本模块是采用前的验证。

本模块只供探针：单摆不登记成数据集场景，不进 datasets、variants、systems 的任何表；第 4 阶段登记的 pendulum
自己定接口（动作维数、调节项），可以照抄这里的物理。已有文件一个字节都没动。

1. 探针单摆：θ̈ = −(g/L)·sin θ − γ·θ̇ + K·u，g/L = 9.81 s⁻²（小角周期 2.006 s，每周期约 8 帧）、γ = 0.2 s⁻¹、
   K = 4 rad/s² 每单位动作。θ = 0 是竖直向下，θ 增大时摆锤往画面右边走。
   - 时间：控制周期 0.25 s。帧 f = 0…33 对应 tick t = f − 1 = −1…32（多一帧 t = −1，这样 t = 0 也有前一帧）；
     动作 u 是 33 个 tick 的标量力矩，第 f 段作用在帧 f 到 f+1 之间。每个 tick 用 RK4 走 SUBSTEPS = 50 步。
   - 动作：v6 的分段规则（段长 1/4/8 tick、20% 零动作），值 ∈ [−0.5, 0.5]（随机流 1）。
   - 初值：θ ∈ U(−π, π)、θ̇ ∈ U(−3, 3) rad/s（随机流 0）。前 200 条里 42 条越过顶点附近（|θ| > 0.9π）、12 条净转过一整圈以上，
     其余是摆动；|θ̇| 最大约 7.2 rad/s，低于 4 fps 的奈奎斯特界 4π rad/s 的 0.6 倍（每帧转角 < π，否则两帧之间
     方向有歧义；测试核前 200 条的最大值 < 0.8 倍）。
   - 随机流：熵 [1703, 2200, 轨迹编号, 流号]，与数据集场景（[1703, 1000+系统序号, …]）不重叠。
   - 画面 256×256×3 uint8：v6 的深色网格背景，支点在画面中心，摆杆是灰色抗锯齿细线（长 80 像素），
     摆锤是橙色高斯光点（σ = 6 像素，同 v7 默认）。画面只由 θ 决定，单帧看不出 θ̇。
2. 三种特征（每个 tick t = 0…32 一个样本，每条轨迹 33 个）：
   - dup：第 t 帧复制两份组成 tubelet，就是 v7 的编码；
   - pair：第 t−1、t 帧按时间顺序组成 tubelet（推荐方案）；
   - dup_concat：第 t−1 帧和第 t 帧的 dup 特征拼接（备选方案“模型同时吃两帧特征”能拿到的信息）。
   两帧的预处理用同一条轨迹同一个重置后的 generator（与 v7 相同的规则），所以两帧的裁剪相同。
3. 三种读出：mean（256 个 token 平均，1408 维）、pool4（4×4 平均池化，22528 维）、full（全部 16×16×1408 = 360448 维）。
4. 线性探针：对偶形式的岭回归（核 K = XXᵀ，按训练集均值精确中心化），目标 sin θ、cos θ、θ̇（第 t 帧的瞬时角速度）、
   θ̄̇ = wrap(θ_t − θ_{t−1})/0.25 s（两帧之间的平均角速度）。按轨迹分 5 折（轨迹编号 mod 5）：第 k 折测试，
   第 k+1 折选 λ，其余训练；选好 λ 后在训练 + 选 λ 两折上重新拟合，报测试折的 R²，5 折取平均。
5. 判据（可改的设计选择，只改这里的常数）：读出按嵌套方式选（S22-fix1）：每个外层折 k 在它选 λ 的那一折（第 k+1 折）
   上挑 pair 的 θ̇ R² 最高的读出，测试折只用来打分，不参与挑选。各折所选读出上，pair 的 θ̇ 测试 R² 5 折平均
   ≥ ADOPT_R2 = 0.9，且比同一读出的 dup 高 ≥ ADOPT_GAIN = 0.1，就采用两帧输入；同时要求所选读出上 dup 的 sin θ、
   cos θ 测试 R²（5 折平均）都 ≥ POSITION_R2 = 0.9，否则说明探针本身没读出位置，结论无效。各读出的 5 折平均另列
   诊断表，不进判据。
6. 生成定义：definition() 是决定实验含义的常数（物理、动作、初值、随机流、画面、tubelet 帧号），编码清单记下它，
   fit 要求每条轨迹的定义与当前代码相同。
"""
import math

import numpy as np
import torch

from . import vjepa
from .systems import COLORS, SIZE, _background

PROBE_STREAM = 2200
G_OVER_L = 9.81
GAMMA = .2
TORQUE_PER_ACTION = 4.
SUBSTEPS = 50
PERIOD = .25
INTERVALS = 33                 # tick −1…32 之间的 33 段
FRAMES = INTERVALS+1           # 帧 0…33 = tick −1…32
THETA0 = math.pi
OMEGA0 = 3.
SEGMENTS = (1, 4, 8)
ZERO_FRACTION = .2
ROD_PX = 80.
BOB_SIGMA = 6.
BOB_COLOR = np.array(COLORS["orange"])
ROD_COLOR = np.array([110., 112., 120.])
ROD_HALF_WIDTH = 1.5

MODES = ("dup", "pair", "dup_concat")
READOUTS = ("mean", "pool4", "full")
TARGETS = ("sin_theta", "cos_theta", "omega", "omega_mean")
FOLDS = 5
LAMBDA_EXPONENTS = tuple(np.arange(-7., 2.01, .5))
ADOPT_R2 = .9
ADOPT_GAIN = .1
POSITION_R2 = .9


def definition() -> dict:
    """决定缓存含义的探针定义（JSON 可序列化）。改任何一项都必须重新编码；fit 与清单逐项比较。"""
    return dict(probe_stream=PROBE_STREAM, rng_entropy=[1703, PROBE_STREAM, "trajectory_id", "stream"],
                streams=dict(initial_state=0, controls=1, preprocess=2),
                g_over_l=G_OVER_L, gamma=GAMMA, torque_per_action=TORQUE_PER_ACTION, substeps=SUBSTEPS, period=PERIOD,
                intervals=INTERVALS, frames=FRAMES, frame_ticks=list(range(-1, INTERVALS)),
                theta0=THETA0, omega0=OMEGA0, segments=list(SEGMENTS), zero_fraction=ZERO_FRACTION,
                image_size=SIZE, rod_px=ROD_PX, bob_sigma=BOB_SIGMA, bob_color=BOB_COLOR.tolist(),
                rod_color=ROD_COLOR.tolist(), rod_half_width=ROD_HALF_WIDTH,
                clips={mode: [list(c) for c in clip_indices(mode)] for mode in ("dup", "pair")})


def rng(trajectory_id: int, stream: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([1703, PROBE_STREAM, int(trajectory_id), int(stream)])))


def preprocess_seed(trajectory_id: int) -> int:
    """每条轨迹的预处理 generator 种子（随机流 2，同 simulation.preprocess_seed 的写法）。"""
    return int(np.random.SeedSequence([1703, PROBE_STREAM, int(trajectory_id), 2]).generate_state(1, dtype=np.uint64)[0])


def controls(trajectory_id: int) -> np.ndarray:
    """33 段的标量动作 [33] float64：段长 ∈ {1,4,8}、20% 零动作、值 ∈ U(−0.5, 0.5)（随机流 1）。"""
    g = rng(trajectory_id, 1); u = np.zeros(INTERVALS); t = 0
    while t < INTERVALS:
        n = int(g.choice(SEGMENTS)); value = 0. if g.random() < ZERO_FRACTION else g.uniform(-.5, .5)
        u[t:t+n] = value; t += n
    return u


def initial_state(trajectory_id: int) -> np.ndarray:
    g = rng(trajectory_id, 0)
    return np.array([g.uniform(-THETA0, THETA0), g.uniform(-OMEGA0, OMEGA0)])


def _rhs(theta, omega, torque, g_over_l, gamma):
    return omega, -g_over_l*math.sin(theta)-gamma*omega+torque


def simulate(state0, u, *, g_over_l=G_OVER_L, gamma=GAMMA, substeps=SUBSTEPS) -> np.ndarray:
    """初值 (θ, θ̇) 和每段动作 u:[L] → 状态 [L+1, 2]（θ 不回绕）。每段 RK4 走 substeps 步，力矩 K·u 段内不变。"""
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 1 or not np.isfinite(u).all():
        raise ValueError("动作必须是有限 float64[L]")
    theta, omega = (float(x) for x in state0)
    h = PERIOD/substeps; out = [(theta, omega)]
    for value in u:
        torque = TORQUE_PER_ACTION*float(value)
        for _ in range(substeps):
            k1 = _rhs(theta, omega, torque, g_over_l, gamma)
            k2 = _rhs(theta+.5*h*k1[0], omega+.5*h*k1[1], torque, g_over_l, gamma)
            k3 = _rhs(theta+.5*h*k2[0], omega+.5*h*k2[1], torque, g_over_l, gamma)
            k4 = _rhs(theta+h*k3[0], omega+h*k3[1], torque, g_over_l, gamma)
            theta += h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); omega += h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        out.append((theta, omega))
    return np.array(out)


def trajectory(trajectory_id: int):
    """→ (状态 [34,2]，帧 f 对应 tick f−1；动作 [33])。"""
    u = controls(trajectory_id)
    return simulate(initial_state(trajectory_id), u), u


def bob_rc(theta) -> np.ndarray:
    """摆锤中心（行、列，像素）[L,2]：θ = 0 竖直向下（行增大），θ 增大往右（列增大）。"""
    theta = np.asarray(theta, dtype=np.float64)
    return np.stack((SIZE/2+ROD_PX*np.cos(theta), SIZE/2+ROD_PX*np.sin(theta)), -1)


def draw(states) -> np.ndarray:
    """状态 [L,2] → 画面 [L,256,256,3] uint8。只用 θ：背景 → 摆杆（到线段的距离 ≤ 1.5 像素线性淡出）→ 摆锤光点。"""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 2 or not np.isfinite(states).all():
        raise ValueError("状态必须是有限 [L,2]")
    row, col = np.meshgrid(np.arange(SIZE)+.5, np.arange(SIZE)+.5, indexing="ij")
    base = _background(); frames = []
    for (r, c) in bob_rc(states[:, 0]):
        pr, pc = SIZE/2, SIZE/2; dr, dc = r-pr, c-pc
        s = np.clip(((row-pr)*dr+(col-pc)*dc)/(dr*dr+dc*dc), 0., 1.)
        d = np.hypot(row-(pr+s*dr), col-(pc+s*dc))
        rod = np.clip(ROD_HALF_WIDTH+.5-d, 0., 1.)[..., None]
        image = base+(ROD_COLOR-base)*rod
        w = np.exp(-((row-r)**2+(col-c)**2)/(2*BOB_SIGMA**2))[..., None]
        frames.append(np.rint(image+(BOB_COLOR-image)*w))
    return np.stack(frames).astype(np.uint8)


def targets(states) -> np.ndarray:
    """状态 [34,2]（tick −1…32）→ 目标 [33,4]（tick 0…32）：sin θ_t、cos θ_t、θ̇_t、wrap(θ_t − θ_{t−1})/0.25。"""
    states = np.asarray(states, dtype=np.float64)
    if states.shape != (FRAMES, 2):
        raise ValueError("状态必须是 [34,2]")
    theta, omega = states[1:, 0], states[1:, 1]
    step = np.angle(np.exp(1j*(states[1:, 0]-states[:-1, 0])))
    return np.stack((np.sin(theta), np.cos(theta), omega, step/PERIOD), -1)


# —————————————————————————— 编码 ——————————————————————————

def clip_indices(mode: str):
    """每种编码送进 tubelet 的帧号（前一帧、后一帧）。dup 编全部 34 帧（dup_concat 要用 tick −1 的），pair 编 33 对。"""
    if mode == "dup":
        return [(f, f) for f in range(FRAMES)]
    if mode == "pair":
        return [(f-1, f) for f in range(1, FRAMES)]
    raise ValueError(f"只编码 dup、pair，不是 {mode}")


def sample_rows(mode: str):
    """每个 tick t = 0…32 的样本在编码结果里的行号：dup → 第 t+1 帧；pair → 第 t 对；dup_concat →（第 t 帧, 第 t+1 帧）的 dup。"""
    ticks = np.arange(INTERVALS)
    if mode == "dup":
        return ticks+1
    if mode == "pair":
        return ticks
    if mode == "dup_concat":
        return ticks, ticks+1
    raise ValueError(mode)


class FrozenClipEncoder(vjepa.FrozenFrameEncoder):
    """同一个冻结编码器，输入是两张预处理图按时间顺序组成的 tubelet（[B,3,2,256,256]）。

    冻结、eval、设备、dtype 的设置和检查与 FrozenFrameEncoder 完全相同（__init__ 直接复用）。prev 和 cur 是同一张图时，
    视频张量与 FrozenFrameEncoder 的“单图复制两份”逐元素相同，所以 dup 就是 v7 的编码。
    """

    def encode_clip(self, prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        if any(m.training for m in self.encoder.modules()):
            raise ValueError("encoder must remain in eval mode")
        if any(p.requires_grad for p in self.encoder.parameters()):
            raise ValueError("encoder parameters must remain frozen")
        if any(p.dtype != torch.float32 or p.device != self.encoder_device for p in self.encoder.parameters()):
            raise ValueError("encoder parameter device/dtype was changed")
        for frame in (prev, cur):
            if (not isinstance(frame, torch.Tensor) or frame.device.type != "cpu" or frame.dtype != torch.float32
                    or frame.ndim != 4 or frame.shape[0] <= 0 or frame.shape[1:] != (3, 256, 256)
                    or not torch.isfinite(frame).all()):
                raise ValueError("preprocessed frames must be finite CPU float32 [B,3,256,256]")
        if prev.shape != cur.shape:
            raise ValueError("prev and cur must have the same batch")
        with torch.inference_mode(False), torch.no_grad(), torch.autocast(self.encoder_device.type, enabled=False):
            video = torch.stack((prev, cur), 2).to(self.encoder_device)
            raw_z = self.encoder(video)
        if (not isinstance(raw_z, torch.Tensor) or raw_z.shape != (prev.shape[0], 256, 1408)
                or not raw_z.is_floating_point() or not torch.isfinite(raw_z).all()):
            raise ValueError("encoder must return one finite raw spatial block [B,256,1408]")
        return raw_z.detach()


def fake_tubelet_encoder():
    """仅供 CPU 流程测试：确定性随机 tubelet 投影 + LayerNorm，[B,3,2,256,256] → [B,256,1408]。
    和真实编码器的 patch 嵌入一样按 2×16×16 的 tubelet 取块，所以 dup 时两帧相同、pair 时含两帧的差别。"""
    g = torch.Generator().manual_seed(0)
    weight = torch.randn(3*2*16*16, 1408, generator=g)/math.sqrt(3*2*16*16)

    def encode(video):
        b = video.shape[0]
        patches = video.reshape(b, 3, 2, 16, 16, 16, 16).permute(0, 3, 5, 1, 2, 4, 6).reshape(b, 256, -1)
        return torch.nn.functional.layer_norm(patches@weight, (1408,))
    return encode


def encode_trajectory(frames, trajectory_id, *, encoder=None, fake=False, batch=34):
    """画面 [34,256,256,3] uint8 → {"dup": [34,256,1408], "pair": [33,256,1408]} float64 numpy。

    真实编码器：每帧前把这条轨迹的预处理 generator 重置到同一状态（v7 规则），再单帧预处理；两帧用各自的预处理图。
    fake：像素 /255（与 encode_v7 --fake-encoder 相同，不做裁剪归一化）。"""
    rgb = torch.from_numpy(np.ascontiguousarray(frames))
    if rgb.shape != (FRAMES, SIZE, SIZE, 3) or rgb.dtype != torch.uint8:
        raise ValueError("画面必须是 uint8 [34,256,256,3]")
    if fake:
        pre = rgb.permute(0, 3, 1, 2).float()/255.
        encode = fake_tubelet_encoder() if encoder is None else encoder
    else:
        from .runtime import cache
        seed = preprocess_seed(trajectory_id); parts = []
        for f in range(FRAMES):
            g = torch.Generator(device="cpu").manual_seed(seed)
            parts.append(vjepa.preprocess_rgb(rgb[f:f+1], spec=cache.preprocess_spec(), generator=g))
        pre = torch.cat(parts)
        encode = None
    out = {}
    for mode in ("dup", "pair"):
        idx = clip_indices(mode); chunks = []
        for s in range(0, len(idx), batch):
            prev = pre[[a for a, _ in idx[s:s+batch]]]; cur = pre[[b for _, b in idx[s:s+batch]]]
            if fake:
                chunks.append(encode(torch.stack((prev, cur), 2)).double())
            else:
                chunks.append(encoder.encode_clip(prev, cur).cpu().double())
        out[mode] = torch.cat(chunks).numpy()
    return out


# —————————————————————————— 线性探针 ——————————————————————————

def readout(z: torch.Tensor, kind: str) -> torch.Tensor:
    """token 场 [n,256,1408]（16×16 行优先）→ 读出特征 [n,d] float32。"""
    n = z.shape[0]
    if z.shape[1:] != (256, 1408):
        raise ValueError("特征必须是 [n,256,1408]")
    z = z.float()
    if kind == "mean":
        return z.mean(1)
    if kind == "pool4":
        return z.reshape(n, 4, 4, 4, 4, 1408).mean((2, 4)).reshape(n, -1)
    if kind == "full":
        return z.reshape(n, -1)
    raise ValueError(kind)


def gram(z: torch.Tensor, kind: str, *, device="cpu", chunk=32768) -> np.ndarray:
    """读出特征的核 K = XXᵀ [n,n] float64。先减去全体样本的均值（只为数值条件；之后按每折的训练集均值精确中心化，
    任何常数平移都不改变中心化后的核）。full 按特征维分块，每块 float32 乘、float64 累加；fp32 matmul 用 highest 精度。"""
    n = z.shape[0]
    before = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        if kind != "full":
            x = torch.cat([readout(z[s:s+512], kind) for s in range(0, n, 512)]).double()
            x = (x-x.mean(0)).to(device)
            return (x@x.T).cpu().numpy()
        flat = z.reshape(n, -1)
        k = torch.zeros(n, n, dtype=torch.float64, device=device)
        for s in range(0, flat.shape[1], chunk):
            x = flat[:, s:s+chunk].to(device).double()
            x = (x-x.mean(0)).float()
            k += (x@x.T).double()
        return k.cpu().numpy()
    finally:
        torch.set_float32_matmul_precision(before)


def concat_gram(k_dup: np.ndarray, n_traj: int) -> np.ndarray:
    """dup_concat 的核：样本 (i,t) 的特征是 [dup(t−1帧), dup(t帧)]，所以 K = K_dup[前,前] + K_dup[后,后]。
    k_dup 的行是 n_traj 条轨迹 × 34 帧，按轨迹排列。"""
    prev, cur = sample_rows("dup_concat")
    a = (np.arange(n_traj)[:, None]*FRAMES+prev[None]).ravel()
    b = (np.arange(n_traj)[:, None]*FRAMES+cur[None]).ravel()
    return k_dup[np.ix_(a, a)]+k_dup[np.ix_(b, b)]


def select_rows(k: np.ndarray, rows_per_traj: int, rows: np.ndarray, n_traj: int) -> np.ndarray:
    idx = (np.arange(n_traj)[:, None]*rows_per_traj+rows[None]).ravel()
    return k[np.ix_(idx, idx)]


def centered(k, rows, cols, fit):
    """按 fit 样本的均值中心化后的核块：k̃(a,b) = k(a,b) − m(a) − m(b) + c，m(a) = mean_{i∈fit} k(a,i)，c = mean m(fit)。"""
    m = k[:, fit].mean(1); c = m[fit].mean()
    return k[np.ix_(rows, cols)]-m[rows][:, None]-m[cols][None, :]+c


def r2(y, pred):
    """逐列 R² = 1 − SSE/SST（SST 用这组样本自己的均值）。"""
    return 1.-((y-pred)**2).sum(0)/((y-y.mean(0))**2).sum(0)


def _ridge_fit(k, y, fit):
    kf = centered(k, fit, fit, fit)
    evals, vecs = np.linalg.eigh(kf)
    evals = np.clip(evals, 0., None)
    mean = y[fit].mean(0)
    proj = vecs.T@(y[fit]-mean)
    scale = max(float(np.trace(kf))/len(fit), 1e-300)
    return vecs, evals, proj, mean, scale


def _ridge_predict(k, fit, rows, model, lam):
    vecs, evals, proj, mean, _ = model
    alpha = vecs@(proj/(evals+lam)[:, None])
    return centered(k, rows, fit, fit)@alpha+mean


def ridge_cv(k: np.ndarray, y: np.ndarray, trajectory_of_row: np.ndarray, folds=FOLDS) -> dict:
    """对偶岭回归，按轨迹分折（轨迹编号 mod folds）。第 f 折测试、第 f+1 折选 λ、其余训练；每个目标各选 λ，
    再在训练 + 选 λ 两折上重拟合，报测试折 R²。λ = trace(K̃)/n · 10^e，e ∈ LAMBDA_EXPONENTS。
    val_r2_per_fold 是每折选 λ 那一折上、所选 λ 的 R²（只用训练折拟合），供 decide() 在不看测试折的情况下挑读出。"""
    fold = np.asarray(trajectory_of_row) % folds
    per_fold, lambdas, at_edge, val_scores = [], [], [], []
    for f in range(folds):
        test = np.flatnonzero(fold == f); val = np.flatnonzero(fold == (f+1) % folds)
        train = np.flatnonzero((fold != f) & (fold != (f+1) % folds))
        if not len(test) or not len(val) or not len(train):
            raise ValueError("每折都要有轨迹：轨迹数至少等于折数")
        model = _ridge_fit(k, y, train)
        scores = np.stack([r2(y[val], _ridge_predict(k, train, val, model, model[4]*10.**e)) for e in LAMBDA_EXPONENTS])
        best = scores.argmax(0)
        val_scores.append(scores.max(0))
        refit = np.concatenate((train, val)); model = _ridge_fit(k, y, refit)
        pred = np.stack([_ridge_predict(k, refit, test, model, model[4]*10.**LAMBDA_EXPONENTS[b])[:, j]
                         for j, b in enumerate(best)], -1)
        per_fold.append(r2(y[test], pred))
        lambdas.append([float(LAMBDA_EXPONENTS[b]) for b in best])
        at_edge.append([bool(b in (0, len(LAMBDA_EXPONENTS)-1)) for b in best])
    per_fold = np.array(per_fold)
    return dict(r2_mean=per_fold.mean(0).tolist(), r2_min=per_fold.min(0).tolist(), r2_per_fold=per_fold.tolist(),
                val_r2_per_fold=np.array(val_scores).tolist(), lambda_exponent=lambdas, lambda_at_grid_edge=at_edge)


def decide(results: dict) -> dict:
    """results[mode][readout] 是 ridge_cv 的输出（目标按 TARGETS 排列）→ 判据（模块说明第 5 条）。

    嵌套选读出：外层折 f 只看 pair 在第 f 折的 val_r2_per_fold（选 λ 那一折）挑 θ̇ 最高的读出（并列取 READOUTS
    里靠前的），再取该读出上 pair、dup 第 f 折的测试 R²。测试折不参与挑选，所以判据数值没有选择偏差。"""
    j = {t: i for i, t in enumerate(TARGETS)}
    rows = []
    for ro in READOUTS:
        pair = results["pair"][ro]["r2_mean"]; dup = results["dup"][ro]["r2_mean"]
        rows.append(dict(readout=ro, pair_omega=pair[j["omega"]], dup_omega=dup[j["omega"]],
                         gain=pair[j["omega"]]-dup[j["omega"]],
                         position_readable=bool(min(dup[j["sin_theta"]], dup[j["cos_theta"]]) >= POSITION_R2)))
    folds = len(results["pair"][READOUTS[0]]["r2_per_fold"])
    per_fold = []
    for f in range(folds):
        ro = max(READOUTS, key=lambda r: (results["pair"][r]["val_r2_per_fold"][f][j["omega"]], -READOUTS.index(r)))
        pair = results["pair"][ro]["r2_per_fold"][f]; dup = results["dup"][ro]["r2_per_fold"][f]
        per_fold.append(dict(fold=f, readout=ro, pair_val_omega=results["pair"][ro]["val_r2_per_fold"][f][j["omega"]],
                             pair_omega=pair[j["omega"]], dup_omega=dup[j["omega"]],
                             dup_sin_theta=dup[j["sin_theta"]], dup_cos_theta=dup[j["cos_theta"]]))
    mean = lambda key: float(np.mean([r[key] for r in per_fold]))
    pair_omega, dup_omega = mean("pair_omega"), mean("dup_omega")
    position = bool(min(mean("dup_sin_theta"), mean("dup_cos_theta")) >= POSITION_R2)
    gain = pair_omega-dup_omega
    adopt = bool(position and pair_omega >= ADOPT_R2 and gain >= ADOPT_GAIN)
    return dict(rule=dict(adopt_r2=ADOPT_R2, adopt_gain=ADOPT_GAIN, position_r2=POSITION_R2, target="omega",
                          readout_selection="nested: per outer fold, highest pair omega R2 on that fold's lambda-selection fold"),
                selected=dict(per_fold=per_fold, pair_omega=pair_omega, dup_omega=dup_omega, gain=gain,
                              position_readable=position),
                selected_readouts=[r["readout"] for r in per_fold], by_readout=rows,
                position_readable=position, adopt_two_frame=adopt)
