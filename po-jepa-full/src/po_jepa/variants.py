"""v7 可调数据集：在基础场景上叠加的“调节项”（导师 11/3：数据集作为贡献，能调节的地方更多）。

一个数据集 = 场景（系统）× 调节项 × 条数。调节项写成短字符串，例如
    double_well~gain=2,axes=x:2000      控制增益 2、只能推 x 方向（欠驱动）
    landscape~shape=ring:2000           势能地形族里的环形槽
    van_der_pol~mu=1.5,color=white:200  物理参数留出 + 外观留出
空调节项就是基础场景，与 v6/v7 试跑逐位一致。

调节项分五类：
- 物理参数（每个场景自己的，见 PARAMS）：如双势阱耦合与倾斜、Van der Pol 的 μ、粒子数、扩散系数；
  S21a 加通用的 process_noise 过程噪声（随机微分方程，只用于粒子类 ODE 场景）；
- 小球控制：gain 增益、axes 可推方向、segments 控制切换长度、zero 零动作比例、noise 执行噪声、delay 执行延迟；
  S21a 加 saturation 执行动作饱和；
- 观察：sigma 光点大小、color 颜色、background 背景、distractors 干扰物个数、pixel_noise 画面噪声；
  S21a 加 occluders 遮挡块、movers 会动的干扰光点；
- 数据采集（S21a）：policy 采集策略 random / goal / mixed；
- 场景族本身（landscape 的 shape）；
- 时间（S24a）：horizon 轨迹长度（控制 tick 数），32 = 原来的 33 帧，64、128 是长时程（65、129 帧）。
  S24b 时间采样：fps 帧率、jitter 帧时间抖动、exposure 曝光（运动模糊）、switch_offset 控制在两帧之间切换。
  这四项不改物理（switch_offset 除外，它改的是下达动作的切换时刻），只改在哪些时刻、怎样取画面（po_jepa.timing）。
- 空间（S24c）：resolution 输出分辨率、zoom 缩小、pan 相机平移、drift 相机缓慢漂移（po_jepa.camera）。只改画面，不改物理；
  相机视野始终包含整块画布（画布外是深色的 SURROUND），所以画面仍能看全状态，相机位置也能从一帧看出来。

模型看到的始终是“下达的”动作；执行噪声、延迟和饱和只作用在物理里（模型看不到），用来测鲁棒性。
除 occluders（被遮住时状态看不到，有意做成部分可观测）外，所有调节项都保持一阶、画面能看全状态（模型只看当前一帧）。
S23b 的五个二阶场景（SECOND_ORDER）本身不是一阶：画面只给位置，速度要由相邻两帧给出；调节项不改变这一点。
随机流：调节项不改变初始状态和动作抽样的随机种子（除非调节项本身改的是动作抽样：axes、segments、zero、policy），
因此同一编号的轨迹在不同调节项下可以配对比较。流号：0 初值、1 动作、2 预处理、3 静止干扰光点、4 像素噪声、
5 执行噪声；S21a 新增 6 过程噪声、7 遮挡块、8 会动的干扰光点、9 采集策略（目标点、探索噪声、mixed 的选择）。
S24a horizon 不用新的随机流：每个流按 tick 顺序抽，长轨迹只是接着抽下去，所以前 32 个 tick 的动作、噪声和画面
与 horizon=32（其余调节项相同）逐位相同（前缀一致，见 docs/segments/S24a.md §2）。
S24b 新增流 10 = 帧时间抖动，流 11 = 不在整 tick 上的帧的像素噪声（按帧号另起，见 po_jepa.timing）。
S24c 新增流 12 = 相机（pan 的方向角）。
"""
from dataclasses import dataclass, field, fields
import math

# 每个场景可调的物理参数：名字 → (默认值, 允许范围)
PARAMS = {
    "double_well": {"coupling": (.25, (0., .6)), "tilt": (0., (-.3, .3))},
    "van_der_pol": {"mu": (.5, (.1, 1.5))},
    "multi_particle": {"n": (3, (1, 5)), "repulsion": (1.5, (0., 3.)), "trap": (.5, (.2, 1.))},
    "advection_diffusion": {"diffusion": (2e-4, (0., 2e-3)), "flow": (.4, (.1, 1.)), "blobs": (3, (1, 5))},
    "landscape": {"depth": (1., (.25, 3.))},
    # S20b：v8 新场景（物理与渲染在 po_jepa.new_systems，那里的 PARAMS 就是这两项；范围的依据见 docs/segments/S20a.md §2）
    "gray_scott": {"feed": (.04, (.035, .05)), "kill": (.06, (.0595, .063)), "inject": (.05, (0., .2))},
    "push": {"mu_s": (.3, (0., 1.)), "slope": (0., (0., .6)), "slope_dir": (0., (-180., 180.)), "size": (.8, (.4, 1.2))},
    # S23b：二阶场景（物理与渲染在 po_jepa.second_order，那里的 PARAMS 就是这五项；范围的依据见 docs/segments/S23a.md §3）
    "pendulum": {"g_over_l": (9.81, (4., 16.)), "damping": (.2, (0., 1.))},
    "bouncing_balls": {"n": (3, (1, 5)), "restitution": (.9, (.5, 1.)), "drag": (.2, (0., 1.))},
    "wave": {"speed": (.15, (.08, .3)), "damping": (.1, (0., .5)), "load": (.25, (0., .4))},
    "cart_pole": {"length": (1., (.6, 1.4)), "mass_ratio": (.3, (.1, 1.)), "friction": (.5, (0., 1.))},
    "spring_chain": {"n": (4, (2, 6)), "stiffness": (16., (9., 25.)), "damping": (.1, (0., .5))},
}
SHAPES = ("multi_well", "ring", "periodic", "obstacles")
COLORS = {"orange": (230., 150., 50.), "white": (235., 235., 235.), "green": (80., 200., 90.), "blue": (70., 120., 235.)}
BACKGROUNDS = ("grid", "plain", "checker", "noise")
SEGMENT_SETS = {"148": (1, 4, 8), "1": (1,), "2": (2,), "4": (4,), "8": (8,)}
POLICIES = ("random", "goal", "mixed")
HORIZONS = (32, 64, 128)       # S24a：轨迹的控制 tick 数（帧数 = horizon+1；二阶场景另多 t = −1 一帧）
# S24b 时间采样（docs/segments/S24b.md）
FPS = (2, 4, 8, 16)             # 帧率；4 = 原来的每 tick 一帧。帧数 = horizon·fps/4 + 1
SWITCH_OFFSETS = (0., .25, .5, .75)   # 下达动作的段边界整体往后挪多少 tick（1/4 tick 的整数倍）
JITTER_MAX = .4                 # 帧时间抖动上限（帧间隔的比例；< 0.5 保证帧时间严格递增）
# S24c 空间采样（docs/segments/S24c.md）
RESOLUTIONS = (192, 256, 320, 384)   # 输出边长（像素）；视野不变，V-JEPA 2 的 16 像素 patch 对应 12/16/20/24 网格
ZOOM_MIN = .5                   # zoom ∈ [0.5, 1]：只缩小（视野始终包含整块画布，画面能看全状态）
DRIFT_MAX = .5                  # rad/s：相机绕画布中心转的角速度上限（一圈 ≥ 12.6 s ≈ 50 tick，“缓慢”）
# S21a：只对 ODE 粒子类场景（状态是 RK4 积分的向量、受控点是状态前两维）成立的调节项不适用的场景
FIELD_SYSTEMS = ("advection_diffusion", "gray_scott", "wave")   # 场方程：没有光点（wave 是 S23b 加的）
# S23b：二阶场景（一帧看不到速度；数据集仍是 33 帧，34 帧见 systems.SecondOrderSystem.frames_with_prev）
SECOND_ORDER = ("pendulum", "bouncing_balls", "wave", "cart_pole", "spring_chain")
NOT_ODE = ("advection_diffusion", "gray_scott", "push")+SECOND_ORDER   # 不走 System.variant_trajectory 的 RK4（过程噪声）
NO_SIGMA = ("bouncing_balls",)                                  # S23b：有光点但不是高斯光点（有物理半径的圆盘），sigma 不适用


@dataclass(frozen=True)
class Variant:
    # 物理参数覆盖（场景相关），按名字排序存成元组，保证可哈希、可复现
    params: tuple = ()
    shape: str | None = None          # 只用于 landscape
    # 小球控制
    gain: float = 1.
    axes: str = "xy"                  # xy / x / y：只能推一个方向 = 欠驱动
    segments: str = "148"             # 控制段长度集合（tick）：148 = v6 的 1/4/8 随机；1/2/4/8 = 固定切换周期
    zero: float = .2                  # 零动作段比例
    noise: float = 0.                 # 执行噪声标准差（与 u 同单位，模型看不到）
    delay: int = 0                    # 执行延迟（tick，模型看不到）
    # 观察
    sigma: float | None = None        # 光点高斯 σ（像素），None = 场景默认
    color: str = "orange"
    background: str = "grid"
    distractors: int = 0              # 静止干扰光点个数（灰色，不受控制）
    pixel_noise: float = 0.           # 每帧独立高斯像素噪声标准差（0–255 刻度）
    # S21a 通用调节项（默认值 = 不起作用，基础场景和已有调节项逐位不变）
    process_noise: float = 0.         # 过程噪声 σ（状态单位/√秒）：dq = f dt + σ dW，随机流 6，只用于 ODE 粒子类场景
    saturation: float | None = None   # 执行动作按分量截断到 [−s, s]（在增益、执行噪声之后；模型看不到），None = 不截断
    occluders: int = 0                # 静止不透明遮挡块个数（48×48 像素，每条轨迹位置不同，随机流 7）
    movers: int = 0                   # 会动的干扰光点个数（绕画面中心匀速转，随机流 8；只看一帧就能推出下一步）
    policy: str = "random"            # 采集策略：random = v6 随机；goal = 朝目标（开环推算）；mixed = 每段各半
    # S24a 时间：轨迹长度（控制 tick 数）。32 = 原来的 33 帧；64、128 是长时程，前 33 帧与 32 逐位相同
    horizon: int = 32
    # S24b 时间采样（默认值 = 不起作用）。fps、jitter、exposure 只改取画面的时刻和方式，不改物理；switch_offset 改下达动作的切换时刻
    fps: int = 4                      # 帧率（帧/秒）：每 4/fps 个 tick 一帧
    jitter: float = 0.                # 帧时间抖动：中间各帧的时刻加 jitter·帧间隔·U(−1,1)（流 10），首尾两帧不动
    exposure: float = 0.              # 曝光时间（帧间隔的比例）：画面是 [t−E, t] 内等距若干时刻画面的平均（只用过去，不看未来）
    switch_offset: float = 0.         # 下达动作的段边界整体往后挪 switch_offset 个 tick（第一段变长、最后一段变短，取值不变）
    # S24c 空间采样（默认值 = 不起作用）。只改画面（po_jepa.camera），不改物理和动作
    resolution: int = 256             # 输出边长（像素），视野不变
    zoom: float = 1.                  # 每个世界单位的像素数相对默认的比例（≤ 1：缩小，画布四周露出 SURROUND）
    pan: float = 0.                   # 相机中心偏离画布中心的距离 / 缩小留出的余量（方向每条轨迹随机，流 12）；要 zoom < 1
    drift: float = 0.                 # 相机中心绕画布中心逆时针转的角速度（rad/s），半径就是 pan 给的偏移；要 pan > 0

    def param(self, system: str, name: str):
        default = PARAMS[system][name][0]
        return dict(self.params).get(name, default)

    @property
    def is_default(self) -> bool:
        return self == Variant()

    def changes_controls(self) -> bool:
        return (self.axes, self.segments, self.zero, self.policy) != ("xy", "148", .2, "random")

    def changes_execution(self) -> bool:
        return self.gain != 1. or self.noise != 0. or self.delay != 0 or self.saturation is not None

    def changes_timing(self) -> bool:
        """S24b：时间采样调节项（帧率、抖动、曝光、两帧之间切换）是否起作用。起作用时画面走 po_jepa.timing。"""
        return (self.fps, self.jitter, self.exposure, self.switch_offset) != (4, 0., 0., 0.)

    def changes_space(self) -> bool:
        """S24c：空间采样调节项（分辨率、缩放、相机平移和漂移）是否起作用。起作用时画面走 po_jepa.timing + po_jepa.camera。"""
        return (self.resolution, self.zoom, self.pan, self.drift) != (256, 1., 0., 0.)

    def resampled(self) -> bool:
        """画面是否走 po_jepa.timing（时间采样或空间采样起作用）。两者都不起作用时走各场景原来的画法（逐位不变）。"""
        return self.changes_timing() or self.changes_space()

    def changes_observation(self) -> bool:
        return ((self.sigma, self.color, self.background, self.distractors, self.pixel_noise, self.occluders, self.movers)
                != (None, "orange", "grid", 0, 0., 0, 0))


_ALIASES = {f.name for f in fields(Variant)} - {"params"}


def parse(system: str, text: str) -> Variant:
    """"gain=2,axes=x,mu=1.5" → Variant；未知名字、越界值直接报错。"""
    if system not in PARAMS:
        raise ValueError(f"unknown system {system!r}")
    if not text:
        if system == "landscape":
            raise ValueError("landscape 需要 shape=…（" + "/".join(SHAPES) + "）")
        return Variant()
    kw, params = {}, {}
    for item in text.split(","):
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise ValueError(f"调节项格式是 name=value，收到 {item!r}")
        if name in kw or name in params:
            raise ValueError(f"调节项 {name} 重复")
        if name in PARAMS[system]:
            default, (lo, hi) = PARAMS[system][name]
            v = int(value) if isinstance(default, int) else float(value)
            if not lo <= v <= hi:
                raise ValueError(f"{system} 的 {name} 必须在 [{lo},{hi}]")
            params[name] = v
        elif name in _ALIASES:
            kw[name] = value
        else:
            raise ValueError(f"{system} 没有调节项 {name}；可用：{sorted(_ALIASES | set(PARAMS[system]))}")
    v = Variant(params=tuple(sorted(params.items())), **_convert(kw))
    validate(system, v)
    return v


def _convert(kw):
    out = {}
    for name, value in kw.items():
        if name in ("gain", "zero", "noise", "sigma", "pixel_noise", "process_noise", "saturation", "jitter", "exposure",
                    "switch_offset", "zoom", "pan", "drift"):
            out[name] = float(value)
        elif name in ("delay", "distractors", "occluders", "movers", "horizon", "fps", "resolution"):
            out[name] = int(value)
        else:
            out[name] = value
    return out


def validate(system: str, v: Variant) -> None:
    if system == "landscape":
        if v.shape not in SHAPES:
            raise ValueError("landscape 的 shape 必须是 " + "/".join(SHAPES))
    elif v.shape is not None:
        raise ValueError("shape 只用于 landscape")
    checks = [(.25 <= v.gain <= 4., "gain ∈ [0.25,4]"), (v.axes in ("xy", "x", "y"), "axes ∈ xy/x/y"),
              (v.segments in SEGMENT_SETS, f"segments ∈ {list(SEGMENT_SETS)}"), (0. <= v.zero <= .8, "zero ∈ [0,0.8]"),
              (0. <= v.noise <= .5, "noise ∈ [0,0.5]"), (0 <= v.delay <= 4, "delay ∈ 0–4 tick"),
              (v.sigma is None or 3. <= v.sigma <= 16., "sigma ∈ [3,16] 像素"), (v.color in COLORS, f"color ∈ {list(COLORS)}"),
              (v.background in BACKGROUNDS, f"background ∈ {BACKGROUNDS}"), (0 <= v.distractors <= 6, "distractors ∈ 0–6"),
              (0. <= v.pixel_noise <= 40., "pixel_noise ∈ [0,40]"),
              (0. <= v.process_noise <= .3, "process_noise ∈ [0,0.3]"),
              (v.saturation is None or .05 <= v.saturation <= 2., "saturation ∈ [0.05,2]"),
              (0 <= v.occluders <= 3, "occluders ∈ 0–3"), (0 <= v.movers <= 4, "movers ∈ 0–4"),
              (v.policy in POLICIES, f"policy ∈ {POLICIES}"),
              (type(v.horizon) is int and v.horizon in HORIZONS, f"horizon ∈ {HORIZONS}"),
              (type(v.fps) is int and v.fps in FPS, f"fps ∈ {FPS}"),
              (0. <= v.jitter <= JITTER_MAX, f"jitter ∈ [0,{JITTER_MAX}]"), (0. <= v.exposure <= 1., "exposure ∈ [0,1]"),
              (v.switch_offset in SWITCH_OFFSETS, f"switch_offset ∈ {SWITCH_OFFSETS}"),
              (type(v.resolution) is int and v.resolution in RESOLUTIONS, f"resolution ∈ {RESOLUTIONS}"),
              (ZOOM_MIN <= v.zoom <= 1., f"zoom ∈ [{ZOOM_MIN},1]"), (0. <= v.pan <= 1., "pan ∈ [0,1]"),
              (0. <= v.drift <= DRIFT_MAX, f"drift ∈ [0,{DRIFT_MAX}] rad/s")]
    for ok, message in checks:
        if not ok:
            raise ValueError("调节项越界：" + message)
    for x in ((v.gain, v.zero, v.noise, v.pixel_noise, v.process_noise, v.jitter, v.exposure, v.switch_offset, v.zoom, v.pan,
               v.drift) + ((v.sigma,) if v.sigma is not None else ())
              + ((v.saturation,) if v.saturation is not None else ())):
        if not math.isfinite(x):
            raise ValueError("调节项必须是有限数")
    if v.pan and v.zoom == 1.:
        raise ValueError("pan 是在缩小留出的余量里平移：要 zoom < 1（zoom = 1 时视野正好是画布，平移会切掉画面）")
    if v.drift and not v.pan:
        raise ValueError("drift 是相机中心绕画布中心转，半径由 pan 给：要 pan > 0")
    if system in FIELD_SYSTEMS and (v.sigma is not None or v.distractors or v.movers):
        raise ValueError(f"{system} 没有光点：sigma/distractors/movers 不适用")
    if system in FIELD_SYSTEMS and v.policy != "random":
        raise ValueError(f"{system} 的动作是整场的流速或注入点速度，没有要推向目标的受控点：policy 只能是 random")
    if system in NOT_ODE and v.process_noise:
        raise ValueError(f"{system} 不是 RK4 积分的粒子类 ODE：process_noise 不适用")
    if system in NO_SIGMA and v.sigma is not None:
        raise ValueError(f"{system} 的球是有物理半径的圆盘，不是高斯光点：sigma 不适用")
    if system in SECOND_ORDER and v.policy != "random":
        # S21a 的 goal 策略按“下达动作 ∝ 受控点速度”做开环推算（System.policy_anchor 的 K）；二阶场景的动作是力，
        # 这个推算不成立（wave 的激振源是速度控制，但它和 gray_scott 的注入点一样没有要推到目标的受控点）
        raise ValueError(f"{system} 是二阶场景，动作是力：S21a 的开环朝目标推算不适用，policy 只能是 random")
    if system == "cart_pole" and v.axes == "y":
        raise ValueError("cart_pole 只受水平推力（u_y 不起作用）：axes=y 等于没有控制")


# S21a 覆盖表：每个通用调节项取一个非默认的代表值，按 validate 判定在哪些场景可用（DATASET.md 的覆盖表由它生成）
GENERAL_SAMPLES = {"gain": "2", "axes": "x", "segments": "2", "zero": "0.5", "noise": "0.1", "delay": "1",
                   "saturation": "0.3", "policy": "goal", "process_noise": "0.1", "sigma": "4", "color": "white",
                   "background": "checker", "distractors": "2", "movers": "2", "occluders": "2", "pixel_noise": "10",
                   "horizon": "128", "fps": "8", "jitter": "0.2", "exposure": "0.5", "switch_offset": "0.5",
                   "resolution": "384", "zoom": "0.75", "pan": "zoom=0.75,pan=0.5", "drift": "zoom=0.75,pan=0.5,drift=0.25"}
KNOB_GROUPS = {"控制": ("gain", "axes", "segments", "zero", "noise", "delay", "saturation"), "数据采集": ("policy",),
               "物理": ("process_noise",),
               "观察": ("sigma", "color", "background", "distractors", "movers", "occluders", "pixel_noise"),
               "时间": ("horizon", "fps", "jitter", "exposure", "switch_offset"),
               "空间": ("resolution", "zoom", "pan", "drift")}


def coverage() -> dict:
    """{通用调节项: 可用场景元组}，按 datasets.SYSTEMS 的顺序；landscape 加上 shape 再判。"""
    from . import datasets
    out = {}
    for name, value in GENERAL_SAMPLES.items():
        ok = []
        for system in datasets.SYSTEMS:
            text = (f"shape={SHAPES[0]}," if system == "landscape" else "") + (value if "=" in value else f"{name}={value}")
            try:
                parse(system, text)
            except ValueError:
                continue
            ok.append(system)
        out[name] = tuple(ok)
    return out


def format(system: str, v: Variant) -> str:
    """规范字符串（按名字排序，只写与默认不同的项）；parse(format(v)) == v。"""
    base = Variant()
    items = dict(v.params)
    for f in fields(Variant):
        if f.name == "params":
            continue
        value = getattr(v, f.name)
        if value != getattr(base, f.name):
            items[f.name] = value
    def show(x):
        if isinstance(x, float):
            return repr(x).rstrip("0").rstrip(".") if "e" not in repr(x) else repr(x)
        return str(x)
    return ",".join(f"{k}={show(items[k])}" for k in sorted(items))
