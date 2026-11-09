"""v7 可调数据集：在基础场景上叠加的“调节项”（导师 11/3：数据集作为贡献，能调节的地方更多）。

一个数据集 = 场景（系统）× 调节项 × 条数。调节项写成短字符串，例如
    double_well~gain=2,axes=x:2000      控制增益 2、只能推 x 方向（欠驱动）
    landscape~shape=ring:2000           势能地形族里的环形槽
    van_der_pol~mu=1.5,color=white:200  物理参数留出 + 外观留出
空调节项就是基础场景，与 v6/v7 试跑逐位一致。

调节项分四类：
- 物理参数（每个场景自己的，见 PARAMS）：如双势阱耦合与倾斜、Van der Pol 的 μ、粒子数、扩散系数；
- 小球控制：gain 增益、axes 可推方向、segments 控制切换长度、zero 零动作比例、noise 执行噪声、delay 执行延迟；
- 观察：sigma 光点大小、color 颜色、background 背景、distractors 干扰物个数、pixel_noise 画面噪声；
- 场景族本身（landscape 的 shape）。

模型看到的始终是“下达的”动作；执行噪声与延迟只作用在物理里（模型看不到），用来测鲁棒性。
所有调节项都保持一阶、画面能看全状态（模型只看当前一帧）。
随机流：调节项不改变初始状态和动作抽样的随机种子（除非调节项本身改的是动作抽样），因此同一编号的轨迹在不同调节项下可以配对比较。
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
}
SHAPES = ("multi_well", "ring", "periodic", "obstacles")
COLORS = {"orange": (230., 150., 50.), "white": (235., 235., 235.), "green": (80., 200., 90.), "blue": (70., 120., 235.)}
BACKGROUNDS = ("grid", "plain", "checker", "noise")
SEGMENT_SETS = {"148": (1, 4, 8), "1": (1,), "2": (2,), "4": (4,), "8": (8,)}


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

    def param(self, system: str, name: str):
        default = PARAMS[system][name][0]
        return dict(self.params).get(name, default)

    @property
    def is_default(self) -> bool:
        return self == Variant()

    def changes_controls(self) -> bool:
        return (self.axes, self.segments, self.zero) != ("xy", "148", .2)

    def changes_execution(self) -> bool:
        return self.gain != 1. or self.noise != 0. or self.delay != 0

    def changes_observation(self) -> bool:
        return (self.sigma, self.color, self.background, self.distractors, self.pixel_noise) != (None, "orange", "grid", 0, 0.)


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
        if name in ("gain", "zero", "noise", "sigma", "pixel_noise"):
            out[name] = float(value)
        elif name in ("delay", "distractors"):
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
              (0. <= v.pixel_noise <= 40., "pixel_noise ∈ [0,40]")]
    for ok, message in checks:
        if not ok:
            raise ValueError("调节项越界：" + message)
    for x in (v.gain, v.zero, v.noise, v.pixel_noise) + ((v.sigma,) if v.sigma is not None else ()):
        if not math.isfinite(x):
            raise ValueError("调节项必须是有限数")
    if system in ("advection_diffusion", "gray_scott") and (v.sigma is not None or v.distractors):
        raise ValueError(f"{system} 没有光点：sigma/distractors 不适用")


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
