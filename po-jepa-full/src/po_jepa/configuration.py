"""S25a：二阶场景在动作探针（tools/probe_actions.py）和规划（tools/plan_v7.py）里用的“构形”：画面上看得到的位置。

二阶场景的隐藏状态里有速度，但一帧画面只给位置（速度要由相邻两帧给出），所以两处都只比位置：
- 探针的 *_px 列（motion）：整个构形的 L2 × 每单位像素数（PX），与 multi_particle 用全部粒子的位置同一口径；
- 规划的状态摘要（controlled）：受控物体的位置，与已有粒子类场景“受控小球位置”同一口径；摆和小车倒立摆的摆锤
  是要摆到目标的东西，所以 pendulum 用摆锤位置，cart_pole 用小车和摆锤的位置。

| 场景 | 构形（世界单位） | 规划摘要 | 每单位像素 |
|---|---|---|---|
| pendulum | 摆锤 (sin θ, −cos θ)（摆长为单位） | 同左 | 80（摆杆 80 像素） |
| cart_pole | (x, x + l·sin θ, l·cos θ)：小车、摆锤 | 同左 | 36 |
| spring_chain | n 个质点的位置 [2n] | 第 0 个质点（受控） | 48 |
| bouncing_balls | n 个球的位置 [2n] | 第 0 个球（受控） | 48 |
| wave | u 的 16×16 平均池化场（256 维，不含 u_t 和激振源） | 同左 | 无（场，按场的 L2） |

wave 与 advection_diffusion、gray_scott 同一类：*_px 列是池化场的 L2（不是像素），规划的目标门槛按场的范数的 5%。
激振源的位置不计入（与 gray_scott 的注入点相同）。

S25b5（Jerry 2026-11-25 定：规划成功判据并入速度、读出加速度）：两帧编码（tubelet=pair）的特征看得到速度，所以另给
构形的时间导数 velocity（世界单位/秒）和规划摘要的时间导数 controlled_velocity，都由隐藏状态解析算出（不做差分）：
| 场景 | 构形速度 | 隐藏状态 |
|---|---|---|
| pendulum | θ̇·(cos θ, sin θ) | (θ, θ̇) |
| cart_pole | (ẋ, ẋ + l·θ̇·cos θ, −l·θ̇·sin θ) | (x, ẋ, θ, θ̇) |
| spring_chain、bouncing_balls | 质点/球的速度 [2n]（规划摘要取第 0 个） | 位置 [2n] + 速度 [2n] |
| wave | u_t 的 16×16 平均池化场（池化是线性的，等于池化 u 的时间导数） | [u 池化 256, u_t 池化 256, p 2] |
velocity 的单位和 configuration 相同再除以秒：乘 PX 是像素/秒，wave 是场的 L2/秒。

输入都是 frames() 的隐藏状态（一维向量；wave 是 514 维池化隐藏状态，不是可续算的整场）。params 是 dict(v.params)。
"""
import numpy as np

from . import second_order, variants

PX = {"pendulum": 80., "cart_pole": 36., "spring_chain": 48., "bouncing_balls": 48., "wave": None}
SUMMARY_NAME = {"pendulum": "摆锤位置 (sin θ, −cos θ)", "cart_pole": "小车和摆锤位置 (x, x + l·sin θ, l·cos θ)",
                "spring_chain": "受控质点（第 0 个）位置", "bouncing_balls": "受控球（第 0 个）位置",
                "wave": "u 的 16×16 池化场（256 维）"}
assert set(PX) == set(SUMMARY_NAME) == set(variants.SECOND_ORDER)
assert second_order.Pendulum.ROD_PX == PX["pendulum"] and second_order.CartPole.SCALE == PX["cart_pole"]
assert second_order.SpringChain.SCALE == PX["spring_chain"] and second_order.BouncingBalls.SCALE == PX["bouncing_balls"]


def _hidden(name, hidden, params):
    h = np.asarray(hidden, dtype=np.float64)
    if name not in PX:
        raise ValueError(f"{name} 不是二阶场景")
    dims = {"pendulum": 2, "cart_pole": 4, "wave": 2*256+2}.get(name)
    if dims is None:
        dims = 4*second_order.param(name, params, "n")
    if h.shape != (dims,):
        raise ValueError(f"{name} 的隐藏状态必须是 [{dims}]，现在是 {list(h.shape)}")
    return h


def configuration(name, hidden, params) -> np.ndarray:
    """隐藏状态 → 构形（见文件头的表）。"""
    h = _hidden(name, hidden, params)
    if name == "pendulum":
        return np.array([np.sin(h[0]), -np.cos(h[0])])
    if name == "cart_pole":
        l = second_order.param(name, params, "length")
        return np.array([h[0], h[0]+l*np.sin(h[2]), l*np.cos(h[2])])
    if name == "wave":
        return h[:256].copy()
    return h[:len(h)//2].copy()                     # spring_chain、bouncing_balls：位置 [n,2] 展平，后一半是速度


def controlled(name, hidden, params) -> np.ndarray:
    """隐藏状态 → 规划的状态摘要（见文件头的表）。"""
    c = configuration(name, hidden, params)
    return c[:2] if name in ("spring_chain", "bouncing_balls") else c


def motion(name, a, b, params) -> float:
    """探针的 *_px 距离：两个隐藏状态的构形之差的 L2 × PX（wave：池化场的 L2）。"""
    d = float(np.linalg.norm(configuration(name, a, params)-configuration(name, b, params)))
    return d if PX[name] is None else d*PX[name]


def velocity(name, hidden, params) -> np.ndarray:
    """S25b5：隐藏状态 → 构形的时间导数 d(configuration)/dt（文件头 S25b5 的表），形状与 configuration 相同。"""
    h = _hidden(name, hidden, params)
    if name == "pendulum":
        return h[1]*np.array([np.cos(h[0]), np.sin(h[0])])
    if name == "cart_pole":
        l = second_order.param(name, params, "length")
        return np.array([h[1], h[1]+l*h[3]*np.cos(h[2]), -l*h[3]*np.sin(h[2])])
    if name == "wave":
        return h[256:512].copy()
    return h[len(h)//2:].copy()                     # spring_chain、bouncing_balls：后一半是速度 [n,2] 展平


def controlled_velocity(name, hidden, params) -> np.ndarray:
    """S25b5：规划摘要（controlled）的时间导数。"""
    v = velocity(name, hidden, params)
    return v[:2] if name in ("spring_chain", "bouncing_balls") else v
