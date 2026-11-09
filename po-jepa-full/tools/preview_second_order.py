"""S23a：二阶场景的样例图（每行一条轨迹，列为 tick −1、0、8、16、24、32），只供肉眼检查，不进实验。

用法：python tools/preview_second_order.py <输出目录>
动作按 v6 规则（1/4/8 段、20% 零动作）用独立的随机数生成，只供预览；正式数据集的动作在 S23b 登记后
由 systems.commanded_controls 给出。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from po_jepa import second_order  # noqa: E402
from po_jepa.variants import Variant  # noqa: E402


def controls(seed):
    g = np.random.default_rng(seed); u = np.zeros((32, 2)); t = 0
    while t < 32:
        n = int(g.choice([1, 4, 8])); u[t:t+n] = 0. if g.random() < .2 else g.uniform(-.5, .5, 2); t += n
    return u


ROWS = {
    "pendulum": [(0, {}), (1, {"g_over_l": 16., "damping": 0.}), (2, {"g_over_l": 4., "damping": 1.})],
    "cart_pole": [(0, {}), (1, {"length": .6, "mass_ratio": .1}), (2, {"length": 1.4, "mass_ratio": 1., "friction": 0.})],
    "spring_chain": [(0, {}), (1, {"n": 6, "stiffness": 9., "damping": 0.}), (2, {"n": 2, "stiffness": 25.})],
    "bouncing_balls": [(0, {}), (1, {"n": 5, "restitution": .5}), (2, {"n": 1, "restitution": 1., "drag": 0.})],
    "wave": [(0, {}), (1, {"speed": .3, "damping": 0.}), (2, {"speed": .08, "load": .4})],
}
COLUMNS = (0, 1, 9, 17, 25, 33)          # 下标 f 对应 tick f−1

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for name, rows in ROWS.items():
    system = second_order.SYSTEMS[name]; strips = []
    for tid, params in rows:
        _, frames = system.frames(tid, controls(tid), params, Variant())
        strips.append(np.concatenate([np.pad(frames[f], ((2, 2), (2, 2), (0, 0))) for f in COLUMNS], 1))
    Image.fromarray(np.concatenate(strips, 0)).save(out/f"{name}.png")
    print(out/f"{name}.png", [p for _, p in rows])
