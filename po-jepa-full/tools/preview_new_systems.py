"""S20a：新增场景的样例图（每行一条轨迹，列为第 0/8/16/24/32 帧），只供肉眼检查，不进实验。

用法：python tools/preview_new_systems.py <输出目录>
动作按 v6 规则（1/4/8 段、20% 零动作）用独立的随机数生成，只供预览；正式数据集的动作在 S20b 登记后
由 systems.commanded_controls 给出。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from po_jepa import new_systems  # noqa: E402
from po_jepa.variants import Variant  # noqa: E402


def controls(seed):
    g = np.random.default_rng(seed); u = np.zeros((32, 2)); t = 0
    while t < 32:
        n = int(g.choice([1, 4, 8])); u[t:t+n] = 0. if g.random() < .2 else g.uniform(-.5, .5, 2); t += n
    return u


ROWS = {
    "gray_scott": [(0, {}), (1, {"feed": .035, "kill": .06}), (2, {"feed": .05, "kill": .063})],
    "push": [(0, {}), (3, {"mu_s": .1}), (5, {"slope": .5, "slope_dir": 45.})],
}

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for name, rows in ROWS.items():
    system = new_systems.SYSTEMS[name]; strips = []
    for tid, params in rows:
        _, frames = system.frames(tid, controls(tid), params, Variant())
        strips.append(np.concatenate([np.pad(frames[t], ((2, 2), (2, 2), (0, 0))) for t in (0, 8, 16, 24, 32)], 1))
    Image.fromarray(np.concatenate(strips, 0)).save(out/f"{name}.png")
    print(out/f"{name}.png", [p for _, p in rows])
