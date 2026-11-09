"""S21b 评测协议（v8）：训练只在基础场景上，测试按协议换调节项。写法参照 CausalWorld（Ahmed et al., ICLR 2021）：
每个协议只动一类变量，最后一个协议把几类叠起来；每个协议在同一批测试轨迹上评。

这里是协议的唯一来源：tools/make_v8_jobs.py 的分布外评测任务、DATASET.md 的协议表（tools/protocol_table.py 生成）、
测试都从这里取。v7 的分布外测试表（tools/make_v7_jobs.py 的 TEST_VARIANTS，仿真补跑也在用）不改；它的 30 个调节项
正好是这里对应协议在 v7 五个场景上的“标准档”，所以 v7 已经编码的变体测试集可以直接复用。

强度三档：轻、标准、重。分布外评测（make_v8_jobs --stage variants）只评标准档；强度曲线（--stage strength）评轻档和重档。
有的协议只有两档（延迟只能取整数 tick；采集策略只有 mixed、goal 两种）。某个场景不适用的协议（variants.validate 会拒绝的，
例如场方程没有光点、push 不加过程噪声）在表里是空的，不评。
"""
from dataclasses import dataclass

from . import datasets, variants

LEVELS = ("轻", "标准", "重")
# v8 的 7 个基础场景（训练场景）：v7 的 5 个（landscape 取多势阱地形，与 v7 相同）+ S20 的两个新场景
SCENES = ("double_well", "van_der_pol", "multi_particle", "advection_diffusion", "landscape~shape=multi_well",
          "gray_scott", "push")


@dataclass(frozen=True)
class Protocol:
    id: str
    name: str
    group: str
    what: str
    # 场景（系统名）→ {档位: 调节项字符串}；同一协议在所有适用场景上用同一个通用调节项时，用 common 给出
    levels: dict


def _common(values: dict) -> dict:
    return {system: dict(values) for system in datasets.SYSTEMS}


PROTOCOLS = (
    Protocol("P1", "物理参数 A", "物理", "每个场景自己的第一个物理参数调大", {
        "double_well": {"轻": "coupling=0.4", "标准": "coupling=0.5", "重": "coupling=0.6"},
        "van_der_pol": {"轻": "mu=0.8", "标准": "mu=1", "重": "mu=1.5"},
        "multi_particle": {"轻": "repulsion=2", "标准": "repulsion=2.5", "重": "repulsion=3"},
        "advection_diffusion": {"轻": "diffusion=0.0005", "标准": "diffusion=0.001", "重": "diffusion=0.002"},
        "landscape": {"轻": "depth=1.5", "标准": "depth=2", "重": "depth=3"},
        "gray_scott": {"轻": "feed=0.043", "标准": "feed=0.046", "重": "feed=0.05"},
        "push": {"轻": "mu_s=0.45", "标准": "mu_s=0.6", "重": "mu_s=0.9"},
    }),
    Protocol("P2", "物理参数 B", "物理", "第二个物理参数（只有一个参数的场景把它往另一边调）", {
        "double_well": {"轻": "tilt=0.1", "标准": "tilt=0.2", "重": "tilt=0.3"},
        "van_der_pol": {"轻": "mu=0.35", "标准": "mu=0.2", "重": "mu=0.1"},
        "multi_particle": {"轻": "trap=0.65", "标准": "trap=0.8", "重": "trap=1"},
        "advection_diffusion": {"轻": "flow=0.6", "标准": "flow=0.8", "重": "flow=1"},
        "landscape": {"轻": "depth=0.75", "标准": "depth=0.5", "重": "depth=0.25"},
        "gray_scott": {"轻": "kill=0.061", "标准": "kill=0.062", "重": "kill=0.063"},
        "push": {"轻": "slope=0.1", "标准": "slope=0.2", "重": "slope=0.25"},
    }),
    Protocol("P3", "过程噪声", "物理", "dq = f dt + σ dW（只用于粒子类 ODE 场景）",
             _common({"轻": "process_noise=0.05", "标准": "process_noise=0.1", "重": "process_noise=0.2"})),
    Protocol("P4", "执行增益", "执行", "执行动作 = 增益 × 下达动作（模型看不到）",
             _common({"轻": "gain=1.5", "标准": "gain=2", "重": "gain=3"})),
    Protocol("P5", "执行延迟", "执行", "执行动作晚若干 tick（模型看不到）",
             _common({"标准": "delay=1", "重": "delay=2"})),
    Protocol("P6", "执行饱和", "执行", "执行动作按分量截断到 [−s, s]（模型看不到）",
             _common({"轻": "saturation=0.4", "标准": "saturation=0.3", "重": "saturation=0.15"})),
    Protocol("P7", "采集策略", "动作分布", "测试轨迹的动作换成朝目标（goal）或一半朝目标（mixed）",
             _common({"轻": "policy=mixed", "标准": "policy=goal"})),
    Protocol("P8", "外观", "观察", "背景和受控物体颜色",
             _common({"轻": "color=white", "标准": "background=checker,color=white", "重": "background=noise,color=blue"})),
    Protocol("P9", "静止干扰光点", "观察", "灰色干扰光点，不受控制",
             _common({"轻": "distractors=1", "标准": "distractors=2", "重": "distractors=4"})),
    Protocol("P10", "会动的干扰光点", "观察", "绕画面中心匀速转的灰色光点",
             _common({"轻": "movers=1", "标准": "movers=2", "重": "movers=4"})),
    Protocol("P11", "像素噪声", "观察", "每帧独立的高斯像素噪声（0–255 刻度）",
             _common({"轻": "pixel_noise=5", "标准": "pixel_noise=10", "重": "pixel_noise=20"})),
    Protocol("P12", "遮挡", "观察", "48×48 像素的静止遮挡块（部分可观测）",
             _common({"轻": "occluders=1", "标准": "occluders=2", "重": "occluders=3"})),
)


def _combined(system, level):
    """P13 组合：P1 + P4 + P8 的同一档叠在一起（物理 × 执行 × 观察）。"""
    parts = [p.levels[system].get(level) for p in PROTOCOLS if p.id in ("P1", "P4", "P8")]
    return None if None in parts else ",".join(parts)


PROTOCOLS = PROTOCOLS+(Protocol("P13", "组合", "组合", "P1、P4、P8 的同一档叠加", {
    system: {level: _combined(system, level) for level in LEVELS if _combined(system, level) is not None}
    for system in datasets.SYSTEMS}),)
BY_ID = {p.id: p for p in PROTOCOLS}


def system_of(scene: str) -> str:
    return scene.partition("~")[0]


def merged(scene: str, extra: str) -> str:
    """基础场景的调节项叠上测试调节项，返回规范字符串（和 make_v7_jobs.merged 相同的规则）。"""
    system, _, base = scene.partition("~")
    items = dict(i.split("=", 1) for i in base.split(",") if i)
    for item in extra.split(","):
        name, _, value = item.partition("=")
        if name in items:
            raise ValueError(f"{scene} 的基础调节项里已经有 {name}，测试调节项不能覆盖它")
        items[name] = value
    return variants.format(system, variants.parse(system, ",".join(f"{k}={v}" for k, v in items.items())))


def applicable(protocol: Protocol, scene: str, level: str) -> bool:
    """这个场景、这一档能不能评：协议有这一档，且叠上基础场景后 variants.validate 接受。"""
    text = protocol.levels.get(system_of(scene), {}).get(level)
    if text is None:
        return False
    try:
        merged(scene, text)
    except ValueError:
        return False
    return True


def tests(scene: str, levels=("标准",), ids=None) -> list[tuple[str, str, str]]:
    """→ [(协议号, 档位, 测试调节项字符串)]，按协议顺序、档位顺序；只列适用的。"""
    out = []
    for p in PROTOCOLS:
        if ids is not None and p.id not in ids:
            continue
        for level in LEVELS:
            if level in levels and applicable(p, scene, level):
                out.append((p.id, level, p.levels[system_of(scene)][level]))
    return out


def table() -> str:
    """DATASET.md 的协议表（Markdown，tools/protocol_table.py 写入）：一张协议总表，一张各场景物理参数的档位表。"""
    def cell(protocol, system, level):
        return protocol.levels.get(system, {}).get(level)

    lines = ["| 协议 | 类别 | 变什么 | 轻 | 标准 | 重 | 不适用的场景 |", "|---|---|---|---|---|---|---|"]
    for p in PROTOCOLS:
        same = {tuple(cell(p, system_of(s), level) for level in LEVELS) for s in SCENES}
        if len(same) == 1:
            shown = [f"`{x}`" if x else "—" for x in next(iter(same))]
        else:
            shown = ["见下表"]*3
        missing = [system_of(s) for s in SCENES if not any(applicable(p, s, level) for level in LEVELS)]
        lines.append(f"| {p.id} {p.name} | {p.group} | {p.what} | " + " | ".join(shown) +
                     f" | {'、'.join(f'`{m}`' for m in missing) or '—'} |")
    lines += ["", "P1、P2 各场景的档位（P13 用 P1 的同一档）：", "",
              "| 场景 | P1 轻 | P1 标准 | P1 重 | P2 轻 | P2 标准 | P2 重 |", "|---|---|---|---|---|---|---|"]
    for s in SCENES:
        row = [f"`{cell(BY_ID[i], system_of(s), level)}`" for i in ("P1", "P2") for level in LEVELS]
        lines.append(f"| `{system_of(s)}` | " + " | ".join(row) + " |")
    counts = {level: sum(len(tests(s, (level,))) for s in SCENES) for level in LEVELS}
    lines += ["", f"适用的测试格（7 个场景合计）：标准档 {counts['标准']} 个，轻档 {counts['轻']} 个，重档 {counts['重']} 个。"
              "P0 分布内就是基础场景的测试段，不另编码。"]
    return "\n".join(lines)
