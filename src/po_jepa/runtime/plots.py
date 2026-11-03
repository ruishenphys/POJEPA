"""v7：收敛曲线（纯 SVG，不依赖 matplotlib）。

每个训练任务自动画：
- loss.svg：训练损失（JEPA、加权时间一致性、加权空间一致性、总损失；对数纵轴，100 步滑动平均）；
- validation.svg：验证集未来 JEPA 距离（选 checkpoint 用的指标）随训练步数；
- curves.json：画图用的原始数据，汇总脚本直接读它画跨种子/跨条件的图。
"""
import json
import math
from pathlib import Path

COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563")


def _smooth(values, window):
    out, total, queue = [], 0., []
    for v in values:
        queue.append(v); total += v
        if len(queue) > window:
            total -= queue.pop(0)
        out.append(total/len(queue))
    return out


def line_chart(series, title, xlabel, ylabel, *, log_y=False, width=720, height=420):
    """series: [(name, xs, ys)] → SVG 字符串。非有限值与（对数轴下）非正值跳过。"""
    pts = []
    for name, xs, ys in series:
        keep = [(x, y) for x, y in zip(xs, ys)
                if y is not None and math.isfinite(y) and (not log_y or y > 0)]
        pts.append((name, keep))
    allx = [x for _, p in pts for x, _ in p]; ally = [y for _, p in pts for _, y in p]
    left, right, top, bottom = 80, 170, 40, 56
    w, h = width-left-right, height-top-bottom
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{left}" y="22" font-size="14" font-weight="bold">{_esc(title)}</text>']
    if not allx:
        svg.append(f'<text x="{left}" y="{top+h/2}">无数据</text></svg>')
        return "\n".join(svg)
    x0, x1 = min(allx), max(allx)
    x1 = x1 if x1 > x0 else x0+1
    f = (lambda y: math.log10(y)) if log_y else (lambda y: y)
    y0, y1 = min(map(f, ally)), max(map(f, ally))
    pad = (y1-y0)*0.05 or (abs(y0)*0.05 or 1.)
    y0, y1 = y0-pad, y1+pad
    sx = lambda x: left+(x-x0)/(x1-x0)*w
    sy = lambda y: top+h-(f(y)-y0)/(y1-y0)*h
    svg.append(f'<rect x="{left}" y="{top}" width="{w}" height="{h}" fill="none" stroke="#9ca3af"/>')
    for i in range(6):
        xv = x0+(x1-x0)*i/5; px = sx(xv)
        svg.append(f'<line x1="{px:.1f}" y1="{top+h}" x2="{px:.1f}" y2="{top+h+5}" stroke="#6b7280"/>'
                   f'<text x="{px:.1f}" y="{top+h+18}" text-anchor="middle">{_fmt(xv)}</text>')
        t = y0+(y1-y0)*i/5; py = top+h-(t-y0)/(y1-y0)*h
        label = _fmt(10**t if log_y else t)
        svg.append(f'<line x1="{left-5}" y1="{py:.1f}" x2="{left+w}" y2="{py:.1f}" stroke="#e5e7eb"/>'
                   f'<text x="{left-8}" y="{py+4:.1f}" text-anchor="end">{label}</text>')
    svg.append(f'<text x="{left+w/2}" y="{height-12}" text-anchor="middle">{_esc(xlabel)}</text>')
    svg.append(f'<text x="18" y="{top+h/2}" text-anchor="middle" transform="rotate(-90 18 {top+h/2})">'
               f'{_esc(ylabel)}{" (log)" if log_y else ""}</text>')
    for i, (name, p) in enumerate(pts):
        color = COLORS[i % len(COLORS)]
        if len(p) == 1:
            svg.append(f'<circle cx="{sx(p[0][0]):.1f}" cy="{sy(p[0][1]):.1f}" r="3" fill="{color}"/>')
        elif p:
            d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in p)
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{d}"/>')
        ly = top+14+18*i
        svg.append(f'<line x1="{left+w+12}" y1="{ly}" x2="{left+w+32}" y2="{ly}" stroke="{color}" stroke-width="3"/>'
                   f'<text x="{left+w+38}" y="{ly+4}">{_esc(name)}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def _fmt(v):
    a = abs(v)
    if a == 0: return "0"
    if a >= 1000: return f"{v:.0f}"
    if a >= 1: return f"{v:.3g}"
    return f"{v:.2e}" if a < 1e-2 else f"{v:.3f}"


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def training_curves(root) -> dict:
    """从一个训练目录读每步损失与每次验证结果。"""
    root = Path(root)
    steps, losses = [], {}
    for path in sorted((root/"updates").glob("*.json")):
        u = json.loads(path.read_text())
        steps.append(u["completed_updates"])
        for key in ("jepa", "weighted_time", "weighted_space", "total", "ac_teacher_forcing", "ac_rollout"):
            if key in u["losses"]:
                losses.setdefault(key, []).append(u["losses"][key])
    validation = []
    for path in sorted((root/"validations").glob("selection-*.json")):
        s = json.loads(path.read_text())
        validation.append(dict(step=s["step"], future_jepa_distance=s["mean"]))
    validation.sort(key=lambda r: r["step"])
    return dict(steps=steps, losses=losses, validation=validation)


def job_figures(output, condition, training_root=None) -> dict:
    """为一个任务写 figures/{curves.json, loss.svg, validation.svg}；identity 无训练，只返回空。"""
    output = Path(output)
    if training_root is None or not (Path(training_root)/"updates").is_dir():
        return {}
    curves = training_curves(training_root)
    folder = output/"figures"; folder.mkdir(parents=True, exist_ok=True)
    (folder/"curves.json").write_text(json.dumps(dict(condition=condition, **curves), indent=1))
    window = max(1, min(100, len(curves["steps"])//20 or 1))
    series = [(k, curves["steps"], _smooth(v, window)) for k, v in curves["losses"].items()
              if any(x not in (None, 0) for x in v)]
    (folder/"loss.svg").write_text(line_chart(series, f"{condition} 训练损失（{window} 步滑动平均）",
                                              "更新步数", "loss", log_y=True))
    v = curves["validation"]
    (folder/"validation.svg").write_text(line_chart(
        [("验证集未来 JEPA 距离", [r["step"] for r in v], [r["future_jepa_distance"] for r in v])],
        f"{condition} 验证集收敛", "更新步数", "future JEPA distance"))
    return dict(curves="figures/curves.json", loss="figures/loss.svg", validation="figures/validation.svg")
