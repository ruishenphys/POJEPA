"""S21a：生成 DATASET.md 里的“通用调节项 × 场景”覆盖表（唯一来源是 po_jepa.variants.coverage，即 validate 的判定）。

  python tools/knob_coverage.py            # 打印表格
  python tools/knob_coverage.py --write    # 改写 DATASET.md 里 <!-- coverage:begin --> 与 <!-- coverage:end --> 之间的部分
  python tools/knob_coverage.py --check    # DATASET.md 与代码不一致时退出码 1（tests/test_s21a_general_knobs.py 也核）
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from po_jepa import datasets, variants  # noqa: E402

BEGIN, END = "<!-- coverage:begin（由 tools/knob_coverage.py 生成，不要手改）-->", "<!-- coverage:end -->"


def table() -> str:
    cov = variants.coverage()
    head = "| 类别 | 调节项 | " + " | ".join(f"`{s}`" for s in datasets.SYSTEMS) + " | 可用场景数 |"
    rows = [head, "|" + "---|"*(len(datasets.SYSTEMS)+3)]
    for group, names in variants.KNOB_GROUPS.items():
        for name in names:
            marks = " | ".join("✓" if s in cov[name] else "—" for s in datasets.SYSTEMS)
            rows.append(f"| {group} | `{name}` | {marks} | {len(cov[name])} |")
    total = sum(len(v) for v in cov.values())
    rows.append(f"\n通用调节项 {len(cov)} 个，“调节项 × 场景”可用格 {total} / {len(cov)*len(datasets.SYSTEMS)}。"
                "物理参数（各场景自己的）见上表。")
    return "\n".join(rows)


def render(text: str) -> str:
    a, b = text.index(BEGIN), text.index(END)
    return text[:a+len(BEGIN)] + "\n" + table() + "\n" + text[b:]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = p.parse_args()
    path = ROOT/"DATASET.md"
    if a.write:
        path.write_text(render(path.read_text()))
    elif a.check:
        text = path.read_text()
        if render(text) != text:
            raise SystemExit("DATASET.md 的覆盖表与 variants.coverage() 不一致：运行 python tools/knob_coverage.py --write")
    else:
        print(table())


if __name__ == "__main__":
    main()
