"""S21b：生成 DATASET.md 里的评测协议表（唯一来源是 po_jepa.protocols）。

  python tools/protocol_table.py            # 打印表格
  python tools/protocol_table.py --write    # 改写 DATASET.md 里 <!-- protocols:begin --> 与 <!-- protocols:end --> 之间的部分
  python tools/protocol_table.py --check    # DATASET.md 与代码不一致时退出码 1（tests/test_s21b_probe_plan.py 也核）
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from po_jepa import protocols  # noqa: E402

BEGIN, END = "<!-- protocols:begin（由 tools/protocol_table.py 生成，不要手改）-->", "<!-- protocols:end -->"


def table() -> str:
    return protocols.table()


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
            raise SystemExit("DATASET.md 的协议表与 po_jepa.protocols 不一致：运行 python tools/protocol_table.py --write")
    else:
        print(table())


if __name__ == "__main__":
    main()
