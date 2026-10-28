"""评价薄入口：等价于统一入口的evaluate子命令。"""
import sys
from po_jepa.runtime.cli import main

if __name__=='__main__':raise SystemExit(main(['evaluate',*sys.argv[1:]]))
