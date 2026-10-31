"""推荐入口：自动捕获完整输出和真实退出码，结果目录可以直接回传。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    """只启动当前解释器的唯一运行实现；help没有缓存、权重或GPU访问。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check','train','evaluate','run','resume','calibrate','sweep'))
    parser.add_argument('--features-root',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-run',type=Path)
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--seed',type=int,choices=(1703,1704,1705),default=1703,help='v6：预登记种子；calibrate 只用 1703')
    args=parser.parse_args(argv)
    # 子进程的工作目录是代码根目录，相对路径须先按调用者当前目录解析成绝对路径。
    if not Path(args.features_root).resolve().is_dir():parser.error(f'--features-root不存在：{Path(args.features_root).resolve()}')
    if args.output.exists():parser.error('--output必须是新目录，不能覆盖旧结果')
    if args.command=='evaluate' and args.training_run is None:parser.error('evaluate需要--training-run')
    if args.command=='resume' and args.resume_from is None:parser.error('resume需要--resume-from')
    if args.command!='resume' and args.resume_from is not None:parser.error('--resume-from仅用于resume')
    if args.command in ('calibrate','sweep') and args.seed!=1703:parser.error('calibrate/sweep只用种子1703；三个种子共用同一组λ')
    if args.resume_from and args.output.resolve().is_relative_to(args.resume_from.resolve()):
        parser.error('续训新输出必须位于旧训练目录之外，不能写入原证据')
    root=Path(__file__).resolve().parent
    # 子进程固定使用本目录的 src，避免环境里以前 pip install -e 过的旧版本（如 code-v4）被导入。
    os.environ['PYTHONPATH']=os.pathsep.join([str(root/'src'),*filter(None,[os.environ.get('PYTHONPATH')])])
    child=[sys.executable,'-B','-X','utf8','-m','po_jepa.runtime.cli',args.command,
        '--features-root',str(Path(args.features_root).resolve()),'--device',args.device,'--output',str(args.output.resolve()/'run')]
    child+=['--seed',str(args.seed)]
    if args.training_run:child+=['--training-run',str(args.training_run.resolve())]
    if args.resume_from:child+=['--resume-from',str(args.resume_from.resolve())]
    command=[sys.executable,'-B','-X','utf8',str(root/'tools/capture.py'),'--cwd',str(root),
        '--output',str(args.output.resolve()/'console'),'--',*child]
    return subprocess.run(command,check=False).returncode


if __name__=='__main__':raise SystemExit(main())
