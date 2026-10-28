"""统一命令行：检查、训练、显式续训、评价或完整运行。"""
import argparse
from pathlib import Path


def main(argv=None):
    """help不读缓存、不构造模型；真实工作仅在显式子命令后执行。"""
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command,help_text in (('check','核依赖、设备、完整梯度、100更新及400窗成本'),('train','预检后训练三个条件'),
        ('resume','从明确旧训练目录恢复到新目录，不覆盖失败证据'),
        ('evaluate','只评价已完成训练的选中点'),('run','预检、训练、留出评价全部顺序执行'),
        ('calibrate','v4：初始化状态下按梯度比标定一致性权重 lambda_t/lambda_s，不训练')):
        task=sub.add_parser(command,help=help_text)
        task.add_argument('--features-root',type=Path,required=True,help='交接包内已包含的冻结特征缓存根目录')
        task.add_argument('--device',default='cuda:0',help='4090/A100的显式cuda:N')
        task.add_argument('--output',type=Path,required=True,help='必须不存在的新结果目录')
        if command=='evaluate':task.add_argument('--training-run',type=Path,required=True)
        if command=='resume':task.add_argument('--resume-from',type=Path,required=True,help='旧结果/run/training目录；仅支持同版源码与环境')
    args=parser.parse_args(argv)
    from .workflow import execute
    return execute(args)


if __name__=='__main__':raise SystemExit(main())
