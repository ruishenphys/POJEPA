"""推荐入口：自动捕获完整输出和真实退出码，结果目录可以直接回传。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    """只启动当前解释器的唯一运行实现；help没有缓存、权重或GPU访问。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check','train','evaluate','run','resume','calibrate','sweep','transfer','d4','spectral-gap'))
    parser.add_argument('--features-root',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-run',type=Path)
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--seed',type=int,choices=(1703,1704,1705),default=1703,help='预登记种子；calibrate 只用 1703')
    parser.add_argument('--system',required=True,choices=('double_well','van_der_pol','multi_particle','advection_diffusion','landscape'),
                        help='v7：物理系统')
    parser.add_argument('--variant',default='',help='v7 可调数据集的调节项，例如 gain=2,axes=x；须与编码时相同（见 src/po_jepa/variants.py）')
    parser.add_argument('--trajectories',type=int,required=True,choices=(200,500,2000),help='v7：轨迹总条数（划分 80/10/10）')
    parser.add_argument('--updates',type=int,required=True,choices=(2000,10000),help='v7：每个条件的训练步数（2000 试跑，10000 正式）')
    parser.add_argument('--condition',choices=('po_jepa_full','po_jepa_no_consistency','matched_transformer','identity',
                        'vjepa2_ac_small','vjepa2_ac','po_jepa_time_only','po_jepa_space_only','transformer_consistency',
                        'fixed_fourier','koopman'),help='v7：run/resume/evaluate 每次只跑一个条件，可并行提交（后 5 个是补跑的新条件，07 §2.6）')
    parser.add_argument('--ac-checkpoint',type=Path,help='v7：vjepa2_ac 需要的官方权重 vjepa2-ac-vitg.pt')
    parser.add_argument('--test-variant',help='v7 transfer：测试集的调节项（--system/--variant/--trajectories 写训练用的基础数据集）')
    args=parser.parse_args(argv)
    # 子进程的工作目录是代码根目录，相对路径须先按调用者当前目录解析成绝对路径。
    if not Path(args.features_root).resolve().is_dir():parser.error(f'--features-root不存在：{Path(args.features_root).resolve()}')
    if args.output.exists():parser.error('--output必须是新目录，不能覆盖旧结果')
    if args.command=='evaluate' and args.training_run is None:parser.error('evaluate需要--training-run')
    if args.command=='resume' and args.resume_from is None:parser.error('resume需要--resume-from')
    if args.command!='resume' and args.resume_from is not None:parser.error('--resume-from仅用于resume')
    if args.command in ('calibrate','sweep') and args.seed!=1703:parser.error('calibrate只用种子1703；三个种子共用同一组λ')
    if args.command=='transfer' and (args.condition is None or not args.test_variant):parser.error('transfer 需要 --condition 和 --test-variant')
    if args.command=='transfer' and args.condition!='identity' and args.training_run is None:parser.error('transfer 需要 --training-run（基础场景上的任务目录）')
    if args.command in ('run','resume','evaluate') and args.condition is None:parser.error('v7 的 run/resume/evaluate 需要 --condition')
    if args.command=='train':parser.error('v7 用 run --condition <条件>，每个条件单独提交')
    if args.condition=='vjepa2_ac' and (args.ac_checkpoint is None or not args.ac_checkpoint.resolve().is_file()):
        parser.error('vjepa2_ac 需要 --ac-checkpoint <vjepa2-ac-vitg.pt>')
    if args.command=='sweep':parser.error('v7 不再 sweep：rho 固定为 1.0，只需 calibrate')
    if args.resume_from and args.output.resolve().is_relative_to(args.resume_from.resolve()):
        parser.error('续训新输出必须位于旧训练目录之外，不能写入原证据')
    # S18（10 §3–§5、16 §5）：d4 读一个已完成任务的选中检查点；spectral-gap 每个场景一次，只在 CPU 上跑
    if args.command=='d4' and (args.condition is None or args.training_run is None):parser.error('d4 需要 --condition 和 --training-run（被读的任务目录）')
    if args.command=='spectral-gap' and (args.condition is not None or args.training_run is not None or args.seed!=1703):
        parser.error('spectral-gap 每个场景只跑一次：不接受 --condition、--training-run，种子只用 1703')
    if args.command=='spectral-gap' and args.device!='cpu':parser.error('spectral-gap 只在 CPU 上跑：--device cpu')
    root=Path(__file__).resolve().parent
    # 子进程固定使用本目录的 src，避免环境里以前 pip install -e 过的旧版本（如 code-v4）被导入。
    env=dict(os.environ)
    env['PYTHONPATH']=os.pathsep.join([str(root/'src'),*filter(None,[os.environ.get('PYTHONPATH')])])
    # v7：协议、数据集与训练步数经环境变量传给子进程（见 po_jepa/datasets.py、runtime/checks.py）
    env['PO_JEPA_PROTOCOL']='v7'
    env['PO_JEPA_DATASET']=f"{args.system}{'~'+args.variant if args.variant else ''}:{args.trajectories}"
    env['PO_JEPA_UPDATES']=str(args.updates)
    if args.ac_checkpoint:env['PO_JEPA_AC_CHECKPOINT']=str(args.ac_checkpoint.resolve())
    child=[sys.executable,'-B','-X','utf8','-m','po_jepa.runtime.cli',args.command,
        '--features-root',str(Path(args.features_root).resolve()),'--device',args.device,'--output',str(args.output.resolve()/'run')]
    child+=['--seed',str(args.seed)]
    if args.condition:child+=['--condition',args.condition]
    if args.command=='transfer':child+=['--test-variant',args.test_variant]
    if args.training_run:child+=['--training-run',str(args.training_run.resolve())]
    if args.resume_from:child+=['--resume-from',str(args.resume_from.resolve())]
    if args.command in ('d4','spectral-gap'):child[child.index('po_jepa.runtime.cli')]='po_jepa.runtime.addon.eval_cli'   # S18：这两个命令的子进程入口
    command=[sys.executable,'-B','-X','utf8',str(root/'tools/capture.py'),'--cwd',str(root),
        '--output',str(args.output.resolve()/'console'),'--',*child]
    return subprocess.run(command,check=False,env=env).returncode


if __name__=='__main__':raise SystemExit(main())
