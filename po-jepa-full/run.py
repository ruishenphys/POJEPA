"""推荐入口：自动捕获完整输出和真实退出码，结果目录可以直接回传。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ADDON_ENTRY=('calibrate-addon','check-addon')   # 16 §5：补跑子命令交给 po_jepa.runtime.addon.entry，v7 的 cli.py 不动


def main(argv=None):
    """只启动当前解释器的唯一运行实现；help没有缓存、权重或GPU访问。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check','train','evaluate','run','resume','calibrate','sweep','transfer',
                        'calibrate-addon','check-addon','d4','spectral-gap','equal-step'))
    parser.add_argument('--features-root',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-run',type=Path)
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--seed',type=int,choices=(1703,1704,1705),default=1703,help='预登记种子；calibrate 只用 1703')
    parser.add_argument('--system',required=True,choices=('double_well','van_der_pol','multi_particle','advection_diffusion','landscape','gray_scott','push','pendulum','bouncing_balls','wave','cart_pole','spring_chain'),
                        help='v7：物理系统')
    parser.add_argument('--variant',default='',help='v7 可调数据集的调节项，例如 gain=2,axes=x；须与编码时相同（见 src/po_jepa/variants.py）')
    parser.add_argument('--trajectories',type=int,required=True,choices=(200,500,2000),help='v7：轨迹总条数（划分 80/10/10）')
    parser.add_argument('--updates',type=int,required=True,choices=(2000,10000),help='v7：每个条件的训练步数（2000 试跑，10000 正式）')
    parser.add_argument('--condition',choices=('po_jepa_full','po_jepa_no_consistency','matched_transformer','identity',
                        'vjepa2_ac_small','vjepa2_ac','po_jepa_time_only','po_jepa_space_only','transformer_consistency',
                        'fixed_fourier','koopman','po_jepa_rollout_sup','dino_wm_small','jepa_wm_small'),help='v7：run/resume/evaluate 每次只跑一个条件，可并行提交（po_jepa_time_only 起 6 个是补跑的新条件，07 §2.6、08 §6；最后 2 个是 DINO-WM、JEPA-WM 参数匹配基线，2026-11-25）')
    parser.add_argument('--ac-checkpoint',type=Path,help='v7：vjepa2_ac 需要的官方权重 vjepa2-ac-vitg.pt')
    parser.add_argument('--test-variant',help='v7 transfer：测试集的调节项（--system/--variant/--trajectories 写训练用的基础数据集）')
    parser.add_argument('--base-calibration',choices=('v7-import','local'),help='S26a，只用于 calibrate-addon/check-addon：基础标定的来源；默认 v7-import（导入的 v7 标定），v8 任务表的补跑用 local（本目录 run.py calibrate 写的）')
    parser.add_argument('--limit-run',type=Path,help='v7.1 equal-step：po_jepa_full 的任务目录；--training-run 的条件截到它实际训练的步数再评测')
    # S28c：S28b 的两个变体开关改成命令行参数（tools/run_jobs.py 按 argv 执行，任务行前面不能加环境变量）；不写时沿用环境变量（S28b 写法），
    # 写了就以参数为准，和已设的环境变量不同时报错，不让两处说法打架
    parser.add_argument('--temporal-partition',choices=('bisection','dyadic'),help='S28b M1：时间一致性切分（默认二等分）；设 PO_JEPA_TEMPORAL_PARTITION')
    parser.add_argument('--readout-init',choices=('xavier','zero'),help='S28b M2：P_out 初始化（默认 Xavier）；设 PO_JEPA_READOUT_INIT')
    # C3（2026-11-22 K 消融 + 方案 B）：同一写法，参数和已设的环境变量不同时报错
    parser.add_argument('--modes',choices=('16','32','64','full'),help='C3：空间基底数 K（默认 16 固定；full = 每个网格 K = token 数）；设 PO_JEPA_MODES')
    parser.add_argument('--complement',choices=('none','shared'),help='C3：补空间（默认 none；shared = 方案 B，前 K 个模态照常、补空间整体保留并参与演化）；设 PO_JEPA_COMPLEMENT')
    # G14（2026-11-25，导师建议试更小的一致性系数）：λ_t / λ_s 在本变体标定值上再乘 0.5 或 0.25；同一写法，和已设的环境变量不同时报错
    parser.add_argument('--time-weight-scale',choices=('1','0.5','0.25'),help='G14：λ_t 乘这个系数（默认 1）；设 PO_JEPA_TIME_WEIGHT_SCALE')
    parser.add_argument('--space-weight-scale',choices=('1','0.5','0.25'),help='G14：λ_s 乘这个系数（默认 1）；设 PO_JEPA_SPACE_WEIGHT_SCALE')
    args=parser.parse_args(argv)
    # 子进程的工作目录是代码根目录，相对路径须先按调用者当前目录解析成绝对路径。
    if not Path(args.features_root).resolve().is_dir():parser.error(f'--features-root不存在：{Path(args.features_root).resolve()}')
    if args.output.exists():parser.error('--output必须是新目录，不能覆盖旧结果')
    if args.command=='evaluate' and args.training_run is None:parser.error('evaluate需要--training-run')
    if args.command=='resume' and args.resume_from is None:parser.error('resume需要--resume-from')
    if args.command!='resume' and args.resume_from is not None:parser.error('--resume-from仅用于resume')
    if args.command in ('calibrate','sweep') and args.seed!=1703:parser.error('calibrate只用种子1703；三个种子共用同一组λ')
    if args.command in ADDON_ENTRY and (args.seed!=1703 or args.condition or args.training_run or args.test_variant):
        parser.error(f'{args.command} 只用种子 1703，不带 --condition、--training-run、--test-variant（16 §3.2、§5）')
    if args.base_calibration and args.command not in ADDON_ENTRY:parser.error('--base-calibration 只用于 calibrate-addon、check-addon')
    if args.command=='transfer' and (args.condition is None or not args.test_variant):parser.error('transfer 需要 --condition 和 --test-variant')
    if args.command=='transfer' and args.condition!='identity' and args.training_run is None:parser.error('transfer 需要 --training-run（基础场景上的任务目录）')
    if args.command=='equal-step' and (args.condition is None or args.training_run is None or args.limit_run is None):
        parser.error('equal-step 需要 --condition、--training-run（该条件的任务目录）和 --limit-run（po_jepa_full 的任务目录）')
    if args.command!='equal-step' and args.limit_run is not None:parser.error('--limit-run 仅用于 equal-step')
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
    for flag,name,value in (('--temporal-partition','PO_JEPA_TEMPORAL_PARTITION',args.temporal_partition),
                            ('--readout-init','PO_JEPA_READOUT_INIT',args.readout_init),
                            ('--modes','PO_JEPA_MODES',args.modes),('--complement','PO_JEPA_COMPLEMENT',args.complement),
                            ('--time-weight-scale','PO_JEPA_TIME_WEIGHT_SCALE',args.time_weight_scale),
                            ('--space-weight-scale','PO_JEPA_SPACE_WEIGHT_SCALE',args.space_weight_scale)):
        if value is None:continue
        if env.get(name,value)!=value:parser.error(f'{flag} {value} 和环境变量 {name}={env[name]} 不一致：去掉其中一个')
        env[name]=value
    # G14-fix1（Codex G14 P1）：按最终传给子进程的环境判断缩放（参数或继承的环境变量都算），子进程里建模型、标定时还会再拦一次
    scaled=any(env.get(name,'1')!='1' for name in ('PO_JEPA_TIME_WEIGHT_SCALE','PO_JEPA_SPACE_WEIGHT_SCALE'))
    if scaled and args.command in ('calibrate','sweep','calibrate-addon','check-addon','equal-step','check'):
        parser.error(f'{args.command} 不收一致性权重缩放（--time-weight-scale/--space-weight-scale 或同名环境变量）：缩放乘在同一变体的标定值上，不另标定；同步数对照两边用同一份预登记，配不上')
    if scaled and args.condition not in (None,'po_jepa_full','po_jepa_time_only','po_jepa_space_only'):
        parser.error('一致性权重缩放只用于 po_jepa_full、po_jepa_time_only、po_jepa_space_only（其他条件的 λ 不来自基础标定或恒为 0）；检查 --time-weight-scale/--space-weight-scale 和同名环境变量')
    if args.ac_checkpoint:env['PO_JEPA_AC_CHECKPOINT']=str(args.ac_checkpoint.resolve())
    child=[sys.executable,'-B','-X','utf8','-m','po_jepa.runtime.addon.entry' if args.command in ADDON_ENTRY else 'po_jepa.runtime.cli',args.command,
        '--features-root',str(Path(args.features_root).resolve()),'--device',args.device,'--output',str(args.output.resolve()/'run')]
    child+=['--seed',str(args.seed)]
    if args.base_calibration:child+=['--base-calibration',args.base_calibration]
    if args.condition:child+=['--condition',args.condition]
    if args.command=='transfer':child+=['--test-variant',args.test_variant]
    if args.training_run:child+=['--training-run',str(args.training_run.resolve())]
    if args.resume_from:child+=['--resume-from',str(args.resume_from.resolve())]
    if args.command in ('d4','spectral-gap'):child[child.index('po_jepa.runtime.cli')]='po_jepa.runtime.addon.eval_cli'   # S18：这两个命令的子进程入口
    if args.limit_run:child+=['--limit-run',str(args.limit_run.resolve())]
    command=[sys.executable,'-B','-X','utf8',str(root/'tools/capture.py'),'--cwd',str(root),
        '--output',str(args.output.resolve()/'console'),'--',*child]
    return subprocess.run(command,check=False,env=env).returncode


if __name__=='__main__':raise SystemExit(main())
