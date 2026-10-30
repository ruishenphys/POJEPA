"""交付工作流：复用原计算，只集中管理真实输入、运行阶段与回传结果。"""
from dataclasses import asdict, fields
import hashlib
import statistics
import json
from pathlib import Path
import platform
import sys
import traceback

import torch
from . import cache, checks, comparison, data, environment, evaluation, resources, training

ROOT=Path(__file__).resolve().parents[3]


def read(path):
    """读取普通JSON；重复键及NaN/Inf拒绝，避免输入摘要含糊。"""
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    def invalid(value):raise ValueError('nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=invalid)


def verify_bundle():
    """核本包源码/配置/公式清单，不依赖开发仓库或旧验收记录。"""
    manifest=read(ROOT/'MANIFEST.json')
    for name,digest in manifest['files_sha256'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or cache.file_sha(path)!=digest:
            raise ValueError('delivery source changed: '+name)
    cache.verify_fixed_protocol()
    return cache.file_sha(ROOT/'MANIFEST.json')


def model_identity(model):
    """实际参数和缓冲区摘要；相同种子不代替初态一致性。"""
    return {n:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(
        v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for n,v in model.state_dict().items()}


def preflight(features,output,device):
    """完整资源门槛：隔离试跑随机状态，成功后才能开始正式比较。"""
    with resources.isolated_randomness(device):
        return _preflight(features,output,device)


def _preflight(features,output,device):
    """核设备/缓存、五Phi三损失梯度、100正常更新与400窗验证成本。"""
    output.mkdir(parents=True,exist_ok=False)
    hardware=checks.device_metadata(device,output)
    assets=data.approved_cache_metadata(features);config=checks.load_protocol()
    model,loss=comparison.build_condition('po_jepa_full',device)
    optimizer=comparison.make_optimizer(model)
    queries=[data.WindowQuery(i,5,d) for i,d in enumerate((1.,2.,4.,8.))]
    batch=data.load_training_batch(features,queries,device=str(device))
    data.write_collections(batch,output/'source-collections')
    torch.cuda.reset_peak_memory_stats(device)
    diagnostic=checks.diagnostic_step(model,optimizer,batch,loss)
    torch.cuda.synchronize(device)
    memory=checks._memory(device)
    del model,optimizer,batch
    costs=resources.resource_check(features,output/'resources',device)
    report=dict(hardware=hardware,cache_identity=assets,diagnostic=diagnostic,gpu_memory=checks._memory(device),
        config_sha256=cache.file_sha(ROOT/'configs/train/default.yaml'),
        v4_overlay_sha256=cache.file_sha(ROOT/'configs/train/v4.yaml'),
        python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,numpy=cache.np.__version__,
        platform=platform.platform(),cpu_threads=torch.get_num_threads(),real_device_check_completed=True)
    report.update(gpu_memory=memory,resources=costs,resource_gate_completed=True)
    cache._write_json(output/'report.json',report)
    cache._write_json(output/'COMPLETE.json',dict(report_sha256=cache.file_sha(output/'report.json')))
    return report


def train_conditions(features,output,device,resume_from=None):
    """三条件各训练一次；实际2000×4共享查询，完整PO与消融的初始权重必须一致。"""
    output.mkdir(parents=True,exist_ok=False)
    registration,sequence,state=comparison.preregister()
    registration['delivery_manifest_sha256']=verify_bundle()
    runtime=environment.runtime_signature(device)
    if resume_from is not None:
        from .resume import verify_training_source
        verify_training_source(resume_from,registration,runtime)
    cache._write_json(output/'preregistration.json',registration)
    cache._write_json(output/'runtime.json',runtime)
    reports={};initial={}
    for condition in ('po_jepa_full','matched_transformer','po_jepa_no_consistency'):
        print('training '+condition,flush=True)
        model,loss=comparison.build_condition(condition,device)
        initial[condition]=model_identity(model)
        if condition=='po_jepa_no_consistency' and initial[condition]!=initial['po_jepa_full']:
            raise ValueError('full and no-consistency actual initial parameters differ')
        optimizer=comparison.make_optimizer(model)
        cache._write_json(output/f'initial-{condition}.json',initial[condition])
        source=Path(resume_from)/condition if resume_from is not None else None
        if source is not None and (source/'COMPLETE.json').exists():
            from .resume import copy_completed_condition
            reports[condition]=copy_completed_condition(source,output/condition,registration,sequence,state)
            del model,optimizer
            continue
        reports[condition]=comparison.run_prepared_condition(condition,model,optimizer,
            sequence=sequence,generator_state=state,registration=registration,features=features,
            output=output/condition,device=device,loss_config=loss,
            validation_interval=checks.load_protocol()['validation']['interval_updates'],
            **({'resume_from':source} if source is not None and source.exists() else {}))
        del model,optimizer
    loss=comparison.LossConfig(**{f.name:checks.load_protocol()['loss'][f.name] for f in fields(comparison.LossConfig)})
    identity=comparison.identity_reference(features,output/'identity',device,loss,registration)
    table=comparison.comparison_table(reports,identity,registration)
    cache._write_json(output/'initial-models.json',initial)
    cache._write_json(output/'report.json',dict(conditions=reports,comparison=table,scientific_conclusions_verified=False))
    cache._write_json(output/'COMPLETE.json',dict(report_sha256=cache.file_sha(output/'report.json')))
    return reports


def verify_selection(root,report):
    """按生产FP64归约从400原值重建所有候选均值，然后核固定未来误差选点。"""
    expected=[dict(trajectory_id=t,start_tick=s,duration_cycles=d) for t in range(160,180)
        for s in (0,5,10,15,20) for d in (1.,2.,4.,8.)]
    budget=report['completed_updates'];interval=checks.load_protocol()['validation']['interval_updates']
    steps=list(range(0,budget+1,interval))
    if steps[-1]!=budget:steps.append(budget)
    if [r['step'] for r in report['validations']]!=steps:raise ValueError('validation schedule differs')
    for row in report['validations']:
        if row['windows']!=400 or row['metric']!='future_jepa_distance':raise ValueError('validation metric/count differs')
        folder=root/'validations'/f"{row['step']:04d}";total=0.
        for offset in range(0,400,4):
            batch=read(folder/f'{offset:04d}.json')
            if batch['queries']!=expected[offset:offset+4]:raise ValueError('validation query differs')
            values=torch.tensor(batch['values']['future_jepa_distance'],dtype=torch.float64)
            if values.shape!=(4,) or not torch.isfinite(values).all() or (values<0).any():raise ValueError('invalid validation values')
            total+=values.sum().item()
        if total/400!=row['mean']:raise ValueError('validation mean differs from original 400 values')
    best=min(report['validations'],key=lambda r:(r['mean'],r['step']))
    if best!=report['best']:raise ValueError('selected checkpoint differs from fixed future-only rule')
    return best


def load_training_run(root):
    """核自身完整训练产物，不以声明 PASS 或任意 checkpoint 代替实际完成证据。"""
    from .resume import verify_ledger
    root=Path(root).resolve()
    if (root/'FAILED.json').exists():raise ValueError('training run retains failure')
    if read(root/'COMPLETE.json')['report_sha256']!=cache.file_sha(root/'report.json'):raise ValueError('training report changed')
    registration=read(root/'preregistration.json')
    actual,sequence,state=comparison.preregister()
    actual['delivery_manifest_sha256']=verify_bundle()
    if registration!=actual:raise ValueError('training preregistration differs from current source/config')
    reports=read(root/'report.json')['conditions']
    if set(reports)!=set(comparison.CONDITIONS):raise ValueError('missing trained comparison condition')
    for condition,report in reports.items():
        folder=root/condition
        if ((folder/'FAILED.json').exists() or report['completed_updates']!=len(sequence)
            or report['query_sequence_sha256']!=registration['query_sequence_sha256']
            or read(folder/'COMPLETE.json')['report_sha256']!=cache.file_sha(folder/'report.json')
            or read(folder/'report.json')!=report or report['evidence_role']!='real_frozen_features'):
            raise ValueError('incomplete real condition: '+condition)
        verify_ledger(folder,sequence,state,registration,report['completed_updates'])
        verify_selection(folder,report)
    return reports


def runtime_comparison(training_root,device):
    """v4：评价时记录当前运行签名并与训练时的 runtime.json 逐项对照；只报告不阻断（换机评价是允许的）。"""
    current=environment.runtime_signature(device)
    source=Path(training_root)/'runtime.json'
    training_runtime=read(source) if source.exists() else None
    if training_runtime is None:
        return dict(current=current,training=None,matches=None,differences=['training runtime.json missing'])
    def flat(value,prefix=''):
        if isinstance(value,dict):
            return {k:v for key,item in value.items() for k,v in flat(item,prefix+key+'.').items()}
        return {prefix[:-1]:value}
    a,b=flat(training_runtime),flat(current)
    differences=sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k))
    return dict(current=current,training=training_runtime,matches=not differences,differences=differences)


def evaluate_conditions(features,training_root,output,device):
    """只读各条件的验证集选中点；不在test选点、不训练，identity也用显式目标设备。"""
    reports=load_training_run(training_root)
    output.mkdir(parents=True,exist_ok=False);records=[]
    runtime=runtime_comparison(Path(training_root),device)
    cache._write_json(output/'runtime.json',runtime)
    for condition in (*comparison.CONDITIONS,'identity'):
        print('evaluating '+condition,flush=True)
        model,loss=comparison.build_condition(condition,device);selected=None
        if condition!='identity':
            selected=reports[condition]['best'];optimizer=comparison.make_optimizer(model)
            evaluation.load_selected_checkpoint(condition,selected,model=model,optimizer=optimizer,
                config=loss,expected_provenance=selected['checkpoint_provenance'],root=Path(training_root)/condition)
            del optimizer
        evaluation.run_prepared_evaluation(model,condition,features,output/condition,loss,
            checkpoint_identity=selected,device=device)
        records.extend(read(p) for p in sorted((output/condition/'rows').glob('*.json')))
        del model
    comparisons=evaluation.paired_comparisons(records,evaluation.build_index())
    cache._write_json(output/'report.json',dict(paired_comparisons=comparisons,runtime_matches_training=runtime['matches'],
        runtime_differences=runtime['differences'],real_evaluation_run=True,scientific_conclusions_verified=False))
    cache._write_json(output/'COMPLETE.json',dict(report_sha256=cache.file_sha(output/'report.json')))


PLACEHOLDER='【待标定】'
CALIBRATED_MARK='calibrate 自动填入（v6'
OVERLAY=ROOT/'configs/train/v4.yaml'


def require_calibrated_weights():
    """v6：v4.yaml 的 time_weight/space_weight 必须是 v6 calibrate 写入的值；占位值或 v5 旧标定都拒绝运行。"""
    lines=[l for l in OVERLAY.read_text(encoding='utf-8').splitlines() if l.strip().startswith(('time_weight:','space_weight:'))]
    if len(lines)!=2 or any(PLACEHOLDER in l or CALIBRATED_MARK not in l for l in lines):
        raise ValueError('一致性权重还没按 v6 方式标定：请先运行 python run.py calibrate（会自动写入 configs/train/v4.yaml）')


def write_calibrated_weights(time_weight,space_weight,tag):
    """把标定结果直接写进 v4.yaml 的 loss 段两行，免去手动改文件；其他行原样保留。"""
    text=OVERLAY.read_text(encoding='utf-8');out=[]
    for line in text.splitlines(keepends=True):
        stripped=line.lstrip()
        for key,value in (('time_weight',time_weight),('space_weight',space_weight)):
            if stripped.startswith(key+':'):
                indent=line[:len(line)-len(stripped)]
                line=f"{indent}{key}: {value!r}    # {CALIBRATED_MARK}，{tag}）\n"
        out.append(line)
    OVERLAY.write_text(''.join(out),encoding='utf-8')
    require_calibrated_weights()


def _gradient_rows(model,sequence,features,device,loss):
    """逐 batch 求 JEPA/时间/空间三项各自的梯度范数，不更新参数。"""
    return [training.consistency_gradient_norms(model,data.load_training_batch(features,q,device=str(device)),loss_config=loss)
            for q in sequence]


def calibrate_weights(features,output,device):
    """v6 λ 标定：lambda_c=rho*median|grad L_JEPA|/median|grad L_c|，在“只用 JEPA 训练 pretrain_updates 步之后”的模型上量。

    v5 在初始化状态下量：那时 P_out 读出很大、一致性损失是训练后的上千倍，梯度比 JEPA 大 100–800 倍，
    λ 被压到 1e-4 量级，正式训练中加权一致性只占总损失约 1e-6，full 与无一致性实际是同一个模型。
    这里先用与正式训练完全相同的初始化、查询序列、学习率调度和裁剪，只用 JEPA（λ=0，即无一致性条件）
    训练 pretrain_updates 步，越过初始化瞬态后，再在紧接着的 batches 个训练 batch 上量梯度。
    初始化状态的数值也一并记录，便于对照。不写 checkpoint；结果写 calibration.json 并自动写入 v4.yaml。
    """
    output.mkdir(parents=True,exist_ok=False)
    config=checks.load_protocol();cal=config['calibration']
    pre,count=cal['pretrain_updates'],cal['batches']
    sequence,_=comparison.learning.make_query_sequence(config)
    if pre+count>len(sequence):raise ValueError('pretrain_updates+batches exceeds the training sequence')
    model,loss=comparison.build_condition('po_jepa_no_consistency',device)
    model.train()
    initial=_gradient_rows(model,sequence[:count],features,device,loss)
    optimizer=comparison.make_optimizer(model);history=[]
    for step in range(pre):
        batch=data.load_training_batch(features,sequence[step],device=str(device))
        result=training.train_step(model,optimizer,batch,loss_config=loss,completed_steps=step,condition='no_consistency')
        history.append(dict(step=result.completed_steps,grad_norm_before_clip=result.grad_norm,**result.loss_values))
        if result.completed_steps%50==0:print(f'calibrate pretrain {result.completed_steps}/{pre} jepa={result.loss_values["jepa"]:.4f}',flush=True)
    rows=_gradient_rows(model,sequence[pre:pre+count],features,device,loss)
    def summary(items):
        return dict(median={k:statistics.median(r[k] for r in items) for k in items[0]},
                    mean={k:sum(r[k] for r in items)/len(items) for k in items[0]})
    after,before=summary(rows),summary(initial)
    median=after['median'];rho=cal['rho']
    result=dict(rho=rho,batches=len(rows),statistic='median',method='v6_after_jepa_pretraining',
        pretrain_updates=pre,pretrain_history=history,per_batch=rows,median=median,mean=after['mean'],
        initial_state=dict(per_batch=initial,**before,
            time_weight=rho*before['median']['grad_jepa']/before['median']['grad_time'],
            space_weight=rho*before['median']['grad_jepa']/before['median']['grad_space']),
        time_weight=rho*median['grad_jepa']/median['grad_time'],space_weight=rho*median['grad_jepa']/median['grad_space'],
        graph=dict(sigma_x=config['model']['graph']['sigma_x'],sigma_z=config['model']['graph']['sigma_z'],
                   max_modes=config['model']['graph']['max_modes']),
        v4_overlay_sha256_before=cache.file_sha(OVERLAY),
        note='time_weight/space_weight 已自动写入 configs/train/v4.yaml；三个种子共用这一组 λ')
    for key in ('time_weight','space_weight'):
        if not (result[key]>0 and result[key]<float('inf')):raise ValueError('calibrated weight must be finite and positive: '+key)
    write_calibrated_weights(result['time_weight'],result['space_weight'],f'JEPA 预训练 {pre} 步后；{output.name}')
    result['v4_overlay_sha256_after']=cache.file_sha(OVERLAY)
    cache._write_json(output/'calibration.json',result)
    print(json.dumps(dict(time_weight=result['time_weight'],space_weight=result['space_weight'],median=median,
        initial_state_weights=dict(time=result['initial_state']['time_weight'],space=result['initial_state']['space_weight'])),
        ensure_ascii=False,indent=1))
    print('已写入 configs/train/v4.yaml，可以直接进行下一步 check。',flush=True)
    return result


def execute(args):
    """统一真实入口：检查失败即停止，保留退出证据；本地CPU测试不走此生产入口。"""
    if args.command=='calibrate':
        verify_bundle();return calibrate_weights(Path(args.features_root),Path(args.output).resolve(),torch.device(args.device)) and 0
    bundle_sha=verify_bundle();device=torch.device(args.device)
    if device.type!='cuda' or device.index is None:raise ValueError('explicit cuda:N target required')
    output=Path(args.output).resolve()
    if args.command=='resume' and output.is_relative_to(Path(args.resume_from).resolve()):
        raise ValueError('resume output must be outside the old training directory')
    output.mkdir(parents=True,exist_ok=False)
    stage='environment'
    try:
        cache._write_json(output/'environment.json',environment.verify_versions())
        stage='calibration'
        require_calibrated_weights()
        stage='preflight'
        receipt=preflight(Path(args.features_root),output/'check',device)
        if args.command in ('train','run','resume'):
            stage='training'
            if args.command=='resume':
                train_conditions(Path(args.features_root),output/'training',device,resume_from=args.resume_from)
            else: train_conditions(Path(args.features_root),output/'training',device)
        if args.command in ('evaluate','run'):
            stage='evaluation'
            train_root=output/'training' if args.command=='run' else Path(args.training_run)
            evaluate_conditions(Path(args.features_root),train_root,output/'evaluation',device)
        if verify_bundle()!=bundle_sha:raise ValueError('source bundle changed during run')
        cache._write_json(output/'COMPLETE.json',dict(command=args.command,manifest_sha256=bundle_sha,
            real_device=receipt['hardware'],scientific_conclusions_verified=False))
        return 0
    except Exception as error:
        cache._write_json(output/'FAILED.json',dict(stage=stage,error_type=type(error).__name__,
            message=str(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
