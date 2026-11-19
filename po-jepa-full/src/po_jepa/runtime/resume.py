"""显式续训：只读旧产物，在新目录接续最后完整验证点，不重标旧版来源。"""
from dataclasses import asdict
from pathlib import Path
import shutil
import hashlib
import torch
from . import cache, comparison, evaluation, learning


def _copystat(src, dst):
    """复制时间戳和权限位；结果盘是 ntfs-3g/fuseblk 时非属主改时间会 PermissionError（v7.1 机哥本机补丁第 2 处）。
    这些只是元数据，续训和完整复用核的是账本、预登记和 SHA，不看时间戳，所以只忽略 PermissionError。
    机哥的补丁是全局替换 shutil.copystat；这里只用于本模块的复制，不改别的模块的 shutil 行为。"""
    try:
        shutil.copystat(src, dst)
    except PermissionError:
        pass


def _copy_file(src, dst):
    """本模块里 copy2（文件到文件）调用的替代：内容照复制，元数据复制失败（PermissionError）不阻断。
    只接受目标文件路径（不像 copy2 那样接受目标目录），返回目标路径。"""
    if Path(dst).is_dir():
        raise IsADirectoryError(f"_copy_file needs a file path, got directory {dst}")
    shutil.copyfile(src, dst)
    _copystat(src, dst)
    return dst


def _copy_tree(src, dst, *, dirs_exist_ok=False):
    """shutil.copytree 的替代（逐层建目录、_copy_file 复制文件，目录的元数据同样容忍 PermissionError）。
    shutil.copytree 在非 Windows 上会把目录 copystat 的失败收集起来最后抛 shutil.Error，所以不能只换 copy_function。"""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=dirs_exist_ok)
    for entry in sorted(src.iterdir()):
        if entry.is_dir():
            _copy_tree(entry, dst/entry.name, dirs_exist_ok=dirs_exist_ok)
        else:
            _copy_file(entry, dst/entry.name)
    _copystat(src, dst)


def read(path):
    """复用严格JSON读取，拒绝重复字段及非有限值。"""
    from .workflow import read as strict_read
    return strict_read(path)


def verify_training_source(source, registration, runtime):
    """旧目录登记须绑定本版源码、协议、缓存消费者及同一完整查询序列。"""
    from . import environment
    relaxed=environment.portable()   # 先解析开关：非法值在签名相同时也要报错（C8-fix1，Codex P2-1）
    source=Path(source).resolve()
    if read(source/'preregistration.json')!=registration:
        raise ValueError('resume preregistration differs from current source/config/query sequence')
    old=read(source/'runtime.json')
    if old!=runtime:
        differences=environment.runtime_differences(old,runtime) or ['<structure>']
        if not relaxed:
            raise ValueError('resume runtime/thread/device signature differs: '+', '.join(differences)
                             +f' (on a cluster that moves jobs between GPUs set {environment.PORTABLE_ENV}=1)')
        other=[k for k in differences if k not in environment.PORTABLE_RESUME_KEYS]
        if other:   # C8：可移植模式只放过机器、卡和 CPU 线程数；版本和计算设置（精度、TF32、确定性开关）必须相同
            raise ValueError('resume runtime differs beyond GPU/host placement: '+', '.join(other))
        import sys
        print(f'{environment.PORTABLE_ENV}=1: resuming on another GPU/host, differing fields: '+', '.join(differences),
              file=sys.stderr, flush=True)
    if not source.is_dir(): raise ValueError('resume training directory missing')


def copy_completed_condition(source, output, registration, sequence, state):
    """已完整完成的条件只核验并复制，避免再次消耗更新预算；旧目录保持只读。"""
    from .workflow import verify_selection
    if Path(output).resolve().is_relative_to(Path(source).resolve()):
        raise ValueError('resume output must be outside the old condition directory')
    report=read(source/'report.json')
    verify_ledger(source,sequence,state,registration,report['completed_updates'],final=True)
    if ((source/'FAILED.json').exists() or report['condition']!=source.name
        or comparison.verify_stopping(report['validations'],report['completed_updates'],len(sequence),
                                      registration.get('early_stopping'))!=report.get('early_stopping')
        or report['evidence_role']!='real_frozen_features'
        or report['query_sequence_sha256']!=registration['query_sequence_sha256']
        or report['preregistration_sha256']!=hashlib.sha256(cache._json_bytes(registration)).hexdigest()
        or read(source/'registration.json')['preregistration']!=registration
        or read(source/'COMPLETE.json')['report_sha256']!=cache.file_sha(source/'report.json')):
        raise ValueError('resume completed condition has inconsistent evidence')
    verify_selection(source,report)
    keep=comparison.kept_checkpoint_steps(report['validations'],registration.get('checkpoint_retention'),source.name)
    for row in report['validations']:
        if row['checkpoint']!=f"checkpoints/step-{row['step']:04d}.pt":
            raise ValueError('resume completed checkpoint path differs')
        present=(source/row['checkpoint']).exists()
        if row['step'] in keep and (not present or cache.file_sha(source/row['checkpoint'])!=row['checkpoint_sha256']):
            raise ValueError('resume completed checkpoint SHA differs')
        if row['step'] not in keep and present:
            raise ValueError('completed run kept a checkpoint the retention rule deletes')
    _copy_tree(source,output)
    return report


def verify_sequence(source,sequence,state,registration):
    """部分恢复和已完成条件跳过均核实际序列/游标，不只信任声明的SHA。"""
    recorded=read(source/'query-sequence.json')
    expected=dict(rows=[[asdict(q) for q in rows] for rows in sequence],
        query_sequence_sha256=registration['query_sequence_sha256'],generator_state=state.tolist(),
        generator_state_stage='after_complete_sequence_pre_generation')
    if not comparison.training._same_native(recorded,expected):
        raise ValueError('resume actual query sequence/generator differs')


def verify_updates(source,sequence,completed,final=None):
    """核每次成功更新的查询和游标；完整条件须恰好有预算内全部记录。

    账本是训练过程留下的逐步记录，不能只凭报告中的总步数判定完整。
    中断条件只核保存点前缀，保留后续未保存更新，供明确回退后重算。
    """
    comparison.training._count(completed,'completed_updates')
    if completed>len(sequence): raise ValueError('update cursor outside registered budget')
    folder=Path(source)/'updates'
    names={f'{number:04d}.json' for number in range(1,completed+1)}
    final=completed==len(sequence) if final is None else final   # 完成的训练（含早停）账本恰好是 1..completed
    if final and {p.name for p in folder.glob('*.json')}!=names:
        raise ValueError('complete update ledger file inventory differs')
    for number in range(1,completed+1):
        update=read(folder/f'{number:04d}.json')
        expected=dict(completed_updates=number,next_batch_index=number,
            queries=[asdict(q) for q in sequence[number-1]])
        if type(update) is not dict or not comparison.training._same_native(
            {key:update.get(key) for key in expected},expected):
            raise ValueError('resume update query/cursor differs')


def verify_ledger(source,sequence,state,registration,completed,final=None):
    """评价、完整复用和中断恢复共用的证据核验，不加载模型或改变训练状态。final=True：已完成的训练。"""
    verify_sequence(Path(source),sequence,state,registration)
    verify_updates(source,sequence,completed,final)


def _prefix(source, sequence, state, registration, interval):
    """核完整验证前缀；不将中断的验证目录或部分更新冒充成功保存点。"""
    from .workflow import verify_selection
    verify_sequence(source,sequence,state,registration)
    selections=[read(p) for p in sorted((source/'validations').glob('selection-*.json'))]
    if not selections: raise ValueError('no complete validated checkpoint; start a new run explicitly')
    step=selections[-1]['step']
    if type(step) is not int or step<0 or step>len(sequence): raise ValueError('resume step outside budget')
    expected=list(range(0,step+1,interval))
    if step==len(sequence) and expected[-1]!=step: expected.append(step)
    if [row['step'] for row in selections]!=expected:
        raise ValueError('resume validation prefix differs')
    best=min(selections,key=lambda row:(row['mean'],row['step']))
    verify_selection(source,dict(completed_updates=step,validations=selections,best=best))
    rule=registration.get('early_stopping')
    for k in range(1,len(selections)):
        if comparison.early_stop_decision(selections[:k],len(sequence),rule)['stop']:
            raise ValueError('resume source trained past an early-stop point')
    # 训练还可能继续的前缀按训练中的保留集核（最好点 ±k 加最近 k 个）；最后一个验证点已经是终点（跑满或触发早停）时，
    # 源目录可能已经压缩成最终保留集，按最终保留集核（v7-P2-1）
    final=selections[-1]['step']==len(sequence) or comparison.early_stop_decision(selections,len(sequence),rule)['stop']
    keep=comparison.kept_checkpoint_steps(selections,registration.get('checkpoint_retention'),source.name,final)
    for row in selections:
        expected=f"checkpoints/step-{row['step']:04d}.pt"
        if row['checkpoint']!=expected:
            raise ValueError('resume checkpoint path/SHA differs')
        if row['step'] in keep and cache.file_sha(source/expected)!=row['checkpoint_sha256']:
            raise ValueError('resume checkpoint path/SHA differs')
    verify_updates(source,sequence,step)
    return step,selections,best


def _copy_prefix(source,output,step,rows):
    """只复制可接续前缀；旧失败和超出保存点的更新仍留在原目录，绝不删除。"""
    for name in ('query-sequence.json','registration.json'):
        _copy_file(source/name,output/name)
    if (source/'source-collections').is_dir():
        _copy_tree(source/'source-collections',output/'source-collections',dirs_exist_ok=True)
    for number in range(1,step+1):
        _copy_file(source/'updates'/f'{number:04d}.json',output/'updates'/f'{number:04d}.json')
    for row in rows:
        if (source/row['checkpoint']).exists():   # 保留规则删掉的检查点不复制（_prefix 已核保留集都在且 SHA 一致）
            _copy_file(source/row['checkpoint'],output/row['checkpoint'])
        name=f"{row['step']:04d}"
        _copy_tree(source/'validations'/name,output/'validations'/name)
        _copy_file(source/'validations'/f'selection-{name}.json',output/'validations'/f'selection-{name}.json')


def prepare_condition(source, output, condition, model, optimizer, sequence, state,
                      registration, features, device, config, interval):
    """核原始状态与新读取缓存来源后恢复模型/AdamW/RNG/游标，再接续原循环。

    只支持同版同运行环境。可能回退至最后完整验证点，随后已执行但未保存的更新
    会重算；回退步数明确记录，不把checkpoint后的失败状态接进优化器。
    """
    source=Path(source).resolve(); record=read(source/'registration.json')
    if record['condition']!=condition or record['preregistration']!=registration:
        raise ValueError('resume condition/preregistration differs')
    step,rows,best=_prefix(source,sequence,state,registration,interval)
    _copy_prefix(source,output,step,rows)
    batch=learning._learning_batch(features,sequence[max(0,step-1)],device,output,registration['query_sequence_sha256'])
    provenance=comparison._provenance(batch.provenance)
    if provenance!=rows[-1]['checkpoint_provenance']:
        raise ValueError('resume current cache/source provenance differs')
    progress=evaluation.load_selected_checkpoint(condition,rows[-1],model=model,optimizer=optimizer,
        config=config,expected_provenance=provenance,root=source)
    if (progress.completed_steps!=step or progress.sampling_state['next_batch_index']!=step
        or not torch.equal(progress.sampling_state['query_generator_state'],state)):
        raise ValueError('resume restored cursor/generator differs')
    cache._write_json(output/'RESUMED.json',dict(source_directory=str(source),restored_step=step,
        checkpoint_sha256=rows[-1]['checkpoint_sha256'],old_failure_preserved=True,
        source_failure_sha256=cache.file_sha(source/'FAILED.json') if (source/'FAILED.json').exists() else None,
        updates_after_checkpoint_replayed=True,automatic_retry=False))
    return step,rows,best,record['evidence_role']
