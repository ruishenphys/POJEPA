"""显式续训：只读旧产物，在新目录接续最后完整验证点，不重标旧版来源。"""
from dataclasses import asdict
from pathlib import Path
import shutil
import hashlib
import torch
from . import cache, comparison, evaluation, learning


def read(path):
    """复用严格JSON读取，拒绝重复字段及非有限值。"""
    from .workflow import read as strict_read
    return strict_read(path)


def verify_training_source(source, registration, runtime):
    """旧目录登记须绑定本版源码、协议、缓存消费者及同一完整查询序列。"""
    source=Path(source).resolve()
    if read(source/'preregistration.json')!=registration:
        raise ValueError('resume preregistration differs from current source/config/query sequence')
    if read(source/'runtime.json')!=runtime:
        raise ValueError('resume runtime/thread/device signature differs')
    if not source.is_dir(): raise ValueError('resume training directory missing')


def copy_completed_condition(source, output, registration, sequence, state):
    """已完整完成的条件只核验并复制，避免再次消耗更新预算；旧目录保持只读。"""
    from .workflow import verify_selection
    if Path(output).resolve().is_relative_to(Path(source).resolve()):
        raise ValueError('resume output must be outside the old condition directory')
    report=read(source/'report.json')
    verify_ledger(source,sequence,state,registration,report['completed_updates'])
    if ((source/'FAILED.json').exists() or report['condition']!=source.name
        or report['completed_updates']!=len(sequence) or report['evidence_role']!='real_frozen_features'
        or report['query_sequence_sha256']!=registration['query_sequence_sha256']
        or report['preregistration_sha256']!=hashlib.sha256(cache._json_bytes(registration)).hexdigest()
        or read(source/'registration.json')['preregistration']!=registration
        or read(source/'COMPLETE.json')['report_sha256']!=cache.file_sha(source/'report.json')):
        raise ValueError('resume completed condition has inconsistent evidence')
    verify_selection(source,report)
    for row in report['validations']:
        if row['checkpoint']!=f"checkpoints/step-{row['step']:04d}.pt":
            raise ValueError('resume completed checkpoint path differs')
        if cache.file_sha(source/row['checkpoint'])!=row['checkpoint_sha256']:
            raise ValueError('resume completed checkpoint SHA differs')
    shutil.copytree(source,output)
    return report


def verify_sequence(source,sequence,state,registration):
    """部分恢复和已完成条件跳过均核实际序列/游标，不只信任声明的SHA。"""
    recorded=read(source/'query-sequence.json')
    expected=dict(rows=[[asdict(q) for q in rows] for rows in sequence],
        query_sequence_sha256=registration['query_sequence_sha256'],generator_state=state.tolist(),
        generator_state_stage='after_complete_sequence_pre_generation')
    if not comparison.training._same_native(recorded,expected):
        raise ValueError('resume actual query sequence/generator differs')


def verify_updates(source,sequence,completed):
    """核每次成功更新的查询和游标；完整条件须恰好有预算内全部记录。

    账本是训练过程留下的逐步记录，不能只凭报告中的总步数判定完整。
    中断条件只核保存点前缀，保留后续未保存更新，供明确回退后重算。
    """
    comparison.training._count(completed,'completed_updates')
    if completed>len(sequence): raise ValueError('update cursor outside registered budget')
    folder=Path(source)/'updates'
    names={f'{number:04d}.json' for number in range(1,completed+1)}
    if completed==len(sequence) and {p.name for p in folder.glob('*.json')}!=names:
        raise ValueError('complete update ledger file inventory differs')
    for number in range(1,completed+1):
        update=read(folder/f'{number:04d}.json')
        expected=dict(completed_updates=number,next_batch_index=number,
            queries=[asdict(q) for q in sequence[number-1]])
        if type(update) is not dict or not comparison.training._same_native(
            {key:update.get(key) for key in expected},expected):
            raise ValueError('resume update query/cursor differs')


def verify_ledger(source,sequence,state,registration,completed):
    """评价、完整复用和中断恢复共用的证据核验，不加载模型或改变训练状态。"""
    verify_sequence(Path(source),sequence,state,registration)
    verify_updates(source,sequence,completed)


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
    for row in selections:
        expected=f"checkpoints/step-{row['step']:04d}.pt"
        if row['checkpoint']!=expected or cache.file_sha(source/expected)!=row['checkpoint_sha256']:
            raise ValueError('resume checkpoint path/SHA differs')
    verify_updates(source,sequence,step)
    return step,selections,best


def _copy_prefix(source,output,step,rows):
    """只复制可接续前缀；旧失败和超出保存点的更新仍留在原目录，绝不删除。"""
    for name in ('query-sequence.json','registration.json'):
        shutil.copy2(source/name,output/name)
    if (source/'source-collections').is_dir():
        shutil.copytree(source/'source-collections',output/'source-collections',dirs_exist_ok=True)
    for number in range(1,step+1):
        shutil.copy2(source/'updates'/f'{number:04d}.json',output/'updates'/f'{number:04d}.json')
    for row in rows:
        shutil.copy2(source/row['checkpoint'],output/row['checkpoint'])
        name=f"{row['step']:04d}"
        shutil.copytree(source/'validations'/name,output/'validations'/name)
        shutil.copy2(source/'validations'/f'selection-{name}.json',output/'validations'/f'selection-{name}.json')


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
