"""用虚构账本检查评价/续训证据；不构造模型、不读真实特征或权重。"""
from dataclasses import asdict
import hashlib
import json
import pytest
import torch
from po_jepa.runtime import comparison, resume, workflow


@pytest.fixture
def ledger(tmp_path):
    """两次更新、每次两条不同查询，使调序和错误游标都可被发现。"""
    sequence=[[comparison.data.WindowQuery(0,0,1.),comparison.data.WindowQuery(1,1,2.)],
        [comparison.data.WindowQuery(2,2,4.),comparison.data.WindowQuery(3,3,8.)]]
    state=torch.Generator().manual_seed(23).get_state()
    registration=dict(query_sequence_sha256=comparison.sequence_sha(sequence))
    (tmp_path/'updates').mkdir()
    comparison.data.cache._write_json(tmp_path/'query-sequence.json',dict(
        rows=[[asdict(q) for q in rows] for rows in sequence],
        query_sequence_sha256=registration['query_sequence_sha256'],generator_state=state.tolist(),
        generator_state_stage='after_complete_sequence_pre_generation'))
    for number,rows in enumerate(sequence,1):
        comparison.data.cache._write_json(tmp_path/'updates'/f'{number:04d}.json',dict(
            completed_updates=number,next_batch_index=number,queries=[asdict(q) for q in rows],
            losses={'fictional':1.}))
    return tmp_path,sequence,state,registration


def test_valid_ledger_is_read_only_and_keeps_rng(ledger):
    """核验只读取证据，不改原文件或CPU随机状态；正常完整记录须通过。"""
    root,sequence,state,registration=ledger
    before={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    rng=torch.get_rng_state().clone()
    resume.verify_ledger(root,sequence,state,registration,len(sequence))
    assert before=={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert torch.equal(torch.get_rng_state(),rng)


@pytest.mark.parametrize('damage',['missing_sequence','sequence_order','sequence_bool','generator',
    'generator_stage','missing_update','query','update_order','cursor','completed','bool_cursor',
    'float_completed','extra_update','misnumbered','duplicate_key'])
def test_corrupt_ledger_refused_before_downstream(ledger,damage):
    """原记录缺失、查询/游标被改或JSON含糊都须报错，不能修补后放行。"""
    root,sequence,state,registration=ledger
    path=root/('query-sequence.json' if damage in ('missing_sequence','sequence_order',
        'sequence_bool','generator','generator_stage') else 'updates/0001.json')
    row=json.loads(path.read_bytes())
    if damage.startswith('missing'):path.unlink()
    elif damage=='extra_update':(root/'updates/0003.json').write_text('{}',encoding='utf-8')
    elif damage=='misnumbered':path.rename(root/'updates/1.json')
    elif damage=='duplicate_key':path.write_text('{"completed_updates":1,"completed_updates":2}',encoding='utf-8')
    else:
        if damage=='sequence_order':row['rows'].reverse()
        elif damage=='sequence_bool':row['rows'][0][1]['trajectory_id']=True
        elif damage=='generator':row['generator_state'][0]^=1
        elif damage=='generator_stage':row['generator_state_stage']='before_generation'
        elif damage=='query':row['queries'][0]['duration_cycles']=12.
        elif damage=='update_order':row['queries'].reverse()
        elif damage=='cursor':row['next_batch_index']=2
        elif damage=='completed':row['completed_updates']=2
        elif damage=='bool_cursor':row['next_batch_index']=True
        elif damage=='float_completed':row['completed_updates']=1.
        path.write_text(json.dumps(row),encoding='utf-8')
    with pytest.raises((ValueError,FileNotFoundError)):
        resume.verify_ledger(root,sequence,state,registration,len(sequence))


def test_partial_prefix_keeps_uncheckpointed_tail(ledger):
    """回到第一步保存点时不验证/复制第二步失败状态；完整模式仍须拒绝它。"""
    root,sequence,state,registration=ledger
    (root/'updates/0002.json').write_text('not a complete update',encoding='utf-8')
    resume.verify_ledger(root,sequence,state,registration,1)
    with pytest.raises(ValueError):resume.verify_ledger(root,sequence,state,registration,2)
    assert (root/'updates/0002.json').read_text(encoding='utf-8')=='not a complete update'


@pytest.mark.parametrize('completed',[-1,True,1.,3])
def test_invalid_consumption_cursor(ledger,completed):
    """内部消费游标也须是预算内原生整数，不能让bool/浮点数冒充完成步数。"""
    root,sequence,state,registration=ledger
    with pytest.raises((TypeError,ValueError)):
        resume.verify_ledger(root,sequence,state,registration,completed)


@pytest.fixture
def closed_run(monkeypatch,tmp_path):
    """完整虚构训练目录；真实400窗选点文件保留，仅checkpoint字节为占位物。

    real_frozen_features只为进入被测证据分支的夹具标记，不代表执行过真实训练。
    """
    sequence=[[comparison.data.WindowQuery(0,0,1.)]]
    state=torch.Generator().manual_seed(23).get_state()
    registration=dict(query_sequence_sha256=comparison.sequence_sha(sequence),delivery_manifest_sha256='fictional')
    monkeypatch.setattr(comparison,'preregister',lambda:(dict(registration),sequence,state))
    monkeypatch.setattr(workflow,'verify_bundle',lambda:'fictional')
    monkeypatch.setattr(workflow.checks,'load_protocol',lambda:{'validation':{'interval_updates':1}})
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);comparison.data.cache._write_json(path,value)
    expected=[dict(trajectory_id=t,start_tick=s,duration_cycles=d) for t in range(160,180)
        for s in (0,5,10,15,20) for d in (1.,2.,4.,8.)]
    reports={};write(tmp_path/'preregistration.json',registration)
    for condition in comparison.CONDITIONS:
        folder=tmp_path/condition;rows=[]
        write(folder/'query-sequence.json',dict(rows=[[asdict(q) for q in batch] for batch in sequence],
            query_sequence_sha256=registration['query_sequence_sha256'],generator_state=state.tolist(),
            generator_state_stage='after_complete_sequence_pre_generation'))
        write(folder/'registration.json',dict(condition=condition,preregistration=registration))
        write(folder/'updates/0001.json',dict(completed_updates=1,next_batch_index=1,queries=[asdict(sequence[0][0])]))
        for step in (0,1):
            checkpoint=folder/f'checkpoints/step-{step:04d}.pt'
            checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes(b'fictional hash-only checkpoint')
            row=dict(step=step,mean=1.,windows=400,metric='future_jepa_distance',
                checkpoint=f'checkpoints/step-{step:04d}.pt',checkpoint_sha256=comparison.data.cache.file_sha(checkpoint))
            rows.append(row)
            for offset in range(0,400,4):write(folder/f'validations/{step:04d}/{offset:04d}.json',
                dict(queries=expected[offset:offset+4],values={'future_jepa_distance':[1.]*4}))
        report=dict(condition=condition,completed_updates=1,evidence_role='real_frozen_features',
            query_sequence_sha256=registration['query_sequence_sha256'],validations=rows,best=rows[0],
            preregistration_sha256=hashlib.sha256(comparison.data.cache._json_bytes(registration)).hexdigest())
        reports[condition]=report;write(folder/'report.json',report)
        write(folder/'COMPLETE.json',dict(report_sha256=comparison.data.cache.file_sha(folder/'report.json')))
    write(tmp_path/'report.json',dict(conditions=reports))
    write(tmp_path/'COMPLETE.json',dict(report_sha256=comparison.data.cache.file_sha(tmp_path/'report.json')))
    assert workflow.load_training_run(tmp_path)==reports
    return tmp_path,sequence,state,registration


def test_evaluation_entry_rejects_missing_actual_sequence(closed_run):
    """回归R05原始反例：合法目录变为缺实际序列后，评价入口必须停止。"""
    root,_,_,_=closed_run
    (root/'po_jepa_full/query-sequence.json').unlink()
    with pytest.raises(FileNotFoundError):workflow.load_training_run(root)


def test_completed_reuse_rejects_wrong_update_before_copy(closed_run,tmp_path):
    """回归R05原始反例：被改错的更新不能复制到新结果，旧证据原样保留。"""
    root,sequence,state,registration=closed_run;source=root/'po_jepa_full'
    path=source/'updates/0001.json';row=json.loads(path.read_bytes())
    row['queries'][0]['trajectory_id']=999;path.write_text(json.dumps(row),encoding='utf-8')
    damaged=path.read_bytes();output=tmp_path/'new'
    with pytest.raises(ValueError,match='query/cursor'):
        resume.copy_completed_condition(source,output,registration,sequence,state)
    assert not output.exists() and path.read_bytes()==damaged


@pytest.mark.parametrize('consumer',['evaluate','completed_reuse'])
@pytest.mark.parametrize('completed',[True,1.])
def test_consumers_reject_noninteger_report_total(closed_run,tmp_path,consumer,completed):
    """总报告即使与自身摘要一致，也不能用bool/浮点数冒充原生完成游标。"""
    root,sequence,state,registration=closed_run;folder=root/'po_jepa_full'
    report=workflow.read(folder/'report.json');report['completed_updates']=completed
    (folder/'report.json').write_text(json.dumps(report),encoding='utf-8')
    (folder/'COMPLETE.json').write_text(json.dumps(dict(report_sha256=comparison.data.cache.file_sha(folder/'report.json'))),encoding='utf-8')
    overall=workflow.read(root/'report.json');overall['conditions']['po_jepa_full']=report
    (root/'report.json').write_text(json.dumps(overall),encoding='utf-8')
    (root/'COMPLETE.json').write_text(json.dumps(dict(report_sha256=comparison.data.cache.file_sha(root/'report.json'))),encoding='utf-8')
    with pytest.raises(ValueError,match='completed_updates'):
        if consumer=='evaluate':workflow.load_training_run(root)
        else:resume.copy_completed_condition(folder,tmp_path/'new',registration,sequence,state)
