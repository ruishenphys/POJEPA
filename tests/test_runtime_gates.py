"""交付修复的CPU边界检查；全部使用虚构设备与小型状态，不访问真实资产。"""
from argparse import Namespace
from dataclasses import asdict, replace
import copy
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from po_jepa.runtime import workflow as w


def test_environment_rejects_wrong_version_before_assets(monkeypatch, tmp_path):
    """错误依赖不能消费GPU、缓存或进入正式训练。"""
    from po_jepa.runtime import environment
    monkeypatch.setattr(environment, 'versions', lambda: dict(python='3.12.14', torch='2.6.0+cu124',
        cuda='12.4', numpy='0.0.0', PyYAML='6.0.2'))
    monkeypatch.setattr(w, 'verify_bundle', lambda: 'fictional')
    monkeypatch.setattr(w, 'preflight', lambda *a: pytest.fail('must stop before GPU/assets'))
    args=Namespace(command='run', device='cuda:0', features_root=tmp_path/'absent', output=tmp_path/'run')
    with pytest.raises(ValueError, match='numpy'): w.execute(args)
    assert w.read(args.output/'FAILED.json')['stage']=='environment'
    assert not (args.output/'COMPLETE.json').exists()


def test_cli_accepts_explicit_resume_without_running(monkeypatch, tmp_path):
    """显式旧目录和新目录进入唯一编排，不自动选择其他运行。"""
    from po_jepa.runtime import cli
    seen=[]
    monkeypatch.setattr(w, 'execute', lambda args: seen.append(args) or 0)
    assert cli.main(['resume', '--features-root', str(tmp_path/'absent'),
        '--resume-from', str(tmp_path/'old'), '--output', str(tmp_path/'new')])==0
    assert seen[0].resume_from==tmp_path/'old'


def test_resume_rejects_nested_output_before_creating_or_computing(monkeypatch,tmp_path):
    """新目录也不能位于旧训练目录内，避免递归复制或写入原证据。"""
    monkeypatch.setattr(w,'verify_bundle',lambda:'fictional')
    args=Namespace(command='resume',device='cuda:0',features_root=tmp_path/'absent',
        resume_from=tmp_path/'old',output=tmp_path/'old/new')
    with pytest.raises(ValueError,match='outside'):w.execute(args)
    assert not args.output.exists()


def test_capture_wrapper_rejects_nested_output_without_creating_console(tmp_path):
    """外层捕获入口也须提前拒绝，否则console会先写进原目录。"""
    import importlib.util
    spec=importlib.util.spec_from_file_location('fictional_capture_entry',w.ROOT/'run.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    old=tmp_path/'old';new=old/'new'
    with pytest.raises(SystemExit) as error:
        entry.main(['resume','--features-root',str(tmp_path/'absent'),
            '--resume-from',str(old),'--output',str(new)])
    assert error.value.code==2 and not old.exists()


def test_resume_runtime_signature_rejects_thread_change(monkeypatch,tmp_path):
    """同源码仍须核原运行线程/库/设备设置，不能靠相同种子推断恢复等价。"""
    from po_jepa.runtime import environment,resume
    runtime=environment.runtime_signature(torch.device('cpu'));registration={'fictional':True}
    w.cache._write_json(tmp_path/'preregistration.json',registration)
    w.cache._write_json(tmp_path/'runtime.json',runtime)
    resume.verify_training_source(tmp_path,registration,runtime)
    monkeypatch.setattr(torch,'get_num_threads',lambda:runtime['cpu_threads']+1)
    with pytest.raises(ValueError,match='signature'):
        resume.verify_training_source(tmp_path,registration,environment.runtime_signature(torch.device('cpu')))


@pytest.mark.parametrize('failure',[False,True])
def test_preflight_resource_gate_blocks_completion(monkeypatch,tmp_path,failure):
    """直接检查真实预检编排；虚构设备替身不会访问CUDA，资源失败不发布完成标记。"""
    monkeypatch.setattr(w.checks,'device_metadata',lambda *a:{'fictional':True})
    monkeypatch.setattr(w.data,'approved_cache_metadata',lambda *a:{'fictional':True})
    monkeypatch.setattr(w.comparison,'build_condition',lambda *a:(object(),object()))
    monkeypatch.setattr(w.comparison,'make_optimizer',lambda *a:object())
    monkeypatch.setattr(w.data,'load_training_batch',lambda *a,**k:object())
    monkeypatch.setattr(w.data,'write_collections',lambda *a:None)
    monkeypatch.setattr(w.checks,'diagnostic_step',lambda *a:{'fictional':True})
    monkeypatch.setattr(w.checks,'_memory',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a:None)
    calls=[]
    def resource(*a):
        calls.append('resource')
        if failure:raise RuntimeError('fictional resource failure')
        return {'completed_updates':100,'validation':{'windows':400}}
    monkeypatch.setattr(w.resources,'resource_check',resource)
    if failure:
        with pytest.raises(RuntimeError,match='resource'):w.preflight(None,tmp_path/'check',torch.device('cpu'))
        assert not (tmp_path/'check/COMPLETE.json').exists()
    else:
        result=w.preflight(None,tmp_path/'check',torch.device('cpu'))
        assert result['resource_gate_completed'] and (tmp_path/'check/COMPLETE.json').exists()
    assert calls==['resource']


@pytest.mark.parametrize('key,value',[('python','3.13.0'),('torch','2.12.0+cu130'),
    ('cuda',None),('numpy','2.4.2'),('PyYAML','6.0.2')])
def test_each_fixed_environment_boundary(monkeypatch,key,value):
    """每项固定依赖分别拒绝；相同CUDA编译版本与Python3.12补丁版本允许。"""
    from po_jepa.runtime import environment as e
    # v4：固定版本与一轮实际运行环境（patho）一致
    versions=dict(python='3.12.13',torch='2.11.0+cu130',cuda='13.0',numpy='2.4.3',PyYAML='6.0.3')
    monkeypatch.setattr(e,'versions',lambda:versions.copy())
    assert e.verify_versions()==versions
    versions[key]=value
    with pytest.raises(ValueError,match=key):e.verify_versions()


@pytest.mark.parametrize('failure',[False,True])
def test_resource_gate_exact_budget_and_rng_isolation(monkeypatch,tmp_path,failure):
    """虚构资源探针核100正常更新/400窗及异常传播，不宣称CPU测得GPU成本。"""
    from po_jepa.runtime import resources as r
    queries=[[w.data.WindowQuery(0,0,1.)]*4 for _ in range(100)]; calls=[]
    monkeypatch.setattr(r.comparison,'build_condition',lambda *a:(object(),object()))
    monkeypatch.setattr(r.comparison,'make_optimizer',lambda *a:object())
    monkeypatch.setattr(r.checks,'make_query_sequence',lambda:queries)
    monkeypatch.setattr(r.data,'load_training_batch',lambda *a,**k:object())
    def update(*a,completed_steps,**k):
        calls.append(completed_steps); random.random();np.random.random();torch.rand(1)
        if failure and completed_steps==3:raise RuntimeError('fictional update failure')
        return SimpleNamespace(completed_steps=completed_steps+1,loss_values={'total':1.})
    monkeypatch.setattr(r.training,'train_step',update)
    def validation(*a):
        calls.append('validation');return dict(windows=400,mean=1.,seconds=.01)
    monkeypatch.setattr(r.comparison,'validate_condition',validation)
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    with r.isolated_randomness(torch.device('cpu')):
        if failure:
            with pytest.raises(RuntimeError,match='fictional'):r.resource_check(None,tmp_path/'cost',torch.device('cpu'))
        else:
            report=r.resource_check(None,tmp_path/'cost',torch.device('cpu'))
            assert report['completed_updates']==100 and report['validation']['windows']==400
            assert report['training_memory'] is None and not report['formal_training_state_reused']
    assert calls==list(range(4)) if failure else calls==list(range(100))+['validation']
    assert random.getstate()==before[0] and np.array_equal(np.random.get_state()[1],before[1][1])
    assert torch.equal(torch.get_rng_state(),before[2])


@pytest.mark.parametrize('condition',w.comparison.CONDITIONS)
@pytest.mark.parametrize('stop_at',[0,2])
def test_actual_resume_matches_continuous_state(monkeypatch,tmp_path,condition,stop_at):
    """真实小模型/AdamW保存恢复：中断后只补剩余更新，参数/动量/RNG与连续执行相同。"""
    from test_training import e06_case,e06_config
    from po_jepa.baselines import TokenTransformer,TransformerSpec
    c=w.comparison; base,batch=e06_case()
    if condition=='matched_transformer':base=TokenTransformer(TransformerSpec(encoder_dim=3))
    config=e06_config() if condition=='po_jepa_full' else e06_config(time_weight=0.,space_weight=0.)
    sequence=[[w.data.WindowQuery(0,index,1.)] for index in range(3)]
    state=torch.Generator().manual_seed(2703).get_state(); count=c.parameter_counts(base)['total']
    registration=dict(parameter_match=c.parameter_match(count,count),query_sequence_sha256=c.sequence_sha(sequence),
        consumer_sha256=c.consumer_sources(),comparison_config_sha256=c.data.cache.file_sha(c.ROOT/'configs/train/comparison.yaml'))
    monkeypatch.setattr(c.data,'load_training_batch',lambda *a,**k:batch)
    monkeypatch.setattr(c.data,'write_collections',lambda *a,**k:{})
    def validate(model,features,output,*a):
        output.mkdir()
        expected=[dict(trajectory_id=t,start_tick=s,duration_cycles=d) for t in range(160,180)
            for s in (0,5,10,15,20) for d in (1.,2.,4.,8.)]
        for offset in range(0,400,4):w.cache._write_json(output/f'{offset:04d}.json',
            dict(queries=expected[offset:offset+4],values={'future_jepa_distance':[1.]*4}))
        return dict(mean=1.,windows=400,metric='future_jepa_distance')
    monkeypatch.setattr(c,'validate_condition',validate)
    # v6：每步都量一致性梯度占比，续训仍须与连续训练逐位一致（测量不改参数、优化器或 RNG）。
    monkeypatch.setattr(w.checks,'load_protocol',lambda:{'validation':{'interval_updates':2},
        'loss':asdict(e06_config()),'monitoring':{'consistency_interval':1,'early_check_updates':10**6,'min_share':.005},'calibration':{'rho':.05}})
    original=c.condition_step; active={'fail':False}; attempted=[]
    def step(*a,**k):
        completed=a[-1];attempted.append(completed)
        random.random();np.random.random();torch.rand(1)
        if active['fail'] and completed==stop_at:raise RuntimeError('fictional interruption')
        return original(*a,**k)
    monkeypatch.setattr(c,'condition_step',step)
    def run(output,source=None):
        model=copy.deepcopy(base);optimizer=c.make_optimizer(model)
        kwargs=dict(sequence=sequence,generator_state=state,registration=registration,features=tmp_path/'absent',
            output=output,device=torch.device('cpu'),loss_config=config,validation_interval=2)
        if source:kwargs['resume_from']=source
        report=c.run_prepared_condition(condition,model,optimizer,**kwargs)
        return model,optimizer,report
    random.seed(31);np.random.seed(31);torch.manual_seed(31)
    reference,reference_optimizer,_=run(tmp_path/'continuous')
    reference_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    first=w.read(tmp_path/'continuous/updates/0001.json')['consistency_gradient']
    if condition=='matched_transformer':assert first is None
    else:
        assert set(first['share'])=={'time','space'} and first['grad_jepa']>0
        if condition=='po_jepa_no_consistency':assert first['share']=={'time':0.,'space':0.}
    random.seed(31);np.random.seed(31);torch.manual_seed(31);active['fail']=True
    with pytest.raises(RuntimeError,match='interruption'):run(tmp_path/'interrupted')
    failure_sha=w.cache.file_sha(tmp_path/'interrupted/FAILED.json')
    active['fail']=False;attempted.clear()
    resumed,optimizer,report=run(tmp_path/'resumed',tmp_path/'interrupted')
    assert attempted==list(range(stop_at,3)) and report['completed_updates']==3
    assert all(torch.equal(value,resumed.state_dict()[key]) for key,value in reference.state_dict().items())
    torch.testing.assert_close(reference_optimizer.state_dict(),optimizer.state_dict(),rtol=0,atol=0)
    assert random.getstate()==reference_rng[0] and np.array_equal(np.random.get_state()[1],reference_rng[1][1])
    assert torch.equal(torch.get_rng_state(),reference_rng[2])
    assert w.cache.file_sha(tmp_path/'interrupted/FAILED.json')==failure_sha
    assert not (tmp_path/'resumed/FAILED.json').exists()
    if condition=='po_jepa_full' and stop_at==2:
        checkpoint=tmp_path/'interrupted/checkpoints/step-0002.pt'
        checkpoint.write_bytes(checkpoint.read_bytes()+b'fictional corruption')
        attempted.clear()
        with pytest.raises(ValueError,match='SHA'):run(tmp_path/'rejected',tmp_path/'interrupted')
        assert attempted==[] and (tmp_path/'rejected/FAILED.json').exists()
        assert w.cache.file_sha(tmp_path/'interrupted/FAILED.json')==failure_sha
