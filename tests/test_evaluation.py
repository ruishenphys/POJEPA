"""F06：全部为虚构 CPU 输入；索引与统计通过不代表真实留出效果。"""

from dataclasses import replace

import importlib

import json

from pathlib import Path

from types import SimpleNamespace

import numpy as np

import pytest

import torch

from test_training import e06_config, v4_loss_fields

@pytest.fixture
def api(monkeypatch):
    """直接核安装后的同一运行模块。"""
    return importlib.import_module("po_jepa.runtime.evaluation")

def fictional_window(api, tid=180, start=0, duration=2., action=.1):
    """沿真实B04控制定义构造虚构16×16窗口，目标与起点明确分开。"""
    c = api.comparison.data.cache
    metadata = dict(trajectory_id=tid, controls=dict(starts_ticks=[0,8,16,24],ends_ticks=[8,16,24,32],
        values=[[action,0.]]*4,units="simulation_position_per_second"))
    controls = c._window_controls(metadata,start,duration,"cpu")
    raw = torch.arange(256*3,dtype=torch.float32).reshape(1,256,3)/100
    x = c.WindowInput(raw,controls,torch.tensor([duration],dtype=torch.float64),
        api.comparison.data.cache.regular_grid(16,16),torch.tensor([start*.25],dtype=torch.float64),"input",tid,"test")
    y = c.WindowTarget(raw.flip(-1)+.3,torch.tensor([(start+duration)*.25],dtype=torch.float64),"target")
    return c.CacheWindow(x,y,dict(actual_manifest_sha256=f"manifest-{tid}",
        generation=dict(evidence_role="real_frozen_encoder"),observation_variant="canonical",fictional=True))

def test_index_1600_and_1200_unique_with_all_times(api):
    """真实固定索引不执行模型；两种网格对都含时间留出，重复行只计一次。"""
    rows=api.build_index(); single=[q for q in rows if q.kind=="point"]; pairs=[q for q in rows if q.kind=="pair"]
    assert len(single)==1600 and len(pairs)==1200
    assert len({q.key("po_jepa_full") for q in rows})==2800
    assert {q.duration_cycles for q in pairs if q.grid_size==8}=={1.,2.,3.,4.,8.,12.}
    assert {s:sum(q.stratum==s for q in single) for s in ("id","time_heldout","grid_heldout","joint_heldout")}==dict(id=800,time_heldout=200,grid_heldout=400,joint_heldout=200)
    assert api.donor_id(199)==180 and api.donor_id(180)==181
    assert all(q.start_tick+q.duration_cycles<=32 for q in rows)

@pytest.mark.parametrize("changes",[dict(trajectory_id=179),dict(trajectory_id=200),dict(start_tick=1),dict(duration_cycles=6.),dict(grid_size=4),dict(kind="other"),dict(grid_size=8,duration_cycles=3.)])
def test_invalid_query_refused(api,changes):
    with pytest.raises((ValueError,TypeError)): api.EvaluationQuery(**(dict(trajectory_id=180,start_tick=0,duration_cycles=2.,grid_size=16,kind="point")|changes))

def test_new_test_loader_and_old_train_boundary(api,monkeypatch,tmp_path):
    """只替换磁盘reader；新入口支持T=12，旧训练入口仍拒绝test和留出时长。"""
    q=api.EvaluationQuery(180,20,12.,16,"point"); w=fictional_window(api,180,20,12.)
    # 生产边界固定1408；夹具在这里扩展维度，而低层指标测试仍用小E。
    raw=w.inputs.raw_z[...,0:1].expand(1,256,1408).clone()
    w=replace(w,inputs=replace(w.inputs,raw_z=raw),target=replace(w.target,raw_z=raw.clone()))
    calls=[]
    def reader(path,**kw): calls.append((path,kw)); return w
    monkeypatch.setattr(api.comparison.data,"approved_cache_metadata",lambda p:dict(manifest_sha256_by_id={180:"manifest-180"}))
    monkeypatch.setattr(api.comparison.data.cache,"read_window",reader)
    assert api.load_test_window(tmp_path,q,"cpu").inputs.trajectory_id==180
    assert calls[0][1]["expected_split"]=="test"
    with pytest.raises(ValueError): api.comparison.data.WindowQuery(180,20,12.)
    with pytest.raises(ValueError): api.comparison.data.load_windows(tmp_path,[api.comparison.data.WindowQuery(180,0,2.)],split="test",device="cpu")
    monkeypatch.setattr(api.comparison.data.cache,"read_window",lambda *a,**k:replace(w,provenance=w.provenance|dict(actual_manifest_sha256="wrong")))
    with pytest.raises(ValueError): api.load_test_window(tmp_path,q,"cpu")

class Affine(torch.nn.Module):
    """虚构解析映射，记录每次输入；不会读取target或任何磁盘。"""
    def __init__(self): super().__init__(); self.calls=[]
    def forward(self,z,controls,duration,grid):
        self.calls.append((z.clone(),controls.source_ids,grid.coords.clone(),duration.clone()))
        u=controls.u_values[:,0,0,None,None]
        return z+u*duration.to(z)[:,None,None]*torch.tensor([1.,-.5,.25])

def test_identity_zero_defects_with_nonzero_future_and_seconds(api):
    from po_jepa.baselines import IdentityPredictor
    from po_jepa.losses import LossConfig
    w=fictional_window(api); donor=fictional_window(api,181,action=-.2)
    model=IdentityPredictor(3).eval()
    q=api.EvaluationQuery(180,0,2.,12,"point")
    values,extra=api.point_metrics(model,w,donor,q,e06_config())
    assert values['future_jepa_distance']>0 and values['tcd_frobenius']==values['delta_act']==0
    assert values['tcd_per_second2']==values['tcd_frobenius']*16
    assert extra['identical_controls'] is False
    pair=api.EvaluationQuery(180,0,2.,12,"pair")
    values,_=api.pair_metrics(model,w,pair)
    assert values['scd_frobenius']==values['scd_rms']==values['scd_per_second']==0

def test_R_coordinates_target_restart_and_no_donor_latent(api):
    from po_jepa.losses import LossConfig,jepa_distance
    w=fictional_window(api); donor=fictional_window(api,181,action=-.2)
    q=api.EvaluationQuery(180,0,2.,12,"point"); model=Affine().eval()
    values,extra=api.point_metrics(model,w,donor,q,e06_config())
    R=api.comparison.build_spatial_transfer(w.inputs.grid,target_height=12,target_width=12)
    x=api.comparison.apply_spatial_transfer(w.inputs.raw_z,R); y=api.comparison.apply_spatial_transfer(w.target.raw_z,R)
    assert torch.equal(model.calls[0][0],x) and torch.equal(model.calls[0][2],R.target.coords)
    assert torch.equal(model.calls[2][0],model.calls[1][0]+.1*torch.tensor([1.,-.5,.25]))
    assert all(c[0].shape==(1,144,3) for c in model.calls)
    changed=replace(donor,inputs=replace(donor.inputs,raw_z=donor.inputs.raw_z+100),target=replace(donor.target,raw_z=donor.target.raw_z-100))
    again,_=api.point_metrics(Affine().eval(),w,changed,q,e06_config())
    assert values==again
    expected=jepa_distance(x+.2*torch.tensor([1.,-.5,.25]),y,config=e06_config()).per_sample.item()
    assert values['future_jepa_distance']==pytest.approx(expected)

@pytest.mark.parametrize('bad',["donor","origin","units"])
def test_donor_mismatch_refused(api,bad):
    from po_jepa.losses import LossConfig
    w=fictional_window(api); d=fictional_window(api,181)
    if bad=='donor': d=replace(d,inputs=replace(d.inputs,trajectory_id=182))
    if bad=='origin': d=replace(d,inputs=replace(d.inputs,controls=replace(d.inputs.controls,source_origin_seconds=torch.tensor([1.],dtype=torch.float64))))
    if bad=='units': d=replace(d,inputs=replace(d.inputs,controls=replace(d.inputs.controls,control_spec=replace(d.inputs.controls.control_spec,control_period_seconds=.5))))
    with pytest.raises(ValueError): api.point_metrics(Affine().eval(),w,d,api.EvaluationQuery(180,0,2.,16,"point"),e06_config())

def test_stats_failures_missing_quantiles_and_macro_denominator(api):
    qs=[api.EvaluationQuery(i,j,2.,16,"point") for i in (180,181) for j in (0,5)]
    def row(q,value,status='ok'):
        return dict(query=api.asdict(q),condition='identity',seed=1703,status=status,
                    values={n:value for n in api.metric_units('point')} if status=='ok' else None,error=None if status=='ok' else dict(type='ValueError',message='fictional failure'))
    rows=[row(qs[0],0.),row(qs[1],2.),row(qs[2],8.)]
    stats=api.summarize_records(rows,qs,('identity',))
    group=next(g for g in stats['groups'] if g['metric']=='future_jepa_distance')
    assert group['expected']==4 and group['valid']==3 and group['not_run']==1
    assert group['window_weighted_mean']==pytest.approx(10/3)
    assert group['trajectory_macro_mean']==pytest.approx(4.5)
    assert group['quantiles']==pytest.approx(np.quantile([0,2,8],[.5,.9,.95,.99],method='linear'))
    failed=api.summarize_records([row(q,0.,'failed') for q in qs],qs,('identity',))
    assert all(g['window_weighted_mean'] is None and g['valid']==0 and g['failed']==4 for g in failed['groups'])
    with pytest.raises(ValueError): api.summarize_records([row(qs[0],float('nan'))],qs,('identity',))

def test_paired_results_do_not_hide_missing_and_tradeoffs(api):
    q=api.EvaluationQuery(180,0,2.,16,"point")
    def row(c,f,t):
        values={n:0. for n in api.metric_units('point')};values.update(future_jepa_distance=f,tcd_frobenius=t,tcd_normalized=t)  # v5：取舍标签按归一化 TCD 判断
        return dict(query=api.asdict(q),condition=c,seed=1703,status='ok',values=values)
    rows=[row('po_jepa_full',3.,1.),row('matched_transformer',2.,2.)]
    out=api.paired_comparisons(rows,[q])
    tf=next(r for r in out if r['reference']=='matched_transformer')
    assert tf['paired_valid']==1 and tf['rows'][0]['tradeoff']=='consistency_lower_prediction_worse'
    assert tf['rows'][0]['full_minus_reference']['future_jepa_distance']==1.
    missing=next(r for r in out if r['reference']=='identity')
    assert missing['paired_valid']==0 and missing['unmatched'][0]['reason']=='reference_not_run'

def test_runner_failure_is_retained_and_no_update(api,monkeypatch,tmp_path):
    from po_jepa.baselines import IdentityPredictor
    from po_jepa.losses import LossConfig
    qs=[api.EvaluationQuery(180,j,2.,16,"point") for j in (0,5)]
    monkeypatch.setattr(api,'load_test_window',lambda root,q,device:fictional_window(api,q.trajectory_id,q.start_tick,q.duration_cycles))
    def broken(*args): raise ValueError('fictional NaN stopped')
    monkeypatch.setattr(api,'point_metrics',broken)
    model=IdentityPredictor(3); rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='fictional NaN'):
        api.run_prepared_evaluation(model,'identity',tmp_path/'none',tmp_path/'run',e06_config(),queries=qs)
    report=json.loads((tmp_path/'run/report.json').read_bytes())
    assert report['counts']==dict(expected=2,completed=1,valid=0,failed=1,not_run=1)
    assert (tmp_path/'run/FAILED.json').exists() and not (tmp_path/'run/COMPLETE.json').exists()
    assert model.training and torch.equal(rng,torch.get_rng_state())

def test_runner_success_per_trajectory_and_no_mode_mutation(api,monkeypatch,tmp_path):
    from po_jepa.baselines import IdentityPredictor
    from po_jepa.losses import LossConfig
    qs=[api.EvaluationQuery(180,0,2.,12,k) for k in ('point','pair')]
    monkeypatch.setattr(api,'load_test_window',lambda root,q,device:fictional_window(api,q.trajectory_id,q.start_tick,q.duration_cycles))
    model=IdentityPredictor(3)
    report=api.run_prepared_evaluation(model,'identity',tmp_path/'none',tmp_path/'run',e06_config(),queries=qs)
    assert report['counts']==dict(expected=2,completed=2,valid=2,failed=0,not_run=0)
    assert report['real_evaluation_run'] is False and model.training
    assert (tmp_path/'run/trajectories/0180.json').exists()
    with pytest.raises(FileExistsError): api.run_prepared_evaluation(model,'identity',tmp_path/'none',tmp_path/'run',e06_config(),queries=qs)

@pytest.mark.parametrize('sign',[1,-1,0])
def test_delta_positive_negative_identical_controls_no_redraw(api,sign):
    w=fictional_window(api); donor=fictional_window(api,181,action=.1 if sign==0 else -.2)
    direction=torch.tensor([1.,-.5,.25]); shift=.2 if sign>=0 else -.4
    w=replace(w,target=replace(w.target,raw_z=w.inputs.raw_z+shift*direction))
    values,extra=api.point_metrics(Affine().eval(),w,donor,api.EvaluationQuery(180,0,2.,16,'point'),e06_config())
    assert (values['delta_act']>0 if sign==1 else values['delta_act']<0 if sign==-1 else values['delta_act']==0)
    assert extra['identical_controls']==(sign==0)

def test_nonzero_hand_metrics_separate_frobenius_rms_and_units(api):
    class GridAffine(torch.nn.Module):
        """虚构F_T^N=(1+T)Z+TN，解析缺陷不依赖被测指标函数。"""
        def forward(self,z,u,t,g):
            return (1+t.to(z)[:,None,None])*z+t.to(z)[:,None,None]*z.shape[1]
    w=fictional_window(api);d=fictional_window(api,181)
    p,_=api.point_metrics(GridAffine().eval(),w,d,api.EvaluationQuery(180,0,2.,16,'point'),e06_config())
    expected=torch.linalg.vector_norm((w.inputs.raw_z.double()+256).reshape(-1)).item()
    assert p['tcd_frobenius']==pytest.approx(expected,rel=1e-5,abs=1e-6)
    assert p['tcd_rms']==pytest.approx(expected/(256*3)**.5,rel=1e-5,abs=1e-6)
    assert p['tcd_per_second2']==p['tcd_frobenius']*16
    s,_=api.pair_metrics(GridAffine().eval(),w,api.EvaluationQuery(180,0,2.,12,'pair'))
    assert s['scd_frobenius']==pytest.approx(112*(144*3)**.5,rel=1e-5,abs=1e-6)
    assert s['scd_rms']==pytest.approx(112,rel=1e-5,abs=1e-6)
    assert s['scd_per_second']==s['scd_frobenius']*4

@pytest.mark.parametrize('condition',['matched_transformer','po_jepa_full','po_jepa_no_consistency'])
def test_selected_checkpoint_real_api_only_fictional_file(api,monkeypatch,tmp_path,condition):
    from test_training import e06_case
    from po_jepa.baselines import TokenTransformer,TransformerSpec
    monkeypatch.setitem(__import__('sys').modules,'e06_training_script',api.comparison.training)
    base,batch=e06_case()
    model=TokenTransformer(TransformerSpec(encoder_dim=3)) if condition=='matched_transformer' else base
    config=e06_config(**v4_loss_fields()) if condition=='po_jepa_full' else e06_config(time_weight=0.,space_weight=0.,**v4_loss_fields())
    opt=api.comparison.make_optimizer(model)
    api.comparison.condition_step(condition,model,opt,batch,config,0)
    path=tmp_path/'fictional.pt';state=dict(kind='explicit_prepared_batches',next_batch_index=1,query_generator_state=torch.Generator().manual_seed(2703).get_state())
    sha=api.comparison.save_checkpoint(condition,path,model,opt,config,1,state,batch.provenance)
    other=TokenTransformer(model.spec) if condition=='matched_transformer' else e06_case()[0]
    optimizer=api.comparison.make_optimizer(other);provenance=api.comparison._provenance(batch.provenance)
    selected=dict(step=1,mean=.2,checkpoint='fictional.pt',checkpoint_sha256=sha,checkpoint_provenance=provenance)
    before={k:v.clone() for k,v in other.state_dict().items()}
    with pytest.raises(ValueError): api.load_selected_checkpoint(condition,selected|dict(step=2),model=other,optimizer=optimizer,config=config,expected_provenance=provenance,root=tmp_path)
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in before.items())
    progress=api.load_selected_checkpoint(condition,selected,model=other,optimizer=optimizer,config=config,expected_provenance=provenance,root=tmp_path)
    assert progress.completed_steps==1 and not other.training
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError): api.load_selected_checkpoint(condition,selected|dict(checkpoint_sha256='0'*64),model=other,optimizer=optimizer,config=config,expected_provenance=provenance,root=tmp_path)

@pytest.mark.parametrize('kind',['po','transformer'])
def test_actual_modules_point_pair_no_grad_no_update(api,monkeypatch,kind):
    from test_training import e06_case
    from po_jepa.baselines import TokenTransformer,TransformerSpec
    monkeypatch.setitem(__import__('sys').modules,'e06_training_script',api.comparison.training)
    model=e06_case()[0] if kind=='po' else TokenTransformer(TransformerSpec(encoder_dim=3))
    model.eval(); next(model.parameters()).grad=torch.ones_like(next(model.parameters()))
    identity=api._state_identity(model);w=fictional_window(api,duration=1.);d=fictional_window(api,181,duration=1.,action=-.2)
    p,_=api.point_metrics(model,w,d,api.EvaluationQuery(180,0,1.,8,'point'),e06_config())
    s,_=api.pair_metrics(model,w,api.EvaluationQuery(180,0,1.,8,'pair'))
    assert all(np.isfinite(v) for v in (*p.values(),*s.values()))
    assert api._state_identity(model)==identity

def test_equal_function_with_redundant_segments_and_source_split(api):
    w=fictional_window(api,duration=8.);d=fictional_window(api,181,duration=8.)
    from po_jepa.actions import from_physical_segments
    controls=from_physical_segments(starts_seconds=torch.tensor([[0.,1.]],dtype=torch.float64),
        ends_seconds=torch.tensor([[1.,2.]],dtype=torch.float64),u_values=torch.tensor([[[.1,0.],[.1,0.]]]),
        n_segments=torch.tensor([2]),segment_valid=torch.tensor([[True,True]]),origin_seconds=torch.zeros(1,dtype=torch.float64),
        control_spec=d.inputs.controls.control_spec,source_ids=('trajectory-0181',))
    assert api.identical_controls(w.inputs.controls,controls,w.inputs.duration)

def test_missing_scd_accuracy_not_silently_invented(api):
    q=api.EvaluationQuery(180,0,3.,8,'pair')
    def row(c): return dict(query=api.asdict(q),condition=c,seed=1703,status='ok',values={n:1. for n in api.metric_units('pair')})
    result=api.paired_comparisons([row('po_jepa_full'),row('identity')],[q])
    assert next(r for r in result if r['reference']=='identity')['rows'][0]['tradeoff']=='accuracy_not_measured_on_this_grid_time'

