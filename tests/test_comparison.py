"""F05 条件、参数匹配和保存的CPU代码验收；不读取真实数组/权重。"""

from dataclasses import replace

import importlib

import json

from pathlib import Path

import pytest

import torch

@pytest.fixture
def api(monkeypatch):
    """直接核安装后的同一运行模块。"""
    return importlib.import_module("po_jepa.runtime.comparison")

@pytest.fixture
def case(api, monkeypatch):
    """只提供虚构训练来源，避免重复加载E06脚本造成类身份不一致。"""
    from test_training import e06_case
    monkeypatch.setitem(__import__("sys").modules, "e06_training_script", api.training)
    return e06_case()

def test_actual_parameter_preregistration_and_shared_queries(api):
    """实际CPU构造计数，不用性能选结构；记录完整共享查询身份及独立配置摘要。"""
    before = torch.get_rng_state().clone()
    registration, sequence, state = api.preregister()
    match = registration["parameter_match"]
    assert match["po_parameters"] == 215778 and match["matched"] is True
    assert match["transformer_parameters"] == 215680 and match["selected"]["feedforward"] == 128
    assert match["relative_difference"] <= .01 and len(match["candidates"]) == 3
    assert len(sequence) == 2000 and registration["query_sequence_sha256"] == api.sequence_sha(sequence)
    assert registration["conditions"] == ["matched_transformer", "po_jepa_no_consistency", "po_jepa_full"]
    assert torch.equal(before, torch.get_rng_state()) and state.dtype == torch.uint8

@pytest.mark.parametrize("count,ok", [(10100, True), (10101, False), (9900, True), (9899, False)])
def test_exact_one_percent_boundary(api, count, ok):
    """固定1%边界不可事后放宽。"""
    assert api.parameter_match(10000, count)["matched"] is ok

def test_po_initialization_identical_and_weights_only_difference(api):
    """两个PO条件同架构同初态，仅批准的双权重及条件标签不同。"""
    full, cf = api.build_condition("po_jepa_full", torch.device("cpu"))
    zero, cz = api.build_condition("po_jepa_no_consistency", torch.device("cpu"))
    assert all(torch.equal(v, zero.state_dict()[k]) for k, v in full.state_dict().items())
    loss = api.check.load_protocol()["loss"]   # v4：权重取自 configs/train/v4.yaml 覆盖层
    assert (cf.time_weight, cf.space_weight) == (loss["time_weight"], loss["space_weight"]) and cf.time_weight > 0 and cf.space_weight > 0
    assert cz.time_weight == cz.space_weight == 0

def test_zero_po_keeps_five_phi_and_actual_unweighted_losses(api, case):
    """零权重仍走原E06五Phi，time/space实值保留，不能伪造为0。"""
    from test_training import e06_config
    model, batch = case
    calls = []
    handle = model.p_out.register_forward_hook(lambda *a: calls.append(1))
    result = api.condition_step("po_jepa_no_consistency", model, api.make_optimizer(model), batch,
        e06_config(time_weight=0., space_weight=0.), 0)
    handle.remove()
    assert len(calls) == 5 and result.completed_steps == 1
    assert result.loss_values["weighted_time"] == result.loss_values["weighted_space"] == 0
    assert result.loss_values["time"] > 0 and result.loss_values["space"] > 0

def test_transformer_only_one_jepa_update_and_optimizer_coverage(api, case):
    """TF仅一次JEPA前向/反传；未算的一致性日志是None，不是误差为0。"""
    from po_jepa.baselines import TokenTransformer, TransformerSpec
    from test_training import e06_config
    _, batch = case
    model = TokenTransformer(TransformerSpec(encoder_dim=3))
    optimizer = api.make_optimizer(model)
    calls = []
    handle = model.register_forward_hook(lambda *a: calls.append(1))
    result = api.condition_step("matched_transformer", model, optimizer, batch, e06_config(time_weight=0., space_weight=0.), 0)
    handle.remove()
    assert len(calls) == 1 and result.completed_steps == 1
    assert result.loss_values["time"] is result.loss_values["space"] is None
    assert result.loss_values["jepa"] == result.loss_values["total"]
    assert [id(p) for p in optimizer.param_groups[0]["params"]] == [id(p) for p in model.parameters()]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        api.condition_step("identity", model, optimizer, batch, e06_config(), 1)

def test_identity_defects_zero_but_future_error_nonzero(api, case):
    """复用E07范数定义：恒等缺陷为0，也可能完全没预测未来。"""
    from po_jepa.baselines import IdentityPredictor
    from test_training import e06_config
    _, batch = case
    model = IdentityPredictor(3).eval()
    metrics = api.evaluate_batch(model, batch.raw_z, batch.controls, batch.duration, batch.transfer,
        batch.raw_target, e06_config())
    assert metrics["tcd_frobenius"].values.eq(0).all() and metrics["scd_frobenius"].values.eq(0).all()
    assert metrics["future_jepa_distance"].values.gt(0).all()
    with pytest.raises(ValueError): api.make_optimizer(model)

def test_transformer_checkpoint_roundtrip_and_bad_sha_no_mutation(api, case, tmp_path):
    """只读写本测试虚构checkpoint；绑定模型/优化器、来源及实际游标，非F07验收。"""
    from po_jepa.baselines import TokenTransformer, TransformerSpec
    from test_training import e06_config, v4_loss_fields
    _, batch = case
    model = TokenTransformer(TransformerSpec(encoder_dim=3))
    optimizer = api.make_optimizer(model)
    config = e06_config(time_weight=0., space_weight=0., **v4_loss_fields())
    api.condition_step("matched_transformer", model, optimizer, batch, config, 0)
    state = dict(kind="explicit_prepared_batches", next_batch_index=1,
        query_generator_state=torch.Generator().manual_seed(2703).get_state())
    path = tmp_path/"fictional.pt"
    digest = api.save_checkpoint("matched_transformer", path, model, optimizer, config, 1, state, batch.provenance)
    target = TokenTransformer(model.spec); target_optimizer = api.make_optimizer(target)
    progress = api.load_transformer_checkpoint(path, digest, target, target_optimizer, batch.provenance)
    assert progress.completed_steps == 1 and progress.sampling_state["next_batch_index"] == 1
    assert all(torch.equal(v, target.state_dict()[k]) for k,v in model.state_dict().items())
    snapshot = {k:v.clone() for k,v in target.state_dict().items()}
    with pytest.raises(ValueError): api.load_transformer_checkpoint(path, "0"*64, target, target_optimizer, batch.provenance)
    assert all(torch.equal(v, target.state_dict()[k]) for k,v in snapshot.items())
    with pytest.raises(FileExistsError):
        api.save_checkpoint("matched_transformer", path, model, optimizer, config, 1, state, batch.provenance)

@pytest.fixture
def loop_case(api, case, monkeypatch, tmp_path):
    """三条件共享的虚构5步控制流；计数/保存为替身，不宣称真实2000步或容量。"""
    model = torch.nn.Module(); model.p_out = torch.nn.Linear(1, 1)
    optimizer = object(); batch = case[1]
    monkeypatch.setattr(api.data, "load_training_batch", lambda *a, **k: batch)
    monkeypatch.setattr(api.data, "write_collections", lambda *a, **k: {})
    sequence = [[api.data.WindowQuery(0, n, 1.)] for n in range(5)]
    state = torch.Generator().manual_seed(2703).get_state()
    registration = dict(parameter_match=api.parameter_match(2, 2), query_sequence_sha256=api.sequence_sha(sequence),
        comparison_config_sha256=api.data.cache.file_sha(api.ROOT/"configs/train/comparison.yaml"),
        consumer_sha256=api.consumer_sources())
    calls = dict(updates=[], validations=[], saved=[])
    def step(condition, model, optimizer, batch, config, completed):
        calls["updates"].append((condition, completed))
        for _ in range(1 if condition == "matched_transformer" else 5): model.p_out(torch.ones(1, 1))
        return api.training.TrainingStepResult(completed+1,
            dict(jepa=1.,time=None if condition == "matched_transformer" else 2.,space=None if condition == "matched_transformer" else 3.,
                weighted_time=0.,weighted_space=0.,total=1.), .1)
    monkeypatch.setattr(api, "condition_step", step)
    metrics = iter([1., .5, .5, .8])
    def validate(*args):
        calls["validations"].append(len(calls["updates"]))
        return dict(mean=next(metrics), windows=1, metric="future_jepa_distance", means={"future_jepa_distance":1.,"tcd_frobenius":99.,"scd_frobenius":88.})
    monkeypatch.setattr(api, "validate_condition", validate)
    def save(condition, path, model, optimizer, config, completed, sampling, provenance):
        calls["saved"].append((completed,sampling["next_batch_index"]))
        path.write_bytes(b"explicit fictional placeholder")
        return api.data.cache.file_sha(path)
    monkeypatch.setattr(api, "save_checkpoint", save)
    from test_training import e06_config
    kwargs = dict(sequence=sequence, generator_state=state, registration=registration,
        features=tmp_path/"fictional", output=tmp_path/"run", device=torch.device("cpu"),
        loss_config=e06_config(), validation_interval=2)
    return model,optimizer,kwargs,calls

@pytest.mark.parametrize("condition", ["matched_transformer", "po_jepa_no_consistency", "po_jepa_full"])
def test_condition_loop_shared_queries_selection_and_counts(api, loop_case, condition):
    """0/2/4/5验证，只按future选点，并列取2；更新与真实游标不重置。"""
    model, optimizer, kwargs, calls = loop_case
    result = api.run_prepared_condition(condition, model, optimizer, **kwargs)
    assert calls["validations"] == [0,2,4,5] and calls["saved"] == [(0,0),(2,2),(4,4),(5,5)]
    assert result["completed_updates"] == 5 and result["best"]["step"] == 2
    assert result["query_sequence_sha256"] == kwargs["registration"]["query_sequence_sha256"]
    assert result["evidence_role"] == "synthetic_development_only"
    row=json.loads((kwargs["output"]/"updates/0001.json").read_bytes())
    assert row["phi_calls"] == (1 if condition == "matched_transformer" else 5)
    assert row["gpu_memory"] is None and row["queries"][0]["trajectory_id"] == 0

@pytest.mark.parametrize("failure", ["update", "nan_validation", "save", "sequence", "capacity"])
def test_loop_failure_blocks_success_and_preserves_evidence(api, loop_case, monkeypatch, failure):
    """关键失败立即停止，不跳过、不改容差；输入身份失败不启动循环。"""
    model, optimizer, kwargs, calls = loop_case
    if failure == "update":
        def fail(*a, **k): raise ValueError("required predictor gradient missing")
        monkeypatch.setattr(api,"condition_step",fail)
    elif failure == "nan_validation":
        monkeypatch.setattr(api,"validate_condition",lambda *a:dict(mean=float("nan")))
    elif failure == "save":
        def fail(*a, **k): raise OSError("fictional checkpoint failure")
        monkeypatch.setattr(api,"save_checkpoint",fail)
    elif failure == "sequence": kwargs["registration"]["query_sequence_sha256"]="0"*64
    else: kwargs["registration"]["parameter_match"]=api.parameter_match(10000,10101)
    with pytest.raises((ValueError,OSError)):
        api.run_prepared_condition("po_jepa_full",model,optimizer,**kwargs)
    assert not (kwargs["output"]/"COMPLETE.json").exists()
    if failure in ("sequence","capacity"): assert not kwargs["output"].exists()
    else: assert (kwargs["output"]/"FAILED.json").exists()

@pytest.mark.parametrize("condition", ["po_jepa_full", "po_jepa_no_consistency"])
def test_po_checkpoint_retains_correct_condition_and_sources(api, case, tmp_path, condition):
    """两种PO都沿原E06存取，不假冒新schema；配置SHA置于独立字段。"""
    from test_training import e06_case, e06_config
    model,batch=case
    cfg=e06_config() if condition == "po_jepa_full" else e06_config(time_weight=0.,space_weight=0.)
    optimizer=api.make_optimizer(model)
    api.condition_step(condition,model,optimizer,batch,cfg,0)
    state=dict(kind="explicit_prepared_batches",next_batch_index=1,query_generator_state=torch.Generator().manual_seed(2703).get_state())
    path=tmp_path/"fictional-po.pt"
    digest=api.save_checkpoint(condition,path,model,optimizer,cfg,1,state,batch.provenance)
    other,_=e06_case(); other_optimizer=api.make_optimizer(other)
    progress=api.training.load_training_checkpoint(path,expected_sha256=digest,model=other,optimizer=other_optimizer,
        loss_config=cfg,condition="full" if condition == "po_jepa_full" else "no_consistency",
        expected_provenance=api._provenance(batch.provenance))
    assert progress.completed_steps == 1
    assert all(torch.equal(value,other.state_dict()[key]) for key,value in model.state_dict().items())
    saved=torch.load(path,map_location="cpu",weights_only=True)
    assert saved["provenance"]["f05_comparison_config_sha256"] == api.data.cache.file_sha(api.ROOT/"configs/train/comparison.yaml")
    assert all(name.endswith(".py") for name in saved["provenance"]["source_files_sha256"])

@pytest.mark.parametrize("failure", [False, True])
def test_validation_400_protocol_restores_state_without_training(api, monkeypatch, tmp_path, failure):
    """400仅指虚构窗口装配计数；指标为替身，不能冒称真实验证或计算成本。"""
    from test_real_training import unit_windows
    from types import SimpleNamespace
    base=unit_windows(api.data); calls=[]
    def windows(root,rows,*,split,device):
        assert split == "validation"; calls.extend(rows); result=[]
        for window,query in zip(base,rows):
            origin=torch.tensor([query.start_tick*.25],dtype=torch.float64)
            inputs=replace(window.inputs,trajectory_id=query.trajectory_id,split=split,observed_time_seconds=origin,
                controls=replace(window.inputs.controls,source_origin_seconds=origin))
            result.append(replace(window,inputs=inputs,target=replace(window.target,target_time_seconds=origin+inputs.duration*.25)))
        return result
    def metrics(model,raw,*args):
        assert not torch.is_grad_enabled() and not any(m.training for m in model.modules())
        if failure: raise ValueError("fictional metric failure")
        return {name:SimpleNamespace(values=torch.full((raw.shape[0],),value)) for name,value in
            [("future_jepa_distance",1.),("tcd_frobenius",2.),("scd_frobenius",3.)]}
    monkeypatch.setattr(api.data,"load_windows",windows)
    monkeypatch.setattr(api,"evaluate_batch",metrics)
    model=torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Dropout(.1)); model.train(); model[1].eval()
    for p in model.parameters(): p.grad=torch.ones_like(p)
    before=api.check._parameter_identity(model); modes=[m.training for m in model.modules()]
    from test_training import e06_config
    if failure:
        with pytest.raises(ValueError): api.validate_condition(model,tmp_path,tmp_path/"val",torch.device("cpu"),e06_config())
    else:
        report=api.validate_condition(model,tmp_path,tmp_path/"val",torch.device("cpu"),e06_config())
        assert report["windows"] == len(calls) == 400 and report["mean"] == 1.
        assert report["means"] == dict(future_jepa_distance=1.,tcd_frobenius=2.,scd_frobenius=3.)
        assert {q.trajectory_id for q in calls} == set(range(160,180))
    assert api.check._parameter_identity(model)==before and [m.training for m in model.modules()]==modes

def test_target_changes_distance_not_predictions(api, case):
    """评价标签只进入d，不能反向渗入任何一次预测或中间重启。"""
    from po_jepa.baselines import TokenTransformer,TransformerSpec
    from test_training import e06_config
    _,batch=case; model=TokenTransformer(TransformerSpec(encoder_dim=3)).eval(); outputs=[]
    handle=model.register_forward_hook(lambda m,a,out:outputs.append(out.detach().clone()))
    first=api.evaluate_batch(model,batch.raw_z,batch.controls,batch.duration,batch.transfer,batch.raw_target,e06_config())
    other_target=batch.raw_target.flip(-1).clone()
    second=api.evaluate_batch(model,batch.raw_z,batch.controls,batch.duration,batch.transfer,other_target,e06_config())
    handle.remove()
    assert len(outputs)==10 and all(torch.equal(a,b) for a,b in zip(outputs[:5],outputs[5:]))
    assert not torch.equal(first["future_jepa_distance"].values,second["future_jepa_distance"].values)

def test_transformer_rejects_extra_optimizer_group_before_mutation(api, case):
    """优化器不能混入额外参数，错误在清梯度/更新之前阻断。"""
    from po_jepa.baselines import TokenTransformer,TransformerSpec
    from test_training import e06_config
    _,batch=case; model=TokenTransformer(TransformerSpec(encoder_dim=3)); optimizer=api.make_optimizer(model)
    optimizer.add_param_group({"params":[torch.nn.Parameter(torch.ones(1))]})
    before={k:v.clone() for k,v in model.state_dict().items()}
    with pytest.raises(ValueError):
        api.condition_step("matched_transformer",model,optimizer,batch,e06_config(time_weight=0.,space_weight=0.),0)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())

@pytest.mark.parametrize("mismatch", [None,"budget","queries","registration","role"])
def test_comparison_records_require_same_actual_inputs(api,mismatch):
    """汇总不能把不同样本/预算/源码版本/开发角色拼成一次比较。"""
    import hashlib
    registration={"parameter_match":api.parameter_match(10000,10000),"query_sequence_sha256":"same fictional sequence"}
    digest=hashlib.sha256(api.data.cache._json_bytes(registration)).hexdigest()
    reports={name:dict(condition=name,completed_updates=5,evidence_role="synthetic_development_only",
        preregistration_sha256=digest,query_sequence_sha256=registration["query_sequence_sha256"],best={"step":0,"mean":1.}) for name in api.CONDITIONS}
    identity=dict(preregistration_sha256=digest,completed_updates=0)
    if mismatch:
        selected=reports[api.CONDITIONS[0]]
        key={"budget":"completed_updates","queries":"query_sequence_sha256","registration":"preregistration_sha256","role":"evidence_role"}[mismatch]
        selected[key]=6 if mismatch == "budget" else "different"
        with pytest.raises(ValueError):api.comparison_table(reports,identity,registration)
    else:
        table=api.comparison_table(reports,identity,registration)
        assert len(table["rows"])==3 and table["heldout_test_evaluation_run"] is False

def test_identity_reference_allocates_no_training(api,monkeypatch,tmp_path):
    """未训练参照也绑定同一登记，但不能意外创建优化器或运行训练更新。"""
    def forbidden(*a,**k):raise AssertionError("identity must not train")
    monkeypatch.setattr(api,"make_optimizer",forbidden)
    monkeypatch.setattr(api,"condition_step",forbidden)
    monkeypatch.setattr(api,"validate_condition",lambda *a:dict(windows=1,mean=1.,development_probe_only=True))
    result=api.identity_reference(tmp_path,tmp_path/"identity",torch.device("cpu"),None,{"consumer_sha256":api.consumer_sources()})
    assert result["completed_updates"]==result["trainable_parameters"]==0


@pytest.mark.parametrize("condition", ["po_jepa_no_consistency", "po_jepa_full"])
def test_condition_loop_residual_readout_counts_two_readouts_per_phi(api, loop_case, monkeypatch, condition):
    """v4 残差读出：每次Phi调用P_out两次，更新记录仍按5次Phi登记。"""
    model, optimizer, kwargs, calls = loop_case
    model.readout = "residual"
    def step(condition, model, optimizer, batch, config, completed):
        calls["updates"].append((condition, completed))
        for _ in range(10): model.p_out(torch.ones(1, 1))
        return api.training.TrainingStepResult(completed+1,
            dict(jepa=1.,time=2.,space=3.,weighted_time=0.,weighted_space=0.,total=1.), .1)
    monkeypatch.setattr(api, "condition_step", step)
    api.run_prepared_condition(condition, model, optimizer, **kwargs)
    row=json.loads((kwargs["output"]/"updates/0001.json").read_bytes())
    assert row["phi_calls"] == 5 and len(row["raw_predictions"]) == 10
