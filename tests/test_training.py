"""E06：用虚构数据核验式17、五Phi更新和自有基本恢复，不证明科研效果。"""

from dataclasses import fields, replace

from fractions import Fraction

import importlib

import math

import numpy as np

import pytest

import torch

from po_jepa.losses import LossConfig, ReducedLoss

def e06_losses():
    """RED在运行时明确暴露缺失接口，避免用导入错误充当算法测试。"""
    module = importlib.import_module("po_jepa.losses")
    assert hasattr(module, "TotalLoss") and hasattr(module, "weighted_total_loss"), "E06式17接口尚未实现"
    return module

def e06_config(**changes):
    """固定A03默认值；仅显式覆盖本测试需要核验的合法或损坏字段。"""
    values = dict(jepa_exponent=1., layer_norm_epsilon=1e-5, time_epsilon=1e-8,
                  space_epsilon=1e-8, time_weight=1e-3, space_weight=1e-3)
    values.update(changes)
    return LossConfig(**values)

def v4_loss_fields():
    """v4 覆盖层中的一致性度量、预热与梯度裁剪字段；与协议比对的 checkpoint 测试需带上它们。"""
    from po_jepa.runtime import checks
    loss = checks.load_protocol()["loss"]
    return dict(consistency_metric=loss["consistency_metric"], consistency_warmup_updates=loss["consistency_warmup_updates"],
                grad_clip_norm=loss["grad_clip_norm"])

def e06_reduced(*, mean=2., dtype=torch.float64, count=2, device="cpu"):
    """独立归约夹具故意让mean不同于per_sample.mean，包装器必须保留调用者值。"""
    return ReducedLoss(torch.arange(1, count+1, dtype=dtype, device=device),
                       torch.tensor(mean, dtype=dtype, device=device, requires_grad=True))

@pytest.mark.parametrize("weights", [(0., 0.), (.25, .5), (Fraction(1, 4), np.float64(.5))])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_e06_total_fields_independent_means_float64_and_analytic_gradient(weights, device):
    """式17手算2+.25*3+.5*5；三个不同样本数不重归约，梯度为1/lambda_t/lambda_s。"""
    module = e06_losses()
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; CPU mandatory")
    jepa = e06_reduced(mean=2., dtype=torch.float32, count=3, device=device)
    time = e06_reduced(mean=3., count=2, device=device)
    space = e06_reduced(mean=5., count=1, device=device)
    config = e06_config(time_weight=weights[0], space_weight=weights[1])
    before = torch.get_rng_state().clone()
    total = module.weighted_total_loss(jepa, time, space, config=config)
    assert tuple(f.name for f in fields(total)) == ("jepa", "time", "space", "weighted_time", "weighted_space", "total")
    assert total.jepa is jepa and total.time is time and total.space is space
    for value in (total.weighted_time, total.weighted_space, total.total):
        assert value.dtype == torch.float64 and value.shape == () and value.device == jepa.mean.device
    assert total.total.item() == 2.+float(weights[0])*3.+float(weights[1])*5.
    total.total.backward()
    assert jepa.mean.grad.item() == 1.
    assert time.mean.grad.item() == float(weights[0])
    assert space.mean.grad.item() == float(weights[1])
    assert torch.equal(before, torch.get_rng_state())
    assert config.time_weight is weights[0] and config.space_weight is weights[1]
    holder = module.TotalLoss(jepa, time, space, "unchanged", None, object())
    assert holder.weighted_time == "unchanged" and holder.weighted_space is None

@pytest.mark.parametrize("bad", ["object", "config", "field_object", "rank", "empty", "mean_shape",
    "jepa_dtype", "time_dtype", "mean_dtype", "negative_sample", "negative_mean", "nan", "inf_zero_weight",
    "sparse", "meta", "inference", "weight_underflow", "product_underflow", "product_overflow", "total_overflow"])
def test_e06_total_rejects_contracts_and_weight_precision(bad):
    """即便双零权重也核查分量；正Real/乘积下溢和有限输入总和溢出须明确失败。"""
    module = e06_losses()
    jepa, time, space = e06_reduced(dtype=torch.float32), e06_reduced(), e06_reduced()
    config = e06_config()
    exception = TypeError if bad in ("object", "config", "field_object", "jepa_dtype", "time_dtype", "mean_dtype") else ValueError
    if bad == "object": jepa = object()
    elif bad == "config": config = object()
    elif bad == "field_object": time = replace(time, per_sample=[1.])
    elif bad == "rank": time = replace(time, per_sample=torch.ones(1, 1, dtype=torch.float64))
    elif bad == "empty": time = replace(time, per_sample=torch.empty(0, dtype=torch.float64))
    elif bad == "mean_shape": time = replace(time, mean=time.mean.reshape(1))
    elif bad == "jepa_dtype": jepa = e06_reduced()
    elif bad == "time_dtype": time = e06_reduced(dtype=torch.float32)
    elif bad == "mean_dtype": time = replace(time, mean=time.mean.float())
    elif bad == "negative_sample": time = replace(time, per_sample=-time.per_sample)
    elif bad == "negative_mean": time = e06_reduced(mean=-1.)
    elif bad == "nan": time = e06_reduced(mean=float("nan"))
    elif bad == "inf_zero_weight":
        time, config = e06_reduced(mean=float("inf")), e06_config(time_weight=0., space_weight=0.)
    elif bad == "sparse": time = replace(time, per_sample=time.per_sample.to_sparse())
    elif bad == "meta": time = replace(time, mean=time.mean.to("meta"))
    elif bad == "inference":
        with torch.inference_mode(): time = replace(time, mean=torch.tensor(1., dtype=torch.float64))
    elif bad == "weight_underflow": config = e06_config(time_weight=Fraction(1, 10**400))
    elif bad == "product_underflow": time, config = e06_reduced(mean=1e-100), e06_config(time_weight=1e-300)
    elif bad == "product_overflow": time, config = e06_reduced(mean=1e200), e06_config(time_weight=1e200)
    elif bad == "total_overflow":
        time, space, config = e06_reduced(mean=1e308), e06_reduced(mean=1e308), e06_config(time_weight=1., space_weight=1.)
    with pytest.raises(exception): module.weighted_total_loss(jepa, time, space, config=config)

def test_e06_total_actual_device_mismatch_and_legal_zero_component():
    """组件本来为0可与正小权重相乘；不同实际设备绝不自动搬运。"""
    module = e06_losses()
    jepa, time, space = e06_reduced(dtype=torch.float32), e06_reduced(mean=0.), e06_reduced(mean=0.)
    result = module.weighted_total_loss(jepa, time, space, config=e06_config(time_weight=1e-300))
    assert result.weighted_time.item() == result.weighted_space.item() == 0.
    if torch.cuda.is_available():
        with pytest.raises(ValueError):
            module.weighted_total_loss(jepa, e06_reduced(device="cuda"), space, config=e06_config())

def e06_script():
    """与生产安装包使用同一模块身份。"""
    return importlib.import_module("po_jepa.runtime.training")

def e06_provenance():
    """虚构开发来源；实际本地源码SHA与不存在的encoder等None分开，绝不伪造。"""
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    names = [f"src/po_jepa/{name}.py" for name in ("actions", "dynamics", "geometry", "losses", "model", "spectral")]
    names.append("src/po_jepa/runtime/training.py")
    return dict(evidence_role="synthetic_development_only", split="development_smoke",
        paper_sha256="d851778ecf4a1f160da9c681a831b3bc9333dae2b60eaeefa6090c0af44aaff1",
        upstream_commit="204698b45b3712590f06245fbfba32d3be539812",
        source_files_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
        encoder_checkpoint_sha256=None, preprocess_sha256=None, feature_manifest_sha256=None, protocol_sha256=None)

def e06_case(*, device="cpu", encoder_dim=3):
    """复用已验E05虚构模型夹具，显式生成冻结raw与已知配对端点，不改旧fixture。"""
    from test_losses import e05_real_case

    module = e06_script()
    model, raw, controls, duration, transfer = e05_real_case(device=device, encoder_dim=encoder_dim)
    model.train()
    raw = raw.detach()
    target = torch.roll(raw, 1, -1)*.8+.03
    controls = replace(controls, u_values=controls.u_values.detach().requires_grad_())
    tau = torch.tensor([.25, .125], dtype=torch.float64)
    sigma = torch.tensor([.375, .25], dtype=torch.float64)
    observed = controls.source_origin_seconds.clone()
    batch = module.TrainingBatch(raw, target, controls, duration, tau, sigma, transfer,
        observed, observed+duration*controls.control_spec.control_period_seconds,
        ("fictional-e06-0", "fictional-e06-1"), e06_provenance())
    return model, batch

def test_e06_step_types_optimizer_exact_coverage_and_frozen_rejection():
    """全部27个既有预测参数有序且唯一，不能意外冻结后再过滤来假装完整。"""
    module = e06_script()
    assert tuple(f.name for f in fields(module.TrainingBatch)) == ("raw_z", "raw_target", "controls", "duration", "tau", "sigma", "transfer", "observed_time_seconds", "target_time_seconds", "sample_ids", "provenance")
    assert tuple(f.name for f in fields(module.TrainingStepResult)) == ("completed_steps", "loss_values", "grad_max_abs", "grad_norm")
    assert tuple(f.name for f in fields(module.CheckpointProgress)) == ("completed_steps", "sampling_state")
    model, _ = e06_case()
    optimizer = module.make_optimizer(model)
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.param_groups[0]["params"] == list(model.parameters())
    assert len({id(p) for p in optimizer.param_groups[0]["params"]}) == 27
    for name, value in dict(lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0).items():
        assert optimizer.defaults[name] == value
    assert all(id(b) not in {id(p) for p in model.parameters()} for b in model.buffers())
    model.dynamics.gamma_head.weight.requires_grad_(False)
    with pytest.raises(ValueError): module.make_optimizer(model)

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("default_cpu", [False, True])
def test_e06_step_real_1408_five_phi_one_backward_manual_vjp_and_adamw(monkeypatch, device, default_cpu):
    """真实五Phi/27参数VJP；CPU默认device上下文中也只计实际C++反向一次。"""
    import copy
    from contextlib import nullcontext
    from unittest.mock import patch
    from torch.autograd import Variable
    from po_jepa.actions import split_action_segments, concatenate_action_segments
    from po_jepa.geometry import apply_spatial_transfer
    from test_model import d12_record

    module = e06_script()
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    model, batch = e06_case(device=device, encoder_dim=1408)
    reference = copy.deepcopy(model)
    optimizer = module.make_optimizer(model)
    queries = d12_record(monkeypatch, model, "prepare_query")
    spectra = d12_record(monkeypatch, model.spectrum, "lowest_eigenpairs")
    calls, steps, backwards = [], [], []
    model.register_forward_hook(lambda owner, args, result: calls.append((args, result)))
    original_step, original_backward = optimizer.step, torch.Tensor.backward
    monkeypatch.setattr(optimizer, "step", lambda *a, **kw: (steps.append(True), original_step(*a, **kw))[1])
    engine = Variable._execution_engine
    class EngineObserver:
        """Tensor.backward会被device mode重新分派；这里只计原生engine的实际调用。"""
        def run_backward(self, *args, **kwargs):
            backwards.append(True)
            return engine.run_backward(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(engine, name)
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    ids = [id(p) for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    context = torch.device("cpu") if default_cpu else nullcontext()
    mode_depth = torch._C._len_torch_function_stack()
    with patch.object(Variable, "_execution_engine", EngineObserver()), context:
        result = module.train_step(model, optimizer, batch, loss_config=e06_config(), completed_steps=4)
    assert torch._C._len_torch_function_stack() == mode_depth
    assert len(calls) == len(queries) == len(spectra) == 5 and len(steps) == len(backwards) == 1
    assert [q[2].initial_coefficients.shape[1] for q in queries] == [20,20,20,20,6]
    assert calls[2][0][0] is calls[1][1] and calls[2][0][0].requires_grad
    assert all(not q[2].spectrum.basis.requires_grad for q in queries)
    assert all(args[0] is not batch.raw_target for args, _ in calls)
    assert torch.equal(rng, torch.get_rng_state())
    assert result.completed_steps == 5 and type(result.grad_max_abs) is float and result.grad_max_abs > 0
    assert set(result.loss_values) == {"jepa", "time", "space", "weighted_time", "weighted_space", "total"}
    assert all(type(v) is float and math.isfinite(v) for v in result.loss_values.values())
    assert batch.raw_z.grad is None and batch.raw_target.grad is None
    assert ids == [id(p) for p in model.parameters()]
    assert all(torch.equal(b, buffers[n]) for n, b in model.named_buffers())
    pair = split_action_segments(batch.controls, tau=batch.tau, sigma=batch.sigma)
    direct = reference(batch.raw_z, concatenate_action_segments(pair.first, pair.second), batch.tau+batch.sigma, batch.transfer.source)
    mid = reference(batch.raw_z, pair.first, batch.tau, batch.transfer.source)
    composed = reference(mid, pair.second, batch.sigma, batch.transfer.source)
    fine = reference(batch.raw_z, batch.controls, batch.duration, batch.transfer.source)
    coarse = reference(apply_spatial_transfer(batch.raw_z, batch.transfer), batch.controls, batch.duration, batch.transfer.target)
    j = torch.abs(torch.nn.functional.layer_norm(direct, (1408,), eps=1e-5)-torch.nn.functional.layer_norm(batch.raw_target, (1408,), eps=1e-5)).mean()
    t = ((direct-composed).double().square().sum((1, 2))/((batch.tau.to(device)*batch.sigma.to(device)).square()+1e-8)).mean()
    s = ((apply_spatial_transfer(fine, batch.transfer)-coarse).double().square().sum((1, 2))/(batch.duration.to(device).square()+1e-8)).mean()
    manual = j.double()+1e-3*t+1e-3*s
    original_backward(manual)
    assert abs(result.loss_values["total"]-manual.item()) <= 1e-10+1e-8*abs(manual.item())
    errors, updates = {}, {}
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-5, rtol=1e-4)
        errors[name] = (actual.grad-expected.grad).abs().max().item()
        update = initial[name].double()-1e-3*expected.grad.double()/(expected.grad.double().abs()+1e-8)
        torch.testing.assert_close(actual, update.float(), atol=1e-5, rtol=1e-4)
        updates[name] = (actual-initial[name]).abs().max().item()
        assert torch.isfinite(optimizer.state[actual]["exp_avg_sq"]).all()
        assert optimizer.state[actual]["step"].item() == 1
    assert updates["p_in.weight"] > 0 and updates["p_out.weight"] > 0
    assert len(errors) == 27
    print("E06 real step", dict(device=str(batch.raw_z.device), losses=result.loss_values, grad_max_abs=result.grad_max_abs,
          Phi_calls=5, native_backward_calls=len(backwards), default_cpu_context=default_cpu, parameter_vjp_count=len(errors),
          VJP_max_abs=max(errors.values()), representative_updates={n: updates[n] for n in ("p_in.weight", "p_out.weight", "dynamics.b_head.weight")}))

@pytest.mark.parametrize("bad", ["condition", "full_zero", "full_one_zero", "no_consistency_nonzero", "batch_type", "raw_graph", "target_graph", "raw_dtype", "target_shape", "time_dtype", "duration_sum", "duration_zero", "observed", "target_time", "target_not_later", "sample_ids", "provenance_missing", "provenance_fake_encoder", "provenance_source_hash", "control_spec", "optimizer_order", "optimizer_missing", "optimizer_config", "frozen_parameter", "completed_steps", "eval", "no_grad", "inference"])
def test_e06_step_early_rejection_preserves_existing_grad_and_parameters(bad):
    """数据角色/配对/条件/覆盖的可预检错误在zero_grad与Phi前拒绝。"""
    from contextlib import nullcontext
    module = e06_script()
    model, batch = e06_case()
    optimizer = module.make_optimizer(model)
    config, condition, completed, context = e06_config(), "full", 0, nullcontext()
    if bad == "condition": condition = "time_only"
    elif bad == "full_zero": config = e06_config(time_weight=0., space_weight=0.)
    elif bad == "full_one_zero": config = e06_config(time_weight=0.)
    elif bad == "no_consistency_nonzero": condition = "no_consistency"
    elif bad == "batch_type": batch = object()
    elif bad == "raw_graph": batch = replace(batch, raw_z=batch.raw_z.clone().requires_grad_())
    elif bad == "target_graph": batch = replace(batch, raw_target=batch.raw_target.clone().requires_grad_())
    elif bad == "raw_dtype": batch = replace(batch, raw_z=batch.raw_z.double())
    elif bad == "target_shape": batch = replace(batch, raw_target=batch.raw_target[:, :1])
    elif bad == "time_dtype": batch = replace(batch, tau=batch.tau.float())
    elif bad == "duration_sum": batch = replace(batch, duration=batch.duration+1e-12)
    elif bad == "duration_zero": batch = replace(batch, duration=torch.zeros_like(batch.duration))
    elif bad == "observed": batch = replace(batch, observed_time_seconds=batch.observed_time_seconds+1e-12)
    elif bad == "target_time": batch = replace(batch, target_time_seconds=batch.target_time_seconds+1e-12)
    elif bad == "target_not_later": batch = replace(batch, target_time_seconds=batch.observed_time_seconds.clone())
    elif bad == "sample_ids": batch = replace(batch, sample_ids=("", "other"))
    elif bad == "provenance_missing": batch.provenance.pop("protocol_sha256")
    elif bad == "provenance_fake_encoder": batch.provenance["encoder_checkpoint_sha256"] = "a"*64
    elif bad == "provenance_source_hash": batch.provenance["source_files_sha256"]["src/po_jepa/model.py"] = "0"*64
    elif bad == "control_spec": batch = replace(batch, controls=replace(batch.controls, control_spec=object()))
    elif bad == "optimizer_order": optimizer.param_groups[0]["params"].reverse()
    elif bad == "optimizer_missing": optimizer.param_groups[0]["params"].pop()
    elif bad == "optimizer_config": optimizer.param_groups[0]["lr"] = .1
    elif bad == "frozen_parameter": model.p_in.weight.requires_grad_(False)
    elif bad == "completed_steps": completed = True
    elif bad == "eval": model.eval()
    elif bad == "no_grad": context = torch.no_grad()
    elif bad == "inference": context = torch.inference_mode()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for p in model.parameters(): p.grad = torch.full_like(p, 7.)
    with context, pytest.raises((ValueError, TypeError)):
        module.train_step(model, optimizer, batch, loss_config=config, completed_steps=completed, condition=condition)
    assert all(torch.equal(p, before[n]) and torch.all(p.grad == 7.) for n, p in model.named_parameters())
    assert not optimizer.state

@pytest.mark.parametrize("failure", ["loss", "gradient", "none_gradient", "adamw_state"])
def test_e06_step_finite_boundaries_never_silently_count_success(monkeypatch, failure):
    """有限grad仍可能令AdamW二阶矩溢出；更新后报错保留证据，不回滚或计步。"""
    module = e06_script()
    model, batch = e06_case()
    optimizer = module.make_optimizer(model)
    steps, original = [], optimizer.step
    monkeypatch.setattr(optimizer, "step", lambda *a, **kw: (steps.append(True), original(*a, **kw))[1])
    if failure == "loss":
        real = module.weighted_total_loss
        monkeypatch.setattr(module, "weighted_total_loss", lambda *a, **kw: replace(real(*a, **kw), total=torch.tensor(float("nan"), dtype=torch.float64)))
    elif failure in ("gradient", "adamw_state"):
        model.p_in.weight.register_hook(lambda g: torch.full_like(g, float("inf") if failure == "gradient" else 1e30))
    else:
        real_backward = torch.Tensor.backward
        def missing_gradient(tensor, *args, **kwargs):
            """仅模拟断开的必需预测支，核验缺失梯度不能被跳过。"""
            real_backward(tensor, *args, **kwargs)
            model.p_in.weight.grad = None
        monkeypatch.setattr(torch.Tensor, "backward", missing_gradient)
    with pytest.raises(ValueError): module.train_step(model, optimizer, batch, loss_config=e06_config(), completed_steps=7)
    assert len(steps) == (1 if failure == "adamw_state" else 0)
    if failure == "adamw_state":
        assert torch.isfinite(model.p_in.weight).all() and not torch.isfinite(optimizer.state[model.p_in.weight]["exp_avg_sq"]).all()

def test_e06_step_no_consistency_full_model_and_two_fresh_graph_updates(monkeypatch):
    """批准对照只双零权重，仍每步5Phi；两步重新构图且日志只含Python标量。"""
    module = e06_script()
    from test_model import d12_record
    model, batch = e06_case()
    optimizer = module.make_optimizer(model)
    queries = d12_record(monkeypatch, model, "prepare_query")
    config = e06_config(time_weight=0., space_weight=0.)
    first = module.train_step(model, optimizer, batch, loss_config=config, completed_steps=0, condition="no_consistency")
    second = module.train_step(model, optimizer, batch, loss_config=config, completed_steps=first.completed_steps, condition="no_consistency")
    assert first.completed_steps == 1 and second.completed_steps == 2 and len(queries) == 10
    assert len({id(q[2]) for q in queries}) == 10
    for result in (first, second):
        assert result.loss_values["total"] == result.loss_values["jepa"]
        assert result.loss_values["weighted_time"] == result.loss_values["weighted_space"] == 0.
        assert all(type(value) is float for value in result.loss_values.values())

def e06_checkpoint_api():
    """保存恢复层独立RED；不以只存字段的类型存在冒称基本恢复已经实现。"""
    module = e06_script()
    assert hasattr(module, "save_training_checkpoint") and hasattr(module, "load_training_checkpoint"), "E06自有checkpoint尚未实现"
    return module

def e06_tree_equal(a, b):
    """核对真实参数/优化器/RNG树，不只比较文件可读或预测shape。"""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return type(b) is type(a) and a.keys() == b.keys() and all(e06_tree_equal(a[k], b[k]) for k in a)
    if type(a) in (tuple, list):
        return type(a) is type(b) and len(a) == len(b) and all(e06_tree_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b

def e06_rng_values(device):
    """全局RNG下一次取值，只读当前训练实际CUDA设备；无关设备不宣称覆盖。"""
    import random
    values = [random.random(), np.random.random(), torch.rand(3)]
    if torch.device(device).type == "cuda": values.append(torch.rand(3, device=device).cpu())
    return values

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_e06_checkpoint_real_update_roundtrip_rng_native_schema_and_next_step(tmp_path, monkeypatch, device):
    """一次真实更新后安全格式同进程往返；逐项核moments、固定buffer、RNG及再更新。"""
    import hashlib
    module = e06_checkpoint_api()
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    model, batch = e06_case(device=device)
    optimizer = module.make_optimizer(model)
    module.train_step(model, optimizer, batch, loss_config=e06_config(), completed_steps=0)
    generator = torch.Generator().manual_seed(739)
    sampling = dict(kind="explicit_prepared_batches", next_batch_index=1, query_generator_state=generator.get_state())
    config = e06_config(time_weight=Fraction(1, 1000), space_weight=np.float64(.001))
    path = tmp_path/"explicit-step1.pt"
    digest = module.save_training_checkpoint(path, model=model, optimizer=optimizer, loss_config=config,
        condition="full", completed_steps=1, sampling_state=sampling, provenance=batch.provenance)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    expected_rng, expected_query = e06_rng_values(device), torch.rand(4, generator=generator)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"schema_version", "completed_steps", "condition", "model_state", "optimizer_state", "optimizer_param_names", "model_config", "loss_config", "optimizer_config", "provenance", "rng_state", "sampling_state", "model_training", "scheduler_state", "scaler_state", "runtime"}
    assert required <= payload.keys() and payload["schema_version"] == 1
    assert payload["scheduler_state"] is payload["scaler_state"] is None
    assert type(payload["rng_state"]["numpy"]["keys"]) is list
    assert all(type(n) is int for n in payload["rng_state"]["numpy"]["keys"])
    assert all(type(v) is (str if k == "consistency_metric" else float) for k, v in payload["loss_config"].items())
    assert isinstance(config.time_weight, Fraction) and isinstance(config.space_weight, np.float64)
    clone, clone_batch = e06_case(device=device)
    clone_optimizer = module.make_optimizer(clone)
    clone.eval()
    ids = [id(p) for p in clone.parameters()]
    load_calls, original_load = [], torch.load
    def check_load(*args, **kwargs):
        """实际加载必须显式weights_only=True/mapcpu，不增加allowlist。"""
        load_calls.append(kwargs.copy())
        return original_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", check_load)
    progress = module.load_training_checkpoint(path, expected_sha256=digest, model=clone, optimizer=clone_optimizer,
        loss_config=config, condition="full", expected_provenance=batch.provenance)
    assert load_calls == [dict(map_location="cpu", weights_only=True)]
    assert progress.completed_steps == 1 and progress.sampling_state["next_batch_index"] == 1
    assert ids == [id(p) for p in clone.parameters()] and clone.training
    assert e06_tree_equal(model.state_dict(), clone.state_dict())
    assert e06_tree_equal(optimizer.state_dict(), clone_optimizer.state_dict())
    assert e06_tree_equal(expected_rng, e06_rng_values(device))
    generator.set_state(progress.sampling_state["query_generator_state"])
    assert torch.equal(expected_query, torch.rand(4, generator=generator))
    for p in clone.parameters():
        assert clone_optimizer.state[p]["exp_avg"].device == p.device
        assert clone_optimizer.state[p]["step"].device.type == "cpu"
    original_result = module.train_step(model, optimizer, batch, loss_config=config, completed_steps=1)
    restored_result = module.train_step(clone, clone_optimizer, clone_batch, loss_config=config, completed_steps=1)
    assert original_result.loss_values == restored_result.loss_values
    assert e06_tree_equal(model.state_dict(), clone.state_dict())
    assert e06_tree_equal(optimizer.state_dict(), clone_optimizer.state_dict())
    print("E06 basic checkpoint", dict(device=device, sha256=digest, completed_steps=progress.completed_steps,
        optimizer_states=len(clone_optimizer.state), RNG_next_values_equal=True, next_update_equal=True,
        schema=payload["schema_version"], F07_cross_process_verified=False))

@pytest.mark.parametrize("bad", ["missing", "sha", "schema", "condition", "loss_config", "loss_bool", "model_config", "optimizer_config", "optimizer_group_bool", "provenance", "names", "shape", "dtype", "keys", "fixed_buffer", "state_nonfinite", "state_shape", "state_key", "step", "rng_python", "rng_python_cache", "rng_numpy", "rng_torch", "rng_cuda", "sampling_kind", "sampling_index", "sampling_generator", "mode", "scheduler", "runtime"])
def test_e06_checkpoint_preflight_rejects_before_mutating_model_optimizer_rng(tmp_path, bad):
    """逐项破坏真实文件并重算合法SHA；格式预检错误不得半加载后才报错。"""
    import copy
    import hashlib
    import random
    module = e06_checkpoint_api()
    model, batch = e06_case()
    optimizer = module.make_optimizer(model)
    module.train_step(model, optimizer, batch, loss_config=e06_config(), completed_steps=0)
    path = tmp_path/"source.pt"
    digest = module.save_training_checkpoint(path, model=model, optimizer=optimizer, loss_config=e06_config(), condition="full",
        completed_steps=1, sampling_state=dict(kind="explicit_prepared_batches", next_batch_index=1, query_generator_state=None), provenance=batch.provenance)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    parameter = next(iter(payload["optimizer_state"]["state"].values()))
    if bad == "missing": path = tmp_path/"absent.pt"
    elif bad == "sha": digest = "0"*64
    elif bad == "schema": payload["schema_version"] = 2
    elif bad == "condition": payload["condition"] = "no_consistency"
    elif bad == "loss_config": payload["loss_config"]["time_weight"] = .01
    elif bad == "loss_bool": payload["loss_config"]["jepa_exponent"] = True
    elif bad == "model_config": payload["model_config"]["projection"]["encoder_dim"] = 4
    elif bad == "optimizer_config": payload["optimizer_config"]["defaults"]["lr"] = .1
    elif bad == "optimizer_group_bool": payload["optimizer_state"]["param_groups"][0]["weight_decay"] = False
    elif bad == "provenance": payload["provenance"]["split"] = "test"
    elif bad == "names": payload["optimizer_param_names"][0].reverse()
    elif bad == "shape": payload["model_state"]["p_in.weight"] = payload["model_state"]["p_in.weight"][:1]
    elif bad == "dtype": payload["model_state"]["p_in.weight"] = payload["model_state"]["p_in.weight"].double()
    elif bad == "keys": payload["model_state"].pop("p_in.bias")
    elif bad == "fixed_buffer": payload["model_state"]["spectrum.q"] = -payload["model_state"]["spectrum.q"]
    elif bad == "state_nonfinite": parameter["exp_avg_sq"].fill_(float("inf"))
    elif bad == "state_shape": parameter["exp_avg"] = parameter["exp_avg"][:1]
    elif bad == "state_key": parameter["unknown"] = torch.tensor(1.)
    elif bad == "step": payload["completed_steps"] = True
    elif bad == "rng_python": payload["rng_state"]["python"] = (1,)
    elif bad == "rng_python_cache": payload["rng_state"]["python"] = (*payload["rng_state"]["python"][:2], True)
    elif bad == "rng_numpy": payload["rng_state"]["numpy"]["keys"][0] = -1
    elif bad == "rng_torch": payload["rng_state"]["torch_cpu"] = torch.zeros(4, dtype=torch.uint8)
    elif bad == "rng_cuda": payload["rng_state"]["cuda"] = dict(device_index=0, state=torch.zeros(8, dtype=torch.uint8))
    elif bad == "sampling_kind": payload["sampling_state"]["kind"] = "hidden_loader"
    elif bad == "sampling_index": payload["sampling_state"]["next_batch_index"] = -1
    elif bad == "sampling_generator": payload["sampling_state"]["query_generator_state"] = torch.zeros(4, dtype=torch.uint8)
    elif bad == "mode": payload["model_training"] = "train"
    elif bad == "scheduler": payload["scheduler_state"] = {}
    elif bad == "runtime": payload["runtime"]["torch"] = "wrong_version"
    if bad not in ("missing", "sha"):
        torch.save(payload, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    clone, _ = e06_case()
    clone_optimizer = module.make_optimizer(clone)
    original_model, original_optimizer = copy.deepcopy(clone.state_dict()), copy.deepcopy(clone_optimizer.state_dict())
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError, FileNotFoundError, RuntimeError)):
        module.load_training_checkpoint(path, expected_sha256=digest, model=clone, optimizer=clone_optimizer,
            loss_config=e06_config(), condition="full", expected_provenance=batch.provenance)
    assert e06_tree_equal(clone.state_dict(), original_model) and e06_tree_equal(clone_optimizer.state_dict(), original_optimizer)
    assert random.getstate() == python_rng and np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert np.random.get_state()[2:] == numpy_rng[2:] and torch.equal(torch.get_rng_state(), cpu_rng)

def test_e06_checkpoint_exclusive_publish_existing_file_and_failure_cleanup(tmp_path, monkeypatch):
    """同目录独占发布，已有目标和竞争产生的旧目标都保留，仅清自己的临时文件。"""
    module = e06_checkpoint_api()
    model, batch = e06_case()
    optimizer = module.make_optimizer(model)
    args = dict(model=model, optimizer=optimizer, loss_config=e06_config(), condition="full", completed_steps=0,
        sampling_state=dict(kind="explicit_prepared_batches", next_batch_index=0, query_generator_state=None), provenance=batch.provenance)
    path = tmp_path/"old.pt"
    path.write_bytes(b"keep original")
    with pytest.raises(FileExistsError): module.save_training_checkpoint(path, **args)
    assert path.read_bytes() == b"keep original"
    fresh = tmp_path/"new.pt"
    def broken_save(*a, **kw):
        """模拟写入失败，不把清理范围扩大到目录或已有checkpoint。"""
        raise OSError("fictional write failure")
    original_save = torch.save
    monkeypatch.setattr(torch, "save", broken_save)
    before = set(tmp_path.iterdir())
    with pytest.raises(OSError, match="fictional"): module.save_training_checkpoint(fresh, **args)
    assert set(tmp_path.iterdir()) == before and path.read_bytes() == b"keep original"
    monkeypatch.setattr(torch, "save", original_save)
    original_link = module.os.link
    def raced_publish(source, destination):
        """模拟预检后另一个写者已发布目标；原生独占发布必须拒绝覆盖。"""
        destination.write_bytes(b"other writer")
        return original_link(source, destination)
    monkeypatch.setattr(module.os, "link", raced_publish)
    with pytest.raises(FileExistsError): module.save_training_checkpoint(fresh, **args)
    assert fresh.read_bytes() == b"other writer" and set(tmp_path.iterdir()) == before | {fresh}

def test_e06_prepared_noncontiguous_decimal_times_and_unrepresentable_endpoint():
    """普通小数采用原tau+sigma；raw允许非连续，巨大物理秒坍缩不能靠容差接收。"""
    module = e06_script()
    model, batch = e06_case()
    tau = torch.tensor([.1,.2], dtype=torch.float64)
    sigma = torch.tensor([.2,.1], dtype=torch.float64)
    duration = tau+sigma
    raw = batch.raw_z.transpose(1,2).contiguous().transpose(1,2)
    target = batch.raw_target.transpose(1,2).contiguous().transpose(1,2)
    batch = replace(batch, raw_z=raw, raw_target=target, tau=tau, sigma=sigma, duration=duration,
        target_time_seconds=batch.observed_time_seconds+duration*batch.controls.control_spec.control_period_seconds)
    assert not raw.is_contiguous() and not target.is_contiguous()
    optimizer = module.make_optimizer(model)
    result = module.train_step(model, optimizer, batch, loss_config=e06_config(), completed_steps=0)
    assert result.completed_steps == 1
    rounded = replace(batch, duration=torch.full((2,), .3, dtype=torch.float64))
    with pytest.raises(ValueError, match="tau\\+sigma"):
        module.train_step(model, optimizer, rounded, loss_config=e06_config(), completed_steps=1)
    observed = torch.full((2,), 2.**52, dtype=torch.float64)
    collapsed = replace(batch, controls=replace(batch.controls, source_origin_seconds=observed.clone()),
        observed_time_seconds=observed, target_time_seconds=observed+duration*batch.controls.control_spec.control_period_seconds)
    with pytest.raises(ValueError): module.train_step(model, optimizer, collapsed, loss_config=e06_config(), completed_steps=1)

def test_e06_target_only_changes_jepa_not_any_phi_or_consistency_and_preserves_inputs():
    """独立未来标签fixture改变时五Phi/时空原值不变；目标只在距离端可见。"""
    import copy
    module = e06_script()
    model, batch = e06_case()
    clone = copy.deepcopy(model)
    outputs, other_outputs = [], []
    model.register_forward_hook(lambda owner, args, result: outputs.append(result.detach().clone()))
    clone.register_forward_hook(lambda owner, args, result: other_outputs.append(result.detach().clone()))
    tensors = [batch.raw_z, batch.raw_target, batch.duration, batch.tau, batch.sigma,
        batch.observed_time_seconds, batch.target_time_seconds, batch.controls.starts, batch.controls.ends,
        batch.controls.u_values, batch.transfer.source.coords, batch.transfer.target.coords]
    before = [(v.clone(), v._version) for v in tensors]
    first = module.train_step(model, module.make_optimizer(model), batch, loss_config=e06_config(), completed_steps=0)
    changed = replace(batch, raw_target=torch.zeros_like(batch.raw_target))
    second = module.train_step(clone, module.make_optimizer(clone), changed, loss_config=e06_config(), completed_steps=0)
    assert len(outputs) == len(other_outputs) == 5
    assert all(torch.equal(a,b) for a,b in zip(outputs, other_outputs))
    assert first.loss_values["time"] == second.loss_values["time"] and first.loss_values["space"] == second.loss_values["space"]
    assert first.loss_values["jepa"] != second.loss_values["jepa"]
    for value, (old, version) in zip(tensors, before):
        assert value._version == version
        torch.testing.assert_close(value, old, atol=0, rtol=0, equal_nan=True)

