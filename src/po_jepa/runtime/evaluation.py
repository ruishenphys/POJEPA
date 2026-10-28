"""留出评价：只读预测、原始分布与失败留证。

单网格 future/TCD/Delta 与成对 SCD 分开索引，沿共同损失与指标。
不训练、不用 test 选 checkpoint；GPU 与 CPU 合成测试结果分别标识。
"""

import argparse

from dataclasses import asdict, dataclass, replace

import hashlib

import io

import math

from pathlib import Path

import torch

from . import comparison as comparison
from po_jepa.actions import clip_action_segments

from po_jepa import metrics

ROOT = Path(__file__).resolve().parents[3]

CONDITIONS = (*comparison.CONDITIONS, "identity")

TRAIN_TIMES = (1., 2., 4., 8.)

TIMES = (1., 2., 3., 4., 8., 12.)

LEVELS = (.5, .9, .95, .99)

@dataclass(frozen=True)
class EvaluationQuery:
    """test 查询，周期与秒分离；grid 为单网格或 R 的目标边长。

    point 共1600点；pair 为两种 R 各六时长，共1200个唯一对。
    网格行重复引用同一 pair 不增加样本数。
    """
    trajectory_id: int
    start_tick: int
    duration_cycles: float
    grid_size: int
    kind: str

    def __post_init__(self):
        """外部索引先验证，拒绝跨split、未登记时长及同网格伪SCD。"""
        if type(self.trajectory_id) is not int or not 180 <= self.trajectory_id < 200:
            raise ValueError("F06 requires test trajectory 180..199")
        if type(self.start_tick) is not int or self.start_tick not in (0,5,10,15,20):
            raise ValueError("unregistered evaluation start")
        if type(self.duration_cycles) not in (int,float) or self.duration_cycles not in TIMES:
            raise ValueError("unregistered evaluation duration")
        if type(self.grid_size) is not int or self.grid_size not in (8,12,16):
            raise ValueError("unregistered grid")
        if self.kind not in ("point","pair"):
            raise ValueError("query kind must be point or pair")
        if self.kind == "pair" and self.grid_size == 16:
            raise ValueError("SCD requires R16->8 or R16->12")
        if self.kind == "point" and self.grid_size == 8 and self.duration_cycles not in TRAIN_TIMES:
            raise ValueError("F01 does not register grid8 heldout-time accuracy points")

    @property
    def stratum(self):
        """由真实时长/网格唯一推得分层，避免标签与计算条件不一致。"""
        unseen_time = self.duration_cycles not in TRAIN_TIMES
        if self.grid_size == 12:
            return "joint_heldout" if unseen_time else "grid_heldout"
        return "time_heldout" if unseen_time else "id"

    def key(self, condition, seed=1703):
        """条件与预登记种子进入唯一键；不将同窗不同条件误认为额外轨迹。"""
        if condition not in CONDITIONS or type(seed) is not int or seed != 1703:
            raise ValueError("condition/seed outside F01/F05 registration")
        return (condition, seed, self.kind, self.trajectory_id, self.start_tick,
                float(self.duration_cycles), self.grid_size, self.stratum)

def donor_id(trajectory_id):
    """test 内下一轨迹循环；不看未来误差或 Delta 重新抽样。"""
    if type(trajectory_id) is not int or not 180 <= trajectory_id < 200:
        raise ValueError("donor recipient must belong to test")
    return 180 + (trajectory_id-180+1) % 20

def build_index():
    """生成固定1600个单网格点及1200个唯一SCD对，不读取数据或执行模型。"""
    single = [(t,g,"point") for t in TIMES for g in (16,8,12) if g != 8 or t in TRAIN_TIMES]
    pairs = [(t,g,"pair") for t in TIMES for g in (8,12)]
    return [EvaluationQuery(tid,start,t,g,kind) for tid in range(180,200)
            for start in (0,5,10,15,20) for t,g,kind in single+pairs]

def load_test_window(features, query, device):
    """核真实 test 冻结窗口，不放宽 train/validation 读取边界。

    实际窗口为有限 FP32 [1,256,1408]、CPU float64 时间和真实 canonical
    来源，按固定 manifest 与精确物理端点核身份，仅显式评价时读取。
    """
    if not isinstance(query, EvaluationQuery):
        raise TypeError("explicit EvaluationQuery required before I/O")
    query.__post_init__()
    approved = comparison.data.approved_cache_metadata(features)
    window = comparison.data.cache.read_window(Path(features)/f"{query.trajectory_id:04d}",
        start_tick=query.start_tick,duration_cycles=query.duration_cycles,expected_split="test",device=str(device))
    x, y, p = window.inputs, window.target, window.provenance
    comparison.data._window_boundary(window,window)
    if (x.trajectory_id != query.trajectory_id or x.split != "test"
        or p["actual_manifest_sha256"] != approved["manifest_sha256_by_id"][query.trajectory_id]
        or p["generation"]["evidence_role"] != "real_frozen_encoder" or p["observation_variant"] != "canonical"
        or float(x.duration[0]) != query.duration_cycles or float(x.observed_time_seconds[0]) != query.start_tick*.25
        or float(y.target_time_seconds[0]) != (query.start_tick+query.duration_cycles)*.25
        or x.controls.source_ids != (f"trajectory-{query.trajectory_id:04d}",)):
        raise ValueError("test identity/source/origin/endpoint mismatch")
    return window

def spatial_view(window, size):
    """同一 R 作用于起点、真实未来和坐标；不重新编码或另造粗网格坐标。"""
    if size == 16:
        return window.inputs.raw_z, window.target.raw_z, window.inputs.grid
    transfer = comparison.build_spatial_transfer(window.inputs.grid,target_height=size,target_width=size)
    return (comparison.apply_spatial_transfer(window.inputs.raw_z,transfer),
            comparison.apply_spatial_transfer(window.target.raw_z,transfer),transfer.target)

def _aligned_pair(window, donor, query):
    """核真实recipient/donor ID、同起点/时长和控制定义；donor latent/标签不用。"""
    if (window.inputs.trajectory_id != query.trajectory_id or window.inputs.split != "test"
        or donor.inputs.trajectory_id != donor_id(query.trajectory_id) or donor.inputs.split != "test"):
        raise ValueError("test recipient/donor identities differ from fixed cyclic mapping")
    for w in (window,donor):
        x = w.inputs
        expected_origin = torch.tensor([query.start_tick*.25],dtype=torch.float64)
        if (x.controls.source_ids != (f"trajectory-{x.trajectory_id:04d}",)
            or not torch.equal(x.duration,torch.tensor([float(query.duration_cycles)],dtype=torch.float64))
            or not torch.equal(x.observed_time_seconds,expected_origin)
            or not torch.equal(x.controls.source_origin_seconds,expected_origin)):
            raise ValueError("control source/local time/physical origin mismatch")
    metrics._action_pair(window.inputs.raw_z,window.inputs.controls,donor.inputs.controls,window.inputs.duration)

def _recipient(window, query):
    """单点与网格对共同核窗口身份和精确时间；原始缓存的真实来源另由reader验证。"""
    x = window.inputs
    if (x.trajectory_id != query.trajectory_id or x.split != "test"
        or (x.grid.height,x.grid.width) != (16,16)
        or x.raw_z.shape[:2] != (1,256) or x.raw_z.dtype != torch.float32
        or x.raw_z.requires_grad or not torch.isfinite(x.raw_z).all()
        or window.target.raw_z.shape != x.raw_z.shape or window.target.raw_z.dtype != x.raw_z.dtype
        or window.target.raw_z.device != x.raw_z.device or window.target.raw_z.requires_grad
        or not torch.isfinite(window.target.raw_z).all()
        or x.controls.control_spec.control_period_seconds != .25
        or x.duration.dtype != torch.float64 or x.duration.device.type != "cpu"
        or not torch.equal(x.duration,torch.tensor([float(query.duration_cycles)],dtype=torch.float64))
        or not torch.equal(x.observed_time_seconds,torch.tensor([query.start_tick*.25],dtype=torch.float64))
        or not torch.equal(window.target.target_time_seconds,torch.tensor([(query.start_tick+query.duration_cycles)*.25],dtype=torch.float64))
        or x.controls.source_ids != (f"trajectory-{query.trajectory_id:04d}",)
        or not torch.equal(x.controls.source_origin_seconds,x.observed_time_seconds)):
        raise ValueError("recipient identity/shape/time/source mismatch")
    x.grid.__post_init__()
    metrics.validate_action_segments(x.controls,query_duration=x.duration)

def identical_controls(executed, alternative, duration):
    """比较整个查询内的分段常值函数；不受padding或冗余分段数量影响。"""
    rows = [clip_action_segments(c,start=torch.zeros_like(duration),end=duration) for c in (executed,alternative)]
    ends = [c.ends[0,c.segment_valid[0]] for c in rows]
    # 取两条路径的边界并集以比较常值函数，不改变动作采样顺序。
    knots = torch.unique(torch.cat([torch.zeros(1,dtype=torch.float64),*ends]),sorted=True)
    mid = (knots[:-1]+knots[1:])/2
    sampled = [c.u_values[0,c.segment_valid[0]][torch.searchsorted(e,mid).to(c.u_values.device)] for c,e in zip(rows,ends)]
    return torch.equal(*sampled)

def _eval_only(model):
    """低层数学入口不暗改模式；外层负责临时eval并在异常时恢复。"""
    if any(m.training for m in model.modules()):
        raise ValueError("evaluation requires eval mode")

def _values(items):
    """只把有限单样本指标转为JSON标量；NaN/Inf抛错，禁止静默填0。"""
    out = {}
    for metric in items:
        if metric.values.shape != (1,) or not torch.isfinite(metric.values).all():
            raise ValueError(f"nonfinite or nonscalar evaluation metric: {metric.name}")
        out[metric.name] = float(metric.values[0])
    return out

def _scale(value):
    """原始表示尺度与JEPA归一化距离分开；FP64统计避免先平方溢出。"""
    x = value.detach().double()
    return dict(mean=float(x.mean()),rms=float(x.square().mean().sqrt()),std=float(x.std(unbiased=False)))

@torch.no_grad()
def point_metrics(model, window, donor, query, config):
    """式14/15与附录 Delta：单网格，第二段仅从预测 raw Z 重启完整 Phi。

    TCD/RMS 沿共同指标；Delta 仅换 donor 控制，不用其起点或标签，
    所有预测器使用相同 JEPA 差定义，不反传或更新优化器。
    """
    _eval_only(model)
    if query.kind != "point": raise ValueError("single-grid point required")
    _recipient(window,query)
    _aligned_pair(window,donor,query)
    raw, target, grid = spatial_view(window,query.grid_size)
    controls, duration = window.inputs.controls, window.inputs.duration
    tau = duration*.5
    pair = comparison.split_action_segments(controls,tau=tau,sigma=tau)
    direct = model(raw,comparison.concatenate_action_segments(pair.first,pair.second),duration,grid)
    first = model(raw,pair.first,tau,grid)
    composed = model(first,pair.second,tau,grid)
    record = comparison.TemporalPredictions(direct,first,composed,direct-composed,tau.clone(),tau.clone())
    future = metrics.SampleMetric(comparison.jepa_distance(direct,target,config=config).per_sample,"future_jepa_distance","dimensionless")
    if isinstance(model,comparison.POJEPA):
        delta = metrics.action_intervention_margin(model,raw,controls,donor.inputs.controls,duration,grid,target,loss_config=config)
    else:
        metrics._action_pair(raw,controls,donor.inputs.controls,duration)
        actual = comparison.jepa_distance(model(raw,controls,duration,grid),target,config=config).per_sample
        alternative = comparison.jepa_distance(model(raw,donor.inputs.controls,duration,grid),target,config=config).per_sample
        delta = metrics.SampleMetric(alternative-actual,"delta_act","dimensionless")
    values = _values((future,metrics.temporal_composition_defect(record),metrics.temporal_composition_rms_defect(record),delta))
    values.update(tcd_per_second2=values["tcd_frobenius"]*16,tcd_rms_per_second2=values["tcd_rms"]*16,
                  tcd_normalized=values["tcd_rms"]/math.sqrt(float(comparison.input_scale(raw)[0])))
    return values,dict(prediction_scale=_scale(direct),target_scale=_scale(target),
                       identical_controls=identical_controls(controls,donor.inputs.controls,duration))

@torch.no_grad()
def pair_metrics(model, window, query):
    """式16/32：同窗规范Z上的R16→8/12两侧交换缺陷，独立于单网格表。"""
    _eval_only(model)
    if query.kind != "pair": raise ValueError("grid pair required")
    _recipient(window,query)
    transfer = comparison.build_spatial_transfer(window.inputs.grid,target_height=query.grid_size,target_width=query.grid_size)
    raw, controls, duration = window.inputs.raw_z, window.inputs.controls, window.inputs.duration
    fine = comparison.apply_spatial_transfer(model(raw,controls,duration,transfer.source),transfer)
    coarse = model(comparison.apply_spatial_transfer(raw,transfer),controls,duration,transfer.target)
    record = comparison.SpatialPredictions(fine,coarse,fine-coarse,duration.clone(),transfer.target)
    values = _values((metrics.spatial_commutation_defect(record),metrics.spatial_commutation_rms_defect(record)))
    values.update(scd_per_second=values["scd_frobenius"]*4,scd_rms_per_second=values["scd_rms"]*4,
                  scd_normalized=values["scd_rms"]/math.sqrt(float(comparison.input_scale(raw)[0])))
    return values,dict(source_grid=[16,16],target_grid=[query.grid_size]*2,
                       fine_after_R_scale=_scale(fine),coarse_scale=_scale(coarse))

def metric_units(kind):
    """忠实Frobenius/RMS和周期/秒各自命名；不混用训练平方损失。"""
    if kind == "point":
        return dict(future_jepa_distance="dimensionless",tcd_frobenius="control_cycle^-2",tcd_rms="control_cycle^-2",
                    delta_act="dimensionless",tcd_per_second2="second^-2",tcd_rms_per_second2="second^-2",
                    tcd_normalized="control_cycle^-2")
    if kind == "pair":
        return dict(scd_frobenius="control_cycle^-1",scd_rms="control_cycle^-1",scd_per_second="second^-1",scd_rms_per_second="second^-1",
                    scd_normalized="control_cycle^-1")
    raise ValueError("unknown metric kind")

def _record_map(records, queries, conditions):
    """先核已声明的共同评价集合、唯一键和有限完整结果，不拼有利子集。"""
    expected = {q.key(c) for c in conditions for q in queries}
    if len(expected) != len(queries)*len(conditions): raise ValueError("duplicate expected query/condition")
    found = {}
    for row in records:
        query = EvaluationQuery(**row["query"])
        key = query.key(row["condition"],row["seed"])
        if key not in expected or key in found or row["status"] not in ("ok","failed"):
            raise ValueError("unexpected/duplicate result identity or status")
        if row["status"] == "ok":
            values = row["values"]
            if set(values) != set(metric_units(query.kind)) or any(type(v) not in (int,float) or not math.isfinite(v) for v in values.values()):
                raise ValueError("successful result requires all finite metrics")
        found[key] = row
    return found

def _group_summary(expected, rows, name, unit):
    """每组按窗口和等权轨迹分别汇总；空组不伪造均值，失败/缺失进入分母清单。"""
    completed = [r for r in rows if r is not None]
    valid = [r for r in completed if r["status"] == "ok"]
    values = [r["values"][name] for r in valid]
    by_trajectory = {}
    for r in valid: by_trajectory.setdefault(r["query"]["trajectory_id"],[]).append(r["values"][name])
    means = [sum(v)/len(v) for v in by_trajectory.values()]
    summary = metrics.summarize_metric(metrics.SampleMetric(torch.tensor(values,dtype=torch.float64),name,unit),quantile_levels=LEVELS) if values else None
    return dict(metric=name,unit=unit,expected=expected,completed=len(completed),valid=len(valid),
        failed=len(completed)-len(valid),not_run=expected-len(completed),status="ok" if valid else "no_valid_values",
        window_weighted_mean=summary.mean if summary else None,trajectory_macro_mean=sum(means)/len(means) if means else None,
        valid_trajectories=len(means),quantile_levels=list(LEVELS),quantiles=list(summary.quantile_values) if summary else None)

def summarize_records(records, queries, conditions=CONDITIONS):
    """按条件/种子/网格或网格对/时长/层级分组，不混池不同Frobenius尺度。"""
    found = _record_map(records,queries,conditions)
    grouped = {}
    for condition in conditions:
        for q in queries:
            key = (condition,1703,q.kind,q.stratum,q.grid_size,float(q.duration_cycles))
            grouped.setdefault(key,[]).append(found.get(q.key(condition)))
    groups = []
    for (condition,seed,kind,stratum,grid,duration),rows in grouped.items():
        for name,unit in metric_units(kind).items():
            groups.append(dict(condition=condition,seed=seed,kind=kind,stratum=stratum,grid_size=grid,
                duration_cycles=duration,source_grid_size=16 if kind=="pair" else grid,
                **_group_summary(len(rows),rows,name,unit)))
    ok = sum(r["status"]=="ok" for r in found.values())
    return dict(groups=groups,counts=dict(expected=len(queries)*len(conditions),completed=len(found),valid=ok,
        failed=len(found)-ok,not_run=len(queries)*len(conditions)-len(found)))

def _tradeoff(future_difference, consistency_difference):
    """只描述本次逐点差值，不写显著/稳健或因果结论；Delta不套越低越好。"""
    if future_difference > 0 and consistency_difference < 0: return "consistency_lower_prediction_worse"
    if future_difference < 0 and consistency_difference < 0: return "both_lower"
    if future_difference < 0: return "prediction_lower_consistency_not_lower"
    return "no_strict_joint_improvement"

def paired_comparisons(records, queries):
    """full-reference逐键配对；未运行/失败也列出，不只保存共同成功窗口。"""
    found = _record_map(records,queries,CONDITIONS)
    results = []
    for reference in ("matched_transformer","po_jepa_no_consistency","identity"):
        paired, unmatched = [], []
        for q in queries:
            full, other = found.get(q.key("po_jepa_full")), found.get(q.key(reference))
            missing = "full_not_run" if full is None else "reference_not_run" if other is None else "full_failed" if full["status"]!="ok" else "reference_failed" if other["status"]!="ok" else None
            if missing:
                unmatched.append(dict(query=asdict(q),reason=missing)); continue
            delta = {n:full["values"][n]-other["values"][n] for n in metric_units(q.kind)}
            tag = "accuracy_not_measured_on_this_grid_time"
            if q.kind == "point":
                tag = _tradeoff(delta["future_jepa_distance"],delta["tcd_normalized"])
            elif not (q.grid_size==8 and q.duration_cycles not in TRAIN_TIMES):
                point = replace(q,kind="point")
                a,b = found.get(point.key("po_jepa_full")),found.get(point.key(reference))
                if a and b and a["status"]==b["status"]=="ok":
                    tag = _tradeoff(a["values"]["future_jepa_distance"]-b["values"]["future_jepa_distance"],delta["scd_normalized"])
                else: tag="paired_accuracy_missing"
            paired.append(dict(query=asdict(q),full_minus_reference=delta,tradeoff=tag))
        results.append(dict(reference=reference,expected=len(queries),paired_valid=len(paired),rows=paired,unmatched=unmatched,
                            delta_interpretation="signed intervention margin; not lower-is-better or causal proof"))
    return results

def load_selected_checkpoint(condition, selected, *, model, optimizer, config, expected_provenance, root):
    """只读指定验证选点路径/SHA，安全加载，不选 latest 或改写训练来源。

    测试指标不得影响 checkpoint 选择；调用者负责完整训练证据准入。
    """
    if condition not in comparison.CONDITIONS or type(selected.get("step")) is not int or selected["step"] < 0:
        raise ValueError("trained condition and explicit successful validation step required")
    if type(selected["mean"]) not in (int,float) or not math.isfinite(selected["mean"]) or selected["mean"] < 0 or selected["checkpoint_provenance"] != expected_provenance:
        raise ValueError("invalid validation future mean or original checkpoint provenance")
    path = (Path(root)/selected["checkpoint"]).resolve()
    if not path.is_relative_to(Path(root).resolve()): raise ValueError("selected checkpoint escaped its run")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != selected["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA mismatch before loading")
    payload = torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    if payload["completed_steps"] != selected["step"]:
        raise ValueError("selected successful step differs before any state restore")
    if condition == "matched_transformer":
        if payload["loss_config"] != comparison.training._loss_config(config,"no_consistency"):
            raise ValueError("selected transformer JEPA configuration differs")
        progress = comparison.load_transformer_checkpoint(path,selected["checkpoint_sha256"],model,optimizer,expected_provenance)
    else:
        progress = comparison.training.load_training_checkpoint(path,expected_sha256=selected["checkpoint_sha256"],model=model,
            optimizer=optimizer,loss_config=config,condition="full" if condition=="po_jepa_full" else "no_consistency",
            expected_provenance=expected_provenance)
    if progress.completed_steps != selected["step"]: raise ValueError("selected successful step differs from checkpoint")
    model.eval()
    return progress

def evaluation_sources():
    """训练来源原样保留，新评价脚本/覆盖决定另存，避免伪造训练时的来源身份。"""
    names = ("src/po_jepa/runtime/evaluation.py","reference/evaluation_coverage.json")
    return dict(training_consumer_sha256=comparison.consumer_sources(),
                evaluation_source_sha256={n:comparison.data.cache.file_sha(ROOT/n) for n in names})

def _state_identity(model):
    """一次评价前后核参数、既有梯度与buffer字节，不在每个GPU窗口重复扫描。"""
    return dict(parameters=comparison.check._parameter_identity(model),buffers={n:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for n,v in model.named_buffers()})

def _write(path, value):
    """沿既有格式保存严格JSON；NaN/Inf必须先转换为显式失败，绝不标准外输出。"""
    comparison.data.cache._write_json(path,value)

def run_prepared_evaluation(model, condition, features, output, config, *, queries=None, checkpoint_identity=None, device=None):
    """只读评价编排，新目录逐点留证，失败即停并统计未运行。

    一次一个条件，不提供优化器或训练步骤；设备显式给定，CPU 合成
    测试身份单独标明。保护参数、已有梯度、buffer 与子模块模式。
    """
    queries = build_index() if queries is None else list(queries)
    if not queries: raise ValueError("nonempty explicit evaluation queries required")
    _record_map([],queries,(condition,))
    expected_type = comparison.IdentityPredictor if condition=="identity" else comparison.TokenTransformer if condition=="matched_transformer" else comparison.POJEPA
    if not isinstance(model,expected_type): raise TypeError("condition/model type mismatch")
    if condition != "identity" and not checkpoint_identity: raise ValueError("selected checkpoint identity required")
    # identity没有参数，必须由调用者明确设备，不能把无参数误判为CPU运行。
    inferred = next(model.parameters(),torch.empty(0)).device
    device = inferred if device is None else torch.device(device)
    if device.type not in ("cpu","cuda") or any(p.device!=device for p in model.parameters()):
        raise ValueError("evaluation device/model mismatch")
    output = Path(output); output.mkdir(parents=True)
    (output/"rows").mkdir(); (output/"trajectories").mkdir()
    _write(output/"registration.json",dict(condition=condition,queries=[asdict(q) for q in queries],checkpoint=checkpoint_identity,sources=evaluation_sources()))
    modes = {m:m.training for m in model.modules()}; identity = _state_identity(model)
    records, error = [], None
    try:
        model.eval()
        for index,q in enumerate(queries):
            row = dict(query=asdict(q),condition=condition,seed=1703,checkpoint=checkpoint_identity,units=metric_units(q.kind))
            try:
                w = load_test_window(features,q,device)
                if q.kind == "point":
                    donor = load_test_window(features,replace(q,trajectory_id=donor_id(q.trajectory_id)),device)
                    values, extra = point_metrics(model,w,donor,q,config)
                    row["donor"] = dict(trajectory_id=donor.inputs.trajectory_id,manifest_sha256=donor.provenance["actual_manifest_sha256"])
                else: values,extra = pair_metrics(model,w,q)
                row.update(status="ok",values=values,diagnostics=extra,error=None,source=w.provenance)
                _record_map([row],[q],(condition,))
            except Exception as exc:
                row.update(status="failed",values=None,error=dict(type=type(exc).__name__,message=str(exc)))
                error = exc
            records.append(row); _write(output/"rows"/f"{index:05d}.json",row)
            if error is not None: break
    finally:
        for module,flag in modes.items(): module.training=flag
        if _state_identity(model) != identity: error = RuntimeError("evaluation changed parameters, gradients or buffers")
    summary = summarize_records(records,queries,(condition,))
    real=device.type=="cuda" and all(row.get("source",{}).get("generation",{}).get("evidence_role")=="real_frozen_encoder" for row in records)
    report = dict(status="failed" if error else "completed_real_evaluation" if real else "completed_fictional_cpu_only",
        evidence_role="real_frozen_features" if real else "fictional_cpu",condition=condition,**summary,
        real_evaluation_run=real,scientific_conclusions_verified=False)
    for tid in sorted({q.trajectory_id for q in queries}):
        subset=[r for r in records if r["query"]["trajectory_id"]==tid]
        expected=[q for q in queries if q.trajectory_id==tid]
        _write(output/"trajectories"/f"{tid:04d}.json",dict(records=subset,summary=summarize_records(subset,expected,(condition,))))
    _write(output/"report.json",report)
    _write(output/("FAILED.json" if error else "COMPLETE.json"),dict(status=report["status"],error=str(error) if error else None))
    if error is not None: raise error
    return report

