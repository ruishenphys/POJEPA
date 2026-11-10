"""10 §4.1–§4.2：D4 读一个已完成任务的验证选中检查点，核数据一致性，不核源码 SHA。

v7 的读法（evaluation.load_selected_checkpoint → training._provenance）会把检查点来源里的 source_files_sha256 和当前
源码逐个比较，po-jepa-full 的源码和 v7 不同，所以 v7 的检查点按原路径读不进来。这里按 10 §4.2 的 6 步读：任务和
训练目录的完成证据、选点规则、查询序列、检查点文件、配置快照、模型状态、来源里的 8 个数据字段都核；
source_files_sha256 原样记进返回值（D4 的 source.json），不核。补跑自己的检查点也走这里。

不读优化器状态、不恢复随机状态，只把 model_state 读进调用者现建的模型，然后 model.eval()。
"""

from dataclasses import asdict

import hashlib

import io

from pathlib import Path

import torch

from .. import cache, checks, comparison, data, learning, training
from . import conditions as addon_conditions

# 10 §4.2 第 6 步：来源里必须和当前一致的数据字段（source_files_sha256 不在其中）
DATA_FIELDS = ("evidence_role", "split", "paper_sha256", "upstream_commit", "encoder_checkpoint_sha256",
               "preprocess_sha256", "feature_manifest_sha256", "protocol_sha256")

PO_LABELS = dict(po_jepa_full="full", po_jepa_no_consistency="no_consistency", po_jepa_time_only="time_only",
                 po_jepa_space_only="space_only", fixed_fourier="full")
TRANSFORMER_LABELS = dict(matched_transformer="no_consistency", transformer_consistency="full")
AC_SCHEMA = "v7_vjepa2_ac_checkpoint_v1"
TRANSFORMER_SCHEMA = "F05_transformer_checkpoint_v1"
D4_CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity", "vjepa2_ac_small", "vjepa2_ac",
                 *addon_conditions.ADDON_CONDITIONS)


def _read(path):
    from ..workflow import read
    return read(path)


def _complete(folder):
    """COMPLETE.json 的 report_sha256 等于 report.json 的实际 SHA，且没有 FAILED.json。"""
    if (folder/"FAILED.json").exists():
        raise ValueError(f"{folder} retains FAILED.json")
    if _read(folder/"COMPLETE.json")["report_sha256"] != cache.file_sha(folder/"report.json"):
        raise ValueError(f"{folder}/report.json differs from its COMPLETE.json")
    return _read(folder/"report.json")


def job_record(job_dir, condition):
    """第 1 步：任务目录的 job.json 和 D4 命令一致，COMPLETE.json 绑定 report.json；返回 (job, report)。"""
    job_dir = Path(job_dir)
    job = _read(job_dir/"job.json")
    expected = dict(condition=condition, dataset=data.datasets.active().tag, seed=checks.model_seed(), updates=checks.v7_updates())
    if "kind" in job or any(job.get(k) != v for k, v in expected.items()):
        raise ValueError(f"job.json differs from the D4 command: expected {expected}, found "
                         f"{ {k: job.get(k) for k in ('kind', *expected)} }")
    return job, _complete(job_dir)


def training_dir(job_dir, job, condition):
    """第 2 步的训练目录：<任务目录>/training/<条件>；续训或只评测的任务复用了旧训练时（目录里没有完成的训练），
    沿 job.json 的 resume_from / training_run 找到旧任务目录，再取它的 training/<条件>，最多跟 8 层。"""
    from ..v7jobs import _job_dir
    folder, record = Path(job_dir), job
    for _ in range(9):
        candidate = folder/"training"/condition
        if (candidate/"COMPLETE.json").exists():
            return candidate
        link = record.get("training_run") or record.get("resume_from")
        if not link:
            break
        folder = _job_dir(link)
        record = _read(folder/"job.json")
        if record.get("condition") != condition:
            raise ValueError("linked old job belongs to another condition")
    raise ValueError(f"no completed training/{condition} under {job_dir} or its linked old jobs")


def _payload_family(condition, payload, model, loss_config):
    """第 4 步：配置快照。返回记进 source.json 的快照。"""
    if condition in PO_LABELS:
        label = PO_LABELS[condition]
        if (type(payload.get("schema_version")) is not int or payload["schema_version"] != 1
                or payload.get("condition") != label
                or not training._same_native(payload.get("model_config"), training._model_config(model))
                or not training._same_native(payload.get("loss_config"), training._loss_config(loss_config, label))):
            raise ValueError(f"{condition} checkpoint condition/model_config/loss_config differs from the current build")
        return dict(schema_version=1, condition=label, model_config=payload["model_config"], loss_config=payload["loss_config"])
    if condition == "koopman":
        if (type(payload.get("schema_version")) is not int or payload["schema_version"] != 1
                or payload.get("condition") != "koopman"
                or not training._same_native(payload.get("model_config"), addon_conditions.koopman_model_config(model))
                or not training._same_native(payload.get("loss_config"), training._loss_config(loss_config, "no_consistency"))):
            raise ValueError("koopman checkpoint condition/model_config/loss_config differs from the current build")
        return dict(schema_version=1, condition="koopman", model_config=payload["model_config"], loss_config=payload["loss_config"])
    if condition in TRANSFORMER_LABELS:
        if (payload.get("schema") != TRANSFORMER_SCHEMA or payload.get("config") != asdict(model.spec)
                or not training._same_native(payload.get("loss_config"),
                                             training._loss_config(loss_config, TRANSFORMER_LABELS[condition]))):
            raise ValueError(f"{condition} checkpoint schema/registered spec/loss_config differs from the current build")
        return dict(schema=TRANSFORMER_SCHEMA, config=payload["config"], loss_config=payload["loss_config"])
    if comparison.is_ac(condition):
        from po_jepa import ac_baseline
        spec = ac_baseline.spec_snapshot(ac_baseline.spec_for(condition))
        if (payload.get("schema") != AC_SCHEMA or payload.get("condition") != condition
                or payload.get("spec") != spec or spec != ac_baseline.spec_snapshot(model.spec)):
            raise ValueError(f"{condition} checkpoint schema/condition/spec differs from ac_baseline.spec_for")
        return dict(schema=AC_SCHEMA, condition=condition, spec=payload["spec"], info=payload.get("info"))
    raise ValueError(f"condition {condition} has no checkpoint")


def current_data_fields(features, step, device):
    """当前树对同一份特征算出的 8 个数据字段：重新组第 max(0, step−1) 批训练批（就是检查点保存时那一批，
    comparison.run_prepared_condition / resume.prepare_condition 的写法），取它的来源。"""
    sequence, _ = learning.make_query_sequence(checks.load_protocol())
    batch = data.load_training_batch(Path(features), sequence[max(0, step-1)], device=str(device))
    return {k: batch.provenance[k] for k in DATA_FIELDS}


def load_v7_selected(job_dir, condition, model, *, loss_config, features, device):
    """10 §4.2：把 job_dir 这个已完成任务的选中检查点读进 model（调用者用 comparison.build_condition 现建），
    返回记进 source.json 的记录。任何一项不符报 ValueError；source_files_sha256 只记录。identity 没有检查点，
    只做第 1 步。"""
    if condition not in D4_CONDITIONS:
        raise ValueError(f"--condition 必须是 {D4_CONDITIONS} 之一")
    job_dir = Path(job_dir).resolve()
    job, report = job_record(job_dir, condition)
    record = dict(job_dir=str(job_dir), job=job, report_sha256=cache.file_sha(job_dir/"report.json"))
    if condition == "identity":
        model.eval()
        return dict(record, training_dir=None, best=None, checkpoint=None)
    from ..workflow import verify_selection
    root = training_dir(job_dir, job, condition).resolve()
    train_report = _complete(root)
    if train_report["best"] != report["best"]:
        raise ValueError("training report best differs from the job report best")
    verify_selection(root, train_report)
    sequence, _ = learning.make_query_sequence(checks.load_protocol())
    if (train_report["completed_updates"] != len(sequence)
            or train_report["query_sequence_sha256"] != comparison.sequence_sha(sequence)):
        raise ValueError("training budget or query sequence differs from the current protocol")
    best = train_report["best"]
    path = (root/best["checkpoint"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("selected checkpoint escaped its training directory")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != best["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA mismatch before loading")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if type(payload) is not dict or payload.get("completed_steps") != best["step"]:
        raise ValueError("checkpoint completed_steps differs from the selected step")
    snapshot = _payload_family(condition, payload, model, loss_config)
    training._check_model_state(payload["model_state"], model)
    provenance = payload["provenance"]
    if not training._same_native(provenance, best["checkpoint_provenance"]):
        raise ValueError("checkpoint provenance differs from the selection record")
    current = current_data_fields(features, best["step"], device)
    found = {k: provenance.get(k) for k in DATA_FIELDS}
    if found != current:
        raise ValueError(f"checkpoint data provenance differs from the current features: "
                         f"{ {k: (found[k], current[k]) for k in DATA_FIELDS if found[k] != current[k]} }")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return dict(record, training_dir=str(root), best=dict(step=best["step"], mean=best["mean"], checkpoint=best["checkpoint"],
                checkpoint_sha256=best["checkpoint_sha256"]), checkpoint=snapshot, data_fields=found,
                source_files_sha256=provenance["source_files_sha256"])
