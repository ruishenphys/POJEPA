"""v7：把一次完整实验拆成互不依赖的小任务（系统 × 种子 × 条件），方便在排队的算力上并行提交。

一个任务 = 一个条件的训练（identity 不训练）+ 用验证集选中的 checkpoint 在测试集上评测 + 收敛图。
同一系统、同一种子的各条件共用同一个预登记（查询序列、源码哈希），汇总时（tools/summarize_v7.py）再逐项核对。
"""
from dataclasses import fields
import hashlib
from pathlib import Path

import torch

from . import cache, checks, comparison, data, environment, evaluation, plots

JOB_CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity",
                  "vjepa2_ac_small", "vjepa2_ac")


def _light_preflight(features, output, device):
    """每个任务都做的轻量检查：GPU 型号/整卡、缓存完整性与系统/条数。完整的梯度与成本检查用 run.py check 单独做一次。"""
    output.mkdir(parents=True, exist_ok=False)
    hardware = checks.device_metadata(device, output) if device.type == "cuda" else dict(device="cpu")
    assets = data.approved_cache_metadata(features)
    return dict(hardware=hardware, cache_identity=assets)


def _model_hash(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _job_dir(path):
    """旧任务目录：run.py 的 --output、其下的 run/ 或 run/job/ 都接受。"""
    if path is None:
        return None
    path = Path(path).resolve()
    for sub in ("run/job", "job", "run"):     # run.py 的输出是 <dir>/run/job/
        if (path/sub/"job.json").exists():
            path = path/sub
            break
    if not (path/"job.json").exists():
        raise ValueError(f"{path} 不是 v7 任务目录（缺 job.json）")
    return path


def run_job(features, output, device, condition, *, resume_from=None, training_run=None):
    """训练（或续训，或复用已完成训练）一个条件并评测。output 必须是新目录。"""
    from .workflow import verify_bundle, verify_selection, read
    from .resume import verify_ledger
    if condition not in JOB_CONDITIONS:
        raise ValueError(f"--condition 必须是 {JOB_CONDITIONS} 之一")
    features, output = Path(features), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = _light_preflight(features, output/"preflight", device)
    config = checks.load_protocol()
    registration, sequence, state = comparison.preregister()
    registration["delivery_manifest_sha256"] = verify_bundle()
    runtime = environment.runtime_signature(device)
    cache._write_json(output/"preregistration.json", registration)
    cache._write_json(output/"runtime.json", runtime)
    job = dict(condition=condition, dataset=data.datasets.active().tag, seed=checks.model_seed(),
               updates=checks.v7_updates(), resume_from=str(resume_from) if resume_from else None,
               training_run=str(training_run) if training_run else None)
    cache._write_json(output/"job.json", job)
    loss = comparison.LossConfig(**{f.name: config["loss"][f.name] for f in fields(comparison.LossConfig)})
    selected, initial_hash, train_report, root = None, None, None, None

    if condition == "identity":
        model, loss = comparison.build_condition("identity", device)
        (output/"training").mkdir()
        train_report = comparison.identity_reference(features, output/"training"/"identity", device, loss, registration)
    else:
        model, loss = comparison.build_condition(condition, device)
        initial_hash = _model_hash(model)
        old = _job_dir(training_run or resume_from)
        if old is not None:
            from .resume import verify_training_source
            if resume_from is not None:
                verify_training_source(old, registration, runtime)
            if read(old/"preregistration.json") != registration:
                raise ValueError("old job preregistration differs from current source/config")
            if read(old/"job.json")["condition"] != condition:
                raise ValueError("old job belongs to another condition")
        source = old/"training"/condition if old is not None else None
        if source is not None and (training_run is not None or (source/"COMPLETE.json").exists()):
            if not (source/"COMPLETE.json").exists():
                raise ValueError("--training-run 的训练没有完成；请用 resume")
            root = source                     # 只评测：复用已完成的训练
            train_report = read(root/"report.json")
        else:
            optimizer = comparison.make_optimizer(model)
            root = output/"training"/condition
            print(f"training {condition}", flush=True)
            train_report = comparison.run_prepared_condition(
                condition, model, optimizer, sequence=sequence, generator_state=state, registration=registration,
                features=features, output=root, device=device, loss_config=loss,
                validation_interval=config["validation"]["interval_updates"],
                **({"resume_from": source} if source is not None and source.exists() else {}))
            del optimizer
        if (root/"FAILED.json").exists() or train_report["completed_updates"] != len(sequence):
            raise ValueError("training incomplete")
        verify_ledger(root, sequence, state, registration, train_report["completed_updates"])
        verify_selection(root, train_report)
        selected = train_report["best"]
        del model
        model, loss = comparison.build_condition(condition, device)
        optimizer = comparison.make_optimizer(model)
        evaluation.load_selected_checkpoint(condition, selected, model=model, optimizer=optimizer, config=loss,
                                            expected_provenance=selected["checkpoint_provenance"], root=root)
        del optimizer

    print(f"evaluating {condition}", flush=True)
    eval_report = evaluation.run_prepared_evaluation(model, condition, features, output/"evaluation"/condition, loss,
                                                     checkpoint_identity=selected, device=device)
    figures = plots.job_figures(output, condition, root)
    report = dict(job=job, preflight=receipt, initial_model_sha256=initial_hash,
                  registration_sha256=hashlib.sha256(cache._json_bytes(registration)).hexdigest(),
                  best=selected, training=dict(completed_updates=train_report.get("completed_updates"),
                                               consistency_monitor=train_report.get("consistency_monitor")),
                  evaluation=dict(status=eval_report["status"], counts=eval_report["counts"]),
                  figures=figures, scientific_conclusions_verified=False)
    cache._write_json(output/"report.json", report)
    cache._write_json(output/"COMPLETE.json", dict(report_sha256=cache.file_sha(output/"report.json")))
    return report
