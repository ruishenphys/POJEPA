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
                  "vjepa2_ac_small", "vjepa2_ac", *comparison.ADDON_CONDITIONS)   # 07 §2.6：补跑的 6 个条件（S27a 加了 po_jepa_rollout_sup）


def _light_preflight(features, output, device):
    """每个任务都做的轻量检查：GPU 型号/整卡、缓存完整性与系统/条数。完整的梯度与成本检查用 run.py check 单独做一次。"""
    output.mkdir(parents=True, exist_ok=False)
    hardware = checks.device_metadata(device, output) if device.type == "cuda" else dict(device="cpu")
    assets = data.approved_cache_metadata(features)
    return dict(hardware=hardware, cache_identity=assets)


def with_variant(job: dict) -> dict:
    """S28b：非默认变体（多段切分、零初始化）记进 job.json 的 variant 键，汇总按它分组；默认变体的 job.json 逐键不变。"""
    variant = checks.v7_variant()
    return job if variant == checks.V7_VARIANT_DEFAULT else dict(job, variant=variant)


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
    job = with_variant(job)
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
        if (root/"FAILED.json").exists() or not (root/"COMPLETE.json").exists():
            raise ValueError("training incomplete")
        # 跑满预算，或按预登记的早停规则恰好在第一个触发点停下（规则从验证序列重算，不信报告里的声明）
        if comparison.verify_stopping(train_report["validations"], train_report["completed_updates"], len(sequence),
                                      registration.get("early_stopping")) != train_report.get("early_stopping"):
            raise ValueError("training early-stopping record differs from the preregistered rule")
        verify_ledger(root, sequence, state, registration, train_report["completed_updates"], final=True)
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
                                               early_stopping=train_report.get("early_stopping"),
                                               checkpoint_retention=train_report.get("checkpoint_retention"),
                                               consistency_monitor=train_report.get("consistency_monitor")),
                  evaluation=dict(status=eval_report["status"], counts=eval_report["counts"]),
                  figures=figures, scientific_conclusions_verified=False)
    cache._write_json(output/"report.json", report)
    cache._write_json(output/"COMPLETE.json", dict(report_sha256=cache.file_sha(output/"report.json")))
    return report


def _linked_training_root(job, job0, condition, registration, read):
    """job 自己的 training/<条件> 没完成时（续训任务复用了旧训练、只评测任务指向旧任务），沿 job.json 的
    training_run / resume_from 找完成的训练，最多跟 8 层，和 addon.v7load.training_dir 同一走法（S28c-fix1）。
    每一层的条件和预登记都要和当前一致。"""
    folder, record = job, job0
    for _ in range(9):
        root = folder/"training"/condition
        if (root/"COMPLETE.json").exists():
            return root
        link = record.get("training_run") or record.get("resume_from")
        if not link:
            return root                       # 没有完成的训练：由调用方按原来的方式报错
        folder = _job_dir(link)
        record = read(folder/"job.json")
        if record.get("condition") != condition:
            raise ValueError(f"{folder} 是另一个条件的任务")
        if read(folder/"preregistration.json") != registration:
            raise ValueError(f"{folder} 的预登记与当前代码/配置不一致")
    raise ValueError(f"{job} 的 training_run / resume_from 链超过 8 层")


def load_for_evaluation(condition, training_run, device):
    """只评测不训练的任务（transfer、S25c 长时程滚动）共用的载入：按当前数据集（训练用的基础场景）现建模型，
    核 training_run 的条件/数据集/种子和预登记（源码、配置、查询序列）与当前一致，载入验证集选中的 checkpoint。
    identity 没有 checkpoint。返回 (model, loss, selected, registration)；selected 是 None 或训练报告的 best。"""
    from .workflow import verify_bundle, verify_selection, read
    base = data.datasets.active()
    registration, _, _ = comparison.preregister()
    registration["delivery_manifest_sha256"] = verify_bundle()
    selected = None
    model, loss = comparison.build_condition(condition, device)
    if condition != "identity":
        old = _job_dir(training_run)
        job0 = read(old/"job.json")
        if job0["condition"] != condition or job0["dataset"] != base.tag or job0["seed"] != checks.model_seed():
            raise ValueError("training run 的条件/数据集/种子与本次参数不一致")
        if read(old/"preregistration.json") != registration:
            raise ValueError("training run 的预登记与当前代码/配置不一致")
        root = _linked_training_root(old, job0, condition, registration, read)
        train_report = read(root/"report.json")
        verify_selection(root, train_report)
        selected = train_report["best"]
        optimizer = comparison.make_optimizer(model)
        evaluation.load_selected_checkpoint(condition, selected, model=model, optimizer=optimizer, config=loss,
                                            expected_provenance=selected["checkpoint_provenance"], root=root)
        del optimizer
    return model, loss, selected, registration


def transfer_job(features, output, device, condition, *, training_run, test_variant):
    """v7 可调数据集的分布外评测：用基础场景上训练好的模型（training_run 是它的任务目录），
    不再训练，直接在“换了调节项”的测试集上评测（features 是 encode_v7.py --variant … --splits test 的输出）。

    测试集与基础场景的测试段编号一一对应（同初始状态、同下达动作，除非调节项改的就是动作抽样），所以可以逐条配对。
    """
    from .workflow import verify_bundle, verify_selection, read
    from po_jepa import variants
    if condition not in JOB_CONDITIONS:
        raise ValueError(f"--condition 必须是 {JOB_CONDITIONS} 之一")
    base = data.datasets.active()
    # 测试调节项叠加在训练数据集的调节项上（同名的以测试为准），例如训练 landscape~shape=multi_well、测试 depth=2
    merged = dict(item.split("=", 1) for item in base.variant.split(",") if item)
    merged.update(item.split("=", 1) for item in test_variant.split(",") if item)
    text = ",".join(f"{k}={v}" for k, v in merged.items())
    test_spec = data.datasets.DatasetSpec(base.system, base.count,
                                          variants.format(base.system, variants.parse(base.system, text)))
    if test_spec == base:
        raise ValueError("--test-variant 与训练用的数据集相同；这不是分布外评测")
    if test_spec.knobs.horizon != base.knobs.horizon:
        # S25c：transfer 只评起点 ≤ 20、时长 ≤ 12 的窗口，长时程缓存的前 33 帧和基础测试集相同，评出来一样；长时程用 tools/long_rollout.py
        raise ValueError("--test-variant 不能改 horizon：长时程测试集用 tools/long_rollout.py 评测")
    if test_spec.knobs.changes_timing():
        # S25d：时间采样测试集的帧数、帧时刻、动作边界不是 33 帧缓存（cache.read_window）的格式；按帧评测用 tools/timing_eval.py
        raise ValueError("--test-variant 不能带时间采样调节项（fps/jitter/exposure/switch_offset）：时间采样测试集用 tools/timing_eval.py 评测")
    if test_spec.knobs.changes_space():
        # S25e：空间采样测试集的特征网格是 (R/16)×(R/16)、预处理是原生分辨率版，不是 cache.read_window 的格式；评测用 tools/space_eval.py
        raise ValueError("--test-variant 不能带空间采样调节项（resolution/zoom/pan/drift）：空间采样测试集用 tools/space_eval.py 评测")
    features, output = Path(features), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, loss, selected, registration = load_for_evaluation(condition, training_run, device)
    job = dict(kind="transfer", condition=condition, dataset=test_spec.tag, train_dataset=base.tag,
               test_variant=test_spec.variant, seed=checks.model_seed(), updates=checks.v7_updates(),
               training_run=str(training_run) if training_run else None)
    job = with_variant(job)
    cache._write_json(output/"job.json", job)
    old_spec = data.datasets._set_for_tests(test_spec)
    try:
        receipt = _light_preflight_test(features, output/"preflight", device)
        print(f"transfer-evaluating {condition}: {base.tag} → {test_spec.tag}", flush=True)
        # 分布外评测只看 16×16 网格上的单点预测（6 个时长，含留出时长 3/12）：问题是换物理/控制/外观，不是换网格；成本约为完整评测的 1/5
        queries = [q for q in evaluation.build_index() if q.kind == "point" and q.grid_size == 16]
        eval_report = evaluation.run_prepared_evaluation(model, condition, features, output/"evaluation"/condition, loss,
                                                         queries=queries, checkpoint_identity=selected, device=device)
    finally:
        data.datasets._set_for_tests(old_spec)
    report = dict(job=job, preflight=receipt, best=selected,
                  registration_sha256=hashlib.sha256(cache._json_bytes(registration)).hexdigest(),
                  evaluation=dict(status=eval_report["status"], counts=eval_report["counts"]),
                  training=dict(), initial_model_sha256=None, figures={}, scientific_conclusions_verified=False)
    cache._write_json(output/"report.json", report)
    cache._write_json(output/"COMPLETE.json", dict(report_sha256=cache.file_sha(output/"report.json")))
    return report


def _light_preflight_test(features, output, device):
    output.mkdir(parents=True, exist_ok=False)
    hardware = checks.device_metadata(device, output) if device.type == "cuda" else dict(device="cpu")
    return dict(hardware=hardware, cache_identity=data.approved_cache_metadata(features, splits=("test",)))


# v7.1（Jerry 2026-11-10）：早停后各条件训练步数不同，另做“同步数对照”作补充：
# 把对照条件截到 po_jepa_full 实际训练的步数 s，在 ≤ s 的验证点里按同一选点规则取最好，再评测同一套测试查询。
# 只对检查点全部保留的小模型做（AC 的检查点按 checkpoints.prune_conditions 精简掉了）。主表仍是各自早停后的最好点。
# full-model：补跑条件（transformer_consistency 等，07 §2.6、08 §6）也是检查点全留的小模型，同样可以截到 po_jepa_full 的步数；
# 任务表（make_v7_jobs 的 D2 段）仍只排前两个，和 v7.1 相同。
EQUAL_STEP_CONDITIONS = ("matched_transformer", "po_jepa_no_consistency", *comparison.ADDON_CONDITIONS)


def equal_step_selection(validations, limit):
    """≤ limit 步的验证点里按同一规则（均值最小、平局取最早）选；limit 必须正好是某个验证点。"""
    if limit not in {r["step"] for r in validations}:
        raise ValueError("equal-step limit must be one of the reference run's validation steps")
    return comparison.best_validation([r for r in validations if r["step"] <= limit])


def _finished_training(job, condition, registration, sequence, state):
    """读一个已完成任务的训练报告，核预登记、完成标记、选点和早停规则；返回 (训练目录, 报告)。"""
    from .workflow import verify_selection, read
    from .resume import verify_ledger
    job0 = read(job/"job.json")
    if job0["condition"] != condition or job0["dataset"] != data.datasets.active().tag or job0["seed"] != checks.model_seed():
        raise ValueError(f"{job} 的条件/数据集/种子与本次参数不一致")
    if read(job/"preregistration.json") != registration or not (job/"COMPLETE.json").exists():
        raise ValueError(f"{job} 没有完成，或预登记与当前代码/配置不一致")
    root = _linked_training_root(job, job0, condition, registration, read)
    report = read(root/"report.json")
    if not (root/"COMPLETE.json").exists() or (root/"FAILED.json").exists():
        raise ValueError(f"{root} 训练没有完成")
    if comparison.verify_stopping(report["validations"], report["completed_updates"], len(sequence),
                                  registration.get("early_stopping")) != report.get("early_stopping"):
        raise ValueError(f"{root} 的早停记录与预登记规则不一致")
    verify_ledger(root, sequence, state, registration, report["completed_updates"], final=True)
    verify_selection(root, report)
    return root, report


def equal_step_job(features, output, device, condition, *, training_run, limit_run):
    """同步数对照：condition 截到 limit_run（po_jepa_full）实际训练的步数，取截断内最好验证点，在完整测试查询上评测。"""
    from .workflow import verify_bundle
    if condition not in EQUAL_STEP_CONDITIONS:
        raise ValueError(f"equal-step 只支持 {EQUAL_STEP_CONDITIONS}")
    # G14-fix1（Codex G14 P2）：两边共用同一份当前预登记，缩放不同的任务配不上；不支持缩放，给明确的错
    checks.require_weight_scale_allowed(None, "equal-step（同步数对照，两边用同一份预登记，缩放与未缩放的任务配不上）")
    retention = comparison.checkpoint_retention_config()
    if retention is not None and condition in retention["prune_conditions"]:
        raise ValueError(f"{condition} 的检查点按 checkpoints.prune_conditions 精简过，不能截断选点")
    features, output = Path(features), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = _light_preflight(features, output/"preflight", device)
    registration, sequence, state = comparison.preregister()
    registration["delivery_manifest_sha256"] = verify_bundle()
    _, limit_report = _finished_training(_job_dir(limit_run), "po_jepa_full", registration, sequence, state)
    root, report = _finished_training(_job_dir(training_run), condition, registration, sequence, state)
    limit = limit_report["completed_updates"]
    selected = equal_step_selection(report["validations"], min(limit, report["completed_updates"]))
    job = dict(kind="equal_step", condition=condition, dataset=data.datasets.active().tag, seed=checks.model_seed(),
               updates=checks.v7_updates(), training_run=str(training_run), limit_run=str(limit_run))
    job = with_variant(job)
    cache._write_json(output/"job.json", job)
    model, loss = comparison.build_condition(condition, device)
    optimizer = comparison.make_optimizer(model)
    evaluation.load_selected_checkpoint(condition, selected, model=model, optimizer=optimizer, config=loss,
                                        expected_provenance=selected["checkpoint_provenance"], root=root)
    del optimizer
    print(f"equal-step evaluating {condition}: limit {limit} updates (po_jepa_full), selected step {selected['step']}", flush=True)
    eval_report = evaluation.run_prepared_evaluation(model, condition, features, output/"evaluation"/condition, loss,
                                                     checkpoint_identity=selected, device=device)
    equal = dict(limit_condition="po_jepa_full", limit_updates=limit, selected_step=selected["step"],
                 own_best_step=report["best"]["step"], own_completed_updates=report["completed_updates"],
                 same_as_main_selection=selected["step"] == report["best"]["step"])
    out = dict(job=job, preflight=receipt, best=selected, equal_step=equal,
               registration_sha256=hashlib.sha256(cache._json_bytes(registration)).hexdigest(),
               evaluation=dict(status=eval_report["status"], counts=eval_report["counts"]),
               training=dict(), initial_model_sha256=None, figures={}, scientific_conclusions_verified=False)
    cache._write_json(output/"report.json", out)
    cache._write_json(output/"COMPLETE.json", dict(report_sha256=cache.file_sha(output/"report.json")))
    return out
