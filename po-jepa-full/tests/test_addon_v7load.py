"""S18a：放宽来源核对的检查点读取（10 §4.1–§4.2）。用小模型在临时目录里造一个“已完成任务”，逐项改坏，
每一项都必须拒绝；只有 source_files_sha256 改了仍能读入，并原样记录。真实 v7 检查点的读取在 tests/addon_d4_smoke.py 里核。"""

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from po_jepa import ac_baseline
from po_jepa.runtime import cache, checks, comparison, data, learning, training
from po_jepa.runtime import workflow as w
from po_jepa.runtime.addon import conditions as addon_conditions
from po_jepa.runtime.addon import v7load
from test_addon_conditions import koopman_model, po, transformer
from test_training import e06_config

SEQ = [f"q{i}" for i in range(6)]
FIELDS = dict(evidence_role="real_frozen_encoder", split="train", paper_sha256="p"*64, upstream_commit="u"*40,
              encoder_checkpoint_sha256="e"*64, preprocess_sha256="r"*64, feature_manifest_sha256="f"*64, protocol_sha256="c"*64)
ZERO = e06_config(time_weight=0., space_weight=0.)        # no_consistency 的损失配置：两个权重都是 0
SOURCES = {"src/po_jepa/model.py": "1"*64, "src/po_jepa/runtime/training.py": "2"*64}


@pytest.fixture
def stubs(monkeypatch):
    """查询序列、选点重建和训练批来源换成可控的替身；其余核对都是真实代码。"""
    monkeypatch.setattr(learning, "make_query_sequence", lambda config: (SEQ, None))
    monkeypatch.setattr(comparison, "sequence_sha", lambda sequence: f"sequence-{len(sequence)}")
    seen = dict(selections=[], batches=[], fields=dict(FIELDS))
    monkeypatch.setattr(w, "verify_selection", lambda root, report: seen["selections"].append((Path(root), report["best"]["step"])))
    def batch(features, query, device):
        seen["batches"].append(query)
        return SimpleNamespace(provenance=dict(seen["fields"], source_files_sha256={"other": "x"}))
    monkeypatch.setattr(data, "load_training_batch", batch)
    return seen


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cache._json_bytes(value))


def complete(folder, report):
    write(folder/"report.json", report)
    write(folder/"COMPLETE.json", dict(report_sha256=cache.file_sha(folder/"report.json")))


def po_payload(model, loss, step, label="full"):
    return dict(schema_version=1, completed_steps=step, condition=label, model_state={k: v.detach().clone() for k, v in model.state_dict().items()},
                model_config=training._model_config(model), loss_config=training._loss_config(loss, label),
                provenance=dict(FIELDS, f03_collections=["c"], source_files_sha256=dict(SOURCES)))


def make_job(root, condition, payload, *, step=4, link=None, with_training=True):
    """root/run/job：job.json、report.json、COMPLETE.json 和 training/<条件>（选中 step 的检查点）。"""
    job_dir = root/"run"/"job"
    job = dict(condition=condition, dataset=data.datasets.active().tag, seed=checks.model_seed(), updates=checks.v7_updates(),
               resume_from=None, training_run=link)
    write(job_dir/"job.json", job)
    best = None
    if payload is not None:
        folder = job_dir/"training"/condition if with_training else root/"elsewhere"
        (folder/"checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(payload, folder/"checkpoints"/f"step-{step:04d}.pt")
        best = dict(step=step, mean=.25, checkpoint=f"checkpoints/step-{step:04d}.pt",
                    checkpoint_sha256=cache.file_sha(folder/"checkpoints"/f"step-{step:04d}.pt"),
                    checkpoint_provenance=copy.deepcopy(payload["provenance"]))
        if with_training:
            complete(folder, dict(best=best, completed_updates=len(SEQ), query_sequence_sha256=f"sequence-{len(SEQ)}"))
    complete(job_dir, dict(best=best, job=job))
    return job_dir


def load(job_dir, condition, model, loss):
    return v7load.load_v7_selected(job_dir, condition, model, loss_config=loss, features=Path("features"), device="cpu")


@pytest.fixture
def po_job(tmp_path, stubs):
    trained, loss = po(), e06_config()
    with torch.no_grad():
        for p in trained.parameters():
            p.add_(.01)
    return make_job(tmp_path/"a", "po_jepa_full", po_payload(trained, loss, 4)), trained, loss


def test_loads_selected_state_and_records_source(po_job, stubs):
    job_dir, trained, loss = po_job
    model = po().train()
    record = load(job_dir, "po_jepa_full", model, loss)
    assert model.training is False
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), trained.state_dict().values()))
    assert record["best"] == dict(step=4, mean=.25, checkpoint="checkpoints/step-0004.pt",
                                  checkpoint_sha256=cache.file_sha(job_dir/"training"/"po_jepa_full"/"checkpoints"/"step-0004.pt"))
    assert record["data_fields"] == FIELDS and record["source_files_sha256"] == SOURCES
    assert record["checkpoint"]["condition"] == "full" and record["checkpoint"]["schema_version"] == 1
    assert record["training_dir"] == str((job_dir/"training"/"po_jepa_full").resolve())
    assert record["report_sha256"] == cache.file_sha(job_dir/"report.json") and record["job"]["condition"] == "po_jepa_full"
    assert stubs["selections"] == [((job_dir/"training"/"po_jepa_full").resolve(), 4)]
    assert stubs["batches"] == ["q3"]                                    # 第 max(0, step−1) 批
    json.dumps(record)                                                   # 能写进 source.json


def test_step_zero_uses_first_batch(tmp_path, stubs):
    model, loss = po(), e06_config()
    load(make_job(tmp_path/"z", "po_jepa_full", po_payload(model, loss, 0), step=0), "po_jepa_full", po(), loss)
    assert stubs["batches"] == ["q0"]


def test_source_files_sha_is_recorded_not_checked(tmp_path, stubs):
    model, loss = po(), e06_config()
    payload = po_payload(model, loss, 4)
    payload["provenance"]["source_files_sha256"] = {"src/po_jepa/model.py": "9"*64}
    record = load(make_job(tmp_path/"s", "po_jepa_full", payload), "po_jepa_full", po(), loss)
    assert record["source_files_sha256"] == {"src/po_jepa/model.py": "9"*64}


def rewrite_report(folder, change):
    report = json.loads((folder/"report.json").read_text())
    change(report)
    complete(folder, report)


@pytest.mark.parametrize("case,message", [
    ("job_condition", "job.json differs"), ("job_kind", "job.json differs"), ("job_seed", "job.json differs"),
    ("failed_marker", "FAILED.json"), ("report_edited", "differs from its COMPLETE.json"),
    ("training_failed", "FAILED.json"), ("best_differs", "best differs from the job report"),
    ("budget", "budget or query sequence"), ("sequence", "budget or query sequence"),
    ("escape", "escaped its training directory"), ("bytes", "SHA mismatch before loading"),
])
def test_job_and_file_refusals(po_job, case, message):
    job_dir, trained, loss = po_job
    train = job_dir/"training"/"po_jepa_full"
    if case.startswith("job_"):
        job = json.loads((job_dir/"job.json").read_text())
        job.update(dict(job_condition=dict(condition="po_jepa_no_consistency"), job_kind=dict(kind="transfer"),
                        job_seed=dict(seed=1704))[case])
        write(job_dir/"job.json", job)
    elif case == "failed_marker":
        write(job_dir/"FAILED.json", {})
    elif case == "report_edited":
        (job_dir/"report.json").write_text((job_dir/"report.json").read_text()+" ")
    elif case == "training_failed":
        write(train/"FAILED.json", {})
    elif case == "best_differs":
        rewrite_report(train, lambda r: r["best"].update(mean=.5))
    elif case == "budget":
        rewrite_report(train, lambda r: r.update(completed_updates=5))
    elif case == "sequence":
        rewrite_report(train, lambda r: r.update(query_sequence_sha256="other"))
    elif case == "escape":
        for folder in (train, job_dir):
            rewrite_report(folder, lambda r: r["best"].update(checkpoint="../../../../elsewhere.pt"))
    elif case == "bytes":
        path = train/"checkpoints"/"step-0004.pt"
        path.write_bytes(path.read_bytes()+b"\0")
    with pytest.raises(ValueError, match=message):
        load(job_dir, "po_jepa_full", po(), loss)


def test_selection_rule_failure_propagates(po_job, monkeypatch):
    job_dir, _, loss = po_job
    def bad(root, report):
        raise ValueError("selected checkpoint differs from fixed future-only rule")
    monkeypatch.setattr(w, "verify_selection", bad)
    with pytest.raises(ValueError, match="future-only rule"):
        load(job_dir, "po_jepa_full", po(), loss)


@pytest.mark.parametrize("case,message", [
    ("steps", "completed_steps differs"), ("label", "condition/model_config/loss_config"),
    ("model_config", "condition/model_config/loss_config"), ("loss_config", "condition/model_config/loss_config"),
    ("state_key", "model state keys mismatch"), ("state_shape", "shape/dtype/finite"), ("state_buffer", "fixed buffer mismatch"),
    ("provenance", "provenance differs from the selection record"), ("data_field", "data provenance differs"),
])
def test_payload_refusals(tmp_path, stubs, case, message):
    model, loss = po(), e06_config()
    payload = po_payload(model, loss, 4)
    state = payload["model_state"]
    buffer = next(n for n, _ in model.named_buffers())
    parameter = next(n for n, _ in model.named_parameters())
    if case == "steps":
        payload["completed_steps"] = 3
    elif case == "label":
        payload["condition"] = "no_consistency"
    elif case == "model_config":
        payload["model_config"] = training._model_config(po(max_modes=4))
    elif case == "loss_config":
        payload["loss_config"] = training._loss_config(e06_config(time_weight=2e-3), "full")
    elif case == "state_key":
        state["renamed"] = state.pop(parameter)
    elif case == "state_shape":
        state[parameter] = torch.zeros(1)
    elif case == "state_buffer":
        state[buffer] = state[buffer]+1
    job_dir = make_job(tmp_path/case, "po_jepa_full", payload)
    if case == "provenance":
        payload["provenance"]["f03_collections"] = ["other"]
        torch.save(payload, job_dir/"training"/"po_jepa_full"/"checkpoints"/"step-0004.pt")
        sha = cache.file_sha(job_dir/"training"/"po_jepa_full"/"checkpoints"/"step-0004.pt")
        for folder in (job_dir/"training"/"po_jepa_full", job_dir):
            rewrite_report(folder, lambda r: r["best"].update(checkpoint_sha256=sha))
    if case == "data_field":
        stubs["fields"]["feature_manifest_sha256"] = "g"*64
    with pytest.raises(ValueError, match=message):
        load(job_dir, "po_jepa_full", po(), loss)


def test_linked_training_directory(tmp_path, stubs):
    """续训或只评测的任务：自己没有 training/<条件>，沿 training_run 找到旧任务；条件不同或没有链接都拒绝。"""
    model, loss = po(), e06_config()
    old = make_job(tmp_path/"old", "po_jepa_full", po_payload(model, loss, 4))
    new = make_job(tmp_path/"new", "po_jepa_full", None, link=str(tmp_path/"old"))
    rewrite_report(new, lambda r: r.update(best=json.loads((old/"report.json").read_text())["best"]))
    record = load(new, "po_jepa_full", po(), loss)
    assert record["training_dir"] == str((old/"training"/"po_jepa_full").resolve())
    other = make_job(tmp_path/"other", "po_jepa_no_consistency", po_payload(model, ZERO, 4, "no_consistency"))
    wrong = make_job(tmp_path/"wrong", "po_jepa_full", None, link=str(tmp_path/"other"))
    with pytest.raises(ValueError, match="another condition"):
        load(wrong, "po_jepa_full", po(), loss)
    lonely = make_job(tmp_path/"lonely", "po_jepa_full", None)
    with pytest.raises(ValueError, match="no completed training"):
        load(lonely, "po_jepa_full", po(), loss)
    assert other.exists()


def test_identity_reads_job_only(tmp_path, stubs):
    from po_jepa.baselines import IdentityPredictor
    job_dir = make_job(tmp_path/"i", "identity", None)
    model = IdentityPredictor(3).train()
    record = load(job_dir, "identity", model, e06_config())
    assert record["training_dir"] is None and record["best"] is None and model.training is False
    assert stubs["selections"] == [] and stubs["batches"] == []


def test_transformer_and_koopman_families(tmp_path, stubs):
    loss = ZERO
    t = transformer()
    payload = dict(schema=v7load.TRANSFORMER_SCHEMA, config=asdict(t.spec), loss_config=training._loss_config(ZERO, "no_consistency"),
                   completed_steps=4, model_state=dict(t.state_dict()), provenance=dict(FIELDS, source_files_sha256=SOURCES))
    record = load(make_job(tmp_path/"t", "matched_transformer", payload), "matched_transformer", transformer(), loss)
    assert record["checkpoint"]["schema"] == v7load.TRANSFORMER_SCHEMA
    with pytest.raises(ValueError, match="schema/registered spec/loss_config"):
        load(make_job(tmp_path/"tc", "transformer_consistency", payload), "transformer_consistency", transformer(), e06_config())
    k = koopman_model()
    payload = dict(schema_version=1, condition="koopman", completed_steps=4, model_state=dict(k.state_dict()),
                   model_config=addon_conditions.koopman_model_config(k), loss_config=training._loss_config(loss, "no_consistency"),
                   provenance=dict(FIELDS, source_files_sha256=SOURCES))
    assert load(make_job(tmp_path/"k", "koopman", payload), "koopman", koopman_model(), loss)["checkpoint"]["condition"] == "koopman"
    bad = dict(payload, loss_config=training._loss_config(e06_config(), "full"))
    with pytest.raises(ValueError, match="koopman checkpoint"):
        load(make_job(tmp_path/"kb", "koopman", bad), "koopman", koopman_model(), loss)


def test_ac_family_snapshot():
    """AC 只核快照（不建模型）：schema、条件、spec 等于 ac_baseline.spec_for 的快照。
    S08b：v7 检查点里的 spec 是加 state_input 字段之前的 asdict，等于现在的 `spec_snapshot`（09 §6.2）；
    带上 state_input 的快照不是 v7 写的，也要拒。"""
    spec = ac_baseline.spec_for("vjepa2_ac_small")
    model = SimpleNamespace(spec=spec)
    snapshot = ac_baseline.spec_snapshot(spec)
    assert snapshot == dict(encoder_dim=1408, width=64, depth=1, heads=4, mlp_ratio=3.5, action_dim=2)
    payload = dict(schema=v7load.AC_SCHEMA, condition="vjepa2_ac_small", spec=snapshot, info=dict(x=1))
    assert v7load._payload_family("vjepa2_ac_small", payload, model, e06_config()) == dict(
        schema=v7load.AC_SCHEMA, condition="vjepa2_ac_small", spec=snapshot, info=dict(x=1))
    for change in (dict(schema="other"), dict(condition="vjepa2_ac"), dict(spec=dict(snapshot, depth=spec.depth+1)),
                   dict(spec=asdict(spec))):
        with pytest.raises(ValueError, match="schema/condition/spec"):
            v7load._payload_family("vjepa2_ac_small", dict(payload, **change), model, e06_config())


def test_condition_list_and_data_fields():
    assert v7load.D4_CONDITIONS == ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity", "vjepa2_ac_small",
                                    "vjepa2_ac", "po_jepa_time_only", "po_jepa_space_only", "transformer_consistency",
                                    "fixed_fourier", "koopman", "po_jepa_rollout_sup")
    assert v7load.PO_LABELS["po_jepa_rollout_sup"] == addon_conditions.LOSS_LABELS["po_jepa_rollout_sup"] == "rollout_sup"   # 08 §6
    assert v7load.DATA_FIELDS == ("evidence_role", "split", "paper_sha256", "upstream_commit", "encoder_checkpoint_sha256",
                                  "preprocess_sha256", "feature_manifest_sha256", "protocol_sha256")
    with pytest.raises(ValueError, match="--condition"):
        v7load.load_v7_selected(Path("x"), "po_jepa_tiny", None, loss_config=None, features=Path("f"), device="cpu")
    assert hashlib.sha256(b"").hexdigest() not in v7load.DATA_FIELDS
    with pytest.raises(ValueError, match="identity has no checkpoint"):
        v7load._payload_family("identity", {}, None, None)            # identity 在第 1 步后就返回，这里是最后一道
