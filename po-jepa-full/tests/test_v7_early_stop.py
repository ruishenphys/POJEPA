"""v7.1：早停、检查点精简、分网格验证、多步滚动评测、同步数对照、任务表（全部 CPU 虚构输入）。"""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from test_evaluation import fictional_window

ROOT = Path(__file__).resolve().parents[1]
RULE = dict(min_updates=6, patience_validations=2)


@pytest.fixture
def api():
    return importlib.import_module("po_jepa.runtime.comparison")


def rows(means, interval=2):
    return [dict(step=i*interval, mean=m, checkpoint=f"checkpoints/step-{i*interval:04d}.pt") for i, m in enumerate(means)]


# —— 早停规则（纯函数）——

def test_decision_counts_validations_since_strict_best(api):
    v = rows([1., .5, .4, .4, .45])              # 第 6 步与第 4 步并列：不算刷新（严格更小才换）
    d = api.early_stop_decision(v, 100, RULE)
    assert d == dict(step=8, best_step=4, validations_since_best=2, stop=True)
    assert not api.early_stop_decision(v[:4], 100, RULE)["stop"]          # 只过了 1 次
    assert not api.early_stop_decision(v, 8, RULE)["stop"]                # 已到预算：不叫早停
    assert not api.early_stop_decision(v, 100, None)["stop"]              # v6：没有规则
    assert api.early_stop_decision(rows([1., .5, .6, .7]), 100, RULE)["stop"]      # 恰好在 min_updates=6 那次验证就可以停
    assert not api.early_stop_decision(rows([1., .9, .8, .9, .9]), 100, dict(min_updates=10, patience_validations=2))["stop"]


def test_verify_stopping_exact_first_trigger(api):
    v = rows([1., .5, .4, .45, .46, .3])
    assert api.verify_stopping(v[:5], 8, 100, RULE)["stopped_early"]
    with pytest.raises(ValueError, match="continued past"):
        api.verify_stopping(v, 10, 100, RULE)
    with pytest.raises(ValueError, match="before its budget"):
        api.verify_stopping(v[:3], 4, 100, RULE)
    full = api.verify_stopping(rows([1., .9, .8]), 4, 4, RULE)
    assert not full["stopped_early"] and full["completed_updates"] == 4
    with pytest.raises(ValueError, match="last validation"):
        api.verify_stopping(v[:5], 9, 100, RULE)


def test_checkpoint_retention_only_for_listed_conditions(api):
    keep = dict(keep_neighbors=2, prune_conditions=["vjepa2_ac"])
    v = rows([1., .9, .8, .3, .7, .6, .65, .66, .67])
    assert api.kept_checkpoint_steps(v, keep, "vjepa2_ac") == {2, 4, 6, 8, 10, 16}
    assert api.kept_checkpoint_steps(v, keep, "matched_transformer") == {r["step"] for r in v}
    assert api.kept_checkpoint_steps(v, None, "vjepa2_ac") == {r["step"] for r in v}
    assert api.kept_checkpoint_steps(rows([1., .5]), dict(keep_neighbors=0, prune_conditions=["vjepa2_ac"]), "vjepa2_ac") == {2}
    n = api.selection_neighbors(v, keep)
    assert [r["step"] for r in n] == [2, 4, 6, 8, 10] and [r["selected"] for r in n] == [False, False, True, False, False]


def test_v7_schedule_registers_rules(monkeypatch):
    from po_jepa.runtime import checks
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    monkeypatch.setenv(checks.UPDATES_ENV, "10000")
    api = importlib.import_module("po_jepa.runtime.comparison")
    assert api.early_stopping_config() == dict(min_updates=5000, patience_validations=4)
    assert api.checkpoint_retention_config() == dict(keep_neighbors=2, prune_conditions=["vjepa2_ac_small", "vjepa2_ac"])
    monkeypatch.delenv(checks.PROTOCOL_ENV)
    assert api.early_stopping_config() is None and api.checkpoint_retention_config() is None


# —— 训练循环 ——

@pytest.fixture
def stop_case(api, monkeypatch, tmp_path):
    from test_comparison import case as _case   # noqa: F401  （夹具源）
    from test_training import e06_case, e06_config
    monkeypatch.setitem(sys.modules, "e06_training_script", api.training)
    batch = e06_case()[1]
    model = torch.nn.Module(); model.p_out = torch.nn.Linear(1, 1)
    monkeypatch.setattr(api.data, "load_training_batch", lambda *a, **k: batch)
    monkeypatch.setattr(api.data, "write_collections", lambda *a, **k: {})
    sequence = [[api.data.WindowQuery(0, n % 21, 1.)] for n in range(20)]
    state = torch.Generator().manual_seed(2703).get_state()
    retention = dict(keep_neighbors=1, prune_conditions=["matched_transformer"])
    monkeypatch.setattr(api, "early_stopping_config", lambda: dict(RULE))
    monkeypatch.setattr(api, "checkpoint_retention_config", lambda: dict(retention))
    registration = dict(parameter_match=api.parameter_match(2, 2), query_sequence_sha256=api.sequence_sha(sequence),
        comparison_config_sha256=api.data.cache.file_sha(api.ROOT/"configs/train/comparison.yaml"),
        consumer_sha256=api.consumer_sources(), early_stopping=dict(RULE), checkpoint_retention=dict(retention))
    calls = dict(updates=[])
    def step(condition, model, optimizer, batch, config, completed):
        calls["updates"].append(completed)
        model.p_out(torch.ones(1, 1))
        return api.training.TrainingStepResult(completed+1, dict(jepa=1., time=None, space=None, weighted_time=0., weighted_space=0., total=1.), .1)
    monkeypatch.setattr(api, "condition_step", step)
    means = iter([1., .5, .4, .45, .46, .3, .2, .1, .1, .1, .1])
    monkeypatch.setattr(api, "validate_condition", lambda *a: dict(mean=next(means), windows=1, metric="future_jepa_distance", means={}))
    def save(condition, path, *a):
        path.write_bytes(f"placeholder {path.name}".encode())
        return api.data.cache.file_sha(path)
    monkeypatch.setattr(api, "save_checkpoint", save)
    kwargs = dict(sequence=sequence, generator_state=state, registration=registration, features=tmp_path/"f",
                  output=tmp_path/"run", device=torch.device("cpu"), loss_config=e06_config(), validation_interval=2)
    return model, kwargs, calls


def test_loop_stops_at_first_trigger_and_prunes(api, stop_case):
    model, kwargs, calls = stop_case
    report = api.run_prepared_condition("matched_transformer", model, object(), **kwargs)
    out = kwargs["output"]
    assert report["completed_updates"] == 8 and len(calls["updates"]) == 8          # 第 8 步验证后停，不再训练
    assert report["early_stopping"]["stopped_early"] and report["best"]["step"] == 4
    assert sorted(p.name for p in (out/"updates").glob("*.json")) == [f"{n:04d}.json" for n in range(1, 9)]
    assert sorted(p.name for p in (out/"checkpoints").glob("*.pt")) == ["step-0002.pt", "step-0004.pt", "step-0006.pt", "step-0008.pt"]
    assert len(list((out/"validations").glob("selection-*.json"))) == 5             # 选点记录全留
    assert report["checkpoint_retention"]["kept_steps"] == [2, 4, 6, 8]
    assert api.verify_stopping(report["validations"], 8, 20, RULE) == report["early_stopping"]


def test_loop_without_rule_runs_full_budget(api, stop_case, monkeypatch):
    model, kwargs, calls = stop_case
    monkeypatch.setattr(api, "early_stopping_config", lambda: None)
    monkeypatch.setattr(api, "checkpoint_retention_config", lambda: None)
    kwargs["registration"].update(early_stopping=None, checkpoint_retention=None)
    report = api.run_prepared_condition("matched_transformer", model, object(), **kwargs)
    assert report["completed_updates"] == 20 and not report["early_stopping"]["stopped_early"]
    assert len(list((kwargs["output"]/"checkpoints").glob("*.pt"))) == 11


def test_loop_refuses_rules_that_differ_from_registration(api, stop_case, monkeypatch):
    from po_jepa.runtime import checks
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    model, kwargs, calls = stop_case
    kwargs["registration"]["early_stopping"] = dict(min_updates=6, patience_validations=3)
    kwargs["registration"]["consumer_sha256"] = api.consumer_sources()   # v7 下多登记几个源码文件
    with pytest.raises(ValueError, match="differ from preregistration"):
        api.run_prepared_condition("matched_transformer", model, object(), **kwargs)
    assert not calls["updates"]


def test_retention_keeps_future_best_neighbours(api):
    """v7-P2-1：训练中保留最好点 ±k 加最近 k 个；以后最好点后移时，它前面的 k 个邻居都还在。"""
    keep = dict(keep_neighbors=2, prune_conditions=["vjepa2_ac"])
    v = rows([1.0, 1.1, 1.2, 1.3, 1.4, .5])
    assert api.kept_checkpoint_steps(v[:5], keep, "vjepa2_ac", final=False) == {0, 2, 4, 6, 8}
    assert api.kept_checkpoint_steps(v[:5], keep, "vjepa2_ac") == {0, 2, 4, 8}              # 训练结束才压缩
    on_disk = None
    for n in range(1, len(v)+1):
        kept = api.kept_checkpoint_steps(v[:n], keep, "vjepa2_ac", final=False)
        assert on_disk is None or kept <= on_disk | {v[n-1]["step"]}                         # 只要求以前留下的或刚写的
        on_disk = kept
    assert api.kept_checkpoint_steps(v, keep, "vjepa2_ac") == {6, 8, 10}
    assert api.kept_checkpoint_steps(rows([1., .5]), dict(keep_neighbors=0, prune_conditions=["vjepa2_ac"]), "vjepa2_ac", final=False) == {2}


def test_loop_pruning_never_drops_a_needed_checkpoint_and_resume_prefix(api, stop_case, monkeypatch):
    """真实循环（写检查点→验证→精简）：最好点在第 10 步后移，中断后 resume._prefix 要求的检查点都在（v7-P2-1）。"""
    from po_jepa.runtime import resume
    model, kwargs, calls = stop_case
    retention = dict(keep_neighbors=2, prune_conditions=["matched_transformer"])
    monkeypatch.setattr(api, "checkpoint_retention_config", lambda: dict(retention))
    monkeypatch.setattr(api, "early_stopping_config", lambda: None)
    kwargs["registration"].update(early_stopping=None, checkpoint_retention=dict(retention))
    means = iter([1.0, 1.1, 1.2, 1.3, 1.4, .5, .6, .7, .8, .9, 1.])
    monkeypatch.setattr(api, "validate_condition", lambda *a: dict(mean=next(means), windows=1, metric="future_jepa_distance", means={}))
    real_prune, seen = api.prune_checkpoints, []
    def checked_prune(output, validations, retention, condition, final=False):
        kept = real_prune(output, validations, retention, condition, final)
        present = {int(p.stem[5:]) for p in (Path(output)/"checkpoints").glob("*.pt")}
        assert set(kept) <= present, f"kept {kept} but only {sorted(present)} on disk"
        seen.append((validations[-1]["step"], final, sorted(present)))
        return kept
    monkeypatch.setattr(api, "prune_checkpoints", checked_prune)
    original = api.condition_step
    def interrupt(condition, model, optimizer, batch, config, completed):
        if completed == 10:
            raise RuntimeError("interrupted after the step-10 validation")
        return original(condition, model, optimizer, batch, config, completed)
    monkeypatch.setattr(api, "condition_step", interrupt)
    kwargs["output"] = kwargs["output"].parent/"training"/"matched_transformer"   # 续训按目录名认条件（training/<条件>）
    with pytest.raises(RuntimeError, match="interrupted"):
        api.run_prepared_condition("matched_transformer", model, object(), **kwargs)
    out = kwargs["output"]
    assert sorted(p.name for p in (out/"checkpoints").glob("*.pt")) == [f"step-{s:04d}.pt" for s in (6, 8, 10)]
    from po_jepa.runtime import workflow
    monkeypatch.setattr(workflow, "verify_selection", lambda *a: None)   # 选点重算要真实验证明细，这里只核检查点保留集
    step, selections, best = resume._prefix(out, kwargs["sequence"], kwargs["generator_state"], kwargs["registration"], 2)
    assert step == 10 and best["step"] == 10
    assert (8, False, [0, 2, 4, 6, 8]) in seen                                                 # 第 8 步还留着 6（旧规则只留 8）


def test_resume_prefix_after_final_compression(api, stop_case, monkeypatch):
    """跑满预算、已压缩成最终保留集、写报告前中断：续训按最终保留集核，不要求训练中才留的检查点（v7-P2-1）。"""
    from po_jepa.runtime import resume, workflow
    model, kwargs, calls = stop_case
    retention = dict(keep_neighbors=2, prune_conditions=["matched_transformer"])
    monkeypatch.setattr(api, "checkpoint_retention_config", lambda: dict(retention))
    monkeypatch.setattr(api, "early_stopping_config", lambda: None)
    kwargs["registration"].update(early_stopping=None, checkpoint_retention=dict(retention))
    kwargs["output"] = kwargs["output"].parent/"training"/"matched_transformer"
    means = iter([1.0, 1.1, 1.2, 1.3, 1.4, .5, .6, .7, .8, .9, 1.])
    monkeypatch.setattr(api, "validate_condition", lambda *a: dict(mean=next(means), windows=1, metric="future_jepa_distance", means={}))
    api.run_prepared_condition("matched_transformer", model, object(), **kwargs)
    out = kwargs["output"]
    for name in ("COMPLETE.json", "report.json"):
        (out/name).unlink()
    assert not (out/"checkpoints"/"step-0018.pt").exists()                                    # 训练中会留的最近第 2 个已压缩掉
    monkeypatch.setattr(workflow, "verify_selection", lambda *a: None)
    step, _, best = resume._prefix(out, kwargs["sequence"], kwargs["generator_state"], kwargs["registration"], 2)
    assert step == 20 and best["step"] == 10


def test_loop_final_prune_compresses_after_budget(api, stop_case, monkeypatch):
    model, kwargs, calls = stop_case
    retention = dict(keep_neighbors=2, prune_conditions=["matched_transformer"])
    monkeypatch.setattr(api, "checkpoint_retention_config", lambda: dict(retention))
    monkeypatch.setattr(api, "early_stopping_config", lambda: None)
    kwargs["registration"].update(early_stopping=None, checkpoint_retention=dict(retention))
    means = iter([1.0, 1.1, 1.2, 1.3, 1.4, .5, .6, .7, .8, .9, 1.])
    monkeypatch.setattr(api, "validate_condition", lambda *a: dict(mean=next(means), windows=1, metric="future_jepa_distance", means={}))
    report = api.run_prepared_condition("matched_transformer", model, object(), **kwargs)
    out = kwargs["output"]
    assert report["completed_updates"] == 20 and report["best"]["step"] == 10
    assert sorted(p.name for p in (out/"checkpoints").glob("*.pt")) == [f"step-{s:04d}.pt" for s in (6, 8, 10, 12, 14, 20)]
    assert report["checkpoint_retention"]["kept_steps"] == [6, 8, 10, 12, 14, 20]


# —— 分网格验证、多步滚动 ——

class Scale(torch.nn.Module):
    """虚构预测器：输出 = 起点 × (1 + 时长/10)，记录每次调用。"""
    def __init__(self):
        super().__init__(); self.w = torch.nn.Parameter(torch.ones(())); self.calls = []
    def forward(self, raw, controls, duration, grid):
        self.calls.append((float(duration[0]), (grid.height, grid.width), controls.u_values[0, controls.segment_valid[0], 0].tolist()))
        return raw*(1+duration.to(raw.dtype).reshape(-1, 1, 1)/10)*self.w


def test_coarse_future_matches_spatial_view(api):
    ev = importlib.import_module("po_jepa.runtime.evaluation")
    from test_training import e06_config
    w = fictional_window(ev, 180, 0, 2.)
    model = Scale().eval()
    got = api.coarse_future(model, w.inputs.raw_z, w.inputs.controls, w.inputs.duration, w.inputs.grid, w.target.raw_z, e06_config())
    for size in (12, 8):
        raw, target, grid = ev.spatial_view(w, size)
        expected = api.jepa_distance(model(raw, w.inputs.controls, w.inputs.duration, grid), target, config=e06_config()).per_sample
        assert torch.equal(got[f"future_jepa_distance_grid{size}"], expected.double())   # 名字写死：不能和 GRID_METRICS 一起错
    assert not torch.equal(got["future_jepa_distance_grid12"], got["future_jepa_distance_grid8"])
    assert {c[1] for c in model.calls} == {(12, 12), (8, 8)}


def test_unit_rollout_steps_one_cycle_with_own_actions():
    ev = importlib.import_module("po_jepa.runtime.evaluation")
    c = ev.comparison.data.cache
    metadata = dict(trajectory_id=180, controls=dict(starts_ticks=[0, 8, 16, 24], ends_ticks=[8, 16, 24, 32],
        values=[[.1, 0.], [.2, 0.], [.3, 0.], [.4, 0.]], units="simulation_position_per_second"))
    controls = c._window_controls(metadata, 6, 4., "cpu")
    raw = torch.ones(1, 256, 3)
    model = Scale().eval()
    out = ev.unit_rollout(model, raw, controls, torch.tensor([4.], dtype=torch.float64), c.regular_grid(16, 16))
    assert [m[0] for m in model.calls] == [1., 1., 1., 1.]
    assert [len(m[2]) for m in model.calls] == [1]*4 and [m[2][0] for m in model.calls] == pytest.approx([.1, .1, .2, .2])   # 第 6–10 tick 各自那 1 周期的动作（第 8 tick 换段）
    assert torch.allclose(out, raw*1.1**4)
    with pytest.raises(ValueError):
        ev.unit_rollout(model, raw, controls, torch.tensor([1.5], dtype=torch.float64), c.regular_grid(16, 16))


def test_rollout_metric_only_v7_grid16(monkeypatch):
    from po_jepa.runtime import checks
    ev = importlib.import_module("po_jepa.runtime.evaluation")
    monkeypatch.delenv(checks.PROTOCOL_ENV, raising=False)
    assert "rollout_future_jepa_distance" not in ev.metric_units("point", 16)
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    assert "rollout_future_jepa_distance" in ev.metric_units("point", 16)
    assert "rollout_future_jepa_distance" not in ev.metric_units("point", 12)
    assert "rollout_future_jepa_distance" not in ev.metric_units("pair", 8)


def test_point_metrics_rollout_value(monkeypatch):
    from po_jepa.runtime import checks
    from test_training import e06_config
    ev = importlib.import_module("po_jepa.runtime.evaluation")
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    q = ev.EvaluationQuery(180, 0, 4., 16, "point")
    w, donor = fictional_window(ev, 180, 0, 4.), fictional_window(ev, 181, 0, 4., action=.3)
    model = Scale().eval()
    values, _ = ev.point_metrics(model, w, donor, q, e06_config())
    rolled = w.inputs.raw_z*1.1**4
    expected = float(ev.comparison.jepa_distance(rolled, w.target.raw_z, config=e06_config()).per_sample[0])
    assert values["rollout_future_jepa_distance"] == pytest.approx(expected, rel=1e-5)
    assert values["rollout_future_jepa_distance"] != pytest.approx(values["future_jepa_distance"])


# —— 同步数对照 ——

def test_equal_step_selection():
    jobs = importlib.import_module("po_jepa.runtime.v7jobs")
    v = rows([1., .5, .6, .4, .3])
    assert jobs.equal_step_selection(v, 4)["step"] == 2
    assert jobs.equal_step_selection(v, 8)["step"] == 8
    with pytest.raises(ValueError):
        jobs.equal_step_selection(v, 5)


def test_equal_step_conditions_and_pruned_guard(api, monkeypatch, tmp_path):
    """full-model：补跑条件也能做同步数对照；检查点被精简的条件（AC）不能截断选点，在建目录前就拒绝。"""
    jobs = importlib.import_module("po_jepa.runtime.v7jobs")
    assert set(jobs.EQUAL_STEP_CONDITIONS) == {"matched_transformer", "po_jepa_no_consistency", *api.ADDON_CONDITIONS}
    assert not {"po_jepa_full", "identity", *api.AC_CONDITIONS} & set(jobs.EQUAL_STEP_CONDITIONS)
    with pytest.raises(ValueError, match="只支持"):
        jobs.equal_step_job(tmp_path, tmp_path/"out", torch.device("cpu"), "vjepa2_ac", training_run=None, limit_run=None)
    monkeypatch.setattr(api, "checkpoint_retention_config",
                        lambda: dict(keep_neighbors=2, prune_conditions=["transformer_consistency"]))
    with pytest.raises(ValueError, match="精简过"):
        jobs.equal_step_job(tmp_path, tmp_path/"out", torch.device("cpu"), "transformer_consistency", training_run=None, limit_run=None)
    assert not (tmp_path/"out").exists()


def test_addon_loader_accepts_rule_stopped_runs(api, monkeypatch):
    """补跑评测（D4、谱隙）读检查点：v7.1 以前的报告要跑满；记了早停的报告按当前规则从验证序列重算。"""
    v7load = importlib.import_module("po_jepa.runtime.addon.v7load")
    monkeypatch.setattr(api, "early_stopping_config", lambda: dict(RULE))
    v = rows([1., .5, .4, .45, .46, .3])
    stopped = dict(validations=v[:5], completed_updates=8, early_stopping=api.verify_stopping(v[:5], 8, 100, RULE))
    assert v7load.stopped_by_rule(stopped, 100)
    assert not v7load.stopped_by_rule(dict(stopped, early_stopping=dict(stopped["early_stopping"], best_step=2)), 100)
    other = dict(min_updates=6, patience_validations=3)
    assert not v7load.stopped_by_rule(dict(stopped, early_stopping=dict(stopped["early_stopping"], rule=other)), 100)
    assert not v7load.stopped_by_rule(dict(validations=v, completed_updates=10,
                                           early_stopping=dict(stopped["early_stopping"], completed_updates=10)), 100)
    assert v7load.stopped_by_rule(dict(validations=v, completed_updates=100), 100)         # 旧报告：没有早停字段、跑满
    assert not v7load.stopped_by_rule(dict(validations=v[:5], completed_updates=8), 100)    # 旧报告没跑满


def test_resume_copies_tolerate_copystat_permission(monkeypatch, tmp_path):
    """v7.1 机哥本机补丁第 2 处：ntfs-3g 上 copystat 报 PermissionError 时，续训的复制照样完成，内容逐字节相同；
    别的错误照常抛出；不全局替换 shutil.copystat。"""
    import shutil
    from po_jepa.runtime import resume
    src = tmp_path/"src"; (src/"validations"/"0002").mkdir(parents=True)
    (src/"a.json").write_bytes(b"{}"); (src/"validations"/"0002"/"m.json").write_bytes(b"[1]")
    real = shutil.copystat
    def denied(*a, **k):
        raise PermissionError("Operation not permitted")
    monkeypatch.setattr(shutil, "copystat", denied)
    resume._copy_tree(src, tmp_path/"dst")
    assert (tmp_path/"dst"/"validations"/"0002"/"m.json").read_bytes() == b"[1]"
    resume._copy_file(src/"a.json", tmp_path/"a.json")
    assert (tmp_path/"a.json").read_bytes() == b"{}"
    with pytest.raises(FileExistsError):
        resume._copy_tree(src, tmp_path/"dst")                           # 目标已存在：和 shutil.copytree 一样拒绝
    resume._copy_tree(src, tmp_path/"dst", dirs_exist_ok=True)
    def broken(*a, **k):
        raise OSError("disk error")
    monkeypatch.setattr(shutil, "copystat", broken)
    with pytest.raises(OSError, match="disk error"):
        resume._copy_file(src/"a.json", tmp_path/"b.json")
    monkeypatch.setattr(shutil, "copystat", real)
    text = (ROOT/"src/po_jepa/runtime/resume.py").read_text(encoding="utf-8")
    assert "shutil.copy2(" not in text and "shutil.copytree(" not in text and "shutil.copystat =" not in text


# —— 任务表 ——

def _jobs(args, cwd=ROOT):
    return subprocess.run([sys.executable, "tools/make_v7_jobs.py", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _sections(text):
    out, cur = {}, None
    for line in text.splitlines():
        if line.startswith("# ====="):
            cur = line.split()[2]; out[cur] = []
        elif cur and line and not line.startswith("#"):
            out[cur].append(line)
    return out


def test_full_jobs_keep_encode_and_calibration_commands(tmp_path):
    """A–C2 段的命令逐字不变（run_jobs.py 按命令文字认已完成），D 段起写进 full/v7.1/。"""
    old = subprocess.run(["git", "show", "4324391:tools/make_v7_jobs.py"], cwd=ROOT, capture_output=True, text=True)
    if old.returncode:
        pytest.skip("基线提交 4324391 不在本地历史里")
    (tmp_path/"tools").mkdir(); (tmp_path/"tools/make_v7_jobs.py").write_text(old.stdout, encoding="utf-8")
    (tmp_path/"src").symlink_to(ROOT/"src")
    before, after = _sections(_jobs(["--stage", "full"], tmp_path)), _sections(_jobs(["--stage", "full"]))
    for name in ("A", "B", "B2", "C", "C2"):
        assert after[name] == before[name]
    assert all("/full/v7.1/" in c and "/full/v7.1/" not in b for c, b in zip(after["D"], before["D"]))
    assert len(after["D2"]) == 5*3*2 and all("run.py equal-step" in c and "--limit-run" in c for c in after["D2"])
    assert after["E"] == ["python tools/summarize_v7.py --results ../results-v7/full/v7.1 --output ../results-v7/full/v7.1/summary"]
    variants = _jobs(["--stage", "variants"])
    assert "--training-run ../results-v7/full/v7.1/" in variants and "--job ../results-v7/full/v7.1/" in variants
    assert _sections(_jobs(["--stage", "pilot"]))["D"][0].endswith("--output ../results-v7/pilot/double_well/seed1703/po_jepa_full")


def test_run_jobs_substitutes_limit_run(tmp_path):
    sys.path.insert(0, str(ROOT/"tools"))
    import run_jobs
    done = tmp_path/"r/full-resume1/run/job"; done.mkdir(parents=True); (done/"COMPLETE.json").write_text("{}")
    (tmp_path/"r/full").mkdir()
    cmd, note = run_jobs.prepare("python run.py equal-step --limit-run r/full --training-run r/full --output r/eq", tmp_path)
    assert "--limit-run r/full-resume1" in cmd and "--training-run r/full-resume1" in cmd


# —— 汇总 ——

def test_summary_curves_union_and_grid_tables(tmp_path):
    sys.path.insert(0, str(ROOT/"tools"))
    import summarize_v7 as sv
    def job(condition, steps, completed, by_grid):
        rows_ = {}
        for i, (g, v) in enumerate(by_grid.items()):
            q = dict(kind="point", trajectory_id=1800+i, start_tick=0, duration_cycles=2., grid_size=g, stratum="id")
            values = dict(future_jepa_distance=v, tcd_normalized=0., delta_act=0.)
            if g == 16:
                values["rollout_future_jepa_distance"] = 2*v
            rows_[sv._key(q)] = dict(query=q, values=values)
        return dict(folder=condition, job=dict(condition=condition, dataset="d", updates=10000, seed=1703),
                    report=dict(registration_sha256="x", initial_model_sha256="i", best=dict(step=steps[-1]),
                                training=dict(completed_updates=completed, early_stopping=dict(stopped_early=completed < 10000))),
                    rows=rows_, curves=dict(validation=[dict(step=s, future_jepa_distance=1.) for s in steps], steps=[], losses={}))
    full = job("po_jepa_full", [0, 500, 1000], 1000, {16: .2, 12: .3, 8: .4})
    tf = job("matched_transformer", [0, 500, 1000, 1500], 1500, {16: .1, 12: .5, 8: .6})
    means = sv.condition_means(full)
    assert means["future_by_grid"] == {"16": .2, "12": .3, "8": .4} and means["rollout"]["rollout_future"] == .4
    assert sv.paired(full, tf, grid=12)["mean"] == pytest.approx(-.2) and sv.paired(full, tf, grid=16)["mean"] == pytest.approx(.1)
    eq = job("matched_transformer", [0, 500, 1000], 1000, {16: .15, 12: .5, 8: .6})
    eq["job"]["kind"] = "equal_step"; eq["report"]["equal_step"] = dict(limit_updates=1000, selected_step=1000, own_best_step=1500)
    problems = []
    groups = {("d", 10000, 1703): {"po_jepa_full": full, "matched_transformer": tf}}
    out = sv.equal_step_rows(groups, [eq], problems)
    assert not problems and out[0]["equal_step"]["16"]["mean"] == pytest.approx(.05) and out[0]["main"]["16"]["mean"] == pytest.approx(.1)
