"""tests/addon_dual_tree.py 的比较器、前置检查、子进程自查和快速失败（S00 §3.5）。

只用 tmp_path 里手工造的小目录和几个假的子进程命令，不跑模型、不需要 v7 树，几秒内跑完。
"""
import functools
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import pytest
import torch

import addon_dual_tree as dual

CONDITIONS = ("po_jepa_full", "identity")
UPDATES = 2
SHA_A = "ab"*32
# 嵌套很深的源码 SHA（EXCLUDED_KEYS 里的键在任何层级都不比）
SOURCE = dict(path="/abs/a", generation=dict(sources=[dict(generator_source_sha256={"src/x.py": "d"*64})]))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _edit(path: Path, change) -> None:
    value = _read(path)
    change(value)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")


def _evaluation(e: Path, condition: str) -> None:
    point = dict(kind="point", trajectory_id=3, start=0, duration_cycles=1.0, grid_size=16, stratum="train")
    pair = dict(kind="pair", trajectory_id=4, start=5, duration_cycles=2.0, grid_size=8, stratum="pair")
    _write(e/"registration.json", dict(condition=condition, queries=[point, pair],
                                       checkpoint=dict(sha256="c"*64, seconds=1.5), sources={"x.py": "d"*64}))
    _write(e/"rows"/"00000.json", dict(query=point, condition=condition, status="ok",
                                       values=dict(future_jepa_distance=0.25, tcd_normalized=0.0),
                                       diagnostics=dict(norm=1.5), donor=dict(trajectory_id=7, manifest_sha256="e"*64),
                                       checkpoint=dict(seconds=1.0), source=SOURCE, error=None))
    _write(e/"rows"/"00001.json", dict(query=pair, condition=condition, status="ok",
                                       values=dict(scd_frobenius=0.5), diagnostics=dict(norm=2.5),
                                       checkpoint=dict(seconds=1.0), source=SOURCE, error=None))
    _write(e/"report.json", dict(condition=condition, status="completed_fictional_cpu_only",
                                 counts=dict(expected=2, completed=2, valid=2, failed=0, not_run=0),
                                 groups=[dict(kind="point", mean=0.25)]))
    _write(e/"COMPLETE.json", dict(status="completed_fictional_cpu_only", error=None))


def _job_report(j: Path, condition: str, *, initial, best) -> None:
    _write(j/"report.json", dict(job=dict(condition=condition, training_run="/abs/somewhere"),
                                 preflight=dict(device="cpu"), initial_model_sha256=initial,
                                 registration_sha256="f"*64, best=best,
                                 training=dict(completed_updates=0 if best is None else UPDATES,
                                               consistency_monitor=None),
                                 evaluation=dict(status="completed_fictional_cpu_only",
                                                 counts=dict(failed=0, valid=2)),
                                 figures={}, scientific_conclusions_verified=False))
    _write(j/"COMPLETE.json", dict(report_sha256="0"*64))


def _validation(path: Path) -> None:
    _write(path, dict(queries=[dict(trajectory_id=1, start=0, duration_cycles=1.0)],
                      source_rows=[dict(path="/abs/feature")], values=dict(future_jepa_distance=[0.125])))


def make_run(root: Path, tree: str) -> Path:
    """最小的子进程输出目录：worker.json、标定三件、po_jepa_full（2 步）和 identity。"""
    _write(root/"worker.json", dict(
        tree=tree, po_jepa_file=f"{tree}/src/po_jepa/__init__.py", po_jepa_modules=["src/po_jepa/__init__.py"],
        verify_bundle_sha256="1"*64, smoke_v7_cpu_sha256="2"*64, ac_reduced_official_spec=dict(dual.REDUCED_OFFICIAL),
        ac_checkpoint_sha256="3"*64, torch_version="2.11.0+cpu", threads=1, env=dict(dual.WORKER_ENV),
        python="3.12.3", seconds=10.0))
    smoke = root/"smoke"
    (smoke/"calibrate").mkdir(parents=True)
    (smoke/"calibration.yaml").write_text("sigma_z: 0.5\ntime_weight: 0.125\n", encoding="utf-8")
    _write(smoke/"calibrate"/"sigma_z.json", dict(sigma_z=0.5, windows=4))
    _write(smoke/"calibrate"/"calibration.json", dict(time_weight=0.125, space_weight=0.25,
                                                      v7_calibration_file=f"{root}/smoke/calibration.yaml",
                                                      note=f"written to {root}"))
    for c in CONDITIONS:
        j = smoke/"jobs"/c
        if c == "identity":
            _job_report(j, c, initial=None, best=None)
            _validation(j/"training"/c/"0000.json")
            _evaluation(j/"evaluation"/c, c)
            continue
        best = dict(step=UPDATES, mean=0.375, checkpoint_sha256="9"*64, seconds=2.0)
        _job_report(j, c, initial=SHA_A, best=best)
        t = j/"training"/c
        for step in range(1, UPDATES+1):
            _write(t/"updates"/f"{step:04d}.json", dict(
                condition=c, completed_updates=step, queries=[dict(trajectory_id=1)],
                losses=dict(total=0.1*step, jepa=0.05), grad_max_abs=0.5, grad_norm_before_clip=1.25,
                learning_rate=1e-4, raw_predictions=[dict(mean=0.0)], raw_target=dict(mean=0.0), phi_calls=5,
                consistency_gradient=None, provenance=dict(path="/abs/x"), update_and_observer_seconds=0.5,
                gpu_memory=None))
        for step in (0, UPDATES):
            _validation(t/"validations"/f"{step:04d}"/"0000.json")
            _write(t/"validations"/f"selection-{step:04d}.json", dict(
                step=step, metric="future_jepa_distance", mean=0.375, means=dict(future_jepa_distance=0.375),
                windows=4, seconds=1.0, checkpoint=f"checkpoints/step-{step:04d}.pt", checkpoint_sha256="8"*64,
                checkpoint_provenance=dict(path="/abs/x")))
        _write(t/"report.json", dict(best=best, validations=[], completed_updates=UPDATES))
        (t/"checkpoints").mkdir(parents=True)
        torch.save(dict(
            schema="test", model_state={"w": torch.arange(6, dtype=torch.float32).reshape(2, 3)/7},
            optimizer_state={"state": {0: {"step": torch.tensor(2.0), "exp_avg": torch.full((2, 3), 0.5)}},
                             "param_groups": [{"lr": 1e-4, "params": [0]}]},
            completed_steps=UPDATES, provenance=dict(path="/abs/x"), seconds=1.0,
            rng_state=dict(python=[3, [1, 2, 3], None], numpy=["MT19937", [5, 6], 0, 0, 0.0],
                           torch_cpu=torch.arange(8, dtype=torch.uint8), cuda=None)),
            t/"checkpoints"/f"step-{UPDATES:04d}.pt")
        _evaluation(j/"evaluation"/c, c)
        r = smoke/"jobs"/f"{c}-evaluate"
        _job_report(r, c, initial=SHA_A, best=best)
        _evaluation(r/"evaluation"/c, c)
    return root


@pytest.fixture
def runs(tmp_path):
    return make_run(tmp_path/"a", "/trees/v7"), make_run(tmp_path/"b", "/trees/full")


def compare(a, b):
    return dual.compare_runs(a, b, conditions=CONDITIONS, updates=UPDATES)


def _fields(result):
    return {(m["path"], m["field"]) for m in result["mismatches"]}


T = "smoke/jobs/po_jepa_full/training/po_jepa_full"
E = "smoke/jobs/po_jepa_full/evaluation/po_jepa_full"


# ---------------------------------------------------------------- §3.5 的 10 种情形

def test_identical_runs_compare_equal_with_positive_counts(runs):
    result = compare(*runs)
    assert result["identical"] is True and result["mismatches"] == [] and result["mismatch_count"] == 0
    assert set(dual.ITEMS) <= set(result["counts"]) and all(n > 0 for n in result["counts"].values())
    c = result["counts"]
    assert (c["worker"], c["calibration"], c["complete"], c["initial_model_sha256"], c["updates"],
            c["validation_files"], c["selection"], c["model_state"], c["evaluation_rows"],
            c["evaluation_rows_rerun"], c["evaluation_reports"], c["job_reports"]) == (1, 3, 3, 2, 2, 3, 2, 1, 6, 2, 3, 3)
    assert c["initial_model_sha256_nonnull"] == 1
    assert (c["all_files"], c["all_json"], c["all_checkpoints"]) == (33, 31, 1)


def test_timing_and_provenance_fields_are_ignored(runs):
    a, b = runs
    def retime(v):
        v["seconds"] = 99.5
    for step in (1, 2):
        _edit(b/T/"updates"/f"{step:04d}.json",
              lambda v: v.update(update_and_observer_seconds=123.0, provenance=dict(path="/other"),
                                 gpu_memory=dict(peak=1)))
    for step in (0, 2):
        _edit(b/T/"validations"/f"selection-{step:04d}.json",
              lambda v: v.update(seconds=7.0, checkpoint_sha256="7"*64, checkpoint_provenance=dict(p=1)))
    _edit(b/"worker.json", retime)
    _edit(b/E/"rows"/"00000.json", lambda v: v["checkpoint"].update(seconds=3.0))
    _edit(b/E/"registration.json", lambda v: v["checkpoint"].update(seconds=9.0))
    _edit(b/"smoke/jobs/po_jepa_full/report.json",
          lambda v: v["job"].update(training_run="/other") or v["best"].update(seconds=5.0))
    _edit(b/"smoke/calibrate/calibration.json", lambda v: v.update(note="elsewhere", v7_calibration_file="/x"))
    assert compare(a, b)["identical"] is True


def test_one_ulp_in_a_loss_is_found(runs):
    a, b = runs
    path = b/T/"updates"/"0001.json"
    x = _read(path)["losses"]["total"]
    _edit(path, lambda v: v["losses"].update(total=math.nextafter(x, math.inf)))
    result = compare(a, b)
    assert result["identical"] is False
    assert (f"{T}/updates/0001.json", "losses.total") in _fields(result)
    m = next(m for m in result["mismatches"] if m["field"] == "losses.total")
    assert m["a"] == x.hex() and m["b"] == math.nextafter(x, math.inf).hex()


def test_negative_zero_in_an_evaluation_value_is_found(runs):
    a, b = runs
    _edit(b/E/"rows"/"00000.json", lambda v: v["values"].update(tcd_normalized=-0.0))
    result = compare(a, b)
    assert result["identical"] is False
    assert (f"{E}/rows/00000.json", "values.tcd_normalized") in _fields(result)


def test_one_ulp_in_model_state_is_found(runs):
    a, b = runs
    path = b/T/"checkpoints"/"step-0002.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    w = payload["model_state"]["w"]
    w[1, 2] = torch.nextafter(w[1, 2], torch.tensor(math.inf))
    path.unlink()
    torch.save(payload, path)
    result = compare(a, b)
    assert result["identical"] is False
    m = next(m for m in result["mismatches"] if m["item"] == "model_state")
    assert m["path"] == f"{T}/checkpoints/step-0002.pt" and m["field"] == "model_state.w"
    assert m["a"]["index"] == m["b"]["index"] == 5


def test_initial_model_hash_difference_is_found(runs):
    a, b = runs
    _edit(b/"smoke/jobs/po_jepa_full/report.json", lambda v: v.update(initial_model_sha256="cd"*32))
    result = compare(a, b)
    assert result["identical"] is False
    assert ("smoke/jobs/po_jepa_full/report.json", "initial_model_sha256") in _fields(result)


def test_missing_evaluation_row_is_found(runs):
    a, b = runs
    (b/E/"rows"/"00001.json").unlink()
    result = compare(a, b)
    assert result["identical"] is False
    assert any(m["path"] == f"{E}/rows" and m["field"].startswith("<names>") for m in result["mismatches"])


def test_update_count_must_equal_updates(runs):
    a, b = runs
    (b/T/"updates"/"0002.json").unlink()
    result = compare(a, b)
    assert result["identical"] is False
    assert any(m["path"] == f"{T}/updates" and m["field"].startswith("<names>") for m in result["mismatches"])


def test_refuses_to_compare_a_tree_with_itself(tmp_path):
    a, b = make_run(tmp_path/"a", "/trees/v7"), make_run(tmp_path/"b", "/trees/v7")
    with pytest.raises(ValueError, match="tree 相同"):
        compare(a, b)


# -- 情形 10：前置检查

def _tree(root: Path, smoke=b"print('smoke')\n") -> Path:
    for name in dual.REQUIRED_FILES:
        (root/name).parent.mkdir(parents=True, exist_ok=True)
        (root/name).write_bytes(b"# x\n")
    (root/"tools/smoke_v7_cpu.py").write_bytes(smoke)
    return root


def _git(tree: Path, *args):
    return subprocess.run(["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def trees(tmp_path, monkeypatch):
    """一棵假的 v7 git 检出（V7_COMMIT 换成它自己的提交）和一棵新树。"""
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    v7, new = _tree(tmp_path/"v7"), _tree(tmp_path/"new")
    _git(v7, "init", "-q")
    _git(v7, "add", "-A")
    _git(v7, "commit", "-q", "-m", "frozen")
    monkeypatch.setattr(dual, "V7_COMMIT", _git(v7, "rev-parse", "HEAD"))
    return v7, new


def test_preflight_accepts_frozen_checkout(trees, tmp_path):
    dual.preflight_checks(*trees, tmp_path/"work", parallel=True)
    dual.preflight_checks(*trees, tmp_path/"work", parallel=False)


def test_preflight_rejects_existing_workdir(trees, tmp_path):
    (tmp_path/"work").mkdir()
    with pytest.raises(ValueError, match="--workdir 必须不存在"):
        dual.preflight_checks(*trees, tmp_path/"work", parallel=False)


@pytest.mark.parametrize("which", [0, 1])
def test_preflight_rejects_workdir_inside_a_tree(trees, which):
    with pytest.raises(ValueError, match="--workdir 不能在树里面"):
        dual.preflight_checks(*trees, trees[which]/"results"/"work", parallel=False)


def test_preflight_rejects_same_tree(trees, tmp_path):
    with pytest.raises(ValueError, match="同一个目录"):
        dual.preflight_checks(trees[0], trees[0]/"src"/"..", tmp_path/"work", parallel=False)


def test_preflight_rejects_missing_required_file(trees, tmp_path):
    (trees[1]/"src/po_jepa/runtime/workflow.py").unlink()
    with pytest.raises(ValueError, match="缺 src/po_jepa/runtime/workflow.py"):
        dual.preflight_checks(*trees, tmp_path/"work", parallel=False)


def test_preflight_rejects_different_smoke_scripts(trees, tmp_path):
    (trees[1]/"tools/smoke_v7_cpu.py").write_bytes(b"print('changed')\n")
    with pytest.raises(ValueError, match="smoke_v7_cpu.py 不同"):
        dual.preflight_checks(*trees, tmp_path/"work", parallel=False)


def test_preflight_rejects_v7_tree_that_is_not_a_git_checkout(tmp_path):
    v7, new = _tree(tmp_path/"v7"), _tree(tmp_path/"new")
    with pytest.raises(ValueError, match="不是 git 检出"):
        dual.preflight_checks(v7, new, tmp_path/"work", parallel=False)


def test_preflight_rejects_v7_tree_that_differs_from_frozen_commit(trees, tmp_path):
    (trees[0]/"src/po_jepa/runtime/workflow.py").write_bytes(b"# edited\n")
    with pytest.raises(ValueError, match="和 .* 不同"):
        dual.preflight_checks(*trees, tmp_path/"work", parallel=False)


def test_preflight_rejects_untracked_file_in_v7_tree(trees, tmp_path):
    (trees[0]/"src/po_jepa/extra.py").write_bytes(b"# new\n")
    with pytest.raises(ValueError, match="未提交或未跟踪"):
        dual.preflight_checks(*trees, tmp_path/"work", parallel=False)


def test_preflight_parallel_needs_two_cores(trees, tmp_path, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    with pytest.raises(ValueError, match="cpu_count"):
        dual.preflight_checks(*trees, tmp_path/"work", parallel=True)
    dual.preflight_checks(*trees, tmp_path/"work", parallel=False)


def test_worker_env_is_exactly_the_seven_fixed_variables():
    assert dual.WORKER_ENV == {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                               "MKL_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE",
                               "MKL_CBWR": "COMPATIBLE,STRICT"}
    assert dual.THREADS == 1
    env = dual.worker_environment({"PATH": "/bin", "PYTHONPATH": "/x", "PO_JEPA_SEED": "1", "OMP_PROC_BIND": "1",
                                   "MKL_VERBOSE": "1", "KMP_AFFINITY": "x", "PYTHONHASHSEED": "7"})
    assert env == {"PATH": "/bin", **dual.WORKER_ENV}


# ---------------------------------------------------------------- 其余报错和“不许空过”

def test_interfaces():
    assert dual.JOB_ORDER == ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity",
                              "vjepa2_ac_small", "vjepa2_ac")
    assert dual.REDUCED_OFFICIAL == dict(width=64, depth=2, heads=4, mlp_ratio=4.0)
    assert not [n for n in dir(dual) if n.startswith("test")]
    assert "po_jepa" not in dual.__dict__


def test_items_missing_on_both_sides_count_as_different(runs):
    a, b = runs
    for root in runs:
        (root/"smoke/calibration.yaml").unlink()
        shutil.rmtree(root/T/"validations"/"0002")
    result = compare(a, b)
    assert result["identical"] is False
    assert ("smoke/calibration.yaml", "<file>") in _fields(result)
    assert (f"{T}/validations", "<step 0002>") in _fields(result)


def test_zero_count_item_is_different(runs):
    a, b = runs
    result = dual.compare_runs(a, b, conditions=("identity",), updates=UPDATES)
    assert result["identical"] is False
    assert ("", "<count>") in _fields(result) and {m["item"] for m in result["mismatches"]} >= {"updates", "model_state"}


def test_failed_rows_and_wrong_hash_format_are_different(runs):
    a, b = runs
    for root in runs:
        _edit(root/E/"rows"/"00001.json", lambda v: v.update(status="failed"))
        _edit(root/E/"report.json", lambda v: v["counts"].update(failed=1))
        _edit(root/"smoke/jobs/identity/report.json", lambda v: v.update(initial_model_sha256=SHA_A))
    fields = _fields(compare(a, b))
    assert {(f"{E}/rows/00001.json", "a: status"), (f"{E}/report.json", "a: counts.failed"),
            ("smoke/jobs/identity/report.json", "a: format")} <= fields


def test_int_and_float_are_different(runs):
    a, b = runs
    _edit(b/T/"updates"/"0002.json", lambda v: v.update(grad_max_abs=1))
    _edit(a/T/"updates"/"0002.json", lambda v: v.update(grad_max_abs=1.0))
    assert (f"{T}/updates/0002.json", "grad_max_abs") in _fields(compare(a, b))


def test_mismatch_list_is_capped(runs):
    a, b = runs
    _edit(a/E/"registration.json", lambda v: v.update(queries=[dict(kind="point", n=i) for i in range(300)]))
    _edit(b/E/"registration.json", lambda v: v.update(queries=[dict(kind="point", n=-i-1) for i in range(300)]))
    result = compare(a, b)
    assert len(result["mismatches"]) == dual.MAX_MISMATCHES < result["mismatch_count"]


def test_compare_command_line(runs, capsys, monkeypatch):
    a, b = runs
    # 命令行用 compare_runs 的默认值（6 个条件、20 步）；这里的小目录只有 2 个条件、2 步
    monkeypatch.setattr(dual, "compare_runs", functools.partial(dual.compare_runs, conditions=CONDITIONS,
                                                                updates=UPDATES))
    assert dual.main(["--compare", str(a), str(b)]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("双树对照：逐位相同")
    _edit(b/T/"updates"/"0001.json", lambda v: v["losses"].update(total=math.nextafter(0.1, 1.0)))
    assert dual.main(["--compare", str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert f"{T}/updates/0001.json" in out and "losses" in out
    assert out.splitlines()[-1] == "双树对照：发现 2 处不同"      # 按项（updates）和整份（all_json）各一处
    _edit(b/"worker.json", lambda v: v.update(tree="/trees/v7"))
    assert dual.main(["--compare", str(a), str(b)]) == 1
    assert "tree 相同" in capsys.readouterr().out


def test_command_line_needs_exactly_one_mode(tmp_path):
    with pytest.raises(SystemExit):
        dual.main([])
    with pytest.raises(SystemExit):
        dual.main(["--workdir", str(tmp_path/"w"), "--compare", "a", "b"])
    with pytest.raises(SystemExit):
        dual.main(["--worker", "--tree", str(tmp_path)])
    with pytest.raises(SystemExit):
        dual.main(["--make-ac-checkpoint", str(tmp_path/"x.pt")])


def test_write_json_matches_cache_and_refuses_existing(tmp_path):
    from po_jepa.runtime import cache
    value = {"b": [1, 1.0, -0.0, 0.1], "a": "λ", "c": None}
    dual.write_json(tmp_path/"mine.json", value)
    cache._write_json(tmp_path/"cache.json", value)
    assert (tmp_path/"mine.json").read_bytes() == (tmp_path/"cache.json").read_bytes()
    assert not (tmp_path/"mine.json.partial").exists()
    with pytest.raises(FileExistsError):
        dual.write_json(tmp_path/"mine.json", value)
    with pytest.raises(ValueError):
        dual.write_json(tmp_path/"nan.json", {"x": math.nan})


def test_worker_runtime_check():
    dual.check_worker_runtime(dict(dual.WORKER_ENV), 1)
    with pytest.raises(RuntimeError, match="get_num_threads"):
        dual.check_worker_runtime(dict(dual.WORKER_ENV), 2)
    for key in dual.WORKER_ENV:
        env = dict(dual.WORKER_ENV)
        del env[key]
        with pytest.raises(RuntimeError, match=key):
            dual.check_worker_runtime(env, 1)
    with pytest.raises(RuntimeError, match="MKL_CBWR"):
        dual.check_worker_runtime({**dual.WORKER_ENV, "MKL_CBWR": "AUTO"}, 1)


def test_bind_tree_replaces_script_directory(tmp_path):
    here = str(Path(dual.__file__).resolve().parent)
    path = [here, "/lib"]
    dual.bind_tree(tmp_path, path)
    assert path == [str(tmp_path/"src"), "/lib"]
    with pytest.raises(RuntimeError, match="sys.path"):
        dual.bind_tree(tmp_path, ["/lib"])


def test_modules_must_live_in_their_tree(tmp_path):
    src = tmp_path/"src"/"po_jepa"
    src.mkdir(parents=True)
    inside = types.ModuleType("po_jepa")
    inside.__file__ = str(src/"__init__.py")
    namespace = types.ModuleType("po_jepa.third_party")
    namespace.__file__, namespace.__path__ = None, [str(src/"third_party")]
    found = dual.check_modules_in_tree({"po_jepa": inside, "po_jepa.third_party": namespace}, tmp_path)
    assert found == ["src/po_jepa/__init__.py", "src/po_jepa/third_party"]
    outside = types.ModuleType("po_jepa.x")
    outside.__file__ = str(tmp_path/"elsewhere"/"x.py")
    with pytest.raises(RuntimeError, match="不在"):
        dual.check_modules_in_tree({"po_jepa.x": outside}, tmp_path)
    nowhere = types.ModuleType("po_jepa.y")
    with pytest.raises(RuntimeError, match="没有文件位置"):
        dual.check_modules_in_tree({"po_jepa.y": nowhere}, tmp_path)


def test_make_ac_checkpoint_refuses_existing_target(tmp_path):
    target = tmp_path/"ac.pt"
    target.write_bytes(b"keep")
    before = list(sys.path)
    with pytest.raises(FileExistsError):
        dual.make_ac_checkpoint(target, tmp_path)
    assert target.read_bytes() == b"keep" and sys.path == before


def test_reduced_spec_differs_from_small_and_loads_like_official(tmp_path):
    """缩小规格能构造，参数量 282,048（≠ vjepa2_ac_small 的 215,552），按假权重的键格式能被 load_official 载入。"""
    from po_jepa import ac_baseline
    spec = ac_baseline.ACSpec(**dual.REDUCED_OFFICIAL)
    official = ac_baseline.OFFICIAL
    with torch.random.fork_rng(devices=[]):        # 不扰动后面测试的全局随机状态
        torch.manual_seed(0)
        model = ac_baseline.ACPredictor(spec)
        small = ac_baseline.ACPredictor(ac_baseline.SMALL)
        assert ac_baseline.parameter_count(model) == 282_048 and ac_baseline.parameter_count(small) == 215_552
        state = {k: v for k, v in model.core.state_dict().items() if not k.startswith("action_encoder.")}
        state.update({"action_encoder.weight": torch.zeros(64, 7), "action_encoder.bias": torch.zeros(64),
                      "state_encoder.weight": torch.zeros(64, 7), "state_encoder.bias": torch.ones(64),
                      "extrinsics_encoder.weight": torch.zeros(64, 6), "extrinsics_encoder.bias": torch.zeros(64)})
        buffer = io.BytesIO()
        torch.save({"predictor": {"module.backbone."+k: v for k, v in state.items()}}, buffer)
        payload = torch.load(io.BytesIO(buffer.getvalue()), map_location="cpu", weights_only=True)["predictor"]
        try:
            ac_baseline.OFFICIAL = spec
            loaded = ac_baseline.ACPredictor(spec)
            report = loaded.load_official(payload)
        finally:
            ac_baseline.OFFICIAL = official
    assert report["loaded"] == 30 and len(report["replaced"]) == 6
    assert torch.equal(loaded.state_token.detach(), torch.ones(64))


# ---------------------------------------------------------------- §3.5 情形 11–14

def _items(result, item):
    return {(m["path"], m["field"]) for m in result["mismatches"] if m["item"] == item}


def test_whole_json_catches_fields_no_item_names(runs):
    a, b = runs
    _edit(b/E/"rows"/"00001.json", lambda v: v.update(error="x"))
    _edit(b/T/"updates"/"0001.json", lambda v: v.update(extra_top_level=1))
    _edit(b/E/"COMPLETE.json", lambda v: v.update(status="changed"))
    result = compare(a, b)
    assert result["identical"] is False
    assert _items(result, "all_json") == {(f"{E}/rows/00001.json", "error"), (f"{T}/updates/0001.json", "extra_top_level"),
                                          (f"{E}/COMPLETE.json", "status")}
    assert {m["item"] for m in result["mismatches"]} == {"all_json"}      # 按项的比较不看这些字段


def test_whole_json_ignores_excluded_keys_at_any_depth(runs):
    a, b = runs
    _edit(b/"smoke/jobs/po_jepa_full/COMPLETE.json", lambda v: v.update(report_sha256="1"*64))
    _edit(b/E/"rows"/"00000.json",
          lambda v: v["source"]["generation"]["sources"][0].update(generator_source_sha256={"src/y.py": "0"*64}))
    for root, value in ((a, 1), (b, 2)):
        _edit(root/E/"rows"/"00001.json", lambda v: v.update(audit=[{"deep": {k: value for k in dual.EXCLUDED_KEYS}}]))
    # 真实输出里随源码变的位置（S08a 的双树对照实测）：评测登记的 sources，任务报告、训练报告里预登记的摘要
    for root, x in ((a, "1"), (b, "2")):
        _edit(root/E/"registration.json", lambda v: v.update(sources=dict(
            training_consumer_sha256={"src/po_jepa/model.py": x*64},
            evaluation_source_sha256={"src/po_jepa/runtime/evaluation.py": x*64})))
        _edit(root/"smoke/jobs/po_jepa_full/report.json", lambda v: v.update(registration_sha256=x*64))
        _edit(root/T/"report.json", lambda v: v.update(preregistration_sha256=x*64))
    assert compare(a, b)["identical"] is True
    assert dual.EXCLUDED_KEYS == {"seconds", "update_and_observer_seconds", "gpu_memory", "checkpoint_sha256",
                                  "report_sha256", "training_run", "provenance", "checkpoint_provenance",
                                  "consumer_sha256", "source_files_sha256", "generator_source_sha256",
                                  "training_consumer_sha256", "evaluation_source_sha256",
                                  "registration_sha256", "preregistration_sha256"}


def test_extra_svg_is_caught(runs):
    a, b = runs
    (b/"smoke/jobs/po_jepa_full/figures").mkdir()
    (b/"smoke/jobs/po_jepa_full/figures/loss.svg").write_text("<svg/>", encoding="utf-8")
    result = compare(a, b)
    assert result["identical"] is False
    m = next(m for m in result["mismatches"] if m["item"] == "all_files")
    assert m["a"] == [] and m["b"] == ["smoke/jobs/po_jepa_full/figures/loss.svg"]


def _rewrite_checkpoint(root: Path, change) -> None:
    path = root/T/"checkpoints"/"step-0002.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    change(payload)
    path.unlink()
    torch.save(payload, path)


def test_checkpoint_rng_state_is_ignored(runs):
    a, b = runs
    def reseed(p):
        p["rng_state"]["torch_cpu"][0] += 1
        p["rng_state"]["python"][1][0] = 99
        p["rng_state"]["numpy"] = ["MT19937", [7], 1, 0, 0.5]
        p["seconds"], p["provenance"] = 4.0, dict(path="/other")
    _rewrite_checkpoint(b, reseed)
    assert compare(a, b)["identical"] is True


def _bump_exp_avg(p):
    t = p["optimizer_state"]["state"][0]["exp_avg"]
    t[0, 1] = torch.nextafter(t[0, 1], torch.tensor(math.inf))


def _swap_optimizer_keys(p):
    o = p["optimizer_state"]
    p["optimizer_state"] = {"param_groups": o["param_groups"], "state": o["state"]}


@pytest.mark.parametrize("change, field", [
    (_bump_exp_avg, "optimizer_state.state.0.exp_avg"),
    (lambda p: p.update(completed_steps=3), "completed_steps"),
    (lambda p: p.update(completed_steps=2.0), "completed_steps"),
    (lambda p: p["optimizer_state"]["param_groups"][0].update(lr=math.nextafter(1e-4, 1.0)),
     "optimizer_state.param_groups[0].lr"),
    (lambda p: p["optimizer_state"]["param_groups"][0].update(params=(0,)), "optimizer_state.param_groups[0].params"),
    (_swap_optimizer_keys, "optimizer_state.<keys>"),
    (lambda p: p["optimizer_state"]["state"][0].update(step=2.0), "optimizer_state.state.0.step"),
    (lambda p: p["optimizer_state"]["state"][0].update(step=torch.tensor(2.0, dtype=torch.float64)),
     "optimizer_state.state.0.step"),
])
def test_checkpoint_whole_comparison_catches_everything_but_rng_state(runs, change, field):
    a, b = runs
    _rewrite_checkpoint(b, change)
    result = compare(a, b)
    assert result["identical"] is False
    assert _items(result, "all_checkpoints") == {(f"{T}/checkpoints/step-0002.pt", field)}
    assert {m["item"] for m in result["mismatches"]} == {"all_checkpoints"}   # model_state 那一项看不到


def test_unreadable_files_are_different_in_the_whole_comparison(runs):
    a, b = runs
    (b/T/"updates"/"0001.json").write_text("{not json", encoding="utf-8")
    (b/E/"COMPLETE.json").write_text("", encoding="utf-8")
    checkpoint = b/T/"checkpoints"/"step-0002.pt"
    checkpoint.unlink()
    torch.save({"model_state": {}, "where": Path("/")}, checkpoint)       # weights_only=True 读不了
    result = compare(a, b)
    assert result["identical"] is False
    unreadable = {(m["item"], m["path"]) for m in result["mismatches"]
                  if m["field"] == "<file>" and m["a"] == "present" and str(m["b"]).startswith("<unreadable")}
    assert {("all_json", f"{T}/updates/0001.json"), ("all_json", f"{E}/COMPLETE.json"),
            ("all_checkpoints", f"{T}/checkpoints/step-0002.pt")} <= unreadable


@pytest.mark.parametrize("sides", ["b", "both"])
def test_worker_env_must_match_and_equal_worker_env(runs, sides):
    a, b = runs
    for root in ((b,) if sides == "b" else runs):
        _edit(root/"worker.json", lambda v: v["env"].update(MKL_CBWR="AUTO"))
    result = compare(a, b)
    assert result["identical"] is False
    fields = _items(result, "worker")
    assert ("worker.json", "b: env == WORKER_ENV") in fields
    if sides == "b":
        assert ("worker.json", "env.MKL_CBWR") in fields and ("worker.json", "a: env == WORKER_ENV") not in fields
    else:
        assert ("worker.json", "a: env == WORKER_ENV") in fields and ("worker.json", "env.MKL_CBWR") not in fields


def test_worker_reduced_spec_must_be_exactly_the_four_fields(runs):
    a, b = runs
    for root in runs:       # 两边相同，但用的是 dataclasses.asdict 的写法（多了字段）
        _edit(root/"worker.json", lambda v: v["ac_reduced_official_spec"].update(encoder_dim=1408, action_dim=2))
    fields = _items(compare(a, b), "worker")
    assert {("worker.json", "a: ac_reduced_official_spec == REDUCED_OFFICIAL"),
            ("worker.json", "b: ac_reduced_official_spec == REDUCED_OFFICIAL")} <= fields


SLEEPER = """import os, signal, sys, time
if sys.argv[2] == "ignore-sigterm":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
open(sys.argv[1] + ".tmp", "w").write(str(os.getpid()))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(600)
"""
FAILER = """import os, sys, time
while not os.path.exists(sys.argv[1]):     # 等 sleep 的那个确实起来了
    time.sleep(0.05)
time.sleep(1)
print("fake worker failed", flush=True)
sys.exit(3)
"""


def _fake_workers(monkeypatch, tmp_path, *, sleeper="default"):
    """v7 的假 worker 1 s 后退出码 3；full 的假 worker sleep 600。"""
    pid_file = tmp_path/"sleeper.pid"
    def command(script, tree, features, checkpoint, output):
        if Path(output).name == "v7":
            return [sys.executable, "-c", FAILER, str(pid_file)]
        return [sys.executable, "-c", SLEEPER, str(pid_file), sleeper]
    monkeypatch.setattr(dual, "worker_command", command)
    return pid_file


def _run_fake(tmp_path, *, parallel=True):
    work = tmp_path/"work"
    work.mkdir()
    return dual.run_workers(work, (("v7", tmp_path), ("full", tmp_path)), tmp_path/"script.py",
                            tmp_path/"features", tmp_path/"ac.pt", dict(os.environ), parallel=parallel)


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_fail_fast_kills_the_other_worker(tmp_path, monkeypatch, capsys):
    assert dual.POLL_SECONDS <= 5 and dual.KILL_GRACE_SECONDS == 30
    pid_file = _fake_workers(monkeypatch, tmp_path)
    started = time.monotonic()
    assert _run_fake(tmp_path) == 1
    assert time.monotonic()-started < 60
    assert _gone(int(pid_file.read_text()))
    out = capsys.readouterr().out
    assert "fake worker failed" in out and "SIGTERM" in out
    assert out.splitlines()[-1].startswith("双树对照：失败（worker v7 退出码 3")


def test_fail_fast_escalates_to_sigkill(tmp_path, monkeypatch, capsys):
    pid_file = _fake_workers(monkeypatch, tmp_path, sleeper="ignore-sigterm")
    monkeypatch.setattr(dual, "POLL_SECONDS", 0.2)
    monkeypatch.setattr(dual, "KILL_GRACE_SECONDS", 1)
    started = time.monotonic()
    assert _run_fake(tmp_path) == 1
    assert time.monotonic()-started < 30
    assert _gone(int(pid_file.read_text())) and "SIGKILL" in capsys.readouterr().out


def test_sequential_stops_after_the_first_failure(tmp_path, monkeypatch):
    pid_file = tmp_path/"sleeper.pid"
    pid_file.write_text("0")        # FAILER 不用等 sleep 的那个
    def command(script, tree, features, checkpoint, output):
        if Path(output).name == "v7":
            return [sys.executable, "-c", FAILER, str(pid_file)]
        return [sys.executable, "-c", "open('started', 'w')"]
    monkeypatch.setattr(dual, "worker_command", command)
    assert _run_fake(tmp_path, parallel=False) == 1
    assert not (tmp_path/"started").exists()


@pytest.mark.parametrize("parallel", [True, False])
def test_workers_that_succeed_return_zero(tmp_path, monkeypatch, parallel):
    monkeypatch.setattr(dual, "worker_command", lambda *args: [sys.executable, "-c", "print('ok')"])
    monkeypatch.setattr(dual, "POLL_SECONDS", 0.2)
    assert _run_fake(tmp_path, parallel=parallel) == 0
    assert (tmp_path/"work"/"v7"/"console.log").read_text() == "ok\n"


def test_summary_line_is_the_last_printed_line(runs, capsys):
    result = compare(*runs)
    line = dual.summary_line(result)
    dual._print_result(result, line)
    assert capsys.readouterr().out.splitlines()[-1] == line and line.startswith("双树对照：逐位相同（")
    assert "all_json 31" in line
    different = dict(result, identical=False, mismatch_count=3)
    assert dual.summary_line(different, Path("/w/dual-tree-report.json")) == \
        "双树对照：发现 3 处不同（见 /w/dual-tree-report.json）"
