"""S28c：小规模验证的任务表（tools/make_small_validation_jobs.py）、run.py 的变体参数、评测工具按任务的变体建模型、
读出和模态工具的跨版本读法（--cross-version），以及汇总（tools/summarize_small_validation.py）的指标、identity 逐位核对、
C-1 选型规则和 C-2 验收判定（docs/segments/S28c.md）。"""

import importlib.util
import re
import json
import os
import shlex
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from po_jepa import datasets
from po_jepa.runtime import checks, v7jobs
from po_jepa.runtime.addon import v7load
from test_s25c_long_horizon import active, real  # noqa: F401
from test_s25f_state_readout import BASE, TEST_IDS, _job, _tool, caches, readout_inputs  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import make_small_validation_jobs as jobs  # noqa: E402
import run_jobs  # noqa: E402

V6 = "/data/v6-features"
SCENES = ("advection_diffusion", "double_well", "van_der_pol")
COUNTS = {"b-v7.1": dict(L1=12, L2=3, R0=3, R1=3, R2=3, R3=108, R4=6, R5=1),
          "c1": dict(C=9, C2=3, D=12, E=5),
          "c2-dyadic-zero": dict(C=3, AC=3, C2=2, D=18, D2=12, R3=30, R4=3, E=2),
          "c2-bisection-xavier": dict(AC=3, C2=2, D=18, D2=12, R3=30, R4=3, E=2)}


def stage_lines(stage):
    if stage == "c1":
        return jobs.job_lines("c1", v6_features=V6)
    if stage.startswith("c2-"):
        partition, init = stage[3:].split("-")
        return jobs.job_lines("c2", partition=partition, init=init)
    return jobs.job_lines(stage)


def sections(stage, tmp_path):
    path = tmp_path/f"jobs-{stage}.txt"
    path.write_text("\n".join(stage_lines(stage))+"\n", encoding="utf-8")
    return run_jobs.parse_sections(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


summary_tool = load_module(ROOT/"tools/summarize_small_validation.py", "s28c_summarize_small_validation")


# —— 任务表 ——

@pytest.mark.parametrize("stage", tuple(COUNTS))
def test_stage_sections_and_counts(stage, tmp_path):
    assert {s["name"]: len(s["commands"]) for s in sections(stage, tmp_path)} == COUNTS[stage]


def test_lists_match_the_runtime():
    from po_jepa.runtime import modal_spectrum
    from po_jepa.runtime.addon import conditions
    assert jobs.SCENES == SCENES == summary_tool.SCENES
    assert set(jobs.C2_CONDITIONS[2:]) <= set(conditions.ADDON_CONDITIONS) and set(jobs.EQUAL_STEP) <= set(v7jobs.EQUAL_STEP_CONDITIONS)
    assert jobs.V71_CONDITIONS == v7jobs.JOB_CONDITIONS[:6]
    assert jobs.MODAL_SCENES == tuple(s for s in SCENES if s in modal_spectrum.LINEAR_SCENES)
    assert jobs.REUSED_FROM_V71 == summary_tool.REUSED_FROM_V71
    assert (jobs.PARTITIONS, jobs.INITS) == (summary_tool.PARTITIONS, summary_tool.INITS)
    assert dict(temporal_partition=jobs.DEFAULT_VARIANT[0], readout_init=jobs.DEFAULT_VARIANT[1]) == checks.V7_VARIANT_DEFAULT


def test_every_training_line_names_its_variant_and_calibration_comes_first(tmp_path):
    for stage in ("c1", "c2-dyadic-zero", "c2-bisection-xavier", "c2-bisection-zero"):
        lines = [l for l in stage_lines(stage) if l.startswith("python run.py")]
        for line in lines:
            argv = shlex.split(line)
            assert argv.count("--temporal-partition") == 1 and argv.count("--readout-init") == 1, line
            assert f"{argv[argv.index('--temporal-partition')+1]}-{argv[argv.index('--readout-init')+1]}" in line, line
    c1 = sections("c1", tmp_path)
    calibrated = {(shlex.split(l)[shlex.split(l).index("--temporal-partition")+1], shlex.split(l)[shlex.split(l).index("--readout-init")+1],
                   shlex.split(l)[shlex.split(l).index("--system")+1]) for l in c1[0]["commands"]}
    assert c1[0]["name"] == "C" and calibrated == {(p, i, s) for p, i in jobs.VARIANTS if (p, i) != jobs.DEFAULT_VARIANT for s in SCENES}
    trained = [shlex.split(l) for l in next(s for s in c1 if s["name"] == "D")["commands"]]
    assert {a[a.index("--condition")+1] for a in trained} == {"po_jepa_full"}
    assert {(a[a.index("--temporal-partition")+1], a[a.index("--readout-init")+1], a[a.index("--system")+1]) for a in trained} == {
        (p, i, s) for p, i in jobs.VARIANTS for s in SCENES}
    assert all(a[a.index("--output")+1] == f"../../results-small-validation/c1/{a[a.index('--temporal-partition')+1]}-"
               f"{a[a.index('--readout-init')+1]}/{a[a.index('--system')+1]}/seed1703/po_jepa_full" for a in trained)
    names = [s["name"] for s in sections("c2-dyadic-zero", tmp_path)]
    assert names.index("C") < names.index("AC") < names.index("D") < names.index("D2") < names.index("R3")
    assert "C" not in [s["name"] for s in sections("c2-bisection-xavier", tmp_path)]


def test_double_well_200_reads_the_v6_directory_only_in_c1():
    c1 = stage_lines("c1")
    dw = [l for l in c1 if l.startswith("python run.py") and "--system double_well" in l]
    assert dw and all(f"--features-root {V6}" in l for l in dw)
    assert not any(V6 in l for l in c1 if "--system double_well" not in l)
    for stage in ("b-v7.1", "c2-dyadic-zero"):
        assert not any(V6 in l for l in stage_lines(stage))
        assert not any(re.search(r"-200(?!0)|--trajectories 200 ", l) for l in stage_lines(stage) if l.startswith("python"))


def test_options_are_checked(capsys):
    for argv, message in ((["--stage", "c1"], "c1 必须给 --v6-features"),
                          (["--stage", "c1", "--v6-features", "rel/path"], "要写绝对路径"),
                          (["--stage", "b-v7.1", "--v6-features", V6], "只用于 --stage c1"),
                          (["--stage", "c2"], "显式给"),
                          (["--stage", "c2", "--temporal-partition", "dyadic"], "显式给"),
                          (["--stage", "c1", "--v6-features", V6, "--readout-init", "zero"], "只用于 --stage c2")):
        with pytest.raises(SystemExit) as error:
            jobs.main(argv)
        assert error.value.code == 2 and message in capsys.readouterr().err


def test_b_stage_reads_the_v71_checkpoints_across_versions():
    lines = stage_lines("b-v7.1")
    readout = [shlex.split(l) for l in lines if l.startswith("python tools/state_readout.py") and (" rollout " in l or " counterfactual " in l)]
    assert len(readout) == 108 and all("--cross-version" in a for a in readout)
    jobs_seen = {a[a.index("--job")+1] for a in readout}
    assert jobs_seen == {f"../../results-v7/full/v7.1/{s}/seed{seed}/{c}" for s in SCENES for seed in (1703, 1704, 1705)
                         for c in jobs.V71_CONDITIONS}
    for a in readout:
        s = a[a.index("--job")+1].split("/")[-3]
        if a[2] == "rollout":
            assert a[a.index("--features")+1] == f"../../features-small-validation/{s}~horizon=128-2000"
            assert a[a.index("--base-features")+1] == f"../../features-v7/{s}-2000"
        else:
            assert a[a.index("--features")+1] == f"../../features-v7/{s}-2000" and "--base-features" not in a
        assert ("--ac-checkpoint" in a) == a[a.index("--job")+1].endswith("/vjepa2_ac")
    modal = [shlex.split(l) for l in lines if l.startswith("python tools/modal_spectrum.py")]
    assert len(modal) == 6 and all("--cross-version" in a and "/advection_diffusion/" in a[a.index("--job")+1] for a in modal)
    c2 = [shlex.split(l) for l in stage_lines("c2-dyadic-zero") if "state_readout.py" in l or "modal_spectrum.py" in l]
    assert c2 and not any("--cross-version" in a or "--base-features" in a for a in c2)
    assert {a[a.index("--probe")+1] for a in c2 if "--probe" in a} == {f"../../results-small-validation/readout/probe/{s}" for s in SCENES}


def test_every_run_py_line_is_accepted_and_sets_the_variant(tmp_path, monkeypatch):
    entry = load_module(ROOT/"run.py", "s28c_run_py")
    seen = []
    monkeypatch.setattr(entry.subprocess, "run", lambda command, **kw: seen.append((command, kw["env"])) or Namespace(returncode=0))
    for name in ("PO_JEPA_TEMPORAL_PARTITION", "PO_JEPA_READOUT_INIT"):
        monkeypatch.delenv(name, raising=False)
    work = tmp_path/"work"/"full-model"/"po-jepa-full"      # 任务行的路径是 ../../ 开头：留在 tmp_path 里
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    counted = 0
    for stage in ("c1", "c2-dyadic-zero", "c2-bisection-xavier"):
        for line in stage_lines(stage):
            if not line.startswith("python run.py"):
                continue
            argv = shlex.split(line)
            for key in ("--features-root", "--training-run", "--limit-run"):
                if key in argv:
                    Path(argv[argv.index(key)+1]).mkdir(parents=True, exist_ok=True)
            seen.clear()
            try:
                assert entry.main(argv[2:]) == 0, line
            except SystemExit as error:
                raise AssertionError(line) from error
            env = seen[0][1]
            assert env["PO_JEPA_TEMPORAL_PARTITION"] == argv[argv.index("--temporal-partition")+1]
            assert env["PO_JEPA_READOUT_INIT"] == argv[argv.index("--readout-init")+1]
            counted += 1
    assert counted == (9+3+12)+(3+3+2+18+12)+(3+2+18+12)


def test_run_py_variant_flags(tmp_path, monkeypatch, capsys):
    entry = load_module(ROOT/"run.py", "s28c_run_py_flags")
    seen = []
    monkeypatch.setattr(entry.subprocess, "run", lambda command, **kw: seen.append(kw["env"]) or Namespace(returncode=0))
    (tmp_path/"f").mkdir()
    base = ["run", "--system", "van_der_pol", "--trajectories", "200", "--updates", "2000", "--condition", "po_jepa_full",
            "--features-root", str(tmp_path/"f")]
    monkeypatch.delenv("PO_JEPA_TEMPORAL_PARTITION", raising=False)
    monkeypatch.setenv("PO_JEPA_READOUT_INIT", "zero")
    assert entry.main([*base, "--output", str(tmp_path/"a")]) == 0          # 不写参数：沿用环境变量（S28b 写法）
    assert "PO_JEPA_TEMPORAL_PARTITION" not in seen[-1] and seen[-1]["PO_JEPA_READOUT_INIT"] == "zero"
    assert entry.main([*base, "--readout-init", "zero", "--temporal-partition", "dyadic", "--output", str(tmp_path/"b")]) == 0
    assert (seen[-1]["PO_JEPA_TEMPORAL_PARTITION"], seen[-1]["PO_JEPA_READOUT_INIT"]) == ("dyadic", "zero")
    with pytest.raises(SystemExit) as error:                               # 参数和环境变量打架
        entry.main([*base, "--readout-init", "xavier", "--output", str(tmp_path/"c")])
    assert error.value.code == 2 and "PO_JEPA_READOUT_INIT=zero" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        entry.main([*base, "--temporal-partition", "trisection", "--output", str(tmp_path/"d")])


def test_job_env_follows_the_job_variant(tmp_path, monkeypatch):
    import plan_v7
    monkeypatch.setattr(os, "environ", dict(os.environ, PO_JEPA_TEMPORAL_PARTITION="dyadic", PO_JEPA_READOUT_INIT="zero"))
    plain = tmp_path/"plain"/"run"/"job"; plain.mkdir(parents=True)
    (plain/"job.json").write_text(json.dumps(dict(seed=1704, updates=2000, dataset="double_well-200", condition="po_jepa_full")))
    plan_v7._job_env(tmp_path/"plain")
    assert (os.environ["PO_JEPA_TEMPORAL_PARTITION"], os.environ["PO_JEPA_READOUT_INIT"]) == ("bisection", "xavier")
    assert (os.environ["PO_JEPA_SEED"], os.environ["PO_JEPA_DATASET"]) == ("1704", "double_well:200")
    assert checks.v7_variant() == checks.V7_VARIANT_DEFAULT
    var = tmp_path/"var"; var.mkdir()
    (var/"job.json").write_text(json.dumps(dict(seed=1703, updates=2000, dataset="double_well-200", condition="po_jepa_full",
                                                variant=dict(temporal_partition="dyadic", readout_init="zero"))))
    plan_v7._job_env(var)
    assert checks.v7_variant() == dict(temporal_partition="dyadic", readout_init="zero") and checks.v7_variant_suffix() == "~dyadic~zeroinit"


# —— 跨版本读检查点 ——

def _cross(seen):
    def load(job_dir, condition, model, *, loss_config, features, device):
        seen.update(job_dir=Path(job_dir), condition=condition, features=Path(features), model=type(model).__name__)
        return dict(job_dir=str(job_dir), best=None if condition == "identity" else dict(step=7, mean=.1, checkpoint="c.pt", checkpoint_sha256="a"*64),
                    source_files_sha256={"src/x.py": "b"*64})
    return load


def _no_same_version(*args, **kwargs):
    raise AssertionError("--cross-version 不能走 v7jobs.load_for_evaluation")


def test_rollout_and_counterfactual_cross_version(caches, readout_inputs, active, tmp_path, monkeypatch):
    base, long = caches
    probe, truth = readout_inputs
    tool = _tool("state_readout")
    seen = {}
    monkeypatch.setattr(v7load, "load_v7_selected", _cross(seen))
    monkeypatch.setattr(v7jobs, "load_for_evaluation", _no_same_version)
    job = _job(tmp_path/"a")
    common = ["--job", str(job), "--probe", str(probe), "--truth", str(truth), "--device", "cpu", "--cross-version"]
    out = tmp_path/"r.json"
    tool.main(["rollout", *common, "--features", str(long), "--horizon", "64", "--base-features", str(base), "--output", str(out)])
    assert seen["features"] == base and seen["condition"] == "identity" and seen["job_dir"].name == "job"
    r = json.loads(out.read_text())
    assert r["registration_sha256"] is None and r["cross_version"]["source_files_sha256"] == {"src/x.py": "b"*64}
    assert [x["trajectory_id"] for x in r["rows"]] == list(TEST_IDS) and r["best"] is None
    with pytest.raises(SystemExit, match="--base-features"):
        tool.main(["rollout", *common, "--features", str(long), "--horizon", "64", "--output", str(tmp_path/"r2.json")])
    seen.clear()
    out = tmp_path/"c.json"
    tool.main(["counterfactual", *common, "--features", str(base), "--output", str(out)])
    assert seen["features"] == base
    c = json.loads(out.read_text())
    assert c["registration_sha256"] is None and "cross_version" in c and c["overall"]["windows"] == 80
    # 同版本的路径：报告和以前一样（有预登记 SHA，没有 cross_version 键）；--base-features 只能和 --cross-version 一起用
    monkeypatch.setattr(v7jobs, "load_for_evaluation", lambda condition, run, device: (
        __import__("po_jepa.baselines", fromlist=["IdentityPredictor"]).IdentityPredictor(),
        __import__("test_training").e06_config(), None, dict(fake=True)))
    out = tmp_path/"same.json"
    tool.main(["counterfactual", *common[:-1], "--features", str(base), "--output", str(out)])
    s = json.loads(out.read_text())
    assert "cross_version" not in s and len(s["registration_sha256"]) == 64
    with pytest.raises(SystemExit, match="只和 --cross-version"):
        tool.main(["rollout", *common[:-1], "--features", str(long), "--horizon", "64", "--base-features", str(base),
                   "--output", str(tmp_path/"r3.json")])
    assert not (tmp_path/"r2.json").exists() and not (tmp_path/"r3.json").exists()


def test_cross_version_refusal_is_reported(caches, readout_inputs, active, tmp_path, monkeypatch):
    base, _ = caches
    probe, truth = readout_inputs
    tool = _tool("state_readout")

    def refuse(*args, **kwargs):
        raise ValueError("checkpoint data provenance differs from the current features")
    monkeypatch.setattr(v7load, "load_v7_selected", refuse)
    with pytest.raises(SystemExit, match="跨版本读检查点没通过.*data provenance"):
        tool.main(["counterfactual", "--job", str(_job(tmp_path/"a")), "--probe", str(probe), "--truth", str(truth),
                   "--features", str(base), "--device", "cpu", "--cross-version", "--output", str(tmp_path/"x.json")])
    assert not (tmp_path/"x.json").exists()


def test_modal_cross_version(tmp_path_factory, real, active, tmp_path, monkeypatch):
    import subprocess
    from test_s25c_long_horizon import _clean_env
    root = tmp_path_factory.mktemp("s28c_modal")
    args = ["--system", "advection_diffusion", "--trajectories", "40", "--output", str(root/"ad")]
    for mode, extra in (("encode", ["--fake-encoder", "--device", "cpu", "--splits", "test"]), ("finalize", ["--splits", "test"])):
        r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/encode_v7.py"), mode, *args, *extra], env=_clean_env(),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    spec = datasets.parse("advection_diffusion:40")
    active(spec)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    seen = {}
    monkeypatch.setattr(v7load, "load_v7_selected", _cross(seen))
    monkeypatch.setattr(v7jobs, "load_for_evaluation", _no_same_version)
    tool = _tool("modal_spectrum")
    out = tmp_path/"m.json"
    tool.main(["--job", str(_job(tmp_path/"a", "po_jepa_full", spec.tag)), "--features", str(root/"ad"), "--output", str(out),
               "--device", "cpu", "--cross-version"])
    r = json.loads(out.read_text())
    assert seen["features"] == root/"ad" and seen["condition"] == "po_jepa_full" and seen["model"] == "POJEPA"
    assert r["registration_sha256"] is None and r["best"] == dict(step=7, checkpoint_sha256="a"*64) and r["comparison"]["samples"] == 20


# —— 汇总 ——

TIMES = (1., 2., 3., 4., 8., 12.)
TRAIN = (1., 2., 4., 8.)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def ckpt(condition, dataset, updates, variant=None):
    """合成任务选中检查点的 SHA（读出文件要记同一个）。"""
    return f"sha-{condition}-{dataset}-{updates}-{summary_tool.variant_label(summary_tool.variant_of(dict(variant=variant)))}"


def make_job(folder, condition, dataset, p, *, seed=1703, updates=2000, variant=None, failed=0, kind=None, registered=True):
    """评测行：2 条轨迹 × 每个 (时长, 网格) 1 个单点（8×8 只有训练时长），网格对 8、12 各 1 个。p 给每个量的值。
    registration.json 登记同一组查询（failed 行是登记的查询之一失败了）。"""
    job_dir = folder/"run"/"job"
    job = dict(condition=condition, dataset=dataset, seed=seed, updates=updates)
    if variant:
        job["variant"] = variant
    if kind:
        job["kind"] = kind
    write_json(job_dir/"job.json", job)
    write_json(job_dir/"report.json", dict(training=dict(completed_updates=updates, early_stopping=dict(stopped_early=False)),
                                           best=dict(step=updates, checkpoint_sha256=ckpt(condition, dataset, updates, variant))))
    write_json(job_dir/"COMPLETE.json", dict(report_sha256="x"))
    evaluation = job_dir/"evaluation"/condition
    queries, i = [], 0
    for tid in (0, 1):
        for t in TIMES:
            for g in (16, 12, 8):
                if g == 8 and t not in TRAIN:
                    continue
                v = dict(future_jepa_distance=p[f"f{g}"]+(p.get("bump", 0.) if g == 16 and t in (3., 12.) else 0.)+tid*p.get("spread", 0.),
                         tcd_normalized=p.get("tcd", 1.), delta_act=p.get("delta", .01))
                if g == 16:
                    v["rollout_future_jepa_distance"] = p.get(f"r{t:g}", v["future_jepa_distance"])
                q = dict(kind="point", trajectory_id=tid, start_tick=0, duration_cycles=t, grid_size=g, stratum=None)
                queries.append(q)
                write_json(evaluation/"rows"/f"{i:04d}.json", dict(status="ok", query=q, values=v))
                i += 1
        for g in (8, 12):
            q = dict(kind="pair", trajectory_id=tid, start_tick=0, duration_cycles=1., grid_size=g, stratum=None)
            queries.append(q)
            write_json(evaluation/"rows"/f"{i:04d}.json", dict(status="ok", query=q, values=dict(scd_normalized=p.get("scd", 1.))))
            i += 1
    for k in range(failed):
        q = dict(kind="point", trajectory_id=9, start_tick=k, duration_cycles=1., grid_size=16, stratum=None)
        queries.append(q)
        write_json(evaluation/"rows"/f"f{k}.json", dict(status="failed", query=q, values={}))
    if registered:
        write_json(evaluation/"registration.json", dict(condition=condition, queries=queries))
    return job_dir


P = dict(f16=1., f12=1.2, f8=1.4, bump=.3, tcd=.5, scd=.2, r8=2., r12=3.)


def test_metrics_definitions(tmp_path):
    j = summary_tool.load_job(make_job(tmp_path/"a", "po_jepa_full", "double_well-200", dict(P, spread=.1), failed=2))
    m = summary_tool.metrics(j)
    assert m["direct16"] == pytest.approx(1.+.3*2/6+.05) and m["heldout16"] == pytest.approx(1.3+.05)
    assert m["grid12"] == pytest.approx(1.2+.3*0+.05) and m["grid8"] == pytest.approx(1.45)
    assert m["rollout16@8"] == 2. and m["rollout16@12"] == 3. and m["tcd"] == .5 and m["scd"] == .2
    assert m["windows"] == dict(point=2*(6+6+4), pair=4, failed=2) and m["finite"] and m["completed_updates"] == 2000


def test_identity_check(tmp_path):
    a = summary_tool.load_job(make_job(tmp_path/"a", "identity", "x-2000", P))
    b = summary_tool.load_job(make_job(tmp_path/"b", "identity", "x-2000", P))
    assert summary_tool.identity_check(a, b) == dict(identical=True, rows=36)
    c = summary_tool.load_job(make_job(tmp_path/"c", "identity", "x-2000", dict(P, f12=1.2+1e-15)))
    r = summary_tool.identity_check(a, c)
    assert not r["identical"] and "12/36 行的值不同" in r["reason"]
    d = summary_tool.load_job(make_job(tmp_path/"d", "identity", "x-2000", P))
    d["rows"].pop(next(iter(d["rows"])))
    assert "没有覆盖登记的查询" in summary_tool.identity_check(a, d)["reason"]
    e = summary_tool.load_job(make_job(tmp_path/"e", "identity", "x-2000", P, failed=1))
    assert "失败" in summary_tool.identity_check(a, e)["reason"]
    assert not summary_tool.identity_check(None, a)["identical"]


def test_identity_check_refuses_duplicate_and_empty_rows(tmp_path):
    """Codex S28c P1-1：重复键不能被排序靠后的行静默覆盖；两边都删光（0 行）不算相同；登记的查询不同也不算。"""
    a = summary_tool.load_job(make_job(tmp_path/"a", "identity", "x-2000", P))
    folder = make_job(tmp_path/"b", "identity", "x-2000", P)
    rows = folder/"evaluation"/"identity"/"rows"
    first = json.loads((rows/"0000.json").read_text())
    write_json(rows/"0000a.json", dict(first, values=dict(first["values"], future_jepa_distance=99.)))   # 同键、不同值，排在正确行之后
    b = summary_tool.load_job(folder)
    assert b["duplicate_rows"] == 1
    r = summary_tool.identity_check(a, b)
    assert not r["identical"] and "重复" in r["reason"]
    empty = []
    for name in ("e1", "e2"):
        f = make_job(tmp_path/name, "identity", "x-2000", P)
        for path in (f/"evaluation"/"identity"/"rows").glob("*.json"):
            path.unlink()
        empty.append(summary_tool.load_job(f))
    r = summary_tool.identity_check(*empty)
    assert not r["identical"] and "没有覆盖登记的查询" in r["reason"]
    unregistered = [summary_tool.load_job(make_job(tmp_path/n, "identity", "x-2000", P, registered=False)) for n in ("u1", "u2")]
    assert "没有登记的评测查询" in summary_tool.identity_check(*unregistered)["reason"]
    other = make_job(tmp_path/"o", "identity", "x-2000", P)
    reg = json.loads((other/"evaluation"/"identity"/"registration.json").read_text())
    reg["queries"][0]["start_tick"] = 1
    write_json(other/"evaluation"/"identity"/"registration.json", reg)
    assert not summary_tool.identity_check(a, summary_tool.load_job(other))["identical"]


def test_finished_attempt_follows_resume_dirs(tmp_path):
    assert summary_tool.finished_attempt(tmp_path/"job") is None
    (tmp_path/"job"/"run"/"job").mkdir(parents=True)
    (tmp_path/"job"/"run"/"job"/"job.json").write_text("{}")             # 第一次没跑完
    done = make_job(tmp_path/"job-resume1", "po_jepa_full", "x-2000", P)
    assert summary_tool.finished_attempt(tmp_path/"job") == done


def c1_tree(root, overrides):
    for p in jobs.PARTITIONS:
        for i in jobs.INITS:
            for s in SCENES:
                values = dict(P, **overrides.get((p, i, s), {}))
                variant = None if (p, i) == jobs.DEFAULT_VARIANT else dict(temporal_partition=p, readout_init=i)
                make_job(root/f"{p}-{i}"/s/"seed1703"/"po_jepa_full", "po_jepa_full", f"{s}-200", values, variant=variant)


def test_c1_rule_adopts_dyadic_and_zero(tmp_path):
    better = dict(r8=1.9, r12=2.9, f16=1.004)               # 两个滚动都更好，直接预测 +0.4%
    c1_tree(tmp_path, {("dyadic", "zero", "double_well"): better, ("dyadic", "zero", "van_der_pol"): better,
                       ("dyadic", "zero", "advection_diffusion"): dict(r8=2.1, f16=.99),
                       ("dyadic", "xavier", "double_well"): dict(f16=1.004)})
    s = summary_tool.c1_summary(tmp_path)
    assert not s["missing"] and not s["problems"]
    d = s["decision"]
    assert (d["temporal_partition"], d["readout_init"]) == ("dyadic", "zero") and d["m1"]["adopt"]
    assert "--temporal-partition dyadic --readout-init zero" in d["command"]
    assert [r["rollout_better"] for r in d["m1"]["scenes"]] == [False, True, True]
    assert s["m1_by_init"]["xavier"]["adopt"] is False                    # Xavier 列只报告：滚动都一样
    md = summary_tool.markdown(s)
    assert "推荐：**dyadic-zero**" in md and "M1 多段切分 vs 二等分（初始化 zero，进判定）" in md


def test_c1_rule_keeps_bisection_when_direct_gets_worse(tmp_path):
    better = dict(r8=1.9, r12=2.9)
    # bump = 0 时 direct16 = f16：van_der_pol 上 dyadic-zero 比 bisection-zero 差 0.51%（> 0.5%）；bisection 下零初始化两个场景差 0.9%
    c1_tree(tmp_path, {("dyadic", "zero", s): better for s in SCENES} | {
        ("dyadic", "zero", "van_der_pol"): dict(better, f16=1.11*1.0051, bump=0.),
        ("bisection", "zero", "double_well"): dict(f16=1.11, bump=0.), ("bisection", "zero", "van_der_pol"): dict(f16=1.11, bump=0.)})
    d = summary_tool.c1_summary(tmp_path)["decision"]
    assert d["temporal_partition"] == "bisection" and not d["m1"]["adopt"]
    assert d["readout_init"] == "xavier" and [r["direct_ok"] for r in d["m2"]["scenes"]] == [True, False, False]


def test_c1_rule_needs_both_rollouts_in_a_scene(tmp_path):
    # 只有 double_well 两个滚动都更好；另两个场景只有 @8 更好 → 1/3 场景，不采用多段切分
    c1_tree(tmp_path, {("dyadic", "zero", "double_well"): dict(r8=1.9, r12=2.9),
                       ("dyadic", "zero", "advection_diffusion"): dict(r8=1.9), ("dyadic", "zero", "van_der_pol"): dict(r8=1.9)})
    d = summary_tool.c1_summary(tmp_path)["decision"]
    assert [r["rollout_better"] for r in d["m1"]["scenes"]] == [False, True, False]
    assert not d["m1"]["adopt"] and d["temporal_partition"] == "bisection"


def test_c1_needs_every_cell_and_matching_dirs(tmp_path):
    c1_tree(tmp_path, {})
    import shutil
    shutil.rmtree(tmp_path/"dyadic-zero"/"van_der_pol")
    s = summary_tool.c1_summary(tmp_path)
    assert s["decision"] is None and s["missing"] == [dict(partition="dyadic", init="zero", scene="van_der_pol")]
    make_job(tmp_path/"bisection-zero"/"extra", "po_jepa_full", "van_der_pol-200", P, variant=dict(temporal_partition="dyadic", readout_init="zero"))
    assert any("目录不在" in p for p in summary_tool.c1_summary(tmp_path)["problems"])


DYZ = dict(temporal_partition="dyadic", readout_init="zero")


def readout_file(root, kind, scene, seed, condition, variant=None, src=None, **values):
    """读出文件；来源字段（训练数据集、检查点、探针、真值、horizon）默认和 c2_tree 的任务对得上，src 改写其中几项。"""
    source = dict(train_dataset=f"{scene}-2000", updates=10000,
                  best=dict(step=10000, checkpoint_sha256=ckpt(condition, f"{scene}-2000", 10000, variant)),
                  probe_sha=f"probe-{scene}", truth_sha256=f"truth-{scene}", horizon=128, curve_times=[8., 16.])
    source.update(src or {})
    probe_sha = source.pop("probe_sha")
    if kind == "rollout":
        body = dict(kind="state_readout_rollout", means={"nmse:T=8": values["nmse"], "nmse:T=16": values["nmse"],
                                                         "floor_nmse:T=8": .01, "floor_nmse:T=16": .01, "vpt@0.1": values["vpt"],
                                                         "vpt@0.05": values["vpt"], "vpt@0.25": values["vpt"], "floor_vpt@0.1": 128},
                    probe=dict(cv_r2=.9, weight_sha256=probe_sha), counts=dict(trajectories=20))
    elif kind == "counterfactual":
        body = dict(kind="state_readout_counterfactual", overall=dict(cf_true=1., cf_pred=.5, explained=.1, gain=values["gain"],
                                                                     cos_mean=values["cos"]), probe=dict(weight_sha256=probe_sha))
        source.pop("horizon"); source.pop("curve_times")
    else:
        body = dict(kind="modal_spectrum", comparison=dict(non_oscillatory=.8, null_non_oscillatory=.06, decay_within_true=.5,
                                                          learned=dict(frequency_quantiles=[0, 1, 2], decay_quantiles=[0, 1, 2])),
                    init_reference=dict(non_oscillatory=.9))
    if kind == "modal":
        source = {}
    write_json(root/kind/scene/f"seed{seed}"/f"{condition}.json", dict(body, condition=condition, seed=seed, **source))


def c2_tree(tmp_path, new=None, old=None, identity=None):
    root, v71 = tmp_path/"c2", tmp_path/"v71"
    variant = dict(temporal_partition="dyadic", readout_init="zero")
    base = dict(po_jepa_full=dict(P, f8=1.3, f12=1.1, r8=1.8, r12=2.7, tcd=.4),
                po_jepa_no_consistency=dict(P, f16=1.002), transformer_consistency=dict(P, f16=.995, r12=2.75),
                fixed_fourier=dict(P, f8=1.5, f12=1.3, r12=3.2), po_jepa_rollout_sup=P, identity=dict(P, f16=2., f12=2., f8=2.))
    reused = dict(matched_transformer=dict(P, f16=.98), vjepa2_ac_small=P, vjepa2_ac=P, po_jepa_full=dict(P, f16=1.01),
                  identity=dict(P, f16=2., f12=2., f8=2.))
    for s in SCENES:
        for c, p in base.items():
            make_job(root/s/"seed1703"/c, c, f"{s}-2000", dict(p, **((new or {}).get((s, c), {}))), updates=10000, variant=variant)
        for c, p in reused.items():
            q = dict(p, **((old or {}).get((s, c), {})))
            if c == "identity" and identity and s in identity:
                q = dict(q, **identity[s])
            make_job(v71/s/"seed1703"/c, c, f"{s}-2000", q, updates=10000)
        rd, v7rd = tmp_path/"readout", tmp_path/"v7readout"
        readout_file(rd, "rollout", s, 1703, "po_jepa_full", DYZ, nmse=.2, vpt=40)
        readout_file(rd, "rollout", s, 1703, "po_jepa_no_consistency", DYZ, nmse=.3, vpt=32)
        readout_file(rd, "counterfactual", s, 1703, "po_jepa_full", DYZ, gain=.4, cos=.5)
        readout_file(v7rd, "rollout", s, 1703, "matched_transformer", nmse=.2005, vpt=40)
        readout_file(v7rd, "counterfactual", s, 1703, "po_jepa_full", gain=.1, cos=.2)
    make_job(root/"equal-step"/"double_well"/"seed1703"/"po_jepa_no_consistency", "po_jepa_no_consistency", "double_well-2000", P,
             updates=10000, variant=variant, kind="equal_step")          # 不进主表
    return root, v71, tmp_path/"readout", tmp_path/"v7readout"


def test_c2_acceptance_passes_on_a_constructed_good_case(tmp_path):
    s = summary_tool.c2_summary(*c2_tree(tmp_path))
    assert not s["problems"] and not s["missing"] and s["variant"] == ["dyadic-zero"]
    assert all(x["identity_gate"]["identical"] for x in s["scenes"].values())
    c = s["criteria"]
    assert c["2_time_consistency"]["pass_"] and c["3_space_consistency"]["pass_"]
    assert c["4_structure"]["pass_"] and c["4_structure"]["vs_transformer_consistency"]["not_worse_scenes"] == dict(
        grid8=3, grid12=3, **{"rollout16@12": 3}, heldout16=3)
    assert c["5_physics"]["pass_"] and c["6_zero_init"]["pass_"]
    assert set(s["scenes"]["double_well"]["v71"]) == {"matched_transformer", "vjepa2_ac_small", "vjepa2_ac", "po_jepa_full", "identity"}
    md = summary_tool.markdown(s)
    assert md.count("**通过**") == 7 and "**未通过**" not in md


def test_c2_acceptance_failures(tmp_path):
    # double_well：full 的直接预测比 no-cons 差 0.6% → 第 2 条不过；van_der_pol 的 identity 和 v7.1 不同 → 不复用 v7.1，第 5、6 条缺场景
    s = summary_tool.c2_summary(*c2_tree(tmp_path, new={("double_well", "po_jepa_full"): dict(f16=1.002*1.006)},
                                         identity={"van_der_pol": dict(f8=2.0000001)}))
    c = s["criteria"]
    assert not s["scenes"]["van_der_pol"]["identity_gate"]["identical"] and s["scenes"]["van_der_pol"]["v71"] == {}
    rows2 = {r["scene"]: r for r in c["2_time_consistency"]["scenes"]}
    assert not c["2_time_consistency"]["pass_"] and not rows2["double_well"]["direct_ok"] and rows2["van_der_pol"]["pass"]
    assert not c["6_zero_init"]["pass_"] and [r["scene"] for r in c["6_zero_init"]["scenes"]] == ["advection_diffusion", "double_well"]
    assert not c["5_physics"]["pass_"] and len(c["5_physics"]["counterfactual"]["scenes"]) == 2
    assert c["3_space_consistency"]["pass_"]


def test_c2_structure_rule(tmp_path):
    # transformer_consistency 在两个场景的 grid12 上更好 1%：grid12 只有 1/3 场景不输 → 不过；fixed_fourier 在两个场景只输 2/4 个指标
    tc = dict(f12=1.089)
    s = summary_tool.c2_summary(*c2_tree(tmp_path, new={("double_well", "transformer_consistency"): tc,
                                                        ("van_der_pol", "transformer_consistency"): tc,
                                                        ("double_well", "fixed_fourier"): dict(f8=1.2),        # 胜 2/4
                                                        ("van_der_pol", "fixed_fourier"): dict(f8=1.2, f12=1.0)}))  # 胜 1/4
    st = s["criteria"]["4_structure"]
    assert st["vs_transformer_consistency"]["not_worse_scenes"]["grid12"] == 1 and not st["vs_transformer_consistency"]["pass_"]
    assert [r["pass_"] for r in st["vs_fixed_fourier"]["scenes"]] == [True, False, False] and not st["vs_fixed_fourier"]["pass_"]
    assert [sum(r["wins"].values()) for r in st["vs_fixed_fourier"]["scenes"]] == [3, 2, 1]


def test_c2_time_rule_needs_tcd_and_counterfactual_needs_positive_cos(tmp_path):
    # double_well：full 的 tcd 比 no-cons 差（其余都更好）→ 第 2 条该场景不过；van_der_pol：gain 更高但 cos_mean < 0 → 反事实不过
    root, v71, rd, v7rd = c2_tree(tmp_path, new={("double_well", "po_jepa_full"): dict(tcd=.6)})
    readout_file(rd, "counterfactual", "van_der_pol", 1703, "po_jepa_full", DYZ, gain=.4, cos=-.1)
    c = summary_tool.c2_summary(root, v71, rd, v7rd)["criteria"]
    rows2 = {r["scene"]: r for r in c["2_time_consistency"]["scenes"]}
    assert rows2["double_well"]["rollout8"] and rows2["double_well"]["direct_ok"] and rows2["double_well"]["tcd"] is False
    assert not rows2["double_well"]["pass"] and not c["2_time_consistency"]["pass_"]
    cf = {r["scene"]: r["pass_"] for r in c["5_physics"]["counterfactual"]["scenes"]}
    assert cf == dict(advection_diffusion=True, double_well=True, van_der_pol=False) and not c["5_physics"]["pass_"]


def test_c2_flags_mixed_variants_and_nonfinite(tmp_path):
    root, v71, rd, v7rd = c2_tree(tmp_path)
    make_job(root/"double_well"/"seed1703"/"po_jepa_full-other", "po_jepa_full", "double_well-2000", P, updates=10000)
    bad = make_job(tmp_path/"c2b"/"double_well"/"seed1703"/"po_jepa_full", "po_jepa_full", "double_well-2000", dict(P, f16=float("nan")),
                   updates=10000)
    s = summary_tool.c2_summary(root, v71, rd, v7rd)
    assert any("混了几个变体" in p for p in s["problems"]) and any("重复" in p for p in s["problems"])
    assert s["criteria"] is None and "不出判定" in summary_tool.markdown(s)
    s = summary_tool.c2_summary(tmp_path/"c2b", v71)
    assert any("非有限" in p for p in s["problems"]) and summary_tool.metrics(summary_tool.load_job(bad))["finite"] is False
    assert [x["condition"] for x in s["invalid"]] == ["po_jepa_full"] and not s["scenes"]["double_well"]["new"]


def _drop_rows(job_dir, condition, keep):
    for path in (job_dir/"evaluation"/condition/"rows").glob("*.json"):
        if not keep(json.loads(path.read_text())["query"]):
            path.unlink()


def test_c2_invalid_jobs_do_not_vote(tmp_path):
    """Codex S28c P1-2：第三个场景的 TC 只剩 16×16 训练时长行（OOD 指标全算不出）→ 这个任务无效，第 4 条缺场景不过；
    full 有失败行 → full 无效，第 2–6 条都不过。"""
    root, v71, rd, v7rd = c2_tree(tmp_path)
    tc = root/"van_der_pol"/"seed1703"/"transformer_consistency"/"run"/"job"
    _drop_rows(tc, "transformer_consistency", lambda q: q["kind"] == "point" and q["grid_size"] == 16 and q["duration_cycles"] in TRAIN)
    s = summary_tool.c2_summary(root, v71, rd, v7rd)
    assert [(x["scene"], x["condition"]) for x in s["invalid"]] == [("van_der_pol", "transformer_consistency")]
    assert "没有覆盖登记的查询" in s["invalid"][0]["reasons"][0]
    st = s["criteria"]["4_structure"]
    assert len(st["vs_transformer_consistency"]["scenes"]) == 2 and not st["vs_transformer_consistency"]["pass_"]
    assert s["criteria"]["2_time_consistency"]["pass_"]                         # 其他条款不受影响
    root2, v712, rd2, v7rd2 = c2_tree(tmp_path/"b")
    full = root2/"double_well"/"seed1703"/"po_jepa_full"
    import shutil
    shutil.rmtree(full)
    make_job(full, "po_jepa_full", "double_well-2000", dict(P, f8=1.3, f12=1.1, r8=1.8, r12=2.7, tcd=.4), updates=10000, variant=DYZ, failed=1)
    c = summary_tool.c2_summary(root2, v712, rd2, v7rd2)["criteria"]
    assert not any(c[k]["pass_"] for k in ("2_time_consistency", "3_space_consistency", "4_structure", "5_physics", "6_zero_init"))


def test_c2_checks_dataset_and_budget(tmp_path):
    """Codex S28c P2-5：本次任务全部写成 200 条 × 2000 步 → 都无效，不出通过。"""
    root, v71, rd, v7rd = c2_tree(tmp_path)
    for path in root.rglob("job.json"):
        job = json.loads(path.read_text())
        write_json(path, dict(job, dataset=job["dataset"].replace("-2000", "-200"), updates=2000))
    s = summary_tool.c2_summary(root, v71, rd, v7rd)
    assert len(s["invalid"]) == 3*6 and all("不是" in x["reasons"][0] for x in s["invalid"])
    assert not any(s["criteria"][k]["pass_"] for k in s["criteria"])
    root, v71, rd, v7rd = c2_tree(tmp_path/"b")
    old = v71/"double_well"/"seed1703"/"matched_transformer"/"run"/"job"/"job.json"
    write_json(old, dict(json.loads(old.read_text()), updates=2000))
    s = summary_tool.c2_summary(root, v71, rd, v7rd)
    assert [x["condition"] for x in s["invalid"]] == ["v7.1 matched_transformer"] and not s["criteria"]["6_zero_init"]["pass_"]


@pytest.mark.parametrize("what", ["dataset", "horizon", "probe", "truth", "checkpoint"])
def test_c2_physics_checks_readout_sources(tmp_path, what):
    """Codex S28c P2-4：读出的训练数据集、horizon、探针、真值不一致，或检查点不是训练任务选中的 → 那一项不进第 5 条。"""
    root, v71, rd, v7rd = c2_tree(tmp_path)
    src = dict(dataset=dict(train_dataset="double_well-200"), horizon=dict(horizon=64), probe=dict(probe_sha="other"),
               truth=dict(truth_sha256="other"), checkpoint=dict(best=dict(step=1, checkpoint_sha256="other")))[what]
    readout_file(rd, "rollout", "double_well", 1703, "po_jepa_no_consistency", DYZ, src=src, nmse=.3, vpt=32)
    s = summary_tool.c2_summary(root, v71, rd, v7rd)
    ph = s["criteria"]["5_physics"]
    assert [r["scene"] for r in ph["vs_no_consistency"]["scenes"]] == ["advection_diffusion", "van_der_pol"]
    assert not ph["vs_no_consistency"]["pass_"] and not ph["pass_"]
    assert any("double_well" in p and "rollout" in p for p in s["problems"])


def test_c1_invalid_cells_block_the_recommendation(tmp_path):
    """Codex S28c P1-2、P2-5：C-1 有失败行或规模不对的格 → 不给推荐。"""
    better = dict(r8=1.9, r12=2.9, f16=1.004)
    c1_tree(tmp_path, {("dyadic", "zero", s): better for s in SCENES})
    assert summary_tool.c1_summary(tmp_path)["decision"]["temporal_partition"] == "dyadic"
    import shutil
    cell = tmp_path/"dyadic-zero"/"double_well"/"seed1703"/"po_jepa_full"
    shutil.rmtree(cell)
    make_job(cell, "po_jepa_full", "double_well-200", dict(P, **better), variant=DYZ, failed=1)
    s = summary_tool.c1_summary(tmp_path)
    assert s["decision"] is None and s["invalid"][0]["key"] == ["dyadic", "zero", "double_well"] and not s["missing"]
    assert "无效" in summary_tool.markdown(s)
    shutil.rmtree(cell)
    make_job(cell, "po_jepa_full", "double_well-200", dict(P, **better), variant=DYZ, updates=10000)
    assert summary_tool.c1_summary(tmp_path)["decision"] is None


def test_data_constants_match_the_job_tables():
    assert summary_tool.C1_DATA == jobs.C1 and summary_tool.C2_DATA == jobs.C2 and summary_tool.LONG_HORIZON == jobs.LONG_HORIZON


def test_readout_summary(tmp_path):
    for seed, nmse in ((1703, .2), (1704, .4)):
        readout_file(tmp_path, "rollout", "double_well", seed, "po_jepa_full", nmse=nmse, vpt=40)
        readout_file(tmp_path, "counterfactual", "double_well", seed, "po_jepa_full", gain=.4, cos=.5)
    readout_file(tmp_path, "modal", "advection_diffusion", 1703, "po_jepa_full")
    write_json(tmp_path/"actions"/"probe-actions-double_well.json", dict(dataset="double_well-2000", by_duration={}, overall=dict(
        identity=1., oracle_delta_act=.1, motion_px=3., action_swap_px=1., oracle_over_identity=.1, windows=8)))
    s = summary_tool.readout_summary(tmp_path, tmp_path/"actions")
    dw = next(e for e in s["across_seeds"] if e["scene"] == "double_well")
    assert dw["seeds"] == [1703, 1704] and dw["rollout"]["nmse_mean"] == pytest.approx(.3) and dw["rollout"]["seeds"] == 2
    assert dw["rollout"]["nmse_mean_seed_std"] == pytest.approx(.1414213562, rel=1e-6)
    ad = next(e for e in s["across_seeds"] if e["scene"] == "advection_diffusion")
    assert ad["modal"]["statistic"] == "non_oscillatory" and ad["modal"]["value"] == .8
    md = summary_tool.markdown(s)
    assert "模态频率对照" in md and "double_well-2000" in md
    write_json(tmp_path/"rollout"/"double_well"/"seed1705"/"po_jepa_full.json", dict(kind="state_readout_rollout", condition="other", seed=1705))
    with pytest.raises(SystemExit, match="不是"):
        summary_tool.readout_summary(tmp_path)


def test_summary_cli(tmp_path):
    root, v71, rd, v7rd = c2_tree(tmp_path)
    assert summary_tool.main(["c2", "--results", str(root), "--v7-results", str(v71), "--readout", str(rd), "--v7-readout", str(v7rd),
                              "--output", str(tmp_path/"out")]) == 0
    assert json.loads((tmp_path/"out"/"summary.json").read_text())["criteria"]["2_time_consistency"]["pass_"]
    assert (tmp_path/"out"/"SUMMARY.md").read_text(encoding="utf-8").startswith("# 小规模验证汇总：c2")


# —— Codex S28c P2-3：训练完成、评测中断后续跑（-resume1 只有 resume_from，没有自己的 training/<条件>）——

def _resumed_layout(tmp_path, registration, *, linked_registration=None, linked_condition="po_jepa_full"):
    orig, resumed = tmp_path/"po_jepa_full"/"run"/"job", tmp_path/"po_jepa_full-resume1"/"run"/"job"
    write_json(orig/"job.json", dict(condition=linked_condition, dataset="double_well-2000", seed=1703, resume_from=None, training_run=None))
    write_json(orig/"preregistration.json", linked_registration or registration)
    write_json(orig/"training"/linked_condition/"report.json", dict(best=dict(step=10, checkpoint_provenance="p")))
    write_json(orig/"training"/linked_condition/"COMPLETE.json", {})
    write_json(resumed/"job.json", dict(condition="po_jepa_full", dataset="double_well-2000", seed=1703,
                                        resume_from=str(tmp_path/"po_jepa_full"), training_run=None))
    write_json(resumed/"preregistration.json", registration)
    return orig, tmp_path/"po_jepa_full-resume1"


def test_load_for_evaluation_follows_resume_from(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from po_jepa.runtime import workflow
    registration = dict(source="s", delivery_manifest_sha256="m")
    monkeypatch.setattr(v7jobs.comparison, "preregister", lambda: (dict(source="s"), None, None))
    monkeypatch.setattr(workflow, "verify_bundle", lambda: "m")
    monkeypatch.setattr(workflow, "verify_selection", lambda root, report: None)
    monkeypatch.setattr(v7jobs.data.datasets, "active", lambda: SimpleNamespace(tag="double_well-2000"))
    monkeypatch.setattr(v7jobs.checks, "model_seed", lambda: 1703)
    monkeypatch.setattr(v7jobs.comparison, "build_condition", lambda c, d: ("model", "loss"))
    monkeypatch.setattr(v7jobs.comparison, "make_optimizer", lambda m: None)
    seen = {}
    monkeypatch.setattr(v7jobs.evaluation, "load_selected_checkpoint", lambda c, s, **kw: seen.update(root=kw["root"], selected=s))
    orig, resumed = _resumed_layout(tmp_path, registration)
    _, _, selected, _ = v7jobs.load_for_evaluation("po_jepa_full", resumed, "cpu")
    assert seen["root"] == orig/"training"/"po_jepa_full" and selected["step"] == 10
    read = lambda p: json.loads(Path(p).read_text())             # noqa: E731
    job = v7jobs._job_dir(resumed)
    assert v7jobs._linked_training_root(job, read(job/"job.json"), "po_jepa_full", registration, read) == orig/"training"/"po_jepa_full"
    assert v7load.training_dir(job, read(job/"job.json"), "po_jepa_full") == orig/"training"/"po_jepa_full"   # 两种读法同一个目录
    for kwargs, message in ((dict(linked_registration=dict(source="old", delivery_manifest_sha256="m")), "预登记"),
                            (dict(linked_condition="po_jepa_no_consistency"), "另一个条件")):
        _, resumed = _resumed_layout(tmp_path/message, registration, **kwargs)
        with pytest.raises(ValueError, match=message):
            v7jobs.load_for_evaluation("po_jepa_full", resumed, "cpu")
