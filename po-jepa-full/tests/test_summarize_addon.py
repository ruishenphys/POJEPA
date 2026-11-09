"""S18b：tools/summarize_addon.py（13 §1–§7）。统计的手算值、精度匹配的边界、自助法、8 项一致性核对、输出的换名，
以及在一棵造出来的结果树上从头跑一遍。"""

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from po_jepa.runtime import cache
from po_jepa.runtime import workflow as w
from po_jepa.runtime.addon import d4

spec = importlib.util.spec_from_file_location("summarize_addon", w.ROOT/"tools"/"summarize_addon.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)

SCENE, UPDATES, TIDS, STARTS = "double_well-200", 2000, (180, 181, 182), (0, 10)
CAL = dict(rho=1., sigma_z=.5, time_weight=.03, space_weight=.01, source="calibrate")
ADDON_CAL = dict(fixed_fourier=dict(rho=1., time_weight=.02, space_weight=.04, source="x"),
                 transformer_consistency=dict(rho=1., time_weight=.05, space_weight=.06, source="x"),
                 sigma_z_used=.5, v7_calibration_sha256="0"*64)
VARIANT = "double_well~gain=2-200"
IDENTITY = dict(dataset_count=200, manifest_sha256_by_id={str(t): f"m{t}" for t in TIDS})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cache._json_bytes(value))


def complete(folder, report):
    write(folder/"report.json", report)
    write(folder/"COMPLETE.json", dict(report_sha256=cache.file_sha(folder/"report.json")))


def queries():
    out = [dict(kind="point", trajectory_id=t, start_tick=s, duration_cycles=d, grid_size=g)
           for t in TIDS for s in STARTS for g in (16, 8, 12) for d in (1., 2., 3., 4., 8., 12.)]
    return out+[dict(kind="pair", trajectory_id=t, start_tick=s, duration_cycles=d, grid_size=g)
                for t in TIDS for s in STARTS for g in (8, 12) for d in (1., 2., 3., 4., 8., 12.)]


def level(condition, seed):
    """每个 (条件, 种子) 的误差水平：po_jepa_full 1.0、其余按名单顺序加 0.1，种子 1704 再加 0.02。"""
    return 1.+.1*sa.CONDITIONS.index(condition)+(.02 if seed == 1704 else 0.)


def future_offset(grid, duration):
    """只有 16×16、T ∈ {1, 2, 4, 8} 的单点进精度和 T1；留出时长 +0.5、12×12 +0.7、两者同时 +0.9（T2），8×8 的单点 +5（不进任何表）。"""
    if grid == 8:
        return 5.
    return {(16, False): 0., (16, True): .5, (12, False): .7, (12, True): .9}[(grid, duration in (3., 12.))]


def run_job(root, condition, seed, *, lam=None, sigma=.5, init="init-full", sequence="seq", rows_shift=0., transfer=None):
    """v7 的 D 任务（transfer 给出测试数据集 tag 时是 V3 变体评测任务：kind "transfer"，没有训练目录）。"""
    folder = (root/"transfer"/transfer if transfer else root/SCENE)/f"seed{seed}"/condition/"run"/"job"
    job = dict(condition=condition, dataset=SCENE, seed=seed, updates=UPDATES, resume_from=None, training_run=None)
    if transfer:
        job.update(kind="transfer", dataset=transfer, train_dataset=SCENE)
    write(folder/"job.json", job)
    base = level(condition, seed)+rows_shift
    for i, q in enumerate(queries()):
        values = dict(future_jepa_distance=base*(1+.01*(q["trajectory_id"]-180))+.001*q["start_tick"]+future_offset(q["grid_size"], q["duration_cycles"]),
                      tcd_normalized=.2*base, delta_act=.01) if q["kind"] == "point" else dict(scd_normalized=.3*base)
        write(folder/"evaluation"/condition/"rows"/f"{i:05d}.json", dict(query=q, values=values, status="ok", condition=condition, seed=seed))
    write(folder/"evaluation"/condition/"registration.json", dict(queries=queries(), condition=condition))
    write(folder/"figures"/"curves.json", dict(condition=condition, steps=[1, 2], losses=dict(jepa=[1., .5]),
                                               validation=[dict(step=0, future_jepa_distance=1.), dict(step=2, future_jepa_distance=.5)]))
    best = None
    if condition != "identity" and not transfer:
        train = folder/"training"/condition
        (train/"checkpoints").mkdir(parents=True)
        tw, sw = lam if lam is not None else dict(
            po_jepa_full=(CAL["time_weight"], CAL["space_weight"]), po_jepa_time_only=(CAL["time_weight"], 0.),
            po_jepa_space_only=(0., CAL["space_weight"]), fixed_fourier=(.02, .04), transformer_consistency=(.05, .06)).get(condition, (0., 0.))
        torch.save(dict(model_config=dict(graph=dict(sigma_z=sigma)), loss_config=dict(time_weight=tw, space_weight=sw)),
                   train/"checkpoints"/"step-0002.pt")
        best = dict(step=2, mean=.5, checkpoint="checkpoints/step-0002.pt")
        for k, seconds in enumerate((.3, .1, .2) if seed == 1703 else (.25, .9, .95)):
            write(train/"updates"/f"{k+1:04d}.json", dict(update_and_observer_seconds=seconds,
                                                         gpu_memory=dict(peak_allocated=100*(k+1), peak_reserved=0, allocated=0, reserved=0)))
        complete(train, dict(best=best, query_sequence_sha256=f"{sequence}-{seed}", completed_updates=3))
    initial = dict(po_jepa_time_only=init, po_jepa_space_only=init, fixed_fourier=init, po_jepa_full="init-full",
                   transformer_consistency="init-mt", matched_transformer="init-mt").get(condition, f"init-{condition}")
    complete(folder, dict(best=best, job=job, preflight=dict(cache_identity=IDENTITY), initial_model_sha256=initial,
                          registration_sha256="r", training={}, evaluation={}))
    return folder


def d4_values(condition, seed, tid):
    base = level(condition, seed)*(1+.01*(tid-180))
    # 留出时长（T = 3、12）的键 ×5、scd_ln 的 12×12 ×3：T1 只取 T ∈ {1, 2, 4, 8}，scd_ln 两个网格一起平均
    values = {n: base*(5. if n.endswith((":T=3", ":T=12")) else 3. if n.startswith("scd_ln:h=12:") else 1.)*.1 for n in d4.value_names()}
    for t, parts in d4.PARTITIONS:
        values[f"pd_in:T={t:g}:{d4.partition_label(parts)}"] = base*.001*d4.partition_x(t, parts)
    for t, _ in d4.HELDOUT:
        values[f"heldout_direct:T={t:g}"], values[f"heldout_partition:T={t:g}"], values[f"heldout_rollout:T={t:g}"] = base, .1, base
    return values


# 等分 tcd_normalized 在 T = 1、2、4、8 上的值（每个窗口相同）：中位数 0.25；再除以 (T/2)² 的话中位数是 0.13125
TCD_NORMALIZED = {"tcd_normalized:T=1": .2, "tcd_normalized:T=2": .25, "tcd_normalized:T=4": .25, "tcd_normalized:T=8": .4,
                  "tcd_normalized:T=3": 9., "tcd_normalized:T=12": 9.}


def d4_job(root, condition, seed, *, passed=True, manifest=None):
    folder = root/"d4"/SCENE/f"seed{seed}"/condition/"run"/"job"
    job = dict(kind="d4", condition=condition, dataset=SCENE, seed=seed, updates=UPDATES)
    write(folder/"job.json", job)
    index = 0
    for t in TIDS:
        for s in STARTS:
            eq = {n: dict(new=TCD_NORMALIZED.get(n, .5), old=0., within_tolerance=True) for n in d4.equivalence_names()}
            write(folder/"rows"/f"{index:05d}.json", dict(index=index, trajectory_id=t, start_tick=s, values=d4_values(condition, seed, t),
                                                         equivalence=eq, triangle_ok={"T=3": True, "T=12": True},
                                                         manifest_sha256=manifest or f"m{t}"))
            index += 1
        write(folder/"long"/f"{t:04d}.json", dict(trajectory_id=t, start_tick=0, manifest_sha256=f"m{t}",
                                                  values={n: level(condition, seed) for n in d4.long_value_names()}))
    metrics = {m: dict(count=1, bitwise_equal=1, max_abs_diff=1e-9 if m == "scd_normalized" else 0., max_rel_diff=0., outside_tolerance=0)
               for m in d4.EQUIVALENCE_METRICS}
    complete(folder, dict(job=job, status="passed" if passed else "failed", equivalence=dict(passed=passed, metrics=metrics),
                          triangle=dict(violations=0, cases=[])))
    write(folder/"source.json", dict(job_dir="x"))
    return folder


def gap_job(root):
    folder = root/"spectral-gap"/SCENE/"run"/"job"
    write(folder/"job.json", dict(kind="spectral_gap", dataset=SCENE, updates=UPDATES, k=16))
    grids = {str(g): dict(windows=6, undefined_ratio=0, ratio=[1., 2., 3., 4., 5.], median_ratio=3., median_ratio_above_1=True) for g in (16, 8, 12)}
    complete(folder, dict(windows=[dict(trajectory_id=t, start_tick=s, manifest_sha256=f"m{t}") for t in TIDS for s in STARTS],
                          summary=dict(grids=grids, keep_sentence=True, decision="keep"), fixed_fourier=dict(cutoff_gap=0.)))


@pytest.fixture
def tree(tmp_path):
    v7, addon, cal = tmp_path/"v7", tmp_path/"addon", tmp_path/"cal"
    for seed in (1703, 1704):
        for c in sa.V7_CONDITIONS:
            run_job(v7, c, seed)
        for c in sa.ADDON_CONDITIONS:
            run_job(addon, c, seed)
        for c in sa.CONDITIONS:
            d4_job(addon, c, seed)
    gap_job(addon)
    run_job(v7, "po_jepa_full", 1703, transfer=VARIANT, rows_shift=.05)
    write(addon/"encoder-identity.json", dict(kind="encoder_identity", verdict="same", encoder=dict(sha256="e"), ac_checkpoint=dict(sha256="a")))
    write(v7/"plan"/"double_well"/"po_jepa_full.json", dict(kind="planning", dataset=SCENE, train_dataset=SCENE, condition="po_jepa_full",
                                                         seed=1703, cem=dict(samples=256), episodes=[{}]*4, model=dict(success_rate=.5, mean_error=1.)))
    cal.mkdir()
    source = tmp_path/"v7tree"/f"{SCENE}-u{UPDATES}.yaml"
    source.parent.mkdir()
    source.write_text(yaml.safe_dump(CAL))
    (cal/f"{SCENE}-u{UPDATES}.yaml").write_text(source.read_text())
    (cal/"addon").mkdir()
    (cal/"addon"/f"{SCENE}-u{UPDATES}.yaml").write_text(yaml.safe_dump(ADDON_CAL))
    write(cal/"IMPORTED.json", dict(schema="addon_v7_calibration_import_v1", files={f"{SCENE}-u{UPDATES}.yaml": dict(
        source=str(source), sha256=cache.file_sha(source), imported_at="2026-11-13T08:00:00+00:00", v7_commit="0ac45fe03bf43b96bf861446b8aa2ada4f4a8987")}))   # S17 §4.5
    return v7, addon, cal, tmp_path


def run_main(tree, name="summary"):
    v7, addon, cal, tmp = tree
    return sa.main(["--v7-results", str(v7), "--addon-results", str(addon), "--output", str(tmp/name), "--calibration-root", str(cal)])


# ---------- 统计的手算值 ----------

def test_macro_is_trajectory_first_and_seed_stats():
    values = {(1, 0): 0., (1, 5): 0., (1, 10): 3., (2, 0): 4.}
    assert sa.by_trajectory(values) == {1: 1., 2: 4.} and sa.macro(values) == 2.5        # 不是 7/4
    s = sa.seed_stats({1703: 1., 1704: 2., 1705: 4.})
    assert s["mean"] == pytest.approx(7/3) and s["sd"] == pytest.approx(math.sqrt(((1-7/3)**2+(2-7/3)**2+(4-7/3)**2)/2)) and s["seeds"] == 3
    assert sa.seed_stats({1703: 1.}) == dict(mean=1., sd=None, seeds=1, per_seed={"1703": 1.})
    assert sa.seed_stats({1703: None})["mean"] is None


def test_paired_and_matching_rules():
    full = {(1, 0): 1., (1, 5): 2., (2, 0): 5., (3, 0): 1.}
    other = {(1, 0): 2., (1, 5): 2., (2, 0): 4.}
    diffs = sa.paired_seed(full, other)
    assert diffs == {1: -.5, 2: 1.}                                            # 轨迹 3 没有配对，不算
    s = sa.paired_stats(diffs)
    assert s == dict(mean=.25, se=pytest.approx(math.sqrt(1.125)/math.sqrt(2)), trajectories=2, full_lower=1)
    assert sa.matched(20., 21.) == (True, 5.) and sa.matched(21., 20.) == (True, 5.)   # 恰好 5% 算匹配
    assert sa.matched(20., 21.000001)[0] is False and sa.matched(None, 1.) == (None, None)
    assert sa.MATCH_TOLERANCE == .05


def test_bootstrap_shares_draws_across_seeds():
    ci = sa.bootstrap_ci([{1: 3., 2: 3., 3: 3.}, {1: 3., 2: 3., 3: 3.}])
    assert (ci["low"], ci["high"], ci["reps"], ci["trajectories"], ci["seed"]) == (3., 3., 2000, 3, 1703)
    ci = sa.bootstrap_ci([{1: 1., 2: 0., 3: 0.}, {1: -1., 2: 0., 3: 0.}])       # 同一组抽样：每次的种子平均恰为 0
    assert ci["low"] == 0. and ci["high"] == 0.
    a, b = {1: 1., 2: 2., 3: 4., 4: 8.}, {1: .5, 2: 1., 3: 0., 4: 2., 9: 100.}   # 轨迹 9 只在一个种子里，不抽
    ci = sa.bootstrap_ci([a, b])
    g = torch.Generator().manual_seed(1703)
    draws = torch.randint(4, (2000, 4), generator=g).tolist()
    ids = [1, 2, 3, 4]
    boot = torch.tensor([(sum(a[ids[i]] for i in d)/4+sum(b[ids[i]] for i in d)/4)/2 for d in draws], dtype=torch.float64)
    low, high = torch.quantile(boot, torch.tensor([.025, .975], dtype=torch.float64)).tolist()
    assert ci["low"] == pytest.approx(low, abs=1e-12) and ci["high"] == pytest.approx(high, abs=1e-12) and ci["trajectories"] == 4
    assert sa.bootstrap_ci([{1: 1.}, {2: 1.}]) is None


# ---------- 找任务 ----------

def test_collect_rejects_two_completed_copies(tmp_path):
    run_job(tmp_path/"a", "identity", 1703)
    resumed = tmp_path/"a"/SCENE/"seed1703"/"identity-resume1"/"run"/"job"
    (resumed).mkdir(parents=True)
    write(resumed/"job.json", read_json(tmp_path/"a"/SCENE/"seed1703"/"identity"/"run"/"job"/"job.json"))
    assert len(sa.collect(tmp_path/"a")) == 1                                   # 没完成的续训目录不算
    complete(resumed, {})
    with pytest.raises(ValueError, match="completed twice"):
        sa.collect(tmp_path/"a")
    for args in (["--v7-results", str(tmp_path/"none"), "--addon-results", str(tmp_path/"a")],
                 ["--v7-results", str(tmp_path/"a"), "--addon-results", str(tmp_path/"none")]):
        with pytest.raises(SystemExit) as error:                               # 结果根不是目录：参数错误，什么都不写
            sa.main([*args, "--output", str(tmp_path/"out")])
        assert error.value.code == 2 and not (tmp_path/"out").exists()


def read_json(path):
    return json.loads(Path(path).read_text())


# ---------- 从头跑一遍 ----------

def test_main_writes_complete_summary(tree):
    v7, addon, cal, tmp = tree
    assert run_main(tree) == 0
    out = tmp/"summary"
    assert sorted(p.name for p in out.iterdir()) == ["checks.json", "figures", "missing.json", "summary.json", "summary.md"]
    s = read_json(out/"summary.json")
    scene = s["scenes"][f"{SCENE}-u{UPDATES}"]
    assert scene["status"] == "ok" and all(c["ok"] for c in s["checks"][f"{SCENE}-u{UPDATES}"])
    # 误差：轨迹 180/181/182 乘 1、1.01、1.02，起点 0、10 加 0、0.01 → 轨迹宏平均 = level·1.01 + 0.005；两个种子平均
    for c in sa.CONDITIONS:
        expected = sum(level(c, seed)*1.01+.005 for seed in (1703, 1704))/2
        assert scene["precision"][c] == pytest.approx(expected, rel=1e-12)
        assert scene["in_distribution"][c]["future"]["mean"] == pytest.approx(expected, rel=1e-12)
        assert [scene["heldout"][c][m]["mean"] for m in sa.T2_METRICS] == pytest.approx([expected+.5, expected+.7, expected+.9], rel=1e-12)
        d4_level = sum(level(c, seed)*1.01 for seed in (1703, 1704))/2               # D4 的值不加起点项
        assert scene["in_distribution"][c]["tcd_ln"]["mean"] == pytest.approx(.1*d4_level, rel=1e-12)
        assert scene["in_distribution"][c]["scd_ln"]["mean"] == pytest.approx(.2*d4_level, rel=1e-12)   # (0.1 + 0.3)/2
    assert scene["in_distribution"]["vjepa2_ac"]["tcd_normalized"] == "not_comparable"
    rows = {(r["reference"], r["metric"]): r for r in scene["comparisons"]}
    r = rows[("po_jepa_no_consistency", "future")]
    e_full, e_nc = scene["precision"]["po_jepa_full"], scene["precision"]["po_jepa_no_consistency"]
    assert r["matched"] is False and r["error_difference_percent"] == pytest.approx((e_nc-e_full)/e_full*100, rel=1e-12)
    assert 9.8 < r["error_difference_percent"] < 9.9
    assert r["mean"] == pytest.approx(-.1*1.01, rel=1e-9) and r["seeds_full_lower"] == 2
    assert r["ci"]["low"] == pytest.approx(-.1*1.01, rel=.02) and r["ci"]["reps"] == 2000
    assert rows[("identity", "future")]["matched"] is None and rows[("vjepa2_ac", "scd_normalized")]["status"] == "not_comparable"
    p = scene["partitions"]["po_jepa_full"]
    assert p["epsilon_hat"]["mean"] == pytest.approx(.25)                    # median(.2, .25, .25, .4)；留出时长 3、12 不进 ε̂
    row = next(x for x in p["partitions"] if x["partition"] == "3+5")
    assert row["x"] == 15. and row["effective_slope"]["mean"] == pytest.approx((1.+1.02)/2*1.01*.001/.25, rel=1e-9)
    assert scene["decomposition"]["po_jepa_full"]["triangle_violations"] == 0
    assert scene["equivalence_max_abs_diff"] == 1e-9
    # 两个种子的 6 条更新记录合在一起取中位数：(.25 + .3)/2；各自的中位数 .2、.9 再取中位数会是 .55
    assert scene["cost"]["po_jepa_full"] == dict(jobs=2, updates=6, median_seconds=pytest.approx(.275), peak_allocated_bytes=300)
    assert s["encoder_identity"]["verdict"] == "same" and s["planning"][0]["samples"] == 256
    assert s["spectral_gap_decision"] == "keep the original sentence (median ratio > 1 on every grid)"
    assert read_json(out/"missing.json") == dict(missing=[])
    # T8：变体的行比基础场景高 0.05·(1 + 0.01·(轨迹 − 180))，轨迹间平均 0.0505
    assert scene["transfer"] == [dict(variant=VARIANT, seed=1703, condition="po_jepa_full", matched=None, error_difference_percent=None,
                                      mean=pytest.approx(.0505, rel=1e-9), se=pytest.approx(.05*.01/math.sqrt(3), rel=1e-6),
                                      trajectories=3, full_lower=0)]
    # 分割差异和长时程也进配对表（13 §2）；AC 的输入尺度量不可比（10 §3.1）
    assert {m for _, m in rows} == set(sa.PAIRED_METRICS) and len(scene["comparisons"]) == len(sa.REFERENCES)*len(sa.PAIRED_METRICS)
    r = rows[("po_jepa_no_consistency", "pd_in")]                            # pd_in = level·0.001·x，7 个分割、3 条轨迹一起平均
    xs = [d4.partition_x(t, parts) for t, parts in d4.PARTITIONS]
    assert r["mean"] == pytest.approx(-.1*1.01*.001*sum(xs)/len(xs), rel=1e-9) and r["matched"] is False
    r = rows[("po_jepa_no_consistency", "long_direct")]                      # 长时程每条轨迹 = level，和轨迹无关
    assert r["mean"] == pytest.approx(-.1, rel=1e-9) and r["ci"]["low"] == pytest.approx(-.1, rel=1e-9)
    assert rows[("vjepa2_ac", "pd_in")]["status"] == rows[("vjepa2_ac_small", "xpd_in")]["status"] == "not_comparable"
    assert rows[("vjepa2_ac", "pd_ln")]["status"] == "ok"
    md = (out/"summary.md").read_text()
    for table in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11"):
        assert f"### {table} " in md or f"## {table} " in md, table
    assert {p.name for p in (out/"figures").iterdir()} == {f"{SCENE}-u{UPDATES}-{n}.svg" for n in
                                                          ("scatter-tcd_ln", "scatter-scd_ln", "pd-vs-x", "validation", "train-jepa")}


def test_rerun_keeps_previous_and_is_identical(tree):
    v7, addon, cal, tmp = tree
    assert run_main(tree) == 0 and run_main(tree) == 0 and run_main(tree) == 0
    assert (tmp/"summary.previous1").is_dir() and (tmp/"summary.previous2").is_dir() and not (tmp/"summary.previous3").exists()
    files = lambda root: {q.relative_to(root).as_posix(): q.read_bytes() for q in sorted(root.rglob("*")) if q.is_file()}
    assert files(tmp/"summary") == files(tmp/"summary.previous1") == files(tmp/"summary.previous2")    # 所有文件逐字节相同
    assert len(files(tmp/"summary")) == 9
    assert not [p for p in tmp.iterdir() if p.name.startswith(".summary.")]


def test_missing_jobs_are_listed(tree):
    v7, addon, cal, tmp = tree
    import shutil
    shutil.rmtree(addon/"d4"/SCENE/"seed1704"/"koopman")
    (addon/"encoder-identity.json").unlink()
    assert run_main(tree) == 0
    missing = read_json(tmp/"summary"/"missing.json")["missing"]
    assert dict(kind="d4", dataset=SCENE, updates=UPDATES, seed=1704, condition="koopman") in missing
    assert any(m["kind"] == "encoder_identity" for m in missing) and len(missing) == 2
    assert "缺" in (tmp/"summary"/"summary.md").read_text()


CORRUPTIONS = [            # (核对项, 重造的任务种类, 条件, 种子, 造法)
    ("features", "d4", "koopman", 1703, dict(manifest="other")),
    ("training_query_sequence", "run", "koopman", 1704, dict(sequence="other")),
    ("sigma_z", "run", "fixed_fourier", 1703, dict(sigma=.6)),
    ("lambda", "run", "po_jepa_time_only", 1704, dict(lam=(.031, 0.))),
    ("lambda", "run", "transformer_consistency", 1703, dict(lam=(.03, .01))),
    ("initial_weights", "run", "po_jepa_space_only", 1703, dict(init="other")),
    ("equivalence", "d4", "identity", 1704, dict(passed=False)),
    ("evaluation_queries", None, "identity", 1703, None),
    ("imported_calibration", None, "source", None, None),     # v7 目录里的标定在导入后被改
    ("imported_calibration", None, "copy", None, None),       # 导入的副本被改（内容读出来相同，只多一行注释）
    ("imported_calibration", None, "schema", None, None),     # 不是 S17 §4.5 的记录
    ("imported_calibration", None, "missing", None, None),    # 没有 IMPORTED.json
]


@pytest.mark.parametrize("item,kind,condition,seed,change", CORRUPTIONS)
def test_each_consistency_check_blocks_the_scene(tree, item, kind, condition, seed, change):
    v7, addon, cal, tmp = tree
    import shutil
    root = v7 if condition in sa.V7_CONDITIONS and kind == "run" else addon
    if kind == "run":
        shutil.rmtree(root/SCENE/f"seed{seed}"/condition)
        run_job(root, condition, seed, **change)
    elif kind == "d4":
        shutil.rmtree(addon/"d4"/SCENE/f"seed{seed}"/condition)
        d4_job(addon, condition, seed, **change)
    elif item == "evaluation_queries":
        write(v7/SCENE/f"seed{seed}"/condition/"run"/"job"/"evaluation"/condition/"registration.json", dict(queries=queries()[:-1]))
    elif condition in ("source", "copy"):
        path = cal.parent/"v7tree"/f"{SCENE}-u{UPDATES}.yaml" if condition == "source" else cal/f"{SCENE}-u{UPDATES}.yaml"
        path.write_text(path.read_text()+"# changed\n")
    elif condition == "schema":
        record = read_json(cal/"IMPORTED.json")
        write(cal/"IMPORTED.json", dict(record, schema="other"))
    else:
        (cal/"IMPORTED.json").unlink()
    assert run_main(tree) == 1
    s = read_json(tmp/"summary"/"summary.json")
    scene = f"{SCENE}-u{UPDATES}"
    assert s["scenes"][scene]["status"] == "blocked" and s["scenes"][scene]["failed"] == [item]
    assert [c["item"] for c in s["checks"][scene] if not c["ok"]] == [item]
    assert not list((tmp/"summary"/"figures").iterdir())
    md = (tmp/"summary"/"summary.md").read_text()
    assert "本场景不出表" in md and "### T1" not in md


def test_duplicate_completed_job_stops_before_writing(tree):
    v7, addon, cal, tmp = tree
    import shutil
    shutil.copytree(v7/SCENE/"seed1703"/"identity", v7/SCENE/"seed1703"/"identity-resume1")
    with pytest.raises(ValueError, match="completed twice"):
        run_main(tree)
    assert not (tmp/"summary").exists()
