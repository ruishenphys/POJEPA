"""S13：DROID 汇总 `tools/droid_summary.py`（docs/design/13-summary.md §8；验收 14 §3 S13）。

假结果目录：两个种子、8 个训练条件和 2 个不训练的对照，评测任务按 S11a/S11b/S12 的文件格式手写（rows、frame_rate.json、
resolution.json、planning/{rank,cem,report}.json、report.json、COMPLETE.json），每个值是 (条件, 种子, 回合, 窗口, 键名) 的
哈希加上条件的偏移，期望值在测试里按定义直接循环算，不调被测的统计函数。训练任务和评测来源（格式等 S09）经假读取函数。
最后一个测试用 S11b 的合成数据根真跑 `evaluate_job` 和 `planning.run`，确认汇总读得懂真实产物。
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from po_jepa.droid import evaluate as ev
from po_jepa.droid import index as ix
from po_jepa.droid import planning as pl
from po_jepa.droid import prepare as pr
from po_jepa.droid import windows as wn
from po_jepa.droid.config import load_config
from po_jepa.runtime import cache
from po_jepa.runtime.addon import spectral_gap as sim_gap
from po_jepa.runtime import workflow as w
from test_droid_index import CONFIGS

spec = importlib.util.spec_from_file_location("droid_summary", w.ROOT/"tools"/"droid_summary.py")
ds = importlib.util.module_from_spec(spec)
sys.modules["droid_summary"] = ds          # dataclass 要在 sys.modules 里找到模块
spec.loader.exec_module(ds)
sa = ds.sa

BASE = load_config(CONFIGS/"pilot.yaml")
CFG = dataclasses.replace(BASE, training=dataclasses.replace(BASE.training, seeds=(1703, 1704)))
SEEDS = CFG.training.seeds
TRAINED = CFG.training.conditions
UNTRAINED = CFG.training.untrained
# 回合：id_test 两个场景（S1 两条、S2 三条，回合数不同才能区分按回合加权和按场景平均），ood_test 三条
EPISODES = {"id_test": {"A": ("LAB1", 11), "B": ("LAB1", 11), "C": ("LAB2", 12), "D": ("LAB2", 12), "E": ("LAB2", 12)},
            "ood_test": {"F": ("REAL", -1), "G": ("RPL", -1), "H": ("REAL", -1)}}
OUTCOME = {"A": "success", "B": "failure", "C": "success", "D": "success", "E": "failure",
           "F": "failure", "G": "success", "H": "success"}
# 条件偏移：po_jepa_full 0；no_consistency 差 2%（匹配）、其余差得多（不匹配）
OFFSET = {"po_jepa_full": 0., "po_jepa_no_consistency": .01, "fixed_fourier": .2, "koopman": .3,
          "matched_transformer": .4, "transformer_consistency": .41, "vjepa2_ac": .6, "vjepa2_ac_small": .7,
          "identity": 1., "vjepa2_ac_zeroshot": .9}
DURATIONS = sorted({*CFG.evaluation.durations_id, *CFG.evaluation.durations_heldout, *CFG.evaluation.durations_long})
SIGMA = .5
LAMBDA = {c: dict(time_weight=.01*(i + 1), space_weight=.02*(i + 1)) for i, c in enumerate(ds.LAMBDA_CONDITIONS)}


def uid(k: str) -> str:
    return f"LAB+0000000{k}+2023-07-07-09h-42m-23s"


def val(*key) -> float:
    """[0, 0.5) 里的确定性值。"""
    return int(hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:12], 16)/16**12/2


def value(c, s, u, wid, name) -> float:
    return .5 + OFFSET[c] + val(c, s, u, wid, name)


def seeds_of(c):
    return (None,) if c in UNTRAINED else SEEDS


def write(path: Path, body) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cache._json_bytes(body))


def windows_of(split):
    """每条回合每个时长两个窗口（起点 0、1）。"""
    out = []
    for k in EPISODES[split]:
        for d in DURATIONS:
            for s in (0, 1):
                out.append(dict(id=wn.window_id(uid(k), d, s), uuid=uid(k), duration=d, start=s, key=k))
    return out


def row_value(c, s, win, name):
    if name == "delta_act" and win["key"] == "A" and win["duration"] == 1 and win["start"] == 0:
        return None                                   # 替身回合被跳过
    return value(c, s, win["uuid"], win["id"], name)


def frame_entries(split):
    return [dict(id=wn.frame_rate_id(uid(k), 15*s), uuid=uid(k), key=k, start=s) for k in EPISODES[split] for s in (0, 1)]


def excluded(entry):
    return entry["key"] in ("B", "G") and entry["start"] == 0


def small(win):
    return win["key"] in ("C", "H") and win["duration"] == 4 and win["start"] == 1


def write_eval(run: Path, c, s, data_sha):
    folder = ds.eval_dir(run, c, s)
    report = dict(schema=ev.SCHEMA_REPORT, job=dict(condition=c, seed=s, config_sha256=CFG.config_sha256),
                  status="passed", axes=dict(windows=True, frame_rate=True, resolution=True), splits={})
    for split in ev.TEST_SPLITS:
        wins = windows_of(split)
        records = [dict(id=x["id"], uuid=x["uuid"], duration=x["duration"], start=x["start"], outcome=OUTCOME[x["key"]],
                        values={n: row_value(c, s, x, n) for n in ev.value_names(x["duration"], CFG)}, triangle_ok=True)
                   for x in wins]
        for i in range(0, len(records), 64):
            write(folder/split/"rows"/f"{i//64:05d}.json",
                  dict(schema=ev.SCHEMA_ROWS, split=split, batch=i//64, windows=records[i:i + 64]))
        names = ev.frame_rate_names(CFG)
        write(folder/split/ev.FRAME_RATE_FILE, dict(schema=ev.SCHEMA_FRAME_RATE, split=split, entries=[
            dict(id=e["id"], uuid=e["uuid"], start_frame=15*e["start"], s5_start_row=3*e["start"], outcome=OUTCOME[e["key"]],
                 excluded="control_out_of_bounds" if excluded(e) else None,
                 values={n: None if excluded(e) else value(c, s, e["uuid"], e["id"], n) for n in names})
            for e in frame_entries(split)]))
        names = ev.resolution_names(CFG)
        write(folder/split/ev.RESOLUTION_FILE, dict(schema=ev.SCHEMA_RESOLUTION, split=split, entries=[
            dict(id=x["id"], uuid=x["uuid"], duration=x["duration"], start=x["start"], outcome=OUTCOME[x["key"]],
                 values={n: value(c, s, x["uuid"], x["id"], "res" + n) for n in names})
            for x in wins if x["duration"] in CFG.evaluation.durations_id]))
        plan = [x for x in wins if x["duration"] in CFG.evaluation.planning_durations]
        head = dict(schema=pl.SCHEMA, split=split)
        write(folder/split/"planning"/"rank.json", dict(head, windows=[
            dict(id=x["id"], uuid=x["uuid"], duration=x["duration"], **{k: value(c, s, x["uuid"], x["id"], k) for k in ds.RANK_KEYS})
            for x in plan]))
        write(folder/split/"planning"/"cem.json", dict(head, windows=[
            dict(id=x["id"], uuid=x["uuid"], duration=x["duration"], small_displacement=small(x),
                 **{k: value(c, s, x["uuid"], x["id"], k) for k in ds.CEM_KEYS}) for x in plan]))
        write(folder/split/"planning"/"report.json", head)
        report["splits"][split] = dict(triangle=dict(violations=0, cases=[]), counts=dict(windows=len(wins)))
    write(folder/"job.json", report["job"])
    write_report(folder, report)


def write_report(folder, report):
    write(folder/"report.json", report)
    (folder/"COMPLETE.json").unlink(missing_ok=True)
    write(folder/"COMPLETE.json", dict(report_sha256=cache.file_sha(folder/"report.json")))


def gap_window(i):
    return dict(grids={str(g): dict(lambda_k=1., lambda_k1=1.5, cutoff_gap=.5, median_gap=.1 + .01*i, ratio=2. + .1*i)
                       for g in sim_gap.GRIDS})


def build(tmp_path, *, skip=()):
    """数据根和运行根；返回 (data, run_root, registry)。registry 是假读取函数要的训练和评测来源。"""
    data, run_root = tmp_path/"data", tmp_path/"runs"
    lines = []
    for split, eps in EPISODES.items():
        for k, (lab, scene) in eps.items():
            lines.append(json.dumps(dict(uuid=uid(k), lab=lab, scene_id=scene, outcome=OUTCOME[k]), sort_keys=True))
    (ix.index_dir(data)).mkdir(parents=True)
    (ix.index_dir(data)/ix.EPISODES_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(ix.index_dir(data)/ix.SPLITS_FILE, dict(fake="splits"))
    write(pr.prepared_dir(data)/pr.ACTION_SCALE_FILE, dict(fake="scale"))
    for s in ("val", *ev.TEST_SPLITS):
        write(pr.prepared_dir(data)/pr.WINDOWS_DIR/f"{s}.json", dict(split=s))
    ref = ds.data_reference(data)
    run = run_root/CFG.name
    for c in (*TRAINED, *UNTRAINED):
        for s in seeds_of(c):
            if (c, s) not in skip:
                write_eval(run, c, s, ref)
    write(run/"calibration"/"sigma_z.json", dict(sigma_z=SIGMA, config_sha256=CFG.config_sha256))
    for c, lam in LAMBDA.items():
        write(run/"calibration"/f"lambda-{c}.json", dict(lam, rho=1.))
    write(run/"check"/"env.json", dict(fake="env"))
    gap = dict(job=dict(config_sha256=CFG.config_sha256, sigma_z=SIGMA,
                        inputs=dict(windows_sha256={s: ref[f"windows/{s}.json"] for s in ev.TEST_SPLITS})),
               summary=sim_gap.summarize([gap_window(i) for i in range(4)]),
               by_split={s: sim_gap.summarize([gap_window(i) for i in range(2)]) for s in ev.TEST_SPLITS},
               fixed_fourier=dict(cutoff_gap=0.))
    folder = run/"eval"/"spectral-gap"
    write(folder/"job.json", gap["job"])
    write_report(folder, gap)
    trains, evals = {}, {}
    train_data = {k: ref[k] for k in ("splits.json", "windows/val.json", "action_scale.json")}
    eval_data = {k: ref[k] for k in ("splits.json", "windows/id_test.json", "windows/ood_test.json", "action_scale.json")}
    for c in TRAINED:
        for s in SEEDS:
            base = ds.INITIAL_SAME_AS.get(c, c)
            trains[(c, s)] = ds.TrainView(
                condition=c, seed=s, config_sha256=CFG.config_sha256, feature_manifest_sha256="f"*64, data_sha256=train_data,
                train_set_sha256="t"*64, sigma_z=SIGMA if c in ds.PO_CLASS else None,
                lambdas=LAMBDA.get(c), initial_model_sha256=f"init-{base}-{s}", updates=500, completed=500,
                log_ranges=((1, 100), (101, 200), (201, 500)), log_problem=None,
                best_checkpoint_sha256=f"ckpt-{c}-{s}",
                step_seconds=(.1 + OFFSET[c], .2 + OFFSET[c], .3 + OFFSET[c]), peak_memory_bytes=1000 + s,
                gpu_hours=.5, resumes=1, spectral_backend="cpu_batched",
                curves=dict(validation=[(0, 1.), (250, .8), (500, .7 + OFFSET[c])], train_jepa=[(1, 1.), (500, .5)]))
            write(run/"train"/c/f"seed{s}"/"COMPLETE.json", dict(fake=True))
            evals[(c, s)] = ds.EvalView(condition=c, seed=s, feature_manifest_sha256="f"*64, data_sha256=eval_data,
                                        checkpoint_sha256=f"ckpt-{c}-{s}")
    for c in UNTRAINED:
        evals[(c, None)] = ds.EvalView(condition=c, seed=None, feature_manifest_sha256="f"*64, data_sha256=eval_data,
                                       checkpoint_sha256=None,
                                       initial_model_sha256=f"init-vjepa2_ac-{SEEDS[0]}" if c == "vjepa2_ac_zeroshot" else None,
                                       construction_seed=SEEDS[0] if c == "vjepa2_ac_zeroshot" else None)
    return data, run_root, dict(trains=trains, evals=evals)


def readers(registry):
    def key(path):
        parts = Path(path).parts
        if parts[-1].startswith("seed"):
            return parts[-2], int(parts[-1][4:])
        return parts[-1], None
    return ds.Readers(train=lambda p: registry["trains"][key(p)], evaluation=lambda p: registry["evals"][key(p)],
                      env=lambda p: dict(source=Path(p).name))


def run_summary(data, run_root, registry):
    return ds.run(CFG, data, run_root, readers(registry))


# ---------------------------------------------------------------- 期望值（直接循环）

def expected_macro(c, s, split, names_durations, outcome=None, source="rows"):
    """回合宏平均：每条回合里 (窗口, 键名) 的值先平均，再在回合间平均。返回 (值, 回合数)。"""
    per = {}
    if source == "rows":
        for x in windows_of(split):
            k = x["key"]
            if outcome and OUTCOME[k] != outcome:
                continue
            for name, durations in names_durations:
                if x["duration"] in durations and name in ev.value_names(x["duration"], CFG):
                    v = row_value(c, s, x, name)
                    if v is not None:
                        per.setdefault(k, []).append(v)
    means = [sum(v)/len(v) for v in per.values()]
    return (sum(means)/len(means) if means else None), len(means)


def test_full_fake_run_outputs_every_table_and_figure(tmp_path):
    """14 §3 S13 第一条：假结果上出 D1–D11 全部表和图；数字对上按定义直接算的值。"""
    data, run_root, registry = build(tmp_path)
    summary, out = run_summary(data, run_root, registry)
    assert summary["status"] == "ok" and summary["failed"] == [] and summary["missing"] == []
    assert out == (run_root/CFG.name/"summary").resolve()
    assert json.loads((out/"COMPLETE.json").read_text())["summary_sha256"] == cache.file_sha(out/"summary.json")
    md = (out/"summary.md").read_text(encoding="utf-8")
    for t in ("D1 ", "D2 ", "D3 ", "D4 ", "D5 ", "D6 ", "D7 ", "D8 ", "D9 ", "D10 ", "D11 "):
        assert t in md
    assert "缺" not in md.split("## D10")[0].split("## id_test")[1].split("#### D6")[0]
    figs = sorted(p.name for p in (out/"figures").iterdir())
    assert figs == sorted([f"{s}-{k}.svg" for s in ev.TEST_SPLITS for k in ("scatter-tcd_ln", "scatter-scd_ln", "pd-vs-x", "frame-rate")]
                          + ["validation.svg", "train-jepa.svg"])
    ids = CFG.evaluation.durations_id
    for split in ev.TEST_SPLITS:
        t = summary["splits"][split]["by_outcome"]["all"]
        for c in ("po_jepa_full", "koopman", "identity"):
            per = {str(s): expected_macro(c, s, split, [("future:g=16", ids)])[0] for s in seeds_of(c)}
            got = t["D1"][c]["future_g16"]
            assert got["per_seed"] == {("untrained" if k == "None" else k): pytest.approx(v, abs=1e-15) for k, v in per.items()}
            assert got["mean"] == pytest.approx(statistics.fmean(per.values()), abs=1e-15)
            assert got["sd"] == (None if len(per) == 1 else pytest.approx(statistics.stdev(per.values()), abs=1e-15))
            scd = expected_macro(c, SEEDS[0] if c != "identity" else None, split,
                                 [("scd_ln:h=8", ids), ("scd_ln:h=12", ids)])[0]
            assert t["D1"][c]["scd_ln"]["per_seed"][str(SEEDS[0]) if c != "identity" else "untrained"] == pytest.approx(scd, abs=1e-15)
        # delta_act：替身被跳过的窗口不进
        assert t["D1"]["po_jepa_full"]["delta_act"]["windows"][str(SEEDS[0])] == len(
            [x for x in windows_of(split) if x["duration"] in ids]) - (split == "id_test")
        # AC 的输入尺度量不可比
        for c in ds.AC:
            assert t["D1"][c]["tcd_normalized"] == "not_comparable" and t["D8"][c]["epsilon_hat"] == "not_comparable"
            assert t["D1"][c]["tcd_ln"] != "not_comparable"
        # D2 留出时长 16 单列
        want = expected_macro("koopman", SEEDS[1], split, [("future:g=16", (16,))])[0]
        assert t["D2"]["koopman"]["future_g16:T=16"]["per_seed"][str(SEEDS[1])] == pytest.approx(want, abs=1e-15)
        # D4：排除的起点不进，个数记上
        fr = t["D4"]["po_jepa_full"]
        assert fr["excluded"] == 1 and fr["entries"] == 2*len(EPISODES[split])
        assert fr["frame_rate:single:f=5"]["windows"][str(SEEDS[0])] == 2*len(EPISODES[split]) - 1
        # D6：随机水平、< 1 cm 的窗口、相对误差不含它们、中位数
        p = t["D6"]
        assert p["random_level"] == pl.random_level(CFG.evaluation.planning_candidates)
        row = p["conditions"]["fixed_fourier"]["4"]
        assert row["small_displacement_windows"] == 1
        n4 = 2*len(EPISODES[split])
        assert row["position_relative"]["windows"][str(SEEDS[0])] == n4 - 1 and row["position_error_m"]["windows"][str(SEEDS[0])] == n4
        s0 = SEEDS[0]
        per_ep = {}
        for x in windows_of(split):
            if x["duration"] == 4:
                per_ep.setdefault(x["key"], []).append(value("fixed_fourier", s0, x["uuid"], x["id"], "gripper_error"))
        means = [statistics.fmean(v) for v in per_ep.values()]
        assert row["gripper_error"]["per_seed"][str(s0)] == pytest.approx(statistics.fmean(means), abs=1e-15)
        assert row["gripper_error"]["median"]["per_seed"][str(s0)] == pytest.approx(statistics.median(means), abs=1e-15)
    # 精度匹配：no_consistency 差 2% 匹配，fixed_fourier 不匹配，identity 不参与
    e = summary["precision"]["errors"]
    rows = {(r["reference"], r["metric"]): r for r in summary["splits"]["ood_test"]["by_outcome"]["all"]["D7"]}
    assert rows[("po_jepa_no_consistency", "future_g16")]["matched"] is True
    assert rows[("fixed_fourier", "future_g16")]["matched"] is False and rows[("identity", "future_g16")]["matched"] is None
    assert rows[("vjepa2_ac", "tcd_normalized")]["status"] == "not_comparable"
    assert rows[("vjepa2_ac", "pd_in")]["status"] == "not_comparable"
    assert e["fixed_fourier"] == summary["splits"]["id_test"]["by_outcome"]["all"]["D1"]["fixed_fourier"]["future_g16"]["mean"]
    # 配对差：full − identity 每个种子一份，回合内平均
    r = rows[("identity", "future_g16")]
    assert set(r["per_seed"]) == {str(s) for s in SEEDS}
    full = expected_macro("po_jepa_full", SEEDS[0], "ood_test", [("future:g=16", ids)])[0]
    ref = expected_macro("identity", None, "ood_test", [("future:g=16", ids)])[0]
    assert r["per_seed"][str(SEEDS[0])]["mean"] == pytest.approx(full - ref, abs=1e-14)   # 每条回合窗口数相同
    assert r["ci"]["clusters"] == 3 and r["seeds_full_lower"] == 2
    ri = {(x["reference"], x["metric"]): x for x in summary["splits"]["id_test"]["by_outcome"]["all"]["D7"]}
    assert ri[("identity", "future_g16")]["ci"]["clusters"] == 2 and ri[("identity", "future_g16")]["ci"]["episodes"] == 5
    # D8：ε̂ 和有效斜率
    d8 = summary["splits"]["id_test"]["by_outcome"]["all"]["D8"]["po_jepa_full"]
    s0 = SEEDS[0]
    tcdn = [row_value("po_jepa_full", s0, x, "tcd_normalized") for x in windows_of("id_test") if x["duration"] in ids]
    eps = statistics.median(tcdn)
    assert d8["epsilon_hat"]["per_seed"][str(s0)] == pytest.approx(eps, abs=1e-15)
    pd = expected_macro("po_jepa_full", s0, "id_test", [("pd_in:3+5", (8,))])[0]
    slope = [r for r in d8["partitions"] if r["partition"] == "3+5"][0]["effective_slope"]
    assert slope["per_seed"][str(s0)] == pytest.approx(pd/(15*eps), rel=1e-14)
    assert d8["triangle_violations"] == 0
    # D10、D11
    assert summary["spectral_gap"]["summary"]["keep_sentence"] is True and summary["spectral_gap"]["fixed_fourier_cutoff_gap"] == 0.
    cost = summary["cost"]["koopman"]
    assert cost == dict(jobs=2, median_step_seconds=pytest.approx(.2 + OFFSET["koopman"]), peak_memory_bytes=1000 + SEEDS[1],
                        gpu_hours=1., resumes=2, spectral_backend=["cpu_batched"])
    assert summary["check_env"] == {"source": "env.json"}


def test_missing_tasks_are_listed_and_marked(tmp_path):
    """缺的任务进 missing.json、表里写“缺”，不影响退出码；某个评测任务没做的轴单独列。"""
    data, run_root, registry = build(tmp_path, skip={("koopman", SEEDS[1]), ("vjepa2_ac_zeroshot", None)})
    run = run_root/CFG.name
    folder = ds.eval_dir(run, "matched_transformer", SEEDS[0])
    shutil.rmtree(folder/"ood_test"/"planning")
    (run/"calibration"/"lambda-fixed_fourier.json").unlink()
    del registry["trains"][("fixed_fourier", SEEDS[1])]
    (run/"train"/"fixed_fourier"/f"seed{SEEDS[1]}"/"COMPLETE.json").unlink()
    registry["evals"].pop(("fixed_fourier", SEEDS[1]))
    shutil.rmtree(ds.eval_dir(run, "fixed_fourier", SEEDS[1]))
    summary, out = run_summary(data, run_root, registry)
    # λ 文件缺而 fixed_fourier seed1703 的训练还在：λ 核对读不到算不通过（13 §4 的规则），其余照常
    assert summary["failed"] == ["λ"]
    assert not (out/"COMPLETE.json").exists() and not (out/"figures").exists()
    for item in ("eval/koopman/seed1704", "eval/vjepa2_ac_zeroshot", "eval/fixed_fourier/seed1704", "train/fixed_fourier/seed1704",
                 "calibrate/lambda-fixed_fourier", f"eval/matched_transformer/seed{SEEDS[0]}:ood_test/planning"):
        assert item in summary["missing"]
    assert json.loads((out/"missing.json").read_text())["missing"] == summary["missing"]
    # 删掉还在的那份 fixed_fourier 训练，λ 核对就只剩缺：全部通过
    del registry["trains"][("fixed_fourier", SEEDS[0])]
    (run/"train"/"fixed_fourier"/f"seed{SEEDS[0]}"/"COMPLETE.json").unlink()
    registry["evals"].pop(("fixed_fourier", SEEDS[0]))
    shutil.rmtree(ds.eval_dir(run, "fixed_fourier", SEEDS[0]))
    summary, out = run_summary(data, run_root, registry)
    assert summary["status"] == "ok" and (out/"COMPLETE.json").exists()
    t = summary["splits"]["ood_test"]["by_outcome"]["all"]
    assert t["D1"]["koopman"]["future_g16"]["seeds"] == 1 and t["D1"]["fixed_fourier"]["future_g16"]["mean"] is None
    assert t["D6"]["conditions"]["matched_transformer"]["1"]["mean_rank"]["seeds"] == 1
    md = (out/"summary.md").read_text(encoding="utf-8")
    assert "| fixed_fourier | 缺 |" in md
    assert (out.parent/"summary.previous1").exists()


# ---------------------------------------------------------------- 13 §8.3 的核对：每项各造一个不一致

def _rewrite_report(run, c, s, change):
    folder = ds.eval_dir(run, c, s)
    report = json.loads((folder/"report.json").read_text())
    change(report)
    (folder/"report.json").unlink()
    write_report(folder, report)


def _drop_window(run, c, s):
    path = ds.eval_dir(run, c, s)/"ood_test"/"rows"/"00000.json"
    body = json.loads(path.read_text())
    body["windows"] = body["windows"][1:]
    path.unlink()
    write(path, body)


def _bad_outcome(run, c, s):
    path = ds.eval_dir(run, c, s)/"id_test"/"rows"/"00000.json"
    body = json.loads(path.read_text())
    body["windows"][0]["outcome"] = "failure" if body["windows"][0]["outcome"] == "success" else "success"
    path.unlink()
    write(path, body)


def _gap(run, change):
    folder = run/"eval"/"spectral-gap"
    report = json.loads((folder/"report.json").read_text())
    change(report)
    (folder/"report.json").unlink()
    write_report(folder, report)


def _train(reg, c, s, **kw):
    reg["trains"][(c, s)] = dataclasses.replace(reg["trains"][(c, s)], **kw)


def _eval(reg, c, s, **kw):
    reg["evals"][(c, s)] = dataclasses.replace(reg["evals"][(c, s)], **kw)


S0, S1 = 1703, 1704
CASES = {
    "配置": [lambda run, reg: _rewrite_report(run, "koopman", S0, lambda r: r["job"].update(config_sha256="x"*64)),
           lambda run, reg: _train(reg, "vjepa2_ac", S1, config_sha256="x"*64),
           lambda run, reg: _gap(run, lambda r: r["job"].update(config_sha256="x"*64))],
    "特征": [lambda run, reg: _eval(reg, "identity", None, feature_manifest_sha256="e"*64),
           lambda run, reg: _train(reg, "koopman", S1, feature_manifest_sha256="e"*64)],
    "划分和窗口": [lambda run, reg: _train(reg, "koopman", S0, data_sha256={**reg["trains"][("koopman", S0)].data_sha256,
                                                                       "windows/val.json": "v"*64}),
               lambda run, reg: _eval(reg, "vjepa2_ac", S1, data_sha256={**reg["evals"][("vjepa2_ac", S1)].data_sha256,
                                                                          "action_scale.json": "a"*64}),
               lambda run, reg: _eval(reg, "vjepa2_ac", S1, data_sha256={**reg["evals"][("vjepa2_ac", S1)].data_sha256,
                                                                          "windows/test.json": "a"*64}),
               lambda run, reg: _gap(run, lambda r: r["job"]["inputs"]["windows_sha256"].update(id_test="w"*64))],
    "训练集": [lambda run, reg: _train(reg, "matched_transformer", S1, train_set_sha256="u"*64)],
    "σ_z": [lambda run, reg: _train(reg, "fixed_fourier", S1, sigma_z=SIGMA*(1 + 1e-12)),
            lambda run, reg: _gap(run, lambda r: r["job"].update(sigma_z=.25)),
            lambda run, reg: (run/"calibration"/"sigma_z.json").unlink()],
    "λ": [lambda run, reg: _train(reg, "transformer_consistency", S0, lambdas=dict(time_weight=.03, space_weight=.07)),
          lambda run, reg: _train(reg, "po_jepa_full", S1, lambdas=None)],
    "初始权重": [lambda run, reg: _train(reg, "fixed_fourier", S1, initial_model_sha256="i"*64),
             lambda run, reg: _train(reg, "transformer_consistency", S0, initial_model_sha256="i"*64),
             lambda run, reg: _eval(reg, "vjepa2_ac_zeroshot", None, initial_model_sha256="i"*64)],
    "训练日志": [lambda run, reg: _train(reg, "koopman", S0, log_ranges=((1, 100), (102, 500))),
             lambda run, reg: _train(reg, "koopman", S0, log_ranges=((1, 100), (100, 500))),
             lambda run, reg: _train(reg, "koopman", S0, log_ranges=((1, 100), (101, 400)), completed=400),
             lambda run, reg: _train(reg, "koopman", S0, completed=499),
             lambda run, reg: _train(reg, "koopman", S0, log_problem="train-log.jsonl does not cover the steps contiguously")],
    "评测": [lambda run, reg: _eval(reg, "vjepa2_ac_small", S1, checkpoint_sha256="c"*64),
           lambda run, reg: _rewrite_report(run, "po_jepa_no_consistency", S0,
                                            lambda r: r["splits"]["ood_test"]["triangle"].update(violations=1)),
           lambda run, reg: _rewrite_report(run, "identity", None, lambda r: r.update(status="failed"))],
    "评测窗口": [lambda run, reg: _drop_window(run, "koopman", S1),
             lambda run, reg: _bad_outcome(run, "vjepa2_ac", S0)],
}


def test_each_consistency_check_fails_on_its_own(tmp_path):
    """14 §3 S13：13 §8.3 的每一项（另加“配置”“评测窗口”）各造一个不一致的例子，只有这一项报错；核对不过时只写
    summary.json（blocked）、checks.json、missing.json 和 summary.md，不出图、不写 COMPLETE.json，命令退出 1。"""
    assert set(CASES) == set(ds.CHECK_ITEMS)
    data, run_root, registry = build(tmp_path/"base")
    clean = ds.summarize(CFG, ds.collect(CFG, data, run_root, readers(registry)))
    assert clean["status"] == "ok"
    for item, cases in CASES.items():
        for k, case in enumerate(cases):
            d, rr, reg = build(tmp_path/f"case{list(CASES).index(item)}-{k}")
            case(rr/CFG.name, reg)
            got = ds.collect(CFG, d, rr, readers(reg))
            summary = ds.summarize(CFG, got)
            assert summary["failed"] == [item], (item, k, [c for c in summary["checks"] if not c["ok"]])
    # 命令行（pilot.yaml 只有种子 1703）：退出 1，只写核对文件
    d, rr, reg = build(tmp_path/"cli")
    _train(reg, "matched_transformer", S0, train_set_sha256="u"*64)
    assert ds.main(["--config", str(CONFIGS/"pilot.yaml"), "--data-root", str(d), "--run-root", str(rr)], readers(reg)) == 1
    path = rr/CFG.name/"summary"
    assert sorted(p.name for p in path.iterdir()) == ["checks.json", "missing.json", "summary.json", "summary.md"]
    body = json.loads((path/"summary.json").read_text())
    assert body["status"] == "blocked" and body["failed"] == ["训练集"] and "splits" not in body
    assert json.loads((path/"checks.json").read_text())["checks"] == body["checks"]


# ---------------------------------------------------------------- 自助法（13 §3、§8.2）

def _independent_ci(per_seed, cluster_of):
    ids = sorted(set.intersection(*[set(d) for d in per_seed]))
    clusters = sorted({cluster_of[u] for u in ids})
    draws = torch.randint(len(clusters), (2000, len(clusters)), generator=torch.Generator().manual_seed(1703)).tolist()
    boots = []
    for row in draws:
        seed_means = []
        for d in per_seed:
            values = [d[u] for j in row for u in ids if cluster_of[u] == clusters[j]]
            seed_means.append(math.fsum(values)/len(values))
        boots.append(sum(seed_means)/len(seed_means))
    q = torch.quantile(torch.tensor(boots, dtype=torch.float64), torch.tensor([.025, .975], dtype=torch.float64)).tolist()
    return q


def test_cluster_bootstrap_by_scene_and_by_episode():
    """14 §3 S13：按场景、按回合两种整群都对。按回合（每群一条）和仿真的 bootstrap_ci 相同；按场景时每个场景回合数相同就
    等于对场景均值做仿真的自助法；回合数不同时等于独立循环重算（抽到的群里按回合加权）。只用所有种子共有的回合。"""
    rng = np.random.default_rng(3)
    eps = [f"e{i:02d}" for i in range(12)]
    per_seed = [{u: float(rng.normal()) for u in eps} for _ in range(3)]
    per_seed[1]["extra"] = 9.                                       # 只在一个种子里的回合不参加
    by_episode = {u: u for u in eps + ["extra"]}
    got = ds.cluster_bootstrap_ci(per_seed, by_episode)
    want = sa.bootstrap_ci(per_seed)
    assert got["low"] == pytest.approx(want["low"], abs=1e-12) and got["high"] == pytest.approx(want["high"], abs=1e-12)
    assert got["clusters"] == got["episodes"] == 12
    # 4 个场景各 3 条
    scene = {u: f"s{i//3}" for i, u in enumerate(eps)} | {"extra": "s9"}
    got = ds.cluster_bootstrap_ci(per_seed, scene)
    means = [{f"s{k}": statistics.fmean(d[u] for u in eps[3*k:3*k + 3]) for k in range(4)} for d in per_seed]
    want = sa.bootstrap_ci(means)
    assert got["low"] == pytest.approx(want["low"], abs=1e-12) and got["high"] == pytest.approx(want["high"], abs=1e-12)
    assert got["clusters"] == 4 and got["episodes"] == 12
    # 场景大小不同（1、2、4、5）
    sizes = [1, 2, 4, 5]
    scene = {u: f"s{k}" for k, n in enumerate(sizes) for u in eps[sum(sizes[:k]):sum(sizes[:k + 1])]}
    scene["extra"] = "s0"
    got = ds.cluster_bootstrap_ci(per_seed, scene)
    low, high = _independent_ci(per_seed, scene)
    assert got["low"] == pytest.approx(low, abs=1e-12) and got["high"] == pytest.approx(high, abs=1e-12)
    assert got["low"] < got["high"]
    assert ds.cluster_bootstrap_ci([], scene) is None and ds.cluster_bootstrap_ci([{"a": 1.}, {"b": 1.}], {"a": "x", "b": "y"}) is None


# ---------------------------------------------------------------- 成败分开、逐字节相同

def test_outcome_halves_recombine_by_episode_count(tmp_path):
    """14 §3 S13：成败分开的两份按回合数加权后等于合并的那份（每个种子、每个量）。"""
    data, run_root, registry = build(tmp_path)
    summary, _ = run_summary(data, run_root, registry)
    checked = 0
    for split in ev.TEST_SPLITS:
        t = summary["splits"][split]["by_outcome"]
        for table in ("D1", "D2", "D3", "D5"):
            for c, row in t["all"][table].items():
                for m, st in row.items():
                    if not isinstance(st, dict) or "per_seed" not in st:
                        continue
                    for seed, v in st["per_seed"].items():
                        parts = [(t[o][table][c][m]["per_seed"][seed], t[o][table][c][m]["episodes"][seed]) for o in ds.OUTCOMES]
                        n = st["episodes"][seed]
                        assert n == sum(k for _, k in parts)
                        assert v*n == pytest.approx(sum(x*k for x, k in parts if k), rel=1e-12)
                        checked += 1
    assert checked > 500


def test_repeated_runs_are_byte_identical(tmp_path):
    """14 §3 S13：同样的输入运行两次，输出逐字节相同。统一入口每次都重算汇总（S11-P1-1 修复），所以内容不变时保留原输出、
    不再多一份 summary.previousN；产物变了（这里少了一个评测任务）才把旧的改名为 summary.previous1 再换上新的。"""
    data, run_root, registry = build(tmp_path)
    run_summary(data, run_root, registry)
    out = run_root/CFG.name/"summary"
    first = tmp_path/"first"
    shutil.copytree(out, first)
    run_summary(data, run_root, registry)
    assert not (out.parent/"summary.previous1").exists()
    files = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file()) and len(files) == 15
    for f in files:
        assert (first/f).read_bytes() == (out/f).read_bytes(), f
    registry["evals"].pop(("koopman", SEEDS[1]))
    shutil.rmtree(ds.eval_dir(run_root/CFG.name, "koopman", SEEDS[1]))
    summary, _ = run_summary(data, run_root, registry)
    assert f"eval/koopman/seed{SEEDS[1]}" in summary["missing"]
    previous = out.parent/"summary.previous1"
    assert all((first/f).read_bytes() == (previous/f).read_bytes() for f in files)
    assert (out/"summary.json").read_bytes() != (first/"summary.json").read_bytes()


# ---------------------------------------------------------------- 真实评测产物（S11a、S11b、S12）

from test_droid_evaluate import LOSS, Identity, Toy  # noqa: E402
from test_droid_evaluate_extra import CFG as EXTRA_CFG, OUTCOME as EXTRA_OUTCOME, SPLIT_ROWS, root, spec as control  # noqa: E402,F401
from po_jepa.droid import batches as bt  # noqa: E402


def test_reads_the_real_evaluation_and_planning_files(root, tmp_path):
    """用 S11b 的合成数据根真跑 identity 和 po_jepa_full（玩具模型）的 `evaluate_job` 与两个测试集的 `planning.run`，
    汇总读出的按时长未来误差、原始帧率、规划名次等于这些任务自己报告里的回合宏平均。"""
    cfg = EXTRA_CFG                   # 只跑 po_jepa_full 和 identity，其余条件是缺
    data, run_root = tmp_path/"data", tmp_path/"runs"
    lines = [json.dumps(dict(uuid=ev_u, lab="LAB", scene_id=i, outcome=o), sort_keys=True)
             for i, (ev_u, o) in enumerate(sorted(root["outcomes"].items()))]
    ix.index_dir(data).mkdir(parents=True)
    (ix.index_dir(data)/ix.EPISODES_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(ix.index_dir(data)/ix.SPLITS_FILE, dict(fake="splits"))
    write(pr.prepared_dir(data)/pr.ACTION_SCALE_FILE, dict(fake="scale"))
    docs = {s: json.loads(json.dumps(d)) for s, d in root["docs"].items()}
    for s in ("val", *ev.TEST_SPLITS):
        write(pr.prepared_dir(data)/pr.WINDOWS_DIR/f"{s}.json", docs.get(s, dict(split=s)))
    ref = ds.data_reference(data)
    run = run_root/cfg.name
    kwargs = dict(splits=docs, prepared=root["prepared"], outcomes=root["outcomes"], reader=root["reader"],
                  control_spec=control(), provenance=root["provenance"], loss_config=LOSS, config=cfg, device="cpu",
                  extra_reader=root["extra"])
    reports = {}
    for c, s, model in (("po_jepa_full", 1703, Toy()), ("identity", None, Identity())):
        folder = ds.eval_dir(run, c, s)
        reports[c] = ev.evaluate_job(folder, model=model, condition=c, seed=s, job={"checkpoint": None}, **kwargs)
        for split in ev.TEST_SPLITS:
            _, kept, _ = ev._selected(docs[split], root["prepared"], "realized")
            inputs = pl.PlanningInputs(docs[split], root["reader"], bt.episode_arrays(root["prepared"], kept, "realized"),
                                       control(), cfg, LOSS)
            reports[(c, split)] = pl.run(model, inputs, folder/split/"planning", "cpu")
    write(run/"calibration"/"sigma_z.json", dict(sigma_z=SIGMA))
    write(run/"calibration"/"lambda-po_jepa_full.json", dict(time_weight=.1, space_weight=.2))
    eval_data = {k: ref[k] for k in ("splits.json", "windows/id_test.json", "windows/ood_test.json", "action_scale.json")}
    reg = dict(trains={("po_jepa_full", 1703): ds.TrainView(
        condition="po_jepa_full", seed=1703, config_sha256=cfg.config_sha256, feature_manifest_sha256="f"*64,
        data_sha256={k: ref[k] for k in ("splits.json", "action_scale.json")}, train_set_sha256="t"*64, sigma_z=SIGMA,
        lambdas=dict(time_weight=.1, space_weight=.2), initial_model_sha256="i", updates=10, completed=10,
        log_ranges=((1, 10),), log_problem=None, best_checkpoint_sha256="k", step_seconds=(.1,), peak_memory_bytes=None, gpu_hours=None,
        resumes=0, spectral_backend=None, curves={})},
        evals={("po_jepa_full", 1703): ds.EvalView("po_jepa_full", 1703, "f"*64, eval_data, "k"),
               ("identity", None): ds.EvalView("identity", None, "f"*64, eval_data, None)})
    write(run/"train"/"po_jepa_full"/"seed1703"/"COMPLETE.json", {})
    summary, out = ds.run(cfg, data, run_root, readers(reg))
    assert summary["status"] == "ok", summary["checks"]
    assert {"eval/po_jepa_full/seed1703", "eval/identity", "train/po_jepa_full/seed1703"}.isdisjoint(summary["missing"])
    assert {"eval/koopman/seed1703", "eval/vjepa2_ac_zeroshot", "eval/spectral-gap", "check/env"} <= set(summary["missing"])
    assert not any(":" in m for m in summary["missing"])          # 两个任务的各轴都齐
    for split in ev.TEST_SPLITS:
        t = summary["splits"][split]["by_outcome"]
        for c, s in (("po_jepa_full", "1703"), ("identity", "untrained")):
            macro = reports[c]["splits"][split]["macro"]
            for d in cfg.evaluation.durations_id:
                for o, label in ((None, "all"), ("success", "success"), ("failure", "failure")):
                    want = macro[str(d)]["future:g=16"][label]["episode_macro_mean"]
                    got = t["all" if o is None else o]["D1_by_duration"][c][f"future_g16:T={d}"]["per_seed"][s]
                    assert got == pytest.approx(want, abs=1e-12)
            fr = reports[c]["splits"][split]["frame_rate"]["macro"]
            for name, entry in fr.items():
                assert t["all"]["D4"][c][f"frame_rate:{name}"]["per_seed"][s] == pytest.approx(entry["all"]["episode_macro_mean"], abs=1e-12)
            plan = reports[(c, split)]["by_duration"]
            for d in cfg.evaluation.planning_durations:
                for key in ds.RANK_KEYS:
                    assert t["all"]["D6"]["conditions"][c][str(d)][key]["per_seed"][s] == pytest.approx(plan[str(d)][key], abs=1e-12)
                assert t["all"]["D6"]["conditions"][c][str(d)]["position_error_m"]["per_seed"][s] == pytest.approx(
                    plan[str(d)]["position_error_m"]["mean"], abs=1e-12)
    ident = summary["splits"]["id_test"]["by_outcome"]["all"]["D6"]["conditions"]["identity"]
    assert ident["1"]["mean_rank"]["mean"] == pytest.approx(pl.random_level(cfg.evaluation.planning_candidates)["mean_rank"])
