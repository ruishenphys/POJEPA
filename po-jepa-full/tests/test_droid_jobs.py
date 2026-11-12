"""S14：DROID 任务图、统一入口、本地运行器（docs/design/12-droid-jobs.md §2–§5.1、§5.3；验收 14 §3 S14）。

期望的任务、依赖、完成标志、资源和 priority 在本文件里按 12 §2/§4 的表逐条写死，不调被测函数当参照。
运行器的测试用假命令（tests 里临时写的 python 脚本）：按行为文件决定退出码、写完成标志、记下开始/结束时刻和
CUDA_VISIBLE_DEVICES。
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import droid  # noqa: E402
import droid_jobs  # noqa: E402
import droid_run  # noqa: E402
from po_jepa.droid.config import load_config  # noqa: E402

# 运行器的锁（fcntl.flock）只在 Linux/macOS 有；Windows 上跳过真起运行器的测试，其余照常收集、运行（R4 审核）
POSIX_RUNNER = pytest.mark.skipif(sys.platform == "win32", reason="droid_run.py runner lock is Linux/macOS only")

CONFIGS = ROOT/"configs"/"droid"
DATA, RUNS = "/data/droid-1.0.1", "/runs/po-jepa-droid"
CONDITIONS = ["po_jepa_full", "po_jepa_no_consistency", "fixed_fourier", "koopman",
              "matched_transformer", "transformer_consistency", "vjepa2_ac", "vjepa2_ac_small"]
LAMBDA = ["po_jepa_full", "fixed_fourier", "transformer_consistency"]
PRIORITY = {"vjepa2_ac": 1, "po_jepa_full": 2, "po_jepa_no_consistency": 2, "fixed_fourier": 2,
            "vjepa2_ac_small": 2, "matched_transformer": 3, "transformer_consistency": 3, "koopman": 3}
# base.yaml resources（12 §2/§4：sigma-z、spectral-gap 用 stats 的）
RES = {"index": (0, 32, 16, 120), "download": (0, 16, 8, 600), "prepare": (0, 8, 32, 240), "stats": (0, 8, 32, 60),
       "encode": (1, 8, 64, 600), "check": (1, 8, 64, 120), "calibrate": (1, 8, 64, 240),
       "train": (1, 8, 96, 1440), "evaluate": (1, 8, 96, 600), "summary": (0, 4, 16, 60)}


def expected(name: str, train_shards: int, seeds: list[int]) -> dict[str, tuple[list[str], str, str]]:
    """12 §2 的表：编号 → (依赖, 完成标志, 资源键)。"""
    shards = [f"train-{k:03d}" for k in range(train_shards)] + ["val", "id_test", "ood_test"]
    run = f"{RUNS}/{name}"
    jobs = {"index": ([], f"{DATA}/index/MANIFEST.json", "index")}
    for s in shards:
        jobs[f"download/{s}"] = (["index"], f"{DATA}/raw/{s}/_complete/{name}.json", "download")
        jobs[f"prepare/{s}"] = ([f"download/{s}"], f"{DATA}/prepared/{s}/_complete/{name}.json", "prepare")
        cleanup = "_raw_deleted" if s.startswith("train-") else "_complete"   # 训练片删完原始文件才算完成（S14-fix1）
        jobs[f"encode/{s}"] = ([f"prepare/{s}"], f"{DATA}/features/{s}/{cleanup}/{name}.json", "encode")
    jobs["prepare/stats"] = ([f"prepare/{s}" for s in shards], f"{DATA}/prepared/_complete/stats-{name}.json", "stats")
    jobs["encode/test-extra"] = (["prepare/id_test", "prepare/ood_test"],
                                 f"{DATA}/features/test_extra/_complete/{name}.json", "encode")
    jobs["calibrate/sigma-z"] = (["prepare/stats"] + [f"encode/{s}" for s in shards],
                                 f"{run}/calibration/sigma_z.json", "stats")
    jobs["check/env"] = (["calibrate/sigma-z"], f"{run}/check/env.json", "check")
    for c in LAMBDA:
        jobs[f"calibrate/lambda-{c}"] = (["calibrate/sigma-z"], f"{run}/calibration/lambda-{c}.json", "calibrate")
    for c in CONDITIONS:
        for seed in seeds:
            jobs[f"train/{c}/seed{seed}"] = (["check/env"] + ([f"calibrate/lambda-{c}"] if c in LAMBDA else []),
                                             f"{run}/train/{c}/seed{seed}/COMPLETE.json", "train")
            jobs[f"eval/{c}/seed{seed}"] = ([f"train/{c}/seed{seed}", "encode/test-extra"],
                                            f"{run}/eval/{c}/seed{seed}/COMPLETE.json", "evaluate")
    for c in ("identity", "vjepa2_ac_zeroshot"):
        jobs[f"eval/{c}"] = (["check/env", "encode/test-extra"], f"{run}/eval/{c}/COMPLETE.json", "evaluate")
    jobs["eval/spectral-gap"] = (["calibrate/sigma-z"], f"{run}/eval/spectral-gap/COMPLETE.json", "stats")
    jobs["summary"] = ([j for j in jobs if j.startswith("eval/")], f"{run}/summary/COMPLETE.json", "summary")
    return jobs


def graph_of(name: str, config=None, **kw) -> dict:
    path = f"configs/droid/{name}.yaml"
    return droid_jobs.build_graph(config or load_config(ROOT/path), path, DATA, RUNS, **kw)


def level_of(jobs: list[dict]) -> dict[str, int]:
    deps = {j["id"]: j["deps"] for j in jobs}
    level: dict[str, int] = {}
    for j in jobs:                       # jobs 按拓扑序：依赖都已算过
        level[j["id"]] = 1 + max((level[d] for d in deps[j["id"]]), default=-1)
    return level


# ---------------------------------------------------------------- 任务图

@pytest.mark.parametrize("name,train_shards,seeds,count", [("pilot", 1, [1703], 40),
                                                            ("full", 20, [1703, 1704, 1705], 129)])
def test_graph_matches_the_12_table(name, train_shards, seeds, count):
    graph = graph_of(name)
    want = expected(name, train_shards, seeds)
    assert len(graph["jobs"]) == count == len(want)
    assert graph["schema"] == "droid_jobs_v1" and graph["config"] == name
    assert graph["config_sha256"] == load_config(CONFIGS/f"{name}.yaml").config_sha256
    assert (graph["data_root"], graph["run_root"]) == (DATA, RUNS)
    by_id = {j["id"]: j for j in graph["jobs"]}
    assert set(by_id) == set(want)
    level = level_of(graph["jobs"])                 # KeyError 就说明不是拓扑序
    for job_id, (deps, complete, res) in want.items():
        j = by_id[job_id]
        stage = job_id.split("/")[0]
        assert sorted(j["deps"]) == sorted(deps), job_id
        assert j["complete"] == complete, job_id
        assert j["stage"] == stage
        assert j["resources"] == dict(zip(("gpus", "cpus", "mem_gb", "time_limit_minutes"), RES[res])), job_id
        assert j["requeue_exit_codes"] == [75]
        argv = ["python", "tools/droid.py", stage, "--config", f"configs/droid/{name}.yaml", "--data-root", DATA,
                "--run-root", RUNS, "--job", job_id]
        if stage == "encode" and job_id.startswith("encode/train-"):
            argv.append("--delete-raw")
        if stage in ("train", "eval"):
            argv += ["--time-limit-minutes", str(RES[res][3])]
        assert j["command"] == argv, job_id
        if stage == "train":
            assert j["priority"] == PRIORITY[job_id.split("/")[1]], job_id
        else:
            assert j["priority"] == level[job_id], job_id
    ids = [j["id"] for j in graph["jobs"]]
    assert ids == sorted(ids, key=lambda i: (level[i], i))       # 拓扑层次，同层按编号
    assert by_id["encode/train-000"]["priority"] == 3 and level["summary"] == 8
    assert len({j["complete"] for j in graph["jobs"]}) == count


def test_cli_is_byte_identical_and_list_sections_are_layers(tmp_path, capsys):
    args = ["--config", "configs/droid/full.yaml", "--data-root", DATA, "--run-root", RUNS]
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        for k in (1, 2):
            assert droid_jobs.main(args + ["--output", str(tmp_path/f"g{k}.json")]) == 0
        assert droid_jobs.main(args + ["--format", "list", "--output", str(tmp_path/"l.txt")]) == 0
        assert droid_jobs.main(args) == 0                         # 标准输出
    finally:
        os.chdir(cwd)
    one, two = (tmp_path/"g1.json").read_bytes(), (tmp_path/"g2.json").read_bytes()
    assert one == two and capsys.readouterr().out.encode("utf-8") == one
    graph = json.loads(one)
    assert graph == graph_of("full")
    sections, current = [], None
    for line in (tmp_path/"l.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("# ====="):
            current = []
            sections.append(current)
        elif line and not line.startswith("#"):
            current.append(shlex.split(line))
    level = level_of(graph["jobs"])
    assert len(sections) == max(level.values()) + 1 == 9
    by_command = {tuple(j["command"]): j for j in graph["jobs"]}
    seen = []
    for k, section in enumerate(sections):
        ids = [by_command[tuple(c)]["id"] for c in section]
        assert sorted(ids) == sorted(i for i, v in level.items() if v == k)
        assert all(d in seen for i in ids for d in by_command_id(graph, i)["deps"])    # 依赖都在前面的段
        seen += ids
    assert len(seen) == len(graph["jobs"]) == 129


def by_command_id(graph: dict, job_id: str) -> dict:
    return next(j for j in graph["jobs"] if j["id"] == job_id)


def test_shards_from_splits_and_use_failures(tmp_path):
    cfg = load_config(CONFIGS/"pilot.yaml")
    small = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, train_episodes=5, shard_size=3))
    order = [f"u{i}" for i in range(9)]
    splits = {"train_order": order, "train_failures": ["u1", "u2"], "val": ["v"], "id_test": ["i"], "ood_test": ["o"]}
    assert droid_jobs.shard_names(small, tmp_path) == ["train-000", "train-001", "val", "id_test", "ood_test"]
    assert droid_jobs.shard_names(small, tmp_path, splits) == ["train-000", "train-001", "val", "id_test", "ood_test"]
    off = dataclasses.replace(small, data=dataclasses.replace(small.data, use_failures=False))
    with pytest.raises(ValueError, match="must exist"):
        droid_jobs.shard_names(off, tmp_path)
    # 去掉 u1、u2 后前 5 条是 u0 u3 u4 u5 u6：落在完整顺序的第 0、1、2 片
    assert droid_jobs.shard_names(off, tmp_path, splits) == ["train-000", "train-001", "train-002",
                                                            "val", "id_test", "ood_test"]
    graph = droid_jobs.build_graph(off, "c.yaml", DATA, RUNS, splits=splits)
    assert [j["id"] for j in graph["jobs"] if j["stage"] == "download"] == [
        "download/id_test", "download/ood_test", "download/train-000", "download/train-001", "download/train-002",
        "download/val"]
    bad = dict(splits, train_order=order[:4])                    # 片号和 ⌈N/shard_size⌉ 对不上就报错
    with pytest.raises(ValueError):
        droid_jobs.shard_names(small, tmp_path, bad)


def _jobs(*specs):
    return [{"id": i, "deps": list(d), "complete": f"/f/{i}"} for i, d in specs]


def test_graph_checks():
    ok = _jobs(("a", []), ("download/x", ["a"]), ("prepare/x", ["download/x"]), ("encode/x", ["prepare/x"]))
    droid_jobs.check_graph(ok, ["x"])
    with pytest.raises(ValueError, match="unknown"):
        droid_jobs.check_graph(ok + _jobs(("z", ["nope"])), ["x"])
    with pytest.raises(ValueError, match="cycle"):
        droid_jobs.check_graph(_jobs(("a", ["b"]), ("b", ["a"])), [])
    dup = ok[:3] + [dict(ok[3], complete="/f/a")]
    with pytest.raises(ValueError, match="not distinct"):
        droid_jobs.check_graph(dup, ["x"])
    with pytest.raises(ValueError, match="lacks"):
        droid_jobs.check_graph(ok[:3], ["x"])
    with pytest.raises(ValueError, match="duplicate"):
        droid_jobs.check_graph(ok + ok[:1], ["x"])
    with pytest.raises(ValueError, match="order"):
        droid_jobs.check_graph([ok[1], ok[0]] + ok[2:], ["x"])


# ---------------------------------------------------------------- 统一入口

def _entry(tmp_path, job, *extra):
    return ["download" if job.startswith("download") else job.split("/")[0], "--config", str(CONFIGS/"pilot.yaml"),
            "--data-root", str(tmp_path/"data"), "--run-root", str(tmp_path/"runs"), "--job", job, *extra]


def test_entry_dispatch(tmp_path, monkeypatch, capsys):
    calls = []

    def fake(stage, write=True, code=0):
        def main(argv):
            calls.append((stage, argv))
            if write:
                job = argv[argv.index("--shard") + 1] if "--shard" in argv else None
                path = {"index": tmp_path/"data/index/MANIFEST.json",
                        "download": tmp_path/f"data/raw/{job}/_complete/pilot.json",
                        "prepare": (tmp_path/"data/prepared/_complete/stats-pilot.json" if job == "stats"
                                    else tmp_path/f"data/prepared/{job}/_complete/pilot.json"),
                        "encode": (tmp_path/"data/features/test_extra/_complete/pilot.json" if job == "test-extra"
                                   else tmp_path/f"data/features/{job}/_raw_deleted/pilot.json"
                                   if (job or "").startswith("train-")
                                   else tmp_path/f"data/features/{job}/_complete/pilot.json")}[stage]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            return code
        return main

    for stage in ("index", "download", "prepare", "encode"):
        monkeypatch.setitem(droid.MAINS, stage, fake(stage))
    cfg, data = str(CONFIGS/"pilot.yaml"), str(tmp_path/"data")
    assert droid.main(_entry(tmp_path, "index")) == 0
    assert droid.main(_entry(tmp_path, "download/val")) == 0
    assert droid.main(_entry(tmp_path, "prepare/stats")) == 0
    assert droid.main(_entry(tmp_path, "encode/train-000", "--delete-raw")) == 0
    assert droid.main(_entry(tmp_path, "encode/test-extra", "--fake-encoder")) == 0
    assert calls == [("index", ["--config", cfg, "--data-root", data]),
                     ("download", ["--config", cfg, "--data-root", data, "--shard", "val"]),
                     ("prepare", ["--config", cfg, "--data-root", data, "--shard", "stats"]),
                     ("encode", ["--config", cfg, "--data-root", data, "--shard", "train-000", "--device", "cuda",
                                 "--delete-raw"]),
                     ("encode", ["--config", cfg, "--data-root", data, "--shard", "test-extra", "--device", "cuda",
                                 "--fake-encoder"])]
    calls.clear()
    assert droid.main(_entry(tmp_path, "download/val")) == 0          # 完成标志已在：什么都不做
    assert calls == [] and "already complete" in capsys.readouterr().out
    monkeypatch.setitem(droid.MAINS, "download", fake("download", write=False))
    assert droid.main(_entry(tmp_path, "download/id_test")) == 1      # 返回 0 却没写完成标志
    monkeypatch.setitem(droid.MAINS, "download", fake("download", write=False, code=75))
    assert droid.main(_entry(tmp_path, "download/ood_test")) == 75    # 退出码原样传出
    calls.clear()
    # S09b：calibrate、check、eval、summary 都接上了；各阶段收到的参数（不写完成标志，入口按失败退出 1）
    for stage in ("calibrate", "check", "eval", "summary"):
        monkeypatch.setitem(droid.MAINS, stage, fake(stage, write=False))
    run = ["--config", cfg, "--data-root", data, "--run-root", str(tmp_path/"runs")]
    assert droid.main(_entry(tmp_path, "calibrate/lambda-po_jepa_full")) == 1
    assert droid.main(_entry(tmp_path, "check/env")) == 1
    assert droid.main(_entry(tmp_path, "eval/koopman/seed1703", "--time-limit-minutes", "30")) == 1
    assert droid.main(_entry(tmp_path, "summary")) == 1
    assert calls == [("calibrate", run + ["--job", "calibrate/lambda-po_jepa_full", "--device", "cuda"]),
                     ("check", run + ["--device", "cuda"]),
                     ("eval", run + ["--job", "eval/koopman/seed1703", "--device", "cuda", "--time-limit-minutes", "30"]),
                     ("summary", run)]
    assert droid.PENDING == {}
    calls.clear()
    # S11-P1-1、RF-P2-4：完成标志在时，calibrate 什么都不做；check、train、eval、summary 照常调用阶段，由阶段核身份
    flags = {"calibrate/lambda-po_jepa_full": tmp_path/"runs/pilot/calibration/lambda-po_jepa_full.json",
             "check/env": tmp_path/"runs/pilot/check/env.json",
             "train/koopman/seed1703": tmp_path/"runs/pilot/train/koopman/seed1703/COMPLETE.json",
             "eval/koopman/seed1703": tmp_path/"runs/pilot/eval/koopman/seed1703/COMPLETE.json",
             "eval/identity": tmp_path/"runs/pilot/eval/identity/COMPLETE.json",
             "summary": tmp_path/"runs/pilot/summary/COMPLETE.json"}
    for job, path in flags.items():
        assert path == droid_jobs.complete_path(job, load_config(CONFIGS/"pilot.yaml"), tmp_path/"data", tmp_path/"runs")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    monkeypatch.setitem(droid.MAINS, "train", fake("train", write=False))
    for job in flags:
        assert droid.main(_entry(tmp_path, job)) == 0
    assert [c[0] for c in calls] == ["check", "train", "eval", "eval", "summary"]
    assert set(droid_jobs.RECHECKED_STAGES) == {"check", "train", "eval", "summary"} == set(droid_run.RECHECKED_STAGES)
    calls.clear()
    for bad in (_entry(tmp_path, "download/train-001"),                            # 试跑只有 train-000
                _entry(tmp_path, "encode/val", "--delete-raw"),                     # 只有训练片带 --delete-raw
                _entry(tmp_path, "download/val", "--time-limit-minutes", "5"),
                ["prepare"] + _entry(tmp_path, "download/val")[1:]):               # 阶段和编号不符
        with pytest.raises((SystemExit, ValueError)):
            droid.main(bad)
    assert calls == []


# ---------------------------------------------------------------- 本地运行器（假命令）

FAKE = r'''
import json, os, sys, time
job, flag, state = sys.argv[1], sys.argv[2], sys.argv[3]
with open(os.path.join(state, "events.jsonl"), "a") as f:
    f.write(json.dumps({"job": job, "event": "start", "t": time.time(),
                        "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\n")
plan_path = os.path.join(state, "plan.json")
plan = json.load(open(plan_path)).get(job, [0]) if os.path.exists(plan_path) else [0]
count_path = os.path.join(state, job.replace("/", "__") + ".count")
n = int(open(count_path).read()) if os.path.exists(count_path) else 0
open(count_path, "w").write(str(n + 1))
action = plan[min(n, len(plan) - 1)]
print(f"fake {job} attempt {n + 1} action {action}", flush=True)
if action == "hang":
    open(os.path.join(state, job.replace("/", "__") + ".hanging"), "w").write("1")
    time.sleep(120)
if isinstance(action, float):
    time.sleep(action)
    action = 0
with open(os.path.join(state, "events.jsonl"), "a") as f:
    f.write(json.dumps({"job": job, "event": "end", "t": time.time()}) + "\n")
if action == 0:
    os.makedirs(os.path.dirname(flag), exist_ok=True)
    open(flag, "w").write("{}")
if action == "noflag":
    action = 0
sys.exit(action)
'''


def fake_graph(tmp_path, specs):
    """specs: (编号, 依赖, gpus, cpus, mem_gb, priority)。"""
    state = tmp_path/"state"
    state.mkdir(exist_ok=True)
    (tmp_path/"fake_job.py").write_text(FAKE)
    jobs = []
    for job_id, deps, gpus, cpus, mem, prio in specs:
        flag = f"runs/fake/flags/{job_id.replace('/', '__')}.done"
        jobs.append({"id": job_id, "stage": job_id.split("/")[0],
                     "command": ["python", "fake_job.py", job_id, flag, str(state)],
                     "resources": {"gpus": gpus, "cpus": cpus, "mem_gb": mem, "time_limit_minutes": 5},
                     "deps": deps, "complete": flag, "requeue_exit_codes": [75], "priority": prio})
    graph = {"schema": "droid_jobs_v1", "config": "fake", "config_sha256": "0"*64, "data_root": "data",
             "run_root": "runs", "jobs": jobs}
    path = tmp_path/"jobs.json"
    path.write_text(json.dumps(graph))
    return path


def plan(tmp_path, **actions):
    (tmp_path/"state"/"plan.json").write_text(json.dumps({k.replace("__", "/"): v for k, v in actions.items()}))


def events(tmp_path):
    path = tmp_path/"state"/"events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def started(tmp_path):
    return [e["job"] for e in events(tmp_path) if e["event"] == "start"]


def runner(tmp_path, path, *extra):
    return droid_run.main([str(path), "--workdir", str(tmp_path), "--poll-seconds", "0.02", *extra])


@POSIX_RUNNER
def test_runner_skips_done_requeues_75_and_logs(tmp_path, capsys):
    path = fake_graph(tmp_path, [("a", [], 0, 1, 1, 0), ("b", ["a"], 0, 1, 1, 1), ("c", ["b"], 0, 1, 1, 2)])
    flag_a = tmp_path/"runs/fake/flags/a.done"
    flag_a.parent.mkdir(parents=True)
    flag_a.write_text("{}")
    plan(tmp_path, b=[75, 75, 0])
    assert runner(tmp_path, path, "--cpus", "4", "--mem-gb", "8") == 0
    assert started(tmp_path) == ["b", "b", "b", "c"]                 # a 已完成，b 重排两次
    log = (tmp_path/"runs/fake/logs/b.log").read_text()
    assert log.count("start (attempt") == 3 and log.count("exit 75") == 2 and log.count("attempt 3 action 0") == 1
    assert (tmp_path/"runs/fake/logs/RUNNER.pid").read_text() == ""        # 锁文件留着，退出时清空、锁随描述符释放
    out = capsys.readouterr().out
    assert "重排第 2 次" in out
    capsys.readouterr()
    assert runner(tmp_path, path, "--cpus", "4", "--mem-gb", "8") == 0        # 全部完成：什么都不跑
    assert started(tmp_path) == ["b", "b", "b", "c"]


@POSIX_RUNNER
def test_runner_rechecks_train_eval_and_summary(tmp_path, capsys):
    """S11-P1-1、RF-P2-4：完成标志已在的 check、train、eval、summary 每次启动运行器都照常运行一次（由阶段核身份）；复核
    失败时下游不开始；其他阶段完成标志在就跳过。dry-run 把它们列成“要复核”。"""
    path = fake_graph(tmp_path, [("calibrate/x", [], 0, 1, 1, 0), ("check/x", ["calibrate/x"], 0, 1, 1, 1),
                                 ("train/x", ["check/x"], 0, 1, 1, 1),
                                 ("eval/x", ["train/x"], 0, 1, 1, 2), ("summary", ["eval/x"], 0, 1, 1, 3)])
    for job in ("calibrate/x", "check/x", "train/x", "eval/x", "summary"):
        flag = tmp_path/f"runs/fake/flags/{job.replace('/', '__')}.done"
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("{}")
    assert runner(tmp_path, path, "--dry-run", "--cpus", "1", "--mem-gb", "1") == 0
    out = capsys.readouterr().out
    assert "已完成  calibrate/x" in out and all(f"已完成、要复核  {j}" in out
                                             for j in ("check/x", "train/x", "eval/x", "summary"))
    assert [line.split()[1] for line in out.splitlines() if line.strip()[:1].isdigit()] == [
        "check/x", "train/x", "eval/x", "summary"]
    assert runner(tmp_path, path, "--cpus", "1", "--mem-gb", "1") == 0
    assert started(tmp_path) == ["check/x", "train/x", "eval/x", "summary"]
    plan(tmp_path, eval__x=[1])                                           # 复核发现身份不同：失败，汇总不开始
    assert runner(tmp_path, path, "--cpus", "1", "--mem-gb", "1") == 1
    assert started(tmp_path)[4:] == ["check/x", "train/x", "eval/x"]
    assert "因上游失败没有开始的：summary" in capsys.readouterr().out
    plan(tmp_path, check__x=[1])                                          # RF-P2-4：环境记录不是当前的，训练不开始
    assert runner(tmp_path, path, "--cpus", "1", "--mem-gb", "1") == 1
    assert started(tmp_path)[7:] == ["check/x"]
    assert "因上游失败没有开始的：eval/x, summary, train/x" in capsys.readouterr().out


@POSIX_RUNNER
def test_runner_lock_is_released_when_writing_the_pid_fails(tmp_path, monkeypatch):
    """S14-P3-1：取得 flock 之后写 PID 出错（磁盘满）时描述符关掉、锁随之释放，同一进程再取能取到（描述符还开着的话 flock
    仍被持有，再取会报“另一个 droid_run.py 正在用”）；释放时清空出错也关描述符。"""
    lock = tmp_path/"logs"/droid_run.LOCK

    def full(*a):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(droid_run.os, "pwrite", full)
    with pytest.raises(OSError, match="No space"):
        droid_run._acquire(lock)
    monkeypatch.undo()
    fd = droid_run._acquire(lock)
    monkeypatch.setattr(droid_run.os, "ftruncate", full)
    with pytest.raises(OSError, match="No space"):
        droid_run._release(fd)
    monkeypatch.undo()
    with pytest.raises(OSError):
        os.fstat(fd)
    droid_run._release(droid_run._acquire(lock))


@POSIX_RUNNER
def test_runner_failure_blocks_downstream_only(tmp_path, capsys):
    path = fake_graph(tmp_path, [("a", [], 0, 1, 1, 0), ("b", ["a"], 0, 1, 1, 1), ("d", ["b"], 0, 1, 1, 2),
                                 ("c", [], 0, 1, 1, 5), ("e", ["c"], 0, 1, 1, 6), ("f", [], 0, 1, 1, 0)])
    plan(tmp_path, a=[3], f=["noflag"])
    assert runner(tmp_path, path, "--cpus", "1", "--mem-gb", "8") == 1
    assert sorted(started(tmp_path)) == ["a", "c", "e", "f"]             # b、d 不开始，c、e 照常跑
    out = capsys.readouterr().out
    assert "失败：a（退出码 3）" in out and str(tmp_path/"runs/fake/logs/a.log") in out
    assert "失败：f（退出 0 但没有完成标志" in out and "因上游失败没有开始的：b, d" in out
    plan(tmp_path, a=[0], f=[0])                                          # 修好后重跑：只补没做完的
    assert runner(tmp_path, path, "--cpus", "1", "--mem-gb", "8") == 0
    assert started(tmp_path)[4:] == ["a", "f", "b", "d"]


@POSIX_RUNNER
def test_runner_priority_then_id(tmp_path):
    path = fake_graph(tmp_path, [("z", [], 0, 1, 1, 1), ("y", [], 0, 1, 1, 2), ("x", [], 0, 1, 1, 2),
                                 ("w", ["z"], 0, 1, 1, 0)])
    assert runner(tmp_path, path, "--cpus", "1", "--mem-gb", "1") == 0
    assert started(tmp_path) == ["z", "w", "x", "y"]


def _max_overlap(evs, pick):
    points = sorted((e["t"], 1 if e["event"] == "start" else -1) for e in evs if pick(e["job"]))
    best = cur = 0
    for _, d in sorted(points, key=lambda p: (p[0], p[1])):
        cur += d
        best = max(best, cur)
    return best


@POSIX_RUNNER
def test_runner_resource_accounting(tmp_path):
    specs = [(f"g{k}", [], 1, 2, 10, 0) for k in range(4)] + [(f"c{k}", [], 0, 3, 10, 1) for k in range(3)] \
        + [("big", [], 0, 1, 25, 2), ("gpu2", [], 2, 1, 1, 3), ("toomuch", [], 3, 1, 1, 0), ("after", ["toomuch"], 0, 1, 1, 0)]
    path = fake_graph(tmp_path, specs)
    plan(tmp_path, **{f"g{k}": [0.4] for k in range(4)}, **{f"c{k}": [0.3] for k in range(3)},
         big=[0.2], gpu2=[0.2])
    assert runner(tmp_path, path, "--gpus", "3,5", "--cpus", "6", "--mem-gb", "30") == 1    # toomuch 要 3 张卡
    evs = events(tmp_path)
    assert sorted(started(tmp_path)) == sorted(["g0", "g1", "g2", "g3", "c0", "c1", "c2", "big", "gpu2"])
    starts = {e["job"]: e for e in evs if e["event"] == "start"}
    assert all(starts[f"g{k}"]["cuda"] in ("3", "5") for k in range(4))
    assert starts["gpu2"]["cuda"] == "3,5" and all(starts[f"c{k}"]["cuda"] == "" for k in range(3))
    assert _max_overlap(evs, lambda j: j.startswith("g")) == 2               # 两张卡，同时最多两个单卡任务
    # 同一时刻在跑的任务：卡数 ≤ 2、CPU ≤ 6、内存 ≤ 30
    res = {s[0]: s[2:5] for s in specs}
    timeline = sorted((e["t"], e["event"], e["job"]) for e in evs)
    live, worst = set(), [0, 0, 0]
    for t, kind, job in sorted(timeline, key=lambda x: (x[0], x[1] != "end")):
        (live.add if kind == "start" else live.discard)(job)
        for k in range(3):
            worst[k] = max(worst[k], sum(res[j][k] for j in live))
    assert worst[0] <= 2 and worst[1] <= 6 and worst[2] <= 30
    cards = [starts[f"g{k}"]["cuda"] for k in range(2)]
    assert sorted(cards) == ["3", "5"]                                         # 两个同时开始的单卡任务分到不同的卡


def test_runner_dry_run_lists_all_in_order(tmp_path, capsys):
    path = fake_graph(tmp_path, [("a", [], 0, 1, 1, 0), ("b", ["a"], 0, 1, 1, 0), ("c", [], 0, 1, 1, 1),
                                 ("d", ["c"], 1, 1, 1, 0)])
    flag = tmp_path/"runs/fake/flags/c.done"
    flag.parent.mkdir(parents=True)
    flag.write_text("{}")
    assert runner(tmp_path, path, "--dry-run", "--cpus", "1", "--mem-gb", "1") == 0
    out = capsys.readouterr().out
    lines = [line.split()[1] for line in out.splitlines() if line.strip()[:1].isdigit()]
    assert lines == ["a", "b", "d"] and "已完成  c" in out
    assert events(tmp_path) == [] and not (tmp_path/"runs/fake/logs").exists()


@POSIX_RUNNER
def test_runner_ctrl_c_then_rerun_finishes_only_the_rest(tmp_path):
    path = fake_graph(tmp_path, [("a", [], 0, 1, 1, 0), ("b", ["a"], 0, 1, 1, 0), ("c", ["b"], 0, 1, 1, 0)])
    plan(tmp_path, b=["hang", 0])
    proc = subprocess.Popen([sys.executable, str(ROOT/"tools"/"droid_run.py"), str(path), "--workdir", str(tmp_path),
                             "--cpus", "1", "--mem-gb", "1", "--poll-seconds", "0.02", "--grace-seconds", "1"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    hanging = tmp_path/"state"/"b.hanging"
    deadline = time.time() + 30
    while not hanging.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert hanging.exists()
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 130, out
    assert started(tmp_path) == ["a", "b"] and not (tmp_path/"runs/fake/flags/b.done").exists()
    assert "interrupted" in (tmp_path/"runs/fake/logs/b.log").read_text()
    assert runner(tmp_path, path, "--cpus", "1", "--mem-gb", "1") == 0
    assert started(tmp_path) == ["a", "b", "b", "c"]                   # a 不重跑


def test_runner_on_the_real_pilot_graph_dry_run(tmp_path, capsys):
    graph = graph_of("pilot")
    path = tmp_path/"jobs-pilot.json"
    path.write_text(json.dumps(graph))
    assert droid_run.main([str(path), "--workdir", str(tmp_path), "--dry-run", "--cpus", "32", "--mem-gb", "256"]) == 0
    lines = [line.split()[1] for line in capsys.readouterr().out.splitlines() if line.strip()[:1].isdigit()]
    assert sorted(lines) == sorted(j["id"] for j in graph["jobs"]) and len(lines) == 40
    pos = {i: k for k, i in enumerate(lines)}
    assert all(pos[d] < pos[j["id"]] for j in graph["jobs"] for d in j["deps"])
    assert lines[0] == "index" and lines[-1] == "summary"


def test_entry_as_a_script(tmp_path):
    """tools/droid.py 作为脚本运行（自己找到 src 和 droid_jobs）：阶段出错退出 1、不写完成标志；完成标志已在时退出 0。"""
    def run(job, stage):
        return subprocess.run([sys.executable, "tools/droid.py", stage, "--config", "configs/droid/pilot.yaml",
                               "--data-root", str(tmp_path/"data"), "--run-root", str(tmp_path/"runs"), "--job", job],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)
    r = run("summary", "summary")
    flag = tmp_path/"runs/pilot/summary/COMPLETE.json"
    assert r.returncode == 1 and "Error" in r.stderr and not flag.exists()      # 空数据根：汇总读不到索引
    flag.parent.mkdir(parents=True)
    flag.write_text("{}")
    r = run("summary", "summary")                         # S11-P1-1：只有完成标志不算完成，汇总照常重算（照样读不到）
    assert r.returncode == 1 and "already complete" not in r.stdout and "Error" in r.stderr
    done = tmp_path/"data/index/MANIFEST.json"            # 只写一次的阶段：完成标志在就退出 0
    done.parent.mkdir(parents=True)
    done.write_text("{}")
    r = run("index", "index")
    assert r.returncode == 0 and "already complete" in r.stdout


@POSIX_RUNNER
def test_entry_runs_the_real_stage_on_a_bare_completion_flag(tmp_path):
    """S11-P1-1 的反例：训练、评测目录里只有一个空的 COMPLETE.json（没有报告、登记、job.json），统一入口进真实阶段代码，
    不报完成。"""
    config = load_config(CONFIGS/"pilot.yaml")
    for job in ("train/koopman/seed1703", "eval/identity", "eval/koopman/seed1703"):
        path = droid_jobs.complete_path(job, config, tmp_path/"data", tmp_path/"runs")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        with pytest.raises((ValueError, OSError)):
            droid.main(_entry(tmp_path, job, "--device", "cpu"))


# ---------------------------------------------------------------- 运行器锁（S14-fix1，Codex S14-P2-1）

LOCK_WORKER = r"""
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import droid_run
lock, go, hold = Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])
while not go.exists():           # 所有进程都已启动、都在等；同时放行，尽量撞在一起
    time.sleep(0.001)
try:
    fd = droid_run._acquire(lock)
except SystemExit as e:
    print("refused", e, flush=True)
    sys.exit(3)
print("acquired", flush=True)
time.sleep(hold)                 # 持锁期间其他进程都该被拒
droid_run._release(fd)
"""


def _lock_workers(tmp_path, n, hold):
    script = tmp_path/"lock_worker.py"
    script.write_text(LOCK_WORKER)
    lock, go = tmp_path/"runs/fake/logs/RUNNER.pid", tmp_path/"go"
    procs = [subprocess.Popen([sys.executable, str(script), str(ROOT/"tools"), str(lock), str(go), str(hold)],
                              stdout=subprocess.PIPE, text=True) for _ in range(n)]
    time.sleep(1.0)
    go.write_text("")
    outs = [p.communicate(timeout=60)[0] for p in procs]
    return [p.returncode for p in procs], outs, lock


@POSIX_RUNNER
def test_runner_lock_is_exclusive_across_processes(tmp_path):
    codes, outs, lock = _lock_workers(tmp_path, 8, 2.0)
    assert sorted(codes) == [0] + [3]*7, outs              # 同时抢：恰好一个取得
    assert sum("acquired" in o for o in outs) == 1
    assert all("另一个 droid_run.py 正在用" in o for o, c in zip(outs, codes) if c == 3)
    assert lock.read_text() == ""
    codes, _, _ = _lock_workers(tmp_path, 1, 0.0)
    assert codes == [0]                                     # 释放后能再取


@POSIX_RUNNER
def test_runner_lock_survives_a_killed_holder_and_names_it(tmp_path):
    lock = tmp_path/"runs/fake/logs/RUNNER.pid"
    holder = subprocess.Popen([sys.executable, "-c", f"""
import sys, time
sys.path.insert(0, {str(ROOT/'tools')!r})
import droid_run
from pathlib import Path
droid_run._acquire(Path({str(lock)!r}))
print("held", flush=True)
time.sleep(60)
"""], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(SystemExit, match=f"进程 {holder.pid}"):
            droid_run._acquire(lock)
        path = fake_graph(tmp_path, [("a", [], 0, 1, 1, 0)])
        with pytest.raises(SystemExit, match="另一个 droid_run.py"):      # 整个运行器也被拒，什么都不跑
            runner(tmp_path, path, "--cpus", "1", "--mem-gb", "1")
        assert started(tmp_path) == []
    finally:
        holder.kill()
        holder.wait()
    fd = droid_run._acquire(lock)                            # 持锁进程被杀：内核释放，没有陈旧锁
    assert lock.read_text() == str(os.getpid())
    droid_run._release(fd)
