"""tools/run_jobs.py：分段、跳过已完成、中断后续训改写、失败停段。"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import run_jobs  # noqa: E402


def _job(path, done):
    path.mkdir(parents=True)
    (path/"job.json").write_text("{}")
    if done:
        (path/"COMPLETE.json").write_text("{}")


def test_parse_sections(tmp_path):
    f = tmp_path/"jobs.txt"
    f.write_text("# 前提\n# ===== A 编码（GPU）=====\npython a1\npython a2\n# ===== B 汇总 =====\npython b\n")
    s = run_jobs.parse_sections(f)
    assert [x["name"] for x in s] == ["A", "B"] and s[0]["commands"] == ["python a1", "python a2"]


def test_prepare_resume_and_substitute(tmp_path):
    cmd = "python run.py run --condition po_jepa_full --device cuda:0 --output r/x"
    assert run_jobs.prepare(cmd, tmp_path) == (cmd, "")            # 没跑过：原样
    _job(tmp_path/"r/x/run/job", done=False)                       # 中断：改成 resume
    new, note = run_jobs.prepare(cmd, tmp_path)
    assert "run.py resume" in new and "--resume-from r/x" in new and "--output r/x-resume1" in new
    _job(tmp_path/"r/x-resume1/run/job", done=True)                # 续训完成：跳过
    assert run_jobs.prepare(cmd, tmp_path)[0] is None
    later = "python tools/plan_v7.py --job r/x --output p.json"   # 后面段自动改指向
    assert "--job r/x-resume1" in run_jobs.prepare(later, tmp_path)[0]


def test_prepare_broken_before_job(tmp_path):
    (tmp_path/"r/c").mkdir(parents=True)
    cmd = "python run.py calibrate --output r/c"
    assert run_jobs.prepare(cmd, tmp_path)[0] == cmd and (tmp_path/"r/c.broken1").exists()
    (tmp_path/"r/y").mkdir(parents=True)
    run_cmd = "python run.py run --output r/y"
    assert run_jobs.prepare(run_cmd, tmp_path)[0] == run_cmd and (tmp_path/"r/y.broken1").exists()


def test_runner_skips_done_and_stops_on_failure(tmp_path):
    (tmp_path/"run.py").write_text("")
    jobs = tmp_path/"jobs.txt"
    jobs.write_text("# ===== A x =====\n"
                    f"{sys.executable} -c \"open('a1','a').write('1')\"\n"
                    f"{sys.executable} -c \"open('a2','a').write('1')\"\n"
                    "# ===== B y =====\n"
                    f"{sys.executable} -c \"import sys; sys.exit(3)\"\n"
                    "# ===== C z =====\n"
                    f"{sys.executable} -c \"open('c','a').write('1')\"\n")
    run = lambda: subprocess.run([sys.executable, str(ROOT/"tools/run_jobs.py"), str(jobs), "--gpus", "0,1",
                                  "--logs", str(tmp_path/"logs")], cwd=tmp_path, capture_output=True, text=True)
    r = run()
    assert r.returncode == 1 and not (tmp_path/"c").exists()
    r = run()                                                      # 再跑：A 段跳过，B 仍失败
    assert r.returncode == 1 and (tmp_path/"a1").read_text() == "1" and (tmp_path/"a2").read_text() == "1"
    assert not (tmp_path/"logs/RUNNING.pid").exists()


def test_make_jobs_pilot_encodes_double_well():
    out = subprocess.run([sys.executable, str(ROOT/"tools/make_v7_jobs.py"), "--stage", "pilot"],
                         capture_output=True, text=True, check=True).stdout
    assert "encode --system double_well --trajectories 200" in out and "unpacked-v3" not in out
