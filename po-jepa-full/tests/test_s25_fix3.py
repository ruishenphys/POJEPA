"""S25-fix3（Codex 复审 S25-fix1 的 P3-2）：tools/encode_v7.py 在每条 manifest 的 generation 里记编码 batch，
finalize 的 report.json 记每条的 batch 和出现过的所有值；S25-fix3 以前编的缓存（manifest 没有这一项）重跑 finalize，
report.json 逐字节不变（encode_report_sha256 不变，已有的评测报告还能配对）。

空间采样汇总并进 pairing_problems 的测试在 tests/test_s25e_space_eval.py。
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from po_jepa.runtime import cache

ROOT = Path(__file__).resolve().parents[1]
TEST_IDS = tuple(range(36, 40))


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")}
    env.update(PYTHONPATH=str(ROOT/"src"), OMP_NUM_THREADS="2")
    return env


def _run(mode, out, *extra):
    args = ["--system", "double_well", "--trajectories", "40", "--output", str(out), "--splits", "test"]
    if mode == "encode":
        args += ["--fake-encoder", "--device", "cpu"]
    return subprocess.run([sys.executable, "-B", str(ROOT/"tools/encode_v7.py"), mode, *args, *extra],
                          env=_env(), capture_output=True, text=True)


def _strip_batch(folder):
    """把一条缓存改回 S25-fix3 以前的样子：manifest 去掉 generation.encode_batch，COMPLETE.json 跟着改。"""
    m = json.loads((folder/"manifest.json").read_bytes())
    del m["generation"]["encode_batch"]
    (folder/"manifest.json").unlink(); (folder/"COMPLETE.json").unlink()
    cache._write_json(folder/"manifest.json", m)
    cache._write_json(folder/"COMPLETE.json", dict(manifest_sha256=cache.file_sha(folder/"manifest.json")))


@pytest.fixture(scope="module")
def encoded(tmp_path_factory):
    """两个分片用不同的 --batch（16、33）编同一套测试段。"""
    out = tmp_path_factory.mktemp("fix3")/"enc"
    for shard, batch in (("0/2", "16"), ("1/2", "33")):
        r = _run("encode", out, "--shard", shard, "--batch", batch)
        assert r.returncode == 0, r.stderr
    return out


def test_manifest_and_report_record_encode_batch(encoded):
    for tid in TEST_IDS:
        m = json.loads((encoded/f"{tid:04d}"/"manifest.json").read_bytes())
        assert m["generation"]["encode_batch"] == (16 if tid % 2 == 0 else 33)
    r = _run("finalize", encoded)
    assert r.returncode == 0, r.stderr
    assert "编码 batch [16, 33]" in r.stdout
    report = json.loads((encoded/"report.json").read_bytes())
    assert report["encode_batches"] == [16, 33] and report["encode_batch_unrecorded"] == 0
    assert {x["trajectory_id"]: x["encode_batch"] for x in report["records"]} == {t: 16 if t % 2 == 0 else 33 for t in TEST_IDS}


def test_legacy_cache_report_unchanged(encoded, tmp_path):
    """全都没记 batch（旧缓存）：报告里没有 batch 字段，和 S25-fix3 以前的 finalize 写的一样；部分没记时数出来。"""
    import shutil
    old = tmp_path/"old"; shutil.copytree(encoded, old)
    for name in ("report.json", "COMPLETE.json"):
        (old/name).unlink(missing_ok=True)
    _strip_batch(old/"0036")
    r = _run("finalize", old)
    assert r.returncode == 0, r.stderr
    report = json.loads((old/"report.json").read_bytes())
    assert report["encode_batches"] == [16, 33] and report["encode_batch_unrecorded"] == 1
    assert "encode_batch" not in next(x for x in report["records"] if x["trajectory_id"] == 36)
    for tid in TEST_IDS[1:]:
        _strip_batch(old/f"{tid:04d}")
    r = _run("finalize", old)
    assert r.returncode == 0, r.stderr
    assert "编码 batch 没记" in r.stdout
    report = json.loads((old/"report.json").read_bytes())
    # S25-fix3 以前 finalize 的字段（tools/encode_v7.py ee1d797）
    assert set(report) == {"mode", "status", "system", "dataset_count", "splits", "splits_encoded", "evidence_roles", "records",
                           "max_abs", "max_relative_rounding_error"}
    assert all(set(x) == {"trajectory_id", "split", "manifest_sha256", "evidence_role", "max_abs", "relative_rounding_error"}
               for x in report["records"])


def test_batch_must_be_positive(tmp_path):
    r = _run("encode", tmp_path/"enc", "--batch", "0")
    assert r.returncode != 0 and "--batch 必须是正整数" in r.stderr and not (tmp_path/"enc").exists()
