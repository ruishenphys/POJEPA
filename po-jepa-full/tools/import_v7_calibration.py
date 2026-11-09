"""16 §3.1：把 v7 的标定文件原字节导入本目录的 configs/calibration/（A0 段，CPU）。

  python tools/import_v7_calibration.py --v7-tree ../../po-jepa --stage full

1. v7 目录的 git HEAD 必须是 V7_COMMITS 里唯一的提交 0ac45fe（token 写法；机哥的正式结果是它跑的）。旧提交 653e2d0、
   911bc09 的模型是模态写法，不能混用，报错（请 git pull 到 0ac45fe）；v7 不应有任何别的改动。
2. 本阶段每个场景读 <v7 目录>/configs/calibration/<tag>-u<步数>.yaml，sigma_z、time_weight、space_weight 都必须是有限正数。
3. 原字节复制到本目录 configs/calibration/ 下同名文件：目标已存在且字节相同就跳过，不同就报错。
4. configs/calibration/IMPORTED.json 给每个文件记源路径、SHA256、导入时间、v7 的 commit（这次读到的 HEAD）；已有的记录原样保留。
   已有的记录要求 sha256 等于源文件的、v7_commit 在 V7_COMMITS 里（也就是 0ac45fe）；记的是旧提交（模态写法）就报错。
先全部核对再写，任何一项不符都不写任何文件；重复运行不改任何文件。本目录的 load_protocol() 因此读到和 v7 逐位相同的 σ_z、λ。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_addon_jobs import STAGES, name  # noqa: E402
from po_jepa.runtime import cache  # noqa: E402
from po_jepa.runtime.workflow import read  # noqa: E402

# v7 接受的提交只有 0ac45fe（token 写法，16 §3.1）：机哥的正式结果是 0ac45fe 跑的；旧提交 653e2d0、911bc09 的模型是
# 模态写法，它们的标定不能和本目录（token 写法）混用，导入时拒绝。
V7_COMMITS = ("0ac45fe03bf43b96bf861446b8aa2ada4f4a8987",)
SCHEMA = "addon_v7_calibration_import_v1"
REQUIRED = ("sigma_z", "time_weight", "space_weight")


def file_names(stage) -> list[str]:
    """本阶段每个场景的标定文件名，和 checks.v7_calibration_path().name 相同：<tag>-u<步数>.yaml。"""
    spec = STAGES[stage]
    return [f"{name(s)}-{spec['n']}-u{spec['updates']}.yaml" for s in spec["scenes"]]


def v7_head(tree: Path) -> str:
    result = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8")
    if result.returncode:
        raise ValueError(f"读不到 {tree} 的 git HEAD：{result.stderr.strip()}")
    return result.stdout.strip()


def _check_values(path: Path, raw: bytes) -> None:
    values = yaml.safe_load(raw.decode("utf-8"))
    if type(values) is not dict:
        raise ValueError(f"{path} 不是 YAML 映射")
    for key in REQUIRED:
        value = values.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{path} 的 {key} 是 {value!r}，应为有限正数（v7 的 calibrate 没有完成？）")


def _replace_json(path: Path, value: dict) -> None:
    """cache._write_json 写到同目录的新临时名，再整体换名（目标可以已存在）。"""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    for stale in (temporary, temporary.with_name(temporary.name+".partial")):
        stale.unlink(missing_ok=True)
    cache._write_json(temporary, value)
    os.replace(temporary, path)


def _copy(source_bytes: bytes, target: Path) -> None:
    """同目录临时文件写完、fsync，再硬链接到目标（目标已存在就失败，不覆盖）。"""
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(source_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def import_calibration(v7_tree, stage, target, *, now=None) -> dict:
    """把本阶段的 v7 标定导入 target（正式用 ROOT/configs/calibration）；返回 {文件名: "imported" | "unchanged"}。"""
    v7_tree, target = Path(v7_tree).resolve(), Path(target)
    head = v7_head(v7_tree)
    if head not in V7_COMMITS:
        raise ValueError(f"v7 目录 {v7_tree} 的 git HEAD 是 {head}，应为 {' 或 '.join(V7_COMMITS)}"
                         "（只接受 0ac45fe 的 token 写法；旧提交是模态写法，请 git pull 到 0ac45fe）")
    record_path = target/"IMPORTED.json"
    record = read(record_path) if record_path.exists() else dict(schema=SCHEMA, files={})
    if record.get("schema") != SCHEMA or type(record.get("files")) is not dict:
        raise ValueError(f"{record_path} 不是导入记录（schema {SCHEMA}）")
    plan = {}
    for file in file_names(stage):
        source = v7_tree/"configs/calibration"/file
        if not source.is_file():
            raise ValueError(f"缺少 v7 标定 {source}（v7 的 C 段没有跑完？）")
        raw = source.read_bytes()
        _check_values(source, raw)
        digest = hashlib.sha256(raw).hexdigest()
        destination = target/file
        if destination.exists() and destination.read_bytes() != raw:
            raise ValueError(f"{destination} 已存在且和 {source} 不同；不覆盖")
        entry = record["files"].get(file)
        if entry is not None and (entry.get("sha256") != digest or entry.get("v7_commit") not in V7_COMMITS):
            raise ValueError(f"{record_path} 里 {file} 的记录和 {source} 不符")
        plan[file] = (source, raw, digest, destination, entry)
    stamp = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    status, added = {}, {}
    target.mkdir(parents=True, exist_ok=True)
    for file, (source, raw, digest, destination, entry) in plan.items():
        status[file] = "unchanged" if destination.exists() else "imported"
        if status[file] == "imported":
            _copy(raw, destination)
        if entry is None:
            added[file] = dict(source=str(source), sha256=digest, imported_at=stamp, v7_commit=head)
    if added:
        _replace_json(record_path, dict(record, files={**record["files"], **added}))
    return status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v7-tree", type=Path, required=True, help="v7 目录（工作目录/po-jepa）")
    p.add_argument("--stage", choices=tuple(STAGES), required=True)
    a = p.parse_args(argv)
    status = import_calibration(a.v7_tree, a.stage, ROOT/"configs/calibration")
    for file, state in status.items():
        print(f"{file}：{'已导入' if state == 'imported' else '已存在且字节相同，跳过'}", flush=True)
    print(f"完成：{len(status)} 个标定文件（configs/calibration/，记录在 IMPORTED.json）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
