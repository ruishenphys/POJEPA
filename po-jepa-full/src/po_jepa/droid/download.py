"""DROID 1.0.1 下载（docs/design/03-droid-download.md）。

阶段 2：一个任务下载一片（02 §6）里本配置用到的每条回合的 `trajectory.h5` 和左外部相机 MP4（§1），
每个文件先用 JSON API 取大小和 MD5，下载到 `.part`，核对通过才换名（§2）；一条回合两个文件到位后写
`DOWNLOADED.json`；全片完成后写片完成记录 `raw/<片名>/_complete/<配置名>.json`（§3）。

网络只走配置里的两个地址前缀：`data.list_api`（对象元数据）和 `data.object_base`（对象内容），测试时换成本地假服务器。
失败分三种：
- 传输失败（连接、HTTP 5xx 等状态、响应体比 Content-Length 短、元数据响应格式不对）按 02 §1 的规则重试
  （`index.RETRY_DELAYS`，共 1 + 5 次）；
- 对象不存在（HTTP 404）不重试，直接失败：划分已冻结，不能悄悄少一条（§3）；
- 下完的文件大小或 MD5 和 JSON API 给的不同：删掉 `.part` 重下，共 `VERIFY_ATTEMPTS` 次，仍不对就失败，报错里写明对象名（§2 第 3 条）。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from po_jepa.droid import index as ix
from po_jepa.droid import splits as sp
from po_jepa.droid.config import DataConfig, DroidConfig, load_config
from po_jepa.runtime import cache

RAW_DIR = "raw"
COMPLETE_DIR = "_complete"
H5_FILE = "trajectory.h5"
METADATA_JSON = "metadata.json"
DOWNLOADED_FILE = "DOWNLOADED.json"
RAW_DELETED_FILE = "RAW_DELETED.json"   # 05 §3 / 03 §4：编码任务删原始文件后写，本段只读它的存在
PART_SUFFIX = ".part"                   # §2：下载中的文件
PARTIAL_SUFFIX = ".partial"             # 14 §2 第 5 条：JSON 的临时文件
DOWNLOADED_SCHEMA = "droid_downloaded_v1"
COMPLETE_SCHEMA = "droid_download_complete_v1"
CHUNK_BYTES = 8 * 1024 * 1024           # §2 第 2 条：流式写，每块 8 MB
VERIFY_ATTEMPTS = 3                     # §2 第 3 条：大小或 MD5 不对时最多下 3 次
EPISODE_THREADS = 16                    # §2：一片里 16 条回合并行
OBJECT_FIELDS = "size,md5Hash"

Clock = Callable[[], datetime]


class ObjectNotFound(RuntimeError):
    """HTTP 404：对象在存储桶里不存在。不重试（§3）。"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _retrying(action: Callable[[], object], what: str, sleep: Callable[[float], None]):
    """02 §1 的重试规则（`index.with_retries`），但 `ObjectNotFound` 立即原样抛出、不重试。"""
    class _Fatal:
        def __init__(self, error: ObjectNotFound):
            self.error = error

    def guarded():
        try:
            return action()
        except ObjectNotFound as error:
            return _Fatal(error)

    result = ix.with_retries(guarded, what, sleep=sleep)
    if isinstance(result, _Fatal):
        raise result.error
    return result


def _open(url: str):
    """GET；404 抛 `ObjectNotFound`，其余 HTTP 错误和连接错误原样抛出（由调用者重试）。"""
    try:
        response = urllib.request.urlopen(url, timeout=ix.TIMEOUT_SECONDS)
    except urllib.error.HTTPError as error:
        error.close()
        if error.code == 404:
            raise ObjectNotFound(f"HTTP 404 for {url}") from None
        raise
    if response.status != 200:
        response.close()
        raise OSError(f"HTTP {response.status} for {url}")
    return response


# ---------------------------------------------------------------- §1 下载哪些

def episode_files(episode: dict, data: DataConfig) -> list[tuple[str, str]]:
    """一条回合的两个文件：[(本地文件名, 对象名)]，先 h5 后 MP4（§1 的表）。"""
    return [(H5_FILE, data.prefix + episode["h5"]),
            (f"{episode['left_serial']}.mp4", data.prefix + episode["left_mp4"])]


def metadata_object_name(episode: dict, data: DataConfig) -> str:
    """这条回合的元数据对象名（02 §3：`folder` 是对象名去掉前缀和文件名）。"""
    return f"{data.prefix}{episode['folder']}/metadata_{episode['uuid']}.json"


def official_metadata(row: dict) -> dict:
    """metadata.jsonl 的一行去掉 `_` 开头的三个字段（`_name`、`_md5`、`_size`），保持原键顺序。"""
    extra = [k for k in row if k.startswith(ix.INTERNAL_PREFIX)]
    if sorted(extra) != sorted(ix.INTERNAL_FIELDS):
        raise ValueError(f"metadata row has internal keys {extra}, expected {list(ix.INTERNAL_FIELDS)}")
    return {k: v for k, v in row.items() if not k.startswith(ix.INTERNAL_PREFIX)}


def metadata_json_bytes(official: dict) -> bytes:
    """`metadata.json` 的字节：原键顺序（不排序，§1），缩进 2，UTF-8，结尾换行。"""
    return (json.dumps(official, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


# ---------------------------------------------------------------- §2 每个文件怎么下

def object_api_url(data: DataConfig, name: str) -> str:
    """JSON API 的对象元数据地址：`{data.list_api}/<URL 编码的对象名>?fields=size,md5Hash`（斜杠也编码）。"""
    if not name.startswith(data.prefix):
        raise ValueError(f"object name {name!r} is not under prefix {data.prefix!r}")
    return (data.list_api.rstrip("/") + "/" + urllib.parse.quote(name, safe="")
            + "?" + urllib.parse.urlencode({"fields": OBJECT_FIELDS}))


def _parse_object_info(raw: bytes, name: str) -> dict:
    body = ix._loads(raw, f"object metadata of {name}")
    if not isinstance(body, dict):
        raise ValueError(f"object metadata of {name} is not a JSON object")
    size, md5 = body.get("size"), body.get("md5Hash")
    if not isinstance(size, str) or not size.isdigit():
        raise ValueError(f"object metadata of {name}: size {size!r} is not a decimal string")
    try:
        digest = base64.b64decode(md5, validate=True) if isinstance(md5, str) else b""
    except ValueError:
        digest = b""
    if len(digest) != 16:
        raise ValueError(f"object metadata of {name}: md5Hash {md5!r} is not a base64 MD5 digest")
    return {"size": int(size), "md5": md5}


def object_info(data: DataConfig, name: str, *, sleep: Callable[[float], None] = time.sleep) -> dict:
    """§2 第 1 步：对象的大小（int）和 MD5（base64，和 GCS 一致）。404 直接失败，其余失败按 02 §1 重试。"""
    url = object_api_url(data, name)

    def attempt() -> dict:
        with _open(url) as response:
            return _parse_object_info(response.read(), name)

    return _retrying(attempt, f"object metadata {name}", sleep)


def _stream(url: str, part: Path) -> None:
    """§2 第 2 步：流式写到 `.part`（每块 8 MB）；响应体比 Content-Length 短算传输失败。"""
    with _open(url) as response, part.open("wb") as file:
        expected = response.headers.get("Content-Length")
        written = 0
        for block in iter(lambda: response.read(CHUNK_BYTES), b""):
            file.write(block)
            written += len(block)
        if expected is not None and written != int(expected):
            raise OSError(f"connection closed after {written} of {expected} bytes for {url}")


def _digests(path: Path) -> tuple[int, str, str]:
    """(大小, MD5 的 base64, SHA-256 十六进制)，按块读。"""
    md5, sha = hashlib.md5(), hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        for block in iter(lambda: file.read(CHUNK_BYTES), b""):
            md5.update(block)
            sha.update(block)
            size += len(block)
    return size, base64.b64encode(md5.digest()).decode("ascii"), sha.hexdigest()


def download_file(data: DataConfig, name: str, target: Path, *,
                  sleep: Callable[[float], None] = time.sleep) -> dict:
    """§2：下载一个对象到 `target`，返回 DOWNLOADED.json 里这个文件的记录 {object, size, md5, sha256}。

    `target` 已存在（上次在两个文件之间中断）时先核大小和 MD5，对就直接用，不对就删掉重下。
    """
    info = object_info(data, name, sleep=sleep)
    if target.exists():
        size, md5, sha = _digests(target)
        if (size, md5) == (info["size"], info["md5"]):
            return {"object": name, "size": size, "md5": md5, "sha256": sha}
        target.unlink()
    url = ix.object_url(data, name)
    part = target.with_name(target.name + PART_SUFFIX)
    problems = []
    for _ in range(VERIFY_ATTEMPTS):
        part.unlink(missing_ok=True)
        _retrying(lambda: _stream(url, part), f"object {name}", sleep)
        size, md5, sha = _digests(part)
        if (size, md5) == (info["size"], info["md5"]):
            os.replace(part, target)
            return {"object": name, "size": size, "md5": md5, "sha256": sha}
        problems.append(f"size {size} md5 {md5}")
        part.unlink()
    raise RuntimeError(f"object {name}: downloaded {VERIFY_ATTEMPTS} times and never matched the listed "
                       f"size {info['size']} md5 {info['md5']} ({'; '.join(problems)})")


# ---------------------------------------------------------------- 一条回合

def episode_dir(data_root: Path | str, shard: str, uuid: str) -> Path:
    return Path(data_root)/RAW_DIR/shard/uuid


def _read_json(path: Path) -> dict:
    value = ix._loads(path.read_bytes(), str(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def read_downloaded(directory: Path, episode: dict, data: DataConfig) -> dict:
    """读 `DOWNLOADED.json` 并核它记的是这条回合的这两个对象（不是就报错：数据根被别的索引写过）。"""
    record = _read_json(directory/DOWNLOADED_FILE)
    expected = {local: name for local, name in episode_files(episode, data)}
    files = record.get("files")
    if (record.get("schema") != DOWNLOADED_SCHEMA or record.get("uuid") != episode["uuid"]
            or not isinstance(files, dict) or sorted(files) != sorted(expected)
            or any(not isinstance(files[k], dict) or files[k].get("object") != expected[k] for k in expected)):
        raise ValueError(f"{directory/DOWNLOADED_FILE} does not record episode {episode['uuid']} "
                         f"with objects {sorted(expected.values())}")
    return record


def _sizes_match(directory: Path, record: dict) -> bool:
    for local, entry in record["files"].items():
        path = directory/local
        if not path.is_file() or path.stat().st_size != entry.get("size"):
            return False
    return True


def _write_bytes(path: Path, body: bytes) -> None:
    temp = path.with_name(path.name + PARTIAL_SUFFIX)
    temp.write_bytes(body)
    os.replace(temp, path)


def download_episode(data: DataConfig, data_root: Path | str, shard: str, episode: dict, metadata_row: dict, *,
                     sleep: Callable[[float], None] = time.sleep, clock: Clock = _utc_now) -> dict:
    """一条回合：返回 {uuid, status: downloaded|skipped|raw_deleted, downloaded_sha256}。

    - 先删这条回合目录里残留的 `.part`（§2 第 4 条）和 JSON 的 `.partial`；
    - `metadata.json` 不存在或内容不同就（重）写；
    - 有 `RAW_DELETED.json`：跳过，不重新下载（§2、§4），`DOWNLOADED.json` 必须还在；
    - 有 `DOWNLOADED.json` 且两个文件的大小都和它记的一致：跳过，不重算 MD5（§2）；
    - 否则删掉旧的 `DOWNLOADED.json`，下载两个文件，再写新的 `DOWNLOADED.json`。
    """
    uuid = episode["uuid"]
    directory = episode_dir(data_root, shard, uuid)
    directory.mkdir(parents=True, exist_ok=True)
    for leftover in sorted(directory.iterdir()):
        if leftover.name.endswith((PART_SUFFIX, PARTIAL_SUFFIX)):
            leftover.unlink()
    metadata = metadata_json_bytes(official_metadata(metadata_row))
    meta_path = directory/METADATA_JSON
    if not meta_path.is_file() or meta_path.read_bytes() != metadata:
        _write_bytes(meta_path, metadata)
    downloaded = directory/DOWNLOADED_FILE
    if (directory/RAW_DELETED_FILE).exists():
        if not downloaded.is_file():
            raise ValueError(f"{directory}: has {RAW_DELETED_FILE} but no {DOWNLOADED_FILE}")
        read_downloaded(directory, episode, data)
        return {"uuid": uuid, "status": "raw_deleted", "downloaded_sha256": cache.file_sha(downloaded)}
    if downloaded.is_file():
        if _sizes_match(directory, read_downloaded(directory, episode, data)):
            return {"uuid": uuid, "status": "skipped", "downloaded_sha256": cache.file_sha(downloaded)}
        downloaded.unlink()
    files = {local: download_file(data, name, directory/local, sleep=sleep)
             for local, name in episode_files(episode, data)}
    record = {"schema": DOWNLOADED_SCHEMA, "uuid": uuid, "downloaded_at": clock().isoformat(timespec="seconds"),
              "files": files}
    cache._write_json(downloaded, record)
    return {"uuid": uuid, "status": "downloaded", "downloaded_sha256": cache.file_sha(downloaded)}


# ---------------------------------------------------------------- §3 片

def complete_path(data_root: Path | str, shard: str, config: DroidConfig) -> Path:
    return Path(data_root)/RAW_DIR/shard/COMPLETE_DIR/f"{config.name}.json"


def _metadata_rows(directory: Path, names: set[str]) -> dict[str, dict]:
    rows = {}
    path = directory/ix.METADATA_FILE
    with path.open(encoding="utf-8") as file:
        for i, line in enumerate(file):
            row = ix._loads(line.encode("utf-8"), f"{path}:{i + 1}")
            if row.get("_name") in names:
                rows[row["_name"]] = row
    missing = sorted(names - set(rows))
    if missing:
        raise ValueError(f"{path} has no rows for {missing[:5]}")
    return rows


def run_download(config: DroidConfig, data_root: Path | str, shard: str, *,
                 sleep: Callable[[float], None] = time.sleep, clock: Clock = _utc_now,
                 threads: int = EPISODE_THREADS) -> dict:
    """任务 `download/<片名>`。片完成记录已存在就什么都不做，返回它（12 §3）。

    片里本配置用到的回合来自 `splits.shard_plan`（02 §6）；任何一条失败（不存在、反复校验失败）整个任务失败、
    不写完成记录（§3：不跳过）。别的配置已经下好的回合靠 `DOWNLOADED.json` 跳过。
    """
    done = complete_path(data_root, shard, config)
    if done.exists():
        return _read_json(done)
    splits = sp.load_splits(data_root, config)
    plan = sp.shard_plan(splits, config.data)
    if shard not in plan:
        raise ValueError(f"shard {shard!r} is not used by config {config.name!r}; its shards are {list(plan)}")
    uuids = plan[shard]
    index = ix.index_dir(data_root)
    by_uuid = {e["uuid"]: e for e in sp.read_episodes(index) if e["selected"]}
    episodes = [by_uuid[u] for u in uuids]
    rows = _metadata_rows(index, {metadata_object_name(e, config.data) for e in episodes})

    def one(episode: dict) -> dict:
        return download_episode(config.data, data_root, shard, episode,
                                rows[metadata_object_name(episode, config.data)], sleep=sleep, clock=clock)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(one, e) for e in episodes]
        try:
            results = [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    record = {
        "schema": COMPLETE_SCHEMA,
        "shard": shard,
        "config": config.name,
        "config_sha256": config.config_sha256,
        "splits_sha256": cache.file_sha(index/ix.SPLITS_FILE),
        "uuids": uuids,
        "downloaded_sha256": [r["downloaded_sha256"] for r in results],
        "code_commit": ix.code_commit(),
    }
    done.parent.mkdir(parents=True, exist_ok=True)
    done.with_name(done.name + PARTIAL_SUFFIX).unlink(missing_ok=True)
    cache._write_json(done, record)
    counts = {status: sum(r["status"] == status for r in results) for status in ("downloaded", "skipped", "raw_deleted")}
    return dict(record, counts=counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DROID 1.0.1 download of one shard (docs/design/03)")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--shard", required=True, help="train-000 … , val, id_test or ood_test (02 §6)")
    args = parser.parse_args(argv)
    result = run_download(load_config(args.config), args.data_root, args.shard)
    print(json.dumps({k: result[k] for k in ("shard", "config") if k in result}
                     | {"episodes": len(result["uuids"]), "counts": result.get("counts", "already complete")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
