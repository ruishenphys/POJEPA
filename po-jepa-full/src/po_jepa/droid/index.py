"""DROID 1.0.1 索引（docs/design/02-droid-index-and-splits.md §1–§4、§7、§8）。

阶段 1 的前半：用 GCS JSON API 列出全部 `metadata_*.json`（§1），逐个下载并核对 MD5（§2），统一字段（§3），
判定入选（§4），写 `index/listing.jsonl`、`index/metadata.jsonl`、`index/episodes.jsonl`；后半（§5–§7，S03）由
`splits.py` 算划分、写 `index/splits.json`；最后写记这四个文件的 `index/MANIFEST.json`（§7）。§8 的抽样核对是 `verify_sample`。

网络只走两个地址前缀：`data.list_api`（列表）和 `data.object_base`（对象内容），都从配置读，测试时换成本地假服务器。
每次请求失败（连接、HTTP 状态、JSON、MD5/大小不对）都按 `RETRY_DELAYS` 重试，重试用尽就报错退出，不跳过任何对象。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from po_jepa.droid.config import DataConfig, DroidConfig, load_config
from po_jepa.runtime import cache

INDEX_DIR = "index"
LISTING_FILE = "listing.jsonl"
METADATA_FILE = "metadata.jsonl"
EPISODES_FILE = "episodes.jsonl"
MANIFEST_FILE = "MANIFEST.json"
SPLITS_FILE = "splits.json"          # 内容由 splits.py 写（02 §5–§7）
INDEX_FILES = (LISTING_FILE, METADATA_FILE, EPISODES_FILE, SPLITS_FILE)
MANIFEST_SCHEMA = "droid_index_v1"

METADATA_GLOB = "**/metadata_*.json"
LIST_FIELDS = "items(name,size,md5Hash,crc32c,updated),nextPageToken"
PAGE_SIZE = 1000
RETRY_DELAYS = (1, 2, 4, 8, 16)      # §1：失败后重试 5 次，间隔 1、2、4、8、16 秒
METADATA_THREADS = 32                # §2
TIMEOUT_SECONDS = 60

OUTCOMES = ("success", "failure")
REASONS = ("duplicate_uuid", "short", "flag_mismatch", "no_scene_id")   # §4 表的顺序
RECORD_FIELDS = ("building", "date", "robot_serial", "user_id", "r2d2_version")   # §3：只作记录
EXTERNAL_CAMERAS = (("ext1", "ext1_cam_serial"), ("ext2", "ext2_cam_serial"))
INTERNAL_PREFIX = "_"
INTERNAL_FIELDS = ("_name", "_md5", "_size")

Getter = Callable[[str], bytes]


def salted_hash(salt: str, kind: str, key: str) -> str:
    """02 §5 的 h(kind, key) = sha256(f"{split_salt}|{kind}|{key}") 的十六进制串。"""
    return hashlib.sha256(f"{salt}|{kind}|{key}".encode("utf-8")).hexdigest()


def http_get(url: str) -> bytes:
    """GET 一个地址，返回完整内容；非 200 或读不完都抛异常（由 `with_retries` 决定是否重试）。"""
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
        if response.status != 200:
            raise OSError(f"HTTP {response.status} for {url}")
        return response.read()


def with_retries(action: Callable[[], object], what: str, *, sleep: Callable[[float], None] = time.sleep):
    """调用 `action`；抛异常就按 RETRY_DELAYS 等待后重试，共 1 + 5 次，仍失败就报错（写明 `what`）。"""
    for attempt, delay in enumerate((*RETRY_DELAYS, None)):
        try:
            return action()
        except Exception as error:   # noqa: BLE001 — 任何失败都按同一规则重试，最后一次的原因放进报错
            if delay is None:
                raise RuntimeError(f"{what}: still failing after {attempt + 1} attempts: {error!r}") from error
            sleep(delay)
    raise AssertionError("unreachable")   # pragma: no cover


def _loads(raw: bytes, what: str):
    """严格 JSON：UTF-8、不收 NaN/Infinity、对象里不能有重复键；对象保持原键顺序。"""
    def pairs(items):
        keys = [k for k, _ in items]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{what}: duplicate JSON keys")
        return dict(items)

    def constant(token):
        raise ValueError(f"{what}: non-finite JSON number {token}")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def _line(value: dict, *, sort_keys: bool) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"), allow_nan=False)


def _json_value(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False, separators=(",", ":"), allow_nan=False)


def _write_lines(path: Path, lines) -> None:
    """先写 `<名>.partial` 再换名（14 §2 第 5 条）；残留的 .partial 直接覆盖。"""
    temp = path.with_name(path.name + ".partial")
    with temp.open("w", encoding="utf-8", newline="\n") as file:
        for line in lines:
            file.write(line + "\n")
    os.replace(temp, path)


def _md5_base64(body: bytes) -> str:
    return base64.b64encode(hashlib.md5(body).digest()).decode("ascii")


# ---------------------------------------------------------------- §1 列表

def list_url(data: DataConfig, token: str | None) -> str:
    query = {"prefix": data.prefix, "matchGlob": METADATA_GLOB, "maxResults": str(PAGE_SIZE), "fields": LIST_FIELDS}
    if token is not None:
        query["pageToken"] = token
    return data.list_api + "?" + urllib.parse.urlencode(query)


def _check_item(item, data: DataConfig) -> dict:
    if not isinstance(item, dict):
        raise ValueError(f"listing item is not an object: {item!r}")
    name, size, md5 = item.get("name"), item.get("size"), item.get("md5Hash")
    if not isinstance(name, str) or not name.startswith(data.prefix):
        raise ValueError(f"listing item name {name!r} is not under prefix {data.prefix!r}")
    if not isinstance(size, str) or not size.isdigit():
        raise ValueError(f"listing item {name}: size {size!r} is not a decimal string")
    try:
        digest = base64.b64decode(md5, validate=True) if isinstance(md5, str) else b""
    except ValueError:
        digest = b""
    if len(digest) != 16:
        raise ValueError(f"listing item {name}: md5Hash {md5!r} is not a base64 MD5 digest")
    return item


def list_metadata(data: DataConfig, *, get: Getter = http_get,
                  sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    """§1：按 nextPageToken 翻页列出全部元数据对象，按 name 排序；条数必须等于 expected_metadata_files。"""
    items: list[dict] = []
    token: str | None = None
    seen_tokens: set[str] = set()
    page = 0
    while True:
        url = list_url(data, token)
        body = with_retries(lambda: _list_page(get(url), page), f"listing page {page} ({url})", sleep=sleep)
        items.extend(_check_item(item, data) for item in body.get("items", []))
        token = body.get("nextPageToken")
        if token is None:
            break
        if not isinstance(token, str) or not token or token in seen_tokens:
            raise ValueError(f"listing page {page}: bad or repeated nextPageToken {token!r}")
        seen_tokens.add(token)
        page += 1
    items.sort(key=lambda item: item["name"])
    names = [item["name"] for item in items]
    duplicates = sorted(n for n, k in Counter(names).items() if k > 1)
    if duplicates:
        raise ValueError(f"listing has duplicate object names: {duplicates[:5]}")
    if len(items) != data.expected_metadata_files:
        raise ValueError(f"listing has {len(items)} metadata files, expected {data.expected_metadata_files} "
                         f"(data.version {data.version} is a fixed release: the source changed, a human must look)")
    return items


def _list_page(raw: bytes, page: int) -> dict:
    body = _loads(raw, f"listing page {page}")
    if not isinstance(body, dict):
        raise ValueError(f"listing page {page} is not a JSON object")
    if not isinstance(body.get("items", []), list):
        raise ValueError(f"listing page {page}: items is not a list")
    return body


# ---------------------------------------------------------------- §2 元数据

def object_url(data: DataConfig, name: str) -> str:
    """对象内容地址：`{data.object_base}/` 后接对象名去掉 `data.prefix` 的部分（URL 编码）。"""
    if not name.startswith(data.prefix):
        raise ValueError(f"object name {name!r} is not under prefix {data.prefix!r}")
    return data.object_base.rstrip("/") + "/" + urllib.parse.quote(name[len(data.prefix):], safe="/")


def _checked_body(get: Getter, url: str, item: dict) -> bytes:
    body = get(url)
    if len(body) != int(item["size"]):
        raise ValueError(f"size {len(body)} != listed {item['size']}")
    if _md5_base64(body) != item["md5Hash"]:
        raise ValueError(f"MD5 {_md5_base64(body)} != listed {item['md5Hash']}")
    return body


def fetch_object(data: DataConfig, item: dict, *, get: Getter = http_get,
                 sleep: Callable[[float], None] = time.sleep) -> bytes:
    """下载一个对象，大小和 MD5 都等于列表里的值才返回；否则按重试规则重下。"""
    url = object_url(data, item["name"])
    return with_retries(lambda: _checked_body(get, url, item), f"metadata object {item['name']}", sleep=sleep)


def fetch_metadata_row(data: DataConfig, item: dict, *, get: Getter = http_get,
                       sleep: Callable[[float], None] = time.sleep) -> dict:
    """下载、核大小和 MD5、解析成一行元数据；三步在同一次尝试里，任一步失败都按重试规则整条重下（Codex S02 #1）。"""
    url = object_url(data, item["name"])
    return with_retries(lambda: parse_metadata(_checked_body(get, url, item), item),
                        f"metadata object {item['name']}", sleep=sleep)


def parse_metadata(body: bytes, item: dict) -> dict:
    """官方字段原样（原键顺序），再加 `_name`、`_md5`、`_size`。"""
    name = item["name"]
    official = _loads(body, name)
    if not isinstance(official, dict):
        raise ValueError(f"{name}: metadata is not a JSON object")
    clash = [k for k in official if k.startswith(INTERNAL_PREFIX)]
    if clash:
        raise ValueError(f"{name}: official metadata already has keys starting with '_': {clash}")
    return dict(official, _name=name, _md5=item["md5Hash"], _size=int(item["size"]))


def fetch_metadata(data: DataConfig, listing: list[dict], *, get: Getter = http_get,
                   sleep: Callable[[float], None] = time.sleep, threads: int = METADATA_THREADS) -> list[dict]:
    """§2：并行下载全部元数据（按列表顺序返回）；任一对象重试用尽就整体报错。"""
    def one(item: dict) -> dict:
        return fetch_metadata_row(data, item, get=get, sleep=sleep)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(one, item) for item in listing]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def read_current_metadata(path: Path, listing: list[dict]) -> list[dict] | None:
    """§2 重跑：`metadata.jsonl` 存在且每行的 `_name`、`_md5` 和新列表逐行一致时返回它，否则 None（重新下载）。"""
    if not path.exists():
        return None
    try:
        rows = [_loads(line.encode("utf-8"), f"{path}:{i + 1}")
                for i, line in enumerate(path.read_text(encoding="utf-8").splitlines())]
    except ValueError:
        return None
    if len(rows) != len(listing):
        return None
    for row, item in zip(rows, listing):
        if not isinstance(row, dict) or row.get("_name") != item["name"] or row.get("_md5") != item["md5Hash"]:
            return None
    return rows


# ---------------------------------------------------------------- §3 统一字段、§4 入选

def _require(row: dict, key: str, kind: type, what: str):
    if key not in row:
        raise ValueError(f"metadata has no {key!r}")
    value = row[key]
    if type(value) is not kind:
        raise ValueError(f"metadata {key!r} must be {what}, got {value!r}")
    return value


def normalize(row: dict, data: DataConfig) -> dict:
    """02 §3：一份元数据（带 `_name`）→ 统一字段；任一规则不满足就报错并写明对象名。"""
    name = row.get("_name")
    try:
        return _normalize(row, data)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name}: {error}") from error


def _plain_path(path: str) -> bool:
    """没有空段、`.`、`..`：这样字符串前缀包含就等于目录包含（Codex S02 #2）。"""
    return all(part not in ("", ".", "..") for part in path.split("/"))


def _normalize(row: dict, data: DataConfig) -> dict:
    name = row["_name"]
    if not name.startswith(data.prefix):
        raise ValueError(f"object name is not under prefix {data.prefix!r}")
    parts = name[len(data.prefix):].split("/")
    if len(parts) < 3 or not _plain_path(name[len(data.prefix):]):
        raise ValueError("object name must be <prefix><lab>/<outcome>/…/metadata_<uuid>.json")
    lab, outcome, file_name = parts[0], parts[1], parts[-1]
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome folder {outcome!r} is not one of {OUTCOMES}")
    uuid = _require(row, "uuid", str, "a string")
    if file_name != f"metadata_{uuid}.json":
        raise ValueError(f"uuid {uuid!r} does not match the file name {file_name!r}")
    success_flag = _require(row, "success", bool, "a bool")
    scene_id = _require(row, "scene_id", int, "an int")
    steps = _require(row, "trajectory_length", int, "an int")
    if steps <= 0:
        raise ValueError(f"trajectory_length must be positive, got {steps}")
    folder = "/".join(parts[:-1])
    h5 = f"{lab}/{_require(row, 'hdf5_path', str, 'a string')}"
    if not h5.startswith(folder + "/") or not _plain_path(h5):
        raise ValueError(f"h5 {h5!r} is not inside the episode folder {folder!r}")
    left_mp4 = f"{lab}/{_require(row, data.camera, str, 'a string')}"
    if not _plain_path(left_mp4):
        raise ValueError(f"left_mp4 {left_mp4!r} has empty, '.' or '..' path segments")
    mp4_dir = folder + "/recordings/MP4/"
    left_file = left_mp4[len(mp4_dir):]
    if not left_mp4.startswith(mp4_dir) or "/" in left_file or not left_file.endswith(".mp4") or left_file == ".mp4":
        raise ValueError(f"left_mp4 {left_mp4!r} is not a .mp4 file directly in {mp4_dir!r}")
    left_serial = left_file[:-len(".mp4")]
    matches = [camera for camera, key in EXTERNAL_CAMERAS if str(row[key]) == left_serial]
    if len(matches) != 1:
        raise ValueError(f"left camera serial {left_serial!r} matches {len(matches)} of the external camera serials "
                         f"({', '.join(f'{k}={row[k]!r}' for _, k in EXTERNAL_CAMERAS)}), expected exactly one")
    episode = {
        "lab": lab, "outcome": outcome, "success_flag": success_flag, "scene_id": scene_id, "steps": steps,
        "uuid": uuid, "folder": folder, "h5": h5, "left_mp4": left_mp4, "left_serial": left_serial,
        "left_is": matches[0], "group": "ood" if lab in data.ood_labs else "id",
    }
    for key in RECORD_FIELDS:
        episode[key] = row[key]
    return episode


def judge(episodes: list[dict], data: DataConfig) -> list[dict]:
    """02 §4：给每条回合加 `reasons`（按 REASONS 的顺序，可多条）和 `selected`（reasons 为空）。"""
    uuid_counts = Counter(e["uuid"] for e in episodes)
    judged = []
    for e in episodes:
        reasons = []
        if uuid_counts[e["uuid"]] > 1:
            reasons.append("duplicate_uuid")
        if e["steps"] < data.min_steps:
            reasons.append("short")
        if e["success_flag"] != (e["outcome"] == "success"):
            reasons.append("flag_mismatch")
        if e["group"] == "id" and e["scene_id"] < 0:
            reasons.append("no_scene_id")
        judged.append(dict(e, selected=not reasons, reasons=reasons))
    return judged


def build_episodes(rows: list[dict], data: DataConfig) -> list[dict]:
    """§3 + §4，按 (uuid, folder) 排序（重复 uuid 的两行靠 folder 区分）。"""
    episodes = judge([normalize(row, data) for row in rows], data)
    return sorted(episodes, key=lambda e: (e["uuid"], e["folder"]))


def summarize(episodes: list[dict]) -> dict:
    selected = [e for e in episodes if e["selected"]]
    return {
        "episodes": len(episodes),
        "selected": len(selected),
        "selected_by_group": dict(sorted(Counter(e["group"] for e in selected).items())),
        "selected_by_lab": dict(sorted(Counter(e["lab"] for e in selected).items())),
        "reasons": {r: sum(r in e["reasons"] for e in episodes) for r in REASONS},
    }


# ---------------------------------------------------------------- §7 输出与入口

def index_dir(data_root: Path | str) -> Path:
    return Path(data_root)/INDEX_DIR


def code_commit() -> str | None:
    """po-jepa-full 所在 git 仓库的 HEAD；不在 git 检出里时为 None（另有 code_manifest_sha256 标识代码）。"""
    result = subprocess.run(["git", "-C", str(cache.ROOT), "rev-parse", "HEAD"], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
    return result.stdout.strip() if result.returncode == 0 else None


def _file_entry(path: Path) -> dict:
    with path.open("rb") as file:
        lines = sum(1 for _ in file)
    return {"sha256": cache.file_sha(path), "lines": lines}


def read_manifest(directory: Path) -> dict:
    manifest = _loads((directory/MANIFEST_FILE).read_bytes(), str(directory/MANIFEST_FILE))
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{directory/MANIFEST_FILE}: schema is not {MANIFEST_SCHEMA!r}")
    files = manifest.get("files")
    if not isinstance(files, dict) or sorted(files) != sorted(INDEX_FILES):
        raise ValueError(f"{directory/MANIFEST_FILE} lists files {sorted(files) if isinstance(files, dict) else files}, "
                         f"this code writes {sorted(INDEX_FILES)}: written by other code, move the index away and rerun")
    return manifest


def check_files(directory: Path, manifest: dict) -> None:
    """index/ 下的文件和清单里的 SHA、行数一致。"""
    for name, entry in manifest["files"].items():
        actual = _file_entry(directory/name)
        if actual != entry:
            raise ValueError(f"{directory/name} does not match {MANIFEST_FILE}: {actual} != {entry}")


def run_index(config: DroidConfig, data_root: Path | str, *, get: Getter = http_get,
              sleep: Callable[[float], None] = time.sleep, threads: int = METADATA_THREADS,
              clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> dict:
    """阶段 1。`index/MANIFEST.json` 已存在就什么都不做（只核它是本代码写的），返回它。"""
    from po_jepa.droid import splits as sp   # splits 依赖本模块，放在函数里避免循环导入
    directory = index_dir(data_root)
    if (directory/MANIFEST_FILE).exists():
        return read_manifest(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = config.data
    listing = list_metadata(data, get=get, sleep=sleep)
    listed_at = clock().isoformat(timespec="seconds")
    _write_lines(directory/LISTING_FILE, (_line(item, sort_keys=True) for item in listing))
    rows = read_current_metadata(directory/METADATA_FILE, listing)
    reused = rows is not None
    if rows is None:
        rows = fetch_metadata(data, listing, get=get, sleep=sleep, threads=threads)
        _write_lines(directory/METADATA_FILE, (_line(row, sort_keys=False) for row in rows))
    episodes = build_episodes(rows, data)
    _write_lines(directory/EPISODES_FILE, (_line(e, sort_keys=True) for e in episodes))
    splits = sp.compute_splits(episodes, data, cache.file_sha(directory/EPISODES_FILE))
    sp.write_splits(directory, splits)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "data_version": data.version,
        "config_keys": {"list_api": data.list_api, "prefix": data.prefix, "camera": data.camera,
                        "expected_metadata_files": data.expected_metadata_files,
                        "min_steps": data.min_steps, "ood_labs": list(data.ood_labs)},
        "listed_at": listed_at,
        "metadata_reused": reused,
        "code_commit": code_commit(),
        "code_manifest_sha256": cache.file_sha(cache.ROOT/"MANIFEST.json"),
        "files": {name: _file_entry(directory/name) for name in INDEX_FILES},
        "counts": summarize(episodes),
    }
    cache._write_json(directory/MANIFEST_FILE, manifest)
    return manifest


def verify_sample(config: DroidConfig, data_root: Path | str, count: int, *, get: Getter = http_get,
                  sleep: Callable[[float], None] = time.sleep) -> dict:
    """§8：按 h("verify", name) 升序取 `count` 份元数据重新下载，和 metadata.jsonl 那一行逐字段（含键顺序）比较。"""
    directory = index_dir(data_root)
    manifest = read_manifest(directory)
    check_files(directory, manifest)
    listing = [_loads(line.encode("utf-8"), LISTING_FILE)
               for line in (directory/LISTING_FILE).read_text(encoding="utf-8").splitlines()]
    if not 0 < count <= len(listing):
        raise ValueError(f"--verify-sample must be between 1 and {len(listing)}, got {count}")
    rows = {row["_name"]: row for row in (_loads(line.encode("utf-8"), METADATA_FILE)
                                          for line in (directory/METADATA_FILE).read_text(encoding="utf-8").splitlines())}
    chosen = sorted(listing, key=lambda item: salted_hash(config.data.split_salt, "verify", item["name"]))[:count]
    for item in chosen:
        fresh = fetch_metadata_row(config.data, item, get=get, sleep=sleep)
        stored = rows.get(item["name"])
        # 按 JSON 文本比较：区分 true/1、300/300.0，也区分键顺序（Codex S02 #3）
        if stored is None or _line(fresh, sort_keys=False) != _line(stored, sort_keys=False):
            differing = sorted(k for k in set(fresh) | set(stored or {})
                               if k not in fresh or k not in (stored or {})
                               or _json_value(fresh[k]) != _json_value(stored[k]))
            raise ValueError(f"{item['name']}: re-downloaded metadata differs from {METADATA_FILE} "
                             f"(fields {differing or 'key order'})")
    return {"checked": count, "first": chosen[0]["name"], "last": chosen[-1]["name"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DROID 1.0.1 index and splits (docs/design/02)")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--verify-sample", type=int, default=0, metavar="N",
                        help="after indexing, re-download N metadata files (02 §8) and compare")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    manifest = run_index(config, args.data_root)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    splits = _loads((index_dir(args.data_root)/SPLITS_FILE).read_bytes(), SPLITS_FILE)
    print(json.dumps({"splits": splits["counts"]}, ensure_ascii=False, indent=2))
    if args.verify_sample:
        print(json.dumps(verify_sample(config, args.data_root, args.verify_sample), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
