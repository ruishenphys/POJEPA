"""S04：DROID 下载（docs/design/03-droid-download.md；验收 14 §3 S04）。

本机起一个只服务内存对象的假存储桶（03 §5）：先用 S02 的列表和元数据接口建一个小索引和划分，
再模拟对象元数据接口（`{list_api}/<编码的对象名>?fields=size,md5Hash`）和对象内容接口，按测试需要返回
500、404、错误内容（同长度、MD5 不对）、诚实的短内容（大小不对）和中途断开的内容（比 Content-Length 短）。
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import random
import re
import threading
import urllib.parse
import zlib
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest
import yaml

from po_jepa.droid import config as config_module
from po_jepa.droid import download as dl
from po_jepa.droid import index as ix
from po_jepa.droid import splits as sp
from po_jepa.runtime import cache
from test_droid_index import CONFIGS, LIST_PATH, OBJECT_PATH, PREFIX, FakeBucket, _config, meta

NOW = datetime(2026, 11, 9, 15, 0, 0, 123456, tzinfo=timezone.utc)   # 记录只到秒
SMALL = dict(id_test_scenes=1, val_scenes=1, episodes_per_scene=1, ood_quota=(1, 0, 0),
             train_episodes=4, shard_size=3)


# ---------------------------------------------------------------- 假存储桶：S02 的两个接口 + 对象元数据接口

class RawBucket(FakeBucket):
    """在 S02 的 FakeBucket 上加对象元数据接口和几种坏内容。各表的值是“接下来还要出几次”。"""

    def __init__(self):
        super().__init__()
        self.fail_info: dict[str, int] = {}
        self.short: dict[str, int] = {}     # 诚实的短内容：Content-Length 等于实际发出的字节（大小不对）
        self.drop: dict[str, int] = {}      # 声明全长，只发一半就断开
        self.info_paths: list[str] = []     # 原样的请求路径（看斜杠有没有编码）
        self.partial: dict[str, int] = {}   # 206 加前一半内容（Content-Length 和发出的一致）
        self.info_size_delta: dict[str, int] = {}   # 对象元数据接口报的大小多出这么多（MD5 照旧）

    def respond(self, raw_path: str) -> tuple[int, bytes, int]:
        parsed = urllib.parse.urlsplit(raw_path)
        query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
        if parsed.path.startswith(LIST_PATH + "/"):
            name = urllib.parse.unquote(parsed.path[len(LIST_PATH) + 1:])
            with self.lock:
                self.info_paths.append(raw_path)
            if name not in self.objects:
                return 404, b"not found", 9
            if self._take(self.fail_info, name):
                return 500, b"boom", 4
            item = self.item(name)
            item["size"] = str(int(item["size"]) + self.info_size_delta.get(name, 0))
            body = json.dumps({key: item[key] for key in query["fields"].split(",")}).encode()
            return 200, body, len(body)
        if parsed.path.startswith(OBJECT_PATH):
            name = PREFIX + urllib.parse.unquote(parsed.path[len(OBJECT_PATH):])
            if name in self.objects and self._take(self.short, name):
                with self.lock:
                    self.object_gets.append(name)
                body = self.objects[name][:-7]
                return 200, body, len(body)
            if name in self.objects and self._take(self.partial, name):
                with self.lock:
                    self.object_gets.append(name)
                body = self.objects[name][:len(self.objects[name])//2]
                return 206, body, len(body)
            if name in self.objects and self._take(self.drop, name):
                with self.lock:
                    self.object_gets.append(name)
                body = self.objects[name]
                return 200, body[:len(body)//2], len(body)
        status, body = self.handle(parsed.path, query)
        return status, body, len(body)


@pytest.fixture
def raw(monkeypatch):
    for key in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(ix, "PAGE_SIZE", 3)
    monkeypatch.setattr(dl, "CHUNK_BYTES", 4096)   # 让每个文件分很多块写
    fake = RawBucket()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, body, declared = fake.respond(self.path)
            self.send_response(status)
            self.send_header("Content-Length", str(declared))
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake.port = server.server_address[1]
    yield fake
    server.shutdown()
    server.server_close()


# 7 个同分布场景各 1 条（一条失败回合）、1 条留出实验室回合：测试、验证各 1 个场景，训练顺序 5 条，
# N = 4、片大小 3 → train-000（3 条）、train-001（1 条）、val、id_test、ood_test。
SPECS = [dict(lab=lab, stamp=f"2023-07-07-09h-4{i}m-3{i}s", scene_id=1000 + i, left="ext2" if i % 2 else "ext1",
              outcome="failure" if i == 3 else "success")
         for i, lab in enumerate(("AUTOLab", "IPRL", "TRI", "CLVR", "AUTOLab", "IRIS", "RAD"))]
SPECS.append(dict(lab="PennPAL", stamp="2023-07-15-17h-18m-49s", scene_id=-1))


def _payload(seed: int, size: int) -> bytes:
    return random.Random(seed).randbytes(size)


def _cfg(fake, **data):
    return _config(fake, count=len(SPECS), **dict(SMALL, **data))


def build(fake: RawBucket, root: Path, **data):
    """建索引和划分，再把每条入选回合的 h5 和 MP4 放进假存储桶。返回 (配置, 入选回合 {uuid: 行})。"""
    for spec in SPECS:
        name, value = meta(**spec)
        fake.put(name, value)
    config = _cfg(fake, **data)
    ix.run_index(config, root, sleep=lambda s: None, threads=2)
    episodes = {e["uuid"]: e for e in sp.read_episodes(ix.index_dir(root)) if e["selected"]}
    for k, e in enumerate(sorted(episodes.values(), key=lambda e: e["uuid"])):
        for local, name in dl.episode_files(e, config.data):
            fake.put(name, _payload(zlib.crc32(f"{k}/{local}".encode()), 30_000 + 1_000*k + (7 if local.endswith(".mp4") else 0)))
    fake.object_gets.clear()
    fake.info_paths.clear()
    return config, episodes


def plan(config, root):
    return sp.shard_plan(sp.load_splits(root, config), config.data)


def run(config, root, shard, slept=None, **kw):
    return dl.run_download(config, root, shard, sleep=(slept.append if slept is not None else lambda s: None),
                           clock=lambda: NOW, **kw)


def edir(root, shard, uuid):
    return Path(root)/"raw"/shard/uuid


def fetched(fake):
    """内容接口被请求过的对象，按回合目录名（uuid）归类不方便，直接给对象名多重集合。"""
    return sorted(fake.object_gets)


# ---------------------------------------------------------------- 正常下载、记录内容

def test_shard_plan_of_the_fixture(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    p = plan(config, tmp_path)
    assert list(p) == ["train-000", "train-001", "val", "id_test", "ood_test"]
    assert [len(p[k]) for k in p] == [3, 1, 1, 1, 1]


def test_normal_download_files_metadata_and_records(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    shard = "train-000"
    uuids = plan(config, tmp_path)[shard]
    result = run(config, tmp_path, shard)
    assert result["counts"] == {"downloaded": 3, "skipped": 0, "raw_deleted": 0}
    rows = {json.loads(line)["_name"]: json.loads(line)
            for line in (tmp_path/"index"/"metadata.jsonl").read_text(encoding="utf-8").splitlines()}
    shas = []
    for uuid in uuids:
        e, d = episodes[uuid], edir(tmp_path, shard, uuid)
        files = dl.episode_files(e, config.data)
        assert files == [("trajectory.h5", PREFIX + e["h5"]), (f"{e['left_serial']}.mp4", PREFIX + e["left_mp4"])]
        assert sorted(p.name for p in d.iterdir()) == sorted(["DOWNLOADED.json", "metadata.json", *(f for f, _ in files)])
        for local, name in files:
            assert (d/local).read_bytes() == raw.objects[name]
        # metadata.json：官方字段、原键顺序、不含 _name/_md5/_size
        row = rows[f"{PREFIX}{e['folder']}/metadata_{uuid}.json"]
        official = json.loads((d/"metadata.json").read_text(encoding="utf-8"))
        assert list(official) == [k for k in row if not k.startswith("_")] and official["uuid"] == uuid
        assert (d/"metadata.json").read_bytes() == (json.dumps(official, ensure_ascii=False, indent=2) + "\n").encode()
        # DOWNLOADED.json：对象名、大小、MD5（base64）、本地 SHA-256、下载时间
        record = json.loads((d/"DOWNLOADED.json").read_text(encoding="utf-8"))
        assert record == {"schema": "droid_downloaded_v1", "uuid": uuid, "downloaded_at": "2026-11-09T15:00:00+00:00",
                          "files": {local: {"object": name, "size": len(raw.objects[name]),
                                            "md5": base64.b64encode(hashlib.md5(raw.objects[name]).digest()).decode(),
                                            "sha256": hashlib.sha256(raw.objects[name]).hexdigest()}
                                    for local, name in files}}
        assert (d/"DOWNLOADED.json").read_bytes() == cache._json_bytes(record)
        shas.append(hashlib.sha256((d/"DOWNLOADED.json").read_bytes()).hexdigest())
    # 片完成记录
    done = tmp_path/"raw"/shard/"_complete"/"pilot.json"
    complete = json.loads(done.read_text(encoding="utf-8"))
    assert complete == {"schema": "droid_download_complete_v1", "shard": shard, "config": "pilot",
                        "config_sha256": config.config_sha256,
                        "splits_sha256": cache.file_sha(tmp_path/"index"/"splits.json"),
                        "uuids": uuids, "downloaded_sha256": shas, "code_commit": ix.code_commit()}
    assert done.read_bytes() == cache._json_bytes(complete)
    assert {k: v for k, v in result.items() if k != "counts"} == complete
    # 只请求了这片的 6 个对象，每个一次元数据、一次内容；对象名在元数据地址里整体编码（含斜杠），只取 size 和 md5Hash
    names = sorted(name for u in uuids for _, name in dl.episode_files(episodes[u], config.data))
    assert fetched(raw) == names
    assert sorted(urllib.parse.unquote(p.split("?")[0][len(LIST_PATH) + 1:]) for p in raw.info_paths) == names
    for path in raw.info_paths:
        encoded, query = path[len(LIST_PATH) + 1:].split("?")
        assert "/" not in encoded and "%2F" in encoded and query == "fields=size%2Cmd5Hash"
    assert not list((tmp_path/"raw").rglob("*.part")) and not list((tmp_path/"raw").rglob("*.partial"))


def test_each_shard_downloads_its_planned_episodes(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    p = plan(config, tmp_path)
    for shard in p:
        run(config, tmp_path, shard)
        assert sorted(d.name for d in (tmp_path/"raw"/shard).iterdir() if d.name != "_complete") == sorted(p[shard])
    # 训练顺序第 5 条（N = 4 之外）没有下载
    order = sp.load_splits(tmp_path, config)["train_order"]
    assert not any((tmp_path/"raw"/s/order[4]).exists() for s in p)


def test_large_file_streams_in_many_chunks(raw, tmp_path):
    """CHUNK_BYTES 调成 4096：一个约 1 MB 的文件分 245 块写，字节和 SHA 都对。"""
    config, episodes = build(raw, tmp_path)
    uuid = plan(config, tmp_path)["val"][0]
    local, name = dl.episode_files(episodes[uuid], config.data)[1]
    raw.put(name, _payload(9, 1_000_003))
    run(config, tmp_path, "val")
    d = edir(tmp_path, "val", uuid)
    assert (d/local).read_bytes() == raw.objects[name]
    record = json.loads((d/"DOWNLOADED.json").read_text(encoding="utf-8"))["files"][local]
    assert record["size"] == 1_000_003 and record["sha256"] == hashlib.sha256(raw.objects[name]).hexdigest()


# ---------------------------------------------------------------- 校验失败、截断、500、404

def _one(config, episodes, root, shard="val"):
    uuid = plan(config, root)[shard][0]
    return uuid, dl.episode_files(episodes[uuid], config.data)


def test_md5_wrong_is_redownloaded(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    uuid, files = _one(config, episodes, tmp_path)
    raw.corrupt[files[1][1]] = 2
    slept = []
    run(config, tmp_path, "val", slept)
    assert (edir(tmp_path, "val", uuid)/files[1][0]).read_bytes() == raw.objects[files[1][1]]
    assert raw.object_gets.count(files[1][1]) == 3 and slept == []   # 校验失败重下不等待
    assert len(raw.info_paths) == 2                              # 对象元数据每个文件只取一次


def test_md5_wrong_three_times_fails_naming_the_object(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    uuid, files = _one(config, episodes, tmp_path)
    raw.corrupt[files[1][1]] = 3
    with pytest.raises(RuntimeError, match=re.escape(f"object {files[1][1]}: downloaded 3 times")):
        run(config, tmp_path, "val")
    d = edir(tmp_path, "val", uuid)
    assert raw.object_gets.count(files[1][1]) == 3
    assert not (d/files[1][0]).exists() and not (d/(files[1][0] + ".part")).exists()
    assert not (d/"DOWNLOADED.json").exists() and not (tmp_path/"raw"/"val"/"_complete").exists()


def test_short_content_counts_as_a_bad_download(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    uuid, files = _one(config, episodes, tmp_path)
    raw.short[files[0][1]] = 2
    run(config, tmp_path, "val")
    assert (edir(tmp_path, "val", uuid)/"trajectory.h5").read_bytes() == raw.objects[files[0][1]]
    assert raw.object_gets.count(files[0][1]) == 3
    raw.short[files[0][1]] = 3
    (tmp_path/"raw"/"val"/"_complete"/"pilot.json").unlink()
    (edir(tmp_path, "val", uuid)/"DOWNLOADED.json").unlink()
    (edir(tmp_path, "val", uuid)/"trajectory.h5").unlink()
    with pytest.raises(RuntimeError, match="never matched the listed size"):
        run(config, tmp_path, "val")


def test_dropped_connection_is_retried_with_backoff(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    uuid, files = _one(config, episodes, tmp_path)
    raw.drop[files[1][1]] = 2
    slept = []
    run(config, tmp_path, "val", slept)
    assert (edir(tmp_path, "val", uuid)/files[1][0]).read_bytes() == raw.objects[files[1][1]]
    assert slept == [1, 2] and raw.object_gets.count(files[1][1]) == 3


def test_dropped_connection_every_time_fails(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    uuid, files = _one(config, episodes, tmp_path)
    raw.drop[files[1][1]] = 99
    slept = []
    with pytest.raises(RuntimeError, match=r"still failing after 6 attempts.*connection closed after"):
        run(config, tmp_path, "val", slept)
    assert slept == [1, 2, 4, 8, 16] and not (tmp_path/"raw"/"val"/"_complete").exists()


def test_500_is_retried_for_content_and_object_metadata(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    uuid, files = _one(config, episodes, tmp_path)
    raw.fail_objects[files[0][1]] = 2
    raw.fail_info[files[1][1]] = 3
    slept = []
    run(config, tmp_path, "val", slept)
    assert sorted(slept) == sorted([1, 2, 1, 2, 4])
    assert (edir(tmp_path, "val", uuid)/"DOWNLOADED.json").exists()


def test_500_forever_fails_after_retries(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    uuid, files = _one(config, episodes, tmp_path)
    raw.fail_info[files[0][1]] = 99
    with pytest.raises(RuntimeError, match=r"object metadata .*still failing after 6 attempts"):
        run(config, tmp_path, "val")


def test_other_2xx_status_is_a_transport_failure(raw, tmp_path):
    """只认 200：206（部分内容）按传输失败重试（等 1 秒），不当成一次下载去核大小。"""
    config, episodes = build(raw, tmp_path)
    uuid, files = _one(config, episodes, tmp_path)
    raw.partial[files[0][1]] = 1
    slept = []
    run(config, tmp_path, "val", slept)
    assert slept == [1] and raw.object_gets.count(files[0][1]) == 2


def test_size_and_md5_must_both_match(raw, tmp_path):
    """对象元数据报的大小不对（MD5 对）：3 次都不算对，失败；只核 MD5 不够。"""
    config, episodes = build(raw, tmp_path)
    uuid, files = _one(config, episodes, tmp_path)
    raw.info_size_delta[files[0][1]] = 1
    with pytest.raises(RuntimeError, match="downloaded 3 times"):
        run(config, tmp_path, "val")
    assert not (edir(tmp_path, "val", uuid)/"trajectory.h5").exists()


def test_404_fails_at_once(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    uuid, files = _one(config, episodes, tmp_path)
    del raw.objects[files[1][1]]
    slept = []
    with pytest.raises(dl.ObjectNotFound, match="HTTP 404"):
        run(config, tmp_path, "val", slept)
    assert slept == [] and sum(files[1][1] in urllib.parse.unquote(p) for p in raw.info_paths) == 1
    assert not (edir(tmp_path, "val", uuid)/"DOWNLOADED.json").exists()


def test_404_on_content_fails_at_once(raw, tmp_path, monkeypatch):
    config, episodes = build(raw, tmp_path)
    uuid, files = _one(config, episodes, tmp_path)
    monkeypatch.setattr(dl, "object_info", lambda data, name, sleep: {"size": 5, "md5": "x"})
    monkeypatch.setattr(ix, "object_url", lambda data, name: f"http://127.0.0.1:{raw.port}{OBJECT_PATH}missing/x.h5")
    slept = []
    with pytest.raises(dl.ObjectNotFound):
        run(config, tmp_path, "val", slept)
    assert slept == []


@pytest.mark.parametrize("body, message", [
    ({"size": 12, "md5Hash": "x"}, "size 12 is not a decimal string"),
    ({"size": "12"}, "md5Hash None is not a base64 MD5 digest"),
    ({"size": "12", "md5Hash": base64.b64encode(b"short").decode()}, "is not a base64 MD5 digest"),
    ([1], "is not a JSON object"),
])
def test_object_metadata_body_checks(body, message):
    with pytest.raises(ValueError, match=message):
        dl._parse_object_info(json.dumps(body).encode(), "robotics/droid_raw/1.0.1/x")


def test_object_api_url_encodes_whole_name_and_needs_prefix(raw, tmp_path):
    config = _cfg(raw)
    url = dl.object_api_url(config.data, PREFIX + "AUTOLab/success/a b+c/trajectory.h5")
    assert url == (config.data.list_api + "/robotics%2Fdroid_raw%2F1.0.1%2FAUTOLab%2Fsuccess%2Fa%20b%2Bc%2F"
                   "trajectory.h5?fields=size%2Cmd5Hash")
    with pytest.raises(ValueError, match="not under prefix"):
        dl.object_api_url(config.data, "other/x.h5")


# ---------------------------------------------------------------- 残留、跳过、重跑

def test_leftover_part_files_are_deleted_and_redownloaded(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    uuid, files = _one(config, episodes, tmp_path)
    d = edir(tmp_path, "val", uuid)
    d.mkdir(parents=True)
    for local, _ in files:
        (d/(local + ".part")).write_bytes(b"half a file")
    (d/"DOWNLOADED.json.partial").write_bytes(b"{")
    (d/"metadata.json.partial").write_bytes(b"{")
    run(config, tmp_path, "val")
    assert sorted(p.name for p in d.iterdir()) == sorted(["DOWNLOADED.json", "metadata.json", *(f for f, _ in files)])
    for local, name in files:
        assert (d/local).read_bytes() == raw.objects[name]


def test_downloaded_with_right_sizes_is_skipped(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    run(config, tmp_path, "train-000")
    uuids = plan(config, tmp_path)["train-000"]
    first = edir(tmp_path, "train-000", uuids[0])
    record_bytes = (first/"DOWNLOADED.json").read_bytes()
    # 内容改了但大小不变：不重算 MD5，照样跳过（§2）
    h5 = first/"trajectory.h5"
    h5.write_bytes(bytes([h5.read_bytes()[0] ^ 1]) + h5.read_bytes()[1:])
    (first/"leftover.part").write_bytes(b"x")
    (tmp_path/"raw"/"train-000"/"_complete"/"pilot.json").unlink()
    raw.object_gets.clear()
    raw.info_paths.clear()
    result = run(config, tmp_path, "train-000")
    assert result["counts"] == {"downloaded": 0, "skipped": 3, "raw_deleted": 0}
    assert raw.object_gets == [] and raw.info_paths == []
    assert (first/"DOWNLOADED.json").read_bytes() == record_bytes and not (first/"leftover.part").exists()


def test_downloaded_with_wrong_size_is_redownloaded(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    run(config, tmp_path, "val")
    uuid, files = _one(config, episodes, tmp_path)
    d = edir(tmp_path, "val", uuid)
    (d/files[1][0]).write_bytes(b"cut")
    (tmp_path/"raw"/"val"/"_complete"/"pilot.json").unlink()
    raw.object_gets.clear()
    result = run(config, tmp_path, "val")
    assert result["counts"]["downloaded"] == 1
    assert raw.object_gets == [files[1][1]]          # h5 大小和 MD5 都对，不重下；MP4 重下
    assert (d/files[1][0]).read_bytes() == raw.objects[files[1][1]]


def test_target_present_without_record_is_verified_not_trusted(raw, tmp_path):
    """上次在两个文件之间中断：h5 已换名、没有 DOWNLOADED.json。h5 对就直接用，不对就重下。"""
    config, episodes = build(raw, tmp_path)
    uuid, files = _one(config, episodes, tmp_path)
    d = edir(tmp_path, "val", uuid)
    d.mkdir(parents=True)
    (d/"trajectory.h5").write_bytes(raw.objects[files[0][1]])
    run(config, tmp_path, "val")
    assert raw.object_gets == [files[1][1]]
    (tmp_path/"raw"/"val"/"_complete"/"pilot.json").unlink()
    (d/"DOWNLOADED.json").unlink()
    good = raw.objects[files[0][1]]
    (d/"trajectory.h5").write_bytes(bytes([good[0] ^ 1]) + good[1:])   # 同样大小、内容不同
    raw.object_gets.clear()
    run(config, tmp_path, "val")
    assert raw.object_gets == [files[0][1]] and (d/"trajectory.h5").read_bytes() == good
    # 不对的旧文件先删：重下也失败时不留下它
    (tmp_path/"raw"/"val"/"_complete"/"pilot.json").unlink()
    (d/"DOWNLOADED.json").unlink()
    (d/"trajectory.h5").write_bytes(bytes([good[0] ^ 1]) + good[1:])
    raw.corrupt[files[0][1]] = 3
    with pytest.raises(RuntimeError, match="downloaded 3 times"):
        run(config, tmp_path, "val")
    assert not (d/"trajectory.h5").exists()


def test_raw_deleted_is_skipped_without_download(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    run(config, tmp_path, "train-000")
    uuids = plan(config, tmp_path)["train-000"]
    first = edir(tmp_path, "train-000", uuids[0])
    sha = cache.file_sha(first/"DOWNLOADED.json")
    for local, _ in dl.episode_files(episodes[uuids[0]], config.data):
        (first/local).unlink()
    (first/"RAW_DELETED.json").write_bytes(b'{"deleted_at": "x"}\n')
    (tmp_path/"raw"/"train-000"/"_complete"/"pilot.json").unlink()
    raw.object_gets.clear()
    raw.info_paths.clear()
    result = run(config, tmp_path, "train-000")
    assert result["counts"] == {"downloaded": 0, "skipped": 2, "raw_deleted": 1}
    assert raw.object_gets == [] and raw.info_paths == [] and result["downloaded_sha256"][0] == sha
    assert not (first/"trajectory.h5").exists()


def test_raw_deleted_without_downloaded_record_fails(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    uuid, _ = _one(config, episodes, tmp_path)
    d = edir(tmp_path, "val", uuid)
    d.mkdir(parents=True)
    (d/"RAW_DELETED.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="has RAW_DELETED.json but no DOWNLOADED.json"):
        run(config, tmp_path, "val")


@pytest.mark.parametrize("edit", [
    lambda r: r.update(uuid="other"),
    lambda r: r.update(schema="droid_downloaded_v0"),
    lambda r: r["files"].pop("trajectory.h5"),
    lambda r: r["files"]["trajectory.h5"].update(object="robotics/droid_raw/1.0.1/elsewhere/trajectory.h5"),
])
def test_downloaded_record_for_other_objects_is_refused(raw, tmp_path, edit):
    config, episodes = build(raw, tmp_path)
    run(config, tmp_path, "val")
    uuid, _ = _one(config, episodes, tmp_path)
    path = edir(tmp_path, "val", uuid)/"DOWNLOADED.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    edit(record)
    path.write_text(json.dumps(record), encoding="utf-8")
    (tmp_path/"raw"/"val"/"_complete"/"pilot.json").unlink()
    with pytest.raises(ValueError, match="does not record episode"):
        run(config, tmp_path, "val")


def test_interrupted_run_resumes_without_redownloading(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    uuids = plan(config, tmp_path)["train-000"]
    late = dl.episode_files(episodes[uuids[2]], config.data)[1][1]
    raw.fail_objects[late] = 99
    with pytest.raises(RuntimeError, match="still failing"):
        run(config, tmp_path, "train-000", threads=1)
    assert not (tmp_path/"raw"/"train-000"/"_complete").exists()
    assert [(edir(tmp_path, "train-000", u)/"DOWNLOADED.json").exists() for u in uuids] == [True, True, False]
    raw.fail_objects.clear()
    raw.object_gets.clear()
    result = run(config, tmp_path, "train-000")
    assert raw.object_gets == [late]                 # 第三条的 h5 已在且 MD5 对，只补 MP4
    assert result["counts"] == {"downloaded": 1, "skipped": 2, "raw_deleted": 0}


def test_complete_record_present_does_nothing(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    first = run(config, tmp_path, "val")
    raw.object_gets.clear()
    raw.info_paths.clear()
    with mock.patch.object(sp, "load_splits", side_effect=AssertionError("must not read")):
        again = run(config, tmp_path, "val")
    assert raw.object_gets == [] and raw.info_paths == []
    assert again == {k: v for k, v in first.items() if k != "counts"}


def test_other_config_reuses_downloaded_episodes(raw, tmp_path):
    """同一数据根：小 N 的配置先下完 train-000 的前 2 条，大 N 的配置只补第 3 条，各写自己的完成记录。"""
    small, episodes = build(raw, tmp_path, train_episodes=2)
    run(small, tmp_path, "train-000")
    big = _cfg(raw)
    with mock.patch.object(config_module, "OOD_TEST_EPISODES", 1):
        big = dataclasses.replace(big, name="full", config_sha256="f"*64)
    raw.object_gets.clear()
    result = run(big, tmp_path, "train-000")
    third = plan(big, tmp_path)["train-000"][2]
    assert sorted(raw.object_gets) == sorted(n for _, n in dl.episode_files(episodes[third], big.data))
    assert result["counts"] == {"downloaded": 1, "skipped": 2, "raw_deleted": 0}
    a = json.loads((tmp_path/"raw"/"train-000"/"_complete"/"pilot.json").read_text(encoding="utf-8"))
    b = json.loads((tmp_path/"raw"/"train-000"/"_complete"/"full.json").read_text(encoding="utf-8"))
    assert b["uuids"][:2] == a["uuids"] and b["downloaded_sha256"][:2] == a["downloaded_sha256"]
    assert len(b["uuids"]) == 3 and b["config_sha256"] == "f"*64


def test_shard_not_used_by_config_is_refused(raw, tmp_path):
    config, _ = build(raw, tmp_path)
    with pytest.raises(ValueError, match="shard 'train-002' is not used by config 'pilot'"):
        run(config, tmp_path, "train-002")


def test_index_must_be_intact(raw, tmp_path):
    config, _ = build(raw, tmp_path)
    with (tmp_path/"index"/"splits.json").open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError, match="does not match MANIFEST.json"):
        run(config, tmp_path, "val")


def test_official_metadata_needs_exactly_the_three_internal_fields():
    assert dl.official_metadata({"b": 1, "_name": "n", "a": 2, "_md5": "m", "_size": 3}) == {"b": 1, "a": 2}
    assert list(dl.official_metadata({"b": 1, "_name": "n", "a": 2, "_md5": "m", "_size": 3})) == ["b", "a"]
    with pytest.raises(ValueError, match="internal keys"):
        dl.official_metadata({"b": 1, "_name": "n", "_md5": "m"})
    with pytest.raises(ValueError, match="internal keys"):
        dl.official_metadata({"_name": "n", "_md5": "m", "_size": 3, "_extra": 1})


def test_metadata_json_rewritten_when_different(raw, tmp_path):
    config, episodes = build(raw, tmp_path)
    run(config, tmp_path, "val")
    uuid, _ = _one(config, episodes, tmp_path)
    path = edir(tmp_path, "val", uuid)/"metadata.json"
    good = path.read_bytes()
    path.write_bytes(b"{}\n")
    (tmp_path/"raw"/"val"/"_complete"/"pilot.json").unlink()
    run(config, tmp_path, "val")
    assert path.read_bytes() == good


def test_main_downloads_one_shard(raw, tmp_path, capsys):
    build(raw, tmp_path)
    base = yaml.safe_load((CONFIGS/"base.yaml").read_text(encoding="utf-8"))
    host = f"http://127.0.0.1:{raw.port}"
    base["data"].update(list_api=host + LIST_PATH, object_base=host + OBJECT_PATH, expected_metadata_files=len(SPECS),
                        **dict(SMALL, ood_quota=list(SMALL["ood_quota"])))
    configs = tmp_path/"configs"
    configs.mkdir()
    (configs/"base.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    pilot = yaml.safe_load((CONFIGS/"pilot.yaml").read_text(encoding="utf-8"))
    pilot["data"]["train_episodes"] = SMALL["train_episodes"]
    (configs/"pilot.yaml").write_text(yaml.safe_dump(pilot, sort_keys=False), encoding="utf-8")
    with mock.patch.object(config_module, "OOD_TEST_EPISODES", 1):
        assert dl.main(["--config", str(configs/"pilot.yaml"), "--data-root", str(tmp_path), "--shard", "id_test"]) == 0
        out = capsys.readouterr().out
        assert '"shard": "id_test"' in out and '"downloaded": 1' in out
        assert dl.main(["--config", str(configs/"pilot.yaml"), "--data-root", str(tmp_path), "--shard", "id_test"]) == 0
        assert "already complete" in capsys.readouterr().out
