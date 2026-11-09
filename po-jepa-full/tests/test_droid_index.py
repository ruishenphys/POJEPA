"""S02：DROID 索引（docs/design/02-droid-index-and-splits.md §1–§4、§7、§8；验收 14 §3 S02）。

网络全部换成本机的假存储桶（http.server，只服务内存里的对象），模拟列表接口（翻页）和对象内容接口，
可以按测试需要返回 500、错误内容。真实元数据上的核对在 `test_real_index_counts`，没有真实索引时跳过。
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import json
import os
import threading
import urllib.parse
import zlib
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest
import yaml

from po_jepa.droid import index as ix
from po_jepa.droid import config as config_module
from po_jepa.droid.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/"configs"/"droid"
PREFIX = "robotics/droid_raw/1.0.1/"
LIST_PATH = "/storage/v1/b/gresearch/o"
OBJECT_PATH = "/gresearch/robotics/droid_raw/1.0.1/"


# ---------------------------------------------------------------- 假存储桶

class FakeBucket:
    """内存里的对象；`fail_pages[k]`/`fail_objects[name]` 是接下来要返回 500 的次数，`corrupt[name]` 是要返回错误内容的次数。"""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.fail_pages: dict[int, int] = {}
        self.fail_objects: dict[str, int] = {}
        self.corrupt: dict[str, int] = {}
        self.list_queries: list[dict] = []
        self.object_gets: list[str] = []
        self.lock = threading.Lock()

    def put(self, name: str, value) -> None:
        self.objects[name] = value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")

    def _take(self, table: dict, key) -> bool:
        with self.lock:
            if table.get(key, 0) > 0:
                table[key] -= 1
                return True
            return False

    def handle(self, path: str, query: dict) -> tuple[int, bytes]:
        if path == LIST_PATH:
            with self.lock:
                self.list_queries.append(query)
            start = int(query.get("pageToken", "0"))
            page = start // int(query["maxResults"])
            if self._take(self.fail_pages, page):
                return 500, b"boom"
            names = sorted(n for n in self.objects if n.startswith(query["prefix"])
                           and n.rsplit("/", 1)[-1].startswith("metadata_") and n.endswith(".json"))
            names = names[::-1]   # 存储桶的返回顺序不保证：倒序返回，检查代码自己排序
            chunk = names[start:start + int(query["maxResults"])]
            body = {"items": [self.item(n) for n in chunk]}
            if start + len(chunk) < len(names):
                body["nextPageToken"] = str(start + len(chunk))
            return 200, json.dumps(body).encode("utf-8")
        if path.startswith(OBJECT_PATH):
            name = PREFIX + urllib.parse.unquote(path[len(OBJECT_PATH):])
            with self.lock:
                self.object_gets.append(name)
            if name not in self.objects:
                return 404, b"not found"
            if self._take(self.fail_objects, name):
                return 500, b"boom"
            body = self.objects[name]
            if self._take(self.corrupt, name):
                body = bytes([body[0] ^ 1]) + body[1:]   # 同样长度、内容不同 → MD5 不对
            return 200, body
        return 404, b"unknown path"

    def item(self, name: str) -> dict:
        body = self.objects[name]
        return {"name": name, "size": str(len(body)),
                "md5Hash": base64.b64encode(hashlib.md5(body).digest()).decode(),
                "crc32c": base64.b64encode(zlib.crc32(body).to_bytes(4, "big")).decode(),
                "updated": "2024-03-18T10:03:49.256Z"}


@pytest.fixture
def bucket(monkeypatch):
    for key in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(ix, "PAGE_SIZE", 3)
    fake = FakeBucket()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
            status, body = fake.handle(parsed.path, query)
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake.port = server.server_address[1]
    yield fake
    server.shutdown()
    server.server_close()


# 几条合成回合也能划分（S03 的 run_index 在写清单前写 splits.json；划分本身在 test_droid_splits.py 里测）。
# 留出实验室测试集正式是 500 条（config.OOD_TEST_EPISODES），构造这份配置时临时改成 1。
SMALL_SPLITS = dict(id_test_scenes=1, val_scenes=1, episodes_per_scene=1, ood_quota=(1, 0, 0))


def _config(bucket: FakeBucket, count: int | None = None, **data):
    cfg = load_config(CONFIGS/"pilot.yaml")
    base = f"http://127.0.0.1:{bucket.port}"
    fields = dict(list_api=base + LIST_PATH, object_base=base + OBJECT_PATH.rstrip("/"),
                  expected_metadata_files=len(bucket.objects) if count is None else count, **SMALL_SPLITS)
    fields.update(data)
    with mock.patch.object(config_module, "OOD_TEST_EPISODES", sum(fields["ood_quota"])):
        return dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, **fields))


# ---------------------------------------------------------------- 合成元数据（字段和真实 1.0.1 一样）

def meta(lab="AUTOLab", outcome="success", stamp="2023-07-07-09h-45m-39s", *, user="5d05c5aa", steps=300,
         scene_id=5207831207, success=None, left="ext1", meta_lab=None, folder_lab=None):
    """返回 (对象名, 元数据)。文件夹名按真实格式 <outcome>/<日期>/<时间文件夹>/。"""
    uuid = f"{lab}+{user}+{stamp}"
    date = stamp[:10]
    sub = f"{outcome}/{date}/Fri_Jul__7_{stamp[11:13]}:{stamp[15:17]}:{stamp[19:21]}_2023"
    ext1, ext2 = "22008760", "24400334"
    left_serial = {"ext1": ext1, "ext2": ext2}[left]
    value = {
        "uuid": uuid, "lab": meta_lab or lab, "user": "Someone", "user_id": user, "date": date, "timestamp": stamp,
        "hdf5_path": f"{sub}/trajectory.h5", "building": "BAIR", "scene_id": scene_id,
        "success": (outcome == "success") if success is None else success, "robot_serial": "fr3-295341-1326595",
        "r2d2_version": "1.3", "current_task": "task", "trajectory_length": steps, "wrist_cam_serial": "18026681",
        "ext1_cam_serial": ext1, "ext2_cam_serial": ext2,
        "ext1_mp4_path": f"{sub}/recordings/MP4/{ext1}.mp4", "ext2_mp4_path": f"{sub}/recordings/MP4/{ext2}.mp4",
        "left_mp4_path": f"{sub}/recordings/MP4/{left_serial}.mp4",
        "right_mp4_path": f"{sub}/recordings/MP4/{ext2 if left == 'ext1' else ext1}.mp4",
    }
    name = f"{PREFIX}{folder_lab or lab}/{sub}/metadata_{uuid}.json"
    return name, value


def _row(name, value):
    return dict(value, _name=name, _md5="x", _size=1)


def _fill(bucket, specs):
    names = []
    for kw in specs:
        name, value = meta(**kw)
        bucket.put(name, value)
        names.append(name)
    return names


SEVEN = [dict(stamp=f"2023-07-07-09h-4{i}m-39s") for i in range(5)] + [
    dict(lab="PennPAL", stamp="2023-07-15-17h-18m-49s", scene_id=-1),
    dict(lab="TRI", meta_lab="tri", outcome="failure", stamp="2023-08-07-17h-12m-42s", left="ext2", scene_id=4242),
]


# ---------------------------------------------------------------- §1 列表

def test_listing_pages_sorts_and_writes(bucket, tmp_path):
    names = _fill(bucket, SEVEN)
    bucket.put(PREFIX + "AUTOLab/success/2023-07-07/x/trajectory.h5", b"not metadata")   # 不匹配 glob，不在列表里
    cfg = _config(bucket, count=7)
    manifest = ix.run_index(cfg, tmp_path)
    assert len(bucket.list_queries) == 3                       # 7 条、每页 3 条
    assert [q.get("pageToken") for q in bucket.list_queries] == [None, "3", "6"]
    first = bucket.list_queries[0]
    assert first == {"prefix": PREFIX, "matchGlob": "**/metadata_*.json", "maxResults": "3",
                     "fields": "items(name,size,md5Hash,crc32c,updated),nextPageToken"}
    lines = (tmp_path/"index"/"listing.jsonl").read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines]
    assert [i["name"] for i in items] == sorted(names)
    assert all(line == json.dumps(i, sort_keys=True, separators=(",", ":")) for line, i in zip(lines, items))
    assert manifest["files"]["listing.jsonl"]["lines"] == 7
    assert not list((tmp_path/"index").glob("*.partial"))


def test_listing_retry_intervals_then_success(bucket):
    _fill(bucket, SEVEN)
    bucket.fail_pages[1] = 5
    slept = []
    items = ix.list_metadata(_config(bucket, count=7).data, sleep=slept.append)
    assert slept == [1, 2, 4, 8, 16]
    assert len(items) == 7


def test_listing_retries_exhausted(bucket):
    _fill(bucket, SEVEN)
    bucket.fail_pages[2] = 6
    slept = []
    with pytest.raises(RuntimeError, match=r"listing page 2 .*still failing after 6 attempts"):
        ix.list_metadata(_config(bucket, count=7).data, sleep=slept.append)
    assert slept == [1, 2, 4, 8, 16]


def test_listing_count_must_match(bucket):
    _fill(bucket, SEVEN)
    with pytest.raises(ValueError, match="listing has 7 metadata files, expected 74896"):
        ix.list_metadata(_config(bucket, count=74896).data)
    with pytest.raises(ValueError, match="listing has 7 metadata files, expected 8"):
        ix.list_metadata(_config(bucket, count=8).data)


def test_listing_item_checks():
    data = load_config(CONFIGS/"pilot.yaml").data
    good = {"name": PREFIX + "a/success/x/metadata_u.json", "size": "12", "md5Hash": base64.b64encode(b"0" * 16).decode()}
    assert ix._check_item(dict(good), data) == good
    for bad, message in ((dict(good, name="other/a.json"), "not under prefix"), (dict(good, size=12), "decimal"),
                         (dict(good, size="-1"), "decimal"), (dict(good, md5Hash="abc"), "MD5"),
                         ({k: v for k, v in good.items() if k != "md5Hash"}, "MD5"), ("x", "not an object")):
        with pytest.raises(ValueError, match=message):
            ix._check_item(bad, data)


def test_listing_page_checks():
    with pytest.raises(ValueError, match="not a JSON object"):
        ix._list_page(b"[]", 0)
    with pytest.raises(ValueError, match="items is not a list"):
        ix._list_page(b'{"items": {}}', 0)
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        ix._list_page(b'{"items": [], "items": []}', 0)
    with pytest.raises(ValueError, match="non-finite"):
        ix._list_page(b'{"items": [], "x": NaN}', 0)


def test_listing_rejects_repeated_token_and_duplicate_names():
    data = load_config(CONFIGS/"pilot.yaml").data
    item = {"name": PREFIX + "a/success/x/metadata_u.json", "size": "1", "md5Hash": base64.b64encode(b"0" * 16).decode()}
    pages = iter([{"items": [item], "nextPageToken": "t"}, {"items": [], "nextPageToken": "t"}])
    with pytest.raises(ValueError, match="repeated nextPageToken"):
        ix.list_metadata(data, get=lambda url: json.dumps(next(pages)).encode())
    pages = iter([{"items": [item], "nextPageToken": "t"}, {"items": [item]}])
    with pytest.raises(ValueError, match="duplicate object names"):
        ix.list_metadata(data, get=lambda url: json.dumps(next(pages)).encode())


# ---------------------------------------------------------------- §2 元数据

def test_metadata_md5_wrong_is_retried(bucket, tmp_path):
    names = _fill(bucket, SEVEN)
    bucket.corrupt[names[3]] = 1
    slept = []
    ix.run_index(_config(bucket), tmp_path, sleep=slept.append)
    assert slept == [1]
    assert bucket.object_gets.count(names[3]) == 2


def test_metadata_md5_wrong_fails_naming_the_object(bucket, tmp_path):
    names = _fill(bucket, SEVEN)
    bucket.corrupt[names[3]] = 6
    slept = []
    with pytest.raises(RuntimeError, match=r"metadata object .*" + "still failing after 6 attempts.*MD5"):
        ix.run_index(_config(bucket), tmp_path, sleep=slept.append)
    assert names[3] in str(bucket.object_gets)
    assert sorted(slept) == [1, 2, 4, 8, 16]
    assert not (tmp_path/"index"/"metadata.jsonl").exists() and not (tmp_path/"index"/"MANIFEST.json").exists()


def test_metadata_500_and_size_checks(bucket):
    names = _fill(bucket, SEVEN)
    data = _config(bucket).data
    item = bucket.item(names[0])
    bucket.fail_objects[names[0]] = 2
    slept = []
    assert ix.fetch_object(data, item, sleep=slept.append) == bucket.objects[names[0]]
    assert slept == [1, 2]
    with pytest.raises(RuntimeError, match="size"):
        ix.fetch_object(data, dict(item, size=str(int(item["size"]) + 1)), sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="HTTPError 404"):
        ix.fetch_object(data, dict(item, name=PREFIX + "missing/metadata_x.json"), sleep=lambda s: None)


def test_metadata_rows_keep_official_order_and_add_three_fields(bucket, tmp_path):
    names = _fill(bucket, SEVEN)
    ix.run_index(_config(bucket), tmp_path)
    lines = (tmp_path/"index"/"metadata.jsonl").read_text(encoding="utf-8").splitlines()
    listed = sorted(names)
    for line, name in zip(lines, listed):
        row = json.loads(line)
        official = json.loads(bucket.objects[name])
        assert list(row) == list(official) + ["_name", "_md5", "_size"]
        assert row["_name"] == name and row["_size"] == len(bucket.objects[name])
        assert row["_md5"] == bucket.item(name)["md5Hash"]


def test_parse_metadata_rejects_bad_bodies():
    item = {"name": "n", "md5Hash": "m", "size": "1"}
    with pytest.raises(ValueError, match="not a JSON object"):
        ix.parse_metadata(b"[1]", item)
    with pytest.raises(ValueError, match="already has keys starting with '_'"):
        ix.parse_metadata(b'{"_name": 1}', item)


def test_object_url_quotes_path_and_needs_prefix():
    data = load_config(CONFIGS/"pilot.yaml").data
    name = PREFIX + "AUTOLab/failure/2023-07-07/Fri_Jul__7_09:45:39_2023/metadata_AUTOLab+5d05c5aa+x.json"
    assert ix.object_url(data, name) == ("https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1/AUTOLab/"
                                         "failure/2023-07-07/Fri_Jul__7_09%3A45%3A39_2023/metadata_AUTOLab%2B5d05c5aa%2Bx.json")
    with pytest.raises(ValueError, match="not under prefix"):
        ix.object_url(data, "robotics/droid_raw/1.0.0/x.json")


# ---------------------------------------------------------------- §3 统一字段

DATA = load_config(CONFIGS/"pilot.yaml").data


def test_normalize_fields():
    name, value = meta()
    e = ix.normalize(_row(name, value), DATA)
    folder = "AUTOLab/success/2023-07-07/Fri_Jul__7_09:45:39_2023"
    assert e == {"lab": "AUTOLab", "outcome": "success", "success_flag": True, "scene_id": 5207831207, "steps": 300,
                 "uuid": "AUTOLab+5d05c5aa+2023-07-07-09h-45m-39s", "folder": folder, "h5": folder + "/trajectory.h5",
                 "left_mp4": folder + "/recordings/MP4/22008760.mp4", "left_serial": "22008760", "left_is": "ext1",
                 "group": "id", "building": "BAIR", "date": "2023-07-07", "robot_serial": "fr3-295341-1326595",
                 "user_id": "5d05c5aa", "r2d2_version": "1.3"}


def test_normalize_lab_from_path_not_metadata():
    name, value = meta(lab="TRI", meta_lab="tri")
    e = ix.normalize(_row(name, value), DATA)
    assert e["lab"] == "TRI" and e["h5"].startswith("TRI/") and e["left_mp4"].startswith("TRI/")


def test_normalize_left_is_ext2_and_group_ood():
    name, value = meta(lab="RPL", left="ext2")
    e = ix.normalize(_row(name, value), DATA)
    assert (e["left_is"], e["left_serial"], e["group"]) == ("ext2", "24400334", "ood")


def test_normalize_serial_may_be_an_int_in_metadata():
    name, value = meta()
    value["ext1_cam_serial"] = 22008760
    assert ix.normalize(_row(name, value), DATA)["left_is"] == "ext1"


def _edit(**changes):
    name, value = meta()
    for key, new in changes.items():
        if new is KeyError:
            del value[key]
        else:
            value[key] = new(value) if callable(new) else new
    return _row(name, value)


BAD_FIELDS = [
    ("uuid", _edit(uuid="AUTOLab+5d05c5aa+2023-07-07-09h-45m-40s"), "does not match the file name"),
    ("h5 outside folder", _edit(hdf5_path="success/2023-07-07/Other/trajectory.h5"), "not inside the episode folder"),
    ("h5 sibling prefix", _edit(hdf5_path=lambda v: v["hdf5_path"].replace("/trajectory.h5", "x/trajectory.h5")),
     "not inside the episode folder"),
    ("mp4 not in MP4", _edit(left_mp4_path=lambda v: v["left_mp4_path"].replace("/MP4/", "/SVO/")), "recordings/MP4"),
    ("mp4 in subfolder", _edit(left_mp4_path=lambda v: v["left_mp4_path"].replace("/MP4/", "/MP4/x/")), "recordings/MP4"),
    ("mp4 suffix", _edit(left_mp4_path=lambda v: v["left_mp4_path"].replace(".mp4", ".svo")), "recordings/MP4"),
    ("mp4 bare", _edit(left_mp4_path=lambda v: v["left_mp4_path"].rsplit("/", 1)[0] + "/.mp4"), "recordings/MP4"),
    ("serial none", _edit(ext1_cam_serial="1", ext2_cam_serial="2"), "matches 0 of the external camera serials"),
    ("serial both", _edit(ext2_cam_serial="22008760"), "matches 2 of the external camera serials"),
    ("success type", _edit(success=1), "'success' must be a bool"),
    ("scene type", _edit(scene_id="5207831207"), "'scene_id' must be an int"),
    ("scene bool", _edit(scene_id=True), "'scene_id' must be an int"),
    ("steps type", _edit(trajectory_length=300.0), "'trajectory_length' must be an int"),
    ("steps zero", _edit(trajectory_length=0), "must be positive"),
    ("uuid type", _edit(uuid=5), "'uuid' must be a string"),
    ("missing uuid", _edit(uuid=KeyError), "has no 'uuid'"),
    ("missing scene", _edit(scene_id=KeyError), "has no 'scene_id'"),
    ("missing ext2", _edit(ext2_cam_serial=KeyError), "ext2_cam_serial"),
    ("missing building", _edit(building=KeyError), "building"),
    ("missing left", _edit(left_mp4_path=KeyError), "has no 'left_mp4_path'"),
]


@pytest.mark.parametrize("label,row,message", BAD_FIELDS, ids=[b[0] for b in BAD_FIELDS])
def test_normalize_rejects(label, row, message):
    with pytest.raises(ValueError, match=message) as info:
        ix.normalize(row, DATA)
    assert row["_name"] in str(info.value)


def test_normalize_rejects_bad_object_names():
    name, value = meta()
    for bad, message in ((name.replace("/success/", "/maybe/"), "outcome folder 'maybe'"),
                         (PREFIX + "AUTOLab/metadata_x.json", "must be <prefix>"),
                         (name.replace("/2023-07-07/", "//"), "must be <prefix>"),
                         (name.replace(PREFIX, "robotics/droid_raw/1.0.0/"), "not under prefix")):
        with pytest.raises(ValueError, match=message):
            ix.normalize(_row(bad, dict(value, success=True)), DATA)


def test_camera_key_comes_from_config():
    name, value = meta()
    value["left_mp4_path"] = "nonsense"
    data = dataclasses.replace(DATA, camera="ext1_mp4_path")
    assert ix.normalize(_row(name, value), data)["left_serial"] == "22008760"


# ---------------------------------------------------------------- §4 入选

def _episodes(*specs):
    return ix.build_episodes([_row(*meta(**kw)) for kw in specs], DATA)


def test_each_reason_alone():
    eps = _episodes(dict(stamp="2023-07-07-09h-40m-00s"),
                    dict(stamp="2023-07-07-09h-41m-00s", steps=241),
                    dict(stamp="2023-07-07-09h-42m-00s", outcome="failure", success=True),
                    dict(stamp="2023-07-07-09h-43m-00s", scene_id=-1),
                    dict(stamp="2023-07-07-09h-44m-00s", steps=242),
                    dict(stamp="2023-07-07-09h-45m-00s", scene_id=0),
                    dict(stamp="2023-07-07-09h-46m-00s", outcome="failure", success=True, steps=10))
    assert [(e["reasons"], e["selected"]) for e in eps] == [
        ([], True), (["short"], False), (["flag_mismatch"], False), (["no_scene_id"], False), ([], True), ([], True),
        (["short", "flag_mismatch"], False)]   # 原因按 02 §4 表的顺序，不按字母


def test_duplicate_uuid_and_several_reasons_at_once():
    eps = _episodes(dict(outcome="success", steps=241), dict(outcome="failure", success=True, steps=300, scene_id=-1))
    assert [e["folder"].split("/")[1] for e in eps] == ["failure", "success"]   # 同 uuid 靠 folder 排序
    assert eps[0]["reasons"] == ["duplicate_uuid", "flag_mismatch", "no_scene_id"]
    assert eps[1]["reasons"] == ["duplicate_uuid", "short"]
    assert not any(e["selected"] for e in eps)


def test_no_scene_id_only_for_in_distribution_labs():
    eps = _episodes(dict(lab="REAL", scene_id=-1), dict(lab="IPRL", scene_id=-1))
    assert {e["lab"]: e["reasons"] for e in eps} == {"REAL": [], "IPRL": ["no_scene_id"]}


def test_min_steps_comes_from_config():
    rows = [_row(*meta(steps=300))]
    assert ix.build_episodes(rows, dataclasses.replace(DATA, min_steps=301))[0]["reasons"] == ["short"]


def test_episodes_file_sorted_with_exact_fields(bucket, tmp_path):
    _fill(bucket, SEVEN + [dict(stamp="2023-07-07-09h-40m-39s", outcome="failure", steps=100)])
    manifest = ix.run_index(_config(bucket), tmp_path)
    lines = (tmp_path/"index"/"episodes.jsonl").read_text(encoding="utf-8").splitlines()
    eps = [json.loads(line) for line in lines]
    assert [(e["uuid"], e["folder"]) for e in eps] == sorted((e["uuid"], e["folder"]) for e in eps)
    assert all(list(e) == sorted(e) for e in eps)
    expected = {"lab", "outcome", "success_flag", "scene_id", "steps", "uuid", "folder", "h5", "left_mp4", "left_serial",
                "left_is", "group", "building", "date", "robot_serial", "user_id", "r2d2_version", "selected", "reasons"}
    assert all(set(e) == expected for e in eps)
    assert manifest["counts"] == {
        "episodes": 8, "selected": 6, "selected_by_group": {"id": 5, "ood": 1},
        "selected_by_lab": {"AUTOLab": 4, "PennPAL": 1, "TRI": 1},
        "reasons": {"duplicate_uuid": 2, "short": 1, "flag_mismatch": 0, "no_scene_id": 0}}
    # PennPAL（留出实验室）scene_id = −1 但入选；TRI 的 metadata 写 tri
    assert {e["lab"] for e in eps} == {"AUTOLab", "PennPAL", "TRI"}
    assert [e for e in eps if e["lab"] == "PennPAL"][0]["reasons"] == []


# ---------------------------------------------------------------- §7 清单、重跑、入口

def test_manifest_contents(bucket, tmp_path):
    _fill(bucket, SEVEN)
    cfg = _config(bucket)
    manifest = ix.run_index(cfg, tmp_path, clock=lambda: datetime(2026, 11, 9, 12, 0, 0, 123456, tzinfo=timezone.utc))
    directory = tmp_path/"index"
    on_disk = json.loads((directory/"MANIFEST.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert (directory/"MANIFEST.json").read_bytes() == ix.cache._json_bytes(manifest)
    assert manifest["schema"] == "droid_index_v1" and manifest["listed_at"] == "2026-11-09T12:00:00+00:00"
    assert sorted(manifest["files"]) == ["episodes.jsonl", "listing.jsonl", "metadata.jsonl", "splits.json"]
    for name, entry in manifest["files"].items():
        lines = len((directory/name).read_bytes().splitlines()) if name == "splits.json" else 7
        assert entry == {"sha256": hashlib.sha256((directory/name).read_bytes()).hexdigest(), "lines": lines}
    assert manifest["config_keys"] == {"list_api": cfg.data.list_api, "prefix": PREFIX, "camera": "left_mp4_path",
                                       "expected_metadata_files": 7, "min_steps": 242,
                                       "ood_labs": ["PennPAL", "REAL", "RPL"]}
    assert manifest["metadata_reused"] is False
    assert manifest["code_manifest_sha256"] == ix.cache.file_sha(ix.cache.ROOT/"MANIFEST.json")
    assert manifest["code_commit"] is None or len(manifest["code_commit"]) == 40


def test_rerun_with_manifest_does_nothing(bucket, tmp_path):
    _fill(bucket, SEVEN)
    first = ix.run_index(_config(bucket), tmp_path)

    def no_network(url):
        raise AssertionError("network used")

    assert ix.run_index(_config(bucket), tmp_path, get=no_network) == first


def test_manifest_from_other_code_is_refused(bucket, tmp_path):
    _fill(bucket, SEVEN)
    ix.run_index(_config(bucket), tmp_path)
    path = tmp_path/"index"/"MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for edit, message in ((lambda m: m["files"].pop("episodes.jsonl"), "this code writes"),
                          (lambda m: m["files"].pop("splits.json"), "this code writes"),   # S02 写的三文件清单
                          (lambda m: m["files"].update({"windows.json": {}}), "this code writes"),
                          (lambda m: m.update(schema="droid_index_v0"), "schema")):
        changed = copy.deepcopy(manifest)
        edit(changed)
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            ix.run_index(_config(bucket), tmp_path)


def test_rerun_reuses_metadata_only_when_every_md5_matches(bucket, tmp_path):
    names = _fill(bucket, SEVEN)
    first = ix.run_index(_config(bucket), tmp_path)
    directory = tmp_path/"index"
    (directory/"MANIFEST.json").unlink()
    bucket.object_gets.clear()
    second = ix.run_index(_config(bucket), tmp_path)
    assert bucket.object_gets == [] and second["metadata_reused"] is True
    assert second["files"] == first["files"]
    # 一个对象变了：列表里的 MD5 变了 → 整份重新下载
    (directory/"MANIFEST.json").unlink()
    name, value = meta(**SEVEN[2])
    bucket.put(name, dict(value, current_task="changed"))
    third = ix.run_index(_config(bucket), tmp_path)
    assert third["metadata_reused"] is False and sorted(bucket.object_gets) == sorted(names)
    assert third["files"]["metadata.jsonl"] != first["files"]["metadata.jsonl"]
    assert third["files"]["episodes.jsonl"] == first["files"]["episodes.jsonl"]   # current_task 不进 episodes


def test_read_current_metadata_mismatch_cases(tmp_path):
    listing = [{"name": "a", "md5Hash": "1"}, {"name": "b", "md5Hash": "2"}]
    path = tmp_path/"metadata.jsonl"
    assert ix.read_current_metadata(path, listing) is None
    rows = [{"k": 1, "_name": "a", "_md5": "1"}, {"k": 2, "_name": "b", "_md5": "2"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert ix.read_current_metadata(path, listing) == rows
    assert ix.read_current_metadata(path, listing[:1]) is None
    assert ix.read_current_metadata(path, [listing[0], dict(listing[1], md5Hash="3")]) is None
    assert ix.read_current_metadata(path, [listing[0], dict(listing[1], name="c")]) is None
    path.write_text('{"_name": "a", "_md5": "1"}\nnot json\n', encoding="utf-8")
    assert ix.read_current_metadata(path, listing) is None


def test_verify_sample_passes_and_picks_by_hash(bucket, tmp_path):
    names = _fill(bucket, SEVEN)
    cfg = _config(bucket)
    ix.run_index(cfg, tmp_path)
    bucket.object_gets.clear()
    report = ix.verify_sample(cfg, tmp_path, 3)
    expected = sorted(names, key=lambda n: hashlib.sha256(f"{cfg.data.split_salt}|verify|{n}".encode()).hexdigest())[:3]
    assert bucket.object_gets == expected
    assert report == {"checked": 3, "first": expected[0], "last": expected[2]}
    for bad in (0, 8):
        with pytest.raises(ValueError, match="between 1 and 7"):
            ix.verify_sample(cfg, tmp_path, bad)


def test_verify_sample_detects_changed_rows(bucket, tmp_path):
    _fill(bucket, SEVEN)
    cfg = _config(bucket)
    ix.run_index(cfg, tmp_path)
    directory = tmp_path/"index"
    path = directory/"metadata.jsonl"
    original = path.read_text(encoding="utf-8")
    # 文件被改了但清单没改 → SHA 不符
    path.write_text(original.replace('"BAIR"', '"BAIX"', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match MANIFEST.json"):
        ix.verify_sample(cfg, tmp_path, 7)
    # 文件和清单一起改（索引自洽但和存储桶不同）→ 逐字段比较报错
    for edited, message in ((original.replace('"BAIR"', '"BAIX"', 1), "fields \\['building'\\]"),
                            (_swap_first_two_keys(original), "key order")):
        path.write_text(edited, encoding="utf-8")
        manifest_path = directory/"MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["metadata.jsonl"] = ix._file_entry(path)
        manifest_path.unlink()
        ix.cache._write_json(manifest_path, manifest)
        with pytest.raises(ValueError, match="re-downloaded metadata differs.*" + message):
            ix.verify_sample(cfg, tmp_path, 7)


def _swap_first_two_keys(text: str) -> str:
    lines = text.splitlines()
    row = json.loads(lines[0])
    keys = list(row)
    keys[0], keys[1] = keys[1], keys[0]
    lines[0] = json.dumps({k: row[k] for k in keys}, ensure_ascii=False, separators=(",", ":"))
    return "".join(line + "\n" for line in lines)


def test_main_runs_index_and_verify(bucket, tmp_path, capsys):
    _fill(bucket, SEVEN)
    base = yaml.safe_load((CONFIGS/"base.yaml").read_text(encoding="utf-8"))
    host = f"http://127.0.0.1:{bucket.port}"
    base["data"].update(list_api=host + LIST_PATH, object_base=host + OBJECT_PATH, expected_metadata_files=7,
                        **dict(SMALL_SPLITS, ood_quota=list(SMALL_SPLITS["ood_quota"])))
    configs = tmp_path/"configs"
    configs.mkdir()
    (configs/"base.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    (configs/"pilot.yaml").write_bytes((CONFIGS/"pilot.yaml").read_bytes())
    with mock.patch.object(config_module, "OOD_TEST_EPISODES", 1):
        assert ix.main(["--config", str(configs/"pilot.yaml"), "--data-root", str(tmp_path/"data"),
                        "--verify-sample", "2"]) == 0
    out = capsys.readouterr().out
    assert '"selected": 7' in out and '"checked": 2' in out and '"ood_test": 1' in out and '"splits"' in out
    assert (tmp_path/"data"/"index"/"MANIFEST.json").exists()


def test_salted_hash_hand_value():
    assert ix.salted_hash("s", "scene", "IPRL/1") == hashlib.sha256(b"s|scene|IPRL/1").hexdigest()


def test_with_retries_counts_attempts():
    calls, slept = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 4:
            raise OSError("down")
        return "ok"

    assert ix.with_retries(flaky, "thing", sleep=slept.append) == "ok"
    assert (len(calls), slept) == (4, [1, 2, 4])


# ---------------------------------------------------------------- 真实元数据（开发机；没有真实索引时跳过）

REAL_ROOT = os.environ.get("PO_JEPA_DROID_DATA_ROOT")
REAL_LABS = {"AUTOLab": 4830, "CLVR": 977, "GuptaLab": 141, "ILIAD": 1905, "IPRL": 2926, "IRIS": 1351, "PennPAL": 1635,
             "RAD": 1221, "RAIL": 872, "REAL": 2004, "RPL": 1655, "TRI": 10586, "WEIRD": 2482}


@pytest.mark.skipif(not REAL_ROOT or not (Path(REAL_ROOT or ".")/"index"/"MANIFEST.json").exists(),
                    reason="set PO_JEPA_DROID_DATA_ROOT to a data root indexed from the real bucket")
def test_real_index_counts():
    directory = Path(REAL_ROOT)/"index"
    manifest = ix.read_manifest(directory)
    ix.check_files(directory, manifest)
    eps = [json.loads(line) for line in (directory/"episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = [e for e in eps if e["selected"]]
    assert len(eps) == 74896 and len(selected) == 32585
    assert sum(e["group"] == "ood" for e in selected) == 5294 and sum(e["group"] == "id" for e in selected) == 27291
    labs = {}
    for e in selected:
        labs[e["lab"]] = labs.get(e["lab"], 0) + 1
    assert labs == REAL_LABS
    assert manifest["counts"]["reasons"] == {"duplicate_uuid": 202, "short": 42207, "flag_mismatch": 15,
                                             "no_scene_id": 1368}
