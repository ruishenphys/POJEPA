"""S06a：DROID 编码和特征格式（docs/design/05-droid-encode.md §1–§3、§5、§6，03 §4；验收 14 §3 S06）。

- 预处理：`preprocess_square(rgb, 256)` 和 v7 `vjepa.preprocess_rgb`（裁剪用 docs/design/droid-golden 里上游
  transforms.py 原样副本中的 `random_resized_crop`、`_get_param_spatial_crop`）逐位相同；裁剪的行列等于官方中心回退。
  设 `PO_JEPA_DROID_GOLDEN_RAW` 时另在 5 条金标准回合的第 0、15 帧上和官方裁剪、归一化的 SHA 比较。
- 假编码器跑通整片：特征文件、每行 SHA、片清单、完成记录、总清单 SHA；特征和独立重算（OpenCV 直接解码 + 同一个假编码器）
  逐字节相同；改一个字节后 `FeatureReader` 报错；帧 SHA 和准备文件不同时报错；§3 的跳过规则；`--delete-raw`。
整片测试沿用 S05 的假存储桶 + 下载 + 准备（tests/test_droid_prepare.py 的 build_root）。
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from po_jepa import vjepa
from po_jepa.droid import download as dl
from po_jepa.droid import encode as en
from po_jepa.droid import features as ft
from po_jepa.droid import prepare as pr
from po_jepa.droid import splits as sp
from po_jepa.droid import video
from po_jepa.runtime import cache
from test_droid_download import raw  # noqa: F401 — fixture（本机假存储桶）
from test_droid_prepare import build_root, mp4_bytes, prepare_all, replace, small  # noqa: F401 — mp4_bytes 是 fixture

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT/"docs"/"design"/"droid-golden"
GOLDEN_RAW = os.environ.get("PO_JEPA_DROID_GOLDEN_RAW")


# ---------------------------------------------------------------- 官方裁剪函数（取自原样副本）

def _official_namespace() -> types.SimpleNamespace:
    """从上游 transforms.py 的原样副本里只取 `_get_param_spatial_crop` 和 `random_resized_crop` 两个函数（整个文件要
    PIL、torchvision 和上游 src 包），在只有 math、random、numpy、torch 的命名空间里执行。"""
    source = (GOLDEN/"official_transforms_204698b.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("_get_param_spatial_crop", "random_resized_crop")]
    assert len(keep) == 2
    namespace = {"math": math, "random": random, "np": np, "torch": torch}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "official_transforms_204698b.py", "exec"), namespace)
    return types.SimpleNamespace(random_resized_crop=namespace["random_resized_crop"],
                                 _get_param_spatial_crop=namespace["_get_param_spatial_crop"])


@pytest.fixture
def official(monkeypatch):
    ns = _official_namespace()
    monkeypatch.setattr(vjepa, "_bound_video_transforms", lambda: ns)
    return ns


def frames(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, *video.FRAME_SHAPE), dtype=np.uint8)


def test_crop_rows_and_columns_are_the_official_center_fallback(official):
    p = cache.preprocess_spec()
    for seed in range(5):
        random.seed(seed)
        np.random.seed(seed)
        i, j, h, w = official._get_param_spatial_crop(p.crop_scale, p.crop_ratio, 720, 1280)
        assert (i, i + h, j, j + w) == (*en.CROP_ROWS, *en.CROP_COLS) == (0, 720, 154, 1126)


@pytest.mark.parametrize("seed", range(3))
def test_preprocess_square_256_is_bit_identical_to_v7(official, seed):
    rgb = frames(3, seed)
    ours = en.preprocess_square(torch.from_numpy(rgb), 256)
    v7 = torch.cat([en.preprocess_v7(f) for f in rgb])
    assert ours.dtype == v7.dtype == torch.float32 and tuple(ours.shape) == (3, 3, 256, 256)
    assert ours.numpy().tobytes() == v7.numpy().tobytes()
    # 极值像素（全 0、全 255）也一样
    edge = np.zeros((2, *video.FRAME_SHAPE), dtype=np.uint8)
    edge[1] = 255
    assert torch.equal(en.preprocess_square(torch.from_numpy(edge), 256),
                       torch.cat([en.preprocess_v7(f) for f in edge]))


def test_preprocess_square_independent_formula():
    rgb = frames(2, 7)
    x = torch.from_numpy(rgb).to(torch.float32).permute(0, 3, 1, 2)[:, :, 0:720, 154:1126]
    resized = torch.nn.functional.interpolate(x, size=(192, 192), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406])*255
    std = torch.tensor([0.229, 0.224, 0.225])*255
    want = (resized - mean[None, :, None, None])/std[None, :, None, None]
    got = en.preprocess_square(torch.from_numpy(rgb), 192)
    assert tuple(got.shape) == (2, 3, 192, 192)
    assert torch.allclose(got, want, atol=1e-5, rtol=0)
    assert tuple(en.preprocess_square(torch.from_numpy(rgb[:1]), 320).shape) == (1, 3, 320, 320)


@pytest.mark.parametrize("bad", ["shape", "dtype", "empty", "size"])
def test_preprocess_square_rejects_bad_input(bad):
    rgb = torch.from_numpy(frames(1))
    kwargs = {"size": 256}
    if bad == "shape":
        rgb = rgb[:, :, :1000]
    elif bad == "dtype":
        rgb = rgb.float()
    elif bad == "empty":
        rgb = rgb[:0]
    else:
        kwargs["size"] = 250
    with pytest.raises(ValueError):
        en.preprocess_square(rgb, **kwargs)


def _golden():
    return json.loads((GOLDEN/"droid_samples_golden.json").read_text(encoding="utf-8"))["episodes"]


@pytest.mark.skipif(not GOLDEN_RAW or not Path(GOLDEN_RAW or ".").is_dir(),
                    reason="set PO_JEPA_DROID_GOLDEN_RAW to a directory with <uuid>/<serial>.mp4 for the golden episodes")
@pytest.mark.parametrize("k", range(5))
def test_golden_frames_match_official_crop_and_normalization(official, k):
    ref = _golden()[k]
    rgb = {i: f for i, f in video.decode_rgb(Path(GOLDEN_RAW)/ref["uuid"]/ref["left_mp4"]) if i in (0, 15)}
    assert len(rgb) == 2
    ours = en.preprocess_square(torch.from_numpy(np.stack([rgb[0], rgb[15]])), 256)      # [2, 3, 256, 256]
    norm = ours.permute(1, 0, 2, 3).contiguous()                                          # 官方 [C, T, H, W]
    assert hashlib.sha256(norm.numpy().tobytes()).hexdigest() == ref["normalized_sha256"]
    assert torch.equal(ours, torch.cat([en.preprocess_v7(rgb[0]), en.preprocess_v7(rgb[15])]))


# ---------------------------------------------------------------- 编码器、块、float16

def test_fake_encoder_token_counts_and_determinism():
    e = en.fake_encoder()
    for size, tokens in ((256, 256), (192, 144), (320, 400)):
        x = en.preprocess_square(torch.from_numpy(frames(2)), size)
        z = e.encode(x)
        assert tuple(z.shape) == (2, tokens, 1408) and z.dtype == torch.float32
        assert torch.equal(z, en.fake_encoder().encode(x))
    assert e.role == "synthetic_development_only" and e.checkpoint_sha256 is None


def test_encode_frames_chunks_in_row_order():
    calls = []
    base = en.fake_encoder()

    def spy(x):
        calls.append(x.shape[0])
        return base.encode(x)

    e = dataclasses.replace(base, encode=spy)
    x = en.preprocess_square(torch.from_numpy(frames(5)), 256)
    out = en.encode_frames(e, x, 2)
    assert calls == [2, 2, 1]
    assert np.array_equal(out, base.encode(x).numpy())
    with pytest.raises(ValueError, match="expected CPU float32"):
        en.encode_frames(dataclasses.replace(base, encode=lambda x: base.encode(x)[:, :255]), x, 2)
    with pytest.raises(ValueError, match="non-finite"):
        en.encode_frames(dataclasses.replace(base, encode=lambda x: base.encode(x)*float("nan")), x, 2)


def test_to_storage_checks_and_rounding():
    raw = np.random.default_rng(0).normal(size=(3, 4, 1408)).astype(np.float32)
    stored, stats = ft.to_storage(raw)
    assert stored.dtype == np.float16 and stats["max_abs"] == float(np.abs(raw).max())
    want = float(np.abs(stored.astype(np.float64) - raw.astype(np.float64)).max()/np.abs(raw).max())
    assert stats["relative_rounding_error"] == want and 0 < want < 1e-3
    for value, message in ((65504.0, "does not fit float16"), (np.inf, "non-finite"), (np.nan, "non-finite")):
        bad = raw.copy()
        bad[1, 2, 3] = value
        with pytest.raises(ValueError, match=message):
            ft.to_storage(bad)
    ok = raw.copy()
    ok[0, 0, 0] = 65503.0
    ft.to_storage(ok)
    with pytest.raises(ValueError, match="float32"):
        ft.to_storage(raw.astype(np.float64))


def test_total_manifest_sha_is_sorted_concatenation():
    shas = {"val": "c"*64, "train-000": "a"*64, "id_test": "b"*64}
    want = hashlib.sha256(("b"*64 + "a"*64 + "c"*64).encode()).hexdigest()
    assert ft.total_manifest_sha256(shas) == want
    with pytest.raises(ValueError):
        ft.total_manifest_sha256({})


def test_architecture_is_the_v7_one():
    tree = ast.parse((ROOT/"tools"/"encode_v7.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "EncoderArchitectureSpec"]
    assert len(calls) == 1
    args = [ast.literal_eval(a) if not isinstance(a, ast.Attribute) else vjepa.UPSTREAM_COMMIT for a in calls[0].args]
    assert tuple(args) == en.ARCHITECTURE_ARGS
    assert calls[0].args[0].attr == "UPSTREAM_COMMIT"


def test_current_requires_every_identity_field(tmp_path):
    stored = np.zeros((2, 3, 1408), dtype=np.float16)
    npy = tmp_path/"u.npy"
    np.save(npy, stored, allow_pickle=False)
    ident = {"evidence_role": "real_frozen_encoder", "encoder_checkpoint_sha256": "e"*64, "preprocess_sha256": "p"*64,
             "batch_frames": 32, "upstream_commit": vjepa.UPSTREAM_COMMIT}
    record = {"shard": "val", "prepared_sha256": "s"*64, "encoder": dict(ident), "file_sha256": cache.file_sha(npy)}
    assert en.current(record, npy, "val", "s"*64, ident)
    assert not en.current(record, npy, "id_test", "s"*64, ident)
    assert not en.current(record, npy, "val", "t"*64, ident)
    for key, value in (("evidence_role", "synthetic_development_only"), ("encoder_checkpoint_sha256", None),
                       ("preprocess_sha256", "q"*64), ("batch_frames", 16), ("upstream_commit", "0"*40)):
        assert not en.current(record, npy, "val", "s"*64, dict(ident, **{key: value})), key
    assert not en.current(dict(record, file_sha256="0"*64), npy, "val", "s"*64, ident)
    assert not en.current(record, tmp_path/"missing.npy", "val", "s"*64, ident)


# ---------------------------------------------------------------- 整片任务（§2、§3）

def setup(raw, tmp_path, mp4_bytes, edit=None):
    cfg, episodes = build_root(raw, tmp_path, mp4_bytes, edit=edit)
    cfg = small(cfg)
    prepare_all(cfg, tmp_path)
    plan = sp.shard_plan(sp.load_splits(tmp_path, cfg), cfg.data)
    return cfg, episodes, plan


def quiet(_):
    pass


def encode_all(cfg, root, plan, encoder=None):
    encoder = encoder or en.fake_encoder()
    return {shard: en.run_encode(cfg, root, shard, encoder, log=quiet) for shard in plan}


def independent_features(mp4: Path, rows: int) -> np.ndarray:
    """不走 encode.py：OpenCV 直接解码第 0、15、… 帧，preprocess_square 一次全部，假编码器一次编完，转 float16。"""
    cap = cv2.VideoCapture(str(mp4))
    keep = []
    i = 0
    while len(keep) < rows:
        ok, frame = cap.read()
        assert ok
        if i % 15 == 0:
            keep.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    x = en.preprocess_square(torch.from_numpy(np.stack(keep)), 256)
    return en.fake_encoder().encode(x).numpy().astype(np.float16)


def test_shard_outputs_manifest_and_total_sha(raw, mp4_bytes, tmp_path):
    cfg, episodes, plan = setup(raw, tmp_path, mp4_bytes)
    results = encode_all(cfg, tmp_path, plan)
    shas = {}
    for shard, uuids in plan.items():
        manifest_path = tmp_path/"features"/shard/f"MANIFEST.{cfg.name}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest_path.read_bytes() == cache._json_bytes(manifest)
        assert [e["uuid"] for e in manifest["episodes"]] == uuids and manifest["skipped"] == []
        assert manifest["counts"] == {"episodes": len(uuids), "encoded": len(uuids), "skipped": 0, "rows": 6*len(uuids)}
        assert manifest["prepare_complete_sha256"] == cache.file_sha(tmp_path/"prepared"/shard/"_complete"/f"{cfg.name}.json")
        assert manifest["encoder"] == {"evidence_role": "synthetic_development_only", "encoder_checkpoint_sha256": None,
                                       "preprocess_sha256": en.square_preprocess_sha256(256), "batch_frames": 32,
                                       "upstream_commit": vjepa.UPSTREAM_COMMIT}
        done = json.loads((tmp_path/"features"/shard/"_complete"/f"{cfg.name}.json").read_text(encoding="utf-8"))
        assert done["manifest_sha256"] == cache.file_sha(manifest_path) and done["config_sha256"] == cfg.config_sha256
        assert results[shard]["statuses"] == {"encoded": len(uuids), "current": 0}
        shas[shard] = done["manifest_sha256"]
        for entry, u in zip(manifest["episodes"], uuids):
            npy, js = ft.feature_paths(tmp_path, shard, u)
            record = json.loads(js.read_text(encoding="utf-8"))
            prepared = pr.load_prepared(tmp_path/"prepared"/shard/f"{u}.json")
            arr = np.load(npy, allow_pickle=False)
            assert arr.dtype == np.float16 and arr.shape == (6, 256, 1408)
            assert entry == {"uuid": u, "feature_json_sha256": cache.file_sha(js), "file_sha256": cache.file_sha(npy),
                             "rows": 6}
            assert record["file_sha256"] == cache.file_sha(npy) and record["shape"] == [6, 256, 1408]
            assert record["row_sha256"] == [hashlib.sha256(r.tobytes()).hexdigest() for r in arr]
            assert record["frame_sha256"] == prepared.info["frame_sha256"]
            assert record["frame_index"] == [0, 15, 30, 45, 60, 75]
            assert record["prepared_sha256"] == prepared.json_sha256
            assert 0 < record["feature_stats"]["relative_rounding_error"] < 1e-3
            mp4 = tmp_path/"raw"/shard/u/f"{episodes[u]['left_serial']}.mp4"
            assert arr.tobytes() == independent_features(mp4, 6).tobytes()
    manifests, total = ft.feature_manifests(tmp_path, cfg)
    assert total == hashlib.sha256("".join(shas[s] for s in sorted(shas)).encode()).hexdigest()
    reader = ft.FeatureReader(tmp_path, manifests)
    u = plan["val"][0]
    rows = reader.rows(u, [5, 0, 3])
    arr = np.load(ft.feature_paths(tmp_path, "val", u)[0])
    assert rows.dtype == torch.float32 and torch.equal(rows, torch.from_numpy(arr[[5, 0, 3]].astype(np.float32)))
    with pytest.raises(IndexError):
        reader.rows(u, [6])
    assert not list((tmp_path/"features").rglob("*.tmp")) and not list((tmp_path/"features").rglob("*.partial"))
    # 完成记录已在：什么都不做，返回它
    before = {p: p.stat().st_mtime_ns for p in (tmp_path/"features").rglob("*")}
    assert en.run_encode(cfg, tmp_path, "ood_test", en.fake_encoder(), log=quiet) == done
    assert {p: p.stat().st_mtime_ns for p in (tmp_path/"features").rglob("*")} == before


def _flip_byte(path: Path, offset_from_end: int) -> None:
    data = bytearray(path.read_bytes())
    data[-offset_from_end] ^= 0x01
    path.write_bytes(bytes(data))


def test_reader_detects_a_changed_byte_and_changed_json(raw, mp4_bytes, tmp_path):
    cfg, _, plan = setup(raw, tmp_path, mp4_bytes)
    encode_all(cfg, tmp_path, plan)
    manifests, _ = ft.feature_manifests(tmp_path, cfg)
    u = plan["id_test"][1]
    npy, js = ft.feature_paths(tmp_path, "id_test", u)
    _flip_byte(npy, 1000)                                 # 最后一行里的一个字节
    reader = ft.FeatureReader(tmp_path, manifests)
    reader.rows(u, [0, 4])                               # 别的行照常
    with pytest.raises(ValueError, match="row 5 does not match its SHA"):
        reader.rows(u, [5])
    # 特征 JSON 改了 → 清单对不上
    record = json.loads(js.read_text(encoding="utf-8"))
    js.write_bytes(cache._json_bytes(dict(record, seconds=0.0)))
    with pytest.raises(ValueError, match="does not match the manifest"):
        ft.FeatureReader(tmp_path, manifests).rows(u, [0])
    # 片清单改了 → 完成记录对不上
    path = tmp_path/"features"/"val"/f"MANIFEST.{cfg.name}.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed after encode/val completed"):
        ft.feature_manifests(tmp_path, cfg)


def test_reader_checks_array_shape(raw, mp4_bytes, tmp_path):
    cfg, _, plan = setup(raw, tmp_path, mp4_bytes)
    en.run_encode(cfg, tmp_path, "val", en.fake_encoder(), log=quiet)
    manifests, _ = ft.feature_manifests(tmp_path, cfg, shards=["val"])
    u = plan["val"][0]
    npy, _ = ft.feature_paths(tmp_path, "val", u)
    np.save(npy, np.load(npy)[:5], allow_pickle=False)
    with pytest.raises(ValueError, match="the JSON says"):
        ft.FeatureReader(tmp_path, manifests).rows(u, [0])


def test_frame_sha_mismatch_is_an_error(raw, mp4_bytes, tmp_path, monkeypatch):
    cfg, _, plan = setup(raw, tmp_path, mp4_bytes)
    original = video.decode_rgb

    def tampered(path):
        for i, frame in original(path):
            if i == 30:
                frame = frame.copy()
                frame[0, 0, 0] ^= 1
            yield i, frame

    monkeypatch.setattr(video, "decode_rgb", tampered)
    with pytest.raises(ValueError, match="frame 30 decodes differently"):
        en.run_encode(cfg, tmp_path, "val", en.fake_encoder(), log=quiet)
    assert not (tmp_path/"features"/"val"/"_complete").exists()


def test_video_shorter_than_prepared_is_an_error(raw, mp4_bytes, tmp_path, monkeypatch):
    cfg, _, plan = setup(raw, tmp_path, mp4_bytes)
    original = video.decode_rgb
    monkeypatch.setattr(video, "decode_rgb", lambda path: ((i, f) for i, f in original(path) if i < 70))
    with pytest.raises(ValueError, match=r"frames \[75\] were not decoded"):
        en.run_encode(cfg, tmp_path, "val", en.fake_encoder(), log=quiet)


def test_rerun_skip_rule(raw, mp4_bytes, tmp_path):
    cfg, _, plan = setup(raw, tmp_path, mp4_bytes)
    shard = "train-000"
    uuids = plan[shard]
    en.run_encode(cfg, tmp_path, shard, en.fake_encoder(), log=quiet)
    before = {u: ft.feature_paths(tmp_path, shard, u)[0].read_bytes() for u in uuids}
    done = tmp_path/"features"/shard/"_complete"/f"{cfg.name}.json"
    manifest = (tmp_path/"features"/shard/f"MANIFEST.{cfg.name}.json").read_bytes()
    done.unlink()
    assert en.run_encode(cfg, tmp_path, shard, en.fake_encoder(), log=quiet)["statuses"] == {"encoded": 0, "current": 3}
    assert (tmp_path/"features"/shard/f"MANIFEST.{cfg.name}.json").read_bytes() == manifest
    # 文件被改（SHA 重算对不上）→ 重编这一条；JSON 没了 → 重编；另一种编码器身份 → 全部重编
    done.unlink()
    _flip_byte(ft.feature_paths(tmp_path, shard, uuids[0])[0], 10)
    ft.feature_paths(tmp_path, shard, uuids[1])[1].unlink()
    assert en.run_encode(cfg, tmp_path, shard, en.fake_encoder(), log=quiet)["statuses"] == {"encoded": 2, "current": 1}
    assert {u: ft.feature_paths(tmp_path, shard, u)[0].read_bytes() for u in uuids} == before
    done.unlink()
    other = dataclasses.replace(en.fake_encoder(), preprocess_sha256="0"*64)
    assert en.run_encode(cfg, tmp_path, shard, other, log=quiet)["statuses"] == {"encoded": 3, "current": 0}


def test_failed_prepared_episode_is_skipped_with_reasons(raw, mp4_bytes, tmp_path):
    def edit(uuid, data):
        if uuid.startswith("PennPAL+acda9df0"):
            data["state_position"][20, 0] = np.nan       # finite 失败：两种动作源都没通过
        if uuid.startswith("PennPAL+acda9df1"):
            data["command_gripper"][15] = 2.5            # 只有 commanded 越界：照样编
    cfg, _, plan = setup(raw, tmp_path, mp4_bytes, edit=edit)
    en.run_encode(cfg, tmp_path, "ood_test", en.fake_encoder(), log=quiet)
    manifest = json.loads((tmp_path/"features"/"ood_test"/f"MANIFEST.{cfg.name}.json").read_text(encoding="utf-8"))
    bad = next(u for u in plan["ood_test"] if u.startswith("PennPAL+acda9df0"))
    assert manifest["skipped"] == [{"uuid": bad, "reasons": ["finite"]}]
    assert [e["uuid"] for e in manifest["episodes"]] == [u for u in plan["ood_test"] if u != bad]
    assert not ft.feature_paths(tmp_path, "ood_test", bad)[0].exists()
    assert "PennPAL+acda9df1" in manifest["episodes"][0]["uuid"]


def test_preconditions(raw, mp4_bytes, tmp_path):
    cfg, _, plan = setup(raw, tmp_path, mp4_bytes)
    with pytest.raises(ValueError, match="is not used by config"):
        en.run_encode(cfg, tmp_path, "train-002", en.fake_encoder(), log=quiet)
    u = plan["val"][0]
    p = tmp_path/"prepared"/"val"/f"{u}.json"
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed after prepare/val completed"):
        en.run_encode(cfg, tmp_path, "val", en.fake_encoder(), log=quiet)
    (tmp_path/"prepared"/"id_test"/"_complete"/f"{cfg.name}.json").unlink()
    with pytest.raises(ValueError, match="prepare/id_test has not completed"):
        en.run_encode(cfg, tmp_path, "id_test", en.fake_encoder(), log=quiet)
    with pytest.raises(ValueError, match="upstream_commit"):
        en.run_encode(replace(cfg, encode=dataclasses.replace(cfg.encode, upstream_commit="0"*40)), tmp_path,
                      "ood_test", en.fake_encoder(), log=quiet)
    with pytest.raises(ValueError, match="encode/train-000 has not completed"):
        ft.feature_manifests(tmp_path, cfg)


# ---------------------------------------------------------------- --delete-raw（03 §4）

def _raw_files(root, shard, u, episodes):
    d = root/"raw"/shard/u
    return d/"trajectory.h5", d/f"{episodes[u]['left_serial']}.mp4"


def test_delete_raw_only_for_training_shards(raw, mp4_bytes, tmp_path):
    cfg, episodes, plan = setup(raw, tmp_path, mp4_bytes)
    for shard in sp.TEST_SHARDS:
        with pytest.raises(ValueError, match="only for training shards"):
            en.run_encode(cfg, tmp_path, shard, en.fake_encoder(), delete_raw=True, log=quiet)
        with pytest.raises(ValueError, match="only for training shards"):
            en.delete_raw_files(cfg, tmp_path, shard)
        assert all(f.is_file() for u in plan[shard] for f in _raw_files(tmp_path, shard, u, episodes))
    assert not (tmp_path/"features"/"val").exists()


def test_delete_raw_after_verified_manifest(raw, mp4_bytes, tmp_path):
    def edit(uuid, data):
        if uuid == keep[0]:
            data["state_gripper"][3] = 1.5                # finite 失败：不编、不删
    keep = []
    cfg0, _ = build_root(raw, tmp_path/"probe", mp4_bytes)
    keep.append(sp.shard_plan(sp.load_splits(tmp_path/"probe", cfg0), cfg0.data)["train-000"][1])
    raw.objects.clear()
    root = tmp_path/"r"
    cfg, episodes = build_root(raw, root, mp4_bytes, edit=edit)
    cfg = small(cfg)
    prepare_all(cfg, root)
    shard = "train-000"
    uuids = sp.shard_plan(sp.load_splits(root, cfg), cfg.data)[shard]
    assert keep[0] in uuids
    result = en.run_encode(cfg, root, shard, en.fake_encoder(), delete_raw=True, log=quiet)
    assert result["raw_deleted"] == {"deleted": 2, "already_deleted": 0, "kept": [keep[0]]}
    manifest = json.loads((root/"features"/shard/f"MANIFEST.{cfg.name}.json").read_text(encoding="utf-8"))
    for entry in manifest["episodes"]:
        u = entry["uuid"]
        d = root/"raw"/shard/u
        assert not any(f.exists() for f in _raw_files(root, shard, u, episodes))
        assert (d/"DOWNLOADED.json").is_file() and (d/"metadata.json").is_file()
        marker = json.loads((d/"RAW_DELETED.json").read_text(encoding="utf-8"))
        assert marker["feature_json_sha256"] == entry["feature_json_sha256"] == cache.file_sha(
            ft.feature_paths(root, shard, u)[1])
        assert marker["files"] == sorted(["trajectory.h5", f"{episodes[u]['left_serial']}.mp4"])
        assert marker["schema"] == "droid_raw_deleted_v1" and marker["uuid"] == u
    assert all(f.is_file() for f in _raw_files(root, shard, keep[0], episodes))
    # 重跑：完成记录在，删除那一步幂等；下载和准备都跳过这些回合
    again = en.run_encode(cfg, root, shard, en.fake_encoder(), delete_raw=True, log=quiet)
    assert again["raw_deleted"] == {"deleted": 0, "already_deleted": 2, "kept": [keep[0]]}
    dl.run_download(cfg, root, shard, sleep=lambda s: None)
    assert not any(f.exists() for e in manifest["episodes"] for f in _raw_files(root, shard, e["uuid"], episodes))
    # 特征过期而原始文件已删 → 报错
    (root/"features"/shard/"_complete"/f"{cfg.name}.json").unlink()
    ft.feature_paths(root, shard, manifest["episodes"][0]["uuid"])[1].unlink()
    with pytest.raises(ValueError, match="raw files were deleted but its features are missing"):
        en.run_encode(cfg, root, shard, en.fake_encoder(), log=quiet)


def test_delete_raw_refuses_when_the_reread_fails(raw, mp4_bytes, tmp_path):
    cfg, episodes, plan = setup(raw, tmp_path, mp4_bytes)
    shard = "train-000"
    en.run_encode(cfg, tmp_path, shard, en.fake_encoder(), log=quiet)
    u = plan[shard][2]
    npy = ft.feature_paths(tmp_path, shard, u)[0]
    _flip_byte(npy, 5)
    with pytest.raises(ValueError, match="does not match its JSON"):
        en.run_encode(cfg, tmp_path, shard, en.fake_encoder(), delete_raw=True, log=quiet)
    assert all(f.is_file() for v in plan[shard] for f in _raw_files(tmp_path, shard, v, episodes))
    assert not list((tmp_path/"raw"/shard).rglob("RAW_DELETED.json"))
    # 行 SHA 也核：文件 SHA 和 JSON 一起改成自洽的错值，行 SHA 仍抓到
    js = ft.feature_paths(tmp_path, shard, u)[1]
    record = json.loads(js.read_text(encoding="utf-8"))
    js.write_bytes(cache._json_bytes(dict(record, file_sha256=cache.file_sha(npy))))
    manifest_path = tmp_path/"features"/shard/f"MANIFEST.{cfg.name}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for e in manifest["episodes"]:
        if e["uuid"] == u:
            e.update(feature_json_sha256=cache.file_sha(js), file_sha256=cache.file_sha(npy))
    manifest_path.write_bytes(cache._json_bytes(manifest))
    done = tmp_path/"features"/shard/"_complete"/f"{cfg.name}.json"
    d = json.loads(done.read_text(encoding="utf-8"))
    done.write_bytes(cache._json_bytes(dict(d, manifest_sha256=cache.file_sha(manifest_path))))
    with pytest.raises(ValueError, match="rows or row SHAs do not match"):
        en.delete_raw_files(cfg, tmp_path, shard)
    assert not list((tmp_path/"raw"/shard).rglob("RAW_DELETED.json"))


def test_command_line_with_fake_encoder(raw, mp4_bytes, tmp_path, monkeypatch, capsys):
    cfg, _, plan = setup(raw, tmp_path, mp4_bytes)
    monkeypatch.setattr(en, "load_config", lambda path: cfg)
    assert en.main(["--config", "x.yaml", "--data-root", str(tmp_path), "--shard", "val", "--fake-encoder"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["shard"] == "val" and out["statuses"] == {"encoded": 2, "current": 0}


def test_interrupted_raw_cleanup_is_finished_by_rerunning_the_job(raw, mp4_bytes, tmp_path, monkeypatch, capsys):
    """S14-fix1（Codex S14-P2-2）：特征完成记录写了、删原始文件删到一半中断，统一入口和运行器都不能当它已完成。"""
    sys.path.insert(0, str(ROOT/"tools"))
    import droid  # noqa: E402
    import droid_jobs  # noqa: E402
    cfg, episodes, plan = setup(raw, tmp_path, mp4_bytes)
    monkeypatch.setattr(droid, "load_config", lambda path: cfg)
    monkeypatch.setattr(en, "load_config", lambda path: cfg)
    shard = "train-000"
    runs = tmp_path/"runs"
    argv = ["encode", "--config", "x.yaml", "--data-root", str(tmp_path), "--run-root", str(runs),
            "--job", f"encode/{shard}", "--delete-raw", "--fake-encoder"]
    job = droid_jobs.find_job(droid_jobs.build_graph(cfg, "x.yaml", str(tmp_path), str(runs)), f"encode/{shard}")
    flag = ft.raw_cleanup_path(tmp_path, shard, cfg)
    assert Path(job["complete"]) == flag and flag.parent.name == "_raw_deleted"
    val = droid_jobs.find_job(droid_jobs.build_graph(cfg, "x.yaml", str(tmp_path), str(runs)), "encode/val")
    assert Path(val["complete"]) == ft.complete_path(tmp_path, "val", cfg)      # 不删原始文件的片不变

    real_unlink, calls = Path.unlink, []

    def crash_on_second_raw_file(self, missing_ok=False):
        if self.parent.parent.name == shard and self.name != dl.RAW_DELETED_FILE and self.suffix in (".h5", ".mp4"):
            calls.append(self)
            if len(calls) == 2:
                raise OSError("simulated crash while deleting raw files")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", crash_on_second_raw_file)
    with pytest.raises(OSError, match="simulated crash"):
        droid.main(argv)
    monkeypatch.setattr(Path, "unlink", real_unlink)
    assert ft.complete_path(tmp_path, shard, cfg).is_file()                 # 特征编完的证据在
    assert not flag.exists()                                                  # 任务没完成
    left = [f for u in plan[shard] for f in _raw_files(tmp_path, shard, u, episodes) if f.exists()]
    assert left
    capsys.readouterr()
    assert droid.main(argv) == 0                                              # 同一条命令重跑：进入编码阶段、删完
    assert "already complete" not in capsys.readouterr().out
    result = json.loads(flag.read_text(encoding="utf-8"))
    manifest = ft.verify_shard(tmp_path, shard, cfg)
    assert result["schema"] == "droid_raw_cleanup_complete_v1" and result["shard"] == shard
    assert result["encode_complete_sha256"] == cache.file_sha(ft.complete_path(tmp_path, shard, cfg))
    assert result["episodes_raw_deleted"] == len(manifest["episodes"])
    for entry in manifest["episodes"]:
        assert not any(f.exists() for f in _raw_files(tmp_path, shard, entry["uuid"], episodes))
    assert droid.main(argv) == 0 and "already complete" in capsys.readouterr().out
