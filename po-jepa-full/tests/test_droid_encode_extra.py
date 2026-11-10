"""S06b：DROID 测试附加编码 `encode/test-extra`（docs/design/05-droid-encode.md §4；验收 14 §3 S06 第四条）。

- 假编码器跑通：`id_test`、`ood_test` 前 `test_extra_episodes` 条的 s5（第 5j 帧，行数 ⌊(V − 1)/5⌋ + 1）、r192（144 token）、
  r320（400 token）；特征和独立重算（OpenCV 直接解码 + preprocess_square + 同一个假编码器）逐字节相同；清单、完成记录、
  读取器逐行核 SHA；跳过没通过的、不补位；帧 SHA 核对；§3 跳过规则；和主特征同帧逐位相同的计数。
- 真实编码器的接线（小编码器桩，S25e 测试的 `_loaded`）：s5 走 v7 预处理 + FrozenFrameEncoder，rR 走 preprocess_square(R)
  + NativeFrameEncoder，身份记对。
整片测试沿用 S05/S06a 的假存储桶 + 下载 + 准备。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from po_jepa import native_encoder as ne
from po_jepa import vjepa
from po_jepa.droid import encode as en
from po_jepa.droid import encode_extra as ex
from po_jepa.droid import features as ft
from po_jepa.droid import prepare as pr
from po_jepa.droid import video
from po_jepa.runtime import cache
from test_droid_download import raw  # noqa: F401 — fixture（本机假存储桶）
from test_droid_encode import _flip_byte, official, quiet, setup  # noqa: F401 — official 是 fixture
from test_droid_prepare import T_FIX, mp4_bytes, replace  # noqa: F401 — mp4_bytes 是 fixture
from test_s25e_space_eval import _loaded

V = T_FIX - 1                       # 夹具视频的帧数（h5 步数 − 1）
F5 = (V - 1)//5 + 1                 # 18
F = (V - 1)//15 + 1                 # 6
KINDS = ("s5", "r192", "r320")


def run(cfg, root, encoders=None):
    return ex.run_test_extra(cfg, root, encoders or ex.fake_encoders(cfg), log=quiet)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def decoded(mp4, indices):
    """不走 encode.py：OpenCV 直接解码，取 `indices` 处的帧。"""
    cap, out, i = cv2.VideoCapture(str(mp4)), {}, 0
    while len(out) < len(indices):
        ok, frame = cap.read()
        assert ok
        if i in indices:
            out[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return np.stack([out[i] for i in indices])


def independent(mp4, indices, size):
    x = en.preprocess_square(torch.from_numpy(decoded(mp4, indices)), size)
    return en.fake_encoder(size).encode(x).numpy().astype(np.float16)


def test_kinds_follow_the_config(raw, mp4_bytes, tmp_path):
    cfg, _, _ = setup(raw, tmp_path, mp4_bytes)
    assert ex.kinds(cfg) == {"s5": {"size": 256, "stride": 5}, "r192": {"size": 192}, "r320": {"size": 320}}
    assert [ex.tokens(s) for s in (192, 256, 320)] == [144, 256, 400]
    for bad in ((), (192, 192), (200,), (256,)):
        with pytest.raises(ValueError, match="extra_resolutions"):
            ex.kinds(replace(cfg, encode=dataclasses.replace(cfg.encode, extra_resolutions=bad)))


def test_outputs_manifest_and_independent_recompute(raw, mp4_bytes, tmp_path):
    cfg, episodes, plan = setup(raw, tmp_path, mp4_bytes)
    result = run(cfg, tmp_path)
    chosen = [(s, u) for s in ex.SHARDS for u in plan[s][:2]]
    assert result["statuses"] == {"encoded": 3*len(chosen), "current": 0}
    manifest_path = tmp_path/"features"/"test_extra"/f"MANIFEST.{cfg.name}.json"
    manifest = read(manifest_path)
    assert manifest_path.read_bytes() == cache._json_bytes(manifest)
    assert [(e["shard"], e["uuid"]) for e in manifest["episodes"]] == chosen and manifest["skipped"] == []
    assert manifest["counts"] == {"episodes": 4, "encoded": 4, "skipped": 0,
                                  "rows": {"s5": 4*F5, "r192": 4*F, "r320": 4*F},
                                  "by_shard": {"id_test": 2, "ood_test": 2}}
    assert manifest["prepare_complete_sha256"] == {
        s: cache.file_sha(tmp_path/"prepared"/s/"_complete"/f"{cfg.name}.json") for s in ex.SHARDS}
    assert manifest["encoder"]["s5"]["preprocess_sha256"] == en.square_preprocess_sha256(256)
    assert manifest["encoder"]["r192"]["preprocess_sha256"] == en.square_preprocess_sha256(192)
    assert manifest["encoder"]["r320"]["preprocess_sha256"] == en.square_preprocess_sha256(320)
    assert all(i["evidence_role"] == "synthetic_development_only" for i in manifest["encoder"].values())
    assert manifest["main_comparison"] == {s: {"main_manifest_sha256": None, "same_encoder": None,
                                               "rows_compared": 0, "rows_identical": 0} for s in ex.SHARDS}
    done = read(tmp_path/"features"/"test_extra"/"_complete"/f"{cfg.name}.json")
    assert done["manifest_sha256"] == cache.file_sha(manifest_path) and done["schema"] == "droid_test_extra_complete_v1"
    shapes = {"s5": (F5, 256, 1408), "r192": (F, 144, 1408), "r320": (F, 400, 1408)}
    s5_frames, main_frames = list(range(0, V, 5)), list(range(0, V, 15))
    for (shard, u), entry in zip(chosen, manifest["episodes"]):
        prepared = pr.load_prepared(tmp_path/"prepared"/shard/f"{u}.json")
        mp4 = tmp_path/"raw"/shard/u/f"{episodes[u]['left_serial']}.mp4"
        for kind in KINDS:
            npy, js = ft.extra_paths(tmp_path, u, kind)
            assert (npy.name, js.name) == (f"{u}.{kind}.npy", f"{u}.{kind}.json")
            arr, record = np.load(npy, allow_pickle=False), read(js)
            assert arr.dtype == np.float16 and arr.shape == shapes[kind] and record["shape"] == list(shapes[kind])
            assert entry["files"][kind] == {"feature_json_sha256": cache.file_sha(js), "file_sha256": cache.file_sha(npy),
                                            "rows": shapes[kind][0]}
            assert record["row_sha256"] == [hashlib.sha256(r.tobytes()).hexdigest() for r in arr]
            assert record["prepared_sha256"] == prepared.json_sha256 and record["shard"] == shard
            assert record["kind"] == kind and record["uuid"] == u
            frames = s5_frames if kind == "s5" else main_frames
            assert record["frame_index"] == frames
            assert record["frame_sha256"] == (prepared.info["frame_sha256_s5"] if kind == "s5"
                                              else prepared.info["frame_sha256"])
            size = {"s5": 256, "r192": 192, "r320": 320}[kind]
            assert arr.tobytes() == independent(mp4, frames, size).tobytes()
    # 读取器：逐行核 SHA；s5 的行号是自己的
    reader = ft.ExtraFeatureReader(tmp_path, ft.read_extra_manifest(tmp_path, cfg)[0])
    u = chosen[1][1]
    arr = np.load(ft.extra_paths(tmp_path, u, "s5")[0])
    assert torch.equal(reader.rows(u, "s5", [17, 0, 3]), torch.from_numpy(arr[[17, 0, 3]].astype(np.float32)))
    assert tuple(reader.rows(u, "r320", [5]).shape) == (1, 400, 1408) and reader.shard(u) == "id_test"
    with pytest.raises(IndexError):
        reader.rows(u, "r192", [F])
    with pytest.raises(KeyError):
        reader.rows(plan["val"][0], "s5", [0])              # 验证片不编附加特征
    assert not list((tmp_path/"features").rglob("*.tmp"))
    # 完成记录已在：什么都不做，返回它
    before = {p: p.stat().st_mtime_ns for p in (tmp_path/"features").rglob("*")}
    assert run(cfg, tmp_path) == done
    assert {p: p.stat().st_mtime_ns for p in (tmp_path/"features").rglob("*")} == before


def test_reader_detects_changed_bytes_and_files(raw, mp4_bytes, tmp_path):
    cfg, _, plan = setup(raw, tmp_path, mp4_bytes)
    run(cfg, tmp_path)
    manifest, _ = ft.read_extra_manifest(tmp_path, cfg)
    u = plan["ood_test"][0]
    npy, js = ft.extra_paths(tmp_path, u, "r192")
    _flip_byte(npy, 1000)                                   # 最后一行
    reader = ft.ExtraFeatureReader(tmp_path, manifest)
    reader.rows(u, "r192", [0, 4])
    with pytest.raises(ValueError, match=r"r192: row 5 does not match its SHA"):
        reader.rows(u, "r192", [5])
    js.write_bytes(cache._json_bytes(dict(read(js), seconds=0.0)))
    with pytest.raises(ValueError, match="does not match the test-extra manifest"):
        ft.ExtraFeatureReader(tmp_path, manifest).rows(u, "r192", [0])
    s5 = ft.extra_paths(tmp_path, u, "s5")[0]
    np.save(s5, np.load(s5)[:F5 - 1], allow_pickle=False)
    with pytest.raises(ValueError, match="the JSON says"):
        ft.ExtraFeatureReader(tmp_path, manifest).rows(u, "s5", [0])
    path = tmp_path/"features"/"test_extra"/f"MANIFEST.{cfg.name}.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed after encode/test-extra completed"):
        ft.read_extra_manifest(tmp_path, cfg)


def test_main_comparison_after_main_encoding(raw, mp4_bytes, tmp_path):
    cfg, _, plan = setup(raw, tmp_path, mp4_bytes)
    en.run_encode(cfg, tmp_path, "id_test", en.fake_encoder(), log=quiet)
    run(cfg, tmp_path)
    comparison = read(tmp_path/"features"/"test_extra"/f"MANIFEST.{cfg.name}.json")["main_comparison"]
    want = 0
    for u in plan["id_test"][:2]:
        main = np.load(ft.feature_paths(tmp_path, "id_test", u)[0])
        s5 = np.load(ft.extra_paths(tmp_path, u, "s5")[0])
        want += sum(main[j].tobytes() == s5[3*j].tobytes() for j in range(F))
    assert comparison["id_test"] == {
        "main_manifest_sha256": cache.file_sha(tmp_path/"features"/"id_test"/f"MANIFEST.{cfg.name}.json"),
        "same_encoder": True, "rows_compared": 2*F, "rows_identical": want}
    assert comparison["ood_test"]["main_manifest_sha256"] is None


def test_first_n_skip_failed_without_backfill(raw, mp4_bytes, tmp_path):
    def edit(uuid, data):
        if uuid in bad:
            data["state_position"][20, 0] = np.nan        # 两种动作源都没通过
    probe_cfg, _, probe_plan = setup(raw, tmp_path/"probe", mp4_bytes)
    bad = {probe_plan["ood_test"][0]}
    raw.objects.clear()
    root = tmp_path/"r"
    cfg, _, plan = setup(raw, root, mp4_bytes, edit=edit)
    assert plan["ood_test"][0] in bad
    run(cfg, root)
    manifest = read(root/"features"/"test_extra"/f"MANIFEST.{cfg.name}.json")
    assert manifest["skipped"] == [{"uuid": plan["ood_test"][0], "shard": "ood_test", "reasons": ["finite"]}]
    assert [e["uuid"] for e in manifest["episodes"] if e["shard"] == "ood_test"] == [plan["ood_test"][1]]
    assert manifest["counts"]["by_shard"] == {"id_test": 2, "ood_test": 1} and manifest["counts"]["episodes"] == 4
    assert not any(ft.extra_paths(root, plan["ood_test"][0], k)[0].exists() for k in KINDS)


def test_frame_sha_is_checked_on_s5_only_frames(raw, mp4_bytes, tmp_path, monkeypatch):
    cfg, _, _ = setup(raw, tmp_path, mp4_bytes)
    original = video.decode_rgb

    def tampered(path):
        for i, frame in original(path):
            if i == 25:                                      # 只是 s5 的帧，不是主特征的
                frame = frame.copy()
                frame[0, 0, 0] ^= 1
            yield i, frame

    monkeypatch.setattr(video, "decode_rgb", tampered)
    with pytest.raises(ValueError, match="frame 25 decodes differently"):
        run(cfg, tmp_path)
    assert not (tmp_path/"features"/"test_extra"/"_complete").exists()
    monkeypatch.setattr(video, "decode_rgb", lambda path: ((i, f) for i, f in original(path) if i < 5*(F5 - 1)))
    with pytest.raises(ValueError, match=rf"frames \[{5*(F5 - 1)}\] were not decoded"):
        run(cfg, tmp_path)


def test_s5_index_and_conflicting_shas():
    info = {"uuid": "u", "video_frames": 11, "frame_sha256_s5": ["a", "b", "c"], "limits": {"s5_stride_frames": 5}}
    assert ex.s5_index(info, 5).tolist() == [0, 5, 10]
    assert ex.s5_index(dict(info, video_frames=15), 5).tolist() == [0, 5, 10]
    for bad, message in ((dict(info, video_frames=16), "3 s5 frame SHAs for V = 16"),
                         (dict(info, video_frames=10), "3 s5 frame SHAs for V = 10"),
                         ({k: v for k, v in info.items() if k != "frame_sha256_s5"}, "no frame_sha256_s5"),
                         (dict(info, limits={"s5_stride_frames": 3}), "s5 stride 3")):
        with pytest.raises(ValueError, match=message):
            ex.s5_index(bad, 5)
    assert ex._wanted({"a": (np.array([0, 15]), ["x", "y"]), "b": (np.array([0, 5]), ["x", "z"])}) == \
        {0: "x", 15: "y", 5: "z"}
    with pytest.raises(ValueError, match="two different SHAs"):
        ex._wanted({"a": (np.array([0, 15]), ["x", "y"]), "b": (np.array([0, 15]), ["x", "q"])})


def test_rerun_skip_rule(raw, mp4_bytes, tmp_path):
    cfg, _, plan = setup(raw, tmp_path, mp4_bytes)
    run(cfg, tmp_path)
    chosen = [u for s in ex.SHARDS for u in plan[s][:2]]
    before = {(u, k): ft.extra_paths(tmp_path, u, k)[0].read_bytes() for u in chosen for k in KINDS}
    done = tmp_path/"features"/"test_extra"/"_complete"/f"{cfg.name}.json"
    manifest = (tmp_path/"features"/"test_extra"/f"MANIFEST.{cfg.name}.json").read_bytes()
    done.unlink()
    assert run(cfg, tmp_path)["statuses"] == {"encoded": 0, "current": 12}
    assert (tmp_path/"features"/"test_extra"/f"MANIFEST.{cfg.name}.json").read_bytes() == manifest
    # 文件被改 → 只重编这一个；JSON 没了 → 重编；kind 被改 → 重编
    done.unlink()
    _flip_byte(ft.extra_paths(tmp_path, chosen[0], "r192")[0], 10)
    ft.extra_paths(tmp_path, chosen[1], "s5")[1].unlink()
    js = ft.extra_paths(tmp_path, chosen[2], "r320")[1]
    assert run(cfg, tmp_path)["statuses"] == {"encoded": 2, "current": 10}
    assert {(u, k): ft.extra_paths(tmp_path, u, k)[0].read_bytes() for u in chosen for k in KINDS} == before
    args = (ft.extra_paths(tmp_path, chosen[2], "r320")[0], "ood_test", "r320", read(js)["prepared_sha256"],
            read(js)["encoder"])
    assert ex.current(read(js), *args) and not ex.current(dict(read(js), kind="r192"), *args)
    # 一种 kind 的编码器身份变了 → 这一种全部重编
    done.unlink()
    encoders = ex.fake_encoders(cfg)
    encoders["r320"] = dataclasses.replace(encoders["r320"], preprocess_sha256="0"*64)
    assert run(cfg, tmp_path, encoders)["statuses"] == {"encoded": 4, "current": 8}


def test_preconditions(raw, mp4_bytes, tmp_path):
    cfg, _, _ = setup(raw, tmp_path, mp4_bytes)
    with pytest.raises(ValueError, match="encoders for"):
        run(cfg, tmp_path, {k: v for k, v in ex.fake_encoders(cfg).items() if k != "r320"})
    with pytest.raises(ValueError, match="upstream_commit"):
        run(replace(cfg, encode=dataclasses.replace(cfg.encode, upstream_commit="0"*40)), tmp_path)
    (tmp_path/"prepared"/"ood_test"/"_complete"/f"{cfg.name}.json").unlink()
    with pytest.raises(ValueError, match="prepare/ood_test has not completed"):
        run(cfg, tmp_path)
    assert not (tmp_path/"features"/"test_extra"/"_complete").exists()
    with pytest.raises(ValueError, match="encode/test-extra has not completed"):
        ft.read_extra_manifest(tmp_path, cfg)


def test_real_encoder_wiring_with_a_tiny_encoder(raw, mp4_bytes, tmp_path, official):
    """s5 = v7 预处理 + FrozenFrameEncoder（和主特征同一个编码器对象），rR = preprocess_square(R) + NativeFrameEncoder。"""
    cfg, episodes, plan = setup(raw, tmp_path, mp4_bytes)
    loaded = _loaded()
    main = en.real_encoder(None, None, "cpu", loaded=loaded)
    encoders = ex.real_encoders(cfg, "cpu", loaded=loaded, main=main)
    assert encoders["s5"] is main and main.preprocess is en.preprocess_v7
    for kind, size in (("r192", 192), ("r320", 320)):
        e = encoders[kind]
        assert (e.role, e.checkpoint_sha256, e.preprocess_sha256) == (
            "real_frozen_encoder", cache.CHECKPOINT_SHA, en.square_preprocess_sha256(size))
    run(cfg, tmp_path, encoders)
    manifest = read(tmp_path/"features"/"test_extra"/f"MANIFEST.{cfg.name}.json")
    assert manifest["encoder"]["s5"]["preprocess_sha256"] == cache.v7_preprocess_sha256()
    u = plan["id_test"][0]
    mp4 = tmp_path/"raw"/"id_test"/u/f"{episodes[u]['left_serial']}.mp4"
    t = lambda n: torch.zeros(n, dtype=torch.float64)                                  # noqa: E731
    kw = dict(preprocess=cache.preprocess_spec(), encoder_device=torch.device("cpu"), autocast_dtype=None)
    s5_rgb = decoded(mp4, list(range(0, V, 5)))
    x = torch.cat([en.preprocess_v7(f) for f in s5_rgb])
    chunks = [vjepa.FrozenFrameEncoder(loaded, **kw).encode_target(x[i:i + 32], target_time_seconds=t(len(x[i:i + 32])))
              .raw_z for i in range(0, len(x), 32)]
    want = torch.cat(chunks).numpy().astype(np.float16)
    assert np.load(ft.extra_paths(tmp_path, u, "s5")[0]).tobytes() == want.tobytes()
    rgb = torch.from_numpy(decoded(mp4, list(range(0, V, 15))))
    for kind, size in (("r192", 192), ("r320", 320)):
        z = ne.NativeFrameEncoder(loaded, resolution=size, **kw).encode_target(
            en.preprocess_square(rgb, size), target_time_seconds=t(F)).raw_z.numpy().astype(np.float16)
        assert z.shape == (F, (size//16)**2, 1408)
        assert np.load(ft.extra_paths(tmp_path, u, kind)[0]).tobytes() == z.tobytes()


def test_command_line(raw, mp4_bytes, tmp_path, monkeypatch, capsys, official):
    cfg, _, plan = setup(raw, tmp_path, mp4_bytes)
    monkeypatch.setattr(en, "load_config", lambda path: cfg)
    assert en.main(["--config", "x.yaml", "--data-root", str(tmp_path), "--shard", "test-extra", "--fake-encoder"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "droid_test_extra_complete_v1" and out["statuses"] == {"encoded": 12, "current": 0}
    mp4 = next((tmp_path/"raw"/"id_test"/plan["id_test"][0]).glob("*.mp4"))
    assert en.main(["--config", "x.yaml", "--data-root", str(tmp_path), "--check-mp4", str(mp4), "--fake-encoder"]) == 0
    check = json.loads(capsys.readouterr().out)
    assert check["extra_resolutions"] == {"r192": {"shape": [1, 144, 1408], "finite": True, "ok": True},
                                          "r320": {"shape": [1, 400, 1408], "finite": True, "ok": True}}
    with pytest.raises(SystemExit):
        en.main(["--config", "x.yaml", "--data-root", str(tmp_path), "--shard", "test-extra", "--fake-encoder",
                 "--delete-raw"])
