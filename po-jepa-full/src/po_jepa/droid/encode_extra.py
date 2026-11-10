"""DROID 测试附加编码 `encode/test-extra`（docs/design/05-droid-encode.md §4；S06b）。

`id_test`、`ood_test` 两片各自按划分列表顺序的前 `test_extra_episodes`（200）条（跳过准备没通过的，不补位）另编三份：

| kind | 文件 | 内容 |
|---|---|---|
| `s5` | `features/test_extra/<uuid>.s5.npy` | float16 [F5, 256, 1408]，第 j 行是第 5j 帧，F5 = ⌊(V − 1)/5⌋ + 1（原始帧率评测，10 §8.4） |
| `r192` | `features/test_extra/<uuid>.r192.npy` | float16 [F, 144, 1408]，取帧位置同主特征（原始分辨率 12×12，10 §8.5） |
| `r320` | `features/test_extra/<uuid>.r320.npy` | float16 [F, 400, 1408]（20×20） |

（5 是 `encode.frame_rate_stride_frames`，192、320 是 `encode.extra_resolutions`。）

- 复用 S06a：`encode.encode_frames`（每块 `batch_frames` 帧、按行序切）、`features.to_storage/row_sha256/write_features`、
  `encode._prepared_records`（准备完成、片计划、准备 JSON 的 SHA）、`encode.encodable`（任一动作源通过就编）、
  `encode.current`（§3 的跳过规则，另核 kind）。
- s5 的编码器和预处理就是主特征的（真实：v7 `preprocess_rgb` + `FrozenFrameEncoder`；假编码器：`preprocess_square(·, 256)`）。
  r192、r320 的预处理是 `encode.preprocess_square(·, R)`（05 §1），真实编码器是同一个已载入编码器按原生边长编码
  （S25e 的 `native_encoder.NativeFrameEncoder`，单帧复制两份，token 行优先）。
- MP4 每条只解码一次：取主特征位置（第 15i 帧）和 s5 位置（第 5j 帧）的并集，逐帧核 RGB 的 SHA 等于准备文件的
  `frame_sha256` / `frame_sha256_s5`（两者在同一帧上必须一致）。
- 清单 `features/test_extra/MANIFEST.<配置名>.json`、完成记录 `features/test_extra/_complete/<配置名>.json`（12 的完成标志）。
  清单另记 s5 第 3j 行和主特征第 j 行（同一帧）逐位相同的行数（05 §4：只作记录）；只有该片主特征在本配置下已完成
  才比，否则记 `main_manifest_sha256: null`。

命令行：`python -m po_jepa.droid.encode --config … --data-root … --shard test-extra [--device cuda] [--fake-encoder] …`。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from po_jepa import native_encoder as ne
from po_jepa import vjepa
from po_jepa.droid import download as dl
from po_jepa.droid import encode as en
from po_jepa.droid import features as ft
from po_jepa.droid import index as ix
from po_jepa.droid import prepare as pr
from po_jepa.droid import video
from po_jepa.droid.config import DroidConfig
from po_jepa.runtime import cache

JOB = "test-extra"                       # 任务 `encode/test-extra`（12）
SHARDS = ("id_test", "ood_test")         # §4：只有两个测试集（验证片不编）
SOURCE_FILES = (*en.SOURCE_FILES, "src/po_jepa/droid/encode_extra.py", "src/po_jepa/native_encoder.py")


def kinds(config: DroidConfig) -> dict[str, dict]:
    """本配置的附加特征种类：{"s5": {"size": 256, "stride": 5}, "r192": {"size": 192}, "r320": {"size": 320}}。"""
    stride, sizes = config.encode.frame_rate_stride_frames, tuple(config.encode.extra_resolutions)
    if type(stride) is not int or stride <= 0:
        raise ValueError("encode.frame_rate_stride_frames must be a positive integer")
    if not sizes or len(set(sizes)) != len(sizes) or any(r not in ne.RESOLUTIONS or r == en.MAIN_SIZE for r in sizes):
        raise ValueError(f"encode.extra_resolutions must be distinct values from {ne.RESOLUTIONS} other than "
                         f"{en.MAIN_SIZE}, got {list(sizes)}")
    out = {f"s{stride}": {"size": en.MAIN_SIZE, "stride": stride}}
    for r in sizes:
        out[f"r{r}"] = {"size": r}
    return out


def tokens(size: int) -> int:
    return (size//en.PATCH)**2


# ---------------------------------------------------------------- 编码器

def fake_encoders(config: DroidConfig) -> dict[str, en.Encoder]:
    """§6 的假编码器：s5 用 `fake_encoder(256)`（和主特征相同），rR 用 `fake_encoder(R)`。"""
    return {kind: en.fake_encoder(spec["size"]) for kind, spec in kinds(config).items()}


def native_real_encoder(loaded: vjepa.LoadedEncoder, device: str, size: int) -> en.Encoder:
    """§1：同一个已载入的编码器按边长 `size` 编码（`NativeFrameEncoder`，单帧复制两份）；预处理 `preprocess_square(·, size)`。"""
    frozen = ne.NativeFrameEncoder(loaded, preprocess=cache.preprocess_spec(), encoder_device=torch.device(device),
                                   autocast_dtype=None, resolution=size)

    def encode(x: torch.Tensor) -> torch.Tensor:
        times = torch.arange(x.shape[0], dtype=torch.float64)      # 只是元数据，编码器不用它
        return frozen.encode_target(x, target_time_seconds=times).raw_z.detach().cpu().to(torch.float32)

    def preprocess(frame: np.ndarray) -> torch.Tensor:
        return en.preprocess_square(torch.from_numpy(np.ascontiguousarray(frame))[None], size)

    return en.Encoder(encode, preprocess, en.REAL_ROLE, cache.CHECKPOINT_SHA, en.square_preprocess_sha256(size),
                      en.load_report(loaded), str(frozen.encoder_device))


def real_encoders(config: DroidConfig, device: str, *, loaded: vjepa.LoadedEncoder,
                  main: en.Encoder | None = None) -> dict[str, en.Encoder]:
    """真实编码器：s5 = 主特征的 256 编码器（v7 预处理），rR = `native_real_encoder(loaded, device, R)`。"""
    main = main or en.real_encoder(None, None, device, loaded=loaded)
    return {kind: (main if "stride" in spec else native_real_encoder(loaded, device, spec["size"]))
            for kind, spec in kinds(config).items()}


# ---------------------------------------------------------------- 一条回合

def s5_index(info: dict, stride: int) -> np.ndarray:
    """§4：第 0、stride、2·stride…… 帧到 V − 1 为止，F5 = ⌊(V − 1)/stride⌋ + 1；要和准备文件的 `frame_sha256_s5` 等长。"""
    v = info.get("video_frames")
    shas = info.get("frame_sha256_s5")
    if (info.get("limits") or {}).get("s5_stride_frames") != stride:
        raise ValueError(f"episode {info.get('uuid')}: prepared with s5 stride "
                         f"{(info.get('limits') or {}).get('s5_stride_frames')}, config says {stride}")
    if not isinstance(shas, list):
        raise ValueError(f"episode {info.get('uuid')}: the prepared JSON has no frame_sha256_s5 "
                         "(prepare/<shard> records it only for the first test_extra_episodes of each test shard)")
    if type(v) is not int or v <= 0 or len(shas) != (v - 1)//stride + 1:
        raise ValueError(f"episode {info.get('uuid')}: {len(shas)} s5 frame SHAs for V = {v} frames "
                         f"(expected ⌊(V − 1)/{stride}⌋ + 1)")
    return np.arange(0, v, stride, dtype=np.int64)


def decode_frames(mp4_path: Path, wanted: dict[int, str]) -> dict[int, np.ndarray]:
    """顺序解码一次，取 `wanted`（帧号 → 准备文件里的 RGB SHA）里的帧，逐帧核 SHA；不同或视频比准备时短就报错。"""
    if not wanted:
        raise ValueError(f"{mp4_path}: no frames requested")
    last, out = max(wanted), {}
    for index, frame in video.decode_rgb(mp4_path):
        if index in wanted:
            if video.frame_sha256(frame) != wanted[index]:
                raise ValueError(f"{mp4_path}: frame {index} decodes differently from the prepared frame SHA; "
                                 "the video or the decoder changed after prepare")
            out[index] = frame
        if index >= last:
            break
    missing = sorted(set(wanted) - set(out))
    if missing:
        raise ValueError(f"{mp4_path}: frames {missing[:5]} were not decoded (video shorter than prepared)")
    return out


def _wanted(targets: dict[str, tuple[np.ndarray, list[str]]]) -> dict[int, str]:
    wanted: dict[int, str] = {}
    for name, (index, shas) in targets.items():
        if len(index) != len(shas):
            raise ValueError(f"{name}: {len(index)} frame positions but {len(shas)} frame SHAs")
        for i, sha in zip(index.tolist(), shas):
            if wanted.setdefault(int(i), sha) != sha:
                raise ValueError(f"frame {i}: the prepared JSON gives two different SHAs ({name})")
    return wanted


def encode_kinds(encoders: dict[str, en.Encoder], todo: list[str], config: DroidConfig, data_root: Path | str,
                 shard: str, episode: dict, prepared: pr.PreparedEpisode, prepared_sha256: str) -> dict[str, dict]:
    """编一条回合的 `todo` 里的几种附加特征，写 `test_extra/<uuid>.<kind>.{npy,json}`；返回 {kind: 特征 JSON 的内容}。"""
    info, uuid = prepared.info, episode["uuid"]
    if info["uuid"] != uuid:
        raise ValueError(f"prepared episode {info['uuid']} is not {uuid}")
    specs = kinds(config)
    main_index = prepared.arrays["frame_index"]
    positions = {}
    for kind in todo:
        spec = specs[kind]
        positions[kind] = ((s5_index(info, spec["stride"]), list(info["frame_sha256_s5"])) if "stride" in spec
                           else (main_index, list(info["frame_sha256"])))
    mp4 = dl.episode_dir(data_root, shard, uuid)/f"{episode['left_serial']}.mp4"
    started = time.perf_counter()
    frames = decode_frames(mp4, _wanted({"frame_sha256": (main_index, list(info["frame_sha256"])), **positions}))
    decode_seconds = time.perf_counter() - started
    out = {}
    for kind in todo:
        started = time.perf_counter()
        index, shas = positions[kind]
        encoder, size = encoders[kind], specs[kind]["size"]
        x = torch.cat([encoder.preprocess(frames[int(i)]) for i in index])
        if tuple(x.shape[1:]) != (3, size, size):
            raise ValueError(f"{kind}: the encoder's preprocess gives {list(x.shape[1:])}, expected [3, {size}, {size}]")
        raw = en.encode_frames(encoder, x, config.encode.batch_frames)
        if raw.shape != (len(index), tokens(size), ft.WIDTH):
            raise ValueError(f"episode {uuid} {kind}: features {list(raw.shape)}, expected "
                             f"[{len(index)}, {tokens(size)}, {ft.WIDTH}]")
        stored, stats = ft.to_storage(raw)
        npy_path, json_path = ft.extra_paths(data_root, uuid, kind)

        def record_for(file_sha: str) -> dict:
            return {
                "schema": ft.EXTRA_SCHEMA,
                "uuid": uuid, "shard": shard, "kind": kind, "rows": int(stored.shape[0]), "shape": list(stored.shape),
                "dtype": ft.STORAGE_DTYPE, "file": npy_path.name, "file_sha256": file_sha,
                "row_sha256": ft.row_sha256(stored),
                "frame_index": [int(i) for i in index],
                "frame_sha256": shas,
                "prepared_sha256": prepared_sha256,
                "encoder": en.identity(encoder, config),
                "encoder_load_report": encoder.load_report,
                "feature_stats": stats,
                "source": {"code_commit": ix.code_commit(),
                           "code_sha256": {name: cache.file_sha(cache.ROOT/name) for name in SOURCE_FILES},
                           **en._environment(encoder.device)},
                "seconds": round(time.perf_counter() - started + decode_seconds/len(todo), 3),
            }

        out[kind], _ = ft.write_features(npy_path, json_path, stored, record_for)
    return out


def current(record: dict | None, npy_path: Path, shard: str, kind: str, prepared_sha256: str, ident: dict) -> bool:
    """§3 的跳过规则（`encode.current`）另要求 kind 一致。"""
    return record is not None and record.get("kind") == kind and en.current(record, npy_path, shard, prepared_sha256,
                                                                            ident)


# ---------------------------------------------------------------- 和主特征比（§4，只作记录）

def main_comparison(config: DroidConfig, data_root: Path | str, shard: str, entries: list[dict], s5_kind: str,
                    s5_ident: dict) -> dict:
    """s5 第 j' 行和主特征第 j 行取的是同一帧（帧号相等）时，两者的行 SHA 是否相同。该片主特征在本配置下没完成就不比。"""
    if not ft.complete_path(data_root, shard, config).is_file():
        return {"main_manifest_sha256": None, "same_encoder": None, "rows_compared": 0, "rows_identical": 0}
    manifest, sha = ft.read_manifest(data_root, shard, config)
    main = {e["uuid"]: e for e in manifest["episodes"]}
    same = manifest["encoder"] == s5_ident
    compared = identical = 0
    if same:
        for entry in entries:
            if entry["uuid"] not in main:
                continue
            _, main_json = ft.feature_paths(data_root, shard, entry["uuid"])
            if cache.file_sha(main_json) != main[entry["uuid"]]["feature_json_sha256"]:
                raise ValueError(f"{main_json} does not match the manifest of {shard}")
            m = ft.read_feature_record(main_json)
            s = ft.read_extra_record(ft.extra_paths(data_root, entry["uuid"], s5_kind)[1], entry["uuid"], s5_kind)
            row_of = {f: j for j, f in enumerate(s["frame_index"])}
            for j, f in enumerate(m["frame_index"]):
                if f not in row_of:
                    raise ValueError(f"episode {entry['uuid']}: main frame {f} is not an s5 frame")
                compared += 1
                identical += m["row_sha256"][j] == s["row_sha256"][row_of[f]]
    return {"main_manifest_sha256": sha, "same_encoder": same, "rows_compared": compared, "rows_identical": identical}


# ---------------------------------------------------------------- 任务

def run_test_extra(config: DroidConfig, data_root: Path | str, encoders: dict[str, en.Encoder], *,
                   log: Callable[[str], None] | None = None) -> dict:
    """任务 `encode/test-extra`。完成记录已存在就什么都不做，返回它。

    前置条件：`prepare/id_test`、`prepare/ood_test` 在本配置下已完成（12）。每片前 `test_extra_episodes` 条：准备没通过的
    记为跳过（原因抄自准备文件，不补位）；每种 kind 已有特征且 §3 的各项一致就跳过；否则解码一次、编需要的几种、写文件。
    """
    if config.encode.upstream_commit != vjepa.UPSTREAM_COMMIT:
        raise ValueError(f"encode.upstream_commit {config.encode.upstream_commit} != {vjepa.UPSTREAM_COMMIT}")
    if config.encode.storage_dtype != ft.STORAGE_DTYPE:
        raise ValueError(f"encode.storage_dtype must be {ft.STORAGE_DTYPE}")
    specs = kinds(config)
    if sorted(encoders) != sorted(specs):
        raise ValueError(f"encoders for {sorted(encoders)}, this config needs {sorted(specs)}")
    done = ft.extra_complete_path(data_root, config)
    if done.exists():
        return ft.read_json(done)
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))
    idents = {kind: en.identity(encoders[kind], config) for kind in specs}
    s5_kind = next(kind for kind, spec in specs.items() if "stride" in spec)
    episodes, skipped, statuses = [], [], {"encoded": 0, "current": 0}
    prepare_complete, comparison = {}, {}
    started = time.perf_counter()
    for shard in SHARDS:
        uuids, shas, index_rows, prepare_complete[shard] = en._prepared_records(config, data_root, shard)
        chosen = uuids[:config.encode.test_extra_episodes]
        shard_entries = []
        for n, uuid in enumerate(chosen, 1):
            json_path, _ = pr.episode_paths(data_root, shard, uuid)
            prepared_record = pr.read_record(json_path)
            if not en.encodable(prepared_record):
                skipped.append({"uuid": uuid, "shard": shard, "reasons": list(prepared_record["reasons"])})
                continue
            records, todo = {}, []
            for kind in specs:
                npy_path, feature_json = ft.extra_paths(data_root, uuid, kind)
                record = ft.read_extra_record(feature_json, uuid, kind) if feature_json.is_file() else None
                if current(record, npy_path, shard, kind, shas[uuid], idents[kind]):
                    records[kind] = record
                    statuses["current"] += 1
                else:
                    todo.append(kind)
            if todo:
                if (dl.episode_dir(data_root, shard, uuid)/dl.RAW_DELETED_FILE).exists():
                    raise ValueError(f"episode {uuid}: raw files were deleted but its test-extra features are "
                                     "missing or stale")
                prepared = pr.load_prepared(json_path)
                if prepared.json_sha256 != shas[uuid]:
                    raise ValueError(f"{json_path} changed while encode/{JOB} was running")
                records |= encode_kinds(encoders, todo, config, data_root, shard, index_rows[uuid], prepared,
                                        shas[uuid])
                statuses["encoded"] += len(todo)
            entry = {"uuid": uuid, "shard": shard,
                     "files": {kind: {"feature_json_sha256": cache.file_sha(ft.extra_paths(data_root, uuid, kind)[1]),
                                      "file_sha256": records[kind]["file_sha256"], "rows": records[kind]["rows"]}
                               for kind in specs}}
            episodes.append(entry)
            shard_entries.append(entry)
            if n == 1 or n % 50 == 0:
                log(f"[encode/{JOB}] {shard} {n}/{len(chosen)} episodes, {time.perf_counter() - started:.0f} s")
        comparison[shard] = main_comparison(config, data_root, shard, shard_entries, s5_kind, idents[s5_kind])
    manifest = {
        "schema": ft.EXTRA_MANIFEST_SCHEMA,
        "config": config.name, "config_sha256": config.config_sha256,
        "splits_sha256": cache.file_sha(ix.index_dir(data_root)/ix.SPLITS_FILE),
        "prepare_complete_sha256": prepare_complete,
        "test_extra_episodes": config.encode.test_extra_episodes,
        "kinds": specs,
        "encoder": idents,
        "episodes": episodes,
        "skipped": skipped,
        "counts": {"episodes": len(episodes) + len(skipped), "encoded": len(episodes), "skipped": len(skipped),
                   "rows": {kind: sum(e["files"][kind]["rows"] for e in episodes) for kind in specs},
                   "by_shard": {s: sum(e["shard"] == s for e in episodes) for s in SHARDS}},
        "main_comparison": comparison,
    }
    path = ft.extra_manifest_path(data_root, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in (path, path.with_name(path.name + ".partial"), done.with_name(done.name + ".partial")):
        stale.unlink(missing_ok=True)
    cache._write_json(path, manifest)
    record = {"schema": ft.EXTRA_COMPLETE_SCHEMA, "config": config.name, "config_sha256": config.config_sha256,
              "manifest_sha256": cache.file_sha(path), "code_commit": ix.code_commit()}
    done.parent.mkdir(parents=True, exist_ok=True)
    cache._write_json(done, record)
    return dict(record, statuses=statuses)


# ---------------------------------------------------------------- 05 §7 第二项（GPU 机器上）

def check_extra(encoders: dict[str, en.Encoder], config: DroidConfig, mp4_path: Path) -> dict:
    """192 和 320 输入各编第 0 帧：输出形状 [1, (R/16)², 1408]、全部有限。"""
    frame = next(iter(video.decode_rgb(mp4_path)), (None, None))[1]
    if frame is None:
        raise ValueError(f"{mp4_path}: no frames decoded")
    out = {}
    for kind, spec in kinds(config).items():
        if "stride" in spec:
            continue
        z = encoders[kind].encode(encoders[kind].preprocess(frame))
        out[kind] = {"shape": list(z.shape), "finite": bool(torch.isfinite(z).all()),
                     "ok": tuple(z.shape) == (1, tokens(spec["size"]), ft.WIDTH) and bool(torch.isfinite(z).all())}
    return out
