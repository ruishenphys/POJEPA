"""DROID 编码（docs/design/05-droid-encode.md §1–§3；S06a）。

阶段 4：任务 `encode/<片名>` 用冻结的 V-JEPA 2 ViT-g 把这片每条准备检查通过的回合的取帧位置编码成特征
（`features.py` 的格式），训练和评测之后只读特征、不跑编码器。

- 编码器和预处理和 v7 完全相同（§1）：`vjepa.build_encoder_architecture` + `load_local_encoder`（权重 SHA 必须等于
  `cache.CHECKPOINT_SHA`）、`vjepa.preprocess_rgb`（每帧前重置 `torch.Generator().manual_seed(0)`，v7 的规则）、
  `FrozenFrameEncoder`（单帧复制两份凑 tubelet，float32，不开 autocast）。每次送 `batch_frames`（32）帧，按行序连续切块。
- `preprocess_square(rgb, size)`：同样的切片 [0:720, 154:1126]、双线性缩到 size×size、同一组均值方差；size = 256 时和
  v7 `preprocess_rgb` 逐位相同（tests/test_droid_encode.py）。主特征的假编码器路径用它（CPU 测试没有上游代码）；
  192、320 的测试附加编码（§4）在 S06b 用它。
- 重新解码 MP4（04 §2），逐帧核 RGB 的 SHA 等于准备文件的 `frame_sha256`（§2 第 1 条），不同就报错。
- 跳过规则（§3）、片清单、完成记录、`--delete-raw`（03 §4）见各函数。

命令行：`python -m po_jepa.droid.encode --config … --data-root … --shard <片名> [--delete-raw] [--device cuda]
[--upstream-root …] [--checkpoint …] [--fake-encoder]`；上游代码和权重默认在数据根的 `upstream/vjepa2` 和
`checkpoints/<encode.checkpoint_file>`（12 §1）。`--check-mp4 <路径>` 做 05 §7 第一项核对（GPU 机器上，合并前）。
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from po_jepa import vjepa
from po_jepa.droid import download as dl
from po_jepa.droid import features as ft
from po_jepa.droid import index as ix
from po_jepa.droid import prepare as pr
from po_jepa.droid import splits as sp
from po_jepa.droid import video
from po_jepa.droid.config import ACTION_SOURCES, DroidConfig, load_config
from po_jepa.runtime import cache

# §1：1280×720 在官方 random_resized_crop（scale 1.777、ratio [0.75, 1.35]）里固定落到中心回退：
# h = 720、w = round(720·1.35) = 972、i = 0、j = (1280 − 972)//2 = 154
CROP_ROWS = (0, 720)
CROP_COLS = (154, 1126)
MAIN_SIZE = 256
PATCH = 16
PREPROCESS_SEED = 0                      # §1：generator 固定 manual_seed(0)，只为满足接口（中心回退没有随机性）
REAL_ROLE = "real_frozen_encoder"
SYNTHETIC_ROLE = "synthetic_development_only"   # §6：假编码器的特征只允许跑冒烟测试
RAW_DELETED_SCHEMA = "droid_raw_deleted_v1"
# 和 tools/encode_v7.py 同一份结构（tests/test_droid_encode.py 按 AST 核对）
ARCHITECTURE_ARGS = (vjepa.UPSTREAM_COMMIT, "configs/train/vitg16/droid-256px-8f.yaml",
                     "vit_giant_xformers", 256, 16, 2, 512, True, True, False, True, True, True)
SOURCE_FILES = ("src/po_jepa/droid/encode.py", "src/po_jepa/droid/features.py", "src/po_jepa/droid/video.py",
                "src/po_jepa/vjepa.py")


# ---------------------------------------------------------------- 预处理（§1）

def _mean_std() -> tuple[torch.Tensor, torch.Tensor]:
    p = cache.preprocess_spec()
    return (255*torch.tensor(p.mean, dtype=torch.float32, device="cpu"),
            255*torch.tensor(p.std, dtype=torch.float32, device="cpu"))


def preprocess_square(rgb: torch.Tensor, size: int) -> torch.Tensor:
    """CPU uint8 RGB [B, 720, 1280, 3] → float32 [B, 3, size, size]：切片 [0:720, 154:1126]，
    `F.interpolate(…, size=(size, size), mode="bilinear", align_corners=False)`，再 (x − 255·mean)/(255·std)。
    每张图按官方的排布（[3, 1, H, W]）单独缩放，和 v7 `preprocess_rgb` 的算术完全相同。"""
    if (not isinstance(rgb, torch.Tensor) or rgb.device.type != "cpu" or rgb.dtype != torch.uint8
            or rgb.ndim != 4 or tuple(rgb.shape[1:]) != video.FRAME_SHAPE or rgb.shape[0] <= 0):
        raise ValueError(f"RGB must be nonempty CPU uint8 [B, {', '.join(map(str, video.FRAME_SHAPE))}]")
    if not isinstance(size, int) or size <= 0 or size % PATCH:
        raise ValueError(f"size must be a positive multiple of {PATCH}")
    mean, std = _mean_std()
    out = []
    with torch.no_grad():
        for image in rgb:
            frame = image.to(torch.float32).permute(2, 0, 1).unsqueeze(1)          # [3, 1, H, W]
            cropped = frame[:, :, CROP_ROWS[0]:CROP_ROWS[1], CROP_COLS[0]:CROP_COLS[1]]
            resized = torch.nn.functional.interpolate(cropped, size=(size, size), mode="bilinear", align_corners=False)
            out.append((resized[:, 0] - mean[:, None, None])/std[:, None, None])
    return torch.stack(out)


def square_preprocess_sha256(size: int) -> str:
    """`preprocess_square(·, size)` 的身份（假编码器和 S06b 附加编码记这个；真实主特征记 v7 的预处理 SHA）。"""
    p = cache.preprocess_spec()
    return hashlib.sha256(cache._json_bytes(dict(
        function="po_jepa.droid.encode.preprocess_square", size=size, crop_rows=list(CROP_ROWS),
        crop_cols=list(CROP_COLS), interpolate="bilinear, align_corners=False", mean=list(p.mean), std=list(p.std),
        input="uint8 RGB 720x1280, 0-255"))).hexdigest()


def preprocess_v7(frame: np.ndarray) -> torch.Tensor:
    """真实编码器的 256 预处理：v7 `vjepa.preprocess_rgb`，每帧前新建 `Generator().manual_seed(0)`（v7 的规则）。"""
    generator = torch.Generator(device="cpu").manual_seed(PREPROCESS_SEED)
    return vjepa.preprocess_rgb(torch.from_numpy(np.ascontiguousarray(frame))[None], spec=cache.preprocess_spec(),
                                generator=generator)


# ---------------------------------------------------------------- 编码器

@dataclass(frozen=True)
class Encoder:
    """编码一块预处理好的帧：float32 [B, 3, S, S] → float32 CPU [B, (S/16)², 1408]。"""

    encode: Callable[[torch.Tensor], torch.Tensor]
    preprocess: Callable[[np.ndarray], torch.Tensor]   # 一帧 uint8 [720, 1280, 3] → [1, 3, 256, 256]
    role: str
    checkpoint_sha256: str | None
    preprocess_sha256: str
    load_report: dict
    device: str


def fake_encoder() -> Encoder:
    """§6：确定性随机 16×16 图块投影（固定种子 0 的 [768, 1408] 矩阵 / 16，再 layer_norm，同 v7 encode_v7 的思路）。
    任意 16 的倍数边长 S 都得到 (S/16)² 个 token。只供 CPU 流程测试：清单里权重 SHA 为 None、`synthetic_development_only`。"""
    weight = torch.randn(3*PATCH*PATCH, ft.WIDTH, generator=torch.Generator().manual_seed(0))/16.

    def encode(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        patches = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)              # B,3,h/16,w/16,16,16
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(b, (h//PATCH)*(w//PATCH), -1)
        return torch.nn.functional.layer_norm(patches @ weight, (ft.WIDTH,))

    return Encoder(encode, lambda frame: preprocess_square(torch.from_numpy(np.ascontiguousarray(frame))[None],
                                                           MAIN_SIZE),
                   SYNTHETIC_ROLE, None, square_preprocess_sha256(MAIN_SIZE),
                   {"kind": "fake random patch projection (seed 0)"}, "cpu")


def real_encoder(upstream_root: Path, checkpoint: Path, device: str) -> Encoder:
    """§1：上游 vjepa2@204698b 的 `vit_giant_xformers`，权重 SHA 必须等于 `cache.CHECKPOINT_SHA`（不对就报错）。"""
    arch = vjepa.EncoderArchitectureSpec(*ARCHITECTURE_ARGS)
    build = vjepa.build_encoder_architecture(upstream_root=Path(upstream_root), architecture=arch,
                                             initialization_seed=0)
    loaded = vjepa.load_local_encoder(build, checkpoint=vjepa.LocalCheckpointSpec(
        Path(checkpoint), cache.CHECKPOINT_SHA, cache.PUBLICATION, "target_encoder", ("module.", "backbone."), True,
        (), ()))
    frozen = vjepa.FrozenFrameEncoder(loaded, preprocess=cache.preprocess_spec(), encoder_device=torch.device(device),
                                      autocast_dtype=None)

    def encode(x: torch.Tensor) -> torch.Tensor:
        times = torch.arange(x.shape[0], dtype=torch.float64)      # 只是元数据，编码器不用它
        return frozen.encode_target(x, target_time_seconds=times).raw_z.detach().cpu().to(torch.float32)

    r = loaded.report
    report = {"path": str(r.path), "sha256": r.sha256, "encoder_key": r.encoder_key, "strict": r.strict,
              "missing_keys": list(r.missing_keys), "unexpected_keys": list(r.unexpected_keys)}
    return Encoder(encode, preprocess_v7, REAL_ROLE, cache.CHECKPOINT_SHA, cache.v7_preprocess_sha256(), report,
                   str(frozen.encoder_device))


def encode_frames(encoder: Encoder, frames: torch.Tensor, batch_frames: int) -> np.ndarray:
    """预处理好的 [F, 3, S, S] 按行序连续切成 `batch_frames` 一块（最后一块可以不满）送进编码器 → float32 [F, N, 1408]。"""
    if frames.ndim != 4 or frames.shape[0] <= 0 or frames.shape[2] != frames.shape[3]:
        raise ValueError("frames must be [F, 3, S, S] with F > 0")
    tokens = (frames.shape[2]//PATCH)**2
    out = []
    for start in range(0, frames.shape[0], batch_frames):
        chunk = frames[start:start + batch_frames]
        z = encoder.encode(chunk)
        if (not isinstance(z, torch.Tensor) or z.dtype != torch.float32 or z.device.type != "cpu"
                or tuple(z.shape) != (chunk.shape[0], tokens, ft.WIDTH)):
            raise ValueError(f"encoder returned {getattr(z, 'dtype', type(z))} {list(getattr(z, 'shape', ()))}, "
                             f"expected CPU float32 [{chunk.shape[0]}, {tokens}, {ft.WIDTH}]")
        out.append(z.numpy())
    raw = np.concatenate(out, axis=0)
    if not np.isfinite(raw).all():
        raise ValueError("encoder output has non-finite values")
    return raw


# ---------------------------------------------------------------- 一条回合（§2）

def decode_rows(mp4_path: Path, frame_index: np.ndarray, frame_sha256: list[str],
                preprocess: Callable[[np.ndarray], torch.Tensor]) -> torch.Tensor:
    """重新解码 MP4（04 §2，顺序 read），取 `frame_index` 处的帧，逐帧核 RGB 的 SHA 等于准备文件，预处理。"""
    wanted = {int(i): k for k, i in enumerate(frame_index)}
    if len(wanted) != len(frame_sha256) or not wanted:
        raise ValueError(f"{mp4_path}: {len(frame_index)} frame positions but {len(frame_sha256)} frame SHAs")
    last, out = int(frame_index[-1]), [None]*len(wanted)
    for index, frame in video.decode_rgb(mp4_path):
        if index in wanted:
            k = wanted[index]
            if video.frame_sha256(frame) != frame_sha256[k]:
                raise ValueError(f"{mp4_path}: frame {index} decodes differently from the prepared frame_sha256 "
                                 f"(row {k}); the video or the decoder changed after prepare")
            out[k] = preprocess(frame)
        if index >= last:
            break
    missing = [int(frame_index[k]) for k, v in enumerate(out) if v is None]
    if missing:
        raise ValueError(f"{mp4_path}: frames {missing[:5]} were not decoded (video shorter than prepared)")
    return torch.cat(out)


@functools.lru_cache(maxsize=None)
def _environment(device: str) -> dict:
    """版本、设备、GPU 型号和驱动（每个设备只查一次）。"""
    gpu, driver = None, None
    if device.startswith("cuda"):
        gpu = torch.cuda.get_device_name(torch.device(device))
        try:
            driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            driver = None
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "numpy": np.__version__, "opencv": video.opencv_version(), "platform": platform.platform(),
            "device": device, "gpu": gpu, "driver": driver}


def identity(encoder: Encoder, config: DroidConfig) -> dict:
    """特征 JSON 里的编码器身份；§3 的跳过规则要求这几项和当前完全一致。"""
    return {"evidence_role": encoder.role, "encoder_checkpoint_sha256": encoder.checkpoint_sha256,
            "preprocess_sha256": encoder.preprocess_sha256, "batch_frames": config.encode.batch_frames,
            "upstream_commit": config.encode.upstream_commit}


def current(record: dict, npy_path: Path, shard: str, prepared_sha256: str, ident: dict) -> bool:
    """§3：已有特征 JSON，且其中的文件 SHA（重新算）、准备文件 JSON 的 SHA、权重 SHA、预处理 SHA、batch_frames
    都和当前一致（另要求片名、证据角色、上游提交一致），就跳过这条。"""
    return (record.get("shard") == shard and record.get("prepared_sha256") == prepared_sha256
            and record.get("encoder") == ident and npy_path.is_file()
            and cache.file_sha(npy_path) == record.get("file_sha256"))


def encode_episode(encoder: Encoder, config: DroidConfig, data_root: Path | str, shard: str, episode: dict,
                   prepared: pr.PreparedEpisode, prepared_sha256: str) -> dict:
    """编一条回合（`episode` 是索引里的回合行，用到 uuid 和 left_serial）并写 `<uuid>.{npy,json}`；返回特征 JSON 的内容。"""
    started = time.perf_counter()
    info, uuid = prepared.info, episode["uuid"]
    if info["uuid"] != uuid:
        raise ValueError(f"prepared episode {info['uuid']} is not {uuid}")
    mp4 = dl.episode_dir(data_root, shard, uuid)/f"{episode['left_serial']}.mp4"
    frames = decode_rows(mp4, prepared.arrays["frame_index"], info["frame_sha256"], encoder.preprocess)
    raw = encode_frames(encoder, frames, config.encode.batch_frames)
    if raw.shape != (info["rows"], ft.TOKENS, ft.WIDTH):
        raise ValueError(f"episode {uuid}: features {list(raw.shape)}, expected [{info['rows']}, {ft.TOKENS}, {ft.WIDTH}]")
    stored, stats = ft.to_storage(raw)
    npy_path, json_path = ft.feature_paths(data_root, shard, uuid)

    def record_for(file_sha: str) -> dict:
        return {
            "schema": ft.FEATURE_SCHEMA,
            "uuid": uuid, "shard": shard, "rows": int(stored.shape[0]), "shape": list(stored.shape),
            "dtype": ft.STORAGE_DTYPE, "file": npy_path.name, "file_sha256": file_sha,
            "row_sha256": ft.row_sha256(stored),
            "frame_index": prepared.arrays["frame_index"].tolist(),
            "frame_sha256": list(info["frame_sha256"]),
            "prepared_sha256": prepared_sha256,
            "encoder": identity(encoder, config),
            "encoder_load_report": encoder.load_report,
            "feature_stats": stats,
            "source": {"code_commit": ix.code_commit(),
                       "code_sha256": {name: cache.file_sha(cache.ROOT/name) for name in SOURCE_FILES},
                       **_environment(encoder.device)},
            "seconds": round(time.perf_counter() - started, 3),
        }

    record, _ = ft.write_features(npy_path, json_path, stored, record_for)
    return record


# ---------------------------------------------------------------- 片任务（§3）

def encodable(record: dict) -> bool:
    """§2：只编准备检查通过的回合。特征只和画面有关、和动作源无关，所以任一动作源通过就编（S06 定，两种配置共用一份特征）。"""
    return any(record.get(f"passed_{s}") is True for s in ACTION_SOURCES)


def _prepared_records(config: DroidConfig, data_root: Path | str, shard: str) -> tuple[list[str], dict, dict, str]:
    """前置条件：本配置的 `prepare/<片>` 已完成，uuid 列表等于片计划，每条准备 JSON 的 SHA 等于完成记录里的。"""
    splits = sp.load_splits(data_root, config)
    plan = sp.shard_plan(splits, config.data)
    if shard not in plan:
        raise ValueError(f"shard {shard!r} is not used by config {config.name!r}; its shards are {list(plan)}")
    uuids = plan[shard]
    done = pr.complete_path(data_root, shard, config)
    if not done.is_file():
        raise ValueError(f"prepare/{shard} has not completed for config {config.name!r} ({done} is missing)")
    complete = ft.read_json(done)
    if complete.get("uuids") != uuids:
        raise ValueError(f"{done}: its uuid list is not this config's plan for {shard}")
    shas = dict(zip(uuids, complete["prepared_sha256"]))
    for uuid in uuids:
        json_path, _ = pr.episode_paths(data_root, shard, uuid)
        if cache.file_sha(json_path) != shas[uuid]:
            raise ValueError(f"{json_path} changed after prepare/{shard} completed")
    episodes = {e["uuid"]: e for e in sp.read_episodes(ix.index_dir(data_root)) if e["uuid"] in shas}
    return uuids, shas, episodes, cache.file_sha(done)


def run_encode(config: DroidConfig, data_root: Path | str, shard: str, encoder: Encoder, *,
               delete_raw: bool = False, log: Callable[[str], None] | None = None) -> dict:
    """任务 `encode/<片名>`。完成记录已存在就不再编码（带 `--delete-raw` 时仍做删除那一步，见 `delete_raw_files`）。

    每条回合：准备没通过的记为跳过（原因抄自准备文件）；已有特征且 §3 的各项一致就跳过；否则重新解码、编码、写文件。
    原始文件已删（`RAW_DELETED.json`）而特征缺失或过期：报错。全部完成后写片清单和完成记录。
    """
    if delete_raw and not shard.startswith(sp.SHARD_PREFIX):
        raise ValueError(f"--delete-raw is only for training shards; {shard} keeps its raw files (03 §4)")
    if config.encode.upstream_commit != vjepa.UPSTREAM_COMMIT:
        raise ValueError(f"encode.upstream_commit {config.encode.upstream_commit} != {vjepa.UPSTREAM_COMMIT}")
    if config.encode.storage_dtype != ft.STORAGE_DTYPE:
        raise ValueError(f"encode.storage_dtype must be {ft.STORAGE_DTYPE}")
    done = ft.complete_path(data_root, shard, config)
    if done.exists():
        result = ft.read_json(done)
        if delete_raw:
            result = dict(result, raw_deleted=delete_raw_files(config, data_root, shard))
        return result
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))   # 进度写 stderr，stdout 只有结果 JSON
    uuids, shas, index_rows, prepare_complete_sha = _prepared_records(config, data_root, shard)
    ident = identity(encoder, config)
    episodes, skipped, statuses = [], [], {"encoded": 0, "current": 0}
    started = time.perf_counter()
    for n, uuid in enumerate(uuids, 1):
        json_path, _ = pr.episode_paths(data_root, shard, uuid)
        prepared_record = pr.read_record(json_path)
        if not encodable(prepared_record):
            skipped.append({"uuid": uuid, "reasons": list(prepared_record["reasons"])})
            continue
        npy_path, feature_json = ft.feature_paths(data_root, shard, uuid)
        record = ft.read_feature_record(feature_json) if feature_json.is_file() else None
        if record is not None and current(record, npy_path, shard, shas[uuid], ident):
            statuses["current"] += 1
        else:
            if (dl.episode_dir(data_root, shard, uuid)/dl.RAW_DELETED_FILE).exists():
                raise ValueError(f"episode {uuid}: raw files were deleted but its features are missing or stale")
            prepared = pr.load_prepared(json_path)
            if prepared.json_sha256 != shas[uuid]:
                raise ValueError(f"{json_path} changed while encode/{shard} was running")
            record = encode_episode(encoder, config, data_root, shard, index_rows[uuid], prepared, shas[uuid])
            statuses["encoded"] += 1
        episodes.append({"uuid": uuid, "feature_json_sha256": cache.file_sha(feature_json),
                         "file_sha256": record["file_sha256"], "rows": record["rows"]})
        if n == 1 or n % 50 == 0:
            log(f"[encode/{shard}] {n}/{len(uuids)} episodes, {time.perf_counter() - started:.0f} s")
    manifest = {
        "schema": ft.MANIFEST_SCHEMA,
        "shard": shard, "config": config.name, "config_sha256": config.config_sha256,
        "splits_sha256": cache.file_sha(ix.index_dir(data_root)/ix.SPLITS_FILE),
        "prepare_complete_sha256": prepare_complete_sha,
        "encoder": ident,
        "episodes": episodes,
        "skipped": skipped,
        "counts": {"episodes": len(uuids), "encoded": len(episodes), "skipped": len(skipped),
                   "rows": sum(e["rows"] for e in episodes)},
    }
    path = ft.manifest_path(data_root, shard, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in (path, path.with_name(path.name + ".partial"), done.with_name(done.name + ".partial")):
        stale.unlink(missing_ok=True)
    cache._write_json(path, manifest)
    record = {"schema": ft.COMPLETE_SCHEMA, "shard": shard, "config": config.name,
              "config_sha256": config.config_sha256, "manifest_sha256": cache.file_sha(path),
              "code_commit": ix.code_commit()}
    done.parent.mkdir(parents=True, exist_ok=True)
    cache._write_json(done, record)
    result = dict(record, statuses=statuses)
    if delete_raw:
        result["raw_deleted"] = delete_raw_files(config, data_root, shard)
    return result


def delete_raw_files(config: DroidConfig, data_root: Path | str, shard: str) -> dict:
    """03 §4：先把清单重新读一遍核对（`features.verify_shard`：每条特征 JSON 的 SHA、`.npy` 整个文件的 SHA、每行 SHA），
    通过后对每条编过的回合先写 `RAW_DELETED.json`（删除时间、特征 JSON 的 SHA、删的文件），再删 h5 和 MP4；
    保留 `metadata.json` 和 `DOWNLOADED.json`。准备没通过、没编的回合不删（S06 定：留着查原因，每片约 1%）。
    只对训练片。"""
    if not shard.startswith(sp.SHARD_PREFIX):
        raise ValueError(f"--delete-raw is only for training shards; {shard} keeps its raw files (03 §4)")
    manifest = ft.verify_shard(data_root, shard, config)
    deleted, already = 0, 0
    for entry in manifest["episodes"]:
        uuid = entry["uuid"]
        directory = dl.episode_dir(data_root, shard, uuid)
        if not (directory/dl.DOWNLOADED_FILE).is_file():
            raise ValueError(f"{directory}: {dl.DOWNLOADED_FILE} is missing; refusing to delete raw files")
        files = sorted(ft.read_json(directory/dl.DOWNLOADED_FILE)["files"])
        marker = directory/dl.RAW_DELETED_FILE
        if marker.exists():
            already += 1
        else:
            cache._write_json(marker, {"schema": RAW_DELETED_SCHEMA, "uuid": uuid, "shard": shard,
                                       "deleted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                       "feature_json_sha256": entry["feature_json_sha256"], "files": files})
            deleted += 1
        for name in files:                       # 标记写好之后再删（中断后重跑会把剩下的删掉）
            (directory/name).unlink(missing_ok=True)
    return {"deleted": deleted, "already_deleted": already, "kept": [s["uuid"] for s in manifest["skipped"]]}


# ---------------------------------------------------------------- 05 §7 真实编码器核对（GPU 机器上）

def check_real(encoder: Encoder, mp4_path: Path, batch_frames: int, rows: int = 32) -> dict:
    """取一条回合的前 `rows` 个取帧位置（每 15 帧），本模块的 `encode_frames` 和直接调同一编码器一次编 `rows` 帧
    的结果逐位相同；`preprocess_square(·, 256)` 和 v7 `preprocess_rgb` 在这些帧上逐位相同。"""
    raw_frames = []
    for index, frame in video.decode_rgb(mp4_path):
        if index % 15 == 0:
            raw_frames.append(frame)
            if len(raw_frames) == rows:
                break
    v7 = torch.cat([preprocess_v7(f) for f in raw_frames])
    square = preprocess_square(torch.from_numpy(np.stack(raw_frames)), MAIN_SIZE)
    ours = encode_frames(encoder, v7, batch_frames)
    direct = encoder.encode(v7).numpy()
    return {"rows": len(raw_frames), "preprocess_square_equals_v7": bool(torch.equal(v7, square)),
            "encode_frames_equals_direct": bool(np.array_equal(ours, direct)),
            "max_abs_difference": float(np.abs(ours - direct).max())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DROID 1.0.1 encoding (docs/design/05): one shard")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--shard", help="train-000 … , val, id_test, ood_test (02 §6)")
    parser.add_argument("--delete-raw", action="store_true", help="training shards only (03 §4)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--upstream-root", type=Path, help="default <data-root>/upstream/vjepa2")
    parser.add_argument("--checkpoint", type=Path, help="default <data-root>/checkpoints/<encode.checkpoint_file>")
    parser.add_argument("--fake-encoder", action="store_true",
                        help="CPU flow tests only: random patch projection, features marked synthetic_development_only")
    parser.add_argument("--check-mp4", type=Path, help="05 §7: compare encode_frames with a direct encoder call")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.fake_encoder:
        encoder = fake_encoder()
    else:
        encoder = real_encoder(args.upstream_root or args.data_root/"upstream"/"vjepa2",
                               args.checkpoint or args.data_root/"checkpoints"/config.encode.checkpoint_file,
                               args.device)
    if args.check_mp4:
        result = check_real(encoder, args.check_mp4, config.encode.batch_frames)
        print(json.dumps(result, indent=2))
        return 0 if result["preprocess_square_equals_v7"] and result["encode_frames_equals_direct"] else 1
    if not args.shard:
        parser.error("--shard is required")
    result = run_encode(config, args.data_root, args.shard, encoder, delete_raw=args.delete_raw)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
