"""DROID 特征文件、片清单、总清单和读取（docs/design/05-droid-encode.md §2、§3、§5）。

一条回合一个 `features/<片名>/<uuid>.npy`（float16 [F, 256, 1408]，第 i 行是第 15i 帧）和字节固定的
`<uuid>.json`（行数、形状、文件 SHA、**每一行的 SHA**、帧 SHA、准备文件 JSON 的 SHA、编码器和预处理的身份）。
`encode/<片名>` 全部完成后写片清单 `features/<片名>/MANIFEST.<配置名>.json`（S06 定：清单和配置有关，按配置名分
文件，同 S05 的 stats）和完成记录 `features/<片名>/_complete/<配置名>.json`。特征总清单 SHA（06 §5 的
`feature_manifest_sha256`）= SHA-256（按片名排序，把每片清单的 SHA 十六进制串依次连接）。

`FeatureReader` 只读需要的行（`np.load(mmap_mode="r")`），每读一行核这一行的 SHA，转成 float32 的 torch 张量。
测试附加（§4，S06b）：`features/test_extra/<uuid>.<kind>.{npy,json}`（kind = s5、r192、r320），清单
`features/test_extra/MANIFEST.<配置名>.json`、完成记录 `features/test_extra/_complete/<配置名>.json`，读取用
`ExtraFeatureReader`（同样逐行核 SHA）。
本模块只管文件格式，不跑编码器（编码在 `encode.py`）。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from po_jepa.droid import index as ix
from po_jepa.droid import splits as sp
from po_jepa.droid.config import DroidConfig
from po_jepa.runtime import cache

FEATURES_DIR = "features"
COMPLETE_DIR = "_complete"
RAW_CLEANUP_DIR = "_raw_deleted"        # 训练片 `encode/<片> --delete-raw` 的完成标志（S14-fix1）
MANIFEST_PREFIX = "MANIFEST."
TEMP_SUFFIX = ".tmp"                    # 05 §2 第 3 条：先写 .tmp 再改名
FEATURE_SCHEMA = "droid_features_v1"
MANIFEST_SCHEMA = "droid_feature_manifest_v1"
COMPLETE_SCHEMA = "droid_encode_complete_v1"
RAW_CLEANUP_SCHEMA = "droid_raw_cleanup_complete_v1"
STORAGE_DTYPE = "float16"
FLOAT16_MAX = 65504.0                   # 05 §2 第 3 条：转之前最大绝对值必须 < 65504
TOKENS = 256                            # 16×16 网格
WIDTH = 1408                            # ViT-g 的特征维数
EXTRA_DIR = "test_extra"                # §4：`encode/test-extra` 的输出目录（S06b）
EXTRA_SCHEMA = "droid_test_extra_features_v1"
EXTRA_MANIFEST_SCHEMA = "droid_test_extra_manifest_v1"
EXTRA_COMPLETE_SCHEMA = "droid_test_extra_complete_v1"


# ---------------------------------------------------------------- 路径

def features_dir(data_root: Path | str) -> Path:
    return Path(data_root)/FEATURES_DIR


def feature_paths(data_root: Path | str, shard: str, uuid: str) -> tuple[Path, Path]:
    """(`<uuid>.npy`, `<uuid>.json`)。"""
    base = features_dir(data_root)/shard
    return base/f"{uuid}.npy", base/f"{uuid}.json"


def manifest_path(data_root: Path | str, shard: str, config: DroidConfig) -> Path:
    return features_dir(data_root)/shard/f"{MANIFEST_PREFIX}{config.name}.json"


def complete_path(data_root: Path | str, shard: str, config: DroidConfig) -> Path:
    return features_dir(data_root)/shard/COMPLETE_DIR/f"{config.name}.json"


def raw_cleanup_path(data_root: Path | str, shard: str, config: DroidConfig) -> Path:
    """训练片编码任务（带 `--delete-raw`）的完成标志：`features/<片>/_raw_deleted/<name>.json`。由
    `encode.delete_raw_files` 在整片原始文件都删完之后才写（S14-fix1）；`complete_path` 只说明特征编完，不说明删完。"""
    return features_dir(data_root)/shard/RAW_CLEANUP_DIR/f"{config.name}.json"


# ---------------------------------------------------------------- 数组

def row_sha256(stored: np.ndarray) -> list[str]:
    """每一行（第一维）的 float16 字节的 SHA-256。"""
    return [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in stored]


def to_storage(raw: np.ndarray) -> tuple[np.ndarray, dict]:
    """float32 特征 → float16（05 §2 第 3 条）：先要求全部有限、最大绝对值 < 65504，再转；返回 (float16 数组,
    {max_abs, relative_rounding_error})，后者 = max|f16 − f32| / max|f32|（v7 encode_v7 同样做法）。"""
    if not isinstance(raw, np.ndarray) or raw.dtype != np.float32 or raw.ndim != 3 or raw.shape[2] != WIDTH:
        raise ValueError(f"features must be float32 [F, N, {WIDTH}], got {getattr(raw, 'dtype', type(raw))} "
                         f"{list(getattr(raw, 'shape', ()))}")
    if not np.isfinite(raw).all():
        raise ValueError("encoder output has non-finite values")
    peak = float(np.abs(raw).max()) if raw.size else 0.0
    if not peak < FLOAT16_MAX:
        raise ValueError(f"encoder output max |z| = {peak} does not fit float16 (must be < {FLOAT16_MAX})")
    stored = raw.astype(np.float16)
    rounding = float(np.abs(stored.astype(np.float64) - raw.astype(np.float64)).max()/max(peak, 1e-12)) if raw.size else 0.0
    return stored, {"max_abs": peak, "relative_rounding_error": rounding}


# ---------------------------------------------------------------- 写和读一条回合

def write_features(npy_path: Path, json_path: Path, stored: np.ndarray,
                   record_for: Callable[[str], dict]) -> tuple[dict, str]:
    """先删旧的 JSON（没有 JSON 就算没做完），写 `<uuid>.npy.tmp` 并算它的 SHA，`record_for(文件 SHA)` 给出 JSON 的内容，
    `.tmp` 改名成 `.npy`，最后写 JSON（`cache._write_json`，字节固定）。返回 (JSON 的内容, JSON 的 SHA)。"""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (json_path, json_path.with_name(json_path.name + ".partial")):
        path.unlink(missing_ok=True)
    temp = npy_path.with_name(npy_path.name + TEMP_SUFFIX)
    with temp.open("wb") as file:
        np.save(file, stored, allow_pickle=False)
    record = record_for(cache.file_sha(temp))
    os.replace(temp, npy_path)
    cache._write_json(json_path, record)
    return record, cache.file_sha(json_path)


def read_json(path: Path) -> dict:
    value = ix._loads(Path(path).read_bytes(), str(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def read_feature_record(json_path: Path) -> dict:
    record = read_json(json_path)
    if record.get("schema") != FEATURE_SCHEMA:
        raise ValueError(f"{json_path}: schema is not {FEATURE_SCHEMA!r}")
    if record.get("uuid") != Path(json_path).stem:
        raise ValueError(f"{json_path}: uuid {record.get('uuid')!r} does not match the file name")
    return record


def check_stored(npy: np.ndarray, record: dict, where: str) -> None:
    """数组的 dtype、形状和 JSON 记的一致，行数等于 rows 和 row_sha256 的长度。"""
    shape = record.get("shape")
    if str(npy.dtype) != STORAGE_DTYPE or list(npy.shape) != shape:
        raise ValueError(f"{where}: array is {npy.dtype} {list(npy.shape)}, the JSON says {STORAGE_DTYPE} {shape}")
    if record.get("rows") != npy.shape[0] or len(record.get("row_sha256") or ()) != npy.shape[0]:
        raise ValueError(f"{where}: rows {record.get('rows')} / {len(record.get('row_sha256') or ())} row SHAs "
                         f"do not match the array's {npy.shape[0]} rows")


# ---------------------------------------------------------------- 片清单和总清单（§3）

def read_manifest(data_root: Path | str, shard: str, config: DroidConfig) -> tuple[dict, str]:
    """读本配置这一片的完成记录和清单：完成记录在、配置 SHA 一致、清单的 SHA 等于完成记录里的。返回 (清单, 清单 SHA)。"""
    done = complete_path(data_root, shard, config)
    if not done.is_file():
        raise ValueError(f"encode/{shard} has not completed for config {config.name!r} ({done} is missing)")
    complete = read_json(done)
    if (complete.get("schema") != COMPLETE_SCHEMA or complete.get("shard") != shard
            or complete.get("config_sha256") != config.config_sha256):
        raise ValueError(f"{done} was not written by encode/{shard} under this config")
    path = manifest_path(data_root, shard, config)
    sha = cache.file_sha(path)
    if sha != complete.get("manifest_sha256"):
        raise ValueError(f"{path} changed after encode/{shard} completed")
    manifest = read_json(path)
    if (manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("shard") != shard
            or manifest.get("config_sha256") != config.config_sha256):
        raise ValueError(f"{path} is not this config's manifest for {shard}")
    return manifest, sha


def total_manifest_sha256(manifest_sha256: dict[str, str]) -> str:
    """特征总清单 SHA（05 §3、06 §5）：按片名排序，把每片清单 SHA 的十六进制串依次连接后取 SHA-256。"""
    if not manifest_sha256:
        raise ValueError("no shard manifests")
    return hashlib.sha256("".join(manifest_sha256[s] for s in sorted(manifest_sha256)).encode("ascii")).hexdigest()


def feature_manifests(data_root: Path | str, config: DroidConfig,
                      shards: Sequence[str] | None = None) -> tuple[dict[str, dict], str]:
    """本配置的片清单（默认是划分的全部片，02 §6）和特征总清单 SHA；任一片没完成或清单被改过就报错。"""
    if shards is None:
        shards = list(sp.shard_plan(sp.load_splits(data_root, config), config.data))
    manifests, shas = {}, {}
    for shard in shards:
        manifests[shard], shas[shard] = read_manifest(data_root, shard, config)
    return manifests, total_manifest_sha256(shas)


def verify_shard(data_root: Path | str, shard: str, config: DroidConfig) -> dict:
    """把一片的清单重新读一遍核对（`--delete-raw` 删之前，03 §4）：每条特征 JSON 的 SHA 等于清单、`.npy` 文件
    整个重算 SHA 等于 JSON、dtype/形状/行数一致、每一行的 SHA 一致。返回清单。"""
    manifest, _ = read_manifest(data_root, shard, config)
    for entry in manifest["episodes"]:
        npy_path, json_path = feature_paths(data_root, shard, entry["uuid"])
        if cache.file_sha(json_path) != entry["feature_json_sha256"]:
            raise ValueError(f"{json_path} does not match the manifest of {shard}")
        record = read_feature_record(json_path)
        if cache.file_sha(npy_path) != record["file_sha256"] or record["file_sha256"] != entry["file_sha256"]:
            raise ValueError(f"{npy_path} does not match its JSON / the manifest of {shard}")
        npy = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        check_stored(npy, record, str(npy_path))
        if record["rows"] != entry["rows"] or row_sha256(npy) != record["row_sha256"]:
            raise ValueError(f"{npy_path}: rows or row SHAs do not match")
    return manifest


# ---------------------------------------------------------------- 读特征（§5）

class FeatureReader:
    """`FeatureReader(data_root, manifests)`：`manifests` 是 {片名: 片清单}（`feature_manifests` 的第一项）。

    第一次读某条回合时核它的特征 JSON 的 SHA 等于清单、`.npy` 的 dtype 和形状等于 JSON（不重算整个文件的 SHA）；
    之后 `rows(uuid, 行号)` 只读需要的行，每一行都核 SHA（float16 字节），不对就报错；返回 float32 的 CPU 张量。
    """

    def __init__(self, data_root: Path | str, manifests: dict[str, dict]) -> None:
        self.data_root = Path(data_root)
        self._entries: dict[str, tuple[str, dict]] = {}
        for shard, manifest in manifests.items():
            if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("shard") != shard:
                raise ValueError(f"{shard}: not a feature manifest of this shard")
            for entry in manifest["episodes"]:
                if entry["uuid"] in self._entries:
                    raise ValueError(f"episode {entry['uuid']} appears in two shard manifests")
                self._entries[entry["uuid"]] = (shard, entry)
        self._open: dict[str, tuple[np.ndarray, dict]] = {}

    def __contains__(self, uuid: str) -> bool:
        return uuid in self._entries

    def uuids(self) -> list[str]:
        return list(self._entries)

    def record(self, uuid: str) -> dict:
        return self._episode(uuid)[1]

    def _episode(self, uuid: str) -> tuple[np.ndarray, dict]:
        if uuid not in self._open:
            if uuid not in self._entries:
                raise KeyError(f"episode {uuid} is not in the feature manifests")
            shard, entry = self._entries[uuid]
            npy_path, json_path = feature_paths(self.data_root, shard, uuid)
            if cache.file_sha(json_path) != entry["feature_json_sha256"]:
                raise ValueError(f"{json_path} does not match the manifest of {shard}")
            record = read_feature_record(json_path)
            npy = np.load(npy_path, mmap_mode="r", allow_pickle=False)
            check_stored(npy, record, str(npy_path))
            self._open[uuid] = (npy, record)
        return self._open[uuid]

    def rows(self, uuid: str, rows: Sequence[int]) -> torch.Tensor:
        """第 `rows` 行（每行 [N, 1408]）→ float32 [len(rows), N, 1408]；每一行先核 SHA。"""
        npy, record = self._episode(uuid)
        return checked_rows(npy, record, rows, f"episode {uuid}")


def checked_rows(npy: np.ndarray, record: dict, rows: Sequence[int], label: str) -> torch.Tensor:
    """读 `npy` 的第 `rows` 行，每一行先核 `record["row_sha256"]`（float16 字节），转成 float32 的 CPU 张量。"""
    out = []
    for r in rows:
        r = int(r)
        if not 0 <= r < npy.shape[0]:
            raise IndexError(f"{label}: row {r} outside [0, {npy.shape[0]})")
        row = np.array(npy[r])
        if hashlib.sha256(row.tobytes()).hexdigest() != record["row_sha256"][r]:
            raise ValueError(f"{label}: row {r} does not match its SHA (the feature file was changed)")
        out.append(row)
    if not out:
        return torch.zeros((0, *npy.shape[1:]), dtype=torch.float32)
    return torch.from_numpy(np.stack(out).astype(np.float32))


# ---------------------------------------------------------------- 测试附加（§4，S06b）

def extra_dir(data_root: Path | str) -> Path:
    return features_dir(data_root)/EXTRA_DIR


def extra_paths(data_root: Path | str, uuid: str, kind: str) -> tuple[Path, Path]:
    """(`test_extra/<uuid>.<kind>.npy`, `test_extra/<uuid>.<kind>.json`)，kind 是 s5、r192、r320。"""
    base = extra_dir(data_root)
    return base/f"{uuid}.{kind}.npy", base/f"{uuid}.{kind}.json"


def extra_manifest_path(data_root: Path | str, config: DroidConfig) -> Path:
    return extra_dir(data_root)/f"{MANIFEST_PREFIX}{config.name}.json"


def extra_complete_path(data_root: Path | str, config: DroidConfig) -> Path:
    return extra_dir(data_root)/COMPLETE_DIR/f"{config.name}.json"


def read_extra_record(json_path: Path, uuid: str, kind: str) -> dict:
    record = read_json(json_path)
    if record.get("schema") != EXTRA_SCHEMA:
        raise ValueError(f"{json_path}: schema is not {EXTRA_SCHEMA!r}")
    if record.get("uuid") != uuid or record.get("kind") != kind:
        raise ValueError(f"{json_path}: uuid/kind {record.get('uuid')!r}/{record.get('kind')!r} != {uuid!r}/{kind!r}")
    return record


def read_extra_manifest(data_root: Path | str, config: DroidConfig) -> tuple[dict, str]:
    """读本配置的测试附加完成记录和清单（同 `read_manifest`）。返回 (清单, 清单 SHA)。"""
    done = extra_complete_path(data_root, config)
    if not done.is_file():
        raise ValueError(f"encode/test-extra has not completed for config {config.name!r} ({done} is missing)")
    complete = read_json(done)
    if complete.get("schema") != EXTRA_COMPLETE_SCHEMA or complete.get("config_sha256") != config.config_sha256:
        raise ValueError(f"{done} was not written by encode/test-extra under this config")
    path = extra_manifest_path(data_root, config)
    sha = cache.file_sha(path)
    if sha != complete.get("manifest_sha256"):
        raise ValueError(f"{path} changed after encode/test-extra completed")
    manifest = read_json(path)
    if manifest.get("schema") != EXTRA_MANIFEST_SCHEMA or manifest.get("config_sha256") != config.config_sha256:
        raise ValueError(f"{path} is not this config's test-extra manifest")
    return manifest, sha


class ExtraFeatureReader:
    """`ExtraFeatureReader(data_root, manifest)`：`manifest` 是 `read_extra_manifest` 的第一项。

    `rows(uuid, kind, 行号)` 和 `FeatureReader.rows` 一样：第一次打开时核特征 JSON 的 SHA 等于清单、`.npy` 的 dtype
    和形状等于 JSON，每读一行核这一行的 SHA。s5 的行号是 s5 文件自己的（第 j 行 = 第 5j 帧），不和主特征混用（§4）。
    """

    def __init__(self, data_root: Path | str, manifest: dict) -> None:
        if manifest.get("schema") != EXTRA_MANIFEST_SCHEMA:
            raise ValueError("not a test-extra manifest")
        self.data_root = Path(data_root)
        self._entries = {}
        for entry in manifest["episodes"]:
            if entry["uuid"] in self._entries:
                raise ValueError(f"episode {entry['uuid']} appears twice in the test-extra manifest")
            self._entries[entry["uuid"]] = entry
        self._open: dict[tuple[str, str], tuple[np.ndarray, dict]] = {}

    def __contains__(self, uuid: str) -> bool:
        return uuid in self._entries

    def uuids(self) -> list[str]:
        return list(self._entries)

    def shard(self, uuid: str) -> str:
        return self._entries[uuid]["shard"]

    def record(self, uuid: str, kind: str) -> dict:
        return self._file(uuid, kind)[1]

    def _file(self, uuid: str, kind: str) -> tuple[np.ndarray, dict]:
        key = (uuid, kind)
        if key not in self._open:
            if uuid not in self._entries:
                raise KeyError(f"episode {uuid} is not in the test-extra manifest")
            files = self._entries[uuid]["files"]
            if kind not in files:
                raise KeyError(f"episode {uuid} has no {kind!r} features (has {sorted(files)})")
            npy_path, json_path = extra_paths(self.data_root, uuid, kind)
            if cache.file_sha(json_path) != files[kind]["feature_json_sha256"]:
                raise ValueError(f"{json_path} does not match the test-extra manifest")
            record = read_extra_record(json_path, uuid, kind)
            npy = np.load(npy_path, mmap_mode="r", allow_pickle=False)
            check_stored(npy, record, str(npy_path))
            self._open[key] = (npy, record)
        return self._open[key]

    def rows(self, uuid: str, kind: str, rows: Sequence[int]) -> torch.Tensor:
        npy, record = self._file(uuid, kind)
        return checked_rows(npy, record, rows, f"episode {uuid} {kind}")
