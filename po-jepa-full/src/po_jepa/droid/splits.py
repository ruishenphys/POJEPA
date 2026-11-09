"""DROID 1.0.1 划分（docs/design/02-droid-index-and-splits.md §5–§7）。

阶段 1（`index`）的后半：从 `index/episodes.jsonl` 的入选回合算出验证集、同分布测试集、留出实验室测试集和
完整训练顺序（§5），写 `index/splits.json`（§7），由 `index.run_index` 在写清单之前调用，清单因此记四个文件。
§6 的片由 `shard_plan` 从划分文件算出。所有排序都用 `index.salted_hash(split_salt, kind, key)` 升序。

`splits.json` 是一个数据根共用的文件：它只依赖入选回合和 `SPLIT_KEYS` 这几个配置键，不依赖 N（`train_episodes`）
和 `use_failures`。`train_order` 是完整训练顺序（04 §6、09 §3.1 的回合池要用它）；`use_failures: false` 时去掉失败
回合由 `train_order(splits, data)` 做，失败回合的 uuid 记在 `train_failures` 里（S03.md §4 第 1 条）。
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path

from po_jepa.droid import index as ix
from po_jepa.droid.config import DataConfig, DroidConfig
from po_jepa.runtime import cache

SPLITS_FILE = ix.SPLITS_FILE
SPLITS_SCHEMA = "droid_splits_v1"
# §7：影响划分的配置键（不含 N 和 use_failures，所以同一数据根下所有配置的划分文件逐字节相同）
SPLIT_KEYS = ("min_steps", "ood_labs", "ood_quota", "id_test_scenes", "val_scenes", "episodes_per_scene")
LISTS = ("id_test", "val", "ood_test", "train_order")
SHARD_PREFIX = "train-"
TEST_SHARDS = ("val", "id_test", "ood_test")   # §6：各自一片，内容就是划分列表


def split_config_keys(data: DataConfig) -> dict:
    return {key: list(value) if isinstance(value, tuple) else value
            for key, value in ((key, getattr(data, key)) for key in SPLIT_KEYS)}


def _check_selected(episodes: list[dict], data: DataConfig) -> list[dict]:
    """入选回合，并核对它们和配置一致（索引是用另一份配置做的就报错，不静默算出另一种划分）。"""
    selected = [e for e in episodes if e["selected"]]
    duplicates = sorted(u for u, k in Counter(e["uuid"] for e in selected).items() if k > 1)
    if duplicates:
        raise ValueError(f"selected episodes have duplicate uuids {duplicates[:5]} (02 §4 excludes them)")
    for e in selected:
        if e["reasons"]:
            raise ValueError(f"{e['uuid']}: selected but has reasons {e['reasons']}")
        if e["group"] != ("ood" if e["lab"] in data.ood_labs else "id"):
            raise ValueError(f"{e['uuid']}: group {e['group']!r} does not match data.ood_labs {list(data.ood_labs)}")
        if e["steps"] < data.min_steps:
            raise ValueError(f"{e['uuid']}: selected with {e['steps']} steps < data.min_steps {data.min_steps}")
        if e["group"] == "id" and e["scene_id"] < 0:
            raise ValueError(f"{e['uuid']}: selected in-distribution episode without scene_id")
        if e["outcome"] not in ix.OUTCOMES:
            raise ValueError(f"{e['uuid']}: outcome {e['outcome']!r}")
    return selected


def compute_splits(episodes: list[dict], data: DataConfig, episodes_sha256: str) -> dict:
    """02 §5：episodes.jsonl 的全部行（含未入选的）→ splits.json 的内容。"""
    def h(kind: str, key: str) -> str:
        return ix.salted_hash(data.split_salt, kind, key)

    selected = _check_selected(episodes, data)
    # 第 1 步：场景键 (lab, scene_id)；同一个 scene_id 出现在不止一个实验室（入选回合，含留出实验室）的不能当验证或测试场景
    labs_of = defaultdict(set)
    for e in selected:
        if e["scene_id"] >= 0:
            labs_of[e["scene_id"]].add(e["lab"])
    excluded = sorted(s for s, labs in labs_of.items() if len(labs) > 1)
    # 第 2 步：候选场景
    in_scene = defaultdict(list)
    for e in selected:
        if e["group"] == "id":
            in_scene[(e["lab"], e["scene_id"])].append(e)
    blocked = set(excluded)
    candidates = sorted((key for key, rows in in_scene.items()
                         if len(rows) >= data.episodes_per_scene and key[1] not in blocked),
                        key=lambda key: (h("scene", f"{key[0]}/{key[1]}"), key))
    needed = data.id_test_scenes + data.val_scenes
    if len(candidates) < needed:
        raise ValueError(f"only {len(candidates)} candidate scenes, need id_test_scenes + val_scenes = {needed}")
    test_scenes, val_scenes = candidates[:data.id_test_scenes], candidates[data.id_test_scenes:needed]

    # 第 3、4 步：每个场景按 h("episode", uuid) 取前 episodes_per_scene 条；列表顺序是“场景顺序，场景内按哈希”
    def pick(scenes):
        return [e["uuid"] for key in scenes
                for e in sorted(in_scene[key], key=lambda e: (h("episode", e["uuid"]), e["uuid"]))[:data.episodes_per_scene]]

    # 第 5 步：按 ood_labs 的顺序，每个实验室按 h("ood", uuid) 取前 ood_quota 条
    ood_test = []
    for lab, quota in zip(data.ood_labs, data.ood_quota):
        pool = sorted((e for e in selected if e["lab"] == lab), key=lambda e: (h("ood", e["uuid"]), e["uuid"]))
        if len(pool) < quota:
            raise ValueError(f"held-out lab {lab} has {len(pool)} selected episodes, ood_quota needs {quota}")
        ood_test.extend(e["uuid"] for e in pool[:quota])

    # 第 6 步：同分布实验室的入选回合，去掉测试和验证场景里的全部回合，按 h("train", uuid) 排序（完整训练顺序）
    held = set(test_scenes) | set(val_scenes)
    train = sorted((e for e in selected if e["group"] == "id" and (e["lab"], e["scene_id"]) not in held),
                   key=lambda e: (h("train", e["uuid"]), e["uuid"]))
    splits = {
        "schema": SPLITS_SCHEMA,
        "salt": data.split_salt,
        "episodes_sha256": episodes_sha256,
        "config_keys": split_config_keys(data),
        "excluded_scene_ids": excluded,
        "id_test_scenes": [list(key) for key in test_scenes],
        "val_scenes": [list(key) for key in val_scenes],
        "id_test": pick(test_scenes),
        "val": pick(val_scenes),
        "ood_test": ood_test,
        "train_order": [e["uuid"] for e in train],
        "train_failures": [e["uuid"] for e in train if e["outcome"] == "failure"],
    }
    splits["counts"] = {"selected": len(selected), "candidate_scenes": len(candidates),
                        **{name: len(splits[name]) for name in (*LISTS, "train_failures")}}
    return splits


def write_splits(directory: Path, splits: dict) -> None:
    """先写 `splits.json.partial` 再换名（和 index 的三个文件一样，重跑时覆盖）；格式是 `cache._json_bytes`。"""
    path = Path(directory)/SPLITS_FILE
    temp = path.with_name(path.name + ".partial")
    temp.write_bytes(cache._json_bytes(splits))
    os.replace(temp, path)


def read_episodes(directory: Path) -> list[dict]:
    path = Path(directory)/ix.EPISODES_FILE
    return [ix._loads(line.encode("utf-8"), f"{path}:{i + 1}")
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines())]


def load_splits(data_root: Path | str, config: DroidConfig) -> dict:
    """读 `index/splits.json`：先核索引清单和四个文件，再核 schema、salt、划分配置键都和 `config` 一致、
    `episodes_sha256` 等于清单里 episodes.jsonl 的 SHA、`counts` 和列表长度一致。冒烟数据根（schema
    `droid_splits_smoke_v1`，14 §4）和任何别的 schema 都拒绝。"""
    directory = ix.index_dir(data_root)
    manifest = ix.read_manifest(directory)
    ix.check_files(directory, manifest)
    splits = ix._loads((directory/SPLITS_FILE).read_bytes(), str(directory/SPLITS_FILE))
    if not isinstance(splits, dict) or splits.get("schema") != SPLITS_SCHEMA:
        raise ValueError(f"{directory/SPLITS_FILE}: schema is not {SPLITS_SCHEMA!r}")
    if splits.get("salt") != config.data.split_salt:
        raise ValueError(f"{directory/SPLITS_FILE}: salt {splits.get('salt')!r} != data.split_salt")
    if splits.get("config_keys") != split_config_keys(config.data):
        raise ValueError(f"{directory/SPLITS_FILE}: config_keys {splits.get('config_keys')} "
                         f"!= this config's {split_config_keys(config.data)}")
    if splits.get("episodes_sha256") != manifest["files"][ix.EPISODES_FILE]["sha256"]:
        raise ValueError(f"{directory/SPLITS_FILE}: episodes_sha256 is not the SHA of {ix.EPISODES_FILE}")
    counts = splits.get("counts", {})
    for name in (*LISTS, "train_failures"):
        if counts.get(name) != len(splits.get(name, ())):
            raise ValueError(f"{directory/SPLITS_FILE}: counts.{name} != len({name})")
    if not set(splits["train_failures"]) <= set(splits["train_order"]):
        raise ValueError(f"{directory/SPLITS_FILE}: train_failures is not a subset of train_order")
    return splits


def train_order(splits: dict, data: DataConfig) -> list[str]:
    """本配置的训练顺序：完整训练顺序；`use_failures: false` 时去掉失败回合，其余保持相对顺序（02 §5 第 6 步）。"""
    if data.use_failures:
        return list(splits["train_order"])
    failures = set(splits["train_failures"])
    return [u for u in splits["train_order"] if u not in failures]


def shard_name(k: int) -> str:
    return f"{SHARD_PREFIX}{k:03d}"


def shard_plan(splits: dict, data: DataConfig) -> dict[str, list[str]]:
    """02 §6：片名 → 本配置在这片里用到的回合（按训练顺序）。

    训练片 k 是完整训练顺序的第 shard_size·k 到 shard_size·(k+1)−1 条，片名和片的范围只由完整训练顺序决定，和 N、
    use_failures 都无关；本配置用到的是 `train_order(splits, data)` 的前 N 条，每片只列其中落在这片里的回合
    （use_failures: true 时就是前 ⌈N/shard_size⌉ 片、最后一片只到第 N 条）。可用的训练回合不到 N 条就报错。
    之后是 `val`、`id_test`、`ood_test` 三片。
    """
    wanted = train_order(splits, data)[:data.train_episodes]
    if len(wanted) < data.train_episodes:
        raise ValueError(f"data.train_episodes = {data.train_episodes} but this config's training order has only "
                         f"{len(wanted)} episodes (use_failures = {data.use_failures})")
    keep = set(wanted)
    order = splits["train_order"]
    position = {u: i for i, u in enumerate(order)}
    plan = {}
    for k in range(position[wanted[-1]]//data.shard_size + 1):
        plan[shard_name(k)] = [u for u in order[k*data.shard_size:(k + 1)*data.shard_size] if u in keep]
    for name in TEST_SHARDS:
        plan[name] = list(splits[name])
    return plan


def feature_rows(steps: int, frames_per_interval: int = 15) -> int:
    """一条回合的特征行数 ⌊(steps − 2)/frames_per_interval⌋ + 1（02 §5 验收表）。"""
    return (steps - 2)//frames_per_interval + 1


def list_digest(uuids: list[str]) -> str:
    """02 §5 验收表的列表摘要：uuid 用 `\\n` 连接后的 SHA-256 前 16 位。"""
    return hashlib.sha256("\n".join(uuids).encode("utf-8")).hexdigest()[:16]


def summary(splits: dict, episodes: list[dict], frames_per_interval: int = 15) -> dict:
    """02 §5 验收表的各项（由划分文件和 episodes 算，供真实数据核对和交接报告用）。"""
    by_uuid = {e["uuid"]: e for e in episodes if e["selected"]}
    first20k = splits["train_order"][:20000]

    def outcomes(uuids):
        c = Counter(by_uuid[u]["outcome"] for u in uuids)
        return [c["success"], c["failure"]]

    def rows(uuids):
        return sum(feature_rows(by_uuid[u]["steps"], frames_per_interval) for u in uuids)

    def long_enough(uuids, d):
        return sum(feature_rows(by_uuid[u]["steps"], frames_per_interval) >= d + 1 for u in uuids)

    return {
        "candidate_scenes": splits["counts"]["candidate_scenes"],
        "first_test_scene": "/".join(map(str, splits["id_test_scenes"][0])),
        "first_val_scene": "/".join(map(str, splits["val_scenes"][0])),
        "first_id_test": splits["id_test"][0],
        "first_ood_test": splits["ood_test"][0],
        "train_first": splits["train_order"][0],
        "train_20000th": splits["train_order"][19999] if len(splits["train_order"]) >= 20000 else None,
        "excluded_scene_ids": splits["excluded_scene_ids"],
        "removed_from_train": sum(e["group"] == "id" for e in by_uuid.values()) - len(splits["train_order"]),
        "success_failure": {"id_test": outcomes(splits["id_test"]), "val": outcomes(splits["val"]),
                            "ood_test": outcomes(splits["ood_test"]), "train_first_20000": outcomes(first20k)},
        "id_test_labs": dict(sorted(Counter(by_uuid[u]["lab"] for u in splits["id_test"]).items())),
        "digests": {name: list_digest(splits[name]) for name in LISTS},
        "feature_rows": {"train_first_20000": rows(first20k), "train_first_500": rows(splits["train_order"][:500]),
                         "id_test": rows(splits["id_test"]), "ood_test": rows(splits["ood_test"]),
                         "val": rows(splits["val"])},
        "long_horizon": {name: [long_enough(splits[name], d) for d in (24, 32)] for name in ("id_test", "ood_test")},
        "counts": splits["counts"],
    }

