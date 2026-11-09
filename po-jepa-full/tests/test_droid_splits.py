"""S03：DROID 划分（docs/design/02-droid-index-and-splits.md §5–§7；验收 14 §3 S03）。

合成回合（episodes.jsonl 的行）上核 02 §5 的每一步和“性质”一段；真实元数据上核 02 §5 的验收表
（`test_real_splits_acceptance_table`，没有真实索引时跳过）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
from pathlib import Path
from unittest import mock

import pytest

from po_jepa.droid import config as config_module
from po_jepa.droid import index as ix
from po_jepa.droid import splits as sp
from po_jepa.droid.config import load_config
from test_droid_index import SEVEN, _config, _fill, bucket  # noqa: F401 — bucket 是假存储桶 fixture

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/"configs"/"droid"
SALT = "po-jepa-droid-1.0.1-split-v1"
SMALL = dict(id_test_scenes=3, val_scenes=2, episodes_per_scene=2, ood_quota=(2, 2, 1), shard_size=4, train_episodes=6)


def h(kind, key):
    return hashlib.sha256(f"{SALT}|{kind}|{key}".encode("utf-8")).hexdigest()


def cfg(**data):
    base = load_config(CONFIGS/"pilot.yaml")
    fields = dict(SMALL, **data)
    with mock.patch.object(config_module, "OOD_TEST_EPISODES", sum(fields["ood_quota"])):
        return dataclasses.replace(base, data=dataclasses.replace(base.data, **fields))


def ep(lab, scene_id, i, *, outcome="success", steps=300, selected=True, reasons=None, group=None):
    ood = lab in ("PennPAL", "REAL", "RPL")
    reasons = [] if selected else (reasons or ["short"])
    return {"uuid": f"{lab}+{scene_id}+{i:04d}", "lab": lab, "scene_id": scene_id, "steps": steps, "outcome": outcome,
            "group": group or ("ood" if ood else "id"), "selected": selected, "reasons": reasons,
            "folder": f"{lab}/{outcome}/x/{scene_id}-{i}", "success_flag": outcome == "success"}


def world():
    """10 个同分布场景（各 2–4 条入选回合，有失败回合）、一个 4 条的场景只有 1 条入选、两个跨实验室的场景号、
    三个留出实验室，加若干未入选回合。"""
    rows = []
    labs = ("AUTOLab", "IPRL", "TRI", "CLVR")
    for s in range(10):
        lab = labs[s % 4]
        for i in range(2 + s % 3):
            rows.append(ep(lab, 1000 + s, i, outcome="failure" if (s + i) % 3 == 0 else "success"))
    rows += [ep("RAIL", 2000, i, selected=i == 0) for i in range(4)]               # 只有 1 条入选：不是候选
    rows += [ep("ILIAD", 3000, i) for i in range(3)] + [ep("WEIRD", 3000, i) for i in range(2)]   # 跨实验室
    rows += [ep("RAD", 4000, i) for i in range(3)] + [ep("PennPAL", 4000, i) for i in range(1)]  # 同分布 + 留出实验室
    for lab, n in (("PennPAL", 3), ("REAL", 4), ("RPL", 2)):
        rows += [ep(lab, -1, 100 + i, outcome="failure" if i == 1 else "success") for i in range(n)]
    rows += [ep("GuptaLab", 5000, i, selected=False, reasons=["duplicate_uuid"]) for i in range(3)]
    return rows


def compute(rows=None, **data):
    return sp.compute_splits(world() if rows is None else rows, cfg(**data).data, "e"*64)


# ---------------------------------------------------------------- §5 每一步

def test_each_step_matches_hand_computation():
    rows = world()
    splits = compute(rows)
    selected = [e for e in rows if e["selected"]]
    by_scene = {}
    for e in selected:
        if e["group"] == "id":
            by_scene.setdefault((e["lab"], e["scene_id"]), []).append(e["uuid"])
    candidates = sorted((k for k, v in by_scene.items() if len(v) >= 2 and k[1] not in (3000, 4000)),
                        key=lambda k: h("scene", f"{k[0]}/{k[1]}"))
    assert len(candidates) == 10 + 0   # RAIL/2000 只有 1 条入选；ILIAD/3000、WEIRD/3000、RAD/4000 被排除
    assert splits["id_test_scenes"] == [list(k) for k in candidates[:3]]
    assert splits["val_scenes"] == [list(k) for k in candidates[3:5]]
    pick = [u for k in candidates[:3] for u in sorted(by_scene[k], key=lambda u: h("episode", u))[:2]]
    assert splits["id_test"] == pick
    assert splits["val"] == [u for k in candidates[3:5] for u in sorted(by_scene[k], key=lambda u: h("episode", u))[:2]]
    ood = []
    for lab, quota in (("PennPAL", 2), ("REAL", 2), ("RPL", 1)):
        ood += sorted((e["uuid"] for e in selected if e["lab"] == lab), key=lambda u: h("ood", u))[:quota]
    assert splits["ood_test"] == ood
    held = set(candidates[:5])
    train = sorted((e["uuid"] for e in selected if e["group"] == "id" and (e["lab"], e["scene_id"]) not in held),
                   key=lambda u: h("train", u))
    assert splits["train_order"] == train
    failures = {e["uuid"] for e in rows if e["outcome"] == "failure"}
    assert splits["train_failures"] == [u for u in train if u in failures] and splits["train_failures"]
    assert splits["excluded_scene_ids"] == [3000, 4000]
    assert splits["counts"] == {"selected": len(selected), "candidate_scenes": 10, "id_test": 6, "val": 4,
                                "ood_test": 5, "train_order": len(train), "train_failures": len(splits["train_failures"])}
    assert splits["schema"] == "droid_splits_v1" and splits["salt"] == SALT and splits["episodes_sha256"] == "e"*64


def test_ood_test_follows_ood_labs_order():
    rows = world()
    splits = compute(rows, ood_labs=("RPL", "PennPAL", "REAL"), ood_quota=(1, 2, 2))
    lab = {e["uuid"]: e["lab"] for e in rows}
    assert [lab[u] for u in splits["ood_test"]] == ["RPL", "PennPAL", "PennPAL", "REAL", "REAL"]
    rpl = sorted((e["uuid"] for e in rows if e["lab"] == "RPL"), key=lambda u: h("ood", u))
    assert splits["ood_test"][0] == rpl[0]


def test_config_keys_exclude_n_and_use_failures():
    assert compute()["config_keys"] == {"min_steps": 242, "ood_labs": ["PennPAL", "REAL", "RPL"], "ood_quota": [2, 2, 1],
                                        "id_test_scenes": 3, "val_scenes": 2, "episodes_per_scene": 2}


def test_scene_never_crosses_train_val_test():
    rows = world()
    splits = compute(rows)
    scene = {e["uuid"]: (e["lab"], e["scene_id"]) for e in rows}
    sets = {name: {scene[u] for u in splits[name]} for name in ("train_order", "val", "id_test")}
    assert not sets["train_order"] & sets["val"] and not sets["train_order"] & sets["id_test"]
    assert not sets["val"] & sets["id_test"]
    assert sets["id_test"] == {tuple(k) for k in splits["id_test_scenes"]}
    assert sets["val"] == {tuple(k) for k in splits["val_scenes"]}
    # 测试和验证场景里没被抽中的回合也不进训练（02 §5 第 6 步）
    held = {tuple(k) for k in splits["id_test_scenes"] + splits["val_scenes"]}
    unpicked = [e["uuid"] for e in rows if e["selected"] and scene[e["uuid"]] in held
                and e["uuid"] not in splits["id_test"] + splits["val"]]
    assert unpicked and not set(unpicked) & set(splits["train_order"])
    # 留出实验室的回合只在留出实验室测试里
    ood = {e["uuid"] for e in rows if e["group"] == "ood"}
    assert set(splits["ood_test"]) <= ood and not ood & set(splits["train_order"] + splits["val"] + splits["id_test"])


def test_cross_lab_scene_ids_excluded_but_stay_in_training():
    rows = world()
    splits = compute(rows)
    for lab, scene in (("ILIAD", 3000), ("WEIRD", 3000), ("RAD", 4000)):
        uuids = [e["uuid"] for e in rows if e["lab"] == lab and e["scene_id"] == scene]
        assert set(uuids) <= set(splits["train_order"])
        assert [lab, scene] not in splits["id_test_scenes"] + splits["val_scenes"]
    # 只在一个实验室出现时就是候选
    alone = [e for e in rows if not (e["lab"] == "WEIRD" and e["scene_id"] == 3000)]
    assert 3000 not in compute(alone)["excluded_scene_ids"] and compute(alone)["counts"]["candidate_scenes"] == 11
    # 未入选回合不算“出现”
    unselected = [dict(e, selected=False, reasons=["short"]) if e["lab"] == "WEIRD" else e for e in rows]
    assert compute(unselected)["excluded_scene_ids"] == [4000]


def test_candidate_needs_episodes_per_scene_selected():
    rows = world()
    assert compute(rows)["counts"]["candidate_scenes"] == 10
    assert compute(rows, episodes_per_scene=3, id_test_scenes=1, val_scenes=1)["counts"]["candidate_scenes"] == 6
    rail = [dict(e, selected=True, reasons=[]) if e["lab"] == "RAIL" else e for e in rows]
    assert compute(rail)["counts"]["candidate_scenes"] == 11


def test_errors_when_too_few_scenes_or_ood_episodes():
    with pytest.raises(ValueError, match="only 10 candidate scenes, need id_test_scenes \\+ val_scenes = 11"):
        compute(id_test_scenes=9, val_scenes=2)
    compute(id_test_scenes=8, val_scenes=2)
    with pytest.raises(ValueError, match="held-out lab RPL has 2 selected episodes, ood_quota needs 3"):
        compute(ood_quota=(1, 1, 3))
    assert len(compute(ood_quota=(3, 4, 2))["ood_test"]) == 9


@pytest.mark.parametrize("edit, message", [
    (lambda rows: rows.append(dict(rows[0])), "duplicate uuids"),
    (lambda rows: rows[0].update(reasons=["short"]), "selected but has reasons"),
    (lambda rows: rows[0].update(group="ood"), "does not match data.ood_labs"),
    (lambda rows: rows[0].update(steps=241), "241 steps < data.min_steps 242"),
    (lambda rows: rows[0].update(scene_id=-1), "without scene_id"),
    (lambda rows: rows[0].update(outcome="unknown"), "outcome"),
])
def test_selected_rows_must_agree_with_config(edit, message):
    rows = world()
    edit(rows)
    with pytest.raises(ValueError, match=message):
        compute(rows)


# ---------------------------------------------------------------- 性质（02 §5 最后一段）

def test_same_input_same_bytes_and_input_order_irrelevant(tmp_path):
    rows = world()
    shuffled = list(rows)
    random.Random(0).shuffle(shuffled)
    a, b = tmp_path/"a", tmp_path/"b"
    a.mkdir(), b.mkdir()
    sp.write_splits(a, compute(rows))
    sp.write_splits(b, compute(shuffled))
    assert (a/"splits.json").read_bytes() == (b/"splits.json").read_bytes()
    assert (a/"splits.json").read_bytes() == ix.cache._json_bytes(compute(rows))
    sp.write_splits(a, compute(rows))   # 重跑覆盖，不报错
    assert not (a/"splits.json.partial").exists()


def test_splits_do_not_depend_on_n_or_use_failures():
    first = ix.cache._json_bytes(compute(train_episodes=1))
    for n in (2, 6, 9):
        for use_failures in (True, False):
            assert ix.cache._json_bytes(compute(train_episodes=n, use_failures=use_failures)) == first


def test_training_set_for_n_is_prefix_of_larger_n():
    splits = compute()
    total = len(splits["train_order"])
    for use_failures in (True, False):
        flat = {}
        for n in range(1, total + 1):
            data = cfg(train_episodes=n, use_failures=use_failures).data
            available = len(sp.train_order(splits, data))
            if n > available:
                with pytest.raises(ValueError, match=f"has only {available} episodes"):
                    sp.shard_plan(splits, data)
                continue
            plan = sp.shard_plan(splits, data)
            flat[n] = [u for name, uuids in plan.items() if name.startswith("train-") for u in uuids]
            assert flat[n] == sp.train_order(splits, data)[:n]
            assert {k: plan[k] for k in ("val", "id_test", "ood_test")} == {k: splits[k] for k in ("val", "id_test", "ood_test")}
        for n in flat:
            for m in flat:
                if n < m:
                    assert flat[m][:n] == flat[n]


def test_use_failures_false_drops_failures_keeping_order():
    splits = compute()
    keep = sp.train_order(splits, cfg(use_failures=False).data)
    failures = set(splits["train_failures"])
    assert keep == [u for u in splits["train_order"] if u not in failures] and len(keep) < len(splits["train_order"])
    assert sp.train_order(splits, cfg().data) == splits["train_order"]


# ---------------------------------------------------------------- §6 片

def test_shard_plan_names_and_ranges():
    splits = compute()
    order = splits["train_order"]
    assert len(order) >= 10
    plan = sp.shard_plan(splits, cfg(train_episodes=6).data)   # shard_size 4：train-000 全部 4 条，train-001 只到第 6 条
    assert list(plan) == ["train-000", "train-001", "val", "id_test", "ood_test"]
    assert plan["train-000"] == order[:4] and plan["train-001"] == order[4:6]
    assert list(sp.shard_plan(splits, cfg(train_episodes=8).data))[:2] == ["train-000", "train-001"]
    assert sp.shard_plan(splits, cfg(train_episodes=9).data)["train-002"] == order[8:9]
    # use_failures: false：片的范围仍按完整训练顺序，每片只列本配置用到的回合
    data = cfg(train_episodes=5, use_failures=False).data
    plan = sp.shard_plan(splits, data)
    wanted = sp.train_order(splits, data)[:5]
    for name, uuids in plan.items():
        if name.startswith("train-"):
            k = int(name[len("train-"):])
            assert uuids == [u for u in order[4*k:4*k + 4] if u in wanted]
    assert sp.shard_name(12) == "train-012"


# ---------------------------------------------------------------- §7 文件、清单、读取

def test_run_index_writes_splits_before_manifest(bucket, tmp_path):  # noqa: F811
    _fill(bucket, SEVEN)
    config = _config(bucket)
    manifest = ix.run_index(config, tmp_path)
    directory = tmp_path/"index"
    episodes = sp.read_episodes(directory)
    expected = sp.compute_splits(episodes, config.data, ix.cache.file_sha(directory/"episodes.jsonl"))
    assert (directory/"splits.json").read_bytes() == ix.cache._json_bytes(expected)
    assert manifest["files"]["splits.json"] == ix._file_entry(directory/"splits.json")
    assert sp.load_splits(tmp_path, config) == expected
    assert expected["excluded_scene_ids"] == [] and expected["id_test_scenes"] in ([["AUTOLab", 5207831207]], [["TRI", 4242]])


def test_run_index_fails_without_manifest_when_splits_fail(bucket, tmp_path):  # noqa: F811
    _fill(bucket, SEVEN)
    with pytest.raises(ValueError, match="candidate scenes"):
        ix.run_index(_config(bucket, val_scenes=5), tmp_path)
    assert not (tmp_path/"index"/"MANIFEST.json").exists()


def _index_dir(tmp_path, splits_value, episodes_text="{}\n"):
    """手搭一个索引目录：五个文件和一份自洽的清单。"""
    directory = tmp_path/"index"
    directory.mkdir(exist_ok=True)
    for name in ("listing.jsonl", "metadata.jsonl", "raw_objects.jsonl"):
        (directory/name).write_text("", encoding="utf-8")
    (directory/"episodes.jsonl").write_text(episodes_text, encoding="utf-8")
    if callable(splits_value):
        splits_value = splits_value(ix.cache.file_sha(directory/"episodes.jsonl"))
    (directory/"splits.json").write_bytes(ix.cache._json_bytes(splits_value))
    manifest = {"schema": "droid_index_v1", "files": {n: ix._file_entry(directory/n) for n in ix.INDEX_FILES}}
    if (directory/"MANIFEST.json").exists():
        (directory/"MANIFEST.json").unlink()
    ix.cache._write_json(directory/"MANIFEST.json", manifest)
    return directory


def test_load_splits_accepts_consistent_file(tmp_path):
    good = lambda sha: dict(compute(), episodes_sha256=sha)   # noqa: E731
    _index_dir(tmp_path, good)
    assert sp.load_splits(tmp_path, cfg()) == good(ix.cache.file_sha(tmp_path/"index"/"episodes.jsonl"))
    assert sp.load_splits(tmp_path, cfg(train_episodes=9, use_failures=False))["schema"] == "droid_splits_v1"


@pytest.mark.parametrize("edit, config, message", [
    (lambda s: s.update(schema="droid_splits_smoke_v1"), {}, "schema is not 'droid_splits_v1'"),
    (lambda s: s.update(salt="other"), {}, "salt 'other' != data.split_salt"),
    (lambda s: None, {"val_scenes": 3}, "config_keys"),
    (lambda s: None, {"min_steps": 200}, "config_keys"),
    (lambda s: s.update(episodes_sha256="0"*64), {}, "episodes_sha256"),
    (lambda s: s["counts"].update(val=5), {}, "counts.val"),
    (lambda s: (s["train_failures"].append("nobody"), s["counts"].update(train_failures=len(s["train_failures"]))),
     {}, "train_failures is not a subset"),
])
def test_load_splits_rejects(tmp_path, edit, config, message):
    def build(sha):
        value = dict(compute(), episodes_sha256=sha)
        edit(value)
        return value

    _index_dir(tmp_path, build)
    with pytest.raises(ValueError, match=message):
        sp.load_splits(tmp_path, cfg(**config))


def test_load_splits_checks_index_manifest(tmp_path):
    directory = _index_dir(tmp_path, lambda sha: dict(compute(), episodes_sha256=sha))
    (directory/"splits.json").write_bytes((directory/"splits.json").read_bytes().replace(b'"val"', b'"vaI"', 1))
    with pytest.raises(ValueError, match="does not match MANIFEST.json"):
        sp.load_splits(tmp_path, cfg())


def test_feature_rows_and_list_digest_hand_values():
    assert [sp.feature_rows(s) for s in (242, 256, 257, 300)] == [17, 17, 18, 20]
    assert sp.list_digest(["a", "b"]) == hashlib.sha256(b"a\nb").hexdigest()[:16]


# ---------------------------------------------------------------- 真实元数据（02 §5 的验收表；没有真实索引时跳过）

REAL_ROOT = os.environ.get("PO_JEPA_DROID_DATA_ROOT")
ACCEPTANCE = {
    "candidate_scenes": 1038,
    "first_test_scene": "IPRL/5445595898",
    "first_val_scene": "AUTOLab/2913242782",
    "first_id_test": "IPRL+41a1536a+2023-11-08-23h-35m-17s",
    "first_ood_test": "PennPAL+acda9df3+2023-07-15-17h-18m-49s",
    "train_first": "CLVR+13759f6e+2023-05-15-22h-28m-19s",
    "train_20000th": "TRI+52ca9b6a+2024-02-06-15h-12m-51s",
    "excluded_scene_ids": [4465064685, 4823049285],
    "removed_from_train": 3524,
    "success_failure": {"id_test": [409, 91], "val": [246, 54], "ood_test": [430, 70],
                        "train_first_20000": [16600, 3400]},
    "id_test_labs": {"AUTOLab": 45, "CLVR": 5, "GuptaLab": 5, "ILIAD": 45, "IPRL": 70, "IRIS": 30, "RAD": 35,
                     "RAIL": 30, "TRI": 200, "WEIRD": 35},
    "digests": {"id_test": "2de8807ef70ace77", "val": "490b22363540ae95", "ood_test": "f3caf536bff734bf",
                "train_order": "b9dc25a09c5fe88a"},
    "feature_rows": {"train_first_20000": 696912, "train_first_500": 17931, "id_test": 18596, "ood_test": 15592,
                     "val": 9128},
    "long_horizon": {"id_test": [259, 141], "ood_test": [268, 156]},
}


@pytest.mark.skipif(not REAL_ROOT or not (Path(REAL_ROOT or ".")/"index"/"MANIFEST.json").exists(),
                    reason="set PO_JEPA_DROID_DATA_ROOT to a data root indexed from the real bucket")
def test_real_splits_acceptance_table():
    config = load_config(CONFIGS/"pilot.yaml")
    splits = sp.load_splits(REAL_ROOT, config)
    directory = ix.index_dir(REAL_ROOT)
    episodes = sp.read_episodes(directory)
    # 文件就是用当前代码从 episodes.jsonl 重算的结果（逐字节）；full 配置读到同一份
    recomputed = sp.compute_splits(episodes, config.data, ix.cache.file_sha(directory/"episodes.jsonl"))
    assert (directory/"splits.json").read_bytes() == ix.cache._json_bytes(recomputed)
    assert sp.load_splits(REAL_ROOT, load_config(CONFIGS/"full.yaml")) == splits
    summary = sp.summary(splits, episodes)
    assert summary.pop("counts") == {"selected": 32141, "candidate_scenes": 1038, "id_test": 500, "val": 300,
                                     "ood_test": 500, "train_order": 23484, "train_failures": 4000}
    assert summary == ACCEPTANCE
    assert len(sp.shard_plan(splits, load_config(CONFIGS/"full.yaml").data)) == 23
    assert list(sp.shard_plan(splits, config.data)) == ["train-000", "val", "id_test", "ood_test"]
