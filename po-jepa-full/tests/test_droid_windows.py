"""S05：DROID 冻结的验证和测试窗口（docs/design/06-droid-windows-and-batches.md §3；验收 14 §3 S05 最后一条）。

全用合成的回合表（行数、是否通过准备检查），按正式配置（每个时长 4 个窗口、附加回合 200 条、规划候选 64 个）生成，
再用 hashlib 独立重算排序核对每条规则。
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from po_jepa.droid import windows as wn
from po_jepa.droid.config import load_config
from po_jepa.runtime import cache
from test_droid_index import CONFIGS

CFG = load_config(CONFIGS/"pilot.yaml")
SALT = CFG.data.window_salt
TEST_DURATIONS = [1, 2, 3, 4, 8, 12, 16, 24, 32]


def H(*parts) -> str:
    return hashlib.sha256("|".join([SALT, *map(str, parts)]).encode("utf-8")).hexdigest()


def table(n: int, seed: int, failed=()):
    rng = np.random.default_rng(seed)
    uuids = [f"LAB+{seed:04d}{i:04d}+2023-07-07-09h-42m-23s" for i in range(n)]
    rows = {u: int(rng.integers(17, 60)) for u in uuids}
    passed = {u: i not in set(failed) for i, u in enumerate(uuids)}
    shas = {u: hashlib.sha256(u.encode()).hexdigest() for u in uuids}
    return uuids, rows, passed, shas


FAILED = (3, 50, 199, 210)   # 3 条在前 200 条里


@pytest.fixture(scope="module")
def test_case():
    uuids, rows, passed, shas = table(230, 1, FAILED)
    content = wn.frozen_windows("id_test", uuids, rows=rows, passed=passed, prepared_sha256=shas,
                                splits_sha256="s"*64, config=CFG)
    return uuids, rows, passed, shas, content


def test_hw_hand_value():
    assert wn.hw("salt", "val", "u", 4, 7) == hashlib.sha256(b"salt|val|u|4|7").hexdigest()
    assert wn.evaluation_durations(CFG) == TEST_DURATIONS
    assert wn.frame_rate_duration(CFG) == 4


def test_test_windows_follow_the_rule(test_case):
    uuids, rows, passed, shas, c = test_case
    expected = []
    for u in uuids:
        if not passed[u]:
            continue
        for d in TEST_DURATIONS:
            starts = sorted(range(rows[u] - d), key=lambda s: H("id_test", u, d, s))[:4]
            expected += [{"id": f"{u}|d{d}|s{s}", "uuid": u, "duration": d, "start": s} for s in starts]
    assert c["windows"] == expected
    assert c["counts"]["windows"] == len(expected) == sum(min(4, max(0, rows[u] - d)) for u in uuids if passed[u]
                                                          for d in TEST_DURATIONS)
    for w in c["windows"]:
        assert 0 <= w["start"] <= rows[w["uuid"]] - 1 - w["duration"]
    assert c["counts"]["windows_by_duration"] == {str(d): sum(w["duration"] == d for w in expected) for d in TEST_DURATIONS}
    assert c["counts"]["episodes"] == 230 and c["counts"]["passed"] == 226 and c["counts"]["failed"] == 4
    assert {w["uuid"] for w in c["windows"]}.isdisjoint({uuids[i] for i in FAILED})
    assert c["prepared_sha256"] == shas   # 含没通过的回合
    assert {k: c[k] for k in ("schema", "split", "salt", "splits_sha256")} == {
        "schema": "droid_windows_v1", "split": "id_test", "salt": SALT, "splits_sha256": "s"*64}


def test_long_durations_need_enough_rows():
    uuids, rows, passed, shas = table(70, 2)
    rows[uuids[0]], rows[uuids[1]] = 25, 33   # F − 1 = 24：d = 32 没有，d = 24 只有 1 个；F − 1 = 32：d = 32 有 1 个
    c = wn.frozen_windows("ood_test", uuids, rows=rows, passed=passed, prepared_sha256=shas, splits_sha256="s", config=CFG)
    by = lambda u, d: [w["start"] for w in c["windows"] if w["uuid"] == u and w["duration"] == d]
    assert by(uuids[0], 32) == [] and by(uuids[0], 24) == [0]
    assert by(uuids[1], 32) == [0] and len(by(uuids[1], 24)) == 4


def test_extra_episodes_skip_failed_without_refill(test_case):
    uuids, rows, passed, shas, c = test_case
    assert c["extra_episodes"] == [u for u in uuids[:200] if passed[u]]
    assert len(c["extra_episodes"]) == 197 == c["counts"]["extra_episodes"]


def test_donors_come_from_other_episodes_on_the_ring(test_case):
    uuids, rows, passed, shas, c = test_case
    windows = {w["id"]: w for w in c["windows"]}
    assert list(c["donors"]) == [w["id"] for w in c["windows"]] and c["counts"]["donors"] == len(windows)
    for d in TEST_DURATIONS:
        ring = sorted((w for w in c["windows"] if w["duration"] == d), key=lambda w: H("donor", "id_test", w["id"]))
        for i, w in enumerate(ring):
            k = next(j for j in range(1, len(ring)) if ring[(i + j) % len(ring)]["uuid"] != w["uuid"])
            donor = windows[c["donors"][w["id"]]]
            assert donor is ring[(i + k) % len(ring)] or donor == ring[(i + k) % len(ring)]
            assert donor["uuid"] != w["uuid"] and donor["duration"] == d


def test_planning_rank_has_exactly_63_distractors_from_other_episodes(test_case):
    uuids, rows, passed, shas, c = test_case
    extra = set(c["extra_episodes"])
    planning = [w for w in c["windows"] if w["uuid"] in extra and w["duration"] in (1, 4)]
    assert list(c["planning"]["rank"]) == [w["id"] for w in planning] and c["counts"]["planning_rank"] == len(planning)
    by_id = {w["id"]: w for w in planning}
    for w in planning:
        ids = c["planning"]["rank"][w["id"]]
        assert len(ids) == 63 == len(set(ids))
        others = [o for o in planning if o["duration"] == w["duration"] and o["uuid"] != w["uuid"]]
        assert ids == [o["id"] for o in sorted(others, key=lambda o: H("distractor", "id_test", w["id"], o["id"]))[:63]]
        assert all(by_id[i]["uuid"] != w["uuid"] and by_id[i]["duration"] == w["duration"] for i in ids)


def test_cem_one_window_per_episode_and_duration(test_case):
    uuids, rows, passed, shas, c = test_case
    cem = c["planning"]["cem"]
    windows = {w["id"]: w for w in c["windows"]}
    keys = [(windows[i]["uuid"], windows[i]["duration"]) for i in cem]
    assert keys == [(u, d) for u in c["extra_episodes"] for d in (1, 4)] and len(set(keys)) == len(keys)
    for i in cem:
        w = windows[i]
        first = sorted(range(rows[w["uuid"]] - w["duration"]), key=lambda s: H("id_test", w["uuid"], w["duration"], s))[0]
        assert w["start"] == first
    assert c["counts"]["cem"] == len(cem) == 2*197


def test_frame_rate_windows(test_case):
    uuids, rows, passed, shas, c = test_case
    extra = set(c["extra_episodes"])
    d4 = [w for w in c["windows"] if w["uuid"] in extra and w["duration"] == 4]
    assert c["frame_rate"] == [{"id": f"{w['uuid']}|fr|f{15*w['start']}", "uuid": w["uuid"], "start_frame": 15*w["start"]}
                               for w in d4]
    assert c["counts"]["frame_rate"] == len(d4) == 4*197
    for f in c["frame_rate"]:
        assert f["start_frame"] + 60 <= 15*(rows[f["uuid"]] - 1)   # 4 个周期之后仍在最后一个取帧位置之内


def test_same_inputs_same_bytes(test_case):
    uuids, rows, passed, shas, c = test_case
    again = wn.frozen_windows("id_test", list(uuids), rows=dict(rows), passed=dict(passed), prepared_sha256=dict(shas),
                              splits_sha256="s"*64, config=CFG)
    assert cache._json_bytes(again) == cache._json_bytes(c)


def test_split_name_and_salt_change_the_picks():
    uuids, rows, passed, shas = table(70, 3)
    a = wn.frozen_windows("id_test", uuids, rows=rows, passed=passed, prepared_sha256=shas, splits_sha256="s", config=CFG)
    b = wn.frozen_windows("ood_test", uuids, rows=rows, passed=passed, prepared_sha256=shas, splits_sha256="s", config=CFG)
    other = dataclasses.replace(CFG, data=dataclasses.replace(CFG.data, window_salt="other-salt"))
    s = wn.frozen_windows("id_test", uuids, rows=rows, passed=passed, prepared_sha256=shas, splits_sha256="s", config=other)
    starts = lambda c: [w["start"] for w in c["windows"]]
    assert starts(a) != starts(b) and starts(a) != starts(s)


def test_val_one_window_per_training_duration_and_only_windows_and_counts():
    uuids, rows, passed, shas = table(300, 4, failed=(7,))
    c = wn.frozen_windows("val", uuids, rows=rows, passed=passed, prepared_sha256=shas, splits_sha256="v", config=CFG)
    assert set(c) == {"schema", "split", "salt", "splits_sha256", "prepared_sha256", "windows", "counts"}
    expected = [{"id": f"{u}|d{d}|s{s}", "uuid": u, "duration": d, "start": s}
                for u in uuids if passed[u] for d in (1, 2, 4, 8)
                for s in sorted(range(rows[u] - d), key=lambda s: H("val", u, d, s))[:1]]
    assert c["windows"] == expected and len(expected) == 299*4
    assert c["counts"] == {"episodes": 300, "passed": 299, "failed": 1, "windows": 1196,
                           "windows_by_duration": {"1": 299, "2": 299, "4": 299, "8": 299}}


def test_errors():
    uuids, rows, passed, shas = table(70, 5)
    kw = dict(rows=rows, passed=passed, prepared_sha256=shas, splits_sha256="s", config=CFG)
    with pytest.raises(ValueError, match="split must be one of"):
        wn.frozen_windows("train", uuids, **kw)
    with pytest.raises(ValueError, match="duplicate uuids"):
        wn.frozen_windows("id_test", uuids + uuids[:1], **kw)
    with pytest.raises(ValueError, match="no rows for"):
        wn.frozen_windows("id_test", uuids, **dict(kw, rows={u: rows[u] for u in uuids[1:]}))
    with pytest.raises(ValueError, match="passed but has rows None"):
        wn.frozen_windows("id_test", uuids, **dict(kw, rows=dict(rows, **{uuids[0]: None})))
    # 没通过的回合可以没有行数（h5 结构不对时准备 JSON 的 rows 是 null）
    wn.frozen_windows("id_test", uuids, **dict(kw, rows=dict(rows, **{uuids[0]: None}), passed=dict(passed, **{uuids[0]: False})))
    with pytest.raises(ValueError, match="needs 63"):
        wn.frozen_windows("id_test", uuids[:16], **kw)   # 16 条 × 4 个窗口：别的回合只有 60 个
    with pytest.raises(ValueError, match="has no donor"):
        wn.frozen_windows("id_test", uuids[:1], **kw)
