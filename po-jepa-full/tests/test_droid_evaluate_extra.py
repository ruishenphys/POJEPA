"""S11b：DROID 评测的原始帧率、原始分辨率、截断间隙（docs/design/10-evaluation.md §8.4、§8.5、§8.8；验收 14 §3 S11）。

合成数据根：两个测试集各 2 条回合，全部是附加回合；准备文件带全帧率的 `state_all`、`command_all`（主特征行的状态和动作
由它们按 04 §3 算出），测试附加特征 s5 / r192 / r320 是和主特征无关的随机 float16（读错文件、读错行都会对不上）。
规划的干扰项数设成 1（只有 4 条回合）。

玩具模型同 S11a：Φ(z, u, T, grid) = a^T·z + (∫u dt)·W + T·(N/256)·v。原始帧率的期望值由全帧率位姿独立按 01 §5.5 手算
（u = poses_to_diffs([s_f, s_{f+F}]) / δ，再除尺度），不走被测的控制构造代码。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math

import numpy as np
import pytest
import torch

from po_jepa.droid import batches as bt
from po_jepa.droid import evaluate as ev
from po_jepa.droid import features as ft
from po_jepa.droid import spectral_gap as sg
from po_jepa.droid import windows as wn
from po_jepa.droid.actions import droid_control_spec, poses_to_diffs
from po_jepa.droid.config import load_config
from po_jepa.droid.poses import compute_new_pose
from po_jepa.droid.prepare import PreparedEpisode, periodic_actions
from po_jepa.geometry import regular_grid
from po_jepa.losses import jepa_distance
from po_jepa.runtime import cache
from po_jepa.spectral import GraphConfig, GraphSpectrum
from test_droid_evaluate import A, LOSS, SCALE, SYNTHETIC, Identity, Toy, uid
from test_droid_index import CONFIGS

BASE = load_config(CONFIGS/"pilot.yaml")
CFG = dataclasses.replace(BASE, evaluation=dataclasses.replace(BASE.evaluation, planning_candidates=2))
SPLIT_ROWS = {"id_test": {"G": 20, "H": 22}, "ood_test": {"J": 21, "K": 19}}
OUTCOME = {"G": "success", "H": "failure", "J": "failure", "K": "success"}
E = ft.WIDTH
KINDS = {"s5": 256, "r192": 144, "r320": 400}


def _poses(rng, steps, step):
    x = np.cumsum(rng.uniform(-step, step, (steps, 7)), axis=0)
    x[:, 6] = np.clip(0.5 + x[:, 6], 0., 1.)
    return x


def _prepared(u: str, rows: int, seed: int, passed=(True, True)) -> PreparedEpisode:
    rng = np.random.default_rng(seed)
    video = 15*(rows - 1) + 1 + 7
    state_all, command_all = _poses(rng, video + 1, 0.002), _poses(rng, video + 1, 0.0025)
    index = np.arange(0, video, 15, dtype=np.int64)
    assert index.shape[0] == rows
    arrays = {"frame_index": index, "state": state_all[index], "command": command_all[index],
              "action_realized": periodic_actions(state_all[index]), "action_commanded": periodic_actions(command_all[index]),
              "state_all": state_all, "command_all": command_all}
    info = {"uuid": u, "rows": rows, "video_frames": video, "passed_realized": passed[0], "passed_commanded": passed[1]}
    return PreparedEpisode(info, hashlib.sha256(f"prep-{u}".encode()).hexdigest(), arrays)


def _write(npy, js, array, extra):
    return ft.write_features(npy, js, array, lambda sha: {
        "rows": array.shape[0], "shape": list(array.shape), "dtype": ft.STORAGE_DTYPE, "file_sha256": sha,
        "row_sha256": ft.row_sha256(array), **extra})


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return build_root(tmp_path_factory.mktemp("droid_eval_extra"))


def build_root(data):
    """合成数据根（`root` 夹具的内容；别的测试文件用另一个目录调用）。"""
    prepared, manifests, docs, extra_entries = {}, {}, {}, []
    for shard, keys in SPLIT_ROWS.items():
        entries = []
        for k, rows in keys.items():
            u = uid(k)
            p = prepared[u] = _prepared(u, rows, 300 + ord(k))
            rng = np.random.default_rng(ord(k) + 7)
            main = rng.standard_normal((rows, ft.TOKENS, E)).astype(np.float16)
            npy, js = ft.feature_paths(data, shard, u)
            rec, js_sha = _write(npy, js, main, {"schema": ft.FEATURE_SCHEMA, "uuid": u, "shard": shard,
                                                 "prepared_sha256": p.json_sha256, "encoder": SYNTHETIC})
            entries.append({"uuid": u, "rows": rows, "file_sha256": rec["file_sha256"], "feature_json_sha256": js_sha})
            files = {}
            for kind, n in KINDS.items():
                index = (list(range(0, p.info["video_frames"], 5)) if kind == "s5"
                         else [int(i) for i in p.arrays["frame_index"]])
                arr = rng.standard_normal((len(index), n, E)).astype(np.float16)
                npy, js = ft.extra_paths(data, u, kind)
                _, sha = _write(npy, js, arr, {"schema": ft.EXTRA_SCHEMA, "uuid": u, "shard": shard, "kind": kind,
                                               "frame_index": index, "prepared_sha256": p.json_sha256})
                files[kind] = {"feature_json_sha256": sha}
            extra_entries.append({"uuid": u, "shard": shard, "files": files})
        manifests[shard] = {"schema": ft.MANIFEST_SCHEMA, "shard": shard, "encoder": SYNTHETIC, "episodes": entries,
                            "config_sha256": CFG.config_sha256}
        uuids = [uid(k) for k in keys]
        docs[shard] = wn.frozen_windows(shard, uuids, rows={u: prepared[u].rows for u in uuids},
                                        passed={u: True for u in uuids},
                                        prepared_sha256={u: prepared[u].json_sha256 for u in uuids},
                                        splits_sha256="s"*64, config=CFG)
    extra = ft.ExtraFeatureReader(data, {"schema": ft.EXTRA_MANIFEST_SCHEMA, "episodes": extra_entries})
    return dict(data=data, prepared=prepared, docs=docs, reader=ft.FeatureReader(data, manifests), extra=extra,
                provenance=bt.training_provenance(CFG, manifests, "f"*64), manifests=manifests,
                outcomes={uid(k): o for k, o in OUTCOME.items()})


def spec(source="realized"):
    return droid_control_spec(SCALE, source=source, config=CFG)


def split_args(root, split, config=CFG, prepared=None, source="realized"):
    prepared = prepared or root["prepared"]
    _, kept, skipped = ev._selected(root["docs"][split], prepared, config.action.source)
    return dict(split=split, doc=root["docs"][split], prepared=prepared, outcomes=root["outcomes"], skipped=skipped,
                episodes=bt.episode_arrays(prepared, kept, config.action.source), extra_reader=root["extra"],
                control_spec=spec(source), loss_config=LOSS, config=config, device="cpu")


def jd(a, b):
    return float(jepa_distance(a, b, config=LOSS).per_sample[0])


def s5(root, u, j):
    return torch.from_numpy(np.load(ft.extra_paths(root["data"], u, "s5")[0])[j].astype(np.float32))[None]


def native(root, u, kind, row):
    return torch.from_numpy(np.load(ft.extra_paths(root["data"], u, kind)[0])[row].astype(np.float32))[None]


def scale_vec(source="realized"):
    s = SCALE[source]
    return np.array([s["position"]]*3 + [s["rotation"]]*3 + [s["gripper"]], dtype=np.float32)


def toy_step(model, x, u_phys_rate, delta, n=256, source="realized"):
    """Φ(x, 一段常值 u（物理每周期值）, δ)：a^δ x + δ(u/尺度)W + δ(N/256)v，float64。"""
    u = (u_phys_rate.astype(np.float32)/scale_vec(source)).astype(np.float64)
    g = torch.from_numpy(delta*u)
    return x.double()*A**delta + (g @ model.w.double())[None, None, :] + delta*n/256*model.v.double()


def expected_frame_rate(root, model, entry, source="realized"):
    """10 §8.4 / 01 §5.5 的手算：起点和目标都取 s5 文件，s5 第 j 行 = 第 5j 帧。"""
    u, f0 = entry["uuid"], entry["start_frame"]
    P = root["prepared"][u].arrays["command_all" if source == "commanded" else "state_all"]
    j0 = f0//5
    z = s5(root, u, j0)
    diff = lambda a, b: np.asarray(poses_to_diffs(P[[a, b]]))[0]
    out = {}
    for f in (5, 10, 15, 20):
        d = f/15
        out[f"single:f={f}"] = jd(toy_step(model, z, diff(f0, f0 + f)/d, d, source=source).float(), s5(root, u, j0 + f//5))
    for f in (10, 15, 20):
        d, x = f/15, z
        for k in range(60//f):
            x = toy_step(model, x, diff(f0 + k*f, f0 + (k + 1)*f)/d, d, source=source).float()
        out[f"rollout:f={f}"] = jd(x, s5(root, u, j0 + 12))
    g = sum((np.asarray(poses_to_diffs(P[[f0 + 15*j, f0 + 15*(j + 1)]]))[0].astype(np.float32)/scale_vec(source)).astype(np.float64)
            for j in range(4))
    direct = z.double()*A**4 + (torch.from_numpy(g) @ model.w.double())[None, None, :] + 4*model.v.double()
    out["direct:f=60"] = jd(direct.float(), s5(root, u, j0 + 12))
    return out


# ---------------------------------------------------------------- 原始帧率

def test_frame_rate_plan_and_names():
    """单步 5/10/20 帧加一个周期的参照 15，滚动 10/20 加 15，直接 60 帧；δ 和滚动步数照 10 §8.4。"""
    assert ev.frame_rate_names(CFG) == ("single:f=5", "single:f=10", "single:f=15", "single:f=20",
                                        "rollout:f=10", "rollout:f=15", "rollout:f=20", "direct:f=60")
    d = ev.frame_rate_definitions(CFG)
    assert [(r["delta"], r["steps"]) for r in d["rollout"]] == [(10/15, 6), (1., 4), (20/15, 3)]
    assert [r["s5_target_offset"] for r in d["single"]] == [1, 2, 3, 4] and d["direct"]["segments"] == 4
    from types import SimpleNamespace as NS                         # 配置本身会拒绝，这里绕过配置核 frame_rate_plan
    for single, roll in (([5, 10, 20], [25]), ([7], [10]), ([5], [40])):
        bad = NS(time=CFG.time, encode=CFG.encode,
                 evaluation=NS(frame_rate_single=single, frame_rate_rollout=roll, frame_rate_total_frames=60))
        with pytest.raises(ValueError):
            ev.frame_rate_plan(bad)


def test_frame_rate_toy_matches_the_hand_computed_controls(root, tmp_path):
    """玩具模型：每个键等于按 01 §5.5 手算的控制值（u/δ 再除尺度、一段 [0, δ)）、滚动每一步用自己的起止帧、起点和目标
    都是 s5 文件的行（不是主特征）。"""
    model = Toy()
    summary = ev.evaluate_frame_rate(tmp_path/"id", model=model, **split_args(root, "id_test"))
    body = json.loads((tmp_path/"id"/ev.FRAME_RATE_FILE).read_text())
    doc = root["docs"]["id_test"]
    assert [r["id"] for r in body["entries"]] == [e["id"] for e in doc["frame_rate"]] and len(doc["frame_rate"]) >= 4
    worst = 0.
    for r in body["entries"]:
        assert r["excluded"] is None and r["outcome"] == OUTCOME[r["uuid"][11]]
        assert r["s5_start_row"] == r["start_frame"]//5 and r["start_frame"] % 15 == 0
        want = expected_frame_rate(root, model, r)
        assert set(r["values"]) == set(want)
        for k, v in want.items():
            assert r["values"][k] == pytest.approx(v, rel=2e-4, abs=2e-6), (r["id"], k)
            worst = max(worst, abs(v))
    assert worst > 1e-2 and summary["counts"]["excluded"] == 0
    # 步长真的不同：同一起点不同键的值不同
    r = body["entries"][0]["values"]
    assert len({round(v, 9) for v in r.values()}) == len(r)
    # 不能用主特征当起点：主特征第 s 行和 s5 第 3s 行不同
    e = body["entries"][0]
    main = root["reader"].rows(e["uuid"], [e["start_frame"]//15])
    assert not torch.equal(main, s5(root, e["uuid"], e["start_frame"]//5))


def test_frame_rate_identity_and_macro(root, tmp_path):
    """identity：每个键等于 d(z_s5, 目标 s5 行)；汇总是回合宏平均、成败分开。"""
    summary = ev.evaluate_frame_rate(tmp_path/"ood", model=Identity(), **split_args(root, "ood_test"))
    rows = json.loads((tmp_path/"ood"/ev.FRAME_RATE_FILE).read_text())["entries"]
    for r in rows:
        j0 = r["start_frame"]//5
        z = s5(root, r["uuid"], j0)
        for f in (5, 10, 15, 20):
            assert r["values"][f"single:f={f}"] == pytest.approx(jd(z, s5(root, r["uuid"], j0 + f//5)), rel=1e-6)
        for k in ("rollout:f=10", "rollout:f=15", "rollout:f=20", "direct:f=60"):
            assert r["values"][k] == pytest.approx(jd(z, s5(root, r["uuid"], j0 + 12)), rel=1e-6)
    for outcome in ("all", "success", "failure"):
        by = {}
        for r in rows:
            if outcome == "all" or r["outcome"] == outcome:
                by.setdefault(r["uuid"], []).append(r["values"]["single:f=5"])
        got = summary["macro"]["single:f=5"][outcome]
        assert got["episodes"] == len(by) == 1 + (outcome == "all")
        assert got["episode_macro_mean"] == pytest.approx(sum(sum(v)/len(v) for v in by.values())/len(by), rel=1e-12)


def test_frame_rate_ac_small_steps(root, tmp_path):
    """AC：一步 5 帧，动作是这 5 帧的物理位姿差（不除以 δ、不除尺度）；时长 δ 一次调用 3δ 步；第 k 次调用的起点状态是
    state_all[f0] 用 compute_new_pose 按前面的小步积分；每次调用的输入是上一次的输出。"""
    calls = []

    def ac_steps(model, raw, actions, start_state, grid):
        assert actions.dtype == torch.float64 and start_state.dtype == torch.float64 and grid.height == 16
        calls.append((raw.clone(), actions.clone(), start_state.clone()))
        return raw*0.5 + actions.sum(1).to(torch.float32).mean(-1)[:, None, None]

    args = split_args(root, "id_test")
    entries = ev._frame_rate_entries(args["doc"], set(), CFG)[:2]
    rec = ev.frame_rate_group(Toy(), entries, prepared=root["prepared"], episodes=args["episodes"],
                              extra_reader=root["extra"], control_spec=spec(), loss_config=LOSS, config=CFG,
                              device="cpu", ac_steps=ac_steps)
    # 单步 4 个键各 1 次，滚动 6 + 4 + 3 次，直接 1 次
    assert len(calls) == 4 + 13 + 1
    sizes = [c[1].shape[1] for c in calls]
    assert sizes == [1, 2, 3, 4] + [2]*6 + [3]*4 + [4]*3 + [12]
    order = [("s", 5), ("s", 10), ("s", 15), ("s", 20)] + [("r", 10, k) for k in range(6)] + \
            [("r", 15, k) for k in range(4)] + [("r", 20, k) for k in range(3)] + [("d", 60)]
    for b, e in enumerate(entries):
        P = root["prepared"][e["uuid"]].arrays["state_all"]
        f0 = e["start_frame"]
        small = np.asarray(poses_to_diffs(P[f0:f0 + 61:5]))
        state = torch.from_numpy(P[f0].copy())[None, None]
        integrated = [P[f0].copy()]
        for a in small:
            state = compute_new_pose(state, torch.from_numpy(a.copy())[None, None])
            integrated.append(state[0, 0].numpy().copy())
        for c, key in zip(calls, order):
            first = 0 if key[0] != "r" else key[2]*key[1]//5
            n = c[1].shape[1]
            assert np.array_equal(c[1][b].numpy(), small[first:first + n]), key
            assert np.array_equal(c[2][b].numpy(), integrated[first]), key
    # 滚动：后一次调用的输入是前一次的输出
    assert torch.equal(calls[5][0], calls[4][0]*0.5 + calls[4][1].sum(1).to(torch.float32).mean(-1)[:, None, None])
    for r in rec:
        assert set(r["values"]) == set(ev.frame_rate_names(CFG)) and all(math.isfinite(v) for v in r["values"].values())


def test_frame_rate_out_of_bounds_and_commanded(root, tmp_path):
    """一个键的 u/δ 越出物理界时整个起点排除并计数（对所有条件都一样）；`commanded` 配置用 command_all 的位姿、跳过
    passed_commanded 为假的回合。"""
    doc = root["docs"]["ood_test"]
    e = doc["frame_rate"][0]
    u = e["uuid"]
    p = root["prepared"][u]
    state_all = p.arrays["state_all"].copy()
    state_all[e["start_frame"] + 5, 6] = state_all[e["start_frame"], 6] + 0.4     # 5 帧夹爪差 0.4 → u/δ = 1.2 > 1
    bumped = dataclasses.replace(p, arrays={**p.arrays, "state_all": state_all})
    prepared = {**root["prepared"], u: bumped}
    summary = ev.evaluate_frame_rate(tmp_path/"oob", model=Toy(), **split_args(root, "ood_test", prepared=prepared))
    rows = json.loads((tmp_path/"oob"/ev.FRAME_RATE_FILE).read_text())["entries"]
    assert summary["counts"]["excluded"] == 1 and summary["excluded"] == [e["id"]]
    bad = next(r for r in rows if r["id"] == e["id"])
    assert bad["excluded"] == "control_out_of_bounds" and all(v is None for v in bad["values"].values())
    assert all(r["excluded"] is None for r in rows if r["id"] != e["id"])

    cmd = dataclasses.replace(CFG, action=dataclasses.replace(CFG.action, source="commanded"))
    gone = uid("G")
    prepared = {**root["prepared"], gone: dataclasses.replace(root["prepared"][gone],
                                                              info={**root["prepared"][gone].info, "passed_commanded": False})}
    summary = ev.evaluate_frame_rate(tmp_path/"cmd", model=Toy(), **split_args(root, "id_test", config=cmd,
                                                                               prepared=prepared, source="commanded"))
    rows = json.loads((tmp_path/"cmd"/ev.FRAME_RATE_FILE).read_text())["entries"]
    assert rows and not any(r["uuid"] == gone for r in rows)
    assert summary["counts"]["entries"] == sum(1 for x in root["docs"]["id_test"]["frame_rate"] if x["uuid"] != gone)
    for r in rows:
        want = expected_frame_rate(root, Toy(), r, source="commanded")
        for k, v in want.items():
            assert r["values"][k] == pytest.approx(v, rel=2e-4, abs=2e-6), (r["id"], k)


def test_frame_rate_and_resolution_files_are_reused(root, tmp_path):
    """已写好的 frame_rate.json / resolution.json 原样复用（模型不再被调用）；起点或窗口不是这一次的就报错。"""
    args = split_args(root, "id_test")
    ev.evaluate_frame_rate(tmp_path, model=Toy(), **args)
    ev.evaluate_resolution(tmp_path, model=Toy(), reader=root["reader"], provenance=root["provenance"], **args)
    before = {n: (tmp_path/n).read_bytes() for n in (ev.FRAME_RATE_FILE, ev.RESOLUTION_FILE)}

    class Never(torch.nn.Module):
        def forward(self, *a):
            raise AssertionError("model called on a reused file")
    a = ev.evaluate_frame_rate(tmp_path, model=Never(), **args)
    b = ev.evaluate_resolution(tmp_path, model=Never(), reader=root["reader"], provenance=root["provenance"], **args)
    assert {n: (tmp_path/n).read_bytes() for n in before} == before and a["counts"]["entries"] and b["counts"]["windows"]
    for name in before:
        body = json.loads(before[name])
        body["entries"] = body["entries"][1:]
        (tmp_path/name).write_text(json.dumps(body))
    with pytest.raises(ValueError):
        ev.evaluate_frame_rate(tmp_path, model=Toy(), **args)
    with pytest.raises(ValueError):
        ev.evaluate_resolution(tmp_path, model=Toy(), reader=root["reader"], provenance=root["provenance"], **args)


# ---------------------------------------------------------------- 原始分辨率

def test_resolution_reads_the_same_rows_on_the_native_grid(root, tmp_path):
    """r192/r320 的第 s、s + d 行，网格 regular_grid(12, 12)、(20, 20)，不重采样；同一批窗口的 16×16 未来误差一起列出。
    只用附加回合里 d ∈ {1, 2, 4, 8} 的窗口。"""
    model, grids = Toy(), []

    class Recording(torch.nn.Module):
        def forward(self, raw, controls, duration, grid):
            grids.append((raw.shape[1], grid))
            return model(raw, controls, duration, grid)

    summary = ev.evaluate_resolution(tmp_path/"id", model=Recording(), reader=root["reader"],
                                     provenance=root["provenance"], **split_args(root, "id_test"))
    rows = json.loads((tmp_path/"id"/ev.RESOLUTION_FILE).read_text())["entries"]
    doc = root["docs"]["id_test"]
    want_ids = [w["id"] for w in doc["windows"] if w["duration"] in (1, 2, 4, 8) and w["uuid"] in doc["extra_episodes"]]
    assert sorted(r["id"] for r in rows) == sorted(want_ids) and summary["counts"]["windows"] == len(want_ids)
    for n, g in grids:
        side = int(math.isqrt(n))
        ref = regular_grid(side, side)
        assert side*side == n and g.height == g.width == side and torch.equal(g.coords, ref.coords)
    assert {n for n, _ in grids} == {256, 144, 400}
    scaled = lambda u: (root["prepared"][u].arrays["action_realized"].astype(np.float32)/scale_vec()).astype(np.float64)
    for r in rows:
        u, d, s = r["uuid"], r["duration"], r["start"]
        g = torch.from_numpy(scaled(u)[s:s + d].sum(0))
        for kind, n in (("r192", 144), ("r320", 400)):
            z, zt = native(root, u, kind, s), native(root, u, kind, s + d)
            phi = z.double()*A**d + (g @ model.w.double())[None, None, :] + d*n/256*model.v.double()
            assert r["values"][f"future:{kind}"] == pytest.approx(jd(phi.float(), zt), rel=2e-4), (r["id"], kind)
        z, zt = root["reader"].rows(u, [s]), root["reader"].rows(u, [s + d])
        phi = z.double()*A**d + (g @ model.w.double())[None, None, :] + d*model.v.double()
        assert r["values"]["future:g=16"] == pytest.approx(jd(phi.float(), zt), rel=2e-4)
    assert set(summary["macro"]) == {"1", "2", "4", "8"}
    # identity：d(z_r, z*_r)
    ev.evaluate_resolution(tmp_path/"idn", model=Identity(), reader=root["reader"], provenance=root["provenance"],
                           **split_args(root, "id_test"))
    for r in json.loads((tmp_path/"idn"/ev.RESOLUTION_FILE).read_text())["entries"]:
        z, zt = native(root, r["uuid"], "r320", r["start"]), native(root, r["uuid"], "r320", r["start"] + r["duration"])
        assert r["values"]["future:r320"] == pytest.approx(jd(z, zt), rel=1e-6)


def test_resolution_checks_the_extra_records(root, tmp_path):
    """附加特征要和准备文件配对（准备 JSON 的 SHA）、行就是主特征的帧。"""
    args = split_args(root, "id_test")
    u = root["docs"]["id_test"]["extra_episodes"][0]
    episodes = dict(args["episodes"])
    episodes[u] = dataclasses.replace(episodes[u], prepared_sha256="0"*64)
    with pytest.raises(ValueError):
        ev.evaluate_resolution(tmp_path/"a", model=Toy(), reader=root["reader"], provenance=root["provenance"],
                               **{**args, "episodes": episodes})
    class OtherPrepared:                     # 主特征配对正常，只有附加特征记的准备 JSON SHA 不同
        def __init__(self, inner):
            self.inner = inner

        def record(self, uuid, kind):
            r = self.inner.record(uuid, kind)
            return {**r, "prepared_sha256": "0"*64} if uuid == u else r

        def rows(self, uuid, kind, rows):
            return self.inner.rows(uuid, kind, rows)
    with pytest.raises(ValueError, match="test-extra features were encoded from another prepared file"):
        ev.evaluate_resolution(tmp_path/"c", model=Toy(), reader=root["reader"], provenance=root["provenance"],
                               **{**args, "extra_reader": OtherPrepared(root["extra"])})
    with pytest.raises(ValueError, match="test-extra features were encoded from another prepared file"):
        ev.evaluate_frame_rate(tmp_path/"d", model=Toy(), **{**args, "extra_reader": OtherPrepared(root["extra"])})
    p = root["prepared"][u]
    shifted = dataclasses.replace(p, arrays={**p.arrays, "frame_index": p.arrays["frame_index"] + 1})
    with pytest.raises(ValueError):
        ev.evaluate_resolution(tmp_path/"b", model=Toy(), reader=root["reader"], provenance=root["provenance"],
                               **{**args, "prepared": {**root["prepared"], u: shifted}})


# ---------------------------------------------------------------- 任务

def test_evaluate_job_with_extra_axes(root, tmp_path):
    """给了附加特征时，任务在两个测试集里另写 frame_rate.json 和 resolution.json，报告的 axes 记上；AC 没有小步前向报错，
    非 AC 给了小步前向报错。"""
    kwargs = dict(seed=1703, job={"checkpoint": None}, splits=root["docs"], prepared=root["prepared"],
                  outcomes=root["outcomes"], reader=root["reader"], control_spec=spec(), provenance=root["provenance"],
                  loss_config=LOSS, config=CFG, device="cpu")
    report = ev.evaluate_job(tmp_path/"j", model=Toy(), condition="po_jepa_full", extra_reader=root["extra"], **kwargs)
    assert report["axes"] == {"windows": True, "frame_rate": True, "resolution": True, "planning": False}
    assert report["job"]["inputs"]["extra_manifest_sha256"] == root["extra"].manifest_sha256
    # S09b：附加特征清单换了（同样的文件、清单里少一条回合）就不复用已完成的任务
    manifest = {"schema": ft.EXTRA_MANIFEST_SCHEMA,
                "episodes": [{"uuid": u, "shard": root["extra"].shard(u), "files": root["extra"]._entries[u]["files"]}
                             for u in root["extra"].uuids()][:-1]}
    other = ft.ExtraFeatureReader(root["data"], manifest)
    assert other.manifest_sha256 != root["extra"].manifest_sha256
    with pytest.raises(ValueError, match="complete for another request"):
        ev.evaluate_job(tmp_path/"j", model=Toy(), condition="po_jepa_full", extra_reader=other, **kwargs)
    for split in ("id_test", "ood_test"):
        assert (tmp_path/"j"/split/ev.FRAME_RATE_FILE).exists() and (tmp_path/"j"/split/ev.RESOLUTION_FILE).exists()
        assert report["splits"][split]["frame_rate"]["counts"]["entries"] == len(root["docs"][split]["frame_rate"])
    plain = ev.evaluate_job(tmp_path/"p", model=Toy(), condition="po_jepa_full", **kwargs)
    assert plain["axes"]["frame_rate"] is False and "frame_rate" not in plain["splits"]["id_test"]
    with pytest.raises(ValueError):
        ev.evaluate_job(tmp_path/"ac", model=Toy(), condition="vjepa2_ac_small", bind=lambda m, s, r: m,
                        extra_reader=root["extra"], **kwargs)
    with pytest.raises(ValueError):
        ev.evaluate_job(tmp_path/"x", model=Toy(), condition="koopman", extra_reader=root["extra"],
                        ac_steps=lambda *a: None, **kwargs)


# ---------------------------------------------------------------- 截断间隙

def independent_spectrum(sigma_z):
    return GraphSpectrum(GraphConfig(q_dim=16, q_seed=1703, sigma_x=0.125, sigma_z=sigma_z, max_modes=16,
                                     diagnostic_atol=1e-10, diagnostic_rtol=1e-8, affinity_source="encoder",
                                     feature_dim=1408, fixed_modes=True))


def test_spectral_gap_windows_values_and_rule(root, tmp_path):
    """d = 1 的全部测试窗口（两个测试集）；λ_16、λ_17 和完整 eigh 的第 16、17 个特征值一致，ratio = 截断间隙 / 前 16 个的
    相邻间隙中位数；三个网格；分位数和 §5 的判断规则；σ_z 进了图；重跑返回同一份报告。"""
    sigma = 9.5
    report = sg.run_spectral_gap(tmp_path/"g", docs=root["docs"], reader=root["reader"], config=CFG, sigma_z=sigma,
                                 provenance={"x": 1})
    want = [(s, w["id"]) for s in ("id_test", "ood_test") for w in root["docs"][s]["windows"] if w["duration"] == 1]
    assert [(w["split"], w["id"]) for w in report["windows"]] == want and len(want) >= 8
    assert report["graph"]["sigma_z"] == sigma and report["job"]["k"] == 16
    spectrum = independent_spectrum(sigma)
    from po_jepa.geometry import apply_spatial_transfer, build_spatial_transfer
    for w in report["windows"][:3]:
        u, s = next((x["uuid"], x["start"]) for d in root["docs"].values() for x in d["windows"] if x["id"] == w["id"])
        z = root["reader"].rows(u, [s]).double()
        for size in (16, 8, 12):
            grid = regular_grid(16, 16)
            x = z
            if size != 16:
                t = build_spatial_transfer(grid, target_height=size, target_width=size)
                x, grid = apply_spatial_transfer(z.float(), t).double(), t.target
            lap = spectrum.normalized_laplacian(spectrum.affinity(x.float(), grid)).laplacian.double()
            lam = torch.linalg.eigvalsh(lap[0])
            got = w["grids"][str(size)]
            assert got["lambda_k"] == pytest.approx(float(lam[15]), abs=1e-7)
            assert got["lambda_k1"] == pytest.approx(float(lam[16]), abs=1e-7)
            gaps = (lam[1:16] - lam[:15]).tolist()
            med = float(np.median(gaps))
            assert got["median_gap"] == pytest.approx(med, abs=1e-7)
            assert got["ratio"] == pytest.approx((float(lam[16]) - float(lam[15]))/med, rel=1e-5)
    for size in ("16", "8", "12"):
        ratios = [w["grids"][size]["ratio"] for w in report["windows"]]
        entry = report["summary"]["grids"][size]
        assert entry["windows"] == len(want)
        assert entry["median_ratio"] == pytest.approx(float(torch.quantile(torch.tensor(ratios, dtype=torch.float64), .5)))
        assert entry["median_ratio_above_1"] == (entry["median_ratio"] > 1)
    assert report["summary"]["keep_sentence"] == all(report["summary"]["grids"][s]["median_ratio_above_1"] for s in ("16", "8", "12"))
    assert set(report["by_split"]) == {"id_test", "ood_test"}
    other = sg.run_spectral_gap(tmp_path/"h", docs=root["docs"], reader=root["reader"], config=CFG, sigma_z=2*sigma,
                                provenance={"x": 1})
    assert other["windows"][0]["grids"]["16"]["lambda_k"] != report["windows"][0]["grids"]["16"]["lambda_k"]
    sha = hashlib.sha256((tmp_path/"g"/"report.json").read_bytes()).hexdigest()
    assert json.loads((tmp_path/"g"/"COMPLETE.json").read_text()) == {"report_sha256": sha}
    assert sg.run_spectral_gap(tmp_path/"g", docs=root["docs"], reader=root["reader"], config=CFG, sigma_z=sigma,
                               provenance={"x": 1}) == report
    with pytest.raises(ValueError, match="complete for another request"):          # S11b-fix1：不再返回旧报告
        sg.run_spectral_gap(tmp_path/"g", docs=root["docs"], reader=root["reader"], config=CFG, sigma_z=1.,
                            provenance={})


def test_spectral_gap_completed_job_needs_the_same_inputs(root, tmp_path):
    """S11b-P1-1：完成后换 sigma_z、来源、配置或 d = 1 测试窗口都报错，不返回旧报告；job.json、report 的 job 或窗口
    列表被改也报错；同样的输入直接返回。"""
    out = tmp_path/"g"
    prov = {"windows_sha256": {"id_test": "a"*64, "ood_test": "b"*64}, "feature_manifest_sha256": "c"*64,
            "sigma_z_sha256": "d"*64}
    run = lambda **over: sg.run_spectral_gap(out, **{**dict(docs=root["docs"], reader=root["reader"], config=CFG,
                                                             sigma_z=9.5, provenance=prov), **over})
    report = run()
    assert run() == report
    doc = root["docs"]["id_test"]
    first = next(i for i, w in enumerate(doc["windows"]) if w["duration"] == 1)
    fewer = {**root["docs"], "id_test": {**doc, "windows": doc["windows"][:first] + doc["windows"][first + 1:]}}
    moved = {**root["docs"], "id_test": {**doc, "windows": [dict(w, start=w["start"] + 1) if i == first else w
                                                            for i, w in enumerate(doc["windows"])]}}
    for over in (dict(sigma_z=9.0), dict(provenance={**prov, "feature_manifest_sha256": "e"*64}),
                 dict(provenance={**prov, "sigma_z_sha256": "f"*64}),
                 dict(config=dataclasses.replace(CFG, config_sha256="0"*64)), dict(docs=fewer), dict(docs=moved)):
        with pytest.raises(ValueError, match="complete for another request"):
            run(**over)
    job_bytes = (out/"job.json").read_bytes()
    (out/"job.json").write_bytes(job_bytes.replace(b"9.5", b"9.0"))
    with pytest.raises(ValueError, match="complete for another request"):
        run()
    (out/"job.json").write_bytes(job_bytes)
    for edit in (lambda b: b["job"].update(sigma_z=9.0), lambda b: b["windows"].pop()):
        body = json.loads((out/"report.json").read_text())
        edit(body)
        (out/"report.json").unlink()
        (out/"COMPLETE.json").unlink()
        (out/"report.json").write_text(json.dumps(body))
        (out/"COMPLETE.json").write_text(json.dumps({"report_sha256": hashlib.sha256((out/"report.json").read_bytes()).hexdigest()}))
        with pytest.raises(ValueError, match="complete for another request"):
            run()


def test_spectral_gap_inputs(root, tmp_path):
    """σ_z 必须是正的有限数、标定时的配置必须是本配置；K、σ_x 要和 DROID 配置相同；job.json 不同报错。"""
    for bad in (0., -1., float("nan"), 3):
        with pytest.raises(ValueError):
            sg.graph_block(bad)
    from po_jepa.droid.calibrate import SIGMA_SCHEMA
    doc = {"schema": SIGMA_SCHEMA, "sigma_z": 7.25, "config_sha256": CFG.config_sha256,
           "feature_manifest_sha256": "f"*64, "action_scale_sha256": "a"*64}
    current = dict(feature_manifest_sha256="f"*64, action_scale_sha256="a"*64)
    (tmp_path/"s.json").write_text(json.dumps(doc))
    assert sg.read_sigma_z(tmp_path/"s.json", CFG, **current)[0] == 7.25
    (tmp_path/"t.json").write_text(json.dumps(dict(doc, config_sha256="0"*64)))
    with pytest.raises(ValueError):
        sg.read_sigma_z(tmp_path/"t.json", CFG, **current)
    for k in current:                       # RF-P1-2：截断间隙读 σ_z 时用全量特征清单和动作尺度核来源
        with pytest.raises(ValueError, match="was calibrated"):
            sg.read_sigma_z(tmp_path/"s.json", CFG, **dict(current, **{k: "0"*64}))
    wrong = dataclasses.replace(CFG, model=dataclasses.replace(CFG.model, sigma_x=0.25))
    with pytest.raises(ValueError):
        sg.build_spectrum(wrong, 7.25)
    (tmp_path/"j").mkdir()
    (tmp_path/"j"/"job.json").write_text("{}")
    with pytest.raises(ValueError):
        sg.run_spectral_gap(tmp_path/"j", docs=root["docs"], reader=root["reader"], config=CFG, sigma_z=7.25, provenance={})
    with pytest.raises(ValueError):
        sg.gap_windows({"id_test": root["docs"]["id_test"]})


def test_spectral_gap_cli_through_the_entry_point(root, tmp_path, monkeypatch):
    """tools/droid.py eval --job eval/spectral-gap：读冻结窗口、两个测试片的特征清单、<name>/calibration/sigma_z.json，
    写完成标志；其他 eval 任务转给 `droid.eval_job.main`（S09b）。"""
    import shutil
    import sys
    sys.path.insert(0, str(cache.ROOT/"tools"))
    import droid  # tools/droid.py
    from po_jepa.droid import prepare as pr
    data = tmp_path/"data"
    shutil.copytree(root["data"], data)
    windows = pr.prepared_dir(data)/pr.WINDOWS_DIR
    windows.mkdir(parents=True)
    for split, doc in root["docs"].items():
        cache._write_json(windows/f"{split}.json", doc)
        path = ft.manifest_path(data, split, BASE)
        cache._write_json(path, {**root["manifests"][split], "config_sha256": BASE.config_sha256})
        done = ft.complete_path(data, split, BASE)
        done.parent.mkdir(parents=True, exist_ok=True)
        cache._write_json(done, {"schema": ft.COMPLETE_SCHEMA, "shard": split, "config_sha256": BASE.config_sha256,
                                 "manifest_sha256": cache.file_sha(path)})
    run = tmp_path/"runs"
    cal = run/BASE.name/"calibration"
    cal.mkdir(parents=True)
    # 这个数据根没有划分：让“全部片”= 两个测试片 + 一个只为清单存在的训练片 train_extra（RF-P1-2：σ_z 按全量清单和动作
    # 尺度核来源）。全量清单因此不等于测试片清单，把测试片清单误当全量传给 read_sigma_z 的实现会在这里失败（R4 审核）。
    extra = "train_extra"
    path = ft.manifest_path(data, extra, BASE)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache._write_json(path, {**root["manifests"]["id_test"], "shard": extra, "config_sha256": BASE.config_sha256})
    done = ft.complete_path(data, extra, BASE)
    done.parent.mkdir(parents=True, exist_ok=True)
    cache._write_json(done, {"schema": ft.COMPLETE_SCHEMA, "shard": extra, "config_sha256": BASE.config_sha256,
                             "manifest_sha256": cache.file_sha(path)})
    real_manifests = ft.feature_manifests
    monkeypatch.setattr(ft, "feature_manifests", lambda d, c, shards=None: real_manifests(
        d, c, shards=[*ev.TEST_SPLITS, extra] if shards is None else shards))
    _, full_sha = ft.feature_manifests(data, BASE)
    _, test_sha = ft.feature_manifests(data, BASE, shards=list(ev.TEST_SPLITS))
    assert full_sha != test_sha
    scale = pr.prepared_dir(data)/pr.ACTION_SCALE_FILE
    if not scale.exists():
        cache._write_json(scale, {"fake": "scale"})
    from po_jepa.droid.calibrate import SIGMA_SCHEMA

    def sigma(value):
        return json.dumps({"schema": SIGMA_SCHEMA, "sigma_z": value, "config_sha256": BASE.config_sha256,
                           "feature_manifest_sha256": full_sha, "action_scale_sha256": cache.file_sha(scale)})
    (cal/"sigma_z.json").write_text(sigma(11.0))
    argv = ["eval", "--config", str(CONFIGS/"pilot.yaml"), "--data-root", str(data), "--run-root", str(run)]
    assert droid.main(argv + ["--job", "eval/spectral-gap"]) == 0
    out = sg.job_dir(run, BASE)
    report = json.loads((out/"report.json").read_text())
    assert report["graph"]["sigma_z"] == 11.0 and (out/"COMPLETE.json").exists()
    assert set(report["job"]["inputs"]) == {"windows_sha256", "feature_manifest_sha256", "sigma_z_sha256"}
    assert report["job"]["inputs"]["feature_manifest_sha256"] == test_sha     # job 身份仍记测试片清单
    assert droid.main(argv + ["--job", "eval/spectral-gap"]) == 0          # 已完成、输入相同：返回 0，报告不变
    assert json.loads((out/"report.json").read_text()) == report
    (cal/"sigma_z.json").write_text(sigma(12.0))
    with pytest.raises(ValueError, match="complete for another request"):  # S11b-fix1：入口不再凭完成标志跳过
        droid.main(argv + ["--job", "eval/spectral-gap"])
    calls = []
    monkeypatch.setitem(droid.MAINS, "eval", lambda a: calls.append(a) or 75)
    assert droid.main(argv + ["--job", "eval/identity"]) == 75
    assert calls == [["--config", str(CONFIGS/"pilot.yaml"), "--data-root", str(data), "--run-root", str(run),
                      "--job", "eval/identity", "--device", "cuda"]]
