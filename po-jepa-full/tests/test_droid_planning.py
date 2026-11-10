"""S12：DROID 离线规划（docs/design/11-droid-planning.md；验收 11 §6、14 §3 S12）。

合成的小数据根：17 条回合（每条 8 行，d = 4 时正好 4 个窗口，其余 16 条回合给出 64 个同时长窗口，够 63 个干扰项），
冻结窗口用 S05 的 `windows.frozen_windows` 生成。期望值一律独立重算：动作直接从准备数组取、除以尺度用 float32 手算，
CEM 种子用 hashlib 手算，名次指标手算。玩具模型：
- identity：输出起点本身，不看动作；
- “代价 = 动作距离”：从控制的 source_ids 里取出窗口编号，输出 目标 + α·q，α = 候选动作（模型单位）和真实动作的距离，
  q 是固定的随机方向（真实动作时 α = 0，输出正好是目标，代价为 0）。
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
from po_jepa.droid import features as ft
from po_jepa.droid import planning as pl
from po_jepa.droid import windows as wn
from po_jepa.droid.actions import droid_control_spec
from po_jepa.droid.config import load_config
from po_jepa.droid.prepare import PreparedEpisode
from test_droid_batches import SYNTHETIC, droid_model
from test_droid_index import CONFIGS

CFG = load_config(CONFIGS/"pilot.yaml")
N_EPISODES, ROWS = 17, 8
SCALE = {"realized": {"position": 0.01, "rotation": 0.05, "gripper": 0.5},
         "commanded": {"position": 0.02, "rotation": 0.04, "gripper": 0.25}}
SCALE7 = np.array([0.01]*3 + [0.05]*3 + [0.5], dtype=np.float32)
LOSS = CFG.loss_config(time_weight=0.0, space_weight=0.0)


def uid(i: int) -> str:
    return f"LAB+{i:08d}+2023-07-07-09h-42m-23s"


UUIDS = [uid(i) for i in range(N_EPISODES)]


def _prepared(u: str, seed: int) -> PreparedEpisode:
    rng = np.random.default_rng(seed)
    arrays = {"action_realized": rng.uniform(-0.02, 0.02, (ROWS - 1, 7)),
              "action_commanded": rng.uniform(-0.02, 0.02, (ROWS - 1, 7)),
              "state": rng.uniform(-0.5, 0.5, (ROWS, 7))}
    info = {"uuid": u, "rows": ROWS, "passed_realized": True, "passed_commanded": True}
    return PreparedEpisode(info, hashlib.sha256(f"prep-{u}".encode()).hexdigest(), arrays)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("droid_planning")
    prepared, stored, entries = {}, {}, []
    for i, u in enumerate(UUIDS):
        prepared[u] = _prepared(u, 500 + i)
        array = np.random.default_rng(900 + i).standard_normal((ROWS, ft.TOKENS, ft.WIDTH)).astype(np.float16)
        npy, js = ft.feature_paths(root, "test-000", u)
        record, js_sha = ft.write_features(npy, js, array, lambda sha, u=u, a=array: {
            "schema": ft.FEATURE_SCHEMA, "uuid": u, "shard": "test-000", "rows": a.shape[0], "shape": list(a.shape),
            "dtype": ft.STORAGE_DTYPE, "file_sha256": sha, "row_sha256": ft.row_sha256(a),
            "prepared_sha256": prepared[u].json_sha256, "encoder": SYNTHETIC})
        entries.append({"uuid": u, "rows": ROWS, "file_sha256": record["file_sha256"], "feature_json_sha256": js_sha})
        stored[u] = array
    manifests = {"test-000": {"schema": ft.MANIFEST_SCHEMA, "shard": "test-000", "encoder": SYNTHETIC, "episodes": entries}}
    windows = wn.frozen_windows("id_test", UUIDS, rows={u: ROWS for u in UUIDS}, passed={u: True for u in UUIDS},
                                prepared_sha256={u: prepared[u].json_sha256 for u in UUIDS}, splits_sha256="s"*64,
                                config=CFG)
    windows = json.loads(json.dumps(windows))            # 和读盘后的文件一样（键是字符串、列表是 list）
    reader = ft.FeatureReader(root, manifests)
    episodes = bt.episode_arrays(prepared, UUIDS, "realized")
    spec = droid_control_spec(SCALE, source="realized", config=CFG)
    inputs = pl.PlanningInputs(windows, reader, episodes, spec, CFG, LOSS)
    return inputs, prepared, stored


def _subset(inputs, rank: int, cem: int):
    """只取前 `rank` 个排名窗口、前 `cem` 个 CEM 窗口（每个窗口 64 个候选一次前向，全部 136 个太慢）。"""
    w = json.loads(json.dumps(inputs.windows))
    keep = list(w["planning"]["rank"])[:rank]
    w["planning"] = {"rank": {k: w["planning"]["rank"][k] for k in keep}, "cem": w["planning"]["cem"][:cem]}
    return dataclasses.replace(inputs, windows=w)


def _table(windows):
    return {w["id"]: w for w in windows["windows"]}


def _true_model_units(prepared, w):
    """窗口 w 的真实动作在模型单位里的值（float32 手算：物理值转 float32 再除以 float32 尺度）。"""
    phys = prepared[w["uuid"]].arrays["action_realized"][w["start"]:w["start"] + w["duration"]].astype(np.float32)
    return torch.from_numpy(phys/SCALE7)


class Identity:
    def __init__(self):
        self.calls = []

    def __call__(self, raw, controls, duration, grid):
        self.calls.append(controls)
        return raw.clone()


class DistanceToy:
    """输出 目标 + α·q：α = ‖候选动作 − 真实动作‖（模型单位，全部有效段），只依赖动作。记录每次调用看到的控制。"""

    def __init__(self, inputs, prepared, stored, seed=7):
        self.table, self.prepared, self.stored = _table(inputs.windows), prepared, stored
        self.q = torch.from_numpy(np.random.default_rng(seed).standard_normal((ft.TOKENS, ft.WIDTH)).astype(np.float32))
        self.calls = []

    def __call__(self, raw, controls, duration, grid):
        self.calls.append(controls)
        wid = controls.source_ids[0].split("|cand=")[0].split("|cem|")[0]
        assert all(i.startswith(wid + "|") for i in controls.source_ids)
        w = self.table[wid]
        target = torch.from_numpy(self.stored[w["uuid"]][w["start"] + w["duration"]].astype(np.float32))
        d = w["duration"]
        assert torch.equal(controls.n_segments, torch.full((raw.shape[0],), d))
        alpha = (controls.u_values[:, :d] - _true_model_units(self.prepared, w)).square().sum((1, 2)).sqrt()
        return target[None] + alpha[:, None, None]*self.q[None]


# ---------------------------------------------------------------- 名次和随机水平（11 §2）

def test_rank_metrics_by_hand_and_random_level():
    c = np.array([2, 1, 2, 3, 2], dtype=np.float32)                 # 真实动作代价 2：1 个更小、2 个相等
    m = pl.rank_metrics(c)
    assert (m["r_lo"], m["ties"], m["mean_rank"], m["top1"]) == (2, 2, 3.0, 0.0)
    assert m["reciprocal_rank"] == pytest.approx((1/2 + 1/3 + 1/4)/3, abs=1e-15)
    m = pl.rank_metrics(np.array([1, 1, 3, 1], dtype=np.float32))   # 第 1 名并列 3 个
    assert (m["r_lo"], m["ties"], m["mean_rank"], m["top1"]) == (1, 2, 2.0, pytest.approx(1/3))
    m = pl.rank_metrics(np.array([0, 1, 2], dtype=np.float32))
    assert (m["r_lo"], m["ties"], m["top1"], m["reciprocal_rank"]) == (1, 0, 1.0, 1.0)
    # “相等”是 float32 逐位相等：差一个 ulp 就不算并列
    c0 = np.float32(1.0)
    m = pl.rank_metrics(np.array([c0, np.nextafter(c0, np.float32(2)), np.nextafter(c0, np.float32(0))], dtype=np.float32))
    assert (m["r_lo"], m["ties"]) == (2, 0)
    r = pl.random_level(64)
    assert r["mean_rank"] == 32.5 and r["top1"] == 1/64
    assert r["reciprocal_rank"] == pytest.approx(math.fsum(1/k for k in range(1, 65))/64) and round(r["reciprocal_rank"], 4) == 0.0741
    with pytest.raises(ValueError):
        pl.rank_metrics(np.array([1, 2], dtype=np.float64))


# ---------------------------------------------------------------- 排名（11 §2、§6）

def test_identity_ties_give_exactly_the_random_level(data):
    assert len(data[0].windows["planning"]["rank"]) == 17*8 and len(data[0].windows["planning"]["cem"]) == 17*2
    inputs = _subset(data[0], 8, 0)                    # 回合 0 的 d = 1、4 各 4 个窗口
    records, skipped = pl.rank_split(Identity(), inputs, "cpu")
    assert skipped == {"rank_windows_skipped": 0}
    assert len(records) == 8
    for r in records:
        assert len(set(r["costs"])) == 1 and (r["r_lo"], r["ties"]) == (1, 63)
        assert r["mean_rank"] == 32.5 and r["top1"] == 1/64
        assert r["reciprocal_rank"] == pytest.approx(math.fsum(1/k for k in range(1, 65))/64, abs=1e-15)
    rep = pl.report(records, [], split="id_test", config=CFG, skipped=skipped)
    for d in ("1", "4"):
        row = rep["by_duration"][d]
        assert row["mean_rank"] == 32.5 and row["top1"] == pytest.approx(1/64)
        assert row["reciprocal_rank"] == pytest.approx(rep["random_level"]["reciprocal_rank"], abs=1e-15)
        assert row["rank_windows"] == 4 and row["rank_episodes"] == 1


def test_candidates_are_the_true_and_distractor_actions_on_the_windows_timeline(data):
    inputs, prepared, stored = data
    inputs = _subset(inputs, 6, 0)
    toy = DistanceToy(inputs, prepared, stored)
    records, _ = pl.rank_split(toy, inputs, "cpu")
    table = _table(inputs.windows)
    for r, controls in zip(records, toy.calls):
        w = table[r["id"]]
        cands = [w, *(table[o] for o in inputs.windows["planning"]["rank"][r["id"]])]
        assert r["candidates"] == [c["id"] for c in cands] and len(cands) == 64
        assert r["r_lo"] == 1 and r["ties"] == 0 and r["top1"] == 1.0 and r["costs"][0] == 0.0
        d, s = w["duration"], w["start"]
        for k, c in enumerate(cands):                                    # 候选 k 的动作 = 窗口 c 自己的动作
            assert torch.equal(controls.u_values[k, :d], _true_model_units(prepared, c))
        assert not torch.equal(controls.u_values[1, :d], controls.u_values[0, :d])   # 干扰项的动作确实不同
        assert torch.equal(controls.source_origin_seconds, torch.full((64,), float(s), dtype=torch.float64))
        assert torch.equal(controls.starts[:, :d], torch.arange(d, dtype=torch.float64).expand(64, d))
        assert torch.equal(controls.ends[:, :d], torch.arange(1, d + 1, dtype=torch.float64).expand(64, d))
    assert len(records) == len(toy.calls)


def test_rank_skips_windows_whose_episode_or_distractor_is_not_in_the_action_table(data):
    inputs = _subset(data[0], 8, 0)
    gone = UUIDS[3]
    smaller = dataclasses.replace(inputs, episodes={u: e for u, e in inputs.episodes.items() if u != gone})
    records, skipped = pl.rank_split(Identity(), smaller, "cpu")
    table = _table(inputs.windows)
    expect = [wid for wid, others in inputs.windows["planning"]["rank"].items()
              if table[wid]["uuid"] != gone and all(table[o]["uuid"] != gone for o in others)]
    assert [r["id"] for r in records] == expect and len(expect) < len(inputs.windows["planning"]["rank"])
    assert skipped["rank_windows_skipped"] == len(inputs.windows["planning"]["rank"]) - len(expect)
    assert all(len(r["costs"]) == 64 for r in records)


def test_header_and_distractor_checks(data):
    inputs, _, _ = data
    bad = json.loads(json.dumps(inputs.windows)); bad["salt"] = "other"
    with pytest.raises(ValueError, match="window_salt"):
        pl.rank_split(Identity(), dataclasses.replace(inputs, windows=bad), "cpu")
    bad = json.loads(json.dumps(inputs.windows)); wid = next(iter(bad["planning"]["rank"]))
    bad["planning"]["rank"][wid] = bad["planning"]["rank"][wid][:-1]
    with pytest.raises(ValueError, match="distinct distractors"):
        pl.rank_split(Identity(), dataclasses.replace(inputs, windows=bad), "cpu")
    bad = json.loads(json.dumps(inputs.windows)); wid = next(iter(bad["planning"]["rank"]))
    same = next(w["id"] for w in bad["windows"] if w["uuid"] == bad["windows"][0]["uuid"] and w["id"] != wid
                and w["duration"] == _table(bad)[wid]["duration"])
    bad["planning"]["rank"][wid][0] = same
    with pytest.raises(ValueError, match="other episodes"):
        pl.rank_split(Identity(), dataclasses.replace(inputs, windows=bad), "cpu")


# ---------------------------------------------------------------- CEM（11 §3、§6）

def test_cem_seed_and_search_bounds():
    salt, wid = CFG.data.window_salt, "LAB+00000001+x|d4|s2"
    assert pl.cem_seed(salt, wid) == int(hashlib.sha256(f"{salt}|cem|{wid}".encode()).hexdigest()[:16], 16)
    spec = droid_control_spec(SCALE, source="realized", config=CFG)
    lo, hi = pl.search_bounds(spec, 1.5)
    assert lo.tolist() == [-1.5]*4 and hi.tolist() == [1.5]*4
    wide = droid_control_spec({**SCALE, "realized": {"position": 0.01, "rotation": 0.05, "gripper": 0.8}},
                              source="realized", config=CFG)
    lo, hi = pl.search_bounds(wide, 1.5)                                   # 1.5 × 0.8 > 夹爪界 1.0：截到 1/0.8
    assert hi.tolist() == [1.5, 1.5, 1.5, 1.0/0.8] and lo.tolist() == [-1.5, -1.5, -1.5, -1.0/0.8]


def _numpy_distance_cem(true, spec, seed, **kw):
    target = np.concatenate([true[:, :3]/SCALE7[:3], true[:, 6:]/SCALE7[6:]], axis=1)
    def cost(a):                                       # a：物理 float32 [S, d, 7]，代价 = 搜索通道的模型单位距离
        u = np.concatenate([a[..., :3]/SCALE7[:3], a[..., 6:]/SCALE7[6:]], axis=-1)
        return np.sqrt(((u - target)**2).sum((1, 2))).astype(np.float32)
    return pl.cem(cost, true, spec, seed=seed, **kw)


def test_cem_follows_the_update_rule_and_error_does_not_increase(data):
    """11 §6 要“净位移误差随轮数单调不增”。代价是逐周期的动作距离，d = 1 时净位移就是这一个周期的位置，逐窗口单调；
    d = 4 时净位移是 4 个周期的和，中间某一轮可以变大（合成数据 17 个 d = 4 窗口里有 1 个第 1 轮变大），只核“最后小于不动”和
    回合平均逐轮单调不增。"""
    inputs, prepared, _ = data
    curves = {1: [], 4: []}
    spec, c = inputs.control_spec, CFG.evaluation.cem
    kw = dict(iterations=c.iterations, samples=c.samples, elites=c.elites, init_std=c.init_std, clip=c.clip)
    for wid in inputs.windows["planning"]["cem"]:
        w = _table(inputs.windows)[wid]
        true = prepared[w["uuid"]].arrays["action_realized"][w["start"]:w["start"] + w["duration"]]
        seed = pl.cem_seed(CFG.data.window_salt, wid)
        res = _numpy_distance_cem(true, spec, seed, **kw)
        # 独立重放：同一串噪声、同样的截断、stable argsort、μ/σ 更新
        rng = np.random.default_rng(seed)
        d = w["duration"]
        mu, sd = np.zeros((d, 4)), np.full((d, 4), 0.5)
        target = np.concatenate([true[:, :3]/SCALE7[:3], true[:, 6:]/SCALE7[6:]], axis=1)
        for rnd in res["rounds"]:
            u = np.clip(mu + sd*rng.standard_normal((256, d, 4)), -1.5, 1.5)
            phys = np.concatenate([u[..., :3]*0.01, np.broadcast_to(true[:, 3:6], (256, d, 3)),
                                   u[..., 3:]*0.5], axis=-1).astype(np.float32)        # 尺度是 float64 的 0.01、0.5
            uu = np.concatenate([phys[..., :3]/SCALE7[:3], phys[..., 6:]/SCALE7[6:]], axis=-1)
            cost = np.sqrt(((uu - target)**2).sum((1, 2))).astype(np.float32)
            elite = u[np.argsort(cost, kind="stable")[:24]]
            mu, sd = elite.mean(0), elite.std(0) + 1e-3
            assert rnd["mu"] == mu.tolist() and rnd["sigma"] == sd.tolist()
            assert rnd["elite_costs"] == np.sort(cost, kind="stable")[:24].tolist()
        assert np.array_equal(res["mu"], mu)
        errs = [pl.cem_errors(np.zeros((d, 4)), true, spec)["position_error_m"]]
        errs += [pl.cem_errors(np.array(r["mu"]), true, spec)["position_error_m"] for r in res["rounds"]]
        curves[d].append(errs)
        if d == 1:
            assert all(b <= a + 1e-12 for a, b in zip(errs, errs[1:])), (wid, errs)
        assert errs[-1] < errs[0]
    mean4 = np.mean(curves[4], axis=0)
    assert len(curves[1]) == len(curves[4]) == 17 and all(b <= a for a, b in zip(mean4, mean4[1:])), mean4


def test_cem_errors_by_hand():
    spec = droid_control_spec(SCALE, source="realized", config=CFG)
    true = np.array([[0.01, 0.0, 0.02, 0.1, 0.2, 0.3, 0.2], [0.01, 0.01, -0.02, 0.1, 0.2, 0.3, -0.5]])
    mu = np.array([[1.0, 0.0, 0.0, 0.2], [0.0, 1.0, 0.0, -1.0]])           # 模型单位：位置 ×0.01、夹爪 ×0.5
    e = pl.cem_errors(mu, true, spec)
    p_hat, p = np.array([0.01, 0.01, 0.0]), np.array([0.02, 0.01, 0.0])
    assert e["position_error_m"] == pytest.approx(np.linalg.norm(p_hat - p), abs=1e-15)
    assert e["gripper_error"] == pytest.approx(abs((0.1 - 0.5) - (0.2 - 0.5)), abs=1e-15)
    assert e["position_error_zero_m"] == pytest.approx(np.linalg.norm(p)) and e["gripper_error_zero"] == pytest.approx(0.3)
    assert e["position_relative"] == pytest.approx(0.01/np.linalg.norm(p)) and e["small_displacement"] is False
    assert pl.cem_errors(np.zeros((1, 4)), np.zeros((1, 7)), spec)["position_relative"] == 0.0
    assert pl.cem_errors(np.zeros((1, 4)), np.array([[0.005, 0, 0, 0, 0, 0, 0]]), spec)["small_displacement"] is True


def test_cem_end_to_end_is_deterministic_keeps_rotation_and_shares_noise(data):
    inputs, prepared, stored = data
    inputs = _subset(inputs, 0, 2)                     # 回合 0 的 d = 1、4
    table = _table(inputs.windows)
    a, b = DistanceToy(inputs, prepared, stored, seed=7), Identity()
    first, _ = pl.cem_split(a, inputs, "cpu")
    again, _ = pl.cem_split(DistanceToy(inputs, prepared, stored, seed=7), inputs, "cpu")
    other, _ = pl.cem_split(b, inputs, "cpu")
    assert json.dumps(first) == json.dumps(again)                         # 同窗口同 seed：逐位相同
    assert [r["id"] for r in first] == inputs.windows["planning"]["cem"] and len(first) == 2
    per_window = 256//64*CFG.evaluation.cem.iterations
    assert len(a.calls) == per_window*len(first)
    for i, r in enumerate(first):
        w = table[r["id"]]
        d = w["duration"]
        assert r["seed"] == pl.cem_seed(CFG.data.window_salt, r["id"])
        rot = _true_model_units(prepared, w)[:, 3:6]
        for controls in a.calls[i*per_window:(i + 1)*per_window]:        # 旋转维在全部样本里都等于真实值
            assert controls.u_values.shape[0] == 64
            assert torch.equal(controls.u_values[:, :d, 3:6], rot.expand(64, d, 3))
            assert torch.equal(controls.source_origin_seconds, torch.full((64,), float(w["start"]), dtype=torch.float64))
        # 换条件（identity）时第一轮的样本相同：共同随机数
        for j in range(256//64):
            assert torch.equal(a.calls[i*per_window + j].u_values, b.calls[i*per_window + j].u_values)
        assert r["position_error_m"] < r["position_error_zero_m"]
        assert r["true_physical"] == prepared[w["uuid"]].arrays["action_realized"][w["start"]:w["start"] + d].tolist()
    # 条件之间的差别只来自代价：identity 全部平局，stable argsort 取前 24 个样本，结果和玩具模型不同
    assert all(x["final_model_units"] != y["final_model_units"] for x, y in zip(first, other))
    assert all(r["rounds"][0]["elite_costs"] == [r["rounds"][0]["elite_costs"][0]]*24 for r in other)


def test_report_macro_averages_by_episode_and_excludes_small_displacements():
    rank = [{"uuid": "A", "duration": 1, "mean_rank": 1.0, "top1": 1.0, "reciprocal_rank": 1.0},
            {"uuid": "A", "duration": 1, "mean_rank": 3.0, "top1": 0.0, "reciprocal_rank": 1/3},
            {"uuid": "B", "duration": 1, "mean_rank": 10.0, "top1": 0.0, "reciprocal_rank": 0.1},
            {"uuid": "B", "duration": 4, "mean_rank": 5.0, "top1": 0.0, "reciprocal_rank": 0.2}]
    cem = [{"uuid": "A", "duration": 4, "position_error_m": 0.02, "gripper_error": 0.1, "position_relative": 0.5,
            "small_displacement": False},
           {"uuid": "B", "duration": 4, "position_error_m": 0.04, "gripper_error": 0.3, "position_relative": 9.0,
            "small_displacement": True},
           {"uuid": "C", "duration": 4, "position_error_m": 0.09, "gripper_error": 0.2, "position_relative": 0.7,
            "small_displacement": False}]
    rep = pl.report(rank, cem, split="ood_test", config=CFG, skipped={"rank_windows_skipped": 0})
    one, four = rep["by_duration"]["1"], rep["by_duration"]["4"]
    assert one["mean_rank"] == pytest.approx((2.0 + 10.0)/2) and one["top1"] == pytest.approx(0.25)   # 宏平均，不是 14/3
    assert one["reciprocal_rank"] == pytest.approx(((1 + 1/3)/2 + 0.1)/2) and one["rank_windows"] == 3
    assert four["mean_rank"] == 5.0 and four["cem_windows"] == 3 and four["cem_small_displacement_windows"] == 1
    assert four["position_error_m"] == {"mean": pytest.approx(0.05), "median": pytest.approx(0.04), "episodes": 3}
    assert four["position_relative"] == {"mean": pytest.approx(0.6), "median": pytest.approx(0.6), "episodes": 2}
    assert one["position_error_m"] == {"mean": None, "median": None, "episodes": 0}
    assert rep["random_level"]["mean_rank"] == 32.5 and rep["cem"]["samples"] == 256


# ---------------------------------------------------------------- AC 的绑定（11 §1、§6）

def test_ac_bind_gets_the_windows_start_state_and_each_batchs_candidates(data):
    inputs, prepared, stored = data
    toy = DistanceToy(inputs, prepared, stored)
    seen = []

    def bind(model, start_state, reference):
        seen.append((start_state, reference))
        def predictor(raw, controls, duration, grid):
            assert controls is reference
            phys = controls.u_values*torch.tensor(controls.control_spec.scale)     # 包装层乘回物理单位
            assert torch.isfinite(phys).all()
            return model(raw, controls, duration, grid)
        return predictor

    small = json.loads(json.dumps(inputs.windows))
    keep = list(small["planning"]["rank"])[:3]
    small["planning"] = {"rank": {k: small["planning"]["rank"][k] for k in keep}, "cem": small["planning"]["cem"][:1]}
    sub = dataclasses.replace(inputs, windows=small)
    rank, _ = pl.rank_split(toy, sub, "cpu", bind=bind)
    cem_records, _ = pl.cem_split(toy, sub, "cpu", bind=bind)
    table = _table(small)
    per_window = 256//64*CFG.evaluation.cem.iterations                 # CEM：4 轮 × 每轮 4 批
    ids = [*keep, *[small["planning"]["cem"][0]]*per_window]
    assert len(seen) == 3 + per_window
    for (state, reference), wid in zip(seen, ids):
        w = table[wid]
        expect = torch.from_numpy(prepared[w["uuid"]].arrays["state"][w["start"]]).expand(64, 7)
        assert state.dtype == torch.float64 and torch.equal(state, expect)   # 干扰项也用窗口 w 的起点状态
        assert reference.u_values.shape[0] == 64
    for (_, reference), wid in zip(seen[:3], keep):
        w = table[wid]
        phys = reference.u_values[:, :w["duration"]]*torch.from_numpy(SCALE7)
        cands = [w, *(table[o] for o in small["planning"]["rank"][wid])]
        want = torch.from_numpy(np.stack([prepared[c["uuid"]].arrays["action_realized"][c["start"]:c["start"] + c["duration"]]
                                          for c in cands]).astype(np.float32))
        assert torch.allclose(phys, want, rtol=1e-6, atol=0)
    assert all(r["r_lo"] == 1 for r in rank) and len(cem_records) == 1
    with pytest.raises(ValueError, match="bind and start_state"):
        pl.costs(toy, torch.zeros(256, 1408), torch.zeros(256, 1408), 0, np.zeros((1, 1, 7), np.float32),
                 inputs.control_spec, ["x|cand=y"], loss_config=LOSS, start_state=torch.zeros(7))


# ---------------------------------------------------------------- 输出（11 §4）和真模型

def test_run_writes_three_files_and_reruns_from_the_finished_directory(data, tmp_path):
    inputs, prepared, stored = data
    small = json.loads(json.dumps(inputs.windows))
    keep = list(small["planning"]["rank"])[:4]
    small["planning"] = {"rank": {k: small["planning"]["rank"][k] for k in keep}, "cem": small["planning"]["cem"][:2]}
    sub = dataclasses.replace(inputs, windows=small)
    out = tmp_path/"id_test"/"planning"
    (tmp_path/"id_test"/"planning.partial").mkdir(parents=True)              # 上次中断的残留
    (tmp_path/"id_test"/"planning.partial"/"rank.json").write_text("{}")
    rep = pl.run(DistanceToy(inputs, prepared, stored), sub, out, "cpu")
    assert sorted(p.name for p in out.iterdir()) == ["cem.json", "rank.json", "report.json"]
    assert not (tmp_path/"id_test"/"planning.partial").exists()
    rank = json.loads((out/"rank.json").read_text()); cem = json.loads((out/"cem.json").read_text())
    assert [r["id"] for r in rank["windows"]] == keep and len(cem["windows"]) == 2
    assert rank["split"] == "id_test" and rank["salt"] == CFG.data.window_salt and len(cem["windows"][0]["rounds"]) == 4
    assert json.loads((out/"report.json").read_text()) == rep and rep["by_duration"]["1"]["top1"] == 1.0
    calls = DistanceToy(inputs, prepared, stored)
    assert pl.run(calls, sub, out, "cpu") == rep and calls.calls == []          # 完整的目录：直接读报告


def test_real_po_jepa_model_runs_one_rank_window(data):
    """v7 协议的 PO-JEPA（动作 7 维）在一个排名窗口上跑通：64 个有限代价，名次在 [1, 64]。"""
    inputs, _, _ = data
    small = json.loads(json.dumps(inputs.windows))
    first = next(iter(small["planning"]["rank"]))
    small["planning"] = {"rank": {first: small["planning"]["rank"][first]}, "cem": []}
    model = droid_model().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    records, _ = pl.rank_split(model, dataclasses.replace(inputs, windows=small), "cpu")
    assert len(records) == 1 and len(records[0]["costs"]) == 64 and all(np.isfinite(records[0]["costs"]))
    assert 1 <= records[0]["r_lo"] <= 64 and len(set(records[0]["costs"])) > 1        # PO-JEPA 看动作
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
