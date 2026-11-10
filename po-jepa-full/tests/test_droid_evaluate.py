"""S11a：DROID 评测的冻结窗口各轴（docs/design/10-evaluation.md §8.1 跳过、§8.2、§8.3、§8.6、§8.7、§8.9；验收 14 §3 S11）。

合成数据根：两个测试集各 3 条回合（行数 33–40，够 d = 32），特征是随机 float16，准备文件是内存里的 `PreparedEpisode`，
冻结窗口用 `windows.frozen_windows` 生成（附加回合数设为 0，不生成规划窗口）。

玩具模型（可手算）：Φ(z, u, T, grid) = a^T·z + (∫u dt)·W + T·(N/256)·v，W 是 [7, E]、v 是 [E] 的固定向量，N 是网格的
token 数。它不满足复合律（a^σ 乘在第一段的动作上）、跨网格也不交换（N 项），所以每个差异都非零。期望值一律用闭式从窗口的
物理动作独立算（不走 split/rollout 代码），再用同一个距离函数比较。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from po_jepa import ln_metrics
from po_jepa.droid import batches as bt
from po_jepa.droid import encode as en
from po_jepa.droid import evaluate as ev
from po_jepa.droid import features as ft
from po_jepa.droid import windows as wn
from po_jepa.droid.actions import droid_control_spec
from po_jepa.droid.config import load_config
from po_jepa.droid.prepare import PreparedEpisode
from po_jepa.geometry import apply_spatial_transfer, build_spatial_transfer, regular_grid
from po_jepa.losses import input_scale, jepa_distance
from test_droid_batches import droid_model
from test_droid_index import CONFIGS

BASE = load_config(CONFIGS/"pilot.yaml")
CFG = dataclasses.replace(BASE, encode=dataclasses.replace(BASE.encode, test_extra_episodes=0))
SPLIT_ROWS = {"id_test": {"A": 34, "B": 36, "C": 40}, "ood_test": {"D": 33, "E": 35, "F": 38}}
OUTCOME = {"A": "success", "B": "failure", "C": "success", "D": "failure", "E": "success", "F": "success"}
SCALE = {"realized": {"position": 0.01, "rotation": 0.05, "gripper": 0.5},
         "commanded": {"position": 0.02, "rotation": 0.04, "gripper": 0.25}}
SYNTHETIC = {"evidence_role": en.SYNTHETIC_ROLE, "encoder_checkpoint_sha256": None, "preprocess_sha256": "p"*64,
             "batch_frames": 32, "upstream_commit": BASE.encode.upstream_commit}
LOSS = CFG.loss_config(time_weight=1e-3, space_weight=1e-3)
A = 0.9
E = ft.WIDTH


def uid(k: str) -> str:
    return f"LAB+0000000{k}+2023-07-07-09h-42m-23s"


def _prepared(u: str, rows: int, seed: int, passed=(True, True)) -> PreparedEpisode:
    rng = np.random.default_rng(seed)
    arrays = {"action_realized": rng.uniform(-0.02, 0.02, (rows - 1, 7)),
              "action_commanded": rng.uniform(-0.02, 0.02, (rows - 1, 7)),
              "state": rng.uniform(-0.5, 0.5, (rows, 7))}
    info = {"uuid": u, "rows": rows, "passed_realized": passed[0], "passed_commanded": passed[1]}
    return PreparedEpisode(info, hashlib.sha256(f"prep-{u}".encode()).hexdigest(), arrays)


class Toy(torch.nn.Module):
    """Φ(z, u, T, grid) = a^T z + (∫u)W + T(N/256)v（逐样本时长、只读有效段）。"""

    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(5)
        self.register_buffer("w", torch.randn(7, E, generator=g)*0.5)
        self.register_buffer("v", torch.randn(E, generator=g)*0.05)

    def forward(self, raw, controls, duration, grid):
        lengths = ((controls.ends - controls.starts)*controls.segment_valid).to(torch.float32)   # [B, J]
        g = (lengths[..., None]*torch.nan_to_num(controls.u_values)).sum(1)                     # [B, 7]
        t = duration.to(torch.float32)
        n = grid.height*grid.width
        return raw*(A**t)[:, None, None] + (g @ self.w)[:, None, :] + (t*n/256)[:, None, None]*self.v


class Identity(torch.nn.Module):
    def forward(self, raw, controls, duration, grid):
        return raw.clone()


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    data = tmp_path_factory.mktemp("droid_eval")
    prepared, manifests, docs = {}, {}, {}
    for shard, keys in SPLIT_ROWS.items():
        entries = []
        for k, rows in keys.items():
            u = uid(k)
            prepared[u] = _prepared(u, rows, 100 + ord(k))
            array = np.random.default_rng(ord(k)).standard_normal((rows, ft.TOKENS, E)).astype(np.float16)
            npy, js = ft.feature_paths(data, shard, u)
            record, js_sha = ft.write_features(npy, js, array, lambda sha, u=u, a=array, s=shard: {
                "schema": ft.FEATURE_SCHEMA, "uuid": u, "shard": s, "rows": a.shape[0], "shape": list(a.shape),
                "dtype": ft.STORAGE_DTYPE, "file_sha256": sha, "row_sha256": ft.row_sha256(a),
                "prepared_sha256": prepared[u].json_sha256, "encoder": SYNTHETIC})
            entries.append({"uuid": u, "rows": rows, "file_sha256": record["file_sha256"], "feature_json_sha256": js_sha})
        manifests[shard] = {"schema": ft.MANIFEST_SCHEMA, "shard": shard, "encoder": SYNTHETIC, "episodes": entries}
        uuids = [uid(k) for k in keys]
        docs[shard] = wn.frozen_windows(shard, uuids, rows={u: prepared[u].rows for u in uuids},
                                        passed={u: True for u in uuids},
                                        prepared_sha256={u: prepared[u].json_sha256 for u in uuids},
                                        splits_sha256="s"*64, config=CFG)
    reader = ft.FeatureReader(data, manifests)
    provenance = bt.training_provenance(CFG, manifests, "f"*64)
    outcomes = {uid(k): o for k, o in OUTCOME.items()}
    return dict(data=data, prepared=prepared, docs=docs, reader=reader, provenance=provenance, outcomes=outcomes,
                manifests=manifests)


def spec(source="realized"):
    return droid_control_spec(SCALE, source=source, config=CFG)


def run_split(root, out, model, split="id_test", config=CFG, prepared=None, bind=None, source="realized"):
    return ev.evaluate_split(out, model=model, split=split, doc=root["docs"][split], prepared=prepared or root["prepared"],
                             outcomes=root["outcomes"], reader=root["reader"], control_spec=spec(source),
                             provenance=root["provenance"], loss_config=LOSS, config=config, device="cpu", bind=bind)


def read_rows(out):
    rows = []
    for p in sorted((out/"rows").glob("*.json")):
        rows += json.loads(p.read_text())["windows"]
    return rows


# ---------------------------------------------------------------- 闭式参照

def stored(root, u, row):
    return torch.from_numpy(np.load(ft.feature_paths(root["data"], shard_of(u), u)[0])[row].astype(np.float32))[None]


def shard_of(u):
    return next(s for s, keys in SPLIT_ROWS.items() if u in [uid(k) for k in keys])


def scaled(root, u, source="realized"):
    """模型单位的每周期动作 float64 [F−1, 7]：物理值 / 尺度（01 §5.3），float32 舍入后再转 float64。"""
    s = np.array([SCALE[source]["position"]]*3 + [SCALE[source]["rotation"]]*3 + [SCALE[source]["gripper"]])
    phys = root["prepared"][u].arrays[f"action_{source}"].astype(np.float32)
    return (phys/s.astype(np.float32)).astype(np.float64)


def integral(u_cycles, lo, hi):
    """∫_lo^hi u dt，u 每周期一段常值（局部时间从 0 起）。"""
    total = np.zeros(7)
    for j in range(len(u_cycles)):
        overlap = min(hi, j + 1) - max(lo, j)
        if overlap > 0:
            total += overlap*u_cycles[j]
    return total


def toy_closed(model, z, u_cycles, parts, n_tokens):
    """闭式 Φ_Π(z)：逐段 x ← a^τ x + (∫_段 u)W + τ(N/256)v。"""
    x = z.double()
    t0 = 0.
    for tau in parts:
        g = torch.from_numpy(integral(u_cycles, t0, t0 + tau))
        x = x*A**tau + (g @ model.w.double())[None, None, :] + tau*n_tokens/256*model.v.double()
        t0 += tau
    return x.float()


def R(h):
    return build_spatial_transfer(regular_grid(16, 16), target_height=h, target_width=h)


def jd(a, b):
    return float(jepa_distance(a, b, config=LOSS).per_sample[0])


def expected_values(root, model, w, donor=None):
    u, d, s = w["uuid"], w["duration"], w["start"]
    z, zt = stored(root, u, s), stored(root, u, s + d)
    uc = scaled(root, u)[s:s + d]
    phi = lambda x, parts, n=256: toy_closed(model, x, uc, parts, n)
    direct = phi(z, (d,))
    sc = math.sqrt(float(input_scale(z)[0]))
    out = {"future:g=16": jd(direct, zt)}
    ids, held, long = CFG.evaluation.durations_id, CFG.evaluation.durations_heldout, CFG.evaluation.durations_long
    if d in ids or d in held:
        half = phi(z, (d/2, d/2))
        out["tcd_ln"] = float(ln_metrics.rms_ln(direct, half)[0])/(d/2)**2
        out["tcd_normalized"] = float((direct.double() - half.double()).square().mean().sqrt())/(d/2)**2/sc
    if d in ids:
        ud = scaled(root, donor["uuid"])[donor["start"]:donor["start"] + d]
        out["delta_act"] = jd(toy_closed(model, z, ud, (d,), 256), zt) - out["future:g=16"]
        for h in (8, 12):
            t = R(h)
            coarse = phi(apply_spatial_transfer(z, t), (d,), h*h)
            fine = apply_spatial_transfer(direct, t)
            out[f"future:g={h}"] = jd(coarse, apply_spatial_transfer(zt, t))
            out[f"scd_ln:h={h}"] = float(ln_metrics.rms_ln(fine, coarse)[0])/d
            out[f"scd_normalized:h={h}"] = float((fine.double() - coarse.double()).square().mean().sqrt())/d/sc
    if d in (3, 12):
        t = R(12)
        out["future:g=12"] = jd(phi(apply_spatial_transfer(z, t), (d,), 144), apply_spatial_transfer(zt, t))
    if d in held:
        comp = phi(z, {3: (1, 2), 12: (8, 4), 16: (8, 8)}[d])
        out |= {"heldout_direct": jd(direct, zt), "heldout_partition": jd(direct, comp), "heldout_rollout": jd(comp, zt)}
    if d in long:
        comp = phi(z, (8.,)*(d//8))
        out |= {"long_direct": jd(direct, zt), "long_composed": jd(comp, zt),
                "long_pd_ln": float(ln_metrics.rms_ln(direct, comp)[0]),
                "long_pd_in": float(ln_metrics.rms_input_scale(direct, comp, z)[0])}
    if d in (4, 8):
        table = {4: [(2, 2), (1,)*4, (.5,)*8], 8: [(4, 4), (2,)*4, (1,)*8, (3, 5)]}[d]
        for parts in table:
            comp = phi(z, parts)
            label = ev.partition_label(tuple(float(p) for p in parts))
            out[f"pd_ln:{label}"] = float(ln_metrics.rms_ln(direct, comp)[0])
            out[f"pd_in:{label}"] = float(ln_metrics.rms_input_scale(direct, comp, z)[0])
        t = R(8)
        for n in (2, 4):
            comp = phi(apply_spatial_transfer(z, t), (d/n,)*n, 64)
            fine = apply_spatial_transfer(direct, t)
            out[f"xpd_ln:n={n}"] = float(ln_metrics.rms_ln(fine, comp)[0])
            out[f"xpd_in:n={n}"] = float(ln_metrics.rms_input_scale(fine, comp, z)[0])
        for parts in {4: [(1, 3), (3, 1)], 8: [(2, 6), (6, 2)]}[d]:
            comp = phi(z, parts)
            label = ev.partition_label(tuple(float(p) for p in parts))
            out[f"tcd_unequal_ln:{label}"] = float(ln_metrics.rms_ln(direct, comp)[0])/(parts[0]*parts[1])
            out[f"tcd_unequal_in:{label}"] = float(ln_metrics.rms_input_scale(direct, comp, z)[0])/(parts[0]*parts[1])
    return out


# ---------------------------------------------------------------- 测试

def test_value_names_follow_the_table():
    """10 §8.2：每个时长只算和它有关的量；d = 4、8 另有 7 个分割、2 个 xpd、2 个不等分 TCD。"""
    n = {d: len(ev.value_names(d, CFG)) for d in (1, 2, 3, 4, 8, 12, 16, 24, 32)}
    assert n == {1: 10, 2: 10, 3: 7, 4: 10 + 6 + 4 + 4, 8: 10 + 8 + 4 + 4, 12: 7, 16: 6, 24: 5, 32: 5}
    assert "delta_act" in ev.value_names(2, CFG) and "delta_act" not in ev.value_names(3, CFG)
    assert "future:g=12" in ev.value_names(12, CFG) and "future:g=12" not in ev.value_names(16, CFG)
    with pytest.raises(ValueError):
        ev.value_names(5, CFG)


def test_toy_model_values_match_the_closed_form(root, tmp_path):
    """每个量对上闭式：控制按周期精确切开、分割的先后顺序、R_8/R_12、替身的动作和本窗口的起点与目标。"""
    model = Toy()
    summary = run_split(root, tmp_path/"id", model)
    doc = root["docs"]["id_test"]
    by_id = {w["id"]: w for w in doc["windows"]}
    rows = read_rows(tmp_path/"id")
    assert len(rows) == len(doc["windows"]) == summary["counts"]["windows"]
    assert {r["duration"] for r in rows} == {1, 2, 3, 4, 8, 12, 16, 24, 32}
    worst = 0.
    for r in rows:
        w = by_id[r["id"]]
        want = expected_values(root, model, w, by_id[doc["donors"][w["id"]]])
        assert sorted(r["values"]) == sorted(ev.value_names(w["duration"], CFG))   # JSON 按键排序
        assert set(r["values"]) == set(want)
        for k, v in want.items():
            assert r["values"][k] == pytest.approx(v, rel=2e-4, abs=2e-6), (r["id"], k)
            worst = max(worst, abs(v))
        assert r["outcome"] == OUTCOME[r["uuid"][11]] and r["triangle_ok"]
    assert worst > 1e-2
    nonzero = {k for r in rows for k, v in r["values"].items() if abs(v) > 1e-6}
    assert {"tcd_ln", "scd_ln:h=8", "pd_ln:3+5", "xpd_ln:n=4", "tcd_unequal_ln:1+3", "heldout_partition",
            "long_pd_in", "delta_act"} <= nonzero


def test_identity_reference(root, tmp_path):
    """identity：未来误差 = d(z, z*_T)（每个网格），所有差异为 0、delta_act 为 0。"""
    run_split(root, tmp_path/"id", Identity(), split="ood_test")
    for r in read_rows(tmp_path/"id"):
        u, d, s = r["uuid"], r["duration"], r["start"]
        z, zt = stored(root, u, s), stored(root, u, s + d)
        assert r["values"]["future:g=16"] == pytest.approx(jd(z, zt), rel=1e-6)
        if d == 2:
            t = R(8)
            assert r["values"]["future:g=8"] == pytest.approx(jd(apply_spatial_transfer(z, t), apply_spatial_transfer(zt, t)), rel=1e-6)
        for k, v in r["values"].items():
            if not (k.startswith("future") or k in ("heldout_direct", "heldout_rollout", "long_direct", "long_composed")):
                assert v == 0., (r["id"], k)


def test_po_jepa_delta_act_uses_the_donor(root):
    """POJEPA 走 v7 action_intervention_margin：等于本窗口起点和目标上、两次前向（替身动作 − 真实动作）的差。"""
    doc = root["docs"]["id_test"]
    group = [w for w in doc["windows"] if w["duration"] == 1][:2]
    by_id = {w["id"]: w for w in doc["windows"]}
    donors = [by_id[doc["donors"][w["id"]]] for w in group]
    assert all(d["uuid"] != w["uuid"] for d, w in zip(donors, group))
    episodes = bt.episode_arrays(root["prepared"], list(root["prepared"]), "realized")
    batch, states = bt.make_eval_batch(group, root["reader"], episodes, spec(), root["provenance"], "cpu")
    donor, _ = bt.make_eval_batch(donors, root["reader"], episodes, spec(), root["provenance"], "cpu")
    model = droid_model().eval()
    values, _ = ev.group_values(model, batch, states, (donor, [True, True]), duration=1, config=CFG, loss_config=LOSS)
    grid = regular_grid(16, 16)
    with torch.no_grad():
        actual = jepa_distance(model(batch.raw_z, batch.controls, batch.duration, grid), batch.raw_target, config=LOSS).per_sample
        alt = jepa_distance(model(batch.raw_z, donor.controls, batch.duration, grid), batch.raw_target, config=LOSS).per_sample
    assert values["delta_act"] == pytest.approx([float(x) for x in alt - actual], abs=1e-6)
    assert values["future:g=16"] == pytest.approx([float(x) for x in actual], rel=1e-6)
    assert all(math.isfinite(v) for vs in values.values() for v in vs)


def test_ac_bind_gets_start_state_and_the_right_reference(root, tmp_path):
    """09 §6.2 的绑定包装（假包装）：起点状态是 state[s]，reference 是本窗口的动作；动作干预另建一个以替身动作为
    reference、起点状态仍是本窗口的包装；分割滚动的每次调用都落在 reference 的范围里。结果和不绑定时相同。"""
    model = Toy()
    calls = []
    doc = root["docs"]["id_test"]
    by_id = {w["id"]: w for w in doc["windows"]}

    def bind(m, start_state, reference):
        assert start_state.dtype == torch.float64 and start_state.shape == (reference.duration.numel(), 7)
        calls.append((start_state.clone(), reference.source_ids))

        def predictor(raw, controls, duration, grid):
            offset = (controls.source_origin_seconds - reference.source_origin_seconds)/controls.control_spec.control_period_seconds
            assert torch.all(offset >= 0) and torch.all(offset + duration <= reference.duration + 1e-12)
            return m(raw, controls, duration, grid)
        return predictor

    true_state = dataclasses.replace(CFG, model=dataclasses.replace(CFG.model, ac_state="true_state"))
    run_split(root, tmp_path/"bound", model, bind=bind)
    run_split(root, tmp_path/"plain", model)
    assert read_rows(tmp_path/"bound") == read_rows(tmp_path/"plain")
    seen_recipient = seen_donor = False
    # 逐组核对：对每个同分布组，恰有一次以替身为 reference 的调用，起点状态等于本组窗口的 state[s]
    for d in CFG.evaluation.durations_id:
        for chunk in bt.chunks([w for w in doc["windows"]]):
            group = [w for w in chunk if w["duration"] == d]
            if not group:
                continue
            want = np.stack([root["prepared"][w["uuid"]].arrays["state"][w["start"]] for w in group])
            own = [c for c in calls if c[1] == tuple(w["id"] for w in group)]
            donor = [c for c in calls if c[1] == tuple(doc["donors"][w["id"]] for w in group)]
            assert len(own) == 1 and len(donor) == 1
            assert np.array_equal(own[0][0].numpy(), want) and np.array_equal(donor[0][0].numpy(), want)
            seen_recipient = seen_donor = True
    assert seen_recipient and seen_donor
    with pytest.raises(ValueError):
        ev.evaluate_job(tmp_path/"job", model=model, condition="vjepa2_ac", seed=1703, job={}, splits=root["docs"],
                        prepared=root["prepared"], outcomes=root["outcomes"], reader=root["reader"], control_spec=spec(),
                        provenance=root["provenance"], loss_config=LOSS, config=true_state, device="cpu")
    with pytest.raises(ValueError):
        ev.evaluate_job(tmp_path/"job2", model=model, condition="po_jepa_full", seed=1703, job={}, splits=root["docs"],
                        prepared=root["prepared"], outcomes=root["outcomes"], reader=root["reader"], control_spec=spec(),
                        provenance=root["provenance"], loss_config=LOSS, config=CFG, device="cpu", bind=bind)


def test_commanded_skips_and_counts(root, tmp_path):
    """`commanded` 配置跳过 passed_commanded 为假的回合（报回合数和窗口数）；替身来自被跳过回合的 delta_act 记 None。"""
    cmd = dataclasses.replace(CFG, action=dataclasses.replace(CFG.action, source="commanded"))
    u = uid("B")
    prepared = {**root["prepared"], u: _prepared(u, 36, 100 + ord("B"), passed=(True, False))}
    prepared[u] = dataclasses.replace(prepared[u], json_sha256=root["prepared"][u].json_sha256)
    summary = run_split(root, tmp_path/"cmd", Toy(), config=cmd, prepared=prepared, source="commanded")
    doc = root["docs"]["id_test"]
    gone = [w for w in doc["windows"] if w["uuid"] == u]
    rows = read_rows(tmp_path/"cmd")
    assert summary["counts"]["skipped_episodes"] == 1 and summary["counts"]["skipped_windows"] == len(gone) > 0
    assert summary["skipped_episodes"] == [u] and not any(r["uuid"] == u for r in rows)
    assert len(rows) == len(doc["windows"]) - len(gone)
    by_id = {w["id"]: w for w in doc["windows"]}
    lost = [r for r in rows if r["duration"] in CFG.evaluation.durations_id and by_id[doc["donors"][r["id"]]]["uuid"] == u]
    assert lost and all(r["values"]["delta_act"] is None for r in lost)
    assert summary["counts"]["delta_act_donor_skipped"] == len(lost)
    kept = [r for r in rows if r["duration"] in CFG.evaluation.durations_id and r not in lost]
    assert all(type(r["values"]["delta_act"]) is float for r in kept)
    # 用 commanded 动作：和闭式（commanded 尺度）一致
    r = next(r for r in rows if r["duration"] == 2)
    w = by_id[r["id"]]
    uc = scaled(root, w["uuid"], "commanded")[w["start"]:w["start"] + 2]
    want = jd(toy_closed(Toy(), stored(root, w["uuid"], w["start"]), uc, (2,), 256), stored(root, w["uuid"], w["start"] + 2))
    assert r["values"]["future:g=16"] == pytest.approx(want, rel=2e-4)


def test_rows_resume_and_cover_each_window_once(root, tmp_path):
    """中断后重跑：已写好的批跳过（字节不变）、缺的和写到一半的批重做（字节与一次跑完相同）；多出来的文件报错。"""
    model = Toy()
    out = tmp_path/"r"
    first = run_split(root, out, model)
    files = sorted((out/"rows").glob("*.json"))
    assert len(files) == first["counts"]["batches"] >= 2
    original = {p.name: p.read_bytes() for p in files}
    files[-1].unlink()
    (out/"rows"/f"{files[-1].name}.partial").write_text("{")
    again = run_split(root, out, model)
    assert again["counts"]["batches_reused"] == len(files) - 1 and again["counts"]["batches_written"] == 1
    assert {p.name: p.read_bytes() for p in sorted((out/"rows").glob("*.json"))} == original
    assert not list((out/"rows").glob("*.partial"))
    assert again["macro"] == first["macro"]
    (out/"rows"/"99999.json").write_text("{}")
    with pytest.raises(ValueError):
        run_split(root, out, model)
    (out/"rows"/"99999.json").unlink()
    body = json.loads(files[0].read_text())
    body["windows"] = body["windows"][1:]
    files[0].unlink()
    files[0].write_text(json.dumps(body))
    with pytest.raises(ValueError):
        run_split(root, out, model)


def test_outcome_macro_means(root, tmp_path):
    """10 §8.7：每个量按回合 outcome 分开汇总一次、合起来一次，都是回合宏平均（先回合内平均）。"""
    summary = run_split(root, tmp_path/"m", Toy())
    rows = read_rows(tmp_path/"m")
    for d, name in ((2, "future:g=16"), (8, "pd_ln:3+5"), (32, "long_composed")):
        for outcome in ("all", "success", "failure"):
            by = {}
            for r in rows:
                if r["duration"] == d and (outcome == "all" or r["outcome"] == outcome):
                    by.setdefault(r["uuid"], []).append(r["values"][name])
            means = [sum(v)/len(v) for v in by.values()]
            got = summary["macro"][str(d)][name][outcome]
            if means:
                assert got["episode_macro_mean"] == pytest.approx(sum(means)/len(means), rel=1e-12)
                assert got["episodes"] == len(by) and got["windows"] == sum(map(len, by.values()))
            else:
                assert got["episode_macro_mean"] is None
    assert summary["macro"]["2"]["future:g=16"]["failure"]["episodes"] == 1        # 只有 B 是 failure


def test_evaluate_job_report_complete_and_checks(root, tmp_path, monkeypatch):
    """评测任务：两个测试集、report.json、COMPLETE.json；重跑直接返回；job.json 不同报错；改了模型状态报错；
    三角不等式有违反时写 FAILED.json；AC 条件把原空间 TCD/SCD 标为不可比。"""
    kwargs = dict(seed=1703, job={"checkpoint": None}, splits=root["docs"], prepared=root["prepared"],
                  outcomes=root["outcomes"], reader=root["reader"], control_spec=spec(), provenance=root["provenance"],
                  loss_config=LOSS, config=CFG, device="cpu")
    class EvalOnly(Toy):
        def forward(self, raw, controls, duration, grid):
            assert not self.training and not torch.is_grad_enabled()
            return super().forward(raw, controls, duration, grid)
    model = EvalOnly()
    report = ev.evaluate_job(tmp_path/"j", model=model, condition="po_jepa_full", **kwargs)
    assert report["status"] == "passed" and set(report["splits"]) == {"id_test", "ood_test"}
    assert report["not_comparable"] == [] and model.training          # 评测时 eval，结束后恢复原模式
    sha = hashlib.sha256((tmp_path/"j"/"report.json").read_bytes()).hexdigest()
    assert json.loads((tmp_path/"j"/"COMPLETE.json").read_text()) == {"report_sha256": sha}
    assert ev.evaluate_job(tmp_path/"j", model=model, condition="po_jepa_full", **kwargs) == report
    with pytest.raises(ValueError):
        ev.evaluate_job(tmp_path/"j2", model=model, condition="po_jepa_full", **{**kwargs, "splits": {"id_test": root["docs"]["id_test"]}})
    ac = ev.evaluate_job(tmp_path/"ac", model=model, condition="vjepa2_ac_small", bind=lambda m, s, r: m, **kwargs)
    assert ac["not_comparable"] == ["tcd_normalized", "scd_normalized:h=8", "scd_normalized:h=12"]
    (tmp_path/"k").mkdir()
    (tmp_path/"k"/"job.json").write_bytes(b'{"condition": "other"}')
    with pytest.raises(ValueError):
        ev.evaluate_job(tmp_path/"k", model=model, condition="po_jepa_full", **kwargs)

    class Drifting(Toy):
        def forward(self, raw, controls, duration, grid):
            self.v.add_(1e-3)
            return super().forward(raw, controls, duration, grid)
    with pytest.raises(RuntimeError):
        ev.evaluate_job(tmp_path/"drift", model=Drifting(), condition="po_jepa_full", **kwargs)
    monkeypatch.setattr(ev.d4, "triangle_holds", lambda a, b, c: False)
    with pytest.raises(ValueError):
        ev.evaluate_job(tmp_path/"tri", model=model, condition="po_jepa_full", **kwargs)
    assert (tmp_path/"tri"/"FAILED.json").exists() and not (tmp_path/"tri"/"COMPLETE.json").exists()
    assert json.loads((tmp_path/"tri"/"report.json").read_text())["status"] == "failed"


# ---------------------------------------------------------------- S11a-fix1（Codex S11a-P1-1、S11a-P2-1）

def other_provenance(root):
    """同一个合成来源，只是某个源码文件的 SHA 不同（假编码器的特征清单 SHA 记 None，换它不改来源）。"""
    prov = json.loads(json.dumps(root["provenance"]))
    name = sorted(prov["source_files_sha256"])[0]
    prov["source_files_sha256"][name] = "0"*64
    assert prov != root["provenance"]
    return prov


def _job_kwargs(root, **over):
    kwargs = dict(seed=1703, job={"checkpoint_sha256": "OLD"}, splits=root["docs"], prepared=root["prepared"],
                  outcomes=root["outcomes"], reader=root["reader"], control_spec=spec(), provenance=root["provenance"],
                  loss_config=LOSS, config=CFG, device="cpu")
    return {**kwargs, **over}


def test_completed_job_is_reused_only_for_the_same_request(root, tmp_path):
    """完成缓存（S11a-P1-1）：同一请求直接返回；换条件、种子、检查点、配置、冻结窗口、准备文件、准备数组（S09b）、outcome、
    来源、两轴开关中任何一项都报错，不返回旧报告；job.json 或 report 里的 job 被改也报错。"""
    out = tmp_path/"j"
    report = ev.evaluate_job(out, model=Toy(), condition="identity", **_job_kwargs(root))
    assert report["job"]["inputs"] == ev.job_inputs(splits=root["docs"], prepared=root["prepared"],
                                                    outcomes=root["outcomes"], provenance=root["provenance"], config=CFG)
    assert ev.evaluate_job(out, model=Toy(), condition="identity", **_job_kwargs(root)) == report
    u = uid("A")
    doc = root["docs"]["id_test"]
    changed = {
        "condition": dict(condition="po_jepa_full"),
        "seed": dict(seed=2026),
        "checkpoint": dict(job={"checkpoint_sha256": "NEW"}),
        "config": dict(config=dataclasses.replace(CFG, config_sha256="0"*64)),
        "windows": dict(splits={**root["docs"], "id_test": {**doc, "windows": doc["windows"][1:]}}),
        "prepared": dict(prepared={**root["prepared"], u: dataclasses.replace(root["prepared"][u], json_sha256="0"*64)}),
        # S09b：准备 JSON 的 SHA 不变、数组本身换了
        "arrays": dict(prepared={**root["prepared"], u: dataclasses.replace(root["prepared"][u], arrays={
            **root["prepared"][u].arrays, "state": root["prepared"][u].arrays["state"] + 1e-3})}),
        "outcomes": dict(outcomes={**root["outcomes"], u: "failure"}),
        "provenance": dict(provenance=other_provenance(root)),
    }
    for name, over in changed.items():
        condition = over.pop("condition", "identity")
        assert ev.job_inputs(**{k: v for k, v in _job_kwargs(root, **over).items()
                                if k in ("splits", "prepared", "outcomes", "provenance", "config")}) != report["job"]["inputs"] \
            or name in ("condition", "seed", "checkpoint", "config"), name
        with pytest.raises(ValueError, match="complete for another request"):
            ev.evaluate_job(out, model=Toy(), condition=condition, **_job_kwargs(root, **over))
    with pytest.raises(ValueError, match="complete for another request"):         # 本次要两轴，旧的没跑
        ev.evaluate_job(out, model=Toy(), condition="identity", extra_reader=SimpleNamespace(manifest_sha256="e"*64),
                        **_job_kwargs(root))
    with pytest.raises(ValueError, match="manifest_sha256"):                     # S09b：附加特征清单要有身份
        ev.evaluate_job(out, model=Toy(), condition="identity", extra_reader=object(), **_job_kwargs(root))
    # 旧任务 identity/OLD，新请求 po_jepa_full/2026/NEW（Codex 的反例）
    with pytest.raises(ValueError, match="complete for another request"):
        ev.evaluate_job(out, model=Toy(), condition="po_jepa_full", **_job_kwargs(root, seed=2026,
                                                                                  job={"checkpoint_sha256": "NEW"}))
    # 只改 job.json（report 和 COMPLETE 不动）、或改 report 的 job 并重签 COMPLETE：都不复用
    job_bytes = (out/"job.json").read_bytes()
    (out/"job.json").write_bytes(job_bytes.replace(b'"OLD"', b'"OTHER"'))
    with pytest.raises(ValueError, match="complete for another request"):
        ev.evaluate_job(out, model=Toy(), condition="identity", **_job_kwargs(root))
    (out/"job.json").write_bytes(job_bytes)
    body = json.loads((out/"report.json").read_text())
    body["job"]["seed"] = 1
    (out/"report.json").write_text(json.dumps(body))
    (out/"COMPLETE.json").write_text(json.dumps({"report_sha256": hashlib.sha256((out/"report.json").read_bytes()).hexdigest()}))
    with pytest.raises(ValueError, match="complete for another request"):
        ev.evaluate_job(out, model=Toy(), condition="identity", **_job_kwargs(root))


def test_unfinished_job_does_not_resume_under_other_inputs(root, tmp_path):
    """没完成的任务：输入身份（来源、outcome）变了不续跑；不变时续跑得到同一份报告。"""
    out = tmp_path/"r"
    report = ev.evaluate_job(out, model=Toy(), condition="identity", **_job_kwargs(root))
    (out/"COMPLETE.json").unlink()
    (out/"report.json").unlink()
    for over in (dict(provenance=other_provenance(root)),
                 dict(outcomes={**root["outcomes"], uid("B"): "success"})):
        with pytest.raises(ValueError, match="differs from this command"):
            ev.evaluate_job(out, model=Toy(), condition="identity", **_job_kwargs(root, **over))
    again = ev.evaluate_job(out, model=Toy(), condition="identity", **_job_kwargs(root))
    assert again["job"] == report["job"] and again["splits"]["id_test"]["counts"]["batches_written"] == 0
    assert all(again["splits"][s]["macro"] == report["splits"][s]["macro"] for s in ("id_test", "ood_test"))


def test_single_window_group_with_a_skipped_donor(root, tmp_path):
    """S11a-P2-1：一批里某个分布内时长只剩一个窗口、它的替身回合被跳过时，这个窗口 delta_act 记 None，其余量照常
    （和整组一起算的值相同），评测不报错。"""
    cmd = dataclasses.replace(CFG, action=dataclasses.replace(CFG.action, source="commanded"))
    u = uid("B")
    prepared = {**root["prepared"], u: dataclasses.replace(_prepared(u, 36, 100 + ord("B"), passed=(True, False)),
                                                           json_sha256=root["prepared"][u].json_sha256)}
    doc = root["docs"]["id_test"]
    by_id = {w["id"]: w for w in doc["windows"]}
    kept, kept_episodes, skipped = ev._selected(doc, prepared, "commanded")
    assert skipped == {u}
    lonely = [w for w in kept if w["duration"] in CFG.evaluation.durations_id and by_id[doc["donors"][w["id"]]]["uuid"] == u]
    healthy = [w for w in kept if w["duration"] in CFG.evaluation.durations_id and w not in lonely]
    assert lonely and healthy
    episodes = bt.episode_arrays(prepared, kept_episodes, "commanded")
    common = dict(model=Toy(), doc=doc, skipped=skipped, reader=root["reader"], episodes=episodes,
                  outcomes=root["outcomes"], control_spec=spec("commanded"), provenance=root["provenance"],
                  loss_config=LOSS, config=cmd, device="cpu")
    full = {r["id"]: r for r in ev.chunk_rows(kept, **common)}
    for w in lonely[:3]:
        [r] = ev.chunk_rows([w], **common)
        assert r["values"]["delta_act"] is None
        assert r["values"]["future:g=16"] is not None and r["triangle_ok"] is True
        want = full[w["id"]]["values"]
        assert set(r["values"]) == set(want)
        for name, v in r["values"].items():
            assert v == pytest.approx(want[name], rel=1e-5, abs=1e-6) if v is not None else want[name] is None, name
    # 同组里另有一个可用窗口时照旧借它的动作占位，可用窗口的 delta_act 是数
    w = next(x for x in lonely if any(y["duration"] == x["duration"] for y in healthy))
    mate = next(y for y in healthy if y["duration"] == w["duration"])
    rows = {r["id"]: r for r in ev.chunk_rows([w, mate], **common)}
    assert rows[w["id"]]["values"]["delta_act"] is None and type(rows[mate["id"]]["values"]["delta_act"]) is float
