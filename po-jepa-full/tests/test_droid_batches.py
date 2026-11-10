"""S07：DROID 训练窗口、批和数据加载器（docs/design/06-droid-windows-and-batches.md §1、§2、§4–§6；验收 14 §3 S07）。

合成的小数据根：4 条回合、两片，特征是随机 float16 [F, 256, 1408]（`features.write_features` 写、手写片清单），准备文件
是内存里的 `PreparedEpisode`（随机的物理动作和状态）。期望值一律独立重算：随机数直接用 PCG64(SeedSequence) 按
06 §2 的抽取顺序重抽，特征行直接 `np.load` 原文件取，动作除以尺度用 float32 手算。
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest
import torch

from po_jepa.droid import batches as bt
from po_jepa.droid import encode as en
from po_jepa.droid import features as ft
from po_jepa.droid import windows as wn
from po_jepa.droid.actions import droid_control_spec
from po_jepa.droid.config import load_config
from po_jepa.droid.prepare import PreparedEpisode
from po_jepa.runtime import cache, checks, comparison, training
from test_droid_index import CONFIGS

BASE = load_config(CONFIGS/"pilot.yaml")
CFG = dataclasses.replace(BASE, training=dataclasses.replace(BASE.training, batch=8, ac_batch_clips=4))
ROWS = {"A": 17, "B": 20, "C": 23, "D": 18}
SHARDS = {"train-000": ["A", "B", "C"], "train-001": ["D"]}
SCALE = {"realized": {"position": 0.01, "rotation": 0.05, "gripper": 0.5},
         "commanded": {"position": 0.02, "rotation": 0.04, "gripper": 0.25}}
SYNTHETIC = {"evidence_role": en.SYNTHETIC_ROLE, "encoder_checkpoint_sha256": None, "preprocess_sha256": "p"*64,
             "batch_frames": 32, "upstream_commit": BASE.encode.upstream_commit}


def uid(k: str) -> str:
    return f"LAB+0000000{k}+2023-07-07-09h-42m-23s"


def _prepared(u: str, rows: int, seed: int, passed=(True, True)) -> PreparedEpisode:
    rng = np.random.default_rng(seed)
    arrays = {"action_realized": rng.uniform(-0.02, 0.02, (rows - 1, 7)),
              "action_commanded": rng.uniform(-0.02, 0.02, (rows - 1, 7)),
              "state": rng.uniform(-0.5, 0.5, (rows, 7))}
    info = {"uuid": u, "rows": rows, "passed_realized": passed[0], "passed_commanded": passed[1]}
    return PreparedEpisode(info, hashlib.sha256(f"prep-{u}".encode()).hexdigest(), arrays)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """(数据根, 片清单, {uuid: PreparedEpisode}, {uuid: 原 float16 数组})。"""
    data = tmp_path_factory.mktemp("droid_batches")
    prepared, stored, manifests = {}, {}, {}
    for i, (shard, keys) in enumerate(SHARDS.items()):
        entries = []
        for k in keys:
            u, rows = uid(k), ROWS[k]
            prepared[u] = _prepared(u, rows, 100 + ord(k))
            array = np.random.default_rng(ord(k)).standard_normal((rows, ft.TOKENS, ft.WIDTH)).astype(np.float16)
            npy, js = ft.feature_paths(data, shard, u)
            record, js_sha = ft.write_features(npy, js, array, lambda sha, u=u, a=array, s=shard: {
                "schema": ft.FEATURE_SCHEMA, "uuid": u, "shard": s, "rows": a.shape[0], "shape": list(a.shape),
                "dtype": ft.STORAGE_DTYPE, "file_sha256": sha, "row_sha256": ft.row_sha256(a),
                "prepared_sha256": prepared[u].json_sha256, "encoder": SYNTHETIC})
            entries.append({"uuid": u, "rows": rows, "file_sha256": record["file_sha256"], "feature_json_sha256": js_sha})
            stored[u] = array
        manifests[shard] = {"schema": ft.MANIFEST_SCHEMA, "shard": shard, "encoder": SYNTHETIC, "episodes": entries}
    return data, manifests, prepared, stored


@pytest.fixture(scope="module")
def setup(root):
    data, manifests, prepared, stored = root
    reader = ft.FeatureReader(data, manifests)
    episodes = bt.episode_arrays(prepared, list(prepared), "realized")
    spec = droid_control_spec(SCALE, source="realized", config=CFG)
    provenance = bt.training_provenance(CFG, manifests, "f"*64)
    return reader, episodes, spec, provenance


def train_set():
    return [uid(k) for k in ROWS]


def rows_of():
    return {uid(k): r for k, r in ROWS.items()}


def gen(seed, step, stream):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, step, stream])))


# ---------------------------------------------------------------- 06 §1、§2

def test_step_rng_and_training_windows_follow_the_draw_order():
    """step_rng 只由（种子，步数，流）决定；每行依次抽回合、时长、起点（独立重抽核对）；起点覆盖 [0, F − 1 − d] 的两端。"""
    a, b = wn.step_rng(1703, 5, 0), gen(1703, 5, 0)
    assert np.array_equal(a.integers(1 << 30, size=8), b.integers(1 << 30, size=8))
    eps, rows = train_set(), rows_of()
    seen = {}
    for step in range(300):
        got = wn.training_windows(eps, rows, seed=1703, step=step, config=CFG)
        assert got == wn.training_windows(eps, rows, seed=1703, step=step, config=CFG)
        g = gen(1703, step, 0)
        want = []
        for _ in range(CFG.training.batch):
            u = eps[int(g.integers(4))]
            d = [1, 2, 4, 8][int(g.integers(4))]
            s = int(g.integers(rows[u] - d))
            want.append({"id": f"{u}|d{d}|s{s}", "uuid": u, "duration": d, "start": s})
        assert got == want
        for w in got:
            assert 0 <= w["start"] <= rows[w["uuid"]] - 1 - w["duration"]
            seen.setdefault((w["uuid"], w["duration"]), set()).add(w["start"])
    u, d = uid("A"), 8
    assert min(seen[(u, d)]) == 0 and max(seen[(u, d)]) == ROWS["A"] - 1 - d
    assert wn.training_windows(eps, rows, seed=1703, step=0, config=CFG) != wn.training_windows(eps, rows, seed=1704, step=0, config=CFG)
    assert wn.training_windows(eps, rows, seed=1703, step=0, config=CFG) != wn.training_windows(eps, rows, seed=1703, step=1, config=CFG)
    assert len(wn.training_windows(eps, rows, seed=1703, step=0, config=BASE)) == 64
    assert len(wn.ac_clips(eps, rows, seed=1703, step=0, config=BASE)) == 16
    with pytest.raises(ValueError):
        wn.step_rng(1703, -1, 0)
    with pytest.raises(ValueError):
        wn.training_windows(eps, {**rows, uid("A"): 8}, seed=1, step=0, config=CFG)   # F = 8 不够 d = 8


def test_ac_clips_use_stream_1_and_stay_inside_the_episode():
    eps, rows = train_set(), rows_of()
    starts = set()
    for step in range(300):
        got = wn.ac_clips(eps, rows, seed=1703, step=step, config=CFG)
        g = gen(1703, step, 1)
        want = []
        for _ in range(CFG.training.ac_batch_clips):
            u = eps[int(g.integers(4))]
            s = int(g.integers(rows[u] - 8))
            want.append({"id": f"{u}|clip|s{s}", "uuid": u, "start": s, "frames": 9})
        assert got == want
        starts |= {w["start"] for w in got if w["uuid"] == uid("A")}
    assert min(starts) == 0 and max(starts) == ROWS["A"] - 9


def test_train_episodes_head_then_drop_failed_keep_order():
    order = [f"LAB+{i:08d}+2023-07-07-09h-42m-23s" for i in range(10)]
    splits = {"train_order": order, "train_failures": [order[1], order[6]]}
    cfg = dataclasses.replace(CFG, data=dataclasses.replace(CFG.data, train_episodes=6))
    prepared = {u: _prepared(u, 20, i, passed=(i not in (2, 7), True)) for i, u in enumerate(order)}
    assert wn.train_episodes(splits, prepared, cfg) == [order[i] for i in (0, 1, 3, 4, 5)]
    off = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, use_failures=False))
    assert wn.train_episodes(splits, prepared, off) == [order[i] for i in (0, 3, 4, 5)]   # 去掉失败回合后前 6 条是 0,2,3,4,5,7，再去掉没通过的 2、7
    commanded = dataclasses.replace(cfg, action=dataclasses.replace(cfg.action, source="commanded"))
    assert wn.train_episodes(splits, prepared, commanded) == order[:6]
    with pytest.raises(ValueError):
        wn.train_episodes(splits, {u: p for u, p in prepared.items() if u != order[4]}, cfg)
    assert wn.list_sha256(order[:2]) == hashlib.sha256(f"{order[0]}\n{order[1]}".encode()).hexdigest()


# ---------------------------------------------------------------- 06 §4

WINDOWS = [("A", 1, 0), ("B", 8, 11), ("C", 3, 19), ("D", 4, 2), ("A", 2, 14)]


def wins(spec=WINDOWS):
    return [{"id": wn.window_id(uid(k), d, s), "uuid": uid(k), "duration": d, "start": s} for k, d, s in spec]


def droid_model():
    """v7 协议的 PO-JEPA，动作维数改成 7（模型核心本来就收任意维数）。"""
    m = checks.load_protocol()["model"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1703)
        return checks.POJEPA(checks.ProjectionConfig(**m["projection"]), checks.GraphSpectrum(checks.GraphConfig(**m["graph"])),
                             checks.ModalDynamics(checks.DynamicsConfig(**{**m["dynamics"], "action_dim": 7})),
                             checks.IntegratorConfig(**m["integrator"]))


def test_training_batch_values_and_v7_checks(root, setup):
    _, _, prepared, stored = root
    reader, episodes, spec, provenance = setup
    batch = bt.make_training_batch(wins(), reader, episodes, spec, provenance, "cpu")
    scale = np.array([0.01]*3 + [0.05]*3 + [0.5], dtype=np.float32)
    assert batch.sample_ids == tuple(w["id"] for w in wins())
    assert batch.controls.source_ids == batch.sample_ids
    assert batch.controls.u_values.shape == (5, 8, 7)
    for b, (k, d, s) in enumerate(WINDOWS):
        u = uid(k)
        assert torch.equal(batch.raw_z[b], torch.from_numpy(stored[u][s].astype(np.float32)))
        assert torch.equal(batch.raw_target[b], torch.from_numpy(stored[u][s + d].astype(np.float32)))
        phys = prepared[u].arrays["action_realized"][s:s + d].astype(np.float32)
        assert torch.equal(batch.controls.u_values[b, :d], torch.from_numpy(phys/scale))
        assert torch.equal(batch.controls.starts[b, :d], torch.arange(d, dtype=torch.float64))
        assert torch.equal(batch.controls.ends[b, :d], torch.arange(1, d + 1, dtype=torch.float64))
        assert batch.controls.segment_valid[b].tolist() == [j < d for j in range(8)]
        assert batch.controls.n_segments[b].item() == d
        assert batch.observed_time_seconds[b].item() == float(s) and batch.target_time_seconds[b].item() == float(s + d)
        assert batch.controls.source_origin_seconds[b].item() == float(s)
        assert batch.duration[b].item() == d and batch.tau[b].item() == d/2 and batch.sigma[b].item() == d/2
    assert batch.duration.dtype == torch.float64 and batch.raw_z.dtype == torch.float32 and not batch.raw_z.requires_grad
    assert (batch.transfer.source.height, batch.transfer.source.width) == (16, 16)
    assert (batch.transfer.target.height, batch.transfer.target.width) == (8, 8)
    model = droid_model()
    training._batch(batch, model)
    _tf_checks(batch, model)


def _tf_checks(batch, model, monkeypatch=None):
    """`comparison._tf_batch` 的全部检查。S08b 之前 `baselines.validate_inputs` 的最后一句只许 2 维动作（09 §6.1 才放宽），
    这里只容忍那一条（它是函数的最后一句，前面的检查都已通过），其余照常。"""
    from po_jepa import baselines
    original = baselines.validate_inputs

    def tolerant(*args, **kwargs):
        try:
            original(*args, **kwargs)
        except ValueError as error:
            if str(error) != "F05 requires two control channels":
                raise
    baselines.validate_inputs = tolerant
    try:
        comparison._tf_batch(batch, model)
    finally:
        baselines.validate_inputs = original


def test_po_jepa_train_step_runs_on_a_droid_batch(setup):
    """端到端：v7 的 `training.train_step`（五次 Φ、三项损失）在 DROID 的 7 维批上跑一步，损失有限。"""
    reader, episodes, spec, provenance = setup
    batch = bt.make_training_batch(wins([("A", 1, 0), ("B", 2, 5), ("C", 4, 3), ("D", 8, 9)]), reader, episodes,
                                   spec, provenance, "cpu")
    model = droid_model()
    optimizer = training.make_optimizer(model)
    loss_config = CFG.loss_config(time_weight=1e-3, space_weight=1e-3)
    result = training.train_step(model, optimizer, batch, loss_config=loss_config, completed_steps=0, condition="full")
    assert result.completed_steps == 1 and all(np.isfinite(v) for v in result.loss_values.values() if v is not None)


def test_padding_is_not_read_and_window_errors(root, setup):
    _, manifests, prepared, _ = root
    reader, episodes, spec, provenance = setup
    u = uid("B")
    spec_b = wins([("B", 8, 10), ("A", 1, 0)])
    poisoned = dict(episodes)
    a = episodes[u].actions.copy()
    a[10 + 8] = np.nan                                     # 窗口 (B, d=8, s=10) 之后的那一步
    poisoned[u] = dataclasses.replace(episodes[u], actions=a)
    bt.make_training_batch(spec_b, reader, poisoned, spec, provenance, "cpu")
    a = episodes[u].actions.copy()
    a[10 + 7] = np.nan                                     # 窗口里的最后一步
    poisoned[u] = dataclasses.replace(episodes[u], actions=a)
    with pytest.raises(ValueError):
        bt.make_training_batch(spec_b, reader, poisoned, spec, provenance, "cpu")
    for bad in ([("B", 8, 12)], [("A", 0, 3)]):                         # s + d > F − 1；d = 0
        with pytest.raises(ValueError):
            bt.make_training_batch(wins(bad), reader, episodes, spec, provenance, "cpu")
    w = wins([("A", 2, 3)])
    w[0]["start"] = 4                                                     # 编号和起点对不上
    with pytest.raises(ValueError):
        bt.make_training_batch(w, reader, episodes, spec, provenance, "cpu")
    other = dict(episodes)
    other[uid("A")] = dataclasses.replace(episodes[uid("A")], prepared_sha256="0"*64)   # 特征出自另一份准备文件
    with pytest.raises(ValueError):
        bt.make_training_batch(wins([("A", 2, 3)]), reader, other, spec, provenance, "cpu")
    with pytest.raises(ValueError):
        bt.episode_arrays({**prepared, uid("A"): _prepared(uid("A"), 17, 1, passed=(False, True))}, [uid("A")], "realized")
    commanded = bt.episode_arrays(prepared, [uid("A")], "commanded")
    assert np.array_equal(commanded[uid("A")].actions, prepared[uid("A")].arrays["action_commanded"])


def test_eval_batch_start_state_and_chunks(root, setup):
    _, _, prepared, _ = root
    reader, episodes, spec, provenance = setup
    batch, states = bt.make_eval_batch(wins(), reader, episodes, spec, provenance, "cpu")
    assert states.dtype == torch.float64 and states.shape == (5, 7)
    for b, (k, d, s) in enumerate(WINDOWS):
        assert torch.equal(states[b], torch.from_numpy(prepared[uid(k)].arrays["state"][s]))
    assert torch.equal(batch.raw_z, bt.make_training_batch(wins(), reader, episodes, spec, provenance, "cpu").raw_z)
    parts = bt.chunks(list(range(130)))
    assert [len(p) for p in parts] == [64, 64, 2] and sum(parts, []) == list(range(130))


def test_ac_clip_batch(root, setup):
    _, _, prepared, stored = root
    reader, episodes, _, _ = setup
    clips = [{"id": wn.clip_id(uid(k), s), "uuid": uid(k), "start": s, "frames": 9} for k, s in (("A", 0), ("C", 14), ("D", 5))]
    batch = bt.make_ac_clip_batch(clips, reader, episodes, "cpu")
    assert batch.features.shape == (3, 9, 256, 1408) and batch.features.dtype == torch.float32
    assert batch.actions.shape == (3, 8, 7) and batch.actions.dtype == torch.float32
    assert batch.states.shape == (3, 9, 7) and batch.states.dtype == torch.float32
    for b, (k, s) in enumerate((("A", 0), ("C", 14), ("D", 5))):
        u = uid(k)
        assert torch.equal(batch.features[b], torch.from_numpy(stored[u][s:s + 9].astype(np.float32)))
        assert torch.equal(batch.actions[b], torch.from_numpy(prepared[u].arrays["action_realized"][s:s + 8].astype(np.float32)))
        assert torch.equal(batch.states[b], torch.from_numpy(prepared[u].arrays["state"][s:s + 9].astype(np.float32)))
    with pytest.raises(ValueError):
        bt.make_ac_clip_batch([{"id": wn.clip_id(uid("A"), 9), "uuid": uid("A"), "start": 9, "frames": 9}], reader, episodes, "cpu")


# ---------------------------------------------------------------- 06 §5、§6

def test_training_provenance(root):
    _, manifests, _, _ = root
    p = bt.training_provenance(CFG, manifests, "f"*64)
    assert p["evidence_role"] == en.SYNTHETIC_ROLE and p["split"] == "development_smoke"
    assert all(p[k] is None for k in ("encoder_checkpoint_sha256", "preprocess_sha256", "feature_manifest_sha256", "protocol_sha256"))
    assert "src/po_jepa/droid/batches.py" in p["source_files_sha256"] and "src/po_jepa/runtime/training.py" in p["source_files_sha256"]
    assert p["source_files_sha256"]["src/po_jepa/droid/windows.py"] == cache.file_sha(cache.ROOT/"src/po_jepa/droid/windows.py")
    real = {"evidence_role": en.REAL_ROLE, "encoder_checkpoint_sha256": "c"*64, "preprocess_sha256": cache.v7_preprocess_sha256()}
    p = bt.training_provenance(CFG, {s: {**m, "encoder": real} for s, m in manifests.items()}, "f"*64)
    assert (p["evidence_role"], p["split"], p["encoder_checkpoint_sha256"], p["feature_manifest_sha256"], p["protocol_sha256"]) \
        == ("droid_training", "train", "c"*64, "f"*64, CFG.config_sha256)
    assert p["preprocess_sha256"] == cache.v7_preprocess_sha256()
    mixed = {"train-000": {**manifests["train-000"], "encoder": real}, "train-001": manifests["train-001"]}
    with pytest.raises(ValueError):
        bt.training_provenance(CFG, mixed, "f"*64)
    with pytest.raises(ValueError):
        bt.training_provenance(CFG, {s: {**m, "encoder": {**real, "preprocess_sha256": "p"*64}} for s, m in manifests.items()}, "f"*64)


@pytest.mark.parametrize("kind", ["train", "ac"])
def test_loader_is_independent_of_workers_and_resumes(root, setup, kind):
    """0、1、2 个工作进程产出的批逐字节相同；从第 k 步开始产出的批和从头产出的第 k 步相同；和直接组装的批相同。"""
    data, manifests, _, _ = root
    reader, episodes, spec, provenance = setup

    def run(start, workers):
        ds = bt.StepBatches(kind, data, manifests, train_set(), episodes, seed=1703, start=start, stop=6, config=CFG)
        return [(int(s), bt.batch_sha256(item)) for s, item in bt.loader(ds, workers=workers, prefetch=2)]
    full = run(0, 0)
    assert [s for s, _ in full] == list(range(6)) and len({h for _, h in full}) == 6
    assert run(0, 1) == full and run(0, 2) == full
    assert run(3, 2) == full[3:] and run(3, 0) == full[3:]
    for step, h in full:
        if kind == "train":
            item = bt.load_windows(wn.training_windows(train_set(), rows_of(), seed=1703, step=step, config=CFG), reader, episodes)
        else:
            item = bt.load_ac_clips(wn.ac_clips(train_set(), rows_of(), seed=1703, step=step, config=CFG), reader, episodes)
        assert bt.batch_sha256(item) == h


def test_training_batches_main_side(root, setup):
    data, manifests, _, _ = root
    reader, episodes, spec, provenance = setup
    ds = bt.StepBatches("train", data, manifests, train_set(), episodes, seed=1703, start=2, stop=5, config=CFG)
    got = list(bt.training_batches(ds, spec, provenance, "cpu", workers=2, prefetch=2))
    assert [s for s, _ in got] == [2, 3, 4]
    for step, batch in got:
        want = bt.make_training_batch(wn.training_windows(train_set(), rows_of(), seed=1703, step=step, config=CFG),
                                      reader, episodes, spec, provenance, "cpu")
        assert batch.sample_ids == want.sample_ids
        for name in ("raw_z", "raw_target", "duration", "observed_time_seconds", "target_time_seconds"):
            assert torch.equal(getattr(batch, name), getattr(want, name))
        assert torch.equal(batch.controls.u_values, want.controls.u_values)
    ac = bt.StepBatches("ac", data, manifests, train_set(), episodes, seed=1703, start=0, stop=2, config=CFG)
    clips = list(bt.training_batches(ac, spec, provenance, "cpu", workers=0))
    assert [s for s, _ in clips] == [0, 1] and clips[0][1].features.shape == (4, 9, 256, 1408)
