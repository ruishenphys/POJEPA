"""S25c：长时程测试集接编码、缓存和滚动评测（docs/segments/S25c.md）。

假特征用 tools/encode_v7.py --fake-encoder（子进程，和 addon_d4_smoke 相同），读取时把 synthetic 标记换成 real
（同 addon_d4_smoke.enter_new_tree 的 as_real）。数据集是 double_well:40（测试段 36–39），只编码测试段。
"""

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from po_jepa import datasets, systems
from po_jepa.baselines import IdentityPredictor
from po_jepa.runtime import cache, comparison, data, long_cache, long_rollout, v7jobs
from po_jepa.runtime.addon import d4, long_window
from test_training import e06_config

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"src"
BASE = datasets.parse("double_well:40")
LONG = datasets.parse("double_well~horizon=64:40")
LONG128 = datasets.parse("double_well~horizon=128:40")
TEST_IDS = tuple(range(36, 40))


def _clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")}
    env.update(PYTHONPATH=str(SRC), OMP_NUM_THREADS="2")
    return env


def _encode(out, variant="", splits="test"):
    args = ["--system", "double_well", "--trajectories", "40", "--output", str(out)]
    if variant:
        args += ["--variant", variant]
    for mode, extra in (("encode", ["--fake-encoder", "--device", "cpu", "--splits", splits]), ("finalize", ["--splits", splits])):
        r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/encode_v7.py"), mode, *args, *extra],
                           env=_clean_env(), capture_output=True, text=True)
        if r.returncode:
            return r
    return r


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    root = tmp_path_factory.mktemp("s25c")
    for name, variant in (("base", ""), ("long", "horizon=64")):
        r = _encode(root/name, variant)
        assert r.returncode == 0, r.stderr
    return root/"base", root/"long"


@pytest.fixture
def real(monkeypatch):
    """读取时当作真实编码器的缓存（只改内存里的 manifest），并清掉进程内的核验缓存。"""
    original = cache._read_v7_cache

    def as_real(path, m, sha):
        m = copy.deepcopy(m); g = m["generation"]
        g.update(evidence_role="real_frozen_encoder", encoder_checkpoint_sha256=cache.CHECKPOINT_SHA,
                 encoder_load_report=dict(sha256=cache.CHECKPOINT_SHA, encoder_key="target_encoder", strict=True,
                                          missing_keys=[], unexpected_keys=[]))
        return original(path, m, sha)

    original_long = long_cache.check_manifest

    def as_real_long(path, m, sha, spec):
        m = copy.deepcopy(m); g = m["generation"]
        g.update(evidence_role="real_frozen_encoder", encoder_checkpoint_sha256=cache.CHECKPOINT_SHA,
                 encoder_load_report=dict(sha256=cache.CHECKPOINT_SHA, encoder_key="target_encoder", strict=True,
                                          missing_keys=[], unexpected_keys=[]))
        return original_long(path, m, sha, spec)
    monkeypatch.setattr(cache, "_read_v7_cache", as_real)
    monkeypatch.setattr(long_cache, "check_manifest", as_real_long)
    monkeypatch.setattr(cache, "_VERIFIED_CACHE", {})
    monkeypatch.setattr(long_cache, "_VERIFIED", {})


@pytest.fixture
def active():
    """切换当前数据集，测试结束恢复。"""
    old = datasets.active()
    yield lambda spec: datasets._set_for_tests(spec)
    datasets._set_for_tests(old)


# —— 编码 ——

def test_encode_long_horizon_needs_test_split_only(tmp_path):
    r = _encode(tmp_path/"enc", "horizon=64", splits="train,test")
    assert r.returncode != 0 and "--splits test" in r.stderr and "长时程" in r.stderr and not (tmp_path/"enc").exists()


def test_long_cache_layout(caches):
    _, long = caches
    report = json.loads((long/"report.json").read_text())
    assert report["variant"] == "horizon=64" and report["splits_encoded"] == ["test"]
    assert sorted(r["trajectory_id"] for r in report["records"]) == list(TEST_IDS)
    m = json.loads((long/"0037"/"manifest.json").read_text())
    assert m["shape"] == [65, 256, 1408] and len(m["frames"]) == 65
    assert [f["tick"] for f in m["frames"]] == list(range(65)) and m["frames"][64]["time_seconds"] == 16.
    assert m["controls"]["ends_ticks"][-1] == 64 and m["controls"]["starts_ticks"][0] == 0
    assert np.load(long/"0037"/"raw_z.npy", mmap_mode="r").shape == (65, 256, 1408)


def test_long_cache_prefix_equals_the_33_frame_cache(caches):
    """前 33 帧：画面哈希、动作（截到 32 的段之前）和特征都与主实验缓存相同。"""
    base, long = caches
    for tid in TEST_IDS:
        a = json.loads((base/f"{tid:04d}"/"manifest.json").read_text())
        b = json.loads((long/f"{tid:04d}"/"manifest.json").read_text())
        assert [f["image_sha256"] for f in a["frames"]] == [f["image_sha256"] for f in b["frames"][:33]]
        za = np.load(base/f"{tid:04d}"/"raw_z.npy"); zb = np.load(long/f"{tid:04d}"/"raw_z.npy")
        assert np.array_equal(za, zb[:33])
        ca, cb = a["controls"], b["controls"]
        k = next(i for i, e in enumerate(ca["ends_ticks"]) if e == 32)     # 32 tick 的最后一段可能被截断
        assert ca["starts_ticks"][:k+1] == cb["starts_ticks"][:k+1] and ca["values"][:k+1] == cb["values"][:k+1]
        assert ca["ends_ticks"][:k] == cb["ends_ticks"][:k]


def test_long_cache_tail_is_the_rendered_long_trajectory(caches, active):
    """tick 32 以后的帧：manifest 的画面哈希 = 长时程数据集 frames() 的画面，特征 = 假编码器对这些画面的输出（float16）。"""
    import hashlib
    _, long = caches
    spec = importlib.util.spec_from_file_location("encode_v7_tool", ROOT/"tools/encode_v7.py")
    enc = importlib.util.module_from_spec(spec); spec.loader.exec_module(enc)
    active(LONG)
    _, frames = systems.get().frames(37)
    m = json.loads((long/"0037"/"manifest.json").read_text())
    assert frames.shape[0] == 65
    assert [f["image_sha256"] for f in m["frames"]] == [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]
    x = torch.from_numpy(frames[[33, 50, 64]]).permute(0, 3, 1, 2).float()/255.
    expected = enc._fake_encoder(torch)(x).numpy().astype(np.float16)
    stored = np.load(long/"0037"/"raw_z.npy")[[33, 50, 64]]
    assert np.allclose(stored.astype(np.float32), expected.astype(np.float32), atol=2e-3)
    assert not np.allclose(stored[2].astype(np.float32), stored[0].astype(np.float32), atol=2e-3)


# —— 缓存读取 ——

def test_long_cache_reads_only_as_the_long_dataset(caches, real, active):
    base, long = caches
    active(BASE)
    with pytest.raises(ValueError):
        data.approved_cache_metadata(long, splits=("test",))
    with pytest.raises(ValueError, match="mismatch"):
        cache.read_window(long/"0036", start_tick=0, duration_cycles=1., expected_split="test", device="cpu")
    with pytest.raises(ValueError, match="not a long-horizon"):
        long_cache.read_complete(long/"0036")
    active(LONG128)                                    # horizon 不同（帧数 129 ≠ 65）也不收
    with pytest.raises(ValueError):
        data.approved_cache_metadata(long, splits=("test",))
    with pytest.raises(ValueError, match="mismatch"):
        long_cache.read_complete(long/"0036")
    active(LONG)
    assert data.approved_cache_metadata(long, splits=("test",))["variant"] == "horizon=64"
    raw, m, _ = long_cache.read_complete(long/"0036")
    assert raw.shape == (65, 256, 1408) and m["trajectory_id"] == 36
    with pytest.raises(ValueError, match="mismatch"):  # 普通窗口（cache.py，写死 33 帧）不收长时程缓存
        cache.read_window(long/"0036", start_tick=0, duration_cycles=1., expected_split="test", device="cpu")
    with pytest.raises(ValueError, match="mismatch"):  # 33 帧的缓存当成长时程读也不收
        long_cache.read_complete(base/"0036")
    active(datasets.parse("double_well~horizon=64:2000"))   # 刚核过的结果不跨数据集复用（同 horizon、条数不同）
    with pytest.raises(ValueError, match="mismatch"):
        long_cache.read_complete(long/"0036")


def test_long_cache_checks_like_cache(caches, real, active, tmp_path):
    """长时程核法里照抄 cache 的几条：特征 SHA、manifest SHA、动作随机流、逐帧来源。"""
    import shutil
    _, long = caches
    active(LONG)
    for name, edit in (("feature", lambda d: (d/"raw_z.npy").write_bytes((d/"raw_z.npy").read_bytes()[:-2]+b"\0\1")),
                       ("manifest", lambda d: (d/"manifest.json").write_text((d/"manifest.json").read_text()+" "))):
        d = tmp_path/name; shutil.copytree(long/"0037", d); edit(d)
        with pytest.raises(ValueError, match="SHA mismatch"):
            long_cache.read_complete(d)
    for name, change in (("controls", lambda m: m["controls"]["values"][0].__setitem__(0, .123)),
                         ("frames", lambda m: m["frames"][40].__setitem__("tick", 41)),
                         ("shape", lambda m: m.__setitem__("shape", [33, 256, 1408])),
                         ("variant", lambda m: m.__setitem__("variant", "gain=2,horizon=64")),
                         ("count", lambda m: m.__setitem__("dataset_count", 2000))):
        m = json.loads((long/"0037"/"manifest.json").read_text()); change(m)
        with pytest.raises(ValueError):
            long_cache.check_manifest(long/"0037", m, "sha", datasets.active())


def test_long_cache_refuses_synthetic_features(caches, active):
    """不打 as_real 补丁：假编码器写的缓存（synthetic）不收。"""
    _, long = caches
    active(LONG)
    m = json.loads((long/"0037"/"manifest.json").read_text())
    assert m["generation"]["evidence_role"] == "synthetic_development_only"
    with pytest.raises(ValueError, match="provenance"):
        long_cache.check_manifest(long/"0037", m, "sha", datasets.active())
    g = m["generation"]
    g.update(encoder_checkpoint_sha256=cache.CHECKPOINT_SHA, encoder_load_report=dict(sha256=cache.CHECKPOINT_SHA,
             encoder_key="target_encoder", strict=True, missing_keys=[], unexpected_keys=[]))
    with pytest.raises(ValueError, match="provenance"):           # 只差 evidence_role 也不收
        long_cache.check_manifest(long/"0037", m, "sha", datasets.active())


@pytest.mark.parametrize("t", [16., 24., 32.])
def test_long_windows_equal_d4_windows_on_the_main_cache(caches, real, active, t):
    """长时程缓存上起点 0、时长 ≤ 32 的窗口（long_cache 那条路、LongControls）与主实验缓存上 D4 的窗口逐位相同。"""
    base, long = caches
    active(BASE)
    a = long_window.read_long_window(base/"0038", start_tick=0, duration_cycles=t, expected_split="test", device="cpu")
    active(LONG)
    b = long_window.read_long_window(long/"0038", start_tick=0, duration_cycles=t, expected_split="test", device="cpu",
                                     allowed=long_rollout.window_times(64))
    assert torch.equal(a.inputs.raw_z, b.inputs.raw_z) and torch.equal(a.target.raw_z, b.target.raw_z)
    for field in ("starts", "ends", "u_values", "n_segments", "segment_valid", "source_origin_seconds", "duration"):
        assert torch.equal(getattr(a.inputs.controls, field), getattr(b.inputs.controls, field)), field
    assert a.inputs.controls.source_ids == b.inputs.controls.source_ids and a.target.image_sha256 == b.target.image_sha256
    assert a.provenance["actual_manifest_sha256"] != b.provenance["actual_manifest_sha256"]


def test_window_controls_long_controls_checks(active):
    active(LONG128)
    m = dict(trajectory_id=36, controls=dict(starts_ticks=list(range(128)), ends_ticks=list(range(1, 129)),
                                              values=[[.001*k, 0.] for k in range(128)], units="simulation_position_per_second"))
    c = long_cache.window_controls(m, 100, 24., "cpu")
    assert float(c.duration[0]) == 24. and int(c.n_segments[0]) == 24
    assert torch.allclose(c.u_values[0, :, 0].double(), torch.tensor([.001*k for k in range(100, 124)], dtype=torch.float64))
    bad = copy.deepcopy(m); bad["controls"]["ends_ticks"][-1] = 127       # 不覆盖 [0, horizon)
    with pytest.raises(ValueError):
        long_cache.window_controls(bad, 0, 16., "cpu")
    bad = copy.deepcopy(m); bad["controls"]["ends_ticks"] = list(range(1, 97)); bad["controls"]["starts_ticks"] = list(range(96))
    bad["controls"]["values"] = bad["controls"]["values"][:96]          # 96 不是 HORIZONS 里的值
    with pytest.raises(ValueError):
        long_cache.window_controls(bad, 0, 16., "cpu")
    bad = copy.deepcopy(m); bad["controls"]["units"] = "pixels"
    with pytest.raises(ValueError, match="units"):
        long_cache.window_controls(bad, 0, 16., "cpu")
    active(LONG)                                                        # 动作覆盖 128，当前数据集是 64
    with pytest.raises(ValueError, match="horizon"):
        long_cache.window_controls(m, 0, 16., "cpu")


def test_long_window_allowed_times(caches, real, active):
    _, long = caches
    active(LONG)
    with pytest.raises(ValueError, match="16, 24 or 32"):          # 不给 allowed：D4 的 LONG_TIMES 不变
        long_window.read_long_window(long/"0036", start_tick=0, duration_cycles=64., expected_split="test", device="cpu")
    w = long_window.read_long_window(long/"0036", start_tick=0, duration_cycles=64., expected_split="test", device="cpu",
                                     allowed=(8., 64.))
    raw = np.load(long/"0036"/"raw_z.npy")
    assert torch.equal(w.target.raw_z[0], torch.from_numpy(raw[64].astype(np.float32)))
    assert float(w.target.target_time_seconds[0]) == 16. and float(w.inputs.controls.duration[0]) == 64.
    with pytest.raises(ValueError, match="past the last"):
        long_window.read_long_window(long/"0036", start_tick=8, duration_cycles=64., expected_split="test", device="cpu",
                                     allowed=(64.,))
    with pytest.raises(ValueError, match="one of"):
        long_window.read_long_window(long/"0036", start_tick=0, duration_cycles=12., expected_split="test", device="cpu",
                                     allowed=(8., 64.))


# —— 滚动评测的定义 ——

def test_times_and_names():
    assert long_rollout.direct_times(128) == (16., 24., 32., 48., 64.) == long_rollout.direct_times(64)
    assert long_rollout.curve_times(128) == tuple(float(8*k) for k in range(1, 17))
    assert long_rollout.curve_times(64) == tuple(float(8*k) for k in range(1, 9))
    assert long_rollout.window_times(64) == long_rollout.curve_times(64)
    names = long_rollout.value_names(128)
    assert len(names) == 5*4+16 and names[:4] == ("long_direct:T=16", "long_composed:T=16", "long_pd_ln:T=16", "long_pd_in:T=16")
    assert names[-1] == "rollout:T=128" and len(long_rollout.value_names(64)) == 28
    for bad in (32, 96, 64.):
        with pytest.raises(ValueError):
            long_rollout.curve_times(bad)


class ActionToy(torch.nn.Module):
    """Φ_T(z) = z + β·T²·q + γ·(∫u_x)·p：不满足半群律、读动作，所以直接预测和 8 步滚动不同。"""
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, raw, controls, duration, grid):
        t = float(duration[0])
        assert float(controls.duration[0]) == t
        self.calls.append(t)
        n = int(controls.n_segments[0])
        integral = float(((controls.ends[0, :n]-controls.starts[0, :n]).double()*controls.u_values[0, :n, 0].double().cpu()).sum())
        q = torch.zeros(raw.shape[-1], dtype=torch.float64); q[0], q[1] = 1., -1.
        p = torch.zeros(raw.shape[-1], dtype=torch.float64); p[2], p[3] = 1., -1.
        return (raw.double()+.002*t*t*q+.5*integral*p).float()


def toy_windows(horizon, tid=36):
    m = dict(trajectory_id=tid, controls=dict(starts_ticks=list(range(horizon)), ends_ticks=list(range(1, horizon+1)),
                                               values=[[.4*math.sin(.3*k), .1] for k in range(horizon)], units="simulation_position_per_second"))
    g = torch.Generator().manual_seed(3)
    frames = torch.randn(horizon+1, 256, 1408, generator=g).to(torch.float16).float()
    out = {}
    for t in long_rollout.window_times(horizon):
        x = cache.WindowInput(frames[:1].clone(), long_cache.window_controls(m, 0, t, "cpu"), torch.tensor([t], dtype=torch.float64),
                              cache.regular_grid(16, 16), torch.tensor([0.], dtype=torch.float64), "img-0", tid, "test")
        y = cache.WindowTarget(frames[int(t):int(t)+1].clone(), torch.tensor([t*.25], dtype=torch.float64), f"img-{int(t)}")
        out[t] = cache.CacheWindow(x, y, dict(actual_manifest_sha256="m", generation=dict(evidence_role="real_frozen_encoder"),
                                              observation_variant="canonical"))
    return out


@pytest.mark.parametrize("horizon", [64, 128])
def test_values_against_d4(active, horizon):
    """composed 和曲线 = 同一次 8 步滚动，等于在时长 t 的窗口上 d4.rollout 的结果；direct = d4.direct_prediction。"""
    active(LONG128 if horizon == 128 else LONG)
    windows, config, model = toy_windows(horizon), e06_config(), ActionToy()
    values = long_rollout.trajectory_values(model, windows, config, horizon)
    assert tuple(values) == long_rollout.value_names(horizon)
    raw = windows[16.].inputs.raw_z
    for t in long_rollout.direct_times(horizon):
        w = windows[t]
        d = d4.direct_prediction(model, w)[0]
        rolled = d4.rollout(model, raw, w.inputs.controls, (8.,)*int(t/8), w.inputs.grid)
        assert values[f"long_direct:T={t:g}"] == d4._jepa(d, w.target.raw_z, config)
        assert values[f"long_composed:T={t:g}"] == d4._jepa(rolled, w.target.raw_z, config) == values[f"rollout:T={t:g}"]
        assert values[f"long_pd_ln:T={t:g}"] == d4._ln(d, rolled) > 0
        assert values[f"long_pd_in:T={t:g}"] == d4._in(d, rolled, raw) > 0
    for t in long_rollout.curve_times(horizon):
        w = windows[t]
        rolled = d4.rollout(model, raw, w.inputs.controls, (8.,)*int(t/8), w.inputs.grid)
        assert values[f"rollout:T={t:g}"] == d4._jepa(rolled, w.target.raw_z, config)
    # 调用：滚动 H/8 次 8 步，直接预测每个时长一次
    model.calls.clear(); long_rollout.trajectory_values(model, windows, config, horizon)
    assert model.calls == [8.]*(horizon//8)+list(long_rollout.direct_times(horizon))


def test_identity_values(active):
    active(LONG)
    windows, config = toy_windows(64), e06_config()
    values = long_rollout.trajectory_values(IdentityPredictor(), windows, config, 64)
    z = windows[8.].inputs.raw_z
    for t in long_rollout.direct_times(64):
        assert values[f"long_direct:T={t:g}"] == values[f"long_composed:T={t:g}"] == d4._jepa(z, windows[t].target.raw_z, config)
        assert values[f"long_pd_ln:T={t:g}"] == 0. and values[f"long_pd_in:T={t:g}"] == 0.


# —— 读窗口和整套评测 ——

def test_load_window_checks(caches, real, active):
    _, long = caches
    active(BASE)
    with pytest.raises(ValueError, match="not the horizon=64"):
        long_rollout.load_window(long, 36, 16., "cpu", 64)
    active(LONG)
    with pytest.raises(ValueError, match="test trajectory"):
        long_rollout.load_window(long, 3, 16., "cpu", 64)
    with pytest.raises(ValueError, match="one of"):
        long_rollout.load_window(long, 36, 12., "cpu", 64)
    with pytest.raises(ValueError):
        long_rollout.load_window(long, 36, 16., "cpu", 128)
    w = long_rollout.load_window(long, 36, 64., "cpu", 64)
    assert w.inputs.trajectory_id == 36 and float(w.target.target_time_seconds[0]) == 16.
    approved = data.approved_cache_metadata
    def other(root, *, splits):
        a = approved(root, splits=splits); a["manifest_sha256_by_id"] = {**a["manifest_sha256_by_id"], 36: "f"*64}
        return a
    data.approved_cache_metadata = other                 # 根目录登记的 manifest 和轨迹目录里的不同
    try:
        with pytest.raises(ValueError, match="identity/source"):
            long_rollout.load_window(long, 36, 16., "cpu", 64)
    finally:
        data.approved_cache_metadata = approved


def test_evaluate_end_to_end(caches, real, active):
    _, long = caches
    active(LONG)
    model, config = ActionToy(), e06_config()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rows, means = long_rollout.evaluate(model, long, config, 64, torch.device("cpu"))
    assert [r["trajectory_id"] for r in rows] == list(TEST_IDS) and model.training      # 训练标志恢复原样
    assert set(means) == set(long_rollout.value_names(64))
    raw = np.load(long/"0039"/"raw_z.npy").astype(np.float32)
    z = torch.from_numpy(raw[:1])
    for t in (16., 64.):
        target = torch.from_numpy(raw[int(t):int(t)+1])
        w = long_rollout.load_window(long, 39, t, "cpu", 64)
        assert torch.equal(w.inputs.raw_z, z) and torch.equal(w.target.raw_z, target)
        assert rows[-1]["values"][f"long_direct:T={t:g}"] == d4._jepa(d4.direct_prediction(model, w)[0], target, config)
    assert means["rollout:T=8"] == pytest.approx(sum(r["values"]["rollout:T=8"] for r in rows)/4, rel=1e-12)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


# —— 工具和任务表 ——

def _tool():
    spec = importlib.util.spec_from_file_location("long_rollout_tool", ROOT/"tools/long_rollout.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_tool_writes_report(caches, real, active, tmp_path, monkeypatch):
    _, long = caches
    active(BASE)
    job = tmp_path/"job"/"run"/"job"; job.mkdir(parents=True)
    (job/"job.json").write_text(json.dumps(dict(seed=1703, updates=10000, dataset=BASE.tag, condition="identity")))
    seen = {}

    def fake_load(condition, training_run, device):
        seen.update(condition=condition, training_run=training_run, dataset=datasets.active().tag)
        return IdentityPredictor(), e06_config(), None, dict(fake=True)
    monkeypatch.setattr(v7jobs, "load_for_evaluation", fake_load)
    monkeypatch.setattr(os, "environ", dict(os.environ))       # _job_env 改环境变量，测试后恢复
    out = tmp_path/"out"/"identity.json"
    tool = _tool()
    assert tool.main(["--job", str(tmp_path/"job"), "--features", str(long), "--horizon", "64", "--output", str(out), "--device", "cpu"]) == 0
    assert seen == dict(condition="identity", training_run=None, dataset=BASE.tag)
    r = json.loads(out.read_text())
    assert (r["kind"], r["dataset"], r["train_dataset"], r["horizon"], r["seed"]) == ("long_rollout", LONG.tag, BASE.tag, 64, 1703)
    assert r["counts"]["trajectories"] == 4 and len(r["rows"]) == 4 and set(r["means"]) == set(long_rollout.value_names(64))
    assert r["tubelet"] is None and r["definitions"]["curve_times"] == list(long_rollout.curve_times(64))
    assert datasets.active() == BASE
    with pytest.raises(SystemExit, match="已存在"):
        tool.main(["--job", str(tmp_path/"job"), "--features", str(long), "--horizon", "64", "--output", str(out), "--device", "cpu"])
    with pytest.raises(ValueError):                              # 特征是 horizon=64，要求 128：数据集对不上
        tool.main(["--job", str(tmp_path/"job"), "--features", str(long), "--horizon", "128", "--output", str(tmp_path/"x.json"),
                   "--device", "cpu"])
    assert not (tmp_path/"x.json").exists()


def test_tool_refuses_long_or_sampled_training_dataset(tmp_path, active, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for dataset in ("double_well~horizon=64-40", "double_well~fps=8-40", "double_well~zoom=0.8-40"):
        active(datasets.parse(dataset.rpartition("-")[0]+":40"))
        job = tmp_path/dataset; job.mkdir()
        (job/"job.json").write_text(json.dumps(dict(seed=1703, updates=1, dataset=dataset, condition="identity")))
        with pytest.raises(SystemExit):
            _tool().main(["--job", str(job), "--features", str(tmp_path), "--output", str(tmp_path/"o.json"), "--device", "cpu"])
    active(BASE)
    job = tmp_path/"transfer"; job.mkdir()
    (job/"job.json").write_text(json.dumps(dict(kind="transfer", seed=1703, updates=1, dataset=BASE.tag, condition="identity")))
    with pytest.raises(SystemExit, match="transfer"):
        _tool().main(["--job", str(job), "--features", str(tmp_path), "--output", str(tmp_path/"o.json"), "--device", "cpu"])


def test_transfer_refuses_horizon(tmp_path, active):
    active(BASE)
    with pytest.raises(ValueError, match="long_rollout"):
        v7jobs.transfer_job(tmp_path, tmp_path/"out", torch.device("cpu"), "identity", training_run=None, test_variant="horizon=128")
    assert not (tmp_path/"out").exists()


def _jobs(*args):
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/make_v8_jobs.py"), *args], env=_clean_env(), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_make_v8_jobs_long_stage():
    from po_jepa import protocols
    lines = _jobs("--stage", "long")
    commands = [l for l in lines if l and not l.startswith("#")]
    encode = [c for c in commands if "encode_v7.py encode" in c]
    final = [c for c in commands if "encode_v7.py finalize" in c]
    rollout = [c for c in commands if "long_rollout.py" in c]
    assert len(encode) == 7*4 and len(final) == 7 and len(rollout) == 7*3*6 and commands[-1].startswith("python tools/summarize_v7.py")
    assert all("--splits test" in c and "horizon=128" in c and "--trajectories 2000" in c for c in encode+final)
    assert "--variant horizon=128,shape=multi_well" in next(c for c in encode if "landscape" in c)
    c = next(c for c in rollout if "landscape" in c and "seed1705/vjepa2_ac " in c)
    assert ("--job ../results-v8/full/landscape~shape=multi_well/seed1705/vjepa2_ac --features "
            "../features-v8/landscape~horizon=128,shape=multi_well-2000 --horizon 128 "
            "--output ../results-v8/full/long/landscape~shape=multi_well/seed1705/vjepa2_ac.json --device cuda:0 --ac-checkpoint") in c
    assert not lines[0].startswith("# 注意")
    second = _jobs("--stage", "long", "--scenes", "pendulum", "--conditions", "identity", "--shards", "1")
    assert second[0].startswith("# 注意：二阶场景 pendulum") and len([l for l in second if l.startswith("python tools/long_rollout.py")]) == 3
    # 其他阶段不变：没有长时程的命令
    for stage in ("pilot", "full", "variants"):
        assert not any("horizon=" in l or "long_rollout" in l for l in _jobs("--stage", stage))
    assert len(protocols.SCENES) == 7


def test_summarize_long_reports(tmp_path):
    spec = importlib.util.spec_from_file_location("summarize_v7", ROOT/"tools/summarize_v7.py")
    s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
    names = long_rollout.value_names(128)

    def report(condition, seed, offset):
        rows = [dict(trajectory_id=t, values={n: offset+.01*t for n in names}) for t in (1800, 1801)]
        return dict(kind="long_rollout", dataset="double_well~horizon=128-2000", updates=10000, condition=condition, seed=seed,
                    definitions=long_rollout.definitions(128), rows=rows,
                    means={n: offset+.01*1800.5 for n in names})
    for c, off in (("po_jepa_full", .1), ("identity", .3)):
        for seed in (1703, 1704):
            path = tmp_path/c/f"{seed}.json"; path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(report(c, seed, off)))
    (tmp_path/"zz_dup.json").write_text(json.dumps(report("identity", 1703, 9.)))
    out = s.long_reports(tmp_path)
    full = next(r for r in out["rows"] if r["condition"] == "po_jepa_full")
    assert full["seeds"] == 2 and full["times"] == [16., 24., 32., 48., 64.]
    assert full["composed"][0] == pytest.approx(.1+18.005)
    cmp = next(c for c in out["comparisons"] if c["comparison"] == "po_jepa_full − identity")
    assert cmp["T"] == 64. and cmp["mean"] == pytest.approx(-.2) and all(x["full_better"] == 2 for x in cmp["per_seed"])
    assert len(out["duplicates"]) == 1
    md = s._markdown(dict(jobs=0, problems=[], across_seeds=[], transfer=[], planning=[], long=out, missing=[], duplicates=[]))
    assert "## 长时程滚动" in md and "8 步滚动 T=64" in md and "po_jepa_full − identity" in md
