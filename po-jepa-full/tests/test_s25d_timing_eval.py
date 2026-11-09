"""S25d：时间采样测试集接编码、缓存和按帧评测（docs/segments/S25d.md）。

假特征用 tools/encode_v7.py --fake-encoder（子进程，同 S25c），读取时把 synthetic 标记换成 real。数据集是 double_well:40
（测试段 36–39；测试段不超过 20 条，所以每条都是 5 个起点 tick 0、5、10、15、20），只编码测试段。
"""

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from po_jepa import ac_baseline, datasets, systems, timing
from po_jepa.baselines import IdentityPredictor
from po_jepa.runtime import cache, data, timing_cache, timing_eval, v7jobs
from po_jepa.runtime.addon import d4
from test_training import e06_config

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"src"
BASE = datasets.parse("double_well:40")
FPS8 = datasets.parse("double_well~fps=8:40")
FPS2 = datasets.parse("double_well~fps=2:40")
FPS16 = datasets.parse("double_well~fps=16:40")
JIT = datasets.parse("double_well~jitter=0.4:40")
SW = datasets.parse("double_well~switch_offset=0.5:40")
TEST_IDS = tuple(range(36, 40))
EXP = datasets.parse("double_well~exposure=1:40")
VARIANTS = {"base": "", "fps8": "fps=8", "fps2": "fps=2", "jit": "jitter=0.4", "sw": "switch_offset=0.5", "exp": "exposure=1"}


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
    root = tmp_path_factory.mktemp("s25d")
    for name, variant in VARIANTS.items():
        r = _encode(root/name, variant)
        assert r.returncode == 0, r.stderr
    return {name: root/name for name in VARIANTS}


def _as_real(m):
    m = copy.deepcopy(m); g = m["generation"]
    g.update(evidence_role="real_frozen_encoder", encoder_checkpoint_sha256=cache.CHECKPOINT_SHA,
             encoder_load_report=dict(sha256=cache.CHECKPOINT_SHA, encoder_key="target_encoder", strict=True,
                                      missing_keys=[], unexpected_keys=[]))
    return m


@pytest.fixture
def real(monkeypatch):
    """读取时当作真实编码器的缓存（只改内存里的 manifest），并清掉进程内的核验缓存。"""
    original, original_t = cache._read_v7_cache, timing_cache.check_manifest
    monkeypatch.setattr(cache, "_read_v7_cache", lambda path, m, sha: original(path, _as_real(m), sha))
    monkeypatch.setattr(timing_cache, "check_manifest", lambda path, m, sha, spec: original_t(path, _as_real(m), sha, spec))
    monkeypatch.setattr(cache, "_VERIFIED_CACHE", {})
    monkeypatch.setattr(timing_cache, "_VERIFIED", {})


@pytest.fixture
def active():
    old = datasets.active()
    yield lambda spec: datasets._set_for_tests(spec)
    datasets._set_for_tests(old)


def _manifest(root, tid):
    return json.loads((root/f"{tid:04d}"/"manifest.json").read_text())


# —— 编码 ——

def test_encode_timing_needs_test_split_and_horizon_32(tmp_path):
    r = _encode(tmp_path/"a", "fps=8", splits="train,test")
    assert r.returncode != 0 and "--splits test" in r.stderr and "timing_eval" in r.stderr and not (tmp_path/"a").exists()
    r = _encode(tmp_path/"b", "fps=8,horizon=64")
    assert r.returncode != 0 and "horizon=32" in r.stderr and not (tmp_path/"b").exists()
    r = _encode(tmp_path/"c", "fps=8,zoom=0.8")
    assert r.returncode != 0 and "空间采样" in r.stderr


def test_timing_cache_layout(caches, active):
    m = _manifest(caches["fps8"], 37)
    assert m["shape"] == [65, 256, 1408] and len(m["frames"]) == 65 and m["variant"] == "fps=8"
    assert m["timing"] == dict(fps=8, jitter=0., exposure=0., switch_offset=0.)
    assert [f["frame"] for f in m["frames"]] == list(range(65))
    assert [f["time_ticks"] for f in m["frames"]] == [k/2 for k in range(65)]
    assert m["frames"][3]["time_seconds"] == .375 and "src/po_jepa/timing.py" in m["generation"]["generator_source_sha256"]
    assert _manifest(caches["fps2"], 37)["shape"] == [17, 256, 1408]
    base = _manifest(caches["base"], 37)
    assert "src/po_jepa/timing.py" not in base["generation"]["generator_source_sha256"] and "timing" not in base
    assert [f["tick"] for f in base["frames"]] == list(range(33))           # 不带时间采样：manifest 格式不变
    jit = _manifest(caches["jit"], 38)
    active(JIT)
    assert [f["time_ticks"] for f in jit["frames"]] == timing.frame_times(38).tolist()
    assert jit["frames"][5]["time_ticks"] != 5. and jit["frames"][32]["time_ticks"] == 32.
    sw = _manifest(caches["sw"], 38)
    active(SW)
    expected = systems.commanded_controls(38)
    assert isinstance(expected, systems.SubtickControls) and set(sw["controls"]) == {"starts_subticks", "ends_subticks", "values", "units"}
    assert sw["controls"]["starts_subticks"] == expected.starts_subticks.tolist() and sw["controls"]["ends_subticks"][-1] == 128
    assert sw["controls"]["starts_subticks"][1] % 4 == 2                  # 边界往后挪了半个 tick


def test_timing_cache_images_and_features(caches, active):
    """画面哈希 = 时间采样数据集 frames() 的画面；特征 = 假编码器的输出；整 tick 的帧和主实验缓存的同一帧相同。"""
    spec = importlib.util.spec_from_file_location("encode_v7_tool", ROOT/"tools/encode_v7.py")
    enc = importlib.util.module_from_spec(spec); spec.loader.exec_module(enc)
    active(FPS8)
    _, frames = systems.get().frames(39)
    m = _manifest(caches["fps8"], 39)
    assert [f["image_sha256"] for f in m["frames"]] == [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]
    x = torch.from_numpy(frames[[1, 31, 64]]).permute(0, 3, 1, 2).float()/255.
    stored = np.load(caches["fps8"]/"0039"/"raw_z.npy")
    assert np.allclose(stored[[1, 31, 64]].astype(np.float32), enc._fake_encoder(torch)(x).numpy().astype(np.float16).astype(np.float32),
                       atol=2e-3)
    base = _manifest(caches["base"], 39); zb = np.load(caches["base"]/"0039"/"raw_z.npy")
    assert [f["image_sha256"] for f in m["frames"][::2]] == [f["image_sha256"] for f in base["frames"]]
    assert np.allclose(stored[::2].astype(np.float32), zb.astype(np.float32), atol=2e-3)
    assert m["frames"][1]["image_sha256"] not in {f["image_sha256"] for f in base["frames"]}


# —— 缓存读取 ——

def test_timing_cache_reads_only_as_its_dataset(caches, real, active):
    active(BASE)
    with pytest.raises(ValueError, match="no time-sampling knob"):
        timing_cache.read_complete(caches["fps8"]/"0036")
    with pytest.raises((ValueError, KeyError)):      # 普通窗口（cache.py，写死 33 帧、每 tick 一帧）不收
        cache.read_window(caches["jit"]/"0036", start_tick=0, duration_cycles=1., expected_split="test", device="cpu")
    active(FPS16)                                    # fps 不同（帧数 129 ≠ 65）
    with pytest.raises(ValueError, match="mismatch"):
        timing_cache.read_complete(caches["fps8"]/"0036")
    active(JIT)
    with pytest.raises((ValueError, KeyError)):      # 抖动的缓存当成 cache.py 的读也不收
        cache.read_window(caches["jit"]/"0036", start_tick=0, duration_cycles=1., expected_split="test", device="cpu")
    with pytest.raises(ValueError, match="mismatch"):   # 33 帧的主实验缓存当成抖动缓存读：variant 不同
        timing_cache.read_complete(caches["base"]/"0036")
    raw, m, _ = timing_cache.read_complete(caches["jit"]/"0036")
    assert raw.shape == (33, 256, 1408) and m["trajectory_id"] == 36
    active(datasets.parse("double_well~jitter=0.4:2000"))   # 核过的结果不跨数据集复用（条数不同）
    with pytest.raises(ValueError, match="mismatch"):
        timing_cache.read_complete(caches["jit"]/"0036")
    active(datasets.parse("double_well~fps=8,horizon=64:40"))
    with pytest.raises(ValueError, match="horizon=32"):
        timing_cache.read_complete(caches["fps8"]/"0036")


def test_timing_cache_checks(caches, real, active, tmp_path):
    active(JIT)
    for name, edit in (("feature", lambda d: (d/"raw_z.npy").write_bytes((d/"raw_z.npy").read_bytes()[:-2]+b"\0\1")),
                       ("manifest", lambda d: (d/"manifest.json").write_text((d/"manifest.json").read_text()+" "))):
        d = tmp_path/name; shutil.copytree(caches["jit"]/"0037", d); edit(d)
        with pytest.raises(ValueError, match="SHA mismatch"):
            timing_cache.read_complete(d)
    root = caches["jit"]/"0037"
    for name, change in (("controls", lambda m: m["controls"]["values"][0].__setitem__(0, .123)),
                         ("time", lambda m: m["frames"][7].__setitem__("time_ticks", 7.)),
                         ("seconds", lambda m: m["frames"][7].__setitem__("time_seconds", 7*.25)),
                         ("time_and_seconds", lambda m: m["frames"][7].update(time_ticks=7., time_seconds=7*.25)),
                         ("frame", lambda m: m["frames"][7].__setitem__("frame", 8)),
                         ("image", lambda m: m["frames"][7].__setitem__("image_sha256", "ab")),
                         ("timing", lambda m: m["timing"].__setitem__("jitter", .2)),
                         ("no_timing", lambda m: m.pop("timing")),
                         ("shape", lambda m: m.__setitem__("shape", [65, 256, 1408])),
                         ("frames", lambda m: m["frames"].pop()),
                         ("units", lambda m: m["controls"].__setitem__("units", "pixels")),
                         ("variant", lambda m: m.__setitem__("variant", "fps=8")),
                         ("subticks", lambda m: m.__setitem__("controls", dict(
                             starts_subticks=[4*s for s in m["controls"]["starts_ticks"]],
                             ends_subticks=[4*e for e in m["controls"]["ends_ticks"]],
                             values=m["controls"]["values"], units=m["controls"]["units"])))):
        m = _as_real(_manifest(caches["jit"], 37)); change(m)
        with pytest.raises((ValueError, KeyError)):
            timing_cache.check_manifest(root, m, "sha", datasets.active())
    m = _as_real(_manifest(caches["jit"], 37))
    timing_cache.check_manifest(root, m, "sha", datasets.active())       # 不改时能过（上面每一处都是它单独报错）
    active(SW)
    m = _as_real(_manifest(caches["sw"], 37))
    timing_cache.check_manifest(caches["sw"]/"0037", m, "sha", datasets.active())
    bad = copy.deepcopy(m); bad["controls"]["starts_subticks"][1] += 1; bad["controls"]["ends_subticks"][0] += 1
    with pytest.raises(ValueError, match="random stream"):
        timing_cache.check_manifest(caches["sw"]/"0037", bad, "sha", datasets.active())
    bad = copy.deepcopy(m); bad["controls"]["starts_ticks"] = [0]
    with pytest.raises(ValueError, match="mixes"):
        timing_cache.check_manifest(caches["sw"]/"0037", bad, "sha", datasets.active())


def test_timing_cache_refuses_synthetic_features(caches, active):
    active(FPS8)
    m = _manifest(caches["fps8"], 37)
    with pytest.raises(ValueError, match="provenance"):
        timing_cache.check_manifest(caches["fps8"]/"0037", m, "sha", datasets.active())
    m["generation"].update(encoder_checkpoint_sha256=cache.CHECKPOINT_SHA, encoder_load_report=dict(sha256=cache.CHECKPOINT_SHA,
                           encoder_key="target_encoder", strict=True, missing_keys=[], unexpected_keys=[]))
    with pytest.raises(ValueError, match="provenance"):           # 只差 evidence_role 也不收
        timing_cache.check_manifest(caches["fps8"]/"0037", m, "sha", datasets.active())


def _controls_equal(a, b):
    for field in ("starts", "ends", "u_values", "n_segments", "segment_valid", "source_origin_seconds", "duration"):
        if not torch.equal(getattr(a, field), getattr(b, field)):
            return False
    return a.source_ids == b.source_ids


@pytest.mark.parametrize("s,n", [(0, 1), (5, 3), (10, 12), (20, 8)])
def test_fps8_even_windows_equal_the_main_cache_windows(caches, real, active, s, n):
    """fps=8 上第 2s 帧到第 2s+2n 帧的窗口 = 主实验缓存上 cache.read_window(s, n)：动作全部字段、时刻、目标画面逐位相同。"""
    active(BASE)
    a = cache.read_window(caches["base"]/"0038", start_tick=s, duration_cycles=float(n), expected_split="test", device="cpu")
    active(FPS8)
    b = timing_cache.read_window(caches["fps8"]/"0038", start_frame=2*s, frames=2*n, expected_split="test", device="cpu")
    assert _controls_equal(a.inputs.controls, b.inputs.controls)
    assert torch.equal(a.inputs.duration, b.inputs.duration) and torch.equal(a.inputs.observed_time_seconds, b.inputs.observed_time_seconds)
    assert torch.equal(a.target.target_time_seconds, b.target.target_time_seconds)
    assert a.inputs.image_sha256 == b.inputs.image_sha256 and a.target.image_sha256 == b.target.image_sha256
    assert torch.allclose(a.inputs.raw_z, b.inputs.raw_z, atol=2e-3) and torch.allclose(a.target.raw_z, b.target.raw_z, atol=2e-3)
    data._window_boundary(b, b)


def test_exposure_windows_match_the_main_cache_except_images(caches, real, active):
    """exposure=1：帧时刻、动作和主实验相同（窗口的动作、时刻逐位相同），画面（运动模糊）不同，第 0 帧除外（窗口早于起点取起点）。"""
    active(BASE)
    a = cache.read_window(caches["base"]/"0037", start_tick=5, duration_cycles=3., expected_split="test", device="cpu")
    active(EXP)
    b = timing_cache.read_window(caches["exp"]/"0037", start_frame=5, frames=3, expected_split="test", device="cpu")
    assert _controls_equal(a.inputs.controls, b.inputs.controls) and torch.equal(a.inputs.duration, b.inputs.duration)
    assert torch.equal(a.target.target_time_seconds, b.target.target_time_seconds)
    assert a.inputs.image_sha256 != b.inputs.image_sha256 and a.target.image_sha256 != b.target.image_sha256
    m, base = _manifest(caches["exp"], 37), _manifest(caches["base"], 37)
    assert m["frames"][0]["image_sha256"] == base["frames"][0]["image_sha256"]


def test_jitter_window_times_and_controls(caches, real, active):
    active(JIT)
    t = timing.frame_times(37)
    w = timing_cache.read_window(caches["jit"]/"0037", start_frame=5, frames=3, expected_split="test", device="cpu")
    assert float(w.inputs.duration[0]) == t[8]-t[5] and float(w.inputs.observed_time_seconds[0]) == t[5]*.25
    assert float(w.target.target_time_seconds[0]) == pytest.approx(t[8]*.25, abs=1e-14)
    raw = np.load(caches["jit"]/"0037"/"raw_z.npy")
    assert torch.equal(w.inputs.raw_z[0], torch.from_numpy(raw[5].astype(np.float32)))
    assert torch.equal(w.target.raw_z[0], torch.from_numpy(raw[8].astype(np.float32)))
    # 动作：独立地把每 tick 一段的下达动作裁到 [t5, t8)，局部时间从 0 起
    c = systems.commanded_controls(37); per_tick = np.repeat(c.values, c.ends_ticks-c.starts_ticks, axis=0)
    edges = [t[5]]+[float(k) for k in range(math.floor(t[5])+1, math.ceil(t[8]))]+[t[8]]
    got = w.inputs.controls; n = int(got.n_segments[0])
    # 相邻同值的 tick 合成一段：按段端点逐个时刻查值
    for left, right in zip(edges[:-1], edges[1:]):
        mid = .5*(left+right)-t[5]
        j = int(((got.starts[0, :n] <= mid) & (got.ends[0, :n] > mid)).nonzero()[0])
        assert torch.allclose(got.u_values[0, j].double(), torch.from_numpy(per_tick[int(math.floor(left))]), atol=1e-7)
    assert float(got.starts[0, 0]) == 0. and float(got.ends[0, n-1]) == t[8]-t[5]
    data._window_boundary(w, w)
    with pytest.raises(ValueError, match="past the last"):
        timing_cache.read_window(caches["jit"]/"0037", start_frame=30, frames=3, expected_split="test", device="cpu")
    for bad in ((-1, 1), (0, 0), (1.0, 1)):
        with pytest.raises(ValueError):
            timing_cache.read_window(caches["jit"]/"0037", start_frame=bad[0], frames=bad[1], expected_split="test", device="cpu")
    with pytest.raises(ValueError, match="cross-split"):
        timing_cache.read_window(caches["jit"]/"0037", start_frame=0, frames=1, expected_split="train", device="cpu")


def test_switch_offset_window_controls(caches, real, active):
    """switch_offset=0.5：窗口的动作边界在半 tick 上，等于 SubtickControls 的边界 / 4，裁到窗口里。"""
    active(SW)
    c = systems.commanded_controls(36)
    w = timing_cache.read_window(caches["sw"]/"0036", start_frame=8, frames=4, expected_split="test", device="cpu")
    got = w.inputs.controls; n = int(got.n_segments[0])
    starts, ends = c.starts_subticks/4, c.ends_subticks/4
    keep = (ends > 8) & (starts < 12)                                       # 段边界 1.5、9.5 → 窗口里 [0,1.5)、[1.5,4)
    assert np.array_equal(got.starts[0, :n].numpy(), np.maximum(starts[keep], 8)-8)
    assert np.array_equal(got.ends[0, :n].numpy(), np.minimum(ends[keep], 12)-8)
    assert torch.equal(got.u_values[0, :n], torch.tensor(c.values[keep], dtype=torch.float32))
    assert any(float(x) % 1 == .5 for x in got.starts[0, 1:n])                  # 至少一个切换落在两帧中间
    full = timing_cache.full_controls(_as_real(_manifest(caches["sw"], 36)), "cpu")
    assert float(full.duration[0]) == 32. and torch.equal(full.starts[0, :len(starts)], torch.from_numpy(starts))


# —— 评测的定义 ——

def test_start_frames_and_steps(active):
    active(FPS2)
    assert timing_eval.start_frames(37) == (0, 2, 5, 7, 10)  # tick 5 → 第 2 帧（tick 4），tick 15 → 第 7 帧（tick 14）
    assert timing_eval.frame_steps(10) == (1, 2, 3, 4) and timing_eval.frame_steps(7) == (1, 2, 3, 4, 8)
    assert timing_eval.frame_steps(16) == ()
    active(FPS8)
    assert timing_eval.start_frames(38) == (0, 10, 20, 30, 40) and timing_eval.frame_steps(40) == (1, 2, 3, 4, 8, 12)
    active(FPS16)
    assert timing_eval.start_frames(39) == (0, 20, 40, 60, 80)
    active(JIT)
    assert timing_eval.start_frames(36) == (0, 5, 10, 15, 20) and timing_eval.frame_steps(20) == (1, 2, 3, 4, 8, 12)
    active(datasets.parse("double_well~fps=8:2000"))     # 大数据集每条 2 个起点（按编号轮换）
    assert timing_eval.start_frames(1800) == (0, 20) and timing_eval.start_frames(1803) == (0, 30)
    active(BASE)
    with pytest.raises(ValueError, match="no time-sampling knob"):
        timing_eval.start_frames(36)
    assert timing_eval.value_names()[:4] == ("direct:n=1", "composed:n=1", "pd_ln:n=1", "pd_in:n=1")
    assert len(timing_eval.value_names()) == 24 and len(timing_eval.value_names(ac=True)) == 30
    from po_jepa import protocols
    assert timing_eval.TESTS == protocols.TIMING_TESTS == ("fps=2", "fps=8", "fps=16", "jitter=0.4", "exposure=1", "switch_offset=0.5")


def _segments(bounds, values, origin=0.):
    """手写的分段常值动作（局部时间、物理原点 origin 秒）。"""
    spec = cache.ControlSpec(2, .25, ("u_x", "u_y"), ("simulation_position_per_second",)*2,
                             (-.5, -.5), (.5, .5), (1., 1.), (0., 0.), "reference/protocol.md")
    n = len(values)
    return cache.from_physical_segments(
        starts_seconds=torch.tensor([bounds[:-1]], dtype=torch.float64)*.25+origin,
        ends_seconds=torch.tensor([bounds[1:]], dtype=torch.float64)*.25+origin,
        u_values=torch.tensor([values], dtype=torch.float32), n_segments=torch.tensor([n]),
        segment_valid=torch.ones((1, n), dtype=torch.bool), origin_seconds=torch.tensor([origin], dtype=torch.float64),
        control_spec=spec, source_ids=("trajectory-0036",))


def test_interval_actions():
    a = _segments([0., .25, 2.], [[.4, 0.], [-.2, .1]])          # 2 tick 的帧间隔（fps=2）：按长度加权
    b = _segments([0., .5], [[.3, -.3]])
    got = timing_eval.interval_actions([a, b])
    assert got.shape == (1, 2, 2)
    assert torch.allclose(got[0, 0].double(), torch.tensor([(.4*.25-.2*1.75)/2, .1*1.75/2], dtype=torch.float64), atol=1e-7)
    assert torch.allclose(got[0, 1], torch.tensor([.3, -.3]))
    # 对照：ac_baseline.tick_actions 只平均第一个 tick，帧间隔 > 1 tick 时不能用它
    assert not torch.allclose(ac_baseline.tick_actions(a, 1)[0, 0], got[0, 0])


@pytest.fixture(scope="module")
def ac():
    torch.manual_seed(0)
    return ac_baseline.ACPredictor(ac_baseline.SMALL).eval()


def test_ac_interpolation(ac):
    g = torch.Generator().manual_seed(1)
    raw = torch.randn(1, 256, 1408, generator=g)
    grid = cache.regular_grid(16, 16)
    c = _segments([0., .5, 1.25, 3.], [[.4, 0.], [-.2, .1], [.1, .1]])
    z0 = torch.nn.functional.layer_norm(raw, (1408,))
    with torch.no_grad():
        full = ac(raw, c, torch.tensor([3.], dtype=torch.float64), grid)
        for d in (1., 1.5, 2., 2.5):                             # 整数、半整数：就是 forward
            assert torch.equal(timing_eval.ac_interpolated(ac, raw, c, torch.tensor([d], dtype=torch.float64), grid),
                               ac(raw, c, torch.tensor([d], dtype=torch.float64), grid))
        # 一般时长：相邻两步线性插值，第 k+1 步的动作 = [k, T) 的平均（同 forward 的 tick_actions）
        steps = ac.rollout(z0, ac_baseline.tick_actions(c, 3), 16, 16)
        assert torch.allclose(steps[:, 2], full, atol=1e-6)
        for d in (.25, 1.75, 2.2):
            got = timing_eval.ac_interpolated(ac, raw, c, torch.tensor([d], dtype=torch.float64), grid)
            k, w = math.floor(d), d-math.floor(d)
            clipped = cache.clip_action_segments(c, start=torch.tensor([0.], dtype=torch.float64), end=torch.tensor([d], dtype=torch.float64))
            partial = ac.rollout(z0, ac_baseline.tick_actions(clipped, k+1), 16, 16)
            lower = z0[0] if k == 0 else partial[0, k-1]
            assert torch.allclose(got[0], (1-w)*lower+w*partial[0, k], atol=1e-6)
            assert not torch.allclose(got[0], partial[0, k], atol=1e-3)
            assert torch.equal(timing_eval.predictor(ac)(raw, clipped, torch.tensor([d], dtype=torch.float64), grid), got)
        with pytest.raises(ValueError, match="half-integer"):        # forward 本身不收这种时长
            ac(raw, clipped, torch.tensor([2.2], dtype=torch.float64), grid)
        # 一帧算一步：fps=4 时（每帧 1 tick）与 forward 的整数时长相同（动作平均的舍入除外）
        pieces = [cache.clip_action_segments(c, start=torch.tensor([float(k)], dtype=torch.float64),
                                             end=torch.tensor([k+1.], dtype=torch.float64)) for k in range(3)]
        assert torch.allclose(timing_eval.ac_frame_step(ac, raw, pieces, grid), full, atol=1e-5)
        # fps=8（每帧半 tick）：一帧一步预测的是 3 个 tick 以后（不知道帧间隔），和时长 1.5 的预测不同
        halves = [cache.clip_action_segments(c, start=torch.tensor([.5*k], dtype=torch.float64),
                                             end=torch.tensor([.5*k+.5], dtype=torch.float64)) for k in range(3)]
        fs = timing_eval.ac_frame_step(ac, raw, halves, grid)
        manual = ac.rollout(z0, timing_eval.interval_actions(halves), 16, 16)[:, -1]
        assert torch.equal(fs, manual)
        assert not torch.allclose(fs, ac(raw, c, torch.tensor([1.5], dtype=torch.float64), grid), atol=1e-3)
    assert timing_eval.predictor(IdentityPredictor()).__class__ is IdentityPredictor


class ActionToy(torch.nn.Module):
    """Φ_T(z) = z + β·T²·q + γ·(∫u_x)·p：不满足半群律、读动作和时长。"""
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


def test_start_values_against_manual(caches, real, active):
    """direct = d4.direct_prediction；composed = 逐帧调用（每帧的窗口动作、输入是上一次的原始预测）；调用次序写死。"""
    active(JIT)
    model, config = ActionToy(), e06_config()
    i = 10
    steps = timing_eval.frame_steps(i)
    windows = {n: timing_eval.load_window(caches["jit"], 36, i, n, "cpu") for n in steps}
    singles = [timing_eval.load_window(caches["jit"], 36, i+m, 1, "cpu") for m in range(max(steps))]
    out = timing_eval.start_values(model, windows, singles, config)
    assert sorted(out) == list(steps) == [1, 2, 3, 4, 8, 12]
    t = timing.frame_times(36)
    raw = windows[1].inputs.raw_z
    for n in steps:
        x = raw
        for m in range(n):
            w = singles[m]
            assert float(w.inputs.duration[0]) == t[i+m+1]-t[i+m]
            x = model(x, w.inputs.controls, w.inputs.duration, w.inputs.grid)
        target = windows[n].target.raw_z
        d = d4.direct_prediction(model, windows[n])[0]
        assert out[n][f"direct:n={n}"] == d4._jepa(d, target, config)
        assert out[n][f"composed:n={n}"] == d4._jepa(x, target, config)
        assert out[n][f"pd_ln:n={n}"] == d4._ln(d, x) and out[n][f"pd_in:n={n}"] == d4._in(d, x, raw)
        if n > 1:
            assert out[n][f"pd_ln:n={n}"] > 0
    model.calls.clear(); timing_eval.start_values(model, windows, singles, config)
    assert model.calls == [float(t[i+m+1]-t[i+m]) for m in range(12)]+[float(t[i+n]-t[i]) for n in steps]
    ident = timing_eval.start_values(IdentityPredictor(), windows, singles, config)
    for n in steps:
        assert ident[n][f"direct:n={n}"] == ident[n][f"composed:n={n}"] == d4._jepa(raw, windows[n].target.raw_z, config)
        assert ident[n][f"pd_ln:n={n}"] == 0.


def test_start_values_ac(caches, real, active, ac):
    active(FPS8)
    i = 0
    windows = {n: timing_eval.load_window(caches["fps8"], 36, i, n, "cpu") for n in (1, 3)}
    singles = [timing_eval.load_window(caches["fps8"], 36, i+m, 1, "cpu") for m in range(3)]
    out = timing_eval.start_values(ac, windows, singles, e06_config())
    assert set(out[3]) == {"direct:n=3", "composed:n=3", "pd_ln:n=3", "pd_in:n=3", "frame_step:n=3"}
    with torch.no_grad():
        direct = d4.direct_prediction(ac, windows[3])[0]           # T = 1.5：半整数，forward 本身
    assert out[3]["direct:n=3"] == d4._jepa(direct, windows[3].target.raw_z, e06_config())
    with torch.no_grad():                                          # 一帧一步：3 个帧间隔、3 步
        fs = timing_eval.ac_frame_step(ac, windows[3].inputs.raw_z, [s.inputs.controls for s in singles], windows[3].inputs.grid)
    assert out[3]["frame_step:n=3"] == d4._jepa(fs, windows[3].target.raw_z, e06_config())


def test_load_window_checks(caches, real, active):
    active(BASE)
    with pytest.raises(ValueError, match="no time-sampling knob"):
        timing_eval.load_window(caches["jit"], 36, 0, 1, "cpu")
    active(JIT)
    with pytest.raises(ValueError, match="test trajectory"):
        timing_eval.load_window(caches["jit"], 3, 0, 1, "cpu")
    approved = data.approved_cache_metadata

    def other(root, *, splits):
        a = approved(root, splits=splits); a["manifest_sha256_by_id"] = {**a["manifest_sha256_by_id"], 36: "f"*64}
        return a
    data.approved_cache_metadata = other
    try:
        with pytest.raises(ValueError, match="identity/source"):
            timing_eval.load_window(caches["jit"], 36, 0, 1, "cpu")
    finally:
        data.approved_cache_metadata = approved
    active(FPS8)                                  # 帧率 8 的数据集读抖动的缓存：根目录 report 的 variant 不同
    with pytest.raises(ValueError):
        timing_eval.load_window(caches["jit"], 36, 0, 1, "cpu")


def test_evaluate_end_to_end(caches, real, active):
    active(FPS2)
    model, config = ActionToy(), e06_config()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rows, means = timing_eval.evaluate(model, caches["fps2"], config, torch.device("cpu"))
    assert model.training and all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    # 起点 tick 0、5、10、15、20 → 第 0、2、5、7、10 帧；帧数只留不超过第 16 帧的
    steps = {0: (1, 2, 3, 4, 8, 12), 2: (1, 2, 3, 4, 8, 12), 5: (1, 2, 3, 4, 8), 7: (1, 2, 3, 4, 8), 10: (1, 2, 3, 4)}
    expected = [(tid, i, n) for tid in TEST_IDS for i in (0, 2, 5, 7, 10) for n in steps[i]]
    assert [(r["trajectory_id"], r["start_frame"], r["frames"]) for r in rows] == expected
    assert all(r["duration_ticks"] == 2.*r["frames"] and r["start_tick"] == 2.*r["start_frame"] for r in rows)
    assert means["windows:n=12"] == 8 and means["windows:n=1"] == 20 and means["duration_ticks:n=3"] == 6.
    assert means["direct:n=4"] == pytest.approx(sum(r["values"]["direct:n=4"] for r in rows if r["frames"] == 4)/20, rel=1e-12)
    assert "frame_step:n=1" not in means and set(k for k in means if ":n=" in k and k.split(":")[0] in timing_eval.KEYS) \
        == set(timing_eval.value_names())


# —— 工具、transfer、任务表、汇总 ——

def _tool():
    spec = importlib.util.spec_from_file_location("timing_eval_tool", ROOT/"tools/timing_eval.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_tool_writes_report(caches, real, active, tmp_path, monkeypatch):
    active(BASE)
    job = tmp_path/"job"/"run"/"job"; job.mkdir(parents=True)
    (job/"job.json").write_text(json.dumps(dict(seed=1703, updates=10000, dataset=BASE.tag, condition="identity")))
    seen = {}

    def fake_load(condition, training_run, device):
        seen.update(condition=condition, training_run=training_run, dataset=datasets.active().tag)
        return IdentityPredictor(), e06_config(), None, dict(fake=True)
    monkeypatch.setattr(v7jobs, "load_for_evaluation", fake_load)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    out = tmp_path/"out"/"identity.json"
    args = ["--job", str(tmp_path/"job"), "--test-variant", "switch_offset=0.5", "--features", str(caches["sw"]), "--device", "cpu"]
    assert _tool().main(args+["--output", str(out)]) == 0
    assert seen == dict(condition="identity", training_run=None, dataset=BASE.tag)
    r = json.loads(out.read_text())
    assert (r["kind"], r["dataset"], r["train_dataset"], r["test_variant"], r["seed"]) == \
        ("timing_eval", SW.tag, BASE.tag, "switch_offset=0.5", 1703)
    assert r["counts"] == dict(windows=len(r["rows"]), trajectories=4) and r["definitions"]["switch_offset"] == .5
    assert r["definitions"]["frame_steps"] == [1, 2, 3, 4, 8, 12] and r["tubelet"] is None
    assert datasets.active() == BASE
    with pytest.raises(SystemExit, match="已存在"):
        _tool().main(args+["--output", str(out)])
    with pytest.raises(ValueError):                    # 特征是 switch_offset，要求 fps=8：数据集对不上
        _tool().main(["--job", str(tmp_path/"job"), "--test-variant", "fps=8", "--features", str(caches["sw"]), "--device", "cpu",
                      "--output", str(tmp_path/"x.json")])
    assert not (tmp_path/"x.json").exists() and datasets.active() == BASE
    for bad, match in (("gain=2", "时间采样调节项"), ("fps=8,gain=2", "时间采样调节项"), ("fps=4", "默认时间采样"), ("", "时间采样调节项")):
        with pytest.raises(SystemExit, match=match):
            _tool().main(["--job", str(tmp_path/"job"), "--test-variant", bad, "--features", str(caches["sw"]), "--device", "cpu",
                          "--output", str(tmp_path/"y.json")])


def test_tool_refuses_sampled_training_dataset(tmp_path, active, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for dataset in ("double_well~horizon=64-40", "double_well~fps=8-40", "double_well~zoom=0.8-40"):
        active(datasets.parse(dataset.rpartition("-")[0]+":40"))
        job = tmp_path/dataset; job.mkdir()
        (job/"job.json").write_text(json.dumps(dict(seed=1703, updates=1, dataset=dataset, condition="identity")))
        with pytest.raises(SystemExit, match="基础场景"):
            _tool().main(["--job", str(job), "--test-variant", "fps=8", "--features", str(tmp_path), "--output", str(tmp_path/"o.json"),
                          "--device", "cpu"])
    active(BASE)
    job = tmp_path/"transfer"; job.mkdir()
    (job/"job.json").write_text(json.dumps(dict(kind="transfer", seed=1703, updates=1, dataset=BASE.tag, condition="identity")))
    with pytest.raises(SystemExit, match="transfer"):
        _tool().main(["--job", str(job), "--test-variant", "fps=8", "--features", str(tmp_path), "--output", str(tmp_path/"o.json"),
                      "--device", "cpu"])


@pytest.mark.parametrize("variant", ["fps=8", "exposure=1", "jitter=0.2", "switch_offset=0.25"])
def test_transfer_refuses_time_sampling(tmp_path, active, variant):
    active(BASE)
    with pytest.raises(ValueError, match="timing_eval"):
        v7jobs.transfer_job(tmp_path, tmp_path/"out", torch.device("cpu"), "identity", training_run=None, test_variant=variant)
    assert not (tmp_path/"out").exists()


def _jobs(*args):
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/make_v8_jobs.py"), *args], env=_clean_env(), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_make_v8_jobs_timing_stage():
    lines = _jobs("--stage", "timing")
    commands = [l for l in lines if l and not l.startswith("#")]
    encode = [c for c in commands if "encode_v7.py encode" in c]
    final = [c for c in commands if "encode_v7.py finalize" in c]
    evals = [c for c in commands if "tools/timing_eval.py" in c]
    assert len(encode) == 7*6*4 and len(final) == 7*6 and len(evals) == 7*6*3*6 and commands[-1].startswith("python tools/summarize_v7.py")
    assert all("--splits test" in c and "--trajectories 2000" in c for c in encode+final)
    assert "--variant fps=16,shape=multi_well" in next(c for c in encode if "landscape" in c and "fps=16" in c)
    c = next(c for c in evals if "landscape" in c and "jitter" in c and "seed1704/vjepa2_ac " in c)
    assert ("--job ../results-v8/full/landscape~shape=multi_well/seed1704/vjepa2_ac --test-variant jitter=0.4 --features "
            "../features-v8/landscape~jitter=0.4,shape=multi_well-2000 --output "
            "../results-v8/full/timing/landscape~jitter=0.4,shape=multi_well/seed1704/vjepa2_ac.json --device cuda:0 --ac-checkpoint") in c
    second = _jobs("--stage", "timing", "--scenes", "wave", "--conditions", "identity", "--shards", "1")
    assert second[0].startswith("# 注意：二阶场景 wave") and len([l for l in second if l.startswith("python tools/timing_eval.py")]) == 18
    for stage in ("pilot", "full", "variants", "long"):
        assert not any("timing_eval" in l or "fps=" in l or "switch_offset" in l for l in _jobs("--stage", stage))


def test_summarize_timing_reports(tmp_path):
    spec = importlib.util.spec_from_file_location("summarize_v7", ROOT/"tools/summarize_v7.py")
    s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)

    def report(condition, seed, offset, ac=False):
        names = timing_eval.value_names(ac)
        rows = [dict(trajectory_id=t, start_frame=i, frames=n, values={k: offset+.01*t for k in names if k.endswith(f":n={n}")})
                for t in (1800, 1801) for i in (0, 20) for n in timing_eval.FRAME_STEPS]
        means = {k: offset+.01*1800.5 for k in names}
        means.update({f"duration_ticks:n={n}": n/2 for n in timing_eval.FRAME_STEPS})
        return dict(kind="timing_eval", dataset="double_well~fps=8-2000", updates=10000, condition=condition, seed=seed,
                    definitions=dict(frame_steps=list(timing_eval.FRAME_STEPS)), rows=rows, means=means)
    for c, off, ac in (("po_jepa_full", .1, False), ("matched_transformer", .3, False), ("vjepa2_ac", .5, True)):
        for seed in (1703, 1704):
            path = tmp_path/c/f"{seed}.json"; path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(report(c, seed, off, ac)))
    (tmp_path/"zz_dup.json").write_text(json.dumps(report("vjepa2_ac", 1703, 9., True)))
    small = report("vjepa2_ac_small", 1703, .7, True); (tmp_path/"s1.json").write_text(json.dumps(small))
    small = report("vjepa2_ac_small", 1704, .7, True); del small["means"]["frame_step:n=12"]
    (tmp_path/"s2.json").write_text(json.dumps(small))
    out = s.timing_reports(tmp_path)
    full = next(r for r in out["rows"] if r["condition"] == "po_jepa_full")
    assert full["seeds"] == 2 and full["frames"] == [1, 2, 3, 4, 8, 12] and full["duration"][1] == 1.
    assert full["direct"][0] == pytest.approx(.1+18.005) and full["frame_step"] == [None]*6
    acrow = next(r for r in out["rows"] if r["condition"] == "vjepa2_ac")
    assert acrow["frame_step"][5] == pytest.approx(.5+18.005)
    smallrow = next(r for r in out["rows"] if r["condition"] == "vjepa2_ac_small")   # 一个种子缺这个键：不拿另一个种子顶替
    assert smallrow["frame_step"][5] is None and smallrow["frame_step"][4] == pytest.approx(.7+18.005)
    cmp = next(c for c in out["comparisons"] if c["comparison"] == "po_jepa_full − matched_transformer" and c["frames"] == 8)
    assert cmp["mean"] == pytest.approx(-.2) and all(x["full_better"] == 4 == x["windows"] for x in cmp["per_seed"])
    assert len(out["duplicates"]) == 1
    md = s._markdown(dict(jobs=0, problems=[], across_seeds=[], transfer=[], planning=[], long=dict(rows=[], comparisons=[]),
                          timing=out, missing=[], duplicates=[]))
    assert "## 时间采样" in md and "T=0.5: " in md and "po_jepa_full − matched_transformer" in md
