"""S25b3：两帧编码（tubelet=pair）加时间采样和空间采样（docs/segments/S25b3.md）。

前一帧固定在 0.25 s（1 tick）前（Jerry 2026-11-22 定），不是视频流里的上一帧。期望值从基础数据集的画法（second_order 自己的
frames_with_prev，不走 timing）、独立写出的帧时刻和独立的假编码器重算，不拿 pair_frames / encode_v7 的返回当参照。
假特征用 tools/encode_v7.py --fake-encoder（子进程），读取时把 synthetic 标记换成 real（同 S25b、S25d、S25e 的测试）。
数据集 <场景>:40，测试段 36–39。
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

from po_jepa import datasets, native_encoder, simulation, systems, timing, tubelet_probe, variants, vjepa
from po_jepa.baselines import IdentityPredictor
from po_jepa.runtime import cache, space_cache, timing_cache, v7jobs
from test_training import e06_config

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"src"
TEST_IDS = tuple(range(36, 40))
TIME_KNOBS = ("fps=2", "fps=8", "fps=16", "jitter=0.4", "exposure=1", "switch_offset=0.5")
SPACE_KNOBS = ("resolution=192", "resolution=384", "zoom=0.75", "zoom=0.75,pan=0.5", "zoom=0.75,pan=0.5,drift=0.25")
CACHES = {"pair": "tubelet=pair", "fps8": "tubelet=pair,fps=8", "dup8": "fps=8", "fps2": "tubelet=pair,fps=2",
          "jit": "tubelet=pair,jitter=0.4,pixel_noise=10", "r192": "tubelet=pair,resolution=192", "dup192": "resolution=192",
          "drift": "tubelet=pair,zoom=0.75,pan=0.5,drift=0.25"}


def spec(variant, system="pendulum"):
    return datasets.parse(f"{system}{'~'+variant if variant else ''}:40")


def _clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")}
    env.update(PYTHONPATH=str(SRC), OMP_NUM_THREADS="2")
    return env


def _encode(out, variant, modes=("encode", "finalize")):
    args = ["--system", "pendulum", "--trajectories", "40", "--output", str(out), "--variant", variant]
    extra = dict(encode=["--fake-encoder", "--device", "cpu", "--splits", "test"], finalize=["--splits", "test"])
    for mode in modes:
        r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/encode_v7.py"), mode, *args, *extra[mode]],
                           env=_clean_env(), capture_output=True, text=True)
        if r.returncode:
            return r
    return r


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    root = tmp_path_factory.mktemp("s25b3")
    for name, variant in CACHES.items():
        r = _encode(root/name, variant)
        assert r.returncode == 0, r.stderr
    return {name: root/name for name in CACHES}


def _as_real(m):
    m = copy.deepcopy(m)
    m["generation"].update(evidence_role="real_frozen_encoder", encoder_checkpoint_sha256=cache.CHECKPOINT_SHA,
                           encoder_load_report=dict(sha256=cache.CHECKPOINT_SHA, encoder_key="target_encoder", strict=True,
                                                    missing_keys=[], unexpected_keys=[]))
    return m


@pytest.fixture
def real(monkeypatch):
    original, original_t, original_s = cache._read_v7_cache, timing_cache.check_manifest, space_cache.check_manifest
    monkeypatch.setattr(cache, "_read_v7_cache", lambda path, m, sha: original(path, _as_real(m), sha))
    monkeypatch.setattr(timing_cache, "check_manifest", lambda path, m, sha, s: original_t(path, _as_real(m), sha, s))
    monkeypatch.setattr(space_cache, "check_manifest", lambda path, m, sha, s: original_s(path, _as_real(m), sha, s))
    for module, name in ((cache, "_VERIFIED_CACHE"), (timing_cache, "_VERIFIED"), (space_cache, "_VERIFIED")):
        monkeypatch.setattr(module, name, {})


@pytest.fixture
def active():
    old = datasets.active()
    yield lambda s: datasets._set_for_tests(s)
    datasets._set_for_tests(old)


def _sha(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def _manifest(folder):
    return json.loads((folder/"manifest.json").read_text())


def _rewrite_manifest(folder, change):
    m = json.loads((folder/"manifest.json").read_text())
    change(m)
    for name in ("manifest.json", "COMPLETE.json"):
        (folder/name).unlink()
    cache._write_json(folder/"manifest.json", m)
    cache._write_json(folder/"COMPLETE.json", dict(manifest_sha256=cache.file_sha(folder/"manifest.json")))


def _base_frames(variant, tid, system="pendulum"):
    """基础数据集（不带时间、空间采样）的 frames_with_prev：second_order 自己的画法，不经过 timing。下标 f = tick f−1。"""
    s = spec(variant, system)
    assert not s.knobs.resampled()
    old = datasets._set_for_tests(s)
    try:
        return systems.get().frames_with_prev(tid)[1]
    finally:
        datasets._set_for_tests(old)


def _pair(active, variant, tid, system="pendulum"):
    active(spec(variant, system))
    return timing.pair_frames(systems.get(), tid)


def _fake_pair(prev, cur):
    """独立重算的假 tubelet 编码（像素 /255，时间顺序（前，后））。"""
    p = torch.from_numpy(prev).permute(0, 3, 1, 2).float()/255.
    c = torch.from_numpy(cur).permute(0, 3, 1, 2).float()/255.
    return tubelet_probe.fake_tubelet_encoder()(torch.stack((p, c), 2)).numpy()


# —— 调节项 ——

@pytest.mark.parametrize("system", variants.SECOND_ORDER)
def test_pair_accepts_time_and_space_sampling(system):
    """二阶场景上 pair 不再给时间、空间采样调节项加限制：pair + 采样能解析 ⇔ 只有采样能解析。一阶场景仍拒绝 pair。"""
    for knob in TIME_KNOBS+SPACE_KNOBS:
        try:
            variants.parse(system, knob)
        except ValueError:
            with pytest.raises(ValueError):
                variants.parse(system, "tubelet=pair,"+knob)
            continue
        v = variants.parse(system, "tubelet=pair,"+knob)
        assert v.tubelet == "pair" and v.resampled()
    with pytest.raises(ValueError, match="一阶场景"):
        variants.parse("double_well", "tubelet=pair,fps=8")
    assert spec("tubelet=pair,fps=8").tag == "pendulum~fps=8,tubelet=pair-40"


# —— 时间采样下的前一帧（timing.pair_frames）——

@pytest.mark.parametrize("system", variants.SECOND_ORDER)
def test_pair_frames_keep_the_video_frames(system, active):
    """pair_frames 的状态和画面就是不带 pair 时的 frames()（逐位）：pair 不改物理、画面和随机流。"""
    for knob in ("fps=8", "jitter=0.4"):
        hidden, prev, images = _pair(active, "tubelet=pair,"+knob, 37, system)
        active(spec(knob, system))
        h, f = systems.get().frames(37)
        assert prev.shape == images.shape == f.shape and prev.dtype == np.uint8
        assert np.array_equal(hidden, h) and np.array_equal(images, f)


@pytest.mark.parametrize("system", variants.SECOND_ORDER)
def test_fps8_prev_frame_is_the_picture_one_tick_earlier(system, active):
    """fps = 8：t_k = k/2，前一帧在 k/2 − 1。偶数 k 落在整 tick 上，等于基础数据集（同样的像素噪声）的 tick k/2 − 1；
    奇数 k ≥ 3 等于第 k − 2 帧；k = 1 在 −0.5 tick，等于 frames(prev=True) 的第 0 帧。不是视频流里的上一帧（第 k − 1 帧）。"""
    tid = 37
    _, prev, images = _pair(active, "tubelet=pair,fps=8,pixel_noise=10", tid, system)
    with_prev = systems.get().frames_with_prev(tid)[1]
    base = _base_frames("pixel_noise=10", tid, system)               # [34]，下标 f = tick f−1
    assert prev.shape[0] == 65
    for k in range(65):
        if k % 2 == 0:
            assert np.array_equal(prev[k], base[k//2]) and np.array_equal(images[k], base[k//2+1])
        elif k >= 3:
            assert np.array_equal(prev[k], images[k-2])
        else:
            assert np.array_equal(prev[k], with_prev[0])
    assert np.array_equal(with_prev[1:], images)
    assert any(not np.array_equal(prev[k], images[k-1]) for k in range(1, 65))


def test_fps2_prev_frame_is_the_skipped_tick(active):
    """fps = 2：t_k = 2k，前一帧在奇数 tick 2k − 1（视频里没有这些帧），等于基础数据集的 tick 2k − 1；k = 0 在 t = −1。
    frames(prev=True) 在 fps = 2 时取不到（往前一个帧间隔是 −2 tick），两帧编码照样能编。"""
    tid = 38
    _, prev, images = _pair(active, "tubelet=pair,fps=2,pixel_noise=10", tid)
    base = _base_frames("pixel_noise=10", tid)
    assert prev.shape[0] == 17
    for k in range(17):
        assert np.array_equal(prev[k], base[2*k]) and np.array_equal(images[k], base[2*k+1])
    with pytest.raises(ValueError, match="取不到"):
        systems.get().frames_with_prev(tid)


def test_fps16_prev_frame_noise_rule(active, monkeypatch):
    """fps = 16：t_k = k/4。前一帧 k ≥ 4 等于第 k − 4 帧，k = 0 等于基础数据集的 t = −1，k = 3（−0.25 tick）等于
    frames(prev=True) 的第 0 帧；k = 1、2（−0.75、−0.5 tick）哪一帧都不是，像素噪声用流 13、键 = 帧号。核每次取噪声的（流, 键）。"""
    tid = 36
    calls = []
    original = timing._offgrid_noise

    def record(trajectory_id, key, sigma, size=systems.SIZE, stream=timing.STREAM_OFFGRID_NOISE):
        calls.append((stream, int(key)))
        return original(trajectory_id, key, sigma, size, stream=stream)
    monkeypatch.setattr(timing, "_offgrid_noise", record)
    _, prev, images = _pair(active, "tubelet=pair,fps=16,pixel_noise=10", tid)
    main = [(11, j+1) for j in range(129) if j % 4]                         # frames()：非整 tick 的第 j 帧，键 j+1
    side = [(11, 0) if k == 3 else (13, k) if k < 3 else (11, k-3) for k in range(129) if k % 4]
    assert calls == main+side
    monkeypatch.setattr(timing, "_offgrid_noise", original)
    with_prev = systems.get().frames_with_prev(tid)[1]
    base = _base_frames("pixel_noise=10", tid)
    assert np.array_equal(prev[0], base[0]) and np.array_equal(prev[3], with_prev[0])
    assert all(np.array_equal(prev[k], images[k-4]) for k in range(4, 129))
    for k in (1, 2):
        assert not any(np.array_equal(prev[k], x) for x in (*images, with_prev[0], base[0]))


def test_jitter_prev_frame_is_drawn_at_its_own_time(active):
    """抖动：帧时刻独立写出（流 10 按帧号抽 33 个 U(−1,1)，首尾不动），前一帧 = 时刻 t_k − 1 的状态按这个时刻画
    （状态用 S24b 核过的 timing.simulate，画图用 second_order 本身）。前一帧不是视频流里的上一帧。"""
    tid = 39
    _, prev, images = _pair(active, "tubelet=pair,jitter=0.4,movers=2", tid)
    v = datasets.active().knobs
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1703, 1000+datasets.SYSTEMS.index("pendulum"), tid, 10])))
    xi = g.uniform(-1., 1., 33)
    t = np.arange(33, dtype=np.float64); t[1:32] += .4*xi[1:32]
    assert np.array_equal(timing.frame_times(tid), t)
    sc = timing._scene(systems.get(), tid, v, 32)
    u4 = systems.applied_subticks(tid, v, systems.commanded_controls(tid))
    states = timing.simulate(sc, u4, t-1.)
    with systems.raw_images():
        raw = sc.draw(states, t-1.)
    assert np.array_equal(prev, np.rint(np.clip(raw, 0., 255.)).astype(np.uint8))
    assert any(not np.array_equal(prev[k], images[k-1]) for k in range(1, 33))


@pytest.mark.parametrize("knob", ["exposure=1", "switch_offset=0.5", "exposure=1,fps=8"])
def test_prev_frame_on_the_frame_grid_keeps_exposure_and_switching(knob, active):
    """前一帧落在帧网格上时就是那一帧（同样长的曝光窗口、同一组执行动作）：fps = 4 时是第 k − 1 帧（k = 0 是
    frames(prev=True) 的第 0 帧），fps = 8 时是第 k − 2 帧。"""
    tid = 37
    _, prev, images = _pair(active, "tubelet=pair,"+knob, tid)
    step = datasets.active().knobs.fps//4
    for k in range(step, len(images)):
        assert np.array_equal(prev[k], images[k-step])
    if step == 1:
        assert np.array_equal(prev[0], systems.get().frames_with_prev(tid)[1][0])


def test_pair_frames_refuses_first_order_scene(active):
    active(spec("fps=8", "double_well"))
    with pytest.raises(ValueError, match="二阶场景"):
        timing.pair_frames(systems.get(), 37)


# —— 编码产物 ——

@pytest.mark.parametrize("name", ["fps8", "fps2", "jit"])
def test_timed_pair_cache_features_and_manifest(caches, active, name):
    """时间采样的两帧缓存：第 k 个特征 = 假 tubelet 编码（t_k − 1 的画面, 第 k 帧）；不是（上一帧, 这一帧）也不是单帧复制。
    manifest 除 variant、特征、tubelet、生成源码外与不带 pair 的同一时间采样数据集相同；tubelet 记录逐帧记前一帧。"""
    s = spec(CACHES[name])
    v = s.knobs
    for tid in TEST_IDS:
        _, prev, images = _pair(active, CACHES[name], tid)
        m = _manifest(caches[name]/f"{tid:04d}")
        z = np.load(caches[name]/f"{tid:04d}"/"raw_z.npy")
        L = 32*v.fps//4+1
        assert z.shape == (L, 256, 1408) and m["shape"] == [L, 256, 1408]
        expected = _fake_pair(prev, images)
        assert np.array_equal(z, expected.astype(np.float64).astype(np.float16))
        x = torch.from_numpy(images).permute(0, 3, 1, 2).float()/255.
        dup = tubelet_probe.fake_tubelet_encoder()(torch.stack((x, x), 2)).numpy()
        video = _fake_pair(images[:-1], images[1:])                      # 视频流里的上一帧
        assert np.abs(z[1:]-video).max() > .1 and np.abs(z-dup).max() > .1
        t = np.array([f["time_ticks"] for f in m["frames"]])
        assert np.array_equal(t, timing.frame_times(tid))
        record = m["tubelet"]
        assert record == variants.timed_tubelet_record(
            [dict(frame=k, time_ticks=float(t[k])-1., time_seconds=(float(t[k])-1.)*.25, image_sha256=_sha(prev[k])) for k in range(L)])
        assert record["prev_frames"][0]["time_ticks"] == -1. and "first_prev" not in record
        assert [f["image_sha256"] for f in m["frames"]] == [_sha(x) for x in images]
        assert m["variant"] == s.variant and m["timing"]["fps"] == v.fps
        assert {"src/po_jepa/timing.py", "src/po_jepa/tubelet_probe.py"} <= set(m["generation"]["generator_source_sha256"])
    assert json.loads((caches[name]/"report.json").read_text())["tubelet"] == "pair"


def test_timed_pair_manifest_matches_the_dup_cache(caches):
    for tid in TEST_IDS:
        a, b = _manifest(caches["fps8"]/f"{tid:04d}"), _manifest(caches["dup8"]/f"{tid:04d}")
        assert "tubelet" not in b
        for key in ("frames", "controls", "timing", "shape", "grid", "preprocess_sha256", "split", "rng"):
            assert a[key] == b[key], key


@pytest.mark.parametrize("name", ["r192", "drift"])
def test_space_pair_cache_features_and_manifest(caches, active, name):
    """空间采样的两帧缓存：帧都在整 tick 上，第 t 个特征 = 原生分辨率的假 tubelet 编码（tick t−1, tick t），画面取
    frames_with_prev（R×R，按各自时刻的相机画）。独立写出 2×16×16 tubelet 的投影，不调 native_encoder.fake_clip_encoder。"""
    s = spec(CACHES[name]); R = s.knobs.resolution; G = R//16
    g = torch.Generator().manual_seed(0)
    weight = torch.randn(3*2*16*16, 1408, generator=g)/math.sqrt(3*2*16*16)
    for tid in TEST_IDS:
        active(s)
        _, frames = systems.get().frames_with_prev(tid)
        assert frames.shape == (34, R, R, 3)
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).double()/255.
        video = torch.stack((x[:-1], x[1:]), 2)                            # [33,3,2,R,R]
        patches = torch.stack([video[:, :, :, 16*i:16*i+16, 16*j:16*j+16].reshape(33, -1)
                               for i in range(G) for j in range(G)], 1)    # 行优先 token
        expected = torch.nn.functional.layer_norm(patches.float()@weight, (1408,)).numpy()
        z = np.load(caches[name]/f"{tid:04d}"/"raw_z.npy")
        assert z.shape == (33, G*G, 1408) and np.allclose(z.astype(np.float32), expected, atol=4e-3, rtol=0)
        m = _manifest(caches[name]/f"{tid:04d}")
        assert m["tubelet"] == variants.tubelet_record(_sha(frames[0])) and m["grid"] == [G, G]
        assert [f["image_sha256"] for f in m["frames"]] == [_sha(f) for f in frames[1:]]
        assert m["space"] == space_cache.space_record(s)
        assert {"src/po_jepa/native_encoder.py", "src/po_jepa/tubelet_probe.py"} <= set(m["generation"]["generator_source_sha256"])
    if name == "r192":
        for tid in TEST_IDS:
            a, b = _manifest(caches["r192"]/f"{tid:04d}"), _manifest(caches["dup192"]/f"{tid:04d}")
            assert "tubelet" not in b and all(a[k] == b[k] for k in ("frames", "controls", "space", "grid", "preprocess_sha256"))


def test_fake_clip_encoder_matches_the_256_tubelet_encoder():
    g = torch.Generator().manual_seed(3)
    video = torch.rand(2, 3, 2, 256, 256, generator=g)
    assert torch.equal(native_encoder.fake_clip_encoder()(video), tubelet_probe.fake_tubelet_encoder()(video))
    with pytest.raises(ValueError):
        native_encoder.fake_clip_encoder()(torch.rand(1, 3, 2, 200, 200))


# —— 读取 ——

def test_timing_cache_reads_pair_only_as_its_dataset(caches, real, active, tmp_path):
    active(spec(CACHES["fps8"]))
    raw, m, _ = timing_cache.read_complete(caches["fps8"]/"0037")
    assert raw.shape == (65, 256, 1408) and m["tubelet"]["mode"] == "pair"
    w = timing_cache.read_window(caches["fps8"]/"0037", start_frame=10, frames=4, expected_split="test", device="cpu")
    assert torch.equal(w.inputs.raw_z, torch.from_numpy(np.array(raw[10:11], dtype=np.float32)))
    with pytest.raises(ValueError):                                     # pair 数据集读 dup 缓存
        timing_cache.read_complete(caches["dup8"]/"0037")
    active(spec("fps=8"))
    with pytest.raises(ValueError):                                     # dup 数据集读 pair 缓存
        timing_cache.read_complete(caches["fps8"]/"0037")
    # 调节项字符串自洽、只有 tubelet 记录不对：timing_cache 自己的核对拒绝
    for i, (source, variant, change) in enumerate((
            ("fps8", CACHES["fps8"], lambda m: m.pop("tubelet")),
            ("fps8", CACHES["fps8"], lambda m: m["tubelet"]["prev_frames"][5].update(
                time_ticks=m["frames"][4]["time_ticks"], time_seconds=m["frames"][4]["time_seconds"])),   # 取成视频里的上一帧
            ("fps8", CACHES["fps8"], lambda m: m.update(tubelet=variants.tubelet_record("0"*64))),         # 不带时间采样的记录
            ("dup8", "fps=8", lambda m: m.update(tubelet=_manifest(caches["fps8"]/"0037")["tubelet"])))):
        folder = tmp_path/f"t{i}"
        shutil.copytree(caches[source]/"0037", folder)
        active(spec(variant))
        timing_cache.read_complete(folder)                              # 没改时能读
        _rewrite_manifest(folder, change)
        timing_cache._VERIFIED.clear()
        with pytest.raises(ValueError, match="tubelet"):
            timing_cache.read_complete(folder)


def test_space_cache_reads_pair_only_as_its_dataset(caches, real, active, tmp_path):
    active(spec(CACHES["r192"]))
    raw, m, _ = space_cache.read_complete(caches["r192"]/"0037")
    assert raw.shape == (33, 144, 1408)
    space_cache.read_window(caches["r192"]/"0037", start_tick=5, duration_cycles=4., expected_split="test", device="cpu")
    with pytest.raises(ValueError):
        space_cache.read_complete(caches["dup192"]/"0037")
    active(spec("resolution=192"))
    with pytest.raises(ValueError):
        space_cache.read_complete(caches["r192"]/"0037")
    for i, (source, variant, change) in enumerate((
            ("r192", CACHES["r192"], lambda m: m.pop("tubelet")),
            ("r192", CACHES["r192"], lambda m: m["tubelet"].update(prev_offset_ticks=1)),
            ("dup192", "resolution=192", lambda m: m.update(tubelet=_manifest(caches["r192"]/"0037")["tubelet"])))):
        folder = tmp_path/f"s{i}"
        shutil.copytree(caches[source]/"0037", folder)
        active(spec(variant))
        space_cache.read_complete(folder)
        _rewrite_manifest(folder, change)
        space_cache._VERIFIED.clear()
        with pytest.raises(ValueError, match="tubelet"):
            space_cache.read_complete(folder)


def _timed_changes():
    def entry(k, **kw):
        return lambda r: r["prev_frames"][k].update(**kw)
    return [
        ("mode", lambda r: r.update(mode="dup")),
        ("clip", lambda r: r.update(clip=variants.PAIR_CLIP)),
        ("offset_ticks_float", lambda r: r.update(prev_offset_ticks=-1.)),
        ("offset_seconds", lambda r: r.update(prev_offset_seconds=-.5)),
        ("preprocess", lambda r: r.update(preprocess="x")),
        ("short", lambda r: r["prev_frames"].pop()),
        ("frame_index", entry(3, frame=4)),
        ("time_previous_video_frame", entry(3, time_ticks=1., time_seconds=.25)),   # fps = 8 的第 3 帧在 1.5，上一帧在 1.0
        ("time_int", entry(2, time_ticks=0, time_seconds=0.)),
        ("seconds", entry(2, time_seconds=.5)),
        ("sha", entry(1, image_sha256="A"*64)),
        ("extra_entry_key", entry(1, tick=0)),
        ("extra_key", lambda r: r.update(first_prev={})),
        ("missing_key", lambda r: r.pop("preprocess")),
    ]


@pytest.mark.parametrize("name,change", _timed_changes())
def test_timed_tubelet_record_is_checked_field_by_field(caches, name, change):
    m = _manifest(caches["fps8"]/"0037")
    v = spec(CACHES["fps8"]).knobs
    variants.check_tubelet_record(m, v)
    bad = copy.deepcopy(m)
    change(bad["tubelet"])
    with pytest.raises(ValueError, match="tubelet"):
        variants.check_tubelet_record(bad, v)


def test_record_kind_follows_the_time_sampling_knobs(caches):
    timed, space = _manifest(caches["fps8"]/"0037"), _manifest(caches["r192"]/"0037")
    variants.check_tubelet_record(space, spec(CACHES["r192"]).knobs)
    with pytest.raises(ValueError):                    # 时间采样数据集配不带时间采样的记录
        variants.check_tubelet_record(dict(timed, tubelet=space["tubelet"]), spec(CACHES["fps8"]).knobs)
    with pytest.raises(ValueError):                    # 空间采样数据集配时间采样的记录
        variants.check_tubelet_record(dict(space, tubelet=timed["tubelet"]), spec(CACHES["r192"]).knobs)


def test_finalize_refuses_inconsistent_timed_record(tmp_path, caches):
    out = tmp_path/"fps8"
    shutil.copytree(caches["fps8"], out)
    assert _encode(out, CACHES["fps8"], modes=("finalize",)).returncode == 0
    _rewrite_manifest(out/"0038", lambda m: m["tubelet"]["prev_frames"][7].update(time_ticks=0., time_seconds=0.))
    r = _encode(out, CACHES["fps8"], modes=("finalize",))
    assert r.returncode != 0 and "tubelet 记录" in r.stderr


# —— 真实编码路径（记录输入的替身）——

def _tool_module(name):
    s = importlib.util.spec_from_file_location(name+"_s25b3", ROOT/"tools"/f"{name}.py")
    module = importlib.util.module_from_spec(s); s.loader.exec_module(module)
    return module


class _Recorder:
    def __init__(self):
        self.calls = []

    def encode_clip(self, prev, cur):
        self.calls.append((prev.clone(), cur.clone()))
        return (prev+2*cur).mean((1, 2, 3))[:, None, None].expand(-1, 4, 3).clone()

    def encode_target(self, *a, **k):
        raise AssertionError("两帧编码不调单帧的 encode_target")


@pytest.mark.parametrize("resolution", [None, 192])
def test_real_path_pairs_prev_frames_preprocess_and_batches(monkeypatch, resolution):
    """prev 给出时：第 k 个 tubelet =（prev[k] 的预处理图, frames[k] 的预处理图），每张图前都重置同一个 generator（种子独立写出），
    空间采样按原生分辨率预处理；65 个 tubelet 按 33 分批 [33, 32]；不收帧时刻；prev 形状不对、不是 pair 时拒绝。"""
    enc = _tool_module("encode_v7")
    R = resolution or 256
    seen = []

    def fake_pre(rgb, *, spec, generator, resolution=None):
        seen.append((int(generator.initial_seed()), resolution, rgb.shape))
        x = rgb.permute(0, 3, 1, 2).float()
        return x+float(torch.rand((), generator=generator))             # generator 的状态影响结果：重置与否看得出来
    monkeypatch.setattr(vjepa, "preprocess_rgb", fake_pre)
    monkeypatch.setattr(native_encoder, "preprocess_native", fake_pre)
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (65, R, R, 3), dtype=np.uint8)
    prev = rng.integers(0, 256, (65, R, R, 3), dtype=np.uint8)
    rec = _Recorder()
    out = enc._encode_trajectory(rec, frames, 37, simulation, cache, torch, np, 33, False, None, resolution, pair=True, prev=prev)
    seed = int(np.random.SeedSequence([1703, 1000+datasets.SYSTEMS.index(datasets.active().system), 37, 2])
               .generate_state(1, dtype=np.uint64)[0]) if datasets.active().system != datasets.DEFAULT_SYSTEM else \
        int(np.random.SeedSequence([1703, 37, 2]).generate_state(1, dtype=np.uint64)[0])
    assert seed == simulation.preprocess_seed(37)
    assert len(seen) == 130 and all(s == (seed, resolution, (1, R, R, 3)) for s in seen)
    assert [c[0].shape[0] for c in rec.calls] == [33, 32] and out.shape == (65, 4, 3)
    offset = float(torch.rand((), generator=torch.Generator().manual_seed(seed)))
    before = torch.cat([c[0] for c in rec.calls]); after = torch.cat([c[1] for c in rec.calls])
    assert torch.equal(before, torch.from_numpy(prev).permute(0, 3, 1, 2).float()+offset)
    assert torch.equal(after, torch.from_numpy(frames).permute(0, 3, 1, 2).float()+offset)
    with pytest.raises(ValueError, match="不收帧时刻"):
        enc._encode_trajectory(rec, frames, 37, simulation, cache, torch, np, 33, False, np.zeros(65), resolution, pair=True, prev=prev)
    with pytest.raises(ValueError, match="一一对应"):
        enc._encode_trajectory(rec, frames, 37, simulation, cache, torch, np, 33, False, None, resolution, pair=True, prev=prev[:-1])
    with pytest.raises(ValueError, match="只用于两帧"):
        enc._encode_trajectory(rec, frames, 37, simulation, cache, torch, np, 33, False, None, resolution, prev=prev)


def test_real_space_path_without_prev_pairs_consecutive_frames(monkeypatch):
    """空间采样（prev 不给）：34 帧 R×R，第 k 个 tubelet =（第 k 帧, 第 k+1 帧），原生分辨率预处理。"""
    enc = _tool_module("encode_v7")
    seen = []

    def fake_pre(rgb, *, spec, generator, resolution=None):
        seen.append(resolution)
        return rgb.permute(0, 3, 1, 2).float()
    monkeypatch.setattr(native_encoder, "preprocess_native", fake_pre)
    monkeypatch.setattr(vjepa, "preprocess_rgb", lambda *a, **k: (_ for _ in ()).throw(AssertionError("要按原生分辨率预处理")))
    frames = np.random.default_rng(1).integers(0, 256, (34, 192, 192, 3), dtype=np.uint8)
    rec = _Recorder()
    enc._encode_trajectory(rec, frames, 36, simulation, cache, torch, np, 33, False, None, 192, pair=True)
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    assert seen == [192]*34 and len(rec.calls) == 1
    assert torch.equal(rec.calls[0][0], x[:-1]) and torch.equal(rec.calls[0][1], x[1:])


class TinyPatch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.proj = torch.nn.Conv3d(3, 1408, (2, 16, 16), (2, 16, 16))
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=g)*.01)
            self.proj.bias.zero_()

    def forward(self, video):
        return self.proj(video).flatten(2).transpose(1, 2)


def _loaded():
    arch = vjepa.EncoderArchitectureSpec(vjepa.UPSTREAM_COMMIT, "x", "vit_giant_xformers", 256, 16, 2, 512,
                                         True, True, False, True, True, True)
    report = vjepa.CheckpointReport(Path("tiny"), "0"*64, "target_encoder", (), (), True, arch)
    return vjepa.LoadedEncoder(TinyPatch(), report)


def test_native_clip_encoder():
    """NativeClipEncoder：R = 256 时与 FrozenClipEncoder 逐位相同；两帧相同时与 NativeFrameEncoder 相同；第 (1,2) 个 token =
    前一帧过时间片 0 的权重 + 后一帧过时间片 1 的权重（手算）；换了时间顺序结果不同；分辨率不对时拒收。"""
    loaded = _loaded()
    kw = dict(preprocess=cache.preprocess_spec(), encoder_device=torch.device("cpu"), autocast_dtype=None)
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(2, 3, 256, 256, generator=g), torch.randn(2, 3, 256, 256, generator=g)
    clip256 = native_encoder.NativeClipEncoder(loaded, resolution=256, **kw)
    assert torch.equal(clip256.encode_clip(a, b), tubelet_probe.FrozenClipEncoder(loaded, **kw).encode_clip(a, b))
    w = loaded.encoder.proj.weight
    t = torch.zeros(2, dtype=torch.float64)
    for r in (192, 384):
        e = native_encoder.NativeClipEncoder(loaded, resolution=r, **kw)
        side = r//16
        p, c = torch.randn(2, 3, r, r, generator=g), torch.randn(2, 3, r, r, generator=g)
        out = e.encode_clip(p, c)
        assert out.shape == (2, side*side, 1408) and (e.grid.height, e.grid.width) == (side, side)
        k = side+2
        expect = (torch.einsum("bchw,dchw->bd", p[:, :, 16:32, 32:48], w[:, :, 0])
                  + torch.einsum("bchw,dchw->bd", c[:, :, 16:32, 32:48], w[:, :, 1]))
        assert torch.allclose(out[:, k], expect, atol=1e-4)
        assert not torch.allclose(e.encode_clip(c, p)[:, k], expect, atol=1e-2)
        same = native_encoder.NativeFrameEncoder(loaded, resolution=r, **kw).encode_target(c, target_time_seconds=t).raw_z
        assert torch.equal(e.encode_clip(c, c), same)
        with pytest.raises(ValueError, match="native resolution"):
            e.encode_clip(a, b)
        with pytest.raises(ValueError, match="same batch"):
            e.encode_clip(p, c[:1])


# —— 评测工具 ——

def _job(tmp_path, dataset):
    job = tmp_path/"job"/"run"/"job"; job.mkdir(parents=True)
    (job/"job.json").write_text(json.dumps(dict(seed=1703, updates=10000, dataset=dataset.tag, condition="identity")))
    return tmp_path/"job"


def test_timing_and_space_eval_on_pair_jobs(caches, real, active, tmp_path, monkeypatch):
    """两帧任务（pendulum~tubelet=pair）上的 timing_eval、space_eval：测试集是训练调节项并上采样调节项（含 tubelet=pair），
    读两帧的测试缓存，报告的 tubelet 是 pair；拿单帧复制的测试缓存来评测被拒。"""
    base = spec("tubelet=pair")
    active(base)
    job = _job(tmp_path, base)
    monkeypatch.setattr(v7jobs, "load_for_evaluation", lambda condition, run, device: (IdentityPredictor(), e06_config(), None, {}))
    monkeypatch.setattr(os, "environ", dict(os.environ))
    timing_tool, space_tool = _tool_module("timing_eval"), _tool_module("space_eval")
    out = tmp_path/"timing.json"
    assert timing_tool.main(["--job", str(job), "--test-variant", "fps=8", "--features", str(caches["fps8"]),
                             "--device", "cpu", "--output", str(out)]) == 0
    r = json.loads(out.read_text())
    assert (r["dataset"], r["train_dataset"], r["tubelet"]) == (spec(CACHES["fps8"]).tag, base.tag, "pair") and r["rows"]
    assert datasets.active() == base
    with pytest.raises(ValueError):
        timing_tool.main(["--job", str(job), "--test-variant", "fps=8", "--features", str(caches["dup8"]),
                          "--device", "cpu", "--output", str(tmp_path/"bad.json")])
    out = tmp_path/"space.json"
    assert space_tool.main(["--job", str(job), "--test-variant", "resolution=192", "--features", str(caches["r192"]),
                            "--base-features", str(caches["pair"]), "--device", "cpu", "--output", str(out)]) == 0
    r = json.loads(out.read_text())
    assert (r["dataset"], r["tubelet"]) == (spec(CACHES["r192"]).tag, "pair") and r["rows"]
    with pytest.raises(ValueError):
        space_tool.main(["--job", str(job), "--test-variant", "resolution=192", "--features", str(caches["dup192"]),
                         "--base-features", str(caches["pair"]), "--device", "cpu", "--output", str(tmp_path/"bad2.json")])
