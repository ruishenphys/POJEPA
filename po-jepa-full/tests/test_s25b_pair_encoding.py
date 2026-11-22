"""S25b 第一段：二阶场景的两帧（pair）编码接进 encode_v7 的缓存（docs/segments/S25b.md）。

假特征用 tools/encode_v7.py --fake-encoder（子进程）；读取时把 synthetic 标记换成 real（同 test_s25c_long_horizon）。
数据集是 pendulum:40（测试段 36–39），只编码测试段。期望值都从画面独立重算，不拿 encode_v7 自己的函数当参照。
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from po_jepa import datasets, simulation, systems, tubelet_probe, variants
from po_jepa.runtime import cache, long_cache

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"src"
DUP = datasets.parse("pendulum:40")
PAIR = datasets.parse("pendulum~tubelet=pair:40")
PAIR_LONG = datasets.parse("pendulum~horizon=64,tubelet=pair:40")
DUP_LONG = datasets.parse("pendulum~horizon=64:40")
TEST_IDS = tuple(range(36, 40))


def _clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")}
    env.update(PYTHONPATH=str(SRC), OMP_NUM_THREADS="2")
    return env


def _tool(tool, *args):
    return subprocess.run([sys.executable, "-B", str(ROOT/"tools"/tool), *args], env=_clean_env(), capture_output=True, text=True)


def _encode(out, variant="", system="pendulum", modes=("encode", "finalize")):
    args = ["--system", system, "--trajectories", "40", "--output", str(out)]
    if variant:
        args += ["--variant", variant]
    extra = dict(encode=["--fake-encoder", "--device", "cpu", "--splits", "test"], finalize=["--splits", "test"])
    for mode in modes:
        r = _tool("encode_v7.py", mode, *args, *extra[mode])
        if r.returncode:
            return r
    return r


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    root = tmp_path_factory.mktemp("s25b")
    for name, variant in (("dup", ""), ("pair", "tubelet=pair"), ("pair_long", "horizon=64,tubelet=pair")):
        r = _encode(root/name, variant)
        assert r.returncode == 0, r.stderr
    return root


def _as_real(m):
    m = copy.deepcopy(m)
    m["generation"].update(evidence_role="real_frozen_encoder", encoder_checkpoint_sha256=cache.CHECKPOINT_SHA,
                           encoder_load_report=dict(sha256=cache.CHECKPOINT_SHA, encoder_key="target_encoder", strict=True,
                                                    missing_keys=[], unexpected_keys=[]))
    return m


@pytest.fixture
def real(monkeypatch):
    original, original_long = cache._read_v7_cache, long_cache.check_manifest
    monkeypatch.setattr(cache, "_read_v7_cache", lambda path, m, sha: original(path, _as_real(m), sha))
    monkeypatch.setattr(long_cache, "check_manifest", lambda path, m, sha, spec: original_long(path, _as_real(m), sha, spec))
    monkeypatch.setattr(cache, "_VERIFIED_CACHE", {})
    monkeypatch.setattr(long_cache, "_VERIFIED", {})


@pytest.fixture
def active():
    old = datasets.active()
    yield lambda spec: datasets._set_for_tests(spec)
    datasets._set_for_tests(old)


def _sha(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def _expected_pair(frames_with_prev):
    """独立重算：第 k 个特征 = 假 tubelet 编码器（(前一帧, 后一帧) 按时间顺序，像素 /255）。"""
    x = torch.from_numpy(frames_with_prev).permute(0, 3, 1, 2).float()/255.
    return tubelet_probe.fake_tubelet_encoder()(torch.stack((x[:-1], x[1:]), 2)).numpy()


# —— 调节项 ——

def test_tubelet_knob_parse_format_and_scope():
    assert variants.TUBELETS == ("dup", "pair") and variants.Variant().tubelet == "dup"
    for system in variants.SECOND_ORDER:
        v = variants.parse(system, "tubelet=pair")
        assert v.tubelet == "pair" and variants.format(system, v) == "tubelet=pair"
        assert variants.format(system, variants.parse(system, "tubelet=dup")) == ""      # 默认值不写进规范字符串
        assert variants.format(system, variants.parse(system, "tubelet=pair,horizon=64")) == "horizon=64,tubelet=pair"
        assert variants.tubelet_label(system, v) == "pair" and variants.tubelet_label(system, variants.Variant()) == "dup"
    assert PAIR.tag == "pendulum~tubelet=pair-40" and DUP.tag == "pendulum-40" and PAIR.variant != DUP.variant
    for system in ("double_well", "van_der_pol", "multi_particle", "advection_diffusion", "gray_scott", "push"):
        with pytest.raises(ValueError, match="一阶场景"):
            variants.parse(system, "tubelet=pair")
        assert variants.tubelet_label(system, variants.Variant()) is None
    with pytest.raises(ValueError, match="一阶场景"):
        variants.parse("landscape", "shape=ring,tubelet=pair")
    with pytest.raises(ValueError, match="tubelet"):
        variants.parse("pendulum", "tubelet=triple")
    assert variants.coverage()["tubelet"] == variants.SECOND_ORDER


@pytest.mark.parametrize("knob", ["fps=8", "jitter=0.2", "exposure=0.5", "switch_offset=0.5", "resolution=384", "zoom=0.75"])
def test_pair_refuses_time_and_space_sampling(knob):
    """时间采样的前一帧要另画（固定 0.25 s 前，Jerry 2026-11-22 定）、空间采样要原生分辨率的两帧编码器：第二段接，这里拒绝。"""
    assert variants.parse("pendulum", knob) is not None          # 不带 pair 时这些调节项本身可用
    with pytest.raises(ValueError, match="还没接"):
        variants.parse("pendulum", f"{knob},tubelet=pair")


@pytest.mark.parametrize("system", variants.SECOND_ORDER)
def test_pair_knob_changes_no_state_frame_or_control(system, active):
    """tubelet 只改编码：同一编号的轨迹在 pair、dup 数据集下的状态、34 帧画面、下达动作逐位相同。"""
    tid = 37
    out = {}
    for name in ("", "tubelet=pair"):
        active(datasets.parse(f"{system}{'~'+name if name else ''}:40"))
        s = systems.get()
        states, frames = s.frames_with_prev(tid)
        states33, frames33 = s.frames(tid)
        c = systems.commanded_controls(tid)
        out[name] = (states, frames, states33, frames33, c.starts_ticks, c.ends_ticks, c.values)
    assert out[""][1].shape == (34, 256, 256, 3)
    for a, b in zip(out[""], out["tubelet=pair"]):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    assert np.array_equal(out[""][1][1:], out[""][3])        # frames() = frames_with_prev() 的下标 1…33


# —— 编码产物 ——

def test_pair_cache_features_are_the_time_ordered_two_frame_tubelets(caches, active):
    active(PAIR); system = systems.get()
    for tid in TEST_IDS:
        states, frames = system.frames_with_prev(tid)
        z = np.load(caches/"pair"/f"{tid:04d}"/"raw_z.npy")
        expected = _expected_pair(frames)
        assert z.shape == (33, 256, 1408) and z.dtype == np.float16
        assert np.array_equal(z, expected.astype(np.float64).astype(np.float16))
        # 顺序是 (前, 后) 不是 (后, 前)；也不是单帧复制
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()/255.
        enc = tubelet_probe.fake_tubelet_encoder()
        reversed_ = enc(torch.stack((x[1:], x[:-1]), 2)).numpy().astype(np.float16)
        dup = enc(torch.stack((x[1:], x[1:]), 2)).numpy().astype(np.float16)
        assert not np.array_equal(z, reversed_) and not np.array_equal(z, dup)
        assert np.abs(z.astype(np.float64)-dup).max() > 1e-2
        # 状态仍是 tick 0…32（与 dup 的缓存相同），不含 t = −1
        assert np.array_equal(np.load(caches/"pair"/f"{tid:04d}"/"states.npy"), states[1:])
        assert np.array_equal(np.load(caches/"dup"/f"{tid:04d}"/"states.npy"), states[1:])


def test_pair_feature_depends_on_previous_frame_only_through_the_tubelet(caches, active):
    """tick t 的特征只由第 t−1、t 两帧决定：与独立编码单个 tubelet 相同（不看别的帧）。"""
    active(PAIR); tid = 38
    _, frames = systems.get().frames_with_prev(tid)
    z = np.load(caches/"pair"/f"{tid:04d}"/"raw_z.npy")
    for t in (0, 1, 17, 32):
        single = _expected_pair(frames[t:t+2])          # 下标 t = tick t−1，下标 t+1 = tick t
        assert single.shape == (1, 256, 1408)
        assert np.abs(z[t].astype(np.float64)-single[0]).max() < 2e-3     # float16 存储 + 分批 BLAS 的舍入


def test_pair_manifest_records(caches, active):
    active(PAIR); tid = 36
    _, frames = systems.get().frames_with_prev(tid)
    m = json.loads((caches/"pair"/f"{tid:04d}"/"manifest.json").read_text())
    d = json.loads((caches/"dup"/f"{tid:04d}"/"manifest.json").read_text())
    assert m["variant"] == "tubelet=pair" and "variant" not in d and "tubelet" not in d
    t = m["tubelet"]
    assert t["mode"] == "pair" and t["prev_offset_ticks"] == -1 and t["prev_offset_seconds"] == -.25
    assert t["first_prev"] == dict(tick=-1, time_seconds=-.25, image_sha256=_sha(frames[0]))
    assert [f["tick"] for f in m["frames"]] == list(range(33))
    assert [f["image_sha256"] for f in m["frames"]] == [_sha(frames[k+1]) for k in range(33)]
    assert m["frames"] == d["frames"] and m["controls"] == d["controls"] and m["shape"] == d["shape"] == [33, 256, 1408]
    assert m["preprocess_sha256"] == d["preprocess_sha256"] == cache.v7_preprocess_sha256()
    assert "src/po_jepa/tubelet_probe.py" in m["generation"]["generator_source_sha256"]
    assert "src/po_jepa/tubelet_probe.py" not in d["generation"]["generator_source_sha256"]
    report = json.loads((caches/"pair"/"report.json").read_text())
    assert report["tubelet"] == "pair" and report["variant"] == "tubelet=pair"
    assert "tubelet" not in json.loads((caches/"dup"/"report.json").read_text())


def test_dup_encoding_of_second_order_scene_is_unchanged(caches, active):
    """不带 tubelet=pair：仍是单帧编码（encode_v7 的单帧假编码器，独立重算）。"""
    active(DUP); tid = 39
    _, frames = systems.get().frames(tid)
    g = torch.Generator().manual_seed(0)
    weight = torch.randn(3*16*16, 1408, generator=g)/16.
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()/255.
    patches = x.unfold(2, 16, 16).unfold(3, 16, 16).permute(0, 2, 3, 1, 4, 5).reshape(33, 256, -1)
    expected = torch.nn.functional.layer_norm(patches@weight, (1408,)).numpy()
    z = np.load(caches/"dup"/f"{tid:04d}"/"raw_z.npy")
    assert np.array_equal(z, expected.astype(np.float64).astype(np.float16))


def test_long_horizon_pair_prefix_matches_main_pair_cache(caches, active):
    active(PAIR_LONG); tid = 37
    _, frames = systems.get().frames_with_prev(tid)
    assert frames.shape[0] == 66
    z = np.load(caches/"pair_long"/f"{tid:04d}"/"raw_z.npy")
    assert z.shape == (65, 256, 1408)
    assert np.array_equal(z, _expected_pair(frames).astype(np.float64).astype(np.float16))
    short = np.load(caches/"pair"/f"{tid:04d}"/"raw_z.npy")
    assert np.abs(z[:33].astype(np.float64)-short).max() < 2e-3
    m = json.loads((caches/"pair_long"/f"{tid:04d}"/"manifest.json").read_text())
    assert m["tubelet"]["first_prev"]["image_sha256"] == _sha(frames[0]) and len(m["frames"]) == 65


# —— 真实编码路径（fake=False）：预处理、tubelet 的组成和分批 ——

class _SpyClip:
    def __init__(self):
        self.calls = []

    def encode_clip(self, prev, cur):
        self.calls.append((prev.clone(), cur.clone()))
        return (prev.float().mean((1, 2, 3))+2*cur.float().mean((1, 2, 3)))[:, None, None].expand(-1, 2, 3).double()

    def encode_target(self, *a, **k):
        raise AssertionError("两帧编码不能走单帧的 encode_target")


def _fake_preprocess(rgb, *, spec, generator):
    assert spec == cache.preprocess_spec() and rgb.shape[0] == 1
    noise = torch.rand(1, generator=generator, dtype=torch.float64)
    return rgb.permute(0, 3, 1, 2)[:, :, ::64, ::64].double()+noise


def _real_path(frames, tid, batch):
    sys.path.insert(0, str(ROOT/"tools"))
    import encode_v7
    spy = _SpyClip()
    z = encode_v7._encode_trajectory(spy, frames, tid, simulation, cache, torch, np, batch, False, pair=True)
    return spy, z


def test_real_pair_path_clips_preprocess_and_batches(active, monkeypatch):
    from po_jepa import vjepa
    monkeypatch.setattr(vjepa, "preprocess_rgb", _fake_preprocess)
    tid = 37
    active(PAIR); _, frames34 = systems.get().frames_with_prev(tid)

    def pre(frame):       # 每帧单独预处理，之前把这条轨迹的 generator 重置到同一状态（流 2 的种子，独立写出）
        g = torch.Generator(device="cpu").manual_seed(int(np.random.SeedSequence(simulation.seed_entropy(tid, 2)).generate_state(1)[0]))
        return _fake_preprocess(torch.from_numpy(frame[None]), spec=cache.preprocess_spec(), generator=g)[0]
    short, z = _real_path(frames34, tid, 33)
    assert [p.shape[0] for p, _ in short.calls] == [33] and z.shape == (33, 2, 3)
    prev, cur = short.calls[0]
    for k in (0, 1, 16, 32):
        assert torch.equal(prev[k], pre(frames34[k])) and torch.equal(cur[k], pre(frames34[k+1]))
    assert torch.equal(prev[1:], cur[:-1])                      # 第 k 个 tubelet 的后一帧就是第 k+1 个的前一帧
    assert not torch.equal(prev, cur)
    # 输出第 k 行 = 替身编码器对第 k 个 tubelet 的值（行序没乱）
    want = (prev.float().mean((1, 2, 3))+2*cur.float().mean((1, 2, 3))).double().numpy()
    assert np.array_equal(z[:, 0, 0], want)
    # 长时程：66 帧 → 65 个 tubelet，按 33 分批，第一批与主缓存那一批逐位相同
    active(PAIR_LONG); _, frames66 = systems.get().frames_with_prev(tid)
    spy, z = _real_path(frames66, tid, 33)
    assert [p.shape[0] for p, _ in spy.calls] == [33, 32] and z.shape == (65, 2, 3)
    assert torch.equal(spy.calls[0][0], prev) and torch.equal(spy.calls[0][1], cur)
    assert torch.equal(spy.calls[1][0][0], spy.calls[0][1][-1])
    assert torch.equal(spy.calls[1][1][-1], pre(frames66[65]))
    # 分批不落下也不重复
    spy, z = _real_path(frames34, tid, 5)
    assert [p.shape[0] for p, _ in spy.calls] == [5]*6+[3] and np.array_equal(z[:, 0, 0], want)


def test_pair_path_argument_checks():
    sys.path.insert(0, str(ROOT/"tools"))
    import encode_v7
    one = np.zeros((1, 256, 256, 3), dtype=np.uint8)
    two = np.zeros((2, 256, 256, 3), dtype=np.uint8)
    enc = tubelet_probe.fake_tubelet_encoder()
    with pytest.raises(ValueError, match="至少要 2 帧"):
        encode_v7._encode_trajectory(enc, one, 0, simulation, cache, torch, np, 33, True, pair=True)
    with pytest.raises(ValueError, match="没接"):
        encode_v7._encode_trajectory(enc, two, 0, simulation, cache, torch, np, 33, True, np.array([0., 1.]), pair=True)
    with pytest.raises(ValueError, match="没接"):
        encode_v7._encode_trajectory(enc, two, 0, simulation, cache, torch, np, 33, True, None, 384, pair=True)
    assert encode_v7._encode_trajectory(enc, two, 0, simulation, cache, torch, np, 33, True, pair=True).shape == (1, 256, 1408)


def test_frozen_clip_encoder_is_the_production_pair_encoder():
    """encode_v7 的真实两帧路径用 S22 的 FrozenClipEncoder；小型 Conv3d tubelet 编码器上核 (前, 后) 的时间顺序和 dup 等价。"""
    from po_jepa import vjepa

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv3d(3, 1408, (2, 16, 16), (2, 16, 16))

        def forward(self, video):
            return self.proj(video).flatten(2).transpose(1, 2)
    torch.manual_seed(0)
    clip = object.__new__(tubelet_probe.FrozenClipEncoder)
    clip.encoder = Tiny().requires_grad_(False).eval()
    clip.encoder_device = torch.device("cpu")
    a, b = torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256)
    w = clip.encoder.proj.weight
    manual = (torch.nn.functional.conv2d(a, w[:, :, 0], stride=16)+torch.nn.functional.conv2d(b, w[:, :, 1], stride=16)
              + clip.encoder.proj.bias[None, :, None, None]).flatten(2).transpose(1, 2)
    out = clip.encode_clip(a, b)
    assert out.shape == (2, 256, 1408) and torch.allclose(out, manual, atol=1e-4)
    assert not torch.allclose(out, clip.encode_clip(b, a), atol=1e-2)
    frame = object.__new__(vjepa.FrozenFrameEncoder)
    frame.encoder, frame.encoder_device = clip.encoder, clip.encoder_device
    assert torch.equal(clip.encode_clip(b, b), frame._encode(b, torch.zeros(2, dtype=torch.float64)))


# —— 缓存读取：pair 与 dup 互相读不到 ——

def test_pair_and_dup_caches_are_not_interchangeable(caches, real, active):
    kw = dict(start_tick=5, duration_cycles=4., expected_split="test", device="cpu")
    active(PAIR)
    w = cache.read_window(caches/"pair"/"0037", **kw)
    z = np.load(caches/"pair"/"0037"/"raw_z.npy").astype(np.float32)
    assert torch.equal(w.inputs.raw_z[0], torch.from_numpy(z[5])) and torch.equal(w.target.raw_z[0], torch.from_numpy(z[9]))
    with pytest.raises(ValueError, match="mismatch"):
        cache.read_window(caches/"dup"/"0037", **kw)
    active(DUP)
    # cache 的进程内核验缓存按路径记、键里没有数据集（正式入口一个进程只有一个数据集）；测试里换数据集要先清掉
    cache._VERIFIED_CACHE.clear()
    assert cache.read_window(caches/"dup"/"0037", **kw).inputs.raw_z.shape == (1, 256, 1408)
    with pytest.raises(ValueError, match="mismatch"):
        cache.read_window(caches/"pair"/"0037", **kw)


def test_feature_root_metadata_is_per_tubelet(caches, active):
    from po_jepa.runtime import data
    active(PAIR)
    assert data._v7_cache_metadata(caches/"pair", ("test",))["variant"] == "tubelet=pair"
    with pytest.raises(ValueError, match="not a complete encode"):
        data._v7_cache_metadata(caches/"dup", ("test",))
    active(DUP)
    with pytest.raises(ValueError, match="not a complete encode"):
        data._v7_cache_metadata(caches/"pair", ("test",))


def test_long_cache_checks_variant_and_tubelet_record(caches, real, active, tmp_path):
    active(PAIR_LONG)
    raw, m, _ = long_cache.read_complete(caches/"pair_long"/"0037")
    assert raw.shape == (65, 256, 1408) and m["tubelet"]["mode"] == "pair"
    active(DUP_LONG)
    with pytest.raises(ValueError, match="mismatch"):
        long_cache.read_complete(caches/"pair_long"/"0037")
    # 只动 tubelet 记录（其余自洽）：pair 数据集缺记录、记录的 mode 不对、dup 数据集多出记录，都拒绝
    path = caches/"pair_long"/"0037"
    good = json.loads((path/"manifest.json").read_text())
    spec = PAIR_LONG
    active(PAIR_LONG)
    assert long_cache.check_manifest(path, good, "x", spec)[1] is not None
    bad = copy.deepcopy(good); del bad["tubelet"]
    with pytest.raises(ValueError, match="tubelet record"):
        long_cache.check_manifest(path, bad, "x", spec)
    bad = copy.deepcopy(good); bad["tubelet"]["mode"] = "dup"
    with pytest.raises(ValueError, match="tubelet record"):
        long_cache.check_manifest(path, bad, "x", spec)
    active(DUP_LONG)
    bad = copy.deepcopy(good); bad["variant"] = "horizon=64"
    with pytest.raises(ValueError, match="tubelet record"):
        long_cache.check_manifest(path, bad, "x", DUP_LONG)


GOOD_SHA = "ab"*32


@pytest.mark.parametrize("change", [
    lambda r: r.clear() or r.update(mode="pair"),                              # 只有 mode
    lambda r: r.update(clip="(tick t, tick t-1)"),                             # 时间顺序
    lambda r: r.update(prev_offset_ticks=1),
    lambda r: r.update(prev_offset_seconds=.25),
    lambda r: r.update(prev_offset_ticks=-1.),                                 # 类型
    lambda r: r.update(preprocess="independent crops"),
    lambda r: r["first_prev"].update(tick=0),
    lambda r: r["first_prev"].update(time_seconds=0.),
    lambda r: r["first_prev"].update(image_sha256="xyz"),
    lambda r: r["first_prev"].update(image_sha256="AB"*32),
    lambda r: r.pop("first_prev"),
    lambda r: r.update(first_prev="x"),
    lambda r: r.update(extra=1),
])
def test_tubelet_record_is_checked_field_by_field(change):
    """Codex 审核 F3：记录逐项等于编码定义才收；每次只改一项，其余不变（没改时通过）。"""
    pair, dup = variants.parse("pendulum", "tubelet=pair"), variants.Variant()
    good = variants.tubelet_record(GOOD_SHA)
    assert good["prev_offset_ticks"] == -1 and good["prev_offset_seconds"] == -.25 and "t-1, tick t" in good["clip"]
    assert good["first_prev"] == dict(tick=-1, time_seconds=-.25, image_sha256=GOOD_SHA)
    variants.check_tubelet_record(dict(tubelet=good), pair)
    variants.check_tubelet_record({}, dup)
    bad = copy.deepcopy(good); change(bad)
    with pytest.raises(ValueError, match="tubelet"):
        variants.check_tubelet_record(dict(tubelet=bad), pair)
    for m in ({}, dict(tubelet=None), dict(tubelet="pair")):
        with pytest.raises(ValueError, match="tubelet"):
            variants.check_tubelet_record(m, pair)
    with pytest.raises(ValueError, match="tubelet"):
        variants.check_tubelet_record(dict(tubelet=good), dup)


def test_long_cache_and_finalize_use_the_full_record_check(caches, real, active, tmp_path):
    path = caches/"pair_long"/"0037"
    good = json.loads((path/"manifest.json").read_text())
    active(PAIR_LONG)
    bad = copy.deepcopy(good); bad["tubelet"] = dict(mode="pair")
    with pytest.raises(ValueError, match="tubelet record"):
        long_cache.check_manifest(path, bad, "x", PAIR_LONG)
    bad = copy.deepcopy(good); bad["tubelet"]["prev_offset_ticks"] = 1
    with pytest.raises(ValueError, match="tubelet record"):
        long_cache.check_manifest(path, bad, "x", PAIR_LONG)
    out = tmp_path/"pair"
    assert _encode(out, "tubelet=pair").returncode == 0
    _rewrite_manifest(out/"0036", lambda m: m.update(tubelet=dict(mode="pair")))
    r = _encode(out, "tubelet=pair", modes=("finalize",))
    assert r.returncode != 0 and "tubelet 记录" in r.stderr
    _rewrite_manifest(out/"0036", lambda m: m.update(tubelet=dict(good["tubelet"], clip="(tick t, tick t-1)")))
    r = _encode(out, "tubelet=pair", modes=("finalize",))
    assert r.returncode != 0 and "tubelet 记录" in r.stderr


# —— transfer：测试集沿用训练数据集的 tubelet（Codex 审核 F1） ——

def test_transfer_inherits_tubelet_and_refuses_overrides(tmp_path, active, monkeypatch):
    from po_jepa.runtime import v7jobs, workflow
    for text in ("tubelet=dup", "tubelet=pair", "gain=2,tubelet=dup", "tubelet=pair,gain=2", " tubelet=dup"):
        with pytest.raises(ValueError, match="不能写 tubelet"):
            workflow.refuse_tubelet_test_variant(text)
    workflow.refuse_tubelet_test_variant("gain=2,damping=0.6")
    # 正式入口和 CPU 冒烟在调 transfer_job 之前都先过这道检查
    source = (SRC/"po_jepa/runtime/workflow.py").read_text(encoding="utf-8")
    assert source.index("refuse_tubelet_test_variant(args.test_variant)") < source.index("report=v7jobs.transfer_job(")
    smoke = (ROOT/"tools/smoke_v7_cpu.py").read_text(encoding="utf-8")
    assert smoke.index("workflow.refuse_tubelet_test_variant(a.transfer[0])") < smoke.index("v7jobs.transfer_job(")
    # 只改别的调节项时测试集继承训练数据集的 pair（走到建目录、载入模型那一步，记下它要读的测试数据集）
    seen = {}

    def stop(condition, training_run, device):
        raise RuntimeError("stop before loading")
    monkeypatch.setattr(v7jobs, "load_for_evaluation", stop)
    original = datasets.DatasetSpec

    def spy(system, count, variant=""):
        seen["variant"] = variant
        return original(system, count, variant)
    monkeypatch.setattr(v7jobs.data.datasets, "DatasetSpec", spy)
    active(PAIR)
    with pytest.raises(RuntimeError, match="stop before loading"):
        v7jobs.transfer_job(tmp_path, tmp_path/"out", torch.device("cpu"), "identity", training_run=None, test_variant="gain=2")
    assert seen["variant"] == "gain=2,tubelet=pair"


def test_noise_floor_tool_refuses_pair_dataset(tmp_path):
    """Codex 审核 F2：ac_context_check floor 的重编码是单帧复制，pair 数据集先拒绝（在读缓存和建编码器之前）。"""
    r = _tool("ac_context_check.py", "floor", "--system", "pendulum", "--trajectories", "40", "--variant", "tubelet=pair",
              "--features", str(tmp_path), "--upstream-root", str(tmp_path), "--checkpoint", str(tmp_path/"x.pt"),
              "--device", "cpu", "--count", "1", "--output", str(tmp_path/"floor.json"))
    assert r.returncode != 0 and "tubelet=pair" in r.stderr and "单帧复制" in r.stderr and not (tmp_path/"floor.json").exists()


def _rewrite_manifest(folder, change):
    m = json.loads((folder/"manifest.json").read_text())
    change(m)
    for name in ("manifest.json", "COMPLETE.json"):
        (folder/name).unlink()
    cache._write_json(folder/"manifest.json", m)
    cache._write_json(folder/"COMPLETE.json", dict(manifest_sha256=cache.file_sha(folder/"manifest.json")))


def test_finalize_refuses_inconsistent_tubelet_record(tmp_path):
    """manifest 自洽（SHA 重算）但 tubelet 记录与调节项不符：finalize 拒绝。先确认没改时 finalize 通过。"""
    out = tmp_path/"pair"
    assert _encode(out, "tubelet=pair").returncode == 0
    _rewrite_manifest(out/"0036", lambda m: m.pop("tubelet"))
    r = _encode(out, "tubelet=pair", modes=("finalize",))
    assert r.returncode != 0 and "tubelet 记录" in r.stderr
    _rewrite_manifest(out/"0036", lambda m: m.update(tubelet=dict(mode="dup")))
    r = _encode(out, "tubelet=pair", modes=("finalize",))
    assert r.returncode != 0 and "tubelet 记录" in r.stderr
    out = tmp_path/"dup"
    assert _encode(out, "").returncode == 0
    _rewrite_manifest(out/"0036", lambda m: m.update(tubelet=dict(mode="pair")))
    r = _encode(out, "", modes=("finalize",))
    assert r.returncode != 0 and "tubelet 记录" in r.stderr
    # 整个目录是另一种 tubelet 编的：调节项不符，finalize 拒绝
    r = _encode(tmp_path/"pair", "", modes=("finalize",))
    assert r.returncode != 0


def test_encode_refuses_pair_for_first_order_scene(tmp_path):
    r = _encode(tmp_path/"x", "tubelet=pair", system="double_well")
    assert r.returncode != 0 and "一阶场景" in r.stderr and not (tmp_path/"x").exists()


# —— 在线编码的工具：两帧数据集先拒绝（第二段接） ——

def test_probe_and_plan_refuse_pair_datasets(tmp_path):
    r = _tool("probe_actions.py", "--system", "pendulum", "--trajectories", "40", "--variant", "tubelet=pair",
              "--output", str(tmp_path/"p.json"), "--device", "cpu", "--fake-encoder", "--max-trajectories", "1")
    assert r.returncode != 0 and "tubelet=pair" in r.stderr and "还没接" in r.stderr and not (tmp_path/"p.json").exists()
    text = (ROOT/"tools/plan_v7.py").read_text(encoding="utf-8")
    guard = text.index('if checked.knobs.tubelet == "pair":')
    assert text.index("for checked in (base, test_spec):") < guard < text.index("comparison.build_condition(condition, device)")
