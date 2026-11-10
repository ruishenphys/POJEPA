"""S25e：空间采样测试集接编码、缓存和评测（docs/segments/S25e.md）。

假特征用 tools/encode_v7.py --fake-encoder（子进程，同 S25c、S25d），读取时把 synthetic 标记换成 real。数据集是 double_well:40
（测试段 36–39；测试段不超过 20 条，所以每条都是 5 个起点 tick 0、5、10、15、20），只编码测试段。
"""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from po_jepa import datasets, native_encoder, protocols, systems, variants, vjepa
from po_jepa.baselines import IdentityPredictor
from po_jepa.runtime import cache, comparison, data, space_cache, space_eval, v7jobs
from po_jepa.runtime.addon import d4
from test_training import e06_config

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"src"
BASE = datasets.parse("double_well:40")
R192 = datasets.parse("double_well~resolution=192:40")
R384 = datasets.parse("double_well~resolution=384:40")
ZOOM = datasets.parse("double_well~zoom=0.75:40")
DRIFT = datasets.parse("double_well~drift=0.5,pan=1,zoom=0.75:40")
TEST_IDS = tuple(range(36, 40))
VARIANTS = {"base": "", "r192": "resolution=192", "r384": "resolution=384", "zoom": "zoom=0.75", "drift": "drift=0.5,pan=1,zoom=0.75"}


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
    root = tmp_path_factory.mktemp("s25e")
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
    original, original_s = cache._read_v7_cache, space_cache.check_manifest
    monkeypatch.setattr(cache, "_read_v7_cache", lambda path, m, sha: original(path, _as_real(m), sha))
    monkeypatch.setattr(space_cache, "check_manifest", lambda path, m, sha, spec: original_s(path, _as_real(m), sha, spec))
    monkeypatch.setattr(cache, "_VERIFIED_CACHE", {})
    monkeypatch.setattr(space_cache, "_VERIFIED", {})


@pytest.fixture
def active():
    old = datasets.active()
    yield lambda spec: datasets._set_for_tests(spec)
    datasets._set_for_tests(old)


def _manifest(root, tid):
    return json.loads((root/f"{tid:04d}"/"manifest.json").read_text())


def _tool_module(name):
    spec = importlib.util.spec_from_file_location(f"{name}_tool_s25e", ROOT/f"tools/{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


# —— 测试项 ——

def test_space_tests_are_valid_for_every_scene():
    assert space_eval.TESTS == protocols.SPACE_TESTS == (
        "resolution=192", "resolution=320", "resolution=384", "zoom=0.75", "pan=1,zoom=0.75", "drift=0.5,pan=1,zoom=0.75")
    for scene in protocols.ALL_SCENES:
        for t in protocols.SPACE_TESTS:
            system = scene.partition("~")[0]
            v = variants.parse(system, protocols.merged(scene, t))
            assert v.changes_space() and not v.changes_timing() and v.horizon == 32
            assert protocols.merged(scene, t).count("zoom=") <= 1
    assert {variants.parse("double_well", t).resolution for t in protocols.SPACE_TESTS} == {192, 256, 320, 384}


# —— 编码 ——

def test_encode_space_needs_test_split_horizon_32_and_no_time_sampling(tmp_path):
    r = _encode(tmp_path/"a", "resolution=192", splits="train,test")
    assert r.returncode != 0 and "--splits test" in r.stderr and "space_eval" in r.stderr and not (tmp_path/"a").exists()
    r = _encode(tmp_path/"b", "resolution=192,horizon=64")
    assert r.returncode != 0 and "horizon=32" in r.stderr and not (tmp_path/"b").exists()
    r = _encode(tmp_path/"c", "fps=8,zoom=0.75")
    assert r.returncode != 0 and "空间采样" in r.stderr and "时间采样" in r.stderr and not (tmp_path/"c").exists()


def test_space_cache_layout(caches):
    m = _manifest(caches["r192"], 37)
    assert m["shape"] == [33, 144, 1408] and m["grid"] == [12, 12] and m["variant"] == "resolution=192"
    assert m["space"] == dict(resolution=192, zoom=1., pan=0., drift=0.)
    assert m["generation"]["preprocess"] == dict(cache._v7_preprocess_record(), crop_size=192, native_resolution=192)
    assert m["preprocess_sha256"] == space_cache.preprocess_sha256(192) != cache.v7_preprocess_sha256()
    assert [f["tick"] for f in m["frames"]] == list(range(33)) and m["frames"][7]["time_seconds"] == 1.75
    for name in ("src/po_jepa/camera.py", "src/po_jepa/timing.py", "src/po_jepa/native_encoder.py"):
        assert name in m["generation"]["generator_source_sha256"]
    assert _manifest(caches["r384"], 37)["shape"] == [33, 576, 1408] and _manifest(caches["r384"], 37)["grid"] == [24, 24]
    z = _manifest(caches["zoom"], 37)
    assert z["shape"] == [33, 256, 1408] and z["grid"] == [16, 16] and z["space"]["zoom"] == .75
    assert z["preprocess_sha256"] == space_cache.preprocess_sha256(256) != cache.v7_preprocess_sha256()  # 256 也和 v7 的摘要不同
    d = _manifest(caches["drift"], 37)
    assert d["space"] == dict(resolution=256, zoom=.75, pan=1., drift=.5)
    base = _manifest(caches["base"], 37)                            # 不带空间采样：manifest 格式不变
    assert "space" not in base and base["grid"] == [16, 16] and base["preprocess_sha256"] == cache.v7_preprocess_sha256()
    assert base["generation"]["preprocess"] == cache._v7_preprocess_record()
    assert "src/po_jepa/native_encoder.py" not in base["generation"]["generator_source_sha256"]
    assert base["controls"] == m["controls"] == z["controls"]       # 空间采样不改动作


def test_space_cache_images_and_features(caches, active):
    """画面哈希 = 空间采样数据集 frames() 的画面（R×R）；特征 = 原生分辨率假编码器的输出；R = 256 时与 v7 的假编码器相同。"""
    enc = _tool_module("encode_v7")
    for name, spec, r in (("r192", R192, 192), ("r384", R384, 384), ("zoom", ZOOM, 256)):
        active(spec)
        _, frames = systems.get().frames(39)
        assert frames.shape == (33, r, r, 3)
        m = _manifest(caches[name], 39)
        assert [f["image_sha256"] for f in m["frames"]] == [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]
        x = torch.from_numpy(frames[[0, 17, 32]]).permute(0, 3, 1, 2).float()/255.
        stored = np.load(caches[name]/"0039"/"raw_z.npy")
        expect = native_encoder.fake_encoder()(x).numpy()
        assert stored.shape == (33, (r//16)**2, 1408)
        assert np.allclose(stored[[0, 17, 32]].astype(np.float32), expect.astype(np.float16).astype(np.float32), atol=2e-3)
        # 独立核 token 顺序：第 k 个 token 是第 (k // G) 行、第 (k % G) 列的 16×16 块
        g = r//16; k = g+3
        patch = x[1, :, 16:32, 48:64].permute(1, 2, 0)                                           # 行 1、列 3（HWC 展平前的块）
        manual = torch.nn.functional.layer_norm(x[1:2, :, 16:32, 48:64].reshape(1, 3*256)@_fake_weight(), (1408,))
        assert patch.shape == (16, 16, 3) and torch.allclose(torch.from_numpy(expect[1, k]), manual[0], atol=1e-5)
    base = np.load(caches["base"]/"0039"/"raw_z.npy")
    x = torch.rand(2, 3, 256, 256, generator=torch.Generator().manual_seed(3))
    assert torch.equal(native_encoder.fake_encoder()(x), enc._fake_encoder(torch)(x))
    zb = _manifest(caches["base"], 39)
    assert not set(f["image_sha256"] for f in _manifest(caches["zoom"], 39)["frames"]) & set(f["image_sha256"] for f in zb["frames"])
    assert not np.allclose(np.load(caches["zoom"]/"0039"/"raw_z.npy").astype(np.float32), base.astype(np.float32), atol=1e-2)


def _fake_weight():
    g = torch.Generator().manual_seed(0)
    return torch.randn(3*16*16, 1408, generator=g)/16.


# —— 原生分辨率的预处理和编码器 ——

STUB = textwrap.dedent('''
    """测试替身：按上游 vjepa2 提交 204698b 的 src/datasets/utils/video/transforms.py 同名函数写（十次尝试、中心回退、双线性）。"""
    import math, random
    import numpy as np
    import torch

    def _get_param_spatial_crop(scale, ratio, height, width, num_repeat=10, log_scale=True, switch_hw=False):
        for _ in range(num_repeat):
            area = height * width
            target_area = random.uniform(*scale) * area
            if log_scale:
                log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
                aspect_ratio = math.exp(random.uniform(*log_ratio))
            else:
                aspect_ratio = random.uniform(*ratio)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if np.random.uniform() < 0.5 and switch_hw:
                w, h = h, w
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width; h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height; w = int(round(h * max(ratio)))
        else:
            w = width; h = height
        return (height - h) // 2, (width - w) // 2, h, w

    def random_resized_crop(images, target_height, target_width, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        height, width = images.shape[2], images.shape[3]
        i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
        cropped = images[:, :, i:i + h, j:j + w]
        return torch.nn.functional.interpolate(cropped, size=(target_height, target_width), mode="bilinear", align_corners=False)
''')


@pytest.fixture
def official(tmp_path, monkeypatch):
    """把替身绑定成 vjepa._bound_video_transforms 要的显式 src 包（测试结束后从 sys.modules 移除）。"""
    from importlib.machinery import ModuleSpec
    root = tmp_path/"upstream"/"src"
    (root/"datasets/utils/video").mkdir(parents=True)
    (root/"datasets/utils/video/transforms.py").write_text(STUB)
    for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
        monkeypatch.delitem(sys.modules, name)
    spec = ModuleSpec("src", loader=None, is_package=True)
    spec.submodule_search_locations = [str(root)]
    monkeypatch.setitem(sys.modules, "src", importlib.util.module_from_spec(spec))
    yield
    for name in [n for n in sys.modules if n.startswith("src.")]:
        del sys.modules[name]


def test_preprocess_native(official):
    spec = cache.preprocess_spec()
    g = torch.Generator().manual_seed(5)
    rgb256 = torch.randint(0, 256, (2, 256, 256, 3), dtype=torch.uint8, generator=g)
    a = vjepa.preprocess_rgb(rgb256, spec=spec, generator=torch.Generator().manual_seed(11))
    b = native_encoder.preprocess_native(rgb256, resolution=256, spec=spec, generator=torch.Generator().manual_seed(11))
    assert torch.equal(a, b)                                        # R = 256 时与 v7 的预处理逐位相同
    mean = 255*torch.tensor(spec.mean)[:, None, None]; std = 255*torch.tensor(spec.std)[:, None, None]
    py, npy = random.getstate(), np.random.get_state()
    for r in (192, 320, 384):
        rgb = torch.randint(0, 256, (2, r, r, 3), dtype=torch.uint8, generator=g)
        out = native_encoder.preprocess_native(rgb, resolution=r, spec=spec, generator=torch.Generator().manual_seed(1))
        # scale 1.777：回退到整张图，R→R 双线性插值是恒等 → 逐像素归一化，不缩放回 256
        manual = (rgb.permute(0, 3, 1, 2).float()-mean)/std
        assert out.shape == (2, 3, r, r) and out.dtype == torch.float32 and torch.equal(out, manual)
    assert random.getstate() == py and all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
                                           for x, y in zip(np.random.get_state(), npy))
    for bad in (dict(rgb=rgb256, resolution=192), dict(rgb=rgb256[..., :2], resolution=256), dict(rgb=rgb256.float(), resolution=256),
                dict(rgb=torch.zeros((1, 200, 200, 3), dtype=torch.uint8), resolution=200)):
        with pytest.raises(ValueError):
            native_encoder.preprocess_native(bad["rgb"], resolution=bad["resolution"], spec=spec, generator=torch.Generator())
    with pytest.raises(ValueError, match="explicit CPU generator"):
        native_encoder.preprocess_native(rgb256, resolution=256, spec=spec, generator=torch.random.default_generator)


class TinyPatch(torch.nn.Module):
    """按 2×16×16 tubelet 嵌入的小编码器（与真实 patch 嵌入同形状，与输入分辨率无关），只供接口测试。"""
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


def test_native_frame_encoder():
    loaded = _loaded()
    kw = dict(preprocess=cache.preprocess_spec(), encoder_device=torch.device("cpu"), autocast_dtype=None)
    g = torch.Generator().manual_seed(0)
    t = torch.zeros(2, dtype=torch.float64)
    x = torch.randn(2, 3, 256, 256, generator=g)
    v7 = vjepa.FrozenFrameEncoder(loaded, **kw).encode_target(x, target_time_seconds=t)
    same = native_encoder.NativeFrameEncoder(loaded, resolution=256, **kw).encode_target(x, target_time_seconds=t)
    assert torch.equal(v7.raw_z, same.raw_z) and same.grid.height == 16                 # R = 256 时逐位相同
    w = loaded.encoder.proj.weight
    for r in (192, 320, 384):
        e = native_encoder.NativeFrameEncoder(loaded, resolution=r, **kw)
        x = torch.randn(2, 3, r, r, generator=g)
        out = e.encode_target(x, target_time_seconds=t)
        side = r//16
        assert out.raw_z.shape == (2, side*side, 1408) and (out.grid.height, out.grid.width) == (side, side)
        assert torch.equal(out.grid.coords, cache.regular_grid(side, side).coords)
        k = side+2                                                                      # 第 1 行、第 2 列（行优先）
        expect = torch.einsum("bchw,dchw->bd", x[:, :, 16:32, 32:48], w[:, :, 0]+w[:, :, 1])   # 单帧复制两份
        assert torch.allclose(out.raw_z[:, k], expect, atol=1e-4)
        with pytest.raises(ValueError, match="native resolution"):
            e.encode_target(torch.randn(2, 3, 256, 256), target_time_seconds=t)
        with pytest.raises(ValueError):
            e.encode_target(x.double(), target_time_seconds=t)
    with pytest.raises(ValueError):
        native_encoder.NativeFrameEncoder(loaded, resolution=200, **kw)
    assert [native_encoder.grid_side(r) for r in (192, 256, 320, 384)] == [12, 16, 20, 24]


def test_encode_trajectory_real_path_uses_native_preprocessing(official):
    """encode_v7 的真实编码器路径（不是 --fake-encoder）：每帧前重置同一个 generator、按原生分辨率预处理、NativeFrameEncoder 编码。"""
    enc = _tool_module("encode_v7")
    from po_jepa import simulation
    kw = dict(preprocess=cache.preprocess_spec(), encoder_device=torch.device("cpu"), autocast_dtype=None)
    loaded = _loaded()
    frames = np.random.default_rng(0).integers(0, 256, (3, 192, 192, 3), dtype=np.uint8)
    e = native_encoder.NativeFrameEncoder(loaded, resolution=192, **kw)
    got = enc._encode_trajectory(e, frames, 37, simulation, cache, torch, np, 2, False, None, 192)
    seed = simulation.preprocess_seed(37)
    pre = torch.cat([native_encoder.preprocess_native(torch.from_numpy(frames[k:k+1]), resolution=192, spec=cache.preprocess_spec(),
                                                      generator=torch.Generator().manual_seed(seed)) for k in range(3)])
    expect = e.encode_target(pre, target_time_seconds=torch.zeros(3, dtype=torch.float64)).raw_z.double().numpy()
    assert got.shape == (3, 144, 1408) and np.array_equal(got, expect)
    with pytest.raises(ValueError):                 # 不按原生分辨率（resolution=None 走 v7 的 256 预处理）时编码器拒收
        enc._encode_trajectory(e, frames, 37, simulation, cache, torch, np, 2, False, None, None)


# —— 缓存读取 ——

def test_space_cache_reads_only_as_its_dataset(caches, real, active):
    active(BASE)
    with pytest.raises(ValueError, match="no space-sampling knob"):
        space_cache.read_complete(caches["r192"]/"0036")
    with pytest.raises((ValueError, KeyError)):      # 普通窗口（cache.py）不收：特征形状、预处理摘要都不同
        cache.read_window(caches["r192"]/"0036", start_tick=0, duration_cycles=1., expected_split="test", device="cpu")
    active(ZOOM)
    with pytest.raises(ValueError, match="mismatch"):     # 256、16×16 的 zoom 缓存，cache.py 也不收（预处理摘要不同）
        cache.read_window(caches["zoom"]/"0036", start_tick=0, duration_cycles=1., expected_split="test", device="cpu")
    with pytest.raises(ValueError, match="mismatch"):     # 主实验缓存当成 zoom 缓存读
        space_cache.read_complete(caches["base"]/"0036")
    raw, m, _ = space_cache.read_complete(caches["zoom"]/"0036")
    assert raw.shape == (33, 256, 1408) and m["trajectory_id"] == 36
    active(datasets.parse("double_well~zoom=0.75:2000"))      # 核过的结果不跨数据集复用（同一目录刚在 :40 下读成功过）
    with pytest.raises(ValueError, match="mismatch"):
        space_cache.read_complete(caches["zoom"]/"0036")
    active(R384)                                          # 分辨率不同
    with pytest.raises(ValueError, match="mismatch"):
        space_cache.read_complete(caches["r192"]/"0036")
    active(datasets.parse("double_well~resolution=192:2000"))   # 条数不同
    with pytest.raises(ValueError, match="mismatch"):
        space_cache.read_complete(caches["r192"]/"0036")
    active(datasets.parse("double_well~resolution=192,horizon=64:40"))
    with pytest.raises(ValueError, match="horizon=32"):
        space_cache.read_complete(caches["r192"]/"0036")
    active(datasets.parse("double_well~fps=8,resolution=192:40"))
    with pytest.raises(ValueError, match="without time sampling"):
        space_cache.read_complete(caches["r192"]/"0036")


def test_space_cache_checks(caches, real, active, tmp_path):
    active(R192)
    for name, edit in (("feature", lambda d: (d/"raw_z.npy").write_bytes((d/"raw_z.npy").read_bytes()[:-2]+b"\0\1")),
                       ("manifest", lambda d: (d/"manifest.json").write_text((d/"manifest.json").read_text()+" "))):
        d = tmp_path/name; shutil.copytree(caches["r192"]/"0037", d); edit(d)
        with pytest.raises(ValueError, match="SHA mismatch"):
            space_cache.read_complete(d)
    root = caches["r192"]/"0037"
    for name, change in (("controls", lambda m: m["controls"]["values"][0].__setitem__(0, .123)),
                         ("starts", lambda m: m["controls"]["starts_ticks"].__setitem__(1, m["controls"]["starts_ticks"][1]+1)),
                         ("tick", lambda m: m["frames"][7].__setitem__("tick", 8)),
                         ("seconds", lambda m: m["frames"][7].__setitem__("time_seconds", 2.)),
                         ("image", lambda m: m["frames"][7].__setitem__("image_sha256", "ab")),
                         ("frames", lambda m: m["frames"].pop()),
                         ("space", lambda m: m["space"].__setitem__("zoom", .75)),
                         ("no_space", lambda m: m.pop("space")),
                         ("shape", lambda m: m.__setitem__("shape", [33, 256, 1408])),
                         ("grid", lambda m: m.__setitem__("grid", [16, 16])),
                         ("preprocess_sha", lambda m: m.__setitem__("preprocess_sha256", cache.v7_preprocess_sha256())),
                         ("preprocess_record", lambda m: m["generation"].__setitem__("preprocess", cache._v7_preprocess_record())),
                         ("variant", lambda m: m.__setitem__("variant", "resolution=384")),
                         ("split", lambda m: m.__setitem__("split", "train")),
                         ("dtype", lambda m: m.__setitem__("dtype", "bfloat16")),
                         ("upstream", lambda m: m["generation"].__setitem__("upstream_commit", "0"*40))):
        m = _as_real(_manifest(caches["r192"], 37)); change(m)
        with pytest.raises((ValueError, KeyError)):
            space_cache.check_manifest(root, m, "sha", datasets.active())
    m = _as_real(_manifest(caches["r192"], 37))
    space_cache.check_manifest(root, m, "sha", datasets.active())       # 不改时能过（上面每一处都是它单独报错）
    # 原生分辨率的预处理摘要和记录一起换成另一个分辨率的（自洽的错值）也不收
    m["preprocess_sha256"] = space_cache.preprocess_sha256(384); m["generation"]["preprocess"] = space_cache.preprocess_record(384)
    with pytest.raises(ValueError, match="mismatch"):
        space_cache.check_manifest(root, m, "sha", datasets.active())


def test_space_cache_refuses_synthetic_features(caches, active):
    active(R192)
    m = _manifest(caches["r192"], 37)
    with pytest.raises(ValueError, match="provenance"):
        space_cache.check_manifest(caches["r192"]/"0037", m, "sha", datasets.active())
    m["generation"].update(encoder_checkpoint_sha256=cache.CHECKPOINT_SHA, encoder_load_report=dict(sha256=cache.CHECKPOINT_SHA,
                           encoder_key="target_encoder", strict=True, missing_keys=[], unexpected_keys=[]))
    with pytest.raises(ValueError, match="provenance"):           # 只差 evidence_role 也不收
        space_cache.check_manifest(caches["r192"]/"0037", m, "sha", datasets.active())


def _controls_equal(a, b):
    for field in ("starts", "ends", "u_values", "n_segments", "segment_valid", "source_origin_seconds", "duration"):
        if not torch.equal(getattr(a, field), getattr(b, field)):
            return False
    return a.source_ids == b.source_ids


@pytest.mark.parametrize("name,s,t", [("zoom", 0, 1), ("r192", 5, 3), ("r384", 10, 12), ("drift", 20, 8)])
def test_space_windows_equal_the_main_windows_except_features(caches, real, active, name, s, t):
    """同一 (轨迹, s, T)：动作全部字段、时长、时刻与主实验窗口逐位相同；网格 G×G；特征就是缓存的第 s、s+T 帧；画面不同。"""
    active(BASE)
    a = cache.read_window(caches["base"]/"0038", start_tick=s, duration_cycles=float(t), expected_split="test", device="cpu")
    spec = dict(zoom=ZOOM, r192=R192, r384=R384, drift=DRIFT)[name]
    active(spec)
    b = space_cache.read_window(caches[name]/"0038", start_tick=s, duration_cycles=float(t), expected_split="test", device="cpu")
    assert _controls_equal(a.inputs.controls, b.inputs.controls)
    assert torch.equal(a.inputs.duration, b.inputs.duration) and torch.equal(a.inputs.observed_time_seconds, b.inputs.observed_time_seconds)
    assert torch.equal(a.target.target_time_seconds, b.target.target_time_seconds)
    assert a.inputs.image_sha256 != b.inputs.image_sha256 and a.target.image_sha256 != b.target.image_sha256
    side = spec.knobs.resolution//16
    assert (b.inputs.grid.height, b.inputs.grid.width) == (side, side) and b.inputs.raw_z.shape == (1, side*side, 1408)
    raw = np.load(caches[name]/"0038"/"raw_z.npy")
    assert torch.equal(b.inputs.raw_z[0], torch.from_numpy(raw[s].astype(np.float32)))
    assert torch.equal(b.target.raw_z[0], torch.from_numpy(raw[s+t].astype(np.float32)))
    assert b.provenance["space"] == space_cache.space_record(spec)
    space_cache.window_boundary(b)


def test_read_window_and_boundary_checks(caches, real, active):
    active(R192)
    for s, t in ((-1, 1.), (21, 1.), (1.0, 1.), (0, 5.), (0, 0.5)):
        with pytest.raises(ValueError):
            space_cache.read_window(caches["r192"]/"0037", start_tick=s, duration_cycles=t, expected_split="test", device="cpu")
    with pytest.raises(ValueError, match="cross-split"):
        space_cache.read_window(caches["r192"]/"0037", start_tick=0, duration_cycles=1., expected_split="train", device="cpu")
    w = space_cache.read_window(caches["r192"]/"0037", start_tick=5, duration_cycles=2., expected_split="test", device="cpu")
    space_cache.window_boundary(w)
    with pytest.raises(ValueError, match="16x16"):                  # 主实验的检查写死 16×16，所以另写了 G×G 版
        data._window_boundary(w, w)
    active(R384)                                                    # 当前分辨率 384：12×12 的窗口不收
    with pytest.raises(ValueError, match="G×G grid"):
        space_cache.window_boundary(w)
    active(R192)
    bad = copy.copy(w.inputs); object.__setattr__(bad, "raw_z", w.inputs.raw_z[:, :100])
    with pytest.raises(ValueError, match="G²"):
        space_cache.window_boundary(cache.CacheWindow(bad, w.target, w.provenance))
    bad = copy.copy(w.target); object.__setattr__(bad, "target_time_seconds", w.target.target_time_seconds+.25)
    with pytest.raises(ValueError, match="endpoint"):
        space_cache.window_boundary(cache.CacheWindow(w.inputs, bad, w.provenance))


# —— 评测 ——

class ActionToy(torch.nn.Module):
    """Φ_T(z) = z + β·T²·q + γ·(∫u_x)·p（逐 token 相同）：读动作和时长，和网格大小无关。"""
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, raw, controls, duration, grid):
        t = float(duration[0])
        assert float(controls.duration[0]) == t and raw.shape[1] == grid.height*grid.width
        self.calls.append((t, grid.height))
        n = int(controls.n_segments[0])
        integral = float(((controls.ends[0, :n]-controls.starts[0, :n]).double()*controls.u_values[0, :n, 0].double().cpu()).sum())
        q = torch.zeros(raw.shape[-1], dtype=torch.float64); q[0], q[1] = 1., -1.
        p = torch.zeros(raw.shape[-1], dtype=torch.float64); p[2], p[3] = 1., -1.
        return (raw.double()+.002*t*t*q+.5*integral*p).float()


def test_windows_and_value_names(active):
    active(R192)
    assert space_eval.windows(37) == tuple((s, t) for s in (0, 5, 10, 15, 20) for t in (1, 2, 3, 4, 8, 12))
    active(datasets.parse("double_well~resolution=192:2000"))      # 大数据集每条 2 个起点（按编号轮换，同主实验）
    assert space_eval.windows(1800) == tuple((s, t) for s in (0, 10) for t in (1, 2, 3, 4, 8, 12))
    assert space_eval.value_names()[:3] == ("direct:T=1", "base_direct:T=1", "gap:T=1") and len(space_eval.value_names()) == 18


def test_evaluate_and_pair_against_manual(caches, real, active):
    model, config = ActionToy(), e06_config()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    active(BASE)
    base_rows = space_eval.evaluate(model, caches["base"], config, torch.device("cpu"), base=True)
    assert {c[1] for c in model.calls} == {16}
    model.calls.clear()
    active(R192)
    rows = space_eval.evaluate(model, caches["r192"], config, torch.device("cpu"))
    assert model.training and all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert {c[1] for c in model.calls} == {12} and len(model.calls) == 4*5*6
    expected = [(tid, s, t) for tid in TEST_IDS for s in (0, 5, 10, 15, 20) for t in (1, 2, 3, 4, 8, 12)]
    assert [(r["trajectory_id"], r["start_tick"], r["duration"]) for r in rows] == expected
    raw = np.load(caches["r192"]/"0039"/"raw_z.npy")
    w = space_eval.load_window(caches["r192"], 39, 15, 8, "cpu")
    direct = d4.direct_prediction(model, w)[0]
    row = next(r for r in rows if (r["trajectory_id"], r["start_tick"], r["duration"]) == (39, 15, 8))
    assert row["value"] == d4._jepa(direct, torch.from_numpy(raw[23:24].astype(np.float32)), config)
    paired, means = space_eval.paired(rows, base_rows)
    for p, r, b in zip(paired, rows, base_rows):
        t = r["duration"]
        assert p["values"] == {f"direct:T={t}": r["value"], f"base_direct:T={t}": b["value"], f"gap:T={t}": r["value"]-b["value"]}
    assert means["windows:T=12"] == 20 and means["gap:T=3"] == pytest.approx(
        sum(p["values"]["gap:T=3"] for p in paired if p["duration"] == 3)/20, rel=1e-12)
    with pytest.raises(ValueError, match="same windows"):
        space_eval.paired(rows, base_rows[:-1])
    with pytest.raises(ValueError, match="same windows"):          # 条数相同但有一个窗口对不上
        space_eval.paired(rows, base_rows[1:]+[dict(base_rows[0], start_tick=1)])
    with pytest.raises(ValueError, match="same windows"):          # 重复的窗口
        space_eval.paired(rows, base_rows[:-1]+base_rows[:1])
    # identity：direct = d(z_s, z_{s+T})（直接从缓存的第 s、s+T 帧算）
    ident = space_eval.evaluate(IdentityPredictor(), caches["r192"], config, torch.device("cpu"))
    r39 = next(r for r in ident if (r["trajectory_id"], r["start_tick"], r["duration"]) == (39, 15, 8))
    assert r39["value"] == d4._jepa(torch.from_numpy(raw[15:16].astype(np.float32)), torch.from_numpy(raw[23:24].astype(np.float32)), config)


def test_load_window_checks(caches, real, active):
    active(BASE)
    with pytest.raises(ValueError, match="no space-sampling knob"):
        space_eval.load_window(caches["r192"], 36, 0, 1, "cpu")
    active(R192)
    with pytest.raises(ValueError, match="test trajectory"):
        space_eval.load_window(caches["r192"], 3, 0, 1, "cpu")
    with pytest.raises(ValueError, match="base test set"):
        space_eval.load_base_window(caches["base"], 36, 0, 1, "cpu")
    approved = data.approved_cache_metadata

    def other(root, *, splits):
        a = approved(root, splits=splits); a["manifest_sha256_by_id"] = {**a["manifest_sha256_by_id"], 36: "f"*64}
        return a
    data.approved_cache_metadata = other
    try:
        with pytest.raises(ValueError, match="identity/source"):
            space_eval.load_window(caches["r192"], 36, 0, 1, "cpu")
        active(BASE)
        with pytest.raises(ValueError, match="identity/source"):
            space_eval.load_base_window(caches["base"], 36, 0, 1, "cpu")
    finally:
        data.approved_cache_metadata = approved
    active(R384)                                  # 384 的数据集读 192 的缓存：根目录 report 的 variant 不同
    with pytest.raises(ValueError):
        space_eval.load_window(caches["r192"], 36, 0, 1, "cpu")
    active(R192)                                  # 读到的窗口端点时刻不对：load_window 要做 G×G 的边界检查
    read = space_cache.read_window

    def shifted(*args, **kw):
        w = read(*args, **kw); y = copy.copy(w.target)
        object.__setattr__(y, "target_time_seconds", w.target.target_time_seconds+.25)
        return cache.CacheWindow(w.inputs, y, w.provenance)
    space_cache.read_window = shifted
    try:
        with pytest.raises(ValueError, match="endpoint"):
            space_eval.load_window(caches["r192"], 36, 0, 1, "cpu")
    finally:
        space_cache.read_window = read


@pytest.mark.parametrize("condition", ["po_jepa_full", "matched_transformer", "vjepa2_ac_small"])
def test_real_model_structures_run_on_native_grids(caches, real, active, condition):
    """真实结构（未训练）的三个模型直接吃 12×12、24×24 的特征：能跑、有限、形状对（只核接口，不看数值）。"""
    active(BASE)
    if condition == "vjepa2_ac_small":                 # build_condition 只在 v7 协议下建 AC；结构同 ac_training.build（未载入权重）
        from po_jepa import ac_baseline
        torch.manual_seed(0)
        model, loss = ac_baseline.ACPredictor(ac_baseline.SMALL), e06_config()
    else:
        model, loss = comparison.build_condition(condition, torch.device("cpu"))
    model.eval()
    for name, spec in (("r192", R192), ("r384", R384)):
        active(spec)
        w = space_eval.load_window(caches[name], 37, 10, 3, "cpu")
        with torch.no_grad():
            out = d4.direct_prediction(model, w)[0]
        side = spec.knobs.resolution//16
        assert out.shape == (1, side*side, 1408) and torch.isfinite(out).all()
        assert space_eval.window_value(model, w, loss) > 0


# —— 工具、transfer、任务表、汇总 ——

def _job(tmp_path, dataset=BASE.tag, kind=None):
    job = tmp_path/"job"/"run"/"job"; job.mkdir(parents=True, exist_ok=True)
    record = dict(seed=1703, updates=10000, dataset=dataset, condition="identity")
    if kind:
        record["kind"] = kind
    (job/"job.json").write_text(json.dumps(record))
    return tmp_path/"job"


def test_tool_writes_report(caches, real, active, tmp_path, monkeypatch):
    active(BASE)
    seen = {}

    def fake_load(condition, training_run, device):
        seen.update(condition=condition, training_run=training_run, dataset=datasets.active().tag)
        return IdentityPredictor(), e06_config(), None, dict(fake=True)
    monkeypatch.setattr(v7jobs, "load_for_evaluation", fake_load)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    job = _job(tmp_path)
    out = tmp_path/"out"/"identity.json"
    args = ["--job", str(job), "--test-variant", "pan=1,zoom=0.75", "--features", str(caches["drift"]),
            "--base-features", str(caches["base"]), "--device", "cpu"]
    with pytest.raises(ValueError):                    # 特征是 drift 的，--test-variant 少了 drift：数据集对不上
        _tool_module("space_eval").main(args+["--output", str(out)])
    assert not out.exists() and datasets.active() == BASE
    args[3] = "drift=0.5,pan=1,zoom=0.75"
    assert _tool_module("space_eval").main(args+["--output", str(out)]) == 0
    assert seen == dict(condition="identity", training_run=None, dataset=BASE.tag)
    r = json.loads(out.read_text())
    assert (r["kind"], r["dataset"], r["train_dataset"], r["test_variant"], r["seed"]) == \
        ("space_eval", DRIFT.tag, BASE.tag, "drift=0.5,pan=1,zoom=0.75", 1703)
    assert r["counts"] == dict(windows=4*5*6, trajectories=4) == dict(windows=len(r["rows"]), trajectories=4)
    assert r["definitions"]["grid"] == [16, 16] and r["definitions"]["drift"] == .5 and r["tubelet"] is None
    assert set(r["cache_identity"]) == {"encode_report_sha256", "base_encode_report_sha256"}
    # identity 在基础测试集上的值 = d(z_s, z_{s+T})（主实验缓存），与空间采样测试集上的不同
    row = r["rows"][7]
    active(BASE)
    w = space_eval.load_base_window(caches["base"], row["trajectory_id"], row["start_tick"], row["duration"], "cpu")
    t = row["duration"]
    assert row["values"][f"base_direct:T={t}"] == d4._jepa(w.inputs.raw_z, w.target.raw_z, e06_config())
    assert row["values"][f"gap:T={t}"] == row["values"][f"direct:T={t}"]-row["values"][f"base_direct:T={t}"] != 0
    assert datasets.active() == BASE
    with pytest.raises(SystemExit, match="已存在"):
        _tool_module("space_eval").main(args+["--output", str(out)])
    for bad, match in (("gain=2", "空间采样调节项"), ("zoom=0.75,gain=2", "空间采样调节项"), ("fps=8", "空间采样调节项"),
                       ("resolution=256", "默认空间采样"), ("", "空间采样调节项")):
        with pytest.raises(SystemExit, match=match):
            _tool_module("space_eval").main(["--job", str(job), "--test-variant", bad, "--features", str(caches["zoom"]),
                                             "--base-features", str(caches["base"]), "--device", "cpu", "--output", str(tmp_path/"y.json")])


def test_tool_refuses_sampled_training_dataset(tmp_path, active, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for dataset in ("double_well~horizon=64-40", "double_well~fps=8-40", "double_well~zoom=0.8-40"):
        active(datasets.parse(dataset.rpartition("-")[0]+":40"))
        job = _job(tmp_path/dataset, dataset)
        with pytest.raises(SystemExit, match="基础场景"):
            _tool_module("space_eval").main(["--job", str(job), "--test-variant", "zoom=0.75", "--features", str(tmp_path),
                                             "--base-features", str(tmp_path), "--output", str(tmp_path/"o.json"), "--device", "cpu"])
    active(BASE)
    job = _job(tmp_path/"transfer", kind="transfer")
    with pytest.raises(SystemExit, match="transfer"):
        _tool_module("space_eval").main(["--job", str(job), "--test-variant", "zoom=0.75", "--features", str(tmp_path),
                                         "--base-features", str(tmp_path), "--output", str(tmp_path/"o.json"), "--device", "cpu"])


@pytest.mark.parametrize("variant", ["resolution=384", "zoom=0.75", "pan=0.5,zoom=0.8"])
def test_transfer_refuses_space_sampling(tmp_path, active, variant):
    active(BASE)
    with pytest.raises(ValueError, match="space_eval"):
        v7jobs.transfer_job(tmp_path, tmp_path/"out", torch.device("cpu"), "identity", training_run=None, test_variant=variant)
    assert not (tmp_path/"out").exists()


def _jobs(*args):
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/make_v8_jobs.py"), *args], env=_clean_env(), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_make_v8_jobs_space_stage():
    lines = _jobs("--stage", "space")
    commands = [l for l in lines if l and not l.startswith("#")]
    encode = [c for c in commands if "encode_v7.py encode" in c]
    final = [c for c in commands if "encode_v7.py finalize" in c]
    evals = [c for c in commands if "tools/space_eval.py" in c]
    assert len(encode) == 7*6*4 and len(final) == 7*6 and len(evals) == 7*6*3*6 and commands[-1].startswith("python tools/summarize_v7.py")
    assert all("--splits test" in c and "--trajectories 2000" in c for c in encode+final)
    assert "--variant drift=0.5,pan=1,shape=multi_well,zoom=0.75" in next(c for c in encode if "landscape" in c and "drift" in c)
    c = next(c for c in evals if "landscape" in c and "resolution=384" in c and "seed1704/vjepa2_ac " in c)
    assert ("--job ../results-v8/full/landscape~shape=multi_well/seed1704/vjepa2_ac --test-variant resolution=384 --features "
            "../features-v8/landscape~resolution=384,shape=multi_well-2000 --base-features ../features-v8/landscape~shape=multi_well-2000 "
            "--output ../results-v8/full/space/landscape~resolution=384,shape=multi_well/seed1704/vjepa2_ac.json --device cuda:0 "
            "--ac-checkpoint") in c
    full = _jobs("--stage", "full")                         # 参照特征目录就是 full 阶段编码的那个
    assert any("--output ../features-v8/landscape~shape=multi_well-2000 " in l for l in full if "encode_v7.py encode" in l)
    second = _jobs("--stage", "space", "--scenes", "wave", "--conditions", "identity", "--shards", "1")
    assert second[0].startswith("# 注意：二阶场景 wave") and len([l for l in second if l.startswith("python tools/space_eval.py")]) == 18
    for stage in ("pilot", "full", "variants", "long", "timing"):
        assert not any("space_eval" in l or "resolution=" in l or "zoom=" in l for l in _jobs("--stage", stage))


_SPACE_REPORT_FIELDS = dict(train_dataset="double_well-2000", registration_sha256="reg",
                            cache_identity=dict(encode_report_sha256="enc", base_encode_report_sha256="base"))


def test_summarize_space_reports(tmp_path):
    s = _tool_module("summarize_v7")

    def report(condition, seed, offset, base=.05):
        rows = []
        for t in (1800, 1801):
            for st in (0, 10):
                for d in space_eval.DURATIONS:
                    v = offset+.01*t
                    rows.append(dict(trajectory_id=t, start_tick=st, duration=d, manifest_sha256=f"s{t}", base_manifest_sha256=f"b{t}",
                                     values={f"direct:T={d}": v, f"base_direct:T={d}": base, f"gap:T={d}": v-base}))
        means = {f"{k}:T={d}": (offset+.01*1800.5 if k == "direct" else base if k == "base_direct" else offset+.01*1800.5-base)
                 for k in space_eval.KEYS for d in space_eval.DURATIONS}
        return dict(kind="space_eval", dataset="double_well~resolution=384-2000", updates=10000, condition=condition, seed=seed,
                    **_SPACE_REPORT_FIELDS, definitions=dict(durations=list(space_eval.DURATIONS)), rows=rows, means=means)
    for c, off, base in (("po_jepa_full", .1, .05), ("matched_transformer", .3, .02)):
        for seed in (1703, 1704):
            path = tmp_path/c/f"{seed}.json"; path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(report(c, seed, off, base)))
    (tmp_path/"zz_dup.json").write_text(json.dumps(report("po_jepa_full", 1703, 9.)))
    half = report("identity", 1703, .5); half["rows"] = half["rows"][1:]; (tmp_path/"i1.json").write_text(json.dumps(half))
    small = report("vjepa2_ac_small", 1703, .7)
    for w in small["rows"]:
        w["manifest_sha256"] = "a"*64                           # 同编号窗口、不同缓存来源
    (tmp_path/"s1.json").write_text(json.dumps(small))
    half = report("identity", 1704, .5); del half["means"]["gap:T=12"]; (tmp_path/"i2.json").write_text(json.dumps(half))
    out = s.space_reports(tmp_path)
    full = next(r for r in out["rows"] if r["condition"] == "po_jepa_full")
    assert full["seeds"] == 2 and full["durations"] == [1, 2, 3, 4, 8, 12]
    assert full["direct"][0] == pytest.approx(.1+18.005) and full["base_direct"][3] == pytest.approx(.05)
    assert full["gap"][5] == pytest.approx(.1+18.005-.05)
    ident = next(r for r in out["rows"] if r["condition"] == "identity")     # 一个种子缺这个键：不拿另一个种子顶替
    assert ident["gap"][5] is None and ident["gap"][4] == pytest.approx(.5+18.005-.05)
    d = next(c for c in out["comparisons"] if c["comparison"] == "po_jepa_full − matched_transformer" and c["metric"] == "direct" and c["T"] == 8)
    g = next(c for c in out["comparisons"] if c["comparison"] == "po_jepa_full − matched_transformer" and c["metric"] == "gap" and c["T"] == 8)
    assert d["mean"] == pytest.approx(-.2) and all(x["full_better"] == 4 == x["windows"] for x in d["per_seed"])
    assert g["mean"] == pytest.approx(-.2-.03) and g["seeds"] == 2
    assert len(out["duplicates"]) == 1
    assert full["gap_sd"][0] == pytest.approx(0., abs=1e-12) and full["direct_sd"][0] is not None
    # 窗口集合不同（identity 1703 少一行）、缓存来源不同（vjepa2_ac_small）：不配对，记进 incomplete，不取交集。
    # S25-fix3：和长时程、时间采样一样，任一种子不通过整个比较不出平均差（原来 identity 还用种子 1704 单独出）
    assert sorted((m["comparison"], [x["seed"] for x in m["per_seed"]]) for m in out["incomplete"]) == [
        ("po_jepa_full − identity", [1703]), ("po_jepa_full − vjepa2_ac_small", [1703])]
    assert not any(c["comparison"] in ("po_jepa_full − identity", "po_jepa_full − vjepa2_ac_small") for c in out["comparisons"])
    md = s._markdown(dict(jobs=0, problems=[], across_seeds=[], transfer=[], planning=[], long=dict(rows=[], comparisons=[]),
                          timing=dict(rows=[], comparisons=[]), space=out, missing=[], duplicates=[]))
    assert "## 空间采样" in md and "| double_well~resolution=384-2000 | 10000 | po_jepa_full | 2 |" in md and "| gap | 8 |" in md
    assert "⚠ 空间采样" in md


def _space_report(condition, seed, direct):
    """两条轨迹 × 一个起点 × 时长 1、2；direct 按行给，base_direct 固定 0。"""
    keys = [(t, 0, d) for t in (1800, 1801) for d in (1, 2)]
    rows = [dict(trajectory_id=t, start_tick=st, duration=d, manifest_sha256=f"s{t}", base_manifest_sha256=f"b{t}",
                 values={f"direct:T={d}": x, f"base_direct:T={d}": 0., f"gap:T={d}": x}) for (t, st, d), x in zip(keys, direct)]
    return dict(kind="space_eval", dataset="double_well~resolution=384-2000", updates=10000, condition=condition, seed=seed,
                **_SPACE_REPORT_FIELDS, definitions=dict(durations=[1, 2]), rows=rows,
                means={f"{k}:T={d}": 1. for k in space_eval.KEYS for d in (1, 2)})


@pytest.mark.parametrize("what", ["missing_row", "duplicate_row", "manifest", "base_manifest", "base_cache", "train_dataset",
                                  "registration", "definitions_differ"])
def test_summarize_space_refuses_unpairable_reports(tmp_path, what):
    """S25-fix3：空间采样配对并进 pairing_problems。原来只核窗口集合和逐窗口 manifest，报告级的训练数据集、预登记、
    评测定义、缓存身份（含基础测试集的编码报告）不核；重复的行被 dict 吞掉只在长度上间接看出。
    反例同时间采样：full [1, 100]、对照 [2, 2]，对照缺最后一行时取交集会让平均差从 48.5 变成 −1。"""
    s = _tool_module("summarize_v7")
    full = _space_report("po_jepa_full", 1703, [1., 1., 1., 100.])
    ref = _space_report("matched_transformer", 1703, [2., 2., 2., 2.])
    (tmp_path/"f.json").write_text(json.dumps(full)); (tmp_path/"r.json").write_text(json.dumps(ref))
    out = s.space_reports(tmp_path)
    assert not out["incomplete"]
    assert next(c for c in out["comparisons"] if c["T"] == 2 and c["metric"] == "direct")["mean"] == pytest.approx(48.5)
    if what == "missing_row":
        ref["rows"] = ref["rows"][:-1]
    elif what == "duplicate_row":
        ref["rows"] = ref["rows"]+[dict(ref["rows"][0], values={"direct:T=1": 50., "base_direct:T=1": 0., "gap:T=1": 50.})]
    elif what == "manifest":
        ref["rows"][3]["manifest_sha256"] = "other"
    elif what == "base_manifest":
        ref["rows"][3]["base_manifest_sha256"] = "other"
    elif what == "base_cache":
        ref["cache_identity"] = dict(ref["cache_identity"], base_encode_report_sha256="other")
    elif what == "train_dataset":
        ref["train_dataset"] = "double_well~gain=2-2000"
    elif what == "registration":
        del ref["registration_sha256"]
    elif what == "definitions_differ":
        ref["definitions"] = dict(durations=[1, 2, 3])
    (tmp_path/"r.json").write_text(json.dumps(ref))
    out = s.space_reports(tmp_path)
    assert not out["comparisons"]
    (inc,) = out["incomplete"]
    assert inc["comparison"] == "po_jepa_full − matched_transformer" and inc["per_seed"][0]["seed"] == 1703
    md = s._markdown(dict(jobs=0, problems=[], across_seeds=[], transfer=[], planning=[], long=dict(rows=[], comparisons=[]),
                          timing=dict(rows=[], comparisons=[]), space=out, missing=[], duplicates=[]))
    assert "⚠ 空间采样" in md


@pytest.mark.parametrize("all_missing", [False, True])
def test_summarize_space_first_report_without_definitions(tmp_path, all_missing):
    """S25-fix3：按文件顺序第一份空间采样报告缺 definitions 时原来整个汇总 KeyError（S25-fix2 只修了长时程、时间采样）。"""
    s = _tool_module("summarize_v7")
    full = _space_report("po_jepa_full", 1703, [1., 1., 1., 100.])
    ref = _space_report("matched_transformer", 1703, [2., 2., 2., 2.])
    del full["definitions"]
    if all_missing:
        del ref["definitions"]
    (tmp_path/"a.json").write_text(json.dumps(full)); (tmp_path/"b.json").write_text(json.dumps(ref))
    out = s.space_reports(tmp_path)
    assert not out["comparisons"]
    (inc,) = out["incomplete"]
    assert inc["comparison"] == ("全部比较" if all_missing else "po_jepa_full − matched_transformer")
    assert bool(out["rows"]) is not all_missing
