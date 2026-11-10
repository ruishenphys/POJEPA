"""S24c 空间采样：调节项 resolution、zoom、pan、drift（docs/segments/S24c.md）。

核的是：解析、校验与覆盖表；默认值不经过相机；单位视野（256、zoom 1、无偏移）下各画图函数与原来逐位相同；背景纹理、遮挡块、
画布边是精确的面积平均（与独立的逐画布像素重叠计算对照）；场的双线性插值；相机偏移（半径、方向、漂移角速度、随机流 12、
视野始终包含画布）；光点落在相机算出的输出位置、大小按缩放变化；隐藏状态不受空间采样影响；与 fps、horizon 组合；
输出分辨率下的像素噪声；曝光窗口里每个取样时刻用自己的相机；从一帧读出相机偏移、推出下一帧；编码、探针、规划拒绝。
写死的数值（视野边长、余量、偏移半径、角速度）不调被测函数推出来（错误清单 2026-11-09 S23a 那条）。
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from po_jepa import camera, datasets, simulation, systems, timing, variants
from po_jepa.camera import View
from po_jepa.variants import Variant

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"src"
ALL = ("double_well", "van_der_pol", "multi_particle", "advection_diffusion", "landscape", "gray_scott", "push",
       "pendulum", "bouncing_balls", "wave", "cart_pole", "spring_chain")
FIELD_GRID = ("gray_scott", "wave")        # 128 格的场：相机里双线性插值，单位视野下不等于原来的 2×2 块复制
IDENTITY = View(256, 1., 0., 0.)


@pytest.fixture
def use():
    old = datasets.active()
    def set_(system, knobs="", count=200):
        text = ("shape=ring," if system == "landscape" and "shape=" not in knobs else "") + knobs
        text = text.rstrip(",")
        count = 40 if system == "gray_scott" and count == 200 else count
        spec = datasets.DatasetSpec(system, count, variants.format(system, variants.parse(system, text)))
        datasets._set_for_tests(spec)
        return spec
    yield set_
    datasets._set_for_tests(old)


def join(*parts):
    return ",".join(p for p in parts if p)


def u4_of(tid):
    v = datasets.active().knobs
    return systems.applied_subticks(tid, v, systems.commanded_controls(tid))


def raw_at(tid, times, view=None):
    """当前数据集第 tid 条在 times 时刻的原始画面（像素噪声前），view=None 时不进相机。"""
    v = datasets.active().knobs
    sc = timing._scene(systems.get(), tid, v, v.horizon)
    states = timing.simulate(sc, u4_of(tid), times)
    with systems.raw_images():
        if view is None:
            return sc.draw(states, np.asarray(times, dtype=np.float64))
        with systems.camera_view(view):
            return sc.draw(states, np.asarray(times, dtype=np.float64))


def clipped_overlap(x, s, n=256):
    """独立实现：足迹 [x − s/2, x + s/2] 与画布像素 [k, k+1)（0 ≤ k < n，不回绕）的重叠长度 / s → [len(x), n]。"""
    A = np.zeros((len(x), n))
    for i, xi in enumerate(x):
        a, b = xi-s/2, xi+s/2
        for k in range(max(0, math.floor(a)), min(n, math.ceil(b))):
            A[i, k] += max(0., min(b, k+1)-max(a, k))/s
    return A


def overlap_matrix(x, s, n=256):
    """独立实现：足迹 [x − s/2, x + s/2] 与画布像素 [k, k+1)（k 按 n 周期回绕）的重叠长度 / s → [len(x), n]。"""
    A = np.zeros((len(x), n))
    for i, xi in enumerate(x):
        a, b = xi-s/2, xi+s/2
        for k in range(math.floor(a), math.ceil(b)):
            A[i, k % n] += max(0., min(b, k+1)-max(a, k))/s
    return A


# —— 解析、格式、覆盖表 ——

def test_parse_format_and_validate():
    v = variants.parse("double_well", "resolution=384,zoom=0.75,pan=0.5,drift=0.25")
    assert (v.resolution, v.zoom, v.pan, v.drift) == (384, .75, .5, .25) and type(v.resolution) is int
    assert variants.format("double_well", v) == "drift=0.25,pan=0.5,resolution=384,zoom=0.75"
    assert v.changes_space() and v.resampled() and not v.changes_timing()
    assert not Variant().changes_space() and not Variant().resampled() and variants.parse("push", "fps=8").resampled()
    assert variants.format("double_well", variants.parse("double_well", "resolution=256,zoom=1,pan=0,drift=0")) == ""
    assert variants.RESOLUTIONS == (192, 256, 320, 384) and variants.ZOOM_MIN == .5 and variants.DRIFT_MAX == .5
    for bad in ("resolution=200", "resolution=512", "resolution=256.0", "zoom=1.2", "zoom=0.4", "zoom=nan",
                "zoom=0.8,pan=1.5", "zoom=0.8,pan=-0.1", "pan=0.5", "zoom=1,pan=0.5", "zoom=0.8,drift=0.2",
                "zoom=0.8,pan=0.5,drift=0.6", "zoom=0.8,pan=0.5,drift=-0.1", "zoom=0.8,pan=0.5,drift=inf"):
        with pytest.raises(ValueError):
            variants.parse("double_well", bad)
    with pytest.raises(ValueError):
        variants.validate("double_well", Variant(resolution=384.))
    for system in ALL:
        text = ("shape=ring," if system == "landscape" else "") + "resolution=192,zoom=0.5,pan=1,drift=0.5"
        assert variants.parse(system, text).resolution == 192


def test_coverage_lists_space_knobs_for_every_scene():
    cov = variants.coverage()
    assert variants.KNOB_GROUPS["空间"] == ("resolution", "zoom", "pan", "drift")
    for name in variants.KNOB_GROUPS["空间"]:
        assert cov[name] == tuple(datasets.SYSTEMS)
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/knob_coverage.py"), "--check"], env=_clean_env(),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout+r.stderr


# —— 默认值不经过相机；单位视野下逐位相同 ——

def test_default_space_knobs_do_not_touch_the_camera(use, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("camera.views 不该被调用")
    monkeypatch.setattr(camera, "views", boom)
    for system, knobs in (("double_well", "pixel_noise=2"), ("push", "fps=8"), ("pendulum", "exposure=0.5"),
                          ("advection_diffusion", "")):
        use(system, knobs)
        _, f = systems.get().frames(1)
        assert f.shape[1:] == (256, 256, 3)
    x = np.arange(256)+.5
    r, c = systems.pixel_axes()
    assert np.array_equal(r, x) and np.array_equal(c, x) and systems.edge_width() == 1.
    row, col = systems.pixel_grid()
    ref = np.meshgrid(x, x, indexing="ij")
    assert np.array_equal(row, ref[0]) and np.array_equal(col, ref[1])


# 在 d7762a7（S24c 改写画图函数之前）上算的整条轨迹（第 3 条）画面 SHA-256 前 16 位：默认画法和非空间调节项的画法逐位不变
GOLDEN = [
    ("double_well", "", "772fe8f272f0c2f6"),
    ("double_well", "background=noise,distractors=2,movers=1,occluders=1", "7d31acf2a52729be"),
    ("van_der_pol", "background=checker,sigma=4", "5e494d6b948dba67"),
    ("multi_particle", "movers=2,n=5", "214f55ca01549c15"),
    ("advection_diffusion", "", "1310cf07d446b1b2"),
    ("advection_diffusion", "background=noise,occluders=2", "ddb5adf2e3e5e6ea"),
    ("landscape", "shape=obstacles,background=checker,distractors=1", "331673a07b4f8533"),
    ("gray_scott", "occluders=1", "91ab73acaf61491c"),
    ("push", "", "1f57d34096d96f2c"),
    ("push", "distractors=2,movers=2,occluders=1", "7482d1287f1eeebb"),
    ("pendulum", "background=checker,movers=1", "e7696498884fc2e1"),
    ("bouncing_balls", "n=5", "5895ed73cb59e783"),
    ("wave", "occluders=1", "f5cba40c3cf2cd75"),
    ("cart_pole", "background=noise", "1dbd94c4017fb231"),
    ("spring_chain", "n=6,movers=1", "d69948fe4871cdf0"),
    ("pendulum", "fps=8,exposure=0.5", "7bbd7b7f8e4fbccb"),
]


@pytest.mark.parametrize("system,knobs,sha", GOLDEN)
def test_default_drawing_matches_the_pre_s24c_hashes(use, system, knobs, sha):
    import hashlib
    use(system, knobs)
    _, f = systems.get().frames(3)
    assert hashlib.sha256(f.tobytes()).hexdigest()[:16] == sha


def test_camera_view_context_restores():
    with systems.camera_view(View(192, 2., -10., 3.)):
        assert systems.pixel_axes()[0].shape == (192,) and systems.edge_width() == 2.
        with systems.camera_view(View(384, .5, 0., 0.)):
            assert systems.edge_width() == 1.
        assert systems.edge_width() == 2.
    assert systems._VIEW[0] is None
    with pytest.raises(RuntimeError):
        with systems.camera_view(View(192, 2., 0., 0.)):
            raise RuntimeError
    assert systems._VIEW[0] is None


CASES = [("double_well", "background=noise,distractors=2,movers=1,occluders=1"),
         ("van_der_pol", "background=checker,sigma=4"),
         ("multi_particle", "n=5,background=plain,movers=2"),
         ("advection_diffusion", "background=noise,occluders=2"),
         ("landscape", "shape=obstacles,background=checker,distractors=1"),
         ("push", "distractors=2,movers=2,occluders=1"),
         ("pendulum", "background=checker,distractors=1,movers=1"),
         ("bouncing_balls", "n=5,occluders=1"),
         ("cart_pole", "background=noise"),
         ("spring_chain", "n=6,movers=1"),
         ("gray_scott", ""), ("wave", "occluders=1")]


@pytest.mark.parametrize("system,knobs", CASES)
def test_identity_view_draws_bit_identical(use, system, knobs):
    """单位视野（256、s = 1、左上角 0）里各画图函数与不进相机时逐位相同：网格坐标、纹理、遮挡块、抗锯齿宽度的改写
    没有改默认画法。gray_scott、wave 的 128 格场在相机里是双线性插值，只核场以外的部分（单独的测试核插值）。"""
    use(system, knobs)
    times = [0., 3., 17., 32.]
    a, b = raw_at(4, times), raw_at(4, times, IDENTITY)
    assert a.shape == b.shape == (4, 256, 256, 3)
    if system in FIELD_GRID:
        assert not np.array_equal(a, b)
    else:
        assert np.array_equal(a, b)


def test_field_grid_scenes_sample_cell_centres_exactly(use):
    """gray_scott、wave：视野像素中心正好落在 128 格的格点中心上时（s = 2，左上角 0），相机里的画面就是原画面每 2×2 块的
    左上像素（原来是最近邻复制，块内四个像素相同）——除了圆环标记。"""
    for system in FIELD_GRID:
        use(system, "background=plain")
        a = raw_at(2, [0., 9.])
        b = raw_at(2, [0., 9.], View(128, 2., 0., 0.))
        # 圆环在原图上的像素位置（原图 − 不带圆环的场色）不好独立算，这里只比远离注入点的像素
        diff = np.abs(a[:, ::2, ::2]-b).max(-1)
        assert np.mean(diff < 1e-9) > .97


SHIFT_CASES = [c for c in CASES if c[0] not in FIELD_GRID]+[("wave", "background=plain"), ("gray_scott", "background=plain")]


@pytest.mark.parametrize("system,knobs", SHIFT_CASES)
def test_shifted_unit_view_is_the_shifted_canvas(use, system, knobs):
    """s = 1、左上角平移整数个画布像素（行 −16、列 +40）的视野 = 原画面平移后的那一块（重叠部分逐位相同）：
    核行、列坐标没有接反、世界 y 的方向、landscape 底图按画布归一化、干扰光点和场跟着相机走。
    gray_scott、wave 的场在相机里是插值（与原来的 2×2 块不同），只核场以外由圆环决定的部分：圆环位置（行、列分开）。"""
    use(system, knobs)
    times = [0., 17.]
    for tid in range(4, 12):
        a = raw_at(tid, times)
        b = raw_at(tid, times, View(256, 1., -16., 40.))
        ref, got = a[:, :240, 40:], b[:, 16:, :216]             # 画布行 0…239、列 40…255
        if system not in FIELD_GRID or all((f[..., 0]+f[..., 2]-2*f[..., 1]).max() > 150 for f in ref):
            break                                               # 场景：圆环在重叠区里的那条
    if system in FIELD_GRID:
        # 品红圆环：R + B − 2G 最大的像素（场的两种颜色这个值都在 ±30 以内，圆环约 290）
        peak = lambda x: [np.unravel_index(np.argmax(f[..., 0]+f[..., 2]-2*f[..., 1]), f.shape[:2]) for f in x]
        score = ref[..., 0]+ref[..., 2]-2*ref[..., 1]
        assert score.max() > 150 and peak(ref) == peak(got)
    else:
        assert np.array_equal(ref, got)


def test_edge_helpers_ramp_over_output_pixels():
    """抗锯齿过渡：s > 1 时过渡宽度 e = s（0.5 在边上、±e/2 处到 1、0）；e = 1 时与原式相同。"""
    from po_jepa import second_order as so
    view = View(64, 2.5, 0., 0.)
    with systems.camera_view(view):
        rows, cols = systems.pixel_grid()
        img = so._disc(np.zeros((64, 64, 3)), (80., 80.), np.ones(3), 20.)[..., 0]
        d = np.hypot(rows-80., cols-80.)
        np.testing.assert_allclose(img, np.clip((20.-d)/2.5+.5, 0., 1.), atol=1e-12)
        seg = so._segment(np.zeros((64, 64, 3)), (40., 10.), (40., 150.), np.ones(3), 1.)[..., 0]
        dist = np.where(cols < 10., np.hypot(rows-40., cols-10.), np.where(cols > 150., np.hypot(rows-40., cols-150.), np.abs(rows-40.)))
        np.testing.assert_allclose(seg, np.clip((1.-dist)/2.5+.5, 0., 1.), atol=1e-9)
        box = so._box(np.zeros((64, 64, 3)), (100.6, 60.6), (10., 30.), np.ones(3))[..., 0]
        inside = (np.abs(rows-100.6) <= 10.-1.25) & (np.abs(cols-60.6) <= 30.-1.25)
        outside = (np.abs(rows-100.6) >= 10.+1.25) | (np.abs(cols-60.6) >= 30.+1.25)
        assert np.all(box[inside] == 1.) and np.all(box[outside] == 0.) and ((box > 0) & (box < 1)).any()


@pytest.mark.parametrize("system,knobs", CASES)
def test_observation_knobs_render_inside_the_camera(use, system, knobs):
    """各场景带观察调节项（背景、干扰光点、会动的干扰光点、遮挡块）在非单位视野里能画，输出 R×R、有限、在 [0,255]。"""
    use(system, join(knobs, "resolution=192,zoom=0.7,pan=0.6"))
    _, f = systems.get().frames(3)
    assert f.shape == (33, 192, 192, 3) and f.dtype == np.uint8 and f.std() > 1


# —— 面积平均：纹理、遮挡块、画布边 ——

VIEWS = [View(192, 256/(192*.5), -128., -128.), View(320, 256/(320*.7), -40.3, -11.7),
         View(384, 256/384, 0., 0.), View(256, 1., -.25, .4), View(192, 256/192, 0., 0.)]


@pytest.mark.parametrize("kind", ["grid", "plain", "checker", "noise"])
@pytest.mark.parametrize("view", VIEWS)
def test_texture_is_the_exact_area_average(kind, view):
    """camera.texture = 画布纹理（256×256 分段常数）在每个输出像素足迹与画布的交集上的面积平均（独立的重叠矩阵；
    Codex 复审 S24c 起不再周期延拓）。完全在画布里的像素与周期延拓的平均相同。"""
    P = systems._texture(kind)
    Ar, Ac = clipped_overlap(view.rows(), view.scale), clipped_overlap(view.cols(), view.scale)
    keep_r, keep_c = Ar.sum(1) > 0, Ac.sum(1) > 0                # 这一轴上与画布有交集的像素（其余的值会被 canvas_mask 盖掉）
    Ar[keep_r] /= Ar[keep_r].sum(1, keepdims=True); Ac[keep_c] /= Ac[keep_c].sum(1, keepdims=True)
    want = np.stack([Ar@P[..., ch]@Ac.T for ch in range(3)], -1)
    got = camera.texture(kind, view)
    assert got.shape == (view.size, view.size, 3) and np.isfinite(got).all()
    keep = keep_r[:, None] & keep_c[None, :]
    np.testing.assert_allclose(got[keep], want[keep], rtol=0, atol=1e-9)
    inside = camera.square_cover(view, 0., 0., 256.) == 1.
    Pr, Pc = overlap_matrix(view.rows(), view.scale), overlap_matrix(view.cols(), view.scale)
    periodic = np.stack([Pr@P[..., ch]@Pc.T for ch in range(3)], -1)
    np.testing.assert_allclose(got[inside], periodic[inside], rtol=0, atol=1e-9)
    with systems.camera_view(view):
        assert np.array_equal(systems._texture(kind), got)


EDGE_VIEWS = VIEWS+[View(192, 2., -63., -64.), View(320, 256/(320*.6), -71.1, -84.9)]


@pytest.mark.parametrize("kind", ["grid", "plain", "checker", "noise"])
@pytest.mark.parametrize("view", EDGE_VIEWS)
def test_texture_and_canvas_edge_together_are_the_area_average(kind, view):
    """Codex 复审 S24c：纹理与画布边合起来（canvas_mask(texture)）= 整个足迹的面积平均，画布里是纹理、画布外是
    SURROUND（独立的不回绕重叠矩阵）。原来纹理先周期延拓平均、再按覆盖比例混色，跨画布边的像素偏色。"""
    P = systems._texture(kind)
    Ar, Ac = clipped_overlap(view.rows(), view.scale), clipped_overlap(view.cols(), view.scale)
    cover = Ar.sum(1)[:, None]*Ac.sum(1)[None, :]
    want = np.stack([Ar@P[..., ch]@Ac.T for ch in range(3)], -1)+camera.SURROUND*(1.-cover)[..., None]
    got = camera.canvas_mask(camera.texture(kind, view), view)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)
    assert ((cover > 0) & (cover < 1)).sum() > view.size          # 确实有跨画布边的像素


def test_canvas_edge_counterexample_from_the_review():
    """Codex 报告的确定反例：View(192, 2, −63, −64)，中心在 (行 0, 列 11) 的输出像素足迹 [−1,1]×[10,12]，grid 背景：
    画布内那半是网格色 (36,40,44)，画布外是 SURROUND (8,10,12)，正确值 (22,25,28)（原实现 (19,22,25)）。"""
    view = View(192, 2., -63., -64.)
    assert view.rows()[31] == 0. and view.cols()[37] == 11.
    got = camera.canvas_mask(camera.texture("grid", view), view)[31, 37]
    np.testing.assert_allclose(got, [22., 25., 28.], atol=1e-12)


@pytest.mark.parametrize("view", VIEWS)
def test_occluder_and_canvas_cover_are_area_fractions(view):
    s = view.scale
    def cover_1d(x, a, b):          # 独立：足迹与 [a, b) 的重叠 / s
        return np.array([max(0., min(xi+s/2, b)-max(xi-s/2, a))/s for xi in x])
    canvas = cover_1d(view.rows(), 0, 256)[:, None]*cover_1d(view.cols(), 0, 256)[None, :]
    np.testing.assert_allclose(camera.square_cover(view, 0., 0., 256.), canvas, atol=1e-12)
    assert abs(canvas.sum()-(256/s)**2) < 1e-6 or view.top > -s      # 视野包含画布时，覆盖比例之和 = 画布面积 / 像素面积
    image = np.full((view.size, view.size, 3), 200.)
    masked = camera.canvas_mask(image, view)
    inside = camera.square_cover(view, 0., 0., 256.) == 1.
    # 完全在画布里的像素判为精确的 1；足迹的边正好落在画布边上时 x ± s/2 有舍入，可能判成 1 − 1e-15 的部分覆盖（一行一列以内）
    assert np.all(canvas[inside] > 1-1e-12) and (canvas > 1-1e-12).sum()-inside.sum() <= 4*view.size
    assert np.array_equal(masked[inside], image[inside])
    np.testing.assert_allclose(masked, camera.SURROUND+(image-camera.SURROUND)*canvas[..., None], atol=1e-9)
    if view.top < -view.scale:
        assert np.array_equal(masked[0, 0], camera.SURROUND) and (canvas == 0).any()
    # 遮挡块：在画布上整数位置的 48×48 方块
    occ = cover_1d(view.rows(), 37, 85)[:, None]*cover_1d(view.cols(), 150, 198)[None, :]
    np.testing.assert_allclose(camera.square_cover(view, 37., 150., 48.), occ, atol=1e-12)


def test_occluders_inside_the_camera(use):
    """遮挡块（流 7）在相机里按面积混合：与画布上原遮挡块掩码的面积平均一致。"""
    use("double_well", "background=plain,occluders=2")
    view = View(192, 256/(192*.6), -60., -70.)
    with systems.raw_images():
        base = systems.compose(np.full((1, 1, 2), -1000.), np.array([[0., 0., 0.]]), 8., datasets.active().knobs, 0, ticks=[0])
        with systems.camera_view(view):
            cam = systems.compose(np.full((1, 1, 2), -1000.), np.array([[0., 0., 0.]]), 8., datasets.active().knobs, 0, ticks=[0])
    mask = np.all(base[0] == systems.OCCLUDER_COLOR, -1).astype(np.float64)            # 画布上的遮挡块
    assert mask.sum() > 2000
    Ar, Ac = overlap_matrix(view.rows(), view.scale), overlap_matrix(view.cols(), view.scale)
    cover = Ar@mask@Ac.T
    want = systems.BACKGROUND+(systems.OCCLUDER_COLOR-systems.BACKGROUND)*cover[..., None]
    # 画布外 overlap_matrix 是周期延拓的（画布外没有遮挡块，周期延拓会把另一边的块带进来），只比画布里面
    inside = camera.square_cover(view, 0., 0., 256.) == 1.
    assert inside.sum() > .9*(256/view.scale-2)**2           # 完全在画布里的像素精确判为 1（不受舍入影响）
    np.testing.assert_allclose(cam[0][inside], want[inside], atol=1e-9)
    full = inside & (cover > 1-1e-12)
    assert full.sum() > 500 and np.array_equal(cam[0][full], np.broadcast_to(systems.OCCLUDER_COLOR, cam[0][full].shape))


class _Corners:
    """遮挡块的随机流替身：integers() 返回给定的左上角（只测混色，不测抽样）。"""
    def __init__(self, corners):
        self.corners = np.asarray(corners)

    def integers(self, lo, hi, shape):
        assert shape == self.corners.shape and lo <= self.corners.min() and self.corners.max() < hi
        return self.corners.copy()


def test_overlapping_occluders_counterexample_from_the_review():
    """Codex 报告的确定反例：View(192, 2, −64, −64)，遮挡块 (37,150)、(37,170)，输出像素 (50,122) 的足迹
    [36,38]×[180,182] 下半被两块同时盖住：并集覆盖 0.5 → (67,70,76)（原实现逐块混色得 0.75 → (88.5,91,98)）。"""
    view = View(192, 2., -64., -64.)
    image = np.broadcast_to(systems.BACKGROUND, (1, 192, 192, 3)).astype(np.float64)
    with systems.camera_view(view):
        out = systems.occlude(image, _Corners([[37, 150], [37, 170]]), 2)
    np.testing.assert_allclose(out[0, 50, 122], [67., 70., 76.], atol=1e-12)


@pytest.mark.parametrize("view", VIEWS+[View(192, 2., -64., -64.)])
def test_overlapping_occluders_mix_by_the_union_area(view):
    """三块互相重叠的遮挡块：相机里每个像素 = 背景与遮挡色按并集覆盖比例混合（独立：画布上的并集掩码 × 重叠矩阵）；
    一块时与逐块混色逐位相同。"""
    corners = [[37, 150], [60, 170], [70, 140]]
    mask = np.zeros((256, 256))
    for r, c in corners:
        mask[r:r+48, c:c+48] = 1.
    Ar, Ac = clipped_overlap(view.rows(), view.scale), clipped_overlap(view.cols(), view.scale)
    cover = Ar@mask@Ac.T
    image = np.broadcast_to(systems.BACKGROUND, (1, view.size, view.size, 3)).astype(np.float64)
    with systems.camera_view(view):
        out = systems.occlude(image, _Corners(corners), 3)[0]
        one = systems.occlude(image, _Corners(corners[:1]), 1)[0]
    want = systems.BACKGROUND+(systems.OCCLUDER_COLOR-systems.BACKGROUND)*cover[..., None]
    np.testing.assert_allclose(out, want, atol=1e-9)
    assert ((cover > 1e-9) & (cover < 1-1e-9)).sum() > 10 and (cover > 1-1e-12).sum() > 100
    c1 = camera.square_cover(view, 37., 150., 48.)[..., None]
    old = np.where(c1 == 1., systems.OCCLUDER_COLOR, image[0]+(systems.OCCLUDER_COLOR-image[0])*c1)
    assert np.array_equal(one, old)


# —— 场的插值 ——

def test_sample_field_bilinear_and_periodic():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(2, 16, 16))
    with systems.camera_view(View(16, 4., 0., 0.)):          # 像素中心 = 格点中心（格子边长 4）
        assert np.array_equal(camera.sample_field(F, 4.), F)
    with systems.camera_view(View(16, 4., 64., -64.)):       # 平移整整一个周期（16 格 × 4）
        assert np.array_equal(camera.sample_field(F, 4.), F)
    with systems.camera_view(View(256, 1., 0., 0.)):
        G = rng.normal(size=(1, 256, 256))
        assert np.array_equal(camera.sample_field(G, 1.), G)
    # 格点中心之间：独立的双线性公式（行在格点 3、4 之间 1/4 处，列在格点 15 与回绕的 0 之间一半处）
    with systems.camera_view(View(1, 1., 4*3.5+1.-.5, 4*15.5+2.-.5)):
        got = camera.sample_field(F, 4.)[:, 0, 0]
    fr, fc = .25, .5
    want = ((1-fr)*((1-fc)*F[:, 3, 15]+fc*F[:, 3, 0])+fr*((1-fc)*F[:, 4, 15]+fc*F[:, 4, 0]))
    np.testing.assert_allclose(got, want, atol=1e-12)
    # 光滑场：插值误差 ≤ (h²/8)·max|f''|
    n = 128
    x = (np.arange(n)+.5)/n
    smooth = np.sin(2*np.pi*x)[None, :, None]*np.cos(4*np.pi*x)[None, None, :]
    view = View(320, 256/(320*.8), -32., -32.)
    with systems.camera_view(view):
        got = camera.sample_field(smooth, 2.)
    r, c = view.rows()/256, view.cols()/256
    exact = np.sin(2*np.pi*r)[:, None]*np.cos(4*np.pi*c)[None, :]
    assert np.abs(got[0]-exact).max() < (1/n)**2/8*(2*np.pi)**2*5


def test_edges_and_rings_scale_with_output_pixels():
    assert systems.edge_width() == 1.
    with systems.camera_view(View(192, 256/(192*.5), 0., 0.)):
        e = systems.edge_width()
        assert e == pytest.approx(8/3)
        d = np.linspace(-20., 30., 50001)
        w = camera.ring_weight(d, 5.)
        # 面积平均的高斯近似：方差 1 + (e² − 1)/12，积分（∫w dd = √(2π)）不变
        assert w.max() == pytest.approx(1/math.sqrt(1+(e*e-1)/12), rel=1e-6)
        assert np.trapezoid(w, d) == pytest.approx(math.sqrt(2*math.pi), rel=1e-6)
    with systems.camera_view(View(384, 2/3, 0., 0.)):
        assert systems.edge_width() == 1.
        d = np.linspace(0., 10., 11)
        assert np.array_equal(camera.ring_weight(d, 5.), np.exp(-(d-5.)**2/2.))


# —— 相机偏移 ——

def test_offsets_radius_direction_drift_and_stream(use):
    use("double_well", "zoom=0.5,pan=0.5,drift=0.4")
    v = datasets.active().knobs
    assert camera.width(v) == 512. and camera.slack(v) == 128.          # 写死：256/0.5，(512 − 256)/2
    t = np.array([0., 1., 2.5, 32.])
    o = camera.offsets(7, t)
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(simulation.seed_entropy(7, 12))))
    phi0 = g.uniform(0., 2*math.pi)
    a = phi0+.4*.25*t                                                     # 0.4 rad/s × 0.25 s/tick
    np.testing.assert_allclose(o[:, 1], 64*np.cos(a), atol=1e-12)       # 列 = ρ cos φ，ρ = 0.5 × 128
    np.testing.assert_allclose(o[:, 0], -64*np.sin(a), atol=1e-12)      # 行 = −ρ sin φ（世界 y 向上，逆时针）
    assert np.allclose(np.hypot(o[:, 0], o[:, 1]), 64.)
    assert not np.allclose(camera.offsets(8, t), o)                       # 每条轨迹方向不同
    use("double_well", "zoom=0.5,pan=0.5")
    assert np.allclose(camera.offsets(7, t), o[0])                        # 没有 drift：不动
    use("double_well", "zoom=0.5")
    assert np.array_equal(camera.offsets(7, t), np.zeros((4, 2)))


@pytest.mark.parametrize("knobs", ["zoom=0.5,pan=1,drift=0.5", "zoom=0.9,pan=1,drift=0.37,resolution=192",
                                   "zoom=0.75,pan=1,resolution=384"])
def test_view_always_contains_the_canvas(use, knobs):
    use("double_well", knobs)
    v = datasets.active().knobs
    t = np.linspace(0., 32., 129)
    for tid in range(20):
        for view in camera.views(tid, t):
            W = view.scale*view.size
            assert view.size == v.resolution and W == pytest.approx(256/v.zoom)
            assert view.top <= 1e-9 and view.left <= 1e-9 and view.top+W >= 256-1e-9 and view.left+W >= 256-1e-9


def test_spot_lands_where_the_camera_says(use):
    """double_well 的光点：画面上的强度质心 = 相机把画布位置 (128 − 48y, 128 + 48x) 映射到输出像素的位置，
    光点的二阶矩 = (σ/s)²（缩放和分辨率把光点按比例放大缩小）。"""
    use("double_well", "background=plain,zoom=0.6,pan=0.8,drift=0.3,resolution=320")
    tid = 3
    times = np.array([0., 10., 31.])
    states, _ = (lambda h: (h[0], None))(systems.get().frames(tid))
    color, bg = np.array(variants.COLORS["orange"]), systems.BACKGROUND
    W = 256/.6; s = W/320
    for t in times.astype(int):
        o = camera.offsets(tid, [float(t)])[0]
        top, left = 128+o[0]-W/2, 128+o[1]-W/2
        view = View(320, s, top, left)
        raw = raw_at(tid, [float(t)], view)[0]
        w = (raw[..., 0]-bg[0])/(color[0]-bg[0])
        i, j = np.meshgrid(np.arange(320)+.5, np.arange(320)+.5, indexing="ij")
        ci, cj = (w*i).sum()/w.sum(), (w*j).sum()/w.sum()
        x, y = states[t]
        assert ci == pytest.approx((128-48*y-top)/s, abs=1e-3) and cj == pytest.approx((128+48*x-left)/s, abs=1e-3)
        var = (w*(j-cj)**2).sum()/w.sum()
        assert math.sqrt(var) == pytest.approx(8/s, rel=1e-3)


# —— 状态、组合、噪声 ——

@pytest.mark.parametrize("system", ALL)
def test_hidden_states_unchanged_by_space_sampling(use, system):
    # 对照用可调实现（background=plain 只改观察）：advection_diffusion 的基础场景走原实现，隐藏状态差约 1e-16（S24a）
    use(system, "background=plain")
    tid = 5
    h0, f0 = systems.get().frames(tid)
    use(system, "background=plain,resolution=192,zoom=0.7,pan=0.6,drift=0.2")
    h, f = systems.get().frames(tid)
    assert np.array_equal(h, h0) and f.shape == (33, 192, 192, 3) and f.dtype == np.uint8
    if system in variants.SECOND_ORDER:
        hp, fp = systems.get().frames_with_prev(tid)
        assert fp.shape == (34, 192, 192, 3) and np.array_equal(fp[1:], f) and np.array_equal(hp[1:], h)


@pytest.mark.parametrize("system", ["double_well", "push", "wave", "advection_diffusion"])
def test_composes_with_fps_and_horizon(use, system):
    space = "resolution=320,zoom=0.8,pan=0.7,drift=0.3,pixel_noise=3"
    tid = 2
    use(system, space)
    h4, f4 = systems.get().frames(tid)
    use(system, join(space, "fps=8"))
    h8, f8 = systems.get().frames(tid)
    assert f8.shape == (65, 320, 320, 3) and np.array_equal(f8[::2], f4) and np.array_equal(h8[::2], h4)
    assert not np.array_equal(f8[1], f8[0])
    use(system, join(space, "horizon=64"))
    hl, fl = systems.get().frames(tid)
    assert fl.shape[0] == 65 and np.array_equal(fl[:33], f4) and np.array_equal(hl[:33], h4)


def test_pixel_noise_at_output_resolution(use):
    """整 tick 的帧：流 4 一次抽 [33,R,R,3] 的第 t 份（R = 192），按块顺序抽与一次抽逐位相同。"""
    use("van_der_pol", "resolution=192,zoom=0.9,pixel_noise=7")
    tid = 9
    _, f = systems.get().frames(tid)
    noise = simulation._rng(tid, 4).normal(0., 7., (33, 192, 192, 3))
    v = datasets.active().knobs
    raw = np.stack([camera.canvas_mask(raw_at(tid, [float(t)], camera.views(tid, [float(t)], v)[0])[0],
                                       camera.views(tid, [float(t)], v)[0]) for t in (0, 13, 32)])
    want = np.rint(np.clip(raw+noise[[0, 13, 32]], 0., 255.)).astype(np.uint8)
    assert np.array_equal(f[[0, 13, 32]], want)
    blocks = timing._NoiseBlocks(simulation._rng(tid, 4), 7., 192)
    assert np.array_equal(blocks.block(5), noise[5])


def test_exposure_uses_each_sample_time_camera(use):
    """drift + exposure：一帧是曝光窗口里每个取样时刻在它自己的相机视野里画（画布外混成 SURROUND）后的平均。"""
    use("pendulum", "zoom=0.6,pan=1,drift=0.5,exposure=1")
    tid = 4
    v = datasets.active().knobs
    _, f = systems.get().frames(tid)
    k = 10
    ts = np.maximum(k-np.arange(9)/8, -1.)          # E = 1 tick，M = 1 + ⌈8⌉ = 9 个时刻
    raws = [camera.canvas_mask(raw_at(tid, [t], camera.views(tid, [t], v)[0])[0], camera.views(tid, [t], v)[0]) for t in ts]
    want = np.rint(np.clip(np.mean(raws, 0), 0., 255.)).astype(np.uint8)
    assert np.array_equal(f[k], want)
    still = np.rint(np.clip(raws[0], 0., 255.)).astype(np.uint8)
    assert not np.array_equal(f[k], still)


def test_camera_offset_is_readable_from_one_frame(use):
    """一阶：从一帧里画布（plain 背景）在画面上的位置读出相机偏移，按 drift 转 0.25·ω 推出下一帧的偏移，与下一帧读出的一致。"""
    use("double_well", "background=plain,zoom=0.5,pan=0.9,drift=0.5,resolution=256")
    _, f = systems.get().frames(6)
    s = 2.                                          # 512/256
    def canvas_box(img):
        inside = np.abs(img.astype(np.float64)-systems.BACKGROUND).max(-1) < 3
        rows, cols = np.where(inside)
        return (rows.min()+rows.max()+1)/2, (cols.min()+cols.max()+1)/2      # 画布中心（输出像素）
    for k in (0, 7, 20):
        r0, c0 = canvas_box(f[k]); r1, c1 = canvas_box(f[k+1])
        o = np.array([(128-r0)*s, (128-c0)*s])       # 视野中心在画布中心的哪边：画布中心在画面上的反方向
        a = .5*.25
        # 行 = −ρ sin φ、列 = ρ cos φ：φ → φ + a 时 行' = 行·cos a − 列·sin a，列' = 列·cos a + 行·sin a
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        o1 = rot@o
        assert np.hypot(*o) == pytest.approx(.9*128, abs=2*s)
        assert abs((128-r1)*s-o1[0]) <= 2*s and abs((128-c1)*s-o1[1]) <= 2*s


# —— 编码、探针、规划拒绝 ——

def _clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")}
    env["PYTHONPATH"] = str(SRC)
    return env


def test_encode_probe_and_plan_refuse_space_sampling(tmp_path):
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/encode_v7.py"), "encode", "--system", "double_well", "--variant",
                        "resolution=384", "--trajectories", "40", "--fake-encoder", "--device", "cpu", "--output", str(tmp_path/"enc")],
                       env=_clean_env(), capture_output=True, text=True)
    assert r.returncode != 0 and "空间采样" in r.stderr and not (tmp_path/"enc").exists()
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/probe_actions.py"), "--system", "push", "--variant", "zoom=0.8",
                        "--trajectories", "40", "--fake-encoder", "--device", "cpu", "--output", str(tmp_path/"p.json")],
                       env=_clean_env(), capture_output=True, text=True)
    assert r.returncode != 0 and "空间采样" in r.stderr and not (tmp_path/"p.json").exists()
    job = tmp_path/"job"; job.mkdir()
    (job/"job.json").write_text(json.dumps(dict(seed=0, updates=1, dataset="double_well~pan=0.5,zoom=0.8-40", condition="identity")))
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/plan_v7.py"), "--job", str(job), "--fake-encoder",
                        "--device", "cpu", "--output", str(tmp_path/"plan.json")], env=_clean_env(), capture_output=True, text=True)
    assert r.returncode != 0 and "空间采样" in r.stderr and not (tmp_path/"plan.json").exists()
