"""S18a：D4 的公式（10 §3.2–§3.7）、等价核对的重算（§4.3）、三角不等式核对和任务入口的输出（§7）。

玩具起点 z[0, n, e] = A_n·t_e，E = 4，t = (1, −1, 1, −1)，A_n 在 16×16 上按 (i+j) 的奇偶取 1.5、0.5（每个 2×2 块里
两个 1.5、两个 0.5）：每个 token 沿 E 均值 0；s²(z) = 1.25；R z（16→8 的 2×2 平均）的幅度恰为 1。目标 z*_T = −z。
三个玩具模型都忽略动作和网格、和 R 可交换：

1. 14 §3 S18 的两个线性玩具（a = 1/40）：半群律精确的 Φ_T(z) = e^{−aT} z，和不满足半群律的 Φ_T(z) = (1 − aT) z。
   记 c_T = 1 − aT、c_Π = Π(1 − aτ_i)：pd_in(T, Π) = |c_T − c_Π|（RMS 和 s 相消），xpd_in(T, n) = |c_T − c_{Π_n}|/√1.25，
   tcd_unequal_in = |c_d − (1 − aτ)(1 − aσ)|/(τσ) = a² = 0.000625，long_pd_in(T) = |c_T − c_8^{T/8}|。
   这两个模型只改尺度，LN 把尺度几乎消掉（差只剩 eps 的影响，1e-8 量级，比 float32 的分辨率还小），所以 LN 空间的量
   和 JEPA 距离的差用下面的第 3 个玩具核。
2. 半群律精确的漂移 Φ_T(z) = z + Tβq，β = 0.01，q = (1, 1, −1, −1)（沿 E 均值 0、和 t 正交）：所有差都为 0。
3. 不满足半群律的漂移 Φ_T(z) = z + T²βq：Φ_Π = z + (Στ_i²)βq，所以 Φ_T − Φ_Π = 2x(Π)βq，
   pd_in(T, Π) = 2x(Π)β/√1.25（正比于 x(Π)），tcd_unequal_in = 2β/√1.25。LN 空间：幅度 A 的 token 上
   LN(A t + S q) = (A t + S q)/√(A² + S² + 1e-5)，记 α = A/√(…)、γ = S/√(…)，则 RMS_LN 的平方是 Δα² + Δγ² 在 token 上的平均，
   JEPA 距离 mean|LN(a) − LN(b)| 是 max(|Δα|, |Δγ|) 在 token 上的平均（目标 −z 的 α = −A/√(A² + 1e-5)、γ = 0）。
模型输出是 float32，线性玩具的比较用 rel=1e-4（c·z 的舍入相对差到 ~3e-5），漂移玩具用 rel=1e-5。
"""

import json
import math

import pytest
import torch

from po_jepa import datasets
from po_jepa.runtime import cache, comparison, evaluation
from po_jepa.runtime.addon import d4, v7load
from test_evaluation import Affine, fictional_window
from test_training import e06_config

A = 1/40
BETA = .01
EPS = 1e-5
E = 4
S5 = math.sqrt(1.25)
Q = torch.tensor([1., 1., -1., -1.])


class Toy(torch.nn.Module):
    """kind：semigroup（e^{−aT}z）、linear（(1 − aT)z）、drift（z + Tβq）、quadratic（z + T²βq）、
    fine_drift（16×16 上 z + Tβq，粗网格上原样返回 z，只用来核 scd_ln 和 xpd 的跨网格一侧）；记录每次调用 (N, T)。"""
    def __init__(self, kind):
        super().__init__()
        self.kind, self.calls = kind, []

    def forward(self, raw, controls, duration, grid):
        t = float(duration[0])
        if raw.shape[1] != grid.height*grid.width or float(controls.duration[0]) != t:
            raise AssertionError("toy received inconsistent grid/controls")
        self.calls.append((raw.shape[1], t))
        if self.kind == "semigroup":
            return (raw.double()*math.exp(-A*t)).float()
        if self.kind == "linear":
            return (raw.double()*(1-A*t)).float()
        if self.kind == "fine_drift" and raw.shape[1] != 256:
            return raw.clone()
        shift = BETA*(t*t if self.kind == "quadratic" else t)
        return (raw.double()+shift*Q.double()).float()


def toy_z():
    i = torch.arange(16).repeat_interleave(16)
    j = torch.arange(16).repeat(16)
    amp = torch.where((i+j) % 2 == 0, 1.5, .5)
    sign = torch.tensor([1., -1., 1., -1.])
    return (amp[:, None]*sign[None, :])[None].to(torch.float32)


def toy_window(duration, tid=36, start=0):
    """动作：32 段各 1 个 tick，u_x = 0.01·(k+1)；目标 −z。"""
    metadata = dict(trajectory_id=tid, controls=dict(starts_ticks=list(range(32)), ends_ticks=list(range(1, 33)),
                    values=[[.01*(k+1), 0.] for k in range(32)], units="simulation_position_per_second"))
    controls = cache._window_controls(metadata, start, float(duration), "cpu")
    z = toy_z()
    x = cache.WindowInput(z, controls, torch.tensor([float(duration)], dtype=torch.float64), cache.regular_grid(16, 16),
                          torch.tensor([start*.25], dtype=torch.float64), "input", tid, "test")
    y = cache.WindowTarget(-z, torch.tensor([(start+duration)*.25], dtype=torch.float64), "target")
    return cache.CacheWindow(x, y, dict(actual_manifest_sha256=f"manifest-{tid}", generation=dict(evidence_role="real_frozen_encoder"),
                                        observation_variant="canonical"))


def toy_windows():
    return {t: toy_window(t) for t in d4.TCD_TIMES}


def long_windows():
    return {t: toy_window(t) for t in d4.LONG_TIMES}


def c(parts):
    return math.prod(1-A*p for p in parts)


def ag(amp, shift):
    n = math.sqrt(amp*amp+shift*shift+EPS)
    return amp/n, shift/n


def drift_ln(s1, s2, amps=(1.5, .5)):
    return math.sqrt(sum((ag(a, s1)[0]-ag(a, s2)[0])**2+(ag(a, s1)[1]-ag(a, s2)[1])**2 for a in amps)/len(amps))


def drift_jepa(s1, s2):
    return sum(max(abs(ag(a, s1)[0]-ag(a, s2)[0]), abs(ag(a, s1)[1]-ag(a, s2)[1])) for a in (1.5, .5))/2


def drift_jepa_target(s1):
    return sum(max(abs(ag(a, s1)[0]+a/math.sqrt(a*a+EPS)), abs(ag(a, s1)[1])) for a in (1.5, .5))/2


def sq(parts):
    return BETA*sum(p*p for p in parts)


# ---- 手算值（10 位有效数字；test_hand_values_match_closed_forms 用上面的公式复核每一个） ----
# 线性玩具 (1 − aT)z，a = 1/40：输入尺度的量
LINEAR_IN = {"pd_in:T=4:2x2": 0.0025, "pd_in:T=4:4x1": 0.003687890625, "pd_in:T=4:8x0.5": 0.004267317001,
             "pd_in:T=8:2x4": 0.01, "pd_in:T=8:4x2": 0.01450625, "pd_in:T=8:8x1": 0.01665180366, "pd_in:T=8:3+5": 0.009375,
             "xpd_in:T=4:n=2": 0.002236067977, "xpd_in:T=4:n=4": 0.003298549652,
             "xpd_in:T=8:n=2": 0.008944271910, "xpd_in:T=8:n=4": 0.01297478444,
             "tcd_unequal_in:d=4:1+3": 0.000625, "tcd_unequal_in:d=4:3+1": 0.000625,
             "tcd_unequal_in:d=8:2+6": 0.000625, "tcd_unequal_in:d=8:6+2": 0.000625}
LINEAR_LONG_IN = {"long_pd_in:T=16": 0.04, "long_pd_in:T=24": 0.112, "long_pd_in:T=32": 0.2096}
# 二次漂移玩具 z + T²βq，β = 0.01
QUADRATIC = {
    "pd_ln:T=4:2x2": 0.1130876301, "pd_in:T=4:2x2": 0.07155417528, "pd_ln:T=4:4x1": 0.1716694584, "pd_in:T=4:4x1": 0.1073312629,
    "pd_ln:T=4:8x0.5": 0.2011698169, "pd_in:T=4:8x0.5": 0.1252198067, "pd_ln:T=8:2x4": 0.2743311937, "pd_in:T=8:2x4": 0.2862167011,
    "pd_ln:T=8:4x2": 0.4661078422, "pd_in:T=8:4x2": 0.4293250517, "pd_ln:T=8:8x1": 0.5728935462, "pd_in:T=8:8x1": 0.500879227,
    "pd_ln:T=8:3+5": 0.2530161583, "pd_in:T=8:3+5": 0.2683281573,
    "xpd_ln:T=4:n=2": 0.07880448292, "xpd_in:T=4:n=2": 0.07155417528, "xpd_ln:T=4:n=4": 0.1186063582, "xpd_in:T=4:n=4": 0.1073312629,
    "xpd_ln:T=8:n=2": 0.2588807694, "xpd_in:T=8:n=2": 0.2862167011, "xpd_ln:T=8:n=4": 0.4077767331, "xpd_in:T=8:n=4": 0.4293250517,
    "tcd_ln:T=1": 0.02980724704, "tcd_ln:T=2": 0.02971093688, "tcd_ln:T=3": 0.02930270154, "tcd_ln:T=4": 0.02827190752,
    "tcd_ln:T=8": 0.01714569961, "tcd_ln:T=12": 0.008190871309,
    "tcd_unequal_ln:d=4:1+3": 0.02806227222, "tcd_unequal_in:d=4:1+3": 0.01788854382,
    "tcd_unequal_ln:d=4:3+1": 0.02806227222, "tcd_unequal_in:d=4:3+1": 0.01788854382,
    "tcd_unequal_ln:d=8:2+6": 0.01606166026, "tcd_unequal_in:d=8:2+6": 0.01788854382,
    "tcd_unequal_ln:d=8:6+2": 0.01606166026, "tcd_unequal_in:d=8:6+2": 0.01788854382,
    "heldout_direct:T=3": 1.991172336, "heldout_partition:T=3": 0.05211260983, "heldout_rollout:T=3": 1.997218977,
    "heldout_direct:T=12": 1.524687624, "heldout_partition:T=12": 0.2119642928, "heldout_rollout:T=12": 1.706162578}
QUADRATIC_LONG = {
    "long_direct:T=16": 1.348607181, "long_composed:T=16": 1.562256505, "long_pd_ln:T=16": 0.2672959394, "long_pd_in:T=16": 1.144866804,
    "long_direct:T=24": 1.169234796, "long_composed:T=24": 1.433816251, "long_pd_ln:T=24": 0.3104311051, "long_pd_in:T=24": 3.434600413,
    "long_direct:T=32": 1.096842704, "long_composed:T=32": 1.348607181, "long_pd_ln:T=32": 0.2887941857, "long_pd_in:T=32": 6.869200827}
IDENTITY_TARGET = 1.999977778     # d(z, −z) = ½Σ_A 2A/√(A² + 1e-5)
# fine_drift 玩具：scd_ln:h=8:T=t = RMS_LN(R z + tβq, R z)/t，R z 的幅度为 1
FINE_DRIFT_SCD8 = {1.: 0.009999575029, 2.: 0.009998450405, 3.: 0.009996577, 4.: 0.009993956263,
                   8.: 0.009976049002, 12.: 0.009946447431}


def test_registered_tables_match_spec():
    assert [(t, d4.partition_label(p), d4.partition_x(t, p)) for t, p in d4.PARTITIONS] == [
        (4., "2x2", 4.), (4., "4x1", 6.), (4., "8x0.5", 7.), (8., "2x4", 16.), (8., "4x2", 24.), (8., "8x1", 28.), (8., "3+5", 15.)]
    assert d4.CROSS == ((4., 2), (4., 4), (8., 2), (8., 4))
    assert d4.UNEQUAL == ((4., (1., 3.)), (4., (3., 1.)), (8., (2., 6.)), (8., (6., 2.)))
    assert d4.HELDOUT == ((3., (1., 2.)), (12., (8., 4.)))
    assert d4.TCD_TIMES == evaluation.TIMES and d4.SCD_GRIDS == (8, 12)
    assert d4.LONG_TIMES == (16., 24., 32.) and d4.LONG_BLOCK == 8.
    assert d4.EQUIVALENCE_TOLERANCE == 1e-6 and d4.TRIANGLE_TOLERANCE == 1e-6
    assert len(d4.value_names()) == 54 and len(set(d4.value_names())) == 54
    assert len(d4.equivalence_names()) == 24 and len(d4.long_value_names()) == 12




def _label(t, parts):
    return f"T={t:g}:{d4.partition_label(parts)}"


def test_hand_values_match_closed_forms():
    """表里每个数都由文件头的闭式复核（这一条不调用 d4，只核表本身）。"""
    for t, parts in d4.PARTITIONS:
        assert LINEAR_IN[f"pd_in:{_label(t, parts)}"] == pytest.approx(abs(c([t])-c(parts)), rel=1e-9)
        assert QUADRATIC[f"pd_ln:{_label(t, parts)}"] == pytest.approx(drift_ln(sq([t]), sq(parts)), rel=1e-9)
        assert QUADRATIC[f"pd_in:{_label(t, parts)}"] == pytest.approx(2*d4.partition_x(t, parts)*BETA/S5, rel=1e-9)
        assert QUADRATIC[f"pd_in:{_label(t, parts)}"] == pytest.approx(abs(sq([t])-sq(parts))/S5, rel=1e-9)
    for t, n in d4.CROSS:
        assert LINEAR_IN[f"xpd_in:T={t:g}:n={n}"] == pytest.approx(abs(c([t])-c([t/n]*n))/S5, rel=1e-9)
        assert QUADRATIC[f"xpd_ln:T={t:g}:n={n}"] == pytest.approx(drift_ln(sq([t]), sq([t/n]*n), amps=(1.,)), rel=1e-9)
        assert QUADRATIC[f"xpd_in:T={t:g}:n={n}"] == pytest.approx(abs(sq([t])-sq([t/n]*n))/S5, rel=1e-9)
    for d, parts in d4.UNEQUAL:
        name = f"d={d:g}:{d4.partition_label(parts)}"
        tau, sigma = parts
        assert LINEAR_IN[f"tcd_unequal_in:{name}"] == pytest.approx(abs(c([d])-c(parts))/(tau*sigma), rel=1e-9)
        assert LINEAR_IN[f"tcd_unequal_in:{name}"] == pytest.approx(A*A, rel=1e-12)
        assert QUADRATIC[f"tcd_unequal_ln:{name}"] == pytest.approx(drift_ln(sq([d]), sq(parts))/(tau*sigma), rel=1e-9)
        assert QUADRATIC[f"tcd_unequal_in:{name}"] == pytest.approx(2*BETA/S5, rel=1e-9)
    for t in d4.TCD_TIMES:
        assert QUADRATIC[f"tcd_ln:T={t:g}"] == pytest.approx(drift_ln(sq([t]), sq([t/2, t/2]))/(t/2)**2, rel=1e-9)
    for t, parts in d4.HELDOUT:
        assert QUADRATIC[f"heldout_direct:T={t:g}"] == pytest.approx(drift_jepa_target(sq([t])), rel=1e-9)
        assert QUADRATIC[f"heldout_partition:T={t:g}"] == pytest.approx(drift_jepa(sq([t]), sq(parts)), rel=1e-9)
        assert QUADRATIC[f"heldout_rollout:T={t:g}"] == pytest.approx(drift_jepa_target(sq(parts)), rel=1e-9)
    for t in d4.LONG_TIMES:
        blocks = [8.]*int(t/8)
        assert LINEAR_LONG_IN[f"long_pd_in:T={t:g}"] == pytest.approx(abs(c([t])-c(blocks)), rel=1e-9)
        assert QUADRATIC_LONG[f"long_direct:T={t:g}"] == pytest.approx(drift_jepa_target(sq([t])), rel=1e-9)
        assert QUADRATIC_LONG[f"long_composed:T={t:g}"] == pytest.approx(drift_jepa_target(sq(blocks)), rel=1e-9)
        assert QUADRATIC_LONG[f"long_pd_ln:T={t:g}"] == pytest.approx(drift_ln(sq([t]), sq(blocks)), rel=1e-9)
        assert QUADRATIC_LONG[f"long_pd_in:T={t:g}"] == pytest.approx(abs(sq([t])-sq(blocks))/S5, rel=1e-9)
    for t, value in FINE_DRIFT_SCD8.items():
        assert value == pytest.approx(drift_ln(BETA*t, 0., amps=(1.,))/t, rel=1e-9)
    assert IDENTITY_TARGET == pytest.approx(drift_jepa_target(0.), rel=1e-9)
    assert set(QUADRATIC) == set(d4.value_names())-{n for n in d4.value_names() if n.startswith("scd_ln")}
    assert set(QUADRATIC_LONG) == set(d4.long_value_names())


@pytest.mark.parametrize("kind", ["semigroup", "drift"])
def test_semigroup_toys_have_zero_differences(kind):
    """半群律精确、和 R 可交换：除 heldout_direct、heldout_rollout 外全部 ≤ 1e-6；两者相等到 1e-6。"""
    values, _, triangle = d4.window_values(Toy(kind), toy_windows(), e06_config())
    for name, value in values.items():
        if not name.startswith(("heldout_direct", "heldout_rollout")):
            assert abs(value) <= 1e-6, name
    for t, _ in d4.HELDOUT:
        assert values[f"heldout_direct:T={t:g}"] == pytest.approx(values[f"heldout_rollout:T={t:g}"], abs=1e-6)
    assert triangle == {"T=3": True, "T=12": True}
    long = d4.long_values(Toy(kind), long_windows(), e06_config())
    for t in d4.LONG_TIMES:
        assert abs(long[f"long_pd_ln:T={t:g}"]) <= 1e-6 and abs(long[f"long_pd_in:T={t:g}"]) <= 1e-6
        assert long[f"long_direct:T={t:g}"] == pytest.approx(long[f"long_composed:T={t:g}"], abs=1e-6)


def test_linear_toy_matches_input_scale_hand_values():
    """(1 − aT)z：输入尺度的量对上 LINEAR_IN（rel 1e-4）；LN 空间的量和 scd_ln 都 ≤ 1e-6（LN 消掉尺度）。"""
    values, _, triangle = d4.window_values(Toy("linear"), toy_windows(), e06_config())
    for name, expected in LINEAR_IN.items():
        assert values[name] == pytest.approx(expected, rel=1e-4), name
    for name, value in values.items():
        if name.startswith(("tcd_ln", "scd_ln", "pd_ln", "xpd_ln", "tcd_unequal_ln")):
            assert abs(value) <= 1e-6, name
    assert triangle == {"T=3": True, "T=12": True}
    long = d4.long_values(Toy("linear"), long_windows(), e06_config())
    for name, expected in LINEAR_LONG_IN.items():
        assert long[name] == pytest.approx(expected, rel=1e-4), name


def test_quadratic_toy_matches_hand_values():
    """z + T²βq：LN 空间、输入尺度和 JEPA 距离的量全部对上 QUADRATIC / QUADRATIC_LONG（rel 1e-5）；scd_ln ≤ 1e-6。"""
    values, _, triangle = d4.window_values(Toy("quadratic"), toy_windows(), e06_config())
    for name, expected in QUADRATIC.items():
        assert values[name] == pytest.approx(expected, rel=1e-5), name
    for t in d4.TCD_TIMES:
        for h in d4.SCD_GRIDS:
            assert abs(values[f"scd_ln:h={h}:T={t:g}"]) <= 1e-6
    assert triangle == {"T=3": True, "T=12": True}
    long = d4.long_values(Toy("quadratic"), long_windows(), e06_config())
    for name, expected in QUADRATIC_LONG.items():
        assert long[name] == pytest.approx(expected, rel=1e-5), name


def test_fine_drift_toy_checks_cross_grid_side():
    """16×16 上 z + Tβq、粗网格上 z：scd_ln:h=8 对上 FINE_DRIFT_SCD8；h=12 按 R z 的逐 token 幅度用同一闭式；
    xpd_ln = RMS_LN(R z + Tβq, R z)（和 n 无关），xpd_in = Tβ/√1.25（s² 用 16×16 起点）；16×16 上的量全为 0。"""
    values, _, _ = d4.window_values(Toy("fine_drift"), toy_windows(), e06_config())
    transfer = comparison.build_spatial_transfer(cache.regular_grid(16, 16), target_height=12, target_width=12)
    amps12 = comparison.apply_spatial_transfer(toy_z(), transfer)[0, :, 0].double().tolist()
    for t in d4.TCD_TIMES:
        assert values[f"scd_ln:h=8:T={t:g}"] == pytest.approx(FINE_DRIFT_SCD8[t], rel=1e-5)
        assert values[f"scd_ln:h=12:T={t:g}"] == pytest.approx(drift_ln(BETA*t, 0., amps=amps12)/t, rel=1e-5)
        assert abs(values[f"tcd_ln:T={t:g}"]) <= 1e-6
    for t, n in d4.CROSS:
        assert values[f"xpd_ln:T={t:g}:n={n}"] == pytest.approx(FINE_DRIFT_SCD8[t]*t, rel=1e-5)
        assert values[f"xpd_in:T={t:g}:n={n}"] == pytest.approx(BETA*t/S5, rel=1e-5)
    for name, value in values.items():
        if name.startswith(("pd_", "tcd_unequal", "heldout_partition")):
            assert abs(value) <= 1e-6, name


def test_shared_tensors_give_bitwise_equal_quantities():
    values, new, _ = d4.window_values(Toy("quadratic"), toy_windows(), e06_config())
    for t in d4.TCD_TIMES:
        if t in (4., 8.):
            assert values[f"pd_ln:{_label(t, (t/2, t/2))}"]/(t/2)**2 == values[f"tcd_ln:T={t:g}"]
    for t, _ in d4.HELDOUT:
        assert values[f"heldout_direct:T={t:g}"] == new[f"future_jepa_distance:T={t:g}"]


def test_call_plan_and_rollout_order():
    """每个窗口 92 次调用、每条长时程轨迹 12 次；分割逐段按表里的先后。"""
    model = Toy("quadratic")
    d4.window_values(model, toy_windows(), e06_config())
    assert len(model.calls) == 92
    durations = [t for _, t in model.calls]
    assert durations[:18] == [1., .5, .5, 2., 1., 1., 3., 1.5, 1.5, 4., 2., 2., 8., 4., 4., 12., 6., 6.]
    assert [n for n, _ in model.calls[18:42]] == [256, 64, 256, 144]*6          # scd：先细网格后粗网格，先 8 后 12
    assert durations[18:42] == [t for t in d4.TCD_TIMES for _ in range(4)]       # scd 的动作不切开
    assert durations[42:68] == [1.]*4+[.5]*8+[2.]*4+[1.]*8+[3., 5.]           # pd：2x2、2x4 复用 tcd 的组合
    assert durations[68:80] == [2., 2., 1., 1., 1., 1., 4., 4., 2., 2., 2., 2.]
    assert {n for n, _ in model.calls[68:80]} == {64}                           # xpd 的组合在 8×8 上
    assert durations[80:88] == [1., 3., 3., 1., 2., 6., 6., 2.]                 # 不等分 TCD：先 τ 后 σ
    assert durations[88:] == [1., 2., 8., 4.]                                   # 留出分解：先 1 后 2，先 8 后 4
    model = Toy("quadratic")
    d4.long_values(model, long_windows(), e06_config())
    assert [t for _, t in model.calls] == [16., 8., 8., 24., 8., 8., 8., 32., 8., 8., 8., 8.]
    assert {n for n, _ in model.calls} == {256}


def test_pieces_split_exactly_and_restart_local_time():
    w = toy_window(8.)
    parts = d4.pieces(w.inputs.controls, (3., 5.))
    assert [float(p.duration[0]) for p in parts] == [3., 5.]
    assert parts[0].u_values[0, :3, 0].tolist() == pytest.approx([.01, .02, .03])
    assert parts[1].u_values[0, 0, 0].item() == pytest.approx(.04)
    assert float(parts[1].starts[0, 0]) == 0.
    assert float(parts[1].source_origin_seconds[0]) == pytest.approx(3*.25)
    eight = d4.pieces(toy_window(4.).inputs.controls, (.5,)*8)
    assert [float(p.duration[0]) for p in eight] == [.5]*8
    assert [p.u_values[0, 0, 0].item() for p in eight] == pytest.approx([.01, .01, .02, .02, .03, .03, .04, .04])


class SegmentCount(torch.nn.Module):
    """输出随动作的段数变：z + 0.01·段数。v7 的 direct 用在 T/2 处切开再接上的动作（切点落在段内时多一段），
    D4 的 Φ_T 必须用同样的动作，否则等价核对对不上（§9 第 1 条）。"""
    def forward(self, z, controls, duration, grid):
        return z+.01*torch.as_tensor(controls.n_segments).to(z)[:, None, None]


def test_equivalence_values_equal_v7_point_and_pair_metrics():
    """同一窗口、同一模型（v7 的 Affine 和上面的 SegmentCount）：D4 重算的三项和 evaluation.point_metrics / pair_metrics 的输出逐位相同。"""
    api = evaluation
    windows = {t: fictional_window(api, 180, 0, t) for t in d4.TCD_TIMES}
    donor = {t: fictional_window(api, 181, 0, t, action=-.2) for t in d4.TCD_TIMES}
    for make in (Affine, SegmentCount):
        _, new, _ = d4.window_values(make().eval(), windows, e06_config())
        for t in d4.TCD_TIMES:
            values, _ = api.point_metrics(make().eval(), windows[t], donor[t], api.EvaluationQuery(180, 0, t, 16, "point"), e06_config())
            assert new[f"future_jepa_distance:T={t:g}"] == values["future_jepa_distance"]
            assert new[f"tcd_normalized:T={t:g}"] == values["tcd_normalized"]
            for h in d4.SCD_GRIDS:
                values, _ = api.pair_metrics(make().eval(), windows[t], api.EvaluationQuery(180, 0, t, h, "pair"))
                assert new[f"scd_normalized:h={h}:T={t:g}"] == values["scd_normalized"]


def test_identity_gives_zero_differences():
    from po_jepa.baselines import IdentityPredictor
    values, _, triangle = d4.window_values(IdentityPredictor(E), toy_windows(), e06_config())
    for name, value in values.items():
        if name.startswith(("heldout_direct", "heldout_rollout")):
            assert value == pytest.approx(IDENTITY_TARGET, rel=1e-6)
        else:
            assert value == 0., name
    assert triangle == {"T=3": True, "T=12": True}
    long = d4.long_values(IdentityPredictor(E), long_windows(), e06_config())
    for t in d4.LONG_TIMES:
        assert long[f"long_direct:T={t:g}"] == long[f"long_composed:T={t:g}"] == pytest.approx(IDENTITY_TARGET, rel=1e-6)
        assert long[f"long_pd_ln:T={t:g}"] == 0. and long[f"long_pd_in:T={t:g}"] == 0.


def test_triangle_and_equivalence_rules():
    assert d4.triangle_holds(1., .4, .6) and d4.triangle_holds(1.+1e-6, .4, .6)
    assert not d4.triangle_holds(1.+2e-6, .4, .6) and not d4.triangle_holds(1., .3, .6)
    assert d4.triangle_holds(.5, 0., .5-.9e-6) and not d4.triangle_holds(.5, 0., .5-1.1e-6)
    assert d4.triangle_holds(4., 1., 3.-3.9e-6) and not d4.triangle_holds(4., 1., 3.-4.1e-6)   # 容差随 direct 放大
    assert d4.compare_equivalence(1.+2**-20, 1.)[:2] == (True, False)
    assert d4.compare_equivalence(1.+2**-19, 1.)[0] is False
    assert d4.compare_equivalence(4.+2**-19, 4.)[0] is True       # 容差随 |旧| 放大
    assert d4.compare_equivalence(.5+.9e-6, .5)[0] is True and d4.compare_equivalence(.5+1.1e-6, .5)[0] is False
    assert d4.compare_equivalence(.25, .25) == (True, True, 0., 0.)
    assert d4.compare_equivalence(0., 0)[3] is None
    for bad in (float("nan"), float("inf"), None, "1.0", True):
        with pytest.raises(ValueError, match="finite number"):
            d4.compare_equivalence(1., bad)


def test_value_order_guards(monkeypatch):
    """window_values / long_values 的键顺序必须等于登记的 value_names / equivalence_names / long_value_names。"""
    names, equivalence = d4.value_names(), d4.equivalence_names()
    monkeypatch.setattr(d4, "value_names", lambda: tuple(reversed(names)))
    with pytest.raises(RuntimeError, match="registered order"):
        d4.window_values(Toy("quadratic"), toy_windows(), e06_config())
    monkeypatch.setattr(d4, "value_names", lambda: names)
    monkeypatch.setattr(d4, "equivalence_names", lambda: tuple(reversed(equivalence)))
    with pytest.raises(RuntimeError, match="registered order"):
        d4.window_values(Toy("quadratic"), toy_windows(), e06_config())
    monkeypatch.setattr(d4, "long_value_names", lambda: ())
    with pytest.raises(RuntimeError, match="long-horizon names differ"):
        d4.long_values(Toy("quadratic"), long_windows(), e06_config())


def _write_reference(folder, queries, seed, mutate=lambda rows: rows):
    """照 v7 evaluation.evaluate_condition 的写法造 evaluation/<条件>/：registration.json、rows/<序号>.json、COMPLETE.json。"""
    from dataclasses import asdict
    (folder/"rows").mkdir(parents=True)
    cache._write_json(folder/"registration.json", dict(condition="po_jepa_full", queries=[asdict(q) for q in queries]))
    rows = [dict(query=asdict(q), condition="po_jepa_full", seed=seed, status="ok",
                 values=dict(future_jepa_distance=.5+i, tcd_normalized=.25+i) if q.kind == "point" else dict(scd_normalized=.125+i))
            for i, q in enumerate(queries)]
    for name, row in mutate(list(enumerate(rows))):
        cache._write_json(folder/"rows"/f"{name:05d}.json", row)
    cache._write_json(folder/"COMPLETE.json", dict(status="completed_fictional_cpu_only", error=None))


def test_reference_rows_reads_v7_layout(tmp_path):
    """reference_rows 按 (kind, 轨迹, 起点, 时长, 网格) 取 v7 的评测行；查询列表、条件、种子、状态、重复行、未完成都核。"""
    from po_jepa.runtime import checks
    old = datasets._set_for_tests(datasets.DatasetSpec("van_der_pol", 40))
    try:
        queries, seed = evaluation.build_index(), checks.model_seed()
        assert len(queries) == 560
        _write_reference(tmp_path/"good"/"evaluation"/"po_jepa_full", queries, seed)
        found = d4.reference_rows(tmp_path/"good", "po_jepa_full")
        assert len(found) == 560
        i = next(k for k, q in enumerate(queries) if (q.kind, q.grid_size, q.duration_cycles) == ("pair", 12, 3.))
        q = queries[i]
        assert found[("pair", q.trajectory_id, q.start_tick, 3., 12)] == dict(scd_normalized=.125+i)
        assert d4._reference_value(found, q.trajectory_id, q.start_tick, "scd_normalized:h=12:T=3") == .125+i
        j = next(k for k, q in enumerate(queries) if (q.kind, q.grid_size) == ("point", 16))
        assert d4._reference_value(found, queries[j].trajectory_id, queries[j].start_tick,
                                   f"tcd_normalized:T={queries[j].duration_cycles:g}") == .25+j
        cases = dict(
            queries=("queries differ", dict(queries=queries[:-1])),
            seed=("another condition/seed", dict(mutate=lambda rows: [(n, dict(r, seed=seed+1) if n == 3 else r) for n, r in rows])),
            condition=("another condition/seed", dict(mutate=lambda rows: [(n, dict(r, condition="identity") if n == 3 else r) for n, r in rows])),
            status=("another condition/seed", dict(mutate=lambda rows: [(n, dict(r, status="failed") if n == 3 else r) for n, r in rows])),
            duplicate=("another condition/seed", dict(mutate=lambda rows: rows+[(99999, rows[0][1])])),
            missing=("1 missing, 0 unregistered", dict(mutate=lambda rows: [(n, r) for n, r in rows if n != i])),
            unregistered=("0 missing, 1 unregistered", dict(mutate=lambda rows: rows+[(99999, dict(rows[0][1], query=dict(
                rows[0][1]["query"], start_tick=rows[0][1]["query"]["start_tick"]+10**6)))])))
        for name, (message, change) in cases.items():
            _write_reference(tmp_path/name/"evaluation"/"po_jepa_full", change.get("queries", queries), seed,
                             change.get("mutate", lambda rows: rows))
            with pytest.raises(ValueError, match=message):
                d4.reference_rows(tmp_path/name, "po_jepa_full")
        (tmp_path/"good"/"evaluation"/"po_jepa_full"/"FAILED.json").write_text("{}")
        with pytest.raises(ValueError, match="did not complete"):
            d4.reference_rows(tmp_path/"good", "po_jepa_full")
        (tmp_path/"good"/"evaluation"/"po_jepa_full"/"FAILED.json").unlink()
        (tmp_path/"good"/"evaluation"/"po_jepa_full"/"COMPLETE.json").unlink()
        with pytest.raises(ValueError, match="did not complete"):
            d4.reference_rows(tmp_path/"good", "po_jepa_full")
    finally:
        datasets._set_for_tests(old)


class RandomToy(torch.nn.Module):
    """每次调用都返回新的标准正态张量（固定种子），和输入、时长无关：JEPA 距离是伪度量，三角不等式照样成立。"""
    def __init__(self, seed):
        super().__init__()
        self.generator = torch.Generator().manual_seed(seed)

    def forward(self, raw, controls, duration, grid):
        return torch.randn(raw.shape, generator=self.generator)


def test_triangle_holds_on_random_predictions():
    """14 §3 S18：随机预测上三角不等式违反数为 0（5 个种子 × 2 个留出时长）；三项都不是 0，核对不是空的。"""
    for seed in range(5):
        values, _, triangle = d4.window_values(RandomToy(seed), toy_windows(), e06_config())
        assert triangle == {"T=3": True, "T=12": True}
        for t, _ in d4.HELDOUT:
            assert min(values[f"heldout_{k}:T={t:g}"] for k in ("direct", "partition", "rollout")) > .3


# ---------- 任务入口（玩具模型、玩具窗口，读写临时目录） ----------

@pytest.fixture
def toy_job(monkeypatch, tmp_path):
    """van_der_pol:40（测试轨迹 36–39，每条 5 个起点 = 20 个窗口）；读窗口、建模型、读检查点、读参照都换成玩具。"""
    old = datasets._set_for_tests(datasets.DatasetSpec("van_der_pol", 40))
    monkeypatch.setattr(evaluation, "load_test_window", lambda features, q, device: toy_window(q.duration_cycles, q.trajectory_id, q.start_tick))
    monkeypatch.setattr(d4, "load_long_test_window", lambda features, tid, t, device: toy_window(t, tid, 0))
    monkeypatch.setattr(comparison, "build_condition", lambda condition, device: (Toy("quadratic"), e06_config()))
    monkeypatch.setattr(v7load, "load_v7_selected", lambda job_dir, condition, model, **kw: dict(job_dir=str(job_dir), toy=True))
    reference = {}
    for tid, start in d4.windows_index():
        _, new, _ = d4.window_values(Toy("quadratic"), {t: toy_window(t, tid, start) for t in d4.TCD_TIMES}, e06_config())
        for name, value in new.items():
            metric, *rest = name.split(":")
            fields = dict(item.split("=") for item in rest)
            key = ("pair", tid, start, float(fields["T"]), int(fields["h"])) if metric == "scd_normalized" else ("point", tid, start, float(fields["T"]), 16)
            reference.setdefault(key, {})[metric] = value
    monkeypatch.setattr(d4, "reference_rows", lambda job_dir, condition: reference)
    job = tmp_path/"v7job"/"run"/"job"
    job.mkdir(parents=True)
    (job/"job.json").write_text("{}")
    yield tmp_path, reference
    datasets._set_for_tests(old)


def _run(tmp, name="out", condition="po_jepa_full"):
    return d4.run_d4(tmp/"features", tmp/name, torch.device("cpu"), condition, training_run=tmp/"v7job")


def test_run_d4_layout_and_byte_identical_rerun(toy_job):
    tmp, _ = toy_job
    report = _run(tmp, "a")
    _run(tmp, "b")
    files = sorted(p.relative_to(tmp/"a").as_posix() for p in (tmp/"a").rglob("*") if p.is_file())
    assert files == sorted(["COMPLETE.json", "job.json", "report.json", "runtime.json", "source.json",
                            *[f"rows/{i:05d}.json" for i in range(20)], *[f"long/{t:04d}.json" for t in range(36, 40)]])
    assert all((tmp/"a"/n).read_bytes() == (tmp/"b"/n).read_bytes() for n in files)
    assert cache._json_bytes(report) == (tmp/"a"/"report.json").read_bytes()
    assert cache.file_sha(tmp/"a"/"report.json") == json.loads((tmp/"a"/"COMPLETE.json").read_text())["report_sha256"]
    eq = report["equivalence"]["metrics"]
    assert {m: (s["count"], s["bitwise_equal"], s["outside_tolerance"]) for m, s in eq.items()} == dict(
        future_jepa_distance=(120, 120, 0), tcd_normalized=(120, 120, 0), scd_normalized=(240, 240, 0))
    assert report["equivalence"]["passed"] is True
    assert report["triangle"]["violations"] == 0 and report["status"] == "passed" and report["problems"] == []
    assert report["counts"] == dict(windows=20, long_trajectories=4, test_trajectories=4)
    assert set(report["trajectory_macro_means"]) == set(d4.value_names())
    assert report["trajectory_macro_means"]["pd_in:T=8:3+5"] == pytest.approx(QUADRATIC["pd_in:T=8:3+5"], rel=1e-5)
    assert report["long_means"]["long_pd_ln:T=24"] == pytest.approx(QUADRATIC_LONG["long_pd_ln:T=24"], rel=1e-5)
    row = json.loads((tmp/"a"/"rows"/"00007.json").read_text())
    assert (row["index"], row["trajectory_id"], row["start_tick"]) == (7, 37, d4.windows_index()[7][1])
    assert set(row["values"]) == set(d4.value_names()) and set(row["equivalence"]) == set(d4.equivalence_names())
    assert row["triangle_ok"] == {"T=3": True, "T=12": True} and row["manifest_sha256"] == "manifest-37"
    job = json.loads((tmp/"a"/"job.json").read_text())
    assert job["kind"] == "d4" and job["condition"] == "po_jepa_full" and job["job_dir"] == str((tmp/"v7job"/"run"/"job").resolve())
    assert json.loads((tmp/"a"/"source.json").read_text())["toy"] is True


def test_run_d4_macro_mean_is_trajectory_first():
    rows = [dict(trajectory_id=1, values=dict(x=0.)), dict(trajectory_id=1, values=dict(x=0.)), dict(trajectory_id=1, values=dict(x=3.)),
            dict(trajectory_id=2, values=dict(x=4.))]
    assert d4._macro(rows, ("x",)) == dict(x=2.5)                 # (1 + 4)/2，不是 7/4


def test_run_d4_equivalence_outside_tolerance_fails(toy_job):
    tmp, reference = toy_job
    reference[("pair", 37, d4.windows_index()[7][1], 3., 12)]["scd_normalized"] += 2e-6
    with pytest.raises(ValueError, match="equivalence outside tolerance"):
        _run(tmp)
    assert not (tmp/"out"/"COMPLETE.json").exists() and (tmp/"out"/"FAILED.json").exists()
    report = json.loads((tmp/"out"/"report.json").read_text())
    assert report["status"] == "failed" and report["equivalence"]["metrics"]["scd_normalized"]["outside_tolerance"] == 1
    assert report["equivalence"]["passed"] is False


def test_run_d4_within_tolerance_passes_and_counts_bitwise(toy_job):
    tmp, reference = toy_job
    key = ("point", 36, d4.windows_index()[0][1], 1., 16)
    reference[key]["future_jepa_distance"] = math.nextafter(reference[key]["future_jepa_distance"], math.inf)
    report = _run(tmp)
    s = report["equivalence"]["metrics"]["future_jepa_distance"]
    assert (s["count"], s["bitwise_equal"], s["outside_tolerance"]) == (120, 119, 0) and s["max_abs_diff"] > 0
    assert (tmp/"out"/"COMPLETE.json").exists()


def test_run_d4_missing_reference_row_fails(toy_job):
    tmp, reference = toy_job
    del reference[("point", 39, d4.windows_index()[-1][1], 12., 16)]
    with pytest.raises(ValueError, match="reference row missing"):
        _run(tmp)
    assert json.loads((tmp/"out"/"FAILED.json").read_text())["stage"] == "windows"


def test_run_d4_triangle_violation_fails(toy_job, monkeypatch):
    tmp, _ = toy_job
    real = d4._jepa
    calls = []
    def fake(a, b, config):
        calls.append(1)
        return 0. if len(calls) == 9 else real(a, b, config)      # 第一个窗口：6 个 future_jepa_distance 之后，第 7、8、9 次是 T=3 的 direct、partition、rollout
    monkeypatch.setattr(d4, "_jepa", fake)
    with pytest.raises(ValueError, match="triangle inequality violated in 1 cases"):
        _run(tmp)
    assert not (tmp/"out"/"COMPLETE.json").exists()
    report = json.loads((tmp/"out"/"report.json").read_text())
    assert report["triangle"]["cases"] == [dict(index=0, trajectory_id=36, start_tick=d4.windows_index()[0][1], heldout="T=3")]


def test_run_d4_refuses_unknown_condition_and_existing_output(toy_job):
    tmp, _ = toy_job
    with pytest.raises(ValueError, match="--condition"):
        _run(tmp, condition="po_jepa_tiny")
    assert not (tmp/"out").exists()
    (tmp/"exists").mkdir()
    with pytest.raises(FileExistsError):
        _run(tmp, "exists")


def test_run_d4_restores_modes_and_refuses_state_change(toy_job, monkeypatch):
    tmp, _ = toy_job
    model = Toy("quadratic")
    model.register_buffer("counter", torch.zeros(1))
    model.train()
    monkeypatch.setattr(comparison, "build_condition", lambda condition, device: (model, e06_config()))
    _run(tmp, "a")
    assert model.training is True
    real = model.forward
    def bump(*args):
        model.counter += 1
        return real(*args)
    monkeypatch.setattr(model, "forward", bump)
    with pytest.raises(RuntimeError, match="changed parameters"):
        _run(tmp, "b")


def test_nonfinite_values_refused():
    class Bad(Toy):
        def forward(self, raw, controls, duration, grid):
            out = super().forward(raw, controls, duration, grid)
            return out*float("inf") if float(duration[0]) == 4. else out
    with pytest.raises(ValueError, match="finite"):          # v7 的 ln_metrics / losses 先拒绝；_finite 是最后一道
        d4.window_values(Bad("quadratic"), toy_windows(), e06_config())
    for bad in (float("nan"), float("inf"), 1, None):
        with pytest.raises(ValueError, match="nonfinite D4 values"):
            d4._finite(dict(x=1., y=bad))
    assert d4._finite(dict(x=0.)) == dict(x=0.)
