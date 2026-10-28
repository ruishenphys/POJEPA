"""B04：虚构控制命令的手算验收；这里没有课题仿真或生产积分器。"""

from dataclasses import replace
import importlib
import importlib.util
import math

import pytest
import torch


def actions():
    """让首次RED清楚指出缺少动作模块，而不是掩盖为其他导入故障。"""
    assert importlib.util.find_spec("po_jepa.actions") is not None, "B04 actions 尚未实现"
    return importlib.import_module("po_jepa.actions")


def t(values):
    """测试时间一律CPU float64，单位由调用处区分物理秒/内部周期。"""
    return torch.tensor(values, dtype=torch.float64)


def fixture(*, device="cpu", gradient=False):
    """虚构单通道命令：周期0.25秒，物理幅值经(u-1)/2变换。"""
    a = actions()
    spec = a.ControlSpec(1, .25, ("fictional_velocity",), ("fictional_m/s",),
                         (-9.,), (9.,), (2.,), (1.,), "fictional B04 test only")
    values = torch.tensor([[3., 5., float("nan"), float("nan")],
                           [-1., 1., 7., float("nan")]], device=device).unsqueeze(-1)
    values.requires_grad_(gradient)
    args = dict(starts_seconds=t([[10., 10.5, math.nan, math.nan], [20., 20.25, 20.75, math.nan]]),
                ends_seconds=t([[10.5, 10.75, math.nan, math.nan], [20.25, 20.75, 21., math.nan]]),
                u_values=values, n_segments=torch.tensor([2, 3]),
                segment_valid=torch.tensor([[True, True, False, False], [True, True, True, False]]),
                origin_seconds=t([10., 20.]), control_spec=spec, source_ids=("fictional-A", "fictional-B"))
    return args, values


def test_clip_split_concat_preserves_batched_control_path():
    a = actions()
    args, _ = fixture()
    before = {k: v.clone() for k, v in args.items() if isinstance(v, torch.Tensor)}
    path = a.from_physical_segments(**args)
    assert torch.equal(path.duration, t([3., 4.]))
    assert torch.equal(path.starts[path.segment_valid], t([0., 2., 0., 1., 3.]))
    assert torch.equal(path.u_values[path.segment_valid].flatten(), torch.tensor([1., 2., -1., 0., 3.]))
    pair = a.split_action_segments(path, tau=t([.75, 1.]), sigma=t([1.5, 2.5]))
    assert isinstance(pair, a.ActionSlicePair)
    assert torch.equal(pair.first.duration, t([.75, 1.]))
    assert torch.equal(pair.second.duration, t([1.5, 2.5]))
    assert torch.equal(pair.second.source_origin_seconds, t([10.1875, 20.25]))
    joined = a.concatenate_action_segments(pair.first, pair.second)
    assert torch.equal(joined.duration, t([2.25, 3.5]))  # 只恢复查询前缀。
    assert joined.source_ids == path.source_ids and joined.control_spec == path.control_spec
    for times, expected in [([0., 0.], [1., -1.]), ([.75, 1.], [1., 0.]),
                            ([2., 3.], [2., 3.]), ([2.249, 3.499], [2., 3.])]:
        actual = a.action_at(joined, time=t(times), active=torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(actual[:, 0], torch.tensor(expected), atol=1e-5, rtol=1e-4)
    nested = a.clip_action_segments(pair.second, start=t([.25, .5]), end=t([1., 2.]))
    direct = a.clip_action_segments(path, start=t([1., 1.5]), end=t([1.75, 3.]))
    assert torch.equal(nested.source_origin_seconds, direct.source_origin_seconds)
    assert torch.equal(nested.duration, direct.duration)
    for key, expected in before.items():
        torch.testing.assert_close(args[key], expected, rtol=0, atol=0, equal_nan=True)


def test_action_at_is_right_continuous_per_sample():
    a = actions()
    args, _ = fixture()
    path = a.from_physical_segments(**args)
    boundary = t([2., 1.])
    just_before = torch.nextafter(boundary, t([0., 0.]))
    active = torch.ones(2, dtype=torch.bool)
    assert torch.equal(a.action_at(path, time=just_before, active=active)[:, 0], torch.tensor([1., -1.]))
    assert torch.equal(a.action_at(path, time=boundary, active=active)[:, 0], torch.tensor([2., 0.]))
    with pytest.raises(ValueError):
        a.action_at(path, time=path.duration, active=active)
    assert torch.equal(a.action_at(path, time=path.duration, active=~active), torch.zeros(2, 1))


def exact_endpoint(path):
    """独立常值精确解：旋转加输入；使用返回的有效段和还原物理单位。"""
    c = torch.zeros(2, dtype=torch.float64)
    spec = path.control_spec
    for j in range(int(path.n_segments[0])):
        seconds = float(path.ends[0, j] - path.starts[0, j]) * spec.control_period_seconds
        u = float(path.u_values[0, j, 0]) * spec.scale[0] + spec.offset[0]
        angle = math.pi/2 * seconds
        co, si = math.cos(angle), math.sin(angle)
        rotation = t([[co, -si], [si, co]])
        c = rotation @ c + t([si, 1-co]) * u
    return c


def test_ordered_controls_change_analytic_endpoint():
    a = actions()
    spec = a.ControlSpec(1, 1., ("test_u",), ("dimensionless",), (0.,), (1.,), (1.,), (0.,), "fictional rotation oracle")
    def path(values, starts, ends):
        return a.from_physical_segments(starts_seconds=t([starts]), ends_seconds=t([ends]),
            u_values=torch.tensor([values], dtype=torch.float32).unsqueeze(-1),
            n_segments=torch.tensor([len(values)]), segment_valid=torch.ones(1, len(values), dtype=torch.bool),
            origin_seconds=t([0.]), control_spec=spec, source_ids=("fictional-oracle",))
    first = path([1., 0.], [0., 1.], [1., 2.])
    reverse = path([0., 1.], [0., 1.], [1., 2.])
    average = path([.5], [0.], [2.])
    for actual, expected in [(first, [-1., 1.]), (reverse, [1., 1.]), (average, [0., 1.])]:
        torch.testing.assert_close(exact_endpoint(actual), t(expected), atol=1e-10, rtol=1e-8)
    pair = a.split_action_segments(first, tau=t([.5]), sigma=t([1.5]))
    torch.testing.assert_close(exact_endpoint(a.concatenate_action_segments(pair.first, pair.second)),
                               t([-1., 1.]), atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_physical_scaling_selected_and_repeated_values_keep_gradient_without_padding(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; CPU is mandatory")
    a = actions()
    args, physical = fixture(device=device, gradient=True)
    path = a.from_physical_segments(**args)
    selected = a.action_at(path, time=t([.5, 3.2]), active=torch.ones(2, dtype=torch.bool))
    selected.sum().backward(retain_graph=True)
    expected = torch.zeros_like(physical)
    expected[0, 0, 0] = .5
    expected[1, 2, 0] = .5
    torch.testing.assert_close(physical.grad, expected, atol=1e-5, rtol=1e-4)
    physical.grad.zero_()
    pair = a.split_action_segments(path, tau=t([.75, 1.]), sigma=t([1.5, 2.5]))
    joined = a.concatenate_action_segments(pair.first, pair.second)
    joined.u_values[joined.segment_valid.to(device)].sum().backward()
    expected.zero_()
    expected[0, :2, 0] = torch.tensor([1., .5], device=device)  # 第一命令在两支出现，梯度相加。
    expected[1, :3, 0] = .5
    torch.testing.assert_close(physical.grad, expected, atol=1e-5, rtol=1e-4)
    assert selected.device == physical.device and joined.u_values.device == physical.device
    assert torch.isfinite(physical.grad).all() and torch.all(physical.grad[~args['segment_valid'].to(device)] == 0)


def test_empty_diagnostics_can_clip_and_concatenate_but_never_supply_active_control():
    a = actions()
    args, _ = fixture()
    path = a.from_physical_segments(**args)
    empty = a.clip_action_segments(path, start=path.duration, end=path.duration)
    assert torch.equal(empty.n_segments, torch.zeros(2, dtype=torch.int64))
    assert empty.starts.shape[1] >= 1 and not empty.segment_valid.any()
    assert torch.equal(empty.duration, t([0., 0.]))
    a.validate_action_segments(empty, query_duration=t([0., 0.]), require_exact_end=True)
    joined = a.concatenate_action_segments(path, empty)
    assert torch.equal(joined.duration, path.duration)
    with pytest.raises(ValueError):
        a.action_at(empty, time=t([0., 0.]), active=torch.tensor([True, False]))
    with pytest.raises(ValueError):
        a.validate_action_segments(empty, query_duration=t([1e-15, 0.]))
    with pytest.raises(ValueError):
        a.split_action_segments(path, tau=t([0., 1.]), sigma=t([1., 1.]))


@pytest.mark.parametrize("bad", ["gap", "overlap", "negative", "nan", "inf", "range", "mask", "count", "time_dtype", "time_gradient"])
def test_invalid_padding_gaps_overlap_and_units_are_rejected(bad):
    a = actions()
    args, _ = fixture()
    if bad == "gap": args['starts_seconds'][0, 1] += .01
    if bad == "overlap": args['starts_seconds'][0, 1] -= .01
    if bad == "negative": args['ends_seconds'][0, 0] = 9.
    if bad == "nan": args['u_values'][0, 0] = math.nan
    if bad == "inf": args['ends_seconds'][0, 0] = math.inf
    if bad == "range": args['u_values'][0, 0] = 10.
    if bad == "mask": args['segment_valid'][0, 2] = True
    if bad == "count": args['n_segments'][0] = 5
    if bad == "time_dtype": args['starts_seconds'] = args['starts_seconds'].float()
    if bad == "time_gradient": args['starts_seconds'].requires_grad_()
    with pytest.raises((TypeError, ValueError)):
        a.from_physical_segments(**args)


def test_incompatible_origin_source_units_and_exact_end_are_rejected():
    a = actions()
    args, _ = fixture()
    path = a.from_physical_segments(**args)
    pair = a.split_action_segments(path, tau=t([1., 1.]), sigma=t([1., 1.]))
    for second in (replace(pair.second, source_origin_seconds=pair.second.source_origin_seconds+.01),
                   replace(pair.second, source_ids=("other", "fictional-B")),
                   replace(pair.second, control_spec=replace(path.control_spec, channel_units=("other_unit",)))):
        with pytest.raises(ValueError): a.concatenate_action_segments(pair.first, second)
    a.validate_action_segments(path, query_duration=t([1., 1.]))
    with pytest.raises(ValueError): a.validate_action_segments(path, query_duration=t([1., 1.]), require_exact_end=True)
    with pytest.raises(TypeError): a.validate_action_segments(path, query_duration=None)
    with pytest.raises(TypeError): a.split_action_segments(replace(path, duration=None), tau=t([1., 1.]), sigma=t([1., 1.]))
    for changes in ({'action_dim': 0}, {'scale': (0.,)}, {'control_period_seconds': math.inf},
                    {'channel_names': ()}, {'definition_source': ''}, {'lower_bounds': (10.,)}):
        with pytest.raises((TypeError, ValueError)): replace(path.control_spec, **changes)


def test_schedule_is_per_sample_never_crosses_boundary_or_reads_action_values():
    a = actions()
    args, _ = fixture()
    path = a.from_physical_segments(**args)
    # meta设备没有数据可读；时间调度只能访问形状和CPU元数据。
    metadata_only = replace(path, u_values=torch.empty(path.u_values.shape, device="meta"))
    time = t([0., 0.]); query = t([2.25, 3.5])
    boundaries = [[2., 3.], [1., 3., 4.]]
    counts = [0, 0]
    for _ in range(20):
        before = time.clone()
        schedule = a.next_step_schedule(metadata_only, time=time, query_duration=query, max_step=.6)
        assert torch.equal(schedule.time, before) and torch.equal(time, before)
        for b in range(2):
            if time[b] == query[b]:
                assert not schedule.active[b] and schedule.delta[b] == 0 and schedule.segment_index[b] == -1
            else:
                j = next(j for j, end in enumerate(boundaries[b]) if float(time[b]) < end)
                delta = min(.6, float(query[b]-time[b]), boundaries[b][j]-float(time[b]))
                assert schedule.active[b] and schedule.segment_index[b] == j
                assert float(schedule.delta[b]) == delta and delta > 0
                counts[b] += 1
        if not schedule.active.any(): break
        time = time + schedule.delta
    else: pytest.fail("schedule did not terminate")
    assert torch.equal(time, query) and counts[0] != counts[1]
    boundary = t([2., 1.]); near = torch.nextafter(boundary, t([0., 0.]))
    tiny = a.next_step_schedule(path, time=near, query_duration=query, max_step=.6)
    assert torch.all(tiny.delta > 0) and torch.equal(near+tiny.delta, boundary)
    at = a.next_step_schedule(path, time=boundary, query_duration=query, max_step=.6)
    assert torch.equal(at.segment_index, torch.tensor([1, 1]))
    short = a.next_step_schedule(path, time=t([0., 0.]), query_duration=t([1., 1.]), max_step=.25)
    long = a.next_step_schedule(path, time=t([0., 0.]), query_duration=path.duration, max_step=.25)
    assert torch.equal(short.delta, long.delta) and torch.equal(short.segment_index, long.segment_index)
    for step in (0., -1., math.nan, math.inf):
        with pytest.raises(ValueError): a.next_step_schedule(path, time=t([0., 0.]), query_duration=query, max_step=step)
    with pytest.raises(ValueError): a.next_step_schedule(path, time=t([3., 0.]), query_duration=query, max_step=.25)


def test_roundoff_alignment_is_local_does_not_swallow_positive_intervals_or_mutate():
    a = actions()
    args, _ = fixture()
    path = a.from_physical_segments(**args)
    altered_starts = path.starts.clone(); altered_starts[0, 1] += 2e-13
    noisy = replace(path, starts=altered_starts)
    snapshot = altered_starts.clone()
    a.validate_action_segments(noisy, query_duration=path.duration, require_exact_end=True)
    assert torch.equal(altered_starts, snapshot)
    assert a.action_at(noisy, time=t([2., 1.]), active=torch.ones(2, dtype=torch.bool))[0, 0] == 2.
    altered_starts[0, 1] += 1e-7
    with pytest.raises(ValueError): a.validate_action_segments(noisy, query_duration=path.duration)
    # 极大的绝对秒原点不能把内部0.01周期的缺口当成舍入误差。
    args, _ = fixture(); args['origin_seconds'] += 1e9
    args['starts_seconds'] += 1e9; args['ends_seconds'] += 1e9
    args['starts_seconds'][0, 1] += .01
    with pytest.raises(ValueError): a.from_physical_segments(**args)
    tiny = a.clip_action_segments(path, start=t([1., 1.]), end=torch.nextafter(t([1., 1.]), t([2., 2.])))
    assert torch.all(tiny.duration > 0) and torch.equal(tiny.n_segments, torch.ones(2, dtype=torch.int64))


@pytest.mark.parametrize("operation", ["validate", "clip", "split", "concat", "action", "schedule"])
def test_public_boundaries_reject_wrong_container_types(operation):
    a = actions()
    calls = {
        "validate": lambda: a.validate_action_segments(None, query_duration=t([1.])),
        "clip": lambda: a.clip_action_segments(None, start=t([0.]), end=t([1.])),
        "split": lambda: a.split_action_segments(None, tau=t([1.]), sigma=t([1.])),
        "concat": lambda: a.concatenate_action_segments(None, None),
        "action": lambda: a.action_at(None, time=t([0.]), active=torch.tensor([True])),
        "schedule": lambda: a.next_step_schedule(None, time=t([0.]), query_duration=t([1.]), max_step=.25),
    }
    with pytest.raises(TypeError):
        calls[operation]()


@pytest.mark.parametrize("sigma", [1e-18, 1e-16])
@pytest.mark.parametrize("batch_size,bad_sample", [(1, 0), (2, 0), (2, 1)])
def test_split_rejects_positive_second_duration_lost_in_endpoint_addition(sigma, batch_size, bad_sample):
    """单样本或虚构批次任一样本的正sigma加不到tau上，必须报错而非返回空片段。"""
    a = actions()
    args, _ = fixture()
    for name in ("starts_seconds", "ends_seconds", "u_values", "n_segments", "segment_valid", "origin_seconds"):
        args[name] = args[name][:batch_size]
    args["source_ids"] = args["source_ids"][:batch_size]
    path = a.from_physical_segments(**args)
    tau = t([1.]*batch_size); durations = tau.clone()
    durations[bad_sample] = sigma
    assert durations[bad_sample] > 0 and (tau+durations)[bad_sample] == tau[bad_sample]
    with pytest.raises(ValueError, match="precision"):
        a.split_action_segments(path, tau=tau, sigma=durations)


def test_split_preserves_representable_positive_duration_smaller_than_alignment_tolerance():
    """一ULP的正区间仍可表示，虽小于1e-12，也不能被容差门槛一概拒绝。"""
    a = actions()
    args, _ = fixture()
    path = a.from_physical_segments(**args)
    tau = t([1., 1.])
    sigma = torch.nextafter(tau, t([2., 2.]))-tau
    assert torch.all((sigma > 0) & (sigma < 1e-12))
    pair = a.split_action_segments(path, tau=tau, sigma=sigma)
    assert torch.equal(pair.second.duration, sigma)
    assert torch.equal(pair.second.n_segments, torch.ones(2, dtype=torch.int64))
    a.validate_action_segments(pair.second, query_duration=sigma, require_exact_end=True)
    assert torch.equal(a.action_at(pair.second, time=t([0., 0.]), active=torch.ones(2, dtype=torch.bool))[:, 0],
                       torch.tensor([1., 0.]))


def d12_raw_path(starts, ends, *, gradient=False):
    """D12虚构原始时间元数据；绕过物理构造对齐，以便暴露真实微小缺口。"""
    a = actions()
    n = len(starts)
    values = torch.arange(1., n+1).reshape(1, n, 1).requires_grad_(gradient)
    spec = a.ControlSpec(1, .25, ("fictional_velocity",), ("unit",),
        (-100.,), (100.,), (1.,), (0.,), "D12 fictional exact split regression")
    return a.ActionSegments(t([starts]), t([ends]), values, torch.tensor([n]),
        torch.ones(1, n, dtype=torch.bool), t([2.]), t([ends[-1]]), spec, ("d12-fictional",))


@pytest.mark.parametrize("tau,sigma", [(.17, .29), (.01, .07)])
def test_d12_split_keeps_authoritative_decimal_durations(tau, sigma):
    """原float64局部时长权威，不能以两次舍入相减替代sigma。"""
    path = d12_raw_path([0.], [1.])
    pair = actions().split_action_segments(path, tau=t([tau]), sigma=t([sigma]))
    for part, duration in ((pair.first, tau), (pair.second, sigma)):
        assert torch.equal(part.duration, t([duration]))
        assert part.starts[0, 0] == 0 and part.ends[0, -1] == duration
    assert torch.equal(pair.second.source_origin_seconds, t([2.+tau*.25]))


def test_d12_split_retains_action_after_rounded_total_and_value_gradients():
    """舍入终点恰为切换点时，保留之后真实正尾段，不能拉长旧动作。"""
    from fractions import Fraction

    tau, sigma = .17, .29
    switch = tau+sigma
    path = d12_raw_path([0., switch], [switch, 1.], gradient=True)
    pair = actions().split_action_segments(path, tau=t([tau]), sigma=t([sigma]))
    second = pair.second
    boundary = float(Fraction.from_float(switch)-Fraction.from_float(tau))
    assert torch.equal(second.n_segments, torch.tensor([2]))
    assert torch.equal(second.starts, t([[0., boundary]]))
    assert torch.equal(second.ends, t([[boundary, sigma]]))
    assert second.ends[0, 1] > second.starts[0, 1]
    assert torch.equal(second.u_values, path.u_values)
    assert second.source_ids == path.source_ids and second.control_spec == path.control_spec
    (pair.first.u_values.sum()+second.u_values.sum()).backward()
    assert torch.equal(path.u_values.grad, torch.tensor([[[2.], [1.]]]))


def test_d12_split_rejects_exact_shortfall_hidden_by_float_addition():
    """.01+.07精确有理和大于float(.08)，源路径真欠覆盖仍须拒绝。"""
    from fractions import Fraction

    assert Fraction.from_float(.01)+Fraction.from_float(.07) > Fraction.from_float(.08)
    assert (.08-.01) == .07
    with pytest.raises(ValueError, match="cover"):
        actions().split_action_segments(d12_raw_path([0.], [.08]), tau=t([.01]), sigma=t([.07]))


def test_d12_former_decimal_oracle_split_exceeds_exact_source_coverage():
    """旧.4/1.6输入的实际有理和超过2；保留该反例而非放宽覆盖或容差。"""
    from fractions import Fraction

    assert Fraction.from_float(.4)+Fraction.from_float(1.6) > Fraction(2)
    with pytest.raises(ValueError, match="cover"):
        actions().split_action_segments(d12_raw_path([0., 1.], [1., 2.]),
            tau=t([.4]), sigma=t([1.6]))


def test_d12_split_rejects_positive_tail_that_collapses_on_local_float64_axis():
    """真实新动作尾段不足局部float64分辨率时明确拒绝，不静默掉段。"""
    with pytest.raises(ValueError, match="precision"):
        actions().split_action_segments(d12_raw_path([0., .08], [.08, 1.]),
            tau=t([.01]), sigma=t([.07]))


@pytest.mark.parametrize("shift", [-1e-14, 1e-14])
def test_d12_split_rejects_raw_micro_gap_or_overlap(shift):
    """先核对原始有效区间，不让既有宽容对齐掩盖真实gap或overlap。"""
    with pytest.raises(ValueError, match="cover|gap|overlap"):
        actions().split_action_segments(d12_raw_path([0., .2+shift], [.2, 1.]),
            tau=t([.17]), sigma=t([.29]))
