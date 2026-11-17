"""v7：数据集规模与划分集中配置；v6 默认值不变。"""
import pytest

from po_jepa import datasets, simulation


@pytest.fixture
def spec2000():
    old = datasets._set_for_tests(datasets.DatasetSpec("double_well", 2000))
    yield datasets.active()
    datasets._set_for_tests(old)


def test_default_is_v6():
    spec = datasets.active()
    assert spec.is_v6_default and (spec.train, spec.validation, spec.test) == (160, 20, 20)
    assert datasets.validation_ids() == list(range(160, 180))
    assert datasets.test_start_ticks(180) == (0, 5, 10, 15, 20)
    assert datasets.donor_id(199) == 180


def test_2000_split_and_subsets(spec2000):
    assert (spec2000.train, spec2000.validation, spec2000.test) == (1600, 200, 200)
    assert simulation.split_for(1599) == "train" and simulation.split_for(1600) == "validation"
    assert simulation.split_for(1999) == "test"
    ids = datasets.validation_ids()
    assert len(ids) == 20 and ids[0] == 1600 and all(1600 <= i < 1800 for i in ids)
    counts = {}
    for tid in spec2000.split_range("test"):
        ticks = datasets.test_start_ticks(tid)
        assert len(ticks) == 2
        for t in ticks: counts[t] = counts.get(t, 0)+1
    assert set(counts.values()) == {80}
    assert datasets.donor_id(1999) == 1804   # 下一条起点相同的轨迹（+5 循环），不是紧邻的 1800
    with pytest.raises(ValueError): simulation.split_for(2000)


def test_double_well_first_200_identical_to_v6(spec2000):
    """扩到 2000 条时前 200 条的初值与控制与 v6 逐位相同（只是划分不同）。"""
    a = simulation.initial_state(123); c = simulation.sample_controls(123)
    old = datasets._set_for_tests(datasets.DatasetSpec("double_well", 200))
    try:
        b = simulation.initial_state(123); d = simulation.sample_controls(123)
    finally:
        datasets._set_for_tests(old)
    assert (a == b).all() and (c.values == d.values).all() and (c.starts_ticks == d.starts_ticks).all()


def test_parse_env():
    assert datasets.parse("van_der_pol:2000") == datasets.DatasetSpec("van_der_pol", 2000)
    for bad in ("double_well", "x:200", "double_well:123"):
        with pytest.raises(ValueError): datasets.parse(bad)


def test_evaluation_donor_shares_registered_start_ticks(spec2000):
    """评测 donor 必须登记了和 recipient 相同的起点，否则 donor 窗口会被 EvaluationQuery 拒绝（v7.1 机哥本机补丁）。
    evaluation.donor_id 与 datasets.donor_id（state_readout、probe_actions 用的）是同一个映射。"""
    from po_jepa.runtime import evaluation
    for tid in spec2000.split_range("test"):
        donor = evaluation.donor_id(tid)
        assert donor != tid and donor in spec2000.split_range("test") and donor == datasets.donor_id(tid)
        assert datasets.test_start_ticks(donor) == datasets.test_start_ticks(tid)
    assert evaluation.donor_id(1800) == 1805 and evaluation.donor_id(1999) == 1804
    assert len({evaluation.donor_id(t) for t in spec2000.split_range("test")}) == 200   # 一一映射


def test_evaluation_donor_registered_window_loads(spec2000):
    """donor 窗口的起点必须通过 EvaluationQuery 的登记检查（v7 第一次正式运行就是在这里失败的）。"""
    from dataclasses import replace
    from po_jepa.runtime import evaluation
    for q in evaluation.build_index():
        replace(q, trajectory_id=evaluation.donor_id(q.trajectory_id))


def test_evaluation_donor_unchanged_for_v6():
    from po_jepa.runtime import evaluation
    assert evaluation.donor_id(199) == 180 and evaluation.donor_id(180) == 181
