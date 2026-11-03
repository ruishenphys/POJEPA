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
    assert datasets.donor_id(1999) == 1800
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
