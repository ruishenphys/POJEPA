"""S08a：一致性监控和提前检查推广到单项消融等新条件，po_jepa_full 的输出与 v7 逐字节相同（07 §2.4）。"""
import json

import pytest

from po_jepa.losses import LossConfig
from po_jepa.runtime import checks, comparison

GOLDEN = {  # 911bc09（0ac45fe 相同）树上实算（S08a 段说明 §7 A14 附复核命令）
    "ok": '{"measurements": 3, "rho": 0.05, "threshold": 0.005, "median_share": {"time": 0.05, "space": 0.05}, "active": true}',
    "off": '{"measurements": 3, "rho": 0.05, "threshold": 0.005, "median_share": {"time": 1e-05, "space": 1e-05}, "active": false}',
    "empty": '{"measurements": 0, "rho": 0.05, "active": false, "median_share": null, "threshold": 0.005}',
}
GOLDEN_EARLY = ('{\n  "active": true,\n  "completed_updates": 4,\n  "measurements": 3,\n  "median_share": {\n'
                '    "space": 0.05,\n    "time": 0.05\n  },\n  "rho": 0.05,\n  "threshold": 0.005\n}\n')
V7_SHARES = {"ok": [1e-6, None, .04, .06, .05], "off": [.05, None, 1e-5, 2e-5, 1e-5], "empty": []}
NEW_CONDITIONS = {"po_jepa_time_only": ((.5, 0.), ("time",)), "po_jepa_space_only": ((0., .25), ("space",)),
                  "transformer_consistency": ((.5, .25), ("time", "space")), "fixed_fourier": ((.5, .25), ("time", "space")),
                  "koopman": ((0., 0.), None)}
NEW_SHARES = {"time-ok": [None, (.04, 0.), (.06, 0.), (.05, 0.)], "space-ok": [None, (0., .04), (0., .06), (0., .05)],
              "both-ok": [None, (.04, .03), (.06, .03), (.05, .03)], "empty": []}
NEW_MEDIANS = {"time-ok": dict(time=.05, space=0.), "space-ok": dict(time=0., space=.05), "both-ok": dict(time=.05, space=.03)}
SINGLE = {"po_jepa_time_only": ((.5, 0.), "time", (1e-5, .05), (.05, 0.)),
          "po_jepa_space_only": ((0., .25), "space", (.05, 1e-5), (0., .05))}


@pytest.fixture
def protocol(monkeypatch):
    real = checks.load_protocol()
    monkeypatch.setattr(comparison.check, "load_protocol", lambda: {**real, "calibration": dict(rho=.05),
        "monitoring": dict(consistency_interval=1, early_check_updates=4, min_share=.005)})


def write(folder, shares):
    """shares：每步一个 (time, space) 或 None（这一步没测）。"""
    (folder/"updates").mkdir(parents=True)
    for step, share in enumerate(shares, 1):
        row = dict(completed_updates=step,
                   consistency_gradient=None if share is None else dict(share=dict(time=share[0], space=share[1])))
        (folder/"updates"/f"{step:04d}.json").write_text(json.dumps(row), encoding="utf-8")


def config(time, space):
    return LossConfig(1., 1e-5, 1., 1., time, space, consistency_warmup_updates=2)


@pytest.mark.parametrize("weights", [(0., 0.), (.5, .25), (.5, 0.)])
def test_full_output_matches_v7_bytes(protocol, tmp_path, weights):
    """po_jepa_full 的汇总和 911bc09（0ac45fe 相同）上实算的逐字节相同（含键序），和权重无关。"""
    for name, shares in V7_SHARES.items():
        write(tmp_path/name, [None if share is None else (share, share) for share in shares])
        summary = comparison.consistency_monitor_summary(tmp_path/name, "po_jepa_full", config(*weights))
        assert json.dumps(summary) == GOLDEN[name]


@pytest.mark.parametrize("weights", [(0., 0.), (.5, .25)])
@pytest.mark.parametrize("condition", ["po_jepa_no_consistency", "matched_transformer", "vjepa2_ac", "vjepa2_ac_small", "identity"])
def test_other_v7_conditions_stay_unmonitored(protocol, tmp_path, condition, weights):
    """其余 v7 名字和 identity（拿到的是协议的正权重）不监控、不做提前检查。"""
    write(tmp_path, [(.05, .05)]*4)
    loss = config(*weights)
    assert comparison.monitored_consistency_terms(condition, loss) is None
    assert comparison.consistency_monitor_summary(tmp_path, condition, loss) is None
    comparison.early_check(tmp_path, condition, loss, 4)
    assert not (tmp_path/"early-check.json").exists()


@pytest.mark.parametrize("condition", list(NEW_CONDITIONS))
def test_new_conditions_check_only_positive_terms(protocol, tmp_path, condition):
    """新条件只看权重 > 0 的项：median_share 只含这些项，active 只看这些项，另加 checked_terms；koopman 不监控。"""
    weights, terms = NEW_CONDITIONS[condition]
    loss = config(*weights)
    assert comparison.monitored_consistency_terms(condition, loss) == terms
    for name, shares in NEW_SHARES.items():
        write(tmp_path/name, shares)
        summary = comparison.consistency_monitor_summary(tmp_path/name, condition, loss)
        if terms is None:
            assert summary is None
            continue
        if name == "empty":
            assert summary == dict(measurements=0, rho=.05, active=False, median_share=None, threshold=.005,
                                   checked_terms=list(terms))
            assert list(summary) == ["measurements", "rho", "active", "median_share", "threshold", "checked_terms"]
            continue
        assert list(summary) == ["measurements", "rho", "threshold", "median_share", "active", "checked_terms"]
        assert summary["measurements"] == 3 and summary["rho"] == .05 and summary["threshold"] == .005
        assert summary["checked_terms"] == list(terms)
        median = {term: NEW_MEDIANS[name][term] for term in terms}
        assert summary["median_share"] == median and list(summary["median_share"]) == list(terms)
        assert summary["active"] is all(value >= .005 for value in median.values())


def test_early_check_full_file_matches_v7(protocol, tmp_path):
    """po_jepa_full 写出的 early-check.json 和 911bc09（0ac45fe 相同）上实算的逐字节相同。"""
    write(tmp_path, [(.05, .05)]*4)
    comparison.early_check(tmp_path, "po_jepa_full", config(0., 0.), 4)
    assert (tmp_path/"early-check.json").read_text(encoding="utf-8") == GOLDEN_EARLY


@pytest.mark.parametrize("condition", list(SINGLE))
def test_early_check_single_term(protocol, tmp_path, condition):
    """单项消融只按被检查的那一项判断：另一项的值再大也不算。"""
    weights, term, weak, strong = SINGLE[condition]
    loss = config(*weights)
    write(tmp_path/"weak", [weak]*4)
    write(tmp_path/"strong", [strong]*4)
    comparison.early_check(tmp_path/"weak", condition, loss, 3)          # 不是检查步，不动作
    assert not (tmp_path/"weak/early-check.json").exists()
    with pytest.raises(comparison.ConsistencyTooWeak, match="一致性太弱"):
        comparison.early_check(tmp_path/"weak", condition, loss, 4)
    record = json.loads((tmp_path/"weak/early-check.json").read_text(encoding="utf-8"))
    assert record["active"] is False and record["checked_terms"] == [term] and list(record["median_share"]) == [term]
    comparison.early_check(tmp_path/"strong", condition, loss, 4)
    record = json.loads((tmp_path/"strong/early-check.json").read_text(encoding="utf-8"))
    assert record["active"] is True and record["checked_terms"] == [term]


def test_new_condition_requires_loss_config(protocol):
    """新条件必须传 LossConfig；po_jepa_full 不读 loss_config。"""
    with pytest.raises(TypeError, match="^loss_config must be a LossConfig$"):
        comparison.monitored_consistency_terms("po_jepa_time_only", dict(time_weight=.5, space_weight=0.))
    assert comparison.monitored_consistency_terms("po_jepa_full", None) == ("time", "space")
