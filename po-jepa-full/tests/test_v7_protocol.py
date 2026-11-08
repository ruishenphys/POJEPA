"""v7 协议覆盖层：步数日程、按数据集标定文件、v6 默认不受影响。"""
import pytest

from po_jepa import datasets
from po_jepa.runtime import checks, workflow


@pytest.fixture
def v7(monkeypatch, tmp_path):
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    old = datasets._set_for_tests(datasets.DatasetSpec("van_der_pol", 2000))
    monkeypatch.setattr(checks, "ROOT", checks.ROOT)   # 显式：标定文件写到临时目录
    monkeypatch.setattr(checks, "v7_calibration_path", lambda: tmp_path/"cal.yaml")
    yield tmp_path/"cal.yaml"
    datasets._set_for_tests(old)


def test_v6_default_unchanged(monkeypatch):
    monkeypatch.delenv(checks.PROTOCOL_ENV, raising=False)
    config = checks.load_protocol()
    assert config["training"]["updates_per_condition"] == 2000 and "v7_uncalibrated" not in config


@pytest.mark.parametrize("updates,interval,warm", [("2000", 200, 200), ("10000", 500, 1000)])
def test_v7_schedules(v7, monkeypatch, updates, interval, warm):
    monkeypatch.setenv(checks.UPDATES_ENV, updates)
    config = checks.load_protocol()
    assert config["training"]["updates_per_condition"] == int(updates)
    assert config["loss"]["lr_decay_updates"] == int(updates)
    assert config["validation"]["interval_updates"] == interval
    assert config["loss"]["consistency_warmup_updates"] == warm
    assert config["calibration"]["rho"] == 1.0 and config["v7_uncalibrated"]
    with pytest.raises(ValueError, match="还没标定"):
        workflow.require_calibrated_weights(swept=True)


def test_v7_calibration_file_applies(v7, monkeypatch):
    monkeypatch.setenv(checks.UPDATES_ENV, "10000")
    monkeypatch.setattr(workflow.checks, "v7_calibration_path", lambda: v7)
    workflow.write_v7_calibration(dict(sigma_z=3.5, time_weight=None, space_weight=None), "test")
    config = checks.load_protocol()
    assert config["model"]["graph"]["sigma_z"] == 3.5 and config.get("v7_uncalibrated")
    workflow.write_v7_calibration(dict(time_weight=0.12, space_weight=8.0, rho=1.0), "test")
    config = checks.load_protocol()
    assert config["loss"]["time_weight"] == 0.12 and config["loss"]["space_weight"] == 8.0
    assert config["model"]["graph"]["sigma_z"] == 3.5 and "v7_uncalibrated" not in config
    workflow.require_calibrated_weights(swept=True)


def test_v7_gpu_whitelist(monkeypatch):
    record = dict(name="NVIDIA H100 80GB HBM3", mig_mode="Disabled", uuid="GPU-x",
                  memory_total_bytes=85*10**9, smi_total_mib=81559.)
    monkeypatch.delenv(checks.PROTOCOL_ENV, raising=False)
    with pytest.raises(ValueError): checks.validate_target_device(record)
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    assert checks.validate_target_device(record) == 80
