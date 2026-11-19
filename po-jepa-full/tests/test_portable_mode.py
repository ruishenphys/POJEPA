"""C8（差距清单 §3）：可移植模式 PO_JEPA_PORTABLE=1 放宽显卡白名单、环境版本和续训同卡三处；默认仍是原来的严格门槛。"""
import pytest
import torch

from po_jepa.runtime import cache, checks, environment, resume


def _record(name="NVIDIA L4", gb=24., mig="Disabled"):
    return dict(name=name, mig_mode=mig, uuid="GPU-abc", memory_total_bytes=int(gb*10**9)+1, smi_total_mib=gb*10**9/1024**2+1)


@pytest.mark.parametrize("value,expected", [(None, False), ("", False), ("0", False), ("1", True)])
def test_portable_switch(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(environment.PORTABLE_ENV, value)
    assert environment.portable() is expected


@pytest.mark.parametrize("value", ["yes", "true", "2", " 1"])
def test_portable_switch_rejects_other_values(monkeypatch, value):
    monkeypatch.setenv(environment.PORTABLE_ENV, value)
    with pytest.raises(ValueError, match="PO_JEPA_PORTABLE"):
        environment.portable()


# ---------------------------------------------------------------- ① 显卡白名单

def test_unlisted_gpu_strict_by_default_and_accepted_when_portable(monkeypatch):
    record = _record()
    with pytest.raises(ValueError, match="F03 target requires.*PO_JEPA_PORTABLE=1"):
        checks.validate_target_device(dict(record))
    monkeypatch.setenv(environment.PORTABLE_ENV, "1")
    seen = dict(record)
    assert checks.validate_target_device(seen) == checks.PORTABLE_MIN_GB == 24
    assert seen == dict(record, portable_unlisted_gpu=True)


@pytest.mark.parametrize("record,match", [
    (_record(mig="Enabled"), "MIG disabled"),
    (_record(gb=16.), "capacity"),
    (dict(_record(), uuid="MIG-abc"), "physical GPU identity"),
])
def test_portable_still_needs_a_full_card(monkeypatch, record, match):
    monkeypatch.setenv(environment.PORTABLE_ENV, "1")
    with pytest.raises(ValueError, match=match):
        checks.validate_target_device(dict(record))


@pytest.mark.parametrize("portable", [False, True])
def test_listed_gpu_unchanged_in_both_modes(monkeypatch, portable):
    if portable:
        monkeypatch.setenv(environment.PORTABLE_ENV, "1")
    record = _record("NVIDIA RTX PRO 6000 Blackwell Server Edition", 96., "[N/A]")
    seen = dict(record)
    assert checks.validate_target_device(seen) == 96 and seen == record          # 名单内的卡不加标记
    with pytest.raises(ValueError, match="MIG"):
        checks.validate_target_device(_record("NVIDIA GeForce RTX 4090", 24., "Enabled"))


# ---------------------------------------------------------------- ② 环境版本

def _versions(**changes):
    return dict(dict(python="3.12.3", torch="2.11.0+cu130", cuda="13.0", numpy="2.4.3", PyYAML="6.0.3"), **changes)


def test_versions_strict_by_default_and_recorded_when_portable(monkeypatch):
    monkeypatch.setattr(environment, "versions", lambda: _versions())
    assert environment.verify_versions() == _versions()                         # 版本对时两种模式都不加字段
    monkeypatch.setattr(environment, "versions", lambda: _versions(python="3.11.9", torch="2.8.0+cu128", cuda="12.8"))
    with pytest.raises(ValueError, match="dependency versions differ.*PO_JEPA_PORTABLE=1"):
        environment.verify_versions()
    monkeypatch.setenv(environment.PORTABLE_ENV, "1")
    got = environment.verify_versions()
    assert got == dict(_versions(python="3.11.9", torch="2.8.0+cu128", cuda="12.8"), portable_version_mismatches=dict(
        torch=dict(expected="2.11.0", actual="2.8.0+cu128"), cuda=dict(expected="13.0", actual="12.8"),
        python=dict(expected="3.12.x", actual="3.11.9")))
    monkeypatch.setattr(environment, "versions", lambda: _versions())
    assert environment.verify_versions() == _versions()


# ---------------------------------------------------------------- ③ 续训换卡

def _source(tmp_path, runtime, registration):
    cache._write_json(tmp_path/"preregistration.json", registration)
    cache._write_json(tmp_path/"runtime.json", runtime)
    return tmp_path


def _moved(runtime):
    moved = dict(runtime, device="cuda:0", gpu_uuid="GPU-other", platform="Linux-other-node", cpu_threads=runtime["cpu_threads"]+8,
                 interop_threads=runtime["interop_threads"]+8)
    moved["environment"] = dict(runtime["environment"], CUDA_VISIBLE_DEVICES="3")
    return moved


def test_runtime_differences_names_nested_fields():
    a = dict(x=1, environment=dict(A="1", B=None), versions=dict(torch="2.11.0"))
    b = dict(x=1, environment=dict(A="2", B=None), versions=dict(torch="2.11.0"), y=0)
    assert environment.runtime_differences(a, b) == ["environment.A", "y"]
    assert environment.runtime_differences(a, a) == []


def test_resume_on_another_gpu_strict_by_default_allowed_when_portable(monkeypatch, tmp_path, capsys):
    runtime = environment.runtime_signature(torch.device("cpu")); registration = dict(fictional=True)
    source = _source(tmp_path, runtime, registration)
    resume.verify_training_source(source, registration, runtime)                 # 相同签名：两种模式都通过
    moved = _moved(runtime)
    with pytest.raises(ValueError, match="signature differs: .*gpu_uuid.*PO_JEPA_PORTABLE=1"):
        resume.verify_training_source(source, registration, moved)
    monkeypatch.setenv(environment.PORTABLE_ENV, "1")
    resume.verify_training_source(source, registration, moved)
    err = capsys.readouterr().err
    assert "PO_JEPA_PORTABLE=1" in err and all(k in err for k in environment.PORTABLE_RESUME_KEYS)
    resume.verify_training_source(source, registration, runtime)
    assert capsys.readouterr().err == ""                                        # 没换卡就不打印


@pytest.mark.parametrize("field,value", [("matmul_tf32", "flip"), ("deterministic", "flip"), ("default_dtype", "torch.float64"),
                                         ("matmul_precision", "medium"), ("versions", "torch"), ("environment", "OMP_NUM_THREADS")])
def test_portable_resume_still_needs_same_versions_and_numerics(monkeypatch, tmp_path, field, value):
    monkeypatch.setenv(environment.PORTABLE_ENV, "1")
    runtime = environment.runtime_signature(torch.device("cpu")); registration = dict(fictional=True)
    source = _source(tmp_path, runtime, registration)
    changed = _moved(runtime)
    if field == "versions":
        changed["versions"] = dict(runtime["versions"], torch="2.8.0")
    elif field == "environment":
        changed["environment"] = dict(changed["environment"], OMP_NUM_THREADS="64")
    else:
        changed[field] = (not runtime[field]) if value == "flip" else value
    with pytest.raises(ValueError, match="beyond GPU/host placement"):
        resume.verify_training_source(source, registration, changed)


def test_portable_resume_still_checks_preregistration(monkeypatch, tmp_path):
    monkeypatch.setenv(environment.PORTABLE_ENV, "1")
    runtime = environment.runtime_signature(torch.device("cpu"))
    source = _source(tmp_path, runtime, dict(fictional=True))
    with pytest.raises(ValueError, match="preregistration"):
        resume.verify_training_source(source, dict(fictional=False), _moved(runtime))
