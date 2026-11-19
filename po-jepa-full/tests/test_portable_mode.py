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
    moved = dict(runtime, gpu_uuid="GPU-other", platform="Linux-other-node", cpu_threads=runtime["cpu_threads"]+8,
                 interop_threads=runtime["interop_threads"]+8)
    moved["environment"] = dict(runtime["environment"], CUDA_VISIBLE_DEVICES="3")
    return moved


def test_runtime_differences_names_nested_fields():
    a = dict(x=1, environment=dict(A="1", B=None), versions=dict(torch="2.11.0"))
    b = dict(x=1, environment=dict(A="2", B=None), versions=dict(torch="2.11.0"), y=0)
    assert environment.runtime_differences(a, b) == ["environment.A", "y"]
    assert environment.runtime_differences(a, a) == []


@pytest.mark.parametrize("old,new,expected", [
    (dict(x=1), dict(x=1, extra={}), ["extra"]),                       # 多一个空字典
    (dict(x=1, extra={}), dict(x=1), ["extra"]),                       # 少一个空字典
    (dict(x=1, extra={}), dict(x=1, extra=None), ["extra"]),           # 空字典换成别的类型
    (dict(x=1, extra=dict(a=1)), dict(x=1, extra="a"), ["extra", "extra.a"]),
    (dict(x=1), dict(x=True), ["x"]),                                  # 1 == True 但类型不同
])
def test_runtime_differences_sees_structure_changes(old, new, expected):
    """C8-fix1（Codex P3-1）：空字典当叶子，类型不同也算不同，结构变化不会被展开吞掉。"""
    assert environment.runtime_differences(old, new) == expected


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


def test_portable_resume_keeps_logical_device(monkeypatch, tmp_path):
    """C8-fix1（Codex P2-2）：逻辑设备号不放过（checkpoint 和 CUDA RNG 都要求它不变），换卡要靠 CUDA_VISIBLE_DEVICES 映射到 cuda:0。"""
    monkeypatch.setenv(environment.PORTABLE_ENV, "1")
    runtime = environment.runtime_signature(torch.device("cpu")); registration = dict(fictional=True)
    source = _source(tmp_path, dict(runtime, device="cuda:0"), registration)
    with pytest.raises(ValueError, match="beyond GPU/host placement: device"):
        resume.verify_training_source(source, registration, dict(_moved(runtime), device="cuda:1"))
    assert "device" not in environment.PORTABLE_RESUME_KEYS


def test_portable_resume_rejects_extra_empty_structure(monkeypatch, tmp_path):
    monkeypatch.setenv(environment.PORTABLE_ENV, "1")
    runtime = environment.runtime_signature(torch.device("cpu")); registration = dict(fictional=True)
    source = _source(tmp_path, dict(runtime, extra={}), registration)
    with pytest.raises(ValueError, match="beyond GPU/host placement: extra"):
        resume.verify_training_source(source, registration, _moved(runtime))


@pytest.mark.parametrize("value", ["yes", "2"])
def test_invalid_switch_rejected_on_success_paths(monkeypatch, tmp_path, value):
    """C8-fix1（Codex P2-1）：开关值非法时，版本全对、名单内的卡、签名相同这三条本来会通过的路径也要报错。"""
    runtime = environment.runtime_signature(torch.device("cpu")); registration = dict(fictional=True)
    source = _source(tmp_path, runtime, registration)
    monkeypatch.setattr(environment, "versions", lambda: _versions())
    monkeypatch.setenv(environment.PORTABLE_ENV, value)
    with pytest.raises(ValueError, match="PO_JEPA_PORTABLE must be"):
        environment.verify_versions()
    with pytest.raises(ValueError, match="PO_JEPA_PORTABLE must be"):
        checks.validate_target_device(_record("NVIDIA RTX PRO 6000 Blackwell Server Edition", 96.))
    with pytest.raises(ValueError, match="PO_JEPA_PORTABLE must be"):
        resume.verify_training_source(source, registration, runtime)


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
