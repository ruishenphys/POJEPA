"""S17：run.py calibrate-addon（16 §3.2）、check-addon（16 §5）和它们的入口 po_jepa.runtime.addon.entry。

标定的流程测试用 test_addon_conditions 的小模型和 e06 合成批（把 300/32 调成 3/2），真实规格的模型只用来核“只用 JEPA 的
版本”的构造和初始权重；真实特征上的完整运行在 tests/addon_entry_smoke.py（端到端冒烟）。"""

import copy
import json
import math
import statistics
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml

from po_jepa import datasets
from po_jepa.fourier import FourierSpectrum
from po_jepa.runtime import cache, checks, comparison, v7jobs
from po_jepa.runtime import workflow as w
from po_jepa.runtime.addon import calibrate, calibration, check_addon, conditions, entry
from test_addon_conditions import po, transformer
from test_training import e06_case, e06_config

CPU = torch.device("cpu")
V7_KEYS = {"rho", "batches", "statistic", "method", "pretrain_updates", "pretrain_history", "per_batch", "median", "mean",
           "initial_state", "time_weight", "space_weight", "graph", "v4_overlay_sha256_before", "note",
           "v7_calibration_file", "v4_overlay_sha256_after"}


@pytest.fixture
def v7(monkeypatch, tmp_path):
    """v7 协议、van_der_pol:2000；v7 标定和补跑标定都写在临时目录，导入记录指向 v7 标定。"""
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    old = datasets._set_for_tests(datasets.DatasetSpec("van_der_pol", 2000))
    folder = tmp_path/"configs/calibration"
    folder.mkdir(parents=True)
    path = folder/"van_der_pol-2000-u10000.yaml"
    monkeypatch.setattr(checks, "v7_calibration_path", lambda: path)
    monkeypatch.setattr(calibration, "addon_calibration_path", lambda: folder/"addon"/path.name)
    w.write_v7_calibration(dict(sigma_z=1.25, time_weight=.03, space_weight=4.0, rho=1.0), "test")
    (folder/"IMPORTED.json").write_bytes(cache._json_bytes(dict(schema="addon_v7_calibration_import_v1", files={
        path.name: dict(source="fictional", sha256=cache.file_sha(path), imported_at="2026-11-13T00:00:00+00:00",
                        v7_commit="0ac45fe03bf43b96bf861446b8aa2ada4f4a8987")})))
    yield path
    datasets._set_for_tests(old)


@pytest.fixture
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def small(v7, monkeypatch):
    """协议的 pretrain_updates/batches 调成 3/2；只用 JEPA 的模型换成小模型，训练批换成 e06 合成批。"""
    full = checks.load_protocol

    def protocol():
        value = copy.deepcopy(full())
        value["calibration"].update(pretrain_updates=3, batches=2)
        return value
    monkeypatch.setattr(checks, "load_protocol", protocol)
    zero = e06_config(time_weight=0., space_weight=0.)
    models = dict(fixed_fourier=lambda: po("fourier"), transformer_consistency=transformer)
    monkeypatch.setattr(calibrate, "_jepa_only", lambda condition, device: (models[condition](), zero))
    monkeypatch.setattr(calibrate.data, "load_training_batch", lambda features, queries, device: e06_case()[1])
    return v7


def run_calibration(tmp_path, name="run"):
    output = tmp_path/"calibrate-van_der_pol"/name
    output.mkdir(parents=True)
    return calibrate.calibrate_addon(tmp_path/"features", output, CPU), output


def test_calibrate_addon_writes_yaml_and_records(small, tmp_path, one_thread):
    payload, output = run_calibration(tmp_path)
    path = calibration.addon_calibration_path()
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "# 仿真补跑标定（run.py calibrate-addon 自动写入，勿手改）：van_der_pol-2000，10000 步"
    value = yaml.safe_load(text)
    assert value == payload and set(value) == calibration.FILE_KEYS
    assert value["sigma_z_used"] == 1.25 and value["v7_calibration_sha256"] == cache.file_sha(small)
    for condition in calibration.CALIBRATED_CONDITIONS:
        entry_value = value[condition]
        assert set(entry_value) == calibration.ENTRY_KEYS and entry_value["rho"] == 1.0
        assert entry_value["source"] == "calibrate-addon：JEPA 预训练 3 步后，2 批梯度范数的中位数；calibrate-van_der_pol"
        record = json.loads((output/f"calibration-{condition}.json").read_text(encoding="utf-8"))
        assert set(record) == V7_KEYS|{"condition", "v7_calibration_sha256", "addon_calibration_file", "base_calibration"}
        assert record["base_calibration"] == "v7-import"
        assert record["condition"] == condition and len(record["per_batch"]) == 2 and len(record["pretrain_history"]) == 3
        assert [r["step"] for r in record["pretrain_history"]] == [1, 2, 3] and len(record["initial_state"]["per_batch"]) == 2
        rows = record["per_batch"]
        for name, key in (("time_weight", "grad_time"), ("space_weight", "grad_space")):     # λ 按同一组梯度范数手算
            expected = 1.0*statistics.median(r["grad_jepa"] for r in rows)/statistics.median(r[key] for r in rows)
            assert entry_value[name] == record[name] == expected and math.isfinite(expected) and expected > 0
        assert calibration.load_addon_weights(condition) == dict(time_weight=entry_value["time_weight"], space_weight=entry_value["space_weight"])
    tc = json.loads((output/"calibration-transformer_consistency.json").read_text(encoding="utf-8"))
    # 登记的初始化下 p_out 为零，一致性梯度恰为 0：初始状态的 λ 记 None（S10b §9 第 13 条），300 步之后的不为 0
    assert all(r["grad_time"] == 0. and r["grad_space"] == 0. for r in tc["initial_state"]["per_batch"])
    assert tc["initial_state"]["time_weight"] is None and tc["initial_state"]["space_weight"] is None
    assert sorted(p.name for p in output.iterdir()) == ["calibration-fixed_fourier.json", "calibration-transformer_consistency.json"]
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_calibrate_addon_is_reproducible(small, tmp_path, one_thread):
    first, out1 = run_calibration(tmp_path, "run")
    text = calibration.addon_calibration_path().read_bytes()
    calibration.addon_calibration_path().unlink()
    second, out2 = run_calibration(tmp_path/"again", "run")
    assert first == second and calibration.addon_calibration_path().read_bytes() == text
    for condition in calibration.CALIBRATED_CONDITIONS:
        assert (out1/f"calibration-{condition}.json").read_bytes() == (out2/f"calibration-{condition}.json").read_bytes()


def scripted(monkeypatch, measured, count=2):
    """梯度范数按顺序给出：每个条件先 count 批初始状态，再 count 批预训练之后；后者来自 measured。"""
    def norms(condition, model, batch, loss):
        norms.calls[condition] = norms.calls.get(condition, 0)+1
        index = norms.calls[condition]-1
        row = dict(jepa=1., time=1., space=1., grad_jepa=1., grad_time=1., grad_space=1.)
        return dict(row, **measured[condition][index-count]) if index >= count else row
    norms.calls = {}
    monkeypatch.setattr(calibrate, "_gradient_norms", norms)


def test_lambda_equals_hand_computed_value(small, tmp_path, monkeypatch):
    """3 批，中位数和均值不同：λ = ρ·中位数/中位数，不是均值。"""
    protocol = checks.load_protocol
    monkeypatch.setattr(checks, "load_protocol", lambda: dict(protocol(), calibration=dict(protocol()["calibration"], batches=3)))
    scripted(monkeypatch, dict(
        fixed_fourier=[dict(grad_jepa=2., grad_time=8., grad_space=.5), dict(grad_jepa=4., grad_time=4., grad_space=1.5),
                       dict(grad_jepa=9., grad_time=1., grad_space=1.)],
        transformer_consistency=[dict(grad_jepa=1., grad_time=3., grad_space=2.), dict(grad_jepa=5., grad_time=1., grad_space=2.),
                                 dict(grad_jepa=3., grad_time=8., grad_space=6.)]), count=3)
    payload, _ = run_calibration(tmp_path)
    assert payload["fixed_fourier"]["time_weight"] == 4./4. and payload["fixed_fourier"]["space_weight"] == 4./1.
    assert payload["transformer_consistency"]["time_weight"] == 3./3. and payload["transformer_consistency"]["space_weight"] == 3./2.


@pytest.mark.parametrize("bad", [dict(grad_time=0.), dict(grad_space=0.)])
def test_zero_denominator_fails_without_files(small, tmp_path, monkeypatch, bad):
    row = dict(grad_jepa=1., grad_time=1., grad_space=1.)
    scripted(monkeypatch, dict(fixed_fourier=[row, row], transformer_consistency=[dict(row, **bad), dict(row, **bad)]))
    key = next(iter(bad))
    with pytest.raises(ValueError, match=f"^transformer_consistency: median {key} = 0.0; λ 的分母为 0"):
        run_calibration(tmp_path)
    assert not calibration.addon_calibration_path().exists()
    assert list((tmp_path/"calibrate-van_der_pol/run").iterdir()) == []


def test_nonpositive_lambda_fails(small, tmp_path, monkeypatch):
    row = dict(grad_jepa=0., grad_time=1., grad_space=1.)
    scripted(monkeypatch, dict(fixed_fourier=[row, row], transformer_consistency=[row, row]))
    with pytest.raises(ValueError, match="^calibrated weight must be finite and positive: fixed_fourier.time_weight$"):
        run_calibration(tmp_path)
    assert not calibration.addon_calibration_path().exists()


def test_failure_in_second_condition_leaves_no_file(small, tmp_path, monkeypatch):
    original = calibrate._pretrain_step

    def step(condition, *args):
        if condition == "transformer_consistency":
            raise RuntimeError("fictional failure in the second condition")
        return original(condition, *args)
    monkeypatch.setattr(calibrate, "_pretrain_step", step)
    with pytest.raises(RuntimeError, match="fictional failure"):
        run_calibration(tmp_path)
    assert not calibration.addon_calibration_path().parent.exists() or not any(calibration.addon_calibration_path().parent.iterdir())
    assert list((tmp_path/"calibrate-van_der_pol/run").iterdir()) == []


def test_existing_file_and_import_record_are_checked(small, tmp_path, monkeypatch):
    path = calibration.addon_calibration_path()
    path.parent.mkdir(parents=True)
    path.write_text("fictional: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="^补跑标定文件已存在"):
        run_calibration(tmp_path)
    assert path.read_text(encoding="utf-8") == "fictional: 1\n"
    path.unlink()
    record = small.parent/"IMPORTED.json"
    good = record.read_bytes()
    record.write_bytes(cache._json_bytes(dict(schema="addon_v7_calibration_import_v1", files={})))
    with pytest.raises(ValueError, match="和 IMPORTED.json 的导入记录不符"):
        run_calibration(tmp_path, "a")
    record.unlink()
    with pytest.raises(ValueError, match="IMPORTED.json：v7 标定必须用 tools/import_v7_calibration.py 导入"):
        run_calibration(tmp_path, "b")
    record.write_bytes(good)
    small.write_text(small.read_text(encoding="utf-8")+"# edited\n", encoding="utf-8")
    with pytest.raises(ValueError, match="导入记录不符"):
        run_calibration(tmp_path, "c")
    small.unlink()
    with pytest.raises(ValueError, match="^缺少 v7 标定"):
        run_calibration(tmp_path, "d")
    assert not path.exists()


def test_readback_failure_removes_file(small, tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "load_addon_weights", lambda condition: dict(time_weight=1., space_weight=1.))
    with pytest.raises(ValueError, match="^add-on calibration file reads back different weights$"):
        run_calibration(tmp_path)
    assert not calibration.addon_calibration_path().exists()


def test_v7_file_changing_during_calibration_fails(small, tmp_path, monkeypatch):
    original = calibrate.calibrate_condition

    def edit(condition, features, device):
        result = original(condition, features, device)
        if condition == "transformer_consistency":
            small.write_text(small.read_text(encoding="utf-8")+"# edited\n", encoding="utf-8")
        return result
    monkeypatch.setattr(calibrate, "calibrate_condition", edit)
    with pytest.raises(ValueError, match="导入记录不符"):
        run_calibration(tmp_path)
    assert not calibration.addon_calibration_path().exists()


def test_calibration_guards(small, tmp_path, monkeypatch):
    """另外三道检查：查询序列不够长；YAML 读不回原值；标定期间 v7 标定连同导入记录一起被换掉。都不留补跑标定文件。"""
    protocol = checks.load_protocol
    monkeypatch.setattr(checks, "load_protocol", lambda: dict(protocol(), calibration=dict(protocol()["calibration"], pretrain_updates=10**9)))
    with pytest.raises(ValueError, match="^pretrain_updates\\+batches exceeds the training sequence$"):
        run_calibration(tmp_path, "a")
    monkeypatch.setattr(checks, "load_protocol", protocol)
    monkeypatch.setattr(calibrate, "yaml", Namespace(safe_dump=yaml.safe_dump, safe_load=lambda text: {}))
    with pytest.raises(ValueError, match="^add-on calibration YAML does not round-trip$"):
        run_calibration(tmp_path, "b")
    monkeypatch.setattr(calibrate, "yaml", yaml)
    original = calibrate.calibrate_condition

    def swap(condition, features, device):
        result = original(condition, features, device)
        if condition == "transformer_consistency":
            small.write_text(small.read_text(encoding="utf-8")+"# re-imported\n", encoding="utf-8")
            record = small.parent/"IMPORTED.json"
            value = json.loads(record.read_text(encoding="utf-8"))
            value["files"][small.name]["sha256"] = cache.file_sha(small)
            record.write_bytes(cache._json_bytes(value))
        return result
    monkeypatch.setattr(calibrate, "calibrate_condition", swap)
    with pytest.raises(ValueError, match="^v7 calibration file changed during calibrate-addon$"):
        run_calibration(tmp_path, "c")
    assert not calibration.addon_calibration_path().exists()
    assert all(list((tmp_path/"calibrate-van_der_pol"/name).iterdir()) == [] for name in "abc")


def test_jepa_only_versions_and_initial_hashes(v7):
    """16 §3.2 第 2 步：fixed_fourier 两个权重为 0、谱是 FourierSpectrum、初始权重和 po_jepa_full 相同；
    transformer_consistency 用的就是 matched_transformer（同一个初始哈希、两个权重为 0）。"""
    model, loss = calibrate._jepa_only("fixed_fourier", CPU)
    full, full_loss = comparison.build_condition("po_jepa_full", CPU)
    assert type(model.spectrum) is FourierSpectrum and (loss.time_weight, loss.space_weight) == (0., 0.)
    assert full_loss.time_weight == .03 and full_loss.space_weight == 4.0
    assert {k: v for k, v in vars(loss).items() if not k.endswith("_weight")} == {k: v for k, v in vars(full_loss).items() if not k.endswith("_weight")}
    assert v7jobs._model_hash(model) == v7jobs._model_hash(full)
    model, loss = calibrate._jepa_only("transformer_consistency", CPU)
    reference, _ = comparison.build_condition("matched_transformer", CPU)
    assert v7jobs._model_hash(model) == v7jobs._model_hash(reference) and (loss.time_weight, loss.space_weight) == (0., 0.)
    with pytest.raises(ValueError, match="^only fixed_fourier and transformer_consistency"):
        calibrate._jepa_only("koopman", CPU)


# ---------- check-addon ----------

def test_initial_relations():
    hashes = dict(po_jepa_full="a", matched_transformer="b", po_jepa_time_only="a", po_jepa_space_only="a",
                  fixed_fourier="a", transformer_consistency="b", koopman="c")
    assert check_addon.initial_relations(hashes) == dict(po_jepa_time_only="po_jepa_full", po_jepa_space_only="po_jepa_full",
        fixed_fourier="po_jepa_full", transformer_consistency="matched_transformer", koopman=None)
    for condition, reference in (("po_jepa_time_only", "po_jepa_full"), ("fixed_fourier", "po_jepa_full"),
                                 ("po_jepa_space_only", "po_jepa_full"), ("transformer_consistency", "matched_transformer")):
        with pytest.raises(ValueError, match=f"^{condition} 的初始权重和 {reference} 不同"):
            check_addon.initial_relations(dict(hashes, **{condition: "x"}))
    assert set(check_addon.REFERENCE) == set(conditions.ADDON_CONDITIONS) and check_addon.TIMED_UPDATES == 100


def test_check_addon_rejects_missing_gradient(v7, monkeypatch, tmp_path):
    """一次更新之后有参数没有梯度就报错（用 identity 换掉单步来造这种情况）。"""
    monkeypatch.setattr(calibration, "load_addon_weights", lambda condition: dict(time_weight=1., space_weight=1.))
    model = po()
    monkeypatch.setattr(comparison, "build_condition", lambda condition, device: (model, e06_config()))
    monkeypatch.setattr(comparison, "_validate_registration", lambda *a: None)
    monkeypatch.setattr(comparison, "make_optimizer", lambda m: None)
    monkeypatch.setattr(check_addon.data, "load_training_batch", lambda *a, **k: None)
    fake = Namespace(loss_values={}, grad_max_abs=0., grad_norm=0.)
    monkeypatch.setattr(comparison, "_observed_update", lambda *a: (fake, []))
    with pytest.raises(ValueError, match="^po_jepa_time_only: parameter .* has no finite gradient$"):
        check_addon._diagnostic("po_jepa_time_only", None, CPU, [], None, [])


def test_check_addon_timed_guards(v7, monkeypatch):
    """计时的更新：步数不对、损失不是有限数、耗时不是正数，各自报错；正常时返回步数和耗时。"""
    monkeypatch.setattr(comparison, "build_condition", lambda condition, device: (None, None))
    monkeypatch.setattr(comparison, "make_optimizer", lambda m: None)
    monkeypatch.setattr(check_addon.data, "load_training_batch", lambda *a, **k: None)
    monkeypatch.setattr(check_addon, "TIMED_UPDATES", 3)

    def step(extra, jepa):
        return lambda *a: Namespace(completed_steps=a[-1]+extra, loss_values=dict(jepa=jepa, time=None))
    for extra, jepa in ((0, 1.), (1, float("nan"))):
        monkeypatch.setattr(comparison, "condition_step", step(extra, jepa))
        with pytest.raises(ValueError, match="^koopman: timed update count or finite losses failed$"):
            check_addon._timed("koopman", None, CPU)
    monkeypatch.setattr(comparison, "condition_step", step(1, 1.))
    timed = check_addon._timed("koopman", None, CPU)
    assert timed["updates"] == 3 and timed["jepa"] == [1., 1., 1.] and timed["seconds"] > 0 and timed["memory"] is None
    monkeypatch.setattr(check_addon, "time", Namespace(perf_counter=lambda: 5.))
    with pytest.raises(ValueError, match="^nonfinite update timing$"):
        check_addon._timed("koopman", None, CPU)


# ---------- 入口 ----------

def args(command, tmp_path, **changes):
    values = dict(command=command, features_root=tmp_path/"features", device="cuda:0", output=tmp_path/"out"/"run", seed=1703,
                  base_calibration="v7-import")
    values.update(changes)
    return Namespace(**values)


@pytest.fixture
def stubs(v7, monkeypatch):
    calls = []
    monkeypatch.setattr(w, "verify_bundle", lambda: "fictional-bundle")
    monkeypatch.setattr(w.environment, "verify_versions", lambda: dict(fictional=True))

    def fake_calibrate(features, output, device, base):
        calls.append(("calibrate", device.type, base))
        path = calibration.addon_calibration_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fictional: 1\n", encoding="utf-8")

    def fake_check(features, output, device):
        calls.append(("check", str(device)))
        cache._write_json(output/"check-addon.json", dict(fictional=True))
        return dict(hardware=dict(device=str(device)))
    monkeypatch.setattr(calibrate, "calibrate_addon", fake_calibrate)
    monkeypatch.setattr(check_addon, "check_addon", fake_check)
    monkeypatch.setattr(calibration, "require_addon_calibration", lambda: calls.append(("addon",)))
    return calls


def test_entry_calibrate_addon_completes(stubs, tmp_path):
    assert entry.execute(args("calibrate-addon", tmp_path, device="cpu")) == 0
    done = json.loads((tmp_path/"out/run/COMPLETE.json").read_text(encoding="utf-8"))
    path = calibration.addon_calibration_path()
    assert done == dict(command="calibrate-addon", manifest_sha256="fictional-bundle", base_calibration="v7-import", addon_calibration_file=str(path),
                        addon_calibration_sha256=cache.file_sha(path), scientific_conclusions_verified=False)
    assert stubs == [("calibrate", "cpu", "v7-import")] and not (tmp_path/"out/run/environment.json").exists()


def test_entry_check_addon_completes(stubs, tmp_path):
    assert entry.execute(args("check-addon", tmp_path)) == 0
    run = tmp_path/"out/run"
    assert json.loads((run/"environment.json").read_text()) == dict(fictional=True)
    done = json.loads((run/"COMPLETE.json").read_text(encoding="utf-8"))
    assert done == dict(command="check-addon", manifest_sha256="fictional-bundle", base_calibration="v7-import", check_sha256=cache.file_sha(run/"check-addon.json"),
                        real_device=dict(device="cuda:0"), scientific_conclusions_verified=False)
    assert stubs == [("addon",), ("check", "cuda:0")]


def test_entry_failures(stubs, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="^explicit cuda:N target required$"):
        entry.execute(args("check-addon", tmp_path, device="cpu"))
    assert not (tmp_path/"out").exists()
    (tmp_path/"configs/calibration/IMPORTED.json").unlink()
    with pytest.raises(ValueError, match="IMPORTED.json"):
        entry.execute(args("calibrate-addon", tmp_path, device="cpu", output=tmp_path/"a/run"))
    failed = json.loads((tmp_path/"a/run/FAILED.json").read_text(encoding="utf-8"))
    assert failed["stage"] == "calibration" and failed["error_type"] == "ValueError" and failed["automatic_retry"] is False
    assert not (tmp_path/"a/run/COMPLETE.json").exists() and stubs == []
    monkeypatch.setattr(w.environment, "verify_versions", lambda: (_ for _ in ()).throw(ValueError("fictional versions differ")))
    with pytest.raises(ValueError, match="fictional versions differ"):
        entry.execute(args("check-addon", tmp_path, output=tmp_path/"b/run"))
    assert json.loads((tmp_path/"b/run/FAILED.json").read_text())["stage"] == "environment"


def test_entry_failure_in_work_and_bundle_change(stubs, v7, tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "calibrate_addon", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fictional work failure")))
    with pytest.raises(RuntimeError):
        entry.execute(args("calibrate-addon", tmp_path, device="cpu", output=tmp_path/"c/run"))
    assert json.loads((tmp_path/"c/run/FAILED.json").read_text())["stage"] == "calibrate-addon"
    bundles = iter(["one", "two"])
    monkeypatch.setattr(w, "verify_bundle", lambda: next(bundles))
    with pytest.raises(ValueError, match="^source bundle changed during run$"):
        entry.execute(args("check-addon", tmp_path, output=tmp_path/"d/run"))
    assert json.loads((tmp_path/"d/run/FAILED.json").read_text())["stage"] == "check-addon"


def test_entry_parser_and_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv(checks.SEED_ENV, "1703")          # entry.main 会设这个环境变量；测试结束时恢复
    seen = []
    monkeypatch.setattr(entry, "execute", lambda a: seen.append(a) or 0)
    base = ["--features-root", str(tmp_path), "--output", str(tmp_path/"o")]
    assert entry.main(["check-addon", *base]) == 0 and seen[0].seed == 1703 and seen[0].device == "cuda:0"
    for bad in (["--seed", "1704"], ["--condition", "koopman"]):
        with pytest.raises(SystemExit):
            entry.main(["calibrate-addon", *base, *bad])
    with pytest.raises(SystemExit):
        entry.main(["d4", *base])
    monkeypatch.undo()
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v6")
    with pytest.raises(ValueError, match="^补跑子命令只用于 v7 协议"):
        entry.execute(args("calibrate-addon", tmp_path))


def test_import_record_must_name_token_v7(v7):
    """Codex 评审（2026-11-09）：补跑入口不只核 SHA，还核导入记录的 v7_commit 是 0ac45fe；旧提交（模态写法）的记录拒绝。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("import_v7_calibration", Path(calibrate.__file__).parents[4]/"tools"/"import_v7_calibration.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert (calibrate.V7_COMMIT,) == tool.V7_COMMITS
    assert calibrate.require_imported_v7_calibration() == cache.file_sha(v7)
    record = v7.parent/"IMPORTED.json"
    files = json.loads(record.read_text(encoding="utf-8"))["files"]
    for old in ("653e2d07477d809e933fdedddebfc77c655b71c5", "911bc09", None):
        entry = dict(files[v7.name], v7_commit=old) if old else {k: x for k, x in files[v7.name].items() if k != "v7_commit"}
        record.write_bytes(cache._json_bytes(dict(schema="addon_v7_calibration_import_v1", files={v7.name: entry})))
        with pytest.raises(ValueError, match="旧提交是模态写法"):
            calibrate.require_imported_v7_calibration()
