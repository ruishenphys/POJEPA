"""S18a：run.py d4 / spectral-gap 的参数检查和子进程命令，addon.eval_cli 的入口流程（16 §5、10 §7）。"""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from po_jepa.runtime import cache, checks, environment
from po_jepa.runtime import workflow as w
from po_jepa.runtime.addon import calibration, d4, eval_cli, spectral_gap


def entry():
    spec = importlib.util.spec_from_file_location("addon_eval_run_entry", w.ROOT/"run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run_py(tmp_path, monkeypatch):
    module = entry()
    seen = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kw: seen.append((command, kw["env"])) or Namespace(returncode=0))
    (tmp_path/"features").mkdir()
    base = ["--features-root", str(tmp_path/"features"), "--system", "double_well", "--trajectories", "200", "--updates", "2000"]
    return module, seen, base, tmp_path


def child_of(command):
    return command[command.index("--")+1:]


def test_run_py_d4_child_command(run_py):
    module, seen, base, tmp = run_py
    (tmp/"v7job").mkdir()
    assert module.main(["d4", *base, "--device", "cuda:1", "--output", str(tmp/"out"), "--seed", "1704",
                        "--condition", "po_jepa_full", "--training-run", str(tmp/"v7job")]) == 0
    command, env = seen[0]
    child = child_of(command)
    assert child[child.index("-m")+1] == "po_jepa.runtime.addon.eval_cli" and child[child.index("-m")+2] == "d4"
    assert child[child.index("--output")+1] == str((tmp/"out").resolve()/"run")
    assert child[child.index("--device")+1] == "cuda:1" and child[child.index("--seed")+1] == "1704"
    assert child[child.index("--condition")+1] == "po_jepa_full"
    assert child[child.index("--training-run")+1] == str((tmp/"v7job").resolve())
    assert command[command.index("--output")+1] == str((tmp/"out").resolve()/"console")
    assert env["PO_JEPA_DATASET"] == "double_well:200" and env["PO_JEPA_UPDATES"] == "2000" and env["PO_JEPA_PROTOCOL"] == "v7"


def test_run_py_spectral_gap_child_command(run_py):
    module, seen, base, tmp = run_py
    assert module.main(["spectral-gap", *base, "--device", "cpu", "--output", str(tmp/"out")]) == 0
    child = child_of(seen[0][0])
    assert child[child.index("-m")+1:child.index("-m")+3] == ["po_jepa.runtime.addon.eval_cli", "spectral-gap"]
    assert "--condition" not in child and "--training-run" not in child and child[child.index("--seed")+1] == "1703"


def test_run_py_other_commands_keep_v7_cli(run_py):
    module, seen, base, tmp = run_py
    assert module.main(["run", *base, "--output", str(tmp/"out"), "--condition", "po_jepa_full"]) == 0
    child = child_of(seen[0][0])
    assert child[child.index("-m")+1:child.index("-m")+3] == ["po_jepa.runtime.cli", "run"]


@pytest.mark.parametrize("extra", [
    ["d4", "--device", "cuda:0", "--condition", "po_jepa_full"],                         # 缺 --training-run
    ["d4", "--device", "cuda:0", "--training-run", "x"],                                 # 缺 --condition
    ["d4", "--device", "cuda:0", "--condition", "vjepa2_ac", "--training-run", "x"],     # vjepa2_ac 缺 --ac-checkpoint
    ["spectral-gap", "--device", "cuda:0"],
    ["spectral-gap", "--device", "cpu", "--condition", "po_jepa_full"],
    ["spectral-gap", "--device", "cpu", "--training-run", "x"],
    ["spectral-gap", "--device", "cpu", "--seed", "1704"],
])
def test_run_py_parse_errors(run_py, extra, capsys):
    module, seen, base, tmp = run_py
    with pytest.raises(SystemExit) as error:
        module.main([extra[0], *base, "--output", str(tmp/"out"), *extra[1:]])
    assert error.value.code == 2 and seen == [] and not (tmp/"out").exists()


@pytest.mark.parametrize("argv", [["d4", "--condition", "po_jepa_full"], ["d4", "--training-run", "x"],
                                  ["spectral-gap", "--seed", "1704"], ["spectral-gap", "--condition", "identity"]])
def test_eval_cli_parse_errors(tmp_path, argv):
    with pytest.raises(SystemExit) as error:
        eval_cli.main([argv[0], "--features-root", str(tmp_path), "--device", "cpu", "--output", str(tmp_path/"o"), *argv[1:]])
    assert error.value.code == 2


@pytest.fixture
def stubs(monkeypatch):
    calls = []
    state = dict(bundle=["b0", "b0"])
    monkeypatch.setattr(w, "verify_bundle", lambda: state["bundle"].pop(0))
    monkeypatch.setattr(environment, "verify_versions", lambda: dict(torch="stub"))
    monkeypatch.setattr(w, "require_calibrated_weights", lambda swept=False: calls.append(("v7_calibration", swept)))
    monkeypatch.setattr(calibration, "require_addon_calibration", lambda: calls.append(("addon_calibration",)))
    monkeypatch.setattr(d4, "run_d4", lambda features, output, device, condition, *, training_run:
                        calls.append(("d4", output.name, str(device), condition, Path(training_run).name)))
    monkeypatch.setattr(spectral_gap, "run_spectral_gap", lambda features, output, device: calls.append(("gap", output.name, str(device))))
    monkeypatch.setenv(checks.SEED_ENV, "1703")
    return calls, state


def test_eval_cli_d4_flow(tmp_path, stubs, monkeypatch):
    calls, state = stubs
    argv = ["d4", "--features-root", str(tmp_path), "--device", "cuda:0", "--output", str(tmp_path/"o"), "--seed", "1705",
            "--condition", "fixed_fourier", "--training-run", str(tmp_path/"v7job")]
    assert eval_cli.main(argv) == 0
    assert calls == [("v7_calibration", True), ("addon_calibration",), ("d4", "job", "cuda:0", "fixed_fourier", "v7job")]
    import os
    assert os.environ[checks.SEED_ENV] == "1705"
    assert json.loads((tmp_path/"o"/"COMPLETE.json").read_text()) == dict(command="d4", condition="fixed_fourier", manifest_sha256="b0",
                                                                         scientific_conclusions_verified=False)
    assert json.loads((tmp_path/"o"/"environment.json").read_text()) == dict(torch="stub")
    calls.clear(); state["bundle"] = ["b0", "b0"]
    assert eval_cli.main([*argv[:5], "--output", str(tmp_path/"o2"), "--condition", "po_jepa_full", "--training-run", "x"]) == 0
    assert ("addon_calibration",) not in calls


def test_eval_cli_refusals(tmp_path, stubs):
    calls, state = stubs
    base = ["--features-root", str(tmp_path), "--condition", "po_jepa_full", "--training-run", "x"]
    for device in ("cpu", "cuda"):
        state["bundle"] = ["b0"]
        with pytest.raises(ValueError, match="explicit cuda:N"):
            eval_cli.main(["d4", *base, "--device", device, "--output", str(tmp_path/"o")])
    state["bundle"] = ["b0"]
    with pytest.raises(ValueError, match="--device cpu"):
        eval_cli.main(["spectral-gap", "--features-root", str(tmp_path), "--device", "cuda:0", "--output", str(tmp_path/"o")])
    assert not (tmp_path/"o").exists()
    state["bundle"] = ["b0", "b1"]
    with pytest.raises(ValueError, match="source bundle changed"):
        eval_cli.main(["spectral-gap", "--features-root", str(tmp_path), "--device", "cpu", "--output", str(tmp_path/"o")])
    failed = json.loads((tmp_path/"o"/"FAILED.json").read_text())
    assert failed["stage"] == "spectral-gap" and failed["automatic_retry"] is False and not (tmp_path/"o"/"COMPLETE.json").exists()
    state["bundle"] = ["b0"]
    with pytest.raises(FileExistsError):
        eval_cli.main(["spectral-gap", "--features-root", str(tmp_path), "--device", "cpu", "--output", str(tmp_path/"o")])


def test_eval_cli_records_failed_stage(tmp_path, stubs, monkeypatch):
    calls, state = stubs
    def uncalibrated(swept=False):
        raise ValueError("not calibrated")
    monkeypatch.setattr(w, "require_calibrated_weights", uncalibrated)
    with pytest.raises(ValueError, match="not calibrated"):
        eval_cli.main(["spectral-gap", "--features-root", str(tmp_path), "--device", "cpu", "--output", str(tmp_path/"o")])
    assert json.loads((tmp_path/"o"/"FAILED.json").read_text())["stage"] == "calibration"
    assert [c for c in calls if c[0] == "gap"] == []
