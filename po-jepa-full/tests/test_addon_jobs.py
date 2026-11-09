"""S17：补跑的任务文件（tools/make_addon_jobs.py，16 §7）、运行器复制件（tools/run_jobs.py，07 §2.8）和 run.py 的补跑子命令分发。"""

import importlib.util
import itertools
import os
import shlex
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import make_addon_jobs as jobs  # noqa: E402
import run_jobs  # noqa: E402

ADDON = ("po_jepa_time_only", "po_jepa_space_only", "transformer_consistency", "fixed_fourier", "koopman")
V7 = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity", "vjepa2_ac_small", "vjepa2_ac")
SCENES = ("double_well", "van_der_pol", "multi_particle", "advection_diffusion", "landscape~shape=multi_well")
COUNTS = dict(pilot=dict(A0=3, C=5, C2=1, D=25, D4=60, E=1), full=dict(A0=3, C=5, C2=1, D=75, D4=170, E=1),
              variants=dict(V3=150, V4=25, V5=1), v7tree=dict(D3=5))
V6 = "/data/v6-features"           # 任意的绝对路径（机哥给的是 v7 交接说明里 V6= 的那个）


def lines(stage):
    return jobs.job_lines(stage, V6 if stage == "pilot" else "")


def sections(stage, tmp_path):
    path = tmp_path/f"jobs-addon-{stage}.txt"
    path.write_text("\n".join(lines(stage))+"\n", encoding="utf-8")
    return run_jobs.parse_sections(path)


def load_run_py():
    spec = importlib.util.spec_from_file_location("addon_jobs_run_py", ROOT/"run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- make_addon_jobs.py ----------

@pytest.mark.parametrize("stage", tuple(COUNTS))
def test_stage_sections_and_counts(stage, tmp_path):
    found = {s["name"]: len(s["commands"]) for s in sections(stage, tmp_path)}
    assert found == COUNTS[stage]
    assert sum(found.values()) == dict(pilot=95, full=255, variants=176, v7tree=5)[stage]


def test_names_match_runtime_lists():
    from po_jepa.runtime import v7jobs
    from po_jepa.runtime.addon import conditions
    assert jobs.ADDON_CONDITIONS == conditions.ADDON_CONDITIONS == ADDON
    assert jobs.V7_CONDITIONS == V7 == v7jobs.JOB_CONDITIONS[:6]
    assert jobs.BASE_SCENES == SCENES and jobs.STAGES["pilot"]["scenes"] == jobs.STAGES["full"]["scenes"] == SCENES


def test_lines_from_spec_examples():
    """16 §7.1–§7.3 里写出来的命令逐字出现。"""
    full, variants, v7tree = (jobs.job_lines(s) for s in ("full", "variants", "v7tree"))
    for line in (
        "python tools/import_v7_calibration.py --v7-tree ../../po-jepa --stage full",
        "python tools/encoder_identity.py --encoder ../../checkpoints/vjepa2-vitg-fpc64-256.pth --ac-checkpoint ../../checkpoints/vjepa2-ac-vitg.pt --output ../../results-addon/full/encoder-identity.json",
        "python tools/addon_preflight.py --v7-results ../../results-v7/full --features ../../features-v7 --stage full --output ../../results-addon/full/preflight.json",
        "python run.py calibrate-addon --system double_well --trajectories 2000 --updates 10000 --features-root ../../features-v7/double_well-2000 --device cuda:0 --output ../../results-addon/full/calibrate-double_well",
        "python run.py check-addon --system van_der_pol --trajectories 2000 --updates 10000 --features-root ../../features-v7/van_der_pol-2000 --device cuda:0 --output ../../results-addon/full/check",
        "python run.py run --system double_well --trajectories 2000 --updates 10000 --seed 1703 --condition fixed_fourier --features-root ../../features-v7/double_well-2000 --device cuda:0 --output ../../results-addon/full/double_well/seed1703/fixed_fourier",
        "python run.py d4 --system double_well --trajectories 2000 --updates 10000 --seed 1703 --condition po_jepa_full --training-run ../../results-v7/full/double_well/seed1703/po_jepa_full --features-root ../../features-v7/double_well-2000 --device cuda:0 --output ../../results-addon/full/d4/double_well/seed1703/po_jepa_full",
        "python run.py d4 --system double_well --trajectories 2000 --updates 10000 --seed 1703 --condition koopman --training-run ../../results-addon/full/double_well/seed1703/koopman --features-root ../../features-v7/double_well-2000 --device cuda:0 --output ../../results-addon/full/d4/double_well/seed1703/koopman",
        "python run.py d4 --system double_well --trajectories 2000 --updates 10000 --seed 1703 --condition vjepa2_ac --training-run ../../results-v7/full/double_well/seed1703/vjepa2_ac --features-root ../../features-v7/double_well-2000 --device cuda:0 --output ../../results-addon/full/d4/double_well/seed1703/vjepa2_ac --ac-checkpoint ../../checkpoints/vjepa2-ac-vitg.pt",
        "python run.py spectral-gap --system double_well --trajectories 2000 --updates 10000 --features-root ../../features-v7/double_well-2000 --device cpu --output ../../results-addon/full/spectral-gap/double_well",
        "python tools/summarize_addon.py --v7-results ../../results-v7/full --addon-results ../../results-addon/full --output ../../results-addon/full/summary",
        "python run.py calibrate-addon --system landscape --variant shape=multi_well --trajectories 2000 --updates 10000 --features-root ../../features-v7/landscape~shape=multi_well-2000 --device cuda:0 --output ../../results-addon/full/calibrate-landscape~shape=multi_well",
    ):
        assert line in full, line
    for line in (
        "python run.py transfer --system double_well --trajectories 2000 --updates 10000 --seed 1703 --condition fixed_fourier --test-variant coupling=0.5 --training-run ../../results-addon/full/double_well/seed1703/fixed_fourier --features-root ../../features-v7/double_well~coupling=0.5-2000 --device cuda:0 --output ../../results-addon/full/transfer/double_well~coupling=0.5/fixed_fourier",
        "python tools/plan_v7.py --job ../../results-addon/full/double_well/seed1703/fixed_fourier --upstream-root ../../vjepa2 --checkpoint ../../checkpoints/vjepa2-vitg-fpc64-256.pth --episodes 100 --output ../../results-addon/full/plan/double_well/fixed_fourier.json --device cuda:0 --no-baselines",
        "python tools/summarize_addon.py --v7-results ../../results-v7/full --addon-results ../../results-addon/full --output ../../results-addon/full/summary",
    ):
        assert line in variants, line
    assert ("python tools/plan_v7.py --job ../results-v7/full/double_well/seed1703/vjepa2_ac --upstream-root ../vjepa2 --checkpoint ../checkpoints/vjepa2-vitg-fpc64-256.pth --episodes 100 --output ../results-addon/full/plan-ac256/double_well/vjepa2_ac.json --device cuda:0 --ac-checkpoint ../checkpoints/vjepa2-ac-vitg.pt --samples 256 --no-baselines"
            in v7tree)


@pytest.mark.parametrize("stage", ("pilot", "full"))
def test_main_stage_paths(stage, tmp_path):
    n, updates, seeds = dict(pilot=(200, 2000, (1703,)), full=(2000, 10000, (1703, 1704, 1705)))[stage]
    names = list(SCENES)
    root = lambda s: V6 if (stage, s) == ("pilot", "double_well") else f"../../features-v7/{s}-{n}"
    feature = lambda argv: {root(s): s for s in names}[opt(argv, "--features-root")]
    by = {s["name"]: [shlex.split(c) for c in s["commands"]] for s in sections(stage, tmp_path)}

    def opt(argv, key):
        return argv[argv.index(key)+1] if key in argv else None

    for argv in itertools.chain(*by.values()):
        if argv[1] == "run.py":
            assert opt(argv, "--trajectories") == str(n) and opt(argv, "--updates") == str(updates)
            system, _, knobs = feature(argv).partition("~")
            assert opt(argv, "--system") == system and opt(argv, "--variant") == (knobs or None)
            assert opt(argv, "--device") == ("cpu" if argv[2] == "spectral-gap" else "cuda:0")
            assert opt(argv, "--output").startswith(f"../../results-addon/{stage}/")
    assert {opt(a, "--features-root") for a in by["C"]} == {root(s) for s in names}
    d = {(opt(a, "--features-root"), int(opt(a, "--seed")), opt(a, "--condition")): opt(a, "--output") for a in by["D"]}
    assert d == {(root(s), seed, c): f"../../results-addon/{stage}/{s}/seed{seed}/{c}"
                 for s in names for seed in seeds for c in ADDON}
    d4 = [a for a in by["D4"] if a[2] == "d4"]
    assert {(opt(a, "--features-root"), int(opt(a, "--seed")), opt(a, "--condition")) for a in d4} == {
        (root(s), seed, c) for s in names for seed in seeds for c in (*V7, *ADDON)}
    for a in d4:
        scene, seed, c = feature(a), opt(a, "--seed"), opt(a, "--condition")
        origin = "results-v7" if c in V7 else "results-addon"
        assert opt(a, "--training-run") == f"../../{origin}/{stage}/{scene}/seed{seed}/{c}"
        assert opt(a, "--output") == f"../../results-addon/{stage}/d4/{scene}/seed{seed}/{c}"
        assert (opt(a, "--ac-checkpoint") == jobs.AC) == (c == "vjepa2_ac") and (opt(a, "--ac-checkpoint") is None) == (c != "vjepa2_ac")
    assert [opt(a, "--output") for a in by["D4"] if a[2] == "spectral-gap"] == [f"../../results-addon/{stage}/spectral-gap/{s}" for s in names]
    text = "\n".join(lines(stage))
    if stage == "pilot":     # double_well 的 C 1、D 5、D4 11、谱间隙 1 条读 v6 目录，preflight 也拿到它
        assert "features-v7/double_well" not in text and text.count(f"--features-root {V6} ") == 18
        assert f"--features ../../features-v7 --v6-features {V6} --stage pilot" in text and "results-v7/pilot" in text
    else:
        assert "--v6-features" not in text and V6 not in text


def test_variant_paths_follow_v7_names(tmp_path):
    from make_v7_jobs import TEST_VARIANTS
    v3 = [shlex.split(c) for c in sections("variants", tmp_path)[0]["commands"]]
    opt = lambda argv, key: argv[argv.index(key)+1]
    expected = {(s.partition("~")[0], t, c, f"../../features-v7/{jobs.name(s, t)}-2000",
                 f"../../results-addon/full/transfer/{jobs.name(s, t)}/{c}", f"../../results-addon/full/{jobs.name(s)}/seed1703/{c}")
                for s in SCENES for t in TEST_VARIANTS[s.partition("~")[0]] for c in ADDON}
    found = [(opt(a, "--system"), opt(a, "--test-variant"), opt(a, "--condition"), opt(a, "--features-root"), opt(a, "--output"),
              opt(a, "--training-run")) for a in v3]
    assert len(found) == len(set(found)) == 150 and set(found) == expected
    assert jobs.name("landscape~shape=multi_well", "depth=2") == "landscape~depth=2,shape=multi_well"


@pytest.mark.parametrize("stage", tuple(COUNTS))
def test_output_is_deterministic(stage):
    extra = ["--v6-features", V6] if stage == "pilot" else []
    run = lambda: subprocess.run([sys.executable, "-B", str(ROOT/"tools/make_addon_jobs.py"), "--stage", stage, *extra],
                                 capture_output=True, check=True).stdout
    first = run()
    assert first == run() and first.decode("utf-8") == "\n".join(lines(stage))+"\n"


def accepts(entry, command, capsys):
    """run.py 的子命令选项里有没有 command：有就打印帮助、退出码 0，没有就是“无效的选项”、退出码 2。"""
    with pytest.raises(SystemExit) as error:
        entry.main([command, "--help"])
    capsys.readouterr()
    assert error.value.code in (0, 2)
    return error.value.code == 0


def test_every_command_is_accepted_without_running(tmp_path, monkeypatch, capsys):
    """每条 run.py 命令过 run.py 的参数检查（子进程换成记录器，不执行）；本段的两个工具和 plan_v7.py 过各自的参数解析。
    d4、spectral-gap 是 S18 的子命令：run.py 已经有就要求通过，还没有就要求恰好报“无效的子命令”。"""
    work = tmp_path/"full-model"/"po-jepa-full"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    for name in ("vjepa2-vitg-fpc64-256.pth", "vjepa2-ac-vitg.pt"):
        (tmp_path/"checkpoints").mkdir(exist_ok=True)
        (tmp_path/"checkpoints"/name).write_bytes(b"")
    entry = load_run_py()
    s18 = accepts(entry, "d4", capsys)                                 # S18 合并以后 run.py 才有 d4、spectral-gap
    assert accepts(entry, "spectral-gap", capsys) == s18
    seen = []
    monkeypatch.setattr(entry.subprocess, "run", lambda command, **kw: seen.append(command) or Namespace(returncode=0))
    import addon_preflight
    import import_v7_calibration
    import plan_v7
    monkeypatch.setattr(import_v7_calibration, "import_calibration", lambda *a, **k: {})
    monkeypatch.setattr(addon_preflight, "preflight", lambda *a, **k: dict(scenes={}, checked_jobs=0))
    monkeypatch.setattr(addon_preflight, "write_once", lambda *a, **k: "written")
    counted = dict(run=0, tool=0, s18=0)
    v6 = str(tmp_path/"v6-features")                                   # 绝对路径，但在临时目录里（下面会建出来）
    listing = {st: jobs.job_lines(st, v6 if st == "pilot" else "") for st in ("pilot", "full", "variants")}
    for stage in ("pilot", "full", "variants"):
        for line in listing[stage]:
            if line.startswith("#"):
                continue
            argv = shlex.split(line)
            assert argv[0] == "python"
            for key in ("--features-root", "--training-run"):     # run.py 要求特征目录存在；S18 的子命令也许要求训练目录存在
                if key in argv:
                    Path(argv[argv.index(key)+1]).mkdir(parents=True, exist_ok=True)
            if argv[1] == "run.py":
                seen.clear()
                if argv[2] in ("d4", "spectral-gap") and not s18:
                    with pytest.raises(SystemExit) as error:
                        entry.main(argv[2:])
                    assert error.value.code == 2
                    counted["s18"] += 1
                    continue
                assert entry.main(argv[2:]) == 0, line
                child = seen[0][seen[0].index("--")+1:]
                module = child[child.index("-m")+1]
                if argv[2] in ("calibrate-addon", "check-addon"):
                    assert module == "po_jepa.runtime.addon.entry"
                elif argv[2] not in ("d4", "spectral-gap"):            # S18 的两个子命令交给 S18 定的模块
                    assert module == "po_jepa.runtime.cli"
                counted["run"] += 1
            elif argv[1] == "tools/import_v7_calibration.py":
                assert import_v7_calibration.main(argv[2:]) == 0
                counted["tool"] += 1
            elif argv[1] == "tools/addon_preflight.py":
                assert addon_preflight.main(argv[2:]) == 0
                counted["tool"] += 1
            elif argv[1] == "tools/plan_v7.py":
                monkeypatch.setattr(sys, "argv", argv[1:])
                parsed = plan_v7._parse()
                assert parsed.no_baselines and parsed.samples == 256 and parsed.elites == 24 and parsed.job.name in ADDON
                counted["tool"] += 1
            else:
                assert argv[1] in ("tools/encoder_identity.py", "tools/summarize_addon.py")   # S18 的工具
    total_run = sum(1 for st in listing for line in listing[st] if line.startswith("python run.py"))
    assert counted["run"]+counted["s18"] == total_run == 5+1+25+60+5+1+75+170+150


def test_v6_features_option(capsys):
    """--v6-features（Jerry 2026-11-08）：试跑必须给、要写绝对路径，其他阶段不能给；路径原样写进 double_well-200 的命令和 preflight。"""
    for argv, message in ((["--stage", "pilot"], "试跑必须给 --v6-features"),
                          (["--stage", "full", "--v6-features", V6], "--v6-features 只用于 --stage pilot"),
                          (["--stage", "variants", "--v6-features", V6], "--v6-features 只用于 --stage pilot"),
                          (["--stage", "pilot", "--v6-features", "../../po-jepa/features"], "--v6-features 要写绝对路径")):
        with pytest.raises(SystemExit) as error:
            jobs.main(argv)
        assert error.value.code == 2 and message in capsys.readouterr().err
    with pytest.raises(ValueError, match="^试跑必须给 --v6-features"):
        jobs.job_lines("pilot")
    pilot = jobs.job_lines("pilot", V6)
    for line in (
        "python tools/addon_preflight.py --v7-results ../../results-v7/pilot --features ../../features-v7 --v6-features /data/v6-features --stage pilot --output ../../results-addon/pilot/preflight.json",
        "python run.py calibrate-addon --system double_well --trajectories 200 --updates 2000 --features-root /data/v6-features --device cuda:0 --output ../../results-addon/pilot/calibrate-double_well",
        "python run.py run --system double_well --trajectories 200 --updates 2000 --seed 1703 --condition koopman --features-root /data/v6-features --device cuda:0 --output ../../results-addon/pilot/double_well/seed1703/koopman",
        "python run.py d4 --system double_well --trajectories 200 --updates 2000 --seed 1703 --condition po_jepa_full --training-run ../../results-v7/pilot/double_well/seed1703/po_jepa_full --features-root /data/v6-features --device cuda:0 --output ../../results-addon/pilot/d4/double_well/seed1703/po_jepa_full",
        "python run.py spectral-gap --system double_well --trajectories 200 --updates 2000 --features-root /data/v6-features --device cpu --output ../../results-addon/pilot/spectral-gap/double_well",
        "python run.py calibrate-addon --system van_der_pol --trajectories 200 --updates 2000 --features-root ../../features-v7/van_der_pol-200 --device cuda:0 --output ../../results-addon/pilot/calibrate-van_der_pol",
        "python run.py check-addon --system van_der_pol --trajectories 200 --updates 2000 --features-root ../../features-v7/van_der_pol-200 --device cuda:0 --output ../../results-addon/pilot/check",
    ):
        assert line in pilot, line
    assert sum(V6 in line for line in pilot) == 19
    assert [jobs.uses_v6(s, n, V6) for s in ("double_well", "van_der_pol", "double_well~gain=2") for n in (200, 2000)] == [
        True, False, False, False, False, False]
    assert not jobs.uses_v6("double_well", 200, "")


def test_v6_features_reads_the_same_directories_as_make_v7_jobs():
    """同一个 --v6-features，补跑试跑每个场景的 --features-root 和 v7 试跑的相同（v7 的相对路径以 po-jepa 为起点，换成补跑的起点后比）。"""
    v7 = subprocess.run([sys.executable, "-B", str(ROOT/"tools/make_v7_jobs.py"), "--stage", "pilot", "--v6-features", V6],
                        capture_output=True, text=True, check=True).stdout.splitlines()

    def roots(job_lines, translate):
        found = {}
        for line in job_lines:
            argv = shlex.split(line)
            if line.startswith("python run.py") and "--features-root" in argv:
                scene = argv[argv.index("--system")+1]+("~"+argv[argv.index("--variant")+1] if "--variant" in argv else "")
                found.setdefault(scene, set()).add(translate(argv[argv.index("--features-root")+1]))
        return found
    expected = roots(v7, lambda path: path.replace("../features-v7/", "../../features-v7/", 1))
    assert expected["double_well"] == {V6} and set(expected) == set(SCENES)
    assert roots(jobs.job_lines("pilot", V6), lambda path: path) == expected
    assert not any("encode_v7.py" in line and "double_well" in line for line in v7)


# ---------- run.py ----------

def test_run_py_routes_addon_commands(tmp_path, monkeypatch):
    entry = load_run_py()
    seen = []
    monkeypatch.setattr(entry.subprocess, "run", lambda command, **kw: seen.append((command, kw["env"])) or Namespace(returncode=0))
    features = tmp_path/"features"
    features.mkdir()
    base = ["--features-root", str(features), "--system", "van_der_pol", "--trajectories", "200", "--updates", "2000"]
    for command in ("calibrate-addon", "check-addon"):
        seen.clear()
        assert entry.main([command, *base, "--output", str(tmp_path/command)]) == 0
        argv, env = seen[0]
        child = argv[argv.index("--")+1:]
        assert child[1:5] == ["-B", "-X", "utf8", "-m"] and child[5:7] == ["po_jepa.runtime.addon.entry", command]
        assert child[7:] == ["--features-root", str(features.resolve()), "--device", "cuda:0", "--output",
                             str((tmp_path/command).resolve()/"run"), "--seed", "1703"]
        assert env["PO_JEPA_PROTOCOL"] == "v7" and env["PO_JEPA_DATASET"] == "van_der_pol:200" and env["PO_JEPA_UPDATES"] == "2000"
        for extra in (["--seed", "1704"], ["--condition", "koopman"], ["--training-run", str(tmp_path)], ["--test-variant", "gain=2"]):
            with pytest.raises(SystemExit) as error:
                entry.main([command, *base, "--output", str(tmp_path/"x"), *extra])
            assert error.value.code == 2
    seen.clear()
    assert entry.main(["check", *base, "--output", str(tmp_path/"check")]) == 0
    assert seen[0][0][seen[0][0].index("-m")+1] == "po_jepa.runtime.cli"
    with pytest.raises(SystemExit):
        entry.main(["fictional-addon", *base, "--output", str(tmp_path/"y")])


# ---------- run_jobs.py 复制件 ----------

def test_runner_defaults_and_lists():
    text = (ROOT/"tools/run_jobs.py").read_text(encoding="utf-8")
    assert 'logs = a.logs or workdir.parent.parent/"logs-addon"/a.jobs.stem' in text
    assert 'raise SystemExit("请在 full-model/po-jepa-full 目录里运行")' in text
    assert ('elif sub in ("calibrate", "calibrate-addon", "check", "check-addon", "transfer", "evaluate", "d4", "spectral-gap"):'
            in text)
    expected = dict(vjepa2_ac=0, vjepa2_ac_small=1, po_jepa_full=1, po_jepa_no_consistency=1, matched_transformer=1,
                    po_jepa_time_only=1, po_jepa_space_only=1, transformer_consistency=1, fixed_fourier=1, koopman=2, identity=2)
    for condition, value in expected.items():
        assert run_jobs.priority(f"python run.py run --condition {condition} --output x") == value
        assert run_jobs.priority(f"python run.py d4 --condition {condition} --training-run t --output x") == value
    assert run_jobs.priority("python run.py calibrate-addon --output x") == 2
    assert run_jobs.priority("python run.py run --condition vjepa2_ac --output x --ac-checkpoint a") == 0


def test_runner_default_log_directory_and_workdir_check(tmp_path):
    work = tmp_path/"full-model"/"po-jepa-full"
    work.mkdir(parents=True)
    listing = tmp_path/"jobs-addon-fictional.txt"
    listing.write_text("# ===== A x\npython -c 1\n", encoding="utf-8")
    run = lambda cwd: subprocess.run([sys.executable, "-B", str(ROOT/"tools/run_jobs.py"), str(listing), "--dry-run"],
                                     cwd=cwd, capture_output=True, text=True)
    missing = run(work)
    assert missing.returncode == 1 and "请在 full-model/po-jepa-full 目录里运行" in missing.stderr
    (work/"run.py").write_text("", encoding="utf-8")
    listed = run(work)
    assert listed.returncode == 0 and "[A-000] python -c 1" in listed.stdout
    assert (tmp_path/"logs-addon"/"jobs-addon-fictional").is_dir() and not (tmp_path/"full-model"/"logs-v7").exists()


@pytest.mark.parametrize("sub", ("calibrate-addon", "check-addon", "d4", "spectral-gap", "transfer", "calibrate", "check", "evaluate"))
def test_prepare_renames_or_skips(tmp_path, sub):
    cmd = f"python run.py {sub} --condition koopman --output r/x"
    assert run_jobs.prepare(cmd, tmp_path) == (cmd, "")
    (tmp_path/"r/x/run").mkdir(parents=True)                       # 半截：改名后重跑
    assert run_jobs.prepare(cmd, tmp_path)[0] == cmd and (tmp_path/"r/x.broken1").is_dir()
    (tmp_path/"r/x/run").mkdir(parents=True)
    (tmp_path/"r/x/run/COMPLETE.json").write_text("{}")             # <输出>/run/COMPLETE.json：已完成
    assert run_jobs.prepare(cmd, tmp_path) == (None, "已完成")
    assert not (tmp_path/"r/x.broken2").exists()


def test_complete_accepts_run_directory_and_v7_layouts(tmp_path):
    for sub in ("", "run/job", "job", "run"):
        path = tmp_path/("p"+sub.replace("/", "_"))
        (path/sub).mkdir(parents=True, exist_ok=True)
        assert not run_jobs.complete(path)
        (path/sub/"COMPLETE.json").write_text("{}")
        assert run_jobs.complete(path)
    later = "python run.py d4 --condition koopman --training-run r/k --output d/k"
    (tmp_path/"r/k/run/job").mkdir(parents=True)
    (tmp_path/"r/k-resume1/run").mkdir(parents=True)
    (tmp_path/"r/k-resume1/run/COMPLETE.json").write_text("{}")
    assert "--training-run r/k-resume1" in run_jobs.prepare(later, tmp_path)[0]


FAKE_RUN = '''import sys
from pathlib import Path
argv = sys.argv[1:]
out = Path(argv[argv.index("--output")+1])
with open("calls.txt", "a", encoding="utf-8") as log:
    log.write(" ".join(argv)+"\\n")
(out/"run"/"job").mkdir(parents=True)
(out/"run"/"job"/"job.json").write_text("{}")
if Path("crash-"+argv[0]).exists():
    sys.exit(9)
(out/"run"/"job"/"COMPLETE.json").write_text("{}")
(out/"run"/"COMPLETE.json").write_text("{}")
'''


def test_runner_resumes_run_renames_d4_and_skips_done(tmp_path):
    """假的 run.py：第一次 run 和 d4 中途退出（留下半截目录），calibrate-addon 完成；重跑同一条命令后 run 改成续训、
    d4 的半截目录改名、calibrate-addon 跳过；下一段的 --training-run 换成续训目录。"""
    work = tmp_path/"full-model"/"po-jepa-full"
    work.mkdir(parents=True)
    (work/"run.py").write_text(FAKE_RUN, encoding="utf-8")
    py = shlex.quote(sys.executable)
    listing = work/"jobs-addon-fictional.txt"
    listing.write_text("# ===== A x\n"
                       f"{py} run.py run --condition koopman --output r/k\n"
                       f"{py} run.py d4 --condition koopman --training-run r/t --output r/d\n"
                       f"{py} run.py calibrate-addon --output r/c\n"
                       "# ===== B y\n"
                       f"{py} run.py d4 --condition koopman --training-run r/k --output r/e\n", encoding="utf-8")
    (work/"r/t/run").mkdir(parents=True)
    (work/"r/t/run/COMPLETE.json").write_text("{}")
    for name in ("run", "d4"):
        (work/f"crash-{name}").write_text("")
    run = lambda: subprocess.run([sys.executable, "-B", str(ROOT/"tools/run_jobs.py"), str(listing), "--gpus", "0"],
                                 cwd=work, capture_output=True, text=True)
    first = run()
    assert first.returncode == 1 and "第 A 段有任务失败" in first.stdout
    assert not (work/"r/e").exists() and (tmp_path/"logs-addon/jobs-addon-fictional").is_dir()
    for name in ("run", "d4"):
        (work/f"crash-{name}").unlink()
    second = run()
    assert second.returncode == 0 and "全部完成。" in second.stdout
    assert (work/"calls.txt").read_text(encoding="utf-8").splitlines() == [
        "run --condition koopman --output r/k", "d4 --condition koopman --training-run r/t --output r/d",
        "calibrate-addon --output r/c",                                                     # 第一次：前两条中途退出
        "resume --condition koopman --output r/k-resume1 --resume-from r/k",                # 第二次：续训
        "d4 --condition koopman --training-run r/t --output r/d",                           # 半截目录改名后重跑
        "d4 --condition koopman --training-run r/k-resume1 --output r/e"]                   # 下一段改指向续训目录
    assert (work/"r/d.broken1/run/job/job.json").exists() and (work/"r/d/run/COMPLETE.json").exists()
    assert (work/"r/k-resume1/run/COMPLETE.json").exists() and not (work/"r/k/run/COMPLETE.json").exists()


def test_v6_features_path_is_shell_quoted():
    """Codex 评审（2026-11-09）：--v6-features 的路径逐个参数 shell 引用，含空格的绝对路径经 shlex.split（run_jobs 的读法）后仍是一个参数。"""
    spaced = "/srv/v6 data/features"
    lines = [line for line in jobs.job_lines("pilot", spaced) if not line.startswith("#")]
    hits = 0
    for line in lines:
        argv = shlex.split(line)
        for flag in ("--features-root", "--v6-features"):
            if flag in argv and argv[argv.index(flag)+1].startswith("/srv"):
                assert argv[argv.index(flag)+1] == spaced, line
                hits += 1
    assert hits >= 4
    assert jobs.job_lines("pilot", V6) == [line.replace(shlex.quote(spaced), V6) for line in jobs.job_lines("pilot", spaced)]
