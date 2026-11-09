"""S18a 的端到端冒烟（开发机专用，不被 pytest 收集）：v7 树训练、评测出来的任务，在新树里做 D4 只评测。

  cd po-jepa-full && python -B tests/addon_d4_smoke.py --workdir /tmp/s18a-d4-smoke                 # 完整跑一遍（约 3 小时）
  cd po-jepa-full && python -B tests/addon_d4_smoke.py --workdir /tmp/s18a-d4-smoke --reuse /tmp/s10b-dual-tree
--reuse 用一次已有的双树对照（S00 tests/addon_dual_tree.py）的工作目录里的 features/、fake-ac/、v7/（只读），省掉前三步。

步骤（每步一个子进程，环境同 addon_dual_tree.WORKER_ENV：1 线程、PYTHONHASHSEED=0）：
1. 假特征：v7 树的 tools/encode_v7.py（van_der_pol:40，--fake-encoder）→ <workdir>/features。
2. 缩小的“官方”AC 假权重：addon_dual_tree.py --make-ac-checkpoint → <workdir>/fake-ac/vjepa2-ac-reduced.pt。
3. v7 树：addon_dual_tree.py --worker --tree <v7 树>（6 个 v7 条件各训练 20 步再评测，另有 5 个只评测任务）→ <workdir>/v7。
4. 新树 --addon-worker：同一份缩小协议，5 个新条件各训练 20 步再评测 → <workdir>/addon/jobs/<条件>。补跑标定用冒烟值
   （两项的 λ 都取 v7 冒烟标定的 λ），写在 <workdir>/addon/addon-calibration.yaml。
5. 新树 --d4-worker：11 个条件各做一次 D4（v7 条件读第 3 步的任务，新条件读第 4 步的任务），po_jepa_full 另对只评测任务
   po_jepa_full-evaluate 做一次（核沿 training_run 找训练目录）；截断间隙；v7load 在真实 v7 检查点上的 7 个改坏核对；
   改动参照记录后等价核对必须失败、改一个最低位必须通过。结果写 <workdir>/d4/d4-smoke.json。
最后一行打印 “D4 冒烟：通过（…）” 或 “D4 冒烟：失败（…）”；退出码 0 / 1。
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

import addon_dual_tree as dual

HERE = Path(__file__).resolve().parent
NEW_TREE = HERE.parent
V7_TREE = NEW_TREE.parent
ADDON = ("po_jepa_time_only", "po_jepa_space_only", "transformer_consistency", "fixed_fourier", "koopman")
UPDATES = 20
SMOKE_ENV = {"PO_JEPA_PROTOCOL": "v7", "PO_JEPA_DATASET": "van_der_pol:40", "PO_JEPA_UPDATES": "2000", "PO_JEPA_SEED": "1703"}
STRIPPED = ("PO_JEPA_", "OMP_", "MKL_", "KMP_")
MUTATIONS = ("checkpoint_bytes", "completed_steps", "model_config", "loss_config", "model_state", "data_field", "source_files_sha256")


def environment():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith(STRIPPED)}
    env.update(dual.WORKER_ENV)
    return env


def step(name, command, cwd, log):
    print(f"{name} …", flush=True)
    started = time.perf_counter()
    with open(log, "xb") as handle:
        code = subprocess.run(command, cwd=cwd, env=environment(), stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT).returncode
    if code:
        tail = "\n".join(Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise SystemExit(f"{tail}\nD4 冒烟：失败（{name} 退出码 {code}，见 {log}）")
    print(f"{name} 完成（{time.perf_counter()-started:.0f} 秒）", flush=True)


def parent(workdir, reuse):
    workdir = Path(workdir).absolute()
    if workdir.exists() or workdir.resolve().is_relative_to(V7_TREE.resolve()):
        raise SystemExit(f"--workdir 必须不存在，且不在仓库里：{workdir}")
    workdir.mkdir(parents=True)
    py = sys.executable
    if reuse is None:
        source = workdir
        step("假特征", [py, "-B", str(V7_TREE/"tools/encode_v7.py"), "encode", "--system", "van_der_pol", "--trajectories", "40",
                       "--output", str(workdir/"features"), "--device", "cpu", "--fake-encoder"], V7_TREE, workdir/"encode.log")
        step("假特征 finalize", [py, "-B", str(V7_TREE/"tools/encode_v7.py"), "finalize", "--system", "van_der_pol", "--trajectories", "40",
                                "--output", str(workdir/"features")], V7_TREE, workdir/"finalize.log")
        step("AC 假权重", [py, "-B", str(HERE/"addon_dual_tree.py"), "--make-ac-checkpoint", str(workdir/"fake-ac/vjepa2-ac-reduced.pt"),
                          "--tree", str(V7_TREE)], NEW_TREE, workdir/"make-ac.log")
        (workdir/"v7").mkdir()
        step("v7 树（约 1 小时 45 分）", [py, "-B", str(HERE/"addon_dual_tree.py"), "--worker", "--tree", str(V7_TREE), "--features",
                                       str(workdir/"features"), "--ac-checkpoint", str(workdir/"fake-ac/vjepa2-ac-reduced.pt"),
                                       "--output", str(workdir/"v7")], V7_TREE, workdir/"v7/console.log")
    else:
        source = Path(reuse).resolve()
    features, checkpoint, v7 = source/"features", source/"fake-ac/vjepa2-ac-reduced.pt", source/"v7"
    for c in (*dual.JOB_ORDER, "po_jepa_full-evaluate"):
        if not (v7/"smoke/jobs"/c/"COMPLETE.json").exists():
            raise SystemExit(f"D4 冒烟：失败（{v7/'smoke/jobs'/c} 没有完成）")
    common = ["--features", str(features), "--ac-checkpoint", str(checkpoint), "--v7", str(v7)]
    (workdir/"addon").mkdir()
    step("新树：5 个新条件（约 40 分）", [py, "-B", str(Path(__file__).resolve()), "--addon-worker", *common, "--output", str(workdir/"addon")],
         NEW_TREE, workdir/"addon/console.log")
    (workdir/"d4").mkdir()
    step("新树：D4（约 1 小时）", [py, "-B", str(Path(__file__).resolve()), "--d4-worker", *common, "--addon", str(workdir/"addon"),
                                 "--output", str(workdir/"d4")], NEW_TREE, workdir/"d4/console.log")
    result = json.loads((workdir/"d4/d4-smoke.json").read_text(encoding="utf-8"))
    print(result["summary_line"])
    return 0 if result["passed"] else 1


# ---------------------------------------------------------------- 新树里的子进程

def enter_new_tree(ac_checkpoint):
    """导入 po_jepa 之前定好环境、路径、线程；真的核一次清单；打 AC 缩小补丁和冒烟补丁（和 v7 树的 smoke_v7_cpu.py --updates 20 相同）。"""
    os.environ.update(SMOKE_ENV)
    os.environ["PO_JEPA_AC_CHECKPOINT"] = str(Path(ac_checkpoint).resolve())
    if Path(sys.path[0] or os.curdir).resolve() != HERE:
        raise RuntimeError("sys.path[0] 应是本脚本所在目录")
    sys.path[0] = str(NEW_TREE/"src")
    torch.set_num_threads(dual.THREADS)
    if torch.get_num_threads() != 1 or any(os.environ.get(k) != v for k, v in dual.WORKER_ENV.items()):
        raise RuntimeError("子进程环境和 WORKER_ENV 不同（请通过父进程运行）")
    from po_jepa.runtime import cache, checks, workflow
    bundle = workflow.verify_bundle()
    from po_jepa import ac_baseline
    ac_baseline.OFFICIAL = ac_baseline.ACSpec(**dual.REDUCED_OFFICIAL)
    original = cache._read_v7_cache
    def as_real(path, m, sha):
        m = copy.deepcopy(m); g = m["generation"]
        g.update(evidence_role="real_frozen_encoder", encoder_checkpoint_sha256=cache.CHECKPOINT_SHA,
                 encoder_load_report=dict(sha256=cache.CHECKPOINT_SHA, encoder_key="target_encoder", strict=True,
                                          missing_keys=[], unexpected_keys=[]))
        return original(path, m, sha)
    cache._read_v7_cache = as_real
    full = checks.load_protocol
    def tiny():
        c = copy.deepcopy(full())
        c["training"]["updates_per_condition"] = UPDATES
        c["loss"].update(lr_decay_updates=UPDATES, lr_warmup_updates=1, consistency_warmup_updates=1)
        c["validation"]["interval_updates"] = max(1, UPDATES//2)
        c["monitoring"].update(early_check_updates=10**9, consistency_interval=2)
        c["calibration"].update(pretrain_updates=2, batches=2)
        return c
    checks.load_protocol = tiny
    workflow.verify_bundle = lambda: "smoke"
    return bundle


def bind_calibrations(v7, addon_output):
    from po_jepa.runtime import checks
    from po_jepa.runtime.addon import calibration
    v7_calibration = Path(v7).resolve()/"smoke/calibration.yaml"
    checks.v7_calibration_path = lambda: v7_calibration
    addon_calibration = Path(addon_output).resolve()/"addon-calibration.yaml"
    calibration.addon_calibration_path = lambda: addon_calibration
    return v7_calibration, addon_calibration


def addon_worker(a):
    started = time.perf_counter()
    bundle = enter_new_tree(a.ac_checkpoint)
    import yaml
    from po_jepa.runtime import cache, checks, v7jobs, workflow
    from po_jepa.runtime.addon import calibration
    v7_calibration, addon_calibration = bind_calibrations(a.v7, a.output)
    v = yaml.safe_load(v7_calibration.read_text(encoding="utf-8"))
    entry = dict(rho=checks.load_protocol()["calibration"]["rho"], time_weight=v["time_weight"], space_weight=v["space_weight"], source="d4 smoke")
    addon_calibration.write_text(yaml.safe_dump(dict(fixed_fourier=entry, transformer_consistency=dict(entry),
                                                     sigma_z_used=checks.load_protocol()["model"]["graph"]["sigma_z"],
                                                     v7_calibration_sha256=cache.file_sha(v7_calibration)), sort_keys=True), encoding="utf-8")
    workflow.require_calibrated_weights(swept=True)
    calibration.require_addon_calibration()
    seconds = {}
    for c in ADDON:
        t0 = time.perf_counter()
        v7jobs.run_job(a.features.resolve(), a.output.resolve()/"jobs"/c, torch.device("cpu"), c)
        seconds[c] = time.perf_counter()-t0
        print(f"{c} {seconds[c]:.0f} s", flush=True)
    cache._write_json(a.output/"worker.json", dict(tree=str(NEW_TREE), verify_bundle_sha256=bundle, seconds=seconds,
                                                   total_seconds=time.perf_counter()-started, threads=torch.get_num_threads()))


def _reseal(folder):
    from po_jepa.runtime import cache
    (folder/"COMPLETE.json").unlink()
    cache._write_json(folder/"COMPLETE.json", dict(report_sha256=cache.file_sha(folder/"report.json")))


def _rewrite(path, change):
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.unlink()
    from po_jepa.runtime import cache
    cache._write_json(path, value)


def mutate(job, case):
    """在 job（po_jepa_full 任务的副本）上做一种改坏；改完把报告、选点记录、COMPLETE.json 重新封好，只让目标那一项不符。"""
    train = job/"training"/"po_jepa_full"
    best = json.loads((train/"report.json").read_text(encoding="utf-8"))["best"]
    path = train/best["checkpoint"]
    if case == "checkpoint_bytes":
        path.write_bytes(path.read_bytes()+b"\0")
        return
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if case == "completed_steps":
        payload["completed_steps"] += 1
    elif case == "model_config":
        payload["model_config"]["graph"]["sigma_z"] *= 2
    elif case == "loss_config":
        payload["loss_config"]["time_weight"] *= 2
    elif case == "model_state":
        name = next(iter(payload["model_state"]))
        payload["model_state"]["renamed."+name] = payload["model_state"].pop(name)
    elif case == "data_field":
        payload["provenance"]["feature_manifest_sha256"] = "0"*64
    elif case == "source_files_sha256":
        payload["provenance"]["source_files_sha256"] = {k: "0"*64 for k in payload["provenance"]["source_files_sha256"]}
    path.unlink()
    torch.save(payload, path)
    from po_jepa.runtime import cache
    sha = cache.file_sha(path)
    def fix(record):
        if record.get("step") == best["step"]:
            record["checkpoint_sha256"] = sha
            record["checkpoint_provenance"] = payload["provenance"]
    def fix_training(report):
        fix(report["best"])
        for row in report["validations"]:
            fix(row)
    _rewrite(train/"report.json", fix_training)
    _reseal(train)
    _rewrite(job/"report.json", lambda r: fix(r["best"]))
    _reseal(job)


EXPECTED = dict(checkpoint_bytes="checkpoint SHA mismatch before loading", completed_steps="completed_steps differs",
                model_config="condition/model_config/loss_config", loss_config="condition/model_config/loss_config",
                model_state="model state keys mismatch", data_field="data provenance differs from the current features")


def d4_worker(a):
    started = time.perf_counter()
    bundle = enter_new_tree(a.ac_checkpoint)
    from po_jepa.runtime import cache, comparison
    from po_jepa.runtime.addon import d4, spectral_gap, v7load
    bind_calibrations(a.v7, a.addon)
    features, out = a.features.resolve(), a.output.resolve()
    device = torch.device("cpu")
    jobs = {c: Path(a.v7).resolve()/"smoke/jobs"/c for c in dual.JOB_ORDER}
    jobs["po_jepa_full-evaluate"] = Path(a.v7).resolve()/"smoke/jobs/po_jepa_full-evaluate"
    jobs.update({c: Path(a.addon).resolve()/"jobs"/c for c in ADDON})
    result, problems = dict(jobs={}, mutations={}, perturbation={}), []
    for name, job in jobs.items():
        condition = name.removesuffix("-evaluate")
        t0 = time.perf_counter()
        report = d4.run_d4(features, out/"jobs"/name, device, condition, training_run=job)
        source = json.loads((out/"jobs"/name/"source.json").read_text(encoding="utf-8"))
        eq = report["equivalence"]["metrics"]
        entry = dict(seconds=time.perf_counter()-t0, status=report["status"], counts=report["counts"],
                     equivalence={m: (s["count"], s["bitwise_equal"], s["max_abs_diff"]) for m, s in eq.items()},
                     triangle_violations=report["triangle"]["violations"], training_dir=source["training_dir"])
        result["jobs"][name] = entry
        print(f"d4 {name} {entry['seconds']:.0f} s {entry['equivalence']}", flush=True)
        if report["status"] != "passed" or report["counts"] != dict(windows=20, long_trajectories=4, test_trajectories=4):
            problems.append(f"{name}: status/counts")
        if any(count != bitwise for count, bitwise, _ in entry["equivalence"].values()):
            problems.append(f"{name}: equivalence not bitwise")
        if name == "po_jepa_full-evaluate" and source["training_dir"] != str((jobs["po_jepa_full"]/"training/po_jepa_full").resolve()):
            problems.append("po_jepa_full-evaluate: linked training directory")
    full = json.loads((out/"jobs/po_jepa_full/report.json").read_text(encoding="utf-8"))
    linked = json.loads((out/"jobs/po_jepa_full-evaluate/report.json").read_text(encoding="utf-8"))
    if full["trajectory_macro_means"] != linked["trajectory_macro_means"] or full["long_means"] != linked["long_means"]:
        problems.append("po_jepa_full and po_jepa_full-evaluate differ")
    t0 = time.perf_counter()
    gap = spectral_gap.run_spectral_gap(features, out/"spectral-gap", "cpu")
    result["spectral_gap"] = dict(seconds=time.perf_counter()-t0, windows=len(gap["windows"]), decision=gap["summary"]["decision"],
                                  median_ratio={g: e["median_ratio"] for g, e in gap["summary"]["grids"].items()})
    for case in MUTATIONS:
        copy_dir = out/"mutations"/case
        shutil.copytree(jobs["po_jepa_full"], copy_dir)
        mutate(copy_dir, case)
        model, loss = comparison.build_condition("po_jepa_full", device)
        try:
            record = v7load.load_v7_selected(copy_dir, "po_jepa_full", model, loss_config=loss, features=features, device=device)
            outcome = "loaded" if case == "source_files_sha256" and set(record["source_files_sha256"].values()) == {"0"*64} else "loaded-unexpectedly"
        except ValueError as error:
            outcome = "refused" if case in EXPECTED and EXPECTED[case] in str(error) else f"wrong error: {error}"
        result["mutations"][case] = outcome
        if outcome not in ("refused", "loaded"):
            problems.append(f"mutation {case}: {outcome}")
    for case, delta in (("outside", None), ("last_bit", "nextafter")):
        copy_dir = out/"perturbation"/case
        shutil.copytree(jobs["identity"], copy_dir)
        rows = sorted((copy_dir/"evaluation/identity/rows").glob("*.json"))
        def change(row):
            value = row["values"]["future_jepa_distance"]
            import math
            row["values"]["future_jepa_distance"] = math.nextafter(value, math.inf) if delta else value+2e-6*max(1., abs(value))
        _rewrite(rows[0], change)
        try:
            report = d4.run_d4(features, out/"perturbation-d4"/case, device, "identity", training_run=copy_dir)
            outcome = (report["status"], report["equivalence"]["metrics"]["future_jepa_distance"]["bitwise_equal"])
        except ValueError as error:
            outcome = ("failed", str(error))
        result["perturbation"][case] = outcome
    if result["perturbation"]["outside"][0] != "failed" or "equivalence outside tolerance" not in result["perturbation"]["outside"][1]:
        problems.append("perturbation outside tolerance was not refused")
    if result["perturbation"]["last_bit"] != ("passed", 119):
        problems.append(f"last-bit perturbation: {result['perturbation']['last_bit']}")
    windows = sum(sum(c for c, _, _ in j["equivalence"].values()) for j in result["jobs"].values())
    bitwise = sum(sum(b for _, b, _ in j["equivalence"].values()) for j in result["jobs"].values())
    passed = not problems
    line = (f"D4 冒烟：{'通过' if passed else '失败'}（{len(result['jobs'])} 个 D4 任务，等价核对逐位相同 {bitwise}/{windows}，"
            f"三角不等式违反 {sum(j['triangle_violations'] for j in result['jobs'].values())}，"
            f"v7load 改坏 {sum(v in ('refused', 'loaded') for v in result['mutations'].values())}/{len(MUTATIONS)}"
            f"{'' if passed else '；' + '；'.join(problems)}）")
    cache._write_json(out/"d4-smoke.json", dict(result, problems=problems, passed=passed, summary_line=line, verify_bundle_sha256=bundle,
                                                seconds=time.perf_counter()-started, threads=torch.get_num_threads()))
    print(line, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", type=Path)
    p.add_argument("--reuse", type=Path)
    p.add_argument("--addon-worker", action="store_true")
    p.add_argument("--d4-worker", action="store_true")
    p.add_argument("--features", type=Path)
    p.add_argument("--ac-checkpoint", type=Path)
    p.add_argument("--v7", type=Path)
    p.add_argument("--addon", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.addon_worker:
        return addon_worker(a) or 0
    if a.d4_worker:
        return d4_worker(a) or 0
    if a.workdir is None:
        p.error("--workdir 必填")
    return parent(a.workdir, a.reuse)


if __name__ == "__main__":
    raise SystemExit(main())
