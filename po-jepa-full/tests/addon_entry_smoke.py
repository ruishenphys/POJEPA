"""S17 端到端冒烟（开发机，CPU，假特征；文件名不以 test_ 开头，pytest 不收集）。

  python -B tests/addon_entry_smoke.py --features <…>/van_der_pol-40 --work <新目录> [--updates 20] [--timed-updates 100]

--features 是 v7 的 tools/encode_v7.py --fake-encoder 编出的 van_der_pol 40 条假特征（目录名必须是 van_der_pol-40）。
本脚本把当前 po-jepa-full 复制到 <work>/full-model/po-jepa-full（不含 docs/、.vscode/、configs/calibration/、缓存目录），
在复制件里按补跑的顺序走一遍，所有产物都在 <work> 下，不改动当前目录：

  v7 侧（造输入）：<work>/po-jepa 是假的 v7 树（git HEAD 0ac45fe），用 v7 的 workflow.calibrate_weights 写它的标定文件；
                   <work>/results-v7/smoke 下写 6 个 v7 条件的任务记录（job.json、report.json 里的缓存身份、COMPLETE.json）
  A0  import_v7_calibration.import_calibration、addon_preflight.preflight（阶段 smoke：van_der_pol、40 条、2000 步、种子 1703）
  C   addon.entry.execute(calibrate-addon)：真实的 verify_bundle（复制件的 MANIFEST）、导入记录核对、补跑标定
  C2  addon.check_addon.check_addon（CPU；entry 的 check-addon 要求 cuda:N，所以直接调用）
  D   5 个新条件各一个 v7jobs.run_job（训练 --updates 步、第 0/中间/最后一步验证、测试集评测），先过 workflow.execute 里同样的两道标定门槛
  P   tools/plan_v7.py --fake-encoder 在 5 个 D 任务上各规划 1 个回合

和 tools/smoke_v7_cpu.py 一样只缩小步数（协议的训练步数、预热、验证间隔、提前检查、标定 3/2 批），并把假特征当作真实特征读。
最后一行打印 {"ok": true, …}。只在 1 线程下跑（OMP_NUM_THREADS=1 MKL_NUM_THREADS=1）。
"""
import argparse
from argparse import Namespace
import copy
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[1]
V7_COMMIT = "0ac45fe03bf43b96bf861446b8aa2ada4f4a8987"
ADDON = ("po_jepa_time_only", "po_jepa_space_only", "transformer_consistency", "fixed_fourier", "koopman")
SKIP = shutil.ignore_patterns("__pycache__", ".pytest_cache", "docs", ".vscode", "calibration", "results", "features")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--timed-updates", type=int, default=100)
    a = p.parse_args()
    started = time.time()
    features, work = a.features.resolve(), a.work.resolve()
    if features.name != "van_der_pol-40" or not (features/"report.json").is_file():
        raise SystemExit("--features 必须是 encode_v7.py --fake-encoder 编出的 van_der_pol-40 目录")
    work.mkdir(parents=True, exist_ok=False)
    root = work/"full-model"/"po-jepa-full"
    shutil.copytree(HERE, root, ignore=SKIP)
    os.environ.update(PO_JEPA_PROTOCOL="v7", PO_JEPA_DATASET="van_der_pol:40", PO_JEPA_UPDATES="2000", PO_JEPA_SEED="1703")
    sys.path[:0] = [str(root/"src"), str(root/"tools")]
    import torch
    from po_jepa.runtime import cache, checks, data, v7jobs, workflow
    from po_jepa.runtime.addon import calibrate, calibration, check_addon, entry
    import addon_preflight
    import import_v7_calibration
    import make_addon_jobs
    assert Path(checks.__file__).is_relative_to(root) and Path(import_v7_calibration.__file__).is_relative_to(root)
    cpu = torch.device("cpu")

    original = cache._read_v7_cache                     # 假特征是 synthetic：只在本脚本里当作真实特征读（同 smoke_v7_cpu.py）

    def as_real(path, m, sha):
        m = copy.deepcopy(m)
        g = m["generation"]
        g.update(evidence_role="real_frozen_encoder", encoder_checkpoint_sha256=cache.CHECKPOINT_SHA,
                 encoder_load_report=dict(sha256=cache.CHECKPOINT_SHA, encoder_key="target_encoder", strict=True,
                                          missing_keys=[], unexpected_keys=[]))
        return original(path, m, sha)
    cache._read_v7_cache = as_real
    full = checks.load_protocol

    def tiny():
        c = copy.deepcopy(full())
        c["training"]["updates_per_condition"] = a.updates
        c["loss"].update(lr_decay_updates=a.updates, lr_warmup_updates=1, consistency_warmup_updates=1)
        c["validation"]["interval_updates"] = max(1, a.updates//2)
        c["monitoring"].update(early_check_updates=10**9, consistency_interval=2)
        c["calibration"].update(pretrain_updates=3, batches=2)
        return c
    checks.load_protocol = tiny
    log = {}

    def step(name):
        log[name] = round(time.time()-started)
        print(f"[{log[name]:5d}s] {name}", flush=True)

    # —— v7 侧：假的 v7 树（HEAD 0ac45fe）和它的标定；6 个 v7 条件的任务记录 ——
    v7_tree, v7_results, results = work/"po-jepa", work/"results-v7"/"smoke", work/"results-addon"/"smoke"
    subprocess.run(["git", "init", "-q", str(v7_tree)], check=True)
    (v7_tree/".git/HEAD").write_text(V7_COMMIT+"\n")
    name = checks.v7_calibration_path().name
    real_path = checks.v7_calibration_path
    checks.v7_calibration_path = lambda: v7_tree/"configs/calibration"/name
    workflow.calibrate_weights(features, v7_results/"calibrate-van_der_pol"/"run", cpu)
    checks.v7_calibration_path = real_path
    identity = data.approved_cache_metadata(features)
    for c in make_addon_jobs.V7_CONDITIONS:
        job = v7_results/"van_der_pol"/"seed1703"/c/"run"/"job"
        job.mkdir(parents=True)
        cache._write_json(job/"job.json", dict(condition=c, dataset="van_der_pol-40", seed=1703, updates=2000, fictional_v7_record=True))
        cache._write_json(job/"report.json", dict(preflight=dict(hardware=dict(device="cpu"), cache_identity=identity)))
        cache._write_json(job/"COMPLETE.json", dict(report_sha256=cache.file_sha(job/"report.json")))
        cache._write_json(job.parent/"COMPLETE.json", dict(command="run", condition=c))
    step("v7 侧输入")

    # —— A0 ——
    make_addon_jobs.STAGES["smoke"] = dict(scenes=("van_der_pol",), n=40, updates=2000, seeds=(1703,))
    status = import_v7_calibration.import_calibration(v7_tree, "smoke", root/"configs/calibration")
    assert status == {name: "imported"} and (root/"configs/calibration"/name).read_bytes() == (v7_tree/"configs/calibration"/name).read_bytes()
    assert import_v7_calibration.import_calibration(v7_tree, "smoke", root/"configs/calibration") == {name: "unchanged"}
    record = addon_preflight.preflight(v7_results, features.parent, "smoke")
    assert record["checked_jobs"] == 6 and addon_preflight.write_once(results/"preflight.json", record) == "written"
    step("A0 导入标定、preflight")

    # —— C ——
    assert entry.execute(Namespace(command="calibrate-addon", features_root=features, device="cpu",
                                   output=results/"calibrate-van_der_pol"/"run", seed=1703)) == 0
    done = json.loads((results/"calibrate-van_der_pol/run/COMPLETE.json").read_text(encoding="utf-8"))
    assert done["manifest_sha256"] == cache.file_sha(root/"MANIFEST.json")
    weights = {c: calibration.load_addon_weights(c) for c in calibration.CALIBRATED_CONDITIONS}
    step("C calibrate-addon")

    # —— C2 ——
    calibration.require_addon_calibration()
    calibrate.require_imported_v7_calibration()
    check_addon.TIMED_UPDATES = a.timed_updates
    out = results/"check"/"run"
    out.mkdir(parents=True)
    report = check_addon.check_addon(features, out, cpu)
    expected = dict(po_jepa_time_only=10, po_jepa_space_only=10, transformer_consistency=6, fixed_fourier=10, koopman=2)
    for c in ADDON:
        r = report["conditions"][c]
        assert r["readouts"] == dict(observed=expected[c], expected=expected[c]) and r["timed"]["updates"] == a.timed_updates
    assert report["conditions"]["koopman"]["koopman_modes"] == 103
    step("C2 check-addon")

    # —— D ——
    reference = report["reference_initial_model_sha256"]
    summary = {}
    for c in ADDON:
        workflow.require_calibrated_weights(swept=True)          # workflow.execute 的两道门槛（07 §2.6）
        if c in calibration.CALIBRATED_CONDITIONS:
            calibration.require_addon_calibration()
        run = results/"van_der_pol"/"seed1703"/c/"run"
        run.mkdir(parents=True)
        job = v7jobs.run_job(features, run/"job", cpu, c)
        assert (run/"job/COMPLETE.json").exists() and job["training"]["completed_updates"] == a.updates
        assert job["evaluation"]["status"] and job["initial_model_sha256"] == report["conditions"][c]["initial_model_sha256"]
        ref = check_addon.REFERENCE[c]
        if ref is not None:
            assert job["initial_model_sha256"] == reference[ref], c
        summary[c] = dict(initial=job["initial_model_sha256"][:12], best_step=job["best"]["step"],
                          monitor=(job["training"]["consistency_monitor"] or {}).get("checked_terms"))
        step(f"D {c}")

    # —— P：plan_v7.py 接受 5 个新条件 ——
    for c in ADDON:
        sys.argv = ["plan_v7.py", "--job", str(results/"van_der_pol"/"seed1703"/c), "--fake-encoder", "--device", "cpu",
                    "--episodes", "1", "--horizon", "4", "--samples", "16", "--iterations", "2", "--no-baselines",
                    "--output", str(results/"plan"/f"{c}.json")]
        runpy.run_path(str(root/"tools/plan_v7.py"), run_name="__main__")
        plan = json.loads((results/"plan"/f"{c}.json").read_text(encoding="utf-8"))
        assert plan["condition"] == c and len(plan["episodes"]) == 1 and plan["oracle"] is None
        step(f"P {c}")
    print(json.dumps(dict(ok=True, work=str(work), seconds=round(time.time()-started), steps=log,
                          weights=weights, d=summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
