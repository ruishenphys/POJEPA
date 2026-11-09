"""S17：tools/import_v7_calibration.py（16 §3.1）和 tools/addon_preflight.py（16 §5、§12），在临时目录里造的假 v7 树和假产物上。"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import addon_preflight as preflight  # noqa: E402
import import_v7_calibration as importer  # noqa: E402
from po_jepa import datasets, simulation, vjepa  # noqa: E402
from po_jepa.runtime import cache, checks  # noqa: E402

FORMAL = "0ac45fe03bf43b96bf861446b8aa2ada4f4a8987"   # v7 正式运行的提交（token 写法；16 §3.1 只接受它）
NOW = "2026-11-13T08:00:00+00:00"
FULL = ["double_well-2000-u10000.yaml", "van_der_pol-2000-u10000.yaml", "multi_particle-2000-u10000.yaml",
        "advection_diffusion-2000-u10000.yaml", "landscape~shape=multi_well-2000-u10000.yaml"]


def calibration_text(tag, sigma=1.5, time=.03, space=4.):
    """和 v7 的 workflow.write_v7_calibration 写出的格式相同。"""
    body = yaml.safe_dump(dict(rho=1.0, sigma_z=sigma, source="JEPA 预训练 300 步后标定；run", space_weight=space,
                               time_weight=time), allow_unicode=True, sort_keys=True)
    return f"# v7 标定（run.py calibrate 自动写入，勿手改）：{tag}，10000 步\n"+body


@pytest.fixture
def v7_tree(tmp_path):
    """假的 v7 树：git HEAD 指向 0ac45fe（只写 HEAD，不需要对象），configs/calibration/ 下有本阶段全部标定文件。"""
    tree = tmp_path/"po-jepa"
    tree.mkdir()
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    (tree/".git/HEAD").write_text(FORMAL+"\n")
    folder = tree/"configs/calibration"
    folder.mkdir(parents=True)
    for stage in ("pilot", "full"):
        for name in importer.file_names(stage):
            (folder/name).write_text(calibration_text(name.rsplit("-u", 1)[0]), encoding="utf-8")
    return tree


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(folder.rglob("*")) if p.is_file()}


def test_file_names_follow_v7_calibration_path(monkeypatch):
    assert importer.file_names("full") == FULL
    assert importer.file_names("pilot") == [n.replace("-2000-u10000", "-200-u2000") for n in FULL]     # 试跑也是 5 个场景
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    for stage, updates, count in (("pilot", "2000", 200), ("full", "10000", 2000)):
        monkeypatch.setenv(checks.UPDATES_ENV, updates)
        for scene, name in zip(importer.STAGES[stage]["scenes"], importer.file_names(stage)):
            system, _, knobs = scene.partition("~")
            old = datasets._set_for_tests(datasets.DatasetSpec(system, count, knobs))
            try:
                assert checks.v7_calibration_path() == checks.ROOT/"configs/calibration"/name
            finally:
                datasets._set_for_tests(old)


def test_import_copies_bytes_records_and_is_idempotent(v7_tree, tmp_path):
    target = tmp_path/"full-model/po-jepa-full/configs/calibration"
    status = importer.import_calibration(v7_tree, "full", target, now=NOW)
    assert status == {name: "imported" for name in FULL}
    source = v7_tree/"configs/calibration"
    for name in FULL:
        assert (target/name).read_bytes() == (source/name).read_bytes()
    record = json.loads((target/"IMPORTED.json").read_text(encoding="utf-8"))
    assert record == dict(schema="addon_v7_calibration_import_v1", files={name: dict(
        source=str((source/name).resolve()), sha256=hashlib.sha256((source/name).read_bytes()).hexdigest(),
        imported_at=NOW, v7_commit=FORMAL) for name in FULL})
    assert (target/"IMPORTED.json").read_bytes() == cache._json_bytes(record)
    before = snapshot(target)
    assert importer.import_calibration(v7_tree, "full", target, now="2026-11-14T00:00:00+00:00") == {n: "unchanged" for n in FULL}
    assert snapshot(target) == before                                           # 重复运行不改任何文件
    assert set(importer.import_calibration(v7_tree, "pilot", target, now=NOW).values()) == {"imported"}
    merged = json.loads((target/"IMPORTED.json").read_text(encoding="utf-8"))["files"]
    assert set(merged) == set(FULL)|set(importer.file_names("pilot")) and all(merged[n] == record["files"][n] for n in FULL)
    assert sorted(p.name for p in target.iterdir()) == sorted([*merged, "IMPORTED.json"])   # 没有残留的临时文件


def test_import_partial_target_adds_missing_record(v7_tree, tmp_path):
    """上次复制完、写 IMPORTED.json 之前中断：目标字节相同就跳过，补上记录。"""
    target = tmp_path/"target"
    target.mkdir()
    (target/FULL[0]).write_bytes((v7_tree/"configs/calibration"/FULL[0]).read_bytes())
    status = importer.import_calibration(v7_tree, "full", target, now=NOW)
    assert status[FULL[0]] == "unchanged" and set(json.loads((target/"IMPORTED.json").read_text())["files"]) == set(FULL)


def test_import_accepts_only_token_v7(v7_tree, tmp_path):
    """16 §3.1：只接受 0ac45fe（token 写法），记录写这次读到的 HEAD。HEAD 是旧提交 653e2d0 或 911bc09（模态写法）时报错、
    不写任何文件；已有记录的 v7_commit 是旧提交时也报错（不再照样跳过），同样不写任何文件。"""
    modal = ("653e2d07477d809e933fdedddebfc77c655b71c5", "911bc09e79e4e92b89a823e083aac7d95edaa029")   # 模态写法的旧 v7 提交
    assert importer.V7_COMMITS == (FORMAL,)
    target = tmp_path/"target"
    for head in modal:
        (v7_tree/".git/HEAD").write_text(head+"\n")
        failing(v7_tree, target, f"git HEAD 是 {head}，应为 {FORMAL}（只接受 0ac45fe 的 token 写法；旧提交是模态写法，请 git pull 到 0ac45fe）")
    assert not target.exists()
    (v7_tree/".git/HEAD").write_text(FORMAL+"\n")
    assert set(importer.import_calibration(v7_tree, "full", target, now=NOW).values()) == {"imported"}
    record = json.loads((target/"IMPORTED.json").read_text(encoding="utf-8"))
    assert {entry["v7_commit"] for entry in record["files"].values()} == {FORMAL}
    for head in modal:
        files = {**record["files"], FULL[3]: dict(record["files"][FULL[3]], v7_commit=head)}
        (target/"IMPORTED.json").write_bytes(cache._json_bytes(dict(record, files=files)))
        failing(v7_tree, target, "IMPORTED.json 里 advection_diffusion-2000-u10000.yaml 的记录和 .* 不符")


def failing(v7_tree, target, message):
    before = snapshot(target) if target.exists() else None
    with pytest.raises(ValueError, match=message):
        importer.import_calibration(v7_tree, "full", target, now=NOW)
    assert (snapshot(target) if target.exists() else None) == before            # 出错时不写任何文件


def test_import_errors(v7_tree, tmp_path):
    target = tmp_path/"target"
    (v7_tree/".git/HEAD").write_text("0"*40+"\n")
    failing(v7_tree, target, f"git HEAD 是 0{{40}}，应为 {FORMAL}（只接受 0ac45fe 的 token 写法；旧提交是模态写法，请 git pull 到 0ac45fe）")
    (v7_tree/".git/HEAD").write_text(FORMAL+"\n")
    failing(tmp_path/"not-a-repo", target, "读不到 .* 的 git HEAD")
    source = v7_tree/"configs/calibration"
    good = (source/FULL[2]).read_bytes()
    (source/FULL[2]).unlink()
    failing(v7_tree, target, "缺少 v7 标定 .*multi_particle-2000-u10000.yaml")
    for broken, key in ((calibration_text("x", time=None), "time_weight"), (calibration_text("x", sigma=0.), "sigma_z"),
                        (calibration_text("x", space=float("nan")), "space_weight"), (calibration_text("x", time=True), "time_weight"),
                        ("- 1\n", None)):
        (source/FULL[2]).write_text(broken, encoding="utf-8")
        failing(v7_tree, target, f"的 {key} 是 .*应为有限正数" if key else "不是 YAML 映射")
    (source/FULL[2]).write_bytes(good)
    target.mkdir()
    (target/FULL[4]).write_bytes(good+b"# changed\n")
    failing(v7_tree, target, "landscape~shape=multi_well-2000-u10000.yaml 已存在且和 .* 不同；不覆盖")
    (target/FULL[4]).unlink()
    importer.import_calibration(v7_tree, "full", target, now=NOW)
    record = json.loads((target/"IMPORTED.json").read_text())
    record["files"][FULL[1]]["sha256"] = "0"*64
    (target/"IMPORTED.json").write_bytes(cache._json_bytes(record))
    failing(v7_tree, target, "IMPORTED.json 里 van_der_pol-2000-u10000.yaml 的记录和 .* 不符")
    (target/"IMPORTED.json").write_bytes(cache._json_bytes(dict(schema="other", files={})))
    failing(v7_tree, target, "不是导入记录")


def test_import_main_writes_into_configs_calibration(v7_tree, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(importer, "ROOT", tmp_path/"new")
    assert importer.main(["--v7-tree", str(v7_tree), "--stage", "pilot"]) == 0
    assert sorted(p.name for p in (tmp_path/"new/configs/calibration").iterdir()) == sorted([*importer.file_names("pilot"), "IMPORTED.json"])
    assert "已导入" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        importer.main(["--v7-tree", str(v7_tree), "--stage", "variants"])


# ---------- addon_preflight.py ----------

STAGE = dict(scenes=("van_der_pol", "landscape~shape=multi_well"), n=40, updates=2000, seeds=(1703, 1704))


def fake_features(root, system, variant, n=40):
    """最小的 v7 编码报告：每条轨迹一个 manifest.json，记录它的 SHA（data._v7_cache_metadata 只读这些）。"""
    root.mkdir(parents=True)
    records = []
    for tid in range(n):
        (root/f"{tid:04d}").mkdir()
        (root/f"{tid:04d}/manifest.json").write_text(json.dumps(dict(trajectory_id=tid, fictional=True)))
        records.append(dict(trajectory_id=tid, manifest_sha256=cache.file_sha(root/f"{tid:04d}/manifest.json")))
    report = dict(status="passed", mode="encode_v7", system=system, dataset_count=n, records=records,
                  **({"variant": variant} if variant else {}))
    (root/"report.json").write_text(json.dumps(report))
    (root/"COMPLETE.json").write_text(json.dumps(dict(report_sha256=cache.file_sha(root/"report.json"))))
    identity = dict(encode_report_sha256=cache.file_sha(root/"report.json"), system=system, dataset_count=n,
                    **({"variant": variant} if variant else {}), manifest_sha256_by_id={str(r["trajectory_id"]): r["manifest_sha256"] for r in records})
    return identity


def fake_job(path, condition, tag, seed, identity, *, done=True):
    job = path/"run/job"
    job.mkdir(parents=True)
    (job/"job.json").write_text(json.dumps(dict(condition=condition, dataset=tag, seed=seed, updates=2000)))
    (job/"report.json").write_text(json.dumps(dict(preflight=dict(hardware=dict(device="cpu"), cache_identity=identity))))
    if done:
        (job/"COMPLETE.json").write_text("{}")
        (path/"run/COMPLETE.json").write_text("{}")


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setitem(preflight.STAGES, "pilot", STAGE)
    features, v7 = tmp_path/"features-v7", tmp_path/"results-v7/pilot"
    identities = {}
    for scene in STAGE["scenes"]:
        system, _, variant = scene.partition("~")
        identities[scene] = fake_features(features/f"{scene}-40", system, variant)
        for seed in STAGE["seeds"]:
            for condition in preflight.V7_CONDITIONS:
                fake_job(v7/scene/f"seed{seed}"/condition, condition, f"{scene}-40", seed, identities[scene])
    return features, v7, identities


def test_preflight_passes_and_records(results, tmp_path):
    features, v7, identities = results
    record = preflight.preflight(v7, features, "pilot")
    assert record["checked_jobs"] == 2*2*6 and list(record["scenes"]) == list(STAGE["scenes"])
    scene = record["scenes"]["landscape~shape=multi_well"]
    assert scene["trajectories"] == 40 and scene["encode_report_sha256"] == identities["landscape~shape=multi_well"]["encode_report_sha256"]
    assert scene["cache_identity_sha256"] == hashlib.sha256(cache._json_bytes(identities["landscape~shape=multi_well"])).hexdigest()
    assert scene["v7_jobs"]["seed1704/vjepa2_ac"] == "landscape~shape=multi_well/seed1704/vjepa2_ac/run/job"
    out = tmp_path/"results-addon/pilot/preflight.json"
    assert preflight.write_once(out, record) == "written" and out.read_bytes() == cache._json_bytes(record)
    assert preflight.write_once(out, record) == "unchanged"
    with pytest.raises(ValueError, match="已存在且内容不同"):
        preflight.write_once(out, dict(record, stage="full"))


def test_preflight_uses_finished_resume_attempt(results):
    features, v7, identities = results
    original = v7/"van_der_pol/seed1703/po_jepa_full"
    os.rename(original, original.with_name("po_jepa_full-resume1"))
    fake_job(original, "po_jepa_full", "van_der_pol-40", 1703, identities["van_der_pol"], done=False)   # 原目录中断，续训完成
    record = preflight.preflight(v7, features, "pilot")
    assert record["scenes"]["van_der_pol"]["v7_jobs"]["seed1703/po_jepa_full"] == "van_der_pol/seed1703/po_jepa_full-resume1/run/job"


def test_preflight_lists_every_problem(results, capsys):
    features, v7, identities = results
    (v7/"van_der_pol/seed1704/identity/run/job/COMPLETE.json").unlink()
    (v7/"van_der_pol/seed1704/identity/run/COMPLETE.json").unlink()                         # 没完成
    job = v7/"van_der_pol/seed1703/matched_transformer/run/job/job.json"
    job.write_text(json.dumps(dict(condition="matched_transformer", dataset="van_der_pol-40", seed=1704, updates=2000)))  # 种子不对
    report = v7/"landscape~shape=multi_well/seed1703/po_jepa_full/run/job/report.json"
    changed = dict(identities["landscape~shape=multi_well"], encode_report_sha256="0"*64)
    report.write_text(json.dumps(dict(preflight=dict(cache_identity=changed))))               # 记的缓存身份不同
    (features/"van_der_pol-40/0007/manifest.json").write_text("{}")                          # 特征被改过
    with pytest.raises(ValueError, match="preflight 没通过：4 项"):
        preflight.preflight(v7, features, "pilot")
    out = capsys.readouterr().out
    assert "seed1704/identity" in out and "matched_transformer/run/job/job.json" in out and "0007" in out
    assert "seed1703/po_jepa_full/run/job 记的特征缓存身份" in out
    (features/"landscape~shape=multi_well-40/COMPLETE.json").write_text("{}")
    with pytest.raises(ValueError, match="preflight 没通过"):
        preflight.preflight(v7, features, "pilot")
    assert "not a complete encode of landscape~shape=multi_well-40" in capsys.readouterr().out


def fake_v6_features(root, monkeypatch, n=200):
    """完整的 v6 特征目录：每条轨迹的 COMPLETE.json、manifest.json、raw_z.npy 都在，manifest 过得了 v6 的逐条读法
    （cache._read_complete_cache_uncached；来源记成 synthetic_development_only，不用核控制序列）。0000 和 0199 的 raw_z 是
    33×256×1408 的零（同一个文件的两个硬链接），其余的只占位（preflight 只完整读首尾两条）。report.json 的 SHA 记在
    reference/cache_asset.json 里：把 data.ROOT 换成临时目录，放一份记着这个 SHA 的记录。返回 v7 任务会记下的缓存身份。"""
    root.mkdir(parents=True)
    zeros = root.parent/"raw_z-zeros.npy"
    np.save(zeros, np.zeros((33, 256, 1408), np.float32))
    feature = cache.file_sha(zeros)
    generation = dict(paper_sha256=cache.PAPER_SHA, protocol_sha256=cache.PROTOCOL_SHA, config_sha256=cache.CONFIG_SHA,
                      upstream_commit=vjepa.UPSTREAM_COMMIT, evidence_role="synthetic_development_only",
                      encoder_checkpoint_sha256=None, preprocess=dict(fictional=True))
    frames = [dict(tick=t, time_seconds=t*.25, encoding_cutoff_seconds=t*.25, image_sha256="0"*64, crop=dict(fictional=True))
              for t in range(33)]
    preprocess = hashlib.sha256(cache._json_bytes(dict(spec=generation["preprocess"], recipe=frames[0]["crop"]))).hexdigest()
    old = datasets._set_for_tests(datasets.DatasetSpec("double_well", n))
    records = []
    try:
        for tid in range(n):
            folder = root/f"{tid:04d}"
            folder.mkdir()
            if tid in (0, n-1):
                os.link(zeros, folder/"raw_z.npy")
            else:
                (folder/"raw_z.npy").write_bytes(b"placeholder")
            manifest = dict(schema_version=1, status="complete", feature_file="raw_z.npy", shape=[33, 256, 1408], dtype="float32",
                            grid=[16, 16], token_order="row_major", trajectory_id=tid, split=simulation.split_for(tid),
                            observation_variant="canonical", generation=generation, feature_sha256=feature,
                            frames=frames, preprocess_sha256=preprocess)
            (folder/"manifest.json").write_text(json.dumps(manifest))
            digest = cache.file_sha(folder/"manifest.json")
            (folder/"COMPLETE.json").write_text(json.dumps(dict(manifest_sha256=digest)))
            records.append(dict(trajectory_id=tid, manifest_sha256=digest))
    finally:
        datasets._set_for_tests(old)
    (root/"report.json").write_text(json.dumps(dict(status="passed", mode="encode", result=dict(records=records))))
    digest = cache.file_sha(root/"report.json")
    (root/"COMPLETE.json").write_text(json.dumps(dict(report_sha256=digest)))
    tree = root.parent/"fake-tree"
    (tree/"reference").mkdir(parents=True)
    (tree/"reference/cache_asset.json").write_text(json.dumps(dict(approved=True, encode_report_sha256=digest)))
    monkeypatch.setattr(preflight.data, "ROOT", tree)
    return dict(accepted_record_sha256=cache.file_sha(tree/"reference/cache_asset.json"), encode_report_sha256=digest,
                manifest_sha256_by_id={str(r["trajectory_id"]): r["manifest_sha256"] for r in records})


@pytest.fixture
def v6_pilot(tmp_path, monkeypatch):
    """试跑只有 double_well、van_der_pol 两个场景、种子 1703；double_well-200 的特征在 v6 目录，van_der_pol 的在 features-v7。"""
    monkeypatch.setitem(preflight.STAGES, "pilot", dict(scenes=("double_well", "van_der_pol"), n=200, updates=2000, seeds=(1703,)))
    v6, features, v7 = tmp_path/"v6/features", tmp_path/"features-v7", tmp_path/"results-v7/pilot"
    identities = dict(double_well=fake_v6_features(v6, monkeypatch),
                      van_der_pol=fake_features(features/"van_der_pol-200", "van_der_pol", "", n=200))
    for scene, identity in identities.items():
        for condition in preflight.V7_CONDITIONS:
            fake_job(v7/scene/"seed1703"/condition, condition, f"{scene}-200", 1703, identity)
    return v6, features, v7, identities


def test_preflight_reads_double_well_200_from_v6_features(v6_pilot, tmp_path, capsys):
    """试跑的 double_well-200 读 --v6-features 的目录，走 v6 的缓存核对（v7 试跑的任务记的也是这份身份）；其他场景照旧。"""
    v6, features, v7, _ = v6_pilot
    record = preflight.preflight(v7, features, "pilot", v6)
    assert record["checked_jobs"] == 12 and record["v6_features"] == str(v6.resolve())
    assert record["scenes"]["double_well"]["features"] == str(v6.resolve()) and record["scenes"]["double_well"]["trajectories"] == 200
    assert record["scenes"]["van_der_pol"]["features"] == str((features/"van_der_pol-200").resolve())
    out = tmp_path/"preflight.json"
    assert preflight.main(["--v7-results", str(v7), "--features", str(features), "--v6-features", str(v6), "--stage", "pilot",
                           "--output", str(out)]) == 0 and json.loads(out.read_text(encoding="utf-8")) == record
    capsys.readouterr()
    with pytest.raises(ValueError, match="preflight 没通过：1 项"):    # 不给：double_well 照 v7 的规则找 features-v7/double_well-200
        preflight.preflight(v7, features, "pilot")
    assert "features-v7/double_well-200" in capsys.readouterr().out


def test_preflight_rejects_metadata_only_and_broken_v6_features(v6_pilot, tmp_path, capsys):
    """git 克隆里的 features/ 只有提交进仓库的元数据：编码报告和缓存身份都和 v7 试跑的相同，只有逐条轨迹的检查拦得下。
    缺一条轨迹的文件、首尾一条的 raw_z 被改，也都在 A0 报出来。每次恰好 1 项。"""
    v6, features, v7, identities = v6_pilot
    clone = tmp_path/"clone/po-jepa/features"
    (clone/"0000").mkdir(parents=True)
    for name in ("report.json", "COMPLETE.json", "0000/manifest.json"):
        (clone/name).write_bytes((v6/name).read_bytes())
    (clone/"registration.json").write_text("{}")
    old = datasets._set_for_tests(datasets.DatasetSpec("double_well", 200))
    try:
        assert preflight._normal(preflight.data.approved_cache_metadata(clone)) == identities["double_well"]
    finally:
        datasets._set_for_tests(old)

    def fails(root, text):
        capsys.readouterr()
        with pytest.raises(ValueError, match="preflight 没通过：1 项"):
            preflight.preflight(v7, features, "pilot", root)
        out = capsys.readouterr().out
        assert text in out, out
    fails(clone, "200 条轨迹缺 COMPLETE.json、manifest.json 或 raw_z.npy（第一条是 0000）；这不是完整的 v6 特征目录")
    (v6/"0123/raw_z.npy").rename(tmp_path/"kept.npy")
    fails(v6, "1 条轨迹缺 COMPLETE.json、manifest.json 或 raw_z.npy（第一条是 0123）")
    (tmp_path/"kept.npy").rename(v6/"0123/raw_z.npy")
    (v6/"0199/raw_z.npy").unlink()                                          # 断开硬链接，再写一份不同的数
    np.save(v6/"0199/raw_z.npy", np.ones((33, 256, 1408), np.float32))
    fails(v6, "轨迹 0199 用 v6 的读法读不出来：ValueError: feature SHA mismatch")


def test_preflight_main_does_not_write_on_failure(results, tmp_path):
    features, v7, _ = results
    out = tmp_path/"results-addon/pilot/preflight.json"
    args = ["--v7-results", str(v7), "--features", str(features), "--stage", "pilot", "--output", str(out)]
    assert preflight.main(args) == 0 and out.exists()
    out.unlink()
    (v7/"van_der_pol/seed1703/identity/run/COMPLETE.json").unlink()
    (v7/"van_der_pol/seed1703/identity/run/job/COMPLETE.json").unlink()
    with pytest.raises(ValueError):
        preflight.main(args)
    assert not out.exists()
