"""S09a：DROID 条件工厂和训练循环（docs/design/09-droid-training.md §2、§4、§5、§8；验收 14 §3 S09）。

合成的小数据根（和 test_droid_batches 同一做法）：训练片 4 条回合、验证片 2 条回合，特征是随机 float16 [F, 256, 1408]，
准备文件是内存里的 `PreparedEpisode`。配置是 pilot.yaml 的缩小版：每步 2 个窗口 / 2 个 AC 片段，U = 30 或 20，
验证间隔 10、一致性测量间隔 2、一致性预热 6、提前检查第 10 步（`Small` 只改这几个由 U 推出的量）。官方规模 AC 用
测试专用的缩小“官方”规格和同格式的假权重（代码路径相同）。逐位比较都在 1 线程下做（14 §1）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import signal
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from po_jepa import ac_baseline
from po_jepa.droid import batches as bt
from po_jepa.droid import checkpoints as ck
from po_jepa.droid import conditions as cd
from po_jepa.droid import encode as en
from po_jepa.droid import features as ft
from po_jepa.droid import train as tr
from po_jepa.droid import windows as wn
from po_jepa.droid.actions import droid_control_spec
from po_jepa.droid.config import CONDITIONS, DroidConfig, load_config
from po_jepa.droid.prepare import PreparedEpisode
from po_jepa.runtime import cache, v7jobs
from po_jepa.third_party import vjepa2_ac
from test_droid_index import CONFIGS

PILOT = load_config(CONFIGS/"pilot.yaml")
ROWS = {"A": 17, "B": 20, "C": 23, "D": 18, "E": 19, "F": 21}
SHARDS = {"train-000": ["A", "B", "C", "D"], "val": ["E", "F"]}
VAL = [("E", 1, 0), ("E", 2, 3), ("E", 4, 5), ("E", 8, 2), ("F", 1, 7), ("F", 2, 0), ("F", 4, 9), ("F", 8, 6)]
SCALE = {"realized": {"position": 0.01, "rotation": 0.05, "gripper": 0.5},
         "commanded": {"position": 0.02, "rotation": 0.04, "gripper": 0.25}}
SYNTHETIC = {"evidence_role": en.SYNTHETIC_ROLE, "encoder_checkpoint_sha256": None, "preprocess_sha256": "p"*64,
             "batch_frames": 32, "upstream_commit": PILOT.encode.upstream_commit}
SIGMA_Z = 40.0
WEIGHTS = {"time_weight": 0.5, "space_weight": 0.5}
REDUCED_OFFICIAL = ac_baseline.ACSpec(width=64, depth=1, heads=4, mlp_ratio=4.0)   # 测试专用的“官方规模”
TIMING = ("seconds", "wait_seconds")


@dataclasses.dataclass(frozen=True)
class Small(DroidConfig):
    """只改由 U 推出的几个量（01 §2 要求 U 是 400 的倍数，测试用 U = 20/30）。"""

    def __post_init__(self) -> None:
        pass

    validation_interval = property(lambda self: 10)
    consistency_interval = property(lambda self: 2)
    consistency_warmup_updates = property(lambda self: 6)
    early_check_updates = property(lambda self: 10)
    lr_warmup_updates = property(lambda self: 2)


def small(updates=30, log_every=4, **training) -> Small:
    t = dataclasses.replace(PILOT.training, updates=updates, batch=2, ac_batch_clips=2, log_every=log_every,
                            seeds=(1703,), **training)
    return Small(**{f.name: getattr(PILOT, f.name) for f in dataclasses.fields(PILOT) if f.name != "training"},
                 training=t)


def uid(k: str) -> str:
    return f"LAB+0000000{k}+2023-07-07-09h-42m-23s"


def _prepared(u: str, rows: int, seed: int) -> PreparedEpisode:
    rng = np.random.default_rng(seed)
    arrays = {"action_realized": rng.uniform(-0.02, 0.02, (rows - 1, 7)),
              "action_commanded": rng.uniform(-0.02, 0.02, (rows - 1, 7)),
              "state": rng.uniform(-0.5, 0.5, (rows, 7))}
    info = {"uuid": u, "rows": rows, "passed_realized": True, "passed_commanded": True}
    return PreparedEpisode(info, hashlib.sha256(f"prep-{u}".encode()).hexdigest(), arrays)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return build_root(tmp_path_factory.mktemp("droid_train"))


def build_root(data):
    """合成数据根（`root` 夹具的内容；别的测试文件用另一个目录调用）。"""
    prepared, manifests = {}, {}
    for shard, keys in SHARDS.items():
        entries = []
        for k in keys:
            u, rows = uid(k), ROWS[k]
            prepared[u] = _prepared(u, rows, 100 + ord(k))
            array = (0.5*np.random.default_rng(ord(k)).standard_normal((rows, ft.TOKENS, ft.WIDTH))).astype(np.float16)
            npy, js = ft.feature_paths(data, shard, u)
            record, js_sha = ft.write_features(npy, js, array, lambda sha, u=u, a=array, s=shard: {
                "schema": ft.FEATURE_SCHEMA, "uuid": u, "shard": s, "rows": a.shape[0], "shape": list(a.shape),
                "dtype": ft.STORAGE_DTYPE, "file_sha256": sha, "row_sha256": ft.row_sha256(a),
                "prepared_sha256": prepared[u].json_sha256, "encoder": SYNTHETIC})
            entries.append({"uuid": u, "rows": rows, "file_sha256": record["file_sha256"], "feature_json_sha256": js_sha})
        manifests[shard] = {"schema": ft.MANIFEST_SCHEMA, "shard": shard, "encoder": SYNTHETIC, "episodes": entries}
    official = data/"vjepa2-ac-vitg.pt"
    torch.save({"predictor": _fake_official()}, official)
    return data, manifests, prepared, official


def _fake_official(seed=3) -> dict:
    """官方键格式的小假权重（含 action/state/extrinsics 三个编码器，带 module. 前缀），编码维 1408。"""
    s = REDUCED_OFFICIAL
    torch.manual_seed(seed)
    core = vjepa2_ac.VisionTransformerPredictorAC(
        img_size=256, patch_size=16, num_frames=2, tubelet_size=2, embed_dim=s.encoder_dim,
        predictor_embed_dim=s.width, depth=s.depth, num_heads=s.heads, mlp_ratio=s.mlp_ratio,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), is_frame_causal=False, use_rope=True,
        action_embed_dim=7, use_silu=False, wide_silu=True)
    return {"module."+k: v.clone() for k, v in core.state_dict().items()}


@pytest.fixture(autouse=True)
def one_thread_and_reduced_official(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(ac_baseline, "OFFICIAL", REDUCED_OFFICIAL)
    yield
    torch.set_num_threads(threads)


def make_data(root, config) -> tr.TrainData:
    data, manifests, prepared, _ = root
    train_set = [uid(k) for k in SHARDS["train-000"]]
    val = [{"id": wn.window_id(uid(k), d, s), "uuid": uid(k), "duration": d, "start": s} for k, d, s in VAL]
    spec = droid_control_spec(SCALE, source="realized", config=config)
    return tr.TrainData(data, manifests, "f"*64, train_set, bt.episode_arrays(prepared, train_set, "realized"),
                        val, bt.episode_arrays(prepared, [uid("E"), uid("F")], "realized"), [], spec,
                        bt.training_provenance(config, manifests, "f"*64), "a"*64, "b"*64,
                        ft.FeatureReader(data, manifests))


def calib(condition, weights=WEIGHTS) -> tr.Calibration:
    if condition in cd.LAMBDA_CONDITIONS:
        return tr.Calibration(SIGMA_Z, "s"*64, dict(weights), "l"*64)
    return tr.Calibration(SIGMA_Z, "s"*64, None, None)


def go(root, out, condition, config, **kw) -> int:
    kw.setdefault("ac_checkpoint", root[3] if condition == "vjepa2_ac" else None)
    return tr.run(out, condition=condition, seed=1703, config=config, data=make_data(root, config),
                  calibration=kw.pop("calibration", calib(condition)), device="cpu", workers=0, **kw)


def payload(path: Path) -> dict:
    return torch.load(io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=True)


def same_tree(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


def log_values(out: Path) -> list[dict]:
    """日志里每一步的全部数值（去掉用时），按步号；加上一致性测量。"""
    steps, shares = [], []
    for row in tr._read_lines(out/tr.LOG_FILE):
        steps += [{k: v for k, v in s.items() if k not in TIMING} for s in row["steps"]]
        shares += row["consistency"]
    return sorted(steps, key=lambda s: s["step"]) + sorted(shares, key=lambda s: s["step"])


def validations(out: Path) -> dict:
    return {p.name: {k: v for k, v in json.loads(p.read_text()).items() if k != "seconds"}
            for p in sorted((out/"validations").glob("*.json"))}


# ---------------------------------------------------------------- 工厂（§2）

def test_factory_counts_relations_and_global_rng(root):
    """§2.3 的参数量逐个断言；§2.4 的初始权重关系；`build` 不改全局随机状态；8 个可训练条件和 2 个对照都能构造。"""
    config = small()
    before = torch.get_rng_state()
    infos = {}
    for c in (*CONDITIONS, *cd.UNTRAINED):
        model, loss, info = cd.build(c, config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu",
                                     weights=WEIGHTS if c in cd.LAMBDA_CONDITIONS else None,
                                     ac_checkpoint=root[3] if c in cd.OFFICIAL_AC else None)
        assert info["initial_model_sha256"] == v7jobs._model_hash(model)
        assert (loss is None) == (c in cd.UNTRAINED)
        infos[c] = info
    assert torch.equal(before, torch.get_rng_state())
    counts = {c: i["parameters"] for c, i in infos.items()}
    assert {c: counts[c] for c in ("po_jepa_full", "po_jepa_no_consistency", "fixed_fourier")} == dict.fromkeys(
        ("po_jepa_full", "po_jepa_no_consistency", "fixed_fourier"), 248_737)
    assert counts["matched_transformer"] == counts["transformer_consistency"] == 249_024
    assert infos["matched_transformer"]["matching"]["selected"]["feedforward"] == 384
    assert counts["vjepa2_ac_small"] == 249_344 and infos["vjepa2_ac_small"]["matching"]["selected"]["mlp_ratio"] == 6.0
    assert counts["koopman"] == 248_512 and infos["koopman"]["matching"]["selected"]["modes"] == 87
    assert counts["identity"] == 0
    assert [r["parameters"] for r in infos["matched_transformer"]["matching"]["candidates"]] == [249_024 - 16*129, 249_024, 249_024 + 16*129]   # 09 §2.3：每单位 FFN 宽 129 个
    lc = dataclasses.replace(config, model=dataclasses.replace(config.model, ac_state="learned_constant"))
    lc = Small(**{f.name: getattr(lc, f.name) for f in dataclasses.fields(lc)})
    m = cd.build("vjepa2_ac_small", config=lc, seed=1703, sigma_z=SIGMA_Z, device="cpu")[2]
    assert m["parameters"] == 248_896 and m["matching"]["selected"]["mlp_ratio"] == 6.0
    for c, ref in cd.REFERENCE.items():
        cd.check_reference(c, infos[c], infos[ref])
    assert infos["koopman"]["initial_model_sha256"] != infos["po_jepa_full"]["initial_model_sha256"]
    other = cd.build("po_jepa_full", config=config, seed=1704, sigma_z=SIGMA_Z, device="cpu", weights=WEIGHTS)[2]
    assert other["initial_model_sha256"] != infos["po_jepa_full"]["initial_model_sha256"]
    with pytest.raises(ValueError, match="different weights"):
        cd.check_reference("fixed_fourier", infos["fixed_fourier"], other)
    assert infos["po_jepa_full"]["spectral_backend"] == dict(eig_backend="cpu_batched", eig_threads=8)
    assert infos["fixed_fourier"]["spectral_backend"] is None
    assert infos["vjepa2_ac"]["ac"]["official_checkpoint_sha256"] == cache.file_sha(root[3])


def test_factory_losses_weights_and_errors(root):
    config = small()
    _, loss, _ = cd.build("po_jepa_full", config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu", weights=WEIGHTS)
    assert (loss.time_weight, loss.space_weight) == (0.5, 0.5)
    assert loss == config.loss_config(time_weight=0.5, space_weight=0.5)
    for c in ("po_jepa_no_consistency", "koopman", "matched_transformer", "vjepa2_ac_small"):
        _, loss, _ = cd.build(c, config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu")
        assert (loss.time_weight, loss.space_weight) == (0., 0.)
    model, _, _ = cd.build("po_jepa_full", config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu", weights=WEIGHTS)
    assert model.dynamics.config.action_dim == 7 and model.spectrum.config.sigma_z == SIGMA_Z
    assert model.dynamics.config.forcing_space == "token"
    with pytest.raises(ValueError, match="calibrated weights"):
        cd.build("transformer_consistency", config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu")
    with pytest.raises(ValueError, match="no consistency weights"):
        cd.build("koopman", config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu", weights=WEIGHTS)
    with pytest.raises(ValueError, match="official weights"):
        cd.build("vjepa2_ac", config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu")
    with pytest.raises(ValueError, match="sigma_z"):
        cd.build("koopman", config=config, seed=1703, sigma_z=0.0, device="cpu")
    tight = dataclasses.replace(config.matching, transformer_feedforward_candidates=(128,))
    tight = Small(**{f.name: getattr(config, f.name) for f in dataclasses.fields(config) if f.name != "matching"},
                  matching=tight)
    with pytest.raises(ValueError, match="not relaxed"):
        cd.build("matched_transformer", config=tight, seed=1703, sigma_z=SIGMA_Z, device="cpu")


# ---------------------------------------------------------------- 训练循环（§4、§5、§8）

def check_complete(root, out, condition, config):
    """14 §3 S09 的“出完整的目录和报告”（U = 30：第 0、10、20、30 步验证，比验收单的 20 步、两次验证多一次）；
    重跑直接退出 0。"""
    U = config.training.updates
    names = {p.name for p in out.iterdir()}
    assert {"registration.json", "checkpoints", "validations", "selection.json", tr.LOG_FILE, tr.STARTS_FILE,
            "report.json", "COMPLETE.json"} <= names and "FAILED.json" not in names
    entries = ck.read_index(out/"checkpoints")
    steps = sorted(e["step"] for e in entries if e["kind"] == "validation")
    assert steps == ([0, 10, 20, 30] if condition != "vjepa2_ac" else sorted({0, 10, 20, 30} & {30, _best(out)}))
    assert {p.name for p in (out/"checkpoints").iterdir()} == {e["path"] for e in entries} | {"index.json"}
    for e in entries:
        assert cache.file_sha(out/"checkpoints"/e["path"]) == e["sha256"]
    report = json.loads((out/"report.json").read_text())
    assert report["completed_steps"] == U and [v["step"] for v in report["validations"]] == [0, 10, 20, 30]
    assert json.loads((out/"COMPLETE.json").read_text()) == {"report_sha256": cache.file_sha(out/"report.json")}
    assert report["resumes"] == 0 and report["registration_sha256"] == cache.file_sha(out/"registration.json")
    best = json.loads((out/"selection.json").read_text())["best"]
    assert best == report["best"] and best["mean"] == min(v["mean"] for v in report["validations"])
    rows = tr._read_lines(out/tr.LOG_FILE)
    tr.check_log_coverage(rows, U)
    assert [(r["first_step"], r["last_step"]) for r in rows] == [(1, 4), (5, 8), (9, 10), (11, 12), (13, 16), (17, 20),
                                                                (21, 24), (25, 28), (29, 30)]
    reg = json.loads((out/"registration.json").read_text())
    assert reg["condition"] == condition and reg["model"]["parameters"] == report["parameters"]
    first = rows[0]["steps"][0]
    if condition in cd.AC_CONDITIONS:
        assert first["p_out_calls"] is None and {"ac_teacher_forcing", "ac_rollout"} <= set(first["losses"])
        assert first["window_ids_sha256"] == wn.list_sha256([c["id"] for c in wn.ac_clips(
            make_data(root, config).train_set, {uid(k): ROWS[k] for k in SHARDS["train-000"]}, seed=1703, step=0,
            config=config)])
    else:
        assert first["p_out_calls"] == cd.expected_readouts(condition, cd.build(
            condition, config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu",
            weights=WEIGHTS if condition in cd.LAMBDA_CONDITIONS else None)[0])
        assert first["window_ids_sha256"] == wn.list_sha256([w["id"] for w in wn.training_windows(
            make_data(root, config).train_set, {uid(k): ROWS[k] for k in SHARDS["train-000"]}, seed=1703, step=0,
            config=config)])
    shares = [m["step"] for r in rows for m in r["consistency"]]
    assert shares == (list(range(2, U + 1, 2)) if condition in cd.LAMBDA_CONDITIONS else [])
    if condition in cd.LAMBDA_CONDITIONS:
        early = json.loads((out/"early-check.json").read_text())
        assert early["completed_updates"] == 10 and early["measurements"] == 3 and early["active"]
        assert report["consistency_monitor"]["measurements"] == 13  # 第 6、8、…、30 步
    else:
        assert report["consistency_monitor"] is None and not (out/"early-check.json").exists()
    assert go(root, out, condition, config) == 0      # 已完成：直接退出 0
    assert len(tr._read_lines(out/tr.STARTS_FILE)) == 1


def _best(out: Path) -> int:
    return json.loads((out/"selection.json").read_text())["best"]["step"]


def _interrupted(root, out, condition, config, monkeypatch):
    """第 13 步收到 SIGTERM、第 21 步时长上限，各退出 75 后重跑同一条命令。"""
    now = [0.0]

    def clock():
        return now[0]

    def after(step):
        if step == 13:
            signal.raise_signal(signal.SIGTERM)
        if step == 21:
            now[0] = 60*(30 - 10) + 1          # 时长上限 30 分钟，余量 10 分钟
    common = dict(time_limit_minutes=30, clock=clock, after_step=after, started=0.0)
    assert go(root, out, condition, config, **common) == 75
    assert ck.latest(ck.read_index(out/"checkpoints"))["path"] == "resume-000013.pt"
    rows = tr._read_lines(out/tr.LOG_FILE)
    assert rows[-1]["last_step"] == 13           # 存档前把不满一行的部分写成一行
    now[0] = 0.0
    assert go(root, out, condition, config, **common) == 75
    assert ck.latest(ck.read_index(out/"checkpoints"))["path"] == "resume-000021.pt"
    assert [e["kind"] for e in ck.read_index(out/"checkpoints")].count("resume") == 1   # 续跑检查点只留最新一个
    now[0] = 0.0
    assert go(root, out, condition, config, **common) == 0


@pytest.mark.parametrize("condition", CONDITIONS)
def test_resume_is_bit_identical(root, tmp_path, monkeypatch, condition):
    """09 §8：30 步、验证间隔 10；第 13 步 SIGTERM、第 21 步快到时长上限，每次退出 75 后重跑；最终的模型状态、
    优化器状态、日志的每个数值、每个验证点的每个值都和不中断的那次逐位相同。"""
    config = small(updates=30)
    a, b = tmp_path/"straight", tmp_path/"resumed"
    assert go(root, a, condition, config) == 0
    check_complete(root, a, condition, config)
    _interrupted(root, b, condition, config, monkeypatch)
    pa, pb = payload(a/"checkpoints/step-000030.pt"), payload(b/"checkpoints/step-000030.pt")
    assert same_tree(pa["model_state"], pb["model_state"]) and same_tree(pa["optimizer_state"], pb["optimizer_state"])
    assert log_values(a) == log_values(b)
    assert validations(a) == validations(b)
    ra, rb = (json.loads((x/"report.json").read_text()) for x in (a, b))
    assert ra["best"]["mean"] == rb["best"]["mean"] and ra["validations"] == rb["validations"]
    assert rb["resumes"] == 2 and ra["resumes"] == 0
    tr.check_log_coverage(tr._read_lines(b/tr.LOG_FILE), 30)


class Killed(BaseException):
    """模拟进程在存档中途被杀（不走 except Exception，不写 FAILED.json）。"""


def test_interrupted_while_saving_removes_orphans(root, tmp_path, monkeypatch):
    """中断在存档中途（检查点写了、index.json 还没更新）：续跑时删掉孤立文件、去掉末步大于检查点步数的日志行，
    结果仍逐位相同。"""
    config = small(updates=30)
    condition = "matched_transformer"     # 存档、索引和孤立文件的逻辑和条件无关，用最快的条件
    a, b = tmp_path/"straight", tmp_path/"killed"
    assert go(root, a, condition, config) == 0
    real = ck.write_index

    def dying(directory, entries):
        if any(e["path"] == "resume-000013.pt" for e in entries):
            raise Killed()
        real(directory, entries)
    monkeypatch.setattr(ck, "write_index", dying)
    with pytest.raises(Killed):
        go(root, b, condition, config, after_step=lambda s: s == 13 and signal.raise_signal(signal.SIGTERM))
    monkeypatch.setattr(ck, "write_index", real)
    assert (b/"checkpoints/resume-000013.pt").exists() and not (b/"FAILED.json").exists()
    assert tr._read_lines(b/tr.LOG_FILE)[-1]["last_step"] == 13
    (b/"checkpoints/.junk.partial").write_bytes(b"x")
    assert go(root, b, condition, config) == 0
    assert not (b/"checkpoints/resume-000013.pt").exists() and not (b/"checkpoints/.junk.partial").exists()
    starts = tr._read_lines(b/tr.STARTS_FILE)
    assert [s["completed"] for s in starts] == [0, 10] and starts[1]["resumed_from"] == "step-000010.pt"
    pa, pb = payload(a/"checkpoints/step-000030.pt"), payload(b/"checkpoints/step-000030.pt")
    assert same_tree(pa["model_state"], pb["model_state"]) and same_tree(pa["optimizer_state"], pb["optimizer_state"])
    assert log_values(a) == log_values(b) and validations(a) == validations(b)
    tr.check_log_coverage(tr._read_lines(b/tr.LOG_FILE), 30)


def test_registration_mismatch_failed_and_retry(root, tmp_path):
    """登记对不上退出 2（不写 FAILED.json）；FAILED.json 在而不带 --retry 时不开始（退出 1）；--retry 接着跑。"""
    config = small(updates=20)
    out = tmp_path/"run"
    assert go(root, out, "matched_transformer", config,
              after_step=lambda s: s == 3 and signal.raise_signal(signal.SIGTERM)) == 75
    reg = (out/"registration.json").read_bytes()
    other = tr.Calibration(SIGMA_Z*2, "t"*64, None, None)
    assert go(root, out, "matched_transformer", config, calibration=other) == 2
    assert (out/"registration.json").read_bytes() == reg and not (out/"FAILED.json").exists()
    cache._write_json(out/"FAILED.json", dict(reason="error"))
    before = sorted(p.name for p in (out/"checkpoints").iterdir())
    assert go(root, out, "matched_transformer", config) == 1
    assert sorted(p.name for p in (out/"checkpoints").iterdir()) == before
    assert go(root, out, "matched_transformer", config, retry=True) == 0
    assert not (out/"FAILED.json").exists() and (out/"COMPLETE.json").exists()


def test_leftover_partial_files_do_not_block_the_rerun(root, tmp_path, monkeypatch):
    """S09a-P2-1：最后写 report.json / COMPLETE.json 时在 `cache._write_json` 改名前中断，留下 *.partial；同一条命令重跑
    （带或不带 --retry）照常写完。先模拟“上次写 report 时中断、这次写失败记了 FAILED.json”的旧行为。另：只有完成标志、
    没有 report.json 的目录不算完成。"""
    config = small(updates=20)
    out = tmp_path/"run"
    assert go(root, out, "matched_transformer", config) == 0
    report = json.loads((out/"report.json").read_text())
    for names in (("report.json.partial",), ("COMPLETE.json.partial", "FAILED.json.partial", "registration.json.partial")):
        (out/"COMPLETE.json").unlink()
        if "COMPLETE.json.partial" not in names:
            (out/"report.json").unlink()
        for name in names:
            (out/name).write_bytes(b'{"half": ')
        assert go(root, out, "matched_transformer", config) == 0
        assert not list(out.glob("*.partial")) and (out/"COMPLETE.json").exists()
        again = json.loads((out/"report.json").read_text())
        assert again["best"] == report["best"] and again["completed_steps"] == 20
        assert json.loads((out/"COMPLETE.json").read_text())["report_sha256"] == cache.file_sha(out/"report.json")
    (out/"COMPLETE.json").unlink()                          # 写 FAILED.json 时也被 .partial 挡住的话会直接抛出：先清
    (out/"report.json").unlink()
    (out/"FAILED.json.partial").write_bytes(b"{")
    monkeypatch.setattr(tr, "selection", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    assert go(root, out, "matched_transformer", config) == 1
    assert json.loads((out/"FAILED.json").read_text())["error"] == "boom"
    monkeypatch.undo()
    assert go(root, out, "matched_transformer", config, retry=True) == 0
    (out/"report.json").unlink()                            # 只剩完成标志：不算完成
    with pytest.raises(ValueError, match="does not match report.json"):
        go(root, out, "matched_transformer", config)


def test_errors_write_failed_json(root, tmp_path, monkeypatch):
    """其他错误写 FAILED.json、退出 1；数据加载器给错步号也算错误。"""
    config = small(updates=20)
    out = tmp_path/"run"
    real = bt.StepBatches.batch
    monkeypatch.setattr(bt.StepBatches, "batch", lambda self, step, reader: (_ for _ in ()).throw(OSError("disk gone"))
                        if step == 5 else real(self, step, reader))
    assert go(root, out, "koopman", config) == 1
    failed = json.loads((out/"FAILED.json").read_text())
    assert failed["reason"] == "error" and "disk gone" in failed["error"]


def test_early_check_stops_weak_consistency(root, tmp_path):
    """09 §4.6：提前检查时一致性占比中位数低于 min_share（0.005）就写 early-check.json 和 FAILED.json、退出 3。"""
    config = small(updates=20)
    out = tmp_path/"weak"
    weak = tr.Calibration(SIGMA_Z, "s"*64, {"time_weight": 1e-9, "space_weight": 1e-9}, "l"*64)
    assert go(root, out, "transformer_consistency", config, calibration=weak) == 3
    early = json.loads((out/"early-check.json").read_text())
    assert early["completed_updates"] == 10 and not early["active"] and early["measurements"] == 3
    assert max(early["median_share"].values()) < 0.005
    assert json.loads((out/"FAILED.json").read_text())["reason"] == "consistency_too_weak"
    assert not (out/"COMPLETE.json").exists()
    assert go(root, out, "transformer_consistency", config, calibration=weak) == 1     # 不自动重排


def test_monitor_summary_and_log_coverage_rules():
    config = small()
    rows = [dict(step=s, share=dict(time=v, space=2*v)) for s, v in ((4, 1.0), (6, 0.001), (8, 0.01), (10, 0.02))]
    summary = cd.monitored_terms
    assert summary("po_jepa_full", config.loss_config(time_weight=1., space_weight=1.)) == ("time", "space")
    assert summary("koopman", config.loss_config(time_weight=0., space_weight=0.)) == ()
    s = tr.monitor_summary(rows, ("time", "space"), config)
    assert s["measurements"] == 3 and s["median_share"] == {"time": 0.01, "space": 0.02} and s["active"]
    rows = [dict(first_step=1, last_step=2, steps=[dict(step=1), dict(step=2)]),
            dict(first_step=3, last_step=3, steps=[dict(step=3)])]
    tr.check_log_coverage(rows, 3)
    with pytest.raises(ValueError):
        tr.check_log_coverage(rows, 4)
    with pytest.raises(ValueError):
        tr.check_log_coverage(rows + [dict(first_step=3, last_step=3, steps=[dict(step=3)])], 3)
    with pytest.raises(ValueError):
        tr.check_log_coverage([rows[1]], 3)


def test_selection_ties_keep_the_earlier_step():
    records = [dict(step=0, mean=1.0), dict(step=10, mean=0.5), dict(step=20, mean=0.5)]
    entries = [dict(path=ck.file_name("validation", s), step=s, kind="validation", sha256="0"*64) for s in (0, 10, 20)]
    assert tr.selection(records, entries)["best"]["step"] == 10
    assert ck.latest(entries + [dict(path="resume-000020.pt", step=20, kind="resume", sha256="1"*64)])["kind"] == "validation"


def test_droid_entry_runs_train(tmp_path, monkeypatch):
    """`tools/droid.py train` 转给 `droid.train.main`，参数含运行根、任务、设备、时长上限和 --retry。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
    import droid
    calls = []
    monkeypatch.setitem(droid.MAINS, "train", lambda argv: calls.append(argv) or 75)
    cfg = str(CONFIGS/"pilot.yaml")
    argv = ["train", "--config", cfg, "--data-root", str(tmp_path/"data"), "--run-root", str(tmp_path/"runs"),
            "--job", "train/koopman/seed1703", "--time-limit-minutes", "20", "--retry", "--device", "cpu"]
    assert droid.main(argv) == 75
    assert calls == [["--config", cfg, "--data-root", str(tmp_path/"data"), "--run-root", str(tmp_path/"runs"),
                      "--job", "train/koopman/seed1703", "--device", "cpu", "--time-limit-minutes", "20", "--retry"]]
    with pytest.raises(SystemExit):
        droid.main(["download", "--config", cfg, "--data-root", str(tmp_path), "--run-root", str(tmp_path),
                    "--job", "download/val", "--retry"])
    assert tr.parse_job("train/koopman/seed1703", PILOT) == ("koopman", 1703)
    for bad in ("train/koopman/seed1704", "train/identity/seed1703", "eval/koopman/seed1703"):
        with pytest.raises(ValueError):
            tr.parse_job(bad, PILOT)


def test_load_data_reads_splits_windows_and_skips_commanded_failures(root, tmp_path, monkeypatch):
    """`load_data`：只读训练片和验证片的准备文件（id_test/ood_test 不读）；验证窗口在 `commanded` 配置下跳过
    `passed_commanded` 为假的回合（10 §8.1）；冻结窗口的 salt 不对就报错。划分、准备文件和特征清单用假的。"""
    data, manifests, prepared, _ = root
    prepared = dict(prepared)
    f = prepared[uid("F")]
    prepared[uid("F")] = PreparedEpisode(dict(f.info, passed_commanded=False), f.json_sha256, f.arrays)
    plan = {"train-000": [uid(k) for k in SHARDS["train-000"]], "val": [uid("E"), uid("F")],
            "id_test": ["never-read"], "ood_test": ["never-read"]}
    read = []
    monkeypatch.setattr(tr.sp, "load_splits", lambda d, c: {"fake": "splits"})
    monkeypatch.setattr(tr.sp, "shard_plan", lambda s, d: plan)
    monkeypatch.setattr(tr.pr, "episode_paths", lambda d, shard, u: (Path(u), None))
    monkeypatch.setattr(tr.pr, "load_prepared", lambda p: read.append(str(p)) or prepared[str(p)])
    monkeypatch.setattr(tr.wn, "train_episodes", lambda s, p, c: plan["train-000"])
    monkeypatch.setattr(tr.ft, "feature_manifests", lambda d, c: (manifests, "f"*64))
    base = tmp_path/"data"
    prep = tr.pr.prepared_dir(base)
    (prep/tr.pr.WINDOWS_DIR).mkdir(parents=True)
    cache._write_json(prep/tr.pr.ACTION_SCALE_FILE, SCALE)
    splits_file = tr.ix.index_dir(base)/tr.ix.SPLITS_FILE
    splits_file.parent.mkdir(parents=True)
    cache._write_json(splits_file, {"fake": "splits"})

    def config_for(source):
        action = dataclasses.replace(PILOT.action, source=source)
        return Small(**{f.name: getattr(PILOT, f.name) for f in dataclasses.fields(PILOT) if f.name != "action"},
                     action=action)
    windows = [{"id": wn.window_id(uid(k), d, s), "uuid": uid(k), "duration": d, "start": s} for k, d, s in VAL]
    doc = {"schema": "droid_windows_v1", "split": "val", "salt": PILOT.data.window_salt, "windows": windows}
    cache._write_json(prep/tr.pr.WINDOWS_DIR/"val.json", doc)
    got = tr.load_data(config_for("realized"), base)
    assert sorted(read) == sorted(plan["train-000"] + plan["val"])
    assert got.val_windows == windows and got.val_skipped == [] and set(got.val_episodes) == {uid("E"), uid("F")}
    assert got.train_set == plan["train-000"] and got.feature_manifest_sha256 == "f"*64
    assert got.val_windows_sha256 == cache.file_sha(prep/tr.pr.WINDOWS_DIR/"val.json")
    assert got.action_scale_sha256 == cache.file_sha(prep/tr.pr.ACTION_SCALE_FILE)
    assert got.splits_sha256 == cache.file_sha(splits_file)
    got = tr.load_data(config_for("commanded"), base)
    assert got.val_windows == [w for w in windows if w["uuid"] == uid("E")] and got.val_skipped == [uid("F")]
    assert set(got.val_episodes) == {uid("E")}
    (prep/tr.pr.WINDOWS_DIR/"val.json").write_text(json.dumps(dict(doc, salt="other")))
    with pytest.raises(ValueError, match="window salt"):
        tr.load_data(config_for("realized"), base)


def test_consistency_share_formula(root):
    """09 §4.6：share = λ·w(s)·|∇L_c|/|∇L_JEPA|，w(s) = min(1, s/W_c)；PO 类用 v7 的三项范数，transformer_consistency
    用通用范数；不改参数和全局随机状态。"""
    from po_jepa.runtime import training
    from po_jepa.runtime.addon import calibration as addon_calibration
    config = small()
    data = make_data(root, config)
    dataset = bt.StepBatches("train", data.data_root, data.manifests, data.train_set, data.episodes, seed=1703,
                             start=0, stop=1, config=config)
    batches = bt.training_batches(dataset, data.control_spec, data.provenance, torch.device("cpu"), workers=0)
    _, batch = next(batches)
    batches.close()
    weights = {"time_weight": 0.3, "space_weight": 0.7}
    for condition, norms_of in (("po_jepa_full", training.consistency_gradient_norms),
                                ("transformer_consistency", addon_calibration.generic_gradient_norms)):
        model, loss, _ = cd.build(condition, config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu", weights=weights)
        params = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state()
        assert loss.consistency_warmup_updates == 6
        got = cd.consistency_share(condition, model, batch, loss, 3)
        assert torch.equal(rng, torch.get_rng_state())
        assert all(torch.equal(params[k], v) for k, v in model.state_dict().items())
        norms = norms_of(model, batch, loss_config=loss)
        assert got["warmup_scale"] == 0.5
        for k, w in (("time", 0.3), ("space", 0.7)):
            assert got["grad_"+k] == norms["grad_"+k] and got["grad_jepa"] == norms["grad_jepa"]
            assert got["share"][k] == w*0.5*norms["grad_"+k]/norms["grad_jepa"]
        assert cd.consistency_share(condition, model, batch, loss, 9)["warmup_scale"] == 1.0


def test_ac_true_state_reaches_step_and_validation(root):
    """`ac_state: true_state`：训练步把片段的真实状态传给 `ac_training.train_step`，验证用绑定起点状态的前向
    （09 §6.2）；和直接调用逐位相同，和不传状态的不同。"""
    from po_jepa.droid import ac as droid_ac
    from po_jepa.losses import jepa_distance
    from po_jepa.runtime import ac_training
    config = small()
    assert config.model.ac_state == "true_state"
    data = make_data(root, config)
    dataset = bt.StepBatches("ac", data.data_root, data.manifests, data.train_set, data.episodes, seed=1703,
                             start=0, stop=1, config=config)
    batches = bt.training_batches(dataset, data.control_spec, data.provenance, torch.device("cpu"), workers=0)
    _, batch = next(batches)
    batches.close()
    results = []
    for how in ("droid", "direct"):
        model, loss, _ = cd.build("vjepa2_ac_small", config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu")
        optimizer = cd.make_optimizer("vjepa2_ac_small", model)
        if how == "droid":
            result, _ = cd.train_step("vjepa2_ac_small", model, optimizer, batch, loss, 0, ac_state="true_state")
        else:
            result, _ = ac_training.train_step(model.ac_condition, model, optimizer, batch.features, batch.actions, loss,
                                               0, states=batch.states)
        results.append((result.loss_values, {k: v.clone() for k, v in model.state_dict().items()}))
    assert results[0][0] == results[1][0] and same_tree(results[0][1], results[1][1])
    with pytest.raises(ValueError, match="true_state AC needs"):      # 这个模型不带状态就不能训练
        ac_training.train_step(model.ac_condition, model, optimizer, batch.features, batch.actions, loss, 1, states=None)
    model, loss, _ = cd.build("vjepa2_ac_small", config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu")
    got = tr.validate(model, condition="vjepa2_ac_small", data=data, config=config, loss_config=loss, device="cpu")
    batch, states = bt.make_eval_batch(data.val_windows, data.reader, data.val_episodes, data.control_spec,
                                       data.provenance, "cpu")
    model.eval()
    with torch.no_grad():
        phi = droid_ac.bind(model, states, batch.controls)
        want = jepa_distance(phi(batch.raw_z, batch.controls, batch.duration, batch.transfer.source), batch.raw_target,
                             config=loss).per_sample.double()
    assert got["values"] == want.tolist() and got["ids"] == [w["id"] for w in data.val_windows]
