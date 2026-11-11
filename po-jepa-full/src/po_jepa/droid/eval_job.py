"""DROID 评测任务 `eval/<条件>/seed<种子>`、`eval/identity`、`eval/vjepa2_ac_zeroshot`（docs/design/10-evaluation.md §8.1、
§8.9，11；S09b 接线）。

一个任务做两个测试集的全部评测轴（`evaluate.evaluate_job`：冻结窗口各轴、原始帧率、原始分辨率）和离线规划（`planning.run`，
写 `<测试集>/planning/`）：

1. 模型：`conditions.build`（σ_z、λ 从 `calibration/` 读，和训练任务相同的 `train.load_calibration`）。训练过的条件用
   `evaluate.load_selected` 读 `train/<条件>/seed<种子>/` 里 `selection.json` 选出的检查点（10 §8.1 的放宽读法）；identity
   不读；vjepa2_ac_zeroshot 直接用官方权重（`<数据根>/checkpoints/vjepa2-ac-vitg.pt`，记 SHA），随机初始化的部分用
   `construction_seed` = 本配置的第一个种子（09 §2.4：初始权重等于这个种子的 vjepa2_ac）。
2. 数据：两个测试集的冻结窗口（salt 要对）、准备文件（`commanded` 配置跳过没通过的回合，在 evaluate 里做）、回合 outcome、
   特征清单（划分的全部片，和训练的来源字典相同）、测试附加特征（`encode/test-extra`）、动作尺度。
3. `source.json`：`load_selected` 返回的来源记录（数据字段核过、源码 SHA 只记录）；不训练的对照记构造信息。已有时必须相同。
4. `job.json`（evaluate 写）：读的检查点（路径、SHA、步数）、`splits.json`、两个测试集窗口文件、`action_scale.json` 的 SHA、
   特征总清单 SHA、σ_z 和 λ 文件的 SHA、`source.json` 的 SHA，加 evaluate 的输入身份。
5. AC：`ac_state: true_state` 时传 `droid.ac.bind`；原始帧率用 `droid.ac.ac_steps`。

中断后重跑同一条命令接着做（rows 按批跳过、frame_rate/resolution/planning 已完成的直接读；输出目录根上
`cache._write_json` 留下的 `*.partial` 先删掉，S09b-P2-2）。`--time-limit-minutes` 只记录（09 §11）：没完成时每次启动往
`starts.jsonl` 追加一行（时长上限、设备），不进 job.json 的请求身份；评测按批落盘，被排队系统停掉后重排同一条命令即可。

  python -m po_jepa.droid.eval_job --config … --data-root … --run-root … --job eval/<条件>/seed<种子> [--device cuda]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from po_jepa.droid import ac as droid_ac
from po_jepa.droid import batches as bt
from po_jepa.droid import conditions as cd
from po_jepa.droid import evaluate as ev
from po_jepa.droid import features as ft
from po_jepa.droid import index as ix
from po_jepa.droid import planning as pl
from po_jepa.droid import prepare as pr
from po_jepa.droid import splits as sp
from po_jepa.droid import train as tr
from po_jepa.droid.actions import droid_control_spec
from po_jepa.droid.config import CONDITIONS, DroidConfig, load_config
from po_jepa.runtime import cache

SOURCE_SCHEMA = "droid_eval_source_v1"
JOB_KIND = "droid_eval_job_v1"
STARTS_FILE = "starts.jsonl"            # 没完成时每次启动一行：时长上限和设备，只记录


@dataclass(frozen=True)
class EvalData:
    """一个评测任务读的数据（`load_data` 从数据根读；测试直接构造）。`data_sha256` 的键是 13 §8.3 的文件名。"""

    splits: dict
    prepared: dict
    outcomes: dict
    reader: ft.FeatureReader
    extra_reader: ft.ExtraFeatureReader | None
    control_spec: object
    provenance: dict
    feature_manifest_sha256: str
    data_sha256: dict


def load_data(config: DroidConfig, data_root: Path | str) -> EvalData:
    data_root = Path(data_root)
    splits = sp.load_splits(data_root, config)
    plan = sp.shard_plan(splits, config.data)
    prepared = {u: pr.load_prepared(pr.episode_paths(data_root, shard, u)[0])
                for shard in ev.TEST_SPLITS for u in plan[shard]}
    root = pr.prepared_dir(data_root)
    docs = {}
    for split in ev.TEST_SPLITS:
        path = root/pr.WINDOWS_DIR/f"{split}.json"
        docs[split] = ix._loads(path.read_bytes(), str(path))
        ev._windows_doc(docs[split], split)
        if docs[split].get("salt") != config.data.window_salt:
            raise ValueError(f"{path} was frozen with another window salt")
    scale_path = root/pr.ACTION_SCALE_FILE
    scale = ix._loads(scale_path.read_bytes(), str(scale_path))
    manifests, manifest_sha = ft.feature_manifests(data_root, config)
    extra, _ = ft.read_extra_manifest(data_root, config)
    files = {"splits.json": ix.index_dir(data_root)/ix.SPLITS_FILE, "action_scale.json": scale_path,
             **{f"windows/{s}.json": root/pr.WINDOWS_DIR/f"{s}.json" for s in ev.TEST_SPLITS}}
    return EvalData(splits=docs, prepared=prepared, outcomes=ev.episode_outcomes(data_root),
                    reader=ft.FeatureReader(data_root, manifests), extra_reader=ft.ExtraFeatureReader(data_root, extra),
                    control_spec=droid_control_spec(scale, source=config.action.source, config=config),
                    provenance=bt.training_provenance(config, manifests, manifest_sha),
                    feature_manifest_sha256=manifest_sha,
                    data_sha256={k: cache.file_sha(p) for k, p in sorted(files.items())})


def parse_job(job: str, config: DroidConfig) -> tuple[str, int | None]:
    """`eval/<条件>/seed<种子>` → (条件, 种子)；`eval/identity`、`eval/vjepa2_ac_zeroshot` → (条件, None)。"""
    parts = job.split("/")
    if len(parts) == 2 and parts[0] == "eval" and parts[1] in config.training.untrained:
        return parts[1], None
    if (len(parts) == 3 and parts[0] == "eval" and parts[1] in CONDITIONS and parts[2].startswith("seed")
            and parts[2][4:].isdigit() and int(parts[2][4:]) in config.training.seeds):
        return parts[1], int(parts[2][4:])
    raise ValueError(f"not an evaluation job of this config: {job!r}")


def eval_dir(run_root: Path | str, config: DroidConfig, condition: str, seed: int | None) -> Path:
    base = Path(run_root)/config.name/"eval"/condition
    return base if seed is None else base/f"seed{seed}"


def construction_seed(config: DroidConfig) -> int:
    """不训练的对照的构造种子：本配置的第一个种子（vjepa2_ac_zeroshot 的初始权重 = 这个种子的 vjepa2_ac，09 §2.4）。"""
    return config.training.seeds[0]


def _sha(value) -> str:
    return hashlib.sha256(cache._json_bytes(value)).hexdigest()


def _write_or_same(path: Path, value: dict) -> None:
    if path.exists():
        if ix._loads(path.read_bytes(), str(path)) != ix._loads(cache._json_bytes(value), "value"):
            raise ValueError(f"{path} differs from this command (checkpoint, calibration or code changed); "
                             "use a new run root")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cache._write_json(path, value)


def run(out: Path | str, *, condition: str, seed: int | None, config: DroidConfig, data: EvalData,
        calibration: tr.Calibration, device, train_dir: Path | str | None = None,
        ac_checkpoint: Path | str | None = None, time_limit_minutes: float | None = None) -> dict:
    """一个评测任务（见模块说明），返回 evaluate 的报告。"""
    out = Path(out)
    device = torch.device(device)
    if not (out/"COMPLETE.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        for stale in out.glob("*.partial"):  # 上次在 cache._write_json 改名前中断（source.json 等；S09b-P2-2）
            if stale.is_file():
                stale.unlink()
        tr._append_line(out/STARTS_FILE, dict(time_limit_minutes=time_limit_minutes, device=str(device)))
    trained = seed is not None
    if trained != (condition in CONDITIONS) or (not trained and condition not in cd.UNTRAINED):
        raise ValueError(f"{condition}/{seed}: trained conditions need a seed, the two untrained ones none")
    build_seed = seed if trained else construction_seed(config)
    model, loss, info = cd.build(condition, config=config, seed=build_seed, sigma_z=calibration.sigma_z, device=device,
                                 weights=calibration.weights,
                                 ac_checkpoint=ac_checkpoint if condition in cd.OFFICIAL_AC else None)
    if loss is None:                                  # 不训练的对照：只用到 jepa_distance 的两项（权重无关）
        loss = config.loss_config(time_weight=0., space_weight=0.)
    if trained:
        if train_dir is None:
            raise ValueError("a trained condition needs its training directory")
        loaded = ev.load_selected(train_dir, condition=condition, seed=seed, model=model, loss_config=loss,
                                  config=config, provenance=data.provenance)
        checkpoint = loaded["checkpoint"]
        source = dict(schema=SOURCE_SCHEMA, condition=condition, seed=seed, **loaded["source"])
    else:
        model.eval()
        checkpoint = None
        source = dict(schema=SOURCE_SCHEMA, condition=condition, seed=None, construction_seed=build_seed,
                      initial_model_sha256=info["initial_model_sha256"], ac=info["ac"],
                      note="untrained control: no checkpoint (10 §8.1)")
    _write_or_same(out/"source.json", source)
    job = dict(kind=JOB_KIND, checkpoint=checkpoint, data_sha256=dict(data.data_sha256),
               feature_manifest_sha256=data.feature_manifest_sha256,
               sigma_z=dict(value=calibration.sigma_z, sha256=calibration.sigma_z_sha256),
               lambda_sha256=calibration.lambda_sha256, source_sha256=_sha(source),
               construction_seed=None if trained else build_seed,
               initial_model_sha256=None if trained else info["initial_model_sha256"],
               ac_checkpoint_sha256=(info["ac"] or {}).get("official_checkpoint_sha256"))
    is_ac = condition in ev.AC_CONDITIONS
    bind = droid_ac.bind if is_ac and config.model.ac_state == "true_state" else None
    ac_steps = droid_ac.ac_steps if is_ac else None
    source_name = config.action.source

    def planning(split: str) -> dict:
        _, kept, _ = ev._selected(data.splits[split], data.prepared, source_name)
        inputs = pl.PlanningInputs(windows=data.splits[split], reader=data.reader,
                                   episodes=bt.episode_arrays(data.prepared, kept, source_name),
                                   control_spec=data.control_spec, config=config, loss_config=loss)
        return pl.run(model, inputs, out/split/"planning", device, bind=bind)

    return ev.evaluate_job(out, model=model, condition=condition, seed=seed, job=job, splits=data.splits,
                           prepared=data.prepared, outcomes=data.outcomes, reader=data.reader,
                           control_spec=data.control_spec, provenance=data.provenance, loss_config=loss, config=config,
                           device=device, bind=bind, extra_reader=data.extra_reader, ac_steps=ac_steps,
                           planning=planning)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--job", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--time-limit-minutes", type=float)
    a = p.parse_args(argv)
    config = load_config(a.config)
    condition, seed = parse_job(a.job, config)
    out = eval_dir(a.run_root, config, condition, seed)
    data = load_data(config, a.data_root)
    calibration = tr.load_calibration(a.run_root, config, condition, feature_manifest_sha256=data.feature_manifest_sha256)
    train_dir = tr.job_dir(a.run_root, config, condition, seed) if seed is not None else None
    ac = Path(a.data_root)/"checkpoints"/tr.OFFICIAL_AC_FILE if condition in cd.OFFICIAL_AC else None
    report = run(out, condition=condition, seed=seed, config=config, data=data, calibration=calibration,
                 device=a.device, train_dir=train_dir, ac_checkpoint=ac, time_limit_minutes=a.time_limit_minutes)
    print(f"{a.job}: {report['status']} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
