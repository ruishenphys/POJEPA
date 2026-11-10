"""DROID 标定（docs/design/09-droid-training.md §3，S09b）：`calibrate/sigma-z` 和 `calibrate/lambda-<条件>`。

σ_z（§3.1，CPU）：和 v7 `workflow.calibrate_sigma_z` 同一规则，只把轨迹和帧换成 DROID 的回合和行。
- 回合池 = 动作尺度的池（04 §6）：`prepared/action_scale.json` 的 `episodes[<action.source>]`，即完整训练顺序前
  `action.scale_episodes` 条里 `passed_<source>` 为真的回合（保持顺序，共 n 条）；取第 ⌊i·n/8⌋ 条，i = 0…7。
- 每条取 `calibration.sigma_z_rows`（0、8、16）行，`FeatureReader` 读（核行 SHA，float16 → float32），再转 float64。
- Q：`torch.Generator().manual_seed(q_seed)` 的 randn(1408, q_dim) 做 QR、按 diag(R) 定号（09 §2.1 配置块的 q_seed、q_dim）。
- 每行 z = Z·Qᵀ，256 个 token 有序对（不含对角）的平方距离全部收集，σ_z = sqrt(torch.median(全部值))。

λ（§3.2）：和 v7 `workflow.calibrate_weights`（16 §3.2 同样做法）同一算法，批换成 DROID 的 stream 0：
- 只用 JEPA 的版本（种子 `calibration.seed`）：po_jepa_full → po_jepa_no_consistency；fixed_fourier → 两个权重为 0 的
  fixed_fourier；transformer_consistency → matched_transformer。
- 第 0–31 步的批上量初始状态的三项梯度范数（只记录）；第 0–299 步只做 JEPA 训练（v7 的单步、本配置 U 推出的学习率
  调度和裁剪）；第 300–331 步的批上量三项梯度范数，λ_c = ρ·median|∇L_JEPA| / median|∇L_c|（分母为 0 报错，16 §3.2）。

输出 `<run-root>/<name>/calibration/sigma_z.json`、`lambda-<条件>.json`（`cache._write_json`，已存在就报错不覆盖：
写出后训练任务就可能读过它）。

  python -m po_jepa.droid.calibrate --config … --data-root … --run-root … --job calibrate/sigma-z|calibrate/lambda-<条件> \\
      [--device cuda] [--workers 2]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch

from po_jepa.droid import batches as bt
from po_jepa.droid import conditions as cd
from po_jepa.droid import features as ft
from po_jepa.droid import index as ix
from po_jepa.droid import prepare as pr
from po_jepa.droid import windows as wn
from po_jepa.droid.config import DroidConfig, load_config
from po_jepa.droid.spectral_gap import graph_block, read_sigma_z
from po_jepa.runtime import cache, comparison, training
from po_jepa.runtime.addon import calibration as addon_calibration

SIGMA_SCHEMA = "droid_sigma_z_v1"
LAMBDA_SCHEMA = "droid_lambda_v1"
SIGMA_FILE = "sigma_z.json"
TOKENS = ft.TOKENS
JEPA_ONLY = dict(po_jepa_full="po_jepa_no_consistency", fixed_fourier="fixed_fourier",
                 transformer_consistency="matched_transformer")          # 09 §3.2 第 2 步
PLACEHOLDER_WEIGHTS = dict(time_weight=1.0, space_weight=1.0)            # 工厂要正数；构造后换成 0，不影响初始化
QUANTILES = (0.05, 0.5, 0.95)


def calibration_dir(run_root: Path | str, config: DroidConfig) -> Path:
    return Path(run_root)/config.name/"calibration"


def lambda_path(run_root: Path | str, config: DroidConfig, condition: str) -> Path:
    return calibration_dir(run_root, config)/f"lambda-{condition}.json"


# ---------------------------------------------------------------- §3.1 σ_z

def q_matrix(q_seed: int, q_dim: int, width: int = ft.WIDTH) -> torch.Tensor:
    """v7 `calibrate_sigma_z` 的 Q（float64 [q_dim, width]）。"""
    g = torch.Generator().manual_seed(q_seed)
    u, r = torch.linalg.qr(torch.randn(width, q_dim, generator=g, dtype=torch.float64))
    return (u*torch.where(r.diagonal() < 0, -1., 1.)).T


def squared_distances(z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """一行特征 Z [256, 1408]（float32）→ z = Z·Qᵀ 的有序 token 对（不含对角）平方距离（float64，256·255 个）。"""
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("a feature row is [tokens, width]")
    projected = z.double()@q.T
    off = ~torch.eye(z.shape[0], dtype=torch.bool)
    return (projected[:, None] - projected[None]).square().sum(-1)[off]


def sigma_z_pool(scale_doc: Mapping, config: DroidConfig) -> list[str]:
    """§3.1 第 1 步的回合池：动作尺度用的同一池（04 §6，`action_scale.json` 的 `episodes[<source>]`）。"""
    if scale_doc.get("schema") != pr.SCALE_SCHEMA:
        raise ValueError("not an action_scale.json")
    pool = scale_doc.get("episodes", {}).get(config.action.source)
    if not isinstance(pool, list) or not all(isinstance(u, str) for u in pool):
        raise ValueError(f"action_scale.json has no episode pool for source {config.action.source!r}")
    if scale_doc.get("pool", {}).get("scale_episodes") != config.action.scale_episodes or len(pool) > config.action.scale_episodes:
        raise ValueError("action_scale.json was made with another action.scale_episodes")
    if len(set(pool)) != len(pool):
        raise ValueError("action_scale.json pool repeats an episode")
    return list(pool)


def sigma_z_picks(pool: Sequence[str], count: int) -> list[str]:
    """第 ⌊i·n/count⌋ 条，i = 0…count−1（v7 `int(i*train/n)`）。"""
    n = len(pool)
    if n < count:
        raise ValueError(f"the sigma_z pool has {n} episodes, fewer than calibration.sigma_z_episodes = {count}")
    return [pool[i*n//count] for i in range(count)]


def calibrate_sigma_z(reader: ft.FeatureReader, pool: Sequence[str], config: DroidConfig) -> dict:
    """§3.1：返回 sigma_z.json 的值部分（调用者补上配置和输入的 SHA）。"""
    cal = config.calibration
    graph = graph_block(1.0)                      # 只取 q_seed、q_dim（σ_z 占位，不用）
    q_seed, q_dim = graph["q_seed"], graph["q_dim"]
    q = q_matrix(q_seed, q_dim)
    picks = sigma_z_picks(pool, cal.sigma_z_episodes)
    rows = list(cal.sigma_z_rows)
    values, row_sha = [], {}
    for u in picks:
        record = reader.record(u)
        if record["rows"] <= max(rows):
            raise ValueError(f"episode {u} has {record['rows']} rows; sigma_z reads rows {rows}")
        z = reader.rows(u, rows)
        if tuple(z.shape[1:]) != (TOKENS, ft.WIDTH):
            raise ValueError(f"episode {u}: feature rows are {list(z.shape[1:])}, expected [{TOKENS}, {ft.WIDTH}]")
        for k in range(len(rows)):
            values.append(squared_distances(z[k], q))
        row_sha[u] = [record["row_sha256"][r] for r in rows]
    pairs = torch.cat(values)
    sigma = float(pairs.median().sqrt())
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma_z = {sigma} is not a positive finite number")
    return dict(sigma_z=sigma, episodes=picks, pool_size=len(pool), rows=rows, pairs=int(pairs.numel()),
                featsq_quantiles_5_50_95=[float(pairs.quantile(x)) for x in QUANTILES], q_seed=q_seed, q_dim=q_dim,
                row_sha256=row_sha, statistic="sqrt(torch.median(all squared distances)); lower middle value when even")


def run_sigma_z(config: DroidConfig, data_root: Path | str, run_root: Path | str) -> dict:
    """`calibrate/sigma-z`：读动作尺度的池和特征，写 `sigma_z.json`。"""
    data_root = Path(data_root)
    out = calibration_dir(run_root, config)/SIGMA_FILE
    if out.exists():
        raise FileExistsError(f"{out} exists; sigma_z is calibrated once per config and never overwritten")
    scale_path = pr.prepared_dir(data_root)/pr.ACTION_SCALE_FILE
    scale = ix._loads(scale_path.read_bytes(), str(scale_path))
    manifests, manifest_sha = ft.feature_manifests(data_root, config)
    value = calibrate_sigma_z(ft.FeatureReader(data_root, manifests), sigma_z_pool(scale, config), config)
    doc = dict(schema=SIGMA_SCHEMA, config_name=config.name, config_sha256=config.config_sha256,
               action_scale_sha256=cache.file_sha(scale_path), feature_manifest_sha256=manifest_sha, **value)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache._write_json(out, doc)
    return doc


# ---------------------------------------------------------------- §3.2 λ

def jepa_only(condition: str, config: DroidConfig, sigma_z: float, device) -> tuple[torch.nn.Module, object, dict]:
    """§3.2 第 2 步：只用 JEPA 的模型和两个权重为 0 的损失配置（种子 `calibration.seed`）。"""
    if condition not in JEPA_ONLY:
        raise ValueError(f"{condition} has no lambda calibration (09 §1: {sorted(JEPA_ONLY)})")
    base = JEPA_ONLY[condition]
    weights = dict(PLACEHOLDER_WEIGHTS) if base in cd.LAMBDA_CONDITIONS else None
    model, loss, info = cd.build(base, config=config, seed=config.calibration.seed, sigma_z=sigma_z, device=device,
                                 weights=weights)
    return model, dataclasses.replace(loss, time_weight=0., space_weight=0.), info


def gradient_norms(condition: str, model, batch, loss) -> dict:
    """三项梯度范数（键 jepa、time、space、grad_jepa、grad_time、grad_space）：transformer_consistency 用通用梯度范数
    （`addon.calibration.generic_gradient_norms`），PO-JEPA 类 `training.consistency_gradient_norms`。"""
    if condition == "transformer_consistency":
        return addon_calibration.generic_gradient_norms(model, batch, loss_config=loss)
    return training.consistency_gradient_norms(model, batch, loss_config=loss)


def pretrain_step(condition: str, model, optimizer, batch, loss, completed: int):
    """只做 JEPA 的一步（16 §3.2 的 `_pretrain_step`）：Transformer 走 matched_transformer 的单步，PO-JEPA 类（含
    fixed_fourier）和 v7 的 calibrate 一样走 `training.train_step(condition="no_consistency")`。"""
    if condition == "transformer_consistency":
        return comparison.condition_step("matched_transformer", model, optimizer, batch, loss, completed)
    return training.train_step(model, optimizer, batch, loss_config=loss, completed_steps=completed,
                               condition="no_consistency")


def _summary(rows: Sequence[Mapping]) -> dict:
    return dict(median={k: statistics.median(r[k] for r in rows) for k in rows[0]},
                mean={k: sum(r[k] for r in rows)/len(rows) for k in rows[0]})


def _batches(data, config: DroidConfig, start: int, stop: int, device, workers: int):
    dataset = bt.StepBatches("train", data.data_root, data.manifests, data.train_set, data.episodes,
                             seed=config.calibration.seed, start=start, stop=stop, config=config)
    return bt.training_batches(dataset, data.control_spec, data.provenance, device, workers=workers)


def calibrate_lambda(condition: str, *, config: DroidConfig, data, sigma_z: float, sigma_z_sha256: str, device,
                     workers: int = bt.WORKERS) -> dict:
    """§3.2。`data` 是 `train.TrainData`（训练集、准备数组、控制规格、来源、片清单）。返回 lambda-<条件>.json 的内容。"""
    cal = config.calibration
    pre, count, rho = cal.pretrain_updates, cal.median_batches, config.model.rho
    if pre + count > config.training.updates:
        raise ValueError("calibration.pretrain_updates + median_batches exceeds training.updates")
    device = torch.device(device)
    model, loss, info = jepa_only(condition, config, sigma_z, device)
    model.train()
    optimizer = cd.make_optimizer(JEPA_ONLY[condition], model)
    window_sha: list[str] = []

    initial = []
    for step, batch in _batches(data, config, 0, count, device, workers):
        window_sha.append(wn.list_sha256(batch.sample_ids))
        initial.append(gradient_norms(condition, model, batch, loss))
        del batch
    history, measured = [], []
    for step, batch in _batches(data, config, 0, pre + count, device, workers):
        sha = wn.list_sha256(batch.sample_ids)
        if step < count and sha != window_sha[step]:
            raise RuntimeError(f"step {step} gave other windows on the second pass")
        if step >= count:
            window_sha.append(sha)
        if step < pre:
            result = pretrain_step(condition, model, optimizer, batch, loss, step)
            if result.completed_steps != step + 1:
                raise RuntimeError("the update did not advance exactly one step")
            history.append(dict(step=result.completed_steps, grad_norm_before_clip=result.grad_norm, **result.loss_values))
            if result.completed_steps % 50 == 0:
                print(f"calibrate/lambda-{condition} pretrain {result.completed_steps}/{pre} "
                      f"jepa={result.loss_values['jepa']:.4f}", flush=True)
        else:
            measured.append(gradient_norms(condition, model, batch, loss))
        del batch
    if len(window_sha) != pre + count or len(measured) != count:
        raise RuntimeError("calibration did not see every batch")
    after, before = _summary(measured), _summary(initial)
    median = after["median"]
    weights, initial_weights = {}, {}
    for name, key in (("time_weight", "grad_time"), ("space_weight", "grad_space")):
        if not median[key] > 0:
            raise ValueError(f"{condition}: median {key} = {median[key]!r}; the lambda denominator is 0 (16 §3.2)")
        weights[name] = float(rho*median["grad_jepa"]/median[key])
        if not (math.isfinite(weights[name]) and weights[name] > 0):
            raise ValueError(f"calibrated weight must be finite and positive: {condition}.{name}")
        initial_weights[name] = (rho*before["median"]["grad_jepa"]/before["median"][key]
                                 if before["median"][key] > 0 else None)        # 只记录（16 §3.2 第 5 步）
    block = cd.model_block(config, sigma_z)["graph"]
    return dict(schema=LAMBDA_SCHEMA, condition=condition, jepa_only_condition=JEPA_ONLY[condition],
                seed=cal.seed, rho=rho, batches=len(measured), statistic="median", method="v6_after_jepa_pretraining",
                pretrain_updates=pre, pretrain_history=history, per_batch=measured, median=median, mean=after["mean"],
                initial_state=dict(per_batch=initial, **before, **initial_weights), **weights,
                graph=dict(sigma_x=block["sigma_x"], sigma_z=block["sigma_z"], max_modes=block["max_modes"]),
                config_name=config.name, config_sha256=config.config_sha256, sigma_z_sha256=sigma_z_sha256,
                initial_model_sha256=info["initial_model_sha256"], feature_manifest_sha256=data.feature_manifest_sha256,
                train_set=dict(count=len(data.train_set), sha256=wn.list_sha256(data.train_set)),
                step_window_ids_sha256=window_sha, device=str(device),
                note="09 §3.2; the three seeds share these weights; no checkpoint is written")


def run_lambda(condition: str, config: DroidConfig, data_root: Path | str, run_root: Path | str, device,
               workers: int = bt.WORKERS) -> dict:
    """`calibrate/lambda-<条件>`：读 σ_z 和训练数据，写 `lambda-<条件>.json`。"""
    from po_jepa.droid import train as tr     # 训练数据的读法（06 §1）和训练任务相同
    out = lambda_path(run_root, config, condition)
    if out.exists():
        raise FileExistsError(f"{out} exists; lambda is calibrated once per config and never overwritten")
    sigma_z, sigma_sha = read_sigma_z(calibration_dir(run_root, config)/SIGMA_FILE, config)
    data = tr.load_data(config, data_root)
    doc = calibrate_lambda(condition, config=config, data=data, sigma_z=sigma_z, sigma_z_sha256=sigma_sha,
                           device=device, workers=workers)
    cache._write_json(out, doc)
    return doc


# ---------------------------------------------------------------- 命令行

def parse_job(job: str) -> str | None:
    """`calibrate/sigma-z` → None；`calibrate/lambda-<条件>` → 条件。"""
    if job == "calibrate/sigma-z":
        return None
    prefix = "calibrate/lambda-"
    if job.startswith(prefix) and job[len(prefix):] in JEPA_ONLY:
        return job[len(prefix):]
    raise ValueError(f"not a calibration job: {job!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--job", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=bt.WORKERS)
    a = p.parse_args(argv)
    config = load_config(a.config)
    condition = parse_job(a.job)
    target = (calibration_dir(a.run_root, config)/SIGMA_FILE if condition is None
              else lambda_path(a.run_root, config, condition))
    if target.exists():
        print(f"{a.job}: already calibrated ({target})")
        return 0
    if condition is None:
        doc = run_sigma_z(config, a.data_root, a.run_root)
        print(f"sigma_z = {doc['sigma_z']:.6g} → {target}")
    else:
        doc = run_lambda(condition, config, a.data_root, a.run_root, a.device, a.workers)
        print(f"lambda-{condition}: time {doc['time_weight']:.6g}, space {doc['space_weight']:.6g} → {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
