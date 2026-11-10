"""16 §3.2：run.py calibrate-addon——transformer_consistency、fixed_fourier 各自的一致性权重 λ。

方法和 v7 的 workflow.calibrate_weights 相同：种子 1703；σ_z 用导入的 v7 标定（load_protocol 读，不重新标定）；只用 JEPA 的
版本（transformer_consistency 用 matched_transformer 的模型，fixed_fourier 用两个权重为 0 的 fixed_fourier）先在初始状态下量
训练查询序列前 batches 批的三项梯度范数（只记录），再用前 pretrain_updates 批只做 JEPA 训练（学习率调度、裁剪和正式训练
相同），然后在紧接着的 batches 批上量三项梯度范数、取中位数：λ_c = ρ·median|∇L_JEPA| / median|∇L_c|。分母为 0 时报错。
两个条件都完成后才写文件：任务目录里的 calibration-<条件>.json，最后一次性写 configs/calibration/addon/<tag>-u<步数>.yaml
（calibration.load_addon_weights 读）。中途失败不留下这两种文件。
"""

import math

import os

import statistics

from dataclasses import fields, replace

from pathlib import Path

import yaml

from .. import cache, checks as check, comparison, data, training
from ..workflow import OVERLAY, ROOT, read
from . import calibration
from po_jepa.fourier import build_fixed_fourier
from po_jepa.losses import LossConfig

IMPORT_RECORD = check.IMPORT_RECORD
# 和 tools/import_v7_calibration.py 的 V7_COMMITS 相同（测试核对）：只接受 token 写法 0ac45fe 导入的 v7 标定
V7_COMMIT = "0ac45fe03bf43b96bf861446b8aa2ada4f4a8987"


def require_imported_v7_calibration() -> str:
    """补跑入口用的 v7 标定必须是 tools/import_v7_calibration.py 导入的那份（16 §3.1）：IMPORTED.json 里有这个文件名的
    记录，记录的 sha256 等于文件当前的 SHA，记录的 v7_commit 是 0ac45fe（旧提交是模态写法，不能混用）。返回这个 SHA。"""
    path = check.v7_calibration_path()
    record = path.parent/IMPORT_RECORD
    if not path.exists():
        raise ValueError(f"缺少 v7 标定 {path}：先运行 python tools/import_v7_calibration.py")
    if not record.exists():
        raise ValueError(f"缺少 {record}：v7 标定必须用 tools/import_v7_calibration.py 导入，不能在本目录里重新标定")
    entry = read(record).get("files", {}).get(path.name)
    digest = cache.file_sha(path)
    if entry is None or entry.get("sha256") != digest:
        raise ValueError(f"{path.name} 和 {IMPORT_RECORD} 的导入记录不符：本目录的 v7 标定被改过或不是导入的")
    if entry.get("v7_commit") != V7_COMMIT:
        raise ValueError(f"{path.name} 的导入记录 v7_commit 是 {entry.get('v7_commit')}，应为 {V7_COMMIT}"
                         "（旧提交是模态写法，请 git pull 到 0ac45fe 后删掉本目录的 v7 标定和 IMPORTED.json 重新导入）")
    return digest


# S26a：补跑标定所依据的基础标定（σ_z、λ）从哪来。v7-import：v7 树（0ac45fe）的标定由 tools/import_v7_calibration.py 导入
# （16 §3.1，补跑在 v7 的 5 个场景上，默认）；local：本目录自己的 run.py calibrate 写的（v8 任务表的 --stage addon：
# 补跑条件和 v8 的 v7 条件用同一份标定，场景含 gray_scott、push 等 v7 没有的）。
BASE_CALIBRATIONS = ("v7-import", "local")
LOCAL_HEADER = "# v7 标定（run.py calibrate 自动写入，勿手改）："   # workflow.write_v7_calibration 的首行前缀


def require_local_calibration() -> str:
    """S26a：local 模式的基础标定是本目录 run.py calibrate 写的那份：首行是 workflow.write_v7_calibration 的表头
    （数据集 tag 和步数都对得上），sigma_z、time_weight、space_weight 是有限正数，且不是导入的 v7 标定
    （IMPORTED.json 里有这个文件名的记录就拒绝：同一个文件不能两种来源混用）。返回这个文件的 SHA。"""
    path = check.v7_calibration_path()
    if not path.exists():
        raise ValueError(f"缺少基础标定 {path}：local 模式要先在本目录跑 python run.py calibrate（v8 任务表的 C 段）")
    record = path.parent/IMPORT_RECORD
    if record.exists() and path.name in read(record).get("files", {}):
        raise ValueError(f"{path.name} 是从 v7 导入的（{IMPORT_RECORD} 有记录），不是本目录 run.py calibrate 写的：local 模式不收；"
                         "v7 场景的补跑用 --base-calibration v7-import，v8 的补跑请在另一份工作树里跑")
    text = path.read_text(encoding="utf-8")
    header = f"{LOCAL_HEADER}{data.datasets.active().tag}，{check.v7_updates()} 步"
    if text.split("\n", 1)[0] != header:
        raise ValueError(f"{path.name} 的首行不是 run.py calibrate 写的表头（应为 {header!r}）：local 模式只收本目录 calibrate 的输出")
    value = yaml.safe_load(text)
    if type(value) is not dict:
        raise ValueError(f"{path.name} 不是 YAML 映射")
    for name in ("sigma_z", "time_weight", "space_weight"):
        number = value.get(name)
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"{path.name} 的 {name} 必须是有限正数，现在是 {number!r}（run.py calibrate 没跑完？）")
    return cache.file_sha(path)


def require_base_calibration(base: str) -> str:
    """S26a：按来源核基础标定，返回它的 SHA。v7-import 就是 require_imported_v7_calibration（行为不变）。"""
    if base == "v7-import":
        return require_imported_v7_calibration()
    if base == "local":
        return require_local_calibration()
    raise ValueError(f"base calibration must be one of {BASE_CALIBRATIONS}, got {base!r}")


def _jepa_only(condition, device):
    """16 §3.2 第 2 步，S10b §4.1：返回 (模型, 两个权重为 0 的损失配置)。"""
    if condition == "fixed_fourier":
        config = check.load_protocol()
        loss = LossConfig(**{f.name: config["loss"][f.name] for f in fields(LossConfig)})
        return build_fixed_fourier(config["model"], check.model_seed(), device), replace(loss, time_weight=0., space_weight=0.)
    if condition == "transformer_consistency":
        return comparison.build_condition("matched_transformer", device)
    raise ValueError("only fixed_fourier and transformer_consistency use the add-on calibration")


def _gradient_norms(condition, model, batch, loss) -> dict:
    """三项梯度范数，键 jepa、time、space、grad_jepa、grad_time、grad_space（16 §3.2 第 5 步）。"""
    if condition == "transformer_consistency":
        return calibration.generic_gradient_norms(model, batch, loss_config=loss)
    return training.consistency_gradient_norms(model, batch, loss_config=loss)


def _pretrain_step(condition, model, optimizer, batch, loss, completed):
    """只做 JEPA 的一步：Transformer 走 matched_transformer 的单步，fixed_fourier 和 v7 的 calibrate 一样走 no_consistency。"""
    if condition == "transformer_consistency":
        return comparison.condition_step("matched_transformer", model, optimizer, batch, loss, completed)
    return training.train_step(model, optimizer, batch, loss_config=loss, completed_steps=completed, condition="no_consistency")


def _summary(rows) -> dict:
    return dict(median={k: statistics.median(r[k] for r in rows) for k in rows[0]},
                mean={k: sum(r[k] for r in rows)/len(rows) for k in rows[0]})


def calibrate_condition(condition, features, device) -> dict:
    """一个条件的标定记录。键是 v7 calibration.json 的键加 condition；v7_calibration_file、v4_overlay_sha256_after 等
    由 calibrate_addon 在两个条件都完成后补上。"""
    config = check.load_protocol()
    cal = config["calibration"]
    pre, count, rho = cal["pretrain_updates"], cal["batches"], cal["rho"]
    sequence, _ = comparison.learning.make_query_sequence(config)
    if pre+count > len(sequence):
        raise ValueError("pretrain_updates+batches exceeds the training sequence")
    model, loss = _jepa_only(condition, device)
    model.train()

    def rows(queries):
        return [_gradient_norms(condition, model, data.load_training_batch(features, q, device=str(device)), loss) for q in queries]

    initial = rows(sequence[:count])
    optimizer = comparison.make_optimizer(model)
    history = []
    for step in range(pre):
        batch = data.load_training_batch(features, sequence[step], device=str(device))
        result = _pretrain_step(condition, model, optimizer, batch, loss, step)
        history.append(dict(step=result.completed_steps, grad_norm_before_clip=result.grad_norm, **result.loss_values))
        if result.completed_steps % 50 == 0:
            print(f'calibrate-addon {condition} pretrain {result.completed_steps}/{pre} jepa={result.loss_values["jepa"]:.4f}', flush=True)
    measured = rows(sequence[pre:pre+count])
    after, before = _summary(measured), _summary(initial)
    median = after["median"]
    weights, initial_weights = {}, {}
    for name, key in (("time_weight", "grad_time"), ("space_weight", "grad_space")):
        if not median[key] > 0:
            raise ValueError(f"{condition}: median {key} = {median[key]!r}; λ 的分母为 0，不能标定（不除出 inf）")
        weights[name] = rho*median["grad_jepa"]/median[key]
        if not (math.isfinite(weights[name]) and weights[name] > 0):
            raise ValueError(f"calibrated weight must be finite and positive: {condition}.{name}")
        # 初始状态只记录：transformer_consistency 的 p_out 零初始化，一致性梯度恰为 0，记 None（S10b §9 第 13 条）
        initial_weights[name] = rho*before["median"]["grad_jepa"]/before["median"][key] if before["median"][key] > 0 else None
    graph = config["model"]["graph"]
    return dict(condition=condition, rho=rho, batches=len(measured), statistic="median", method="v6_after_jepa_pretraining",
                pretrain_updates=pre, pretrain_history=history, per_batch=measured, median=median, mean=after["mean"],
                initial_state=dict(per_batch=initial, **before, **initial_weights), **weights,
                graph=dict(sigma_x=graph["sigma_x"], sigma_z=graph["sigma_z"], max_modes=graph["max_modes"]),
                v4_overlay_sha256_before=cache.file_sha(OVERLAY),
                note="仿真补跑标定；λ 写进 configs/calibration/addon/，三个种子共用")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def _publish(path: Path, text: str) -> None:
    """同目录临时文件写完、fsync，再硬链接发布（目标已存在就失败，不覆盖）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def calibrate_addon(features, output, device, base="v7-import") -> dict:
    """16 §3.2：output 是已建好的空目录。依次标定 CALIBRATED_CONDITIONS，全部完成后写记录和补跑标定文件，返回写入的内容。
    base（S26a）是基础标定的来源（BASE_CALIBRATIONS）；记录里的 v7_calibration_* 指的就是这份基础标定，另记 base_calibration。"""
    path = calibration.addon_calibration_path()
    if path.exists():
        raise ValueError(f"补跑标定文件已存在：{path}。它可能已被训练任务用过，不覆盖；确实要重标定请先告诉 Jerry")
    v7_file = check.v7_calibration_path()
    v7_sha = require_base_calibration(base)
    results = {c: calibrate_condition(c, Path(features), device) for c in calibration.CALIBRATED_CONDITIONS}
    if require_base_calibration(base) != v7_sha:
        raise ValueError("v7 calibration file changed during calibrate-addon")
    protocol = check.load_protocol()
    source = (f"calibrate-addon：JEPA 预训练 {protocol['calibration']['pretrain_updates']} 步后，"
              f"{protocol['calibration']['batches']} 批梯度范数的中位数；{output.parent.name}")
    payload = {c: dict(rho=r["rho"], time_weight=r["time_weight"], space_weight=r["space_weight"], source=source)
               for c, r in results.items()}
    payload.update(sigma_z_used=protocol["model"]["graph"]["sigma_z"], v7_calibration_sha256=v7_sha)
    header = f"# 仿真补跑标定（run.py calibrate-addon 自动写入，勿手改）：{data.datasets.active().tag}，{check.v7_updates()} 步\n"
    text = header+yaml.safe_dump(payload, allow_unicode=True, sort_keys=True)
    if yaml.safe_load(text) != payload:
        raise ValueError("add-on calibration YAML does not round-trip")
    for c, r in results.items():
        cache._write_json(output/f"calibration-{c}.json", dict(r, v7_calibration_file=_relative(v7_file),
            v7_calibration_sha256=v7_sha, base_calibration=base, addon_calibration_file=_relative(path), v4_overlay_sha256_after=cache.file_sha(OVERLAY)))
    _publish(path, text)
    try:
        for c in calibration.CALIBRATED_CONDITIONS:      # 训练任务读这个文件时的核对（S10b §4.2）必须通过
            if calibration.load_addon_weights(c) != dict(time_weight=results[c]["time_weight"], space_weight=results[c]["space_weight"]):
                raise ValueError("add-on calibration file reads back different weights")
    except Exception:
        path.unlink()
        raise
    print(yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), flush=True)
    print(f"已写入 {_relative(path)}，可以开始 D 段。", flush=True)
    return payload
