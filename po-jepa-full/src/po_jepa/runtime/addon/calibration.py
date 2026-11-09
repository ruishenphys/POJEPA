"""16 §3.2、08 §3：补跑标定文件的读取和核对，transformer_consistency 的三项梯度范数。

本模块只读补跑标定；写文件的 calibrate-addon 在 S17。文件 configs/calibration/addon/<tag>-u<步数>.yaml 的格式（16 §3.2）：

    fixed_fourier: {rho: …, space_weight: …, time_weight: …, source: …}
    transformer_consistency: {rho: …, space_weight: …, time_weight: …, source: …}
    sigma_z_used: …
    v7_calibration_sha256: …
"""

import math

from pathlib import Path

import torch

import yaml

from .. import checks as check
from .. import data
from po_jepa.generic_consistency import spatial_predictions, temporal_predictions
from po_jepa.losses import jepa_distance, spatial_consistency_loss, temporal_consistency_loss

CALIBRATED_CONDITIONS = ("fixed_fourier", "transformer_consistency")
ENTRY_KEYS = frozenset({"rho", "space_weight", "time_weight", "source"})
FILE_KEYS = frozenset({*CALIBRATED_CONDITIONS, "sigma_z_used", "v7_calibration_sha256"})


def addon_calibration_path() -> Path:
    """configs/calibration/addon/<tag>-u<步数>.yaml：文件名和 checks.v7_calibration_path() 的相同（16 §3.2）。"""
    return check.ROOT/"configs/calibration/addon"/check.v7_calibration_path().name


def load_addon_weights(condition: str) -> dict:
    """读补跑标定并核对，返回 dict(time_weight=float, space_weight=float)。

    要求：condition 是 fixed_fourier 或 transformer_consistency；文件存在；顶层键恰好是 FILE_KEYS；这一项的键恰好是
    ENTRY_KEYS；两个 λ 是有限正数（int 或 float，不收 bool）；rho 等于协议的 calibration.rho；sigma_z_used 等于协议的
    model.graph.sigma_z；v7_calibration_sha256 等于 checks.v7_calibration_path() 当前文件的 SHA。任一不符报 ValueError。
    """
    if condition not in CALIBRATED_CONDITIONS:
        raise ValueError("only fixed_fourier and transformer_consistency use the add-on calibration")
    path = addon_calibration_path()
    if not path.exists():
        raise ValueError(f"补跑标定文件不存在：{path}；请先运行 python run.py calibrate-addon")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if type(value) is not dict or set(value) != FILE_KEYS:
        raise ValueError("add-on calibration must contain exactly fixed_fourier, transformer_consistency, sigma_z_used, v7_calibration_sha256")
    entry = value[condition]
    if type(entry) is not dict or set(entry) != ENTRY_KEYS:
        raise ValueError(f"add-on calibration entry {condition} must contain exactly rho, space_weight, time_weight, source")
    weights = {}
    for name in ("time_weight", "space_weight"):
        weight = entry[name]
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"add-on calibration {condition}.{name} must be a finite positive number")
        weights[name] = float(weight)
    protocol = check.load_protocol()
    if entry["rho"] != protocol["calibration"]["rho"]:
        raise ValueError("add-on calibration rho differs from the protocol calibration.rho")
    if value["sigma_z_used"] != protocol["model"]["graph"]["sigma_z"]:
        raise ValueError("add-on calibration sigma_z_used differs from the protocol sigma_z")
    v7 = check.v7_calibration_path()
    if not v7.exists() or value["v7_calibration_sha256"] != data.cache.file_sha(v7):
        raise ValueError("add-on calibration was made against a different v7 calibration file")
    return weights


def require_addon_calibration() -> None:
    """07 §2.6：transformer_consistency、fixed_fourier 的任务开始前，两项补跑标定都要能读出并通过核对。"""
    for condition in CALIBRATED_CONDITIONS:
        load_addon_weights(condition)


def generic_gradient_norms(model, batch, *, loss_config) -> dict:
    """08 §3、16 §3.2：transformer_consistency 的三项梯度范数，键和 training.consistency_gradient_norms 相同。

    同一批、和 addon.conditions.transformer_consistency_step 相同的 6 次前向（JEPA 1 次、时间 3 次、空间 2 次），
    三项分别用 torch.autograd.grad 对全部 Transformer 参数求导；不写 .grad、不改参数和优化器、不用随机数。
    """
    from .. import comparison
    parameters = [p for _, p in comparison._tf_parameters(model)]
    comparison._tf_batch(batch, model)
    prediction = model(batch.raw_z, batch.controls, batch.duration, batch.transfer.source)
    jepa = jepa_distance(prediction, batch.raw_target, config=loss_config).mean
    temporal = temporal_predictions(model, batch.raw_z, batch.controls, batch.tau, batch.sigma, batch.transfer.source)
    time = temporal_consistency_loss(temporal, config=loss_config).mean
    spatial = spatial_predictions(model, batch.raw_z, batch.controls, batch.duration, batch.transfer)
    space = spatial_consistency_loss(spatial, config=loss_config).mean
    result = dict(jepa=jepa.item(), time=time.item(), space=space.item())
    for name, loss in (("jepa", jepa), ("time", time), ("space", space)):
        grads = torch.autograd.grad(loss, parameters, retain_graph=name != "space", allow_unused=True)
        result["grad_"+name] = math.sqrt(sum(float(g.double().square().sum()) for g in grads if g is not None))
    return result
