"""DROID 的条件工厂（docs/design/09-droid-training.md §2，S09a）。

`build(condition, *, config, seed, sigma_z, device, weights=None, ac_checkpoint=None)` 返回 (模型, 损失配置, 登记信息)：

- PO-JEPA 类（po_jepa_full、po_jepa_no_consistency、fixed_fourier）用 §2.1 的模型配置块：po-jepa-full 的
  `configs/train/default.yaml` 深度合并 `v4.yaml`（和 v7 `checks.load_protocol()` 的前两步相同），再合并 `v7.yaml` 的
  `model` 段（token 写法），只改 `dynamics.action_dim = 7`、`graph.sigma_z = σ_z`；构造顺序和 `checks.build_full_model`
  相同，`fork_rng` 里 `manual_seed(seed)`。fixed_fourier 用 S10 的 `fourier.build_fixed_fourier`（同一配置块、同一种子）。
- matched_transformer、transformer_consistency：`TokenTransformer(TransformerSpec(action_dim=7, width=64, heads=4,
  layers=1, feedforward=按 §2.3 选, residual=True))`，其余字段取默认；koopman：`KoopmanConfig(modes=按 08 §5.6 选,
  action_dim=7)`，S10 的 `build_koopman`。
- vjepa2_ac、vjepa2_ac_zeroshot：官方规模 `ACSpec(action_dim=7, state_input=model.ac_state)`，载入官方权重
  （`ac_checkpoint`，S08b 的 `load_official`）；vjepa2_ac_small：`ACSpec(width=64, depth=1, heads=4, mlp_ratio=按 §2.3 选,
  action_dim=7, state_input=model.ac_state)`，从头训练。identity：`IdentityPredictor(action_dim=7)`。
- 损失：po_jepa_full、fixed_fourier、transformer_consistency 的两个权重由调用者从 `lambda-<条件>.json` 读出后经
  `weights` 传进来（S09a 定：工厂不读文件），其余条件两个权重为 0（AC 只用它的学习率调度）。
- 求谱后端：GraphSpectrum 的 `eig_backend = model.spectral_backend`、`eig_threads = resources.train.cpus`（09 §6.3）。

参数量按 §2.3 在构造时现算，任一条件超过 `matching.tolerance` 就报错。所有构造都在 `fork_rng` 里做，不改全局随机状态。
登记信息里的初始权重哈希用 v7 `v7jobs._model_hash`；§2.4 的关系由 `REFERENCE` 给出，训练任务开始时核。
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from po_jepa import ac_baseline
from po_jepa.baselines import IdentityPredictor, TokenTransformer, TransformerSpec
from po_jepa.droid.config import CONDITIONS, DroidConfig
from po_jepa.dynamics import DynamicsConfig, ModalDynamics
from po_jepa.fourier import FourierSpectrum, build_fixed_fourier
from po_jepa.koopman import KoopmanConfig, KoopmanPredictor, build_koopman, choose_modes
from po_jepa.koopman import parameter_count as koopman_parameter_count
from po_jepa.losses import LossConfig
from po_jepa.model import POJEPA, IntegratorConfig, ProjectionConfig
from po_jepa.runtime import ac_training, cache, checks, comparison, training, v7jobs
from po_jepa.runtime.addon import calibration as addon_calibration
from po_jepa.runtime.addon import conditions as addon_conditions
from po_jepa.spectral import GraphConfig, GraphSpectrum

ACTION_DIM = 7
UNTRAINED = ("identity", "vjepa2_ac_zeroshot")
ALL = (*CONDITIONS, *UNTRAINED)
PO_CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "fixed_fourier")
TRANSFORMER_CONDITIONS = ("matched_transformer", "transformer_consistency")
AC_CONDITIONS = ("vjepa2_ac", "vjepa2_ac_small")
OFFICIAL_AC = ("vjepa2_ac", "vjepa2_ac_zeroshot")
LAMBDA_CONDITIONS = ("po_jepa_full", "fixed_fourier", "transformer_consistency")   # 09 §1：各有一条 calibrate/lambda
# training._loss_config 的标签（检查点里存的 condition / loss_config 快照）
LOSS_LABELS = dict(po_jepa_full="full", po_jepa_no_consistency="no_consistency", fixed_fourier="full",
                   koopman="no_consistency", matched_transformer="no_consistency", transformer_consistency="full",
                   vjepa2_ac="no_consistency", vjepa2_ac_small="no_consistency")
# 09 §2.4：同一种子下初始权重哈希必须相等的参照条件
REFERENCE = dict(po_jepa_no_consistency="po_jepa_full", fixed_fourier="po_jepa_full",
                 transformer_consistency="matched_transformer", vjepa2_ac_zeroshot="vjepa2_ac")
TRANSFORMER_FIXED = dict(width=64, heads=4, layers=1, residual=True)     # 09 §2.2
AC_SMALL_FIXED = dict(width=64, depth=1, heads=4)                        # 09 §2.2
MODEL_SECTIONS = ("projection", "graph", "dynamics", "integrator")


# ---------------------------------------------------------------- §2.1 模型配置块

def model_block(config: DroidConfig, sigma_z: float) -> dict:
    """09 §2.1：default.yaml ← v4.yaml 的 `model`，再合并 v7.yaml 的 `model` 段；只改 action_dim 和 sigma_z。
    读之前 `cache.verify_fixed_protocol()`（这三份文件必须和 v7 逐字节相同）。K、σ_x 必须等于 DROID 配置的。"""
    if type(sigma_z) is not float or not math.isfinite(sigma_z) or sigma_z <= 0:
        raise ValueError(f"sigma_z must be a positive finite float, got {sigma_z!r}")
    cache.verify_fixed_protocol()
    root = cache.ROOT/"configs"/"train"
    base = yaml.safe_load((root/"default.yaml").read_text(encoding="utf-8"))
    overlay = yaml.safe_load((root/"v4.yaml").read_text(encoding="utf-8"))
    merged = checks._merge_v4(base, {k: v for k, v in overlay.items() if k not in ("schema_version", "protocol_id")})
    v7 = yaml.safe_load((root/"v7.yaml").read_text(encoding="utf-8"))
    merged = checks._merge_v4(merged, dict(model=v7["model"]))
    block = {k: dict(merged["model"][k]) for k in MODEL_SECTIONS}
    if block["dynamics"].get("forcing_space") != "token":
        raise ValueError("the DROID model block must use the v7 token forcing (09 §2.1)")
    graph = block["graph"]
    if graph["max_modes"] != config.model.modes or graph["sigma_x"] != config.model.sigma_x:
        raise ValueError(f"the PO-JEPA graph (K = {graph['max_modes']}, sigma_x = {graph['sigma_x']}) differs from the "
                         f"DROID config (modes {config.model.modes}, sigma_x {config.model.sigma_x})")
    block["dynamics"]["action_dim"] = ACTION_DIM
    graph["sigma_z"] = sigma_z
    return block


def _po(block: dict, seed: int, device, *, fourier: bool = False) -> POJEPA:
    """和 `checks.build_full_model` 相同的构造顺序；fork_rng 里 manual_seed(seed)。fixed_fourier 用 S10 的构造函数。"""
    if fourier:
        return build_fixed_fourier(block, seed, torch.device(device))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = POJEPA(ProjectionConfig(**block["projection"]), GraphSpectrum(GraphConfig(**block["graph"])),
                       ModalDynamics(DynamicsConfig(**block["dynamics"])), IntegratorConfig(**block["integrator"]))
    return model.to(device)


def po_parameters(block: dict) -> int:
    """PO-JEPA（token 写法、7 维动作）的可训练参数量；CPU 上构造一次计数（不扰动全局随机状态）。"""
    return comparison.parameter_counts(_po(block, 0, "cpu"))["total"]


# ---------------------------------------------------------------- §2.3 参数量匹配

def _match(po: int, rows: list[dict], tolerance: float, what: str) -> dict:
    """绝对差最小，平局取规格字典序较小的（v7 `choose_architecture` 的规则）；超过容差报错。"""
    if not rows:
        raise ValueError(f"no {what} candidates")
    for r in rows:
        r["absolute_difference"] = abs(r["parameters"] - po)
    best = min(rows, key=lambda r: (r["absolute_difference"], tuple(r["spec"].values())))
    if best["absolute_difference"] > tolerance*po:
        raise ValueError(f"{what}: no candidate within {tolerance:.2%} of the PO-JEPA count {po} "
                         f"(best {best['parameters']}); not relaxed (09 §2.3)")
    return dict(rule="minimum_absolute_parameter_difference_then_lexicographic_spec", po_parameters=po,
                tolerance=tolerance, candidates=rows, selected=best["spec"], parameters=best["parameters"],
                absolute_difference=best["absolute_difference"], relative_difference=best["absolute_difference"]/po)


def transformer_spec(feedforward: int) -> TransformerSpec:
    return TransformerSpec(action_dim=ACTION_DIM, feedforward=feedforward, **TRANSFORMER_FIXED)


def ac_small_spec(config: DroidConfig, mlp_ratio: float) -> ac_baseline.ACSpec:
    return ac_baseline.ACSpec(mlp_ratio=mlp_ratio, action_dim=ACTION_DIM, state_input=config.model.ac_state,
                              **AC_SMALL_FIXED)


def official_ac_spec(config: DroidConfig) -> ac_baseline.ACSpec:
    """官方规模（结构字段取 `ac_baseline.OFFICIAL`），7 维动作、`model.ac_state` 的状态输入。"""
    return dataclasses.replace(ac_baseline.OFFICIAL, action_dim=ACTION_DIM, state_input=config.model.ac_state)


def _count(build) -> int:
    with torch.random.fork_rng(devices=[]):
        return sum(p.numel() for p in build().parameters() if p.requires_grad)


def match_transformer(config: DroidConfig, po: int) -> dict:
    rows = [dict(spec=asdict(s), parameters=_count(lambda s=s: TokenTransformer(s)))
            for s in (transformer_spec(w) for w in config.matching.transformer_feedforward_candidates)]
    return _match(po, rows, config.matching.tolerance, "matched_transformer feedforward")


def match_ac_small(config: DroidConfig, po: int) -> dict:
    rows = [dict(spec=ac_baseline.spec_snapshot(s), parameters=_count(lambda s=s: ac_baseline.ACPredictor(s)))
            for s in (ac_small_spec(config, r) for r in config.matching.ac_small_mlp_ratio_candidates)]
    return _match(po, rows, config.matching.tolerance, "vjepa2_ac_small mlp_ratio")


def match_koopman(config: DroidConfig, po: int) -> dict:
    """08 §5.6：K 取任意正整数里使参数量最接近的（相等取较小的），只有这一个候选。"""
    k = choose_modes(po, action_dim=ACTION_DIM)
    spec = asdict(KoopmanConfig(modes=k, action_dim=ACTION_DIM))
    return _match(po, [dict(spec=spec, parameters=koopman_parameter_count(KoopmanConfig(**spec)))],
                  config.matching.tolerance, "koopman modes")


# ---------------------------------------------------------------- §2.2 每个条件

def _weights(condition: str, weights) -> tuple[float, float]:
    if condition in LAMBDA_CONDITIONS:
        if (not isinstance(weights, dict) or set(weights) != {"time_weight", "space_weight"}
                or any(type(weights[k]) is not float or not math.isfinite(weights[k]) or weights[k] <= 0
                       for k in weights)):
            raise ValueError(f"{condition} needs calibrated weights {{time_weight, space_weight}} (positive finite "
                             f"floats from lambda-{condition}.json), got {weights!r}")
        return weights["time_weight"], weights["space_weight"]
    if weights is not None:
        raise ValueError(f"{condition} has no consistency weights (09 §2.2)")
    return 0., 0.


def model_spec(condition: str, model) -> dict:
    """登记和检查点里的规格快照（v7 各自的写法）。"""
    if isinstance(model, POJEPA):
        return training._model_config(model)
    if isinstance(model, TokenTransformer):
        return asdict(model.spec)
    if isinstance(model, KoopmanPredictor):
        return addon_conditions.koopman_model_config(model)
    if isinstance(model, ac_baseline.ACPredictor):
        return ac_baseline.spec_snapshot(model.spec)
    if isinstance(model, IdentityPredictor):
        return dict(encoder_dim=model.encoder_dim, action_dim=ACTION_DIM)
    raise TypeError(f"unknown model type for {condition}")


def _official(model: ac_baseline.ACPredictor, ac_checkpoint) -> dict:
    if ac_checkpoint is None or not Path(ac_checkpoint).is_file():
        raise ValueError(f"vjepa2_ac needs the official weights file (vjepa2-ac-vitg.pt, {ac_baseline.OFFICIAL_CHECKPOINT_URL}), "
                         f"got {ac_checkpoint!r}")
    state, digest = ac_training._official_state(Path(ac_checkpoint))
    return dict(official_checkpoint_sha256=digest, official_load=model.load_official(state))


def build(condition: str, *, config: DroidConfig, seed: int, sigma_z: float, device,
          weights: dict | None = None, ac_checkpoint: Path | str | None = None) -> tuple[torch.nn.Module, LossConfig | None, dict]:
    """09 §2：(模型, 损失配置, 登记信息)。不训练的两个对照没有损失配置（None）。

    登记信息：condition、seed、model_type、spec（规格快照）、parameters、po_parameters、matching（§2.3，无匹配的为 None）、
    initial_model_sha256、spectral_backend（GraphSpectrum 的后端和份数，其余为 None）、ac（官方权重 SHA、载入记录，其余 None）。
    """
    if condition not in ALL:
        raise ValueError(f"unknown DROID condition {condition!r}")
    if type(seed) is not int or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    device = torch.device(device)
    time_weight, space_weight = _weights(condition, weights)
    block = model_block(config, sigma_z)
    po = po_parameters(block)
    matching, ac = None, None
    if condition in PO_CONDITIONS:
        model = _po(block, seed, device, fourier=condition == "fixed_fourier")
    elif condition in TRANSFORMER_CONDITIONS:
        matching = match_transformer(config, po)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = TokenTransformer(TransformerSpec(**matching["selected"]))
        model = model.to(device)
    elif condition == "koopman":
        matching = match_koopman(config, po)
        model = build_koopman(KoopmanConfig(**matching["selected"]), seed, device)
    elif condition in OFFICIAL_AC or condition == "vjepa2_ac_small":
        if condition == "vjepa2_ac_small":
            matching = match_ac_small(config, po)
            spec = ac_baseline.ACSpec(**matching["selected"])
        else:
            spec = official_ac_spec(config)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = ac_baseline.ACPredictor(spec)
        ac = dict(init="official_checkpoint" if condition in OFFICIAL_AC else "scratch")
        if condition in OFFICIAL_AC:
            ac.update(_official(model, ac_checkpoint))
        elif ac_checkpoint is not None:
            raise ValueError("vjepa2_ac_small is trained from scratch; it takes no official weights")
        model = model.to(device)
        model.ac_condition = "vjepa2_ac" if condition in OFFICIAL_AC else condition   # 优化器峰值、存档（ac_training）
        model.ac_info = dict(condition=condition, spec=ac_baseline.spec_snapshot(spec), **ac,
                             parameters=ac_baseline.parameter_count(model))
    else:   # identity
        model = IdentityPredictor(encoder_dim=block["projection"]["encoder_dim"], action_dim=ACTION_DIM).to(device)
    backend = None
    if isinstance(model, POJEPA) and type(model.spectrum) is GraphSpectrum:     # FourierSpectrum 不求谱
        model.spectrum.eig_backend = config.model.spectral_backend
        model.spectrum.eig_threads = config.resources.train.cpus
        backend = dict(eig_backend=model.spectrum.eig_backend, eig_threads=model.spectrum.eig_threads)
    elif isinstance(model, POJEPA) and type(model.spectrum) is not FourierSpectrum:
        raise TypeError("unexpected spectrum module")
    parameters = comparison.parameter_counts(model)["total"]
    if matching is not None and parameters != matching["parameters"]:
        raise ValueError(f"{condition}: built model has {parameters} parameters, the matching chose {matching['parameters']}")
    if condition in PO_CONDITIONS and parameters != po:
        raise ValueError(f"{condition}: {parameters} parameters, PO-JEPA has {po}")
    loss = None
    if condition not in UNTRAINED:
        loss = config.loss_config(time_weight=time_weight, space_weight=space_weight)
        training._loss_config(loss, LOSS_LABELS[condition])
    info = dict(condition=condition, seed=seed, model_type=type(model).__name__, spec=model_spec(condition, model),
                parameters=parameters, po_parameters=po, matching=matching,
                initial_model_sha256=v7jobs._model_hash(model), spectral_backend=backend, ac=ac)
    return model, loss, info


def reference_hash(condition: str, *, config: DroidConfig, seed: int, sigma_z: float) -> str:
    """09 §2.4 的参照条件（po_jepa_full、matched_transformer）在 CPU 上构造后的初始权重哈希；权重不影响初始化，
    po_jepa_full 用占位权重 1.0 构造。官方 AC 不在这里算（vjepa2_ac_zeroshot 不训练，和 vjepa2_ac 是同一构造）。"""
    if condition not in ("po_jepa_full", "matched_transformer"):
        raise ValueError(f"{condition} is not a 09 §2.4 reference of a trained condition")
    weights = dict(time_weight=1.0, space_weight=1.0) if condition in LAMBDA_CONDITIONS else None
    return build(condition, config=config, seed=seed, sigma_z=sigma_z, device="cpu", weights=weights)[2]["initial_model_sha256"]


def check_reference(condition: str, info: dict, reference_info: dict) -> None:
    """09 §2.4：`condition` 的初始权重哈希等于参照条件（REFERENCE）的。"""
    if REFERENCE.get(condition) != reference_info["condition"]:
        raise ValueError(f"{reference_info['condition']} is not the 09 §2.4 reference of {condition}")
    if info["seed"] != reference_info["seed"] or info["initial_model_sha256"] != reference_info["initial_model_sha256"]:
        raise ValueError(f"{condition} and {reference_info['condition']} start from different weights (09 §2.4)")


# ---------------------------------------------------------------- 优化器、单步、一致性占比

def make_optimizer(condition: str, model) -> torch.optim.Optimizer:
    """09 §2.2 的优化器列：PO-JEPA 类 `training.make_optimizer`，Transformer 类 v7 的 AdamW，Koopman 同 Transformer
    （07 §2.4），AC `ac_training.make_optimizer`（权重衰减 0.04，偏置和一维参数不衰减）。"""
    if condition in PO_CONDITIONS:
        return training.make_optimizer(model)
    if condition in TRANSFORMER_CONDITIONS:
        return comparison.make_optimizer(model)
    if condition == "koopman":
        return addon_conditions.make_koopman_optimizer(model)
    if condition in AC_CONDITIONS:
        return ac_training.make_optimizer(model, model.ac_condition)
    raise ValueError(f"{condition} is not trained")


def expected_readouts(condition: str, model) -> int | None:
    """07 §2.4：一次更新里 p_out 被调用的次数；AC 没有 p_out（None）。"""
    if condition in PO_CONDITIONS:
        return 5*checks.readouts_per_phi(model)
    return dict(matched_transformer=1, transformer_consistency=6, koopman=2).get(condition)


def train_step(condition: str, model, optimizer, batch, loss_config: LossConfig, completed: int, *, ac_state: str):
    """一步更新（09 §4.3）：返回 (TrainingStepResult, 本步学习率)。PO-JEPA 类走 `training.train_step`（fixed_fourier
    经 S10 的分发核谱模块类型），matched_transformer 走 `comparison.condition_step`，transformer_consistency、koopman 走
    S10 的单步，AC 走 `ac_training.train_step`（`ac_state: true_state` 时传片段的真实状态）。"""
    if condition in AC_CONDITIONS:
        states = batch.states if ac_state == "true_state" else None
        return ac_training.train_step(model.ac_condition, model, optimizer, batch.features, batch.actions, loss_config,
                                      completed, states=states)
    if condition in ("po_jepa_full", "po_jepa_no_consistency"):
        result = training.train_step(model, optimizer, batch, loss_config=loss_config, completed_steps=completed,
                                     condition=LOSS_LABELS[condition])
    elif condition == "matched_transformer":
        result = comparison.condition_step(condition, model, optimizer, batch, loss_config, completed)
    elif condition in ("fixed_fourier", "transformer_consistency", "koopman"):
        result = addon_conditions.condition_step(condition, model, optimizer, batch, loss_config, completed)
    else:
        raise ValueError(f"{condition} is not trained")
    return result, training.scheduled_lr(loss_config, completed)


def monitored_terms(condition: str, loss_config: LossConfig) -> tuple[str, ...]:
    """09 §4.6：至少一个一致性权重 > 0 的条件看权重 > 0 的项（按 time、space 的顺序）；其余条件不看（空元组）。"""
    return tuple(name for name, weight in (("time", loss_config.time_weight), ("space", loss_config.space_weight))
                 if weight > 0)


def consistency_share(condition: str, model, batch, loss_config: LossConfig, completed: int) -> dict:
    """09 §4.6：本步批上的三项梯度范数和 share = λ·w(s)·|∇L_c|/|∇L_JEPA|（v7 `comparison.consistency_share` 的公式和键；
    transformer_consistency 用通用梯度范数）。不改参数、优化器和随机状态。"""
    if condition == "transformer_consistency":
        norms = addon_calibration.generic_gradient_norms(model, batch, loss_config=loss_config)
    elif condition in PO_CONDITIONS:
        norms = training.consistency_gradient_norms(model, batch, loss_config=loss_config)
    else:
        raise ValueError(f"{condition} has no consistency terms to monitor")
    scale = training.consistency_warmup_scale(loss_config, completed)
    weights = dict(time=loss_config.time_weight, space=loss_config.space_weight)
    return dict(**norms, warmup_scale=scale,
                share={k: w*scale*norms["grad_"+k]/norms["grad_jepa"] if norms["grad_jepa"] > 0 else None
                       for k, w in weights.items()})
