"""v7：V-JEPA 2-AC 基线的构造、优化器、片段读取、训练步与 checkpoint。

训练照官方 app/vjepa_droid/train.py（提交 204698b）：
- 片段：从共享查询序列的 (轨迹, 起点) 取 9 帧（8 次转移，官方 8 帧 4 fps 片段相当），每 tick 一个动作；
- 损失：teacher forcing（8 帧上下文一次前向，逐帧预测下一帧）+ 2 步自回归（auto_steps=2），
  均为 LayerNorm 后的 L1，loss = jloss + sloss；
- 优化器：AdamW(β=(0.9,0.999), eps=1e-8)，权重衰减 0.04（偏置与一维参数不衰减），梯度裁剪 1.0；
  学习率曲线与 PO-JEPA 同形（预热 5%、余弦降到峰值的 10%），峰值：官方规模微调 1e-4，缩小版从头 4.25e-4（官方峰值）；
- 官方规模在 CUDA 上用 bf16 autocast（官方 dtype: bfloat16），缩小版 float32。
与 PO-JEPA 的信息差别：AC 训练时看到片段内全部中间帧（官方做法），PO-JEPA 只用起点和终点。这让基线更强，报告里注明。
DROID（S08b，09 §6.2）：`train_step(…, states=…)` 在 `state_input: true_state` 时收真实状态 [B, 9, 7]，teacher forcing
用 states[:, :-1]，第 n 步自回归用 states[:, :n+1]（和官方 train.py 相同，训练时用真实状态）；动作是官方物理单位
（DROID 片段批 `droid.batches.make_ac_clip_batch`）。规格快照一律 `ac_baseline.spec_snapshot`，v7 的逐字节不变。

DINO-WM、JEPA-WM 参数匹配版（2026-11-25，照 jepa-wms @ 13cf1d9 的 app/vjepa_wm/train.py、video_wm.py 和两份 Metaworld 最终配置
mw_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_1roll_save_seed1.yaml、mw_4f_fsk5_ask1_r224_pred_AdaLN_ftprop_depth6_repro_2roll_save.yaml）：
- 片段 4 帧（num_hist 3 + num_pred 1），和 AC 用同一批 (轨迹, 起点)，只取前 4 帧、前 3 个 tick 动作；
- 损失 MSE（l2_loss_weight 1），目标是 LN 后的特征（和 AC 同一空间）；teacher forcing：前 3 帧一次前向，逐帧预测第 1…3 帧；
- dino_wm_small：只有 teacher forcing（rollout_steps 1），loss = L_tf / 2（上游 predictor_loss / (rollout_steps + 1)）；
- jepa_wm_small：再加 1 步顺序滚动（rollout_steps 2）：前缀 t ∈ {0, 1} 均匀取一个，上下文 = [真值第 0…t 帧, teacher forcing 的第 t+1 帧预测]
  （整段停梯度，rollout_stop_gradient），动作第 0…t+1 个，截最近 3 帧（ctxt_window_train_rollout 3），预测第 t+2 帧；
  loss = L_tf / 2 + L_roll / 2（上游两项各除以 rollout_steps）。上游每批用 torch.randint 取 t，这里按 (模型种子, 已完成步数)
  确定地取（同一分布，不碰全局随机状态，续训逐位可复现）；
- 优化器：AdamW(β=(0.9,0.999), eps=1e-8)，权重衰减 1e-7（上游 weight_decay），偏置与一维参数不衰减；峰值学习率 5e-4（上游 ref_lr），
  曲线和 AC 一样与 PO-JEPA 同形（上游是常数学习率、无预热，这里为和其余条件同一套步数/早停口径改成同形曲线）；float32。
"""
import hashlib
import io
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from po_jepa import ac_baseline
from po_jepa.losses import LossConfig
from . import cache, data, training

CLIP_FRAMES = 9
AUTO_STEPS = 2
WEIGHT_DECAY = 0.04
PEAK_LR = {"vjepa2_ac": 1e-4, "vjepa2_ac_small": 4.25e-4, "dino_wm_small": 5e-4, "jepa_wm_small": 5e-4}
# 2026-11-25：DINO-WM、JEPA-WM（jepa-wms 配置）
WM_CLIP_FRAMES = 4          # num_hist 3 + num_pred 1
WM_WEIGHT_DECAY = 1e-7
WM_ROLLOUT_STEPS = {"dino_wm_small": 1, "jepa_wm_small": 2}   # 上游 rollout_steps（1 = 只有 teacher forcing）


def clip_frames(condition) -> int:
    return WM_CLIP_FRAMES if condition in ac_baseline.WM_CONDITIONS else CLIP_FRAMES


def weight_decay(condition) -> float:
    return WM_WEIGHT_DECAY if condition in ac_baseline.WM_CONDITIONS else WEIGHT_DECAY
CHECKPOINT_ENV = "PO_JEPA_AC_CHECKPOINT"
SCHEMA = "v7_vjepa2_ac_checkpoint_v1"


def _official_state(path: Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    content = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    for key in ("predictor", "ema_predictor"):
        if isinstance(content, dict) and key in content:
            return content[key], digest
    raise KeyError("official AC checkpoint has no 'predictor' entry")


def build(condition: str, device) -> tuple[ac_baseline.ACPredictor, dict]:
    """模型初始化 RNG 局部隔离（种子同 PO-JEPA）；官方规模从 PO_JEPA_AC_CHECKPOINT 指向的官方权重载入。"""
    from . import checks
    spec = ac_baseline.spec_for(condition)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(checks.model_seed())
        model = ac_baseline.ACPredictor(spec)
    info = ac_baseline.describe(condition)
    if condition == "vjepa2_ac":
        path = os.environ.get(CHECKPOINT_ENV)
        if not path or not Path(path).is_file():
            raise ValueError("vjepa2_ac 需要官方权重：run.py 加 --ac-checkpoint <vjepa2-ac-vitg.pt>（下载地址 "
                             + ac_baseline.OFFICIAL_CHECKPOINT_URL+"）")
        state, digest = _official_state(Path(path))
        info.update(official_checkpoint_sha256=digest, official_load=model.load_official(state))
    info["parameters"] = ac_baseline.parameter_count(model)
    return model.to(device), info


def make_optimizer(model, condition):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return torch.optim.AdamW([dict(params=decay, weight_decay=weight_decay(condition)), dict(params=no_decay, weight_decay=0.)],
                             lr=PEAK_LR[condition], betas=(.9, .999), eps=1e-8)


def load_clips(features, rows, device, frames=CLIP_FRAMES) -> tuple[torch.Tensor, torch.Tensor]:
    """共享查询的每行 (轨迹, 起点) → frames 帧 [B,frames,256,1408] float32 与 frames−1 个 tick 动作 [B,frames−1,2]（训练段）。
    AC 9 帧；DINO-WM、JEPA-WM 4 帧（clip_frames），同一起点的前 4 帧。"""
    if frames not in (CLIP_FRAMES, WM_CLIP_FRAMES):
        raise ValueError("clip length must be the AC or the DINO-WM/JEPA-WM one")
    clips, actions = [], []
    for q in rows:
        if data.datasets.active().split_for(q.trajectory_id) != "train":
            raise ValueError("AC clips must come from the train split")
        raw, m, _ = cache._read_complete_cache(Path(features)/f"{q.trajectory_id:04d}")
        start = q.start_tick
        if start+frames > raw.shape[0]:
            raise ValueError("clip exceeds trajectory")
        clips.append(torch.from_numpy(np.array(raw[start:start+frames], dtype=np.float32, copy=True)))
        c = m["controls"]; per_tick = np.zeros((32, 2))
        for s, e, v in zip(c["starts_ticks"], c["ends_ticks"], c["values"]):
            per_tick[s:e] = v
        actions.append(torch.tensor(per_tick[start:start+frames-1], dtype=torch.float32))
    return torch.stack(clips).to(device), torch.stack(actions).to(device)


def _lr(condition, loss_config, completed):
    return training.scheduled_lr(loss_config, completed)*PEAK_LR[condition]/training.BASE_LR


def _clip_states(model, clips, states):
    if not model.true_state:
        if states is not None:
            raise ValueError("learned_constant AC takes no states")
        return None
    b, t = clips.shape[:2]
    if (not isinstance(states, torch.Tensor) or states.shape != (b, t, 7) or states.device != clips.device
            or not states.is_floating_point() or not torch.isfinite(states).all()):
        raise ValueError("true_state AC needs finite states [B, clip frames, 7] on the clip device")
    return states


def train_step(condition, model, optimizer, clips, actions, loss_config: LossConfig, completed: int, states=None):
    """官方 teacher forcing + 2 步自回归；返回与其他条件同格式的 TrainingStepResult。
    `states`：true_state 时真实状态 [B, 9, 7]（S08b），learned_constant 时必须为 None（v7）。
    DINO-WM、JEPA-WM 走 `_wm_losses`（见文件头）。"""
    if condition in ac_baseline.WM_CONDITIONS:
        return _wm_step(condition, model, optimizer, clips, actions, loss_config, completed, states)
    states = _clip_states(model, clips, states)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    e = clips.shape[-1]
    autocast = clips.device.type == "cuda" and condition == "vjepa2_ac"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        h = F.layer_norm(clips, (e,))
        z_tf = model.predict_frames(h[:, :-1], actions, 16, 16,
                                    None if states is None else states[:, :-1])          # 预测第 1..8 帧
        jloss = (z_tf.float()-h[:, 1:]).abs().mean()
        frames = [h[:, 0], z_tf[:, 0]]
        for n in range(1, AUTO_STEPS):
            nxt = model.predict_frames(torch.stack(frames, 1), actions[:, :n+1], 16, 16,
                                       None if states is None else states[:, :n+1])[:, -1]
            frames.append(nxt)
        z_ar = torch.stack(frames[1:], 1)
        sloss = (z_ar.float()-h[:, 1:AUTO_STEPS+1]).abs().mean()
        loss = jloss+sloss
    if not torch.isfinite(loss):
        raise ValueError("nonfinite AC loss")
    loss.backward()
    parameters = [p for p in model.parameters()]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError("AC gradient missing or nonfinite")
    maximum = max(p.grad.detach().abs().max().item() for p in parameters)
    grad_norm = training.clip_gradients(parameters, loss_config.grad_clip_norm)
    lr = _lr(condition, loss_config, completed)
    saved = [g["lr"] for g in optimizer.param_groups]
    for g in optimizer.param_groups: g["lr"] = lr
    try:
        optimizer.step()
    finally:
        for g, value in zip(optimizer.param_groups, saved): g["lr"] = value
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("updated AC parameter nonfinite")
    losses = dict(jepa=jloss.detach().item(), time=None, space=None, weighted_time=0., weighted_space=0.,
                  total=loss.detach().item(), ac_teacher_forcing=jloss.detach().item(), ac_rollout=sloss.detach().item())
    return training.TrainingStepResult(completed+1, losses, maximum, grad_norm), lr


def rollout_prefix(completed: int) -> int:
    """JEPA-WM 顺序滚动的前缀 t ∈ {0, …, 4 − rollout_steps − 1} = {0, 1}：按 (模型种子, 已完成步数) 确定地均匀取。"""
    from . import checks
    span = WM_CLIP_FRAMES-WM_ROLLOUT_STEPS["jepa_wm_small"]
    return int(np.random.default_rng([checks.model_seed(), int(completed)]).integers(span))


def _wm_losses(condition, model, clips, actions, completed):
    """→ (loss, 记录的分项)。clips [B,4,N,E]（N = 256），actions [B,3,m]；MSE，目标是 LN 后的特征。"""
    if model.spec.conditioning == "token" or clips.shape[1] != WM_CLIP_FRAMES or actions.shape[1] != WM_CLIP_FRAMES-1:
        raise ValueError("DINO-WM / JEPA-WM train on 4-frame clips with 3 actions")
    e = clips.shape[-1]
    h = F.layer_norm(clips, (e,))
    z_tf = model.predict_frames(h[:, :-1], actions, 16, 16)                      # 预测第 1..3 帧
    tf = F.mse_loss(z_tf, h[:, 1:])
    steps = WM_ROLLOUT_STEPS[condition]
    if steps == 1:
        return tf/2, dict(ac_teacher_forcing=tf.detach().item())
    t = rollout_prefix(completed)
    window = model.spec.context_frames
    context = torch.cat([h[:, :t+1], z_tf[:, t:t+1]], 1).detach()[:, -window:]   # rollout_stop_gradient
    nxt = model.predict_frames(context, actions[:, :t+2][:, -window:], 16, 16)[:, -1]
    roll = F.mse_loss(nxt, h[:, t+2])
    return tf/2+roll/2, dict(ac_teacher_forcing=tf.detach().item(), ac_rollout=roll.detach().item(), ac_rollout_prefix=t)


def _wm_step(condition, model, optimizer, clips, actions, loss_config, completed, states):
    if states is not None:
        raise ValueError("DINO-WM / JEPA-WM take no states")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, parts = _wm_losses(condition, model, clips, actions, completed)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite AC loss")
    loss.backward()
    parameters = [p for p in model.parameters()]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError("AC gradient missing or nonfinite")
    maximum = max(p.grad.detach().abs().max().item() for p in parameters)
    grad_norm = training.clip_gradients(parameters, loss_config.grad_clip_norm)
    lr = _lr(condition, loss_config, completed)
    saved = [g["lr"] for g in optimizer.param_groups]
    for g in optimizer.param_groups: g["lr"] = lr
    try:
        optimizer.step()
    finally:
        for g, value in zip(optimizer.param_groups, saved): g["lr"] = value
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("updated AC parameter nonfinite")
    losses = dict(jepa=parts["ac_teacher_forcing"], time=None, space=None, weighted_time=0., weighted_space=0.,
                  total=loss.detach().item(), **parts)
    return training.TrainingStepResult(completed+1, losses, maximum, grad_norm), lr


def save_checkpoint(condition, path, model, optimizer, completed, sampling, provenance, info):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    device = next(model.parameters()).device
    payload = dict(schema=SCHEMA, condition=condition, spec=ac_baseline.spec_snapshot(model.spec), info=info,
                   model_state=training._cpu_tree(dict(model.state_dict())),
                   optimizer_state=training._cpu_tree(optimizer.state_dict()), provenance=provenance,
                   completed_steps=completed, sampling_state=training._sampling(sampling), model_training=model.training,
                   rng_state=training._rng_state(device))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="."+path.name, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name); torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        return cache.file_sha(path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def load_checkpoint(condition, path, expected_sha, model, optimizer, expected_provenance):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("checkpoint SHA mismatch before loading")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (payload["schema"] != SCHEMA or payload["condition"] != condition or payload["spec"] != ac_baseline.spec_snapshot(model.spec)
            or not training._same_native(payload["provenance"], expected_provenance)):
        raise ValueError("AC checkpoint condition/spec/provenance mismatch")
    device = next(model.parameters()).device
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    model.train(payload["model_training"])
    legacy = training._check_rng(payload["rng_state"], device)
    random.setstate(payload["rng_state"]["python"]); np.random.set_state(legacy)
    torch.set_rng_state(payload["rng_state"]["torch_cpu"])
    if device.type == "cuda": torch.cuda.set_rng_state(payload["rng_state"]["cuda"]["state"], device=device)
    return training.CheckpointProgress(payload["completed_steps"], training._sampling(payload["sampling_state"]))
