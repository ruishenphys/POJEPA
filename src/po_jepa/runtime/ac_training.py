"""v7：V-JEPA 2-AC 基线的构造、优化器、片段读取、训练步与 checkpoint。

训练照官方 app/vjepa_droid/train.py（提交 204698b）：
- 片段：从共享查询序列的 (轨迹, 起点) 取 9 帧（8 次转移，官方 8 帧 4 fps 片段相当），每 tick 一个动作；
- 损失：teacher forcing（8 帧上下文一次前向，逐帧预测下一帧）+ 2 步自回归（auto_steps=2），
  均为 LayerNorm 后的 L1，loss = jloss + sloss；
- 优化器：AdamW(β=(0.9,0.999), eps=1e-8)，权重衰减 0.04（偏置与一维参数不衰减），梯度裁剪 1.0；
  学习率曲线与 PO-JEPA 同形（预热 5%、余弦降到峰值的 10%），峰值：官方规模微调 1e-4，缩小版从头 4.25e-4（官方峰值）；
- 官方规模在 CUDA 上用 bf16 autocast（官方 dtype: bfloat16），缩小版 float32。
与 PO-JEPA 的信息差别：AC 训练时看到片段内全部中间帧（官方做法），PO-JEPA 只用起点和终点。这让基线更强，报告里注明。
"""
from dataclasses import asdict
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
PEAK_LR = {"vjepa2_ac": 1e-4, "vjepa2_ac_small": 4.25e-4}
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
    return torch.optim.AdamW([dict(params=decay, weight_decay=WEIGHT_DECAY), dict(params=no_decay, weight_decay=0.)],
                             lr=PEAK_LR[condition], betas=(.9, .999), eps=1e-8)


def load_clips(features, rows, device) -> tuple[torch.Tensor, torch.Tensor]:
    """共享查询的每行 (轨迹, 起点) → 9 帧 [B,9,256,1408] float32 与 8 个 tick 动作 [B,8,2]（训练段）。"""
    clips, actions = [], []
    for q in rows:
        if data.datasets.active().split_for(q.trajectory_id) != "train":
            raise ValueError("AC clips must come from the train split")
        raw, m, _ = cache._read_complete_cache(Path(features)/f"{q.trajectory_id:04d}")
        start = q.start_tick
        if start+CLIP_FRAMES > raw.shape[0]:
            raise ValueError("clip exceeds trajectory")
        clips.append(torch.from_numpy(np.array(raw[start:start+CLIP_FRAMES], dtype=np.float32, copy=True)))
        c = m["controls"]; per_tick = np.zeros((32, 2))
        for s, e, v in zip(c["starts_ticks"], c["ends_ticks"], c["values"]):
            per_tick[s:e] = v
        actions.append(torch.tensor(per_tick[start:start+CLIP_FRAMES-1], dtype=torch.float32))
    return torch.stack(clips).to(device), torch.stack(actions).to(device)


def _lr(condition, loss_config, completed):
    return training.scheduled_lr(loss_config, completed)*PEAK_LR[condition]/training.BASE_LR


def train_step(condition, model, optimizer, clips, actions, loss_config: LossConfig, completed: int):
    """官方 teacher forcing + 2 步自回归；返回与其他条件同格式的 TrainingStepResult。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    e = clips.shape[-1]
    autocast = clips.device.type == "cuda" and condition == "vjepa2_ac"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        h = F.layer_norm(clips, (e,))
        z_tf = model.predict_frames(h[:, :-1], actions, 16, 16)                         # 预测第 1..8 帧
        jloss = (z_tf.float()-h[:, 1:]).abs().mean()
        frames = [h[:, 0], z_tf[:, 0]]
        for n in range(1, AUTO_STEPS):
            nxt = model.predict_frames(torch.stack(frames, 1), actions[:, :n+1], 16, 16)[:, -1]
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


def save_checkpoint(condition, path, model, optimizer, completed, sampling, provenance, info):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    device = next(model.parameters()).device
    payload = dict(schema=SCHEMA, condition=condition, spec=asdict(model.spec), info=info,
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
    if (payload["schema"] != SCHEMA or payload["condition"] != condition or payload["spec"] != asdict(model.spec)
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
