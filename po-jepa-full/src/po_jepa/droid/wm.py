"""DROID 上的 DINO-WM、JEPA-WM 参数匹配基线（dino_wm_small、jepa_wm_small；docs/design/09-droid-training.md §14）。

模型就是仿真路径的那两个（`ac_baseline.ACPredictor` 的 conditioning = feature / adaln，核心在 `wm_baselines.py`，
上游代码 `third_party/jepa_wms.py`，facebookresearch/jepa-wms @ 13cf1d9，CC BY-NC 4.0），只把动作维改成 7（DROID 的
物理单位动作，和 AC 同一套 tick 动作），前馈宽按 DROID 的 PO-JEPA（248,737 参数）重新匹配（`conditions.match_wm`）。
仿真训练会读到的源码（ac_baseline、ac_training、wm_baselines、jepa_wms）一个字节都不改；DROID 特有的三处写在这里：

1. 训练片段：和 AC 同一批 (回合, 起点) 的 9 帧片段（`batches.ACClipBatch`），只取前 4 帧、前 3 个动作
   （上游 num_hist 3 + num_pred 1；和仿真 `ac_training.load_clips(frames=4)` 同一取法）。
2. 损失：和仿真 `ac_training._wm_losses` 逐项相同（MSE，目标是 LN 后的特征；DINO-WM L_tf/2；JEPA-WM L_tf/3 + L_roll/2，
   1 步顺序滚动、上下文停梯度、窗口 3）。唯一的差别是滚动前缀 t ∈ {0, 1} 按 (本任务种子, 已完成步数) 取：
   仿真用 `checks.model_seed()`（仿真协议里的种子），DROID 不走仿真协议，那里永远是 1703，三个种子会取到同一串前缀。
   测试核对：同一前缀时损失和梯度与 `ac_training._wm_losses` 逐位相同。
3. 优化器：`ac_training.make_optimizer`（AdamW，wd 1e-7，偏置和一维参数不衰减，峰值 5e-4，曲线和其余条件同形），
   jepa_wm_small 的 β 换成上游 DROID 最终配置的 (0.9, 0.995)（droid_4fpcs_fps4_r256_dv3vitl_asp1_pred_AdaLN_depth12_
   noprop_repro_2roll_4n.yaml；Metaworld 配置和仿真版是 (0.9, 0.999)）；dino_wm_small 上游 DROID 配置也是 (0.9, 0.999)，不变。

两者都不吃本体状态（no-prop）：上游 DROID 的两份最终配置都是 noprop（AdaLN：`proprio_encoding: none`；dino_wm：
`proprio_emb_dim: 0`），ftprop 只在 Metaworld 上用。所以 `ac_state: true_state` 对它们不起作用：训练不传状态，验证、
评测、规划都不绑定起点状态（`conditions.binds_state` 为假），原始帧率评测的小步前向忽略状态。这和 vjepa2_ac、
vjepa2_ac_small（DROID 上吃真实状态）不同，比较时要注明。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from po_jepa import ac_baseline
from po_jepa.runtime import ac_training, training

WM_CONDITIONS = ac_baseline.WM_CONDITIONS           # ("dino_wm_small", "jepa_wm_small")
CLIP_FRAMES = ac_training.WM_CLIP_FRAMES            # 4
ROLLOUT_STEPS = ac_training.WM_ROLLOUT_STEPS        # dino 1（只有 teacher forcing），jepa 2
GRID = 16                                           # 训练特征 16×16
DROID_BETAS = {"dino_wm_small": (0.9, 0.999), "jepa_wm_small": (0.9, 0.995)}   # 上游 DROID 最终配置
PROPRIO = "none"                                    # 上游 DROID 配方：no-prop


def _check(condition: str) -> None:
    if condition not in WM_CONDITIONS:
        raise ValueError(f"{condition} is not a DINO-WM / JEPA-WM condition")


def make_optimizer(condition: str, model) -> torch.optim.Optimizer:
    """`ac_training.make_optimizer` 的分组、衰减和峰值；β 取上游 DROID 配置（见文件头第 3 条）。"""
    _check(condition)
    optimizer = ac_training.make_optimizer(model, condition)
    for group in optimizer.param_groups:
        group["betas"] = DROID_BETAS[condition]
    return optimizer


def rollout_prefix(seed: int, completed: int) -> int:
    """JEPA-WM 顺序滚动的前缀 t ∈ {0, …, 4 − rollout_steps − 1} = {0, 1}：按 (本任务种子, 已完成步数) 确定地均匀取
    （和 `ac_training.rollout_prefix` 同一公式，只把仿真协议的种子换成 DROID 任务的种子）。"""
    if type(seed) is not int or seed < 0 or type(completed) is not int or completed < 0:
        raise ValueError("seed and completed must be non-negative ints")
    span = CLIP_FRAMES - ROLLOUT_STEPS["jepa_wm_small"]
    return int(np.random.default_rng([seed, completed]).integers(span))


def clips(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """AC 片段批 → 前 4 帧 [C, 4, 256, 1408] 和前 3 个物理动作 [C, 3, 7]（第 k 个动作把第 k 帧带到第 k + 1 帧）。"""
    if batch.features.shape[1] < CLIP_FRAMES or batch.actions.shape[1] != batch.features.shape[1] - 1:
        raise ValueError("AC clip batch must hold at least 4 frames and one action per transition")
    return batch.features[:, :CLIP_FRAMES], batch.actions[:, :CLIP_FRAMES - 1]


def losses(condition: str, model, frames: torch.Tensor, actions: torch.Tensor, prefix: int | None):
    """→ (loss, 记录的分项)；和 `ac_training._wm_losses` 逐项相同，前缀由调用者给（DINO-WM 不用，传 None）。"""
    _check(condition)
    if (model.spec.conditioning == "token" or model.true_state or frames.shape[1] != CLIP_FRAMES
            or actions.shape[1] != CLIP_FRAMES - 1 or actions.shape[2] != 7):
        raise ValueError("DINO-WM / JEPA-WM train on 4-frame clips with 3 seven-dimensional actions and no states")
    e = frames.shape[-1]
    h = F.layer_norm(frames, (e,))
    z_tf = model.predict_frames(h[:, :-1], actions, GRID, GRID)                   # 预测第 1..3 帧
    tf = F.mse_loss(z_tf, h[:, 1:])
    steps = ROLLOUT_STEPS[condition]
    if steps == 1:
        if prefix is not None:
            raise ValueError("DINO-WM has no rollout term, so no prefix")
        return tf/2, dict(ac_teacher_forcing=tf.detach().item())
    if prefix not in range(CLIP_FRAMES - steps):
        raise ValueError(f"rollout prefix must be in [0, {CLIP_FRAMES - steps})")
    t = prefix
    window = model.spec.context_frames
    context = torch.cat([h[:, :t+1], z_tf[:, t:t+1]], 1).detach()[:, -window:]   # rollout_stop_gradient
    nxt = model.predict_frames(context, actions[:, :t+2][:, -window:], GRID, GRID)[:, -1]
    roll = F.mse_loss(nxt, h[:, t+2])
    return tf/(steps+1) + roll/steps, dict(ac_teacher_forcing=tf.detach().item(), ac_rollout=roll.detach().item(),
                                           ac_rollout_prefix=t)


def train_step(condition: str, model, optimizer, batch, loss_config, completed: int, *, seed: int):
    """一步更新：返回 (TrainingStepResult, 本步学习率)，格式和检查同 `ac_training._wm_step`（梯度裁剪、学习率曲线、
    有限性检查都一样）。不传状态（no-prop）。"""
    _check(condition)
    frames, actions = clips(batch)
    prefix = rollout_prefix(seed, completed) if ROLLOUT_STEPS[condition] > 1 else None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, parts = losses(condition, model, frames, actions, prefix)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite AC loss")
    loss.backward()
    parameters = [p for p in model.parameters()]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError("AC gradient missing or nonfinite")
    maximum = max(p.grad.detach().abs().max().item() for p in parameters)
    grad_norm = training.clip_gradients(parameters, loss_config.grad_clip_norm)
    lr = ac_training._lr(condition, loss_config, completed)
    saved = [g["lr"] for g in optimizer.param_groups]
    for g in optimizer.param_groups:
        g["lr"] = lr
    try:
        optimizer.step()
    finally:
        for g, value in zip(optimizer.param_groups, saved):
            g["lr"] = value
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("updated AC parameter nonfinite")
    values = dict(jepa=parts["ac_teacher_forcing"], time=None, space=None, weighted_time=0., weighted_space=0.,
                  total=loss.detach().item(), **parts)
    return training.TrainingStepResult(completed+1, values, maximum, grad_norm), lr
