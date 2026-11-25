"""V-JEPA 2-AC 训练的 2 步自回归损失有没有错位（JEPA-WM 附录 C，arXiv 2512.24497v3：官方代码“拿真值 z1 拼预测 ẑ2、
被训练去输出 ẑ2”）。用一个精确的“神谕”预测器核对 runtime/ac_training.train_step 的对齐：

  h_{t+1} = roll(h_t, 1 + a_t)  （按通道循环移位，LayerNorm 不变；移位量取决于第 t 个 tick 的动作）

神谕 predict_frames 对第 t 帧用第 t 个动作做同样的移位（逐帧因果）。对齐正确时 teacher forcing 和 2 步滚动两项损失都恰好为 0；
动作错一位会让两项损失都非零（第 2 个测试）；第 3 个测试让预测和真值可分（乘 α），按神谕公式独立算出两项损失的期望值
并核对第 2 步的上下文，所以上下文拿真值、目标错一位、取错输出位置、滚动项复用 teacher forcing 项都会失败（Codex C3x P2-1）。"""
import torch
from torch import nn

from po_jepa.runtime import ac_training

from test_droid_s08b import _loss

E, N, B, T = 8, 4, 3, 9


class OracleAC(nn.Module):
    true_state = False

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = []

    def predict_frames(self, frames, actions, height, width, states):
        assert states is None and frames.ndim == 4 and actions.shape[:2] == frames.shape[:2]   # 第 k 帧配第 k 个动作
        self.calls.append(frames.shape[1])
        out = [torch.roll(frames[:, t], shifts=1+int(a), dims=-1)
               for t in range(frames.shape[1]) for a in [actions[0, t, 0].item()]]
        return self.scale*torch.stack(out, 1)


def clip_and_actions(seed=0):
    g = torch.Generator().manual_seed(seed)
    acts = torch.zeros(B, T-1, 2)
    acts[:, :, 0] = torch.randint(0, 3, (T-1,), generator=g).float()   # 每个样本同一串动作（神谕按样本 0 取），移位 1–3
    frames = [torch.randn(B, N, E, generator=g)]
    for t in range(T-1):
        frames.append(torch.roll(frames[-1], shifts=1+int(acts[0, t, 0]), dims=-1))
    return torch.stack(frames, 1), acts


def test_rollout_loss_is_aligned_with_the_next_frame():
    clips, acts = clip_and_actions()
    model = OracleAC()
    opt = torch.optim.SGD(model.parameters(), lr=0.)
    res, _ = ac_training.train_step("vjepa2_ac_small", model, opt, clips, acts, _loss(), 0)
    assert model.calls == [T-1, 2]                     # teacher forcing 一次 8 帧，第 2 步自回归的上下文 [h0, ẑ1]
    assert res.loss_values["ac_teacher_forcing"] < 1e-6
    assert res.loss_values["ac_rollout"] < 1e-6


def test_misaligned_actions_would_be_caught():
    """对照：神谕错用下一个 tick 的动作（错一位）时，两项损失都明显非零（说明上面的 0 不是平凡的）。"""
    clips, acts = clip_and_actions(1)
    model = OracleAC()
    shifted = torch.cat([acts[:, 1:], acts[:, :1]], 1)
    assert not torch.equal(shifted, acts)
    res, _ = ac_training.train_step("vjepa2_ac_small", model, torch.optim.SGD(model.parameters(), lr=0.), clips, shifted, _loss(), 0)
    assert res.loss_values["ac_teacher_forcing"] > 1e-2 and res.loss_values["ac_rollout"] > 1e-2


def test_second_step_context_is_true_start_then_own_prediction():
    """附录 C 说的错法是“真值 z1 拼预测 ẑ2”。这里神谕输出乘 α = 1.001（预测和真值可分），记下两次调用的输入输出：
    第 2 步的上下文必须是 [真值 h0, 第 1 步自己的预测 ẑ1 = α h1]，最后一帧输出 α² h2 和 h2 比。期望损失只由输入和神谕公式构造：
      L_TF = mean |α h_{1:9} − h_{1:9}|,   L_AR = mean |[α h1, α² h2] − [h1, h2]|。"""
    clips, acts = clip_and_actions(2)
    model = OracleAC()
    with torch.no_grad():
        model.scale.fill_(1.001)
    seen = []
    original = model.predict_frames
    model.predict_frames = lambda frames, *a: seen.append((frames.detach().clone(), out := original(frames, *a))) or out
    alpha = 1.001
    res, _ = ac_training.train_step("vjepa2_ac_small", model, torch.optim.SGD(model.parameters(), lr=0.), clips, acts, _loss(), 0)
    h = torch.nn.functional.layer_norm(clips, (E,))
    expected_tf = (alpha*h[:, 1:]-h[:, 1:]).abs().mean().item()
    expected_ar = (torch.stack([alpha*h[:, 1], alpha**2*h[:, 2]], 1)-h[:, 1:3]).abs().mean().item()
    assert abs(res.loss_values["ac_teacher_forcing"]-expected_tf) <= 1e-4*expected_tf   # float32 舍入
    assert abs(res.loss_values["ac_rollout"]-expected_ar) <= 1e-4*expected_ar
    assert expected_ar > 1.2*expected_tf                       # 两项可分：滚动项复用 teacher forcing 项会被抓到
    # 梯度也经过滚动项（α > 1 时 |·| 处处可导）：d/dα L_TF = mean|h_{1:9}|，d/dα L_AR = mean([|h1|, 2α|h2|])
    expected_grad = h[:, 1:].abs().mean().item()+torch.stack([h[:, 1].abs(), 2*alpha*h[:, 2].abs()], 1).mean().item()
    assert abs(res.grad_norm-expected_grad) <= 1e-4*expected_grad
    (tf_in, tf_out), (ar_in, ar_out) = seen
    assert torch.equal(tf_in, h[:, :-1])
    assert torch.equal(ar_in[:, 0], h[:, 0]) and torch.equal(ar_in[:, 1], tf_out[:, 0].detach())
    assert not torch.allclose(ar_in[:, 1], h[:, 1])            # 第二帧上下文是预测，不是真值
