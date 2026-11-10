"""DROID 上 V-JEPA 2-AC 的起点状态绑定和小步前向（docs/design/09-droid-training.md §6.2、10 §8.4、§8.6，S08b）。

- `bind(model, start_state, reference)`：`ac_state: true_state` 时评测（S11）、规划（S12）、验证（S09）用的只读包装，
  前向签名和其他模型相同 `phi(raw, controls, duration, grid)`。`reference` 是窗口完整的 `ActionSegments`，
  `start_state` 是窗口起点的真实状态（CPU float64 [B, 7]，物理单位）。每次调用先算这次调用的起点相对窗口起点的周期数
  τ = (controls.source_origin_seconds − reference.source_origin_seconds) / P，再用 `reference` 里 [0, τ) 的物理动作把状态
  积分过去：整周期 j < ⌊τ⌋ 各做一次 `compute_new_pose(状态, a_j)`，不满一个周期的余下 r = τ − ⌊τ⌋ 做一次
  `compute_new_pose(状态, r·a_⌊τ⌋)`（09 §6.2 “非整数起点”）；a_j 是 `ac_baseline.physical_tick_actions` 给的第 j 个周期的
  物理动作（时间平均 × scale + offset）。然后调用 `model(raw, controls, duration, grid, start_state=积分后的状态)`。
  τ 越出 [0, reference.duration] 就报错；控制单位、来源编号、批大小必须和 `reference` 相同。动作干预时调用者另建一个以
  替身动作为 `reference`、起点状态仍是本窗口的包装（evaluate、planning 已这样做）。
- `ac_steps(model, raw, actions, start_state, grid)`：原始帧率评测的小步前向（S11b 的接口），转交
  `ACPredictor.small_steps`；learned_constant 模式下忽略 `start_state`。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from po_jepa.ac_baseline import ACPredictor, physical_tick_actions
from po_jepa.actions import ActionSegments, validate_action_segments
from po_jepa.droid.poses import compute_new_pose
from po_jepa.geometry import GridSpec

SNAP = 1e-9     # τ 离整数不到这么多个周期就当整数（物理秒相减的舍入）


def _check_state(start_state, batch: int, name: str = "start_state") -> Tensor:
    if (not isinstance(start_state, Tensor) or start_state.dtype != torch.float64 or start_state.device.type != "cpu"
            or start_state.shape != (batch, 7) or not torch.isfinite(start_state).all()):
        raise ValueError(f"{name} must be finite CPU float64 [B, 7] (physical units)")
    return start_state


def offsets(reference: ActionSegments, controls: ActionSegments) -> Tensor:
    """这次调用的起点相对窗口起点的周期数 τ，CPU float64 [B]；离整数 ≤ SNAP 的取整，越出 [0, 窗口时长] 报错。"""
    if controls.control_spec != reference.control_spec or controls.source_ids != reference.source_ids:
        raise ValueError("call controls must share the reference's control spec and source ids")
    if controls.source_origin_seconds.shape != reference.source_origin_seconds.shape:
        raise ValueError("call controls and reference must have the same batch")
    period = reference.control_spec.control_period_seconds
    tau = (controls.source_origin_seconds.double() - reference.source_origin_seconds.double())/period
    tau = torch.where((tau - tau.round()).abs() <= SNAP, tau.round(), tau)
    if (not torch.isfinite(tau).all() or torch.any(tau < 0)
            or torch.any(tau > reference.duration.double() + SNAP)):
        raise ValueError("call start lies outside the reference window [0, duration]")
    return tau


def advance(start_state: Tensor, physical: Tensor, tau: Tensor) -> Tensor:
    """把起点状态按 `physical`（[B, K, 7] float64，每周期的物理动作）积分 τ 个周期（09 §6.2），CPU float64 [B, 7]。
    同一步上需要积分的样本一起调用 compute_new_pose（逐行独立，结果和逐个调用相同）。"""
    state = start_state.clone()
    whole = torch.floor(tau).to(torch.int64)
    rest = tau - whole.double()
    for j in range(int(whole.max()) if whole.numel() else 0):
        rows = (whole > j).nonzero().flatten()
        if rows.numel():
            state[rows] = compute_new_pose(state[rows][:, None], physical[rows, j][:, None])[:, 0]
    rows = (rest > 0).nonzero().flatten()
    if rows.numel():
        part = rest[rows, None]*physical[rows, whole[rows]]
        state[rows] = compute_new_pose(state[rows][:, None], part[:, None])[:, 0]
    return state


class BoundAC:
    """`bind` 返回的只读包装；不是 nn.Module，不持有参数，`model` 的状态不变。"""

    def __init__(self, model: ACPredictor, start_state: Tensor, reference: ActionSegments):
        if not isinstance(model, ACPredictor) or not model.true_state:
            raise ValueError("bind is for AC models with state_input true_state")
        if not isinstance(reference, ActionSegments) or reference.control_spec.action_dim != 7:
            raise ValueError("reference must be the window's 7-dimensional ActionSegments")
        validate_action_segments(reference, query_duration=reference.duration)
        b = reference.duration.shape[0]
        self._model = model
        self._start = _check_state(start_state, b).clone()
        self._reference = reference
        steps = max(1, int(math.ceil(float(reference.duration.max()))))
        self._physical = physical_tick_actions(reference, steps)                       # CPU float64 [B, K, 7]

    @property
    def model(self) -> ACPredictor:
        return self._model

    def state_at(self, controls: ActionSegments) -> Tensor:
        """这次调用起点的状态（CPU float64 [B, 7]）。"""
        return advance(self._start, self._physical, offsets(self._reference, controls))

    def __call__(self, raw: Tensor, controls: ActionSegments, duration: Tensor, grid: GridSpec) -> Tensor:
        return self._model(raw, controls, duration, grid, start_state=self.state_at(controls))


def bind(model: ACPredictor, start_state: Tensor, reference: ActionSegments) -> BoundAC:
    """09 §6.2 的 `droid.ac.bind`；evaluate、planning 的 `bind` 参数就传这个函数。"""
    return BoundAC(model, start_state, reference)


def ac_steps(model: ACPredictor, raw: Tensor, actions: Tensor, start_state: Tensor, grid: GridSpec) -> Tensor:
    """10 §8.4 的小步前向（S11b 的 `ac_steps` 接口）：`actions` 物理单位 CPU float64 [B, n, 7]，`start_state` 这次调用
    起点的状态 CPU float64 [B, 7]；返回 n 个小步之后的预测。learned_constant 模式不用状态。"""
    if not isinstance(model, ACPredictor):
        raise TypeError("ac_steps needs an ACPredictor")
    _check_state(start_state, raw.shape[0])
    return model.small_steps(raw, actions, start_state if model.true_state else None, grid)
