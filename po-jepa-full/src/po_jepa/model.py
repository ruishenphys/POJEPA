"""图1的逐 token 投影与完整 Z→Z 预测器。

P_in/P_out 为 1408↔64 线性投影，采用 Xavier uniform 与零偏置。
每次完整 Phi 重建图谱和上下文；零时长也经谱投影与读出，不强制恒等。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .actions import ActionSegments
from .dynamics import DynamicsContext, IntegrationResult, IntegratorConfig, ModalDynamics
from .geometry import GridSpec
from .spectral import GraphSpectrum, SpectralResult, modal_analysis, modal_synthesis


@dataclass(frozen=True)
class ProjectionConfig:
    """无量纲投影通道配置：编码器宽 E，动力学宽固定 64。

    生产 E=1408；小 E 仅用于合成测试。
    """

    encoder_dim: int
    dynamic_dim: int
    # v4：readout="residual" 时 Z_hat=Z+P_out(Psi C_tau)-P_out(Psi C_0)，零时长严格恒等；
    # "direct" 保持一轮 Z_hat=P_out(Psi C_tau)。
    readout: str = "direct"

    def __post_init__(self) -> None:
        """分配参数前拒绝布尔数、非整数和未约定宽度。"""
        if self.readout not in ("direct", "residual"):
            raise ValueError("readout must be 'direct' or 'residual'")
        if type(self.encoder_dim) is not int or self.encoder_dim <= 0:
            raise ValueError("encoder_dim must be a positive integer")
        if type(self.dynamic_dim) is not int or self.dynamic_dim != 64:
            raise ValueError("dynamic_dim must be the fixed first-round width 64")


@dataclass(frozen=True)
class QueryContext:
    """单次 Phi 起点的工作量，仅本次查询或内部续算有效。

    X [B,N,64]、C0 [B,K,64] 为模型 float32 并保留梯度；grid 为 CPU
    float64，谱 lambda/Psi 无梯度，g/A/B/两偏置保留图。字段只读，不跨优化步缓存。
    """

    projected: Tensor
    grid: GridSpec
    spectrum: SpectralResult
    initial_coefficients: Tensor
    dynamics: DynamicsContext


@dataclass(frozen=True)
class PredictionTrace:
    """同次实际预测和诊断，使用后释放计算图。

    raw_prediction [B,N,E]、synthesized [B,N,64] 为模型 float32；query 与
    integration 保存起点和实际积分量，逐步快照仅 collect_trace 时保留。
    式15重启只接 raw_prediction，不接整个 trace 或隐藏状态。
    """

    raw_prediction: Tensor
    query: QueryContext
    integration: IntegrationResult
    synthesized: Tensor


class POJEPA(nn.Module):
    """共享所有网格参数的完整预测器，P_in/P_out 对应 p_in/p_out。

    每次完整预测重建图谱/上下文；内部续算显式复用 query。无跨查询成员
    缓存；forward 返回 raw，trace_forward 额外保留同次中间量而不重跑网络。
    """

    def __init__(self, projection: ProjectionConfig, spectrum: GraphSpectrum,
                 dynamics: ModalDynamics, integrator: IntegratorConfig) -> None:
        """构造 E→64 与 64→E 两层线性投影，新增 129E+64 个参数。

        CPU float32 分配，仅消耗调用者 torch RNG；无激活、LN 或残差，不改变
        注入组件和全局默认设备/精度，设备转移由调用者负责。
        """
        super().__init__()
        if not isinstance(projection, ProjectionConfig):
            raise TypeError("projection must be a ProjectionConfig")
        self.encoder_dim = projection.encoder_dim
        self.dynamic_dim = projection.dynamic_dim
        self.readout = projection.readout
        self.spectrum = spectrum
        self.dynamics = dynamics
        self.integrator = integrator
        self.p_in = nn.Linear(self.encoder_dim, self.dynamic_dim, bias=True,
                              device="cpu", dtype=torch.float32)
        self.p_out = nn.Linear(self.dynamic_dim, self.encoder_dim, bias=True,
                               device="cpu", dtype=torch.float32)
        # 仅初始化新投影层，避免递归重置已注入组件。
        for layer in (self.p_in, self.p_out):
            nn.init.xavier_uniform_(layer.weight, gain=1.)
            nn.init.zeros_(layer.bias)

    def _validate_components(self) -> None:
        """生产预测前核对真实谱、动力学与积分组件。"""
        for name, expected in (("spectrum", GraphSpectrum), ("dynamics", ModalDynamics),
                               ("integrator", IntegratorConfig)):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be a {expected.__name__} for real prediction")

    def forward(self, raw_z: Tensor, controls: ActionSegments,
                duration: Tensor, grid: GridSpec) -> Tensor:
        """图1、式5—9：已知起点与控制经完整 Phi 得到 raw 预测。

        raw_z/输出为同网格同序的模型 float32 [B,N,E]，N=H*W；grid 已知，
        controls 单位见 ControlSpec；duration 为无梯度 CPU float64 [B] 非负周期。
        依次投影、构图/求谱、分析、A/B/N 积分、合成、读出。仅谱分解停止
        梯度，X/C/g/动作/参数与预测反馈保留允许路径，服从外部 no_grad。
        零时长仍为 P_out(Psi Psi^T P_in(Z))；式15第二段只接 raw 输出完整重建。
        接口不接未来观察或标签，不保存逐步 trace。
        """
        return self._predict_raw_with_trace(raw_z, controls, duration, grid, collect_trace=False).raw_prediction

    def trace_forward(self, raw_z: Tensor, controls: ActionSegments,
                      duration: Tensor, grid: GridSpec) -> PredictionTrace:
        """与 forward 同次完整 Phi，额外返回 PredictionTrace，不重跑网络。

        输入/输出沿 forward：模型 float32 [B,N,E]，无梯度 CPU float64 [B]
        非负周期与已知控制/网格。K=min(max_modes,N)，仅特征值和基底停梯度。
        零时长无积分步但仍谱投影读出；完整重启只接 raw_prediction，使用后释放 trace。
        """
        return self._predict_raw_with_trace(raw_z, controls, duration, grid, collect_trace=True)

    def prepare_query(self, raw_z: Tensor, grid: GridSpec) -> QueryContext:
        """图1：raw Z [B,N,E]→X、最低 K 谱、C0 与完整起点动力学上下文。

        输入有限模型 float32，N=H*W，允许非连续视图及预测反馈梯度。
        P_in→W→L→min(max_modes,N) 特征对→Psi^T X 和 Pool/条件化；仅谱
        分支停梯度。原始 CPU float64 坐标在动力学入口显式转模型 float32。
        错组件/网格/形状/精度/设备报错，不修补或缓存跨查询图。
        """
        self._validate_components()
        if not isinstance(raw_z, Tensor) or not isinstance(grid, GridSpec):
            raise TypeError("raw_z must be a tensor and grid must be a GridSpec")
        if raw_z.is_nested or raw_z.layout != torch.strided or raw_z.is_meta:
            raise ValueError("raw_z must use a real dense model device")
        x = self.project_in(raw_z)
        graph = self.spectrum.config
        # affinity 内部只取 detach 副本；encoder 模式用编码器输出 Z 建图，不经过可训练 P_in。
        affinity = self.spectrum.affinity(raw_z if graph.affinity_source == "encoder" else x, grid)
        laplacian = self.spectrum.normalized_laplacian(affinity).laplacian
        if graph.fixed_modes and x.shape[1] < graph.max_modes:
            raise ValueError("fixed_modes requires every grid to have at least K tokens")
        modes = graph.max_modes if graph.fixed_modes else min(graph.max_modes, x.shape[1])
        spectrum = self.spectrum.lowest_eigenpairs(laplacian, k=modes, model_device=x.device)
        initial = modal_analysis(x, spectrum.basis)
        context = self.dynamics.prepare_context(x, spectrum.eigenvalues, spectrum.basis, grid.coords)
        return QueryContext(x, grid, spectrum, initial, context)

    def integrate_internal_continuation(
        self, query: QueryContext, current_coefficients: Tensor,
        controls: ActionSegments, duration: Tensor,
    ) -> IntegrationResult:
        """内部诊断：C [B,K,64] 在既有 query 上继续积分，不重建完整 Phi。

        C 为模型 float32；控制已知，duration 为无梯度 CPU float64 [B] 非负
        周期。保持 query/梯度，不投影、求谱、Pool 或读出；不得跨优化步，也
        不能代替式15从 raw 预测重新构建的短支。
        """
        self._validate_components()
        if not isinstance(query, QueryContext):
            raise TypeError("query must be a QueryContext")
        return self.dynamics.integrate_segments(current_coefficients, controls, query.dynamics,
            duration=duration, config=self.integrator)

    def _predict_raw_with_trace(
        self, raw_z: Tensor, controls: ActionSegments, duration: Tensor, grid: GridSpec,
        *, collect_trace: bool = False,
    ) -> PredictionTrace:
        """唯一内核：raw Z→新 query→完整积分→Psi 合成→P_out。

        Z/预测为模型 float32 [B,N,E]，控制已知，duration 为无梯度 CPU
        float64 [B] 非负周期。零时长仍谱投影读出，不补 LN/残差或强制恒等。
        完整第二段仅接 raw_prediction；collect_trace 只保留同次实际步骤。
        """
        query = self.prepare_query(raw_z, grid)
        integration = self.dynamics.integrate_segments(query.initial_coefficients, controls,
            query.dynamics, duration=duration, config=self.integrator, collect_trace=collect_trace)
        synthesized = modal_synthesis(integration.final, query.spectrum.basis)
        raw = self.project_out(synthesized)
        if self.readout == "residual":
            # 偏置在两项中精确相消；C_tau=C_0 时差为 0，输出逐位等于输入 Z。
            initial = modal_synthesis(query.initial_coefficients, query.spectrum.basis)
            raw = raw_z + (raw - self.project_out(initial))
        return PredictionTrace(raw, query, integration, synthesized)

    def project_in(self, raw_z: Tensor) -> Tensor:
        """图1：Z [B,N,E] float32→X [B,N,64]，逐 token 计算 Z W_in^T+b_in。

        同设备无量纲特征；保留当前或预测反馈 Z 的梯度，不 detach 或归一化。
        """
        return self._project(raw_z, self.p_in, "raw_z")

    def project_out(self, synthesized: Tensor) -> Tensor:
        """图1：X_hat [B,N,64] float32→raw Z_hat [B,N,E]，逐 token 线性读出。

        输入同设备并保留梯度；不做谱合成、LN 或残差，不强制与 P_in 互逆。
        """
        return self._project(synthesized, self.p_out, "synthesized")

    @staticmethod
    def _project(x: Tensor, layer: nn.Linear, name: str) -> Tensor:
        """两条入口共用边界核对；实际Linear保留全部允许的反向传播路径。"""
        if not isinstance(x, Tensor):
            raise TypeError(f"{name} must be a tensor")
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != layer.in_features:
            raise ValueError(f"{name} must have shape [positive B, positive N, {layer.in_features}]")
        if x.dtype != torch.float32 or layer.weight.dtype != torch.float32 or layer.bias.dtype != torch.float32:
            raise ValueError(f"{name} and projection parameters must be float32")
        if x.device != layer.weight.device or x.device != layer.bias.device:
            raise ValueError(f"{name} and projection parameters must share a device")
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} must contain finite values")
        result = layer(x)
        if result.dtype != torch.float32 or not torch.isfinite(result).all():
            raise ValueError("projection output must be finite float32")
        return result
