"""式8—11：共享条件动力学与分段 exponential-Euler 积分。

起点 g=mean_N(X)，[lambda,g] 经共享 65→64 SiLU 干路生成 A/B。
完整右端为 Ac+Ba+N；N 将内容/谱/空间分数相加后稀疏聚合，再加 M_a。
每次查询固定起点上下文，每步重算当前 C/动作交互，调度不跨控制边界。
网络宽度与初始化是固定实现选择；非线性完整流不继承自由块稳定性保证。
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal

import torch
from torch import Tensor, nn

from .actions import (ActionSegments, action_at, clip_action_segments,
                      next_step_schedule, validate_action_segments)


@dataclass(frozen=True)
class DynamicsConfig:
    """冻结的动力学结构与初始化配置。

    action_dim=m，单位由控制协议声明；dynamic_dim/condition_width=64，
    query_dim/两偏置宽=32，action_hidden_width=64，top_r=16。
    initial_damping=.1/控制周期，head_weight_std=1e-3；其余宽度/特征无量纲。
    """

    action_dim: int
    dynamic_dim: int
    condition_width: int
    query_dim: int
    top_r: int
    spectral_bias_width: int
    spatial_bias_width: int
    action_hidden_width: int
    initial_damping: float
    head_weight_std: float

    def __post_init__(self) -> None:
        """分配参数前验证严格整数/正有限实数，拒绝bool与未登记宽度。"""
        for name in ("action_dim", "top_r"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, width in (("dynamic_dim", 64), ("condition_width", 64), ("query_dim", 32),
                            ("spectral_bias_width", 32), ("spatial_bias_width", 32), ("action_hidden_width", 64)):
            value = getattr(self, name)
            if type(value) is not int or value != width:
                raise ValueError(f"{name} must be the fixed first-round width {width}")
        for name in ("initial_damping", "head_weight_std"):
            value = getattr(self, name)
            try:
                valid = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value) and value > 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be a finite positive real number")


@dataclass(frozen=True)
class IntegratorConfig:
    """固定 exponential_euler 配置，不保存计算图。

    phi1_threshold/order=1e-4/6，max_step=.25 内部周期，endpoint_rtol=1e-12
    无量纲。单步 delta 由调用者显式给定，分段调度负责 max_step 截断。
    """

    method: Literal["exponential_euler"]
    max_step: float
    endpoint_rtol: float
    phi1_threshold: float
    phi1_order: int

    def __post_init__(self) -> None:
        """拒绝bool、非有限时间参数及未登记的phi1阈值/阶数，不静默修补。"""
        if not isinstance(self.method, str) or self.method != "exponential_euler":
            raise ValueError("method must be exponential_euler")
        for name in ("max_step", "endpoint_rtol", "phi1_threshold"):
            value = getattr(self, name)
            try:
                valid = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value) and value > 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be a finite positive real number")
        if self.phi1_threshold != 1e-4:
            raise ValueError("phi1_threshold must be the fixed first-round value 1e-4")
        if type(self.phi1_order) is not int or self.phi1_order != 6:
            raise ValueError("phi1_order must be the fixed first-round integer 6")


def _validate_model_tensor(value: Tensor, layer: nn.Linear, name: str) -> None:
    """只核对公共张量边界；不搬设备、不转精度、不改输入或允许的梯度。"""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.is_nested or value.layout != torch.strided or value.is_meta or layer.weight.is_meta or layer.bias.is_meta:
        raise ValueError(f"{name} and condition parameters must use a real dense device, not meta")
    if value.dtype != torch.float32 or layer.weight.dtype != torch.float32 or layer.bias.dtype != torch.float32:
        raise ValueError(f"{name} and condition parameters must be float32")
    if value.device != layer.weight.device or value.device != layer.bias.device:
        raise ValueError(f"{name} and condition parameters must share a device")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")


@dataclass(frozen=True)
class _FreeCoefficients:
    """自由块系数：raw_gamma/gamma/omega [B,K,32]，A [B,K,32,2,2]。

    同模型有限 float32；gamma/omega/A 单位为内部周期^-1。通道对
    (2r,2r+1) 属同一模态。保留 g/干路/双头梯度，lambda 为无梯度常量。
    """

    raw_gamma: Tensor
    gamma: Tensor
    omega: Tensor
    a_blocks: Tensor


@dataclass(frozen=True)
class ConditionedCoefficients:
    """同次共享 H 产生 A/B，冻结字段不冻结张量梯度。

    raw_gamma/gamma/omega [B,K,32]，A [B,K,32,2,2]，B [B,K,64,m]，
    均为模型有限 float32。A 单位周期^-1，B 列单位为系数/周期/相应动作单位。
    保留 g/H/三头梯度，lambda 无梯度，不跨查询缓存。
    """

    raw_gamma: Tensor
    gamma: Tensor
    omega: Tensor
    a_blocks: Tensor
    b_matrix: Tensor


@dataclass(frozen=True)
class ContentTerms:
    """当前内容项：q/key [B,K,32]、value [B,K,64]、scores [B,K,K]。

    同设备有限 float32；q/key/scores 无量纲，value 单位系数/周期。
    保留当前 C 和投影权重梯度，未作稀疏聚合。
    """

    query: Tensor
    key: Tensor
    value: Tensor
    scores: Tensor


@dataclass(frozen=True)
class SpatialDescriptors:
    """固定空间描述：moments [B,K,5]、overlap [B,K,K]、pair_features [B,K,K,16]。

    同模型 float32、无量纲、无梯度。五矩为 Psi² 加权 [x,y,x²,y²,xy]；
    重叠为绝对值乘积之和，不是退化为单位阵的正交内积。张量只读。
    """

    moments: Tensor
    overlap: Tensor
    pair_features: Tensor


@dataclass(frozen=True)
class SparseAttentionResult:
    """稀疏索引/分数/权重 [B,K,r_eff] 与 aggregate [B,K,64]。

    索引为同设备无梯度 int64；其余为有限 float32。分数/权重无量纲，
    aggregate 单位系数/周期。索引离散，仅选中分数和值保留梯度。
    """

    neighbor_index: Tensor
    selected_scores: Tensor
    weights: Tensor
    aggregate: Tensor


@dataclass(frozen=True)
class DynamicsContext:
    """一次完整查询固定的起点上下文，不跨优化步缓存。

    lambda [B,K]、Psi [B,N,K] 与 coords [N,2] 为模型 float32 无梯度常量；
    coords 由 CPU float64 网格在 prepare_context 唯一转换。pooled=g [B,64]，
    同次 A/B 与谱/空间偏置 [B,K,K] 保留图。新起点/网格/优化步须重建；张量只读。
    """

    eigenvalues: Tensor
    basis: Tensor
    coords: Tensor
    pooled: Tensor
    coefficients: ConditionedCoefficients
    spectral_bias: Tensor
    spatial_bias: Tensor


@dataclass(frozen=True)
class NonlinearResult:
    """同次完整交互诊断：content、三项总分、稀疏注意力与动作残差。

    scores 为无量纲 float32 [B,K,K]；action_residual/nonlinear 为
    [B,K,64] 系数/周期，nonlinear=aggregate+M_a(C,a)。保留 C/a/参数梯度。
    """

    content: ContentTerms
    spectral_scores: Tensor
    spatial_scores: Tensor
    total_scores: Tensor
    attention: SparseAttentionResult
    action_residual: Tensor
    nonlinear: Tensor


@dataclass(frozen=True)
class RHSResult:
    """式8：free=Ac、action_linear=Ba、interaction=N，derivative 为三项和。

    各项同模型有限 float32 [B,K,64]，单位系数/周期，保留同次计算图。
    只用当前 C、已知动作及固定起点上下文；不乘时长或加 C。
    """

    free: Tensor
    action_linear: Tensor
    interaction: NonlinearResult
    derivative: Tensor


@dataclass(frozen=True)
class IntegrationCoefficients:
    """E=exp(A delta)、H=积分 exp(A s)ds，形状 [B,K,32,2,2]。

    E 无量纲，H 单位周期；同率 dtype/device，模型 float32，CPU float64
    可作解析诊断。由正 delta 生成，保留 gamma/omega 梯度。
    """

    exponential: Tensor
    forcing_integral: Tensor


@dataclass(frozen=True)
class StepResult:
    """更新后 C [B,K,64]、同次左端 rhs 与 E/H，全部保留计算图。

    C 为模型 float32 无量纲状态；rhs 单位系数/周期，不额外重跑诊断或缓存。
    """

    coefficients: Tensor
    rhs: RHSResult
    propagator: IntegrationCoefficients


@dataclass(frozen=True)
class IntegrationStep:
    """一次实际积分步的快照，仅 collect_trace 时保留。

    batch_index/segment_index 为 CPU int64 [B_active]，start_time/delta
    为 CPU float64 同形状周期；before 为模型 float32 [B_active,K,64]，
    result 为同次 StepResult。保留图并按实际顺序记录，使用后释放。
    """

    batch_index: Tensor
    start_time: Tensor
    delta: Tensor
    segment_index: Tensor
    before: Tensor
    result: StepResult


@dataclass(frozen=True)
class IntegrationResult:
    """分段终点 final [B,K,64] 保留图，n_steps 为 CPU int64 [B]。

    计数仅含真实正步；零查询保持 initial。steps 仅 collect_trace=True
    时含实际快照，不代表完整 raw Phi 的零时长恒等。
    """

    final: Tensor
    n_steps: Tensor
    steps: tuple[IntegrationStep, ...]


class ModalDynamics(nn.Module):
    """跨样本、网格和模态数共享 A/B、内容、谱/空间偏置与动作残差网络。

    g=pool_context(X)，H 为 [lambda,g] 条件特征。完整 Phi 重建起点量；
    同次积分保持 g/A/B/偏置且不 detach，动作切换不重置 C 或重算 g。
    计算图不存成员；新起点、网格或优化步必须重新准备上下文。
    """

    def __init__(self, config: DynamicsConfig) -> None:
        """按固定注册顺序初始化，保留随机数消耗顺序。

        显式 CPU float32；隐藏层 Xavier uniform(gain=1)、偏置 0，输出头
        Normal(0,head_weight_std)。raw_gamma 偏置以 d+log(-expm1(-d)) 实现
        稳定逆 softplus，omega 偏置 0；非零头权重允许率随条件变化。
        B 按 c*m+d 排列；q/key 宽32、value 宽64，谱 MLP 3→32→1、
        空间 MLP 16→32→1、M_a (64+m)→64→64，隐藏 SiLU。
        M_a 无输出激活或零动作门控；有限正 float32 无法表示的初始化量直接拒绝。
        仅消费调用者 torch RNG，不重设种子、默认精度/设备或既有参数。
        """
        super().__init__()
        if not isinstance(config, DynamicsConfig):
            raise TypeError("config must be a DynamicsConfig")
        self.config = config
        self.condition_trunk = nn.Sequential(
            nn.Linear(config.dynamic_dim+1, config.condition_width, bias=True,
                      device="cpu", dtype=torch.float32),
            nn.SiLU(),
        )
        nn.init.xavier_uniform_(self.condition_trunk[0].weight, gain=1.)
        nn.init.zeros_(self.condition_trunk[0].bias)
        for name in ("initial_damping", "head_weight_std"):
            value = torch.tensor(float(getattr(config, name)), device="cpu", dtype=torch.float32)
            if not torch.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be representable as finite positive float32 for initialization")
        self.gamma_head = nn.Linear(64, 32, bias=True, device="cpu", dtype=torch.float32)
        self.omega_head = nn.Linear(64, 32, bias=True, device="cpu", dtype=torch.float32)
        nn.init.normal_(self.gamma_head.weight, mean=0., std=config.head_weight_std)
        nn.init.normal_(self.omega_head.weight, mean=0., std=config.head_weight_std)
        damping = float(config.initial_damping)
        nn.init.constant_(self.gamma_head.bias, damping+math.log(-math.expm1(-damping)))
        nn.init.zeros_(self.omega_head.bias)
        self.b_head = nn.Linear(64, 64*config.action_dim, bias=True, device="cpu", dtype=torch.float32)
        nn.init.normal_(self.b_head.weight, mean=0., std=config.head_weight_std)
        nn.init.zeros_(self.b_head.bias)
        self.q_proj = nn.Linear(64, 32, bias=False, device="cpu", dtype=torch.float32)
        self.k_proj = nn.Linear(64, 32, bias=False, device="cpu", dtype=torch.float32)
        self.v_proj = nn.Linear(64, 64, bias=False, device="cpu", dtype=torch.float32)
        for layer in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(layer.weight, gain=1.)
        self.spectral_bias_net = nn.Sequential(
            nn.Linear(3, config.spectral_bias_width, bias=True, device="cpu", dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(config.spectral_bias_width, 1, bias=True, device="cpu", dtype=torch.float32),
        )
        nn.init.xavier_uniform_(self.spectral_bias_net[0].weight, gain=1.)
        nn.init.zeros_(self.spectral_bias_net[0].bias)
        nn.init.normal_(self.spectral_bias_net[2].weight, mean=0., std=config.head_weight_std)
        nn.init.zeros_(self.spectral_bias_net[2].bias)
        self.spatial_bias_net = nn.Sequential(
            nn.Linear(16, config.spatial_bias_width, bias=True, device="cpu", dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(config.spatial_bias_width, 1, bias=True, device="cpu", dtype=torch.float32),
        )
        nn.init.xavier_uniform_(self.spatial_bias_net[0].weight, gain=1.)
        nn.init.zeros_(self.spatial_bias_net[0].bias)
        nn.init.normal_(self.spatial_bias_net[2].weight, mean=0., std=config.head_weight_std)
        nn.init.zeros_(self.spatial_bias_net[2].bias)
        self.action_residual_net = nn.Sequential(
            nn.Linear(64+config.action_dim, config.action_hidden_width, bias=True, device="cpu", dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(config.action_hidden_width, 64, bias=True, device="cpu", dtype=torch.float32),
        )
        nn.init.xavier_uniform_(self.action_residual_net[0].weight, gain=1.)
        nn.init.zeros_(self.action_residual_net[0].bias)
        nn.init.normal_(self.action_residual_net[2].weight, mean=0., std=config.head_weight_std)
        nn.init.zeros_(self.action_residual_net[2].bias)
        if not all(torch.isfinite(parameter).all() for parameter in self.parameters()):
            raise ValueError("initialized condition parameters must be finite float32")

    def _validate_context(self, context: DynamicsContext) -> None:
        """核固定上下文及嵌套 A/B 合同，不重算网络或强迫输出带梯度。"""
        if not isinstance(context, DynamicsContext):
            raise TypeError("context must be DynamicsContext")
        if not isinstance(context.coefficients, ConditionedCoefficients):
            raise TypeError("context coefficients must be ConditionedCoefficients")
        for name in ("eigenvalues", "basis", "coords", "pooled", "spectral_bias", "spatial_bias"):
            _validate_model_tensor(getattr(context, name), self.condition_trunk[0], name)
        for name in ("eigenvalues", "basis", "coords"):
            if getattr(context, name).requires_grad:
                raise ValueError(f"{name} must be constant without gradients")
        eigenvalues, basis = context.eigenvalues, context.basis
        if eigenvalues.ndim != 2 or eigenvalues.shape[0] < 1 or eigenvalues.shape[1] < 1:
            raise ValueError("context eigenvalues must have shape [positive B, positive K]")
        batch, modes = eigenvalues.shape
        if basis.ndim != 3 or basis.shape[0] != batch or basis.shape[2] != modes or basis.shape[1] < modes:
            raise ValueError("context basis must have matching shape [B, N >= K, K]")
        if context.coords.shape != (basis.shape[1], 2) or (context.coords < 0).any() or (context.coords > 1).any():
            raise ValueError("context coords must have matching shape [N,2] and values in [0,1]")
        if context.pooled.shape != (batch, 64):
            raise ValueError("context pooled must have matching shape [B,64]")
        if context.spectral_bias.shape != (batch, modes, modes) or context.spatial_bias.shape != (batch, modes, modes):
            raise ValueError("context biases must have matching shape [B,K,K]")
        shapes = (("raw_gamma", (batch, modes, 32)), ("gamma", (batch, modes, 32)),
                  ("omega", (batch, modes, 32)), ("a_blocks", (batch, modes, 32, 2, 2)),
                  ("b_matrix", (batch, modes, 64, self.config.action_dim)))
        for name, shape in shapes:
            value = getattr(context.coefficients, name)
            _validate_model_tensor(value, self.condition_trunk[0], f"context {name}")
            if value.shape != shape:
                raise ValueError(f"context {name} must have matching shape {shape}")

    def prepare_context(
        self, x: Tensor, eigenvalues: Tensor, basis: Tensor, coords: Tensor,
    ) -> DynamicsContext:
        """起点 X/谱/网格→本次固定 g、A/B 与两项偏置。

        X 为模型 float32 [B,N,64] 并保留梯度；lambda [B,K]/Psi [B,N,K]
        为同设备无梯度 float32 常量，1<=K<=N。coords 为 CPU float64 [N,2]
        有限无梯度 [0,1] 坐标，在此唯一转模型 float32。依次均值 Pool→A/B→
        谱偏置→空间描述/偏置。保留 g/A/B/偏置图，不重求谱或缓存跨查询量。
        """
        for name, value in (("x", x), ("eigenvalues", eigenvalues), ("basis", basis)):
            _validate_model_tensor(value, self.condition_trunk[0], name)
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != 64:
            raise ValueError("x must have shape [positive B, positive N, 64]")
        if eigenvalues.ndim != 2 or eigenvalues.shape[0] != x.shape[0] or not 1 <= eigenvalues.shape[1] <= x.shape[1]:
            raise ValueError("eigenvalues must have matching shape [B, 1 <= K <= N]")
        if basis.shape != (x.shape[0], x.shape[1], eigenvalues.shape[1]):
            raise ValueError("basis must have matching shape [B,N,K]")
        if eigenvalues.requires_grad or basis.requires_grad:
            raise ValueError("eigenvalues and basis must be constants without gradients from C04")
        if not isinstance(coords, Tensor):
            raise TypeError("coords must be a tensor")
        if coords.is_nested or coords.layout != torch.strided or coords.device.type != "cpu" or coords.dtype != torch.float64:
            raise ValueError("coords must use the original dense CPU float64 grid protocol")
        if coords.shape != (x.shape[1], 2) or coords.requires_grad or not torch.isfinite(coords).all() or (coords < 0).any() or (coords > 1).any():
            raise ValueError("coords must have shape [N,2], be finite in [0,1] and have no gradients")
        coords_model = coords.to(device=x.device, dtype=torch.float32)
        pooled = self.pool_context(x)
        conditioned = self.condition(eigenvalues, pooled)
        spectral = self.spectral_bias(eigenvalues)
        descriptors = self.spatial_descriptors(basis, coords_model)
        spatial = self.spatial_bias(descriptors)
        context = DynamicsContext(eigenvalues, basis, coords_model, pooled, conditioned, spectral, spatial)
        self._validate_context(context)
        return context

    def free_rhs(self, coefficients: Tensor, a_blocks: Tensor) -> Tensor:
        """式8/9：C [B,K,64] 与 A [B,K,32,2,2]→Ac [B,K,64]。

        同设备有限 float32，允许非连续输入；A 单位周期^-1，输出系数/周期。
        按 (0,1)…(62,63) 通道对左乘 A，不混模态、转置 A、乘时长或加 C，
        保留 C/A 梯度。
        """
        for name, value in (("coefficients", coefficients), ("a_blocks", a_blocks)):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.is_nested or value.layout != torch.strided or value.is_meta:
                raise ValueError(f"{name} must use a real dense device, not meta")
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite float32 values")
        if coefficients.ndim != 3 or coefficients.shape[0] < 1 or coefficients.shape[1] < 1 or coefficients.shape[2] != 64:
            raise ValueError("coefficients must have shape [positive B, positive K, 64]")
        batch, modes = coefficients.shape[:2]
        if a_blocks.shape != (batch, modes, 32, 2, 2) or a_blocks.device != coefficients.device:
            raise ValueError("a_blocks must have matching shape [B,K,32,2,2] and share the coefficients device")
        result = torch.einsum("bkrpq,bkrq->bkrp", a_blocks, coefficients.reshape(batch, modes, 32, 2)).reshape(batch, modes, 64)
        if result.dtype != torch.float32 or not torch.isfinite(result).all():
            raise ValueError("free contribution must be finite float32")
        return result

    def nonlinear(self, coefficients: Tensor, action: Tensor, context: DynamicsContext) -> NonlinearResult:
        """附录交互：当前 C/a 重算 N=总分稀疏聚合+M_a(C,a)。

        C [B,K,64]、a [B,m] 与 context 同模型 float32/device、严格匹配 B/K。
        重算 q/k/v 和内容分数，加固定谱/空间偏置后 top-r/softmax 聚合，
        再加当前动作残差。保持 C/a/参数梯度；不重算起点上下文或重置状态。
        输出 [B,K,64] 系数/周期，不含 Ac/Ba/积分；非有限求和报错。
        """
        self._validate_context(context)
        content = self.content_terms(coefficients)
        if coefficients.shape[:2] != context.eigenvalues.shape:
            raise ValueError("coefficients must match context batch and modes")
        total = content.scores+context.spectral_bias+context.spatial_bias
        attention = self.sparse_attention(total, content.value)
        residual = self.action_residual(coefficients, action)
        nonlinear = attention.aggregate+residual
        if nonlinear.dtype != torch.float32 or not torch.isfinite(nonlinear).all():
            raise ValueError("nonlinear contribution must be finite float32")
        return NonlinearResult(content, context.spectral_bias, context.spatial_bias, total, attention, residual, nonlinear)

    def rhs(self, coefficients: Tensor, action: Tensor, context: DynamicsContext) -> RHSResult:
        """式8：同次 C/a 返回 Ac、Ba、N 与 dC/dt=Ac+Ba+N。

        C [B,K,64]、a [B,m] 沿同模型有限 float32 合同，允许梯度和非连续输入。
        A/B 起点固定，交互当前重算；不乘时长、更新 C 或重建上下文。
        分支相加溢出直接报错，不裁剪。
        """
        self._validate_context(context)
        free = self.free_rhs(coefficients, context.coefficients.a_blocks)
        linear = self.action_rhs(action, context.coefficients.b_matrix)
        interaction = self.nonlinear(coefficients, action, context)
        derivative = free+linear+interaction.nonlinear
        if derivative.dtype != torch.float32 or not torch.isfinite(derivative).all():
            raise ValueError("complete derivative must be finite float32")
        return RHSResult(free, linear, interaction, derivative)

    def integration_coefficients(
        self, *, gamma: Tensor, omega: Tensor, delta: Tensor, config: IntegratorConfig,
    ) -> IntegrationCoefficients:
        """固定 A=-gamma I+omega J 的 E/H，不求 A 逆或加 epsilon。

        gamma/omega 为同设备 [B,K,32] 有限 float32，或均 CPU float64；
        gamma>=0、率单位周期^-1。delta 为无梯度 CPU float64 [B] 正周期。
        率转 double，z=(-gamma+i omega)delta 用 complex128；|z|<1e-4 时
        phi1 用 0…6 阶 Horner，否则 expm1(z)/z。未选分支先换安全参数，
        避免零点/溢出污染反向；E 用阻尼旋转闭式，H=delta*phi1。
        结果转回率 dtype 并保留率梯度；合法指数下溢为 0，非有限值报错。
        """
        if not isinstance(config, IntegratorConfig):
            raise TypeError("config must be an IntegratorConfig")
        for name, value in (("gamma", gamma), ("omega", omega), ("delta", delta)):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.is_nested or value.layout != torch.strided or value.is_meta:
                raise ValueError(f"{name} must use a real dense device, not meta")
            if value.dtype not in (torch.float32, torch.float64) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite real float32 or float64 values")
        if gamma.ndim != 3 or gamma.shape[0] < 1 or gamma.shape[1] < 1 or gamma.shape[2] != 32 or omega.shape != gamma.shape:
            raise ValueError("gamma and omega must have matching shape [positive B, positive K, 32]")
        if gamma.dtype != omega.dtype or gamma.device != omega.device:
            raise ValueError("gamma and omega must share dtype and device")
        if gamma.dtype == torch.float64 and gamma.device.type != "cpu":
            raise ValueError("float64 analytic integration coefficients are supported only on CPU")
        if delta.device.type != "cpu" or delta.dtype != torch.float64 or delta.shape != gamma.shape[:1]:
            raise ValueError("delta must be CPU float64 with shape [B]")
        if delta.requires_grad or (delta <= 0).any() or (gamma < 0).any():
            raise ValueError("delta must be positive without gradients and gamma must be nonnegative")
        time = delta.to(device=gamma.device).reshape(-1, 1, 1)
        damping = gamma.to(dtype=torch.float64)*time
        angle = omega.to(dtype=torch.float64)*time
        if not torch.isfinite(damping).all() or not torch.isfinite(angle).all():
            raise ValueError("float64 damping and angle products must be finite")
        z = torch.complex(-damping, angle)
        small = z.abs() < config.phi1_threshold
        # 未选分支也保持定义良好，where本身不能阻止其NaN进入反向。
        ratio_arg = torch.where(small, torch.ones_like(z), z)
        series_arg = torch.where(small, z, torch.zeros_like(z))
        ratio = torch.expm1(ratio_arg)/ratio_arg
        series = torch.full_like(z, 1/math.factorial(config.phi1_order+1))
        for order in range(config.phi1_order-1, -1, -1):
            series = 1/math.factorial(order+1)+series_arg*series
        h = time*torch.where(small, series, ratio)
        decay = torch.exp(-damping)
        real, imaginary = decay*torch.cos(angle), decay*torch.sin(angle)
        exponential = torch.stack((torch.stack((real, -imaginary), dim=-1),
                                   torch.stack((imaginary, real), dim=-1)), dim=-2).to(dtype=gamma.dtype)
        forcing_integral = torch.stack((torch.stack((h.real, -h.imag), dim=-1),
                                        torch.stack((h.imag, h.real), dim=-1)), dim=-2).to(dtype=gamma.dtype)
        if any(not torch.isfinite(value).all() for value in (ratio, series, h, exponential, forcing_integral)):
            raise ValueError("integration coefficients must be finite at working and output precision")
        return IntegrationCoefficients(exponential, forcing_integral)

    def exponential_euler_step(
        self, coefficients: Tensor, action: Tensor, context: DynamicsContext, *,
        delta: Tensor, config: IntegratorConfig,
    ) -> StepResult:
        """C_next=E C+H(Ba+N(C,a))，仅求一次左端 rhs。

        C [B,K,64]、a [B,m] 为模型 float32，context 固定；delta 为
        无梯度 CPU float64 [B] 正周期。64 通道还原为32对，E/H 分别左乘后还原。
        forcing 不含 Ac，避免自由项重复积分；当前 N 每次重算，保留全部梯度。
        不自动截步、查控制区间或更新时钟；这是状态依赖 N 的一阶近似。
        """
        result = self.rhs(coefficients, action, context)
        propagator = self.integration_coefficients(gamma=context.coefficients.gamma,
            omega=context.coefficients.omega, delta=delta, config=config)
        forcing = result.action_linear+result.interaction.nonlinear
        if not torch.isfinite(forcing).all():
            raise ValueError("step forcing must be finite float32")
        batch, modes = coefficients.shape[:2]
        paired = coefficients.reshape(batch, modes, 32, 2)
        paired_forcing = forcing.reshape(batch, modes, 32, 2)
        updated = (torch.einsum("bkrpq,bkrq->bkrp", propagator.exponential, paired)
                   +torch.einsum("bkrpq,bkrq->bkrp", propagator.forcing_integral, paired_forcing)).reshape(batch, modes, 64)
        if updated.dtype != torch.float32 or not torch.isfinite(updated).all():
            raise ValueError("updated coefficients must be finite float32")
        return StepResult(updated, result, propagator)

    def integrate_segments(
        self, initial: Tensor, controls: ActionSegments, context: DynamicsContext, *,
        duration: Tensor, config: IntegratorConfig, collect_trace: bool = False,
    ) -> IntegrationResult:
        """按已知右连续控制及独立查询时长推进 C，不重启完整 Phi。

        initial 为模型有限 float32 [B,K,64]；有效控制同设备 [B,J,m]，padding
        不读。duration 为无梯度 CPU float64 [B] 非负周期，仅用于调度。
        先精确核路径覆盖/原端点，再截取、调度与取当前动作；每轮仅 active
        子 batch 进单步，固定 lambda/Psi/g/A/B/偏置，重算当前 N，不跨动作边界。
        非原地恢复 batch；零查询不取动作，正短区间必须推进。返回 final、
        逐样本步数与可选同次 trace；保留图，trace 使用后释放。
        """
        if not isinstance(config, IntegratorConfig):
            raise TypeError("config must be an IntegratorConfig")
        if config.endpoint_rtol != 1e-12:
            raise ValueError("D11 supports only endpoint_rtol=1e-12")
        if type(collect_trace) is not bool:
            raise TypeError("collect_trace must be bool")
        _validate_model_tensor(initial, self.condition_trunk[0], "initial")
        if initial.ndim != 3 or min(initial.shape[:2]) < 1 or initial.shape[2] != 64:
            raise ValueError("initial must have shape [positive B, positive K,64]")
        self._validate_context(context)
        if initial.shape[:2] != context.eigenvalues.shape:
            raise ValueError("initial must match context batch and modes")
        if not isinstance(controls, ActionSegments):
            raise TypeError("controls must be ActionSegments")
        batch = initial.shape[0]
        for name, value in (("duration", duration), ("control duration", controls.duration)):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a tensor")
            if (value.is_nested or value.layout != torch.strided or value.is_meta
                    or value.device.type != "cpu" or value.dtype != torch.float64
                    or value.shape != (batch,) or value.requires_grad):
                raise ValueError(f"{name} must be CPU float64 [B] without gradients")
            if not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"{name} must be finite and nonnegative")
        # 端点容差不能补足控制路径实际未覆盖的正尾段。
        if (duration > controls.duration).any():
            raise ValueError("control path does not strictly cover requested duration")
        if not isinstance(controls.u_values, Tensor):
            raise TypeError("u_values must be a tensor")
        if (controls.u_values.is_nested or controls.u_values.layout != torch.strided
                or controls.u_values.is_meta or controls.u_values.device != initial.device):
            raise ValueError("control values must use the model's real dense device")
        validate_action_segments(controls, query_duration=duration)
        if controls.control_spec.action_dim != self.config.action_dim:
            raise ValueError("control action dimension must match dynamics")
        for b, count in enumerate(controls.n_segments):
            n = int(count)
            if n and (controls.starts[b, 0] != 0
                      or not torch.equal(controls.starts[b, 1:n], controls.ends[b, :n-1])
                      or controls.ends[b, n-1] != controls.duration[b]):
                raise ValueError("effective raw segments must strictly cover [0,duration) without gap or overlap")
        time = torch.zeros_like(duration)
        clipped = clip_action_segments(controls, start=time, end=duration)
        if not torch.equal(clipped.duration, duration):
            raise ValueError("clipped duration must equal requested duration exactly")
        current = initial
        n_steps = torch.zeros((batch,), dtype=torch.int64, device="cpu")
        steps = []
        max_step = float(config.max_step)
        while (time < duration).any():
            schedule = next_step_schedule(clipped, time=time, query_duration=duration, max_step=max_step)
            if not torch.equal(schedule.time, time) or not torch.equal(schedule.active, time < duration):
                raise ValueError("schedule must preserve the actual clock and remaining positive intervals")
            indices = torch.nonzero(schedule.active).flatten()
            model_indices = indices.to(device=initial.device)
            start = time.index_select(0, indices)
            delta = schedule.delta.index_select(0, indices)
            segment = schedule.segment_index.index_select(0, indices)
            next_time = start+delta
            boundary = torch.minimum(clipped.ends[indices, segment], duration.index_select(0, indices))
            tolerance = config.endpoint_rtol*torch.maximum(torch.ones_like(boundary),
                torch.maximum(duration.index_select(0, indices).abs(), boundary.abs()))
            if ((delta <= 0).any() or (next_time <= start).any()
                    or (next_time > boundary+tolerance).any()):
                raise ValueError("positive integration step must advance without crossing a boundary")
            # 只向下修正已经发生的舍入越界，不向前跳过边界前的正剩余区间。
            next_time = torch.minimum(next_time, boundary)
            if (next_time <= start).any():
                raise ValueError("positive integration interval cannot be represented")
            action = action_at(clipped, time=schedule.time, active=schedule.active).index_select(0, model_indices)
            fixed = context.coefficients
            active_context = DynamicsContext(
                context.eigenvalues.index_select(0, model_indices), context.basis.index_select(0, model_indices),
                context.coords, context.pooled.index_select(0, model_indices),
                ConditionedCoefficients(*(getattr(fixed, name).index_select(0, model_indices) for name in
                    ("raw_gamma", "gamma", "omega", "a_blocks", "b_matrix"))),
                context.spectral_bias.index_select(0, model_indices), context.spatial_bias.index_select(0, model_indices),
            )
            before = current.index_select(0, model_indices)
            result = self.exponential_euler_step(before, action, active_context, delta=delta, config=config)
            current = current.index_copy(0, model_indices, result.coefficients)
            time = time.index_copy(0, indices, next_time)
            n_steps = n_steps+schedule.active.to(dtype=torch.int64)
            if collect_trace:
                steps.append(IntegrationStep(indices, start, delta, segment, before, result))
        if not torch.equal(time, duration):
            raise ValueError("actual integration clock must reach the original requested duration")
        return IntegrationResult(current, n_steps, tuple(steps))

    def action_residual(self, coefficients: Tensor, action: Tensor) -> Tensor:
        """M_a：当前 C [B,K,64] 与已知 a [B,m]→[B,K,64] 系数/周期。

        同模型有限 float32，动作单位见控制协议，仅沿 K 共享，不广播 batch。
        [C,a] 经 Linear→SiLU→Linear，保留 C/a/四参数梯度；动作梯度沿 K 累加。
        无输出激活或零动作门控，M_a(C,0) 可非零；不乘时长、加 C/Ba/注意力。
        """
        for layer in (self.action_residual_net[0], self.action_residual_net[2]):
            _validate_model_tensor(coefficients, layer, "coefficients")
            _validate_model_tensor(action, layer, "action")
            for parameter in (layer.weight, layer.bias):
                if parameter.is_nested or parameter.layout != torch.strided or not torch.isfinite(parameter).all():
                    raise ValueError("action residual parameters must be dense finite float32")
        if coefficients.ndim != 3 or coefficients.shape[0] < 1 or coefficients.shape[1] < 1 or coefficients.shape[2] != 64:
            raise ValueError("coefficients must have shape [positive B, positive K, 64]")
        if action.shape != (coefficients.shape[0], self.config.action_dim):
            raise ValueError("action must have shape [matching B, action_dim]")
        features = torch.cat((coefficients, action.unsqueeze(1).expand(-1, coefficients.shape[1], -1)), dim=-1)
        result = self.action_residual_net(features)
        if result.dtype != torch.float32 or not torch.isfinite(result).all():
            raise ValueError("action residual must be finite float32")
        return result

    def sparse_attention(self, total_scores: Tensor, value: Tensor) -> SparseAttentionResult:
        """逐行稳定 top-r→选中槽 softmax→来源 value 加权和。

        scores [B,K,K] 无量纲，value [B,K,64] 为系数/周期，同设备有限
        float32，允许非连续输入。r_eff=min(top_r,K)，稳定降序；完全并列
        时小来源列 j 优先，自身参与候选但不强制。仅从原分数 gather，不再
        除 sqrt(d_q)；Y_k=sum_l alpha_kl V_j，不建立稠密权重。
        索引离散，未选分数梯度为0，r_eff=1 的分数零梯度合法；共享值梯度累加。
        """
        for name, tensor in (("total_scores", total_scores), ("value", value)):
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a tensor")
            if tensor.is_nested or tensor.layout != torch.strided or tensor.is_meta:
                raise ValueError(f"{name} must use a real dense device, not meta")
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain finite float32 values")
        if total_scores.ndim != 3 or total_scores.shape[0] < 1 or total_scores.shape[1] < 1 or total_scores.shape[2] != total_scores.shape[1]:
            raise ValueError("total_scores must have shape [positive B, positive K, K]")
        batch, modes, _ = total_scores.shape
        if value.shape != (batch, modes, 64):
            raise ValueError("value must have shape [matching B, matching K, 64]")
        if value.device != total_scores.device:
            raise ValueError("total_scores and value must share a device")
        neighbor_index = torch.argsort(total_scores, dim=-1, descending=True, stable=True)[..., :min(self.config.top_r, modes)]
        selected_scores = total_scores.gather(-1, neighbor_index)
        weights = torch.softmax(selected_scores, dim=-1)
        batch_index = torch.arange(batch, device=value.device, dtype=torch.int64)[:, None, None]
        selected_value = value[batch_index, neighbor_index]
        aggregate = (weights.unsqueeze(-1)*selected_value).sum(dim=-2)
        if any(tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()
               for tensor in (selected_scores, weights, aggregate)):
            raise ValueError("sparse attention outputs must be finite float32")
        return SparseAttentionResult(neighbor_index, selected_scores, weights, aggregate)

    def spatial_descriptors(self, basis: Tensor, coords: Tensor) -> SpatialDescriptors:
        """Psi [B,N,K] 与 coords [N,2]→五矩、绝对重叠和有序16维特征。

        同模型有限 float32 无梯度常量，1<=K<=N；coords 在 [0,1] 并逐 token
        对应 Psi。p=Psi²，mu=sum_i p_i[x,y,x²,y²,xy]，rho=abs(Psi)^T abs(Psi)。
        不物化 [B,N,K,K]，不换成正交内积；按接收 k、候选 j 拼
        [mu_k,mu_j,abs差,rho]。不 QR、加度/面积权或 sqrt(N)；不保证重根旋转不变。
        """
        for name, value in (("basis", basis), ("coords", coords)):
            _validate_model_tensor(value, self.spatial_bias_net[0], name)
            if value.requires_grad:
                raise ValueError(f"{name} must be constant without gradients")
        if basis.ndim != 3 or basis.shape[0] < 1 or basis.shape[1] < 1 or not 1 <= basis.shape[2] <= basis.shape[1]:
            raise ValueError("basis must have shape [positive B, positive N, 1 <= K <= N]")
        if coords.shape != (basis.shape[1], 2) or (coords < 0).any() or (coords > 1).any():
            raise ValueError("coords must have shape [N,2] with values in [0,1]")
        x, y = coords.unbind(dim=-1)
        coordinate_features = torch.stack((x, y, x.square(), y.square(), x*y), dim=-1)
        moments = basis.square().transpose(1, 2)@coordinate_features
        absolute_basis = basis.abs()
        overlap = absolute_basis.transpose(1, 2)@absolute_basis
        modes = basis.shape[2]
        receiver = moments.unsqueeze(2).expand(-1, -1, modes, -1)
        candidate = moments.unsqueeze(1).expand(-1, modes, -1, -1)
        pair_features = torch.cat((receiver, candidate, (receiver-candidate).abs(), overlap.unsqueeze(-1)), dim=-1)
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in (moments, overlap, pair_features)):
            raise ValueError("spatial descriptors must be finite float32")
        return SpatialDescriptors(moments, overlap, pair_features)

    def spatial_bias(self, descriptors: SpatialDescriptors) -> Tensor:
        """固定 pair_features [B,K,K,16] 经共享16→32→1 SiLU 得 b_x [B,K,K]。

        完整描述子为同模型有限 float32 无梯度；有序 mu_k/mu_j 允许非对称、
        负值与非零对角，不归一化。仅移末单通道轴，保留 B=K=1；四参数求导。
        """
        if not isinstance(descriptors, SpatialDescriptors):
            raise TypeError("descriptors must be SpatialDescriptors")
        for layer in (self.spatial_bias_net[0], self.spatial_bias_net[2]):
            for name in ("moments", "overlap", "pair_features"):
                value = getattr(descriptors, name)
                _validate_model_tensor(value, layer, name)
                if value.requires_grad:
                    raise ValueError(f"{name} must be constant without gradients")
            for parameter in (layer.weight, layer.bias):
                if parameter.is_nested or parameter.layout != torch.strided or not torch.isfinite(parameter).all():
                    raise ValueError("spatial bias parameters must be dense finite float32")
        moments, overlap, features = descriptors.moments, descriptors.overlap, descriptors.pair_features
        if moments.ndim != 3 or moments.shape[0] < 1 or moments.shape[1] < 1 or moments.shape[2] != 5:
            raise ValueError("moments must have shape [positive B, positive K, 5]")
        batch, modes = moments.shape[:2]
        if overlap.shape != (batch, modes, modes) or features.shape != (batch, modes, modes, 16):
            raise ValueError("overlap/pair_features must have matching shapes [B,K,K]/[B,K,K,16]")
        bias = self.spatial_bias_net(features).squeeze(-1)
        if bias.dtype != torch.float32 or not torch.isfinite(bias).all():
            raise ValueError("spatial bias must be finite float32")
        return bias

    def spectral_bias(self, eigenvalues: Tensor) -> Tensor:
        """lambda [B,K]→[lambda_k,lambda_j,abs差]→b_lambda [B,K,K]。

        谱为同模型有限无梯度 float32；误传可训练 lambda 直接报错。
        共享3→32→1 SiLU，参数求导，不向 eig/P_in 反传。仅移末单通道轴，
        保留 B=K=1；不排序/裁剪原谱或归一化分数，允许负值、非对称与非零对角。
        """
        for layer in (self.spectral_bias_net[0], self.spectral_bias_net[2]):
            _validate_model_tensor(eigenvalues, layer, "eigenvalues")
            for parameter in (layer.weight, layer.bias):
                if parameter.is_nested or parameter.layout != torch.strided or not torch.isfinite(parameter).all():
                    raise ValueError("spectral bias parameters must be dense finite float32")
        if eigenvalues.ndim != 2 or eigenvalues.shape[0] < 1 or eigenvalues.shape[1] < 1:
            raise ValueError("eigenvalues must have shape [positive B, positive K]")
        if eigenvalues.requires_grad:
            raise ValueError("eigenvalues must be constants without gradients from C04")
        modes = eigenvalues.shape[1]
        receiver = eigenvalues.unsqueeze(2).expand(-1, -1, modes)
        candidate = eigenvalues.unsqueeze(1).expand(-1, modes, -1)
        features = torch.stack((receiver, candidate, (receiver-candidate).abs()), dim=-1)
        bias = self.spectral_bias_net(features).squeeze(-1)
        if bias.dtype != torch.float32 or not torch.isfinite(bias).all():
            raise ValueError("spectral bias must be finite float32")
        return bias

    def content_terms(self, coefficients: Tensor) -> ContentTerms:
        """当前 C [B,K,64]→q/key [B,K,32]、v [B,K,64] 与 q key^T/sqrt(32)。

        同模型有限 float32，允许非连续输入与无梯度推理；保留 C/投影权重梯度。
        分数 [B,K,K] 无量纲，允许负值、非对称、非零对角，不加偏置或 softmax。
        """
        if not isinstance(coefficients, Tensor):
            raise TypeError("coefficients must be a tensor")
        projections = ((self.q_proj, 32), (self.k_proj, 32), (self.v_proj, 64))
        for name, value in (("coefficients", coefficients),
                            *(("content weight", layer.weight) for layer, _ in projections)):
            if value.is_nested or value.layout != torch.strided or value.is_meta:
                raise ValueError(f"{name} must use a real dense device, not meta")
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite float32 values")
            if value.device != coefficients.device:
                raise ValueError("coefficients and content weights must share a device")
        if coefficients.ndim != 3 or coefficients.shape[0] < 1 or coefficients.shape[1] < 1 or coefficients.shape[2] != 64:
            raise ValueError("coefficients must have shape [positive B, positive K, 64]")
        if any(layer.weight.shape != (width, 64) or layer.bias is not None for layer, width in projections):
            raise ValueError("content projections must use the registered weight shapes without bias")
        query = self.q_proj(coefficients)
        key = self.k_proj(coefficients)
        value = self.v_proj(coefficients)
        if any(result.dtype != torch.float32 or not torch.isfinite(result).all() for result in (query, key, value)):
            raise ValueError("content projections must be finite float32")
        scores = torch.bmm(query, key.transpose(1, 2))/math.sqrt(self.config.query_dim)
        if scores.dtype != torch.float32 or not torch.isfinite(scores).all():
            raise ValueError("content scores must be finite float32")
        return ContentTerms(query, key, value, scores)

    def pool_context(self, x: Tensor) -> Tensor:
        """X [B,N,64]→g [B,64]，仅沿 token 轴取算术均值。

        X 为同模型有限 float32；当前起点或预测反馈均可，不 detach X/g。
        固定均值反向 dX=J/N；参数跨网格共享不保证空间重采样一致。
        """
        _validate_model_tensor(x, self.condition_trunk[0], "x")
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != 64:
            raise ValueError("x must have shape [positive B, positive N, 64]")
        pooled = x.mean(dim=1)
        if pooled.dtype != torch.float32 or not torch.isfinite(pooled).all():
            raise ValueError("pooled context must be finite float32")
        return pooled

    def _condition_features(self, eigenvalues: Tensor, pooled: Tensor) -> Tensor:
        """lambda [B,K] 与 g [B,64]→H [B,K,64]，H=SiLU(W[lambda,g]+b)。

        同模型有限 float32、严格 B 匹配；lambda 无梯度，g 仅沿 K 展开。
        保留 g/参数梯度，不把 k/K/网格/动作/时长加入条件。
        """
        _validate_model_tensor(eigenvalues, self.condition_trunk[0], "eigenvalues")
        _validate_model_tensor(pooled, self.condition_trunk[0], "pooled")
        if eigenvalues.ndim != 2 or eigenvalues.shape[0] < 1 or eigenvalues.shape[1] < 1:
            raise ValueError("eigenvalues must have shape [positive B, positive K]")
        if pooled.ndim != 2 or pooled.shape != (eigenvalues.shape[0], 64):
            raise ValueError("pooled must have shape [matching B, 64]")
        if eigenvalues.requires_grad:
            raise ValueError("eigenvalues must be constants without gradients from C04")
        # 仅沿模态轴复用本样本上下文，不混合batch或引入模态编号。
        inputs = torch.cat((eigenvalues.unsqueeze(-1), pooled.unsqueeze(1).expand(-1, eigenvalues.shape[1], -1)), dim=-1)
        features = self.condition_trunk(inputs)
        if features.dtype != torch.float32 or not torch.isfinite(features).all():
            raise ValueError("condition features must be finite float32")
        return features

    def _condition_free(self, eigenvalues: Tensor, pooled: Tensor) -> _FreeCoefficients:
        """式9/10：lambda/g 经共享 H 生成32个同模态阻尼旋转块。

        输入同模型有限 float32，lambda [B,K] 无梯度、g [B,64] 保留梯度。
        gamma=softplus(raw_gamma)>=0，omega 有符号，A=[[-gamma,-omega],
        [omega,-gamma]]，率单位周期^-1。保留干路/双头图，不调用 B 头；
        极负 raw 导致 gamma 下溢为0合法，不加阻尼下限。
        """
        features = self._condition_features(eigenvalues, pooled)
        return self._free_from_features(features)

    def _free_from_features(self, features: Tensor) -> _FreeCoefficients:
        """从共享 H [B,K,64] 依次计算 raw_gamma、softplus、omega 与式9/10的 A。

        H 为已验证模型 float32，保留 H/双头梯度；不计算干路或 B。
        """
        _validate_model_tensor(features, self.gamma_head, "gamma head features")
        _validate_model_tensor(features, self.omega_head, "omega head features")
        raw_gamma = self.gamma_head(features)
        gamma = torch.nn.functional.softplus(raw_gamma)
        omega = self.omega_head(features)
        a_blocks = torch.stack((torch.stack((-gamma, -omega), dim=-1),
                                torch.stack((omega, -gamma), dim=-1)), dim=-2)
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all()
               for value in (raw_gamma, gamma, omega, a_blocks)):
            raise ValueError("free coefficients must be finite float32")
        return _FreeCoefficients(raw_gamma, gamma, omega, a_blocks)

    def condition(self, eigenvalues: Tensor, pooled: Tensor) -> ConditionedCoefficients:
        """lambda [B,K]、g [B,64] 经唯一共享 H 生成同次 A/B。

        同模型有限 float32，lambda 无梯度，g 保留梯度；B_flat 按 c*m+d
        还原 [B,K,64,m]，不拼动作、时长或模态编号，不缓存上下文。
        """
        features = self._condition_features(eigenvalues, pooled)
        free = self._free_from_features(features)
        _validate_model_tensor(features, self.b_head, "B head features")
        b_flat = self.b_head(features)
        if b_flat.dtype != torch.float32 or not torch.isfinite(b_flat).all():
            raise ValueError("B coefficients must be finite float32")
        b_matrix = b_flat.reshape(*features.shape[:2], 64, self.config.action_dim)
        return ConditionedCoefficients(free.raw_gamma, free.gamma, free.omega, free.a_blocks, b_matrix)

    def action_rhs(self, action: Tensor, b_matrix: Tensor) -> Tensor:
        """式8：B [B,K,64,m] 与 a [B,m] 仅收缩动作轴，返回 Ba [B,K,64]。

        同设备有限 float32，允许非连续输入；动作仅沿 K 共享，保留 a/B 梯度。
        输出单位系数/周期；不乘时长、加 C、换算秒或缩放动作。
        """
        for name, value in (("action", action), ("b_matrix", b_matrix)):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.is_nested or value.layout != torch.strided or value.is_meta:
                raise ValueError(f"{name} must use a real dense device, not meta")
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite float32 values")
        if action.ndim != 2 or action.shape[0] < 1 or action.shape[1] != self.config.action_dim:
            raise ValueError("action must have shape [positive B, action_dim]")
        if b_matrix.ndim != 4 or b_matrix.shape[:1] != action.shape[:1] or b_matrix.shape[1] < 1 or b_matrix.shape[2:] != (64, self.config.action_dim):
            raise ValueError("b_matrix must have shape [matching B, positive K, 64, action_dim]")
        if action.device != b_matrix.device:
            raise ValueError("action and b_matrix must share a device")
        result = torch.einsum("bkcd,bd->bkc", b_matrix, action)
        if result.dtype != torch.float32 or not torch.isfinite(result).all():
            raise ValueError("linear action contribution must be finite float32")
        return result

    def free_propagate(
        self,
        coefficients: Tensor,
        *,
        gamma: Tensor,
        omega: Tensor,
        duration: Tensor,
    ) -> Tensor:
        """式11：固定 A=-gamma I+omega J 的精确自由解 S_tau C。

        C [B,K,64]、gamma/omega [B,K,32] 同设备 float32，或均 CPU float64。
        gamma>=0，率单位周期^-1；duration 为无梯度 CPU float64 [B] 非负周期。
        率/时长转 double，a=exp(-gamma*tau)cos(omega*tau)、
        b=exp(-gamma*tau)sin(omega*tau)，转回 C 精度后逐对计算
        (c0',c1')=(a*c0-b*c1,b*c0+a*c1)，保留 C/两率梯度。
        零时长沿同闭式恒等；式12半群/式13非扩张仅适用于区间固定同组率，
        不推广为完整 raw Phi(0)=Z 或非线性稳定性。合法指数下溢允许，非有限量报错。
        """
        for name, value in (("coefficients", coefficients), ("gamma", gamma),
                            ("omega", omega), ("duration", duration)):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.is_nested or value.layout != torch.strided or value.is_meta:
                raise ValueError(f"{name} must use a real dense device, not meta")
            if value.dtype not in (torch.float32, torch.float64):
                raise ValueError(f"{name} must be a real float32 or float64 tensor")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite values")
        if coefficients.ndim != 3 or coefficients.shape[0] < 1 or coefficients.shape[1] < 1 or coefficients.shape[2] != 64:
            raise ValueError("coefficients must have shape [positive B, positive K, 64]")
        if gamma.shape != (*coefficients.shape[:2], 32) or omega.shape != gamma.shape:
            raise ValueError("gamma and omega must have matching shape [B, K, 32]")
        if any(value.dtype != coefficients.dtype or value.device != coefficients.device for value in (gamma, omega)):
            raise ValueError("coefficients, gamma and omega must share dtype and device")
        if coefficients.dtype == torch.float64 and coefficients.device.type != "cpu":
            raise ValueError("float64 analytic propagation is supported only on CPU")
        if duration.device.type != "cpu" or duration.dtype != torch.float64 or duration.shape != coefficients.shape[:1]:
            raise ValueError("duration must be CPU float64 with shape [B]")
        if duration.requires_grad or (duration < 0).any() or (gamma < 0).any():
            raise ValueError("duration must be nonnegative without gradients and gamma must be nonnegative")
        time = duration.to(device=gamma.device).reshape(-1, 1, 1)
        damping = gamma.to(dtype=torch.float64)*time
        angle = omega.to(dtype=torch.float64)*time
        if not torch.isfinite(damping).all() or not torch.isfinite(angle).all():
            raise ValueError("float64 damping and angle products must be finite")
        decay = torch.exp(-damping)
        a = (decay*torch.cos(angle)).to(dtype=coefficients.dtype)
        b = (decay*torch.sin(angle)).to(dtype=coefficients.dtype)
        c0, c1 = coefficients[..., 0::2], coefficients[..., 1::2]
        result = torch.stack((a*c0-b*c1, b*c0+a*c1), dim=-1).flatten(start_dim=-2)
        if not torch.isfinite(result).all():
            raise ValueError("propagated coefficients must be finite")
        return result
