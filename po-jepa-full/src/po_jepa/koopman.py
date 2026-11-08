"""08 §5：Koopman 对照——可学空间基底上的模态系数，逐对阻尼旋转加线性动作项，按段精确积分，残差读出。

模态之间没有相互作用，γ、ω 不随上下文变化，只用 JEPA 损失训练。输入输出合同和 TokenTransformer.forward 相同。
"""

import math

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .actions import ActionSegments, clip_action_segments, validate_action_segments
from .fourier import dct_basis, mode_order
from .geometry import GridSpec, apply_spatial_transfer, build_spatial_transfer, regular_grid

BASIS_HEIGHT, BASIS_WIDTH = 16, 16
BASIS_TOKENS = BASIS_HEIGHT*BASIS_WIDTH


@dataclass(frozen=True)
class KoopmanConfig:
    """08 §5.7：Koopman 的模型配置；检查点的 model_config 存它的快照。"""

    modes: int
    action_dim: int
    encoder_dim: int = 1408
    dynamic_dim: int = 64
    basis_grid: tuple = (16, 16)
    basis_init: str = "dct2"
    gamma_init: float = 0.1
    action_gain_std: float = 0.001
    phi1_series_threshold: float = 1e-4
    readout: str = "residual"

    def __post_init__(self) -> None:
        """分配参数前拒绝非法维度和偏离 08 §5 的固定选择。"""
        for name in ("modes", "action_dim", "encoder_dim"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.dynamic_dim) is not int or self.dynamic_dim != 64:
            raise ValueError("dynamic_dim must be 64 (32 complex channel pairs)")
        if type(self.basis_grid) is not tuple or self.basis_grid != (BASIS_HEIGHT, BASIS_WIDTH):
            raise ValueError("basis_grid must be the tuple (16, 16)")
        if self.basis_init != "dct2" or self.readout != "residual":
            raise ValueError("Koopman uses the dct2 basis initialization and the residual readout")
        for name in ("gamma_init", "action_gain_std"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive float")
        if type(self.phi1_series_threshold) is not float or self.phi1_series_threshold != 1e-4:
            raise ValueError("phi1_series_threshold must be 1e-4")


def parameter_count(config: KoopmanConfig) -> int:
    """08 §5.2：2·E·D + D + E + K·(256 + D + D·m)；E=1408、D=64 时为 181,696 + K·(320 + 64m)。"""
    if not isinstance(config, KoopmanConfig):
        raise TypeError("config must be a KoopmanConfig")
    e, d, k, m = config.encoder_dim, config.dynamic_dim, config.modes, config.action_dim
    return 2*e*d+d+e+k*(BASIS_TOKENS+d+d*m)


def choose_modes(po_parameters: int, *, action_dim: int, encoder_dim: int = 1408) -> int:
    """08 §5.6：使 |parameter_count(K) − PO-JEPA 参数量| 最小的正整数 K，相等时取较小的 K。"""
    if type(po_parameters) is not int or po_parameters <= 0:
        raise ValueError("po_parameters must be a positive integer")
    def difference(k):
        return abs(parameter_count(KoopmanConfig(modes=k, action_dim=action_dim, encoder_dim=encoder_dim))-po_parameters)
    k = 1
    while difference(k+1) < difference(k):   # 参数量随 K 严格递增，差先减后增
        k += 1
    return k


def phi1(w: Tensor, threshold: float) -> Tensor:
    """φ1(w) = (e^w − 1)/w：|w| < threshold 时用 1 + w/2 + w²/6，否则用 expm1(w)/w。

    两个分支都先换成安全参数（未选分支的 0/0 不会把 NaN 带进反向），再按 |w| 选。
    """
    small = w.abs() < threshold
    ratio_arg = torch.where(small, torch.ones_like(w), w)
    series_arg = torch.where(small, w, torch.zeros_like(w))
    ratio = torch.expm1(ratio_arg)/ratio_arg
    series = 1+series_arg/2+series_arg*series_arg/6
    return torch.where(small, series, ratio)


def _validate_inputs(raw: Tensor, controls: ActionSegments, duration: Tensor, grid: GridSpec,
                     config: KoopmanConfig, device: torch.device) -> None:
    """与 baselines.validate_inputs 相同的输入合同，只把动作维 2 换成 config.action_dim。"""
    if (not isinstance(raw, Tensor) or raw.layout != torch.strided or raw.is_nested or raw.is_meta
        or raw.dtype != torch.float32 or raw.ndim != 3 or min(raw.shape) <= 0
        or raw.shape[-1] != config.encoder_dim or raw.device != device or not torch.isfinite(raw).all()):
        raise ValueError("raw must be finite model-device float32 [B,N,encoder_dim]")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be GridSpec")
    grid.__post_init__()
    if raw.shape[1] != grid.height*grid.width:
        raise ValueError("raw N must match actual grid")
    validate_action_segments(controls, query_duration=duration)
    if controls.u_values.shape[0] != raw.shape[0] or controls.u_values.device != device:
        raise ValueError("control B/device must match raw")
    if controls.control_spec.action_dim != config.action_dim:
        raise ValueError("control channels must equal the Koopman action_dim")


class KoopmanPredictor(nn.Module):
    """08 §5：Φ(Z, u, T; h) = Z + P_out(W_h c_T) − P_out(W_h c_0)。"""

    def __init__(self, config: KoopmanConfig) -> None:
        """按 08 §5.2 的表的顺序创建参数；只有 p_in、p_out、action_gain 消耗调用者的 torch 随机数。"""
        super().__init__()
        if not isinstance(config, KoopmanConfig):
            raise TypeError("config must be a KoopmanConfig")
        config.__post_init__()
        self.config = config
        self.encoder_dim, self.dynamic_dim, self.readout = config.encoder_dim, config.dynamic_dim, config.readout
        self.p_in = nn.Linear(config.encoder_dim, config.dynamic_dim, bias=True, device="cpu", dtype=torch.float32)
        self.p_out = nn.Linear(config.dynamic_dim, config.encoder_dim, bias=True, device="cpu", dtype=torch.float32)
        for layer in (self.p_in, self.p_out):
            nn.init.xavier_uniform_(layer.weight, gain=1.)
            nn.init.zeros_(layer.bias)
        pairs = config.dynamic_dim//2
        self.basis = nn.Parameter(dct_basis(BASIS_HEIGHT, BASIS_WIDTH, mode_order(config.modes)).to(dtype=torch.float32))
        self.raw_gamma = nn.Parameter(torch.full((config.modes, pairs), math.log(math.expm1(config.gamma_init)),
                                                 dtype=torch.float32, device="cpu"))
        self.omega = nn.Parameter(torch.zeros(config.modes, pairs, dtype=torch.float32, device="cpu"))
        self.action_gain = nn.Parameter(torch.empty(config.modes, config.dynamic_dim, config.action_dim,
                                                    dtype=torch.float32, device="cpu"))
        nn.init.normal_(self.action_gain, mean=0., std=config.action_gain_std)

    def grid_basis(self, grid: GridSpec) -> Tensor:
        """08 §5.3 第 1 步：W_h [N_h, K]。16×16 上就是 basis；其他网格用和 token 相同的重采样 R（梯度传回 basis）。

        16×16 的坐标必须和 regular_grid(16,16) 逐位相同；其他网格的坐标必须和 R 的目标坐标逐位相同。
        """
        source = regular_grid(BASIS_HEIGHT, BASIS_WIDTH)
        if (grid.height, grid.width) == (BASIS_HEIGHT, BASIS_WIDTH):
            if not torch.equal(grid.coords, source.coords):
                raise ValueError("grid coordinates must equal the 16x16 basis grid or its spatial transfer")
            return self.basis
        transfer = build_spatial_transfer(source, target_height=grid.height, target_width=grid.width)
        if not torch.equal(transfer.target.coords, grid.coords):
            raise ValueError("grid coordinates must equal the 16x16 basis grid or its spatial transfer")
        return apply_spatial_transfer(self.basis[None], transfer)[0]

    def analysis(self, x: Tensor, basis: Tensor) -> Tensor:
        """08 §5.3 第 2 步：c_0 = (256 / N_h) · W_hᵀ X，[B, K, 64]；16×16 上就是 Wᵀ X（乘 1.0 是精确的）。"""
        return torch.matmul(basis.transpose(0, 1), x)*(BASIS_TOKENS/basis.shape[0])

    def integrate(self, initial: Tensor, controls: ActionSegments, duration: Tensor) -> Tensor:
        """08 §5.3 第 3、4 步：每对通道 z = c[2i] + i·c[2i+1]，λ = −softplus(raw_gamma) + i·ω，逐段
        z ← e^{λΔ} z + Δ·φ1(λΔ)·φ，φ = B·u 同样配对。e^{λΔ}、Δ·φ1(λΔ) 用 complex128 算，乘之前转 complex64。

        段取 [0, T) 内的有效段（clip_action_segments，和 TokenTransformer 相同），Δ_j = 段长；
        padding 段不读，行里没有第 j 段时该行原样保留，所以 T = 0 时 c_T 与 c_0 逐位相同。
        """
        clipped = clip_action_segments(controls, start=torch.zeros_like(duration), end=duration)
        valid = clipped.segment_valid
        lengths = torch.where(valid, clipped.ends-clipped.starts, torch.zeros_like(clipped.ends))
        total = lengths.sum(1)
        if any(abs(float(t)-float(d)) > 1e-12 for t, d in zip(total, duration)):   # 08 §5.3 第 4 步：绝对差 1e-12
            raise ValueError("effective segment lengths must sum to the query duration")
        batch, modes, width = initial.shape
        state = torch.view_as_complex(initial.reshape(batch, modes, width//2, 2))
        rate = torch.complex(-F.softplus(self.raw_gamma.to(dtype=torch.float64)), self.omega.to(dtype=torch.float64))
        device = initial.device
        for j in range(valid.shape[1]):
            active = valid[:, j].to(device)
            if not bool(active.any()):
                continue
            delta = lengths[:, j].to(device=device)[:, None, None]
            w = rate[None]*delta
            propagator = torch.exp(w).to(dtype=torch.complex64)
            forcing_integral = (delta*phi1(w, self.config.phi1_series_threshold)).to(dtype=torch.complex64)
            action = torch.where(valid[:, j, None].to(device), clipped.u_values[:, j], torch.zeros_like(clipped.u_values[:, j]))
            forcing = torch.einsum("kcm,bm->bkc", self.action_gain, action)
            phi = torch.view_as_complex(forcing.reshape(batch, modes, width//2, 2))
            state = torch.where(active[:, None, None], propagator*state+forcing_integral*phi, state)
        return torch.view_as_real(state).reshape(batch, modes, width)

    def forward(self, raw: Tensor, controls: ActionSegments, duration: Tensor, grid: GridSpec) -> Tensor:
        """raw Z [B,N_h,E] 模型 float32，duration 为 CPU float64 [B]（控制周期）→ 同形 raw 预测；p_out 调用两次。"""
        _validate_inputs(raw, controls, duration, grid, self.config, self.p_in.weight.device)
        basis = self.grid_basis(grid)
        initial = self.analysis(self.p_in(raw), basis)
        final = self.integrate(initial, controls, duration)
        output = self.p_out(torch.matmul(basis, final))
        # 偏置在两项中精确相消；c_T = c_0 时差为 0，输出逐位等于输入 Z（和 POJEPA 的残差读出相同的写法）。
        output = raw+(output-self.p_out(torch.matmul(basis, initial)))
        if not torch.isfinite(output).all():
            raise ValueError("nonfinite Koopman raw prediction")
        return output


def build_koopman(config: KoopmanConfig, seed: int, device: torch.device) -> KoopmanPredictor:
    """08 §1.2、§5.2：fork_rng 里用 seed 设种子后构造（和 comparison._make_transformer 相同），再转到 device。

    仿真传 KoopmanConfig(modes=choose_modes(PO 参数量, action_dim=2), action_dim=2) 和 checks.model_seed()。
    """
    if type(seed) is not int:
        raise TypeError("seed must be an int")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = KoopmanPredictor(config)
    return model.to(device)
