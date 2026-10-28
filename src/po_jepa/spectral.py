"""式5—7：双核 W、规范化 L、最低 K 谱与欧氏模态分析/合成。

空间核与固定 Q 特征核相乘，分母为带宽平方，无额外 1/2。
CPU float64 构图/求谱并停止整个谱分支梯度，X 的其他预测路径保留。
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .geometry import GridSpec


@dataclass(frozen=True)
class GraphConfig:
    """固定 Q/带宽/谱配置，不参加优化。

    q_dim=16、q_seed=1703、sigma_x=.25、sigma_z=1、max_modes=128；
    双精度诊断容差 1e-10/1e-8。带宽为无量纲尺度；非单位带宽可用于公式诊断。
    """

    q_dim: int
    q_seed: int
    sigma_x: float
    sigma_z: float
    max_modes: int
    diagnostic_atol: float
    diagnostic_rtol: float
    # v4：affinity_source="encoder" 时用 stop-gradient 编码器输出 Z [B,N,feature_dim] 建图，
    # Q 为 [16,feature_dim]；"projected" 保持一轮行为（X=P_in(Z).detach()，宽 64）。
    affinity_source: str = "projected"
    feature_dim: int = 64
    # v4：fixed_modes=True 时所有网格都取同一个 K=max_modes，N<K 直接报错，不再 min(K,N)。
    fixed_modes: bool = False

    def __post_init__(self) -> None:
        """拒绝非法维度、torch种子范围和数值，不把布尔开关当整数。"""
        if type(self.q_dim) is not int or self.q_dim != 16:
            raise ValueError("q_dim must be the fixed first-round width 16")
        if self.affinity_source not in ("projected", "encoder"):
            raise ValueError("affinity_source must be 'projected' or 'encoder'")
        if type(self.feature_dim) is not int or self.feature_dim < self.q_dim:
            raise ValueError("feature_dim must be an integer >= q_dim")
        if self.affinity_source == "projected" and self.feature_dim != 64:
            raise ValueError("projected affinity uses the 64-wide dynamic features")
        if type(self.fixed_modes) is not bool:
            raise ValueError("fixed_modes must be bool")
        if type(self.q_seed) is not int or not -(2**63) <= self.q_seed < 2**64:
            raise ValueError("q_seed must be an integer in the torch seed range")
        if type(self.max_modes) is not int or self.max_modes <= 0:
            raise ValueError("max_modes must be a positive integer")
        for name in ("sigma_x", "sigma_z", "diagnostic_atol", "diagnostic_rtol"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
            if value < 0 or (name.startswith("sigma") and value == 0):
                raise ValueError(f"{name} must be positive for bandwidths, nonnegative for tolerances")


@dataclass(frozen=True)
class LaplacianResult:
    """式6：degree [B,N] 为 W 行和，L [B,N,N]=I-D^-1/2 W D^-1/2。

    CPU float64、无量纲、无梯度，只读，供后续求谱。
    """

    degree: Tensor
    laplacian: Tensor


@dataclass(frozen=True)
class SpectralDiagnostics:
    """原生CPU float64谱诊断，不是训练损失或统一理论界的证明。

    residual_norms:[B,K]为每列||L psi-lambda psi||_2；orthogonality_error:[B]
    为||Psi^T Psi-I||_F。consecutive_gaps:[B,max(K-1,0)]，cutoff_gap:[B]
    为第K+1减第K个原生特征值，全K时None。sign_pivot_indices:[B,K]为
    CPU int64最大绝对分量的首个token索引；其余字段CPU float64，全部无梯度。
    间隙不能从float32副本重算；小间隙/截入重根说明敏感性，不自动换K或基底。
    """

    residual_norms: Tensor
    orthogonality_error: Tensor
    consecutive_gaps: Tensor
    cutoff_gap: Tensor | None
    sign_pivot_indices: Tensor


@dataclass(frozen=True)
class SpectralResult:
    """最低 K 模态及其模型副本，字段和张量只读。

    eigenvalues_cpu [B,K]/basis_cpu [B,N,K] 为 CPU float64；同形
    模型副本为目标设备 float32，均无梯度。先 double 规范符号再复制，
    不重新正交化；重根方向不唯一，按完整重根簇子空间核验。
    """

    eigenvalues_cpu: Tensor
    basis_cpu: Tensor
    eigenvalues: Tensor
    basis: Tensor
    diagnostics: SpectralDiagnostics


class GraphSpectrum(nn.Module):
    """固定 Q/带宽、双核 W、规范化 L 与最低 K 谱。

    q [16,64] 与两个标量带宽为 float32 持久 buffer，无可训练参数。
    每次重新构图，不缓存 X/W；Q 专属种子与模型全局 RNG 分离。
    """

    def __init__(self, config: GraphConfig) -> None:
        """专属 CPU double 正态→薄 QR→符号规范→float32 Q。

        R 负对角翻转对应 U 列，零符号取 +1；Q 行单位范数，无额外缩放。
        只消费独立 CPU generator，不改变全局 RNG、设备或精度。
        """
        super().__init__()
        if not isinstance(config, GraphConfig):
            raise TypeError("config must be a GraphConfig")
        self.config = config
        generator = torch.Generator(device="cpu").manual_seed(config.q_seed)
        normal = torch.randn(config.feature_dim, config.q_dim, generator=generator, device="cpu", dtype=torch.float64)
        u, r = torch.linalg.qr(normal, mode="reduced")
        signs = torch.where(r.diagonal() < 0, -1., 1.)
        self.register_buffer("q", (u*signs).T.contiguous().to(dtype=torch.float32))
        self.register_buffer("sigma_x", torch.tensor(config.sigma_x, device="cpu", dtype=torch.float32))
        self.register_buffer("sigma_z", torch.tensor(config.sigma_z, device="cpu", dtype=torch.float32))
        self._working_buffers()

    def _working_buffers(self) -> tuple[Tensor, Tensor, Tensor]:
        """只读检查实际buffer后转CPU double；不重新QR、不修正损坏状态。

        q的float32量化不会被转double消除，正交性用模型侧1e-5/1e-4标准。
        带宽保存后及分母平方均须有限且严格正，不能加epsilon掩盖不可表示值。
        返回Q_work及两个平方分母，仅用于当前构图，不写回持久状态。
        """
        work = []
        for name, shape in (("q", (16, self.config.feature_dim)), ("sigma_x", ()), ("sigma_z", ())):
            value = getattr(self, name)
            if (name not in self._buffers or name in self._non_persistent_buffers_set
                or not isinstance(value, Tensor) or value.shape != shape
                or value.dtype != torch.float32 or value.device.type == "meta" or value.requires_grad):
                raise ValueError(f"{name} must remain a persistent fixed float32 buffer with shape {shape}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite values")
            work.append(value.to(device="cpu", dtype=torch.float64))
        q, sigma_x, sigma_z = work
        if not torch.allclose(q @ q.T, torch.eye(16, device="cpu", dtype=torch.float64), atol=1e-5, rtol=1e-4):
            raise ValueError("q must retain orthonormal rows within float32 storage precision")
        sx2, sz2 = sigma_x.square(), sigma_z.square()
        if sigma_x <= 0 or sigma_z <= 0 or sx2 <= 0 or sz2 <= 0 or not torch.isfinite(sx2+sz2):
            raise ValueError("bandwidth buffers and squared denominators must be finite and positive")
        return q, sx2, sz2

    def affinity(self, x: Tensor, grid: GridSpec) -> Tensor:
        """式5：X [B,N,64] 模型 float32 与网格→W [B,N,N] CPU float64。

        坐标为无梯度 CPU float64 [N,2]、无量纲 [0,1]，N=H*W。仅将
        x.detach() 副本转 double 构图，原 X 供模态分析/Pool 反传。
        Q 投影后直接差分平方，两个高斯核相乘；不做 Gram 距离、clamp、
        归一化或对称/对角修补。远距下溢 0 合法，非有限结果报错。
        """
        if not isinstance(x, Tensor) or not isinstance(grid, GridSpec):
            raise TypeError("x must be a tensor and grid must be a GridSpec")
        grid.__post_init__()
        width = self.config.feature_dim
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] != grid.height*grid.width or x.shape[2] != width:
            raise ValueError(f"x must have shape [positive B, grid height*width, {width}]")
        if x.dtype != torch.float32 or x.device.type == "meta" or not torch.isfinite(x).all():
            raise ValueError("x must be a finite float32 tensor on a real model device")
        q, sx2, sz2 = self._working_buffers()
        x_graph = x.detach().to(device="cpu", dtype=torch.float64)
        projected = x_graph @ q.T
        spatial_sq = (grid.coords[:, None, :] - grid.coords[None, :, :]).square().sum(-1)
        feature_sq = (projected[:, :, None, :] - projected[:, None, :, :]).square().sum(-1)
        affinity = torch.exp(-spatial_sq/sx2) * torch.exp(-feature_sq/sz2)
        if not torch.isfinite(affinity).all():
            raise ValueError("affinity must contain finite values")
        return affinity

    def normalized_laplacian(self, affinity: Tensor) -> LaplacianResult:
        """式6：W [B,N,N]→degree [B,N]、L [B,N,N]，CPU float64 无梯度。

        W 须在 [0,1]、单位对角、数值对称；不连通和非对角下溢 0 合法。
        行和 D 后两侧各乘 D^-1/2，不改成随机游走矩阵。无 epsilon、裁剪
        或对称修补；容差仅判断对称性，不改变 W。
        """
        if not isinstance(affinity, Tensor):
            raise TypeError("affinity must be a tensor")
        if affinity.layout != torch.strided or affinity.is_nested or affinity.is_complex():
            raise ValueError("affinity must be a dense real tensor")
        if (affinity.ndim != 3 or affinity.shape[0] < 1 or affinity.shape[1] < 1
            or affinity.shape[1] != affinity.shape[2]):
            raise ValueError("affinity must have shape [positive B, positive N, N]")
        if affinity.device.type != "cpu" or affinity.dtype != torch.float64 or affinity.requires_grad:
            raise ValueError("affinity must be CPU float64 without gradient")
        if not torch.isfinite(affinity).all():
            raise ValueError("affinity must contain finite values")
        if torch.any((affinity < 0) | (affinity > 1)) or not torch.all(affinity.diagonal(dim1=-2, dim2=-1) == 1):
            raise ValueError("first-round affinity must lie in [0,1] with an exact unit diagonal")
        atol, rtol = self.config.diagnostic_atol, self.config.diagnostic_rtol
        if not torch.allclose(affinity, affinity.transpose(-1, -2), atol=atol, rtol=rtol):
            raise ValueError("affinity must be symmetric within diagnostic tolerances")
        degree = affinity.sum(dim=-1)
        if not torch.isfinite(degree).all() or torch.any(degree <= 0):
            raise ValueError("affinity degree must be finite and positive")
        inv_sqrt_degree = degree.rsqrt()
        nodes = affinity.shape[-1]
        laplacian = (torch.eye(nodes, dtype=affinity.dtype, device=affinity.device)[None, :, :]
                     - inv_sqrt_degree[:, :, None] * affinity * inv_sqrt_degree[:, None, :])
        if not torch.isfinite(laplacian).all():
            raise ValueError("normalized laplacian must contain finite values")
        if not torch.allclose(laplacian, laplacian.transpose(-1, -2), atol=atol, rtol=rtol):
            raise ValueError("normalized laplacian must be symmetric within diagnostic tolerances")
        return LaplacianResult(degree, laplacian)

    def lowest_eigenpairs(self, laplacian: Tensor, *, k: int,
                          model_device: torch.device) -> SpectralResult:
        """CPU float64 L [B,N,N]→最低 k 特征对与原生谱诊断，1<=k<=N。

        在 no_grad 下对 detach 副本 eigh(UPLO='L')，整个谱分支停止梯度。
        最大绝对分量的首个 token 为符号 pivot，不旋转重根子空间。原 L
        计算残差与正交误差，完整 double 谱计算间隙；容差内小负值原样保留。
        返回 CPU float64 与显式模型设备 float32 副本，均无梯度；CUDA 不可用报错。
        """
        if not isinstance(laplacian, Tensor):
            raise TypeError("laplacian must be a tensor")
        if laplacian.layout != torch.strided or laplacian.is_nested or laplacian.is_complex():
            raise ValueError("laplacian must be a dense real tensor")
        if (laplacian.ndim != 3 or laplacian.shape[0] < 1 or laplacian.shape[1] < 1
            or laplacian.shape[1] != laplacian.shape[2]):
            raise ValueError("laplacian must have shape [positive B, positive N, N]")
        if laplacian.device.type != "cpu" or laplacian.dtype != torch.float64:
            raise ValueError("laplacian must be CPU float64")
        if not torch.isfinite(laplacian).all():
            raise ValueError("laplacian must contain finite values")
        atol, rtol = self.config.diagnostic_atol, self.config.diagnostic_rtol
        if not torch.allclose(laplacian, laplacian.transpose(-1, -2), atol=atol, rtol=rtol):
            raise ValueError("laplacian must be symmetric within diagnostic tolerances")
        nodes = laplacian.shape[-1]
        if type(k) is not int or not 1 <= k <= nodes:
            raise ValueError("k must be an integer in [1,N]")
        if not isinstance(model_device, torch.device):
            raise TypeError("model_device must be a torch.device")
        if model_device.type not in ("cpu", "cuda"):
            raise ValueError("first-round model_device must be CPU or available CUDA")
        if model_device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("requested CUDA is unavailable")
            index = torch.cuda.current_device() if model_device.index is None else model_device.index
            if not 0 <= index < torch.cuda.device_count():
                raise ValueError("requested CUDA index does not exist")
            model_device = torch.device("cuda", index)
        with torch.no_grad():
            matrix = laplacian.detach()
            values, vectors = torch.linalg.eigh(matrix, UPLO="L")
            if not torch.isfinite(values).all() or not torch.isfinite(vectors).all():
                raise ValueError("eigensolver must return finite eigenpairs")
            if torch.any(values < -atol) or torch.any(values > 2+atol+2*rtol):
                raise ValueError("full laplacian spectrum must lie in [0,2] within diagnostic tolerances")
            selected = vectors[:, :, :k]
            pivots = selected.abs().argmax(dim=1)
            pivot_values = selected.gather(1, pivots[:, None, :]).squeeze(1)
            signs = torch.where(pivot_values < 0, -torch.ones_like(pivot_values), torch.ones_like(pivot_values))
            basis_cpu = (selected*signs[:, None, :]).detach()
            eigenvalues_cpu = values[:, :k].detach().clone()
            residuals = torch.linalg.vector_norm(matrix @ basis_cpu - basis_cpu*eigenvalues_cpu[:, None, :], dim=1)
            gram_error = basis_cpu.transpose(-1, -2) @ basis_cpu - torch.eye(k, device="cpu", dtype=torch.float64)[None]
            orthogonality = torch.linalg.matrix_norm(gram_error, ord="fro", dim=(-2, -1))
            gaps = values[:, 1:k]-values[:, :k-1]
            cutoff = values[:, k]-values[:, k-1] if k < nodes else None
            diagnostics = SpectralDiagnostics(residuals, orthogonality, gaps, cutoff, pivots)
            if any(not torch.isfinite(t).all() for t in (residuals, orthogonality, gaps)) or (cutoff is not None and not torch.isfinite(cutoff).all()):
                raise ValueError("spectral diagnostics must contain finite values")
            return SpectralResult(eigenvalues_cpu, basis_cpu,
                                  eigenvalues_cpu.to(device=model_device, dtype=torch.float32).detach(),
                                  basis_cpu.to(device=model_device, dtype=torch.float32).detach(), diagnostics)


def _validate_modal_tensors(values: Tensor, basis: Tensor) -> None:
    """核同设备 float32 模态张量：正 B/N、1<=K<=N、特征宽 64。

    允许非连续视图；Psi 必须无梯度，误传直接报错，不额外 detach、QR 或求谱。
    """
    for name, tensor in (("values", values), ("basis", basis)):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if tensor.is_nested or tensor.layout != torch.strided or tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be a dense real float32 tensor")
        if tensor.device.type == "meta":
            raise ValueError(f"{name} must have an actual computation device")
        if tensor.ndim != 3 or any(size <= 0 for size in tensor.shape):
            raise ValueError(f"{name} must have three nonempty axes")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain finite values")
    if values.device != basis.device:
        raise ValueError("values and basis must be on the same device")
    if values.shape[0] != basis.shape[0] or values.shape[2] != 64:
        raise ValueError("batch must match exactly and feature width must be 64")
    if basis.shape[2] > basis.shape[1]:
        raise ValueError("basis must satisfy 1 <= K <= N")
    if basis.requires_grad:
        raise ValueError("basis must be the stop-gradient C04 model constant")


def modal_analysis(x: Tensor, basis: Tensor) -> Tensor:
    """式7：C=Psi^T X，X [B,N,64]、Psi [B,N,K]→C [B,K,64]。

    同设备 float32 无量纲特征，不加面积/度权或 sqrt(N)。Psi 无梯度，
    X/P_in 保留梯度；固定 Psi 时 dX=Psi G。K<N 为子空间投影，一般有损。
    """
    _validate_modal_tensors(x, basis)
    if x.shape[1] != basis.shape[1]:
        raise ValueError("analysis requires matching spatial N axes")
    coefficients = torch.bmm(basis.transpose(1, 2), x)
    if coefficients.dtype != torch.float32 or not torch.isfinite(coefficients).all():
        raise ValueError("modal coefficients must be finite float32 values")
    return coefficients


def modal_synthesis(coefficients: Tensor, basis: Tensor) -> Tensor:
    """图1：X_hat=Psi C，C [B,K,64]、Psi [B,N,K]→[B,N,64]。

    同设备 float32，无面积权、归一化、偏置或残差。Psi 无梯度，C 正常
    反传，dC=Psi^T H；与分析组合为 Psi Psi^T X，不保证投影读出互逆。
    """
    _validate_modal_tensors(coefficients, basis)
    if coefficients.shape[1] != basis.shape[2]:
        raise ValueError("synthesis requires matching modal K axes")
    field = torch.bmm(basis, coefficients)
    if field.dtype != torch.float32 or not torch.isfinite(field).all():
        raise ValueError("synthesized field must be finite float32 values")
    return field
