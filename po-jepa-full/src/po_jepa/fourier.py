"""08 §4：固定 Fourier 对照的谱模块——二维 DCT-II 基底和解析特征值，替换 PO-JEPA 的自适应图谱。

FourierSpectrum 继承 GraphSpectrum，不加参数、不加 buffer（q、sigma_x、sigma_z 照常构造和保存，
构造时消耗的随机数相同），只重写 spectrum_for：基底和特征值只由网格和 sigma_x 决定，
source 只用来核形状、batch 和设备，数值不参与计算。
"""

import math

from dataclasses import dataclass

import torch
from torch import Tensor

from .dynamics import DynamicsConfig, IntegratorConfig, ModalDynamics
from .geometry import GridSpec, regular_grid
from .model import POJEPA, ProjectionConfig
from .spectral import GraphConfig, GraphSpectrum, SpectralDiagnostics, SpectralResult


@dataclass(frozen=True)
class FourierConfig(GraphConfig):
    """GraphConfig 只多一个字段 basis（只接受 "dct2"）；training._model_config 的 graph 快照因此和 full 不同。"""

    basis: str = "dct2"

    def __post_init__(self) -> None:
        """先做 GraphConfig 的全部检查，再要求 basis == "dct2"。"""
        super().__post_init__()
        if self.basis != "dct2":
            raise ValueError("FourierConfig.basis must be 'dct2'")


def mode_order(count: int) -> tuple:
    """08 §4.3：按 (p²+q², max(p,q), p) 升序的前 count 个 (p, q)，与网格无关。

    p 是 x 方向频率，q 是 y 方向频率。前 count 个模态的 p、q 都小于 count（(0,0)…(0,count−1)
    已有 count 个，p²+q² 都小于 count²），所以只需在 [0,count)² 里排序。
    """
    if type(count) is not int or count <= 0:
        raise ValueError("mode count must be a positive integer")
    candidates = [(p, q) for p in range(count) for q in range(count)]
    return tuple(sorted(candidates, key=lambda mode: (mode[0]**2+mode[1]**2, max(mode), mode[0]))[:count])


def dct_basis(height: int, width: int, modes) -> Tensor:
    """08 §4.3：u_pq(x,y) = c_p^W c_q^H cos(πp(x+.5)/W) cos(πq(y+.5)/H)，token n = y·W + x。

    c_0^N = √(1/N)，c_p^N = √(2/N)（p > 0）。返回 CPU float64 [H·W, len(modes)]，各列在 H·W 个 token 上正交归一。
    """
    if any(type(v) is not int or v <= 0 for v in (height, width)):
        raise ValueError("grid height/width must be positive integers")
    modes = tuple(modes)
    if not modes or any(type(m) is not tuple or len(m) != 2 or any(type(v) is not int or v < 0 for v in m) for m in modes):
        raise ValueError("modes must be a nonempty sequence of (p, q) nonnegative integer pairs")
    if any(p >= width or q >= height for p, q in modes):
        raise ValueError("grid does not resolve the requested Fourier modes")
    y, x = torch.meshgrid(torch.arange(height, dtype=torch.float64, device="cpu"),
                          torch.arange(width, dtype=torch.float64, device="cpu"), indexing="ij")
    x, y = x.reshape(-1), y.reshape(-1)
    columns = []
    for p, q in modes:
        cp = math.sqrt((1. if p == 0 else 2.)/width)
        cq = math.sqrt((1. if q == 0 else 2.)/height)
        columns.append(cp*cq*torch.cos(math.pi*p*(x+.5)/width)*torch.cos(math.pi*q*(y+.5)/height))
    return torch.stack(columns, 1)


def fourier_eigenvalues(modes, sigma_x_squared) -> Tensor:
    """08 §4.4：λ_pq = 1 − exp(−σ_x² π² (p²+q²) / 4)，CPU float64 [len(modes)]；与网格大小无关。"""
    s = torch.tensor([p*p+q*q for p, q in modes], dtype=torch.float64, device="cpu")
    return 1-torch.exp(-sigma_x_squared*math.pi**2*s/4)


class FourierSpectrum(GraphSpectrum):
    """08 §4.2：固定 DCT-II 基底的谱模块；不加参数和 buffer，只重写 spectrum_for。"""

    def __init__(self, config: FourierConfig) -> None:
        """要求 FourierConfig，其余和 GraphSpectrum 完全相同（q 用独立的 CPU generator，不消耗全局随机数）。"""
        if not isinstance(config, FourierConfig):
            raise TypeError("config must be a FourierConfig")
        super().__init__(config)

    def spectrum_for(self, source: Tensor, grid: GridSpec, *, k: int,
                     model_device: torch.device) -> SpectralResult:
        """08 §4.5：返回和 GraphSpectrum.lowest_eigenpairs 同类型、同形状、同精度的 SpectralResult。

        source 只核形状、batch、设备和有限值（和 affinity 的输入检查相同）；grid 必须是规则格点中心；
        前 min(k+1, N) 个模态在这个网格上都要能表示（第 k+1 个用于 cutoff_gap）。
        """
        if not isinstance(source, Tensor) or not isinstance(grid, GridSpec):
            raise TypeError("x must be a tensor and grid must be a GridSpec")
        grid.__post_init__()
        width = self.config.feature_dim
        if source.ndim != 3 or source.shape[0] < 1 or source.shape[1] != grid.height*grid.width or source.shape[2] != width:
            raise ValueError(f"x must have shape [positive B, grid height*width, {width}]")
        if source.dtype != torch.float32 or source.device.type == "meta" or not torch.isfinite(source).all():
            raise ValueError("x must be a finite float32 tensor on a real model device")
        _, sx2, _ = self._working_buffers()
        nodes = grid.height*grid.width
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
        if not torch.allclose(grid.coords, regular_grid(grid.height, grid.width).coords, atol=1e-12, rtol=0.):
            raise ValueError("fixed Fourier basis requires regular cell-center coordinates")
        modes = mode_order(min(k+1, nodes))
        if any(p >= grid.width or q >= grid.height for p, q in modes):
            raise ValueError("grid does not resolve the first k+1 Fourier modes")
        batch = source.shape[0]
        with torch.no_grad():
            values = fourier_eigenvalues(modes, sx2)
            basis_cpu = dct_basis(grid.height, grid.width, modes[:k])[None].repeat(batch, 1, 1)
            eigenvalues_cpu = values[:k][None].repeat(batch, 1)
            residuals = torch.zeros(batch, k, dtype=torch.float64, device="cpu")
            gram_error = basis_cpu.transpose(-1, -2) @ basis_cpu - torch.eye(k, device="cpu", dtype=torch.float64)[None]
            orthogonality = torch.linalg.matrix_norm(gram_error, ord="fro", dim=(-2, -1))
            gaps = eigenvalues_cpu[:, 1:]-eigenvalues_cpu[:, :-1]
            cutoff = (values[k]-values[k-1]).repeat(batch) if k < nodes else None
            pivots = basis_cpu.abs().argmax(dim=1)
            diagnostics = SpectralDiagnostics(residuals, orthogonality, gaps, cutoff, pivots)
            if any(not torch.isfinite(t).all() for t in (basis_cpu, eigenvalues_cpu, orthogonality, gaps)) or (cutoff is not None and not torch.isfinite(cutoff).all()):
                raise ValueError("spectral diagnostics must contain finite values")
            return SpectralResult(eigenvalues_cpu, basis_cpu,
                                  eigenvalues_cpu.to(device=model_device, dtype=torch.float32).detach(),
                                  basis_cpu.to(device=model_device, dtype=torch.float32).detach(), diagnostics)


def build_fixed_fourier(model_config: dict, seed: int, device: torch.device) -> POJEPA:
    """08 §1.2、§4.2：和 checks.build_full_model 相同的顺序和种子，只把 GraphSpectrum 换成 FourierSpectrum。

    model_config 是含 projection/graph/dynamics/integrator 四块的模型配置（仿真传 load_protocol()["model"]，
    DROID 传 09 §2.1 的配置块）；seed 是初始化种子（仿真传 checks.model_seed()）。构造在 fork_rng 里做，不改全局随机状态。
    """
    if type(seed) is not int:
        raise TypeError("seed must be an int")
    m = model_config
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = POJEPA(ProjectionConfig(**m["projection"]), FourierSpectrum(FourierConfig(**m["graph"], basis="dct2")),
            ModalDynamics(DynamicsConfig(**m["dynamics"])), IntegratorConfig(**m["integrator"]))
    return model.to(device)
