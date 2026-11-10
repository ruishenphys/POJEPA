"""求谱后端（docs/design/09-droid-training.md §6.3，S08b）。

`GraphSpectrum.lowest_eigenpairs` 里那一次 `torch.linalg.eigh(L, UPLO="L")` 按 `GraphSpectrum.eig_backend` 分发到这里；
前后的检查、符号 pivot、残差、正交误差、间隙都不变，输入输出都是 CPU float64。

| 后端 | 做法 |
|---|---|
| `cpu_batched` | v7：整批一次 `torch.linalg.eigh(L, UPLO="L")`（默认，走的就是 v7 原来那一行） |
| `cpu_threads` | 按样本切成 `threads` 份，线程池里各自 `eigh`，按原顺序拼回 |
| `cuda_float64` | L 转到模型所在的 GPU（仍 float64）`eigh`，结果拷回 CPU float64 |

后端只影响速度和舍入，不是配置字段、不进 `_model_config` 快照；DROID 的条件工厂从 `model.spectral_backend` 设到
`model.spectrum.eig_backend` 并写进登记（S09）。哪个后端算“一致”由 `check/env` 按 `agreement` 判（09 §6.3）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os

import torch
from torch import Tensor

BACKENDS = ("cpu_batched", "cpu_threads", "cuda_float64")
DEFAULT = "cpu_batched"
EIGENVALUE_ATOL = 1e-9        # 09 §6.3：特征值最大绝对差
PROJECTOR_ATOL = 1e-6         # 前 k 个特征向量张成的投影矩阵之差的 Frobenius 范数
GAP_FLOOR = 1e-6              # 只在截断间隙 ≥ 此值的样本上比投影


def check_backend(name) -> str:
    if name not in BACKENDS:
        raise ValueError(f"eig_backend must be one of {BACKENDS}, got {name!r}")
    return name


def eigh(matrix: Tensor, backend: str = DEFAULT, *, threads: int | None = None,
         device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    """CPU float64 [B,N,N] → (特征值 [B,N]，特征向量 [B,N,N])，CPU float64，升序，UPLO='L'。"""
    check_backend(backend)
    if not isinstance(matrix, Tensor) or matrix.device.type != "cpu" or matrix.dtype != torch.float64 or matrix.ndim != 3:
        raise ValueError("eigh backends take a CPU float64 [B,N,N] tensor")
    if backend == "cpu_batched":
        return torch.linalg.eigh(matrix, UPLO="L")
    if backend == "cpu_threads":
        n = (os.cpu_count() or 1) if threads is None else threads
        if type(n) is not int or n < 1:
            raise ValueError("threads must be a positive int")
        parts = [p for p in torch.tensor_split(matrix, min(n, matrix.shape[0])) if p.shape[0]]
        if len(parts) == 1:
            return torch.linalg.eigh(matrix, UPLO="L")
        with ThreadPoolExecutor(max_workers=len(parts)) as pool:
            results = list(pool.map(lambda p: torch.linalg.eigh(p, UPLO="L"), parts))
        return torch.cat([v for v, _ in results]), torch.cat([w for _, w in results])
    if device is None or device.type != "cuda":
        raise ValueError("cuda_float64 needs the model on a CUDA device")
    values, vectors = torch.linalg.eigh(matrix.to(device), UPLO="L")
    return values.to("cpu", torch.float64), vectors.to("cpu", torch.float64)


def agreement(reference: tuple[Tensor, Tensor], other: tuple[Tensor, Tensor], k: int) -> dict:
    """09 §6.3 的一致标准：特征值最大绝对差 ≤ 1e-9；截断间隙 λ_k − λ_{k−1} ≥ 1e-6 的样本上，前 k 个特征向量
    张成的投影矩阵 VVᵀ 之差的 Frobenius 范数 ≤ 1e-6。`reference` 一般是 cpu_batched 的结果。"""
    (v0, w0), (v1, w1) = reference, other
    if v0.shape != v1.shape or w0.shape != w1.shape or not 1 <= k < v0.shape[1]:
        raise ValueError("eigenpairs must match in shape and 1 <= k < N")
    value_diff = float((v0-v1).abs().max())
    gap = v0[:, k]-v0[:, k-1]
    use = gap >= GAP_FLOOR
    p0 = w0[:, :, :k] @ w0[:, :, :k].transpose(-1, -2)
    p1 = w1[:, :, :k] @ w1[:, :, :k].transpose(-1, -2)
    proj = torch.linalg.matrix_norm(p0-p1, ord="fro", dim=(-2, -1))
    proj_max = float(proj[use].max()) if bool(use.any()) else 0.
    return dict(max_eigenvalue_diff=value_diff, max_projector_diff=proj_max, compared_samples=int(use.sum()),
                samples=int(v0.shape[0]), agree=value_diff <= EIGENVALUE_ATOL and proj_max <= PROJECTOR_ATOL)
