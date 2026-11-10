"""S25e：冻结 V-JEPA 2 按原生分辨率编码空间采样测试集的画面（resolution = 192/256/320/384 → 12/16/20/24 网格）。

v7 的编码（vjepa.preprocess_rgb + vjepa.FrozenFrameEncoder）写死 256×256 输入、16×16 网格。空间采样（S24c）的画面是
原生渲染的 R×R（R = resolution，视野不变），这里按原生分辨率送进同一个冻结编码器，不缩放回 256：

- 预处理 `preprocess_native`：和 vjepa.preprocess_rgb 逐行相同（同一个官方 random_resized_crop、同一组 scale/ratio、
  同一个均值方差、每图一个子种子），只把输出边长 256 换成 R。scale = 1.777 > 1 时官方的十次尝试都不成立
  （目标面积大于整图），回退到“整张图”（宽高比 1 在 [0.75, 1.35] 里），再把 R×R 双线性插值到 R×R（align_corners=False，
  源坐标正好落在像素中心上，是恒等）。所以对 R×R 的画面它就是逐像素的 (x − 255·mean)/(255·std)；R = 256 时与
  vjepa.preprocess_rgb 逐位相同。
- 编码 `NativeFrameEncoder`：同一个已加载的编码器（冻结、eval、设备、dtype 的设置和检查原样继承 FrozenFrameEncoder），
  单图复制两份凑 tubelet，输入 [B,3,R,R]，输出 raw [B,(R/16)²,1408]，token 行优先（同 v7）。上游 VisionTransformer
  （提交 204698b）默认 handle_nonsquare_inputs=True：forward 按输入算 H_patches = W_patches = R/16 交给每层的
  RoPEAttention.separate_positions，位置就是整数 0…R/16 − 1（不像 AC 预测器那样缩放到 16 格）；patch 嵌入是步长 16 的
  Conv3d，与分辨率无关。所以 192 的位置是 256 的子集，320、384 的位置超出 256 训练时见过的 0…15（外推）。
  R = 256 时与 FrozenFrameEncoder 逐位相同（测试核）。

只用于空间采样测试集的编码（tools/encode_v7.py，--splits test）；训练、v7 缓存和 cache.py 都不经过这里。
"""

import random

import numpy as np
import torch
from torch import Tensor

from . import vjepa
from .geometry import regular_grid
from .variants import RESOLUTIONS

PATCH = 16
ENCODER_DIM = 1408


def grid_side(resolution: int) -> int:
    """输出边长 R → 特征网格边长 R/16（12/16/20/24）。"""
    if type(resolution) is not int or resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}")
    return resolution//PATCH


def preprocess_native(rgb: Tensor, *, resolution: int, spec: vjepa.PreprocessSpec, generator: torch.Generator) -> Tensor:
    """CPU RGB [B,R,R,3] → CPU float32 [B,3,R,R]。vjepa.preprocess_rgb 的原生分辨率版（只把 spec.crop_size 换成 R）。"""
    grid_side(resolution)
    if not isinstance(spec, vjepa.PreprocessSpec):
        raise TypeError("spec must be PreprocessSpec")
    spec.__post_init__()
    if (not isinstance(rgb, Tensor) or rgb.device.type != "cpu" or rgb.dtype != spec.input_dtype
            or rgb.ndim != 4 or rgb.shape[0] <= 0 or tuple(rgb.shape[1:]) != (resolution, resolution, 3)):
        raise ValueError("RGB must match explicit dtype and nonempty CPU [B,R,R,3] at the native resolution")
    if not torch.isfinite(rgb).all() or torch.any((rgb < spec.input_range[0]) | (rgb > spec.input_range[1])):
        raise ValueError("RGB must be finite and within the explicitly declared pixel range")
    if (not isinstance(generator, torch.Generator) or generator.device.type != "cpu"
            or generator is torch.random.default_generator):
        raise ValueError("supply a separate explicit CPU generator")
    official = vjepa._bound_video_transforms()
    mean = 255*torch.tensor(spec.mean, dtype=torch.float32, device="cpu")
    std = 255*torch.tensor(spec.std, dtype=torch.float32, device="cpu")
    frames = []
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.no_grad():
            for image in rgb:
                seed = int(torch.randint(0, 2**32, (), generator=generator, dtype=torch.int64, device="cpu"))
                random.seed(seed)
                np.random.seed(seed)
                frame = image.to(torch.float32).clone().permute(2, 0, 1).unsqueeze(1)
                cropped = official.random_resized_crop(frame, resolution, resolution,
                                                        scale=spec.crop_scale, ratio=spec.crop_ratio)
                frames.append(((cropped[:, 0]-mean[:, None, None])/std[:, None, None]))
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return torch.stack(frames)


class NativeFrameEncoder(vjepa.FrozenFrameEncoder):
    """同一个冻结编码器按原生分辨率 R 编码单图（复制两份凑 tubelet）：[B,3,R,R] → raw [B,(R/16)²,1408]。

    __init__ 复用 FrozenFrameEncoder（冻结、eval、设备、float32），只多记 resolution 并把 grid 换成 (R/16)×(R/16)；
    encode_start / encode_target 原样继承（它们调用这里的 _encode，返回的 grid 是 self.grid）。
    """

    def __init__(self, loaded, *, preprocess, encoder_device, autocast_dtype, resolution: int):
        side = grid_side(resolution)
        super().__init__(loaded, preprocess=preprocess, encoder_device=encoder_device, autocast_dtype=autocast_dtype)
        self.resolution = resolution
        self.grid = regular_grid(side, side)

    def _encode(self, frame: Tensor, time_seconds: Tensor) -> Tensor:
        r, tokens = self.resolution, (self.resolution//PATCH)**2
        if any(m.training for m in self.encoder.modules()):
            raise ValueError("encoder must remain in eval mode")
        if any(p.requires_grad for p in self.encoder.parameters()):
            raise ValueError("encoder parameters must remain frozen")
        if any(p.dtype != torch.float32 or p.device != self.encoder_device for p in self.encoder.parameters()):
            raise ValueError("encoder parameter device/dtype was changed")
        if (not isinstance(frame, Tensor) or frame.device.type != "cpu" or frame.dtype != torch.float32
                or frame.ndim != 4 or frame.shape[0] <= 0 or tuple(frame.shape[1:]) != (3, r, r)
                or not torch.isfinite(frame).all()):
            raise ValueError("preprocessed frame must be finite CPU float32 [B,3,R,R] at the encoder's native resolution")
        if (not isinstance(time_seconds, Tensor) or time_seconds.device.type != "cpu"
                or time_seconds.dtype != torch.float64 or time_seconds.shape != (frame.shape[0],)
                or time_seconds.requires_grad or not torch.isfinite(time_seconds).all()):
            raise ValueError("time must be finite CPU float64 [B] physical seconds without gradient")
        with torch.inference_mode(False), torch.no_grad(), torch.autocast(self.encoder_device.type, enabled=False):
            video = frame.to(self.encoder_device).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            raw_z = self.encoder(video)
        if (not isinstance(raw_z, Tensor) or raw_z.shape != (frame.shape[0], tokens, ENCODER_DIM)
                or not raw_z.is_floating_point() or not torch.isfinite(raw_z).all()):
            raise ValueError("encoder must return one finite raw spatial block [B,(R/16)^2,1408]")
        return raw_z


def fake_encoder():
    """仅供 CPU 流程测试：确定性随机 16×16 patch 投影 + LayerNorm，[L,3,R,R] → [L,(R/16)²,1408]。
    权重与 tools/encode_v7.py 的 _fake_encoder 相同（R = 256 时逐位相同）。"""
    g = torch.Generator().manual_seed(0)
    weight = torch.randn(3*PATCH*PATCH, ENCODER_DIM, generator=g)/16.

    def encode(x):
        side = grid_side(int(x.shape[-1]))
        if tuple(x.shape[1:]) != (3, side*PATCH, side*PATCH):
            raise ValueError("fake encoder expects square [L,3,R,R]")
        patches = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)          # L,3,G,G,16,16
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(x.shape[0], side*side, -1)
        return torch.nn.functional.layer_norm(patches@weight, (ENCODER_DIM,))
    return encode
