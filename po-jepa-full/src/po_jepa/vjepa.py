"""本地官方编码器的构造、加载与冻结单图编码。

显式固定源码与文件来源；导入不读取资源。随机架构、合成加载测试
和真实预训练加载证据分别记录。
"""

from dataclasses import dataclass
import hashlib
import importlib
from importlib.machinery import ModuleSpec
import importlib.util
from pathlib import Path
import re
import random
import subprocess
import sys
from typing import Literal

import torch
import numpy as np
from torch import nn, Tensor

from .geometry import GridSpec, regular_grid


UPSTREAM_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"


@dataclass(frozen=True)
class EncoderArchitectureSpec:
    """显式编码器架构；容量帧数不等于预测时可见历史。"""

    upstream_commit: str
    config_source: str
    model_name: str
    crop_size: int
    patch_size: int
    tubelet_size: int
    max_num_frames: int
    use_rope: bool
    use_sdpa: bool
    use_silu: bool
    wide_silu: bool
    uniform_power: bool
    use_activation_checkpointing: bool


@dataclass(frozen=True)
class LocalCheckpointSpec:
    """一个已明确来源的本地文件及事先审阅的键规则；无备用路径。"""

    path: Path
    expected_sha256: str
    publication_source: str
    encoder_key: Literal["encoder", "target_encoder"]
    strip_prefixes: tuple[str, ...]
    strict: bool
    allowed_missing: tuple[str, ...]
    allowed_unexpected: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointReport:
    """实际加载事实；不把发布链接或合成文件当成真实权重证据。"""

    path: Path
    sha256: str
    encoder_key: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    strict: bool
    architecture: EncoderArchitectureSpec


@dataclass(frozen=True)
class ArchitectureBuild:
    """CPU float32 随机 encoder 与构造凭据；尚未证明预训练已加载。"""

    encoder: nn.Module
    architecture: EncoderArchitectureSpec
    initialization_seed: int
    evidence_role: Literal["random_architecture_only"]


@dataclass(frozen=True)
class LoadedEncoder:
    """同一个已加载编码器及实际报告，冻结/eval/设备转移由包装器完成。"""

    encoder: nn.Module
    report: CheckpointReport


def _check_src_origins(source_root: Path) -> None:
    """拒绝进程内其他同名 src 包，避免官方绝对导入串到别处。"""
    for name, module in tuple(sys.modules.items()):
        if name != "src" and not name.startswith("src."):
            continue
        paths = ([module.__file__] if getattr(module, "__file__", None)
                 else list(getattr(module, "__path__", ())))
        if not paths or any(not Path(p).resolve().is_relative_to(source_root) for p in paths):
            raise ImportError(f"src module has foreign origin: {name}: {paths}")


def _import_official_encoder(upstream_root: Path, commit: str):
    """显式核对本地 pin 与实际源码路径，仅导入官方模型，不调用 Hub/trainer。"""
    root = Path(upstream_root).resolve(strict=True)
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"upstream commit must be {UPSTREAM_COMMIT}, got {commit}")
    actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if actual != commit:
        raise ValueError(f"upstream commit mismatch: expected {commit}, actual {actual}")
    source_root = root / "src"
    _check_src_origins(source_root)
    if "src" not in sys.modules:
        # 上游 src 没有 __init__.py；只绑定显式根目录，避免修改全局 sys.path。
        spec = ModuleSpec("src", loader=None, is_package=True)
        spec.submodule_search_locations = [str(source_root)]
        sys.modules["src"] = importlib.util.module_from_spec(spec)
    official = importlib.import_module("src.models.vision_transformer")
    _check_src_origins(source_root)
    if Path(official.__file__).resolve() != source_root / "models" / "vision_transformer.py":
        raise ImportError(f"unexpected encoder source: {official.__file__}")
    return official


def build_encoder_architecture(*, upstream_root: Path,
                               architecture: EncoderArchitectureSpec,
                               initialization_seed: int) -> ArchitectureBuild:
    """构造真实 CPU float32 编码器架构，尚未加载预训练。

    显式本地源码、固定参数与种子定义 1408宽/40层/22头。临时 CPU RNG
    和默认精度退出后恢复；不调用 torch.manual_seed 或改 CUDA 随机状态。
    """
    expected = EncoderArchitectureSpec(
        UPSTREAM_COMMIT, architecture.config_source, "vit_giant_xformers", 256, 16, 2,
        512, True, True, False, True, True, True,
    )
    if not architecture.config_source.strip() or architecture != expected:
        raise ValueError("architecture differs from fixed A06 encoder specification")
    official = _import_official_encoder(upstream_root, architecture.upstream_commit)
    previous_dtype = torch.get_default_dtype()
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(initialization_seed)
            torch.set_default_dtype(torch.float32)
            encoder = official.vit_giant_xformers(
                img_size=architecture.crop_size, patch_size=architecture.patch_size,
                tubelet_size=architecture.tubelet_size,
                # 官方参数名是 num_frames；max_num_frames 会被 **kwargs 静默吞掉。
                num_frames=architecture.max_num_frames,
                use_rope=architecture.use_rope, use_sdpa=architecture.use_sdpa,
                use_silu=architecture.use_silu, wide_silu=architecture.wide_silu,
                uniform_power=architecture.uniform_power,
                use_activation_checkpointing=architecture.use_activation_checkpointing,
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    if (encoder.num_frames, encoder.img_height, encoder.img_width,
        encoder.patch_size, encoder.tubelet_size) != (512, 256, 256, 16, 2):
        raise ValueError("official encoder did not apply the requested frame/grid capacity")
    if any(t.device.type != "cpu" or (t.is_floating_point() and t.dtype != torch.float32)
           for t in (*encoder.parameters(), *encoder.buffers())):
        raise ValueError("encoder construction must produce CPU float32 parameters/buffers")
    return ArchitectureBuild(encoder, architecture, initialization_seed, "random_architecture_only")


def _checkpoint_identity(checkpoint: LocalCheckpointSpec) -> tuple[Path, str]:
    """在反序列化或分配大架构之前，核实显式普通文件、来源与 SH6。"""
    path = Path(checkpoint.path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"checkpoint must be an explicit local regular file: {path}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint.expected_sha256):
        raise ValueError("expected SHA256 must contain exactly 64 hexadecimal characters")
    if not checkpoint.publication_source.strip():
        raise ValueError("checkpoint publication_source must be explicit and nonempty")
    if checkpoint.encoder_key not in ("encoder", "target_encoder"):
        raise ValueError("encoder_key must be explicitly encoder or target_encoder")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != checkpoint.expected_sha256.lower():
        raise ValueError(f"SHA256 mismatch: expected {checkpoint.expected_sha256}, actual {actual}")
    return path.resolve(), actual


def load_local_encoder(build: ArchitectureBuild, *, checkpoint: LocalCheckpointSpec) -> LoadedEncoder:
    """严格核实一个本地 state_dict 并更新 build.encoder，返回同一对象和报告。

    先验证哈希，再以 CPU/weights_only=True 读取；异常原样向上传递，不换键、
    文件或反序列化方式。按显式顺序各清理一次前缀，冲突/shape 错始终失败。
    strict=False 的实际缺键/多键须精确等于事先允许列表。无冻结、eval、CUDA
    或随机补权重；调用者不得再把旧 build 当成一份独立模型。
    """
    path, sha256 = _checkpoint_identity(checkpoint)
    content = torch.load(path, map_location="cpu", weights_only=True)
    state = content[checkpoint.encoder_key]  # 无备用 key；KeyError 保留实际请求。
    cleaned = {}
    for key, value in state.items():
        for prefix in checkpoint.strip_prefixes:
            if not prefix:
                raise ValueError("strip_prefixes cannot contain an empty prefix")
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in cleaned:
            raise ValueError(f"checkpoint prefix collision: {key}")
        cleaned[key] = value
    expected = build.encoder.state_dict()
    # 先核所有 shape，避免 load_state_dict 失败前已经更新一部分正确参数。
    mismatches = [f"{key}: checkpoint {tuple(value.shape)}, model {tuple(expected[key].shape)}"
                  for key, value in cleaned.items()
                  if key in expected and value.shape != expected[key].shape]
    if mismatches:
        raise RuntimeError("checkpoint shape mismatch: " + "; ".join(mismatches))
    missing = tuple(sorted(expected.keys() - cleaned.keys()))
    unexpected = tuple(sorted(cleaned.keys() - expected.keys()))
    if ((checkpoint.strict and (missing or unexpected))
        or set(missing) != set(checkpoint.allowed_missing)
        or set(unexpected) != set(checkpoint.allowed_unexpected)):
        raise RuntimeError(f"checkpoint key mismatch: missing_keys={missing}, unexpected_keys={unexpected}")
    result = build.encoder.load_state_dict(cleaned, strict=checkpoint.strict)
    report = CheckpointReport(path, sha256, checkpoint.encoder_key,
                              tuple(sorted(result.missing_keys)), tuple(sorted(result.unexpected_keys)),
                              checkpoint.strict, build.architecture)
    return LoadedEncoder(build.encoder, report)


@dataclass(frozen=True)
class PreprocessSpec:
    """声明 CPU RGB [B,H,W,3] 的 dtype、0–255 范围与固定 DROID 变换。

    来源须显式给定，不根据像素值猜尺度；输出归一化后不限于 [0,1]。
    """

    input_dtype: torch.dtype
    input_range: tuple[float, float]
    crop_size: int
    crop_scale: tuple[float, float]
    crop_ratio: tuple[float, float]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    source: str

    def __post_init__(self) -> None:
        """固定变换参数是已登记选择；拒绝静默改成 0–1 或 Hub 预处理。"""
        if not isinstance(self.input_dtype, torch.dtype) or not (
            self.input_dtype == torch.uint8 or self.input_dtype.is_floating_point
        ):
            raise ValueError("input_dtype must describe uint8 or floating RGB pixels")
        if (self.input_range != (0., 255.) or self.crop_size != 256
            or self.crop_scale != (1.777, 1.777) or self.crop_ratio != (.75, 1.35)
            or self.mean != (.485, .456, .406) or self.std != (.229, .224, .225)):
            raise ValueError("preprocess differs from the fixed 0–255 DROID transform")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("preprocess source must be explicit")


@dataclass(frozen=True)
class EncodedStart:
    """预测起点：raw_z [B,N,E] 无梯度，只含当前单图；时间为物理秒。

    observed_time_seconds 为 CPU float64 [B]；grid/checkpoint/preprocess
    解释 token 空间顺序和来源，不携带未来图像、轨迹或隐藏状态。
    """

    raw_z: Tensor
    observed_time_seconds: Tensor
    grid: GridSpec
    checkpoint: CheckpointReport
    preprocess: PreprocessSpec


@dataclass(frozen=True)
class EncodedTarget:
    """未来标签：与起点同样约定的 raw_z，仅供损失/评价标签端。

    target_time_seconds 为 CPU float64 [B] 物理秒；未来图像单独编码，
    不得把本对象送入起点预测分支。encoder 内部 norm 保留，外部 LN
    属于后续 JEPA 距离，不能写回这些 raw token。
    """

    raw_z: Tensor
    target_time_seconds: Tensor
    grid: GridSpec
    checkpoint: CheckpointReport
    preprocess: PreprocessSpec


def adapt_raw_tokens(raw_z: Tensor, grid: GridSpec, *, encoder_dim: int,
                     model_device: torch.device) -> Tensor:
    """图1 raw Z→模型设备 float32，不改变 token 轴/顺序或反馈梯度。

    输入有限实浮点 [B,N,E]，N=grid.height*grid.width、E=encoder_dim；
    生产 [B,256,1408]，粗网格和合成小 E 合法。特征无量纲，时间不转换。
    只执行原生 Tensor.to，不 LN、偏移、投影、重编码或 detach，允许
    非连续视图。no_grad 下同设备 float32 可返回原对象，不保证独立存储。
    转换前后核有限性，防止有限 float64 转 float32 溢出；仅 CPU/可用 CUDA。
    起点可见性、标签来源与 stop-gradient 由编码/缓存编排保证。
    """
    if not isinstance(raw_z, Tensor):
        raise TypeError("raw_z must be a tensor")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    if type(encoder_dim) is not int:
        raise TypeError("encoder_dim must be a native integer")
    if encoder_dim <= 0:
        raise ValueError("encoder_dim must be positive")
    if not isinstance(model_device, torch.device):
        raise TypeError("model_device must be an explicit torch.device")
    grid.__post_init__()
    if raw_z.is_nested or raw_z.layout != torch.strided or raw_z.is_meta:
        raise ValueError("raw_z must be a real dense tensor")
    if torch.is_inference(raw_z):
        raise ValueError("raw_z must be an ordinary tensor, not an inference tensor")
    if raw_z.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("raw_z must use a supported real floating dtype")
    if (raw_z.ndim != 3 or any(n <= 0 for n in raw_z.shape)
        or raw_z.shape[1:] != (grid.height*grid.width, encoder_dim)):
        raise ValueError("raw_z shape must be [positive B, grid.height*grid.width, encoder_dim]")
    if not torch.isfinite(raw_z).all():
        raise ValueError("raw_z must contain finite values")
    if model_device.type not in ("cpu", "cuda"):
        raise ValueError("model_device must be CPU or CUDA")
    actual_device = torch.device("cpu")
    if model_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("model_device CUDA is unavailable")
        index = torch.cuda.current_device() if model_device.index is None else model_device.index
        if index >= torch.cuda.device_count():
            raise ValueError("model_device CUDA index is out of range")
        actual_device = torch.device("cuda", index)
    # 用原请求转换，CPU带索引和无索引CUDA都遵守原生to的实际设备/别名语义。
    adapted = raw_z.to(device=model_device, dtype=torch.float32)
    if adapted.device != actual_device:
        raise ValueError("raw_z conversion did not reach the actual model_device")
    if not torch.isfinite(adapted).all():
        raise ValueError("raw_z conversion must remain finite float32")
    return adapted


def _bound_video_transforms():
    """复用显式绑定的官方 src 裁剪，不猜路径、搜索安装目录或调用 trainer。

    须先绑定指定上游根；实际模块来源不符则拒绝，避免同名外部包混入。
    """
    if "src" not in sys.modules or len(getattr(sys.modules["src"], "__path__", ())) != 1:
        raise RuntimeError("first explicitly bind the official source using the B02 build workflow")
    source_root = Path(next(iter(sys.modules["src"].__path__))).resolve()
    _check_src_origins(source_root)
    module = importlib.import_module("src.datasets.utils.video.transforms")
    _check_src_origins(source_root)
    if Path(module.__file__).resolve() != source_root / "datasets/utils/video/transforms.py":
        raise ImportError("unexpected official video transform source")
    return module


def preprocess_rgb(rgb: Tensor, *, spec: PreprocessSpec, generator: torch.Generator) -> Tensor:
    """CPU RGB [B,H,W,3]→CPU float32 [B,3,256,256]。

    复用官方 random_resized_crop 十次尝试/中心回退及
    (pixel-255*mean)/(255*std)；scale=1.777 通常触发回退，不改增强分布。
    显式 generator 每图抽一个子种子，Python/NumPy 全局状态在 finally
    恢复。本入口串行使用，不提供并发 RNG 隔离，不把 batch 当时间轴。
    """
    if not isinstance(spec, PreprocessSpec):
        raise TypeError("spec must be PreprocessSpec")
    spec.__post_init__()
    if (not isinstance(rgb, Tensor) or rgb.device.type != "cpu" or rgb.dtype != spec.input_dtype
        or rgb.ndim != 4 or rgb.shape[-1] != 3 or any(n <= 0 for n in rgb.shape)):
        raise ValueError("RGB must match explicit dtype and nonempty CPU [B,H,W,3]")
    if not torch.isfinite(rgb).all() or torch.any((rgb < spec.input_range[0]) | (rgb > spec.input_range[1])):
        raise ValueError("RGB must be finite and within the explicitly declared pixel range")
    if (not isinstance(generator, torch.Generator) or generator.device.type != "cpu"
        or generator is torch.random.default_generator):
        raise ValueError("supply a separate explicit CPU generator")
    official = _bound_video_transforms()
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
                # clone 防止官方代码或后续修改原地改变调用者图像。
                frame = image.to(torch.float32).clone().permute(2, 0, 1).unsqueeze(1)
                cropped = official.random_resized_crop(frame, spec.crop_size, spec.crop_size,
                                                        scale=spec.crop_scale, ratio=spec.crop_ratio)
                frames.append(((cropped[:, 0]-mean[:, None, None])/std[:, None, None]))
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return torch.stack(frames)


class FrozenFrameEncoder:
    """同一个已加载编码器冻结编码独立端点，不额外创建模型。

    单图复制两份满足 tubelet=2，不增加运动历史；返回 raw [B,256,1408]。
    仅接当前调用的预处理图像和物理时间，不融合历史或访问未来端点。
    """

    def __init__(self, loaded: LoadedEncoder, *, preprocess: PreprocessSpec,
                 encoder_device: torch.device, autocast_dtype: torch.dtype | None) -> None:
        """显式设置原模型的设备、float32、eval 和冻结；不读取权重或自动启用 AMP。"""
        if not isinstance(loaded, LoadedEncoder):
            raise TypeError("FrozenFrameEncoder requires LoadedEncoder")
        if not isinstance(preprocess, PreprocessSpec):
            raise TypeError("preprocess must be PreprocessSpec")
        preprocess.__post_init__()
        if autocast_dtype is not None:
            raise ValueError("autocast dtype not verified for this adapter; explicitly use None")
        self.encoder = loaded.encoder.to(device=encoder_device, dtype=torch.float32)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.checkpoint, self.preprocess = loaded.report, preprocess
        # .to("cuda") 实际落在 cuda:当前序号；记录实际设备，避免合法别名被误判为外部改动。
        self.encoder_device = next(self.encoder.parameters()).device
        self.grid = regular_grid(16, 16)

    def _encode(self, frame: Tensor, time_seconds: Tensor) -> Tensor:
        """共享端点计算，只从当前调用的单张图生成 raw；异常时不默默修复模式。"""
        if any(m.training for m in self.encoder.modules()):
            raise ValueError("encoder must remain in eval mode")
        if any(p.requires_grad for p in self.encoder.parameters()):
            raise ValueError("encoder parameters must remain frozen")
        if any(p.dtype != torch.float32 or p.device != self.encoder_device for p in self.encoder.parameters()):
            raise ValueError("encoder parameter device/dtype was changed")
        if (not isinstance(frame, Tensor) or frame.device.type != "cpu" or frame.dtype != torch.float32
            or frame.ndim != 4 or frame.shape[0] <= 0 or frame.shape[1:] != (3, 256, 256)
            or not torch.isfinite(frame).all()):
            raise ValueError("preprocessed frame must be finite CPU float32 [B,3,256,256]")
        if (not isinstance(time_seconds, Tensor) or time_seconds.device.type != "cpu"
            or time_seconds.dtype != torch.float64 or time_seconds.shape != (frame.shape[0],)
            or time_seconds.requires_grad or not torch.isfinite(time_seconds).all()):
            raise ValueError("time must be finite CPU float64 [B] physical seconds without gradient")
        # 明确关闭 AMP；即使外层使用 inference_mode，也返回可用于后续投影反向的普通张量。
        with torch.inference_mode(False), torch.no_grad(), torch.autocast(self.encoder_device.type, enabled=False):
            video = frame.to(self.encoder_device).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            raw_z = self.encoder(video)
        if (not isinstance(raw_z, Tensor) or raw_z.shape != (frame.shape[0], 256, 1408)
            or not raw_z.is_floating_point() or not torch.isfinite(raw_z).all()):
            raise ValueError("encoder must return one finite raw spatial block [B,256,1408]")
        return raw_z

    def encode_start(self, preprocessed_frame: Tensor, *, observed_time_seconds: Tensor) -> EncodedStart:
        """只编码当前图；返回起点 raw token、其时间和来源，不接受任何未来端点。"""
        raw_z = self._encode(preprocessed_frame, observed_time_seconds)
        return EncodedStart(raw_z, observed_time_seconds.clone(), self.grid, self.checkpoint, self.preprocess)

    def encode_target(self, preprocessed_frame: Tensor, *, target_time_seconds: Tensor) -> EncodedTarget:
        """单独编码未来标签并明确 detach；不更新起点或隐式历史。"""
        raw_z = self._encode(preprocessed_frame, target_time_seconds).detach()
        return EncodedTarget(raw_z, target_time_seconds.clone(), self.grid, self.checkpoint, self.preprocess)
