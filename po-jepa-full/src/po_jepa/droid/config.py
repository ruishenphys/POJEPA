"""DROID 三层配置（docs/design/01-droid-config.md §1–§3、§6）。

`load_config(path)` 读同目录的 `base.yaml`，用 `path` 这份 overlay 按 v7 `checks._merge_v4` 的规则深度合并，
再构造冻结的 `DroidConfig` 并校验。overlay 只能改 01 §2 的键；合并后缺键、多键、类型不对、§3 的任一校验
不满足都报错，不补默认值。`config_sha256` 是合并后配置的 `cache._json_bytes` 的 SHA-256。
"""

from __future__ import annotations

import hashlib
import math
import re
import typing
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from po_jepa.losses import LossConfig
from po_jepa.runtime import cache, checks

SCHEMA = "droid_config_v1"
BASE_FILE = "base.yaml"
BASE_NAME = "base"
DERIVED_KEY = "config_sha256"   # DroidConfig 里唯一不来自配置文件的字段
# 01 §2：overlay 能改的键（点号路径）；resources 下的任何键都能改。
OVERLAY_KEYS = frozenset({"name", "data.train_episodes", "training.updates", "training.seeds",
                          "model.spectral_backend"})
OVERLAY_PREFIX = "resources."
OOD_TEST_EPISODES = 500          # 00 §3.1 第 5 项：留出实验室测试集 500 条
TRAINING_DURATIONS = (1, 2, 4, 8)  # v7 的 WindowQuery 和时间一致性的半程都依赖它
CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "fixed_fourier", "koopman",
              "matched_transformer", "transformer_consistency", "vjepa2_ac", "vjepa2_ac_small")  # 08 §1.2，顺序固定
ACTION_SOURCES = ("realized", "commanded")
AC_STATES = ("true_state", "learned_constant")
SPECTRAL_BACKENDS = ("cpu_batched", "cpu_threads", "cuda_float64")  # 09 §6.3
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")   # name 是运行根下的子目录名
# v7 协议里 loss_config 照搬的五项（01 §6）
V7_LOSS_ITEMS = ("jepa_exponent", "layer_norm_epsilon", "time_epsilon", "space_epsilon", "consistency_metric")


@dataclass(frozen=True)
class DataConfig:
    version: str
    list_api: str
    object_base: str
    prefix: str
    expected_metadata_files: int
    camera: str
    min_steps: int
    ood_labs: tuple[str, ...]
    ood_quota: tuple[int, ...]
    id_test_scenes: int
    val_scenes: int
    episodes_per_scene: int
    train_episodes: int
    shard_size: int
    split_salt: str
    window_salt: str
    use_failures: bool


@dataclass(frozen=True)
class TimeConfig:
    frames_per_interval: int
    nominal_interval_seconds: float
    min_interval_seconds: float
    max_interval_seconds: float


@dataclass(frozen=True)
class ActionConfig:
    source: str
    scale_episodes: int
    scale_quantile: float
    position_bound_m: float
    rotation_bound_rad: float
    gripper_bound: float


@dataclass(frozen=True)
class EncodeConfig:
    upstream_commit: str
    checkpoint_file: str
    batch_frames: int
    storage_dtype: str
    test_extra_episodes: int
    frame_rate_stride_frames: int
    extra_resolutions: tuple[int, ...]


@dataclass(frozen=True)
class ModelConfig:
    grid: int
    consistency_grid: int
    test_grid: int
    modes: int
    sigma_x: float
    rho: float
    spectral_backend: str
    ac_state: str


@dataclass(frozen=True)
class TrainingConfig:
    durations: tuple[int, ...]
    batch: int
    ac_batch_clips: int
    ac_clip_frames: int
    updates: int
    lr_min: float
    grad_clip: float
    checkpoint_minutes: int
    stop_margin_minutes: int
    log_every: int
    seeds: tuple[int, ...]
    conditions: tuple[str, ...]
    untrained: tuple[str, ...]


@dataclass(frozen=True)
class MatchingConfig:
    tolerance: float
    transformer_feedforward_candidates: tuple[int, ...]
    ac_small_mlp_ratio_candidates: tuple[float, ...]


@dataclass(frozen=True)
class MonitoringConfig:
    min_share: float


@dataclass(frozen=True)
class CalibrationConfig:
    seed: int
    pretrain_updates: int
    median_batches: int
    sigma_z_episodes: int
    sigma_z_rows: tuple[int, ...]


@dataclass(frozen=True)
class CemConfig:
    iterations: int
    samples: int
    elites: int
    init_std: float
    clip: float


@dataclass(frozen=True)
class EvaluationConfig:
    durations_id: tuple[int, ...]
    durations_heldout: tuple[int, ...]
    durations_long: tuple[int, ...]
    windows_per_duration: int
    composed_step: int
    frame_rate_single: tuple[int, ...]
    frame_rate_rollout: tuple[int, ...]
    frame_rate_total_frames: int
    planning_durations: tuple[int, ...]
    planning_candidates: int
    cem: CemConfig


@dataclass(frozen=True)
class StageResources:
    gpus: int
    cpus: int
    mem_gb: int
    time_limit_minutes: int


@dataclass(frozen=True)
class ResourcesConfig:
    index: StageResources
    download: StageResources
    prepare: StageResources
    stats: StageResources
    encode: StageResources
    check: StageResources
    calibrate: StageResources
    train: StageResources
    evaluate: StageResources
    summary: StageResources


@dataclass(frozen=True)
class DroidConfig:
    """01 §3 的全部键（嵌套冻结 dataclass），外加合并后配置的 `config_sha256`。"""

    schema: str
    name: str
    data: DataConfig
    time: TimeConfig
    action: ActionConfig
    encode: EncodeConfig
    model: ModelConfig
    training: TrainingConfig
    matching: MatchingConfig
    monitoring: MonitoringConfig
    calibration: CalibrationConfig
    evaluation: EvaluationConfig
    resources: ResourcesConfig
    config_sha256: str

    def __post_init__(self) -> None:
        _validate(self)

    # 01 §2：由 training.updates 推出的量。U 是 400 的倍数（_validate），下面的乘法都是精确整数，等于 round(·)。
    @property
    def lr_warmup_updates(self) -> int:
        return self.training.updates * 5 // 100

    @property
    def consistency_warmup_updates(self) -> int:
        return self.training.updates * 10 // 100

    @property
    def validation_interval(self) -> int:
        return self.training.updates // 20

    @property
    def early_check_updates(self) -> int:
        return self.consistency_warmup_updates + self.training.updates * 4 // 100

    @property
    def consistency_interval(self) -> int:
        return self.training.updates // 400

    def loss_config(self, *, time_weight: float, space_weight: float) -> LossConfig:
        """v7 的 LossConfig（01 §6）：五项取 v7 协议，调度取本配置，两个权重由调用者给（09 §3）。"""
        protocol = v7_loss_protocol()
        return LossConfig(**{k: protocol[k] for k in V7_LOSS_ITEMS},
                          time_weight=time_weight, space_weight=space_weight,
                          consistency_warmup_updates=self.consistency_warmup_updates,
                          grad_clip_norm=self.training.grad_clip, lr_min=self.training.lr_min,
                          lr_decay_updates=self.training.updates, lr_warmup_updates=self.lr_warmup_updates)


def v7_loss_protocol() -> dict:
    """`checks.load_protocol()` 在 v7 下 `loss` 块里 `loss_config` 要的五项（V7_LOSS_ITEMS）。

    不读、不改进程环境变量（Codex S01 #2：原来临时改 os.environ，别的线程会看到）。照 `load_protocol` 的前两步
    合并 default.yaml 和 v4.yaml；v7 在其上合并的 schedules、calibration、model 不碰这五项，这里逐个核对，
    以后谁在 v7.yaml 里改了这五项就报错，不会悄悄读到旧值。和 `load_protocol()` 的一致性由测试核。
    """
    cache.verify_fixed_protocol()
    root = checks.ROOT/"configs"/"train"
    base = _read_mapping(root/"default.yaml")
    overlay = _read_mapping(root/"v4.yaml")
    merged = checks._merge_v4(base, {k: v for k, v in overlay.items() if k not in ("schema_version", "protocol_id")})
    v7 = _read_mapping(root/"v7.yaml")
    later = [*v7["schedules"].values(), {"calibration": v7["calibration"]}, {"model": v7.get("model", {})}]
    touched = sorted({k for block in later for k in (block.get("loss") or {}) if k in V7_LOSS_ITEMS})
    if touched:
        raise ValueError(f"configs/train/v7.yaml overrides loss items {touched}; v7_loss_protocol must read them too")
    return {k: merged["loss"][k] for k in V7_LOSS_ITEMS}


def load_config(path: Path | str) -> DroidConfig:
    """01 §1：同目录 base.yaml + overlay（path 本身是 base.yaml 时 overlay 为空，name 仍是 base 会报错）。"""
    path = Path(path)
    base_path = path.parent/BASE_FILE
    base = _read_mapping(base_path)
    overlay = {} if path.resolve() == base_path.resolve() else _read_mapping(path)
    _check_overlay(base, overlay, "")
    merged = checks._merge_v4(base, overlay)
    if DERIVED_KEY in merged:   # 由本函数算出的字段不能出现在配置文件里，否则会被覆盖而不报多键（Codex S01 #1）
        raise ValueError(f"config has unknown keys: {DERIVED_KEY} (it is computed, not read from the files)")
    _build(DroidConfig, dict(merged, config_sha256="pending"), "")   # 先按类型和 §3 校验，报错比 JSON 编码的报错更清楚
    sha = hashlib.sha256(cache._json_bytes(merged)).hexdigest()
    return _build(DroidConfig, dict(merged, config_sha256=sha), "")


def _read_mapping(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: config file must be a YAML mapping")
    return value


def _check_overlay(base: dict, overlay: dict, prefix: str) -> None:
    for key, value in overlay.items():
        full = prefix + str(key)
        if not isinstance(base, dict) or key not in base:
            raise ValueError(f"overlay key {full!r} is not in base.yaml")
        if full in OVERLAY_KEYS or full.startswith(OVERLAY_PREFIX):
            if isinstance(value, dict):
                _check_overlay(base[key], value, full + ".")
            continue
        if isinstance(value, dict) and isinstance(base[key], dict):
            _check_overlay(base[key], value, full + ".")
            continue
        raise ValueError(f"overlay key {full!r} may not be overridden (01 §2)")


def _build(cls, raw, where: str):
    """按 dataclass 的注解逐键构造：缺键、多键、类型不对都报错；列表转成 tuple。"""
    if not isinstance(raw, dict):
        raise TypeError(f"config section {where.rstrip('.') or '<root>'!r} must be a mapping")
    names = [f.name for f in fields(cls)]
    missing = [n for n in names if n not in raw]
    if missing:
        raise ValueError(f"config is missing keys: {', '.join(where + n for n in missing)}")
    extra = sorted(str(k) for k in raw if k not in names)
    if extra:
        raise ValueError(f"config has unknown keys: {', '.join(where + k for k in extra)}")
    hints = typing.get_type_hints(cls)
    return cls(**{n: _value(hints[n], raw[n], where + n) for n in names})


def _value(kind, value, where: str):
    if isinstance(kind, type) and hasattr(kind, "__dataclass_fields__"):
        return _build(kind, value, where + ".")
    if typing.get_origin(kind) is tuple:
        item = typing.get_args(kind)[0]
        if not isinstance(value, list):
            raise TypeError(f"{where} must be a list")
        return tuple(_value(item, v, f"{where}[{i}]") for i, v in enumerate(value))
    if kind is bool:
        if type(value) is not bool:
            raise TypeError(f"{where} must be true or false")
        return value
    if kind is int:
        if type(value) is not int:
            raise TypeError(f"{where} must be an integer")
        return value
    if kind is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise TypeError(f"{where} must be a finite real number")
        return float(value)
    if kind is str:
        if type(value) is not str or not value.strip():
            raise TypeError(f"{where} must be a non-empty string")
        return value
    raise AssertionError(f"unsupported config type for {where}: {kind!r}")   # pragma: no cover


def _validate(c: DroidConfig) -> None:
    """01 §2、§3 的校验；任一不满足就报错。"""
    if c.schema != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}")
    if c.name == BASE_NAME:
        raise ValueError("name is still 'base': run with an overlay (pilot.yaml/full.yaml) that sets its own name")
    if not NAME_PATTERN.fullmatch(c.name):
        raise ValueError("name must be a plain directory name (letters, digits, '_', '.', '-')")
    d = c.data
    if len(d.ood_quota) != len(d.ood_labs) or sum(d.ood_quota) != OOD_TEST_EPISODES:
        raise ValueError(f"data.ood_quota must match data.ood_labs one to one and sum to {OOD_TEST_EPISODES}")
    for key in ("id_test_scenes", "val_scenes", "episodes_per_scene", "train_episodes", "shard_size"):
        if getattr(d, key) <= 0:
            raise ValueError(f"data.{key} must be a positive integer")
    # 两个 salt 非空：由 _value 对所有字符串的“非空”检查保证
    t = c.training
    if t.durations != TRAINING_DURATIONS:
        raise ValueError("training.durations must be [1, 2, 4, 8]")
    if t.conditions != CONDITIONS:
        raise ValueError(f"training.conditions must be exactly {list(CONDITIONS)} in this order")
    if not t.seeds or len(set(t.seeds)) != len(t.seeds):
        raise ValueError("training.seeds must be non-empty and distinct")
    if t.updates <= 0:
        raise ValueError("training.updates must be a positive integer")
    if t.updates % 20:
        raise ValueError("training.updates must be divisible by 20 (validation interval U/20)")
    if t.updates % 400:
        raise ValueError("training.updates must be divisible by 400 (consistency interval U/400)")
    if c.action.source not in ACTION_SOURCES:
        raise ValueError(f"action.source must be one of {ACTION_SOURCES}")
    if c.model.ac_state not in AC_STATES:
        raise ValueError(f"model.ac_state must be one of {AC_STATES}")
    if c.model.spectral_backend not in SPECTRAL_BACKENDS:
        raise ValueError(f"model.spectral_backend must be one of {SPECTRAL_BACKENDS}")
    e, stride = c.evaluation, c.encode.frame_rate_stride_frames
    if stride <= 0 or any(f <= 0 or f % stride for f in (*e.frame_rate_single, *e.frame_rate_rollout)):
        raise ValueError("evaluation.frame_rate_single and frame_rate_rollout must be positive multiples of "
                         "encode.frame_rate_stride_frames")
    if any(e.frame_rate_total_frames % f for f in e.frame_rate_rollout):
        raise ValueError("evaluation.frame_rate_total_frames must be a multiple of every frame_rate_rollout entry")

