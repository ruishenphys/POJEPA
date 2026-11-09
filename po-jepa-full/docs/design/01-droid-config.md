# 01 DROID 配置与 7 维动作

2026-11-08 · 第一版

本文件定 DROID 实验的配置格式、所有固定取值和 7 维动作的单位约定。代码在 `src/po_jepa/droid/config.py` 和 `src/po_jepa/droid/actions.py`（S01）。

## 1. 三层配置

| 文件 | 内容 |
|---|---|
| `configs/droid/base.yaml` | 所有固定选择（§3 的全部键） |
| `configs/droid/pilot.yaml` | 试跑：只覆盖 §2 允许的键 |
| `configs/droid/full.yaml` | 正式运行：只覆盖 §2 允许的键 |

`load_config(path) -> DroidConfig`：读 `base.yaml`，再用 overlay 深度合并（字典逐键递归，其余整体替换，和 v7 `checks._merge_v4` 同一规则），然后构造冻结的 dataclass 并校验。

- overlay 里出现 §2 名单以外的键，或出现 base 里没有的键：报错。
- 合并后缺任何键：报错，不补默认值。
- `config_sha256` = 合并后配置的规范 JSON（`cache._json_bytes`：排好序的键、固定分隔符）的 SHA-256。它进每个产物的清单和训练登记。
- 数据根和运行根不进配置文件，由命令行 `--data-root`、`--run-root` 给出（换机器不改配置 SHA）；运行根下的子目录名用配置的 `name`。

## 2. overlay 能改的键

只有这些：`name`、`data.train_episodes`、`training.updates`、`training.seeds`、`model.spectral_backend`（求谱用哪个求解器，只影响速度和舍入，09 §6.3）、`resources.*`（各阶段的卡数、核数、内存、时长上限）。

| 键 | pilot | full |
|---|---|---|
| `name` | `pilot` | `full` |
| `data.train_episodes` | 500 | 20000 |
| `training.updates` | 2000 | 30000（试跑测速后可在 full.yaml 里改，改后配置 SHA 随之变化） |
| `training.seeds` | [1703] | [1703, 1704, 1705] |

由 `training.updates` 推出、不单独配置的量（`DroidConfig` 的属性，不是键）：

| 量 | 规则 | pilot | full |
|---|---|---|---|
| 学习率预热 | round(0.05·U) | 100 | 1500 |
| 一致性预热 | round(0.10·U) | 200 | 3000 |
| 验证间隔 | U / 20（必须整除，否则报错） | 100 | 1500 |
| 早检步数 | 一致性预热 + round(0.04·U)（v7 的 1400/10000 同比例） | 280 | 4200 |
| 一致性占比的测量间隔 | U / 400（必须整除）；早检时正好有 17 次测量 | 5 | 75 |

## 3. base.yaml 的全部键和取值

```yaml
schema: droid_config_v1
name: base                      # base 本身不能直接用来跑（name 必须被 overlay 改掉，否则报错）
data:
  version: "1.0.1"
  list_api: "https://storage.googleapis.com/storage/v1/b/gresearch/o"
  object_base: "https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1"
  prefix: "robotics/droid_raw/1.0.1/"
  expected_metadata_files: 74896
  camera: left_mp4_path         # 和官方 AC 训练用的相机相同
  min_steps: 242                # h5 步数下限；视频比 h5 少 1 帧，242 步 = 241 帧 = 16 个周期
  ood_labs: [PennPAL, REAL, RPL]
  ood_quota: [167, 167, 166]    # 与 ood_labs 一一对应
  id_test_scenes: 100
  val_scenes: 60
  episodes_per_scene: 5
  train_episodes: 20000         # overlay 覆盖
  shard_size: 1000
  split_salt: "po-jepa-droid-1.0.1-split-v1"
  window_salt: "po-jepa-droid-1.0.1-windows-v1"
  use_failures: true            # 开关；false 时训练顺序里跳过失败回合（测试集不变）
time:
  frames_per_interval: 15
  nominal_interval_seconds: 1.0
  min_interval_seconds: 0.8     # 准备阶段的时间检查（04 §4）
  max_interval_seconds: 1.6
action:
  source: realized              # realized（官方，实际位姿差）| commanded（指令位姿差）
  scale_episodes: 500           # 训练顺序前 500 条（通过检查的）
  scale_quantile: 0.99
  position_bound_m: 1.0         # 物理界，超出算准备检查失败
  rotation_bound_rad: 3.2
  gripper_bound: 1.0
encode:
  upstream_commit: "204698b45b3712590f06245fbfba32d3be539812"
  checkpoint_file: "vjepa2-vitg-fpc64-256.pth"
  batch_frames: 32
  storage_dtype: float16
  test_extra_episodes: 200      # 每个测试集的前 200 条另编
  frame_rate_stride_frames: 5
  extra_resolutions: [192, 320]
model:
  grid: 16
  consistency_grid: 8
  test_grid: 12
  modes: 16
  sigma_x: 0.125
  rho: 1.0
  spectral_backend: cpu_batched # cpu_batched（v7）| cpu_threads | cuda_float64，09 §6.3；full.yaml 按 check/env 的结果改
  ac_state: true_state          # true_state（官方）| learned_constant（v7 做法）
training:
  durations: [1, 2, 4, 8]
  batch: 64
  ac_batch_clips: 16
  ac_clip_frames: 9
  updates: 30000                # overlay 覆盖
  lr_min: 1.0e-4
  grad_clip: 1.0
  checkpoint_minutes: 30
  stop_margin_minutes: 10
  log_every: 100
  seeds: [1703, 1704, 1705]     # overlay 覆盖
  conditions: [po_jepa_full, po_jepa_no_consistency, fixed_fourier, koopman,
               matched_transformer, transformer_consistency, vjepa2_ac, vjepa2_ac_small]
  untrained: [identity, vjepa2_ac_zeroshot]
matching:                       # 参数量匹配的候选，只按参数量选（09 §2.3）
  tolerance: 0.01
  transformer_feedforward_candidates: [368, 384, 400]   # S00t：token 写法 PO-JEPA 248,737（09 §2.3）
  ac_small_mlp_ratio_candidates: [5.75, 6.0, 6.25]
monitoring:
  min_share: 0.005              # 加权一致性梯度至少是 JEPA 梯度的 0.5%（v7 同值，09 §4.6）
calibration:
  seed: 1703
  pretrain_updates: 300
  median_batches: 32
  sigma_z_episodes: 8           # 训练顺序前 500 条里等距取 8 条（09 §3.1）
  sigma_z_rows: [0, 8, 16]
evaluation:
  durations_id: [1, 2, 4, 8]
  durations_heldout: [3, 12, 16]
  durations_long: [24, 32]
  windows_per_duration: 4
  composed_step: 8
  frame_rate_single: [5, 10, 20]       # 单步的帧数 = 1/3、2/3、4/3 个周期
  frame_rate_rollout: [10, 20]         # 用 2/3 和 4/3 周期的步长滚到 4 个周期
  frame_rate_total_frames: 60
  planning_durations: [1, 4]
  planning_candidates: 64
  cem: {iterations: 4, samples: 256, elites: 24, init_std: 0.5, clip: 1.5}
resources:                      # 每个阶段一份：gpus、cpus、mem_gb、time_limit_minutes（12 §2）
  index:    {gpus: 0, cpus: 32, mem_gb: 16,  time_limit_minutes: 120}
  download: {gpus: 0, cpus: 16, mem_gb: 8,   time_limit_minutes: 600}
  prepare:  {gpus: 0, cpus: 8,  mem_gb: 32,  time_limit_minutes: 240}
  stats:    {gpus: 0, cpus: 8,  mem_gb: 32,  time_limit_minutes: 60}
  encode:   {gpus: 1, cpus: 8,  mem_gb: 64,  time_limit_minutes: 600}
  check:    {gpus: 1, cpus: 8,  mem_gb: 64,  time_limit_minutes: 120}
  calibrate: {gpus: 1, cpus: 8, mem_gb: 64,  time_limit_minutes: 240}
  train:    {gpus: 1, cpus: 8,  mem_gb: 96,  time_limit_minutes: 1440}
  evaluate: {gpus: 1, cpus: 8,  mem_gb: 96,  time_limit_minutes: 600}
  summary:  {gpus: 0, cpus: 4,  mem_gb: 16,  time_limit_minutes: 60}
```

校验（任一不满足就报错）：`ood_quota` 与 `ood_labs` 等长且和为 500；`id_test_scenes`、`val_scenes`、`episodes_per_scene`、`train_episodes`、`shard_size` 为正整数；`training.durations` 必须是 [1,2,4,8]（v7 的 `WindowQuery` 和时间一致性的半程都依赖它）；`training.conditions` 是 08 §1.2 的 8 个名字、顺序固定；`seeds` 非空、互不相同；两个 `salt` 非空；三个开关取值在注释列出的范围内；`frame_rate_single` 和 `frame_rate_rollout` 都是 `frame_rate_stride_frames` 的倍数，`frame_rate_total_frames` 是 `frame_rate_rollout` 每一项的倍数。

**开关**（`action.source`、`model.ac_state`、`data.use_failures`）的默认值就是主实验。开关只能在 base.yaml 的副本里改，改了就是另一个配置名、另一个运行目录；pilot 和 full 不允许改开关。

## 4. 时间

- 一个控制周期 = 15 个录制帧。DROID 的 MP4 文件头写 60 fps，实际约 14 Hz（相机时间戳间隔约 71 ms），所以 15 帧实测约 1.06 秒；这和官方 V-JEPA 2-AC 训练时的步长相同（官方按文件头 60 fps、4 fps 取帧，每 15 帧一帧）。
- 模型只看周期数。`ControlSpec.control_period_seconds = 1.0`（名义值）；`TrainingBatch` 里的“物理秒”一律取 `周期序号 × 1.0`，这样 v7 `_batch` 的端点核对（目标秒 = 起点秒 + 时长 × 周期）在 float64 下精确成立。实测秒数只记在准备文件里（04 §3），不进模型。
- 特征第 i 行是第 15i 个视频帧（第 i 个周期边界）。

## 5. 7 维动作

### 5.1 定义（官方，`source: realized`）

对一条回合，取视频帧序号 `idx = [0, 15, 30, …]`（04 §2），状态 `s = concat(observation/robot_state/cartesian_position, observation/robot_state/gripper_position[:, None])[idx]`，float64 [F, 7]。动作 `a = poses_to_diffs(s)`，[F−1, 7]，第 j 行是周期 j（从第 j 行特征到第 j+1 行特征）的动作：

```python
def poses_to_diffs(poses):            # 逐字取自 vjepa2@204698b app/vjepa_droid/droid.py
    xyz = poses[:, :3]
    thetas = poses[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    xyz_diff = xyz[1:] - xyz[:-1]
    angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    angle_diff = np.stack([d for d in angle_diff], axis=0)
    closedness = poses[:, -1:]
    closedness_delta = closedness[1:] - closedness[:-1]
    return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)
```

函数体一字不改（含 scipy 的 `Rotation`），S05 用 5 条金标准回合核对逐位相同（14 §3 S05）。上游是 `DROIDVideoDataset` 的方法（带 `self`），这里写成模块函数，函数体按 AST 相同；上游文件原样放在 `droid-golden/official_droid_204698b.py`。scipy 由 S01 加进 `pyproject.toml`，锁 `scipy==1.16.3`（1.15.3、1.16.3、1.17.1、1.18.1 在 5 条金标准回合上都和参照逐位相同）。

`source: commanded` 时把状态换成 `concat(action/cartesian_position, action/gripper_position[:, None])[idx]`，其余相同。只作开关，主实验不用。

### 5.2 尺度

位置、旋转、夹爪三组各一个尺度：训练顺序前 500 条（跳过准备检查失败的，04 §5）的全部周期动作，按组把 3 个（夹爪 1 个）通道的绝对值拼在一起，取 `numpy.quantile(…, 0.99, method="linear")`（float64）。只除尺度、不减均值，零动作仍是零。结果写 `prepared/action_scale.json`（04 §6），两种动作源各算一份。

### 5.3 `ControlSpec`

`droid_control_spec(scale: dict, *, source: str, config: DroidConfig) -> ControlSpec`（S01 补：界和周期取自 `config`，所以要传配置）。`scale` 是 `prepared/action_scale.json` 的内容：每个动作源一个键（`realized`、`commanded`），值正好是 `{position, rotation, gripper}` 三个正的有限 float；其余顶层键（uuid 列表、分位数、算法）只作记录。definition_source 里的 SHA 是 `cache._json_bytes(scale)` 的 SHA-256，等于 `_write_json` 写出的文件的 SHA。

| 字段 | 值 |
|---|---|
| `action_dim` | 7 |
| `control_period_seconds` | 1.0 |
| `channel_names` | `("dx", "dy", "dz", "rx", "ry", "rz", "grip")` |
| `channel_units` | `("m", "m", "m", "rad", "rad", "rad", "1")` |
| `lower_bounds` / `upper_bounds` | 位置 ∓1.0，旋转 ∓3.2，夹爪 ∓1.0（物理单位，取自配置） |
| `scale` | 位置尺度 ×3、旋转尺度 ×3、夹爪尺度 ×1 |
| `offset` | 全 0 |
| `definition_source` | `"DROID 1.0.1 <source> pose diffs, vjepa2@204698b poses_to_diffs, 15 frames per interval, action_scale.json sha256=<…>"` |

`ControlSpec` 是 v7 的类，不改。模型里的动作值 = 物理值 / 尺度（v7 `from_physical_segments` 的规则）。

### 5.4 AC 的动作单位

官方 AC 吃官方单位（物理值，不除尺度）。AC 包装层从 `controls.control_spec.scale` 乘回物理值，每个周期一个动作 token（09 §6.2）。原始帧率评测里 AC 的一步是重采样后的一帧间隔，动作是这一步的物理位姿差，不做换算（10 §8.4）。

### 5.5 非整周期的动作值

原始帧率评测里，PO-JEPA 类、Transformer 和 Koopman 的一步长 δ 个周期（δ = 1/3、2/3、4/3）。这一步的控制值取“每周期的位姿差”：`u = poses_to_diffs([s_f, s_{f+15δ}]) / δ`，再除尺度。这样常速运动在任何步长下给出同一个控制值，和训练时“一个周期一个位姿差”一致（10 §8.4）。

## 6. 代码接口（S01）

```python
@dataclass(frozen=True)
class DroidConfig:            # 字段和 §3 一一对应，嵌套的冻结 dataclass；外加 config_sha256: str
    ...
    @property
    def lr_warmup_updates(self) -> int: ...
    @property
    def consistency_warmup_updates(self) -> int: ...
    @property
    def validation_interval(self) -> int: ...
    @property
    def early_check_updates(self) -> int: ...
    @property
    def consistency_interval(self) -> int: ...
    def loss_config(self, *, time_weight: float, space_weight: float) -> LossConfig: ...

def load_config(path: Path) -> DroidConfig: ...
def poses_to_diffs(poses: np.ndarray) -> np.ndarray: ...       # §5.1，逐字
def droid_control_spec(scale: dict, *, source: str, config: DroidConfig) -> ControlSpec: ...
```

`loss_config` 返回 v7 的 `LossConfig`（不加字段）：`jepa_exponent`、`layer_norm_epsilon`、`time_epsilon`、`space_epsilon`、`consistency_metric` 取 v7 协议（`checks.load_protocol()` 在 v7 下的值）里的同名项，`grad_clip_norm = 1.0`、`lr_min = 1e-4`、`lr_decay_updates = U`、`lr_warmup_updates`、`consistency_warmup_updates` 取上表，两个权重由调用者给（09 §3）。
