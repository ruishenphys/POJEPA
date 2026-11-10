# 09 DROID 训练、标定、续跑

2026-11-08 · 第一版

本文件定 DROID 上每个条件怎么构造、σ_z 和 λ 怎么标定、训练循环（验证、选点、检查点、断点续跑、退出码、日志），以及第二期对共享模块的三处小改（S08b）。代码在 `src/po_jepa/droid/conditions.py`、`calibrate.py`、`train.py`、`checkpoints.py`、`check.py`（S09），S08b 的改动在 `baselines.py`、`ac_baseline.py`、`runtime/ac_training.py`、`spectral.py` 和新文件 `spectral_backends.py`。

## 1. 任务

| 任务 | 做什么 | 依赖 | 输出（运行根 `<run-root>/<name>/` 下） |
|---|---|---|---|
| `calibrate/sigma-z` | §3.1（CPU） | 训练片的特征 | `calibration/sigma_z.json` |
| `calibrate/lambda-<条件>` | §3.2，po_jepa_full、fixed_fourier、transformer_consistency 各一条（GPU） | sigma-z | `calibration/lambda-<条件>.json` |
| `check/env` | §7（GPU） | sigma-z | `check/env.json` |
| `train/<条件>/seed<种子>` | §4（GPU），8 个条件 × 种子 | sigma-z；需要 λ 的条件还依赖自己的 lambda；check/env | `train/<条件>/seed<种子>/` |

命令统一是 `python tools/droid.py <阶段> --config configs/droid/<name>.yaml --data-root … --run-root … --job <任务名>`，任务图见 12。

## 2. 条件工厂 `droid.conditions.build`

```python
def build(condition: str, *, config: DroidConfig, seed: int, sigma_z: float,
          device: torch.device, ac_checkpoint: Path | None = None) -> tuple[nn.Module, LossConfig, dict]
```

返回模型、损失配置和登记信息（参数量、规格快照、匹配记录、初始权重哈希）。

### 2.1 模型配置块

PO-JEPA 类的配置块 = po-jepa-full 里 `configs/train/default.yaml` 深度合并 `configs/train/v4.yaml`（和 v7 `checks.load_protocol()` 的前两步相同，不合并 v7 的日程和标定）之后的 `model` 段，再合并 v7 `configs/train/v7.yaml` 的 `model` 段（`dynamics.forcing_space: token`，S00t；不合并就是旧的模态写法，参数量和下表都不对），然后只改两项：`dynamics.action_dim = 7`，`graph.sigma_z = sigma_z`（§3.1 的值）。读之前先做 v7 的 `cache.verify_fixed_protocol()`，所以这两份文件必须和 v7 逐字节相同。其余全部沿用：投影 1408→64 残差读出，K = 16，σ_x = 0.125，Q 的种子 1703，指数欧拉积分、内部最大步长 0.25 个周期。

### 2.2 每个条件

| 条件 | 构造 | 损失 | 优化器 |
|---|---|---|---|
| po_jepa_full | `POJEPA(ProjectionConfig, GraphSpectrum(GraphConfig), ModalDynamics(DynamicsConfig), IntegratorConfig)`，顺序和 `checks.build_full_model` 完全相同，`torch.random.fork_rng(devices=[])` 里 `manual_seed(seed)` | `config.loss_config(time_weight=λ_t, space_weight=λ_s)`，λ 取 `lambda-po_jepa_full.json` | `training.make_optimizer` |
| po_jepa_no_consistency | 同上 | 两个权重 0 | 同上 |
| fixed_fourier | S10 的构造函数，显式传 §2.1 的配置块和 `seed`（08 §1.2） | λ 取 `lambda-fixed_fourier.json` | 同上 |
| koopman | S10 的构造函数，`KoopmanConfig(modes=87, action_dim=7)`，显式传 `seed` | 两个权重 0 | 和 Transformer 相同的 AdamW（07 §2.4） |
| matched_transformer | `TokenTransformer(TransformerSpec(action_dim=7, width=64, heads=4, layers=1, feedforward=384, residual=True))`，其余字段取 v7 默认；同样 `fork_rng` + `manual_seed(seed)` | 两个权重 0 | v7 的 Transformer AdamW |
| transformer_consistency | 和 matched_transformer 完全相同 | λ 取 `lambda-transformer_consistency.json` | 同上 |
| vjepa2_ac | `ACPredictor(ACSpec(action_dim=7, state_input=config.model.ac_state))`（官方规模），载入官方权重（§6.2） | v7 AC 的官方损失；学习率调度用 `config.loss_config(0, 0)`，峰值 1e-4 | `ac_training.make_optimizer` |
| vjepa2_ac_small | `ACSpec(width=64, depth=1, heads=4, mlp_ratio=6.0, action_dim=7, state_input=config.model.ac_state)`，从头训练 | 同上，峰值 4.25e-4 | 同上 |
| identity（不训练） | `IdentityPredictor(action_dim=7)` | — | — |
| vjepa2_ac_zeroshot（不训练） | 和 vjepa2_ac 相同的构造和载入，不训练 | — | — |

所有模型的初始化种子 = 运行种子（1703、1704、1705），标定时 = `calibration.seed`（1703）。初始权重哈希用 v7 `v7jobs._model_hash`。

### 2.3 参数量匹配

在构造时按规则现算，不看任何数据和结果，写进登记：

| 条件 | 候选（`matching` 配置段，01 §3） | 规则 | 结果 | 和 PO-JEPA（248,737，token 写法）相差 |
|---|---|---|---|---|
| matched_transformer、transformer_consistency | FFN 宽 [368, 384, 400] | 绝对差最小，平局取规格字典序较小的（v7 `choose_architecture` 同一规则） | 384 → 249,024 | 287（0.115%） |
| vjepa2_ac_small | mlp_ratio [5.75, 6.0, 6.25] | 同上 | 6.0 → 249,344 | 607（0.244%） |
| koopman | K 为任意正整数 | 08 §5.6 | 87 → 248,512 | 225（0.090%） |

v7 的候选（FFN 112/128/144，AC 的 mlp_ratio 2.0）在 7 维动作下都差 8% 以上，所以 DROID 另定候选：取使相差最小的值和它两侧各一档。任一条件超过 1% 就报错，不放宽。`ac_state: learned_constant` 时 AC 缩小版没有状态编码器、多一个常向量，同一组候选里仍是 6.0（248,896，相差 0.064%）。测试断言上表的每个数。（S00t 之后按 token 写法手算：Transformer 每单位 FFN 宽 129 个参数、7 维动作比 2 维多 320；AC 每 0.25 的 mlp_ratio 2,064 个；DROID 段实现时由测试核对）模态写法时的旧数（PO-JEPA 236,898；FFN 288；mlp_ratio 4.5；K 72）作废。

### 2.4 初始权重的关系

同一种子下（`check/env` 和每个训练任务开始时都核）：po_jepa_no_consistency 和 fixed_fourier 的初始哈希等于 po_jepa_full 的；transformer_consistency 的等于 matched_transformer 的；vjepa2_ac_zeroshot 的等于 vjepa2_ac 的（同一份官方权重，随机初始化的部分用同一种子）。

## 3. 标定

### 3.1 σ_z（`calibrate/sigma-z`）

和 v7 `workflow.calibrate_sigma_z` 同一规则，只把轨迹和帧换成 DROID 的回合和行：

1. 回合池：完整训练顺序（不受 `use_failures` 影响，02 §5）的前 `action.scale_episodes`（500）条里 `passed_<source>` 为真的回合，保持顺序，共 n 条（n ≤ 500，和动作尺度用同一池，04 §6）。取第 ⌊i·n/8⌋ 条，i = 0…7。试跑和正式运行取到同样的 8 条，所以 σ_z 相同。
2. 每条取特征第 0、8、16 行（每条回合至少 17 行，04 §1），用 `FeatureReader` 读（核行 SHA，float16 → float32），再转 float64。
3. Q：`torch.Generator().manual_seed(q_seed)`，`u, r = torch.linalg.qr(torch.randn(1408, q_dim, generator=g, dtype=torch.float64))`，`Q = (u * where(diag(r) < 0, −1, 1)).T`（[16, 1408]，q_seed = 1703，q_dim = 16，取自 §2.1 的配置块）。
4. 每一行 z = Z·Qᵀ（[256, 16]），收集 256 个 token 两两（有序对、不含对角）的平方距离，共 8 × 3 × 256 × 255 = 1,566,720 个值。
5. σ_z = sqrt(`torch.median`(全部值))（偶数个时取较小的中间值，和 v7 相同）。

`sigma_z.json`：`sigma_z`、8 条回合的 uuid、行号、平方距离的 5%/50%/95% 分位数、Q 的种子和维数、`config_sha256`、`feature_manifest_sha256`、所读每行的 SHA。σ_z 必须有限且为正。

### 3.2 λ（`calibrate/lambda-<条件>`）

和 v7 `workflow.calibrate_weights` 同一算法（16 §3.2 同样做法），批换成 DROID 的：

1. 种子 `calibration.seed`（1703），σ_z 取 §3.1 的值。
2. 建只用 JEPA 的版本：po_jepa_full 用 po_jepa_no_consistency；fixed_fourier 用两个权重为 0 的 fixed_fourier；transformer_consistency 用 matched_transformer。
3. 批：`step_rng(calibration.seed, step, 0)` 第 0–331 步的窗口（06 §2），用 `make_training_batch` 组装。
4. 初始状态下在第 0–31 步的批上量三项梯度范数（只记录）。
5. 第 0–299 步只做 JEPA 训练：v7 的单步、学习率调度（这个配置自己的 U 推出的预热）和裁剪，和正式训练相同。
6. 在第 300–331 步的批上量三项梯度范数：POJEPA 类用 `training.consistency_gradient_norms`，transformer_consistency 用 `addon.calibration.generic_gradient_norms`。
7. λ_c = ρ · median|∇L_JEPA| / median|∇L_c|，ρ = `model.rho`（1.0）。两个 λ 都必须有限且为正。

`lambda-<条件>.json` 的字段和 v7 的 `calibration.json` 相同（`rho`、`batches`、`statistic`、`method`、`pretrain_updates`、`pretrain_history`、`per_batch`、`median`、`mean`、`initial_state`、`time_weight`、`space_weight`、`graph`），另加 `config_sha256`、`sigma_z_sha256`（`sigma_z.json` 的 SHA）、`initial_model_sha256`、每一步窗口编号列表的 SHA。不写检查点。

λ 每个配置标一次（试跑和正式运行的 U 不同，预热长度不同，所以分别标），三个种子共用。

## 4. 训练循环（`train/<条件>/seed<种子>`）

### 4.1 目录

```
<run-root>/<name>/train/<条件>/seed<种子>/
  registration.json         # §4.2，第一次启动时写，之后每次启动都核对
  checkpoints/
    index.json              # 现存检查点的列表：路径、步数、种类、SHA；每次存档后原子重写
    step-000000.pt …        # 验证点检查点
    resume-012345.pt        # 续跑检查点，只留最新一个
  validations/step-000000.json …
  selection.json            # 到目前为止每个验证点的均值和当前最优
  train-log.jsonl           # §5
  report.json, COMPLETE.json
  FAILED.json               # 只在出错时
```

### 4.2 登记

`registration.json`（`cache._write_json`）：`schema: droid_train_registration_v1`、条件、种子、配置名和 `config_sha256`、训练集（条数、uuid 列表的 SHA，06 §1）、`action_scale.json` 和 `windows/val.json` 的 SHA、`feature_manifest_sha256`、`sigma_z.json` 和 λ 文件的 SHA 与取值、模型类型和规格快照、参数量和匹配记录（§2.3）、初始权重哈希、`spectral_backend`、损失配置快照（`training._loss_config`）、优化器配置、来源字典（06 §5）、AC 的官方权重 SHA 和载入记录、python/torch/numpy 版本。

每次启动先按当前代码和配置重算一遍。文件已存在且内容不同就报错退出（退出码 2）：代码、配置或上游产物变了，不能接着旧的跑。要重跑只能换运行根或手动删目录。

### 4.3 主循环

```
如果 COMPLETE.json 存在：直接退出 0
如果 FAILED.json 存在且没有 --retry：报错退出 1（--retry 时删掉它，从最新检查点接着跑）
核对登记（§4.2）
构造模型和优化器（§2）
从 checkpoints/index.json 取步数最大的检查点载入（同一步有两种时取验证点的）；没有就从头开始：
    在第 0 步验证并存 step-000000.pt（§4.4）
删掉 checkpoints/ 下不在 index.json 里的文件（上次存档时中断留下的）
把 train-log.jsonl 里末步 > completed 的行去掉
数据加载器从 completed 开始按步号产出批（06 §6）
for step in completed … U−1：
    batch = 第 step 步的批（核对加载器给的步号 == step）
    单步：PO-JEPA 类 training.train_step(…, condition=标签)；Transformer 类和 Koopman 走 07 §2.4 的单步；AC 走 ac_training.train_step（§6.2）
    completed = step + 1
    一致性占比（§4.6）、写日志（§5）
    completed == early_check_updates 时做提前检查（§4.6）
    completed % V == 0 或 completed == U：验证、存验证点检查点（§4.4）
    否则，如果距上次存档 ≥ checkpoint_minutes，或收到信号，或离时长上限不到 stop_margin_minutes：存续跑检查点（§4.5）
    收到信号或快到时长上限：存档后退出 75
选点、写 report.json 和 COMPLETE.json，退出 0
```

V = U/20（01 §2）。批的内容只由（种子，步号）决定，载入检查点后 RNG 状态也恢复，所以续跑和不中断的结果在 CPU 上逐位相同（§8 测试）；GPU 上不保证逐位（v7 也不保证），日志能看出差别。

### 4.4 验证和选点

- 用 `windows/val.json` 的 1,200 个窗口，按文件顺序每 64 个一批（最后一批 48 个），模型 eval 模式、`no_grad`。
- 指标只有未来误差：对每个窗口做一次直接前向 `model(raw_z, controls, duration, grid16)`，`jepa_distance(prediction, raw_target).per_sample`；AC 用绑定起点状态的前向（§6.2）。所有条件用同一种前向，这也是测试时的未来误差（10 §8）。和 v7 不同，验证时不算 TCD 和 SCD（v7 也不用它们选点），省时间。
- 验证批的组装同 06 §4，另外带出每个窗口起点行的真实状态 `state[s]`（float64 [B, 7]，只给 AC 用）。
- `validations/step-<6 位>.json`：每个窗口的编号和值、均值、用时。
- 选点：均值严格更小才换最优，平局留更早的；第 0 步也参加（和 v7 相同）。`selection.json` 每次验证后原子重写。
- 每个验证点存 `checkpoints/step-<6 位>.pt`。vjepa2_ac 的检查点约 3.6 GB（含 AdamW 状态），只保留当前最优和最近一个验证点，其余在新文件写好之后删掉；其他条件全部保留（每个几 MB）。

### 4.5 检查点和续跑

- 格式完全用 v7 的保存函数和字段：PO-JEPA 类 `training.save_training_checkpoint`，matched_transformer `comparison.save_checkpoint`，transformer_consistency 和 koopman 用 S10 的保存函数，AC 用 `ac_training.save_checkpoint`（规格快照见 §6.2）。
- `sampling_state = {"kind": "explicit_prepared_batches", "next_batch_index": completed, "query_generator_state": None}`，满足 v7 `training._sampling` 的要求：DROID 的批由（种子，步号）算出，没有要保存的生成器状态。
- v7 的保存函数拒绝覆盖已有文件，所以每次存档用新文件名；写好后把它加进 `index.json`（先写临时文件再改名），再删掉旧的续跑检查点。
- 载入：PO-JEPA 类用 `training.load_training_checkpoint`（传 DROID 的损失配置和来源）；AC 用 `ac_training.load_checkpoint`；v7 的 `comparison.load_transformer_checkpoint` 会拿 v7 协议的损失配置去比，DROID 不能用，所以 `droid/checkpoints.py` 里另写一个 Transformer 的载入函数，检查项和 v7 那个逐条相同，只是期望的损失配置换成 DROID 的。都先核 `index.json` 里记的 SHA。
- 载入后要求检查点的步数等于 `next_batch_index`，且不大于 U。

### 4.6 一致性占比和提前检查

对至少一个一致性权重 > 0 的条件（po_jepa_full、fixed_fourier、transformer_consistency）：

- 每 `monitoring.consistency_interval` 步（U/400：试跑 5、正式 75）在本步的批上量一次三项梯度范数，算 share = λ·w(s)·|∇L_c|/|∇L_JEPA|，和 v7 `comparison.consistency_share` 相同（transformer_consistency 用通用梯度范数）。
- 第 `early_check_updates` 步（01 §2：试跑 280、正式 4200）取一致性预热结束后各次测量的中位数，任一个权重 > 0 的项低于 `monitoring.min_share`（0.005）就写 `early-check.json` 和 `FAILED.json`（原因 `consistency_too_weak`），退出 3，不自动重排。每个条件各有 17 次测量参加判断。
- 训练结束时写一致性监控汇总进 `report.json`（v7 `consistency_monitor_summary` 的字段）。

### 4.7 时长上限和退出码

- 时长上限由命令行 `--time-limit-minutes` 给（任务图按 `resources.train.time_limit_minutes` 填），从进程启动开始算。
- 捕获 SIGTERM 和 SIGUSR1：只设一个标志，当前这一步做完再存档退出。
- 每步结束时如果距上限不到 `stop_margin_minutes`（10 分钟），同样存档退出。存档前先把日志里未满 100 步的那部分写成一行（§5）。

| 退出码 | 含义 | 任务图怎么处理（12） |
|---|---|---|
| 0 | 完成，或早就完成 | 下游可以开始 |
| 75 | 已存续跑检查点，没跑完 | 重排同一条命令 |
| 3 | 一致性太弱，提前停止 | 失败，等人看 |
| 2 | 登记对不上 | 失败，等人看 |
| 1 | 其他错误（写了 FAILED.json） | 失败；查清后用 `--retry` 接着跑 |

## 5. 日志和报告

v7 每步写一份完整的 JSON；DROID 一个任务 3 万步，改成 `train-log.jsonl` 里每 100 步一行（存档前把不满 100 步的部分也写成一行）。每行：

- `first_step`、`last_step`（完成的步数，从 1 计）；
- 这几步每一步的：六项损失（AC 另加 `ac_teacher_forcing`、`ac_rollout`）、裁剪前梯度范数、梯度最大绝对值、学习率、单步用时、等数据的时间、窗口编号列表的 SHA（`sha256("\n".join(sample_ids))`）、p_out 调用次数；
- 这一段里的一致性占比测量（步号、三项范数、share）；
- GPU 峰值显存、GPU 型号。

续跑后，日志的各行按 `first_step` 排起来正好覆盖 1…U，不重不漏（`report.json` 写之前核）。需要查某一步用了哪些窗口时，按种子和步号用 06 §2 的函数重算，再和日志里的 SHA 对。

`report.json`：条件、种子、完成步数、最优验证点（步数、均值、检查点路径和 SHA）、全部验证点、一致性监控汇总、总用时、续跑次数、参数量、初始权重哈希、`registration.json` 的 SHA。`COMPLETE.json` 记 `report.json` 的 SHA。

## 6. 共享模块的第二期改动（S08b）

这些改动都要求 v7 的 6 个条件逐位不变（07 §1、§5 的核对方法照用）。

### 6.1 Transformer 和 identity 接受 7 维动作

- `TransformerSpec.__post_init__`：`action_dim` 允许 2 或 7（原来只许 2）。字段本来就有，不加新字段，v7 的快照不变。
- `TokenTransformer.__init__`：`self.action_embed = nn.Linear(spec.action_dim + 2, spec.width)`（原来写死 4）。m = 2 时就是 `Linear(4, width)`，构造顺序和随机数消耗不变。
- `baselines.validate_inputs(…, action_dim=2)`：新关键字参数，默认 2，检查 `controls.control_spec.action_dim == action_dim`。`TokenTransformer.forward` 和 `comparison._tf_batch` 传 `self.spec.action_dim`。
- `IdentityPredictor(encoder_dim=1408, action_dim=2)`：新构造参数，只用于输入检查，不建参数。

### 6.2 AC 的官方输入模式

- `ACSpec`：`action_dim` 允许 2 或 7；新字段 `state_input: str = "learned_constant"`，取值 `learned_constant`（v7 做法）或 `true_state`（官方做法，要求 action_dim = 7）。新增 `spec_snapshot(spec)`：`asdict(spec)` 去掉取默认值的 `state_input`；`ac_baseline.describe` 和 `ac_training` 存档、读档里的 `asdict(model.spec)` 都换成它，v7 的规格快照因此逐字节不变。
- 构造：
  - action_dim = 7 时保留官方的 `action_encoder`（`Linear(7, width)`），vjepa2_ac 从官方权重载入，不再随机初始化；action_dim = 2 时和 v7 相同（换成新的 `Linear(2, width)`）。
  - `true_state` 时保留官方的 `state_encoder`（`Linear(7, width)`，vjepa2_ac 从官方权重载入），不建 `state_token`；`learned_constant` 时和 v7 相同。
  - `extrinsics_encoder` 都删掉（官方 `use_extrinsics=False`，本来就不用）。
  - `load_official` 的“被替换的键”名单随模式变：只有真正被替换的编码器不载入，其余键必须一一对应、形状相同。
- `predict_frames(context, actions, height, width, states=None)`：`true_state` 时要求 `states` [B, T, 7]，状态 token = `state_encoder(states)`；`learned_constant` 时要求 `states is None`。
- 前向 `forward(raw, controls, duration, grid, *, start_state=None)`：
  - 7 维时，每个周期的物理动作 = `tick_actions(controls, steps)` × `control_spec.scale` + `control_spec.offset`（官方单位，01 §5.4）；2 维时和 v7 相同。
  - `true_state` 时要求 `start_state`（[B, 7] 物理单位）；第 k+1 帧的状态 = `compute_new_pose(第 k 帧的状态, 第 k 个物理动作)`，在 CPU float64 上算，转 float32 送进模型。`compute_new_pose` 逐字复制自官方 `notebooks/utils/mpc_utils.py`（204698b）：位置相加，旋转 R_new = R(Δ)·R，夹爪相加后截到 [0, 1]。放在 `src/po_jepa/droid/poses.py`，S06 用金标准回合核对它和官方函数逐位相同。
  - 时长：v7 只接受整数和半整数（半整数取相邻两步平均）。7 维时放宽到任意正时长：t 不是整数时取相邻两步的线性插值 (1 − r)·ẑ_⌊t⌋ + r·ẑ_⌊t⌋+1，r = t − ⌊t⌋，ẑ_0 是 LN 后的起点。半整数时它就是 v7 的平均。2 维时 v7 的检查不变。
  - 非整数起点上的状态（只出现在 d = 1 的时间一致性半程，起点在 τ = 0.5）：对最后那个不满一个周期的部分用 r × 该周期的物理动作做一次 `compute_new_pose`。
- 训练步 `ac_training.train_step(…, states=None)`：`true_state` 时要求 `states` [B, 9, 7]（真实状态，06 §4），teacher forcing 用 `states[:, :-1]`，第 n 步自回归用 `states[:, :n+1]`，和官方 `app/vjepa_droid/train.py` 相同；训练时用真实状态，滚动预测（验证、测试、规划）时状态由动作积分，和官方规划的做法一致。不传时和 v7 相同。
- 绑定起点状态：DROID 的评测代码用 `droid.ac.bind(model, start_state, reference)` 得到一个只读的包装，前向签名和其他模型相同（`(raw, controls, duration, grid)`）。`reference` 是这个窗口完整的 `ActionSegments`；每次调用时，按 `controls.source_origin_seconds` 减去 `reference.source_origin_seconds` 得到这次调用的起点相对窗口起点的周期数，用 `reference` 里这段时间的动作把状态从窗口起点积分过去（时间一致性和分割滚动里后面几段的起点），再调用上面的前向；起点不在 `reference` 的范围内就报错。动作干预时另建一个以替身动作为 `reference`、起点状态仍是本窗口的包装。这样 10 §3 和 §8 的指标可以对 AC 用同一套代码。
- 官方规模在 CUDA 上照 v7 用 bf16 autocast，缩小版 float32。

### 6.3 求谱后端

- `GraphSpectrum` 加一个普通属性 `eig_backend`（不是配置字段、不是 buffer，不进 `_model_config` 快照），默认 `"cpu_batched"`。`lowest_eigenpairs` 只把其中那一次 `torch.linalg.eigh` 换成按后端分发，前后的检查、符号 pivot、残差、正交误差、间隙都不变，仍在 CPU float64 上算。默认值走的就是 v7 原来的那一行。
- 三个后端，写在新文件 `src/po_jepa/spectral_backends.py`：

| 后端 | 做法 |
|---|---|
| `cpu_batched` | v7：`torch.linalg.eigh(L, UPLO="L")`，整批一次 |
| `cpu_threads` | 按样本切成 `cpus` 份，线程池里各自 `eigh`，按原顺序拼回 |
| `cuda_float64` | L 转到模型所在的 GPU（仍 float64）`eigh`，结果拷回 CPU float64 |

- 后端由配置的 `model.spectral_backend` 给出（overlay 允许改这一项，01 §2），工厂把它设到 `model.spectrum.eig_backend`，写进登记；续跑时登记要相同。FourierSpectrum 不求谱，不受影响。
- 选哪个：`check/env`（§7）在 8 个真实训练批上比较三个后端。某个后端算“一致”的条件是：特征值和 cpu_batched 的最大绝对差 ≤ 1e-9；截断间隙 ≥ 1e-6 的样本上，前 16 个特征向量张成的投影矩阵之差的 Frobenius 范数 ≤ 1e-6。正式运行在 full.yaml 里选用一致的后端里最快的那个；都不一致就用 cpu_batched。比较结果和测速写进 `check/env.json`，论文附录据此写明用的是哪个求解器。
- 不做：同一步里同一个输入的谱算了好几次（5 次 Φ 里有 3 次的输入相同），加缓存会改 v7 的代码路径，所以不加。

### 6.4 v7 条件不变的核对

07 §5 的四项照做。另加：6 个 v7 条件在 po-jepa-full 里用默认参数构造的 `TransformerSpec`、`ACSpec` 快照（`spec_snapshot`）和 v7 检查点里的逐字节相同；`eig_backend` 取默认值时，v7 双树冒烟逐位相同。

## 7. 环境检查 `check/env`

在一张训练卡上，用 `step_rng(1703, step, 0)` 的真实批：

1. 环境：GPU 型号和驱动、torch、CUDA、CPU 核数、特征所在磁盘的读速度（读 500 个随机行计时）。
2. 每个可训练条件：构造、参数量和 §2.3 的表一致、初始权重哈希的关系（§2.4）、20 步训练（损失和梯度全部有限、每个参数都有梯度、p_out 调用次数符合 07 §2.4）、第 6–20 步单步用时的中位数、峰值显存。
3. 由单步用时推算每个训练任务的总时长（U 步加 20 次验证），写进 `check/env.json`。试跑之后据此定 full.yaml 的 U 和各阶段的时长上限。
4. 求谱后端的比较（§6.3）。
5. 数据加载器：不跑模型，连续产出 200 步的批，记每秒步数；要求比最快那个条件的单步快，否则提示特征存储太慢。

任一项不通过就退出 1，训练任务不开始。

## 8. 测试（CPU，假特征，S09）

下面凡是“逐位相同”的比较都在 1 线程下做（14 §1）。

- 工厂：8 个可训练条件和 2 个对照都能构造；§2.3 表里的参数量逐个断言；初始权重哈希的关系成立；`build` 不改全局随机状态。
- σ_z：在小规模假特征上和直接调用 v7 `calibrate_sigma_z` 的同一算法（换成同样的行）逐位相同；回合池和 8 条的选法按 §3.1。
- λ：po_jepa_full 的标定在假特征上和 v7 `calibrate_weights` 的步骤一一对应（同样的批给出同样的梯度范数和 λ）。
- 续跑：每个可训练条件（vjepa2_ac 用测试专用的缩小假权重，代码路径相同）跑 30 步、验证间隔 10。在第 13 步模拟 SIGTERM、第 21 步模拟快到时长上限，每次退出 75 后重跑同一条命令，最终的 `model_state`、优化器状态、日志的每个数值、每个验证点的每个值都和不中断的那次逐位相同。
- 中断在存档中途（检查点写了、`index.json` 还没更新）：续跑时删掉孤立文件，结果仍逐位相同。
- 登记对不上时退出 2；FAILED.json 存在而不带 `--retry` 时不开始。
- 日志行覆盖 1…U 不重不漏；续跑后去掉的行正好是末步大于检查点步数的那些。
- S08b：v7 原有测试全过；`TransformerSpec`、`ACSpec` 的 v7 快照逐字节不变；`spec_snapshot` 去掉默认的 `state_input`；`true_state` 模式下 AC 的训练步和一份按官方 `train.py` 逻辑手写的参照逐位相同（假的小规格）；`compute_new_pose` 和官方函数逐位相同；三个求谱后端在 CPU 上（cuda 跳过）结果一致。

## 9. S08b 实现补注（2026-11-10）

代码和验收单见 `docs/segments/S08b.md`。实现时定的（不改口径）：官方规模按结构字段判断（v7 时与 `spec == OFFICIAL` 相同）；物理动作在 CPU float64 上由 `tick_actions` × scale + offset 算出（`ac_baseline.physical_tick_actions`），状态积分用它；`droid.ac.bind` 的起点 τ 离整数 ≤ 1e-9 个周期取整、越出 [0, 窗口时长] 报错；`ac_steps`（`droid.ac.ac_steps` → `ACPredictor.small_steps`）在 learned_constant 模式下忽略状态；`cpu_threads` 的份数是 `GraphSpectrum.eig_threads`（S09 工厂按 `resources.train.cpus` 设）；`check/env` 的一致标准用 `spectral_backends.agreement`。§8 “true_state 训练步和官方 train.py 参照逐位相同”实际是两项损失和除 `predictor_embed.bias` 外的全部参数逐位相同，这一项因官方先摊平各帧再过 Linear、求和顺序不同差约 1.5e-7（相对），改成摊平会动 v7 的数，不改。

## 10. S09a 实现补注（2026-11-10）

代码和验收单见 `docs/segments/S09a.md`。S09 拆成两段：S09a 是工厂（§2）和训练循环（§4、§5）；σ_z 和 λ 标定（§3）、`check/env`（§7）、评测读检查点和 S13 的读取函数留给 S09b。实现时定的（不改口径）：

- λ 由训练任务从 `calibration/lambda-<条件>.json` 读（time_weight、space_weight 为有限正数；文件里记了 `config_sha256`、`sigma_z_sha256` 就必须对上），再传给 `conditions.build`；工厂不读文件。
- 新加 `starts.jsonl`：每次启动一行（启动时的完成步数和从哪个检查点接上），`report.json` 的续跑次数 = 行数 − 1。
- `selection.json` 每次启动按 `validations/` 和 `index.json` 重建，中断在验证文件写完、选点没写之间也能接上。
- 验证窗口在 `commanded` 配置下跳过 `passed_commanded` 为假的回合（同 10 §8.1，用 `evaluate._selected`），窗口数和跳过的回合记进登记。
- 求谱后端只在 `type(model.spectrum) is GraphSpectrum` 时设（`FourierSpectrum` 是它的子类，fixed_fourier 不求谱）。
- §8 的续跑测试按原文跑 30 步；14 §3 的“20 步、两次验证、完整目录”并进同一次运行核（第 0、10、20、30 步验证）。测试每步 2 个窗口 / 2 个 AC 片段，CPU 上 PO 类一步约 1.5 秒。

## 11. S09b 实现补注（2026-11-10）

代码和验收单见 `docs/segments/S09b.md`。实现 §3 标定（`droid/calibrate.py`）、§7 `check/env`（`droid/check.py`）、10 §8.1 的读检查点（`evaluate.load_selected`）、评测任务入口（`droid/eval_job.py`）和 13 §8.3 的三个读取函数。实现时定的（不改口径，Jerry 可以改）：

- σ_z 的回合池直接取 `action_scale.json` 记的 `episodes[<source>]`（04 §6 的同一池），不重算。
- `check/env` 跑 20 步时 λ 用占位值 1（真 λ 要等标定；只测时长和有限性，不影响参数量和初始哈希）；总时长 = U·单步中位数 + (U/V + 1)·一次验证用时（第 0 步也验证，§7 第 3 条写的“20 次”按 U/V 算）。
- 数据加载器比最快条件的单步慢时按“不通过”处理（退出 1、写 FAILED.json），不只是提示。
- 不训练的对照（identity、vjepa2_ac_zeroshot）的构造种子取配置的第一个种子；评测任务的 `--time-limit-minutes` 只记录（评测按批落盘，重排同一条命令接着做）。
- 离线规划在 `evaluate_job` 的每个测试集循环里、写 COMPLETE.json 之前做，报告的 `axes.planning` 记上。
- 训练登记新加 `splits_sha256`（`index/splits.json` 的 SHA），汇总的 `data_sha256` 由它、验证窗口和动作尺度的 SHA 组成。
- 补 S11-fix1 留下的两处输入身份：`job_inputs` 加准备数组本身的 SHA（`prepared_arrays_sha256`）和测试附加特征清单的 SHA（清单内容的规范 JSON 的 SHA，`ExtraFeatureReader.manifest_sha256`）。主特征读取器在合成数据下的身份仍只靠来源字典（假编码器时特征清单 SHA 记 None），留作已知缺口。
- `load_selected` 拿到的模型不是这个条件的类时报 ValueError（原来是 AttributeError）。
