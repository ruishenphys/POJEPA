# 10 评测

本文件先写仿真部分。DROID 特有的评测轴（原始帧率、原始分辨率、留出实验室、离线规划）在补写 DROID 规格时加到 §8。

## 1. 原则

- 测试窗口、查询和指标在训练前固定，不按测试结果改。
- 每个训练条件只评测验证集选出的那个检查点（验证集未来误差最小，平局取更早的）。
- 评测全程 `model.eval()`、不求梯度，评测前后核对参数、梯度和 buffer 没变（沿用 v7 的 `_state_identity`）。
- 记号：一个测试窗口 w 是（测试轨迹，起点 tick s）。z 是起点的 16×16 特征，u 是从 s 开始的记录动作，z*_T 是第 s+T 个 tick 的特征。Φ^h_T(z; u) 是网格 h 上时长 T 的一次完整预测。

## 2. v7 的标准评测（不改）

补跑的 5 个新条件在各自的 D 任务里跑和 v7 完全相同的标准评测（`evaluation.run_prepared_evaluation`），这样新旧条件的标准指标直接可比：

- 400 个测试窗口：200 条测试轨迹，每条两个起点 START_TICKS[i%5] 和 START_TICKS[(i+2)%5]。
- 单点查询：时长 1、2、3、4、8、12，网格 16×16 和 12×12；8×8 只在训练时长 1、2、4、8。指标 `future_jepa_distance`、`tcd_rms`、`tcd_normalized`、`delta_act` 等。
- 网格对查询：每个时长上 16→8 和 16→12，指标 `scd_rms`、`scd_normalized` 等。
- 变体（分布外）评测：只评 16×16 单点（`v7jobs.transfer_job`）。
- 规划：`tools/plan_v7.py`，规则见论文 R12。

## 3. D4：只评测的新指标

D4 对每个条件、每个种子跑一次（v7 的 6 个条件读 v7 的检查点，补跑的 5 个读补跑的检查点），在同样的 400 个窗口上算下面这些量，另外在全部测试轨迹上算长时程（§3.6）。所有量都按窗口记录，汇总时先在轨迹内平均（13 §3）。

### 3.1 两种度量

对同一网格上的两个预测 a、b（[1, N, 1408]），逐样本算：

- **LN 空间**：RMS_LN(a, b) = √( mean_{N,E} (LN(a) − LN(b))² )。LN 和 JEPA 损失里的完全相同（逐 token、eps = 1e-5、无仿射）。**跨模型比较（包括 AC）用这个**：AC 的输出本来就在 LN 空间，原空间的 TCD 对它没有可比性。
- **输入尺度**：RMS_in(a, b) = √( mean_{N,E} (a − b)² ) / √s²(z)，s²(z) = `losses.input_scale(z)`（16×16 起点）。这就是 v7 `tcd_normalized` 的做法，**对照理论用这个**（定理里的范数是表示空间里的）。

两种度量都由 `ln_metrics.py` 提供，float64 计算，返回 [B]。

记号：Φ_Π(z) 表示按分割 Π = (τ_1, …, τ_n) 逐段滚动：每一段都是一次完整的 Φ 调用，输入是上一段输出的原始预测（和 `temporal_predictions` 的 composed 一样，不 LN、不截断梯度、不用真实中间帧），动作用 `split_action_segments` 按段精确切开。

### 3.2 LN 空间的 TCD 和 SCD

| 量 | 定义 | 取值 |
|---|---|---|
| `tcd_ln` | RMS_LN(Φ_T(z), Φ_{T/2}∘Φ_{T/2}(z)) / (T/2)² | T ∈ {1, 2, 3, 4, 8, 12}，16×16 |
| `scd_ln` | RMS_LN(R Φ^16_T(z), Φ^h_T(R z)) / T | h ∈ {8, 12}，T ∈ {1, 2, 3, 4, 8, 12} |

归一化（除以 τσ、除以 T）和 v7 的 `tcd_rms`、`scd_rms` 相同。R 是 v7 的 `build_spatial_transfer`（16→8 为 2×2 平均，16→12 为双线性）。

### 3.3 分割差异（定理控制的量）

pd(T, Π) = 距离(Φ_T(z), Φ_Π(z))，两种度量都报，**不除以时间**。每个分割同时记 x(Π) = (T² − Σ τ_i²)/2，定理预测 pd 大致随 x 线性增长、斜率约 ε_t e^{LT}。

| T | 分割 | x |
|---|---|---|
| 4 | 2 段 × 2 | 4 |
| 4 | 4 段 × 1 | 6 |
| 4 | 8 段 × 0.5 | 7 |
| 8 | 2 段 × 4 | 16 |
| 8 | 4 段 × 2 | 24 |
| 8 | 8 段 × 1 | 28 |
| 8 | 3 + 5 | 15 |

等分 2 段的 pd 和同一时长的 TCD 用的是同一对预测（TCD 再除以 (T/2)²），实现时可以复用，也可以重算，结果必须一致。

### 3.4 跨网格分割差异

xpd(T, n) = 距离(R Φ^16_T(z), Φ^8_{Π_n}(R z))，Π_n 是 n 段等分，T ∈ {4, 8}，n ∈ {2, 4}。度量在 8×8 上算；输入尺度版本仍用 16×16 起点的 s²。这是定理里“不同网格、不同时间分段”两种完整分割之间的差。

### 3.5 不等分 TCD

`tcd_unequal`(d; τ, σ) = 距离(Φ_d(z), Φ_σ∘Φ_τ(z)) / (τσ)，两种度量都报：

| d | (τ, σ) |
|---|---|
| 4 | (1, 3)、(3, 1) |
| 8 | (2, 6)、(6, 2) |

训练只约束等分，这一项检验缺陷能不能推广到非等分的切法。

### 3.6 留出时长的误差分解

留出时长 T* 拆成训练过的时长：3 = 1 + 2，12 = 8 + 4（先 1 后 2，先 8 后 4）。d(a, b) 用 JEPA 距离（mean |LN(a) − LN(b)|，即 `jepa_distance(...).per_sample`，它是伪度量）：

| 项 | 定义 |
|---|---|
| `heldout_direct` | d(Φ_{T*}(z), z*_{T*}) |
| `heldout_partition` | d(Φ_{T*}(z), Φ_Π(z)) |
| `heldout_rollout` | d(Φ_Π(z), z*_{T*}) |

每个窗口核对三角不等式 direct ≤ partition + rollout + 1e-6·max(1, direct)，记违反的个数（应为 0，不为 0 说明实现有错，任务失败）。

### 3.7 长时程

- 窗口：全部测试轨迹（正式运行 200 条），起点 tick 0，T ∈ {16, 24, 32}。仿真轨迹有 33 帧（`TRAJECTORY_TICKS = 32`），T = 32 正好用到最后一帧。
- 指标：`long_direct` = d(Φ_T(z), z*_T)；`long_composed` = d(Φ_{8,…,8}(z), z*_T)（T/8 段，每段 8）；两者之间的 pd 两种度量都报。identity 条件给出 d(z, z*_T) 作参照。
- 读数据：新函数 `addon.long_window.read_long_window(path, *, start_tick, duration_cycles, expected_split, device)`，和 `cache.read_window` 相同，只是允许 T ∈ {16, 24, 32}，并要求 start_tick + T 不超过缓存的最后一帧。内部复用 `cache._read_complete_cache` 和 `cache._window_controls`；`cache.read_window` 本身不改。

### 3.8 AC 的特殊处理

- AC 只接受整数或半整数时长；D4 用到的时长（含 0.5 段）都满足。半整数时长按 v7 的做法取相邻两步的平均。
- AC 的分割滚动把上一段输出（已在 LN 空间）当作下一段的原始输入，AC 内部再做一次 LN。
- 汇总里 AC 只参与 LN 空间指标的比较，原空间的 `tcd_normalized`、`scd_normalized` 标为不可比。

### 3.9 identity

identity 的所有差异都等于 0（Φ = Id）。它只作参照，不进精度匹配的比较（13 §2）。

### 3.10 成本

每个窗口约 80–100 次模型调用（不复用时的上界），和 v7 标准评测（约 100 次）同一量级；长时程每条轨迹 12 次调用，只占很小一部分。官方规模 AC 每次调用要逐步自回归，最慢。实际耗时以试跑为准。

## 4. 读 v7 检查点和等价核对

### 4.1 为什么要放宽

v7 的检查点在 `provenance.source_files_sha256` 里记了 v7 源码的 SHA，`training._provenance` 读入时会和当前源码逐个比较，po-jepa-full 里的源码不同，所以 v7 的检查点按原路径读不进来。D4 用 `addon.v7load.load_v7_selected(job_dir, condition, model)` 读，**核对数据一致性，不核对源码 SHA**，源码 SHA 原样记进 D4 的报告。

### 4.2 读取步骤

1. 用 `v7jobs._job_dir` 找到任务目录（`<输出>/run/job/`）。`job.json` 里的条件、数据集 tag、种子、步数必须和 D4 命令一致。`COMPLETE.json` 的 `report_sha256` 必须等于 `report.json` 的实际 SHA。
2. 训练目录 = `<任务目录>/training/<条件>`；它的 `report.json` 的 `best` 必须和任务报告的 `best` 相同。重跑 `workflow.verify_selection`（只读数据，和源码无关）确认选点规则；`completed_updates` 等于当前查询序列的长度，`query_sequence_sha256` 等于当前序列的 SHA。
3. 选中的检查点：路径不越出训练目录；文件 SHA 等于 `best.checkpoint_sha256`；`torch.load(weights_only=True)`；`completed_steps == best.step`。
4. 配置快照：PO-JEPA 类比较 `model_config` 和新树里现建模型的 `training._model_config`，以及 `loss_config` 和 `_loss_config(loss, 标签)`；Transformer 比较登记规格和 `loss_config`；AC 要求 `schema == "v7_vjepa2_ac_checkpoint_v1"` 且规格等于 `ac_baseline.spec_for(condition)`。任何一项不等就报错。
5. `training._check_model_state`（名字、形状、精度），然后 `load_state_dict(strict=True)`、`model.eval()`。不读优化器状态。
6. 来源里的数据字段必须和当前一致：`evidence_role`、`split`、`paper_sha256`、`upstream_commit`、`encoder_checkpoint_sha256`、`preprocess_sha256`、`feature_manifest_sha256`、`protocol_sha256`。`source_files_sha256` 原样记录，不核对。

补跑自己的检查点也走同一个函数（这时源码 SHA 本来就一致，照样只记录）。

### 4.3 等价核对

D4 在同样的窗口上重算 v7 的三项标准指标，和被读检查点所在任务自己的评测记录（`evaluation/<条件>/rows/*.json`）逐条比较：

- `future_jepa_distance`：16×16，T ∈ {1, 2, 3, 4, 8, 12}
- `tcd_normalized`：16×16，同样的 T
- `scd_normalized`：16→8 和 16→12，同样的 T

容差 |新 − 旧| ≤ 1e-6 · max(1, |旧|)。报告每项的条数、逐位相同的条数、最大绝对差和最大相对差。同一台机器上预期逐位相同；超出容差任务就失败，不写 COMPLETE.json，说明新树读 v7 检查点或复制来的计算路径有问题，要先查清楚。identity 也做这项核对（参照是 v7 identity 任务的记录）。

## 5. 截断间隙

论文说 K = 16 时第 K 个特征值位于谱的稠密部分以下。v7 的谱诊断算了 λ_{K+1} − λ_K，但没有保存。

- 图只由编码器特征、Q（种子 1703）、σ_x 和本场景的 σ_z 决定，和条件、种子无关，所以**每个场景只跑一次**（`run.py spectral-gap`，CPU）。
- 对 400 个测试窗口的起点帧，在 16×16、8×8（R z）、12×12（R z）上用 po_jepa_full 的 `GraphSpectrum` 求最低 16 个特征对，记录 λ_16、λ_17、`cutoff_gap`、前 16 个模态相邻间隙的中位数，以及两者之比。
- 报告每个场景、每个网格上这些量的 5、25、50、75、95 分位数。
- 预先定好的判断规则：每个场景、每个网格上比值的中位数都大于 1，论文保留原句；否则改成“K = 16 is fixed in advance and shared across grids”。
- fixed_fourier 的截断间隙由公式给出，恒为 0（08 §4.3），在同一张表里列出。

## 6. 编码器核对

官方 AC 权重 `vjepa2-ac-vitg.pt` 里自带一份编码器。如果它和我们用的 `vjepa2-vitg-fpc64-256.pth` 的 `target_encoder` 不是同一份，零样本 AC 吃到的就是它没见过的特征。

- `tools/encoder_identity.py`，CPU 任务，A0 段。
- 两边去掉 `module.`、`backbone.` 前缀后按名字配对，比较 AC 权重里的 `encoder`（和 `target_encoder`，如果有）与 fpc64-256 的 `target_encoder`（和 `encoder`）。
- 报告：配对上的张量数、按存储精度逐位相同的个数、最大绝对差、两边各自配不上的键、精度不同的键。全部配对、全部逐位相同才判“同一份”。
- 结果写进论文。

## 7. 输出格式

`run.py d4` 的输出目录：

```
<输出>/job.json            # 条件、场景、种子、步数、被读的任务目录
<输出>/source.json         # 被读检查点的 best 记录、文件 SHA、v7 源码 SHA（原样）、核对过的数据字段
<输出>/rows/<序号>.json     # 每个窗口一条：窗口、§3.2–§3.6 的全部量、等价核对的新旧值
<输出>/long/<轨迹>.json     # 每条轨迹一条：§3.7 的量
<输出>/report.json         # 条数、等价核对汇总、三角不等式违反数、各量的轨迹宏平均
<输出>/COMPLETE.json       # report.json 的 SHA；只有全部核对通过才写
```

所有 JSON 用 `cache._write_json`（排好序的键、固定格式），同样的输入写两次字节相同。

## 8. DROID 评测轴（待写）

同分布、留出实验室两个测试集；时长留出 3、12、16；长时程 24、32；原始帧率（单步 1/3、2/3、4/3，用 2/3 和 4/3 的步长滚到 4）；原始分辨率（192 输入的 12×12、320 输入的 20×20）；离线规划。见 DROID 架构文档 rev 14，补写时并入本文件。§3 的指标 DROID 上同样计算，时长按 DROID 的训练时长调整。
