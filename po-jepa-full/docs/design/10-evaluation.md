# 10 评测

§1–§7 是仿真部分，§8 是 DROID 的评测轴（离线规划另见 11）。

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

D4 对每个条件、每个种子跑一次（v7 的 6 个条件读 v7 的检查点，补跑的 5 个读补跑的检查点），在同样的 400 个窗口上算下面这些量，另外在全部测试轨迹上算长时程（§3.7）。所有量都按窗口记录，汇总时先在轨迹内平均（13 §3）。

### 3.1 两种度量

对同一网格上的两个预测 a、b（[1, N, 1408]），逐样本算：

- **LN 空间**：RMS_LN(a, b) = √( mean_{N,E} (LN(a) − LN(b))² )。LN 的定义和 JEPA 损失里的相同（逐 token 沿 E、总体方差、eps = 1e-5、无仿射），但在 float64 上算。**跨模型比较（包括 AC）用这个**：AC 的输出本来就在 LN 空间，原空间的 TCD 对它没有可比性。
- **输入尺度**：RMS_in(a, b) = √( mean_{N,E} (a − b)² ) / √s²(z)，s²(z) = `losses.input_scale(z)`（16×16 起点）。这就是 v7 `tcd_normalized` 的做法，**对照理论用这个**（定理里的范数是表示空间里的）。

两种度量都由 `ln_metrics.py` 提供（`rms_ln(a, b)`、`rms_input_scale(a, b, raw_z)`，S08a）：

- 输入 a、b 是同形、同设备、全部有限的 float32 [B, N, E]（其他精度的输出由调用者先转成 float32）；`raw_z` 是 float32 [B, N0, E]，N0 可以和 N 不同（跨网格的 xpd 在 8×8 上算，s² 仍用 16×16 起点）。
- a、b 先各自转 float64 再相减；返回 float64 [B]，和输入同设备，不带梯度。
- 不做时间归一化：tcd_ln、scd_ln、各种 pd 的归一化由调用者按 §3.2–§3.5 做。
- 因为先转 float64，`rms_input_scale(direct, composed, z)/(τσ)` 和 v7 的 `tcd_normalized`（v7 先在 float32 里相减）不是逐位相同，相对差在 1e-7 以内（float32 相减的舍入界约 6e-8）。所以 §4.3 的等价核对不用 `ln_metrics`，照抄 v7 的 float32 写法（`point_metrics`、`pair_metrics` 里的那几行）；D4 自己的量才用 `ln_metrics`。

记号：Φ_Π(z) 表示按分割 Π = (τ_1, …, τ_n) 逐段滚动：每一段都是一次完整的 Φ 调用，输入是上一段输出的原始预测（和 `temporal_predictions` 的 composed 一样，不 LN、不截断梯度、不用真实中间帧），动作用 `split_action_segments` 按段精确切开。

Φ_T(z) 的动作写法：v7 有两种，`point_metrics` 的 `direct` 把动作在 T/2 处 `split_action_segments` 切开再 `concatenate_action_segments` 接上，`pair_metrics` 和 `delta_act` 用不切开的 u；T/2 落在一个 tick 内部时（T = 1、3）两种写法的控制段不同，数值可以不同。`tcd_ln`、pd、xpd 的细网格一侧、`tcd_unequal`、留出分解、长时程都用 `point_metrics` 的写法（等价核对的 `future_jepa_distance`、`tcd_normalized` 必须用它才能和 v7 的记录对上，同一个张量也可以复用）；`scd_ln` 用 `pair_metrics` 的写法（不切开），和 v7 的 `scd_normalized` 一致（Jerry 2026-11-08 定：按推荐做法）。受影响的论文数字是 `tcd_ln:T=1`、`tcd_ln:T=3`、`heldout_*:T=3`（其余时长 T/2 是整数 tick）。

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

等分 2 段的 pd 和同一时长的 TCD 用的是同一对预测（TCD 再除以 (T/2)²），实现时可以复用，也可以重算：对同一对张量调用同一个函数，结果逐位相同。

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

v7 的检查点在 `provenance.source_files_sha256` 里记了 v7 源码的 SHA，`training._provenance` 读入时会和当前源码逐个比较，po-jepa-full 里的源码不同，所以 v7 的检查点按原路径读不进来。D4 用 `addon.v7load.load_v7_selected(job_dir, condition, model, *, loss_config, features, device)` 读（第 4 步要比 `loss_config`，第 6 步要读特征组训练批），**核对数据一致性，不核对源码 SHA**，源码 SHA 原样记进 D4 的报告。

### 4.2 读取步骤

1. 用 `v7jobs._job_dir` 找到任务目录（`<输出>/run/job/`）。`job.json` 里的条件、数据集 tag、种子、步数必须和 D4 命令一致。`COMPLETE.json` 的 `report_sha256` 必须等于 `report.json` 的实际 SHA。
2. 训练目录 = `<任务目录>/training/<条件>`。续训和只评测的任务（v7 的 `run.py resume`、`run.py evaluate --training-run`）自己没有完成的训练目录，沿 `job.json` 的 `training_run`、`resume_from` 找旧任务，最多 8 层，旧任务的条件必须相同；`source.json` 的 `training_dir` 记实际用的目录。训练目录的 `report.json` 的 `best` 必须和任务报告的 `best` 相同。重跑 `workflow.verify_selection`（只读数据，和源码无关）确认选点规则；`completed_updates` 等于当前查询序列的长度，`query_sequence_sha256` 等于当前序列的 SHA。
3. 选中的检查点：路径不越出训练目录；文件 SHA 等于 `best.checkpoint_sha256`；`torch.load(weights_only=True)`；`completed_steps == best.step`；检查点的 `provenance` 等于训练报告 `best.checkpoint_provenance`（两份都是保存时写的，不涉及源码）。
4. 配置快照：PO-JEPA 类比较 `model_config` 和新树里现建模型的 `training._model_config`，以及 `loss_config` 和 `_loss_config(loss, 标签)`；Transformer 比较登记规格和 `loss_config`；AC 要求 `schema == "v7_vjepa2_ac_checkpoint_v1"` 且规格等于 `ac_baseline.spec_for(condition)`。任何一项不等就报错。
5. `training._check_model_state`（名字、形状、精度），然后 `load_state_dict(strict=True)`、`model.eval()`。不读优化器状态。
6. 来源里的数据字段必须和当前一致：`evidence_role`、`split`、`paper_sha256`、`upstream_commit`、`encoder_checkpoint_sha256`、`preprocess_sha256`、`feature_manifest_sha256`、`protocol_sha256`。“当前值”取法：重新组检查点保存时的那一批训练批（第 `max(0, step−1)` 批，和 `comparison.run_prepared_condition`、`resume.prepare_condition` 的组法相同），取它的 `provenance` 里的这 8 个字段，所以比的是同一份特征、同一个协议在新树里算出来的值。`source_files_sha256` 原样记录，不核对。

补跑自己的检查点也走同一个函数（这时源码 SHA 本来就一致，照样只记录）。

### 4.3 等价核对

D4 在同样的窗口上重算 v7 的三项标准指标，和被读检查点所在任务自己的评测记录（`evaluation/<条件>/rows/*.json`）逐条比较：

- `future_jepa_distance`：16×16，T ∈ {1, 2, 3, 4, 8, 12}
- `tcd_normalized`：16×16，同样的 T
- `scd_normalized`：16→8 和 16→12，同样的 T

容差 |新 − 旧| ≤ 1e-6 · max(1, |旧|)。报告每项的条数、逐位相同的条数、最大绝对差和最大相对差。同一台机器、同样的设备和线程数下多半逐位相同，但 GPU 和多线程 CPU 上的规约顺序可能让末位不同（07 §5.2），所以判据只用容差，逐位相同的条数只作报告；超出容差任务就失败，不写 COMPLETE.json，说明新树读 v7 检查点或复制来的计算路径有问题，要先查清楚。identity 也做这项核对（参照是 v7 identity 任务的记录）。

参照行的要求：被读任务的 `evaluation/<条件>/` 必须有 `COMPLETE.json`、没有 `FAILED.json`；`registration.json` 的 `queries` 必须等于当前 `evaluation.build_index()`（每个窗口 28 条）；参照行按 (kind, 轨迹, 起点, 时长, 网格) 对齐。缺行、重复行、条件或种子不同、状态不是 ok、参照值不是有限数，任务都失败。

## 5. 截断间隙

论文说 K = 16 时第 K 个特征值位于谱的稠密部分以下。v7 的谱诊断算了 λ_{K+1} − λ_K，但没有保存。

- 图只由编码器特征、Q（种子 1703）、σ_x 和本场景的 σ_z 决定，和条件、种子无关，所以**每个场景只跑一次**（`run.py spectral-gap`，CPU）。
- 对 400 个测试窗口的起点帧，在 16×16、8×8（R z）、12×12（R z）上用 po_jepa_full 的 `GraphSpectrum` 求最低 16 个特征对，记录 λ_16、λ_17、`cutoff_gap`、前 16 个模态相邻间隙的中位数，以及两者之比。
- 报告每个场景、每个网格上这些量的 5、25、50、75、95 分位数。
- 预先定好的判断规则：每个场景、每个网格上比值的中位数都大于 1，论文保留原句；否则改成“K = 16 is fixed in advance and shared across grids”。
- 比值和规则的读法：每个窗口、每个网格算 ratio = `cutoff_gap` / median(15 个相邻间隙)，再取 400 个窗口 ratio 的中位数（`torch.quantile` 线性插值，偶数个时就是中间两个的平均）；“大于 1”是严格大于；某个窗口的相邻间隙中位数为 0 时 ratio 记 None、计入 `undefined_ratio`、不进分位数，全是 None 时判“不保留原句”（Jerry 2026-11-08 定：按推荐做法）。
- fixed_fourier 的截断间隙由公式给出，恒为 0（08 §4.3），在同一张表里列出。

## 6. 编码器核对

官方 AC 权重 `vjepa2-ac-vitg.pt` 里自带一份编码器。如果它和我们用的 `vjepa2-vitg-fpc64-256.pth` 的 `target_encoder` 不是同一份，零样本 AC 吃到的就是它没见过的特征。

- `tools/encoder_identity.py`，CPU 任务，A0 段。
- 两边去掉 `module.`、`backbone.` 前缀后按名字配对，比较 AC 权重里的 `encoder`（和 `target_encoder`，如果有）与 fpc64-256 的 `target_encoder`（和 `encoder`）。
- 报告：配对上的张量数、按存储精度逐位相同的个数、最大绝对差、两边各自配不上的键、精度不同的键。全部配对、全部逐位相同才判“同一份”。
- 结论取哪一对：主比较是 AC `encoder` 对 fpc64-256 `target_encoder`（v7 训练和 D4 用的就是 `target_encoder`），`verdict` 取主比较的（`same`/`different`/`missing_keys`），其余配对也算、写进 `comparisons`；逐位相同按存储字节比，不转精度；前缀 `module.`、`backbone.` 各去一次，去掉后重名报错。论文里写的那句话以 `verdict` 为准（Jerry 2026-11-08 定：按推荐做法）。
- 结果写进论文。

## 7. 输出格式

`run.py d4 --output X` 的任务目录是 `X/run/job/`（下面的 `<输出>` 就是 `X/run/job`），和 v7 的 `run`/`evaluate` 相同：`X/console/` 是 `tools/capture.py` 的输出，`X/run/environment.json` 和 `X/run/COMPLETE.json`（`eval_cli` 写；出错时是 `FAILED.json`）的位置也和 v7 相同，`v7jobs._job_dir(X)` 能找到任务目录：

```
<输出>/job.json            # 条件、场景、种子、步数、被读的任务目录
<输出>/source.json         # 被读检查点的 best 记录、文件 SHA、v7 源码 SHA（原样）、核对过的数据字段
<输出>/rows/<序号>.json     # 每个窗口一条：窗口、§3.2–§3.6 的全部量、等价核对的新旧值
<输出>/long/<轨迹>.json     # 每条轨迹一条：§3.7 的量
<输出>/report.json         # 条数、等价核对汇总、三角不等式违反数、各量的轨迹宏平均
<输出>/COMPLETE.json       # report.json 的 SHA；只有全部核对通过才写
```

完成标志因此写在 `X/run/COMPLETE.json` 和 `X/run/job/COMPLETE.json` 两处，S17 复制的运行器认 `run/`，v7 的写法认 `run/job/`（07 §2.8）。`run.py spectral-gap --output X` 同样：`X/run/job/` 下写 `job.json`、`report.json`、`COMPLETE.json`（16 §5）。

所有 JSON 用 `cache._write_json`（排好序的键、固定格式），同样的输入写两次字节相同。

## 8. DROID 评测轴

代码在 `src/po_jepa/droid/evaluate.py`（S11）。§3 的指标定义（两种度量、分割滚动 Φ_Π、归一化）DROID 上原样使用，只把“tick”换成“周期”，窗口换成 06 §3 的冻结窗口。

### 8.1 任务和读检查点

- 每个（条件，种子）一个评测任务 `eval/<条件>/seed<种子>`，两个测试集 `id_test`、`ood_test` 都在里面做，离线规划（11）也在里面做。identity 和 vjepa2_ac_zeroshot 不训练，各一个任务 `eval/<条件>`。共 8 × 3 + 2 = 26 个。
- 训练过的条件读 `selection.json` 选出的检查点（09 §4.4）。读法和 §4.2 一样放宽：核文件 SHA、步数、配置快照、`model_state`，以及来源里的数据字段（`feature_manifest_sha256` 等），`source_files_sha256` 只记录，因为评测代码可能在训练之后才修好。vjepa2_ac_zeroshot 直接载入官方权重并记 SHA。
- 评测全程 eval 模式、`no_grad`，前后核对参数没变（§1）。
- `action.source: commanded` 的配置里，跳过 `passed_commanded` 为假的回合，报告跳过的回合数和窗口数。

### 8.2 记号

窗口 w =（回合，时长 d，起点行 s），z 是第 s 行特征，z*_T 是第 s+T 行，u 是第 s 到 s+T−1 个周期的动作（01 §5）。d(a, b) 是 JEPA 距离（mean |LN(a) − LN(b)|）。未来误差一律用一次直接前向 `model(z, u, T, grid)`，所有条件相同（09 §4.4）。R_8 是 2×2 平均，R_12 是双线性（v7 `build_spatial_transfer`）。

每个窗口只算和它的时长 d 有关的量（T = d）：

| 轴 | T | 网格 | 量 |
|---|---|---|---|
| 同分布时长 | 1、2、4、8 | 16×16 | 未来误差；`tcd_ln`、`tcd_normalized`（§3.2 和 v7 的定义）；`delta_act`（§8.3） |
|  | 1、2、4、8 | 8×8 | 未来误差 d(Φ^8_T(R_8 z), R_8 z*_T)；`scd_ln`、`scd_normalized`（16→8） |
| 网格留出 | 1、2、4、8 | 12×12 | 未来误差 d(Φ^12_T(R_12 z), R_12 z*_T)；`scd_ln`、`scd_normalized`（16→12） |
| 时长留出 | 3、12、16 | 16×16 | 未来误差；`tcd_ln`、`tcd_normalized`；误差分解（§3.6，3 = 1+2、12 = 8+4、16 = 8+8） |
| 时长和网格同时留出 | 3、12 | 12×12 | 未来误差 |
| 长时程 | 24、32 | 16×16 | 直接预测和 8 步滚动（3 段、4 段）的未来误差，两者之间的 pd（§3.7）；只用够长的回合（06 §3） |
| 分割差异 | 4、8 | 16×16、8×8 | §3.3 的 7 个分割、§3.4 的跨网格 xpd（n = 2、4）、§3.5 的不等分 TCD |

identity 给出每一项的参照（未来误差是 d(z, z*_T)，所有差异为 0）。

### 8.3 动作干预 Δ_act

Δ_act = d(Φ_T(z; u_donor), z*_T) − d(Φ_T(z; u), z*_T)，u_donor 是替身窗口（06 §3，同一测试集、同一时长、别的回合）的动作，起点和目标仍是本窗口的。POJEPA 类走 v7 `metrics.action_intervention_margin`，其余条件走两次前向（v7 `evaluation.point_metrics` 的同一分支）。AC 的起点状态用本窗口的，状态再按替身动作积分（09 §6.2 的包装，以替身动作为参照）。

### 8.4 原始帧率

只用附加回合（每个测试集前 200 条，06 §3）的 s5 特征，起点和目标都从 s5 文件读（05 §4）。起点帧 f0 = 15s（s 取附加回合里 d = 4 的窗口的起点），s5 行号 j0 = 3s。

- **单步**：δ ∈ {1/3, 2/3, 4/3} 个周期（5、10、20 帧）。目标是 s5 第 j0 + 3δ 行。控制值 u = `poses_to_diffs([s_{f0}, s_{f0+15δ}])` / δ 再除尺度（01 §5.5），一段 [0, δ)，时长 δ（float64，目标秒 = 起点秒 + δ，按这个式子算，保证端点核对精确成立）。
- **滚动**：步长 δ ∈ {2/3, 4/3}，滚 4/δ 步（6 步、3 步）到 4 个周期（60 帧，s5 第 j0 + 12 行）。第 k 步的输入是上一步的原始预测，控制值按这一步自己的起止帧算。
- **参照**：同一起点在 s5 上的 δ = 1 单步（15 帧）、δ = 1 滚 4 步、以及一次直接预测 4 个周期，让“步长变了”和“预测得远”两种影响分得开。
- **AC**：一步就是重采样后的一帧间隔（5 帧），动作是这 5 帧的物理位姿差，不除以 δ（01 §5.4）。时长 δ 的一次调用 = 连续 3δ 个这样的小步，状态从 `state_all[f0]` 起按动作积分；滚动时每次调用都从上一次的输出重新开始上下文，和其他模型的复合方式一致。
- 指标：每一项的未来误差 d(预测, 目标)；identity 作参照。

### 8.5 原始分辨率

只用附加回合里 d ∈ {1, 2, 4, 8} 的窗口。起点和目标都换成 r192（12×12 = 144 个 token）或 r320（20×20 = 400 个 token）的同一行（05 §4），网格用 `regular_grid(12, 12)` 或 `regular_grid(20, 20)`，模型直接在这个网格上预测，不做重采样。指标是未来误差；同一批窗口上 256 输入的 16×16 未来误差一起列出作对照。PO-JEPA 类在 144 或 400 个 token 上建图求谱；Koopman 的基底按 08 §5.3 换到这个网格；AC 按 v7 的做法把 RoPE 位置缩放到 16×16。

### 8.6 AC 的特殊处理

- 非整数时长用相邻两步的线性插值，半整数时就是 v7 的平均（09 §6.2）。`tcd_ln` 在 T = 1、3 时要用到 0.5、1.5 的半程，这两个时长上 AC 的 TCD 部分量的是插值本身，汇总表里单独列出（论文同样写明）。
- AC 只参加 LN 空间指标的跨模型比较，原空间的 `tcd_normalized`、`scd_normalized` 标为不可比（§3.8）。
- `true_state` 时，所有调用都用 09 §6.2 的绑定起点状态的包装；分割滚动里后一段的起点状态由前面各段的动作积分得到。

### 8.7 成败分开报

每个逐窗口的量都按回合的 outcome（success / failure，取自 `index/episodes.jsonl`）分开汇总一次，再合起来汇总一次（13 §8）。

### 8.8 截断间隙

和 §5 相同的做法，每个配置跑一次 CPU 任务 `eval/spectral-gap`：两个测试集里 d = 1 的全部窗口的起点帧，在 16×16、8×8、12×12 上用 σ_z 的 DROID 标定值建图，报告 λ_16、λ_17、`cutoff_gap` 和相邻间隙中位数之比的分位数，判断规则同 §5。

### 8.9 输出

```
<run-root>/<name>/eval/<条件>/seed<种子>/
  job.json                 # 条件、种子、读的检查点（路径、SHA、步数）、配置 SHA
  source.json              # 来源里核对过的数据字段；源码 SHA 原样记录
  <测试集>/rows/<序号>.json  # 每批一份：每个窗口的编号、回合 outcome、§8.2–§8.5 的全部量
  <测试集>/frame_rate.json   # §8.4，每个起点一条
  <测试集>/resolution.json   # §8.5，每个窗口一条
  <测试集>/planning/…        # 11
  report.json              # 条数、跳过数、三角不等式违反数（必须为 0）、各量的回合宏平均
  COMPLETE.json
```

中断后重跑同一条命令：`rows/` 下已有的文件直接跳过（每个文件先写临时文件再改名，存在就是完整的；内容只由窗口文件和检查点决定），从缺的那批接着做。`report.json` 写之前核对每个窗口恰好出现一次。
