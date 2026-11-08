# 08 条件

## 1. 条件总表

### 1.1 仿真（v7 的 6 个加补跑的 5 个）

| 条件 | 来源 | 模型 | λ_t | λ_s | 损失标签 | 每次更新的 Φ 次数 | 每次更新的 p_out 次数 | 参数量（m = 2） |
|---|---|---|---|---|---|---|---|---|
| po_jepa_full | v7 | POJEPA + GraphSpectrum | v7 标定 | v7 标定 | full | 5 | 10 | 215,778 |
| po_jepa_no_consistency | v7 | 同上 | 0 | 0 | no_consistency | 5 | 10 | 215,778 |
| matched_transformer | v7 | TokenTransformer（登记规格，残差读出） | 0 | 0 | no_consistency | 1 | 1 | v7 登记值（与 PO-JEPA 相差 < 1%） |
| identity | v7 | IdentityPredictor | — | — | — | 不训练 | — | 0 |
| vjepa2_ac | v7 | 官方规模 AC，从官方权重微调 | 官方损失 | — | — | — | — | 约 3 亿 |
| vjepa2_ac_small | v7 | 同结构缩小，从头训练 | 官方损失 | — | — | — | — | v7 登记值（相差 < 1%） |
| po_jepa_time_only | 补跑 | 与 full 完全相同 | v7 full 的 λ_t | 0 | time_only | 5 | 10 | 215,778 |
| po_jepa_space_only | 补跑 | 与 full 完全相同 | 0 | v7 full 的 λ_s | space_only | 5 | 10 | 215,778 |
| transformer_consistency | 补跑 | 与 matched_transformer 完全相同 | 补跑标定 | 补跑标定 | full | 6 | 6 | 同 matched_transformer |
| fixed_fourier | 补跑 | POJEPA + FourierSpectrum | 补跑标定 | 补跑标定 | full | 5 | 10 | 215,778 |
| koopman | 补跑 | KoopmanPredictor，K = 76 | 0 | 0 | no_consistency | 1 | 2 | 215,744 |

“v7 标定”指 v7 的 calibrate 任务写出的 `configs/calibration/<场景>-<条数>-u<步数>.yaml`，补跑时原样导入（16 §3.1）；“补跑标定”见 16 §3.2。σ_z 所有 PO-JEPA 类条件都用导入的 v7 值（fixed_fourier 不建图，用不到 σ_z，但它的 buffer 里存的是同一个值，保证初始权重和 full 相同）。

### 1.2 DROID（第二期以后）

8 个训练条件：po_jepa_full、po_jepa_no_consistency、fixed_fourier、koopman、matched_transformer、transformer_consistency、vjepa2_ac、vjepa2_ac_small；2 个不训练的对照：identity、vjepa2_ac_zeroshot。单项消融只在仿真上做。DROID 上 m = 7，Koopman 的 K = 72（§5.6）；PO-JEPA 的参数量按 7 维动作重新数（236,898）。

DROID 复用 S10 写的 fixed_fourier 和 koopman 构造函数，所以这两个构造函数要把模型配置块（projection/graph/dynamics/integrator，Koopman 另有 `KoopmanConfig`）和初始化种子作为显式参数：仿真调用时传 `load_protocol()["model"]` 和 `check.model_seed()`，结果和 §4.2、§5.2 写的完全相同；DROID 调用时传 09 §2.1 的配置块和运行种子。Transformer 和 AC 在 DROID 上的规格见 09 §2.2。

### 1.3 初始权重的关系（补跑要核对）

同一场景、同一种子下：

- po_jepa_time_only、po_jepa_space_only、fixed_fourier 的 `initial_model_sha256`（`v7jobs._model_hash`）必须等于 v7 那次 po_jepa_full 的值。fixed_fourier 也能相等，因为 `FourierSpectrum` 不加参数和 buffer，构造时消耗的随机数和 `GraphSpectrum` 相同。
- transformer_consistency 的初始哈希必须等于 v7 那次 matched_transformer 的值。
- koopman 没有可对照的 v7 条件，只登记自己的值。

## 2. 单项一致性消融（po_jepa_time_only、po_jepa_space_only）

- 模型、初始化、查询序列、学习率调度、梯度裁剪、验证选点和 po_jepa_full 完全相同，只是一个权重为 0。
- 非零的那个权重直接用导入的 v7 full 标定值，不重新标定。理由：v7 的 λ_t、λ_s 是两项分别按“加权一致性梯度的中位数 = ρ × JEPA 梯度的中位数”（ρ = 1）算出来的，互不依赖，单开一项时这个比例的含义不变。
- 一致性预热、梯度占比监控和提前检查照常进行，只检查权重 > 0 的那一项（07 §2.4）。占比中位数低于 `monitoring.min_share`（0.005）就提前停止，和 full 的规则相同。
- 单步用 `training.train_step(..., condition="time_only")` 或 `"space_only"`：仍算 5 次 Φ 和两个一致性项，权重为 0 的那项乘 0 后进总损失。

## 3. transformer_consistency

- 模型就是 v7 登记的 matched_transformer（同一个 `TokenTransformer` 规格、同一个初始化种子），在 JEPA 损失之外加时间和空间一致性损失。
- `generic_consistency.py` 提供 `temporal_predictions(model, raw_z, controls, tau, sigma, grid)` 和 `spatial_predictions(model, raw_z, controls, duration, transfer)`：逐行照抄 `losses.py` 里的同名函数，只去掉 `isinstance(model, POJEPA)` 这一条检查，改成要求：`model` 是 `nn.Module`；`forward` 恰好 4 个没有默认值的位置参数，按位置是 `(raw, controls, duration, grid)`（只核个数和种类，不核名字：POJEPA 的第一个参数叫 `raw_z`，`TokenTransformer` 的叫 `raw`）；有正整数属性 `encoder_dim`（`spatial_predictions` 本来就读它）。返回同样的 `TemporalPredictions`、`SpatialPredictions`，交给 `losses.py` 里原有的 `temporal_consistency_loss`、`spatial_consistency_loss`、`weighted_total_loss`，这三个函数不改。
- **JEPA 项和 matched_transformer 逐位相同。** PO-JEPA 的 `train_step` 把 JEPA 损失放在时间一致性的长支 direct 上，direct 用的是切开再拼接的动作 u1⊕u2。对 PO-JEPA 这和原动作等价，但 Transformer 每个动作段生成一个 token，拼接后在 τ 处多一个段边界，输入就变了。为了让两个 Transformer 只差一致性损失，transformer_consistency 的 JEPA 项另做一次和 matched_transformer 完全相同的前向（原动作、原时长），不复用 direct。
- 单步（`addon.conditions.transformer_consistency_step`）：
  1. 计步检查、`_loss_config(loss, "full")`、Transformer 参数表（`comparison._tf_optimizer`）、批检查（`comparison._tf_batch`）、显式训练模式。
  2. `zero_grad(set_to_none=True)`。
  3. `prediction = model(batch.raw_z, batch.controls, batch.duration, batch.transfer.source)`；`jepa = jepa_distance(prediction, batch.raw_target)`（和 matched_transformer 的单步相同）。
  4. `temporal = temporal_predictions(model, batch.raw_z, batch.controls, batch.tau, batch.sigma, batch.transfer.source)`（3 次前向）；`time = temporal_consistency_loss(temporal)`。
  5. `spatial = spatial_predictions(model, batch.raw_z, batch.controls, batch.duration, batch.transfer)`（2 次前向：16×16 和 8×8）；`space = spatial_consistency_loss(spatial)`。
  6. `weighted_total_loss(jepa, time, space, consistency_scale=consistency_warmup_scale(loss, completed))`。
  7. 六项损失都有限才 backward；每个参数的梯度都要存在且有限；`clip_gradients`（1.0）；`scheduled_step`；更新后参数和 AdamW 状态有限。
- 每次更新 6 次前向、6 次 p_out 读出。一致性预热的第 0 步（权重 × 0）梯度和 matched_transformer 的第 0 步逐位相同，测试用这一点核对。
- λ 用自己的补跑标定（16 §3.2），方法和 v7 相同：只用 JEPA 训练 300 步，再在接下来 32 个训练批上量三项梯度范数，取中位数，λ_c = ρ·median|∇L_JEPA| / median|∇L_c|，ρ = 1。梯度范数用 `addon.calibration.generic_gradient_norms`：同一批、和单步相同的 6 次前向，三项分别对全部 Transformer 参数求导。
- 检查点格式和 matched_transformer 相同，`loss_config` 存 `"full"` 标签的快照。

## 4. fixed_fourier

### 4.1 是什么

把 PO-JEPA 的自适应图谱换成固定的二维余弦（DCT-II）基底，图的特征值换成同一个空间核在连续区域上的特征值。模态动力学、读出、损失、λ 的标定方法都不变。它衡量“随场景变化的几何”值多少。

### 4.2 代码结构

- `FourierConfig(GraphConfig)`：冻结的 dataclass，只多一个字段 `basis: str = "dct2"`，`__post_init__` 先调父类，再要求 `basis == "dct2"`。因为多了这个字段，`training._model_config` 存下的 `graph` 快照和 full 不同，fixed_fourier 的检查点不会被误当成 full 读入。
- `FourierSpectrum(GraphSpectrum)`：不加参数、不加 buffer，只重写 `spectrum_for`。`affinity`、`normalized_laplacian` 不调用（它们仍然存在，`q`、`sigma_z` buffer 照常保存）。
- 构造：和 `checks.build_full_model` 相同的顺序和种子，只把 `GraphSpectrum(GraphConfig(**m["graph"]))` 换成 `FourierSpectrum(FourierConfig(**m["graph"], basis="dct2"))`。`build_full_model` 自身要求 `config == load_protocol()`，不改它，在 `fourier.py` 里另写一个同样结构的构造函数 `build_fixed_fourier(配置块, seed, device)`（不放 `runtime/addon/`，DROID 也要用；Koopman 的 `build_koopman` 同样放在 `koopman.py`）。

### 4.3 基底

对 H×W 网格（token 序号 n = y·W + x）：

- 模态 (p, q)，p ∈ [0, W) 是 x 方向频率，q ∈ [0, H) 是 y 方向频率：
  u_pq(x, y) = c_p^W · c_q^H · cos(π p (x + 0.5) / W) · cos(π q (y + 0.5) / H)，
  其中 c_0^N = √(1/N)，c_p^N = √(2/N)（p > 0）。各列在 N = H·W 个 token 上正交归一。
- 排序：按 (p² + q², max(p, q), p) 升序，取前 K = 16 个。
- 前 16 个模态和对应的特征值（σ_x = 0.125，由 §4.4 的公式算出；可作测试金标准）：

| 序号 | (p, q) | p²+q² | λ |
|---|---|---|---|
| 1 | (0,0) | 0 | 0 |
| 2, 3 | (0,1), (1,0) | 1 | 0.037819 |
| 4 | (1,1) | 2 | 0.074209 |
| 5, 6 | (0,2), (2,0) | 4 | 0.142910 |
| 7, 8 | (1,2), (2,1) | 5 | 0.175325 |
| 9 | (2,2) | 8 | 0.265397 |
| 10, 11 | (0,3), (3,0) | 9 | 0.293179 |
| 12, 13 | (1,3), (3,1) | 10 | 0.319911 |
| 14, 15 | (2,3), (3,2) | 13 | 0.394191 |
| 16 | (0,4) | 16 | 0.460359 |
| （第 17 个，不用） | (4,0) | 16 | 0.460359 |

- **K = 16 正好切开一对简并模态**：(0,4) 进、(4,0) 不进，所以 fixed_fourier 的截断间隙在每个网格上都是 0。保持 K = 16 是为了和 PO-JEPA 的模态数一致；论文里写明排序规则和这一点。
- 符号：直接用上面的余弦，不做 GraphSpectrum 那种按最大分量翻符号。同一个连续函数在不同网格上取样，符号天然一致；2×2 平均池化把 16×16 的 u_pq 映成 ½·cos(πp/32)·cos(πq/32)·u_pq^{8×8}（p, q < 8；½ 是两种网格归一化常数之比），所以跨网格时模态一一对应，方向不变。
- 网格要求：K ≤ H·W，且每个方向的频率范围够用（8×8、12×12、16×16、20×20 都满足）。不满足时 `spectrum_for` 报错：坐标不是规则格点中心（和 `regular_grid(H, W)` 的差超过 1e-12）报 `fixed Fourier basis requires regular cell-center coordinates`；前 min(k+1, N) 个模态在这个网格上表示不了报 `grid does not resolve the first k+1 Fourier modes`（第 k+1 个用于 `cutoff_gap`）。

### 4.4 特征值

λ_pq = 1 − exp(−σ_x² π² (p² + q²) / 4)，σ_x 取 `sigma_x` buffer 的值（0.125，float32 转 float64 是精确的）。

推导：v7 的空间核是 k(r) = exp(−|r|²/σ_x²)（`spectral.affinity`，坐标在 [0,1]²），二维傅里叶变换 ∝ exp(−σ_x²|ξ|²/4)。密度均匀、远离边界时，归一化拉普拉斯 I − D^{−1/2} W D^{−1/2} 在频率 ξ 上的特征值是 1 − k̂(ξ)/k̂(0)；单位正方形上的 Neumann 余弦模态 cos(πp x)cos(πq y) 对应 ξ = (πp, πq)。这个值和网格大小无关，所有网格用同一组。

### 4.5 `spectrum_for` 的输出

和 `GraphSpectrum.lowest_eigenpairs` 返回的 `SpectralResult` 同类型、同形状、同精度：

| 字段 | 内容 |
|---|---|
| `basis_cpu` | CPU float64 [B, N, K]，§4.3 的基底在 batch 维复制 |
| `eigenvalues_cpu` | CPU float64 [B, K]，§4.4 的值 |
| `basis`、`eigenvalues` | 上面两项转成模型设备 float32，无梯度 |
| `diagnostics.residual_norms` | 全 0（这组基底按定义就是它的特征向量） |
| `diagnostics.orthogonality_error` | 由 `basis_cpu` 实算 ‖ΨᵀΨ − I‖_F |
| `diagnostics.consecutive_gaps` | 前 K 个公式特征值的相邻差，[B, K−1] |
| `diagnostics.cutoff_gap` | 按同一排序的第 K+1 个减第 K 个公式特征值；K = N 时为 None（和 GraphSpectrum 相同）。K = 16 时恒为 0 |
| `diagnostics.sign_pivot_indices` | 每列绝对值最大的第一个 token 的序号（只记录，不翻符号） |

`source` 只用来核对形状、batch 大小和设备（和 `affinity` 的输入检查相同），数值不参与计算，因此对 encoder 特征的梯度和 full 一样在求谱处截断。

## 5. koopman

### 5.1 是什么

有限维的模态系数向量，在一个可学的空间基底上展开；每对系数在与上下文无关的阻尼旋转下演化，加一个线性动作项，按段精确积分；模态之间没有相互作用；读出和 PO-JEPA 一样是残差形式。只用 JEPA 损失训练。基底大小按参数量和 PO-JEPA 匹配。灵感来自 Koopman Dreamer（li2026koopman）。它检验的是：在谱稳定性之外，函数空间的结构本身有没有用。

### 5.2 参数

| 参数 | 形状 | 初始化 |
|---|---|---|
| `p_in` | Linear(1408 → 64)，带偏置 | `xavier_uniform_(gain=1)`，偏置 0（和 POJEPA 的投影层相同） |
| `p_out` | Linear(64 → 1408)，带偏置 | 同上 |
| `basis` W | [256, K] | 16×16 网格上前 K 个 DCT-II 模态（§4.3 的公式和排序，确定性，不用随机数） |
| `raw_gamma` | [K, 32] | 常数 softplus⁻¹(0.1) = log(e^{0.1} − 1) ≈ −2.252168 |
| `omega` | [K, 32] | 0 |
| `action_gain` B | [K, 64, m] | Normal(0, 0.001) |

构造时 `torch.random.fork_rng(devices=[])` 并用 `check.model_seed()` 设种子（和 Transformer 相同），按上表顺序创建；只有 `p_in`、`p_out`、B 消耗随机数。参数量 = 2·1408·64 + 64 + 1408 + K·(256 + 64 + 64m) = 181,696 + K·(320 + 64m)。

### 5.3 前向 Φ(Z, u, T; 网格 h)

输入输出合同和 `TokenTransformer.forward` 相同：raw Z [B, N_h, 1408] 模型 float32，`ActionSegments`，duration 为 CPU float64 [B]（≥ 0，单位是控制周期），`GridSpec`。

1. **基底换到当前网格。** 16×16 上 W_h = W。其他网格用和 token 完全相同的重采样：`transfer = build_spatial_transfer(regular_grid(16,16), target_height=H, target_width=W)`，把 W 当成 [1, 256, K] 的“特征”做 `apply_spatial_transfer`，得到 W_h [N_h, K]（梯度传回 W）。要求 `transfer.target.coords` 和输入 grid 的坐标逐位相同，否则报错；16×16 也核（输入坐标和 `regular_grid(16,16)` 逐位相同），坐标顺序不同的 16×16 网格不能被静默接受。8×8 是 2×2 平均，12×12 是双线性，DROID 的 20×20 是双线性放大。
   - **DROID 原始分辨率（10 §8.5，S11 改）**：那里的网格是 `regular_grid(12,12)`、`regular_grid(20,20)`，不是重采样的目标。12×12 的双线性目标坐标和规则格点只差约 1e-16，20×20 放大时 R 在边缘按 edge 截断，目标坐标的边缘一圈和格点中心差 0.00625。基底仍是 W_h = R W（值不变，等于 W 在格点中心的双线性插值、边缘截断），只把坐标核对改成“输入坐标和 `regular_grid(H, W)` 的差 ≤ 1e-12”。仿真用到的三种网格两种规则都能过。
2. **分析（面积加权内积）。** X = P_in(Z)，c_0 = (256 / N_h) · W_hᵀ X，形状 [B, K, 64]。16×16 上就是 Wᵀ X。粗网格上 W_h 的秩可能小于 K（8×8 只有 64 个 token，而 K = 76），这个做法不需要求逆。
3. **动力学。** 64 个通道按 (2i, 2i+1) 分成 32 对。第 k 个模态第 i 对写成复数 z = c[..., 2i] + i·c[..., 2i+1]，λ_ki = −γ_ki + i·ω_ki，γ = softplus(raw_gamma) > 0。动作项 f = B·u ∈ R^{K×64}，同样配对成复数 φ。
4. **按段精确积分。** 先 `validate_action_segments(controls, query_duration=duration)`。对每个样本按时间顺序取有效段 j，段长 Δ_j = max(0, min(end_j, T) − max(start_j, 0))，段内动作 u_j 为常数：
   z ← e^{λΔ_j} z + Δ_j · φ1(λΔ_j) · φ_j，其中 φ1(w) = (e^w − 1)/w。
   - e^{λΔ} 和 Δ·φ1(λΔ) 用 complex128 算（γ、ω 转 float64，保留梯度），乘到 float32 系数前转回 complex64。
   - |w| < 1e-4 时 φ1(w) 用 1 + w/2 + w²/6，否则直接算 `expm1(w)/w`；两个分支都先换成安全参数再用 `torch.where` 选，w = 0 处反向有限。
   - 无效（padding）段 Δ = 0，等于不变。各段 Δ 之和与 T 的差超过 1e-12 就报错。
   - T = 0 时没有有效段，c_T 与 c_0 逐位相同。
5. **残差读出。** Ẑ = Z + P_out(W_h c_T) − P_out(W_h c_0)。每次前向调用 p_out 两次。T = 0 时 Ẑ = Z 逐位成立。

### 5.4 训练

- 只用 JEPA 损失，单步和 matched_transformer 的单步结构相同：一次前向 `model(batch.raw_z, batch.controls, batch.duration, batch.transfer.source)`（16×16），`jepa_distance` 对 `batch.raw_target`，`_loss_config(loss, "no_consistency")`，梯度全部存在且有限，裁剪 1.0，`scheduled_step`。
- 优化器：AdamW，lr 由调度给出（峰值 1e-3），betas (0.9, 0.999)，eps 1e-8，weight decay 0，全部参数恰好一次。
- 训练只在 16×16 上；8×8、12×12 上的预测只在评测时出现。

### 5.5 不做的事

不加模态间相互作用、不加非线性项、不随上下文改变 γ、ω、不加一致性损失、不学习时间缩放。

### 5.6 K 怎么定

K = 使 |参数量(K) − PO-JEPA 参数量| 最小的正整数，相等时取较小的 K。

| 场景 | m | PO-JEPA | K | Koopman | 相差 |
|---|---|---|---|---|---|
| 仿真 | 2 | 215,778 | 76 | 215,744 | 34（0.016%） |
| DROID | 7 | 236,898 | 72 | 236,992 | 94（0.040%） |

代码里按规则算 K，写进检查点的 `model_config`；`validate_registration` 对 koopman 核 K 等于对登记的 PO-JEPA 参数量按规则取的值、实际参数量等于公式。v7 的 `preregister()` 不改，`check-addon` 的报告里另记 K。测试断言仿真得 76、DROID 得 72。

参数量的口径和 v7 相同，含 `p_out.bias` 的 1,408 个参数。PO-JEPA 和 Koopman 都是残差读出 P_out(a) − P_out(b)，偏置精确相消、梯度恒为 0，所以这 1,408 个参数从不训练；matched_transformer 的读出是 Z + P_out(·)，偏置会训练。按可训练参数算，PO-JEPA 比 Transformer 少 1,408 个（约 0.65%），对基线有利。

### 5.7 检查点

字段和 v7 PO-JEPA 检查点（`training.save_training_checkpoint`，`schema_version` 1）相同，`condition` 为 `"koopman"`，`scheduler_state`、`scaler_state` 为 None（`schema_version`、`completed_steps`、`condition`、`model_state`、`optimizer_state`、`optimizer_param_names`、`model_config`、`loss_config`、`optimizer_config`、`provenance`、`rng_state`、`sampling_state`、`model_training`、`scheduler_state`、`scaler_state`、`runtime`），`model_config` 是 `KoopmanConfig` 的快照：

```python
@dataclass(frozen=True)
class KoopmanConfig:
    modes: int            # K
    action_dim: int       # m
    encoder_dim: int = 1408
    dynamic_dim: int = 64
    basis_grid: tuple = (16, 16)
    basis_init: str = "dct2"
    gamma_init: float = 0.1
    action_gain_std: float = 0.001
    phi1_series_threshold: float = 1e-4
    readout: str = "residual"
```

读回时核对快照逐项相等、`model_state` 的名字形状精度相同，再 `load_state_dict(strict=True)`。
