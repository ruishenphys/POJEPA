# PO-JEPA v5（在 v4 第二版基础上，撤回逐路径 LN 的一致性度量）

**v5 只改了一处方法：一致性缺陷的度量。** 其余与 v4 第二版完全相同，下文 v4 的说明仍然有效。

- **原因：** v4 第一次 GPU 运行中，P3 的逐路径 LN 让模型可以把两条路径的共同分量放大约 21 倍，一致性缺陷趋于 0，JEPA 却升到约 0.7，即"放大捷径"。详见 `v4结果分析.md` 第 4 节。
- **新度量：** δ_h(A,B) = mean_(N,E)(A−B)² / s²。其中 s² = Z_t 在 (N,E) 上的方差，逐样本计算并停梯度；空间项用细网格输入的方差。两条路径除以同一个数，所以：
  - 放大输出只会让 δ 变大；
  - 残差读出里始终含有 Z，缩小也不行；
  - d_h = sqrt(δ_h) 是真正的度量（范数差一个常数倍），P4 对定理的弱化可以撤回。
- **配置：** `configs/train/v4.yaml` 中 `loss.consistency_metric: input_scale`。`ln_mean`、`raw_sum` 仍然可选，用于对照。
- **代码：** `losses.py`（`input_scale`、`_consistency_difference`，TemporalPredictions / SpatialPredictions 新增 `input_scale` 字段）。
- **评价：** evaluation 新增 `tcd_normalized` = tcd_rms / s 和 `scd_normalized` = scd_rms / s，与训练用的是同一个度量。"一致性与精度的取舍"标签改为按归一化指标判断；原始范数的 `tcd_frobenius` 等列保留，作为补充。
- **λ 必须重新标定。** 度量变了，v4 标定的 λ 不能沿用，`v4.yaml` 里的 time_weight / space_weight 仍是占位值。
- **测试：** `tests/test_v4.py` 新增 4 项，CPU 全量 1746 项通过。新增测试覆盖：
  - 放大捷径在 ln_mean 下使缺陷趋于 0，在 input_scale 下缺陷不变；
  - 显式公式；
  - 尺度等变性；
  - 空间项使用细网格尺度；
  - 覆盖层默认值。
- **学习率衰减（导师 2026-10-29）：** 学习率在 2000 次更新内从 1e-3 余弦衰减到 1e-4，之后保持 1e-4。
  - 配置：`loss.lr_min: 0.0001`、`loss.lr_decay_updates: 2000`；lr_decay_updates 设为 0 就恢复恒定 1e-3。
  - 实现：`training.scheduled_lr` / `scheduled_step`，只在每次 step 时临时设置 lr，然后恢复登记值。调度只取决于已完成的步数，没有额外状态，续训会自动接上同一条曲线。
  - 日志：每步的 `learning_rate` 写进 updates/*.json。
  - PO-JEPA 和 Transformer 用同一条曲线。
- **状态：** 导师 2026-10-29 确认问题出在损失函数，其余实现没有问题，同意按新度量改。没有做 GPU 训练。

---

# PO-JEPA v4 整改说明

本目录是在 `code-only/po-jepa`（v3，原样保留未动）的副本上做的整改，对应导师 2026-10-27 确认的 P1–P7。
数据、编码器、划分协议与 `configs/train/default.yaml` 完全不变，一轮的冻结特征缓存可以直接复用。所有 v4 设置都在覆盖层 `configs/train/v4.yaml` 里，由 `checks.load_protocol()` 深度合并到 default.yaml 上。

## 1. 改了什么

| 编号 | 内容 | 代码位置 | 配置项（v4.yaml） |
|---|---|---|---|
| P1 | 建图改用 stop-gradient 的编码器输出 Z（1408 维），不再经过可训练的 P_in。Q ∈ R^{16×1408}，固定随机正交，种子 1703 | `spectral.py`（GraphConfig.affinity_source / feature_dim）、`model.py` prepare_query | `graph.affinity_source: encoder`、`feature_dim: 1408`、`sigma_z`【待标定】 |
| P2 | K 固定，各网格（16×16 / 12×12 / 8×8）相同；任何网格 N<K 时直接报错 | `model.py` prepare_query、`checks.py` diagnostic_step | `graph.max_modes: 16`、`fixed_modes: true` |
| P3 | 一致性损失放到 LN 空间（两条路径各做无参 LayerNorm），并按 N·E 取均值 | `losses.py`（_consistency_difference / _consistency_reduce） | `loss.consistency_metric: ln_mean` |
| P6 | 残差读出 Ẑ = Z + P_out(ΨC_τ) − P_out(ΨC_0)，τ=0 时严格等于 Z | `model.py` _predict_raw_with_trace | `projection.readout: residual` |
| P7 | 一致性项权重因子：λ_t、λ_s 在初始化时按梯度比标定后固定（逐 batch 梯度范数取中位数），并加线性 warmup w(s)=min(1, s/200) | `losses.py` weighted_total_loss(consistency_scale)、`training.py` consistency_warmup_scale / consistency_gradient_norms、`workflow.py` calibrate_weights、新命令 `run.py calibrate` | `loss.time_weight`、`space_weight`【待标定】、`consistency_warmup_updates: 200`、`calibration.rho: 0.05`、`batches: 32` |
| 环境 | 版本门槛改为 patho 环境：torch 2.11.0 / CUDA 13.0 / numpy 2.4.3 / PyYAML 6.0.3；设备门槛接受 RTX PRO 6000 Blackwell（96 GB） | `environment.py`、`checks.py`、`pyproject.toml` | — |

原来的行为都还在，只是默认关闭：`affinity_source: projected`、`readout: direct`、`consistency_metric: raw_sum`、`fixed_modes: false` 就是 v3。

清单文件：
- `SOURCE_MAP.json`：8 个改动文件的 candidate_sha256 和 executable_ast_sha256 已刷新（operation 记为 `v4_modified`）。
- `MANIFEST.json`：已重算。
- `VALIDATION_STATUS.json`：已标注 `mathematical_core_changed: true`。

`configs/train/v4.yaml` 需要你填标定值，所以**不进 MANIFEST**。它的 sha256 会写进 preflight 报告的 `v4_overlay_sha256` 字段，用来对账。

## 2. CPU 上做过的验证（未在 GPU 上跑，未做正式训练）

所有结果在 `tools/v4_validation/` 下，脚本可以重跑。

1. **单元测试**：`pytest` 全部通过（Python 3.12 / torch 2.11.0 CPU）。新增的 `tests/test_v4.py` 有 10 项，覆盖以下内容：
   - 覆盖层合并；
   - encoder 建图不受 P_in 影响；
   - projected 模式下 Q 与 v3 逐位相同；
   - 三种网格上 K 都固定为 16，且 N<K 时报错；
   - 残差读出在 τ=0 时严格恒等；
   - LN 度量对输出缩放的不变性，以及显式公式；
   - warmup 和权重；
   - 非法配置被拒绝。
2. **梯度检查**（`gradcheck_v4.py`，float64，缩小维度的代理模型）：解析梯度与有限差分（谱固定）的最大相对误差为 4.6e-8，P_in、P_out、动力学三组参数都通过。谱随参数一起变时，完整损失下会有差异。原因是复合路径的第二段会用中间预测重新建图，而图本身是 stop-gradient 的。这是 detached 设计下的预期行为，不是 bug。
3. **初始化检查**（`init_check_v4.py`，生产配置）：
   - Φ₀ 严格等于输入：是。
   - 三种网格都是 K=16，各有 1 个近零特征值，λ_K 约 0.37–0.43。
4. **损失量级**：同一初始化下，v3 的一致性项是 10⁴ 量级。v4 在 LN 空间下，时间项约 1.6、空间项约 0.37，JEPA 约 0.66，三者处在同一量级。
5. **短训练**（`short_train_v4.py`，替身数据、80 步、batch 4、AdamW lr 1e-3、单种子）。表中是验证集 JEPA，越低越好；不训练的 identity 基线为 0.259。

   | 条件 | 第 0 步 | 第 80 步 | 结果文件 |
   |---|---|---|---|
   | 无一致性（λ=0） | 0.682 | 0.298 | `short_train_v4_result.json` |
   | 完整，ρ=0.2（λ_t=0.022，λ_s=0.92） | 0.682 | 0.687（不降） | `short_train_v4_result.json` |
   | 只开时间项，ρ=0.2 | 0.682 | 0.417 | `…_time_only.json` |
   | 只开空间项，ρ=0.2 | 0.682 | 0.384 | `…_space_only.json` |
   | 完整，ρ=0.05（λ_t=0.005，λ_s=0.23） | 0.682 | 0.296 | `…_rho005.json` |

   结论：
   - 代码通路是通的。无一致性和 ρ=0.05 两个条件下，JEPA 在 80 步内从 0.68 降到 0.30，已经接近 identity 基线。残差读出让 identity 成为模型可达的特例。
   - ρ=0.2 时，两项一致性同时开会明显拖慢 JEPA；只开其中一项也会拖慢，但 JEPA 仍在下降。
   - 因此 `v4.yaml` 的默认值改为 ρ=0.05。论文 P7 中写的 ρ=0.2 也要相应改成 0.05，敏感性实验仍报告 {0.05, 0.2, 0.5}。
   - 局限：这是替身数据、单种子、80 步的冒烟结果，只说明方向，不代表真实特征上的结论。

## 3. 在 GPU 机器上怎么跑（按顺序）

环境用 `conda activate patho`。在 `po-jepa` 根目录下运行，`./features` 换成你的冻结特征目录。给别人代跑时，请直接把 `运行清单_给算力方.md` 发给对方。

**第 1 步：标定 σ_z，选定 K。** CPU 即可，约 1 分钟：

```
python tools/v4_validation/real_feature_probe.py --code . --features-root <features 目录>
```

把最后一行"建议"里的 `sigma_z` 填进 `configs/train/v4.yaml` 的 `graph.sigma_z`，把 `K` 填进 `graph.max_modes`（16 或 32）。

**第 2 步：标定一致性权重。** 这一步只做前向和反向，不更新参数，几分钟即可：

```
python run.py calibrate --features-root "./features" --device cuda:0 --output "./results/calibrate-01"
```

终端会打印 `time_weight` 和 `space_weight`，同样的内容也存在 `calibration.json`。把这两个值填进 `v4.yaml` 的 `loss.time_weight` 和 `loss.space_weight`。
**以后只要改了 σ_z 或 K，就要重新做这一步。**

**第 3 步：检查。**

```
python run.py check --features-root "./features" --device cuda:0 --output "./results/check-v4-01"
```

**第 4 步：正式训练和评估。** 命令和 v3 相同，见 README.md：

```
python run.py run --features-root "./features" --device cuda:0 --output "./results/experiment-v4-01"
```

## 4. 第一次 GPU 运行后的纯代码修复（2026-10-28 第二版）

这些修复不改论文中的方法和公式。编号对应 `v4问题对照表.md`。

| 编号 | 内容 | 代码位置 |
|---|---|---|
| C2 | λ 标定改用逐 batch 梯度范数的**中位数**，batch 数从 8 改为 32；`calibration.json` 同时保存逐 batch 数值、中位数和均值 | `workflow.py` calibrate_weights；`v4.yaml` calibration.batches |
| C3 | 全局梯度范数裁剪，在 optimizer.step 前执行，PO-JEPA 和 Transformer 都适用；每步日志记录裁剪前的范数 `grad_norm_before_clip` | `losses.py` LossConfig.grad_clip_norm；`training.py` clip_gradients / train_step；`comparison.py` condition_step；`v4.yaml` loss.grad_clip_norm: 1.0 |
| C5 | checkpoint 的 `model_config.projection.readout` 记录实际读出方式（之前一律写 direct） | `training.py` _model_config |
| C7 | `run.py` 先把 `--features-root` 解析成绝对路径再交给子进程，路径不存在时直接报错 | `run.py` |
| E02 | 评价时记录当前运行签名，并和训练时的 `runtime.json` 逐项对照；结果写进评价目录的 `runtime.json` 和 report 的 `runtime_matches_training` / `runtime_differences`，只报告不阻断 | `workflow.py` runtime_comparison / evaluate_conditions |
| E03 | check 的 100 次试跑报告新增 `numeric_health`，只报告不阻断：前后 10 次 JEPA 均值、梯度范数中位数和最大值、尖峰次数（超过中位数 100 倍）、最后一个 batch 上的 ‖Ẑ−Z‖/‖Z‖ | `resources.py` numeric_health / readout_ratio |
| E04 | 同一进程内缓存已经完整校验过的轨迹，不再每读一个窗口就把整条轨迹的 SHA 重算一遍；任一文件的 mtime 或大小变化时重新校验 | `cache.py` _read_complete_cache |

`v4.yaml` 中的 `sigma_z` 和 `max_modes` 已经按第一次 GPU 运行的第 1 步结果填好（12.4616、16）。λ 必须用新版重新标定。

`tests/test_v4.py` 新增了 9 项测试，覆盖上表各项。

## 5. 已知的未完成项

- 2026-10-28 已修复：check 和训练循环里的 P_out 调用计数原先按直接读出写死为 5 次，残差读出下实际是 10 次，导致第 3 步 check 失败。现在按 `checks.readouts_per_phi` 计数，并补了 CPU 测试。

- `metrics.py` 里的评价指标 TCD/SCD 目前仍是 raw 范数。论文 P5 要求改用 d_h，这一项计划在拿到第一次 GPU 结果后再改，以免和训练损失的改动混在一起。
- 训练后建议检查时间缺陷与 τσ 的关系是否接近线性，用来确认 (τσ)² 归一化成立。
- ρ 的默认值是 0.05。正式训练建议同时跑 ρ=0（无一致性）作对照；如果 GPU 时间允许，再加跑 ρ=0.2。
