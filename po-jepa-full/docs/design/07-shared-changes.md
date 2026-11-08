# 07 共享模块改动

从 v7（提交 `911bc09`）复制来的模块，只允许做本文件列出的改动。新功能尽量放进新文件；必须动旧文件时，只加可选分支，v7 已有的 6 个条件（po_jepa_full、po_jepa_no_consistency、matched_transformer、identity、vjepa2_ac、vjepa2_ac_small）走的代码路径一行不变。

## 1. 总规则

1. **v7 条件逐位不变。** 同样的输入、种子和设备，v7 条件在 po-jepa-full 里算出的参数、损失、检查点里的 `model_state` 和逐窗口评测值都和 v7 树里逐位相同。验证方法见 §5。
2. **配置快照不变。** v7 检查点里存了 `model_config`（`training._model_config`）、`loss_config`、Transformer 规格和 AC 规格，D4 读 v7 检查点时要和新树里现建的模型逐项比较（10 §4）。所以：
   - 不给 `GraphConfig`、`DynamicsConfig`、`IntegratorConfig`、`ProjectionConfig`、`LossConfig` 加字段。
   - 必须给 `TransformerSpec`、`ACSpec` 加字段时（第二期的 7 维动作），新字段取默认值时不进快照，保证 v7 的快照逐字节相同。
   - 只对新条件有意义的配置放进新的配置类（例如 `FourierConfig`），或者放进运行登记，不进 v7 类的快照。
3. **新条件用名字分发。** 新条件名单 `ADDON_CONDITIONS = ("po_jepa_time_only", "po_jepa_space_only", "transformer_consistency", "fixed_fourier", "koopman")` 定义在 `runtime/addon/conditions.py`。共享函数里加一行“是新条件就交给 addon 模块”，放在 v7 原有分支之前；v7 原有的分支和检查不动。
4. **v7 的名单常量不动。** `comparison.CONDITIONS`、`comparison.V7_CONDITIONS`、`comparison.AC_CONDITIONS` 保持原值（`load_comparison_config` 和 `preregister` 依赖它们）。新增 `comparison.ALL_CONDITIONS = (*V7_CONDITIONS, *ADDON_CONDITIONS)`，只用在“这个条件名合不合法”的检查上。

## 2. 第零期要改的地方

S08a 做 §2.1–§2.3、两个新文件 `generic_consistency.py` 和 `ln_metrics.py`，以及 §2.4 表里监控那一行；§2.4 其余各行和 §2.5–§2.7 随新条件一起在 S10 做；§2.8 在 S17 做（`refresh_manifests.py` 的排除项在 S00 做）。

### 2.1 `model.py`：求谱入口委托给谱模块

`POJEPA.prepare_query` 现在在函数里依次调用 `self.spectrum.affinity`、`normalized_laplacian`、`lowest_eigenpairs`。改成一次调用：

```python
modes = graph.max_modes if graph.fixed_modes else min(graph.max_modes, x.shape[1])   # 不变
# fixed_modes 的检查不变，挪到求谱之前（只改变配置错误时报错的先后，不影响任何合法输入的结果）
spectrum = self.spectrum.spectrum_for(raw_z if graph.affinity_source == "encoder" else x, grid,
                                      k=modes, model_device=x.device)
```

`GraphSpectrum.spectrum_for(source, grid, *, k, model_device)`（新方法，见 2.2）内部按原来的顺序调用那三个函数，所以 v7 条件的结果逐位相同。固定 Fourier 的谱模块重写这个方法（08 §4）。

### 2.2 `spectral.py`：新增 `spectrum_for`

```python
def spectrum_for(self, source: Tensor, grid: GridSpec, *, k: int, model_device: torch.device) -> SpectralResult:
    """建图并求最低 k 个特征对；与 v7 prepare_query 里原有的三步完全相同。"""
    affinity = self.affinity(source, grid)
    laplacian = self.normalized_laplacian(affinity).laplacian
    return self.lowest_eigenpairs(laplacian, k=k, model_device=model_device)
```

只加这个方法，`GraphSpectrum` 的其他部分和 `GraphConfig` 不动。

### 2.3 `runtime/training.py`：单项消融的损失标签

`_loss_config(config, condition)` 现在只认 `"full"`（两个权重都 > 0）和 `"no_consistency"`（两个都 = 0）。加两个标签：

| 标签 | 要求 | 用于 |
|---|---|---|
| `"time_only"` | time_weight > 0，space_weight = 0 | po_jepa_time_only |
| `"space_only"` | time_weight = 0，space_weight > 0 | po_jepa_space_only |

其余标签和报错不变。`train_step(..., condition=label)`、`save_training_checkpoint`、`load_training_checkpoint` 都把标签原样传给 `_loss_config`，不用另改。`train_step` 仍然每步算 5 次 Φ 和两个一致性项（权重为 0 的那一项照算、乘 0），这和 v7 的 no_consistency 一样，保证单项消融和 full 的计算图只差权重。

### 2.4 `runtime/comparison.py`：条件分发

下面每个函数开头加一行分发，v7 分支不动：

| 函数 | 新条件怎么处理 |
|---|---|
| `build_condition` | `addon.conditions.build(condition, device)`，返回 `(model, loss)`；loss 的两个权重按 08 §1 的表设置 |
| `make_optimizer` | POJEPA 类（含 fixed_fourier）仍走 `training.make_optimizer`；Koopman 走 addon 里和 Transformer 相同设置的 AdamW；transformer_consistency 走原来的 Transformer 分支 |
| `condition_step` | time_only/space_only/fixed_fourier 调 `training.train_step`，标签见 08 §1；transformer_consistency 和 koopman 调 addon 里的单步（08 §3、§5） |
| `save_checkpoint` / 读检查点 | POJEPA 类走 `training.save_training_checkpoint`；transformer_consistency 走 Transformer 的保存函数，但 `loss_config` 用 `"full"` 标签；koopman 走 addon 的保存函数（格式见 08 §5.7） |
| `_validate_registration` | 接受 `ALL_CONDITIONS` |
| `_observed_update` 的读出次数检查 | 每次更新 `p_out` 被调用的次数：po_jepa 类和 fixed_fourier 为 5 × `readouts_per_phi`（残差读出时 = 10）；transformer_consistency 为 6；koopman 为 2。记录里的 `phi_calls` 相应为 5、6、1 |
| `consistency_share` | POJEPA 类用 `training.consistency_gradient_norms`；transformer_consistency 用 `addon.calibration.generic_gradient_norms`（同一批、和单步相同的 6 次前向，08 §3）；koopman 返回 None |
| `consistency_monitor_summary`、`early_check` | 原来只对 po_jepa_full 生效。改成对“至少一个一致性权重 > 0”的条件生效，只检查权重 > 0 的那几项；po_jepa_full 两项都检查，输出的键和值与 v7 相同。新条件的输出另加一个键 `checked_terms`（例如 `["time"]`） |

### 2.5 `runtime/evaluation.py`

- `CONDITIONS` 末尾追加 `ADDON_CONDITIONS`（`EvaluationQuery.key` 靠它判断条件名合法）。所有汇总函数调用时都显式传了 `(condition,)`，追加不影响 v7 的输出。
- `run_prepared_evaluation` 的 `expected_type`：transformer_consistency → `TokenTransformer`；time_only、space_only、fixed_fourier → `POJEPA`（fixed_fourier 另要求 `type(model.spectrum) is FourierSpectrum`，其余要求 `type(model.spectrum) is GraphSpectrum`）；koopman → `KoopmanPredictor`。
- `load_selected_checkpoint`：新条件交给 `addon.conditions.load_selected`，检查项和 v7 分支相同（路径不越界、SHA、步数、损失配置）。
- `point_metrics` 不改：`isinstance(model, POJEPA)` 为真的条件（含 fixed_fourier 和两个单项消融）走 v7 的 `action_intervention_margin`，其余走通用的两次前向。

### 2.6 `runtime/v7jobs.py`、`run.py`、`runtime/checks.py`

- `v7jobs.JOB_CONDITIONS` 末尾追加 `ADDON_CONDITIONS`；`run_job`、`transfer_job` 不用另改。
- `run.py` 的 `--condition` 选项加 5 个新条件；新增子命令 `calibrate-addon`、`check-addon`、`d4`、`spectral-gap`（16 §5）。`run`、`resume`、`transfer`、`evaluate` 的行为不变。
- `checks.require`/`workflow.require_calibrated_weights`：新条件里 transformer_consistency 和 fixed_fourier 还要求补跑标定文件存在（16 §3.2），其余新条件只要求导入的 v7 标定存在。`load_protocol()` 不改。

### 2.7 来源登记

`comparison.consumer_sources()` 加上新文件：`src/po_jepa/fourier.py`、`koopman.py`、`generic_consistency.py`、`ln_metrics.py` 和 `src/po_jepa/runtime/addon/*.py`。这些路径满足 `training._provenance` 对键名的要求（以 `src/po_jepa/` 开头、`.py` 结尾）。

### 2.8 `tools/` 的复制件

| 文件 | 改什么 |
|---|---|
| `tools/run_jobs.py` | 默认日志目录改成 `workdir.parent.parent/"logs-addon"/<任务文件名>`（即 工作目录/logs-addon/…）；启动检查的提示改成“请在 full-model/po-jepa-full 目录里运行”；把半截输出改名重跑的子命令名单扩成 `calibrate`、`calibrate-addon`、`check`、`check-addon`、`transfer`、`evaluate`、`d4`、`spectral-gap`；`priority` 里 transformer_consistency、fixed_fourier、两个单项消融排 1，koopman 排 2 |
| `tools/refresh_manifests.py` | `EXCLUDE_DIRS` 加 `docs`，改设计文档不会改变 MANIFEST，也就不会让续训的登记对不上 |
| `tools/plan_v7.py` | 接受新条件（读检查点走 `evaluation.load_selected_checkpoint`）；其余不变 |
| `tools/make_v7_jobs.py`、`tools/summarize_v7.py` | 原样保留，不用于补跑；补跑用新的 `make_addon_jobs.py`、`summarize_addon.py` |

## 3. 第二期（S08b，DROID）要改的地方

先列出来，细节在 09 里写：

- `baselines.py`：`TransformerSpec` 和输入检查接受 m 维动作（DROID m = 7）。新字段取默认值 2 时不进快照。
- `ac_baseline.py`：新增“官方输入”模式（保留官方的 7 维动作编码器和状态编码器，用真实本体状态）。v7 的 2 维替换模式不变，`ACSpec` 新字段取默认值时不进快照。
- 求谱后端：GPU float64 或 CPU 多线程逐样本。后端选择**不放进 `GraphConfig`**（否则 v7 的 `model_config` 快照会变），由运行登记里单独的 `spectral_backend` 字段给出，所有 PO-JEPA 类条件用同一个；默认仍是 v7 的 CPU 批量求解。

## 4. 新文件

| 文件 | 内容 | 规格 |
|---|---|---|
| `src/po_jepa/fourier.py` | `FourierConfig`、`FourierSpectrum`、DCT-II 基底和解析特征值 | 08 §4 |
| `src/po_jepa/koopman.py` | `KoopmanConfig`、`KoopmanPredictor` | 08 §5 |
| `src/po_jepa/generic_consistency.py` | 不限模型类型的 `temporal_predictions`、`spatial_predictions` | 08 §3 |
| `src/po_jepa/ln_metrics.py` | LN 空间的差异度量 | 10 §3.1 |
| `src/po_jepa/runtime/addon/conditions.py` | 新条件的构造、优化器、单步、检查点 | 08 |
| `src/po_jepa/runtime/addon/calibration.py` | 导入 v7 标定、补跑标定、通用梯度范数 | 16 §3 |
| `src/po_jepa/runtime/addon/v7load.py` | 读 v7 检查点（放宽来源核对） | 10 §4 |
| `src/po_jepa/runtime/addon/d4.py`、`long_window.py`、`spectral_gap.py`、`encoder_identity.py` | D4 评测和几项核对 | 10 §3–§6 |

## 5. 怎么验证“v7 条件逐位不变”

1. **v7 原有测试**：`tests/` 下 v7 的测试在 po-jepa-full 里全部通过，不改测试本身（MANIFEST 用 refresh_manifests 重算后，交付测试按新清单核对）。
2. **双树对照冒烟**：在 CPU 上用假特征，分别在 v7 树和 po-jepa-full 树里对 6 个 v7 条件各跑 20 步训练加评测（`tools/smoke_v7_cpu.py` 的同一组输入）。要求：初始权重哈希、第 20 步的 `model_state`、每一步的损失值、全部逐窗口评测值逐位相同。
3. **真实检查点等价核对**：D4 在每个 v7 检查点上重算 v7 的标准指标，和 v7 自己的评测记录比较（10 §4）。这一步在机哥的机器上随 D4 一起跑。
4. **快照核对**：对 6 个 v7 条件，新树里现建模型的 `_model_config`、`_loss_config`、Transformer 规格、AC 规格序列化后和 v7 检查点里存的逐字节相同。
