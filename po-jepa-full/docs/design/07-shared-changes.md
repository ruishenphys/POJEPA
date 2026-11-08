# 07 共享模块改动

从 v7（提交 `911bc09`）复制来的模块，只允许做本文件列出的改动。新功能尽量放进新文件；必须动旧文件时，只加可选分支，v7 已有的 6 个条件（po_jepa_full、po_jepa_no_consistency、matched_transformer、identity、vjepa2_ac、vjepa2_ac_small）走的代码路径一行不变；分发用的判断行可以换成对 v7 条件真值相同、不读新参数的写法（例如 `early_check` 的第一行，§2.4）。

## 1. 总规则

1. **v7 条件逐位不变。** 同样的输入、种子和设备（CPU 上是 1 线程和 §5.2 的固定环境），v7 条件在 po-jepa-full 里算出的参数、损失、检查点里的 `model_state` 和逐窗口评测值都和 v7 树里逐位相同。验证方法见 §5。
2. **配置快照不变。** v7 检查点里存了 `model_config`（`training._model_config`）、`loss_config`、Transformer 规格和 AC 规格，D4 读 v7 检查点时要和新树里现建的模型逐项比较（10 §4）。所以：
   - 不给 `GraphConfig`、`DynamicsConfig`、`IntegratorConfig`、`ProjectionConfig`、`LossConfig` 加字段。
   - 必须给 `TransformerSpec`、`ACSpec` 加字段时（第二期的 7 维动作），新字段取默认值时不进快照，保证 v7 的快照逐字节相同。
   - 只对新条件有意义的配置放进新的配置类（例如 `FourierConfig`），或者放进运行登记，不进 v7 类的快照。
3. **新条件用名字分发。** 新条件名单 `ADDON_CONDITIONS = ("po_jepa_time_only", "po_jepa_space_only", "transformer_consistency", "fixed_fourier", "koopman")` 定义在 `runtime/addon/conditions.py`。共享函数里加一行“是新条件就交给 addon 模块”，放在 v7 原有分支之前；v7 原有的分支和检查不动。
4. **v7 的名单常量不动。** `comparison.CONDITIONS`、`comparison.V7_CONDITIONS`、`comparison.AC_CONDITIONS` 保持原值（`load_comparison_config` 和 `preregister` 依赖它们）。新增 `comparison.ALL_CONDITIONS = (*V7_CONDITIONS, *ADDON_CONDITIONS)`，只用在“这个可训练条件名合不合法”的检查上（S10 定义，S17 的入口检查用；v7 分支里的名单检查保持 `V7_CONDITIONS`）。它和 `V7_CONDITIONS` 一样不含 identity（identity 不训练）；评测侧的名单是 `evaluation.CONDITIONS`，本来就含 identity（§2.5）。

## 2. 第零期要改的地方

S08a 做 §2.1–§2.3、两个新文件 `generic_consistency.py` 和 `ln_metrics.py`，以及 §2.4 表里监控那一行；§2.4 其余各行和 §2.5–§2.7 随新条件一起在 S10 做；§2.8 在 S17 做（`refresh_manifests.py` 的排除项在 S00 做）。

### 2.1 `model.py`：求谱入口委托给谱模块

`POJEPA.prepare_query` 现在在函数里依次调用 `self.spectrum.affinity`、`normalized_laplacian`、`lowest_eigenpairs`。改成一次调用：

```python
if graph.fixed_modes and x.shape[1] < graph.max_modes:                              # 原文不动，从求谱之后挪到求谱之前
    raise ValueError("fixed_modes requires every grid to have at least K tokens")
modes = graph.max_modes if graph.fixed_modes else min(graph.max_modes, x.shape[1])   # 不变
spectrum = self.spectrum.spectrum_for(raw_z if graph.affinity_source == "encoder" else x, grid,
                                      k=modes, model_device=x.device)
```

检查挪到求谱之前只改变配置错误时报错的先后，不影响任何合法输入的结果。`prepare_query` 通过 `self.spectrum.spectrum_for(...)` 调用，`spectrum_for` 通过 `self.affinity` 等实例属性调用（v7 测试在实例上替换 `lowest_eigenpairs`，S10 的 `FourierSpectrum` 重写 `spectrum_for`）。

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

两个新标签不符时各报一句新的 `ValueError`（`condition time_only requires a positive time weight and a zero space weight`、`condition space_only requires a zero time weight and a positive space weight`）；`full`、`no_consistency` 不符和其他未知标签仍报 v7 原来那一句。`-0.0` 算 0。`train_step(..., condition=label)`、`save_training_checkpoint`、`load_training_checkpoint` 都把标签原样传给 `_loss_config`，不用另改。`train_step` 仍然每步算 5 次 Φ 和两个一致性项（权重为 0 的那一项照算、乘 0），这和 v7 的 no_consistency 一样，保证单项消融和 full 的计算图只差权重。

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
| `evaluate_batch` 的类型白名单 | 加 `KoopmanPredictor`（训练中的验证要用；对 v7 模型真值不变） |
| `consistency_share` | POJEPA 类用 `training.consistency_gradient_norms`；transformer_consistency 用 `addon.calibration.generic_gradient_norms`（同一批、和单步相同的 6 次前向，08 §3）；koopman 返回 None |
| `consistency_monitor_summary`、`early_check` | 原来只对 po_jepa_full 生效。新函数 `monitored_consistency_terms(condition, loss_config)` 决定看哪几项：po_jepa_full 固定 `("time", "space")`、不读权重（v7 测试用零权重调它，输出的键、键序和值与 v7 相同，po_jepa_full 仍走 v7 原文）；其余 v7 名字和 identity 返回 None（identity 拿到的是协议的正权重，所以按名字排除）；新条件看权重 > 0 的项（按 time、space 的顺序），一项都没有返回 None。新条件的汇总是 `measurements, rho, threshold, median_share, active, checked_terms`：`median_share` 只含被检查的项，`active` 只看这些项，`checked_terms` 是 list（例如 `["time"]`）。S08a 做到这里，此时对 v7 名字和 identity 以外的任何名字都按权重判断；S10 加了 `ADDON_CONDITIONS` 之后改成只认这 5 个新名字，其他名字报 `ValueError` |

### 2.5 `runtime/evaluation.py`

- `CONDITIONS` 末尾追加 `ADDON_CONDITIONS`（`EvaluationQuery.key` 靠它判断条件名合法）。所有汇总函数调用时都显式传了 `(condition,)`，追加不影响 v7 的输出。
- `run_prepared_evaluation` 的 `expected_type`：transformer_consistency → `TokenTransformer`；time_only、space_only、fixed_fourier → `POJEPA`（fixed_fourier 另要求 `type(model.spectrum) is FourierSpectrum`，其余要求 `type(model.spectrum) is GraphSpectrum`）；koopman → `KoopmanPredictor`。
- `load_selected_checkpoint`：新条件交给 `addon.conditions.load_selected`，检查项和 v7 分支相同（路径不越界、SHA、步数、损失配置）。
- `point_metrics` 不改：`isinstance(model, POJEPA)` 为真的条件（含 fixed_fourier 和两个单项消融）走 v7 的 `action_intervention_margin`，其余走通用的两次前向。

### 2.6 `runtime/v7jobs.py`、`run.py`、`runtime/checks.py`

- `v7jobs.JOB_CONDITIONS` 末尾追加 `ADDON_CONDITIONS`；`run_job`、`transfer_job` 不用另改。
- `run.py` 的 `--condition` 选项加 5 个新条件（S10）；新增子命令 `calibrate-addon`、`check-addon`（S17）、`d4`、`spectral-gap`（S18）（16 §5）。`run`、`resume`、`transfer`、`evaluate` 的行为不变。
- 补跑标定的门槛放在 `workflow.execute` 里、紧接 `require_calibrated_weights` 之后，按 `args.condition` 判断，失败阶段仍记为 `calibration`（`require_calibrated_weights` 本身还被不带条件名的 `write_calibrated_weights` 调用，所以不改它）：新条件里 transformer_consistency 和 fixed_fourier 还要求补跑标定文件存在（16 §3.2），其余新条件只要求导入的 v7 标定存在。`load_protocol()` 不改。

### 2.7 来源登记

`comparison.consumer_sources()` 在 `is_v7()` 分支里加上新文件：`src/po_jepa/fourier.py`、`koopman.py`、`generic_consistency.py`、`ln_metrics.py` 和 `src/po_jepa/runtime/addon/*.py`（用 glob，S17、S18 加的文件自动进来）。新树里 v7 条件的 `consumer_sha256` 因此比 v7 树多几个键；S08a 以后它本来就和 v7 树不同，v7 树的运行不在新树里续训。这些路径满足 `training._provenance` 对键名的要求（以 `src/po_jepa/` 开头、`.py` 结尾）。

### 2.8 `tools/` 的复制件

| 文件 | 改什么 |
|---|---|
| `tools/run_jobs.py` | 默认日志目录改成 `workdir.parent.parent/"logs-addon"/<任务文件名>`（即 工作目录/logs-addon/…）；启动检查的提示改成“请在 full-model/po-jepa-full 目录里运行”；把半截输出改名重跑的子命令名单扩成 `calibrate`、`calibrate-addon`、`check`、`check-addon`、`transfer`、`evaluate`、`d4`、`spectral-gap`；`priority` 里 transformer_consistency、fixed_fourier、两个单项消融排 1，koopman 排 2 |
| `tools/refresh_manifests.py` | `EXCLUDE_DIRS` 加 `docs` 和 `.vscode`，`MUTABLE` 加 `README.md`（连同它上面的注释共三行，原文见 S00 段说明 §3.2）。改设计文档、编辑器设置或项目状态页都不改变 MANIFEST，`verify_bundle` 不会因此失败，续训和只评测的登记也不会对不上。注意 `EXCLUDE_DIRS` 按路径里任何一级目录名匹配，所以以后任何一层叫 `docs` 或 `.vscode` 的目录（如 `src/po_jepa/docs/`）都不进 MANIFEST；新代码不要放进这样命名的目录 |
| `tools/plan_v7.py` | 接受新条件（读检查点走 `evaluation.load_selected_checkpoint`）；其余不变 |
| `tools/make_v7_jobs.py`、`tools/summarize_v7.py` | 原样保留，不用于补跑；补跑用新的 `make_addon_jobs.py`、`summarize_addon.py` |

## 3. 第二期（S08b，DROID）要改的地方

先列出来，细节在 09 里写：

- `baselines.py`：`TransformerSpec` 和输入检查接受 m 维动作（DROID m = 7）。新字段取默认值 2 时不进快照。
- `ac_baseline.py`：新增“官方输入”模式（保留官方的 7 维动作编码器和状态编码器，用真实本体状态）。v7 的 2 维替换模式不变，`ACSpec` 新字段取默认值时不进快照。
- `addon.conditions.koopman_step` 用 `comparison._tf_batch` 核批，它经 `baselines.validate_inputs` 写死 2 维动作；随 Transformer 的 m 维动作一起放开。
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

1. **v7 原有测试**：`tests/` 下 v7 的测试在 po-jepa-full 里通过，不改测试本身。没有 CUDA 13.0 版 torch 的机器上，`tests/test_delivery.py` 有 4 个测试因 `environment.verify_versions` 报 CUDA 版本不同而失败，未改动的 v7 树同样如此；只允许这 4 个失败、报错相同（GPU 机器上应全部通过）。
   测试里的 `verify_bundle` 全都打了补丁，没有测试读 MANIFEST，清单错了测试照样全绿。所以清单另外显式核对：改了进清单的文件，就在 po-jepa-full 下运行 `tools/refresh_manifests.py` 重算并提交；然后在新目录里真的调用一次 `workflow.verify_bundle()` 和 `cache.verify_fixed_protocol()`，必须通过（14 §2）。
2. **双树对照冒烟** `tests/addon_dual_tree.py`（S00 写，S08a、S10 原样复用；只在同时有 v7 树的开发机上跑，pytest 不收集它）：
   - 输入：v7 的 `encode_v7.py --fake-encoder` 生成的假特征（van_der_pol，40 条轨迹，两棵树读同一份）。每棵树各起一个子进程，运行本树的 `tools/smoke_v7_cpu.py --updates 20 --seed 1703 --jobs po_jepa_full,po_jepa_no_consistency,matched_transformer,identity,vjepa2_ac_small,vjepa2_ac`：每个条件训练 20 步（第 0、10、20 步验证）加评测，identity 以外的 5 个条件另做一次只评测重跑。
   - vjepa2_ac：CPU 上做不了官方规模（约 3 亿参数，要官方权重），两棵树都换成测试专用的缩小规格 `ACSpec(width=64, depth=2, heads=4, mlp_ratio=4.0)` 和同一份假权重，走的代码路径（载入官方权重、微调、检查点、评测）和正式运行相同；官方规模的数值等价由第 3 项核。
   - 只在 1 线程、固定环境下跑：`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OMP_DYNAMIC=FALSE`、`MKL_DYNAMIC=FALSE`、`MKL_CBWR=COMPATIBLE,STRICT`、`PYTHONHASHSEED=0`。实测 v7 树自己在 2 线程、4 线程下连跑两次就不能逐位复现（标定的 λ 就不同），多线程下两棵树的不同分不清是代码还是线程调度造成的。
   - 逐位相同的项：标定结果、初始权重哈希、每一步的训练记录（损失、梯度量、学习率、调用次数）、每个验证文件的逐窗口值和选点、第 20 步的 `model_state`、每一条测试评测行（含只评测重跑）和评测汇总。不比耗时、绝对路径和来源字段（源码 SHA 在 S08a 以后会合法地变化）。除了按项点名比较，还对 `smoke/` 下的全部文件做整份比较：文件集合相同，全部 JSON 和全部检查点在删掉一组固定的键（耗时、含耗时的文件 SHA、绝对路径、源码 SHA、从不播种的全局随机状态）以后相同，所以表里没点名的字段也不会漏掉。完整的比较表、排除的键和每项的条数在 `docs/segments/S00.md` §3.4.5。
   - 一次约 1 小时 50 分（两棵树并行，各占 1 个核；S00 实测 1:47:37，机器有别的负载时到 2:18:26），时间几乎都花在评测上，要放到后台跑，并在独立会话里起（`setsid`），免得被后台命令的 2 小时时限打断。
3. **真实检查点等价核对**：D4 在每个 v7 检查点上重算 v7 的标准指标，和 v7 自己的评测记录比较（10 §4）。这一步在机哥的机器上随 D4 一起跑。
4. **快照核对**：对 6 个 v7 条件，新树里现建模型的 `_model_config`、`_loss_config`、Transformer 规格、AC 规格序列化后和 v7 检查点里存的逐字节相同。
