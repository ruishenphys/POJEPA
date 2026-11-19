# po-jepa-full

完整模型的主干：仿真补跑和 DROID 实验共用这一份代码。放在仓库的 `full-model` 分支上；v7 的代码在 `v7` 分支（2026-11-10 起本分支根目录不再放 v7 副本）。

## 现在的状态

**设计规格已全部写完，代码按 [15-workflow.md](docs/design/15-workflow.md) 分段写，进度和缺口见仓库根目录的 [FULL_MODEL_STATUS.md](../FULL_MODEL_STATUS.md)。** 现在能跑正式实验的是 `v7` 分支（v7.1）；给机哥的补跑说明是 [ADDON_RUN_GUIDE.md](ADDON_RUN_GUIDE.md)。

## 先读哪些

| 文件 | 内容 | 状态 |
|---|---|---|
| [docs/design/00-overview.md](docs/design/00-overview.md) | 目标、已定决策、分期、原则、文档索引 | 已写 |
| [docs/design/16-sim-addon.md](docs/design/16-sim-addon.md) | 仿真补跑的完整方案：条件、标定、v7 检查点复用、新指标、任务文件、给机哥的说明 | 已写 |
| [docs/design/07-shared-changes.md](docs/design/07-shared-changes.md) | 从 v7 复制来的模块要改的每一处，以及“默认行为逐位不变”怎么验 | 已写 |
| [docs/design/08-conditions.md](docs/design/08-conditions.md) | 全部条件的定义，固定 Fourier 和 Koopman 的公式 | 已写 |
| [docs/design/10-evaluation.md](docs/design/10-evaluation.md) | 评测：v7 原有指标、新增的 LN 空间指标、分割差异、长时程、截断间隙 | 已写 |
| [docs/design/13-summary.md](docs/design/13-summary.md) | 汇总：精度匹配规则、数据一致性核对、统计方法、表和图 | 已写 |
| [docs/design/14-testing-and-acceptance.md](docs/design/14-testing-and-acceptance.md) | 测试分层和每段的验收标准 | 已写（含 DROID 各段） |
| [docs/design/15-workflow.md](docs/design/15-workflow.md) | 开发流程：分段、三个 agent 加 Codex、合并条件、时间表 | 已写 |
| [01](docs/design/01-droid-config.md)–[06](docs/design/06-droid-windows-and-batches.md)、[09](docs/design/09-droid-training.md)、[11](docs/design/11-droid-planning.md)、[12](docs/design/12-droid-jobs.md) | DROID 的配置、索引划分、下载、准备、编码、窗口、训练循环、规划、任务图；DROID 的评测、汇总、验收在 10 §8、13 §8、14 §3 | 已写 |
| [docs/design/droid-golden/](docs/design/droid-golden/README.md) | 5 条真实 DROID 回合上官方代码算出的参照值和生成脚本 | 已写 |

## 目录（代码阶段）

```
po-jepa-full/
  run.py                    # 从 v7 复制，加补跑的子命令和条件
  src/po_jepa/              # 从 v7 复制（S00 时是 911bc09，S00t 跟进到 0ac45fe）；新文件 fourier.py、koopman.py、generic_consistency.py、ln_metrics.py
  src/po_jepa/runtime/addon/  # 仿真补跑：条件分发、标定、v7 检查点读取、D4 评测、截断间隙、编码器核对
  src/po_jepa/droid/        # DROID 数据入口（第二期以后）
  tools/                    # make_addon_jobs.py、import_v7_calibration.py、summarize_addon.py、run_jobs.py（复制后改默认日志目录）
  configs/                  # 从 v7 复制；configs/calibration/ 放导入的 v7 标定和补跑标定
  reference/                # 从 v7 原样复制（协议文件逐字节相同）
  tests/                    # v7 原有测试，加 test_addon_*.py、test_droid_*.py
  docs/design/              # 本目录的设计规格
  docs/segments/            # 每段的提示词、验收单和审核记录
```

## 在别的集群上跑（可移植模式，C8）

默认的三道门槛是按机哥那台机器定的：显卡白名单（RTX 4090、RTX PRO 6000、A100，v7 另加 H100/H200/H800/A800/L40S）、环境版本写死（Python 3.12、torch 2.11.0、CUDA 13.0、numpy 2.4.3、PyYAML 6.0.3）、仿真续训必须落在同一张物理卡上（新旧两份 `runtime.json` 逐项相同）。在别的集群上跑时，任务脚本里设

```bash
export PO_JEPA_PORTABLE=1
```

就放宽这三处（不设或设成 `0` 时和以前完全相同；设成别的值直接报错）：

| 门槛 | 放宽后 | 记在哪里 |
|---|---|---|
| 显卡白名单（`runtime/checks.py`） | 名单外的卡也收，但仍要整张物理卡（UUID 以 `GPU-` 开头、MIG 未开）、显存 ≥ 24 GB（这是身份下限，不保证装得下；正式跑 `vjepa2_ac` 前先按实际批量和序列长度测一次峰值显存） | 硬件记录里 `portable_unlisted_gpu: true`，`nominal_capacity_gb` 记 24 |
| 环境版本（`runtime/environment.py`） | 版本不同也能跑 | `environment.json` 的 `portable_version_mismatches` 列出每一项的期望值和实际值 |
| 续训同卡（`runtime/resume.py`） | 续训可以换卡、换节点：只放过 `platform`、`gpu_uuid`、`cpu_threads`、`interop_threads`、`CUDA_VISIBLE_DEVICES`；逻辑设备号 `device` 仍须相同（检查点和 CUDA 随机流都记了它），换卡时用 `CUDA_VISIBLE_DEVICES` 把新卡映射进来、进程里始终用默认的 `cuda:0`（`tools/run_jobs.py` 就是这样起任务的，别另传 `--device cuda:1`）；库版本、精度和 TF32、确定性开关、其余线程环境变量仍须相同；预登记仍须逐项相同 | 标准错误输出打一行不同的字段；新目录和旧目录的 `runtime.json` 都在，可以逐项对 |

放宽的只是“在哪台机器、哪个环境上跑”的准入，不改模型、损失、随机流、选点和评测。换了版本或换了卡，结果不保证和原环境逐位相同；同一组对照（同一场景、同一种子的各条件）应在同一种卡、同一套环境上跑完，CPU 线程数（`OMP_NUM_THREADS`、`torch.get_num_threads()`）和谱分解后端也要相同：GPU 训练里图亲和度、谱分解仍在 CPU 上算（`spectral.py`），线程数不同浮点结果可能不同。开关设成 `0`、`1` 以外的值时，无论这三道门槛本来是否通过都直接报错。`run.py` 用 `PYTHONPATH=src` 读本目录的源码，不需要 `pip install`，所以 `pyproject.toml` 里写死的版本不拦运行。
