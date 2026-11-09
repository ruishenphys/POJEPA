# po-jepa-full

完整模型的主干：仿真补跑和 DROID 实验共用这一份代码。放在仓库的 `full-model` 分支上，和根目录的 v7 代码互不影响。

## 现在的状态

**设计规格已全部写完，代码按 [15-workflow.md](docs/design/15-workflow.md) 分段开始写**（Jerry 2026-11-08 授权规格写完即开工）。代码写好以前，仓库根目录的 v7 代码是唯一能跑的版本，机哥从 2026-11-09 起用它做正式运行，期间 v7 不做任何改动。

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
