# 小规模验证运行说明（S28c，2026-11-17）

给跑卡的人看。目的：用小算力确认完整模型符合论文理论、学到物理，通过后再把同一份代码交给导师集群跑正式规模（方案 `full-model-next-trial` §5）。
只评测和小规模训练，不是正式训练。

## 0. 准备

- 用一份**新 clone** 的 full-model 分支（`工作目录/full-model/po-jepa-full`），不要和 v7 场景补跑（make_addon_jobs.py、导入 v7 标定）共用工作树。
- 目录约定和 v7、补跑相同（相对 `po-jepa-full`）：`../../features-v7/`（v7 编好的特征，含 `<场景>-200`、`<场景>-2000`）、
  `../../results-v7/full/v7.1/`（v7.1 正式运行的任务目录，**检查点要在**）、`../../checkpoints/`、`../../vjepa2/`。
- 新写的目录：`../../features-small-validation/`（长时程测试集）、`../../results-small-validation/`（全部结果）。
- 场景 advection_diffusion、double_well、van_der_pol，种子 1703。

## 1. 生成任务文件

```
python tools/make_small_validation_jobs.py --stage b-v7.1 > jobs-sv-b-v7.1.txt
python tools/make_small_validation_jobs.py --stage c1 --v6-features "$V6" > jobs-sv-c1.txt
```

`$V6` 是 v7 试跑用的 v6 double_well-200 特征目录的绝对路径（v7 交接说明里 `V6=` 那一行，和 v7 试跑、补跑试跑同一个）。

## 2. 跑（两份可以同时排，互不依赖）

```
python tools/run_jobs.py jobs-sv-b-v7.1.txt --gpus 0,1
python tools/run_jobs.py jobs-sv-c1.txt --gpus 0,1
```

- **b-v7.1**（B 层诊断，只评测）：编码 3 个场景的长时程测试集（horizon=128，只编测试段）→ 动作探针 → 拟合线性探针（验证段，full 读出约 10 GB 内存）
  → 真值（CPU）→ 在 v7.1 的 6 个条件 × 3 个种子上读长时程物理量（VPT）和反事实 → advection_diffusion 的模态频率对照 → 汇总。
  v7.1 检查点是 v7 分支代码训练的，读出工具带 `--cross-version`（核数据来源和配置快照，不核源码 SHA）；核不过会直接报错，不会读错模型。
- **c1**（C-1 筛选）：PO-JEPA 的 2×2（二等分/多段切分 × Xavier/零初始化），每格 3 个场景 × 200 条 × 2000 步。
  三个非默认变体先各自 `run.py calibrate`（9 条），默认变体用 git 里 v7 试跑的标定。
- 断了直接重跑同一条命令（run_jobs 会跳过完成的、续训没完成的）。

## 3. C-1 之后

`../../results-small-validation/c1/summary/SUMMARY.md` 里有 2×2 对照和**推荐的 C-2 变体**（规则写在 `tools/summarize_small_validation.py` 文件头），
末尾给出生成 C-2 任务文件的命令，形如：

```
python tools/make_small_validation_jobs.py --stage c2 --temporal-partition dyadic --readout-init zero > jobs-sv-c2.txt
python tools/run_jobs.py jobs-sv-c2.txt --gpus 0,1
```

先把 C-1 的汇总发回来，由 Jerry 这边确认变体再跑 C-2。C-2 要等 b-v7.1 的 R1、R2（探针、真值）和 L1、L2（长时程测试集）完成。
C-2 训练 po_jepa_full、po_jepa_no_consistency、transformer_consistency、fixed_fourier、po_jepa_rollout_sup（每个场景 2000 条 × 10000 步），
另跑 3 条 identity（不训练）：它和 v7.1 的 identity 逐位相同，才复用 v7.1 的 matched_transformer、AC-small、AC，不重训。
如果 C-2 变体是多段切分，po_jepa_rollout_sup 也是 2^k 段有真值滚动（和 po_jepa_full 同一串 k，S28d），任务文件开头有一行说明；用的代码必须含 S28d，S28d 以前在多段切分下标的补跑标定文件会被拒绝。

## 4. 回传

用 `tools/pack_results.py`（说明见 `RESULTS_RETURN_PACK.md`；模型权重 `.pt` 默认不打包），不要用 zip。在**工作目录**里（`full-model/` 所在的目录）：

```bash
nvidia-smi > nvidia-smi.txt
python full-model/po-jepa-full/tools/pack_results.py pack results-small-validation logs-addon/jobs-sv-* \
    full-model/po-jepa-full/configs/calibration full-model/po-jepa-full/jobs-sv-*.txt nvidia-smi.txt \
    --output results-small-validation.pack.sqlite
```

`configs/calibration` 要带上：非默认变体的标定（`*~dyadic*.yaml`、`*~zeroinit.yaml`）和 `addon/` 下的补跑标定是跑的时候写进这个目录的。
`logs-addon/jobs-sv-*` 是 run_jobs 给这几份任务文件的默认日志目录。
