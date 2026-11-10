# v7.1: restarting the paused full run (2026-11-10)

正式运行暂停后，代码加了下面几项，**训练和评测（D 段起）要用新代码从头跑**；编码、标定、检查（A–C2 段）的结果原样复用，不用重跑。

## 改了什么

| 项 | 内容 |
|---|---|
| 早停 | 第 5000 步之后，如果最好的验证点之后已经连续 4 次验证（2000 步）没有刷新最好值，就在这次验证后停。每个条件同一规则；选 checkpoint 的规则不变；学习率日程不变（仍按 10000 步）。 |
| 检查点精简 | 只对 V-JEPA 2-AC 两个条件（每个 3–4 GB）：只留最好验证点前后各 2 个，再加最近一个（续训用）。PO-JEPA、Transformer 每个几 MB，全部保留。 |
| 验证按网格拆开 | 每次验证另外记 12×12、8×8 的直接预测误差，只记录，不参与选点。 |
| 多步滚动评测 | 测试集 16×16 单点另算“每步 1 个周期、走满 T 步”的自回归滚动误差。 |
| 同步数对照（D2 段） | 把 Transformer 和 no-consistency 截到 po_jepa_full 实际训练的步数，再评一次测试集（补充对照，较快）。 |
| 结果位置 | D 段起的结果写进 `../results-v7/full/v7.1/`，不和暂停前跑了一半的任务混在一起。 |

每一步的损失、梯度和优化器都没有改。

## 怎么做（在 `po-jepa` 目录里）

```bash
# 1. 确认没有 run_jobs.py 在跑，代码没被改过（只应看到 jobs-*.txt 这类自己生成的文件）
git status --short
# 2. 拉新代码
git pull
git log -1 --oneline            # 应该看到 v7.1 字样
# 3. 重新生成任务表（文件名不变，A–C2 段的命令和以前逐字相同，会被认成已完成）
python tools/make_v7_jobs.py --stage full > jobs-full.txt
python tools/make_v7_jobs.py --stage variants > jobs-variants.txt
# 4. 先看一眼要跑什么：A–C2 段应显示“已完成”，从 D 段开始待跑
python tools/run_jobs.py jobs-full.txt --gpus 0,1 --dry-run
# 5. 正式跑（和以前一样，放在 tmux/screen 里）
python tools/run_jobs.py jobs-full.txt --gpus 0,1 && python tools/run_jobs.py jobs-variants.txt --gpus 0,1
```

- 第 4 步如果 A–C2 段不是“已完成”（比如暂停时 C2 还没跑），照常跑就行，它们不受这次改动影响。
- **暂停前的半截结果（`../results-v7/full/<场景>/…`）先不要删**，等 Jerry 决定。新代码不会读它们，也不会从它们续训。
- 中断了还是重新运行同一条 `run_jobs.py` 命令。

## 跑完回传

和 `RESULTS_RETURN.md` 第 3 节一样，把打包路径里的 `results-v7/full` 换成 `results-v7/full/v7.1`。
