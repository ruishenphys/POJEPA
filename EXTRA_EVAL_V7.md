# 补充评测：用已训练好的模型补算几项指标（不训练）

`tools/extra_eval_v7.py` 读取每个已完成任务正式评测所用的检查点，在 16×16 测试窗口上补算：

| 指标 | 含义 |
|---|---|
| `direct` | 直接预测 Φ_T(Z_t) 到真实第 t+T 帧的 JEPA 距离；动作和正式评测一样先在 T/2 处切开再拼接，数值等于正式的 `future_jepa_distance` |
| `half` | 先 T/2 再 T/2 的组合预测到真值的距离（时间一致性损失里那条组合路径） |
| `unit` | 每步 1 个周期、走满 T 步的 rollout 到真值的距离（直接调用正式评测的 `unit_rollout`） |
| `p2_4`、`p4_2` | 6 = 2+4、6 = 4+2（只对 T = 6） |
| `dc_half` | 直接预测与组合预测之间的 JEPA 距离 |
| `id` | 当前帧到真值的距离（identity 参照） |

时长 1、2、3、4、6、8、12、16 个周期；6 和 16 不在正式评测里，单独标注。只读，不改训练、选点或正式结果表。条件：po_jepa_full、po_jepa_no_consistency、matched_transformer、identity（不含两个 V-JEPA 2-AC）。

## 身份核对（2026-11-11 第三轮审核后加）

- 检查点：任务的 `COMPLETE.json` 必须绑定顶层 `report.json`，`report.json` 的 `best` 必须等于 `evaluation/<条件>/registration.json` 登记的检查点，训练目录（本地 `training/<条件>`，没有时按 `job.json` 的 `training_run` / `resume_from` 找，与 `run_job` 相同）的 `COMPLETE.json` 绑定的训练报告也必须选同一个点；再核文件 SHA、步数、训练来源记录、条件和架构。
- 特征：所用缓存的身份必须等于任务 `report.json` 里 preflight 记录的缓存身份（即正式评测读的那份）；同标签的另一份缓存会被拒绝。
- 输出：每个结果文件带 `identity`（度量版本、任务报告 SHA、正式评测登记 SHA、检查点、特征摘要、训练步数、测试条数、时长）。同名文件只在 `identity` 完全一致时跳过；不一致直接报错，不覆盖也不复用；写入是先写临时文件再改名。同一结果目录里有两个相同数据集/种子/条件的已完成任务时 `run` 直接停下。
- 汇总：拒收没有 `identity` 的旧文件（a305bfe 写的）和放错位置的文件；同一数据集/种子下各条件的缓存、步数、测试条数必须一致；full 对 no-consistency 只在共同窗口上配对，并写出配对数和各自未配对数；某条件在某时长没有窗口时均值记为空。

**a305bfe 写的旧输出不能用**（Transformer 的 `direct`、`dc_half` 输入与正式评测不同，且没有身份记录），请换一个新的 `--output` 目录重跑。

## 怎么跑（在 `po-jepa` 目录里）

```bash
git pull            # 只改 tools/extra_eval_v7.py、tests/test_extra_eval_v7.py 和本文件（都不在 MANIFEST 里）；正在跑的任务不受影响
python -m pytest -q tests/test_extra_eval_v7.py      # 约 10 秒，CPU，不需要本地标定文件

# 试跑结果（GPU 被正式运行占着就用 CPU：约 2 小时；空闲 GPU 上约 10–20 分钟）
python tools/extra_eval_v7.py run --results-root ../results-v7/pilot --features-base ../features-v7 \
    --output ../extra-eval-v7-r2/pilot --device cpu
python tools/extra_eval_v7.py summarize --output ../extra-eval-v7-r2/pilot
```

- 每个任务训练时用的特征目录从 `<任务>/console/command.json` 里读，`--features-base` 只是找不到记录时的备用。
- 中断后重跑同一条命令，身份一致的已完成任务会跳过，写了一半的文件会重算。
- 正式运行（2000 条）出结果后同样适用：`--results-root ../results-v7/full/v7.1 --output ../extra-eval-v7-r2/full`，建议加 `--max-test-trajectories 50` 控制时间。

## 回传

把 `../extra-eval-v7-r2/` 整个文件夹打包发回即可（只有 JSON，几 MB）：

```bash
zip -r extra-eval-v7-r2.zip ../extra-eval-v7-r2
```
