# 补充评测：用已训练好的模型补算几项指标（不训练）

`tools/extra_eval_v7.py` 读取每个已完成任务里按验证集选中的检查点（与正式评测同一个文件、同一个 SHA），在 16×16 测试窗口上补算：

| 指标 | 含义 |
|---|---|
| `direct` | 直接预测 Φ_T(Z_t) 到真实第 t+T 帧的 JEPA 距离（与正式评测相同） |
| `half` | 先 T/2 再 T/2 的组合预测到真值的距离（时间一致性损失里那条组合路径） |
| `unit` | 每步 1 个周期、走满 T 步的 rollout 到真值的距离 |
| `p2_4`、`p4_2` | 6 = 2+4、6 = 4+2（只对 T = 6） |
| `dc_half` | 直接预测与组合预测之间的 JEPA 距离 |
| `id` | 当前帧到真值的距离（identity 参照） |

时长 1、2、3、4、6、8、12、16 个周期；6 和 16 不在正式评测里，单独标注。只读，不改训练、选点或正式结果表。条件：po_jepa_full、po_jepa_no_consistency、matched_transformer、identity（不含两个 V-JEPA 2-AC）。

## 怎么跑（在 `po-jepa` 目录里）

```bash
git pull            # 只新增 tools/extra_eval_v7.py、tests/test_extra_eval_v7.py 和本文件；正在跑的任务不受影响
python -m pytest -q tests/test_extra_eval_v7.py      # 约 10 秒，CPU

# 试跑结果（GPU 被正式运行占着就用 CPU：约 2 小时；空闲 GPU 上约 10–20 分钟）
python tools/extra_eval_v7.py run --results-root ../results-v7/pilot --features-base ../features-v7 \
    --output ../extra-eval-v7/pilot --device cpu
python tools/extra_eval_v7.py summarize --output ../extra-eval-v7/pilot
```

- 每个任务训练时用的特征目录从 `<任务>/console/command.json` 里读，`--features-base` 只是找不到记录时的备用。
- 中断后重跑同一条命令，已完成的任务会跳过。
- 正式运行（2000 条）出结果后同样适用：`--results-root ../results-v7/full/v7.1 --output ../extra-eval-v7/full`，建议加 `--max-test-trajectories 50` 控制时间。

## 回传

把 `../extra-eval-v7/` 整个文件夹打包发回即可（只有 JSON，几 MB）：

```bash
zip -r extra-eval-v7.zip ../extra-eval-v7
```
