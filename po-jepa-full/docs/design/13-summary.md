# 13 汇总

`tools/summarize_addon.py` 把 v7 正式运行和仿真补跑的结果合在一起出表和图。DROID 的汇总用同一套统计和表格代码，只换输入目录（待 09、12 补写时接上）。

```
python tools/summarize_addon.py --v7-results ../../results-v7/full --addon-results ../../results-addon/full \
    --output ../../results-addon/full/summary
```

## 1. 读哪些结果

- 任务目录的识别和 v7 的 `summarize_v7.find_jobs` 相同：有 `job.json`、`COMPLETE.json`、`report.json`。一个任务被续训过时（`原目录-resumeN`），用完成了的那一次；同一任务有两次都完成算错误。
- 读入：v7 的 D 任务（6 个条件 × 5 个场景 × 3 个种子）、补跑的 D 任务（5 × 5 × 3）、D4（11 × 5 × 3）、截断间隙（5）、编码器核对（1）、变体评测（v7 的 V3 和补跑的 V3）、规划（v7 的 V4、补跑的 V4、D3 的 AC 256 样本）。
- 缺的任务列进 `missing.json`，其余照常汇总，表里对应格子写“缺”。

## 2. 精度匹配规则（看结果之前定死）

恒等预测天然 TCD = SCD = 0，所以“缺陷小”本身不是优点；比较缺陷和泛化时必须在精度相当的预测器之间比。

- **同分布误差** e_c：场景内，条件 c 在 16×16、训练时长 1、2、4、8 的单点查询上的 `future_jepa_distance`，先在每条轨迹内平均，再在轨迹间平均，最后在 3 个种子间平均。
- **匹配**：两个训练过的条件 a、b 满足 |e_a − e_b| ≤ 0.05 · min(e_a, e_b) 才算精度匹配。
- **用在哪**：所有比较缺陷（TCD、SCD、分割差异）和泛化（留出时长、留出网格、变体、长时程）的配对表，每一行都标“匹配 / 不匹配”和误差相差的百分比。论文的证伪条件 1（“精度相当时降低 TCD/SCD 不改善泛化”）只用匹配的配对判断。
- identity 不参与匹配，在表里单独一行。
- **散点图**：每个场景一张，横轴同分布误差（每个种子一个点），纵轴缺陷（`tcd_ln` 在 T ∈ {1, 2, 4, 8} 上的平均；另一张用 `scd_ln` 在 h ∈ {8, 12}、T ∈ {1, 2, 4, 8} 上的平均），identity 用不同标记画在纵轴 0 处。
- 5% 是规格里定的阈值，改动只能在看结果之前、经 Jerry 同意。

## 3. 统计

- **层次**：窗口 → 轨迹内平均 → 轨迹间平均（一个场景、一个种子、一个条件得到一个数）→ 3 个种子的平均和标准差。表里报“种子平均 ± 种子标准差”。
- **配对比较**（和 v7 的 `summarize_v7.paired` 一致）：对同一个窗口键，算 po_jepa_full 减对照，在每条轨迹内平均，报轨迹间的均值、标准误、full 更好的轨迹数。每个种子各报一次。
- **跨种子的置信区间**：按轨迹做整群自助法。每次有放回地抽测试轨迹编号（同一组编号用于所有种子和两个条件），在抽出的轨迹上重算种子平均的配对差，重复 2000 次（`torch.Generator` 种子 1703），取 2.5 和 97.5 分位数。
- 不做多重比较校正，但所有比较都列出来，不挑着报。

## 4. 数据一致性核对

汇总前逐场景核对；任何一项不一致就报错停下，不出这个场景的表：

| 项 | 要求 |
|---|---|
| 特征 | 所有任务记录的特征缓存身份（`preflight` 里的 `cache_identity`、各 manifest 的 SHA）相同 |
| 训练查询序列 | 同一种子下所有训练条件的 `query_sequence_sha256` 相同 |
| 评测查询 | 所有任务 `evaluation/<条件>/registration.json` 里的查询列表相同 |
| σ_z | 所有 PO-JEPA 类条件（含 fixed_fourier）用的 σ_z 等于导入的 v7 标定值 |
| λ | time_only 的 λ_t 等于 v7 full 的 λ_t；space_only 的 λ_s 等于 v7 full 的 λ_s；transformer_consistency、fixed_fourier 的 λ 等于补跑标定文件里的值 |
| 初始权重 | 同一种子下 time_only、space_only、fixed_fourier 的 `initial_model_sha256` 等于 v7 full 的值；transformer_consistency 的等于 v7 matched_transformer 的值 |
| 等价核对 | 每个 D4 任务的等价核对都通过（10 §4.3） |
| 导入的标定 | `configs/calibration/IMPORTED.json` 里记的源文件 SHA 等于 v7 目录里文件的当前 SHA |

## 5. 表

| 表 | 内容 |
|---|---|
| T1 同分布 | 每个场景、每个条件：未来误差、`tcd_ln`、`scd_ln`、`tcd_normalized`、`scd_normalized`、`delta_act`（AC 的原空间 TCD/SCD 标“不可比”） |
| T2 留出 | 留出时长（3、12）、留出网格（12×12）、两者同时留出的未来误差 |
| T3 配对 | po_jepa_full 对每个对照的配对差和置信区间，带精度匹配标记 |
| T4 消融 | po_jepa_full、time_only、space_only、no_consistency 四者的 T1–T2 指标并排（单项消融的主表） |
| T5 分割差异 | §3.3 的 pd 对 x，以及跨网格 xpd、不等分 TCD；每个条件附有效斜率 pd / (x · ε̂)，ε̂ 取同一批窗口上等分 `tcd_normalized`（T ∈ {1, 2, 4, 8}）的中位数 |
| T6 误差分解 | 留出时长 3、12 上 direct、partition、rollout 三项和三角不等式违反数 |
| T7 长时程 | T = 16、24、32 的直接预测误差和 8 步滚动误差，identity 作参照 |
| T8 变体 | 每个场景 6 个变体上相对基础场景的误差变化 |
| T9 规划 | 成功率；AC 同时列 v7 的 64 样本和 D3 的 256 样本结果 |
| T10 截断间隙 | 10 §5 的分位数和判断结果；fixed_fourier 一行为 0 |
| T11 核对与成本 | 编码器核对结论；等价核对的最大差；每个条件每次更新的耗时中位数和显存峰值（来自训练时每步的记录） |

## 6. 图（SVG，沿用 v7 `plots.line_chart` 的画法）

- 精度—缺陷散点（§2）。
- 每个条件的 pd 对 x，叠加 ε̂·x 参照线。
- 收敛曲线：每个任务的 `figures/curves.json` 合在一张图里（训练损失、验证未来误差）。

## 7. 输出

```
<输出>/summary.json     # 全部数字，表和图都从它生成
<输出>/summary.md       # 上面的表
<输出>/figures/*.svg
<输出>/missing.json     # 缺的任务
<输出>/checks.json      # §4 每一项核对的结果
```

先写到临时目录，全部写完再整体换名到 `<输出>`；已有的 `<输出>` 改名为 `<输出>.previousN` 保留。这样 E 段和 V5 段可以反复运行，每次都得到当时已有结果的完整汇总。
