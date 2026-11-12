# 13 汇总

`tools/summarize_addon.py` 把 v7 正式运行和仿真补跑的结果合在一起出表和图（§1–§7）。DROID 的汇总用同一套统计和表格代码，见 §8。

```
python tools/summarize_addon.py --v7-results ../../results-v7/full --addon-results ../../results-addon/full \
    --output ../../results-addon/full/summary
```

## 1. 读哪些结果

- 任务目录的识别和 v7 的 `summarize_v7.find_jobs` 相同：有 `job.json`、`COMPLETE.json`、`report.json`。一个任务被续训过时（`原目录-resumeN`），用完成了的那一次；同一任务有两次都完成算错误：在写任何文件之前报 `ValueError`（列出两个目录），不像 v7 的 `summarize_v7` 那样只用第一个、把第二个记进 `duplicates`。
- 读入：v7 的 D 任务（6 个条件 × 5 个场景 × 3 个种子）、补跑的 D 任务（5 × 5 × 3）、D4（11 × 5 × 3）、截断间隙（5）、编码器核对（1）、变体评测（v7 的 V3 和补跑的 V3）、规划（v7 的 V4、补跑的 V4、D3 的 AC 256 样本）。
- 缺的任务列进 `missing.json`，其余照常汇总，表里对应格子写“缺”。E 段在试跑和正式运行都要跑，所以种子和场景不写死，从找到的任务里取：`missing.json` 列的是“找到的场景 × 找到的种子 × 11 个条件”里缺的 D、D4 任务，以及缺的截断间隙和编码器核对。一个种子或一个场景的任务全部缺时不会列出来，要看最后一行的任务数和 `summary.json` 每个格子的 `seeds`。变体评测和规划报告不列缺（数目随变体表变）。

## 2. 精度匹配规则（看结果之前定死）

恒等预测天然 TCD = SCD = 0，所以“缺陷小”本身不是优点；比较缺陷和泛化时必须在精度相当的预测器之间比。

- **同分布误差** e_c：场景内，条件 c 在 16×16、训练时长 1、2、4、8 的单点查询上的 `future_jepa_distance`，先在每条轨迹内平均，再在轨迹间平均，最后在 3 个种子间平均。
- **匹配**：两个训练过的条件 a、b 满足 |e_a − e_b| ≤ 0.05 · min(e_a, e_b) 才算精度匹配。
- **用在哪**：所有比较缺陷（TCD、SCD、分割差异）和泛化（留出时长、留出网格、变体、长时程）的配对表，每一行都标“匹配 / 不匹配”和误差相差的百分比。论文的证伪条件 1（“精度相当时降低 TCD/SCD 不改善泛化”）只用匹配的配对判断。
- **配对表的范围**：T3 有 15 个量：T1 的 6 个、T2 的 3 个、`pd_ln`、`pd_in`（7 个分割的窗口值一起平均）、`xpd_ln`、`xpd_in`（4 个一起平均）、`long_direct`、`long_composed`（3 个时长一起平均）；变体的匹配标记写在 T8 的每一行（那一行的条件对 po_jepa_full）。不等分 TCD、留出分解、各个分割单独的配对不进 T3（在 T5、T6 里按条件列）。AC 在 4 个输入尺度的量（`tcd_normalized`、`scd_normalized`、`pd_in`、`xpd_in`）上标“不可比”（10 §3.1、§3.8）（Jerry 2026-11-08 定：按推荐做法）。
- identity 不参与匹配，在表里单独一行。
- **散点图**：每个场景一张，横轴同分布误差（每个种子一个点），纵轴缺陷（`tcd_ln` 在 T ∈ {1, 2, 4, 8} 上的平均；另一张用 `scd_ln` 在 h ∈ {8, 12}、T ∈ {1, 2, 4, 8} 上的平均）。v7 的 `plots.line_chart` 只画折线，所以每个条件画一条按横轴排序的折线（每个种子一个点），identity 一条画在纵轴 0、图例写“identity（纵轴 0）”，不另写散点函数。
- 5% 是规格里定的阈值，改动只能在看结果之前、经 Jerry 同意。

## 3. 统计

- **层次**：窗口 → 轨迹内平均 → 轨迹间平均（一个场景、一个种子、一个条件得到一个数）→ 3 个种子的平均和标准差。表里报“种子平均 ± 种子标准差”。
- **配对比较**（和 v7 的 `summarize_v7.paired` 一致）：对同一个窗口键，算 po_jepa_full 减对照，在每条轨迹内平均，报轨迹间的均值、标准误、full 更低的轨迹数（`full_lower`：full − 对照 < 0 的轨迹数；对误差和缺陷更低就是更好，对 `delta_act` 不是，所以表头写“full 更低”，不替读者判断好坏）。每个种子各报一次。
- **跨种子的置信区间**：按轨迹做整群自助法。每次有放回地抽测试轨迹编号（同一组编号用于所有种子和两个条件），在抽出的轨迹上重算种子平均的配对差，重复 2000 次（`torch.Generator` 种子 1703），取 2.5 和 97.5 分位数。
  - 细节：(a) 抽样的轨迹集合是所有种子的配对差里都有的轨迹编号的交集（正式运行每个场景 200 条测试轨迹全有，交集就是全部）；(b) 每一个比较（每个场景 × 对照 × 量）都从 `torch.Generator().manual_seed(1703)` 重新开始抽，所以同一场景里所有比较用的是同一组 2000 × 200 的编号；(c) 每次抽样里，每个种子在抽出的轨迹（重复的照算）上平均配对差，再对种子平均；(d) 分位数用 `torch.quantile` 的线性插值（float64）（Jerry 2026-11-08 定：按推荐做法）。
- 不做多重比较校正，但所有比较都列出来，不挑着报。

## 4. 数据一致性核对

汇总前逐场景核对；任何一项不一致就报错停下，不出这个场景的表和图：这个场景只在 `checks.json` 和 `summary.json` 里记 `status: "blocked"` 和 `failed`，其他场景照常汇总，最后退出码 1（S17 的运行器把退出码 1 记为失败，所以 E 段会显示失败）：

| 项 | 要求 |
|---|---|
| 特征 | 所有任务记录的特征缓存身份（`preflight` 里的 `cache_identity`、各 manifest 的 SHA）相同 |
| 训练查询序列 | 同一种子下所有训练条件的 `query_sequence_sha256` 相同 |
| 评测查询 | 所有任务 `evaluation/<条件>/registration.json` 里的查询列表相同 |
| σ_z | 所有 PO-JEPA 类条件（含 fixed_fourier）用的 σ_z 等于导入的 v7 标定值 |
| λ | time_only 的 λ_t 等于 v7 full 的 λ_t；space_only 的 λ_s 等于 v7 full 的 λ_s；transformer_consistency、fixed_fourier 的 λ 等于补跑标定文件里的值 |
| 初始权重 | 同一种子下 time_only、space_only、fixed_fourier 的 `initial_model_sha256` 等于 v7 full 的值；transformer_consistency 的等于 v7 matched_transformer 的值 |
| 等价核对 | 每个 D4 任务的等价核对都通过（10 §4.3） |
| 导入的标定 | `configs/calibration/IMPORTED.json` 的 `schema` 是 `addon_v7_calibration_import_v1`；`files` 的每一项 `sha256` 等于 `source`（v7 目录里的原文件）的当前 SHA，也等于导入的副本的 SHA；`imported_at`、`v7_commit` 不核（S17 的入口已经核过） |

σ_z 和 λ：v7 和补跑的任务报告里没有直接记这两个数，从每个 D 任务训练目录里选中检查点的 `model_config.graph.sigma_z` 和 `loss_config.time_weight`/`space_weight` 读（`torch.load(weights_only=True)`）；续训、只评测的任务沿 `training_run`/`resume_from` 找训练目录（10 §4.2 第 2 步）。读不到算核对不通过。缺 `registration.json` 的任务算“评测查询”不通过；没有训练目录的训练条件算“训练查询序列”“σ_z”“λ”不通过。这些都是报错停下的那一类，不算缺。

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
| T11 核对与成本 | 编码器核对结论；等价核对的最大差；每个条件每次更新的耗时中位数和显存峰值（来自训练时每步的记录：一个场景里这个条件所有种子的全部 `updates/*.json` 合在一起，取 `update_and_observer_seconds` 的中位数（v7 每步记录里只有这一个耗时，含观察器）和 `gpu_memory.peak_allocated` 的最大值；CPU 上是 `null`，表里写“缺”） |

T1、T2 各量取哪些查询（文件头 `ROW_METRICS`、`D4_GROUPS`）：`future`、`tcd_normalized`、`delta_act` 取 16×16、训练时长 1、2、4、8 的单点查询（和 §2 的 e_c 相同）；`scd_normalized` 取 16→8、16→12 两种、时长 1、2、4、8；`tcd_ln` 取 T ∈ {1, 2, 4, 8}，`scd_ln` 取 h ∈ {8, 12}、T ∈ {1, 2, 4, 8}（和 §2 的散点图纵轴相同）；同一个窗口的几个时长、几个网格先在窗口内一起平均（窗口键带上量的名字，在轨迹内和窗口一起平均）。T2：留出时长是 16×16 上 T ∈ {3, 12}，留出网格是 12×12 上 T ∈ {1, 2, 4, 8}，两者同时是 12×12 上 T ∈ {3, 12}（Jerry 2026-11-08 定：按推荐做法）。

T5 的 ε̂ 和有效斜率：ε̂ 是每个 (场景, 种子, 条件) 上 D4 窗口 × 4 个时长共 1600 个 `tcd_normalized` 的中位数（D4 等价核对里重算的值，等于 v7 记录的值），不再除 (T/2)²：v7 的 `tcd_normalized` 已经除过 τσ = (T/2)²，正是等分 2 段的 x，所以它本来就是“每单位 x 的输入尺度缺陷”，和 pd_in / x 同量纲。有效斜率 = pd_in（轨迹宏平均）/(x · ε̂)，每个种子一个值，表里报种子平均 ± SD；pd_ln 不算斜率（ε̂ 是输入尺度的）。T5 的图里参照线是 po_jepa_full 的 ε̂·x（Jerry 2026-11-08 定：按推荐做法）。

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

## 8. DROID 汇总

`summary` 任务（`python tools/droid.py summary …`，12 §2）用 §2、§3 的同一套统计和表格代码，只换输入和聚类单位。

### 8.1 读哪些结果

运行根 `<name>/` 下的全部评测任务（26 个）、`eval/spectral-gap`、每个训练任务的 `report.json` 和 `train-log.jsonl`、`check/env.json`、标定文件。缺的任务列进 `missing.json`，表里写“缺”。

### 8.2 统计

- **层次**：窗口 → 回合内平均 → 回合间平均（一个测试集、一个种子、一个条件得到一个数）→ 3 个种子的平均和标准差。同分布测试集每个场景正好 5 条回合，所以回合间平均也等于场景间平均。
- **精度匹配**：e_c 取同分布测试集上 16×16、T ∈ {1, 2, 4, 8} 的未来误差，其余同 §2（5% 规则不变）。留出实验室测试集的比较沿用同分布测试集上的匹配结论，不另算。
- **配对比较**：po_jepa_full 减每个对照，按窗口键配对，回合内平均。
- **自助法的整群单位**：同分布测试集按场景（100 个），留出实验室测试集按回合（500 条；留出实验室只有 3 个，太少，不按实验室抽）。2000 次，种子 1703，同一组抽样用于所有种子和条件。
- **成败分开**：每张表都再按回合的 outcome 分成成功、失败两份各算一遍（架构决策 3）。

### 8.3 数据一致性核对

| 项 | 要求 |
|---|---|
| 特征 | 所有训练和评测任务记录的 `feature_manifest_sha256` 相同 |
| 划分和窗口 | 所有任务记录的 `splits.json`、`windows/*.json`、`action_scale.json` 的 SHA 相同 |
| 训练集 | 所有训练任务的训练集 uuid 列表 SHA 相同 |
| σ_z | 所有 PO-JEPA 类条件登记的 σ_z 等于 `sigma_z.json` 的值 |
| λ | po_jepa_full、fixed_fourier、transformer_consistency 登记的 λ 等于各自的标定文件 |
| 初始权重 | 同一种子下 09 §2.4 的关系成立 |
| 训练日志 | 每个训练任务的日志覆盖 1…U 不重不漏，完成步数等于 U |
| 评测 | 每个评测任务读的检查点 SHA 等于对应训练任务 `selection.json` 的最优；三角不等式违反数为 0 |

第三轮审核修复（e9016a8，见 `docs/segments/review-fix-e9016a8.md`）：读每个评测和训练目录时先核报告、`job.json`、`source.json`（或训练登记）里的 (条件, 种子) 都等于目录槽位、`job.json` 等于报告的 job，不等就报错退出（RF-P1-1：目录槽位不是身份）；“评测”另核 source.json 记的训练登记 SHA 等于该训练目录的；“σ_z”另核 `sigma_z.json` 的动作尺度 SHA 等于数据根的、截断间隙、PO-JEPA 类训练和 check/env 读的是同一份 `sigma_z.json`（RF-P1-2）；“特征”把 `sigma_z.json` 和 `check/env.json` 记的特征总清单也算进去；“配置”包括 `check/env.json`（RF-P2-4）。

第四轮审核修复（fe48a5c，见 `docs/segments/review-fix-fe48a5c.md`）：训练读取器 `read_train` 另读 `selection.json` 选中的检查点（先核文件 SHA，再用 mmap 只取来源字典），检查点里训练时写下的 `droid_training`（条件、种子、registration.json 的文件 SHA）必须等于登记和槽位，缺这一项也拒绝（R4-1-P1-1：登记、报告、source、job 能一起改写再重新绑定，检查点里的这一项不重新训练就改不了）。

### 8.4 表

| 表 | 内容 |
|---|---|
| D1 主表 | 两个测试集各一份：未来误差（16×16、8×8，T = 1、2、4、8）、`tcd_ln`、`scd_ln`、`tcd_normalized`、`scd_normalized`、`delta_act` |
| D2 留出 | 时长留出（3、12、16）、网格留出（12×12）、两者同时留出 |
| D3 长时程 | T = 24、32 的直接预测和 8 步滚动，identity 作参照，注明用到的窗口数 |
| D4 原始帧率 | 单步 1/3、2/3、4/3 和两种步长的滚动，附 δ = 1 的参照 |
| D5 原始分辨率 | 12×12（192 输入）、20×20（320 输入）和同一批窗口上 16×16 的对照 |
| D6 规划 | 动作排名的三项指标（附随机水平）、CEM 的净位移误差和相对误差 |
| D7 配对 | po_jepa_full 对每个对照的配对差、置信区间和精度匹配标记 |
| D8 分割差异和误差分解 | 10 §3.3–§3.6 的量 |
| D9 成功和失败 | D1、D2 的主要指标按 outcome 分开 |
| D10 截断间隙 | 10 §8.8 |
| D11 成本 | 每个条件每步耗时的中位数、显存峰值、GPU 小时（来自训练日志）、续跑次数、求谱后端 |

图：每个测试集一张精度—缺陷散点；pd 对 x；收敛曲线；原始帧率下误差随步长的变化。输出目录结构和 §7 相同，放在 `<name>/summary/`。

### 8.5 S13 实现补注（2026-11-10）

代码 `tools/droid_summary.py`，测试 `tests/test_droid_summary.py`，说明和验收单 `docs/segments/S13.md`。统计、精度匹配、表格格式和画图复用 `tools/summarize_addon.py`。实现时定的几条（不改上面的口径）：

- 训练任务的产物（09 §4–§5）、`check/env.json` 和评测任务 job.json/source.json 里的来源字段格式由 S09 定，汇总只经过三个读取函数（`read_train`、`read_eval_view`、`read_env`，S09b 写好：`read_train` 读 registration/report/selection/train-log、`read_eval_view` 读 `eval_job` 写的 job.json 并核 source.json 的 SHA、`read_env` 只认通过的 env.json），§8.3 的核对和 D11 对着 `TrainView`、`EvalView` 写好；`tools/droid.py summary` 已接上（S09b）。
- 场景 = `index/episodes.jsonl` 的 (lab, scene_id)。自助法在抽到的群里按回合加权平均：每个场景回合数相同时就是场景平均，群是单条回合时和 §3 的按轨迹自助法逐位相同。
- 未来误差在 D1 里把 T = 1、2、4、8 合在一起（和 13 §5 T1 相同），另出按时长分开的表；AC 的 `tcd_ln` 在 T = 1、3 上是插值，在按时长的表里单列（10 §8.6）。D2 时长留出既给 3、12、16 合起来的值（进 D7），也给每个时长。
- ε̂（D8）取一个 (测试集, 种子, 条件) 上同分布时长全部窗口的 `tcd_normalized` 的中位数：DROID 每个窗口只有一个时长，没有仿真那种“同一批窗口 × 4 个时长”。
- 不训练的对照（identity、vjepa2_ac_zeroshot）只有一个任务，种子标签 `untrained`，配对时和 po_jepa_full 的每个种子配。vjepa2_ac_zeroshot 的初始权重和它登记的 `construction_seed` 那个种子的 vjepa2_ac 比（09 §2.4）。
- 核对另加两项：“配置”（所有任务的 config_sha256 = 本配置）和“评测窗口”（各条件评测的窗口、原始帧率起点、分辨率窗口、规划窗口编号相同；rows 的 outcome = 索引；id_test 回合都有场景）。任一项不通过：只写 summary.json（`status: blocked`）、checks.json、missing.json、summary.md，不出表和图、不写 COMPLETE.json，退出 1。
- 读不到 σ_z 或 λ 标定文件而有对应的训练任务时，算核对不通过（13 §4 的规则）；对照的训练任务缺时“初始权重”不比、只进 missing.json；评测任务对应的训练任务缺时“评测”不通过。
