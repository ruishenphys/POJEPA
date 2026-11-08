# 14 测试与验收

本文件先写第零期（仿真补跑）的五段。DROID 各段的验收要点在架构文档 rev 14 的分段表里，补写 DROID 规格时并入这里。

## 1. 测试分层

| 层 | 在哪跑 | 内容 | 谁跑 |
|---|---|---|---|
| 单元测试 | CPU，几分钟 | 每个新函数的手算值、边界和报错 | 编写 agent 和审核 agent 都跑 |
| v7 回归 | CPU | v7 原有的 `tests/` 在 po-jepa-full 里不改一字全部通过 | 每段都跑 |
| 双树对照冒烟 | CPU，假特征 | 07 §5.2：同样输入下 v7 树和新树的 6 个 v7 条件逐位相同 | S00、S08a、S10 都跑 |
| 端到端冒烟 | CPU，假特征、小规模 | 每个补跑子命令和任务文件从头跑到尾（标定、训练 20 步、D4、汇总） | S17、S18 |
| 真实试跑 | 机哥的两卡机器 | jobs-addon-pilot.txt（16 §7.4），含真实 v7 检查点上的等价核对 | 合并后，正式补跑之前 |

所有测试在 `po-jepa-full/` 下用 `python -m pytest tests -q` 运行；新测试文件命名 `tests/test_addon_*.py`。测试不联网、不需要 GPU、不读真实特征，假特征用 v7 测试里已有的合成方法生成。

## 2. 通用验收条件（每段都要满足）

1. 本段测试和 v7 回归全部通过，输出贴在 PR 里。
2. 只改了本段规格允许改的文件；`git diff` 里没有别的文件。
3. 没有放宽任何已有检查（报错条件、容差、类型检查）。
4. 新增的每个报错都有一条触发它的测试。
5. 所有写文件的地方先写临时文件再换名；JSON 用 `cache._write_json`。

## 3. 分段验收

### S00 脚手架

- `po-jepa-full/` 下有 v7 提交 `911bc09` 的 `src/`、`configs/`、`reference/`、`tests/`、`tools/`、`run.py`、`pyproject.toml`、`DATASET.md`，和 v7 树逐字节相同（`diff -r` 只允许 MANIFEST.json、SOURCE_MAP.json 不同，以及新加的 `.vscode/`、`docs/`、`README.md`）。
- `tools/refresh_manifests.py` 排除 `docs/`（07 §2.8）后重算 MANIFEST；`workflow.verify_bundle` 通过。
- `reference/` 下的协议文件 SHA 和 v7 相同，`data.cache.verify_fixed_protocol` 通过。
- v7 回归测试全部通过。
- 双树对照冒烟脚本 `tests/addon_dual_tree.py`（只在开发机上跑，需要同时有 v7 树）写好并通过：6 个 v7 条件各 20 步，初始权重哈希、每步损失、第 20 步 `model_state`、全部逐窗口评测值逐位相同。
- `.vscode/settings.json` 让 VS Code 识别 `src/` 和 pytest。

### S08a 共享小改（第零期部分）

范围：07 §2.1–§2.3，新文件 `generic_consistency.py`、`ln_metrics.py`，07 §2.4 表里 `consistency_monitor_summary` 和 `early_check` 的推广。

- `GraphSpectrum.spectrum_for` 的结果和依次调用三个函数逐位相同（随机输入，16×16、8×8、12×12 各一次）。
- 双树对照冒烟逐位相同。
- `_loss_config`：4 个标签 × 4 种权重组合（双正、双零、只有时间、只有空间）的接受/拒绝矩阵和 07 §2.3 的表一致；原有的两种组合报错信息不变。
- `train_step(condition="time_only")` 和 `"space_only"` 能跑；权重为 0 的那一项对总损失的贡献恰为 0。
- `generic_consistency`：对同一个 POJEPA 模型和输入，两个函数的输出（全部字段）和 `losses.py` 里的原函数逐位相同；对 `TokenTransformer` 能前向，反传后每个参数都有有限梯度。
- `ln_metrics`：小张量上和手算公式一致（float64，相对误差 < 1e-12）；RMS_LN(a, a) = 0；输入尺度版本和 v7 的 `tcd_normalized` 算法在同一对张量上一致。
- 监控推广：用 v7 测试里的更新记录，po_jepa_full 的 `consistency_monitor_summary` 输出（键和值）和 v7 树的输出逐字节相同；time_only 只检查时间项，space_only 只检查空间项，`checked_terms` 正确。

### S10 新对照

范围：08 全部，07 §2.4（除监控两行）、§2.5–§2.7。

- **fixed_fourier**
  - 8×8、12×12、16×16、20×20 上基底正交归一：‖ΨᵀΨ − I‖_F < 1e-12（float64）。
  - 前 16 个模态的 (p, q) 和特征值与 08 §4.3 的表一致（特征值误差 < 1e-6）。
  - 2×2 平均池化后 u_pq^{16} = ½·cos(πp/32)·cos(πq/32)·u_pq^{8}，误差 < 1e-12。
  - `spectrum_for` 输出的字段、形状、精度、设备和 `GraphSpectrum` 相同；`cutoff_gap` 为 0；`residual_norms` 为 0。
  - `_model_config` 的 `graph` 多出 `basis: "dct2"`，其余和 full 相同。
- **初始权重**：同一种子下 time_only、space_only、fixed_fourier 的 `_model_hash` 等于 po_jepa_full 的；transformer_consistency 的等于 matched_transformer 的。
- **transformer_consistency**：每步 6 次前向、6 次读出；第 0 步（预热系数 0）的参数更新和 matched_transformer 的第 0 步逐位相同；第 1 步起一致性项有非零梯度。
- **koopman**
  - K 的规则给出 m = 2 时 76、m = 7 时 72，参数量 215,744 和 236,992。
  - T = 0 时输出与输入逐位相同。
  - 同一个常数动作段拆成两半积分，结果和不拆在 1e-6（相对）内一致。
  - φ1 在 |w| = 1e-4 两侧的两种算法相对差 < 1e-10。
  - 16×16 上 c_0 = WᵀX 逐位相同；网格坐标和重采样目标不一致时报错；8×8 上 K = 76 > 64 时能前向。
  - 初始化：`basis` 列正交归一，`raw_gamma` 全为 softplus⁻¹(0.1)，`omega` 全 0。
- **5 个新条件**都能在 CPU 合成的仿真批上：前向、反传（全部参数有有限梯度）、跑 20 步、存检查点、读回后 `model_state` 逐位相同；第 10 步中断后续训到第 20 步，和不中断的结果逐位相同。
- **评测分发**：5 个新条件都能跑 `run_prepared_evaluation`（假特征、几个窗口），`expected_type` 检查对错类型报错。
- **检查点隔离**：fixed_fourier 的检查点不能被当成 po_jepa_full 读入（报配置不一致），反之亦然；transformer_consistency 和 matched_transformer 之间同样。

### S17 补跑入口

范围：16 §3、§5 的 `run`、`calibrate-addon`、`check-addon`、`transfer`，`import_v7_calibration.py`、`addon_preflight.py`、`make_addon_jobs.py`、`run_jobs.py` 和 `refresh_manifests.py` 的复制件、`plan_v7.py` 的复制件、`补跑说明.md`。

- `import_v7_calibration.py`：在临时目录里造一个假的 v7 树（含 git HEAD），复制后字节相同、`IMPORTED.json` 正确；HEAD 不对、文件缺、λ 为空、目标已存在且不同，各自报错；重复运行不改任何文件。
- `calibrate-addon`（假特征，把 300/32 调成 3/2 的测试配置）：写出的 yaml 结构和 16 §3.2 相同；λ 等于用同样梯度范数手算的值；两个条件都完成才写文件（中途失败不留半个文件）。
- `check-addon` 在假特征上通过，报告里有 5 个条件的参数量、读出次数、初始哈希关系。
- `make_addon_jobs.py`：4 个阶段的任务数等于 16 §7.5；每条命令能被 `run.py` 的参数解析接受（用 `--help` 之外的干跑模式核对）；路径和 16 §7 一致；同样参数运行两次输出逐字节相同。
- `run_jobs.py` 复制件：默认日志目录、半截输出改名的子命令名单、优先级和 07 §2.8 一致；用假命令测试中断后重跑会跳过完成的、续训 `run`、改名 `d4` 的半截目录。
- `plan_v7.py` 复制件：对 v7 条件，用假编码器在 CPU 上的输出（去掉来源字段）和 v7 树里的结果逐字节相同；接受 5 个新条件。
- 端到端冒烟：假特征上 A0 → C → C2 → D（5 个条件各 20 步）全部完成。

### S18 补跑评测和汇总

范围：10 §3–§7，13 全部。

- `v7load`：用 v7 树在假特征上训练出的检查点，在新树里读入成功；改动以下任一项都报错：检查点字节、`best.step`、`model_config`、`loss_config`、`model_state` 的一个名字、来源里的任一数据字段。`source_files_sha256` 不同不报错，但原样出现在 `source.json`。
- 等价核对：用 v7 树的评测记录作参照，新树重算的三项标准指标逐位相同；人为改动参照里一个值（超过容差）后任务失败、不写 COMPLETE.json。
- D4 各量：用一个可手算的线性玩具模型（Φ_T(z) = e^{−aT} z，模型上的预测满足半群律）核对：所有分割差异、不等分 TCD 和跨网格差异为 0（跨网格时用对 R 可交换的玩具模型）；再用一个故意不满足半群律的玩具模型（Φ_T(z) = (1 − aT) z），分割差异等于手算值。
- 三角不等式核对：随机预测上违反数为 0；构造的违反情形会被检测出来并让任务失败。
- `read_long_window`：T = 16、24、32 读到的起点和终点帧、动作和 `read_window` 拼出来的一致；超出最后一帧报错；`cache.read_window` 未被修改（文件 SHA 和 v7 相同）。
- 截断间隙：在一个已知谱的玩具图上 λ_16、λ_17 和 `cutoff_gap` 等于解析值；输出分位数正确。
- 编码器核对：两个造出来的小权重文件（一份相同、一份改了一个元素、一份缺一个键）分别得到“同一份”“不同”“缺键”的结论。
- `summarize_addon.py`：在假结果目录上出完整的表和图；精度匹配的判断在边界值（恰好 5%）上判为匹配；数据一致性的每一项各造一个不一致的例子，都会报错；自助法用固定种子时结果可复现；重复运行输出逐字节相同。
- 端到端冒烟：假特征上 D4（11 个条件，AC 用缩小规格）→ 截断间隙 → 汇总全部完成。

## 4. 真实试跑的通过标准（合并之后、正式补跑之前）

- jobs-addon-pilot.txt 全部完成，没有失败任务。
- 55 个 D4 任务的等价核对全部通过；记录逐位相同的比例。
- 补跑试跑里 time_only、space_only、fixed_fourier 的初始权重哈希等于 v7 试跑里 po_jepa_full 的值；transformer_consistency 的等于 matched_transformer 的值。
- 每个新条件的每步耗时写进 16 §10，算出正式补跑的时长。
- 试跑的汇总表能完整生成。

试跑只用来核对流程和测速，不看数字下结论，也不根据试跑结果改任何超参数。
