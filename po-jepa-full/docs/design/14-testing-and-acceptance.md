# 14 测试与验收

§3 先列第零期（仿真补跑）的五段，再列 DROID 各段；§4 是仿真补跑的真实试跑标准，DROID 的试跑关口在 12 §6。

## 1. 测试分层

| 层 | 在哪跑 | 内容 | 谁跑 |
|---|---|---|---|
| 单元测试 | CPU，几分钟 | 每个新函数的手算值、边界和报错 | 编写 agent 和审核 agent 都跑 |
| v7 回归 | CPU，约 4 分钟 | v7 原有的 `tests/` 在 po-jepa-full 里不改一字通过；没有 CUDA 13.0 版 torch 的机器上只允许 `test_delivery.py` 的 4 个 CUDA 版本失败（07 §5.1） | 每段都跑 |
| 双树对照冒烟 | CPU，假特征，1 线程，约 2 小时，后台跑 | 07 §5.2：同样输入下 v7 树和新树的 6 个 v7 条件逐位相同 | S00、S08a、S10 都跑 |
| 端到端冒烟 | CPU，假特征、小规模 | 每个补跑子命令和任务文件从头跑到尾（标定、训练 20 步、D4、汇总） | S17、S18 |
| 真实试跑 | 机哥的两卡机器 | jobs-addon-pilot.txt（16 §7.4），含真实 v7 检查点上的等价核对 | 合并后，正式补跑之前 |

所有测试在 `po-jepa-full/` 下用 `python -m pytest tests -q` 运行；新测试文件命名 `tests/test_addon_*.py`。测试不联网、不需要 GPU、不读真实特征，假特征用 v7 测试里已有的合成方法生成。

“逐位相同”的测试（续跑和不中断比、两次运行比、新旧代码比）在 1 线程下做：同一进程里的两次运行用 `torch.set_num_threads(1)`（比完恢复原值）；跨进程的比较另加 07 §5.2 的固定环境变量。v7 在多线程 CPU 上本身就不能逐位复现。

下文“测试全过”“全绿”都指除了上表那 4 个 CUDA 版本失败以外没有失败。

## 2. 通用验收条件（每段都要满足）

1. 本段测试和 v7 回归全部通过（§1 的 4 个 CUDA 版本失败除外），输出贴在 PR 里。
2. 只改了本段规格允许改的文件；`git diff` 里没有别的文件。
3. 没有放宽任何已有检查（报错条件、容差、类型检查）。
4. 新增的每个报错都有一条触发它的测试。
5. 所有写文件的地方先写临时文件再换名；JSON 用 `cache._write_json`（双树对照脚本的父进程不能导入任何一棵树的 `po_jepa`，用一个行为相同的小函数）。
6. 改了进清单的文件（后缀 `.py .yaml .md .json .toml .tex`，`docs/`、`.vscode/`、`README.md`、`补跑说明.md` 以外），就在 `po-jepa-full/` 下重跑 `tools/refresh_manifests.py`，新的 `MANIFEST.json` 和 `SOURCE_MAP.json` 一起提交（改了 `SOURCE_MAP` 登记的模块而不重算，`test_bundle_core_math_matches_source_map` 会失败）；在新目录里真的调用一次 `workflow.verify_bundle()` 和 `cache.verify_fixed_protocol()` 并通过（07 §5.1）。

## 3. 分段验收

### S00 脚手架

- 从 v7 提交 `911bc09` 复制：`po-jepa-full/` 下的 `src/`、`configs/`、`reference/`、`run.py`、`pyproject.toml`、`DATASET.md`、`SOURCE_MAP.json` 和 911bc09 逐字节相同；`tests/` 只多 `addon_dual_tree.py`、`test_addon_dual_tree.py` 两个文件；`tools/` 只有 `refresh_manifests.py` 差 07 §2.8 的三行；另加 `.vscode/settings.json`，`MANIFEST.json` 重算。`README.md` 和 `docs/` 在设计提交里已有，本段不动。
- 新目录里重算的 MANIFEST 有 90 项，不含 `docs/`、`.vscode/` 下的文件和 `README.md`；`workflow.verify_bundle` 在新目录通过。
- `reference/` 下的协议文件 SHA 和 v7 相同，`cache.verify_fixed_protocol` 通过。
- v7 回归测试通过（§1 的 4 个 CUDA 版本失败除外，和根目录的 v7 树相同）。
- 双树对照冒烟脚本 `tests/addon_dual_tree.py` 和它的比较器单元测试写好，按 07 §5.2 跑一遍逐位相同。
- `.vscode/settings.json` 让 VS Code 识别 `src/` 和 pytest。
- 逐条的验收命令和期望输出见 `docs/segments/S00.md` §7（A1–A22）。

### S08a 共享小改（第零期部分）

范围：07 §2.1–§2.3，新文件 `generic_consistency.py`、`ln_metrics.py`，07 §2.4 表里 `consistency_monitor_summary` 和 `early_check` 的推广。

- `GraphSpectrum.spectrum_for` 的结果和依次调用三个函数逐位相同（随机输入，16×16、8×8、12×12 各一次）；`model_device` 原样转交（CPU 上只能用替身的 `lowest_eigenpairs` 核）。
- 双树对照冒烟逐位相同。
- `_loss_config`：4 个标签 × 4 种权重组合（双正、双零、只有时间、只有空间）的接受/拒绝矩阵和 07 §2.3 的表一致；原有的两种组合报错信息不变。
- `train_step(condition="time_only")` 和 `"space_only"` 能跑；权重为 0 的那一项对总损失的贡献恰为 0。
- `generic_consistency`：对同一个 POJEPA 模型和输入，两个函数的输出（全部字段）和 `losses.py` 里的原函数逐位相同，比较用不等分的 τ、σ（例如 τ = T/4），否则 τ、σ 互换查不出来；对 `TokenTransformer` 能前向；`p_out` 是零初始化的，所以先把 `p_out.weight` 用 std = 0.02 重设，再要求反传后每个参数的梯度存在、有限，且梯度平方和 > 0。
- `ln_metrics`：小张量上和手算公式一致（float64，相对误差 < 1e-12），a、b 独立生成、量级不同（b = a + 小扰动时 float32 相减是精确的，分不出先转 float64 和后转）；RMS_LN(a, a) = 0；输入尺度版本和 v7 的 `tcd_normalized` 算法在同一对张量上相对差 ≤ 1e-7（10 §3.1）。
- 监控推广：po_jepa_full 的 `consistency_monitor_summary` 输出和 `early-check.json` 和 v7 树的逐字节相同。单元测试里没法跑 v7 树，所以用在 911bc09 上实算的金标准字符串，并给出在 911bc09 上复核它的命令；双树对照的冒烟把提前检查关掉了，`early_check` 只靠这条核。time_only 只检查时间项，space_only 只检查空间项，`checked_terms` 正确；identity 和其余 v7 名字不监控。

### S10 新对照

范围：08 全部，07 §2.4（除监控两行）、§2.5–§2.7。分成两段：S10a 是 `fourier.py`、`koopman.py` 两个新模块和它们的测试（下面 fixed_fourier、koopman 两组里的公式项），S10b 是其余各项和双树对照；逐条验收见 `docs/segments/S10a.md`、`S10b.md` 的 §7。

- **fixed_fourier**
  - 8×8、12×12、16×16、20×20 上基底正交归一：‖ΨᵀΨ − I‖_F < 1e-12（float64）。
  - 前 16 个模态的 (p, q) 和特征值与 08 §4.3 的表一致（特征值误差 < 1e-6）。
  - 2×2 平均池化后 u_pq^{16} = ½·cos(πp/32)·cos(πq/32)·u_pq^{8}，误差 < 1e-12。
  - `spectrum_for` 输出的字段、形状、精度、设备和 `GraphSpectrum` 相同；`cutoff_gap` 为 0；`residual_norms` 为 0。
  - `_model_config` 的 `graph` 多出 `basis: "dct2"`，其余和 full 相同。
- **初始权重**：同一种子下 time_only、space_only、fixed_fourier 的 `_model_hash` 等于 po_jepa_full 的；transformer_consistency 的等于 matched_transformer 的。
- **transformer_consistency**：每步 6 次前向、6 次读出；第 0 步（预热系数 0）的参数更新和 matched_transformer 的第 0 步逐值相同（一致性项贡献 ±0，JEPA 梯度为 −0 的位置可能变成 +0，用 `torch.equal` 比）；第 1 步起一致性项有非零梯度（初始化时 `p_out` 为零、一致性梯度恰为 0，测试另用扰动过 `p_out` 的模型核）。
- **koopman**
  - K 的规则给出 m = 2 时 76、m = 7 时 72，参数量 215,744 和 236,992。
  - T = 0 时输出与输入逐位相同。
  - 同一个常数动作段拆成两半积分，结果和不拆在 1e-6（相对）内一致。
  - φ1 在 |w| = 1e-4 两侧的两种算法相对差 < 1e-10。
  - 16×16 上 c_0 = WᵀX 逐位相同；网格坐标和重采样目标不一致时报错；8×8 上 K = 76 > 64 时能前向。
  - 初始化：`basis` 列正交归一，`raw_gamma` 全为 softplus⁻¹(0.1)，`omega` 全 0。
- **5 个新条件**都能在 CPU 合成的仿真批上：前向、反传（全部参数有有限梯度；残差读出的条件里除 `p_out.bias` 外每个梯度非零，`p_out.bias` 的梯度恰为 0）、跑 20 步、存检查点、读回后 `model_state` 逐位相同；在第 10 步和第 15 步各中断一次（都从第 10 步的检查点续），续训到第 20 步，和不中断的结果逐位相同。
- **评测分发**：5 个新条件都能跑 `run_prepared_evaluation`（假特征、几个窗口），`expected_type` 检查对错类型报错。
- **检查点隔离**：fixed_fourier 的检查点不能被当成 po_jepa_full 读入（报配置不一致），反之亦然；transformer_consistency 和 matched_transformer 之间同样。

### S17 补跑入口

范围：16 §3、§5 的 `run`、`calibrate-addon`、`check-addon`、`transfer`，`import_v7_calibration.py`、`addon_preflight.py`、`make_addon_jobs.py`、`run_jobs.py` 的复制件、`plan_v7.py` 的复制件、`补跑说明.md`（`refresh_manifests.py` 的排除项已在 S00 做完，本段只在 `MUTABLE` 加 `补跑说明.md`，+2 −2，07 §2.8）。另外允许改 S10b 的测试 `tests/test_addon_conditions.py` 里 `test_consumer_sources_register_new_files` 的 2 行（+3 −2）：期望名单的后半段按 `runtime/addon/` 的 glob 取，另核 S10b 的 3 个文件仍在（07 §2.7；S10b 的测试写法以后照这个）。

- `import_v7_calibration.py`：在临时目录里造一个假的 v7 树（含 git HEAD），复制后字节相同、`IMPORTED.json` 正确；HEAD 不对、文件缺、λ 为空、目标已存在且不同，各自报错；重复运行不改任何文件。
- `calibrate-addon`（假特征，把 300/32 调成 3/2 的测试配置）：写出的 yaml 结构和 16 §3.2 相同；λ 等于用同样梯度范数手算的值；两个条件都完成才写文件（中途失败不留半个文件）。
- `check-addon` 在假特征上通过，报告里有 5 个条件的参数量、读出次数、初始哈希关系。
- `make_addon_jobs.py`：4 个阶段的任务数等于 16 §7.5；每条命令能被 `run.py` 的参数解析接受（用 `--help` 之外的干跑模式核对）；路径和 16 §7 一致；同样参数运行两次输出逐字节相同。
- `run_jobs.py` 复制件：默认日志目录、半截输出改名的子命令名单、优先级和 07 §2.8 一致；用假命令测试中断后重跑会跳过完成的、续训 `run`、改名 `d4` 的半截目录。
- `plan_v7.py` 复制件：对 v7 条件，用假编码器在 CPU 上的输出（去掉来源字段）和 v7 树里的结果逐字节相同；接受 5 个新条件。
- 端到端冒烟：假特征上 A0 → C → C2 → D（5 个条件各 20 步）全部完成。

### S18 补跑评测和汇总

范围：10 §3–§7，13 全部。另外允许改的已有测试只有 S10b 的 `tests/test_addon_conditions.py`：S17 已合并时不碰它；S17 还没合并时照抄 S17 的那 2 行（逐字相同，改完和 S17 的结果是同一份），这样两段可以并行、谁先合并终态都相同。

- `v7load`：用 v7 树在假特征上训练出的检查点，在新树里读入成功；改动以下任一项都报错：检查点字节、`best.step`、`model_config`、`loss_config`、`model_state` 的一个名字、来源里的任一数据字段。`source_files_sha256` 不同不报错，但原样出现在 `source.json`。
- 等价核对：用 v7 树的评测记录作参照，新树重算的三项标准指标逐位相同；人为改动参照里一个值（超过容差）后任务失败、不写 COMPLETE.json。
- D4 各量：用一个可手算的线性玩具模型（Φ_T(z) = e^{−aT} z，模型上的预测满足半群律）核对：所有分割差异、不等分 TCD 和跨网格差异为 0（跨网格时用对 R 可交换的玩具模型）；再用一个故意不满足半群律的玩具模型（Φ_T(z) = (1 − aT) z），分割差异等于手算值。
- 三角不等式核对：随机预测上违反数为 0；构造的违反情形会被检测出来并让任务失败。
- `read_long_window`：T = 16、24、32 读到的起点和终点帧、动作和 `read_window` 拼出来的一致；超出最后一帧报错；`cache.read_window` 未被修改（文件 SHA 和 v7 相同）。
- 截断间隙：在一个已知谱的玩具图上 λ_16、λ_17 和 `cutoff_gap` 等于解析值；输出分位数正确。
- 编码器核对：两个造出来的小权重文件（一份相同、一份改了一个元素、一份缺一个键）分别得到“同一份”“不同”“缺键”的结论。
- `summarize_addon.py`：在假结果目录上出完整的表和图；精度匹配的判断在边界值（恰好 5%）上判为匹配；数据一致性的每一项各造一个不一致的例子，都会报错；自助法用固定种子时结果可复现；重复运行输出逐字节相同。
- 端到端冒烟：假特征上 D4（11 个条件，AC 用缩小规格）→ 截断间隙 → 汇总全部完成。

### DROID 各段（第一到第四期）

DROID 段的测试文件命名 `tests/test_droid_*.py`。金标准回合的参照值在 `docs/design/droid-golden/`（说明见那里的 README）；需要真实视频、真实元数据或 GPU 的测试在缺这些东西时跳过，并在 PR 里贴出在开发机或 GPU 机器上跑过的输出。

#### S01 配置和 7 维动作（01）

- `load_config`：base 加 overlay 的合并结果正确；overlay 里出现名单外的键、base 里没有的键、合并后缺键、`name` 仍是 base，各自报错；同样的文件两次读出的 `config_sha256` 相同，键的书写顺序不影响它。
- pilot 和 full 的推导量（学习率预热、一致性预热、验证间隔、早检步数、一致性测量间隔）等于 01 §2 的表；不能整除时报错。
- 01 §3 的每条校验各有一个触发它的例子。
- `poses_to_diffs` 的函数体和上游 `app/vjepa_droid/droid.py`（204698b）里的逐字相同（按 AST 比较），在金标准回合上的输出和参照逐位相同。
- `droid_control_spec` 的每个字段等于 01 §5.3；`loss_config` 只有两个权重来自参数，其余等于 01 §6。

#### S02 索引（02 §1–§4、§7）

- 假的列表接口（本地服务器）：翻页、失败重试的间隔、重试耗尽报错、条数不等于 74,896 报错、MD5 不对报错。
- 02 §3 的每条字段规则（含 `tri`/`TRI`、uuid 和文件名不一致、`h5` 不在回合文件夹下、`left_mp4` 不在 `recordings/MP4/` 下、相机序列号对不上）各有测试。
- 02 §4 的四种原因各自能被判出，一条回合可以同时有多种原因。
- 开发机上在真实元数据上跑：入选 32,585 条，各实验室条数和 02 §4 一致；`--verify-sample 100` 通过。

#### S03 划分（02 §5–§7）

- 在真实元数据上（开发机）：02 §5 验收表的每一项都相等。
- 在合成元数据上：同一场景不跨训练、验证、同分布测试；任何 N 的训练集是更大 N 的前缀；改 N 时验证集和两个测试集不变；跨实验室的场景号被排除出候选；同样输入两次输出逐字节相同。

#### S04 下载（03）

本地假服务器上：正常下载；MD5 不对时重下、3 次后失败并写明对象名；内容被截断；500 错误重试；404 直接失败；启动时删掉残留的 `.part`；有 `DOWNLOADED.json` 且大小对时跳过；有 `RAW_DELETED.json` 时跳过；片完成记录的内容；中断后重跑不重复下载已完成的回合。

#### S05 准备（04，含 06 §3 的冻结窗口）

- 金标准回合：`frame_index`、`state`、`command`、两种动作、`capture_ms`、每帧的 SHA 和参照逐位相同。
- 合成 h5 和视频覆盖 04 §4 的每一项检查失败；失败的回合照样写输出，`passed_*` 和 `reasons` 正确。
- `load_prepared` 发现数组被改动时报错；JSON 两次写出字节相同。
- `prepare/stats`：动作尺度在合成数据上等于手算的分位数；回合池按 04 §6 取；任一片失败比例超过 1% 时任务失败。
- 冻结窗口：条数和 06 §3 的规则一致；起点在 [0, F − 1 − d] 内；替身和干扰项都来自别的回合，干扰项正好 63 个；CEM 每条回合每个 d 只有一个窗口；同样输入两次输出字节相同；`val.json` 只有 `windows` 和 `counts`。

#### S06 编码（05）

- 金标准帧上 `preprocess_square(rgb, 256)` 和 v7 `preprocess_rgb` 逐位相同；裁剪的行列和 05 §1 一致。
- 假编码器跑通：特征文件、每行 SHA、片清单、总清单 SHA；改动一个字节后 `FeatureReader` 报错；帧 SHA 和准备文件不同时报错；05 §3 的跳过规则。
- `--delete-raw` 只在清单重读核对通过后才删，并写 `RAW_DELETED.json`；验证片和测试片不删。
- 测试附加：s5 的行数等于 ⌊(V − 1)/5⌋ + 1；r192、r320 的 token 数为 144、400。
- `compute_new_pose`（`droid/poses.py`）和官方函数在金标准状态和动作上逐位相同。
- 合并前在 GPU 机器上做 05 §7 的两项核对。

#### S07 窗口和批（06）

- `step_rng` 只由（种子，步数，流）决定；同一种子下 PO-JEPA 类、Transformer 类、Koopman 第 k 步拿到同样的 64 个窗口；起点范围正确。
- `make_training_batch` 的输出通过 v7 `training._batch` 和 `comparison._tf_batch` 的全部检查；端点秒数逐位精确；padding 段的 NaN 不被读到。
- 数据加载器用 0、1、2 个工作进程产出的批逐字节相同；从第 k 步开始产出的批和从头产出的第 k 步相同。
- AC 片段的形状、单位和取值等于 06 §4；`make_eval_batch` 带出的起点状态正确。

#### S08b 共享小改（第二期部分，09 §6）

- v7 原有测试全部通过；双树对照冒烟逐位相同；6 个 v7 条件的 `TransformerSpec`、`ACSpec` 快照和 v7 逐字节相同。
- 7 维的 Transformer、identity、两种 AC 都能前向和反传，每个参数都有有限梯度。
- AC `true_state` 模式：训练步和按官方 `train.py` 逻辑手写的参照逐位相同（小规格）；`load_official` 在两种模式下载入和替换的键正确（用测试专用的小假权重）；状态按 `compute_new_pose` 积分；非整数时长的插值在半整数上等于 v7 的平均。
- 三个求谱后端：CPU 上 `cpu_threads` 和 `cpu_batched` 在随机 Laplacian 上满足 09 §6.3 的一致标准；默认后端的输出和 v7 逐位相同；`cuda_float64` 的测试没有 GPU 时跳过，合并前在 GPU 机器上跑。

#### S09 训练循环（09）

09 §8 的全部测试；再加：CPU 上 8 个可训练条件（vjepa2_ac 用小假权重）各跑 20 步、两次验证，出完整的目录和报告。

#### S11 评测（10 §8）

- 用可手算的玩具模型核对 10 §8.2–§8.5 的每个量；identity 的未来误差等于手算的 d(z, z*_T)，所有差异为 0。
- 原始帧率的控制值等于按 01 §5.5 手算的值；滚动的每一步用的是这一步自己的起止帧；AC 的小步动作不除以 δ。
- 原始分辨率读的是 r192、r320 的同一行，网格坐标正确。
- `delta_act` 用替身的动作、本窗口的起点和目标。
- 放宽读法：改检查点字节、步数、配置快照、数据字段任一项都报错，源码 SHA 不同只记录。
- `commanded` 配置正确跳过并计数；中断后重跑跳过已写好的批，每个窗口恰好出现一次。

#### S12 离线规划（11）

11 §6 的全部测试。

#### S13 汇总（13 §8）

在假结果目录上出 D1–D11 全部表和图；13 §8.3 的每一项各造一个不一致的例子，都会报错；自助法按场景、按回合两种整群都对；成败分开的两份加起来和合并的那份一致（按回合数加权）；重复运行输出逐字节相同。

#### S14 任务图和本地运行器（12 §2–§5.1、§5.3）

- 试跑 40 个、正式 129 个任务，依赖和完成标志等于 12 §2 的表；没有环；同样参数两次输出逐字节相同；`--format list` 的分段是拓扑层。
- 本地运行器（假命令）：跳过已完成的；退出 75 时重排；失败的任务阻止它的下游、不影响别的；资源记账不超额；Ctrl-C 后重跑只补没做完的；`--dry-run` 列出全部任务和顺序。

#### S15 集群导出（12 §5.2）

- 生成的 sbatch 脚本包含 12 §5.2 列的每一项；`submit.sh` 按拓扑序提交、依赖写对、跳过已完成的任务。
- 合并前在真实集群上用假任务核对 12 §5.2 的三件事。

#### S16 端到端冒烟

用 5 条金标准回合造一个冒烟数据根（3 条训练、1 条验证、1 条同时当两个测试集；`splits.json` 的 schema 标成 `droid_splits_smoke_v1`，正式配置拒绝读它）。在 GPU 机器上从下载跑到汇总：真实编码器、8 个条件各训练 20 步（含一次续跑）、全部评测轴和规划、汇总表。全部任务完成，没有失败。

### DROID 的试跑关口

见 12 §6。

## 4. 真实试跑的通过标准（合并之后、正式补跑之前）

- jobs-addon-pilot.txt 全部完成，没有失败任务。
- 55 个 D4 任务的等价核对全部通过；记录逐位相同的比例。
- 补跑试跑里 time_only、space_only、fixed_fourier 的初始权重哈希等于 v7 试跑里 po_jepa_full 的值；transformer_consistency 的等于 matched_transformer 的值。
- 每个新条件的每步耗时写进 16 §10，算出正式补跑的时长。
- 试跑的汇总表能完整生成。

试跑只用来核对流程和测速，不看数字下结论，也不根据试跑结果改任何超参数。
