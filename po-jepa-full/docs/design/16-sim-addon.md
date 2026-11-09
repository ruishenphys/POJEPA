# 16 仿真补跑

## 1. 做什么、为什么在仿真上做

v7 正式运行（机哥，2026-11-09 起）结束后，在同一台两卡机器、同一批仿真特征上补四件事：

| 代号 | 内容 | 训练新模型 | 任务数（正式） |
|---|---|---|---|
| D | 5 个新条件 × 5 个场景 × 3 个种子的训练和标准评测 | 是 | 75 |
| D4 | 11 个条件的只评测新指标（v7 的 6 个读 v7 检查点）+ 每个场景的截断间隙 | 否 | 165 + 5 |
| D3 | 官方规模 AC 的规划用 256 个样本重跑（用 v7 目录里的代码） | 否 | 5 |
| V | 新条件的变体（分布外）评测和规划 | 否 | 150 + 25 |

在仿真上补，是因为论文自己把 Transformer+consistency 和固定 Fourier 当作关键对照（证伪条件 2 和 5 靠它们），而有真值状态、物理变体和规划成功率的仿真最能说明问题；这三个对照 DROID 本来就要写，仿真只多一个数据入口。v7 不改，主实验结果不受影响。

## 2. 条件

新条件 po_jepa_time_only、po_jepa_space_only、transformer_consistency、fixed_fourier、koopman，定义见 [08-conditions.md](08-conditions.md)。训练协议（查询序列、学习率调度、10,000 步、验证每 500 步、选点规则、种子 1703/1704/1705、提前检查）和 v7 完全相同，都由 `configs/train/v7.yaml` 的 10000 步日程给出。

## 3. 标定

### 3.1 导入 v7 的标定

`tools/import_v7_calibration.py --v7-tree ../../po-jepa --stage full`：

1. 要求 v7 目录的 git HEAD 是 `0ac45fe03bf43b96bf861446b8aa2ada4f4a8987`，否则报错（v7 的代码不应有任何改动）。只接受 0ac45fe：机哥的正式结果是 0ac45fe 跑的；旧提交（653e2d0、911bc09）的模型是模态写法，不能混用。`IMPORTED.json` 记下实际读到的 HEAD。只核 HEAD，不核工作区是否干净：v7 的 calibrate 会在 `configs/calibration/` 下写未跟踪的文件，机器上 v7 的工作区本来就不干净；标定文件本身的值由第 2 步核，字节由 SHA 记下。
2. 对每个场景，读 `<v7 目录>/configs/calibration/<tag>-u<步数>.yaml`（tag 如 `double_well-2000`、`landscape~shape=multi_well-2000`），要求 `sigma_z`、`time_weight`、`space_weight` 都不为空。
3. 原字节复制到 po-jepa-full 的 `configs/calibration/` 下同名文件。目标已存在且字节相同就跳过，不同就报错。
4. 写 `configs/calibration/IMPORTED.json`：每个文件的源路径、SHA、导入时间、v7 的 commit。

第 3、4 步：`IMPORTED.json` 已有这个文件的记录时，`sha256` 相同、`v7_commit` 是 0ac45fe 就算同一个文件，跳过；`sha256` 不同或 `v7_commit` 不是 0ac45fe 就报错。

这样 po-jepa-full 的 `load_protocol()` 不用改，读到的 σ_z、λ 和 v7 正式运行逐位相同。`configs/calibration/` 不进 MANIFEST（v7 原来就排除了它）。

### 3.2 补跑标定（transformer_consistency 和 fixed_fourier 各自的 λ）

`run.py calibrate-addon`，每个场景一条任务，依次标定两个条件，方法和 v7 的 `workflow.calibrate_weights` 相同：

1. 种子固定 1703（和 v7 的 calibrate 相同，不传 `--seed`）。σ_z 用导入的值，不重新标定。
2. 建只用 JEPA 的版本：transformer_consistency 用 matched_transformer 的模型；fixed_fourier 用两个权重为 0 的 fixed_fourier。
3. 在初始状态下量前 32 个训练批的三项梯度范数（只记录）。
4. 用训练查询序列的前 300 批只做 JEPA 训练（学习率调度、裁剪和正式训练相同）。
5. 在第 301–332 批上量三项梯度范数（transformer_consistency 用 `addon.calibration.generic_gradient_norms`，fixed_fourier 用 `training.consistency_gradient_norms`），取中位数，λ_c = ρ·median|∇L_JEPA| / median|∇L_c|，ρ = `calibration.rho` = 1。λ 必须有限且为正；分母为 0 时报错，不除出 inf（`<条件>: median grad_<项> = 0.0; λ 的分母为 0，不能标定（不除出 inf）`）。transformer_consistency 在登记的初始化下一致性梯度恰为 0（残差读出的 `p_out` 零初始化），所以第 3 步会记到 0，这是正常的：初始状态那一组（只记录）分母为 0 时 λ 记 `None`，不报错；第 5 步在 300 步之后量，分母一般非零。
6. 两个条件都完成后，一次性写 `configs/calibration/addon/<tag>-u<步数>.yaml`。这个文件已存在就报错（`补跑标定文件已存在：…确实要重标定请先告诉 Jerry`），不覆盖：文件一旦写出，同一场景的 D 任务就可能已经读过它，重标定会让已经训练的任务和后来的任务用不同的 λ。下面的例子是示意（流格式）；实际写的是首行注释加 `yaml.safe_dump(sort_keys=True)` 的块格式，和 v7 的 `write_v7_calibration` 一致，读出来的值相同：

```yaml
# 仿真补跑标定（run.py calibrate-addon 自动写入，勿手改）：<tag>，<步数> 步
fixed_fourier: {rho: 1.0, space_weight: …, time_weight: …, source: …}
transformer_consistency: {rho: 1.0, space_weight: …, time_weight: …, source: …}
sigma_z_used: …
v7_calibration_sha256: …   # 导入的 v7 标定文件的 SHA
```

逐批记录写在任务输出目录的 `run/calibration-<条件>.json`：v7 的 calibration.json 的 17 个键全有、含义相同，另加 `condition`、`v7_calibration_sha256`、`addon_calibration_file` 三个；`note` 的文字改成补跑的说法。三个种子共用这一组 λ（和 v7 一样）。

读这个文件时（S10 的 `addon.calibration`，训练任务开始前）核对：顶层键恰好是上面这五个；每个条件的键恰好是 `rho`、`space_weight`、`time_weight`、`source`，两个 λ 都是有限正数；`rho` 等于协议的 `calibration.rho`；`sigma_z_used` 等于协议的 σ_z；`v7_calibration_sha256` 等于当前导入的 v7 标定文件的 SHA。任何一项不符就报错。

## 4. 复用 v7 的产物

- 特征：直接读 `../../features-v7/<场景>-<条数>`，不重新编码（试跑的 double_well-200 例外，读 v7 试跑用的 v6 特征目录，§7.4）。每个任务的 preflight 记录缓存身份，汇总时核对一致（13 §4）。
- 检查点和评测记录：D4 用放宽来源核对的读法（10 §4），并逐条核对重算的标准指标和 v7 的记录一致。
- 规划的三个参照（不动、随机、真值规划）只在 v7 的 identity 规划任务里跑过，补跑的规划任务都带 `--no-baselines`，汇总时从 v7 读参照。

## 5. 新子命令和工具

| 命令 | 作用 | 输出 |
|---|---|---|
| `run.py run --condition <新条件>` | 训练 + 标准评测，和 v7 的 run 相同（含续训） | `<输出>/run/job/…`，结构同 v7 |
| `run.py calibrate-addon` | §3.2 | `configs/calibration/addon/…`、任务目录 |
| `run.py check-addon` | 依赖、设备；5 个新条件各自一批上的完整梯度覆盖、读出次数、参数量、初始权重哈希关系（08 §1.3）、100 步耗时 | `<输出>/run/check-addon.json` |
| `run.py d4 --condition C --training-run <任务目录>` | 10 §3、§4 | 10 §7 |
| `run.py spectral-gap` | 10 §5（CPU） | `<输出>/run/job/` 下的 `job.json`、`report.json`、`COMPLETE.json`（和 D 任务、D4 任务同名，13 的汇总按这三个文件找任务） |
| `run.py transfer --condition <新条件>` | 变体评测，和 v7 相同 | 同 v7 |
| `tools/import_v7_calibration.py` | §3.1 | `configs/calibration/` |
| `tools/encoder_identity.py` | 10 §6（CPU） | 一个 JSON |
| `tools/addon_preflight.py` | 核对 v7 的 D 任务都已完成（含续训目录）、特征目录齐全且和 v7 任务记录的缓存身份一致。试跑带 `--v6-features "$V6"`（和生成器的同一个绝对路径）：double_well-200 换成这个目录，走 v6 的缓存核对（`data.approved_cache_metadata` 的 v6 分支），和 v7 试跑任务记的缓存身份比；另要求 0000–0199 每条轨迹的 `COMPLETE.json`、`manifest.json`、`raw_z.npy` 都在，并对 0000 和 0199 用 v6 训练时的逐条读法（`cache._read_complete_cache_uncached`，含 `raw_z.npy` 的 SHA）完整读一遍，所以只有元数据的目录（git 克隆里的 `features/`）也在 A0 停下 | 一个 JSON |
| `tools/plan_v7.py`（po-jepa-full 的复制件） | 接受新条件的规划 | 同 v7 |
| `tools/make_addon_jobs.py --stage pilot/full/variants/v7tree` | 生成任务文件（§7）。`--v6-features` 和 v7 `make_v7_jobs.py` 的同名选项语义相同：试跑必须给、其他阶段不能给，而且必须是绝对路径（否则退出码 2） | 标准输出 |
| `tools/summarize_addon.py` | 13 | 13 §7 |
| `tools/run_jobs.py`（复制件） | 和 v7 的运行器相同，改动见 07 §2.8 | 日志在 工作目录/logs-addon/ |

设备和门槛：check-addon 和 v7 的 check 一样要求 `cuda:N`、写 `environment.json`；calibrate-addon 和 v7 的 calibrate 一样不要求。这两个都先核导入的 v7 标定就是导入的那份（`IMPORTED.json`）；check-addon 另要求补跑标定文件能读（它要建 transformer_consistency、fixed_fourier）。`run.py d4` 要显式的 `cuda:N`（和 v7 的非标定命令相同）；`run.py spectral-gap` 只收 `cpu`。

所有写文件的工具都先写临时文件再换名，中断后重跑同一条命令即可。

## 6. D3：官方规模 AC 规划用 256 个样本

v7 的 V4 段里官方规模 AC 用了 64 个样本（`--samples 64`，`plan_v7.py` 第 171 行按比例给出 24 个精英），其他条件 256 个样本、24 个精英。为了所有预测器用同一个规划器，D3 用 **v7 目录里的 `plan_v7.py` 原样**重跑，只把 `--samples` 改成 256（精英仍是 24）。目标图像由独立的随机数（种子 1704）生成，和 v7 的 V4 完全相同。每个场景约 3 小时，两张卡约 7.5 小时。跑完后论文里去掉“官方规模 AC 用 64 个样本”的注；v7 的 64 样本结果留在汇总表里作参照。

## 7. 任务文件

在 `full-model/po-jepa-full/` 里用 `make_addon_jobs.py` 生成，格式和 v7 相同（“# =====”分段，段内互不依赖，前一段全部完成才开始下一段）。下面以正式运行为例；`landscape` 场景写作 `--system landscape --variant shape=multi_well`，目录名 `landscape~shape=multi_well`。

### 7.1 jobs-addon-full.txt

```
# ===== A0 准备（CPU，3 条）
python tools/import_v7_calibration.py --v7-tree ../../po-jepa --stage full
python tools/encoder_identity.py --encoder ../../checkpoints/vjepa2-vitg-fpc64-256.pth --ac-checkpoint ../../checkpoints/vjepa2-ac-vitg.pt --output ../../results-addon/full/encoder-identity.json
python tools/addon_preflight.py --v7-results ../../results-v7/full --features ../../features-v7 --stage full --output ../../results-addon/full/preflight.json
# ===== C 补跑标定（GPU，每个场景 1 条，共 5 条）
python run.py calibrate-addon --system double_well --trajectories 2000 --updates 10000 --features-root ../../features-v7/double_well-2000 --device cuda:0 --output ../../results-addon/full/calibrate-double_well
…（其余 4 个场景）
# ===== C2 检查（GPU，1 条）
python run.py check-addon --system van_der_pol --trajectories 2000 --updates 10000 --features-root ../../features-v7/van_der_pol-2000 --device cuda:0 --output ../../results-addon/full/check
# ===== D 训练+评测（GPU，75 条）
python run.py run --system double_well --trajectories 2000 --updates 10000 --seed 1703 --condition fixed_fourier --features-root ../../features-v7/double_well-2000 --device cuda:0 --output ../../results-addon/full/double_well/seed1703/fixed_fourier
…（5 个场景 × 3 个种子 × 5 个新条件）
# ===== D4 只评测（GPU 165 条 + CPU 5 条）
python run.py d4 --system double_well --trajectories 2000 --updates 10000 --seed 1703 --condition po_jepa_full --training-run ../../results-v7/full/double_well/seed1703/po_jepa_full --features-root ../../features-v7/double_well-2000 --device cuda:0 --output ../../results-addon/full/d4/double_well/seed1703/po_jepa_full
python run.py d4 … --condition koopman --training-run ../../results-addon/full/double_well/seed1703/koopman …
python run.py d4 … --condition vjepa2_ac --training-run ../../results-v7/full/double_well/seed1703/vjepa2_ac … --ac-checkpoint ../../checkpoints/vjepa2-ac-vitg.pt
…（5 个场景 × 3 个种子 × 11 个条件；identity 的 --training-run 指向 v7 的 identity 任务，作为等价核对的参照）
python run.py spectral-gap --system double_well --trajectories 2000 --updates 10000 --features-root ../../features-v7/double_well-2000 --device cpu --output ../../results-addon/full/spectral-gap/double_well
…（其余 4 个场景）
# ===== E 汇总（CPU，1 条）
python tools/summarize_addon.py --v7-results ../../results-v7/full --addon-results ../../results-addon/full --output ../../results-addon/full/summary
```

运行器重跑时会把 `--training-run` 自动换成实际完成的那次（续训目录 `-resumeN`），规则和 v7 相同。

D 段按场景、种子、`ADDON_CONDITIONS` 的顺序（time_only 在前）生成，上面 D 段的例子行是其中一条（逐字出现在输出里）。运行器在段内按 `priority` 稳定排序后分卡，顺序只影响哪条先跑，不影响结果。

### 7.2 jobs-addon-variants.txt

```
# ===== V3 变体评测（GPU，150 条，种子 1703，只评 16×16 单点）
python run.py transfer --system double_well --trajectories 2000 --updates 10000 --seed 1703 --condition fixed_fourier --test-variant coupling=0.5 --training-run ../../results-addon/full/double_well/seed1703/fixed_fourier --features-root ../../features-v7/double_well~coupling=0.5-2000 --device cuda:0 --output ../../results-addon/full/transfer/double_well~coupling=0.5/fixed_fourier
…（5 个场景 × 6 个变体 × 5 个新条件；变体表和 v7 的 TEST_VARIANTS 相同）
# ===== V4 规划（GPU，25 条，种子 1703，每条 100 个回合）
python tools/plan_v7.py --job ../../results-addon/full/double_well/seed1703/fixed_fourier --upstream-root ../../vjepa2 --checkpoint ../../checkpoints/vjepa2-vitg-fpc64-256.pth --episodes 100 --output ../../results-addon/full/plan/double_well/fixed_fourier.json --device cuda:0 --no-baselines
…
# ===== V5 汇总（CPU，1 条）
python tools/summarize_addon.py --v7-results ../../results-v7/full --addon-results ../../results-addon/full --output ../../results-addon/full/summary
```

变体特征由 v7 的 jobs-variants.txt 编码，要求 v7 的变体阶段先跑完。

### 7.3 jobs-addon-v7tree.txt（在 v7 目录里跑）

```
# ===== D3 官方规模 AC 规划，256 个样本（GPU，5 条，种子 1703）
python tools/plan_v7.py --job ../results-v7/full/double_well/seed1703/vjepa2_ac --upstream-root ../vjepa2 --checkpoint ../checkpoints/vjepa2-vitg-fpc64-256.pth --episodes 100 --output ../results-addon/full/plan-ac256/double_well/vjepa2_ac.json --device cuda:0 --ac-checkpoint ../checkpoints/vjepa2-ac-vitg.pt --samples 256 --no-baselines
…（其余 4 个场景）
```

路径相对于 `工作目录/po-jepa`。用 v7 自己的 `run_jobs.py` 跑，日志在 工作目录/logs-v7/jobs-addon-v7tree/。

### 7.4 jobs-addon-pilot.txt（试跑）

和 7.1 结构相同，改用 v7 试跑的产物：200 条轨迹、2000 步、只有种子 1703、5 个场景全跑。double_well-200 读 v7 试跑用的 v6 特征目录（`--v6-features "$V6"`，绝对路径），其他场景 `../../features-v7/<场景>-200`；双势阱的 18 条命令（C 1、D 5、D4 11、谱间隙 1）的 `--features-root` 都是它，A0 的 preflight 行带同一个参数。结果路径 `../../results-v7/pilot`、`../../results-addon/pilot`，导入 `<tag>-u2000.yaml`（double_well-200 也照常从 v7 试跑导入 `double_well-200-u2000.yaml`）。任务数：A0 3、C 5、C2 1、D 25、D4 55 + 5、E 1。目的是在真实 GPU 上把全流程走一遍、在真实 v7 检查点上做等价核对，并测出每个新条件每步的耗时，用来算准正式运行的时长。

### 7.5 任务数合计

| 文件 | 任务数 |
|---|---|
| 试跑 | 95 |
| 正式 | 255 |
| 变体 | 176 |
| v7 目录 | 5 |

## 8. 目录

```
工作目录/
  po-jepa/            # v7（不动）；D3 在这里跑
  full-model/         # git clone -b full-model
    po-jepa-full/     # 补跑的所有命令在这里执行
  vjepa2/  checkpoints/
  features-v7/  results-v7/  logs-v7/      # v7 的产物（只读）
  results-addon/{pilot,full}/              # 补跑的产物
  logs-addon/<任务文件名>/                   # 补跑的日志
```

## 9. 给机哥的补充说明（草稿，代码合并后定稿为 po-jepa-full/补跑说明.md）

环境和 v7 相同，不用装新包。顺序：

1. 等 v7 的 `jobs-full.txt` 和 `jobs-variants.txt` 都跑完。D3 的任务文件照常在 `full-model/po-jepa-full` 里用新代码的 `make_addon_jobs.py --stage v7tree` 生成（第 5 步）；补跑代码晚了而要先跑 D3 的话，Jerry 直接把 §7.3 的 5 行发给机哥（`--stage v7tree` 的输出就是这 5 行加两行说明）。
2. 在工作目录里：`git clone -b full-model https://github.com/ruishenphys/POJEPA.git full-model`，然后 `cd full-model/po-jepa-full`。
3. 试跑（约半天）：
   ```bash
   V6=<v6 特征目录的绝对路径>        # 和 v7 交接说明里的 V6= 那一行完全相同
   python tools/make_addon_jobs.py --stage pilot --v6-features "$V6" > jobs-addon-pilot.txt
   python tools/run_jobs.py jobs-addon-pilot.txt --gpus 0,1
   ```
   `V6` 和 v7 交接说明的相同，就是 v7 试跑给 `make_v7_jobs.py --stage pilot --v6-features "$V6"` 的那个绝对路径（在原来的 v6 工作目录里；记不清时从 v7 的 `po-jepa/jobs-pilot.txt` 里 double_well 行的 `--features-root` 读出来），这个目录要保留到补跑结束。不要用克隆里的 `po-jepa/features/`、`full-model/features/`：它们只有提交进仓库的元数据，没有 `raw_z.npy`，A0 的 preflight 会停下。
   打印“有任务失败，已停下”就把 `../../logs-addon/` 和 `../../results-addon/pilot/` 打包发给 Jerry，先不往下跑。
4. 正式和变体：
   ```bash
   python tools/make_addon_jobs.py --stage full > jobs-addon-full.txt
   python tools/run_jobs.py jobs-addon-full.txt --gpus 0,1
   python tools/make_addon_jobs.py --stage variants > jobs-addon-variants.txt
   python tools/run_jobs.py jobs-addon-variants.txt --gpus 0,1
   ```
5. D3：
   ```bash
   python tools/make_addon_jobs.py --stage v7tree > jobs-addon-v7tree.txt
   cd ../../po-jepa
   python tools/run_jobs.py ../full-model/po-jepa-full/jobs-addon-v7tree.txt --gpus 0,1
   ```
6. 打包发回（在工作目录里，不带权重）：
   ```bash
   zip -r results-addon.zip results-addon logs-addon logs-v7/jobs-addon-v7tree full-model/po-jepa-full/configs/calibration full-model/po-jepa-full/jobs-addon-*.txt -x '*.pt'
   ```

断了重跑同一条命令即可，规则和 v7 一样。

## 10. 算力估计

- **D**：time_only、space_only 每步成本和 v7 的 po_jepa_full 相同；fixed_fourier 省掉建图和求谱，更快；transformer_consistency 每步 6 次 Transformer 前向，约为 matched_transformer 的 6 倍；koopman 每步 1 次前向，可以忽略。记 v7 正式运行里一个 po_jepa_full 任务的实测时长为 t，D 段约 60t 卡时，两张卡除以 2。t 约 1.5 小时的话，D 段约 2 天。
- **D4**：每个任务的成本和该条件 v7 评测相当（10 §3.10），合计约为 v7 D 段评测部分的 1.8 倍（165 / 90）。
- **D3**：约 15 卡时，两张卡约 7.5 小时。
- **C、C2、A0、E**：合计约 1–2 小时。
- **变体**：V3 只评 16×16 单点，很快；V4 和 v7 的 V4 中 PO-JEPA 类任务相当。
- 试跑结束后用实测的每步耗时更新这一节。

## 11. 时间表

| 日期 | 事 |
|---|---|
| 11-08 | 补跑规格推上 full-model；Jerry 确认 |
| 11-09 起 | 机哥跑 v7 正式运行（到约 11-13，含变体阶段） |
| 11-09 至 11-12 | 第零期开发：S00、S08a、S10、S17、S18，审核通过后合并（15） |
| 约 11-13 | v7 跑完后，补跑试跑（约半天）；如果代码晚了，先跑 D3 |
| 11-13 至 11-16 | 补跑正式和变体（约 2–3 天） |
| 约 11-16 至 17 | 结果回来，汇总，把数字填进论文 |

## 12. 风险和应对

| 情况 | 怎么办 |
|---|---|
| v7 有 D 任务没完成 | A0 的 preflight 列出来，补跑停在 A0；先让 v7 补完 |
| 试跑的 `--v6-features` 给错（不是 v7 试跑用的那个目录、只有元数据的克隆 `features/`） | A0 的 preflight 对试跑的 double_well-200 走 v6 的缓存核对、逐条核 200 条轨迹的 `COMPLETE.json`、`manifest.json`、`raw_z.npy`，并完整读首尾两条的 `raw_z.npy`；不对就列出来，补跑停在 A0。没给或给了相对路径时生成器直接报错（退出码 2） |
| 等价核对不过 | D4 任务失败，段停下。先查新树读 v7 检查点的路径，不放宽容差 |
| 单项消融被提前检查停掉（一致性梯度占比低于 0.005） | 和 v7 的规则一样算失败，列在日志里；这本身是结果，Jerry 决定是否报告或调 ρ（调 ρ 就是另一个配置，要重新登记） |
| fixed_fourier 或 transformer_consistency 标定出的 λ 不是有限正数 | 标定任务失败，段停下，查原因，不手填 λ |
| 补跑代码到 11-13 还没合并 | 先跑 D3；v7 的机器空着时也可以先跑 v7 的变体阶段，补跑顺延 |
| 磁盘 | 补跑的检查点很小（PO-JEPA 类每个约几 MB），不需要额外空间；不新编码特征 |

## 13. 结果进论文的哪里

| 结果 | 论文位置 |
|---|---|
| D 的标准指标 | 主实验表和消融表里三个新对照和两个单项消融的行 |
| D4 的 LN 空间 TCD/SCD | 跨模型的一致性比较，包括 AC |
| 分割差异、跨网格分割差异、不等分 | 定理那一节后面的实证段和附录图 |
| 误差分解 | “From consistency to held-out durations”段的数字 |
| 长时程 | OOD 表的 Rollout 行（16–32 个周期） |
| 截断间隙 | 第 235 行附近那句话，按 10 §5 的规则保留或改写 |
| 编码器核对 | AC 基线的说明 |
| D3 | 规划结果表；删掉 64 样本的注 |
| 精度匹配 | 证伪条件 1 和配对比较表的标记 |

Overleaf 先按本规格改好文字，数字的位置写 TODO 注释。
