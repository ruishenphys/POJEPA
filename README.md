# PO-JEPA v7

> **2026-11-10 v7.1：正式运行暂停后要换代码重跑训练，先看 [RESTART_V7_1.md](RESTART_V7_1.md)。**

**帮忙跑 v7 实验的同学：完整步骤在 [HANDOFF.md](HANDOFF.md)，照着复制命令就行，不需要改任何代码或配置。** 数据集说明见 [DATASET.md](DATASET.md)。

> **2026-11-08 更新**：和最早那版相比有两处变化。① 按导师意见改了 PO-JEPA 的动力学写法（动作驱动和非线性交互改在 token 空间计算），这是正式运行要用的模型，代码因此更新了；PO-JEPA 的两个条件每步比原来慢约 1.6 倍，下面的时间已经按这个调过。② 试跑里的 double_well 改用你机器上 v6 时的特征目录，试跑的命令换了，见下面，HANDOFF.md 第二节是同一段。命令和其他步骤都没变。已经 clone 过的，在 `po-jepa` 目录里执行 `git pull`，再看 `git log -1 --oneline`，里面应该有 `token-space` 字样。

## 流程一览

| 步骤 | 做什么 | 两张 RTX PRO 6000 大约要 |
|---|---|---|
| 一、准备 | 装环境（版本必须一致）、clone 本仓库的 v7 分支和 V-JEPA 2、下载两个权重（HANDOFF.md 第一节） | 只做一次 |
| 二、试跑 | 5 个场景 × 200 条轨迹，2000 步，1 个种子；double_well 用 v6 的特征 | 9–11 小时 |
| 三、正式运行 | 5 个场景 × 2000 条轨迹，10000 步，3 个种子，再做分布外评测和规划 | 4–4.5 天 |
| 四、传回结果 | 打包 `results-v7`、`logs-v7` 发给 Jerry（HANDOFF.md 第五节） | 几分钟 |

## 试跑（新版命令）

试跑的 double_well 是 v6 用过的那 200 条数据。v7 对这份数据只认 v6 编码出的那份特征，在本机重新编码的会在标定时被拒绝（报 `approved F02 encode report SHA mismatch`），所以试跑要指向 v6 时的特征目录：就是 v6 运行时 `--features-root` 指的那个 `features` 目录，里面有 `0000` 到 `0199` 共 200 个子目录。新 clone 的 `po-jepa/features/` 里只有几个登记文件、没有特征数组，不能用。

把下面第二行换成那个目录的绝对路径，核对三项，再生成任务：

```bash
cd po-jepa
V6=/v6/的/features/目录的/绝对路径
sha256sum "$V6/report.json"                    # 必须是 e464495fee79f2f0367d99fca456c8ecf7066f329e37935ce2c7ae87083b269e
ls -d "$V6"/[0-9][0-9][0-9][0-9] | wc -l       # 必须是 200
ls "$V6"/0000/raw_z.npy "$V6"/0199/raw_z.npy   # 两个文件都在
python tools/make_v7_jobs.py --stage pilot --v6-features "$V6" > jobs-pilot.txt
python tools/run_jobs.py jobs-pilot.txt --gpus 0,1
```

三项里有一项不对就先别跑，告诉 Jerry。生成的 `jobs-pilot.txt` 一共 57 条任务：double_well 不再编码，它的标定和 6 条训练任务都读 `$V6`，其余 4 个场景和以前一样在本机编码。

- 最后打印 **“全部完成。”**：试跑通过，直接开始正式运行，同时把试跑结果打包发给 Jerry，不用等他回复。
- 打印 **“有任务失败，已停下”**：把 `../logs-v7/` 和 `../results-v7/` 打包发给 Jerry，先不要开始正式运行。

## 正式运行（命令没变）

```bash
python tools/make_v7_jobs.py --stage full > jobs-full.txt
python tools/make_v7_jobs.py --stage variants > jobs-variants.txt
python tools/run_jobs.py jobs-full.txt --gpus 0,1 && python tools/run_jobs.py jobs-variants.txt --gpus 0,1
```

正式运行每个场景是 2000 条轨迹，全部在本机编码，不用 v6 的特征。

中断了（关机、断电、Ctrl-C、显卡出错）就重新运行同一条 `run_jobs.py` 命令，不用删任何东西。报错先查HANDOFF.md 第六节的表，别的报错把 `../logs-v7/` 发给 Jerry。

---

## 附：v3 时期的旧说明（仅供参考，跑 v7 不用看）

# PO-JEPA v3 运行说明

## 交付内容

在交付文件所在目录先校验并解压，再进入代码目录：

```bash
python -c "import hashlib,pathlib; p=pathlib.Path('po-jepa-v3-handoff.zip'); expected=pathlib.Path('SHA256SUMS.txt').read_text().split()[0]; actual=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(); print(actual); assert actual==expected,'SHA256 mismatch'"
python -m zipfile -e "po-jepa-v3-handoff.zip" "./unpacked-v3"
cd "./unpacked-v3/po-jepa"
```

`unpacked-v3` 必须是新目录。代码、固定配置、参考公式和已有冻结特征缓存均在包内：

```text
po-jepa/
├── README.md
├── run.py
├── app/  evals/  src/       # 训练、评价及计算模块
├── configs/  reference/    # 固定参数、公式与来源记录
├── tests/                 # CPU 单元测试
├── features/              # 已完成的 200 条冻结特征缓存
│   ├── report.json  COMPLETE.json  registration.json
│   └── 0000/ … 0199/      # 各含 raw_z.npy、manifest.json、COMPLETE.json
└── MANIFEST.json           # 代码与配置摘要
```

缓存约 **9.52 GB**，共603个文件。每条特征为 `float32[33,256,1408]`，只缓存冻结编码器输出 `Z`。直接使用 `features/`，不重新生成数据、不重新编码、不下载编码器权重。原始缓存元数据保持原样，运行时按相对路径读取。

压缩包校验值见同目录 `SHA256SUMS.txt`。解压需使用支持 ZIP64 的工具；压缩包与解压内容同时保留需约19.1 GB，另预留运行结果空间。

## 环境

使用已准备并激活的 Python 环境：

| 项目 | 要求 |
|---|---|
| GPU | 完整 RTX 4090 24GB，或完整 A100 40/80GB；MIG 关闭 |
| 驱动 | 能运行 `nvidia-smi`，支持 PyTorch CUDA 12.4 运行时 |
| Python | 3.12.x |
| PyTorch | 2.6.0，编译 CUDA 版本12.4 |
| NumPy / PyYAML | 2.5.2 / 6.0.2 |
| setuptools | 78.1.0 |

以下命令均从解压后的 `po-jepa` 目录执行，Linux和Windows通用。安装只使用本地代码，不下载依赖：

```bash
python --version
nvidia-smi
python -m pip install --no-index --no-deps --no-build-isolation -e .
python run.py --help
```

若环境缺项，先按表格补齐环境。保留解压目录；editable安装依赖该目录。不要修改源码、固定配置或缓存元数据。

## 检查和完整运行

先运行资源检查。所有输出目录必须是尚不存在的新目录：

```bash
python run.py check --features-root "./features" --device cuda:0 --output "./results/check-01"
```

检查包括设备与版本、缓存来源、完整模型五次Phi/三损失反传诊断、独立模型100次正常更新和400窗口验证成本。资源试跑状态不用于正式训练。确认 `results/check-01/console/exit-code.txt` 为 `0`，且 `results/check-01/run/COMPLETE.json` 存在；查看 `run/check/report.json` 中耗时与峰值显存，确认可安排完整运行。

随后执行正式训练与留出评价：

```bash
python run.py run --features-root "./features" --device cuda:0 --output "./results/experiment-01"
```

正式训练包含完整PO-JEPA、参数量匹配Transformer、去除两项一致性损失的PO-JEPA，三个条件各2000次更新、每批4窗口。只按验证集未来误差选checkpoint；固定选点后，评价三个条件和未训练identity，每条件1600个单网格点及1200个唯一网格对。

真实运行命令 `run/train/evaluate/resume` 均会重新执行完整资源预检，需计入额外时间。出现错误返回非零退出码并留存证据，不自动重试或调整预算、参数、容差。

## 分步运行与续训

需要分开训练和评价时，使用下列命令代替上面的 `run`：

```bash
python run.py train --features-root "./features" --device cuda:0 --output "./results/train-01"
python run.py evaluate --features-root "./features" --device cuda:0 --training-run "./results/train-01/run/training" --output "./results/eval-01"
```

评价只读验证集选中的checkpoint，不更新该模型。评价和续训复用均核实际查询序列、生成器状态、逐步更新记录及完成/下一批游标；记录缺失或不一致即停止，不自动补写。

训练中断后保留旧目录，明确指定旧 `run/training` 和一个新输出目录：

```bash
python run.py resume --features-root "./features" --device cuda:0 --resume-from "./results/experiment-01/run/training" --output "./results/resume-01"
python run.py evaluate --features-root "./features" --device cuda:0 --training-run "./results/resume-01/run/training" --output "./results/eval-resume-01"
```

`resume`只续完训练。已完成条件核验后复用；中断条件恢复最后完整验证点的模型、AdamW、随机状态与游标，补齐剩余预算。每200更新保存验证点；保存点后的未保存更新可能重算，恢复点记在 `RESUMED.json`。

续训要求同一代码、配置、缓存、物理GPU、Python/库版本、线程与计算设置；不自动兼容旧版包或跨设备恢复。新输出必须位于旧训练目录之外。缺完整保存点或摘要损坏时，停止并回传证据，不覆盖旧结果或修改账本。

## 回传结果

回传执行过的每个 `results/<运行名>/` **完整目录**。失败目录也必须保留，包括续训使用的旧失败目录。目录中应包含：

- `console/`：完整stdout/stderr、实际命令、真实退出码；
- `run/environment.json` 和 `run/check/`：版本、设备、缓存来源、梯度诊断、耗时与显存；
- `run/training/`：预登记、实际查询与更新记录、验证结果和全部checkpoint；
- `run/evaluation/`：逐窗口/轨迹记录、汇总与配对比较；
- 各阶段生成的 `COMPLETE.json` 或 `FAILED.json`。未执行阶段没有产物。

例如完整运行需回传 `results/check-01/` 与 `results/experiment-01/`；如续训，再一并回传 `results/resume-01/` 和对应评价目录。不必重复回传 `features/` 或官方编码器权重。

## 验证状态

代码已完成CPU检查和独立审查。**尚未完成4090/A100真实资源检查、正式训练及真实留出评价**；实际显存、总耗时、长期数值稳定性与科研收益由目标设备输出确认。资源检查通过不等于科研结论成立。
