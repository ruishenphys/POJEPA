> **在 GPU 上运行 v5：只看 [`运行清单_给算力方.md`](运行清单_给算力方.md)。** 改动说明见 [`README_v4.md`](README_v4.md)。下文是一轮（v3）的原始说明，仅供参考。

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
