# 结果回传：去重打包（2026-11-17）

补跑、仿真数据集的正式训练和 DROID 的正式训练，回传结果都用 `tools/pack_results.py`，不再用 `zip -r`。补跑说明 `ADDON_RUN_GUIDE.md` 第六节已经改成这里的命令。

## 为什么要换

v7.1 正式结果的 zip 解压后有 61 GB、194 万个 JSON 文件，压缩后还有 5.27 GB。原因是每个逐步训练记录、每条评测记录都把同一份约 14～18 KB 的检查点和编码器来源整段重写一遍，而 zip 一个文件一个文件单独压缩，跨文件的重复压不掉。完整模型的仿真路径沿用同一套写法，仿真数据集有 12 个场景，不换的话包会更大。

`tools/pack_results.py` 把重复内容只存一次，其他地方记引用，再整段压缩，最后写成**一个** SQLite 文件。打包是无损的：每个原文件都记了 SHA256，打完会把所有文件逐个还原、逐字节核对一遍，解包时也一样核对，对不上就报错。只用 Python 标准库（3.9 及以上），不用装别的包。训练、评测代码和实验数值都不受影响。

实测：v7 试跑结果（15 万个文件，4.46 GB，zip 0.40 GB）打包后 **32.9 MB**，逐个还原核对全部通过，4 进程约 5 分钟。v7.1 正式结果按同样比例估约 0.5 GB（原 zip 5.27 GB）。

模型权重 `.pt` 默认不打包；其余（包括 `source-collections/`、`rows/`）都打进去，去重后几乎不占地方。

## 一、拿到工具

任务在跑时不要在 `full-model` 里 `git pull`（会让正在跑和续训的任务报 `delivery source changed`）。只把工具取出来，不动当前代码：

```bash
cd full-model
git fetch origin full-model
git show origin/full-model:po-jepa-full/tools/pack_results.py > ../pack_results.py
cd ..
```

下面的命令都在**工作目录**里执行（`full-model/` 所在的目录），工具就是 `pack_results.py`。

## 二、打包

补跑（替代 `ADDON_RUN_GUIDE.md` 原来的 zip 命令，路径相同；还没跑 D3 时去掉 `logs-v7/jobs-addon-v7tree`）：

```bash
nvidia-smi > nvidia-smi.txt
python pack_results.py pack results-addon logs-addon logs-v7/jobs-addon-v7tree \
    full-model/po-jepa-full/configs/calibration full-model/po-jepa-full/jobs-addon-*.txt nvidia-smi.txt \
    --output results-addon.pack.sqlite
```

仿真数据集的正式训练（任务表默认写到 `results-v8/`，日志在 `logs-addon/<任务文件名>/`）：

```bash
python pack_results.py pack results-v8 logs-addon full-model/po-jepa-full/configs/calibration \
    full-model/po-jepa-full/jobs-*.txt nvidia-smi.txt --output results-sim-formal.pack.sqlite
```

DROID：结果和日志都在 `<运行根>/<配置名>/` 下（特征在数据根，不用传）。要打包的路径必须在当前目录里，所以先进运行根的上一层：

```bash
cd /runs          # 运行根是 /runs/po-jepa-droid 时
python <工作目录>/pack_results.py pack po-jepa-droid/full --output droid-full.pack.sqlite
```

- 命令里列的路径必须都存在，还没生成的先去掉，否则会报 `FileNotFoundError`。要打包的那部分任务请先跑完。
- 默认用 8 个进程（CPU 少就用 CPU 数），可以加 `--workers 4`。打包只读，不改动、不删除任何文件。v7.1 规模（61 GB）估计要 1 小时左右，主进程内存 1～2 GB（估）。
- 最后两行应该是 `verified N files: every rebuilt file matches its sha256 and size` 和 `wrote ....pack.sqlite: ... MB`。把 `.pack.sqlite` 发给 Jerry，不用再压缩。
- 被打断时只会留下 `.pack.sqlite.partial`，重跑同一条命令即可。报错就把屏幕输出发给 Jerry。
- 不想打包的部分用 `--exclude '<模式>'`（可以写多次；`*` 也匹配 `/`，和 `zip -x` 一样）。

## 三、Jerry 这边解包（Windows 也一样）

工具从 GitHub 的 `full-model` 分支取 `po-jepa-full/tools/pack_results.py`，和 `.pack.sqlite` 放在一起：

```bash
python pack_results.py list results-addon.pack.sqlite --summary 4          # 按前 4 层目录汇总大小
python pack_results.py extract results-addon.pack.sqlite --output unpacked --include "*/report.json"
python pack_results.py extract results-addon.pack.sqlite --output unpacked   # 全部解出
python pack_results.py verify results-addon.pack.sqlite                      # 只核对、不写文件
```

解出的文件和算力方机器上的原文件逐字节相同（修改时间也恢复）。目标位置已有同名文件时：内容相同就跳过，不同就报错，不会覆盖。
