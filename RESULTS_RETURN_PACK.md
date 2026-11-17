# 结果回传：改用去重打包（2026-11-17）

正式运行以后回传结果都用这里的命令，**不再用 `RESULTS_RETURN.md` 第 2、3 节和 `RESTART_V7_1.md`「跑完回传」里的 `zip -r`**。那两个文件在 `MANIFEST.json` 里，为了不改动交付清单的哈希，原文没有动。

本文件和 `tools/pack_results.py` 都不在 `MANIFEST.json` 里：`git pull` 拿到它们，不影响正在跑或要续训的任务。

## 为什么要换

v7.1 正式结果的 zip 解压后有 61 GB、194 万个 JSON 文件，压缩后还有 5.27 GB。原因是每个逐步训练记录、每条评测记录都把同一份约 14～18 KB 的检查点和编码器来源整段重写一遍，而 zip 一个文件一个文件单独压缩，跨文件的重复压不掉。

`tools/pack_results.py` 把这些重复内容只存一次，其他地方记引用，再整段压缩，最后写成**一个** SQLite 文件。打包是无损的：每个原文件都记了 SHA256，打完会把所有文件逐个还原、逐字节核对一遍，解包时也一样核对，对不上就报错。只用 Python 标准库（3.9 及以上），不用装别的包。训练、评测代码和实验数值都不受影响。

## 实测和估算

| 数据 | 原始 | zip | 打包后 | 耗时（4 进程） |
|---|---|---|---|---|
| v7 试跑结果 `results-v7-pilot-lite.zip` 的全部内容（15 万个文件） | 4.46 GB | 0.40 GB | **32.9 MB**（逐个还原核对全部通过） | 约 5 分钟 |
| v7.1 正式结果（194 万个文件，估） | 61 GB | 5.27 GB | **约 0.5 GB**（按试跑比例估） | 约 1 小时（估，8 进程会更快） |

`source-collections/`（每步一份的数据来源记录）和 `rows/`（与 `trajectories/` 内容相同的逐条评测明细）打包后几乎不占地方（在试跑里两者合计只多 3 MB），所以不再排除，包里是完整的结果目录。只有模型权重 `.pt` 默认不打包。

## 一、拉代码（在 `po-jepa` 目录里）

```bash
git pull
ls tools/pack_results.py
```

如果正在跑的任务不允许 `git pull`，也可以只把工具取出来，不动当前代码：
`git fetch origin v7 && git show origin/v7:tools/pack_results.py > ../pack_results.py`，下面的命令里把 `po-jepa/tools/pack_results.py` 换成 `pack_results.py`。

## 二、打包（在工作目录里执行，也就是 `results-v7/` 所在的目录）

分布外和规划的结果（`transfer/`、`plan/`）以及跑完后的汇总：

```bash
nvidia-smi > nvidia-smi.txt
python po-jepa/tools/pack_results.py pack results-v7/full/v7.1/transfer results-v7/full/v7.1/plan \
    results-v7/full/v7.1/summary logs-v7 po-jepa/configs/calibration po-jepa/jobs-*.txt nvidia-smi.txt \
    --output results-v7-final.pack.sqlite
```

以后整个正式运行目录都要回传时：

```bash
python po-jepa/tools/pack_results.py pack results-v7/full/v7.1 logs-v7 po-jepa/configs/calibration \
    po-jepa/jobs-*.txt nvidia-smi.txt --output results-v7-full.pack.sqlite
```

- 默认用 8 个进程（CPU 少就用 CPU 数），可以加 `--workers 4` 少占一点 CPU。打包只读结果目录，不改动、不删除任何文件；但要打包的那部分任务请先跑完，跑到一半的目录打进去也是半截的。
- 命令里列的路径必须都存在，还没生成的（比如汇总没跑时的 `summary`）先从命令里去掉，否则会报 `FileNotFoundError`。
- 最后两行应该是 `verified N files: every rebuilt file matches its sha256 and size` 和 `wrote results-v7-....pack.sqlite: ... MB`，把这个 `.pack.sqlite` 文件发给 Jerry 就行。不用再压缩，里面已经压过了。
- 中途出错或被打断时，不会留下 `.pack.sqlite`，只会留下 `.pack.sqlite.partial`，直接重跑同一条命令即可。如果报错，把屏幕输出发给 Jerry。
- 主进程需要一些内存记录重复内容的位置，正式规模估计 1～2 GB。

## 三、Jerry 这边解包（Windows 也一样）

工具从 GitHub 的 `v7` 分支取 `tools/pack_results.py`，和 `.pack.sqlite` 放在一起：

```bash
# 看包里有什么（按前 5 层目录汇总大小）
python pack_results.py list results-v7-final.pack.sqlite --summary 5
# 只解出要用的部分，比如每个任务的 report 和训练曲线（几秒钟）
python pack_results.py extract results-v7-final.pack.sqlite --output unpacked --include "*/report.json" --include "*/curves.json"
# 全部解出（大小和原目录一样，正式规模要 60 GB 左右的空闲空间）
python pack_results.py extract results-v7-final.pack.sqlite --output unpacked
# 只核对、不写文件
python pack_results.py verify results-v7-final.pack.sqlite
```

- `--include` / `--exclude` 可以写多次，按包里存的路径匹配（`*` 也匹配 `/`，和 `zip -x` 一样）。
- 解出的文件和机哥机器上的原文件逐字节相同（修改时间也恢复）。目标位置已有同名文件时：内容相同就跳过，不同就报错，不会覆盖。

## 四、本机要保留的

和以前一样：`features-v7/` 和 `results-v7/` 里的 `.pt` 都先别删；打包不会删除或移动任何原文件。
