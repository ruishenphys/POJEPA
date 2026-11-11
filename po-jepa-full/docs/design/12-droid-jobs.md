# 12 DROID 任务图与后端

2026-11-08 · 第一版

所有阶段都写成互相独立的任务，任务之间只通过磁盘上的产物和完成标志联系。任务图是一份 JSON，由 `tools/droid_jobs.py` 从配置生成，不含任何集群相关的东西；三个后端（本机、SLURM、容器列表）各自把它翻译成能跑的形式。代码在 `tools/droid.py`（所有阶段的统一入口）、`tools/droid_jobs.py`、`tools/droid_run.py`、`tools/droid_slurm.py`（S14、S15）。

## 1. 根目录

| 根 | 内容 | 谁写 |
|---|---|---|
| 数据根 `--data-root` | `index/`、`raw/`、`prepared/`、`features/`，以及 `upstream/vjepa2`（上游代码，提交 204698b）和 `checkpoints/`（`vjepa2-vitg-fpc64-256.pth`、`vjepa2-ac-vitg.pt`） | 索引到编码的任务；按数据版本命名，所有配置共用 |
| 运行根 `--run-root` | `<name>/calibration/`、`check/`、`train/`、`eval/`、`summary/`、`logs/` | 标定以后的任务；一个配置一个子目录 |

上游代码和两份权重不由任务下载，由人放好；`check/env` 和编码任务开始时核对上游的 git 提交（必须是 204698b）和编码器权重的 SHA（必须等于 v7 `cache.CHECKPOINT_SHA`），不对就退出 1；AC 权重的 SHA 记进 `check/env.json` 和用到它的每个任务的登记，同一配置里所有任务必须相同。

估计的磁盘用量（正式运行）：原始文件同时最多约 100 GB（训练片编码后删，03 §4），测试片的原始文件 20 多 GB，准备文件约 10 GB，特征约 520 GB（含测试片和测试附加），运行根约 100 GB（主要是官方规模 AC 的检查点）。

## 2. 任务

| 阶段 | 任务编号 | 条数（试跑 / 正式） | 依赖 | 完成标志 |
|---|---|---|---|---|
| 索引 | `index` | 1 / 1 | — | `index/MANIFEST.json` |
| 下载 | `download/<片>` | 4 / 23 | `index` | `raw/<片>/_complete/<name>.json` |
| 准备 | `prepare/<片>` | 4 / 23 | `download/<片>` | `prepared/<片>/_complete/<name>.json` |
| 统计 | `prepare/stats` | 1 / 1 | 全部 `prepare/<片>` | `prepared/_complete/stats-<name>.json` |
| 编码 | `encode/<片>` | 4 / 23 | `prepare/<片>` | 训练片 `features/<片>/_raw_deleted/<name>.json`（原始文件删完才写，S14-fix1）；`val`、`id_test`、`ood_test` 是 `features/<片>/_complete/<name>.json` |
| 测试附加 | `encode/test-extra` | 1 / 1 | `prepare/id_test`、`prepare/ood_test` | `features/test_extra/_complete/<name>.json` |
| σ_z | `calibrate/sigma-z` | 1 / 1 | `prepare/stats`、全部 `encode/<片>` | `<name>/calibration/sigma_z.json` |
| 环境检查 | `check/env` | 1 / 1 | `calibrate/sigma-z` | `<name>/check/env.json` |
| λ | `calibrate/lambda-<条件>` | 3 / 3 | `calibrate/sigma-z` | `<name>/calibration/lambda-<条件>.json` |
| 训练 | `train/<条件>/seed<种子>` | 8 / 24 | `check/env`；三个带一致性的条件另依赖自己的 λ | `<name>/train/<条件>/seed<种子>/COMPLETE.json` |
| 评测 | `eval/<条件>/seed<种子>` | 8 / 24 | 对应的训练任务、`encode/test-extra` | `<name>/eval/<条件>/seed<种子>/COMPLETE.json` |
| 不训练的对照 | `eval/identity`、`eval/vjepa2_ac_zeroshot` | 2 / 2 | `check/env`、`encode/test-extra` | 同上 |
| 截断间隙 | `eval/spectral-gap` | 1 / 1 | `calibrate/sigma-z` | `<name>/eval/spectral-gap/COMPLETE.json` |
| 汇总 | `summary` | 1 / 1 | 全部评测任务 | `<name>/summary/COMPLETE.json` |

正式运行共 129 个任务，试跑 40 个。片是 `train-000` 到 `train-019`（试跑只有 `train-000`）和 `val`、`id_test`、`ood_test`（02 §6）。编码训练片时带 `--delete-raw`（03 §4），测试片和验证片不带。

下载、准备、编码按片串起来：一片下完就能准备，准备完就能编码，不用等全部下完。训练任务之间互不依赖，可以同时跑。

S14 补注：训练片号在 `use_failures: true` 时按 ⌈N/shard_size⌉ 推出，不读索引；数据根里已有 `index/splits.json` 时用 `splits.shard_plan` 并核对两者相同；`use_failures: false` 的片号取决于失败回合的位置，必须先有索引。汇总依赖全部 `eval/*`（含 `eval/spectral-gap`）。统一入口 `tools/droid.py` 目前接上 index、download、prepare、encode 四个阶段（转给 S02–S06b 各模块的 `main()`），其余阶段在 S09、S11–S13 写好之前退出 2、不写文件。S11b 接上了 `eval/spectral-gap`（`droid.spectral_gap.main`），其余 eval 任务仍退出 2。S09a 接上 train；S09b 接上 calibrate（`droid.calibrate.main`）、check（`droid.check.main`）、其余 eval 任务（`droid.eval_job.main`）和 summary，统一入口不再有退出 2 的阶段。

## 3. 每个任务的共同约定

- 命令：`python tools/droid.py <阶段> --config configs/droid/<name>.yaml --data-root <数据根> --run-root <运行根> --job <任务编号>`，训练和评测另加 `--time-limit-minutes <上限>`，训练任务在工作目录 `po-jepa-full/` 下运行。
- 开始时完成标志已存在：什么都不做，退出 0。审核修复（a5f4edb，S11-P1-1）：只对 index、download、prepare、encode、calibrate、check 如此（产物只写一次，下游读时核身份）；train、eval、summary 照常调用阶段，由阶段先核本次请求的身份再复用（训练核 COMPLETE.json 绑定 report.json，评测核 source.json、job.json，汇总按当前产物重算、内容不变就不换输出）。
- 完成标志最后写，先写临时文件再改名。
- 中断后重跑同一条命令就接着做：下载跳过已有 `DOWNLOADED.json` 的回合，准备和编码跳过输出已核对通过的回合，训练从检查点接着跑（09 §4.5），评测跳过已写好的批（10 §8.9）。
- 退出码：0 完成；75 已存档、要重排同一条命令；其他非 0 是失败，不自动重试（09 §4.7）。
- 日志：标准输出和标准错误由后端存到 `<运行根>/<name>/logs/<任务编号的 / 换成 __>.log`，重排时追加。

## 4. 任务图 JSON

`python tools/droid_jobs.py --config configs/droid/full.yaml --data-root … --run-root … --output jobs-full.json`：

```json
{
  "schema": "droid_jobs_v1",
  "config": "full",
  "config_sha256": "…",
  "data_root": "/data/droid-1.0.1",
  "run_root": "/runs/po-jepa-droid",
  "jobs": [
    {
      "id": "encode/train-007",
      "stage": "encode",
      "command": ["python", "tools/droid.py", "encode", "--config", "configs/droid/full.yaml",
                  "--data-root", "/data/droid-1.0.1", "--run-root", "/runs/po-jepa-droid",
                  "--job", "encode/train-007", "--delete-raw"],
      "resources": {"gpus": 1, "cpus": 8, "mem_gb": 64, "time_limit_minutes": 600},
      "deps": ["prepare/train-007"],
      "complete": "/data/droid-1.0.1/features/train-007/_complete/full.json",
      "requeue_exit_codes": [75],
      "priority": 3
    }
  ]
}
```

- `jobs` 按拓扑序、同层按编号排序；同样的参数运行两次输出逐字节相同（`cache._write_json`）。
- 资源取配置的 `resources.<阶段>`（01 §3），`calibrate/sigma-z` 和 `eval/spectral-gap` 用 `stats` 的资源（CPU）。
- `priority`：越小越先排。训练里 vjepa2_ac 为 1（最慢），其余 PO-JEPA 类和 AC 缩小版为 2，Transformer 类和 Koopman 为 3；其他阶段按拓扑层次。只影响同时有多个任务可跑时的顺序。
- 生成时核对：每个依赖都存在、没有环、完成标志互不相同、每个片都有下载、准备、编码三个任务。

## 5. 后端

### 5.1 本机 `tools/droid_run.py`

`python tools/droid_run.py jobs-full.json --gpus 0,1 --cpus 32 --mem-gb 256`：

- 依赖都完成（看完成标志）的任务进入可跑队列，按 `priority`、再按编号取；GPU 任务独占一张卡（设 `CUDA_VISIBLE_DEVICES`），CPU 和内存按资源数记账，不超过命令行给的总量。
- 退出 75：立即重排同一条命令。其他非 0：记进失败列表，依赖它的任务不再开始，别的照常跑；全部能跑的跑完后列出失败任务的日志路径，退出 1。
- 本机没有时长上限，训练任务传配置里的上限，到点同样会存档退出 75 再被重排，所以本机和集群上的行为一致。
- 断电或 Ctrl-C 后重新运行同一条命令即可：完成的跳过，没完成的从断点接着做。审核修复（a5f4edb，S11-P1-1）：train、eval、summary 的任务即使完成标志在，每次启动运行器也照常运行一次（同 §3），复核失败的下游不开始；`--dry-run` 把它们列成“已完成、要复核”。
- `--dry-run` 只打印将要运行的顺序。
- S14 补注：排在前面的任务资源不够时，后面放得下的先跑；CPU 任务的 `CUDA_VISIBLE_DEVICES` 为空；命令的 `python` 换成运行器自己的解释器；每个任务最多重排 `--max-requeues`（默认 100）次；退出 0 但没写完成标志、要的资源超过本机总量都算失败；同一运行根同时只能有一个运行器（`logs/RUNNER.pid` 上的 flock 排他锁，运行期间一直持有，进程退出或被杀时内核释放；锁文件不删，内容是持锁进程的 PID，S14-fix1）；Ctrl-C 时把 SIGINT 转给正在跑的任务，等 `--grace-seconds`（默认 60）秒后强制结束，退出 130。

这和 v7 的 `run_jobs.py` 是同一种用法，只是按依赖调度而不是按段。

### 5.2 SLURM `tools/droid_slurm.py`

`python tools/droid_slurm.py jobs-full.json --out sbatch-full/ [--partition … --account … --qos …]` 生成：

- 每个任务一个 sbatch 脚本：`--job-name`（任务编号）、`--time`、`--gres=gpu:<n>`、`--cpus-per-task`、`--mem`、`--output`（日志，追加模式）、`--requeue`、`--signal=B:USR1@600`。脚本体运行任务命令；退出码是 75 时执行 `scontrol requeue $SLURM_JOB_ID`。
- 一个 `submit.sh`：按拓扑序 `sbatch --parsable` 提交，依赖写成 `--dependency=afterok:<上游作业号…>`；已完成（完成标志存在）的任务不提交，依赖它的任务不再写这条依赖。提交记录写 `submitted.json`（任务编号 → 作业号）。
- 集群相关的东西（分区、账户、模块加载、conda 环境）都是命令行参数或模板文件，不进任务图。

集群类型要等导师给出（架构决策 9）。S15 合并前必须在真实集群上用一个假任务核对三件事：收到 USR1 后存档退出、退出 75 后被重排并接着跑、重排期间下游任务保持等待而不是被取消。核对不过就改脚本模板（例如改成退出 0 后由脚本自己 `sbatch` 一个续跑作业），任务本身的约定不变。

### 5.3 容器列表

`python tools/droid_jobs.py … --format list` 输出 v7 那种按“# =====”分段的纯文本（每段是一个拓扑层），给没有排队系统、只能一条条跑命令的环境（云端容器、Kubernetes 任务）用；每条命令仍然遵守 §3 的约定，跑完一段再跑下一段。

## 6. 试跑关口

正式运行之前先跑 `jobs-pilot.json`（N = 500、U = 2000、1 个种子）。通过标准：

- 40 个任务全部完成，没有失败。
- 至少有一个训练任务真的被时长上限打断并续跑过：`pilot.yaml` 里把 `resources.train.time_limit_minutes` 设成 20，训练任务都会续跑几次；续跑后的日志覆盖 1…U 不重不漏。
- `check/env.json` 给出每个条件的单步用时和求谱后端的比较；据此在 `full.yaml` 里定 `training.updates`、`model.spectral_backend` 和各阶段的 `time_limit_minutes`（这些只能在看任何评测结果之前定）。
- 试跑的汇总表能完整生成。

试跑只用来核对流程和测速，不看数字下结论，不根据试跑结果改任何模型或训练超参数。
