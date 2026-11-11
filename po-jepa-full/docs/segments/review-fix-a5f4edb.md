# review-fix-a5f4edb：修第二轮朋友 Codex 审核（a5f4edb）的 full-model 意见

基于 full-model a5f4edb。来源：朋友 Codex 审核报告 `POJEPA_review_a5f4edb_2026-11-11.md` 第 1、3、5、6、7 节（1 P1、5 P2、
1 P3）和同一批探针里 Claude 顺带发现的一条（评测的 `--time-limit-minutes`）。v7 分支的 v7-P2-1 不在这里。全部不改数值
口径：正常命令下的模型、损失、批、评测数字逐位不变，改的只是“换了东西、或中断后”能否被拒绝、能否接着跑。

## S11-P1-1 统一入口和运行器凭完成标志跳过阶段身份核对

- `tools/droid.py`：完成标志在时只有 index、download、prepare、encode、calibrate、check 直接退出 0（这些产物只写一次，下游读
  时核身份：特征清单 SHA、σ_z/λ 的配置和条件）；train、eval、summary（`droid_jobs.RECHECKED_STAGES`）照常调用阶段。
  训练核 COMPLETE.json 绑定 report.json（没有 report.json 也拒绝）；评测由 `load_selected`、`source.json`、`job.json`
  核本次请求（S11a-fix1 那套）；汇总每次按当前产物重算。原来只给 `eval/spectral-gap` 开的例外并进这条规则。
- `tools/droid_run.py`：启动时 train、eval、summary 即使有完成标志也排进待跑，照常运行一次；复核失败时下游不开始。
  `--dry-run` 列成“已完成、要复核”。代价：每次重启运行器，已完成的评测任务要重读数据和检查点各一次（没实测，估计每个几分钟）。
- `tools/droid_summary.write_output`：新算出的输出和已有的逐文件逐字节相同时保留原目录，不再多一份 `summary.previousN`；
  不同（例如新增了评测任务）才把旧的改名后换上新的。
- 测试：入口对空 `{}` 完成标志的 train、eval（训练过的和不训练的）、summary 进真实阶段代码并报错；fake 分派里
  calibrate/check 跳过、train/eval/summary 照常调用；脚本方式空 COMPLETE 的 summary 退出 1；运行器复核与阻塞下游；
  汇总内容不变不换目录、少一个评测任务就换。

## S09b-P2-1 评测不核训练种子

- `evaluate.load_selected` 加必填参数 `seed`：报告的 `seed` 必须等于它；`registration.json` 的文件 SHA 等于报告记的
  `registration_sha256`，登记的 schema、条件、种子也必须是本次请求的。`source.json` 的 `registration_sha256` 改记登记
  文件实际的 SHA（正常情况和原来的值相同）。
- 测试：1703 的目录请求 1704 拒绝；登记被改/被删、登记的条件或种子被改（报告重新绑定）拒绝；报告和登记一起改成 1704
  时请求 1703 拒绝；`eval_job.run` 用错种子的目录在写 source.json 之前就拒绝。

## S09a-P2-2 λ 文件不核条件名和特征清单

- `train.read_lambda` 改为必须给 `condition`、`feature_manifest_sha256`，可选 `train_set`：schema = `droid_lambda_v1`、内容
  的 `condition` 等于要读的条件、`config_sha256`、`sigma_z_sha256`、`feature_manifest_sha256` 都对上（原来只在记了时核
  前两项）；训练任务另核 `train_set`（{count, sha256}）。`load_calibration` 同步加参数；训练和评测的 `main` 改成先读数据
  再读标定。
- 测试：把 po_jepa_full 的 λ 复制成 `lambda-fixed_fourier.json` 拒绝；换特征清单、训练集、σ_z、schema，缺字段，各自拒绝；
  同样内容换个文件名照样读得出。

## S09a-P2-1、S09b-P2-2 `*.partial` 残留挡住重跑

- `cache._write_json` 用 `open("xb")` 写 `<名>.partial` 再改名；中断在改名前，同名临时文件留下，下次写就 FileExistsError。
  `runtime/cache.py` 是 v7 共享模块，不改；改在调用方：训练任务（未完成时）在进入训练前删目录根上的 `*.partial` 文件
  （report、COMPLETE、FAILED、登记）；评测任务（未完成时）在写 source.json 之前删目录根上的 `*.partial` 文件。
  已完成的目录不动。
- 测试：训练留 report.json.partial，或 COMPLETE.json.partial + FAILED.json.partial + registration.json.partial，重跑都
  完成且报告的选点不变；FAILED.json.partial 在时出错仍能写 FAILED.json，`--retry` 后完成；评测留 source.json.partial
  和 job.json.partial，重跑完成。

## S13-P2-1 汇总的日志核对只看行首尾

- `droid_summary.read_train` 对日志行调用训练端同一条 `train.check_log_coverage`，结果记进 `TrainView.log_problem`
  （通过为 None）；“训练日志”核对在原来的首尾覆盖之外，`log_problem` 非空也算不通过。
- 测试：真实训练目录，第一行保持 first/last 不变、行内删掉一步或重复一步，读取器都报出来；核对表里 `log_problem` 非空
  单独一例不通过。

## S14-P3-1 运行器取锁失败时描述符没关

- `_acquire` 取得描述符后任何失败（含写 PID 时磁盘满）都先关描述符再抛出；`_release` 清空失败也关描述符。
- 测试：注入 `os.pwrite` 抛 OSError(28)，同一进程再取锁能取到；`os.ftruncate` 抛错时描述符已关闭。

## 评测的 `--time-limit-minutes` 没有生效（P3）

- 09 §11 定的是“只记录”，但原来哪里都没记。现在评测任务没完成时每次启动往评测目录的 `starts.jsonl` 追加一行
  （`time_limit_minutes`、`device`），不进 job.json 的请求身份（改时长上限不应让已做的批作废）；完成后复核不再追加。

## 顺手：poses_to_diffs 存档测试的跨平台容差

- `tests/test_droid_actions.py` 照 S06a-fix1 的做法：同一环境执行上游原方法逐位相同 + 和存档金标准差 < 1e-12。AST 核对不变，
  `droid/actions.py` 没改。

## 没做的

- 训练任务完成后的复核只核 COMPLETE.json 绑定 report.json，不重算登记（重算要读全部训练数据、建模型，且登记里记了
  Python/torch 版本，升级环境后所有已完成任务都会被拒）。换 λ/σ_z 后的旧训练由汇总的“σ_z”“λ”核对和评测读检查点时
  的损失配置快照拦下。
- 审核报告第 8 节第 4 条（同步数对照的“等步数”口径）要 Jerry 定，没动。
