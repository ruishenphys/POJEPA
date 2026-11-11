# review-fix-e9016a8：修第三轮朋友 Codex 审核（e9016a8）的 full-model 意见

基于 full-model e9016a8。来源：朋友 Codex 第三轮审核报告 `POJEPA_review_round3_2026-11-11.md` 第 1 节（2 P1、4 P2）。
第 2 节（v7 extra_eval）不在这里。全部不改数值口径：正常命令下的模型、损失、批、选点、评测数字不变；改的是“目录被复制、
标定或特征换过、两个进程撞到同一目录、中断后”能否被拒绝、能否接着跑。唯一的格式变化是检查点存的来源字典多一项
`droid_training`（见 RF-P2-1），检查点文件的 SHA 因此和旧代码写的不同，权重本身不变。

## RF-P1-1 独立汇总把目录名当种子身份

- 核实：属实。`collect` 按 `(条件, 种子)` 的目录槽位收集，只比训练和评测之间的 SHA，没把记录里的条件、种子和槽位比；
  把 seed1703 的训练和评测目录原样复制到 seed1704 槽位，汇总 `status: ok`。
- `tools/droid_summary.collect`：每个评测目录先核 `job.json` 等于 `report.json` 的 job，再核报告、`job.json`、读取器给的
  来源（真实读取器读 job.json 和 source.json）里的 (条件, 种子) 都等于槽位（种子连类型一起比，1704.0、True 不算），每个
  训练目录核读取器给的登记 (条件, 种子) 等于槽位；任一不等报错，不读行文件。
- `read_eval_view`：source.json 的 schema、条件、种子必须等于 job.json 的，并带出它记的训练 `registration_sha256`；“评测”
  核对另比它和训练目录 registration.json 的 SHA（`TrainView.registration_sha256`）。
- 测试：整个评测目录复制、只有来源是另一个种子、训练登记是另一个种子、job.json 换成别的任务的，都报错；`read_eval_view`
  的 source/job 不一致三例；“评测”核对里登记 SHA 不同单独一例不通过。

## RF-P1-2 σ_z 没有与当前特征缓存和动作尺度绑定

- 核实：属实。`run_sigma_z` 写了 `feature_manifest_sha256`、`action_scale_sha256`，`read_sigma_z` 只核数值、配置 SHA
  也是有才核；清单 A 下标定的 σ_z 可以被清单 C 的训练读进去，λ 标定同样。
- `spectral_gap.sigma_z_doc`（新）：schema、配置 SHA、数值、两项来源 SHA 都必须在。`read_sigma_z(path, config, *,
  feature_manifest_sha256, action_scale_sha256)`：两项都必须等于调用者按当前数据根算出的（全部片的特征总清单）。调用者：
  训练 main、λ 标定（改成先读数据再读 σ_z）、评测 main（动作尺度取 `data_sha256["action_scale.json"]`）、check/env
  main、截断间隙 main（另算一次全量清单给 σ_z，job.json 里记的仍是测试片清单，job 身份不变）。
- 评测读 λ 时传所选训练登记的训练集（`eval_job.registered_train_set`）；`load_selected` 之后核评测读到的 σ_z、λ 文件
  SHA 和取值等于训练登记的（`check_training_calibration`）。
- 汇总的“σ_z”另核 `sigma_z.json` 的动作尺度 SHA 等于数据根的、截断间隙 `inputs.sigma_z_sha256`、PO-JEPA 类训练登记、
  check/env 记的 σ_z 文件 SHA 都等于当前文件；“特征”把 `sigma_z.json` 记的特征总清单算进去。
- 测试：read_sigma_z 换特征、换动作尺度、缺字段、schema 不对；λ 标定拒绝另一份特征上的 σ_z 且不写文件；load_calibration
  两种来源不符；截断间隙的两项；评测读到的 σ_z（SHA 或取值）、λ 与训练登记不同；汇总四个新例各自只让一项不通过。

## RF-P2-1 种子校验可被一致的元数据改写绕过

- 核实：属实。检查点只存批的来源字典，没有训练种子或登记指纹；登记和报告一起改成 1704 并重新绑定后，请求 1704 照样
  读进原检查点；登记的 `config_sha256` 也没比。
- `train.checkpoint_provenance`：存进检查点的来源 = 批的来源字典 + `droid_training` = {condition, seed,
  registration_sha256}（v7 的保存函数原样存来源，`training._provenance` 只要求固定字段在）；续训读回时整份比较。
  `evaluate.load_selected` 核这一项等于本次请求的条件、种子和 registration.json 的文件 SHA，并核登记的配置 SHA。
- 旧检查点：DROID 还没有真实训练产物（全部只到 CPU 验证），所以不做旧格式兼容，没有这一项的检查点直接拒绝。
- 测试：检查点里记的身份；登记和报告改成 1704 后请求 1704 拒绝；登记的配置 SHA 改了（重新绑定）拒绝。

## RF-P2-2 帧率、分辨率轴无法从写入中断续跑

- 核实：属实。`frame_rate.json`、`resolution.json` 在 `<测试集>/` 下，评测任务只清根目录的 `*.partial`。
- `evaluate._drop_partial`：两个轴在写结果前删自己那一个 `<名>.partial`（只删这一个，目录锁保证没有别的写者）。
- 测试：两个测试集各留下两个轴的 `.partial`，评测照常做完、不留临时文件。

## RF-P2-3 同一输出目录的两个进程可互删正在写的临时文件

- 核实：属实。运行器的锁只防两个运行器，直接启动的阶段不受保护。
- 新模块 `droid/joblock.py`：`<输出目录>/job.lock` 上的排他锁（Linux/macOS `fcntl.flock`，Windows `msvcrt.locking`），
  不等待，取不到抛 `JobBusy`。训练（从看完成标志起，取锁失败不写 FAILED.json）、评测、截断间隙任务全程持锁。
- 测试：第一个训练停在 registration.json.partial 改名前时第二次启动报 JobBusy、不删对方文件、不写 FAILED.json，第一个
  照常做完；评测目录被锁时第二次启动报错、目录里只多锁文件。

## RF-P2-4 check/env 可被空文件或其他请求的通过记录绕过

- 核实：属实。统一入口、运行器只看 env.json 在不在，`check.main` 见到文件就退出 0，汇总只核 schema 和 passed。
- `check.verify_env`：已有记录必须是通过记录，配置 SHA、特征总清单 SHA、σ_z 文件 SHA 等于按当前配置、数据根、运行根算
  出的；`check.main` 在 env.json 已有时只算清单、读 σ_z、核记录，不重跑检查。`check` 进 `RECHECKED_STAGES`，统一入口和
  运行器每次都调用阶段复核，复核失败时依赖它的训练不开始。汇总的“配置”“特征”“σ_z”把 env.json 记的三项算进去。
- 测试：verify_env 的通过、三种不符、空记录、未通过记录；命令行在已有记录时不重跑、`{}` 和别的特征下的记录报错；入口
  和运行器把 check 列为复核阶段，复核失败阻塞下游。

## 没做的

- 直接启动 `train`（不经运行器）时不读 check/env.json：设计 09 §7 的前置由任务图的依赖保证，运行器现在会复核它。
- 汇总阶段的输出目录没加锁（`write_output` 先写临时目录再整体换名，不删别人的临时文件）。
