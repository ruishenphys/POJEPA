# review-fix-fe48a5c：修第四轮朋友 Codex 复审（fe48a5c）的 full-model 意见

基于 full-model fe48a5c。来源：第四轮复审报告 `POJEPA_review_round4_codex-20261112.md` 第 1 节（1 P1，另两处测试覆盖问题）。
不改数值口径：正常命令下的模型、损失、批、选点、评测和汇总数字不变；检查点格式不变。

## R4-1-P1-1 独立汇总不读检查点里的训练身份

- 核实：属实。`tools/droid_summary.read_train` 只读 registration.json、report.json、selection.json，从不打开检查点；把 seed1703
  的训练和评测目录复制到 seed1704 槽位，同时重标登记、报告、source、job 并重新绑定各 SHA，`_check_slot` 全部通过，
  汇总 `status: ok`、种子 [1703, 1704]。评测端 `evaluate.load_selected` 会核检查点的 `droid_training`（RF-P2-1），汇总没有。
- `droid_summary.selected_checkpoint_identity`（新），`read_train` 在核完登记和报告后调用：选中的检查点必须在
  `checkpoints/` 下、文件 SHA 等于 selection.json 记的；用 `torch.load(mmap=True, weights_only=True)` 只取来源字典（官方规模
  AC 的权重不整份读进内存），`provenance.droid_training` 必须等于 {登记的条件、登记的种子（类型也要是 int）、registration.json
  的文件 SHA}；缺这一项拒绝。登记的 (条件, 种子) 又已被 `collect` 核等于槽位，所以一致重标的目录在汇总里报错。
- 防护边界：连检查点一起重写（重新保存、重新绑定所有 SHA）属于蓄意伪造，不防。
- 测试：`test_read_train_on_a_real_training_directory` 用真实训练产物一致重标成 seed1704（检查点不动）报错，换检查点字节
  报错，并核 `Readers()` 的默认训练读取器就是 `read_train`；`test_selected_checkpoint_identity` 覆盖种子、条件、登记 SHA
  不符、旧格式无此项、种子是 1704.0、SHA 不符、路径逃出 checkpoints/。

## 两处测试覆盖（报告第 8 点和测试表）

- 谱间隙 CLI 测试夹具里全量清单等于测试片清单（`test_droid_evaluate_extra.py`），把测试片清单误当全量传给 `read_sigma_z`
  的实现也能通过。改为夹具多一个只有清单的片 `train_extra`，断言两者不同、job 里记的仍是测试片清单；临时把实现改成传
  测试片清单，这个测试失败（核过后已还原）。
- `tools/droid_run.py` 顶层 `import fcntl` 使 `test_droid_jobs.py` 在 Windows 上整个文件收集失败。`fcntl` 挪进 `_acquire`；
  真起运行器或运行器锁的 10 个测试在 Windows 上标 skip（运行器锁仍只支持 Linux/macOS，同前），其余照常运行。

## 验证（CPU 验证级，Python 3.12.3、torch 2.11.0+cpu）

- 相关 4 个文件 summary、calibrate_check、evaluate_extra、jobs：54 个里 53 过，失败的 1 个是新夹具漏建目录，修后该测试过。
- 按约定修 Codex 意见不做故意改坏；上面那一处“传测试片清单”是为确认审核指出的测试缺口确实补上。全量回归见交接。
