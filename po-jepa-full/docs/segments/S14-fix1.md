# S14-fix1：运行器锁与删原始文件的完成语义（Codex S14-P2-1、S14-P2-2）

基于 full-model 797063e。来源：朋友 Codex 审核报告（e651582，2026-11-10）S14 节。

## S14-P2-1 运行器锁非原子

- 原来 `_acquire` 先 `exists()` 再 `write_text(pid)`，两个运行器同时启动都能取得。
- 改为 `fcntl.flock(LOCK_EX | LOCK_NB)`，描述符在整个运行期间持有，`_release` 清空内容并关闭。进程被杀时内核释放锁，
  不再有陈旧锁判断。锁文件不删（删了别的进程可能锁在已删的旧 inode 上）。只支持 Linux/macOS（运行器本来就用
  `start_new_session`/`killpg`）。
- 测试：8 个独立进程同时放行抢锁，恰好 1 个取得；持锁进程在时 `_acquire` 和整个运行器都被拒并报出 PID；持锁进程
  被 SIGKILL 后能立即取得。

## S14-P2-2 完成标志早于删原始文件

- 训练片 `encode/<片>`（带 `--delete-raw`）的任务完成标志改成 `features/<片>/_raw_deleted/<name>.json`
  （`features.raw_cleanup_path`），由 `encode.delete_raw_files` 在整片删完之后写，内容绑定特征完成记录和片清单的 SHA。
  特征完成记录 `features/<片>/_complete/<name>.json` 不变，仍是编码完成的证据，读特征的代码不受影响。
- `tools/droid_jobs.complete_path` 对训练片返回新标志，统一入口 `tools/droid.py` 和运行器 `droid_run.py` 都看任务图的
  `complete`，所以两层同时生效；`val`、`id_test`、`ood_test`、`test-extra` 的标志不变。
- 中断后重跑：入口看到新标志不在就进入编码阶段，`run_encode` 见特征已完成，跳过编码、重做幂等的删除，写标志。
- 测试：真实入口 `droid.main(['encode', …, '--delete-raw', '--fake-encoder'])`，删到第 2 个原始文件时模拟崩溃；
  特征完成记录在、新标志不在；同一条命令重跑返回 0、全部删完、写标志；再跑一次才输出 already complete。
- 不改特征数值、编码、划分和删除范围。

## 验证

- `tests/test_droid_jobs.py` 16 过（含 2 个新锁测试）；`tests/test_droid_encode.py` 新恢复测试过。
- 改坏 5 处（去掉 flock、训练片标志改回旧路径 ×2 个测试、标志在删除前返回、释放不清空），全部被抓到。
