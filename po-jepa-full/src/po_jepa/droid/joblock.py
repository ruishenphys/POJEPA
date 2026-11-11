"""一个任务输出目录的排他锁（RF-P2-3）。

训练、评测、截断间隙任务在“看完成标志 → 删上次留下的 `*.partial` → 写文件”整个过程中持有 `<输出目录>/job.lock` 上的
排他锁：手工启动和运行器重试同时落到同一个目录时，后来的进程立即报错退出，不会删掉另一个进程正在写的临时文件。

锁跟着打开的文件描述符走（Linux/macOS 用 `fcntl.flock`，Windows 用 `msvcrt.locking` 锁第一个字节），进程不管怎么退出
（包括被杀）都会释放，没有陈旧锁要处理；锁文件不删（删了别的进程可能锁在已删的旧文件上）。文件内容是当前持锁进程的
PID，只给人看。和运行器自己的 `logs/RUNNER.pid`（`tools/droid_run.py`）是两把锁：那把只防两个运行器，这把防的是
直接启动的阶段。
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator

LOCK_FILE = "job.lock"


class JobBusy(RuntimeError):
    """另一个进程正在用这个任务目录。"""


def _try_lock(fd: int) -> bool:
    try:
        import fcntl
    except ImportError:                       # Windows
        import msvcrt
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


@contextlib.contextmanager
def hold(directory: Path | str) -> Iterator[Path]:
    """建目录、取 `job.lock` 的排他锁（不等待：取不到就 `JobBusy`），退出时释放。返回锁文件路径。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/LOCK_FILE
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if not _try_lock(fd):
            try:
                holder = os.read(fd, 64).decode(errors="replace").strip() or "?"
            except OSError:                   # Windows 上被锁住的字节读不出来
                holder = "?"
            raise JobBusy(f"another process (pid {holder}) is running the job in {directory}; "
                          "wait for it or stop it before starting this one again")
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid():>10}".encode())   # 定长，Windows 锁的第一个字节一直在文件里
        yield lock
    finally:
        os.close(fd)
