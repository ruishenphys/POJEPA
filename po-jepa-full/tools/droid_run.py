"""在一台机器上按依赖跑完 DROID 任务图（docs/design/12-droid-jobs.md §5.1；S14）。

  python tools/droid_run.py jobs-full.json --gpus 0,1 --cpus 32 --mem-gb 256
  python tools/droid_run.py jobs-full.json --gpus 0,1 --cpus 32 --mem-gb 256 --dry-run

- 依赖都完成（看完成标志）的任务进入可跑队列，按 priority、再按编号取；当前排在前面的放不下时，后面放得下的先跑。
  GPU 任务独占它要的卡数（设 CUDA_VISIBLE_DEVICES），CPU 任务的 CUDA_VISIBLE_DEVICES 为空；CPU 和内存按任务图里的
  资源数记账，不超过命令行给的总量。要的资源比总量还多的任务一开始就记为失败。
- 退出 75：立即重排同一条命令（每个任务最多 --max-requeues 次）。其他非 0，或退出 0 但没写完成标志：记进失败列表，
  依赖它的任务不再开始，别的照常跑；全部能跑的跑完后列出失败任务的日志路径，退出 1。
- 本机没有时长上限：训练任务带着配置里的 --time-limit-minutes，到点存档退出 75 再被重排，和集群上的行为一致。
- 断电或 Ctrl-C 后重新运行同一条命令即可：完成的跳过，没完成的从断点接着做（各任务自己续跑，12 §3）。Ctrl-C 时
  把 SIGINT 转给正在跑的任务，等它们退出（最多 --grace-seconds 秒，之后强制结束），退出 130。
- 日志：`<run_root>/<config>/logs/<任务编号的 / 换成 __>.log`，重排和重跑时追加。同一个日志目录同时只能有一个运行器
  （`logs/RUNNER.pid` 上的 flock 排他锁，运行期间一直持有；只支持 Linux/macOS）。
- 命令在 --workdir（默认本文件所在的 po-jepa-full/）下运行；命令的第一个词是 `python` 时换成运行本脚本的解释器。
- --dry-run 只打印将要运行的顺序（假定每个任务都成功、一次跑一个），不写任何文件。

和 v7 的 run_jobs.py 是同一种用法，只是按依赖调度而不是按段。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

SCHEMA = "droid_jobs_v1"
LOCK = "RUNNER.pid"
INTERRUPTED = 130
DEFAULT_WORKDIR = Path(__file__).resolve().parents[1]


def load_graph(path: Path) -> dict:
    graph = json.loads(Path(path).read_text(encoding="utf-8"))
    if graph.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema is not {SCHEMA!r}")
    ids = [j["id"] for j in graph["jobs"]]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate job ids")
    known = set(ids)
    for j in graph["jobs"]:
        missing = [d for d in j["deps"] if d not in known]
        if missing:
            raise SystemExit(f"{path}: job {j['id']} depends on unknown jobs {missing}")
    return graph


def log_path(graph: dict, workdir: Path, job_id: str) -> Path:
    """12 §3（和 droid_jobs.log_path 相同的规则）。"""
    return workdir/graph["run_root"]/graph["config"]/"logs"/(job_id.replace("/", "__") + ".log")


def flag(workdir: Path, job: dict) -> Path:
    return workdir/job["complete"]


def order_key(job: dict) -> tuple:
    return (job["priority"], job["id"])


def argv_of(job: dict) -> list[str]:
    argv = list(job["command"])
    if argv and argv[0] == "python":
        argv[0] = sys.executable
    return argv


class Pool:
    """卡、CPU、内存的记账。"""

    def __init__(self, gpus: list[str], cpus: int, mem_gb: float):
        self.order = list(gpus)
        self.free_gpus, self.cpus, self.mem = list(gpus), cpus, mem_gb
        self.total = (len(gpus), cpus, mem_gb)

    def feasible(self, r: dict) -> bool:
        return r["gpus"] <= self.total[0] and r["cpus"] <= self.total[1] and r["mem_gb"] <= self.total[2]

    def fits(self, r: dict) -> bool:
        return r["gpus"] <= len(self.free_gpus) and r["cpus"] <= self.cpus and r["mem_gb"] <= self.mem

    def take(self, r: dict) -> list[str]:
        cards, self.free_gpus = self.free_gpus[:r["gpus"]], self.free_gpus[r["gpus"]:]
        self.cpus -= r["cpus"]
        self.mem -= r["mem_gb"]
        return cards

    def give(self, r: dict, cards: list[str]) -> None:
        self.free_gpus = sorted(self.free_gpus + cards, key=self.order.index)
        self.cpus += r["cpus"]
        self.mem += r["mem_gb"]


def dry_run(graph: dict, workdir: Path) -> int:
    jobs = {j["id"]: j for j in graph["jobs"]}
    done = {i for i, j in jobs.items() if flag(workdir, j).exists()}
    todo = [i for i in jobs if i not in done]
    print(f"{graph['config']}: {len(jobs)} 个任务，已完成 {len(done)} 个，待跑 {len(todo)} 个（假定都成功、一次跑一个的顺序）")
    for i in sorted(done, key=lambda i: order_key(jobs[i])):
        print(f"  已完成  {i}")
    simulated, n = set(done), 0
    while len(simulated) < len(jobs):
        ready = [jobs[i] for i in todo if i not in simulated and all(d in simulated for d in jobs[i]["deps"])]
        job = min(ready, key=order_key)
        n += 1
        r = job["resources"]
        print(f"  {n:4d}. {job['id']}  [priority {job['priority']}, GPU {r['gpus']}, CPU {r['cpus']}, "
              f"{r['mem_gb']} GB]  {shlex.join(job['command'])}")
        simulated.add(job["id"])
    return 0


def _acquire(lock: Path) -> int:
    """取运行器锁：`flock` 排他锁，从取得到运行器退出一直持有（S14-fix1：原先先看文件在不在再写 PID，两个运行器
    可以同时取得）。锁跟着打开的文件描述符走，进程不管怎么退出（包括被杀）内核都会释放，没有陈旧锁要处理；锁文件
    不删（删了别的进程可能锁在已删的旧文件上）。文件内容只是当前持锁进程的 PID，给人看。返回描述符，`_release` 关。"""
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = os.pread(fd, 64, 0).decode(errors="replace").strip() or "?"
        os.close(fd)
        raise SystemExit(f"另一个 droid_run.py 正在用 {lock.parent}（进程 {holder}）") from None
    os.ftruncate(fd, 0)
    os.pwrite(fd, str(os.getpid()).encode(), 0)
    return fd


def _release(fd: int) -> None:
    os.ftruncate(fd, 0)
    os.close(fd)


def run(graph: dict, workdir: Path, gpus: list[str], cpus: int, mem_gb: float, *, poll: float = 1.0,
        max_requeues: int = 100, grace: float = 60.0) -> int:
    jobs = {j["id"]: j for j in graph["jobs"]}
    pool = Pool(gpus, cpus, mem_gb)
    done = {i for i, j in jobs.items() if flag(workdir, j).exists()}
    pending = {i for i in jobs if i not in done}
    failed: dict[str, str] = {}
    blocked: set[str] = set()
    running: dict[str, tuple[subprocess.Popen, list[str], float]] = {}
    requeues = {i: 0 for i in jobs}
    print(f"{graph['config']}: {len(jobs)} 个任务，已完成 {len(done)} 个，待跑 {len(pending)} 个"
          f"（GPU {','.join(gpus) or '无'}，CPU {cpus}，内存 {mem_gb} GB）", flush=True)
    for i in sorted(pending, key=lambda i: order_key(jobs[i])):
        if not pool.feasible(jobs[i]["resources"]):
            failed[i] = f"要的资源 {jobs[i]['resources']} 超过本机总量（GPU {len(gpus)}，CPU {cpus}，{mem_gb} GB）"
            pending.discard(i)
            print(f"  [{i}] 失败：{failed[i]}", flush=True)

    def start(job: dict) -> None:
        r = job["resources"]
        cards = pool.take(r)
        log = log_path(graph, workdir, job["id"])
        log.parent.mkdir(parents=True, exist_ok=True)
        argv = argv_of(job)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(cards), PYTHONUNBUFFERED="1")
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} start (attempt {requeues[job['id']] + 1}, "
                    f"CUDA_VISIBLE_DEVICES={','.join(cards)!r})\n### {shlex.join(job['command'])}\n")
        out = open(log, "ab")
        try:
            proc = subprocess.Popen(argv, cwd=workdir, env=env, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        finally:
            out.close()
        running[job["id"]] = (proc, cards, time.time())
        print(f"  [{job['id']}] 开始{('（卡 ' + ','.join(cards) + '）') if cards else ''}", flush=True)

    def finish(job_id: str, code: int) -> None:
        _, cards, began = running.pop(job_id)
        job = jobs[job_id]
        pool.give(job["resources"], cards)
        minutes = (time.time() - began)/60
        log = log_path(graph, workdir, job_id)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"### {time.strftime('%Y-%m-%d %H:%M:%S')} exit {code}\n")
        if code in job["requeue_exit_codes"]:
            requeues[job_id] += 1
            if requeues[job_id] > max_requeues:
                failed[job_id] = f"退出 {code} 已重排 {max_requeues} 次"
                print(f"  [{job_id}] 失败：{failed[job_id]}，日志 {log}", flush=True)
                return
            pending.add(job_id)
            print(f"  [{job_id}] 退出 {code}（已存档），重排第 {requeues[job_id]} 次，{minutes:.1f} 分钟", flush=True)
        elif code != 0:
            failed[job_id] = f"退出码 {code}"
            print(f"  [{job_id}] 失败（退出码 {code}，{minutes:.1f} 分钟），日志 {log}", flush=True)
        elif not flag(workdir, job).exists():
            failed[job_id] = f"退出 0 但没有完成标志 {job['complete']}"
            print(f"  [{job_id}] 失败：{failed[job_id]}，日志 {log}", flush=True)
        else:
            done.add(job_id)
            print(f"  [{job_id}] 完成，{minutes:.1f} 分钟", flush=True)

    def block() -> None:
        changed = True
        while changed:
            changed = False
            for i in sorted(pending):
                if any(d in failed or d in blocked for d in jobs[i]["deps"]):
                    pending.discard(i)
                    blocked.add(i)
                    changed = True

    try:
        while True:
            for job_id, (proc, _, _) in list(running.items()):
                code = proc.poll()
                if code is not None:
                    finish(job_id, code)
            block()
            ready = sorted((jobs[i] for i in pending if all(d in done for d in jobs[i]["deps"])), key=order_key)
            for job in ready:
                if pool.fits(job["resources"]):
                    pending.discard(job["id"])
                    start(job)
            if not running:      # 资源全空还没有能开始的任务：剩下的都在等失败的上游（已记 blocked）或无法就绪
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        print(f"\n收到中断，把 SIGINT 转给 {len(running)} 个正在跑的任务，等它们退出……", flush=True)
        for proc, _, _ in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
        deadline = time.time() + grace
        for job_id, (proc, _, _) in running.items():
            try:
                proc.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            with open(log_path(graph, workdir, job_id), "a", encoding="utf-8") as f:
                f.write(f"### {time.strftime('%Y-%m-%d %H:%M:%S')} interrupted (exit {proc.returncode})\n")
        print("已停下。重新运行同一条命令会跳过完成的任务、接着做没完成的。", flush=True)
        return INTERRUPTED
    left = sorted(pending)
    print(f"\n完成 {len(done)}/{len(jobs)} 个任务。", flush=True)
    for i in sorted(failed):
        print(f"  失败：{i}（{failed[i]}）\n        日志：{log_path(graph, workdir, i)}", flush=True)
    if blocked:
        print(f"  因上游失败没有开始的：{', '.join(sorted(blocked))}", flush=True)
    if left:      # 不该发生：没有失败的上游却永远不就绪
        print(f"  没能开始的：{', '.join(left)}", flush=True)
    return 1 if failed or blocked or left else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs", type=Path, help="tools/droid_jobs.py 生成的任务图 JSON")
    p.add_argument("--gpus", default="", help="用哪几张卡，逗号分隔（例如 0,1）；不给就只跑 CPU 任务")
    p.add_argument("--cpus", type=int, required=True)
    p.add_argument("--mem-gb", type=float, required=True)
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="命令的工作目录（默认 po-jepa-full/）")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=1.0)
    p.add_argument("--max-requeues", type=int, default=100)
    p.add_argument("--grace-seconds", type=float, default=60.0)
    a = p.parse_args(argv)
    graph = load_graph(a.jobs)
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    if len(set(gpus)) != len(gpus):
        p.error("--gpus lists a card twice")
    workdir = a.workdir.resolve()
    if a.dry_run:
        return dry_run(graph, workdir)
    lock = workdir/graph["run_root"]/graph["config"]/"logs"/LOCK
    fd = _acquire(lock)
    try:
        return run(graph, workdir, gpus, a.cpus, a.mem_gb, poll=a.poll_seconds, max_requeues=a.max_requeues,
                   grace=a.grace_seconds)
    finally:
        _release(fd)


if __name__ == "__main__":
    sys.exit(main())
