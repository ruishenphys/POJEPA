"""在一台多卡机器上自动跑完 make_v7_jobs.py 生成的任务列表（没有排队系统时用）。

  python tools/run_jobs.py jobs-pilot.txt --gpus 0,1
  python tools/run_jobs.py jobs-full.txt  --gpus 0,1
  python tools/run_jobs.py jobs-variants.txt --gpus 0,1

- 按 “# =====” 分段：一段里的命令分给各张卡同时跑（每张卡同一时间只跑一条），整段完成后才开始下一段。
- 每条命令的输出存到 --logs 下（默认 ../logs-v7/<任务文件名>/），完成的命令留一个 .done 标记。
- **断了（关机、断电、Ctrl-C、某条报错）直接重新运行同一条命令即可**：
  已完成的跳过；编码会跳过已完成的轨迹；训练任务（run.py run）自动改成从原目录续训（resume，输出到 原目录-resumeN）；
  其他 GPU 任务（calibrate / check / transfer）把半截的输出目录改名为 原目录.brokenN 后重跑。
  后面段里用 --training-run / --job 指向的训练目录，如果是靠续训完成的，会自动换成续训的新目录。
- 某段有命令失败：同段其他命令照常跑完，然后停下，不进入下一段，并列出失败命令的日志。重新运行会再试失败的那几条。
"""
import argparse
import hashlib
import os
from pathlib import Path
import queue
import re
import shlex
import subprocess
import sys
import threading
import time


def parse_sections(path):
    sections, current = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = re.match(r"#\s*=====\s*(\S+)\s*(.*?)\s*=*$", line)
        if m:
            current = dict(name=m.group(1), title=m.group(2), commands=[])
            sections.append(current)
        elif line and not line.startswith("#"):
            if current is None:
                current = dict(name="0", title="", commands=[])
                sections.append(current)
            current["commands"].append(line)
    return [s for s in sections if s["commands"]]


def option(argv, name):
    return argv[argv.index(name)+1] if name in argv else None


def set_option(argv, name, value):
    argv = list(argv)
    if name in argv:
        argv[argv.index(name)+1] = value
    else:
        argv += [name, value]
    return argv


def complete(path):
    path = Path(path)
    return any((path/sub/"COMPLETE.json").exists() for sub in ("", "run/job", "job"))


def has_job(path):
    path = Path(path)
    return any((path/sub/"job.json").exists() for sub in ("", "run/job", "job"))


def attempts(path):
    """原目录和它的续训目录（原目录-resume1、-resume2…），按先后排列。"""
    path = Path(path)
    found = [path] if path.exists() else []
    k = 1
    while Path(f"{path}-resume{k}").exists():
        found.append(Path(f"{path}-resume{k}")); k += 1
    return found


def finished_attempt(path):
    for p in attempts(path):
        if complete(p):
            return p
    return None


def aside(path):
    k = 1
    while Path(f"{path}.broken{k}").exists():
        k += 1
    Path(path).rename(f"{path}.broken{k}")
    return f"{path}.broken{k}"


def prepare(cmd, workdir):
    """把一条命令改写成“可以接着跑”的形式；返回 (要执行的命令, 说明) 或 (None, 说明) 表示已经完成。"""
    argv = shlex.split(cmd)
    notes = []
    for name in ("--training-run", "--job"):   # 后面段指向的训练目录：换成实际完成的那一次
        target = option(argv, name)
        if target and not complete(workdir/target):
            done = finished_attempt(workdir/target)
            if done is not None:
                argv = set_option(argv, name, os.path.relpath(done, workdir))
                notes.append(f"{name} 改用 {os.path.relpath(done, workdir)}")
    script = argv[1] if len(argv) > 1 else ""
    sub = argv[2] if len(argv) > 2 else ""
    out = option(argv, "--output")
    if script == "run.py" and out and (workdir/out).exists():
        out_path = workdir/out
        if sub == "run":
            done = finished_attempt(out_path)
            if done is not None:
                return None, f"已完成（{os.path.relpath(done, workdir)}）"
            tried = [p for p in attempts(out_path) if has_job(p)]
            if tried:
                source = tried[-1]
                k = len(attempts(out_path))
                argv[2] = "resume"
                argv = set_option(argv, "--resume-from", os.path.relpath(source, workdir))
                argv = set_option(argv, "--output", f"{out}-resume{k}")
                notes.append(f"上次中断，从 {os.path.relpath(source, workdir)} 续训，输出到 {out}-resume{k}")
            else:
                notes.append(f"上次中断在开始训练之前，旧目录改名为 {os.path.relpath(aside(out_path), workdir)}")
        elif sub in ("calibrate", "check", "transfer", "evaluate"):
            if complete(out_path):
                return None, "已完成"
            notes.append(f"上次没跑完，旧目录改名为 {os.path.relpath(aside(out_path), workdir)}")
    return shlex.join(argv), "；".join(notes)


def priority(cmd):
    """长任务先跑，两张卡的总时间更短。"""
    if "--condition vjepa2_ac " in cmd + " ":
        return 0
    if "--condition po_jepa" in cmd or "--condition matched_transformer" in cmd or "--condition vjepa2_ac_small" in cmd:
        return 1
    return 2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs", type=Path, help="make_v7_jobs.py 生成的任务文件")
    p.add_argument("--gpus", default="0,1", help="用哪几张卡，逗号分隔（默认 0,1）")
    p.add_argument("--logs", type=Path, help="日志目录（默认 ../logs-v7/<任务文件名>）")
    p.add_argument("--only", help="只跑这些段，逗号分隔（例如 A,B）")
    p.add_argument("--dry-run", action="store_true", help="只列出将要执行的命令")
    a = p.parse_args()
    workdir = Path.cwd()
    if not (workdir/"run.py").exists():
        raise SystemExit("请在 code-v7/po-jepa 目录里运行")
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    logs = a.logs or workdir.parent/"logs-v7"/a.jobs.stem
    logs.mkdir(parents=True, exist_ok=True)
    lock = logs/"RUNNING.pid"
    if lock.exists() and not a.dry_run:
        try:
            os.kill(int(lock.read_text()), 0)
            raise SystemExit(f"另一个 run_jobs.py 正在用 {logs}（进程 {lock.read_text()}）")
        except (ProcessLookupError, ValueError):
            pass
    if not a.dry_run:
        lock.write_text(str(os.getpid()))
    sections = parse_sections(a.jobs)
    only = set(a.only.split(",")) if a.only else None
    try:
        for section in sections:
            if only and section["name"] not in only:
                continue
            ok = run_section(section, gpus, logs, workdir, a.dry_run)
            if not ok:
                print(f"\n第 {section['name']} 段有任务失败，已停下（没有进入下一段）。看上面列出的日志；"
                      f"修好后重新运行同一条命令会接着跑。", flush=True)
                return 1
        print("\n全部完成。", flush=True)
        return 0
    finally:
        if not a.dry_run and lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()


def run_section(section, gpus, logs, workdir, dry_run):
    items = []
    for i, cmd in enumerate(section["commands"]):
        key = f"{section['name']}-{i:03d}"
        tag = hashlib.sha256(cmd.encode()).hexdigest()[:12]
        items.append(dict(key=key, cmd=cmd, done=logs/f"{key}.{tag}.done", log=logs/f"{key}.log"))
    todo = [it for it in items if not it["done"].exists()]
    print(f"\n===== {section['name']} {section['title']}：共 {len(items)} 条，已完成 {len(items)-len(todo)} 条，"
          f"待跑 {len(todo)} 条（{len(gpus)} 张卡）", flush=True)
    if dry_run:
        for it in todo:
            print(f"  [{it['key']}] {it['cmd']}")
        return True
    q = queue.Queue()
    for it in sorted(todo, key=lambda it: priority(it["cmd"])):
        q.put(it)
    failed, lock = [], threading.Lock()

    def worker(gpu):
        while True:
            try:
                it = q.get_nowait()
            except queue.Empty:
                return
            with lock:
                cmd, note = prepare(it["cmd"], workdir)
            if cmd is None:
                it["done"].write_text(it["cmd"]+"\n"+note+"\n")
                print(f"  [{it['key']}] 跳过：{note}", flush=True)
                continue
            start = time.time()
            print(f"  [{it['key']}] 卡 {gpu} 开始{('（'+note+'）') if note else ''}：{cmd}", flush=True)
            with open(it["log"], "a", encoding="utf-8") as log:
                log.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} 卡 {gpu}\n### {cmd}\n" + (f"### {note}\n" if note else ""))
                log.flush()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
                code = subprocess.call(cmd, shell=True, cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT)
            minutes = (time.time()-start)/60
            if code == 0:
                it["done"].write_text(it["cmd"]+"\n"+cmd+"\n")
                print(f"  [{it['key']}] 卡 {gpu} 完成，{minutes:.0f} 分钟", flush=True)
            else:
                with lock:
                    failed.append(it)
                print(f"  [{it['key']}] 卡 {gpu} 失败（退出码 {code}，{minutes:.0f} 分钟），日志 {it['log']}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for it in failed:
        print(f"  失败：[{it['key']}] {it['cmd']}\n        日志：{it['log']}", flush=True)
    return not failed


if __name__ == "__main__":
    sys.exit(main())
