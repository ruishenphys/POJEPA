"""16 §5、§12：补跑开始前核对 v7 的产物（A0 段，CPU）。

  python tools/addon_preflight.py --v7-results ../../results-v7/full --features ../../features-v7 --stage full \\
      --output ../../results-addon/full/preflight.json
  python tools/addon_preflight.py --v7-results ../../results-v7/pilot --features ../../features-v7 \\
      --v6-features "$V6" --stage pilot --output ../../results-addon/pilot/preflight.json

1. v7 的 D 段：本阶段每个场景 × 种子 × 6 个 v7 条件的任务目录（含续训目录 原目录-resumeN，规则和 run_jobs.py 相同）
   有一次已完成，任务记录（job.json）的条件、数据集、种子对得上。
2. 特征：每个场景的 <features>/<场景>-<条数> 是完整的 v7 编码（data.approved_cache_metadata），每条轨迹的 manifest.json
   还在且 SHA 等于编码报告里记的值；缓存身份和上面每个 v7 任务 report.json 里记的 preflight.cache_identity 逐项相同。
   给了 --v6-features（试跑）时，double_well-200 的特征是那个目录（v6 编出的，approved_cache_metadata 走 v6 的核对），
   条件和 make_addon_jobs.py、make_v7_jobs.py 的同名选项相同；其他场景照旧。这个目录另要求 0000–0199 每条轨迹的
   COMPLETE.json、manifest.json、raw_z.npy 都在，首尾两条用 v6 的逐条读法完整读一遍（v6_trajectories）。
有缺的或对不上的全部列出来后报错退出（补跑停在 A0），不写输出；都通过才写 --output。
重跑：输出已存在且内容相同就跳过，不同就报错。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_addon_jobs import STAGES, V7_CONDITIONS, name, split_scene, uses_v6  # noqa: E402
from run_jobs import finished_attempt  # noqa: E402
from po_jepa import datasets  # noqa: E402
from po_jepa.runtime import cache, data, v7jobs  # noqa: E402
from po_jepa.runtime.workflow import read  # noqa: E402

SCHEMA = "addon_preflight_v1"
V6_FILES = ("COMPLETE.json", "manifest.json", "raw_z.npy")


def _normal(value):
    """JSON 往返（整数键变成字符串），和 report.json 里记下的形式可比。"""
    return json.loads(json.dumps(value, allow_nan=False))


def v6_trajectories(root: Path, n: int, problems: list) -> bool:
    """--v6-features 的目录：每条轨迹的三个文件都在，首尾两条用 v6 的逐条读法（cache._read_complete_cache_uncached：
    核 manifest、来源、raw_z 的 SHA、shape 和有限性）完整读一遍。git 克隆里的 features/ 只有提交进仓库的元数据
    （report.json、COMPLETE.json、registration.json、0000/manifest.json），编码报告和缓存身份都对得上，只有这里拦得下。"""
    missing = [f"{tid:04d}" for tid in range(n) if not all((root/f"{tid:04d}"/name).is_file() for name in V6_FILES)]
    if missing:
        problems.append(f"特征 {root}：{len(missing)} 条轨迹缺 COMPLETE.json、manifest.json 或 raw_z.npy（第一条是 {missing[0]}）；"
                        "这不是完整的 v6 特征目录（git 克隆里的 features/ 只有元数据）")
        return False
    for tid in (0, n-1):
        try:
            cache._read_complete_cache_uncached(root/f"{tid:04d}")
        except Exception as error:
            problems.append(f"特征 {root}：轨迹 {tid:04d} 用 v6 的读法读不出来：{type(error).__name__}: {error}")
            return False
    return True


def scene_identity(features: Path, scene: str, n: int, problems: list, *, v6=False):
    """场景的特征缓存身份（v7 任务 preflight 记的同一个函数），并逐条核 manifest.json；v6=True 时另核 v6_trajectories。
    出错记进 problems，返回 None。"""
    system, knobs = split_scene(scene)
    old = datasets._set_for_tests(datasets.DatasetSpec(system, n, knobs))
    try:
        identity = data.approved_cache_metadata(features)
        if v6 and not v6_trajectories(features, n, problems):
            return None
    except Exception as error:
        problems.append(f"特征 {features}：{type(error).__name__}: {error}")
        return None
    finally:
        datasets._set_for_tests(old)
    for tid, digest in identity["manifest_sha256_by_id"].items():
        path = features/f"{tid:04d}"/"manifest.json"
        if not path.is_file() or cache.file_sha(path) != digest:
            problems.append(f"特征 {features}：轨迹 {tid:04d} 的 manifest.json 缺失或和编码报告不符")
    return _normal(identity)


def preflight(v7_results, features, stage, v6_features=None) -> dict:
    """返回要写的记录；有问题就把全部问题列出来后报 ValueError。"""
    spec = STAGES[stage]
    v7_results, features = Path(v7_results).resolve(), Path(features).resolve()
    v6 = Path(v6_features).resolve() if v6_features else None
    problems, scenes = [], {}
    for scene in spec["scenes"]:
        from_v6 = uses_v6(scene, spec["n"], v6)
        root = v6 if from_v6 else features/f"{name(scene)}-{spec['n']}"
        identity = scene_identity(root, scene, spec["n"], problems, v6=from_v6)
        tag = f"{name(scene)}-{spec['n']}"
        jobs = {}
        for seed in spec["seeds"]:
            for condition in V7_CONDITIONS:
                path = v7_results/name(scene)/f"seed{seed}"/condition
                done = finished_attempt(path)
                if done is None:
                    problems.append(f"v7 任务没有完成：{path}（含 -resumeN）")
                    continue
                job_dir = v7jobs._job_dir(done)
                job = read(job_dir/"job.json")
                if (job.get("condition"), job.get("dataset"), job.get("seed"), job.get("updates")) != (
                        condition, tag, seed, spec["updates"]):
                    problems.append(f"v7 任务记录对不上：{job_dir/'job.json'}")
                    continue
                recorded = read(job_dir/"report.json")["preflight"]["cache_identity"]
                if identity is not None and recorded != identity:
                    problems.append(f"v7 任务 {job_dir} 记的特征缓存身份和 {root} 现在的不同")
                jobs[f"seed{seed}/{condition}"] = os.path.relpath(job_dir, v7_results)
        if identity is not None:
            scenes[name(scene)] = dict(features=str(root), encode_report_sha256=identity["encode_report_sha256"],
                                       cache_identity_sha256=hashlib.sha256(cache._json_bytes(identity)).hexdigest(),
                                       trajectories=len(identity["manifest_sha256_by_id"]), v7_jobs=jobs)
    if problems:
        for line in problems:
            print("  "+line, flush=True)
        raise ValueError(f"preflight 没通过：{len(problems)} 项（见上），补跑停在 A0")
    return dict(schema=SCHEMA, stage=stage, v7_results=str(v7_results), features=str(features), v6_features=str(v6) if v6 else None,
                v7_conditions=list(V7_CONDITIONS), seeds=list(spec["seeds"]), scenes=scenes,
                checked_jobs=sum(len(s["v7_jobs"]) for s in scenes.values()))


def write_once(path: Path, value: dict) -> str:
    """不存在就用 cache._write_json 写；已存在且字节相同就跳过；不同就报错。"""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != cache._json_bytes(value):
            raise ValueError(f"{path} 已存在且内容不同；不覆盖")
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(path.name+".partial").unlink(missing_ok=True)
    cache._write_json(path, value)
    return "written"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v7-results", type=Path, required=True, help="results-v7/<pilot|full>")
    p.add_argument("--features", type=Path, required=True, help="features-v7")
    p.add_argument("--v6-features", type=Path, default=None,
                   help="试跑：v7 试跑用的 v6 double_well-200 特征目录（绝对路径，和 make_v7_jobs.py --v6-features 给的相同）")
    p.add_argument("--stage", choices=tuple(STAGES), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    record = preflight(a.v7_results, a.features, a.stage, a.v6_features)
    state = write_once(a.output, record)
    print(f"preflight 通过：{len(record['scenes'])} 个场景，{record['checked_jobs']} 个 v7 任务；"
          f"{'已写入' if state == 'written' else '已存在且相同'} {a.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
