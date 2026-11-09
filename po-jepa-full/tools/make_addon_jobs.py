"""仿真补跑的任务文件（16 §7）：每行一个独立任务，按 “# =====” 分段，格式和 make_v7_jobs.py 相同。

  python tools/make_addon_jobs.py --stage pilot --v6-features "$V6" > jobs-addon-pilot.txt
                                                                               # 试跑：5 个场景 × 200 条、2000 步、种子 1703
  python tools/make_addon_jobs.py --stage full     > jobs-addon-full.txt       # 正式：5 个场景 × 2000 条、10000 步、3 个种子
  python tools/make_addon_jobs.py --stage variants > jobs-addon-variants.txt   # 新条件的分布外评测和规划（full 完成后）
  python tools/make_addon_jobs.py --stage v7tree   > jobs-addon-v7tree.txt     # D3：在 v7 目录里用 v7 的 run_jobs.py 跑

前三个文件在 工作目录/full-model/po-jepa-full 里用本目录的 tools/run_jobs.py 跑，路径相对于这个目录；v7tree 的路径
相对于 工作目录/po-jepa（16 §8）。段与段之间有依赖（前一段全部完成才开始下一段），同一段里的命令互不依赖。
试跑的 double_well-200 读 --v6-features 给的目录：v7 里 200 条 double_well 走 v6 的缓存核对，只认 v6 编出的特征，所以 v7 试跑
是用 make_v7_jobs.py --stage pilot --v6-features "$V6" 生成的（Jerry 2026-11-08 定；V6 是 v6 特征目录的绝对路径），补跑必须
读同一个目录。语义和 v7 的同名选项相同（只换 double_well、200 条、没有调节项的特征目录）；试跑必须给，其他阶段不能给。
git 克隆里的 po-jepa/features/ 只有元数据、没有 raw_z.npy，不是这个目录。
"""
import argparse
import shlex
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_v7_jobs import BASE_SCENES, CONDITIONS as V7_CONDITIONS, SEEDS, TEST_VARIANTS, merged, split_scene  # noqa: E402

# 和 po_jepa.runtime.addon.conditions.ADDON_CONDITIONS 相同（测试核对）；这里不导入 torch
ADDON_CONDITIONS = ("po_jepa_time_only", "po_jepa_space_only", "transformer_consistency", "fixed_fourier", "koopman")
STAGES = dict(pilot=dict(scenes=BASE_SCENES, n=200, updates=2000, seeds=SEEDS[:1]),
              full=dict(scenes=BASE_SCENES, n=2000, updates=10000, seeds=SEEDS))
CHECK_SCENE = "van_der_pol"
FEATURES, RESULTS_V7, RESULTS = "../../features-v7", "../../results-v7", "../../results-addon"
ENCODER, AC = "../../checkpoints/vjepa2-vitg-fpc64-256.pth", "../../checkpoints/vjepa2-ac-vitg.pt"
UPSTREAM, V7_TREE = "../../vjepa2", "../../po-jepa"
V6_REQUIRED = ("试跑必须给 --v6-features：v7 试跑的 double_well-200 读的是 v6 的特征目录，补跑要读同一个目录。给 v7 试跑用的"
               "同一个绝对路径（v7 交接说明里 V6= 那一行；也就是 po-jepa/jobs-pilot.txt 里 double_well 那几行的 --features-root），"
               "写成 --v6-features \"$V6\"；git 克隆里的 po-jepa/features/ 只有元数据，不是这个目录")


def name(scene, extra=None):
    """目录名：系统[~调节项]，和 make_v7_jobs.py 的相同（变体测试集的调节项叠加在基础场景的调节项上）。"""
    system, knobs = split_scene(scene) if extra is None else (split_scene(scene)[0], merged(scene, extra))
    return system if not knobs else f"{system}~{knobs}"


def uses_v6(scene, n, v6_features) -> bool:
    """和 make_v7_jobs.py 的 feats 相同的条件：给了 --v6-features 时，double_well、200 条、没有调节项的场景读那个目录。"""
    return bool(v6_features) and scene == "double_well" and n == 200


def ds(scene, n):
    system, knobs = split_scene(scene)
    return f"--system {system}" + (f" --variant {knobs}" if knobs else "") + f" --trajectories {n}"


def main_lines(stage, v6_features=""):
    """pilot / full：A0 → C → C2 → D → D4 → E（16 §7.1、§7.4）。"""
    spec = STAGES[stage]
    scenes, n, updates, seeds = spec["scenes"], spec["n"], spec["updates"], spec["seeds"]
    v7, root = f"{RESULTS_V7}/{stage}", f"{RESULTS}/{stage}"
    common = f"--updates {updates}"

    def feats(scene):
        return shlex.quote(v6_features) if uses_v6(scene, n, v6_features) else f"{FEATURES}/{name(scene)}-{n}"

    lines = ["# ===== A0 准备（CPU，3 条）",
             f"python tools/import_v7_calibration.py --v7-tree {V7_TREE} --stage {stage}",
             f"python tools/encoder_identity.py --encoder {ENCODER} --ac-checkpoint {AC} --output {root}/encoder-identity.json",
             f"python tools/addon_preflight.py --v7-results {v7} --features {FEATURES}"
             + (f" --v6-features {shlex.quote(v6_features)}" if v6_features else "") + f" --stage {stage} --output {root}/preflight.json",
             f"# ===== C 补跑标定（GPU，每个场景 1 条，共 {len(scenes)} 条）"]
    for s in scenes:
        lines.append(f"python run.py calibrate-addon {ds(s, n)} {common} --features-root {feats(s)} --device cuda:0 "
                     f"--output {root}/calibrate-{name(s)}")
    lines += ["# ===== C2 检查（GPU，1 条）",
              f"python run.py check-addon {ds(CHECK_SCENE, n)} {common} --features-root {feats(CHECK_SCENE)} --device cuda:0 "
              f"--output {root}/check",
              f"# ===== D 训练+评测（GPU，{len(scenes)*len(seeds)*len(ADDON_CONDITIONS)} 条）"]
    for s in scenes:
        for seed in seeds:
            for c in ADDON_CONDITIONS:
                lines.append(f"python run.py run {ds(s, n)} {common} --seed {seed} --condition {c} --features-root {feats(s)} "
                             f"--device cuda:0 --output {root}/{name(s)}/seed{seed}/{c}")
    every = (*V7_CONDITIONS, *ADDON_CONDITIONS)
    lines.append(f"# ===== D4 只评测（GPU {len(scenes)*len(seeds)*len(every)} 条 + CPU {len(scenes)} 条）")
    for s in scenes:
        for seed in seeds:
            for c in every:
                train = f"{v7 if c in V7_CONDITIONS else root}/{name(s)}/seed{seed}/{c}"
                extra = f" --ac-checkpoint {AC}" if c == "vjepa2_ac" else ""
                lines.append(f"python run.py d4 {ds(s, n)} {common} --seed {seed} --condition {c} --training-run {train} "
                             f"--features-root {feats(s)} --device cuda:0 --output {root}/d4/{name(s)}/seed{seed}/{c}{extra}")
    for s in scenes:
        lines.append(f"python run.py spectral-gap {ds(s, n)} {common} --features-root {feats(s)} --device cpu "
                     f"--output {root}/spectral-gap/{name(s)}")
    lines += ["# ===== E 汇总（CPU，1 条）",
              f"python tools/summarize_addon.py --v7-results {v7} --addon-results {root} --output {root}/summary"]
    return lines


def variant_lines():
    """新条件的分布外评测和规划（16 §7.2）；变体表和 v7 的 TEST_VARIANTS 相同，变体特征由 v7 的 jobs-variants.txt 编码。"""
    spec = STAGES["full"]
    n, updates = spec["n"], spec["updates"]
    v7, root = f"{RESULTS_V7}/full", f"{RESULTS}/full"
    tests = [(s, t) for s in BASE_SCENES for t in TEST_VARIANTS[split_scene(s)[0]]]
    lines = ["# 前提：jobs-addon-full.txt 的 D 段和 v7 的 jobs-variants.txt 都已跑完。",
             f"# ===== V3 变体评测（GPU，{len(tests)*len(ADDON_CONDITIONS)} 条，种子 1703，只评 16×16 单点）"]
    for s, t in tests:
        for c in ADDON_CONDITIONS:
            lines.append(f"python run.py transfer {ds(s, n)} --updates {updates} --seed 1703 --condition {c} --test-variant {t} "
                         f"--training-run {root}/{name(s)}/seed1703/{c} --features-root {FEATURES}/{name(s, t)}-{n} "
                         f"--device cuda:0 --output {root}/transfer/{name(s, t)}/{c}")
    lines.append(f"# ===== V4 规划（GPU，{len(BASE_SCENES)*len(ADDON_CONDITIONS)} 条，种子 1703，每条 100 个回合）")
    for s in BASE_SCENES:
        for c in ADDON_CONDITIONS:
            lines.append(f"python tools/plan_v7.py --job {root}/{name(s)}/seed1703/{c} --upstream-root {UPSTREAM} "
                         f"--checkpoint {ENCODER} --episodes 100 --output {root}/plan/{name(s)}/{c}.json --device cuda:0 --no-baselines")
    lines += ["# ===== V5 汇总（CPU，1 条）",
              f"python tools/summarize_addon.py --v7-results {v7} --addon-results {root} --output {root}/summary"]
    return lines


def v7tree_lines():
    """D3（16 §6、§7.3）：v7 目录里的 plan_v7.py 原样，官方规模 AC 用 256 个样本重跑；路径相对于 工作目录/po-jepa。"""
    lines = ["# 在 工作目录/po-jepa（v7 目录）里用 v7 自己的 tools/run_jobs.py 跑；路径相对于那个目录。",
             f"# ===== D3 官方规模 AC 规划，256 个样本（GPU，{len(BASE_SCENES)} 条，种子 1703）"]
    for s in BASE_SCENES:
        lines.append(f"python tools/plan_v7.py --job ../results-v7/full/{name(s)}/seed1703/vjepa2_ac --upstream-root ../vjepa2 "
                     f"--checkpoint ../checkpoints/vjepa2-vitg-fpc64-256.pth --episodes 100 "
                     f"--output ../results-addon/full/plan-ac256/{name(s)}/vjepa2_ac.json --device cuda:0 "
                     f"--ac-checkpoint ../checkpoints/vjepa2-ac-vitg.pt --samples 256 --no-baselines")
    return lines


def job_lines(stage, v6_features=""):
    if stage == "pilot" and not v6_features:
        raise ValueError(V6_REQUIRED)
    if stage != "pilot" and v6_features:
        raise ValueError("--v6-features 只用于 --stage pilot（只有试跑的 double_well 是 200 条）")
    if v6_features and not Path(v6_features).is_absolute():
        raise ValueError(f"--v6-features 要写绝对路径（和 v7 试跑用的相同），不是 {v6_features}")
    if stage in STAGES:
        return main_lines(stage, v6_features)
    return variant_lines() if stage == "variants" else v7tree_lines()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("pilot", "full", "variants", "v7tree"), required=True)
    p.add_argument("--v6-features", default="",
                   help="试跑必须给：v7 试跑用的 v6 double_well-200 特征目录，和 make_v7_jobs.py --v6-features 给的是同一个绝对路径"
                        "（v7 交接说明的 V6=）；git 克隆里的 po-jepa/features/ 只有元数据，不是它")
    a = p.parse_args(argv)
    try:
        lines = job_lines(a.stage, a.v6_features)
    except ValueError as error:
        p.error(str(error))
    sys.stdout.write("\n".join(lines)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
