"""生成 v7 要提交的全部命令（每行一个独立任务），方便排队系统逐条提交。

  python tools/make_v7_jobs.py --stage pilot    > jobs-pilot.txt      # 试跑：每个场景 200 条、2000 步、1 个种子
  python tools/make_v7_jobs.py --stage full     > jobs-full.txt       # 正式：每个场景 2000 条、10000 步、3 个种子
  python tools/make_v7_jobs.py --stage variants > jobs-variants.txt   # 可调数据集：分布外评测 + 规划（在 full 完成后）

输出按 “# =====” 分段，段与段之间有依赖（前一段全部完成才能开始下一段），同一段里的命令互不依赖、可以同时跑在不同的卡上。
每条 GPU 命令都写 --device cuda:0：排队系统给每个任务分一张卡时，用 CUDA_VISIBLE_DEVICES 指定是哪张。
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from po_jepa import variants  # noqa: E402

BASE_SCENES = ("double_well", "van_der_pol", "multi_particle", "advection_diffusion", "landscape~shape=multi_well")
CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity", "vjepa2_ac_small", "vjepa2_ac")
SEEDS = (1703, 1704, 1705)
# 分布外评测用的测试调节项：每个场景 2 个物理参数 + 2 个控制 + 2 个外观（训练都在基础场景上）
TEST_VARIANTS = {
    "double_well": ("coupling=0.5", "tilt=0.2", "gain=2", "delay=1", "background=checker,color=white", "distractors=2"),
    "van_der_pol": ("mu=1", "mu=0.2", "gain=2", "delay=1", "background=checker,color=white", "distractors=2"),
    "multi_particle": ("repulsion=2.5", "trap=0.8", "gain=2", "delay=1", "background=checker,color=white", "distractors=2"),
    "advection_diffusion": ("diffusion=0.001", "flow=0.8", "gain=2", "delay=1", "background=checker,color=white", "pixel_noise=10"),
    "landscape": ("depth=2", "depth=0.5", "gain=2", "delay=1", "background=checker,color=white", "distractors=2"),
}


def split_scene(scene):
    system, _, knobs = scene.partition("~")
    return system, variants.format(system, variants.parse(system, knobs))


def merged(scene, extra):
    system, base = split_scene(scene)
    items = dict(i.split("=", 1) for i in base.split(",") if i)
    items.update(i.split("=", 1) for i in extra.split(",") if i)
    return variants.format(system, variants.parse(system, ",".join(f"{k}={v}" for k, v in items.items())))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("pilot", "full", "variants"), required=True)
    p.add_argument("--scenes", default=",".join(BASE_SCENES),
                   help="逗号分隔的场景（系统[~调节项]）；去掉 landscape 可省约 20%% 算力")
    p.add_argument("--shards", type=int, default=4, help="每个场景的编码切成几片（每片一张卡）")
    p.add_argument("--features", default="../features-v7")
    p.add_argument("--results", default="../results-v7")
    p.add_argument("--upstream-root", default="../vjepa2")
    p.add_argument("--encoder", default="../checkpoints/vjepa2-vitg-fpc64-256.pth")
    p.add_argument("--ac-checkpoint", default="../checkpoints/vjepa2-ac-vitg.pt")
    p.add_argument("--v6-features", default="",
                   help="可选：试跑的 double_well-200 改用已有的 v6 特征目录（不给就和其他场景一样在本机编码）")
    p.add_argument("--plan-episodes", type=int, default=100)
    a = p.parse_args()
    scenes = [s for s in a.scenes.split(",") if s]
    n, updates = (200, 2000) if a.stage == "pilot" else (2000, 10000)
    seeds = SEEDS[:1] if a.stage == "pilot" else SEEDS
    root = f"{a.results}/{'pilot' if a.stage == 'pilot' else 'full'}"
    enc = f"--upstream-root {a.upstream_root} --checkpoint {a.encoder}"

    def name(scene, extra=None):
        system, knobs = split_scene(scene) if extra is None else (split_scene(scene)[0], merged(scene, extra))
        return system if not knobs else f"{system}~{knobs}"

    def feats(scene, extra=None):
        if a.v6_features and scene == "double_well" and n == 200 and extra is None:
            return a.v6_features
        return f"{a.features}/{name(scene, extra)}-{n}"

    def ds(scene):
        system, knobs = split_scene(scene)
        return f"--system {system}" + (f" --variant {knobs}" if knobs else "") + f" --trajectories {n}"

    if a.stage == "variants":
        print(f"# 前提：--stage full 的 A–D 段已完成（模型在基础场景上训练好），本文件只做评测，不再训练。")
        tests = [(s, t) for s in scenes for t in TEST_VARIANTS[split_scene(s)[0]]]
        print(f"# ===== V1 编码变体测试集（GPU，{len(tests)*a.shards} 条；只编码测试段 200 条，编号与基础场景测试段一一对应）=====")
        for s, t in tests:
            system, knobs = split_scene(s)[0], merged(s, t)
            for i in range(a.shards):
                print(f"python tools/encode_v7.py encode --system {system} --variant {knobs} --trajectories {n} --splits test {enc} "
                      f"--output {feats(s, t)} --device cuda:0 --shard {i}/{a.shards}")
        print("# ===== V2 汇总编码（CPU）=====")
        for s, t in tests:
            print(f"python tools/encode_v7.py finalize --system {split_scene(s)[0]} --variant {merged(s, t)} --trajectories {n} "
                  f"--splits test --output {feats(s, t)}")
        print(f"# ===== V3 分布外评测（GPU，{len(tests)*len(CONDITIONS)} 条，种子 1703；只评 16×16 网格单点，较快）=====")
        for s, t in tests:
            for c in CONDITIONS:
                run = "" if c == "identity" else f" --training-run {root}/{name(s)}/seed1703/{c}"
                extra = f" --ac-checkpoint {a.ac_checkpoint}" if c == "vjepa2_ac" else ""
                print(f"python run.py transfer {ds(s)} --updates {updates} --seed 1703 --condition {c} --test-variant {t}{run} "
                      f"--features-root {feats(s, t)} --device cuda:0 --output {root}/transfer/{name(s, t)}/{c}{extra}")
        print(f"# ===== V4 规划（GPU，{len(scenes)*len(CONDITIONS)} 条，种子 1703，每条 {a.plan_episodes} 个回合；"
              "不动/随机/真值规划三个参照只在 identity 那条里跑）=====")
        for s in scenes:
            for c in CONDITIONS:
                extra = (f" --ac-checkpoint {a.ac_checkpoint} --samples 64" if c == "vjepa2_ac" else "") + \
                        ("" if c == "identity" else " --no-baselines")
                print(f"python tools/plan_v7.py --job {root}/{name(s)}/seed1703/{c} {enc} --episodes {a.plan_episodes} "
                      f"--output {root}/plan/{name(s)}/{c}.json --device cuda:0{extra}")
        print("# ===== V5 汇总（CPU）=====")
        print(f"python tools/summarize_v7.py --results {root} --output {root}/summary")
        return

    encode = [s for s in scenes if not (a.v6_features and s == "double_well" and n == 200)]
    print(f"# ===== A 编码（GPU，{len(encode)*a.shards} 条，互不依赖）=====")
    for s in encode:
        for i in range(a.shards):
            print(f"python tools/encode_v7.py encode {ds(s)} {enc} --output {feats(s)} --device cuda:0 --shard {i}/{a.shards}")
    print("# ===== B 汇总编码（CPU，A 段全部完成后）=====")
    for s in encode:
        print(f"python tools/encode_v7.py finalize {ds(s)} --output {feats(s)}")
    if a.stage == "full" and "double_well" in scenes:
        print("# ===== B2 动作探针（GPU，1 条，约 10 分钟）：动作在冻结特征里有多强，回答 v6 的 delta_act≈0 =====")
        print(f"python tools/probe_actions.py --system double_well --trajectories {n} {enc} "
              f"--output {root}/probe-actions-double_well.json --device cuda:0")
    print("# ===== C 标定（GPU，B 段完成后；每个场景 1 条）=====")
    for s in scenes:
        print(f"python run.py calibrate {ds(s)} --updates {updates} --features-root {feats(s)} --device cuda:0 "
              f"--output {root}/calibrate-{name(s)}")
    print("# ===== C2 检查（GPU，C 段完成后跑 1 条：依赖、设备、完整梯度、100 步成本；每种 GPU 型号做一次）=====")
    print(f"python run.py check {ds(scenes[min(1, len(scenes)-1)])} --updates {updates} --features-root {feats(scenes[min(1, len(scenes)-1)])} "
          f"--device cuda:0 --output {root}/check")
    print(f"# ===== D 训练+评测（GPU，C 段对应场景完成后；{len(scenes)*len(seeds)*len(CONDITIONS)} 条，互不依赖）=====")
    for s in scenes:
        for seed in seeds:
            for c in CONDITIONS:
                extra = f" --ac-checkpoint {a.ac_checkpoint}" if c == "vjepa2_ac" else ""
                print(f"python run.py run {ds(s)} --updates {updates} --seed {seed} --condition {c} "
                      f"--features-root {feats(s)} --device cuda:0 --output {root}/{name(s)}/seed{seed}/{c}{extra}")
    print("# ===== E 汇总（CPU，D 段完成后；缺的任务会列出来，可以先汇总已完成的）=====")
    print(f"python tools/summarize_v7.py --results {root} --output {root}/summary")


if __name__ == "__main__":
    main()
