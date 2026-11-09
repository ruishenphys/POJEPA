"""生成 v8 要提交的全部命令（每行一个独立任务），格式和 make_v7_jobs.py 相同：按 “# =====” 分段，段与段之间有依赖，
同一段里的命令互不依赖、可以同时跑在不同的卡上；GPU 命令都写 --device cuda:0（排队系统用 CUDA_VISIBLE_DEVICES 分卡）。

  python tools/make_v8_jobs.py --stage pilot    > jobs-v8-pilot.txt      # 试跑：7 个场景各 200 条、2000 步、1 个种子
  python tools/make_v8_jobs.py --stage full     > jobs-v8-full.txt       # 正式：各 2000 条、10000 步、3 个种子
  python tools/make_v8_jobs.py --stage variants > jobs-v8-variants.txt   # 分布外评测（评测协议的标准档）+ 规划（full 之后）
  python tools/make_v8_jobs.py --stage strength > jobs-v8-strength.txt   # 强度曲线（轻档、重档；variants 之后）

和 v7 的区别（S21b）：
- 场景是 po_jepa.protocols.SCENES：v7 的 5 个加 gray_scott、push。
- 分布外测试表来自评测协议 po_jepa.protocols（P1–P13，三档强度；DATASET.md 有协议表），不是 v7 的 TEST_VARIANTS。
  v7 的 TEST_VARIANTS 和 make_v7_jobs.py 一字不改：v7 正式实验和仿真补跑（make_addon_jobs.py）都在用它。
- 正式阶段的动作探针（B2）每个场景都跑，不只 double_well。
- 默认目录是 ../features-v8、../results-v8，和 v7 分开。同一个场景、同一个调节项的数据在 v7 树和 v8 树上逐位相同
  （S20b、S21a、S21b 的跨树对照），想复用 v7 已编码的特征，可以用 --features 指到 v7 的特征目录。
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from po_jepa import protocols, variants  # noqa: E402

CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity", "vjepa2_ac_small", "vjepa2_ac")
SEEDS = (1703, 1704, 1705)
# 规划的真值参照（oracle）每次 CEM 的样本数：真值仿真在 CPU 上逐个跑，新场景每个 tick 贵得多
# （gray_scott 约 60 ms、push 约 30 ms，粒子类约 3 ms），默认 64 会让 identity 那条规划任务占着 GPU 跑几个小时（S21b 实测）
ORACLE_SAMPLES = {"gray_scott": 16, "push": 32}


def split_scene(scene):
    system, _, knobs = scene.partition("~")
    return system, variants.format(system, variants.parse(system, knobs))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("pilot", "full", "variants", "strength"), required=True)
    p.add_argument("--scenes", default=",".join(protocols.SCENES), help="逗号分隔的场景（系统[~调节项]）")
    p.add_argument("--protocols", default="", help="只要这些协议，逗号分隔，例如 P1,P4（默认全部）")
    p.add_argument("--conditions", default=",".join(CONDITIONS), help="分布外评测和规划评哪些条件")
    p.add_argument("--shards", type=int, default=4, help="每个场景的编码切成几片（每片一张卡）")
    p.add_argument("--features", default="../features-v8")
    p.add_argument("--results", default="../results-v8")
    p.add_argument("--upstream-root", default="../vjepa2")
    p.add_argument("--encoder", default="../checkpoints/vjepa2-vitg-fpc64-256.pth")
    p.add_argument("--ac-checkpoint", default="../checkpoints/vjepa2-ac-vitg.pt")
    p.add_argument("--plan-episodes", type=int, default=100)
    a = p.parse_args()
    scenes = [s for s in a.scenes.split(",") if s]
    for s in scenes:
        if s not in protocols.SCENES:
            raise SystemExit(f"{s} 不是 v8 的基础场景：{protocols.SCENES}")
    ids = [i for i in a.protocols.split(",") if i] or None
    if ids and not set(ids) <= set(protocols.BY_ID):
        raise SystemExit(f"没有这些协议：{sorted(set(ids)-set(protocols.BY_ID))}")
    conditions = [c for c in a.conditions.split(",") if c]
    if not set(conditions) <= set(CONDITIONS):
        raise SystemExit(f"没有这些条件：{sorted(set(conditions)-set(CONDITIONS))}")
    n, updates = (200, 2000) if a.stage == "pilot" else (2000, 10000)
    seeds = SEEDS[:1] if a.stage == "pilot" else SEEDS
    root = f"{a.results}/{'pilot' if a.stage == 'pilot' else 'full'}"
    enc = f"--upstream-root {a.upstream_root} --checkpoint {a.encoder}"

    def name(scene, extra=None):
        system, knobs = split_scene(scene) if extra is None else (split_scene(scene)[0], protocols.merged(scene, extra))
        return system if not knobs else f"{system}~{knobs}"

    def feats(scene, extra=None):
        return f"{a.features}/{name(scene, extra)}-{n}"

    def ds(scene):
        system, knobs = split_scene(scene)
        return f"--system {system}" + (f" --variant {knobs}" if knobs else "") + f" --trajectories {n}"

    if a.stage in ("variants", "strength"):
        levels = ("标准",) if a.stage == "variants" else ("轻", "重")
        print("# 前提：--stage full 的 A–D 段已完成（模型在基础场景上训练好）" +
              ("" if a.stage == "variants" else "；强度曲线和标准档共用训练好的模型") + "，本文件只做评测，不再训练。")
        print(f"# 评测协议：po_jepa.protocols（DATASET.md 的协议表），档位 {'、'.join(levels)}"
              + (f"，协议 {','.join(ids)}" if ids else "") + "。")
        tests = [(s, pid, level, t) for s in scenes for pid, level, t in protocols.tests(s, levels, ids)]
        print(f"# ===== V1 编码变体测试集（GPU，{len(tests)*a.shards} 条；只编码测试段 200 条，编号与基础场景测试段一一对应）=====")
        for s, pid, level, t in tests:
            for i in range(a.shards):
                print(f"python tools/encode_v7.py encode --system {split_scene(s)[0]} --variant {protocols.merged(s, t)} "
                      f"--trajectories {n} --splits test {enc} --output {feats(s, t)} --device cuda:0 --shard {i}/{a.shards}")
        print("# ===== V2 汇总编码（CPU）=====")
        for s, pid, level, t in tests:
            print(f"python tools/encode_v7.py finalize --system {split_scene(s)[0]} --variant {protocols.merged(s, t)} "
                  f"--trajectories {n} --splits test --output {feats(s, t)}")
        print(f"# ===== V3 分布外评测（GPU，{len(tests)*len(conditions)} 条，种子 1703；只评 16×16 网格单点）=====")
        for s, pid, level, t in tests:
            for c in conditions:
                run = "" if c == "identity" else f" --training-run {root}/{name(s)}/seed1703/{c}"
                extra = f" --ac-checkpoint {a.ac_checkpoint}" if c == "vjepa2_ac" else ""
                print(f"python run.py transfer {ds(s)} --updates {updates} --seed 1703 --condition {c} --test-variant {t}{run} "
                      f"--features-root {feats(s, t)} --device cuda:0 --output {root}/transfer/{name(s, t)}/{c}{extra}")
        if a.stage == "variants":
            print(f"# ===== V4 规划（GPU，{len(scenes)*len(conditions)} 条，种子 1703，每条 {a.plan_episodes} 个回合，基础场景 = P0；"
                  "不动/随机/真值规划三个参照只在 identity 那条里跑）=====")
            for s in scenes:
                for c in conditions:
                    extra = (f" --ac-checkpoint {a.ac_checkpoint} --samples 64" if c == "vjepa2_ac" else "")
                    if c != "identity":
                        extra += " --no-baselines"
                    elif split_scene(s)[0] in ORACLE_SAMPLES:
                        extra += f" --oracle-samples {ORACLE_SAMPLES[split_scene(s)[0]]}"
                    print(f"python tools/plan_v7.py --job {root}/{name(s)}/seed1703/{c} {enc} --episodes {a.plan_episodes} "
                          f"--output {root}/plan/{name(s)}/{c}.json --device cuda:0{extra}")
        print(f"# ===== V5 汇总（CPU）=====")
        print(f"python tools/summarize_v7.py --results {root} --output {root}/summary")
        return

    print(f"# ===== A 编码（GPU，{len(scenes)*a.shards} 条，互不依赖）=====")
    for s in scenes:
        for i in range(a.shards):
            print(f"python tools/encode_v7.py encode {ds(s)} {enc} --output {feats(s)} --device cuda:0 --shard {i}/{a.shards}")
    print("# ===== B 汇总编码（CPU，A 段全部完成后）=====")
    for s in scenes:
        print(f"python tools/encode_v7.py finalize {ds(s)} --output {feats(s)}")
    if a.stage == "full":
        print(f"# ===== B2 动作探针（GPU，{len(scenes)} 条，每条约 10 分钟）：动作在冻结特征里有多强（反事实上限 vs identity）=====")
        for s in scenes:
            system, knobs = split_scene(s)
            print(f"python tools/probe_actions.py --system {system}" + (f" --variant {knobs}" if knobs else "") +
                  f" --trajectories {n} {enc} --output {root}/probe-actions-{name(s)}.json --device cuda:0")
    print("# ===== C 标定（GPU，B 段完成后；每个场景 1 条）=====")
    for s in scenes:
        print(f"python run.py calibrate {ds(s)} --updates {updates} --features-root {feats(s)} --device cuda:0 "
              f"--output {root}/calibrate-{name(s)}")
    probe = scenes[min(1, len(scenes)-1)]
    print("# ===== C2 检查（GPU，C 段完成后跑 1 条：依赖、设备、完整梯度、100 步成本；每种 GPU 型号做一次）=====")
    print(f"python run.py check {ds(probe)} --updates {updates} --features-root {feats(probe)} --device cuda:0 --output {root}/check")
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
