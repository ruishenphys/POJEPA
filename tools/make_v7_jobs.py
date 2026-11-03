"""生成 v7 要提交的全部命令（每行一个独立任务），方便排队系统逐条提交。

  python tools/make_v7_jobs.py --stage pilot > jobs-pilot.txt
  python tools/make_v7_jobs.py --stage full  > jobs-full.txt

输出分 5 段，段与段之间有依赖（前一段全部完成才能开始下一段），同一段里的命令互不依赖、可以同时跑在不同的卡上：
  A 编码（每个系统按 --shards 切片）  B 汇总编码（CPU，几秒）  C 标定（每个系统 1 个）  C2 检查（1 条）
  D 训练+评测（系统 × 种子 × 条件）  E 汇总（CPU）
每条 GPU 命令都写 --device cuda:0：排队系统给每个任务分一张卡时，用 CUDA_VISIBLE_DEVICES 指定是哪张。
"""
import argparse

SYSTEMS = ("double_well", "van_der_pol", "multi_particle", "advection_diffusion")
CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity", "vjepa2_ac_small", "vjepa2_ac")
SEEDS = (1703, 1704, 1705)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("pilot", "full"), required=True)
    p.add_argument("--shards", type=int, default=4, help="每个系统的编码切成几片（每片一张卡）")
    p.add_argument("--features", default="../features-v7")
    p.add_argument("--results", default="../results-v7")
    p.add_argument("--upstream-root", default="../vjepa2")
    p.add_argument("--encoder", default="../checkpoints/vjepa2-vitg-fpc64-256.pth")
    p.add_argument("--ac-checkpoint", default="../checkpoints/vjepa2-ac-vitg.pt")
    p.add_argument("--v6-features", default="../unpacked-v3/po-jepa/features",
                   help="试跑的 double_well-200 直接用 v6 的特征，不重新编码")
    a = p.parse_args()
    n, updates = (200, 2000) if a.stage == "pilot" else (2000, 10000)
    seeds = SEEDS[:1] if a.stage == "pilot" else SEEDS
    root = f"{a.results}/{a.stage}"

    def feats(system):
        return a.v6_features if (system == "double_well" and n == 200) else f"{a.features}/{system}-{n}"

    encode = [s for s in SYSTEMS if not (s == "double_well" and n == 200)]
    print(f"# ===== A 编码（GPU，{len(encode)*a.shards} 条，互不依赖）=====")
    for s in encode:
        for i in range(a.shards):
            print(f"python tools/encode_v7.py encode --system {s} --trajectories {n} --upstream-root {a.upstream_root} "
                  f"--checkpoint {a.encoder} --output {feats(s)} --device cuda:0 --shard {i}/{a.shards}")
    print(f"# ===== B 汇总编码（CPU，A 段全部完成后）=====")
    for s in encode:
        print(f"python tools/encode_v7.py finalize --system {s} --trajectories {n} --output {feats(s)}")
    print(f"# ===== C 标定（GPU，B 段完成后；每个系统 1 条）=====")
    for s in SYSTEMS:
        print(f"python run.py calibrate --system {s} --trajectories {n} --updates {updates} --features-root {feats(s)} "
              f"--device cuda:0 --output {root}/calibrate-{s}")
    print("# ===== C2 检查（GPU，C 段完成后跑 1 条：依赖、设备、完整梯度、100 步成本；每种 GPU 型号做一次）=====")
    print(f"python run.py check --system {SYSTEMS[1]} --trajectories {n} --updates {updates} --features-root {feats(SYSTEMS[1])} "
          f"--device cuda:0 --output {root}/check")
    print(f"# ===== D 训练+评测（GPU，C 段对应系统完成后；{len(SYSTEMS)*len(seeds)*len(CONDITIONS)} 条，互不依赖）=====")
    for s in SYSTEMS:
        for seed in seeds:
            for c in CONDITIONS:
                extra = f" --ac-checkpoint {a.ac_checkpoint}" if c == "vjepa2_ac" else ""
                print(f"python run.py run --system {s} --trajectories {n} --updates {updates} --seed {seed} --condition {c} "
                      f"--features-root {feats(s)} --device cuda:0 --output {root}/{s}/seed{seed}/{c}{extra}")
    print("# ===== E 汇总（CPU，D 段完成后；缺的任务会列出来，可以先汇总已完成的）=====")
    print(f"python tools/summarize_v7.py --results {root} --output {root}/summary")


if __name__ == "__main__":
    main()
