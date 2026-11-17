"""小规模验证（方案 `full-model-next-trial` §5，2026-11-17）的任务文件：每行一个独立任务，按 “# =====” 分段，格式和
make_v7_jobs.py 相同，在 工作目录/full-model/po-jepa-full 里用 tools/run_jobs.py 跑，路径相对于这个目录。

  python tools/make_small_validation_jobs.py --stage b-v7.1 > jobs-sv-b-v7.1.txt
      # B 层诊断（只评测）：在 v7.1 正式运行的检查点上读物理量（冻结线性探针 + 长时程 VPT）、反事实动作、线性场景模态对照
  python tools/make_small_validation_jobs.py --stage c1 --v6-features "$V6" > jobs-sv-c1.txt
      # C-1 筛选：PO-JEPA 的 2×2（二等分/多段切分 × Xavier/零初始化），3 个场景 × 200 条 × 2000 步，种子 1703，共 12 条训练
  python tools/make_small_validation_jobs.py --stage c2 --temporal-partition <P> --readout-init <I> > jobs-sv-c2.txt
      # C-2 对照：C-1 选定的变体，3 个场景 × 2000 条 × 10000 步，种子 1703；po_jepa_full、po_jepa_no_consistency、
      # transformer_consistency、fixed_fourier、po_jepa_rollout_sup 共 15 条训练，加 3 条 identity（逐位核对，决定能不能复用 v7.1）

C-2 的变体必须显式给：由 C-1 的汇总（tools/summarize_small_validation.py c1）按预先写好的规则推荐，Jerry 可以改。
三个阶段的先后：b-v7.1 和 c1 互不依赖，可以同时排；c2 在 c1 之后，它的 R 段（新检查点的读出）要用 b-v7.1 的探针和真值。

共同的约定：
- 场景 advection_diffusion、double_well、van_der_pol（方案 §5.3：v7.1 里区分力最大的 3 个），种子 1703。
- 特征复用 v7 树编好的 `../../features-v7/<场景>-<条数>`（不重编码）；C-1 的 double_well-200 和 v7 试跑一样读 v6 的特征目录
  （--v6-features，绝对路径；make_addon_jobs.py 的 V6_REQUIRED 说明）。长时程测试集（horizon=128，只编码测试段）是新编的，
  写在 `../../features-small-validation/`。
- 结果写 `../../results-small-validation/`：b-v7.1/、readout/（探针、真值、动作探针：和模型无关，C-2 复用）、c1/<变体>/、c2/<变体>/。
  变体目录名是 <切分>-<初始化>（bisection-xavier、dyadic-zero 等），每个变体一个结果根，summarize_v7 / summarize_addon 不会把两个变体混在一组。
- 变体用 run.py 的 --temporal-partition / --readout-init（S28c），任务行里总是两个都写，不依赖外面的环境变量。
- 每个非默认变体先 run.py calibrate 重新标定 λ（S28b：标定文件按变体分，`<场景>-<条数>-u<步数>~dyadic~zeroinit.yaml`，没标定就拒跑）；
  默认变体（bisection-xavier）用 git 里 v7.1 的标定（configs/calibration/，损失定义没变）。C-2 的补跑条件（transformer_consistency、
  fixed_fourier、po_jepa_rollout_sup）用 run.py calibrate-addon --base-calibration local 在同一变体下标定。
- 这个任务表和 make_addon_jobs.py（v7 场景补跑，导入 v7 标定）不能共用一份工作树（base-calibration 两种来源互斥，S26a）；
  用一份新 clone 的 full-model 跑。
"""
import argparse
import shlex
from pathlib import Path
import sys

SCENES = ("advection_diffusion", "double_well", "van_der_pol")
SEED = 1703
SEEDS_V71 = (1703, 1704, 1705)
PARTITIONS = ("bisection", "dyadic")
INITS = ("xavier", "zero")
DEFAULT_VARIANT = ("bisection", "xavier")
VARIANTS = tuple((p, i) for p in PARTITIONS for i in INITS)
C1 = dict(n=200, updates=2000)
C2 = dict(n=2000, updates=10000)
# v7.1 正式运行的 6 个条件（B 层读它们的检查点）；C-2 训练的 5 个条件（方案 §5.3）和数值自检 identity
V71_CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity", "vjepa2_ac_small", "vjepa2_ac")
C2_CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "transformer_consistency", "fixed_fourier", "po_jepa_rollout_sup")
# C-2 不重训、复用 v7.1 种子 1703 的条件（前提：identity 逐位相同，summarize_small_validation.py c2 核）
REUSED_FROM_V71 = ("matched_transformer", "vjepa2_ac_small", "vjepa2_ac")
EQUAL_STEP = ("po_jepa_no_consistency", "transformer_consistency", "fixed_fourier", "po_jepa_rollout_sup")
MODAL_SCENES = ("advection_diffusion",)   # runtime/modal_spectrum.LINEAR_SCENES 里在 SCENES 中的
LONG_HORIZON = 128
CHECK_SCENE = "van_der_pol"
FEATURES, LONG_FEATURES = "../../features-v7", "../../features-small-validation"
RESULTS_V71, OUT = "../../results-v7/full/v7.1", "../../results-small-validation"
ENCODER, AC, UPSTREAM = "../../checkpoints/vjepa2-vitg-fpc64-256.pth", "../../checkpoints/vjepa2-ac-vitg.pt", "../../vjepa2"
V6_REQUIRED = ("c1 必须给 --v6-features：v7 里 200 条的 double_well 读的是 v6 的特征目录（v7 试跑同样），给 v7 试跑用的同一个"
               "绝对路径（v7 交接说明里 V6= 那一行）")


def variant_name(partition, init) -> str:
    return f"{partition}-{init}"


def variant_flags(partition, init) -> str:
    return f"--temporal-partition {partition} --readout-init {init}"


def ds(scene, n) -> str:
    return f"--system {scene} --trajectories {n}"


def long_name(scene) -> str:
    return f"{scene}~horizon={LONG_HORIZON}"


def feats(scene, n, v6_features="") -> str:
    if v6_features and scene == "double_well" and n == 200:
        return shlex.quote(v6_features)
    return f"{FEATURES}/{scene}-{n}"


def long_feats(scene) -> str:
    return f"{LONG_FEATURES}/{long_name(scene)}-{C2['n']}"


def ac_flag(condition) -> str:
    return f" --ac-checkpoint {AC}" if condition == "vjepa2_ac" else ""


def readout_lines(job_root, out, conditions, seeds, *, cross_version, scenes=SCENES):
    """R 段：长时程读出（VPT）和反事实（每个 场景×种子×条件 各 1 条），线性场景加模态对照（只 PO-JEPA 条件）。
    探针和真值在 {OUT}/readout/ 下（b-v7.1 段写，C-2 复用）。cross_version：任务目录是别的代码版本训练的（v7.1）。"""
    n = C2["n"]
    flag = " --cross-version" if cross_version else ""
    lines = [f"# ===== R3 长时程读出、反事实（GPU，{2*len(scenes)*len(seeds)*len(conditions)} 条，互不依赖）====="]
    for s in scenes:
        for seed in seeds:
            for c in conditions:
                common = (f"--job {job_root}/{s}/seed{seed}/{c} --probe {OUT}/readout/probe/{s} "
                          f"--truth {OUT}/readout/truth/{s}.json")
                base = f" --base-features {feats(s, n)}" if cross_version else ""
                lines.append(f"python tools/state_readout.py rollout {common} --features {long_feats(s)} --horizon {LONG_HORIZON}"
                             f"{base}{flag} --output {out}/rollout/{s}/seed{seed}/{c}.json --device cuda:0{ac_flag(c)}")
                lines.append(f"python tools/state_readout.py counterfactual {common} --features {feats(s, n)}{flag} "
                             f"--output {out}/counterfactual/{s}/seed{seed}/{c}.json --device cuda:0{ac_flag(c)}")
    modal = [s for s in scenes if s in MODAL_SCENES]
    po = [c for c in conditions if c.startswith("po_jepa")]
    lines.append(f"# ===== R4 模态频率对照（GPU，{len(modal)*len(seeds)*len(po)} 条；只有线性场景、PO-JEPA 条件；探索性）=====")
    for s in modal:
        for seed in seeds:
            for c in po:
                lines.append(f"python tools/modal_spectrum.py --job {job_root}/{s}/seed{seed}/{c} --features {feats(s, n)}{flag} "
                             f"--output {out}/modal/{s}/seed{seed}/{c}.json --device cuda:0")
    return lines


def b_lines(shards):
    """B 层（方案 §5.2 第 1–3 条）在 v7.1 正式运行的检查点上：只评测，不训练。"""
    n = C2["n"]
    enc = f"--upstream-root {UPSTREAM} --checkpoint {ENCODER}"
    out = f"{OUT}/b-v7.1"
    lines = ["# 前提：v7.1 正式运行（v7 分支）的任务目录在 ../../results-v7/full/v7.1/，检查点还在；特征在 ../../features-v7/。",
             "# 检查点是 v7 分支的代码训练的，读出和模态工具用 --cross-version（addon.v7load，核数据来源和配置快照，不核源码 SHA）。",
             f"# ===== L1 编码长时程测试集（GPU，{len(SCENES)*shards} 条；horizon={LONG_HORIZON}，只编码测试段 200 条 × {LONG_HORIZON+1} 帧）====="]
    for s in SCENES:
        for i in range(shards):
            lines.append(f"python tools/encode_v7.py encode --system {s} --variant horizon={LONG_HORIZON} --trajectories {n} --splits test "
                         f"{enc} --output {long_feats(s)} --device cuda:0 --shard {i}/{shards}")
    lines.append("# ===== L2 汇总编码（CPU）=====")
    for s in SCENES:
        lines.append(f"python tools/encode_v7.py finalize --system {s} --variant horizon={LONG_HORIZON} --trajectories {n} --splits test "
                     f"--output {long_feats(s)}")
    lines.append(f"# ===== R0 动作探针（GPU，{len(SCENES)} 条，每条约 10 分钟）：换动作后真实终点特征和画面变多少（动作信号上限）=====")
    for s in SCENES:
        lines.append(f"python tools/probe_actions.py {ds(s, n)} {enc} --features-root {feats(s, n)} "
                     f"--output {OUT}/readout/probe-actions-{s}.json --device cuda:0")
    lines.append(f"# ===== R1 拟合探针（GPU，{len(SCENES)} 条；主实验缓存的验证段，full 读出约 10 GB 内存）=====")
    for s in SCENES:
        lines.append(f"python tools/state_readout.py fit {ds(s, n)} --features {feats(s, n)} --output {OUT}/readout/probe/{s} --device cuda:0")
    lines.append(f"# ===== R2 真值（CPU，{len(SCENES)} 条；反事实重新仿真）=====")
    for s in SCENES:
        lines.append(f"python tools/state_readout.py truth {ds(s, n)} --features {feats(s, n)} --long-features {long_feats(s)} "
                     f"--horizon {LONG_HORIZON} --output {OUT}/readout/truth/{s}.json")
    lines += readout_lines(RESULTS_V71, out, V71_CONDITIONS, SEEDS_V71, cross_version=True)
    lines += ["# ===== R5 汇总（CPU，1 条）=====",
              f"python tools/summarize_small_validation.py readout --results {out} --actions {OUT}/readout --output {out}/summary"]
    return lines


def c1_lines(v6_features):
    """C-1 筛选（方案 §5.3）：只训 po_jepa_full，4 个变体 × 3 个场景。"""
    n, updates = C1["n"], C1["updates"]
    root = f"{OUT}/c1"
    others = [v for v in VARIANTS if v != DEFAULT_VARIANT]
    lines = ["# 前提：v7 试跑的 200 条特征在 ../../features-v7/<场景>-200（double_well-200 读 --v6-features）。",
             f"# 默认变体 {variant_name(*DEFAULT_VARIANT)} 用 git 里 v7 试跑的标定（configs/calibration/<场景>-200-u2000.yaml），不重标。",
             f"# ===== C 标定（GPU，{len(others)*len(SCENES)} 条；每个非默认变体、每个场景 1 条）====="]
    for v in others:
        for s in SCENES:
            lines.append(f"python run.py calibrate {ds(s, n)} --updates {updates} {variant_flags(*v)} --features-root {feats(s, n, v6_features)} "
                         f"--device cuda:0 --output {root}/calibrate/{variant_name(*v)}/{s}")
    lines.append(f"# ===== C2 检查（GPU，{len(others)} 条；每个非默认变体在 {CHECK_SCENE} 上 1 条：依赖、设备、完整梯度、100 步成本）=====")
    for v in others:
        lines.append(f"python run.py check {ds(CHECK_SCENE, n)} --updates {updates} {variant_flags(*v)} "
                     f"--features-root {feats(CHECK_SCENE, n, v6_features)} --device cuda:0 --output {root}/check/{variant_name(*v)}")
    lines.append(f"# ===== D 训练+评测（GPU，{len(VARIANTS)*len(SCENES)} 条，互不依赖；只训 po_jepa_full）=====")
    for v in VARIANTS:
        for s in SCENES:
            lines.append(f"python run.py run {ds(s, n)} --updates {updates} --seed {SEED} --condition po_jepa_full {variant_flags(*v)} "
                         f"--features-root {feats(s, n, v6_features)} --device cuda:0 --output {root}/{variant_name(*v)}/{s}/seed{SEED}/po_jepa_full")
    lines.append(f"# ===== E 汇总（CPU，{len(VARIANTS)+1} 条）：每个变体一张 v7 汇总表，再出 2×2 对照和 C-2 变体的推荐 =====")
    for v in VARIANTS:
        lines.append(f"python tools/summarize_v7.py --results {root}/{variant_name(*v)} --output {root}/{variant_name(*v)}/summary")
    lines.append(f"python tools/summarize_small_validation.py c1 --results {root} --output {root}/summary")
    return lines


def c2_lines(partition, init):
    """C-2 对照（方案 §5.3）：C-1 定下的变体，5 个条件 + identity，3 个场景。"""
    n, updates = C2["n"], C2["updates"]
    v = (partition, init)
    name = variant_name(*v)
    root = f"{OUT}/c2/{name}"
    flags = variant_flags(*v)
    lines = [f"# 变体 {name}（C-1 汇总推荐，或 Jerry 指定）。前提：b-v7.1 的 R1、R2 已完成（探针和真值），L1–L2 的长时程测试集已编好。",
             f"# matched_transformer、vjepa2_ac_small、vjepa2_ac 不重训，复用 v7.1 种子 {SEED}（E 段先核 identity 和 v7.1 逐位相同）。"]
    if partition == "dyadic":
        lines.append("# 注意：po_jepa_rollout_sup 在多段切分变体下仍是有真值的两段滚动（S28b 固定二等分）；要不要改成 2^k 段等 Jerry 定，"
                     "定了要改代码的话，先改代码再跑带 rollout_sup 的行（AC 标定也含 rollout_sup 的 λ_r）。")
    if v != DEFAULT_VARIANT:
        lines.append(f"# ===== C 标定（GPU，{len(SCENES)} 条；本变体的 σ_z、λ_t、λ_s）=====")
        for s in SCENES:
            lines.append(f"python run.py calibrate {ds(s, n)} --updates {updates} {flags} --features-root {feats(s, n)} "
                         f"--device cuda:0 --output {root}/calibrate/{s}")
    else:
        lines.append("# 默认变体：用 git 里 v7.1 的标定（configs/calibration/<场景>-2000-u10000.yaml），不重标。")
    lines.append(f"# ===== AC 补跑标定（GPU，{len(SCENES)} 条；transformer_consistency、fixed_fourier 的 λ，po_jepa_rollout_sup 的 λ_r）=====")
    for s in SCENES:
        lines.append(f"python run.py calibrate-addon {ds(s, n)} --updates {updates} {flags} --features-root {feats(s, n)} "
                     f"--base-calibration local --device cuda:0 --output {root}/calibrate-addon/{s}")
    lines += [f"# ===== C2 检查（GPU，2 条；{CHECK_SCENE}）=====",
              f"python run.py check {ds(CHECK_SCENE, n)} --updates {updates} {flags} --features-root {feats(CHECK_SCENE, n)} "
              f"--device cuda:0 --output {root}/check",
              f"python run.py check-addon {ds(CHECK_SCENE, n)} --updates {updates} {flags} --features-root {feats(CHECK_SCENE, n)} "
              f"--base-calibration local --device cuda:0 --output {root}/check-addon"]
    conditions = (*C2_CONDITIONS, "identity")
    lines.append(f"# ===== D 训练+评测（GPU，{len(SCENES)*len(conditions)} 条，互不依赖；identity 不训练，只作逐位核对）=====")
    for s in SCENES:
        for c in conditions:
            lines.append(f"python run.py run {ds(s, n)} --updates {updates} --seed {SEED} --condition {c} {flags} "
                         f"--features-root {feats(s, n)} --device cuda:0 --output {root}/{s}/seed{SEED}/{c}")
    lines.append(f"# ===== D2 同步数对照（GPU，{len(SCENES)*len(EQUAL_STEP)} 条，只评测）：截到 po_jepa_full 实际训练的步数 =====")
    for s in SCENES:
        for c in EQUAL_STEP:
            lines.append(f"python run.py equal-step {ds(s, n)} --updates {updates} --seed {SEED} --condition {c} {flags} "
                         f"--training-run {root}/{s}/seed{SEED}/{c} --limit-run {root}/{s}/seed{SEED}/po_jepa_full "
                         f"--features-root {feats(s, n)} --device cuda:0 --output {root}/equal-step/{s}/seed{SEED}/{c}")
    lines += readout_lines(root, f"{root}/readout", C2_CONDITIONS, (SEED,), cross_version=False)
    lines += ["# ===== E 汇总（CPU，2 条）：本变体的 v7 汇总表；再和 v7.1（复用的 TF、AC，B 层底数）合起来出方案 §5.4 的验收表 =====",
              f"python tools/summarize_v7.py --results {root} --output {root}/summary --addon-conditions "
              f"{','.join(c for c in C2_CONDITIONS if c not in ('po_jepa_full', 'po_jepa_no_consistency'))}",
              f"python tools/summarize_small_validation.py c2 --results {root} --v7-results {RESULTS_V71} "
              f"--readout {root}/readout --v7-readout {OUT}/b-v7.1 --output {root}/acceptance"]
    return lines


def job_lines(stage, *, v6_features="", partition=None, init=None, shards=4):
    if stage != "c1" and v6_features:
        raise ValueError("--v6-features 只用于 --stage c1（只有 C-1 的 double_well 是 200 条）")
    if stage != "c2" and (partition or init):
        raise ValueError("--temporal-partition / --readout-init 只用于 --stage c2（C-1 四个变体都排）")
    if stage == "b-v7.1":
        return b_lines(shards)
    if stage == "c1":
        if not v6_features:
            raise ValueError(V6_REQUIRED)
        if not Path(v6_features).is_absolute():
            raise ValueError(f"--v6-features 要写绝对路径（和 v7 试跑用的相同），不是 {v6_features}")
        return c1_lines(v6_features)
    if partition not in PARTITIONS or init not in INITS:
        raise ValueError("--stage c2 要显式给 --temporal-partition 和 --readout-init（C-1 汇总推荐的变体，或 Jerry 指定的）")
    return c2_lines(partition, init)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("b-v7.1", "c1", "c2"), required=True)
    p.add_argument("--v6-features", default="", help="c1 必须给：v7 试跑用的 v6 double_well-200 特征目录（绝对路径）")
    p.add_argument("--temporal-partition", choices=PARTITIONS, help="c2 必须给：C-1 定下的时间一致性切分")
    p.add_argument("--readout-init", choices=INITS, help="c2 必须给：C-1 定下的 P_out 初始化")
    p.add_argument("--shards", type=int, default=4, help="b-v7.1：长时程测试集的编码切成几片（每片一张卡）")
    a = p.parse_args(argv)
    try:
        lines = job_lines(a.stage, v6_features=a.v6_features, partition=a.temporal_partition, init=a.readout_init, shards=a.shards)
    except ValueError as error:
        p.error(str(error))
    sys.stdout.write("\n".join(lines)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
