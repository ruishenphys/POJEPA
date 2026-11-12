"""生成 v8 要提交的全部命令（每行一个独立任务），格式和 make_v7_jobs.py 相同：按 “# =====” 分段，段与段之间有依赖，
同一段里的命令互不依赖、可以同时跑在不同的卡上；GPU 命令都写 --device cuda:0（排队系统用 CUDA_VISIBLE_DEVICES 分卡）。

  python tools/make_v8_jobs.py --stage pilot    > jobs-v8-pilot.txt      # 试跑：7 个场景各 200 条、2000 步、1 个种子
  python tools/make_v8_jobs.py --stage full     > jobs-v8-full.txt       # 正式：各 2000 条、10000 步、3 个种子
  python tools/make_v8_jobs.py --stage variants > jobs-v8-variants.txt   # 分布外评测（评测协议的标准档）+ 规划（full 之后）
  python tools/make_v8_jobs.py --stage strength > jobs-v8-strength.txt   # 强度曲线（轻档、重档；variants 之后）
  python tools/make_v8_jobs.py --stage long     > jobs-v8-long.txt       # 长时程滚动（16–64 步；full 之后，S25c）
  python tools/make_v8_jobs.py --stage timing   > jobs-v8-timing.txt     # 时间采样按帧评测（帧率、抖动、曝光、切换；full 之后，S25d）
  python tools/make_v8_jobs.py --stage space    > jobs-v8-space.txt      # 空间采样评测（分辨率、缩小、平移、漂移；full 之后，S25e）
  python tools/make_v8_jobs.py --stage readout  > jobs-v8-readout.txt    # 物理状态读出、VPT、反事实、模态频率（full、long 之后，S25f）

和 v7 的区别（S21b）：
- 场景是 po_jepa.protocols.SCENES：v7 的 5 个加 gray_scott、push。
- 分布外测试表来自评测协议 po_jepa.protocols（P1–P13，三档强度；DATASET.md 有协议表），不是 v7 的 TEST_VARIANTS。
  v7 的 TEST_VARIANTS 和 make_v7_jobs.py 一字不改：v7 正式实验和仿真补跑（make_addon_jobs.py）都在用它。
- 正式阶段的动作探针（B2）每个场景都跑，不只 double_well。
- 默认目录是 ../features-v8、../results-v8，和 v7 分开。同一个场景、同一个调节项的数据在 v7 树和 v8 树上逐位相同
  （S20b、S21a、S21b 的跨树对照），想复用 v7 已编码的特征，可以用 --features 指到 v7 的特征目录。

S25a：五个二阶场景（protocols.SECOND_ORDER_SCENES）能排了，但要显式写 --scenes（默认仍是 7 个一阶场景，任务表不变）。
它们现在按单帧复制（dup）编码，特征里看不到速度，只适合当“看不到速度时有多差”的对照；两帧（pair）编码等 S22 探针结果
在 S25b 接，接好以后并进默认场景。排了二阶场景时，输出的第一行注释会写明这一点。

S25c：--stage long 排长时程测试集（<场景>~horizon=128 的测试段，每类 200 条 × 129 帧，只编码测试段）的编码、汇总和
tools/long_rollout.py 滚动评测（3 个种子 × 各条件；模型是 full 阶段在基础场景上训练好的，不再训练）。

S25d：--stage timing 排时间采样测试集（<场景>~<po_jepa.protocols.TIMING_TESTS 之一> 的测试段，每类 200 条，只编码测试段）的
编码、汇总和 tools/timing_eval.py 按帧评测（3 个种子 × 各条件；同样只评测）。

S25e：--stage space 排空间采样测试集（<场景>~<po_jepa.protocols.SPACE_TESTS 之一> 的测试段，每类 200 条，只编码测试段，
按原生分辨率编码）的编码、汇总和 tools/space_eval.py 评测（3 个种子 × 各条件；同一组窗口在基础测试集上配对；同样只评测）。
S25f：--stage readout 排 tools/state_readout.py 的探针拟合（每个场景一个，主实验缓存验证段）和真值（CPU）、长时程滚动的
读出和 VPT、反事实预测（3 个种子 × 各条件），以及 tools/modal_spectrum.py 的模态频率对照（只排线性场景
advection_diffusion、spring_chain、wave 里在 --scenes 中的，只排 PO-JEPA 条件）。要用 --stage full 的主实验缓存和模型、
--stage long 的长时程缓存，本文件不编码、不训练。

S26a：ICML 计划的 N1 基线和 N2 消融就是仿真补跑的条件（ADDON_CONDITIONS：po_jepa_time_only、po_jepa_space_only、
transformer_consistency、fixed_fourier、koopman；08-conditions.md），2026-11-12 加了 po_jepa_rollout_sup（有真值两段滚动的对照，
08 §6；Jerry 定 12 个场景都跑）。它们默认不进 v8 任务表（前 5 个 Jerry 2026-11-10 定：可选）：
  python tools/make_v8_jobs.py --stage addon       > jobs-v8-addon.txt        # full 的 A–C 段之后：补跑标定、检查、训练+评测
  python tools/make_v8_jobs.py --stage addon-pilot > jobs-v8-addon-pilot.txt  # 同上，pilot 规模（pilot 的 A–C 段之后）
补跑标定用本目录 run.py calibrate 写的基础标定（--base-calibration local），和 v8 的 v7 条件同一份 σ_z、λ。
addon 阶段可以用 --conditions 只排其中几个（例如只排 po_jepa_rollout_sup：--conditions po_jepa_rollout_sup）；AC 补跑标定
每个场景 1 条，三个要标定的条件（transformer_consistency、fixed_fourier、po_jepa_rollout_sup）总是一起标。
训练好以后，variants、strength、long、timing、space、readout 各阶段用 --conditions 选这些条件即可评测（目录结构和 v7 条件相同）。
注意：v8 和“v7 场景上的仿真补跑”（make_addon_jobs.py，导入 v7 标定）不能在同一份工作树里跑——两者的基础标定文件同名
（configs/calibration/<场景>-<条数>-u<步数>.yaml），补跑标定也同名。local 模式遇到导入的 v7 标定会直接报错。
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from po_jepa import protocols, variants  # noqa: E402

CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity", "vjepa2_ac_small", "vjepa2_ac")
# S26a：和 po_jepa.runtime.addon.conditions.ADDON_CONDITIONS 相同（测试核对）；这里不导入 torch
ADDON_CONDITIONS = ("po_jepa_time_only", "po_jepa_space_only", "transformer_consistency", "fixed_fourier", "koopman",
                    "po_jepa_rollout_sup")   # 08 §6：有真值两段滚动的对照（2026-11-12 加）
SEEDS = (1703, 1704, 1705)
# 规划的真值参照（oracle）每次 CEM 的样本数：真值仿真在 CPU 上逐个跑，新场景每个 tick 贵得多
# （gray_scott 约 60 ms、push 约 30 ms，粒子类约 3 ms），默认 64 会让 identity 那条规划任务占着 GPU 跑几个小时（S21b 实测）
# S25a：二阶场景每个 tick 本机 CPU 实测 wave 约 12 ms，其余 ≤ 3 ms（pendulum 0.8、bouncing_balls 2.6、cart_pole 2.0、
# spring_chain 0.1；double_well 3.9 作对照），只有 wave 降到 32（docs/segments/S25a.md §4）
ORACLE_SAMPLES = {"gray_scott": 16, "push": 32, "wave": 32}
LONG_HORIZON = 128   # S25c：长时程测试集的 horizon（variants.HORIZONS 的最大值）
MODAL_SCENES = ("advection_diffusion", "spring_chain", "wave")   # S25f：有解析线性模态的场景（runtime/modal_spectrum.LINEAR_SCENES）


def split_scene(scene):
    system, _, knobs = scene.partition("~")
    return system, variants.format(system, variants.parse(system, knobs))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("pilot", "full", "variants", "strength", "long", "timing", "space", "readout", "addon", "addon-pilot"), required=True)
    p.add_argument("--scenes", default=",".join(protocols.SCENES), help="逗号分隔的场景（系统[~调节项]）")
    p.add_argument("--protocols", default="", help="只要这些协议，逗号分隔，例如 P1,P4（默认全部）")
    p.add_argument("--conditions", default=None,
                   help="variants、strength、long、timing、space、readout 评哪些条件（默认 v7 的条件）；可以加补跑的 6 个（S26a、08 §6，"
                        "要先跑 --stage addon）；addon、addon-pilot 阶段只能写补跑条件，不写时 6 个全排；pilot、full 不收这个参数")
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
        if s not in protocols.ALL_SCENES:
            raise SystemExit(f"{s} 不是 v8 的基础场景：{protocols.ALL_SCENES}")
    ids = [i for i in a.protocols.split(",") if i] or None
    if ids and not set(ids) <= set(protocols.BY_ID):
        raise SystemExit(f"没有这些协议：{sorted(set(ids)-set(protocols.BY_ID))}")
    # 默认值按阶段定：只有确实没写 --conditions 时才用默认（S27a 审核：不能拿字符串是否等于 v7 列表来判断“没写”）
    addon_stage = a.stage in ("addon", "addon-pilot")
    if a.conditions is None:
        conditions = list(ADDON_CONDITIONS if addon_stage else CONDITIONS)
    else:
        conditions = [c for c in a.conditions.split(",") if c]
    if not set(conditions) <= {*CONDITIONS, *ADDON_CONDITIONS}:
        raise SystemExit(f"没有这些条件：{sorted(set(conditions)-{*CONDITIONS, *ADDON_CONDITIONS})}")
    if a.stage in ("pilot", "full") and a.conditions is not None:
        raise SystemExit(f"--conditions 只用于 variants、strength、long、timing、space、readout、addon；{a.stage} 的条件是固定的")
    if addon_stage:
        if not conditions or not set(conditions) <= set(ADDON_CONDITIONS) or len(set(conditions)) != len(conditions):
            raise SystemExit(f"{a.stage} 的 --conditions 只能是不重复的补跑条件：{ADDON_CONDITIONS}")
        conditions = [c for c in ADDON_CONDITIONS if c in conditions]   # 按 ADDON_CONDITIONS 的顺序排
    pilot = a.stage in ("pilot", "addon-pilot")
    n, updates = (200, 2000) if pilot else (2000, 10000)
    seeds = SEEDS[:1] if pilot else SEEDS
    root = f"{a.results}/{'pilot' if pilot else 'full'}"
    second = [s for s in scenes if s in protocols.SECOND_ORDER_SCENES]
    if second:
        print(f"# 注意：二阶场景 {','.join(second)} 按单帧复制（dup）编码，特征里看不到速度（pair 编码等 S22 探针结果，S25b 接）。")
    enc = f"--upstream-root {a.upstream_root} --checkpoint {a.encoder}"

    def name(scene, extra=None):
        system, knobs = split_scene(scene) if extra is None else (split_scene(scene)[0], protocols.merged(scene, extra))
        return system if not knobs else f"{system}~{knobs}"

    def feats(scene, extra=None):
        return f"{a.features}/{name(scene, extra)}-{n}"

    def ds(scene):
        system, knobs = split_scene(scene)
        return f"--system {system}" + (f" --variant {knobs}" if knobs else "") + f" --trajectories {n}"

    if a.stage in ("addon", "addon-pilot"):
        base = "pilot" if pilot else "full"
        print(f"# S26a 补跑条件（N1 基线、N2 消融；08 §6 的有真值滚动对照）：{', '.join(conditions)}。")
        print(f"# 前提：--stage {base} 的 A–C 段已在本工作树完成（特征编好、run.py calibrate 写了基础标定）；"
              "不能和 make_addon_jobs.py（v7 场景的补跑）共用一份工作树。")
        print(f"# ===== AC 补跑标定（GPU，每个场景 1 条，共 {len(scenes)} 条；transformer_consistency、fixed_fourier 的 λ，"
              "po_jepa_rollout_sup 的 λ_r）=====")
        for s in scenes:
            print(f"python run.py calibrate-addon {ds(s)} --updates {updates} --features-root {feats(s)} --base-calibration local "
                  f"--device cuda:0 --output {root}/calibrate-addon-{name(s)}")
        probe = scenes[min(1, len(scenes)-1)]
        print("# ===== AC2 检查（GPU，AC 段完成后跑 1 条；每种 GPU 型号做一次）=====")
        print(f"python run.py check-addon {ds(probe)} --updates {updates} --features-root {feats(probe)} --base-calibration local "
              f"--device cuda:0 --output {root}/check-addon")
        print(f"# ===== AD 训练+评测（GPU，AC 段对应场景完成后；{len(scenes)*len(seeds)*len(conditions)} 条，互不依赖）=====")
        for s in scenes:
            for seed in seeds:
                for c in conditions:
                    print(f"python run.py run {ds(s)} --updates {updates} --seed {seed} --condition {c} "
                          f"--features-root {feats(s)} --device cuda:0 --output {root}/{name(s)}/seed{seed}/{c}")
        print("# ===== AE 汇总（CPU，AD 段完成后；和 v7 条件同一张表，每组都要有本次排的补跑条件）=====")
        print(f"python tools/summarize_v7.py --results {root} --output {root}/summary --addon-conditions {','.join(conditions)}")
        return

    if a.stage == "long":
        long = f"horizon={LONG_HORIZON}"
        print("# 前提：--stage full 的 A–D 段已完成（模型在基础场景上训练好），本文件只做评测，不再训练。")
        print(f"# 长时程测试集：<场景>~{long} 的测试段（每条 {LONG_HORIZON+1} 帧，float16 每条约 93 MB），"
              "前 33 帧和主实验测试段逐位相同；指标见 src/po_jepa/runtime/long_rollout.py。")
        print(f"# ===== L1 编码长时程测试集（GPU，{len(scenes)*a.shards} 条；只编码测试段）=====")
        for s in scenes:
            for i in range(a.shards):
                print(f"python tools/encode_v7.py encode --system {split_scene(s)[0]} --variant {protocols.merged(s, long)} "
                      f"--trajectories {n} --splits test {enc} --output {feats(s, long)} --device cuda:0 --shard {i}/{a.shards}")
        print("# ===== L2 汇总编码（CPU）=====")
        for s in scenes:
            print(f"python tools/encode_v7.py finalize --system {split_scene(s)[0]} --variant {protocols.merged(s, long)} "
                  f"--trajectories {n} --splits test --output {feats(s, long)}")
        print(f"# ===== L3 长时程滚动（GPU，{len(scenes)*len(seeds)*len(conditions)} 条，互不依赖）=====")
        for s in scenes:
            for seed in seeds:
                for c in conditions:
                    extra = f" --ac-checkpoint {a.ac_checkpoint}" if c == "vjepa2_ac" else ""
                    print(f"python tools/long_rollout.py --job {root}/{name(s)}/seed{seed}/{c} --features {feats(s, long)} "
                          f"--horizon {LONG_HORIZON} --output {root}/long/{name(s)}/seed{seed}/{c}.json --device cuda:0{extra}")
        print("# ===== L4 汇总（CPU）=====")
        print(f"python tools/summarize_v7.py --results {root} --output {root}/summary")
        return

    if a.stage == "readout":
        long = f"horizon={LONG_HORIZON}"
        out = f"{root}/readout"
        print("# 前提：--stage full 的 A–D 段（主实验缓存和模型）和 --stage long 的 L1–L2（长时程缓存）已完成；本文件不编码、不训练。")
        print("# 指标见 src/po_jepa/runtime/state_readout.py、src/po_jepa/runtime/modal_spectrum.py（模态频率是探索性的）。")
        print(f"# ===== R1 拟合探针（GPU，{len(scenes)} 条；主实验缓存的验证段，full 读出约 10 GB 内存）=====")
        for s in scenes:
            print(f"python tools/state_readout.py fit {ds(s)} --features {feats(s)} --output {out}/probe/{name(s)} --device cuda:0")
        print(f"# ===== R2 真值（CPU，{len(scenes)} 条；反事实重新仿真，gray_scott 最慢）=====")
        for s in scenes:
            print(f"python tools/state_readout.py truth {ds(s)} --features {feats(s)} --long-features {feats(s, long)} "
                  f"--horizon {LONG_HORIZON} --output {out}/truth/{name(s)}.json")
        print(f"# ===== R3 长时程读出、反事实（GPU，{2*len(scenes)*len(seeds)*len(conditions)} 条，互不依赖）=====")
        for s in scenes:
            for seed in seeds:
                for c in conditions:
                    extra = f" --ac-checkpoint {a.ac_checkpoint}" if c == "vjepa2_ac" else ""
                    common = (f"--job {root}/{name(s)}/seed{seed}/{c} --probe {out}/probe/{name(s)} "
                              f"--truth {out}/truth/{name(s)}.json")
                    print(f"python tools/state_readout.py rollout {common} --features {feats(s, long)} --horizon {LONG_HORIZON} "
                          f"--output {out}/rollout/{name(s)}/seed{seed}/{c}.json --device cuda:0{extra}")
                    print(f"python tools/state_readout.py counterfactual {common} --features {feats(s)} "
                          f"--output {out}/counterfactual/{name(s)}/seed{seed}/{c}.json --device cuda:0{extra}")
        modal = [s for s in scenes if split_scene(s)[0] in MODAL_SCENES]
        po = [c for c in conditions if c.startswith("po_jepa")]
        print(f"# ===== R4 模态频率对照（GPU，{len(modal)*len(seeds)*len(po)} 条；只有线性场景、PO-JEPA 条件；探索性）=====")
        for s in modal:
            for seed in seeds:
                for c in po:
                    print(f"python tools/modal_spectrum.py --job {root}/{name(s)}/seed{seed}/{c} --features {feats(s)} "
                          f"--output {out}/modal/{name(s)}/seed{seed}/{c}.json --device cuda:0")
        return

    if a.stage == "timing":
        tests = [(s, t) for s in scenes for t in protocols.TIMING_TESTS]
        print("# 前提：--stage full 的 A–D 段已完成（模型在 4 fps 的基础场景上训练好），本文件只做评测，不再训练。")
        print(f"# 时间采样测试集：{', '.join(protocols.TIMING_TESTS)}；每条帧数 = 32·fps/4 + 1（fps=16 时 129 帧，float16 每条约 93 MB），"
              "整 tick 的帧和主实验测试段逐位相同；指标见 src/po_jepa/runtime/timing_eval.py。")
        print(f"# ===== T1 编码时间采样测试集（GPU，{len(tests)*a.shards} 条；只编码测试段）=====")
        for s, t in tests:
            for i in range(a.shards):
                print(f"python tools/encode_v7.py encode --system {split_scene(s)[0]} --variant {protocols.merged(s, t)} "
                      f"--trajectories {n} --splits test {enc} --output {feats(s, t)} --device cuda:0 --shard {i}/{a.shards}")
        print("# ===== T2 汇总编码（CPU）=====")
        for s, t in tests:
            print(f"python tools/encode_v7.py finalize --system {split_scene(s)[0]} --variant {protocols.merged(s, t)} "
                  f"--trajectories {n} --splits test --output {feats(s, t)}")
        print(f"# ===== T3 按帧评测（GPU，{len(tests)*len(seeds)*len(conditions)} 条，互不依赖）=====")
        for s, t in tests:
            for seed in seeds:
                for c in conditions:
                    extra = f" --ac-checkpoint {a.ac_checkpoint}" if c == "vjepa2_ac" else ""
                    print(f"python tools/timing_eval.py --job {root}/{name(s)}/seed{seed}/{c} --test-variant {t} --features {feats(s, t)} "
                          f"--output {root}/timing/{name(s, t)}/seed{seed}/{c}.json --device cuda:0{extra}")
        print("# ===== T4 汇总（CPU）=====")
        print(f"python tools/summarize_v7.py --results {root} --output {root}/summary")
        return

    if a.stage == "space" and "koopman" in conditions:
        # S26a：koopman 的基底是 16×16 的，换网格时要求网格坐标等于 16→G 重采样的目标坐标（koopman.grid_basis）；空间采样的
        # 原生网格 12×12、20×20、24×24 是 regular_grid(G, G)，对不上（20、24 边缘差 0.006–0.010），评测会直接报错。
        # 08 §5.3 给 DROID 原生分辨率写了放宽规则（S11 才实现），仿真空间采样要不要照用另定。
        raise SystemExit("koopman 还不能评空间采样（原生网格和 16×16 基底的重采样坐标对不上，见 08 §5.3），--conditions 里去掉 koopman")
    if a.stage == "space":
        tests = [(s, t) for s in scenes for t in protocols.SPACE_TESTS]
        print("# 前提：--stage full 的 A–D 段已完成（模型在 256、相机不动的基础场景上训练好，基础特征在 --features 下），本文件只做评测，不再训练。")
        print(f"# 空间采样测试集：{', '.join(protocols.SPACE_TESTS)}；每条 33 帧，按原生分辨率编码，特征网格 = resolution/16"
              "（384 时 24×24，float16 每条约 53 MB）；指标见 src/po_jepa/runtime/space_eval.py。")
        print(f"# ===== S1 编码空间采样测试集（GPU，{len(tests)*a.shards} 条；只编码测试段）=====")
        for s, t in tests:
            for i in range(a.shards):
                print(f"python tools/encode_v7.py encode --system {split_scene(s)[0]} --variant {protocols.merged(s, t)} "
                      f"--trajectories {n} --splits test {enc} --output {feats(s, t)} --device cuda:0 --shard {i}/{a.shards}")
        print("# ===== S2 汇总编码（CPU）=====")
        for s, t in tests:
            print(f"python tools/encode_v7.py finalize --system {split_scene(s)[0]} --variant {protocols.merged(s, t)} "
                  f"--trajectories {n} --splits test --output {feats(s, t)}")
        print(f"# ===== S3 评测（GPU，{len(tests)*len(seeds)*len(conditions)} 条，互不依赖）=====")
        for s, t in tests:
            for seed in seeds:
                for c in conditions:
                    extra = f" --ac-checkpoint {a.ac_checkpoint}" if c == "vjepa2_ac" else ""
                    print(f"python tools/space_eval.py --job {root}/{name(s)}/seed{seed}/{c} --test-variant {t} --features {feats(s, t)} "
                          f"--base-features {feats(s)} --output {root}/space/{name(s, t)}/seed{seed}/{c}.json --device cuda:0{extra}")
        print("# ===== S4 汇总（CPU）=====")
        print(f"python tools/summarize_v7.py --results {root} --output {root}/summary")
        return

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
