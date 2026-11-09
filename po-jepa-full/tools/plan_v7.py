"""v7 可调数据集的“规划”任务：把小球（或染料）推到目标，报成功率（与真实数据上的“规划成功率”同一类指标）。

每个回合：测试轨迹 tid 的第 start 个 tick 为起点；目标状态渲染并编码成目标特征 z_goal。规划器在模型里搜动作：
  CEM（交叉熵法）在 H 个 tick 的分段常数动作上搜索，代价 = JEPA 距离 d(Φ(z_now, u, 剩余时长), z_goal)；
  MPC：每次只执行前 --replan 个 tick（在真值仿真里执行），重新渲染、编码当前画面再规划剩下的。
目标 = 从起点按一段随机动作（每 2 tick 换一次）走 H 个 tick 后的状态（保证可达，且与“不动”的结果拉开距离）。
20 次重抽都达不到门槛的回合记为目标无效（报告的 goal_valid、goals），不进成功率和误差统计。
同时给出三个参照（同一批回合）：
  zero   ：一直不动；
  random ：随机动作；
  oracle ：同样的 CEM，但代价用真值仿真器里的状态距离（能做到多好的上界）。
成功判据（--success 可改）：最终状态误差 ≤ 0.5 × 不动时的误差（对所有场景通用）；小球场景另报受控小球到目标的平均距离（像素）。
“状态误差”用的状态摘要（报告里的 state_summary）：小球场景是受控小球位置；advection_diffusion 是 16×16 平均浓度场；
S21b 新场景 gray_scott 是 U、V 的 16×16 池化场（512 维，和 advection_diffusion 同一类），push 是方块位姿
(b_x, b_y, size/2·θ)，也就是要把方块推到目标，推杆在哪不算。push 的真值规划（oracle）代价用完整状态
(c_x, c_y, b_x, b_y, size/2·θ) 到目标状态的距离：只看方块时推杆碰到方块之前代价处处相同，CEM 无从下手；模型规划的代价
本来就是到目标画面（推杆也在画面里）的特征距离。push 的目标动作不是纯随机（随机动作很少碰到方块）：
每 2 个 tick 一段，方向 = 起点时推杆指向方块中心的方向 ± 30° 内均匀，幅值 ∈ [0.25, 0.5]，同样要求方块位姿和“不动”差 ≥ 0.15。

用法（GPU，需要冻结编码器做目标与重规划编码）：
  python tools/plan_v7.py --job ../results-v7/full/double_well/seed1703/po_jepa_full \
      --upstream-root ../vjepa2 --checkpoint ../checkpoints/vjepa2-vitg-fpc64-256.pth \
      --episodes 100 --horizon 8 --output ../results-v7/full/plan/double_well-po_jepa_full-seed1703.json --device cuda:0
--job 是 run.py run 的输出目录（模型、数据集、种子都从里面读，不用再写）。
--test-variant 可选：在换了调节项的场景里规划（分布外规划）。
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def summary(system_name, system, v, state):
    """规划误差用的状态摘要（见文件头）。state 是 variant_trajectory / variant_fields 的一帧。"""
    import numpy as np
    if system_name == "advection_diffusion":
        return state.reshape(16, 16, 16, 16).mean((1, 3)).reshape(-1)
    if system_name == "gray_scott":
        return system.hidden(state[None])[0][:512]               # U、V 池化场；注入点不计入
    if system_name == "push":
        return np.array([state[2], state[3], .5*v.param("push", "size")*state[4]])   # 方块位姿
    return state[:2]                                             # 受控小球（多粒子场景是第 0 个）


def oracle_target(system_name, system, v, state):
    """真值规划的代价用的状态：push 是完整状态（见文件头），其他场景就是 summary。"""
    import numpy as np
    if system_name == "push":
        return np.array([state[0], state[1], state[2], state[3], .5*v.param("push", "size")*state[4]])
    return summary(system_name, system, v, state)


def goal_controls(system_name, state, horizon, goal_rng):
    """目标动作 [H,2]：每 2 个 tick 一段。push 朝方块推（见文件头），其他场景均匀随机（与 S21b 之前逐位相同）。"""
    import numpy as np
    n = (horizon+1)//2
    if system_name == "push":
        d = state[2:4]-state[:2]
        angle = math.atan2(d[1], d[0])+goal_rng.uniform(-math.pi/6, math.pi/6, n)
        size = goal_rng.uniform(.25, .5, n)
        return np.repeat(np.stack((size*np.cos(angle), size*np.sin(angle)), 1), 2, 0)[:horizon]
    return np.repeat(goal_rng.uniform(-.5, .5, (n, 2)), 2, 0)[:horizon]


def goal_far_enough(g, zero_final, relative):
    """目标门槛：与“不动”的终点差 ≥ 0.05 × 目标范数（场：advection_diffusion、gray_scott）或 ≥ 0.15（其他场景）。"""
    import numpy as np
    gap = np.linalg.norm(g-zero_final)
    return bool(gap >= (.05*np.linalg.norm(g) if relative else .15))


def _parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--upstream-root", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--ac-checkpoint", type=Path, help="条件是 vjepa2_ac 时需要")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--horizon", type=int, default=8, help="目标在多少个控制 tick（0.25 s）之后")
    p.add_argument("--replan", type=int, default=2, help="每执行几个 tick 重新规划一次")
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--elites", type=int, default=24)
    p.add_argument("--oracle-samples", type=int, default=64, help="oracle 每次 CEM 的样本数（真值仿真在 CPU 上逐个跑）")
    p.add_argument("--success", type=float, default=.5)
    p.add_argument("--test-variant", default=None)
    p.add_argument("--no-baselines", action="store_true", help="不跑 zero/random/oracle 参照（它们与模型无关，每个场景只需在一个任务里跑）")
    p.add_argument("--fake-encoder", action="store_true", help="仅 CPU 流程测试")
    return p.parse_args()


def _job_env(job_dir):
    for sub in ("run/job", "job", "run", "."):
        if (job_dir/sub/"job.json").exists():
            job_dir = (job_dir/sub).resolve(); break
    job = json.loads((job_dir/"job.json").read_text())
    os.environ.update(PO_JEPA_PROTOCOL="v7", PO_JEPA_SEED=str(job["seed"]), PO_JEPA_UPDATES=str(job["updates"]))
    scene, _, count = job["dataset"].rpartition("-")
    os.environ["PO_JEPA_DATASET"] = f"{scene}:{count}"
    return job_dir, job


def main():
    a = _parse()
    job_dir, job = _job_env(a.job)
    if a.ac_checkpoint:
        os.environ["PO_JEPA_AC_CHECKPOINT"] = str(a.ac_checkpoint.resolve())
    sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"tools"))
    import numpy as np
    import torch
    from torch.nn import functional as F
    from po_jepa import datasets, simulation, systems, variants
    from po_jepa.actions import ControlSpec, from_physical_segments
    from po_jepa.geometry import regular_grid
    from po_jepa.runtime import cache, comparison, evaluation
    from po_jepa.runtime.workflow import read
    import encode_v7

    device = torch.device(a.device)
    condition = job["condition"]
    base = datasets.active()
    # —— 模型：与 transfer 相同，按任务目录里验证集选中的 checkpoint 载入 ——
    model, loss = comparison.build_condition(condition, device)
    if condition != "identity":
        root = job_dir/"training"/condition
        if not (root/"COMPLETE.json").exists() and job.get("training_run"):
            root = Path(job["training_run"])/"run"/"job"/"training"/condition
        selected = read(root/"report.json")["best"]
        opt = comparison.make_optimizer(model)
        evaluation.load_selected_checkpoint(condition, selected, model=model, optimizer=opt, config=loss,
                                            expected_provenance=selected["checkpoint_provenance"], root=root)
        del opt
    model.eval()
    if a.test_variant is not None:
        datasets._set_for_tests(datasets.DatasetSpec(base.system, base.count,
                                                     variants.format(base.system, variants.parse(base.system, a.test_variant))))
    spec = datasets.active(); v = spec.knobs; system = systems.get()

    # —— 编码器 ——
    if a.fake_encoder:
        encoder = encode_v7._fake_encoder(torch)
    else:
        from po_jepa import vjepa
        arch = vjepa.EncoderArchitectureSpec(vjepa.UPSTREAM_COMMIT, "configs/train/vitg16/droid-256px-8f.yaml",
            "vit_giant_xformers", 256, 16, 2, 512, True, True, False, True, True, True)
        build = vjepa.build_encoder_architecture(upstream_root=a.upstream_root, architecture=arch, initialization_seed=0)
        loaded = vjepa.load_local_encoder(build, checkpoint=vjepa.LocalCheckpointSpec(
            a.checkpoint, cache.CHECKPOINT_SHA, cache.PUBLICATION, "target_encoder", ("module.", "backbone."), True, (), ()))
        encoder = vjepa.FrozenFrameEncoder(loaded, preprocess=cache.preprocess_spec(), encoder_device=device, autocast_dtype=None)

    def encode(frame, tid):
        z = encode_v7._encode_trajectory(encoder, frame[None], tid, simulation, cache, torch, np, 1, a.fake_encoder)
        return torch.from_numpy(z).float().to(device)            # [1,256,1408]

    field_scene = spec.system == "advection_diffusion"
    new_scene = spec.system in ("gray_scott", "push")             # S21b：没有 v6 原实现，基础场景也走可调实现
    relative_gap = field_scene or spec.system == "gray_scott"     # 场：目标与“不动”的差按场的范数的 5% 判

    def simulate(tid, state, start, actions, render=True):
        """从 state（tick=start）起执行下达动作 actions[k,2]，返回 (新状态, 新画面)。render=False 不画（真值规划的代价只看状态）。"""
        k = actions.shape[0]
        starts = np.arange(32, dtype=np.int64); vals = np.zeros((32, 2))
        vals[start:start+k] = np.clip(actions, -.5, .5)
        c = simulation.TrajectoryControls(tid, starts, starts+1, vals)
        if field_scene:
            f = system.variant_fields(tid, v, controls=c, start=start, field=state, ticks=k)[-1]
            return f, picture(f, tid, start+k) if render else None
        q = system.variant_trajectory(tid, v, controls=c, start=start, state=state, ticks=k)[-1]
        return q, picture(q, tid, start+k) if render else None

    def picture(state, tid, tick):
        """与数据集同一套渲染：基础场景用原渲染函数，有调节项时用可调渲染。tick：这一帧的时刻（会动的干扰光点、像素噪声要用）。"""
        if field_scene:
            return (system.render_fields(state[None]) if v.is_default else system.draw_fields(state[None], tid, v, ticks=[tick]))[0]
        if v.is_default and not new_scene:
            return system.render(state[None])[0]
        return system.draw(state[None], tid, v, ticks=[tick])[0]

    summary_state = lambda state: summary(spec.system, system, v, state)
    oracle_state = lambda state: oracle_target(spec.system, system, v, state)
    goal_actions = lambda state: goal_controls(spec.system, state, a.horizon, goal_rng)

    spec_c = ControlSpec(2, .25, ("u_x", "u_y"), ("simulation_position_per_second",)*2,
                         (-.5, -.5), (.5, .5), (1., 1.), (0., 0.), "reference/protocol.md")

    def controls_for(u):                                          # u: [B,k,2] → 每 tick 一段
        B, k, _ = u.shape
        s = torch.arange(k, dtype=torch.float64)[None].repeat(B, 1)*.25
        return from_physical_segments(starts_seconds=s, ends_seconds=s+.25, u_values=u.float().to(device),
                                      n_segments=torch.full((B,), k, dtype=torch.int64),
                                      segment_valid=torch.ones((B, k), dtype=torch.bool), origin_seconds=torch.zeros(B, dtype=torch.float64),
                                      control_spec=spec_c, source_ids=tuple(f"plan-{i}" for i in range(B)))

    grid = regular_grid(16, 16)

    @torch.no_grad()
    def model_cost(z, z_goal, u):
        B, k, _ = u.shape
        out = []
        for i in range(0, B, 64):
            uu = u[i:i+64]
            pred = model(z.expand(uu.shape[0], -1, -1).contiguous(), controls_for(uu),
                         torch.full((uu.shape[0],), float(k), dtype=torch.float64), grid)
            ln = lambda x: F.layer_norm(x, (x.shape[-1],), eps=1e-5)
            out.append((ln(pred)-ln(z_goal)).abs().mean((1, 2)))
        return torch.cat(out).cpu().numpy()

    def oracle_cost(tid, state, start, goal_state, u):
        return np.array([np.linalg.norm(oracle_state(simulate(tid, state, start, ui, render=False)[0])-goal_state) for ui in u])

    def cem(cost, k, rng, samples):
        mu, sd = np.zeros((k, 2)), np.full((k, 2), .3)
        elites = max(2, a.elites*samples//a.samples)
        for _ in range(a.iterations):
            u = np.clip(mu+sd*rng.standard_normal((samples, k, 2)), -.5, .5)
            elite = u[np.argsort(cost(u))[:elites]]
            mu, sd = elite.mean(0), elite.std(0)+1e-3
        return mu

    test = list(spec.split_range("test"))
    rng = np.random.default_rng(1703)
    goal_rng = np.random.default_rng(1704)      # 目标单独一条随机流：换模型、换规划参数时目标不变
    rows, started = [], time.perf_counter()
    for ep in range(a.episodes):
        tid = test[ep % len(test)]; start = int(datasets.START_TICKS[(ep//len(test)) % 5])
        if start+a.horizon > 32:
            start = 32-a.horizon
        states0 = (system.variant_fields(tid, v) if field_scene else system.variant_trajectory(tid, v))
        s0 = states0[start]
        # 目标：同起点按随机下达动作（每 2 个 tick 换一次）走 H 个 tick；与“不动”的结果差得足够远才用（最多重抽 20 次）
        zero_final = summary_state(simulate(tid, s0, start, np.zeros((a.horizon, 2)), render=False)[0])
        for _ in range(20):
            gu = goal_actions(s0)
            goal_state, goal_frame = simulate(tid, s0, start, gu)
            g = summary_state(goal_state)
            valid = goal_far_enough(g, zero_final, relative_gap)
            if valid:
                break
        # S21b（Codex 审核）：20 次都没达到门槛的目标记为无效，不进成功率和误差统计（规划照常跑，随机流消耗不变）
        z_goal = encode(goal_frame, tid); g_oracle = oracle_state(goal_state)
        result = {}
        for planner in ("model",) if a.no_baselines else ("model", "oracle", "random"):
            state, t, frame = s0, start, None
            if planner == "random":
                state = simulate(tid, s0, start, rng.uniform(-.5, .5, (a.horizon, 2)), render=False)[0]
            else:
                left = a.horizon
                frame_z = encode(picture(s0, tid, start), tid) if planner == "model" else None
                while left > 0:
                    if planner == "model":
                        plan = cem(lambda u: model_cost(frame_z, z_goal, torch.from_numpy(u)), left, rng, a.samples)
                    else:
                        plan = cem(lambda u: oracle_cost(tid, state, t, g_oracle, u), left, rng, a.oracle_samples)
                    step = min(a.replan, left)
                    state, frame = simulate(tid, state, t, plan[:step])
                    t += step; left -= step
                    if planner == "model" and left > 0:
                        frame_z = encode(frame, tid)
            err = float(np.linalg.norm(summary_state(state)-g))
            result[planner] = err
        zero_err = float(np.linalg.norm(zero_final-g))
        rows.append(dict(trajectory_id=tid, start_tick=start, goal_valid=bool(valid), zero=zero_err, **result))
        shown = "  ".join(f"{k} {val:.3f}" for k, val in result.items())
        print(f"[{condition} {spec.tag}] 回合 {ep+1}/{a.episodes}：{shown}  zero {zero_err:.3f}" + ("" if valid else "  （目标无效）") +
              f"  {time.perf_counter()-started:.0f}s", flush=True)

    scored = [r for r in rows if r["goal_valid"]]
    if not scored:
        raise SystemExit(f"{a.episodes} 个回合的目标都没达到门槛（与“不动”差得不够远）：时域 --horizon {a.horizon} 太短？")

    scale = 1. if relative_gap else (48. if spec.system in ("double_well", "multi_particle", "landscape", "push") else 27.)
    def stats(name):
        e = np.array([r[name] for r in scored]); z = np.array([r["zero"] for r in scored])
        ok = (e <= a.success*np.maximum(z, 1e-9))
        return dict(success_rate=float(ok.mean()), mean_error=float(e.mean()),
                    mean_error_px=None if relative_gap else float(e.mean()*scale), relative_to_zero=float((e/np.maximum(z, 1e-9)).mean()))
    summary_name = {"advection_diffusion": "16×16 平均浓度场", "gray_scott": "U、V 的 16×16 池化场（512 维）",
                    "push": "方块位姿 (b_x, b_y, size/2·θ)"}.get(spec.system, "受控小球位置")
    report = dict(kind="planning", job=str(job_dir), condition=condition, train_dataset=base.tag, dataset=spec.tag,
                  seed=job["seed"], horizon_ticks=a.horizon, replan_ticks=a.replan, state_summary=summary_name,
                  cem=dict(samples=a.samples, oracle_samples=a.oracle_samples, iterations=a.iterations, elites=a.elites), success_rule=f"误差 ≤ {a.success} × 不动时的误差",
                  model=stats("model"), oracle=None if a.no_baselines else stats("oracle"),
                  random=None if a.no_baselines else stats("random"),
                  zero=dict(mean_error=float(np.mean([r["zero"] for r in scored]))), episodes=rows,
                  goals=dict(valid=len(scored), invalid=len(rows)-len(scored),
                             rule="与“不动”的差 ≥ 0.05 × 目标范数（场）或 ≥ 0.15（其他）；20 次重抽都不够的回合不计分"),
                  encoder="fake" if a.fake_encoder else "vjepa2-vitg", scientific_conclusions_verified=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("condition", "dataset", "model", "oracle", "random", "zero")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
