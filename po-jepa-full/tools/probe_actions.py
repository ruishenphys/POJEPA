"""v7 诊断：动作在冻结特征里到底有多大信号？（回答“delta_act≈1e-6 是数据问题还是模型问题”）

对测试集里的 (轨迹, 起点, 时长)，用和评测 delta_act 完全相同的 donor（测试集下一条轨迹、同起点同时长的控制），
从真实的起点状态出发、换成 donor 的控制重新仿真，渲染终点画面，用同一个冻结编码器、同一个预处理随机种子编码，
得到“如果换了动作，终点特征本来会是什么”(Z_cf)。然后报告：
- identity 误差   d(Z_t, Z_{t+τ})                      ：完全不预测的误差
- 动作上限 Δ_act* d(Z_cf, Z_{t+τ})                     ：一个完美模型的 delta_act（换动作后误差应该涨多少）
- 像素位移        |q_cf − q_true|、|q_{t+τ} − q_t|        ：同一件事在画面里有多大
d 是 PO-JEPA 统一的 JEPA 距离（双侧无仿射 LayerNorm 后的 MAE）。
*_px 列：粒子类场景是状态的 L2 × 每单位像素数；advection_diffusion 是 16×16 平均浓度场的 L2；S21b 新场景 push 是推杆和
方块中心 4 个位置坐标的 L2 × 48（不含朝向角），gray_scott 是 U、V 池化场（512 维）的 L2（不是像素，注入点不计入）；
S25a 二阶场景只比画面上看得到的位置（构形，见 po_jepa.configuration）：pendulum 摆锤 × 80、cart_pole 小车和摆锤 × 36、
spring_chain 和 bouncing_balls 全部质点/球 × 48，wave 是 u 的 16×16 池化场（256 维）的 L2（不是像素）。
二阶场景现在按单帧复制（dup）编码，和 encode_v7 一致：特征里看不到速度；两帧（pair）编码在 S25b 接（等 S22 探针结果）。
若 Δ_act* 和 identity 误差同量级，而训练好的模型 delta_act≈1e-6，说明是模型/训练没学会用动作，不是数据里没有动作信号。

用法（GPU 上几分钟；和 encode_v7.py 用同一个编码器与权重）：
  python tools/probe_actions.py --system double_well --trajectories 200 \
      --upstream-root ../vjepa2 --checkpoint ../checkpoints/vjepa2-vitg-fpc64-256.pth \
      --features-root ../features-v7/double_well-200 --output ../results-v7/probe-actions-double_well.json --device cuda:0
--features-root 可选：给了就核对真实终点帧的重新编码与缓存一致（检查编码可复现），不给就只用重新编码的特征。
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DURATIONS = (1, 2, 4, 8)
NEW_SYSTEMS = ("gray_scott", "push")      # S21b：走 System.variant_trajectory 的续算接口（基础场景也走，没有 v6 原实现）
# S25a：二阶场景（variants.SECOND_ORDER）也走同一个续算接口（SecondOrderSystem：完整状态续算、hidden()、按 tick 画）
_FULL = {}                                # S21b：新场景每条轨迹的整条仿真只算一次（gray_scott 约 5 秒/条）


def motion(system_name, a, b, scale, params=None):
    """*_px 列的距离（见文件头）。a、b 是 frames() 的隐藏状态；params：二阶场景的物理参数 dict(v.params)（构形要用）。"""
    import numpy as np
    from po_jepa import configuration, variants
    if system_name in variants.SECOND_ORDER:
        if params is None:
            raise ValueError(f"{system_name} 是二阶场景：motion 要给物理参数 params")
        return configuration.motion(system_name, a, b, params)
    if system_name == "push":
        return float(np.linalg.norm(a[:4]-b[:4])*48.)
    if system_name == "gray_scott":
        return float(np.linalg.norm(a[:512]-b[:512]))
    return float(np.linalg.norm(a-b)*scale)


def _parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", required=True)
    p.add_argument("--trajectories", type=int, required=True)
    p.add_argument("--variant", default="", help="可调数据集的调节项（与 encode_v7.py 相同写法）")
    p.add_argument("--upstream-root", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--features-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-trajectories", type=int, default=20, help="最多用多少条测试轨迹（每条 2 个起点 × 4 个时长）")
    p.add_argument("--fake-encoder", action="store_true", help="仅 CPU 流程测试")
    return p.parse_args()


def counterfactual_end(system, simulation, tid, donor, start, duration):
    """从 tid 的真实起点状态出发，用 donor 在 [start, start+duration) 的控制仿真 → (真实终点状态, 反事实终点状态, 反事实终点画面)。"""
    import numpy as np
    from po_jepa import datasets, systems
    from po_jepa import variants
    v = datasets.active().knobs
    if system.name in NEW_SYSTEMS+variants.SECOND_ORDER:       # S21b；S25a 二阶场景同一条路径：从第 start 帧的完整状态续算；隐藏状态用 hidden() 换成 frames() 的那种
        if (tid, datasets.active().tag) not in _FULL:
            _FULL.clear(); _FULL[tid, datasets.active().tag] = system.variant_trajectory(tid, v)
        full = _FULL[tid, datasets.active().tag]
        q = system.variant_trajectory(tid, v, controls=systems.commanded_controls(donor), start=start,
                                      state=full[start], ticks=duration)[-1]
        return (system.hidden(full[start+duration][None])[0], system.hidden(q[None])[0],
                system.draw(q[None], tid, v, ticks=[start+duration])[0])
    if not v.is_default:      # v7 可调数据集：同一套可调仿真与渲染
        controls = systems.commanded_controls(donor)
        if system.name == "advection_diffusion":
            fields = system.variant_fields(tid, v)
            field = system.variant_fields(tid, v, controls=controls, start=start, field=fields[start], ticks=duration)[-1]
            pool = lambda f: f.reshape(16, 16, 16, 16).mean((1, 3)).reshape(-1)
            return pool(fields[start+duration]), pool(field), system.draw_fields(field[None], tid, v, ticks=[start+duration])[0]
        states = system.variant_trajectory(tid, v)
        q = system.variant_trajectory(tid, v, controls=controls, start=start, state=states[start], ticks=duration)[-1]
        return states[start+duration], q, system.draw(q[None], tid, v, ticks=[start+duration])[0]   # S21a：movers 要知道 tick
    controls = simulation.sample_controls(donor)
    if system.name == "advection_diffusion":
        fields = system.fields(tid)
        k = np.fft.fftfreq(fields.shape[-1], d=1./fields.shape[-1])*2*np.pi
        ky, kx = np.meshgrid(k, k, indexing="ij")
        spectrum = np.fft.fft2(fields[start]); seg = 0
        for tick in range(start, start+duration):
            while controls.ends_ticks[seg] <= tick: seg += 1
            wx, wy = system.FLOW*controls.values[seg]
            spectrum = spectrum*np.exp((-system.D*(kx**2+ky**2)-1j*(kx*wx-ky*wy))*.25)
        field = np.fft.ifft2(spectrum).real
        pool = lambda f: f.reshape(16, 16, 16, 16).mean((1, 3)).reshape(-1)
        return pool(fields[start+duration]), pool(field), system.render_fields(field[None])[0]
    states = system.reference_trajectory(tid)
    field = system.field if hasattr(system, "field") else simulation.vector_field
    q = states[start].copy(); h = .25/128; seg = 0
    for tick in range(start, start+duration):
        while controls.ends_ticks[seg] <= tick: seg += 1
        u = controls.values[seg]
        for _ in range(128):
            k1 = field(q, u); k2 = field(q+h*k1/2, u); k3 = field(q+h*k2/2, u); k4 = field(q+h*k3, u)
            q = q+h*(k1+2*k2+2*k3+k4)/6
    return states[start+duration], q, system.render(q[None])[0]


def main():
    a = _parse()
    os.environ["PO_JEPA_DATASET"] = f"{a.system}{'~'+a.variant if a.variant else ''}:{a.trajectories}"
    sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"tools"))
    import numpy as np
    import torch
    from torch.nn import functional as F
    from po_jepa import datasets, simulation, systems, variants
    from po_jepa.runtime import cache
    import encode_v7

    spec = datasets.active(); system = systems.get(); device = torch.device(a.device)
    if spec.knobs.horizon != systems.TICKS:
        # S24a：探针按 33 帧取起点和时长（start+tau ≤ 32）；长时程的探针在第 6 阶段接
        raise SystemExit(f"horizon={spec.knobs.horizon} 是长时程数据集：动作探针还没接（S24a 只做数据生成）")
    if spec.knobs.changes_timing():
        # S24b：探针按每 tick 一帧、TrajectoryControls 取起点和时长；时间采样的探针在第 6 阶段接
        raise SystemExit(f"{spec.scene} 带时间采样调节项：动作探针还没接（S24b 只做数据生成）")
    if spec.knobs.changes_space():
        # S24c：探针按单帧重画（systems.draw，不经过相机）；空间采样的探针在第 6 阶段接
        raise SystemExit(f"{spec.scene} 带空间采样调节项：动作探针还没接（S24c 只做数据生成）")
    if a.fake_encoder:
        encoder = encode_v7._fake_encoder(torch)
    else:
        if a.upstream_root is None or a.checkpoint is None:
            raise SystemExit("需要 --upstream-root 和 --checkpoint（与 encode_v7.py 相同）")
        from po_jepa import vjepa
        arch = vjepa.EncoderArchitectureSpec(vjepa.UPSTREAM_COMMIT, "configs/train/vitg16/droid-256px-8f.yaml",
            "vit_giant_xformers", 256, 16, 2, 512, True, True, False, True, True, True)
        build = vjepa.build_encoder_architecture(upstream_root=a.upstream_root, architecture=arch, initialization_seed=0)
        loaded = vjepa.load_local_encoder(build, checkpoint=vjepa.LocalCheckpointSpec(
            a.checkpoint, cache.CHECKPOINT_SHA, cache.PUBLICATION, "target_encoder", ("module.", "backbone."), True, (), ()))
        encoder = vjepa.FrozenFrameEncoder(loaded, preprocess=cache.preprocess_spec(), encoder_device=device, autocast_dtype=None)

    def encode(frames, tid):
        return torch.from_numpy(encode_v7._encode_trajectory(encoder, frames, tid, simulation, cache, torch, np,
                                                             33, a.fake_encoder)).float()

    def d(x, y):
        return float((F.layer_norm(x, (x.shape[-1],), eps=1e-5)-F.layer_norm(y, (y.shape[-1],), eps=1e-5)).abs().mean())

    test = list(spec.split_range("test"))[:a.max_trajectories]
    rows, started = [], time.perf_counter()
    scale = 48. if system.name == "double_well" else getattr(system, "SCALE", 1.)   # advection_diffusion：*_px 列是 16×16 平均浓度场的 L2 变化，不是像素
    params = dict(spec.knobs.params)            # S25a：二阶场景的构形要用物理参数（cart_pole 的摆长、质点数 n）
    for i, tid in enumerate(test):
        donor = datasets.donor_id(tid)
        states, frames = system.frames(tid)
        z = encode(frames, tid)                                     # [33,256,1408]
        if a.features_root is not None:
            path = a.features_root/f"{tid:04d}"/"raw_z.npy"
            if path.exists():
                stored = torch.from_numpy(np.load(path).astype(np.float32))
                gap = d(stored, z)
                if gap > 1e-2:
                    raise SystemExit(f"轨迹 {tid}: 重新编码与缓存不一致（d={gap:.3g}），检查编码器/权重/预处理")
        for start in datasets.test_start_ticks(tid):
            for tau in DURATIONS:
                if start+tau > 32:
                    continue
                true_end, cf_state, cf_frame = counterfactual_end(system, simulation, tid, donor, start, tau)
                if not np.allclose(true_end, states[start+tau], atol=1e-6):
                    raise SystemExit("真实终点重新仿真不一致")
                z_cf = encode(cf_frame[None], tid)[0]
                rows.append(dict(trajectory_id=tid, donor_id=donor, start_tick=start, duration_cycles=tau,
                                 identity=d(z[start], z[start+tau]), oracle_delta_act=d(z_cf, z[start+tau]),
                                 motion_px=motion(system.name, states[start+tau], states[start], scale, params),
                                 action_swap_px=motion(system.name, cf_state, true_end, scale, params)))
        print(f"[{spec.tag}] {i+1}/{len(test)} 条，{time.perf_counter()-started:.0f} s", flush=True)

    def summary(sel):
        keys = ("identity", "oracle_delta_act", "motion_px", "action_swap_px")
        return {k: float(np.mean([r[k] for r in sel])) for k in keys} | dict(windows=len(sel),
                oracle_over_identity=float(np.mean([r["oracle_delta_act"] for r in sel])/np.mean([r["identity"] for r in sel])))
    report = dict(system=a.system, dataset=spec.tag, encoder="fake" if a.fake_encoder else "vjepa2-vitg",
                  overall=summary(rows), by_duration={str(t): summary([r for r in rows if r["duration_cycles"] == t]) for t in DURATIONS},
                  rows=rows, note="oracle_delta_act = 完美模型的 delta_act；与训练好模型的 delta_act（评测报告）对比",
                  **({"tubelet": "dup", "motion_units": "构形（po_jepa.configuration），二阶场景单帧复制编码看不到速度"}
                     if spec.system in variants.SECOND_ORDER else {}))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(dict(overall=report["overall"], by_duration=report["by_duration"]), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
