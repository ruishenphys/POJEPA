"""CPU 对比：动力学 forcing_space="modal"（v7 现有模型）vs "token"（导师 2026-11-08 建议）。

数据：v7 仿真器真实轨迹画面（默认 double_well，200 条划分），用确定性随机 patch 投影 + LN 当编码器
（tools/encode_v7.py --fake-encoder 同一个，1408 维，仅开发用，不是 V-JEPA 特征）。σ_z 按 v7 标定规则在
这批特征上重标。两个模型除动力学右端外完全相同（同协议 v4 覆盖层：K=16、编码器建图、残差读出、学习率日程、
梯度裁剪），只用 JEPA 损失训练（no_consistency），以隔离动力学改动的影响。

报告：
  - 验证集 JEPA 误差（按时长 1/2/4/8 周期）与恒等基线（Ẑ=Z）
  - 规范敏感度：把图谱特征向量随机翻符号后，预测位移 Ẑ−Z 的相对变化（理论上 token 版为 0）
用法（po-jepa 根目录）：
  python tools/token_forcing/compare_cpu.py --steps 600 --output <目录>
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"tools")]

ap = argparse.ArgumentParser()
ap.add_argument("--system", default="double_well")
ap.add_argument("--count", type=int, default=200)
ap.add_argument("--train-ids", type=int, default=80, help="训练段前多少条")
ap.add_argument("--val-ids", type=int, default=20)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--seeds", default="1703")
ap.add_argument("--spaces", default="modal,token")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--cache", type=Path, required=True, help="假编码特征缓存目录（可复用）")
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
os.environ["PO_JEPA_DATASET"] = f"{args.system}:{args.count}"

import numpy as np
import torch

from po_jepa import datasets, systems
from po_jepa.dynamics import DynamicsConfig, IntegratorConfig, ModalDynamics
from po_jepa.geometry import regular_grid
from po_jepa.losses import LossConfig, jepa_distance
from po_jepa.model import POJEPA, ProjectionConfig
from po_jepa.runtime import cache, checks, data
from po_jepa.spectral import GraphConfig, GraphSpectrum, SpectralResult
from encode_v7 import _fake_encoder

torch.set_num_threads(args.threads)
spec = datasets.active()
train_ids = list(spec.split_range("train"))[:args.train_ids]
val_ids = list(spec.split_range("validation"))[:args.val_ids]
grid = regular_grid(16, 16)


# ---------- 特征 ----------
def features(tid):
    path = args.cache/f"{args.system}-{tid:04d}.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _, frames = systems.get().frames(tid)
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()/255.
        with torch.no_grad():
            z = _fake_encoder(torch)(x)
        np.save(path, z.numpy().astype(np.float16))
    return torch.from_numpy(np.load(path).astype(np.float32))


t0 = time.time()
FEAT = {tid: features(tid) for tid in train_ids+val_ids}
print(f"特征就绪 {len(FEAT)} 条，{time.time()-t0:.0f}s", flush=True)
CONTROLS = {tid: systems.commanded_controls(tid) for tid in FEAT}


def window(tid, start, duration):
    c = CONTROLS[tid]
    m = dict(trajectory_id=tid, controls=dict(starts_ticks=c.starts_ticks.tolist(), ends_ticks=c.ends_ticks.tolist(),
                                              values=c.values.tolist(), units="simulation_position_per_second"))
    ctl = cache._window_controls(m, start, float(duration), "cpu")
    return FEAT[tid][start], FEAT[tid][start+int(duration)], ctl


def make_batch(rows):
    zs, ts, cs = zip(*(window(*r) for r in rows))
    return (torch.stack(zs), torch.stack(ts), data.stack_controls(list(cs)),
            torch.tensor([float(r[2]) for r in rows], dtype=torch.float64))


# ---------- 协议与 σ_z ----------
cfg = checks.load_protocol()
m = cfg["model"]
g = torch.Generator().manual_seed(m["graph"]["q_seed"])
u, r = torch.linalg.qr(torch.randn(1408, m["graph"]["q_dim"], generator=g, dtype=torch.float64))
q = (u*torch.where(r.diagonal() < 0, -1., 1.)).T
off = ~torch.eye(256, dtype=torch.bool)
pairs = []
for tid in [train_ids[int(i*len(train_ids)/8)] for i in range(8)]:
    for tick in (0, 10, 20):
        zq = FEAT[tid][tick].double()@q.T
        pairs.append((zq[:, None]-zq[None]).square().sum(-1)[off])
sigma_z = float(torch.cat(pairs).median().sqrt())
graph_cfg = dict(m["graph"], sigma_z=sigma_z)
print(f"σ_z={sigma_z:.4f}", flush=True)
loss_cfg = LossConfig(**{k: cfg["loss"][k] for k in LossConfig.__dataclass_fields__})


def build(space, seed):
    torch.manual_seed(seed)
    return POJEPA(ProjectionConfig(**m["projection"]), GraphSpectrum(GraphConfig(**graph_cfg)),
                  ModalDynamics(DynamicsConfig(**{**m["dynamics"], "forcing_space": space})), IntegratorConfig(**m["integrator"]))


# ---------- 验证集与规范扰动 ----------
DURATIONS = (1., 2., 4., 8.)
val_rows = [(tid, s, d) for tid in val_ids for s in (0, 12) for d in DURATIONS]
VAL = [make_batch(val_rows[i:i+8]) for i in range(0, len(val_rows), 8)]


class SignFlip:
    """包装 lowest_eigenpairs：对每个样本的每个模态独立随机翻符号（同一子空间内的合法另一种选取）。"""
    def __init__(self, spectrum, seed):
        self.spectrum, self.g = spectrum, torch.Generator().manual_seed(seed)
        self.original = spectrum.lowest_eigenpairs

    def __call__(self, laplacian, *, k, model_device):
        s = self.original(laplacian, k=k, model_device=model_device)
        flip = torch.where(torch.rand(s.basis_cpu.shape[0], 1, k, generator=self.g) < .5, -1., 1.).double()
        b = s.basis_cpu*flip
        return SpectralResult(s.eigenvalues_cpu, b, s.eigenvalues, b.to(device=s.basis.device, dtype=torch.float32), s.diagnostics)


def evaluate(model):
    model.eval()
    per = {d: [] for d in DURATIONS}; ident = {d: [] for d in DURATIONS}; gauge = []
    with torch.no_grad():
        for z, target, ctl, dur in VAL:
            pred = model(z, ctl, dur, grid)
            err = jepa_distance(pred, target, config=loss_cfg).per_sample
            idn = jepa_distance(z, target, config=loss_cfg).per_sample
            for i, d in enumerate(dur.tolist()):
                per[d].append(float(err[i])); ident[d].append(float(idn[i]))
            flipper = SignFlip(model.spectrum, 0)
            model.spectrum.lowest_eigenpairs = flipper
            try:
                pred2 = model(z, ctl, dur, grid)
            finally:
                del model.spectrum.lowest_eigenpairs
            move = (pred-z).flatten(1).norm(dim=1)
            gauge += ((pred2-pred).flatten(1).norm(dim=1)/move.clamp_min(1e-12)).tolist()
    model.train()
    mean = lambda v: sum(v)/len(v)
    return dict(jepa=mean([x for d in DURATIONS for x in per[d]]), by_duration={str(int(d)): mean(per[d]) for d in DURATIONS},
                identity=mean([x for d in DURATIONS for x in ident[d]]),
                identity_by_duration={str(int(d)): mean(ident[d]) for d in DURATIONS},
                gauge_rel_change_median=float(np.median(gauge)), gauge_rel_change_max=float(max(gauge)))


def lr_at(step):
    lc = cfg["loss"]; warm = max(1, args.steps//20); top, low = 1e-3, lc["lr_min"]
    if step < warm:
        return 1e-5+(top-1e-5)*step/warm
    p = (step-warm)/max(1, args.steps-warm)
    return low+(top-low)*.5*(1+math.cos(math.pi*p))


report = dict(system=args.system, encoder="fake_patch_projection_ln_1408 (development only)", sigma_z=sigma_z,
              steps=args.steps, batch=args.batch, train_trajectories=len(train_ids), val_windows=len(val_rows), runs={})
args.output.mkdir(parents=True, exist_ok=True)
for seed in [int(s) for s in args.seeds.split(",")]:
    for space in args.spaces.split(","):
        model = build(space, seed); model.train()
        n_params = sum(p.numel() for p in model.parameters())
        dyn_params = sum(p.numel() for p in model.dynamics.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0)
        gen = torch.Generator().manual_seed(seed+7)
        hist = [dict(step=0, **evaluate(model))]
        print(space, seed, "params", n_params, "dyn", dyn_params, hist[-1], flush=True)
        log = []; start = time.time()
        for step in range(args.steps):
            rows = [(train_ids[int(torch.randint(len(train_ids), (), generator=gen))], int(torch.randint(21, (), generator=gen)),
                     DURATIONS[int(torch.randint(4, (), generator=gen))]) for _ in range(args.batch)]
            z, target, ctl, dur = make_batch(rows)
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            loss = jepa_distance(model(z, ctl, dur, grid), target, config=loss_cfg).mean
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["loss"]["grad_clip_norm"]))
            opt.step()
            log.append(dict(step=step+1, jepa=float(loss), grad_norm=gn))
            if (step+1) % max(1, args.steps//6) == 0 or step+1 == args.steps:
                hist.append(dict(step=step+1, seconds=time.time()-start, **evaluate(model)))
                print(space, seed, hist[-1], flush=True)
        report["runs"][f"{space}-{seed}"] = dict(space=space, seed=seed, parameters=n_params, dynamics_parameters=dyn_params,
                                                 val=hist, train=log, seconds=time.time()-start)
        (args.output/"result.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved", args.output/"result.json")
