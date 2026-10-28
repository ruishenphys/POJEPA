"""v4 CPU 短训练冒烟实验（替身特征，不读真实缓存、不启动正式训练）。

替身数据：每帧 Z = 共享通道均值 + 平滑位置场 + 亮斑(中心 c) + 噪声，总 std≈3（与真实 Z 一致）；
亮斑中心按已知动作移动 c(t+τ) = c(t) + 0.08·u·τ，因此“未来特征”确实依赖起点与动作。
条件：po_jepa_full（λ 按 ρ=0.2 梯度比在初始化时标定，前 10% 预热）、po_jepa_no_consistency（λ=0）、identity（不训练）。
用法（po-jepa 根目录）：python tools/v4_validation/short_train_v4.py [--steps 60]
"""
import argparse, json, math, sys, time
from dataclasses import replace
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src")]
from po_jepa.actions import ControlSpec, from_physical_segments
from po_jepa.dynamics import DynamicsConfig, IntegratorConfig, ModalDynamics
from po_jepa.geometry import build_spatial_transfer, regular_grid
from po_jepa.losses import (LossConfig, jepa_distance, temporal_predictions, temporal_consistency_loss,
                            spatial_predictions, spatial_consistency_loss, weighted_total_loss)
from po_jepa.model import POJEPA, ProjectionConfig
from po_jepa.runtime import checks, training
from po_jepa.spectral import GraphConfig, GraphSpectrum

ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=60); ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--rho", type=float); ap.add_argument("--terms", default="time,space", help="完整条件中启用的一致性项"); ap.add_argument("--tag", default=""); ap.add_argument("--full-only", action="store_true")
args = ap.parse_args()
torch.set_num_threads(4)
cfg = checks.load_protocol(); m = cfg["model"]; E = 1408
grid = regular_grid(16, 16); transfer = build_spatial_transfer(grid, target_height=8, target_width=8)
coords = grid.coords.float()
spec = ControlSpec(2, .25, ("u_x", "u_y"), ("p/s", "p/s"), (-.5, -.5), (.5, .5), (1., 1.), (0., 0.), "surrogate")

g0 = torch.Generator().manual_seed(0)
shared = torch.randn(E, generator=g0)
om = torch.randn(32, 2, generator=g0)/.3; ph = torch.rand(32, generator=g0)*2*math.pi
pos = (torch.cos(coords@om.T+ph)*math.sqrt(2/32))@torch.randn(32, E, generator=g0)
spot_dir = torch.randn(E, generator=g0)

def frame(center, g):
    bump = torch.exp(-((coords[None]-center[:, None])**2).sum(-1)/(2*.06**2))[..., None]
    var = pos[None]+3*bump*spot_dir+.5*torch.randn(center.shape[0], 256, E, generator=g)
    return (math.sqrt(.7)*shared+math.sqrt(.3)*var/1.25)*3

durations = (.5, 1., 2., 4.)
def batch(g):
    b = args.batch
    c = torch.rand(b, 2, generator=g)*.4+.3
    u = torch.rand(b, 2, generator=g)-.5
    tau = torch.tensor([durations[int(i)] for i in torch.randint(4, (b,), generator=g)], dtype=torch.float64)
    sig = torch.tensor([durations[int(i)] for i in torch.randint(4, (b,), generator=g)], dtype=torch.float64)
    total = tau+sig
    ctl = from_physical_segments(starts_seconds=torch.zeros(b, 1, dtype=torch.float64), ends_seconds=(total*.25)[:, None],
        u_values=u[:, None, :].float(), n_segments=torch.ones(b, dtype=torch.int64), segment_valid=torch.ones(b, 1, dtype=torch.bool),
        origin_seconds=torch.zeros(b, dtype=torch.float64), control_spec=spec, source_ids=tuple(f"w{i}" for i in range(b)))
    z = frame(c, g); target = frame(c+.08*u*total[:, None].float(), g)
    return z, target, ctl, tau, sig

def build():
    torch.manual_seed(cfg["training"]["initial_model_seed"])
    return POJEPA(ProjectionConfig(**m["projection"]), GraphSpectrum(GraphConfig(**m["graph"])),
                  ModalDynamics(DynamicsConfig(**m["dynamics"])), IntegratorConfig(**m["integrator"]))

def losses(model, data, lc, scale):
    z, target, ctl, tau, sig = data
    t = temporal_predictions(model, z, ctl, tau, sig, grid)
    j = jepa_distance(t.direct, target, config=lc)
    s = spatial_predictions(model, z, ctl, tau+sig, transfer)
    return weighted_total_loss(j, temporal_consistency_loss(t, config=lc), spatial_consistency_loss(s, config=lc), config=lc, consistency_scale=scale)

val = [batch(torch.Generator().manual_seed(10_000+i)) for i in range(3)]
def evaluate(model, lc):
    with torch.no_grad():
        rows = [losses(model, d, lc, 1.) for d in val]
    return {k: sum(float(getattr(r, k).mean) for r in rows)/len(rows) for k in ("jepa", "time", "space")}
identity = sum(float(jepa_distance(d[0], d[1], config=LossConfig(1., 1e-5, 1e-8, 1e-8, 0., 0.)).mean) for d in val)/len(val)

base = LossConfig(**{k: cfg["loss"][k] for k in LossConfig.__dataclass_fields__})
# λ 标定：初始化状态、前 4 个训练 batch，ρ=0.2（与 run.py calibrate 同一规则）
rho = cfg["calibration"]["rho"] if args.rho is None else args.rho; cal_model = build(); cal_model.train(); params = list(cal_model.parameters())
norms = {"jepa": [], "time": [], "space": []}
for i in range(4):
    z, target, ctl, tau, sig = batch(torch.Generator().manual_seed(i))
    t = temporal_predictions(cal_model, z, ctl, tau, sig, grid); s = spatial_predictions(cal_model, z, ctl, tau+sig, transfer)
    parts = dict(jepa=jepa_distance(t.direct, target, config=base).mean, time=temporal_consistency_loss(t, config=base).mean,
                 space=spatial_consistency_loss(s, config=base).mean)
    for k, v in parts.items():
        gr = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
        norms[k].append(math.sqrt(sum(float(x.double().square().sum()) for x in gr if x is not None)))
mean = {k: sum(v)/len(v) for k, v in norms.items()}
lam_t, lam_s = rho*mean["jepa"]/mean["time"], rho*mean["jepa"]/mean["space"]
terms = set(args.terms.split(","))
lam_t, lam_s = (lam_t if "time" in terms else 0.), (lam_s if "space" in terms else 0.)
warm = max(1, args.steps//10)
conditions = {"po_jepa_full": replace(base, time_weight=lam_t, space_weight=lam_s, consistency_warmup_updates=warm),
              "po_jepa_no_consistency": replace(base, time_weight=0., space_weight=0., consistency_warmup_updates=0)}
if args.full_only: conditions.pop("po_jepa_no_consistency")
report = dict(terms=sorted(terms), steps=args.steps, batch=args.batch, identity_val_jepa=identity, calibration=dict(rho=rho, grad_norm_mean=mean, lambda_t=lam_t, lambda_s=lam_s),
              graph=dict(sigma_x=m["graph"]["sigma_x"], sigma_z=m["graph"]["sigma_z"], K=m["graph"]["max_modes"]), conditions={})
print("identity val JEPA", identity, "lambda", lam_t, lam_s, flush=True)
for name, lc in conditions.items():
    model = build(); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0)
    hist = [dict(step=0, **evaluate(model, lc))]
    train_log = []; t0 = time.time(); g = torch.Generator().manual_seed(1)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        L = losses(model, batch(g), lc, training.consistency_warmup_scale(lc, step))
        L.total.backward(); opt.step()
        train_log.append(dict(step=step+1, jepa=float(L.jepa.mean), time=float(L.time.mean), space=float(L.space.mean), total=float(L.total)))
        if (step+1) % max(1, args.steps//4) == 0:
            hist.append(dict(step=step+1, **evaluate(model, lc)))
            print(name, hist[-1], f"{time.time()-t0:.0f}s", flush=True)
    report["conditions"][name] = dict(val=hist, train=train_log, seconds=time.time()-t0)
out = Path(__file__).with_name(f"short_train_v4_result{args.tag}.json")
out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved", out)
