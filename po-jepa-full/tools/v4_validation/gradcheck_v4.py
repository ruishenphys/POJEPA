"""v4 有限差分梯度检验（CPU、float64 诊断进程）：完整损失 = JEPA + w·(λ_t L_time + λ_s L_space)，LN 空间、残差读出、编码器特征建图。

用法：python tools/v4_validation/gradcheck_v4.py   （在 po-jepa 根目录下运行）
只在本进程内把 po_jepa 模块里的 torch.float32 指向 float64，以消除单精度差分噪声；不改任何文件。
通过标准看“冻结谱”有限差分（autograd 的定义就是谱视为常数）。“实时重算谱”只在时间一致性第二段不同：
那一段从中间预测出发重建图，中间预测依赖参数，而图按设计停梯度（detached），所以实时差分会多出这部分，属预期。
"""
import json, math, sys, types
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src")]
torch.set_num_threads(4)
import po_jepa.actions, po_jepa.dynamics, po_jepa.geometry, po_jepa.losses, po_jepa.model, po_jepa.spectral
class P(types.ModuleType):
    def __getattr__(s, k): return getattr(torch, k)
proxy = P("torchproxy"); proxy.float32 = torch.float64
for name, m in list(sys.modules.items()):
    if name.startswith("po_jepa") and getattr(m, "torch", None) is torch: m.torch = proxy
from po_jepa.actions import ControlSpec, from_physical_segments
from po_jepa.dynamics import DynamicsConfig, IntegratorConfig, ModalDynamics
from po_jepa.geometry import build_spatial_transfer, regular_grid
from po_jepa.losses import (LossConfig, jepa_distance, temporal_predictions, temporal_consistency_loss,
                            spatial_predictions, spatial_consistency_loss, weighted_total_loss)
from po_jepa.model import POJEPA, ProjectionConfig
from po_jepa.spectral import GraphConfig, GraphSpectrum
import yaml
E = 16
cfg = yaml.safe_load((ROOT/"configs/train/default.yaml").read_text(encoding="utf-8"))["model"]
torch.manual_seed(1703)
model = POJEPA(ProjectionConfig(E, 64, readout="residual"),
               GraphSpectrum(GraphConfig(16, 1703, .125, 4., 8, 1e-10, 1e-8, affinity_source="encoder", feature_dim=E, fixed_modes=True)),
               ModalDynamics(DynamicsConfig(**cfg["dynamics"])), IntegratorConfig(**cfg["integrator"])).double()
spec = ControlSpec(2, .25, ("u_x", "u_y"), ("p/s", "p/s"), (-.5, -.5), (.5, .5), (1., 1.), (0., 0.), "gradcheck")
B = 2
controls = from_physical_segments(starts_seconds=torch.zeros(B, 1, dtype=torch.float64), ends_seconds=torch.tensor([[.25], [.5]], dtype=torch.float64),
    u_values=torch.tensor([[[.2, -.1]], [[-.3, .4]]], dtype=torch.float64), n_segments=torch.ones(B, dtype=torch.int64),
    segment_valid=torch.ones(B, 1, dtype=torch.bool), origin_seconds=torch.zeros(B, dtype=torch.float64), control_spec=spec, source_ids=("a", "b"))
g = torch.Generator().manual_seed(5)
raw = (torch.randn(1, 1, E, generator=g, dtype=torch.float64)*2+torch.randn(B, 64, E, generator=g, dtype=torch.float64))
target = raw+.3*torch.randn(raw.shape, generator=g, dtype=torch.float64)
grid = regular_grid(8, 8); transfer = build_spatial_transfer(grid, target_height=4, target_width=4)
tau = torch.tensor([.5, 1.], dtype=torch.float64); sigma = tau.clone()

def total(cfg_, scale):
    t = temporal_predictions(model, raw, controls, tau, sigma, grid)
    j = jepa_distance(t.direct, target, config=cfg_)
    s = spatial_predictions(model, raw, controls, tau+sigma, transfer)
    return weighted_total_loss(j, temporal_consistency_loss(t, config=cfg_), spatial_consistency_loss(s, config=cfg_),
                               config=cfg_, consistency_scale=scale)

orig = po_jepa.spectral.GraphSpectrum.lowest_eigenpairs
class Freeze:
    mode, cache, i = "live", [], 0
    def __call__(s, self_, lap, **kw):
        if s.mode == "live": return orig(self_, lap, **kw)
        if s.mode == "record": r = orig(self_, lap, **kw); s.cache.append(r); return r
        r = s.cache[s.i]; s.i += 1; return r
F = Freeze(); po_jepa.spectral.GraphSpectrum.lowest_eigenpairs = lambda self_, lap, **kw: F(self_, lap, **kw)

def run(name, cfg_, scale):
    model.zero_grad(set_to_none=True)
    F.mode, F.cache = "record", []
    L = total(cfg_, scale); L.total.backward()
    out = dict(case=name, losses=dict(jepa=float(L.jepa.mean), time=float(L.time.mean), space=float(L.space.mean), total=float(L.total)), groups={})
    torch.manual_seed(11)
    groups = {"P_out": [], "P_in": [], "dynamics": []}
    for n, p in model.named_parameters():
        groups["P_out" if n.startswith("p_out") else "P_in" if n.startswith("p_in") else "dynamics"].append(p)
    for gname, ps in groups.items():
        vs = [torch.randn_like(p) for p in ps]; nrm = math.sqrt(sum(float((v*v).sum()) for v in vs)); vs = [v/nrm for v in vs]
        analytic = sum(float((p.grad*v).sum()) for p, v in zip(ps, vs))
        res = dict(autograd=analytic)
        for mode in ("frozen", "live"):
            h = 1e-5; vals = []
            for sgn in (1, -1):
                with torch.no_grad():
                    for p, v in zip(ps, vs): p.add_(sgn*h*v)
                F.mode = "replay" if mode == "frozen" else "live"; F.i = 0
                with torch.no_grad(): vals.append(float(total(cfg_, scale).total))
                with torch.no_grad():
                    for p, v in zip(ps, vs): p.sub_(sgn*h*v)
            res["fd_"+mode] = (vals[0]-vals[1])/(2*h)
        res["rel_err_frozen"] = abs(res["fd_frozen"]-analytic)/max(abs(analytic), 1e-12)
        res["rel_err_live"] = abs(res["fd_live"]-analytic)/max(abs(analytic), 1e-12)
        out["groups"][gname] = res
    return out

ln = LossConfig(1., 1e-5, 1e-8, 1e-8, .07, .3, consistency_metric="ln_mean", consistency_warmup_updates=200)
results = [run("仅 JEPA（预热第 0 步，w=0）", ln, 0.), run("完整损失（w=1，λ_t=0.07，λ_s=0.3，LN 空间）", ln, 1.)]
worst = max(r["groups"][g]["rel_err_frozen"] for r in results for g in r["groups"])
summary = dict(results=results, worst_relative_error=worst, passed=worst < 1e-5)
(Path(__file__).with_name("gradcheck_v4_result.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=1))
