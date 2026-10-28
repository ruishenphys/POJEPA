"""v4 生产配置（E=1408，16×16/12×12/8×8）初始化检查：Φ₀ 恒等、各网格 K、谱是否连通。替身特征，CPU。"""
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src")]
from po_jepa.actions import ControlSpec, from_physical_segments
from po_jepa.geometry import build_spatial_transfer, regular_grid, apply_spatial_transfer
from po_jepa.runtime import checks
torch.set_grad_enabled(False)
cfg = checks.load_protocol(); model = checks.build_full_model(cfg, torch.device("cpu"))
spec = ControlSpec(2, .25, ("u_x", "u_y"), ("p/s", "p/s"), (-.5, -.5), (.5, .5), (1., 1.), (0., 0.), "check")
g = torch.Generator().manual_seed(0)
z16 = (torch.randn(1, 1, 1408, generator=g)*2.5+torch.randn(2, 256, 1408, generator=g)*1.6).float()
grid16 = regular_grid(16, 16)
ctl = from_physical_segments(starts_seconds=torch.zeros(2, 1, dtype=torch.float64), ends_seconds=torch.full((2, 1), .25, dtype=torch.float64),
    u_values=torch.tensor([[[.1, .2]], [[-.3, 0.]]]), n_segments=torch.ones(2, dtype=torch.int64), segment_valid=torch.ones(2, 1, dtype=torch.bool),
    origin_seconds=torch.zeros(2, dtype=torch.float64), control_spec=spec, source_ids=("a", "b"))
out = dict(config=dict(graph=cfg["model"]["graph"], projection=cfg["model"]["projection"], loss=cfg["loss"]))
out["phi0_equals_input_exactly"] = bool(torch.equal(model(z16, ctl, torch.zeros(2, dtype=torch.float64), grid16), z16))
grids = {}
for h in (16, 12, 8):
    t = build_spatial_transfer(grid16, target_height=h, target_width=h)
    zh = apply_spatial_transfer(z16, t) if h != 16 else z16
    q = model.prepare_query(zh, t.target if h != 16 else grid16)
    lam = q.spectrum.eigenvalues_cpu
    grids[f"{h}x{h}"] = dict(K=int(q.spectrum.basis.shape[-1]), near_zero_eigs=int((lam < 1e-6).sum(-1).max()),
                             lambda_K=float(lam[:, -1].mean()), cutoff_gap=float(q.spectrum.diagnostics.cutoff_gap.mean()))
out["grids"] = grids
print(json.dumps(out, ensure_ascii=False, indent=1))
(Path(__file__).with_name("init_check_v4_result.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
