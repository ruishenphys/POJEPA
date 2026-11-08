"""PO-JEPA 建图复核：在真实冻结特征上标定 σ_z，并给出 K=16 还是 32 的建议。

只读、CPU、约 1 分钟；不改生产代码，不训练。
定稿方案（导师确认）：建图特征 = stop-gradient 的编码器输出 Z（不再经过可训练 P_in），
W_ij = exp(-|x_i-x_j|^2/σ_x^2) * exp(-|Q Z_i - Q Z_j|^2/σ_z^2)，Q ∈ R^{16×1408} 固定随机正交（种子 1703），
σ_z 在训练集上一次标定后固定，K 跨网格相同。

用法（任选一种特征来源，在 PO-JEPA一轮验证 目录下运行）：
  python 建图与一致性诊断/real_feature_probe.py --code code-only/po-jepa --zip po-jepa-v3-handoff.zip
  python 建图与一致性诊断/real_feature_probe.py --code code-only/po-jepa --features-root <解压后的 features 目录>
可选 --checkpoint <step-0000.pt>：额外报告旧方案（Q·P_in(Z) 建图）作对照。
输出：脚本同目录 real_feature_probe_result.json，最后一行打印建议。
"""
import argparse, io, json, math, sys, zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

p = argparse.ArgumentParser()
p.add_argument("--code", required=True)
p.add_argument("--features-root")
p.add_argument("--zip")
p.add_argument("--checkpoint")
p.add_argument("--ids", default="0,20,40,60,80,100,120,140")   # 只用训练集轨迹（0-159）
p.add_argument("--ticks", default="0,10,20")
a = p.parse_args()
sys.path.insert(0, str(Path(a.code)/"src"))
from po_jepa.geometry import regular_grid
torch.set_grad_enabled(False)

def load_z(tid):
    name = f"{tid:04d}/raw_z.npy"
    if a.features_root:
        return np.load(Path(a.features_root)/name, mmap_mode="r")
    with zipfile.ZipFile(a.zip) as zf:
        member = [n for n in zf.namelist() if n.endswith("features/"+name)][0]
        return np.load(io.BytesIO(zf.read(member)))

# 固定 Q: 1408→16，与 GraphSpectrum 相同的 QR+符号规范流程
g = torch.Generator().manual_seed(1703)
u, r = torch.linalg.qr(torch.randn(1408, 16, generator=g, dtype=torch.float64))
Q_Z = (u*torch.where(r.diagonal() < 0, -1., 1.)).T
projections = {"Z_detached(定稿)": lambda z: z@Q_Z.T}
if a.checkpoint:
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    def find(d):
        if isinstance(d, dict):
            if "p_in.weight" in d: return d
            for v in d.values():
                r_ = find(v)
                if r_ is not None: return r_
    sd = find(ck)
    Wp, bp, Qx = sd["p_in.weight"].double(), sd["p_in.bias"].double(), sd["spectrum.q"].double()
    projections["P_in(Z)(旧方案)"] = lambda z: (z@Wp.T+bp)@Qx.T

def spatial(h):
    c = regular_grid(h, h).coords
    return (c[:, None]-c[None]).square().sum(-1)
SP = {16: spatial(16), 8: spatial(8), 12: spatial(12)}

def eig(fs, sx, sz, h):
    Wm = torch.exp(-SP[h]/sx**2)*torch.exp(-fs/sz**2); d = Wm.sum(-1)
    L = torch.eye(h*h, dtype=torch.float64)-Wm/d.sqrt()[:, None]/d.sqrt()[None]
    v, V = torch.linalg.eigh(L)
    return Wm, v, V

def fsq(pz): return (pz[:, None]-pz[None]).square().sum(-1)

def resize(z, h):   # 与 geometry.py 一致：8×8 平均池化，12×12 双线性 align_corners=False
    f = z.reshape(16, 16, -1).permute(2, 0, 1)[None]
    f = F.avg_pool2d(f, 16//h) if 16 % h == 0 else F.interpolate(f, size=(h, h), mode="bilinear", align_corners=False)
    return f[0].permute(1, 2, 0).reshape(h*h, -1)

frames = []
for tid in [int(x) for x in a.ids.split(",")]:
    zz = load_z(tid)
    for t in [int(x) for x in a.ticks.split(",")]:
        frames.append((tid, t, torch.from_numpy(np.array(zz[t], dtype=np.float64)),
                       torch.from_numpy(np.array(zz[t+1], dtype=np.float64))))

report = {"Z_std": float(torch.stack([f[2] for f in frames]).std()), "n_frames": len(frames), "modes": {}}
for mode, proj in projections.items():
    pairs = torch.cat([fsq(proj(f[2]))[~torch.eye(256, dtype=bool)] for f in frames])
    med = pairs.median().item(); sz_cal = math.sqrt(med)
    m = {"featsq_allpairs_quantiles(5,50,95%)": [pairs.quantile(q).item() for q in (.05, .5, .95)],
         "calibrated_sigma_z": sz_cal}
    _, v1, _ = eig(fsq(proj(frames[0][2])), .25, 1.0, 16)
    m["sigma_z=1,sigma_x=.25 近零特征值数(首帧)"] = int((v1 < 1e-6).sum())
    for sx in (.125, .25):
        rows = []
        for tid, t, zt, zn in frames:
            _, v, V = eig(fsq(proj(zt)), sx, sz_cal, 16)
            gen = torch.Generator().manual_seed(tid*100+t)
            res = {"near0": int((v < 1e-6).sum())}
            for tag, z2 in (("noise1pct", zt+0.01*zt.std()*torch.randn(zt.shape, generator=gen, dtype=torch.float64)),
                            ("next_frame", zn)):
                _, _, V2 = eig(fsq(proj(z2)), sx, sz_cal, 16)
                for K in (16, 32, 64):
                    s = torch.linalg.svdvals(V[:, :K].T@V2[:, :K])
                    res[f"{tag}_sinTheta_K{K}"] = math.sqrt(max(K-float(s.square().sum()), 0))
            for K in (16, 32, 64):
                res[f"lam16x16_K{K}"] = v[K-1].item(); res[f"relgap_K{K}"] = ((v[K]-v[K-1])/v[K]).item()
            for h in (8, 12):     # 跨网格：同一 σ_z、同一 K 下 λ_K 是否对应
                _, vh, _ = eig(fsq(proj(resize(zt, h))), sx, sz_cal, h)
                for K in (16, 32):
                    res[f"lam{h}x{h}_K{K}"] = vh[K-1].item()
            rows.append(res)
        m[f"sigma_x={sx}"] = {k: float(np.mean([r_[k] for r_ in rows])) for k in rows[0]}
    report["modes"][mode] = m

# 建议规则：在定稿方案、σ_x=0.125 下，若 K=32 的“下一帧子空间漂移”≤0.3 且 16/12/8 网格的 λ_32 相差 ≤0.1，则取 32，否则取 16
d = report["modes"]["Z_detached(定稿)"]["sigma_x=0.125"]
ok32 = d["next_frame_sinTheta_K32"] <= 0.3 and max(abs(d["lam16x16_K32"]-d["lam8x8_K32"]), abs(d["lam16x16_K32"]-d["lam12x12_K32"])) <= 0.1
report["建议"] = {"sigma_x": 0.125, "sigma_z": report["modes"]["Z_detached(定稿)"]["calibrated_sigma_z"], "K": 32 if ok32 else 16,
                 "依据": {k: d[k] for k in ("next_frame_sinTheta_K16", "next_frame_sinTheta_K32", "lam16x16_K32", "lam12x12_K32", "lam8x8_K32")}}
out = Path(__file__).with_name("real_feature_probe_result.json")
out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=1, ensure_ascii=False))
print("saved", out)
print("建议:", json.dumps(report["建议"], ensure_ascii=False))
