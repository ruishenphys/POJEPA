"""S25f：物理状态读出（冻结线性探针）、长时程滚动的物理单位误差和有效预测时长（VPT）、反事实预测（v8 规划 §1.6）。

把“特征距离”换成“物理量误差”：在真实画面的冻结特征上拟合一个线性探针 R（特征 → 真值状态），再用同一个 R 读模型
预测出来的特征。探针只拟合一次（每个数据集一个，和模型、条件、种子无关），拟合用主实验缓存的验证段（33 帧），
所以从不碰测试轨迹，也不碰训练模型用过的训练段。

1. 读出目标（target）：和动作探针 tools/probe_actions.py 的 *_px 列同一个量，‖target(a) − target(b)‖ = motion(a, b)（测试核）：
   | 场景 | 目标 | 单位 |
   |---|---|---|
   | double_well、multi_particle、landscape | 隐藏状态（位置）× 48 | 像素 |
   | van_der_pol | 相平面点 × 27 | 像素 |
   | advection_diffusion | 16×16 平均浓度场（256 维） | 场的 L2 |
   | gray_scott | U、V 池化场（512 维，不含注入点） | 场的 L2 |
   | push | 推杆和方块中心 4 个坐标 × 48（不含朝向角） | 像素 |
   | 二阶场景 | 构形（po_jepa.configuration，只有位置）× 每单位像素；wave 是 u 的池化场 | 像素；wave 场的 L2 |
   二阶场景现在按单帧复制（dup）编码，特征看不到速度，所以只读位置（S25b 接 pair 编码后可以加速度）。
2. 探针的输入：每个 token 先做无仿射 LayerNorm（eps = 1e-5，和 JEPA 距离同一个），再取读出 kind（tubelet_probe.readout：
   mean 1408 维、pool4 22528 维、full 360448 维，默认 full）。LN 让探针只看损失约束的那部分（JEPA 距离对每个 token 的
   平移和缩放不敏感，模型的原始预测在这两个方向上没被约束）。
3. 拟合：对偶岭回归（核 K = XXᵀ，按拟合样本的均值精确中心化，tubelet_probe 的写法）。λ = trace(K̃)/n·10^e，
   e ∈ tubelet_probe.LAMBDA_EXPONENTS，所有输出维共用一个 λ：验证段按轨迹分 5 折（轨迹编号 mod 5；轨迹不到 5 条时折数 = 轨迹数），选合并 R²
   （1 − ΣSSE/ΣSST，对全部输出维求和）最大的 λ，再用全部验证样本重拟合。存成原始形式 W = X̃ᵀα：
   R(z) = (x − μ_x)·W + μ_y。V = 拟合样本目标的总方差 mean‖y − ȳ‖²（NMSE 的分母）。
4. 长时程滚动（tools/state_readout.py rollout）：S25c 的 8 步滚动（同一个预测，测试核 JEPA 距离与 long_rollout 的
   rollout:T 逐位相同），每 8 个 tick 读一次：
   - readout:T=t = ‖R(ẑ_t) − y_t‖（物理单位），floor:T=t = ‖R(z*_t) − y_t‖（探针在真实特征上的误差，读出的下限）；
   - nmse:T=t = ‖R(ẑ_t) − y_t‖²/V，floor_nmse:T=t 同理；
   - vpt@ε = 最大的 t，使得 t 以前（含 t）所有曲线点 nmse ≤ ε（tick；第一个点就超过时是 0），ε ∈ VPT_THRESHOLDS；
     floor_vpt@ε 同理（探针本身能撑多久：floor_vpt < H 的地方，vpt 不能解释成模型的问题）。分辨率是 8 个 tick。
5. 反事实（tools/state_readout.py counterfactual）：主实验测试窗口（每条测试轨迹的登记起点 × 时长 1、2、4、8），
   donor 是 delta_act 的同一个（测试集下一条轨迹、同起点同时长的动作）。ẑ_A = Φ(z, u)，ẑ_B = Φ(z, u_donor)，
   Δ̂ = R(ẑ_B) − R(ẑ_A)；真值 Δ = target(s_B) − target(s_A)，s_B 是从同一个真实起点状态、同一份噪声换成 donor 动作的
   重新仿真（probe_actions.counterfactual_end）。每个窗口记 ‖Δ‖、‖Δ̂‖、‖Δ̂ − Δ‖、cos(Δ̂, Δ)，以及 ẑ_A 的读出误差和
   探针在真实终点上的读出误差；汇总 explained = 1 − Σ‖Δ̂ − Δ‖²/Σ‖Δ‖²（identity 恰好是 0）、
   gain = Σ⟨Δ̂, Δ⟩/Σ‖Δ‖²（沿真实方向预测出了多大比例）。动作与 donor 完全相同的窗口（Δ ≡ 0）不进 cos、explained、gain。
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from po_jepa import configuration, datasets, tubelet_probe, variants

KINDS = tubelet_probe.READOUTS
DEFAULT_KIND = "full"
FOLDS = 5
LAMBDA_EXPONENTS = tubelet_probe.LAMBDA_EXPONENTS
LN_EPS = 1e-5
VPT_THRESHOLDS = (.05, .1, .25)
CF_DURATIONS = (1., 2., 4., 8.)
SCALE = {"double_well": 48., "van_der_pol": 27., "multi_particle": 48., "advection_diffusion": 1., "landscape": 48.}
FIELD_TARGETS = ("advection_diffusion", "gray_scott", "wave")
PROBE_SCHEMA = "po_jepa.state_readout.v1"


def units(name) -> str:
    return "field_l2" if name in FIELD_TARGETS else "px"


def target(name, hidden, params) -> np.ndarray:
    """frames() 的一个隐藏状态 → 读出目标（文件头第 1 条的表）。params：dict(v.params)（二阶场景的构形要用）。"""
    h = np.asarray(hidden, dtype=np.float64)
    if h.ndim != 1:
        raise ValueError("target 一次只收一个隐藏状态 [D]")
    if name in variants.SECOND_ORDER:
        c = configuration.configuration(name, h, params)
        return c if configuration.PX[name] is None else c*configuration.PX[name]
    if name == "push":
        return h[:4]*48.
    if name == "gray_scott":
        return h[:512].copy()
    if name not in SCALE:
        raise ValueError(f"没有 {name} 的读出目标")
    return h*SCALE[name]


def targets(name, states, params) -> np.ndarray:
    return np.stack([target(name, s, params) for s in states])


def features(z: torch.Tensor, kind: str) -> torch.Tensor:
    """[n,256,1408] → 逐 token 无仿射 LN 后的读出特征 [n,d] float32。"""
    if kind not in KINDS:
        raise ValueError(f"readout kind must be one of {KINDS}")
    z = torch.as_tensor(z).float()
    return tubelet_probe.readout(F.layer_norm(z, (z.shape[-1],), eps=LN_EPS), kind)


@dataclass(frozen=True)
class Probe:
    """R(z) = (features(z) − mean_x)·weight + mean_y。weight [d,o] float32，mean_x [d] float32，mean_y [o] float64。"""
    kind: str
    weight: torch.Tensor
    mean_x: torch.Tensor
    mean_y: np.ndarray
    variance: float
    meta: dict

    def to(self, device):
        return Probe(self.kind, self.weight.to(device), self.mean_x.to(device), self.mean_y, self.variance, self.meta)

    @torch.no_grad()
    def __call__(self, z: torch.Tensor) -> np.ndarray:
        """z [n,256,1408]（任意精度、任意设备）→ 读出 [n,o] float64。"""
        x = features(z.to(self.weight.device), self.kind)
        before = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            out = ((x-self.mean_x)@self.weight).double().cpu().numpy()+self.mean_y
        finally:
            torch.set_float32_matmul_precision(before)
        if not np.isfinite(out).all():
            raise ValueError("state readout produced nonfinite values")
        return out


def _pooled_r2(y, pred):
    return 1.-float(((y-pred)**2).sum())/float(((y-y.mean(0))**2).sum())


def select_lambda(k: np.ndarray, y: np.ndarray, groups: np.ndarray, folds=FOLDS) -> dict:
    """按轨迹分折（编号 mod folds）的交叉验证：每个 λ 的合并 R²（各折测试样本的 SSE 相加，SST 用全体样本的均值）。"""
    folds = min(folds, len(set(np.asarray(groups).tolist())))       # 轨迹少于 5 条时（只在 CPU 小数据集上）折数 = 轨迹数
    if folds < 2:
        raise ValueError("拟合探针至少要 2 条轨迹")
    fold = np.asarray(groups) % folds
    sse = np.zeros(len(LAMBDA_EXPONENTS))
    for f in range(folds):
        test = np.flatnonzero(fold == f); train = np.flatnonzero(fold != f)
        if not len(test) or not len(train):
            raise ValueError("每折都要有轨迹：拟合用的轨迹数至少等于折数")
        model = tubelet_probe._ridge_fit(k, y, train)
        for i, e in enumerate(LAMBDA_EXPONENTS):
            pred = tubelet_probe._ridge_predict(k, train, test, model, model[4]*10.**e)
            sse[i] += float(((y[test]-pred)**2).sum())
    sst = float(((y-y.mean(0))**2).sum())
    if not sst > 0:
        raise ValueError("读出目标在拟合样本上没有方差")
    r2 = 1.-sse/sst
    best = int(np.argmax(r2))
    return dict(best=best, folds=folds, lambda_exponent=float(LAMBDA_EXPONENTS[best]), cv_r2=r2.tolist(),
                at_grid_edge=bool(best in (0, len(LAMBDA_EXPONENTS)-1)))


def fit(x: torch.Tensor, y: np.ndarray, groups: np.ndarray, kind: str, *, device="cpu", chunk=32768) -> tuple:
    """x [n,d] = features(z, kind)（调用者按轨迹逐条算好拼起来，省一份 [n,256,1408] 的内存）、y [n,o]、
    groups [n]（轨迹编号）→ (Probe, 拟合记录)。"""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    if kind not in KINDS:
        raise ValueError(f"readout kind must be one of {KINDS}")
    if x.ndim != 2 or x.dtype != torch.float32 or y.ndim != 2 or len(y) != x.shape[0] or groups.shape != (len(y),):
        raise ValueError("x [n,d] float32、y [n,o]、groups [n] 的样本数必须相同")
    if not torch.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("probe inputs must be finite")
    k = _gram(x, device, chunk)
    choice = select_lambda(k, y, groups)
    fit_rows = np.arange(len(y))
    vecs, evals, proj, mean_y, scale = tubelet_probe._ridge_fit(k, y, fit_rows)
    lam = scale*10.**choice["lambda_exponent"]
    alpha = vecs@(proj/(evals+lam)[:, None])                                # [n,o]
    pred = tubelet_probe.centered(k, fit_rows, fit_rows, fit_rows)@alpha+mean_y
    mean_x = _mean(x, chunk)
    a = torch.from_numpy(alpha).to(device)
    weight = torch.zeros(x.shape[1], y.shape[1], dtype=torch.float64)
    for s in range(0, x.shape[1], chunk):
        xc = (x[:, s:s+chunk].double()-mean_x[s:s+chunk]).to(device)
        weight[s:s+chunk] = (xc.T@a).cpu()
    variance = float(((y-mean_y)**2).sum(1).mean())
    record = dict(kind=kind, folds=choice["folds"], samples=int(len(y)), trajectories=int(len(set(groups.tolist()))), outputs=int(y.shape[1]),
                  lambda_exponent=choice["lambda_exponent"], lambda_value=float(lam), lambda_at_grid_edge=choice["at_grid_edge"],
                  cv_r2_by_lambda=dict(zip((f"{e:g}" for e in LAMBDA_EXPONENTS), choice["cv_r2"])),
                  cv_r2=choice["cv_r2"][choice["best"]], fit_r2=_pooled_r2(y, pred), variance=variance,
                  cv_rmse=math.sqrt(max(0., 1.-choice["cv_r2"][choice["best"]])*variance))
    probe = Probe(kind, weight.float(), mean_x.float(), mean_y, variance, record)
    return probe, record


def _mean(x: torch.Tensor, chunk) -> torch.Tensor:
    """按特征维分块的 float64 列均值（不整块转 float64）。"""
    return torch.cat([x[:, s:s+chunk].double().mean(0) for s in range(0, x.shape[1], chunk)])


def _gram(x: torch.Tensor, device, chunk) -> np.ndarray:
    """K = XXᵀ [n,n] float64：先减去全体样本均值（只为数值条件，之后按拟合样本精确中心化），按特征维分块。"""
    before = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        mean = _mean(x, chunk)
        k = torch.zeros(x.shape[0], x.shape[0], dtype=torch.float64, device=device)
        for s in range(0, x.shape[1], chunk):
            xc = (x[:, s:s+chunk].double()-mean[s:s+chunk]).to(device)
            k += xc@xc.T
        return k.cpu().numpy()
    finally:
        torch.set_float32_matmul_precision(before)


# —— 存取 ——

def save(probe: Probe, folder: Path, extra: dict) -> dict:
    """folder/weight.npy（float32 [d,o]）、mean_x.npy、probe.json（含两个数组文件的 SHA）。folder 必须不存在。"""
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"{folder} 已存在，不覆盖")
    tmp = folder.with_name(folder.name+".partial")
    tmp.mkdir(parents=True)
    np.save(tmp/"weight.npy", probe.weight.cpu().numpy().astype(np.float32))
    np.save(tmp/"mean_x.npy", probe.mean_x.cpu().numpy().astype(np.float32))
    meta = dict(schema=PROBE_SCHEMA, **extra, fit=probe.meta, mean_y=probe.mean_y.tolist(),
                weight_sha256=_sha(tmp/"weight.npy"), mean_x_sha256=_sha(tmp/"mean_x.npy"))
    (tmp/"probe.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, allow_nan=False))
    tmp.rename(folder)
    return meta


def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load(folder: Path, *, dataset_tag: str, system: str) -> tuple:
    """→ (Probe, meta)。核 schema、数据集、两个数组文件的 SHA 和形状。"""
    folder = Path(folder)
    meta = json.loads((folder/"probe.json").read_text())
    if meta.get("schema") != PROBE_SCHEMA:
        raise ValueError("not a state-readout probe")
    if meta.get("dataset") != dataset_tag or meta.get("system") != system:
        raise ValueError(f"probe was fitted on {meta.get('dataset')}, not {dataset_tag}")
    for name in ("weight", "mean_x"):
        if _sha(folder/f"{name}.npy") != meta[f"{name}_sha256"]:
            raise ValueError(f"probe {name}.npy SHA mismatch")
    weight = torch.from_numpy(np.load(folder/"weight.npy"))
    mean_x = torch.from_numpy(np.load(folder/"mean_x.npy"))
    fit_record = meta["fit"]
    mean_y = np.asarray(meta["mean_y"], dtype=np.float64)
    if (weight.dtype != torch.float32 or mean_x.dtype != torch.float32 or weight.ndim != 2
            or weight.shape != (mean_x.shape[0], len(mean_y)) or len(mean_y) != fit_record["outputs"]
            or fit_record["kind"] not in KINDS):
        raise ValueError("probe arrays have the wrong shape or dtype")
    return Probe(fit_record["kind"], weight, mean_x, mean_y, float(fit_record["variance"]), fit_record), meta


# —— 长时程滚动的读出（S25c 的滚动曲线）——

def rollout_predictions(model, full_window, horizon) -> dict:
    """S25c long_rollout.trajectory_values 的 8 步滚动，原样：{t: ẑ_t}，t = 8, 16, …, H（测试核与它逐位相同）。"""
    from .addon import d4
    from . import long_rollout
    raw, grid = full_window.inputs.raw_z, full_window.inputs.grid
    out, x = {}, raw
    for piece, t in zip(d4.pieces(full_window.inputs.controls, (long_rollout.BLOCK,)*int(horizon/long_rollout.BLOCK)),
                        long_rollout.curve_times(horizon)):
        x = model(x, piece, d4._cycles(long_rollout.BLOCK), grid)
        out[t] = x
    return out


def vpt(times, nmse, threshold) -> float:
    """最大的 t：t 以前（含）所有点 nmse ≤ threshold；第一个点就超过时 0。times 升序。"""
    best = 0.
    for t, e in zip(times, nmse):
        if not e <= threshold:
            break
        best = float(t)
    return best


def rollout_names(horizon) -> tuple:
    from . import long_rollout
    from .addon import d4
    times = long_rollout.curve_times(horizon)
    names = [f"{k}:T={d4._t(t)}" for t in times for k in ("readout", "floor", "nmse", "floor_nmse")]
    names += [f"{k}@{e:g}" for e in VPT_THRESHOLDS for k in ("vpt", "floor_vpt")]
    return tuple(names)


@torch.no_grad()
def rollout_values(model, probe, windows, truth, horizon) -> dict:
    """一条长时程测试轨迹。windows：{t: CacheWindow}，t 取遍 long_rollout.window_times(H)（同一条轨迹、起点 0，
    long_rollout.load_window 读出；windows[t].target.raw_z 是真实特征 z*_t）；truth：{t: y_t}（读出目标，物理单位）。"""
    from . import long_rollout
    from .addon import d4
    times = long_rollout.curve_times(horizon)
    preds = rollout_predictions(model, windows[float(horizon)], horizon)
    pred_y = probe(torch.cat([preds[t] for t in times]))
    real_y = probe(torch.cat([windows[t].target.raw_z for t in times]))
    values, nmse, floor = {}, [], []
    for i, t in enumerate(times):
        y = np.asarray(truth[t], dtype=np.float64)
        if y.shape != pred_y[i].shape:
            raise ValueError("truth and probe outputs differ in shape")
        e, f = float(np.linalg.norm(pred_y[i]-y)), float(np.linalg.norm(real_y[i]-y))
        values[f"readout:T={d4._t(t)}"] = e
        values[f"floor:T={d4._t(t)}"] = f
        values[f"nmse:T={d4._t(t)}"] = e*e/probe.variance
        values[f"floor_nmse:T={d4._t(t)}"] = f*f/probe.variance
        nmse.append(e*e/probe.variance); floor.append(f*f/probe.variance)
    for eps in VPT_THRESHOLDS:
        values[f"vpt@{eps:g}"] = vpt(times, nmse, eps)
        values[f"floor_vpt@{eps:g}"] = vpt(times, floor, eps)
    return d4._finite({k: values[k] for k in rollout_names(horizon)})


# —— 反事实预测 ——

CF_KEYS = ("cf_true", "cf_pred", "cf_error", "cf_inner", "actual_readout", "actual_floor")


@torch.no_grad()
def counterfactual_values(model, probe, window, donor, y_actual, y_counterfactual) -> dict:
    """一个测试窗口（evaluation.load_test_window 读出的 recipient 和 donor，调用者已核 _aligned_pair）。
    y_actual = target(s_A)（真实终点），y_counterfactual = target(s_B)（换 donor 动作重新仿真的终点）。"""
    from .addon import d4
    raw, grid, duration = window.inputs.raw_z, window.inputs.grid, window.inputs.duration
    actual = model(raw, window.inputs.controls, duration, grid)
    alternative = model(raw, donor.inputs.controls, duration, grid)
    r_a, r_b, r_real = probe(torch.cat((actual, alternative, window.target.raw_z)))
    y_a = np.asarray(y_actual, dtype=np.float64); y_b = np.asarray(y_counterfactual, dtype=np.float64)
    if y_a.shape != r_a.shape or y_b.shape != r_a.shape:
        raise ValueError("truth and probe outputs differ in shape")
    true, pred = y_b-y_a, r_b-r_a
    values = dict(cf_true=float(np.linalg.norm(true)), cf_pred=float(np.linalg.norm(pred)),
                  cf_error=float(np.linalg.norm(pred-true)), cf_inner=float(pred@true),
                  actual_readout=float(np.linalg.norm(r_a-y_a)), actual_floor=float(np.linalg.norm(r_real-y_a)))
    return d4._finite(values)


def counterfactual_summary(rows) -> dict:
    """rows：每个窗口 dict(values=…, identical_controls=bool)。均值取全部窗口；cos、explained、gain 只用动作不同的窗口。"""
    if not rows:
        return {}
    out = {k: float(np.mean([r["values"][k] for r in rows])) for k in CF_KEYS if k != "cf_inner"}
    moved = [r["values"] for r in rows if not r["identical_controls"]]
    total = sum(v["cf_true"]**2 for v in moved)
    cos = [v["cf_inner"]/(v["cf_pred"]*v["cf_true"]) for v in moved if v["cf_pred"] > 0 and v["cf_true"] > 0]
    out.update(windows=len(rows), windows_with_different_actions=len(moved),
               explained=None if not total > 0 else 1.-sum(v["cf_error"]**2 for v in moved)/total,
               gain=None if not total > 0 else sum(v["cf_inner"] for v in moved)/total,
               cos_mean=float(np.mean(cos)) if cos else None, cos_windows=len(cos))
    return out
