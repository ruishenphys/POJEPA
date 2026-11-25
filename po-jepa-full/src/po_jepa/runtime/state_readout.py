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
   S25b5（Jerry 2026-11-25 定，新口径）：二阶场景另读速度。位置探针（上表）不变；另拟合一个速度探针，目标是构形的时间导数
   velocity_target = configuration.velocity × 每单位像素（像素/秒；wave 是 u_t 的池化场，场的 L2/秒），同一套特征、
   同一套拟合（单独选 λ，所以位置探针的数值和以前的拟合法逐位相同），存在探针目录的 velocity/ 子目录（probe.json 的
   quantity = "velocity"）。两帧编码（tubelet=pair）的特征看得到速度；单帧复制（dup）也照样拟合，作为对照（看不到速度时
   速度探针的 R² 应该明显更低）。一阶场景和 push 的状态里没有速度，什么都不变。滚动和反事实的速度指标在键名前加 vel_
   （vel_readout:T=t、vel_nmse:T=t、vel_vpt@ε、vel_cf_error 等），定义与位置的逐条相同，只是 R、y、V 换成速度探针的；
   二阶场景的读出结果从 S25b5 起多了这些键，和 S25b5 之前的结果不可比（之前没有速度，探针和真值文件要重做）。
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
   donor 是 delta_act 的同一个（datasets.donor_id：测试集内下一条登记起点相同的轨迹，test ≤ 20 条时就是下一条、2000 条时是编号 +5；同起点同时长的动作）。ẑ_A = Φ(z, u)，ẑ_B = Φ(z, u_donor)，
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
VELOCITY_DIR = "velocity"          # S25b5：速度探针在位置探针目录下的子目录
VELOCITY_PREFIX = "vel_"           # S25b5：滚动、反事实里速度指标的键名前缀


def has_velocity(name) -> bool:
    """S25b5：二阶场景另读速度。"""
    return name in variants.SECOND_ORDER


def velocity_units(name) -> str:
    return units(name)+"_per_s"


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


def velocity_target(name, hidden, params) -> np.ndarray:
    """S25b5：二阶场景的一个隐藏状态 → 速度读出目标 = configuration.velocity × 每单位像素（wave 是 u_t 池化场，不乘）。
    与 target 同一个比例，所以 velocity_target 是 target 沿真实轨迹的时间导数（测试核）。"""
    h = np.asarray(hidden, dtype=np.float64)
    if h.ndim != 1:
        raise ValueError("velocity_target 一次只收一个隐藏状态 [D]")
    if not has_velocity(name):
        raise ValueError(f"{name} 的状态里没有速度（只有二阶场景读速度）")
    c = configuration.velocity(name, h, params)
    return c if configuration.PX[name] is None else c*configuration.PX[name]


def velocity_targets(name, states, params) -> np.ndarray:
    return np.stack([velocity_target(name, s, params) for s in states])


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


def load(folder: Path, *, dataset_tag: str, system: str, quantity: str = "position") -> tuple:
    """→ (Probe, meta)。核 schema、读的量（S25b5：position 或 velocity；S25b5 之前的探针没有 quantity 键，算 position）、
    数据集、两个数组文件的 SHA 和形状。"""
    folder = Path(folder)
    if not (folder/"probe.json").exists():
        raise ValueError(f"{folder} 下没有 probe.json" + ("（S25b5 起二阶场景要速度探针：用当前代码重新跑 fit）"
                                                         if quantity == "velocity" else ""))
    meta = json.loads((folder/"probe.json").read_text())
    if meta.get("schema") != PROBE_SCHEMA:
        raise ValueError("not a state-readout probe")
    if meta.get("quantity", "position") != quantity:
        raise ValueError(f"probe reads {meta.get('quantity', 'position')}, not {quantity}")
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


def rollout_names(horizon, velocity=False) -> tuple:
    """velocity=True（S25b5，二阶场景）时位置的键后面再接同样一套 vel_ 开头的速度键。"""
    from . import long_rollout
    from .addon import d4
    times = long_rollout.curve_times(horizon)
    names = [f"{k}:T={d4._t(t)}" for t in times for k in ("readout", "floor", "nmse", "floor_nmse")]
    names += [f"{k}@{e:g}" for e in VPT_THRESHOLDS for k in ("vpt", "floor_vpt")]
    if velocity:
        names += [VELOCITY_PREFIX+k for k in names]
    return tuple(names)


def _curve(probe, preds, reals, truth, times, prefix) -> dict:
    """一个量（位置或速度）的读出曲线和 VPT。preds、reals：[len(times), …] 的预测特征和真实特征；truth：{t: y_t}。"""
    from .addon import d4
    pred_y, real_y = probe(preds), probe(reals)
    values, nmse, floor = {}, [], []
    for i, t in enumerate(times):
        y = np.asarray(truth[t], dtype=np.float64)
        if y.shape != pred_y[i].shape:
            raise ValueError("truth and probe outputs differ in shape")
        e, f = float(np.linalg.norm(pred_y[i]-y)), float(np.linalg.norm(real_y[i]-y))
        values[f"{prefix}readout:T={d4._t(t)}"] = e
        values[f"{prefix}floor:T={d4._t(t)}"] = f
        values[f"{prefix}nmse:T={d4._t(t)}"] = e*e/probe.variance
        values[f"{prefix}floor_nmse:T={d4._t(t)}"] = f*f/probe.variance
        nmse.append(e*e/probe.variance); floor.append(f*f/probe.variance)
    for eps in VPT_THRESHOLDS:
        values[f"{prefix}vpt@{eps:g}"] = vpt(times, nmse, eps)
        values[f"{prefix}floor_vpt@{eps:g}"] = vpt(times, floor, eps)
    return values


@torch.no_grad()
def rollout_values(model, probe, windows, truth, horizon, velocity_probe=None, velocity_truth=None) -> dict:
    """一条长时程测试轨迹。windows：{t: CacheWindow}，t 取遍 long_rollout.window_times(H)（同一条轨迹、起点 0，
    long_rollout.load_window 读出；windows[t].target.raw_z 是真实特征 z*_t）；truth：{t: y_t}（读出目标，物理单位）。
    S25b5：velocity_probe、velocity_truth（{t: 速度目标}）同时给时另算 vel_ 开头的速度指标（同一个滚动预测）。"""
    from . import long_rollout
    from .addon import d4
    if (velocity_probe is None) != (velocity_truth is None):
        raise ValueError("velocity_probe 和 velocity_truth 要一起给")
    times = long_rollout.curve_times(horizon)
    preds = rollout_predictions(model, windows[float(horizon)], horizon)
    pz = torch.cat([preds[t] for t in times])
    rz = torch.cat([windows[t].target.raw_z for t in times])
    values = _curve(probe, pz, rz, truth, times, "")
    if velocity_probe is not None:
        values.update(_curve(velocity_probe, pz, rz, velocity_truth, times, VELOCITY_PREFIX))
    return d4._finite({k: values[k] for k in rollout_names(horizon, velocity_probe is not None)})


# —— 反事实预测 ——

CF_KEYS = ("cf_true", "cf_pred", "cf_error", "cf_inner", "actual_readout", "actual_floor")


def _cf(probe, z, y_actual, y_counterfactual, prefix) -> dict:
    """一个量的反事实指标。z = cat(ẑ_A, ẑ_B, z*_A)。"""
    r_a, r_b, r_real = probe(z)
    y_a = np.asarray(y_actual, dtype=np.float64); y_b = np.asarray(y_counterfactual, dtype=np.float64)
    if y_a.shape != r_a.shape or y_b.shape != r_a.shape:
        raise ValueError("truth and probe outputs differ in shape")
    true, pred = y_b-y_a, r_b-r_a
    values = dict(cf_true=float(np.linalg.norm(true)), cf_pred=float(np.linalg.norm(pred)),
                  cf_error=float(np.linalg.norm(pred-true)), cf_inner=float(pred@true),
                  actual_readout=float(np.linalg.norm(r_a-y_a)), actual_floor=float(np.linalg.norm(r_real-y_a)))
    return {prefix+k: v for k, v in values.items()}


@torch.no_grad()
def counterfactual_values(model, probe, window, donor, y_actual, y_counterfactual,
                          velocity_probe=None, v_actual=None, v_counterfactual=None) -> dict:
    """一个测试窗口（evaluation.load_test_window 读出的 recipient 和 donor，调用者已核 _aligned_pair）。
    y_actual = target(s_A)（真实终点），y_counterfactual = target(s_B)（换 donor 动作重新仿真的终点）。
    S25b5：velocity_probe、v_actual = velocity_target(s_A)、v_counterfactual = velocity_target(s_B) 一起给时另算 vel_ 开头的键。"""
    from .addon import d4
    given = (velocity_probe is not None, v_actual is not None, v_counterfactual is not None)
    if any(given) and not all(given):
        raise ValueError("velocity_probe、v_actual、v_counterfactual 要一起给")
    raw, grid, duration = window.inputs.raw_z, window.inputs.grid, window.inputs.duration
    actual = model(raw, window.inputs.controls, duration, grid)
    alternative = model(raw, donor.inputs.controls, duration, grid)
    z = torch.cat((actual, alternative, window.target.raw_z))
    values = _cf(probe, z, y_actual, y_counterfactual, "")
    if velocity_probe is not None:
        values.update(_cf(velocity_probe, z, v_actual, v_counterfactual, VELOCITY_PREFIX))
    return d4._finite(values)


def counterfactual_summary(rows) -> dict:
    """rows：每个窗口 dict(values=…, identical_controls=bool)。均值取全部窗口；cos、explained、gain 只用动作不同的窗口。
    S25b5：values 里有 vel_ 开头的速度键时，另给 velocity = 同样定义的速度汇总（动作相同的窗口同样不进 cos 等）。"""
    out = _cf_summary(rows, "")
    if rows and VELOCITY_PREFIX+"cf_true" in rows[0]["values"]:
        out["velocity"] = _cf_summary(rows, VELOCITY_PREFIX)
    return out


def _cf_summary(rows, prefix) -> dict:
    if not rows:
        return {}
    rows = [dict(r, values={k: r["values"][prefix+k] for k in CF_KEYS}) for r in rows]
    out = {k: float(np.mean([r["values"][k] for r in rows])) for k in CF_KEYS if k != "cf_inner"}
    moved = [r["values"] for r in rows if not r["identical_controls"]]
    total = sum(v["cf_true"]**2 for v in moved)
    cos = [v["cf_inner"]/(v["cf_pred"]*v["cf_true"]) for v in moved if v["cf_pred"] > 0 and v["cf_true"] > 0]
    out.update(windows=len(rows), windows_with_different_actions=len(moved),
               explained=None if not total > 0 else 1.-sum(v["cf_error"]**2 for v in moved)/total,
               gain=None if not total > 0 else sum(v["cf_inner"] for v in moved)/total,
               cos_mean=float(np.mean(cos)) if cos else None, cos_windows=len(cos))
    return out
