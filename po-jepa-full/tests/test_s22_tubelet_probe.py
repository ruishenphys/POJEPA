"""S22 两帧 tubelet 探针（tubelet_probe.py、tools/tubelet_probe.py）：单摆物理、画面、编码帧号、对偶岭回归、判据、端到端。
S22-fix1 补：真实预处理分支、全部 67 行、full 核精度到 R² 层面、嵌套选读出、缓存生成身份。"""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from po_jepa import tubelet_probe as tp, vjepa
from po_jepa.systems import _background

ROOT = Path(__file__).resolve().parents[1]


# —————————————————————————— 物理 ——————————————————————————

def energy(states, g_over_l=tp.G_OVER_L):
    return .5*states[:, 1]**2-g_over_l*np.cos(states[:, 0])


def test_conservative_energy_and_small_angle_period():
    s = tp.simulate([2.5, 0.], np.zeros(40), gamma=0.)
    e = energy(s)
    assert np.abs(e-e[0]).max() < 1e-8*abs(e[0])          # RK4、h = 5 ms：10 秒内能量相对漂移 < 1e−8
    # 小角：θ(t) = θ0·cos(ω0 t)，ω0 = √(g/L)
    s = tp.simulate([1e-3, 0.], np.zeros(32), gamma=0.)
    t = np.arange(33)*tp.PERIOD
    assert np.allclose(s[:, 0], 1e-3*np.cos(math.sqrt(tp.G_OVER_L)*t), atol=1e-9)


def test_damping_torque_and_convergence():
    # 线性化 + 阻尼 + 常力矩的解析解：θ'' + γθ' + ω0²θ = K·u
    w0, g, u = math.sqrt(tp.G_OVER_L), tp.GAMMA, .3
    s = tp.simulate([0., 0.], np.full(8, u*1e-4), substeps=50)
    wd = math.sqrt(w0**2-g**2/4); t = np.arange(9)*tp.PERIOD; c = tp.TORQUE_PER_ACTION*u*1e-4/w0**2
    exact = c*(1-np.exp(-g*t/2)*(np.cos(wd*t)+g/(2*wd)*np.sin(wd*t)))
    assert np.allclose(s[:, 0], exact, rtol=1e-6, atol=1e-14)
    # 无动作时阻尼让能量单调减
    s = tp.simulate([2., 1.], np.zeros(32))
    assert (np.diff(energy(s)) < 0).all()
    # 子步数加倍，结果几乎不变（RK4 收敛）
    st, u = tp.trajectory(7)
    fine = tp.simulate(tp.initial_state(7), u, substeps=200)
    assert np.abs(fine-st).max() < 1e-7


def test_streams_controls_and_states():
    s1, u1 = tp.trajectory(3); s2, u2 = tp.trajectory(3)
    assert np.array_equal(s1, s2) and np.array_equal(u1, u2)
    assert not np.array_equal(tp.trajectory(4)[0], s1)
    assert s1.shape == (34, 2) and u1.shape == (33,)
    assert np.array_equal(s1[0], tp.initial_state(3))
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1703, 2200, 3, 0])))
    assert np.array_equal(tp.initial_state(3), [g.uniform(-np.pi, np.pi), g.uniform(-3, 3)])
    # 预处理种子是随机流 2（独立重算，不调被测函数做参照）
    for tid in (0, 3, 199):
        expect = int(np.random.SeedSequence([1703, 2200, tid, 2]).generate_state(1, dtype=np.uint64)[0])
        assert tp.preprocess_seed(tid) == expect
    # 动作分段：每段长度 ∈ {1,4,8}（最后一段可被截断）、值 ∈ [−0.5,0.5]
    for tid in range(30):
        u = tp.controls(tid)
        assert np.abs(u).max() <= .5
        g = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1703, 2200, tid, 1])))
        t = 0
        while t < 33:
            n = int(g.choice([1, 4, 8])); v = 0. if g.random() < .2 else g.uniform(-.5, .5)
            assert np.all(u[t:t+n] == v); t += n


def test_velocity_below_nyquist_and_both_regimes():
    states = [tp.trajectory(i)[0] for i in range(200)]
    omega = max(np.abs(s[:, 1]).max() for s in states)
    assert omega*tp.PERIOD < .8*math.pi          # 每帧转角 < 0.8π：两帧之间方向没有歧义
    wrapped = [np.abs(np.angle(np.exp(1j*s[:, 0]))) for s in states]
    assert sum((w > .9*math.pi).any() for w in wrapped) >= 20     # 有越过顶点的
    assert sum(w.max() < math.pi/2 for w in wrapped) >= 50       # 也有一直在下半圈摆动的


# —————————————————————————— 画面与目标 ——————————————————————————

def test_draw_depends_on_theta_only_and_shows_bob():
    s = np.array([[.7, 0.], [.7, 5.], [-2.2, -1.]])
    f = tp.draw(s)
    assert f.shape == (3, 256, 256, 3) and f.dtype == np.uint8
    assert np.array_equal(f[0], f[1])           # 单帧看不出 θ̇
    for frame, theta in zip(f[1:], s[1:, 0]):
        diff = np.abs(frame.astype(float)-_background()).sum(-1)
        bob = (frame[..., 0] > 150) & (frame[..., 2] < 120)           # 橙色
        r, c = np.argwhere(bob).mean(0)+.5
        assert np.hypot(r-tp.bob_rc(theta)[0], c-tp.bob_rc(theta)[1]) < 1.
        # 摆杆：支点到摆锤的中点附近有灰线
        mid = (np.array([128., 128.])+tp.bob_rc(theta))/2
        assert diff[int(mid[0]), int(mid[1])] > 100
    # 约定：θ = 0 竖直向下（行 128+80），θ = π/2 在右边（列 128+80）
    assert np.allclose(tp.bob_rc([0., math.pi/2]), [[208., 128.], [128., 208.]])
    with pytest.raises(ValueError):
        tp.draw(np.array([[np.nan, 0.]]))


def test_targets_alignment():
    s, _ = tp.trajectory(11)
    y = tp.targets(s)
    assert y.shape == (33, 4)
    assert np.allclose(y[:, 0], np.sin(s[1:, 0])) and np.allclose(y[:, 1], np.cos(s[1:, 0]))
    assert np.array_equal(y[:, 2], s[1:, 1])
    d = s[1:, 0]-s[:-1, 0]
    assert np.allclose(y[:, 3], d/tp.PERIOD)      # 每帧转角 < π 时 wrap 不改差值
    # 跨过 ±π 时 wrap 生效
    jump = np.zeros((34, 2)); jump[:, 0] = np.linspace(3.0, 3.0+33*.5, 34); jump[:, 0] = np.angle(np.exp(1j*jump[:, 0]))
    assert np.allclose(tp.targets(jump)[:, 3], .5/tp.PERIOD)


def test_clip_and_sample_indices():
    assert tp.clip_indices("dup") == [(f, f) for f in range(34)]
    assert tp.clip_indices("pair") == [(t, t+1) for t in range(33)]     # 时间顺序：前一帧在前
    assert np.array_equal(tp.sample_rows("dup"), np.arange(1, 34))
    assert np.array_equal(tp.sample_rows("pair"), np.arange(33))
    prev, cur = tp.sample_rows("dup_concat")
    assert np.array_equal(prev, np.arange(33)) and np.array_equal(cur, np.arange(1, 34))


# —————————————————————————— 编码 ——————————————————————————

class TinyTubelet(torch.nn.Module):
    """按 2×16×16 tubelet 嵌入的小编码器（与真实 patch 嵌入同形状），只供接口测试。"""
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.proj = torch.nn.Conv3d(3, 1408, (2, 16, 16), (2, 16, 16))
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=g)*.01)
            self.proj.bias.zero_()

    def forward(self, video):
        return self.proj(video).flatten(2).transpose(1, 2)


def loaded_tiny():
    arch = vjepa.EncoderArchitectureSpec(vjepa.UPSTREAM_COMMIT, "x", "vit_giant_xformers", 256, 16, 2, 512,
                                         True, True, False, True, True, True)
    report = vjepa.CheckpointReport(Path("tiny"), "0"*64, "target_encoder", (), (), True, arch)
    return vjepa.LoadedEncoder(TinyTubelet(), report)


def preprocess():
    from po_jepa.runtime import cache
    return cache.preprocess_spec()


def test_clip_encoder_dup_equals_v7_and_pair_order():
    loaded = loaded_tiny()
    clip = tp.FrozenClipEncoder(loaded, preprocess=preprocess(), encoder_device=torch.device("cpu"), autocast_dtype=None)
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(2, 3, 256, 256, generator=g), torch.randn(2, 3, 256, 256, generator=g)
    frame = vjepa.FrozenFrameEncoder(loaded, preprocess=preprocess(), encoder_device=torch.device("cpu"), autocast_dtype=None)
    v7 = frame.encode_target(b, target_time_seconds=torch.zeros(2, dtype=torch.float64)).raw_z
    assert torch.equal(clip.encode_clip(b, b), v7)            # dup 就是 v7 的单帧复制
    w = loaded.encoder.proj.weight
    expect = torch.einsum("bchw,dchw->bd", a[:, :, :16, :16], w[:, :, 0])+torch.einsum("bchw,dchw->bd", b[:, :, :16, :16], w[:, :, 1])
    assert torch.allclose(clip.encode_clip(a, b)[:, 0], expect, atol=1e-4)    # 第 0 个时间片是前一帧
    assert not torch.allclose(clip.encode_clip(a, b), clip.encode_clip(b, a))
    with pytest.raises(ValueError):
        clip.encode_clip(a, b[:1])
    with pytest.raises(ValueError):
        clip.encode_clip(a.double(), b)
    loaded.encoder.train()
    with pytest.raises(ValueError, match="eval"):
        clip.encode_clip(a, b)
    loaded.encoder.eval(); loaded.encoder.proj.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        clip.encode_clip(a, b)


def test_encode_trajectory_fake_rows_match_clips():
    s, _ = tp.trajectory(2)
    frames = tp.draw(s)
    out = tp.encode_trajectory(frames, 2, fake=True, batch=7)
    assert out["dup"].shape == (34, 256, 1408) and out["pair"].shape == (33, 256, 1408)
    enc = tp.fake_tubelet_encoder()
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()/255.
    for mode in ("dup", "pair"):
        assert len(tp.clip_indices(mode)) == len(out[mode])
        for r in range(len(out[mode])):
            p, c = tp.clip_indices(mode)[r]
            ref = enc(torch.stack((x[p:p+1], x[c:c+1]), 2)).double().numpy()[0]
            assert np.allclose(out[mode][r], ref, atol=1e-6)
    with pytest.raises(ValueError):
        tp.encode_trajectory(frames[:33], 2, fake=True)


def test_encode_trajectory_real_preprocess_branch(monkeypatch):
    """fake=False：每帧预处理前把 generator 重置到 preprocess_seed(轨迹) 的同一状态，一次只处理一帧、按帧号顺序；
    编码器收到的 67 个 tubelet 的前、后帧正是 clip_indices 的预处理图，输出逐行等于单独编码。
    预处理换成记录用的替身（真实裁剪要上游仓库），替身和真函数一样从 generator 抽一个种子决定“裁剪”。"""
    tid = 5
    s, _ = tp.trajectory(tid)
    frames = tp.draw(s)
    calls = []

    def stub(rgb, *, spec, generator):
        calls.append(dict(rgb=rgb.clone(), spec=spec, state=generator.get_state().clone()))
        seed = int(torch.randint(0, 2**32, (), generator=generator, dtype=torch.int64))
        shift = seed % 13
        return (rgb.permute(0, 3, 1, 2).float().roll(shift, -1)-100.)/50.

    monkeypatch.setattr(tp.vjepa, "preprocess_rgb", stub)
    clip = tp.FrozenClipEncoder(loaded_tiny(), preprocess=preprocess(), encoder_device=torch.device("cpu"), autocast_dtype=None)
    seen = []
    real_encode = clip.encode_clip
    monkeypatch.setattr(clip, "encode_clip", lambda prev, cur: seen.append((prev, cur)) or real_encode(prev, cur))
    out = tp.encode_trajectory(frames, tid, encoder=clip, fake=False, batch=10)
    # 预处理：34 次、每次一帧、按顺序、规格是 v7 的、初始 generator 状态都是同一个重置状态
    start = torch.Generator().manual_seed(tp.preprocess_seed(tid)).get_state()
    assert len(calls) == 34
    for f, c in enumerate(calls):
        assert torch.equal(c["rgb"], torch.from_numpy(frames[f:f+1]))
        assert c["spec"] == preprocess() and torch.equal(c["state"], start)
    # 独立重算每帧的预处理图（同一个重置状态 → 同一个裁剪）
    pre = [stub(torch.from_numpy(frames[f:f+1]), spec=preprocess(), generator=torch.Generator().manual_seed(tp.preprocess_seed(tid)))
           for f in range(34)]
    calls.clear()
    prevs = torch.cat([p for p, _ in seen]); curs = torch.cat([c for _, c in seen])
    clips = tp.clip_indices("dup")+tp.clip_indices("pair")
    assert len(prevs) == 67
    for i, (a, b) in enumerate(clips):
        assert torch.equal(prevs[i], pre[a][0]) and torch.equal(curs[i], pre[b][0])
    for mode in ("dup", "pair"):
        for r, (a, b) in enumerate(tp.clip_indices(mode)):
            ref = real_encode(pre[a], pre[b]).double().numpy()[0]
            assert np.array_equal(out[mode][r], ref)


# —————————————————————————— 探针 ——————————————————————————

def test_readouts():
    z = torch.randn(3, 256, 1408, dtype=torch.float64)
    grid = z.reshape(3, 16, 16, 1408)
    assert torch.allclose(tp.readout(z, "mean"), z.mean(1).float(), atol=1e-6)
    pool = torch.stack([grid[:, 4*i:4*i+4, 4*j:4*j+4].mean((1, 2)) for i in range(4) for j in range(4)], 1)
    assert torch.allclose(tp.readout(z, "pool4"), pool.reshape(3, -1).float(), atol=1e-6)
    assert tp.readout(z, "full").shape == (3, 256*1408)


def test_gram_concat_and_centering_exact():
    g = torch.Generator().manual_seed(3)
    z = (torch.randn(2*34, 256, 1408, generator=g)+5.).half()
    for kind in tp.READOUTS:
        x = tp.readout(z, kind).double()
        k = tp.gram(z, kind, chunk=50000)
        xc = x-x.mean(0)
        ref = (xc@xc.T).numpy()
        assert np.abs(k-ref).max() < 1e-5*np.abs(ref).max()
    # dup_concat 的核 = 显式拼接特征的核（中心化后相同）
    x = tp.readout(z, "mean").double().numpy(); k = x@x.T
    prev, cur = tp.sample_rows("dup_concat")
    feats = np.concatenate([np.concatenate((x[i*34+prev], x[i*34+cur]), 1) for i in range(2)])
    fit = np.arange(0, 66, 2)
    kc = tp.centered(tp.concat_gram(k, 2), np.arange(66), np.arange(66), fit)
    fc = feats-feats[fit].mean(0)
    assert np.allclose(kc, fc@fc.T, atol=1e-6*np.abs(fc@fc.T).max())
    # select_rows 取 dup 的第 t+1 帧
    sel = tp.select_rows(k, 34, tp.sample_rows("dup"), 2)
    idx = np.concatenate([i*34+np.arange(1, 34) for i in range(2)])
    assert np.array_equal(sel, k[np.ix_(idx, idx)])


def test_full_gram_precision_at_r2_level():
    """强相关特征（大的共同偏移 + 低秩信号 + 小噪声，float16 存储）：full 读出分块 float64 乘、float64 累加的核，
    与 float64 参考核在核本身、最小 λ 的预测和 5 折 R² 上一致。S22-fix2（Codex S22-P2-1）：原 fp32 块乘在 macOS ARM64 上
    预测差 1.7e−2、R² 差 3e−3；在 Linux x86 上是核 4e−8、预测 1.8e−4、R² 2.8e−5，也过不了下面收紧后的阈值（float64 实测约
    核 4e−16、预测 4e−12、R² 1e−12）。"""
    g = torch.Generator().manual_seed(4)
    n_traj, per = 5, 17
    n = n_traj*per
    common = 3.+torch.randn(1, 256, 1408, generator=g)
    factors = torch.randn(n, 3, generator=g)
    basis = torch.randn(3, 256, 1408, generator=g)*.05
    z = (common+torch.einsum("nk,kij->nij", factors, basis)+.002*torch.randn(n, 256, 1408, generator=g)).half()
    x = z.reshape(n, -1).double(); xc = x-x.mean(0)
    ref = (xc@xc.T).numpy()
    k = tp.gram(z, "full")
    assert k.dtype == np.float64 and np.abs(k-ref).max() < 1e-12*np.abs(ref).max()
    y = np.stack((factors[:, 0].numpy(), (factors[:, 1]+.5*factors[:, 2]).numpy(), np.random.default_rng(0).normal(size=n)), -1)
    traj = np.repeat(np.arange(n_traj), per)
    fit, rows = np.flatnonzero(traj != 0), np.flatnonzero(traj == 0)
    for kk in (k, ref):
        kk.setflags(write=False)
    m_ref, m = tp._ridge_fit(ref, y, fit), tp._ridge_fit(k, y, fit)
    lam = m_ref[4]*10.**min(tp.LAMBDA_EXPONENTS)
    p_ref, p = tp._ridge_predict(ref, fit, rows, m_ref, lam), tp._ridge_predict(k, fit, rows, m, lam)
    assert np.abs(p-p_ref).max() < 1e-8*np.abs(p_ref).max()
    a, b = tp.ridge_cv(k, y, traj), tp.ridge_cv(ref, y, traj)
    assert a["r2_mean"][0] > .99 and a["r2_mean"][1] > .99
    assert np.allclose(a["r2_per_fold"], b["r2_per_fold"], rtol=0, atol=1e-8)
    assert a["lambda_exponent"] == b["lambda_exponent"]


def test_dual_ridge_equals_primal():
    g = np.random.default_rng(0)
    x = g.normal(size=(60, 8))+3.; y = g.normal(size=(60, 2))
    fit, rows = np.arange(40), np.arange(40, 60); lam = .7
    model = tp._ridge_fit(x@x.T, y, fit)
    dual = tp._ridge_predict(x@x.T, fit, rows, model, lam)
    mu, my = x[fit].mean(0), y[fit].mean(0)
    w = np.linalg.solve((x[fit]-mu).T@(x[fit]-mu)+lam*np.eye(8), (x[fit]-mu).T@(y[fit]-my))
    assert np.allclose(dual, (x[rows]-mu)@w+my, atol=1e-10)


def test_ridge_cv_signal_noise_and_folds():
    g = np.random.default_rng(1)
    n_traj, per = 30, 33
    x = g.normal(size=(n_traj*per, 20))
    y = np.stack((x@g.normal(size=20)+.05*g.normal(size=len(x)), g.normal(size=len(x))), -1)
    traj = np.repeat(np.arange(n_traj), per)
    res = tp.ridge_cv(x@x.T, y, traj)
    assert res["r2_mean"][0] > .99 and res["r2_mean"][1] < .05
    assert len(res["r2_per_fold"]) == 5
    with pytest.raises(ValueError, match="每折"):
        tp.ridge_cv(x[:3*per]@x[:3*per].T, y[:3*per], traj[:3*per])


def test_ridge_cv_splits_by_trajectory():
    """同一条轨迹的样本不能同时进训练和测试：特征 = 轨迹编号的独热，目标 = 每条轨迹一个随机常数。
    按轨迹分折时测试轨迹的独热在训练里从没出现，R² ≤ 0；按样本分折时会接近 1。"""
    g = np.random.default_rng(2)
    n_traj, per = 20, 33
    traj = np.repeat(np.arange(n_traj), per)
    x = np.eye(n_traj)[traj]
    y = np.repeat(g.normal(size=(n_traj, 1)), per, 0)
    assert tp.ridge_cv(x@x.T, y, traj)["r2_mean"][0] <= 0.
    assert tp.ridge_cv(x@x.T, y, np.arange(len(traj)))["r2_mean"][0] > .9


def table(pair_omega, dup_omega, dup_pos=.99, val=None):
    """每个读出、每折相同的 ridge_cv 结果；val[ro] 给 pair 各折选 λ 那一折的 θ̇ R²（默认等于测试 R²）。"""
    res = {}
    for mode, om in (("pair", pair_omega), ("dup", dup_omega), ("dup_concat", 0.)):
        res[mode] = {}
        for ro in tp.READOUTS:
            o = om[ro] if isinstance(om, dict) else om
            row = [dup_pos if mode == "dup" else .99]*2+[o, o]
            v = [row[:2]+[val[ro][f]]*2 for f in range(5)] if (val and mode == "pair") else [row]*5
            res[mode][ro] = dict(r2_mean=row, r2_per_fold=[row]*5, val_r2_per_fold=v)
    return res


def test_decide_rule():
    assert tp.decide(table(.95, .1))["adopt_two_frame"]
    assert not tp.decide(table(.85, .1))["adopt_two_frame"]
    assert not tp.decide(table(.95, .9))["adopt_two_frame"]                 # 提升不到 0.1
    assert not tp.decide(table(.95, .1, dup_pos=.5))["adopt_two_frame"]     # 位置都读不出：探针无效
    d = tp.decide(table(dict(mean=.5, pool4=.93, full=.6), .1))
    assert d["selected_readouts"] == ["pool4"]*5 and d["adopt_two_frame"]
    assert d["by_readout"][1]["readout"] == "pool4" and d["by_readout"][1]["pair_omega"] == .93


def test_decide_selects_readout_on_validation_folds_only():
    """读出由每折选 λ 那一折上的 pair θ̇ R² 挑，测试 R² 不参与挑选：测试上 full 最高（0.95），但验证上 mean 最高，
    判据取 mean 的测试值（0.85）→ 不采用；按测试均值挑最大会选 full 并误判采用。"""
    test = dict(mean=.85, pool4=.80, full=.95)
    val = dict(mean=[.9]*5, pool4=[.8]*5, full=[.7]*5)
    d = tp.decide(table(test, .1, val=val))
    assert d["selected_readouts"] == ["mean"]*5 and not d["adopt_two_frame"]
    assert d["selected"]["pair_omega"] == pytest.approx(.85) and d["selected"]["gain"] == pytest.approx(.75)
    # 每折可以选不同的读出，判据是各折所选读出测试 R² 的平均
    val = dict(mean=[.9, .1, .1, .1, .1], pool4=[.1]*5, full=[.1, .9, .9, .9, .9])
    d = tp.decide(table(test, .1, val=val))
    assert d["selected_readouts"] == ["mean"]+["full"]*4
    assert d["selected"]["pair_omega"] == pytest.approx((.85+4*.95)/5) and d["adopt_two_frame"]
    # dup 的 θ̇ 取同一个所选读出（full）上的值
    d = tp.decide(table(test, dict(mean=.5, pool4=.4, full=.2), val=dict(mean=[.1]*5, pool4=[.1]*5, full=[.9]*5)))
    assert d["selected"]["dup_omega"] == pytest.approx(.2) and d["selected"]["gain"] == pytest.approx(.75)
    # 位置用所选读出上 dup 的 sin/cos
    for bad in ([.5, .99, .1, .1], [.99, .5, .1, .1]):        # sin、cos 各单独读不出（其他读出都读得出）
        t = table(test, .1, val=dict(mean=[.9]*5, pool4=[.1]*5, full=[.1]*5))
        for f in range(5):
            t["dup"]["mean"]["r2_per_fold"][f] = list(bad)
        assert not tp.decide(t)["position_readable"]
    # ridge_cv 给出 val_r2_per_fold：每折、每个目标所选 λ 在验证折上的 R²
    g = np.random.default_rng(5)
    x = g.normal(size=(20*33, 6)); y = np.stack((x@g.normal(size=6), g.normal(size=len(x))), -1)
    res = tp.ridge_cv(x@x.T, y, np.repeat(np.arange(20), 33))
    assert np.array(res["val_r2_per_fold"]).shape == (5, 2) and min(r[0] for r in res["val_r2_per_fold"]) > .99


# —————————————————————————— 工具端到端 ——————————————————————————

def run_tool(*args):
    return subprocess.run([sys.executable, str(ROOT/"tools"/"tubelet_probe.py"), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_tool_end_to_end_fake(tmp_path):
    out = tmp_path/"probe"
    for shard in ("0/2", "1/2"):
        r = run_tool("encode", "--trajectories", "10", "--output", str(out), "--fake-encoder", "--device", "cpu", "--shard", shard)
        assert r.returncode == 0, r.stderr
    m = json.loads((out/"0003"/"manifest.json").read_text())
    assert m["features"]["pair"]["clips"][0] == [0, 1] and m["features"]["dup"]["shape"] == [34, 256, 1408]
    assert m["generation"]["identity"]["evidence_role"] == "synthetic_development_only"
    assert np.array_equal(np.load(out/"0003"/"states.npy"), tp.trajectory(3)[0])
    r = run_tool("fit", "--trajectories", "10", "--output", str(out), "--device", "cpu")
    assert r.returncode == 0, r.stderr
    rep = json.loads((out/"report.json").read_text())
    assert rep["evidence_role"] == "synthetic_development_only" and rep["samples"] == 330
    assert rep["generation_identity"] == m["generation"]["identity"] and len(rep["manifest_sha256"]) == 10
    assert set(rep["fit_source_sha256"]) == set(rep["generation_identity"]["generation_source_sha256"])
    assert rep["decision"]["rule"]["readout_selection"].startswith("nested")
    assert set(rep["results"]) == set(tp.MODES)
    j = tp.TARGETS.index("omega")
    # 假编码器也是按 tubelet 取块：单帧复制读不出角速度，两帧能读出一部分
    assert rep["results"]["dup"]["pool4"]["r2_mean"][j] < .1
    assert rep["results"]["pair"]["pool4"]["r2_mean"][j] > rep["results"]["dup"]["pool4"]["r2_mean"][j]+.2
    assert rep["results"]["dup"]["pool4"]["r2_mean"][0] > .9
    # 断点续跑：已完成的跳过；缺一条时 fit 拒绝；特征被改时 fit 拒绝
    before = (out/"0004"/"pair.npy").stat().st_mtime_ns
    assert run_tool("encode", "--trajectories", "10", "--output", str(out), "--fake-encoder", "--device", "cpu").returncode == 0
    assert (out/"0004"/"pair.npy").stat().st_mtime_ns == before
    (out/"0004"/"COMPLETE.json").unlink()
    r = run_tool("fit", "--trajectories", "10", "--output", str(out), "--device", "cpu")
    assert r.returncode != 0 and "没编码完" in r.stderr
    assert run_tool("encode", "--trajectories", "10", "--output", str(out), "--fake-encoder", "--device", "cpu").returncode == 0
    arr = np.load(out/"0005"/"dup.npy"); arr[0, 0, 0] += 1; np.save(out/"0005"/"dup.npy", arr)
    r = run_tool("fit", "--trajectories", "10", "--output", str(out), "--device", "cpu")
    assert r.returncode != 0 and "SHA mismatch" in r.stderr


def test_tool_jobs_and_guards(tmp_path):
    r = run_tool("jobs", "--shards", "3", "--output", "../features-v8/tubelet-probe-200")
    lines = [x for x in r.stdout.splitlines() if x.startswith("python")]
    assert len(lines) == 4 and all("--trajectories 200" in x for x in lines)
    assert [x.split("--shard ")[1] for x in lines[:3]] == ["0/3", "1/3", "2/3"] and " fit " in lines[3]
    assert r.stdout.index("T1") < r.stdout.index("T2")
    r = run_tool("encode", "--trajectories", "10", "--output", str(tmp_path), "--device", "cpu")
    assert r.returncode != 0 and "--upstream-root" in r.stderr
    assert run_tool("fit", "--trajectories", "4", "--output", str(tmp_path)).returncode != 0


# —————————————————————————— 缓存生成身份（S22-fix1） ——————————————————————————

def tool_module():
    spec = importlib.util.spec_from_file_location("tubelet_probe_tool", ROOT/"tools"/"tubelet_probe.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def encoded(tmp_path_factory):
    out = tmp_path_factory.mktemp("ident")/"probe"
    tool = tool_module()
    tool.main(["encode", "--trajectories", "5", "--output", str(out), "--fake-encoder", "--device", "cpu"])
    return out


def rewrite(folder, change):
    """把第 folder 条的清单里的生成身份按 change 改掉，再把身份 SHA、清单 SHA、COMPLETE 全部改成自洽的。"""
    from po_jepa.runtime import cache
    tool = tool_module()
    m = json.loads((folder/"manifest.json").read_bytes())
    change(m)
    m["generation"]["identity_sha256"] = tool.identity_sha(m["generation"]["identity"])
    (folder/"manifest.json").write_bytes(cache._json_bytes(m))
    (folder/"COMPLETE.json").write_bytes(cache._json_bytes(dict(manifest_sha256=cache.file_sha(folder/"manifest.json"))))


def real_identity(tool):
    from po_jepa.runtime import cache
    arch = tool.v7_architecture()
    report = vjepa.CheckpointReport(Path("w.pth"), cache.CHECKPOINT_SHA, "target_encoder", (), (), True, arch)
    return tool.generation_identity(tool.REAL, tool.encoder_identity(report, tool.v7_load_spec("w.pth"), arch))


def load(tool, out, n=5):
    from po_jepa.runtime import cache
    return tool.load(out, n, "pair", np, cache)


@pytest.fixture
def cache_copy(encoded, tmp_path):
    import shutil
    out = tmp_path/"probe"; shutil.copytree(encoded, out)
    return out


def test_identity_recorded_and_consistent_cache_loads(cache_copy):
    tool = tool_module()
    _, _, ident, shas = load(tool, cache_copy)
    assert ident["evidence_role"] == tool.SYNTHETIC and ident["definition"] == tp.definition()
    assert ident["definition"]["clips"]["pair"][0] == [0, 1] and len(shas) == 5
    # 真实编码器身份（v7 权重、上游提交、架构、加载规则、预处理）能通过：先证明正例可过，下面的反例才只差一项
    real = real_identity(tool)
    for t in range(5):
        rewrite(cache_copy/f"{t:04d}", lambda m: m["generation"].__setitem__("identity", copy.deepcopy(real)))
    assert load(tool, cache_copy)[2]["evidence_role"] == tool.REAL


@pytest.mark.parametrize("name,change,match", [
    ("mixed_source", lambda i: i["generation_source_sha256"].__setitem__("src/po_jepa/tubelet_probe.py", "0"*64), "混了两套"),
    ("mixed_role", lambda i: i.__setitem__("evidence_role", "synthetic_development_only") or
        i.__setitem__("encoder", copy.deepcopy(tool_module().FAKE_ENCODER)) or i.__setitem__("preprocess", None), "混了两套"),
])
def test_mixed_identities_rejected(cache_copy, name, change, match):
    """所有文件 SHA 都自洽，只有一条轨迹的生成身份与其他条不同 → 拒绝。"""
    tool = tool_module(); real = real_identity(tool)
    for t in range(5):
        rewrite(cache_copy/f"{t:04d}", lambda m: m["generation"].__setitem__("identity", copy.deepcopy(real)))
    load(tool, cache_copy)
    rewrite(cache_copy/"0003", lambda m: change(m["generation"]["identity"]))
    with pytest.raises(ValueError, match=match):
        load(tool, cache_copy)


@pytest.mark.parametrize("field,value", [
    ("checkpoint_sha256", "1"*64), ("upstream_commit", "deadbeef"), ("encoder_key", "encoder"),
    ("missing_keys", ["x"]), ("initialization_seed", 1), ("architecture", None), ("load_rules", None), ("extra_field", 1),
])
def test_real_role_needs_v7_encoder(cache_copy, field, value):
    """全部轨迹一致地声称真实编码器，但编码器身份有一项不是 v7 的 → 拒绝（不再只看 evidence_role 字符串）。"""
    tool = tool_module(); real = real_identity(tool)
    if field == "architecture":
        value = dict(real["encoder"]["architecture"], tubelet_size=1)
    if field == "load_rules":
        value = dict(real["encoder"]["load_rules"], strip_prefixes=["module."])
    real["encoder"][field] = value
    for t in range(5):
        rewrite(cache_copy/f"{t:04d}", lambda m: m["generation"].__setitem__("identity", copy.deepcopy(real)))
    with pytest.raises(ValueError, match="v7"):
        load(tool, cache_copy)


def test_real_role_needs_v7_preprocess(cache_copy):
    tool = tool_module()
    for change in (lambda p: p.__setitem__("rule", "one_generator_for_all_frames"),
                   lambda p: p["spec"].__setitem__("crop_size", 224)):
        real = real_identity(tool); change(real["preprocess"])
        for t in range(5):
            rewrite(cache_copy/f"{t:04d}", lambda m: m["generation"].__setitem__("identity", copy.deepcopy(real)))
        with pytest.raises(ValueError, match="preprocess"):
            load(tool, cache_copy)


def test_definition_schema_and_clips_checked(cache_copy):
    tool = tool_module()
    # 全部轨迹一致，但探针定义与当前代码不同（pair 帧号反序）→ 拒绝
    def reverse(m):
        m["generation"]["identity"]["definition"]["clips"]["pair"] = [[b, a] for a, b in tp.clip_indices("pair")]
        m["features"]["pair"]["clips"] = [[b, a] for a, b in tp.clip_indices("pair")]
    for t in range(5):
        rewrite(cache_copy/f"{t:04d}", reverse)
    with pytest.raises(ValueError, match="探针定义"):
        load(tool, cache_copy)


def test_feature_clips_and_schema_checked(cache_copy):
    tool = tool_module()
    rewrite(cache_copy/"0002", lambda m: m["features"]["pair"].__setitem__("clips", [[b, a] for a, b in tp.clip_indices("pair")]))
    with pytest.raises(ValueError, match="帧号或形状"):
        load(tool, cache_copy)


def test_old_schema_and_bad_identity_sha_rejected(cache_copy):
    tool = tool_module()
    rewrite(cache_copy/"0001", lambda m: m.__setitem__("schema", "s22_tubelet_probe_v1"))
    with pytest.raises(ValueError, match="清单格式"):
        load(tool, cache_copy)


def test_identity_sha_must_match_content(cache_copy):
    from po_jepa.runtime import cache
    tool = tool_module(); folder = cache_copy/"0004"
    m = json.loads((folder/"manifest.json").read_bytes())
    m["generation"]["identity"]["generation_source_sha256"]["tools/tubelet_probe.py"] = "0"*64     # 不改身份 SHA
    (folder/"manifest.json").write_bytes(cache._json_bytes(m))
    (folder/"COMPLETE.json").write_bytes(cache._json_bytes(dict(manifest_sha256=cache.file_sha(folder/"manifest.json"))))
    with pytest.raises(ValueError, match="身份与它的 SHA"):
        load(tool, cache_copy)


def test_resume_refuses_other_identity(cache_copy):
    """续跑：同一身份的已完成轨迹跳过；身份不同的已完成轨迹不覆盖、不混用，直接报错。"""
    tool = tool_module()
    before = (cache_copy/"0002"/"pair.npy").stat().st_mtime_ns
    tool.main(["encode", "--trajectories", "5", "--output", str(cache_copy), "--fake-encoder", "--device", "cpu"])
    assert (cache_copy/"0002"/"pair.npy").stat().st_mtime_ns == before
    saved = {n: (cache_copy/"0002"/n).read_bytes() for n in ("manifest.json", "COMPLETE.json")}
    rewrite(cache_copy/"0002", lambda m: m["generation"]["identity"]["generation_source_sha256"].__setitem__("src/po_jepa/vjepa.py", "0"*64))
    manifest = (cache_copy/"0002"/"manifest.json").read_bytes()
    with pytest.raises(SystemExit, match="生成身份与本次不同"):
        tool.main(["encode", "--trajectories", "5", "--output", str(cache_copy), "--fake-encoder", "--device", "cpu"])
    assert (cache_copy/"0002"/"manifest.json").read_bytes() == manifest
    assert (cache_copy/"0002"/"pair.npy").stat().st_mtime_ns == before
    # 已完成但清单被改过（COMPLETE 里的 SHA 对不上）：也不覆盖，报错（先把 0002 复原，只差这一项）
    for n, b in saved.items():
        (cache_copy/"0002"/n).write_bytes(b)
    tool.main(["encode", "--trajectories", "5", "--output", str(cache_copy), "--fake-encoder", "--device", "cpu"])
    from po_jepa.runtime import cache
    (cache_copy/"0003"/"COMPLETE.json").write_bytes(cache._json_bytes(dict(manifest_sha256="0"*64)))
    with pytest.raises(SystemExit, match="清单 SHA 不符"):
        tool.main(["encode", "--trajectories", "5", "--output", str(cache_copy), "--fake-encoder", "--device", "cpu"])
