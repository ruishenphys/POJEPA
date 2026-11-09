"""S22 两帧 tubelet 探针（tubelet_probe.py、tools/tubelet_probe.py）：单摆物理、画面、编码帧号、对偶岭回归、判据、端到端。"""
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
    for mode, rows in (("dup", (0, 5, 33)), ("pair", (0, 12, 32))):
        for r in rows:
            p, c = tp.clip_indices(mode)[r]
            ref = enc(torch.stack((x[p:p+1], x[c:c+1]), 2)).double().numpy()[0]
            assert np.allclose(out[mode][r], ref, atol=1e-6)
    with pytest.raises(ValueError):
        tp.encode_trajectory(frames[:33], 2, fake=True)


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


def test_decide_rule():
    def table(pair_omega, dup_omega, dup_pos=.99):
        res = {}
        for mode, om in (("pair", pair_omega), ("dup", dup_omega), ("dup_concat", 0.)):
            res[mode] = {ro: dict(r2_mean=[dup_pos if mode == "dup" else .99]*2+[om, om]) for ro in tp.READOUTS}
        return res
    assert tp.decide(table(.95, .1))["adopt_two_frame"]
    assert not tp.decide(table(.85, .1))["adopt_two_frame"]
    assert not tp.decide(table(.95, .9))["adopt_two_frame"]                 # 提升不到 0.1
    assert not tp.decide(table(.95, .1, dup_pos=.5))["adopt_two_frame"]     # 位置都读不出：探针无效
    t = table(.5, .1); t["pair"]["pool4"]["r2_mean"][2] = .93
    d = tp.decide(t)
    assert d["best_readout"] == "pool4" and d["adopt_two_frame"]


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
    assert m["generation"]["evidence_role"] == "synthetic_development_only"
    assert np.array_equal(np.load(out/"0003"/"states.npy"), tp.trajectory(3)[0])
    r = run_tool("fit", "--trajectories", "10", "--output", str(out), "--device", "cpu")
    assert r.returncode == 0, r.stderr
    rep = json.loads((out/"report.json").read_text())
    assert rep["evidence_role"] == "synthetic_development_only" and rep["samples"] == 330
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
