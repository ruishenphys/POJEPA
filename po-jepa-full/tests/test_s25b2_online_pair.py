"""S25b 第二段之一：探针、规划、特征噪声下限三个工具的在线两帧（pair）编码（docs/segments/S25b2.md）。

定义：Z_t = E([P(I_{t−1}), P(I_t)])，前一帧永远是 1 tick 前的画面，时间顺序（前，后）。期望值都从画面和物理独立重算
（variant_trajectory、draw、fake_tubelet_encoder），不拿被测工具自己的函数当参照。
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from po_jepa import datasets, simulation, systems, tubelet_probe, variants
from po_jepa.runtime import cache
from test_s25c_long_horizon import active, real  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"src"
PAIR40 = datasets.parse("pendulum~tubelet=pair:40")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"s25b2_{name}", ROOT/"tools"/f"{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PO_JEPA_")}
    env.update(PYTHONPATH=str(SRC), OMP_NUM_THREADS="2")
    return env


def _run(tool, *args):
    return subprocess.run([sys.executable, "-B", str(ROOT/"tools"/tool), *args], env=_clean_env(), capture_output=True, text=True,
                          cwd=ROOT)


def _spec(system, knobs, count=200):
    return datasets.DatasetSpec(system, count, variants.format(system, variants.parse(system, knobs)))


def _per_tick(c):
    """下达动作逐 tick 展开 [32,2]（独立写，不调 systems.command_ticks）。"""
    out = np.zeros((32, 2))
    for s, e, val in zip(c.starts_ticks, c.ends_ticks, c.values):
        out[s:e] = val
    return out


def _tick_controls(tid, values):
    return simulation.TrajectoryControls(tid, np.arange(32), np.arange(1, 33), np.asarray(values, dtype=np.float64))


def _fake_pair(prev, cur):
    """假 tubelet 编码器对一对画面（uint8 [256,256,3]）的特征 [256,1408]：像素 /255，时间顺序（前，后）。"""
    x = torch.from_numpy(np.stack((prev, cur))).permute(0, 3, 1, 2).float()/255.          # [2,3,256,256]
    return tubelet_probe.fake_tubelet_encoder()(torch.stack((x[0], x[1]), 1)[None])[0]


def _d(x, y):
    ln = lambda z: F.layer_norm(z, (z.shape[-1],), eps=1e-5)
    return float((ln(x)-ln(y)).abs().mean())


# ———————————————————————————— 探针 ————————————————————————————

@pytest.mark.parametrize("name,knobs", [("pendulum", "pixel_noise=5,movers=1,tubelet=pair"), ("bouncing_balls", "gain=2,tubelet=pair"),
                                        ("wave", "speed=0.25,tubelet=pair"), ("cart_pole", "delay=1,tubelet=pair"),
                                        ("spring_chain", "n=3,tubelet=pair")])
def test_counterfactual_clip_is_the_last_two_ticks_of_the_counterfactual_path(active, name, knobs):
    active(_spec(name, knobs)); probe = _load("probe_actions"); S = systems.get(); v = datasets.active().knobs
    probe._FULL.clear()
    tid, donor, start = 181, 182, 5
    states34, frames34 = S.frames_with_prev(tid)                # 下标 f = tick f−1
    full = S.variant_trajectory(tid, v)
    own, other = _per_tick(systems.commanded_controls(tid)), _per_tick(systems.commanded_controls(donor))
    for tau in (1, 2, 8):
        # 换成自己的动作 = 数据集本身：两帧就是数据集的 tick start+τ−1、start+τ（frames_with_prev 下标 start+τ、start+τ+1）
        true_end, cf, clip = probe.counterfactual_end(S, simulation, tid, tid, start, tau, with_prev=True)
        assert clip.shape == (2, 256, 256, 3) and clip.dtype == np.uint8
        assert np.array_equal(clip, frames34[start+tau:start+tau+2]) and np.array_equal(cf, states34[start+tau+1])
        # 换成 donor 的动作：独立续算（窗口前是自己的下达动作，窗口内是 donor 的）
        mixed = np.concatenate((own[:start], other[start:start+tau], own[start+tau:]))
        path = S.variant_trajectory(tid, v, controls=_tick_controls(tid, mixed), start=start, state=full[start], ticks=tau)
        true_end, cf, clip = probe.counterfactual_end(S, simulation, tid, donor, start, tau, with_prev=True)
        assert np.array_equal(clip[1], S.draw(path[-1:], tid, v, ticks=[start+tau])[0])
        assert np.array_equal(clip[0], S.draw(path[-2:-1], tid, v, ticks=[start+tau-1])[0])
        assert np.array_equal(true_end, states34[start+tau+1]) and np.array_equal(cf, S.hidden(path[-1:])[0])
        # 不带 with_prev 的返回没变：一帧，就是两帧里的后一帧
        _, _, single = probe.counterfactual_end(S, simulation, tid, donor, start, tau)
        assert single.shape == (256, 256, 3) and np.array_equal(single, clip[1])
        if tau == 1:      # 前一帧是真实起点那一帧（不是反事实终点自己，也不是真实终点的前一帧以外的东西）
            assert np.array_equal(clip[0], frames34[start+1]) and not np.array_equal(clip[0], clip[1])
        elif tau == 8:    # 前一帧在反事实轨迹上，不是真实轨迹的那一帧（τ = 2、delay = 1 时窗口第一个 tick 执行的还是自己的动作）
            assert not np.array_equal(clip[0], frames34[start+tau])


def test_counterfactual_with_prev_is_second_order_only(active):
    probe = _load("probe_actions")
    for system, knobs in (("double_well", "gain=2"), ("push", "")):
        active(_spec(system, knobs)); probe._FULL.clear()
        with pytest.raises(ValueError, match="二阶场景"):
            probe.counterfactual_end(systems.get(), simulation, 181, 182, 5, 2, with_prev=True)


def _probe(tmp_path, variant, *extra, count="200", name="probe.json"):
    out = tmp_path/name
    r = _run("probe_actions.py", "--system", "pendulum", "--trajectories", count, *(("--variant", variant) if variant else ()),
             "--fake-encoder", "--device", "cpu", "--max-trajectories", "1", "--output", str(out), *extra)
    return r, out


def test_probe_report_on_pair_dataset_equals_independent_two_frame_features(tmp_path, active):
    r, out = _probe(tmp_path, "tubelet=pair")
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["tubelet"] == "pair" and rep["dataset"] == "pendulum~tubelet=pair-200" and "两帧编码" in rep["motion_units"]
    active(_spec("pendulum", "tubelet=pair")); S = systems.get(); v = datasets.active().knobs
    tid = 180; donor = datasets.donor_id(tid)
    _, frames34 = S.frames_with_prev(tid)
    z = lambda t: _fake_pair(frames34[t], frames34[t+1])          # tick t 的特征：(tick t−1, tick t) = 下标 (t, t+1)
    full = S.variant_trajectory(tid, v)
    own, other = _per_tick(systems.commanded_controls(tid)), _per_tick(systems.commanded_controls(donor))
    assert {(x["start_tick"], x["duration_cycles"]) for x in rep["rows"]} == {
        (s, t) for s in datasets.test_start_ticks(tid) for t in (1, 2, 4, 8) if s+t <= 32}
    off_path = 0
    for row in rep["rows"]:
        s, tau = row["start_tick"], row["duration_cycles"]
        mixed = np.concatenate((own[:s], other[s:s+tau], own[s+tau:]))
        path = S.variant_trajectory(tid, v, controls=_tick_controls(tid, mixed), start=s, state=full[s], ticks=tau)
        prev, end = S.draw(path[-2:], tid, v, ticks=[s+tau-1, s+tau])
        assert row["identity"] == pytest.approx(_d(z(s), z(s+tau)), rel=1e-6)
        want = _d(_fake_pair(prev, end), z(s+tau))
        assert row["oracle_delta_act"] == pytest.approx(want, rel=1e-6)
        # 测试有分辨力：时间顺序反了、前一帧取成终点自己（单帧复制），值都不同
        for wrong in (_fake_pair(end, prev), _fake_pair(end, end)):
            assert abs(_d(wrong, z(s+tau))-row["oracle_delta_act"]) > 1e-4*row["identity"]
        # 前一帧取成真实轨迹的那一帧：只要反事实轨迹在 tick s+τ−1 已经和真实轨迹画得不一样，值就不同
        if not np.array_equal(prev, frames34[s+tau]):
            assert tau > 1 and abs(_d(_fake_pair(frames34[s+tau], end), z(s+tau))-row["oracle_delta_act"]) > 1e-6*row["identity"]
            off_path += 1
    assert off_path >= 5


def test_probe_on_dup_second_order_dataset_is_unchanged(tmp_path, active):
    """不带 tubelet=pair：仍是单帧假编码器对 33 帧各自的特征（S25a 的口径）。"""
    sys.path.insert(0, str(ROOT/"tools"))
    import encode_v7
    r, out = _probe(tmp_path, "")
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["tubelet"] == "dup" and rep["motion_units"] == "构形（po_jepa.configuration），二阶场景单帧复制编码看不到速度"
    active(_spec("pendulum", "")); _, frames = systems.get().frames(180)
    z = encode_v7._fake_encoder(torch)(torch.from_numpy(frames).permute(0, 3, 1, 2).float()/255.)
    for row in rep["rows"]:
        assert row["identity"] == pytest.approx(_d(z[row["start_tick"]], z[row["start_tick"]+row["duration_cycles"]]), rel=1e-6)


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    root = tmp_path_factory.mktemp("s25b2")
    for name, variant in (("dup", ""), ("pair", "tubelet=pair")):
        args = ["--system", "pendulum", "--trajectories", "40", "--output", str(root/name), *(("--variant", variant) if variant else ())]
        for mode, extra in (("encode", ["--fake-encoder", "--device", "cpu", "--splits", "test"]), ("finalize", ["--splits", "test"])):
            r = _run("encode_v7.py", mode, *args, *extra)
            assert r.returncode == 0, r.stderr[-3000:]
    return root


def test_probe_reencoding_matches_the_pair_cache_and_not_the_dup_cache(tmp_path, caches):
    r, out = _probe(tmp_path, "tubelet=pair", "--features-root", str(caches/"pair"), count="40")
    assert r.returncode == 0, r.stderr[-3000:]
    assert json.loads(out.read_text(encoding="utf-8"))["rows"][0]["trajectory_id"] == 36
    r, out = _probe(tmp_path, "tubelet=pair", "--features-root", str(caches/"dup"), count="40", name="bad.json")
    assert r.returncode != 0 and "重新编码与缓存不一致" in r.stderr and not out.exists()


class _Clip:
    """记录输入的 FrozenClipEncoder 替身：只有 encode_clip（调单帧的 encode_target 会直接报错）。"""
    built = []

    def __init__(self, loaded, *, preprocess, encoder_device, autocast_dtype):
        assert preprocess == cache.preprocess_spec() and autocast_dtype is None
        self.loaded, self.calls = loaded, []
        self.project = tubelet_probe.fake_tubelet_encoder()
        _Clip.built.append(self)

    def encode_clip(self, prev, cur):
        self.calls.append((prev.clone(), cur.clone()))
        return self.project(torch.stack((prev, cur), 2))


def _unit_preprocess(rgb, *, spec, generator):
    """代替 vjepa.preprocess_rgb：和 --fake-encoder 路径同一个输入（[1,3,256,256]，0–1）。"""
    assert rgb.shape[0] == 1
    return rgb.permute(0, 3, 1, 2).float()/255.


@pytest.fixture
def real_clip(monkeypatch):
    """真实编码路径的替身：不载权重，两帧编码器换成 _Clip；单帧的 FrozenFrameEncoder 一被构造就报错。"""
    from po_jepa import vjepa
    _Clip.built.clear()
    monkeypatch.setattr(vjepa, "build_encoder_architecture", lambda **kw: "build")
    monkeypatch.setattr(vjepa, "load_local_encoder", lambda build, checkpoint: "loaded")
    monkeypatch.setattr(vjepa, "preprocess_rgb", _unit_preprocess)
    monkeypatch.setattr(tubelet_probe, "FrozenClipEncoder", _Clip)
    def forbidden(*args, **kw):
        raise AssertionError("两帧数据集不能用单帧复制的 FrozenFrameEncoder")
    monkeypatch.setattr(vjepa, "FrozenFrameEncoder", forbidden)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    return _Clip


def test_probe_real_path_uses_the_clip_encoder_in_time_order(tmp_path, active, real_clip, monkeypatch):
    r, fake_out = _probe(tmp_path, "tubelet=pair", name="fake.json")
    assert r.returncode == 0, r.stderr[-3000:]
    active(_spec("pendulum", "tubelet=pair"))
    probe = _load("probe_actions"); out = tmp_path/"real.json"
    monkeypatch.setattr(sys, "argv", ["probe_actions.py", "--system", "pendulum", "--trajectories", "200", "--variant", "tubelet=pair",
                                      "--upstream-root", str(tmp_path), "--checkpoint", str(tmp_path/"x.pt"), "--device", "cpu",
                                      "--max-trajectories", "1", "--output", str(out)])
    probe.main()
    enc, = real_clip.built
    _, frames34 = systems.get().frames_with_prev(180)
    pre = torch.from_numpy(frames34).permute(0, 3, 1, 2).float()/255.
    prev, cur = enc.calls[0]                                    # 整条轨迹：33 个 tubelet 一批，(第 k 帧, 第 k+1 帧)
    assert torch.equal(prev, pre[:-1]) and torch.equal(cur, pre[1:])
    assert all(p.shape[0] == 1 for p, _ in enc.calls[1:]) and len(enc.calls) == 1+len(json.loads(out.read_text())["rows"])
    # 预处理和投影都与 --fake-encoder 相同，所以两份报告的数值应相同
    real_rows, fake_rows = json.loads(out.read_text())["rows"], json.loads(fake_out.read_text())["rows"]
    assert json.loads(out.read_text())["encoder"] == "vjepa2-vitg"
    for a, b in zip(real_rows, fake_rows, strict=True):
        assert a["oracle_delta_act"] == pytest.approx(b["oracle_delta_act"], rel=1e-5) and a["identity"] == pytest.approx(b["identity"], rel=1e-5)


# ———————————————————————————— 规划 ————————————————————————————

@pytest.mark.parametrize("name,knobs", [("pendulum", "delay=1,tubelet=pair"), ("cart_pole", "tubelet=pair"), ("double_well", "gain=2")])
def test_rollout_path_is_every_tick_of_the_continuation(active, name, knobs):
    active(_spec(name, knobs)); plan = _load("plan_v7"); S = systems.get(); v = datasets.active().knobs
    tid, start = 181, 5
    full = S.variant_trajectory(tid, v)
    history = _per_tick(systems.commanded_controls(tid)); history[start:] = 0.
    actions = np.random.default_rng(0).uniform(-.5, .5, (4, 2))
    path = plan.rollout_path(S, v, False, tid, full[start], start, actions, history)
    assert path.shape[0] == 5 and np.array_equal(path[0], full[start])
    values = np.zeros((32, 2)); values[:start] = history[:start]
    for j in range(1, 5):                                       # 第 j 个 = 只走 j 个 tick 的终点（独立续算）
        values[start:start+j] = actions[:j]; values[start+j:] = 0.
        want = S.variant_trajectory(tid, v, controls=_tick_controls(tid, values), start=start, state=full[start], ticks=j)[-1]
        assert np.array_equal(path[j], want)
    assert np.array_equal(plan.rollout(S, v, False, tid, full[start], start, actions, history), path[-1])


def test_plan_requires_the_same_tubelet_for_training_and_test():
    plan = _load("plan_v7")
    pair, dup = _spec("pendulum", "tubelet=pair"), _spec("pendulum", "")
    pair_gain, dup_gain = _spec("pendulum", "gain=2,tubelet=pair"), _spec("pendulum", "gain=2")
    for a, b in ((pair, pair), (dup, dup), (pair, pair_gain), (dup, dup_gain)):
        plan.check_same_tubelet(a, b)
    for a, b in ((pair, dup), (dup, pair), (pair, dup_gain), (dup, pair_gain)):
        with pytest.raises(SystemExit, match="tubelet"):
            plan.check_same_tubelet(a, b)
    # 在建模型、读 checkpoint 之前核
    text = (ROOT/"tools/plan_v7.py").read_text(encoding="utf-8")
    assert text.index("check_same_tubelet(base, test_spec)") < text.index("comparison.build_condition(condition, device)")


def _job(tmp_path, dataset):
    job = tmp_path/"job"; job.mkdir(exist_ok=True)
    (job/"job.json").write_text(json.dumps(dict(seed=1703, updates=2000, dataset=dataset, condition="identity")))
    return job


SMALL = ("--episodes", "1", "--horizon", "3", "--samples", "8", "--iterations", "1", "--oracle-samples", "4")


def test_plan_cli_on_pair_job_and_test_variants(tmp_path):
    job = _job(tmp_path, "pendulum~tubelet=pair-200")
    base = ("plan_v7.py", "--job", str(job), "--fake-encoder", "--device", "cpu", *SMALL, "--replan", "1")
    r = _run(*base, "--output", str(tmp_path/"a.json"))
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads((tmp_path/"a.json").read_text(encoding="utf-8"))
    assert rep["tubelet"] == "pair" and rep["dataset"] == rep["train_dataset"] == "pendulum~tubelet=pair-200"
    assert "1 tick" in rep["tubelet_prev_frame"] and all(np.isfinite(rep["episodes"][0][k]) for k in ("zero", "model", "oracle", "random"))
    # --test-variant 是整个替换：不写 tubelet=pair 就成了单帧复制的测试集，拒绝；写上就能跑
    r = _run(*base, "--test-variant", "gain=2", "--output", str(tmp_path/"b.json"))
    assert r.returncode != 0 and "tubelet" in r.stderr and not (tmp_path/"b.json").exists()
    r = _run(*base, "--test-variant", "gain=2,tubelet=pair", "--no-baselines", "--output", str(tmp_path/"c.json"))
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads((tmp_path/"c.json").read_text(encoding="utf-8"))
    assert (rep["tubelet"], rep["train_dataset"], rep["dataset"]) == ("pair", "pendulum~tubelet=pair-200", "pendulum~gain=2,tubelet=pair-200")
    # 单帧复制上训练的任务不能在两帧测试集上规划；不带 tubelet 的照旧是 dup，报告没有 tubelet_prev_frame
    job = _job(tmp_path, "pendulum-200")
    base = ("plan_v7.py", "--job", str(job), "--fake-encoder", "--device", "cpu", *SMALL, "--replan", "1", "--no-baselines")
    r = _run(*base, "--test-variant", "tubelet=pair", "--output", str(tmp_path/"d.json"))
    assert r.returncode != 0 and "tubelet" in r.stderr and not (tmp_path/"d.json").exists()
    r = _run(*base, "--output", str(tmp_path/"e.json"))
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads((tmp_path/"e.json").read_text(encoding="utf-8"))
    assert rep["tubelet"] == "dup" and "tubelet_prev_frame" not in rep


def _plan_in_process(tmp_path, monkeypatch, active, dataset_knobs, *extra, fake=True):
    """在本进程跑 plan_v7.main，记下每次送编码器的画面和每次续算。→ (报告, 送编码器的画面列表, 续算记录, 模块)。"""
    sys.path.insert(0, str(ROOT/"tools"))
    import encode_v7
    spec = _spec("pendulum", dataset_knobs); active(spec)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    plan = _load("plan_v7")
    encoded, paths = [], []
    original_encode, original_path = encode_v7._encode_trajectory, plan.rollout_path
    def spy_encode(encoder, frames, *args, **kw):
        encoded.append((np.array(frames), kw.get("pair", False)))
        return original_encode(encoder, frames, *args, **kw)
    def spy_path(system, v, field_scene, tid, state, start, actions, history):
        path = original_path(system, v, field_scene, tid, state, start, actions, history)
        paths.append(dict(tid=tid, start=start, actions=np.array(actions), path=path.copy()))
        return path
    monkeypatch.setattr(encode_v7, "_encode_trajectory", spy_encode)
    monkeypatch.setattr(plan, "rollout_path", spy_path)
    out = tmp_path/"plan.json"
    monkeypatch.setattr(sys, "argv", ["plan_v7.py", "--job", str(_job(tmp_path, spec.tag)), "--device", "cpu", "--output", str(out),
                                      *(("--fake-encoder",) if fake else ("--upstream-root", str(tmp_path), "--checkpoint", str(tmp_path/"x.pt"))),
                                      *SMALL, "--no-baselines", *extra])       # extra 在后：同名参数以它为准
    plan.main()
    return json.loads(out.read_text(encoding="utf-8")), encoded, paths, plan


@pytest.mark.parametrize("replan", [1, 2])
def test_plan_encodes_goal_start_and_replan_frames_with_the_one_tick_earlier_frame(tmp_path, monkeypatch, active, replan):
    rep, encoded, paths, plan = _plan_in_process(tmp_path, monkeypatch, active, "tubelet=pair", "--replan", str(replan))
    S = systems.get(); v = datasets.active().knobs
    tid, start, H = 180, 0, 3                                    # 第一个回合：测试段第一条、START_TICKS[0]
    ep, = rep["episodes"]
    assert (ep["trajectory_id"], ep["start_tick"], ep["goal_valid"]) == (tid, start, True)
    assert all(pair and frames.shape == (2, 256, 256, 3) for frames, pair in encoded)
    assert len(encoded) == 2+(H-1)//replan                       # 目标、起点、每次执行后还有剩余时的重规划
    _, frames34 = S.frames_with_prev(tid)
    full = S.variant_trajectory(tid, v)
    # 目标：目标动作（流 1704，每 2 tick 一段）从起点走 H 个 tick，与“不动”的摆锤位置差 ≥ 0.15 才用（独立重抽）；
    # 前一帧是它在 tick start+H−1 的状态
    bob = lambda q: np.array([np.sin(q[0]), -np.cos(q[0])])
    go = lambda u: S.variant_trajectory(tid, v, controls=_tick_controls(tid, np.concatenate((np.zeros((start, 2)), u, np.zeros((32-start-H, 2))))),
                                        start=start, state=full[start], ticks=H)
    still = bob(go(np.zeros((H, 2)))[-1]); goal_rng = np.random.default_rng(1704)
    for _ in range(20):
        gu = np.repeat(goal_rng.uniform(-.5, .5, ((H+1)//2, 2)), 2, 0)[:H]
        goal_path = go(gu)
        if np.linalg.norm(bob(goal_path[-1])-still) >= .15:
            break
    else:
        raise AssertionError("没抽到有效目标")
    assert ep["zero"] == pytest.approx(float(np.linalg.norm(still-bob(goal_path[-1]))), abs=1e-12)
    want_goal = S.draw(goal_path[-2:], tid, v, ticks=[start+H-1, start+H])
    assert np.array_equal(encoded[0][0], want_goal) and not np.array_equal(want_goal[0], want_goal[1])
    # 起点：测试轨迹自己的 (tick start−1, tick start)；start = 0 时前一帧是 t = −1 那一帧
    assert np.array_equal(encoded[1][0], frames34[start:start+2])
    # 重规划：--no-baselines 时最后 ⌈H/replan⌉ 次续算就是模型规划实际执行的那几段（之前的是“不动”参照和目标），
    # 首尾相接地从起点走到 tick start+H；前一帧是执行轨迹在每段终点前 1 tick 的状态
    executed, t, state = paths[-(-(-H//replan)):], start, full[start]
    for q in executed:
        assert q["tid"] == tid and q["start"] == t and np.array_equal(q["path"][0], state) and len(q["actions"]) == min(replan, start+H-t)
        t += len(q["actions"]); state = q["path"][-1]
    assert t == start+H and len(executed) == -(-H//replan)
    for k, p in enumerate(executed[:-1]):
        end = p["start"]+len(p["actions"])
        want = S.draw(p["path"][-2:], tid, v, ticks=[end-1, end])
        assert np.array_equal(encoded[2+k][0], want)
        previous = encoded[1+k][0]
        if replan == 1:   # 每次只走 1 tick：前一帧就是上一次送编码器的后一帧（上一步执行后的画面）
            assert np.array_equal(want[0], previous[1])
        else:             # 走 2 tick：前一帧是中间那个 tick，不是上一次观测的画面
            assert not np.array_equal(want[0], previous[1])
    # 报告里的模型误差 = 执行轨迹终点到目标的受控构形距离（独立算：摆锤 (sin θ, −cos θ)）
    assert ep["model"] == pytest.approx(float(np.linalg.norm(bob(executed[-1]["path"][-1])-bob(goal_path[-1]))), abs=1e-12)


def test_plan_start_frame_at_a_later_tick_uses_the_previous_dataset_frame(tmp_path, monkeypatch, active):
    """第 21 个回合轮到 START_TICKS[1] = 5（pendulum:200 的测试段 20 条）：起点的两帧是数据集的 tick 4、5。"""
    rep, encoded, _, _ = _plan_in_process(tmp_path, monkeypatch, active, "tubelet=pair", "--replan", "3", "--episodes", "21")
    assert len(encoded) == 2*21                                  # 每回合：目标、起点（--replan 3 = 一次走完，没有重规划）
    S = systems.get(); v = datasets.active().knobs
    for ep, tid, start in ((0, 180, 0), (1, 181, 0), (20, 180, 5)):
        row = rep["episodes"][ep]
        assert (row["trajectory_id"], row["start_tick"]) == (tid, start)
        _, frames34 = S.frames_with_prev(tid)                    # 下标 f = tick f−1：(tick start−1, tick start) = 下标 (start, start+1)
        assert np.array_equal(encoded[2*ep+1][0], frames34[start:start+2])
        full = S.variant_trajectory(tid, v)
        assert np.array_equal(encoded[2*ep+1][0][1], S.draw(full[start:start+1], tid, v, ticks=[start])[0])
        if start:
            assert np.array_equal(encoded[2*ep+1][0][0], S.draw(full[start-1:start], tid, v, ticks=[start-1])[0])


def test_plan_real_path_uses_the_clip_encoder(tmp_path, monkeypatch, active, real_clip):
    rep, encoded, _, _ = _plan_in_process(tmp_path, monkeypatch, active, "tubelet=pair", "--replan", "1", fake=False)
    enc, = real_clip.built
    assert rep["encoder"] == "vjepa2-vitg" and len(enc.calls) == len(encoded) == 4
    for (prev, cur), (frames, _) in zip(enc.calls, encoded, strict=True):       # 时间顺序（前，后），每次 1 个 tubelet
        pre = torch.from_numpy(frames).permute(0, 3, 1, 2).float()/255.
        assert prev.shape[0] == 1 and torch.equal(prev[0], pre[0]) and torch.equal(cur[0], pre[1])


def test_plan_on_dup_dataset_still_encodes_single_frames(tmp_path, monkeypatch, active):
    rep, encoded, _, _ = _plan_in_process(tmp_path, monkeypatch, active, "", "--replan", "1")
    assert rep["tubelet"] == "dup" and all(not pair and frames.shape == (1, 256, 256, 3) for frames, pair in encoded)
    _, frames = systems.get().frames(180)
    assert np.array_equal(encoded[1][0][0], frames[0])


# ———————————————————————————— 特征噪声下限 ————————————————————————————

def _floor_tool(monkeypatch):
    from po_jepa import vjepa
    tool = _load("ac_context_check")
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(vjepa, "preprocess_rgb", _unit_preprocess)
    project = tubelet_probe.fake_tubelet_encoder()
    seen = dict(batches=[], paired=[])
    class Fake:
        def encode_clip(self, prev, cur):
            seen["batches"].append(prev.shape[0]); seen.setdefault("first", (prev.clone(), cur.clone()))
            return project(torch.stack((prev, cur), 2))
    def build(root, ckpt, device, paired=False):
        seen["paired"].append(paired)
        return Fake()
    monkeypatch.setattr(tool, "_real_encoder", build)
    return tool, seen


def _floor_args(features, tmp_path, variant="tubelet=pair"):
    return ["floor", "--system", "pendulum", "--trajectories", "40", *(("--variant", variant) if variant else ()),
            "--features", str(features), "--upstream-root", str(tmp_path), "--checkpoint", str(tmp_path/"x.pt"),
            "--device", "cpu", "--count", "2"]


def test_floor_tool_reencodes_pair_cache_with_two_frames(caches, real, active, tmp_path, monkeypatch):
    tool, seen = _floor_tool(monkeypatch)
    active(datasets.parse("double_well:200"))
    out = tmp_path/"floor.json"
    tool.main([*_floor_args(caches/"pair", tmp_path), "--output", str(out)])
    r = json.loads(out.read_text())
    assert (r["kind"], r["dataset"], r["trajectories"], r["tubelet"]) == ("feature_noise_floor", PAIR40.tag, [36, 37], "pair")
    assert seen["paired"] == [True] and seen["batches"] == [33, 33, *[1]*33]*2      # 按 tubelet 计：缓存的批大小两次，再逐个一次
    s = r["summary"]
    assert s["rerun"] == dict(mean=0., max=0.)
    assert s["vs_cache"]["mean"] == pytest.approx(s["rounding"]["mean"], rel=1e-3) and 0 < s["rounding"]["mean"] < 1e-2
    assert s["adjacent"]["mean"] > 10*s["vs_cache"]["mean"]
    # 第一批的输入就是 34 帧的 (第 k 帧, 第 k+1 帧)
    _, frames34 = systems.get().frames_with_prev(36)
    pre = torch.from_numpy(frames34).permute(0, 3, 1, 2).float()/255.
    assert torch.equal(seen["first"][0], pre[:-1]) and torch.equal(seen["first"][1], pre[1:])


def test_floor_tool_checks_the_prev_frame_and_the_tubelet_record(caches, real, active, tmp_path, monkeypatch):
    tool, seen = _floor_tool(monkeypatch)
    active(datasets.parse("double_well:200"))
    original = systems.SecondOrderSystem.frames_with_prev
    def changed(index):
        def frames_with_prev(self, tid):
            states, frames = original(self, tid)
            frames = frames.copy(); frames[index] = 255-frames[index]
            return states, frames
        return frames_with_prev
    for index in (0, 4):             # t = −1 那一帧（只记在 tubelet.first_prev）和中间一帧
        monkeypatch.setattr(systems.SecondOrderSystem, "frames_with_prev", changed(index))
        with pytest.raises(SystemExit, match="哈希不符"):
            tool.main([*_floor_args(caches/"pair", tmp_path), "--output", str(tmp_path/f"bad{index}.json")])
        assert not (tmp_path/f"bad{index}.json").exists()
    monkeypatch.setattr(systems.SecondOrderSystem, "frames_with_prev", original)
    # manifest 的 tubelet 记录被拿掉：拒绝
    read = cache._read_complete_cache
    def stripped(path):
        raw, m, sha = read(path)
        m = json.loads(json.dumps(m)); m.pop("tubelet", None)
        return raw, m, sha
    monkeypatch.setattr(cache, "_read_complete_cache", stripped)
    with pytest.raises(SystemExit, match="tubelet 记录"):
        tool.main([*_floor_args(caches/"pair", tmp_path), "--output", str(tmp_path/"bad.json")])
    assert not (tmp_path/"bad.json").exists()


def test_floor_tool_on_dup_second_order_cache_is_single_frame(caches, real, active, tmp_path, monkeypatch):
    """不带 tubelet=pair 的二阶场景：照旧单帧复制重编码（_real_encoder 不带 paired），报告没有 tubelet 字段。"""
    sys.path.insert(0, str(ROOT/"tools"))
    import encode_v7
    from types import SimpleNamespace
    tool, seen = _floor_tool(monkeypatch)
    project = encode_v7._fake_encoder(torch)
    class Single:
        def encode_target(self, chunk, *, target_time_seconds):
            seen["batches"].append(chunk.shape[0])
            return SimpleNamespace(raw_z=project(chunk.float()))
    monkeypatch.setattr(tool, "_real_encoder", lambda root, ckpt, device: Single())
    active(datasets.parse("double_well:200"))
    out = tmp_path/"floor.json"
    tool.main([*_floor_args(caches/"dup", tmp_path, variant=""), "--output", str(out)])
    r = json.loads(out.read_text())
    assert r["dataset"] == "pendulum-40" and "tubelet" not in r and seen["batches"] == [33, 33, *[1]*33]*2
    assert r["summary"]["vs_cache"]["mean"] == pytest.approx(r["summary"]["rounding"]["mean"], rel=1e-3)
    # 两帧数据集读单帧复制的缓存（或反过来）：缓存身份就对不上
    for features, variant in ((caches/"dup", "tubelet=pair"), (caches/"pair", "")):
        with pytest.raises((SystemExit, ValueError, RuntimeError)):
            tool.main([*_floor_args(features, tmp_path, variant=variant), "--output", str(tmp_path/"x.json")])
        assert not (tmp_path/"x.json").exists()
