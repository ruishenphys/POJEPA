"""C7（差距清单 §3）：tools/ac_context_check.py 的 AC 上下文截断核查和特征噪声下限。

假特征同 S25f：tools/encode_v7.py --fake-encoder（子进程），读取时用 real 夹具当作真实编码器的缓存；数据集 double_well:40（测试段 36–39）。
"""
from dataclasses import fields
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from po_jepa import ac_baseline, datasets
from po_jepa.geometry import regular_grid
from po_jepa.losses import LossConfig
from po_jepa.runtime import ac_training, checks, comparison, evaluation, v7jobs
from test_model import d12_controls
from test_s25c_long_horizon import _encode, active, real  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
BASE = datasets.parse("double_well:40")
TOOL_FILE = "tools/ac_context_check.py"


def _tool():
    spec = importlib.util.spec_from_file_location("ac_context_check_tool", ROOT/TOOL_FILE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _loss():
    c = checks.load_protocol()["loss"]
    return LossConfig(**{f.name: c[f.name] for f in fields(LossConfig)})


def _model(seed=0):
    torch.manual_seed(seed)
    return ac_baseline.ACPredictor(ac_baseline.SMALL).eval()


def _case(T, seed=3):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(1, 256, 1408, generator=g)
    values = (torch.rand(4, 2, generator=g)-.5).tolist()
    # 4 段分段常数控制，段界不在整 tick 上（动作的 tick 平均要按窗口取对）
    bounds = [0., T*.3, T*.55, T*.8, float(T)]
    controls = d12_controls([[(bounds[i], bounds[i+1], tuple(values[i])) for i in range(4)]])
    return raw, controls, torch.tensor([float(T)], dtype=torch.float64), regular_grid(16, 16)


# ---------------------------------------------------------------- 滑动窗口上下文

@pytest.mark.parametrize("T", [1, 3, 12])
def test_full_context_is_bitwise_the_direct_prediction(T):
    tool, model = _tool(), _model()
    raw, controls, d, grid = _case(T)
    with torch.no_grad():
        assert torch.equal(tool.windowed_rollout(model, raw, controls, d, grid, None), model(raw, controls, d, grid))
        for w in (T, T+1, 16):                                  # 窗口不小于 T：不截断，同一串运算
            assert torch.equal(tool.windowed_rollout(model, raw, controls, d, grid, w), model(raw, controls, d, grid))


def test_window_one_matches_unit_rollout():
    """W = 1 和 v7.1 的逐步滚动只差每步对预测帧多做的一次 LN（unit_rollout 每步走 forward）。"""
    tool, model = _tool(), _model(1)
    raw, controls, d, grid = _case(12)
    with torch.no_grad():
        a = tool.windowed_rollout(model, raw, controls, d, grid, 1)
        b = evaluation.unit_rollout(model, raw, controls, d, grid)
        direct = model(raw, controls, d, grid)
    # 实测（CPU，12 步）：最大差 1.2e-4、平均差 1.2e-5；每步补做 LN 后最大差降到 1.5e-5（剩下的是切分动作段的舍入）
    assert (a-b).abs().max() < 5e-4 and (a-b).abs().mean() < 5e-5
    assert (a-direct).abs().max() > 1e-3                        # 未训练的模型上截断确实改了结果


def test_window_truncates_context_and_actions(monkeypatch):
    """T = 12、W = 8：第 k 步的上下文是第 max(0,k−7)…k 帧（长度 1…8 然后一直 8），动作是同样那几帧的 tick 动作，帧是滚动出来的那几帧。"""
    tool, model = _tool(), _model(2)
    raw, controls, d, grid = _case(12)
    actions = ac_baseline.tick_actions(controls, 12)
    calls, original = [], model.predict_frames
    def spy(context, acts, height, width, states=None):
        out = original(context, acts, height, width, states)
        calls.append((context.clone(), acts.clone(), out[:, -1].clone()))
        return out
    monkeypatch.setattr(model, "predict_frames", spy)
    with torch.no_grad():
        tool.windowed_rollout(model, raw, controls, d, grid, 8)
    assert [c[0].shape[1] for c in calls] == [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8]
    frames = [torch.nn.functional.layer_norm(raw, (1408,))]+[c[2] for c in calls]
    for k, (context, acts, _) in enumerate(calls):
        lo = max(0, k+1-8)
        assert torch.equal(acts, actions[:, lo:k+1])
        assert torch.equal(context, torch.stack(frames[lo:k+1], 1))


def test_windowed_rollout_refusals():
    tool, model = _tool(), _model()
    raw, controls, d, grid = _case(2)
    with pytest.raises(ValueError, match="window"):
        tool.windowed_rollout(model, raw, controls, d, grid, 0)
    half = d12_controls([[(0., 1.5, (.1, .2))]])
    with pytest.raises(ValueError, match="正整数时长"):
        tool.windowed_rollout(model, raw, half, torch.tensor([1.5], dtype=torch.float64), grid, 8)
    with pytest.raises(ValueError, match="V-JEPA 2-AC"):
        tool.windowed_rollout(torch.nn.Identity(), raw, controls, d, grid, 8)
    with pytest.raises(ValueError, match="一个窗口"):
        tool.windowed_rollout(model, raw.expand(2, -1, -1), controls, d, grid, 8)


def test_context_values_and_summary():
    tool, model, loss = _tool(), _model(4), _loss()
    rows = []
    for T in (4, 12):
        raw, controls, d, grid = _case(T, seed=T)
        target = torch.randn(1, 256, 1408, generator=torch.Generator().manual_seed(100+T))
        values, gap = tool.context_values(model, raw, target, controls, d, grid, loss)
        assert set(values) == set(tool.method_names()) and gap == 0.
        assert values["full_context"] == values["direct"]
        for w in tool.WINDOWS:
            if T <= w:
                assert values[f"window{w}"] == values["direct"]
        with torch.no_grad():
            w8 = tool.windowed_rollout(model, raw, controls, d, grid, 8)
        assert values["window8"] == pytest.approx(float(comparison.jepa_distance(w8, target, config=loss).per_sample[0]), abs=0)
        rows.append(dict(duration_cycles=float(T), values=values, max_abs_full_minus_direct=gap))
    rows.append(dict(duration_cycles=12., values={**rows[-1]["values"], "window8": rows[-1]["values"]["direct"]-1.},
                     max_abs_full_minus_direct=0.))
    s = tool.summarize_context(rows)
    assert set(s) == {"4", "12"} and s["4"]["points"] == 1 and s["12"]["points"] == 2
    assert s["4"]["window8_minus_direct"] == dict(mean=0., better_points=0, worse_points=0)
    diff = [rows[1]["values"]["window8"]-rows[1]["values"]["direct"], -1.]
    assert s["12"]["window8_minus_direct"]["mean"] == pytest.approx(np.mean(diff))
    assert s["12"]["window8_minus_direct"]["better_points"] == sum(x < 0 for x in diff)
    assert s["12"]["window8_minus_direct"]["worse_points"] == sum(x > 0 for x in diff)
    assert diff[0] != 0                                          # 未训练模型上截断改了结果，配对差有一正一负或两负
    assert s["12"]["means"]["direct"] == pytest.approx(rows[1]["values"]["direct"])


def test_tool_is_not_a_training_consumer_source(monkeypatch):
    """加这个工具不改任何预登记：它不在训练的源码身份里（runtime/data、comparison 的 consumer_sources）。"""
    monkeypatch.setenv("PO_JEPA_PROTOCOL", "v7")
    from po_jepa.runtime import data
    assert TOOL_FILE not in data.consumer_sources() and TOOL_FILE not in comparison.consumer_sources()
    assert not any("ac_context" in name for name in comparison.consumer_sources())


# ---------------------------------------------------------------- 端到端：context

@pytest.fixture(scope="module")
def base_cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("c7")
    r = _encode(root/"base", "", splits="test")
    assert r.returncode == 0, r.stderr
    return root/"base"


def _job(root, condition, dataset=BASE.tag):
    job = root/"job"/"run"/"job"; job.mkdir(parents=True)
    (job/"job.json").write_text(json.dumps(dict(seed=1703, updates=10000, dataset=dataset, condition=condition)))
    return root/"job"


def test_context_tool(base_cache, real, active, tmp_path, monkeypatch):
    tool = _tool()
    active(datasets.parse("double_well:200"))                   # 先停在别的数据集：工具必须按 job.json 设
    monkeypatch.setattr(os, "environ", dict(os.environ))
    seen, model = {}, _model(5)
    def load(condition, training_run, device):
        seen.update(condition=condition, training_run=Path(training_run), dataset=datasets.active().tag)
        return model, _loss(), dict(step=40, checkpoint_sha256="c"*64), dict(fake=True)
    monkeypatch.setattr(v7jobs, "load_for_evaluation", load)
    job = _job(tmp_path, "vjepa2_ac_small")
    out = tmp_path/"out"/"ac.json"
    args = ["context", "--job", str(job), "--features", str(base_cache), "--device", "cpu", "--max-trajectories", "1"]
    tool.main([*args, "--output", str(out)])
    assert seen["condition"] == "vjepa2_ac_small" and seen["dataset"] == BASE.tag and datasets.active() == BASE
    r = json.loads(out.read_text())
    assert (r["kind"], r["dataset"], r["condition"], r["training_context_frames"], r["grid"]) == (
        "ac_context_check", BASE.tag, "vjepa2_ac_small", 8, 16)
    starts = datasets.test_start_ticks(36)
    assert [(x["trajectory_id"], x["start_tick"], x["duration_cycles"]) for x in r["rows"]] == [
        (36, s, t) for s in starts for t in evaluation.TIMES]
    assert r["counts"] == dict(trajectories=1, points=len(starts)*len(evaluation.TIMES), all_test_trajectories=False)
    assert all(x["max_abs_full_minus_direct"] == 0. for x in r["rows"]) and set(r["by_duration"]) == {"1", "2", "3", "4", "8", "12"}
    # 一个点独立重算：T = 12 截断到 8 帧
    q = evaluation.EvaluationQuery(36, starts[0], 12., 16, "point")
    w = evaluation.load_test_window(base_cache, q, torch.device("cpu"))
    with torch.no_grad():
        pred = tool.windowed_rollout(model, w.inputs.raw_z, w.inputs.controls, w.inputs.duration, w.inputs.grid, 8)
    want = float(comparison.jepa_distance(pred, w.target.raw_z, config=_loss()).per_sample[0])
    row = next(x for x in r["rows"] if (x["start_tick"], x["duration_cycles"]) == (starts[0], 12.))
    assert row["values"]["window8"] == pytest.approx(want, rel=1e-6)
    assert r["by_duration"]["12"]["means"]["window8"] == pytest.approx(
        np.mean([x["values"]["window8"] for x in r["rows"] if x["duration_cycles"] == 12.]))
    with pytest.raises(SystemExit, match="已存在"):
        tool.main([*args, "--output", str(out)])
    with pytest.raises(SystemExit, match="只核 AC 条件"):
        tool.main(["context", "--job", str(_job(tmp_path/"po", "po_jepa_full")), "--features", str(base_cache),
                   "--device", "cpu", "--output", str(tmp_path/"po.json")])
    assert not (tmp_path/"po.json").exists()


# ---------------------------------------------------------------- 特征噪声下限

def test_floor_values_definitions():
    tool, loss = _tool(), _loss()
    g = np.random.default_rng(0)
    first = g.normal(size=(5, 4, 1408))
    stored = first.astype(np.float16)
    again = first.copy(); single = first+1e-3*g.normal(size=first.shape)
    v = tool.floor_values(first, again, single, stored, np.float16, loss)
    assert set(v) == {"rerun", "batch", "rounding", "vs_cache", "adjacent"}
    assert v["rerun"] == dict(mean=0., max=0.)
    # 缓存就是新编码按 float16 存（新编码先转 float32 再舍入，和 float64 直接舍入只差个别最低位）
    assert 0 < v["rounding"]["mean"] < 1e-2 and v["vs_cache"]["mean"] == pytest.approx(v["rounding"]["mean"], rel=1e-3)
    assert v["batch"]["mean"] > v["rounding"]["mean"]
    def d(x, y):
        return comparison.jepa_distance(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)),
                                        config=loss).per_sample.double().numpy()
    assert v["adjacent"]["mean"] == pytest.approx(float(d(stored[:-1], stored[1:]).mean()))
    assert v["adjacent"]["mean"] > 100*v["rounding"]["mean"]


def _fake_preprocess(rgb, *, spec, generator):
    """代替 vjepa.preprocess_rgb：和 encode_v7 的 --fake-encoder 路径同一个输入（[1,3,256,256]，0–1）。"""
    return rgb.permute(0, 3, 1, 2).float()/255.


def test_floor_tool(base_cache, real, active, tmp_path, monkeypatch):
    """假编码器重编码假缓存：同一状态编码两次 = float16 存储舍入，重跑为 0；画面哈希不符时拒绝。"""
    import sys
    sys.path.insert(0, str(ROOT/"tools"))
    import encode_v7
    from po_jepa import vjepa
    tool = _tool()
    active(datasets.parse("double_well:200"))
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(vjepa, "preprocess_rgb", _fake_preprocess)
    project = encode_v7._fake_encoder(torch)
    batches = []
    class Fake:
        def encode_target(self, chunk, *, target_time_seconds):
            batches.append(chunk.shape[0])
            return SimpleNamespace(raw_z=project(chunk.float()))
    monkeypatch.setattr(tool, "_real_encoder", lambda root, ckpt, device: Fake())
    out = tmp_path/"floor.json"
    args = ["floor", "--system", "double_well", "--trajectories", "40", "--features", str(base_cache),
            "--upstream-root", str(tmp_path), "--checkpoint", str(tmp_path/"x.pt"), "--device", "cpu", "--count", "2"]
    tool.main([*args, "--output", str(out)])
    r = json.loads(out.read_text())
    assert (r["kind"], r["dataset"], r["trajectories"]) == ("feature_noise_floor", BASE.tag, [36, 37]) and datasets.active() == BASE
    assert batches == [33, 33, *[1]*33]*2                                   # 缓存的批大小两次，再逐帧一次
    s = r["summary"]
    assert s["rerun"] == dict(mean=0., max=0.)
    assert s["vs_cache"]["mean"] == pytest.approx(s["rounding"]["mean"], rel=1e-3) and 0 < s["rounding"]["mean"] < 1e-2
    assert s["adjacent"]["mean"] > 10*s["vs_cache"]["mean"]
    assert all(x["stored_dtype"] == "float16" and x["encode_batch"] == 33 for x in r["rows"])
    with pytest.raises(SystemExit, match="已存在"):
        tool.main([*args, "--output", str(out)])
    from po_jepa import systems
    original = systems.System.frames if hasattr(systems, "System") else None
    def shifted(self, tid):
        states, frames = original(self, tid)
        frames = frames.copy(); frames[3] = 255-frames[3]
        return states, frames
    if original is not None:
        monkeypatch.setattr(systems.System, "frames", shifted)
        with pytest.raises(SystemExit, match="哈希不符"):
            tool.main([*args, "--output", str(tmp_path/"bad.json")])
        assert not (tmp_path/"bad.json").exists()
