"""DROID 上的 dino_wm_small、jepa_wm_small（docs/design/09-droid-training.md §14，2026-11-26）。

合成数据根和缩小配置沿用 test_droid_train（同一 `root` 夹具、`small()`、`make_data`）。核对：参数匹配和 09 §14 的表、
构造（7 维动作、no-prop、种子）、训练步只用片段前 4 帧和前 3 个动作且不读状态、损失和仿真 `ac_training._wm_losses`
同一前缀时逐位相同、滚动前缀按任务种子取、验证和评测不绑定状态、任务图 `--conditions`。
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from po_jepa import ac_baseline
from po_jepa.droid import batches as bt
from po_jepa.droid import check as chk
from po_jepa.droid import conditions as cd
from po_jepa.droid import evaluate as ev
from po_jepa.droid import train as tr
from po_jepa.droid import wm
from po_jepa.droid.config import CONDITIONS
from po_jepa.losses import jepa_distance
from po_jepa.runtime import ac_training
from test_droid_train import SIGMA_Z, Small, make_data, root, same_tree, small  # noqa: F401（root 是夹具）

from test_droid_jobs import CONFIGS, DATA, RUNS, expected, graph_of  # noqa: E402（先导入它，tools/ 才在 sys.path 上）

import droid_jobs  # noqa: E402

WM = ("dino_wm_small", "jepa_wm_small")
TABLE = dict(dino_wm_small=(248_694, "mlp_hidden", 318, 149), jepa_wm_small=(248_508, "mlp_ratio", 2.9375, 129*4))
PO = 248_737


def _batch(data, config, step=0):
    dataset = bt.StepBatches("ac", data.data_root, data.manifests, data.train_set, data.episodes, seed=1703,
                             start=step, stop=step + 1, config=config)
    batches = bt.training_batches(dataset, data.control_spec, data.provenance, torch.device("cpu"), workers=0)
    _, batch = next(batches)
    batches.close()
    return batch


def _build(c, config, seed=1703):
    return cd.build(c, config=config, seed=seed, sigma_z=SIGMA_Z, device="cpu")


def _learned_constant(config):
    lc = dataclasses.replace(config, model=dataclasses.replace(config.model, ac_state="learned_constant"))
    return Small(**{f.name: getattr(lc, f.name) for f in dataclasses.fields(lc)})


# ---------------------------------------------------------------- 条件表、参数匹配、构造

def test_conditions_and_parameter_table():
    assert CONDITIONS[-2:] == WM and set(WM) <= set(cd.AC_CONDITIONS) and set(WM) <= set(ev.AC_CONDITIONS)
    assert not set(WM) & set(cd.STATE_AC) and not set(WM) & set(ev.STATE_AC_CONDITIONS)
    assert cd.LOSS_LABELS["dino_wm_small"] == cd.LOSS_LABELS["jepa_wm_small"] == "no_consistency"
    for c in WM:
        assert chk.expected_parameters(c, small()) == TABLE[c][0]


@pytest.mark.parametrize("condition", WM)
def test_matching_follows_the_ac_small_rule(condition):
    config = small()
    want, field, value, per_step = TABLE[condition]
    match = cd.match_wm(condition, config, PO)
    assert match["selected"][field] == value and match["parameters"] == want
    assert match["relative_difference"] < 0.001 and match["po_parameters"] == PO
    counts = [r["parameters"] for r in match["candidates"]]
    assert counts == [want - per_step, want, want + per_step]          # 候选两侧各一档（09 §14）
    for row in match["candidates"]:                                    # 其余结构字段和仿真版相同，只换动作维和匹配项
        spec = ac_baseline.ACSpec(**row["spec"])
        sim = ac_baseline.spec_for(condition)
        assert dataclasses.replace(spec, action_dim=2, **{field: getattr(sim, field)}) == sim
        assert spec.action_dim == 7 and spec.state_input == "learned_constant"


@pytest.mark.parametrize("condition", WM)
def test_build_is_seeded_no_prop_and_state_independent(condition):
    config = small()
    model, loss, info = _build(condition, config)
    assert isinstance(model, ac_baseline.ACPredictor) and not model.true_state
    assert model.spec.conditioning == ("feature" if condition == "dino_wm_small" else "adaln")
    assert model.ac_condition == condition and model.wm_seed == 1703
    assert info["parameters"] == TABLE[condition][0] == ac_baseline.parameter_count(model)
    assert info["ac"]["proprio"] == "none" and info["ac"]["init"] == "scratch"
    assert info["ac"]["source"] == ac_baseline.JEPA_WMS_SOURCE
    assert (loss.time_weight, loss.space_weight) == (0., 0.)
    again = _build(condition, config)[2]["initial_model_sha256"]
    other = _build(condition, config, seed=1704)
    assert again == info["initial_model_sha256"] != other[2]["initial_model_sha256"] and other[0].wm_seed == 1704
    lc = _build(condition, _learned_constant(config))[2]           # no-prop：ac_state 不影响构造
    assert lc["initial_model_sha256"] == info["initial_model_sha256"] and lc["parameters"] == info["parameters"]
    assert not cd.binds_state(condition, config) and cd.binds_state("vjepa2_ac_small", config)
    with pytest.raises(ValueError, match="no official weights"):
        cd.build(condition, config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu", ac_checkpoint="x.pt")
    with pytest.raises(ValueError, match="no consistency weights"):
        cd.build(condition, config=config, seed=1703, sigma_z=SIGMA_Z, device="cpu",
                 weights=dict(time_weight=1., space_weight=1.))


def test_optimizer_betas_and_peak():
    config = small()
    for c, betas in (("dino_wm_small", (0.9, 0.999)), ("jepa_wm_small", (0.9, 0.995))):
        model, _, _ = _build(c, config)
        opt = cd.make_optimizer(c, model)
        ref = ac_training.make_optimizer(model, c)
        assert [g["betas"] for g in opt.param_groups] == [betas, betas]
        assert [g["weight_decay"] for g in opt.param_groups] == [g["weight_decay"] for g in ref.param_groups] == [1e-7, 0.]
        assert [g["lr"] for g in opt.param_groups] == [5e-4, 5e-4]
        assert [[id(p) for p in g["params"]] for g in opt.param_groups] == [[id(p) for p in g["params"]] for g in ref.param_groups]


# ---------------------------------------------------------------- 损失、训练步

def test_rollout_prefix_uses_the_job_seed():
    assert all(wm.rollout_prefix(s, k) in (0, 1) for s in (1703, 1704) for k in range(50))
    a = [wm.rollout_prefix(1703, k) for k in range(64)]
    b = [wm.rollout_prefix(1704, k) for k in range(64)]
    assert a != b and 0 < sum(a) < 64
    with pytest.raises(ValueError):
        wm.rollout_prefix(-1, 0)


@pytest.mark.parametrize("condition", WM)
def test_losses_equal_the_simulation_formula(root, condition, monkeypatch):
    """同一前缀时，DROID 的损失和梯度与仿真 `ac_training._wm_losses` 逐位相同（同一模型、同一 4 帧片段）。"""
    config = small()
    batch = _batch(make_data(root, config), config)
    frames, actions = wm.clips(batch)
    assert frames.shape[1:] == (4, 256, 1408) and actions.shape[1:] == (3, 7)
    assert torch.equal(frames, batch.features[:, :4]) and torch.equal(actions, batch.actions[:, :3])
    for prefix in ((None,) if condition == "dino_wm_small" else (0, 1)):
        grads = []
        for how in ("droid", "sim"):
            model, _, _ = _build(condition, config)
            if how == "droid":
                loss, parts = wm.losses(condition, model, frames, actions, prefix)
            else:
                monkeypatch.setattr(ac_training, "rollout_prefix", lambda completed, p=prefix: p)
                loss, parts = ac_training._wm_losses(condition, model, frames, actions, 0)
            loss.backward()
            grads.append((loss.item(), parts, {n: p.grad.clone() for n, p in model.named_parameters()}))
        assert grads[0][0] == grads[1][0] and grads[0][1] == grads[1][1] and same_tree(grads[0][2], grads[1][2])
    model, _, _ = _build(condition, config)
    with pytest.raises(ValueError):
        wm.losses(condition, model, batch.features, batch.actions, 0)                      # 9 帧不收


@pytest.mark.parametrize("condition", WM)
def test_train_step_reads_four_frames_and_no_states(root, condition):
    """训练步只看片段前 4 帧、前 3 个动作，不读真实状态：把后面的帧、动作和全部状态换掉，结果逐位不变；
    改第 3 帧（第 4 帧）会变。走 `conditions.train_step`（ac_state: true_state 的配置）。"""
    config = small()
    assert config.model.ac_state == "true_state"
    batch = _batch(make_data(root, config), config)
    noisy = dataclasses.replace(batch, features=torch.cat([batch.features[:, :4], torch.randn_like(batch.features[:, 4:])], 1),
                                actions=torch.cat([batch.actions[:, :3], torch.randn_like(batch.actions[:, 3:])], 1),
                                states=torch.randn_like(batch.states))
    changed = dataclasses.replace(batch, features=batch.features.clone())
    changed.features[:, 3] = torch.randn_like(changed.features[:, 3])   # 常数平移会被 LN 消掉，换成随机
    out = {}
    for name, b in (("base", batch), ("noisy", noisy), ("changed", changed)):
        model, loss, _ = _build(condition, config)
        result, lr = cd.train_step(condition, model, cd.make_optimizer(condition, model), b, loss, 0,
                                   ac_state=config.model.ac_state)
        out[name] = (result.loss_values, lr, {k: v.clone() for k, v in model.state_dict().items()})
    assert out["base"][0] == out["noisy"][0] and same_tree(out["base"][2], out["noisy"][2])
    assert out["base"][0]["total"] != out["changed"][0]["total"]
    assert out["base"][1] == ac_training._lr(condition, loss, 0)
    if condition == "jepa_wm_small":
        assert out["base"][0]["ac_rollout_prefix"] == wm.rollout_prefix(1703, 0)
        assert set(out["base"][0]) >= {"ac_teacher_forcing", "ac_rollout", "total", "jepa"}
    else:
        assert "ac_rollout" not in out["base"][0]


@pytest.mark.parametrize("condition", WM)
def test_validation_runs_without_state_binding(root, condition):
    """验证（09 §4.4）对 no-prop 条件用模型本身前向（不绑定起点状态），ac_state: true_state 也一样。"""
    config = small()
    data = make_data(root, config)
    model, loss, _ = _build(condition, config)
    got = tr.validate(model, condition=condition, data=data, config=config, loss_config=loss, device="cpu")
    batch, _ = bt.make_eval_batch(data.val_windows, data.reader, data.val_episodes, data.control_spec,
                                  data.provenance, "cpu")
    model.eval()
    with torch.no_grad():
        want = jepa_distance(model(batch.raw_z, batch.controls, batch.duration, batch.transfer.source), batch.raw_target,
                             config=loss).per_sample.double()
    assert got["values"] == want.tolist() and all(torch.isfinite(want))


# ---------------------------------------------------------------- 任务图 --conditions

def test_jobs_filter_by_condition():
    full = graph_of("full")
    assert "conditions" not in full
    got = graph_of("full", conditions=droid_jobs.parse_conditions("dino_wm_small,jepa_wm_small"))
    ids = {j["id"] for j in got["jobs"]}
    want = {k for k in expected("full", 20, [1703, 1704, 1705])
            if not k.startswith(("train/", "eval/", "calibrate/lambda-")) and k != "summary"}
    want |= {f"{s}/{c}/seed{seed}" for s in ("train", "eval") for c in WM for seed in (1703, 1704, 1705)}
    assert ids == want and got["conditions"] == list(WM)
    by_id = {j["id"]: j for j in full["jobs"]}
    for j in got["jobs"]:                                   # 和完整任务图里的同一任务完全相同（命令、完成标志、依赖）
        assert j == by_id[j["id"]]
    lam = graph_of("full", conditions=("po_jepa_full", "identity"))
    lam_ids = {j["id"] for j in lam["jobs"]}
    assert {"calibrate/lambda-po_jepa_full", "eval/identity", "train/po_jepa_full/seed1703"} <= lam_ids
    assert not any(i.startswith("calibrate/lambda-") and i != "calibrate/lambda-po_jepa_full" for i in lam_ids)
    assert "eval/spectral-gap" not in lam_ids and "summary" not in lam_ids and "eval/vjepa2_ac_zeroshot" not in lam_ids
    assert "只排条件 po_jepa_full,identity" in droid_jobs.to_list(lam)
    for bad in ("", "dino_wm_small,", "dino_wm_small,dino_wm_small", "nope"):
        with pytest.raises(ValueError):
            droid_jobs.parse_conditions(bad)


def test_jobs_cli_conditions(tmp_path):
    out = tmp_path/"jobs.json"
    assert droid_jobs.main(["--config", str(CONFIGS/"pilot.yaml"), "--data-root", DATA, "--run-root", RUNS,
                            "--output", str(out), "--conditions", "jepa_wm_small"]) == 0
    graph = json.loads(out.read_text(encoding="utf-8"))
    trains = [j["id"] for j in graph["jobs"] if j["stage"] == "train"]
    assert trains == ["train/jepa_wm_small/seed1703"] and graph["conditions"] == ["jepa_wm_small"]
    assert next(j for j in graph["jobs"] if j["id"] == trains[0])["priority"] == 2


# ---------------------------------------------------------------- DWM1-P2：共享模型源码进训练来源

def test_shared_model_sources_are_in_the_training_provenance(root):
    """只改共享核心（例如 wm_baselines.py 的前向，参数、规格、初始化都不变）时训练来源必须变：这些文件都登记、摘要是文件的
    实际 SHA；把其中一项换成别的摘要，`training._provenance` 的实际 SHA 核对就拒绝（续训同样经这个核对）。"""
    from po_jepa.runtime import cache, training
    _, manifests, _, _ = root
    p = bt.training_provenance(small(), manifests, "f"*64)
    for name in ("src/po_jepa/wm_baselines.py", "src/po_jepa/third_party/jepa_wms.py", "src/po_jepa/ac_baseline.py",
                 "src/po_jepa/runtime/ac_training.py", "src/po_jepa/third_party/vjepa2_ac.py"):
        assert p["source_files_sha256"][name] == cache.file_sha(cache.ROOT/name)
        changed = dict(p, source_files_sha256=dict(p["source_files_sha256"], **{name: "0"*64}))
        with pytest.raises(ValueError, match="source SHA mismatch"):
            training._provenance(changed)
