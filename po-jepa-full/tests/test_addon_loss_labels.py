"""S08a：training._loss_config 的单项消融标签 time_only、space_only，以及它们在 train_step 和检查点里的用法（07 §2.3）。"""
from dataclasses import asdict

import pytest
import torch

from po_jepa.runtime import training

from test_model import d12_record
from test_training import e06_case, e06_config

V7_MESSAGE = "condition must be full with two positive weights or no_consistency with two zero weights"
TIME_MESSAGE = "condition time_only requires a positive time weight and a zero space weight"
SPACE_MESSAGE = "condition space_only requires a zero time weight and a positive space weight"
WEIGHTS = {"both": (1e-3, 2e-3), "none": (0., 0.), "time": (1e-3, 0.), "space": (0., 2e-3)}
ACCEPT = {("full", "both"), ("no_consistency", "none"), ("time_only", "time"), ("space_only", "space")}
REJECT = {"full": V7_MESSAGE, "no_consistency": V7_MESSAGE, "time_only": TIME_MESSAGE, "space_only": SPACE_MESSAGE,
          "ablation": V7_MESSAGE, "": V7_MESSAGE}
LABELS = ("full", "no_consistency", "time_only", "space_only", "ablation", "")
SINGLE = {"time_only": (WEIGHTS["time"], TIME_MESSAGE), "space_only": (WEIGHTS["space"], SPACE_MESSAGE)}


def label_config(weights):
    time_weight, space_weight = weights
    return e06_config(time_weight=time_weight, space_weight=space_weight)


@pytest.mark.parametrize("weights", list(WEIGHTS))
@pytest.mark.parametrize("label", LABELS)
def test_loss_config_label_matrix(label, weights):
    """6 个标签 × 4 种权重：接受的返回原生快照，其余报对应的 ValueError（文字精确相等）。"""
    config = label_config(WEIGHTS[weights])
    if (label, weights) in ACCEPT:
        native = training._loss_config(config, label)
        assert native == {k: v if isinstance(v, str) else float(v) for k, v in asdict(config).items()}
        assert all(type(v) is float or type(v) is str for v in native.values())
    else:
        with pytest.raises(ValueError) as error:
            training._loss_config(config, label)
        assert str(error.value) == REJECT[label]


def test_negative_zero_counts_as_zero_weight():
    """-0.0 算零权重，和 v7 的 weights == (0., 0.) 判法一致。"""
    assert training._loss_config(e06_config(time_weight=1e-3, space_weight=-0.), "time_only")["space_weight"] == 0.
    assert training._loss_config(e06_config(time_weight=-0., space_weight=2e-3), "space_only")["time_weight"] == 0.


@pytest.mark.parametrize("label", ["time_only", "space_only"])
def test_train_step_single_term_label(monkeypatch, label):
    """单项标签照样算 5 次 Φ 和两个一致性项；零权重那项对 total 的贡献恰为 0；双正权重配单项标签报错。"""
    weights, message = SINGLE[label]
    model, batch = e06_case()
    optimizer = training.make_optimizer(model)
    queries = d12_record(monkeypatch, model, "prepare_query")
    config = label_config(weights)
    result = training.train_step(model, optimizer, batch, loss_config=config, completed_steps=0, condition=label)
    values = result.loss_values
    assert result.completed_steps == 1
    assert len(queries) == 5
    assert values["time"] > 0 and values["space"] > 0
    zero, live = ("weighted_space", "weighted_time") if label == "time_only" else ("weighted_time", "weighted_space")
    assert values[zero] == 0. and values[live] > 0
    assert values["total"] == values["jepa"]+values[live]
    with pytest.raises(ValueError, match="^"+message+"$"):
        training.train_step(model, optimizer, batch, loss_config=e06_config(), completed_steps=1, condition=label)


@pytest.mark.parametrize("label", ["time_only", "space_only"])
def test_checkpoint_round_trip_keeps_label(tmp_path, label):
    """检查点的 condition 存标签本身，loss_config 存单项快照；同标签读回逐项相同，换标签加对调的权重读报 mismatch。"""
    weights, _ = SINGLE[label]
    model, batch = e06_case()
    optimizer = training.make_optimizer(model)
    config = label_config(weights)
    training.train_step(model, optimizer, batch, loss_config=config, completed_steps=0, condition=label)
    path = tmp_path/"step-0001.pt"
    digest = training.save_training_checkpoint(path, model=model, optimizer=optimizer, loss_config=config,
        condition=label, completed_steps=1,
        sampling_state=dict(kind="explicit_prepared_batches", next_batch_index=1, query_generator_state=None),
        provenance=batch.provenance)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["condition"] == label
    assert payload["loss_config"] == training._loss_config(config, label)
    clone, _ = e06_case()
    clone_optimizer = training.make_optimizer(clone)
    progress = training.load_training_checkpoint(path, expected_sha256=digest, model=clone, optimizer=clone_optimizer,
        loss_config=config, condition=label, expected_provenance=batch.provenance)
    assert progress.completed_steps == 1
    expected, restored = model.state_dict(), clone.state_dict()
    assert list(expected) == list(restored)
    assert all(torch.equal(expected[name], restored[name]) for name in expected)
    other = "space_only" if label == "time_only" else "time_only"
    swapped = label_config(weights[::-1])
    fresh, _ = e06_case()
    fresh_optimizer = training.make_optimizer(fresh)
    with pytest.raises(ValueError, match="mismatch"):
        training.load_training_checkpoint(path, expected_sha256=digest, model=fresh, optimizer=fresh_optimizer,
            loss_config=swapped, condition=other, expected_provenance=batch.provenance)
