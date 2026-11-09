"""S01：DROID 三层配置（docs/design/01-droid-config.md §1–§3、§6；验收 14 §3 S01）。"""

import copy
import dataclasses
import hashlib
import os
from pathlib import Path

import pytest
import yaml

from po_jepa.droid import config as dc
from po_jepa.losses import LossConfig
from po_jepa.runtime import cache, checks

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/"configs"/"droid"
BASE = yaml.safe_load((CONFIGS/"base.yaml").read_text(encoding="utf-8"))
PILOT = dict(name="pilot", data=dict(train_episodes=500), training=dict(updates=2000, seeds=[1703]))


def _write(tmp_path: Path, overlay, *, base_edit=None, base=None) -> Path:
    """在 tmp_path 写一份 base.yaml（可改）和 overlay，返回 overlay 的路径。"""
    raw = copy.deepcopy(BASE if base is None else base)
    if base_edit is not None:
        base_edit(raw)
    (tmp_path/"base.yaml").write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    path = tmp_path/"run.yaml"
    path.write_text(overlay if isinstance(overlay, str) else yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    return path


def _load(tmp_path, overlay=PILOT, **kw):
    return dc.load_config(_write(tmp_path, overlay, **kw))


def _merged(overlay):
    return checks._merge_v4(BASE, overlay)


# ---------------------------------------------------------------- 合并、SHA、推导量

@pytest.mark.parametrize("name,overlay", [("pilot", PILOT),
                                          ("full", dict(name="full", data=dict(train_episodes=20000),
                                                        training=dict(updates=30000, seeds=[1703, 1704, 1705])))])
def test_shipped_overlays_merge_onto_base(name, overlay):
    c = dc.load_config(CONFIGS/f"{name}.yaml")
    assert yaml.safe_load((CONFIGS/f"{name}.yaml").read_text(encoding="utf-8")) == overlay
    merged = _merged(overlay)
    assert c.config_sha256 == hashlib.sha256(cache._json_bytes(merged)).hexdigest()
    assert c.name == name and c.schema == "droid_config_v1"
    assert c.data.train_episodes == overlay["data"]["train_episodes"]
    assert c.training.updates == overlay["training"]["updates"]
    assert c.training.seeds == tuple(overlay["training"]["seeds"])
    # 其余键全部来自 base：逐键和合并结果比（tuple 对 list、float 对 int 按值比）
    def same(obj, raw):
        if dataclasses.is_dataclass(obj):
            assert sorted(f.name for f in dataclasses.fields(obj)) == sorted(raw)
            for key, value in raw.items():
                same(getattr(obj, key), value)
        elif isinstance(obj, tuple):
            assert len(obj) == len(raw)
            for a, b in zip(obj, raw):
                same(a, b)
        else:
            assert type(obj) is (float if isinstance(raw, float) else type(raw)) and obj == raw
    same(dataclasses.replace(c), dict(merged, config_sha256=c.config_sha256))
    assert c.evaluation.cem == dc.CemConfig(iterations=4, samples=256, elites=24, init_std=0.5, clip=1.5)
    assert c.resources.calibrate == dc.StageResources(gpus=1, cpus=8, mem_gb=64, time_limit_minutes=240)


@pytest.mark.parametrize("name,expected", [("pilot", (100, 200, 100, 280, 5)), ("full", (1500, 3000, 1500, 4200, 75))])
def test_derived_quantities_match_table(name, expected):
    c = dc.load_config(CONFIGS/f"{name}.yaml")
    got = (c.lr_warmup_updates, c.consistency_warmup_updates, c.validation_interval,
           c.early_check_updates, c.consistency_interval)
    assert got == expected and all(type(x) is int for x in got)
    u = c.training.updates
    assert got == (round(0.05*u), round(0.10*u), u//20, round(0.10*u)+round(0.04*u), u//400)
    # 早检时正好有 17 次一致性占比测量（一致性预热结束那次算第 1 次）
    assert (c.early_check_updates - c.consistency_warmup_updates)//c.consistency_interval + 1 == 17


def test_sha_is_stable_and_independent_of_key_order(tmp_path):
    a = _load(tmp_path)
    assert _load(tmp_path).config_sha256 == a.config_sha256
    def reverse(d):
        return {k: reverse(v) if isinstance(v, dict) else v for k, v in reversed(list(d.items()))}
    other = tmp_path/"other"
    other.mkdir()
    b = dc.load_config(_write(other, reverse(PILOT), base=reverse(BASE)))
    assert b.config_sha256 == a.config_sha256 and b == a
    assert dc.load_config(CONFIGS/"full.yaml").config_sha256 != a.config_sha256


def test_allowed_overlay_keys_change_the_sha(tmp_path):
    base_sha = _load(tmp_path).config_sha256
    for i, extra in enumerate((dict(model=dict(spectral_backend="cpu_threads")),
                               dict(resources=dict(train=dict(time_limit_minutes=2000))),
                               dict(resources=dict(encode=dict(gpus=2, cpus=16))))):
        sub = tmp_path/str(i)
        sub.mkdir()
        c = dc.load_config(_write(sub, checks._merge_v4(PILOT, extra)))
        assert c.config_sha256 != base_sha
    assert c.resources.encode == dc.StageResources(gpus=2, cpus=16, mem_gb=64, time_limit_minutes=600)


def test_switches_change_only_in_a_base_copy(tmp_path):
    c = _load(tmp_path, base_edit=lambda b: b["action"].update(source="commanded"))
    assert c.action.source == "commanded" and c.config_sha256 != dc.load_config(CONFIGS/"pilot.yaml").config_sha256
    for i, extra in enumerate((dict(action=dict(source="commanded")), dict(model=dict(ac_state="learned_constant")),
                               dict(data=dict(use_failures=False)))):
        sub = tmp_path/str(i)
        sub.mkdir()
        with pytest.raises(ValueError, match="may not be overridden"):
            dc.load_config(_write(sub, checks._merge_v4(PILOT, extra)))


def test_config_is_frozen():
    c = dc.load_config(CONFIGS/"pilot.yaml")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.training.updates = 10
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.name = "x"


# ---------------------------------------------------------------- overlay 和键的报错

@pytest.mark.parametrize("extra,message", [
    (dict(training=dict(batch=32)), "'training.batch' may not be overridden"),
    (dict(schema="droid_config_v2"), "'schema' may not be overridden"),
    (dict(data=5), "'data' may not be overridden"),
    (dict(data=dict(foo=1)), "'data.foo' is not in base.yaml"),
    (dict(extra_section=dict(a=1)), "'extra_section' is not in base.yaml"),
    (dict(resources=dict(train=dict(foo=1))), "'resources.train.foo' is not in base.yaml"),
    (dict(resources=dict(gpu_farm=dict(gpus=1))), "'resources.gpu_farm' is not in base.yaml"),
])
def test_overlay_key_errors(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        _load(tmp_path, checks._merge_v4(PILOT, extra))


def test_overlay_must_be_a_mapping(tmp_path):
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        _load(tmp_path, "- name\n- pilot\n")


def test_missing_and_unknown_keys_after_merge(tmp_path):
    with pytest.raises(ValueError, match=r"missing keys: data\.camera"):
        _load(tmp_path, base_edit=lambda b: b["data"].pop("camera"))
    with pytest.raises(ValueError, match=r"missing keys: evaluation\.cem\.elites"):
        _load(tmp_path, base_edit=lambda b: b["evaluation"]["cem"].pop("elites"))
    with pytest.raises(ValueError, match=r"missing keys: monitoring"):
        _load(tmp_path, base_edit=lambda b: b.pop("monitoring"))
    with pytest.raises(ValueError, match=r"unknown keys: action\.offset"):
        _load(tmp_path, base_edit=lambda b: b["action"].update(offset=0.0))


def test_name_still_base_or_not_a_directory_name(tmp_path):
    with pytest.raises(ValueError, match="name is still 'base'"):
        _load(tmp_path, dict(data=dict(train_episodes=500)))
    with pytest.raises(ValueError, match="name is still 'base'"):
        dc.load_config(CONFIGS/"base.yaml")
    with pytest.raises(ValueError, match="name is still 'base'"):
        _load(tmp_path, "")
    with pytest.raises(ValueError, match="plain directory name"):
        _load(tmp_path, dict(PILOT, name="../pilot"))


@pytest.mark.parametrize("edit,message", [
    (lambda b: b["data"].update(shard_size=True), r"data\.shard_size must be an integer"),
    (lambda b: b["data"].update(shard_size=1000.0), r"data\.shard_size must be an integer"),
    (lambda b: b["data"].update(use_failures=1), r"data\.use_failures must be true or false"),
    (lambda b: b["model"].update(sigma_x="0.125"), r"model\.sigma_x must be a finite real number"),
    (lambda b: b["model"].update(sigma_x=float("nan")), r"model\.sigma_x must be a finite real number"),
    (lambda b: b["model"].update(rho=True), r"model\.rho must be a finite real number"),
    (lambda b: b["data"].update(camera=" "), r"data\.camera must be a non-empty string"),
    (lambda b: b["data"].update(split_salt=""), r"data\.split_salt must be a non-empty string"),
    (lambda b: b["data"].update(window_salt=""), r"data\.window_salt must be a non-empty string"),
    (lambda b: b.update(time=5), r"config section 'time' must be a mapping"),
])
def test_type_errors(tmp_path, edit, message):
    with pytest.raises(TypeError, match=message):
        _load(tmp_path, base_edit=edit)


@pytest.mark.parametrize("training,message", [(dict(updates=2000, seeds=1703), r"training\.seeds must be a list"),
                                              (dict(updates=2000, seeds=[1703, "1704"]), r"training\.seeds\[1\] must be an integer"),
                                              (dict(updates=2000.0, seeds=[1703]), r"training\.updates must be an integer")])
def test_overlay_type_errors(tmp_path, training, message):
    with pytest.raises(TypeError, match=message):
        _load(tmp_path, dict(PILOT, training=training))


def test_overlay_train_episodes_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match=r"data\.train_episodes must be a positive integer"):
        _load(tmp_path, dict(PILOT, data=dict(train_episodes=0)))


def test_integer_valued_float_is_accepted_as_float(tmp_path):
    c = _load(tmp_path, base_edit=lambda b: b["model"].update(rho=1))
    assert type(c.model.rho) is float and c.model.rho == 1.0


# ---------------------------------------------------------------- 01 §3 的每条校验

@pytest.mark.parametrize("edit,message", [
    (lambda b: b.update(schema="droid_config_v0"), "schema must be"),
    (lambda b: b["data"].update(ood_quota=[250, 250]), "ood_quota must match"),
    (lambda b: b["data"].update(ood_quota=[167, 167, 167]), "ood_quota must match"),
    (lambda b: b["data"].update(id_test_scenes=0), r"data\.id_test_scenes must be a positive integer"),
    (lambda b: b["data"].update(val_scenes=-1), r"data\.val_scenes must be a positive integer"),
    (lambda b: b["data"].update(episodes_per_scene=0), r"data\.episodes_per_scene must be a positive integer"),
    (lambda b: b["data"].update(shard_size=0), r"data\.shard_size must be a positive integer"),
    (lambda b: b["training"].update(durations=[1, 2, 4]), r"training\.durations must be \[1, 2, 4, 8\]"),
    (lambda b: b["training"].update(durations=[1, 2, 4, 16]), r"training\.durations must be \[1, 2, 4, 8\]"),
    (lambda b: b["training"]["conditions"].reverse(), r"training\.conditions must be exactly"),
    (lambda b: b["training"]["conditions"].pop(), r"training\.conditions must be exactly"),
    (lambda b: b["action"].update(source="velocity"), r"action\.source must be one of"),
    (lambda b: b["model"].update(ac_state="none"), r"model\.ac_state must be one of"),
    (lambda b: b["model"].update(spectral_backend="gpu"), r"model\.spectral_backend must be one of"),
    (lambda b: b["evaluation"].update(frame_rate_single=[5, 10, 21]), "frame_rate_single and frame_rate_rollout"),
    (lambda b: b["evaluation"].update(frame_rate_rollout=[10, 12]), "frame_rate_single and frame_rate_rollout"),
    (lambda b: b["evaluation"].update(frame_rate_single=[0, 10, 20]), "frame_rate_single and frame_rate_rollout"),
    (lambda b: b["encode"].update(frame_rate_stride_frames=0), "frame_rate_single and frame_rate_rollout"),
    (lambda b: b["evaluation"].update(frame_rate_total_frames=50), "frame_rate_total_frames must be a multiple"),
])
def test_base_validation_rules(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, base_edit=edit)


@pytest.mark.parametrize("seeds,message", [([], "non-empty and distinct"), ([1703, 1703], "non-empty and distinct")])
def test_seed_rules(tmp_path, seeds, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, dict(PILOT, training=dict(updates=2000, seeds=seeds)))


@pytest.mark.parametrize("updates,message", [(0, "positive integer"), (-400, "positive integer"),
                                             (2010, "divisible by 20"), (2020, "divisible by 400")])
def test_updates_must_divide(tmp_path, updates, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, dict(PILOT, training=dict(updates=updates, seeds=[1703])))


def test_updates_multiple_of_400_is_accepted(tmp_path):
    c = _load(tmp_path, dict(PILOT, training=dict(updates=400, seeds=[1703])))
    assert (c.lr_warmup_updates, c.consistency_warmup_updates, c.validation_interval,
            c.early_check_updates, c.consistency_interval) == (20, 40, 20, 56, 1)


# ---------------------------------------------------------------- loss_config

def test_loss_config_takes_only_the_weights_from_arguments(monkeypatch):
    c = dc.load_config(CONFIGS/"full.yaml")
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v7")
    monkeypatch.delenv(checks.SEED_ENV, raising=False)
    monkeypatch.delenv(checks.UPDATES_ENV, raising=False)
    v7 = checks.load_protocol()["loss"]
    monkeypatch.undo()
    got = c.loss_config(time_weight=0.0123, space_weight=0.0456)
    assert type(got) is LossConfig
    assert {f.name for f in dataclasses.fields(LossConfig)} == {
        *dc.V7_LOSS_ITEMS, "time_weight", "space_weight", "consistency_warmup_updates", "grad_clip_norm",
        "lr_min", "lr_decay_updates", "lr_warmup_updates"}
    assert got == LossConfig(jepa_exponent=v7["jepa_exponent"], layer_norm_epsilon=v7["layer_norm_epsilon"],
                             time_epsilon=v7["time_epsilon"], space_epsilon=v7["space_epsilon"],
                             consistency_metric=v7["consistency_metric"], time_weight=0.0123, space_weight=0.0456,
                             consistency_warmup_updates=3000, grad_clip_norm=1.0, lr_min=1e-4,
                             lr_decay_updates=30000, lr_warmup_updates=1500)
    assert got.consistency_metric == "input_scale"
    assert c.loss_config(time_weight=0.0, space_weight=0.0) == dataclasses.replace(got, time_weight=0.0, space_weight=0.0)


def test_loss_config_ignores_and_restores_process_environment(monkeypatch):
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v6")
    monkeypatch.setenv(checks.SEED_ENV, "999999")      # 不是 v7 预登记的种子；load_protocol 直接读会报错
    monkeypatch.setenv(checks.UPDATES_ENV, "2000")
    before = dict(os.environ)
    got = dc.load_config(CONFIGS/"pilot.yaml").loss_config(time_weight=0.1, space_weight=0.2)
    assert dict(os.environ) == before
    assert (got.lr_decay_updates, got.lr_warmup_updates, got.consistency_warmup_updates) == (2000, 100, 200)
    with pytest.raises(ValueError, match="preregistered seeds"):
        checks.load_protocol()


def test_v7_loss_protocol_matches_load_protocol_in_v7_mode(monkeypatch):
    """五项等于 v7 环境下 `checks.load_protocol()` 的值（v6 下目前也相同）；本函数自己不读环境（Codex S01 #2）。"""
    got = dc.v7_loss_protocol()
    assert list(got) == list(dc.V7_LOSS_ITEMS)
    for protocol in ("v7", "v6"):
        monkeypatch.setenv(checks.PROTOCOL_ENV, protocol)
        monkeypatch.delenv(checks.SEED_ENV, raising=False)
        monkeypatch.delenv(checks.UPDATES_ENV, raising=False)
        loss = checks.load_protocol()["loss"]
        assert got == {k: loss[k] for k in dc.V7_LOSS_ITEMS}
    monkeypatch.undo()

    def boom(*args, **kwargs):
        raise AssertionError("v7_loss_protocol must not go through the environment-driven protocol loader")

    for name in ("load_protocol", "is_v7", "v7_updates", "_apply_seed"):
        monkeypatch.setattr(checks, name, boom)
    monkeypatch.setenv(checks.SEED_ENV, "999999")
    monkeypatch.setenv(checks.PROTOCOL_ENV, "v6")
    before = dict(os.environ)
    assert dc.v7_loss_protocol() == got and dict(os.environ) == before


def test_v7_loss_protocol_refuses_v7_overrides_of_the_five_items(monkeypatch):
    real = dc._read_mapping

    def edited(path):
        value = real(path)
        if path.name == "v7.yaml":
            value = copy.deepcopy(value)
            next(iter(value["schedules"].values()))["loss"]["time_epsilon"] = 2e-8
        return value

    monkeypatch.setattr(dc, "_read_mapping", edited)
    with pytest.raises(ValueError, match=r"overrides loss items \['time_epsilon'\]"):
        dc.v7_loss_protocol()


def test_loss_config_takes_clip_and_lr_min_from_config(tmp_path):
    def edit(raw):
        raw["training"].update(grad_clip=0.25, lr_min=2.0e-4)
    got = _load(tmp_path, base_edit=edit).loss_config(time_weight=0.1, space_weight=0.2)
    assert (got.grad_clip_norm, got.lr_min) == (0.25, 2.0e-4)


def test_computed_sha_key_in_config_files_is_rejected(tmp_path):
    """config_sha256 是算出来的；写在 base.yaml 里要按多键报错，不能被悄悄覆盖（Codex S01 #1）。"""
    with pytest.raises(ValueError, match="unknown keys: config_sha256"):
        _load(tmp_path, base_edit=lambda raw: raw.update(config_sha256="ignored"))
    with pytest.raises(ValueError, match="'config_sha256' is not in base.yaml"):
        _load(tmp_path, overlay=dict(PILOT, config_sha256="x"))
