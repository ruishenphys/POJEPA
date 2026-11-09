"""S01：DROID 7 维动作与 ControlSpec（docs/design/01-droid-config.md §5、§6；验收 14 §3 S01）。"""

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy
import pytest
import scipy.spatial.transform
import yaml

from po_jepa.actions import ControlSpec
from po_jepa.droid import actions as da
from po_jepa.droid import config as dc
from po_jepa.runtime import cache

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT/"docs"/"design"/"droid-golden"
OFFICIAL = GOLDEN/"official_droid_204698b.py"
# 上游 git 里的原始字节（LF）；reference/upstream_sources.json 登记的是 CRLF 检出的 SHA
OFFICIAL_LF_SHA = "25f3fe88d6d29af0460155f5e0f2b2d05e005f6201a2090bf351fc3daee1d549"
SCALE = {"realized": {"position": 0.0123, "rotation": 0.0456, "gripper": 0.25},
         "commanded": {"position": 0.0222, "rotation": 0.0555, "gripper": 0.5},
         "episodes": ["a", "b"], "quantile": 0.99, "method": "numpy.quantile linear float64"}


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1
    return found[0]


def test_vendored_upstream_file_is_the_registered_one():
    raw = OFFICIAL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == OFFICIAL_LF_SHA
    registered = json.loads((ROOT/"reference"/"upstream_sources.json").read_text(encoding="utf-8"))
    crlf = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    assert hashlib.sha256(crlf).hexdigest() == registered["upstream_raw_sha256"]["app/vjepa_droid/droid.py"]


def test_poses_to_diffs_body_is_verbatim_upstream():
    upstream = _function(ast.parse(OFFICIAL.read_text(encoding="utf-8")), "poses_to_diffs")
    ours = _function(ast.parse(inspect.getsource(da)), "poses_to_diffs")
    assert [a.arg for a in upstream.args.args] == ["self", "poses"]
    assert [a.arg for a in ours.args.args] == ["poses"]
    assert not ours.decorator_list and ours.returns is None
    dump = lambda f: [ast.dump(s, include_attributes=False) for s in f.body]
    assert dump(ours) == dump(upstream)
    # 函数体里的名字解析到上游用的同一个对象（`np` 是 numpy，`Rotation` 是 scipy 的）
    assert da.np is numpy and da.Rotation is scipy.spatial.transform.Rotation


def test_poses_to_diffs_matches_golden_episodes_bit_for_bit():
    golden = json.loads((GOLDEN/"droid_samples_golden.json").read_text(encoding="utf-8"))
    assert len(golden["episodes"]) == 5
    for ep in golden["episodes"]:
        states = numpy.array(ep["stride15_states"], dtype=numpy.float64)
        got = da.poses_to_diffs(states)
        ref = numpy.array(ep["official_actions"], dtype=numpy.float64)
        assert got.dtype == numpy.float64 and got.shape == (len(states) - 1, 7) == ref.shape
        assert numpy.array_equal(got, ref), ep["uuid"]
        assert got.tobytes() == ref.tobytes()


def test_poses_to_diffs_hand_values():
    poses = numpy.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0],
                         [0.4, 0.0, 0.3, 0.0, 0.0, 0.25, 0.5],
                         [0.4, 0.0, 0.3, 0.0, 0.0, 0.25, 0.5]])
    got = da.poses_to_diffs(poses)
    expected = numpy.array([[0.3, -0.2, 0.0, 0.0, 0.0, 0.25, 0.5], [0.0] * 7])
    assert got.shape == (2, 7)
    numpy.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)
    # 旋转差是相对旋转 R_{t+1} R_t^T 的欧拉角，不是欧拉角相减：绕 x 转 0.3 后再绕 z 转 0.2
    r0 = scipy.spatial.transform.Rotation.from_euler("xyz", [0.3, 0.0, 0.0])
    r1 = scipy.spatial.transform.Rotation.from_euler("z", 0.2) * r0
    poses = numpy.zeros((2, 7))
    poses[0, 3:6], poses[1, 3:6] = r0.as_euler("xyz"), r1.as_euler("xyz")
    numpy.testing.assert_allclose(da.poses_to_diffs(poses)[0, 3:6], [0.0, 0.0, 0.2], atol=1e-12)


@pytest.mark.parametrize("source", ["realized", "commanded"])
def test_control_spec_fields(source, tmp_path):
    config = dc.load_config(ROOT/"configs"/"droid"/"full.yaml")
    spec = da.droid_control_spec(SCALE, source=source, config=config)
    path = tmp_path/"action_scale.json"
    cache._write_json(path, SCALE)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    p, r, g = (SCALE[source][k] for k in ("position", "rotation", "gripper"))
    assert spec == ControlSpec(
        action_dim=7, control_period_seconds=1.0,
        channel_names=("dx", "dy", "dz", "rx", "ry", "rz", "grip"),
        channel_units=("m", "m", "m", "rad", "rad", "rad", "1"),
        lower_bounds=(-1.0, -1.0, -1.0, -3.2, -3.2, -3.2, -1.0),
        upper_bounds=(1.0, 1.0, 1.0, 3.2, 3.2, 3.2, 1.0),
        scale=(p, p, p, r, r, r, g), offset=(0.0,) * 7,
        definition_source=(f"DROID 1.0.1 {source} pose diffs, vjepa2@204698b poses_to_diffs, "
                           f"15 frames per interval, action_scale.json sha256={sha}"))
    assert all(type(x) is float for x in (*spec.lower_bounds, *spec.upper_bounds, *spec.scale, *spec.offset))


def test_control_spec_bounds_come_from_config(tmp_path):
    raw = yaml.safe_load((ROOT/"configs"/"droid"/"base.yaml").read_text(encoding="utf-8"))
    raw["action"].update(position_bound_m=0.5, rotation_bound_rad=2.0, gripper_bound=0.75)
    (tmp_path/"base.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    (tmp_path/"pilot.yaml").write_text((ROOT/"configs"/"droid"/"pilot.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    spec = da.droid_control_spec(SCALE, source="realized", config=dc.load_config(tmp_path/"pilot.yaml"))
    assert spec.upper_bounds == (0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 0.75)
    assert spec.lower_bounds == tuple(-x for x in spec.upper_bounds)


def _bad(edit):
    scale = copy.deepcopy(SCALE)
    edit(scale)
    return scale


@pytest.mark.parametrize("scale,source,message", [
    (SCALE, "velocity", "source must be one of"),
    (_bad(lambda s: s.pop("commanded")), "commanded", "must contain a 'commanded' mapping"),
    ([0.1, 0.2, 0.3], "realized", "must contain a 'realized' mapping"),
    (_bad(lambda s: s["realized"].pop("gripper")), "realized", "must give exactly"),
    (_bad(lambda s: s["realized"].update(extra=1.0)), "realized", "must give exactly"),
    (_bad(lambda s: s["realized"].update(position=0.0)), "realized", r"realized\.position must be a positive finite float"),
    (_bad(lambda s: s["realized"].update(rotation=-0.1)), "realized", r"realized\.rotation must be a positive finite float"),
    (_bad(lambda s: s["realized"].update(gripper=float("inf"))), "realized", r"realized\.gripper must be"),
    (_bad(lambda s: s["realized"].update(gripper=float("nan"))), "realized", r"realized\.gripper must be"),
    (_bad(lambda s: s["realized"].update(gripper=1)), "realized", r"realized\.gripper must be"),
    (_bad(lambda s: s["realized"].update(gripper=True)), "realized", r"realized\.gripper must be"),
])
def test_control_spec_rejects_bad_scale(scale, source, message):
    config = dc.load_config(ROOT/"configs"/"droid"/"pilot.yaml")
    with pytest.raises(ValueError, match=message):
        da.droid_control_spec(scale, source=source, config=config)
