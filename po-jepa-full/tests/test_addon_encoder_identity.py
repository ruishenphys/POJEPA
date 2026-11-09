"""S18a：编码器核对（10 §6）。去前缀、逐位比较、三种结论，以及 tools/encoder_identity.py 的输出。"""

import importlib.util
import json

import pytest
import torch

from po_jepa.runtime import cache
from po_jepa.runtime.addon import encoder_identity as ei
from po_jepa.runtime import workflow as w


def weights():
    g = torch.Generator().manual_seed(7)
    return {"blocks.0.attn.qkv.weight": torch.randn(12, 4, generator=g), "blocks.0.attn.qkv.bias": torch.randn(12, generator=g),
            "norm.weight": torch.ones(4), "pos_embed": torch.randn(1, 3, 4, generator=g)}


def prefixed(state, prefix):
    return {prefix+k: v.clone() for k, v in state.items()}


def test_strip_prefixes_like_load_official():
    s = weights()
    assert ei.strip(prefixed(s, "module.backbone.")).keys() == s.keys()
    assert ei.strip(prefixed(s, "module.")).keys() == s.keys() and ei.strip(prefixed(s, "backbone.")).keys() == s.keys()
    assert list(ei.strip({"backbone.module.x": torch.zeros(1)})) == ["module.x"]    # 先 module. 后 backbone.，各至多一次
    assert list(ei.strip({"module.module.x": torch.zeros(1)})) == ["module.x"]
    with pytest.raises(ValueError, match="prefix collision"):
        ei.strip({"module.a": torch.zeros(1), "a": torch.zeros(1)})
    with pytest.raises(TypeError, match="not a tensor"):
        ei.strip({"a": 1.})


def test_same_different_and_missing():
    s = weights()
    same = ei.compare(prefixed(s, "module.backbone."), prefixed(s, "module."))
    assert same == dict(paired=4, bitwise_equal=4, max_abs_diff=0., unmatched_ac=[], unmatched_encoder=[], dtype_mismatch=[],
                        shape_mismatch=[], verdict="same")
    changed = prefixed(s, "")
    changed["norm.weight"][2] = torch.nextafter(torch.tensor(1.), torch.tensor(2.))      # float32 的 1 + 2^-23
    r = ei.compare(s, changed)
    assert r["verdict"] == "different" and r["bitwise_equal"] == 3 and r["max_abs_diff"] == pytest.approx(2.**-23)
    half = dict(s, **{"norm.weight": s["norm.weight"].to(torch.bfloat16)})
    r = ei.compare(half, s)
    assert r["verdict"] == "different" and r["dtype_mismatch"] == [dict(name="norm.weight", ac="torch.bfloat16", encoder="torch.float32")]
    assert r["max_abs_diff"] == 0.                                            # 数值相同也不算逐位相同
    reshaped = dict(s, **{"pos_embed": s["pos_embed"].reshape(3, 4)})
    r = ei.compare(reshaped, s)
    assert r["verdict"] == "different" and r["shape_mismatch"] == ["pos_embed"] and r["bitwise_equal"] == 3
    signed = dict(s, **{"norm.weight": torch.tensor([1., 1., 1., -0.])}), dict(s, **{"norm.weight": torch.tensor([1., 1., 1., 0.])})
    assert ei.compare(*signed)["verdict"] == "different"                     # 按存储的字节比：−0 和 +0 不同
    extra = dict(s, **{"head.weight": torch.zeros(2)})
    r = ei.compare(extra, s)
    assert r["verdict"] == "missing_keys" and r["unmatched_ac"] == ["head.weight"] and r["bitwise_equal"] == 4
    r = ei.compare(s, extra)
    assert r["verdict"] == "missing_keys" and r["unmatched_encoder"] == ["head.weight"]


def test_identity_report_pairs_and_primary():
    s = weights()
    ac = dict(encoder=prefixed(s, "module.backbone."), predictor={"x": torch.zeros(1)})
    enc = dict(target_encoder=prefixed(s, "module."), encoder=dict(prefixed(s, "module."), **{"module.norm.weight": torch.zeros(4)}))
    r = ei.identity_report(ac, enc)
    assert r["primary"] == dict(ac_key="encoder", encoder_key="target_encoder") and r["verdict"] == "same"
    assert [(c["ac_key"], c["encoder_key"], c["primary"], c["verdict"]) for c in r["comparisons"]] == [
        ("encoder", "target_encoder", True, "same"), ("encoder", "encoder", False, "different")]
    ac["target_encoder"] = prefixed(s, "")
    assert len(ei.identity_report(ac, enc)["comparisons"]) == 4
    r = ei.identity_report(ac, dict(target_encoder=enc["encoder"]))
    assert r["verdict"] == "different"
    with pytest.raises(KeyError, match="AC checkpoint has no 'encoder'"):
        ei.identity_report(dict(target_encoder=s), enc)
    with pytest.raises(KeyError, match="encoder checkpoint has no 'target_encoder'"):
        ei.identity_report(ac, dict(encoder=s))
    with pytest.raises(TypeError, match="must load as dicts"):
        ei.identity_report([s], enc)


def load_tool():
    spec = importlib.util.spec_from_file_location("encoder_identity_tool", w.ROOT/"tools"/"encoder_identity.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_tool_writes_report_and_refuses_overwrite(tmp_path):
    s = weights()
    torch.save(dict(encoder=prefixed(s, "module.backbone."), predictor={}), tmp_path/"ac.pt")
    torch.save(dict(target_encoder=prefixed(s, "module."), encoder=prefixed(s, "module.")), tmp_path/"enc.pth")
    tool = load_tool()
    out = tmp_path/"nested"/"encoder-identity.json"
    args = ["--encoder", str(tmp_path/"enc.pth"), "--ac-checkpoint", str(tmp_path/"ac.pt"), "--output", str(out)]
    assert tool.main(args) == 0
    report = json.loads(out.read_text())
    assert report["kind"] == "encoder_identity" and report["verdict"] == "same" and report["scientific_conclusions_verified"] is False
    assert report["encoder"] == dict(path=str((tmp_path/"enc.pth").resolve()), sha256=cache.file_sha(tmp_path/"enc.pth"))
    assert report["ac_checkpoint"]["sha256"] == cache.file_sha(tmp_path/"ac.pt")
    assert out.read_bytes() == cache._json_bytes(report)
    with pytest.raises(SystemExit):
        tool.main(args)
    with pytest.raises(SystemExit):
        tool.main(["--encoder", str(tmp_path/"missing.pth"), "--ac-checkpoint", str(tmp_path/"ac.pt"), "--output", str(tmp_path/"x.json")])
    assert not (tmp_path/"x.json").exists()
