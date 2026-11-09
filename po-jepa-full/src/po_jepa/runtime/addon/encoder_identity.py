"""10 §6：官方 AC 权重 vjepa2-ac-vitg.pt 里自带的编码器，和我们用的 vjepa2-vitg-fpc64-256.pth 的 target_encoder
是不是同一份（CPU；入口 tools/encoder_identity.py，A0 段）。

两边的状态字典按 ac_baseline.load_official 的写法依次去掉 "module."、"backbone." 前缀（每个至多一次），去掉后
重名报错，然后按名字配对。比较对：AC 的 encoder（和 target_encoder，如果有）× fpc64-256 的 target_encoder（和
encoder，如果有）；主比较是 AC encoder 对 fpc64-256 target_encoder，结论 verdict 取主比较的：

- "missing_keys"：任一边有配不上的键；
- "different"：全部配上，但有张量形状或精度不同，或按存储精度不逐位相同；
- "same"：全部配上、形状和精度相同、全部逐位相同。

逐位相同按存储的字节比（reshape(-1) 后 view(torch.uint8)），不转精度；最大绝对差在形状相同的配对上用 float64 算。
"""

import torch

PREFIXES = ("module.", "backbone.")
AC_KEYS = ("encoder", "target_encoder")
ENCODER_KEYS = ("target_encoder", "encoder")
PRIMARY = ("encoder", "target_encoder")


def strip(state: dict) -> dict:
    cleaned = {}
    for key, value in state.items():
        for prefix in PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in cleaned:
            raise ValueError(f"prefix collision after stripping: {key}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state entry {key} is not a tensor")
        cleaned[key] = value
    return cleaned


def _bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                       b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))


def compare(ac_state: dict, encoder_state: dict) -> dict:
    """两份（已去前缀的）状态字典的比较记录。"""
    ac, enc = strip(ac_state), strip(encoder_state)
    names = sorted(set(ac) & set(enc))
    dtype_mismatch, shape_mismatch, equal, max_abs = [], [], 0, 0.
    for name in names:
        a, b = ac[name], enc[name]
        if a.shape != b.shape:
            shape_mismatch.append(name)
            continue
        if a.dtype != b.dtype:
            dtype_mismatch.append(dict(name=name, ac=str(a.dtype), encoder=str(b.dtype)))
        elif _bytes_equal(a, b):
            equal += 1
        if a.numel():
            max_abs = max(max_abs, float((a.detach().cpu().double()-b.detach().cpu().double()).abs().max()))
    unmatched_ac, unmatched_encoder = sorted(set(ac)-set(enc)), sorted(set(enc)-set(ac))
    if unmatched_ac or unmatched_encoder:
        verdict = "missing_keys"
    elif dtype_mismatch or shape_mismatch or equal != len(names):
        verdict = "different"
    else:
        verdict = "same"
    return dict(paired=len(names), bitwise_equal=equal, max_abs_diff=max_abs, unmatched_ac=unmatched_ac,
                unmatched_encoder=unmatched_encoder, dtype_mismatch=dtype_mismatch, shape_mismatch=shape_mismatch,
                verdict=verdict)


def identity_report(ac_checkpoint: dict, encoder_checkpoint: dict) -> dict:
    """两份 torch.load 出来的检查点（顶层是 dict）→ 报告的比较部分。主比较缺键报 KeyError。"""
    if type(ac_checkpoint) is not dict or type(encoder_checkpoint) is not dict:
        raise TypeError("checkpoints must load as dicts")
    for owner, content, key in (("AC", ac_checkpoint, PRIMARY[0]), ("encoder", encoder_checkpoint, PRIMARY[1])):
        if key not in content:
            raise KeyError(f"{owner} checkpoint has no '{key}' entry")
    comparisons = []
    for a in AC_KEYS:
        for b in ENCODER_KEYS:
            if a in ac_checkpoint and b in encoder_checkpoint:
                comparisons.append(dict(ac_key=a, encoder_key=b, primary=(a, b) == PRIMARY,
                                        **compare(ac_checkpoint[a], encoder_checkpoint[b])))
    primary = next(c for c in comparisons if c["primary"])
    return dict(primary=dict(ac_key=PRIMARY[0], encoder_key=PRIMARY[1]), verdict=primary["verdict"], comparisons=comparisons)
