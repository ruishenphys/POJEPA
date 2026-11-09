"""10 §6 编码器核对（CPU，A0 段）：官方 AC 权重里的编码器和 fpc64-256 的 target_encoder 是不是同一份。

用法（在 po-jepa-full 目录里）：
  python tools/encoder_identity.py --encoder ../../checkpoints/vjepa2-vitg-fpc64-256.pth \
      --ac-checkpoint ../../checkpoints/vjepa2-ac-vitg.pt --output ../../results-addon/full/encoder-identity.json
输出是一个 JSON（结论 verdict 是 same / different / missing_keys，见 src/po_jepa/runtime/addon/encoder_identity.py）。
--output 必须不存在；先写临时文件再换名。只读两份权重，不需要 GPU。
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--ac-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    import torch
    from po_jepa.runtime import cache
    from po_jepa.runtime.addon import encoder_identity
    if a.output.exists():
        p.error(f"--output 已存在：{a.output}")
    paths = dict(encoder=a.encoder.resolve(), ac_checkpoint=a.ac_checkpoint.resolve())
    for name, path in paths.items():
        if not path.is_file():
            p.error(f"--{name.replace('_', '-')} 不是文件：{path}")
    files = {name: dict(path=str(path), sha256=cache.file_sha(path)) for name, path in paths.items()}
    content = {name: torch.load(path, map_location="cpu", weights_only=True) for name, path in paths.items()}
    report = dict(kind="encoder_identity", **files, **encoder_identity.identity_report(content["ac_checkpoint"], content["encoder"]),
                  scientific_conclusions_verified=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cache._write_json(a.output, report)
    print(f"编码器核对：{report['verdict']} → {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
