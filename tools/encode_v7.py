"""v7 特征编码：仿真 → 渲染 → 冻结 V-JEPA 2 ViT-g 编码 → 逐轨迹缓存。

和 v3 的编码完全相同的编码器（同一权重 SHA、同一上游提交、同一 DROID 预处理、单帧复制两份凑 tubelet）。
不同点：系统和条数可选；特征默认存 float16（约 24 MB/条，2000 条约 48 GB），训练读取时转回 float32。

用法（在 code-v7/po-jepa 目录里）：
  # 编码；多卡时每张卡跑一个分片，例如 4 张卡：--shard 0/4 … --shard 3/4
  python tools/encode_v7.py encode --system van_der_pol --trajectories 2000 \
      --upstream-root ../vjepa2 --checkpoint ../checkpoints/vjepa2-vitg-fpc64-256.pth \
      --output ../features-v7/van_der_pol-2000 --device cuda:0 --shard 0/1
  # 全部分片完成后汇总（几秒，不用 GPU）
  python tools/encode_v7.py finalize --system van_der_pol --trajectories 2000 --output ../features-v7/van_der_pol-2000

中途断了直接重跑同一条命令：已完成（有 COMPLETE.json）的轨迹会跳过，未完成的残留会被清掉重做。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def _parse():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("encode", "finalize"))
    parser.add_argument("--system", required=True)
    parser.add_argument("--trajectories", type=int, required=True)
    parser.add_argument("--variant", default="", help="可调数据集的调节项，例如 gain=2,axes=x（见 src/po_jepa/variants.py）")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", default="0/1", help="i/n：只编码 trajectory_id %% n == i 的轨迹")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--splits", default="train,validation,test",
                        help="只编码这些划分；变体测试集用 --splits test（只能用于评测，编号与基础场景测试段一一对应）")
    parser.add_argument("--batch", type=int, default=33, help="每次送进编码器的帧数")
    parser.add_argument("--fake-encoder", action="store_true",
                        help="仅供 CPU 流程测试：用确定性随机投影代替编码器，产物标记为 synthetic，训练入口会拒绝")
    return parser.parse_args()


def main():
    args = _parse()
    os.environ["PO_JEPA_DATASET"] = f"{args.system}{'~'+args.variant if args.variant else ''}:{args.trajectories}"
    sys.path.insert(0, str(ROOT/"src"))
    import numpy as np
    import torch
    from po_jepa import datasets, simulation, systems, vjepa
    from po_jepa.runtime import cache

    spec = datasets.active()
    out = args.output.resolve()

    if args.mode == "finalize":
        return finalize(out, spec, cache, _splits(args.splits))

    index, _, total = args.shard.partition("/")
    index, total = int(index), int(total)
    if not 0 <= index < total:
        raise SystemExit("--shard 必须是 i/n 且 0 <= i < n")
    out.mkdir(parents=True, exist_ok=True)
    system = systems.get()
    dtype = {"float16": np.float16, "float32": np.float32}[args.dtype]
    device = torch.device(args.device)

    if args.fake_encoder:
        encoder, load_report, role = _fake_encoder(torch), dict(sha256=None, encoder_key=None, strict=None,
                                                                missing_keys=[], unexpected_keys=[]), "synthetic_development_only"
    else:
        if args.upstream_root is None or args.checkpoint is None:
            raise SystemExit("encode 需要 --upstream-root（vjepa2 仓库，提交 %s）和 --checkpoint" % vjepa.UPSTREAM_COMMIT)
        arch = vjepa.EncoderArchitectureSpec(vjepa.UPSTREAM_COMMIT, "configs/train/vitg16/droid-256px-8f.yaml",
            "vit_giant_xformers", 256, 16, 2, 512, True, True, False, True, True, True)
        build = vjepa.build_encoder_architecture(upstream_root=args.upstream_root, architecture=arch, initialization_seed=0)
        loaded = vjepa.load_local_encoder(build, checkpoint=vjepa.LocalCheckpointSpec(
            args.checkpoint, cache.CHECKPOINT_SHA, cache.PUBLICATION, "target_encoder",
            ("module.", "backbone."), True, (), ()))
        frozen = vjepa.FrozenFrameEncoder(loaded, preprocess=cache.preprocess_spec(), encoder_device=device,
                                          autocast_dtype=None)
        encoder = frozen
        r = loaded.report
        load_report = dict(path=str(r.path), sha256=r.sha256, encoder_key=r.encoder_key, strict=r.strict,
                           missing_keys=list(r.missing_keys), unexpected_keys=list(r.unexpected_keys),
                           architecture=dict(vars(r.architecture)))
        role = "real_frozen_encoder"

    generator_sources = {name: cache.file_sha(ROOT/name) for name in
                         ("src/po_jepa/simulation.py", "src/po_jepa/systems.py", "src/po_jepa/variants.py", "src/po_jepa/vjepa.py",
                          "src/po_jepa/datasets.py", "tools/encode_v7.py")}
    environment = dict(python=sys.version.split()[0], torch=torch.__version__, cuda_runtime=torch.version.cuda,
                       numpy=np.__version__, platform=platform.platform(), encoder_device=str(device),
                       device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    wanted = _splits(args.splits)
    ids = [t for t in range(spec.count) if t % total == index and spec.split_for(t) in wanted]
    started = time.perf_counter(); done = 0
    for tid in ids:
        folder = out/f"{tid:04d}"
        if (folder/"COMPLETE.json").exists():
            continue
        if folder.exists():
            shutil.rmtree(folder)       # 上次中断留下的半成品
        folder.mkdir()
        states, frames = system.frames(tid)
        raw = _encode_trajectory(encoder, frames, tid, simulation, cache, torch, np, args.batch, args.fake_encoder)
        stored = raw.astype(dtype)
        if not np.isfinite(stored).all():
            raise ValueError(f"trajectory {tid}: features overflow {args.dtype}")
        rounding = float(np.abs(stored.astype(np.float64)-raw).max()/max(np.abs(raw).max(), 1e-12))
        np.save(folder/"raw_z.npy", stored, allow_pickle=False)
        np.save(folder/"states.npy", states, allow_pickle=False)
        controls = systems.commanded_controls(tid)
        manifest = dict(
            schema_version=cache.V7_SCHEMA, status="complete", system=spec.system, dataset_count=spec.count,
            **({"variant": spec.variant} if spec.variant else {}),
            trajectory_id=tid, split=spec.split_for(tid), shape=list(stored.shape), dtype=args.dtype,
            feature_file="raw_z.npy", feature_sha256=cache.file_sha(folder/"raw_z.npy"),
            states_file="states.npy", states_sha256=cache.file_sha(folder/"states.npy"),
            grid=[16, 16], token_order="row_major", observation_variant="canonical",
            preprocess_sha256=cache.v7_preprocess_sha256(),
            controls=dict(starts_ticks=controls.starts_ticks.tolist(), ends_ticks=controls.ends_ticks.tolist(),
                          values=controls.values.tolist(), units="simulation_position_per_second"),
            frames=[dict(tick=t, time_seconds=t*.25, image_sha256=hashlib.sha256(frames[t].tobytes()).hexdigest())
                    for t in range(33)],
            feature_stats=dict(max_abs=float(np.abs(raw).max()), relative_rounding_error=rounding),
            rng=dict(seed_entropy=simulation.seed_entropy(tid, 0), preprocess_seed=simulation.preprocess_seed(tid)),
            generation=dict(evidence_role=role, encoder_checkpoint_sha256=None if args.fake_encoder else cache.CHECKPOINT_SHA,
                            encoder_load_report=load_report, upstream_commit=vjepa.UPSTREAM_COMMIT,
                            preprocess=cache._v7_preprocess_record(), generator_source_sha256=generator_sources,
                            environment=environment,
                            preprocess_rule="reset_same_trajectory_generator_state_before_each_single_frame"))
        cache._write_json(folder/"manifest.json", manifest)
        cache._write_json(folder/"COMPLETE.json", dict(manifest_sha256=cache.file_sha(folder/"manifest.json")))
        done += 1
        if done % 20 == 0 or done == 1:
            rate = (time.perf_counter()-started)/done
            left = sum(1 for t in ids if not (out/f"{t:04d}"/"COMPLETE.json").exists())
            print(f"[{spec.tag} shard {index}/{total}] 已编码 {done} 条，约 {rate:.1f} s/条，本分片剩余 {left} 条", flush=True)
    print(f"[{spec.tag} shard {index}/{total}] 完成", flush=True)


def _encode_trajectory(encoder, frames, tid, simulation, cache, torch, np, batch, fake):
    """按 v3 规则：每帧前把同一条轨迹的预处理 generator 重置到同一初始状态，再单帧预处理并编码。"""
    rgb = torch.from_numpy(frames)
    if fake:
        x = rgb.permute(0, 3, 1, 2).float()/255.
        return encoder(x).numpy().astype(np.float64)
    from po_jepa import vjepa
    seed = simulation.preprocess_seed(tid)
    pre = []
    for t in range(rgb.shape[0]):
        g = torch.Generator(device="cpu").manual_seed(seed)
        pre.append(vjepa.preprocess_rgb(rgb[t:t+1], spec=cache.preprocess_spec(), generator=g))
    pre = torch.cat(pre)
    out = []
    for s in range(0, pre.shape[0], batch):
        chunk = pre[s:s+batch]
        times = torch.arange(s, s+chunk.shape[0], dtype=torch.float64)*.25
        out.append(encoder.encode_target(chunk, target_time_seconds=times).raw_z.cpu().double())
    return torch.cat(out).numpy()


def _fake_encoder(torch):
    """确定性随机 16×16 patch 投影（仅流程测试）：[L,3,256,256] → [L,256,1408]。"""
    g = torch.Generator().manual_seed(0)
    weight = torch.randn(3*16*16, 1408, generator=g)/16.
    def encode(x):
        patches = x.unfold(2, 16, 16).unfold(3, 16, 16)              # L,3,16,16,16,16
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(x.shape[0], 256, -1)
        return torch.nn.functional.layer_norm(patches@weight, (1408,))
    return encode


def _splits(text):
    names = tuple(s for s in ("train", "validation", "test") if s in text.split(","))
    if not names or set(text.split(",")) - set(names):
        raise SystemExit("--splits 只能是 train/validation/test 的逗号组合")
    return names


def finalize(out, spec, cache, splits=("train", "validation", "test")):
    records, missing = [], []
    for tid in (t for t in range(spec.count) if spec.split_for(t) in splits):
        folder = out/f"{tid:04d}"
        if not (folder/"COMPLETE.json").exists():
            missing.append(tid); continue
        complete = json.loads((folder/"COMPLETE.json").read_bytes())
        if complete["manifest_sha256"] != cache.file_sha(folder/"manifest.json"):
            raise ValueError(f"trajectory {tid}: manifest SHA mismatch")
        m = json.loads((folder/"manifest.json").read_bytes())
        if m["system"] != spec.system or m["dataset_count"] != spec.count or m.get("variant", "") != spec.variant:
            raise ValueError(f"trajectory {tid}: encoded for {m['system']}-{m['dataset_count']}")
        records.append(dict(trajectory_id=tid, split=m["split"], manifest_sha256=complete["manifest_sha256"],
                            evidence_role=m["generation"]["evidence_role"],
                            max_abs=m["feature_stats"]["max_abs"],
                            relative_rounding_error=m["feature_stats"]["relative_rounding_error"]))
    if missing:
        raise SystemExit(f"还有 {len(missing)} 条没编码完（例如 {missing[:5]}），先把所有分片跑完")
    roles = {r["evidence_role"] for r in records}
    report = dict(mode="encode_v7", status="passed", system=spec.system, dataset_count=spec.count,
                  **({"variant": spec.variant} if spec.variant else {}),
                  splits=dict(train=spec.train, validation=spec.validation, test=spec.test),
                  **({"splits_encoded": list(splits)} if tuple(splits) != ("train", "validation", "test") else {}),
                  evidence_roles=sorted(roles), records=records,
                  max_abs=max(r["max_abs"] for r in records),
                  max_relative_rounding_error=max(r["relative_rounding_error"] for r in records))
    for name in ("report.json", "COMPLETE.json"):
        if (out/name).exists(): (out/name).unlink()
    cache._write_json(out/"report.json", report)
    cache._write_json(out/"COMPLETE.json", dict(report_sha256=cache.file_sha(out/"report.json")))
    print(f"{spec.tag}：{len(records)} 条全部完成；特征最大绝对值 {report['max_abs']:.3g}，"
          f"float16 最大相对舍入误差 {report['max_relative_rounding_error']:.2e}")


if __name__ == "__main__":
    main()
