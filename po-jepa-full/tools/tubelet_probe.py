"""S22 两帧 tubelet 探针：探针单摆 → 冻结 V-JEPA 2 编码（dup = v7 的单帧复制、pair = 真实相邻两帧）→ 线性探针读角速度。

定义和判据见 src/po_jepa/tubelet_probe.py 的模块说明和 docs/segments/S22.md。约 1 GPU 小时（v8 规划 §1.4）。

用法（在 po-jepa-full 目录里）：
  # 1) 编码；多卡时每张卡一片：--shard 0/2、--shard 1/2
  python tools/tubelet_probe.py encode --trajectories 200 --output ../features-v8/tubelet-probe-200 \
      --upstream-root ../vjepa2 --checkpoint ../checkpoints/vjepa2-vitg-fpc64-256.pth --device cuda:0 --shard 0/1
  # 2) 全部分片完成后拟合探针（读约 10 GB 特征；有 GPU 时几分钟）
  python tools/tubelet_probe.py fit --trajectories 200 --output ../features-v8/tubelet-probe-200 --device cuda:0
  # 打印给排队系统的任务表（和 make_v8_jobs.py 同一种格式）
  python tools/tubelet_probe.py jobs --shards 2 > jobs-v8-tubelet.txt

结果：<output>/report.json（每种编码、每种读出、每个目标 5 折的 R²，判据），屏幕上打印摘要表。
中途断了直接重跑同一条命令：已完成（有 COMPLETE.json）的轨迹会跳过，未完成的残留会被清掉重做。
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("src/po_jepa/tubelet_probe.py", "src/po_jepa/systems.py", "src/po_jepa/vjepa.py", "tools/tubelet_probe.py")


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("encode", "fit", "jobs"))
    p.add_argument("--trajectories", type=int, default=200)
    p.add_argument("--output", type=Path, default=Path("../features-v8/tubelet-probe-200"))
    p.add_argument("--upstream-root", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard", default="0/1", help="i/n：只编码 trajectory_id %% n == i 的轨迹")
    p.add_argument("--shards", type=int, default=2, help="jobs：编码切成几片（每片一张卡）")
    p.add_argument("--batch", type=int, default=34, help="每次送进编码器的 tubelet 数")
    p.add_argument("--fake-encoder", action="store_true",
                   help="仅供 CPU 流程测试：确定性随机 tubelet 投影代替编码器，产物和报告都标记为 synthetic")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    sys.path.insert(0, str(ROOT/"src"))
    if args.trajectories < 5:
        raise SystemExit("--trajectories 至少 5（5 折，每折至少一条轨迹）")
    if args.mode == "jobs":
        return jobs(args)
    if args.mode == "encode":
        return encode(args)
    return fit(args)


def jobs(args):
    out, n = args.output, args.trajectories
    common = f"--trajectories {n} --output {out}"
    print("# S22 两帧 tubelet 探针（约 1 GPU 小时）。格式同 make_v8_jobs.py：段与段之间有依赖，同一段的命令可以同时跑。")
    print("# ===== T1 编码（每片一张卡）")
    for i in range(args.shards):
        print(f"python tools/tubelet_probe.py encode {common} --upstream-root ../vjepa2 "
              f"--checkpoint ../checkpoints/vjepa2-vitg-fpc64-256.pth --device cuda:0 --shard {i}/{args.shards}")
    print("# ===== T2 拟合探针（等 T1 全部完成）")
    print(f"python tools/tubelet_probe.py fit {common} --device cuda:0")


def _encoder(args, torch):
    from po_jepa import tubelet_probe as tp, vjepa
    from po_jepa.runtime import cache
    if args.fake_encoder:
        return None, dict(sha256=None), "synthetic_development_only"
    if args.upstream_root is None or args.checkpoint is None:
        raise SystemExit("encode 需要 --upstream-root（vjepa2 仓库，提交 %s）和 --checkpoint" % vjepa.UPSTREAM_COMMIT)
    # 与 tools/encode_v7.py 同一个编码器、同一份权重和加载规则
    arch = vjepa.EncoderArchitectureSpec(vjepa.UPSTREAM_COMMIT, "configs/train/vitg16/droid-256px-8f.yaml",
        "vit_giant_xformers", 256, 16, 2, 512, True, True, False, True, True, True)
    build = vjepa.build_encoder_architecture(upstream_root=args.upstream_root, architecture=arch, initialization_seed=0)
    loaded = vjepa.load_local_encoder(build, checkpoint=vjepa.LocalCheckpointSpec(
        args.checkpoint, cache.CHECKPOINT_SHA, cache.PUBLICATION, "target_encoder", ("module.", "backbone."), True, (), ()))
    encoder = tp.FrozenClipEncoder(loaded, preprocess=cache.preprocess_spec(), encoder_device=torch.device(args.device),
                                   autocast_dtype=None)
    r = loaded.report
    return encoder, dict(path=str(r.path), sha256=r.sha256, encoder_key=r.encoder_key, strict=r.strict,
                         missing_keys=list(r.missing_keys), unexpected_keys=list(r.unexpected_keys)), "real_frozen_encoder"


def encode(args):
    import numpy as np
    import torch
    from po_jepa import tubelet_probe as tp, vjepa
    from po_jepa.runtime import cache
    index, _, total = args.shard.partition("/")
    index, total = int(index), int(total)
    if not 0 <= index < total:
        raise SystemExit("--shard 必须是 i/n 且 0 <= i < n")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    encoder, load_report, role = _encoder(args, torch)
    environment = dict(python=sys.version.split()[0], torch=torch.__version__, cuda_runtime=torch.version.cuda,
                       numpy=np.__version__, platform=platform.platform(), encoder_device=args.device)
    sources = {name: cache.file_sha(ROOT/name) for name in SOURCES}
    ids = [t for t in range(args.trajectories) if t % total == index]
    started, done = time.perf_counter(), 0
    for tid in ids:
        folder = out/f"{tid:04d}"
        if (folder/"COMPLETE.json").exists():
            continue
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        states, u = tp.trajectory(tid)
        frames = tp.draw(states)
        feats = tp.encode_trajectory(frames, tid, encoder=encoder, fake=args.fake_encoder, batch=args.batch)
        files = {}
        for mode, raw in feats.items():
            stored = raw.astype(np.float16)
            if not np.isfinite(stored).all():
                raise ValueError(f"trajectory {tid}: {mode} features overflow float16")
            np.save(folder/f"{mode}.npy", stored, allow_pickle=False)
            files[mode] = dict(file=f"{mode}.npy", sha256=cache.file_sha(folder/f"{mode}.npy"), shape=list(stored.shape),
                               clips=tp.clip_indices(mode),
                               relative_rounding_error=float(np.abs(stored.astype(np.float64)-raw).max()/max(np.abs(raw).max(), 1e-12)))
        np.save(folder/"states.npy", states, allow_pickle=False)
        manifest = dict(schema="s22_tubelet_probe_v1", status="complete", trajectory_id=tid, dataset_count=args.trajectories,
                        features=files, dtype="float16", grid=[16, 16], token_order="row_major",
                        states=dict(file="states.npy", sha256=cache.file_sha(folder/"states.npy"),
                                    columns=["theta_rad", "omega_rad_per_s"], frame_ticks=list(range(-1, 33))),
                        controls=u.tolist(),
                        frames=[dict(tick=f-1, image_sha256=hashlib.sha256(frames[f].tobytes()).hexdigest())
                                for f in range(tp.FRAMES)],
                        physics=dict(g_over_l=tp.G_OVER_L, gamma=tp.GAMMA, torque_per_action=tp.TORQUE_PER_ACTION,
                                     substeps=tp.SUBSTEPS, period=tp.PERIOD),
                        rng=dict(entropy=[1703, tp.PROBE_STREAM, tid], preprocess_seed=tp.preprocess_seed(tid)),
                        generation=dict(evidence_role=role, encoder_load_report=load_report,
                                        encoder_checkpoint_sha256=None if args.fake_encoder else cache.CHECKPOINT_SHA,
                                        upstream_commit=vjepa.UPSTREAM_COMMIT, source_sha256=sources, environment=environment,
                                        preprocess_rule=None if args.fake_encoder else
                                        "reset_same_trajectory_generator_state_before_each_single_frame"))
        cache._write_json(folder/"manifest.json", manifest)
        cache._write_json(folder/"COMPLETE.json", dict(manifest_sha256=cache.file_sha(folder/"manifest.json")))
        done += 1
        if done == 1 or done % 20 == 0:
            print(f"[tubelet shard {index}/{total}] 已编码 {done} 条，约 {(time.perf_counter()-started)/done:.1f} s/条", flush=True)
    print(f"[tubelet shard {index}/{total}] 完成", flush=True)


def load(out, n, mode, np, cache):
    """读全部轨迹的一种编码 → ([n·L,256,1408] float16, 状态 [n,34,2], 证据角色集合)。核 COMPLETE、清单 SHA 和特征 SHA。"""
    feats, states, roles, missing = [], [], set(), []
    for tid in range(n):
        folder = out/f"{tid:04d}"
        if not (folder/"COMPLETE.json").exists():
            missing.append(tid); continue
        complete = json.loads((folder/"COMPLETE.json").read_bytes())
        if complete["manifest_sha256"] != cache.file_sha(folder/"manifest.json"):
            raise ValueError(f"trajectory {tid}: manifest SHA mismatch")
        m = json.loads((folder/"manifest.json").read_bytes())
        if m["trajectory_id"] != tid or m["dataset_count"] != n:
            raise ValueError(f"trajectory {tid}: 清单是第 {m['trajectory_id']} 条 / 共 {m['dataset_count']} 条")
        for name, rec in ((mode, m["features"][mode]), ("states", m["states"])):
            if cache.file_sha(folder/rec["file"]) != rec["sha256"]:
                raise ValueError(f"trajectory {tid}: {name} SHA mismatch")
        feats.append(np.load(folder/m["features"][mode]["file"], allow_pickle=False))
        states.append(np.load(folder/"states.npy", allow_pickle=False))
        roles.add(m["generation"]["evidence_role"])
    if missing:
        raise SystemExit(f"还有 {len(missing)} 条没编码完（例如 {missing[:5]}），先把所有分片跑完")
    return np.concatenate(feats), np.stack(states), roles


def fit(args):
    import numpy as np
    import torch
    from po_jepa import tubelet_probe as tp
    from po_jepa.runtime import cache
    out, n = args.output.resolve(), args.trajectories
    device = torch.device(args.device)
    grams, roles, states = {}, set(), None
    for mode in ("dup", "pair"):
        z, s, r = load(out, n, mode, np, cache)
        roles |= r
        if states is None:
            states = s
        elif not np.array_equal(states, s):
            raise ValueError("dup 和 pair 读到的状态不一致")
        z = torch.from_numpy(z)
        rows = tp.FRAMES if mode == "dup" else tp.INTERVALS
        if z.shape[0] != n*rows:
            raise ValueError(f"{mode} 特征行数 {z.shape[0]} ≠ {n}×{rows}")
        for ro in tp.READOUTS:
            k = tp.gram(z, ro, device=device)
            if mode == "dup":
                grams[("dup_concat", ro)] = tp.concat_gram(k, n)
                grams[("dup", ro)] = tp.select_rows(k, tp.FRAMES, tp.sample_rows("dup"), n)
            else:
                grams[("pair", ro)] = k
        del z
    y = np.concatenate([tp.targets(s) for s in states])
    traj = np.repeat(np.arange(n), tp.INTERVALS)
    results = {mode: {ro: tp.ridge_cv(grams[(mode, ro)], y, traj) for ro in tp.READOUTS} for mode in tp.MODES}
    decision = tp.decide(results)
    synthetic = roles != {"real_frozen_encoder"}
    report = dict(mode="s22_tubelet_probe", trajectories=n, samples=int(len(y)), targets=list(tp.TARGETS),
                  folds=tp.FOLDS, lambda_exponents=list(tp.LAMBDA_EXPONENTS), evidence_roles=sorted(roles),
                  evidence_role="synthetic_development_only" if synthetic else "real_frozen_encoder",
                  target_std=y.std(0).tolist(), results=results, decision=decision,
                  source_sha256={name: cache.file_sha(ROOT/name) for name in SOURCES})
    for name in ("report.json",):
        if (out/name).exists():
            (out/name).unlink()
    cache._write_json(out/"report.json", report)
    print(f"{'synthetic（假编码器，不是证据）' if synthetic else '真实编码器'}：{n} 条轨迹、{len(y)} 个样本，5 折测试 R² 平均")
    print("编码        读出    " + "  ".join(f"{t:>10}" for t in tp.TARGETS))
    for mode in tp.MODES:
        for ro in tp.READOUTS:
            print(f"{mode:<11} {ro:<6} " + "  ".join(f"{v:>10.3f}" for v in results[mode][ro]["r2_mean"]))
    d = decision
    print(f"判据：pair 的 θ̇ R² ≥ {tp.ADOPT_R2} 且比 dup 高 ≥ {tp.ADOPT_GAIN}（最好的读出 {d['best_readout']}）；"
          f"位置可读 {d['position_readable']} → 采用两帧输入：{d['adopt_two_frame']}")
    print(f"报告：{out/'report.json'}")


if __name__ == "__main__":
    main()
