# 05 DROID 编码与特征格式

2026-11-08 · 第一版

阶段 4：用冻结的 V-JEPA 2 ViT-g 把每条回合的取帧位置编码成特征，训练和评测之后只读特征、不跑编码器。每片一个 GPU 任务 `encode/<片名>`，另加 `encode/test-extra`。代码在 `src/po_jepa/droid/encode.py` 和 `src/po_jepa/droid/features.py`（S06 拆两段：S06a 主特征、片清单、读取、`--delete-raw`、`droid/poses.py`；S06b 测试附加 §4）。

## 1. 编码器和预处理（和 v7 完全相同）

- 编码器：上游 `vjepa2@204698b` 的 `vit_giant_xformers`，权重 `vjepa2-vitg-fpc64-256.pth`，用 v7 的 `vjepa.build_encoder_architecture` 和 `load_local_encoder` 载入，权重 SHA 必须等于 v7 `cache.CHECKPOINT_SHA`。
- 256 输入：直接用 v7 的 `vjepa.preprocess_rgb(rgb, spec=cache.preprocess_spec(), generator=…)` 和 `FrozenFrameEncoder`（单帧复制两份凑 tubelet，float32，不开 autocast）。1280×720 的输入在官方 `random_resized_crop` 里固定落到中心回退：取第 0–719 行、第 154–1125 列（972×720），双线性缩到 256×256，没有随机性；5 条金标准回合上和官方函数、和手写切片加 `F.interpolate(mode="bilinear", align_corners=False)` 都逐位相同。generator 固定 `torch.Generator().manual_seed(0)`，只为满足接口。
- 192、320 输入（只用于测试附加）：v7 的 `PreprocessSpec` 只接受 256，不改它。新函数 `preprocess_square(rgb, size)`：同样的切片 [0:720, 154:1126]，`F.interpolate(…, size=(size, size), mode="bilinear", align_corners=False)`，再按 v7 同一组均值和方差归一化。要求 `preprocess_square(rgb, 256)` 和 v7 的 `preprocess_rgb` 逐位相同（S06 测试）。编码直接调同一个已载入编码器的 `forward`（单帧复制两份），192 得 12×12 = 144 个 token，320 得 20×20 = 400 个；上游编码器会按输入尺寸处理位置编码。
- 每次送进编码器 `batch_frames`（32）帧：按行序连续切块，最后一块可以不满。块大小固定，因为 GPU 上同一帧在不同块大小下的结果不保证逐位相同。

## 2. 主特征

对每条回合（只编准备检查通过的，04 §4；没通过的在片清单里记为跳过）：

1. 重新解码 MP4（04 §2），取 `frame_index` 处的帧，逐帧算 RGB 的 SHA 和准备文件里的 `frame_sha256` 比较，不同就报错。
2. 预处理、编码，得到 float32 [F, 256, 1408]；检查全部有限。
3. 转 float16 存成 `features/<片名>/<uuid>.npy`（先写 `.tmp` 再改名）。转之前检查最大绝对值 < 65504，并记下最大相对舍入误差（v7 同样做法）。
4. 写 `features/<片名>/<uuid>.json`：`rows`、`shape`、`dtype`、文件 SHA、**每一行的 SHA**（float16 字节）、`frame_sha256`（抄自准备文件，便于下游直接核）、准备文件 JSON 的 SHA、编码器权重 SHA、预处理 SHA（v7 `cache.v7_preprocess_sha256()`）、上游提交、`batch_frames`、最大相对舍入误差、代码 SHA、GPU 型号和驱动、torch 版本、用时。

特征第 i 行就是第 15i 帧（第 i 个周期边界）。每条回合平均约 35 行；训练前 2 万条共 696,912 行，float16 约 500 GB。

## 3. 片完成和清单

重跑时，一条回合已有特征 JSON，且其中的文件 SHA、准备文件 JSON 的 SHA、编码器权重 SHA、预处理 SHA、`batch_frames` 都和当前一致（文件 SHA 重新算一遍），就跳过这条。所以试跑编过的回合正式运行直接复用。

全部回合完成后，`encode/<片名>` 写：

- `features/<片名>/MANIFEST.<配置名>.json`：这一片在该配置下的 uuid 列表，每条的 JSON SHA、文件 SHA、行数；跳过的回合和原因（S06a 改：原写 `MANIFEST.json`，但试跑和正式运行的 `train-000` 回合列表不同，所以按配置名分文件，同 04 §6 的统计）。
- `features/<片名>/_complete/<配置名>.json`：MANIFEST 的 SHA、配置 SHA。

全部片完成后，训练任务读取前先生成**特征总清单**的 SHA：按片名排序（Python 字符串顺序），把每片 MANIFEST 的 SHA（64 位十六进制串）依次连接后取 SHA-256（`features.total_manifest_sha256`）。它就是 `TrainingBatch.provenance["feature_manifest_sha256"]`（06 §5）。

带 `--delete-raw` 时（只用于训练片），清单写完并重新读一遍核对通过后删原始 MP4 和 h5（03 §4）。

S06a 补充（实现时定，见 `docs/segments/S06a.md` §4）：
- “准备检查通过”取两种动作源任一通过（`passed_realized` 或 `passed_commanded`）：特征只和画面有关，两种配置共用一份；两种都没通过的回合记为跳过，原因抄自准备文件的 `reasons`。
- 跳过规则另要求片名、证据角色（`evidence_role`）、上游提交一致；特征 JSON 的 `encoder` 字段就是这几项加权重 SHA、预处理 SHA、`batch_frames`。
- 重新读一遍核对 = 每条特征 JSON 的 SHA 等于清单、`.npy` 整个文件重算 SHA、dtype/形状/行数、每一行的 SHA。对每条回合先写 `RAW_DELETED.json`（`schema`、删除时间、特征 JSON 的 SHA、删的文件名），再删文件；中断后重跑会删掉剩下的。准备没通过、没编码的回合不删原始文件（留着查原因）。
- 完成记录已在时任务不再编码；带 `--delete-raw` 时仍做删除那一步（幂等）。
- 假编码器的预处理是 `preprocess_square(·, 256)`（和 v7 `preprocess_rgb` 逐位相同，但 CPU 测试没有上游代码），特征 JSON 记它自己的预处理 SHA（`encode.square_preprocess_sha256(256)`），不冒充 v7 的。

## 4. 测试附加 `encode/test-extra`

对 `id_test` 和 `ood_test` 两片各自的前 200 条（按划分列表顺序，跳过没通过检查的，不补位）：

| 文件 | 内容 | 用途 |
|---|---|---|
| `features/test_extra/<uuid>.s5.npy` | float16 [F5, 256, 1408]，第 j 行是第 5j 帧，F5 = ⌊(V − 1)/5⌋ + 1 | 原始帧率评测（10 §8.4） |
| `features/test_extra/<uuid>.r192.npy` | float16 [F, 144, 1408]，取帧位置同主特征 | 原始分辨率 12×12 |
| `features/test_extra/<uuid>.r320.npy` | float16 [F, 400, 1408] | 原始分辨率 20×20 |

每个文件旁边同样写 JSON（每行 SHA 等）。s5 的帧 SHA 核对准备文件里的 `frame_sha256_s5`。

s5 文件里第 3j 行和主特征第 j 行是同一帧，但编码时的块不同，GPU 上不保证逐位相同。原始帧率评测的起点和终点都从 s5 文件读，不混用主特征；两者逐位相同的比例记进清单，只作记录。

约 400 条 × (约 110 + 2 × 37) 行 ≈ 7.5 万帧。

S06b 补充（实现时定，见 `docs/segments/S06b.md` §4）：
- 代码在 `src/po_jepa/droid/encode_extra.py`，命令行 `python -m po_jepa.droid.encode … --shard test-extra`；读取用 `features.ExtraFeatureReader`（逐行核 SHA）。
- “没通过检查”同 S06a：两种动作源都没通过才跳过（特征只和画面有关）；评测按 06 §3 只用 `passed_realized` 的附加回合，是这里编的子集。
- 每条回合 MP4 只解码一次（主特征位置和 s5 位置的并集），同一帧在 `frame_sha256` 和 `frame_sha256_s5` 里的 SHA 必须相同。
- s5 的编码器和预处理就是主特征的（v7 `preprocess_rgb` + `FrozenFrameEncoder`）；r192、r320 用 `preprocess_square(·, R)` 和同一个已载入编码器的 `native_encoder.NativeFrameEncoder`（S25e）。
- 清单 `features/test_extra/MANIFEST.<配置名>.json`，每条回合记三种文件的特征 JSON SHA、文件 SHA、行数；跳过的回合记片名和原因；完成记录 `features/test_extra/_complete/<配置名>.json`。
- 和主特征逐位相同的计数（`main_comparison`，按帧号配对）只在该片主特征于本配置下已完成时才算，否则记 `main_manifest_sha256: null`。
- §7 第二项：`--check-mp4` 另把第 0 帧按 192、320 各编一次，形状和有限性都对才返回 0。

## 5. 读特征

`features.FeatureReader(root, manifest)`：

- `np.load(path, mmap_mode="r")`，只读需要的行；
- 每读一行就核这一行的 SHA（float16 字节，约 0.4 毫秒），不对就报错；
- 转成 float32 的 torch 张量返回。

训练时在数据加载进程里读和核，不占 GPU（06 §6）。

## 6. 测试用的假编码器

CPU 测试不用真实权重：沿用 v7 `encode_v7.py` 里的假编码器思路（固定随机投影：把 16×16 的图块拉平后乘一个固定种子的随机矩阵，得到 [B, 256, 1408]），192 和 320 输入相应得到 144 和 400 个 token。假编码器写的清单里 `encoder_checkpoint_sha256` 为 null、`evidence_role` 为 `synthetic_development_only`，训练读到这种特征时只允许跑冒烟测试。

## 7. 真实编码器核对（GPU 机器上；S06a 合入时没有 GPU，交给算力方）

- 取一条金标准回合的前 32 个取帧位置，用本模块编码，和直接用 v7 `FrozenFrameEncoder` 在同一块大小下编码的结果逐位相同（`python -m po_jepa.droid.encode --config … --data-root … --check-mp4 <金标准 MP4>`，退出码 0 即通过）。
- 192 和 320 输入各编 1 帧，输出形状正确、全部有限。
