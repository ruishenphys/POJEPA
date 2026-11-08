# 05 DROID 编码与特征格式

2026-11-08 · 第一版

阶段 4：用冻结的 V-JEPA 2 ViT-g 把每条回合的取帧位置编码成特征，训练和评测之后只读特征、不跑编码器。每片一个 GPU 任务 `encode/<片名>`，另加 `encode/test-extra`。代码在 `src/po_jepa/droid/encode.py` 和 `src/po_jepa/droid/features.py`（S06）。

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

特征第 i 行就是第 15i 帧（第 i 个周期边界）。每条回合平均约 35 行；训练前 2 万条共 695,986 行，float16 约 500 GB。

## 3. 片完成和清单

全部回合完成后，`encode/<片名>` 写：

- `features/<片名>/MANIFEST.json`：这一片在该配置下的 uuid 列表，每条的 JSON SHA、文件 SHA、行数；跳过的回合和原因。
- `features/<片名>/_complete/<配置名>.json`：MANIFEST 的 SHA、配置 SHA。

全部片完成后，训练任务读取前先生成**特征总清单**的 SHA：按片名排序，把每片 MANIFEST 的 SHA 依次连接后取 SHA-256。它就是 `TrainingBatch.provenance["feature_manifest_sha256"]`（06 §5）。

带 `--delete-raw` 时（只用于训练片），清单写完并重新读一遍核对通过后删原始 MP4 和 h5（03 §4）。

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

## 5. 读特征

`features.FeatureReader(root, manifest)`：

- `np.load(path, mmap_mode="r")`，只读需要的行；
- 每读一行就核这一行的 SHA（float16 字节，约 0.4 毫秒），不对就报错；
- 转成 float32 的 torch 张量返回。

训练时在数据加载进程里读和核，不占 GPU（06 §5）。

## 6. 测试用的假编码器

CPU 测试不用真实权重：沿用 v7 `encode_v7.py` 里的假编码器思路（固定随机投影：把 16×16 的图块拉平后乘一个固定种子的随机矩阵，得到 [B, 256, 1408]），192 和 320 输入相应得到 144 和 400 个 token。假编码器写的清单里 `encoder_checkpoint_sha256` 为 null、`evidence_role` 为 `synthetic_development_only`，训练读到这种特征时只允许跑冒烟测试。

## 7. 真实编码器核对（S06 合并前，在 GPU 机器上）

- 取一条金标准回合的前 32 个取帧位置，用本模块编码，和直接用 v7 `FrozenFrameEncoder` 在同一块大小下编码的结果逐位相同。
- 192 和 320 输入各编 1 帧，输出形状正确、全部有限。
