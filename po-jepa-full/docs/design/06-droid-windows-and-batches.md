# 06 DROID 窗口和批

2026-11-08 · 第一版

本文件定训练时每一步取哪些窗口、冻结的验证和测试窗口怎么生成，以及怎样把它们组装成 v7 的 `TrainingBatch` 和 AC 的片段。代码在 `src/po_jepa/droid/windows.py` 和 `src/po_jepa/droid/batches.py`（S07）。

## 1. 训练集

`train_episodes(splits, prepared, config) -> list[str]`：取 `splits["train_order"]`（`use_failures: false` 时已去掉失败回合，02 §5）的前 `train_episodes` 条，再去掉 `passed_<source>` 为假的回合，保持原顺序。结果的条数、uuid 列表的 SHA 写进训练登记。正式运行的条数会略少于 20,000（去掉的条数在 `prepared/STATS.json` 里）。

每条回合的可用行数 F 取自准备文件 JSON 的 `rows`。

## 2. 每一步的训练窗口

只由种子和步数决定，不依赖数据加载器的状态，所以续跑后取到的批和不中断时完全相同。

```python
def step_rng(seed: int, step: int, stream: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, step, stream])))
```

**PO-JEPA 类、Transformer 类和 Koopman**（stream 0）：第 `step` 步（从 0 计）的 `batch`（64）行，逐行依次抽：

1. 回合 `e = episodes[rng.integers(len(episodes))]`；
2. 时长 `d = durations[rng.integers(4)]`（durations = [1, 2, 4, 8]）；
3. 起点 `s = rng.integers(F_e − d)`，即 s ∈ [0, F_e − 1 − d]。

同一种子下这几个条件在同一步看到完全相同的 64 个窗口。

**AC 两个条件**（stream 1）：每步 `ac_batch_clips`（16）个 9 帧片段，逐个抽回合 `e` 和起点 `s = rng.integers(F_e − 8)`（片段是第 s 到 s+8 行）。

**标定**（09 §3）用同一个函数，种子取 `calibration.seed`，步数从 0 数到 331（300 步预训练加 32 批）。

窗口编号统一写成 `f"{uuid}|d{d}|s{s}"`（AC 片段写 `f"{uuid}|clip|s{s}"`）。

## 3. 冻结的验证和测试窗口

由 `prepare/stats` 生成（04 §6），开训前和配置、划分一起登记，之后不改。只含 `passed_realized` 为真的回合。排序哈希 `hw(*parts) = sha256("|".join([window_salt, *map(str, parts)]))`。

**验证**（`val`）：每条回合、每个训练时长 d ∈ {1, 2, 4, 8}，候选起点 s ∈ [0, F − 1 − d] 按 `hw("val", uuid, d, s)` 排序取第 1 个。共 300 × 4 = 1,200 个窗口。

**测试**（`id_test`、`ood_test`）：每条回合、每个评测时长 d ∈ {1, 2, 3, 4, 8, 12, 16, 24, 32}，候选起点 s ∈ [0, F − 1 − d]（F − 1 < d 时没有），按 `hw(split, uuid, d, s)` 排序取前 min(4, 候选数) 个。

**附加回合**：每个测试集按划分列表顺序的前 200 条（`test_extra_episodes`，跳过没通过检查的，不补位）。只有它们参加原始帧率、原始分辨率和离线规划三个评测轴。

**动作干预的替身**：每个测试集、每个时长 d，把该时长的全部窗口按 `hw("donor", split, id)` 排序成一个环；窗口 w 的替身是环上 w 之后第一个来自别的回合的窗口。Δ_act 用替身窗口的动作代替真实动作（10 §8.3）。

**离线规划**（11）：每个测试集的附加回合里 d ∈ {1, 4} 的窗口。
- 排名：窗口 w 的 63 个干扰项 = 同一测试集、同一时长、来自别的回合的规划窗口，按 `hw("distractor", split, w.id, other.id)` 排序取前 63 个。
- CEM：每条附加回合、每个 d 只用排在第 1 的那个窗口。

**原始帧率**（10 §8.4）：附加回合里 d = 4 的每个窗口，起点帧 `15 s`，从这里出发的单步和滚动都在 s5 特征上取（05 §4）。窗口编号 `f"{uuid}|fr|f{15 s}"`。

输出 `prepared/windows/<split>.json`：

```json
{
  "schema": "droid_windows_v1",
  "split": "id_test",
  "salt": "po-jepa-droid-1.0.1-windows-v1",
  "splits_sha256": "<index/splits.json 的 SHA>",
  "prepared_sha256": {"<uuid>": "<准备 JSON 的 SHA>", ...},
  "extra_episodes": ["<uuid>", ...],
  "windows": [{"id": "...", "uuid": "...", "duration": 4, "start": 7}, ...],
  "donors": {"<window id>": "<donor window id>", ...},
  "planning": {"rank": {"<window id>": ["<63 个 id>"]}, "cem": ["<window id>", ...]},
  "frame_rate": [{"id": "...", "uuid": "...", "start_frame": 105}, ...],
  "counts": {...}
}
```

窗口按（回合在划分列表里的顺序，时长，哈希顺序）排列。`val.json` 只有 `windows` 和 `counts`。

## 4. 组装 `TrainingBatch`

`make_training_batch(windows, reader, prepared, control_spec, provenance, device) -> training.TrainingBatch`，v7 `_batch` 的全部检查必须原样通过：

| 字段 | 值 |
|---|---|
| `raw_z` / `raw_target` | 特征第 s 行 / 第 s + d 行，float32 [B, 256, 1408]，放到模型设备，无梯度 |
| `controls` | 用 v7 `actions.from_physical_segments` 构造：第 j 段（j < d）`starts_seconds = s + j`、`ends_seconds = s + j + 1`（float64，乘名义周期 1.0），`u_values` = 物理动作 `action_<source>[s + j]` 转 float32 [B, J, 7]，J = 本批最大的 d，多出的位置 `segment_valid = False`，时间填 0、动作值填 NaN（v7 只读有效段）；`origin_seconds = s`；`control_spec` 用 01 §5.3 |
| `duration` | d，CPU float64 [B] |
| `tau`、`sigma` | d / 2，CPU float64 [B]（v7 时间一致性的等分半程） |
| `transfer` | `build_spatial_transfer(regular_grid(16, 16), target_height=8, target_width=8)`（和 v7 训练相同） |
| `observed_time_seconds` | s × 1.0 |
| `target_time_seconds` | (s + d) × 1.0 |
| `sample_ids` | 窗口编号的元组 |
| `provenance` | §5 |

**AC 片段**：`make_ac_clip_batch(clips, reader, prepared, source, device)` 返回 `features` float32 [16, 9, 256, 1408]、`actions` float64→float32 [16, 8, 7]（物理单位，第 k 个是 `action_<source>[s + k]`）、`states` float32 [16, 9, 7]（`state[s : s+9]`，`ac_state: true_state` 时用）。v7 的 `ac_training` 片段读取只认仿真缓存，DROID 用这个函数代替，训练步本身复用（09 §6.2）。

**验证和测试批**：冻结窗口按文件顺序每 64 个一批，组装规则同上（评测时不需要 `tau`、`sigma` 的那些检查，但仍用同一个函数）。`make_eval_batch` 另返回每个窗口起点行的真实状态 `state[s]`（float64 [B, 7]，物理单位），只给 AC 的官方输入模式用（09 §6.2）。

## 5. 来源字段

每个训练任务开始时算一次，之后每批共用（v7 `training._provenance` 的要求）：

| 键 | 值 |
|---|---|
| `evidence_role` | `"droid_training"`；假编码器特征时为 `"synthetic_development_only"`（此时 `split` 必须是 `"development_smoke"`，四个 SHA 为 None） |
| `split` | `"train"` |
| `paper_sha256` | v7 固定值 `d851778ecf4a1f160da9c681a831b3bc9333dae2b60eaeefa6090c0af44aaff1` |
| `upstream_commit` | `204698b45b3712590f06245fbfba32d3be539812` |
| `source_files_sha256` | v7 要求的 7 个源文件，加 `src/po_jepa/droid/` 下参与训练的全部 `.py` |
| `encoder_checkpoint_sha256` | 特征清单里的权重 SHA |
| `preprocess_sha256` | v7 `cache.v7_preprocess_sha256()` |
| `feature_manifest_sha256` | 05 §3 的特征总清单 SHA |
| `protocol_sha256` | 配置 SHA（01 §1） |

v7 每步写一份完整的来源 JSON；DROID 一个任务 3 万步，不这样做，改成每 100 步一行日志（09 §5），日志里记这 100 步每一步窗口编号列表的 SHA，需要时可按种子和步数重建任意一批。

## 6. 读取速度

每步读 64 × 2 行（每行 256 × 1408 × 2 字节 ≈ 0.7 MB），约 90 MB；AC 每步读 16 × 9 行，约 100 MB。特征要放在 SSD 或共享高速存储上。数据加载用 `torch.utils.data.DataLoader`，`IterableDataset` 从 `completed_steps` 开始按步号产出批（读行、核行 SHA、转 float32，都在工作进程里），2 个工作进程，预取 4 步；主进程只把张量搬到 GPU。批的内容只由步号决定，和工作进程数无关。
