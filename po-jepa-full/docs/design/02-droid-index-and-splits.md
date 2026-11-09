# 02 DROID 索引与划分

2026-11-08 · 第一版

阶段 1（一个 CPU 任务 `index`，要联网，约 15 分钟）：列出存储桶里全部回合的元数据和原始文件，统一字段，判定入选，生成冻结的划分文件。代码在 `src/po_jepa/droid/index.py`（S02）和 `src/po_jepa/droid/splits.py`（S03）。

本文件里的条数都是在 2026-11-08 拉下的全部 74,896 份元数据、2026-11-09 列出的原始文件上按本规则算出来的（S03b 加 `missing_raw` 后由项目文件夹 `po-jepa-s03b/doc_numbers.py` 重算），S02/S03 的测试和真实运行都要和它们一致。

## 1. 列表

用 GCS JSON API 列对象（不需要凭据）：

```
GET {data.list_api}?prefix={data.prefix}&matchGlob=**/metadata_*.json&maxResults=1000
    &fields=items(name,size,md5Hash,crc32c,updated),nextPageToken
```

按 `nextPageToken` 翻页；每页失败重试 5 次（间隔 1、2、4、8、16 秒），仍失败就报错退出。全部条目按 `name` 排序后写 `index/listing.jsonl`（每行一个对象，键排好序）。条数必须等于 `data.expected_metadata_files`（74,896），否则报错：1.0.1 版是固定发布，条数变了说明数据源变了，要人来看。

**原始文件（S03b）**：再用同一个接口列两次，`matchGlob` 分别是 `**/trajectory.h5` 和 `**/recordings/MP4/*.mp4`，`fields=items(name,size,md5Hash),nextPageToken`，翻页、重试、条目检查（名字在前缀下、大小是十进制串、MD5 是 base64 摘要）和名字不能重复都同上。当前存储桶列出 74,970 个 h5、326,151 个 MP4（含右相机、腕部相机和 stereo），条数记进清单的 `raw_listed`，不设期望值（缺文件的回合由 §4 的 `missing_raw` 剔除，条数变了会体现在 §4、§5 的验收值上）。这两份列表只用来判 §4 的 `missing_raw` 和写 `index/raw_objects.jsonl`（§7），不单独存盘。

## 2. 读元数据

对每个条目下载 `https://storage.googleapis.com/gresearch/<name>`，核对 MD5（列表里的 `md5Hash` 是 MD5 摘要的 base64）。32 个线程并行，单个文件重试规则同上。

`index/metadata.jsonl`：按列表顺序每行一份元数据，原样保留官方字段，再加三个字段 `_name`（对象名）、`_md5`、`_size`。写完后整份文件的 SHA 进清单。重跑时如果 `metadata.jsonl` 已存在且每行的 `_md5` 和新列表一致，跳过下载。

## 3. 统一字段

对每份元数据算出：

| 字段 | 规则 |
|---|---|
| `lab` | 对象名第 4 段：`robotics/droid_raw/1.0.1/<lab>/<outcome>/…`。不用元数据里的 `lab` 字段（有 55 条写成 `tri`，路径里是 `TRI`） |
| `outcome` | 对象名第 5 段，只能是 `success` 或 `failure`，否则报错 |
| `success_flag` | 元数据 `success`，必须是 bool |
| `scene_id` | 元数据 `scene_id`，必须是 int；−1 表示没有场景号 |
| `steps` | 元数据 `trajectory_length`，必须是正 int |
| `uuid` | 元数据 `uuid`，必须和对象名末尾 `metadata_<uuid>.json` 一致 |
| `folder` | 对象名去掉前缀和文件名，例如 `AUTOLab/success/2023-07-07/Fri_Jul__7_18:50:52_2023` |
| `h5` | `<lab>/<元数据 hdf5_path>`；必须以 `folder` 开头，文件名必须是 `trajectory.h5`（§1 的列表只列这个名字，否则会被误判成缺文件；S03b） |
| `left_mp4` | `<lab>/<元数据 left_mp4_path>`；必须在 `<folder>/recordings/MP4/` 下 |
| `left_serial` | `left_mp4` 的文件名去掉 `.mp4` |
| `left_is` | `ext1` 或 `ext2`（和元数据里两个外部相机序列号比对；都不是就报错） |
| `group` | `lab` 在 `data.ood_labs` 里为 `ood`，否则 `id` |

另外原样带上 `building`、`date`、`robot_serial`、`user_id`、`r2d2_version`，只作记录。

## 4. 入选

逐条判定，原因可以同时有多条，原因列表为空才入选：

| 原因 | 条件 | 全库条数 |
|---|---|---|
| `duplicate_uuid` | 同一个 uuid 在全库出现不止一次 | 202（101 对，同一时间戳分别放在 success 和 failure 文件夹，长度不同） |
| `short` | `steps < data.min_steps`（242） | 42,207 |
| `flag_mismatch` | `success_flag != (outcome == "success")` | 15 |
| `no_scene_id` | `group == "id"` 且 `scene_id < 0` | 1,368 |
| `missing_raw` | `{data.prefix}<h5>` 不在 §1 的 h5 列表里，或 `{data.prefix}<left_mp4>` 不在 MP4 列表里（S03b） | 1,200（其中只因为这一条落选的 444 条：缺左相机 MP4 426、缺 h5 13、都缺 5） |

结果：入选 32,141 条，其中留出实验室 5,133 条、同分布实验室 27,008 条。各实验室入选条数：AUTOLab 4,763、CLVR 967、GuptaLab 134、ILIAD 1,831、IPRL 2,925、IRIS 1,313、PennPAL 1,580、RAD 1,190、RAIL 862、REAL 1,989、RPL 1,564、TRI 10,572、WEIRD 2,451。

> S03b 之前（没有 `missing_raw`）入选 32,585 条；多剔除的 444 条正好是 S04 在存储桶里查出的缺文件回合（`po-jepa-s04/raw_listing_result.json`）。

> 架构文档 rev 14 写的留出实验室 5,303 条没有去掉成败标记不一致的 7 条和重复编号的 2 条；以本文件为准。

`index/episodes.jsonl`：每条回合一行（含未入选的），字段为 §3 的全部字段加 `selected`（bool）和 `reasons`（按上表顺序的列表）。按 `uuid`、再按 `folder` 排序（重复编号的两行靠 `folder` 区分）。

`index/raw_objects.jsonl`（S03b）：和 `episodes.jsonl` 同样的行数和顺序，每行 `{"uuid", "folder", "h5", "left_mp4"}`（键排好序），`h5`、`left_mp4` 是 §1 列表里这个对象的条目 `{"name", "size", "md5Hash"}`，不在列表里为 `null`（这时这条回合有 `missing_raw`）。下载（03）可以拿它核对象元数据接口给的大小和 MD5。

## 5. 划分

全部用 `h(kind, key) = sha256(f"{split_salt}|{kind}|{key}")` 的十六进制串排序，升序。只用入选的回合。

1. **场景键**是 `(lab, scene_id)`（lab 已按 §3 统一）。同一个 scene_id 出现在不止一个实验室的（统一后有 4465064685、4823049285 两个），不能当验证或测试场景，它们的回合仍可进训练。
2. **候选场景**：同分布实验室里、入选回合 ≥ `episodes_per_scene`（5）条、且不在第 1 条排除名单里的场景，共 1,038 个。按 `h("scene", f"{lab}/{scene_id}")` 排序。
3. **同分布测试**：前 `id_test_scenes`（100）个场景，每个场景的入选回合按 `h("episode", uuid)` 排序取前 5 条，共 500 条。列表顺序是“场景顺序，场景内按哈希”。
4. **验证**：接下来的 `val_scenes`（60）个场景，同样每个取 5 条，共 300 条。
5. **留出实验室测试**：按 `ood_labs` 的顺序，每个实验室的入选回合按 `h("ood", uuid)` 排序取前 `ood_quota` 条（PennPAL 167、REAL 167、RPL 166），共 500 条。
6. **训练顺序**：同分布实验室的入选回合，去掉第 3、4 步选中的 160 个场景里的全部回合（不只是被抽中的 5 条，共去掉 3,524 条），剩下 23,484 条按 `h("train", uuid)` 排序，这是**完整训练顺序**。`use_failures: false` 时在这一步之后再去掉失败回合（保持其余回合的相对顺序）；这一步不写进 `splits.json`（它的 `train_order` 始终是完整训练顺序，04 §6、09 §3.1 的回合池要用），失败回合的 uuid 记在 `train_failures`，由 `splits.train_order(splits, data)` 按开关去掉（S03 定，见 `docs/segments/S03.md` §4）。
7. 训练集是本配置训练顺序（`splits.train_order(splits, data)`）的前 `train_episodes` 条中通过准备检查的那些（04 §5）。

性质（S03 测试逐条核）：同一场景不跨训练、验证、同分布测试；任何 N 的训练集是更大 N 的前缀；划分不依赖 N，改 N 时验证集和两个测试集不变；同样输入两次输出逐字节相同。

在当前元数据上的结果（S03 的验收值）：

| 项 | 值 |
|---|---|
| 候选场景 | 1,038 |
| 第一个测试场景 | `IPRL/5445595898` |
| 第一个验证场景 | `AUTOLab/2913242782` |
| 同分布测试第一条 | `IPRL+41a1536a+2023-11-08-23h-35m-17s` |
| 留出实验室测试第一条 | `PennPAL+acda9df3+2023-07-15-17h-18m-49s` |
| 训练顺序第一条 / 第 20,000 条 | `CLVR+13759f6e+2023-05-15-22h-28m-19s` / `TRI+52ca9b6a+2024-02-06-15h-12m-51s` |
| 成功 / 失败 | 同分布测试 409 / 91；验证 246 / 54；留出实验室 430 / 70；训练前 2 万条 16,600 / 3,400 |
| 同分布测试的实验室分布 | AUTOLab 45、CLVR 5、GuptaLab 5、ILIAD 45、IPRL 70、IRIS 30、RAD 35、RAIL 30、TRI 200、WEIRD 35 |
| 各列表 uuid 用 `\n` 连接后的 SHA-256 前 16 位 | 同分布测试 `2de8807ef70ace77`；验证 `490b22363540ae95`；留出实验室 `f3caf536bff734bf`；完整训练顺序 `b9dc25a09c5fe88a` |
| 特征行数（每条 ⌊(steps−2)/15⌋+1 行） | 训练前 2 万条 696,912；前 500 条 17,931；同分布测试 18,596；留出实验室 15,592；验证 9,128 |
| 够 24 / 32 个周期长时程的回合 | 同分布测试 259 / 141；留出实验室 268 / 156 |
| 完整训练顺序 / 其中失败 / 去掉失败后 | 23,484 / 4,000 / 19,484 |

S03b 重冻前后对照：测试和验证场景（100 + 60 个）完全相同；同分布测试、验证、留出实验室测试分别保留 S03 的 494、299、490 条，其余由同一场景（留出实验室为同一实验室）里哈希排在后面的回合补上；训练前 2 万条和 S03 的前 2 万条重合 19,776 条。

## 6. 片

- 训练片 `train-000`、`train-001`……：完整训练顺序里第 1000k 到 1000k+999 条是第 k 片。一个配置用到前 ⌈N/1000⌉ 片；最后一片只取到第 N 条（试跑 N = 500 时 `train-000` 只有 500 条）。`use_failures: false` 的配置片的范围不变，每片只列它用到的回合（本配置训练顺序的前 N 条落在这片里的），用到的片因此多几片；本配置训练顺序不到 N 条就报错（`splits.shard_plan`）。
- `val`、`id_test`、`ood_test` 各自一片，内容就是划分列表。
- 片名只由训练顺序决定，和 N 无关；片在某个配置下的回合列表写进该配置的片完成记录（03 §3、04 §5、05 §3），所以试跑和正式运行可以共用一个数据根。

## 7. 输出

`index/splits.json`：

```json
{
  "schema": "droid_splits_v1",
  "salt": "po-jepa-droid-1.0.1-split-v1",
  "episodes_sha256": "<index/episodes.jsonl 的 SHA>",
  "config_keys": {"min_steps": 242, "ood_labs": [...], "ood_quota": [...], "id_test_scenes": 100,
                  "val_scenes": 60, "episodes_per_scene": 5},
  "excluded_scene_ids": [4465064685, 4823049285],
  "id_test_scenes": [["IPRL", 5445595898], ...],
  "val_scenes": [[...], ...],
  "id_test": ["<uuid>", ...],
  "val": [...],
  "ood_test": [...],
  "train_order": [...],
  "train_failures": [...],
  "counts": {"selected": 32141, "candidate_scenes": 1038, "id_test": 500, "val": 300, "ood_test": 500,
             "train_order": 23484, "train_failures": 4000}
}
```

文件用 `cache._json_bytes` 写（键排序、缩进 2）。`config_keys` 只放影响划分的键（不放 N 和 `use_failures`），所以同一数据根下的所有配置（试跑、正式运行、开关副本）的划分文件逐字节相同；`splits.load_splits` 读时核 schema、salt、`config_keys` 和本配置一致、`episodes_sha256` 和清单一致。`index/MANIFEST.json` 记五个文件（`listing.jsonl`、`metadata.jsonl`、`episodes.jsonl`、`raw_objects.jsonl`、`splits.json`）的 SHA、条数、代码提交、列表时间和两份原始文件列表的条数 `raw_listed`。S03 写的四文件清单会被新代码拒绝；升级办法同 S03.md §4 第 6 条：删掉 `index/MANIFEST.json` 重跑 `index`（重新列表，`metadata.jsonl` 复用）。

## 8. 真实存储桶核对

S02 合并前，在能联网的机器上跑一次 `tools/droid.py index --verify-sample 100`：从列表里按 `h("verify", name)` 取 100 份元数据重新下载，和 `metadata.jsonl` 里的那一行逐字段比较。另外 `episodes.jsonl` 的入选条数和 §4 的表一致。S03b 合并前同样重跑了一次（见 `docs/segments/S03b.md`）。
