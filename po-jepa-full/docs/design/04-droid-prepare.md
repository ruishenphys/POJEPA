# 04 DROID 准备

2026-11-08 · 第一版

阶段 3：从下载的 h5 和 MP4 里取出每条回合要用的全部数值，做检查；再用一个统计任务算动作尺度、冻结验证和测试窗口。每片一个 CPU 任务 `prepare/<片名>`，另加一个 `prepare/stats`。代码在 `src/po_jepa/droid/prepare.py` 和 `src/po_jepa/droid/video.py`（S05）。

## 1. 帧和步的对齐

- h5 有 T 步，MP4 有 V = T − 1 帧。沿用官方对齐：第 i 帧对第 i 步（官方 `states[indices]` 和 `vr.get_batch(indices)` 用同一组序号）。h5 的最后一步没有画面，不用。画面运动和机械臂运动的相关性在 ±1 帧内分不出差别，所以不另做对齐。
- 取帧位置 `frame_index = [0, 15, 30, …, 15(F−1)]`，F = ⌊(V − 1)/15⌋ + 1。入选回合 T ≥ 242，所以 F ≥ 17。
- 特征第 i 行、动作第 i 行、状态第 i 行都按这个序号对齐（01 §4）。

## 2. 解码

`video.decode_rgb(path) -> Iterator[(index, frame)]`：`cv2.VideoCapture(path)` 顺序 `read()` 到返回 False，每帧 `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`，得到 uint8 [720, 1280, 3]。不 seek、不读文件头帧率、不按时间取帧。每个进程 `cv2.setNumThreads(1)`。

OpenCV、PyAV、decord 在这批视频上解出的像素逐位相同（已在 5 条回合上核过），所以用 OpenCV，不新增依赖。OpenCV 的版本记进清单。

## 3. 每条回合的输出

`prepared/<片名>/<uuid>.npz`（`numpy.savez`，不压缩）：

| 数组 | dtype、形状 | 内容 |
|---|---|---|
| `frame_index` | int64 [F] | §1 |
| `state` | float64 [F, 7] | `observation/robot_state/cartesian_position` 和 `gripper_position` 在 `frame_index` 处 |
| `command` | float64 [F, 7] | `action/cartesian_position` 和 `action/gripper_position` 在 `frame_index` 处 |
| `action_realized` | float64 [F−1, 7] | `poses_to_diffs(state)`（01 §5.1） |
| `action_commanded` | float64 [F−1, 7] | `poses_to_diffs(command)` |
| `state_all` | float64 [T, 7] | 全部步的状态（原始帧率评测和 AC 的状态输入要用） |
| `command_all` | float64 [T, 7] | 全部步的指令位姿 |
| `capture_ms` | int64 [T] | `observation/timestamp/cameras/<left_serial>_estimated_capture` |
| `interval_seconds` | float64 [F−1] | `diff(capture_ms[frame_index]) / 1000` |

npz 的 zip 头带写入时间，文件字节不固定，所以不用文件 SHA。旁边写 `prepared/<片名>/<uuid>.json`（`cache._write_json`，字节固定）：

- `uuid`、`shard`、`lab`、`scene_id`、`outcome`、`steps`（T）、`video_frames`（V）、`rows`（F）、`video`（宽、高、文件头帧率，只作记录）、`skip_action_count`（`observation/timestamp/skip_action` 的和，只作记录）；
- `arrays_sha256`：每个数组 `sha256(np.ascontiguousarray(a).tobytes())`，加上 dtype 和形状；
- `frame_sha256`：`frame_index` 每一帧解码后 RGB uint8 的 SHA-256（编码阶段重新解码时逐帧核对，05 §2）；
- 对 `val`、`id_test`、`ood_test` 三片每片的前 `test_extra_episodes`（200）条，另记 `frame_sha256_s5`：第 0、5、10……帧（到 V − 1 为止）的 SHA；
- `checks`：§4 每一项的结果和数值；`passed_realized`、`passed_commanded`：两种动作源下是否通过；`reasons`：没通过的检查名；
- `source`：两个原始文件的 SHA-256（取自 `DOWNLOADED.json`）、代码提交、h5py 和 OpenCV 的版本。

读取函数 `load_prepared(path) -> PreparedEpisode` 先核 JSON 里的 `arrays_sha256`，对不上就报错。

S05 补充（实现时定，见 `docs/segments/S05.md` §4）：
- h5 结构不对、或视频帧数超过 h5 步数（取帧位置越界）时数组无法构造：只写 JSON，`rows`、`arrays_sha256`、`frame_sha256` 为 null，不写 npz（有旧的就删掉）；算不了的检查记 `passed: null`，不进 `reasons`，两种动作源都判为没通过。
- 位姿在取帧位置上有非有限值时，这个动作源的周期动作整段记 NaN（不调用 `poses_to_diffs`），`action_bounds_<源>` 判失败。
- JSON 另有 `schema`（`droid_prepared_v1`）和 `limits`（取帧和检查用到的配置值；同一数据根下不同就报错，不能混用两套口径）；`source` 另记 numpy、scipy 的版本。
- 重跑时跳过的条件：JSON 在、记的两个源文件 SHA 和 `DOWNLOADED.json` 相同、片名相同、有数组时 npz 在、该有 `frame_sha256_s5` 时有。原始文件已删（`RAW_DELETED.json`）而输出缺失或过期：报错。

## 4. 检查

| 检查 | 条件 |
|---|---|
| `h5_structure` | 上表用到的 6 个数据集（状态位姿、状态夹爪、指令位姿、指令夹爪、左相机 `estimated_capture`、`skip_action`；S05 核对：原文写 7 个是笔误）都在，第一维都是 T（等于元数据的 `trajectory_length`），状态和指令位姿是 [T, 6]，夹爪、时间戳、`skip_action` 是 [T]；位姿和夹爪是浮点、时间戳是有符号整数、`skip_action` 是 bool 或整数；文件打不开也算这一项失败 |
| `video_frames` | V = T − 1 |
| `video_size` | 每一帧都是 720×1280×3 |
| `finite` | 状态和指令位姿全部有限；状态的夹爪在 [0, 1] 内 |
| `timestamps` | `capture_ms` 严格递增；每个 `interval_seconds` 在 [`min_interval_seconds`, `max_interval_seconds`]（[0.8, 1.6]）内 |
| `action_bounds_realized` / `action_bounds_commanded` | 该动作源的每个周期动作在物理界内（位置 1.0 m、旋转 3.2 rad、夹爪 1.0） |

`passed_<源>` = 前五项都通过且对应的 `action_bounds_<源>` 通过。检查失败的回合照样写输出文件（便于查看），但下游一律跳过：训练集不用它，验证和测试窗口不含它（§6）。

5 条样本回合上实测：每个周期 1.05–1.10 秒，时间戳严格递增，最大位移 0.13 m、转角 0.82 rad、夹爪 0.89，都远在界内。

## 5. 片任务

`prepare/<片名>`：对这片在该配置下的每条回合（02 §6），没有输出或输出的源文件 SHA 和 `DOWNLOADED.json` 对不上就重做，8 个进程并行。全部完成后写 `prepared/<片名>/_complete/<配置名>.json`：配置 SHA、uuid 列表、每条 JSON 的 SHA、通过和失败的条数。

## 6. 统计任务 `prepare/stats`

依赖全部 `prepare/<片名>`（核每条准备 JSON 的 SHA 和片完成记录一致）。下面 1、2 两项和配置无关（试跑和正式运行共用）：文件不存在就写，已存在就重算并要求逐字节相同，不同就报错。完成记录 `prepared/_complete/stats-<配置名>.json` 记三类文件的 SHA（S05）。

1. **动作尺度** `prepared/action_scale.json`：取完整训练顺序（含失败回合，不受 `use_failures` 影响）的前 `scale_episodes`（500）条，每种动作源只用其中 `passed_<源>` 为真的回合（通常只少几条），按 01 §5.2 算各自的三个尺度。记下用到的 uuid 列表、分位数、算法（`numpy.quantile`，`method="linear"`，float64）。三个尺度都必须 > 0。格式按 S01 的读取方（01 §5.3）：顶层 `realized`、`commanded` 两个键，各自正好是 `{position, rotation, gripper}` 三个 float；记录类字段放在别的顶层键里。这 500 条就是 `train-000` 的前 500 条，试跑正好全部准备过，所以试跑就能算出来，正式运行沿用。σ_z 的标定用同一个回合池（09 §3.1）。`use_failures: false` 的配置不重算这些文件，要求同一数据根下主配置的 `prepare/stats` 已完成。
2. **冻结窗口** `prepared/windows/{val,id_test,ood_test}.json`：按 06 §3 的规则生成，只含 `passed_realized` 为真的回合。
3. **统计** `prepared/stats/<配置名>.json`（S05 改：各片在不同配置下的回合列表不同，统计和配置有关，所以按配置名分文件；可重算，重跑时覆盖）：每片通过和失败的条数（失败按本配置的 `action.source`）、失败原因计数、每片 `rows` 的总数、训练片去掉的回合（`train_excluded`）、`interval_seconds` 的分位数（1%、50%、99%，所有能构造数组的回合）。任一片的失败比例超过 3%（恰好 3% 不算；Jerry 2026-11-10 定，原 1%：真实数据约 1.3% 回合视频帧数 ≠ T − 1，按 §4 跳过，见 `docs/segments/S05.md` §6）时写完统计后任务失败、不写 1、2 两项和完成记录：说明有系统性问题，要人来看，不能悄悄少掉很多回合。

`commanded` 动作源只在开关打开时用到：它的尺度已在 `action_scale.json` 里；窗口文件仍是上面那一份，评测时再跳过 `passed_commanded` 为假的回合并报告跳过的条数。
