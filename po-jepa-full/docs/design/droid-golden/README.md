# DROID 金标准参照

5 条真实 DROID 1.0.1 回合上、用官方代码（`facebookresearch/vjepa2` 提交 `204698b45b3712590f06245fbfba32d3be539812`）算出的参照值。S01、S05、S06、S08b 的测试拿新代码的输出和这里逐位比较（14 §3）。

| 文件 | 内容 |
|---|---|
| `droid_samples_golden.json` | 每条回合的参照值（下表） |
| `make_golden.py` | 生成上面这个文件的脚本 |
| `official_transforms_204698b.py` | 上游 `src/datasets/utils/video/transforms.py` 的原样副本（逐字节相同），裁剪参照用 |
| `LICENSE-vjepa2` | 上游的 MIT 许可证；`official_transforms_204698b.py` 和 `make_golden.py` 里抄的 `poses_to_diffs` 都在它之下 |

原始视频和 h5 不放进仓库（每条几十 MB）。需要时按 uuid 从索引里查到对象名，用 03 的下载代码取回，或者直接下载：

```
https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1/<lab>/<outcome>/<日期>/<文件夹>/trajectory.h5
https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1/<lab>/<outcome>/<日期>/<文件夹>/recordings/MP4/<左外部相机序列号>.mp4
```

每条回合放一个目录，里面是 `metadata.json`（官方元数据）、`trajectory.h5` 和那个 MP4，然后运行（上游代码放在一个单独的目录里，按提交 204698b 检出）：

```
python -I make_golden.py <vjepa2 目录> droid_samples_golden.json <回合目录1> … <回合目录5>
```

## 五条回合

| uuid | h5 步数 | 取帧位置数 |
|---|---|---|
| RAIL+80edfcb1+2023-04-12-16h-21m-07s | 216 | 15 |
| TRI+938130c4+2023-08-07-17h-12m-42s | 158 | 11 |
| IPRL+7790ec0a+2023-04-21-21h-16m-34s | 288 | 20 |
| ILIAD+7ae1bcff+2023-05-12-16h-48m-11s | 1139 | 76 |
| AUTOLab+5d05c5aa+2023-07-07-09h-42m-23s | 472 | 32 |

前两条短于 242 步，不会入选实验（02 §4），只用来核对对齐、动作和预处理。

## 每条回合的字段

| 字段 | 内容 |
|---|---|
| `h5_steps`、`video_frames` | T 和 V（V = T − 1，04 §1） |
| `container_fps`、`capture_dt_ms_median`、`skip_action_count` | 只作记录（文件头 60 fps；相机时间戳间隔的中位数；h5 里 skip_action 的和） |
| `stride15_indices` | 取帧位置 0、15、30……（04 §1） |
| `stride15_states` | 这些位置上的 7 维状态（位置、欧拉角、夹爪） |
| `official_actions` | 官方 `poses_to_diffs` 的输出 |
| `official_new_pose` | 官方 `notebooks/utils/mpc_utils.compute_new_pose(状态[j], 动作[j])`，j = 0…F−2（09 §6.2）；和下一个真实状态的差小于 1e-12 |
| `decoded_frame_sha256` | OpenCV 解码后第 0、15、30、V−1 帧 RGB 的 SHA-256 |
| `official_crop_shape`、`official_crop_sha256` | 官方 `random_resized_crop` 对第 0、15 帧的输出 |
| `manual_crop_equal` | 手写切片 [0:720, 154:1126] 加双线性缩放是否和官方逐位相同（五条都是 true） |
| `normalized_sha256`、`normalized_sample` | 官方均值方差归一化后的 SHA 和三个取样值 |
| `action_abs_max` | 每个动作通道的最大绝对值 |
