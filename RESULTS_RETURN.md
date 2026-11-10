# 试跑结果回传说明（2026-11-10，替代《交接说明.md》第五节的打包命令）

只把分析要用的数据传回来，模型权重和特征都留在你本机。这个文件不在 `MANIFEST.json` 里，`git pull` 拉到它不影响正在跑的任务；不 pull、直接在 GitHub 网页上看也行。**正在跑的正式运行不用停。**

## 1. 先看一下大小（在工作目录里执行）

```bash
du -sh results-v7 logs-v7 features-v7
find results-v7 -name '*.pt' | wc -l
```

`features-v7/` 是编码好的输入特征（几十 GB），`.pt` 是模型权重，其中 V-JEPA 2-AC 基线每个约 3–4 GB。这两样都**不用传**。

## 2. 打包试跑结果（只发这一个包）

```bash
zip -r results-v7-pilot-lite.zip results-v7/pilot logs-v7 po-jepa/configs/calibration po-jepa/jobs-pilot.txt \
  -x '*.pt' '*/rows/*' '*/matched_transformer/updates/*' '*/vjepa2_ac/updates/*' '*/vjepa2_ac_small/updates/*'
nvidia-smi > nvidia-smi.txt && zip results-v7-pilot-lite.zip nvidia-smi.txt
ls -lh results-v7-pilot-lite.zip
```

预计几 GB 以内。如果超过 20 GB，先别传，把第 1 步的输出发给 Jerry。

去掉的是：全部权重（`.pt`）、逐条评测明细 `rows/`（和 `trajectories/` 内容重复）、三个基线的逐步训练记录。保留的是：各任务的 report、验证选点、评测汇总、汇总表、标定配置、日志，以及 PO-JEPA 两个条件的逐步训练记录（一致性分析要用）。

## 3. 正式运行跑完后

同样的命令，把 `results-v7/pilot` 换成 `results-v7/full`、`jobs-pilot.txt` 换成 `jobs-*.txt`，文件名改成 `results-v7-full-lite.zip`。

## 4. 本机要保留的

`features-v7/` 和 `results-v7/` 里的 `.pt` 都先别删。之后如果要做诊断，Jerry 会单独要 PO-JEPA 两个条件的权重（每个只有几 MB）。
