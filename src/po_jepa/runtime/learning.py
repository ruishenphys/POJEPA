"""固定训练查询编排，复用完整预测器与式14—17更新。

仅安排批次、更新、验证和保存；真实入口由统一工作流检查资源与来源。
"""

import argparse

from dataclasses import asdict, fields, replace

import math

from pathlib import Path

import time

import torch

from . import checks as check

from . import data as data

from . import training as training

from po_jepa.losses import LossConfig

ROOT = Path(__file__).resolve().parents[3]

LOSS_NAMES = {"jepa", "time", "space", "weighted_time", "weighted_space", "total"}

def make_query_sequence(config: dict) -> tuple[list, torch.Tensor]:
    """CPU generator2703 逐行抽轨迹→起点→时长，不消耗模型全局 RNG。

    输出全部 2000×4 查询及预生成完成时的 ByteTensor 状态。训练按序列
    游标读取，不从该末状态再抽下一批；时长为周期，物理秒由数据接口换算。
    """
    if config != check.load_protocol():
        raise ValueError("F04 requires the complete unchanged F01 protocol")
    generator = torch.Generator(device="cpu").manual_seed(config["randomness"]["window_seed"])
    sequence = []
    for _ in range(config["training"]["updates_per_condition"]):
        rows = []
        for _ in range(config["training"]["microbatch"]):
            tid = int(torch.randint(160, (), generator=generator))
            start = int(torch.randint(21, (), generator=generator))
            durations = config["queries"]["jepa_duration_cycles"]
            duration = durations[int(torch.randint(len(durations), (), generator=generator))]
            rows.append(data.WindowQuery(tid, start, duration))
        sequence.append(rows)
    return sequence, generator.get_state().clone()

def _learning_batch(features, rows, device, output, sequence_sha):
    """复用 train-only 真实组批；新增消费者身份，不改冻结 Z 的历史生成来源。"""
    batch = data.load_training_batch(features, rows, device=str(device))
    data.write_collections(batch, output/"source-collections")
    provenance = dict(batch.provenance)
    provenance["source_files_sha256"] = dict(provenance.get("source_files_sha256", {}))
    name = "src/po_jepa/runtime/learning.py"
    provenance["source_files_sha256"][name] = data.cache.file_sha(ROOT/name)
    provenance["f04_query_sequence_sha256"] = sequence_sha
    provenance["f04_generator_state_stage"] = "after_complete_sequence_pre_generation"
    return replace(batch, provenance=provenance)

