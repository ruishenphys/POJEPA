"""B02 接口测试：小模型与合成文件不能作为真实预训练验收证据。"""

import importlib

import importlib.util

import json

import os

from pathlib import Path

import random

import runpy

import socket

import subprocess

import sys

from dataclasses import replace

from types import ModuleType

import numpy as np

import pytest

import torch

from torch import nn

PIN = "204698b45b3712590f06245fbfba32d3be539812"

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_vjepa.py"

def e02_real_predictor_and_controls(*, encoder_dim, batch=1):
    """虚构短控制/小K测试配置，仍运行真实谱、完整动力学、积分和两层读出。"""
    from po_jepa.actions import ActionSegments, ControlSpec
    from po_jepa.dynamics import DynamicsConfig, IntegratorConfig, ModalDynamics
    from po_jepa.model import POJEPA, ProjectionConfig
    from po_jepa.spectral import GraphConfig, GraphSpectrum

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1702)
        model = POJEPA(ProjectionConfig(encoder_dim, 64),
            GraphSpectrum(GraphConfig(16, 1703, .25, 1., 3, 1e-10, 1e-8)),
            ModalDynamics(DynamicsConfig(2, 64, 64, 32, 16, 32, 32, 64, .1, 1e-3)),
            IntegratorConfig("exponential_euler", .25, 1e-12, 1e-4, 6))
    spec = ControlSpec(2, .05, ("fictional_u", "fictional_v"), ("unit", "unit"),
        (-1., -1.), (1., 1.), (1., 1.), (0., 0.), "E02 fictional known simulation commands")
    duration = torch.full((batch,), .125, dtype=torch.float64)
    controls = ActionSegments(torch.zeros(batch, 1, dtype=torch.float64), duration[:, None].clone(),
        torch.tensor([.25, -.125], dtype=torch.float32).expand(batch, 1, 2).clone(),
        torch.ones(batch, dtype=torch.int64), torch.ones(batch, 1, dtype=torch.bool),
        torch.zeros(batch, dtype=torch.float64), duration, spec, tuple(f"e02-fictional-{b}" for b in range(batch)))
    return model, controls, duration

