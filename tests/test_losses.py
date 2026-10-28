"""E03/T-L1：虚构端点的官方对照、独立LN/梯度及真实预测器连接。"""

from dataclasses import FrozenInstanceError, MISSING, fields, replace

import importlib

import importlib.util

import inspect

from contextlib import nullcontext

import os

from pathlib import Path

import random

import runpy

import sys

import numpy as np

import pytest

import torch

from torch import nn

def api():
    """首次RED在测试体内明确指向缺失的E03生产接口，避免收集导入错误。"""
    assert importlib.util.find_spec("po_jepa.losses") is not None, "E03 JEPA距离尚未实现"
    module = importlib.import_module("po_jepa.losses")
    assert all(hasattr(module, name) for name in ("LossConfig", "ReducedLoss", "jepa_distance"))
    return module

def configuration(**changes):
    """显式六字段；后四项只作合法性检查，不参与本步JEPA计算。"""
    values = dict(jepa_exponent=1., layer_norm_epsilon=1e-5, time_epsilon=1e-8,
                  space_epsilon=1e-8, time_weight=1e-3, space_weight=1e-3)
    values.update(changes)
    return api().LossConfig(**values)

def e04_real_case(*, device="cpu", encoder_dim=3):
    """虚构异步两样本/不同动作段数/NaN padding，真实64动态模型和短控制格。"""
    from test_vjepa import e02_real_predictor_and_controls
    from test_model import d12_controls
    from po_jepa.geometry import regular_grid

    model, _, _ = e02_real_predictor_and_controls(encoder_dim=encoder_dim, batch=2)
    model.to(device)
    rows = [[(0., .125, (.2, -.4)), (.125, .5, (-.7, .3)), (.5, 1., (.4, .6)), (1., 1.5, (.9, .8))],
            [(0., .375, (.8, .1)), (.375, 1., (-.1, .9)), (1., 1.5, (-.6, -.2))]]
    controls = d12_controls(rows, device=device)
    controls.u_values.requires_grad_()
    raw = torch.linspace(-.7, .9, 2*6*encoder_dim, dtype=torch.float32, device=device).reshape(2, 6, encoder_dim).requires_grad_()
    tau, sigma = torch.tensor([.25, .125], dtype=torch.float64), torch.tensor([.375, .625], dtype=torch.float64)
    return model, raw, controls, tau, sigma, regular_grid(2, 3)

def e05_transfer_and_independent_matrix(case):
    """独立手算R不由生产R作用单位阵生成；含池化、half-pixel及边缘延拓。"""
    from po_jepa.geometry import regular_grid, build_spatial_transfer
    from test_geometry import reference_bilinear

    if case == "pool":
        source, target = (4, 6), (2, 3)
        ry = torch.tensor([[.5, .5, 0., 0.], [0., 0., .5, .5]], dtype=torch.float64)
        rx = torch.tensor([[.5, .5, 0., 0., 0., 0.], [0., 0., .5, .5, 0., 0.], [0., 0., 0., 0., .5, .5]], dtype=torch.float64)
        matrix = torch.kron(ry, rx)
    elif case == "bilinear":
        source, target, matrix = (2, 3), (3, 2), reference_bilinear()
    elif case == "edge":
        source, target = (1, 2), (1, 4)
        matrix = torch.tensor([[1., 0.], [.75, .25], [.25, .75], [0., 1.]], dtype=torch.float64)
    elif case == "identity":
        source, target, matrix = (2, 3), (2, 3), torch.eye(6, dtype=torch.float64)
    else:
        assert case == "small"
        source, target, matrix = (1, 2), (1, 1), torch.tensor([[.5, .5]], dtype=torch.float64)
    transfer = build_spatial_transfer(regular_grid(*source), target_height=target[0], target_width=target[1])
    return transfer, matrix

def e05_real_case(*, device="cpu", encoder_dim=3):
    """虚构4×6→2×3、K20→6/r16、异步控制；不修改正式网格或加载encoder。"""
    from po_jepa.dynamics import DynamicsConfig, IntegratorConfig, ModalDynamics
    from po_jepa.model import POJEPA, ProjectionConfig
    from po_jepa.spectral import GraphConfig, GraphSpectrum
    from test_model import d12_controls

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1705)
        model = POJEPA(ProjectionConfig(encoder_dim, 64),
            GraphSpectrum(GraphConfig(16, 1703, .25, 1., 20, 1e-10, 1e-8)),
            ModalDynamics(DynamicsConfig(2, 64, 64, 32, 16, 32, 32, 64, .1, 1e-3)),
            IntegratorConfig("exponential_euler", .25, 1e-12, 1e-4, 6)).to(device)
    index = torch.arange(2*24*encoder_dim, dtype=torch.float32, device=device).reshape(2, 24, encoder_dim)
    raw = (.2*torch.sin(index/37)+.1*torch.cos(index/11)).requires_grad_()
    controls = d12_controls([[(0., .375, (.2, -.4)), (.375, 1., (-.7, .3)), (1., 1.5, (.9, .8))],
                            [(0., .125, (.8, .1)), (.125, .5, (-.3, .6)), (.5, 1., (.1, -.2)), (1., 1.5, (-.6, -.2))]], device=device)
    controls.u_values.requires_grad_()
    transfer, _ = e05_transfer_and_independent_matrix("pool")
    return model, raw, controls, torch.tensor([.625, .375], dtype=torch.float64), transfer

