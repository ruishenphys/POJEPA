"""S08a：generic_consistency 的式15/16预测对 POJEPA 与 losses 逐位相同，对 TokenTransformer 能前向反传，坏模型报错（08 §3）。"""
import dataclasses

import pytest
import torch
from torch import nn

from po_jepa import generic_consistency as generic
from po_jepa import losses
from po_jepa.baselines import TokenTransformer, TransformerSpec
from po_jepa.geometry import GridSpec, build_spatial_transfer, regular_grid
from po_jepa.losses import LossConfig

from test_losses import e05_real_case
from test_v4 import _controls as v4_controls, _model as v4_model, _z as v4_z

CONFIG = LossConfig(1., 1e-5, 1e-8, 1e-8, .5, .25, consistency_metric="input_scale")
PREDICTOR_MESSAGE = "model.forward must take exactly four positional arguments (raw, controls, duration, grid)"


@pytest.fixture
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pojepa_cases():
    model, raw, controls, duration, transfer = e05_real_case()
    yield "e05", model, raw.detach(), controls, duration, transfer
    model = v4_model()
    yield ("v4", model, v4_z(), v4_controls([2., 3.], [[.1, -.2], [0., .3]]),
           torch.tensor([2., 3.], dtype=torch.float64), build_spatial_transfer(regular_grid(8, 8), target_height=4, target_width=4))


def run(module, model, raw, controls, duration, transfer):
    tau = duration*.25                      # tau != sigma，这样 tau、sigma 互换能被测出来
    temporal = module.temporal_predictions(model, raw, controls, tau, duration-tau, transfer.source)
    spatial = module.spatial_predictions(model, raw, controls, duration, transfer)
    return temporal, spatial


def transformer_case():
    spec = TransformerSpec(encoder_dim=32, residual=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1703)
        model = TokenTransformer(spec)
        nn.init.normal_(model.p_out.weight, std=.02)   # 残差读出的 p_out 是零初始化，不扰动的话 p_out 以前的梯度全是 0
    raw = v4_z()
    controls = v4_controls([2., 3.], [[.1, -.2], [0., .3]])
    duration = torch.tensor([2., 3.], dtype=torch.float64)
    transfer = build_spatial_transfer(regular_grid(8, 8), target_height=4, target_width=4)
    return model, raw, controls, duration, transfer


class Scripted(nn.Module):
    """第 i 次前向返回常数场 values[i]（形状同输入）；用来造出差值溢出。"""
    encoder_dim = 32
    def __init__(self, values):
        super().__init__()
        self.values, self.calls = list(values), 0
    def forward(self, raw, controls, duration, grid):
        value = self.values[self.calls]
        self.calls += 1
        return torch.full_like(raw, value)


class NotAModule:
    """有 __call__ 和 forward，但不是 nn.Module。"""
    encoder_dim = 32
    def __call__(self, raw, controls, duration, grid):
        return self.forward(raw, controls, duration, grid)
    def forward(self, raw, controls, duration, grid):
        return raw


class ThreeArgs(nn.Module):
    encoder_dim = 32
    def forward(self, raw, controls, duration):
        return raw


class VarArgs(nn.Module):
    encoder_dim = 32
    def forward(self, *args):
        return args[0]


class Defaulted(nn.Module):
    encoder_dim = 32
    def forward(self, raw, controls, duration, grid=None):
        return raw


class KeywordOnly(nn.Module):
    encoder_dim = 32
    def forward(self, raw, controls, duration, *, grid):
        return raw


class NoEncoderDim(nn.Module):
    def forward(self, raw, controls, duration, grid):
        return raw


class BoolEncoderDim(NoEncoderDim):
    encoder_dim = True


BAD_MODELS = {
    "NotAModule": (NotAModule, "model must be an nn.Module"),
    "ThreeArgs": (ThreeArgs, PREDICTOR_MESSAGE),
    "VarArgs": (VarArgs, PREDICTOR_MESSAGE),
    "Defaulted": (Defaulted, PREDICTOR_MESSAGE),
    "KeywordOnly": (KeywordOnly, PREDICTOR_MESSAGE),
    "NoEncoderDim": (NoEncoderDim, "model must expose a positive integer encoder_dim"),
    "BoolEncoderDim": (BoolEncoderDim, "model must expose a positive integer encoder_dim"),
}


def assert_same_predictions(a, b, target_grid):
    """两份预测逐字段：张量 torch.equal、dtype 相同、requires_grad 相同；target_grid 是同一个 transfer.target 对象。"""
    assert type(a) is type(b)
    for field in dataclasses.fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, GridSpec):
            assert x is y is target_grid, field.name
            continue
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor), field.name
        assert x.dtype == y.dtype and x.requires_grad == y.requires_grad, field.name
        assert torch.equal(x, y), field.name


@pytest.mark.parametrize("index", [0, 1])
def test_pojepa_outputs_and_gradients_match_losses_bitwise(one_thread, index):
    """同一个 POJEPA（e05：projected 4×6→2×3；v4：encoder 8×8→4×4）先后用 losses 和 generic：输出、损失、梯度逐位相同。"""
    _, model, raw, controls, duration, transfer = list(pojepa_cases())[index]
    results = []
    for module in (losses, generic):
        model.zero_grad(set_to_none=True)
        temporal, spatial = run(module, model, raw, controls, duration, transfer)
        time = losses.temporal_consistency_loss(temporal, config=CONFIG)
        space = losses.spatial_consistency_loss(spatial, config=CONFIG)
        (time.mean+space.mean).backward()
        gradients = {name: None if p.grad is None else p.grad.detach().clone() for name, p in model.named_parameters()}
        results.append((temporal, spatial, time, space, gradients))
    (t0, s0, time0, space0, g0), (t1, s1, time1, space1, g1) = results
    assert type(t1) is losses.TemporalPredictions and type(s1) is losses.SpatialPredictions
    assert_same_predictions(t0, t1, transfer.target)
    assert_same_predictions(s0, s1, transfer.target)
    assert s0.target_grid is s1.target_grid is transfer.target
    assert torch.equal(time0.per_sample, time1.per_sample) and torch.equal(space0.per_sample, space1.per_sample)
    assert list(g0) == list(g1) and any(g is not None for g in g0.values())
    for name in g0:
        assert (g0[name] is None) == (g1[name] is None), name
        assert g0[name] is None or torch.equal(g0[name], g1[name]), name


def test_transformer_forward_counts_and_finite_gradients():
    """TokenTransformer：losses 拒绝它；generic 时间 3 次、空间 2 次前向；两项损失各自反传，梯度都存在、有限、平方和 > 0。"""
    model, raw, controls, duration, transfer = transformer_case()
    model.train()
    with pytest.raises(TypeError, match="^model must be a POJEPA$"):
        losses.temporal_predictions(model, raw, controls, duration/2, duration/2, transfer.source)
    sizes = []
    handle = model.register_forward_hook(lambda module, args, output: sizes.append(args[0].shape[1]))
    try:
        temporal, spatial = run(generic, model, raw, controls, duration, transfer)
    finally:
        handle.remove()
    assert sizes == [64, 64, 64, 64, 16]
    for loss in (losses.temporal_consistency_loss(temporal, config=CONFIG),
                 losses.spatial_consistency_loss(spatial, config=CONFIG)):
        model.zero_grad(set_to_none=True)
        loss.mean.backward(retain_graph=True)
        squared = torch.zeros((), dtype=torch.float64)
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            squared = squared+parameter.grad.detach().to(dtype=torch.float64).square().sum()
        assert squared > 0


@pytest.mark.parametrize("bad", list(BAD_MODELS))
@pytest.mark.parametrize("function", ["temporal", "spatial"])
def test_rejects_non_predictors(function, bad):
    """不是 nn.Module、forward 不是恰好 4 个无默认值的位置参数、encoder_dim 不是正整数（含 bool），都报 TypeError。"""
    cls, message = BAD_MODELS[bad]
    _, raw, controls, duration, transfer = transformer_case()
    model = cls()
    with pytest.raises(TypeError) as error:
        if function == "temporal":
            generic.temporal_predictions(model, raw, controls, duration/2, duration/2, transfer.source)
        else:
            generic.spatial_predictions(model, raw, controls, duration, transfer)
    assert str(error.value) == message


@pytest.mark.parametrize("case", ["t_controls", "t_grid", "t_overflow", "s_controls", "s_transfer", "s_width", "s_overflow"])
def test_copied_checks_still_raise(case):
    """从 losses 照抄来的检查原样报错（类型和文字精确相等）。"""
    model, raw, controls, duration, transfer = transformer_case()
    d = duration
    if case == "t_controls":
        kind, message = TypeError, "controls must be ActionSegments"
        call = lambda: generic.temporal_predictions(model, raw, "controls", d/2, d/2, transfer.source)
    elif case == "t_grid":
        kind, message = TypeError, "grid must be a GridSpec"
        call = lambda: generic.temporal_predictions(model, raw, controls, d/2, d/2, (8, 8))
    elif case == "t_overflow":
        kind, message = ValueError, "temporal raw float32 difference must remain finite"
        call = lambda: generic.temporal_predictions(Scripted([3e38, 0., -3e38]), raw, controls, d/2, d/2, transfer.source)
    elif case == "s_controls":
        kind, message = TypeError, "controls must be ActionSegments"
        call = lambda: generic.spatial_predictions(model, raw, "controls", d, transfer)
    elif case == "s_transfer":
        kind, message = TypeError, "transfer must be a SpatialTransfer"
        call = lambda: generic.spatial_predictions(model, raw, controls, d, transfer.target)
    elif case == "s_width":
        kind, message = ValueError, "raw_z N/E must match transfer.source and model.encoder_dim"
        call = lambda: generic.spatial_predictions(model, raw[..., :16].contiguous(), controls, d, transfer)
    else:
        # [3e38, -3e38] 会先在 2×2 平均池化里溢出（报 restricted_prediction 的有限性），测不到差值这一条。
        kind, message = ValueError, "spatial raw float32 difference must remain finite"
        call = lambda: generic.spatial_predictions(Scripted([5e37, -3.3e38]), raw, controls, d, transfer)
    with pytest.raises(kind) as error:
        call()
    assert str(error.value) == message
