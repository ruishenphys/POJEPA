"""S26a：补跑条件在空间采样（S25e）的原生网格上能不能评。fixed_fourier 能跑；koopman 的 16×16 基底换到 12×12、24×24
时坐标核对不过，直接报错（make_v8_jobs --stage space 因此拒绝 koopman，见 docs/segments/S26a.md §4）。"""

import pytest
import torch

from po_jepa.fourier import build_fixed_fourier
from po_jepa.koopman import build_koopman
from po_jepa.runtime import checks, space_eval
from po_jepa.runtime.addon import conditions, d4
from test_s25e_space_eval import BASE, R192, R384, active, caches, real  # noqa: F401
from test_training import e06_config


def test_fixed_fourier_runs_on_native_grids(caches, real, active):  # noqa: F811
    active(BASE)
    config = checks.load_protocol()
    model = build_fixed_fourier(config["model"], 1703, torch.device("cpu")).eval()
    for name, spec in (("r192", R192), ("r384", R384)):
        active(spec)
        w = space_eval.load_window(caches[name], 37, 10, 3, "cpu")
        with torch.no_grad():
            out = d4.direct_prediction(model, w)[0]
        side = spec.knobs.resolution//16
        assert out.shape == (1, side*side, 1408) and torch.isfinite(out).all()
        assert space_eval.window_value(model, w, e06_config()) > 0


def test_koopman_refuses_native_grids(caches, real, active):  # noqa: F811
    active(BASE)
    model = build_koopman(conditions.koopman_config(checks.load_protocol()), 1703, torch.device("cpu")).eval()
    for name, spec in (("r192", R192), ("r384", R384)):
        active(spec)
        w = space_eval.load_window(caches[name], 37, 10, 3, "cpu")
        with torch.no_grad(), pytest.raises(ValueError, match="grid coordinates must equal"):
            d4.direct_prediction(model, w)
