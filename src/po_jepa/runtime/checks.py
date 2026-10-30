"""完整更新诊断与100次更新/400窗口验证成本检查。

五次 Phi、式14—17与 AdamW 共同复用训练实现。显式调用才读缓存或
构造模型，目标设备须符合协议；资源试跑状态不保存为正式 checkpoint。
"""

from contextlib import ExitStack

from dataclasses import asdict, fields

from pathlib import Path

import argparse

import csv

import hashlib

import json

import os

import platform

import shutil

import statistics

import subprocess

import sys

import time

import traceback

from unittest.mock import patch

import torch

import yaml

from . import data as data
from . import training as training
from po_jepa.dynamics import DynamicsConfig, IntegratorConfig, ModalDynamics

from po_jepa.losses import LossConfig

from po_jepa.metrics import future_error

from po_jepa.model import POJEPA, ProjectionConfig

from po_jepa.spectral import GraphConfig, GraphSpectrum

ROOT = Path(__file__).resolve().parents[3]

def load_protocol() -> dict:
    """核原稿与固定协议原字节，缺项或改动报错，不补默认研究参数。"""
    data.cache.verify_fixed_protocol()
    base = yaml.safe_load((ROOT/"configs/train/default.yaml").read_text(encoding="utf-8"))
    overlay = yaml.safe_load((ROOT/"configs/train/v4.yaml").read_text(encoding="utf-8"))
    return _apply_seed(_merge_v4(base, {k: v for k, v in overlay.items() if k not in ("schema_version", "protocol_id")}))

SEED_ENV = "PO_JEPA_SEED"

def _apply_seed(config: dict) -> dict:
    """v6 多种子：环境变量 PO_JEPA_SEED 只能取 default.yaml 预登记的 1703 或 later_robustness_seeds。

    同时改模型初始化种子与训练查询序列种子（window_seed 平移同样的量）；验证/测试窗口固定不变。
    未设置时与 v5 完全相同。
    """
    value = os.environ.get(SEED_ENV)
    if value is None:
        return config
    training = config["training"]
    base_seed = training["initial_model_seed"]
    allowed = [base_seed, *training["later_robustness_seeds"]]
    if not value.isdigit() or int(value) not in allowed:
        raise ValueError(f"{SEED_ENV} must be one of the preregistered seeds {allowed}")
    seed = int(value)
    return dict(config, training=dict(training, initial_model_seed=seed),
                randomness=dict(config["randomness"], window_seed=config["randomness"]["window_seed"]+seed-base_seed))

def model_seed() -> int:
    """当前运行的模型初始化种子（默认 1703），写进登记、评价行和汇总。"""
    return load_protocol()["training"]["initial_model_seed"]

def _merge_v4(base: dict, overlay: dict) -> dict:
    """v4 覆盖层深度合并：字典逐键递归，其他值整体替换；不修改输入。"""
    merged = dict(base)
    for key, value in overlay.items():
        merged[key] = _merge_v4(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return merged

def build_full_model(config: dict, device: torch.device) -> POJEPA:
    """固定协议构造 1408→64 完整预测器，CPU 初始化 RNG 局部隔离。"""
    if config != load_protocol():
        raise ValueError("F03 model requires the complete unchanged F01 protocol")
    m = config["model"]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["training"]["initial_model_seed"])
        model = POJEPA(ProjectionConfig(**m["projection"]), GraphSpectrum(GraphConfig(**m["graph"])),
            ModalDynamics(DynamicsConfig(**m["dynamics"])), IntegratorConfig(**m["integrator"]))
    return model.to(device)

def make_query_sequence() -> list:
    """固定100×4查询；独立CPU generator2703，逐行抽轨迹→起点→训练时长。"""
    cfg = load_protocol()
    generator = torch.Generator(device="cpu").manual_seed(cfg["randomness"]["window_seed"])
    durations = cfg["queries"]["jepa_duration_cycles"]
    sequence = []
    for _ in range(100):
        rows = []
        for _ in range(cfg["training"]["microbatch"]):
            tid = int(torch.randint(160, (), generator=generator))
            start = int(torch.randint(21, (), generator=generator))
            duration = durations[int(torch.randint(len(durations), (), generator=generator))]
            rows.append(data.WindowQuery(tid, start, float(duration)))
        sequence.append(rows)
    return sequence

def validate_target_device(record: dict) -> int:
    """核设备类别、物理GPU身份和MIG；容量下界用标称十进制GB，原始容量另记。

    这只是硬件身份门槛，不能替代完整训练步的实际峰值显存证明。
    0须显式Disabled；unknown/Enabled/MIG设备不放行，无CPU或小模型回退。
    """
    name = record["name"]
    if name == "NVIDIA GeForce RTX 4090":
        capacity = 24
        if record["mig_mode"].lower() not in ("[n/a]", "n/a", "disabled", "not supported"):
            raise ValueError("4090 MIG/partition identity is not resolved")
    elif "RTX PRO 6000 Blackwell" in name:
        # 与一轮实际运行机一致（RTX PRO 6000 Blackwell 96GB）。
        capacity = 96
        if record["mig_mode"].lower() not in ("[n/a]", "n/a", "disabled", "not supported"):
            raise ValueError("RTX PRO 6000 MIG/partition identity is not resolved")
    elif "A100" in name and any(label in name for label in ("40GB", "80GB")):
        capacity = 80 if "80GB" in name else 40
        if record["mig_mode"].lower() != "disabled":
            raise ValueError("F03 requires a full A100 with MIG explicitly Disabled")
    else:
        raise ValueError("F03 target requires full RTX4090 or full A100; no smaller-device substitute")
    if (not record["uuid"].startswith("GPU-") or record["memory_total_bytes"] < capacity*10**9
        or record["smi_total_mib"]*1024**2 < capacity*10**9):
        raise ValueError("physical GPU identity/capacity does not match the full target")
    return capacity

def device_metadata(device: torch.device, output: Path) -> dict:
    """按Torch实际UUID匹配nvidia-smi，避免可见GPU重排时把别的卡当目标。"""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("real F03 checks require an explicit available CUDA target")
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise ValueError("nvidia-smi is required to verify GPU UUID/capacity/MIG")
    args = [executable, "--query-gpu=uuid,name,memory.total,memory.free,mig.mode.current,driver_version", "--format=csv,noheader,nounits"]
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    (output/"gpu-inventory-stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output/"gpu-inventory-stderr.txt").write_text(result.stderr, encoding="utf-8")
    data.cache._write_json(output/"gpu-inventory-command.json", dict(argv=args, exit_code=result.returncode))
    if result.returncode:
        raise RuntimeError("nvidia-smi failed; original streams retained")
    props = torch.cuda.get_device_properties(device)
    uuid = "GPU-"+str(props.uuid).removeprefix("GPU-")
    rows = [[part.strip() for part in row] for row in csv.reader(result.stdout.splitlines())]
    matches = [row for row in rows if len(row) == 6 and row[0] == uuid]
    if len(matches) != 1:
        raise ValueError("CUDA UUID cannot be uniquely matched to a complete physical GPU")
    row = matches[0]
    free, total = torch.cuda.mem_get_info(device)
    record = dict(device=str(device), name=props.name, uuid=uuid, memory_total_bytes=props.total_memory,
        torch_free_bytes=free, torch_total_bytes=total, smi_name=row[1], smi_total_mib=float(row[2]),
        smi_free_mib=float(row[3]), mig_mode=row[4], driver_version=row[5],
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if record["name"] != record["smi_name"]:
        raise ValueError("Torch/nvidia-smi target names disagree")
    record["nominal_capacity_gb"] = validate_target_device(record)
    return record

def tensor_stats(value: torch.Tensor) -> dict:
    """诊断只留标量/形状；复制到CPU后立即归约，不保留训练图或完整激活。"""
    cpu = value.detach().to("cpu")
    if not torch.isfinite(cpu).all():
        raise ValueError("nonfinite diagnostic tensor; stop without retry")
    numbers = cpu.double()
    return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
        requires_grad=value.requires_grad, has_grad_fn=value.grad_fn is not None,
        finite=True, mean=numbers.mean().item(), std=numbers.std(unbiased=False).item(),
        frobenius=numbers.square().sum().sqrt().item())

class TrainingObserver:
    """只用于独立诊断更新的可移除观察器；不改变返回值或增加backward次数。"""
    def __init__(self, model):
        """保存原方法引用；所有记录只含普通数值，不跨更新持有计算图。"""
        self.model = model
        self.original_query = model.prepare_query
        self.original_nonlinear = model.dynamics.nonlinear
        self.stack = ExitStack()
        self.queries, self.predictions, self.loss_routes = [], [], {}
        self.nonlinear_calls = 0
        self.first_nonlinear = None
        self.feedback_gradient = None

    def _query(self, raw, grid):
        """记录真实同次X/C/g/A/B及两偏置；第三次Phi的raw反馈用一次反传hook核对。"""
        q = self.original_query(raw, grid)
        self.queries.append(dict(raw=tensor_stats(raw), projected=tensor_stats(q.projected),
            coefficients=tensor_stats(q.initial_coefficients), context=tensor_stats(q.dynamics.pooled),
            eigenvalues_cpu=tensor_stats(q.spectrum.eigenvalues_cpu), basis=tensor_stats(q.spectrum.basis),
            A=tensor_stats(q.dynamics.coefficients.a_blocks), B=tensor_stats(q.dynamics.coefficients.b_matrix),
            spectral_bias=tensor_stats(q.dynamics.spectral_bias), spatial_bias=tensor_stats(q.dynamics.spatial_bias)))
        if len(self.queries) == 3 and raw.requires_grad:
            self.stack.callback(raw.register_hook(self._feedback).remove)
        return q

    def _feedback(self, gradient):
        """只读取联合backward产生的中间反馈梯度，不累计非叶.grad。"""
        self.feedback_gradient = tensor_stats(gradient)

    def _nonlinear(self, *args, **kwargs):
        """统计N的真实调用；仅第一次保存内容分数、两偏置、M_a和总残差标量。"""
        result = self.original_nonlinear(*args, **kwargs)
        self.nonlinear_calls += 1
        if self.first_nonlinear is None:
            self.first_nonlinear = {name: tensor_stats(value) for name, value in dict(
                content_score=result.content.scores, spectral_bias=result.spectral_scores,
                spatial_bias=result.spatial_scores, M_a=result.action_residual, N=result.nonlinear).items()}
        return result

    def _prediction(self, owner, inputs, output):
        """记录实际raw读出（五次Phi，残差读出时每次两次），不另跑预测，也不保存Tensor。"""
        self.predictions.append(tensor_stats(output))

    def _loss(self, name, original):
        """记录三项mean到总损失的真实梯度；不用额外图、retain_graph或分次反传。"""
        def wrapped(*args, **kwargs):
            """保留原损失对象，只增加不修改梯度的观察hook。"""
            result = original(*args, **kwargs)
            self.loss_routes[name] = dict(requires_grad=result.mean.requires_grad, backward_seen=False)
            if result.mean.requires_grad:
                def observe(gradient):
                    """返回None保留原梯度；只存标量，不累计非叶.grad。"""
                    self.loss_routes[name].update(backward_seen=True, gradient=tensor_stats(gradient))
                self.stack.callback(result.mean.register_hook(observe).remove)
            return result
        return wrapped

    def __enter__(self):
        """只在本次诊断作用域替换观察包装；离开后恢复原方法及全部hooks。"""
        self.stack.enter_context(patch.object(self.model, "prepare_query", self._query))
        self.stack.enter_context(patch.object(self.model.dynamics, "nonlinear", self._nonlinear))
        self.stack.callback(self.model.p_out.register_forward_hook(self._prediction).remove)
        for name, symbol in (("jepa", "jepa_distance"), ("time", "temporal_consistency_loss"), ("space", "spatial_consistency_loss")):
            self.stack.enter_context(patch.object(training, symbol, self._loss(name, getattr(training, symbol))))
        return self

    def __exit__(self, *error):
        """无论成功失败都移除观察器；不抑制原异常，不回滚模型或重试。"""
        return self.stack.__exit__(*error)

def readouts_per_phi(model) -> int:
    """v4 残差读出每次Phi调用P_out两次（P_out(ΨC_τ)与P_out(ΨC_0)），直接读出一次。"""
    return 2 if getattr(model, "readout", "direct") == "residual" else 1

def diagnostic_step(model, optimizer, batch, config) -> dict:
    """一次真实联合更新；诊断快照成本不计作正常更新耗时。"""
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    buffers = {name: b.detach().cpu().clone() for name, b in model.named_buffers()}
    with TrainingObserver(model) as observer:
        result = training.train_step(model, optimizer, batch, loss_config=config, completed_steps=0)
    if len(observer.queries) != 5 or len(observer.predictions) != 5*readouts_per_phi(model) or not observer.nonlinear_calls:
        raise ValueError("complete five-Phi/N routes were not observed")
    if observer.feedback_gradient is None or not all(v["backward_seen"] for v in observer.loss_routes.values()):
        raise ValueError("required feedback/loss gradient route missing")
    graph = model.spectrum.config
    fine = graph.max_modes if graph.fixed_modes else min(graph.max_modes, batch.raw_z.shape[1])
    coarse = graph.max_modes if graph.fixed_modes else min(graph.max_modes, batch.transfer.target.height*batch.transfer.target.width)
    if [q["basis"]["shape"][-1] for q in observer.queries] != [fine]*4+[coarse]:
        raise ValueError("full fine/coarse spectral modes were not used")
    if any(q["basis"]["requires_grad"] or q["eigenvalues_cpu"]["requires_grad"] for q in observer.queries):
        raise ValueError("entire eig must remain stopped")
    parameters = {name: dict(gradient=tensor_stats(p.grad), update=tensor_stats(p.detach().cpu()-before[name]))
                  for name, p in model.named_parameters()}
    if any(b.requires_grad or b.grad is not None or not torch.equal(b.detach().cpu(), buffers[name])
           for name, b in model.named_buffers()):
        raise ValueError("fixed predictor buffers changed or acquired gradients")
    return dict(evidence_role=batch.provenance["evidence_role"], phi_calls=5,
        query_records=observer.queries, raw_predictions=observer.predictions,
        raw_start=tensor_stats(batch.raw_z), raw_target=tensor_stats(batch.raw_target),
        first_nonlinear=observer.first_nonlinear, nonlinear_calls=observer.nonlinear_calls,
        feedback_gradient=observer.feedback_gradient, loss_routes=observer.loss_routes,
        parameters=parameters, fixed_buffers_unchanged=True, losses=result.loss_values,
        completed_updates=result.completed_steps, diagnostic_overhead_included=True,
        future_jepa_distance_mean=result.loss_values["jepa"], extra_backward_or_parameter_loss_decomposition=False)

def _memory(device) -> dict:
    """调用前须同步；allocated/reserved峰值都按本次更新重置后的统计报告。"""
    return dict(peak_allocated=torch.cuda.max_memory_allocated(device), peak_reserved=torch.cuda.max_memory_reserved(device),
                allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device))

def _parameter_identity(model) -> dict:
    """validation前后核参数/已有梯度原字节，避免把评价变成隐式更新或清零。"""
    def digest(value):
        """None和零梯度分开记录，内存字节只用于比较，不写参数文件。"""
        return None if value is None else hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return {name: dict(parameter=digest(p), gradient=digest(p.grad)) for name, p in model.named_parameters()}

def validation_cost(model, features, output, device, config) -> dict:
    """eval/no-grad下处理20×5×4=400窗口，只计既定future_jepa_distance和成本。"""
    queries = [data.WindowQuery(tid, start, duration) for tid in range(160, 180)
               for start in (0, 5, 10, 15, 20) for duration in (1., 2., 4., 8.)]
    modes = {module: module.training for module in model.modules()}
    identity = _parameter_identity(model)
    total, count = 0., 0
    (output/"validation").mkdir()
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    try:
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(queries), 4):
                rows = queries[offset:offset+4]
                windows = data.load_windows(features, rows, split="validation", device=str(device))
                for window in windows:
                    data._window_boundary(window, windows[0])
                metric = future_error(model, torch.cat([w.inputs.raw_z for w in windows]),
                    data.stack_controls([w.inputs.controls for w in windows]),
                    torch.cat([w.inputs.duration for w in windows]), windows[0].inputs.grid,
                    torch.cat([w.target.raw_z for w in windows]), loss_config=config)
                if metric.name != "future_jepa_distance" or not torch.isfinite(metric.values).all():
                    raise ValueError("invalid validation metric")
                values = metric.values.detach().cpu().double()
                total += values.sum().item()
                count += values.numel()
                data.cache._write_json(output/"validation"/f"{offset:04d}.json",
                    dict(queries=[asdict(q) for q in rows], values=values.tolist(), metric=metric.name))
    finally:
        for module, mode in modes.items():
            module.training = mode
    torch.cuda.synchronize(device)
    if count != 400 or _parameter_identity(model) != identity:
        raise ValueError("validation count or unchanged parameter/gradient contract failed")
    return dict(windows=count, batch=4, metric="future_jepa_distance", mean=total/count,
                seconds=time.perf_counter()-start_time, includes_loading_and_validation_logging=True,
                parameters_and_gradients_unchanged=True, modes_restored=True)

