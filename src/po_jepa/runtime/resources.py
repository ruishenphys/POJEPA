"""正式比较前的资源试跑；使用原更新和验证算法，丢弃试跑状态。"""
from contextlib import contextmanager
from dataclasses import asdict
import math
import random
import time
import numpy as np
import torch
from . import cache, checks, comparison, data, training


@contextmanager
def isolated_randomness(device):
    """试跑不改变正式训练的Python/NumPy/CPU/CUDA随机状态及采样顺序。"""
    python_state=random.getstate(); numpy_state=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
            yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)


def synchronize(device):
    """CUDA异步完成后才计时；CPU分支仅用于虚构输入行为验收。"""
    if device.type=='cuda': torch.cuda.synchronize(device)


def resource_check(features, output, device):
    """100次正常联合更新和400窗生产验证；不保存为正式训练checkpoint。

    输入沿用固定100×4训练查询，正常更新没有诊断快照开销。返回实际耗时/显存，
    不发明原协议没有的秒数阈值；异常、非有限损失或缺少计数均阻断正式训练。
    """
    output.mkdir(parents=True,exist_ok=False)
    model,config=comparison.build_condition('po_jepa_full',device)
    optimizer=comparison.make_optimizer(model); sequence=checks.make_query_sequence()
    if len(sequence)!=100 or any(len(rows)!=4 for rows in sequence):
        raise ValueError('resource check requires exactly 100 updates of four windows')
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    synchronize(device); started=time.perf_counter(); completed=0; trace=[]; ratio=None
    for rows in sequence:
        batch=data.load_training_batch(features,rows,device=str(device))
        result=training.train_step(model,optimizer,batch,loss_config=config,completed_steps=completed)
        if result.completed_steps!=completed+1 or any(
            value is not None and not math.isfinite(value) for value in result.loss_values.values()):
            raise ValueError('resource update count or finite losses failed')
        completed=result.completed_steps
        trace.append(dict(jepa=result.loss_values.get('jepa'),grad_max_abs=getattr(result,'grad_max_abs',None),
                          grad_norm=getattr(result,'grad_norm',None)))
        cache._write_json(output/f'update-{completed:04d}.json',dict(
            completed_updates=completed,queries=[asdict(row) for row in rows],losses=result.loss_values,**trace[-1]))
        if completed==len(sequence):ratio=readout_ratio(model,batch)
        del batch
    synchronize(device); seconds=time.perf_counter()-started
    memory=checks._memory(device) if device.type=='cuda' else None
    validation=comparison.validate_condition(model,features,output/'validation',device,config)
    if (completed!=100 or validation['windows']!=400 or not math.isfinite(seconds) or seconds<=0
        or not math.isfinite(validation['seconds']) or validation['seconds']<=0
        or not math.isfinite(validation['mean']) or validation['mean']<0):
        raise ValueError('incomplete/nonfinite resource acceptance')
    return dict(completed_updates=completed,microbatch=4,hundred_update_seconds=seconds,
        includes_loading_and_update_logging=True,diagnostic_overhead_included=False,
        training_memory=memory,validation=validation,
        combined_memory=checks._memory(device) if device.type=='cuda' else None,
        formal_training_state_reused=False,numeric_health=numeric_health(trace,ratio))


@torch.no_grad()
def readout_ratio(model,batch):
    """v4 E03：100 次更新后，在最后一个 batch 上量直接预测的改动幅度 ‖Ẑ−Z‖/‖Z‖（逐样本）。"""
    if not isinstance(model,training.POJEPA):return None
    direct=training.temporal_predictions(model,batch.raw_z,batch.controls,batch.tau,batch.sigma,batch.transfer.source).direct
    change=(direct-batch.raw_z).double().flatten(1).norm(dim=1)/batch.raw_z.double().flatten(1).norm(dim=1)
    return [float(v) for v in change]


def numeric_health(trace,ratio):
    """v4 E03：100 次试跑的数值健康摘要，只报告不阻断（阈值尚未事先登记）。

    jepa_decreased：后 10 次 JEPA 均值是否低于前 10 次；gradient_spikes：梯度范数超过中位数 100 倍的次数；
    readout_change_ratio：最后 batch 上 ‖Ẑ−Z‖/‖Z‖，明显大于 1 说明输出在放大。
    """
    jepa=[r['jepa'] for r in trace if r['jepa'] is not None]
    norms=[r['grad_norm'] for r in trace if r['grad_norm'] is not None]
    median=float(np.median(norms)) if norms else None
    first,last=(float(np.mean(jepa[:10])),float(np.mean(jepa[-10:]))) if len(jepa)>=20 else (None,None)
    return dict(jepa_first10_mean=first,jepa_last10_mean=last,
        jepa_decreased=None if first is None else last<first,
        grad_norm_median=median,grad_norm_max=max(norms) if norms else None,
        gradient_spikes=None if median is None else sum(n>100*median for n in norms),
        readout_change_ratio=ratio,report_only=True)
