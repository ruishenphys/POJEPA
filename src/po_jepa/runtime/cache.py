"""逐轨迹冻结特征缓存与窄窗口读取。

导入无 I/O；生产读取核真实 canonical 身份，不以 synthetic 测试替身回退。
"""

from dataclasses import asdict, dataclass

import argparse

import hashlib

import json

from pathlib import Path

import platform

import random

import sys

ROOT=Path(__file__).resolve().parents[3]

import numpy as np

import torch

from po_jepa import vjepa

from po_jepa.vjepa import PreprocessSpec

from po_jepa.actions import ControlSpec,ActionSegments,from_physical_segments,clip_action_segments

from po_jepa.geometry import GridSpec,regular_grid,build_spatial_transfer

from po_jepa.simulation import TrajectoryControls,sample_controls,split_for

PAPER_SHA="d851778ecf4a1f160da9c681a831b3bc9333dae2b60eaeefa6090c0af44aaff1"

CONFIG_SHA="afef31e904f74f68f027ce79bb3a617ee917ab7e81336282bf7c6765dd983ba4"

PROTOCOL_SHA="09a45ff10b5d256e8dbed279f8f5e269055877b7623195e52d889fef5da853fd"

CHECKPOINT_SHA="67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e"

PUBLICATION="https://huggingface.co/facebook/vjepa2-vitg-fpc64-256/resolve/875c192b7b704b87d1e1d99345769632dd5f739a/original/model.pth"

def file_sha(path: Path) -> str:
    """按块核原字节SHA，不把大权重或特征整体载入内存，不进行网络访问。"""
    digest=hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda:file.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()

def _json_bytes(value: dict) -> bytes:
    """确定JSON编码用于来源摘要；调用者不得把自身摘要写进同一份JSON。"""
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False,default=str)+"\n").encode("utf-8")

def _write_json(path: Path, value: dict) -> None:
    """仅为本次新建目录原子写文件；已有文件明确报错，不覆盖完成产物。"""
    if path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+".partial")
    with temp.open("xb") as file:file.write(_json_bytes(value))
    temp.replace(path)

def verify_fixed_protocol() -> None:
    """生成和消费都先核固定定义；不以旧来源哈希推定当前文件仍相同。"""
    for name,value in (("reference/main.tex",PAPER_SHA),
        ("configs/train/default.yaml",CONFIG_SHA),
        ("reference/protocol.md",PROTOCOL_SHA)):
        if file_sha(ROOT/name)!=value:raise ValueError("fixed protocol changed: "+name)

def preprocess_spec() -> PreprocessSpec:
    """固定 0–255 官方预处理，不使用 Hub 默认参数或新增数据增强。"""
    return PreprocessSpec(torch.uint8,(0.,255.),256,(1.777,1.777),(.75,1.35),
        (.485,.456,.406),(.229,.224,.225),"F01 fixed RGB; B03 official DROID transform")

def _source_hashes(names: tuple[str,...]) -> dict:
    """分别记录生成或消费实际使用的自有源码，不用新代码冒充旧生成记录。"""
    return {name:file_sha(ROOT/name) for name in names}

@dataclass(frozen=True)
class WindowInput:
    """数值输入只有raw_z/controls/duration/grid；时间和来源字段仅审计，不作模型embedding。

    raw_z:[1,256,1408]普通FP32，控制已知，时间CPU64秒/周期分离；不含隐藏q。
    """
    raw_z: torch.Tensor
    controls: ActionSegments
    duration: torch.Tensor
    grid: GridSpec
    observed_time_seconds: torch.Tensor
    image_sha256: str
    trajectory_id: int
    split: str

@dataclass(frozen=True)
class WindowTarget:
    """单独的真实未来标签；不作为Phi输入，目标时间CPU64[1]物理秒。"""
    raw_z: torch.Tensor
    target_time_seconds: torch.Tensor
    image_sha256: str

@dataclass(frozen=True)
class CacheWindow:
    """训练/评价调用者分开取inputs和target；无全轨迹、隐藏q或未来索引器句柄。"""
    inputs: WindowInput
    target: WindowTarget
    provenance: dict

_VERIFIED_CACHE={}

def _read_complete_cache(path: Path) -> tuple[np.ndarray,dict,str]:
    """v4：同一进程内复用已完整核验的轨迹，避免每读一个窗口都重算整条轨迹的SHA。

    键含三个文件的 mtime_ns/size 与固定来源SHA；任一文件被改动、出现失败标记或来源常量变化都会重新完整核验。
    """
    path=Path(path).resolve(strict=True)
    if (path/"failure.json").exists():raise ValueError("cache has retained failure marker")
    key=(str(path),PAPER_SHA,PROTOCOL_SHA,CONFIG_SHA,CHECKPOINT_SHA,
         *((s.st_mtime_ns,s.st_size) for s in ((path/n).stat() for n in ("COMPLETE.json","manifest.json","raw_z.npy"))))
    if key not in _VERIFIED_CACHE:
        _VERIFIED_CACHE[key]=_read_complete_cache_uncached(path)
    return _VERIFIED_CACHE[key]

def _read_complete_cache_uncached(path: Path) -> tuple[np.ndarray,dict,str]:
    """只读明确完成缓存；核SHA、实际shape/dtype/有限性和逐端点来源，不启用pickle。"""
    path=Path(path).resolve(strict=True)
    for name in ("COMPLETE.json","manifest.json","raw_z.npy"):
        if (path/name).resolve().parent!=path:raise ValueError("cache child path escapes trajectory directory")
    if (path/"failure.json").exists():raise ValueError("cache has retained failure marker")
    complete=json.loads((path/"COMPLETE.json").read_bytes())
    manifest_sha=file_sha(path/"manifest.json")
    if complete.get("manifest_sha256")!=manifest_sha:raise ValueError("manifest SHA mismatch")
    m=json.loads((path/"manifest.json").read_bytes());verify_fixed_protocol()
    if (m.get("schema_version")!=1 or m.get("status")!="complete" or m.get("feature_file")!="raw_z.npy"
        or m.get("shape")!=[33,256,1408] or m.get("dtype")!="float32"
        or m.get("grid")!=[16,16] or m.get("token_order")!="row_major"
        or m["split"]!=split_for(m["trajectory_id"])
        or m.get("observation_variant") not in ("canonical","diagnostic_future_rgb","diagnostic_current_rgb")):
        raise ValueError("cache schema/identity/grid mismatch")
    g=m["generation"]
    if (g["paper_sha256"]!=PAPER_SHA or g["protocol_sha256"]!=PROTOCOL_SHA
        or g["config_sha256"]!=CONFIG_SHA or g["upstream_commit"]!=vjepa.UPSTREAM_COMMIT):
        raise ValueError("cache fixed provenance mismatch")
    role=g["evidence_role"]
    if role not in ("real_frozen_encoder","synthetic_development_only"):
        raise ValueError("unknown generation evidence role")
    if role=="real_frozen_encoder":
        report=g["encoder_load_report"]
        if (g["encoder_checkpoint_sha256"]!=CHECKPOINT_SHA or report["sha256"]!=CHECKPOINT_SHA
            or report["encoder_key"]!="target_encoder" or report["strict"] is not True
            or report["missing_keys"] or report["unexpected_keys"]):raise ValueError("invalid real encoder provenance")
    elif g["encoder_checkpoint_sha256"] is not None:raise ValueError("synthetic cache cannot claim pretrained checkpoint")
    if file_sha(path/"raw_z.npy")!=m["feature_sha256"]:raise ValueError("feature SHA mismatch")
    raw=np.load(path/"raw_z.npy",allow_pickle=False,mmap_mode="r")
    if raw.dtype!=np.float32 or raw.shape!=(33,256,1408) or not np.isfinite(raw).all():
        raise ValueError("invalid actual raw Z array")
    if role=="real_frozen_encoder":
        expected=sample_controls(m["trajectory_id"]);c=m["controls"]
        if any(not np.array_equal(np.asarray(c[key]),value) for key,value in
            (("starts_ticks",expected.starts_ticks),("ends_ticks",expected.ends_ticks),("values",expected.values))):
            raise ValueError("real cache controls differ from fixed F01 random stream")
    if len(m["frames"])!=33:raise ValueError("incomplete endpoint provenance")
    for tick,f in enumerate(m["frames"]):
        if (type(f["tick"]) is not int or f["tick"]!=tick or f["time_seconds"]!=tick*.25 or f["encoding_cutoff_seconds"]!=tick*.25
            or type(f["image_sha256"]) is not str or len(f["image_sha256"])!=64
            or any(ch not in "0123456789abcdef" for ch in f["image_sha256"]) or f["crop"]!=m["frames"][0]["crop"]):
            raise ValueError("endpoint source/crop mismatch")
    expected_preprocess=hashlib.sha256(_json_bytes(dict(spec=g["preprocess"],recipe=m["frames"][0]["crop"]))).hexdigest()
    if m["preprocess_sha256"]!=expected_preprocess:raise ValueError("preprocess metadata SHA mismatch")
    return raw,m,manifest_sha

def _window_controls(m: dict,start_tick: int,total: float,device: str):
    """物理控制转换后裁剪，局部周期从0开始，保留起点物理秒原点。"""
    data=m["controls"];c=TrajectoryControls(m["trajectory_id"],np.array(data["starts_ticks"],dtype=np.int64),
        np.array(data["ends_ticks"],dtype=np.int64),np.array(data["values"],dtype=np.float64))
    c.__post_init__()
    if data["units"]!="simulation_position_per_second":raise ValueError("unknown control units")
    spec=ControlSpec(2,.25,("u_x","u_y"),("simulation_position_per_second",)*2,
        (-.5,-.5),(.5,.5),(1.,1.),(0.,0.),"reference/protocol.md")
    n=len(c.starts_ticks)
    full=from_physical_segments(starts_seconds=torch.tensor(c.starts_ticks[None]*.25,dtype=torch.float64,device="cpu"),
        ends_seconds=torch.tensor(c.ends_ticks[None]*.25,dtype=torch.float64,device="cpu"),u_values=torch.tensor(c.values[None],dtype=torch.float32,device=device),
        n_segments=torch.tensor([n],dtype=torch.int64,device="cpu"),segment_valid=torch.ones((1,n),dtype=torch.bool,device="cpu"),
        origin_seconds=torch.zeros(1,dtype=torch.float64,device="cpu"),control_spec=spec,source_ids=(f"trajectory-{m['trajectory_id']:04d}",))
    return clip_action_segments(full,start=torch.tensor([float(start_tick)],dtype=torch.float64,device="cpu"),
        end=torch.tensor([start_tick+total],dtype=torch.float64,device="cpu"))

def read_window(path: Path,*,start_tick: int,duration_cycles: float,expected_split: str,device: str) -> CacheWindow:
    """读明确ID缓存的起点/整数未来标签；不随机抽窗口，不给模型全缓存或隐藏状态。"""
    if type(start_tick) is not int or not 0<=start_tick<=20:
        raise ValueError("F01 start must be integer tick in [0,20]")
    if type(duration_cycles) not in (int,float) or duration_cycles not in (1.,2.,3.,4.,8.,12.):
        raise ValueError("supervised endpoint must use an allowed integer duration")
    raw,m,manifest_sha=_read_complete_cache(Path(path));end=int(start_tick+duration_cycles)
    if m["split"]!=expected_split:raise ValueError("cross-split cache/window request")
    with torch.inference_mode(False),torch.no_grad():
        # 显式复制离开memmap/外层inference，后续可训练投影允许用其输入反传。
        z=torch.from_numpy(np.array(raw[start_tick:start_tick+1],copy=True)).to(device).clone()
        target=torch.from_numpy(np.array(raw[end:end+1],copy=True)).to(device).clone()
        duration=torch.tensor([float(duration_cycles)],dtype=torch.float64,device="cpu")
        observed=torch.tensor([start_tick*.25],dtype=torch.float64,device="cpu")
        controls=_window_controls(m,start_tick,float(duration_cycles),device)
        inputs=WindowInput(z,controls,duration,regular_grid(16,16),observed,
            m["frames"][start_tick]["image_sha256"],m["trajectory_id"],m["split"])
        label=WindowTarget(target,torch.tensor([end*.25],dtype=torch.float64,device="cpu"),m["frames"][end]["image_sha256"])
    provenance=dict(actual_manifest_sha256=manifest_sha,generation=m["generation"],preprocess_sha256=m["preprocess_sha256"],
        observation_variant=m["observation_variant"])
    return CacheWindow(inputs,label,provenance)

def training_batch_fields(window: CacheWindow) -> dict:
    """仅组装 train 来源字段；当前消费者源码与历史生成来源分别记录。"""
    x,y=window.inputs,window.target
    if x.split!="train":raise ValueError("TrainingBatch assembly is train-only")
    if x.duration.numel()!=1 or float(x.duration[0]) not in (1.,2.,4.,8.):
        raise ValueError("heldout duration cannot enter F01 JEPA training")
    if window.provenance["observation_variant"]!="canonical":raise ValueError("diagnostic RGB caches cannot be used for training")
    real=window.provenance["generation"]["evidence_role"]=="real_frozen_encoder"
    names=tuple("src/po_jepa/"+n+".py" for n in ("actions","dynamics","geometry","losses","model","spectral","vjepa","simulation"))
    sources=_source_hashes(names+("src/po_jepa/runtime/training.py","src/po_jepa/runtime/cache.py"))
    provenance=dict(evidence_role="real_frozen_features" if real else "synthetic_development_only",
        split="train" if real else "development_smoke",paper_sha256=PAPER_SHA,upstream_commit=vjepa.UPSTREAM_COMMIT,
        source_files_sha256=sources,encoder_checkpoint_sha256=CHECKPOINT_SHA if real else None,
        preprocess_sha256=window.provenance["preprocess_sha256"] if real else None,
        feature_manifest_sha256=window.provenance["actual_manifest_sha256"] if real else None,
        protocol_sha256=PROTOCOL_SHA if real else None)
    tau=x.duration/2;sigma=x.duration/2
    if not torch.equal(x.duration,tau+sigma):raise ValueError("duration split is not exact")
    return dict(raw_z=x.raw_z,raw_target=y.raw_z,controls=x.controls,duration=x.duration,tau=tau,sigma=sigma,
        transfer=build_spatial_transfer(x.grid,target_height=8,target_width=8),observed_time_seconds=x.observed_time_seconds,
        target_time_seconds=y.target_time_seconds,sample_ids=(f"{x.trajectory_id:04d}:{float(x.observed_time_seconds[0])}",),
        provenance=provenance)

DIAGNOSTIC_PROTOCOL=dict(version="f02_signal_probe_v1",trajectory_ids=[0,1,2,3],
    repeat_encodings=3,control_vectors=[[-.5,0.],[.5,0.]],horizon_cycles=8.,
    noise_margin=10.,jepa_floor=.00001,identity_to_position_floor=.01,
    raw_atol=.00001,raw_rtol=.0001,policy="review_required_if_ambiguous_no_posthoc_threshold")

