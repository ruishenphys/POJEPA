"""依赖准入：只核版本，不安装、不升级，不访问GPU或研究数据。"""
import platform
import os
import torch
import numpy as np
import yaml


def versions():
    """记录实际解释器、库及Torch编译CUDA版本，区别于显卡驱动版本。"""
    return dict(python=platform.python_version(), torch=str(torch.__version__),
        cuda=torch.version.cuda, numpy=np.__version__, PyYAML=yaml.__version__)


def verify_versions():
    """按已交付固定环境核对；Python允许3.12补丁版本，其他研究依赖严格固定。"""
    actual=versions()
    # 与一轮实际运行环境（conda env patho）一致。
    expected=dict(torch='2.11.0', cuda='13.0', numpy='2.4.3', PyYAML='6.0.3')
    mismatches={key:dict(expected=value, actual=actual[key]) for key,value in expected.items()
        if (actual[key].split('+',1)[0] if key=='torch' else actual[key])!=value}
    if actual['python'].split('.')[:2]!=['3','12']:
        mismatches['python']=dict(expected='3.12.x', actual=actual['python'])
    if mismatches: raise ValueError('dependency versions differ: '+str(mismatches))
    return actual


def runtime_signature(device):
    """续训核线程、计算设置与物理GPU；相同签名不等于跨进程逐位一致保证。"""
    uuid=None if device.type!='cuda' else str(torch.cuda.get_device_properties(device).uuid)
    keys=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG',
        'CUDA_VISIBLE_DEVICES','PYTHONHASHSEED')
    return dict(versions=versions(),platform=platform.platform(),device=str(device),gpu_uuid=uuid,
        cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        default_dtype=str(torch.get_default_dtype()),matmul_precision=torch.get_float32_matmul_precision(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        environment={key:os.environ.get(key) for key in keys})
