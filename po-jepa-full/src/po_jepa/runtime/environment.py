"""依赖准入：只核版本，不安装、不升级，不访问GPU或研究数据。"""
import platform
import os
import torch
import numpy as np
import yaml


# C8（交给导师集群前）：可移植模式。默认（不设或设成 0）和以前逐字节相同的严格门槛；设成 1 时放宽三处：
# 显卡白名单（checks.validate_target_device）、环境版本（verify_versions）、续训必须同一张物理卡（resume.verify_training_source）。
# 放宽的内容都记进产物：hardware 记录的 portable_unlisted_gpu、environment.json 的 portable_version_mismatches、
# 续训时新旧两份 runtime.json（新目录和 --resume-from 指的旧目录各一份）。数学、损失、随机流和选点都不受这个开关影响。
PORTABLE_ENV='PO_JEPA_PORTABLE'
# 续训时可以不同的运行签名字段：落在哪台机器、哪张物理卡、几个 CPU 线程。逻辑设备号 device 不放过（C8-fix1，Codex P2-2）：
# checkpoint 的 _runtime 和 CUDA RNG 的 device_index 都要求逻辑号不变，所以换卡靠 CUDA_VISIBLE_DEVICES 映射、进程里始终用 cuda:0。
# CPU 线程数会改谱分解等 CPU 上算的浮点结果（spectral.py），放过它是可移植模式的有意取舍，同一组对照要用同一线程数。
PORTABLE_RESUME_KEYS=('platform','gpu_uuid','cpu_threads','interop_threads','environment.CUDA_VISIBLE_DEVICES')


def portable():
    """PO_JEPA_PORTABLE：不设、空或 0 = 严格（默认）；1 = 可移植模式。其他值直接报错，不猜。"""
    value=os.environ.get(PORTABLE_ENV,'')
    if value not in ('','0','1'):
        raise ValueError(f'{PORTABLE_ENV} must be unset, 0 or 1, got {value!r}')
    return value=='1'


def versions():
    """记录实际解释器、库及Torch编译CUDA版本，区别于显卡驱动版本。"""
    return dict(python=platform.python_version(), torch=str(torch.__version__),
        cuda=torch.version.cuda, numpy=np.__version__, PyYAML=yaml.__version__)


def verify_versions():
    """按已交付固定环境核对；Python允许3.12补丁版本，其他研究依赖严格固定。"""
    relaxed=portable()      # 先解析开关：非法值在版本全对时也要报错（C8-fix1，Codex P2-1）
    actual=versions()
    # 与一轮实际运行环境（conda env patho）一致。
    expected=dict(torch='2.11.0', cuda='13.0', numpy='2.4.3', PyYAML='6.0.3')
    mismatches={key:dict(expected=value, actual=actual[key]) for key,value in expected.items()
        if (actual[key].split('+',1)[0] if key=='torch' else actual[key])!=value}
    if actual['python'].split('.')[:2]!=['3','12']:
        mismatches['python']=dict(expected='3.12.x', actual=actual['python'])
    if mismatches:
        if not relaxed:
            raise ValueError('dependency versions differ: '+str(mismatches)
                             +f' (on another cluster set {PORTABLE_ENV}=1 to run anyway; the differences are then recorded)')
        return dict(actual,portable_version_mismatches=mismatches)
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


def runtime_differences(old,new):
    """两份 runtime_signature 逐项（嵌套键用点号）不同的字段名，排好序。

    空字典当成一个叶子（C8-fix1，Codex P3-1）：否则多出或少了一个空字典展开成零项，结构差异就看不见了。"""
    def flat(value,prefix=''):
        if isinstance(value,dict) and value:
            return {k:v for key,item in value.items() for k,v in flat(item,prefix+key+'.').items()}
        return {prefix[:-1]:value}
    a,b=flat(old),flat(new)
    return sorted(k for k in a.keys()|b.keys() if k not in a or k not in b or type(a[k]) is not type(b[k]) or a[k]!=b[k])
