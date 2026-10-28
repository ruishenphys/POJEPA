"""受控仿真的独立参考真值与 RGB 渲染。

q 为位置，时间为秒，u 为位置/秒；参数来自固定仿真协议。
参考真值积分独立于 PO-JEPA 动力学，不向预测器暴露隐藏 q。
"""
from dataclasses import dataclass
import math

import numpy as np

CONTROL_PERIOD_SECONDS = .25
TRAJECTORY_TICKS = 32


def _trajectory_id(value: int) -> None:
    """只接受正式ID 0…199；bool不是轨迹编号，不猜测字符串或越界ID。"""
    if type(value) is not int or not 0 <= value < 200:
        raise ValueError("trajectory_id must be an integer in [0,199]")


def split_for(trajectory_id: int) -> str:
    """先按完整轨迹固定train/validation/test；任何窗口继承此划分。"""
    _trajectory_id(trajectory_id)
    return "train" if trajectory_id < 160 else "validation" if trajectory_id < 180 else "test"


def _rng(trajectory_id: int, stream_id: int) -> np.random.Generator:
    """每条轨迹/用途独立PCG64，不读取或消耗NumPy全局随机状态。"""
    _trajectory_id(trajectory_id)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([1703,trajectory_id,stream_id])))


def initial_state(trajectory_id: int) -> np.ndarray:
    """初始q:[2] float64；先抽|x|再抽y，偶数ID的x为正，隐藏q不进模型。"""
    generator=_rng(trajectory_id,0)
    x=generator.uniform(.05,1.5)*(1 if trajectory_id%2==0 else -1)
    return np.array([x,generator.uniform(-1.5,1.5)],dtype=np.float64)


def preprocess_seed(trajectory_id: int) -> int:
    """stream2派生uint64种子供独立torch CPU generator；不消耗其他流。"""
    _trajectory_id(trajectory_id)
    return int(np.random.SeedSequence([1703,trajectory_id,2]).generate_state(1,dtype=np.uint64)[0])


@dataclass(frozen=True)
class TrajectoryControls:
    """整轨迹已知控制：int64[S] tick区间、float64[S,2]物理命令，无隐藏状态。

    半开段覆盖[0,32)，值不随切分缩放。数组须视为只读，使用前再次核验；
    这不是可训练参数，也不是整段位移。源ID限定轨迹和随机流。
    """
    trajectory_id: int
    starts_ticks: np.ndarray
    ends_ticks: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        """拒绝空段、漏段、非法持续长度/幅值及NaN，不用裁剪修补输入。"""
        _trajectory_id(self.trajectory_id)
        for value in (self.starts_ticks,self.ends_ticks):
            if not isinstance(value,np.ndarray) or value.dtype!=np.int64 or value.ndim!=1 or not value.size:
                raise ValueError("control ticks must be nonempty int64[S]")
        s=self.starts_ticks.size
        if (self.ends_ticks.shape!=(s,) or not isinstance(self.values,np.ndarray)
            or self.values.dtype!=np.float64 or self.values.shape!=(s,2)
            or not np.isfinite(self.values).all() or np.any(np.abs(self.values)>.5)):
            raise ValueError("commands must be finite float64[S,2] in physical bounds")
        durations=self.ends_ticks-self.starts_ticks
        if (self.starts_ticks[0]!=0 or self.ends_ticks[-1]!=32
            or not np.array_equal(self.ends_ticks[:-1],self.starts_ticks[1:])
            or np.any(durations<=0) or not np.isin(durations[:-1],[1,4,8]).all()
            or durations[-1]>8):
            raise ValueError("controls must exactly cover [0,32) with the F01 segment lengths")


def sample_controls(trajectory_id: int) -> TrajectoryControls:
    """stream1依次抽时长、零事件、非零时两幅值；只截最后段，不看状态或未来。"""
    generator=_rng(trajectory_id,1);starts=[];ends=[];values=[];tick=0
    while tick<32:  # 真正的变长控制段递推；每段消耗顺序由协议固定。
        length=int(generator.choice([1,4,8]))
        value=np.zeros(2,dtype=np.float64) if generator.random()<.2 else generator.uniform(-.5,.5,2)
        starts.append(tick);tick=min(tick+length,32);ends.append(tick);values.append(value)
    return TrajectoryControls(trajectory_id,np.array(starts,dtype=np.int64),
        np.array(ends,dtype=np.int64),np.array(values,dtype=np.float64))


def _positions(q: np.ndarray) -> None:
    """外部q必须为有限float64[...,2]且在闭域内；不自动转dtype或clamp。"""
    if (not isinstance(q,np.ndarray) or q.dtype!=np.float64 or q.ndim<1
        or q.shape[-1]!=2 or q.size==0 or not np.isfinite(q).all() or np.any(np.abs(q)>2)):
        raise ValueError("q must be finite float64[...,2] in [-2,2]^2")


def vector_field(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """协议中的两条一阶方程；q/u 为同形 float64 [...,2]，返回位置/秒。"""
    _positions(q)
    if (not isinstance(u,np.ndarray) or u.dtype!=np.float64 or u.shape!=q.shape
        or not np.isfinite(u).all() or np.any(np.abs(u)>.5)):
        raise ValueError("u must match q as finite float64[...,2] in [-.5,.5]")
    x,y=q[...,0],q[...,1]
    return np.stack((x-x**3-.25*y+u[...,0],-y-.25*x+u[...,1]),axis=-1)


def rk4_step(q: np.ndarray, u: np.ndarray, step_seconds: float) -> np.ndarray:
    """独立真值RK4一步，四stage都用同段u；输出float64同形q，不调用模型积分器。"""
    if (isinstance(step_seconds,bool) or not isinstance(step_seconds,(int,float))
        or not math.isfinite(step_seconds) or step_seconds<=0):
        raise ValueError("RK4 physical step must be a finite positive real")
    h=step_seconds
    k1=vector_field(q,u);k2=vector_field(q+h*k1/2,u)
    k3=vector_field(q+h*k2/2,u);k4=vector_field(q+h*k3,u)
    result=q+h*(k1+2*k2+2*k3+k4)/6
    _positions(result)
    return result


def reference_trajectory(trajectory_id: int, *, controls: TrajectoryControls | None=None,
                         steps_per_tick: int=128) -> np.ndarray:
    """返回33个整数tick端点[33,2] float64；参考128/半步256，秒钟不累计浮点加法。

    RK4是递推，保留时间循环；每tick内部都使用当前控制，跨边界后才切段。
    既不生成半周期监督标签，也不暴露q给预测器；数值越界立即报错。
    """
    _trajectory_id(trajectory_id)
    if type(steps_per_tick) is not int or steps_per_tick not in (128,256):
        raise ValueError("only the registered 128/256 reference steps are allowed")
    c=sample_controls(trajectory_id) if controls is None else controls
    if not isinstance(c,TrajectoryControls) or c.trajectory_id!=trajectory_id:
        raise ValueError("control trajectory identity mismatch")
    c.__post_init__()
    q=initial_state(trajectory_id);result=[q.copy()];h=.25/steps_per_tick;segment=0
    for tick in range(32):
        while c.ends_ticks[segment]<=tick:segment+=1
        for _ in range(steps_per_tick):q=rk4_step(q,c.values[segment],h)
        result.append(q.copy())
    return np.stack(result)


def half_step_check(trajectory_id: int) -> dict:
    """逐端点/坐标保存h与h/2差、既有double组合判据；单轨迹结果不冒称全200通过。"""
    coarse=reference_trajectory(trajectory_id,steps_per_tick=128)
    fine=reference_trajectory(trajectory_id,steps_per_tick=256)
    error=np.abs(coarse-fine);allowed=1e-10+1e-8*np.abs(fine)
    passed=error<=allowed
    return dict(trajectory_id=trajectory_id,passed=bool(passed.all()),
        reference_steps_per_tick=128,half_steps_per_tick=256,
        reference_states=coarse.tolist(),half_step_states=fine.tolist(),atol=1e-10,rtol=1e-8,
        absolute_error=error.tolist(),allowed_error=allowed.tolist(),
        coordinate_pass=passed.tolist(),failed_coordinates=np.argwhere(~passed).tolist())


def render_positions(q: np.ndarray) -> np.ndarray:
    """q:[L,2] float64→[L,256,256,3] uint8；向量化固定像素中心/网格/亮斑及ties-even。

    隐藏q仅在此用于画图；输出不带ID、时间、状态叠字或随机外观。
    调用者按轨迹/小块调用，避免为200条同时制造大中间数组。
    """
    _positions(q)
    if q.ndim!=2:raise ValueError("render expects float64[L,2]")
    row,col=np.meshgrid(np.arange(256),np.arange(256),indexing="ij")
    background=np.where(((row%32==0)|(col%32==0))[...,None],
                        np.array([36.,40.,44.]),np.array([24.,28.,32.]))
    center_col=128+48*q[:,0,None,None];center_row=128-48*q[:,1,None,None]
    weight=np.exp(-((col+.5-center_col)**2+(row+.5-center_row)**2)/128)
    return np.rint(background[None]+(np.array([230.,150.,50.])-background)[None]*weight[...,None]).astype(np.uint8)
