"""v7：当前数据集（物理系统 + 轨迹条数 + 划分）的唯一来源。

v6 及以前把"200 条、160/20/20"写死在十几处；v7 把它们集中到这里。
默认值（未设置环境变量时）就是 v6 的双势阱 200 条，因此 v6 的缓存、测试和结果都不受影响。

run.py 用 --system / --trajectories 设置环境变量 PO_JEPA_DATASET="<system>:<count>"，
子进程启动时读取一次；同一进程内不允许中途切换，避免一半代码用旧划分。
"""
from dataclasses import dataclass
import os

ENV = "PO_JEPA_DATASET"
DEFAULT_SYSTEM = "double_well"
DEFAULT_COUNT = 200
# v7 登记的系统名；仿真与渲染见 po_jepa.systems。
SYSTEMS = ("double_well", "van_der_pol", "multi_particle", "advection_diffusion", "landscape")
# 允许的轨迹条数；划分固定为 80% / 10% / 10% 的完整轨迹，按编号顺序切。
COUNTS = (40, 200, 500, 2000)


@dataclass(frozen=True)
class DatasetSpec:
    """系统名、总条数与三段划分（按编号连续切分，划分单位是完整轨迹）。"""
    system: str
    count: int
    # v7 可调数据集：规范化的调节项字符串（见 po_jepa.variants），空 = 基础场景
    variant: str = ""

    def __post_init__(self):
        if self.system not in SYSTEMS:
            raise ValueError(f"unknown system {self.system!r}; registered: {SYSTEMS}")
        if type(self.count) is not int or self.count not in COUNTS:
            raise ValueError(f"trajectory count must be one of {COUNTS}")
        from . import variants
        if variants.format(self.system, variants.parse(self.system, self.variant)) != self.variant:
            raise ValueError(f"调节项须写成规范形式：{variants.format(self.system, variants.parse(self.system, self.variant))!r}")

    @property
    def knobs(self):
        """解析后的调节项（po_jepa.variants.Variant）。"""
        from . import variants
        return variants.parse(self.system, self.variant)

    @property
    def train(self) -> int:
        return self.count*8//10

    @property
    def validation(self) -> int:
        return self.count//10

    @property
    def test(self) -> int:
        return self.count-self.train-self.validation

    def split_range(self, split: str) -> range:
        """train/validation/test 对应的连续编号区间。"""
        a, b = self.train, self.train+self.validation
        if split == "train":
            return range(0, a)
        if split == "validation":
            return range(a, b)
        if split == "test":
            return range(b, self.count)
        raise ValueError("split must be train/validation/test")

    def split_for(self, trajectory_id: int) -> str:
        self.check_id(trajectory_id)
        return ("train" if trajectory_id < self.train
                else "validation" if trajectory_id < self.train+self.validation else "test")

    def check_id(self, trajectory_id: int) -> None:
        if type(trajectory_id) is not int or not 0 <= trajectory_id < self.count:
            raise ValueError(f"trajectory_id must be an integer in [0,{self.count-1}]")

    @property
    def is_v6_default(self) -> bool:
        """v6 原始数据（双势阱 200 条）：沿用 v6 的缓存格式和来源核验。"""
        return self.system == DEFAULT_SYSTEM and self.count == DEFAULT_COUNT and not self.variant

    @property
    def scene(self) -> str:
        """场景名（系统 + 调节项），不含条数。"""
        return self.system if not self.variant else f"{self.system}~{self.variant}"

    @property
    def tag(self) -> str:
        return f"{self.scene}-{self.count}"


def parse(value: str | None) -> DatasetSpec:
    if not value:
        return DatasetSpec(DEFAULT_SYSTEM, DEFAULT_COUNT)
    scene, sep, count = value.rpartition(":")
    if not sep or not count.isdigit():
        raise ValueError(f"{ENV} must look like 'double_well:2000' or 'double_well~gain=2:2000', got {value!r}")
    system, _, knobs = scene.partition("~")
    from . import variants
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; registered: {SYSTEMS}")
    return DatasetSpec(system, int(count), variants.format(system, variants.parse(system, knobs)))


_ACTIVE = parse(os.environ.get(ENV))


def active() -> DatasetSpec:
    """当前进程的数据集；只在启动时从环境变量读取一次。"""
    return _ACTIVE


def _set_for_tests(spec: DatasetSpec) -> DatasetSpec:
    """仅供测试切换；返回旧值以便恢复。正式入口只用环境变量。"""
    global _ACTIVE
    old, _ACTIVE = _ACTIVE, spec
    return old


START_TICKS = (0, 5, 10, 15, 20)
VALIDATION_TRAJECTORIES = 20   # v6 是 20 条 × 5 起点 × 4 时长 = 400 窗口；v7 保持同样的验证成本


def validation_ids(spec: DatasetSpec | None = None) -> list[int]:
    """训练中选 checkpoint 用的验证轨迹：v6 是全部 20 条；更大数据集在验证段内等距取 20 条（固定、不看结果）。"""
    spec = active() if spec is None else spec
    ids = list(spec.split_range("validation"))
    if len(ids) <= VALIDATION_TRAJECTORIES:
        return ids
    step = len(ids)/VALIDATION_TRAJECTORIES
    return [ids[int(i*step)] for i in range(VALIDATION_TRAJECTORIES)]


def test_start_ticks(trajectory_id: int, spec: DatasetSpec | None = None) -> tuple[int, ...]:
    """测试起点：v6 默认数据每条 5 个起点；更大数据集每条 2 个、按编号轮换，五个起点总体上出现次数相同。"""
    spec = active() if spec is None else spec
    if spec.split_for(trajectory_id) != "test":
        raise ValueError("test start ticks requested for a non-test trajectory")
    if spec.test <= 20:
        return START_TICKS
    i = trajectory_id-spec.split_range("test").start
    return tuple(sorted((START_TICKS[i % 5], START_TICKS[(i+2) % 5])))


def donor_id(trajectory_id: int, spec: DatasetSpec | None = None) -> int:
    """test 内下一条轨迹（循环）。"""
    spec = active() if spec is None else spec
    r = spec.split_range("test")
    if trajectory_id not in r:
        raise ValueError("donor recipient must belong to test")
    return r.start+(trajectory_id-r.start+1) % len(r)
