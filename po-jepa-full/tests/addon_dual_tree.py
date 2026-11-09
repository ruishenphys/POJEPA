"""开发机专用：双树对照冒烟（07 §5.2；S00 段说明 §3.4）。

同一份假特征、同一组参数下，v7 树（仓库根目录，冻结在 0ac45fe）和 po-jepa-full 树各起一个子进程，
运行本树的 tools/smoke_v7_cpu.py --updates 20 --seed 1703 --jobs <6 个 v7 条件>，然后逐位比较
标定、初始权重哈希、每步训练记录、验证逐窗口值和选点、最后一步的 model_state、全部测试评测行
（含只评测重跑）和评测汇总。耗时、绝对路径和来源字段不比（源码 SHA 在 S08a 以后会合法地变化）。
另外对 smoke/ 下的全部输出做整份比较：文件集合相同，全部 JSON 和全部检查点在递归删掉 EXCLUDED_KEYS
（检查点另删 rng_state）以后相同，所以没点名的字段也不会漏掉。

只在 1 线程和固定环境变量（WORKER_ENV）下跑：v7 自己在多线程 CPU 上就不能逐位复现。
vjepa2_ac 在 CPU 上做不了官方规模，两棵树换成同一个缩小规格 REDUCED_OFFICIAL 和同一份假权重。

pytest 不收集本文件（文件名不是 test_*.py）；比较器的单元测试在 tests/test_addon_dual_tree.py。
父进程不导入 po_jepa（不能和任何一棵树绑定），只有子进程导入各自树里的 po_jepa。

用法（在 po-jepa-full 目录里；完整跑一遍约 2 小时，放后台）：
  python -B tests/addon_dual_tree.py --workdir /tmp/s00-dual-tree
  python -B tests/addon_dual_tree.py --compare <A 的输出目录> <B 的输出目录>
退出码：逐位相同为 0；有任何不同或任何一步失败为 1。
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

V7_COMMIT = "0ac45fe"
FROZEN_PATHS = ("src", "configs", "reference", "tests", "tools", "run.py", "pyproject.toml", "DATASET.md",
                "MANIFEST.json", "SOURCE_MAP.json")
REQUIRED_FILES = ("src/po_jepa/runtime/workflow.py", "tools/smoke_v7_cpu.py")
JOB_ORDER = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity", "vjepa2_ac_small", "vjepa2_ac")
REDUCED_OFFICIAL = dict(width=64, depth=2, heads=4, mlp_ratio=4.0)
THREADS = 1
WORKER_ENV = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
              "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE",
              "MKL_CBWR": "COMPATIBLE,STRICT"}
STRIPPED_ENV_PREFIXES = ("PO_JEPA_", "OMP_", "MKL_", "KMP_")
SYSTEM, TRAJECTORIES, UPDATES, SEED = "van_der_pol", 40, 20, 1703
# 和 smoke_v7_cpu.py 自己设的值相同；po_jepa.datasets 在导入时读一次，必须在导入 po_jepa 之前设
SMOKE_ENV = {"PO_JEPA_PROTOCOL": "v7", "PO_JEPA_DATASET": f"{SYSTEM}:{TRAJECTORIES}",
             "PO_JEPA_UPDATES": "2000", "PO_JEPA_SEED": str(SEED)}
AC_CHECKPOINT = Path("fake-ac")/"vjepa2-ac-reduced.pt"
UNTRAINED = "identity"
MAX_MISMATCHES = 200
TAIL_LINES = 40
POLL_SECONDS = 5            # 并行时轮询两个子进程的间隔（不超过 5 s）
KILL_GRACE_SECONDS = 30     # 一个子进程失败后，另一个收到 SIGTERM 30 s 内不退出就 SIGKILL
WORKER_SAME_FIELDS = ("threads", "torch_version", "smoke_v7_cpu_sha256", "ac_reduced_official_spec",
                      "ac_checkpoint_sha256", "env")
# 整份比较时在任何层级删掉的键：耗时；含耗时和来源的文件的 SHA；绝对路径；源码 SHA（S08a 以后会合法地变）；
# preregistration.json 的摘要（它含 consumer_sha256，所以随源码变；preregistration.json 本身照样整份比较）
EXCLUDED_KEYS = frozenset({"seconds", "update_and_observer_seconds", "gpu_memory", "checkpoint_sha256",
                           "report_sha256", "training_run", "provenance", "checkpoint_provenance", "consumer_sha256",
                           "source_files_sha256", "generator_source_sha256",
                           "training_consumer_sha256", "evaluation_source_sha256",
                           "registration_sha256", "preregistration_sha256"})
# 检查点另删 Python/NumPy/torch 的全局随机状态：训练只用显式生成器，全局状态从不播种，进程之间本来就不同
CHECKPOINT_EXCLUDED_KEYS = EXCLUDED_KEYS | {"rng_state"}
WHOLE_CALIBRATION = "smoke/calibrate/calibration.json"
UPDATE_FIELDS = ("completed_updates", "queries", "losses", "grad_max_abs", "grad_norm_before_clip", "learning_rate",
                 "raw_predictions", "raw_target", "phi_calls", "consistency_gradient")
VALIDATION_FIELDS = ("queries", "values")
SELECTION_FIELDS = ("step", "metric", "mean", "means", "windows")
TRAINING_BEST_FIELDS = ("best.step", "best.mean")
JOB_REPORT_FIELDS = ("best.step", "training", "evaluation")
ROW_FIELDS = ("query", "status", "values", "diagnostics")
CALIBRATION_DROPPED = ("v7_calibration_file", "note")
# 每一项都必须真的比过（数量 > 0），否则算不同
ITEMS = ("worker", "calibration", "complete", "initial_model_sha256", "updates", "validation_files", "selection",
         "training_best", "model_state", "evaluation_registration", "evaluation_rows", "evaluation_reports",
         "job_reports", "all_files", "all_json", "all_checkpoints")
_MISSING = object()


# ---------------------------------------------------------------- 写文件（父进程不导入 po_jepa）

def _json_bytes(value) -> bytes:
    """与 po_jepa.runtime.cache._json_bytes 相同的编码。"""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=str)
            + "\n").encode("utf-8")


def write_json(path: Path, value) -> None:
    """行为与 cache._write_json 相同：目标已存在就报错，先写 .partial 再换名。"""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name+".partial")
    with temp.open("xb") as file:
        file.write(_json_bytes(value))
    temp.replace(path)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- 前置检查和子进程环境

def _git(tree: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")    # 只读：不让 git status 顺手改写索引
    return subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True, env=env)


def preflight_checks(v7_tree: Path, new_tree: Path, workdir: Path, *, parallel: bool) -> None:
    """S00 §3.4.2 第 0 步，按顺序检查，任一条不满足就报 ValueError。"""
    if parallel:
        cores = os.cpu_count()
        if cores is None or cores < 2:
            raise ValueError(f"并行模式要求 os.cpu_count() >= 2（现在是 {cores}）；单核机器请加 --sequential")
    v7, new, work = Path(v7_tree).resolve(), Path(new_tree).resolve(), Path(workdir)
    if os.path.lexists(work):
        raise ValueError(f"--workdir 必须不存在：{work}")
    work = work.resolve()
    for tree in (v7, new):
        if work.is_relative_to(tree):
            raise ValueError(f"--workdir 不能在树里面（{tree}）：{work}")
    if v7 == new:
        raise ValueError(f"两棵树是同一个目录：{v7}")
    for tree in (v7, new):
        for name in REQUIRED_FILES:
            if not (tree/name).is_file():
                raise ValueError(f"{tree} 缺 {name}")
    if (v7/"tools/smoke_v7_cpu.py").read_bytes() != (new/"tools/smoke_v7_cpu.py").read_bytes():
        raise ValueError("两棵树的 tools/smoke_v7_cpu.py 不同（输入就不同了）")
    top = _git(v7, "rev-parse", "--show-toplevel")
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != v7:
        raise ValueError(f"v7 树不是 git 检出：{v7}")
    if _git(v7, "diff", "--quiet", V7_COMMIT, "--", *FROZEN_PATHS).returncode != 0:
        raise ValueError(f"v7 树和 {V7_COMMIT} 不同（git diff {V7_COMMIT} -- {' '.join(FROZEN_PATHS)}）")
    status = _git(v7, "status", "--porcelain", "--", *FROZEN_PATHS)
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError(f"v7 树有未提交或未跟踪的文件：{status.stdout.strip() or status.stderr.strip()}")


def worker_environment(base=None) -> dict:
    """复制当前环境，删掉 PYTHONPATH、PO_JEPA_*、OMP_*、MKL_*、KMP_*，再设 WORKER_ENV。"""
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key == "PYTHONPATH" or key.startswith(STRIPPED_ENV_PREFIXES):
            del env[key]
    env.update(WORKER_ENV)
    return env


def check_worker_runtime(environ, threads: int) -> None:
    """子进程自查：1 线程，WORKER_ENV 的变量都在、值相同（防止绕过父进程直接调 --worker）。"""
    if threads != THREADS:
        raise RuntimeError(f"torch.get_num_threads() 是 {threads}，要求 {THREADS}")
    wrong = {k: environ.get(k) for k, v in WORKER_ENV.items() if environ.get(k) != v}
    if wrong:
        raise RuntimeError(f"子进程环境和 WORKER_ENV 不同：{wrong}（请通过父进程运行）")


def bind_tree(tree: Path, path: list) -> None:
    """去掉 path[0]（本脚本所在的 tests 目录），把 <tree>/src 插到最前面。"""
    here = Path(__file__).resolve().parent
    if not path or Path(path[0] or os.curdir).resolve() != here:
        raise RuntimeError(f"sys.path[0] 应是本脚本所在目录 {here}，实际是 {path[0] if path else None}")
    del path[0]
    path.insert(0, str(Path(tree)/"src"))


def check_modules_in_tree(modules: dict, tree: Path) -> list[str]:
    """每个 po_jepa 模块的文件（命名空间包是它的目录）都在 <tree>/src 下；返回相对 <tree> 的路径。"""
    tree = Path(tree).resolve()
    src = tree/"src"
    found = []
    for name in sorted(modules):
        module = modules[name]
        origin = getattr(module, "__file__", None)
        places = [origin] if origin else list(getattr(module, "__path__", None) or [])
        if not places:
            raise RuntimeError(f"{name} 没有文件位置")
        for place in places:
            place = Path(place).resolve()
            if not place.is_relative_to(src):
                raise RuntimeError(f"{name} 不在 {src} 下：{place}")
            found.append(place.relative_to(tree).as_posix())
    return found


def _po_jepa_modules() -> dict:
    return {n: m for n, m in list(sys.modules.items()) if n == "po_jepa" or n.startswith("po_jepa.")}


def _enter_tree(tree: Path) -> None:
    """子进程（--worker、--make-ac-checkpoint）共用：在导入 po_jepa 之前定好环境、路径和线程。"""
    os.environ.update(SMOKE_ENV)
    bind_tree(tree, sys.path)
    torch.set_num_threads(THREADS)
    check_worker_runtime(os.environ, torch.get_num_threads())
    import po_jepa
    check_modules_in_tree({"po_jepa": po_jepa}, tree)


# ---------------------------------------------------------------- 子进程

def make_ac_checkpoint(path: Path, tree: Path) -> None:
    """缩小的“官方”AC 假权重（S00 §3.4.4），格式同 vjepa2-ac-vitg.pt：{"predictor": {module.backbone.<键>: 张量}}。"""
    path = Path(path).absolute()
    if os.path.lexists(path):
        raise FileExistsError(path)
    tree = Path(tree).resolve()
    _enter_tree(tree)
    from po_jepa import ac_baseline
    torch.manual_seed(0)
    model = ac_baseline.ACPredictor(ac_baseline.ACSpec(**REDUCED_OFFICIAL))
    state = {k: v for k, v in model.core.state_dict().items() if not k.startswith("action_encoder.")}
    width = REDUCED_OFFICIAL["width"]
    state["action_encoder.weight"] = torch.randn(width, 7)
    state["action_encoder.bias"] = torch.zeros(width)
    state["state_encoder.weight"] = torch.randn(width, 7)*0.02
    state["state_encoder.bias"] = torch.randn(width)*0.02
    state["extrinsics_encoder.weight"] = torch.randn(width, 6)*0.02
    state["extrinsics_encoder.bias"] = torch.zeros(width)
    payload = {"predictor": {"module.backbone."+k: v for k, v in state.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".partial")
    with temp.open("xb") as file:
        torch.save(payload, file)
        file.flush()
        os.fsync(file.fileno())
    temp.replace(path)
    print(json.dumps(dict(ok=True, path=str(path), keys=len(payload["predictor"]), sha256=_file_sha(path)),
                     ensure_ascii=False), flush=True)


def worker(tree: Path, features: Path, ac_checkpoint: Path, output: Path) -> None:
    """一棵树一个子进程（S00 §3.4.3）：真的核一次清单，打 AC 缩小补丁，调本树的 smoke_v7_cpu.main()。"""
    started = time.perf_counter()
    tree, features, ac_checkpoint, output = (Path(p).resolve() for p in (tree, features, ac_checkpoint, output))
    os.environ["PO_JEPA_AC_CHECKPOINT"] = str(ac_checkpoint)
    _enter_tree(tree)
    import po_jepa
    from po_jepa.runtime import workflow
    bundle = workflow.verify_bundle()          # 真的核：smoke 稍后才把它换成返回 "smoke"
    from po_jepa import ac_baseline
    # 必须在 smoke 之前打：comparison._ac_parameter_counts 有 lru_cache，第一次调用就定下了
    ac_baseline.OFFICIAL = ac_baseline.ACSpec(**REDUCED_OFFICIAL)
    smoke_path = tree/"tools"/"smoke_v7_cpu.py"
    spec = importlib.util.spec_from_file_location("smoke_v7_cpu", smoke_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)            # 这样它的 ROOT 就是 <tree>
    sys.argv = [str(smoke_path), "--system", SYSTEM, "--trajectories", str(TRAJECTORIES),
                "--features-root", str(features), "--output", str(output/"smoke"),
                "--updates", str(UPDATES), "--seed", str(SEED), "--jobs", ",".join(JOB_ORDER)]
    module.main()
    modules = check_modules_in_tree(_po_jepa_modules(), tree)
    from po_jepa.runtime import cache
    cache._write_json(output/"worker.json", dict(
        tree=str(tree), po_jepa_file=str(Path(po_jepa.__file__).resolve()), po_jepa_modules=modules,
        verify_bundle_sha256=bundle, smoke_v7_cpu_sha256=_file_sha(smoke_path),
        # 只取这 4 个字段：S08b 会给 ACSpec 加字段，两棵树的 asdict 那时就不同了
        ac_reduced_official_spec={k: getattr(ac_baseline.OFFICIAL, k) for k in REDUCED_OFFICIAL},
        ac_checkpoint_sha256=_file_sha(ac_checkpoint), torch_version=torch.__version__,
        threads=torch.get_num_threads(), env={k: os.environ[k] for k in WORKER_ENV}, python=sys.version,
        seconds=time.perf_counter()-started))


# ---------------------------------------------------------------- 比较器

def _not_json(value) -> str:
    """检查点里非 JSON 的值（例如 torch.dtype）：带上类型名，不同类型不会被当成相同。"""
    return f"<{type(value).__name__}> {value!r}"


def _canon(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=_not_json)


def _drop_keys(value, keys):
    """递归删掉任何层级上名字在 keys 里的字典键（保持其余键的顺序）。"""
    if isinstance(value, dict):
        return {k: _drop_keys(v, keys) for k, v in value.items() if not (isinstance(k, str) and k in keys)}
    if isinstance(value, list):
        return [_drop_keys(v, keys) for v in value]
    if isinstance(value, tuple):
        return tuple(_drop_keys(v, keys) for v in value)
    return value


def _join(field: str, key) -> str:
    return f"{field}.{key}" if field else str(key)


def _hexify(value):
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, dict):
        return {k: _hexify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hexify(v) for v in value]
    return value


def _show(value):
    """报告里的值：浮点数写成 float.hex()；太长的截断。"""
    if value is _MISSING:
        return "<missing>"
    shown = _hexify(value)
    text = _canon(shown)
    return shown if len(text) <= 2000 else text[:2000]+"…"


def _leaf_diffs(a, b, field):
    """逐层找到最小的不同之处，产出 (字段路径, A 的值, B 的值)。"""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            va, vb = a.get(key, _MISSING), b.get(key, _MISSING)
            if va is _MISSING or vb is _MISSING or _canon(va) != _canon(vb):
                yield from _leaf_diffs(va, vb, f"{field}.{key}" if field else str(key))
        return
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (va, vb) in enumerate(zip(a, b)):
            if _canon(va) != _canon(vb):
                yield from _leaf_diffs(va, vb, f"{field}[{i}]")
        return
    yield field, a, b


def _get(value, field: str):
    """按点分路径取值；中间是 None 就返回 None（identity 的 best），缺键返回 _MISSING。"""
    for key in field.split("."):
        if value is None:
            return None
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _flat_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def _scalar(value):
    return value.hex() if isinstance(value, float) else value


def _tensor_diff(ta: torch.Tensor, tb: torch.Tensor):
    """dtype、shape 和原始字节都相同返回 None，否则返回 (A 的说明, B 的说明)。"""
    if ta.dtype != tb.dtype or tuple(ta.shape) != tuple(tb.shape):
        return dict(dtype=str(ta.dtype), shape=list(ta.shape)), dict(dtype=str(tb.dtype), shape=list(tb.shape))
    ba, bb = _flat_bytes(ta), _flat_bytes(tb)
    if torch.equal(ba, bb):
        return None
    first = int((ba != bb).nonzero()[0, 0])//max(1, ta.element_size())
    fa, fb = ta.detach().cpu().reshape(-1), tb.detach().cpu().reshape(-1)
    return dict(index=first, value=_scalar(fa[first].item())), dict(index=first, value=_scalar(fb[first].item()))


def _tree_diffs(a, b, field):
    """检查点内容的递归比较：张量比 dtype/shape/字节，字典连键序一起比，其他值比规范化字符串和类型。"""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            yield field, type(a).__name__, type(b).__name__
        else:
            diff = _tensor_diff(a, b)
            if diff is not None:
                yield (field, *diff)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            yield _join(field, "<keys>"), [str(k) for k in a], [str(k) for k in b]
        for key in [k for k in a if k in b]:
            yield from _tree_diffs(a[key], b[key], _join(field, key))
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            yield field, f"{type(a).__name__}[{len(a)}]", f"{type(b).__name__}[{len(b)}]"
            return
        for i, (va, vb) in enumerate(zip(a, b)):
            yield from _tree_diffs(va, vb, f"{field}[{i}]")
        return
    if type(a) is not type(b) or _canon(a) != _canon(b):
        yield field, a if type(a) in (int, float, str, bool, type(None)) else _not_json(a), \
            b if type(b) in (int, float, str, bool, type(None)) else _not_json(b)


class _Comparison:
    def __init__(self, a: Path, b: Path):
        self.a, self.b = Path(a), Path(b)
        self.counts = {}
        self.mismatches = []
        self.total = 0

    def count(self, item, n=1):
        self.counts[item] = self.counts.get(item, 0)+n

    def differ(self, item, path, field, a, b):
        self.total += 1
        if len(self.mismatches) < MAX_MISMATCHES:
            self.mismatches.append(dict(item=item, path=path, field=field, a=_show(a), b=_show(b)))

    # -- 读文件
    @staticmethod
    def _read(path: Path, reader):
        """返回 (状态, 值)；读不了也算不同，不让比较器中途退出。"""
        if not path.is_file():
            return "<missing>", None
        try:
            return "present", reader(path)
        except Exception as error:
            return f"<unreadable: {type(error).__name__}: {error}>", None

    def pair(self, item, rel, reader=None):
        """两边都读到才返回 (A, B)；任一边缺或读不了记一处不同，返回 None。"""
        reader = reader or (lambda p: json.loads(p.read_text(encoding="utf-8")))
        (sa, va), (sb, vb) = self._read(self.a/rel, reader), self._read(self.b/rel, reader)
        if sa != "present" or sb != "present":
            self.differ(item, rel, "<file>", sa, sb)
            return None
        return va, vb

    def complete(self, rel):
        if self.pair("complete", rel) is not None:
            self.count("complete")

    def names(self, item, rel, pattern, *, dirs=False, expected=None) -> list[str]:
        """两边的文件（或子目录）名集合必须相同、非空，给了 expected 时还要等于它；返回两边共有的名字。"""
        found = []
        for root in (self.a, self.b):
            folder = root/rel
            found.append({p.name for p in folder.glob(pattern) if (p.is_dir() if dirs else p.is_file())}
                         if folder.is_dir() else None)
        na, nb = found
        if na is None or nb is None or na != nb or not na or (expected is not None and na != set(expected)):
            self.differ(item, rel, "<names>" if expected is None else f"<names> (expected {len(expected)})",
                        sorted(na) if na is not None else _MISSING, sorted(nb) if nb is not None else _MISSING)
        return sorted((na or set()) & (nb or set()))

    # -- 比值
    def values(self, item, rel, field, va, vb):
        if va is _MISSING or vb is _MISSING:
            self.differ(item, rel, field, va, vb)
        elif _canon(va) != _canon(vb):
            for sub, a, b in _leaf_diffs(va, vb, field):
                self.differ(item, rel, sub, a, b)

    def fields(self, item, rel, fields, transform=None):
        """比 JSON 文件里的这些字段（点分路径；None 表示整份）；两边都在才计数。"""
        got = self.pair(item, rel)
        if got is None:
            return None
        va, vb = got if transform is None else (transform(got[0]), transform(got[1]))
        for field in fields:
            self.values(item, rel, field or "", va if field is None else _get(va, field),
                        vb if field is None else _get(vb, field))
        self.count(item)
        return got

    def tensors(self, item, rel, key="model_state"):
        got = self.pair(item, rel, lambda p: torch.load(p, map_location="cpu", weights_only=True))
        if got is None:
            return
        sa, sb = _get(got[0], key), _get(got[1], key)
        if not isinstance(sa, dict) or not isinstance(sb, dict) or not sa:
            self.differ(item, rel, key, "dict" if isinstance(sa, dict) and sa else _MISSING,
                        "dict" if isinstance(sb, dict) and sb else _MISSING)
            return
        if list(sa) != list(sb):
            self.differ(item, rel, f"{key}.<keys>", list(sa), list(sb))
        for name in [n for n in sa if n in sb]:
            ta, tb = sa[name], sb[name]
            field = f"{key}.{name}"
            if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
                self.differ(item, rel, field, type(ta).__name__, type(tb).__name__)
            else:
                diff = _tensor_diff(ta, tb)
                if diff is not None:
                    self.differ(item, rel, field, *diff)
        self.count(item)

    # -- 各项
    def worker(self):
        got = self.pair("worker", "worker.json")
        if got is None:
            return
        wa, wb = got
        ta, tb = _get(wa, "tree"), _get(wb, "tree")
        if isinstance(ta, str) and isinstance(tb, str) and os.path.realpath(ta) == os.path.realpath(tb):
            raise ValueError(f"两份 worker.json 的 tree 相同（{ta}）：拒绝拿一棵树和自己比")
        for side, w in (("a", wa), ("b", wb)):
            tree, file, modules = _get(w, "tree"), _get(w, "po_jepa_file"), _get(w, "po_jepa_modules")
            ok = (isinstance(tree, str) and isinstance(file, str)
                  and Path(file).is_relative_to(Path(tree)/"src") and isinstance(modules, list) and modules
                  and all(isinstance(m, str) and not Path(m).is_absolute() and ".." not in Path(m).parts
                          and Path(m).parts[:1] == ("src",) for m in modules))
            if not ok:
                self.differ("worker", "worker.json", f"{side}: po_jepa_file/po_jepa_modules in own tree",
                            dict(tree=tree, po_jepa_file=file), None)
            # 子进程确实在固定环境、缩小规格下跑（不只是两边相同）
            for field, name, expected in (("env", "WORKER_ENV", WORKER_ENV),
                                          ("ac_reduced_official_spec", "REDUCED_OFFICIAL", REDUCED_OFFICIAL)):
                value = _get(w, field)
                if value is _MISSING or _canon(value) != _canon(expected):
                    self.differ("worker", "worker.json", f"{side}: {field} == {name}", value, expected)
        for field in WORKER_SAME_FIELDS:
            self.values("worker", "worker.json", field, _get(wa, field), _get(wb, field))
        self.count("worker")

    def calibration(self):
        got = self.pair("calibration", "smoke/calibration.yaml", lambda p: p.read_bytes())
        if got is not None:
            if got[0] != got[1]:
                self.differ("calibration", "smoke/calibration.yaml", "<bytes>",
                            got[0].decode("utf-8", "replace"), got[1].decode("utf-8", "replace"))
            self.count("calibration")
        self.fields("calibration", "smoke/calibrate/sigma_z.json", (None,))
        self.fields("calibration", "smoke/calibrate/calibration.json", (None,),
                    transform=lambda v: {k: x for k, x in v.items() if k not in CALIBRATION_DROPPED}
                    if isinstance(v, dict) else v)

    def job_report(self, rel, condition, *, rerun):
        """J/report.json。初始权重哈希：训练条件必须是 64 位十六进制，identity 必须是 null；
        第一次任务的哈希记在 initial_model_sha256 项下，只评测重跑的随任务报告一起比。"""
        fields = (("initial_model_sha256",) if rerun else ())+JOB_REPORT_FIELDS
        got = self.fields("job_reports", rel, fields)
        if got is None:
            return
        for side, report in zip(("a", "b"), got):
            value = _get(report, "initial_model_sha256")
            good = (value is None if condition == UNTRAINED else
                    isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value))
            if not good:
                self.differ("initial_model_sha256", rel, f"{side}: format",
                            value, "null" if condition == UNTRAINED else "64 hex")
        if not rerun:
            self.values("initial_model_sha256", rel, "initial_model_sha256",
                        _get(got[0], "initial_model_sha256"), _get(got[1], "initial_model_sha256"))
            self.count("initial_model_sha256")
            if condition != UNTRAINED:
                self.count("initial_model_sha256_nonnull")

    def training(self, t, updates):
        expected = [f"{i:04d}.json" for i in range(1, updates+1)]
        for name in self.names("updates", f"{t}/updates", "*.json", expected=expected):
            self.fields("updates", f"{t}/updates/{name}", UPDATE_FIELDS)
        steps = self.names("validation_files", f"{t}/validations", "*", dirs=True)
        for step in ("0000", f"{updates:04d}"):     # 不写死 0/10/20：第一步和最后一步必须有
            if step not in steps:
                self.differ("validation_files", f"{t}/validations", f"<step {step}>",
                            *("present" if (root/t/"validations"/step).is_dir() else _MISSING
                              for root in (self.a, self.b)))
        for step in steps:
            for name in self.names("validation_files", f"{t}/validations/{step}", "*.json"):
                self.fields("validation_files", f"{t}/validations/{step}/{name}", VALIDATION_FIELDS)
        selections = self.names("selection", f"{t}/validations", "selection-*.json",
                                expected=[f"selection-{s}.json" for s in steps])
        for name in selections:
            self.fields("selection", f"{t}/validations/{name}", SELECTION_FIELDS)
        self.fields("training_best", f"{t}/report.json", TRAINING_BEST_FIELDS)
        self.tensors("model_state", f"{t}/checkpoints/step-{updates:04d}.pt")

    def evaluation(self, e, *, rerun):
        got = self.fields("evaluation_registration", f"{e}/registration.json", ("queries",))
        queries = _get(got[0], "queries") if got is not None else None
        expected = [f"{i:05d}.json" for i in range(len(queries))] if isinstance(queries, list) else None
        for name in self.names("evaluation_rows", f"{e}/rows", "*.json", expected=expected):
            rel = f"{e}/rows/{name}"
            row = self.fields("evaluation_rows", rel, ROW_FIELDS)
            if row is None:
                continue
            if rerun:
                self.count("evaluation_rows_rerun")
            for side, r in zip(("a", "b"), row):
                if _get(r, "status") != "ok":
                    self.differ("evaluation_rows", rel, f"{side}: status", _get(r, "status"), "ok")
            if _get(row[0], "query.kind") == "point" or _get(row[1], "query.kind") == "point":
                self.values("evaluation_rows", rel, "donor.trajectory_id",
                            _get(row[0], "donor.trajectory_id"), _get(row[1], "donor.trajectory_id"))
        report = self.fields("evaluation_reports", f"{e}/report.json", (None,))
        for side, r in zip(("a", "b"), report or ()):
            if _get(r, "counts.failed") != 0:
                self.differ("evaluation_reports", f"{e}/report.json", f"{side}: counts.failed",
                            _get(r, "counts.failed"), 0)

    # -- 整份比较：防止上面没点名的字段或文件漏掉
    def whole_output(self):
        found = []
        for root in (self.a, self.b):
            smoke = root/"smoke"
            found.append({p.relative_to(root).as_posix() for p in smoke.rglob("*") if p.is_file()}
                         if smoke.is_dir() else None)
        na, nb = found
        if na is None or nb is None or na != nb or not na:
            self.differ("all_files", "smoke", "<names> (only in A / only in B)",
                        sorted(na-(nb or set())) if na is not None else _MISSING,
                        sorted(nb-(na or set())) if nb is not None else _MISSING)
        common = sorted((na or set()) & (nb or set()))
        self.count("all_files", len(common))
        for rel in common:
            if rel.endswith(".json"):
                self.whole_json(rel)
            elif rel.endswith(".pt"):
                self.whole_checkpoint(rel)

    def whole_json(self, rel):
        got = self.pair("all_json", rel)
        if got is None:
            return
        va, vb = (_drop_keys(v, EXCLUDED_KEYS) for v in got)
        if rel == WHOLE_CALIBRATION:        # 含绝对路径
            va, vb = ({k: x for k, x in v.items() if k not in CALIBRATION_DROPPED} if isinstance(v, dict) else v
                      for v in (va, vb))
        self.values("all_json", rel, "", va, vb)
        self.count("all_json")

    def whole_checkpoint(self, rel):
        got = self.pair("all_checkpoints", rel, lambda p: torch.load(p, map_location="cpu", weights_only=True))
        if got is None:
            return
        va, vb = (_drop_keys(v, CHECKPOINT_EXCLUDED_KEYS) for v in got)
        for field, a, b in _tree_diffs(va, vb, ""):
            self.differ("all_checkpoints", rel, field, a, b)
        self.count("all_checkpoints")

    def condition(self, c, updates):
        j = f"smoke/jobs/{c}"
        self.complete(f"{j}/COMPLETE.json")
        self.job_report(f"{j}/report.json", c, rerun=False)
        if c == UNTRAINED:
            t = f"{j}/training/{c}"
            for name in self.names("validation_files", t, "*.json"):
                self.fields("validation_files", f"{t}/{name}", VALIDATION_FIELDS)
        else:
            self.training(f"{j}/training/{c}", updates)
        self.evaluation(f"{j}/evaluation/{c}", rerun=False)
        if c != UNTRAINED:
            r = f"smoke/jobs/{c}-evaluate"
            self.complete(f"{r}/COMPLETE.json")
            self.job_report(f"{r}/report.json", c, rerun=True)
            self.evaluation(f"{r}/evaluation/{c}", rerun=True)


def compare_runs(a: Path, b: Path, *, conditions=JOB_ORDER, updates: int = 20) -> dict:
    """逐位比较两个子进程输出目录（worker.json + smoke/）；返回 identical、counts、mismatches、mismatch_count。

    JSON 值按 json.dumps(sort_keys=True) 规范化后比字符串（等价于 float64 逐位相同，分得出 -0.0/0.0、1/1.0）；
    张量比键列表（含顺序）、dtype、shape 和原始字节。任何一项两边都缺、数量为 0、或文件名集合不同都算不同。
    除了按项点名的比较，还做整份比较（all_files、all_json、all_checkpoints，见 _Comparison.whole_output）。
    两份 worker.json 的 tree 相同时报 ValueError。
    """
    run = _Comparison(a, b)
    run.worker()
    run.calibration()
    for c in conditions:
        run.condition(c, updates)
    run.whole_output()
    for item in ITEMS:
        if not run.counts.get(item):
            run.differ(item, "", "<count>", run.counts.get(item, 0), "> 0")
    return dict(identical=run.total == 0, counts=dict(run.counts), mismatches=run.mismatches,
                mismatch_count=run.total)


# ---------------------------------------------------------------- 父进程

def _counts_text(counts: dict) -> str:
    return "、".join(f"{k} {v}" for k, v in counts.items())


def summary_line(result: dict, report_path=None) -> str:
    """打印的最后一行（也原样写进 dual-tree-report.json 的 summary_line）。"""
    if result["identical"]:
        return f"双树对照：逐位相同（{_counts_text(result['counts'])}）"
    where = f"（见 {report_path}）" if report_path else ""
    return f"双树对照：发现 {result['mismatch_count']} 处不同{where}"


def _print_result(result: dict, line: str) -> None:
    for item, n in result["counts"].items():
        print(f"  {item}: {n}")
    for m in result["mismatches"]:
        print(f"  不同：{m['item']}  {m['path']}  {m['field']}  A={_canon(m['a'])}  B={_canon(m['b'])}")
    hidden = result["mismatch_count"]-len(result["mismatches"])
    if hidden > 0:
        print(f"  （另有 {hidden} 处不同未列出）")
    print(line)


def _start(cmd, *, cwd, env, log: Path) -> subprocess.Popen:
    with open(log, "xb") as handle:
        return subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=handle,
                                stderr=subprocess.STDOUT)


def _tail(log: Path, lines: int = TAIL_LINES) -> str:
    try:
        return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError as error:
        return f"<读不到 {log}：{error}>"


def _failed(name, code, log: Path) -> int:
    print(f"{name} 失败（退出码 {code}），{log} 的最后 {TAIL_LINES} 行：")
    print(_tail(log))
    print(f"双树对照：失败（{name} 退出码 {code}，见 {log}）")
    return 1


def worker_command(script: Path, tree: Path, features: Path, checkpoint: Path, output: Path) -> list[str]:
    """一棵树的子进程命令（§3.4.2 第 3 步）。"""
    return [sys.executable, "-B", str(script), "--worker", "--tree", str(tree), "--features", str(features),
            "--ac-checkpoint", str(checkpoint), "--output", str(output)]


def _stop(proc: subprocess.Popen, name: str) -> None:
    """先 SIGTERM，KILL_GRACE_SECONDS 内不退出再 SIGKILL；等它真的结束（回收进程）。"""
    if proc.poll() is not None:
        return
    print(f"结束 worker {name}（SIGTERM）", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        print(f"worker {name} {KILL_GRACE_SECONDS} s 内没有退出，SIGKILL", flush=True)
        proc.kill()
        proc.wait()


def run_workers(workdir: Path, trees, script: Path, features: Path, checkpoint: Path, env: dict, *,
                parallel: bool) -> int:
    """每棵树一个子进程，输出到 <workdir>/<name>/。并行时每 POLL_SECONDS 轮询一次，
    第一个非 0 退出出现时立即结束另一个（不等它跑完），返回 1；全部成功返回 0。"""
    jobs = []
    for name, tree in trees:
        out = Path(workdir)/name
        out.mkdir()
        print(f"worker {name}（{tree}）：进度见 {out/'console.log'}", flush=True)
        jobs.append((name, worker_command(script, tree, features, checkpoint, out), tree, out/"console.log"))
    if not parallel:
        for name, cmd, tree, log in jobs:
            code = _start(cmd, cwd=tree, env=env, log=log).wait()
            if code:
                return _failed(f"worker {name}", code, log)
        return 0
    running = [(name, _start(cmd, cwd=tree, env=env, log=log), log) for name, cmd, tree, log in jobs]
    try:
        while running:
            for name, proc, log in list(running):
                code = proc.poll()
                if code is None:
                    continue
                running.remove((name, proc, log))
                if code:
                    for other, other_proc, _ in running:
                        _stop(other_proc, other)
                    running = []
                    return _failed(f"worker {name}", code, log)
            if running:
                time.sleep(POLL_SECONDS)
        return 0
    finally:
        for other, other_proc, _ in running:    # 父进程自己出错（例如被中断）时也不留下子进程
            _stop(other_proc, other)


def run_dual_tree(v7_tree: Path, new_tree: Path, workdir: Path, *, parallel: bool) -> int:
    started = time.perf_counter()
    preflight_checks(v7_tree, new_tree, workdir, parallel=parallel)
    v7, new, workdir = Path(v7_tree).resolve(), Path(new_tree).resolve(), Path(workdir).resolve()
    workdir.mkdir(parents=True)
    logs = workdir/"logs"
    logs.mkdir()
    env, py, script = worker_environment(), sys.executable, new/"tests"/"addon_dual_tree.py"
    features, checkpoint = workdir/"features", workdir/AC_CHECKPOINT
    stages = {}
    encode = [py, "-B", str(v7/"tools"/"encode_v7.py")]
    data = ["--system", SYSTEM, "--trajectories", str(TRAJECTORIES), "--output", str(features)]
    for name, cmd in (("encode", [*encode, "encode", *data, "--device", "cpu", "--fake-encoder"]),
                      ("finalize", [*encode, "finalize", *data]),
                      ("ac_checkpoint", [py, "-B", str(script), "--make-ac-checkpoint", str(checkpoint),
                                         "--tree", str(v7)])):
        print(f"{name} …", flush=True)
        t0 = time.perf_counter()
        code = _start(cmd, cwd=v7, env=env, log=logs/f"{name}.log").wait()
        stages[name] = time.perf_counter()-t0
        if code:
            return _failed(name, code, logs/f"{name}.log")
    t0 = time.perf_counter()
    workers = (("v7", v7), ("full", new))
    if run_workers(workdir, workers, script, features, checkpoint, env, parallel=parallel):
        return 1
    stages["workers"] = time.perf_counter()-t0
    t0 = time.perf_counter()
    result = compare_runs(workdir/"v7", workdir/"full")
    stages["compare"] = time.perf_counter()-t0
    stages["total"] = time.perf_counter()-started
    report_path = workdir/"dual-tree-report.json"
    line = summary_line(result, report_path)
    write_json(report_path, dict(**result, summary_line=line, workers={name: json.loads(
        (workdir/name/"worker.json").read_text(encoding="utf-8")) for name, _ in workers}, stage_seconds=stages))
    total = round(stages["total"])
    print(f"各阶段耗时（秒）：{json.dumps({k: round(v, 1) for k, v in stages.items()}, ensure_ascii=False)}；"
          f"总耗时 {total//3600}:{total//60 % 60:02d}:{total % 60:02d}")
    _print_result(result, line)
    return 0 if result["identical"] else 1


def _parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve()
    p = argparse.ArgumentParser(description="双树对照冒烟（S00 §3.4）")
    p.add_argument("--workdir", type=Path, help="完整模式：必须不存在，且不在任何一棵树里")
    p.add_argument("--v7-tree", type=Path, default=here.parents[2], help="v7 树（默认仓库根目录）")
    p.add_argument("--new-tree", type=Path, default=here.parents[1], help="新树（默认 po-jepa-full）")
    p.add_argument("--sequential", action="store_true", help="先 v7 后新树，不并行")
    p.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"), help="只比较两个已有的子进程输出目录")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--make-ac-checkpoint", type=Path, metavar="PATH", help=argparse.SUPPRESS)
    p.add_argument("--tree", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--features", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--ac-checkpoint", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None) -> int:
    p = _parser()
    a = p.parse_args(argv)
    modes = [m for m, on in (("--workdir", a.workdir is not None), ("--compare", a.compare is not None),
                             ("--worker", a.worker), ("--make-ac-checkpoint", a.make_ac_checkpoint is not None)) if on]
    if len(modes) != 1:
        p.error("必须恰好给出 --workdir、--compare 之一")
    if a.worker:
        if None in (a.tree, a.features, a.ac_checkpoint, a.output):
            p.error("--worker 需要 --tree、--features、--ac-checkpoint、--output")
        worker(a.tree, a.features, a.ac_checkpoint, a.output)
        return 0
    if a.make_ac_checkpoint is not None:
        if a.tree is None:
            p.error("--make-ac-checkpoint 需要 --tree")
        make_ac_checkpoint(a.make_ac_checkpoint, a.tree)
        return 0
    try:
        if a.compare is not None:
            result = compare_runs(*a.compare)
            _print_result(result, summary_line(result))
            return 0 if result["identical"] else 1
        return run_dual_tree(a.v7_tree, a.new_tree, a.workdir, parallel=not a.sequential)
    except ValueError as error:
        print(f"双树对照：失败（{error}）")
        return 1


if __name__ == "__main__":
    sys.exit(main())
