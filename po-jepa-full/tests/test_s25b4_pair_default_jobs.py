"""S25b4：二阶场景按两帧编码（<场景>~tubelet=pair）并进协议表和 make_v8_jobs 的默认场景。

说明和验收单 docs/segments/S25b4.md。期望值写死或从调节项字符串独立拼（错误清单：测试不拿被测函数当参照）。
"""
from pathlib import Path
import re
import subprocess
import sys

import pytest

from po_jepa import protocols, variants

ROOT = Path(__file__).resolve().parents[1]
FIRST = ("double_well", "van_der_pol", "multi_particle", "advection_diffusion", "landscape~shape=multi_well",
         "gray_scott", "push")
SECOND = ("pendulum", "bouncing_balls", "wave", "cart_pole", "spring_chain")
PAIR = tuple(s+"~tubelet=pair" for s in SECOND)
CONDITIONS = ("po_jepa_full", "po_jepa_no_consistency", "matched_transformer", "identity", "vjepa2_ac_small", "vjepa2_ac",
              "dino_wm_small", "jepa_wm_small")   # 2026-11-25
ADDON = ("po_jepa_time_only", "po_jepa_space_only", "transformer_consistency", "fixed_fourier", "koopman", "po_jepa_rollout_sup")


def _jobs(*args):
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/make_v8_jobs.py"), *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def _cmds(*args):
    return [l for l in _jobs(*args) if l and not l.startswith("#")]


def _arg(line, flag):
    m = re.search(rf"{flag} (\S+)", line)
    return None if m is None else m.group(1)


def _knobs(text):
    """调节项字符串 → 按名排序的 {名: 值}（独立于 variants.format 的写法比较）。"""
    return dict(sorted(i.split("=", 1) for i in text.split(",") if i))


def _scene_of_path(path):
    """../features-v8/<场景>-<条数> 或 ../results-v8/full/<场景>/seed… → (系统, {调节项})。"""
    name = path.split("/")[2] if path.startswith("../features-v8/") else path.split("/")[3]
    name = re.sub(r"-\d+$", "", name) if path.startswith("../features-v8/") else name
    system, _, knobs = name.partition("~")
    return system, _knobs(knobs)


# —— 协议表 ——

def test_scene_lists():
    assert protocols.SCENES == FIRST and protocols.SECOND_ORDER_SCENES == SECOND
    assert protocols.PAIR_SCENES == PAIR
    assert protocols.DEFAULT_SCENES == FIRST+PAIR and len(protocols.DEFAULT_SCENES) == 12
    assert set(protocols.JOB_SCENES) == set(FIRST+PAIR+SECOND) and len(protocols.JOB_SCENES) == 17
    for s in PAIR:
        system, _, knobs = s.partition("~")
        assert variants.format(system, variants.parse(system, knobs)) == "tubelet=pair"   # 已是规范写法（目录名不变形）
        assert variants.parse(system, knobs).tubelet == "pair"


@pytest.mark.parametrize("scene", PAIR)
def test_pair_scene_has_same_cells_and_every_test_dataset_is_pair(scene):
    system = scene.partition("~")[0]
    cells = protocols.tests(scene, protocols.LEVELS)
    assert cells == protocols.tests(system, protocols.LEVELS) and cells
    for _, _, t in cells:
        merged = protocols.merged(scene, t)
        assert _knobs(merged) == {**_knobs(t), "tubelet": "pair"}
        assert variants.parse(system, merged).tubelet == "pair"
    for t in (*protocols.TIMING_TESTS, *protocols.SPACE_TESTS, "horizon=128"):
        assert _knobs(protocols.merged(scene, t)) == {**_knobs(t), "tubelet": "pair"}


def test_cell_counts_are_written_out():
    def count(scenes, level):
        return sum(len(protocols.tests(s, (level,))) for s in scenes)
    assert [count(FIRST, lv) for lv in protocols.LEVELS] == [75, 82, 77]   # S21b 的 82 格不变
    assert [count(PAIR, lv) for lv in protocols.LEVELS] == [48, 53, 53]
    assert [count(SECOND, lv) for lv in protocols.LEVELS] == [48, 53, 53]   # 单帧复制对照，格子相同


def test_table_states_pair_cells_and_dataset_md_is_current():
    text = protocols.table()
    assert "二阶 5 个场景合计标准档 53 个，轻档 48 个，重档 53 个。两者都在任务表默认里" in text
    assert "`pendulum~g_over_l=14,tubelet=pair`" in text
    assert "现在是单帧复制编码" not in text
    r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/protocol_table.py"), "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout+r.stderr


# —— 任务表默认 ——

def test_default_full_stage_has_12_scenes_and_second_order_is_pair():
    lines = _jobs("--stage", "full")
    assert not lines[0].startswith("# 注意：二阶场景")          # 默认里没有单帧复制的二阶场景
    cmds = [l for l in lines if l and not l.startswith("#")]
    runs = [l for l in cmds if l.startswith("python run.py run ")]
    assert len(runs) == 12*3*8 == 288
    assert sum("encode_v7.py encode" in l for l in cmds) == 12*4
    assert sum("probe_actions.py" in l for l in cmds) == 12 and sum("run.py calibrate " in l for l in cmds) == 12
    want = {("double_well", ""), ("van_der_pol", ""), ("multi_particle", ""), ("advection_diffusion", ""),
            ("landscape", "shape=multi_well"), ("gray_scott", ""), ("push", ""),
            *((s, "tubelet=pair") for s in SECOND)}
    got = {(_arg(l, "--system"), _arg(l, "--variant") or "") for l in cmds if "--system" in l}
    assert got == want
    for l in runs:
        system, knobs = _arg(l, "--system"), _knobs(_arg(l, "--variant") or "")
        assert _scene_of_path(_arg(l, "--features-root")) == (system, knobs)
        assert _scene_of_path(_arg(l, "--output")) == (system, knobs)
        assert (knobs.get("tubelet") == "pair") == (system in SECOND)


def _second_order_lines(cmds):
    out = []
    for l in cmds:
        names = [p for p in re.findall(r"\.\./(?:features|results)-v8/\S+", l)]
        if any(s in l for s in SECOND):
            out.append((l, names))
    return out


@pytest.mark.parametrize("stage", ["pilot", "variants", "strength", "long", "timing", "space", "readout", "addon", "addon-pilot"])
def test_every_second_order_line_of_every_stage_is_pair(stage):
    cmds = _cmds("--stage", stage)
    lines = _second_order_lines(cmds)
    assert lines
    for l, paths in lines:
        if "--system" in l:
            assert "tubelet=pair" in (_arg(l, "--variant") or ""), l
        for p in paths:
            if p.startswith("../results-v8/") and p.split("/")[3] in ("summary",):
                continue
            assert "tubelet=pair" in p, (p, l)
        tv = _arg(l, "--test-variant")
        assert tv is None or "tubelet" not in tv, l      # transfer / timing / space 拒绝 --test-variant 里写 tubelet


def test_transfer_timing_space_read_the_merged_pair_test_set():
    """测试集 = 训练数据集的调节项叠上 --test-variant；特征目录名要正好是它（含 tubelet=pair）。"""
    checked = 0
    for stage in ("variants", "timing", "space"):
        for l in _cmds("--stage", stage):
            if not any(f"/{s}~" in l for s in SECOND):
                continue
            tv = _arg(l, "--test-variant")
            if tv is None:
                continue
            if "run.py transfer" in l:
                train = (_arg(l, "--system"), _knobs(_arg(l, "--variant")))
                assert _arg(l, "--training-run") is None or _scene_of_path(_arg(l, "--training-run")) == train
            else:
                train = _scene_of_path(_arg(l, "--job"))
            feats = _scene_of_path(_arg(l, "--features-root") or _arg(l, "--features"))
            assert train[1]["tubelet"] == "pair"
            assert feats == (train[0], {**train[1], **_knobs(tv)}), l
            checked += 1
    assert checked == 8*53 + 3*6*8*5 + 3*6*8*5     # V3 标准档 53 格 × 8 条件；T3、S3 各 5 场景 × 6 项 × 3 种子 × 8 条件


def test_plan_lines_plan_on_the_pair_base_scene_without_test_variant():
    plan = [l for l in _cmds("--stage", "variants") if "plan_v7.py" in l]
    assert len(plan) == 12*8
    for l in plan:
        assert "--test-variant" not in l
        job = _arg(l, "--job")
        system, knobs = _scene_of_path(job)
        if system in SECOND:
            assert knobs == {"tubelet": "pair"} and _arg(l, "--output").split("/")[4] == f"{system}~tubelet=pair"
        oracle = _arg(l, "--oracle-samples")
        if job.endswith("/identity"):
            assert oracle == {"gray_scott": "16", "push": "32", "wave": "32"}.get(system)
        else:
            assert oracle is None


def test_addon_and_readout_default_counts():
    addon = _cmds("--stage", "addon")
    assert sum(l.startswith("python run.py run ") for l in addon) == 12*3*len(ADDON)
    assert sum("calibrate-addon" in l for l in addon) == 12
    readout = _cmds("--stage", "readout")
    assert sum("state_readout.py fit" in l for l in readout) == 12
    modal = [l for l in readout if "modal_spectrum.py" in l]
    # 线性场景 advection_diffusion、spring_chain~tubelet=pair、wave~tubelet=pair × 3 种子 × 2 个 PO-JEPA 条件
    assert len(modal) == 3*3*2
    assert {_scene_of_path(_arg(l, "--job"))[0] for l in modal} == {"advection_diffusion", "spring_chain", "wave"}
    assert all("tubelet=pair" in l for l in modal if "advection_diffusion" not in l)


def test_dup_control_still_explicit_and_bad_scenes_refused():
    lines = _jobs("--stage", "full", "--scenes", "pendulum")
    assert lines[0].startswith("# 注意：二阶场景 pendulum") and "dup" in lines[0] and "tubelet=pair" in lines[0]
    cmds = [l for l in lines if l and not l.startswith("#")]
    assert all("tubelet" not in l for l in cmds) and sum(l.startswith("python run.py run ") for l in cmds) == 24
    mixed = _jobs("--stage", "full", "--scenes", "pendulum,pendulum~tubelet=pair")
    assert mixed[0] == ("# 注意：二阶场景 pendulum 按单帧复制（dup）编码，特征里看不到速度，是对照；"
                        "两帧编码要写 <场景>~tubelet=pair（默认场景就是这个写法）。")
    for bad in ("double_well~tubelet=pair", "pendulum~tubelet=dup", "pendulum~tubelet=pair,g_over_l=14"):
        r = subprocess.run([sys.executable, "-B", str(ROOT/"tools/make_v8_jobs.py"), "--stage", "full", "--scenes", bad],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0 and "不是 v8 的基础场景" in r.stderr, bad


def test_readout_and_long_paths_belong_to_the_same_scene():
    """S25b4-fix1（Codex S25b4 P3）：读出、长时程每条命令的探针、真值、基础特征、长时程特征都要是本场景本调节项的，
    不只是“带 tubelet=pair”（wave 读 pendulum 的探针、长时程特征写成基础特征，原来的测试都放过）。"""
    def scene(system, knobs):
        return system + (("~" + ",".join(f"{k}={v}" for k, v in sorted(knobs.items()))) if knobs else "")

    def feats(system, knobs):
        return f"../features-v8/{scene(system, knobs)}-2000"

    readout, checked = _cmds("--stage", "readout"), 0
    for l in readout:
        if "state_readout.py" not in l:
            continue
        if " fit " in l or " truth " in l:
            system, knobs = _arg(l, "--system"), _knobs(_arg(l, "--variant") or "")
        else:
            system, knobs = _scene_of_path(_arg(l, "--job"))
        name = scene(system, knobs)
        long = feats(system, {**knobs, "horizon": "128"})
        if " fit " in l:
            assert _arg(l, "--features") == feats(system, knobs) and _arg(l, "--output") == f"../results-v8/full/readout/probe/{name}"
        elif " truth " in l:
            assert _arg(l, "--features") == feats(system, knobs) and _arg(l, "--long-features") == long
            assert _arg(l, "--output") == f"../results-v8/full/readout/truth/{name}.json"
        else:
            assert _arg(l, "--probe") == f"../results-v8/full/readout/probe/{name}"
            assert _arg(l, "--truth") == f"../results-v8/full/readout/truth/{name}.json"
            kind = "rollout" if "state_readout.py rollout" in l else "counterfactual"
            assert _arg(l, "--features") == (long if kind == "rollout" else feats(system, knobs)), l
            assert _arg(l, "--output").startswith(f"../results-v8/full/readout/{kind}/{name}/seed")
        checked += 1
    assert checked == 12 + 12 + 2*12*3*8
    long_cmds = [l for l in _cmds("--stage", "long") if "long_rollout.py" in l]
    for l in long_cmds:
        system, knobs = _scene_of_path(_arg(l, "--job"))
        assert _arg(l, "--features") == feats(system, {**knobs, "horizon": "128"})
        assert _arg(l, "--output").startswith(f"../results-v8/full/long/{scene(system, knobs)}/seed")
    assert len(long_cmds) == 12*3*8
