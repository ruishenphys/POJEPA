# GPU tasks: handoff (2026-11-11)

This file lists every PO-JEPA task that needs a GPU, which ones you can run now, which are still blocked, and what to send back.
Code references: `full-model` branch e9016a8 (code in `po-jepa-full/`) and `v7` branch a305bfe (v7.1 + extra evaluation).
Times, memory and disk figures marked *(est.)* are estimates, not measurements.

## 1. Summary

| # | Task | Where | Status | GPU | Time on 2× RTX PRO 6000 |
|---|---|---|---|---|---|
| T0 | v7.1 formal run (`RESTART_V7_1.md`) | `v7` branch, `po-jepa/` | **Running. Keep going, do not change anything.** | both cards | as planned |
| T1 | Extra evaluation of finished runs (`EXTRA_EVAL_V7.md`) | `v7` branch, `po-jepa/` | Pilot part: ready (CPU is fine). Formal part: after T0 | optional | pilot 10–20 min on a free GPU, ~2 h on CPU |
| T2 | First-GPU checks for `full-model` | `full-model/po-jepa-full/` | **Ready.** Run when one card is free | 1 card, ~5 GB *(est.)* | ~10 min *(est.)* |
| T3 | S22 two-frame tubelet probe | `full-model/po-jepa-full/` | **Ready.** Run when a card is free (skip if you already ran it) | 1–2 cards, ~5 GB *(est.)* | ≤1 GPU-hour *(est.)* |
| T4 | Simulation add-on runs | `full-model/po-jepa-full/` | **Blocked**: waits for T0, then a code update for v7.1 | both cards | pilot ½ day, formal + variants 2–3 days, D3 7.5 h *(est.)* |
| T5 | v8 dataset extension runs | `full-model/po-jepa-full/` | **Blocked**: waits for T0 results, method decisions and the S22 result | both cards | see §7 |
| T6 | DROID real-robot pipeline | `full-model/po-jepa-full/` | **Blocked**: waits for decisions and two code segments | ≥40 GB card *(est.)* | 200–330 GPU-hours *(est.)* |

Order: T0 first and undisturbed. T2 and T3 are short; slot them in only when a card is idle, otherwise run them right after T0. T4–T6 get their own instructions when they are unblocked.

## 2. Hardware and environment

- Linux machine. The DROID runner (`tools/droid_run.py`) uses `flock` and process groups and does not run on native Windows.
- Same Python environment as v7 (`conda activate pojepa`): Python 3.12.x, `torch==2.11.0` built for CUDA 13.0 (`2.11.0+cu130`), `numpy==2.4.3`, `PyYAML==6.0.3`, `scipy==1.16.3`. The production entry points refuse any other versions (`dependency versions differ`).
- The DROID check in T2 also needs `opencv-python-headless==4.13.0.92` and `h5py==3.15.1` (from `po-jepa-full/pyproject.toml`). Install them only if missing (see T2), with `--no-deps` so pip cannot touch numpy or torch.
- GPU memory: the official-scale V-JEPA 2-AC baseline (`vjepa2_ac`, about 300 M parameters, bf16 fine-tuning with AdamW) needs a large card; the ICML plan estimates ≥40 GB. Every task that trains or checks it (T0, T4 D3, T6) cannot run on small cards. Encoding with the frozen ViT-g (about 1.0 B parameters, float32 by protocol, ~3.8 GiB of weights) needs about 5 GB *(est.)*.
- Shared assets from the v7 work directory, used as they are: `vjepa2/` (upstream code at commit `204698b45b3712590f06245fbfba32d3be539812`) and `checkpoints/vjepa2-vitg-fpc64-256.pth` (SHA-256 `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e`). DROID (T6) additionally needs `checkpoints/vjepa2-ac-vitg.pt`.

All commands below use one variable for the work directory, the folder that already holds `po-jepa/`, `vjepa2/` and `checkpoints/`:

```bash
WORK=/absolute/path/to/your/work/directory
conda activate pojepa
```

## 3. T0: v7.1 formal run (running)

Follow `RESTART_V7_1.md` on the `v7` branch, as you are doing now. Do not edit the `v7` branch or delete the half-finished results from before the pause (`results-v7/full/<scene>/…`). When `run_jobs.py` has printed `全部完成。` ("all done") for both `jobs-full.txt` and `jobs-variants.txt`, return the results as described in the last section of `RESTART_V7_1.md`.

## 4. T1: extra evaluation (`EXTRA_EVAL_V7.md`)

Run it exactly as `EXTRA_EVAL_V7.md` on the `v7` branch says. It only reads finished checkpoints and adds three new files, so it does not affect T0.

- Pilot results: available now; use `--device cpu` while both cards are busy.
- Formal results: after T0 finishes, with `--results-root ../results-v7/full/v7.1 --output ../extra-eval-v7/full --max-test-trajectories 50`.
- Send back `extra-eval-v7.zip` (JSON only, a few MB).

## 5. T2: first-GPU checks (~10 minutes, one card)

These confirm that the `full-model` code works on a real GPU. Nothing here trains a model or writes into the v7 directories.

```bash
# 1. Get the full-model code next to po-jepa/ (skip the clone if $WORK/full-model already exists; then just pull)
cd $WORK
[ -d full-model ] || git clone -b full-model https://github.com/ruishenphys/POJEPA.git full-model
cd $WORK/full-model && git pull && git log -1 --oneline
cd $WORK/full-model/po-jepa-full

# 2. Environment: must show Python 3.12.x, 2.11.0+cu130, 13.0 and your GPU name
python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
# This must print a path inside full-model/po-jepa-full/src (not the v7 copy):
PYTHONPATH=src python -c "import po_jepa; print(po_jepa.__file__)"

# 3. Delivery tests (8 tests). The 4 tests that fail in every CPU-only full regression are in this file; here all 8 must pass.
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -v tests/test_delivery.py

# 4. GPU eigensolver (S08b): float64 eigh on CUDA must agree with the CPU solver. Expect "1 passed", not "skipped".
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -rs tests/test_droid_s08b.py -k cuda_float64_agrees

# 5. Real-encoder check for DROID encoding (05 §7): one public DROID video, 38,604,360 bytes
python -c "import cv2, h5py" || pip install --no-deps opencv-python-headless==4.13.0.92 h5py==3.15.1
python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"   # must still be 2.4.3 and 2.11.0+cu130
mkdir -p $WORK/droid-check
curl -L -o $WORK/droid-check/23897859.mp4 \
  "https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1/ILIAD/success/2023-05-12/Fri_May_12_16:48:11_2023/recordings/MP4/23897859.mp4"
ls -l $WORK/droid-check/23897859.mp4
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m po_jepa.droid.encode \
  --config configs/droid/pilot.yaml --data-root $WORK/droid-check \
  --upstream-root $WORK/vjepa2 --checkpoint $WORK/checkpoints/vjepa2-vitg-fpc64-256.pth \
  --device cuda --check-mp4 $WORK/droid-check/23897859.mp4 | tee $WORK/droid-check/check_mp4.json
echo "exit=${PIPESTATUS[0]}"
```

Pass criteria:

| Step | Pass |
|---|---|
| 3 | `8 passed`, including the 4 tests below |
| 4 | `1 passed` (a skip means CUDA was not visible) |
| 5 | `exit=0`; the JSON shows `"preprocess_square_equals_v7": true`, `"encode_frames_equals_direct": true` and `"ok": true` for both extra resolutions (192 and 320) |

The 4 tests that the CPU-only full regressions report as failed (e.g. 3,507 passed, 147 skipped, 4 failed on full-model e9016a8) are all in step 3:

- `tests/test_delivery.py::test_workflow_failure_stops_and_preserves_stage[preflight]`
- `tests/test_delivery.py::test_workflow_failure_stops_and_preserves_stage[training]`
- `tests/test_delivery.py::test_workflow_failure_stops_and_preserves_stage[evaluation]`
- `tests/test_delivery.py::test_workflow_calls_all_phases_once_without_old_stage_records`

They fail on CPU machines because the production entry point checks the environment first (`environment.verify_versions`: Python 3.12.x, torch 2.11.0 built for CUDA 13.0, numpy 2.4.3, PyYAML 6.0.3) and a CPU-only torch has no CUDA version. In the v7 environment they must pass. If any of them fails with `dependency versions differ`, the environment is not the v7 one; stop and send the output.

Send back: the terminal output of steps 2–5 (copy into a text file) and `$WORK/droid-check/check_mp4.json`.

## 6. T3: S22 two-frame tubelet probe (≤1 GPU-hour, est.)

Purpose: decide whether the second-order scenes (pendulum, cart-pole, …) should be encoded from two real consecutive frames instead of one duplicated frame. Definitions and the decision rule are in `po-jepa-full/docs/segments/S22.md`. Scale: 200 trajectories × 67 tubelets = 13,400 encoder calls; features about 9.7 GB on disk; the fit step peaks at about 10 GB of host RAM. If you already ran S22 earlier, skip this and just send the report.

```bash
cd $WORK/full-model/po-jepa-full
OUT=$WORK/features-v8/tubelet-probe-200
mkdir -p $OUT

# Encode: one shard per card, both in parallel
CUDA_VISIBLE_DEVICES=0 python tools/tubelet_probe.py encode --trajectories 200 --output $OUT \
  --upstream-root $WORK/vjepa2 --checkpoint $WORK/checkpoints/vjepa2-vitg-fpc64-256.pth \
  --device cuda:0 --shard 0/2 > $OUT-encode0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python tools/tubelet_probe.py encode --trajectories 200 --output $OUT \
  --upstream-root $WORK/vjepa2 --checkpoint $WORK/checkpoints/vjepa2-vitg-fpc64-256.pth \
  --device cuda:0 --shard 1/2 > $OUT-encode1.log 2>&1 &
wait
tail -n 2 $OUT-encode0.log $OUT-encode1.log        # both should end with "[tubelet shard i/2] 完成" ("done")

# Fit the probe (a few minutes on a GPU)
CUDA_VISIBLE_DEVICES=0 python tools/tubelet_probe.py fit --trajectories 200 --output $OUT --device cuda:0 | tee $OUT-fit.txt
```

- Only one card free: run a single encode command with `--shard 0/1` instead of the two above.
- Interrupted: rerun the same command. Finished trajectories are skipped; half-written ones are redone.
- Out of GPU memory: add `--batch 8` to the encode command (default 34). Use the same value for every shard of one output folder.
- Send back: `$OUT/report.json`, `$OUT-fit.txt` and the two encode logs.

## 7. Blocked tasks (do not start yet)

### T4: simulation add-on runs

Five extra conditions on the v7 scenes (`po_jepa_time_only`, `po_jepa_space_only`, `transformer_consistency`, `fixed_fourier`, `koopman`), plus extra evaluations and D3 (official-scale AC planning with 256 samples). They reuse v7 features, checkpoints and calibrations, so no re-encoding.

Blocked because the add-on code was written against v7 (0ac45fe) and reads `results-v7/full/<scene>`, while v7.1 writes to `results-v7/full/v7.1/` and changed the source. After T0 finishes we will release a code update and a new English version of `po-jepa-full/ADDON_RUN_GUIDE.md`; follow that, not the current version. Size per the current guide: pilot 95 jobs (~½ day), formal 255 + variants 176 jobs (2–3 days), D3 5 jobs (7.5 h), all *(est.)* on your two cards.

### T5: v8 dataset extension

12 scenes (v7's 5 plus `gray_scott`, `push`, `pendulum`, `bouncing_balls`, `wave`, `cart_pole`, `spring_chain`). Job tables come from `tools/make_v8_jobs.py` (stages `pilot`, `full`, `variants`, `strength`, `long`, `timing`, `space`, `readout`, optional `addon`). Pilot: 7 scenes × 200 trajectories × 2,000 updates × 1 seed; full: 2,000 trajectories × 10,000 updates × 3 seeds.

Blocked on: the v7.1 results and the method choices that follow from them (zero-initialised residual readout for PO-JEPA, the temporal-consistency fix, whether v8 uses the v7.1 early-stopping rule); the S22 result (T3), which decides the second-order encoding (segment S25b). For scale, the ICML plan estimates 510–1,140 GPU-hours for all simulation experiments together *(est.)*.

### T6: DROID

Stages: `index`, `download`, `prepare`, `stats` (CPU) → `encode` (GPU) → `check/env` → `calibrate` → `train` (8 conditions × 3 seeds × 30,000 updates, incl. official-scale `vjepa2_ac`) → `eval` → `summary`. Job graph: `tools/droid_jobs.py` (pilot 40 jobs, formal 129 jobs); local runner: `tools/droid_run.py`. A pilot (500 episodes, 2,000 updates, 1 seed) comes first and fixes the formal step budget and time limits from `check/env`.

Resources *(est.)*: 200–330 GPU-hours; disk about 520 GB of features, up to 100 GB of raw videos at a time, 20+ GB of test raw files, about 10 GB of prepared files and about 100 GB under the run root (mostly AC checkpoints, ~3.6 GB each).

Blocked on: whether DROID goes into the ICML submission (advisor and Jerry); segment S16 (5-episode end-to-end smoke test on GPU, not written yet); segment S15 (cluster export, only if DROID runs on the school cluster; needs the cluster type); `configs/droid/pilot.yaml` still needs `resources.train.time_limit_minutes: 20` so the pilot exercises resume, as `docs/design/12-droid-jobs.md` §6 requires.

## 8. What to send back

| Task | Files |
|---|---|
| T0 | as in `RESTART_V7_1.md` |
| T1 | `extra-eval-v7.zip` |
| T2 | terminal output of steps 2–5, `droid-check/check_mp4.json` |
| T3 | `features-v8/tubelet-probe-200/report.json`, `tubelet-probe-200-fit.txt`, the two encode logs |

If anything fails, stop that task, keep its folder as it is and send the full terminal output; the other tasks can continue.
