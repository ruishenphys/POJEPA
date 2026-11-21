# GPU tasks other than training: handoff (updated 2026-11-18)

This file lists every PO-JEPA task that needs a GPU but is **not a training run**: checks, calibration, evaluation, probes and physics diagnostics. For each one it gives the entry point, what it reads, the rough GPU time, what it depends on and what to send back.
Training runs (C-1 and C-2 training lines, the formal training of the simulation dataset and DROID) are out of scope here, except where they share a job file with the tasks below.

Code references (2026-11-18):
- `full-model` branch **349f7b6** (code in `po-jepa-full/`).
- Two open pull requests on `full-model`, which some tasks below depend on:
  - **#8** `seg/S28c-fix1`: fixes the small-validation summary admission rules, reading checkpoints of resumed jobs, readout provenance checks and the stage budget checks.
  - **#7** `seg/S28d-dyadic-rollout-sup`: `po_jepa_rollout_sup` under the dyadic variant and its λ_r calibration. A Codex review is still pending.
- `v7` branch **d1b9ec4** (v7.1 training code 748ca22, the extra-evaluation tool and the results pack tool).

Times, memory and disk figures marked *(est.)* are estimates, not measurements. One "GPU-hour" means one hour of one RTX PRO 6000.

Naming: "formal training" means the full-scale runs on the 12-scene simulation dataset and on DROID; both use the same `full-model` code. `v8` in some file and folder names (`tools/make_v8_jobs.py`, `features-v8/`, `results-v8/`) is only an old code name for the simulation dataset.

## 1. Summary

| # | Task | Status | GPU time *(est.)* | Depends on |
|---|---|---|---|---|
| G1 | First-GPU checks: the 4 delivery tests that need CUDA 13.0, the CUDA float64 eigensolver test, the DROID real-encoder check | **Ready** | ~10 min, 1 card | nothing |
| G2 | S22 two-frame tubelet probe | **Ready** (skip if you already ran it, just send the report) | ≤1 GPU-hour | nothing |
| G3 | Extra evaluation of the v7.1 formal runs (`EXTRA_EVAL_V7.md`, `v7` branch) | **Ready** (v7.1 finished 2026-11-16) | 1–3 GPU-hours | v7.1 checkpoints still on disk |
| G4 | Small validation, B layer on the v7.1 checkpoints (`b-v7.1` job file, 139 commands, evaluation only) | **Ready after #8 is merged** | 5–12 GPU-hours, not measured | v7.1 checkpoints; PR #8 |
| G5 | Small validation C-1: calibration and checks (12 of the 29 commands in the `c1` job file) | **Ready after #8 is merged** | 2–4 GPU-hours for these 12 lines | PR #8 |
| G6 | Small validation C-2: calibration, checks, identity check, equal-step evaluation, readouts (63 of the 84 commands in the `c2` job file) | **Blocked** | 10–20 GPU-hours | C-1 summary and Jerry's choice of variant; G4 steps R1, R2, L1, L2; PR #8; PR #7 if the variant is dyadic |
| G7 | Other evaluation tasks that cannot run yet | **Blocked** (code or decisions missing) | see §8 | see §8 |
| G8 | AC checks (C7): AC direct prediction with its context cut to the 8 frames it was trained on, and the feature noise floor (same state encoded twice) | **Ready once its pull request is merged** (pull only after the running job files are finished) | ~1–3 GPU-hours *(est.)* for seed 1703 | v7.1 AC checkpoints; the encoder weights and `vjepa2` checkout used for encoding |
| G9 | C-2 shrinkage check (`tools/shrinkage_check.py`): readout slope / spread / motion gain and LN-feature spread of the C-2 checkpoints on the registered test windows | **Ready once its pull request is merged** | ~1–2 GPU-hours *(est.)*, 21 commands | C-2 checkpoints, readout probes and truth files under `results-small-validation/` (all from G4/G6) |

Suggested order: G1 and G2 first (short). Then G3, G4 and the `c1` job file in parallel on the two cards. G6 comes after the C-1 summary has been sent back and Jerry has confirmed the variant.

**Do not `git pull` in `full-model` while any job from that clone is running or still has to resume.** The run entry points check the delivery manifest and stop with `delivery source changed` if the code changed under them. So ask Jerry to merge #8 (and #7, if its review is done by then) before you clone or pull for G4 and G5. If #7 is merged later, pull only after all G4 and C-1 jobs are finished, then generate the `c2` job file.

## 2. Hardware and environment

- Linux machine. The DROID runner (`tools/droid_run.py`) uses `flock` and process groups and does not run on native Windows.
- Same Python environment as v7 (`conda activate pojepa`): Python 3.12.x, `torch==2.11.0` built for CUDA 13.0 (`2.11.0+cu130`), `numpy==2.4.3`, `PyYAML==6.0.3`, `scipy==1.16.3`. The production entry points refuse any other versions (`dependency versions differ`).
- The DROID check in G1 also needs `opencv-python-headless==4.13.0.92` and `h5py==3.15.1` (from `po-jepa-full/pyproject.toml`). Install them only if they are missing, with `--no-deps` so pip cannot touch numpy or torch.
- The official-scale V-JEPA 2-AC (`vjepa2_ac`) needs a large card (≥40 GB *(est.)*). In this file it is only read for evaluation (G3 does not include it; G4 and G6 read its v7.1 checkpoints), never trained.
- Shared assets from the v7 work directory, used as they are: `vjepa2/` (upstream code at commit `204698b45b3712590f06245fbfba32d3be539812`) and `checkpoints/vjepa2-vitg-fpc64-256.pth` (SHA-256 `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e`).

All commands below use one variable for the work directory, the folder that already holds `po-jepa/` (v7), `vjepa2/`, `checkpoints/`, `features-v7/` and `results-v7/`:

```bash
WORK=/absolute/path/to/your/work/directory
conda activate pojepa
```

Get the `full-model` code once (a fresh clone, not shared with the v7 add-on work tree), after #8 is merged:

```bash
cd $WORK
[ -d full-model ] || git clone -b full-model https://github.com/ruishenphys/POJEPA.git full-model
cd $WORK/full-model && git pull && git log -1 --oneline
```

## 3. G1: first-GPU checks (~10 minutes, one card)

These confirm that the `full-model` code works on a real GPU. Nothing here trains a model or writes into the v7 directories.

```bash
cd $WORK/full-model/po-jepa-full

# 1. Environment: must show Python 3.12.x, 2.11.0+cu130, 13.0 and your GPU name
python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
# This must print a path inside full-model/po-jepa-full/src (not the v7 copy):
PYTHONPATH=src python -c "import po_jepa; print(po_jepa.__file__)"

# 2. Delivery tests (8 tests, including the 4 that fail on every CPU-only machine). Here all 8 must pass.
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -v tests/test_delivery.py

# 3. GPU eigensolver (S08b): float64 eigh on CUDA must agree with the CPU solver. Expect "1 passed", not "skipped".
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -rs tests/test_droid_s08b.py -k cuda_float64_agrees

# 4. Real-encoder check for DROID encoding (05 §7): one public DROID video, 38,604,360 bytes
python -c "import cv2, h5py" || pip install --no-deps opencv-python-headless==4.13.0.92 h5py==3.15.1
python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"   # must still be 2.4.3 and 2.11.0+cu130
mkdir -p $WORK/gpu-checks
curl -L -o $WORK/gpu-checks/23897859.mp4 \
  "https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1/ILIAD/success/2023-05-12/Fri_May_12_16:48:11_2023/recordings/MP4/23897859.mp4"
ls -l $WORK/gpu-checks/23897859.mp4
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m po_jepa.droid.encode \
  --config configs/droid/pilot.yaml --data-root $WORK/gpu-checks \
  --upstream-root $WORK/vjepa2 --checkpoint $WORK/checkpoints/vjepa2-vitg-fpc64-256.pth \
  --device cuda --check-mp4 $WORK/gpu-checks/23897859.mp4 | tee $WORK/gpu-checks/check_mp4.json
echo "exit=${PIPESTATUS[0]}"
```

Pass criteria:

| Step | Pass |
|---|---|
| 2 | `8 passed`, including the 4 tests below |
| 3 | `1 passed` (a skip means CUDA was not visible) |
| 4 | `exit=0`; the JSON shows `"preprocess_square_equals_v7": true`, `"encode_frames_equals_direct": true` and `"ok": true` for both extra resolutions (192 and 320) |

The 4 tests that every CPU-only full regression reports as failed (for example 3,678 passed, 148 skipped, 4 failed on `full-model` 349f7b6) are all in step 2:

- `tests/test_delivery.py::test_workflow_failure_stops_and_preserves_stage[preflight]`
- `tests/test_delivery.py::test_workflow_failure_stops_and_preserves_stage[training]`
- `tests/test_delivery.py::test_workflow_failure_stops_and_preserves_stage[evaluation]`
- `tests/test_delivery.py::test_workflow_calls_all_phases_once_without_old_stage_records`

They fail on CPU machines because the production entry point checks the environment first (`environment.verify_versions`: Python 3.12.x, torch 2.11.0 built for CUDA 13.0, numpy 2.4.3, PyYAML 6.0.3) and a CPU-only torch has no CUDA version. They do not do GPU computation. In the v7 environment they must pass. If any of them fails with `dependency versions differ`, the environment is not the v7 one; stop and send the output.

Save the terminal output of steps 1–4 as `$WORK/gpu-checks/g1-terminal.txt`. Return it with G2 (§9).

## 4. G2: S22 two-frame tubelet probe (≤1 GPU-hour, est.)

Purpose: decide whether the 5 second-order scenes (pendulum, bouncing_balls, wave, cart_pole, spring_chain) are encoded from two real consecutive frames instead of one duplicated frame. Their formal training waits for this result. Definitions and the decision rule are in `po-jepa-full/docs/segments/S22.md`. Scale: 200 trajectories × 67 tubelets = 13,400 encoder calls; features about 9.7 GB on disk; the fit step peaks at about 10 GB of host RAM. If you already ran S22, skip this and just send the report.

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
tail -n 2 $OUT-encode0.log $OUT-encode1.log        # both should end with the "[tubelet shard i/2]" line that marks the shard as done

# Fit the probe (a few minutes on a GPU)
CUDA_VISIBLE_DEVICES=0 python tools/tubelet_probe.py fit --trajectories 200 --output $OUT --device cuda:0 | tee $OUT-fit.txt
```

- Only one card free: run a single encode command with `--shard 0/1` instead of the two above.
- Interrupted: rerun the same command. Finished trajectories are skipped; half-written ones are redone.
- Out of GPU memory: add `--batch 8` to the encode command (default 34). Use the same value for every shard of one output folder.

## 5. G3: extra evaluation of the v7.1 formal runs (1–3 GPU-hours, est.)

Run it as `EXTRA_EVAL_V7.md` on the `v7` branch says. It only reads finished checkpoints (po_jepa_full, po_jepa_no_consistency, matched_transformer, identity; not the two AC baselines) and computes direct, composed and unit-rollout errors at 1–16 periods. It does not train.

```bash
cd $WORK/po-jepa
git pull                                       # v7 at d1b9ec4 or later; the extra-eval and pack files are not in the manifest
python -m pytest -q tests/test_extra_eval_v7.py # ~10 s, CPU
CUDA_VISIBLE_DEVICES=0 python tools/extra_eval_v7.py run --results-root ../results-v7/full/v7.1 \
    --features-base ../features-v7 --output ../extra-eval-v7-r2/full --max-test-trajectories 50 --device cuda:0
python tools/extra_eval_v7.py summarize --output ../extra-eval-v7-r2/full
```

- If the pilot part (`--results-root ../results-v7/pilot --output ../extra-eval-v7-r2/pilot`) has not been run yet, run it as well; it takes 10–20 minutes on a GPU.
- Outputs of the first version of the tool (commit a305bfe, folder `extra-eval-v7/`) are not valid; the tool refuses them. Leave that folder as it is.
- Interrupted: rerun the same command; finished tasks with a matching identity are skipped.

## 6. G4: small validation, B layer on the v7.1 checkpoints (`b-v7.1`, evaluation only)

Full instructions: `po-jepa-full/SMALL_VALIDATION_GUIDE.md` §0–§2 (in Chinese). It reads the v7.1 formal checkpoints (trained by the `v7` branch code) with `--cross-version`: the tool checks the data provenance and the config snapshot and stops without writing output if they do not match, so it cannot read the wrong model.

```bash
cd $WORK/full-model/po-jepa-full
python tools/make_small_validation_jobs.py --stage b-v7.1 > jobs-sv-b-v7.1.txt
python tools/run_jobs.py jobs-sv-b-v7.1.txt --gpus 0,1
```

What the 139 commands do (scenes advection_diffusion, double_well, van_der_pol):

| Step | Commands | Device | What |
|---|---|---|---|
| L1 | 12 | GPU | encode the long-horizon test sets (horizon 128, test split only, 200 trajectories × 129 frames per scene, 4 shards) into `../../features-small-validation/` |
| L2 | 3 | CPU | finalize the encodings |
| R0 | 3 | GPU | action probe: how much the true endpoint features change when the action changes (~10 min each) |
| R1 | 3 | GPU | fit the frozen linear state probes on the validation split (~10 GB host RAM for the full readout) |
| R2 | 3 | CPU | ground truth for the counterfactuals (re-simulation) |
| R3 | 108 | GPU | long-horizon physical readout (VPT) and counterfactual actions on 6 conditions × 3 seeds × 3 scenes, including the official-scale AC |
| R4 | 6 | GPU | modal frequency comparison on advection_diffusion |
| R5 | 1 | CPU | summary |

- Needs: `../../results-v7/full/v7.1/` with its checkpoints, `../../features-v7/`, `../../vjepa2/`, `../../checkpoints/`.
- GPU time *(est.)*: L1 about 1 GPU-hour (77,400 frames at the measured ~37 frames/s/card, plus rendering); R0 + R1 about 1 GPU-hour; R3 + R4 were not measured (each readout is 16 forward passes of 8 ticks), rough guess 3–10 GPU-hours.
- R1, R2, L1 and L2 are reused by C-2 (G6), so do not delete `../../features-small-validation/` or `../../results-small-validation/readout/`.
- Interrupted: rerun the same `run_jobs.py` command.

## 7. G5 and G6: calibration, checks and evaluation lines of C-1 and C-2

These lines sit in the same job files as the C-1 and C-2 training lines. `run_jobs.py` runs the whole file; you do not have to separate them. They are listed here so the GPU time outside training is visible.

### G5: C-1 (`c1` job file, ready after #8)

```bash
cd $WORK/full-model/po-jepa-full
python tools/make_small_validation_jobs.py --stage c1 --v6-features "$V6" > jobs-sv-c1.txt
python tools/run_jobs.py jobs-sv-c1.txt --gpus 0,1
```

`$V6` is the absolute path of the v6 `double_well-200` feature folder used in the v7 pilot (the `V6=` line in the v7 handoff). Non-training lines: 9 × `run.py calibrate` (the 3 non-default variants × 3 scenes; the default variant uses the calibration in git) and 3 × `run.py check` (one per non-default variant on van_der_pol: dependencies, device, full gradient, 100-step cost), about 2–4 GPU-hours *(est.)*. The other 17 lines are 12 short training runs (200 trajectories × 2,000 updates, only `po_jepa_full`) and 5 CPU summaries.

PR #8 matters here because its summary fixes decide which C-1 cells may vote; without it the C-1 recommendation can be wrong. After C-1, send back the pack (§9) and wait: Jerry confirms the variant for C-2.

### G6: C-2 (`c2` job file, blocked)

Blocked until: (1) the C-1 summary is back and Jerry has confirmed the variant; (2) G4 steps L1, L2, R1, R2 are finished; (3) PR #8 is merged; (4) if the variant uses `--temporal-partition dyadic`, PR #7 is merged as well, because the `po_jepa_rollout_sup` loss and its λ_r calibration change under that variant. Jerry will send the exact command, of the form:

```bash
python tools/make_small_validation_jobs.py --stage c2 --temporal-partition <bisection|dyadic> --readout-init <xavier|zero> > jobs-sv-c2.txt
python tools/run_jobs.py jobs-sv-c2.txt --gpus 0,1
```

Non-training lines (63 of 84; counted from the tool's output): 3 × `calibrate` (σ_z, λ_t, λ_s on 2,000 trajectories), 3 × `calibrate-addon` (λ for transformer_consistency and fixed_fourier, λ_r for rollout_sup), `check` + `check-addon`, 3 identity jobs (no training; they must match v7.1 bit for bit so the v7.1 Transformer and AC results can be reused), 12 × `equal-step` (evaluation only, cut at the step where po_jepa_full stopped), 30 readouts and counterfactuals, 3 modal spectra, 2 summaries. About 10–20 GPU-hours *(est.)*. The other 15 lines are the C-2 training runs (2,000 trajectories × 10,000 updates), about 50–60 GPU-hours *(est.)*.

## 8. G7: evaluation tasks that cannot run yet

| Task | Why it is blocked |
|---|---|
| Simulation add-on extra evaluations and D3 (official-scale AC planning, 256 samples, ~7.5 h *(est.)*) | The add-on code reads `results-v7/full/<scene>` from v7 0ac45fe, but v7.1 writes `results-v7/full/v7.1/`; a code segment has to adapt it first |
| Evaluation-only stages of the simulation dataset (`variants`, `strength`, `long`, `timing` = frame-rate and timing changes, `space`, `readout` in `tools/make_v8_jobs.py`) | They read the formal-training checkpoints; planned for the advisor's cluster after the formal training |
| S25b: re-encoding the 5 second-order scenes from two frames | Waits for the G2 result and a code segment |
| DROID encoding, `check/env`, calibration, evaluation, and the S16 five-episode smoke test | Waits for decisions (DROID in the submission, cluster information) and segments S15/S16 (`configs/droid/pilot.yaml` now sets `resources.train.time_limit_minutes: 20`, C9) |

## 8b. G8: AC checks (C7, `tools/ac_context_check.py`)

Evaluation only; it changes no training or evaluation code and is not part of any preregistration. **Pull the commit that adds it only after every job file in that clone (`c1`, `b-v7.1`) is finished**, as for any other pull (§1).

- `context`: on the 16×16 test points of one AC job, recomputes the direct prediction with the context cut to the last W ∈ {1, 2, 4, 8} frames (AC was trained on 9-frame clips, so at most 8 context frames; the 12-cycle direct prediction uses up to 12), next to the full-context prediction (identical to the stored direct prediction) and the v7.1 step-by-step rollout. Question it answers: is "12-cycle rollout 0.094 < 12-cycle direct 0.154" caused by the context going beyond 8 frames?
- `floor`: re-renders the first 8 test trajectories of one scene, checks the image hashes against the cache, re-encodes them with the same frozen encoder and preprocessing as `encode_v7.py`, and reports the distance to the cached features (and its parts: run-to-run, batch size, float16 storage), next to the distance between adjacent frames. Same metric as `future_jepa_distance`.

```bash
cd $WORK/full-model/po-jepa-full
R=../../results-v7/full/v7.1; F=../../features-v7; O=../../results-small-validation/ac-check
for s in advection_diffusion double_well van_der_pol; do
  for c in vjepa2_ac vjepa2_ac_small; do
    python tools/ac_context_check.py context --job $R/$s/seed1703/$c --cross-version --features $F/$s-2000 \
        --ac-checkpoint ../../checkpoints/vjepa2-ac-vitg.pt --output $O/context/$s/seed1703/$c.json --device cuda:0
  done
  python tools/ac_context_check.py floor --system $s --trajectories 2000 --features $F/$s-2000 \
      --upstream-root ../../vjepa2 --checkpoint ../../checkpoints/vjepa2-vitg-fpc64-256.pth \
      --output $O/floor/$s.json --device cuda:0
done
```

Each command refuses to overwrite an existing `--output` and writes it only at the end (via a `.partial` file), so after a failure just rerun that line. Send back `results-small-validation/ac-check/` together with the G4–G6 pack (§9; the pack command there already includes `results-small-validation`).

## 8c. G9: C-2 shrinkage check (`tools/shrinkage_check.py`)

Evaluation only, no training; it reads the selected checkpoint of each C-2 job (the same one the C-2 evaluation used) and changes no training or evaluation code. C-2 is finished, so pulling the commit that adds it is safe (§1 still applies to anything else running in that clone).

Question it answers: on double_well the full model's future JEPA distance is about equal to `po_jepa_no_consistency`, but its physical readout error is ~40% worse. Are its predictions shrunk toward the mean or toward "nothing moves"? On the same test windows as `state_readout.py counterfactual` (each test trajectory × registered start × T ∈ {1, 2, 4, 8}, 16×16), with the same probe and truth file, it reports per T: readout spread ratio and regression slope of prediction on truth, the share of the true displacement that is predicted (motion gain), the same in LayerNorm feature space, and raw feature norms. Formulas are in the file header.

```bash
cd $WORK/full-model/po-jepa-full
R=../../results-small-validation; C=$R/c2/dyadic-zero; F=../../features-v7; O=$C/shrinkage
for s in advection_diffusion double_well van_der_pol; do
  for c in identity po_jepa_full po_jepa_no_consistency po_jepa_rollout_sup fixed_fourier transformer_consistency; do
    python tools/shrinkage_check.py --job $C/$s/seed1703/$c --cross-version --probe $R/readout/probe/$s \
        --truth $R/readout/truth/$s.json --features $F/$s-2000 --output $O/$s/seed1703/$c.json --device cuda:0
  done
  python tools/shrinkage_check.py --job ../../results-v7/full/v7.1/$s/seed1703/matched_transformer --cross-version \
      --probe $R/readout/probe/$s --truth $R/readout/truth/$s.json \
      --features $F/$s-2000 --output $O/$s/seed1703/v7.1_matched_transformer.json --device cuda:0
done
```

`--cross-version` is required on every line: the C-2 jobs were trained on d5a8fc1, and the commit that adds this tool changes `MANIFEST.json`, so the default loader would refuse them with `training run 的预登记与当前代码/配置不一致`. The cross-version loader checks the job record, selected step, query sequence, checkpoint SHA, configuration snapshot and data provenance instead of the source hashes. Each command refuses to overwrite an existing `--output`; after a failure just rerun that line. Send back `results-small-validation/c2/dyadic-zero/shrinkage/` (a few MB) with the pack command of §9, for example `python $TOOL pack results-small-validation/c2/dyadic-zero/shrinkage nvidia-smi.txt --output c2-shrinkage.pack.sqlite`.

## 9. What to send back

Use the de-duplicating pack tool `tools/pack_results.py` (only the Python standard library; `.pt` weights are skipped by default), not zip. Run it in `$WORK`. Every path you list must exist; drop the ones that have not been produced yet. The last two lines must be `verified N files: …` and `wrote ….pack.sqlite: … MB`. Send the `.pack.sqlite` files to Jerry as they are.

```bash
cd $WORK
nvidia-smi > nvidia-smi.txt
TOOL=full-model/po-jepa-full/tools/pack_results.py

# G1 + G2 (small)
python $TOOL pack gpu-checks/g1-terminal.txt gpu-checks/check_mp4.json \
    features-v8/tubelet-probe-200/report.json features-v8/tubelet-probe-200-fit.txt \
    features-v8/tubelet-probe-200-encode0.log features-v8/tubelet-probe-200-encode1.log nvidia-smi.txt \
    --output gpu-checks.pack.sqlite

# G3 (a few MB)
python $TOOL pack extra-eval-v7-r2 nvidia-smi.txt --output extra-eval-v7-r2.pack.sqlite

# G4, G5, G6 (after each job file is finished; same command, it includes whatever exists)
python $TOOL pack results-small-validation logs-addon/jobs-sv-* \
    full-model/po-jepa-full/configs/calibration full-model/po-jepa-full/jobs-sv-*.txt nvidia-smi.txt \
    --output results-small-validation.pack.sqlite
```

- `configs/calibration` must be included: the non-default variant calibrations (`*~dyadic*.yaml`, `*~zeroinit.yaml`) and the add-on calibrations under `addon/` are written there during the runs.
- `logs-addon/jobs-sv-*` is the default log folder of `run_jobs.py` for these job files.
- Interrupted packing leaves only `.pack.sqlite.partial`; rerun the same command. More options in `po-jepa-full/RESULTS_RETURN_PACK.md`.

If anything fails, stop that task, keep its folder as it is and send the full terminal output; the other tasks can continue.
