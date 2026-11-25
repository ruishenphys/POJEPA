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
| G11 | C-3 mode-count ablation and scheme B (`c3` job files, batch 1 and batch 2): new PO-JEPA variants that change only the number of spatial basis vectors and never drop information (K = 16, 32, 64 plus the kept complement; all tokens) | **Ready once its pull request is merged**; batch 2 only after Jerry looks at batch 1 | batch 1 ~55 GPU-hours, batch 2 ~25 GPU-hours *(est.)* | C-2 results under `results-small-validation/c2/dyadic-zero/` (reference rows, **keep them**); readout probes and truth files (G4) |
| G10 | C-2 truncation check (`tools/truncation_check.py`): oracle ceilings of the PO-JEPA output subspace, the C-2 Transformer's change projected onto the 16 graph modes, and the trained PO-JEPA attention and right-hand-side sizes, on the registered test windows of all three grids | **Ready once its pull request is merged** | ~3–6 GPU-hours *(est.)*, 21 commands | C-2 checkpoints under `results-small-validation/c2/dyadic-zero/` (G6); **please keep them** |
| G12 | C-1 2×2 readout (§8f): physical readout and counterfactual action response of the 12 C-1 checkpoints (bisection/dyadic × Xavier/zero `P_out`), evaluation only, probes and truth fitted on the 200-trajectory datasets | **Ready** (`full-model` c379bc4 or later; #14 and #15 are merged) | ~1 GPU-hour *(est.)*, 48 commands (12 of them optional) | C-1 job folders under `results-small-validation/c1/` with their selected checkpoints; the v6 `double_well-200` features (`$V6`, as for C-1) |
| G13 | Second-order scenes in the formal-training job file (§8g): `tools/make_v8_jobs.py` now lists all 12 scenes by default, the 5 second-order scenes encoded from two frames (`<scene>~tubelet=pair`) | **Nothing to run now**: it runs with the formal training of the simulation dataset (advisor's cluster) | part of the formal training; the 5 scenes add about 70% to every stage, see §8g | formal training; `full-model` with S25b4 merged |
| G14 | C-1-scale isolation of the double_well "moves too little" shrinkage (§8h): `po_jepa_no_consistency`, temporal-only and spatial-only consistency, two more seeds of the key cells, and a λ_t scan (×0.5, ×0.25), all 200 trajectories × 2,000 updates on double_well and van_der_pol; then action response and motion gain | **Ready once its pull request is merged**; pull only after the C-3 batch-1 remainder in that clone has finished | ~10–15 GPU-hours *(est.)*: 29 short training runs + 66 evaluation lines, ~6–8 hours on two cards | the C-1 clone (its 9 `*-200-u2000~*.yaml` calibrations, the C-1 job folders) and the G12 probes and truth files under `results-small-validation/c1-readout/` |
| G15 | C-3 extra training with the dyadic-xavier fix (§8i): `Kfull` and `K16c` × dyadic temporal partition × Xavier-initialised `P_out`, one `po_jepa_full` per variant and scene (6 runs), with calibration, readout, modal check, G9 shrinkage check and summary (`c3-xavier` job file, 36 commands); plus sending back the batch-1 checkpoints for the `K16c` divergence analysis | **Ready once its pull request is merged**; run it **after** the C-3 batch-1 remainder (§8e) and only then pull | ~20–24 GPU-hours *(est.)*, ~11 hours on two cards | readout probes and truth files under `results-small-validation/readout/` (G4) |

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
| S25b: encoding the 5 second-order scenes from two frames | **Code done** (S25b, S25b2, S25b3, S25b4); it is part of the formal-training job file now, see §8g (G13) |
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


## 8d. G10: C-2 truncation check (`tools/truncation_check.py`)

Evaluation only, no training; it reads the selected checkpoint of each C-2 job (the same one the C-2 evaluation used) and changes no training or evaluation code. Like G9 it needs `--cross-version` on every line (the C-2 jobs were trained on d5a8fc1 and this commit changes `MANIFEST.json`). **Please do not delete the C-2 checkpoints**: G10 and later diagnostics read them.

Question it answers: the PO-JEPA change is `Z_hat − Z = Ψ_16 (C_τ − C_0) W_outᵀ`, so it can only use the 16 lowest graph modes of the start frame. Is that what keeps its direct prediction below the Transformer's? On the C-2 test windows (each test trajectory × registered start × T ∈ {1, 2, 4, 8}, grids 16×16, 12×12, 8×8, the same points as the C-2 evaluation), with the evaluation metric (per-token LayerNorm, mean absolute difference):

- `ceiling`: how far the identity error can be reduced by *any* change in a given output subspace when the true future frame is known (Adam, 6000 steps; the optimised value is a lower bound of the true ceiling): the 16 graph modes, the 16 modes with the trained PO-JEPA readout, 64 modes, and a per-token change in a 64-dim channel space (the Transformer's form). The heaviest step (est. 40–80 min per scene).
- `project`: the trained `transformer_consistency` prediction, with its change projected onto those 16 modes (and onto 64), scored again; next to `po_jepa_full` on the same windows. A drop after the projection shows that what the Transformer learned uses the removed directions; it is not by itself a proof that the truncation explains the whole gap (the projection is not the best point of the subspace under this metric). It also re-scores the stored C-2 evaluation rows and reports the largest difference (`evaluation_check`, expected ≈ 0).
- `dynamics`: for the trained PO-JEPA variants, the top-r attention (max weight, entropy) and the sizes of the terms of dC/dt at every integration step.

```bash
cd $WORK/full-model/po-jepa-full
C=../../results-small-validation/c2/dyadic-zero; F=../../features-v7; O=$C/truncation
for s in advection_diffusion double_well van_der_pol; do
  J=$C/$s/seed1703
  python tools/truncation_check.py ceiling --job $J/po_jepa_full --cross-version --features $F/$s-2000 \
      --output $O/ceiling/$s.json --device cuda:0
  python tools/truncation_check.py project --job $J/transformer_consistency --spectrum-job $J/po_jepa_full --cross-version \
      --features $F/$s-2000 --output $O/project/$s/transformer_consistency.json --device cuda:0
  for c in po_jepa_full po_jepa_no_consistency po_jepa_rollout_sup fixed_fourier; do
    python tools/truncation_check.py dynamics --job $J/$c --cross-version --features $F/$s-2000 \
        --output $O/dynamics/$s/$c.json --device cuda:0
  done
  python tools/truncation_check.py project --job $J/po_jepa_no_consistency --spectrum-job $J/po_jepa_full --cross-version \
      --features $F/$s-2000 --output $O/project/$s/po_jepa_no_consistency.json --device cuda:0
done
```

The `ceiling` lines can run on the second card (`--device cuda:1`) while the other lines run on the first. Each command refuses to overwrite an existing `--output` and writes it only at the end (via a `.partial` file), so after a failure just rerun that line. `ceiling` keeps one duration's windows (400 × 2 frames, ~1.2 GB) in CPU memory and about 3 GB on the GPU with the default `--batch 100`; lower `--batch` if memory is short (the result does not depend on it). Send back `results-small-validation/c2/dyadic-zero/truncation/` (well under 10 MB), for example `python $TOOL pack results-small-validation/c2/dyadic-zero/truncation nvidia-smi.txt --output c2-truncation.pack.sqlite` (§9).

## 8e. G11: C-3 mode-count ablation and scheme B (`c3` job files)

This one **trains**: 18 new PO-JEPA runs in two batches (plan `small-validation-results/c3-plan/2026-11-22_mode-ablation-plan.md`, decided by Jerry on 2026-11-22). All of them are the C-2 setting `dyadic-zero` (dyadic temporal partition, zero-initialised `P_out`), 2,000 trajectories × 10,000 updates, seed 1703, on the three C-2 scenes; only the spatial part changes, through two new `run.py` options:

| Variant | `--modes` | `--complement` | What it is |
|---|---|---|---|
| `K16c` | `16` | `shared` | scheme B: the 16 lowest graph modes as now, plus the whole orthogonal complement kept as a token field that evolves with one shared damped rotation `A(λ_⊥, g)` (λ_⊥ = mean of the cut eigenvalues). No information is dropped. |
| `Kfull` | `full` | `none` | every grid uses all its tokens as modes (K = 256 / 144 / 64 on 16×16 / 12×12 / 8×8) |
| `K32c` | `32` | `shared` | scheme B with K = 32: 32 lowest modes plus the kept complement. No information is dropped. |
| `K64c` | `64` | `shared` | scheme B with K = 64. On 8×8 (64 tokens) the complement is empty, so it is the same as all modes there. No information is dropped. |

Every variant keeps all the information (Jerry, 2026-11-22: only the number of explicit basis vectors changes). The truncated K16 from C-2 stays in the summary as the information-losing reference.

K16 (the current model) and `transformer_consistency` are **not retrained**: the summary reuses the C-2 runs, after checking that the batch-1 `identity` evaluation matches C-2 bit for bit. **Please keep the C-2 job folders.** The parameter count is unchanged (227,617) for every variant.

Pulling: the commit that adds this changes the model code and `MANIFEST.json`. Pull only after every job file running in that clone is finished (§1). The C-2 diagnostics G9 and G10 already use `--cross-version`, so they still read the C-2 checkpoints after the pull.

```bash
cd $WORK/full-model/po-jepa-full
git pull
python tools/make_small_validation_jobs.py --stage c3 --batch 1 > jobs-sv-c3-b1.txt
python tools/run_jobs.py jobs-sv-c3-b1.txt --gpus 0,1
```

Batch 1 (`K16c`, `Kfull`; 72 commands): 6 × `calibrate` (σ_z, λ_t, λ_s per variant and scene; the files are written as `configs/calibration/<scene>-2000-u10000~dyadic~zeroinit~K16c.yaml` / `~Kfull.yaml`), 2 × `check`, 12 training runs (`po_jepa_full` and `po_jepa_no_consistency` for each variant and scene) + 3 `identity` evaluations, 6 × `equal-step`, 24 readouts and counterfactuals + 4 modal spectra, 12 × `shrinkage_check.py` (the G9 check on the new runs), 3 summaries. About 55 GPU-hours *(est.)*, ~28 hours on two cards. `Kfull` may take 0–20% longer per run than C-2 (not measured).

Send batch 1 back (§9) and wait: Jerry decides from it whether batch 2 is needed. Only then:

```bash
python tools/make_small_validation_jobs.py --stage c3 --batch 2 > jobs-sv-c3-b2.txt
python tools/run_jobs.py jobs-sv-c3-b2.txt --gpus 0,1
```

Batch 2 (`K32c`, `K64c`, only `po_jepa_full`; 31 commands): 6 × `calibrate`, 2 × `check`, 6 training runs, 12 readouts + 2 modal spectra, 3 summaries. About 25 GPU-hours *(est.)*, ~13 hours on two cards (the kept complement makes each run about as heavy as `Kfull`).

Results go to `results-small-validation/c3/dyadic-zero-<variant>/`; the joint summary (`summarize_small_validation.py c3`) to `results-small-validation/c3/summary/`. Pack command (§9 has the general rules; `configs/calibration` must be included for the new `~K*` files):

```bash
cd $WORK
python $TOOL pack results-small-validation/c3 logs-addon/jobs-sv-c3-* full-model/po-jepa-full/configs/calibration \
    full-model/po-jepa-full/jobs-sv-c3-*.txt nvidia-smi.txt --output c3-batch1.pack.sqlite
```

(For batch 2 use `--output c3-batch2.pack.sqlite`; the same command picks up both batches.)

### Batch 1 remainder (2026-11-25): run only the missing sections, in the same clone, without pulling

Batch 1 stopped in section D on 2026-11-24 (19 of 72 done): `K16c` `po_jepa_full` diverged during training on `double_well` (step 3336) and `van_der_pol` (step 5927), and the evaluation of the two `van_der_pol` `po_jepa_no_consistency` runs failed in the closed-loop rollout (`eigh` did not converge). `run_jobs.py` stops at the first section with a failure, so D2, R3, R4, G9 and E never started.

**Do not rerun the whole job file**: it would first retry the two diverged trainings, which are deterministic and would diverge again at the same step (about 3 GPU-hours wasted). Run only the evaluation sections, one command per section, so that a failure in one section does not block the next:

```bash
cd $WORK/full-model/po-jepa-full        # the same clone that ran batch 1 (0d35288); do NOT git pull before this is finished
for s in R3 R4 G9 E; do python tools/run_jobs.py jobs-sv-c3-b1.txt --gpus 0,1 --only $s; done
```

- Use the existing `jobs-sv-c3-b1.txt`; do not regenerate it. These lines read checkpoints trained by this clone and have no `--cross-version`, so they must run **before** the pull for §8i (after the pull they would stop with `delivery source changed`).
- Expected failures, which are fine: the lines that read `K16c` `po_jepa_full` on `double_well` and `van_der_pol` (no checkpoint), and possibly the `van_der_pol` `po_jepa_no_consistency` lines. Everything else should finish. `run_jobs.py` prints the failed log paths at the end of each section; keep them.
- Each `for` step ends with either `全部完成。` or `第 … 段有任务失败，已停下`; both are fine here, the loop goes on to the next section.
- D2 (6 equal-step evaluations, ~7 GPU-hours *(est.)*) is **skipped**: the `Kfull` full runs trained the full 10,000 updates without early stopping, so it adds little.
- R3 (24 commands, ~3 min each), R4 (4), G9 (12, ~1.5 min each), E (3): about 2 GPU-hours *(est.)*.

Send back with the batch-1 pack command above, but `--output c3-batch1b.pack.sqlite`.

## 8f. G12: C-1 2×2 readout (physical readout and action response, evaluation only)

Evaluation only, no training, no new code: it runs the existing `tools/encode_v7.py` and `tools/state_readout.py` (the G4 tools) on the 12 C-1 checkpoints. Like G9 and G10, every readout line uses `--cross-version` (the C-1 jobs were trained on an older commit and later commits changed `MANIFEST.json`), so it runs on the current `full-model` and also after #14 (G10) and #15 (C-3) are merged (C-3 leaves the default model and its configuration snapshot unchanged).

Question it answers: in C-2 (`dyadic-zero`), `po_jepa_full` on double_well responds to a change of actions only about half as much as `po_jepa_no_consistency` (counterfactual amplitude 0.18 against 0.33), but the v7.1 `po_jepa_full` (bisection + Xavier) does not (0.34). The two changes between them are the dyadic temporal partition and the zero-initialised `P_out` (S28b). C-1 trained all four combinations (200 trajectories × 2,000 updates, seed 1703), but never ran the readout on them. This task does, so we can tell which of the two changes causes the shrinkage.

Because C-1 used the 200-trajectory datasets, the probes and truth files of G4 (fitted on the 2,000-trajectory datasets) cannot be reused: the tools refuse a probe or truth file of another dataset. So the job file first encodes a 200-trajectory long test set (only its 20 test trajectories × 129 frames), fits a probe on the 200-trajectory validation split, and writes the truth files, all under `results-small-validation/c1-readout/`. The `double_well-200` features are the v6 folder that C-1 read (`$V6`, the same absolute path as in §7 G5).

```bash
cd $WORK/full-model/po-jepa-full
V6=/absolute/path/of/the/v6/double_well-200/features    # the same value you used for the c1 job file (§7 G5)
R=../../results-small-validation; O=$R/c1-readout; LF=../../features-small-validation
ENC="--upstream-root ../../vjepa2 --checkpoint ../../checkpoints/vjepa2-vitg-fpc64-256.pth"
SCENES="advection_diffusion double_well van_der_pol"
VARIANTS="bisection-xavier bisection-zero dyadic-xavier dyadic-zero"
feats() { if [ "$1" = double_well ]; then echo "$V6"; else echo "../../features-v7/$1-200"; fi; }
long() { echo "$LF/$1~horizon=128-200"; }
readout() {  # $1 job folder, $2 scene, $3 output group, $4 condition
  P="--probe $O/probe/$2 --truth $O/truth/$2.json"
  echo "python tools/state_readout.py counterfactual --job $1 $P --features $(feats $2) --cross-version --output $O/$3/counterfactual/$2/seed1703/$4.json --device cuda:0"
  echo "python tools/state_readout.py rollout --job $1 $P --features $(long $2) --horizon 128 --base-features $(feats $2) --cross-version --output $O/$3/rollout/$2/seed1703/$4.json --device cuda:0"
}
{
echo "# ===== L1 encode the 200-trajectory long test sets (GPU, 3 commands: 20 test trajectories x 129 frames per scene) ====="
for s in $SCENES; do echo "python tools/encode_v7.py encode --system $s --variant horizon=128 --trajectories 200 --splits test $ENC --output $(long $s) --device cuda:0 --shard 0/1"; done
echo "# ===== L2 finalize the encodings (CPU, 3 commands) ====="
for s in $SCENES; do echo "python tools/encode_v7.py finalize --system $s --variant horizon=128 --trajectories 200 --splits test --output $(long $s)"; done
echo "# ===== R1 fit the probes on the 200-trajectory validation split (GPU, 3 commands) ====="
for s in $SCENES; do echo "python tools/state_readout.py fit --system $s --trajectories 200 --features $(feats $s) --output $O/probe/$s --device cuda:0"; done
echo "# ===== R2 truth files (CPU, 3 commands; counterfactual ends re-simulated) ====="
for s in $SCENES; do echo "python tools/state_readout.py truth --system $s --trajectories 200 --features $(feats $s) --long-features $(long $s) --horizon 128 --output $O/truth/$s.json"; done
echo "# ===== R3 counterfactual and long rollout of the C-1 2x2 checkpoints (GPU, 24 commands, independent) ====="
for v in $VARIANTS; do for s in $SCENES; do readout $R/c1/$v/$s/seed1703/po_jepa_full $s $v po_jepa_full; done; done
echo "# ===== P optional reference: v7 pilot (200 x 2000, bisection-xavier, v7 code) full and no-consistency (GPU, 12 commands) ====="
for c in po_jepa_full po_jepa_no_consistency; do for s in $SCENES; do readout ../../results-v7/pilot/$s/seed1703/$c $s v7-pilot $c; done; done
} > jobs-sv-c1-readout.txt
grep -c '^python' jobs-sv-c1-readout.txt      # must print 48
python tools/run_jobs.py jobs-sv-c1-readout.txt --gpus 0,1
```

- Sections: L1 (3) → L2 (3) → R1 (3) → R2 (3) → R3 (24) → P (12). `run_jobs.py` starts a section only after the previous one has finished, logs go to `../../logs-addon/jobs-sv-c1-readout/`, and rerunning the same command skips finished lines. Each readout refuses to overwrite an existing `--output`.
- Run it in the clone where C-1 ran (the one used for C-2). The readout rebuilds each model with the λ of its variant, read from `configs/calibration/<scene>-200-u2000~dyadic.yaml`, `~zeroinit.yaml` and `~dyadic~zeroinit.yaml` (9 files written by the C-1 `calibrate` lines, not in git); the cross-version loader compares that configuration with the checkpoint and refuses on any difference. In a fresh clone, copy those 9 files over first.
- **Section P is optional.** It reads the v7 pilot jobs (`../../results-v7/pilot/<scene>/seed1703/po_jepa_full` and `po_jepa_no_consistency`, trained on the v7 branch before v7.1) as a same-scale reference without the dyadic / zero changes, and the only same-scale no-consistency run we have. If that folder is gone, or its lines fail (for example `跨版本读检查点没通过`), just leave them failed and send the log; R3 is what matters. It is the last section, so a failure there blocks nothing.
- If an R3 line stops with `跨版本读检查点没通过` or a missing checkpoint, the C-1 selected checkpoint was probably cleaned up; please tell us which variants still have `run/job/training/po_jepa_full/` with its checkpoint file.
- Time *(est., scaled from the G4 runs: a 2,000-trajectory counterfactual took ~170 s and a rollout ~300 s; here each has 10× fewer test trajectories)*: L1 a few minutes per scene, R1/R2 1–2 minutes each, each R3/P line ~1 minute plus model loading. About 1 GPU-hour in total, well under an hour on two cards. Disk: ~2 GB of new features per scene (20 × 129 frames, float16) under `features-small-validation/`.

Order with the other tasks tonight: **G9 → G10 → G12 → G11 batch 1 (C-3)**. G9 and G10 need no pull if that clone already has them. G12 is cheap and its answer may change how we read C-3 (all C-3 variants are `dyadic-zero`), so run it right after pulling the C-3 merge, before starting the C-3 batch-1 job file (~55 GPU-hours, ~28 hours on two cards) in the same clone; once C-3 batch 1 is running, do not pull again (§1). If C-3 is not merged yet when you get to it, G12 can also run on the `full-model` commit you already have (d5a8fc1 or later): the readout tools and `--cross-version` are unchanged since C-2, and it does not depend on #14 or #15.

Send back (a few MB plus logs):

```bash
cd $WORK
python $TOOL pack results-small-validation/c1-readout logs-addon/jobs-sv-c1-readout \
    full-model/po-jepa-full/jobs-sv-c1-readout.txt nvidia-smi.txt --output c1-readout.pack.sqlite
```

Do not include `features-small-validation/` in the pack (only the readouts are needed).

## 8g. G13: second-order scenes in the formal-training job file (two-frame encoding, S25b4)

Nothing to run now. This section records what changed in the formal-training job file of the simulation dataset, so that whoever generates it later gets the two-frame scenes.

- `tools/make_v8_jobs.py` without `--scenes` now lists **12 scenes**: the 7 first-order ones as before, plus `pendulum~tubelet=pair`, `bouncing_balls~tubelet=pair`, `wave~tubelet=pair`, `cart_pole~tubelet=pair`, `spring_chain~tubelet=pair` (`po_jepa.protocols.DEFAULT_SCENES`).
- `tubelet=pair` means every feature is the frozen V-JEPA 2 encoding of (the frame 1 tick = 0.25 s earlier, this frame), so velocity is visible (G2: θ̇ readout R² 0.962 with two frames, 0.003 with one duplicated frame). The encoder cost per feature is the same as before; each trajectory renders one extra frame (t = −1).
- The knob travels with the scene name, so every stage is two-frame: `--variant tubelet=pair` on encoding, training, calibration and probes; feature and result folders are named `<scene>~tubelet=pair` (test sets `<scene>~<test knob>,tubelet=pair`); `run.py transfer`, `timing_eval.py` and `space_eval.py` add their `--test-variant` to the training knobs (so `--test-variant` never contains `tubelet`); the V4 planning lines plan on the base scene and read the dataset from `--job`.
- The old one-frame (duplicated) encoding of the second-order scenes is still available as a control, only when asked for explicitly (`--scenes pendulum,...`); the job file then starts with a warning line.
- The first-order scenes and their lines are unchanged.

Commands per stage (counted with `grep -c '^python'` on the tool's output; 2,000 trajectories, 3 seeds; `pilot` is 200 trajectories, 1 seed; `addon` lists all 6 add-on conditions):

| Stage | 7 first-order scenes | 5 two-frame second-order scenes | Default (12 scenes) |
|---|---|---|---|
| pilot | 86 | 62 | 146 |
| full | 177 | 127 | 302 |
| variants | 945 | 614 | 1558 |
| strength | 1673 | 1112 | 2784 |
| long | 162 | 116 | 277 |
| timing | 967 | 691 | 1657 |
| space | 967 | 691 | 1657 |
| readout | 272 | 202 | 474 |
| addon | 135 | 97 | 230 |

(The default column is slightly less than the sum: the check and summary lines are shared.) The `full` stage has 12 × 3 × 6 = 216 training runs, matching the formal-training plan.

Not decided yet (Jerry): whether the planning success criterion of the second-order scenes should also count velocity (now position only), and whether the readout should add acceleration. Until then the job file uses the current position-only criteria.

## 8h. G14: what causes the double_well shrinkage (C-1 scale, trains 29 short runs)

Question: in C-2 (`dyadic-zero`) and in C-1, `po_jepa_full` on double_well predicts only about half the motion of `po_jepa_no_consistency` and responds to actions much less (G9, G12). G12 found it only in the dyadic + zero-initialised `P_out` cell. Is the consistency loss itself what pulls the model toward "move less", and which of its two terms? Is it one seed? Does a smaller λ_t remove it (advisor's suggestion, 2026-11-25; Jerry chose to scale λ_t only, λ_s stays at its calibrated value)?

Facts that shaped the design (Claude, CPU-checked on the current code):
- The temporal partition only enters through the temporal-consistency loss. With both weights zero (`po_jepa_no_consistency`), bisection and dyadic give the same JEPA loss bit for bit and the same gradient up to float32 rounding (`tests/test_g14_weight_scale.py`). So "no consistency" is trained once per init (`bisection-*` folder names), and one `dyadic-zero` no-consistency run on double_well is kept only as a same-setting second realisation (a noise reference).
- The four C-1 `po_jepa_full` cells (seed 1703) already exist; they are reused, not retrained. Only their G9 motion gain is new (section X, `--cross-version`).

New option (this pull request): `run.py --time-weight-scale {1,0.5,0.25}` (and `--space-weight-scale`, unused here) multiplies the λ_t of the variant's own calibration file; the calibration file name does not change, `calibrate` refuses the option, only `po_jepa_full`, `po_jepa_time_only` and `po_jepa_space_only` accept it. The scale is written into `job.json` (`variant.time_weight_scale`), the evaluation tools restore it, and `summarize_v7.py` groups such runs as `<dataset>~dyadic~zeroinit~lt0.5`. Without the option nothing changes.

The 29 training runs (all `--trajectories 200 --updates 2000`, the C-1 setting, K = 16 as in C-1):

| Folder (`results-small-validation/c1-isolation/`) | Condition | Seeds | Scenes | Asks |
|---|---|---|---|---|
| `bisection-xavier` | `po_jepa_no_consistency` | 1703 | DW, VdP | baseline without consistency, Xavier `P_out` |
| `bisection-zero` | `po_jepa_no_consistency` | 1703, 1704, 1705 | DW, VdP | does the zero init alone shrink? (3 seeds: the reference of the main comparison) |
| `dyadic-zero` | `po_jepa_no_consistency` | 1703 | DW | noise reference (same setting as the row above) |
| `dyadic-zero` | `po_jepa_time_only` | 1703 | DW, VdP | temporal consistency alone |
| `bisection-zero` | `po_jepa_space_only` | 1703 | DW, VdP | spatial consistency alone |
| `dyadic-zero` | `po_jepa_full` | 1704, 1705 | DW, VdP | is the C-1 shrinkage (seed 1703) reproducible? |
| `dyadic-xavier` | `po_jepa_full` | 1704, 1705 | DW, VdP | the chosen fix (dyadic + Xavier), 3 seeds with C-1 |
| `dyadic-zero~lt0.5`, `~lt0.25` | `po_jepa_full`, λ_t × 0.5 / × 0.25 | 1703 | DW, VdP | λ_t scan on the shrinking cell |
| `dyadic-xavier~lt0.5`, `~lt0.25` | `po_jepa_full`, λ_t × 0.5 / × 0.25 | 1703 | DW, VdP | λ_t scan on the fix |

Every training line is the normal `run.py run` (same evaluation as the main table). Each new job then gets `state_readout.py counterfactual` (action response A) and `shrinkage_check.py` (G9 motion gain), with the 200-trajectory probes and truth files that G12 wrote. advection_diffusion is left out: its probe cv R² is 0.735, below the 0.9 we need to read a ratio.

Where and when: in the clone where C-1 and C-3 ran (it has the 9 C-1 calibration files `configs/calibration/<scene>-200-u2000~*.yaml`, which are not in git). Finish the C-3 batch-1 remainder there first, then `git pull` (this changes `MANIFEST.json`; do not pull while any job of that clone is still running), then:

```bash
cd $WORK/full-model/po-jepa-full
V6=/absolute/path/of/the/v6/double_well-200/features    # the same value as in §7 G5 and §8f G12
R=../../results-small-validation; O=$R/c1-isolation; P=$R/c1-readout     # P: probes and truth files written by G12
SCENES="double_well van_der_pol"
feats() { if [ "$1" = double_well ]; then echo "$V6"; else echo "../../features-v7/$1-200"; fi; }
flags() { case $1 in
  bisection-xavier) echo "--temporal-partition bisection --readout-init xavier";;
  bisection-zero)   echo "--temporal-partition bisection --readout-init zero";;
  dyadic-xavier)    echo "--temporal-partition dyadic --readout-init xavier";;
  dyadic-zero)      echo "--temporal-partition dyadic --readout-init zero";; esac; }
# cells: <variant> <condition> <seeds> <scenes> [λ_t scale]  (the λ_t scan: Tomas 2026-11-25, Jerry chose "scale λ_t only")
CELLS="bisection-xavier po_jepa_no_consistency 1703 double_well,van_der_pol
bisection-zero po_jepa_no_consistency 1703,1704,1705 double_well,van_der_pol
dyadic-zero po_jepa_no_consistency 1703 double_well
dyadic-zero po_jepa_time_only 1703 double_well,van_der_pol
bisection-zero po_jepa_space_only 1703 double_well,van_der_pol
dyadic-zero po_jepa_full 1704,1705 double_well,van_der_pol
dyadic-xavier po_jepa_full 1704,1705 double_well,van_der_pol
dyadic-zero po_jepa_full 1703 double_well,van_der_pol 0.5
dyadic-zero po_jepa_full 1703 double_well,van_der_pol 0.25
dyadic-xavier po_jepa_full 1703 double_well,van_der_pol 0.5
dyadic-xavier po_jepa_full 1703 double_well,van_der_pol 0.25"
each() {  # calls $1 <variant> <condition> <seed> <scene> <λ_t scale> for every training job
  echo "$CELLS" | while read v c seeds scenes lt; do
    for seed in ${seeds//,/ }; do for s in ${scenes//,/ }; do $1 $v $c $seed $s ${lt:-1}; done; done; done; }
dir() { if [ "$5" = 1 ]; then echo "$1"; else echo "$1~lt$5"; fi; }   # result folder: same suffix as summarize_v7 (~lt0.5)
scale() { if [ "$5" != 1 ]; then echo " --time-weight-scale $5"; fi; }
job() { echo "$O/$(dir $@)/$4/seed$3/$2"; }
train() { echo "python run.py run --system $4 --trajectories 200 --updates 2000 --seed $3 --condition $2 $(flags $1)$(scale $@) --features-root $(feats $4) --device cuda:0 --output $(job $@)"; }
pt() { echo "--probe $P/probe/$1 --truth $P/truth/$1.json --features $(feats $1)"; }
cf() { echo "python tools/state_readout.py counterfactual --job $(job $@) $(pt $4) --output $O/counterfactual/$(dir $@)/$4/seed$3/$2.json --device cuda:0"; }
g9() { echo "python tools/shrinkage_check.py --job $(job $@) $(pt $4) --output $O/shrinkage/$(dir $@)/$4/seed$3/$2.json --device cuda:0"; }
{
echo "# ===== X G9 motion gain on the existing C-1 po_jepa_full jobs (GPU, 8 commands, evaluation only, --cross-version) ====="
for v in bisection-xavier bisection-zero dyadic-xavier dyadic-zero; do for s in $SCENES; do
  echo "python tools/shrinkage_check.py --job $R/c1/$v/$s/seed1703/po_jepa_full $(pt $s) --cross-version --output $O/shrinkage/c1-$v/$s/seed1703/po_jepa_full.json --device cuda:0"
done; done
echo "# ===== D training + evaluation, 200 trajectories x 2000 updates (GPU, 29 commands, independent) ====="
each train
echo "# ===== R counterfactual action response of the new jobs (GPU, 29 commands) ====="
each cf
echo "# ===== G9 motion gain of the new jobs (GPU, 29 commands) ====="
each g9
} > jobs-sv-c1-isolation.txt
grep -c '^python' jobs-sv-c1-isolation.txt      # must print 95
for s in X D R G9; do python tools/run_jobs.py jobs-sv-c1-isolation.txt --gpus 0,1 --only $s; done
```

- Sections: X (8, evaluation of existing C-1 jobs, no dependency) → D (29 training runs) → R (29) → G9 (29). One `run_jobs.py --only` per section as above, so a failure in one section does not block the next; rerunning skips finished lines; every tool refuses to overwrite an existing output.
- Time *(est.)*: each D line is one C-1-size run (200 × 2,000 plus the standard evaluation, roughly 20–30 minutes); R and G9 lines ~1–2 minutes each. About 10–15 GPU-hours, 6–8 hours on two cards.
- If an X line stops with a missing checkpoint, that C-1 folder was cleaned; leave it failed and tell us which.

Send back (a few MB plus logs):

```bash
cd $WORK
python $TOOL pack results-small-validation/c1-isolation logs-addon/jobs-sv-c1-isolation \
    full-model/po-jepa-full/jobs-sv-c1-isolation.txt nvidia-smi.txt --output c1-isolation.pack.sqlite
```

## 8i. G15: C-3 extra training with the dyadic-xavier fix (`c3-xavier` job file)

This one **trains** (6 new runs). Background: in C-2 and C-3 the full model "moves too little" on `double_well` (G9: its motion gain is about half that of `po_jepa_no_consistency`). The C-1 2×2 readout (G12) showed this happens only when the dyadic temporal partition and the zero-initialised `P_out` are combined; dyadic + Xavier does not shrink. Jerry decided on 2026-11-25 to keep the dyadic partition (the advisor's choice) and go back to Xavier initialisation of `P_out` (as in v7.1). This section checks the fix at C-2 scale on two structures: `Kfull` (every grid uses all its tokens as modes; best in C-3 batch 1) and `K16c` (16 modes plus the kept complement; the same K on every grid, as the advisor asked). In batch 1 (dyadic-zero) `K16c` `po_jepa_full` diverged on `double_well` and `van_der_pol`; the `K16c` lines here show whether it still diverges with Xavier `P_out`.

| Setting | Value |
|---|---|
| Variants | `Kfull`: `--temporal-partition dyadic --readout-init xavier --modes full --complement none`; `K16c`: `... --modes 16 --complement shared` (calibration files `configs/calibration/<scene>-2000-u10000~dyadic~Kfull.yaml` / `~dyadic~K16c.yaml`, separate from batch 1's `~dyadic~zeroinit~*.yaml`) |
| Scenes | `advection_diffusion`, `double_well`, `van_der_pol`, 2,000 trajectories × 10,000 updates, seed 1703 |
| Condition | `po_jepa_full` only. The references (`po_jepa_no_consistency` and the dyadic-zero full of each variant, `transformer_consistency`) are reused from C-3 batch 1 and C-2, not retrained. **Please keep those job folders.** |
| Results | `results-small-validation/c3/dyadic-xavier-Kfull/`, `results-small-validation/c3/dyadic-xavier-K16c/` |

Order: first finish the C-3 batch-1 remainder (§8e) in the current clone. Then pull and generate the job file:

```bash
cd $WORK/full-model/po-jepa-full
git pull                                 # only after every job file of this clone is finished (§1)
python tools/make_small_validation_jobs.py --stage c3-xavier > jobs-sv-c3-xavier.txt
python tools/run_jobs.py jobs-sv-c3-xavier.txt --gpus 0,1
```

36 commands: 6 × `calibrate`, 2 × `check` (`van_der_pol`), 6 training runs, 12 readouts and counterfactuals (R3), 2 modal spectra (R4, `advection_diffusion`), 6 × `shrinkage_check.py` (G9), 2 summaries. About 20–24 GPU-hours *(est.)*, ~11 hours on two cards (C-3 batch 1 took about 3–3.5 GPU-hours per training run including its calibration; a run that diverges stops early). G14 (§8h, if present) can share this clone and run after or alongside it.

If a training run diverges (loss to 1e3 or more, gradient norm to 1e16, as `K16c` did in batch 1), keep its folder and log and go on; the other lines are independent except the R3, R4 and G9 lines of that run, which will fail.

Send back:

```bash
cd $WORK
python $TOOL pack results-small-validation/c3/dyadic-xavier-Kfull results-small-validation/c3/dyadic-xavier-K16c \
    logs-addon/jobs-sv-c3-xavier full-model/po-jepa-full/configs/calibration \
    full-model/po-jepa-full/jobs-sv-c3-xavier.txt nvidia-smi.txt --output c3-xavier.pack.sqlite
```

### `K16c` divergence: send back the batch-1 checkpoints (no GPU, can be done any time)

The step-by-step training logs of the two diverged runs are already back (in `c3-batch1-partial-2026-11-24.pack.sqlite`): on `double_well` the gradient first jumps at about step 3040–3140 and the loss explodes from about 3180; on `van_der_pol` from about step 5880. To see which part of the model runs away (the shared complement rotation `A(λ_⊥, g)`, the attention, the residual heads), Jerry needs the PO-JEPA checkpoints of those runs and of two references at the same steps. They are small (a few MB each) and `pack_results.py` skips `*.pt` by default, so this is a separate pack with `--no-default-excludes`:

```bash
cd $WORK
R=results-small-validation/c3
python $TOOL pack \
    $R/dyadic-zero-K16c/double_well/seed1703/po_jepa_full/run/job/training/po_jepa_full/checkpoints \
    $R/dyadic-zero-K16c/van_der_pol/seed1703/po_jepa_full/run/job/training/po_jepa_full/checkpoints \
    $R/dyadic-zero-K16c/double_well/seed1703/po_jepa_no_consistency/run/job/training/po_jepa_no_consistency/checkpoints \
    $R/dyadic-zero-K16c/van_der_pol/seed1703/po_jepa_no_consistency/run/job/training/po_jepa_no_consistency/checkpoints \
    $R/dyadic-zero-Kfull/double_well/seed1703/po_jepa_full/run/job/training/po_jepa_full/checkpoints \
    $R/dyadic-zero-Kfull/van_der_pol/seed1703/po_jepa_full/run/job/training/po_jepa_full/checkpoints \
    --no-default-excludes --exclude '*.partial' --output c3-k16c-checkpoints.pack.sqlite
```

About 100 files, roughly 0.3 GB *(est.)*. If a path does not exist, drop it and say which one. Do not delete or move these checkpoints afterwards.

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
