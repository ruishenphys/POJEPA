# Calibration files in git (2026-11-17)

The ten `*.yaml` files here are the v7 calibrations (sigma_z, lambda_t, lambda_s, rho) that the v7.1
formal run actually used (v7 branch, code 748ca22 plus the two local fixes now merged into full-model).

- `*-200-u2000.yaml`: pilot (200 trajectories, 2000 updates), written 2026-11-09 by `run.py calibrate`.
- `*-2000-u10000.yaml`: formal (2000 trajectories, 10000 updates), written 2026-11-10 by `run.py calibrate` (C stage).
- Source: the v7.1 return package (`POJEPA/configs/calibration/`); byte-identical to the copies in
  `results-v7-pilot-lite.zip`. The operator restored them after a fresh clone and checked them against
  `results-v7/full/calibrate-*/run/`.

`checks.load_protocol()` reads `configs/calibration/<tag>-u<updates>.yaml`, so a fresh clone of this branch
now reuses these values for these five scenes. They were calibrated for the current loss definitions.
If a later change alters a loss term (for example the temporal consistency normalisation), the matching
lambda must be recalibrated with `run.py calibrate`; do not keep using these values silently.

This directory is excluded from MANIFEST.json (`tools/refresh_manifests.py`, EXCLUDE_DIRS), because
`run.py calibrate` rewrites these files at run time.
