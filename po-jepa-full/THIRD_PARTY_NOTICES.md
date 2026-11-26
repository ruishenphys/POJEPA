# Third-party code

| File | Upstream | License |
|---|---|---|
| `src/po_jepa/third_party/vjepa2_ac.py` | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) @ 204698b — `src/models/ac_predictor.py`, `src/models/utils/modules.py`, `src/utils/tensors.py` | MIT |
| `src/po_jepa/third_party/jepa_wms.py` | [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms) @ 13cf1d9 — `app/plan_common/models/vit.py` (DINO-WM predictor, itself from [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) and lucidrains/vit-pytorch), `app/plan_common/models/AdaLN_vit.py` | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

The jepa-wms code is used here for non-commercial academic research only, with attribution:
Terver et al., "What Drives Success in Physical Planning with Joint-Embedding Predictive World Models",
arXiv:2512.24497 (TMLR 2026); Zhou, Pan, LeCun, Pinto, "DINO-WM: World Models on Pre-trained Visual Features
enable Zero-shot Planning", arXiv:2411.04983. The changes made to the copied code are listed at the top of
`jepa_wms.py`; the adaptation to frozen V-JEPA 2 features (input/output projections, parameter matching,
context window, losses) is in `src/po_jepa/wm_baselines.py`, `src/po_jepa/ac_baseline.py` and
`src/po_jepa/runtime/ac_training.py`; the DROID adaptation (7-dimensional actions, parameter matching against the
DROID PO-JEPA, no proprioception, the DROID optimizer betas of the upstream DROID configs) is in
`src/po_jepa/droid/wm.py` and `src/po_jepa/droid/conditions.py`.
