# previous_codes - archive of the first round of experiments

Files were only **moved and given a header comment** (purpose, original name, caveats). No logic was changed, so
the hard-coded Windows paths (`D:\Datasets\Medical\brisc2025\...`) and relative weight folders
(`EfficientNets\brisc_effnet_b0_v1.pth`, ...) still refer to the original machine and are not expected to run here
as-is. The new, parametrised code is in `../shortcut_audit/`. Model weights (`*.pth`) and the YOLO sample-batch
images were not included.

| folder | what it holds |
|---|---|
| `01_data_preparation/` | morphometric feature extraction (+ its CSV), zero / Gaussian / inpainted / foreground-only dataset creation |
| `02_morphometrics_viewers/` | interactive matplotlib viewers for brain hull, tumor hull and measurements |
| `03_baseline_classifiers/` | EfficientNet-B0/B1/B2, VGG16/19, Xception, InceptionV3 train + test scripts |
| `04_shortcut_audit/` | multi-model audits on masked/inpainted/foreground data, SRS + CDM, Grad-CAM, saved report + CDM plots |
| `05_spatial_prior_analysis/` | random forest on location vs shape features |
| `06_mitigation_experiments/` | causal-shortcut loss (B2), SupCon (B0), dual-branch Siamese ResNet50 |
| `07_yolo_experiments/` | YOLOv8 classification / segmentation baselines, adaptive-margin, cross-view adversarial variants (+ metrics) |

## Known issues found while organising (kept, not fixed)
* `04_shortcut_audit/audit_inpainted_foreground.py` evaluates the **train** split, so its "clean" accuracy (~99.9 %) is not held-out accuracy.
* Zero-out / Gaussian audits use `val`, inpainted/foreground use `train`: the numbers are not directly comparable.
* Xception is built with `timm 'xception'` in most scripts but `'legacy_xception'` in `xai_gradcam_gallery.py`.
* "SRS" is probability-based in `cdm_and_srs.py` but accuracy-based in `compare_b2_baseline_vs_causal.py`.
* The causal-shortcut loss is trained and tested on the same zero-out intervention (circular).
* `train_random_forest.py` and the SupCon run have no saved results.
