# Handcrafted BRISC 2025 feature study

The workflow decomposes each MRI into F01–F25 and trains classical classifiers on each feature group and on named combinations. It also trains a matched full-image CNN and a mask-guided CNN for comparison. Each Python stage runs directly from PyCharm and gets its settings from `config.py`.

## Get the branch

For a new checkout, run the following in a terminal.

```bash
git clone --branch codex/brisc-handcrafted-features https://github.com/Ampmalekpour/mri_brisc.git
cd mri_brisc
```

From an existing checkout, run these commands.

```bash
git fetch origin
git switch --track origin/codex/brisc-handcrafted-features
```

If you already have the branch locally, use `git switch codex/brisc-handcrafted-features` and `git pull --ff-only`.

## Configure PyCharm

1. Open the `mri_brisc` folder in PyCharm and create a project virtual environment with Python 3.11 or 3.12.
2. In PyCharm's Terminal, install the feature extraction and classical ML dependencies using the project interpreter.

   ```bash
   python -m pip install -r handcrafted/requirements.txt
   ```

   For step 6, install PyTorch. The CPU installation is shown below. A CUDA-enabled PyTorch installation can also be used.

   ```bash
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. In `handcrafted/config.py`, set `DATA_ROOT` to the extracted folder that contains `classification_task` and `segmentation_task`. The repository's existing `BRISC_ROOT` environment variable also works. Set `OUTPUT_DIR` if you want a separate results directory. [Official BRISC data on Zenodo](https://doi.org/10.5281/zenodo.17524350)
4. Right-click each step script in the Project pane and choose **Run**. Its path is resolved independently of the run configuration's working directory.

```text
brisc2025/
  classification_task/train/<class>/<scan>.jpg
  classification_task/test/<class>/<scan>.jpg
  segmentation_task/train/masks/<scan>.png
  segmentation_task/test/masks/<scan>.png
```

## Run the stages

| Run this script | What it prints | Where to inspect results under `handcrafted/outputs/` |
| --- | --- | --- |
| `s01_index.py` | Class counts, mask coverage, duplicate groups and test sizes | `index/manifest.csv`, `coverage.csv`, `groups.csv`, plane and duplicate tables |
| `s02_geometry.py` | F01–F11 coverage plus examples such as centroid, area and circularity | `features/F01.csv` through `F11.csv`, `features/summaries/`, `qc/step2/` |
| `s03_appearance.py` | F12–F20 including intensity, texture, radial architecture and boundary contrast | `features/F12.csv` through `F20.csv`, alternative texture binning tables and `qc/step3/` |
| `s04_context.py` | F21–F25 including ring measurements, HOG, local patches and distant tissue | `features/F21.csv` through `F25.csv`, `F23_patches.npz`, `qc/step4/` |
| `s05_train_classical.py` | Grouped CV balanced accuracy, leakage-reduced accuracy, balanced accuracy and macro F1 per model | `classical/metrics.csv`, `per_class.csv`, `confusions.csv`, `predictions/`, `cv/`, `models/` |
| `s06_train_cnn.py` | Per-epoch training loss and grouped validation scores, then test metrics | `cnn/history/`, `metrics.csv`, `predictions/` and `models/` |
| `s07_report.py` | Leading configurations based on training CV | `report/REPORT.md`, a score plot, feature-group gains and paired CNN comparisons |
| `s08_robustness.py` | Features most sensitive to mask erosion, dilation, smoothing, JPEG, intensity and resampling | `robustness/feature_sensitivity.csv` and per-image changes |

Start with steps 1 and 2, then inspect `index/coverage.csv` and the overlay PNGs. In the overlays, cyan traces the estimated head outline. Step 2 and 3 overlays color the tumor from rim to core. Step 4 overlays show the lesion and outer rings.

For a smaller first training run, edit these values in `config.py` and run steps 1, 2, 5 and 7. Step 5 reads the extraction tables needed by the selected experiments.

```python
EXPERIMENT_IDS = ("E00", "E01", "E02", "E03", "E04", "E05", "E07", "4C02")
MODELS = ("logistic", "svm")
```

With `EXPERIMENT_IDS = ()`, step 5 runs all 25 single-group probes and the named combinations in `experiments.py`. The default six model families are multinomial logistic regression, shrinkage LDA, RBF SVM, random forest, XGBoost and KNN. Small geometry probes also get a depth-three tree. By default, LDA uses fold-fitted PCA capped at 128 components for wide inputs. The full sweep and CNN training take longer than the first run.

Run steps 3 and 4 before selecting appearance, context or visual experiments. Run step 6 to include E21 and E22, then regenerate the report. Step 7 can also report the classical or CNN results by themselves. For a CPU trial, set `CNN_SEEDS = (42,)`, `CNN_EPOCHS = 2` and `CNN_SIZE = 64`. Use the default seeds and epoch budget for a fuller comparison.

## Feature groups and experiments

| Groups | Measurements |
| --- | --- |
| F01–F03 | Native image and JPEG metadata, padding and background statistics, and filename-derived imaging plane |
| F04–F06 | Image- and head-relative position, lesion extent, Feret diameters and inscribed radii |
| F07–F11 | Circularity and other shape ratios, Hu, Zernike and Fourier descriptors, contour roughness, topology, and spatial and centered occupancy grids |
| F12–F14 | Grayscale moments and percentiles, local and distant reference normalization, and ROI histograms |
| F15–F17 | Masked GLCM, GLRLM, GLSZM, GLDM, NGTDM and GLDZM summaries, LBP, Gabor, LoG and Haar wavelets |
| F18–F20 | Bright and dark intensity compartments, core-to-rim profiles and along-normal margin contrast |
| F21–F22 | Unrestricted and head-restricted rings, estimated peripheral contact, midline and mirrored tissue measurements |
| F23–F25 | HOG for four views, fold-fitted BoVW/VLAD, low-resolution MRI and DCT, and pooled distant tissue patches |

`experiments.py` lists every column selector. E00 is the small geometry-and-plane probe. E01–E20 progress from individual cues to combined models. E13, E13H and E13U use ring features without paired lesion intensity terms. V01–V06 isolate the four HOG views, BoVW and VLAD. P01 and P02 run fold-fitted PCA of the centered mask and lesion appearance. T16, T32 and T64 compare texture quantization. 4C01–4C04 are four-class image-only controls. Set `VOCABULARY_SIZES = (32, 64, 128, 256)` to compare codebook sizes in grouped CV.

Set `RUN_EXHAUSTIVE_SUBSETS = True` to add all 255 nonempty combinations of eight feature families and an intercept-only baseline with `SUBSET_MODEL`. This writes `all_subsets.csv`, `family_increments.csv` and `family_shapley.csv`. Exact subset Shapley contributions describe changes in grouped training CV balanced accuracy.

## Read the results

- **Check the cohort.** The primary three-class study includes glioma, meningioma and pituitary scans with usable masks, including for image-only probes and both CNNs. The four-class series also includes no-tumor scans and uses image-derived inputs. Enable `CNN_RUN_FOUR_CLASS` to add its full-image CNN.
- **Use the leakage-reduced result for the primary comparison.** The official result includes every test case in the cohort. The leakage-reduced result excludes test cases that share a hash-duplicate or optional patient group with the official training split. Validation for model selection comes from grouped folds of training. BRISC supplies slice-level identifiers. If you have verified patient IDs, set `PATIENT_CSV` to a file with `sample_id,patient_id`; the sample IDs match the paths in the index.
- **Read the definitions and coverage together.** Blank feature cells mark regions where a measurement is undefined. Imputation, scaling, PCA and visual vocabularies fit inside the training folds. Texture pairs and texture filter neighborhoods use valid ROI pixels. Distances and sizes are measured from 2-D slices in pixels or fractions. Head-relative measurements use an estimated foreground outline and bounding box. Anatomical landmark distances require separate validated landmarks. Bright and dark compartments are grayscale measurements. The texture matrices use custom 2-D definitions and fixed grayscale bins.
- **Compare the same cases.** The paired report joins predictions by sample ID and reports agreement, error overlap and metric differences with group-level bootstrap intervals. `score_*` columns are probabilities for probabilistic models and decision scores for SVM. Macro AUROC is the mean one-vs-rest AUROC when the evaluated view contains all classes. Balanced accuracy and macro F1 average over classes represented in that view.
- **Use matched CNN variants.** Both variants use the same architecture, cases, training schedule and split for a given seed. They train from scratch. E21 receives the full MRI and a zero second channel. E22 receives the full MRI and a tumor mask as its second channel. The original repo's `shortcut_audit/` provides complementary CNN interventions.

Dataset paths, feature tables, visual overlays and model artifacts are generated locally. JSON settings files, CV memberships and per-image scores record the run. Use a distinct `OUTPUT_DIR` when you want to keep two experiment runs side by side.
