# shortcut_audit - does a brain-tumor CNN diagnose from the lesion or from where it sits?

## The question and the logic
Tumor **location is partly legitimate evidence** (pituitary tumors sit in the sella, meningiomas on the dura), so
"the model uses location" is not automatically a shortcut. The study therefore measures, in order:

| step | script | question it answers |
|---|---|---|
| 0 | `s00_build_index.py` | Is the data clean? Cross-split near-duplicates (BRISC has no patient id), class x plane / image-size confounds, mask coverage |
| 1 | `s01_extract_features.py` | Brain/tumor geometry: radial depth, distance to dura, edge contact, area, solidity ... |
| 2 | `s02_cue_probes.py` | How much does each **cue alone** (plane, location, shape, silhouette) say about the class, without any CNN? |
| 3 | `s03_train.py` | Models: several architectures x seeds; probe models that see only the inpainted image / only the tumor crop / only the silhouette; the mitigation (`--p_relocate`) |
| 4 | `s04_location_typicality.py` | **Natural experiment**: accuracy for tumors in typical vs atypical locations (no image editing) |
| 5 | `s05_interventions.py` | Erasure sweep (zero / Gaussian / inpaint, dilated 0-20 px), **placebo** silhouette, **tumor transplant** (same location / cue conflict), foreground-only |
| 6 | `s06_xai.py` | Grad-CAM + SmoothGrad quantified against the mask (mass in tumor, pointing game, deletion/insertion vs random/oracle, randomisation sanity check) |
| 7 | `s07_make_report.py` | One `REPORT.md` with every table, mean over seeds, CIs where applicable |

Why this differs from the first round (`../previous_codes`): held-out TEST split only; seeds + bootstrap CIs; the
erased-tumor test is separated from "black blob" effects (placebo) and from residual pathology (dilation sweep);
location is tested without editing pixels; the mitigation is judged on interventions it was **not** trained on;
XAI is scored, not just displayed.

## Getting the code on your computer (from a new folder)
```bash
git clone -b claude/modest-mayer-mqsj89 https://github.com/Ampmalekpour/mri_brisc.git
cd mri_brisc
```
If the folder already exists: `git fetch origin claude/modest-mayer-mqsj89 && git checkout claude/modest-mayer-mqsj89`.
In PyCharm: *File > Open* the `mri_brisc` folder, then *Settings > Project > Python Interpreter* and create a venv.

## Setup
```bash
pip install -r shortcut_audit/requirements.txt      # install the CUDA build of torch from pytorch.org if you have a GPU
```
1. Open `shortcut_audit/config.py`, set `DATA_ROOT` (default `D:\Datasets\Medical\brisc2025`) - or set env var `BRISC_ROOT`.
   Expected layout: `classification_task/<train|val|test>/<class>/*.jpg` and `segmentation_task/<split>/{images,masks}/...`
   (a mask is any image whose path contains "mask" and whose file stem equals the classification image stem).
2. In `config.py` check `TRAIN_SPLIT / VAL_SPLIT / TEST_SPLIT`. **Every reported number uses the test split.**
   If your `val` folder was carved out of train, make sure it is not also in `test`; s00 reports leakage.
3. In PyCharm set the *working directory* of each run configuration to `shortcut_audit/`.

## Running
Smoke test without real data (synthetic fake BRISC, minutes on CPU):
```bash
python tools/make_synthetic_brisc.py C:\tmp\fake_brisc
set BRISC_ROOT=C:\tmp\fake_brisc & set AUDIT_PRETRAINED=0 & set AUDIT_EPOCHS=12 & set AUDIT_SIZE=112 & set AUDIT_LR=2e-3
python s00_build_index.py && python s01_extract_features.py && python s02_cue_probes.py
python s03_train.py --arch resnet18 --seed 0 && python s05_interventions.py --run resnet18_clean_s0
```
Real run: `python s00_build_index.py`, `s01`, `s02`, then `python run_all.py --quick` (1 arch, 1 seed) and, once happy, `python run_all.py`.
Everything is resumable (trained runs are skipped). Results appear in `shortcut_audit/outputs/results/` (git-ignored).

## Status / honesty notes
* The code was **smoke-tested end-to-end only on synthetic data** (resnet18, CPU). It has **not** been run on real BRISC, so expect to
  adjust paths/mask discovery on first contact; s00 prints warnings if masks are not found.
* Inpainting uses OpenCV Telea (fast, on the fly), not the biharmonic method of the first round; results are not identical.
* ViT Grad-CAM uses the last block's `norm1` tokens - a known approximation. Pretrained timm weights need internet the first time.
* Interpretation rule of thumb: an XAI map on the tumor does **not** show the model uses tumor *appearance* rather than *location*;
  that comes from s04 (typicality gap) and s05 (placebo, transplant cue-conflict). Single-model/single-seed differences are noise - read the CIs.
