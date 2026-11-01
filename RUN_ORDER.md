# Run order

## Before running

Set up the files as described in README.md and install the packages in requirements.txt.

Run every script from its own folder, because all paths in the configuration section at the top of each script are relative to that folder. For example:

```
cd Model
MPLBACKEND=Agg python Model_Training.py 2>&1 | tee run.log
```

MPLBACKEND=Agg saves figures to the output folder instead of opening plot windows. Every script resumes where it stopped: if a run is interrupted, start it again and finished seeds, folds and evaluations are loaded instead of recomputed.

Each training script runs five training seeds (42 to 46) with the five held-out label sets masked at test time, five leave-classes-out trainings (one per fold, seed 42), and the external validation on JSIEC and ODIR-5K. No settings need to be changed to reproduce the paper; only the paths in the configuration section may need adjusting to your layout.

Times below are measured training times on one NVIDIA A30 (24 GB). Evaluation adds well under an hour per script, and the first run of the model also builds the preprocessing caches.

## 1. Model

Run first. It builds the preprocessing caches in Preprocessing_Cache/ that every other script reuses, and its output is read by the ablations and the analyses.

| Script | Time |
|---|---|
| Model/Model_Training.py | about 14 h |

## 2. Ablations

Run after the model. A1, A2, A3, A4 and A8 do not change the first training phase, so they copy it from Model/Output instead of retraining it (set by PHASE_A_SOURCE in their configuration). The ablations are independent of each other and can run in any order, or in parallel on separate GPUs.

| Script folder | Time |
|---|---|
| Ablations/No_Residual_Branch (A1) | about 7 h |
| Ablations/Fixed_Withholding (A2) | about 9 h |
| Ablations/Uniform_Withholding (A3) | about 7 h |
| Ablations/No_Explanation_Consistency (A4) | about 9 h |
| Ablations/No_LoRA (A5) | about 17 h |
| Ablations/Linear_Probe (A6) | about 16 h |
| Ablations/No_Compactness_Separation (A7) | about 13 h |
| Ablations/No_FOV_Masking (A8) | about 7 h |
| Ablations/No_Augmentation (A9) | about 10 h |
| Ablations/Anatomy_Grounding (A10) | about 15 h |
| Ablations/Image_Rotation (A11) | about 15 h |

A10 and A11 also detect the fovea and optic disc on every image on their first run (the localiser is downloaded once).

## 3. Baselines

Run after the model (they reuse its preprocessing caches). The baselines are independent of each other and of the ablations.

| Script folder | Time |
|---|---|
| Baselines/UIOS (B1) | about 18 h |
| Baselines/RetFound (B2) | about 8 h |
| Baselines/RetFound_DINO (B3) | about 9 h |
| Baselines/ML_Decoder (B4) | about 9 h |
| Baselines/ResNet50_ASL (B5) | about 9 h |
| Baselines/SwinT_ASL (B6) | about 8 h |

## 4. Robustness analysis

Run after the model and ablations A9, A10 and A11 have finished. Both scripts only evaluate the finished models (all five seeds) on RFMiD, JSIEC and ODIR-5K; nothing is trained. Both write to Robustness_Analysis/Output/.

| Script | Models | Time |
|---|---|---|
| Robustness_Analysis/Corruption_Analysis.py | RetinaOWL, A9 | several hours (not timed) |
| Robustness_Analysis/Rotation_Analysis.py | RetinaOWL, A10, A11 | several hours (not timed) |

## 5. Additions

Run after the model and all six baselines have finished; both scripts read their output folders (set in CTRL_MODELS and MODELS in the configuration sections).

| Script | What it does | Time |
|---|---|---|
| Additions/Protocol_Controls.py | Gallery and mixed-case analyses and the deletion test from saved outputs, then five size-matched RetinaOWL trainings (one per fold) | about 6.5 h |
| Additions/External_Bootstrap_CI.py | Bootstrap confidence intervals from saved external predictions; runs on CPU | a few minutes |

## Total

About 14 h for the model, 125 h for all ablations, 61 h for all baselines, several hours for the robustness analysis and about 6.5 h for the additions on one A30. Steps 2 and 3 can be split across several GPUs.
