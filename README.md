# RetinaOWL

## What RetinaOWL is

RetinaOWL is an open-world, multi-label model for colour fundus photographs: it predicts the 45 retinal diseases of RFMiD that it was trained on and, at the same time, flags diseases outside that vocabulary as unknown, without discarding its known-disease predictions for the same image. A frozen DINOv2 ViT-S/14 backbone adapted with rank-8 LoRA encodes the image into patch tokens. A prototype disease memory predicts the known diseases (one learned prototype per disease), and an explained/residual decomposer pools the token evidence that no active prototype explains into a learned unknown score and a residual map. The final unknown score is the rank mean of this residual score, the negated maximum logit and a k-nearest-neighbour feature distance. Training hides a fresh, frequency-tier-stratified set of seven diseases in every batch (episodic withholding), so the model learns to score images whose diseases it cannot explain while keeping supervision for the diseases it can, including images that carry both. The repository also contains the comparison of masked-class evaluation (classes trained, masked at test time) with leave-classes-out retraining (classes never trained) on identical test images, for RetinaOWL, six baselines and eleven ablations.

## What is in the repository

All code is in Python scripts, one per experiment. Each script writes its outputs (checkpoints, logs, predictions, results and a reproducibility record) to an output folder next to it and resumes from where it stopped if run again.

- Model/Model_Training.py: RetinaOWL. Builds the preprocessing caches, trains five seeds, runs the masked-class (withheld-class fold) evaluation, the leave-classes-out benchmark and the external validation on JSIEC and ODIR-5K.
- Ablations/: one script per ablation, each in its own folder: No_Residual_Branch (A1), Fixed_Withholding (A2), Uniform_Withholding (A3), No_Explanation_Consistency (A4), No_LoRA (A5), Linear_Probe (A6), No_Compactness_Separation (A7), No_FOV_Masking (A8), No_Augmentation (A9), Anatomy_Grounding (A10), Image_Rotation (A11).
- Baselines/: one script per baseline, each in its own folder: UIOS (B1), RetFound (B2, RETFound-MAE), RetFound_DINO (B3, RETFound-DINOv2), ML_Decoder (B4), ResNet50_ASL (B5), SwinT_ASL (B6).
- Robustness_Analysis/: Corruption_Analysis.py (RetinaOWL and A9 under six image corruptions) and Rotation_Analysis.py (RetinaOWL, A10 and A11 under rotations). Both evaluate finished models; nothing is retrained.
- Additions/Protocol_Controls.py: controls for the masked-class versus leave-classes-out comparison (kNN gallery controls, mixed versus unknown-only cases with operating points, residual-map deletion test, size-matched training control).
- Additions/External_Bootstrap_CI.py: patient-clustered bootstrap confidence intervals for the external (JSIEC, ODIR-5K) unknown-detection AUROC of every model, from saved predictions.
- Preprocessing_Cache/JSIEC_class_to_RFMiD_mapping.csv and Preprocessing_Cache/ODIR_keyword_to_RFMiD_mapping.csv: the mappings of JSIEC classes and ODIR-5K diagnostic keywords to RFMiD classes (known, unknown, normal or omitted). They define every external label and are not available elsewhere.
- Model/Output/reproducibility/ and Additions/Output_Protocol_Controls/reproducibility/: reference copies of the five held-out label sets, the validation calibration split, the class tiers, the images removed for leave-classes-out and for the size-matched control, and the selected checkpoints. The scripts regenerate these from fixed seeds; the copies let you check that your run used the same folds and splits.
- requirements.txt: the Python packages and versions used.

## Expected file structure

Paths in every script are relative to the script's own folder and are set in the configuration section at the top of each script. The layout below is the one the scripts expect. Folders marked "downloaded" are not in the repository (see the next section); folders marked "created" are written by the scripts.

```
Root/
    README.md
    RUN_ORDER.md
    requirements.txt
    Model/
        Model_Training.py
        Output/                               created
    Ablations/
        No_Residual_Branch/                   one script per folder; output written here
        Fixed_Withholding/
        Uniform_Withholding/
        No_Explanation_Consistency/
        No_LoRA/
        Linear_Probe/
        No_Compactness_Separation/
        No_FOV_Masking/
        No_Augmentation/
        Anatomy_Grounding/
        Image_Rotation/
    Baselines/
        UIOS/                                 one script per folder; output written here
        RetFound/
        RetFound_DINO/
        ML_Decoder/
        ResNet50_ASL/
        SwinT_ASL/
    Robustness_Analysis/
        Corruption_Analysis.py
        Rotation_Analysis.py
        Output/                               created
    Additions/
        Protocol_Controls.py
        External_Bootstrap_CI.py
        Output_Protocol_Controls/             created
        Output_External_CI/                   created
    Baseline_Models/                          downloaded
        B1.pth
        B2.pth
        B3.pth
        B5.pth
        B6.pth
    Dataset/                                  downloaded
        DINOv2/
            config.json
            pytorch_model.bin
        RFMID/
            Training_Set/Training_Set/Training/
            Training_Set/Training_Set/RFMiD_Training_Labels.csv
            Evaluation_Set/Evaluation_Set/Validation/
            Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv
            Test_Set/Test_Set/Test/
            Test_Set/Test_Set/RFMiD_Testing_Labels.csv
        JSIEC/
            0.0.Normal/
            0.1.Tessellated fundus/
            ...                               the 39 class folders as distributed
        ODIR-5K/
            Training Set/Annotation/training annotation (English).xlsx
            Training Set/Images/
            Off-site Test Set/Annotation/off-site test annotation (English).xlsx
            Off-site Test Set/Images/
            On-site Test Set/Annotation/on-site test annotation (English).xlsx
            On-site Test Set/Images/
    Preprocessing_Cache/
        JSIEC_class_to_RFMiD_mapping.csv      in the repository
        ODIR_keyword_to_RFMiD_mapping.csv     in the repository
        RFMiD/                                created on the first run
        JSIEC/                                created on the first run
        ODIR/                                 created on the first run
```

## Downloads

Install the packages with `pip install -r requirements.txt` (Python 3.10, PyTorch with CUDA; a single NVIDIA GPU with at least 24 GB was used).

Datasets and backbone (place as shown in the file structure):

| Resource | Place in | Source |
|---|---|---|
| RFMiD | Dataset/RFMID/ | https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid |
| JSIEC (1000 fundus images, 39 categories) | Dataset/JSIEC/ | https://www.kaggle.com/datasets/linchundan/fundusimage1000 |
| ODIR-5K (OIA-ODIR) | Dataset/ODIR-5K/ | https://github.com/nkicsl/OIA-ODIR |
| DINOv2 ViT-S/14 (config.json and pytorch_model.bin) | Dataset/DINOv2/ | https://huggingface.co/facebook/dinov2-small |

Baseline pretrained weights (download each file and save it under the name in the first column):

| Saved as | Used by | Original file | Source |
|---|---|---|---|
| Baseline_Models/B1.pth | UIOS (B1) | resnet50-0676ba61.pth (torchvision ResNet-50, ImageNet V1) | https://download.pytorch.org/models/resnet50-0676ba61.pth |
| Baseline_Models/B2.pth | RETFound-MAE (B2) | RETFound_mae_natureCFP.pth | https://huggingface.co/YukunZhou/RETFound_mae_natureCFP |
| Baseline_Models/B3.pth | RETFound-DINOv2 (B3) | RETFound_dinov2_meh.pth | https://huggingface.co/YukunZhou/RETFound_dinov2_meh |
| Baseline_Models/B5.pth | ResNet-50 + ASL (B5) | resnet50-11ad3fa6.pth (torchvision ResNet-50, ImageNet V2) | https://download.pytorch.org/models/resnet50-11ad3fa6.pth |
| Baseline_Models/B6.pth | Swin-T + ASL (B6) | swin_t-704ceda3.pth (torchvision Swin-T, ImageNet V1) | https://download.pytorch.org/models/swin_t-704ceda3.pth |

ML-Decoder (B4) has no separate weights file; it uses the DINOv2 ViT-S/14 in Dataset/DINOv2/. The two RETFound repositories on Hugging Face require accepting their terms and logging in before the files can be downloaded.

Downloaded automatically on first use: the fovea and optic-disc localiser of fundus_image_toolbox (used by Anatomy_Grounding, Image_Rotation and Rotation_Analysis).

## Running the code

See RUN_ORDER.md for the order in which to run the scripts and how long each one takes.
