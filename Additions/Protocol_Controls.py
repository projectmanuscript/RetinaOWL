#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys

# ---- paths -----------------------------------------------------------------------------------
OUTPUT_ROOT   = "./Output"                              # overridden by the control block at the end of §0
RFMID_ROOT    = "../Dataset/RFMID"                      # Training_Set/ Evaluation_Set/ Test_Set/
RFMID_CACHE   = "../Preprocessing_Cache/RFMiD"  # <split>/images/<ID>.png -- built if missing
DINOV2_ROOT   = "../Dataset/DINOv2"                     # config.json + pytorch_model.bin (ViT-S/14)
JSIEC_ROOT    = "../Dataset/JSIEC"                      # the 39 class folders
ODIR_ROOT     = "../Dataset/ODIR-5K"                    # Training Set/ Off-site Test Set/ On-site Test Set/
JSIEC_CACHE   = "../Preprocessing_Cache/JSIEC"  # built if missing
ODIR_CACHE    = "../Preprocessing_Cache/ODIR"   # built if missing
JSIEC_MAP_CSV = "../Preprocessing_Cache/JSIEC_class_to_RFMiD_mapping.csv"
ODIR_MAP_CSV  = "../Preprocessing_Cache/ODIR_keyword_to_RFMiD_mapping.csv"

# ---- run control -----------------------------------------------------------------------------
SEEDS          = [42, 43, 44, 45, 46]
SEEDS_THIS_RUN = [42, 43, 44, 45, 46]
GPU_DEVICE     = 0
RUN_EXTERNAL   = True
RUN_LCO        = True
LCO_FOLDS      = [1, 2, 3, 4, 5]
LCO_SEEDS      = [42]

# ---- data and preprocessing ------------------------------------------------------------------
IMG_SIZE, PATCH_SIZE = 518, 14
FOV_THRESH = 12                         # 0-255; darker pixels are camera padding, not retina
RFMID_LAYOUT = {
    "train": ("Training_Set/Training_Set/Training",      "Training_Set/Training_Set/RFMiD_Training_Labels.csv"),
    "val":   ("Evaluation_Set/Evaluation_Set/Validation", "Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv"),
    "test":  ("Test_Set/Test_Set/Test",                  "Test_Set/Test_Set/RFMiD_Testing_Labels.csv"),
}
TIER_EDGES = {"head": 100, "medium": 20, "tail": 5}     # training positives; below 5 = ultra_rare

# ---- model -----------------------------------------------------------------------------------
KP                  = 1        # prototypes per disease
ATTN_TEMPERATURE    = 0.07     # prototype similarity temperature (fixed)
DECOMP_TAU_E        = 0.3      # explanation temperature (fixed buffer); log_Te is learnable
TOP_K_RESIDUAL_FRAC = 0.10     # fraction of tokens in the residual top-k mean
LORA_RANK, LORA_ALPHA = 8, 16  # on query, key, value and output projections of every block
DROPOUT_P           = 0.0
USE_FOV_MASK, FOV_ERODE_RADIUS, FOV_TOKEN_KEEP = True, 14, 0.5

# ---- optimisation ----------------------------------------------------------------------------
NUM_EPOCHS_PHASE_A, NUM_EPOCHS_PHASE_B = 50, 30
MICRO_BATCH_SIZE, ACCUM_STEPS, EVAL_BATCH_SIZE, NUM_WORKERS = 8, 2, 16, 2
LR_HEADS_A, LR_LORA_A = 1e-3, 1e-4
LR_HEADS_B, LR_LORA_B = 3e-4, 3e-5
WEIGHT_DECAY     = 1e-4
CKPT_EVERY_STEPS = 200

# ---- augmentation (training split only) --------------------------------------------------------
HFLIP_P = 0.5
USE_PHOTO_AUG = True
PHOTO_AUG = {"brightness": 0.30, "contrast": 0.30, "saturation": 0.20, "hue": 0.02,
             "gamma_range": (0.80, 1.25), "gamma_p": 0.50, "blur_p": 0.20, "blur_sigma": (0.10, 1.50)}

# ---- losses ------------------------------------------------------------------------------------
ASL_GAMMA_POS, ASL_GAMMA_NEG, ASL_CLIP, ASL_EPS = 0.0, 4.0, 0.05, 1e-8
LAMBDA_CMP, LAMBDA_SEP, SEP_MARGIN = 0.05, 0.1, 0.5
LAMBDA_U, LAMBDA_EXP = 0.5, 0.1
POS_WEIGHT_U_CAP, POS_WEIGHT_U_EPISODES, POS_WEIGHT_U_SEED = 50.0, 4000, SEEDS[0]

# ---- withholding, folds, calibration, selection --------------------------------------------------
STRATIFIED_WITHHOLD = {"head": 1, "medium": 1, "tail": 2, "ultra_rare": 3}   # 7 classes per episode / fold
FOLD_SEED, N_FOLDS = 20260101, 5
CALIB_SPLIT_SEED, CALIB_FRAC = 42, 0.5
EVAL_SEED_BASE, N_EVAL_SEEDS_TRAIN = 1000, 10
BURN_IN_FRAC, KNOWN_MAP_TOL = 0.5, 0.05
SELECTION_KEY = "ensemble_auroc_dis_mean"   # validation u_hat + energy ensemble AUROC on diseased images
CONFORMAL_ALPHA = 0.10                      # unknown operating point: 90% sensitivity target on calibration

# ---- unknown detector (fixed before any evaluation) ----------------------------------------------
KNN_K, KNN_FEATURE = 50, "patch"
ENS3_COMPONENTS = ("uhat", "maxlogit", "knn")     # equal weights
PRIMARY_DETECTOR = "ens3"

# ---- statistics ----------------------------------------------------------------------------------
N_BOOT, N_PERM, N_BOOT_EXT = 2000, 10000, 1000

# ---- external datasets ---------------------------------------------------------------------------
EXT_DATASETS = ["JSIEC", "ODIR"]
ODIR_SPLITS = {"Training Set":      ("train",   "Annotation/training annotation (English).xlsx"),
               "Off-site Test Set": ("offsite", "Annotation/off-site test annotation (English).xlsx"),
               "On-site Test Set":  ("onsite",  "Annotation/on-site test annotation (English).xlsx")}
ODIR_IMAGE_DIR = "Images"

# ---- GPU selection: before anything imports torch ---------------------------------------------------
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE)
if "torch" in sys.modules:
    import torch as _probe
    if _probe.cuda.is_available() and _probe.cuda.is_initialized():
        raise RuntimeError("CUDA was initialised before the GPU pin; restart the kernel and Run All.")
RUN_NAME = "RetinaOWL_protocol_controls"

# ==== PROTOCOL CONTROLS: every setting this notebook adds or overrides ==============================================
# Applied last, so it overrides the training values above. Keep this block when you paste in the paths above.
OUTPUT_ROOT = "./Output_Protocol_Controls"               # everything this notebook writes (new folder)
CTRL_MODELS = {                                          # display name -> finished training output folder (read only)
    "RetinaOWL":        "../Model/Output",
    "UIOS":             "../Baselines/UIOS/Output",
    "RETFound-MAE":     "../Baselines/RetFound/Output",
    "RETFound-DINOv2":  "../Baselines/RetFound_DINO/Output",
    "ML-Decoder":       "../Baselines/ML_Decoder/Output",
    "ResNet-50":        "../Baselines/ResNet50_ASL/Output",
    "Swin-T":           "../Baselines/SwinT_ASL/Output",
    # "A1 no residual branch": "../Ablations/No_Residual_Branch/Output",
}
CTRL_REFERENCE = "RetinaOWL"                             # differences are reported as reference - model
CTRL_SEEDS = [42, 43, 44, 45, 46]                        # training seeds of the withheld-class folds
CTRL_LCO_SEED = 42                                       # the leave-classes-out seed of every run

# ---- §17 kNN gallery controls ---------------------------------------------------------------------------------------
RUN_GALLERY_CONTROL = True
CTRL_GALLERY_DRAWS = 20                                  # random size-matched galleries per seed and fold
CTRL_GALLERY_SEED = 2027

# ---- §18 mixed vs unknown-only cases, operating points --------------------------------------------------------------
RUN_CASE_SPLIT = True
OP_TPR = 0.90                                            # FPR is reported at this TPR
CTRL_MIN_CASES = 5                                       # a fold enters a subset's mean only with >= this many positives
CTRL_N_BOOT, CTRL_BOOT_SEED = 1000, 7                    # paired image bootstrap (same resample for every model)

# ---- §19 residual-map deletion test (RetinaOWL checkpoints of CTRL_REFERENCE) ---------------------------------------
RUN_DELETION = True
DEL_SEEDS = [42, 43, 44, 45, 46]
DEL_FRAC = 0.10                                          # fraction of each image's retina tokens deleted
DEL_RANKINGS = ["heatmap", "residual"]                   # heatmap = alpha_U * (1 - w), the drawn map; residual = 1 - w
DEL_FILLS = ["blur", "mean"]                             # blur: Gaussian blur within the retina; mean: retina mean colour
DEL_BLUR_SIGMA = 8.0                                     # pixels at 518 px
DEL_RANDOM_REPEATS = 3                                   # random 10% draws per image
DEL_SEED = 2027

# ---- §20 size-matched training control ------------------------------------------------------------------------------
RUN_SIZE_MATCHED = True
SM_FOLDS = [1, 2, 3, 4, 5]
SM_SEED = 42                                             # training seed (the leave-classes-out seed)
SM_REMOVAL_SEED = 2027                                   # which random images are removed (fold f uses seed + f)
SM_MATCH_VAL = True                                      # also remove as many validation images as leave-classes-out
RUN_EXTERNAL, RUN_LCO = False, False                     # nothing external or leave-classes-out is run here
CONTROL_SETTINGS = ['RUN_NAME', 'OUTPUT_ROOT', 'CTRL_MODELS', 'CTRL_REFERENCE', 'CTRL_SEEDS', 'CTRL_LCO_SEED', 'RUN_GALLERY_CONTROL', 'CTRL_GALLERY_DRAWS', 'CTRL_GALLERY_SEED', 'RUN_CASE_SPLIT', 'OP_TPR', 'CTRL_MIN_CASES', 'CTRL_N_BOOT', 'CTRL_BOOT_SEED', 'RUN_DELETION', 'DEL_SEEDS', 'DEL_FRAC', 'DEL_RANKINGS', 'DEL_FILLS', 'DEL_BLUR_SIGMA', 'DEL_RANDOM_REPEATS', 'DEL_SEED', 'RUN_SIZE_MATCHED', 'SM_FOLDS', 'SM_SEED', 'SM_REMOVAL_SEED', 'SM_MATCH_VAL', 'RUN_EXTERNAL', 'RUN_LCO']
# ===================================================================================================================
print(f"[cfg] {RUN_NAME} | seeds {SEEDS_THIS_RUN} | Phase A {NUM_EPOCHS_PHASE_A} + Phase B {NUM_EPOCHS_PHASE_B} epochs "
      f"| external={RUN_EXTERNAL} | leave-classes-out={RUN_LCO} folds {LCO_FOLDS} seeds {LCO_SEEDS} | out {OUTPUT_ROOT}")


# In[ ]:


import json as _json
_cfg_here = os.path.join(OUTPUT_ROOT, "reproducibility", "config.json")
if os.path.exists(_cfg_here) and _json.load(open(_cfg_here)).get("RUN_NAME") != RUN_NAME:
    raise RuntimeError(f"OUTPUT_ROOT={OUTPUT_ROOT!r} already holds the outputs of "
                       f"{_json.load(open(_cfg_here)).get('RUN_NAME')!r}; choose another OUTPUT_ROOT.")
_missing = [n for n in ['RUN_NAME', 'OUTPUT_ROOT', 'CTRL_MODELS', 'CTRL_REFERENCE', 'CTRL_SEEDS', 'CTRL_LCO_SEED', 'RUN_GALLERY_CONTROL', 'CTRL_GALLERY_DRAWS', 'CTRL_GALLERY_SEED', 'RUN_CASE_SPLIT', 'OP_TPR', 'CTRL_MIN_CASES', 'CTRL_N_BOOT', 'CTRL_BOOT_SEED', 'RUN_DELETION', 'DEL_SEEDS', 'DEL_FRAC', 'DEL_RANKINGS', 'DEL_FILLS', 'DEL_BLUR_SIGMA', 'DEL_RANDOM_REPEATS', 'DEL_SEED', 'RUN_SIZE_MATCHED', 'SM_FOLDS', 'SM_SEED', 'SM_REMOVAL_SEED', 'SM_MATCH_VAL', 'RUN_EXTERNAL', 'RUN_LCO'] if n not in globals()]
if _missing:
    raise NameError(f"§0 lacks {_missing}: the control block at the end of §0 was removed or not run.")
import glob, json, math, random, re, shutil, time, socket, platform, hashlib
import importlib.metadata as _im
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TVT
import torchvision.transforms.functional as TF
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

SESSION_T0 = time.time()

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(SEEDS[0])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
if DEVICE.type != "cuda":
    raise RuntimeError("No GPU visible: training a ViT-S/14 at 1,369 tokens is not feasible on CPU.")

CKPT_ROOT_OUT = os.path.join(OUTPUT_ROOT, "Model_Checkpoints")
LOG_DIR       = os.path.join(OUTPUT_ROOT, "logs")
RESULTS_DIR   = os.path.join(OUTPUT_ROOT, "results")
JSON_DIR      = os.path.join(OUTPUT_ROOT, "json")
FIG_DIR       = os.path.join(OUTPUT_ROOT, "figures")
PRED_DIR      = os.path.join(OUTPUT_ROOT, "predictions")
FEAT_DIR      = os.path.join(OUTPUT_ROOT, "features")
REPRO_DIR     = os.path.join(OUTPUT_ROOT, "reproducibility")
for _d in (OUTPUT_ROOT, CKPT_ROOT_OUT, LOG_DIR, RESULTS_DIR, JSON_DIR, FIG_DIR, PRED_DIR, FEAT_DIR, REPRO_DIR):
    os.makedirs(_d, exist_ok=True)

def seed_dirs(seed):
    root = os.path.join(CKPT_ROOT_OUT, f"SEED = {seed}")
    d = {"root": root, "phaseA": os.path.join(root, "phaseA"), "phaseB": os.path.join(root, "phaseB")}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d

def _clean(o):
    """JSON-safe copy: NaN/inf -> null, numpy -> python, DataFrames -> records."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, pd.DataFrame):
        return _clean(o.to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return _clean(o.to_dict())
    return o

def write_json(obj, path):
    with open(path, "w") as fh:
        json.dump(_clean(obj), fh, indent=2)

def sha256_file(path, chunk=1 << 20, n=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest() if n is None else h.hexdigest()[:n]

# ---- environment record ------------------------------------------------------------------------
KEY_PACKAGES = ["torch", "torchvision", "transformers", "timm", "fundus_image_toolbox", "numpy", "pandas",
                "scikit-learn", "scipy", "pillow", "matplotlib", "openpyxl"]
ENV_MANIFEST = {"python": sys.version.split()[0], "platform": platform.platform(), "host": socket.gethostname(),
                "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "packages": {}}
for _p in KEY_PACKAGES:
    try:
        ENV_MANIFEST["packages"][_p] = _im.version(_p)
    except Exception:
        ENV_MANIFEST["packages"][_p] = None
_freeze = sorted({f"{d.metadata['Name']}=={d.version}" for d in _im.distributions() if d.metadata["Name"]},
                 key=str.lower)
with open(os.path.join(REPRO_DIR, "requirements_freeze.txt"), "w") as fh:
    fh.write("\n".join(_freeze) + "\n")
write_json(ENV_MANIFEST, os.path.join(REPRO_DIR, "environment.json"))
print(f"[env] {ENV_MANIFEST['gpu']} ({ENV_MANIFEST['gpu_memory_gb']} GB) | python {ENV_MANIFEST['python']} | "
      + " | ".join(f"{k}={v}" for k, v in ENV_MANIFEST["packages"].items()))
print(f"[env] wrote reproducibility/environment.json and requirements_freeze.txt ({len(_freeze)} packages)")


# In[ ]:


PREFLIGHT_ERRORS, PREFLIGHT_WARNINGS = [], []
def _err(m):  PREFLIGHT_ERRORS.append(m);  print(f"[preflight] FAIL  {m}")
def _warn(m): PREFLIGHT_WARNINGS.append(m); print(f"[preflight] WARN  {m}")
def _ok(m):   print(f"[preflight] ok    {m}")

_probe = os.path.join(OUTPUT_ROOT, ".write_probe")
try:
    open(_probe, "w").write("ok"); os.remove(_probe)
    _ok(f"OUTPUT_ROOT writable ({shutil.disk_usage(OUTPUT_ROOT).free / 2**30:.0f} GB free)")
except Exception as e:
    _err(f"OUTPUT_ROOT not writable: {e}")

DATA_PATHS = {}
for _sp, (_img, _csv) in RFMID_LAYOUT.items():
    _ip, _cp = os.path.join(RFMID_ROOT, _img), os.path.join(RFMID_ROOT, _csv)
    if not os.path.exists(_cp):
        _err(f"RFMiD {_sp} labels missing: {_cp}")
        continue
    DATA_PATHS[_sp] = {"img_dir": _ip, "csv": _cp}
    _have_cache = os.path.isdir(os.path.join(RFMID_CACHE, _sp, "images"))
    if os.path.isdir(_ip):
        _ok(f"RFMiD {_sp:5}: labels + {len(os.listdir(_ip))} raw images")
    elif _have_cache:
        _warn(f"RFMiD {_sp}: raw images not mounted; the cache must be complete")
    else:
        _err(f"RFMiD {_sp}: neither raw images ({_ip}) nor a cache exist")

_cfg = os.path.join(DINOV2_ROOT, "config.json")
_w = [f for f in ("model.safetensors", "pytorch_model.bin") if os.path.exists(os.path.join(DINOV2_ROOT, f))]
if not os.path.exists(_cfg) or not _w:
    _err(f"DINOv2 at {DINOV2_ROOT}: needs config.json + weights")
else:
    _c = json.load(open(_cfg))
    (_ok if (_c.get("hidden_size"), _c.get("patch_size")) == (384, 14) else _err)(
        f"DINOv2 config hidden_size={_c.get('hidden_size')} patch_size={_c.get('patch_size')} (need 384, 14)")

if RUN_EXTERNAL:
    for _ds, _root, _cache, _map in (("JSIEC", JSIEC_ROOT, JSIEC_CACHE, JSIEC_MAP_CSV),
                                     ("ODIR", ODIR_ROOT, ODIR_CACHE, ODIR_MAP_CSV)):
        if _ds not in EXT_DATASETS:
            continue
        (_ok if os.path.exists(_map) else _err)(f"{_ds} mapping: {_map}")
        if os.path.isdir(_root):
            _ok(f"{_ds} original dataset: {_root}")
        elif os.path.exists(os.path.join(_cache, "cache_meta.json")):
            _warn(f"{_ds} original not mounted; using the cache at {_cache}")
        else:
            _err(f"{_ds}: neither the original ({_root}) nor a cache ({_cache}) exists")

for _m in ("transformers", "torchvision", "sklearn", "scipy", "matplotlib") + (("openpyxl",) if RUN_EXTERNAL else ()):
    try:
        __import__(_m)
    except ImportError:
        _err(f"python package '{_m}' missing")
try:
    __import__("fundus_image_toolbox"); _ok("fundus_image_toolbox importable (FOV crop for cache building)")
except ImportError:
    _warn("fundus_image_toolbox not importable -- fine only if every cache is already complete")

print("=" * 78)
if PREFLIGHT_ERRORS:
    raise SystemExit(f"Preflight failed with {len(PREFLIGHT_ERRORS)} problem(s); fix the paths in section 0.")
print("  PREFLIGHT PASSED" + (f" with {len(PREFLIGHT_WARNINGS)} warning(s)" if PREFLIGHT_WARNINGS else ""))
print("=" * 78)


# In[ ]:


train_df = pd.read_csv(DATA_PATHS["train"]["csv"])
val_df   = pd.read_csv(DATA_PATHS["val"]["csv"])
test_df  = pd.read_csv(DATA_PATHS["test"]["csv"])
for _d in (train_df, val_df, test_df):
    _d.columns = [("ID" if c.strip().lower() == "id" else c) for c in _d.columns]
    _d["ID"] = _d["ID"].astype(str)

DISEASE_COLS = [c for c in train_df.columns if c not in ("ID", "Disease_Risk")]
NUM_CLASSES = len(DISEASE_COLS)
assert NUM_CLASSES == 45 and all(c in val_df and c in test_df for c in DISEASE_COLS)
pos_counts = train_df[DISEASE_COLS].sum(axis=0)

def tier_of(n):
    if n >= TIER_EDGES["head"]:   return "head"
    if n >= TIER_EDGES["medium"]: return "medium"
    if n >= TIER_EDGES["tail"]:   return "tail"
    return "ultra_rare"
TIER_LABELS = pos_counts.apply(tier_of)
TIERS = ("head", "medium", "tail", "ultra_rare")
pd.DataFrame({"class": DISEASE_COLS, "train_positives": pos_counts.values.astype(int),
              "val_positives": val_df[DISEASE_COLS].sum().values.astype(int),
              "test_positives": test_df[DISEASE_COLS].sum().values.astype(int),
              "tier": TIER_LABELS.values}).to_csv(os.path.join(REPRO_DIR, "class_tiers.csv"), index=False)
print(f"[data] {NUM_CLASSES} classes | train {len(train_df)} | val {len(val_df)} | test {len(test_df)} | tiers: "
      + ", ".join(f"{t}={int((TIER_LABELS == t).sum())}" for t in TIERS) + " | wrote reproducibility/class_tiers.csv")


# In[ ]:


_FIT = None
def get_fit():
    global _FIT
    if _FIT is None:
        import fundus_image_toolbox as _f
        _FIT = _f
    return _FIT

def crop_to_png(src, dst):
    crop = get_fit().crop(src, size=IMG_SIZE)
    if not isinstance(crop, Image.Image):
        crop = Image.fromarray(np.asarray(crop).astype(np.uint8))
    crop = crop.convert("RGB")
    if crop.size != (IMG_SIZE, IMG_SIZE):
        raise ValueError(f"crop returned {crop.size}")
    tmp = dst + ".tmp.png"
    crop.save(tmp)
    os.replace(tmp, dst)                           # an interrupted run never leaves a partial PNG

def _rfmid_raw(split, image_id):
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        p = os.path.join(DATA_PATHS[split]["img_dir"], f"{image_id}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"RFMiD {split} image {image_id} not found in {DATA_PATHS[split]['img_dir']}")

def ensure_rfmid_cache(split, df):
    img_dir = os.path.join(RFMID_CACHE, split, "images")
    os.makedirs(img_dir, exist_ok=True)
    ids = df["ID"].tolist()
    missing = [i for i in ids if not os.path.exists(os.path.join(img_dir, f"{i}.png"))]
    if missing:
        print(f"[cache rfmid {split}] building {len(missing)}/{len(ids)} crops from raw ...")
        t0 = time.time()
        for n, i in enumerate(missing):
            crop_to_png(_rfmid_raw(split, i), os.path.join(img_dir, f"{i}.png"))
            if (n + 1) % 400 == 0:
                print(f"[cache rfmid {split}] {n+1}/{len(missing)} ({time.time()-t0:.0f}s)")
    else:
        print(f"[cache rfmid {split}] complete ({len(ids)} crops) -> reused")
    man = pd.DataFrame({"ID": ids, "png_sha256": [sha256_file(os.path.join(img_dir, f"{i}.png"), n=16) for i in ids]})
    _tmp = os.path.join(RFMID_CACHE, split, f".manifest.{os.getpid()}.csv")
    man.to_csv(_tmp, index=False)
    os.replace(_tmp, os.path.join(RFMID_CACHE, split, "manifest.csv"))
    return hashlib.sha256("".join(man["png_sha256"]).encode()).hexdigest()[:16]

CACHE_FINGERPRINT = {sp: ensure_rfmid_cache(sp, d) for sp, d in (("train", train_df), ("val", val_df), ("test", test_df))}
print(f"[cache rfmid] fingerprints {CACHE_FINGERPRINT}")


# In[ ]:


class RFMiDDataset(Dataset):
    """Cached 518x518 crops. Training augmentation: horizontal flip, colour jitter, gamma, mild blur."""
    def __init__(self, split, df, augment=False, hflip_p=0.0):
        self.df = df.reset_index(drop=True).copy()
        self.img_root = os.path.join(RFMID_CACHE, split, "images")
        self.augment, self.hflip_p = augment, hflip_p
        self.jitter = (TVT.ColorJitter(PHOTO_AUG["brightness"], PHOTO_AUG["contrast"],
                                       PHOTO_AUG["saturation"], PHOTO_AUG["hue"])
                       if augment and USE_PHOTO_AUG else None)
        self.labels = self.df[DISEASE_COLS].values.astype(np.float32)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        iid = self.df.at[idx, "ID"]
        img = Image.open(os.path.join(self.img_root, f"{iid}.png")).convert("RGB")
        if self.hflip_p > 0.0 and random.random() < self.hflip_p:
            img = TF.hflip(img)
        if self.jitter is not None:
            img = self.jitter(img)
            if random.random() < PHOTO_AUG["gamma_p"]:
                img = TF.adjust_gamma(img, random.uniform(*PHOTO_AUG["gamma_range"]))
            if random.random() < PHOTO_AUG["blur_p"]:
                img = TF.gaussian_blur(img, kernel_size=5, sigma=random.uniform(*PHOTO_AUG["blur_sigma"]))
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img_t, torch.from_numpy(self.labels[idx]), iid

def make_loaders(tr_df, va_df):
    tr = RFMiDDataset("train", tr_df, augment=True, hflip_p=HFLIP_P)
    tr_eval = RFMiDDataset("train", tr_df)
    va = RFMiDDataset("val", va_df)
    return (tr, tr_eval, va,
            DataLoader(tr, batch_size=MICRO_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True),
            DataLoader(tr_eval, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
            DataLoader(va, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS))

TRAIN_DF_FULL, VAL_DF_FULL, TEST_DF_FULL = train_df.copy(), val_df.copy(), test_df.copy()
train_ds, train_eval_ds, val_ds, train_loader, train_eval_loader, val_loader = make_loaders(train_df, val_df)
test_ds = RFMiDDataset("test", test_df)
test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# full-split loaders kept separately: the leave-classes-out section rebinds the training ones
TRAIN_EVAL_LOADER_FULL, VAL_LOADER_FULL, TEST_LOADER_FULL = train_eval_loader, val_loader, test_loader

VAL_LABELS_NP, TEST_LABELS_NP = val_ds.labels, test_ds.labels
VAL_DISEASED, TEST_DISEASED = VAL_LABELS_NP.sum(1) > 0, TEST_LABELS_NP.sum(1) > 0
assert MICRO_BATCH_SIZE * ACCUM_STEPS == 16
_x, _y, _ = next(iter(train_loader))
assert _x.shape[1:] == (3, IMG_SIZE, IMG_SIZE) and _y.shape[1] == NUM_CLASSES
print(f"[data] loaders ready | effective batch {MICRO_BATCH_SIZE * ACCUM_STEPS} | augmentation "
      f"{'on' if USE_PHOTO_AUG else 'off'} (training only)")


# In[ ]:


from transformers import Dinov2Model

class Dinov2Backbone(nn.Module):
    """forward_features returns (B, 1+N, D): CLS token at index 0, final layer norm applied."""
    def __init__(self, hf_model):
        super().__init__()
        self.hf = hf_model
        self.embed_dim = hf_model.config.hidden_size
        self.num_prefix_tokens = 1
    def forward_features(self, img_norm):
        return self.hf(pixel_values=img_norm, interpolate_pos_encoding=True).last_hidden_state

_hf, _info = Dinov2Model.from_pretrained(DINOV2_ROOT, output_loading_info=True)
assert not _info["missing_keys"] and not _info.get("mismatched_keys"), f"DINOv2 did not load cleanly: {_info}"
backbone = Dinov2Backbone(_hf).eval()
for p in backbone.parameters():
    p.requires_grad = False
backbone = backbone.to(DEVICE)
EMBED_DIM, NUM_PREFIX_TOKENS = backbone.embed_dim, backbone.num_prefix_tokens
GRID_SIZE = IMG_SIZE // PATCH_SIZE
NUM_PATCHES = GRID_SIZE * GRID_SIZE
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

class LoRALinear(nn.Module):
    """y = W0 x + (alpha / r) B A x with W0 frozen and B initialised to zero."""
    def __init__(self, base, r=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

# transformers names the four attention projections either query/key/value/output.dense or q/k/v/o_proj
LORA_SCHEMES = {"query_key_value": ("attention.attention.query", "attention.attention.key",
                                    "attention.attention.value", "attention.output.dense"),
                "q_k_v_o_proj":    ("attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.o_proj")}
_linear = [n for n, m in backbone.named_modules() if isinstance(m, nn.Linear)]
_n_layers = len(backbone.hf.encoder.layer)
LORA_SCHEME = next(k for k, suf in LORA_SCHEMES.items() if len([n for n in _linear if n.endswith(suf)]) == 4 * _n_layers)
with torch.no_grad():
    _dummy = (torch.rand(2, 3, IMG_SIZE, IMG_SIZE, device=DEVICE) - IMAGENET_MEAN) / IMAGENET_STD
    _ref_out = backbone.forward_features(_dummy)
for _name in [n for n in _linear if n.endswith(LORA_SCHEMES[LORA_SCHEME])]:
    _parent = backbone.get_submodule(_name.rsplit(".", 1)[0])
    setattr(_parent, _name.rsplit(".", 1)[1], LoRALinear(getattr(_parent, _name.rsplit(".", 1)[1])))
for n, p in backbone.named_parameters():
    p.requires_grad = ".lora_" in n
LORA_PARAMS = [p for n, p in backbone.named_parameters() if ".lora_" in n]
with torch.no_grad():
    assert torch.equal(backbone.forward_features(_dummy), _ref_out), "LoRA at init must reproduce the frozen backbone"
assert _ref_out.shape[1:] == (1 + NUM_PATCHES, EMBED_DIM)

def lora_state_dict(m):
    return {k: v.detach().cpu().clone() for k, v in m.backbone.state_dict().items() if ".lora_" in k}

def _translate_lora_keys(sd):
    dst = LORA_SCHEMES[LORA_SCHEME]
    out = {}
    for k, v in sd.items():
        for scheme, src in LORA_SCHEMES.items():
            if scheme != LORA_SCHEME:
                for a, b in zip(src, dst):
                    k = k.replace(f".{a}.lora_", f".{b}.lora_")
        out[k] = v
    return out

def load_lora_state_dict(m, sd):
    sd = _translate_lora_keys(sd)
    expected = {k for k in m.backbone.state_dict() if ".lora_" in k}
    assert expected == set(sd), f"LoRA key mismatch: missing {sorted(expected - set(sd))[:3]}"
    missing, unexpected = m.backbone.load_state_dict(sd, strict=False)
    assert not unexpected and all(".lora_" not in k for k in missing)

print(f"[backbone] DINOv2 ViT-S/14 | {_n_layers} blocks | LoRA r={LORA_RANK} alpha={LORA_ALPHA} on {4*_n_layers} "
      f"linears ({sum(p.numel() for p in LORA_PARAMS):,} params) | init check OK")


# In[ ]:


@torch.no_grad()
def make_token_mask(img_norm):
    img01 = img_norm * IMAGENET_STD + IMAGENET_MEAN
    pix = (img01.amax(dim=1, keepdim=True) > (FOV_THRESH / 255.0)).float()
    r = FOV_ERODE_RADIUS
    eroded = -F.max_pool2d(-pix, kernel_size=2 * r + 1, stride=1, padding=r)
    frac = F.avg_pool2d(eroded, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    tok = (frac >= FOV_TOKEN_KEEP).reshape(frac.shape[0], -1)
    empty = tok.sum(dim=1) < 8
    if empty.any():
        tok[empty] = True
    return tok

def softmax_energy(logits):
    """-logsumexp over the active logits (higher = more unknown); -inf (withheld) logits contribute nothing."""
    return -torch.logsumexp(logits, dim=1)

class DiseaseMemory(nn.Module):
    def __init__(self, embed_dim, num_classes, kp=KP, tau=ATTN_TEMPERATURE, dropout_p=DROPOUT_P):
        super().__init__()
        self.num_classes, self.kp, self.tau = num_classes, kp, tau
        self.scale = embed_dim ** -0.5
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.label_queries = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.prototypes = nn.Parameter(torch.randn(num_classes, kp, embed_dim) * 0.02)
        self.token_dropout = nn.Dropout(dropout_p)
        self.evidence_dropout = nn.Dropout(dropout_p)
    def forward(self, z, active_mask=None):
        z = self.token_dropout(z)
        q, k, v = self.Wq(self.label_queries), self.Wk(z), self.Wv(z)
        attn = (torch.einsum("ca,bna->bcn", q, k) * self.scale).softmax(dim=-1)
        g = self.evidence_dropout(torch.einsum("bcn,bna->bca", attn, v))
        sim = torch.einsum("bca,cka->bck", F.normalize(g, dim=-1), F.normalize(self.prototypes, dim=-1)) / self.tau
        if active_mask is not None:
            sim = sim.masked_fill(~active_mask.view(1, -1, 1), float("-inf"))
        logits = torch.logsumexp(sim, dim=-1)
        if active_mask is not None:
            logits = logits.masked_fill(~active_mask.view(1, -1), float("-inf"))
        return logits, g, attn

class ResidualDecomposer(nn.Module):
    def __init__(self, embed_dim, top_k_frac=TOP_K_RESIDUAL_FRAC, tau_e=DECOMP_TAU_E):
        super().__init__()
        self.top_k_frac = top_k_frac
        self.register_buffer("tau_e", torch.tensor(float(tau_e)))
        self.delta = nn.Parameter(torch.tensor(0.0))
        self.log_Te = nn.Parameter(torch.tensor(0.0))
        self.Wu = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_attn = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.unknown_mlp = nn.Sequential(nn.Linear(2 * embed_dim + 1, embed_dim), nn.GELU(),
                                         nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
                                         nn.Linear(embed_dim // 2, 1))
    def forward(self, z, prototypes, global_img_emb, active_mask=None, token_mask=None):
        B, N, D = z.shape
        p_bar = F.normalize(prototypes, dim=-1)
        p_bar = (p_bar[active_mask] if active_mask is not None else p_bar).reshape(-1, D)
        e = (torch.einsum("bnd,md->bnm", F.normalize(z, dim=-1), p_bar) / self.tau_e).max(dim=-1).values
        w = torch.sigmoid((e - self.delta) / torch.exp(self.log_Te))
        r = (1.0 - w.unsqueeze(-1)) * z
        attn_logits = torch.tanh(self.Wu(r)) @ self.v_attn
        if token_mask is not None:
            attn_logits = attn_logits.masked_fill(~token_mask, float("-inf"))
        alpha_U = torch.softmax(attn_logits, dim=-1)
        g_U = torch.einsum("bn,bnd->bd", alpha_U, r)
        resid_mass = 1.0 - w
        K = max(1, int(round(self.top_k_frac * N)))
        if token_mask is not None:
            rm = resid_mass.masked_fill(~token_mask, float("-inf"))
            topk_vals = torch.topk(rm, k=K, dim=-1).values
            bad = torch.isinf(topk_vals)
            if bad.any():          # fewer than K retina tokens: pad with that image's largest retina value
                safe_max = rm.masked_fill(torch.isinf(rm), torch.finfo(rm.dtype).min).max(dim=1, keepdim=True).values
                topk_vals = torch.where(bad, safe_max.expand_as(topk_vals), topk_vals)
        else:
            topk_vals = torch.topk(resid_mass, k=K, dim=-1).values
        u_hat_logit = self.unknown_mlp(torch.cat([g_U, topk_vals.mean(-1, keepdim=True), global_img_emb], -1)).squeeze(-1)
        hm = alpha_U * resid_mass
        if token_mask is not None:
            hm = hm * token_mask.float()
        return {"w": w, "e": e, "g_U": g_U, "u_hat": torch.sigmoid(u_hat_logit), "u_hat_logit": u_hat_logit,
                "heatmap": hm.reshape(B, GRID_SIZE, GRID_SIZE)}

class RetinaOWL(nn.Module):
    def __init__(self, backbone, memory, decomposer):
        super().__init__()
        self.backbone, self.memory, self.decomposer = backbone, memory, decomposer
    def tokens(self, img_norm):
        return self.backbone.forward_features(img_norm)[:, NUM_PREFIX_TOKENS:, :]
    def forward_known_only(self, img_norm, active_mask=None):
        """Phase A: memory only (the decomposer contributes nothing to the known loss)."""
        logits, g, attn = self.memory(self.tokens(img_norm), active_mask)
        return {"logits": logits, "g": g, "attn": attn}
    def forward(self, img_norm, active_mask=None):
        z = self.tokens(img_norm)
        logits, g, attn = self.memory(z, active_mask)
        token_mask = make_token_mask(img_norm) if USE_FOV_MASK else None
        d = self.decomposer(z, self.memory.prototypes, z.mean(dim=1), active_mask, token_mask=token_mask)
        tm = (token_mask if token_mask is not None else torch.ones(z.shape[:2], dtype=torch.bool, device=z.device))
        tmf = tm.float().unsqueeze(-1)
        feat = F.normalize((z.float() * tmf).sum(1) / tmf.sum(1).clamp(min=1.0), dim=-1)   # kNN feature
        return {"logits": logits, "g": g, "attn": attn, "feat": feat, **d}

def reinit_lora(seed):
    set_seed(seed)
    n = 0
    for m in backbone.modules():
        if isinstance(m, LoRALinear):
            nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5)); nn.init.zeros_(m.lora_B); n += 1
    return n

def build_model(seed):
    """Fresh LoRA draw and fresh heads for `seed` (the frozen ViT weights are shared)."""
    global model, memory, decomposer
    reinit_lora(seed)
    set_seed(seed)
    memory = DiseaseMemory(EMBED_DIM, NUM_CLASSES).to(DEVICE)
    decomposer = ResidualDecomposer(EMBED_DIM).to(DEVICE)
    model = RetinaOWL(backbone, memory, decomposer).to(DEVICE)
    return model

model = build_model(SEEDS[0])
N_TRAINABLE = sum(p.numel() for p in model.parameters() if p.requires_grad)
with torch.no_grad():                                   # Kp = 1: the logit is exactly cosine / tau; masking -> -inf
    _z = torch.randn(2, 9, EMBED_DIM, device=DEVICE)
    _lg, _g, _ = memory(_z)
    assert torch.allclose(_lg, torch.einsum("bca,cka->bck", F.normalize(_g, dim=-1),
                                            F.normalize(memory.prototypes, dim=-1)).squeeze(-1) / memory.tau, atol=1e-4)
    _am = torch.ones(NUM_CLASSES, dtype=torch.bool, device=DEVICE); _am[1] = False
    assert torch.isinf(memory(_z, _am)[0][:, 1]).all()
print(f"[model] Kp={KP} | trainable parameters {N_TRAINABLE:,} (LoRA {sum(p.numel() for p in LORA_PARAMS):,} + "
      f"heads {N_TRAINABLE - sum(p.numel() for p in LORA_PARAMS):,}) | self-tests OK")


# In[ ]:


TIER_TO_INDICES = {t: [i for i, c in enumerate(DISEASE_COLS) if TIER_LABELS[c] == t] for t in TIERS}
TIER_TO_INDICES_FULL = {t: list(v) for t, v in TIER_TO_INDICES.items()}
WITHHOLD_K = sum(STRATIFIED_WITHHOLD.values())
LCO_MASKED = set()            # classes masked in every pass; empty except inside the leave-classes-out section

def sample_withheld_set(rng=None):
    rng = rng if rng is not None else np.random
    chosen = []
    for t, n in STRATIFIED_WITHHOLD.items():
        if n > 0:
            chosen.extend(rng.choice(TIER_TO_INDICES[t], size=n, replace=False).tolist())
    return set(chosen)

def make_active_mask(withheld_idx):
    mask = torch.ones(NUM_CLASSES, dtype=torch.bool, device=DEVICE)
    off = set(withheld_idx) | LCO_MASKED
    if off:
        mask[list(off)] = False
    return mask

def pseudo_unknown_target(labels, withheld_idx):
    if not withheld_idx:
        return torch.zeros(labels.shape[0], device=labels.device)
    widx = torch.tensor(sorted(withheld_idx), device=labels.device)
    return (labels.index_select(1, widx).sum(dim=1) > 0).float()

def unknown_target_np(labels_np, withheld_idx):
    return (labels_np[:, sorted(withheld_idx)].sum(axis=1) > 0).astype(np.float32)

def build_label_holdout_folds(n_folds=N_FOLDS, seed=FOLD_SEED):
    rng = np.random.RandomState(seed)
    remaining = {t: [int(i) for i in rng.permutation(TIER_TO_INDICES_FULL[t])] for t in STRATIFIED_WITHHOLD}
    folds = []
    for _ in range(n_folds):
        chosen = []
        for t, n in STRATIFIED_WITHHOLD.items():
            if n <= 0:
                continue
            if len(remaining[t]) < n:                # ultra-rare has fewer than 5 x 3 classes: refill
                remaining[t] = [int(i) for i in rng.permutation(TIER_TO_INDICES_FULL[t])]
            chosen.extend(remaining[t][:n]); remaining[t] = remaining[t][n:]
        folds.append(set(chosen))
    return folds

FOLDS = build_label_holdout_folds()
_rows = []
for f, w in enumerate(FOLDS):
    assert len(w) == WITHHOLD_K
    for i in sorted(w):
        c = DISEASE_COLS[i]
        _rows.append({"fold": f + 1, "class": c, "tier": TIER_LABELS[c],
                      "train_positives": int(TRAIN_DF_FULL[c].sum()), "val_positives": int(VAL_DF_FULL[c].sum()),
                      "test_positives": int(TEST_DF_FULL[c].sum())})
    _wn = [DISEASE_COLS[i] for i in sorted(w)]
    _rows_img = {sp: int((d[_wn].sum(axis=1) > 0).sum()) for sp, d in (("train", TRAIN_DF_FULL), ("val", VAL_DF_FULL),
                                                                       ("test", TEST_DF_FULL))}
    print(f"[folds] fold {f+1}: {_wn} | images carrying one: train {_rows_img['train']} | val {_rows_img['val']} "
          f"| test {_rows_img['test']}")
HELDOUT = pd.DataFrame(_rows)
HELDOUT.to_csv(os.path.join(REPRO_DIR, "heldout_label_sets.csv"), index=False)
write_json({"fold_seed": FOLD_SEED, "composition": STRATIFIED_WITHHOLD,
            "folds": {f + 1: sorted(DISEASE_COLS[i] for i in w) for f, w in enumerate(FOLDS)}},
           os.path.join(REPRO_DIR, "heldout_label_sets.json"))
print(f"[folds] {N_FOLDS} fixed held-out sets (seed {FOLD_SEED}) -> reproducibility/heldout_label_sets.csv/.json")


# In[ ]:


FULL_MASK = torch.ones(NUM_CLASSES, dtype=torch.bool, device=DEVICE)   # active classes in Phase A

def asymmetric_loss(logits, targets):
    logits, targets = logits.float(), targets.float()
    p = torch.sigmoid(logits)
    p_m = (p - ASL_CLIP).clamp(min=0.0)
    with torch.no_grad():
        w_pos, w_neg = (1.0 - p).pow(ASL_GAMMA_POS), p_m.pow(ASL_GAMMA_NEG)
    loss = -(targets * w_pos * torch.log(p.clamp(min=ASL_EPS))
             + (1.0 - targets) * w_neg * torch.log((1.0 - p_m).clamp(min=ASL_EPS)))
    return loss.mean()

def estimate_unknown_pos_weight(df, n_episodes=POS_WEIGHT_U_EPISODES, cap=POS_WEIGHT_U_CAP, seed=POS_WEIGHT_U_SEED):
    labels_all = torch.tensor(df[DISEASE_COLS].values.astype(np.float32))
    rng = np.random.RandomState(seed)
    n_pos = n_tot = 0
    for _ in range(n_episodes):
        u = pseudo_unknown_target(labels_all, sample_withheld_set(rng))
        n_pos += int(u.sum().item()); n_tot += int(u.shape[0])
    return float(min(cap, (n_tot - n_pos) / max(n_pos, 1))), n_pos / max(n_tot, 1)

POS_WEIGHT_U, U_TRAIN_PREVALENCE = estimate_unknown_pos_weight(train_df)
POS_WEIGHT_U_T = torch.tensor(POS_WEIGHT_U, device=DEVICE)

def compactness_loss(g, prototypes, labels, active_mask):
    dist2 = (F.normalize(g, dim=-1).unsqueeze(2) - F.normalize(prototypes, dim=-1).unsqueeze(0)).pow(2).sum(-1)
    supervised = labels * active_mask.view(1, -1).float()
    return (dist2.min(dim=-1).values * supervised).sum() / supervised.sum().clamp(min=1.0)

def separation_loss(prototypes, active_mask):
    p_active = prototypes[active_mask]
    Ca = p_active.shape[0]
    if Ca < 2:
        return torch.tensor(0.0, device=prototypes.device)
    p_n = F.normalize(p_active, dim=-1).reshape(Ca * p_active.shape[1], -1)
    d = (torch.cdist(p_n, p_n, p=2) ** 2).view(Ca, p_active.shape[1], Ca, p_active.shape[1]).min(3).values.min(1).values
    v = F.relu(SEP_MARGIN - d).masked_fill(torch.eye(Ca, dtype=torch.bool, device=prototypes.device), 0.0)
    return (v ** 2).sum() / max(Ca * (Ca - 1), 1)

def explanation_consistency_loss(attn, w, labels, active_mask, withheld_idx):
    """Known positives should be explained (high w under their attention); withheld positives should not."""
    w_bar = torch.einsum("bcn,bn->bc", attn, w)
    active_pos = (labels > 0.5) & active_mask.view(1, -1)
    term = ((1 - w_bar) * active_pos.float()).sum() / active_pos.float().sum().clamp(min=1.0)
    if withheld_idx:
        widx = torch.tensor(sorted(withheld_idx), device=labels.device)
        wp = labels.index_select(1, widx) > 0.5
        term = term + (w_bar.index_select(1, widx) * wp.float()).sum() / wp.float().sum().clamp(min=1.0)
    return term

def compute_known_loss(logits, g, labels):
    ai = FULL_MASK.nonzero(as_tuple=True)[0]
    asl = asymmetric_loss(logits.index_select(1, ai), labels.index_select(1, ai))
    total = asl + LAMBDA_CMP * compactness_loss(g, memory.prototypes, labels, FULL_MASK) \
                + LAMBDA_SEP * separation_loss(memory.prototypes, FULL_MASK)
    return total

def compute_full_loss(out, labels, active_mask, withheld_idx, u_target):
    ai = active_mask.nonzero(as_tuple=True)[0]
    parts = {"asl": asymmetric_loss(out["logits"].index_select(1, ai), labels.index_select(1, ai)),
             "cmp": compactness_loss(out["g"], memory.prototypes, labels, active_mask),
             "sep": separation_loss(memory.prototypes, active_mask),
             "unknown_bce": F.binary_cross_entropy_with_logits(out["u_hat_logit"], u_target, pos_weight=POS_WEIGHT_U_T),
             "exp": explanation_consistency_loss(out["attn"], out["w"], labels, active_mask, withheld_idx)}
    total = (parts["asl"] + LAMBDA_CMP * parts["cmp"] + LAMBDA_SEP * parts["sep"]
             + LAMBDA_U * parts["unknown_bce"] + LAMBDA_EXP * parts["exp"])
    return total, {k: float(v.item()) for k, v in parts.items()} | {"total": float(total.item())}

print(f"[loss] unknown pos_weight={POS_WEIGHT_U:.3f} (training pseudo-unknown prevalence {U_TRAIN_PREVALENCE:.3f})")


# In[ ]:


def _degenerate(y):
    y = np.asarray(y)
    return y.sum() == 0 or y.sum() == len(y)

def safe_auroc(y, s):
    return float("nan") if _degenerate(y) else float(roc_auc_score(y, s))

def safe_auprc(y, s):
    return float("nan") if _degenerate(y) else float(average_precision_score(y, s))

def fpr_at_tpr(y, s, target=0.95):
    if _degenerate(y):
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.searchsorted(tpr, target, side="left"))
    return 1.0 if i >= len(tpr) else float(fpr[i])

def detection_error(y, s):
    if _degenerate(y):
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.min(0.5 * (1.0 - tpr) + 0.5 * fpr))

def aurc(y, s):
    """Area under the risk-coverage curve when the highest unknown scores are referred first."""
    if len(y) == 0:
        return float("nan")
    ys = np.asarray(y)[np.argsort(s, kind="stable")]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1)).mean())

def unknown_metric_block(y, s, prefix=""):
    return {f"{prefix}auroc": safe_auroc(y, s), f"{prefix}auprc": safe_auprc(y, s),
            f"{prefix}fpr95": fpr_at_tpr(y, s), f"{prefix}det_err": detection_error(y, s),
            f"{prefix}aurc": aurc(y, s)}

def tiered_evaluation(y_true, y_score, cols, tier_labels):
    """Per-class AP; 'primary' mAP averages the evaluable classes outside the ultra-rare tier."""
    ap = {}
    for i, c in enumerate(cols):
        ap[c] = float("nan") if _degenerate(y_true[:, i]) else float(average_precision_score(y_true[:, i], y_score[:, i]))
    tiers = {}
    for t in TIERS:
        v = [ap[c] for c in cols if tier_labels[c] == t and not math.isnan(ap[c])]
        tiers[t] = {"n_evaluable": len(v), "mean_ap": float(np.mean(v)) if v else float("nan")}
    prim = [ap[c] for c in cols if tier_labels[c] != "ultra_rare" and not math.isnan(ap[c])]
    return {"per_class_ap": ap, "tier_results": tiers, "primary_macro_map": float(np.mean(prim)) if prim else float("nan")}

def prediction_sets(p, t, cols_idx=None):
    m = p >= t[None, :]
    if cols_idx is not None:
        keep = np.zeros(p.shape[1], dtype=bool); keep[cols_idx] = True
        m = m & keep[None, :]
    return m

def set_metrics(pred_set, y, cols_idx=None):
    if cols_idx is not None:
        pred_set, y = pred_set[:, cols_idx], y[:, cols_idx] > 0.5
    else:
        y = y > 0.5
    inter, union = (pred_set & y).sum(1), (pred_set | y).sum(1)
    exact = (pred_set == y).all(1)
    npos = y.sum(0)
    cov = np.divide((pred_set & y).sum(0), npos, out=np.full(y.shape[1], np.nan), where=npos > 0)
    return {"exact_match": float(exact.mean()), "jaccard": float(np.where(union > 0, inter / np.maximum(union, 1), 1.0).mean()),
            "avg_set_size": float(pred_set.sum(1).mean()),
            "macro_coverage": float(np.nanmean(cov)) if np.isfinite(cov).any() else float("nan"), "exact_vec": exact}

def prf_metrics(pred_set, y, cols_idx=None):
    if cols_idx is not None:
        pred_set, y = pred_set[:, cols_idx], y[:, cols_idx] > 0.5
    else:
        y = y > 0.5
    tp = (pred_set & y).sum(0).astype(np.float64); fp = (pred_set & ~y).sum(0).astype(np.float64)
    fn = (~pred_set & y).sum(0).astype(np.float64); tn = (~pred_set & ~y).sum(0).astype(np.float64)
    ev = (tp + fn) > 0                                  # a class with no positive is not evaluable
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    sens = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0)
    spec = np.divide(tn, tn + fp, out=np.full_like(tp, np.nan), where=(tn + fp) > 0)
    return {"macro_f1": float(f1[ev].mean()) if ev.any() else float("nan"),
            "micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1e-9)),
            "balanced_acc": float(np.nanmean(0.5 * (sens + spec))),
            "per_class_f1": f1, "per_class_sens": sens, "per_class_spec": spec}

def macro_auroc_auprc(p, y, cols_idx=None):
    idx = range(p.shape[1]) if cols_idx is None else cols_idx
    au = [roc_auc_score(y[:, c], p[:, c]) for c in idx if not _degenerate(y[:, c])]
    ap = [average_precision_score(y[:, c], p[:, c]) for c in idx if not _degenerate(y[:, c])]
    return {"macro_auroc": float(np.mean(au)) if au else float("nan"),
            "macro_auprc": float(np.mean(ap)) if ap else float("nan"), "n_evaluable": len(au)}

ECE_BINS = 15
def expected_calibration_error(p, y, n_bins=ECE_BINS):
    p, y = np.asarray(p).ravel(), (np.asarray(y).ravel() > 0.5).astype(np.float64)
    if len(p) == 0:
        return float("nan")
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(n_bins) if (b == k).any()))

def brier_score(p, y, cols_idx=None):
    if cols_idx is not None:
        p, y = p[:, cols_idx], y[:, cols_idx]
    return float(np.mean((p - (y > 0.5)) ** 2))

def _rank01(x):
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    return order / max(len(x) - 1, 1)

def bootstrap_ci(fn, n, n_boot=N_BOOT, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    v = [x for x in (fn(rng.randint(0, n, n)) for _ in range(n_boot)) if x == x]
    return (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))) if v else (np.nan, np.nan)

def paired_permutation_test(y, sa, sb, n_perm=N_PERM, seed=0):
    """Two-sided paired permutation test on an AUROC difference (scores rank-transformed first)."""
    a, b = _rank01(np.asarray(sa, np.float64)), _rank01(np.asarray(sb, np.float64))
    obs = safe_auroc(y, a) - safe_auroc(y, b)
    if not np.isfinite(obs):
        return float("nan"), float("nan")
    rng, hits = np.random.RandomState(seed), 0
    for _ in range(n_perm):
        sw = rng.rand(len(y)) < 0.5
        if abs(safe_auroc(y, np.where(sw, b, a)) - safe_auroc(y, np.where(sw, a, b))) >= abs(obs) - 1e-12:
            hits += 1
    return float(obs), float((hits + 1) / (n_perm + 1))

def benjamini_hochberg(pvals, q=0.05):
    p = np.asarray(pvals, np.float64); n = len(p); o = np.argsort(p); ranked = p[o]
    passed = ranked <= q * (np.arange(1, n + 1) / n)
    k = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
    rej = np.zeros(n, bool)
    if k:
        rej[o[:k]] = True
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(adj, 0.0, 1.0)
    return rej, out

def conformal_scalar_threshold(s_cal, y_cal, alpha=CONFORMAL_ALPHA):
    """t = s_(k) of the calibration positives with k = floor(alpha (n+1)): >= 1 - alpha sensitivity."""
    s = np.sort(np.asarray(s_cal)[np.asarray(y_cal) > 0.5])
    k = int(np.floor(alpha * (len(s) + 1)))
    return float(s[k - 1]) if k >= 1 else float("-inf")

_y, _s = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])
assert safe_auroc(_y, _s) == 1.0 and fpr_at_tpr(_y, _s) == 0.0 and detection_error(_y, _s) == 0.0
assert abs(aurc(_y, _s) - np.mean([0, 0, 1 / 3, 0.5])) < 1e-12
_ps = prediction_sets(np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0.5, 0.5]))
assert set_metrics(_ps, np.array([[1, 0], [0, 1]]))["jaccard"] == 1.0
assert benjamini_hochberg([0.001, 0.04, 0.9])[0].tolist() == [True, False, False]
print("[metrics] library defined | toy checks OK")


# In[ ]:


def _norm(img):
    return (img.to(DEVICE, non_blocking=True) - IMAGENET_MEAN) / IMAGENET_STD

@torch.no_grad()
def evaluate_known_map(loader, m):
    m.eval()
    sc, lab = [], []
    for img, labels, _ in loader:
        with torch.autocast(device_type="cuda", enabled=USE_AMP):
            out = m.forward_known_only(_norm(img), active_mask=None)
        sc.append(torch.sigmoid(out["logits"]).float().cpu().numpy()); lab.append(labels.numpy())
    m.train()
    return tiered_evaluation(np.concatenate(lab), np.concatenate(sc), DISEASE_COLS, TIER_LABELS)

EVAL_SEEDS_TRAIN = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + N_EVAL_SEEDS_TRAIN))
SELECTION_DETECTORS = ["uhat", "energy", "maxlogit", "ensemble", "ensemble_ml"]

@torch.no_grad()
def evaluate_unknown(loader, m, seeds, diseased_mask):
    """Validation unknown detection under per-batch random withholding (the checkpoint-selection score)."""
    m.eval()
    per = {n: {"auroc": [], "auprc": [], "auroc_dis": []} for n in SELECTION_DETECTORS}
    for s in seeds:
        rng = np.random.RandomState(s)
        acc = {"uhat": [], "energy": [], "maxlogit": [], "u": []}
        for img, labels, _ in loader:
            me = sample_withheld_set(rng)
            with torch.autocast(device_type="cuda", enabled=USE_AMP):
                out = m(_norm(img), make_active_mask(me))
            lg = out["logits"].float()
            acc["uhat"].append(out["u_hat"].float().cpu().numpy())
            acc["energy"].append(softmax_energy(lg).cpu().numpy())
            acc["maxlogit"].append((-lg.max(dim=1).values).cpu().numpy())
            acc["u"].append(pseudo_unknown_target(labels.to(DEVICE), me).cpu().numpy())
        sc = {k: np.concatenate(v) for k, v in acc.items()}
        u = sc.pop("u")
        if _degenerate(u):
            continue
        sc["ensemble"] = 0.5 * _rank01(sc["uhat"]) + 0.5 * _rank01(sc["energy"])
        sc["ensemble_ml"] = 0.5 * _rank01(sc["uhat"]) + 0.5 * _rank01(sc["maxlogit"])
        for n in SELECTION_DETECTORS:
            per[n]["auroc"].append(safe_auroc(u, sc[n])); per[n]["auprc"].append(safe_auprc(u, sc[n]))
            per[n]["auroc_dis"].append(safe_auroc(u[diseased_mask], sc[n][diseased_mask]))
    m.train()
    res = {}
    for n in SELECTION_DETECTORS:
        for k, v in per[n].items():
            v = [x for x in v if x == x]
            res[f"{n}_{k}_mean"] = float(np.mean(v)) if v else float("nan")
            res[f"{n}_{k}_std"] = float(np.std(v)) if v else float("nan")
    return res

@torch.no_grad()
def collect_pass(loader, m, withheld_idx=None):
    """withheld_idx=None -> all classes active (except any leave-classes-out classes)."""
    m.eval()
    act = None if (withheld_idx is None and not LCO_MASKED) else make_active_mask(withheld_idx or [])
    acc = {k: [] for k in ("probs", "uhat", "energy", "maxlogit", "feat", "labels")}
    ids = []
    for img, labels, bid in loader:
        with torch.autocast(device_type="cuda", enabled=USE_AMP):
            out = m(_norm(img), act)
        lg = out["logits"].float()
        acc["probs"].append(torch.sigmoid(lg).cpu().numpy())
        acc["uhat"].append(out["u_hat"].float().cpu().numpy())
        acc["energy"].append(softmax_energy(lg).cpu().numpy())
        acc["maxlogit"].append((-lg.max(dim=1).values).cpu().numpy())
        acc["feat"].append(out["feat"].float().cpu().numpy())
        acc["labels"].append(labels.numpy())
        ids.extend([str(v) for v in bid])
    res = {k: np.concatenate(v) for k, v in acc.items()}
    res["ids"] = np.array(ids)
    return res

# ---- validation calibration split (iterative stratification) -------------------------------------
def iterative_stratification(Y, frac, seed):
    rng = np.random.RandomState(seed)
    Yb = Y > 0.5
    n = Yb.shape[0]
    r = np.array([frac, 1.0 - frac])
    c_j = n * r
    c_ij = Yb.sum(0).astype(np.float64)[:, None] * r[None, :]
    assigned = -np.ones(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    while remaining.any():
        counts = Yb[remaining].sum(0)
        pos = np.where(counts > 0)[0]
        if len(pos) == 0:
            idxs = np.where(remaining)[0]; rng.shuffle(idxs)
            for i in idxs:
                j = int(np.argmax(c_j)); assigned[i] = j; c_j[j] -= 1.0; remaining[i] = False
            break
        lab = int(pos[np.argmin(counts[pos])])
        idxs = np.where(remaining & Yb[:, lab])[0]; rng.shuffle(idxs)
        for i in idxs:
            m_ = c_ij[lab]
            ties = np.where(m_ == m_.max())[0]
            if len(ties) > 1:
                ties = ties[c_j[ties] == c_j[ties].max()]
            j = int(ties[0]) if len(ties) == 1 else int(rng.choice(ties))
            assigned[i] = j
            for l2 in np.where(Yb[i])[0]:
                c_ij[l2, j] -= 1.0
            c_j[j] -= 1.0; remaining[i] = False
    return assigned == 0

CALIB_MASK = iterative_stratification(VAL_LABELS_NP, CALIB_FRAC, CALIB_SPLIT_SEED)
HALF_IS_CALIB = dict(zip(VAL_DF_FULL["ID"].astype(str), CALIB_MASK))
pd.DataFrame({"ID": VAL_DF_FULL["ID"], "half": np.where(CALIB_MASK, "calibration", "evaluation")}).to_csv(
    os.path.join(REPRO_DIR, "val_calibration_split.csv"), index=False)

THRESH_GRID = np.round(np.arange(0.01, 1.00, 0.01), 4)
def fit_global_threshold(p_cal, y_cal):
    best_t, best_f1 = 0.5, -1.0
    for g in THRESH_GRID:
        f1 = prf_metrics(prediction_sets(p_cal, np.full(p_cal.shape[1], g)), y_cal)["macro_f1"]
        if np.isfinite(f1) and f1 > best_f1:
            best_f1, best_t = float(f1), float(g)
    return best_t, best_f1

print(f"[cal] validation halves: calibration {int(CALIB_MASK.sum())} | evaluation {int((~CALIB_MASK).sum())} "
      f"-> reproducibility/val_calibration_split.csv")


# In[ ]:


JOURNAL_PATH = os.path.join(LOG_DIR, f"training_journal_{'_'.join(map(str, SEEDS_THIS_RUN))}.jsonl")
JOURNAL_GLOB = os.path.join(LOG_DIR, "training_journal*.jsonl")
PHASE_A_KEYS = ["train_map", "val_map", "gap"]
PHASE_B_KEYS = ["train_loss", "val_known_map"] + [f"{n}_{k}_mean" for n in SELECTION_DETECTORS
                                                  for k in ("auroc", "auprc", "auroc_dis")]
PHASE_B_BURN_IN = int(BURN_IN_FRAC * NUM_EPOCHS_PHASE_B)

def journal_append(arm, seed, phase, epoch, metrics, extra=None):
    rec = {"ts": time.time(), "arm": arm, "seed": int(seed), "phase": phase, "epoch": int(epoch),
           "metrics": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v))
                       for k, v in metrics.items()}, **(extra or {})}
    with open(JOURNAL_PATH, "a") as fh:
        fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())

def journal_load():
    out = []
    for p in sorted(glob.glob(JOURNAL_GLOB)):
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out

def journal_history(arm, seed, phase, keys, upto):
    recs = {int(r["epoch"]): r for r in journal_load()
            if r.get("arm") == arm and int(r.get("seed", -1)) == int(seed) and r.get("phase") == phase}
    eps = [e for e in sorted(recs) if e < upto]
    if eps != list(range(len(eps))):
        return None, 0
    return {k: [recs[e]["metrics"].get(k) for e in eps] for k in keys}, len(eps)

def save_checkpoint(path, m, optimizer, epoch, step, best_val_map, history, include_decomposer=True):
    payload = {"epoch": epoch, "step": step, "best_val_map": best_val_map, "history": history,
               "memory_state": m.memory.state_dict(), "lora_state": lora_state_dict(m),
               "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
               "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(),
               "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
               "config": {"kp": KP, "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "img_size": IMG_SIZE,
                          "disease_cols": DISEASE_COLS, "lora_scheme": LORA_SCHEME}}
    if include_decomposer:
        payload["decomposer_state"] = m.decomposer.state_dict()
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)

def load_checkpoint(path, m, optimizer=None, load_decomposer=True):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m.memory.load_state_dict(ck["memory_state"], strict=True)
    if load_decomposer and "decomposer_state" in ck:
        m.decomposer.load_state_dict(ck["decomposer_state"], strict=True)
    load_lora_state_dict(m, ck["lora_state"])
    if optimizer is not None and ck.get("optimizer_state") is not None:
        optimizer.load_state_dict(ck["optimizer_state"])
    try:
        torch.set_rng_state(ck["torch_rng_state"].cpu())
        if len(ck["cuda_rng_state"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in ck["cuda_rng_state"]])
        np.random.set_state(ck["numpy_rng_state"]); random.setstate(ck["python_rng_state"])
    except Exception as e:
        print(f"[ckpt] RNG state not restored ({type(e).__name__}); weights are loaded")
    return ck["epoch"], ck["step"], ck["best_val_map"], ck["history"]

def find_resume_checkpoint(ckpt_dir):
    best, best_step = None, -1
    for p in glob.glob(os.path.join(ckpt_dir, "step_*.pt")) + glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")):
        try:
            s = torch.load(p, map_location="cpu", weights_only=False)["step"]
        except Exception:
            continue
        if s > best_step:
            best, best_step = p, s
    return best

def prune_checkpoints(ckpt_dir, keep_latest=2, keep=("best.pt", "selected.pt")):
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")), key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))
    for p in (cands[:-keep_latest] if keep_latest else cands):
        if os.path.basename(p) not in keep:
            os.remove(p)
    for p in glob.glob(os.path.join(ckpt_dir, "step_*.pt")):
        os.remove(p)

def resume_point(seed, arm, phase, dirs, keys, empty_history):
    latest = find_resume_checkpoint(dirs["phaseA" if phase == "A" else "phaseB"])
    start, step, best, hist = 0, 0, -1.0, empty_history
    if latest is not None:
        start, step, best, hist = load_checkpoint(latest, model, None, load_decomposer=(phase == "B"))
        print(f"[phase{phase} s{seed}] resumed from {os.path.basename(latest)} -> epoch {start}")
        jh, n = journal_history(arm, seed, phase, keys, upto=start)
        if jh is not None and n == start:
            hist = jh
    return start, step, best, hist, latest

def _optimizer(lr_heads, lr_lora, with_decomposer):
    heads = list(memory.parameters()) + (list(decomposer.parameters()) if with_decomposer else [])
    return torch.optim.AdamW([{"params": heads, "lr": lr_heads}, {"params": LORA_PARAMS, "lr": lr_lora}],
                             weight_decay=WEIGHT_DECAY)

def run_phase_a(seed, dirs, arm):
    opt = _optimizer(LR_HEADS_A, LR_LORA_A, with_decomposer=False)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    start, step, best, hist, latest = resume_point(seed, arm, "A", dirs, PHASE_A_KEYS, {k: [] for k in PHASE_A_KEYS})
    if latest is not None:
        load_checkpoint(latest, model, opt, load_decomposer=False)
    for epoch in range(start, NUM_EPOCHS_PHASE_A):
        t0 = time.time()
        model.train(); model.backbone.eval()
        losses = []
        opt.zero_grad(set_to_none=True)
        for i, (img, labels, _) in enumerate(train_loader):
            labels = labels.to(DEVICE)
            with torch.autocast(device_type="cuda", enabled=USE_AMP):
                out = model.forward_known_only(_norm(img), active_mask=None)
                loss = compute_known_loss(out["logits"], out["g"], labels)
            scaler.scale(loss / ACCUM_STEPS).backward()
            losses.append(loss.item())
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
                if step % CKPT_EVERY_STEPS == 0:
                    save_checkpoint(os.path.join(dirs["phaseA"], f"step_{step}.pt"), model, opt, epoch, step, best,
                                    hist, include_decomposer=False)
        tr_map = evaluate_known_map(train_eval_loader, model)["primary_macro_map"]
        va_map = evaluate_known_map(val_loader, model)["primary_macro_map"]
        hist["train_map"].append(tr_map); hist["val_map"].append(va_map); hist["gap"].append(tr_map - va_map)
        is_best = va_map > best
        if is_best:
            best = va_map
            save_checkpoint(os.path.join(dirs["phaseA"], "best.pt"), model, opt, epoch, step, best, hist,
                            include_decomposer=False)
        print(f"  [phaseA s{seed}] epoch {epoch+1}/{NUM_EPOCHS_PHASE_A} | loss={np.mean(losses):.4f} | "
              f"train mAP={tr_map:.4f} | val mAP={va_map:.4f}{'  <- best' if is_best else ''} | {(time.time()-t0)/60:.1f} min")
        save_checkpoint(os.path.join(dirs["phaseA"], f"epoch_{epoch}.pt"), model, opt, epoch + 1, step, best, hist,
                        include_decomposer=False)
        journal_append(arm, seed, "A", epoch, {"train_map": tr_map, "val_map": va_map, "gap": tr_map - va_map,
                                               "train_loss": float(np.mean(losses))}, {"epoch_sec": time.time() - t0})
        prune_checkpoints(dirs["phaseA"], keep_latest=2)
    return hist

def run_phase_b(seed, dirs, baseline, arm):
    floor = (1.0 - KNOWN_MAP_TOL) * baseline
    opt = _optimizer(LR_HEADS_B, LR_LORA_B, with_decomposer=True)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    start, step, best, hist, latest = resume_point(seed, arm, "B", dirs, PHASE_B_KEYS, {k: [] for k in PHASE_B_KEYS})
    if latest is not None:
        load_checkpoint(latest, model, opt, load_decomposer=True)
    for epoch in range(start, NUM_EPOCHS_PHASE_B):
        t0 = time.time()
        model.train(); model.backbone.eval()
        losses, parts_sum, nb = [], {}, 0
        opt.zero_grad(set_to_none=True)
        me = act = None
        for i, (img, labels, _) in enumerate(train_loader):
            labels = labels.to(DEVICE)
            if i % ACCUM_STEPS == 0:                     # one withheld set per effective batch
                me = sample_withheld_set()
                act = make_active_mask(me)
            u_t = pseudo_unknown_target(labels, me)
            with torch.autocast(device_type="cuda", enabled=USE_AMP):
                out = model(_norm(img), act)
                loss, parts = compute_full_loss(out, labels, act, me, u_t)
            scaler.scale(loss / ACCUM_STEPS).backward()
            losses.append(parts["total"]); nb += 1
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
                if step % CKPT_EVERY_STEPS == 0:
                    save_checkpoint(os.path.join(dirs["phaseB"], f"step_{step}.pt"), model, opt, epoch, step, best, hist)
        known_map = evaluate_known_map(val_loader, model)["primary_macro_map"]
        ue = evaluate_unknown(val_loader, model, EVAL_SEEDS_TRAIN, VAL_DISEASED)
        rec = {"train_loss": float(np.mean(losses)), "val_known_map": known_map, **ue}
        for k in PHASE_B_KEYS:
            hist[k].append(rec[k])
        status = "burn-in" if epoch < PHASE_B_BURN_IN else ("eligible" if known_map >= floor else "below floor")
        print(f"  [phaseB s{seed}] epoch {epoch+1}/{NUM_EPOCHS_PHASE_B} | loss={rec['train_loss']:.4f} | "
              f"val known mAP={known_map:.4f} (floor {floor:.4f}) | {SELECTION_KEY}={ue[SELECTION_KEY]:.4f} | "
              + " ".join(f"{k}={v / nb:.3f}" for k, v in parts_sum.items() if k != "total")
              + f" | {status} | {(time.time()-t0)/60:.1f} min")
        save_checkpoint(os.path.join(dirs["phaseB"], f"epoch_{epoch}.pt"), model, opt, epoch + 1, step, known_map, hist)
        journal_append(arm, seed, "B", epoch, {**rec, **{f"loss_{k}": v / nb for k, v in parts_sum.items()}},
                       {"epoch_sec": time.time() - t0, "known_floor": floor,
                        "eligible": bool(epoch >= PHASE_B_BURN_IN and known_map >= floor)})
        for p in glob.glob(os.path.join(dirs["phaseB"], "step_*.pt")):
            os.remove(p)
    return hist

def apply_selection(seed, dirs, hist_b, baseline):
    floor = (1.0 - KNOWN_MAP_TOL) * baseline
    n = len(hist_b["val_known_map"])
    elig = [i for i in range(n) if i >= PHASE_B_BURN_IN and hist_b["val_known_map"][i] >= floor
            and hist_b[SELECTION_KEY][i] is not None and np.isfinite(hist_b[SELECTION_KEY][i])]
    print(f"[select s{seed}] rule: epoch >= {PHASE_B_BURN_IN + 1} and val known mAP >= {floor:.4f}; argmax {SELECTION_KEY}")
    for i in range(n):
        print(f"[select s{seed}]   epoch {i+1:2d} | known mAP {hist_b['val_known_map'][i]:.4f} | "
              f"{SELECTION_KEY} {hist_b[SELECTION_KEY][i]:.4f} | {'eligible' if i in elig else '-'}")
    if not elig:
        raise RuntimeError(f"seed {seed}: no Phase B epoch meets the selection rule; report it, do not relax the rule")
    pick = max(elig, key=lambda i: hist_b[SELECTION_KEY][i])
    sel = os.path.join(dirs["phaseB"], "selected.pt")
    shutil.copy(os.path.join(dirs["phaseB"], f"epoch_{pick}.pt"), sel)
    load_checkpoint(sel, model, None, load_decomposer=True)
    re_map = evaluate_known_map(val_loader, model)["primary_macro_map"]
    assert abs(re_map - hist_b["val_known_map"][pick]) < 1e-3, "selected.pt does not reproduce its recorded score"
    prune_checkpoints(dirs["phaseB"], keep_latest=0, keep=("selected.pt",))
    print(f"[select s{seed}] selected epoch {pick+1} | known mAP {hist_b['val_known_map'][pick]:.4f} | "
          f"{SELECTION_KEY} {hist_b[SELECTION_KEY][pick]:.4f} | reload check {re_map:.4f}")
    return pick + 1, sel

def train_one_seed(seed, arm="full"):
    dirs = seed_dirs(seed)
    sel, meta_path = os.path.join(dirs["phaseB"], "selected.pt"), os.path.join(dirs["root"], "train_meta.json")
    build_model(seed)
    if os.path.exists(sel) and os.path.exists(meta_path):
        load_checkpoint(sel, model, None, load_decomposer=True); model.eval()
        print(f"[train] seed {seed} ({arm}): already trained -- selected weights reloaded")
        return json.load(open(meta_path))
    t0 = time.time()
    hist_a = run_phase_a(seed, dirs, arm)
    t_a = time.time() - t0
    va = np.array(hist_a["val_map"], dtype=np.float64)
    assert len(va) == NUM_EPOCHS_PHASE_A, "Phase A incomplete"
    baseline = float(np.median(va[len(va) // 2:]))
    print(f"[phaseA s{seed}] baseline (median of 2nd half) {baseline:.4f} | Phase B floor {(1-KNOWN_MAP_TOL)*baseline:.4f}")
    if find_resume_checkpoint(dirs["phaseB"]) is None:
        load_checkpoint(os.path.join(dirs["phaseA"], "best.pt"), model, None, load_decomposer=False)
    t0 = time.time()
    hist_b = run_phase_b(seed, dirs, baseline, arm)
    t_b = time.time() - t0
    ep, sel = apply_selection(seed, dirs, hist_b, baseline)
    meta = {"seed": seed, "arm": arm, "phase_a_baseline": baseline, "phase_b_floor": (1 - KNOWN_MAP_TOL) * baseline,
            "selected_epoch": ep, "selected_ckpt": sel, "selected_ckpt_sha256": sha256_file(sel),
            "phaseA_sec": t_a, "phaseB_sec": t_b, "history_A": hist_a, "history_B": hist_b}
    write_json(meta, meta_path)
    write_json({k: v for k, v in meta.items() if k != "history_A"}, os.path.join(LOG_DIR, f"selection_{arm}_seed{seed}.json"))
    print(f"[train] seed {seed} ({arm}): Phase A {t_a/60:.0f} min | Phase B {t_b/60:.0f} min | selected epoch {ep}")
    return meta

print(f"[train] driver ready | burn-in {PHASE_B_BURN_IN} epochs | journal {os.path.basename(JOURNAL_PATH)}")


# In[ ]:


DETECTORS = ["uhat", "energy", "maxlogit", "knn", "ensemble", "ensemble_ml", "ens_mk", "ens3", "ens3_ref"]
COMPARISONS = [("ens3", "ensemble_ml"), ("ens3", "ens_mk"), ("ens3", "ensemble"), ("ens3", "maxlogit"),
               ("knn", "maxlogit")]

@torch.no_grad()
def knn_score(ref, qry, k=KNN_K):
    R = torch.from_numpy(np.ascontiguousarray(ref, dtype=np.float32)).to(DEVICE)
    k = min(k, R.shape[0])
    out = []
    for i in range(0, len(qry), 2048):
        Q = torch.from_numpy(np.ascontiguousarray(qry[i:i + 2048], dtype=np.float32)).to(DEVICE)
        out.append((1.0 - torch.topk(Q @ R.T, k=k, dim=1).values[:, -1]).cpu().numpy())
    return np.concatenate(out).astype(np.float64)

def ref_percentile(x, ref):
    r = np.sort(np.asarray(ref, np.float64))
    return np.searchsorted(r, np.asarray(x, np.float64), side="right") / max(len(r), 1)

def detector_scores(uhat, energy, maxlogit, knn, ref=None):
    s = {"uhat": np.asarray(uhat, np.float64), "energy": np.asarray(energy, np.float64),
         "maxlogit": np.asarray(maxlogit, np.float64), "knn": np.asarray(knn, np.float64)}
    s["ensemble"] = 0.5 * _rank01(s["uhat"]) + 0.5 * _rank01(s["energy"])
    s["ensemble_ml"] = 0.5 * _rank01(s["uhat"]) + 0.5 * _rank01(s["maxlogit"])
    s["ens_mk"] = 0.5 * _rank01(s["maxlogit"]) + 0.5 * _rank01(s["knn"])
    s["ens3"] = sum(_rank01(s[c]) for c in ENS3_COMPONENTS) / len(ENS3_COMPONENTS)
    s["ens3_ref"] = (sum(ref_percentile(s[c], ref[c]) for c in ENS3_COMPONENTS) / len(ENS3_COMPONENTS)
                     if ref is not None else np.full(len(s["uhat"]), np.nan))
    return s

_t = detector_scores([0.1, 0.9], [0, 0], [1, 2], [0.2, 0.1])
assert abs(_t["ens3"][1] - 2 / 3) < 1e-12
_r = np.random.RandomState(0).randn(300, 8).astype(np.float32) + 3; _r /= np.linalg.norm(_r, axis=1, keepdims=True)
_q = np.vstack([_r[:3], -_r[:1]]); assert knn_score(_r, _q, k=10).argmax() == 3

def evaluate_seed(m, seed, verbose=True):
    """Thresholds from the validation calibration half; one pass per fold on test; nothing fitted on test."""
    t_all = time.time()
    m.eval()
    R = {"run": RUN_NAME, "seed": int(seed)}
    # ---- 1. validation closed pass -> decision threshold
    PVc = collect_pass(VAL_LOADER_FULL, m)
    cal = np.array([bool(HALF_IS_CALIB[i]) for i in PVc["ids"]])
    thr, cal_f1 = fit_global_threshold(PVc["probs"][cal], PVc["labels"][cal])
    T = np.full(NUM_CLASSES, thr)
    R["threshold"] = {"global": thr, "calibration_macro_f1": cal_f1}
    # ---- 2. known diseases on test (all classes active)
    PTc = collect_pass(TEST_LOADER_FULL, m)
    Y, p = PTc["labels"], PTc["probs"]
    tie = tiered_evaluation(Y, p, DISEASE_COLS, TIER_LABELS)
    ps = prediction_sets(p, T)
    pm, sm, mac = prf_metrics(ps, Y), set_metrics(ps, Y), macro_auroc_auprc(p, Y)
    R["known"] = {"known_map": tie["primary_macro_map"], "known_macro_auroc": mac["macro_auroc"],
                  "known_macro_auprc": mac["macro_auprc"], "macro_f1": pm["macro_f1"], "micro_f1": pm["micro_f1"],
                  "balanced_acc": pm["balanced_acc"], "set_size": sm["avg_set_size"], "coverage": sm["macro_coverage"],
                  "exact_match": sm["exact_match"], "jaccard": sm["jaccard"], "gt_set_size": float(Y.sum(1).mean()),
                  "ece": expected_calibration_error(p, Y), "brier": brier_score(p, Y)}
    R["per_class"] = pd.DataFrame([{"disease": c, "tier": TIER_LABELS[c], "train_n": int(pos_counts[c]),
                                    "test_n": int(Y[:, i].sum()), "auroc": safe_auroc(Y[:, i], p[:, i]),
                                    "ap": tie["per_class_ap"][c], "f1": float(pm["per_class_f1"][i]),
                                    "sensitivity": float(pm["per_class_sens"][i]),
                                    "specificity": float(pm["per_class_spec"][i])} for i, c in enumerate(DISEASE_COLS)])
    R["per_tier"] = pd.DataFrame([{"tier": t, "n_classes": int((TIER_LABELS == t).sum()),
                                   "known_map": tie["tier_results"][t]["mean_ap"],
                                   "macro_auroc": macro_auroc_auprc(p, Y, [i for i, c in enumerate(DISEASE_COLS)
                                                                         if TIER_LABELS[c] == t])["macro_auroc"],
                                   "macro_f1": prf_metrics(ps, Y, [i for i, c in enumerate(DISEASE_COLS)
                                                                   if TIER_LABELS[c] == t])["macro_f1"]} for t in TIERS])
    # ---- 3. kNN reference features: training images (no augmentation)
    PTr = collect_pass(TRAIN_EVAL_LOADER_FULL, m)
    Ytr, Ftr = PTr["labels"], PTr["feat"]
    np.savez_compressed(os.path.join(FEAT_DIR, f"rfmid_patch_seed{seed}.npz"), train=Ftr.astype(np.float16),
                        val=PVc["feat"].astype(np.float16), test=PTc["feat"].astype(np.float16),
                        train_ids=PTr["ids"], val_ids=PVc["ids"], test_ids=PTc["ids"])
    dis_v, dis_t = PVc["labels"].sum(1) > 0, Y.sum(1) > 0
    # ---- 4. withheld-class folds
    rows, pooled_y, pooled_s, long_rows = [], [], {n: [] for n in DETECTORS}, []
    for f, w in enumerate(FOLDS):
        keep = unknown_target_np(Ytr, w) < 0.5
        pv, pt = collect_pass(VAL_LOADER_FULL, m, w), collect_pass(TEST_LOADER_FULL, m, w)
        kv, kt = knn_score(Ftr[keep], PVc["feat"]), knn_score(Ftr[keep], PTc["feat"])
        ref = {"uhat": pv["uhat"], "maxlogit": pv["maxlogit"], "knn": kv}
        Sv = detector_scores(pv["uhat"], pv["energy"], pv["maxlogit"], kv, ref=ref)
        St = detector_scores(pt["uhat"], pt["energy"], pt["maxlogit"], kt, ref=ref)
        uv, ut = unknown_target_np(PVc["labels"], w), unknown_target_np(Y, w)
        t_e = conformal_scalar_threshold(Sv["ens3_ref"][cal], uv[cal])
        flag = St["ens3_ref"] >= t_e
        r = {"fold": f + 1, "withheld": ";".join(sorted(DISEASE_COLS[i] for i in w)), "n_reference": int(keep.sum()),
             "n_pos": int(ut.sum()), "n_pos_dis": int(ut[dis_t].sum()), "n_dis": int(dis_t.sum()), "t_ens3": t_e,
             "op_sensitivity": float(flag[(ut > 0.5) & dis_t].mean()) if ((ut > 0.5) & dis_t).any() else np.nan,
             "op_specificity": float((~flag)[(ut < 0.5) & dis_t].mean())}
        for n in DETECTORS:
            r.update(unknown_metric_block(ut[dis_t], St[n][dis_t], prefix=f"{n}_dis_"))
            r.update(unknown_metric_block(ut, St[n], prefix=f"{n}_all_"))
            r[f"{n}_val_dis_auroc"] = safe_auroc(uv[dis_v], Sv[n][dis_v])
        act = [c for c in range(NUM_CLASSES) if c not in w]
        dm = (ut > 0.5) & (Y[:, act].sum(1) > 0)
        r.update({"n_mixed": int(dm.sum()), "mixed_jaccard": np.nan, "mixed_exact": np.nan, "J_mixed": np.nan})
        if dm.sum() >= 5:
            smx = set_metrics(prediction_sets(pt["probs"][dm], T, cols_idx=act), Y[dm], cols_idx=act)
            r.update({"mixed_jaccard": smx["jaccard"], "mixed_exact": smx["exact_match"],
                      "J_mixed": float((smx["exact_vec"] * flag[dm]).mean())})
        rows.append(r)
        pooled_y.append(ut[dis_t])
        for n in DETECTORS:
            pooled_s[n].append(St[n][dis_t])
        long_rows.append(pd.DataFrame({"fold": f + 1, "ID": PTc["ids"], "unknown": ut.astype(int),
                                       "diseased": dis_t.astype(int), **{n: St[n] for n in DETECTORS}}))
    FU = pd.DataFrame(rows)
    pd.concat(long_rows).to_csv(os.path.join(PRED_DIR, f"rfmid_test_fold_scores_seed{seed}.csv"), index=False)
    pd.DataFrame({"ID": PTc["ids"], **{f"p_{c}": p[:, j] for j, c in enumerate(DISEASE_COLS)},
                  **{f"y_{c}": Y[:, j] for j, c in enumerate(DISEASE_COLS)}}).to_csv(
        os.path.join(PRED_DIR, f"rfmid_test_probs_seed{seed}.csv"), index=False)
    # ---- 5. pooled CIs and paired tests (diseased test images, all folds)
    up = np.concatenate(pooled_y); sp = {n: np.concatenate(v) for n, v in pooled_s.items()}
    R["detector_ci"] = pd.DataFrame([{"detector": n, "auroc_pooled": safe_auroc(up, sp[n]),
                                      **dict(zip(("ci_lo", "ci_hi"), bootstrap_ci(lambda ix, s=sp[n]: safe_auroc(up[ix], s[ix]),
                                                                                  len(up), seed=7)))}
                                     for n in DETECTORS])
    tests = [paired_permutation_test(up, sp[a], sp[b], seed=11) for a, b in COMPARISONS]
    rej, padj = benjamini_hochberg([t[1] for t in tests])
    R["comparisons"] = pd.DataFrame([{"a": a, "b": b, "delta_auroc": d, "p": pv_, "p_bh": pa, "reject_bh05": bool(rj)}
                                     for (a, b), (d, pv_), pa, rj in zip(COMPARISONS, tests, padj, rej)])
    R["fold_unknown"] = FU
    R["headline"] = {"seed": seed, "threshold": thr, **R["known"],
                     **{f"{n}_auroc": float(FU[f"{n}_dis_auroc"].mean()) for n in DETECTORS},
                     **{f"ens3_{k}": float(FU[f"ens3_dis_{k}"].mean()) for k in ("auprc", "fpr95", "det_err", "aurc")},
                     "ens3_auroc_all": float(FU["ens3_all_auroc"].mean()),
                     "unknown_prevalence": float((FU["n_pos_dis"] / FU["n_dis"]).mean()),
                     **{k: float(np.nanmean(FU[k])) for k in ("t_ens3", "op_sensitivity", "op_specificity",
                                                              "mixed_jaccard", "mixed_exact", "J_mixed")},
                     "n_unknown_dis": float(FU["n_pos_dis"].mean()), "n_images": int(len(Y))}
    R["eval_sec"] = time.time() - t_all
    if verbose:
        h = R["headline"]
        print(f"[eval s{seed}] threshold {thr:.2f} | known mAP {h['known_map']:.4f} | macro-F1 {h['macro_f1']:.4f} | "
              f"known AUROC {h['known_macro_auroc']:.4f} | ECE {h['ece']:.4f} | set size {h['set_size']:.2f}")
        print(f"[eval s{seed}] unknown vs known disease (mean of {N_FOLDS} folds) | "
              + " | ".join(f"{n}={h[f'{n}_auroc']:.4f}" for n in ("uhat", "maxlogit", "knn", "ensemble", "ensemble_ml",
                                                                    "ens_mk", "ens3", "ens3_ref")))
        print(f"[eval s{seed}] ens3 AUPRC {h['ens3_auprc']:.4f} | FPR95 {h['ens3_fpr95']:.4f} | operating point "
              f"sens {h['op_sensitivity']:.3f} / spec {h['op_specificity']:.3f} | mixed Jaccard {h['mixed_jaccard']:.4f} | "
              f"J_mixed {h['J_mixed']:.4f} | {R['eval_sec']/60:.1f} min")
        for _, c in R["comparisons"].iterrows():
            print(f"[eval s{seed}]   {c.a:>6} vs {c.b:11} | delta AUROC {c.delta_auroc:+.4f} | p_BH {c.p_bh:.4f}"
                  f"{'  *' if c.reject_bh05 else ''}")
    return R

print(f"[eval] detectors {DETECTORS} | primary = {PRIMARY_DETECTOR} | comparisons {COMPARISONS}")


# In[ ]:


_CFG_NAMES = ["RUN_NAME", "SEEDS", "SEEDS_THIS_RUN", "IMG_SIZE", "PATCH_SIZE", "FOV_THRESH", "TIER_EDGES", "KP",
              "ATTN_TEMPERATURE", "DECOMP_TAU_E", "TOP_K_RESIDUAL_FRAC", "LORA_RANK", "LORA_ALPHA", "DROPOUT_P",
              "USE_FOV_MASK", "FOV_ERODE_RADIUS", "FOV_TOKEN_KEEP", "NUM_EPOCHS_PHASE_A", "NUM_EPOCHS_PHASE_B",
              "MICRO_BATCH_SIZE", "ACCUM_STEPS", "EVAL_BATCH_SIZE", "LR_HEADS_A", "LR_LORA_A", "LR_HEADS_B", "LR_LORA_B",
              "WEIGHT_DECAY", "CKPT_EVERY_STEPS", "HFLIP_P", "USE_PHOTO_AUG", "PHOTO_AUG", "ASL_GAMMA_POS",
              "ASL_GAMMA_NEG", "ASL_CLIP", "LAMBDA_CMP", "LAMBDA_SEP", "SEP_MARGIN", "LAMBDA_U", "LAMBDA_EXP",
              "POS_WEIGHT_U_CAP", "POS_WEIGHT_U_EPISODES", "POS_WEIGHT_U_SEED", "POS_WEIGHT_U", "STRATIFIED_WITHHOLD",
              "FOLD_SEED", "N_FOLDS", "CALIB_SPLIT_SEED", "CALIB_FRAC", "EVAL_SEED_BASE", "N_EVAL_SEEDS_TRAIN",
              "EVAL_SEEDS_TRAIN", "BURN_IN_FRAC", "KNOWN_MAP_TOL", "SELECTION_KEY", "CONFORMAL_ALPHA", "KNN_K",
              "KNN_FEATURE", "ENS3_COMPONENTS", "PRIMARY_DETECTOR", "DETECTORS", "COMPARISONS", "N_BOOT", "N_PERM",
              "N_BOOT_EXT", "ECE_BINS", "EXT_DATASETS", "ODIR_SPLITS", "RUN_LCO", "LCO_FOLDS", "LCO_SEEDS", "LORA_SCHEME",
              "N_TRAINABLE", "NUM_CLASSES"]
CONFIG = {n: globals()[n] for n in _CFG_NAMES}
CONFIG["disease_cols"] = DISEASE_COLS
CONFIG["tiers"] = TIER_LABELS.to_dict()
CONFIG["heldout_label_sets"] = {f + 1: sorted(DISEASE_COLS[i] for i in w) for f, w in enumerate(FOLDS)}
CONFIG["backbone"] = {"name": "DINOv2 ViT-S/14", "embed_dim": EMBED_DIM, "n_blocks": _n_layers,
                      "frozen": True, "adapters": "LoRA on query/key/value/output projections"}
CONFIG["paths"] = {k: os.path.abspath(v) for k, v in (("OUTPUT_ROOT", OUTPUT_ROOT), ("RFMID_ROOT", RFMID_ROOT),
                   ("RFMID_CACHE", RFMID_CACHE), ("DINOV2_ROOT", DINOV2_ROOT), ("JSIEC_ROOT", JSIEC_ROOT),
                   ("ODIR_ROOT", ODIR_ROOT), ("JSIEC_CACHE", JSIEC_CACHE), ("ODIR_CACHE", ODIR_CACHE),
                   ("JSIEC_MAP_CSV", JSIEC_MAP_CSV), ("ODIR_MAP_CSV", ODIR_MAP_CSV))}
write_json(CONFIG, os.path.join(REPRO_DIR, "config.json"))

PROVENANCE = {"rfmid_labels_sha256": {sp: sha256_file(DATA_PATHS[sp]["csv"]) for sp in DATA_PATHS},
              "rfmid_cache_fingerprint": CACHE_FINGERPRINT,
              "dinov2_weights_sha256": {f: sha256_file(os.path.join(DINOV2_ROOT, f)) for f in
                                        ("model.safetensors", "pytorch_model.bin", "config.json")
                                        if os.path.exists(os.path.join(DINOV2_ROOT, f))}}
for _ds, _p in (("JSIEC", JSIEC_MAP_CSV), ("ODIR", ODIR_MAP_CSV)):
    if RUN_EXTERNAL and os.path.exists(_p):
        PROVENANCE[f"{_ds.lower()}_mapping_sha256"] = sha256_file(_p)
write_json(PROVENANCE, os.path.join(REPRO_DIR, "data_provenance.json"))
print(f"[repro] config.json ({len(CONFIG)} entries) and data_provenance.json written to {REPRO_DIR}")


# In[ ]:


_m = build_model(SEEDS_THIS_RUN[0])
_img, _lab, _ = next(iter(val_loader))
with torch.no_grad(), torch.autocast(device_type="cuda", enabled=USE_AMP):
    _out = _m(_norm(_img), make_active_mask(sample_withheld_set(np.random.RandomState(0))))
_tm = make_token_mask(_norm(_img))
assert torch.isfinite(_out["u_hat"]).all() and _out["feat"].shape == (_img.shape[0], EMBED_DIM)
assert float(_out["heatmap"].reshape(_img.shape[0], -1)[~_tm].abs().max()) < 1e-6, "heatmap must be zero off-retina"
_ue = evaluate_unknown(val_loader, _m, EVAL_SEEDS_TRAIN[:1], VAL_DISEASED)
assert np.isfinite(_ue[SELECTION_KEY])
print(f"[smoke] forward, FOV mask ({int(_tm.sum(1).min())}-{int(_tm.sum(1).max())} retina tokens), heatmap, "
      f"selection score OK")


# In[ ]:


import warnings
from scipy.stats import wilcoxon

assert CTRL_REFERENCE in CTRL_MODELS, f"CTRL_REFERENCE={CTRL_REFERENCE!r} is not in CTRL_MODELS"
for _d in ("results", "predictions", "reproducibility"):
    os.makedirs(os.path.join(OUTPUT_ROOT, _d), exist_ok=True)
_HELDOUT_HERE = {str(f + 1): sorted(DISEASE_COLS[i] for i in w) for f, w in enumerate(FOLDS)}
TEST_IDS = TEST_DF_FULL["ID"].astype(str).values
Y_TEST = TEST_DF_FULL.set_index(TEST_DF_FULL["ID"].astype(str))[DISEASE_COLS]
Y_TRAIN = TRAIN_DF_FULL.set_index(TRAIN_DF_FULL["ID"].astype(str))[DISEASE_COLS]
INPUT_FILES = []                 # every file read from a finished run -> reproducibility/input_manifest.csv

def read_exact(path, **kw):
    """Scores are read back bit-exactly. pandas' default float parser can be off by one unit in the last place; the
    rank ensemble has many exactly tied values, and a one-ulp shift breaks those ties differently from the training run."""
    return pd.read_csv(path, float_precision="round_trip", **kw)

def _have(p, errs=None, what=""):
    if os.path.exists(p):
        INPUT_FILES.append(os.path.abspath(p))
        return True
    if errs is not None:
        errs.append(f"{what}: missing {os.path.abspath(p)}")
    return False

RUNS, _errs, _warns = {}, [], []
for _name, _src in CTRL_MODELS.items():
    _cp = os.path.join(_src, "reproducibility", "config.json")
    if not _have(_cp, _errs, _name):
        continue
    _cfg = json.load(open(_cp))
    _bad = [k for k, here in (("disease_cols", DISEASE_COLS), ("heldout_label_sets", _HELDOUT_HERE), ("KNN_K", KNN_K),
                              ("CALIB_SPLIT_SEED", CALIB_SPLIT_SEED), ("CONFORMAL_ALPHA", CONFORMAL_ALPHA))
            if _cfg.get(k) != _clean(here)]
    if _bad:
        _errs.append(f"{_name}: {_bad} differ between {_cp} and this notebook")
        continue
    _info = {"src": _src, "run_name": _cfg.get("RUN_NAME"), "config": _cfg,
             "ens3": list(_cfg.get("ENS3_COMPONENTS", ENS3_COMPONENTS)), "seeds": [], "features": [], "lco_folds": []}
    _have(os.path.join(_src, "results", "rfmid_fold_unknown.csv"), _errs, _name)
    for _s in CTRL_SEEDS:
        if _have(os.path.join(_src, "predictions", f"rfmid_test_fold_scores_seed{_s}.csv"), _errs, _name):
            _info["seeds"].append(_s)
        if _have(os.path.join(_src, "features", f"rfmid_patch_seed{_s}.npz")):
            _info["features"].append(_s)
    if _have(os.path.join(_src, "results", "lco_results.csv")):
        _info["lco_folds"] = [f for f in range(1, N_FOLDS + 1) if _have(
            os.path.join(_src, "predictions", f"lco_fold{f}_test_scores_seed{CTRL_LCO_SEED}.csv"))]
    if not _info["features"]:
        _warns.append(f"{_name}: no saved features -> left out of §17")
    elif len(_info["features"]) < len(CTRL_SEEDS):
        _warns.append(f"{_name}: features for seeds {_info['features']} only -> §17 uses these")
    if len(_info["lco_folds"]) < N_FOLDS:
        _warns.append(f"{_name}: leave-classes-out scores for folds {_info['lco_folds']} only")
    RUNS[_name] = _info
for _w in _warns:
    print(f"[runs] WARN  {_w}")
if _errs:
    for _e in _errs:
        print(f"[runs] FAIL  {_e}")
    raise RuntimeError(f"{len(_errs)} problem(s) with CTRL_MODELS; fix the folders in §0")
assert "uhat" in RUNS[CTRL_REFERENCE]["ens3"], "the reference model must have the residual branch (u_hat)"

def load_scores(name):
    """One long table per run: protocol, seed, fold, ID, unknown, diseased, every detector, t_ens3, saved metrics."""
    info, parts = RUNS[name], []
    FU = read_exact(os.path.join(info["src"], "results", "rfmid_fold_unknown.csv"))
    for s in info["seeds"]:
        d = read_exact(os.path.join(info["src"], "predictions", f"rfmid_test_fold_scores_seed{s}.csv"))
        fu = FU[FU.seed == s][["fold", "t_ens3", "op_specificity", "ens3_dis_auroc", "n_reference"]]
        assert len(fu) == N_FOLDS, f"{name}: rfmid_fold_unknown.csv has {len(fu)} folds for seed {s}"
        d = d.merge(fu.rename(columns={"op_specificity": "op_specificity_saved", "ens3_dis_auroc": "ens3_auroc_saved"}),
                    on="fold", how="left", validate="many_to_one")
        d["protocol"], d["seed"] = "withheld", s
        parts.append(d)
    if info["lco_folds"]:
        LR = read_exact(os.path.join(info["src"], "results", "lco_results.csv"))
        for f in info["lco_folds"]:
            r = LR[(LR.fold == f) & (LR.seed == CTRL_LCO_SEED)]
            assert len(r) == 1, f"{name}: lco_results.csv has {len(r)} rows for fold {f} seed {CTRL_LCO_SEED}"
            d = read_exact(os.path.join(info["src"], "predictions", f"lco_fold{f}_test_scores_seed{CTRL_LCO_SEED}.csv"))
            d["fold"], d["seed"], d["protocol"] = f, CTRL_LCO_SEED, "lco"
            d["t_ens3"], d["op_specificity_saved"] = float(r.t_ens3.iloc[0]), float(r.op_specificity.iloc[0])
            d["ens3_auroc_saved"], d["n_reference"] = float(r.ens3_auroc.iloc[0]), int(r.n_train.iloc[0])
            parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    D["ID"] = D["ID"].astype(str)
    return D

SCORES = {n: load_scores(n) for n in RUNS}

def case_masks(ids, fold):
    """Unknown positives of fold f split by whether the image also carries an active (known) class."""
    Y = Y_TEST.loc[ids].values > 0.5
    w = sorted(FOLDS[fold - 1])
    act = [c for c in range(NUM_CLASSES) if c not in FOLDS[fold - 1]]
    unk, known, dis = Y[:, w].any(1), Y[:, act].any(1), Y.any(1)
    return {"diseased": dis, "negative": dis & ~unk, "all": unk, "mixed": unk & known, "unknown_only": unk & ~known}

def _auc(pos, neg):
    """Mann-Whitney AUROC (ties count 1/2); NaN when a side is empty. Equal to roc_auc_score."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    n = np.sort(neg)
    lo, hi = np.searchsorted(n, pos, side="left"), np.searchsorted(n, pos, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))

_r = np.random.RandomState(0); _yy = _r.rand(200) < 0.3; _ss = np.round(_r.rand(200), 1)
assert abs(_auc(_ss[_yy], _ss[~_yy]) - safe_auroc(_yy, _ss)) < 1e-12

def _ci(v, alpha=0.05):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))) if len(v) else (np.nan, np.nan)

def _pfmt(p, n_boot=CTRL_N_BOOT):
    return "n/a" if not np.isfinite(p) else (f"<= {1.0 / n_boot:.3f}" if p <= 1.0 / n_boot + 1e-12 else f"{p:.3f}")

def _proto_order(protos):
    first = ["withheld", "lco", "size_matched"]
    return [p for p in first if p in list(protos)] + [p for p in protos if p not in first]

def _boot_p(d):
    """Two-sided bootstrap p-value of a difference (share of resamples on the other side of 0, doubled)."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if not len(d):
        return np.nan
    return float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / len(d))))

_cnt = []
for f in range(1, N_FOLDS + 1):
    m = case_masks(TEST_IDS, f)
    _cnt.append({"fold": f, "withheld": ";".join(_HELDOUT_HERE[str(f)]), "n_diseased": int(m["diseased"].sum()),
                 "n_known_disease_negatives": int(m["negative"].sum()), "n_unknown": int(m["all"].sum()),
                 "n_mixed": int(m["mixed"].sum()), "n_unknown_only": int(m["unknown_only"].sum())})
CASE_COUNTS = pd.DataFrame(_cnt)
CASE_COUNTS.to_csv(os.path.join(RESULTS_DIR, "case_counts_per_fold.csv"), index=False)
print(CASE_COUNTS.to_string(index=False))
for _n, _i in RUNS.items():
    print(f"[runs] {_n:18} {_i['run_name']:28} ens3 = {'+'.join(_i['ens3']):22} | withheld seeds {_i['seeds']} | "
          f"features {_i['features']} | leave-classes-out folds {_i['lco_folds']}")


# In[ ]:


GAL_ROWS, GAL_CHECK = [], []
if RUN_GALLERY_CONTROL:
    for name, info in RUNS.items():
        if not info["features"]:
            print(f"[gallery] {name}: no saved features -> skipped")
            continue
        D, comps = SCORES[name], info["ens3"]
        for s in info["features"]:
            Z = np.load(os.path.join(info["src"], "features", f"rfmid_patch_seed{s}.npz"))
            Ftr, Fte = Z["train"].astype(np.float32), Z["test"].astype(np.float32)
            tr_ids, te_ids = Z["train_ids"].astype(str), Z["test_ids"].astype(str)
            assert len(set(tr_ids)) == len(Y_TRAIN) and len(set(te_ids)) == len(Y_TEST), f"{name} seed {s}: features cover other images"
            Ytr = Y_TRAIN.loc[tr_ids].values > 0.5
            rng = np.random.RandomState(CTRL_GALLERY_SEED + s)
            for f in range(1, N_FOLDS + 1):
                w = sorted(FOLDS[f - 1])
                g = D[(D.protocol == "withheld") & (D.seed == s) & (D.fold == f)].set_index("ID").loc[te_ids]
                dis, u = g.diseased.values > 0.5, g.unknown.values
                keep = ~Ytr[:, w].any(1)
                assert int(keep.sum()) == int(g.n_reference.iloc[0]), f"{name} s{s} f{f}: gallery size differs from n_reference"
                lco = D[(D.protocol == "lco") & (D.fold == f)]
                if len(lco):
                    assert int(lco.n_reference.iloc[0]) == int(keep.sum()), \
                        f"{name} f{f}: leave-classes-out training set ({int(lco.n_reference.iloc[0])}) != masked gallery ({int(keep.sum())})"

                def _score(gm):
                    k = knn_score(Ftr[gm], Fte)
                    comp = {"uhat": g.uhat.values, "maxlogit": g.maxlogit.values, "knn": k}
                    e3 = sum(_rank01(comp[c]) for c in comps) / len(comps)
                    return k, safe_auroc(u[dis], k[dis]), safe_auroc(u[dis], e3[dis])

                base = {"model": name, "seed": s, "fold": f, "n_removed_by_protocol": int((~keep).sum())}
                k0, ka, ea = _score(keep)
                GAL_CHECK.append({"model": name, "seed": s, "fold": f, "knn_max_abs_diff": float(np.abs(k0 - g.knn.values).max()),
                                  "ens3_auroc_abs_diff": abs(ea - float(g.ens3_auroc_saved.iloc[0]))})
                GAL_ROWS.append({**base, "variant": "protocol", "draw": 0, "n_gallery": int(keep.sum()),
                                 "n_withheld_images_in_gallery": 0, "knn_auroc": ka, "ens3_auroc": ea})
                _, ka, ea = _score(np.ones(len(Ftr), bool))
                GAL_ROWS.append({**base, "variant": "full", "draw": 0, "n_gallery": len(Ftr),
                                 "n_withheld_images_in_gallery": int((~keep).sum()), "knn_auroc": ka, "ens3_auroc": ea})
                if (~keep).sum() > keep.sum():
                    print(f"[gallery] {name} s{s} f{f}: fewer other images ({int(keep.sum())}) than fold images "
                          f"({int((~keep).sum())}) -> no size-matched gallery")
                for r in range(CTRL_GALLERY_DRAWS if (~keep).sum() <= keep.sum() else 0):
                    gm = np.ones(len(Ftr), bool)
                    gm[rng.choice(np.where(keep)[0], int((~keep).sum()), replace=False)] = False
                    _, ka, ea = _score(gm)
                    GAL_ROWS.append({**base, "variant": "full_size_matched", "draw": r, "n_gallery": int(gm.sum()),
                                     "n_withheld_images_in_gallery": int((~keep).sum()), "knn_auroc": ka, "ens3_auroc": ea})
        print(f"[gallery] {name}: {len(info['features'])} seeds x {N_FOLDS} folds x {2 + CTRL_GALLERY_DRAWS} galleries")
    GCK = pd.DataFrame(GAL_CHECK)
    print(f"[gallery] self-check: max |kNN recomputed - saved| = {GCK.knn_max_abs_diff.max():.2e} | "
          f"max |ens3 AUROC recomputed - reported| = {GCK.ens3_auroc_abs_diff.max():.2e}")
    if GCK.knn_max_abs_diff.max() > 1e-3 or GCK.ens3_auroc_abs_diff.max() > 5e-3:
        print(GCK.sort_values("ens3_auroc_abs_diff", ascending=False).head(10).to_string(index=False))
        raise RuntimeError("the recomputed protocol gallery does not reproduce the saved scores: wrong features or labels")
    print("[gallery] every masked-class gallery equals the leave-classes-out training set of the same fold (asserted)")

    GAL = pd.DataFrame(GAL_ROWS)
    GAL.to_csv(os.path.join(RESULTS_DIR, "gallery_control_per_draw.csv"), index=False)
    _pf = GAL.groupby(["model", "variant", "seed", "fold"])[["knn_auroc", "ens3_auroc", "n_gallery"]].mean().reset_index()
    _pf = _pf.groupby(["model", "variant", "fold"])[["knn_auroc", "ens3_auroc", "n_gallery"]].mean().reset_index()
    _rows = []
    for name in GAL.model.unique():
        D = SCORES[name]
        lco = D[D.protocol == "lco"].groupby("fold").first()
        LR = (read_exact(os.path.join(RUNS[name]["src"], "results", "lco_results.csv"))
              if RUNS[name]["lco_folds"] else pd.DataFrame(columns=["fold", "seed", "knn_auroc", "ens3_auroc"]))
        LR = LR[LR.seed == CTRL_LCO_SEED].set_index("fold")
        for f in list(range(1, N_FOLDS + 1)) + ["mean"]:
            sel = _pf[_pf.model == name] if f == "mean" else _pf[(_pf.model == name) & (_pf.fold == f)]
            row = {"model": name, "fold": f}
            for v in ("protocol", "full", "full_size_matched"):
                x = sel[sel.variant == v]
                row[f"knn_{v}"], row[f"ens3_{v}"] = x.knn_auroc.mean(), x.ens3_auroc.mean()
                row[f"n_gallery_{v}"] = x.n_gallery.mean()
            lf = LR if f == "mean" else LR[LR.index == f]
            row["knn_lco"] = lf.knn_auroc.mean() if len(lf) else np.nan
            row["ens3_lco"] = lf.ens3_auroc.mean() if len(lf) else np.nan
            row["n_gallery_lco"] = (lco.n_reference.mean() if f == "mean" else
                                    (lco.n_reference.get(f, np.nan) if len(lco) else np.nan))
            _rows.append(row)
    GAL_SUM = pd.DataFrame(_rows)
    GAL_SUM.to_csv(os.path.join(RESULTS_DIR, "gallery_control_summary.csv"), index=False)
    print("\n" + "=" * 116)
    print("  kNN GALLERY CONTROLS | masked-class encoder | unknown vs known disease AUROC (mean over seeds; size-matched: and draws)")
    print("=" * 116)
    print(f"  {'model':18} {'fold':>4} | {'kNN: protocol':>13} {'full':>7} {'full,size':>9} {'LCO':>7} | "
          f"{'ens3: protocol':>14} {'full':>7} {'full,size':>9} {'LCO':>7} | gallery n (protocol = LCO / full)")
    for _, r in GAL_SUM.iterrows():
        print(f"  {r.model:18} {str(r.fold):>4} | {r.knn_protocol:13.3f} {r.knn_full:7.3f} {r.knn_full_size_matched:9.3f} "
              f"{r.knn_lco:7.3f} | {r.ens3_protocol:14.3f} {r.ens3_full:7.3f} {r.ens3_full_size_matched:9.3f} "
              f"{r.ens3_lco:7.3f} | {r.n_gallery_protocol:.0f} = {r.n_gallery_lco:.0f} / {r.n_gallery_full:.0f}")
else:
    GAL_SUM = pd.DataFrame()
    print("[gallery] RUN_GALLERY_CONTROL=False -- skipped")


# In[ ]:


CASE_SUBSETS = ("all", "mixed", "unknown_only")
_TPR_TAG = f"fpr_at_tpr{int(round(100 * OP_TPR))}"

def case_rows(name, D):
    rows, configs = [], []
    for (proto, s, f), g in D.groupby(["protocol", "seed", "fold"]):
        g = g.set_index("ID").reindex(TEST_IDS)
        assert not g.ens3.isna().any(), f"{name} {proto} s{s} f{f}: test images missing from the scores"
        m = case_masks(TEST_IDS, int(f))
        assert (m["all"] == (g.unknown.values > 0.5)).all(), f"{name} {proto} f{f}: unknown targets differ from the labels"
        assert (m["diseased"] == (g.diseased.values > 0.5)).all()
        neg, flag = m["negative"], g.ens3_ref.values >= g.t_ens3.values
        r = {"model": name, "protocol": proto, "seed": int(s), "fold": int(f), "t_ens3": float(g.t_ens3.iloc[0]),
             "n_negative": int(neg.sum()), "op_specificity": float((~flag[neg]).mean()),
             "op_specificity_saved": float(g.op_specificity_saved.iloc[0]), "ens3_auroc_saved": float(g.ens3_auroc_saved.iloc[0])}
        _e = g.ens3.values.astype(float)
        r["tie_exact_auroc_shift"] = safe_auroc(m["all"][m["all"] | neg], np.round(_e, 12)[m["all"] | neg]) - \
            safe_auroc(m["all"][m["all"] | neg], _e[m["all"] | neg])
        for sub in CASE_SUBSETS:
            pos = m[sub]; sel = pos | neg
            r[f"n_{sub}"] = int(pos.sum())
            for det in ("ens3", "uhat", "maxlogit", "knn"):
                x = g[det].values.astype(float)
                r[f"{sub}_{det}_auroc"] = (safe_auroc(pos[sel], x[sel]) if pos.any() and np.isfinite(x).all() else np.nan)
            r[f"{sub}_{_TPR_TAG}"] = fpr_at_tpr(pos[sel], g.ens3.values[sel], OP_TPR) if pos.any() else np.nan
            r[f"{sub}_op_sensitivity"] = float(flag[pos].mean()) if pos.any() else np.nan
        rows.append(r)
        configs.append({"model": name, "protocol": proto, "seed": int(s), "fold": int(f), "s": g.ens3.values.astype(float),
                        "negative": neg, **{sub: m[sub] for sub in CASE_SUBSETS}})
    return rows, configs

def diagnose_case_mismatch(CS):
    """For every mismatching (run, protocol, seed, fold): which of the run's own files disagree, and how."""
    CS = CS.assign(d_auroc=(CS.all_ens3_auroc - CS.ens3_auroc_saved).abs(),
                   d_spec=(CS.op_specificity - CS.op_specificity_saved).abs())
    bad = CS[(CS.d_auroc >= 1e-6) | (CS.d_spec >= 1e-6)].sort_values(["d_auroc", "d_spec"], ascending=False)
    print(f"[cases] {len(bad)} of {len(CS)} (run, protocol, seed, fold) rows disagree with what the run reported")
    print(bad.groupby(["model", "protocol"]).agg(rows=("fold", "size"), max_d_auroc=("d_auroc", "max"),
                                                  max_d_spec=("d_spec", "max")).to_string())
    for _, r in bad.head(12).iterrows():
        info, D = RUNS[r.model], SCORES[r.model]
        g = D[(D.protocol == r.protocol) & (D.seed == r.seed) & (D.fold == r.fold)].set_index("ID").reindex(TEST_IDS)
        comps = info["ens3"]
        e_rebuilt = sum(_rank01(g[c].values.astype(float)) for c in comps) / len(comps)
        if r.protocol == "withheld":
            sc_p = os.path.join(info["src"], "predictions", f"rfmid_test_fold_scores_seed{r.seed}.csv")
            res_p = os.path.join(info["src"], "results", "rfmid_fold_unknown.csv")
            js_p = os.path.join(info["src"], "json", f"results_seed{r.seed}.json")
            js = None
            if os.path.exists(js_p):
                fu = [x for x in json.load(open(js_p)).get("fold_unknown", []) if int(x.get("fold", -1)) == r.fold]
                js = fu[0].get("ens3_dis_auroc") if fu else None
        else:
            sc_p = os.path.join(info["src"], "predictions", f"lco_fold{r.fold}_test_scores_seed{r.seed}.csv")
            res_p = os.path.join(info["src"], "results", "lco_results.csv")
            js_p = os.path.join(info["src"], "LCO", f"fold{r.fold}", f"results_lco_fold{r.fold}_seed{r.seed}.json")
            js = json.load(open(js_p)).get("ens3_auroc") if os.path.exists(js_p) else None
        e = g.ens3.values.astype(float)
        print(f"\n  {r.model} | {r.protocol} | seed {r.seed} | fold {r.fold}")
        print(f"    AUROC from the scores file {r.all_ens3_auroc:.6f} | in the results table {r.ens3_auroc_saved:.6f} | "
              f"in the run's JSON {js if js is None else f'{js:.6f}'}")
        print(f"    specificity from the scores file {r.op_specificity:.6f} | in the results table {r.op_specificity_saved:.6f}")
        print(f"    ens3 column vs rank mean of its own components {'+'.join(comps)}: max |diff| "
              f"{np.abs(e - e_rebuilt).max():.2e} | images sharing an ens3 value with another: "
              f"{int(pd.Series(np.round(e, 12)).duplicated(keep=False).sum())}/{len(e)}")
        print(f"    file times: scores {time.ctime(os.path.getmtime(sc_p))} | results table {time.ctime(os.path.getmtime(res_p))}"
              + (f" | JSON {time.ctime(os.path.getmtime(js_p))}" if os.path.exists(js_p) else " | JSON missing"))
    print("\n  Reading: JSON = results table but != scores file -> the scores file was rewritten by a later evaluation "
          "(or the reverse);\n  ens3 column != rank mean of its components -> the ens3 column was produced by another rule;"
          "\n  only the specificity differs -> t_ens3 or ens3_ref in the results table belongs to another evaluation.")

CASE_ELIGIBLE = {sub: {f: int(case_masks(TEST_IDS, f)[sub].sum()) >= CTRL_MIN_CASES for f in range(1, N_FOLDS + 1)}
                 for sub in CASE_SUBSETS}

def boot_fold_means(configs, n_boot=CTRL_N_BOOT, seed=CTRL_BOOT_SEED):
    """{(model, protocol, subset): array} -- index 0 = observed value, 1.. = bootstrap resamples. The value is the mean
    over seeds of the mean over eligible folds of the ens3 AUROC. Every config sees the same resample of test images."""
    groups = {}
    for c in configs:
        groups.setdefault((c["model"], c["protocol"]), []).append(c)
    rng = np.random.RandomState(seed)
    n = len(TEST_IDS)
    resamples = [np.arange(n)] + [rng.randint(0, n, n) for _ in range(n_boot)]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for key, cs in groups.items():
            for sub in CASE_SUBSETS:
                cs_e = [c for c in cs if CASE_ELIGIBLE[sub][c["fold"]]]
                vals = np.full(len(resamples), np.nan)
                if cs_e:
                    for b, ix in enumerate(resamples):
                        per_seed = {}
                        for c in cs_e:
                            sc = c["s"][ix]
                            per_seed.setdefault(c["seed"], []).append(_auc(sc[c[sub][ix]], sc[c["negative"][ix]]))
                        vals[b] = np.mean([np.nanmean(v) for v in per_seed.values()])
                out[(key[0], key[1], sub)] = vals
    return out

def summarise_cases(CS, BOOT, reference=CTRL_REFERENCE):
    summ, vs = [], []
    for (name, proto), g in CS.groupby(["model", "protocol"], sort=False):
        for sub in CASE_SUBSETS:
            el = g[g.fold.map(CASE_ELIGIBLE[sub])]
            per_seed = el.groupby("seed")[[f"{sub}_ens3_auroc", f"{sub}_{_TPR_TAG}", f"{sub}_op_sensitivity",
                                           "op_specificity"] + [f"{sub}_{d}_auroc" for d in ("uhat", "maxlogit", "knn")]].mean()
            v = BOOT.get((name, proto, sub))
            lo, hi = _ci(v[1:]) if v is not None else (np.nan, np.nan)
            row = {"model": name, "protocol": proto, "subset": sub, "n_seeds": int(g.seed.nunique()),
                   "folds_used": ";".join(str(f) for f in sorted(el.fold.unique())),
                   "n_positives_per_seed": int(el[el.seed == el.seed.min()][f"n_{sub}"].sum()) if len(el) else 0,
                   "n_negatives_per_seed": int(el[el.seed == el.seed.min()]["n_negative"].sum()) if len(el) else 0,
                   "ens3_auroc": float(per_seed[f"{sub}_ens3_auroc"].mean()) if len(el) else np.nan,
                   "ens3_auroc_sd_seeds": float(per_seed[f"{sub}_ens3_auroc"].std(ddof=1)) if len(per_seed) > 1 else np.nan,
                   "ci_lo": lo, "ci_hi": hi,
                   _TPR_TAG: float(per_seed[f"{sub}_{_TPR_TAG}"].mean()) if len(el) else np.nan,
                   "op_sensitivity": float(per_seed[f"{sub}_op_sensitivity"].mean()) if len(el) else np.nan,
                   "op_specificity": float(per_seed["op_specificity"].mean()) if len(el) else np.nan,
                   **{f"{d}_auroc": float(per_seed[f"{sub}_{d}_auroc"].mean()) if len(el) else np.nan
                      for d in ("uhat", "maxlogit", "knn")}}
            if v is not None and sub == "mixed" and (name, proto, "unknown_only") in BOOT:
                d = v - BOOT[(name, proto, "unknown_only")]
                row.update({"mixed_minus_unknown_only": float(d[0]), "mmu_ci_lo": _ci(d[1:])[0], "mmu_ci_hi": _ci(d[1:])[1],
                            "mmu_p": _boot_p(d[1:])})
            summ.append(row)
            ref = BOOT.get((reference, proto, sub))
            if name != reference and v is not None and ref is not None:
                d = ref - v
                vs.append({"protocol": proto, "subset": sub, "reference": reference, "model": name,
                           "reference_auroc": float(ref[0]), "model_auroc": float(v[0]), "delta": float(d[0]),
                           "ci_lo": _ci(d[1:])[0], "ci_hi": _ci(d[1:])[1], "p_boot": _boot_p(d[1:])})
    return pd.DataFrame(summ), pd.DataFrame(vs)

def print_case_tables(SUM, VS, title):
    print("\n" + "=" * 132)
    print(f"  {title}")
    print("=" * 132)
    for proto in _proto_order(SUM.protocol.unique()):
        print(f"  protocol: {proto}   (ens3 AUROC [95% CI] | FPR at {OP_TPR:.0%} TPR | sensitivity at t_ens3; "
              f"last column: specificity at t_ens3)")
        print(f"  {'model':18} | " + " | ".join(f"{s:^34}" for s in CASE_SUBSETS) + " | spec@op")
        for name in SUM.model.unique():
            g = SUM[(SUM.protocol == proto) & (SUM.model == name)].set_index("subset")
            if g.empty:
                continue
            cells = [f"{g.at[s, 'ens3_auroc']:.3f} [{g.at[s, 'ci_lo']:.3f},{g.at[s, 'ci_hi']:.3f}] "
                     f"{g.at[s, _TPR_TAG]:.2f} {g.at[s, 'op_sensitivity']:.2f}" for s in CASE_SUBSETS]
            print(f"  {name:18} | " + " | ".join(f"{c:^34}" for c in cells) + f" | {g.at['all', 'op_specificity']:.3f}")
    if len(VS):
        print(f"\n  paired differences {CTRL_REFERENCE} - model (ens3 AUROC, 95% bootstrap CI, two-sided bootstrap p)")
        for proto in _proto_order(VS.protocol.unique()):
            for _, r in VS[VS.protocol == proto].iterrows():
                print(f"  {r.protocol:12} {r.subset:13} {r.model:18} {r.delta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] "
                      f"p {_pfmt(r.p_boot)}")

if RUN_CASE_SPLIT:
    _rows, CASE_CONFIGS = [], []
    for name in RUNS:
        r, c = case_rows(name, SCORES[name])
        _rows += r; CASE_CONFIGS += c
    CS = pd.DataFrame(_rows)
    _d1 = (CS.all_ens3_auroc - CS.ens3_auroc_saved).abs().max()
    _d2 = (CS.op_specificity - CS.op_specificity_saved).abs().max()
    print(f"[cases] self-check: max |all-cases AUROC - reported| = {_d1:.2e} | max |specificity - reported| = {_d2:.2e}")
    if not (_d1 < 1e-6 and _d2 < 1e-6):
        diagnose_case_mismatch(CS)
        raise RuntimeError("recomputed AUROC / specificity differ from what the runs reported: see the diagnosis above")
    CS.to_csv(os.path.join(RESULTS_DIR, "case_split_per_fold.csv"), index=False)
    t0 = time.time()
    CASE_BOOT = boot_fold_means(CASE_CONFIGS)
    CASE_SUM, CASE_VS = summarise_cases(CS, CASE_BOOT)
    CASE_SUM.to_csv(os.path.join(RESULTS_DIR, "case_split_summary.csv"), index=False)
    CASE_VS.to_csv(os.path.join(RESULTS_DIR, "case_split_vs_reference.csv"), index=False)
    print(f"[cases] {len(CS)} (run, protocol, seed, fold) rows | bootstrap {CTRL_N_BOOT} resamples in {time.time()-t0:.0f} s")
    print(f"[cases] folds used (>= {CTRL_MIN_CASES} positives): " +
          " | ".join(f"{s}: {[f for f, ok in CASE_ELIGIBLE[s].items() if ok]}" for s in CASE_SUBSETS))
    print_case_tables(CASE_SUM, CASE_VS, "MIXED vs UNKNOWN-ONLY | unknown vs known-disease AUROC of ens3 | RFMiD test")
else:
    CS, CASE_SUM, CASE_VS, CASE_CONFIGS = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
    print("[cases] RUN_CASE_SPLIT=False -- skipped")


# In[ ]:


_DEL_KEYS = ["IMG_SIZE", "PATCH_SIZE", "FOV_THRESH", "KP", "ATTN_TEMPERATURE", "DECOMP_TAU_E", "TOP_K_RESIDUAL_FRAC",
             "LORA_RANK", "LORA_ALPHA", "DROPOUT_P", "USE_FOV_MASK", "FOV_ERODE_RADIUS", "FOV_TOKEN_KEEP", "FOLD_SEED",
             "N_FOLDS", "NUM_CLASSES"]

@torch.no_grad()
def decompose(m, z, act, tm):
    """The residual branch on given tokens, with a given token mask (exactly what forward() does after the backbone)."""
    return m.decomposer(z, m.memory.prototypes, z.mean(dim=1), act, token_mask=tm)

def select_top(score, tm, frac):
    """Per image: the round(frac * retina tokens) retina tokens with the highest score."""
    B, N = score.shape
    k = (tm.sum(1).float() * frac).round().clamp(min=1).long()
    order = score.float().masked_fill(~tm, float("-inf")).argsort(dim=1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(N, device=score.device).expand(B, N))
    return rank < k[:, None]

def delete_patches(x, sel, fill, pix):
    """x: (B, 3, H, W) in [0, 1]; sel: (B, N) token selection; pix: (B, 1, H, W) retina pixels."""
    pm = F.interpolate(sel.view(-1, 1, GRID_SIZE, GRID_SIZE).float(), scale_factor=PATCH_SIZE, mode="nearest") * pix
    if fill == "blur":
        kz = 2 * int(round(3 * DEL_BLUR_SIGMA)) + 1
        rep = (TF.gaussian_blur(x * pix, [kz, kz], [DEL_BLUR_SIGMA, DEL_BLUR_SIGMA]) /
               TF.gaussian_blur(pix, [kz, kz], [DEL_BLUR_SIGMA, DEL_BLUR_SIGMA]).clamp(min=1e-3))
    elif fill == "mean":
        rep = ((x * pix).sum((2, 3), keepdim=True) / pix.sum((2, 3), keepdim=True).clamp(min=1.0)).expand_as(x)
    else:
        raise ValueError(fill)
    return x * (1 - pm) + rep.clamp(0, 1) * pm

DEL_ROWS = []
if RUN_DELETION:
    _src = RUNS[CTRL_REFERENCE]["src"]
    _bad = {k: (RUNS[CTRL_REFERENCE]["config"].get(k), _clean(globals()[k])) for k in _DEL_KEYS
            if RUNS[CTRL_REFERENCE]["config"].get(k) != _clean(globals()[k])}
    if _bad:
        raise RuntimeError(f"{CTRL_REFERENCE} model settings differ from this notebook (source, here): {_bad}")
    assert GRID_SIZE * PATCH_SIZE == IMG_SIZE
    _conds = ([f"top_{rk}_{fl}" for rk in DEL_RANKINGS for fl in DEL_FILLS] +
              [f"random{r}_{fl}" for r in range(DEL_RANDOM_REPEATS) for fl in DEL_FILLS])
    t0 = time.time()
    for s in DEL_SEEDS:
        ck = os.path.join(_src, "Model_Checkpoints", f"SEED = {s}", "phaseB", "selected.pt")
        sp = os.path.join(_src, "predictions", f"rfmid_test_fold_scores_seed{s}.csv")
        if not (_have(ck) and _have(sp)):
            raise FileNotFoundError(f"{CTRL_REFERENCE} seed {s}: needs {ck} and {sp}")
        saved = read_exact(sp); saved["ID"] = saved["ID"].astype(str)
        saved = saved.set_index(["fold", "ID"])["uhat"]
        build_model(s)
        load_checkpoint(ck, model, None, load_decomposer=True)
        model.eval()
        gen = torch.Generator(device=DEVICE).manual_seed(DEL_SEED + s)
        acts = {f: make_active_mask(FOLDS[f - 1]) for f in range(1, N_FOLDS + 1)}
        first = True
        for img, labels, ids in TEST_LOADER_FULL:
            x = img.to(DEVICE)
            xn = _norm(img)
            pix = (x.amax(dim=1, keepdim=True) > FOV_THRESH / 255.0).float()
            tm = make_token_mask(xn) if USE_FOV_MASK else torch.ones(x.shape[0], GRID_SIZE ** 2, dtype=torch.bool, device=DEVICE)
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=USE_AMP):
                z0 = model.tokens(xn)
                base = {f: decompose(model, z0, acts[f], tm) for f in acts}
                if first:                                  # decompose() must equal the full forward pass
                    ref = model(xn, acts[1])
                    assert torch.allclose(ref["u_hat"].float(), base[1]["u_hat"].float(), atol=1e-5)
                    first = False
                res = {f: {} for f in acts}
                rand_sel = [select_top(torch.rand(tm.shape, generator=gen, device=DEVICE), tm, DEL_FRAC)
                            for _ in range(DEL_RANDOM_REPEATS)]
                for fl in DEL_FILLS:
                    for r, sel in enumerate(rand_sel):
                        zr = model.tokens(_norm(delete_patches(x, sel, fl, pix)))
                        for f in acts:
                            res[f][f"random{r}_{fl}"] = decompose(model, zr, acts[f], tm)
                    for rk in DEL_RANKINGS:
                        for f in acts:
                            score = (base[f]["heatmap"].reshape(x.shape[0], -1) if rk == "heatmap"
                                     else 1.0 - base[f]["w"])
                            sel = select_top(score, tm, DEL_FRAC)
                            zt = model.tokens(_norm(delete_patches(x, sel, fl, pix)))
                            res[f][f"top_{rk}_{fl}"] = decompose(model, zt, acts[f], tm)
            Y = labels.numpy() > 0.5
            for f in acts:
                m_ = {"unknown": Y[:, sorted(FOLDS[f - 1])].any(1), "diseased": Y.any(1)}
                u0 = base[f]["u_hat"].float().cpu().numpy(); l0 = base[f]["u_hat_logit"].float().cpu().numpy()
                cols = {c: (res[f][c]["u_hat"].float().cpu().numpy(), res[f][c]["u_hat_logit"].float().cpu().numpy())
                        for c in _conds}
                for i, iid in enumerate(ids):
                    row = {"seed": s, "fold": f, "ID": str(iid), "unknown": int(m_["unknown"][i]),
                           "diseased": int(m_["diseased"][i]), "n_retina_tokens": int(tm[i].sum()),
                           "uhat": float(u0[i]), "uhat_logit": float(l0[i]), "uhat_saved": float(saved.loc[(f, str(iid))])}
                    for c, (uu, ll) in cols.items():
                        row[f"uhat__{c}"], row[f"logit__{c}"] = float(uu[i]), float(ll[i])
                    DEL_ROWS.append(row)
        print(f"[deletion] seed {s} done | {(time.time() - t0) / 60:.1f} min")
    DEL = pd.DataFrame(DEL_ROWS)
    _dd = (DEL.uhat - DEL.uhat_saved).abs().max()
    print(f"[deletion] self-check: max |u_hat recomputed - saved| = {_dd:.4f}")
    if _dd > 0.01:
        raise RuntimeError("recomputed u_hat differs from the saved scores: wrong checkpoint, seed or data")
    DEL.to_csv(os.path.join(RESULTS_DIR, "deletion_per_image.csv"), index=False)

    _subsets = {"all test images": np.ones(len(DEL), bool), "unknown-positive": DEL.unknown.values == 1,
                "known disease only": (DEL.diseased.values == 1) & (DEL.unknown.values == 0),
                "normal": DEL.diseased.values == 0}
    _summ = []
    _brng = np.random.RandomState(CTRL_BOOT_SEED)
    for fl in DEL_FILLS:
        rnd_u = DEL[[f"uhat__random{r}_{fl}" for r in range(DEL_RANDOM_REPEATS)]].mean(1).values
        rnd_l = DEL[[f"logit__random{r}_{fl}" for r in range(DEL_RANDOM_REPEATS)]].mean(1).values
        for rk in DEL_RANKINGS:
            top_u, top_l = DEL[f"uhat__top_{rk}_{fl}"].values, DEL[f"logit__top_{rk}_{fl}"].values
            for sub, msk in _subsets.items():
                if not msk.any():
                    continue
                d_top, d_rnd = DEL.uhat.values - top_u, DEL.uhat.values - rnd_u
                per_img = pd.DataFrame({"ID": DEL.ID.values[msk], "diff": (d_top - d_rnd)[msk]}).groupby("ID")["diff"].mean().values
                boots = [per_img[_brng.randint(0, len(per_img), len(per_img))].mean() for _ in range(2000)]
                try:
                    p_w = float(wilcoxon(per_img).pvalue) if len(per_img) >= 5 and np.any(per_img != 0) else np.nan
                except ValueError:
                    p_w = np.nan
                _summ.append({"fill": fl, "ranking": rk, "subset": sub, "n_image_folds": int(msk.sum()),
                              "n_images": len(per_img), "uhat_mean": float(DEL.uhat.values[msk].mean()),
                              "drop_targeted": float(d_top[msk].mean()), "drop_random": float(d_rnd[msk].mean()),
                              "drop_targeted_relative": float(d_top[msk].mean() / DEL.uhat.values[msk].mean()),
                              "drop_random_relative": float(d_rnd[msk].mean() / DEL.uhat.values[msk].mean()),
                              "difference": float(per_img.mean()), "diff_ci_lo": _ci(boots)[0], "diff_ci_hi": _ci(boots)[1],
                              "p_wilcoxon": p_w, "share_targeted_larger": float((d_top[msk] > d_rnd[msk]).mean()),
                              "logit_drop_targeted": float((DEL.uhat_logit.values - top_l)[msk].mean()),
                              "logit_drop_random": float((DEL.uhat_logit.values - rnd_l)[msk].mean())})
    DEL_SUM = pd.DataFrame(_summ)
    DEL_SUM.to_csv(os.path.join(RESULTS_DIR, "deletion_summary.csv"), index=False)
    print("\n" + "=" * 128)
    print(f"  RESIDUAL-MAP DELETION | {CTRL_REFERENCE}, seeds {DEL_SEEDS} | {DEL_FRAC:.0%} of retina tokens | "
          f"drop in u_hat = original - edited (mean over image-folds)")
    print("=" * 128)
    print(f"  {'fill':5} {'ranking':8} {'subset':20} {'n':>6} {'u_hat':>6} | {'targeted':>8} {'random':>7} | "
          f"{'difference [95% CI]':>26} {'Wilcoxon p':>10} | {'targeted > random':>17}")
    for _, r in DEL_SUM.iterrows():
        print(f"  {r.fill:5} {r.ranking:8} {r.subset:20} {r.n_image_folds:6d} {r.uhat_mean:6.3f} | {r.drop_targeted:8.4f} "
              f"{r.drop_random:7.4f} | {r.difference:+.4f} [{r.diff_ci_lo:+.4f}, {r.diff_ci_hi:+.4f}] {r.p_wilcoxon:10.1e} | "
              f"{r.share_targeted_larger:16.1%}")
else:
    DEL_SUM = pd.DataFrame()
    print("[deletion] RUN_DELETION=False -- skipped")


# In[ ]:


SM_ROOT = os.path.join(OUTPUT_ROOT, "SizeMatched")
_TRAIN_KEYS = ["IMG_SIZE", "PATCH_SIZE", "FOV_THRESH", "TIER_EDGES", "KP", "ATTN_TEMPERATURE", "DECOMP_TAU_E",
               "TOP_K_RESIDUAL_FRAC", "LORA_RANK", "LORA_ALPHA", "DROPOUT_P", "USE_FOV_MASK", "FOV_ERODE_RADIUS",
               "FOV_TOKEN_KEEP", "NUM_EPOCHS_PHASE_A", "NUM_EPOCHS_PHASE_B", "MICRO_BATCH_SIZE", "ACCUM_STEPS",
               "LR_HEADS_A", "LR_LORA_A", "LR_HEADS_B", "LR_LORA_B", "WEIGHT_DECAY", "HFLIP_P", "USE_PHOTO_AUG",
               "PHOTO_AUG", "ASL_GAMMA_POS", "ASL_GAMMA_NEG", "ASL_CLIP", "LAMBDA_CMP", "LAMBDA_SEP", "SEP_MARGIN",
               "LAMBDA_U", "LAMBDA_EXP", "POS_WEIGHT_U_CAP", "POS_WEIGHT_U_EPISODES", "POS_WEIGHT_U_SEED",
               "STRATIFIED_WITHHOLD", "FOLD_SEED", "N_FOLDS", "CALIB_SPLIT_SEED", "CALIB_FRAC", "EVAL_SEED_BASE",
               "N_EVAL_SEEDS_TRAIN", "BURN_IN_FRAC", "KNOWN_MAP_TOL", "SELECTION_KEY", "CONFORMAL_ALPHA", "KNN_K",
               "KNN_FEATURE", "ENS3_COMPONENTS"]
_SM_NAMES = ("train_df", "val_df", "train_ds", "train_eval_ds", "val_ds", "train_loader", "train_eval_loader",
             "val_loader", "VAL_LABELS_NP", "VAL_DISEASED", "TIER_TO_INDICES", "LCO_MASKED", "FULL_MASK",
             "POS_WEIGHT_U", "POS_WEIGHT_U_T", "U_TRAIN_PREVALENCE", "CKPT_ROOT_OUT", "LOG_DIR", "JOURNAL_PATH",
             "JOURNAL_GLOB", "RUN_NAME")
_SM_SAVED = None

if RUN_SIZE_MATCHED:
    _bad = {k: (RUNS[CTRL_REFERENCE]["config"].get(k), _clean(CONFIG[k])) for k in _TRAIN_KEYS
            if RUNS[CTRL_REFERENCE]["config"].get(k) != _clean(CONFIG[k])}
    if _bad:
        raise RuntimeError(f"training settings differ from {CTRL_REFERENCE}'s config.json (reference, here): {_bad}")

# ---- which images are removed: fixed in advance, written before any training ------------------------------------------
SM_REMOVED, _rm, _plan = {}, [], []
for _f in range(1, N_FOLDS + 1):
    _wn = [DISEASE_COLS[i] for i in sorted(FOLDS[_f - 1])]
    _rng = np.random.RandomState(SM_REMOVAL_SEED + _f)
    _p = {"fold": _f}
    for _sp, _d in (("train", TRAIN_DF_FULL), ("val", VAL_DF_FULL)):
        _has = (_d[_wn].sum(axis=1) > 0).values
        _n = int(_has.sum()) if (_sp == "train" or SM_MATCH_VAL) else 0
        _pool = np.sort(_d.loc[~_has, "ID"].astype(str).values)
        assert _n <= len(_pool), f"fold {_f} {_sp}: cannot remove {_n} of {len(_pool)} images"
        SM_REMOVED[(_f, _sp)] = set(_rng.choice(_pool, _n, replace=False).tolist())
        _rm += [{"fold": _f, "split": _sp, "ID": i} for i in sorted(SM_REMOVED[(_f, _sp)], key=lambda v: (len(v), v))]
        _p.update({f"{_sp}_lco_removed": int(_has.sum()), f"{_sp}_size_matched_removed": _n,
                   f"{_sp}_size_matched_n": len(_d) - _n, f"{_sp}_fold_class_images_kept": int(_has.sum())})
    _plan.append(_p)
pd.DataFrame(_rm).to_csv(os.path.join(REPRO_DIR, "size_matched_removed_images.csv"), index=False)
SM_PLAN = pd.DataFrame(_plan)
print(SM_PLAN.to_string(index=False))
_lco_rm_p = os.path.join(RUNS[CTRL_REFERENCE]["src"], "reproducibility", "lco_removed_images.csv")
if _have(_lco_rm_p):                                      # the counts must equal the reference run's leave-classes-out
    _lrm = pd.read_csv(_lco_rm_p).groupby(["fold", "split"]).size()
    for _, _r in SM_PLAN.iterrows():
        assert _lrm.get((_r.fold, "train"), 0) == _r.train_size_matched_removed, f"fold {_r.fold}: train count differs"
        if SM_MATCH_VAL:
            assert _lrm.get((_r.fold, "val"), 0) == _r.val_size_matched_removed, f"fold {_r.fold}: val count differs"
    print(f"[size-matched] removal counts equal {CTRL_REFERENCE}'s lco_removed_images.csv for every fold")

def sm_configure(fold):
    """Rebind every global the training driver reads to the fold-f size-matched setting (all 45 classes active)."""
    global _SM_SAVED, train_df, val_df, train_ds, train_eval_ds, val_ds, train_loader, train_eval_loader, val_loader
    global VAL_LABELS_NP, VAL_DISEASED, TIER_TO_INDICES, LCO_MASKED, FULL_MASK, POS_WEIGHT_U, POS_WEIGHT_U_T
    global U_TRAIN_PREVALENCE, CKPT_ROOT_OUT, LOG_DIR, JOURNAL_PATH, JOURNAL_GLOB, RUN_NAME
    assert _SM_SAVED is None, "sm_configure called twice without sm_restore"
    _SM_SAVED = {n: globals()[n] for n in _SM_NAMES}
    train_df = TRAIN_DF_FULL[~TRAIN_DF_FULL["ID"].astype(str).isin(SM_REMOVED[(fold, "train")])].reset_index(drop=True)
    val_df = VAL_DF_FULL[~VAL_DF_FULL["ID"].astype(str).isin(SM_REMOVED[(fold, "val")])].reset_index(drop=True)
    train_ds, train_eval_ds, val_ds, train_loader, train_eval_loader, val_loader = make_loaders(train_df, val_df)
    VAL_LABELS_NP = val_ds.labels
    VAL_DISEASED = VAL_LABELS_NP.sum(axis=1) > 0
    LCO_MASKED = set()
    FULL_MASK = make_active_mask([])
    TIER_TO_INDICES = {t: list(v) for t, v in TIER_TO_INDICES_FULL.items()}
    POS_WEIGHT_U, U_TRAIN_PREVALENCE = estimate_unknown_pos_weight(train_df)
    POS_WEIGHT_U_T = torch.tensor(POS_WEIGHT_U, device=DEVICE)
    froot = os.path.join(SM_ROOT, f"fold{fold}")
    CKPT_ROOT_OUT, LOG_DIR = os.path.join(froot, "Model_Checkpoints"), os.path.join(froot, "logs")
    for d in (froot, CKPT_ROOT_OUT, LOG_DIR):
        os.makedirs(d, exist_ok=True)
    JOURNAL_PATH = os.path.join(LOG_DIR, f"training_journal_{SM_SEED}.jsonl")
    JOURNAL_GLOB = os.path.join(LOG_DIR, "training_journal*.jsonl")
    RUN_NAME = f"RetinaOWL_SizeMatched_fold{fold}"
    wn = [DISEASE_COLS[i] for i in sorted(FOLDS[fold - 1])]
    print(f"[size-matched f{fold}] fold classes (kept in training): {wn}")
    print(f"[size-matched f{fold}] train {len(train_df)}/{len(TRAIN_DF_FULL)} | val {len(val_df)}/{len(VAL_DF_FULL)} | "
          f"active classes {int(FULL_MASK.sum())} | unknown pos_weight {POS_WEIGHT_U:.3f} | output {froot}")
    return froot

def sm_restore():
    global _SM_SAVED
    if _SM_SAVED is None:
        return
    for n, v in _SM_SAVED.items():
        globals()[n] = v
    _SM_SAVED = None
    assert not LCO_MASKED and bool(FULL_MASK.all()) and len(train_df) == len(TRAIN_DF_FULL), "state not restored"

def evaluate_size_matched(m, fold, seed):
    """Test split under fold f's mask; thresholds from the full validation calibration half (as in both protocols)."""
    w = FOLDS[fold - 1]
    active = [c for c in range(NUM_CLASSES) if c not in w]
    m.eval()
    P = collect_pass(TEST_LOADER_FULL, m, w)
    Y = P["labels"]; dis = Y.sum(1) > 0; u = unknown_target_np(Y, w)
    PTr = collect_pass(TRAIN_EVAL_LOADER_FULL, m)                      # every training image, encoded by this model
    keep = unknown_target_np(PTr["labels"], w) < 0.5                  # the protocol gallery: no fold-f class
    own = keep & ~np.isin(PTr["ids"], list(SM_REMOVED[(fold, "train")]))
    pv = collect_pass(VAL_LOADER_FULL, m, w)
    calv = np.array([bool(HALF_IS_CALIB[i]) for i in pv["ids"]])
    uv = unknown_target_np(pv["labels"], w)
    S, extra = {}, {}
    for gname, gm in (("protocol", keep), ("own", own)):
        kv, kt = knn_score(PTr["feat"][gm], pv["feat"]), knn_score(PTr["feat"][gm], P["feat"])
        ref = {"uhat": pv["uhat"], "maxlogit": pv["maxlogit"], "knn": kv}
        St = detector_scores(P["uhat"], P["energy"], P["maxlogit"], kt, ref=ref)
        Sv = detector_scores(pv["uhat"], pv["energy"], pv["maxlogit"], kv, ref=ref)
        if gname == "protocol":
            S, SV = St, Sv
        else:
            extra = {"knn_own_auroc": safe_auroc(u[dis], St["knn"][dis]), "ens3_own_auroc": safe_auroc(u[dis], St["ens3"][dis]),
                     "n_gallery_own": int(gm.sum())}
    thr, _ = fit_global_threshold(pv["probs"][calv][:, active], pv["labels"][calv][:, active])
    ps = prediction_sets(P["probs"], np.full(NUM_CLASSES, thr), cols_idx=active)
    pm = prf_metrics(ps, Y, active)
    t_e = conformal_scalar_threshold(SV["ens3_ref"][calv], uv[calv])
    flag = S["ens3_ref"] >= t_e
    out = {"fold": fold, "seed": seed, "withheld": [DISEASE_COLS[i] for i in sorted(w)], "n_train": int(len(train_df)),
           "n_val": int(len(val_df)), "n_gallery": int(keep.sum()), **extra,
           "n_images": int(len(Y)), "n_unknown": int(u[dis].sum()), "threshold": float(thr),
           "macro_f1": pm["macro_f1"], "micro_f1": pm["micro_f1"],
           "known_map": tiered_evaluation(Y[:, active], P["probs"][:, active], [DISEASE_COLS[i] for i in active],
                                          TIER_LABELS)["primary_macro_map"],
           "t_ens3": t_e, "op_sensitivity": float(flag[(u > 0.5) & dis].mean()) if ((u > 0.5) & dis).any() else np.nan,
           "op_specificity": float((~flag)[(u < 0.5) & dis].mean())}
    for n in DETECTORS:
        b = unknown_metric_block(u[dis], S[n][dis])
        out[f"{n}_auroc"] = b["auroc"]
        if n == "ens3":
            out.update({f"ens3_{k}": b[k] for k in ("auprc", "fpr95", "det_err", "aurc")})
    pd.DataFrame({"ID": P["ids"], "unknown": u.astype(int), "diseased": dis.astype(int), **{n: S[n] for n in DETECTORS}}) \
      .to_csv(os.path.join(PRED_DIR, f"size_matched_fold{fold}_test_scores_seed{seed}.csv"), index=False)
    return out

print(f"[size-matched] machinery ready | outputs under {SM_ROOT} | removed IDs -> reproducibility/size_matched_removed_images.csv")


# In[ ]:


SM_RESULTS, SM_FAILURES = [], []
if RUN_SIZE_MATCHED:
    for _f in SM_FOLDS:
        _out = os.path.join(SM_ROOT, f"fold{_f}", f"results_size_matched_fold{_f}_seed{SM_SEED}.json")
        if os.path.exists(_out) and os.path.exists(os.path.join(PRED_DIR, f"size_matched_fold{_f}_test_scores_seed{SM_SEED}.csv")):
            SM_RESULTS.append(json.load(open(_out)))
            print(f"[size-matched] fold {_f} seed {SM_SEED}: already evaluated -- loaded")
            continue
        print(f"\n{'='*84}\n[size-matched] fold {_f} | seed {SM_SEED} | {(time.time()-SESSION_T0)/3600:.2f} h into this session\n{'='*84}")
        try:
            sm_configure(_f)
            _meta = train_one_seed(SM_SEED, arm=f"size_matched_fold{_f}")
            _r = evaluate_size_matched(model, _f, SM_SEED)
            _r.update({"selected_epoch": _meta["selected_epoch"], "phase_a_baseline": _meta["phase_a_baseline"],
                       "selected_ckpt_sha256": _meta.get("selected_ckpt_sha256")})
        except Exception as _e:
            import traceback; traceback.print_exc()
            SM_FAILURES.append({"fold": _f, "seed": SM_SEED, "error": f"{type(_e).__name__}: {_e}"})
            continue
        finally:
            sm_restore()
        write_json(_r, _out)
        SM_RESULTS.append(_clean(_r))
        print(f"[size-matched] fold {_f} | known mAP {_r['known_map']:.4f} | " +
              " | ".join(f"{n}={_r[f'{n}_auroc']:.4f}" for n in ("uhat", "maxlogit", "knn", "ens3")))
else:
    print("[size-matched] RUN_SIZE_MATCHED=False -- skipped")
for _x in SM_FAILURES:
    print(f"[size-matched] FAILED fold {_x['fold']}: {_x['error']}")

SM_CMP = pd.DataFrame()
if SM_RESULTS:
    SMR = pd.DataFrame(SM_RESULTS)
    SMR.to_csv(os.path.join(RESULTS_DIR, "size_matched_results.csv"), index=False)
    _src = RUNS[CTRL_REFERENCE]["src"]
    FU = read_exact(os.path.join(_src, "results", "rfmid_fold_unknown.csv"))
    LR = (read_exact(os.path.join(_src, "results", "lco_results.csv")) if RUNS[CTRL_REFERENCE]["lco_folds"]
          else pd.DataFrame(columns=["fold", "seed"]))
    LR = LR[LR.seed == CTRL_LCO_SEED].set_index("fold")
    dets = ("ens3", "uhat", "maxlogit", "knn")
    rows = []
    for f in sorted(SMR.fold):
        r = {"fold": f, "n_train_size_matched": int(SMR.set_index("fold").at[f, "n_train"]),
             "n_train_lco": int(LR.at[f, "n_train"]) if f in LR.index else np.nan}
        for d in dets:
            r[f"{d}_withheld_seed{SM_SEED}"] = float(FU[(FU.seed == SM_SEED) & (FU.fold == f)][f"{d}_dis_auroc"].mean())
            r[f"{d}_withheld_mean_seeds"] = float(FU[FU.fold == f][f"{d}_dis_auroc"].mean())
            r[f"{d}_lco"] = float(LR.at[f, f"{d}_auroc"]) if f in LR.index else np.nan
            r[f"{d}_size_matched"] = float(SMR.set_index("fold").at[f, f"{d}_auroc"])
        r["knn_size_matched_own_gallery"] = float(SMR.set_index("fold").at[f, "knn_own_auroc"])
        r["ens3_size_matched_own_gallery"] = float(SMR.set_index("fold").at[f, "ens3_own_auroc"])
        rows.append(r)
    SM_CMP = pd.DataFrame(rows)
    _mean = SM_CMP.drop(columns="fold").mean(numeric_only=True).to_dict()
    SM_CMP = pd.concat([SM_CMP, pd.DataFrame([{"fold": "mean", **_mean}])], ignore_index=True)
    _m = SM_CMP[SM_CMP.fold == "mean"].iloc[0]
    _gap = _m[f"ens3_withheld_seed{SM_SEED}"] - _m["ens3_lco"]
    SM_GAP_SHARE = (_m[f"ens3_withheld_seed{SM_SEED}"] - _m["ens3_size_matched"]) / _gap if _gap and np.isfinite(_gap) else np.nan
    SM_CMP.to_csv(os.path.join(RESULTS_DIR, "size_matched_vs_protocols.csv"), index=False)
    print("\n" + "=" * 112)
    print(f"  SIZE-MATCHED TRAINING | {CTRL_REFERENCE} seed {SM_SEED} | unknown vs known disease AUROC, fold classes masked")
    print("=" * 112)
    print(f"  {'fold':>4} {'n train SM/LCO':>15} | " + " | ".join(f"{d + ': withheld  LCO  size-m.':>29}" for d in ("ens3", "knn")))
    for _, r in SM_CMP.iterrows():
        print(f"  {str(r.fold):>4} {r.n_train_size_matched:7.0f}/{r.n_train_lco:<7.0f} | " +
              " | ".join(f"{r[f'{d}_withheld_seed{SM_SEED}']:16.3f} {r[f'{d}_lco']:5.3f} {r[f'{d}_size_matched']:7.3f}"
                         for d in ("ens3", "knn")))
    print(f"  share of the withheld - leave-classes-out ens3 gap reproduced by size matching alone: {SM_GAP_SHARE:.1%}")


# In[ ]:


SM_CASE_SUM, SM_CASE_VS = pd.DataFrame(), pd.DataFrame()
if SM_RESULTS and RUN_CASE_SPLIT:
    _parts = []
    for _r in SM_RESULTS:
        _f = int(_r["fold"])
        d = read_exact(os.path.join(PRED_DIR, f"size_matched_fold{_f}_test_scores_seed{SM_SEED}.csv"))
        d["ID"] = d["ID"].astype(str)
        d["fold"], d["seed"], d["protocol"] = _f, SM_SEED, "size_matched"
        d["t_ens3"], d["op_specificity_saved"], d["ens3_auroc_saved"] = _r["t_ens3"], _r["op_specificity"], _r["ens3_auroc"]
        _parts.append(d)
    _ref = SCORES[CTRL_REFERENCE]
    _three = pd.concat([_ref[(_ref.protocol == "withheld") & (_ref.seed == SM_SEED)].assign(protocol=f"withheld_seed{SM_SEED}"),
                        _ref[_ref.protocol == "lco"], pd.concat(_parts)], ignore_index=True)
    _three = _three[_three.fold.isin([int(r["fold"]) for r in SM_RESULTS])]
    _rows, _cfgs = case_rows(CTRL_REFERENCE, _three)
    SM_CS = pd.DataFrame(_rows)
    assert (SM_CS.all_ens3_auroc - SM_CS.ens3_auroc_saved).abs().max() < 1e-6
    SM_CS.to_csv(os.path.join(RESULTS_DIR, "size_matched_case_split_per_fold.csv"), index=False)
    for c in _cfgs:                                   # protocols compared pairwise: name the configs by protocol
        c["model"] = c["protocol"]
    _b = boot_fold_means(_cfgs)
    SM_CASE_SUM, _ = summarise_cases(SM_CS.assign(model=SM_CS.protocol), _b, reference="size_matched")
    _vs = []
    for sub in CASE_SUBSETS:
        for other in (f"withheld_seed{SM_SEED}", "lco"):
            a, b = _b.get(("size_matched", "size_matched", sub)), _b.get((other, other, sub))
            if a is None or b is None:
                continue
            d = a - b
            _vs.append({"subset": sub, "comparison": f"size_matched - {other}", "size_matched": float(a[0]),
                        other: float(b[0]), "delta": float(d[0]), "ci_lo": _ci(d[1:])[0], "ci_hi": _ci(d[1:])[1],
                        "p_boot": _boot_p(d[1:])})
    SM_CASE_VS = pd.DataFrame(_vs)
    SM_CASE_SUM.to_csv(os.path.join(RESULTS_DIR, "size_matched_case_split_summary.csv"), index=False)
    SM_CASE_VS.to_csv(os.path.join(RESULTS_DIR, "size_matched_vs_protocols_bootstrap.csv"), index=False)
    print(f"[summary] {CTRL_REFERENCE}, folds {sorted(SM_CS.fold.unique())}: ens3 AUROC by protocol and case type")
    print(SM_CASE_SUM[["model", "subset", "folds_used", "ens3_auroc", "ci_lo", "ci_hi", _TPR_TAG, "op_specificity"]]
          .rename(columns={"model": "protocol"}).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for _, v in SM_CASE_VS.iterrows():
        print(f"  {v.subset:13} {v.comparison:32} {v.delta:+.3f} [{v.ci_lo:+.3f}, {v.ci_hi:+.3f}] p {_pfmt(v.p_boot)}")

def _f3(v):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.3f}"

L = [f"# {RUN_NAME}: numbers for the paper", "",
     f"Generated {time.strftime('%Y-%m-%d %H:%M')} | reference {CTRL_REFERENCE} | runs: {', '.join(RUNS)}", ""]
if len(GAL_SUM):
    L += ["## kNN gallery (§17)", ""]
    for name in GAL_SUM.model.unique():
        r = GAL_SUM[(GAL_SUM.model == name) & (GAL_SUM.fold == "mean")].iloc[0]
        L.append(f"* {name}: masked-class kNN AUROC {_f3(r.knn_protocol)} with the protocol gallery (= the leave-classes-out "
                 f"training images, mean n = {r.n_gallery_protocol:.0f}); {_f3(r.knn_full)} with the withheld-class images put "
                 f"back (n = {r.n_gallery_full:.0f}); {_f3(r.knn_full_size_matched)} with them put back at the "
                 f"leave-classes-out size; leave-classes-out {_f3(r.knn_lco)}. ens3: {_f3(r.ens3_protocol)} / "
                 f"{_f3(r.ens3_full)} / {_f3(r.ens3_full_size_matched)} / LCO {_f3(r.ens3_lco)}.")
    L.append("")
if len(CASE_SUM):
    L += ["## Mixed vs unknown-only (§18), ens3 AUROC [95% CI]", ""]
    for proto in _proto_order(CASE_SUM.protocol.unique()):
        L.append(f"**{proto}**")
        for name in CASE_SUM.model.unique():
            g = CASE_SUM[(CASE_SUM.protocol == proto) & (CASE_SUM.model == name)].set_index("subset")
            if g.empty:
                continue
            L.append(f"* {name}: " + "; ".join(f"{s} {_f3(g.at[s, 'ens3_auroc'])} [{_f3(g.at[s, 'ci_lo'])}, "
                                              f"{_f3(g.at[s, 'ci_hi'])}]" for s in CASE_SUBSETS)
                     + f"; FPR at {OP_TPR:.0%} TPR {_f3(g.at['all', _TPR_TAG])}; specificity at t_ens3 "
                     f"{_f3(g.at['all', 'op_specificity'])}")
        L.append("")
    if len(CASE_VS):
        L.append(f"Paired differences {CTRL_REFERENCE} - model on mixed cases:")
        for _, r in CASE_VS[CASE_VS.subset == "mixed"].iterrows():
            L.append(f"* {r.protocol}, {r.model}: {r.delta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}], p {_pfmt(r.p_boot)}")
        L.append("")
if len(DEL_SUM):
    L += ["## Residual-map deletion (§19)", ""]
    for _, r in DEL_SUM[DEL_SUM.subset.isin(["all test images", "unknown-positive"])].iterrows():
        L.append(f"* {r.fill}, ranked by {r.ranking}, {r.subset}: deleting the top {DEL_FRAC:.0%} lowers u_hat by "
                 f"{r.drop_targeted:.4f} ({r.drop_targeted_relative:.1%} of the mean) vs {r.drop_random:.4f} "
                 f"({r.drop_random_relative:.1%}) for a random {DEL_FRAC:.0%}; difference {r.difference:+.4f} "
                 f"[{r.diff_ci_lo:+.4f}, {r.diff_ci_hi:+.4f}], Wilcoxon p = {r.p_wilcoxon:.1e}; targeted > random for "
                 f"{r.share_targeted_larger:.0%} of image-folds.")
    L.append("")
if len(SM_CMP):
    r = SM_CMP[SM_CMP.fold == "mean"].iloc[0]
    L += ["## Size-matched training (§20)", "",
          f"* ens3 AUROC, mean over folds {sorted(int(x['fold']) for x in SM_RESULTS)}: withheld-class (seed {SM_SEED}) "
          f"{_f3(r[f'ens3_withheld_seed{SM_SEED}'])}, size-matched {_f3(r.ens3_size_matched)}, leave-classes-out "
          f"{_f3(r.ens3_lco)}. Size matching reproduces {SM_GAP_SHARE:.0%} of the gap.",
          f"* kNN AUROC: withheld-class {_f3(r[f'knn_withheld_seed{SM_SEED}'])}, size-matched {_f3(r.knn_size_matched)}, "
          f"leave-classes-out {_f3(r.knn_lco)}."]
    for _, v in SM_CASE_VS.iterrows():
        L.append(f"* {v.subset}: {v.comparison} {v.delta:+.3f} [{v.ci_lo:+.3f}, {v.ci_hi:+.3f}], p {_pfmt(v.p_boot)}")
    L.append("")
open(os.path.join(RESULTS_DIR, "CONTROLS_SUMMARY.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))


# In[ ]:


write_json({**{n: globals()[n] for n in CONTROL_SETTINGS},
            "runs": {n: {"folder": os.path.abspath(i["src"]), "run_name": i["run_name"], "ens3": i["ens3"],
                         "withheld_seeds": i["seeds"], "feature_seeds": i["features"], "lco_folds": i["lco_folds"]}
                     for n, i in RUNS.items()},
            "case_folds_used": {s: [f for f, ok in CASE_ELIGIBLE[s].items() if ok] for s in CASE_SUBSETS}
            if RUN_CASE_SPLIT else None},
           os.path.join(REPRO_DIR, "control_settings.json"))
pd.DataFrame([{"path": p, "bytes": os.path.getsize(p), "sha256": sha256_file(p)} for p in sorted(set(INPUT_FILES))]) \
  .to_csv(os.path.join(REPRO_DIR, "input_manifest.csv"), index=False)

_sel = []
for p in sorted(glob.glob(os.path.join(SM_ROOT, "**", "selection_*.json"), recursive=True)):
    j = json.load(open(p))
    _sel.append({"arm": j.get("arm"), "seed": j.get("seed"), "selected_epoch": j.get("selected_epoch"),
                 "phase_a_baseline": j.get("phase_a_baseline"), "phase_b_floor": j.get("phase_b_floor"),
                 "selected_ckpt": os.path.relpath(j.get("selected_ckpt", ""), OUTPUT_ROOT),
                 "selected_ckpt_sha256": j.get("selected_ckpt_sha256"), "record": os.path.relpath(p, OUTPUT_ROOT)})
pd.DataFrame(_sel).to_csv(os.path.join(REPRO_DIR, "checkpoint_selection.csv"), index=False)

ENV_MANIFEST["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
ENV_MANIFEST["session_hours"] = round((time.time() - SESSION_T0) / 3600, 2)
write_json(ENV_MANIFEST, os.path.join(REPRO_DIR, "environment.json"))

README = f"""# {RUN_NAME}: reproduction record

Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')} on {ENV_MANIFEST['gpu']} (python {ENV_MANIFEST['python']},
torch {ENV_MANIFEST['packages'].get('torch')}).

## Inputs
* The finished training runs listed in `control_settings.json -> runs`; every file read from them is listed with its
  SHA-256 in `input_manifest.csv`.
* RFMiD and the DINOv2 weights as in `config.json -> paths` (hashes in `data_provenance.json`).

## Reproduce
1. Install the package versions in `requirements_freeze.txt`.
2. Point the paths in §0 at the data and set `CTRL_MODELS` to the same finished runs (check `input_manifest.csv`).
3. Run the notebook top to bottom with the settings in `control_settings.json`. Random choices are seeded:
   gallery draws CTRL_GALLERY_SEED={CTRL_GALLERY_SEED} + training seed; bootstrap CTRL_BOOT_SEED={CTRL_BOOT_SEED};
   random deletions DEL_SEED={DEL_SEED} + seed; size-matched removals SM_REMOVAL_SEED={SM_REMOVAL_SEED} + fold
   (must reproduce `size_matched_removed_images.csv`); training seed SM_SEED={SM_SEED}.
4. Size-matched trainings resume: a fold with `selected.pt` and `train_meta.json` under `SizeMatched/fold<f>/` is only
   re-evaluated (checkpoints listed with SHA-256 in `checkpoint_selection.csv`).

## Notes
* Mixed precision and non-deterministic GPU kernels: a re-run reproduces the numbers closely, not bit-for-bit.
* Nothing is fitted on test images: thresholds come from the RFMiD validation calibration half.
* Numbers for the paper: `results/CONTROLS_SUMMARY.md`.
"""
open(os.path.join(REPRO_DIR, "README.md"), "w").write(README)

_man_path = os.path.join(REPRO_DIR, "output_manifest.csv")
_man = []
for p in sorted(Path(OUTPUT_ROOT).rglob("*")):
    if p.is_file() and str(p) != _man_path and not p.name.endswith(".tmp"):
        _man.append({"path": str(p.relative_to(OUTPUT_ROOT)), "bytes": p.stat().st_size, "sha256": sha256_file(str(p))})
pd.DataFrame(_man).to_csv(_man_path, index=False)
print(f"[repro] {len(_man)} files hashed -> reproducibility/output_manifest.csv")
for f in sorted(os.listdir(REPRO_DIR)):
    print(f"    {f:36} {os.path.getsize(os.path.join(REPRO_DIR, f)) / 1024:8.1f} KB")
print(f"\n[done] {(time.time() - SESSION_T0) / 3600:.2f} h this session")

