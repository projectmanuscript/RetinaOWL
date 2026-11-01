#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys

# ---- paths -----------------------------------------------------------------------------------
OUTPUT_ROOT   = "./Output/Corruption"                              # overridden by the robustness block at the end of §0
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
SEEDS_THIS_RUN = [42, 43, 44, 45, 46]   # split across two notebooks/GPUs if wanted; outputs never collide
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
RUN_NAME = "RetinaOWL_robustness"

# ==== ROBUSTNESS (inference only): every setting this notebook adds or overrides =================================
# Applied last, so it overrides the training values above. Keep this block when you paste in the paths above.
OUTPUT_ROOT = "./Output/Corruption"                     # where the robustness results go (new folder)
ABLATION = "robustness: inference of finished models"
ROB_MODELS = {                                          # each model's finished training output folder (read only)
    "RetinaOWL":                  {"source": "../Model/Output",                         "suites": ["corruption"]},
    "A9_no_augmentation":         {"source": "../Ablations/No_Augmentation/Output",     "suites": ["corruption"]},
}
ROB_REFERENCE = "RetinaOWL"                             # the ablations are compared with this model
ROB_SEEDS = [42, 43, 44, 45, 46]                        # training seeds (original checkpoints only, no leave-classes-out)
ROB_DATASETS = ["RFMiD", "JSIEC", "ODIR"]
ROB_CORRUPTIONS = {                                     # 3 severities each; applied to the retina only (border stays black)
    "gaussian_noise": [0.04, 0.08, 0.12],               # sigma, intensities in [0, 1]
    "gaussian_blur": [1.5, 3.0, 5.0],                   # sigma in pixels at 518 px (defocus)
    "jpeg": [30, 15, 8],                                # JPEG quality
    "underexposure": [0.70, 0.50, 0.35],                # intensity factor
    "haze": [0.15, 0.30, 0.45],                         # white veil weight (media opacity / cataract-like)
    "uneven_illumination": [0.70, 0.50, 0.30],          # gain at the dark side of a linear illumination ramp
}
ROB_ANGLES = [15, 30, 45, 90, 180]                      # degrees, counter-clockwise about the image centre
LANDMARK_DIR = "../Ablations/Anatomy_Grounding/Preprocessing_Cache/Landmarks"          # A10 / A11: the landmark CSVs those runs wrote
RFMID_ROT_CACHE = "../Ablations/Image_Rotation/Preprocessing_Cache/RFMiD"   # A11: its rotation-normalised crops (<split>/images)
JSIEC_ROT_CACHE = "../Ablations/Image_Rotation/Preprocessing_Cache/JSIEC"
ODIR_ROT_CACHE = "../Ablations/Image_Rotation/Preprocessing_Cache/ODIR"
ROB_LANDMARK_ROT_DIR = "../Ablations/Image_Rotation/Preprocessing_Cache/Landmarks"   # landmarks detected on rotated images (cache)
ROB_WORKERS, ROB_BATCH = 4, 32                          # data-loading processes; images per batch
ROB_CHECK_TOL = 0.01                                    # max |clean recomputed - original| before the run stops
RUN_EXTERNAL, RUN_LCO = True, False                     # external caches are needed; nothing is retrained
ROT_FILL = 0                                            # A11 alignment: fill for exposed corners (as in A11)
ABLATION_SETTINGS = ['RUN_NAME', 'OUTPUT_ROOT', 'ABLATION', 'ROB_MODELS', 'ROB_REFERENCE', 'ROB_SEEDS', 'ROB_DATASETS', 'ROB_CORRUPTIONS', 'ROB_ANGLES', 'LANDMARK_DIR', 'RFMID_ROT_CACHE', 'JSIEC_ROT_CACHE', 'ODIR_ROT_CACHE', 'ROB_LANDMARK_ROT_DIR', 'ROB_WORKERS', 'ROB_BATCH', 'ROB_CHECK_TOL', 'RUN_EXTERNAL', 'RUN_LCO', 'ROT_FILL']
# ==================================================================================================================
print(f"[cfg] {RUN_NAME} | seeds {SEEDS_THIS_RUN} | Phase A {NUM_EPOCHS_PHASE_A} + Phase B {NUM_EPOCHS_PHASE_B} epochs "
      f"| external={RUN_EXTERNAL} | leave-classes-out={RUN_LCO} folds {LCO_FOLDS} seeds {LCO_SEEDS} | out {OUTPUT_ROOT}")


# In[ ]:


import json as _json
_cfg_here = os.path.join(OUTPUT_ROOT, "reproducibility", "config.json")
if os.path.exists(_cfg_here) and _json.load(open(_cfg_here)).get("RUN_NAME") != RUN_NAME:
    raise RuntimeError(f"OUTPUT_ROOT={OUTPUT_ROOT!r} already holds the outputs of "
                       f"{_json.load(open(_cfg_here)).get('RUN_NAME')!r}; choose another OUTPUT_ROOT.")
_missing = [n for n in ['RUN_NAME', 'OUTPUT_ROOT', 'ABLATION', 'ROB_MODELS', 'ROB_REFERENCE', 'ROB_SEEDS', 'ROB_DATASETS', 'ROB_CORRUPTIONS', 'ROB_ANGLES', 'LANDMARK_DIR', 'RFMID_ROT_CACHE', 'JSIEC_ROT_CACHE', 'ODIR_ROT_CACHE', 'ROB_LANDMARK_ROT_DIR', 'ROB_WORKERS', 'ROB_BATCH', 'ROB_CHECK_TOL', 'RUN_EXTERNAL', 'RUN_LCO', 'ROT_FILL'] if n not in globals()]
if _missing:
    raise NameError(f"§0 lacks {_missing}: the robustness block at the end of §0 was removed or not run.")
import glob, json, math, random, re, shutil, time, socket, platform, hashlib, unicodedata
import importlib.metadata as _im
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
import matplotlib.pyplot as plt

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

# ---- robustness: the finished models (original checkpoints of every training seed) -------------------------------
MODEL_INFO = {}
for _mn, _spec in ROB_MODELS.items():
    _src = _spec["source"]
    _cp = os.path.join(_src, "reproducibility", "config.json")
    if not os.path.exists(_cp):
        _err(f"{_mn}: no reproducibility/config.json in {os.path.abspath(_src)} (point ROB_MODELS at its output folder)")
        continue
    _c = json.load(open(_cp))
    _kind = "anatomy" if _c.get("USE_ANATOMY_GROUNDING") else ("rotnorm" if _c.get("USE_ROTATION_NORM") else "plain")
    _uns = {k: v for k, v in (("USE_RESIDUAL_BRANCH", False), ("USE_LORA", False), ("HEAD_TYPE", "linear"),
                              ("USE_FOV_MASK", False)) if _c.get(k, None) == v}
    if _uns:
        _err(f"{_mn}: this notebook scores RetinaOWL-architecture models only (source has {_uns})")
    _ck = {s: os.path.join(_src, "Model_Checkpoints", f"SEED = {s}", "phaseB", "selected.pt") for s in ROB_SEEDS}
    _js = {s: os.path.join(_src, "json", f"results_seed{s}.json") for s in ROB_SEEDS}
    _ms = [p for s in ROB_SEEDS for p in (_ck[s], _js[s]) if not os.path.exists(p)]
    if _ms:
        _err(f"{_mn}: missing {len(_ms)} checkpoint/result files, e.g. {_ms[0]}")
    if not set(_spec["suites"]) <= {"corruption", "rotation"}:
        _err(f"{_mn}: suites must be 'corruption' and/or 'rotation'")
    MODEL_INFO[_mn] = {"source": _src, "cfg": _c, "kind": _kind, "run_name": _c.get("RUN_NAME"), "ckpt": _ck,
                       "json": _js, "suites": list(_spec["suites"])}
    _ok(f"{_mn}: {_c.get('RUN_NAME')} ({_kind}) | {len(ROB_SEEDS)} original checkpoints in {os.path.abspath(_src)}")
if ROB_REFERENCE not in ROB_MODELS:
    _err(f"ROB_REFERENCE={ROB_REFERENCE!r} is not in ROB_MODELS")
NEED_LANDMARKS = any(i["kind"] == "anatomy" or (i["kind"] == "rotnorm" and "rotation" in i["suites"])
                     for i in MODEL_INFO.values())
NEED_ROTATED = any(i["kind"] == "rotnorm" for i in MODEL_INFO.values())
if NEED_LANDMARKS:
    (_ok if os.path.isdir(LANDMARK_DIR) else _warn)(f"landmarks: {LANDMARK_DIR}"
                                                    + ("" if os.path.isdir(LANDMARK_DIR) else " missing -- detected now"))
if NEED_ROTATED:
    for _d in [os.path.join(RFMID_ROT_CACHE, sp, "images") for sp in ("train", "val", "test")] + \
              [os.path.join(JSIEC_ROT_CACHE, "images"), os.path.join(ODIR_ROT_CACHE, "images")]:
        (_ok if os.path.isdir(_d) else _warn)(f"A11 rotation-normalised crops: {_d}"
                                              + ("" if os.path.isdir(_d) else " missing -- built now"))

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


LM_COLS = ["fovea_x", "fovea_y", "od_x", "od_y"]
LM_MIN_DIST, LM_MAX_DIST = 0.08 * IMG_SIZE, 0.60 * IMG_SIZE
LANDMARKS, EXT_LANDMARKS, LANDMARK_SUMMARY = {}, {}, []
_LM_MODEL = None

def get_landmark_model():
    global _LM_MODEL
    if _LM_MODEL is None:
        try:
            _LM_MODEL, _ = get_fit().load_fovea_od_model("default", device=str(DEVICE), return_test_dataloader=False)
        except Exception as e:
            raise RuntimeError(f"the fovea/optic-disc localiser could not be loaded ({type(e).__name__}: {e}). It downloads "
                               f"an EfficientNet-B3 on first use (internet needed once); afterwards the CSVs in "
                               f"{LANDMARK_DIR} are reused") from e
    return _LM_MODEL

def landmark_ok(p):
    p = np.asarray(p, dtype=float)
    if p.shape != (4,) or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > IMG_SIZE):
        return False
    return LM_MIN_DIST <= math.hypot(p[0] - p[2], p[1] - p[3]) <= LM_MAX_DIST

def _write_atomic(df, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def ensure_landmarks(tag, img_dir, ids):
    """Fovea/OD landmarks for `ids` (cached PNGs in img_dir); returns a DataFrame indexed by ID with LM_COLS."""
    os.makedirs(LANDMARK_DIR, exist_ok=True)
    path, ids = os.path.join(LANDMARK_DIR, f"{tag}.csv"), [str(i) for i in ids]
    rows = (pd.read_csv(path, dtype={"ID": str}, keep_default_na=False, na_values=[""]).to_dict("records")
            if os.path.exists(path) else [])
    have = {r["ID"] for r in rows}
    todo = [i for i in ids if i not in have]
    if todo:
        m, t0 = get_landmark_model(), time.time()
        print(f"[landmarks {tag}] detecting fovea + optic disc on {len(todo)} cached crops ...")
        for n, i in enumerate(todo):
            err = ""
            try:
                p = np.asarray(m.predict(Image.open(os.path.join(img_dir, f"{i}.png")).convert("RGB")), dtype=float).ravel()
                p = p[:4] if p.size >= 4 else np.full(4, np.nan)
            except Exception as e:
                p, err = np.full(4, np.nan), f"{type(e).__name__}: {str(e)[:120]}"
            rows.append({"ID": i, **{f"raw_{c}": float(v) for c, v in zip(LM_COLS, p)},
                         "detected_ok": int(landmark_ok(p)), "error": err})
            if (n + 1) % 500 == 0 or n + 1 == len(todo):
                _write_atomic(pd.DataFrame(rows), path)
                print(f"[landmarks {tag}] {n + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows).drop_duplicates("ID", keep="last").set_index("ID")
    ok = df["detected_ok"].astype(int) == 1
    if not ok.any():
        raise RuntimeError(f"[landmarks {tag}] no plausible detection at all -- check the localiser")
    med = df.loc[ok, [f"raw_{c}" for c in LM_COLS]].median()
    for c in LM_COLS:
        df[c] = np.where(ok, df[f"raw_{c}"], med[f"raw_{c}"])
    out = df.loc[ids, LM_COLS + ["detected_ok"]].copy()
    d = np.hypot(out.fovea_x - out.od_x, out.fovea_y - out.od_y)
    rec = {"set": tag, "images": len(out), "rejected_to_median": int((out.detected_ok == 0).sum()),
           "median_disc_fovea_px": float(np.median(d)), "od_left_of_fovea": float((out.od_x < out.fovea_x).mean()),
           "file": os.path.abspath(path), "file_sha256": sha256_file(path)}
    LANDMARK_SUMMARY.append(rec)
    pd.DataFrame(LANDMARK_SUMMARY).to_csv(os.path.join(REPRO_DIR, "landmarks_summary.csv"), index=False)
    print(f"[landmarks {tag}] {rec['images']} images | {rec['rejected_to_median']} rejected -> median position | "
          f"median disc-fovea distance {rec['median_disc_fovea_px']:.0f} px | disc left of fovea in "
          f"{rec['od_left_of_fovea']:.0%}")
    return out

if NEED_LANDMARKS:                               # A10 / A11 only
    for _sp, _d in (("train", train_df), ("val", val_df), ("test", test_df)):
        LANDMARKS[_sp] = ensure_landmarks(f"rfmid_{_sp}", os.path.join(RFMID_CACHE, _sp, "images"), _d["ID"])
else:
    print("[landmarks] not needed by the selected models")


# In[ ]:


ROTATION_SUMMARY = []

def rotate_align(pil_img, fovea, od, fill=ROT_FILL):
    """Rotate about the centre so the fovea -> disc axis is horizontal, fovea left, disc right.
    Returns (image, fovea', od', degrees)."""
    W, H = pil_img.size
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    phi = -np.arctan2(od[1] - fovea[1], od[0] - fovea[0])
    Rz = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    fwd = lambda R, p: tuple(R @ np.array([p[0] - cx, p[1] - cy]) + np.array([cx, cy]))
    R = Rz(phi); f2, o2 = fwd(R, fovea), fwd(R, od)
    if o2[0] < f2[0]:
        phi += np.pi; R = Rz(phi); f2, o2 = fwd(R, fovea), fwd(R, od)
    a_, b_ = Rz(-phi)[0]; d_, e_ = Rz(-phi)[1]          # PIL's affine maps output -> input
    out = pil_img.transform((W, H), Image.AFFINE, (a_, b_, cx - (a_ * cx + b_ * cy), d_, e_, cy - (d_ * cx + e_ * cy)),
                            resample=Image.BICUBIC, fillcolor=(fill, fill, fill))
    return out, f2, o2, float(np.degrees(phi))

for _ang in (0.0, 37.0, 90.0, 180.0, -120.0):             # geometry check over orientations
    _f = np.array([259.0, 259.0]); _r = np.radians(_ang)
    _o = _f + 130 * np.array([np.cos(_r), np.sin(_r)])
    _, _f2, _o2, _ = rotate_align(Image.new("RGB", (IMG_SIZE, IMG_SIZE)), tuple(_f), tuple(_o))
    assert _o2[0] > _f2[0] and abs(_o2[1] - _f2[1]) < 1e-6, f"rotation geometry failed at {_ang} deg"

def ensure_rotated(tag, src_dir, dst_dir, lm, ids):
    os.makedirs(dst_dir, exist_ok=True)
    csv_path = os.path.join(os.path.dirname(dst_dir), "rotation.csv")
    ids = [str(i) for i in ids]
    old = pd.read_csv(csv_path, dtype={"ID": str}).set_index("ID") if os.path.exists(csv_path) else pd.DataFrame()
    rows = {i: r for i, r in old.to_dict("index").items()} if len(old) else {}
    def stale(i):
        r = rows.get(i)
        if r is None or not os.path.exists(os.path.join(dst_dir, f"{i}.png")):
            return True
        return not np.allclose([r[f"src_{c}"] for c in LM_COLS], lm.loc[i, LM_COLS].values.astype(float), atol=1e-3)
    todo = [i for i in ids if stale(i)]
    if todo:
        t0 = time.time()
        print(f"[rotate {tag}] writing {len(todo)}/{len(ids)} rotation-normalised crops ...")
        for n, i in enumerate(todo):
            img = Image.open(os.path.join(src_dir, f"{i}.png")).convert("RGB")
            p = lm.loc[i, LM_COLS].values.astype(float)
            ok = int(lm.loc[i, "detected_ok"]) == 1
            if ok:
                img, f2, o2, deg = rotate_align(img, (p[0], p[1]), (p[2], p[3]))
            else:
                f2, o2, deg = (p[0], p[1]), (p[2], p[3]), 0.0
            tmp = os.path.join(dst_dir, f"{i}.tmp.png")
            img.save(tmp); os.replace(tmp, os.path.join(dst_dir, f"{i}.png"))
            rows[i] = {**{f"src_{c}": float(v) for c, v in zip(LM_COLS, p)}, "angle_deg": deg, "rot_ok": int(ok),
                       "fovea_x": f2[0], "fovea_y": f2[1], "od_x": o2[0], "od_y": o2[1]}
            if (n + 1) % 500 == 0 or n + 1 == len(todo):
                _write_atomic(pd.DataFrame.from_dict(rows, orient="index").rename_axis("ID").reset_index(), csv_path)
                print(f"[rotate {tag}] {n + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
    else:
        print(f"[rotate {tag}] complete ({len(ids)} crops) -> reused")
    return pd.DataFrame.from_dict(rows, orient="index").loc[ids]

if NEED_ROTATED:
    for _sp, _d in (("train", train_df), ("val", val_df), ("test", test_df)):
        ensure_rotated(f"rfmid_{_sp}", os.path.join(RFMID_CACHE, _sp, "images"),
                       os.path.join(RFMID_ROT_CACHE, _sp, "images"), LANDMARKS[_sp], _d["ID"])


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


from scipy import ndimage as _ndi
YCOLS = [f"y_{c}" for c in DISEASE_COLS]
DECISIONS = {"KNOWN", "UNKNOWN", "NORMAL", "OMIT"}
_KW_SPLIT = re.compile("[,，]")            # ASCII comma and the full-width comma
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def norm_term(s):
    s = unicodedata.normalize("NFKC", str(s)).replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", s).strip().casefold()

def split_terms(dataset, raw):
    """JSIEC: the folder name is the single term. ODIR: split the keyword cell on both commas."""
    raw = "" if raw is None or (isinstance(raw, float) and math.isnan(raw)) else str(raw)
    if dataset == "JSIEC":
        return [raw.strip()] if raw.strip() else []
    out, seen = [], set()
    for part in _KW_SPLIT.split(raw):
        p = part.strip()
        if norm_term(p) and norm_term(p) not in seen:
            seen.add(norm_term(p)); out.append(p)
    return out

def load_mapping(path, key_col):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    need = {key_col, "decision", "rfmid_class", "debatable_flag", "rationale"}
    assert need <= set(df.columns), f"{path}: columns {sorted(need - set(df.columns))} missing"
    problems, mapping = [], {}
    for i, r in df.iterrows():
        key, dec, cls, deb = r[key_col].strip(), r["decision"].strip().upper(), r["rfmid_class"].strip(), r["debatable_flag"].strip()
        if not key:
            problems.append(f"row {i+2}: empty {key_col}"); continue
        if dec not in DECISIONS:
            problems.append(f"row {i+2} '{key}': decision '{dec}' not in {sorted(DECISIONS)}")
        if dec == "KNOWN" and cls not in DISEASE_COLS:
            problems.append(f"row {i+2} '{key}': KNOWN but rfmid_class '{cls}' is not an RFMiD class")
        if dec != "KNOWN" and cls:
            problems.append(f"row {i+2} '{key}': {dec} rows must not carry an rfmid_class ('{cls}')")
        if deb not in ("", "0", "1"):
            problems.append(f"row {i+2} '{key}': debatable_flag '{deb}' is not empty/0/1")
        k = norm_term(key)
        if k in mapping:
            problems.append(f"row {i+2} '{key}': duplicates '{mapping[k]['term']}' after normalisation")
        mapping[k] = {"term": key, "decision": dec, "rfmid_class": cls if dec == "KNOWN" else "",
                      "debatable": deb == "1", "rationale": r["rationale"]}
    if problems:
        raise ValueError(f"{path} has {len(problems)} problem(s):\n  " + "\n  ".join(problems))
    return df, mapping

def _sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def parse_jsiec(root):
    rows = []
    for folder in sorted(os.listdir(root)):
        fdir = os.path.join(root, folder)
        if not os.path.isdir(fdir):
            continue
        for fn in sorted(os.listdir(fdir)):
            if os.path.isfile(os.path.join(fdir, fn)) and os.path.splitext(fn)[1].lower() in IMG_EXTS:
                rel = os.path.join(folder, fn)
                rows.append({"image_id": "jsiec_" + _sha1(rel.replace(os.sep, "/"))[:16], "dataset": "JSIEC",
                             "split": "all", "patient_id": None, "eye": "", "source_relpath": rel,
                             "keywords_raw": folder})
    df = pd.DataFrame(rows)
    df["patient_id"] = df["image_id"]                  # no patient information in JSIEC
    assert df["image_id"].is_unique
    print(f"[jsiec] {len(df)} images in {df['keywords_raw'].nunique()} folders")
    return df

def parse_odir(root):
    need = ["ID", "Left-Fundus", "Right-Fundus", "Left-Diagnostic Keywords", "Right-Diagnostic Keywords"]
    rows = []
    for folder, (tag, ann) in ODIR_SPLITS.items():
        x = pd.read_excel(os.path.join(root, folder, ann), dtype=str, keep_default_na=False)
        x.columns = [str(c).strip() for c in x.columns]
        assert not [c for c in need if c not in x.columns], f"ODIR {tag}: workbook lacks {need}"
        x = x[x["ID"].astype(str).str.strip() != ""]
        on_disk = {f.lower(): f for f in os.listdir(os.path.join(root, folder, ODIR_IMAGE_DIR))}
        for _, r in x.iterrows():
            pid = str(r["ID"]).strip()
            for eye in ("Left", "Right"):
                fn = str(r[f"{eye}-Fundus"]).strip()
                rows.append({"image_id": f"odir_{tag}_{os.path.splitext(fn)[0]}", "dataset": "ODIR", "split": tag,
                             "patient_id": f"{tag}_{pid}", "eye": eye.lower(),
                             "source_relpath": os.path.join(folder, ODIR_IMAGE_DIR, on_disk.get(fn.lower(), fn)),
                             "keywords_raw": r[f"{eye}-Diagnostic Keywords"]})
    df = pd.DataFrame(rows)
    assert df["image_id"].is_unique, "ODIR: duplicated eye ids"
    print(f"[odir] {len(df)} eyes from {df['patient_id'].nunique()} patients | "
          + " | ".join(f"{t}={int((df['split'] == t).sum())}" for t in df["split"].unique()))
    return df

def build_labels(dataset, ann, mapping):
    """(labels, exclusions, terms_long, unmapped): a pure function of the annotations and the mapping."""
    lab_rows, exc_rows, term_rows, unmapped = [], [], [], []
    for r in ann.itertuples(index=False):
        terms = split_terms(dataset, r.keywords_raw)
        recs = []
        for t in terms:
            m = mapping.get(norm_term(t))
            if m is None:
                unmapped.append({"image_id": r.image_id, "dataset": dataset, "split": r.split, "patient_id": r.patient_id,
                                 "eye": r.eye, "source_relpath": r.source_relpath, "keywords_raw": r.keywords_raw,
                                 "unmapped_term": t, "unmapped_term_normalised": norm_term(t)})
                continue
            recs.append(m)
            term_rows.append({"image_id": r.image_id, "term_raw": t, "term_mapped": m["term"], "decision": m["decision"],
                              "rfmid_class": m["rfmid_class"], "debatable": int(m["debatable"])})
        kept = [m for m in recs if m["decision"] != "OMIT"]
        omitted = [m["term"] for m in recs if m["decision"] == "OMIT"]
        known = sorted({m["rfmid_class"] for m in kept if m["decision"] == "KNOWN"})
        unk = sorted({m["term"] for m in kept if m["decision"] == "UNKNOWN"})
        normal = any(m["decision"] == "NORMAL" for m in kept)
        base = {"image_id": r.image_id, "dataset": dataset, "split": r.split, "patient_id": r.patient_id,
                "eye": r.eye, "source_relpath": r.source_relpath, "keywords_raw": r.keywords_raw}
        if not terms:
            exc_rows.append({**base, "reason": "no_terms"}); continue
        if not kept:
            exc_rows.append({**base, "reason": "omit_class" if dataset == "JSIEC" else "all_terms_omitted",
                             "detail": "; ".join(omitted)}); continue
        if normal and (known or unk):
            exc_rows.append({**base, "reason": "normal_with_disease_terms", "detail": "; ".join(m["term"] for m in kept)})
            continue
        row = {**base, "terms_retained": "; ".join(m["term"] for m in kept), "known_classes": ";".join(known),
               "unknown_terms": ";".join(unk), "omit_terms": "; ".join(omitted), "n_omit_terms": len(omitted),
               "debatable_terms": "; ".join(m["term"] for m in kept if m["debatable"]),
               "has_debatable": int(any(m["debatable"] for m in kept)), "is_normal": int(normal),
               "has_known": int(bool(known)), "is_unknown": int(bool(unk)), "is_mixed": int(bool(known) and bool(unk)),
               "is_diseased": int(bool(known) or bool(unk))}
        row.update({f"y_{c}": int(c in known) for c in DISEASE_COLS})
        lab_rows.append(row)
    return pd.DataFrame(lab_rows), pd.DataFrame(exc_rows), pd.DataFrame(term_rows), pd.DataFrame(unmapped)

# ---- crop + quality indicators -----------------------------------------------------------------
QC_COLS = ["blur_lapvar", "illum_mean", "illum_std", "illum_cv", "fov_frac", "fov_completeness"]
MAN_COLS = ["image_id", "source_relpath", "source_sha256_16", "orig_w", "orig_h", "ok", "error"]

def quality_indicators(img_u8, grid=8):
    m = img_u8.max(axis=2) > FOV_THRESH
    gray = img_u8.mean(axis=2)
    if m.sum() < 100:
        return {"blur_lapvar": np.nan, "illum_mean": np.nan, "illum_std": np.nan, "illum_cv": np.nan,
                "fov_frac": float(m.mean()), "fov_completeness": 0.0}
    mu = float(gray[m].mean())
    bh, bw = gray.shape[0] // grid, gray.shape[1] // grid
    bm = [float(gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw][m[i*bh:(i+1)*bh, j*bw:(j+1)*bw]].mean())
          for i in range(grid) for j in range(grid) if m[i*bh:(i+1)*bh, j*bw:(j+1)*bw].mean() > 0.5] or [mu]
    return {"blur_lapvar": float(_ndi.laplace(gray)[m].var()) / (mu ** 2 + 1e-6), "illum_mean": mu,
            "illum_std": float(gray[m].std()), "illum_cv": float(np.std(bm) / (np.mean(bm) + 1e-6)),
            "fov_frac": float(m.mean()), "fov_completeness": float(min(m.mean() / (math.pi / 4.0), 1.0))}

def _source_info(src):
    out = {"source_sha256_16": "", "orig_w": np.nan, "orig_h": np.nan}
    if src and os.path.exists(src):
        out["source_sha256_16"] = sha256_file(src, n=16)
        with Image.open(src) as im:
            out["orig_w"], out["orig_h"] = im.size
    return out

def _preprocess_one(image_id, src, dst):
    """Crop one original into the cache and compute its quality indicators. Errors are returned, not raised."""
    out = {"image_id": image_id, "ok": 0, "error": ""}
    try:
        if not os.path.exists(src):
            raise FileNotFoundError(f"source image not found: {src}")
        out.update(_source_info(src))
        crop_to_png(src, dst)
        out.update(quality_indicators(np.array(Image.open(dst).convert("RGB"))))
        out["ok"] = 1
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return out

def _qc_one(image_id, src, dst):
    out = {"image_id": image_id, "ok": 1, "error": ""}
    try:
        out.update(_source_info(src))
        arr = np.array(Image.open(dst).convert("RGB"))
        assert arr.shape[:2] == (IMG_SIZE, IMG_SIZE), f"cached PNG is {arr.shape[:2]}"
        out.update(quality_indicators(arr))
    except Exception as e:
        out.update({"ok": 0, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    return out

DS_SPECS = {"JSIEC": {"root": JSIEC_ROOT, "cache": JSIEC_CACHE, "map_csv": JSIEC_MAP_CSV, "key_col": "jsiec_class",
                      "parser": parse_jsiec},
            "ODIR":  {"root": ODIR_ROOT, "cache": ODIR_CACHE, "map_csv": ODIR_MAP_CSV, "key_col": "odir_keyword",
                      "parser": parse_odir}}

def build_cache(ds):
    spec = DS_SPECS[ds]
    root, cache = spec["root"], spec["cache"]
    img_dir = os.path.join(cache, "images")
    os.makedirs(img_dir, exist_ok=True)
    t0 = time.time()
    mapping_df, mapping = load_mapping(spec["map_csv"], spec["key_col"])
    have_src = os.path.isdir(root)
    ann = (spec["parser"](root) if have_src else
           pd.read_csv(os.path.join(cache, "annotations_original.csv"), dtype=str, keep_default_na=False))
    labels, excl, terms_long, unmapped = build_labels(ds, ann, mapping)
    if len(unmapped):
        p = os.path.join(cache, "UNMAPPED_TERMS.csv")
        unmapped.to_csv(p, index=False)
        for t, n in unmapped.groupby("unmapped_term").size().sort_values(ascending=False).items():
            print(f"    {n:5d} x  {t!r}")
        raise RuntimeError(f"{ds}: {unmapped['unmapped_term_normalised'].nunique()} term(s) are not in "
                           f"{spec['map_csv']} ({unmapped['image_id'].nunique()} images, listed in {p}); "
                           f"add them to the mapping and re-run")
    if os.path.exists(os.path.join(cache, "UNMAPPED_TERMS.csv")):
        os.remove(os.path.join(cache, "UNMAPPED_TERMS.csv"))

    man_path, qc_path = os.path.join(cache, "manifest.csv"), os.path.join(cache, "qc.csv")
    man = pd.read_csv(man_path, dtype={"image_id": str}).set_index("image_id").to_dict("index") if os.path.exists(man_path) else {}
    qc_old = pd.read_csv(qc_path, dtype={"image_id": str}).set_index("image_id").to_dict("index") if os.path.exists(qc_path) else {}
    rel = dict(zip(ann["image_id"], ann["source_relpath"]))
    png = lambda i: os.path.join(img_dir, f"{i}.png")
    src = lambda i: os.path.join(root, rel[i]) if have_src else ""
    ev_ids = labels["image_id"].tolist()
    crop_ids = [i for i in ev_ids if not os.path.exists(png(i))]
    qc_ids = [i for i in ev_ids if os.path.exists(png(i)) and (i not in qc_old or i not in man)]
    if not have_src:      # an image that failed in an earlier build stays excluded with its recorded error
        crop_ids = [i for i in crop_ids if not (i in man and str(man[i].get("ok", "")) in ("0", "0.0", "False"))]
        if crop_ids:
            raise RuntimeError(f"{ds}: {len(crop_ids)} image(s) are not cached and the original is not mounted at {root}")
    print(f"[cache {ds}] {len(ev_ids)} evaluable | {len(ev_ids) - len(crop_ids)} cached | {len(crop_ids)} to crop | "
          f"{len(qc_ids)} need quality indicators | original {'mounted' if have_src else 'not mounted'}")

    def _merge(r):
        man[r["image_id"]] = {"source_relpath": rel.get(r["image_id"], ""),
                              **{k: r.get(k, man.get(r["image_id"], {}).get(k, "")) for k in
                                 ("source_sha256_16", "orig_w", "orig_h")}, "ok": r["ok"], "error": r["error"]}
        if r["ok"]:
            qc_old[r["image_id"]] = {k: r[k] for k in QC_COLS}
    def _flush():
        pd.DataFrame([{"image_id": k, **v} for k, v in man.items()]).reindex(columns=MAN_COLS).to_csv(man_path, index=False)
        pd.DataFrame([{"image_id": k, **v} for k, v in qc_old.items()]).to_csv(qc_path, index=False)
    for n, i in enumerate(crop_ids):
        _merge(_preprocess_one(i, src(i), png(i)))
        if (n + 1) % 200 == 0 or n + 1 == len(crop_ids):
            print(f"[cache {ds}] cropped {n+1}/{len(crop_ids)} ({time.time()-t0:.0f}s)"); _flush()
    for i in qc_ids:
        _merge(_qc_one(i, src(i), png(i)))

    bad = [i for i in ev_ids if not os.path.exists(png(i)) or i not in qc_old]
    if bad:
        b = labels[labels["image_id"].isin(bad)][["image_id", "dataset", "split", "patient_id", "eye",
                                                  "source_relpath", "keywords_raw"]].copy()
        b["reason"], b["detail"] = "preprocessing_failed", [str(man.get(i, {}).get("error", "")) for i in b["image_id"]]
        excl = pd.concat([excl, b], ignore_index=True)
        labels = labels[~labels["image_id"].isin(bad)].reset_index(drop=True)
        print(f"[cache {ds}] {len(bad)} image(s) excluded: preprocessing failed (reasons in exclusions.csv)")
    qc = pd.DataFrame([{"image_id": i, **qc_old[i]} for i in labels["image_id"]])
    manifest = pd.DataFrame([{"image_id": k, **v} for k, v in man.items()]).reindex(columns=MAN_COLS)
    m2 = manifest[manifest["image_id"].isin(labels["image_id"]) & manifest["source_sha256_16"].notna()
                  & (manifest["source_sha256_16"].astype(str) != "")]
    dups = m2[m2["source_sha256_16"].duplicated(keep=False)].merge(
        labels[["image_id", "known_classes", "unknown_terms", "is_normal"]], on="image_id")
    if len(dups):
        sig = dups["known_classes"] + "|" + dups["unknown_terms"] + "|" + dups["is_normal"].astype(str)
        dups = dups.assign(label_conflict=(sig.groupby(dups["source_sha256_16"]).transform("nunique") > 1).astype(int))
        print(f"[cache {ds}] {dups['source_sha256_16'].nunique()} groups of byte-identical source images "
              f"({len(dups)} images) -- kept, listed in duplicates.csv")

    ann.to_csv(os.path.join(cache, "annotations_original.csv"), index=False)
    labels.to_csv(os.path.join(cache, "labels.csv"), index=False)
    excl.reindex(columns=["image_id", "dataset", "split", "patient_id", "eye", "source_relpath", "keywords_raw",
                          "reason", "detail"]).to_csv(os.path.join(cache, "exclusions.csv"), index=False)
    terms_long.to_csv(os.path.join(cache, "terms_long.csv"), index=False)
    qc.to_csv(qc_path, index=False)
    manifest.to_csv(man_path, index=False)
    dups.to_csv(os.path.join(cache, "duplicates.csv"), index=False)
    mapping_df.to_csv(os.path.join(cache, "mapping_snapshot.csv"), index=False)
    meta = {"dataset": ds, "built_or_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mapping_file": os.path.abspath(spec["map_csv"]), "mapping_sha256": sha256_file(spec["map_csv"]),
            "disease_cols": DISEASE_COLS,
            "preprocessing": {"method": "fundus_image_toolbox.crop(path, size=518) -> RGB -> PNG", "img_size": IMG_SIZE,
                              "fundus_image_toolbox": ENV_MANIFEST["packages"].get("fundus_image_toolbox")},
            "label_protocol": {"OMIT": "term dropped (JSIEC: class folder dropped)", "unmapped": "run stops",
                               "normal_with_disease_terms": "excluded", "odir_keyword_split": "',' and U+FF0C, trimmed",
                               "odir_patient_level_columns": "not used"},
            "counts": {"source_images": int(len(ann)), "evaluable": int(len(labels)),
                       "excluded": {k: int(v) for k, v in excl["reason"].value_counts().items()} if len(excl) else {},
                       "normal": int(labels["is_normal"].sum()),
                       "known_only": int((labels["has_known"].astype(bool) & ~labels["is_unknown"].astype(bool)).sum()),
                       "unknown_only": int((labels["is_unknown"].astype(bool) & ~labels["has_known"].astype(bool)).sum()),
                       "mixed": int(labels["is_mixed"].sum()), "has_debatable": int(labels["has_debatable"].sum())},
            "labels_sha256": sha256_file(os.path.join(cache, "labels.csv"))}
    json.dump(meta, open(os.path.join(cache, "cache_meta.json"), "w"), indent=2)
    print(f"[cache {ds}] ready in {time.time()-t0:.0f}s: {meta['counts']}")

def load_cache(ds):
    cache = DS_SPECS[ds]["cache"]
    meta = json.load(open(os.path.join(cache, "cache_meta.json")))
    assert meta["disease_cols"] == DISEASE_COLS, f"{ds}: cache was built for a different class order"
    L = pd.read_csv(os.path.join(cache, "labels.csv"), dtype={"image_id": str, "patient_id": str}, keep_default_na=False)
    for c in ("is_normal", "has_known", "is_unknown", "is_mixed", "is_diseased", "has_debatable", "n_omit_terms") + tuple(YCOLS):
        L[c] = L[c].astype(int)
    missing = [i for i in L["image_id"] if not os.path.exists(os.path.join(cache, "images", f"{i}.png"))]
    assert not missing, f"{ds}: {len(missing)} cached PNGs missing, e.g. {missing[:3]}"
    return L, meta

EXT_CACHE = {}
if RUN_EXTERNAL:
    for ds in EXT_DATASETS:
        _meta_p = os.path.join(DS_SPECS[ds]["cache"], "cache_meta.json")
        _cur = (os.path.exists(_meta_p) and
                json.load(open(_meta_p)).get("mapping_sha256") == sha256_file(DS_SPECS[ds]["map_csv"]))
        if _cur:
            print(f"[cache {ds}] current for this mapping file -> reused read-only")
        else:
            build_cache(ds)
        EXT_CACHE[ds] = load_cache(ds)
        L, meta = EXT_CACHE[ds]
        print(f"[ext {ds}] evaluable {len(L)} | normal {int(L.is_normal.sum())} | known only "
              f"{int((L.has_known.astype(bool) & ~L.is_unknown.astype(bool)).sum())} | unknown only "
              f"{int((L.is_unknown.astype(bool) & ~L.has_known.astype(bool)).sum())} | mixed {int(L.is_mixed.sum())} | "
              f"excluded {meta['counts']['excluded']}")
        PROVENANCE[f"{ds.lower()}_cache"] = {"path": os.path.abspath(DS_SPECS[ds]["cache"]),
                                             "mapping_sha256": meta["mapping_sha256"],
                                             "labels_sha256": sha256_file(os.path.join(DS_SPECS[ds]["cache"], "labels.csv")),
                                             "counts": meta["counts"]}
    write_json(PROVENANCE, os.path.join(REPRO_DIR, "data_provenance.json"))
    print("[repro] data_provenance.json updated with the external cache fingerprints")
else:
    print("[ext] RUN_EXTERNAL=False -- skipped")


# In[ ]:


EXT_RES_DIR = os.path.join(RESULTS_DIR, "external")
EXT_PRED_DIR = os.path.join(PRED_DIR, "external")
for _d in (EXT_RES_DIR, EXT_PRED_DIR):
    os.makedirs(_d, exist_ok=True)

class ExternalPNG(Dataset):
    def __init__(self, ids, img_dir):
        self.ids, self.img_dir = list(ids), img_dir
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, f"{self.ids[i]}.png")).convert("RGB")
        assert img.size == (IMG_SIZE, IMG_SIZE), f"{self.ids[i]}: {img.size}"
        return (torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0,
                torch.zeros(NUM_CLASSES), self.ids[i])

def _cluster_groups(clusters):
    _, inv = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)

def cluster_bootstrap(fn, groups, n_boot=N_BOOT_EXT, seed=0):
    rng = np.random.RandomState(seed)
    v = [x for x in (fn(np.concatenate([groups[g] for g in rng.randint(0, len(groups), len(groups))]))
                     for _ in range(n_boot)) if x == x]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)

EXT_VARIANTS = {"primary": lambda L: np.ones(len(L), bool),
                "no_debatable": lambda L: L["has_debatable"].values == 0,
                "no_omit_terms": lambda L: L["n_omit_terms"].values == 0}

def ext_evaluate(ds, seed, L, P, thr, ref, variant, t_op):
    idx = np.where(EXT_VARIANTS[variant](L))[0]
    Lv = L.iloc[idx].reset_index(drop=True)
    Y, p, n = Lv[YCOLS].values.astype(np.float64), P["probs"][idx].astype(np.float64), len(idx)
    unk, dis = Lv["is_unknown"].values.astype(bool), Lv["is_diseased"].values.astype(bool)
    known_dis = Lv["has_known"].values.astype(bool) & ~unk
    covered = [j for j in range(NUM_CLASSES) if 0 < Y[:, j].sum() < n]
    S = detector_scores(P["uhat"][idx], P["energy"][idx], P["maxlogit"][idx], P["knn"][idx], ref=ref)
    mac = macro_auroc_auprc(p, Y, covered)
    ps = prediction_sets(p, np.full(NUM_CLASSES, thr))
    pm, sm = prf_metrics(ps, Y, covered), set_metrics(ps, Y, covered)
    head = {"dataset": ds, "seed": seed, "variant": variant, "n_images": n, "n_unknown": int(unk.sum()),
            "n_known_diseased": int(known_dis.sum()), "n_normal": int(Lv["is_normal"].sum()),
            "unknown_prevalence": float(unk[dis].mean()) if dis.any() else np.nan, "n_covered_classes": len(covered),
            "threshold": thr, "known_map": tiered_evaluation(Y, p, DISEASE_COLS, TIER_LABELS)["primary_macro_map"],
            "known_macro_auroc": mac["macro_auroc"], "known_macro_auprc": mac["macro_auprc"],
            "macro_f1": pm["macro_f1"], "micro_f1": pm["micro_f1"], "set_size": float(ps[:, covered].sum(1).mean()),
            "jaccard": sm["jaccard"], "ece": expected_calibration_error(p[:, covered], Y[:, covered]),
            "brier": brier_score(p, Y, covered)}
    flag = S["ens3_ref"] >= t_op
    head.update({"t_ens3": t_op, "op_sensitivity": float(flag[unk].mean()) if unk.any() else np.nan,
                 "op_specificity": float((~flag)[known_dis].mean()) if known_dis.any() else np.nan})
    mixed = Lv["is_mixed"].values.astype(bool)
    head.update({"n_mixed": int(mixed.sum()), "mixed_jaccard": np.nan, "mixed_exact": np.nan, "J_mixed": np.nan})
    if mixed.sum() >= 5:
        smx = set_metrics(ps[mixed], Y[mixed], covered)
        head.update({"mixed_jaccard": smx["jaccard"], "mixed_exact": smx["exact_match"],
                     "J_mixed": float((smx["exact_vec"] * flag[mixed]).mean())})
    urows = []
    for pop, mk in (("diseased", dis), ("all", np.ones(n, bool))):
        y = unk[mk].astype(int)
        for d in DETECTORS:
            urows.append({"dataset": ds, "seed": seed, "variant": variant, "population": pop, "detector": d,
                          "n": int(mk.sum()), "n_pos": int(y.sum()), **unknown_metric_block(y, S[d][mk])})
            if pop == "diseased":
                head[f"{d}_auroc"] = urows[-1]["auroc"]
                if d == "ens3":
                    head.update({f"ens3_{k}": urows[-1][k] for k in ("auprc", "fpr95", "det_err", "aurc")})
            elif d == "ens3":
                head["ens3_auroc_all"] = urows[-1]["auroc"]
    ci = {}
    if variant == "primary":
        yd, groups = unk[dis].astype(int), _cluster_groups(Lv["patient_id"].values[dis])
        for d in ("ens3", "ensemble_ml", "knn", "ens_mk", "maxlogit"):
            ci[f"{d}_auroc_ci"] = cluster_bootstrap(lambda ix, s=S[d][dis]: safe_auroc(yd[ix], s[ix]), groups, seed=7)
        ci["ens3_minus_ens_mk_ci"] = cluster_bootstrap(
            lambda ix: safe_auroc(yd[ix], S["ens3"][dis][ix]) - safe_auroc(yd[ix], S["ens_mk"][dis][ix]), groups, seed=9)
    trows = []
    terms = [[t for t in s.split(";") if t] for s in Lv["unknown_terms"]]
    for t in sorted({t for tl in terms for t in tl}):
        pos = np.array([t in tl for tl in terms]); mk = pos | known_dis
        trows.append({"dataset": ds, "seed": seed, "variant": variant, "unknown_term": t, "n_pos": int(pos.sum()),
                      **{f"{d}_auroc": safe_auroc(pos[mk].astype(int), S[d][mk]) for d in DETECTORS}})
    return head, urows, trows, ci, S

def load_selected(seed):
    build_model(seed)
    load_checkpoint(os.path.join(seed_dirs(seed)["phaseB"], "selected.pt"), model, None, load_decomposer=True)
    model.eval()
    return model

print("[ext] evaluation functions defined (the external evaluation itself runs in §R)")


# In[ ]:


import copy, io
from PIL import ImageFilter

assert set(ROB_DATASETS) <= {"RFMiD", "JSIEC", "ODIR"}
# ---- each model's settings must match the settings this notebook evaluates with ------------------------------------------
_must_match = ["IMG_SIZE", "PATCH_SIZE", "FOV_THRESH", "KP", "ATTN_TEMPERATURE", "DECOMP_TAU_E", "TOP_K_RESIDUAL_FRAC",
               "LORA_RANK", "LORA_ALPHA", "DROPOUT_P", "USE_FOV_MASK", "FOV_ERODE_RADIUS", "FOV_TOKEN_KEEP",
               "ENS3_COMPONENTS", "KNN_K", "FOLD_SEED", "N_FOLDS", "CALIB_SPLIT_SEED", "CALIB_FRAC", "CONFORMAL_ALPHA",
               "NUM_CLASSES"]
_folds_here = {str(f + 1): sorted(DISEASE_COLS[i] for i in w) for f, w in enumerate(FOLDS)}
for _mn, _i in MODEL_INFO.items():
    _c = _i["cfg"]
    _bad = {k: (_c.get(k), _clean(globals()[k])) for k in _must_match if _c.get(k) != _clean(globals()[k])}
    if _c.get("heldout_label_sets") != _folds_here:
        _bad["heldout_label_sets"] = "differ"
    if _c.get("disease_cols") != DISEASE_COLS:
        _bad["disease_cols"] = "differ"
    if _i["kind"] in ("anatomy", "rotnorm"):
        _bad.update({k: (_c.get(k), globals()[k]) for k in ("LM_MIN_DIST", "LM_MAX_DIST")
                     if _c.get(k) is not None and abs(float(_c[k]) - float(globals()[k])) > 1e-6})
    if _bad:
        raise RuntimeError(f"{_mn}: settings differ from this notebook (model, here): {_bad}")
print("[rob] models: " + " | ".join(f"{m} ({i['kind']}, suites {i['suites']})" for m, i in MODEL_INFO.items()))

# ---- image sources -------------------------------------------------------------------------------------------------
ORIG_DIR = {"RFMiD": os.path.join(RFMID_CACHE, "test", "images"),
            "JSIEC": os.path.join(DS_SPECS["JSIEC"]["cache"], "images"),
            "ODIR": os.path.join(DS_SPECS["ODIR"]["cache"], "images")}
A11_DIR = {"RFMiD": os.path.join(RFMID_ROT_CACHE, "test", "images"), "JSIEC": os.path.join(JSIEC_ROT_CACHE, "images"),
           "ODIR": os.path.join(ODIR_ROT_CACHE, "images")}
IDS = {"RFMiD": TEST_DF_FULL["ID"].astype(str).tolist()}
for _ds in ("JSIEC", "ODIR"):
    IDS[_ds] = EXT_CACHE[_ds][0]["image_id"].astype(str).tolist()
CLEAN_LM = {}
if NEED_LANDMARKS:
    CLEAN_LM["RFMiD"] = LANDMARKS["test"]
    for _ds in ("JSIEC", "ODIR"):
        EXT_LANDMARKS[_ds] = ensure_landmarks(_ds.lower(), ORIG_DIR[_ds], IDS[_ds])
        CLEAN_LM[_ds] = EXT_LANDMARKS[_ds]
if NEED_ROTATED:
    for _ds in ("JSIEC", "ODIR"):
        ensure_rotated(_ds.lower(), ORIG_DIR[_ds], A11_DIR[_ds], EXT_LANDMARKS[_ds], IDS[_ds])

# ---- corruptions and rotation ------------------------------------------------------------------------------------
def _img_rng(iid, tag):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f"{iid}|{tag}".encode()).digest()[:8], "little"))

def corrupt(img, kind, level, iid):
    """Corrupt the retina of a PIL RGB crop; pixels outside its original field of view stay black."""
    a = np.asarray(img, dtype=np.float32) / 255.0
    fov = a.max(axis=2) > (FOV_THRESH / 255.0)
    if kind == "gaussian_noise":
        a = a + _img_rng(iid, kind).normal(0.0, level, a.shape).astype(np.float32)
    elif kind == "gaussian_blur":
        a = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=level)), dtype=np.float32) / 255.0
    elif kind == "jpeg":
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=int(level)); buf.seek(0)
        a = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    elif kind == "underexposure":
        a = a * level
    elif kind == "haze":
        a = (1.0 - level) * a + level
    elif kind == "uneven_illumination":
        th = _img_rng(iid, kind).uniform(0.0, 2.0 * np.pi)
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]].astype(np.float32)
        proj = (xx - xx.mean()) * np.cos(th) + (yy - yy.mean()) * np.sin(th)
        pr = proj[fov] if fov.any() else proj.ravel()
        t = np.clip((proj - pr.min()) / max(float(pr.max() - pr.min()), 1e-6), 0.0, 1.0)
        a = a * (1.0 - (1.0 - level) * t)[..., None]
    else:
        raise ValueError(kind)
    a = np.clip(a, 0.0, 1.0)
    a[~fov] = 0.0
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))

ROT_CENTER = ((IMG_SIZE - 1) / 2.0, (IMG_SIZE - 1) / 2.0)

def rotate_image(img, angle):
    return img.rotate(angle, resample=Image.BICUBIC, center=ROT_CENTER, fillcolor=(0, 0, 0))

def rotate_points(xy, angle):
    """Where points (..., 2) land after rotate_image(angle): counter-clockwise on screen, y pointing down."""
    t, c = np.radians(angle), np.array(ROT_CENTER)
    d = np.asarray(xy, dtype=float) - c
    return np.stack([c[0] + d[..., 0] * np.cos(t) + d[..., 1] * np.sin(t),
                     c[1] - d[..., 0] * np.sin(t) + d[..., 1] * np.cos(t)], axis=-1)

def _rob_write(df, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

CONDITIONS = [{"suite": "clean", "corruption": "none", "severity": 0, "level": 0.0}]
if any("corruption" in i["suites"] for i in MODEL_INFO.values()):
    CONDITIONS += [{"suite": "corruption", "corruption": k, "severity": i + 1, "level": float(v)}
                   for k, lv in ROB_CORRUPTIONS.items() for i, v in enumerate(lv)]
if any("rotation" in i["suites"] for i in MODEL_INFO.values()):
    CONDITIONS += [{"suite": "rotation", "corruption": "rotation", "severity": int(a), "level": float(a)} for a in ROB_ANGLES]

def models_for(cond):
    return [m for m, i in MODEL_INFO.items() if cond["suite"] == "clean" or cond["suite"] in i["suites"]]

# ---- checks: deterministic, border stays black, rotation geometry -------------------------------------------------
_im = Image.open(os.path.join(ORIG_DIR["RFMiD"], f"{IDS['RFMiD'][0]}.png")).convert("RGB")
_fov = np.asarray(_im).max(axis=2) > FOV_THRESH
for _k, _lv in ROB_CORRUPTIONS.items():
    _a, _b = np.asarray(corrupt(_im, _k, _lv[-1], "x")), np.asarray(corrupt(_im, _k, _lv[-1], "x"))
    assert np.array_equal(_a, _b) and _a[~_fov].max(initial=0) == 0 and _a.shape == (IMG_SIZE, IMG_SIZE, 3), _k
    assert not np.array_equal(_a, np.asarray(_im)), f"{_k} left the image unchanged"
_dot = Image.new("RGB", (IMG_SIZE, IMG_SIZE)); _p = (380.0, 150.0)
_dot.putpixel((int(_p[0]), int(_p[1])), (255, 255, 255))
for _ang in (30, 90, 180):
    _r = np.asarray(rotate_image(_dot, _ang)).max(axis=2).astype(float)
    _yx = np.unravel_index(np.argmax(_r), _r.shape)
    assert np.hypot(*(rotate_points(_p, _ang) - np.array([_yx[1], _yx[0]]))) < 1.5, f"rotation geometry at {_ang}"
print(f"[rob] {len(CONDITIONS)} conditions: " + " | ".join(
    f"{s} {sum(c['suite'] == s for c in CONDITIONS)} ({', '.join(models_for(next(c for c in CONDITIONS if c['suite'] == s)))})"
    for s in ("clean", "corruption", "rotation") if any(c["suite"] == s for c in CONDITIONS)) + " | transform checks OK")


# In[ ]:


# ---- landmarks of rotated images (A10, A11) -----------------------------------------------------------------------
LANDMARK_ROT_STATS = []
_ROT_LM = {}

def rotated_landmarks(ds, angle):
    """Fovea / optic-disc detections on the rotated crops of `ds` (cached), with the acceptance rule of the clean
    landmarks; rejected detections take the median accepted position of this set and angle."""
    if (ds, angle) in _ROT_LM:
        return _ROT_LM[(ds, angle)]
    os.makedirs(ROB_LANDMARK_ROT_DIR, exist_ok=True)
    path = os.path.join(ROB_LANDMARK_ROT_DIR, f"{ds.lower()}_rot{int(angle):+04d}.csv")
    ids = IDS[ds]
    rows = (pd.read_csv(path, dtype={"ID": str}, keep_default_na=False, na_values=[""]).to_dict("records")
            if os.path.exists(path) else [])
    have = {r["ID"] for r in rows}
    todo = [i for i in ids if i not in have]
    if todo:
        m, t0 = get_landmark_model(), time.time()
        print(f"[landmarks {ds} {angle:+d} deg] detecting on {len(todo)} rotated crops ...")
        for n, i in enumerate(todo):
            err = ""
            try:
                im = rotate_image(Image.open(os.path.join(ORIG_DIR[ds], f"{i}.png")).convert("RGB"), angle)
                p = np.asarray(m.predict(im), dtype=float).ravel()
                p = p[:4] if p.size >= 4 else np.full(4, np.nan)
            except Exception as e:
                p, err = np.full(4, np.nan), f"{type(e).__name__}: {str(e)[:120]}"
            rows.append({"ID": i, **{f"raw_{c}": float(v) for c, v in zip(LM_COLS, p)},
                         "detected_ok": int(landmark_ok(p)), "error": err})
            if (n + 1) % 500 == 0 or n + 1 == len(todo):
                _rob_write(pd.DataFrame(rows), path)
                print(f"[landmarks {ds} {angle:+d} deg] {n + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows).drop_duplicates("ID", keep="last").set_index("ID").loc[ids]
    ok = df["detected_ok"].astype(int).values == 1
    raw = df[[f"raw_{c}" for c in LM_COLS]].values.astype(float)
    med = np.median(raw[ok], axis=0) if ok.any() else np.full(4, IMG_SIZE / 2.0)
    lm = np.where(ok[:, None], raw, med[None, :])
    clean = CLEAN_LM[ds].loc[ids]
    both = ok & (clean["detected_ok"].astype(int).values == 1)
    ef = np.hypot(*(lm[:, 0:2] - rotate_points(clean[["fovea_x", "fovea_y"]].values, angle)).T)[both]
    eo = np.hypot(*(lm[:, 2:4] - rotate_points(clean[["od_x", "od_y"]].values, angle)).T)[both]
    rec = {"dataset": ds, "angle": int(angle), "images": len(ids), "accepted": float(ok.mean()),
           "median_error_fovea_px": float(np.median(ef)) if len(ef) else np.nan,
           "median_error_disc_px": float(np.median(eo)) if len(eo) else np.nan,
           "share_disc_error_over_50px": float(np.mean(eo > 50)) if len(eo) else np.nan}
    LANDMARK_ROT_STATS.append(rec)
    print(f"[landmarks {ds} {angle:+d} deg] accepted {rec['accepted']:.1%} | median error fovea "
          f"{rec['median_error_fovea_px']:.1f} px, disc {rec['median_error_disc_px']:.1f} px")
    out = pd.DataFrame(lm, index=ids, columns=LM_COLS).assign(detected_ok=ok.astype(int))
    _ROT_LM[(ds, angle)] = out
    return out

# ---- the shared input pipeline: one transformed image per test image, every model's input derived from it --------------
class SharedSet(Dataset):
    """Yields (plain input, A11 input, A10 landmarks, row index). `plain` goes to RetinaOWL, A9 and A10."""
    def __init__(self, ids, plain_dir, a11_dir, plain_fn=None, a11_fn=None, lm=None, need_plain=True, need_a11=False):
        self.ids, self.plain_dir, self.a11_dir = list(ids), plain_dir, a11_dir
        self.plain_fn, self.a11_fn, self.lm = plain_fn, a11_fn, lm
        self.need_plain, self.need_a11 = need_plain, need_a11
    def __len__(self):
        return len(self.ids)
    @staticmethod
    def _t(img):
        assert img.size == (IMG_SIZE, IMG_SIZE)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    def __getitem__(self, i):
        iid = self.ids[i]
        xp = xa = torch.zeros(1)
        img = None
        if self.need_plain or self.a11_dir is None:
            img = Image.open(os.path.join(self.plain_dir, f"{iid}.png")).convert("RGB")
            if self.plain_fn is not None:
                img = self.plain_fn(img, iid)
            if self.need_plain:
                xp = self._t(img)
        if self.need_a11:
            if self.a11_dir is not None:             # clean / corruption: A11's own aligned crop
                ia = Image.open(os.path.join(self.a11_dir, f"{iid}.png")).convert("RGB")
                xa = self._t(self.a11_fn(ia, iid) if self.a11_fn is not None else ia)
            else:                                    # rotation: align the same rotated pixels the other models see
                xa = self._t(self.a11_fn(img, iid))
        lm = torch.from_numpy(self.lm[i]) if self.lm is not None else torch.zeros(4)
        return xp, xa, lm, i

def condition_set(cond, ds, models):
    kinds = {MODEL_INFO[m]["kind"] for m in models}
    need_plain, need_a11, need_lm = bool(kinds & {"plain", "anatomy"}), "rotnorm" in kinds, "anatomy" in kinds
    ids = IDS[ds]
    if cond["suite"] == "clean":
        lm = CLEAN_LM[ds].loc[ids, LM_COLS].values.astype(np.float32) if need_lm else None
        return SharedSet(ids, ORIG_DIR[ds], A11_DIR[ds], lm=lm, need_plain=need_plain, need_a11=need_a11)
    if cond["suite"] == "corruption":
        k, lv = cond["corruption"], cond["level"]
        fn = lambda img, iid: corrupt(img, k, lv, iid)
        lm = CLEAN_LM[ds].loc[ids, LM_COLS].values.astype(np.float32) if need_lm else None
        return SharedSet(ids, ORIG_DIR[ds], A11_DIR[ds], plain_fn=fn, a11_fn=fn, lm=lm,
                         need_plain=need_plain, need_a11=need_a11)
    ang = int(cond["level"])
    rl = rotated_landmarks(ds, ang) if (need_lm or need_a11) else None
    fo = ({i: (r.fovea_x, r.fovea_y, r.od_x, r.od_y, int(r.detected_ok)) for i, r in rl.iterrows()}
          if need_a11 else {})
    def a11_fn(img, iid, fo=fo):
        fx, fy, ox, oy, ok = fo[iid]
        return rotate_align(img, (fx, fy), (ox, oy))[0] if ok else img
    lm = rl[LM_COLS].values.astype(np.float32) if need_lm else None
    return SharedSet(ids, ORIG_DIR[ds], None, plain_fn=lambda img, iid, ang=ang: rotate_image(img, ang),
                     a11_fn=a11_fn, lm=lm, need_plain=True, need_a11=need_a11)
print("[rob] shared input pipeline defined")


# In[ ]:


# ---- A10's anatomy-grounded tokenizer (as in the A10 notebook) -------------------------------------------------------
EPS = 1e-6
PATCH_CENTERS = ((torch.stack(torch.meshgrid(torch.arange(GRID_SIZE, dtype=torch.float32),
                                             torch.arange(GRID_SIZE, dtype=torch.float32), indexing="ij"), -1)
                  .flip(-1).reshape(-1, 2) + 0.5) * PATCH_SIZE).to(DEVICE)      # (N, 2) as (x, y) pixels

def compute_polar_coords(landmarks, patch_centers=None, img_size=IMG_SIZE):
    """landmarks (B, 4) = fovea_x, fovea_y, od_x, od_y -> rho, theta (B, N); delta_disc, delta_fovea (B, N, 2)."""
    pc = PATCH_CENTERS if patch_centers is None else patch_centers
    f, d = landmarks[:, 0:2].unsqueeze(1), landmarks[:, 2:4].unsqueeze(1)
    q = pc.unsqueeze(0).expand(landmarks.shape[0], -1, -1)
    rho = torch.norm(q - d, dim=-1) / (torch.norm(f - d, dim=-1) + EPS)
    theta = torch.atan2((q - d)[..., 1], (q - d)[..., 0]) - torch.atan2((f - d)[..., 1], (f - d)[..., 0])
    theta = torch.atan2(torch.sin(theta), torch.cos(theta))
    return rho, theta, (q - d) / img_size, (q - f) / img_size

class AnatomyGroundedTokenizer(nn.Module):
    """z'_q = z_q + MLP_polar(rho, sin theta, cos theta) + MLP_landmark(delta_disc, delta_fovea)."""
    def __init__(self, embed_dim, hidden=64):
        super().__init__()
        self.polar_mlp = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))
        self.landmark_mlp = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))
    def forward(self, patch_tokens, landmarks):
        rho, theta, dd, df = compute_polar_coords(landmarks.to(patch_tokens.device).float())
        e = self.polar_mlp(torch.stack([rho, torch.sin(theta), torch.cos(theta)], -1)) \
            + self.landmark_mlp(torch.cat([dd, df], -1))
        return patch_tokens + e.to(patch_tokens.dtype)

# ---- every model's original checkpoints, side by side -----------------------------------------------------------------
def load_robust_model(name, seed):
    info = MODEL_INFO[name]
    build_model(seed)
    load_checkpoint(info["ckpt"][seed], model, None, load_decomposer=True)
    m = copy.deepcopy(model).eval()
    m.tokenizer = None
    if info["kind"] == "anatomy":
        ck = torch.load(info["ckpt"][seed], map_location=DEVICE, weights_only=False)
        tok = AnatomyGroundedTokenizer(EMBED_DIM, hidden=int(info["cfg"].get("ANATOMY_HIDDEN", 64))).to(DEVICE)
        tok.load_state_dict(ck["tokenizer_state"], strict=True)
        m.tokenizer = tok.eval()
    return m

@torch.no_grad()
def forward_masks(m, x, masks, lm=None):
    """One backbone pass, then the disease memory and the residual branch once per class mask (same as m(x, mask))."""
    z_raw = m.tokens(x)
    z = m.tokenizer(z_raw, lm) if m.tokenizer is not None else z_raw
    tm = make_token_mask(x) if USE_FOV_MASK else None
    tmf = (tm if tm is not None else torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)).float().unsqueeze(-1)
    feat = F.normalize((z_raw.float() * tmf).sum(1) / tmf.sum(1).clamp(min=1.0), dim=-1)     # kNN feature (raw tokens)
    outs = []
    for am in masks:
        logits, _, _ = m.memory(z, am)
        d = m.decomposer(z, m.memory.prototypes, z.mean(dim=1), am, token_mask=tm)
        outs.append((logits.float(), d["u_hat"].float()))
    return feat, outs

FOLD_MASKS = [make_active_mask(w) for w in FOLDS]
MODELS = {(mn, s): load_robust_model(mn, s) for mn in MODEL_INFO for s in ROB_SEEDS}
# the shortcut equals the model's own forward (RetinaOWL architecture; A10 is checked through the clean results)
_m0 = MODELS[(next(m for m, i in MODEL_INFO.items() if i["kind"] != "anatomy") if any(
    i["kind"] != "anatomy" for i in MODEL_INFO.values()) else next(iter(MODEL_INFO)), ROB_SEEDS[0])]
if _m0.tokenizer is None:
    _xb = _norm(torch.stack([SharedSet(IDS["RFMiD"][:4], ORIG_DIR["RFMiD"], None)[j][0] for j in range(4)]))
    _f, _o = forward_masks(_m0, _xb, [None, FOLD_MASKS[0]])
    with torch.no_grad():
        for _j, _am in enumerate([None, FOLD_MASKS[0]]):
            _ref = _m0(_xb, _am)
            assert torch.allclose(_o[_j][0], _ref["logits"].float(), atol=1e-3) and \
                torch.allclose(_o[_j][1], _ref["u_hat"].float(), atol=1e-4), "forward_masks differs from the model's forward"
        assert torch.allclose(_f, _ref["feat"].float(), atol=1e-4)
print(f"[rob] {len(MODELS)} checkpoints loaded ({len(MODEL_INFO)} models x {len(ROB_SEEDS)} seeds) | forward shortcut check OK")

# ---- one pass of a set of images through a set of models ------------------------------------------------------------
def run_models(dset, models, masks):
    """Returns {(model, seed): {"feat": (N, D), "logits": [per mask], "uhat": [per mask]}} in dataset order."""
    loader = DataLoader(dset, batch_size=ROB_BATCH, shuffle=False, num_workers=ROB_WORKERS)
    acc = {(mn, s): {"feat": [], "logits": [[] for _ in masks], "uhat": [[] for _ in masks]}
           for mn in models for s in ROB_SEEDS}
    order = []
    for xp, xa, lm, ix in loader:
        order.append(ix.numpy())
        xp_n = _norm(xp) if xp.dim() == 4 else None
        xa_n = _norm(xa) if xa.dim() == 4 else None
        lm_d = lm.to(DEVICE)
        for mn in models:
            kind = MODEL_INFO[mn]["kind"]
            x = xa_n if kind == "rotnorm" else xp_n
            for s in ROB_SEEDS:
                with torch.autocast(device_type="cuda", enabled=USE_AMP):
                    feat, outs = forward_masks(MODELS[(mn, s)], x, masks, lm_d if kind == "anatomy" else None)
                a = acc[(mn, s)]
                a["feat"].append(feat.float().cpu().numpy())
                for j, (lg, uh) in enumerate(outs):
                    a["logits"][j].append(lg.cpu().numpy()); a["uhat"][j].append(uh.cpu().numpy())
    assert np.array_equal(np.concatenate(order), np.arange(len(dset))), "row order drifted"
    return {k: {"feat": np.concatenate(v["feat"]), "logits": [np.concatenate(l) for l in v["logits"]],
                "uhat": [np.concatenate(u) for u in v["uhat"]]} for k, v in acc.items()}

def _blocks(r):
    return [{"uhat": r["uhat"][j], "energy": softmax_energy(torch.from_numpy(r["logits"][j])).numpy(),
             "maxlogit": -r["logits"][j].max(axis=1)} for j in range(len(r["logits"]))]

# ---- clean references of every model (validation and training images, each model's own clean input) ------------------
t0 = time.time()
_all = list(MODEL_INFO)
_kinds = {MODEL_INFO[m]["kind"] for m in _all}
def _clean_set(split, df):
    lm = LANDMARKS[split].loc[df["ID"].astype(str), LM_COLS].values.astype(np.float32) if "anatomy" in _kinds else None
    return SharedSet(df["ID"].astype(str).tolist(), os.path.join(RFMID_CACHE, split, "images"),
                     os.path.join(RFMID_ROT_CACHE, split, "images"), lm=lm,
                     need_plain=bool(_kinds & {"plain", "anatomy"}), need_a11="rotnorm" in _kinds)
RV = run_models(_clean_set("val", VAL_DF_FULL), _all, [None] + FOLD_MASKS)
RT = run_models(_clean_set("train", TRAIN_DF_FULL), _all, [None])
Ytr = TRAIN_DF_FULL[DISEASE_COLS].values.astype(np.float32)
calm = np.array([bool(HALF_IS_CALIB[i]) for i in VAL_DF_FULL["ID"].astype(str)])
CAL = {}
for (mn, s), rv in RV.items():
    SRC = json.load(open(MODEL_INFO[mn]["json"][s]))
    fu = pd.DataFrame(SRC["fold_unknown"])
    Ftr, bv = RT[(mn, s)]["feat"], _blocks(rv)
    thr_re, _ = fit_global_threshold(1.0 / (1.0 + np.exp(-rv["logits"][0][calm])), VAL_LABELS_NP[calm])
    C = {"thr": float(SRC["threshold"]["global"]), "thr_recomputed": float(thr_re), "Ftr": Ftr, "keep": [], "ref": [],
         "t_e": [float(v) for v in fu["t_ens3"].values], "t_op": float(np.nanmedian(fu["t_ens3"].astype(float)))}
    for f, w in enumerate(FOLDS):
        keep = unknown_target_np(Ytr, w) < 0.5
        C["keep"].append(keep)
        C["ref"].append({"uhat": bv[f + 1]["uhat"], "maxlogit": bv[f + 1]["maxlogit"], "knn": knn_score(Ftr[keep], rv["feat"])})
    C["ref_ext"] = {"uhat": bv[0]["uhat"], "maxlogit": bv[0]["maxlogit"], "knn": knn_score(Ftr, rv["feat"])}
    CAL[(mn, s)] = C
for mn in _all:
    print(f"[rob] {mn}: thresholds " + ", ".join(f"s{s} {CAL[(mn, s)]['thr']:.2f} (recomputed {CAL[(mn, s)]['thr_recomputed']:.2f})"
                                               for s in ROB_SEEDS))
print(f"[rob] clean references of {len(CAL)} checkpoints in {(time.time() - t0) / 60:.1f} min")

ROB_METRICS = ["known_map", "known_macro_auroc", "macro_f1", "micro_f1", "ece", "set_size"] + \
              [f"{d}_auroc" for d in DETECTORS] + ["ens3_auprc", "ens3_fpr95", "op_sensitivity", "op_specificity",
                                                   "flag_rate_normal"]

def rfmid_metrics(key, P):
    C, Y = CAL[key], TEST_LABELS_NP
    p = P["probs"]
    T = np.full(NUM_CLASSES, C["thr"])
    tie = tiered_evaluation(Y, p, DISEASE_COLS, TIER_LABELS)
    ps = prediction_sets(p, T)
    pm, sm, mac = prf_metrics(ps, Y), set_metrics(ps, Y), macro_auroc_auprc(p, Y)
    out = {"known_map": tie["primary_macro_map"], "known_macro_auroc": mac["macro_auroc"], "macro_f1": pm["macro_f1"],
           "micro_f1": pm["micro_f1"], "ece": expected_calibration_error(p, Y), "set_size": sm["avg_set_size"]}
    dis = Y.sum(1) > 0
    rows = []
    for f, w in enumerate(FOLDS):
        kt = knn_score(C["Ftr"][C["keep"][f]], P["feat"])
        o = P["fold"][f]
        St = detector_scores(o["uhat"], o["energy"], o["maxlogit"], kt, ref=C["ref"][f])
        ut = unknown_target_np(Y, w)
        flag = St["ens3_ref"] >= C["t_e"][f]
        r = {f"{d}_auroc": safe_auroc(ut[dis], St[d][dis]) for d in DETECTORS}
        r.update({f"ens3_{k}": v for k, v in unknown_metric_block(ut[dis], St["ens3"][dis]).items()})
        r["op_sensitivity"] = float(flag[(ut > 0.5) & dis].mean()) if ((ut > 0.5) & dis).any() else np.nan
        r["op_specificity"] = float((~flag)[(ut < 0.5) & dis].mean())
        r["flag_rate_normal"] = float(flag[~dis].mean()) if (~dis).any() else np.nan
        rows.append(r)
    fr = pd.DataFrame(rows)
    out.update({c: float(np.nanmean(fr[c])) for c in fr.columns})
    return out

EXT_VARIANTS["robust"] = lambda L: np.ones(len(L), bool)        # the primary population, without bootstrap CIs

def external_metrics(ds, key, P):
    C = CAL[key]
    L = EXT_CACHE[ds][0]
    h, _, _, _, S = ext_evaluate(ds, key[1], L, P, C["thr"], C["ref_ext"], "robust", C["t_op"])
    nrm = L["is_normal"].values.astype(bool)
    h["flag_rate_normal"] = float((S["ens3_ref"][nrm] >= C["t_op"]).mean()) if nrm.any() else np.nan
    return {k: h.get(k, np.nan) for k in ROB_METRICS}


# In[ ]:


# ---- run every condition once for all its models (resumable: finished results are kept and skipped) ---------------------
ROWS_PATH = os.path.join(RESULTS_DIR, "robustness_per_seed.csv")
ROB_ROWS = pd.read_csv(ROWS_PATH).to_dict("records") if os.path.exists(ROWS_PATH) else []
_done = {(r["model"], r["suite"], r["corruption"], int(r["severity"]), r["dataset"]) for r in ROB_ROWS}
write_json({"RUN_NAME": RUN_NAME, "ROB_MODELS": {m: {"source": os.path.abspath(i["source"]), "run_name": i["run_name"],
                                                     "kind": i["kind"], "suites": i["suites"]} for m, i in MODEL_INFO.items()},
            "ROB_SEEDS": ROB_SEEDS, "ROB_DATASETS": ROB_DATASETS, "ROB_CORRUPTIONS": ROB_CORRUPTIONS,
            "ROB_ANGLES": ROB_ANGLES, "CONDITIONS": CONDITIONS, "ROB_METRICS": ROB_METRICS,
            "checkpoints_sha256": {f"{m}/seed{s}": sha256_file(MODEL_INFO[m]["ckpt"][s]) for m in MODEL_INFO
                                   for s in ROB_SEEDS}},
           os.path.join(REPRO_DIR, "robustness_config.json"))

def check_clean_against_source():
    """The clean condition recomputed here must reproduce each model's original evaluation."""
    rows = []
    for mn, info in MODEL_INFO.items():
        ext = os.path.join(info["source"], "results", "external", "external_per_seed.csv")
        E = pd.read_csv(ext) if os.path.exists(ext) else pd.DataFrame()
        E = E[E.variant == "primary"] if len(E) else E
        for r in [r for r in ROB_ROWS if r["model"] == mn and r["suite"] == "clean"]:
            if r["dataset"] == "RFMiD":
                h = json.load(open(info["json"][r["seed"]]))["headline"]
                for k in ("known_map", "macro_f1", "ens3_auroc", "maxlogit_auroc", "knn_auroc", "op_specificity"):
                    rows.append({"model": mn, "dataset": "RFMiD", "seed": r["seed"], "metric": k,
                                 "original": h.get(k), "recomputed": r[k]})
            elif len(E):
                e = E[(E.dataset == r["dataset"]) & (E.seed == r["seed"])]
                for k in ("known_map", "macro_f1", "ens3_auroc", "op_specificity"):
                    if len(e):
                        rows.append({"model": mn, "dataset": r["dataset"], "seed": r["seed"], "metric": k,
                                     "original": float(e[k].iloc[0]), "recomputed": r[k]})
    D = pd.DataFrame(rows)
    D["abs_diff"] = (D.original.astype(float) - D.recomputed.astype(float)).abs()
    D.to_csv(os.path.join(REPRO_DIR, "robustness_clean_check.csv"), index=False)
    worst = D.groupby("model").abs_diff.max()
    print("[rob] clean check against the original results (max |difference| per model): "
          + " | ".join(f"{m} {v:.5f}" for m, v in worst.items()))
    if float(np.nanmax(D.abs_diff)) > ROB_CHECK_TOL:
        print(D.sort_values("abs_diff", ascending=False).head(10).to_string(index=False))
        raise RuntimeError("the clean recomputation does not reproduce a model's original results: wrong folder in "
                           "ROB_MODELS, changed caches or changed settings")

for cond in CONDITIONS:
    for ds in ROB_DATASETS:
        todo = [m for m in models_for(cond)
                if (m, cond["suite"], cond["corruption"], int(cond["severity"]), ds) not in _done]
        if not todo:
            continue
        t0 = time.time()
        masks = [None] + FOLD_MASKS if ds == "RFMiD" else [None]
        res = run_models(condition_set(cond, ds, todo), todo, masks)
        for (mn, s), r in res.items():
            b = _blocks(r)
            probs = 1.0 / (1.0 + np.exp(-r["logits"][0]))
            if ds == "RFMiD":
                met = rfmid_metrics((mn, s), {"probs": probs, "feat": r["feat"], "fold": b[1:]})
            else:
                met = external_metrics(ds, (mn, s), {"probs": probs, **b[0], "knn": knn_score(CAL[(mn, s)]["Ftr"], r["feat"])})
            ROB_ROWS.append({"model": mn, **cond, "dataset": ds, "seed": s, **met})
        _rob_write(pd.DataFrame(ROB_ROWS), ROWS_PATH)
        cur = pd.DataFrame([r for r in ROB_ROWS if (r["suite"], r["corruption"], int(r["severity"]), r["dataset"]) ==
                            (cond["suite"], cond["corruption"], int(cond["severity"]), ds)])
        print(f"[rob] {cond['suite']:10} {cond['corruption']:20} {cond['severity']:>3} | {ds:5} | {(time.time() - t0) / 60:4.1f} min | "
              + " | ".join(f"{m}: mAP {g.known_map.mean():.3f} ens3 {g.ens3_auroc.mean():.3f}" for m, g in cur.groupby("model", sort=False)))
    if cond["suite"] == "clean":
        check_clean_against_source()
if LANDMARK_ROT_STATS:
    pd.DataFrame(LANDMARK_ROT_STATS).drop_duplicates(["dataset", "angle"], keep="last").to_csv(
        os.path.join(RESULTS_DIR, "robustness_landmarks.csv"), index=False)
print(f"[rob] all conditions done -> {ROWS_PATH}")


# In[ ]:


# ---- summaries: mean (sd) over seeds, and the change from clean ------------------------------------------------------
RR = pd.DataFrame(ROB_ROWS)
_k = ["model", "suite", "corruption", "severity", "level", "dataset"]
SUM = RR.groupby(_k, sort=False)[ROB_METRICS].agg(["mean", "std"])
SUM.columns = [f"{a}_{b}" for a, b in SUM.columns]
SUM.reset_index().to_csv(os.path.join(RESULTS_DIR, "robustness_summary.csv"), index=False)
CL = RR[RR.suite == "clean"].set_index(["model", "dataset", "seed"])[ROB_METRICS]
DR = RR[RR.suite != "clean"].copy()
for c in ROB_METRICS:
    DR[c] = DR[c].values - CL.loc[list(zip(DR.model, DR.dataset, DR.seed)), c].values
DR.to_csv(os.path.join(RESULTS_DIR, "robustness_change_from_clean_per_seed.csv"), index=False)
DS_ = DR.groupby(_k, sort=False)[ROB_METRICS].agg(["mean", "std"])
DS_.columns = [f"{a}_{b}" for a, b in DS_.columns]
DS_.reset_index().to_csv(os.path.join(RESULTS_DIR, "robustness_change_from_clean.csv"), index=False)
if (DR.suite == "corruption").any():
    MC = DR[DR.suite == "corruption"].groupby(["model", "corruption", "dataset", "seed"])[ROB_METRICS].mean().reset_index()
    MCS = MC.groupby(["model", "corruption", "dataset"], sort=False)[ROB_METRICS].agg(["mean", "std"])
    MCS.columns = [f"{a}_{b}" for a, b in MCS.columns]
    MCS.reset_index().to_csv(os.path.join(RESULTS_DIR, "robustness_mean_over_severities.csv"), index=False)

print("Change from clean, mean over seeds and severities: known mAP / ens3 AUROC")
for su in ("corruption", "rotation"):
    sub = DR[DR.suite == su]
    if not len(sub):
        continue
    t = sub.groupby(["corruption", "dataset", "model"], sort=False)[["known_map", "ens3_auroc"]].mean().unstack("model")
    print(f"  {su}")
    print(t.round(4).to_string())

# ---- paired comparison with the reference model under the same conditions --------------------------------------------
from scipy import stats as _st
rows = []
for mn in [m for m in MODEL_INFO if m != ROB_REFERENCE]:
    for (su, co, sev, ds), g in DR[DR.model == mn].groupby(["suite", "corruption", "severity", "dataset"], sort=False):
        gr = DR[(DR.model == ROB_REFERENCE) & (DR.suite == su) & (DR.corruption == co) & (DR.severity == sev) & (DR.dataset == ds)]
        seeds = sorted(set(g.seed) & set(gr.seed))
        if not seeds:
            continue
        for mk in ("known_map", "macro_f1", "ens3_auroc", "uhat_auroc", "op_specificity", "flag_rate_normal"):
            da = g.set_index("seed").loc[seeds, mk].values.astype(float)
            dr = gr.set_index("seed").loc[seeds, mk].values.astype(float)
            dd, ok = da - dr, np.isfinite(da - dr)
            p = float(_st.ttest_rel(da[ok], dr[ok]).pvalue) if ok.sum() >= 3 and np.nanstd(dd) > 0 else np.nan
            rows.append({"model": mn, "reference": ROB_REFERENCE, "suite": su, "corruption": co, "severity": sev,
                         "dataset": ds, "metric": mk, "n_seeds": int(ok.sum()), "change_model": float(np.nanmean(da)),
                         "change_reference": float(np.nanmean(dr)), "difference_of_changes": float(np.nanmean(dd)),
                         "p_paired": p})
CMP = pd.DataFrame(rows)
if len(CMP):
    CMP.to_csv(os.path.join(RESULTS_DIR, f"robustness_vs_{ROB_REFERENCE}.csv"), index=False)
    print(f"\nChange from clean, model minus {ROB_REFERENCE} (mean over severities); * = p < 0.05 at >= 1 severity")
    for mk in ("known_map", "ens3_auroc", "op_specificity"):
        t = CMP[CMP.metric == mk].groupby(["model", "corruption", "dataset"], sort=False).agg(
            d=("difference_of_changes", "mean"), sig=("p_paired", lambda v: bool((v < 0.05).any())))
        print(f"  {mk}")
        for (m, co, ds), r in t.iterrows():
            print(f"    {m:28} {co:20} {ds:5} {r.d:+.4f}{' *' if r.sig else ''}")

# ---- figures ---------------------------------------------------------------------------------------------------
S_ = SUM.reset_index()
for su, xs, xl in (("corruption", None, "severity"), ("rotation", None, "rotation (degrees)")):
    ms = [m for m in MODEL_INFO if su in MODEL_INFO[m]["suites"]]
    if not ms:
        continue
    cols = list(ROB_CORRUPTIONS) if su == "corruption" else ["rotation"]
    for mk, lab in (("ens3_auroc", "Unknown AUROC (ens3)"), ("known_map", "Known mAP"), ("op_specificity", "Known-disease specificity")):
        fig, ax = plt.subplots(len(ROB_DATASETS), len(cols), figsize=(3.2 * len(cols) + 1, 2.8 * len(ROB_DATASETS)), squeeze=False)
        for i, ds in enumerate(ROB_DATASETS):
            for j, co in enumerate(cols):
                for m in ms:
                    c0 = S_[(S_.model == m) & (S_.suite == "clean") & (S_.dataset == ds)][f"{mk}_mean"]
                    g = S_[(S_.model == m) & (S_.corruption == co) & (S_.dataset == ds)].sort_values("severity")
                    if len(g) and len(c0):
                        ax[i, j].plot([0] + g.severity.tolist(), [float(c0.iloc[0])] + g[f"{mk}_mean"].tolist(), marker="o", label=m)
                ax[i, j].set_title(f"{ds}: {co}", fontsize=9); ax[i, j].set_xlabel(xl, fontsize=8)
                ax[i, j].set_xticks(sorted({0} | set(S_[S_.corruption == co].severity.astype(int))))
        ax[0, 0].legend(fontsize=7)
        fig.suptitle(lab)
        plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, f"robustness_{su}_{mk}.png"), dpi=130); plt.show()

