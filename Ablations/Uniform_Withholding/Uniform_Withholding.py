#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys

# ---- paths -----------------------------------------------------------------------------------
OUTPUT_ROOT   = "./Output"       # writable; everything this notebook produces
PHASE_A_SOURCE = "../../Model/Output"                             # RetinaOWL output folder whose Phase A is reused ("" = train here)
RFMID_ROOT    = "../../Dataset/RFMID"                      # Training_Set/ Evaluation_Set/ Test_Set/
RFMID_CACHE   = "../../Preprocessing_Cache/RFMiD"  # <split>/images/<ID>.png -- built if missing
DINOV2_ROOT   = "../../Dataset/DINOv2"                     # config.json + pytorch_model.bin (ViT-S/14)
JSIEC_ROOT    = "../../Dataset/JSIEC"                      # the 39 class folders
ODIR_ROOT     = "../../Dataset/ODIR-5K"                    # Training Set/ Off-site Test Set/ On-site Test Set/
JSIEC_CACHE   = "../../Preprocessing_Cache/JSIEC"  # built if missing
ODIR_CACHE    = "../../Preprocessing_Cache/ODIR"   # built if missing
JSIEC_MAP_CSV = "../../Preprocessing_Cache/JSIEC_class_to_RFMiD_mapping.csv"
ODIR_MAP_CSV  = "../../Preprocessing_Cache/ODIR_keyword_to_RFMiD_mapping.csv"

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
RUN_NAME = "RetinaOWL_A3_uniform_withholding"
ABLATION = "A3_uniform_withholding"
WITHHOLDING_MODE = "uniform"     # training sampler only; selection draws and evaluation folds stay stratified
print(f"[cfg] {RUN_NAME} | seeds {SEEDS_THIS_RUN} | Phase A {NUM_EPOCHS_PHASE_A} + Phase B {NUM_EPOCHS_PHASE_B} epochs "
      f"| external={RUN_EXTERNAL} | leave-classes-out={RUN_LCO} folds {LCO_FOLDS} seeds {LCO_SEEDS} | out {OUTPUT_ROOT}")


# In[ ]:


import glob, json, math, random, re, shutil, time, socket, platform, hashlib, unicodedata, importlib
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
from sklearn.covariance import LedoitWolf
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

print("=" * 78)
if PREFLIGHT_ERRORS:
    raise SystemExit(f"Preflight failed with {len(PREFLIGHT_ERRORS)} problem(s); fix the paths in section 0.")
print(f"  PREFLIGHT PASSED" + (f" with {len(PREFLIGHT_WARNINGS)} warning(s)" if PREFLIGHT_WARNINGS else ""))
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

def sample_training_withheld_set(rng=None):
    """Phase B training sampler of this ablation: WITHHOLD_K classes drawn uniformly from the trained classes,
    ignoring frequency tiers. Same number of classes as the stratified sampler."""
    rng = rng if rng is not None else np.random
    pool = [i for i in range(NUM_CLASSES) if i not in LCO_MASKED]
    return set(int(i) for i in rng.choice(pool, size=WITHHOLD_K, replace=False))

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
        u = pseudo_unknown_target(labels_all, sample_training_withheld_set(rng))
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


def sampler_profile(fn, name, n=POS_WEIGHT_U_EPISODES, seed=POS_WEIGHT_U_SEED, df=None):
    df = train_df if df is None else df
    Y = df[DISEASE_COLS].values > 0.5
    rng = np.random.RandomState(seed)
    tier_idx = {t: set(TIER_TO_INDICES[t]) for t in TIERS}
    sizes, counts, prev = [], {t: [] for t in TIERS}, []
    for _ in range(n):
        w = fn(rng)
        sizes.append(len(w))
        for t in TIERS:
            counts[t].append(len(w & tier_idx[t]))
        prev.append(float(Y[:, sorted(w)].any(axis=1).mean()))
    row = {"sampler": name, "draws": n, "set_size_min": int(min(sizes)), "set_size_max": int(max(sizes)),
           "share_no_head": float(np.mean(np.array(counts["head"]) == 0)), "unknown_prevalence_mean": float(np.mean(prev)),
           "unknown_prevalence_sd": float(np.std(prev))}
    for t in TIERS:
        row[f"{t}_mean"], row[f"{t}_sd"] = float(np.mean(counts[t])), float(np.std(counts[t]))
    return row

SAMPLER_PROFILE = pd.DataFrame([sampler_profile(sample_training_withheld_set, "uniform (training, this ablation)"),
                                sampler_profile(sample_withheld_set, "tier-stratified (RetinaOWL; selection draws here)")])
assert (SAMPLER_PROFILE[["set_size_min", "set_size_max"]] == WITHHOLD_K).all().all(), "both samplers must withhold 7 classes"
SAMPLER_PROFILE.to_csv(os.path.join(REPRO_DIR, "training_sampler.csv"), index=False)
for _, r in SAMPLER_PROFILE.iterrows():
    print(f"[sampler] {r.sampler:50} | size {r.set_size_min}-{r.set_size_max} | per-tier mean "
          + " ".join(f"{t} {r[f'{t}_mean']:.2f}±{r[f'{t}_sd']:.2f}" for t in TIERS)
          + f" | no head class {r.share_no_head:.1%} | unknown prevalence {r.unknown_prevalence_mean:.3f}")
print(f"[sampler] unknown-loss pos_weight from the uniform sampler: {POS_WEIGHT_U:.3f}")


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
            if i % ACCUM_STEPS == 0:                     # one withheld set per effective batch (uniform, 7 classes)
                me = sample_training_withheld_set()
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
    import_phase_a(seed, arm, dirs)
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


PHASE_A_KEYS_CHECKED = ["IMG_SIZE", "PATCH_SIZE", "FOV_THRESH", "TIER_EDGES", "KP", "ATTN_TEMPERATURE", "LORA_RANK",
                        "LORA_ALPHA", "DROPOUT_P", "NUM_EPOCHS_PHASE_A", "MICRO_BATCH_SIZE", "ACCUM_STEPS", "LR_HEADS_A",
                        "LR_LORA_A", "WEIGHT_DECAY", "HFLIP_P", "USE_PHOTO_AUG", "PHOTO_AUG", "ASL_GAMMA_POS",
                        "ASL_GAMMA_NEG", "ASL_CLIP", "LAMBDA_CMP", "LAMBDA_SEP", "SEP_MARGIN", "FOLD_SEED", "N_FOLDS",
                        "STRATIFIED_WITHHOLD", "NUM_CLASSES"]
PHASE_A_IMPORTS = []

def _phase_a_source_dirs(arm):
    if arm == "full":
        root = PHASE_A_SOURCE
    else:
        m_ = re.fullmatch(r"lco_fold(\d+)", arm)
        assert m_, f"unknown arm {arm}"
        root = os.path.join(PHASE_A_SOURCE, "LCO", f"fold{m_.group(1)}")
    return os.path.join(root, "Model_Checkpoints"), os.path.join(root, "logs")

def _check_source_config():
    p = os.path.join(PHASE_A_SOURCE, "reproducibility", "config.json")
    if not os.path.exists(p):
        return f"no {p}"
    src = json.load(open(p))
    mine = _clean({k: globals()[k] for k in PHASE_A_KEYS_CHECKED})
    diff = [k for k in PHASE_A_KEYS_CHECKED if src.get(k) != mine[k]]
    if diff:
        raise RuntimeError(f"PHASE_A_SOURCE differs in Phase A settings {diff}: set PHASE_A_SOURCE = '' to train Phase A here")
    prov = os.path.join(PHASE_A_SOURCE, "reproducibility", "data_provenance.json")
    if os.path.exists(prov) and json.load(open(prov)).get("rfmid_cache_fingerprint") != CACHE_FINGERPRINT:
        raise RuntimeError("PHASE_A_SOURCE was trained on a different RFMiD cache (fingerprints differ)")
    return None

def import_phase_a(seed, arm, dirs):
    if not PHASE_A_SOURCE:
        return
    if glob.glob(os.path.join(dirs["phaseA"], "*.pt")):
        return                                                  # already here (imported or trained)
    ck_root, log_root = _phase_a_source_dirs(arm)
    src = os.path.join(ck_root, f"SEED = {seed}", "phaseA")
    last = os.path.join(src, f"epoch_{NUM_EPOCHS_PHASE_A - 1}.pt")
    if not (os.path.exists(last) and os.path.exists(os.path.join(src, "best.pt"))):
        print(f"[phaseA import] seed {seed} ({arm}): not finished in {src} -- training Phase A here")
        return
    problem = _check_source_config()
    if problem:
        print(f"[phaseA import] seed {seed} ({arm}): {problem} -- training Phase A here")
        return
    ck = torch.load(last, map_location="cpu", weights_only=False)
    assert ck["epoch"] == NUM_EPOCHS_PHASE_A and len(ck["history"]["val_map"]) == NUM_EPOCHS_PHASE_A
    assert ck["config"]["disease_cols"] == DISEASE_COLS and ck["config"]["kp"] == KP
    assert (ck["config"]["lora_rank"], ck["config"]["lora_alpha"]) == (LORA_RANK, LORA_ALPHA)
    for f in ("best.pt", os.path.basename(last)):
        shutil.copy(os.path.join(src, f), os.path.join(dirs["phaseA"], f))
    n = 0
    for p in sorted(glob.glob(os.path.join(log_root, "training_journal*.jsonl"))):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("arm") == arm and int(r.get("seed", -1)) == int(seed) and r.get("phase") == "A":
                r["imported_from"] = os.path.abspath(p)
                with open(JOURNAL_PATH, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
                n += 1
    rec = {"seed": seed, "arm": arm, "source": os.path.abspath(src), "best_sha256": sha256_file(os.path.join(src, "best.pt")),
           "last_sha256": sha256_file(last), "journal_records": n}
    PHASE_A_IMPORTS.append(rec)
    with open(os.path.join(REPRO_DIR, "phase_a_imports.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[phaseA import] seed {seed} ({arm}): copied best.pt + epoch {NUM_EPOCHS_PHASE_A} checkpoint "
          f"and {n} journal records from {src}")

print(f"[phaseA import] source: {PHASE_A_SOURCE or '(none: Phase A trained here)'}")


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


_CFG_NAMES = ["RUN_NAME", "ABLATION", "WITHHOLDING_MODE", "PHASE_A_SOURCE", "SEEDS", "SEEDS_THIS_RUN", "IMG_SIZE", "PATCH_SIZE", "FOV_THRESH", "TIER_EDGES", "KP",
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


RESULTS = {}
for s in SEEDS:
    p = os.path.join(JSON_DIR, f"results_seed{s}.json")
    if os.path.exists(p):
        RESULTS[s] = json.load(open(p))
        print(f"[run] seed {s}: already evaluated -- loaded")

for s in SEEDS_THIS_RUN:
    if s in RESULTS:
        continue
    print(f"\n{'='*84}\n[run] seed {s} | {(time.time()-SESSION_T0)/3600:.2f} h into this session\n{'='*84}")
    meta = train_one_seed(s)
    res = evaluate_seed(model, s)
    res["train_meta"] = {k: v for k, v in meta.items() if not k.startswith("history")}
    res["history_A"], res["history_B"] = meta.get("history_A"), meta.get("history_B")
    write_json(res, os.path.join(JSON_DIR, f"results_seed{s}.json"))
    RESULTS[s] = _clean(res)
print(f"\n[run] {len(RESULTS)}/{len(SEEDS)} seeds evaluated: {sorted(RESULTS)}")


# In[ ]:


def stack(key, source=None):
    source = RESULTS if source is None else source
    out = []
    for s in sorted(source):
        d = source[s].get(key)
        if d:
            df = pd.DataFrame(d if isinstance(d, list) else [d]); df.insert(0, "seed", s); out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

H = pd.DataFrame([RESULTS[s]["headline"] for s in sorted(RESULTS)])
H.to_csv(os.path.join(RESULTS_DIR, "rfmid_per_seed.csv"), index=False)
for key, name in (("per_class", "rfmid_per_class"), ("per_tier", "rfmid_per_tier"), ("fold_unknown", "rfmid_fold_unknown"),
                  ("detector_ci", "rfmid_detector_ci"), ("comparisons", "rfmid_detector_comparisons")):
    df = stack(key)
    if len(df):
        df.to_csv(os.path.join(RESULTS_DIR, f"{name}.csv"), index=False)
print(f"RFMiD test | mean +/- sd over {len(H)} seeds")
for k in ("threshold", "known_map", "known_macro_auroc", "macro_f1", "micro_f1", "set_size", "ece", "uhat_auroc",
          "maxlogit_auroc", "knn_auroc", "ensemble_auroc", "ensemble_ml_auroc", "ens_mk_auroc", "ens3_auroc",
          "ens3_ref_auroc", "ens3_auprc", "ens3_fpr95", "op_sensitivity", "op_specificity", "mixed_jaccard", "J_mixed"):
    v = pd.to_numeric(H[k], errors="coerce")
    print(f"  {k:20} {v.mean():.4f} +/- {v.std(ddof=1) if len(v) > 1 else float('nan'):.4f}")
_c = stack("comparisons")
for (a, b), g in _c.groupby(["a", "b"], sort=False):
    print(f"  {a:>6} vs {b:11} | mean delta {g.delta_auroc.mean():+.4f} | BH-significant in {int(g.reject_bh05.sum())}/{len(g)} seeds")

J = pd.DataFrame([{"seed": r["seed"], "phase": r["phase"], "epoch": r["epoch"] + 1, **r["metrics"]} for r in journal_load()
                  if r.get("arm") == "full"])
if len(J):
    J.to_csv(os.path.join(LOG_DIR, "training_curves.csv"), index=False)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for s, g in J.groupby("seed"):
        a, b = g[g.phase == "A"], g[g.phase == "B"]
        ax[0].plot(a.epoch, a.val_map, label=f"seed {s}")
        ax[1].plot(b.epoch, b.val_known_map, label=f"seed {s}")
        ax[2].plot(b.epoch, b[SELECTION_KEY], label=f"seed {s}")
    for a_, t in zip(ax, ("Phase A: validation mAP", "Phase B: validation known mAP", f"Phase B: {SELECTION_KEY}")):
        a_.set_title(t, fontsize=9); a_.set_xlabel("epoch"); a_.grid(alpha=0.3); a_.legend(fontsize=7, frameon=False)
    ax[1].axvline(PHASE_B_BURN_IN, color="black", ls=":", lw=1)
    plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=150, bbox_inches="tight"); plt.show()


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

EXT_HEAD, EXT_UNK, EXT_TERM, EXT_CI = [], [], [], []
if RUN_EXTERNAL:
    assert not LCO_MASKED, "external validation uses the full 45-class models"
    _seeds = [s for s in sorted(RESULTS) if os.path.exists(os.path.join(seed_dirs(s)["phaseB"], "selected.pt"))]
    print(f"[ext] seeds {_seeds} on {EXT_DATASETS}")
    for s in _seeds:
        t0 = time.time()
        m = load_selected(s)
        thr = float(RESULTS[s]["threshold"]["global"])
        t_op = float(np.nanmedian(pd.DataFrame(RESULTS[s]["fold_unknown"])["t_ens3"].astype(float)))
        Ftr = collect_pass(TRAIN_EVAL_LOADER_FULL, m)["feat"]
        pv = collect_pass(VAL_LOADER_FULL, m)
        ref = {"uhat": pv["uhat"], "maxlogit": pv["maxlogit"], "knn": knn_score(Ftr, pv["feat"])}
        for ds in EXT_DATASETS:
            L, _ = EXT_CACHE[ds]
            P = collect_pass(DataLoader(ExternalPNG(L["image_id"], os.path.join(DS_SPECS[ds]["cache"], "images")),
                                        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS), m)
            assert list(P["ids"]) == list(L["image_id"]), "row order drifted"
            P["knn"] = knn_score(Ftr, P["feat"])
            for v in [v for v in EXT_VARIANTS if v != "no_omit_terms" or ds == "ODIR"]:
                if v == "no_debatable" and int(L["has_debatable"].sum()) == 0:
                    continue
                h, u, t, ci, S = ext_evaluate(ds, s, L, P, thr, ref, v, t_op)
                EXT_HEAD.append(h); EXT_UNK += u; EXT_TERM += t
                if ci:
                    EXT_CI.append({"dataset": ds, "seed": s, **{f"{k}_lo": a for k, (a, b) in ci.items()},
                                   **{f"{k}_hi": b for k, (a, b) in ci.items()}})
                if v == "primary":
                    pd.DataFrame({"image_id": L["image_id"].values, "patient_id": L["patient_id"].values,
                                  "is_unknown": L["is_unknown"].values, "is_diseased": L["is_diseased"].values,
                                  **{d: S[d] for d in DETECTORS},
                                  **{f"p_{c}": P["probs"][:, j] for j, c in enumerate(DISEASE_COLS)}}).to_csv(
                        os.path.join(EXT_PRED_DIR, f"{ds}_seed{s}.csv"), index=False)
            hp = [h for h in EXT_HEAD if h["dataset"] == ds and h["seed"] == s and h["variant"] == "primary"][0]
            print(f"[ext {ds} s{s}] known mAP {hp['known_map']:.4f} | known AUROC {hp['known_macro_auroc']:.4f} | "
                  f"macro-F1 {hp['macro_f1']:.4f} | unknown vs known disease: "
                  + " | ".join(f"{d}={hp[f'{d}_auroc']:.4f}" for d in ("uhat", "maxlogit", "knn", "ens_mk", "ens3", "ens3_ref")))
        print(f"[ext] seed {s} done in {(time.time()-t0)/60:.1f} min")

    EH, EU, ET, EC = pd.DataFrame(EXT_HEAD), pd.DataFrame(EXT_UNK), pd.DataFrame(EXT_TERM), pd.DataFrame(EXT_CI)
    EH.to_csv(os.path.join(EXT_RES_DIR, "external_per_seed.csv"), index=False)
    EU.to_csv(os.path.join(EXT_RES_DIR, "external_unknown.csv"), index=False)
    ET.to_csv(os.path.join(EXT_RES_DIR, "external_unknown_by_term.csv"), index=False)
    EC.to_csv(os.path.join(EXT_RES_DIR, "external_ci.csv"), index=False)
    _num = [c for c in EH.columns if c not in ("dataset", "seed", "variant")]
    ES = EH.groupby(["dataset", "variant"], sort=False)[_num].agg(["mean", "std"])
    ES.columns = [f"{a}_{b}" for a, b in ES.columns]
    ES.reset_index().to_csv(os.path.join(EXT_RES_DIR, "external_summary.csv"), index=False)
    ET.groupby(["dataset", "variant", "unknown_term", "n_pos"])[[f"{d}_auroc" for d in DETECTORS]].mean() \
      .reset_index().to_csv(os.path.join(EXT_RES_DIR, "external_unknown_by_term_summary.csv"), index=False)
    for ds in EXT_DATASETS:
        for v, g in EH[EH.dataset == ds].groupby("variant", sort=False):
            print(f"  [{ds}] {v:14} n={int(g.n_images.mean())} | ens3 {g.ens3_auroc.mean():.4f} | ens_mk "
                  f"{g.ens_mk_auroc.mean():.4f} | knn {g.knn_auroc.mean():.4f} | known AUROC {g.known_macro_auroc.mean():.4f}")
    if len(EC):
        print("  patient-clustered 95% CIs (diseased images):")
        for _, r in EC.iterrows():
            print(f"    {r.dataset:6} s{int(r.seed)} | ens3 [{r.ens3_auroc_ci_lo:.3f}, {r.ens3_auroc_ci_hi:.3f}] | "
                  f"ens3 - ens_mk [{r.ens3_minus_ens_mk_ci_lo:+.3f}, {r.ens3_minus_ens_mk_ci_hi:+.3f}]")
    print(f"[ext] tables in {EXT_RES_DIR}")


# In[ ]:


LCO_ROOT = os.path.join(OUTPUT_ROOT, "LCO")
_LCO_SAVED = None
_LCO_NAMES = ("train_df", "val_df", "train_ds", "train_eval_ds", "val_ds", "train_loader", "train_eval_loader",
              "val_loader", "VAL_LABELS_NP", "VAL_DISEASED", "TIER_TO_INDICES", "LCO_MASKED", "FULL_MASK",
              "POS_WEIGHT_U", "POS_WEIGHT_U_T", "U_TRAIN_PREVALENCE", "CKPT_ROOT_OUT", "LOG_DIR", "JOURNAL_PATH",
              "JOURNAL_GLOB", "RUN_NAME")

_rm = []
for _f in range(1, N_FOLDS + 1):
    _wn = [DISEASE_COLS[i] for i in sorted(FOLDS[_f - 1])]
    for _sp, _d in (("train", TRAIN_DF_FULL), ("val", VAL_DF_FULL)):
        for _id in _d.loc[_d[_wn].sum(axis=1) > 0, "ID"]:
            _rm.append({"fold": _f, "split": _sp, "ID": _id})
pd.DataFrame(_rm).to_csv(os.path.join(REPRO_DIR, "lco_removed_images.csv"), index=False)

def lco_configure(fold):
    """Rebind every global the training driver reads to the fold-f leave-classes-out setting."""
    global _LCO_SAVED, train_df, val_df, train_ds, train_eval_ds, val_ds, train_loader, train_eval_loader, val_loader
    global VAL_LABELS_NP, VAL_DISEASED, TIER_TO_INDICES, LCO_MASKED, FULL_MASK, POS_WEIGHT_U, POS_WEIGHT_U_T
    global U_TRAIN_PREVALENCE, CKPT_ROOT_OUT, LOG_DIR, JOURNAL_PATH, JOURNAL_GLOB, RUN_NAME
    assert _LCO_SAVED is None, "lco_configure called twice without lco_restore"
    _LCO_SAVED = {n: globals()[n] for n in _LCO_NAMES}
    w = sorted(FOLDS[fold - 1]); wn = [DISEASE_COLS[i] for i in w]
    train_df = TRAIN_DF_FULL[TRAIN_DF_FULL[wn].sum(axis=1).values == 0].reset_index(drop=True)
    val_df = VAL_DF_FULL[VAL_DF_FULL[wn].sum(axis=1).values == 0].reset_index(drop=True)
    train_ds, train_eval_ds, val_ds, train_loader, train_eval_loader, val_loader = make_loaders(train_df, val_df)
    VAL_LABELS_NP = val_ds.labels
    VAL_DISEASED = VAL_LABELS_NP.sum(axis=1) > 0
    LCO_MASKED = set(w)
    FULL_MASK = make_active_mask([])
    TIER_TO_INDICES = {t: [i for i in TIER_TO_INDICES_FULL[t] if i not in w] for t in TIERS}
    for t, n in STRATIFIED_WITHHOLD.items():
        assert len(TIER_TO_INDICES[t]) >= n, f"tier {t} has only {len(TIER_TO_INDICES[t])} classes left"
    POS_WEIGHT_U, U_TRAIN_PREVALENCE = estimate_unknown_pos_weight(train_df)
    POS_WEIGHT_U_T = torch.tensor(POS_WEIGHT_U, device=DEVICE)
    froot = os.path.join(LCO_ROOT, f"fold{fold}")
    CKPT_ROOT_OUT, LOG_DIR = os.path.join(froot, "Model_Checkpoints"), os.path.join(froot, "logs")
    for d in (froot, CKPT_ROOT_OUT, LOG_DIR):
        os.makedirs(d, exist_ok=True)
    JOURNAL_PATH = os.path.join(LOG_DIR, f"training_journal_{'_'.join(map(str, LCO_SEEDS))}.jsonl")
    JOURNAL_GLOB = os.path.join(LOG_DIR, "training_journal*.jsonl")
    RUN_NAME = f"RetinaOWL_LCO_fold{fold}"
    print(f"[lco f{fold}] never trained: {wn}")
    print(f"[lco f{fold}] train {len(train_df)}/{len(TRAIN_DF_FULL)} | val {len(val_df)}/{len(VAL_DF_FULL)} | active "
          f"classes {int(FULL_MASK.sum())} | unknown pos_weight {POS_WEIGHT_U:.3f} | output {froot}")
    return froot

def lco_restore():
    global _LCO_SAVED
    if _LCO_SAVED is None:
        return
    for n, v in _LCO_SAVED.items():
        globals()[n] = v
    _LCO_SAVED = None
    assert not LCO_MASKED and bool(FULL_MASK.all()), "leave-classes-out state not restored"

def evaluate_lco(m, fold, seed):
    """Test split; identical images and targets to the withheld-class folds; the fold's classes were never trained."""
    w = FOLDS[fold - 1]
    active = [c for c in range(NUM_CLASSES) if c not in w]
    m.eval()
    P = collect_pass(TEST_LOADER_FULL, m, w)
    Y = P["labels"]; dis = Y.sum(1) > 0; u = unknown_target_np(Y, w)
    Ftr = collect_pass(train_eval_loader, m)["feat"]                        # the model's own training images
    pv = collect_pass(VAL_LOADER_FULL, m, w)
    kv = knn_score(Ftr, pv["feat"])
    ref = {"uhat": pv["uhat"], "maxlogit": pv["maxlogit"], "knn": kv}
    S = detector_scores(P["uhat"], P["energy"], P["maxlogit"], knn_score(Ftr, P["feat"]), ref=ref)
    Sv = detector_scores(pv["uhat"], pv["energy"], pv["maxlogit"], kv, ref=ref)
    calv = np.array([bool(HALF_IS_CALIB[i]) for i in pv["ids"]])
    # decision threshold of this model: the global macro-F1 rule on the calibration half, active classes only
    thr, _ = fit_global_threshold(pv["probs"][calv][:, active], pv["labels"][calv][:, active])
    ps = prediction_sets(P["probs"], np.full(NUM_CLASSES, thr), cols_idx=active)
    pm = prf_metrics(ps, Y, active)
    uv = unknown_target_np(pv["labels"], w)
    t_e = conformal_scalar_threshold(Sv["ens3_ref"][calv], uv[calv])
    flag = S["ens3_ref"] >= t_e
    dm = (u > 0.5) & (Y[:, active].sum(1) > 0)
    out = {"fold": fold, "seed": seed, "withheld": [DISEASE_COLS[i] for i in sorted(w)], "n_train": int(len(train_df)),
           "n_images": int(len(Y)), "n_unknown": int(u[dis].sum()), "unknown_prevalence": float(u[dis].mean()),
           "threshold": float(thr), "macro_f1": pm["macro_f1"], "micro_f1": pm["micro_f1"],
           "set_size": float(ps[:, active].sum(1).mean()),
           "known_map": tiered_evaluation(Y[:, active], P["probs"][:, active], [DISEASE_COLS[i] for i in active],
                                          TIER_LABELS)["primary_macro_map"],
           "known_macro_auroc": macro_auroc_auprc(P["probs"], Y, active)["macro_auroc"],
           "ece": expected_calibration_error(P["probs"][:, active], Y[:, active]), "t_ens3": t_e,
           "op_sensitivity": float(flag[(u > 0.5) & dis].mean()) if ((u > 0.5) & dis).any() else np.nan,
           "op_specificity": float((~flag)[(u < 0.5) & dis].mean()),
           "n_mixed": int(dm.sum()), "mixed_jaccard": np.nan, "mixed_exact": np.nan, "J_mixed": np.nan}
    if dm.sum() >= 5:
        smx = set_metrics(ps[dm], Y[dm], cols_idx=active)
        out.update({"mixed_jaccard": smx["jaccard"], "mixed_exact": smx["exact_match"],
                    "J_mixed": float((smx["exact_vec"] * flag[dm]).mean())})
    for n in DETECTORS:
        b = unknown_metric_block(u[dis], S[n][dis])
        out[f"{n}_auroc"] = b["auroc"]
        if n == "ens3":
            out.update({f"ens3_{k}": b[k] for k in ("auprc", "fpr95", "det_err", "aurc")})
            out["ens3_auroc_all"] = safe_auroc(u, S[n])
    pd.DataFrame({"ID": P["ids"], "unknown": u.astype(int), "diseased": dis.astype(int), **{n: S[n] for n in DETECTORS}}) \
      .to_csv(os.path.join(PRED_DIR, f"lco_fold{fold}_test_scores_seed{seed}.csv"), index=False)
    return out

print(f"[lco] machinery ready | outputs under {LCO_ROOT} | removed image IDs -> reproducibility/lco_removed_images.csv")


# In[ ]:


LCO_RESULTS, LCO_FAILURES = [], []
if RUN_LCO:
    for _f in LCO_FOLDS:
        for _s in LCO_SEEDS:
            _out = os.path.join(LCO_ROOT, f"fold{_f}", f"results_lco_fold{_f}_seed{_s}.json")
            if os.path.exists(_out):
                LCO_RESULTS.append(json.load(open(_out)))
                print(f"[lco] fold {_f} seed {_s}: already evaluated -- loaded")
                continue
            print(f"\n{'='*84}\n[lco] fold {_f} | seed {_s} | {(time.time()-SESSION_T0)/3600:.2f} h into this session\n{'='*84}")
            try:
                lco_configure(_f)
                _meta = train_one_seed(_s, arm=f"lco_fold{_f}")
                _r = evaluate_lco(model, _f, _s)
                _r.update({"selected_epoch": _meta["selected_epoch"], "phase_a_baseline": _meta["phase_a_baseline"],
                           "selected_ckpt_sha256": _meta.get("selected_ckpt_sha256")})
            except Exception as _e:
                # e.g. no epoch met the selection rule: reported, never relaxed; the other folds continue
                import traceback; traceback.print_exc()
                LCO_FAILURES.append({"fold": _f, "seed": _s, "error": f"{type(_e).__name__}: {_e}"})
                continue
            finally:
                lco_restore()
            write_json(_r, _out)
            LCO_RESULTS.append(_clean(_r))
            print(f"[lco] fold {_f} seed {_s} | known mAP {_r['known_map']:.4f} | " +
                  " | ".join(f"{n}={_r[f'{n}_auroc']:.4f}" for n in ("uhat", "maxlogit", "knn", "ensemble_ml", "ens_mk", "ens3")))
else:
    print("[lco] RUN_LCO=False -- skipped")

if LCO_RESULTS:
    LR = pd.DataFrame(LCO_RESULTS)
    LR.to_csv(os.path.join(RESULTS_DIR, "lco_results.csv"), index=False)
    _fp = stack("fold_unknown")
    dets = ("uhat", "maxlogit", "knn", "ensemble_ml", "ens_mk", "ens3")
    print("\n" + "=" * 118)
    print("  LEAVE-CLASSES-OUT vs WITHHELD-CLASS FOLDS | RFMiD test | unknown vs known disease AUROC | same images and targets")
    print("=" * 118)
    print(f"  {'fold':>4} {'never-trained classes':40} | " + " | ".join(f"{d:>15}" for d in dets) + " | known mAP")
    rows = []
    for f, g in LR.groupby("fold"):
        fpm = _fp[_fp.fold == f] if len(_fp) else pd.DataFrame()
        cells = []
        for d in dets:
            lco = g[f"{d}_auroc"].mean()
            fp = fpm[f"{d}_dis_auroc"].mean() if len(fpm) else np.nan
            cells.append(f"{lco:.3f} ({fp:.3f})"); rows.append({"fold": f, "detector": d, "lco": lco, "withheld_fold": fp})
        print(f"  {int(f):4d} {', '.join(g.withheld.iloc[0])[:40]:40} | " + " | ".join(f"{c:>15}" for c in cells)
              + f" | {g.known_map.mean():.4f}")
    CMP = pd.DataFrame(rows)
    CMP.to_csv(os.path.join(RESULTS_DIR, "lco_vs_withheld_folds.csv"), index=False)
    _m = CMP.groupby("detector", sort=False)[["lco", "withheld_fold"]].mean()
    print("  mean" + " " * 42 + "| " + " | ".join(f"{_m.loc[d, 'lco']:.3f} ({_m.loc[d, 'withheld_fold']:.3f})".rjust(15)
                                                for d in dets))
    print("  cell = leave-classes-out (withheld-class folds of the full models, same fold)")
for _x in LCO_FAILURES:
    print(f"[lco] FAILED fold {_x['fold']} seed {_x['seed']}: {_x['error']}")


# In[ ]:


FINAL_ROWS = [
    ("n_images", "Images evaluated"), ("n_unknown", "Unknown-positive images"),
    ("unknown_prevalence", "Unknown prevalence among diseased (AUPRC chance)"),
    ("threshold", "Fitted decision threshold (global, macro-F1)"),
    ("known_map", "Known mAP"), ("known_macro_auroc", "Known macro-AUROC"), ("macro_f1", "Macro-F1"),
    ("micro_f1", "Micro-F1"), ("set_size", "Mean prediction-set size"), ("ece", "ECE (15 bins)"),
    ("ens3_auroc", "Unknown AUROC (ens3, vs known disease)"), ("ens3_auprc", "Unknown AUPRC (ens3)"),
    ("ens3_fpr95", "FPR at 95% TPR (ens3)"), ("ens3_aurc", "AURC (ens3, lower is better)"),
    ("ens3_auroc_all", "Unknown AUROC (ens3, vs all images)"),
    ("t_ens3", "Unknown operating threshold t_ens3 (ens3_ref)"),
    ("op_sensitivity", "Unknown sensitivity at t_ens3"), ("op_specificity", "Known-disease specificity at t_ens3"),
    ("mixed_jaccard", "Mixed Jaccard"), ("J_mixed", "J_mixed (exact known set and flagged)"),
    ("uhat_auroc", "Unknown AUROC: u_hat"), ("maxlogit_auroc", "Unknown AUROC: max-logit"),
    ("energy_auroc", "Unknown AUROC: energy"), ("knn_auroc", "Unknown AUROC: kNN (k=50)"),
    ("ensemble_auroc", "Unknown AUROC: u_hat + energy"), ("ensemble_ml_auroc", "Unknown AUROC: u_hat + max-logit"),
    ("ens_mk_auroc", "Unknown AUROC: max-logit + kNN (no u_hat)"), ("ens3_ref_auroc", "Unknown AUROC: ens3_ref")]

SETTINGS = {}
if RESULTS:
    _h = pd.DataFrame([RESULTS[s]["headline"] for s in sorted(RESULTS)]).rename(columns={"n_unknown_dis": "n_unknown"})
    SETTINGS["RFMiD (withheld-class folds)"] = _h
if LCO_RESULTS:
    SETTINGS["RFMiD (leave-classes-out)"] = pd.DataFrame(LCO_RESULTS)
if EXT_HEAD:
    _eh = pd.DataFrame(EXT_HEAD)
    for ds, lab in (("JSIEC", "JSIEC"), ("ODIR", "ODIR-5K")):
        g = _eh[(_eh.dataset == ds) & (_eh.variant == "primary")]
        if len(g):
            SETTINGS[lab] = g

_rows, _fmt = [], []
for key, label in FINAL_ROWS:
    r, rf = {"metric": label, "key": key}, {"metric": label}
    for st, df in SETTINGS.items():
        v = pd.to_numeric(df[key], errors="coerce") if key in df else pd.Series(dtype=float)
        v = v[np.isfinite(v)]
        mu, sd = (float(v.mean()) if len(v) else np.nan), (float(v.std(ddof=1)) if len(v) > 1 else np.nan)
        r.update({f"{st} | mean": mu, f"{st} | sd": sd, f"{st} | n": int(len(v))})
        if not len(v):
            rf[st] = "n/a"
        elif key in ("n_images", "n_unknown"):
            rf[st] = f"{mu:.0f}" + (f" ± {sd:.0f}" if np.isfinite(sd) and sd > 0 else "")
        else:
            rf[st] = f"{mu:.4f}" + (f" ± {sd:.4f}" if np.isfinite(sd) else "")
    _rows.append(r); _fmt.append(rf)
FINAL = pd.DataFrame(_rows)
FINAL_FMT = pd.DataFrame(_fmt).set_index("metric")
FINAL.to_csv(os.path.join(RESULTS_DIR, "FINAL_METRICS_ALL_DATASETS.csv"), index=False)
FINAL_FMT.to_csv(os.path.join(RESULTS_DIR, "FINAL_METRICS_ALL_DATASETS_formatted.csv"))
with open(os.path.join(RESULTS_DIR, "FINAL_METRICS_ALL_DATASETS.md"), "w") as fh:
    fh.write("| metric | " + " | ".join(FINAL_FMT.columns) + " |\n|---|" + "---|" * len(FINAL_FMT.columns) + "\n")
    for k, row in FINAL_FMT.iterrows():
        fh.write(f"| {k} | " + " | ".join(row.values) + " |\n")

pd.set_option("display.max_colwidth", 60, "display.width", 250)
print("FINAL METRICS | mean ± sd | " + " | ".join(f"{st}: n={len(df)} {'folds' if 'leave' in st else 'seeds'}"
                                                 for st, df in SETTINGS.items()))
print(FINAL_FMT.to_string())
print(f"\n-> {os.path.join(RESULTS_DIR, 'FINAL_METRICS_ALL_DATASETS.csv')} (+ _formatted.csv, .md)")

_dets = ["uhat", "maxlogit", "knn", "ensemble_ml", "ens_mk", "ens3"]
if SETTINGS:
    fig, ax = plt.subplots(figsize=(11, 4))
    wbar = 0.8 / len(SETTINGS)
    for k, (st, df) in enumerate(SETTINGS.items()):
        mu = [pd.to_numeric(df[f"{d}_auroc"], errors="coerce").mean() for d in _dets]
        sd = [pd.to_numeric(df[f"{d}_auroc"], errors="coerce").std(ddof=1) for d in _dets]
        ax.bar(np.arange(len(_dets)) + k * wbar, mu, wbar * 0.92, yerr=sd, capsize=2, label=st)
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.set_xticks(np.arange(len(_dets)) + 0.4 - wbar / 2)
    ax.set_xticklabels(["u_hat", "max-logit", "kNN", "u_hat+max-logit", "max-logit+kNN", "ens3"])
    ax.set_ylabel("unknown AUROC (vs known disease)"); ax.set_ylim(0.3, 1.0); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "unknown_auroc_all_settings.png"), dpi=200, bbox_inches="tight")
    plt.show()


# In[ ]:


_thr = []
for s in sorted(RESULTS):
    _thr.append({"setting": "rfmid_full", "seed": s, "fold": None, "decision_threshold": RESULTS[s]["threshold"]["global"],
                 "calibration_macro_f1": RESULTS[s]["threshold"].get("calibration_macro_f1"), "t_ens3": None})
    for r in RESULTS[s]["fold_unknown"]:
        _thr.append({"setting": "rfmid_full", "seed": s, "fold": r["fold"], "decision_threshold": None,
                     "calibration_macro_f1": None, "t_ens3": r["t_ens3"]})
for r in LCO_RESULTS:
    _thr.append({"setting": "lco", "seed": r["seed"], "fold": r["fold"], "decision_threshold": r["threshold"],
                 "calibration_macro_f1": None, "t_ens3": r["t_ens3"]})
pd.DataFrame(_thr).to_csv(os.path.join(REPRO_DIR, "fitted_thresholds.csv"), index=False)

_sel = []
for p in sorted(glob.glob(os.path.join(OUTPUT_ROOT, "**", "selection_*.json"), recursive=True)):
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
torch {ENV_MANIFEST['packages'].get('torch')}, transformers {ENV_MANIFEST['packages'].get('transformers')}).

## Reproduce
1. Install the exact package versions in `requirements_freeze.txt` (the key ones are in `environment.json`).
2. Place RFMiD, the DINOv2 ViT-S/14 weights, JSIEC, ODIR-5K and the two mapping CSVs as in `config.json -> paths`
   (check the hashes in `data_provenance.json`), or point the paths in section 0 of the notebook at them.
3. Run the notebook top to bottom. The configuration in section 0 must match `config.json`; the seeds are
   SEEDS={SEEDS}, FOLD_SEED={FOLD_SEED}, CALIB_SPLIT_SEED={CALIB_SPLIT_SEED}, EVAL_SEED_BASE={EVAL_SEED_BASE},
   POS_WEIGHT_U_SEED={POS_WEIGHT_U_SEED}, LCO_SEEDS={LCO_SEEDS}.
4. The held-out label sets (`heldout_label_sets.csv`), the calibration split (`val_calibration_split.csv`) and the
   leave-classes-out removals (`lco_removed_images.csv`) are regenerated from those seeds and must match the files here.
5. Selected checkpoints are listed with their SHA-256 in `checkpoint_selection.csv`; to re-evaluate without training,
   place them at the same relative paths and rerun (training is skipped when `selected.pt` and `train_meta.json` exist).

## Notes
* Mixed precision and non-deterministic GPU kernels mean a re-run reproduces the numbers closely, not bit-for-bit.
* Nothing is fitted on RFMiD test, JSIEC or ODIR images: thresholds come from the RFMiD validation calibration half,
  checkpoints are selected on RFMiD validation only.
* Headline table: `results/FINAL_METRICS_ALL_DATASETS.csv`.
"""
open(os.path.join(REPRO_DIR, "README.md"), "w").write(README)

_man_path = os.path.join(REPRO_DIR, "output_manifest.csv")
_man = []
for p in sorted(Path(OUTPUT_ROOT).rglob("*")):
    if p.is_file() and str(p) != _man_path and not p.name.endswith(".tmp"):
        _man.append({"path": str(p.relative_to(OUTPUT_ROOT)), "bytes": p.stat().st_size, "sha256": sha256_file(str(p))})
pd.DataFrame(_man).to_csv(_man_path, index=False)
print(f"[repro] {len(_man)} files hashed -> reproducibility/output_manifest.csv")
print("[repro] contents of reproducibility/:")
for f in sorted(os.listdir(REPRO_DIR)):
    print(f"    {f:32} {os.path.getsize(os.path.join(REPRO_DIR, f)) / 1024:8.1f} KB")
print(f"\n[done] {(time.time() - SESSION_T0) / 3600:.2f} h this session")

