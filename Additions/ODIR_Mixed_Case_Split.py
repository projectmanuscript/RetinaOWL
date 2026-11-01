#!/usr/bin/env python
# coding: utf-8

# In[ ]:


OUTPUT_ROOT = "./Output_ODIR_Case_Split"
ODIR_LABELS = "../Preprocessing_Cache/ODIR/labels.csv"     # the ODIR cache the training notebooks built

MODELS = {                                    # display name -> training output folder (read only)
    "RetinaOWL":        "../Model/Output",
    "UIOS":             "../Baselines/UIOS/Output",
    "RETFound-MAE":     "../Baselines/RetFound/Output",
    "RETFound-DINOv2":  "../Baselines/RetFound_DINO/Output",
    "ML-Decoder":       "../Baselines/ML_Decoder/Output",
    "ResNet-50":        "../Baselines/ResNet50_ASL/Output",
    "Swin-T":           "../Baselines/SwinT_ASL/Output",
    # "A1 no residual branch": "../Ablations/No_Residual_Branch/Output",
}
REFERENCE = "RetinaOWL"                       # paired differences are reported as REFERENCE - model
SEEDS = [42, 43, 44, 45, 46]
VARIANTS = ["primary", "no_debatable", "no_omit_terms"]
SUBSETS = ["all", "mixed", "unknown_only"]
DETECTORS = ["ens3", "uhat", "maxlogit", "knn"]          # all-NaN columns (u_hat of the closed-set baselines) are skipped
OP_TPR = 0.90                                 # FPR is reported at this TPR
MIN_CASES = 20                                # subsets with fewer positives are flagged (*) as too small to interpret
N_BOOT, BOOT_SEED = 2000, 7                   # patient-clustered bootstrap; same draws for every model
CHECK_TOL = 1e-6                              # max |recomputed - reported| (scores are read back bit-exactly)


# In[ ]:


import os, sys, json, hashlib, platform, time
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
import matplotlib.pyplot as plt

RES_DIR, FIG_DIR, REPRO_DIR = (os.path.join(OUTPUT_ROOT, d) for d in ("results", "figures", "reproducibility"))
for _d in (RES_DIR, FIG_DIR, REPRO_DIR):
    os.makedirs(_d, exist_ok=True)
ENV = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
       "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
       "started": time.strftime("%Y-%m-%d %H:%M:%S")}
print("[env]", ENV)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_exact(path, **kw):
    """Scores are read back bit-exactly (pandas' default parser can be off by one unit in the last place, which breaks
    the many exact ties of a rank ensemble differently from the training run)."""
    return pd.read_csv(path, float_precision="round_trip", **kw)


# In[ ]:


ERRORS, WARNS, INPUT_FILES = [], [], []
assert REFERENCE in MODELS, f"REFERENCE={REFERENCE!r} is not in MODELS"
assert set(VARIANTS) <= {"primary", "no_debatable", "no_omit_terms"} and "primary" in VARIANTS

if not os.path.exists(ODIR_LABELS):
    raise FileNotFoundError(f"ODIR_LABELS: {os.path.abspath(ODIR_LABELS)} does not exist (the labels.csv of the ODIR cache)")
LAB = pd.read_csv(ODIR_LABELS, dtype={"image_id": str, "patient_id": str}, keep_default_na=False)
LAB_SHA = sha256(ODIR_LABELS)
INPUT_FILES.append(("labels", os.path.abspath(ODIR_LABELS)))
for c in ("has_known", "is_unknown", "is_mixed", "is_diseased", "is_normal", "has_debatable", "n_omit_terms"):
    LAB[c] = LAB[c].astype(int)
assert (LAB.is_mixed == (LAB.has_known & LAB.is_unknown)).all(), "labels.csv: is_mixed != has_known & is_unknown"
assert (LAB.is_diseased == (LAB.has_known | LAB.is_unknown)).all(), "labels.csv: is_diseased != has_known | is_unknown"

RUNS = {}
for name, folder in MODELS.items():
    paths = {"config": os.path.join(folder, "reproducibility", "config.json"),
             "provenance": os.path.join(folder, "reproducibility", "data_provenance.json"),
             "summary": os.path.join(folder, "results", "external", "external_per_seed.csv"),
             **{s: os.path.join(folder, "predictions", "external", f"ODIR_seed{s}.csv") for s in SEEDS}}
    miss = [p for p in paths.values() if not os.path.exists(p)]
    if miss:
        ERRORS += [f"{name}: missing {os.path.abspath(p)}" for p in miss]
        continue
    INPUT_FILES += [(name, os.path.abspath(p)) for p in paths.values()]
    cfg = json.load(open(paths["config"]))
    used = json.load(open(paths["provenance"])).get("odir_cache", {}).get("labels_sha256")
    if used is None:
        WARNS.append(f"{name}: data_provenance.json has no ODIR label hash; label agreement is checked row by row only")
    elif used != LAB_SHA:
        ERRORS.append(f"{name}: evaluated with an ODIR label file of SHA-256 {used[:12]}…, ODIR_LABELS is {LAB_SHA[:12]}…")
    RUNS[name] = {"folder": folder, "run_name": cfg.get("RUN_NAME"), "ens3": list(cfg["ENS3_COMPONENTS"]), "paths": paths}
for w in WARNS:
    print(f"[inputs] WARN  {w}")
if ERRORS:
    print("\n".join("[inputs] FAIL  " + e for e in ERRORS))
    raise RuntimeError(f"{len(ERRORS)} problem(s) with the inputs; fix ODIR_LABELS / MODELS in §0")

PRED = {}
for name, info in RUNS.items():
    for s in SEEDS:
        d = read_exact(info["paths"][s], dtype={"image_id": str, "patient_id": str})
        assert list(d.image_id) == list(LAB.image_id), f"{name} seed {s}: images differ from ODIR_LABELS (order or set)"
        assert (d.is_unknown.values == LAB.is_unknown.values).all() and (d.is_diseased.values == LAB.is_diseased.values).all(), \
            f"{name} seed {s}: is_unknown / is_diseased differ from ODIR_LABELS"
        PRED[(name, s)] = d
    SUM = read_exact(info["paths"]["summary"])
    info["summary"] = SUM[SUM.dataset == "ODIR"].set_index(["seed", "variant"])
    print(f"[inputs] {name:18} {str(info['run_name']):28} ens3 = {'+'.join(info['ens3']):22} | seeds {SEEDS} | labels match")
print(f"[inputs] ODIR_LABELS {os.path.abspath(ODIR_LABELS)} | SHA-256 {LAB_SHA[:16]}… | {len(LAB)} images")


# In[ ]:


VARIANT_MASK = {"primary": np.ones(len(LAB), bool),
                "no_debatable": LAB.has_debatable.values == 0,
                "no_omit_terms": LAB.n_omit_terms.values == 0}

def subsets(v):
    """Masks over the variant's images (variant order = label order restricted to the variant)."""
    L = LAB[VARIANT_MASK[v]]
    unk, known = L.is_unknown.values.astype(bool), L.has_known.values.astype(bool)
    return {"negative": known & ~unk, "all": unk, "mixed": unk & known, "unknown_only": unk & ~known,
            "diseased": unk | known, "patient": L.patient_id.values}

COMP = []
for v in VARIANTS:
    m = subsets(v)
    COMP.append({"variant": v, "n_images": int(VARIANT_MASK[v].sum()), "n_normal_excluded": int((~m["diseased"]).sum()),
                 "n_negative_known_only": int(m["negative"].sum()), "n_unknown": int(m["all"].sum()),
                 "n_mixed": int(m["mixed"].sum()), "n_unknown_only": int(m["unknown_only"].sum()),
                 "patients_unknown": int(pd.Series(m["patient"][m["all"]]).nunique()),
                 "patients_diseased": int(pd.Series(m["patient"][m["diseased"]]).nunique())})
COMP = pd.DataFrame(COMP)
COMP.to_csv(os.path.join(RES_DIR, "odir_case_counts.csv"), index=False)
print(COMP.to_string(index=False))

_t = []
for v in VARIANTS:
    L = LAB[VARIANT_MASK[v]]
    for sub, mk in (("mixed", (L.is_unknown == 1) & (L.has_known == 1)), ("unknown_only", (L.is_unknown == 1) & (L.has_known == 0))):
        terms = pd.Series([t.strip() for s in L.loc[mk, "unknown_terms"] for t in s.split(";") if t.strip()])
        for t, n in terms.value_counts().items():
            _t.append({"variant": v, "subset": sub, "unknown_term": t, "n_images": int(n)})
        if mk.any():
            kc = pd.Series([c for s in L.loc[mk, "known_classes"] for c in s.split(";") if c])
            if len(kc):
                for c, n in kc.value_counts().items():
                    _t.append({"variant": v, "subset": sub, "known_class_alongside": c, "n_images": int(n)})
TERMS = pd.DataFrame(_t)
TERMS.to_csv(os.path.join(RES_DIR, "odir_case_terms.csv"), index=False)
for sub in ("mixed", "unknown_only"):
    x = TERMS[(TERMS.variant == "primary") & (TERMS.subset == sub) & TERMS.unknown_term.notna()].head(8)
    print(f"\n[terms] primary, {sub}: " + "; ".join(f"{r.unknown_term} ({r.n_images})" for r in x.itertuples()))
if "known_class_alongside" in TERMS:
    x = TERMS[(TERMS.variant == "primary") & (TERMS.subset == "mixed") & TERMS.known_class_alongside.notna()].head(8)
    print("[terms] primary, mixed, known disease alongside: " + "; ".join(f"{r.known_class_alongside} ({r.n_images})" for r in x.itertuples()))


# In[ ]:


def _rank01(x):
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    return order / max(len(x) - 1, 1)

def safe_auroc(y, s):
    y = np.asarray(y)
    return float("nan") if y.sum() == 0 or y.sum() == len(y) else float(roc_auc_score(y, s))

def fpr_at_tpr(y, s, target):
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.searchsorted(tpr, target, side="left"))
    return 1.0 if i >= len(tpr) else float(fpr[i])

SCORE = {}           # (model, seed, variant) -> {detector: scores over the variant's images}
ROWS, CHECK = [], []
for name, info in RUNS.items():
    for s in SEEDS:
        d = PRED[(name, s)]
        for v in VARIANTS:
            dv = d[VARIANT_MASK[v]]
            comp = {c: dv[c].values.astype(float) for c in ("uhat", "maxlogit", "knn")}
            e3 = sum(_rank01(comp[c]) for c in info["ens3"]) / len(info["ens3"])
            sc = {"ens3": e3, **comp}
            SCORE[(name, s, v)] = sc
            m = subsets(v)
            neg = m["negative"]
            t_op = float(info["summary"].at[(s, v), "t_ens3"])
            flag = dv.ens3_ref.values >= t_op
            if v == "primary":
                CHECK.append({"model": name, "seed": s, "variant": v, "check": "ens3 column",
                              "diff": float(np.abs(e3 - dv.ens3.values).max())})
            r = {"model": name, "seed": s, "variant": v, "t_ens3": t_op, "n_negative": int(neg.sum()),
                 "op_specificity": float((~flag[neg]).mean())}
            for sub in SUBSETS:
                pos = m[sub]; sel = pos | neg
                r[f"n_{sub}"] = int(pos.sum())
                for det in DETECTORS:
                    x = sc[det]
                    r[f"{sub}_{det}_auroc"] = safe_auroc(pos[sel], x[sel]) if np.isfinite(x).all() else np.nan
                r[f"{sub}_fpr_at_tpr{int(round(100 * OP_TPR))}"] = fpr_at_tpr(pos[sel], e3[sel], OP_TPR)
                r[f"{sub}_op_sensitivity"] = float(flag[pos].mean()) if pos.any() else np.nan
            ROWS.append(r)
            CHECK.append({"model": name, "seed": s, "variant": v, "check": "ens3 AUROC",
                          "diff": abs(r["all_ens3_auroc"] - float(info["summary"].at[(s, v), "ens3_auroc"]))})
            CHECK.append({"model": name, "seed": s, "variant": v, "check": "specificity at t_ens3",
                          "diff": abs(r["op_specificity"] - float(info["summary"].at[(s, v), "op_specificity"]))})
PS = pd.DataFrame(ROWS)
CK = pd.DataFrame(CHECK)
CK.to_csv(os.path.join(RES_DIR, "self_check.csv"), index=False)
PS.to_csv(os.path.join(RES_DIR, "odir_case_split_per_seed.csv"), index=False)
print(CK.groupby("check")["diff"].max().to_string())
if CK["diff"].max() > CHECK_TOL:
    print(CK.sort_values("diff", ascending=False).head(15).to_string(index=False))
    raise RuntimeError("recomputed scores differ from what the runs reported: wrong label file, folder or seed")
print(f"[check] every model, seed and variant reproduces its reported ens3 AUROC and specificity (max diff {CK['diff'].max():.1e})")


# In[ ]:


def weighted_auc_batch(score, pos, neg, W):
    """AUROC (ties count 1/2) for each row of the weight matrix W (B x n); pos/neg are boolean masks over n."""
    s = score[pos | neg]
    W = W[:, pos | neg]
    p, q = pos[pos | neg], neg[pos | neg]
    _, g = np.unique(s, return_inverse=True)                       # dense rank of each score
    G = sp.csr_matrix((np.ones(len(g)), (np.arange(len(g)), g)), shape=(len(g), g.max() + 1))
    Wn = W * q                                                     # negative weights
    neg_per = np.asarray(Wn @ G)                                   # B x groups
    below = np.cumsum(neg_per, axis=1) - neg_per                   # negatives strictly below each group
    num = ((W * p) * (below[:, g] + 0.5 * neg_per[:, g])).sum(1)
    den = (W * p).sum(1) * Wn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den

_r = np.random.RandomState(0); _s = np.round(_r.rand(300), 2); _p = _r.rand(300) < 0.3; _n = ~_p & (_r.rand(300) < 0.8)
_w = _r.poisson(1.0, size=(3, 300)).astype(float)
for _b in range(3):
    _ix = np.repeat(np.arange(300), _w[_b].astype(int)); _ix = _ix[_p[_ix] | _n[_ix]]
    assert abs(weighted_auc_batch(_s, _p, _n, _w[_b:_b + 1])[0] - roc_auc_score(_p[_ix], _s[_ix])) < 1e-12

def bootstrap(v, models=None, n_boot=N_BOOT, seed=BOOT_SEED, chunk=250):
    """{(model, subset): array of length 1 + n_boot}; index 0 = observed (unit weights)."""
    models = list(RUNS) if models is None else models
    m = subsets(v)
    pid = m["patient"]
    upat, inv = np.unique(pid, return_inverse=True)
    rng = np.random.RandomState(seed)
    out = {(mo, sub): [np.array([np.nan])] for mo in models for sub in SUBSETS}
    draws = [np.ones((1, len(upat)))] + [rng.multinomial(len(upat), np.full(len(upat), 1.0 / len(upat)), size=min(chunk, n_boot - k)).astype(float)
                                        for k in range(0, n_boot, chunk)]
    for mo in models:
        for sub in SUBSETS:
            vals = []
            for C in draws:
                W = C[:, inv]                                       # image weight = patient multiplicity
                per_seed = [weighted_auc_batch(SCORE[(mo, s, v)]["ens3"], m[sub], m["negative"], W) for s in SEEDS]
                vals.append(np.mean(per_seed, axis=0))
            out[(mo, sub)] = np.concatenate(vals)
    return out

def ci(x, a=0.05):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (float(np.percentile(x, 100 * a / 2)), float(np.percentile(x, 100 * (1 - a / 2)))) if len(x) else (np.nan, np.nan)

def boot_p(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    return float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / len(d)))) if len(d) else np.nan

t0 = time.time()
BOOT = {v: bootstrap(v) for v in VARIANTS}
for v in VARIANTS:                                                 # observed value = seed mean of the per-seed AUROCs
    for mo in RUNS:
        for sub in SUBSETS:
            obs = PS[(PS.model == mo) & (PS.variant == v)][f"{sub}_ens3_auroc"].mean()
            assert abs(BOOT[v][(mo, sub)][0] - obs) < 1e-9, (v, mo, sub)
print(f"[bootstrap] {N_BOOT} patient-clustered replicates x {len(VARIANTS)} variants x {len(RUNS)} models in {time.time()-t0:.0f} s")


# In[ ]:


TPR_TAG = f"fpr_at_tpr{int(round(100 * OP_TPR))}"
def pfmt(p):
    return "n/a" if not np.isfinite(p) else (f"<= {1.0 / N_BOOT:.4f}" if p <= 1.0 / N_BOOT + 1e-12 else f"{p:.3f}")

SUMM, VS = [], []
for v in VARIANTS:
    for mo in RUNS:
        g = PS[(PS.model == mo) & (PS.variant == v)]
        for sub in SUBSETS:
            b = BOOT[v][(mo, sub)]
            row = {"variant": v, "model": mo, "subset": sub, "n_positive": int(g[f"n_{sub}"].iloc[0]),
                   "few_cases": bool(int(g[f"n_{sub}"].iloc[0]) < MIN_CASES),
                   "n_negative": int(g.n_negative.iloc[0]), "ens3_auroc": float(b[0]), "ci_lo": ci(b[1:])[0],
                   "ci_hi": ci(b[1:])[1], "seed_sd": float(g[f"{sub}_ens3_auroc"].std(ddof=1)),
                   **{f"{d}_auroc": float(g[f"{sub}_{d}_auroc"].mean()) for d in DETECTORS if d != "ens3"},
                   TPR_TAG: float(g[f"{sub}_{TPR_TAG}"].mean()), "op_sensitivity": float(g[f"{sub}_op_sensitivity"].mean()),
                   "op_specificity": float(g.op_specificity.mean())}
            if sub == "mixed":
                dd = b - BOOT[v][(mo, "unknown_only")]
                row.update({"mixed_minus_unknown_only": float(dd[0]), "mmu_ci_lo": ci(dd[1:])[0], "mmu_ci_hi": ci(dd[1:])[1],
                            "mmu_p": boot_p(dd[1:])})
            SUMM.append(row)
            if mo != REFERENCE:
                dd = BOOT[v][(REFERENCE, sub)] - b
                VS.append({"variant": v, "subset": sub, "reference": REFERENCE, "model": mo,
                           "reference_auroc": float(BOOT[v][(REFERENCE, sub)][0]), "model_auroc": float(b[0]),
                           "delta": float(dd[0]), "ci_lo": ci(dd[1:])[0], "ci_hi": ci(dd[1:])[1], "p_boot": boot_p(dd[1:])})
SUMM, VS = pd.DataFrame(SUMM), pd.DataFrame(VS)
SUMM.to_csv(os.path.join(RES_DIR, "odir_case_split_summary.csv"), index=False)
VS.to_csv(os.path.join(RES_DIR, "odir_case_split_vs_reference.csv"), index=False)

for v in VARIANTS:
    c = COMP[COMP.variant == v].iloc[0]
    print("\n" + "=" * 126)
    print(f"  ODIR-5K | {v} | ens3 AUROC [95% CI], seed mean | negatives (known disease only) {c.n_negative_known_only} | "
          f"mixed {c.n_mixed} | unknown-only {c.n_unknown_only}")
    print("=" * 126)
    print(f"  {'model':18} | {'all unknown':^22} | {'mixed':^22} | {'unknown-only':^22} | {'mixed - only':^22} | "
          f"{'FPR@' + str(int(OP_TPR * 100)) + 'TPR':>9} {'spec@op':>8}")
    for mo in RUNS:
        g = SUMM[(SUMM.variant == v) & (SUMM.model == mo)].set_index("subset")
        cells = [f"{g.at[s, 'ens3_auroc']:.3f} [{g.at[s, 'ci_lo']:.3f},{g.at[s, 'ci_hi']:.3f}]{'*' if g.at[s, 'few_cases'] else ''}"
                 for s in SUBSETS]
        mm = g.loc["mixed"]
        cells.append(f"{mm.mixed_minus_unknown_only:+.3f} [{mm.mmu_ci_lo:+.3f},{mm.mmu_ci_hi:+.3f}]")
        print(f"  {mo:18} | " + " | ".join(f"{x:^22}" for x in cells) +
              f" | {g.at['all', TPR_TAG]:9.3f} {g.at['all', 'op_specificity']:8.3f}")
    if (c[["n_mixed", "n_unknown_only"]] < MIN_CASES).any():
        print(f"  * fewer than {MIN_CASES} positives: the interval is not reliable; do not interpret this subset")
    vv = VS[VS.variant == v]
    if len(vv):
        print(f"  paired {REFERENCE} - model: " + " ; ".join(
            f"{r.model} {r.subset} {r.delta:+.3f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}] p {pfmt(r.p_boot)}"
            for r in vv[vv.subset == "mixed"].itertuples()))


# In[ ]:


plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 7})
COL = {"mixed": "#2b4c7e", "unknown_only": "#b5562f"}
MK = {"mixed": ("o", COL["mixed"]), "unknown_only": ("s", "white")}
fig, axes = plt.subplots(1, len(VARIANTS), figsize=(2.4 * len(VARIANTS), 0.32 * len(RUNS) + 0.9), sharey=True)
axes = np.atleast_1d(axes)
names = list(RUNS)
for ax, v in zip(axes, VARIANTS):
    for k, sub in enumerate(("mixed", "unknown_only")):
        g = SUMM[(SUMM.variant == v) & (SUMM.subset == sub)].set_index("model").loc[names]
        y = np.arange(len(names)) + (k - 0.5) * 0.28
        ax.errorbar(g.ens3_auroc, y, xerr=[g.ens3_auroc - g.ci_lo, g.ci_hi - g.ens3_auroc], fmt="none",
                    ecolor=COL[sub], elinewidth=0.9, capsize=1.5)
        ax.scatter(g.ens3_auroc, y, marker=MK[sub][0], s=16, facecolor=MK[sub][1], edgecolor=COL[sub], linewidth=0.9,
                   zorder=3, label=sub.replace("_", "-"))
    c = COMP[COMP.variant == v].iloc[0]
    ax.axvline(0.5, color="0.6", lw=0.7, ls=":")
    small = " (too few cases)" if min(c.n_mixed, c.n_unknown_only) < MIN_CASES else ""
    ax.set_title(f"{v.replace('_', ' ')}{small}\nmixed {c.n_mixed} / unknown-only {c.n_unknown_only}", fontsize=7)
    ax.set_xlabel("ens3 AUROC vs known disease")
    ax.grid(axis="x", color="0.92", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_yticks(np.arange(len(names)), names)
axes[0].invert_yaxis()
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3)
fig.tight_layout(rect=(0, 0, 1, 0.9))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"odir_case_split.{ext}"), dpi=600, bbox_inches="tight")
plt.show()


# In[ ]:


def f3(x):
    return "n/a" if not np.isfinite(x) else f"{x:.3f}"

L = ["# ODIR-5K: mixed vs unknown-only cases", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} | reference {REFERENCE} | "
     f"seeds {SEEDS} | {N_BOOT} patient-clustered bootstrap replicates", ""]
for v in VARIANTS:
    c = COMP[COMP.variant == v].iloc[0]
    L += [f"## {v} (negatives {c.n_negative_known_only}, mixed {c.n_mixed}, unknown-only {c.n_unknown_only})", "",
          "| model | all unknown | mixed | unknown-only | mixed - unknown-only | FPR@90%TPR | spec. at t_ens3 |",
          "|---|---|---|---|---|---|---|"]
    for mo in RUNS:
        g = SUMM[(SUMM.variant == v) & (SUMM.model == mo)].set_index("subset")
        mm = g.loc["mixed"]
        L.append(f"| {mo} | " + " | ".join(f"{f3(g.at[s, 'ens3_auroc'])} [{f3(g.at[s, 'ci_lo'])}, {f3(g.at[s, 'ci_hi'])}]"
                                           + ("*" if g.at[s, "few_cases"] else "") for s in SUBSETS)
                 + f" | {mm.mixed_minus_unknown_only:+.3f} [{mm.mmu_ci_lo:+.3f}, {mm.mmu_ci_hi:+.3f}]"
                 + f" | {f3(g.at['all', TPR_TAG])} | {f3(g.at['all', 'op_specificity'])} |")
    if (c[["n_mixed", "n_unknown_only"]] < MIN_CASES).any():
        L.append(f"\n\\* fewer than {MIN_CASES} positive images: not interpretable.")
    L.append("")
    vv = VS[VS.variant == v]
    for sub in SUBSETS:
        L.append(f"{REFERENCE} - model, {sub}: " + "; ".join(
            f"{r.model} {r.delta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] p {pfmt(r.p_boot)}" for r in vv[vv.subset == sub].itertuples()))
    L.append("")
open(os.path.join(RES_DIR, "ODIR_CASE_SPLIT_SUMMARY.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

pd.DataFrame([{"input": n, "path": p, "bytes": os.path.getsize(p), "sha256": sha256(p)} for n, p in INPUT_FILES]) \
  .to_csv(os.path.join(REPRO_DIR, "input_files_sha256.csv"), index=False)
json.dump({"ODIR_LABELS": os.path.abspath(ODIR_LABELS), "ODIR_LABELS_sha256": LAB_SHA,
           "MODELS": {k: os.path.abspath(v) for k, v in MODELS.items()},
           "ens3_components": {k: v["ens3"] for k, v in RUNS.items()}, "REFERENCE": REFERENCE, "SEEDS": SEEDS,
           "VARIANTS": VARIANTS, "SUBSETS": SUBSETS, "OP_TPR": OP_TPR, "N_BOOT": N_BOOT, "BOOT_SEED": BOOT_SEED,
           "negatives": "has_known & ~is_unknown (normal images excluded)", "mixed": "is_unknown & has_known",
           "unknown_only": "is_unknown & ~has_known", "resampling_unit": "patient_id",
           "estimand": "ens3 AUROC averaged over SEEDS", "operating_threshold": "each run's t_ens3 from external_per_seed.csv"},
          open(os.path.join(REPRO_DIR, "config.json"), "w"), indent=2)
json.dump({**ENV, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(REPRO_DIR, "environment.json"), "w"), indent=2)
print(f"[done] results in {os.path.abspath(RES_DIR)} | figure in {os.path.abspath(FIG_DIR)}")

