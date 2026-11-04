# -*- coding: utf-8 -*-
"""MRgRT-Seg - Phase 3 (ablation 18,8 M) : evaluation des checkpoints, derivee de
phase2_kaggle_eval/phase2_eval_kaggle.py. Les fonctions de metriques (surface_stats, overlap_stats,
resample_to_native, grid_of, metriques "legacy") sont COPIEES A L'IDENTIQUE (verifie par
phase3_ablation/tests/test_eval_identique.py).

Differences avec la phase 2 :
  - checkpoints cherches dans RUNS_ROOT (sorties de l'entrainement de la meme session) puis INPUT_ROOT ;
  - modele reconstruit a partir de run_meta.json (ecrit par scripts/train.py, branche ablation-18m) ;
    sans run_meta.json (checkpoints de phase 2, controle E3) : detection d'apres les poids, comme en phase 2 ;
  - cle d'un run = (variante, graine, pli) ; colonnes supplementaires "seed" et "tag" en fin de CSV ;
  - le pli de validation est verifie contre la liste enregistree dans run_meta.json.

Sorties : OUT (defaut /kaggle/working/ablation_eval) : metrics_long.csv, runs_summary.csv, histories/,
predictions.zip, splits.json, env.json, errors.log (meme format que la phase 2 + seed, tag).
"""
import csv
import json
import os
import re
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

# =============================== PARAMETRES ===============================
INPUT_ROOT = Path(os.environ.get("P2_INPUT_ROOT", "/kaggle/input"))
RUNS_ROOT = Path(os.environ.get("P3_RUNS_ROOT", "/kaggle/working/runs"))   # sorties de train.py
OUT = Path(os.environ.get("P2_OUT", "/kaggle/working/ablation_eval"))
REPO_DIR = Path(os.environ.get("P2_REPO_DIR", "/kaggle/working/mrgrt-seg"))
REPO_URL = "https://github.com/NSSIRI/mrgrt-seg.git"
REPO_COMMIT = os.environ.get("P3_REPO_COMMIT", "0a36a1e")   # branche ablation-18m (si le depot est absent)
MONAI_VERSION = "1.5.2"
SPACING = (1.5, 1.5, 3.0)        # grille d'entrainement / d'evaluation historique
PATCH = (128, 128, 64)
SW_OVERLAP = 0.5
SURF_TOL_MM = 2.0
SURF_TOL_VOX = 2.0               # tolerance "2" des CSV existants (exprimee en voxels)
EVAL_NATIVE = True               # metriques aussi dans la grille d'origine du label
SAVE_PRED = True                 # masques predits (grille d'origine), NIfTI uint8
MAX_NATIVE_VOXELS = 400_000_000  # garde-fou memoire pour l'espace natif
DEVICE_OVERRIDE = os.environ.get("P2_DEVICE", "auto")   # "auto" | "cuda" | "cpu"
ONLY_VARIANTS = None             # ex. ["segresnet"] ; None = tout ce qui est trouve
ONLY_FOLDS = None                # ex. [0]
MAX_PATIENTS = None              # ex. 2 pour un test rapide
SKIP_INSTALL = os.environ.get("P2_SKIP_INSTALL", "0") == "1"
if os.environ.get("P2_ONLY_FOLDS"):
    ONLY_FOLDS = [int(x) for x in os.environ["P2_ONLY_FOLDS"].split(",")]
if os.environ.get("P2_MAX_PATIENTS"):
    MAX_PATIENTS = int(os.environ["P2_MAX_PATIENTS"])
# ==========================================================================

ORGANS = ["poumon_g", "poumon_d", "coeur", "oesophage"]    # classes 1..4
NUM_CLASSES = 5
# Phase 4 (6 OAR) : P3_ORGANS="poumon_g,poumon_d,coeur,oesophage,moelle,aorte" ; defaut = 4 OAR (phases 2-3)
if os.environ.get("P3_ORGANS"):
    ORGANS = os.environ["P3_ORGANS"].split(",")
    NUM_CLASSES = len(ORGANS) + 1
N_EXPECTED = int(os.environ.get("P3_N_PATIENTS", "187"))
VARIANT_FEATURES = {
    "unet_base": [32, 64, 128, 256, 512],     # configs/default.yaml
    "unet_opt": [48, 96, 192, 384, 832],      # configs/unet_fair.yaml
}
SEGRESNET_KW = {"init_filters": 32, "blocks_down": [1, 2, 2, 4],
                "blocks_up": [1, 1, 1], "dropout_prob": 0.2}   # configs/default.yaml

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "histories").mkdir(exist_ok=True)
T_START = time.time()


def log_error(msg: str) -> None:
    print(f"[ERREUR] {msg}", flush=True)
    with open(OUT / "errors.log", "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def sh(cmd, cwd=None) -> int:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd).returncode


# ---------------------------------------------------------------- 1) dependances
print("\n[1] Dependances", flush=True)
if not SKIP_INSTALL:
    if sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", f"monai=={MONAI_VERSION}"]) != 0:
        sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "monai"])
    for mod, pkg in [("nibabel", "nibabel"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
                     ("yaml", "pyyaml"), ("pandas", "pandas")]:
        try:
            __import__(mod)
        except ImportError:
            sh([sys.executable, "-m", "pip", "install", "-q", pkg])

# ---------------------------------------------------------------- 2) code du projet
print("\n[2] Code du projet", flush=True)
if not (REPO_DIR / "src").exists():
    if sh(["git", "clone", "--quiet", REPO_URL, REPO_DIR]) != 0:
        raise SystemExit("Clonage GitHub impossible : verifier que Internet est active.")
    if sh(["git", "checkout", "--quiet", REPO_COMMIT], cwd=REPO_DIR) != 0:
        log_error(f"commit {REPO_COMMIT} introuvable : on garde la branche par defaut")
try:
    REPO_HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR,
                               capture_output=True, text=True).stdout.strip() or "copie-locale"
except Exception:
    REPO_HEAD = "copie-locale"
sys.path.insert(0, str(REPO_DIR))
print(f"   code : {REPO_DIR} @ {REPO_HEAD}", flush=True)

import numpy as np                      # noqa: E402
import scipy.ndimage as ndi             # noqa: E402
from scipy.spatial import cKDTree       # noqa: E402
import gc as _gc                         # noqa: E402
import torch                            # noqa: E402
import nibabel as nib                   # noqa: E402
import monai                            # noqa: E402

# Meme correctif que scripts/evaluate.py (cuCIM indisponible / non compilable sur Kaggle)
try:
    import monai.metrics.utils as _mmu

    def _scipy_binary_erosion(arr, *args, **kwargs):
        if hasattr(arr, "get"):
            arr = arr.get()
        if hasattr(arr, "cpu"):
            arr = arr.cpu().numpy()
        return ndi.binary_erosion(np.asarray(arr).astype(bool))

    _mmu.cucim_binary_erosion = _scipy_binary_erosion
    if hasattr(_mmu, "has_cucim"):
        _mmu.has_cucim = False
except Exception as e:  # pragma: no cover
    print(f"[warn] patch cuCIM impossible : {e}")

from monai.data import MetaTensor                                   # noqa: E402
from monai.inferers import sliding_window_inference                  # noqa: E402
from monai.transforms import AsDiscrete, ResampleToMatch  # noqa: E402

from src.data.dataset import list_patients, get_val_transforms       # noqa: E402
from src.data.splits import make_5fold_splits, load_fold             # noqa: E402
from src.models.factory import build_model, count_parameters         # noqa: E402
from src.eval.metrics import build_metrics                           # noqa: E402

warnings.filterwarnings("ignore", message=".*is all 0, this may result in nan/inf distance.*")


# ---------------------------------------------------------------- 3) device
def pick_device() -> str:
    if DEVICE_OVERRIDE in ("cuda", "cpu"):
        return DEVICE_OVERRIDE
    if not torch.cuda.is_available():
        return "cpu"
    try:  # certains GPU (P100, sm_60) ne sont plus supportes par les versions recentes de PyTorch
        conv = torch.nn.Conv3d(1, 2, 3).cuda()
        _ = conv(torch.randn(1, 1, 8, 8, 8, device="cuda")).sum().item()
        return "cuda"
    except Exception as e:
        print(f"[warn] GPU inutilisable ({e}) -> CPU", flush=True)
        return "cpu"


DEVICE = pick_device()
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else None
print(f"\n[3] Device : {DEVICE} {GPU_NAME or ''}", flush=True)


# ---------------------------------------------------------------- 4) dataset + plis
def find_dataset(root: Path) -> Path:
    best, best_n = None, 0
    frontier = [root]
    for _ in range(5):                                  # profondeur max 5
        nxt = []
        for d in frontier:
            try:
                subs = [p for p in d.iterdir() if p.is_dir()]
            except (PermissionError, FileNotFoundError):
                continue
            n = sum(1 for p in subs if p.name.startswith("PATIENT_s"))
            if n > best_n:
                best, best_n = d, n
            nxt.extend(p for p in subs if not p.name.startswith("PATIENT_s"))
        frontier = nxt
    if best is None:
        raise SystemExit(f"Aucun dossier PATIENT_s* trouve sous {root} : ajouter le dataset en Input.")
    return best


DATA = find_dataset(INPUT_ROOT)
sample = sorted(p for p in DATA.iterdir() if p.name.startswith("PATIENT_s"))[0]
IMG_FN = "image.nii" if (sample / "image.nii").exists() else "image.nii.gz"
LBL_FN = "label.nii" if (sample / "label.nii").exists() else "label.nii.gz"
items = list_patients(str(DATA), IMG_FN, LBL_FN)
folds = make_5fold_splits(items, n_folds=5, seed=42)
sizes = [len(f["val"]) for f in folds]
print(f"\n[4] Dataset : {DATA} ({len(items)} patients, {IMG_FN}/{LBL_FN}) ; plis val = {sizes}", flush=True)
if len(items) != N_EXPECTED:
    log_error(f"{len(items)} patients au lieu de {N_EXPECTED} : les plis ne correspondront PAS aux CSV existants")
(OUT / "splits.json").write_text(json.dumps(folds, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 5) checkpoints
def detect_variant(sd: dict):
    if "convInit.conv.weight" in sd:
        return "segresnet"
    w = sd.get("model.0.conv.weight")
    if w is not None:
        c0 = int(w.shape[0])
        return {32: "unet_base", 48: "unet_opt"}.get(c0, f"unet_c{c0}")
    return "unknown"


VARIANT_ALIASES = {"e3_unet_opt": "unet_opt", "e2_segresnet": "segresnet"}   # memes noms qu'en phase 2
REF_FOLDS = {f["fold"]: f["val"] for f in folds}

print("\n[5] Recherche des checkpoints best.pt", flush=True)
cands = []
roots = [r for r in (RUNS_ROOT, INPUT_ROOT) if r.exists()]
seen = set()
for p in sorted(q for r in roots for q in r.rglob("best.pt")):
    if p.resolve() in seen:
        continue
    seen.add(p.resolve())
    m = re.fullmatch(r"(.+)_fold(\d+)", p.parent.name)
    if not m:
        continue
    meta_path = p.parent / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        log_error(f"lecture impossible {p} : {e}")
        continue
    sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    if meta:
        stem = Path(meta["config"]).stem
        variant, seed, tag = VARIANT_ALIASES.get(stem, stem), int(meta["train_seed"]), meta["tag"]
        family, fold = meta["model"], int(meta["fold"])
        if fold != int(m.group(2)):
            log_error(f"pli {fold} (run_meta) != dossier {p.parent.name}")
            continue
        if int(meta.get("split_seed", -1)) != 42 or meta.get("val_patients") != REF_FOLDS.get(fold):
            log_error(f"{p} : pli de validation de run_meta.json different de splits.json (graine 42) -> ignore")
            continue
    else:                                   # checkpoint de phase 2 (controle E3) : comme phase2_eval_kaggle.py
        if m.group(1) not in ("unet", "segresnet"):
            continue
        family, fold = m.group(1), int(m.group(2))
        variant, seed, tag = detect_variant(sd), 42, "phase2"
        if not variant.startswith(family):
            log_error(f"poids {variant} dans un dossier {family}_fold : {p}")
    cands.append({
        "path": str(p), "family": family, "fold": fold, "variant": variant, "seed": seed, "tag": tag,
        "meta": meta,
        "train_seed": meta.get("train_seed") if meta else None,
        "split_seed": meta.get("split_seed") if meta else None,
        "train_commit": meta.get("git_commit") if meta else None,
        "train_monai": meta.get("monai_version") if meta else None,
        "train_torch": meta.get("torch_version") if meta else None,
        "best_epoch": ck.get("epoch") if isinstance(ck, dict) else None,
        "best_val_dsc": ck.get("val_dsc") if isinstance(ck, dict) else None,
        "cfg_epochs": cfg.get("epochs"), "cfg_lr": cfg.get("lr"),
        "cfg_loss": (cfg.get("loss") or {}).get("name"),
        "mtime": p.stat().st_mtime, "size_mb": round(p.stat().st_size / 2**20, 1),
    })
    del ck, sd

runs = {}
for c in sorted(cands, key=lambda c: -c["mtime"]):
    key = (c["variant"], c["seed"], c["fold"])
    c["selected"] = key not in runs
    if c["selected"]:
        runs[key] = c
for c in cands:
    print(f"   {'OK ' if c['selected'] else 'dup'} {c['variant']:<16} s{c['seed']} fold{c['fold']}  "
          f"epoch={c['best_epoch']} val_dsc={c['best_val_dsc']}  {c['path']}", flush=True)
if not runs:
    raise SystemExit("Aucun best.pt trouve : ajouter les sorties des notebooks d'entrainement en Input.")
if ONLY_VARIANTS:
    runs = {k: v for k, v in runs.items() if k[0] in ONLY_VARIANTS}
if ONLY_FOLDS is not None:
    runs = {k: v for k, v in runs.items() if k[2] in ONLY_FOLDS}


# ---------------------------------------------------------------- 6) historiques
def export_history(run: dict) -> None:
    rdir = Path(run["path"]).parent
    hist, src = None, None
    if (rdir / "history.npz").exists():
        z = np.load(rdir / "history.npz")
        hist, src = {k: z[k].tolist() for k in z.files}, "history.npz"
    if (rdir / "last.pt").exists():
        try:
            ck = torch.load(rdir / "last.pt", map_location="cpu", weights_only=False)
            run["last_epoch"] = ck.get("epoch")
            run["epochs_no_improve"] = ck.get("epochs_no_improve")
            run["best_dsc_last"] = ck.get("best_dsc")
            if hist is None and isinstance(ck.get("history"), dict):
                hist, src = ck["history"], "last.pt"
            del ck
        except Exception as e:
            log_error(f"last.pt illisible {rdir} : {e}")
    run["history_source"] = src
    if not hist:
        return
    n = max(len(v) for v in hist.values())
    path = OUT / "histories" / f"{run['variant']}_s{run['seed']}_fold{run['fold']}_history.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = [k for k in ("train_loss", "val_dsc", "lr") if k in hist] + \
               [k for k in hist if k not in ("train_loss", "val_dsc", "lr")]
        w.writerow(["epoch"] + keys)
        for i in range(n):
            w.writerow([i + 1] + [hist[k][i] if i < len(hist[k]) else "" for k in keys])
    run["history_epochs"] = n


# ---------------------------------------------------------------- 7) metriques
STRUCT = ndi.generate_binary_structure(3, 1)


def _bbox(u: np.ndarray, margin: int = 1):
    sl = []
    for ax in range(u.ndim):
        proj = np.any(u, axis=tuple(i for i in range(u.ndim) if i != ax))
        idx = np.flatnonzero(proj)
        sl.append(slice(max(int(idx[0]) - margin, 0), min(int(idx[-1]) + margin + 1, u.shape[ax])))
    return tuple(sl)


def surface_stats(p: np.ndarray, g: np.ndarray, spacing, tol: float) -> dict:
    """Distances de surface bord-a-bord (meme definition que MONAI : bords = masque XOR erosion 6-connexe ;
    distances euclidiennes en mm entre centres de voxels de bord, calculees par KD-tree).

    hd95 = max(P95(pred->gt), P95(gt->pred)) ; hdmax = max des deux maxima ;
    assd = moyenne des distances des deux directions concatenees ;
    sdsc = fraction des points de bord (des deux masques) a <= tol de l'autre bord.
    """
    out = {"hd95": np.nan, "hdmax": np.nan, "assd": np.nan, "sdsc": np.nan}
    P, G = bool(p.any()), bool(g.any())
    if not P and not G:
        return out
    if P != G:                          # un des deux masques est vide
        out["sdsc"] = 0.0
        return out
    sl = _bbox(p | g)
    p, g = np.pad(p[sl], 1), np.pad(g[sl], 1)
    ep = p & ~ndi.binary_erosion(p, structure=STRUCT)
    eg = g & ~ndi.binary_erosion(g, structure=STRUCT)
    # plus proche voisin entre centres de voxels de bord (identique a l'EDT, beaucoup moins de memoire)
    sp = np.asarray(spacing, dtype=np.float64)
    pts_p, pts_g = np.argwhere(ep) * sp, np.argwhere(eg) * sp
    del p, g, ep, eg
    d_pg = cKDTree(pts_g).query(pts_p, workers=-1)[0]
    d_gp = cKDTree(pts_p).query(pts_g, workers=-1)[0]
    both = np.concatenate([d_pg, d_gp])
    out["hd95"] = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    out["hdmax"] = float(max(d_pg.max(), d_gp.max()))
    out["assd"] = float(both.mean())
    out["sdsc"] = float(((d_pg <= tol).sum() + (d_gp <= tol).sum()) / both.size)
    return out


def overlap_stats(p: np.ndarray, g: np.ndarray, vox_ml: float) -> dict:
    tp = int(np.count_nonzero(p & g))
    npred, ngt = int(np.count_nonzero(p)), int(np.count_nonzero(g))
    fp, fn = npred - tp, ngt - tp
    nan = float("nan")
    return {
        "dsc": 2 * tp / (2 * tp + fp + fn) if ngt > 0 else nan,     # NaN si GT vide (comme MONAI)
        "iou": tp / (tp + fp + fn) if ngt > 0 else nan,
        "precision": tp / npred if npred > 0 else nan,
        "recall": tp / ngt if ngt > 0 else nan,
        "vol_gt_ml": ngt * vox_ml, "vol_pred_ml": npred * vox_ml,
        "vol_diff_ml": (npred - ngt) * vox_ml,
        "vol_rel_err": (npred - ngt) / ngt if ngt > 0 else nan,
        "empty_gt": int(ngt == 0), "empty_pred": int(npred == 0),
    }


def resample_to_native(pred: np.ndarray, src_affine, dst_affine: np.ndarray, dst_shape,
                       max_vox: int = 8_000_000) -> np.ndarray:
    """Plus proche voisin MONAI (ResampleToMatch) de la grille reechantillonnee vers la grille d'origine,
    par tranches pour borner la memoire (resultat identique au reechantillonnage en un bloc)."""
    dst_shape = tuple(int(x) for x in dst_shape)
    out = np.zeros(dst_shape, dtype=np.uint8)
    ax = int(np.argmax(dst_shape))
    per_slice = int(np.prod(dst_shape)) // dst_shape[ax]
    step = max(1, max_vox // max(per_slice, 1))
    src = MetaTensor(torch.from_numpy(pred[None].astype(np.float32)), affine=src_affine)
    for a in range(0, dst_shape[ax], step):
        b = min(a + step, dst_shape[ax])
        shift = np.eye(4)
        shift[ax, 3] = a
        shp = list(dst_shape)
        shp[ax] = b - a
        dst = MetaTensor(torch.zeros([1] + shp, dtype=torch.float32),
                         affine=torch.as_tensor(dst_affine @ shift, dtype=torch.float64))
        r = resample(src, img_dst=dst)
        sl = [slice(None)] * 3
        sl[ax] = slice(a, b)
        out[tuple(sl)] = np.rint(r[0].cpu().numpy()).astype(np.uint8)
        del r, dst
    return out


def grid_of(affine) -> tuple:
    a = np.asarray(affine, dtype=np.float64)[:3, :3]
    spacing = tuple(float(x) for x in np.sqrt((a ** 2).sum(axis=0)))
    return spacing, abs(float(np.linalg.det(a))) / 1000.0


FIELDS = ["variant", "fold", "patient_id", "organ", "space",
          "dsc", "iou", "precision", "recall",
          "hd95_mm", "hdmax_mm", "assd_mm", "sdsc_2mm",
          "vol_gt_ml", "vol_pred_ml", "vol_diff_ml", "vol_rel_err", "empty_gt", "empty_pred",
          "hd95_vox", "sdsc_2vox", "dsc_legacy", "iou_legacy", "hd95_legacy_vox", "sdsc_legacy_2vox",
          "sp_x", "sp_y", "sp_z", "t_infer_s", "t_metrics_s", "seed", "tag"]
METRICS_CSV = OUT / "metrics_long.csv"


def append_rows(rows: list) -> None:
    new = not METRICS_CSV.exists()
    with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def done_keys() -> set:
    if not METRICS_CSV.exists():
        return set()
    with open(METRICS_CSV, newline="", encoding="utf-8") as f:
        return {(r["variant"], r["seed"], r["fold"], r["patient_id"]) for r in csv.DictReader(f)}


# ---------------------------------------------------------------- 8) boucle d'evaluation
val_tf = get_val_transforms(spacing=list(SPACING), intensity_params={"clip_percentiles": [0.5, 99.5]},
                            modality="mri")
post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
post_label = AsDiscrete(to_onehot=NUM_CLASSES)
legacy = build_metrics(NUM_CLASSES, surface_tol_mm=SURF_TOL_VOX)   # identique a evaluate.py
resample = ResampleToMatch(mode="nearest", padding_mode="zeros")
already = done_keys()

print(f"\n[6] Evaluation de {len(runs)} run(s)", flush=True)
for (variant, seed, fold), run in sorted(runs.items()):
    t_run = time.time()
    tag = run["tag"]
    print(f"\n=== {variant} s{seed} fold{fold} ({run['path']}) ===", flush=True)
    try:
        export_history(run)
        ck = torch.load(run["path"], map_location="cpu", weights_only=False)
        sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
        meta = run["meta"]
        if meta:                                   # reconstruction exacte depuis run_meta.json
            model = build_model(meta["model"], meta["in_channels"], meta["out_channels"],
                                features=meta["features"], segresnet_kwargs=meta["segresnet_kwargs"],
                                unet_kwargs=meta["unet_kwargs"])
        elif variant == "segresnet":
            model = build_model("segresnet", 1, NUM_CLASSES, segresnet_kwargs=dict(SEGRESNET_KW))
        elif variant in VARIANT_FEATURES:
            model = build_model("unet", 1, NUM_CLASSES, features=VARIANT_FEATURES[variant])
        else:
            raise ValueError(f"architecture non reconnue : {variant}")
        model.load_state_dict(sd, strict=True)
        model = model.to(DEVICE).eval()
        run["n_params"] = count_parameters(model)
        if meta and run["n_params"] != meta.get("n_params"):
            raise ValueError(f"{run['n_params']} parametres != run_meta.json ({meta.get('n_params')})")
        del ck, sd
    except Exception as e:
        log_error(f"{variant} s{seed} fold{fold} : chargement impossible : {e}\n{traceback.format_exc()}")
        run["status"] = "load_error"
        continue

    _, val_items = load_fold(folds, fold, items)
    if MAX_PATIENTS:
        val_items = val_items[:MAX_PATIENTS]
    pred_dir = OUT / "predictions" / variant / f"s{seed}" / f"fold{fold}"
    if SAVE_PRED:
        pred_dir.mkdir(parents=True, exist_ok=True)
    n_ok, n_skip, dsc_acc = 0, 0, {o: [] for o in ORGANS}
    for i, it in enumerate(val_items, 1):
        pid = it["patient_id"]
        if (variant, str(seed), str(fold), pid) in already:      # reprise : deja evalue lors d'un run precedent
            n_skip += 1
            continue
        rows = []
        try:
            d = val_tf({"image": it["image"], "label": it["label"]})
            img, lbl = d["image"], d["label"]
            t0 = time.time()
            with torch.no_grad():
                logits = sliding_window_inference(
                    inputs=img.unsqueeze(0).to(DEVICE), roi_size=PATCH, sw_batch_size=2,
                    predictor=model, overlap=SW_OVERLAP, mode="gaussian",
                    sw_device=DEVICE, device="cpu")
            t_inf = time.time() - t0
            t1 = time.time()
            logits = logits.as_tensor() if isinstance(logits, MetaTensor) else logits
            lbl_t = lbl.as_tensor() if isinstance(lbl, MetaTensor) else lbl

            # --- metriques "legacy" : exactement le calcul des CSV existants
            p_oh = post_pred(logits[0]).unsqueeze(0).cpu()
            g_oh = post_label(lbl_t).unsqueeze(0).cpu()
            leg = {}
            for name, metric in legacy.items():
                metric.reset()
                metric(y_pred=p_oh, y=g_oh)
                leg[name] = metric.aggregate().cpu().numpy().flatten()
            del p_oh, g_oh

            pred = torch.argmax(logits[0], dim=0).numpy().astype(np.uint8)
            gt = np.rint(lbl_t[0].numpy()).astype(np.uint8)
            del logits
            sp_rs, vox_rs = grid_of(img.affine)

            # --- grille reechantillonnee
            for c, organ in enumerate(ORGANS, start=1):
                pc, gc = pred == c, gt == c
                r = {"variant": variant, "fold": fold, "patient_id": pid, "organ": organ,
                     "seed": seed, "tag": tag,
                     "space": "resampled", "sp_x": sp_rs[0], "sp_y": sp_rs[1], "sp_z": sp_rs[2],
                     "t_infer_s": round(t_inf, 2)}
                r.update(overlap_stats(pc, gc, vox_rs))
                s_mm = surface_stats(pc, gc, sp_rs, SURF_TOL_MM)
                s_vx = surface_stats(pc, gc, (1.0, 1.0, 1.0), SURF_TOL_VOX)
                r.update({"hd95_mm": s_mm["hd95"], "hdmax_mm": s_mm["hdmax"], "assd_mm": s_mm["assd"],
                          "sdsc_2mm": s_mm["sdsc"], "hd95_vox": s_vx["hd95"], "sdsc_2vox": s_vx["sdsc"],
                          "dsc_legacy": float(leg["dsc"][c - 1]), "iou_legacy": float(leg["iou"][c - 1]),
                          "hd95_legacy_vox": float(leg["hd95"][c - 1]),
                          "sdsc_legacy_2vox": float(leg["surface_dsc"][c - 1])})
                rows.append(r)
                dsc_acc[organ].append(r["dsc"])

            # --- grille d'origine du label
            if EVAL_NATIVE or SAVE_PRED:
                lab_nii = nib.load(it["label"])
                n_nat = int(np.prod(lab_nii.shape[:3]))
                if n_nat > MAX_NATIVE_VOXELS:
                    raise RuntimeError(f"volume natif trop grand ({n_nat / 1e6:.0f} M voxels) : espace natif ignore")
                gnat = np.rint(np.asanyarray(lab_nii.dataobj)).astype(np.uint8)
                aff_nat = np.asarray(lab_nii.affine, dtype=np.float64)
                pnat = resample_to_native(pred, img.affine.clone(), aff_nat, gnat.shape)
                _gc.collect()
                if pnat.shape != gnat.shape:
                    raise RuntimeError(f"forme native {pnat.shape} != {gnat.shape}")
                if SAVE_PRED:
                    nib.save(nib.Nifti1Image(pnat, aff_nat), str(pred_dir / f"{pid}_pred.nii.gz"))
                if EVAL_NATIVE:
                    sp_n, vox_n = grid_of(aff_nat)
                    for c, organ in enumerate(ORGANS, start=1):
                        pc, gc = pnat == c, gnat == c
                        r = {"variant": variant, "fold": fold, "patient_id": pid, "organ": organ,
                             "seed": seed, "tag": tag,
                             "space": "native", "sp_x": sp_n[0], "sp_y": sp_n[1], "sp_z": sp_n[2],
                             "t_infer_s": round(t_inf, 2)}
                        r.update(overlap_stats(pc, gc, vox_n))
                        s_mm = surface_stats(pc, gc, sp_n, SURF_TOL_MM)
                        r.update({"hd95_mm": s_mm["hd95"], "hdmax_mm": s_mm["hdmax"],
                                  "assd_mm": s_mm["assd"], "sdsc_2mm": s_mm["sdsc"]})
                        rows.append(r)
            t_met = round(time.time() - t1, 2)
            for r in rows:
                r["t_metrics_s"] = t_met
            append_rows(rows)
            n_ok += 1
            _gc.collect()
            msg = " ".join(f"{o[:5]}={rows[k]['dsc']:.3f}" for k, o in enumerate(ORGANS))
            print(f"  [{i}/{len(val_items)}] {pid} inf={t_inf:.1f}s met={t_met:.1f}s | DSC {msg}", flush=True)
        except Exception as e:
            log_error(f"{variant} s{seed} fold{fold} {pid} : {e}\n{traceback.format_exc()}")
    run["n_patients_evaluated"] = n_ok + n_skip
    run["run_minutes"] = round((time.time() - t_run) / 60, 1)
    run["status"] = "ok" if n_ok + n_skip == len(val_items) else "partial"
    means = {o: float(np.nanmean(v)) if v else float("nan") for o, v in dsc_acc.items()}
    print(f"  -> {variant} s{seed} fold{fold} : {n_ok} evalues + {n_skip} deja faits / {len(val_items)} ; "
          "DSC moyen (ce run) " + " ".join(f"{o}={m:.3f}" for o, m in means.items()), flush=True)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ---------------------------------------------------------------- 9) archive des masques predits
# Un seul fichier predictions.zip au lieu de centaines de NIfTI : telechargement plus rapide.
pred_root = OUT / "predictions"
if SAVE_PRED and pred_root.exists() and any(pred_root.rglob("*.nii.gz")):
    import shutil
    try:
        shutil.make_archive(str(OUT / "predictions"), "zip", root_dir=str(pred_root))
        shutil.rmtree(pred_root)
        print(f"\nMasques predits archives dans {OUT / 'predictions.zip'}", flush=True)
    except Exception as e:
        log_error(f"archivage des predictions impossible : {e}")

# ---------------------------------------------------------------- 10) recapitulatif
keys = ["variant", "seed", "tag", "fold", "selected", "status", "path", "best_epoch", "best_val_dsc", "last_epoch",
        "epochs_no_improve", "best_dsc_last", "cfg_epochs", "cfg_lr", "cfg_loss", "n_params",
        "history_source", "history_epochs", "n_patients_evaluated", "run_minutes", "size_mb",
        "train_seed", "split_seed", "train_commit", "train_monai", "train_torch"]
with open(OUT / "runs_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for c in sorted(cands, key=lambda c: (c["variant"], c["seed"], c["fold"], not c["selected"])):
        w.writerow(c)

env = {
    "python": sys.version.split()[0], "torch": torch.__version__, "monai": monai.__version__,
    "numpy": np.__version__, "nibabel": nib.__version__,
    "scipy": __import__("scipy").__version__, "device": DEVICE, "gpu": GPU_NAME,
    "repo_head": REPO_HEAD, "dataset": str(DATA), "n_patients": len(items), "val_sizes": sizes,
    "image_file": IMG_FN, "spacing": SPACING, "patch": PATCH, "overlap": SW_OVERLAP,
    "surface_tol_mm": SURF_TOL_MM, "minutes_total": round((time.time() - T_START) / 60, 1),
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
}
(OUT / "env.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
print(f"\nTermine en {env['minutes_total']} min. Sorties : {OUT}", flush=True)
print("Recuperation : python phase2_kaggle_eval/17_recuperer_sorties.py <compte/slug> <dossier> <regex>")
