"""Batch evaluation : evalue les 5 folds d'un modele en boucle.

A coller dans une seule cellule de notebook Kaggle. Le notebook doit avoir
en Input :
  1. Le DATASET : mrgrt-oar-thorax-clean-v2
  2. Les 5 OUTPUTS du notebook d'entrainement : mrgrt-train-<MODEL>-fold0, ..., fold4
     (via Add Input -> Notebook Output, faire 5 fois)

Le script auto-detecte le path du checkpoint pour chaque fold, lance l'eval
et sauvegarde un CSV par fold dans /kaggle/working/.

Ajuster MODEL ci-dessous (unet ou segresnet) puis Save & Run All.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# =====================================================================
# A AJUSTER
# =====================================================================
GITHUB_USER = "NSSIRI"
GITHUB_REPO_NAME = "mrgrt-seg"
GITHUB_BRANCH = "main"

MODEL = "unet"       # "unet" ou "segresnet" — change selon le compte
FOLDS = [0, 1, 2, 3, 4]   # evalue les 5 folds. Reduire si certains deja faits.
CONFIG = "configs/default.yaml"

# Dataset (auto-detect entre 2 paths possibles)
KAGGLE_INPUT_CANDIDATES = [
    "/kaggle/input/mrgrt-oar-thorax-clean-v2",
    "/kaggle/input/datasets/abdelhalimnssiri/mrgrt-oar-thorax-clean-v2",
]
# =====================================================================

WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / GITHUB_REPO_NAME


def run(cmd, cwd=None, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}"
          + (f"   (cwd={cwd})" if cwd else ""))
    return subprocess.run(cmd, cwd=cwd, check=check, shell=isinstance(cmd, str),
                          stdout=sys.stdout, stderr=sys.stderr)


# 1) Clone du repo (public, pas de secret)
print(f"\n[1] Repo {GITHUB_USER}/{GITHUB_REPO_NAME}")
repo_url = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO_NAME}.git"
if REPO_DIR.exists():
    subprocess.run(["git", "remote", "set-url", "origin", repo_url], cwd=REPO_DIR,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run(["git", "fetch", "origin"], cwd=REPO_DIR)
    run(["git", "reset", "--hard", f"origin/{GITHUB_BRANCH}"], cwd=REPO_DIR)
else:
    run(["git", "clone", "--branch", GITHUB_BRANCH, repo_url, str(REPO_DIR)])

# 2) Dependances
print("\n[2] Dependances")
run([sys.executable, "-m", "pip", "install", "-q",
     "monai>=1.3", "nibabel>=5.1", "SimpleITK>=2.3", "einops>=0.7",
     "scikit-image>=0.22", "scipy>=1.11"])

import torch
print(f"   torch={torch.__version__} | CUDA={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU : {torch.cuda.get_device_name(0)}")

# 3) Auto-detect dataset
print(f"\n[3] Dataset")
KAGGLE_INPUT_DATASET = None
for cand in KAGGLE_INPUT_CANDIDATES:
    if Path(cand).exists() and any(Path(cand).glob("PATIENT_s*")):
        KAGGLE_INPUT_DATASET = cand
        n_pat = sum(1 for _ in Path(cand).glob("PATIENT_s*"))
        print(f"   trouve : {cand} ({n_pat} patients)")
        break
if KAGGLE_INPUT_DATASET is None:
    print("   ERREUR : dataset non trouve. Contenu de /kaggle/input :")
    for p in Path("/kaggle/input").iterdir():
        print(f"     {p}")
    sys.exit(1)

# 4) Symlink data/
data_link = REPO_DIR / "data"
if data_link.exists() or data_link.is_symlink():
    (data_link.unlink() if data_link.is_symlink() else shutil.rmtree(data_link))
data_link.symlink_to(KAGGLE_INPUT_DATASET)

# 5) Detect format
sample = next(Path(KAGGLE_INPUT_DATASET).glob("PATIENT_s*"))
img_fn = "image.nii" if (sample / "image.nii").exists() else "image.nii.gz"
lbl_fn = "label.nii" if (sample / "label.nii").exists() else "label.nii.gz"
print(f"   Format : {img_fn} / {lbl_fn}")

# 6) Auto-detect checkpoint pour chaque fold
print(f"\n[4] Recherche checkpoints pour {MODEL} folds {FOLDS}")
notebooks_root = Path("/kaggle/input/notebooks")
if not notebooks_root.exists():
    print(f"   ERREUR : {notebooks_root} n'existe pas")
    sys.exit(1)

ckpts = {}
for fold in FOLDS:
    # Le script train.py sauvegarde toujours dans runs/<MODEL>_fold<N>/best.pt
    # quelle que soit la facon dont le notebook Kaggle est nomme.
    # Donc on cherche cette structure interne (rglob) dans TOUS les input notebooks.
    pattern = f"**/{MODEL}_fold{fold}/best.pt"
    found = list(notebooks_root.rglob(pattern))
    if found:
        # Si plusieurs notebooks ont le meme checkpoint, prendre le plus recent
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        ckpts[fold] = found[0]
        if len(found) > 1:
            print(f"   fold {fold} : {found[0]}  ({len(found)} candidats, on prend le plus recent)")
        else:
            print(f"   fold {fold} : {found[0]}")
    else:
        print(f"   fold {fold} : NOT FOUND (skip)")

if not ckpts:
    print("ERREUR : aucun checkpoint trouve. Verifie les Inputs (Add Input -> Notebook Output -> mrgrt-train-<model>-fold<N>)")
    sys.exit(1)

# 7) Lancer eval pour chaque fold
print(f"\n[5] Lancement des {len(ckpts)} evaluations")
print("=" * 70)
results = {}
for fold, ckpt in ckpts.items():
    print(f"\n--- Fold {fold} ---")
    out_csv = f"/kaggle/working/{MODEL}_fold{fold}_metrics.csv"
    cmd = [sys.executable, "scripts/evaluate.py",
           "--model", MODEL, "--fold", str(fold), "--config", CONFIG,
           "--ckpt", str(ckpt),
           "--image_filename", img_fn, "--label_filename", lbl_fn,
           "--device", "cuda" if torch.cuda.is_available() else "cpu",
           "--out", out_csv]
    sys.stdout.flush()
    ret = subprocess.run(cmd, cwd=REPO_DIR, check=False)
    results[fold] = (ret.returncode, out_csv)
    print(f"--- Fold {fold} : exit code {ret.returncode}, CSV : {out_csv} ---")

# 8) Recap final
print("\n" + "=" * 70)
print("RECAP FINAL")
print("=" * 70)
for fold, (rc, csv) in sorted(results.items()):
    status = "OK" if rc == 0 else "FAIL"
    print(f"  fold {fold} : {status:<5}  ->  {csv}")

print(f"\nApres Save Version, telecharge les CSV depuis l'onglet Output.")
