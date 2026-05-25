"""Script d'EVALUATION a coller dans un notebook Kaggle.

Evalue un checkpoint deja entraine (best.pt) sur son fold de validation et
produit le CSV par-patient (DSC, HD95, Surface DSC, IoU par OAR).

PRE-REQUIS dans le notebook Kaggle (Settings -> Add Input) :
  1. Le DATASET : mrgrt-oar-thorax-clean-v2  (les 187 patients .nii)
  2. L'OUTPUT du notebook d'entrainement (qui contient runs/<model>_fold<N>/best.pt)
     via Add Input -> Notebook Output -> ton notebook d'entrainement
  + Internet ON (pour cloner le repo) + Secret GITHUB_TOKEN

Ajuster les variables ci-dessous puis Save & Run All.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# =====================================================================
# A AJUSTER
# =====================================================================
GITHUB_USER = "<TON_USER_GITHUB>"
GITHUB_REPO_NAME = "mrgrt-seg"
GITHUB_BRANCH = "main"
GITHUB_TOKEN_SECRET_NAME = "GITHUB_TOKEN"

KAGGLE_INPUT_DATASET = "/kaggle/input/mrgrt-oar-thorax-clean-v2"  # le dataset 187 patients (.nii)
# Dossier ou Kaggle a monte l'output du notebook d'entrainement.
# Apres "Add Input -> Notebook Output", regarde le panneau de droite pour le chemin exact.
CHECKPOINT_INPUT = "/kaggle/input/mrgrt-train-unet-fold0"  # <-- slug de ton notebook training

MODEL = "unet"        # "unet" ou "segresnet" (doit matcher le checkpoint)
FOLD = 0
CONFIG = "configs/default.yaml"
# =====================================================================

WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / GITHUB_REPO_NAME
_token = None


def run(cmd, cwd=None, check=True):
    printable = " ".join(cmd) if isinstance(cmd, list) else cmd
    if _token:
        printable = printable.replace(_token, "***")
    print(f"  $ {printable}" + (f"   (cwd={cwd})" if cwd else ""))
    return subprocess.run(cmd, cwd=cwd, check=check, shell=isinstance(cmd, str),
                          stdout=sys.stdout, stderr=sys.stderr)


# 0) Token GitHub
if GITHUB_TOKEN_SECRET_NAME:
    from kaggle_secrets import UserSecretsClient
    _token = UserSecretsClient().get_secret(GITHUB_TOKEN_SECRET_NAME)
    print(f"[0] Token GitHub recupere (longueur={len(_token)})")
repo_url = (f"https://{GITHUB_USER}:{_token}@github.com/{GITHUB_USER}/{GITHUB_REPO_NAME}.git"
            if _token else f"https://github.com/{GITHUB_USER}/{GITHUB_REPO_NAME}.git")

# 1) Clone / pull
print(f"\n[1] Repo {GITHUB_USER}/{GITHUB_REPO_NAME}")
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

# 3) Lien data/ -> dataset Kaggle
print(f"\n[3] Lien data/ -> {KAGGLE_INPUT_DATASET}")
if not Path(KAGGLE_INPUT_DATASET).exists():
    raise RuntimeError(f"Dataset non monte : {KAGGLE_INPUT_DATASET}")
data_link = REPO_DIR / "data"
if data_link.exists() or data_link.is_symlink():
    (data_link.unlink() if data_link.is_symlink() else shutil.rmtree(data_link))
data_link.symlink_to(KAGGLE_INPUT_DATASET)
n_pat = sum(1 for _ in Path(KAGGLE_INPUT_DATASET).glob("PATIENT_s*"))
print(f"   {n_pat} patients visibles")

# 4) Localiser le checkpoint best.pt dans l'output monte
print(f"\n[4] Recherche du checkpoint dans {CHECKPOINT_INPUT}")
if not Path(CHECKPOINT_INPUT).exists():
    raise RuntimeError(f"Output training non monte : {CHECKPOINT_INPUT}. "
                       f"Settings -> Add Input -> Notebook Output -> ton notebook training.")
candidates = list(Path(CHECKPOINT_INPUT).rglob(f"{MODEL}_fold{FOLD}/best.pt"))
if not candidates:
    candidates = list(Path(CHECKPOINT_INPUT).rglob("best.pt"))
if not candidates:
    raise RuntimeError(f"Aucun best.pt trouve dans {CHECKPOINT_INPUT}")
ckpt = candidates[0]
print(f"   Checkpoint trouve : {ckpt}")

# 5) Detecter le format des fichiers du dataset (.nii ou .nii.gz)
sample = next(Path(KAGGLE_INPUT_DATASET).glob("PATIENT_s*"))
img_fn = "image.nii" if (sample / "image.nii").exists() else "image.nii.gz"
lbl_fn = "label.nii" if (sample / "label.nii").exists() else "label.nii.gz"
print(f"   Format detecte : {img_fn} / {lbl_fn}")

# 6) Lancer l'evaluation
out_csv = f"/kaggle/working/{MODEL}_fold{FOLD}_metrics.csv"
cmd = [sys.executable, "scripts/evaluate.py",
       "--model", MODEL, "--fold", str(FOLD), "--config", CONFIG,
       "--ckpt", str(ckpt),
       "--image_filename", img_fn, "--label_filename", lbl_fn,
       "--device", "cuda" if torch.cuda.is_available() else "cpu",
       "--out", out_csv]
print(f"\n[5] Evaluation : {' '.join(cmd)}")
print("=" * 60); sys.stdout.flush()
ret = subprocess.run(cmd, cwd=REPO_DIR, check=False)
print("=" * 60)
print(f"\nExit code : {ret.returncode}")
print(f"CSV : {out_csv}")
print("\nApres Save Version, telecharge le CSV depuis l'onglet Output.")
