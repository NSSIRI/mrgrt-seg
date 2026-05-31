"""Script XAI batch a coller dans un notebook Kaggle.

Genere les cartes SEG-GRAD-CAM 3D + 3 metriques + Adebayo sanity check
sur les 5 folds d'un modele (UNet ou SegResNet) en une seule run.

Pre-requis Inputs (Settings -> Add Input) :
  1. Dataset : mrgrt-oar-thorax-clean-v2 (187 patients, partage compte 1)
  2. 5 notebooks outputs : <user>/<model>-fold-0, fold-1, ..., fold-4

Settings : GPU T4 x2 + Internet On.
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
MODEL = "segresnet"            # "unet" ou "segresnet"
FOLDS = [0, 1, 2, 3, 4]
SKIP_ADEBAYO = False           # True pour gagner du temps
MAX_PATIENTS = None            # int pour debug (ex: 5)
# =====================================================================

WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / GITHUB_REPO_NAME


def run(cmd, cwd=None, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}"
          + (f"   (cwd={cwd})" if cwd else ""))
    return subprocess.run(cmd, cwd=cwd, check=check, shell=isinstance(cmd, str),
                          stdout=sys.stdout, stderr=sys.stderr)


# 1) Clone repo public
print(f"\n[1] Repo {GITHUB_USER}/{GITHUB_REPO_NAME}")
url = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO_NAME}.git"
if REPO_DIR.exists():
    subprocess.run(["git", "remote", "set-url", "origin", url], cwd=REPO_DIR, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run(["git", "fetch", "origin"], cwd=REPO_DIR)
    run(["git", "reset", "--hard", "origin/main"], cwd=REPO_DIR)
else:
    run(["git", "clone", "--branch", "main", url, str(REPO_DIR)])

# 2) Dependances
print("\n[2] Dependances")
run([sys.executable, "-m", "pip", "install", "-q",
     "monai>=1.3", "nibabel>=5.1", "SimpleITK>=2.3", "einops>=0.7",
     "scikit-image>=0.22", "scipy>=1.11"])

import torch
print(f"   torch={torch.__version__} | CUDA={torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    print(f"   GPU : {gpu}")
    if "P100" in gpu:
        print("   [warn] P100 sm_60 incompatible PyTorch cu128 -> bascule sur CPU")
        DEVICE = "cpu"
    else:
        DEVICE = "cuda"
else:
    DEVICE = "cpu"

# 3) Auto-detect dataset
print("\n[3] Auto-detect dataset")
DS_CANDIDATES = [
    "/kaggle/input/mrgrt-oar-thorax-clean-v2",
    "/kaggle/input/datasets/abdelhalimnssiri/mrgrt-oar-thorax-clean-v2",
]
DATASET = None
for cand in DS_CANDIDATES:
    if Path(cand).exists() and any(Path(cand).glob("PATIENT_s*")):
        DATASET = cand
        n_pat = sum(1 for _ in Path(cand).glob("PATIENT_s*"))
        print(f"   trouve : {cand} ({n_pat} patients)")
        break
if DATASET is None:
    sys.exit("ERREUR : dataset non trouve dans /kaggle/input/")

# 4) Symlink data/ -> dataset
data_link = REPO_DIR / "data"
if data_link.exists() or data_link.is_symlink():
    (data_link.unlink() if data_link.is_symlink() else shutil.rmtree(data_link))
data_link.symlink_to(DATASET)

# 5) Detect format
sample = next(Path(DATASET).glob("PATIENT_s*"))
img_fn = "image.nii" if (sample / "image.nii").exists() else "image.nii.gz"
lbl_fn = "label.nii" if (sample / "label.nii").exists() else "label.nii.gz"
print(f"   Format : {img_fn} / {lbl_fn}")

# 6) Auto-detect 5 checkpoints
print(f"\n[4] Recherche checkpoints {MODEL} folds {FOLDS}")
nb_root = Path("/kaggle/input/notebooks")
ckpts = {}
for fold in FOLDS:
    pattern = f"**/{MODEL}_fold{fold}/best.pt"
    found = list(nb_root.rglob(pattern)) if nb_root.exists() else []
    if found:
        ckpts[fold] = found[0]
        print(f"   fold {fold} : {found[0]}")
    else:
        print(f"   fold {fold} : NOT FOUND (skip)")
if not ckpts:
    sys.exit("ERREUR : aucun checkpoint trouve. Verifie Add Input -> Notebook Output -> <user>/<model>-fold-N.")

# 7) Lancer XAI analysis pour chaque fold
print(f"\n[5] Lancement XAI analysis ({len(ckpts)} folds, device={DEVICE})")
print("=" * 70)
out_root = Path("/kaggle/working/results/xai")
out_root.mkdir(parents=True, exist_ok=True)

for fold, ckpt in ckpts.items():
    print(f"\n--- Fold {fold} ---")
    cmd = [sys.executable, "scripts/run_xai_analysis.py",
           "--model", MODEL, "--fold", str(fold),
           "--ckpt", str(ckpt),
           "--config", "configs/default.yaml",
           "--image_filename", img_fn, "--label_filename", lbl_fn,
           "--device", DEVICE,
           "--out_dir", str(out_root)]
    if SKIP_ADEBAYO:
        cmd.append("--skip_adebayo")
    if MAX_PATIENTS:
        cmd += ["--max_patients", str(MAX_PATIENTS)]
    sys.stdout.flush()
    ret = subprocess.run(cmd, cwd=REPO_DIR, check=False)
    print(f"--- Fold {fold} exit code : {ret.returncode} ---")

# 8) Recap
print("\n=== Fichiers XAI produits ===")
for f in sorted(out_root.rglob("*.csv")):
    print(f"  {f}")
print("\n=== XAI batch FINI ===")
print("Save Version puis telecharge depuis l'onglet Output.")
