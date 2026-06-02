"""Adebayo cascading weight randomization sanity check - batch Kaggle.

Pour chaque (model in {unet, segresnet}) x (fold in 0..4):
  - Charge le checkpoint best.pt
  - Sur un sous-ensemble de N_PATIENTS_PER_FOLD patients de validation
  - Cascading randomization de la sortie vers l'entree (couches Conv3d)
  - A chaque etape on calcule la heatmap SEG-GRAD-CAM 3D (esophage par defaut)
  - On rapporte : SSIM(heatmap_step, heatmap_pristine), IoR, pointing, entropy

Sortie : results/adebayo/<model>_fold<N>_adebayo_full.csv
Colonnes : patient_id, organ, step, layer_randomized, ssim_vs_pristine,
           in_organ_ratio, pointing_accuracy, spatial_entropy

Pre-requis Inputs Kaggle:
  1. mrgrt-oar-thorax-clean-v2
  2. Notebooks outputs : <user>/<model>-fold-0..4 (10 notebooks au total)
Settings : GPU T4 x2 + Internet On.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# =====================================================================
# CONFIGURATION
# =====================================================================
GITHUB_USER = "NSSIRI"
GITHUB_REPO_NAME = "mrgrt-seg"
MODELS = ["unet", "segresnet"]      # comparaison Adebayo entre les 2 archis
FOLDS = [0, 1, 2, 3, 4]
N_PATIENTS_PER_FOLD = 5             # 5 patients/fold x 5 folds = 25 par modele
TARGET_ORGANS = [1, 2, 3, 4]        # poumon_g, poumon_d, coeur, oesophage
# Note: chaque patient x organ x layer = 1 forward+backward; ~40 conv layers => prevoir ~5h sur T4
# Pour debug rapide: TARGET_ORGANS = [4]  (oesophage seulement)
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
    DEVICE = "cpu" if "P100" in gpu else "cuda"
else:
    DEVICE = "cpu"

# 3) Dataset
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

data_link = REPO_DIR / "data"
if data_link.exists() or data_link.is_symlink():
    (data_link.unlink() if data_link.is_symlink() else shutil.rmtree(data_link))
data_link.symlink_to(DATASET)

sample = next(Path(DATASET).glob("PATIENT_s*"))
img_fn = "image.nii" if (sample / "image.nii").exists() else "image.nii.gz"
lbl_fn = "label.nii" if (sample / "label.nii").exists() else "label.nii.gz"
print(f"   Format : {img_fn} / {lbl_fn}")

# 4) Detect checkpoints pour les 2 modeles x 5 folds
print(f"\n[4] Recherche checkpoints {MODELS} x folds {FOLDS}")
nb_root = Path("/kaggle/input/notebooks")
all_ckpts = {}  # {(model, fold): path}
for model in MODELS:
    for fold in FOLDS:
        pattern = f"**/{model}_fold{fold}/best.pt"
        found = list(nb_root.rglob(pattern)) if nb_root.exists() else []
        if found:
            all_ckpts[(model, fold)] = found[0]
            print(f"   {model} fold {fold} : {found[0]}")
        else:
            print(f"   {model} fold {fold} : NOT FOUND (skip)")
if not all_ckpts:
    sys.exit("ERREUR : aucun checkpoint trouve. Verifie Add Input -> Notebook Output -> <user>/<model>-fold-N.")

# 5) Run Adebayo cascading pour chaque combo
print(f"\n[5] Adebayo cascading ({len(all_ckpts)} combos x N_PATIENTS_PER_FOLD={N_PATIENTS_PER_FOLD})")
out_root = Path("/kaggle/working/results/adebayo")
out_root.mkdir(parents=True, exist_ok=True)

for (model, fold), ckpt in all_ckpts.items():
    print(f"\n--- {model} fold {fold} ---")
    cmd = [sys.executable, "scripts/run_adebayo_analysis.py",
           "--model", model, "--fold", str(fold),
           "--ckpt", str(ckpt),
           "--config", "configs/default.yaml",
           "--image_filename", img_fn, "--label_filename", lbl_fn,
           "--device", DEVICE,
           "--n_patients", str(N_PATIENTS_PER_FOLD),
           "--target_organs", *[str(o) for o in TARGET_ORGANS],
           "--out_dir", str(out_root)]
    sys.stdout.flush()
    ret = subprocess.run(cmd, cwd=REPO_DIR, check=False)
    print(f"--- {model} fold {fold} exit code : {ret.returncode} ---")

# 6) Recap
print("\n=== Fichiers Adebayo produits ===")
for f in sorted(out_root.rglob("*.csv")):
    print(f"  {f}")
print("\n=== Adebayo batch FINI ===")
print("Save Version puis telecharge depuis l'onglet Output.")
