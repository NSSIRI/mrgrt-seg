"""Script XAI a coller dans un notebook Kaggle.

Genere les cartes SEG-GRAD-CAM 3D et le sanity check (cascading randomization,
Adebayo 2018) pour un checkpoint entraine. Produit les visualisations PNG
+ CSV de metriques + CSV du sanity check.

PRE-REQUIS dans le notebook Kaggle (Settings -> Add Input) :
  1. Dataset : mrgrt-oar-thorax-clean-v2 (les 187 patients)
  2. Notebook Output du training (contient runs/<model>_fold<N>/best.pt)
  3. Internet ON + Secret GITHUB_TOKEN (sinon repo public)

Ajuster les variables ci-dessous puis Save & Run All.
"""
import os, shutil, subprocess, sys
from pathlib import Path

# ============== A AJUSTER ==============
GITHUB_USER = "<TON_USER_GITHUB>"
GITHUB_REPO_NAME = "mrgrt-seg"
GITHUB_BRANCH = "main"
GITHUB_TOKEN_SECRET_NAME = "GITHUB_TOKEN"

KAGGLE_INPUT_DATASET = "/kaggle/input/mrgrt-oar-thorax-clean-v2"
CHECKPOINT_INPUT = "/kaggle/input/mrgrt-train-segresnet-fold0"  # slug du training

MODEL = "segresnet"
FOLD = 0
CONFIG = "configs/default.yaml"
N_PATIENTS = 3  # nombre de patients pour les visualisations + sanity check
# =======================================

WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / GITHUB_REPO_NAME
_token = None


def run(cmd, cwd=None, check=True):
    printable = " ".join(cmd) if isinstance(cmd, list) else cmd
    if _token: printable = printable.replace(_token, "***")
    print(f"  $ {printable}" + (f"   (cwd={cwd})" if cwd else ""))
    return subprocess.run(cmd, cwd=cwd, check=check, shell=isinstance(cmd, str),
                          stdout=sys.stdout, stderr=sys.stderr)


# 0) Token GitHub (facultatif)
if GITHUB_TOKEN_SECRET_NAME:
    try:
        from kaggle_secrets import UserSecretsClient
        _token = UserSecretsClient().get_secret(GITHUB_TOKEN_SECRET_NAME)
        print(f"[0] Token GitHub recupere (longueur={len(_token)})")
    except Exception as e:
        print(f"[0] Pas de secret '{GITHUB_TOKEN_SECRET_NAME}' ({type(e).__name__}). "
              f"Clone sans auth (le repo doit etre public).")
        _token = None

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

# 2) Dependances (scikit-image pour SSIM 3D du sanity check)
print("\n[2] Dependances")
run([sys.executable, "-m", "pip", "install", "-q",
     "monai>=1.3", "nibabel>=5.1", "SimpleITK>=2.3", "einops>=0.7",
     "scikit-image>=0.22", "scipy>=1.11"])

import torch
print(f"   torch={torch.__version__} | CUDA={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU : {torch.cuda.get_device_name(0)}")

# 3) Lien data
print(f"\n[3] Lien data/ -> {KAGGLE_INPUT_DATASET}")
if not Path(KAGGLE_INPUT_DATASET).exists():
    raise RuntimeError(f"Dataset non monte : {KAGGLE_INPUT_DATASET}")
data_link = REPO_DIR / "data"
if data_link.exists() or data_link.is_symlink():
    (data_link.unlink() if data_link.is_symlink() else shutil.rmtree(data_link))
data_link.symlink_to(KAGGLE_INPUT_DATASET)
n = sum(1 for _ in Path(KAGGLE_INPUT_DATASET).glob("PATIENT_s*"))
print(f"   {n} patients visibles")

# 4) Localiser best.pt
print(f"\n[4] Recherche du checkpoint dans {CHECKPOINT_INPUT}")
if not Path(CHECKPOINT_INPUT).exists():
    raise RuntimeError(f"Output training non monte : {CHECKPOINT_INPUT}")
cands = list(Path(CHECKPOINT_INPUT).rglob(f"{MODEL}_fold{FOLD}/best.pt"))
if not cands:
    cands = list(Path(CHECKPOINT_INPUT).rglob("best.pt"))
if not cands:
    raise RuntimeError(f"Aucun best.pt dans {CHECKPOINT_INPUT}")
ckpt = cands[0]
print(f"   Checkpoint : {ckpt}")

# 5) Detecter format fichiers
sample = next(Path(KAGGLE_INPUT_DATASET).glob("PATIENT_s*"))
img_fn = "image.nii" if (sample / "image.nii").exists() else "image.nii.gz"
lbl_fn = "label.nii" if (sample / "label.nii").exists() else "label.nii.gz"
print(f"   Format detecte : {img_fn} / {lbl_fn}")
# Override la config pour pointer sur le bon format (le repo a peut-etre image.nii)
import yaml
cfg_path = REPO_DIR / CONFIG
cfg = yaml.safe_load(cfg_path.read_text())
cfg["data"]["image_filename"] = img_fn
cfg["data"]["label_filename"] = lbl_fn
cfg_path.write_text(yaml.dump(cfg))
print(f"   Config mise a jour avec les bons noms de fichiers")

# 6) Lancer XAI
out_dir = f"/kaggle/working/xai_{MODEL}_fold{FOLD}"
cmd = [sys.executable, "scripts/xai_analysis.py",
       "--model", MODEL, "--fold", str(FOLD), "--config", CONFIG,
       "--ckpt", str(ckpt),
       "--n_patients", str(N_PATIENTS),
       "--device", "cuda" if torch.cuda.is_available() else "cpu",
       "--out_dir", out_dir]
print(f"\n[5] XAI analysis : {' '.join(cmd)}")
print("=" * 60); sys.stdout.flush()
ret = subprocess.run(cmd, cwd=REPO_DIR, check=False)
print("=" * 60)
print(f"\nExit code : {ret.returncode}")

# 7) Recap des fichiers produits
print(f"\n[6] Fichiers produits dans {out_dir} :")
for f in sorted(Path(out_dir).glob("*")):
    sz = f.stat().st_size / 1024
    print(f"   {f.name:<40} {sz:>8.1f} KB")

print("\nApres Save Version, telecharge depuis l'onglet Output :")
print(f"  - xai_metrics.csv         (localization, pointing, sparsity)")
print(f"  - sanity_check_ssim.csv   (cascading randomization)")
print(f"  - *.png                   (visualisations Figure 4)")
