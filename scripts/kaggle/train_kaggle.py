"""Script unique a coller dans une cellule de notebook Kaggle.

Ce script :
  1) Recupere le token GitHub depuis Kaggle Secrets (pour repo prive)
  2) Clone (ou met a jour) le repo GitHub mrgrt-seg dans /kaggle/working/
  3) Installe les dependances Python manquantes
  4) Pointe le dossier data/ vers le dataset Kaggle monte en input
  5) Si un last.pt d'une session precedente est disponible (via "notebook output
     as input"), il est copie dans runs/<model>_fold<fold>/last.pt
  6) Lance l'entrainement avec --resume auto (et --epochs si EPOCHS != None)

Ajuster les variables ci-dessous avant le premier run.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# =====================================================================
# A AJUSTER
# =====================================================================
# --- Repo GitHub ---
GITHUB_USER = "<TON_USER_GITHUB>"   # ton username GitHub
GITHUB_REPO_NAME = "mrgrt-seg"      # nom du repo
GITHUB_BRANCH = "main"
# Pour un repo PRIVE : creer un secret Kaggle nomme GITHUB_TOKEN dans
# Add-ons -> Secrets (cf README). Le script lit ce secret automatiquement.
# Pour un repo PUBLIC : laisser GITHUB_TOKEN_SECRET_NAME = None.
GITHUB_TOKEN_SECRET_NAME = "GITHUB_TOKEN"

# --- Dataset Kaggle ---
KAGGLE_INPUT_DATASET = "/kaggle/input/mrgrt-oar-thorax"

# --- Entrainement ---
MODEL = "unet"        # "unet" ou "segresnet"
FOLD = 0              # 0..4
CONFIG = "configs/default.yaml"
# Override le nombre d'epochs (utile pour SANITY-RUN).
# Mettre 5 pour valider que tout tourne sur GPU (~30 min).
# Mettre None pour utiliser cfg.train.epochs (300 par defaut, vrai entrainement).
EPOCHS = 5

# --- Reprise (Plan B si Kaggle coupe) ---
# Si tu reprends une session : ajouter ton notebook precedent en Input et
# mettre ici son chemin (ex: "/kaggle/input/mrgrt-train-unet-fold0").
# Sinon : None.
PREVIOUS_NOTEBOOK_INPUT = None
# =====================================================================

WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / GITHUB_REPO_NAME
RUNS_DIR = WORK_DIR / "runs"
_token = None


def run(cmd, cwd=None, check=True):
    printable = " ".join(cmd) if isinstance(cmd, list) else cmd
    if _token:
        printable = printable.replace(_token, "***")
    print(f"  $ {printable}" + (f"   (cwd={cwd})" if cwd else ""))
    return subprocess.run(
        cmd, cwd=cwd, check=check, shell=isinstance(cmd, str),
        stdout=sys.stdout, stderr=sys.stderr,
    )


# 0) Token GitHub
if GITHUB_TOKEN_SECRET_NAME:
    try:
        from kaggle_secrets import UserSecretsClient
        _token = UserSecretsClient().get_secret(GITHUB_TOKEN_SECRET_NAME)
        print(f"[0] Token GitHub recupere depuis le secret '{GITHUB_TOKEN_SECRET_NAME}' (longueur={len(_token)})")
    except Exception as e:
        raise RuntimeError(
            f"Echec lecture du secret Kaggle '{GITHUB_TOKEN_SECRET_NAME}'. "
            f"Verifie qu'il existe : Add-ons -> Secrets. Erreur : {e}"
        )

if _token:
    repo_url = f"https://{GITHUB_USER}:{_token}@github.com/{GITHUB_USER}/{GITHUB_REPO_NAME}.git"
else:
    repo_url = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO_NAME}.git"

# 1) Clone / pull
print(f"\n[1] Repo : github.com/{GITHUB_USER}/{GITHUB_REPO_NAME} (branche {GITHUB_BRANCH})")
if REPO_DIR.exists():
    print(f"   Repo deja present a {REPO_DIR}, on pull.")
    subprocess.run(["git", "remote", "set-url", "origin", repo_url],
                   cwd=REPO_DIR, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run(["git", "fetch", "origin"], cwd=REPO_DIR)
    run(["git", "checkout", GITHUB_BRANCH], cwd=REPO_DIR)
    run(["git", "reset", "--hard", f"origin/{GITHUB_BRANCH}"], cwd=REPO_DIR)
else:
    run(["git", "clone", "--branch", GITHUB_BRANCH, repo_url, str(REPO_DIR)])

# 2) Dependances
print("\n[2] Installation des dependances (MONAI, nibabel, etc.)")
deps = ["monai>=1.3", "nibabel>=5.1", "SimpleITK>=2.3", "einops>=0.7",
        "pydicom>=2.4", "rt-utils>=1.2"]
run([sys.executable, "-m", "pip", "install", "-q", *deps])

import torch
print(f"   torch={torch.__version__} | CUDA dispo : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("Pas de GPU detecte ! Active 'GPU T4 x2' dans Settings.")

# 3) Symlink data
print(f"\n[3] Lien data/ -> {KAGGLE_INPUT_DATASET}")
if not Path(KAGGLE_INPUT_DATASET).exists():
    raise RuntimeError(f"Dataset non monte : {KAGGLE_INPUT_DATASET}. Ajoute-le dans Settings -> Add Input.")
data_link = REPO_DIR / "data"
if data_link.exists() or data_link.is_symlink():
    if data_link.is_symlink():
        data_link.unlink()
    else:
        shutil.rmtree(data_link)
data_link.symlink_to(KAGGLE_INPUT_DATASET)
n_patients = sum(1 for _ in Path(KAGGLE_INPUT_DATASET).glob("PATIENT_s*"))
print(f"   {n_patients} patients visibles via {data_link}")

# 4) Liens runs/ et reprise eventuelle
RUNS_DIR.mkdir(parents=True, exist_ok=True)
run_subdir = RUNS_DIR / f"{MODEL}_fold{FOLD}"
run_subdir.mkdir(parents=True, exist_ok=True)
runs_in_repo = REPO_DIR / "runs"
if runs_in_repo.exists() or runs_in_repo.is_symlink():
    if runs_in_repo.is_symlink():
        runs_in_repo.unlink()
    else:
        shutil.rmtree(runs_in_repo)
runs_in_repo.symlink_to(RUNS_DIR)

print(f"\n[4] Recherche d'un checkpoint precedent...")
if PREVIOUS_NOTEBOOK_INPUT and Path(PREVIOUS_NOTEBOOK_INPUT).exists():
    candidates = list(Path(PREVIOUS_NOTEBOOK_INPUT).rglob(f"{MODEL}_fold{FOLD}/last.pt"))
    if candidates:
        src = candidates[0]
        dst = run_subdir / "last.pt"
        shutil.copy2(src, dst)
        print(f"   Checkpoint repris : {src} -> {dst}")
        best_src = src.parent / "best.pt"
        if best_src.exists():
            shutil.copy2(best_src, run_subdir / "best.pt")
            print(f"   best.pt egalement repris.")
    else:
        print(f"   Aucun {MODEL}_fold{FOLD}/last.pt trouve dans {PREVIOUS_NOTEBOOK_INPUT} (Run from scratch).")
else:
    print("   Pas de notebook precedent fourni. Run from scratch.")

# 5) Entrainement
train_cmd = [sys.executable, "scripts/train.py",
             "--model", MODEL, "--fold", str(FOLD),
             "--config", CONFIG, "--resume", "auto"]
if EPOCHS is not None:
    train_cmd += ["--epochs", str(EPOCHS)]

print(f"\n[5] Entrainement : {' '.join(train_cmd)}")
print("=" * 60)
sys.stdout.flush()
ret = subprocess.run(train_cmd, cwd=REPO_DIR, check=False)
print("=" * 60)
print(f"\nExit code train.py : {ret.returncode}")

# 6) Recap
print(f"\n[6] Fichiers ecrits dans /kaggle/working/runs/{MODEL}_fold{FOLD}/ :")
for f in sorted(run_subdir.glob("*")):
    size_mb = f.stat().st_size / 1024**2
    print(f"   {f.name:<20} {size_mb:>8.1f} MB")

print("\nPour reprendre dans une nouvelle session :")
print("  1) Save Version (commit).")
print("  2) Settings -> Add Input -> Notebook Output -> ce notebook.")
print(f"  3) Mettre PREVIOUS_NOTEBOOK_INPUT = '/kaggle/input/<slug-de-ce-notebook>'")
print("  4) Save & Run All.")
