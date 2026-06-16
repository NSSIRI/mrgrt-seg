#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrateur Kaggle — pousse automatiquement les 10 entraînements
(5 folds x 2 modèles) répartis sur plusieurs comptes Kaggle, via le CLI `kaggle`.

A LANCER DEPUIS TON TERMINAL LOCAL (pas dans le sandbox), où :
  - `pip install kaggle` est fait,
  - chaque compte Kaggle a son kaggle.json (token API), cf. ACCOUNTS ci-dessous,
  - chaque compte est phone-verifié (Settings -> Phone Verification) pour le GPU,
  - le dataset OAR est accessible au compte (public, partagé, ou ré-uploadé),
  - (repo privé) un secret Kaggle `GITHUB_TOKEN` existe dans CHAQUE compte.

Par défaut le script est en DRY-RUN (il affiche seulement ce qu'il ferait).
Ajoute --push pour réellement créer/pousser les kernels.

Exemples :
    python scripts/kaggle/launch_all.py                # aperçu (dry-run)
    python scripts/kaggle/launch_all.py --push         # pousse tout
    python scripts/kaggle/launch_all.py --push --only unet      # seulement U-Net
    python scripts/kaggle/launch_all.py --push --epochs 5       # sanity-run 5 epochs
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# =====================================================================
# 1) A AJUSTER — comptes, dataset, repo
# =====================================================================
GITHUB_USER   = "NSSIRI"
GITHUB_SECRET = "GITHUB_TOKEN"          # nom du secret Kaggle ; None si repo public
DATASET_SLUG  = "abdelhalimnssiri/mrgrt-oar-thorax-clean-v2"
DATASET_MOUNT = "/kaggle/input/mrgrt-oar-thorax-clean-v2"   # chemin monté dans le kernel
EPOCHS_DEFAULT = 150

# Chaque compte : username Kaggle -> dossier contenant SON kaggle.json.
# (Kaggle lit le token via la variable d'env KAGGLE_CONFIG_DIR.)
ACCOUNTS = {
    "abdelhalimnssiri": str(Path.home() / ".kaggle"),          # compte principal
    # "compte2_kaggle":   str(Path.home() / ".kaggle_compte2"), # décommente et ajoute
    # "compte3_kaggle":   str(Path.home() / ".kaggle_compte3"),
}

# Quels (modèle, config) entraîner. Le U-Net "équitable" 18.8M utilise unet_fair.yaml.
MODEL_CONFIG = {
    "unet":      "configs/unet_fair.yaml",   # ~18.8 M params (capacité alignée)
    "segresnet": "configs/default.yaml",
}
FOLDS = [0, 1, 2, 3, 4]
# =====================================================================

HERE = Path(__file__).resolve().parent
TRAIN_CELL = HERE / "train_kaggle.py"
BUILD_DIR  = HERE / "_build_kernels"     # dossiers générés (un par job)


def patch_cell(src: str, *, model: str, fold: int, config: str, epochs) -> str:
    """Remplace les variables de tête de train_kaggle.py pour ce job."""
    def setv(code, name, value):
        # value est déjà une repr Python (avec guillemets si str)
        return re.sub(rf'(?m)^{name}\s*=.*$', f'{name} = {value}', code, count=1)
    src = setv(src, "GITHUB_USER", repr(GITHUB_USER))
    src = setv(src, "GITHUB_REPO_NAME", repr("mrgrt-seg"))
    src = setv(src, "GITHUB_TOKEN_SECRET_NAME", repr(GITHUB_SECRET))
    src = setv(src, "KAGGLE_INPUT_DATASET", repr(DATASET_MOUNT))
    src = setv(src, "MODEL", repr(model))
    src = setv(src, "FOLD", str(fold))
    src = setv(src, "CONFIG", repr(config))
    src = setv(src, "EPOCHS", str(epochs) if epochs is not None else "None")
    src = setv(src, "PREVIOUS_NOTEBOOK_INPUT", "None")
    return src


def build_kernel(account: str, model: str, fold: int, config: str, epochs) -> Path:
    """Génère le dossier kernel (notebook .ipynb + kernel-metadata.json)."""
    cell = patch_cell(TRAIN_CELL.read_text(encoding="utf-8"),
                      model=model, fold=fold, config=config, epochs=epochs)
    job_dir = BUILD_DIR / f"{account}__{model}_fold{fold}"
    job_dir.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None,
                   "metadata": {}, "outputs": [],
                   "source": cell.splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    (job_dir / "train_kaggle.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {
        "id": f"{account}/mrgrt-train-{model}-fold{fold}",
        "title": f"MRgRT train {model} fold{fold}",
        "code_file": "train_kaggle.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [DATASET_SLUG],
        "competition_sources": [], "kernel_sources": [],
    }
    (job_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return job_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="pousse réellement (sinon dry-run)")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT,
                    help="nb d'epochs (5 = sanity-run)")
    ap.add_argument("--only", choices=["unet", "segresnet"], default=None)
    args = ap.parse_args()

    if not TRAIN_CELL.exists():
        sys.exit(f"ERREUR : {TRAIN_CELL} introuvable.")
    if not ACCOUNTS:
        sys.exit("ERREUR : renseigne au moins un compte dans ACCOUNTS.")

    models = [args.only] if args.only else list(MODEL_CONFIG)
    jobs = [(m, f) for m in models for f in FOLDS]
    accounts = list(ACCOUNTS)

    print(f"{len(jobs)} jobs à répartir sur {len(accounts)} compte(s) "
          f"| epochs={args.epochs} | {'PUSH' if args.push else 'DRY-RUN'}\n")

    for i, (model, fold) in enumerate(jobs):
        account = accounts[i % len(accounts)]          # round-robin
        config = MODEL_CONFIG[model]
        job_dir = build_kernel(account, model, fold, config, args.epochs)
        kid = f"{account}/mrgrt-train-{model}-fold{fold}"
        env = dict(os.environ, KAGGLE_CONFIG_DIR=ACCOUNTS[account])
        cmd = ["kaggle", "kernels", "push", "-p", str(job_dir)]
        print(f"[{i+1:>2}/{len(jobs)}] {account:<20} {model}_fold{fold}  ->  {kid}")
        print(f"        KAGGLE_CONFIG_DIR={ACCOUNTS[account]}")
        print(f"        $ {' '.join(cmd)}   (config={config})")
        if args.push:
            r = subprocess.run(cmd, env=env)
            print(f"        exit={r.returncode}")
        print()

    print("Suivi des runs :  kaggle kernels status <account>/mrgrt-train-<model>-fold<k>")
    if not args.push:
        print("\n(DRY-RUN — relance avec --push pour exécuter réellement.)")


if __name__ == "__main__":
    main()
