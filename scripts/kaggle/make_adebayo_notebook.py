"""Genere un notebook Kaggle adebayo_kaggle.ipynb + kernel-metadata.json
proprement (UTF-8 sans BOM) pour le batch Adebayo cascading randomization.

Usage :
    # Run unique : 2 modeles x 5 folds dans un seul notebook
    python make_adebayo_notebook.py --username nssiri02 \\
        --out_dir C:\\tmp\\kaggle_adebayo

    # Run separe par modele (pour repartir entre 2 comptes Kaggle)
    python make_adebayo_notebook.py --username abdelhalimnssiri --model unet \\
        --out_dir C:\\tmp\\kaggle_adebayo_unet
    python make_adebayo_notebook.py --username nssiri02 --model segresnet \\
        --out_dir C:\\tmp\\kaggle_adebayo_segres
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--username", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", choices=["unet", "segresnet", "both"], default="both",
                   help="Filtre le batch sur un seul modele (defaut: les 2)")
    p.add_argument("--adebayo_py", default=None,
                   help="Chemin vers adebayo_kaggle.py (defaut: meme dossier)")
    p.add_argument("--n_patients", type=int, default=5,
                   help="Nb de patients par fold (defaut 5; 2 pour dry-run rapide)")
    p.add_argument("--target_organs", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="Organes (1=poumon_g, 2=poumon_d, 3=coeur, 4=oesophage). "
                        "Pour dry-run rapide : --target_organs 4 (oesophage seul)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.adebayo_py:
        ada_py = Path(args.adebayo_py)
    else:
        ada_py = Path(__file__).parent / "adebayo_kaggle.py"
    if not ada_py.exists():
        sys.exit(f"ERREUR : adebayo_kaggle.py absent : {ada_py}")
    code = ada_py.read_text(encoding="utf-8")

    # Override MODELS, N_PATIENTS_PER_FOLD, TARGET_ORGANS
    if args.model == "unet":
        code = code.replace('MODELS = ["unet", "segresnet"]', 'MODELS = ["unet"]')
    elif args.model == "segresnet":
        code = code.replace('MODELS = ["unet", "segresnet"]', 'MODELS = ["segresnet"]')

    code = code.replace("N_PATIENTS_PER_FOLD = 5",
                        f"N_PATIENTS_PER_FOLD = {args.n_patients}")

    organs_str = "[" + ", ".join(str(o) for o in args.target_organs) + "]"
    code = code.replace("TARGET_ORGANS = [1, 2, 3, 4]",
                        f"TARGET_ORGANS = {organs_str}")

    # Notebook avec UNE cellule de code
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb_path = out_dir / "adebayo_kaggle.ipynb"
    nb_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False),
                       encoding="utf-8")
    print(f"OK : {nb_path}")

    # Kernel-metadata.json
    model_slug = args.model if args.model != "both" else "all"
    kernel_id = f"{args.username}/mrgrt-adebayo-{model_slug}-batch"
    title = f"MRgRT Adebayo Cascading ({model_slug})"

    # Notebook sources : tous les training notebooks dont on aura besoin
    if args.model == "unet":
        models_for_srcs = ["unet"]
    elif args.model == "segresnet":
        models_for_srcs = ["segresnet"]
    else:
        models_for_srcs = ["unet", "segresnet"]
    notebook_srcs = [f"{args.username}/{m}-fold-{i}"
                     for m in models_for_srcs for i in range(5)]

    meta = {
        "id": kernel_id,
        "title": title,
        "code_file": "adebayo_kaggle.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": ["abdelhalimnssiri/mrgrt-oar-thorax-clean-v2"],
        "competition_sources": [],
        "kernel_sources": notebook_srcs,
    }

    meta_path = out_dir / "kernel-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"OK : {meta_path}")
    print(f"\nKernel ID : {kernel_id}")
    print(f"Title     : {title}")
    print(f"N patients/fold : {args.n_patients}")
    print(f"Target organs   : {args.target_organs}")
    print(f"Sources         :")
    for s in notebook_srcs:
        print(f"   - {s}")


if __name__ == "__main__":
    main()
