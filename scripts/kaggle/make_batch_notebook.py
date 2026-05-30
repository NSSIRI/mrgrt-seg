"""Genere un notebook Kaggle batch_eval.ipynb + kernel-metadata.json
proprement (UTF-8 sans BOM, JSON garanti correct via nbformat/json).

Usage :
    python make_batch_notebook.py --model unet --username abdelhalimnssiri \
        --out_dir C:\\tmp\\kaggle_batch_eval_unet
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["unet", "segresnet"])
    p.add_argument("--username", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_py", default=None,
                   help="Chemin vers batch_eval.py (defaut: meme dossier)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lire batch_eval.py en UTF-8
    if args.batch_py:
        batch_py = Path(args.batch_py)
    else:
        batch_py = Path(__file__).parent / "batch_eval.py"
    if not batch_py.exists():
        sys.exit(f"ERREUR : batch_eval.py absent : {batch_py}")
    code = batch_py.read_text(encoding="utf-8")

    # Forcer MODEL = "..."
    if 'MODEL = "unet"' in code:
        code = code.replace('MODEL = "unet"', f'MODEL = "{args.model}"', 1)
    elif 'MODEL = "segresnet"' in code:
        code = code.replace('MODEL = "segresnet"', f'MODEL = "{args.model}"', 1)
    else:
        print("[warn] Ligne MODEL non trouvee, on continue quand meme")

    # Construire le notebook .ipynb proprement
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
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb_path = out_dir / "batch_eval.ipynb"
    nb_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK : {nb_path}")

    # Kernel-metadata.json
    # Slug convention : {username}/{model}-fold-{N} pour les 5 folds
    kernel_id = f"{args.username}/mrgrt-eval-{args.model}-batch"
    title = f"MRgRT Eval {args.model.capitalize()} Batch"
    notebook_srcs = [f"{args.username}/{args.model}-fold-{i}" for i in range(5)]

    meta = {
        "id": kernel_id,
        "title": title,
        "code_file": "batch_eval.ipynb",
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
    print(f"Sources   :")
    for s in notebook_srcs:
        print(f"   - {s}")


if __name__ == "__main__":
    main()
