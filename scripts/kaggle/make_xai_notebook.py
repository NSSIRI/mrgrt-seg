"""Genere un notebook Kaggle xai_kaggle.ipynb + kernel-metadata.json
proprement (UTF-8 sans BOM) pour le batch XAI 5 folds.

Usage :
    python make_xai_notebook.py --model segresnet --username nssiri02 \
        --out_dir C:\\tmp\\kaggle_xai_segresnet
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
    p.add_argument("--xai_py", default=None,
                   help="Chemin vers xai_kaggle.py (defaut: meme dossier)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lire xai_kaggle.py en UTF-8
    if args.xai_py:
        xai_py = Path(args.xai_py)
    else:
        xai_py = Path(__file__).parent / "xai_kaggle.py"
    if not xai_py.exists():
        sys.exit(f"ERREUR : xai_kaggle.py absent : {xai_py}")
    code = xai_py.read_text(encoding="utf-8")

    # Forcer MODEL = "..." (defaut "segresnet" dans xai_kaggle.py)
    if 'MODEL = "segresnet"' in code:
        code = code.replace('MODEL = "segresnet"', f'MODEL = "{args.model}"', 1)
    elif 'MODEL = "unet"' in code:
        code = code.replace('MODEL = "unet"', f'MODEL = "{args.model}"', 1)
    else:
        print("[warn] Ligne MODEL non trouvee, on continue quand meme")

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

    nb_path = out_dir / "xai_kaggle.ipynb"
    nb_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK : {nb_path}")

    # Kernel-metadata.json
    kernel_id = f"{args.username}/mrgrt-xai-{args.model}-batch"
    title = f"MRgRT XAI {args.model.capitalize()} Batch"
    # Slug convention : <username>/<model>-fold-<N>
    notebook_srcs = [f"{args.username}/{args.model}-fold-{i}" for i in range(5)]

    meta = {
        "id": kernel_id,
        "title": title,
        "code_file": "xai_kaggle.ipynb",
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
