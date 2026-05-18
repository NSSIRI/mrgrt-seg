"""Script d'entrainement d'un fold pour un modele donne.

Exemples :
    python scripts/train.py --model unet --fold 0 --config configs/default.yaml
    python scripts/train.py --model unet --fold 0 --resume            # auto-detect last.pt
    python scripts/train.py --model unet --fold 0 --resume runs/x/last.pt
    python scripts/train.py --model unet --fold 0 --epochs 5          # sanity-run
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import (list_patients, get_train_transforms,
                              get_val_transforms, make_loaders)
from src.data.splits import make_5fold_splits, load_fold
from src.models.factory import build_model, count_parameters
from src.train.losses import build_loss
from src.train.trainer import Trainer


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "segresnet"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_root", default=None,
                        help="Override config.data.root.")
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        help=("Reprendre depuis un checkpoint. Sans argument : cherche last.pt "
              "dans le output_dir du run. Avec un chemin : utilise ce fichier."),
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=("Override cfg.train.epochs. Utile pour un sanity-run rapide "
              "(--epochs 5) sans toucher au YAML."),
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.epochs is not None:
        print(f"Override : epochs {cfg['train']['epochs']} -> {args.epochs}")
        cfg["train"]["epochs"] = args.epochs
    set_seed(cfg["experiment"]["seed"])

    items = list_patients(cfg["data"]["root"],
                          cfg["data"]["image_filename"],
                          cfg["data"]["label_filename"])
    folds = make_5fold_splits(items, n_folds=cfg["splits"]["n_folds"],
                              seed=cfg["experiment"]["seed"])
    train_items, val_items = load_fold(folds, args.fold, items)
    print(f"Fold {args.fold} : train={len(train_items)} | val={len(val_items)}")

    modality = cfg["data"].get("modality", "mri")
    intensity_params = cfg["data"].get("intensity_params") or (
        {"clip_percentiles": cfg["data"].get("intensity_clip_percentiles",
                                              [0.5, 99.5])}
        if modality == "mri" else
        {"hu_window": cfg["data"].get("hu_window", [-1000.0, 400.0])}
    )
    train_tf = get_train_transforms(
        spacing=cfg["data"]["spacing"],
        intensity_params=intensity_params,
        patch_size=cfg["data"]["patch_size"],
        modality=modality,
    )
    val_tf = get_val_transforms(
        spacing=cfg["data"]["spacing"],
        intensity_params=intensity_params,
        modality=modality,
    )
    train_loader, val_loader = make_loaders(
        train_items, val_items, train_tf, val_tf,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        cache_rate=cfg["data"]["cache_rate"],
    )

    model = build_model(
        name=args.model,
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        features=cfg["model"]["features"],
        segresnet_kwargs=cfg["model"].get("segresnet"),
    )
    print(f"Modele {args.model} | parametres : {count_parameters(model):,}")

    loss_fn = build_loss(cfg["train"]["loss"])
    train_cfg = {
        **cfg["train"],
        "patch_size": cfg["data"]["patch_size"],
        "sliding_window_overlap": cfg["eval"]["sliding_window_overlap"],
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        num_classes=cfg["data"]["num_classes"],
        cfg=train_cfg, device=device,
    )

    output_dir = Path(cfg["experiment"]["output_dir"]) / \
        f"{args.model}_fold{args.fold}"

    # Resolution de --resume :
    #   None         -> pas de reprise (run from scratch)
    #   "auto"       -> last.pt dans output_dir si present, sinon from scratch
    #   "<chemin>"   -> charge ce chemin (erreur si absent)
    resume_path = None
    if args.resume == "auto":
        candidate = output_dir / "last.pt"
        if candidate.exists():
            resume_path = candidate
            print(f"Auto-resume : {candidate}")
        else:
            print(f"Auto-resume : pas de last.pt dans {output_dir}, run from scratch.")
    elif args.resume is not None:
        resume_path = Path(args.resume)

    history = trainer.fit(
        epochs=cfg["train"]["epochs"],
        output_dir=output_dir,
        resume_from=resume_path,
    )
    np.savez(output_dir / "history.npz",
             **{k: np.asarray(v) for k, v in history.items()})
    print(f"Entrainement termine. Best : {output_dir / 'best.pt'} | "
          f"Last : {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
