"""Evaluation d'un checkpoint entraine sur le fold de validation correspondant.

Calcule par patient et par OAR : DSC, IoU, HD95, Surface DSC (2 mm).
Sauvegarde un CSV par-patient (unite statistique = le patient) directement
utilisable pour les box plots et les tests de Wilcoxon de l'article.

Exemples :
    python scripts/evaluate.py --model unet --fold 0 \
        --ckpt runs/unet_fold0/best.pt --out results/unet_fold0_metrics.csv

    # Agreger les 5 folds d'un modele :
    for f in 0 1 2 3 4; do
        python scripts/evaluate.py --model unet --fold $f \
            --ckpt runs/unet_fold$f/best.pt --out results/unet_fold$f_metrics.csv
    done
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import (list_patients, get_val_transforms, make_loaders,
                              get_train_transforms)
from src.data.splits import make_5fold_splits, load_fold
from src.models.factory import build_model
from src.eval.metrics import build_metrics, post_processors

from monai.inferers import sliding_window_inference


CLASS_NAMES = ["poumon_g", "poumon_d", "coeur", "oesophage"]  # classes 1..4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "segresnet"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default=None,
                        help="Chemin du checkpoint. Defaut: runs/<model>_fold<fold>/best.pt")
    parser.add_argument("--data_root", default=None, help="Override config.data.root")
    parser.add_argument("--image_filename", default=None,
                        help="Override config.data.image_filename (ex: image.nii.gz en local)")
    parser.add_argument("--label_filename", default=None,
                        help="Override config.data.label_filename (ex: label.nii.gz en local)")
    parser.add_argument("--out", default=None,
                        help="Chemin du CSV de sortie. Defaut: results/<model>_fold<fold>_metrics.csv")
    parser.add_argument("--device", default=None, help="cuda ou cpu (auto si non specifie)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.image_filename:
        cfg["data"]["image_filename"] = args.image_filename
    if args.label_filename:
        cfg["data"]["label_filename"] = args.label_filename

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["data"]["num_classes"]

    ckpt_path = Path(args.ckpt) if args.ckpt else \
        Path(cfg["experiment"]["output_dir"]) / f"{args.model}_fold{args.fold}" / "best.pt"
    if not ckpt_path.exists():
        sys.exit(f"ERREUR : checkpoint introuvable : {ckpt_path}")

    out_path = Path(args.out) if args.out else \
        ROOT / "results" / f"{args.model}_fold{args.fold}_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Reconstruire EXACTEMENT le meme split que l'entrainement ---
    items = list_patients(cfg["data"]["root"],
                          cfg["data"]["image_filename"],
                          cfg["data"]["label_filename"])
    folds = make_5fold_splits(items, n_folds=cfg["splits"]["n_folds"],
                              seed=cfg["experiment"]["seed"])
    _, val_items = load_fold(folds, args.fold, items)
    print(f"Fold {args.fold} : {len(val_items)} patients de validation")

    modality = cfg["data"].get("modality", "mri")
    intensity_params = cfg["data"].get("intensity_params") or (
        {"clip_percentiles": cfg["data"].get("intensity_clip_percentiles", [0.5, 99.5])}
        if modality == "mri" else
        {"hu_window": cfg["data"].get("hu_window", [-1000.0, 400.0])}
    )
    val_tf = get_val_transforms(spacing=cfg["data"]["spacing"],
                                intensity_params=intensity_params, modality=modality)
    # train_tf factice (make_loaders en exige un) ; cache_rate=0 pour l'eval
    dummy_train_tf = val_tf
    _, val_loader = make_loaders(
        val_items, val_items, dummy_train_tf, val_tf,
        batch_size=cfg["train"]["batch_size"], num_workers=2, cache_rate=0.0,
    )

    # --- Modele + checkpoint ---
    model = build_model(
        name=args.model,
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        features=cfg["model"]["features"],
        segresnet_kwargs=cfg["model"].get("segresnet"),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Checkpoint charge : {ckpt_path}"
          + (f" (epoch {ckpt.get('epoch', '?')}, val_dsc {ckpt.get('val_dsc', '?')})"
             if isinstance(ckpt, dict) else ""))

    # --- Metriques (par patient, reduction='none') ---
    metrics = build_metrics(num_classes, surface_tol_mm=cfg["eval"].get("surface_dsc_tolerance_mm", 2.0))
    post_pred, post_label = post_processors(num_classes)
    patch_size = cfg["data"]["patch_size"]
    overlap = cfg["eval"].get("sliding_window_overlap", 0.5)

    rows = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            pid = val_items[i]["patient_id"]
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = sliding_window_inference(
                inputs=x, roi_size=patch_size, sw_batch_size=2,
                predictor=model, overlap=overlap, mode="gaussian",
            )
            pred = post_pred(logits[0]).unsqueeze(0)
            lbl = post_label(y[0]).unsqueeze(0)

            row = {"patient_id": pid}
            for mname, metric in metrics.items():
                metric.reset()
                metric(y_pred=pred, y=lbl)
                vals = metric.aggregate()  # shape [1, num_classes-1]
                vals = vals.cpu().numpy().flatten()
                for c, cname in enumerate(CLASS_NAMES):
                    v = float(vals[c]) if c < len(vals) else float("nan")
                    row[f"{mname}_{cname}"] = v
            rows.append(row)
            print(f"  [{i+1}/{len(val_items)}] {pid} : "
                  f"DSC moy = {np.nanmean([row[f'dsc_{c}'] for c in CLASS_NAMES]):.3f}")

    # --- Sauvegarde CSV ---
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    print(f"\nCSV ecrit : {out_path} ({len(rows)} patients)")

    # --- Recap moyennes ---
    print("\n=== Moyennes (sur les patients du fold) ===")
    for mname in metrics:
        for cname in CLASS_NAMES:
            col = [r[f"{mname}_{cname}"] for r in rows]
            print(f"  {mname:>12} {cname:<11}: {np.nanmean(col):.3f} +/- {np.nanstd(col):.3f}")


if __name__ == "__main__":
    main()
