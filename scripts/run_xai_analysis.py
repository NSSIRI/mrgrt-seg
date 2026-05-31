"""Runner XAI : calcule les heatmaps SEG-GRAD-CAM 3D + 3 metriques + Adebayo
pour un modele entraine, sur tous les patients d'un fold de validation.

Produit :
  - results/xai/<model>_fold<N>_xai_per_patient.csv : metriques par patient/OAR
  - results/xai/<model>_fold<N>_adebayo.csv         : courbe SSIM cascading
  - results/xai/<model>_fold<N>_examples/           : 3 heatmaps qualitatives (npy)

Usage :
  python scripts/run_xai_analysis.py --model unet --fold 0 \
      --ckpt runs/unet_fold0/best.pt --out_dir results/xai
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Permettre l'import depuis n'importe ou
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Patch cuCIM avant tout import MONAI metric (cas Kaggle cu128)
try:
    import monai.metrics.utils as _mmu
    import scipy.ndimage as _scipy_ndi
    def _scipy_erosion(arr, *a, **k):
        if hasattr(arr, "get"): arr = arr.get()
        if hasattr(arr, "cpu"): arr = arr.cpu().numpy()
        return _scipy_ndi.binary_erosion(np.asarray(arr).astype(bool))
    _mmu.cucim_binary_erosion = _scipy_erosion
    if hasattr(_mmu, "has_cucim"):
        _mmu.has_cucim = False
except Exception:
    pass

from src.data.dataset import list_patients, get_val_transforms, make_loaders
from src.data.splits import make_5fold_splits, load_fold
from src.models.factory import build_model
from src.xai.grad_cam import SegGradCAM3D
from src.xai.metrics import compute_all_metrics
from src.xai.adebayo import cascading_weight_randomization

from monai.inferers import sliding_window_inference


CLASS_NAMES = ["poumon_g", "poumon_d", "coeur", "oesophage"]  # classes 1..4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["unet", "segresnet"], required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--ckpt", required=True, help="Chemin du best.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--image_filename", default=None)
    ap.add_argument("--label_filename", default=None)
    ap.add_argument("--out_dir", default="results/xai")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_patients", type=int, default=None,
                    help="Limite le nombre de patients evalues (debug)")
    ap.add_argument("--skip_adebayo", action="store_true",
                    help="Saute le sanity check Adebayo (long)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root: cfg["data"]["root"] = args.data_root
    if args.image_filename: cfg["data"]["image_filename"] = args.image_filename
    if args.label_filename: cfg["data"]["label_filename"] = args.label_filename

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["data"]["num_classes"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruire le split fold N
    items = list_patients(cfg["data"]["root"],
                          cfg["data"]["image_filename"],
                          cfg["data"]["label_filename"])
    folds = make_5fold_splits(items, n_folds=cfg["splits"]["n_folds"],
                              seed=cfg["experiment"]["seed"])
    _, val_items = load_fold(folds, args.fold, items)
    if args.max_patients:
        val_items = val_items[:args.max_patients]
    print(f"Fold {args.fold} : {len(val_items)} patients de validation")

    modality = cfg["data"].get("modality", "mri")
    intensity_params = cfg["data"].get("intensity_params") or (
        {"clip_percentiles": cfg["data"].get("intensity_clip_percentiles", [0.5, 99.5])}
        if modality == "mri" else {"hu_window": cfg["data"].get("hu_window", [-1000.0, 400.0])}
    )
    val_tf = get_val_transforms(spacing=cfg["data"]["spacing"],
                                intensity_params=intensity_params, modality=modality)
    dummy_train_tf = val_tf
    _, val_loader = make_loaders(val_items, val_items, dummy_train_tf, val_tf,
                                 batch_size=1, num_workers=0, cache_rate=0.0)

    # Modele
    model = build_model(
        name=args.model,
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        features=cfg["model"]["features"],
        segresnet_kwargs=cfg["model"].get("segresnet"),
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Checkpoint charge : {args.ckpt}")

    # GradCAM
    cam = SegGradCAM3D(model, target_layer="auto")
    patch_size = cfg["data"]["patch_size"]

    # ----- 1) Boucle patients : 3 metriques XAI par OAR -----
    rows = []
    saved_examples = 0
    examples_dir = out_dir / f"{args.model}_fold{args.fold}_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    def center_crop_to_patch(x_tensor, y_arr, patch):
        """Center-crop l'image et le label a la patch_size (128x128x64).
        Si plus petit que patch dans un axe, pad avec des zeros.
        Retourne (x_cropped_tensor, y_cropped_array, slices).
        """
        D, H, W = x_tensor.shape[-3:]
        pd, ph, pw = patch
        x_np = x_tensor[0, 0].cpu().numpy()
        # Pad si plus petit que patch
        pad = [(0, max(0, pd - D)), (0, max(0, ph - H)), (0, max(0, pw - W))]
        if any(p[1] > 0 for p in pad):
            x_np = np.pad(x_np, pad, mode="constant", constant_values=0)
            y_arr = np.pad(y_arr, pad, mode="constant", constant_values=0)
        D2, H2, W2 = x_np.shape
        # Center crop
        sd = (D2 - pd) // 2
        sh = (H2 - ph) // 2
        sw = (W2 - pw) // 2
        slices = (slice(sd, sd + pd), slice(sh, sh + ph), slice(sw, sw + pw))
        x_cropped = x_np[slices]
        y_cropped = y_arr[slices]
        x_t = torch.from_numpy(x_cropped).float().unsqueeze(0).unsqueeze(0).to(device)
        return x_t, y_cropped, slices

    for i, batch in enumerate(val_loader):
        pid = val_items[i]["patient_id"]
        x_full = batch["image"].to(device)
        y_full = batch["label"][0, 0].cpu().numpy().astype(np.int32)

        # CRITIQUE : center-crop a patch_size pour eviter shape mismatch + OOM
        try:
            x, y, _ = center_crop_to_patch(x_full, y_full, tuple(patch_size))
        except Exception as e:
            print(f"  [{i+1}/{len(val_items)}] {pid} : crop FAIL ({e})")
            continue

        row = {"patient_id": pid}
        for c, cname in enumerate(CLASS_NAMES, start=1):
            gt_mask = (y == c)
            if not gt_mask.any():
                for m in ("in_organ_ratio", "pointing_accuracy", "spatial_entropy", "organ_baseline"):
                    row[f"{m}_{cname}"] = float("nan")
                continue

            try:
                heat = cam.compute(x, target_class=c, patch_size=tuple(patch_size))
                if isinstance(heat, torch.Tensor):
                    heat = heat.cpu().numpy()
                # Libere memoire GPU apres chaque forward+backward
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if heat.shape != gt_mask.shape:
                    h_t = torch.from_numpy(heat).float().unsqueeze(0).unsqueeze(0)
                    h_t = F.interpolate(h_t, size=gt_mask.shape, mode="trilinear", align_corners=False)
                    heat = h_t[0, 0].numpy()
            except Exception as e:
                print(f"  [{i+1}/{len(val_items)}] {pid} {cname} : ECHEC heatmap ({e})")
                for m in ("in_organ_ratio", "pointing_accuracy", "spatial_entropy", "organ_baseline"):
                    row[f"{m}_{cname}"] = float("nan")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            m = compute_all_metrics(heat, gt_mask)
            for k, v in m.items():
                row[f"{k}_{cname}"] = v

            # Sauve 3 exemples qualitatifs (3 premiers patients avec esophage GT)
            if saved_examples < 3 and cname == "oesophage":
                np.save(examples_dir / f"{pid}_{cname}_heatmap.npy", heat.astype(np.float32))
                np.save(examples_dir / f"{pid}_{cname}_gtmask.npy", gt_mask.astype(np.uint8))
                np.save(examples_dir / f"{pid}_image_crop.npy", x[0, 0].detach().cpu().numpy().astype(np.float32))
                saved_examples += 1

        rows.append(row)
        print(f"  [{i+1}/{len(val_items)}] {pid} : iom_g={row.get('in_organ_ratio_poumon_g', float('nan')):.3f}  iom_d={row.get('in_organ_ratio_poumon_d', float('nan')):.3f}  iom_h={row.get('in_organ_ratio_coeur', float('nan')):.3f}  iom_o={row.get('in_organ_ratio_oesophage', float('nan')):.3f}")

    # CSV per patient
    out_csv = out_dir / f"{args.model}_fold{args.fold}_xai_per_patient.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nCSV XAI per patient : {out_csv} ({len(rows)} patients)")

    # Recap moyennes
    print("\n=== Moyennes XAI ===")
    for m in ("in_organ_ratio", "pointing_accuracy", "spatial_entropy", "organ_baseline"):
        for cname in CLASS_NAMES:
            col = f"{m}_{cname}"
            vals = [r[col] for r in rows if not np.isnan(r.get(col, float("nan")))]
            if vals:
                print(f"  {m:>18} {cname:<11} : {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    # ----- 2) Adebayo cascading (1 seul patient pour gain de temps) -----
    if not args.skip_adebayo and len(val_items) > 0:
        print("\n[Adebayo] Cascading weight randomization (1 patient test)...")
        # Re-charger le batch du premier patient et center-crop
        first_batch = next(iter(val_loader))
        y_first = first_batch["label"][0, 0].cpu().numpy().astype(np.int32)
        x_test, _, _ = center_crop_to_patch(first_batch["image"].to(device), y_first, tuple(patch_size))
        target_c = 4  # esophage : le plus instructif
        cam.cleanup()

        def compute_saliency_of(m: torch.nn.Module) -> np.ndarray:
            local_cam = SegGradCAM3D(m, target_layer="auto")
            try:
                h = local_cam.compute(x_test, target_class=target_c, patch_size=tuple(patch_size))
                arr = h.cpu().numpy() if isinstance(h, torch.Tensor) else h
            finally:
                local_cam.cleanup()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return arr

        results = cascading_weight_randomization(model, compute_saliency_of, seed=42)
        ade_csv = out_dir / f"{args.model}_fold{args.fold}_adebayo.csv"
        with open(ade_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"CSV Adebayo : {ade_csv}")
        print(f"   SSIM pristine={results[0]['ssim_vs_pristine']:.3f} -> "
              f"all-random={results[-1]['ssim_vs_pristine']:.3f}")

    print("\n=== XAI analysis FINI ===")


if __name__ == "__main__":
    main()
