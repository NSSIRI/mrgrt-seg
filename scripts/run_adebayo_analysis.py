"""Runner Adebayo cascading weight randomization (extended).

Pour un fold donne et un modele entraine, boucle sur N patients de validation
et, pour chaque patient + chaque organe cible :
  - calcule la heatmap pristine
  - randomise progressivement les couches Conv3d de la sortie vers l'entree
  - apres chaque randomisation, recalcule la heatmap et reporte
    (SSIM vs pristine, in_organ_ratio, pointing_accuracy, spatial_entropy)

Produit : results/adebayo/<model>_fold<N>_adebayo_full.csv
Colonnes : patient_id, organ, step, layer_randomized, ssim_vs_pristine,
           in_organ_ratio, pointing_accuracy, spatial_entropy
"""
from __future__ import annotations
import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Patch cuCIM pour Kaggle cu128
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
from src.xai.adebayo import list_conv_layers, randomize_layer, ssim_3d


CLASS_NAMES = {1: "poumon_g", 2: "poumon_d", 3: "coeur", 4: "oesophage"}


def center_crop_to_patch(x_tensor, y_arr, patch, device):
    D, H, W = x_tensor.shape[-3:]
    pd, ph, pw = patch
    x_np = x_tensor[0, 0].cpu().numpy()
    pad = [(0, max(0, pd - D)), (0, max(0, ph - H)), (0, max(0, pw - W))]
    if any(p[1] > 0 for p in pad):
        x_np = np.pad(x_np, pad, mode="constant", constant_values=0)
        y_arr = np.pad(y_arr, pad, mode="constant", constant_values=0)
    D2, H2, W2 = x_np.shape
    sd = (D2 - pd) // 2
    sh = (H2 - ph) // 2
    sw = (W2 - pw) // 2
    slices = (slice(sd, sd + pd), slice(sh, sh + ph), slice(sw, sw + pw))
    x_cropped = x_np[slices]
    y_cropped = y_arr[slices]
    x_t = torch.from_numpy(x_cropped).float().unsqueeze(0).unsqueeze(0).to(device)
    return x_t, y_cropped


def compute_heatmap(model, x, target_class, patch_size):
    """Calcule la heatmap SEG-GRAD-CAM 3D, retourne np.ndarray 3D."""
    local_cam = SegGradCAM3D(model, target_layer="auto")
    try:
        if hasattr(local_cam, "compute"):
            h = local_cam.compute(x, target_class=target_class, patch_size=tuple(patch_size))
        else:
            h = local_cam(x, target_class=target_class)
        if isinstance(h, torch.Tensor):
            h = h.detach().cpu().numpy()
        # squeeze batch si necessaire
        if h.ndim == 4 and h.shape[0] == 1:
            h = h[0]
        return np.asarray(h, dtype=np.float32)
    finally:
        if hasattr(local_cam, "cleanup"):
            local_cam.cleanup()
        elif hasattr(local_cam, "remove_hooks"):
            local_cam.remove_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["unet", "segresnet"], required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--image_filename", default=None)
    ap.add_argument("--label_filename", default=None)
    ap.add_argument("--out_dir", default="results/adebayo")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_patients", type=int, default=5,
                    help="Nb de patients du fold val a traiter (default 5)")
    ap.add_argument("--target_organs", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="Organes cibles (1=poumon_g, 2=poumon_d, 3=coeur, 4=oesophage)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root: cfg["data"]["root"] = args.data_root
    if args.image_filename: cfg["data"]["image_filename"] = args.image_filename
    if args.label_filename: cfg["data"]["label_filename"] = args.label_filename

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split fold
    items = list_patients(cfg["data"]["root"],
                          cfg["data"]["image_filename"],
                          cfg["data"]["label_filename"])
    folds = make_5fold_splits(items, n_folds=cfg["splits"]["n_folds"],
                              seed=cfg["experiment"]["seed"])
    _, val_items = load_fold(folds, args.fold, items)
    val_items = val_items[:args.n_patients]
    print(f"Fold {args.fold} ({args.model}) : {len(val_items)} patients pour Adebayo")

    modality = cfg["data"].get("modality", "mri")
    intensity_params = cfg["data"].get("intensity_params") or (
        {"clip_percentiles": cfg["data"].get("intensity_clip_percentiles", [0.5, 99.5])}
        if modality == "mri" else {"hu_window": cfg["data"].get("hu_window", [-1000.0, 400.0])}
    )
    val_tf = get_val_transforms(spacing=cfg["data"]["spacing"],
                                intensity_params=intensity_params, modality=modality)
    _, val_loader = make_loaders(val_items, val_items, val_tf, val_tf,
                                 batch_size=1, num_workers=0, cache_rate=0.0)
    patch_size = cfg["data"]["patch_size"]

    # Charger une fois le modele pristine
    pristine_model = build_model(
        name=args.model,
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        features=cfg["model"]["features"],
        segresnet_kwargs=cfg["model"].get("segresnet"),
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    pristine_model.load_state_dict(state)
    pristine_model.eval()
    print(f"Checkpoint charge : {args.ckpt}")

    n_conv = len(list_conv_layers(pristine_model))
    print(f"Modele {args.model} : {n_conv} couches Conv3d (cascading top-down)")

    # Pre-charger les patients (image + label) crop a patch
    print("Pre-chargement patients (center-crop patch)...")
    patient_data = []
    for i, batch in enumerate(val_loader):
        pid = val_items[i]["patient_id"]
        x_full = batch["image"].to(device)
        y_full = batch["label"][0, 0].cpu().numpy().astype(np.int32)
        x, y = center_crop_to_patch(x_full, y_full, tuple(patch_size), device)
        patient_data.append({"pid": pid, "x": x.cpu(), "y": y})  # x en CPU pour economiser
        print(f"  loaded {pid}")

    # ----- Cascading randomization commune pour tous les patients -----
    # On randomise UNE FOIS la sequence de layers et on recalcule TOUS les
    # patients a chaque etape. C'est plus rapide que de tout deepcopy par patient.

    rows = []  # final CSV rows

    def evaluate_all_patients(model, step, layer_name):
        """Pour ce modele (a ce stade de randomisation), calcule heatmaps
        pour tous les patients et tous les organes cibles."""
        for pdata in patient_data:
            pid, y = pdata["pid"], pdata["y"]
            x = pdata["x"].to(device)
            for c in args.target_organs:
                gt_mask = (y == c)
                cname = CLASS_NAMES.get(c, f"class{c}")
                if not gt_mask.any():
                    rows.append({
                        "patient_id": pid, "organ": cname, "step": step,
                        "layer_randomized": layer_name,
                        "ssim_vs_pristine": float("nan"),
                        "in_organ_ratio": float("nan"),
                        "pointing_accuracy": float("nan"),
                        "spatial_entropy": float("nan"),
                    })
                    continue
                try:
                    heat = compute_heatmap(model, x, c, patch_size)
                    if heat.shape != gt_mask.shape:
                        h_t = torch.from_numpy(heat).float().unsqueeze(0).unsqueeze(0)
                        h_t = F.interpolate(h_t, size=gt_mask.shape,
                                            mode="trilinear", align_corners=False)
                        heat = h_t[0, 0].numpy()
                    # SSIM contre pristine: on stocke pristine au step 0
                    key_pristine = (pid, cname)
                    if step == 0:
                        pristine_cache[key_pristine] = heat.copy()
                        ssim = 1.0
                    else:
                        ref = pristine_cache.get(key_pristine)
                        ssim = ssim_3d(ref, heat) if ref is not None else float("nan")
                    metrics = compute_all_metrics(heat, gt_mask)
                    rows.append({
                        "patient_id": pid, "organ": cname, "step": step,
                        "layer_randomized": layer_name,
                        "ssim_vs_pristine": ssim,
                        "in_organ_ratio": metrics.get("in_organ_ratio", float("nan")),
                        "pointing_accuracy": metrics.get("pointing_accuracy", float("nan")),
                        "spatial_entropy": metrics.get("spatial_entropy", float("nan")),
                    })
                except Exception as e:
                    print(f"    [step {step}] {pid} {cname} : ECHEC ({e})")
                    rows.append({
                        "patient_id": pid, "organ": cname, "step": step,
                        "layer_randomized": layer_name,
                        "ssim_vs_pristine": float("nan"),
                        "in_organ_ratio": float("nan"),
                        "pointing_accuracy": float("nan"),
                        "spatial_entropy": float("nan"),
                    })
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Step 0 : pristine
    pristine_cache = {}  # (pid, cname) -> heatmap np.ndarray
    print("\n[Step 0] Pristine")
    evaluate_all_patients(pristine_model, step=0, layer_name="<pristine>")

    # Steps 1..N : cascading randomization
    work_model = copy.deepcopy(pristine_model)
    work_model.eval()
    layers = list_conv_layers(work_model)[::-1]  # output -> input
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    for i, (name, layer) in enumerate(layers, start=1):
        randomize_layer(layer, generator=gen)
        print(f"[Step {i}/{len(layers)}] randomized layer: {name}")
        evaluate_all_patients(work_model, step=i, layer_name=name)

    # Sauve CSV
    out_csv = out_dir / f"{args.model}_fold{args.fold}_adebayo_full.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "patient_id", "organ", "step", "layer_randomized",
            "ssim_vs_pristine", "in_organ_ratio",
            "pointing_accuracy", "spatial_entropy"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV : {out_csv} ({len(rows)} rows)")
    print("=== Adebayo analysis FINI ===")


if __name__ == "__main__":
    main()
