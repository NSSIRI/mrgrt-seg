"""Figures 2 et 3 de l'article : box plots des metriques par OAR.

Lit les CSV par-patient produits par scripts/evaluate.py et trace des box plots
comparatifs. Deux modes :

  --mode ablation   : compare deux datasets (filtre vs non filtre) pour un modele
  --mode arch       : compare deux architectures (U-Net vs SegResNet)

Exemples :
    # Figure 3 : U-Net vs SegResNet (DSC)
    python paper/figures/generate_boxplots.py --mode arch \
        --a "results/unet_fold*.csv" --name_a "U-Net" \
        --b "results/segresnet_fold*.csv" --name_b "SegResNet" \
        --metric dsc --out paper/figures/figure3_arch_dsc.png

    # Figure 2 : filtre vs non filtre (DSC)
    python paper/figures/generate_boxplots.py --mode ablation \
        --a "results_filtered/unet_fold*.csv" --name_a "Filtered (n=187)" \
        --b "results_full/unet_fold*.csv" --name_b "Unfiltered (n=616)" \
        --metric dsc --out paper/figures/figure2_ablation_dsc.png
"""
from __future__ import annotations
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["poumon_g", "poumon_d", "coeur", "oesophage"]
CLASS_LABELS = {"poumon_g": "Left lung", "poumon_d": "Right lung",
                "coeur": "Heart", "oesophage": "Esophagus"}
METRIC_LABELS = {"dsc": "Dice Similarity Coefficient", "hd95": "HD95 (mm)",
                 "surface_dsc": "Surface DSC @ 2 mm", "iou": "IoU"}


def load_values(pattern, metric):
    """Retourne {classe: [valeurs par patient]}."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"Aucun fichier pour {pattern}")
    data = {c: [] for c in CLASS_NAMES}
    for fp in files:
        with open(fp) as f:
            for r in csv.DictReader(f):
                for c in CLASS_NAMES:
                    col = f"{metric}_{c}"
                    if col in r and r[col] not in ("", "nan"):
                        try:
                            data[c].append(float(r[col]))
                        except ValueError:
                            pass
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ablation", "arch"], required=True)
    p.add_argument("--a", required=True); p.add_argument("--name_a", default="A")
    p.add_argument("--b", required=True); p.add_argument("--name_b", default="B")
    p.add_argument("--metric", default="dsc", choices=list(METRIC_LABELS))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    da = load_values(args.a, args.metric)
    db = load_values(args.b, args.metric)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    positions_a = np.arange(len(CLASS_NAMES)) * 2.0
    positions_b = positions_a + 0.7

    col_a, col_b = "#3b78c2", "#c25a3b"
    bp_a = ax.boxplot([da[c] for c in CLASS_NAMES], positions=positions_a,
                      widths=0.55, patch_artist=True, showfliers=False)
    bp_b = ax.boxplot([db[c] for c in CLASS_NAMES], positions=positions_b,
                      widths=0.55, patch_artist=True, showfliers=False)
    for box in bp_a["boxes"]: box.set(facecolor=col_a, alpha=0.7)
    for box in bp_b["boxes"]: box.set(facecolor=col_b, alpha=0.7)
    for bp in (bp_a, bp_b):
        for med in bp["medians"]: med.set(color="black", linewidth=1.5)

    ax.set_xticks(positions_a + 0.35)
    ax.set_xticklabels([CLASS_LABELS[c] for c in CLASS_NAMES])
    ax.set_ylabel(METRIC_LABELS[args.metric])
    if args.metric in ("dsc", "surface_dsc", "iou"):
        ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=col_a, alpha=0.7, label=args.name_a),
                       Patch(facecolor=col_b, alpha=0.7, label=args.name_b)],
              loc="lower right")
    title = ("Figure 2. Impact of quality filtering" if args.mode == "ablation"
             else "Figure 3. Architecture comparison")
    ax.set_title(f"{title} - {METRIC_LABELS[args.metric]}", fontsize=12, fontweight="bold")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Figure ecrite : {out}")
    print(f"Figure ecrite : {out.with_suffix('.pdf')}")
    # Recap medianes
    for c in CLASS_NAMES:
        ma = np.median(da[c]) if da[c] else float("nan")
        mb = np.median(db[c]) if db[c] else float("nan")
        print(f"  {CLASS_LABELS[c]:<11}: {args.name_a}={ma:.3f}  {args.name_b}={mb:.3f}")


if __name__ == "__main__":
    main()
