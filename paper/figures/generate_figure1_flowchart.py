"""Figure 1 de l'article : flowchart CONSORT-like du filtrage qualite.

Genere un diagramme propre 616 -> N patients retenus, avec les raisons
d'exclusion. Les nombres sont parametrables en haut du script (a ajuster avec
les chiffres finaux apres confirmation du dataset).

Usage :
    python paper/figures/generate_figure1_flowchart.py
    # produit paper/figures/figure1_flowchart.png (300 DPI) et .pdf (vectoriel)

Si _quality_summary.json est present, les nombres sont lus automatiquement.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# =====================================================================
# NOMBRES — a ajuster avec les chiffres finaux
# (lus depuis _quality_summary.json si disponible)
# =====================================================================
N_TOTAL = 616
N_KEPT = 187          # <-- chiffre final apres fix "empty organ files"
EXCLUSIONS = [        # (label, count) — peut depasser N_excluded car multi-raisons
    ("Lung touching image boundary (probable cut-off)", 201),
    ("Cranio-caudal FOV < 120 mm", 105),
    ("Lung volume < 300 mL", 82),
    ("Esophagus volume < 5 mL", 77),
    ("Empty / missing organ annotation", 116),
]
SUMMARY_JSON = Path(__file__).resolve().parents[2] / "data_thorax_complet" / "_quality_summary.json"
# =====================================================================


def try_load_summary():
    global N_TOTAL, N_KEPT
    if SUMMARY_JSON.exists():
        try:
            s = json.loads(SUMMARY_JSON.read_text())
            N_TOTAL = s.get("n_input", N_TOTAL)
            N_KEPT = s.get("n_kept", N_KEPT)
            print(f"Nombres lus depuis {SUMMARY_JSON}: total={N_TOTAL}, kept={N_KEPT}")
        except Exception as e:
            print(f"(lecture JSON echouee, valeurs par defaut: {e})")
    else:
        print(f"(pas de _quality_summary.json, valeurs par defaut: total={N_TOTAL}, kept={N_KEPT})")


def box(ax, x, y, w, h, text, fc="#eef3fb", ec="#2c5d8f"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.4, color="#444"))


def main():
    try_load_summary()
    n_excluded = N_TOTAL - N_KEPT

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")

    # Boite 1 : dataset total
    box(ax, 2.5, 10.3, 5, 1.1,
        f"TotalSegmentator MRI v2.0.0\n{N_TOTAL} MRI volumes\n(50 anatomical regions)",
        fc="#dce9f7")
    # Boite 2 : OAR thoraciques
    box(ax, 2.5, 8.6, 5, 1.0,
        f"Thoracic OAR mapping\n(left lung, right lung, heart, esophagus)")
    # Boite exclusions (a droite)
    excl_text = f"Excluded by quality filter\n(n = {n_excluded}; multiple reasons possible):\n" + \
        "\n".join(f"  - {lab}: {cnt}" for lab, cnt in EXCLUSIONS)
    box(ax, 5.3, 5.4, 4.5, 2.4, excl_text, fc="#fbe9e9", ec="#a33")
    # Boite finale : retenus
    box(ax, 2.5, 4.2, 5, 1.1,
        f"Quality-filtered cohort\n{N_KEPT} patients\n(complete thoracic anatomy)",
        fc="#e6f5e6", ec="#2f7d32")
    # Boite CV
    box(ax, 2.5, 2.4, 5, 1.0,
        "Patient-stratified 5-fold\ncross-validation (seed = 42)")

    # Fleches
    arrow(ax, 5, 10.3, 5, 9.6)
    arrow(ax, 5, 8.6, 5, 5.3)
    arrow(ax, 5, 7.0, 5.3, 6.6)   # vers exclusions
    arrow(ax, 5, 4.2, 5, 3.4)

    ax.text(5, 11.7, "Figure 1. Dataset quality filtering flow", ha="center",
            fontsize=12, fontweight="bold")

    out_png = Path(__file__).resolve().parent / "figure1_flowchart.png"
    out_pdf = Path(__file__).resolve().parent / "figure1_flowchart.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Figure ecrite : {out_png}")
    print(f"Figure ecrite : {out_pdf}")


if __name__ == "__main__":
    main()
