"""English article figures 8 and 9 (PNG 300 dpi + PDF) -> paper/figures_en/. Read-only on all data.

  fig8_thoracic_region_cord_aorta : spinal cord and aorta, full extent vs lung slices (phase4_6oar/thorax)
  fig9_xai_matched_layers         : Seg-Grad-CAM in-organ ratio at 4 matched layers, D187 and D115
                                    (results/phase3_ablation/xai_matched)
Same style as make_figures_en.py. Raw per-patient values only (no smoothing, no subsampling).

    .venv\\Scripts\\python.exe paper/figures_en/make_figures_xai_en.py
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EN = {"poumon_g": "Left lung", "poumon_d": "Right lung", "coeur": "Heart", "oesophage": "Oesophagus",
      "moelle": "Spinal cord", "aorte": "Aorta"}
SEG, UNET = "#eb6834", "#1baf7a"
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.family": "DejaVu Sans"})
RNG = np.random.default_rng(20261106)   # jitter only


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  figure", name)


def strip_box(ax, v, x, color, w=0.32):
    ax.scatter(x + RNG.uniform(-w / 3, w / 3, len(v)), v, s=4, color=color, alpha=0.4, linewidths=0, zorder=2)
    ax.boxplot(v, positions=[x], widths=w, showfliers=False, patch_artist=True, zorder=3,
               boxprops=dict(facecolor="none", edgecolor=INK, lw=0.9), medianprops=dict(color=INK, lw=1.4),
               whiskerprops=dict(color=INK2, lw=0.8), capprops=dict(color=INK2, lw=0.8))


def fig8():
    d = pd.read_csv(ROOT / "phase4_6oar/thorax/metrics_thorax_long.csv")
    d["m"] = d.variant.map({"unet_6oar": "U-Net", "segresnet_6oar": "SegResNet"})
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2))
    for i, (met, lab) in enumerate([("dsc", "DSC"), ("hd95_mm", "HD95 (mm, log scale)")]):
        for j, org in enumerate(["moelle", "aorte"]):
            ax = axes[i, j]
            for k, region in enumerate(["complet", "thorax"]):
                for q, (mod, col) in enumerate([("U-Net", UNET), ("SegResNet", SEG)]):
                    v = d[(d.region == region) & (d.organ == org) & (d.m == mod)][met].dropna().to_numpy()
                    strip_box(ax, v, k * 1.0 + (q - 0.5) * 0.38, col)
            ax.set_xticks([0, 1], ["Full annotated extent", "Lung slices only"])
            ax.set_xlim(-0.6, 1.6)
            if met == "hd95_mm":
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(EN[org], fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(lab)
    h = [axes[0, 0].scatter([], [], color=c, label=m) for m, c in (("U-Net", UNET), ("SegResNet", SEG))]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, -0.01, "D115, 115 patients, 5 folds, native grid. Lung slices = slices containing lung in the reference; "
             "surface distances from full-mask boundaries.", fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "fig8_thoracic_region_cord_aorta")


def fig9():
    res = ROOT / "results/phase3_ablation/xai_matched"
    a = pd.concat(pd.read_csv(f) for f in glob.glob(str(res / "*/abl18m-xai-e3*-matched/xai_matched/xai_matched_long.csv")))
    b = pd.concat(pd.read_csv(f) for f in glob.glob(str(res / "*/abl18m-xai6-*/xai_matched/xai_matched_long.csv")))
    a["m"] = a.variant.map({"unet_opt": "U-Net", "segresnet": "SegResNet"})
    b["m"] = b.variant.map({"unet6": "U-Net", "seg6": "SegResNet"})
    assert a.groupby("m").patient_id.nunique().eq(187).all() and b.groupby("m").patient_id.nunique().eq(115).all()
    levels = ["enc_profond", "dec_demi", "dec_pre_tete", "sortie"]
    ticks = ["Enc.\ndeep", "Dec.\n1/2", "Pre-\nhead", "Out-\nput"]
    rows = [("D187: E3 (U-Net) vs E2 (SegResNet)", a, ["poumon_g", "poumon_d", "coeur", "oesophage"]),
            ("D115: U-Net vs SegResNet (six OAR)", b, ["poumon_g", "poumon_d", "coeur", "oesophage", "moelle", "aorte"])]
    fig, axes = plt.subplots(2, 6, figsize=(17, 6.6), sharey=True)
    for r, (title, d, organs) in enumerate(rows):
        for c in range(6):
            ax = axes[r, c]
            if c >= len(organs):
                ax.axis("off")
                continue
            org = organs[c]
            for i, lv in enumerate(levels):
                for q, (mod, col) in enumerate([("U-Net", UNET), ("SegResNet", SEG)]):
                    v = d[(d.m == mod) & (d.level == lv) & (d.organ == org)].in_organ_ratio.dropna().to_numpy()
                    strip_box(ax, v, i + (q - 0.5) * 0.36, col, w=0.28)
            ax.set_xticks(range(4), ticks, fontsize=7)
            ax.set_title(EN[org], fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel("In-organ ratio")
        axes[r, 0].annotate(title, xy=(0, 1.13), xycoords="axes fraction", fontsize=10, color=INK, weight="bold")
    h = [axes[0, 0].scatter([], [], color=c, label=m) for m, c in (("U-Net", UNET), ("SegResNet", SEG))]
    axes[0, 5].legend(handles=h, loc="center", fontsize=10)
    fig.text(0.01, -0.01, "Seg-Grad-CAM, 128x128x64 field centred on the four thoracic OAR. Enc. deep = deepest encoder "
             "stage; Dec. 1/2 = decoder at 1/2 resolution; Pre-head = last feature map before the output projection; "
             "Output = logits. A uniform map gives an in-organ ratio equal to the organ's volume fraction.",
             fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "fig9_xai_matched_layers")


if __name__ == "__main__":
    fig8()
    fig9()
