# -*- coding: utf-8 -*-
"""High-resolution Seg-Grad-CAM overlays on MRI (matched layers), from the example maps saved by the Kaggle jobs.

Source (read-only): results/phase3_ablation/xai_matched/*/*/xai_matched/examples/*.npz
  image / label : 128x128x64 crop (RAS, 1.5 x 1.5 x 3 mm, centred on the 4 thoracic OAR)
  <organ>__<level> : Seg-Grad-CAM map in [0, 1] for that organ and layer level
In-organ ratios printed on the panels are read from xai_matched_long.csv (same run, same patient).

Display: radiological convention (patient right on image left); nearest-neighbour display (no added smoothing or
interpolation); map shown with constant opacity, values < 0.05 transparent; target organ reference in cyan.

    .venv\\Scripts\\python.exe phase3_ablation/xai_matched/figures/make_gradcam_images.py
Writes PNG (300 dpi) + PDF in phase3_ablation/xai_matched/figures/.
"""
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parents[2] / "results" / "phase3_ablation" / "xai_matched"
EN = {"coeur": "Heart", "oesophage": "Oesophagus", "moelle": "Spinal cord", "aorte": "Aorta"}
CLS = {"coeur": 3, "oesophage": 4, "moelle": 5, "aorte": 6}
LEVELS = [("enc_profond", "Deepest encoder"), ("dec_pre_tete", "Pre-head map"), ("sortie", "Output logits")]
MODELS = {"D187": [("unet_opt", "E3 U-Net"), ("segresnet", "E2 SegResNet")],
          "D115": [("unet6", "U-Net"), ("seg6", "SegResNet")]}
SURF, INK, MUTED = "#fcfcfb", "#0b0b0b", "#8a8984"
plt.rcParams.update({"figure.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
                     "font.family": "DejaVu Sans"})
base = plt.get_cmap("inferno")(np.linspace(0, 1, 256))
CMAP = ListedColormap(base)
CMAP.set_under(alpha=0.0)
THR = 0.05

npz = {}
for f in glob.glob(str(RES / "*/*/xai_matched/examples/*.npz")):
    m = re.search(r"(unet_opt|segresnet|unet6|seg6)_fold(\d)_(PATIENT_s\d+)\.npz$", f)
    npz[(m.group(1), m.group(3))] = (f, int(m.group(2)))
long = pd.concat(pd.read_csv(f) for f in glob.glob(str(RES / "*/*/xai_matched/xai_matched_long.csv")))
IOR = long.set_index(["variant", "patient_id", "level", "organ"]).in_organ_ratio


def axial(arr, z):
    """Crop array (x, y, z) in RAS -> axial slice in radiological display (rows = anterior up)."""
    return np.fliplr(arr[:, :, z].T)


def best_slice(lab, c):
    """Axial slice with the largest area of organ c among the slices that contain lung (thoracic study);
    falls back to all slices if the organ never appears together with lung."""
    area = (lab == c).sum((0, 1)).astype(float)
    lung = ((lab == 1) | (lab == 2)).any((0, 1))
    if (area * lung).max() > 0:
        area = area * lung
    return int(np.argmax(area))


def panel(ax, img, lo, hi, heat, mask, title=None, note=None):
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
    im = None
    if heat is not None:
        im = ax.imshow(np.ma.masked_less(heat, THR), cmap=CMAP, vmin=0, vmax=1, alpha=0.6, origin="lower",
                       interpolation="nearest")
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=["#22d3ee"], linewidths=1.0)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8.5, color=INK, pad=3)
    if note:
        ax.text(0.03, 0.04, note, transform=ax.transAxes, color="white", fontsize=7.5,
                bbox=dict(facecolor="black", alpha=0.55, lw=0, pad=1.5))
    return im


def patient_figure(cohort, pid, organs):
    (va, la), (vb, lb) = MODELS[cohort]
    za, zb = np.load(npz[(va, pid)][0]), np.load(npz[(vb, pid)][0])
    fold = npz[(va, pid)][1]
    img = za["image"].astype(np.float32)
    lab = za["label"]
    assert np.array_equal(lab, zb["label"]), "crops differ between models"
    lo, hi = np.percentile(img, [0.5, 99.5])
    ncol = 1 + 2 * len(LEVELS)
    fig, axes = plt.subplots(len(organs), ncol, figsize=(1.75 * ncol, 1.95 * len(organs) + 0.6), squeeze=False)
    im = None
    for r, org in enumerate(organs):
        c = CLS[org]
        z = best_slice(lab, c)
        mask = axial(lab, z) == c
        base_img = axial(img, z)
        panel(axes[r, 0], base_img, lo, hi, None, mask, "MRI + reference" if r == 0 else None)
        axes[r, 0].set_ylabel(EN[org], fontsize=9.5, color=INK)
        axes[r, 0].text(0.03, 0.04, f"slice {z}", transform=axes[r, 0].transAxes, color="white", fontsize=7)
        axes[r, 0].text(0.02, 0.5, "R", transform=axes[r, 0].transAxes, color="white", fontsize=7, va="center")
        axes[r, 0].text(0.94, 0.5, "L", transform=axes[r, 0].transAxes, color="white", fontsize=7, va="center")
        for k, (v, vz, lab_m) in enumerate(((va, za, la), (vb, zb, lb))):
            for j, (lv, lvlab) in enumerate(LEVELS):
                key = f"{org}__{lv}"
                ax = axes[r, 1 + k * len(LEVELS) + j]
                if key not in vz.files:
                    ax.axis("off")
                    continue
                heat = axial(vz[key].astype(np.float32), z)
                ior = IOR.get((v, pid, lv, org), np.nan)
                im = panel(ax, base_img, lo, hi, heat, mask, f"{lab_m}\n{lvlab}" if r == 0 else None,
                           f"3D in-organ {ior:.2f}" if np.isfinite(ior) else "organ not predicted")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    cax = fig.add_axes([0.35, 0.045, 0.3, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Seg-Grad-CAM (min–max normalised per map; < 0.05 transparent)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cohort_txt = ("D187, four-OAR models (phase-2 checkpoints)" if cohort == "D187"
                  else "D115, six-OAR models")
    fig.suptitle(f"{pid.replace('PATIENT_', '')} — {cohort_txt}, validation fold {fold}; cyan: reference",
                 fontsize=10, color=INK, y=0.995)
    name = f"gradcam_{cohort}_{pid.replace('PATIENT_', '')}"
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("  figure", name)


def gallery(cohort, organ, level="sortie"):
    (va, la), (vb, lb) = MODELS[cohort]
    pids = sorted({p for (v, p) in npz if v == va} & {p for (v, p) in npz if v == vb})
    fig, axes = plt.subplots(2, len(pids), figsize=(2.3 * len(pids), 5.9), squeeze=False)
    im = None
    for j, pid in enumerate(pids):
        za, zb = np.load(npz[(va, pid)][0]), np.load(npz[(vb, pid)][0])
        img, lab = za["image"].astype(np.float32), za["label"]
        lo, hi = np.percentile(img, [0.5, 99.5])
        z = best_slice(lab, CLS[organ])
        mask = axial(lab, z) == CLS[organ]
        for r, (v, vz, lm) in enumerate(((va, za, la), (vb, zb, lb))):
            heat = axial(vz[f"{organ}__{level}"].astype(np.float32), z)
            ior = IOR.get((v, pid, level, organ), np.nan)
            im = panel(axes[r, j], axial(img, z), lo, hi, heat, mask,
                       pid.replace("PATIENT_", "") if r == 0 else None,
                       f"3D in-organ {ior:.2f}" if np.isfinite(ior) else "organ not predicted")
            if j == 0:
                axes[r, j].set_ylabel(lm, fontsize=9.5, color=INK)
    lvl = dict(LEVELS)[level]
    fig.suptitle(f"{EN[organ]} — Seg-Grad-CAM at the {lvl.lower()} level ({cohort}); cyan: reference",
                 fontsize=10, color=INK)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.14, wspace=0.06, hspace=0.12)
    cax = fig.add_axes([0.35, 0.06, 0.3, 0.02])
    fig.colorbar(im, cax=cax, orientation="horizontal").ax.tick_params(labelsize=7)
    name = f"gradcam_gallery_{cohort}_{organ}_{level}"
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("  figure", name)


if __name__ == "__main__":
    for cohort, organs in (("D187", ["coeur", "oesophage"]), ("D115", ["coeur", "oesophage", "moelle", "aorte"])):
        (va, _), (vb, _) = MODELS[cohort]
        for pid in sorted({p for (v, p) in npz if v == va} & {p for (v, p) in npz if v == vb}):
            patient_figure(cohort, pid, organs)
    for lv in ("sortie", "dec_pre_tete"):
        gallery("D187", "oesophage", lv)
        gallery("D115", "oesophage", lv)
    gallery("D115", "aorte", "sortie")
    gallery("D115", "moelle", "sortie")
