# -*- coding: utf-8 -*-
"""Six-OAR segmentation figures on MRI: reference vs U-Net vs SegResNet (phase 4, D115, native grid).

Sources (read-only):
  data/<patient>/image.nii.gz                      MRI (native grid)
  data_6oar/<patient>/label.nii.gz                 reference labels (1 L lung, 2 R lung, 3 heart, 4 oesophagus,
                                                   5 spinal cord, 6 aorta)
  results/phase3_ablation/*/abl18m-{unet6,seg6}-s42-f*/ablation_eval/predictions.zip   predicted masks (native grid)
  results/phase3_ablation/*/abl18m-*6-s42-f*/ablation_eval/metrics_long.csv            DSC printed on the panels
Display: radiological convention (patient right on image left), nearest-neighbour (no smoothing or interpolation),
filled masks with constant opacity + contour. Voxel aspect ratio from the native spacing.

    .venv\\Scripts\\python.exe phase4_6oar/figures/segmentations/make_segmentation_images.py
"""
import glob
import gzip
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORG = ["poumon_g", "poumon_d", "coeur", "oesophage", "moelle", "aorte"]
EN = {"poumon_g": "Left lung", "poumon_d": "Right lung", "coeur": "Heart", "oesophage": "Oesophagus",
      "moelle": "Spinal cord", "aorte": "Aorta"}
COL = {"poumon_g": "#2a78d6", "poumon_d": "#6250d6", "coeur": "#e0383e", "oesophage": "#1baf7a",
       "moelle": "#f2b51d", "aorte": "#eb6834"}
SURF, INK = "#fcfcfb", "#0b0b0b"
plt.rcParams.update({"figure.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9, "font.family": "DejaVu Sans"})
LABCMAP = ListedColormap(["#00000000"] + [COL[o] for o in ORG])
ALPHA = 0.40
PATIENTS = ["PATIENT_s0001", "PATIENT_s0002", "PATIENT_s0009", "PATIENT_s0017", "PATIENT_s0181"]

zips = {}
for r in glob.glob(str(ROOT / "results/phase3_ablation/*/abl18m-*6-s42-f*/ablation_eval/predictions.zip")):
    model = "U-Net" if "unet6" in r else "SegResNet"
    with zipfile.ZipFile(r) as z:
        for m in z.namelist():
            mm = re.search(r"(PATIENT_s\d+)_pred\.nii\.gz$", m)
            if mm:
                zips[(model, mm.group(1))] = (r, m)
met = pd.concat(pd.read_csv(f) for f in glob.glob(str(ROOT / "results/phase3_ablation/*/abl18m-*6-s42-f*/ablation_eval/metrics_long.csv")))
met = met[met.space == "native"]
met["m"] = met.variant.map({"unet_6oar": "U-Net", "segresnet_6oar": "SegResNet"})
DSC = met.set_index(["m", "patient_id", "organ"]).dsc


def canon(img):
    return nib.as_closest_canonical(img)


def load_case(pid):
    im = canon(nib.load(str(ROOT / "data" / pid / "image.nii.gz")))
    gt = canon(nib.load(str(ROOT / "data_6oar" / pid / "label.nii.gz")))
    vols = {"Reference": np.asarray(gt.dataobj).astype(np.uint8)}
    for model in ("U-Net", "SegResNet"):
        r, m = zips[(model, pid)]
        with zipfile.ZipFile(r) as z:
            p = canon(nib.Nifti1Image.from_bytes(gzip.decompress(z.read(m))))
        assert p.shape == gt.shape and np.allclose(p.affine, gt.affine, atol=1e-3), pid
        vols[model] = np.asarray(p.dataobj).astype(np.uint8)
    return np.asarray(im.dataobj, dtype=np.float32), vols, im.header.get_zooms()[:3]


def views(G, zooms):
    """Slices (RAS canonical arrays, x = toward patient right... displayed radiologically)."""
    lung_z = np.flatnonzero(((G == 1) | (G == 2)).any((0, 1)))
    z_a = int(np.argmax(((G == 3) | (G == 4)).sum((0, 1))))                         # heart / oesophagus level
    z_b = int(lung_z.min() + 0.80 * (lung_z.max() - lung_z.min()))                  # upper thorax
    heart = np.argwhere(G == 3)
    y_c = int(np.median(heart[:, 1])) if len(heart) else G.shape[1] // 2           # coronal through heart
    cord = np.argwhere(G == 5)
    x_s = int(np.median(cord[:, 0])) if len(cord) else G.shape[0] // 2             # sagittal through cord
    sx, sy, sz = zooms
    # sagittal: anterior on the left (patient facing left)
    return [(f"Axial, heart level (slice {z_a})", lambda A: np.fliplr(A[:, :, z_a].T), sy / sx),
            (f"Axial, upper thorax (slice {z_b})", lambda A: np.fliplr(A[:, :, z_b].T), sy / sx),
            (f"Coronal (slice {y_c})", lambda A: np.fliplr(A[:, y_c, :].T), sz / sx),
            (f"Sagittal (slice {x_s})", lambda A: np.fliplr(A[x_s, :, :].T), sz / sy)]


def draw(ax, img2d, lab2d, lo, hi, asp):
    ax.imshow(img2d, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest", aspect=asp)
    ax.imshow(np.ma.masked_equal(lab2d, 0), cmap=LABCMAP, vmin=0, vmax=6, alpha=ALPHA, origin="lower",
              interpolation="nearest", aspect=asp)
    for k, o in enumerate(ORG, start=1):
        if (lab2d == k).any():
            ax.contour(lab2d == k, levels=[0.5], colors=[COL[o]], linewidths=0.9)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def legend(fig, y=0.01):
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[o], alpha=0.8, label=EN[o]) for o in ORG]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, y))


def patient_figure(pid):
    I, vols, zooms = load_case(pid)
    lo, hi = np.percentile(I, [0.5, 99.5])
    vw = views(vols["Reference"], zooms)
    fig, axes = plt.subplots(3, len(vw) + 1, figsize=(3.0 * len(vw) + 2.6, 9.2),
                             gridspec_kw={"width_ratios": [1] * len(vw) + [0.85]})
    for r, name in enumerate(("Reference", "U-Net", "SegResNet")):
        for c, (title, f, asp) in enumerate(vw):
            ax = axes[r, c]
            draw(ax, f(I), f(vols[name]), lo, hi, asp)
            if r == 0:
                ax.set_title(title, fontsize=9.5, color=INK)
            if c == 0:
                ax.set_ylabel(name, fontsize=11, color=INK)
            left, right = ("R", "L") if c < 3 else ("A", "P")
            ax.text(0.02, 0.5, left, transform=ax.transAxes, color="white", fontsize=7, va="center")
            ax.text(0.95, 0.5, right, transform=ax.transAxes, color="white", fontsize=7, va="center")
        ax = axes[r, -1]
        ax.axis("off")
        if name != "Reference":
            lines = [f"DSC ({name})"] + [f"{EN[o]:<12}{DSC.get((name, pid, o), np.nan):6.3f}" for o in ORG]
            ax.text(0.02, 0.5, "\n".join(lines), va="center", family="monospace", fontsize=8.5, color=INK)
        else:
            ax.text(0.02, 0.5, "Reference labels\n(TotalSegmentator MRI\n+ six-OAR cohort)", va="center",
                    fontsize=8.5, color=INK)
    legend(fig)
    fig.suptitle(f"{pid.replace('PATIENT_', '')} — six-OAR segmentation (cross-validation prediction, native grid)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    name = f"seg6_{pid.replace('PATIENT_', '')}"
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("  figure", name)


def gallery(pids, model="SegResNet", tag=""):
    fig, axes = plt.subplots(2, len(pids), figsize=(2.9 * len(pids), 6.6), squeeze=False)
    for j, pid in enumerate(pids):
        I, vols, zooms = load_case(pid)
        lo, hi = np.percentile(I, [0.5, 99.5])
        title, f, asp = views(vols["Reference"], zooms)[0]
        for r, name in enumerate(("Reference", model)):
            draw(axes[r, j], f(I), f(vols[name]), lo, hi, asp)
            if j == 0:
                axes[r, j].set_ylabel(name, fontsize=11, color=INK)
        mean = np.nanmean([DSC.get((model, pid, o), np.nan) for o in ORG])
        axes[0, j].set_title(pid.replace("PATIENT_", ""), fontsize=9.5, color=INK)
        axes[1, j].text(0.03, 0.04, f"mean DSC {mean:.3f}", transform=axes[1, j].transAxes, color="white",
                        fontsize=8, bbox=dict(facecolor="black", alpha=0.55, lw=0, pad=1.5))
    legend(fig)
    sub = {"best_to_worst": "(two best, median and two worst cases by mean DSC over the six OAR)"}.get(tag, "")
    fig.suptitle(f"Axial slice at the heart level — reference vs {model} {sub}".strip(), fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    name = f"seg6_gallery_{model.replace('-', '')}{('_' + tag) if tag else ''}"
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("  figure", name)


if __name__ == "__main__":
    for pid in PATIENTS:
        patient_figure(pid)
    gallery(PATIENTS, "SegResNet")
    gallery(PATIENTS, "U-Net")
    # best, median and worst SegResNet cases (mean DSC over the six OAR)
    m = met[met.m == "SegResNet"].groupby("patient_id").dsc.mean().sort_values()
    picks = [m.index[-1], m.index[-2], m.index[len(m) // 2], m.index[1], m.index[0]]
    gallery(picks, "SegResNet", "best_to_worst")
    for pid in (m.index[-1], m.index[len(m) // 2], m.index[0]):
        if pid not in PATIENTS:
            patient_figure(pid)
