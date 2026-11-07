# -*- coding: utf-8 -*-
"""Thorax figures combining six-OAR segmentation and Seg-Grad-CAM heat maps (D115, U-Net vs SegResNet).

The heat maps exist only in the 128x128x64 crop saved by the XAI jobs (RAS, 1.5 x 1.5 x 3 mm, centred on the
four thoracic OAR). To overlay the segmentations on the same grid, the reference label and the cross-validation
prediction (native grid, predictions.zip) are passed through the SAME validation transform as in training/XAI
(nearest-neighbour for labels) and cropped with the SAME rule as xai_matched_kaggle.load_crop. A check verifies
that the recomputed image crop equals the saved one and that the recomputed reference crop equals the saved label.

Display: radiological convention, nearest-neighbour (no added smoothing), heat map with constant opacity and
values < 0.05 transparent; reference contour cyan, predicted contour white dashed.

    .venv\\Scripts\\python.exe phase3_ablation/xai_matched/figures/make_gradcam_seg_thorax.py
"""
import glob
import gzip
import re
import sys
import tempfile
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
sys.path.insert(0, str(ROOT))
from src.data.dataset import get_val_transforms  # noqa: E402

RES = ROOT / "results" / "phase3_ablation" / "xai_matched"
PATCH = (128, 128, 64)
ORG = ["poumon_g", "poumon_d", "coeur", "oesophage", "moelle", "aorte"]
EN = {"poumon_g": "Left lung", "poumon_d": "Right lung", "coeur": "Heart", "oesophage": "Oesophagus",
      "moelle": "Spinal cord", "aorte": "Aorta"}
COL = {"poumon_g": "#2a78d6", "poumon_d": "#6250d6", "coeur": "#e0383e", "oesophage": "#1baf7a",
       "moelle": "#f2b51d", "aorte": "#eb6834"}
HEAT_ORG = ["coeur", "oesophage", "moelle", "aorte"]
MODELS = [("seg6", "SegResNet", "segresnet_6oar"), ("unet6", "U-Net", "unet_6oar")]
SURF, INK = "#fcfcfb", "#0b0b0b"
plt.rcParams.update({"figure.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9, "font.family": "DejaVu Sans"})
LABCMAP = ListedColormap(["#00000000"] + [COL[o] for o in ORG])
HEAT = ListedColormap(plt.get_cmap("inferno")(np.linspace(0, 1, 256)))
VAL_TF = get_val_transforms(spacing=[1.5, 1.5, 3.0], intensity_params={"clip_percentiles": [0.5, 99.5]}, modality="mri")

npz = {}
# fold-first examples (abl18m-xai6-*/xai_matched) + thorax patients (abl18m-xai6t-*/xai_thorax)
for f in glob.glob(str(RES / "*/abl18m-xai6*/xai_*/examples/*.npz")):
    m = re.search(r"(unet6|seg6)_fold(\d)_(PATIENT_s\d+)\.npz$", f)
    npz[(m.group(1), m.group(3))] = f
zips = {}
for r in glob.glob(str(ROOT / "results/phase3_ablation/*/abl18m-*6-s42-f*/ablation_eval/predictions.zip")):
    v = "unet_6oar" if "unet6" in r else "segresnet_6oar"
    with zipfile.ZipFile(r) as z:
        for n in z.namelist():
            mm = re.search(r"(PATIENT_s\d+)_pred\.nii\.gz$", n)
            if mm:
                zips[(v, mm.group(1))] = (r, n)
long = pd.concat(pd.read_csv(f) for f in glob.glob(str(RES / "*/abl18m-xai6*/xai_*/xai_matched_long.csv")))
long = long.drop_duplicates(["variant", "fold", "patient_id", "level", "organ"])
IOR = long.set_index(["variant", "patient_id", "level", "organ"]).in_organ_ratio
met = pd.concat(pd.read_csv(f) for f in glob.glob(str(ROOT / "results/phase3_ablation/*/abl18m-*6-s42-f*/ablation_eval/metrics_long.csv")))
met = met[met.space == "native"]
DSC = met.set_index(["variant", "patient_id", "organ"]).dsc


def crop_slices(lab):
    """Same rule as xai_matched_kaggle.load_crop (pad to PATCH, centre on the bbox of classes 1-4)."""
    pad = [(0, max(0, p - s)) for s, p in zip(lab.shape, PATCH)]
    labp = np.pad(lab, pad)
    fg = np.argwhere((labp > 0) & (labp <= 4))
    c = (fg.min(0) + fg.max(0)) // 2
    st = [int(min(max(ci - p // 2, 0), s - p)) for ci, p, s in zip(c, PATCH, labp.shape)]
    return pad, tuple(slice(s, s + p) for s, p in zip(st, PATCH))


def to_crop(pid, label_path):
    d = VAL_TF({"image": str(ROOT / "data" / pid / "image.nii.gz"), "label": str(label_path)})
    return d["image"][0].numpy().astype(np.float32), d["label"][0].numpy().astype(np.int16)


def load_patient(pid):
    img, ref = to_crop(pid, ROOT / "data_6oar" / pid / "label.nii.gz")
    pad, sl = crop_slices(ref)
    img_c, ref_c = np.pad(img, pad)[sl], np.pad(ref, pad)[sl]
    saved = np.load(npz[("seg6", pid)])
    err_img = float(np.abs(img_c - saved["image"].astype(np.float32)).max())
    same_lab = float((ref_c == saved["label"]).mean())   # voxel agreement local vs Kaggle transform
    ref_c = saved["label"].astype(np.int16)             # reference = exactly the label used by the XAI job
    preds = {}
    with tempfile.TemporaryDirectory() as td:
        for _, name, var in MODELS:
            r, n = zips[(var, pid)]
            with zipfile.ZipFile(r) as z:
                p = Path(td) / f"{name}.nii.gz"
                p.write_bytes(z.read(n))
            _, pr = to_crop(pid, p)
            preds[name] = np.pad(pr, pad)[sl]
    maps = {name: np.load(npz[(k, pid)]) for k, name, _ in MODELS}
    return img_c, ref_c, preds, maps, err_img, same_lab


def ax_slice(a, z):
    return np.fliplr(a[:, :, z].T)


def choose_slice(ref):
    """Axial slice with lung where heart, oesophagus, spinal cord and aorta are all present; largest heart area."""
    pres = np.stack([(ref == k).any((0, 1)) for k in (1, 3, 4, 5, 6)]).all(0)
    area = (ref == 3).sum((0, 1)) * pres
    return int(np.argmax(area)) if area.max() > 0 else int(np.argmax((ref == 3).sum((0, 1))))


def show_base(ax, img2d, lo, hi):
    ax.imshow(img2d, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def figure(pid, level="sortie"):
    img, ref, preds, maps, err_img, same_lab = load_patient(pid)
    print(f"  {pid}: max |image crop - saved| = {err_img:.2e} ; reference voxel agreement = {same_lab:.5f}")
    assert err_img < 1e-2 and same_lab >= 0.995, "crop reproduction failed"
    z = choose_slice(ref)
    lo, hi = np.percentile(img, [0.5, 99.5])
    ncol = 2 + len(HEAT_ORG)
    fig, axes = plt.subplots(2, ncol, figsize=(2.45 * ncol, 6.9))
    im = None
    for r, (key, name, var) in enumerate(MODELS):
        base = ax_slice(img, z)
        # col 0: reference segmentation ; col 1: predicted segmentation
        for c, (lab3, ttl) in enumerate(((ref, "Reference"), (preds[name], f"{name} prediction"))):
            ax = axes[r, c]
            show_base(ax, base, lo, hi)
            lab2 = ax_slice(lab3, z)
            ax.imshow(np.ma.masked_equal(lab2, 0), cmap=LABCMAP, vmin=0, vmax=6, alpha=0.40, origin="lower",
                      interpolation="nearest")
            for k, o in enumerate(ORG, start=1):
                if (lab2 == k).any():
                    ax.contour(lab2 == k, levels=[0.5], colors=[COL[o]], linewidths=0.8)
            ax.set_title(ttl if r == 0 or c == 1 else "", fontsize=8.5, color=INK)
            if c == 0:
                ax.set_ylabel(name, fontsize=11, color=INK)
        # heat maps per organ
        for j, org in enumerate(HEAT_ORG):
            ax = axes[r, 2 + j]
            show_base(ax, base, lo, hi)
            h = ax_slice(maps[name][f"{org}__{level}"].astype(np.float32), z)
            im = ax.imshow(np.ma.masked_less(h, 0.05), cmap=HEAT, vmin=0, vmax=1, alpha=0.6, origin="lower",
                           interpolation="nearest")
            k = ORG.index(org) + 1
            g2, p2 = ax_slice(ref, z) == k, ax_slice(preds[name], z) == k
            if g2.any():
                ax.contour(g2, levels=[0.5], colors=["#22d3ee"], linewidths=1.0)
            if p2.any():
                ax.contour(p2, levels=[0.5], colors=["white"], linewidths=0.9, linestyles="dashed")
            ior, dsc = IOR.get((key, pid, level, org), np.nan), DSC.get((var, pid, org), np.nan)
            ax.set_title(f"{EN[org]} heat map" if r == 0 else "", fontsize=8.5, color=INK)
            ax.text(0.03, 0.04, f"3D DSC {dsc:.2f} | 3D in-organ {ior:.2f}", transform=ax.transAxes, color="white",
                    fontsize=7, bbox=dict(facecolor="black", alpha=0.55, lw=0, pad=1.5))
        for c in range(ncol):
            axes[r, c].text(0.02, 0.5, "R", transform=axes[r, c].transAxes, color="white", fontsize=7, va="center")
            axes[r, c].text(0.94, 0.5, "L", transform=axes[r, c].transAxes, color="white", fontsize=7, va="center")
    lvl = {"sortie": "output logits", "dec_pre_tete": "pre-head map", "enc_profond": "deepest encoder"}[level]
    fig.suptitle(f"{pid.replace('PATIENT_', '')} — six-OAR segmentation and Seg-Grad-CAM ({lvl}), thoracic crop, "
                 f"axial slice {z}", fontsize=10.5, color=INK)
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[o], alpha=0.8, label=EN[o]) for o in ORG] + \
        [plt.Line2D([], [], color="#22d3ee", lw=1.5, label="Reference contour"),
         plt.Line2D([], [], color="black", lw=1.2, ls="dashed", label="Predicted contour (white)")]
    fig.legend(handles=h, loc="lower left", ncol=4, frameon=False, fontsize=7.8, bbox_to_anchor=(0.01, 0.0))
    fig.subplots_adjust(left=0.04, right=0.995, top=0.90, bottom=0.14, wspace=0.05, hspace=0.16)
    cax = fig.add_axes([0.70, 0.07, 0.25, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Seg-Grad-CAM (< 0.05 transparent)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    name = f"seg_gradcam_thorax_{pid.replace('PATIENT_', '')}_{level}"
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("  figure", name)


if __name__ == "__main__":
    pids = sorted({p for (k, p) in npz if k == "seg6"} & {p for (k, p) in npz if k == "unet6"})
    for pid in pids:
        for lv in ("sortie", "dec_pre_tete"):
            figure(pid, lv)
