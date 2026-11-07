# -*- coding: utf-8 -*-
"""Thorax-focused six-OAR segmentation figures (reference vs U-Net vs SegResNet), cropped to the thorax.

Same sources and display rules as make_segmentation_images.py (imported). Only patients whose examination covers
the lungs completely (lung labels touch neither the first nor the last axial slice) are used.
Every view is cropped to the bounding box of the lungs and heart in the reference (+ 10 mm margin):
four axial levels from apex to base (85 / 62 / 40 / 18 % of the lung extent above its base), one coronal slice
through the heart and one sagittal slice through the spinal cord.

    .venv\\Scripts\\python.exe phase4_6oar/figures/segmentations/make_thorax_images.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import make_segmentation_images as S

HERE = Path(__file__).resolve().parent
PATIENTS = ["PATIENT_s0547", "PATIENT_s0478", "PATIENT_s0215", "PATIENT_s0202", "PATIENT_s0175"]
LEVELS = [(0.85, "near apex"), (0.62, "upper"), (0.40, "middle"), (0.18, "near base")]
MARGIN_MM = 10.0


def full_coverage_patients():
    out = []
    for p in sorted((S.ROOT / "data_6oar").glob("PATIENT_s*/label.nii.gz")):
        a = np.asarray(nib.as_closest_canonical(nib.load(str(p))).dataobj)
        lz = np.flatnonzero(((a == 1) | (a == 2)).any((0, 1)))
        if lz.min() > 0 and lz.max() < a.shape[2] - 1:
            out.append(p.parent.name)
    return out


def thorax_box(G, zooms):
    thor = (G == 1) | (G == 2) | (G == 3)
    idx = np.argwhere(thor)
    mx, my, mz = (int(round(MARGIN_MM / z)) for z in zooms)
    lo = np.maximum(idx.min(0) - [mx, my, mz], 0)
    hi = np.minimum(idx.max(0) + [mx, my, mz] + 1, G.shape)
    lz = np.flatnonzero(((G == 1) | (G == 2)).any((0, 1)))
    return lo, hi, lz


def thorax_views(G, zooms):
    (x0, y0, z0), (x1, y1, z1), lz = thorax_box(G, zooms)
    sx, sy, sz = zooms
    v = []
    for frac, name in LEVELS:
        z = int(round(lz.min() + frac * (lz.max() - lz.min())))
        v.append((f"Axial, {name} (slice {z})", lambda A, z=z: np.fliplr(A[x0:x1, y0:y1, z].T), sy / sx, ("R", "L")))
    heart = np.argwhere(G == 3)
    yc = int(np.median(heart[:, 1]))
    cord = np.argwhere(G == 5)
    xs = int(np.median(cord[:, 0]))
    v.append((f"Coronal (slice {yc})", lambda A: np.fliplr(A[x0:x1, yc, z0:z1].T), sz / sx, ("R", "L")))
    v.append((f"Sagittal (slice {xs})", lambda A: np.fliplr(A[xs, y0:y1, z0:z1].T), sz / sy, ("A", "P")))
    return v


def patient_figure(pid):
    I, vols, zooms = S.load_case(pid)
    lo, hi = np.percentile(I, [0.5, 99.5])
    vw = thorax_views(vols["Reference"], zooms)
    fig, axes = plt.subplots(3, len(vw) + 1, figsize=(2.55 * len(vw) + 2.4, 8.4),
                             gridspec_kw={"width_ratios": [1] * len(vw) + [0.95]})
    for r, name in enumerate(("Reference", "U-Net", "SegResNet")):
        for c, (title, f, asp, (left, right)) in enumerate(vw):
            ax = axes[r, c]
            S.draw(ax, f(I), f(vols[name]), lo, hi, asp)
            if r == 0:
                ax.set_title(title, fontsize=8.5, color=S.INK)
            if c == 0:
                ax.set_ylabel(name, fontsize=11, color=S.INK)
            ax.text(0.02, 0.5, left, transform=ax.transAxes, color="white", fontsize=7, va="center")
            ax.text(0.94, 0.5, right, transform=ax.transAxes, color="white", fontsize=7, va="center")
        ax = axes[r, -1]
        ax.axis("off")
        if name == "Reference":
            meta = S.pd.read_csv(S.ROOT / "TotalsegmentatorMRI_dataset_v200/meta.csv", sep=";",
                                 encoding="utf-8-sig").set_index("image_id").loc[pid.replace("PATIENT_", "")]
            txt = (f"Reference labels\n\n{str(meta.study_type).replace('unkown', 'unknown')}\n"
                   f"{meta.manufacturer} {meta.magnetic_field_strength} T\nsequence {meta.scanning_sequence}\n"
                   f"voxel {zooms[0]:.2f} x {zooms[1]:.2f} x {zooms[2]:.2f} mm")
        else:
            txt = "\n".join([f"DSC ({name})"] + [f"{S.EN[o]:<12}{S.DSC.get((name, pid, o), np.nan):6.3f}" for o in S.ORG])
        ax.text(0.02, 0.5, txt, va="center", family="monospace", fontsize=8.2, color=S.INK)
    S.legend(fig)
    fig.suptitle(f"{pid.replace('PATIENT_', '')} — six-OAR segmentation, thorax-focused views "
                 f"(cropped to lungs and heart + {MARGIN_MM:.0f} mm)", fontsize=11, color=S.INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    name = f"thorax_{pid.replace('PATIENT_', '')}"
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("  figure", name)


def gallery(pids, level=0.40):
    fig, axes = plt.subplots(3, len(pids), figsize=(2.6 * len(pids), 8.0), squeeze=False)
    for j, pid in enumerate(pids):
        I, vols, zooms = S.load_case(pid)
        lo, hi = np.percentile(I, [0.5, 99.5])
        (x0, y0, _), (x1, y1, _), lz = thorax_box(vols["Reference"], zooms)
        z = int(round(lz.min() + level * (lz.max() - lz.min())))
        asp = zooms[1] / zooms[0]
        for r, name in enumerate(("Reference", "U-Net", "SegResNet")):
            f = lambda A: np.fliplr(A[x0:x1, y0:y1, z].T)  # noqa: E731
            S.draw(axes[r, j], f(I), f(vols[name]), lo, hi, asp)
            if j == 0:
                axes[r, j].set_ylabel(name, fontsize=11, color=S.INK)
            if name != "Reference":
                mean = np.nanmean([S.DSC.get((name, pid, o), np.nan) for o in S.ORG])
                axes[r, j].text(0.03, 0.04, f"mean DSC {mean:.3f}", transform=axes[r, j].transAxes,
                                color="white", fontsize=7.5, bbox=dict(facecolor="black", alpha=0.55, lw=0, pad=1.5))
        axes[0, j].set_title(pid.replace("PATIENT_", ""), fontsize=9.5, color=S.INK)
    S.legend(fig)
    fig.suptitle(f"Mid-thorax axial slice ({int(level * 100)} % of the lung extent above its base), "
                 "patients with complete lung coverage, ordered by SegResNet mean DSC (left = highest)",
                 fontsize=11, color=S.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    name = "thorax_gallery_mid"
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("  figure", name)


if __name__ == "__main__":
    full = full_coverage_patients()
    print(f"{len(full)} patients with complete lung coverage")
    for pid in PATIENTS:
        assert pid in full, pid
        patient_figure(pid)
    # gallery: 8 full-coverage patients spread over the SegResNet mean-DSC range
    m = S.met[(S.met.m == "SegResNet") & S.met.patient_id.isin(full)].groupby("patient_id").dsc.mean().sort_values()
    picks = [m.index[int(round(q * (len(m) - 1)))] for q in np.linspace(1, 0, 8)]
    gallery(picks)
