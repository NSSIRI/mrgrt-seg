# -*- coding: utf-8 -*-
"""DLAS manuscript (six thoracic OAR, SegResNet, MONAI) : tables, outlier analysis and figures from RAW files.

Read-only sources:
  TotalsegmentatorMRI_dataset_v200/meta.csv                              cohort metadata
  data/, data_6oar/                                                      images / 6-OAR reference labels
  results/phase3_ablation/*/abl18m-{seg6,unet6}-s42-f*/                  per-patient metrics (native grid),
      ablation_eval/{metrics_long.csv,runs_summary.csv,histories/,predictions.zip}, training logs
  phase4_6oar/thorax/metrics_thorax_long.csv                             thoracic-region metrics
Writes paper/dlas_6oar/tables/*.csv and paper/dlas_6oar/figures/*.png (300 dpi). Raw values only (no smoothing).

    .venv\\Scripts\\python.exe paper/dlas_6oar/analysis_dlas.py
"""
import glob
import gzip
import json
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.stats import mannwhitneyu, fisher_exact  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
ORG = ["poumon_g", "poumon_d", "coeur", "oesophage", "moelle", "aorte"]
EN = {"poumon_g": "Left lung", "poumon_d": "Right lung", "coeur": "Heart", "oesophage": "Oesophagus",
      "moelle": "Spinal cord", "aorte": "Aorta"}
SEG, UNET = "#eb6834", "#1baf7a"
OCOL = {"poumon_g": "#2a78d6", "poumon_d": "#6250d6", "coeur": "#e0383e", "oesophage": "#1baf7a",
        "moelle": "#f2b51d", "aorte": "#eb6834"}
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.family": "DejaVu Sans"})
# Pre-specified per-organ failure thresholds (a single DSC 0.7 threshold is uninformative for the oesophagus,
# whose median DSC is ~0.68): DSC < 0.7 for lungs, heart, spinal cord and aorta; DSC < 0.5 for the oesophagus.
THR = {"poumon_g": 0.7, "poumon_d": 0.7, "coeur": 0.7, "oesophage": 0.5, "moelle": 0.7, "aorte": 0.7}

RUNS = sorted(glob.glob(str(ROOT / "results/phase3_ablation/*/abl18m-*6-s42-f*")))


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  figure", name)


def medq(s, k=3):
    s = s.dropna()
    return f"{s.median():.{k}f} [{s.quantile(.25):.{k}f}–{s.quantile(.75):.{k}f}]"


# ------------------------------------------------------------------ data
met = pd.concat(pd.read_csv(Path(r) / "ablation_eval/metrics_long.csv") for r in RUNS)
met = met[met.space == "native"].drop_duplicates(["variant", "fold", "patient_id", "organ"])
met["m"] = met.variant.map({"segresnet_6oar": "SegResNet", "unet_6oar": "U-Net"})
assert met.groupby("m").patient_id.nunique().eq(115).all()
seg = met[met.m == "SegResNet"]
meta = pd.read_csv(ROOT / "TotalsegmentatorMRI_dataset_v200/meta.csv", sep=";", encoding="utf-8-sig")
meta["patient_id"] = "PATIENT_" + meta.image_id
ids = sorted(seg.patient_id.unique())
mt = meta[meta.patient_id.isin(ids)].copy()
assert len(mt) == 115
thx = pd.read_csv(ROOT / "phase4_6oar/thorax/metrics_thorax_long.csv")

# ------------------------------------------------------------------ T1 cohort
age = pd.to_numeric(mt.age, errors="coerce")
st = pd.to_numeric(mt.slice_thickness, errors="coerce")
fs = pd.to_numeric(mt.magnetic_field_strength, errors="coerce")
rows = [("Patients (examinations)", "115"),
        ("Age, years, median [IQR] (available)", f"{age.median():.0f} [{age.quantile(.25):.0f}–{age.quantile(.75):.1f}] (n = {age.notna().sum()})"),
        ("Age range, years", f"{age.min():.0f}–{age.max():.0f}"),
        ("Sex: male / female / not reported", f"{(mt.gender == 'm').sum()} / {(mt.gender == 'f').sum()} / {mt.gender.isna().sum()}"),
        ("Centres (A / B / C / D)", " / ".join(str((mt.institute == c).sum()) for c in "ABCD")),
        ("Field strength: 1.5 T / 3 T / not reported", f"{(fs == 1.5).sum()} / {(fs == 3.0).sum()} / {fs.isna().sum()}"),
        ("Manufacturer: Siemens / GE / Philips / unknown", " / ".join(str((mt.manufacturer == x).sum()) for x in ("siemens", "ge", "philips", "unkown"))),
        ("Sequence: gradient echo / spin echo / other or unknown",
         f"{(mt.scanning_sequence == 'GR').sum()} / {(mt.scanning_sequence == 'SE').sum()} / {(~mt.scanning_sequence.isin(['GR', 'SE'])).sum()}"),
        ("Slice thickness, mm, median [range]", f"{st.median():.1f} [{st.min():.2f}–{st.max():.1f}] (n = {st.notna().sum()})"),
        ("Study type (top): abdomen / heart / chest / unknown / other",
         f"{mt.study_type.str.contains('abdomen', na=False).sum()} / {(mt.study_type == 'mri heart').sum()} / "
         f"{(mt.study_type == 'mri chest').sum()} / {(mt.study_type == 'unkown').sum()} / "
         f"{115 - mt.study_type.str.contains('abdomen', na=False).sum() - (mt.study_type == 'mri heart').sum() - (mt.study_type == 'mri chest').sum() - (mt.study_type == 'unkown').sum()}")]
pd.DataFrame(rows, columns=["Characteristic", "Value"]).to_csv(TAB / "T1_cohort.csv", index=False)

# native voxel spacing from the reference labels (all 115)
sp = []
for pid in ids:
    h = nib.load(str(ROOT / "data_6oar" / pid / "label.nii.gz"))
    a = h.affine[:3, :3]
    z = np.sqrt((a ** 2).sum(0))
    sp.append({"patient_id": pid, "inplane_mm": float(min(z[0], z[1])), "slice_mm": float(z[2]),
               "nx": h.shape[0], "ny": h.shape[1], "nz": h.shape[2]})
sp = pd.DataFrame(sp)
sp.to_csv(TAB / "native_spacing.csv", index=False)

# ------------------------------------------------------------------ T2 contours per OAR and per fold
folds = seg.groupby("patient_id").fold.first()
vol = seg.pivot_table(index="patient_id", columns="organ", values="vol_gt_ml")
t2 = []
for o in ORG:
    r = {"OAR": EN[o]}
    for k in range(5):
        r[f"Fold {k} (validation)"] = int((vol.loc[folds[folds == k].index, o] > 0).sum())
    r["Total"] = int((vol[o] > 0).sum())
    r["Reference volume, mL, median [IQR]"] = medq(vol[o], 1)
    t2.append(r)
pd.DataFrame(t2).to_csv(TAB / "T2_contours.csv", index=False)

# ------------------------------------------------------------------ training statistics
hist, rs, ep_time = [], [], []
for r in RUNS:
    s = pd.read_csv(Path(r) / "ablation_eval/runs_summary.csv")
    for _, x in s.iterrows():
        rs.append({"model": x.variant, "fold": x.fold, "best_epoch": x.best_epoch, "best_val_dsc": x.best_val_dsc,
                   "n_params": x.n_params, "monai": x.train_monai, "torch": x.train_torch})
    for h in glob.glob(str(Path(r) / "ablation_eval/histories/*_history.csv")):
        d = pd.read_csv(h)
        mm = re.search(r"(segresnet_6oar|unet_6oar)_s42_fold(\d)", h)
        d["model"], d["fold"] = mm.group(1), int(mm.group(2))
        hist.append(d)
    log = next(Path(r).glob("abl18m-*.log"))
    secs = {}
    for e, t in re.findall(r"\[Epoch\s+(\d+)\][^\"]*?\((\d+\.\d)s\)", log.read_text(encoding="utf-8", errors="replace")):
        secs[int(e)] = float(t)                      # dedupe by epoch number (stdout can be duplicated)
    mdl = "segresnet_6oar" if "seg6" in r else "unet_6oar"
    ep_time += [{"model": mdl, "fold": int(r[-1]), "epoch": e, "sec": t} for e, t in secs.items()]
rs = pd.DataFrame(rs).drop_duplicates(["model", "fold"])
hist = pd.concat(hist).drop_duplicates(["model", "fold", "epoch"])
ep = pd.DataFrame(ep_time)
tr = (rs.groupby("model").agg(best_epoch_median=("best_epoch", "median"), best_epoch_min=("best_epoch", "min"),
                              best_epoch_max=("best_epoch", "max"), best_val_dsc_median=("best_val_dsc", "median"),
                              n_params=("n_params", "first"), monai=("monai", "first"), torch=("torch", "first"))
      .join(ep.groupby("model").agg(sec_per_epoch_median=("sec", "median"), n_epochs_logged=("sec", "size")))
      .join(ep.groupby(["model", "fold"]).sec.sum().groupby("model").median().rename("train_hours_median") / 3600))
inf = seg.groupby("patient_id").t_infer_s.first()
tr.loc["segresnet_6oar", "infer_s_median"] = inf.median()
tr.loc["unet_6oar", "infer_s_median"] = met[met.m == "U-Net"].groupby("patient_id").t_infer_s.first().median()
tr.round(4).to_csv(TAB / "training_stats.csv")

# ------------------------------------------------------------------ T4 geometric evaluation (SegResNet)
t4 = []
for o in ORG:
    s = seg[seg.organ == o]
    r = {"OAR": EN[o], "n": s.patient_id.nunique(), "DSC": medq(s.dsc), "HD95 (mm)": medq(s.hd95_mm, 1),
         "ASSD (mm)": medq(s.assd_mm, 2), "Surface DSC 2 mm": medq(s.sdsc_2mm),
         "Relative volume error": medq(s.vol_rel_err, 3),
         "Empty predictions": int(s.empty_pred.sum())}
    if o in ("moelle", "aorte"):
        q = thx[(thx.variant == "segresnet_6oar") & (thx.organ == o) & (thx.region == "thorax")]
        r.update({"DSC, lung slices": medq(q.dsc), "HD95, lung slices (mm)": medq(q.hd95_mm, 1),
                  "ASSD, lung slices (mm)": medq(q.assd_mm, 2)})
    t4.append(r)
pd.DataFrame(t4).to_csv(TAB / "T4_geometric_SegResNet.csv", index=False)
ub = []
for o in ORG:
    s, u = seg[seg.organ == o], met[(met.m == "U-Net") & (met.organ == o)]
    ub.append({"OAR": EN[o], "DSC U-Net": medq(u.dsc), "DSC SegResNet": medq(s.dsc),
               "HD95 U-Net": medq(u.hd95_mm, 1), "HD95 SegResNet": medq(s.hd95_mm, 1)})
pd.DataFrame(ub).to_csv(TAB / "T4b_baseline_UNet.csv", index=False)
# per-fold mean DSC (SegResNet)
seg.groupby(["fold", "organ"]).dsc.mean().unstack()[ORG].round(3).to_csv(TAB / "perfold_mean_dsc_SegResNet.csv")

# ------------------------------------------------------------------ outliers (SegResNet)
seg2 = seg.merge(mt[["patient_id", "institute", "magnetic_field_strength", "scanning_sequence", "study_type",
                     "manufacturer", "slice_thickness", "age", "gender"]], on="patient_id", how="left")
seg2 = seg2.merge(sp, on="patient_id", how="left")
seg2["fail"] = [d < THR[o] if pd.notna(d) else True for d, o in zip(seg2.dsc, seg2.organ)]
fails = seg2[seg2.fail].copy()


def n_components(pid, organ_idx, pred):
    lab, n = ndi.label(pred == organ_idx, structure=np.ones((3, 3, 3)))
    if n == 0:
        return 0, np.nan
    sizes = ndi.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    return int(n), float(1 - sizes.max() / sizes.sum())


zips = {}
for r in RUNS:
    if "seg6" in r:
        z = zipfile.ZipFile(Path(r) / "ablation_eval/predictions.zip")
        for m in z.namelist():
            mm = re.search(r"(PATIENT_s\d+)_pred", m)
            if mm:
                zips[mm.group(1)] = (Path(r) / "ablation_eval/predictions.zip", m)


def load_pred(pid):
    zp, m = zips[pid]
    with zipfile.ZipFile(zp) as z:
        return nib.Nifti1Image.from_bytes(gzip.decompress(z.read(m)))


extra = []
for pid in sorted(fails.patient_id.unique()):
    pn = load_pred(pid)
    pred = np.asarray(pn.dataobj).astype(np.uint8)
    for o in fails[fails.patient_id == pid].organ:
        n, frac = n_components(pid, ORG.index(o) + 1, pred)
        extra.append({"patient_id": pid, "organ": o, "n_components": n, "frac_outside_largest": frac})
fails = fails.merge(pd.DataFrame(extra), on=["patient_id", "organ"], how="left")


def cause(r):
    c = []
    if r.empty_pred == 1:
        c.append("no prediction")
    if pd.notna(r.frac_outside_largest) and r.frac_outside_largest > 0.10:
        c.append(f"fragmented ({r.frac_outside_largest:.0%} outside largest component)")
    if pd.notna(r.vol_rel_err) and r.vol_rel_err < -0.4:
        c.append(f"under-segmentation (volume {r.vol_rel_err:+.0%})")
    if pd.notna(r.vol_rel_err) and r.vol_rel_err > 0.4:
        c.append(f"over-segmentation (volume {r.vol_rel_err:+.0%})")
    if r.slice_mm >= 6:
        c.append(f"thick slices ({r.slice_mm:.1f} mm)")
    if r.scanning_sequence not in ("GR", "SE"):
        c.append(f"atypical sequence ({r.scanning_sequence})")
    return "; ".join(c) if c else "boundary disagreement without gross error"


fails["probable_cause"] = fails.apply(cause, axis=1)
fails = fails.sort_values(["organ", "dsc"])
fails[["patient_id", "fold", "organ", "dsc", "hd95_mm", "vol_gt_ml", "vol_pred_ml", "vol_rel_err", "empty_pred",
       "n_components", "frac_outside_largest", "slice_mm", "inplane_mm", "magnetic_field_strength",
       "scanning_sequence", "study_type", "institute", "probable_cause"]].round(4).to_csv(TAB / "outliers_SegResNet.csv", index=False)

# association of failures with acquisition factors (patient level: any failure on a non-oesophagus OAR,
# and oesophagus failure separately)
pat = seg2.groupby("patient_id").agg(slice_mm=("slice_mm", "first"), inst=("institute", "first"),
                                     field=("magnetic_field_strength", "first"), seq=("scanning_sequence", "first"),
                                     study=("study_type", "first"))
pat["fail_major"] = seg2[(seg2.organ != "oesophage")].groupby("patient_id").fail.any()
pat["fail_oeso"] = seg2[seg2.organ == "oesophage"].groupby("patient_id").fail.any()
assoc = []
for flag in ("fail_major", "fail_oeso"):
    a, b = pat[pat[flag]].slice_mm, pat[~pat[flag]].slice_mm
    assoc.append({"outcome": flag, "factor": "native slice thickness (mm)", "failed_median": a.median(),
                  "ok_median": b.median(), "test": "Mann-Whitney", "p": mannwhitneyu(a, b).pvalue,
                  "n_failed": len(a), "n_ok": len(b)})
    for fac, val in (("field", 3.0), ("seq", "GR"), ("study", "mri heart")):
        x = pat[fac] == val
        tab = [[(x & pat[flag]).sum(), (x & ~pat[flag]).sum()], [(~x & pat[flag]).sum(), (~x & ~pat[flag]).sum()]]
        assoc.append({"outcome": flag, "factor": f"{fac} = {val}", "failed_median": tab[0][0], "ok_median": tab[0][1],
                      "test": "Fisher exact (counts: with factor failed / ok)", "p": fisher_exact(tab).pvalue,
                      "n_failed": int(pat[flag].sum()), "n_ok": int((~pat[flag]).sum())})
pd.DataFrame(assoc).round(4).to_csv(TAB / "outliers_association.csv", index=False)

# ------------------------------------------------------------------ figures
# F1 boxplots DSC / HD95 / ASSD per OAR (SegResNet, U-Net baseline in grey-green)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
rng = np.random.default_rng(20261106)
for ax, (col, lab, logy) in zip(axes, [("dsc", "DSC", False), ("hd95_mm", "HD95 (mm, log scale)", True),
                                       ("assd_mm", "ASSD (mm, log scale)", True)]):
    for i, o in enumerate(ORG):
        for j, (mname, c) in enumerate((("U-Net", UNET), ("SegResNet", SEG))):
            v = met[(met.m == mname) & (met.organ == o)][col].dropna().to_numpy()
            x = i + (j - 0.5) * 0.36
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=4, color=c, alpha=0.4, linewidths=0, zorder=2)
            ax.boxplot(v, positions=[x], widths=0.3, showfliers=False, patch_artist=True, zorder=3,
                       boxprops=dict(facecolor="none", edgecolor=INK, lw=0.9), medianprops=dict(color=INK, lw=1.3))
    ax.set_xticks(range(6), [EN[o].replace(" ", "\n") for o in ORG], fontsize=8)
    ax.set_ylabel(lab)
    if logy:
        ax.set_yscale("log")
h = [axes[0].scatter([], [], color=c, label=n) for n, c in (("U-Net (baseline)", UNET), ("SegResNet", SEG))]
fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout()
save(fig, "Fig2_boxplots_DSC_HD95_ASSD")

# F3 learning curves (raw per-epoch values, 5 folds)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for mdl, c in (("unet_6oar", UNET), ("segresnet_6oar", SEG)):
    for k in range(5):
        d = hist[(hist.model == mdl) & (hist.fold == k)].sort_values("epoch")
        lab = ("U-Net" if mdl == "unet_6oar" else "SegResNet") if k == 0 else None
        axes[0].plot(d.epoch, d.train_loss, color=c, lw=0.8, alpha=0.8, label=lab)
        axes[1].plot(d.epoch, d.val_dsc, color=c, lw=0.8, alpha=0.8, label=lab)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss (Dice + CE)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation mean DSC (6 OAR)")
axes[1].legend(loc="lower right")
fig.tight_layout()
save(fig, "Fig3_learning_curves")

# F4 overlays: median and worst patients (SegResNet mean DSC over 6 OAR)
mean_dsc = seg.groupby("patient_id").dsc.mean().sort_values()
picks = {"median": mean_dsc.index[len(mean_dsc) // 2], "worst": mean_dsc.index[0]}
fig, axes = plt.subplots(2, 3, figsize=(12, 8.2))
for r, (kind, pid) in enumerate(picks.items()):
    img = nib.as_closest_canonical(nib.load(str(ROOT / "data" / pid / "image.nii.gz")))
    gt = nib.as_closest_canonical(nib.load(str(ROOT / "data_6oar" / pid / "label.nii.gz")))
    pr = nib.as_closest_canonical(load_pred(pid))
    I, G, P = np.asarray(img.dataobj, float), np.asarray(gt.dataobj).astype(int), np.asarray(pr.dataobj).astype(int)
    zz = int(np.argmax(((G == 3) | (G == 4)).sum((0, 1))))          # axial slice with most heart + oesophagus
    yy = int(np.argmax((G > 0).sum((0, 2))))                          # coronal slice with most OAR
    lo, hi = np.percentile(I, [0.5, 99.5])
    for c_, (title, im, g, p, asp) in enumerate([
            ("Axial", I[:, :, zz].T, G[:, :, zz].T, P[:, :, zz].T, img.header.get_zooms()[1] / img.header.get_zooms()[0]),
            ("Coronal", I[:, yy, :].T, G[:, yy, :].T, P[:, yy, :].T, img.header.get_zooms()[2] / img.header.get_zooms()[0])]):
        ax = axes[r, c_]
        # canonical RAS arrays (+x = patient right) -> flip left/right for the radiological convention
        im, g, p = np.fliplr(im), np.fliplr(g), np.fliplr(p)
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect=asp)
        ax.text(0.02, 0.5, "R", transform=ax.transAxes, color="white", fontsize=9, va="center", weight="bold")
        ax.text(0.96, 0.5, "L", transform=ax.transAxes, color="white", fontsize=9, va="center", weight="bold")
        for k, o in enumerate(ORG, start=1):
            if (g == k).any():
                ax.contour(g == k, levels=[0.5], colors=[OCOL[o]], linewidths=1.2)
            if (p == k).any():
                ax.contour(p == k, levels=[0.5], colors=[OCOL[o]], linewidths=1.0, linestyles="dashed")
        ax.set_title(f"{title}", fontsize=9)
        ax.axis("off")
    ax = axes[r, 2]
    ax.axis("off")
    d = seg[seg.patient_id == pid].set_index("organ").loc[ORG]
    txt = [f"{'Median' if kind == 'median' else 'Worst'} case ({pid.replace('PATIENT_', '')})",
           f"mean DSC over 6 OAR = {mean_dsc[pid]:.3f}", ""] + \
          [f"{EN[o]:<12} DSC {d.loc[o, 'dsc']:.3f}  HD95 {d.loc[o, 'hd95_mm']:.1f} mm" for o in ORG]
    ax.text(0.0, 0.95, "\n".join(txt), va="top", family="monospace", fontsize=8.5, color=INK)
hl = [plt.Line2D([], [], color=OCOL[o], lw=2, label=EN[o]) for o in ORG] + \
     [plt.Line2D([], [], color=INK2, lw=1.2, label="Reference (solid)"),
      plt.Line2D([], [], color=INK2, lw=1.0, ls="dashed", label="SegResNet (dashed)")]
fig.legend(handles=hl, loc="lower center", ncol=8, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.03, 1, 1))
save(fig, "Fig4_overlays_median_worst")
json.dump({k: v for k, v in picks.items()}, open(TAB / "overlay_patients.json", "w"))
print("done")
