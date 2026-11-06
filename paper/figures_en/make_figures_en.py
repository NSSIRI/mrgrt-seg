"""English versions of the article figures (PNG 300 dpi + PDF) -> paper/figures_en/. Read-only on all data.

Same data and same style as the French report figures (results/analyse_complete, phase3_ablation/figures,
phase4_6oar/figures); only the labels are translated. Raw values only (no smoothing).

    .venv\\Scripts\\python.exe paper/figures_en/make_figures_en.py
"""
import gzip
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
AB = ROOT / "phase3_ablation/analyse"
P2 = ROOT / "results/phase2_eval"
sys.path.insert(0, str(AB))
from ablation_stats import load as load_ablation  # noqa: E402

ORG4 = ["poumon_g", "poumon_d", "coeur", "oesophage"]
EN = {"poumon_g": "Left lung", "poumon_d": "Right lung", "coeur": "Heart", "oesophage": "Oesophagus",
      "moelle": "Spinal cord", "aorte": "Aorta"}
E1C, E2C, E3C, VC = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.family": "DejaVu Sans"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  figure", name)


def boxes(ax, groups, labels_ticks, logy=False):
    data = [g[0] for g in groups]
    bp = ax.boxplot(data, positions=[g[1] for g in groups], widths=0.7, patch_artist=True,
                    medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                    flierprops={"marker": ".", "ms": 2.5, "mec": MUTED, "alpha": 0.5})
    for b, g in zip(bp["boxes"], groups):
        b.set_facecolor(g[2])
        b.set_edgecolor(SURF)
        b.set_linewidth(1.5)
    ax.set_xticks(*labels_ticks)
    if logy:
        ax.set_yscale("log")


# ---------------------------------------------------------------- Fig. 1 : E1 / E3 / E2, 187 patients
def fig1():
    c = pd.read_csv(P2 / "analyse/combined_long.csv")
    c = c[c.space == "native"]
    order = [("E1_UNet_base", "E1 U-Net 7.9 M", E1C), ("E3_UNet_opt", "E3 U-Net 18.8 M", E3C),
             ("E2_SegResNet", "E2 SegResNet 18.8 M", E2C)]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, metric, ylab, logy in [(axs[0], "dsc", "DSC", False), (axs[1], "hd95_mm", "HD95 (mm, log scale)", True)]:
        groups = [(c[(c.model == m) & (c.organ == o)][metric].dropna().to_numpy(), i * 4 + j, col)
                  for i, o in enumerate(ORG4) for j, (m, _, col) in enumerate(order)]
        boxes(ax, groups, ([i * 4 + 1 for i in range(4)], [EN[o] for o in ORG4]), logy)
        ax.set_ylabel(ylab)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in order], [lab for _, lab, _ in order],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    fig.text(0.01, -0.03, "187 patients, native label grid, 5-fold cross-validation. Empty predictions excluded from HD95.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig1_capacity_matched_boxplots")


# ---------------------------------------------------------------- Fig. 2 : paired differences vs E3
LAB = {"A1": "A1 U-Net + residual units", "A2": "A2 U-Net + GroupNorm", "A3": "A3 SegResNet + InstanceNorm"}


def fig2():
    q2 = pd.read_csv(AB / "stats_q2_variantes.csv")
    q1 = pd.read_csv(AB / "stats_q1_graines.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, metric, xl in [(axs[0], "dsc", "Paired difference in DSC vs E3 (median, 95 % CI)"),
                           (axs[1], "hd95_mm", "Paired difference in HD95 vs E3, mm (median, 95 % CI)")]:
        y, ticks, labs = 0, [], []
        for o in ORG4:
            rows = [("E2 SegResNet", E2C, q1[(q1.metrique == metric) & (q1.organe == o) &
                                              (q1.comparaison == "E2-E3 graine 42")].iloc[0])]
            for v in ["A3", "A1", "A2"]:
                rows.append((LAB[v], VC, q2[(q2.metrique == metric) & (q2.organe == o) & (q2.variante == v) &
                                            (q2.reference == "E3")].iloc[0]))
            thr = q2[(q2.metrique == metric) & (q2.organe == o)].seuil_graine.iloc[0]
            ax.fill_between([-thr, thr], y + 0.5, y - len(rows) + 0.5, color=GRID, alpha=0.8, lw=0, zorder=0)
            for lab, col, r in rows:
                ax.plot([r.ic_bas, r.ic_haut], [y, y], color=col, lw=2, solid_capstyle="round")
                ax.plot(r.med_diff, y, "o", color=col, ms=6, mec=SURF, mew=1.2)
                ticks.append(y)
                labs.append(f"{EN[o]} — {lab}")
                y -= 1
            y -= 0.7
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(ticks, labs, fontsize=7.5)
        ax.set_xlabel(xl)
    axs[1].set_yticklabels([])
    fig.text(0.01, -0.03, "Grey band: training-seed variability (largest |95 % CI bound| of the seed effect of E3 and E2). "
             "187 patients, native grid.", fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "fig2_paired_differences_vs_E3")


# ---------------------------------------------------------------- Fig. 3 : share of the gap closed
def fig3():
    g = pd.read_csv(AB / "stats_part_ecart.csv")
    g = g[g.metrique == "dsc"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs = np.arange(4)
    for k, v in enumerate(["A2", "A1", "A3"]):
        s = g[g.variante == v].set_index("organe").loc[ORG4]
        x = xs + (k - 1) * 0.22
        ax.errorbar(x, s.part_comblee, yerr=[s.part_comblee - s.ic_bas, s.ic_haut - s.part_comblee], fmt="o",
                    color=VC, ms=6, mec=SURF, mew=1.2, elinewidth=2, capsize=0)
        for xi, yi in zip(x, s.part_comblee):
            ax.annotate(v, (xi, yi), textcoords="offset points", xytext=(7, -3), fontsize=7.5, color=INK2)
    ax.axhline(0, color=E3C, lw=1.5)
    ax.axhline(1, color=E2C, lw=1.5)
    ax.text(3.55, 0.04, "E3 level", color=INK2, fontsize=7.5)
    ax.text(3.55, 1.04, "E2 level", color=INK2, fontsize=7.5)
    ax.set_xticks(xs, [EN[o] for o in ORG4])
    ax.set_ylabel("Share of the E3 → E2 DSC gap closed")
    fig.text(0.01, -0.04, "Share = median(variant − E3) / median(E2 − E3), paired by patient; 95 % bootstrap CI "
             "(10,000 resamples). A1, A2: modified U-Net; A3: modified SegResNet.", fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "fig3_share_of_gap_closed")


# ---------------------------------------------------------------- Fig. 4 : seed reproducibility
def fig4(df):
    fm = df[df.m.str.startswith(("E2", "E3"))].groupby(["m", "fold", "organ"]).dsc.mean().reset_index()
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, o in zip(axs, ORG4):
        for mod, col in [("E3", E3C), ("E2", E2C)]:
            a = fm[(fm.m == f"{mod}_s42") & (fm.organ == o)].set_index("fold").dsc
            b = fm[(fm.m == f"{mod}_s123") & (fm.organ == o)].set_index("fold").dsc
            ax.scatter(a, b.reindex(a.index), color=col, s=30, edgecolor=SURF, lw=1, label=mod, zorder=3)
        lo, hi = fm[fm.organ == o].dsc.min() - 0.02, fm[fm.organ == o].dsc.max() + 0.02
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(EN[o], fontsize=10, color=INK)
        ax.set_xlabel("Fold mean DSC, seed 42")
    axs[0].set_ylabel("Fold mean DSC, seed 123")
    axs[0].legend(loc="upper left", fontsize=8)
    fig.text(0.01, -0.06, "Each point is one fold (mean over its patients). Diagonal: perfect reproducibility.",
             fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "fig4_seed_reproducibility")


# ---------------------------------------------------------------- Fig. 5 : robustness
def fig5():
    r = pd.read_csv(AB / "ablation_masks.csv")
    order = [("E3_s42", "E3", E3C), ("A2", "A2", VC), ("A1", "A1", VC), ("E2_s42", "E2", E2C), ("A3", "A3", VC)]
    frag = r.groupby(["model", "organ"]).apply(lambda g: 100 * (g.frac_out_lcc > 0.01).mean(), include_groups=False)
    lat = r.drop_duplicates(["patient_id", "model"]).assign(sw=lambda t: t[["swap_G", "swap_D"]].max(axis=1) > 0.05)
    lat = lat.groupby("model").sw.mean() * 100
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [3, 1]})
    ax = axs[0]
    for k, (m, lab, c) in enumerate(order):
        vals = [frag.get((m, o), np.nan) for o in ORG4]
        ax.bar(np.arange(4) + (k - 2) * 0.16, vals, width=0.15, color=c)
        for i, v in enumerate(vals):
            ax.text(i + (k - 2) * 0.16, v + 1, lab, ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(range(4), [EN[o] for o in ORG4])
    ax.set_ylabel("% of patients with > 1 % of the predicted\nvolume outside the largest component")
    ax = axs[1]
    ax.bar(range(5), [lat.get(m, np.nan) for m, _, _ in order], color=[c for _, _, c in order])
    ax.set_xticks(range(5), [lab for _, lab, _ in order])
    ax.set_ylabel("% of patients, left/right lung confusion > 5 %")
    fig.tight_layout()
    save(fig, "fig5_fragmentation_laterality")


# ---------------------------------------------------------------- Fig. 6 : six OAR
def fig6():
    runs = []
    for f in sorted((ROOT / "results/phase3_ablation").glob("*/abl18m-*6-s42-f*/ablation_eval/metrics_long.csv")):
        d = pd.read_csv(f)
        runs.append(d[d.space == "native"])
    df = pd.concat(runs).drop_duplicates(["variant", "fold", "patient_id", "organ"])
    org6 = ORG4 + ["moelle", "aorte"]
    order = [("unet_6oar", "U-Net 18.8 M", E3C), ("segresnet_6oar", "SegResNet 18.8 M", E2C)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, metric, ylab, logy in [(axs[0], "dsc", "DSC", False), (axs[1], "hd95_mm", "HD95 (mm, log scale)", True)]:
        groups = [(df[(df.variant == v) & (df.organ == o)][metric].dropna().to_numpy(), i * 3 + j, col)
                  for i, o in enumerate(org6) for j, (v, _, col) in enumerate(order)]
        boxes(ax, groups, ([i * 3 + 0.5 for i in range(6)], [EN[o] for o in org6]), logy)
        ax.set_ylabel(ylab)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in order], [lab for _, lab, _ in order],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig.text(0.01, -0.03, "Six-OAR cohort (115 patients, 5 folds of 23), native grid, seed 42.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig6_six_oar_boxplots")


# ---------------------------------------------------------------- Fig. 7 : qualitative overlays (same 2 patients as the report)
def fig7():
    c = pd.read_csv(P2 / "analyse/combined_long.csv")
    c = c[c.space == "native"]
    zips = {"E3_UNet_opt": [P2 / "abdelhalimnssiri_b/phase2_eval/predictions.zip", P2 / "nssiri02_b/phase2_eval/predictions.zip"],
            "E2_SegResNet": [P2 / "nssiri02_b/phase2_eval/predictions.zip"]}
    var = {"E3_UNet_opt": "unet_opt", "E2_SegResNet": "segresnet"}
    col = {"E3_UNet_opt": E3C, "E2_SegResNet": E2C}

    def pred(m, fold, pid):
        for z in zips[m]:
            with zipfile.ZipFile(z) as zf:
                n = f"{var[m]}/fold{fold}/{pid}_pred.nii.gz"
                if n in zf.namelist():
                    return np.asanyarray(nib.Nifti1Image.from_bytes(gzip.decompress(zf.read(n))).dataobj).astype(np.uint8)
        raise FileNotFoundError(pid)

    e3 = c[(c.model == "E3_UNet_opt") & (c.organ == "oesophage")].dropna(subset=["dsc"]).sort_values("dsc")
    picks = [("median E3 oesophagus DSC", e3.iloc[len(e3) // 2]), ("worst E3 oesophagus DSC", e3.iloc[0])]
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    for r, (why, row) in enumerate(picks):
        pid, fold = row.patient_id, int(row.fold)
        img = np.asanyarray(nib.load(str(ROOT / "data" / pid / "image.nii.gz")).dataobj).astype(np.float32)
        gt = np.rint(np.asanyarray(nib.load(str(ROOT / "data" / pid / "label.nii.gz")).dataobj)).astype(np.uint8)
        preds = {m: pred(m, fold, pid) for m in var}
        ref = gt == 4 if (gt == 4).any() else gt == 3
        z = int(np.round(ndi.center_of_mass(ref)[2]))
        lo, hi = np.percentile(img[..., z], [0.5, 99.5])
        sl = lambda a: np.rot90(a[..., z])  # noqa: E731
        d = c[(c.patient_id == pid) & (c.organ == "oesophage")].set_index("model").dsc
        for k, (title, m) in enumerate([("Reference", None), ("E3 U-Net 18.8 M", "E3_UNet_opt"), ("E2 SegResNet", "E2_SegResNet")]):
            ax = axs[r, k]
            ax.imshow(sl(img), cmap="gray", vmin=lo, vmax=hi)
            for cl in range(1, 5):
                if (gt[..., z] == cl).any():
                    ax.contour(sl(gt == cl).astype(float), levels=[0.5], colors=["white"], linewidths=1.0,
                               linestyles="--" if m else "-")
                if m and (preds[m][..., z] == cl).any():
                    ax.contour(sl(preds[m] == cl).astype(float), levels=[0.5], colors=[col[m]], linewidths=1.4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(title + (f"\nOesophagus DSC {d[m]:.2f}" if m else f"\n{pid}, fold {fold} — {why}"), fontsize=8.5)
    fig.text(0.01, 0.0, "White: reference (dashed in prediction columns). Colour: model prediction (4 OAR). Axial slice "
             "through the centre of the reference oesophagus; display orientation not radiological.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1), h_pad=2.5)
    save(fig, "fig7_qualitative_overlays")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4(load_ablation())
    fig5()
    fig6()
    fig7()
