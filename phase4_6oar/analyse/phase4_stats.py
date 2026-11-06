"""Phase 4 - 6 OAR (cohorte D115) : SegResNet vs U-Net 18,8 M, 5 plis, graine 42. Lecture seule.

Statistiques par OAR (grille native) : mediane [Q1-Q3] ; difference appariee SegResNet - U-Net (Wilcoxon bilateral,
Holm par metrique sur les 6 OAR, IC 95 % bootstrap 10 000 tirages, r de rang biserial) ; DSC moyen par pli.
Figures : boites a moustaches (DSC, HD95) et differences appariees. Sorties dans phase4_6oar/analyse et /figures.

    .venv\\Scripts\\python.exe phase4_6oar/analyse/phase4_stats.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT.parent / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261106)
NBOOT = 10_000
ORG = ["poumon_g", "poumon_d", "coeur", "oesophage", "moelle", "aorte"]
ORGL = {"poumon_g": "Poumon G", "poumon_d": "Poumon D", "coeur": "Cœur", "oesophage": "Œsophage",
        "moelle": "Moelle", "aorte": "Aorte"}
METRICS = {"dsc": 1, "hd95_mm": -1, "assd_mm": -1, "sdsc_2mm": 1, "hdmax_mm": -1}
SEG, UNET = "#eb6834", "#1baf7a"
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.family": "DejaVu Sans"})


def load():
    runs = []
    for f in sorted((ROOT / "results/phase3_ablation").glob("*/abl18m-*6-s42-f*/ablation_eval/metrics_long.csv")):
        d = pd.read_csv(f)
        runs.append(d[d.space == "native"])
    df = pd.concat(runs, ignore_index=True).drop_duplicates(["variant", "fold", "patient_id", "organ"])
    df["m"] = df.variant.map({"segresnet_6oar": "SegResNet", "unet_6oar": "U-Net"})
    n = df.groupby(["m", "organ"]).patient_id.nunique()
    print("patients par modele x OAR :", n.unstack().to_dict())
    return df


def paired(x, y, sense):
    d = (x - y).dropna().to_numpy()
    nz = d[d != 0]
    rk = rankdata(np.abs(nz))
    r = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum() if len(nz) else np.nan
    p = wilcoxon(d, zero_method="wilcox").pvalue if len(nz) else 1.0
    bm = np.median(d[RNG.integers(0, len(d), size=(NBOOT, len(d)))], axis=1)
    return {"n": len(d), "med_diff": np.median(d), "ic_bas": np.percentile(bm, 2.5), "ic_haut": np.percentile(bm, 97.5),
            "pct_seg_meilleur": 100 * (d * sense > 0).mean(), "p": p, "r_rb": r}


def holm(p):
    p = np.asarray(p, float)
    o, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    df = load()
    desc = (df.groupby(["m", "organ"])[list(METRICS)]
            .agg(lambda x: f"{np.nanmedian(x):.3f} [{np.nanpercentile(x, 25):.3f}-{np.nanpercentile(x, 75):.3f}]"))
    desc.to_csv(OUT / "p4_descriptifs.csv")
    rows = []
    for metric, sense in METRICS.items():
        w = df.pivot_table(index=["patient_id", "organ"], columns="m", values=metric)
        block = [{"metrique": metric, "organe": o, **paired(w.xs(o, level="organ").SegResNet,
                                                            w.xs(o, level="organ")["U-Net"], sense)} for o in ORG]
        for b, q in zip(block, holm([b["p"] for b in block])):
            b["p_holm"] = q
        rows += block
    pr = pd.DataFrame(rows)
    pr.to_csv(OUT / "p4_comparaisons.csv", index=False)
    folds = df.groupby(["m", "fold", "organ"]).dsc.mean().unstack("organ")[ORG]
    folds.to_csv(OUT / "p4_dsc_par_pli.csv")
    empty = df.groupby(["m", "organ"]).empty_pred.sum().unstack()[ORG]
    empty.to_csv(OUT / "p4_predictions_vides.csv")

    # figures
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, metric, logy in [(axs[0], "dsc", False), (axs[1], "hd95_mm", True)]:
        data, pos, cols = [], [], []
        for i, o in enumerate(ORG):
            for j, (m, c) in enumerate([("U-Net", UNET), ("SegResNet", SEG)]):
                data.append(df[(df.m == m) & (df.organ == o)][metric].dropna().to_numpy())
                pos.append(i * 3 + j)
                cols.append(c)
        bp = ax.boxplot(data, positions=pos, widths=0.75, patch_artist=True, medianprops={"color": INK},
                        flierprops={"marker": ".", "ms": 2.5, "mec": MUTED, "alpha": 0.5})
        for b, c in zip(bp["boxes"], cols):
            b.set_facecolor(c)
            b.set_edgecolor(SURF)
        ax.set_xticks([i * 3 + 0.5 for i in range(6)], [ORGL[o] for o in ORG])
        ax.set_ylabel({"dsc": "DSC", "hd95_mm": "HD95 (mm, log)"}[metric])
        if logy:
            ax.set_yscale("log")
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (UNET, SEG)]
    fig.legend(hs, ["U-Net 18,8 M", "SegResNet"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.01, -0.03, "Cohorte D115 (115 patients, 5 plis de 23), grille native, graine 42.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"p4_fig1_boxplots.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, metric, xl in [(axs[0], "dsc", "SegResNet − U-Net : DSC"), (axs[1], "hd95_mm", "SegResNet − U-Net : HD95 (mm)")]:
        s = pr[pr.metrique == metric].set_index("organe").loc[ORG]
        y = np.arange(6)[::-1]
        ax.hlines(y, s.ic_bas, s.ic_haut, color=SEG, lw=2)
        ax.plot(s.med_diff, y, "o", color=SEG, ms=6, mec=SURF, mew=1.2)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(y, [ORGL[o] for o in ORG])
        ax.set_xlabel(xl + " (médiane appariée, IC 95 %)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"p4_fig2_differences.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(desc[["dsc", "hd95_mm", "sdsc_2mm"]].to_string())
    print(pr[pr.metrique.isin(["dsc", "hd95_mm", "sdsc_2mm"])][["metrique", "organe", "n", "med_diff", "ic_bas", "ic_haut",
                                                               "pct_seg_meilleur", "p_holm", "r_rb"]].round(4).to_string(index=False))
    print(folds.round(3))
    print("predictions vides :\n", empty)


if __name__ == "__main__":
    main()
