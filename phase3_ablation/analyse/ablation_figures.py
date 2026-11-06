"""Figures de l'ablation 18,8 M (PNG 300 dpi + PDF) a partir des CSV de ablation_stats.py / ablation_masks.py.

Couleurs : E3 aqua #1baf7a, E2 orange #eb6834 (palette du projet), variantes violet #6250d6 (emplacement 7) ;
3 teintes validees toutes paires (scripts/validate_palette.js : PASS) ; les variantes sont identifiees par leur
etiquette, jamais par la couleur seule.

    .venv\\Scripts\\python.exe phase3_ablation/analyse/ablation_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

A = Path(__file__).resolve().parent
FIG = A.parent / "figures"
FIG.mkdir(exist_ok=True)
ORG = ["poumon_g", "poumon_d", "coeur", "oesophage"]
ORGL = {"poumon_g": "Poumon G", "poumon_d": "Poumon D", "coeur": "Cœur", "oesophage": "Œsophage"}
E3C, E2C, VC = "#1baf7a", "#eb6834", "#6250d6"
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
LAB = {"A1": "A1 U-Net + résiduel", "A2": "A2 U-Net + GroupNorm", "A3": "A3 SegResNet + InstanceNorm"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.family": "DejaVu Sans"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  figure", name)


def fig_forest():
    q2 = pd.read_csv(A / "stats_q2_variantes.csv")
    q1 = pd.read_csv(A / "stats_q1_graines.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, metric, xl in [(axs[0], "dsc", "Différence appariée de DSC vs E3 (médiane, IC 95 %)"),
                           (axs[1], "hd95_mm", "Différence appariée de HD95 vs E3, mm (médiane, IC 95 %)")]:
        y, ticks, labs = 0, [], []
        for o in ORG:
            rows = [("E2 SegResNet", E2C, q1[(q1.metrique == metric) & (q1.organe == o) &
                                              (q1.comparaison == "E2-E3 graine 42")].iloc[0])]
            for v in ["A3", "A1", "A2"]:
                rows.append((LAB[v], VC, q2[(q2.metrique == metric) & (q2.organe == o) & (q2.variante == v) &
                                            (q2.reference == "E3")].iloc[0]))
            thr = q2[(q2.metrique == metric) & (q2.organe == o)].seuil_graine.iloc[0]
            ax.axvspan(-thr, thr, ymin=0, ymax=1, color=GRID, alpha=0.0)
            ax.fill_between([-thr, thr], y + 0.5, y - len(rows) + 0.5, color=GRID, alpha=0.8, lw=0, zorder=0)
            for lab, col, r in rows:
                ax.plot([r.ic_bas, r.ic_haut], [y, y], color=col, lw=2, solid_capstyle="round")
                ax.plot(r.med_diff, y, "o", color=col, ms=6, mec=SURF, mew=1.2)
                ticks.append(y)
                labs.append(f"{ORGL[o]} — {lab}")
                y -= 1
            y -= 0.7
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(ticks, labs, fontsize=7.5)
        ax.set_xlabel(xl)
    axs[1].set_yticklabels([])
    fig.text(0.01, -0.03, "Bande grise : variabilité due à la graine d'entraînement (plus grande borne |IC 95 %| de "
             "l'effet graine de E3 et E2). 187 patients, grille native.", fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "abl_fig1_differences_vs_E3")


def fig_gap():
    g = pd.read_csv(A / "stats_part_ecart.csv")
    g = g[g.metrique == "dsc"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs = np.arange(len(ORG))
    for k, v in enumerate(["A2", "A1", "A3"]):
        s = g[g.variante == v].set_index("organe").loc[ORG]
        x = xs + (k - 1) * 0.22
        ax.errorbar(x, s.part_comblee, yerr=[s.part_comblee - s.ic_bas, s.ic_haut - s.part_comblee], fmt="o",
                    color=VC, ms=6, mec=SURF, mew=1.2, elinewidth=2, capsize=0)
        for xi, yi in zip(x, s.part_comblee):
            ax.annotate(v, (xi, yi), textcoords="offset points", xytext=(7, -3), fontsize=7.5, color=INK2)
    ax.axhline(0, color=E3C, lw=1.5)
    ax.axhline(1, color=E2C, lw=1.5)
    ax.text(len(ORG) - 0.45, 0.04, "niveau E3", color=INK2, fontsize=7.5)
    ax.text(len(ORG) - 0.45, 1.04, "niveau E2", color=INK2, fontsize=7.5)
    ax.set_xticks(xs, [ORGL[o] for o in ORG])
    ax.set_ylabel("Part de l'écart E3 → E2 comblée (DSC)")
    fig.text(0.01, -0.04, "Part = méd(variante − E3) / méd(E2 − E3), appariée par patient ; IC 95 % bootstrap "
             "(10 000 tirages). A1, A2 : U-Net modifié ; A3 : SegResNet modifié.", fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "abl_fig2_part_ecart_comblee")


def fig_boxes():
    import sys
    sys.path.insert(0, str(A))
    from ablation_stats import load
    df = load()
    order = [("E3_s42", "E3", E3C), ("A2", "A2", VC), ("A1", "A1", VC), ("E2_s42", "E2", E2C), ("A3", "A3", VC)]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.4), sharey=False)
    for ax, o in zip(axs, ORG):
        data = [df[(df.m == m) & (df.organ == o)].dsc.dropna().to_numpy() for m, _, _ in order]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, medianprops={"color": INK, "lw": 1.2},
                        flierprops={"marker": ".", "ms": 2.5, "mec": MUTED, "alpha": 0.5},
                        whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        for b, (_, _, c) in zip(bp["boxes"], order):
            b.set_facecolor(c)
            b.set_edgecolor(SURF)
        ax.set_xticks(range(1, 6), [lab for _, lab, _ in order])
        ax.set_title(ORGL[o], fontsize=10, color=INK)
    axs[0].set_ylabel("DSC (187 patients, grille native)")
    fig.text(0.01, -0.05, "E3 U-Net 18,8 M ; A2 = E3 + GroupNorm ; A1 = U-Net résiduel ; E2 SegResNet ; "
             "A3 = E2 avec InstanceNorm. Graine 42.", fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "abl_fig3_boxplots_dsc")


def fig_seeds():
    import sys
    sys.path.insert(0, str(A))
    from ablation_stats import load
    df = load()
    fm = df[df.m.str.startswith(("E2", "E3"))].groupby(["m", "fold", "organ"]).dsc.mean().reset_index()
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, o in zip(axs, ORG):
        for mod, col in [("E3", E3C), ("E2", E2C)]:
            a = fm[(fm.m == f"{mod}_s42") & (fm.organ == o)].set_index("fold").dsc
            b = fm[(fm.m == f"{mod}_s123") & (fm.organ == o)].set_index("fold").dsc
            ax.scatter(a, b.reindex(a.index), color=col, s=30, edgecolor=SURF, lw=1, label=mod, zorder=3)
        lo = fm[fm.organ == o].dsc.min() - 0.02
        hi = fm[fm.organ == o].dsc.max() + 0.02
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(ORGL[o], fontsize=10, color=INK)
        ax.set_xlabel("DSC moyen du pli, graine 42")
    axs[0].set_ylabel("graine 123")
    axs[0].legend(loc="upper left", fontsize=8)
    fig.text(0.01, -0.06, "Chaque point = un pli (moyenne des patients). Diagonale : reproductibilité parfaite.",
             fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "abl_fig4_reproductibilite_graines")


def fig_robust():
    p = A / "ablation_masks.csv"
    if not p.exists():
        print("  (ablation_masks.csv absent : figure de robustesse sautee)")
        return
    r = pd.read_csv(p)
    order = [("E3_s42", "E3", E3C), ("A2", "A2", VC), ("A1", "A1", VC), ("E2_s42", "E2", E2C), ("A3", "A3", VC)]
    frag = r.groupby(["model", "organ"]).apply(lambda g: 100 * (g.frac_out_lcc > 0.01).mean(), include_groups=False)
    lat = r.drop_duplicates(["patient_id", "model"]).assign(sw=lambda t: t[["swap_G", "swap_D"]].max(axis=1) > 0.05)
    lat = lat.groupby("model").sw.mean() * 100
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [3, 1]})
    ax = axs[0]
    for k, (m, lab, c) in enumerate(order):
        vals = [frag.get((m, o), np.nan) for o in ORG]
        ax.bar(np.arange(4) + (k - 2) * 0.16, vals, width=0.15, color=c)
        for i, v in enumerate(vals):
            ax.text(i + (k - 2) * 0.16, v + 1, lab, ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(range(4), [ORGL[o] for o in ORG])
    ax.set_ylabel("% patients avec > 1 % du volume\nhors de la plus grande composante")
    ax = axs[1]
    ax.bar(range(5), [lat.get(m, np.nan) for m, _, _ in order], color=[c for _, _, c in order])
    ax.set_xticks(range(5), [lab for _, lab, _ in order])
    ax.set_ylabel("% patients, confusion G/D > 5 %")
    fig.tight_layout()
    save(fig, "abl_fig5_robustesse")


if __name__ == "__main__":
    fig_forest()
    fig_gap()
    fig_boxes()
    fig_seeds()
    fig_robust()
