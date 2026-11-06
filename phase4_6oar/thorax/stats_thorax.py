# -*- coding: utf-8 -*-
"""Statistiques des metriques 6 OAR limitees a la region thoracique (coupes pulmonaires) vs etendue complete.

Lit phase4_6oar/thorax/metrics_thorax_long.csv. Memes conventions que phase4_6oar/analyse/phase4_stats.py :
difference appariee SegResNet - U-Net, Wilcoxon bilateral, Holm par metrique et par region sur les 6 OAR,
IC 95 % bootstrap (10 000 tirages, graine 20261106), r de rang biserial.
Ecrit : stats_thorax.csv (par region x OAR x metrique), couverture_thorax.csv, fig_thorax.(png|pdf).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, wilcoxon  # noqa: E402

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261106)
NBOOT = 10_000
ORG = ["poumon_g", "poumon_d", "coeur", "oesophage", "moelle", "aorte"]
ORGL = {"poumon_g": "Poumon G", "poumon_d": "Poumon D", "coeur": "Cœur", "oesophage": "Œsophage",
        "moelle": "Moelle", "aorte": "Aorte"}
METRICS = {"dsc": 1, "hd95_mm": -1, "assd_mm": -1, "sdsc_2mm": 1}
SEG, UNET = "#eb6834", "#1baf7a"

df = pd.read_csv(HERE / "metrics_thorax_long.csv")
df["m"] = df.variant.map({"segresnet_6oar": "SegResNet", "unet_6oar": "U-Net"})
assert df.groupby("m").patient_id.nunique().eq(115).all(), df.groupby("m").patient_id.nunique()


def paired(x, y, sense):
    d = (x - y).dropna().to_numpy()
    nz = d[d != 0]
    rk = rankdata(np.abs(nz))
    r = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum() if len(nz) else np.nan
    p = wilcoxon(d, zero_method="wilcox").pvalue if len(nz) else 1.0
    bm = np.median(d[RNG.integers(0, len(d), size=(NBOOT, len(d)))], axis=1)
    return {"n": len(d), "med_diff": np.median(d), "ic_bas": np.percentile(bm, 2.5),
            "ic_haut": np.percentile(bm, 97.5), "pct_seg_meilleur": 100 * (d * sense > 0).mean(), "p": p, "r_rb": r}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    adj[o] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    return adj


rows = []
for region in ("complet", "thorax"):
    for met, sense in METRICS.items():
        block = []
        for org in ORG:
            w = df[(df.region == region) & (df.organ == org)].pivot_table(
                index="patient_id", columns="m", values=met)
            r = {"region": region, "metrique": met, "organe": org}
            for mod in ("U-Net", "SegResNet"):
                v = w[mod].dropna()
                r.update({f"{mod}_med": v.median(), f"{mod}_q1": v.quantile(.25), f"{mod}_q3": v.quantile(.75)})
            r.update(paired(w["SegResNet"], w["U-Net"], sense))
            block.append(r)
        ph = holm([b["p"] for b in block])
        for b, q in zip(block, ph):
            b["p_holm"] = q
        rows += block
S = pd.DataFrame(rows)
S.round(5).to_csv(HERE / "stats_thorax.csv", index=False)

cov = (df[df.region == "thorax"].groupby(["organ"])
       .agg(n_coupes_poumon_med=("n_coupes_poumon", "median"),
            frac_gt_thorax_med=("frac_gt_dans_thorax", "median"),
            frac_gt_thorax_min=("frac_gt_dans_thorax", "min"),
            frac_gt_thorax_q1=("frac_gt_dans_thorax", lambda s: s.quantile(.25)),
            pct_patients_gt_hors_thorax=("frac_gt_dans_thorax", lambda s: 100 * (s < 0.999).mean()))
       .reindex(ORG))
cov.round(4).to_csv(HERE / "couverture_thorax.csv")

# figure : DSC et HD95, complet vs thorax, points bruts (pas de lissage)
fig, axes = plt.subplots(2, 6, figsize=(15, 6), facecolor="#fcfcfb")
for i, met in enumerate(["dsc", "hd95_mm"]):
    for j, org in enumerate(ORG):
        ax = axes[i, j]
        for k, (region, mod) in enumerate([(r, m) for r in ("complet", "thorax") for m in ("U-Net", "SegResNet")]):
            v = df[(df.region == region) & (df.organ == org) & (df.m == mod)][met].dropna().to_numpy()
            x = k + (0.4 if region == "thorax" else 0)
            ax.scatter(x + RNG.uniform(-0.12, 0.12, len(v)), v, s=4, alpha=0.35, linewidths=0,
                       color=SEG if mod == "SegResNet" else UNET)
            ax.boxplot(v, positions=[x], widths=0.4, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="none", lw=0.9), medianprops=dict(color="#0b0b0b", lw=1.3))
        ax.set_xticks([0.5, 2.9], ["complet", "thorax"], fontsize=8)
        if met == "hd95_mm":
            ax.set_yscale("log")
        if i == 0:
            ax.set_title(ORGL[org], fontsize=10)
        if j == 0:
            ax.set_ylabel({"dsc": "DSC", "hd95_mm": "HD95 (mm, log)"}[met])
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
for mod, col in (("U-Net", UNET), ("SegResNet", SEG)):
    axes[0, -1].scatter([], [], color=col, label=mod)
axes[0, -1].legend(frameon=False, fontsize=8, loc="lower right")
fig.text(0.01, 0.005, "Cohorte D115 (115 patients, 5 plis), grille native ; thorax = coupes contenant du poumon "
         "dans la référence.", fontsize=7, color="#8a8984")
fig.tight_layout(rect=(0, 0.02, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_thorax.{ext}", dpi=300)
print("ecrit dans", HERE)
