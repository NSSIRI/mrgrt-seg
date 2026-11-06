# -*- coding: utf-8 -*-
"""Analyse du XAI a couches appariees (E3 U-Net vs E2 SegResNet, 187 patients, 5 plis).

Lit results/phase3_ablation/xai_matched/*/*/xai_matched/{xai_matched_long,adebayo_matched_long}.csv (lecture seule).
Ecrit dans phase3_ablation/xai_matched/analyse/ : xai_stats_apparies.csv, xai_couverture.csv,
xai_adebayo_resume.csv, xai_ancien_vs_nouveau.csv, fig_xai_matched.(png|pdf).
Tests : Wilcoxon apparie par patient (E2 - E3), Holm par metrique sur 4 niveaux x 4 organes,
IC 95 % bootstrap de la mediane des differences (10 000 tirages, graine 20261106), r rang-biserial.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = ROOT / "results" / "phase3_ablation" / "xai_matched"
LEVELS = ["enc_profond", "dec_demi", "dec_pre_tete", "sortie"]
ORGANS = ["poumon_g", "poumon_d", "coeur", "oesophage"]
METRICS = ["in_organ_ratio", "pointing_accuracy", "activation_dice", "spatial_entropy", "centroid_distance_mm"]
RNG = np.random.default_rng(20261106)

L = pd.concat(pd.read_csv(f) for f in glob.glob(str(RES / "*/*/xai_matched/xai_matched_long.csv")))
A = pd.concat(pd.read_csv(f) for f in glob.glob(str(RES / "*/*/xai_matched/adebayo_matched_long.csv")))
L["model"] = L.variant.map({"unet_opt": "E3", "segresnet": "E2"})
A["model"] = A.variant.map({"unet_opt": "E3", "segresnet": "E2"})
assert L.groupby("model").patient_id.nunique().eq(187).all()


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    tmp = np.empty(m)
    tmp[idx] = np.minimum(adj, 1)
    out[ok] = tmp
    return out


# --- couverture : organes hors recadrage, absence de prediction
cov = (L.groupby(["model", "level", "organ"])
       .agg(n=("patient_id", "size"), organe_absent_crop=("n_gt_crop", lambda s: int((s == 0).sum())),
            frac_crop_mediane=("frac_gt_in_crop", "median"),
            pred_vide=("n_pred_crop", lambda s: int((s == 0).sum())),
            metrique_nan=("in_organ_ratio", lambda s: int(s.isna().sum())))
       .reset_index())
cov[cov.level == "sortie"].drop(columns="level").to_csv(HERE / "xai_couverture.csv", index=False)

# --- statistiques appariees E2 - E3
rows = []
W = L.pivot_table(index=["patient_id", "level", "organ"], columns="model", values=METRICS + ["organ_baseline"])
for met in METRICS:
    for lv in LEVELS:
        for org in ORGANS:
            d = W.xs((lv, org), level=("level", "organ"))[met].dropna()
            e3, e2 = d["E3"].to_numpy(), d["E2"].to_numpy()
            diff = e2 - e3
            r = {"metrique": met, "niveau": lv, "organe": org, "n_paires": len(d),
                 "E3_mediane": np.median(e3), "E3_q1": np.percentile(e3, 25), "E3_q3": np.percentile(e3, 75),
                 "E2_mediane": np.median(e2), "E2_q1": np.percentile(e2, 25), "E2_q3": np.percentile(e2, 75),
                 "E3_moyenne": e3.mean(), "E2_moyenne": e2.mean(), "diff_mediane": np.median(diff)}
            bs = np.median(diff[RNG.integers(0, len(diff), (10000, len(diff)))], axis=1)
            r["ic95_bas"], r["ic95_haut"] = np.percentile(bs, [2.5, 97.5])
            nz = diff[diff != 0]
            if len(nz) >= 10:
                r["p"] = wilcoxon(e2, e3, zero_method="wilcox").pvalue
                rk = pd.Series(np.abs(nz)).rank().to_numpy()
                r["r_rang_biserial"] = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
            else:
                r["p"], r["r_rang_biserial"] = np.nan, np.nan
            if met == "in_organ_ratio":   # gain sur une carte uniforme (organ_baseline = volume organe / volume crop)
                b = W.xs((lv, org), level=("level", "organ"))["organ_baseline"].loc[d.index]
                r["E3_gain_vs_uniforme"] = np.median(e3 / b["E3"].to_numpy())
                r["E2_gain_vs_uniforme"] = np.median(e2 / b["E2"].to_numpy())
            rows.append(r)
S = pd.DataFrame(rows)
S["p_holm"] = np.nan
for met in METRICS:
    m = S.metrique == met
    S.loc[m, "p_holm"] = holm(S.loc[m, "p"])
S.round(5).to_csv(HERE / "xai_stats_apparies.csv", index=False)

# --- Adebayo : SSIM / Spearman apres randomisation de la 1re couche (sortie) et de toutes les couches
ad = []
for (mod, lv, fold), g in A.groupby(["model", "level", "fold"]):
    g = g.sort_values("step")
    ad.append({"model": mod, "niveau": lv, "pli": fold, "n_couches": int(g.step.max()),
               "ssim_etape1": g.ssim_vs_pristine.iloc[1], "spearman_etape1": g.spearman_vs_pristine.iloc[1],
               "ssim_finale": g.ssim_vs_pristine.iloc[-1], "spearman_finale": g.spearman_vs_pristine.iloc[-1],
               "ssim_max_apres_0": g.ssim_vs_pristine.iloc[1:].max()})
AD = pd.DataFrame(ad)
AD.round(4).to_csv(HERE / "xai_adebayo_resume.csv", index=False)

# --- ancien XAI (target_layer="auto", recadrage centre) vs niveaux equivalents du nouveau
old = {"E3": glob.glob(str(ROOT / "results/UNET VS SEGRESNET/xai_unet_opt/unet_fold*_xai_per_patient.csv")),
       "E2": glob.glob(str(ROOT / "results/xai_final/segresnet_fold*_xai_per_patient.csv"))}
eq = {"E3": "enc_profond", "E2": "sortie"}   # couches visees par l'ancien code (verifiees par construction du modele)
cmp_rows = []
for mod, files in old.items():
    o = pd.concat(pd.read_csv(f) for f in files)
    for org in ORGANS:
        for met in ("in_organ_ratio", "pointing_accuracy"):
            n = L[(L.model == mod) & (L.level == eq[mod]) & (L.organ == org)][met]
            cmp_rows.append({"model": mod, "organe": org, "metrique": met, "ancien_mediane": o[f"{met}_{org}"].median(),
                             "ancien_nan": int(o[f"{met}_{org}"].isna().sum()), "nouveau_niveau": eq[mod],
                             "nouveau_mediane": n.median(), "nouveau_nan": int(n.isna().sum())})
pd.DataFrame(cmp_rows).round(4).to_csv(HERE / "xai_ancien_vs_nouveau.csv", index=False)

# --- figure : in-organ ratio par niveau et organe, points bruts (pas de lissage)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

COL = {"E3": "#1baf7a", "E2": "#eb6834"}
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True, facecolor="#fcfcfb")
for ax, org in zip(axes, ORGANS):
    for i, lv in enumerate(LEVELS):
        for j, mod in enumerate(["E3", "E2"]):
            v = L[(L.model == mod) & (L.level == lv) & (L.organ == org)].in_organ_ratio.dropna().to_numpy()
            x = i + (j - 0.5) * 0.36
            ax.scatter(x + RNG.uniform(-0.08, 0.08, len(v)), v, s=4, color=COL[mod], alpha=0.35, linewidths=0)
            ax.boxplot(v, positions=[x], widths=0.28, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="none", edgecolor="#0b0b0b", lw=0.9),
                       medianprops=dict(color="#0b0b0b", lw=1.4), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    ax.set_xticks(range(4), ["enc.\nprofond", "déc.\n1/2", "déc.\npré-tête", "sortie"], fontsize=8)
    ax.set_title(org.replace("_", " "), fontsize=10)
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)
axes[0].set_ylabel("in-organ ratio (Seg-Grad-CAM)")
for mod in COL:
    axes[-1].scatter([], [], color=COL[mod], label=f"{mod} ({'U-Net' if mod == 'E3' else 'SegResNet'})")
axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_xai_matched.{ext}", dpi=300)
print("ecrit dans", HERE)
