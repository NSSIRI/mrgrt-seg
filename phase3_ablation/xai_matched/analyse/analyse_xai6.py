# -*- coding: utf-8 -*-
"""Analyse du XAI a couches appariees, phase 4 (6 OAR, cohorte D115, 115 patients, 5 plis) : U-Net vs SegResNet.

Lit results/phase3_ablation/xai_matched/*/abl18m-xai6-*/xai_matched/*.csv (lecture seule).
Memes tests que analyse_xai_matched.py : Wilcoxon apparie (SegResNet - U-Net), Holm par metrique sur
4 niveaux x 6 organes, IC 95 % bootstrap (10 000 tirages, graine 20261106), r rang-biserial.
Ecrit : xai6_stats_apparies.csv, xai6_couverture.csv, xai6_adebayo_resume.csv, fig_xai6.(png|pdf).
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
RES = HERE.parents[2] / "results" / "phase3_ablation" / "xai_matched"
LEVELS = ["enc_profond", "dec_demi", "dec_pre_tete", "sortie"]
ORGANS = ["poumon_g", "poumon_d", "coeur", "oesophage", "moelle", "aorte"]
METRICS = ["in_organ_ratio", "pointing_accuracy", "activation_dice", "spatial_entropy", "centroid_distance_mm"]
RNG = np.random.default_rng(20261106)

L = pd.concat(pd.read_csv(f) for f in glob.glob(str(RES / "*/abl18m-xai6-*/xai_matched/xai_matched_long.csv")))
A = pd.concat(pd.read_csv(f) for f in glob.glob(str(RES / "*/abl18m-xai6-*/xai_matched/adebayo_matched_long.csv")))
L["model"] = L.variant.map({"unet6": "U-Net", "seg6": "SegResNet"})
A["model"] = A.variant.map({"unet6": "U-Net", "seg6": "SegResNet"})
assert L.groupby("model").patient_id.nunique().eq(115).all()
assert not L.duplicated(["variant", "fold", "patient_id", "level", "organ"]).any()


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


cov = (L[L.level == "sortie"].groupby(["model", "organ"])
       .agg(n=("patient_id", "size"), frac_crop_mediane=("frac_gt_in_crop", "median"),
            frac_crop_q1=("frac_gt_in_crop", lambda s: s.quantile(.25)),
            organe_absent_crop=("n_gt_crop", lambda s: int((s == 0).sum())),
            pred_vide=("n_pred_crop", lambda s: int((s == 0).sum())),
            metrique_nan=("in_organ_ratio", lambda s: int(s.isna().sum())))
       .reset_index())
cov.to_csv(HERE / "xai6_couverture.csv", index=False)

rows = []
W = L.pivot_table(index=["patient_id", "level", "organ"], columns="model", values=METRICS + ["organ_baseline"])
for met in METRICS:
    for lv in LEVELS:
        for org in ORGANS:
            d = W.xs((lv, org), level=("level", "organ"))[met].dropna()
            u, s = d["U-Net"].to_numpy(), d["SegResNet"].to_numpy()
            diff = s - u
            r = {"metrique": met, "niveau": lv, "organe": org, "n_paires": len(d),
                 "UNet_mediane": np.median(u), "UNet_q1": np.percentile(u, 25), "UNet_q3": np.percentile(u, 75),
                 "SegResNet_mediane": np.median(s), "SegResNet_q1": np.percentile(s, 25),
                 "SegResNet_q3": np.percentile(s, 75), "diff_mediane": np.median(diff)}
            bs = np.median(diff[RNG.integers(0, len(diff), (10000, len(diff)))], axis=1)
            r["ic95_bas"], r["ic95_haut"] = np.percentile(bs, [2.5, 97.5])
            nz = diff[diff != 0]
            if len(nz) >= 10:
                r["p"] = wilcoxon(s, u, zero_method="wilcox").pvalue
                rk = pd.Series(np.abs(nz)).rank().to_numpy()
                r["r_rang_biserial"] = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
            else:
                r["p"], r["r_rang_biserial"] = np.nan, np.nan
            if met == "in_organ_ratio":
                b = W.xs((lv, org), level=("level", "organ"))["organ_baseline"].loc[d.index]
                r["UNet_gain_vs_uniforme"] = np.median(u / b["U-Net"].to_numpy())
                r["SegResNet_gain_vs_uniforme"] = np.median(s / b["SegResNet"].to_numpy())
            rows.append(r)
S = pd.DataFrame(rows)
S["p_holm"] = np.nan
for met in METRICS:
    m = S.metrique == met
    S.loc[m, "p_holm"] = holm(S.loc[m, "p"])
S.round(5).to_csv(HERE / "xai6_stats_apparies.csv", index=False)

ad = []
for (mod, lv, fold), g in A.groupby(["model", "level", "fold"]):
    g = g.sort_values("step")
    ad.append({"model": mod, "niveau": lv, "pli": fold, "n_couches": int(g.step.max()),
               "ssim_etape1": g.ssim_vs_pristine.iloc[1], "spearman_etape1": g.spearman_vs_pristine.iloc[1],
               "ssim_finale": g.ssim_vs_pristine.iloc[-1], "ssim_max_apres_0": g.ssim_vs_pristine.iloc[1:].max()})
pd.DataFrame(ad).round(4).to_csv(HERE / "xai6_adebayo_resume.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

COL = {"U-Net": "#1baf7a", "SegResNet": "#eb6834"}
fig, axes = plt.subplots(1, 6, figsize=(20, 4.3), sharey=True, facecolor="#fcfcfb")
for ax, org in zip(axes, ORGANS):
    for i, lv in enumerate(LEVELS):
        for j, mod in enumerate(["U-Net", "SegResNet"]):
            v = L[(L.model == mod) & (L.level == lv) & (L.organ == org)].in_organ_ratio.dropna().to_numpy()
            x = i + (j - 0.5) * 0.36
            ax.scatter(x + RNG.uniform(-0.08, 0.08, len(v)), v, s=5, color=COL[mod], alpha=0.4, linewidths=0)
            ax.boxplot(v, positions=[x], widths=0.28, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="none", edgecolor="#0b0b0b", lw=0.9),
                       medianprops=dict(color="#0b0b0b", lw=1.4), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    ax.set_xticks(range(4), ["enc.\nprofond", "déc.\n1/2", "déc.\npré-tête", "sortie"], fontsize=8)
    ax.set_title(org.replace("_", " "), fontsize=10)
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)
axes[0].set_ylabel("in-organ ratio (Seg-Grad-CAM)")
handles = [axes[-1].scatter([], [], color=COL[mod], label=mod) for mod in COL]
fig.legend(handles=handles, frameon=False, fontsize=9, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.text(0.01, 0.005, "Phase 4 : cohorte D115 (115 patients, 5 plis), recadrage 128x128x64 centré sur les 4 OAR "
         "thoraciques ; moelle et aorte évaluées dans ce champ.", fontsize=7, color="#8a8984")
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_xai6.{ext}", dpi=300)
print("ecrit dans", HERE)
