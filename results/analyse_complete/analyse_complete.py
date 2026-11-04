"""Analyse complete U-Net / SegResNet (E1, E2, E3 ; E4 nnU-Net en appoint) a partir des fichiers EXISTANTS.

Lecture seule : aucun fichier de results/ ou data/ n'est modifie. Sorties dans results/analyse_complete/ :
  tableaux/*.csv   inventaire, descriptifs, comparaisons appariees, plis, courbes, temps, controles
  figures/*.png|pdf (300 dpi)

    .venv\\Scripts\\python.exe results/analyse_complete/analyse_complete.py

Sources (definitions verifiees dans phase2_kaggle_eval/phase2_eval_kaggle.py) :
  results/phase2_eval/analyse/combined_long.csv      E1-E3, 187 patients, grilles native et reechantillonnee
  results/phase2_eval/analyse/nnunet_e4_metrics_long.csv  E4, 76 patients (plis 0-1), grille native
  results/phase2_eval/analyse/histories_all.csv      train_loss, val_dsc, lr par epoque
  results/phase3_ablation/_journaux/*/*.log          journaux Kaggle (temps par epoque)
  results/phase2_eval/*/phase2_eval/predictions.zip  masques predits (grille native)
  data/PATIENT_s*/label.nii.gz                       en-tetes (nombre de voxels, pour la specificite)
"""
from __future__ import annotations

import io
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
from scipy.stats import rankdata, wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "results/phase2_eval"
OUT = Path(__file__).resolve().parent
TAB, FIG = OUT / "tableaux", OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261104)
NBOOT = 10_000

MODELS = {"E1_UNet_base": "E1 U-Net 7,9 M", "E3_UNet_opt": "E3 U-Net 18,8 M", "E2_SegResNet": "E2 SegResNet 18,8 M"}
ORDER = ["E1_UNet_base", "E3_UNet_opt", "E2_SegResNet"]
ORGANS = {"poumon_g": "Poumon G", "poumon_d": "Poumon D", "coeur": "Cœur", "oesophage": "Œsophage"}
METRICS = {  # colonne -> (libelle, sens : +1 plus grand = mieux, -1 plus petit = mieux)
    "dsc": ("DSC", 1), "iou": ("IoU", 1), "hd95_mm": ("HD95 (mm)", -1), "assd_mm": ("ASSD (mm)", -1),
    "recall": ("Sensibilité", 1), "specificity": ("Spécificité", 1), "precision": ("Précision", 1),
    "sdsc_2mm": ("Surface-DSC 2 mm", 1), "abs_vol_err": ("|ΔV|/V", -1),
}
COMPARISONS = [("E2_SegResNet", "E3_UNet_opt"), ("E3_UNet_opt", "E1_UNet_base"), ("E2_SegResNet", "E1_UNet_base")]

# palette validee du projet (scripts/validate_palette.js : PASS ; contraste < 3:1 pour E3/E4 -> legendes + tableaux)
COL = {"E1_UNet_base": "#2a78d6", "E2_SegResNet": "#eb6834", "E3_UNet_opt": "#1baf7a", "E4_nnUNet": "#eda100"}
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.titlecolor": INK,
    "legend.frameon": False, "font.family": "DejaVu Sans",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  figure", name)


# =========================================================== 1. inventaire
def inventory():
    files = [P2 / "analyse/combined_long.csv", P2 / "analyse/nnunet_e4_metrics_long.csv",
             P2 / "analyse/histories_all.csv"]
    files += sorted(P2.glob("*/phase2_eval/metrics_long.csv")) + sorted(P2.glob("*/phase2_eval/runs_summary.csv"))
    files += sorted(P2.glob("*/phase2_eval/predictions.zip")) + sorted(P2.glob("*/phase2_eval/splits.json"))
    files += sorted((ROOT / "results/phase3_ablation/_journaux").glob("*/*.log"))
    files += [ROOT / p for p in ("configs/default.yaml", "configs/unet_fair.yaml", "src/models/factory.py",
                                 "src/train/trainer.py", "src/data/dataset.py", "src/data/splits.py",
                                 "src/train/losses.py", "scripts/train.py", "phase2_kaggle_eval/phase2_eval_kaggle.py")]
    rows = [{"fichier": str(f.relative_to(ROOT)).replace("\\", "/"), "octets": f.stat().st_size} for f in files if f.exists()]
    pd.DataFrame(rows).to_csv(TAB / "00_inventaire.csv", index=False)
    return len(rows)


# =========================================================== 2. donnees
def load_metrics():
    c = pd.read_csv(P2 / "analyse/combined_long.csv")
    c = c[c.space == "native"].copy()
    # specificite : TN = N - TP - FP - FN, N = nombre de voxels de la grille native du label (en-tete NIfTI)
    nvox = {}
    for pid in c.patient_id.unique():
        h = nib.load(str(ROOT / "data" / pid / "label.nii.gz")).header
        nvox[pid] = int(np.prod(h.get_data_shape()[:3]))
    vox_ml = c.sp_x * c.sp_y * c.sp_z / 1000.0
    ngt = np.rint(c.vol_gt_ml / vox_ml)
    npred = np.rint(c.vol_pred_ml / vox_ml)
    tp = np.rint(c.dsc * (ngt + npred) / 2)
    fp = npred - tp
    n = c.patient_id.map(nvox)
    tn = n - ngt - fp
    c["specificity"] = tn / (tn + fp)
    c["abs_vol_err"] = c.vol_rel_err.abs()
    # controle : le TP reconstruit redonne la precision et la sensibilite du CSV
    chk = pd.DataFrame({"precision": (tp / npred - c.precision).abs(), "recall": (tp / ngt - c.recall).abs()})
    e4 = pd.read_csv(P2 / "analyse/nnunet_e4_metrics_long.csv")
    e4 = e4[e4.space == "native"].copy()
    e4["model"] = "E4_nnUNet"
    e4["abs_vol_err"] = e4.vol_rel_err.abs()
    return c, e4, float(np.nanmax(chk.values))


# =========================================================== 3. controles (fuite, coherence)
def controls(c):
    rows = []
    for src in sorted(P2.glob("*/phase2_eval/splits.json")):
        folds = json.loads(src.read_text())
        val = [p for f in folds for p in f["val"]]
        disj = all(not (set(f["train"]) & set(f["val"])) for f in folds)
        rows.append({"controle": f"plis {src.parent.parent.name}", "plis_disjoints_train_val": disj,
                     "val_uniques": len(set(val)), "val_total": len(val), "tailles_val": [len(f["val"]) for f in folds]})
    ref = json.loads((P2 / "abdelhalimnssiri_b/phase2_eval/splits.json").read_text())
    pf = {p: f["fold"] for f in ref for p in f["val"]}
    for m in ORDER:
        sub = c[c.model == m].drop_duplicates("patient_id")
        rows.append({"controle": f"{m} : pli de chaque patient = splits.json",
                     "ok": bool((sub.patient_id.map(pf) == sub.fold).all()), "patients": len(sub)})
    pd.DataFrame(rows).to_csv(TAB / "01_controles_plis_fuite.csv", index=False)
    return rows


# =========================================================== 4. statistiques
def boot_median_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan
    idx = RNG.integers(0, len(x), size=(NBOOT, len(x)))
    med = np.median(x[idx], axis=1)
    return tuple(np.percentile(med, [2.5, 97.5]))


def descriptives(df, models):
    rows = []
    for m in models:
        for o in ORGANS:
            s = df[(df.model == m) & (df.organ == o)]
            for k, (lab, _) in METRICS.items():
                if k not in s:
                    continue
                x = s[k].to_numpy(float)
                v = x[~np.isnan(x)]
                if len(v) == 0:      # metrique non disponible pour ce modele (ex. specificite de E4)
                    continue
                lo, hi = boot_median_ci(v)
                rows.append({"modele": m, "organe": o, "metrique": k, "n": len(v), "n_nan": int(np.isnan(x).sum()),
                             "moyenne": v.mean(), "ecart_type": v.std(ddof=1), "mediane": np.median(v),
                             "q1": np.percentile(v, 25), "q3": np.percentile(v, 75),
                             "ic95_mediane_bas": lo, "ic95_mediane_haut": hi})
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired(df, comps):
    rows = []
    for k, (lab, sense) in METRICS.items():
        block = []
        for a, b in comps:
            for o in ORGANS:
                A = df[(df.model == a) & (df.organ == o)].set_index("patient_id")[k]
                B = df[(df.model == b) & (df.organ == o)].set_index("patient_id")[k]
                d = (A - B.reindex(A.index)).dropna()
                if len(d) == 0:
                    continue
                n_excl = int(len(A) - len(d))
                dd = d.to_numpy()
                nz = dd[dd != 0]
                rk = rankdata(np.abs(nz))
                r_rb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum() if len(nz) else np.nan
                p = wilcoxon(dd, zero_method="wilcox", alternative="two-sided").pvalue if len(nz) > 0 else 1.0
                idx = RNG.integers(0, len(dd), size=(NBOOT, len(dd)))
                bm = np.median(dd[idx], axis=1)
                better = (dd * sense > 0).mean() * 100
                block.append({"metrique": k, "comparaison": f"{a} - {b}", "organe": o, "n_paires": len(dd),
                              "n_exclus": n_excl, "mediane_diff": np.median(dd),
                              "ic95_bas": np.percentile(bm, 2.5), "ic95_haut": np.percentile(bm, 97.5),
                              "moyenne_diff": dd.mean(), "pct_A_meilleur": better, "pct_egalites": (dd == 0).mean() * 100,
                              "p_wilcoxon": p, "r_rang_biserial": r_rb})
        ph = holm([r["p_wilcoxon"] for r in block])
        for r, q in zip(block, ph):
            r["p_holm"] = q
        rows += block
    return pd.DataFrame(rows)


# =========================================================== 5. courbes d'apprentissage
def curves():
    h = pd.read_csv(P2 / "analyse/histories_all.csv")
    rows = []
    for (m, f), g in h.groupby(["model", "fold"]):
        g = g.sort_values("epoch")
        v, tl, ep = g.val_dsc.to_numpy(), g.train_loss.to_numpy(), g.epoch.to_numpy()
        ib = int(np.argmax(v))
        last = slice(max(0, len(v) - 30), len(v))
        slope_v = np.polyfit(ep[last], v[last], 1)[0] * 30    # variation sur les 30 dernieres epoques
        slope_t = np.polyfit(ep[last], tl[last], 1)[0] * 30
        rows.append({"modele": m, "pli": f, "epoques": len(v), "meilleure_epoque": int(ep[ib]),
                     "val_dsc_max": v[ib], "val_dsc_finale": v[-1], "perte_apres_max": v[ib] - v[-1],
                     "epoque_95pct_du_max": int(ep[np.argmax(v >= 0.95 * v[ib])]),
                     "pente_val_dsc_30ep": slope_v, "pente_train_loss_30ep": slope_t,
                     "instabilite_sd_dval_30ep": float(np.std(np.diff(v[last]), ddof=1)),
                     "train_loss_finale": tl[-1]})
    return h, pd.DataFrame(rows)


# =========================================================== 6. temps
def training_times():
    rows = []
    for p in sorted((ROOT / "results/phase3_ablation/_journaux").glob("*/*.log")):
        ev = json.loads(p.read_text(encoding="utf-8"))
        txt = "".join(e.get("data", "") for e in ev if e.get("stream_name") == "stdout")
        # chaque ligne d'epoque apparait deux fois dans le journal : dedoublonnage par numero d'epoque
        t = list({int(e): float(x) for e, x in re.findall(r"\[Epoch\s+(\d+)\][^\n]*?\(([\d.]+)s\)", txt)}.values())
        par = re.findall(r"parametres : ([\d,]+)", txt)
        fold = re.findall(r"Fold (\d) : train", txt)
        if not t or not par:
            rows.append({"notebook": p.parent.name, "statut": "journal sans entrainement (session ulterieure)"})
            continue
        n = int(par[0].replace(",", ""))
        model = {7918061: "E1_UNet_base", 18802973: "E3_UNet_opt", 18796101: "E2_SegResNet"}[n]
        rows.append({"notebook": p.parent.name, "statut": "ok", "modele": model, "pli": int(fold[0]),
                     "parametres": n, "epoques": len(t), "s_par_epoque_mediane": float(np.median(t)),
                     "heures_epoques": sum(t) / 3600, "heures_session": max(e["time"] for e in ev) / 3600})
    df = pd.DataFrame(rows)
    ok = df[df.statut == "ok"].drop_duplicates(["modele", "pli"])   # unet-fold-0 = copie de unet-fold-0-opt
    return df, ok


def inference_times(c):
    t = c.drop_duplicates(["model", "patient_id"])[["model", "fold", "patient_id", "t_infer_s"]].copy()
    t["exclu"] = (t.model == "E1_UNet_base") & (t.fold == 0)     # pli 0 de E1 evalue sur CPU (local)
    return t


# =========================================================== 7. figures
def fig_curves(h, cs):
    for col, name, ylab in [("train_loss", "fig01_courbes_train_loss", "Perte d'entraînement (DiceCE)"),
                            ("val_dsc", "fig02_courbes_val_dsc", "DSC moyen de validation (4 OAR)")]:
        fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
        for ax, m in zip(axs, ORDER):
            g = h[h.model == m]
            for f, gf in g.groupby("fold"):
                ax.plot(gf.epoch, gf[col], color=COL[m], lw=0.8, alpha=0.35)
            mean = g.groupby("epoch")[col].mean()
            ax.plot(mean.index, mean.values, color=COL[m], lw=2, label="moyenne des 5 plis")
            if col == "val_dsc":
                b = cs[cs.modele == m]
                ax.scatter(b.meilleure_epoque, b.val_dsc_max, s=22, color=COL[m], edgecolor=SURF, lw=1.2,
                           zorder=3, label="meilleure époque (best.pt)")
            ax.set_title(MODELS[m])
            ax.set_xlabel("Époque")
        axs[0].set_ylabel(ylab)
        axs[0].legend(loc="lower right" if col == "val_dsc" else "upper right", fontsize=7.5)
        fig.text(0.01, -0.10, "Traits fins : plis individuels. Perte de validation et DSC d'entraînement non "
                 "enregistrés par trainer.py.", fontsize=7, color=MUTED)
        save(fig, name)


def fig_boxplots(c):
    sel = [("dsc", False), ("hd95_mm", True), ("assd_mm", True), ("recall", False)]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, (k, logy) in zip(axs.flat, sel):
        pos, data, cols = [], [], []
        for i, o in enumerate(ORGANS):
            for j, m in enumerate(ORDER):
                x = c[(c.model == m) & (c.organ == o)][k].dropna().to_numpy()
                data.append(x)
                pos.append(i * 4 + j)
                cols.append(COL[m])
        bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=True,
                        medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": MUTED},
                        capprops={"color": MUTED}, flierprops={"marker": ".", "ms": 2.5, "mec": MUTED, "alpha": 0.5})
        for b, cc in zip(bp["boxes"], cols):
            b.set_facecolor(cc)
            b.set_edgecolor(SURF)
            b.set_linewidth(1.5)
        ax.set_xticks([i * 4 + 1 for i in range(4)], list(ORGANS.values()))
        ax.set_ylabel(METRICS[k][0] + (" — échelle log" if logy else ""))
        if logy:
            ax.set_yscale("log")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[m]) for m in ORDER]
    fig.legend(handles, [MODELS[m] for m in ORDER], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, -0.01, "187 patients, grille native. Distances : prédictions vides exclues (E1 : voir tableau 03).",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig03_boxplots_par_organe")


def fig_heatmap(desc):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.1))
    for ax, k, cmap, fmt in [(axs[0], "dsc", "Greens", "{:.3f}"), (axs[1], "hd95_mm", "Oranges", "{:.1f}")]:
        d = desc[desc.metrique == k].pivot(index="modele", columns="organe", values="mediane").loc[ORDER, list(ORGANS)]
        im = ax.imshow(d.values, cmap=cmap, aspect="auto")
        vmin, vmax = np.nanmin(d.values), np.nanmax(d.values)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                v = d.values[i, j]
                dark = (v - vmin) / (vmax - vmin + 1e-12) > 0.55
                ax.text(j, i, fmt.format(v).replace(".", ","), ha="center", va="center", fontsize=8,
                        color=SURF if dark else INK)
        ax.set_xticks(range(4), list(ORGANS.values()), rotation=20, ha="right")
        ax.set_yticks(range(3), [MODELS[m] for m in ORDER])
        ax.grid(False)
        ax.set_title(f"{METRICS[k][0]} médian" + (" (plus foncé = plus grand)" if k == "hd95_mm" else ""))
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    save(fig, "fig04_heatmap_medianes")


def fig_forest(pr):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, k in zip(axs, ["dsc", "hd95_mm"]):
        s = pr[pr.metrique == k]
        y = 0
        ticks, labs = [], []
        for o in ORGANS:
            for comp, ccol in [("E2_SegResNet - E3_UNet_opt", COL["E2_SegResNet"]),
                               ("E3_UNet_opt - E1_UNet_base", COL["E3_UNet_opt"]),
                               ("E2_SegResNet - E1_UNet_base", COL["E1_UNet_base"])]:
                r = s[(s.organe == o) & (s.comparaison == comp)].iloc[0]
                ax.plot([r.ic95_bas, r.ic95_haut], [y, y], color=ccol, lw=2, solid_capstyle="round")
                ax.plot(r.mediane_diff, y, "o", color=ccol, ms=6, mec=SURF, mew=1.2)
                y -= 1
            ticks.append(y + 2)
            labs.append(ORGANS[o])
            y -= 0.8
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(ticks, labs)
        ax.set_xlabel(f"Différence appariée de {METRICS[k][0]} (médiane, IC 95 % bootstrap)")
    hs = [plt.Line2D([], [], color=c_, lw=2, marker="o") for c_ in (COL["E2_SegResNet"], COL["E3_UNet_opt"], COL["E1_UNet_base"])]
    fig.legend(hs, ["E2 − E3 (même taille)", "E3 − E1 (effet de la taille, U-Net)", "E2 − E1"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig05_differences_appariees")


def fig_times(tinf, ttr):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
    ax = axs[0]
    data = [tinf[(tinf.model == m) & ~tinf.exclu].t_infer_s.to_numpy() for m in ORDER]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops={"color": INK},
                    flierprops={"marker": ".", "ms": 2.5, "mec": MUTED})
    for b, m in zip(bp["boxes"], ORDER):
        b.set_facecolor(COL[m])
        b.set_edgecolor(SURF)
    ax.set_xticks([1, 2, 3], [MODELS[m] for m in ORDER], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Inférence par patient (s, log) — T4")
    ax = axs[1]
    for j, m in enumerate(ORDER):
        s = ttr[ttr.modele == m].sort_values("pli")
        ax.bar(s.pli + (j - 1) * 0.27, s.heures_session, width=0.25, color=COL[m], label=MODELS[m])
    ax.set_xticks(range(5), [f"pli {k}" for k in range(5)])
    ax.set_ylabel("Durée de la session d'entraînement (h)")
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.text(0.01, -0.04, "Inférence : E1 pli 0 exclu (évalué sur CPU). Entraînement : plis sans journal "
             "exploitable absents (voir tableau 06).", fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "fig06_temps_inference_entrainement")


def fig_masks(c):
    """Superposition réf./prédictions E3 et E2 : patient à DSC œsophage E3 médian et patient au pire DSC œsophage E3."""
    zips = {"E3_UNet_opt": [P2 / "abdelhalimnssiri_b/phase2_eval/predictions.zip", P2 / "nssiri02_b/phase2_eval/predictions.zip"],
            "E2_SegResNet": [P2 / "nssiri02_b/phase2_eval/predictions.zip"]}
    var = {"E3_UNet_opt": "unet_opt", "E2_SegResNet": "segresnet"}

    def load_pred(m, fold, pid):
        for z in zips[m]:
            with zipfile.ZipFile(z) as zf:
                name = f"{var[m]}/fold{fold}/{pid}_pred.nii.gz"
                if name in zf.namelist():
                    return np.asanyarray(nib.Nifti1Image.from_bytes(
                        __import__("gzip").decompress(zf.read(name))).dataobj).astype(np.uint8)
        raise FileNotFoundError(f"{m} {pid}")

    e3 = c[(c.model == "E3_UNet_opt") & (c.organ == "oesophage")].dropna(subset=["dsc"]).sort_values("dsc")
    picks = [("DSC œsophage E3 médian", e3.iloc[len(e3) // 2]), ("pire DSC œsophage E3", e3.iloc[0])]
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    rows = []
    for r, (why, row) in enumerate(picks):
        pid, fold = row.patient_id, int(row.fold)
        img_n = nib.load(str(ROOT / "data" / pid / "image.nii.gz"))
        lab_n = nib.load(str(ROOT / "data" / pid / "label.nii.gz"))
        img = np.asanyarray(img_n.dataobj).astype(np.float32)
        gt = np.rint(np.asanyarray(lab_n.dataobj)).astype(np.uint8)
        preds = {m: load_pred(m, fold, pid) for m in ("E3_UNet_opt", "E2_SegResNet")}
        # coupe axiale passant par le centre de l'œsophage de référence (sinon du cœur)
        ref = gt == 4 if (gt == 4).any() else gt == 3
        z = int(np.round(ndi.center_of_mass(ref)[2]))
        lo, hi = np.percentile(img[..., z], [0.5, 99.5])
        sl = lambda a: np.rot90(a[..., z])   # noqa: E731
        dscs = c[(c.patient_id == pid) & (c.organ == "oesophage")].set_index("model").dsc
        for k, (title, m) in enumerate([("Référence", None), ("E3 U-Net 18,8 M", "E3_UNet_opt"),
                                        ("E2 SegResNet", "E2_SegResNet")]):
            ax = axs[r, k]
            ax.imshow(sl(img), cmap="gray", vmin=lo, vmax=hi)
            for cl in range(1, 5):
                if (gt[..., z] == cl).any():
                    ax.contour(sl(gt == cl).astype(float), levels=[0.5], colors=["white"], linewidths=1.0,
                               linestyles="--" if m else "-")
                if m and (preds[m][..., z] == cl).any():
                    ax.contour(sl(preds[m] == cl).astype(float), levels=[0.5], colors=[COL[m]], linewidths=1.4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            sub = f"\nDSC œsophage {dscs[m]:.2f}".replace(".", ",") if m else f"\n{pid}, pli {fold} — {why}"
            ax.set_title(title + sub, fontsize=8.5)
        rows.append({"patient": pid, "pli": fold, "choix": why, "coupe_axiale": z,
                     "dsc_oeso_E3": dscs["E3_UNet_opt"], "dsc_oeso_E2": dscs["E2_SegResNet"]})
    fig.text(0.01, 0.0, "Blanc : référence (pointillé sur les colonnes prédiction). Couleur : prédiction du modèle "
             "(4 OAR). Coupe axiale par le centre de l'œsophage de référence ; orientation d'affichage non radiologique.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1), h_pad=2.5)
    save(fig, "fig07_superpositions_masques")
    pd.DataFrame(rows).to_csv(TAB / "09_patients_figures_masques.csv", index=False)


# =========================================================== main
def main():
    n_inv = inventory()
    c, e4, chk = load_metrics()
    print(f"inventaire : {n_inv} fichiers | controle TP reconstruit (precision/sensibilite) : ecart max {chk:.2e}")
    ctl = controls(c)
    desc = descriptives(c, ORDER)
    desc.to_csv(TAB / "02_descriptifs_E1_E2_E3_187.csv", index=False)
    pr = paired(c, COMPARISONS)
    pr.to_csv(TAB / "03_comparaisons_appariees_187.csv", index=False)
    # E4 : 76 patients (plis 0-1), checkpoint final, protocole different -> exploratoire
    c76 = pd.concat([c[c.fold.isin([0, 1])], e4], ignore_index=True)
    descriptives(c76, ORDER + ["E4_nnUNet"]).to_csv(TAB / "04_descriptifs_avec_E4_76.csv", index=False)
    paired(c76, [("E4_nnUNet", "E2_SegResNet"), ("E4_nnUNet", "E3_UNet_opt")]).to_csv(
        TAB / "05_comparaisons_E4_76.csv", index=False)
    folds = (c.groupby(["model", "fold", "organ"]).dsc.mean().unstack("organ")[list(ORGANS)])
    folds["moyenne_4_OAR"] = folds.mean(axis=1)
    folds.to_csv(TAB / "07_dsc_moyen_par_pli.csv")
    sd = folds.groupby(level=0).std(ddof=1)
    sd.to_csv(TAB / "07b_ecart_type_entre_plis.csv")
    h, cs = curves()
    cs.to_csv(TAB / "08_courbes_apprentissage.csv", index=False)
    tall, ttr = training_times()
    tall.to_csv(TAB / "06_temps_entrainement_journaux.csv", index=False)
    tinf = inference_times(c)
    ti = tinf[~tinf.exclu].groupby("model").t_infer_s.describe(percentiles=[0.25, 0.5, 0.75])
    ti.to_csv(TAB / "06b_temps_inference.csv")
    fig_curves(h, cs)
    fig_boxplots(c)
    fig_heatmap(desc)
    fig_forest(pr)
    fig_times(tinf, ttr)
    fig_masks(c)
    print("termine :", OUT)


if __name__ == "__main__":
    main()
