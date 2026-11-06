"""Ablation 18,8 M - analyse statistique finale (grille native, 187 patients). Lecture seule.

Modeles : E3 (U-Net 18,8 M) et E2 (SegResNet) graine 42 [phase 2] et graine 123 [phase 3, bloc A] ;
          A1 U-Net + num_res_units=2, A2 U-Net + GroupNorm(8), A3 SegResNet + InstanceNorm [graine 42, bloc B].

Q1 graines : medianes ; ecart E2 - E3 par graine et sur les valeurs moyennees par patient sur les 2 graines ;
             effet de la graine (s123 - s42) par modele ; ecart-type entre graines des moyennes de pli.
Q2-Q4 variantes : chaque variante vs E3 et vs E2 (graine 42) ; Wilcoxon apparie bilateral, Holm par metrique sur
             variantes x OAR x reference (24 tests) ; IC 95 % bootstrap de la mediane des differences (10 000 tirages) ;
             r de rang bisérial ; part de l'ecart comblee = med(V - E3) / med(E2 - E3) avec IC bootstrap apparie.
Regle d'interpretation (TACHE §4) : un effet plus petit que la variabilite entre graines n'est pas interpretable.
             Seuil par OAR et metrique = plus grande borne absolue des IC 95 % de l'effet graine (E3 et E2).

    .venv\\Scripts\\python.exe phase3_ablation/analyse/ablation_stats.py
Sorties : phase3_ablation/analyse/stats_*.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261106)
NBOOT = 10_000
ORG = ["poumon_g", "poumon_d", "coeur", "oesophage"]
METRICS = {"dsc": 1, "hd95_mm": -1, "assd_mm": -1, "sdsc_2mm": 1}   # +1 : plus grand = mieux
NAMES = {("unet_res", 42): "A1", ("unet_gn", 42): "A2", ("segresnet_in", 42): "A3",
         ("unet_opt", 123): "E3_s123", ("segresnet", 123): "E2_s123"}


def load():
    c = pd.read_csv(ROOT / "results/phase2_eval/analyse/combined_long.csv")
    c = c[(c.space == "native") & c.model.isin(["E3_UNet_opt", "E2_SegResNet"])].copy()
    c["m"] = c.model.map({"E3_UNet_opt": "E3_s42", "E2_SegResNet": "E2_s42"})
    runs = []
    for f in sorted((ROOT / "results/phase3_ablation").glob("*/abl18m-*/ablation_eval/metrics_long.csv")):
        if "ctrl" in str(f):
            continue
        d = pd.read_csv(f)
        runs.append(d[d.space == "native"])
    a = pd.concat(runs, ignore_index=True).drop_duplicates(["variant", "seed", "fold", "patient_id", "organ"])
    a["m"] = [NAMES.get((v, int(s))) for v, s in zip(a.variant, a.seed)]
    a = a[a.m.notna()]
    cols = ["m", "fold", "patient_id", "organ"] + list(METRICS)
    df = pd.concat([c[cols], a[cols]], ignore_index=True)
    n = df.groupby(["m", "organ"]).patient_id.nunique()
    assert (n == 187).all(), n[n != 187]
    return df


def wide(df, metric):
    return df.pivot_table(index=["patient_id", "organ"], columns="m", values=metric)


def paired(x, y, sense):
    """Statistiques appariees de d = x - y (paires completes)."""
    d = (x - y).dropna().to_numpy()
    nz = d[d != 0]
    rk = rankdata(np.abs(nz))
    r = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum() if len(nz) else np.nan
    p = wilcoxon(d, zero_method="wilcox").pvalue if len(nz) else 1.0
    idx = RNG.integers(0, len(d), size=(NBOOT, len(d)))
    bm = np.median(d[idx], axis=1)
    return {"n": len(d), "med_diff": np.median(d), "ic_bas": np.percentile(bm, 2.5), "ic_haut": np.percentile(bm, 97.5),
            "pct_x_meilleur": 100 * (d * sense > 0).mean(), "p": p, "r_rb": r}


def holm(p):
    p = np.asarray(p, float)
    o, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    df = load()
    med = df.groupby(["m", "organ"])[list(METRICS)].median().reset_index()
    med.to_csv(OUT / "stats_medianes.csv", index=False)

    # ---------- Q1 : graines
    q1 = []
    for metric, sense in METRICS.items():
        w = wide(df, metric)
        for o in ORG:
            s = w.xs(o, level="organ")
            avg_e3 = s[["E3_s42", "E3_s123"]].mean(axis=1)
            avg_e2 = s[["E2_s42", "E2_s123"]].mean(axis=1)
            for lab, x, y in [("E2-E3 graine 42", s.E2_s42, s.E3_s42), ("E2-E3 graine 123", s.E2_s123, s.E3_s123),
                              ("E2-E3 moyenne des 2 graines", avg_e2, avg_e3),
                              ("effet graine E3 (s123-s42)", s.E3_s123, s.E3_s42),
                              ("effet graine E2 (s123-s42)", s.E2_s123, s.E2_s42)]:
                q1.append({"metrique": metric, "organe": o, "comparaison": lab, **paired(x, y, sense)})
    q1 = pd.DataFrame(q1)
    q1.to_csv(OUT / "stats_q1_graines.csv", index=False)

    # ecart-type entre graines des moyennes de pli (par modele, OAR) ; ecart-type entre plis pour comparaison
    fm = df[df.m.isin(["E3_s42", "E3_s123", "E2_s42", "E2_s123"])].groupby(["m", "fold", "organ"]).dsc.mean().reset_index()
    fm["modele"], fm["graine"] = fm.m.str[:2], fm.m.str[-3:].str.lstrip("s")
    sd = (fm.groupby(["modele", "organ", "fold"]).dsc.std(ddof=1).groupby(["modele", "organ"]).mean()
          .rename("sd_entre_graines_moy_plis").reset_index())
    sd2 = fm.groupby(["modele", "graine", "organ"]).dsc.std(ddof=1).groupby(["modele", "organ"]).mean().rename("sd_entre_plis").reset_index()
    sd.merge(sd2, on=["modele", "organ"]).to_csv(OUT / "stats_q1_variabilite_plis.csv", index=False)

    # seuil d'interpretabilite : plus grande borne absolue des IC de l'effet graine (E3, E2)
    thr = (q1[q1.comparaison.str.startswith("effet graine")]
           .assign(b=lambda t: t[["ic_bas", "ic_haut"]].abs().max(axis=1))
           .groupby(["metrique", "organe"]).b.max().rename("seuil_graine"))

    # ---------- Q2-Q4 : variantes
    rows = []
    for metric, sense in METRICS.items():
        w = wide(df, metric)
        block = []
        for v in ["A1", "A2", "A3"]:
            for ref in ["E3_s42", "E2_s42"]:
                for o in ORG:
                    s = w.xs(o, level="organ")
                    block.append({"metrique": metric, "variante": v, "reference": ref[:2], "organe": o,
                                  **paired(s[v], s[ref], sense)})
        ph = holm([b["p"] for b in block])
        for b, q in zip(block, ph):
            b["p_holm"] = q
        rows += block
    q2 = pd.DataFrame(rows).merge(thr.reset_index(), left_on=["metrique", "organe"], right_on=["metrique", "organe"])
    q2["interpretable"] = (q2.p_holm < 0.05) & (q2.med_diff.abs() > q2.seuil_graine) & \
                          (np.sign(q2.ic_bas) == np.sign(q2.ic_haut))
    q2.to_csv(OUT / "stats_q2_variantes.csv", index=False)

    # part de l'ecart comblee (DSC et HD95), IC bootstrap apparie (reechantillonnage des patients)
    gap = []
    for metric in ["dsc", "hd95_mm"]:
        w = wide(df, metric)
        for v in ["A1", "A2", "A3"]:
            for o in ORG:
                s = w.xs(o, level="organ")[[v, "E3_s42", "E2_s42"]].dropna()
                a, e3, e2 = s[v].to_numpy(), s.E3_s42.to_numpy(), s.E2_s42.to_numpy()
                part = np.median(a - e3) / np.median(e2 - e3)
                idx = RNG.integers(0, len(s), size=(NBOOT, len(s)))
                num, den = np.median((a - e3)[idx], axis=1), np.median((e2 - e3)[idx], axis=1)
                ok = den != 0
                bt = num[ok] / den[ok]
                gap.append({"metrique": metric, "variante": v, "organe": o, "n": len(s), "part_comblee": part,
                            "ic_bas": np.percentile(bt, 2.5), "ic_haut": np.percentile(bt, 97.5)})
    pd.DataFrame(gap).to_csv(OUT / "stats_part_ecart.csv", index=False)

    # sensibilite : references moyennees sur les 2 graines
    sens = []
    w = wide(df, "dsc")
    for v in ["A1", "A2", "A3"]:
        for refname, cols in [("E3 (moy. 2 graines)", ["E3_s42", "E3_s123"]), ("E2 (moy. 2 graines)", ["E2_s42", "E2_s123"])]:
            for o in ORG:
                s = w.xs(o, level="organ")
                sens.append({"variante": v, "reference": refname, "organe": o, **paired(s[v], s[cols].mean(axis=1), 1)})
    pd.DataFrame(sens).to_csv(OUT / "stats_sensibilite_ref_moyennee.csv", index=False)

    pd.set_option("display.width", 220)
    print("== Medianes DSC\n", med.pivot(index="m", columns="organ", values="dsc")[ORG].round(3))
    print("\n== Q1 (DSC)\n", q1[q1.metrique == "dsc"][["organe", "comparaison", "med_diff", "ic_bas", "ic_haut", "p", "r_rb"]].round(4).to_string(index=False))
    print("\n== Variabilite\n", pd.read_csv(OUT / "stats_q1_variabilite_plis.csv").round(4).to_string(index=False))
    print("\n== Q2 (DSC)\n", q2[q2.metrique == "dsc"][["variante", "reference", "organe", "med_diff", "ic_bas", "ic_haut", "pct_x_meilleur", "p_holm", "r_rb", "seuil_graine", "interpretable"]].round(4).to_string(index=False))
    print("\n== Q2 (HD95)\n", q2[q2.metrique == "hd95_mm"][["variante", "reference", "organe", "med_diff", "ic_bas", "ic_haut", "p_holm", "seuil_graine", "interpretable"]].round(3).to_string(index=False))
    print("\n== Part de l'ecart\n", pd.read_csv(OUT / "stats_part_ecart.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
