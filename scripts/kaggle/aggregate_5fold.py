"""Agregation des 5-fold cross-validation pour UNet et SegResNet.

Lit les CSV de results/batch_unet/ et results/batch_segresnet/, agrege
les 5 folds par modele, calcule moyennes ± std par OAR/metrique, fait
les tests Wilcoxon paired UNet vs SegResNet avec correction de Bonferroni.

Produit :
  - results/aggregated_5fold.csv : tableau final pour la publication
  - results/wilcoxon_results.csv : tests statistiques
  - results/per_patient_5fold.csv : tous les 187 patients par modele

Usage :
  python scripts/kaggle/aggregate_5fold.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[ATTENTION] scipy absent, tests Wilcoxon desactives")
    print("  Pour installer : pip install scipy")


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ORGANS = ["poumon_g", "poumon_d", "coeur", "oesophage"]
METRICS = ["dsc", "iou", "hd95", "surface_dsc"]


def load_model_folds(model: str) -> pd.DataFrame:
    """Charge tous les CSV d'un modele et les concatene avec colonne 'fold'."""
    batch_dir = RESULTS / f"batch_{model}"
    if not batch_dir.exists():
        # Fallback : chercher dans results/
        candidates = sorted(RESULTS.glob(f"{model}_fold*_metrics.csv"))
        if not candidates:
            sys.exit(f"ERREUR : aucun CSV trouve pour {model}")
    else:
        candidates = sorted(batch_dir.rglob(f"{model}_fold*_metrics.csv"))

    dfs = []
    for csv in candidates:
        # Extraire le numero de fold du nom
        import re
        m = re.search(rf"{model}_fold(\d+)_metrics\.csv", csv.name)
        if not m:
            continue
        fold = int(m.group(1))
        df = pd.read_csv(csv)
        df["fold"] = fold
        df["model"] = model
        dfs.append(df)
    if not dfs:
        sys.exit(f"ERREUR : aucun fold trouve pour {model}")
    return pd.concat(dfs, ignore_index=True)


def aggregate_metrics(df: pd.DataFrame, model: str) -> dict:
    """Calcule moyenne ± std par OAR/metrique sur tous les patients."""
    out = {"model": model, "n_patients": len(df)}
    for m in METRICS:
        for o in ORGANS:
            col = f"{m}_{o}"
            if col in df.columns:
                vals = df[col].replace([np.inf, -np.inf], np.nan).dropna().values
                out[f"{col}_mean"] = round(float(vals.mean()), 4) if len(vals) else np.nan
                out[f"{col}_std"] = round(float(vals.std()), 4) if len(vals) else np.nan
                out[f"{col}_n"] = len(vals)
    # DSC moyen global (moyenne des 4 OAR par patient)
    dsc_cols = [f"dsc_{o}" for o in ORGANS]
    if all(c in df.columns for c in dsc_cols):
        per_patient_mean = df[dsc_cols].mean(axis=1).values
        out["dsc_mean_global"] = round(float(per_patient_mean.mean()), 4)
        out["dsc_std_global"] = round(float(per_patient_mean.std()), 4)
    return out


def wilcoxon_per_oar(unet: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    """Tests Wilcoxon paired (par patient) UNet vs SegResNet pour chaque OAR x metrique."""
    if not HAS_SCIPY:
        return pd.DataFrame()
    # Merger par patient_id pour assurer le pairing
    merged = unet[["patient_id"] + [f"{m}_{o}" for m in METRICS for o in ORGANS]].merge(
        seg[["patient_id"] + [f"{m}_{o}" for m in METRICS for o in ORGANS]],
        on="patient_id", suffixes=("_unet", "_seg")
    )
    rows = []
    for m in METRICS:
        for o in ORGANS:
            col = f"{m}_{o}"
            u = merged[f"{col}_unet"].replace([np.inf, -np.inf], np.nan).values
            s = merged[f"{col}_seg"].replace([np.inf, -np.inf], np.nan).values
            mask = ~(np.isnan(u) | np.isnan(s))
            u, s = u[mask], s[mask]
            if len(u) < 5:
                continue
            # Direction : pour HD95 plus petit = mieux, donc test inverse
            if m == "hd95":
                alt = "greater"  # H1: UNet > SegResNet (UNet pire)
            else:
                alt = "less"     # H1: UNet < SegResNet (UNet pire)
            try:
                stat, p = stats.wilcoxon(u, s, alternative=alt)
            except Exception:
                stat, p = np.nan, np.nan
            rows.append({
                "metric": m, "organ": o, "n_pairs": len(u),
                "unet_mean": round(float(u.mean()), 4),
                "seg_mean": round(float(s.mean()), 4),
                "delta": round(float(s.mean() - u.mean()), 4),
                "wilcoxon_stat": round(float(stat), 2) if not np.isnan(stat) else None,
                "p_value": float(p) if not np.isnan(p) else None,
            })
    df = pd.DataFrame(rows)
    # Correction Bonferroni
    n_tests = len(df)
    df["p_bonferroni"] = df["p_value"].apply(lambda x: min(x * n_tests, 1.0) if x is not None else None)
    df["significant_005"] = df["p_bonferroni"].apply(lambda x: x is not None and x < 0.05)
    return df


def main():
    print("=" * 70)
    print(" AGREGATION 5-FOLD CV")
    print("=" * 70)

    # 1) Charger les 2 modeles
    print("\n[1] Chargement des CSV...")
    unet = load_model_folds("unet")
    seg = load_model_folds("segresnet")
    print(f"   UNet      : {len(unet)} lignes ({unet['fold'].nunique()} folds, {unet['patient_id'].nunique()} patients uniques)")
    print(f"   SegResNet : {len(seg)} lignes ({seg['fold'].nunique()} folds, {seg['patient_id'].nunique()} patients uniques)")

    # 2) Sauver les per-patient
    per_pat = pd.concat([unet, seg], ignore_index=True)
    per_pat_path = RESULTS / "per_patient_5fold.csv"
    per_pat.to_csv(per_pat_path, index=False)
    print(f"   sauve : {per_pat_path}")

    # 3) Agregation
    print("\n[2] Agregation par modele...")
    rows = [aggregate_metrics(unet, "unet"), aggregate_metrics(seg, "segresnet")]
    agg = pd.DataFrame(rows)
    agg_path = RESULTS / "aggregated_5fold.csv"
    agg.to_csv(agg_path, index=False)
    print(f"   sauve : {agg_path}")

    # 4) Tableau lisible
    print("\n" + "=" * 70)
    print(" TABLEAU FINAL UNet vs SegResNet (5-fold, n=187)")
    print("=" * 70)
    for m in METRICS:
        print(f"\n--- {m.upper()} ---")
        for o in ORGANS:
            col = f"{m}_{o}"
            u_mean = agg.loc[agg["model"] == "unet", f"{col}_mean"].values[0]
            u_std = agg.loc[agg["model"] == "unet", f"{col}_std"].values[0]
            s_mean = agg.loc[agg["model"] == "segresnet", f"{col}_mean"].values[0]
            s_std = agg.loc[agg["model"] == "segresnet", f"{col}_std"].values[0]
            delta = s_mean - u_mean
            arrow = "↑" if delta > 0 else "↓"
            print(f"  {o:<12}  UNet: {u_mean:.3f}±{u_std:.3f}  |  SegRes: {s_mean:.3f}±{s_std:.3f}  |  Δ={delta:+.3f} {arrow}")

    # DSC global
    u_g = agg.loc[agg["model"] == "unet", "dsc_mean_global"].values[0]
    u_gs = agg.loc[agg["model"] == "unet", "dsc_std_global"].values[0]
    s_g = agg.loc[agg["model"] == "segresnet", "dsc_mean_global"].values[0]
    s_gs = agg.loc[agg["model"] == "segresnet", "dsc_std_global"].values[0]
    print(f"\n  DSC GLOBAL    UNet: {u_g:.3f}±{u_gs:.3f}  |  SegRes: {s_g:.3f}±{s_gs:.3f}  |  Δ={s_g-u_g:+.3f}")

    # 5) Wilcoxon
    if HAS_SCIPY:
        print("\n[3] Tests Wilcoxon paired (UNet vs SegResNet) + Bonferroni...")
        wil = wilcoxon_per_oar(unet, seg)
        wil_path = RESULTS / "wilcoxon_results.csv"
        wil.to_csv(wil_path, index=False)
        print(f"   sauve : {wil_path}")
        print()
        print("Resultats significatifs (p_bonferroni < 0.05) :")
        sig = wil[wil["significant_005"]]
        if len(sig) == 0:
            print("   AUCUN — peut-etre n trop petit, ou pas de difference")
        else:
            for _, r in sig.iterrows():
                print(f"   {r['metric']:>4} {r['organ']:<12}  UNet={r['unet_mean']:.3f}  SegRes={r['seg_mean']:.3f}  delta={r['delta']:+.3f}  p={r['p_bonferroni']:.2e}")

    print("\n" + "=" * 70)
    print(" Outputs :")
    print(f"   - {agg_path}")
    print(f"   - {per_pat_path}")
    if HAS_SCIPY:
        print(f"   - {wil_path}")
    print("=" * 70)
    print()
    print("Prochaines etapes :")
    print("  1. Verifie le tableau ci-dessus pour confirmer SegResNet > UNet")
    print("  2. Genere les figures (boxplots, courbes loss)")
    print("  3. Demarre la redaction Publication 2 avec ces chiffres")


if __name__ == "__main__":
    main()
