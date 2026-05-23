"""Comparaison statistique de deux configurations (architectures ou datasets).

Agrege les CSV par-patient produits par evaluate.py, apparie les patients par
patient_id, et effectue un test de Wilcoxon signe apparie avec correction de
Bonferroni. Produit la table de comparaison de l'article (Table 3 / ablation).

Exemples :
    # U-Net vs SegResNet (5 folds chacun, dataset filtre)
    python scripts/compare_results.py \
        --a "results/unet_fold*.csv" --name_a "U-Net" \
        --b "results/segresnet_fold*.csv" --name_b "SegResNet" \
        --metric dsc --out results/comparison_unet_vs_segresnet.csv

    # Ablation : meme modele, dataset filtre vs non filtre
    python scripts/compare_results.py \
        --a "results_filtered/unet_fold*.csv" --name_a "Filtered (n=187)" \
        --b "results_full/unet_fold*.csv"     --name_b "Unfiltered (n=616)" \
        --metric dsc
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.metrics import wilcoxon_signed_rank, bonferroni

CLASS_NAMES = ["poumon_g", "poumon_d", "coeur", "oesophage"]


def load_pooled(pattern: str, metric: str) -> dict:
    """Charge et concatene les CSV (multi-folds) -> {classe: {patient_id: valeur}}."""
    import csv
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"ERREUR : aucun fichier ne correspond a {pattern}")
    data = {c: {} for c in CLASS_NAMES}
    for fp in files:
        with open(fp) as f:
            reader = csv.DictReader(f)
            for r in reader:
                pid = r["patient_id"]
                for c in CLASS_NAMES:
                    col = f"{metric}_{c}"
                    if col in r and r[col] not in ("", "nan"):
                        try:
                            data[c][pid] = float(r[col])
                        except ValueError:
                            pass
    print(f"  {pattern} -> {len(files)} fichiers, "
          f"{len(data[CLASS_NAMES[0]])} patients")
    return data


def bootstrap_ci(values, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    boots = [np.median(rng.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    return (float(np.median(values)),
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Glob CSV config A")
    parser.add_argument("--b", required=True, help="Glob CSV config B")
    parser.add_argument("--name_a", default="A")
    parser.add_argument("--name_b", default="B")
    parser.add_argument("--metric", default="dsc",
                        choices=["dsc", "iou", "hd95", "surface_dsc"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"Config A ({args.name_a}) :")
    data_a = load_pooled(args.a, args.metric)
    print(f"Config B ({args.name_b}) :")
    data_b = load_pooled(args.b, args.metric)

    # Tests par classe (sur patients communs aux deux configs)
    results = []
    pvals = []
    for c in CLASS_NAMES:
        common = sorted(set(data_a[c]) & set(data_b[c]))
        if not common:
            print(f"ATTENTION : aucun patient commun pour {c}")
            continue
        va = [data_a[c][p] for p in common]
        vb = [data_b[c][p] for p in common]
        med_a, lo_a, hi_a = bootstrap_ci(va)
        med_b, lo_b, hi_b = bootstrap_ci(vb)
        w = wilcoxon_signed_rank(va, vb)
        pvals.append(w["pvalue"])
        results.append({
            "oar": c, "n": len(common),
            f"{args.name_a}_median": med_a, f"{args.name_a}_ci_low": lo_a, f"{args.name_a}_ci_high": hi_a,
            f"{args.name_b}_median": med_b, f"{args.name_b}_ci_low": lo_b, f"{args.name_b}_ci_high": hi_b,
            "median_diff": w["median_diff"], "pvalue": w["pvalue"],
        })

    # Bonferroni sur l'ensemble des classes testees
    corrected = bonferroni(pvals, n_tests=len(pvals))
    for r, pc in zip(results, corrected):
        r["pvalue_bonferroni"] = pc
        r["significant"] = pc < 0.05

    # Affichage
    print(f"\n=== Comparaison {args.name_a} vs {args.name_b} (metrique: {args.metric}) ===")
    print(f"{'OAR':<11} {'n':>4} {args.name_a[:12]:>14} {args.name_b[:12]:>14} "
          f"{'diff':>8} {'p':>9} {'p_bonf':>9} {'sig':>4}")
    for r in results:
        print(f"{r['oar']:<11} {r['n']:>4} "
              f"{r[f'{args.name_a}_median']:>14.3f} {r[f'{args.name_b}_median']:>14.3f} "
              f"{r['median_diff']:>8.3f} {r['pvalue']:>9.4f} "
              f"{r['pvalue_bonferroni']:>9.4f} {'OUI' if r['significant'] else 'non':>4}")

    if args.out:
        import csv
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\nTable ecrite : {args.out}")


if __name__ == "__main__":
    main()
