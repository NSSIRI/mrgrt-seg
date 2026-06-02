"""Aggregate the per-fold Adebayo cascading-randomization CSVs into a single
flat table ready for plotting (Figure B / Figure 8 of Pub 3).

Reads:  results/adebayo/{arch}_fold{0..4}_adebayo_full.csv  (one per arch x fold)
Writes:
  - results/adebayo/adebayo_aggregated.csv  (long-form, ready for seaborn/pandas)
  - results/adebayo/adebayo_summary.csv     (per-arch per-step mean+sd of SSIM)
  - results/adebayo/adebayo_quality_report.txt

Each input CSV has columns:
    patient_id, organ, step, layer_randomized, ssim_vs_pristine,
    in_organ_ratio, pointing_accuracy, spatial_entropy
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


ARCHS = ["unet", "segresnet"]
ORGANS = ["poumon_g", "poumon_d", "coeur", "oesophage"]


def load_one(path: Path, arch: str, fold: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["arch"] = arch
    df["fold"] = fold
    # Coerce numerics
    for c in ("step", "ssim_vs_pristine", "in_organ_ratio",
              "pointing_accuracy", "spatial_entropy"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def gather(adebayo_dir: Path) -> pd.DataFrame:
    parts = []
    for arch in ARCHS:
        for fold in range(5):
            p = adebayo_dir / f"{arch}_fold{fold}_adebayo_full.csv"
            if not p.exists():
                # also accept legacy short name
                p2 = adebayo_dir / f"{arch}_fold{fold}_adebayo.csv"
                if p2.exists():
                    p = p2
                else:
                    continue
            parts.append(load_one(p, arch, fold))
            print(f"  loaded {p.name}  ({len(parts[-1])} rows)")
    if not parts:
        raise SystemExit(
            f"No Adebayo CSV found in {adebayo_dir}. "
            "Run the Kaggle pipeline first (scripts/kaggle/12_run_adebayo_batch.ps1)."
        )
    return pd.concat(parts, ignore_index=True)


def quality_report(df: pd.DataFrame) -> str:
    out = ["=== Adebayo aggregation quality report ==="]
    out.append(f"Total rows           : {len(df)}")
    out.append(f"Unique architectures : {sorted(df['arch'].unique())}")
    out.append(f"Folds covered        : {sorted(df['fold'].unique())}")
    out.append(f"Organs               : {sorted(df['organ'].dropna().unique())}")
    out.append(f"Max step             : {int(df['step'].max())}")
    out.append("")
    out.append("Per-arch coverage:")
    for arch in ARCHS:
        sub = df[df["arch"] == arch]
        n_pat = sub["patient_id"].nunique() if "patient_id" in sub else 0
        out.append(f"  {arch:9s} : {len(sub):6d} rows, "
                   f"{n_pat:4d} patients, "
                   f"{sub['fold'].nunique()} folds")
    return "\n".join(out) + "\n"


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per (arch, step) Mean ± SD of SSIM-vs-pristine, averaging over patients/folds."""
    g = df.groupby(["arch", "step"])["ssim_vs_pristine"].agg(["mean", "std", "count"])
    g = g.reset_index().rename(columns={"mean": "ssim_mean",
                                          "std": "ssim_sd",
                                          "count": "n"})
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adebayo_dir",
                    default="results/adebayo",
                    help="Directory containing the per-arch per-fold CSVs")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write the aggregated files (default = adebayo_dir)")
    args = ap.parse_args()

    in_dir = Path(args.adebayo_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = gather(in_dir)

    long_path = out_dir / "adebayo_aggregated.csv"
    df.to_csv(long_path, index=False)
    print(f"\nWrote {long_path}  ({len(df)} rows)")

    summary = summarise(df)
    summary_path = out_dir / "adebayo_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}  ({len(summary)} rows)")

    report = quality_report(df)
    report_path = out_dir / "adebayo_quality_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
