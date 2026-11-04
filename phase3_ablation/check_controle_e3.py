"""Controle obligatoire (etape 3) : la reevaluation du best.pt E3 pli 0 par eval_ablation.py doit reproduire
results/phase2_eval/abdelhalimnssiri_b/phase2_eval/metrics_long.csv a moins de 1e-6.

    python phase3_ablation/check_controle_e3.py [metrics_long.csv du job ctrl-e3-f0]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results/phase2_eval/abdelhalimnssiri_b/phase2_eval/metrics_long.csv"
NEW = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "results/phase3_ablation/abdelhalimnssiri/abl18m-ctrl-e3-f0/ablation_eval/metrics_long.csv"
KEY = ["variant", "fold", "patient_id", "organ", "space"]
SKIP = {"t_infer_s", "t_metrics_s", "seed", "tag"}
TOL = 1e-6

ref = pd.read_csv(REF)
new = pd.read_csv(NEW)
ref = ref[(ref.variant == "unet_opt") & (ref.fold == 0)]
new = new[(new.variant == "unet_opt") & (new.fold == 0)]
m = ref.merge(new, on=KEY, how="outer", suffixes=("_ref", "_new"), indicator=True)
missing = m[m["_merge"] != "both"]
cols = [c for c in ref.columns if c not in KEY and c not in SKIP and pd.api.types.is_numeric_dtype(ref[c])]
worst = {}
for c in cols:
    a, b = m[c + "_ref"].to_numpy(float), m[c + "_new"].to_numpy(float)
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.where(both_nan, 0.0, np.abs(a - b))
    worst[c] = float(np.nanmax(np.where(np.isnan(d), np.inf, d))) if len(d) else 0.0
ecart = max(worst.values()) if worst else float("inf")
print(f"lignes : reference {len(ref)} | controle {len(new)} | non appariees {len(missing)}")
for c, v in sorted(worst.items(), key=lambda kv: -kv[1])[:8]:
    print(f"   {c:<18} ecart max {v:.3g}")
ok = len(missing) == 0 and len(ref) > 0 and ecart < TOL
print(f"ECART MAXIMAL {ecart:.3g} -> {'CONTROLE REUSSI' if ok else 'CONTROLE ECHOUE'} (seuil {TOL})")
sys.exit(0 if ok else 1)
