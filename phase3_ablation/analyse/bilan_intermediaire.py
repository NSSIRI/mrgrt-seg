"""Bilan intermediaire de l'ablation (lecture seule) : regroupe les metrics_long.csv deja recuperes
(results/phase3_ablation/<compte>/<job>/ablation_eval/) et les compare a E3/E2 sur les MEMES patients (grille native).

    .venv\\Scripts\\python.exe phase3_ablation/analyse/bilan_intermediaire.py

Exploratoire : plis incomplets, une graine ; pas de test statistique ici (fait sur les 5 plis dans le rapport final).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ORG = ["poumon_g", "poumon_d", "coeur", "oesophage"]

runs = []
for f in sorted((ROOT / "results/phase3_ablation").glob("*/abl18m-*/ablation_eval/metrics_long.csv")):
    if "ctrl" in f.parts[-3]:
        continue
    d = pd.read_csv(f)
    runs.append(d[d.space == "native"])
ab = pd.concat(runs, ignore_index=True).drop_duplicates(["variant", "seed", "fold", "patient_id", "organ"])
ref = pd.read_csv(ROOT / "results/phase2_eval/analyse/combined_long.csv")
ref = ref[ref.space == "native"].replace({"model": {"E3_UNet_opt": "E3", "E2_SegResNet": "E2"}})
ref = ref[ref.model.isin(["E3", "E2"])]
piv = ref.pivot_table(index=["patient_id", "organ"], columns="model", values=["dsc", "hd95_mm"])

rows = []
for (v, s), g in ab.groupby(["variant", "seed"]):
    folds = sorted(g.fold.unique())
    for o in ORG:
        x = g[g.organ == o].set_index("patient_id")
        p = piv.xs(o, level="organ").reindex(x.index)
        d_e3 = (x.dsc - p[("dsc", "E3")]).dropna()
        d_e2 = (x.dsc - p[("dsc", "E2")]).dropna()
        gap = (p[("dsc", "E2")] - p[("dsc", "E3")]).median()
        rows.append({"variante": v, "graine": s, "plis": ",".join(map(str, folds)), "n": len(x), "organe": o,
                     "DSC_var": x.dsc.median(), "DSC_E3": p[("dsc", "E3")].median(), "DSC_E2": p[("dsc", "E2")].median(),
                     "diff_vs_E3": d_e3.median(), "diff_vs_E2": d_e2.median(),
                     "pct_mieux_que_E3": 100 * (d_e3 > 0).mean(),
                     "part_ecart_comble": d_e3.median() / gap if gap else np.nan,
                     "HD95_var": x.hd95_mm.median(), "HD95_E3": p[("hd95_mm", "E3")].median(),
                     "HD95_E2": p[("hd95_mm", "E2")].median()})
t = pd.DataFrame(rows)
t.to_csv(OUT / "bilan_intermediaire.csv", index=False)
pd.set_option("display.width", 200)
print(t.round(3).to_string(index=False))
