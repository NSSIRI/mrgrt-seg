"""Phase 4 (preparation) : combien d'examens ont la moelle epiniere et l'aorte annotees et visibles dans le thorax ?

Lecture seule de TotalsegmentatorMRI_dataset_v200/<sXXXX>/segmentations/*.nii.gz. Sortie : ce dossier.

Pour chaque examen (616) et pour la cohorte D187 (data/PATIENT_sXXXX) :
  - nombre de voxels de spinal_cord, aorta, lung_left, lung_right ;
  - etendue cranio-caudale des poumons (coupes axiales, orientation RAS canonique) ;
  - couverture = fraction des coupes pulmonaires contenant la structure (moelle, aorte) ;
  - aorte : couverture sur les 75 % inferieurs des coupes pulmonaires (les apex depassent la crosse aortique) ;
  - troncature = la structure touche la premiere ou la derniere coupe du volume (normal pour moelle et aorte,
    qui depassent le thorax : indicatif seulement).

    .venv\\Scripts\\python.exe results/phase4_organes/comptage_moelle_aorte.py
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "TotalsegmentatorMRI_dataset_v200"
OUT = Path(__file__).resolve().parent
STRUCT = ["lung_left", "lung_right", "spinal_cord", "aorta"]


def load(sid, name):
    p = RAW / sid / "segmentations" / f"{name}.nii.gz"
    if not p.exists():
        return None
    img = nib.as_closest_canonical(nib.load(str(p)))          # axe 2 = inferieur -> superieur
    return np.asanyarray(img.dataobj) > 0


def main():
    d187 = {p.name.replace("PATIENT_", "") for p in (ROOT / "data").iterdir() if p.is_dir()}
    rows = []
    for sdir in sorted(p for p in RAW.iterdir() if p.is_dir() and p.name.startswith("s")):
        sid = sdir.name
        m = {k: load(sid, k) for k in STRUCT}
        r = {"examen": sid, "cohorte_D187": sid in d187}
        for k, v in m.items():
            r[f"vox_{k}"] = int(v.sum()) if v is not None else -1
        lungs = [m[k] for k in ("lung_left", "lung_right") if m[k] is not None]
        zl = np.flatnonzero(np.any(np.logical_or.reduce(lungs), axis=(0, 1))) if lungs else np.array([])
        r["coupes_poumons"] = len(zl)
        for k in ("spinal_cord", "aorta"):
            v = m[k]
            if v is None or not v.any():
                r[f"couv_{k}"], r[f"tronque_{k}"], r[f"couv75_{k}"] = 0.0, False, 0.0
                continue
            zk = np.any(v, axis=(0, 1))
            r[f"couv_{k}"] = float(zk[zl].mean()) if len(zl) else np.nan
            r[f"tronque_{k}"] = bool(zk[0] or zk[-1])
            # les apex pulmonaires depassent la crosse aortique : couverture sur les 75 % inferieurs des coupes pulmonaires
            low = zl[: max(1, int(round(0.75 * len(zl))))] if len(zl) else zl
            r[f"couv75_{k}"] = float(zk[low].mean()) if len(low) else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "comptage_moelle_aorte_par_examen.csv", index=False)

    lines = []
    for name, sub in [("616 examens", df), ("cohorte D187", df[df.cohorte_D187]),
                      ("examens avec poumons G et D", df[(df.vox_lung_left > 0) & (df.vox_lung_right > 0)])]:
        n = len(sub)
        def cnt(mask):
            return f"{int(mask.sum())} ({100 * mask.mean():.0f} %)" if n else "0"
        both = lambda t: (sub.couv_spinal_cord >= t) & (sub.couv75_aorta >= t)  # noqa: E731
        lines.append({"sous_ensemble": name, "n": n,
                      "moelle_presente": cnt(sub.vox_spinal_cord > 0), "aorte_presente": cnt(sub.vox_aorta > 0),
                      "moelle_couv>=0.95": cnt(sub.couv_spinal_cord >= 0.95),
                      "aorte_couv75bas>=0.95": cnt(sub.couv75_aorta >= 0.95),
                      "les_deux>=0.90": cnt(both(0.90)), "les_deux>=0.95": cnt(both(0.95))})
    s = pd.DataFrame(lines)
    s.to_csv(OUT / "comptage_moelle_aorte_resume.csv", index=False)
    print(s.T.to_string())
    d = df[df.cohorte_D187]
    print("\nD187, couverture (quantiles 5/25/50 %) : moelle",
          np.round(np.nanpercentile(d.couv_spinal_cord, [5, 25, 50]), 3),
          "| aorte (75 % inf.)", np.round(np.nanpercentile(d.couv75_aorta, [5, 25, 50]), 3))


if __name__ == "__main__":
    main()
