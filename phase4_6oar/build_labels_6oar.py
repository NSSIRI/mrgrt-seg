"""Phase 4 - labels a 6 OAR pour la cohorte D115.

Cohorte : patients D187 dont la moelle couvre >= 95 % des coupes pulmonaires ET l'aorte >= 95 % des 75 %
inferieurs des coupes pulmonaires (results/phase4_organes/comptage_moelle_aorte_par_examen.csv).

Label = label D187 existant (data/PATIENT_s*/label.nii.gz : 1 poumon G, 2 poumon D, 3 coeur, 4 oesophage, INCHANGES)
        + 5 moelle epiniere + 6 aorte (TotalsegmentatorMRI_dataset_v200/sXXXX/segmentations), ajoutes
        uniquement sur les voxels de fond (priorite aux 4 OAR historiques, comme prepare_totalsegmentator.py).

Lecture seule de data/ et du jeu brut. Ecrit data_6oar/PATIENT_sXXXX/label.nii.gz (pas d'image : sur Kaggle,
l'image vient du dataset existant) + data_6oar/manifest_6oar.csv.

    .venv\\Scripts\\python.exe phase4_6oar/build_labels_6oar.py
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "TotalsegmentatorMRI_dataset_v200"
SRC = ROOT / "data"
DST = ROOT / "data_6oar"
COUNTS = ROOT / "results/phase4_organes/comptage_moelle_aorte_par_examen.csv"
THRESH = 0.95
NEW = {"spinal_cord": 5, "aorta": 6}


def main():
    c = pd.read_csv(COUNTS)
    c = c[c.cohorte_D187 & (c.couv_spinal_cord >= THRESH) & (c.couv75_aorta >= THRESH)]
    rows = []
    for sid in sorted(c.examen):
        pid = f"PATIENT_{sid}"
        lab_nii = nib.load(str(SRC / pid / "label.nii.gz"))
        lab = np.asanyarray(lab_nii.dataobj).astype(np.uint8)
        r = {"patient_id": pid}
        for name, cls in NEW.items():
            m_nii = nib.load(str(RAW / sid / "segmentations" / f"{name}.nii.gz"))
            if m_nii.shape != lab_nii.shape or not np.allclose(m_nii.affine, lab_nii.affine, atol=1e-4):
                raise RuntimeError(f"{pid} {name} : grille differente du label D187")
            m = np.asanyarray(m_nii.dataobj) > 0
            r[f"conflits_{name}"] = int((m & (lab > 0)).sum())     # voxels deja attribues a un OAR historique
            lab[m & (lab == 0)] = cls
        for k in range(1, 7):
            r[f"vox_{k}"] = int((lab == k).sum())
        out = DST / pid
        out.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(lab, lab_nii.affine, lab_nii.header), str(out / "label.nii.gz"))
        rows.append(r)
    m = pd.DataFrame(rows)
    m.to_csv(DST / "manifest_6oar.csv", index=False)
    print(f"{len(m)} patients ecrits dans {DST}")
    print("classes vides :", {k: int((m[f'vox_{k}'] == 0).sum()) for k in range(1, 7)})
    print("conflits (voxels deja attribues, mediane / max) :",
          {n: (int(m[f'conflits_{n}'].median()), int(m[f'conflits_{n}'].max())) for n in NEW})


if __name__ == "__main__":
    main()
