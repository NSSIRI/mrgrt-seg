"""Genere 5 patients synthetiques (NIfTI) pour tester le pipeline.

Cree :
    data/
      PATIENT_001/
        image.nii.gz   -> volume "IRM" synthetique 192x192x64
        label.nii.gz   -> masque multiclasses 5 OAR (BG + poumons G/D + coeur + oesophage)
      PATIENT_002/
        ...
      PATIENT_005/

Anatomie tres approximative (formes geometriques), juste pour valider le pipeline.
Pour de vraies donnees, voir HPC_MARWAN.md (TotalSegmentator MRI, AAPM Thoracic).

Usage :
    python make_demo_data.py
    (depuis le dossier mrgrt_seg, apres avoir active le venv)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib


def synth_patient(seed: int, shape=(192, 192, 64)):
    """Genere un volume + label synthetiques. Anatomie thoracique tres simplifiee."""
    rng = np.random.default_rng(seed)
    H, W, D = shape

    # Image : bruit gaussien + gradients pour simuler un IRM
    img = rng.normal(loc=200, scale=30, size=shape).astype(np.float32)
    # Ajouter un fond plus sombre autour pour simuler l'air
    Y, X, Z = np.ogrid[:H, :W, :D]
    cy, cx = H // 2, W // 2
    body_mask = ((X - cx) ** 2 / (W * 0.4) ** 2 +
                 (Y - cy) ** 2 / (H * 0.35) ** 2) <= 1.0
    img[~body_mask] = rng.normal(loc=20, scale=10, size=img.shape)[~body_mask]

    # Label : 0 = BG, 1 = poumon_g, 2 = poumon_d, 3 = coeur, 4 = oesophage
    label = np.zeros(shape, dtype=np.uint8)

    # Poumons : 2 ellipsoides decalees a gauche et a droite
    poumon_g_offset = (-W // 6 + int(rng.normal(0, 3)), -H // 12, 0)
    poumon_d_offset = ( W // 6 + int(rng.normal(0, 3)), -H // 12, 0)
    for cls_id, offset in [(1, poumon_g_offset), (2, poumon_d_offset)]:
        ox, oy, oz = offset
        mask = (((X - (cx + ox)) ** 2) / (W * 0.18) ** 2 +
                ((Y - (cy + oy)) ** 2) / (H * 0.22) ** 2 +
                ((Z - D // 2) ** 2) / (D * 0.35) ** 2) <= 1.0
        label[mask] = cls_id
        # Image plus sombre dans les poumons (parenchyme = peu de protons en IRM)
        img[mask] = rng.normal(loc=80, scale=15, size=img.shape)[mask]

    # Coeur : ellipsoide centrale legerement decalee a gauche
    heart_offset = (-W // 24, H // 14, 0)
    ox, oy, oz = heart_offset
    heart_mask = (((X - (cx + ox)) ** 2) / (W * 0.10) ** 2 +
                  ((Y - (cy + oy)) ** 2) / (H * 0.12) ** 2 +
                  ((Z - D // 2) ** 2) / (D * 0.25) ** 2) <= 1.0
    label[heart_mask] = 3
    img[heart_mask] = rng.normal(loc=250, scale=20, size=img.shape)[heart_mask]

    # Oesophage : petit cylindre vertical, posterieur, central
    eso_x = cx + int(rng.normal(0, 2))
    eso_y = cy + H // 8  # posterieur
    eso_mask = ((X - eso_x) ** 2 + (Y - eso_y) ** 2) <= (5 ** 2)
    eso_mask = eso_mask & (Z >= D // 4) & (Z <= D * 3 // 4)
    label[eso_mask] = 4
    img[eso_mask] = rng.normal(loc=180, scale=10, size=img.shape)[eso_mask]

    # Spacing realiste : 1.5 x 1.5 x 3 mm
    affine = np.diag([1.5, 1.5, 3.0, 1.0])
    return img, label, affine


def main():
    root = Path("data")
    root.mkdir(exist_ok=True)
    print(f"Generation des donnees demo dans : {root.resolve()}")
    for i in range(1, 6):
        pid = f"PATIENT_{i:03d}"
        patient_dir = root / pid
        patient_dir.mkdir(exist_ok=True)
        img, lbl, affine = synth_patient(seed=i)
        nib.save(nib.Nifti1Image(img.astype(np.float32), affine),
                 patient_dir / "image.nii.gz")
        nib.save(nib.Nifti1Image(lbl.astype(np.uint8), affine),
                 patient_dir / "label.nii.gz")
        n_vox_per_class = [int((lbl == c).sum()) for c in range(5)]
        print(f"  {pid} cree. Voxels par classe : {n_vox_per_class}")
    print()
    print("Termine. Vous pouvez maintenant :")
    print("  1. Visualiser dans 3D Slicer : ouvrir data/PATIENT_001/image.nii.gz")
    print("     puis charger label.nii.gz comme segmentation par-dessus")
    print("  2. Tester le pipeline :")
    print("     python scripts/train.py --model unet --fold 0 --config configs/default.yaml")
    print("     (sur CPU ce sera lent : reduire epochs dans default.yaml pour test)")


if __name__ == "__main__":
    main()
