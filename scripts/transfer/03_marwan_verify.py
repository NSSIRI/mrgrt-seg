#!/usr/bin/env python3
"""Vérification d'intégrité du dataset transféré sur MARWAN.

À exécuter SUR MARWAN, après le rsync, avec l'env mrgrt-seg activé :

    cd ~/mrgrt-seg
    source activate mrgrt-seg
    python scripts/transfer/03_marwan_verify.py

Le script vérifie :
  - le nombre de patients (attendu : 250)
  - la présence des deux fichiers (image.nii.gz, label.nii.gz) pour chacun
  - l'intégrité NIfTI : chaque fichier s'ouvre, shape cohérente image/label
  - la distribution des classes dans les labels (sanity)
  - la taille totale et la moyenne par patient

Code de retour 0 si tout OK, 1 sinon (utile pour CI / scripting).
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter

EXPECTED_N_PATIENTS = 250
EXPECTED_N_CLASSES = 5     # background + 4 OAR (poumon_g, poumon_d, coeur, oesophage)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Tolérance : le pipeline génère parfois de tout petits labels (organes peu
# visibles sur certaines coupes). On considère normal jusqu'à 5% de patients
# n'ayant pas toutes les classes ; au-delà, on lève un warning.
MAX_FRACTION_MISSING_CLASSES = 0.05


def main() -> int:
    print("=" * 60)
    print(" Vérification dataset MRgRT — post-transfert")
    print("=" * 60)
    print(f"Dossier inspecté : {DATA_DIR}")

    if not DATA_DIR.exists():
        print(f"ERREUR : dossier introuvable : {DATA_DIR}")
        print("Le symlink data/ pointe-t-il bien vers $SCRATCH/mrgrt-seg/data/ ?")
        print("Vérifier avec : ls -la ~/mrgrt-seg/data")
        return 1

    # 1) Lister les patients
    patients = sorted([p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("PATIENT_s")])
    n = len(patients)
    print(f"\n[1] Nombre de patients : {n}")
    if n != EXPECTED_N_PATIENTS:
        print(f"    ATTENTION : attendu {EXPECTED_N_PATIENTS}, trouvé {n}")
    else:
        print(f"    OK ({n} == {EXPECTED_N_PATIENTS})")

    # 2) Présence des fichiers
    missing_image, missing_label, empty_files = [], [], []
    for p in patients:
        img = p / "image.nii.gz"
        lbl = p / "label.nii.gz"
        if not img.exists():
            missing_image.append(p.name)
        elif img.stat().st_size == 0:
            empty_files.append(f"{p.name}/image.nii.gz")
        if not lbl.exists():
            missing_label.append(p.name)
        elif lbl.stat().st_size == 0:
            empty_files.append(f"{p.name}/label.nii.gz")

    print(f"\n[2] Présence des fichiers :")
    print(f"    Sans image.nii.gz : {len(missing_image)}")
    print(f"    Sans label.nii.gz : {len(missing_label)}")
    print(f"    Fichiers vides    : {len(empty_files)}")
    if missing_image[:5]:
        print(f"    Exemples sans image : {missing_image[:5]}")
    if missing_label[:5]:
        print(f"    Exemples sans label : {missing_label[:5]}")
    if empty_files[:5]:
        print(f"    Exemples vides      : {empty_files[:5]}")

    # 3) Intégrité NIfTI
    print(f"\n[3] Intégrité NIfTI (chargement des fichiers)...")
    try:
        import nibabel as nib
    except ImportError:
        print("    nibabel non installé. Sauter cette vérif ? Non — installe-le :")
        print("      pip install nibabel")
        return 1
    import numpy as np

    nifti_errors = []
    shape_mismatches = []
    class_counter: Counter = Counter()
    patients_missing_classes = 0
    total_voxels_per_class: dict[int, int] = {}
    sample_shapes = []

    for i, p in enumerate(patients, 1):
        img_path = p / "image.nii.gz"
        lbl_path = p / "label.nii.gz"
        if not img_path.exists() or not lbl_path.exists():
            continue
        try:
            img_nii = nib.load(str(img_path))
            lbl_nii = nib.load(str(lbl_path))
            img_shape = img_nii.shape
            lbl_shape = lbl_nii.shape
            if img_shape != lbl_shape:
                shape_mismatches.append(f"{p.name}: img={img_shape}, lbl={lbl_shape}")
            if len(sample_shapes) < 5:
                sample_shapes.append((p.name, img_shape))
            lbl_arr = lbl_nii.get_fdata().astype(np.int32)
            classes_present = set(int(c) for c in np.unique(lbl_arr))
            class_counter.update(classes_present)
            if len(classes_present) < EXPECTED_N_CLASSES:
                patients_missing_classes += 1
            # comptage voxels par classe (échantillonnage tous les 10 patients
            # pour éviter d'être lent)
            if i % 10 == 0:
                for c in classes_present:
                    total_voxels_per_class[c] = total_voxels_per_class.get(c, 0) + int((lbl_arr == c).sum())
        except Exception as e:
            nifti_errors.append(f"{p.name}: {type(e).__name__}: {e}")

        if i % 50 == 0:
            print(f"    ... {i}/{n}")

    print(f"    Erreurs de chargement : {len(nifti_errors)}")
    print(f"    Shape image != label  : {len(shape_mismatches)}")
    if nifti_errors[:3]:
        print(f"    Exemples erreurs   : {nifti_errors[:3]}")
    if shape_mismatches[:3]:
        print(f"    Exemples mismatch  : {shape_mismatches[:3]}")
    print(f"    Exemples de shapes : {sample_shapes}")

    # 4) Distribution des classes
    print(f"\n[4] Classes présentes (sur {n} patients) :")
    class_names = ["background", "poumon_g", "poumon_d", "coeur", "oesophage"]
    for c in range(EXPECTED_N_CLASSES):
        name = class_names[c] if c < len(class_names) else f"classe_{c}"
        present = class_counter.get(c, 0)
        frac = present / max(n, 1) * 100
        print(f"    classe {c} ({name:<11}): présent chez {present}/{n} patients ({frac:.1f}%)")

    miss_frac = patients_missing_classes / max(n, 1)
    print(f"\n    Patients avec moins de {EXPECTED_N_CLASSES} classes : "
          f"{patients_missing_classes} ({miss_frac*100:.1f}%)")
    if miss_frac > MAX_FRACTION_MISSING_CLASSES:
        print(f"    NOTE : > {MAX_FRACTION_MISSING_CLASSES*100:.0f}% des patients "
              f"n'ont pas toutes les classes. À investiguer (organe coupé, "
              f"erreur de conversion DICOM-RT ?)")

    # 5) Taille totale
    print(f"\n[5] Volumétrie :")
    total_bytes = sum(f.stat().st_size for p in patients for f in p.iterdir() if f.is_file())
    print(f"    Taille totale : {total_bytes / 1024**3:.2f} GB")
    print(f"    Moyenne / patient : {total_bytes / max(n, 1) / 1024**2:.1f} MB")

    # 6) Verdict
    print("\n" + "=" * 60)
    fatal = (
        n != EXPECTED_N_PATIENTS
        or missing_image
        or missing_label
        or empty_files
        or nifti_errors
        or shape_mismatches
    )
    if fatal:
        print(" VERDICT : PROBLEMES DETECTES — voir les sections ci-dessus.")
        print("=" * 60)
        return 1
    print(" VERDICT : OK — dataset prêt pour l'entraînement.")
    print("")
    print(" Prochain pas : sanity-run sur 1 fold, peu d'epochs :")
    print("   sbatch scripts/slurm/train_one.sbatch unet 0")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
