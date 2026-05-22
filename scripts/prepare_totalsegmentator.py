"""Convertit le dataset TotalSegmentator MRI vers la structure attendue par
le pipeline mrgrt_seg.

Entree (apres extraction du zip) :
    raw_totalseg/
        s0001/
            mri.nii.gz   (ou ct.nii.gz selon les cas)
            segmentations/
                lung_upper_lobe_left.nii.gz
                lung_upper_lobe_right.nii.gz
                ...
                heart.nii.gz
                esophagus.nii.gz

Sortie :
    data/PATIENT_s0001/
        image.nii.gz
        label.nii.gz    (multi-classes : 0=bg, 1=poumon_g, 2=poumon_d, 3=coeur, 4=oesophage)

Usage :
    cd C:\\Users\\Lenovo\\Desktop\\mrgrt_seg
    .venv\\Scripts\\activate
    python scripts\\prepare_totalsegmentator.py \\
        --src raw_totalseg \\
        --dst data \\
        [--max 5]                # limiter pour test rapide

Le script :
- fusionne les lobes pulmonaires (5 lobes -> poumon gauche/droit)
- combine en un seul masque multiclasse
- preserve la geometrie spatiale (affine NIfTI)
- ecrit un resume JSON
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib


# Mapping organe TotalSegmentator -> classe projet
# Gere les deux conventions :
#   - CT : poumons decomposes en 5 lobes
#   - MRI : poumons en blocs (lung_left, lung_right) ou variantes
ORGAN_TO_CLASS = {
    # ---- POUMON GAUCHE (classe 1) ----
    # Convention MRI (TotalSegmentator MRI v2)
    "lung_left":              1,
    # Convention CT (TotalSegmentator CT)
    "lung_upper_lobe_left":   1,
    "lung_lower_lobe_left":   1,
    # Variantes vues dans d'autres datasets
    "left_lung":              1,
    "lung_l":                 1,

    # ---- POUMON DROIT (classe 2) ----
    "lung_right":             2,
    "lung_upper_lobe_right":  2,
    "lung_middle_lobe_right": 2,
    "lung_lower_lobe_right":  2,
    "right_lung":             2,
    "lung_r":                 2,

    # ---- COEUR (classe 3) ----
    # MRI : souvent en sous-structures
    "heart":                  3,
    "heart_myocardium":       3,
    "heart_atrium_left":      3,
    "heart_atrium_right":     3,
    "heart_ventricle_left":   3,
    "heart_ventricle_right":  3,
    "myocardium":             3,
    "atrium_left":            3,
    "atrium_right":           3,
    "ventricle_left":         3,
    "ventricle_right":        3,

    # ---- OESOPHAGE (classe 4) ----
    "esophagus":              4,
    "oesophagus":             4,
}

CLASS_NAMES = {
    1: "poumon_g",
    2: "poumon_d",
    3: "coeur",
    4: "oesophage",
}


def find_image(patient_dir: Path) -> Path | None:
    """Trouve le fichier image principal (mri.nii.gz, ct.nii.gz, ou autre)."""
    for name in ["mri.nii.gz", "ct.nii.gz", "image.nii.gz",
                 "MRI.nii.gz", "CT.nii.gz"]:
        p = patient_dir / name
        if p.exists():
            return p
    # fallback : tout .nii.gz au niveau racine, hors segmentations/
    for p in patient_dir.glob("*.nii.gz"):
        return p
    return None


def convert_patient(src_dir: Path, dst_dir: Path) -> dict:
    """Convertit un patient TotalSegmentator vers le format projet.

    Returns
    -------
    dict avec resume : organes trouves, classes presentes, etc.
    """
    img_path = find_image(src_dir)
    if img_path is None:
        return {"status": "error", "reason": "image principale introuvable"}

    seg_dir = src_dir / "segmentations"
    if not seg_dir.exists():
        return {"status": "error", "reason": "dossier segmentations/ absent"}

    # Charger l'image
    img_nii = nib.load(str(img_path))
    img_data = np.asanyarray(img_nii.dataobj)
    img_shape = img_data.shape
    affine = img_nii.affine

    # Construire le masque multi-classes
    label = np.zeros(img_shape, dtype=np.uint8)
    organs_found = []
    organs_missing = []
    classes_present = set()

    for organ_name, class_id in ORGAN_TO_CLASS.items():
        organ_file = seg_dir / f"{organ_name}.nii.gz"
        if not organ_file.exists():
            organs_missing.append(organ_name)
            continue
        organ_nii = nib.load(str(organ_file))
        organ_data = np.asanyarray(organ_nii.dataobj) > 0
        if organ_data.shape != img_shape:
            return {"status": "error",
                    "reason": f"shape mismatch pour {organ_name} "
                              f"({organ_data.shape} vs {img_shape})"}
        # CRITIQUE : verifier que l'organe a des voxels non-nuls.
        # TotalSegmentator MRI v2 contient des fichiers d'organe VIDES (tous a 0)
        # pour les patients ou l'organe n'est pas dans le FOV. Sans ce check,
        # `classes_present` est augmente meme avec un fichier vide, ce qui
        # produit des label.nii.gz vides qui passent le filtre qualite.
        n_vox_organ = int(organ_data.sum())
        if n_vox_organ < 10:  # seuil tres bas (10 voxels = ~0.5 mL)
            organs_missing.append(f"{organ_name} (fichier present mais vide)")
            continue
        # Priorite : on n'ecrase que les voxels background
        label[organ_data & (label == 0)] = class_id
        organs_found.append(organ_name)
        classes_present.add(class_id)

    if not organs_found:
        return {"status": "error",
                "reason": "aucun organe d'interet trouve dans segmentations/"}

    # Sauvegarde
    dst_dir.mkdir(parents=True, exist_ok=True)
    nib.save(img_nii, str(dst_dir / "image.nii.gz"))
    nib.save(nib.Nifti1Image(label, affine, img_nii.header),
             str(dst_dir / "label.nii.gz"))

    # Statistiques par classe (nombre de voxels)
    voxel_counts = {}
    for c in sorted(classes_present):
        voxel_counts[CLASS_NAMES[c]] = int((label == c).sum())

    return {
        "status": "ok",
        "image_shape": list(img_shape),
        "image_source": img_path.name,
        "organs_found": sorted(organs_found),
        "classes_present": sorted(CLASS_NAMES[c] for c in classes_present),
        "voxel_counts": voxel_counts,
        "organs_missing_optional": sorted(organs_missing),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Dossier source (ex: raw_totalseg/)")
    parser.add_argument("--dst", required=True,
                        help="Dossier destination (ex: data/)")
    parser.add_argument("--max", type=int, default=None,
                        help="Limiter le nombre de patients traites.")
    parser.add_argument("--min_classes", type=int, default=2,
                        help="Nombre minimum de classes d'OAR thoraciques "
                             "requises pour garder le patient (defaut: 2). "
                             "Les patients avec moins seront EXCLUS (non thoracique).")
    parser.add_argument("--require_lungs", action="store_true",
                        help="Exiger les 2 poumons pour garder le patient.")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        sys.exit(f"ERREUR : src introuvable : {src}")

    # Lister les dossiers patients (heuristique : ceux qui ont segmentations/)
    candidates = sorted([p for p in src.iterdir()
                         if p.is_dir() and (p / "segmentations").exists()])
    if not candidates:
        # essai : sous-niveau (au cas ou raw_totalseg/dataset/s0001/...)
        for sub in src.iterdir():
            if sub.is_dir():
                sub_cands = sorted([p for p in sub.iterdir()
                                    if p.is_dir() and (p / "segmentations").exists()])
                if sub_cands:
                    candidates = sub_cands
                    print(f"Sous-niveau detecte : {sub}")
                    break
    if not candidates:
        sys.exit(f"ERREUR : aucun patient (avec segmentations/) trouve dans {src}")

    if args.max:
        candidates = candidates[:args.max]
    print(f"Patients a convertir : {len(candidates)}")

    summary = {"patients": {}, "kept": [], "excluded": [], "errors": []}
    n_ok, n_excluded, n_err = 0, 0, 0
    import shutil
    for patient_dir in candidates:
        pid = patient_dir.name
        out_name = f"PATIENT_{pid}"
        out_dir = dst / out_name
        print(f"  -> {pid} ... ", end="", flush=True)
        result = convert_patient(patient_dir, out_dir)
        summary["patients"][pid] = result

        if result["status"] != "ok":
            print(f"ECHEC : {result['reason']}")
            summary["errors"].append(pid)
            n_err += 1
            continue

        classes_present = result["classes_present"]
        n_classes = len(classes_present)
        has_both_lungs = ("poumon_g" in classes_present
                          and "poumon_d" in classes_present)

        # Filtres d'inclusion
        excluded_reason = None
        if n_classes < args.min_classes:
            excluded_reason = f"seulement {n_classes} classe(s) trouvee(s) " \
                              f"(< {args.min_classes} requis)"
        if args.require_lungs and not has_both_lungs:
            excluded_reason = "poumons gauche+droit non tous deux presents"

        if excluded_reason:
            # Patient non thoracique : on supprime le dossier produit
            if out_dir.exists():
                shutil.rmtree(out_dir)
            print(f"EXCLU ({excluded_reason})")
            summary["excluded"].append({"pid": pid, "reason": excluded_reason,
                                        "classes_found": classes_present})
            n_excluded += 1
        else:
            classes_str = "+".join(classes_present)
            print(f"OK [{n_classes}/4] ({classes_str})")
            summary["kept"].append({"pid": pid, "classes": classes_present})
            n_ok += 1

    (dst / "_conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print("=" * 60)
    print(f" BILAN : {n_ok} gardes  |  {n_excluded} exclus  |  {n_err} erreurs")
    print("=" * 60)
    if n_excluded > 0:
        print(f" Patients exclus (non thoraciques, < {args.min_classes} OAR) :")
        for e in summary["excluded"][:10]:
            print(f"   - {e['pid']} : {e['reason']}")
        if len(summary["excluded"]) > 10:
            print(f"   ... et {len(summary['excluded']) - 10} autres")
    print()
    print(f"Resume detaille (JSON) : {dst / '_conversion_summary.json'}")
    print()
    print("Structure produite pour les patients gardes :")
    if summary["kept"]:
        first = summary["kept"][0]["pid"]
        print(f"  {dst}/PATIENT_{first}/image.nii.gz")
        print(f"  {dst}/PATIENT_{first}/label.nii.gz")


if __name__ == "__main__":
    main()
