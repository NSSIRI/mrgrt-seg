"""Filtre qualite pour le dataset MRgRT segmentation OAR thorax.

Lit les NIfTI deja convertis (sortie de prepare_totalsegmentator.py) et applique
3 criteres pour ne garder que les patients ou le thorax est CLINIQUEMENT COMPLET.

Criteres d'exclusion (un seul suffit) :
  1. BOUNDARY : un OAR labellise touche un bord de l'image (cut-off probable)
  2. VOLUME   : un OAR present a un volume anormalement petit (< seuil)
  3. FOV      : etendue cranio-caudale (axe Z) trop courte (< ~20 cm)

Mode "rapport seul" (--dry-run) : produit le rapport JSON sans rien copier.
Mode "production"             : copie les patients qui passent dans dst/

Usage :
    python scripts/transfer/05_filter_quality.py \\
        --src /scratch/users/a.nssiri/mrgrt-seg/data \\
        --dst /scratch/users/a.nssiri/mrgrt-seg/data_thorax_complet \\
        [--dry-run]                  # rapport sans copie
        [--min_lung_ml 500]          # poumon minimum 500 mL
        [--min_heart_ml 100]         # coeur minimum 100 mL
        [--min_esophagus_ml 5]       # oesophage tres petit, 5 mL min
        [--min_fov_cc_mm 180]        # FOV cranio-caudal min 18 cm
        [--strict_boundary]          # rejeter aussi si TOUT OAR touche un bord
                                     # (par defaut on tolere oesophage et coeur
                                     # qui touchent souvent legitimement les
                                     # bords du FOV thoracique standard)

Le script ecrit egalement un CSV par-patient avec toutes les mesures, utile
pour expliquer les choix de seuil dans le manuscrit de these.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

CLASS_NAMES = {1: "poumon_g", 2: "poumon_d", 3: "coeur", 4: "oesophage"}


def voxel_volume_ml(affine: np.ndarray) -> float:
    """Volume d'un voxel en mL a partir de l'affine NIfTI (mm3 -> mL)."""
    # |det(affine[:3,:3])| en mm3 / 1000 = mL
    return abs(np.linalg.det(affine[:3, :3])) / 1000.0


def fov_cc_mm(affine: np.ndarray, shape: tuple[int, int, int]) -> float:
    """Etendue cranio-caudale en mm.

    Heuristique : l'axe cranio-caudal est celui dont le vecteur de l'affine
    a la plus grande composante Z (en valeur absolue) en RAS+. Pour la
    plupart des IRM standardisees, c'est l'axe k (3eme axe du volume).
    """
    # Norme de chaque vecteur colonne * taille de cet axe
    extents_mm = []
    for axis in range(3):
        spacing = float(np.linalg.norm(affine[:3, axis]))
        extents_mm.append(spacing * shape[axis])
    # On prend l'axe le plus probable d'etre cranio-caudal :
    # celui dont la composante Z (3eme ligne de l'affine) est dominante.
    z_components = np.abs(affine[2, :3])
    cc_axis = int(np.argmax(z_components))
    return extents_mm[cc_axis]


def touches_boundary(mask: np.ndarray, class_id: int, margin: int = 1) -> bool:
    """True si le masque de classe `class_id` touche un bord de l'image.

    `margin` : nombre de voxels en marge — par defaut 1 (la classe touche
    physiquement le bord). Augmenter a 2-3 pour etre plus strict.
    """
    region = mask == class_id
    if not region.any():
        return False
    # Verifie chaque face
    for axis in range(3):
        # Premiere face : voxels d'indice 0..margin-1
        sl_start = [slice(None)] * 3
        sl_start[axis] = slice(0, margin)
        if region[tuple(sl_start)].any():
            return True
        # Derniere face : voxels d'indice -margin..end
        sl_end = [slice(None)] * 3
        sl_end[axis] = slice(-margin, None)
        if region[tuple(sl_end)].any():
            return True
    return False


def analyze_patient(patient_dir: Path) -> dict:
    """Calcule toutes les metriques qualite pour un patient."""
    img_path = patient_dir / "image.nii.gz"
    lbl_path = patient_dir / "label.nii.gz"
    if not (img_path.exists() and lbl_path.exists()):
        return {"status": "error", "reason": "image ou label absent"}

    img_nii = nib.load(str(img_path))
    lbl_nii = nib.load(str(lbl_path))
    lbl = np.asanyarray(lbl_nii.dataobj).astype(np.int32)

    affine = img_nii.affine
    shape = img_nii.shape
    vox_ml = voxel_volume_ml(affine)
    fov_cc = fov_cc_mm(affine, shape)

    classes_present = sorted(int(c) for c in np.unique(lbl) if c > 0)
    volumes_ml = {}
    boundaries = {}
    for c in classes_present:
        n_vox = int((lbl == c).sum())
        volumes_ml[CLASS_NAMES[c]] = round(n_vox * vox_ml, 2)
        boundaries[CLASS_NAMES[c]] = touches_boundary(lbl, c)

    return {
        "status": "ok",
        "shape": list(shape),
        "voxel_ml": round(vox_ml, 4),
        "fov_cc_mm": round(fov_cc, 1),
        "classes_present": [CLASS_NAMES[c] for c in classes_present],
        "volumes_ml": volumes_ml,
        "boundaries": boundaries,
    }


def evaluate_quality(metrics: dict, args) -> tuple[bool, list[str]]:
    """Retourne (keep, reasons_if_excluded)."""
    if metrics["status"] != "ok":
        return False, [f"erreur: {metrics.get('reason', '?')}"]

    reasons = []
    # FOV cranio-caudal
    if metrics["fov_cc_mm"] < args.min_fov_cc_mm:
        reasons.append(f"FOV cranio-caudal trop court "
                       f"({metrics['fov_cc_mm']:.0f} mm < {args.min_fov_cc_mm})")

    # Volumes
    thresholds_ml = {
        "poumon_g":  args.min_lung_ml,
        "poumon_d":  args.min_lung_ml,
        "coeur":     args.min_heart_ml,
        "oesophage": args.min_esophagus_ml,
    }
    for organ, vol in metrics["volumes_ml"].items():
        thr = thresholds_ml.get(organ)
        if thr is not None and vol < thr:
            reasons.append(f"{organ} trop petit ({vol:.0f} mL < {thr} mL)")

    # Boundary touching
    # Par defaut : poumon_g et poumon_d strict ; coeur et oesophage tolerants
    # (l'oesophage et le coeur peuvent legitimement toucher z=0 ou z=max
    # quand le FOV thoracique standard est utilise)
    strict_organs = {"poumon_g", "poumon_d"}
    if args.strict_boundary:
        strict_organs = set(metrics["boundaries"].keys())
    for organ, touches in metrics["boundaries"].items():
        if touches and organ in strict_organs:
            reasons.append(f"{organ} touche un bord (cut-off probable)")

    return (len(reasons) == 0), reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Dossier source (apres prepare_totalsegmentator)")
    parser.add_argument("--dst", required=True,
                        help="Dossier destination (filtre)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Rapport seul, ne copie rien")
    parser.add_argument("--min_lung_ml", type=float, default=500,
                        help="Volume minimum par poumon en mL (defaut 500)")
    parser.add_argument("--min_heart_ml", type=float, default=100,
                        help="Volume minimum coeur en mL (defaut 100)")
    parser.add_argument("--min_esophagus_ml", type=float, default=5,
                        help="Volume minimum oesophage en mL (defaut 5)")
    parser.add_argument("--min_fov_cc_mm", type=float, default=180,
                        help="FOV cranio-caudal minimum en mm (defaut 180)")
    parser.add_argument("--strict_boundary", action="store_true",
                        help="Rejeter aussi si coeur/oesophage touche un bord")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        sys.exit(f"ERREUR : src introuvable : {src}")

    patients = sorted([p for p in src.iterdir() if p.is_dir() and p.name.startswith("PATIENT_")])
    print(f"Analyse de {len(patients)} patients...")

    kept, excluded, errors = [], [], []
    rows = []  # pour CSV

    for i, pdir in enumerate(patients, 1):
        m = analyze_patient(pdir)
        keep, reasons = evaluate_quality(m, args)

        # Ligne CSV
        row = {"patient": pdir.name}
        if m["status"] == "ok":
            row.update({
                "shape": "x".join(str(s) for s in m["shape"]),
                "fov_cc_mm": m["fov_cc_mm"],
                "voxel_ml": m["voxel_ml"],
                "classes": "+".join(m["classes_present"]),
                "vol_poumon_g_ml": m["volumes_ml"].get("poumon_g", 0),
                "vol_poumon_d_ml": m["volumes_ml"].get("poumon_d", 0),
                "vol_coeur_ml":    m["volumes_ml"].get("coeur", 0),
                "vol_oesophage_ml": m["volumes_ml"].get("oesophage", 0),
                "boundary_poumon_g": int(m["boundaries"].get("poumon_g", False)),
                "boundary_poumon_d": int(m["boundaries"].get("poumon_d", False)),
                "boundary_coeur":    int(m["boundaries"].get("coeur", False)),
                "boundary_oesophage": int(m["boundaries"].get("oesophage", False)),
                "kept": int(keep),
                "exclusion_reasons": " | ".join(reasons) if reasons else "",
            })
        else:
            row.update({
                "shape": "", "fov_cc_mm": "", "voxel_ml": "",
                "classes": "", "vol_poumon_g_ml": "", "vol_poumon_d_ml": "",
                "vol_coeur_ml": "", "vol_oesophage_ml": "",
                "boundary_poumon_g": "", "boundary_poumon_d": "",
                "boundary_coeur": "", "boundary_oesophage": "",
                "kept": 0,
                "exclusion_reasons": reasons[0] if reasons else "erreur inconnue",
            })
            errors.append(pdir.name)
        rows.append(row)

        if keep:
            kept.append(pdir.name)
            tag = "KEEP"
        else:
            excluded.append({"patient": pdir.name, "reasons": reasons})
            tag = "EXCLU"

        if i % 25 == 0 or i == len(patients):
            print(f"  [{i}/{len(patients)}] {pdir.name}: {tag}"
                  + (f"  ({reasons[0]})" if reasons else ""))

    # CSV
    dst.mkdir(parents=True, exist_ok=True)
    csv_path = dst / "_quality_report.csv"
    import csv
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nRapport CSV detaille : {csv_path}")

    # JSON summary
    summary = {
        "thresholds": vars(args),
        "n_input": len(patients),
        "n_kept": len(kept),
        "n_excluded": len(excluded),
        "n_errors": len(errors),
        "kept": kept,
        "excluded": excluded,
        "errors": errors,
    }
    json_path = dst / "_quality_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Resume JSON         : {json_path}")

    # Copie
    if args.dry_run:
        print(f"\nMode --dry-run : aucun fichier copie.")
    else:
        print(f"\nCopie de {len(kept)} patients dans {dst}/ ...")
        for pname in kept:
            src_p = src / pname
            dst_p = dst / pname
            if dst_p.exists():
                continue  # idempotent
            shutil.copytree(src_p, dst_p)
        print(f"Copie terminee.")

    # Bilan
    print("\n" + "=" * 60)
    print(f" BILAN : {len(kept)} gardes  |  {len(excluded)} exclus  |  {len(errors)} erreurs")
    print(f" (sur {len(patients)} patients d'entree)")
    print("=" * 60)
    print(f"\nTop 5 raisons d'exclusion :")
    reason_counts = {}
    for e in excluded:
        for r in e["reasons"]:
            # On groupe par debut de phrase (avant les chiffres)
            key = r.split("(")[0].strip()
            reason_counts[key] = reason_counts.get(key, 0) + 1
    for key, n in sorted(reason_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"   {n:3d} x  {key}")


if __name__ == "__main__":
    main()
