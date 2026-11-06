"""Ablation 18,8 M - robustesse des masques (grille native, 187 patients, 7 modeles). Lecture seule.

Memes definitions que results/phase2_eval/E2_vs_E3_18M/analyse/e2e3_masks.py :
  - composantes connexes (connexite 26) par OAR et fraction du volume predit hors de la plus grande composante ;
  - confusion gauche/droite des poumons : fraction du poumon G de reference predite poumon D (swap_G) et inversement.
Modeles : E3 et E2 graine 42 (phase 2), E3 et E2 graine 123, A1 unet_res, A2 unet_gn, A3 segresnet_in (graine 42).

    .venv\\Scripts\\python.exe phase3_ablation/analyse/ablation_masks.py [n_processus]
Sortie : phase3_ablation/analyse/ablation_masks.csv (1 ligne par patient x modele x OAR)
"""
import gzip
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P2 = ROOT / "results/phase2_eval"
R3 = ROOT / "results/phase3_ablation"
ORGANS = ["poumon_g", "poumon_d", "coeur", "oesophage"]
STRUCT26 = ndi.generate_binary_structure(3, 3)


def sources():
    """(modele, pli) -> (zip, nom interne prefixe)."""
    src = {}
    for k in range(5):
        src[("E3_s42", k)] = (P2 / ("nssiri02_b" if k == 4 else "abdelhalimnssiri_b") / "phase2_eval/predictions.zip",
                              f"unet_opt/fold{k}/")
        src[("E2_s42", k)] = (P2 / "nssiri02_b/phase2_eval/predictions.zip", f"segresnet/fold{k}/")
    names = {"unet_res": "A1", "unet_gn": "A2", "segresnet_in": "A3", "unet_opt": "E3_s123", "segresnet": "E2_s123"}
    for z in R3.glob("*/abl18m-*/ablation_eval/predictions.zip"):
        if "ctrl" in str(z):
            continue
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                parts = n.split("/")            # variante/sGRAINE/foldK/PID_pred.nii.gz
                if len(parts) == 4 and parts[0] in names:
                    m = names[parts[0]]
                    if m.startswith(("E3", "E2")) and parts[1] != "s123":
                        continue
                    src[(m, int(parts[2][4:]))] = (z, f"{parts[0]}/{parts[1]}/{parts[2]}/")
    return src


def load(z, name):
    return np.rint(np.asanyarray(nib.Nifti1Image.from_bytes(gzip.decompress(z.read(name))).dataobj)).astype(np.uint8)


def n_cc(m):
    lab, n = ndi.label(m, structure=STRUCT26)
    if n <= 1:
        return n, 0.0
    # tailles des composantes sans copie int64 du volume (meme resultat que np.bincount(lab.ravel())[1:])
    sizes = np.asarray(ndi.sum_labels(np.ones(lab.shape, dtype=np.uint8), lab, index=np.arange(1, n + 1)))
    return n, float(1 - sizes.max() / sizes.sum())


def work(args):
    pid, fold, src = args
    G = np.rint(np.asanyarray(nib.load(str(ROOT / "data" / pid / "label.nii.gz")).dataobj)).astype(np.uint8)
    rows = []
    zips = {}
    for (m, k), (zpath, prefix) in src.items():
        if k != fold:
            continue
        zf = zips.setdefault(zpath, zipfile.ZipFile(zpath))
        P = load(zf, f"{prefix}{pid}_pred.nii.gz")
        assert P.shape == G.shape, (pid, m)
        gl, gr = G == 1, G == 2
        swap_g = float((P[gl] == 2).mean()) if gl.any() else np.nan
        swap_d = float((P[gr] == 1).mean()) if gr.any() else np.nan
        for c, organ in enumerate(ORGANS, start=1):
            n, frac = n_cc(P == c)
            rows.append({"patient_id": pid, "fold": fold, "model": m, "organ": organ, "n_cc": n, "frac_out_lcc": frac,
                         "swap_G": swap_g, "swap_D": swap_d})
    for zf in zips.values():
        zf.close()
    return rows


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    src = sources()
    models = sorted({m for m, _ in src})
    assert all((m, k) in src for m in models for k in range(5)), "archive manquante"
    print("modeles :", models, flush=True)
    c = pd.read_csv(P2 / "analyse/combined_long.csv")
    pat = c.drop_duplicates("patient_id")[["patient_id", "fold"]].sort_values("patient_id").values.tolist()
    t0, out = time.time(), []
    with Pool(nproc) as pool:
        for i, rows in enumerate(pool.imap_unordered(work, [(p, int(f), src) for p, f in pat]), 1):
            out += rows
            if i % 20 == 0 or i == len(pat):
                print(f"  {i}/{len(pat)} patients ({time.time() - t0:.0f} s)", flush=True)
    R = pd.DataFrame(out)
    R.to_csv(OUT / "ablation_masks.csv", index=False)
    print("lignes :", len(R), "| modeles :", R.model.nunique())
