# -*- coding: utf-8 -*-
"""Phase 4 : metriques 6 OAR limitees a l'etendue cranio-caudale des poumons (region thoracique).

Pourquoi : les labels moelle (5) et aorte (6) de data_6oar/ reprennent toute l'etendue TotalSegmentator ; chez
beaucoup de patients ils descendent dans l'abdomen (et parfois le cou). Les metriques publiees pour ces organes
n'etaient donc pas strictement thoraciques.

Methode (grille native du label, comme phase4_stats.py) :
  - region thoracique = coupes (axe le plus proche de S) contenant du poumon G ou D dans la reference ;
  - recouvrement (DSC, IoU, precision, sensibilite, volumes) : voxels de cette region seulement ;
  - surfaces (HD95, HDmax, ASSD, Surface-DSC 2 mm) : bords des masques ENTIERS (pas de face de coupe
    artificielle), on ne garde que les points de bord situes dans la region et on mesure leur distance au bord
    entier de l'autre masque ; meme definition que eval_ablation.surface_stats sinon.
Fonctions de recouvrement et de bord reprises a l'identique de phase3_ablation/eval_ablation.py (extraction ast).
Controle : sans restriction, le recalcul doit redonner metrics_long.csv (ecart max < 1e-6).

Lecture seule : data_6oar/, results/phase3_ablation/*/abl18m-*6-s42-f*/ablation_eval/.
Ecrit phase4_6oar/thorax/ : metrics_thorax_long.csv, controle_sans_restriction.csv.

    .venv\\Scripts\\python.exe phase4_6oar/thorax/metrics_thorax.py [n_processus]
"""
import ast
import glob
import io
import re
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABELS = ROOT / "data_6oar"
RUNS = sorted(glob.glob(str(ROOT / "results/phase3_ablation/*/abl18m-*6-s42-f*/ablation_eval")))
ORGANS = ["poumon_g", "poumon_d", "coeur", "oesophage", "moelle", "aorte"]
SURF_TOL_MM = 2.0


def _load_eval_functions():
    src = (ROOT / "phase3_ablation/eval_ablation.py").read_text(encoding="utf-8")
    keep = [n for n in ast.parse(src).body
            if (isinstance(n, ast.FunctionDef) and n.name in {"_bbox", "surface_stats", "overlap_stats", "grid_of"})
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "STRUCT" for t in n.targets))]
    ns = {"np": np, "ndi": ndi, "cKDTree": cKDTree}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "eval_ablation.py", "exec"), ns)
    return ns


EV = _load_eval_functions()
STRUCT = EV["STRUCT"]


def cc_axis(affine):
    """Axe de voxel le plus proche de l'axe S (cranio-caudal) du monde."""
    a = np.asarray(affine, float)[:3, :3]
    return int(np.argmax(np.abs(a[2]) / np.sqrt((a ** 2).sum(0))))


def surface_restricted(p, g, spacing, tol, keep):
    """Comme eval_ablation.surface_stats, mais seuls les points de bord ou keep est vrai sont mesures
    (keep=None : aucune restriction -> doit redonner surface_stats)."""
    out = {"hd95": np.nan, "hdmax": np.nan, "assd": np.nan, "sdsc": np.nan}
    P, G = bool(p.any()), bool(g.any())
    if not P and not G:
        return out
    if P != G:
        out["sdsc"] = 0.0
        return out
    sl = EV["_bbox"](p | g)
    pp, gg = np.pad(p[sl], 1), np.pad(g[sl], 1)
    ep = pp & ~ndi.binary_erosion(pp, structure=STRUCT)
    eg = gg & ~ndi.binary_erosion(gg, structure=STRUCT)
    sp = np.asarray(spacing, dtype=np.float64)
    pts_p, pts_g = np.argwhere(ep), np.argwhere(eg)
    if keep is not None:
        kk = np.pad(keep[sl], 1)
        qp, qg = pts_p[kk[tuple(pts_p.T)]], pts_g[kk[tuple(pts_g.T)]]
    else:
        qp, qg = pts_p, pts_g
    if len(qp) == 0 or len(qg) == 0:      # un des deux bords absent de la region
        out["sdsc"] = 0.0 if (len(qp) + len(qg)) else np.nan
        return out
    d_pg = cKDTree(pts_g * sp).query(qp * sp, workers=1)[0]
    d_gp = cKDTree(pts_p * sp).query(qg * sp, workers=1)[0]
    both = np.concatenate([d_pg, d_gp])
    out["hd95"] = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    out["hdmax"] = float(max(d_pg.max(), d_gp.max()))
    out["assd"] = float(both.mean())
    out["sdsc"] = float(((d_pg <= tol).sum() + (d_gp <= tol).sum()) / both.size)
    return out


def one_patient(args):
    run_dir, member, variant, fold = args
    pid = re.search(r"(PATIENT_s\d+)_pred", member).group(1)
    with zipfile.ZipFile(Path(run_dir) / "predictions.zip") as z:
        data = z.read(member)
    pn = nib.Nifti1Image.from_bytes(__import__("gzip").decompress(data))
    gn = nib.load(str(LABELS / pid / "label.nii.gz"))
    pred = np.asarray(pn.dataobj).astype(np.uint8)
    gt = np.rint(np.asanyarray(gn.dataobj)).astype(np.uint8)
    aff = np.asarray(gn.affine, float)
    if pred.shape != gt.shape or not np.allclose(pn.affine, aff, atol=1e-3):
        raise RuntimeError(f"{pid} : grille prediction != label")
    sp, vox = EV["grid_of"](aff)
    ax = cc_axis(aff)
    lung_sl = np.flatnonzero(np.any((gt == 1) | (gt == 2), axis=tuple(i for i in range(3) if i != ax)))
    keep = np.zeros(gt.shape, bool)
    idx = [slice(None)] * 3
    idx[ax] = slice(int(lung_sl.min()), int(lung_sl.max()) + 1)
    keep[tuple(idx)] = True
    rows = []
    for c, organ in enumerate(ORGANS, start=1):
        pc, gc = pred == c, gt == c
        base = {"variant": variant, "fold": fold, "patient_id": pid, "organ": organ,
                "n_coupes_poumon": len(lung_sl), "axe_cc": ax,
                "frac_gt_dans_thorax": float((gc & keep).sum() / gc.sum()) if gc.any() else np.nan,
                "frac_pred_dans_thorax": float((pc & keep).sum() / pc.sum()) if pc.any() else np.nan}
        for region, kp in (("complet", None), ("thorax", keep)):
            r = dict(base, region=region)
            r.update(EV["overlap_stats"](pc if kp is None else pc & kp, gc if kp is None else gc & kp, vox))
            s = surface_restricted(pc, gc, sp, SURF_TOL_MM, kp)
            r.update({"hd95_mm": s["hd95"], "hdmax_mm": s["hdmax"], "assd_mm": s["assd"], "sdsc_2mm": s["sdsc"]})
            if kp is None:   # controle de la reimplementation des surfaces
                ref = EV["surface_stats"](pc, gc, sp, SURF_TOL_MM)
                r["ecart_surface_vs_eval"] = float(np.nanmax(np.abs(np.array(
                    [s[k] - ref[k] for k in ("hd95", "hdmax", "assd", "sdsc")], float)))) \
                    if any(np.isfinite([ref[k] for k in ref])) else 0.0
            rows.append(r)
    return rows


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tasks = []
    for rd in RUNS:
        ml = pd.read_csv(Path(rd) / "metrics_long.csv", usecols=["variant", "fold"]).iloc[0]
        with zipfile.ZipFile(Path(rd) / "predictions.zip") as z:
            members = [m for m in z.namelist() if m.endswith("_pred.nii.gz")]
        tasks += [(rd, m, ml.variant, int(ml.fold)) for m in members]
    print(f"{len(RUNS)} runs, {len(tasks)} predictions", flush=True)
    rows = []
    with Pool(nproc) as pool:
        for i, r in enumerate(pool.imap_unordered(one_patient, tasks, chunksize=2), 1):
            rows += r
            if i % 25 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)
    df = pd.DataFrame(rows).sort_values(["variant", "fold", "patient_id", "organ", "region"])
    df.to_csv(HERE / "metrics_thorax_long.csv", index=False)

    # controle : region "complet" == metrics_long.csv (grille native)
    ref = pd.concat(pd.read_csv(Path(rd) / "metrics_long.csv") for rd in RUNS)
    ref = ref[ref.space == "native"]
    m = df[df.region == "complet"].merge(ref, on=["variant", "fold", "patient_id", "organ"], suffixes=("", "_ref"))
    cols = ["dsc", "iou", "precision", "recall", "vol_gt_ml", "vol_pred_ml", "hd95_mm", "hdmax_mm", "assd_mm", "sdsc_2mm"]
    ctl = pd.DataFrame({c: [np.nanmax(np.abs(m[c] - m[c + "_ref"]))] for c in cols})
    ctl["n_lignes"] = len(m)
    ctl["ecart_surface_reimpl_max"] = df.ecart_surface_vs_eval.max()
    ctl.to_csv(HERE / "controle_sans_restriction.csv", index=False)
    print(ctl.T.to_string())


if __name__ == "__main__":
    main()
