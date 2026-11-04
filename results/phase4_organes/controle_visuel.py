"""Controle visuel : patients D187 a faible couverture de la moelle ou de l'aorte.

Coupe coronale passant par le centre des poumons (moelle : coupe sagittale mediane) avec les contours
des poumons, de la moelle et de l'aorte (masques bruts TotalSegmentator). Lecture seule.

    .venv\\Scripts\\python.exe results/phase4_organes/controle_visuel.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "TotalsegmentatorMRI_dataset_v200"
OUT = Path(__file__).resolve().parent
COL = {"lungs": "#2a78d6", "spinal_cord": "#eda100", "aorta": "#eb6834"}


def canon(p):
    return nib.as_closest_canonical(nib.load(str(p)))


def main():
    df = pd.read_csv(OUT / "comptage_moelle_aorte_par_examen.csv")
    d = df[df.cohorte_D187]
    low = d[(d.couv_spinal_cord < 0.95) | (d.couv75_aorta < 0.95)].copy()
    low["score"] = low[["couv_spinal_cord", "couv75_aorta"]].min(axis=1)
    pick = low.sort_values("score").iloc[np.linspace(0, len(low) - 1, 12).astype(int)]
    fig, axs = plt.subplots(3, 4, figsize=(12, 10))
    for ax, (_, r) in zip(axs.flat, pick.iterrows()):
        sid = r.examen
        img = np.asanyarray(canon(RAW / sid / "mri.nii.gz").dataobj).astype(np.float32)
        seg = {k: np.asanyarray(canon(RAW / sid / "segmentations" / f"{k}.nii.gz").dataobj) > 0
               for k in ("lung_left", "lung_right", "spinal_cord", "aorta")}
        lungs = seg["lung_left"] | seg["lung_right"]
        # coupe sagittale par la moelle si presente, sinon par le milieu des poumons
        ref = seg["spinal_cord"] if seg["spinal_cord"].any() else lungs
        x = int(np.round(np.argwhere(ref)[:, 0].mean()))
        sl = lambda a: np.rot90(a[x])  # noqa: E731   (A-P horizontal, S en haut)
        lo, hi = np.percentile(img[x], [1, 99])
        ax.imshow(sl(img), cmap="gray", vmin=lo, vmax=hi, aspect="auto")
        for k, m, c in [("poumons", lungs, COL["lungs"]), ("moelle", seg["spinal_cord"], COL["spinal_cord"]),
                        ("aorte", seg["aorta"], COL["aorta"])]:
            if m[x].any():
                ax.contour(sl(m).astype(float), levels=[0.5], colors=[c], linewidths=1.2)
        ax.set_title(f"{sid}  moelle {r.couv_spinal_cord:.2f} | aorte {r.couv75_aorta:.2f}".replace(".", ","),
                     fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    hs = [plt.Line2D([], [], color=c, lw=2) for c in COL.values()]
    fig.legend(hs, ["poumons (coupe sagittale)", "moelle", "aorte"], loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "controle_visuel_faible_couverture.png", dpi=150)
    pick[["examen", "couv_spinal_cord", "couv75_aorta", "coupes_poumons"]].to_csv(
        OUT / "controle_visuel_cas.csv", index=False)
    print(f"{len(low)} patients D187 sous 0,95 ; 12 montres")


if __name__ == "__main__":
    main()
