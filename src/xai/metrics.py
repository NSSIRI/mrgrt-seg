"""Metriques quantitatives d'explicabilite pour les cartes SEG-GRAD-CAM 3D.

3 metriques par patient et par OAR :
  - in_organ_ratio   : fraction de la masse de la heatmap qui tombe dans le GT mask
  - pointing_accuracy : 1 si l'argmax voxel de la heatmap est dans le GT mask, 0 sinon
  - spatial_entropy  : entropie de Shannon (normalisee) de la heatmap

References :
  Vondrick et al. ICCV 2013 (pointing game)
  Vinogradova et al. AAAI 2020 (SEG-GRAD-CAM)
  Adebayo et al. NeurIPS 2018 (sanity checks)
"""
from __future__ import annotations
import numpy as np


def in_organ_ratio(heatmap: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-12) -> float:
    """Fraction de la masse totale de la heatmap qui tombe dans le mask GT.

    Parameters
    ----------
    heatmap : 3D array, valeurs >=0 (typiquement L1-normalisee dans [0,1])
    gt_mask : 3D array bool (True ou !=0 dans l'organe cible)

    Returns
    -------
    float dans [0, 1]. 1 = saliency parfaitement localise sur l'organe.
    """
    heat = np.asarray(heatmap, dtype=np.float64)
    gt = np.asarray(gt_mask).astype(bool)
    total = float(heat.sum())
    if total < eps:
        return float("nan")
    inside = float(heat[gt].sum())
    return inside / total


def pointing_accuracy(heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
    """1.0 si argmax(heatmap) tombe dans le GT mask, 0.0 sinon.

    Variante stricte du pointing-game de Vondrick et al. (ICCV 2013),
    adaptee a la segmentation dense.

    Parameters
    ----------
    heatmap : 3D array
    gt_mask : 3D array bool

    Returns
    -------
    0.0 ou 1.0
    """
    heat = np.asarray(heatmap)
    gt = np.asarray(gt_mask).astype(bool)
    if not gt.any():
        return float("nan")  # pas de GT, score indefini
    idx = np.unravel_index(np.argmax(heat), heat.shape)
    return float(gt[idx])


def spatial_entropy(heatmap: np.ndarray, eps: float = 1e-12) -> float:
    """Entropie de Shannon (en nats) de la distribution voxel-wise de la heatmap.

    Definie comme H = -sum p_v log p_v ou p_v = heatmap[v] / sum(heatmap).
    Plus faible = attention concentree (focal).
    Plus eleve = attention diffuse.

    Pour comparer entre patients, normaliser par log(N_voxels) pour avoir une
    valeur dans [0, 1] (1 = uniforme).
    """
    heat = np.asarray(heatmap, dtype=np.float64).ravel()
    total = float(heat.sum())
    if total < eps:
        return float("nan")
    p = heat / total
    # Eviter log(0)
    p_pos = p[p > 0]
    H = float(-(p_pos * np.log(p_pos)).sum())
    # Normaliser par log(N_voxels_non_nuls) pour avoir [0, 1]
    n = int((p > 0).sum())
    if n <= 1:
        return 0.0
    H_max = float(np.log(n))
    return H / H_max if H_max > 0 else float("nan")


def organ_size_baseline(gt_mask: np.ndarray) -> float:
    """Baseline pour in_organ_ratio : ratio du volume de l'organe sur le volume total.

    Une heatmap UNIFORME aurait in_organ_ratio = organ_size_baseline.
    Si in_organ_ratio > baseline, le modele est plus focal que random.
    """
    gt = np.asarray(gt_mask).astype(bool)
    return float(gt.sum() / gt.size) if gt.size > 0 else float("nan")


def compute_all_metrics(heatmap: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Calcule les 4 metriques (3 + baseline) en une fois.

    Returns
    -------
    dict avec cles : in_organ_ratio, pointing_accuracy, spatial_entropy, organ_baseline
    """
    return {
        "in_organ_ratio": in_organ_ratio(heatmap, gt_mask),
        "pointing_accuracy": pointing_accuracy(heatmap, gt_mask),
        "spatial_entropy": spatial_entropy(heatmap),
        "organ_baseline": organ_size_baseline(gt_mask),
    }
