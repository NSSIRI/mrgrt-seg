"""Metriques de segmentation : DSC, IoU, HD95, Surface DSC, ASSD."""
from __future__ import annotations
from typing import Sequence

import numpy as np
from monai.metrics import (
    DiceMetric, MeanIoU, HausdorffDistanceMetric, SurfaceDiceMetric,
)
from monai.transforms import AsDiscrete, Compose


def build_metrics(num_classes: int, surface_tol_mm: float = 2.0):
    class_thresholds = [surface_tol_mm] * (num_classes - 1)
    return {
        "dsc": DiceMetric(include_background=False, reduction="none"),
        "iou": MeanIoU(include_background=False, reduction="none"),
        "hd95": HausdorffDistanceMetric(include_background=False,
                                        percentile=95, reduction="none"),
        "surface_dsc": SurfaceDiceMetric(class_thresholds=class_thresholds,
                                         include_background=False,
                                         reduction="none"),
    }


def post_processors(num_classes: int):
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = Compose([AsDiscrete(to_onehot=num_classes)])
    return post_pred, post_label


def wilcoxon_signed_rank(scores_a, scores_b) -> dict:
    """Wilcoxon signed-rank test apparie + effect size r = |Z| / sqrt(N).

    Cohen (1988) : r = 0.10 small, 0.30 medium, 0.50 large.
    Cible publication MIA / Med Phys : r >= 0.5 souvent attendu pour des
    differences "cliniquement significatives".
    """
    from scipy.stats import wilcoxon, norm
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = a - b
    n_pairs = int((diff != 0).sum())
    if n_pairs == 0:
        return {"statistic": 0.0, "pvalue": 1.0, "mean_diff": 0.0,
                "median_diff": 0.0, "effect_size_r": 0.0, "n_pairs": 0}
    stat, p = wilcoxon(a, b, alternative="two-sided")
    # Effect size r = |Z| / sqrt(N) ; Z reconstruit a partir de p bilateral
    # (approximation normale, valide pour N >= ~15-20).
    p_one = max(p / 2.0, 1e-300)
    z = abs(norm.isf(p_one))  # |Z|
    r = float(z / np.sqrt(n_pairs)) if n_pairs > 0 else 0.0
    return {"statistic": float(stat), "pvalue": float(p),
            "mean_diff": float(diff.mean()),
            "median_diff": float(np.median(diff)),
            "effect_size_r": r,
            "n_pairs": n_pairs}


def bonferroni(pvals, n_tests=None):
    n = n_tests if n_tests is not None else len(pvals)
    return [min(p * n, 1.0) for p in pvals]
