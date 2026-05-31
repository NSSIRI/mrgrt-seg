"""Sanity check d'Adebayo et al. (NeurIPS 2018) pour cartes de saillance.

Le test principal est la "cascading weight randomization" :
On randomise progressivement les couches du modele, du decoder (couches finales)
vers l'encoder (couches initiales), et on mesure la similarite (SSIM) entre
la heatmap originale et la nouvelle a chaque etape.

Une methode de saillance fiable doit produire une chute monotone de SSIM :
la heatmap doit devenir progressivement aleatoire au fur et a mesure que les
couches utiles sont detruites.

References :
  Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B.
  Sanity Checks for Saliency Maps. NeurIPS 2018.
"""
from __future__ import annotations
import copy
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def list_conv_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Liste les couches Conv3d du modele, dans l'ordre des modules (top-down).

    Pour le cascading, on veut randomiser de la fin vers le debut.
    """
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv3d):
            layers.append((name, mod))
    return layers


def randomize_layer(layer: nn.Module, generator: torch.Generator | None = None) -> None:
    """Re-initialise les poids d'une couche Conv3d in-place (Kaiming uniform).

    Reset bias a 0.
    """
    if not isinstance(layer, nn.Conv3d):
        return
    nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_in", nonlinearity="relu", generator=generator)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def ssim_3d(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM simplifie pour comparer 2 heatmaps 3D.

    Implementation minimaliste basee sur la formule standard SSIM :
      SSIM = (2*mu_a*mu_b + C1)(2*sigma_ab + C2) /
             ((mu_a^2 + mu_b^2 + C1)(sigma_a^2 + sigma_b^2 + C2))
    Avec C1 = (0.01 * dynamic_range)^2, C2 = (0.03 * dynamic_range)^2.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    L = max(a.max() - a.min(), b.max() - b.min(), 1e-12)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2)
    return float(num / den) if den > 1e-12 else 0.0


def cascading_weight_randomization(
    model: nn.Module,
    compute_saliency: Callable[[nn.Module], np.ndarray],
    seed: int = 42,
) -> List[dict]:
    """Lance la cascading weight randomization d'Adebayo et al.

    Parameters
    ----------
    model : modele original, deja entraine
    compute_saliency : callable qui recoit un modele et retourne une heatmap 3D
                       (avec la meme entree fixe pour tous les appels)
    seed : pour la reproducibilite des re-initialisations

    Returns
    -------
    list[dict] : 1 dict par etape, avec :
      - step : index de l'etape (0 = pristine, 1 = derniere couche randomisee, ...)
      - layer_randomized : nom de la couche qu'on vient de re-initialiser
      - ssim_vs_pristine : SSIM entre la heatmap de cette etape et la heatmap originale
    """
    model = copy.deepcopy(model)  # ne pas modifier l'original
    model.eval()

    # Saliency pristine (etape 0)
    pristine = compute_saliency(model)
    results = [{"step": 0, "layer_randomized": "<pristine>", "ssim_vs_pristine": 1.0}]

    # Layers du dernier au premier (top-down -> reverse)
    layers = list_conv_layers(model)[::-1]
    gen = torch.Generator(device="cpu").manual_seed(seed)

    for i, (name, layer) in enumerate(layers, start=1):
        randomize_layer(layer, generator=gen)
        try:
            sal = compute_saliency(model)
            s = ssim_3d(pristine, sal)
        except Exception as e:
            s = float("nan")
        results.append({"step": i, "layer_randomized": name, "ssim_vs_pristine": s})

    return results
