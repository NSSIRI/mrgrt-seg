"""SEG-GRAD-CAM 3D pour la segmentation semantique medicale.

Reference : Vinogradova K., Dibrov A., Myers G. (AAAI Workshop 2020).
"""
from __future__ import annotations
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _find_last_conv3d(model: nn.Module) -> nn.Module:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            last = m
    if last is None:
        raise RuntimeError("Aucune couche Conv3d trouvee dans le modele.")
    return last


def _resolve_target_layer(model, target):
    if isinstance(target, nn.Module):
        return target
    if target == "auto":
        return _find_last_conv3d(model)
    for name, mod in model.named_modules():
        if name == target:
            return mod
    raise ValueError(f"Couche cible introuvable : {target}")


class SegGradCAM3D:
    """SEG-GRAD-CAM 3D - carte d'activation par voxel pour une classe donnee."""

    def __init__(self, model: nn.Module, target_layer="auto"):
        self.model = model.eval()
        self.layer = _resolve_target_layer(model, target_layer)
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd_handle = self.layer.register_forward_hook(self._save_act)
        self._bwd_handle = self.layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _module, _inp, output):
        self._activations = output

    def _save_grad(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0]

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    # Alias plus pratique pour le runner XAI
    def cleanup(self):
        self.remove_hooks()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass

    def compute(self, x, target_class, patch_size=None):
        """Alias pour __call__ qui retourne le 1er sample en numpy/tensor 3D.

        Si l'input est un volume entier (B=1, C=1, D, H, W), le tensor cam
        retourne par __call__ a shape (1, D, H, W). On retourne le [0] = (D, H, W).
        """
        cam = self(x, target_class)  # tensor (B, D, H, W) ou (1, D, H, W)
        if cam.dim() == 4 and cam.shape[0] == 1:
            cam = cam[0]
        return cam

    @torch.enable_grad()
    def __call__(self, x, target_class, mask_subset=None, normalize=True):
        x = x.requires_grad_(False).clone()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if mask_subset is None:
            preds = logits.argmax(dim=1, keepdim=True)
            mask_subset = (preds == target_class).float()
        score = (logits[:, target_class:target_class + 1] * mask_subset).sum()
        score.backward(retain_graph=False)
        grads = self._gradients
        acts = self._activations
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear",
                            align_corners=False).squeeze(1)
        if normalize:
            B = cam.shape[0]
            flat = cam.view(B, -1)
            mins = flat.min(dim=1, keepdim=True).values
            maxs = flat.max(dim=1, keepdim=True).values
            cam = (flat - mins) / (maxs - mins + 1e-8)
            if B == 1:
                cam = cam.view(*x.shape[2:]).unsqueeze(0)
            else:
                cam = cam.view(B, *x.shape[2:])
        return cam


def in_organ_ratio(heatmap, gt_mask):
    h = np.asarray(heatmap, dtype=float)
    m = np.asarray(gt_mask, dtype=bool)
    total = h.sum()
    if total <= 0:
        return float("nan")
    return float(h[m].sum() / total)


def pointing_accuracy(heatmap, gt_mask):
    h = np.asarray(heatmap, dtype=float)
    m = np.asarray(gt_mask, dtype=bool)
    if h.size == 0 or not m.any():
        return 0
    idx = np.unravel_index(np.argmax(h), h.shape)
    return int(bool(m[idx]))


def spatial_entropy(heatmap, eps=1e-12):
    h = np.asarray(heatmap, dtype=float).ravel()
    h = np.clip(h, 0, None)
    s = h.sum()
    if s <= 0:
        return float("nan")
    p = h / s
    H = -np.sum(p * np.log(p + eps))
    H_max = np.log(p.size + eps)
    return float(H / H_max) if H_max > 0 else float("nan")


def xai_metrics(heatmap, gt_mask):
    return {
        "in_organ_ratio": in_organ_ratio(heatmap, gt_mask),
        "pointing_accuracy": pointing_accuracy(heatmap, gt_mask),
        "spatial_entropy": spatial_entropy(heatmap),
    }
