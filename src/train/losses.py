"""Pertes Dice + Cross-Entropy et Focal Tversky alternative."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss


def build_loss(cfg: dict):
    name = cfg.get("name", "dice_ce").lower()
    if name == "dice_ce":
        return DiceCELoss(
            include_background=cfg.get("include_background", False),
            to_onehot_y=True, softmax=True,
            lambda_dice=cfg.get("dice_weight", 1.0),
            lambda_ce=cfg.get("ce_weight", 1.0),
        )
    elif name == "focal_tversky":
        return FocalTverskyLoss(
            alpha=cfg.get("alpha", 0.7), beta=cfg.get("beta", 0.3),
            gamma=cfg.get("gamma", 0.75),
            include_background=cfg.get("include_background", False),
        )
    else:
        raise ValueError(f"Loss inconnue : {name}")


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75,
                 smooth=1.0, include_background=False):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, target):
        n_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target.squeeze(1).long(), num_classes=n_classes)
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()
        if not self.include_background:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]
        dims = (0, 2, 3, 4)
        tp = (probs * target_oh).sum(dims)
        fn = ((1 - probs) * target_oh).sum(dims)
        fp = (probs * (1 - target_oh)).sum(dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1 - tversky).pow(self.gamma).mean()
