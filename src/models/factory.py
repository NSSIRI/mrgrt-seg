"""Factory : construction des architectures U-Net et SegResNet (ResUNet).

L'objectif est d'aligner au maximum les hyperparamètres structurels (nombre de
niveaux, nombre de canaux par niveau) pour que la comparaison U-Net vs ResUNet
soit équitable.
"""
from __future__ import annotations
import torch.nn as nn
from monai.networks.nets import UNet, SegResNet


def _norm_arg(norm):
    """Convertit une norme lue en YAML (liste) en argument MONAI (tuple).

    ["GROUP", {"num_groups": 8}] -> ("GROUP", {"num_groups": 8}) ; une chaine reste inchangee.
    """
    if isinstance(norm, list):
        return tuple(norm)
    return norm


def build_model(name: str, in_channels: int, out_channels: int,
                features=(32, 64, 128, 256, 512),
                segresnet_kwargs: dict | None = None,
                unet_kwargs: dict | None = None):
    """Construit le modèle 3D demandé.

    unet_kwargs (optionnel, ablation phase 3) : num_res_units, norm, act, dropout,
    out_head. Sans unet_kwargs, le U-Net est strictement celui des resultats E1/E3
    (plain, InstanceNorm, PReLU). out_head=k : le U-Net sort k canaux, suivis d'une
    convolution 1x1x1 (k -> out_channels) ; utile quand la norme de la couche de
    sortie impose un nombre de canaux divisible (GroupNorm(8) avec num_res_units>0).
    """
    name = name.lower()
    if name == "unet":
        uk = dict(unet_kwargs or {})
        out_head = uk.pop("out_head", None)
        unknown = set(uk) - {"num_res_units", "norm", "act", "dropout"}
        if unknown:
            raise ValueError(f"unet_kwargs inconnus : {sorted(unknown)}")
        kw = dict(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_head or out_channels,
            channels=tuple(features),
            strides=(2, 2, 2, 2),
            num_res_units=0,
            norm="instance",
        )
        for k, v in uk.items():
            kw[k] = _norm_arg(v) if k in ("norm", "act") else v
        net = UNet(**kw)
        if out_head:
            return nn.Sequential(net, nn.Conv3d(out_head, out_channels, kernel_size=1))
        return net
    elif name == "segresnet":
        # IMPORTANT : pour GroupNorm dans MONAI il faut passer un tuple
        # (norm_name, params_dict). Avec une simple string "group", MONAI
        # appelle GroupNorm() sans num_groups -> TypeError.
        # On utilise GroupNorm avec 8 groupes (default MONAI SegResNet).
        kw = dict(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.2,
            norm=("GROUP", {"num_groups": 8}),
        )
        if segresnet_kwargs:
            kw.update(segresnet_kwargs)
        if "norm" in kw:
            kw["norm"] = _norm_arg(kw["norm"])
        return SegResNet(**kw)
    else:
        raise ValueError(f"Modele inconnu : {name}. Choix: 'unet' | 'segresnet'.")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
