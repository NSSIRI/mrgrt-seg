"""Factory : construction des architectures U-Net et SegResNet (ResUNet).

L'objectif est d'aligner au maximum les hyperparamètres structurels (nombre de
niveaux, nombre de canaux par niveau) pour que la comparaison U-Net vs ResUNet
soit équitable.
"""
from __future__ import annotations
from monai.networks.nets import UNet, SegResNet


def build_model(name: str, in_channels: int, out_channels: int,
                features=(32, 64, 128, 256, 512),
                segresnet_kwargs: dict | None = None):
    """Construit le modèle 3D demandé."""
    name = name.lower()
    if name == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(features),
            strides=(2, 2, 2, 2),
            num_res_units=0,
            norm="instance",
        )
    elif name == "segresnet":
        kw = dict(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.2,
            norm="group",
        )
        if segresnet_kwargs:
            kw.update(segresnet_kwargs)
        return SegResNet(**kw)
    else:
        raise ValueError(f"Modele inconnu : {name}. Choix: 'unet' | 'segresnet'.")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
