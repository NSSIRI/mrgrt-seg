"""Dataset NIfTI + transforms MONAI pour segmentation OAR thorax MRgRT et CT.

Le pipeline est pilote par un flag `modality` ("mri" ou "ct").
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRangePercentilesd, ScaleIntensityRanged, CropForegroundd,
    SpatialPadd, RandCropByPosNegLabeld, RandRotate90d, RandShiftIntensityd,
    RandAdjustContrastd, RandGaussianNoised, RandBiasFieldd, RandAffined,
    EnsureTyped, ToTensord,
)


def _intensity_transform(modality: str, params: dict):
    modality = modality.lower()
    if modality == "mri":
        clip = params.get("clip_percentiles", [0.5, 99.5])
        return ScaleIntensityRangePercentilesd(
            keys=["image"], lower=clip[0], upper=clip[1],
            b_min=0.0, b_max=1.0, clip=True,
        )
    elif modality == "ct":
        hu = params.get("hu_window", [-1000.0, 400.0])
        return ScaleIntensityRanged(
            keys=["image"], a_min=hu[0], a_max=hu[1],
            b_min=0.0, b_max=1.0, clip=True,
        )
    else:
        raise ValueError(f"modality inconnue : {modality}")


def _modality_specific_aug(modality: str) -> list:
    modality = modality.lower()
    if modality == "mri":
        return [
            RandBiasFieldd(keys=["image"], degree=3, coeff_range=(0.0, 0.1), prob=0.2),
            RandGaussianNoised(keys=["image"], std=0.02, prob=0.2),
        ]
    elif modality == "ct":
        return [
            RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.3),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.2),
        ]
    return []


def list_patients(root, image_filename="image.nii.gz", label_filename="label.nii.gz"):
    root = Path(root)
    items = []
    for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        img = patient_dir / image_filename
        lbl = patient_dir / label_filename
        if img.exists() and lbl.exists():
            items.append({"image": str(img), "label": str(lbl),
                          "patient_id": patient_dir.name})
    if not items:
        raise FileNotFoundError(f"Aucun patient valide trouve dans {root}")
    return items


def get_train_transforms(spacing, intensity_params, patch_size,
                         modality="mri", num_samples=2):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing,
                 mode=("bilinear", "nearest")),
        _intensity_transform(modality, intensity_params),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=patch_size, pos=2, neg=1,
            num_samples=num_samples, image_key="image",
        ),
        RandAffined(keys=["image", "label"], prob=0.3,
                    rotate_range=(0.26, 0.26, 0.26),
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest")),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3, spatial_axes=(0, 1)),
        RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.3),
        *_modality_specific_aug(modality),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms(spacing, intensity_params, modality="mri"):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing,
                 mode=("bilinear", "nearest")),
        _intensity_transform(modality, intensity_params),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])


def make_loaders(train_items, val_items, transforms_train, transforms_val,
                 batch_size=2, num_workers=4, cache_rate=0.5):
    train_ds = CacheDataset(data=train_items, transform=transforms_train,
                            cache_rate=cache_rate, num_workers=num_workers)
    val_ds = CacheDataset(data=val_items, transform=transforms_val,
                          cache_rate=cache_rate, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
