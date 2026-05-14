"""Smoke test - valide l'environnement local sans donnees reelles.

Ce que ce script teste :
1. Imports principaux (torch, monai, nibabel, numpy, scipy, sklearn, pandas, yaml)
2. Construction U-Net 3D et SegResNet 3D
3. Forward pass sur tenseur aleatoire 1x1x32x32x16 (taille reduite pour CPU)
4. Calcul de la loss DiceCELoss
5. Backward pass (calcul du gradient)
6. Test des transforms MONAI sur un tenseur synthetique
7. Test de la metrique DiceMetric

Si tout passe : votre environnement est pret, vous pouvez basculer sur MARWAN
pour les vrais entrainements.

Note : sur CPU, le forward pass d'un patch reel (128x128x64) prendrait des minutes.
On utilise donc une taille reduite (32x32x16) pour valider la mecanique en quelques
secondes.
"""
from __future__ import annotations
import sys
import time
import platform
import traceback


def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    section("1. Environnement Python")
    print(f"Python    : {sys.version.split()[0]}")
    print(f"OS        : {platform.system()} {platform.release()}")
    print(f"Machine   : {platform.machine()}")

    # ---- 2. Imports ----
    section("2. Imports des bibliotheques")
    libs = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("monai", "monai"),
        ("nibabel", "nibabel"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
        ("yaml", "pyyaml"),
        ("matplotlib", "matplotlib"),
        ("SimpleITK", "SimpleITK"),
    ]
    failures = []
    for mod_name, pkg_name in libs:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  [OK]  {pkg_name:18s} {ver}")
        except Exception as e:
            failures.append((pkg_name, str(e)))
            print(f"  [ERR] {pkg_name:18s} {e}")
    if failures:
        print()
        print(f"ECHEC : {len(failures)} bibliotheque(s) manquante(s).")
        for pkg, err in failures:
            print(f"  -> Installer : pip install {pkg}")
        return 1

    import torch
    import numpy as np

    # ---- 3. Verification GPU/CPU ----
    section("3. GPU disponible ?")
    if torch.cuda.is_available():
        print(f"  GPU CUDA detecte : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go")
        device = "cuda"
    else:
        print("  Pas de GPU CUDA. On tourne en CPU (lent mais OK pour smoke test).")
        device = "cpu"
    print(f"  PyTorch : {torch.__version__}")

    # ---- 4. Construction des modeles ----
    section("4. Construction U-Net 3D et SegResNet 3D")
    try:
        from monai.networks.nets import UNet, SegResNet
        unet = UNet(
            spatial_dims=3, in_channels=1, out_channels=5,
            channels=(16, 32, 64, 128),  # reduit pour CPU
            strides=(2, 2, 2),
            num_res_units=0, norm="instance",
        ).to(device)
        n_unet = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        print(f"  [OK] U-Net      : {n_unet:,} parametres")

        segres = SegResNet(
            spatial_dims=3, in_channels=1, out_channels=5,
            init_filters=16, blocks_down=(1, 2, 2),
            blocks_up=(1, 1), dropout_prob=0.1,
            norm=("GROUP", {"num_groups": 8}),  # MONAI exige num_groups explicite
        ).to(device)
        n_seg = sum(p.numel() for p in segres.parameters() if p.requires_grad)
        print(f"  [OK] SegResNet  : {n_seg:,} parametres")
    except Exception as e:
        print(f"  [ERR] Construction modele : {e}")
        traceback.print_exc()
        return 1

    # ---- 5. Forward pass ----
    section("5. Forward pass sur tenseur aleatoire (1, 1, 32, 32, 16)")
    try:
        x = torch.randn(1, 1, 32, 32, 16, device=device)
        t0 = time.time()
        with torch.no_grad():
            y_unet = unet(x)
            y_seg  = segres(x)
        dt = time.time() - t0
        print(f"  [OK] Sortie U-Net      : {tuple(y_unet.shape)}  (attendu (1,5,32,32,16))")
        print(f"  [OK] Sortie SegResNet  : {tuple(y_seg.shape)}")
        print(f"  Temps total           : {dt*1000:.0f} ms")
    except Exception as e:
        print(f"  [ERR] Forward pass : {e}")
        traceback.print_exc()
        return 1

    # ---- 6. Loss + backward ----
    section("6. Loss DiceCE + backward pass")
    try:
        from monai.losses import DiceCELoss
        loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        target = torch.randint(0, 5, (1, 1, 32, 32, 16), device=device).long()
        x.requires_grad_(False)
        unet.train()
        logits = unet(x)
        loss = loss_fn(logits, target)
        loss.backward()
        # verifier qu'au moins un parametre a un gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in unet.parameters())
        print(f"  [OK] DiceCELoss        : {loss.item():.4f}")
        print(f"  [OK] Gradient calcule  : {has_grad}")
    except Exception as e:
        print(f"  [ERR] Loss/backward : {e}")
        traceback.print_exc()
        return 1

    # ---- 7. Transforms MONAI ----
    section("7. Transforms MONAI (sur volume synthetique)")
    try:
        import nibabel as nib
        import tempfile, os
        from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
                                      Orientationd, Spacingd, ScaleIntensityRangePercentilesd,
                                      ToTensord)
        # Cree un NIfTI synthetique en memoire
        with tempfile.TemporaryDirectory() as td:
            arr_img = (np.random.rand(64, 64, 32) * 1000).astype(np.float32)
            arr_lbl = (np.random.randint(0, 5, (64, 64, 32))).astype(np.int16)
            img_path = os.path.join(td, "image.nii.gz")
            lbl_path = os.path.join(td, "label.nii.gz")
            nib.save(nib.Nifti1Image(arr_img, np.eye(4)), img_path)
            nib.save(nib.Nifti1Image(arr_lbl, np.eye(4)), lbl_path)
            tf = Compose([
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 3.0),
                         mode=("bilinear", "nearest")),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.5, upper=99.5,
                    b_min=0.0, b_max=1.0, clip=True),
                ToTensord(keys=["image", "label"]),
            ])
            out = tf({"image": img_path, "label": lbl_path})
            print(f"  [OK] Image apres transform  : shape={tuple(out['image'].shape)}")
            print(f"       intensite [{out['image'].min():.3f}, {out['image'].max():.3f}]")
            print(f"  [OK] Label apres transform  : shape={tuple(out['label'].shape)}")
            print(f"       classes uniques={sorted(set(out['label'].flatten().tolist()))}")
    except Exception as e:
        print(f"  [ERR] Transforms : {e}")
        traceback.print_exc()
        return 1

    # ---- 8. Metriques ----
    section("8. Metrique DiceMetric")
    try:
        from monai.metrics import DiceMetric
        from monai.transforms import AsDiscrete
        metric = DiceMetric(include_background=False, reduction="mean")
        post_pred = AsDiscrete(argmax=True, to_onehot=5)
        post_lbl  = AsDiscrete(to_onehot=5)
        # On utilise les sorties precedentes
        with torch.no_grad():
            y_pred = unet(x).cpu()
        y_pred_oh = post_pred(y_pred[0]).unsqueeze(0)
        y_lbl_oh  = post_lbl(target[0].cpu()).unsqueeze(0)
        metric(y_pred=[y_pred_oh[0]], y=[y_lbl_oh[0]])
        dsc = metric.aggregate().item()
        print(f"  [OK] DSC (random vs random) : {dsc:.4f}  (proche de 0 attendu)")
    except Exception as e:
        print(f"  [ERR] DiceMetric : {e}")
        traceback.print_exc()
        return 1

    # ---- Outils data prep ----
    section("9. Outils data prep (DICOM/RT)")
    for mod_name, pkg_name in [("pydicom", "pydicom"), ("rt_utils", "rt-utils")]:
        try:
            __import__(mod_name)
            print(f"  [OK] {pkg_name}")
        except ImportError:
            print(f"  [WARN] {pkg_name} non installe (pour conversion DICOM)")

    section("RESULTAT GLOBAL")
    print()
    print("  >>> SMOKE TEST REUSSI <<<")
    print()
    print("  Votre environnement local est valide. Vous pouvez :")
    print("   - Lancer le notebook 00_pedagogical_pipeline.ipynb")
    print("   - Ouvrir le code dans VS Code, le parcourir, deboguer")
    print("   - Pousser le code sur git, puis basculer sur MARWAN pour entrainer")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
