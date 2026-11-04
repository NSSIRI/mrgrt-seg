"""Tests CPU de l'etape 1 (branche ablation-18m). Sans pytest :

    .venv\\Scripts\\python.exe phase3_ablation/tests/test_ablation.py

Verifie : parametres E2/E3 inchanges (configs historiques ET copies d'ablation),
parametres de chaque variante (tableau §2), passe avant 1x1x64x64x32 -> 1x5x64x64x32,
plis identiques a splits.json pour les graines 42 / 123 / 2024, run_meta.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.models.factory import build_model, count_parameters  # noqa: E402
from src.data.splits import make_5fold_splits  # noqa: E402
import train  # noqa: E402  (scripts/train.py)

SPLITS_REF = ROOT / "results/phase2_eval/abdelhalimnssiri_b/phase2_eval/splits.json"

# (config, --model, parametres attendus) ; valeurs : phase3_ablation/variants_params.csv
CASES = [
    ("configs/unet_fair.yaml", "unet", 18_802_973),            # E3 historique
    ("configs/default.yaml", "segresnet", 18_796_101),         # E2 historique
    ("configs/ablation/e3_unet_opt.yaml", "unet", 18_802_973),
    ("configs/ablation/e2_segresnet.yaml", "segresnet", 18_796_101),
    ("configs/ablation/unet_res.yaml", "unet", 18_918_214),           # A1
    ("configs/ablation/unet_gn.yaml", "unet", 18_806_749),            # A2
    ("configs/ablation/segresnet_in.yaml", "segresnet", 18_789_381),  # A3
    ("configs/ablation/segresnet_nodrop.yaml", "segresnet", 18_796_101),  # A4
    ("configs/ablation/unet_res_gn.yaml", "unet", 18_929_350),        # A5
]
ABLATION_CFGS = [c for c, _, _ in CASES if c.startswith("configs/ablation/")]


def load(cfg_path):
    return yaml.safe_load((ROOT / cfg_path).read_text(encoding="utf-8"))


def build_from_cfg(cfg, model):
    # meme appel que scripts/train.py
    return build_model(name=model, in_channels=cfg["model"]["in_channels"],
                       out_channels=cfg["model"]["out_channels"],
                       features=cfg["model"]["features"],
                       segresnet_kwargs=cfg["model"].get("segresnet"),
                       unet_kwargs=cfg["model"].get("unet"))


def test_params_and_forward():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 64, 32)
    for cfg_path, model, expected in CASES:
        m = build_from_cfg(load(cfg_path), model).eval()
        n = count_parameters(m)
        assert n == expected, f"{cfg_path}: {n} != {expected}"
        assert abs(n - 18_800_000) / 18_800_000 < 0.01, f"{cfg_path}: hors ±1 % de 18,8 M"
        with torch.no_grad():
            shape = tuple(m(x).shape)
        assert shape == (1, 5, 64, 64, 32), f"{cfg_path}: sortie {shape}"
        print(f"  ok {cfg_path:<42} {n:>11,}  sortie {shape}")


def test_default_unet_unchanged():
    # sans unet_kwargs : meme architecture qu'au commit 05d473e (num_res_units=0, InstanceNorm)
    from monai.networks.nets import UNet
    a = build_model("unet", 1, 5, features=(48, 96, 192, 384, 832))
    b = UNet(spatial_dims=3, in_channels=1, out_channels=5, channels=(48, 96, 192, 384, 832),
             strides=(2, 2, 2, 2), num_res_units=0, norm="instance")
    assert type(a) is UNet and str(a) == str(b)
    try:
        build_model("unet", 1, 5, unet_kwargs={"num_res_unit": 2})
        raise AssertionError("cle inconnue acceptee")
    except ValueError:
        pass
    print("  ok U-Net par defaut identique a 05d473e ; cle inconnue refusee")


def test_structure():
    # A4 a le meme nombre de parametres que E2 : on verifie la structure, pas seulement le compte
    import torch.nn as nn

    def kinds(cfg_path, model):
        m = build_from_cfg(load(cfg_path), model)
        return m, {type(x).__name__ for x in m.modules()}

    m, k = kinds("configs/ablation/e2_segresnet.yaml", "segresnet")
    assert m.dropout_prob == 0.2 and "GroupNorm" in k and "InstanceNorm3d" not in k
    m, k = kinds("configs/ablation/segresnet_nodrop.yaml", "segresnet")
    assert m.dropout_prob is None and not any(isinstance(x, nn.Dropout3d) for x in m.modules())
    m, k = kinds("configs/ablation/segresnet_in.yaml", "segresnet")
    assert "InstanceNorm3d" in k and "GroupNorm" not in k
    m, k = kinds("configs/ablation/unet_gn.yaml", "unet")
    assert "GroupNorm" in k and "InstanceNorm3d" not in k and "PReLU" in k
    m, k = kinds("configs/ablation/unet_res.yaml", "unet")
    assert "ResidualUnit" in k and "InstanceNorm3d" in k
    m, k = kinds("configs/ablation/unet_res_gn.yaml", "unet")
    assert isinstance(m, nn.Sequential) and "ResidualUnit" in k and "GroupNorm" in k
    print("  ok structure : E2 GN+dropout, A4 sans dropout, A3 InstanceNorm, A2 GN, A1/A5 residuels")


def test_splits():
    ref = json.loads(SPLITS_REF.read_text(encoding="utf-8"))
    patients = sorted(p.name for p in (ROOT / "data").iterdir() if p.is_dir())
    assert patients == sorted(pid for f in ref for pid in f["val"]), "data/ != patients de splits.json"
    items = [{"patient_id": p} for p in patients]   # list_patients trie aussi par dossier
    cfgs = ABLATION_CFGS + ["configs/unet_fair.yaml", "configs/default.yaml"]
    for cfg_path in cfgs:
        cfg = load(cfg_path)
        for cli_seed in (None, 42, 123, 2024):
            train_seed, split_seed = train.resolve_seeds(cfg, cli_seed)
            assert train_seed == (cli_seed if cli_seed is not None else 42)
            folds = make_5fold_splits(items, n_folds=cfg["splits"]["n_folds"], seed=split_seed)
            for f, r in zip(folds, ref):
                assert f["val"] == r["val"] and sorted(f["train"]) == sorted(r["train"]), \
                    f"{cfg_path} seed={cli_seed} pli {f['fold']} different"
    print(f"  ok plis identiques a splits.json : {len(cfgs)} configs x graines None/42/123/2024 "
          f"({[len(f['val']) for f in ref]})")


def test_run_meta():
    cfg = load("configs/ablation/unet_res.yaml")
    m = build_from_cfg(cfg, "unet")
    args = types.SimpleNamespace(tag="unet_res_s123", model="unet",
                                 config="configs/ablation/unet_res.yaml", fold=0)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "unet_res_s123_fold0"
        train._write_run_meta(out, args, cfg, 123, 42, [{"patient_id": "PATIENT_s0003"}],
                              count_parameters(m))
        meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["train_seed"] == 123 and meta["split_seed"] == 42
    assert meta["unet_kwargs"] == {"num_res_units": 2} and meta["n_params"] == 18_918_214
    assert meta["val_patients"] == ["PATIENT_s0003"] and meta["monai_version"]
    print(f"  ok run_meta.json (monai {meta['monai_version']}, commit {str(meta['git_commit'])[:7]})")


def test_cli_help():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--help"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "--seed" in r.stdout and "--tag" in r.stdout, r.stderr[-500:]
    print("  ok train.py --help expose --seed et --tag")


if __name__ == "__main__":
    import monai
    print(f"MONAI {monai.__version__} | torch {torch.__version__}")
    if monai.__version__ != "1.5.2":
        print("  ATTENTION : MONAI != 1.5.2, les nombres de parametres peuvent differer")
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("TOUS LES TESTS PASSENT")
