"""Tests CPU de la phase 4 (6 OAR, D115), sans pytest :

    .venv\\Scripts\\python.exe phase4_6oar/tests/test_6oar.py

1. labels data_6oar : 115 patients, valeurs 0..6, classes 1-4 identiques aux labels D187, grilles identiques ;
2. configs phase 4 : 7 sorties, nombre de parametres, passe avant ;
3. plis D115 : KFold(5, graine 42) deterministe et disjoint ;
4. chargement MONAI (transformations de validation) d'un patient : 7 classes presentes apres reechantillonnage ;
5. evaluation de bout en bout (eval_ablation.py, P3_ORGANS a 6 OAR) d'un checkpoint initialise au hasard, 1 patient.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train  # noqa: E402
from src.models.factory import build_model, count_parameters  # noqa: E402
from src.data.splits import make_5fold_splits  # noqa: E402
from src.data.dataset import get_val_transforms  # noqa: E402

D6 = ROOT / "data_6oar"
ORGANS6 = "poumon_g,poumon_d,coeur,oesophage,moelle,aorte"
CFGS = {"configs/phase4/segresnet_6oar.yaml": ("segresnet", 18_796_167),
        "configs/phase4/unet_6oar.yaml": ("unet", 18_808_159)}


def patients():
    return sorted(p.name for p in D6.iterdir() if p.is_dir())


def test_labels():
    pids = patients()
    assert len(pids) == 115, len(pids)
    for pid in pids[::23]:                                  # echantillon de 5 patients
        a = nib.load(str(D6 / pid / "label.nii.gz"))
        b = nib.load(str(ROOT / "data" / pid / "label.nii.gz"))
        la, lb = np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)
        assert a.shape == b.shape and np.allclose(a.affine, b.affine)
        assert set(np.unique(la)) == set(range(7)), (pid, np.unique(la))
        assert np.array_equal(np.where(la <= 4, la, 0), lb), f"{pid} : classes 1-4 modifiees"
    print(f"  ok 115 patients ; echantillon : classes 0-6 presentes, classes 1-4 identiques a D187")


def test_configs():
    for f, (m, n) in CFGS.items():
        c = yaml.safe_load((ROOT / f).read_text(encoding="utf-8"))
        assert c["data"]["num_classes"] == 7 == c["model"]["out_channels"] == len(c["data"]["class_names"])
        assert c["splits"]["seed"] == 42 and c["train"]["epochs"] == 150
        net = build_model(m, 1, 7, features=c["model"]["features"], segresnet_kwargs=c["model"].get("segresnet"),
                          unet_kwargs=c["model"].get("unet")).eval()
        assert count_parameters(net) == n, (f, count_parameters(net))
        with torch.no_grad():
            assert tuple(net(torch.randn(1, 1, 64, 64, 32)).shape) == (1, 7, 64, 64, 32)
    print("  ok configs : 7 sorties, 18 796 167 / 18 808 159 parametres, passe avant")


def test_folds():
    items = [{"patient_id": p} for p in patients()]
    a, b = make_5fold_splits(items, seed=42), make_5fold_splits(items, seed=42)
    assert a == b
    val = [p for f in a for p in f["val"]]
    assert len(val) == len(set(val)) == 115 and all(not set(f["train"]) & set(f["val"]) for f in a)
    print(f"  ok plis D115 : {[len(f['val']) for f in a]}")


def test_transform():
    pid = patients()[0]
    tf = get_val_transforms(spacing=[1.5, 1.5, 3.0], intensity_params={"clip_percentiles": [0.5, 99.5]}, modality="mri")
    d = tf({"image": str(ROOT / "data" / pid / "image.nii.gz"), "label": str(D6 / pid / "label.nii.gz")})
    u = set(np.unique(np.rint(d["label"].numpy())).astype(int))
    assert u == set(range(7)), u
    print(f"  ok transformations de validation : {pid} classes {sorted(u)}")


def test_eval_smoke():
    cfg = yaml.safe_load((ROOT / "configs/phase4/unet_6oar.yaml").read_text(encoding="utf-8"))
    torch.manual_seed(0)
    m = build_model("unet", 1, 7, features=cfg["model"]["features"])
    pids = patients()
    val = make_5fold_splits([{"patient_id": p} for p in pids], seed=42)[0]["val"]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        data6 = d / "data6"
        for pid in pids:                                    # equivalent local de data6/ (liens physiques)
            (data6 / pid).mkdir(parents=True)
            os.link(ROOT / "data" / pid / "image.nii.gz", data6 / pid / "image.nii.gz")
            os.link(D6 / pid / "label.nii.gz", data6 / pid / "label.nii.gz")
        rdir = d / "runs" / "unet6_s42_fold0"
        args = types.SimpleNamespace(tag="unet6_s42", model="unet", config="configs/phase4/unet_6oar.yaml", fold=0)
        train._write_run_meta(rdir, args, cfg, 42, 42, [{"patient_id": p} for p in val], count_parameters(m))
        torch.save({"model_state": m.state_dict(), "epoch": 1, "val_dsc": 0.0, "config": cfg["train"]}, rdir / "best.pt")
        env = dict(os.environ, P2_INPUT_ROOT=str(data6), P3_RUNS_ROOT=str(d / "runs"), P2_OUT=str(d / "out"),
                   P2_REPO_DIR=str(ROOT), P2_DEVICE="cpu", P2_SKIP_INSTALL="1", P2_MAX_PATIENTS="1",
                   P3_ORGANS=ORGANS6, P3_N_PATIENTS="115", PYTHONWARNINGS="ignore")
        r = subprocess.run([sys.executable, str(ROOT / "phase3_ablation/eval_ablation.py")], env=env,
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert not (d / "out" / "errors.log").exists(), (d / "out" / "errors.log").read_text()
        rows = list(csv.DictReader(open(d / "out" / "metrics_long.csv", encoding="utf-8")))
    assert len(rows) == 12 and {r["organ"] for r in rows} == set(ORGANS6.split(",")), len(rows)
    assert all(r["variant"] == "unet_6oar" and r["seed"] == "42" for r in rows)
    print(f"  ok evaluation 6 OAR de bout en bout (1 patient, CPU) : 12 lignes, {val[0]}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("TOUS LES TESTS PASSENT")
