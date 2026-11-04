"""Controle de eval_ablation.py (CPU, sans pytest) :

    .venv\\Scripts\\python.exe phase3_ablation/tests/test_eval_identique.py [--smoke]

1. Les fonctions et constantes de metriques sont identiques (AST) a phase2_kaggle_eval/phase2_eval_kaggle.py.
2. --smoke : evaluation de bout en bout sur CPU d'un checkpoint A1 initialise au hasard (1 patient du pli 0),
   avec run_meta.json ecrit par scripts/train.py ; verifie le format de sortie (colonnes seed / tag).
"""
from __future__ import annotations

import ast
import csv
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "phase2_kaggle_eval/phase2_eval_kaggle.py"
P3 = ROOT / "phase3_ablation/eval_ablation.py"

SAME_FUNCS = ["_bbox", "surface_stats", "overlap_stats", "resample_to_native", "grid_of", "pick_device",
              "find_dataset", "detect_variant", "export_history", "append_rows", "log_error", "sh"]
SAME_ASSIGNS = ["SPACING", "PATCH", "SW_OVERLAP", "SURF_TOL_MM", "SURF_TOL_VOX", "ORGANS", "NUM_CLASSES",
                "VARIANT_FEATURES", "SEGRESNET_KW", "STRUCT", "MONAI_VERSION", "val_tf", "post_pred",
                "post_label", "legacy", "resample"]


def defs(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs, assigns = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            funcs[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assigns.setdefault(node.targets[0].id, ast.dump(node.value))
    return funcs, assigns


def test_identical_metrics():
    f2, a2 = defs(P2)
    f3, a3 = defs(P3)
    for name in SAME_FUNCS:
        if name == "export_history":     # seul changement autorise : nom du fichier (graine)
            continue
        assert f2[name] == f3[name], f"fonction {name} differente de la phase 2"
    for name in SAME_ASSIGNS:
        assert a2[name] == a3[name], f"{name} different de la phase 2"
    # boucle de calcul : memes appels (overlap_stats / surface_stats / legacy) dans le meme ordre
    src2, src3 = P2.read_text(encoding="utf-8"), P3.read_text(encoding="utf-8")
    for frag in ['r.update(overlap_stats(pc, gc, vox_rs))',
                 's_mm = surface_stats(pc, gc, sp_rs, SURF_TOL_MM)',
                 's_vx = surface_stats(pc, gc, (1.0, 1.0, 1.0), SURF_TOL_VOX)',
                 'pnat = resample_to_native(pred, img.affine.clone(), aff_nat, gnat.shape)',
                 's_mm = surface_stats(pc, gc, sp_n, SURF_TOL_MM)',
                 'predictor=model, overlap=SW_OVERLAP, mode="gaussian",']:
        assert src2.count(frag) == src3.count(frag) == 1, frag
    print(f"  ok {len(SAME_FUNCS) - 1} fonctions + {len(SAME_ASSIGNS)} constantes identiques a la phase 2")


def test_smoke():
    import torch
    import yaml
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import train
    from src.models.factory import build_model, count_parameters
    from src.data.splits import make_5fold_splits

    cfg = yaml.safe_load((ROOT / "configs/ablation/unet_res.yaml").read_text(encoding="utf-8"))
    torch.manual_seed(0)
    m = build_model("unet", 1, 5, features=cfg["model"]["features"], unet_kwargs=cfg["model"]["unet"])
    patients = sorted(p.name for p in (ROOT / "data").iterdir() if p.is_dir())
    val = make_5fold_splits([{"patient_id": p} for p in patients], seed=42)[0]["val"]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rdir = d / "runs" / "unet_res_s123_fold0"
        args = types.SimpleNamespace(tag="unet_res_s123", model="unet",
                                     config="configs/ablation/unet_res.yaml", fold=0)
        train._write_run_meta(rdir, args, cfg, 123, 42, [{"patient_id": p} for p in val],
                              count_parameters(m))
        torch.save({"model_state": m.state_dict(), "epoch": 1, "val_dsc": 0.0,
                    "config": cfg["train"]}, rdir / "best.pt")
        env = dict(os.environ, P2_INPUT_ROOT=str(ROOT / "data"), P3_RUNS_ROOT=str(d / "runs"),
                   P2_OUT=str(d / "out"), P2_REPO_DIR=str(ROOT), P2_DEVICE="cpu", P2_SKIP_INSTALL="1",
                   P2_MAX_PATIENTS="1", PYTHONWARNINGS="ignore")
        r = subprocess.run([sys.executable, str(P3)], env=env, capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        rows = list(csv.DictReader(open(d / "out" / "metrics_long.csv", encoding="utf-8")))
        summ = list(csv.DictReader(open(d / "out" / "runs_summary.csv", encoding="utf-8")))
        assert not (d / "out" / "errors.log").exists(), (d / "out" / "errors.log").read_text()
    assert len(rows) == 8, len(rows)                           # 4 OAR x 2 grilles
    assert {r["space"] for r in rows} == {"resampled", "native"}
    assert all(r["variant"] == "unet_res" and r["seed"] == "123" and r["tag"] == "unet_res_s123"
               and r["fold"] == "0" and r["patient_id"] == val[0] for r in rows)
    assert summ[0]["n_params"] == "18918214" and summ[0]["status"] == "ok" and summ[0]["split_seed"] == "42"
    print(f"  ok evaluation de bout en bout (1 patient, CPU) : 8 lignes, seed/tag presents, {val[0]}")


if __name__ == "__main__":
    print("test_identical_metrics")
    test_identical_metrics()
    if "--smoke" in sys.argv:
        print("test_smoke")
        test_smoke()
    print("TESTS OK")
