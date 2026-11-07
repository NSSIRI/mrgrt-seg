#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lanceur Kaggle - XAI a couches appariees (E3 vs E2, checkpoints de phase 2). DRY-RUN par defaut.

    python phase3_ablation/xai_matched/launch_xai.py                       # construit les 2 notebooks
    python phase3_ablation/xai_matched/launch_xai.py --push --jobs xai-e3  # pousse CE job seulement
    python phase3_ablation/xai_matched/launch_xai.py --status
    python phase3_ablation/xai_matched/launch_xai.py --fetch

Les notebooks d'entrainement de phase 2 sont prives : chaque job tourne sur le compte qui les possede.
  xai-e3   : abdelhalimnssiri, E3 plis 0-3 (unet-fold-{0..3}-opt)
  xai-e3e2 : nssiri02, E3 pli 4 (unet-fold-4-opt) + E2 plis 0-4 (segresnet-fold-{0..4})
Authentification et verification du compte : fonctions de launch_ablation.py (token jamais lu).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from launch_ablation import (DATASET_SLUG, PY_KAGGLE, ROOT, TORCH_PIN, check_account,  # noqa: E402
                             kaggle, kenv)

CELL = HERE / "xai_matched_kaggle.py"
BUILD = HERE / "_build_kernels"
PLAN = HERE / "runs_xai.csv"
RESULTS = ROOT / "results" / "phase3_ablation" / "xai_matched"
REPO_COMMIT = "0a36a1e"   # src/ (donnees, plis, modeles) ; le script XAI lui-meme est integre au notebook

JOBS = [
    {"job": "xai-e3", "compte": "abdelhalimnssiri", "heures_estimees": 1.0,
     "sources": [f"abdelhalimnssiri/unet-fold-{k}-opt" for k in range(4)],
     "remarque": "E3 plis 0-3 (150 patients)"},
    {"job": "xai-e3e2", "compte": "nssiri02", "heures_estimees": 1.5,
     "sources": ["nssiri02/unet-fold-4-opt"] + [f"nssiri02/segresnet-fold-{k}" for k in range(5)],
     "remarque": "E3 pli 4 + E2 plis 0-4 (37 + 187 patients)"},
]
# Phase 4 (6 OAR, D115) : checkpoints des notebooks abl18m-{seg6,unet6}-s42-f<k>, prives -> job sur le compte proprietaire
P4_OWNERS = {"pokie02": ["seg6-s42-f0", "seg6-s42-f2", "unet6-s42-f1", "unet6-s42-f3"],
             "kozani": ["seg6-s42-f1", "seg6-s42-f3"],
             "kozani1": ["seg6-s42-f4", "unet6-s42-f2"],
             "kakachi11": ["unet6-s42-f0", "unet6-s42-f4"]}
LABELS_SLUG = "mrgrt-6oar-labels-d115"   # un dataset prive par compte (phase4_6oar/launch_6oar.py)
for owner, runs6 in P4_OWNERS.items():
    JOBS.append({"job": f"xai6-{owner}", "compte": owner, "heures_estimees": 0.4 * len(runs6),
                 "sources": [f"{owner}/abl18m-{r}" for r in runs6],
                 "datasets": [DATASET_SLUG, f"{owner}/{LABELS_SLUG}"],
                 "env": {"XAI_LABELS6": f"/kaggle/input/datasets/{owner}/{LABELS_SLUG}",
                         "XAI_ORGANS": "poumon_g,poumon_d,coeur,oesophage,moelle,aorte",
                         "XAI_N_PATIENTS": "115", "XAI_VARIANTS": "unet6,seg6"},
                 "remarque": "6 OAR : " + ", ".join(runs6)})
# Figures thoraciques : cartes des 5 patients a couverture pulmonaire complete (s0547, s0478, s0215, s0202, s0175),
# uniquement sur les plis qui les valident ; cartes sauvees pour chaque patient, sans Adebayo.
THORAX_PATIENTS = "PATIENT_s0547,PATIENT_s0478,PATIENT_s0215,PATIENT_s0202,PATIENT_s0175"
P4_THORAX = {"kozani": (["seg6-s42-f1", "seg6-s42-f3"], "1,3"),
             "pokie02": (["unet6-s42-f1", "unet6-s42-f3"], "1,3"),
             "kozani1": (["seg6-s42-f4"], "4"),
             "kakachi11": (["unet6-s42-f4"], "4")}
for owner, (runs6, folds) in P4_THORAX.items():
    JOBS.append({"job": f"xai6t-{owner}", "compte": owner, "heures_estimees": 0.3,
                 "sources": [f"{owner}/abl18m-{r}" for r in runs6],
                 "datasets": [DATASET_SLUG, f"{owner}/{LABELS_SLUG}"],
                 "env": {"XAI_LABELS6": f"/kaggle/input/datasets/{owner}/{LABELS_SLUG}",
                         "XAI_ORGANS": "poumon_g,poumon_d,coeur,oesophage,moelle,aorte",
                         "XAI_N_PATIENTS": "115", "XAI_VARIANTS": "unet6,seg6", "XAI_ONLY_FOLDS": folds,
                         "XAI_PATIENTS": THORAX_PATIENTS, "XAI_EXAMPLE_ALL": "1", "XAI_ADEBAYO": "0",
                         "XAI_OUT": "/kaggle/working/xai_thorax"},
                 "remarque": "cartes thorax : " + ", ".join(runs6)})
FIELDS = ["job", "compte", "kernel_id", "heures_estimees", "statut", "date_push", "date_fin", "remarque"]


def load_plan():
    old = {r["job"]: r for r in csv.DictReader(open(PLAN, encoding="utf-8"))} if PLAN.exists() else {}
    jobs = []
    for j in JOBS:
        j = dict(j, kernel_id=f"{j['compte']}/abl18m-{j['job']}-matched", statut="a_lancer", date_push="", date_fin="")
        if j["job"] in old:
            for k in ("statut", "date_push", "date_fin"):
                j[k] = old[j["job"]][k]
        jobs.append(j)
    return jobs


def save_plan(jobs):
    with open(PLAN, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(jobs)


def build(job):
    d = BUILD / job["job"]
    d.mkdir(parents=True, exist_ok=True)
    env = {"XAI_INPUT_ROOT": "/kaggle/input", "XAI_OUT": "/kaggle/working/xai_matched",
           "XAI_REPO_DIR": "/kaggle/working/mrgrt-seg", "XAI_REPO_COMMIT": REPO_COMMIT, **job.get("env", {})}
    src = CELL.read_text(encoding="utf-8")
    cell = (TORCH_PIN + "import os\n" + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
            + f"exec(compile({src!r}, 'xai_matched_kaggle.py', 'exec'))\n")
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": cell.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {"id": job["kernel_id"], "title": job["kernel_id"].split("/", 1)[1], "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": job.get("datasets", [DATASET_SLUG]), "competition_sources": [],
            "kernel_sources": job["sources"]}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--jobs", default="")
    ap.add_argument("--repush", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    jobs = load_plan()
    sel = [s for s in args.jobs.split(",") if s]
    unknown = set(sel) - {j["job"] for j in jobs}
    if unknown:
        sys.exit(f"jobs inconnus : {sorted(unknown)}")

    if args.status or args.fetch:
        for j in jobs:
            if j["statut"] == "a_lancer" or (sel and j["job"] not in sel):
                continue
            code, out = kaggle(j["compte"], "kernels", "status", j["kernel_id"])
            st = re.search(r'status "?(?:KernelWorkerStatus\.)?(\w+)', out)
            st = st.group(1).lower() if st else "inconnu"
            print(f"{j['job']:<10} {j['kernel_id']:<45} {st}")
            if st in ("complete", "error", "cancel_acknowledged", "cancelled"):
                j["date_fin"] = j["date_fin"] or time.strftime("%Y-%m-%d %H:%M")
                j["statut"] = st
                if args.fetch and st == "complete":
                    dest = RESULTS / j["compte"] / j["kernel_id"].split("/", 1)[1]
                    subprocess.run([PY_KAGGLE, str(ROOT / "phase2_kaggle_eval/17_recuperer_sorties.py"),
                                    j["kernel_id"], str(dest), r"\.(csv|json|log|txt|npz)$"], env=kenv(j["compte"]))
            elif st in ("running", "queued"):
                j["statut"] = st
        save_plan(jobs)
        return

    if args.push and not sel:
        sys.exit("--push exige --jobs <liste>.")
    print(f"{len(jobs)} jobs | {'PUSH' if args.push else 'DRY-RUN'} | commit src {REPO_COMMIT}")
    for j in [j for j in jobs if not sel or j["job"] in sel]:
        d = build(j)
        print(f"  {j['job']:<10} {j['kernel_id']:<45} {j['heures_estimees']:.1f} h  {j['remarque']}  statut={j['statut']}")
        print(f"     sources : {', '.join(j['sources'])}")
        if not args.push:
            continue
        if j["statut"] != "a_lancer" and not args.repush:
            print("     deja pousse : ignore")
            continue
        check_account(j["compte"])
        code, out = kaggle(j["compte"], "kernels", "push", "-p", str(d))
        print("     " + out.replace("\n", "\n     "))
        ok = code == 0 and "error" not in out.lower()
        j["statut"], j["date_push"] = ("pousse", time.strftime("%Y-%m-%d %H:%M")) if ok else ("echec_push", "")
    save_plan(jobs)
    if not args.push:
        print(f"\n(DRY-RUN : rien n'a ete envoye ; notebooks dans {BUILD})")


if __name__ == "__main__":
    main()
