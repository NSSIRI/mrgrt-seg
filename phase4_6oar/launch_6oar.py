#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lanceur Kaggle - phase 4 (6 OAR, cohorte D115). Reutilise phase3_ablation/launch_ablation.py (cellules,
authentification, statut, recuperation) ; seuls le plan, les datasets et l'environnement d'evaluation changent.

DRY-RUN par defaut :
    python phase4_6oar/launch_6oar.py                                   # plan + notebooks
    python phase4_6oar/launch_6oar.py --prepare-datasets                # dossiers a televerser (local)
    python phase4_6oar/launch_6oar.py --create-datasets                 # televerse (prive, 1 par compte)
    python phase4_6oar/launch_6oar.py --jobs seg6-s42-f0 --push         # pousse CES jobs
    python phase4_6oar/launch_6oar.py --status | --fetch [--with-zip]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("la", ROOT / "phase3_ablation/launch_ablation.py")
la = importlib.util.module_from_spec(spec)
spec.loader.exec_module(la)

PLAN = HERE / "runs_plan_6oar.csv"
BUILD = HERE / "_build_kernels"
LABELS = ROOT / "data_6oar"
LABELS_SLUG = "mrgrt-6oar-labels-d115"          # un dataset prive par compte : <compte>/mrgrt-6oar-labels-d115
ORGANS6 = "poumon_g,poumon_d,coeur,oesophage,moelle,aorte"
N_PATIENTS = 115
REPO_COMMIT = None   # rempli par --commit (commit pousse contenant configs/phase4) ; obligatoire pour --push
MODELS = {   # job -> (modele, config, heures estimees : entrainement + evaluation ; D115 = 115/187 des patients)
    "seg6": ("segresnet", "configs/phase4/segresnet_6oar.yaml", round(6.59 * 115 / 187 + 0.4, 2)),
    "unet6": ("unet", "configs/phase4/unet_6oar.yaml", round(3.14 * 115 / 187 + 0.4, 2)),
}


def plan() -> list[dict]:
    jobs = []
    for k in range(5):
        for key, (m, c, h) in MODELS.items():
            jobs.append({"job": f"{key}-s42-f{k}", "bloc": "6OAR", "variante": key, "modele": m, "config": c,
                         "graine": 42, "pli": k, "epoques": 150, "heures_estimees": h,
                         "remarque": "estimation proportionnelle au nombre de patients : a confirmer"})
    load = {a: 0.0 for a in la.ACCOUNTS}
    for j in sorted(jobs, key=lambda j: -j["heures_estimees"]):
        j["compte"] = min(load, key=load.get)
        load[j["compte"]] += j["heures_estimees"]
    for j in jobs:
        j["kernel_id"] = f"{j['compte']}/abl18m-{la.slug(j['job'])}"
        j.update(semaine="apres ablation", statut="a_lancer", date_push="", date_fin="")
    if PLAN.exists():
        old = {r["job"]: r for r in csv.DictReader(open(PLAN, encoding="utf-8"))}
        for j in jobs:
            o = old.get(j["job"])
            if o and o["statut"] != "a_lancer":
                for k in ("compte", "kernel_id", "statut", "date_push", "date_fin"):
                    j[k] = o[k]
    return jobs


def save(jobs):
    with open(PLAN, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=la.FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(jobs)


def build(job, commit) -> Path:
    d = BUILD / job["job"]
    d.mkdir(parents=True, exist_ok=True)
    owner = job["compte"]
    jl = [{"model": job["modele"], "config": job["config"], "fold": int(job["pli"]), "seed": int(job["graine"]),
           "tag": f"{job['variante']}_s{job['graine']}"}]
    tc = la.train_cell(jl, int(job["epoques"]))
    tc = la.setv(tc, "REPO_COMMIT", repr(commit or "A_RENSEIGNER"))
    tc = la.setv(tc, "LABELS_6OAR", repr(f"/kaggle/input/datasets/{owner}/{LABELS_SLUG}"))
    ec = la.eval_cell()
    ec = ec.replace("import os\n", "import os\n"
                    "os.environ['P2_INPUT_ROOT'] = '/kaggle/working/mrgrt-seg/data6'\n"
                    f"os.environ['P3_ORGANS'] = {ORGANS6!r}\n"
                    f"os.environ['P3_N_PATIENTS'] = '{N_PATIENTS}'\n", 1)
    if commit:
        ec = ec.replace(f"os.environ['P3_REPO_COMMIT'] = {la.REPO_COMMIT!r}", f"os.environ['P3_REPO_COMMIT'] = {commit!r}")
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": c.splitlines(keepends=True)} for c in (tc, ec)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    sl = job["kernel_id"].split("/", 1)[1]
    meta = {"id": job["kernel_id"], "title": sl, "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [la.DATASET_SLUG, f"{owner}/{LABELS_SLUG}"], "competition_sources": [],
            "kernel_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def prepare_datasets() -> dict:
    out = {}
    for owner in la.ACCOUNTS:
        d = HERE / f"_dataset_{owner}"
        d.mkdir(exist_ok=True)
        for lab in sorted(LABELS.glob("PATIENT_s*/label.nii.gz")):
            dst = d / f"{lab.parent.name}_label.nii.gz"
            if not dst.exists():
                shutil.copy2(lab, dst)
        shutil.copy2(LABELS / "manifest_6oar.csv", d / "manifest_6oar.csv")
        meta = {"title": LABELS_SLUG, "id": f"{owner}/{LABELS_SLUG}", "licenses": [{"name": "CC-BY-4.0"}],
                "description": "Labels 6 OAR (1 poumon G, 2 poumon D, 3 coeur, 4 oesophage, 5 moelle, 6 aorte) "
                               "pour 115 patients de TotalSegmentator MRI v2.0 (CC BY 4.0) ; images dans "
                               f"{la.DATASET_SLUG}."}
        (d / "dataset-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        n = len(list(d.glob("PATIENT_s*_label.nii.gz")))
        out[owner] = (d, n)
        print(f"   {owner}: {d} ({n} labels)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default=None, help="commit pousse sur ablation-18m contenant configs/phase4")
    ap.add_argument("--jobs", default="")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--prepare-datasets", action="store_true")
    ap.add_argument("--create-datasets", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--with-zip", action="store_true")
    args = ap.parse_args()

    if args.prepare_datasets or args.create_datasets:
        ds = prepare_datasets()
        if args.create_datasets:
            for owner, (d, n) in ds.items():
                la.check_account(owner)
                code, out = la.kaggle(owner, "datasets", "create", "-p", str(d))
                print(f"   {owner}: code {code}\n     " + out.replace("\n", "\n     "))
        return

    jobs = plan()
    if args.status or args.fetch:
        # reutilise la logique du lanceur de phase 3 sur ce plan
        la.PLAN = PLAN
        la.load_plan = lambda: jobs
        sys.argv = [sys.argv[0]] + (["--status"] if args.status else ["--fetch"]) + \
                   (["--with-zip"] if args.with_zip else []) + (["--jobs", args.jobs] if args.jobs else [])
        return la.main()

    sel = [s for s in args.jobs.split(",") if s]
    by = {j["job"]: j for j in jobs}
    if any(s not in by for s in sel):
        sys.exit(f"jobs inconnus : {[s for s in sel if s not in by]}")
    if args.push and (not sel or not args.commit):
        sys.exit("--push exige --jobs <liste> et --commit <sha pousse>")
    tot = {a: sum(j["heures_estimees"] for j in jobs if j["compte"] == a) for a in la.ACCOUNTS}
    print(f"{len(jobs)} jobs 6 OAR | {'PUSH' if args.push else 'DRY-RUN'} | commit {args.commit or 'non fixe'} | "
          + " ; ".join(f"{a} {h:.1f} h" for a, h in tot.items()))
    checked = set()
    for j in ([by[s] for s in sel] if sel else jobs):
        d = build(j, args.commit)
        print(f"  {j['job']:<14} {j['kernel_id']:<42} {j['heures_estimees']:5.2f} h  statut={j['statut']}")
        if not args.push or j["statut"] != "a_lancer":
            continue
        if j["compte"] not in checked:
            la.check_account(j["compte"])
            checked.add(j["compte"])
        code, out = la.kaggle(j["compte"], "kernels", "push", "-p", str(d))
        print("     " + out.replace("\n", "\n     "))
        j["statut"] = "pousse" if code == 0 and "error" not in out.lower() else "echec_push"
        j["date_push"] = time.strftime("%Y-%m-%d %H:%M")
    save(jobs)


if __name__ == "__main__":
    main()
