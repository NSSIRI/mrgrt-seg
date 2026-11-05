#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lanceur Kaggle - phase 3 (ablation 18,8 M), option Standard. Inspire de scripts/kaggle/launch_all.py (non modifie).

DRY-RUN par defaut : construit les notebooks dans phase3_ablation/_build_kernels/ et met a jour
phase3_ablation/runs_plan.csv, sans rien envoyer.

    python phase3_ablation/launch_ablation.py                         # plan + notebooks (DRY-RUN)
    python phase3_ablation/launch_ablation.py --jobs ctrl-e3-f0 --push  # pousse CES jobs seulement
    python phase3_ablation/launch_ablation.py --status                # kaggle kernels status des jobs pousses
    python phase3_ablation/launch_ablation.py --fetch [--with-zip]    # recupere les sorties des jobs termines

Authentification : KAGGLE_API_TOKEN=<chemin du fichier token> (jamais lu par ce script), puis
`kaggle config view` doit montrer le bon compte et ACCESS_TOKEN avant chaque push.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "_build_kernels"
PLAN = HERE / "runs_plan.csv"
CELL_TRAIN = HERE / "train_ablation_kaggle.py"
CELL_EVAL = HERE / "eval_ablation.py"
RESULTS = ROOT / "results" / "phase3_ablation"
KAGGLE = r"C:\Users\Lenovo\AppData\Roaming\Python\Python311\Scripts\kaggle.exe"
PY_KAGGLE = r"C:\Users\Lenovo\AppData\Local\Programs\Python\Python311\python.exe"   # interpreteur ou le paquet kaggle est installe

REPO_COMMIT = "0a36a1e"
DATASET_SLUG = "abdelhalimnssiri/mrgrt-oar-thorax-clean-v2"
DATASET_MOUNT = "/kaggle/input/datasets/abdelhalimnssiri/mrgrt-oar-thorax-clean-v2"
ACCOUNTS = {   # compte -> fichier token (chemin seulement ; le contenu n'est jamais lu ici)
    "abdelhalimnssiri": Path.home() / ".kaggle" / "access_token_compte1",
    "nssiri02": Path.home() / ".kaggle" / "access_token_compte2",
    "nasri022026": Path.home() / ".kaggle" / "access_token_compte3",   # compte d'un collegue (accord, 04/11/2026)
    "kozani": Path.home() / ".kaggle" / "access_token_compte4",        # compte d'un collegue (accord, 05/11/2026)
    "kozani1": Path.home() / ".kaggle" / "access_token_compte5",       # compte d'un collegue (accord, 05/11/2026)
}
QUOTA_H = 30.0     # heures GPU / semaine / compte (a verifier sur la page du compte)

VARIANTS = {   # variante -> (modele, config, heures estimees par run : entrainement + evaluation)
    "unet_opt": ("unet", "configs/ablation/e3_unet_opt.yaml", 3.14 + 0.15),    # mesure etape 0
    "segresnet": ("segresnet", "configs/ablation/e2_segresnet.yaml", 6.59 + 0.20),
    "unet_res": ("unet", "configs/ablation/unet_res.yaml", 3.9 + 0.2),         # mesure pli 0 (le smoke de 331 s/epoque etait trompeur)
    "unet_gn": ("unet", "configs/ablation/unet_gn.yaml", 3.14 + 0.15),         # suppose = E3
    "segresnet_in": ("segresnet", "configs/ablation/segresnet_in.yaml", 6.59 + 0.20),  # suppose = E2
}
FORCED_ACCOUNT = {"segresnet_in-s42-f2": "nasri022026"}   # v1 interrompue sur abdelhalimnssiri (quota presque epuise)
SPLIT_VARIANTS = {"unet_res"}   # entrainement en 2 sessions (reprise depuis last.pt)
FIELDS = ["job", "bloc", "variante", "modele", "config", "graine", "pli", "epoques", "compte", "kernel_id",
          "heures_estimees", "semaine", "statut", "date_push", "date_fin", "remarque"]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")


# ------------------------------------------------------------------ plan
def standard_plan() -> list[dict]:
    jobs = [
        {"job": "ctrl-e3-f0", "bloc": "controle", "variante": "unet_opt", "graine": 42, "pli": 0, "epoques": 0,
         "compte": "abdelhalimnssiri", "heures_estimees": 0.25,
         "remarque": "evaluation seule du best.pt E3 pli 0 de phase 2 (ecart < 1e-6 attendu)"},
        {"job": "smoke-1ep", "bloc": "controle", "variante": "unet_res+unet_gn+segresnet_in", "graine": 42,
         "pli": 0, "epoques": 1, "heures_estimees": 0.6,
         "remarque": "1 epoque A1, A2, A3 a la suite : memoire GPU, temps par epoque ; pas d'evaluation"},
    ]
    for v in ("unet_res", "unet_gn", "segresnet_in"):                       # bloc B
        for k in range(5):
            if v in SPLIT_VARIANTS:   # trop long pour une session : arret a MAX_TRAIN_HOURS puis reprise
                jobs.append({"job": f"{v}-s42-f{k}-p1", "bloc": "B", "variante": v, "graine": 42, "pli": k,
                             "epoques": 150, "heures_estimees": 4.1,
                             "remarque": "mesure pli 0 : 79 s/epoque, 3,9 h -> une seule session suffit"})
                jobs.append({"job": f"{v}-s42-f{k}-p2", "bloc": "B", "variante": v, "graine": 42, "pli": k,
                             "epoques": 150, "heures_estimees": 0.0,
                             "remarque": "a lancer SEULEMENT si p1 a ecrit runs/A_REPRENDRE (non necessaire au pli 0)"})
            else:
                jobs.append({"job": f"{v}-s42-f{k}", "bloc": "B", "variante": v, "graine": 42, "pli": k,
                             "epoques": 150})
    for v in ("unet_opt", "segresnet"):                                     # bloc A, graine 123
        for k in range(5):
            jobs.append({"job": f"{v}-s123-f{k}", "bloc": "A", "variante": v, "graine": 123, "pli": k,
                         "epoques": 150})
    for j in jobs:
        if j["variante"] in VARIANTS:
            m, c, h = VARIANTS[j["variante"]]
            j.update(modele=m, config=c)
            j.setdefault("heures_estimees", h)
        else:
            j.update(modele="", config="")
        j.setdefault("remarque", "")
    assign(jobs)
    for j in jobs:
        j["kernel_id"] = f"{j['compte']}/abl18m-{slug(j['job'])}"
        j.update(statut="a_lancer", date_push="", date_fin="")
    return jobs


def assign(jobs):
    """Repartition gloutonne (plus long d'abord) sur le compte le moins charge ; semaine = remplissage a QUOTA_H."""
    load = {a: 0.0 for a in ACCOUNTS}
    by_id = {j["job"]: j for j in jobs}
    for j in sorted(jobs, key=lambda j: -float(j["heures_estimees"])):
        if j["job"].endswith("-p2"):          # place avec sa partie 1 (le notebook source doit etre du meme compte)
            continue
        if not j.get("compte"):
            j["compte"] = min(load, key=load.get)
        load[j["compte"]] += float(j["heures_estimees"])
        p2 = by_id.get(j["job"][:-3] + "-p2") if j["job"].endswith("-p1") else None
        if p2:
            p2["compte"] = j["compte"]
            load[j["compte"]] += float(p2["heures_estimees"])
    for a in ACCOUNTS:   # ordre d'execution : controles, bloc B pli par pli, puis bloc A
        order = sorted([j for j in jobs if j["compte"] == a],
                       key=lambda j: ({"controle": 0, "B": 1, "A": 2}[j["bloc"]], int(j["pli"]), j["job"]))
        week, used = 1, 0.0
        for j in order:
            h = float(j["heures_estimees"])
            if used + h > QUOTA_H:
                week, used = week + 1, 0.0
            used += h
            j["semaine"] = week


def load_plan() -> list[dict]:
    fresh = standard_plan()
    if PLAN.exists():   # garde statut / dates / compte des jobs deja pousses
        old = {r["job"]: r for r in csv.DictReader(open(PLAN, encoding="utf-8"))}
        for j in fresh:
            o = old.get(j["job"])
            if o and o["statut"] != "a_lancer":
                for k in ("compte", "kernel_id", "statut", "date_push", "date_fin", "semaine"):
                    j[k] = o[k]
    # comptes imposes (ex. job relance sur un compte qui a encore du quota)
    for j in fresh:
        if j["statut"] == "a_lancer" and j["job"] in FORCED_ACCOUNT:
            j["compte"] = FORCED_ACCOUNT[j["job"]]
            j["kernel_id"] = f"{j['compte']}/abl18m-{slug(j['job'])}"
            j["date_push"] = ""
    # une partie 2 suit toujours le compte REEL de sa partie 1 (notebook source prive du meme compte)
    by_id = {j["job"]: j for j in fresh}
    for j in fresh:
        p1 = by_id.get(j["job"][:-3] + "-p1") if j["job"].endswith("-p2") else None
        if p1 and j["statut"] == "a_lancer" and j["compte"] != p1["compte"]:
            j["compte"] = p1["compte"]
            j["kernel_id"] = f"{j['compte']}/abl18m-{slug(j['job'])}"
    return fresh


def save_plan(jobs):
    with open(PLAN, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(jobs)


# ------------------------------------------------------------------ notebooks
def setv(code: str, name: str, value: str) -> str:
    new, n = re.subn(rf"(?m)^{name}\s*=.*$", lambda _: f"{name} = {value}", code, count=1)
    assert n == 1, name
    return new


def train_cell(jobs_list, epochs, resume=False) -> str:
    c = CELL_TRAIN.read_text(encoding="utf-8")
    c = setv(c, "RESUME", repr("auto") if resume else "None")
    c = setv(c, "REPO_COMMIT", repr(REPO_COMMIT))
    c = setv(c, "KAGGLE_INPUT_DATASET", repr(DATASET_MOUNT))
    c = setv(c, "EPOCHS", str(epochs))
    c = re.sub(r"(?ms)^JOBS = \[\n.*?^\]\n", lambda _: "JOBS = " + json.dumps(jobs_list, indent=1) + "\n", c, count=1)
    assert "JOBS = [\n {" in c
    return c


def eval_cell(only_folds=None) -> str:
    env = {"P3_RUNS_ROOT": "/kaggle/working/runs", "P2_OUT": "/kaggle/working/ablation_eval",
           "P2_REPO_DIR": "/kaggle/working/mrgrt-seg", "P3_REPO_COMMIT": REPO_COMMIT}
    head = "import os\n" + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
    if only_folds is not None:
        head += f"os.environ['P2_ONLY_FOLDS'] = {','.join(map(str, only_folds))!r}\n"
    # evaluation sautee si l'entrainement a ete interrompu (fichier A_REPRENDRE ecrit par la cellule d'entrainement)
    src = CELL_EVAL.read_text(encoding="utf-8")
    return (head + "from pathlib import Path as _P\n"
            "if _P('/kaggle/working/runs/A_REPRENDRE').exists():\n"
            "    print('Entrainement interrompu : evaluation sautee (reprise necessaire).')\n"
            "else:\n"
            f"    exec(compile({src!r}, 'eval_ablation.py', 'exec'))\n")


TORCH_PIN = (  # notebook d'evaluation seule : torch 2.10.0 installe avant tout import de torch
    "import subprocess, sys\n"
    "assert 'torch' not in sys.modules\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.10.0', '--index-url', "
    "'https://download.pytorch.org/whl/cu128'], check=True)\n"
    "import torch\n"
    "assert torch.__version__.startswith('2.10.0'), torch.__version__\n"
    "print('torch', torch.__version__)\n\n")


def build(job: dict) -> Path:
    d = BUILD / job["job"]
    d.mkdir(parents=True, exist_ok=True)
    kernel_sources, cells = [], []
    if job["job"] == "ctrl-e3-f0":
        kernel_sources = ["abdelhalimnssiri/unet-fold-0-opt"]
        cells = [TORCH_PIN + eval_cell(only_folds=[0]).replace(
            "os.environ['P2_ONLY_FOLDS']", "os.environ['P2_SKIP_INSTALL'] = '0'\nos.environ['P2_ONLY_FOLDS']", 1)]
    elif job["job"] == "smoke-1ep":
        jl = [{"model": VARIANTS[v][0], "config": VARIANTS[v][1], "fold": 0, "seed": 42, "tag": f"smoke_{v}"}
              for v in ("unet_res", "unet_gn", "segresnet_in")]
        # 3 jobs a la suite (un seul GPU a la fois) : 3 executions de la cellule d'entrainement
        cells = [train_cell([x], 1) for x in jl]
    else:
        jl = [{"model": job["modele"], "config": job["config"], "fold": int(job["pli"]),
               "seed": int(job["graine"]), "tag": f"{job['variante']}_s{job['graine']}"}]
        is_p2 = job["job"].endswith("-p2")
        if is_p2:
            kernel_sources = [job["kernel_id"][:-3] + "-p1"]
        cells = [train_cell(jl, int(job["epoques"]), resume=is_p2), eval_cell()]
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": c.splitlines(keepends=True)} for c in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    sl = job["kernel_id"].split("/", 1)[1]
    meta = {"id": job["kernel_id"], "title": sl, "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [DATASET_SLUG], "competition_sources": [], "kernel_sources": kernel_sources}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


# ------------------------------------------------------------------ CLI kaggle
def kenv(account: str) -> dict:
    return dict(os.environ, KAGGLE_API_TOKEN=str(ACCOUNTS[account]), PYTHONUTF8="1")


def kaggle(account: str, *args) -> tuple[int, str]:
    r = subprocess.run([KAGGLE, *args], env=kenv(account), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "outdated" not in l)
    return r.returncode, out


def check_account(account: str) -> None:
    code, out = kaggle(account, "config", "view")
    user = re.search(r"username:\s*(\S+)", out)
    meth = re.search(r"auth_method:\s*(\S+)", out)
    if code or not user or user.group(1) != account or not meth or meth.group(1) != "ACCESS_TOKEN":
        sys.exit(f"ERREUR authentification {account} : username={user and user.group(1)} "
                 f"methode={meth and meth.group(1)}")
    print(f"   compte verifie : {account} (ACCESS_TOKEN)")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="pousse reellement les jobs de --jobs")
    ap.add_argument("--jobs", default="", help="liste de jobs separes par des virgules (obligatoire avec --push)")
    ap.add_argument("--repush", action="store_true", help="avec --push : nouvelle version d'un job deja pousse")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--with-zip", action="store_true", help="avec --fetch : telecharge aussi predictions.zip")
    args = ap.parse_args()

    jobs = load_plan()
    by_id = {j["job"]: j for j in jobs}
    selected = [s for s in args.jobs.split(",") if s]
    unknown = [s for s in selected if s not in by_id]
    if unknown:
        sys.exit(f"jobs inconnus : {unknown}")

    if args.status or args.fetch:
        for j in jobs:
            if j["statut"] in ("a_lancer",) or (selected and j["job"] not in selected):
                continue
            code, out = kaggle(j["compte"], "kernels", "status", j["kernel_id"])
            st = re.search(r'status "?(?:KernelWorkerStatus\.)?(\w+)', out)
            st = st.group(1).lower() if st else "inconnu"
            print(f"{j['job']:<20} {j['kernel_id']:<45} {st}")
            if st in ("complete", "error", "cancel_acknowledged", "cancelled"):
                if not j["date_fin"]:
                    j["date_fin"] = time.strftime("%Y-%m-%d %H:%M")
                j["statut"] = st
                if args.fetch and st == "complete":
                    dest = RESULTS / j["compte"] / j["kernel_id"].split("/", 1)[1]
                    small = r"\.(csv|json|log|txt|yaml|npz)$"
                    for pat in [small] + ([r"predictions\.zip$"] if args.with_zip else []):
                        subprocess.run([PY_KAGGLE, str(ROOT / "phase2_kaggle_eval/17_recuperer_sorties.py"),
                                        j["kernel_id"], str(dest), pat], env=kenv(j["compte"]))
            elif st in ("running", "queued"):
                j["statut"] = st
        save_plan(jobs)
        return

    todo = [by_id[s] for s in selected] if selected else jobs
    print(f"{len(jobs)} jobs au plan | {'PUSH' if args.push else 'DRY-RUN'} | commit {REPO_COMMIT}")
    for a in ACCOUNTS:
        hs = [float(j["heures_estimees"]) for j in jobs if j["compte"] == a]
        weeks = max(int(j["semaine"]) for j in jobs if j["compte"] == a)
        print(f"   {a:<18} {len(hs):>2} jobs  {sum(hs):6.1f} h estimees  {weeks} semaine(s) a {QUOTA_H:.0f} h")
    if args.push and not selected:
        sys.exit("--push exige --jobs <liste> : rien n'est pousse en masse.")
    checked = set()
    for j in todo:
        d = build(j)
        print(f"  [{j['bloc']:<8}] {j['job']:<20} {j['kernel_id']:<45} {float(j['heures_estimees']):5.2f} h  "
              f"semaine {j['semaine']}  statut={j['statut']}")
        if not args.push:
            continue
        if j["statut"] != "a_lancer" and not args.repush:
            print(f"     deja pousse ({j['statut']}) : ignore")
            continue
        if j["compte"] not in checked:
            check_account(j["compte"])
            checked.add(j["compte"])
        code, out = kaggle(j["compte"], "kernels", "push", "-p", str(d))
        print("     " + out.replace("\n", "\n     "))
        if code == 0 and "error" not in out.lower():
            j["statut"], j["date_push"] = "pousse", time.strftime("%Y-%m-%d %H:%M")
        else:
            j["statut"] = "echec_push"
    save_plan(jobs)
    print(f"\nPlan : {PLAN}" + ("" if args.push else "\n(DRY-RUN : rien n'a ete envoye ; --push --jobs <liste> pour lancer)"))


if __name__ == "__main__":
    main()
