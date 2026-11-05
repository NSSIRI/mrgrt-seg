#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordonnanceur automatique (accord de l'utilisateur du 04/11/2026 : "une fois une place vide, passer au job suivant").

Une passe toutes les POLL secondes jusqu'au premier evenement, puis s'arrete (Claude est notifie, verifie, relance) :
  - statut Kaggle des jobs pousses ; job termine -> recuperation des petits fichiers ; job en erreur -> signale
    (jamais relance automatiquement) ;
  - pour chaque compte avec moins de MAX_PAR_COMPTE sessions actives : lance le prochain job "a_lancer" du plan
    (ablation : bloc B puis bloc A ; puis phase 4 6 OAR), si le quota glissant sur 7 jours le permet ;
  - partie 2 d'A1 : lancee seulement si la partie 1 a ecrit runs/A_REPRENDRE, sinon marquee "inutile".

    python phase3_ablation/auto_scheduler.py [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


la = load("la", ROOT / "phase3_ablation/launch_ablation.py")
l6 = load("l6", ROOT / "phase4_6oar/launch_6oar.py")

MAX_PAR_COMPTE = 2
QUOTA_7J = 30.0            # h GPU par compte sur 7 jours glissants (a verifier sur la page des comptes)
POLL = 300
PHASE4_COMMIT = "65a63af"  # commit pousse contenant configs/phase4
ACTIFS = {"pousse", "running", "queued"}
SMALL = r"^(ablation_eval/|runs/[^/]+\.csv$|runs/A_REPRENDRE$|runs/[^/]+/(run_meta\.json|history\.npz)$)"


def now():
    return dt.datetime.now()


def status_of(j):
    code, out = la.kaggle(j["compte"], "kernels", "status", j["kernel_id"])
    m = re.search(r'status "?(?:KernelWorkerStatus\.)?(\w+)', out)
    return m.group(1).lower() if m else "inconnu"


def fetch(j):
    dest = la.RESULTS / j["compte"] / j["kernel_id"].split("/", 1)[1]
    subprocess.run([la.PY_KAGGLE, str(ROOT / "phase2_kaggle_eval/17_recuperer_sorties.py"), j["kernel_id"],
                    str(dest), SMALL], env=la.kenv(j["compte"]), capture_output=True, text=True)
    return dest


def hours_7d(jobs, account):
    lim = now() - dt.timedelta(days=7)
    h = 0.0
    for j in jobs:
        if j["compte"] != account or not j.get("date_push"):
            continue
        try:
            t = dt.datetime.strptime(j["date_push"], "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        if t >= lim:
            h += 0.2 if j["statut"].startswith("error") else float(j["heures_estimees"])
    return h


def one_pass(dry):
    events = []
    ab = la.load_plan()
    p4 = l6.plan()
    by_ab = {j["job"]: j for j in ab}

    # 1) suivi des jobs actifs
    for plan in (ab, p4):
        for j in plan:
            if j["statut"] not in ACTIFS:
                continue
            st = status_of(j)
            if st in ("running", "queued"):
                j["statut"] = st
            elif st == "complete":
                j["statut"], j["date_fin"] = "complete", now().strftime("%Y-%m-%d %H:%M")
                dest = fetch(j)
                reprise = (dest / "runs" / "A_REPRENDRE").exists()
                events.append(f"TERMINE {j['job']} ({j['compte']})" + (" -> reprise necessaire (A_REPRENDRE)" if reprise else ""))
            elif st in ("error", "cancel_acknowledged", "cancelled"):
                j["statut"], j["date_fin"] = "error", now().strftime("%Y-%m-%d %H:%M")
                events.append(f"ECHEC {j['job']} ({j['compte']}) : a examiner, pas de relance automatique")

    # 2) parties 2 d'A1 : utiles seulement si la partie 1 a ete interrompue
    for j in ab:
        if j["job"].endswith("-p2") and j["statut"] == "a_lancer":
            p1 = by_ab.get(j["job"][:-3] + "-p1")
            if p1 and p1["statut"] == "complete":
                d1 = la.RESULTS / p1["compte"] / p1["kernel_id"].split("/", 1)[1]
                if not (d1 / "runs" / "A_REPRENDRE").exists():
                    j["statut"] = "inutile"

    # 3) remplissage des places libres
    def eligible(j):
        if j["statut"] != "a_lancer" or j["bloc"] == "controle":
            return False
        if j["job"].endswith("-p2"):
            p1 = by_ab.get(j["job"][:-3] + "-p1")
            return bool(p1 and p1["statut"] == "complete")
        return True

    order = {"B": 1, "A": 2, "6OAR": 3}
    for acc in la.ACCOUNTS:
        actifs = sum(1 for plan in (ab, p4) for j in plan if j["compte"] == acc and j["statut"] in ACTIFS)
        while actifs < MAX_PAR_COMPTE:
            cands = sorted([j for j in ab + p4 if j["compte"] == acc and eligible(j)],
                           key=lambda j: (order.get(j["bloc"], 9), int(j["pli"]), j["job"]))
            # la phase 4 ne demarre qu'une fois tous les jobs de l'ablation lances
            if any(j["bloc"] in ("A", "B") and eligible(j) for j in ab):
                cands = [j for j in cands if j["bloc"] != "6OAR"]
            if not cands:
                break
            used = hours_7d(ab + p4, acc)
            fits = [c for c in cands if used + float(c["heures_estimees"]) <= QUOTA_7J]
            if not fits:   # info seulement : ne compte pas comme un evenement (pas d'arret de l'ordonnanceur)
                print(f"  quota {acc} : {used:.1f} h sur 7 jours, {cands[0]['job']} attend", flush=True)
                break
            j = fits[0]    # le premier job du plan qui tient dans le quota restant
            if dry:
                events.append(f"(dry-run) LANCERAIT {j['job']} sur {acc}")
                j["statut"] = "pousse"
            else:
                la.check_account(acc)
                d = l6.build(j, PHASE4_COMMIT) if j["bloc"] == "6OAR" else la.build(j)
                code, out = la.kaggle(acc, "kernels", "push", "-p", str(d))
                ok = code == 0 and "error" not in out.lower()
                j["statut"] = "pousse" if ok else "echec_push"
                j["date_push"] = now().strftime("%Y-%m-%d %H:%M")
                events.append(f"{'LANCE' if ok else 'ECHEC_PUSH'} {j['job']} sur {acc} ({float(j['heures_estimees']):.1f} h)")
            actifs += 1
    if not dry:
        la.save_plan(ab)
        l6.save(p4)
    return events, ab, p4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    while True:
        ev, ab, p4 = one_pass(args.dry_run)
        restant = sum(1 for j in ab + p4 if j["statut"] == "a_lancer" and not j["job"].endswith("-p2"))
        actifs = sum(1 for j in ab + p4 if j["statut"] in ACTIFS)
        if ev or args.dry_run:
            print(now().strftime("%Y-%m-%d %H:%M"), "|", " ; ".join(ev) or "aucun evenement",
                  f"| actifs {actifs} | a lancer {restant}", flush=True)
            return
        if actifs == 0 and restant == 0:
            print("TOUT EST TERMINE", flush=True)
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
