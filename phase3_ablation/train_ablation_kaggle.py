"""Cellule Kaggle - phase 3 (ablation 18,8 M). Derivee de scripts/kaggle/train_kaggle.py (non modifie).

Rempli automatiquement par phase3_ablation/launch_ablation.py (variables de la section A AJUSTER).
Une session :
  1) clone le depot public a la branche ablation-18m, commit fige (REPO_COMMIT) ;
  2) installe monai==1.5.2 (fige) et les dependances de train_kaggle.py ;
  3) relie data/ au dataset Kaggle et runs/ a /kaggle/working/runs ;
  2b) torch==2.10.0 (cu128) fige : meme version que les entrainements E2/E3 ;
  4) entraine 1 job (ou 2 jobs en parallele, un par GPU, si len(JOBS) == 2) ; arret propre a MAX_TRAIN_HOURS
     (fichier runs/A_REPRENDRE, evaluation sautee) et reprise dans une 2e session avec RESUME = "auto" :
       scripts/train.py --model M --fold k --config C --seed S --tag T --epochs E ;
     la memoire GPU est relevee toutes les 30 s (nvidia-smi) dans runs/gpu_mem_<tag>_f<k>.csv ;
  5) la cellule suivante du notebook (phase3_ablation/eval_ablation.py) evalue les plis entraines.
Aucun secret n'est lu : le depot est public.
"""

import csv
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# =====================================================================
# A AJUSTER (remplace par launch_ablation.py)
# =====================================================================
GITHUB_USER = "NSSIRI"
GITHUB_REPO_NAME = "mrgrt-seg"
GITHUB_BRANCH = "ablation-18m"
REPO_COMMIT = "0a36a1e"
KAGGLE_INPUT_DATASET = "/kaggle/input/datasets/abdelhalimnssiri/mrgrt-oar-thorax-clean-v2"
MONAI_VERSION = "1.5.2"
TORCH_VERSION = "2.10.0"          # = environnement des entrainements E2/E3 (journaux : torch 2.10.0+cu128)
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
EPOCHS = 150
# Arret propre avant la limite de session Kaggle : last.pt (ecrit a chaque epoque) permet la reprise
MAX_TRAIN_HOURS = 10.5
# Reprise : None, ou "auto" = chercher <tag>_fold<k>/last.pt sous /kaggle/input (sortie de la session precedente)
RESUME = None
# Phase 4 : dossier monte du dataset de labels 6 OAR (None = phases 2-3, labels 4 OAR du dataset D187)
LABELS_6OAR = None
# Liste de jobs : 1 job = protocole standard ; 2 jobs = essai en parallele (1 par GPU, num_workers=2 chacun)
JOBS = [
    {"model": "unet", "config": "configs/ablation/unet_res.yaml", "fold": 0, "seed": 42, "tag": "unet_res_s42"},
]
# =====================================================================

WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / GITHUB_REPO_NAME
RUNS_DIR = WORK_DIR / "runs"
T0 = time.time()


def run(cmd, cwd=None, check=True):
    print("  $ " + " ".join(str(c) for c in cmd) + (f"   (cwd={cwd})" if cwd else ""), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd, check=check)


# 1) Code
print(f"[1] github.com/{GITHUB_USER}/{GITHUB_REPO_NAME} branche {GITHUB_BRANCH} @ {REPO_COMMIT}")
if not (REPO_DIR / ".git").exists():
    run(["git", "clone", "--quiet", "--branch", GITHUB_BRANCH,
         f"https://github.com/{GITHUB_USER}/{GITHUB_REPO_NAME}.git", REPO_DIR])
run(["git", "checkout", "--quiet", REPO_COMMIT], cwd=REPO_DIR)
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()
print(f"   HEAD = {head}")
if not head.startswith(REPO_COMMIT):
    raise RuntimeError(f"commit attendu {REPO_COMMIT}, obtenu {head}")

# 2) Dependances : torch et MONAI figes (torch installe AVANT tout import de torch dans ce noyau)
print(f"\n[2] torch=={TORCH_VERSION} (cu128) + monai=={MONAI_VERSION} (figes) + dependances de train_kaggle.py")
if "torch" in sys.modules:
    raise RuntimeError("torch deja importe dans ce noyau : impossible de changer de version")
run([sys.executable, "-m", "pip", "install", "-q", f"torch=={TORCH_VERSION}", "--index-url", TORCH_INDEX])
run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", f"monai=={MONAI_VERSION}"])
run([sys.executable, "-m", "pip", "install", "-q", "nibabel>=5.1", "SimpleITK>=2.3", "einops>=0.7",
     "pydicom>=2.4", "rt-utils>=1.2"])
import monai  # noqa: E402
import torch  # noqa: E402
print(f"   monai={monai.__version__} | torch={torch.__version__} | GPU : "
      + ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))
if monai.__version__ != MONAI_VERSION:
    raise RuntimeError(f"MONAI {monai.__version__} != {MONAI_VERSION}")
if not torch.__version__.startswith(TORCH_VERSION):
    raise RuntimeError(f"torch {torch.__version__} != {TORCH_VERSION}")
if not torch.cuda.is_available():
    raise RuntimeError("Pas de GPU : choisir l'accelerateur GPU T4 x2.")
if len(JOBS) > torch.cuda.device_count():
    raise RuntimeError(f"{len(JOBS)} jobs pour {torch.cuda.device_count()} GPU")

# 3) data/ et runs/
print(f"\n[3] data/ -> {KAGGLE_INPUT_DATASET}")
if not Path(KAGGLE_INPUT_DATASET).exists():
    raise RuntimeError(f"Dataset non monte : {KAGGLE_INPUT_DATASET}")
for link, target in [(REPO_DIR / "data", Path(KAGGLE_INPUT_DATASET)), (REPO_DIR / "runs", RUNS_DIR)]:
    target.mkdir(parents=True, exist_ok=True) if target == RUNS_DIR else None
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"{link} existe deja (dossier) : on ne le supprime pas")
    link.symlink_to(target)
n_patients = sum(1 for _ in Path(KAGGLE_INPUT_DATASET).glob("PATIENT_s*"))
print(f"   {n_patients} patients")
if n_patients != 187:
    raise RuntimeError(f"{n_patients} patients au lieu de 187 : les plis ne correspondraient pas")

# Phase 4 : data6/PATIENT_sXXXX/{image.nii -> dataset D187, label.nii.gz -> dataset de labels 6 OAR}
if LABELS_6OAR:
    import nibabel as nib
    import numpy as np
    root6 = Path(LABELS_6OAR)
    if not root6.exists():   # chemin de montage different : on cherche le dossier du dataset par son nom
        cands = [p for p in Path("/kaggle/input").rglob(root6.name) if p.is_dir()]
        if not cands:
            raise RuntimeError(f"dataset de labels 6 OAR non monte : {LABELS_6OAR}")
        root6 = cands[0]
    # fichiers a plat dans le dataset : PATIENT_sXXXX_label.nii.gz, que Kaggle peut decompresser en .nii
    src_lab = sorted(set(root6.rglob("PATIENT_s*_label.nii.gz")) | set(root6.rglob("PATIENT_s*_label.nii")))
    if not src_lab:
        raise RuntimeError(f"aucun label 6 OAR sous {root6}")
    data6 = REPO_DIR / "data6"
    data6.mkdir(exist_ok=True)
    import gzip
    for lab in src_lab:
        pid = lab.name.split("_label.nii")[0]
        img = Path(KAGGLE_INPUT_DATASET) / pid / "image.nii"
        hi, hl = nib.load(str(img)), nib.load(str(lab))
        if hi.shape[:3] != hl.shape[:3] or not np.allclose(hi.affine, hl.affine, atol=1e-3):
            raise RuntimeError(f"{pid} : grille du label 6 OAR differente de l'image")
        (data6 / pid).mkdir(exist_ok=True)
        if not (data6 / pid / "image.nii").exists():
            (data6 / pid / "image.nii").symlink_to(img)
        dst = data6 / pid / "label.nii.gz"            # nom attendu par configs/phase4 (label_filename)
        if not dst.exists():
            if lab.name.endswith(".gz"):
                dst.symlink_to(lab)
            else:                                     # label decompresse par Kaggle : recompresse localement
                with open(lab, "rb") as fi, gzip.open(dst, "wb", compresslevel=1) as fo:
                    fo.write(fi.read())
    if len({p.name for p in data6.iterdir() if p.is_dir()}) != 115:
        raise RuntimeError("data6 : nombre de patients different de 115")
    print(f"   data6 : {len(src_lab)} patients (labels 6 OAR, grilles verifiees)")


# 4) Memoire GPU (nvidia-smi toutes les 30 s)
def gpu_monitor(stop, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "gpu", "mem_used_mib", "mem_total_mib", "util_pct", "ram_used_gib"])
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
                with open("/proc/meminfo") as m:
                    mi = {l.split(":")[0]: int(l.split()[1]) for l in m}
                ram = (mi["MemTotal"] - mi["MemAvailable"]) / 2**20
                for line in out.strip().splitlines():
                    i, used, tot, util = [x.strip() for x in line.split(",")]
                    w.writerow([round(time.time() - T0), i, used, tot, util, round(ram, 2)])
                f.flush()
            except Exception as e:  # le suivi ne doit jamais arreter l'entrainement
                print(f"[gpu_monitor] {e}", flush=True)
            stop.wait(30)


# Reprise : copie de last.pt / best.pt / history.npz / run_meta.json de la session precedente
if RESUME == "auto":
    import shutil
    for job in JOBS:
        name = f"{job['tag']}_fold{job['fold']}"
        found = sorted(Path("/kaggle/input").rglob(f"{name}/last.pt"))
        if not found:
            raise RuntimeError(f"RESUME='auto' : {name}/last.pt introuvable sous /kaggle/input")
        dst = RUNS_DIR / name
        dst.mkdir(parents=True, exist_ok=True)
        for fn in ("last.pt", "best.pt", "history.npz", "run_meta.json"):
            if (found[0].parent / fn).exists():
                shutil.copy2(found[0].parent / fn, dst / fn)
        print(f"   reprise {name} depuis {found[0].parent}")

stop = threading.Event()
GPU_CSV = RUNS_DIR / ("gpu_mem_" + "+".join(f"{j['tag']}_f{j['fold']}" for j in JOBS) + ".csv")
threading.Thread(target=gpu_monitor, args=(stop, GPU_CSV), daemon=True).start()


# 5) Entrainement
def job_config(job):
    """2 jobs en parallele : copie de la config avec num_workers=2 (seule difference), sinon config inchangee."""
    if len(JOBS) == 1:
        return job["config"]
    import yaml
    cfg = yaml.safe_load((REPO_DIR / job["config"]).read_text())
    cfg["train"]["num_workers"] = 2
    out = RUNS_DIR / f"{job['tag']}_fold{job['fold']}_config_nw2.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out)


def pump(proc, prefix):
    for line in proc.stdout:
        print(prefix + line, end="", flush=True)


procs = []
for gpu, job in enumerate(JOBS):
    cmd = [sys.executable, "-u", "scripts/train.py", "--model", job["model"], "--fold", str(job["fold"]),
           "--config", job_config(job), "--seed", str(job["seed"]), "--tag", job["tag"],
           "--epochs", str(EPOCHS), "--resume", "auto"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    prefix = f"[{job['tag']}_f{job['fold']}] " if len(JOBS) > 1 else ""
    print(f"\n[5] GPU {gpu} : {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, cwd=REPO_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t = threading.Thread(target=pump, args=(p, prefix), daemon=True)
    t.start()
    procs.append((job, p, t))

codes, interrupted = [], False
deadline = T0 + MAX_TRAIN_HOURS * 3600
for job, p, t in procs:
    try:
        codes.append(p.wait(timeout=max(1.0, deadline - time.time())))
    except subprocess.TimeoutExpired:   # arret propre : last.pt de la derniere epoque terminee est conserve
        print(f"\n[!] {MAX_TRAIN_HOURS} h atteintes : arret de {job['tag']} pli {job['fold']} (reprise necessaire)",
              flush=True)
        p.terminate()
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
        codes.append(0)
        interrupted = True
    t.join()
stop.set()
A_REPRENDRE = RUNS_DIR / "A_REPRENDRE"
if interrupted:
    A_REPRENDRE.write_text("entrainement interrompu avant la fin : relancer avec RESUME='auto'\n")
elif A_REPRENDRE.exists():
    A_REPRENDRE.unlink()
print("=" * 60)
for (job, _, _), code in zip(procs, codes):
    rdir = RUNS_DIR / f"{job['tag']}_fold{job['fold']}"
    files = ", ".join(f"{f.name} ({f.stat().st_size / 2**20:.1f} Mo)" for f in sorted(rdir.glob("*")))
    print(f"{job['tag']} pli {job['fold']} : code {code} | {files}")
print(f"Entrainement : {(time.time() - T0) / 3600:.2f} h")
if any(codes):
    raise RuntimeError(f"train.py a echoue : codes {codes}")
