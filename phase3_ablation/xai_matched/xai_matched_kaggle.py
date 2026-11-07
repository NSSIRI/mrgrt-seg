# -*- coding: utf-8 -*-
"""Seg-Grad-CAM 3D a couches APPARIEES pour E3 (U-Net 18,8 M) et E2 (SegResNet 18,8 M).

Pourquoi : l'analyse XAI precedente (scripts/run_xai_analysis.py, target_layer="auto") visait la
derniere Conv3d, soit le goulot pour l'U-Net (384->832, 1/16) et la tete de sortie 1x1 pour
SegResNet (conv_final). Les deux modeles n'etaient donc pas compares au meme niveau.

Ici, une seule passe avant/arriere par classe capture TOUS les niveaux a la fois (hooks de tenseur) :
  enc_profond  : derniere etape de l'encodeur  (U-Net : bloc du fond, 1/16 ; SegResNet : down_layers.3, 1/8)
  dec_demi     : decodeur a 1/2 de resolution (U-Net : model.1 = concat skip + up ; SegResNet : up_layers.1)
  dec_pre_tete : derniere carte avant la projection de sortie
                 (U-Net : model.1, 1/2, car la tete model.2 est la ConvTranspose finale ; SegResNet : up_layers.2, 1/1)
  sortie       : logits (U-Net : model.2 ; SegResNet : conv_final) = reglage de l'ancienne analyse SegResNet
Pour l'U-Net, dec_demi et dec_pre_tete designent le meme module (structure MONAI) : c'est voulu et documente.

Formulation Seg-Grad-CAM inchangee (Vinogradova et al. 2020) : score = somme des logits de la classe c sur
le masque predit de c ; poids = moyenne spatiale des gradients ; ReLU ; interpolation trilineaire ; min-max.
Recadrage 128x128x64 centre sur la boite englobante des 4 OAR de reference (l'ancien recadrage etait centre
sur le volume, d'ou ~12 % d'organes hors champ) ; la fraction de chaque organe dans le recadrage est notee.
Metriques : memes definitions que src/xai/metrics.py (copiees ici pour ne pas dependre de la version locale
non commitee), centroid_distance en mm (espacement 1,5 x 1,5 x 3).
Test d'Adebayo : randomisation en cascade (Conv3d ET ConvTranspose3d, de la sortie vers l'entree),
SSIM (meme formule que src/xai/adebayo.py) + correlation de Spearman, sur le 1er patient de chaque pli.

Lecture seule des donnees et des checkpoints. Sorties dans XAI_OUT (defaut /kaggle/working/xai_matched).
"""
import csv
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

# =============================== PARAMETRES ===============================
INPUT_ROOT = Path(os.environ.get("XAI_INPUT_ROOT", "/kaggle/input"))          # dataset
CKPT_ROOT = Path(os.environ.get("XAI_CKPT_ROOT", str(INPUT_ROOT)))           # sorties des notebooks d'entrainement
OUT = Path(os.environ.get("XAI_OUT", "/kaggle/working/xai_matched"))
REPO_DIR = Path(os.environ.get("XAI_REPO_DIR", "/kaggle/working/mrgrt-seg"))
REPO_URL = "https://github.com/NSSIRI/mrgrt-seg.git"
REPO_COMMIT = os.environ.get("XAI_REPO_COMMIT", "0a36a1e")
MONAI_VERSION = "1.5.2"
SKIP_INSTALL = os.environ.get("XAI_SKIP_INSTALL", "0") == "1"
SPACING = (1.5, 1.5, 3.0)
PATCH = (128, 128, 64)
MAX_PATIENTS = int(os.environ["XAI_MAX_PATIENTS"]) if os.environ.get("XAI_MAX_PATIENTS") else None
ONLY_FOLDS = [int(x) for x in os.environ["XAI_ONLY_FOLDS"].split(",")] if os.environ.get("XAI_ONLY_FOLDS") else None
RANDOM_INIT = os.environ.get("XAI_RANDOM_INIT", "")        # test local : "unet_opt,segresnet" sans checkpoint
DO_ADEBAYO = os.environ.get("XAI_ADEBAYO", "1") == "1"
# sous-ensemble de patients (ex. figures thoraciques) ; vide = tous les patients du pli (comportement d'origine)
ONLY_PATIENTS = [p for p in os.environ.get("XAI_PATIENTS", "").split(",") if p]
EXAMPLE_ALL = os.environ.get("XAI_EXAMPLE_ALL", "0") == "1"   # sauver les cartes de CHAQUE patient traite
VARIANTS = os.environ.get("XAI_VARIANTS", "unet_opt,segresnet").split(",")   # E3, E2
N_EXPECTED = int(os.environ.get("XAI_N_PATIENTS", "187"))
# Phase 4 (6 OAR, cohorte D115) : labels 6 OAR + images D187, checkpoints avec run_meta.json
LABELS6 = os.environ.get("XAI_LABELS6", "")                # dossier (ou dataset monte) des labels 6 OAR ; vide = phase 2
D187_ROOT = Path(os.environ.get("XAI_D187", "/kaggle/input/datasets/abdelhalimnssiri/mrgrt-oar-thorax-clean-v2"))
# ==========================================================================

ORGANS = ["poumon_g", "poumon_d", "coeur", "oesophage"]   # classes 1..4
if os.environ.get("XAI_ORGANS"):                           # phase 4 : "...,oesophage,moelle,aorte"
    ORGANS = os.environ["XAI_ORGANS"].split(",")
NUM_CLASSES = len(ORGANS) + 1
LEVELS = ["enc_profond", "dec_demi", "dec_pre_tete", "sortie"]
LAYERS = {
    "unet": {"enc_profond": "model.1.submodule.1.submodule.1.submodule.1.submodule",
             "dec_demi": "model.1", "dec_pre_tete": "model.1", "sortie": "model.2"},
    "segresnet": {"enc_profond": "down_layers.3", "dec_demi": "up_layers.1",
                  "dec_pre_tete": "up_layers.2", "sortie": "conv_final"},
}
VARIANT_FEATURES = {"unet_base": [32, 64, 128, 256, 512], "unet_opt": [48, 96, 192, 384, 832]}
SEGRESNET_KW = {"init_filters": 32, "blocks_down": [1, 2, 2, 4], "blocks_up": [1, 1, 1], "dropout_prob": 0.2}
EXAMPLE_LEVELS = ["enc_profond", "dec_pre_tete", "sortie"]
EXAMPLE_ORGANS = [3, 4] + list(range(5, NUM_CLASSES))      # coeur, oesophage (+ moelle, aorte en phase 4)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "examples").mkdir(exist_ok=True)
T0 = time.time()


def log_error(msg):
    print(f"[ERREUR] {msg}", flush=True)
    with open(OUT / "errors.log", "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def sh(cmd, cwd=None):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd).returncode


print("\n[1] Dependances", flush=True)
if not SKIP_INSTALL:
    if sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", f"monai=={MONAI_VERSION}"]) != 0:
        sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "monai"])
    for mod, pkg in [("nibabel", "nibabel"), ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("yaml", "pyyaml")]:
        try:
            __import__(mod)
        except ImportError:
            sh([sys.executable, "-m", "pip", "install", "-q", pkg])

print("\n[2] Code du projet", flush=True)
if not (REPO_DIR / "src").exists():
    if sh(["git", "clone", "--quiet", REPO_URL, REPO_DIR]) != 0:
        raise SystemExit("Clonage GitHub impossible : verifier que Internet est active.")
    if sh(["git", "checkout", "--quiet", REPO_COMMIT], cwd=REPO_DIR) != 0:
        raise SystemExit(f"commit {REPO_COMMIT} introuvable")
REPO_HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True,
                           text=True).stdout.strip() or "copie-locale"
sys.path.insert(0, str(REPO_DIR))
print(f"   code : {REPO_DIR} @ {REPO_HEAD}", flush=True)

import numpy as np                          # noqa: E402
import torch                                # noqa: E402
import torch.nn as nn                       # noqa: E402
import torch.nn.functional as F             # noqa: E402
import scipy.ndimage as ndi                 # noqa: E402
from scipy.stats import spearmanr           # noqa: E402
import monai                                # noqa: E402

from src.data.dataset import list_patients, get_val_transforms   # noqa: E402
from src.data.splits import make_5fold_splits                     # noqa: E402
from src.models.factory import build_model, count_parameters      # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[3] Device : {DEVICE} {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else ''} | "
      f"torch {torch.__version__} | monai {monai.__version__}", flush=True)


# ---------------------------------------------------------------- dataset + plis
def find_dataset(root):
    best, best_n, frontier = None, 0, [root]
    for _ in range(5):
        nxt = []
        for d in frontier:
            try:
                subs = [p for p in d.iterdir() if p.is_dir()]
            except (PermissionError, FileNotFoundError):
                continue
            n = sum(1 for p in subs if p.name.startswith("PATIENT_s"))
            if n > best_n:
                best, best_n = d, n
            nxt.extend(p for p in subs if not p.name.startswith("PATIENT_s"))
        frontier = nxt
    if best is None:
        raise SystemExit(f"Aucun dossier PATIENT_s* sous {root}")
    return best


def _link(src, dst):
    try:
        dst.symlink_to(src)
    except OSError:          # Windows sans droit de lien symbolique (test local)
        import shutil
        shutil.copy2(src, dst)


def build_data6(labels_root, d187, dest):
    """Comme la cellule d'entrainement de phase 4 : data6/PATIENT/{image -> D187, label.nii.gz -> labels 6 OAR}.
    Accepte les labels a plat (PATIENT_sXXXX_label.nii[.gz], dataset Kaggle) ou par dossier (PATIENT_sXXXX/label.nii.gz,
    data_6oar/ local). Grilles image/label verifiees."""
    import gzip
    import nibabel as nib
    root6 = Path(labels_root)
    if not root6.exists():
        cands = [p for p in INPUT_ROOT.rglob(root6.name) if p.is_dir()]
        if not cands:
            raise SystemExit(f"labels 6 OAR non montes : {labels_root}")
        root6 = cands[0]
    src = {lab.name.split("_label.nii")[0]: lab for lab in
           sorted(set(root6.rglob("PATIENT_s*_label.nii.gz")) | set(root6.rglob("PATIENT_s*_label.nii")))}
    src.update({lab.parent.name: lab for lab in sorted(root6.glob("PATIENT_s*/label.nii.gz"))})
    if not src:
        raise SystemExit(f"aucun label 6 OAR sous {root6}")
    dest.mkdir(parents=True, exist_ok=True)
    for pid, lab in src.items():
        img = next((d187 / pid / n for n in ("image.nii", "image.nii.gz") if (d187 / pid / n).exists()), None)
        if img is None:
            raise SystemExit(f"{pid} : image absente de {d187}")
        hi, hl = nib.load(str(img)), nib.load(str(lab))
        if hi.shape[:3] != hl.shape[:3] or not np.allclose(hi.affine, hl.affine, atol=1e-3):
            raise SystemExit(f"{pid} : grille du label 6 OAR differente de l'image")
        (dest / pid).mkdir(exist_ok=True)
        li = dest / pid / img.name
        if not li.exists():
            _link(img.resolve(), li)
        dst = dest / pid / "label.nii.gz"
        if not dst.exists():
            if lab.name.endswith(".gz"):
                _link(lab.resolve(), dst)
            else:
                with open(lab, "rb") as fi, gzip.open(dst, "wb", compresslevel=1) as fo:
                    fo.write(fi.read())
    print(f"   data6 : {len(src)} patients ({root6})", flush=True)
    return dest


if LABELS6:
    DATA = build_data6(LABELS6, find_dataset(D187_ROOT) if D187_ROOT.exists() else find_dataset(INPUT_ROOT),
                       OUT.parent / "data6")
else:
    DATA = find_dataset(INPUT_ROOT)
sample =sorted(p for p in DATA.iterdir() if p.name.startswith("PATIENT_s"))[0]
IMG_FN = "image.nii" if (sample / "image.nii").exists() else "image.nii.gz"
LBL_FN = "label.nii" if (sample / "label.nii").exists() else "label.nii.gz"
items = list_patients(str(DATA), IMG_FN, LBL_FN)
folds = make_5fold_splits(items, n_folds=5, seed=42)
VAL = {f["fold"]: f["val"] for f in folds}
print(f"\n[4] Dataset : {DATA} ({len(items)} patients) ; plis val = {[len(VAL[k]) for k in sorted(VAL)]}", flush=True)
if len(items) != N_EXPECTED:
    log_error(f"{len(items)} patients au lieu de {N_EXPECTED} : plis differents des CSV de reference")
by_pid = {it["patient_id"]: it for it in items}
val_tf = get_val_transforms(spacing=list(SPACING), intensity_params={"clip_percentiles": [0.5, 99.5]},
                            modality="mri")


# ---------------------------------------------------------------- checkpoints
def detect_variant(sd):
    if "convInit.conv.weight" in sd:
        return "segresnet"
    w = sd.get("model.0.conv.weight")
    if w is not None:
        return {32: "unet_base", 48: "unet_opt"}.get(int(w.shape[0]), f"unet_c{int(w.shape[0])}")
    return "unknown"


def make_model(variant, meta=None):
    if meta:   # phase 4 : reconstruction exacte depuis run_meta.json
        return build_model(meta["model"], meta["in_channels"], meta["out_channels"], features=meta["features"],
                           segresnet_kwargs=meta["segresnet_kwargs"], unet_kwargs=meta["unet_kwargs"])
    if variant == "segresnet":
        return build_model("segresnet", 1, NUM_CLASSES, segresnet_kwargs=dict(SEGRESNET_KW))
    return build_model("unet", 1, NUM_CLASSES, features=VARIANT_FEATURES[variant])


runs = {}
print("\n[5] Checkpoints", flush=True)
if RANDOM_INIT:
    for v in RANDOM_INIT.split(","):
        for k in (ONLY_FOLDS or [0]):
            runs[(v, k)] = {"variant": v, "fold": k, "path": "<init aleatoire>", "epoch": None, "val_dsc": None}
            if LABELS6:   # test local phase 4 : architecture lue dans un run_meta.json (poids non telecharges)
                metas = [json.loads(q.read_text(encoding="utf-8")) for q in CKPT_ROOT.rglob("run_meta.json")]
                metas = [mt for mt in metas if mt["tag"].split("_s")[0] == v and int(mt["fold"]) == k]
                if not metas:
                    raise SystemExit(f"aucun run_meta.json pour {v} pli {k} sous {CKPT_ROOT}")
                runs[(v, k)]["meta"] = metas[0]
else:
    seen = set()
    for p in sorted(CKPT_ROOT.rglob("best.pt"), key=lambda q: -q.stat().st_mtime):
        if p.resolve() in seen:
            continue
        seen.add(p.resolve())
        meta_path = p.parent / "run_meta.json"
        if LABELS6:   # phase 4 : uniquement des runs avec run_meta.json, pli de validation verifie
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            v, k = meta["tag"].split("_s")[0], int(meta["fold"])
            if v not in VARIANTS:
                continue
            if int(meta.get("split_seed", -1)) != 42 or meta.get("val_patients") != VAL.get(k):
                log_error(f"{p} : pli de validation de run_meta.json different des plis recalcules -> ignore")
                continue
            if meta.get("out_channels") != NUM_CLASSES:
                log_error(f"{p} : {meta.get('out_channels')} sorties au lieu de {NUM_CLASSES} -> ignore")
                continue
            if (v, k) not in runs:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                runs[(v, k)] = {"variant": v, "fold": k, "path": str(p), "meta": meta,
                                "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
                                "val_dsc": ck.get("val_dsc") if isinstance(ck, dict) else None}
                del ck
            continue
        m = re.fullmatch(r"(unet|segresnet)_fold(\d+)", p.parent.name)
        if not m or meta_path.exists():    # phase 2 uniquement (pas de run_meta.json)
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            log_error(f"lecture impossible {p} : {e}")
            continue
        sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
        v, k = detect_variant(sd), int(m.group(2))
        if v not in VARIANTS or not v.startswith(m.group(1)):
            print(f"   ignore {v} {p}")
            continue
        if (v, k) not in runs:
            runs[(v, k)] = {"variant": v, "fold": k, "path": str(p),
                            "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
                            "val_dsc": ck.get("val_dsc") if isinstance(ck, dict) else None}
        del ck, sd
if ONLY_FOLDS is not None:
    runs = {key: r for key, r in runs.items() if key[1] in ONLY_FOLDS}
for r in runs.values():
    print(f"   {r['variant']:<10} fold{r['fold']}  epoch={r['epoch']} val_dsc={r['val_dsc']}  {r['path']}", flush=True)
if not runs:
    raise SystemExit("Aucun best.pt trouve : ajouter les notebooks d'entrainement en Input.")


# ---------------------------------------------------------------- Seg-Grad-CAM multi-niveaux
class MultiLevelCAM:
    def __init__(self, model, layer_names):
        self.model, self.names = model, layer_names
        mods = dict(model.named_modules())
        missing = [n for n in set(layer_names.values()) if n not in mods]
        if missing:
            raise ValueError(f"couches introuvables : {missing}")
        self.acts, self.grads, self.handles = {}, {}, []
        for n in sorted(set(layer_names.values())):
            self.handles.append(mods[n].register_forward_hook(self._hook(n)))

    def _hook(self, name):
        def f(_m, _i, out):
            self.acts[name] = out
            if out.requires_grad:
                out.register_hook(lambda g: self.grads.__setitem__(name, g))
        return f

    def close(self):
        for h in self.handles:
            h.remove()

    @torch.enable_grad()
    def __call__(self, x, classes):
        """x : (1,1,D,H,W). Retourne {classe: {niveau: carte (D,H,W) numpy}}, {classe: n voxels predits}, shapes."""
        self.model.zero_grad(set_to_none=True)
        self.acts, self.grads = {}, {}
        logits = self.model(x)
        pred = logits.argmax(1, keepdim=True)
        shapes = {lv: list(self.acts[n].shape[1:]) for lv, n in self.names.items()}
        out, npred = {}, {}
        for i, c in enumerate(classes):
            self.grads = {}
            mask = (pred == c).float()
            npred[c] = int(mask.sum().item())
            score = (logits[:, c:c + 1] * mask).sum()
            score.backward(retain_graph=i < len(classes) - 1)
            out[c] = {}
            for lv, n in self.names.items():
                a, g = self.acts[n], self.grads.get(n)
                if g is None:
                    out[c][lv] = np.zeros(x.shape[2:], np.float32)
                    continue
                w = g.mean(dim=(2, 3, 4), keepdim=True)
                cam = F.relu((w * a).sum(1, keepdim=True)).detach()
                cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0]
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                out[c][lv] = cam.float().cpu().numpy()
        del logits
        return out, npred, shapes


# ---------------------------------------------------------------- metriques (= src/xai/metrics.py)
def in_organ_ratio(h, g):
    t = float(h.sum())
    return float(h[g].sum()) / t if t >= 1e-12 else float("nan")


def pointing_accuracy(h, g):
    return float(g[np.unravel_index(np.argmax(h), h.shape)]) if g.any() else float("nan")


def spatial_entropy(h):
    h = h.astype(np.float64).ravel()
    t = h.sum()
    if t < 1e-12:
        return float("nan")
    p = h / t
    pp = p[p > 0]
    n = pp.size
    return 0.0 if n <= 1 else float(-(pp * np.log(pp)).sum() / np.log(n))


def boundary_energy_ratio(h, g, band=2):
    t = float(h.sum())
    if not g.any() or t < 1e-12:
        return float("nan")
    b = ndi.binary_dilation(g, iterations=band) & ~ndi.binary_erosion(g, iterations=band)
    return float(h[b].sum()) / t


def activation_dice(h, g, pct=90.0):
    nz = h[h > 0]
    if not g.any() or nz.size < 10:
        return float("nan")
    a = h >= np.percentile(nz, pct)
    return 2.0 * int((a & g).sum()) / (int(a.sum()) + int(g.sum()) + 1e-12)


def centroid_distance_mm(h, g, spacing=SPACING):
    t = float(h.sum())
    if not g.any() or t < 1e-12:
        return float("nan")
    idx = np.indices(h.shape, dtype=np.float64)
    cs = np.array([(idx[i] * h).sum() / t for i in range(3)])
    cg = np.array([idx[i][g].mean() for i in range(3)])
    return float(np.linalg.norm((cs - cg) * np.array(spacing)))


def all_metrics(h, g):
    h = h.astype(np.float64)
    return {"in_organ_ratio": in_organ_ratio(h, g), "pointing_accuracy": pointing_accuracy(h, g),
            "spatial_entropy": spatial_entropy(h), "boundary_energy_ratio": boundary_energy_ratio(h, g),
            "activation_dice": activation_dice(h, g), "centroid_distance_mm": centroid_distance_mm(h, g),
            "organ_baseline": float(g.sum() / g.size)}


def ssim_3d(a, b):   # = src/xai/adebayo.py
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    ma, mb, va, vb = a.mean(), b.mean(), a.var(), b.var()
    cov = float(((a - ma) * (b - mb)).mean())
    L = max(a.max() - a.min(), b.max() - b.min(), 1e-12)
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    den = (ma ** 2 + mb ** 2 + C1) * (va + vb + C2)
    return float((2 * ma * mb + C1) * (2 * cov + C2) / den) if den > 1e-12 else 0.0


# ---------------------------------------------------------------- donnees
def load_crop(pid):
    d = val_tf(dict(by_pid[pid]))
    img = d["image"][0].numpy().astype(np.float32)
    lab = d["label"][0].numpy().astype(np.int16)
    pad = [(0, max(0, p - s)) for s, p in zip(img.shape, PATCH)]
    img, lab = np.pad(img, pad), np.pad(lab, pad)
    # centre = boite des 4 OAR thoraciques (classes 1-4) : identique a la phase 2 ; en phase 4, la moelle et
    # l'aorte descendent souvent dans l'abdomen et deplaceraient le recadrage hors du thorax
    fg = np.argwhere((lab > 0) & (lab <= 4))
    c = (fg.min(0) + fg.max(0)) // 2 if len(fg) else np.array(img.shape) // 2
    st = [int(min(max(ci - p // 2, 0), s - p)) for ci, p, s in zip(c, PATCH, img.shape)]
    sl = tuple(slice(s, s + p) for s, p in zip(st, PATCH))
    frac = {k: (float((lab[sl] == k).sum()) / max(1, int((lab == k).sum()))) for k in range(1, NUM_CLASSES)}
    return img[sl], lab[sl], frac


# ---------------------------------------------------------------- reprise
LONG = OUT / "xai_matched_long.csv"
ADEB = OUT / "adebayo_matched_long.csv"
FIELDS = ["variant", "fold", "patient_id", "level", "layer", "feat_shape", "organ", "n_gt_crop", "frac_gt_in_crop",
          "n_pred_crop", "in_organ_ratio", "pointing_accuracy", "spatial_entropy", "boundary_energy_ratio",
          "activation_dice", "centroid_distance_mm", "organ_baseline"]
done = set()
if LONG.exists():
    with open(LONG, newline="", encoding="utf-8") as f:
        done = {(r["variant"], int(r["fold"]), r["patient_id"]) for r in csv.DictReader(f)}


def append(path, fields, rows):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


ADE_FIELDS = ["variant", "fold", "patient_id", "organ", "step", "layer_randomized", "level", "n_pred_crop",
              "ssim_vs_pristine", "spearman_vs_pristine"]


def adebayo(model, family, variant, fold, pid, x, cls=4):
    import copy
    m = copy.deepcopy(model).eval()
    cam = MultiLevelCAM(m, LAYERS[family])
    try:
        ref, npred, _ = cam(x, [cls])
        ref = ref[cls]
        rows = [{"variant": variant, "fold": fold, "patient_id": pid, "organ": ORGANS[cls - 1], "step": 0,
                 "layer_randomized": "<pristine>", "level": lv, "n_pred_crop": npred[cls],
                 "ssim_vs_pristine": 1.0, "spearman_vs_pristine": 1.0} for lv in LEVELS]
        layers = [(n, mod) for n, mod in m.named_modules() if isinstance(mod, (nn.Conv3d, nn.ConvTranspose3d))][::-1]
        gen = torch.Generator(device=next(m.parameters()).device).manual_seed(42)
        sub = np.random.default_rng(0).choice(x[0, 0].numel(), 200_000, replace=False)
        for i, (n, mod) in enumerate(layers, start=1):
            nn.init.kaiming_uniform_(mod.weight, a=0, mode="fan_in", nonlinearity="relu", generator=gen)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
            cur, npred, _ = cam(x, [cls])
            for lv in LEVELS:
                a, b = ref[lv], cur[cls][lv]
                rho = spearmanr(a.ravel()[sub], b.ravel()[sub]).statistic if b.std() > 0 and a.std() > 0 else float("nan")
                rows.append({"variant": variant, "fold": fold, "patient_id": pid, "organ": ORGANS[cls - 1],
                             "step": i, "layer_randomized": n, "level": lv, "n_pred_crop": npred[cls],
                             "ssim_vs_pristine": ssim_3d(a, b), "spearman_vs_pristine": float(rho)})
        return rows
    finally:
        cam.close()
        del m
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- boucle
info = {"repo_head": REPO_HEAD, "torch": torch.__version__, "monai": monai.__version__, "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, "patch": PATCH, "spacing": SPACING,
        "layers": LAYERS, "runs": {}}
print(f"\n[6] XAI de {len(runs)} run(s)", flush=True)
for (variant, fold), run in sorted(runs.items()):
    t_run = time.time()
    meta = run.get("meta")
    family = meta["model"] if meta else ("segresnet" if variant == "segresnet" else "unet")
    try:
        model = make_model(variant, meta)
        if meta and count_parameters(model) != meta.get("n_params"):
            raise ValueError(f"{count_parameters(model)} parametres != run_meta.json ({meta.get('n_params')})")
        if run["path"] != "<init aleatoire>":
            ck = torch.load(run["path"], map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model_state"] if "model_state" in ck else ck, strict=True)
            del ck
        model = model.to(DEVICE).eval()
        for p in model.parameters():
            p.requires_grad_(True)
        cam = MultiLevelCAM(model, LAYERS[family])
    except Exception as e:
        log_error(f"{variant} fold{fold} : chargement impossible : {e}\n{traceback.format_exc()}")
        continue
    pids = VAL[fold][:MAX_PATIENTS] if MAX_PATIENTS else VAL[fold]
    if ONLY_PATIENTS:
        pids = [p for p in pids if p in ONLY_PATIENTS]
        if not pids:
            print(f"   {variant} fold{fold} : aucun patient demande dans ce pli", flush=True)
            cam.close()
            continue
    print(f"\n=== {variant} fold{fold} : {len(pids)} patients, {count_parameters(model):,} parametres ===", flush=True)
    shapes_run = None
    for j, pid in enumerate(pids):
        if (variant, fold, pid) in done:
            continue
        t = time.time()
        try:
            img, lab, frac = load_crop(pid)
            x = torch.from_numpy(img)[None, None].to(DEVICE)
            present = [c for c in range(1, NUM_CLASSES) if (lab == c).any()]
            maps, npred, shapes = cam(x, present)
            shapes_run = shapes
            rows = []
            for c in range(1, NUM_CLASSES):
                g = lab == c
                for lv in LEVELS:
                    base = {"variant": variant, "fold": fold, "patient_id": pid, "level": lv,
                            "layer": LAYERS[family][lv], "feat_shape": "x".join(map(str, shapes[lv])),
                            "organ": ORGANS[c - 1], "n_gt_crop": int(g.sum()), "frac_gt_in_crop": round(frac[c], 4),
                            "n_pred_crop": npred.get(c, 0)}
                    if c in maps:
                        base.update(all_metrics(maps[c][lv], g))
                    rows.append(base)
            append(LONG, FIELDS, rows)
            if j == 0 or EXAMPLE_ALL:   # exemples qualitatifs (1er patient du pli, ou tous si EXAMPLE_ALL)
                np.savez_compressed(OUT / "examples" / f"{variant}_fold{fold}_{pid}.npz",
                                    image=img.astype(np.float16), label=lab.astype(np.uint8),
                                    **{f"{ORGANS[c - 1]}__{lv}": maps[c][lv].astype(np.float16)
                                       for c in EXAMPLE_ORGANS if c in maps for lv in EXAMPLE_LEVELS})
                if DO_ADEBAYO and 4 in present:
                    t_a = time.time()
                    cam.close()   # sinon deepcopy recopie les hooks du modele original
                    try:
                        append(ADEB, ADE_FIELDS, adebayo(model, family, variant, fold, pid, x))
                    finally:
                        cam = MultiLevelCAM(model, LAYERS[family])
                    print(f"   Adebayo {pid} : {time.time() - t_a:.0f} s", flush=True)
            io = {lv: np.nanmean([r.get("in_organ_ratio", np.nan) for r in rows if r["level"] == lv]) for lv in LEVELS}
            print(f"  [{j + 1}/{len(pids)}] {pid} {time.time() - t:5.1f} s  in-organ moyen : " +
                  "  ".join(f"{lv}={v:.3f}" for lv, v in io.items()), flush=True)
        except Exception as e:
            log_error(f"{variant} fold{fold} {pid} : {e}\n{traceback.format_exc()}")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    cam.close()
    info["runs"][f"{variant}_fold{fold}"] = {"checkpoint": run["path"], "best_epoch": run["epoch"],
                                             "best_val_dsc": run["val_dsc"], "n_params": count_parameters(model),
                                             "feat_shapes": shapes_run, "seconds": round(time.time() - t_run, 1)}
    (OUT / "run_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

info["total_seconds"] = round(time.time() - T0, 1)
(OUT / "run_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
print(f"\nFINI en {info['total_seconds'] / 60:.1f} min -> {OUT}", flush=True)
