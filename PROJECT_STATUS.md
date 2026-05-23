# PROJECT STATUS — MRgRT OAR Segmentation (état au 19 mai 2026)

> Ce fichier sert à reprendre le projet depuis n'importe quelle session/compte Claude.
> Coller ce fichier ou demander à Claude de le lire pour restaurer le contexte.

## Vue d'ensemble du projet

Thèse doctorale : segmentation deep learning des organes à risque (OAR) thoraciques
en radiothérapie guidée par IRM (MRgRT). Comparaison U-Net 3D vs SegResNet, avec
filtre qualité des données et explicabilité (SEG-GRAD-CAM 3D).

- **Repo GitHub** : github.com/NSSIRI/mrgrt-seg (privé)
- **Cluster HPC** : MARWAN (a.nssiri@hpc-login.marwan.ma)
- **Dataset source** : TotalSegmentator MRI v2.0.0 (Zenodo 10.5281/zenodo.14710732)
- **Backup compute** : Kaggle (notebook prêt dans scripts/kaggle/)

## CE QUI EST FAIT

### Données
- [x] Dataset TotalSeg MRI v2.0.0 téléchargé sur MARWAN (5.1 GB, MD5 vérifié)
- [x] Converti en NIfTI multi-classes (5 classes : bg + poumon_g/d, coeur, oesophage)
- [x] 616 patients bruts dans $SCRATCH/mrgrt-seg/downloads/raw_totalseg/
- [x] Filtre qualité appliqué : 303 patients retenus (sur 616)
  - Seuils : FOV cranio-caudal >= 120 mm, poumon >= 300 mL, coeur >= 50 mL,
    oesophage >= 5 mL, poumons ne touchent pas les bords
  - Dataset propre dans $SCRATCH/mrgrt-seg/data_thorax_complet/
  - Dataset complet (616) dans $SCRATCH/mrgrt-seg/data_full/ (backup)

### Code (tout sur GitHub, branche main)
- [x] scripts/train.py — entraînement, supporte --resume et --epochs
- [x] src/train/trainer.py — boucle training avec resume-from-checkpoint (last.pt + best.pt atomiques)
- [x] scripts/transfer/01_marwan_setup.sh — setup SCRATCH + symlink data/
- [x] scripts/transfer/03_marwan_verify.py — vérification intégrité dataset
- [x] scripts/transfer/04_marwan_download_and_prepare.sh — download Zenodo + convert
- [x] scripts/transfer/05_filter_quality.py — filtre qualité (boundary/volume/FOV)
- [x] scripts/slurm/train_one.sbatch — 1 fold, avec --qos=gpu et --account=gpu_users
- [x] scripts/slurm/train_5fold_array.sbatch — 5 folds en array
- [x] scripts/kaggle/ — notebook + README pour entraînement Kaggle GPU (backup)
- [x] paper/article_draft.md — draft article scientifique (Intro + Methods rédigés)

### Environnement MARWAN
- [x] Module : Anaconda3/2024.02-1 + CUDA/11.4.1
- [x] Env conda : mrgrt-seg (activé auto via ~/.bashrc)
- [x] Stack : torch 2.7.1+cu118, monai 1.5.2, numpy 2.4.6, scipy 1.17.1, nibabel 5.4.2
- [x] SLURM : sbatch dans /cm/shared/apps/slurm/current/bin (dans PATH via ~/.bashrc)
- [x] Partitions : gpu-prodq (7j, saturée), gpu-testq (1h max, plus rapide pour tests)
- [x] QOS requise : --qos=gpu, compte : --account=gpu_users

## CE QUI EST EN COURS

- [ ] Sanity-run GPU : sbatch train_one.sbatch unet 0 (--epochs 5) sur gpu-testq
  - Job 444388 en attente PD au dernier point connu
  - Objectif : valider que le pipeline tourne sur GPU avant le vrai entraînement

## CE QUI RESTE À FAIRE

1. [ ] Confirmer le sanity-run (loss descend, DSC monte, last.pt créé)
2. [ ] Switcher config sur data_thorax_complet (303 patients) pour le vrai entraînement
3. [ ] Vrai entraînement 5-fold U-Net (300 epochs) sur gpu-prodq
4. [ ] Vrai entraînement 5-fold SegResNet (300 epochs)
5. [ ] Ablation : ré-entraîner sur data_full (616) pour comparaison avant/après filtrage
6. [ ] Évaluation : DSC, HD95, Surface DSC, Wilcoxon + Bonferroni
7. [ ] Module XAI : SEG-GRAD-CAM 3D + sanity checks (Adebayo cascading randomization)
8. [ ] Remplir Results + Discussion dans paper/article_draft.md
9. [ ] Figures : flowchart 616->303, box plots, saliency maps

## COMMANDES CLÉS POUR REPRENDRE

```bash
# Connexion MARWAN
ssh a.nssiri@hpc-login.marwan.ma
cd ~/mrgrt-seg && git pull

# L'env conda s'active auto (dans .bashrc). Sinon :
module load Anaconda3/2024.02-1 && source activate mrgrt-seg
export SCRATCH=/scratch/users/a.nssiri

# Vérifier l'état des jobs
squeue -u a.nssiri

# Lancer un sanity-run rapide (gpu-testq, ~30 min)
sbatch --partition=gpu-testq --time=00:55:00 scripts/slurm/train_one.sbatch unet 0 configs/default.yaml --epochs 5

# Lancer le vrai entraînement (gpu-prodq, attente longue mais job 10h+)
sbatch scripts/slurm/train_one.sbatch unet 0 configs/default.yaml

# Suivre les logs
tail -f logs/train_mrgrt_train_*.out
```

## POINTS DE VIGILANCE / DÉCISIONS EN ATTENTE

- **Choix dataset training** : utiliser data_thorax_complet (303, propre) comme principal,
  et data_full (616) pour l'ablation comparative. Modifier data.root dans configs/default.yaml.
- **gpu-prodq saturée** : 96 jobs en queue, 2 GPU. Utiliser gpu-testq pour les tests rapides.
- **Article** : titre + journal à valider avec l'encadrant (Physica Medica recommandé).
- **Seuils filtre qualité** : défendables mais à discuter avec l'encadrant.
- **2nd baseline** : SegResNet choisi ; envisager nnU-Net mais ouvre critique "auto-config vs vanilla".

## CONTEXTE POUR CLAUDE (nouvelle session)

Si tu lis ceci dans une nouvelle session : le projet est un pipeline complet de
segmentation OAR thoracique MRgRT. L'utilisateur (Abdelhalim Nssiri, doctorant en
physique médicale) travaille sur MARWAN (HPC) avec Kaggle en backup. Le dossier local
est sur son PC (Windows), synchronisé via GitHub. Communique en français. La prochaine
étape concrète est de confirmer le sanity-run GPU puis lancer le vrai entraînement 5-fold.
