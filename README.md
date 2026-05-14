# MRgRT - Segmentation OAR Thorax (version locale PC)

Pour valider l'environnement avant de basculer sur HPC MARWAN.

## Demarrage rapide

1. Double-clic sur `setup_local.bat` (sur le Bureau, dans ce dossier)
   - Cree le venv `.venv`
   - Installe PyTorch CPU + MONAI + dependances
   - Lance `smoke_test.py` automatiquement

2. Si tout est vert, votre env est valide. Vous pouvez :
   - Lancer le notebook : `jupyter notebook notebooks/00_pedagogical_pipeline.ipynb`
   - Pousser le code sur git, puis basculer sur MARWAN pour entrainer

## Limitations sur PC sans GPU

- Pas d'entrainement reel (trop lent : plusieurs jours par fold)
- Smoke test : OK
- Inference sur 1-2 patients : lent mais OK
- Pour les vrais entrainements : HPC MARWAN

## Structure

```
mrgrt_seg/
  setup_local.bat       # creation venv + install + smoke test
  smoke_test.py         # validation env (autonome)
  requirements_cpu.txt  # deps Python
  configs/              # default.yaml (IRM), ct.yaml (CT)
  src/                  # code Python
    data/, models/, train/, eval/, xai/
  scripts/              # train.py, evaluate.py, etc.
```

## Reactivation du venv plus tard

```powershell
cd %USERPROFILE%\Desktop\mrgrt_seg
.venv\Scripts\activate
```
