# Transfert du dataset vers MARWAN — procédure

Objectif : déposer les 250 patients (~1.85 GB) du dossier `data/` local vers
`$SCRATCH/mrgrt-seg/data/` sur MARWAN, avec un symlink `~/mrgrt-seg/data` qui
pointe dessus. Les scripts SLURM utilisent `data/` en chemin relatif et donc
fonctionneront tels quels après le symlink.

## Pourquoi $SCRATCH et pas $HOME ?

Sur la plupart des clusters HPC (MARWAN inclus), `$HOME` a un quota petit
(souvent 10–50 GB), est sauvegardé, et est plus lent en I/O. `$SCRATCH` est
prévu pour les gros datasets et l'I/O d'entraînement : quota large, plus
rapide, non sauvegardé (ce n'est pas grave : les NIfTI sont sur ton PC et GitHub).

## Pré-requis

- Côté local (PC) : WSL2 Ubuntu avec `rsync` installé. Tester : `wsl rsync --version`.
- Côté MARWAN : compte actif, accès SSH OK, `$SCRATCH` défini.
- Côté GitHub : code déjà cloné sur MARWAN dans `~/mrgrt-seg/`.

## Procédure en 3 étapes

### Étape 1 — Préparer MARWAN (1 fois)

Sur **MARWAN** (via SSH ou MobaXterm) :

```bash
bash scripts/transfer/01_marwan_setup.sh
```

Ce script :
- vérifie que `$SCRATCH` est défini ;
- crée `$SCRATCH/mrgrt-seg/data/` ;
- crée le symlink `~/mrgrt-seg/data → $SCRATCH/mrgrt-seg/data` ;
- affiche le chemin résolu de `$SCRATCH` (à noter pour l'étape 2).

### Étape 2 — Transférer depuis le PC (WSL2)

Sur le **PC local**, ouvrir WSL2 Ubuntu :

```bash
# Renseigner les 3 variables une fois pour toutes
export MARWAN_USER="ton_login_marwan"
export MARWAN_HOST="hpc.marwan.ma"          # à ajuster si différent
export MARWAN_SCRATCH="/scratch/$MARWAN_USER"  # valeur affichée par l'étape 1

# Lancer
bash /mnt/c/Users/Lenovo/Desktop/mrgrt_seg/scripts/transfer/02_local_rsync.sh
```

Le script utilise `rsync -avP --partial` :
- `-a` : préserve permissions, timestamps, etc.
- `-v` : verbeux
- `-P` : affiche la progression, et reprend les fichiers partiels en cas de coupure
- `--partial` : garde les transferts incomplets pour reprise

Tu peux **interrompre (Ctrl+C) et relancer** : rsync reprend où il s'était arrêté.

Temps estimé pour 1.85 GB : 5–25 min selon ton upload (10–50 Mbps).

### Étape 3 — Vérifier sur MARWAN

Sur **MARWAN** :

```bash
cd ~/mrgrt-seg
source activate mrgrt-seg
python scripts/transfer/03_marwan_verify.py
```

Le script vérifie :
- nombre de patients (attendu : 250) ;
- présence des 2 fichiers (`image.nii.gz`, `label.nii.gz`) pour chacun ;
- intégrité NIfTI : chaque fichier s'ouvre, dimensions cohérentes ;
- distribution des classes dans les labels (sanity check).

Si tout est vert, le dataset est prêt pour `sbatch scripts/slurm/train_one.sbatch unet 0`.

## Dépannage

**rsync : Permission denied**
→ Mauvais login ou clé SSH non configurée. Tester d'abord `ssh $MARWAN_USER@$MARWAN_HOST` manuellement.

**rsync : No such file or directory côté serveur**
→ L'étape 1 n'a pas été faite, ou `$MARWAN_SCRATCH` mal renseigné. Refaire l'étape 1.

**Le symlink data/ est cassé après le transfert**
→ rsync a peut-être remplacé le symlink par un dossier. Vérifier avec `ls -la ~/mrgrt-seg/data`. Si oui, refaire l'étape 1 et **utiliser `--keep-dirlinks` (-K)** dans le rsync (déjà présent dans le script).

**Lenteur**
→ Pour beaucoup de petits fichiers, rsync fait un round-trip par fichier. Si > 30 min sans avancer, basculer sur tar+ssh :
```bash
tar cf - data/ | ssh $MARWAN_USER@$MARWAN_HOST "cd $MARWAN_SCRATCH/mrgrt-seg && tar xf -"
```
