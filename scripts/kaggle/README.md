# Workflow Kaggle — entraînement MRgRT avec reprise inter-sessions

Kaggle propose un GPU gratuit (T4 16 GB ou P100 16 GB) mais limite chaque
session à **12 h max** et donne **30 h/semaine** de quota. Avec ton config
actuel (300 epochs, 250 patients, U-Net 3D patch 128³, validation
sliding-window à chaque epoch), un fold complet va prendre ~8–15 h. Donc :

- **Plan A (par défaut)** : un fold = une session. Vise un fold qui tient en 12 h grâce à `early_stopping_patience: 30`.
- **Plan B (si on dépasse)** : on découpe en plusieurs sessions grâce à `last.pt` (checkpoint complet ré-écrit à chaque epoch).

Le code est déjà prêt pour les deux cas : `--resume auto` détecte un `last.pt`
et reprend exactement où la session précédente s'est arrêtée (optimizer,
scheduler, scaler AMP, history, best_dsc, compteur early stopping inclus).

## Pré-requis

- Compte Kaggle vérifié par téléphone (sinon GPU pas dispo).
- Le repo `mrgrt-seg` poussé sur GitHub (public ou avec token PAT).
- Le dataset uploadé comme **Kaggle Dataset** privé (étape 1 ci-dessous).

## Étape 1 — Créer le Kaggle Dataset (une seule fois, ~15 min)

Le dataset complet fait 1.85 GB. On l'upload une fois, puis on le **monte en
lecture seule** dans chaque notebook.

1. Aller sur https://www.kaggle.com/datasets → **+ New Dataset**
2. **Title** : `mrgrt-oar-thorax` (le slug devient `<ton_user>/mrgrt-oar-thorax`)
3. **Subtitle** : "OAR thorax MRgRT, 250 patients, 5 classes"
4. **Visibility** : Private
5. Glisser-déposer le dossier `data/` complet (ou ses 250 sous-dossiers). Upload via la WebUI fonctionne pour 1.85 GB mais l'API CLI est plus fiable :
   ```bash
   pip install kaggle
   # Récupère kaggle.json depuis kaggle.com/settings -> Create New Token
   mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   cd C:\Users\Lenovo\Desktop\mrgrt_seg
   kaggle datasets init -p data/
   # éditer data/dataset-metadata.json (title, id "ton_user/mrgrt-oar-thorax")
   kaggle datasets create -p data/ --dir-mode zip
   ```
6. Attendre la fin de l'upload (l'écran affiche "Available").

## Étape 2 — Créer le notebook Kaggle

1. https://www.kaggle.com/code → **+ New Notebook**
2. **Settings** (panneau de droite) :
   - **Accelerator** : `GPU T4 x2` (ou P100 si dispo)
   - **Internet** : `On` (indispensable pour `git clone` et `pip install`)
   - **Environment** : `Pin to original environment` (reproductibilité)
3. **Add Input** (panneau de droite) :
   - Ajouter ton dataset `ton_user/mrgrt-oar-thorax` → monté dans `/kaggle/input/mrgrt-oar-thorax/`
4. Coller le contenu de [`train_kaggle.py`](train_kaggle.py) dans une seule
   cellule du notebook (ou utiliser `%load train_kaggle.py`).
5. **Ajuster les 4 variables au début du script** :
   - `GITHUB_REPO` : URL HTTPS de ton repo
   - `GITHUB_BRANCH` : branche à pull
   - `KAGGLE_INPUT_DATASET` : `/kaggle/input/mrgrt-oar-thorax`
   - `MODEL` / `FOLD` / `CONFIG` : ce que tu veux entraîner

## Étape 3 — Premier run

1. Cliquer **Save Version** → **Save & Run All (Commit)**
2. Aller te coucher (ou regarder un film) — la session tourne pendant 12 h max.
3. Quand la session se termine (par toi ou par timeout), Kaggle archive le
   contenu de `/kaggle/working/` dans la version du notebook.
4. Le checkpoint `last.pt` est dans `/kaggle/working/runs/<model>_fold<fold>/last.pt`.

## Étape 4 — Reprendre dans une nouvelle session

C'est le pattern qui permet de dépasser la limite des 12 h.

1. Dans le notebook actuel, **Settings → Add Input → Notebook Output** :
   ajouter ton propre notebook (sa dernière version) comme input.
2. Kaggle monte alors les fichiers de la session précédente dans
   `/kaggle/input/<ton-notebook>/`.
3. Le script `train_kaggle.py` détecte automatiquement ce dossier et copie
   `last.pt` au bon endroit avant de relancer l'entraînement avec `--resume auto`.
4. **Save Version → Save & Run All** : nouvelle session, reprise où on en était.

Le script gère ça tout seul si tu suis la convention de nommage des dossiers.

## Étape 5 — Récupérer les résultats

Quand l'entraînement est fini (early stopping ou epochs atteint) :

1. **Download** depuis l'onglet Output du notebook : `best.pt`, `last.pt`, `history.npz`.
2. Optionnel : créer un Kaggle Dataset à partir de ces outputs pour les conserver durablement.

## Quotas et bonnes pratiques

- **30 h GPU/semaine** : un fold complet (8–15 h) en consomme la moitié. Donc 2 folds/semaine max sur Kaggle gratuit.
- **Cache** : Kaggle propose `Persistent Storage` qui survit entre les sessions. À activer si on veut éviter le pattern "notebook-output-as-input".
- **Pas de TPU** : ce code utilise `cuda` explicitement (autocast, GradScaler). Ne pas changer pour TPU sans refacto.
- **Surveillance** : Kaggle Notebooks ont un timeout d'inactivité de quelques minutes côté UI mais le job continue. On revoit l'avancement dans l'onglet "Logs" de la dernière version.
- **Risque de variabilité** : Kaggle peut couper sans préavis si quota dépassé ou maintenance. Le checkpoint atomique à chaque epoch limite la perte à une epoch max.

## Workflow recommandé en pratique

Pour 5 folds × 2 modèles (U-Net, SegResNet) = 10 runs :

- **Semaine 1** : `unet_fold0`, `unet_fold1` (~20 h Kaggle)
- **Semaine 2** : `unet_fold2`, `unet_fold3` (~20 h Kaggle)
- **Semaine 3** : `unet_fold4`, `segresnet_fold0` (~20 h Kaggle)
- ... etc, ou 5 folds sur MARWAN si dispo entre-temps

En parallèle, dès que MARWAN est OK, basculer dessus pour les runs canoniques
(reproductibles, citables) et garder Kaggle pour XAI / eval rapides.
