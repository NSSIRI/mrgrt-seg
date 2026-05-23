# RUNS TRACKING — Entraînements 5-fold x 2 modèles

> Suivi des 10 runs principaux (+ ablation + multi-seed optionnels).
> Mettre à jour ce tableau à chaque lancement / fin de run.
> Le patient est l'unité statistique : toutes les métriques sont par patient.

## Convention de nommage
- Notebook Kaggle : `mrgrt-train-<model>-fold<N>`
- Dossier de sortie : `runs/<model>_fold<N>/` (best.pt, last.pt, history.npz)
- CSV d'éval : `results/<model>_fold<N>_metrics.csv`

## Dataset utilisé
- **Principal** : `data/` = cohorte filtrée qualité, **N = 187 patients** (.nii non compressé)
  - ⚠️ NOTE : 187 après le fix "reject empty organ files". L'ancien chiffre 303 (et 250)
    correspondait à des versions antérieures du filtre. CONFIRMER le N final avant publication.
- **Ablation (optionnel)** : `data_full/` = 616 patients (non filtré) pour comparaison

---

## TABLEAU PRINCIPAL — 10 runs (5 folds x 2 modèles)

| # | Modèle | Fold | Compte Kaggle | Notebook | GPU | Statut | Epoch atteint | best val_dsc | CSV éval | Date |
|---|--------|------|---------------|----------|-----|--------|---------------|--------------|----------|------|
| 1 | U-Net | 0 | abdelhalimnssiri | mrgrt-train-unet-fold0 | ? | 🟡 en cours | 87/150 | ? | — | |
| 2 | U-Net | 1 | | | | ⚪ à lancer | | | | |
| 3 | U-Net | 2 | | | | ⚪ à lancer | | | | |
| 4 | U-Net | 3 | | | | ⚪ à lancer | | | | |
| 5 | U-Net | 4 | | | | ⚪ à lancer | | | | |
| 6 | SegResNet | 0 | ABDELHALIM | mrgrt-train-segresnet-fold0 | ? | ⚪ à lancer | | | | |
| 7 | SegResNet | 1 | | | | ⚪ à lancer | | | | |
| 8 | SegResNet | 2 | | | | ⚪ à lancer | | | | |
| 9 | SegResNet | 3 | | | | ⚪ à lancer | | | | |
| 10 | SegResNet | 4 | | | | ⚪ à lancer | | | | |

Légende statut : ⚪ à lancer · 🟡 en cours · 🟢 terminé OK · 🔴 échec

---

## ABLATION (optionnel mais = résultat clé de l'article)

Même modèle (U-Net), dataset filtré (187) vs non filtré (616), pour quantifier l'impact du filtrage.

| # | Modèle | Fold | Dataset | Statut | best val_dsc | CSV |
|---|--------|------|---------|--------|--------------|-----|
| A1 | U-Net | 0 | filtré (187) | (= run #1) | | |
| A2 | U-Net | 0 | non filtré (616) | ⚪ à lancer | | |

---

## MULTI-SEED (optionnel, pour estimation de variance)

À faire seulement sur 1-2 folds représentatifs si le temps GPU le permet.

| # | Modèle | Fold | Seed | Statut | best val_dsc |
|---|--------|------|------|--------|--------------|
| S1 | U-Net | 0 | 42 (= run #1) | | |
| S2 | U-Net | 0 | 123 | ⚪ à lancer | |
| S3 | U-Net | 0 | 2024 | ⚪ à lancer | |

---

## CHECKLIST POST-RUN (à faire pour chaque run terminé)

Pour chaque run 🟢 terminé :
- [ ] Télécharger best.pt + last.pt + history.npz depuis l'output Kaggle
- [ ] Placer dans runs/<model>_fold<N>/ localement
- [ ] Noter le GPU utilisé (torch.cuda.get_device_name) — ne pas comparer les TEMPS entre GPU différents
- [ ] Lancer l'éval : `python scripts/evaluate.py --model <m> --fold <N> --ckpt runs/<m>_fold<N>/best.pt`
- [ ] Vérifier que le CSV results/<m>_fold<N>_metrics.csv est créé
- [ ] Cocher la ligne dans le tableau principal

## QUAND LES 10 RUNS SONT FINIS

1. Comparaison architectures :
   ```
   python scripts/compare_results.py \
       --a "results/unet_fold*.csv" --name_a "U-Net" \
       --b "results/segresnet_fold*.csv" --name_b "SegResNet" \
       --metric dsc --out results/comparison_arch.csv
   ```
   Refaire pour --metric hd95 et --metric surface_dsc

2. Ablation (si fait) :
   ```
   python scripts/compare_results.py \
       --a "results_filtered/unet_fold*.csv" --name_a "Filtered" \
       --b "results_full/unet_fold*.csv" --name_b "Unfiltered" --metric dsc
   ```

3. Remplir les Tables 2 et 3 de paper/article_draft.md avec les sorties

4. Générer les box plots (Figure 2 et 3) — script à créer
