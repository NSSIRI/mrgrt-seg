# Analyse complète U-Net / SegResNet — segmentation des OAR thoraciques sur IRM

*04/11/2026. Toutes les valeurs proviennent des tableaux `tableaux/*.csv`, recalculés par `analyse_complete.py` à partir des fichiers existants (lecture seule ; aucun fichier original modifié). Les valeurs déjà publiées dans le rapport E2 vs E3 sont reproduites à l'identique : DSC médian E3 0,893 / 0,915 / 0,825 / 0,502 et E2 0,935 / 0,946 / 0,886 / 0,679.*

**Légende :**
- **[O]** résultat observé ;
- **[H]** hypothèse ou interprétation, non démontrée ;
- **[M]** information manquante.

---

## 1. Inventaire des fichiers analysés

Liste complète, avec les tailles, dans `tableaux/00_inventaire.csv` (43 fichiers).

| Source | Contenu | Utilisation |
|---|---|---|
| `results/phase2_eval/analyse/combined_long.csv` | E1–E3, 187 patients × 4 OAR × 2 grilles | Métriques (grille **native**) |
| `results/phase2_eval/analyse/nnunet_e4_metrics_long.csv` | E4, 76 patients (plis 0–1) | Comparaison exploratoire |
| `results/phase2_eval/analyse/histories_all.csv` | `train_loss`, `val_dsc`, `lr` par époque, 15 runs | Courbes |
| `results/phase3_ablation/_journaux/*/*.log` | Journaux Kaggle (15 notebooks) | Temps d'entraînement, paramètres, GPU |
| `results/phase2_eval/*/phase2_eval/predictions.zip` | Masques prédits, grille native | Figure 7 |
| `results/phase2_eval/*/phase2_eval/splits.json` | Plis | Contrôle de fuite |
| `data/PATIENT_s*/label.nii.gz` | En-têtes NIfTI | Nombre de voxels (spécificité) |
| `configs/*.yaml`, `src/`, `scripts/train.py`, `phase2_eval_kaggle.py` | Code | Revue méthodologique (§9) |

**Définition des colonnes, vérifiée dans `phase2_eval_kaggle.py` :**
- **DSC, IoU, précision, sensibilité** : calculés voxel à voxel sur la grille native du label.
- **HD95** : maximum des deux 95ᵉ percentiles dirigés, en mm. Les distances sont mesurées entre les voxels de bord, définis par érosion 6-connexe.
- **ASSD** : moyenne des distances des deux directions réunies, en mm.
- **Prédiction vide** : les distances valent NaN.
- **Spécificité** : absente des CSV, je l'ai **dérivée** :
  - TN = N − TP − FP − FN, où N est le nombre de voxels de la grille native ;
  - TP est reconstruit à partir du DSC et des volumes ;
  - contrôle : ce TP redonne la précision et la sensibilité du CSV à 1e-16 près.

**Informations manquantes [M] :**
- **Perte de validation et DSC d'entraînement** : `trainer.py` ne les enregistre pas. Seuls `train_loss`, `val_dsc` et `lr` sont disponibles.
- **Consommation mémoire GPU** : jamais enregistrée pour E1–E4. Elle sera relevée pendant la phase 3 (`gpu_mem_*.csv`).
- **Temps d'entraînement de E1 pli 0, E1 pli 2 et E2 pli 1** :
  - E1 pli 0 a été entraîné hors Kaggle, et son journal n'est pas disponible ;
  - pour E1 pli 2 et E2 pli 1, le journal Kaggle conservé est celui d'une session ultérieure, sans entraînement.
- **Temps d'inférence de E4** : non disponible.
- **Temps d'inférence de E1 pli 0** : l'évaluation a tourné sur CPU, donc ce pli est exclu de la comparaison.

## 2. Architectures et protocole

| | E1 U-Net base | E3 U-Net opt | E2 SegResNet | E4 nnU-Net (appoint) |
|---|---|---|---|---|
| Implémentation | MONAI `UNet`, 5 niveaux, plain | idem, canaux élargis | MONAI `SegResNet`, blocs [1,2,2,4]/[1,1,1] | PlainConvUNet, 6 niveaux |
| Canaux | 32→512 | 48→832 | init_filters 32 | 32→320 |
| Normalisation / activation | InstanceNorm / PReLU | InstanceNorm / PReLU | GroupNorm(8) / ReLU | nnU-Net par défaut |
| Paramètres [O] | 7 918 061 | 18 802 973 | 18 796 101 | non recalculé ici |
| Protocole | commun (ci-dessous) | commun | commun | **différent** : SGD, 100 époques, deep supervision, mirroring, checkpoint final |

**Protocole commun E1–E3 [O, code et journaux] :**
- **Prétraitement** : orientation RAS ; rééchantillonnage à 1,5 × 1,5 × 3 mm ; intensités ramenées des percentiles 0,5–99,5 vers [0, 1].
- **Patchs** : 128 × 128 × 64, ratio positifs/négatifs 2:1, batch 2.
- **Augmentations** : affine, rot90, gamma, champ de biais, bruit ; pas de flip.
- **Optimisation** : perte DiceCE ; AdamW 1e-4 ; LR cosinus jusqu'à 1e-6 ; AMP ; 150 époques.
- **Sélection** : `best.pt` = meilleur `val_dsc` sur le pli de validation.
- **Inférence** : fenêtre glissante, recouvrement 0,5, pondération gaussienne.

**Comparabilité :**
- **E1, E2, E3** : mêmes plis, même code, même évaluation. Ils sont **comparables**.
- **E4** : sous-échantillon de 76 patients et protocole différent. Les comparaisons avec E4 sont **exploratoires**.

## 3. Tableau comparatif des performances (187 patients, grille native)

Médiane [Q1–Q3]. Moyenne ± écart-type et IC 95 % bootstrap de la médiane : `tableaux/02_descriptifs_E1_E2_E3_187.csv`.

| Métrique | OAR | E1 U-Net 7,9 M | E3 U-Net 18,8 M | E2 SegResNet 18,8 M |
|---|---|---|---|---|
| **DSC** | Poumon G | 0,882 [0,798–0,927] | 0,893 [0,818–0,934] | **0,935 [0,892–0,959]** |
| | Poumon D | 0,907 [0,818–0,941] | 0,915 [0,828–0,945] | **0,946 [0,905–0,967]** |
| | Cœur | 0,800 [0,659–0,865] | 0,825 [0,695–0,882] | **0,886 [0,823–0,925]** |
| | Œsophage | 0,414 [0,252–0,534] | 0,502 [0,302–0,629] | **0,679 [0,554–0,753]** |
| **IoU** | Poumon G | 0,789 | 0,806 | **0,878** |
| | Poumon D | 0,830 | 0,844 | **0,897** |
| | Cœur | 0,667 | 0,702 | **0,795** |
| | Œsophage | 0,261 | 0,335 | **0,514** |
| **HD95 (mm)** | Poumon G | 15,6 [7,0–104,1] | 12,2 [6,0–81,5] | **6,0 [3,5–15,0]** |
| | Poumon D | 12,1 [6,0–67,5] | 11,0 [4,8–56,8] | **5,0 [2,9–12,0]** |
| | Cœur | 20,5 [10,3–44,6] | 18,3 [9,4–31,7] | **11,9 [6,2–19,4]** |
| | Œsophage | 19,8 [10,5–42,1] | 13,3 [8,1–31,0] | **7,8 [4,7–18,2]** |
| **ASSD (mm)** | Poumon G | 3,29 | 2,95 | **1,33** |
| | Poumon D | 3,05 | 2,50 | **1,12** |
| | Cœur | 4,79 | 4,41 | **2,78** |
| | Œsophage | 4,70 | 3,39 | **1,87** |
| **Sensibilité** | Poumon G | 0,900 | 0,909 | **0,946** |
| | Poumon D | 0,926 | 0,932 | **0,959** |
| | Cœur | 0,800 | 0,825 | **0,891** |
| | Œsophage | 0,441 | 0,473 | **0,697** |
| **Précision** | Poumon G | 0,905 | 0,916 | **0,940** |
| | Poumon D | 0,911 | 0,927 | **0,949** |
| | Cœur | 0,851 | 0,861 | **0,901** |
| | Œsophage | 0,478 | 0,577 | **0,688** |
| **Spécificité** | 4 OAR | 0,998–1,000 | 0,998–1,000 | 0,999–1,000 |

**Lecture de la spécificité.** Elle dépasse 0,998 partout, parce que les vrais négatifs (tout le fond de l'image) dominent largement. Elle ne discrimine pas les modèles et ne devrait pas figurer comme critère principal [O]. Les écarts, tous inférieurs à 1e-3, sont « significatifs » mais sans portée pratique.

**Prédictions vides** (distances non définies, exclues des tests sur les distances) :

| Modèle | Œsophage | Cœur |
|---|---|---|
| E1 | 3 | — |
| E3 | 4 | — |
| E2 | 1 | 1 |

**Effectifs et coût de calcul [O]** (tableaux 06 et 06b) :

| | E1 | E3 | E2 |
|---|---|---|---|
| Inférence médiane par patient (T4) | 0,47 s (n = 149) | 0,53 s | **4,99 s** |
| Temps par époque (médiane des plis) | 69–78 s | 73–76 s | 150–163 s |
| Durée d'une session d'entraînement | 2,9–3,3 h | 3,1–3,3 h | 6,3–6,8 h |

[H] E1 (7,9 M) et E3 (18,8 M) ont **le même temps par époque** : pour le U-Net, l'entraînement semble limité par le chargement et l'augmentation des données (CPU), pas par le GPU. SegResNet coûte deux fois plus de temps par époque et environ 9 fois plus à l'inférence.

## 4. Courbes d'apprentissage

Figures 1 et 2 ; valeurs dans `tableaux/08_courbes_apprentissage.csv`.

**Observé [O] :**
- **Meilleure époque** : 117 à 147 pour tous les runs. Le `best.pt` est donc toujours choisi en fin d'entraînement.
- **95 % du maximum** atteint vers les époques 68 à 89.
- **Après le meilleur point, le `val_dsc` ne baisse presque pas** : entre 0,001 et 0,006 de perte.
- **Pente de `val_dsc` sur les 30 dernières époques** : toujours positive, de +0,001 à +0,009. La perte d'entraînement continue de baisser.
- **Instabilité** (écart-type des variations d'une époque à l'autre sur les 30 dernières époques) : 0,002 à 0,005, faible et comparable entre modèles.
- **Plateau atteint** : environ 0,68 (E1), 0,72 (E3) et 0,81 (E2).
- **Perte d'entraînement finale** : 0,56–0,59 (E1), 0,51–0,53 (E3), 0,41–0,44 (E2).
- **Aucun early stopping**, sauf E1 pli 4, arrêté à l'époque 147.

**Interprétation [H] :**
- **Pas de signe de surapprentissage** dans la fenêtre observée : le `val_dsc` ne décroît pas. Ce diagnostic reste **incomplet**, car la perte de validation n'a pas été enregistrée [M].
- **Léger sous-apprentissage possible** : les courbes montent encore à 150 époques. Mais le LR cosinus atteint 1e-6 à l'époque 150, donc prolonger l'entraînement demanderait de changer le calendrier de LR.
- **E2 atteint une perte d'entraînement plus basse** et un meilleur `val_dsc`. Rien n'indique que l'avantage de SegResNet vienne d'une meilleure régularisation plutôt que d'un meilleur ajustement aux données. Les deux restent possibles.

## 5. Résultats par organe

Figures 3, 4 et 7.

- **Poumons** : bonnes performances pour tous les modèles (DSC médian ≥ 0,88). Les différences portent surtout sur les **queues de distribution** : HD95 Q3 de 81 à 104 mm pour les U-Net contre 15 mm pour E2. Le rapport E2 vs E3 relie ces queues à des fragments parasites et à des confusions gauche/droite. La figure 7 (rangée 1) montre aussi un poumon presque entièrement manqué par E3 sur une IRM au champ inhabituel.
- **Cœur** : niveau intermédiaire (DSC 0,80–0,89) ; HD95 médian de 12 à 20 mm.
- **Œsophage** : **l'organe le plus difficile pour tous les modèles**. DSC médian 0,41 (E1), 0,50 (E3) et 0,68 (E2). Les U-Net **sous-segmentent** : la sensibilité (0,44–0,47) est inférieure à la précision (0,48–0,58), et le rapport E2 vs E3 a mesuré une erreur volumique signée médiane de −16 % pour E3. La figure 7 (rangée 2) montre un cas où E3 obtient un DSC de 0,00 pour l'œsophage alors que E2 atteint 0,45.
- **Erreurs de localisation** : prédictions vides et HDmax de plusieurs centaines de mm (rapport E2 vs E3, §4). Elles sont plus fréquentes avec les U-Net.

## 6. Comparaison U-Net / SegResNet

Figure 5 ; tableau 03.

| Comparaison (DSC, médiane des différences appariées [IC 95 %]) | Poumon G | Poumon D | Cœur | Œsophage |
|---|---|---|---|---|
| **E2 − E3** (même taille) | +0,027 [0,023 ; 0,035] | +0,025 [0,020 ; 0,028] | +0,052 [0,039 ; 0,062] | **+0,152 [0,136 ; 0,168]** |
| **E3 − E1** (taille seule, U-Net) | +0,011 [0,008 ; 0,013] | +0,006 [0,005 ; 0,009] | +0,019 [0,015 ; 0,022] | +0,080 [0,065 ; 0,092] |
| E2 − E1 | +0,040 | +0,033 | +0,072 | +0,216 |

| Comparaison (HD95, mm) | Poumon G | Poumon D | Cœur | Œsophage |
|---|---|---|---|---|
| E2 − E3 | −2,8 [−4,3 ; −1,9] | −2,9 [−4,6 ; −1,8] | −2,7 [−4,1 ; −1,9] | −4,4 [−5,2 ; −3,0] |
| E3 − E1 | −0,8 [−1,4 ; −0,2] | −0,4 [−1,0 ; −0,2] | −0,9 [−1,4 ; −0,1] | −3,2 [−5,1 ; −2,6] |

**Observé [O] :**
- **À capacité égale**, E2 est meilleur que E3 chez 90 à 97 % des patients pour le DSC, selon l'OAR.
- **Passer de 7,9 M à 18,8 M paramètres** n'améliore le U-Net que modestement : environ 3 fois moins que l'écart E2 − E3, sauf pour l'œsophage, où le gain atteint la moitié de cet écart.
- **La capacité n'explique donc qu'une partie de l'écart E1 → E2.** L'essentiel subsiste à taille égale.

**Exploratoire, E4 nnU-Net (76 patients, tableau 05) :**
- E4 dépasse E2 en DSC : +0,005 à +0,008 pour les poumons et le cœur, **+0,070 [0,047 ; 0,090]** pour l'œsophage.
- Pas de différence de HD95 entre E4 et E2 après correction de Holm (p_Holm = 1,0).
- Protocole, checkpoint (final contre meilleur) et effectif diffèrent : ce résultat ne permet **pas** de classer les architectures.

## 7. Analyse statistique

**Méthode :**
- **Unité d'analyse** : le patient (n = 187), avec des paires constituées par patient et par OAR.
- **Test** : Wilcoxon apparié bilatéral ; correction de Holm par métrique sur 12 tests (4 OAR × 3 comparaisons).
- **Taille d'effet** : r de rang bisérial.
- **Incertitude** : IC 95 % bootstrap de la médiane des différences (10 000 tirages, graine fixée).
- **Les distributions sont asymétriques** (HD95 à longue queue, DSC borné), d'où ce test non paramétrique et la préférence pour la médiane.

**Résultats [O] :**
- **E2 − E3** : p_Holm < 1e-6 pour DSC, IoU, HD95, ASSD, sensibilité et précision, sur les 4 OAR. r = 0,89–0,97 pour le DSC et −0,53 à −0,72 pour le HD95 : des effets **grands**.
- **E3 − E1** : p_Holm ≤ 0,005. r = 0,51–0,73 pour le DSC (effet moyen à grand) et −0,24 à −0,49 pour le HD95 (effet faible à moyen).
- **Variabilité entre plis** (écart-type des DSC moyens par pli, moyenne des 4 OAR) : 0,021 (E1), 0,017 (E3), 0,014 (E2), contre un écart E2 − E3 de 0,03 à 0,15. L'écart reste nettement supérieur à la variabilité **entre plis** ; la variabilité **entre graines** n'est pas mesurée [M].

**Limites statistiques :**
1. **Une seule graine par modèle et par pli.** La variance d'entraînement n'est pas estimée ; c'est l'objet de la phase 3, bloc A.
2. **Tests non indépendants entre OAR** : un même patient contribue aux 4 OAR. Holm reste valide pour chaque famille de tests, mais les tests ne sont pas indépendants.
3. **Pas de jeu de test séparé** : les métriques sont mesurées sur le pli qui a servi à choisir l'époque (§9, point 1). Le biais est probablement faible [H], car seule l'époque est sélectionnée, et il touche tous les modèles de la même façon. Il existe néanmoins.
4. **Distances** : les patients à prédiction vide sont exclus (jusqu'à 5 paires). Ce sont des échecs, donc les distances sont légèrement **optimistes**, surtout pour les U-Net.
5. **E4** : n = 76 et protocole différent ; analyse exploratoire seulement.

## 8. Figures

Toutes en PNG 300 dpi et en PDF, dans `figures/`. Palette du projet, vérifiée par le validateur (daltonisme : PASS) ; les légendes et tableaux sont toujours fournis.

| Figure | Contenu |
|---|---|
| `fig01_courbes_train_loss` | Perte d'entraînement, 5 plis et moyenne, par modèle |
| `fig02_courbes_val_dsc` | DSC de validation, avec la meilleure époque de chaque pli |
| `fig03_boxplots_par_organe` | DSC, HD95 (log), ASSD (log), sensibilité ; 3 modèles × 4 OAR |
| `fig04_heatmap_medianes` | DSC et HD95 médians, modèle × OAR |
| `fig05_differences_appariees` | Différences appariées avec IC 95 % (E2−E3, E3−E1, E2−E1) |
| `fig06_temps_inference_entrainement` | Inférence par patient, durée d'entraînement par pli |
| `fig07_superpositions_masques` | Référence, E3 et E2 sur deux patients : DSC œsophage E3 médian et pire cas |

Les courbes de perte de validation et de DSC d'entraînement demandées ne peuvent pas être tracées : ces données n'existent pas [M].

## 9. Revue des scripts : points à corriger ou à signaler

Aucun fichier n'a été modifié.

1. **Sélection du modèle sur le jeu d'évaluation** (`trainer.py`). Le `best.pt` et l'early stopping utilisent le pli de validation, qui sert ensuite à rapporter les métriques. → Pour l'article, soit le dire, soit garder un jeu de test séparé ou utiliser le checkpoint final. Ici, le `val_dsc` final est à moins de 0,006 du maximum, donc l'impact est probablement faible [H].
2. **`warmup_epochs: 5` n'est pas implémenté.** Le `lr` de l'époque 1 vaut déjà 9,999e-05 dans les historiques [O]. Or le brouillon `Publication_Combined_..._v1.md` (§2.6) décrit un « 5-epoch linear warm-up » : **c'est inexact, à corriger.**
3. **Journalisation insuffisante** : pas de perte de validation, pas de DSC d'entraînement, pas de DSC par organe pendant l'entraînement, pas de mémoire GPU. → À ajouter dans une future version, sur une branche dédiée.
4. **Métrique de sélection différente de la métrique rapportée.** Le `val_dsc` est la moyenne des 4 OAR sur la grille rééchantillonnée ; les résultats sont rapportés par OAR sur la grille native. C'est cohérent mais à mentionner.
5. **`experiment.seed` règle à la fois l'entraînement et les plis.** C'est corrigé sur la branche `ablation-18m` (`splits.seed`).
6. **MONAI non figé pendant l'entraînement** (`monai>=1.3`) : la version réellement utilisée pour E1–E3 n'est pas vérifiable [M]. L'évaluation, elle, a utilisé 1.5.2.
7. **Validation croisée non stratifiée** par centre, champ ou séquence : des plis déséquilibrés sont possibles. La faible variabilité entre plis (≤ 0,033 par OAR) suggère que l'effet est limité [H].
8. **Fuite de données** : aucune entre entraînement et validation au niveau des identifiants [O]. Les plis sont disjoints, 187 patients uniques, et chaque patient est dans le même pli pour les 3 modèles (tableau 01). **Non vérifié** : si TotalSegmentator MRI contient plusieurs examens d'un même sujet sous des identifiants différents, la séparation par identifiant ne suffirait pas [M].
9. **Pas de flip en augmentation** : choix cohérent avec des labels latéralisés (poumon G/D), à justifier dans l'article.
10. **Le brouillon d'article combiné** compare E1 et non E3, rapporte des p < 10⁻²⁹⁹ (impossible avec un Wilcoxon apparié sur 187 patients : le minimum exact est d'environ 5 × 10⁻⁵⁷) et mélange deux sources de Dice (son `[FLAG]` de la §3.2). → Le refaire à partir des tableaux de ce dossier.

## 10. Discussion critique et conclusion

**Réponses aux questions :**
- **Quelle architecture est la meilleure ?** E2 SegResNet, pour toutes les métriques et tous les OAR, à capacité égale avec E3 [O]. Sur 76 patients, nnU-Net (E4) fait un peu mieux en DSC, surtout pour l'œsophage, mais avec un protocole différent [O, exploratoire].
- **Quels organes sont les mieux segmentés ?** Les poumons (DSC médian 0,88–0,95), puis le cœur.
- **Quels organes restent difficiles ?** L'œsophage, pour tous les modèles (0,41–0,68 ; E4 0,745 sur 76 patients), avec une sous-segmentation par les U-Net.
- **Les résultats sont-ils reproductibles ?** Ils sont stables **entre plis** (écart-type ≤ 0,033), et l'avantage de E2 se retrouve dans les 20 couples pli × OAR (rapport E2 vs E3). La reproductibilité **entre graines** n'est pas mesurée [M].
- **Y a-t-il du surapprentissage ?** Aucun signe dans les courbes disponibles. Diagnostic incomplet, faute de perte de validation.
- **Les différences sont-elles significatives ?** Oui, avec des effets grands pour E2 − E3 (r ≥ 0,89 sur le DSC). Mais cela vaut pour une seule graine et sans jeu de test séparé.
- **Qu'est-ce qui peut expliquer les différences ? [H]** La taille n'en explique qu'une partie (E3 − E1 est petit). Restent les composants propres à SegResNet : blocs résiduels, GroupNorm, dropout, suréchantillonnage, profondeur. L'ablation de la phase 3 (A1–A3) teste les trois premiers.
- **Quelles améliorations méthodologiques ?**
  - plusieurs graines ;
  - jeu de test séparé ou test externe sur de vraies IRM MRgRT ;
  - journalisation complète ;
  - stratification par centre ou séquence ;
  - nnU-Net sur les 5 plis ;
  - vérifier l'absence de doublons de sujets.

**Conclusion.** Sur 187 IRM diagnostiques thoraciques, avec le même protocole et le même nombre de paramètres, SegResNet dépasse nettement un U-Net plain pour les 4 OAR. L'écart est faible pour les poumons (+0,03 de DSC), mais grand pour le cœur (+0,05) et surtout l'œsophage (+0,15). Il apparaît aussi sur les distances de surface, où le U-Net produit davantage de fragments et d'erreurs de localisation. Augmenter la capacité du U-Net n'en comble qu'une petite partie. Ces conclusions valent pour une seule graine d'entraînement, une validation croisée sans test externe et des IRM diagnostiques (pas des IRM de planification MRgRT). Les mécanismes en jeu relèvent encore de l'hypothèse, que la phase 3 doit tester.
