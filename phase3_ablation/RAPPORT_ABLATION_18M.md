# Ablation U-Net / SegResNet à capacité égale (≈ 18,8 M de paramètres)

*Rapport du 06/11/2026. Toutes les valeurs proviennent des CSV de `phase3_ablation/analyse/`, produits par `ablation_stats.py`, `ablation_masks.py` et `ablation_figures.py`, et de leur relecture directe. Aucun fichier existant n'a été modifié. Les figures sont dans `phase3_ablation/figures/`.*

**Légende :**
- **[O]** résultat observé ;
- **[L]** résultat de la littérature (texte non relu dans ce travail sauf mention) ;
- **[H]** interprétation possible, non démontrée.

---

## 0. Synthèse

- **Q1, graines : l'écart E2 − E3 est reproductible [O].** Réentraînés avec une autre graine (123), E2 et E3 donnent presque les mêmes résultats.
  - L'effet de la graine sur le DSC médian apparié reste ≤ 0,008, avec des IC 95 % dans ±0,016.
  - L'écart E2 − E3 vaut +0,024 à +0,152 avec la graine 42, et +0,024 à +0,146 avec la graine 123.
- **Q2, connexions résiduelles (A1) : elles comblent environ la moitié de l'écart en DSC [O].** C'est 50 à 62 % selon l'organe, avec des IC entre 40 et 76 %.
  - Elles font presque disparaître les fragments lointains : HDmax médian des poumons de 24 mm contre 152 mm pour E3, 2 composantes connexes au lieu de 9.
  - La confusion gauche/droite des poumons concerne 4 patients au lieu de 22 (McNemar p = 1e-5).
  - A1 reste inférieur à E2 sur toutes les métriques.
- **Q3, normalisation : elle n'explique pas l'avantage de SegResNet [O].**
  - **A2 (U-Net + GroupNorm) dégrade le U-Net** : −0,018 à −0,136 de DSC face à E3, avec des fragments plus nombreux.
  - **A3 (SegResNet + InstanceNorm) fait au moins aussi bien qu'E2** : +0,005 à +0,046 de DSC. Cet écart dépasse la variabilité due à la graine.
- **Tous les effets significatifs dépassent le seuil de variabilité due à la graine**, sauf deux écarts de HD95 sur le poumon G. Ils restent donc interprétables selon la règle fixée a priori (§4).
- **Limite principale :** A1 et A3 n'ont qu'une graine, et A1 change davantage qu'une simple connexion de saut (§5).

---

## 1. Plan réalisé et contrôles

| Bloc | Contenu | Runs | Statut |
|---|---|---|---|
| Contrôles | Réévaluation de E3 pli 0 ; smoke run d'une époque | 2 | ✅ |
| B (Q2–Q3) | A1, A2, A3 × 5 plis, graine 42 | 15 | ✅ (A3 pli 2 relancé après une interruption Kaggle) |
| A (Q1) | E3 et E2 × 5 plis, graine 123 | 10 | ✅ |

Les questions Q4 (sans dropout) et Q5 (résiduel + GroupNorm) étaient hors de l'option Standard choisie. Elles ne sont pas traitées.

**Contrôles [O] :**
- **Environnement identique pour les 25 runs :** commit `0a36a1e`, torch 2.10.0+cu128, MONAI 1.5.2, Tesla T4. Les runs sont répartis sur 7 comptes Kaggle, ce qui n'a aucune incidence sur les calculs.
- **Mêmes plis que la phase 2 :** 187 patients par variante, et liste de validation de chaque `run_meta.json` identique à `splits.json`.
- **Évaluation :**
  - les fonctions de métriques sont identiques à celles de la phase 2 (vérifié par test) ;
  - la réévaluation du checkpoint E3 du pli 0 reproduit **exactement** la phase 2 sur la grille native, avec un écart de 0,0 ;
  - sur la grille rééchantillonnée, un seul voxel diffère (patient s0031, poumon D). C'est dû à l'image Kaggle (Python 3.13, numpy 2.1) : l'écart est identique sous torch 2.10 et 2.11. Le critère « < 1e-6 » n'est donc rempli que sur la grille native.
- **Différences avec la phase 2 :**
  - E2 et E3 graine 42 ont été entraînés avec `monai>=1.3` (version non vérifiable) et torch 2.10.0+cu128. Tous les runs de la phase 3 utilisent MONAI 1.5.2.
  - L'effet graine mesuré ci-dessous inclut donc aussi cette éventuelle différence de version. Il reste très faible (§2).

---

## 2. Q1 : stabilité selon la graine d'entraînement

**DSC médian (187 patients, grille native) [O] :**

| Modèle | Poumon G | Poumon D | Cœur | Œsophage |
|---|---|---|---|---|
| E3 graine 42 | 0,893 | 0,915 | 0,825 | 0,502 |
| E3 graine 123 | 0,891 | 0,915 | 0,829 | 0,501 |
| E2 graine 42 | 0,935 | 0,946 | 0,886 | 0,679 |
| E2 graine 123 | 0,937 | 0,946 | 0,883 | 0,679 |

**Écart E2 − E3 (DSC, médiane appariée [IC 95 %], `stats_q1_graines.csv`) [O] :**

| OAR | Graine 42 | Graine 123 | Moyenne des 2 graines par patient |
|---|---|---|---|
| Poumon G | +0,027 [0,023 ; 0,035] | +0,032 [0,025 ; 0,041] | +0,030 [0,024 ; 0,039] |
| Poumon D | +0,025 [0,019 ; 0,030] | +0,024 [0,019 ; 0,032] | +0,026 [0,021 ; 0,032] |
| Cœur | +0,052 [0,040 ; 0,062] | +0,048 [0,042 ; 0,052] | +0,047 [0,043 ; 0,057] |
| Œsophage | +0,152 [0,136 ; 0,168] | +0,146 [0,124 ; 0,163] | +0,144 [0,121 ; 0,169] |

Pour toutes ces comparaisons, p < 1e-20 (Wilcoxon apparié) et r de rang bisérial de 0,87 à 0,98.

**Effet de la graine (graine 123 − graine 42, DSC) [O] :**
- E3 : −0,008 à +0,0003, avec des IC dans [−0,016 ; +0,003] ;
- E2 : −0,0006 à +0,004, avec des IC dans [−0,003 ; +0,015] ;
- aucun n'est significatif (p ≥ 0,10, sans correction).

**Variabilité au niveau des plis (DSC, `stats_q1_variabilite_plis.csv`) [O] :**
- l'écart-type entre graines des moyennes de pli vaut 0,003 à 0,010 ;
- il est 3 à 10 fois plus petit que l'écart-type entre plis (0,015 à 0,031), et 5 à 15 fois plus petit que l'écart E2 − E3.

**Réponse à Q1 :** oui. L'écart E2 − E3 est reproductible d'une graine d'entraînement à l'autre, plis fixés (Figure `abl_fig4`).

---

## 3. Q2–Q3 : composants

**Différences appariées de DSC (médiane [IC 95 %], % de patients améliorés, `stats_q2_variantes.csv`) [O] :**

| Variante | OAR | vs E3 | % > E3 | vs E2 | Part de l'écart E3 → E2 comblée [IC 95 %] |
|---|---|---|---|---|---|
| **A1** U-Net résiduel | Poumon G | +0,017 [0,012 ; 0,021] | 87 % | −0,010 [−0,012 ; −0,008] | 0,62 [0,47 ; 0,74] |
| | Poumon D | +0,014 [0,010 ; 0,019] | 87 % | −0,008 [−0,011 ; −0,006] | 0,58 [0,44 ; 0,76] |
| | Cœur | +0,026 [0,019 ; 0,035] | 82 % | −0,019 [−0,023 ; −0,016] | 0,50 [0,40 ; 0,64] |
| | Œsophage | +0,081 [0,067 ; 0,089] | 82 % | −0,068 [−0,084 ; −0,050] | 0,53 [0,43 ; 0,60] |
| **A2** U-Net + GroupNorm | Poumon G | −0,021 [−0,027 ; −0,017] | 16 % | −0,055 | −0,78 [−1,05 ; −0,58] |
| | Poumon D | −0,018 [−0,025 ; −0,010] | 21 % | −0,046 | −0,72 [−1,04 ; −0,41] |
| | Cœur | −0,030 [−0,038 ; −0,023] | 20 % | −0,085 | −0,58 [−0,81 ; −0,43] |
| | Œsophage | −0,136 [−0,148 ; −0,119] | 16 % | −0,273 | −0,90 [−1,07 ; −0,74] |
| **A3** SegResNet + InstanceNorm | Poumon G | +0,041 [0,032 ; 0,050] | 97 % | **+0,006 [0,005 ; 0,008]** | 1,51 [1,26 ; 1,78] |
| | Poumon D | +0,034 [0,028 ; 0,042] | 95 % | **+0,005 [0,003 ; 0,005]** | 1,37 [1,19 ; 1,62] |
| | Cœur | +0,062 [0,052 ; 0,073] | 96 % | **+0,013 [0,009 ; 0,017]** | 1,20 [1,06 ; 1,44] |
| | Œsophage | +0,199 [0,171 ; 0,219] | 96 % | **+0,046 [0,036 ; 0,061]** | 1,31 [1,15 ; 1,47] |

**Tests et seuil d'interprétation :**
- Wilcoxon apparié bilatéral, correction de Holm par métrique sur 24 tests (3 variantes × 4 OAR × 2 références) ;
- toutes les comparaisons de DSC ont p_Holm < 1e-10, et toutes celles de Surface-DSC et d'ASSD p_Holm < 1e-5.

**HD95 (médiane appariée, mm) [O] :**
- **A1 − E3** : −1,7 à −2,3 ;
- **A3 − E3** : −4,3 à −5,2 ;
- **A2 − E3** : +1,5 à +6,3 ;
- **A1 − E2** : +0,5 à +1,1 ;
- **A3 − E2** : −0,5 à −1,0 (p_Holm ≤ 0,03).

**Volumes [O] :** l'erreur volumique médiane |ΔV|/V de l'œsophage vaut 0,376 pour E3, 0,267 pour A1, 0,659 pour A2, 0,227 pour E2 et 0,183 pour A3.

**Sensibilité (`stats_sensibilite_ref_moyennee.csv`) [O] :** les conclusions sont les mêmes si E3 et E2 sont remplacés par leur moyenne sur les deux graines. Par exemple, A1 − E3 pour l'œsophage passe de +0,081 à +0,077, et A3 − E2 pour l'œsophage de +0,046 à +0,043.

Figures `abl_fig1` (différences appariées et bande de variabilité due à la graine), `abl_fig2` (part de l'écart comblée) et `abl_fig3` (boîtes à moustaches).

---

## 4. Robustesse (masques, grille native)

*Figure `abl_fig5`.*

**HDmax médian, en mm (`stats` relus sur `metrics_long`) [O] :**

| Modèle | Poumon G | Poumon D | Cœur | Œsophage |
|---|---|---|---|---|
| E3 | 151,8 | 139,4 | 61,8 | 27,3 |
| A2 | 152,5 | 139,9 | 105,0 | 37,7 |
| **A1** | **24,1** | **24,2** | 34,7 | 19,2 |
| E2 | 21,1 | 20,9 | 26,3 | 13,3 |
| A3 | 18,1 | 18,8 | 24,9 | 11,8 |

Le HDmax médian des poumons passe de plus de 139 mm pour les U-Net plain (E3, A2) à environ 24 mm avec les unités résiduelles (A1). C'est la signature d'une forte réduction des fragments lointains, quantifiée ci-dessous.

**Fragments (`stats_fragments.csv`) [O]** : nombre médian de composantes connexes (connexité 26) / pourcentage de patients ayant plus de 1 % du volume prédit hors de la plus grande composante.

| Modèle | Poumon G | Poumon D | Cœur | Œsophage |
|---|---|---|---|---|
| E3 graine 42 | 9 / 32,6 % | 9 / 35,8 % | 6 / 34,2 % | 2 / 39,6 % |
| E3 graine 123 | 9 / 32,1 % | 7 / 31,0 % | 5 / 26,2 % | 2 / 34,2 % |
| A2 | 14 / 44,9 % | 13 / 41,2 % | 8 / 41,2 % | 3 / 49,2 % |
| **A1** | **2 / 16,6 %** | **2 / 19,3 %** | **3 / 16,0 %** | **1 / 29,4 %** |
| E2 graine 42 | 1 / 14,4 % | 1 / 11,2 % | 1 / 12,3 % | 1 / 29,9 % |
| E2 graine 123 | 1 / 11,2 % | 1 / 15,0 % | 1 / 17,1 % | 1 / 27,3 % |
| A3 | 1 / 11,2 % | 1 / 12,3 % | 1 / 13,4 % | 1 / 27,8 % |

**Confusion gauche/droite des poumons** (plus de 5 % d'un poumon de référence prédit comme le poumon opposé, `stats_lateralite.csv`) [O] :

| Modèle | E3 s42 | E3 s123 | A2 | A1 | E2 s42 | E2 s123 | A3 |
|---|---|---|---|---|---|---|---|
| Patients | 22 (11,8 %) | 17 (9,1 %) | 19 (10,2 %) | **4 (2,1 %)** | 8 (4,3 %) | 7 (3,7 %) | **3 (1,6 %)** |

McNemar exact face à E3 graine 42 :

| Modèle | Discordances (seul le modèle se trompe / seul E3 se trompe) | p |
|---|---|---|
| A1 | 0 / 18 | 1e-5 |
| A3 | 0 / 19 | < 1e-5 |
| E2 | 1 / 15 | 5e-4 |
| A2 | 5 / 8 | 0,58 (non significatif) |

**Contrôle :** pour E2 et E3 graine 42, ce script redonne exactement les chiffres de la phase 2 (`e2e3_fragments.csv`, `e2e3_laterality_187.csv` : 22 et 8 patients).

**Lecture :**
- Les unités résiduelles (A1) ramènent le U-Net presque au niveau de SegResNet pour la fragmentation et la latéralité.
- La GroupNorm (A2) aggrave la fragmentation, sans changer significativement la confusion gauche/droite.
- La graine change peu ces indicateurs.

**Prédictions vides (œsophage) [O] :** E3 4 ; A1 2 ; A2 6 ; E2 1 ; A3 0. Ces cas sont exclus des distances, ce qui rend légèrement optimistes les distances des modèles qui ont le plus de prédictions vides.

---

## 5. Littérature [L]

- **Connexions résiduelles.** He et al. (CVPR 2016, ResNet) les ont introduites pour faciliter l'optimisation des réseaux profonds. SegResNet (Myronenko, BrainLes/MICCAI 2018) reprend des blocs résiduels dans un encodeur–décodeur 3D. Les présets « ResEnc » de nnU-Net (Isensee et al., « nnU-Net Revisited », MICCAI 2024) rapportent un gain des encodeurs résiduels sur plusieurs jeux de données (non relu ici, à vérifier).
- **GroupNorm.** Wu & He (ECCV 2018) l'ont proposée pour les petits batchs. Myronenko 2018 l'utilise dans SegResNet. nnU-Net utilise par défaut l'InstanceNorm (Isensee et al., Nature Methods 2021).
- **Comparaisons d'architectures à capacité égale pour les OAR thoraciques en IRM :** je n'ai pas identifié d'ablation de ce type dans les références déjà réunies (Ribeiro et al. 2023 et les articles de la phase 2). L'absence d'une telle étude n'est **pas vérifiée** par une recherche systématique.

---

## 6. Comparaison

- **Connexions résiduelles.** Le gain observé avec A1 va dans le même sens que la littérature sur les encodeurs résiduels.
- **Normalisation.** Que GroupNorm n'apporte rien, voire dégrade, est cohérent avec le fait que nnU-Net obtient de bons résultats avec l'InstanceNorm. En revanche, la GroupNorm a été motivée par le cas des petits batchs (Wu & He), alors qu'ici le batch vaut 2. Notre résultat va donc à l'encontre de l'attente intuitive [O + L].

---

## 7. Interprétation possible [H]

1. **Les unités résiduelles expliquent environ la moitié de l'écart E3 → E2.** Leur effet le plus net porte sur la cohérence spatiale : beaucoup moins de fragments lointains, HDmax divisé par environ 6 pour les poumons. Une explication plausible est un meilleur contexte et un apprentissage plus stable grâce aux sauts d'identité. Ce n'est pas démontré ici.
2. **La normalisation n'est pas le facteur.** Le SegResNet de MONAI fonctionne au moins aussi bien avec l'InstanceNorm. Avec GroupNorm(8), le U-Net plain se dégrade : la première couche, à 48 canaux, n'a que 6 canaux par groupe. Cela suggère une interaction entre la GroupNorm et l'architecture plain plutôt qu'un défaut de la GroupNorm en soi.
3. **L'autre moitié de l'écart** vient vraisemblablement des différences non testées ici : profondeur et nombre de blocs (SegResNet [1, 2, 2, 4] contre une convolution par niveau), suréchantillonnage (trilinéaire contre convolution transposée), addition contre concaténation pour les connexions de saut, ReLU contre PReLU, dropout. Ce reste n'est pas attribuable avec ces données.

---

## 8. Limites

1. **Une seule graine pour A1, A2 et A3.** La variabilité due à la graine est mesurée sur E2 et E3, et supposée semblable pour les variantes [H]. C'est raisonnable vu sa faiblesse (≤ 0,01 de DSC), mais ce n'est pas vérifié.
2. **A1 ne teste pas une connexion de saut isolée.** Avec `num_res_units=2`, MONAI remplace la convolution unique de chaque niveau par une unité résiduelle à 2 sous-convolutions, et ajoute une unité résiduelle après chaque convolution du décodeur. Pour garder ≈ 18,8 M de paramètres, les canaux passent de 48–832 à 32–504. A1 combine donc trois changements : sauts d'identité, réseau plus profond et canaux plus étroits.
3. **Une seule configuration de GroupNorm (8 groupes)**, sans recherche du nombre de groupes.
4. **Ni Q4 (dropout) ni Q5 (résiduel + GroupNorm).**
5. **Pas de jeu de test séparé.** L'époque est choisie sur le pli de validation, comme en phase 2. Le biais est commun à tous les modèles.
6. **Données :** IRM diagnostiques TotalSegmentator (4 centres, 1,5 et 3 T), et non des IRM de planification MRgRT ; labels en partie semi-automatiques ; pas de test externe.
7. **Environnement :** E2 et E3 graine 42 ont été entraînés avec une version de MONAI non figée ; les autres runs avec MONAI 1.5.2. L'effet graine, qui inclut cette différence, reste minime.
8. **Distances :** les prédictions vides sont exclues ; voir les effectifs dans les CSV.

---

## 9. Fichiers

| Fichier | Contenu |
|---|---|
| `analyse/ablation_stats.py` → `stats_*.csv` | Médianes, Q1, variabilité, Q2–Q3, part de l'écart, sensibilité |
| `analyse/ablation_masks.py` → `ablation_masks.csv` | Fragments, confusion G/D |
| `analyse/ablation_figures.py` → `figures/abl_fig1…5` | Figures PNG (300 dpi) et PDF |
| `runs_plan.csv`, `results/phase3_ablation/<compte>/<job>/` | Plan d'exécution et sorties brutes de chaque run |
