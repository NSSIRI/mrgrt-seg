# Phase 4 : métriques 6 OAR limitées à la région thoracique

Date : 06/11/2026. Cohorte D115 (115 patients, 5 plis, graine 42). Grille native. Masques prédits existants (`predictions.zip`) : aucun réentraînement, aucun calcul GPU.

**Scripts :**
- `metrics_thorax.py` : recalcul ;
- `stats_thorax.py` : statistiques et figure.

**Sorties :**
- `metrics_thorax_long.csv`, `stats_thorax.csv`, `couverture_thorax.csv` ;
- `controle_sans_restriction.csv` ;
- `fig_thorax.png` et `fig_thorax.pdf`.

## Pourquoi

`build_labels_6oar.py` ajoute la moelle (5) et l'aorte (6) avec toute leur étendue TotalSegmentator. Le critère « ≥ 95 % des coupes pulmonaires » sert à **sélectionner les patients**, mais **ne coupe pas les labels**. Les métriques publiées pour ces deux organes incluaient donc des portions abdominales et parfois cervicales.

## Méthode

- **Région thoracique** : les coupes, selon l'axe le plus proche de l'axe S, qui contiennent du poumon G ou D dans la référence.
- **Recouvrement** (DSC, IoU, précision, sensibilité, volumes) : voxels de la région seulement.
- **Surfaces** (HD95, HDmax, ASSD, Surface-DSC 2 mm) : les bords sont extraits des masques **entiers**, puis seuls les points situés dans la région sont mesurés, par rapport au bord entier de l'autre masque. On évite ainsi les faces de coupe artificielles, identiques dans la prédiction et dans la référence, qui amélioreraient faussement les distances.
- **Fonctions** reprises à l'identique de `phase3_ablation/eval_ablation.py`.
- **Statistiques** : SegResNet − U-Net, Wilcoxon apparié, correction de Holm sur les 6 OAR par métrique et par région, IC 95 % bootstrap.

**Contrôle :** sans restriction, le recalcul retrouve `metrics_long.csv`. L'écart maximal est de 6e-14 sur 1 380 lignes, et la ré-implémentation des surfaces est identique à `surface_stats` (écart 0).

## Résultats observés

**Part de la référence située dans la région thoracique** (médiane, Q1, minimum) :

| OAR | Médiane | Q1 | Min | Patients avec une partie hors thorax |
|---|---|---|---|---|
| Moelle | 0,38 | 0,27 | 0,07 | 98 % |
| Aorte | 0,57 | 0,41 | 0,07 | 97 % |
| Œsophage | 1,00 | 1,00 | 0,38 | 14 % |
| Cœur | 1,00 | 1,00 | 0,96 | 4 % |

En médiane, seuls 38 % du volume de moelle et 57 % du volume d'aorte étaient réellement thoraciques.

**Moelle et aorte, médianes (étendue complète → région thoracique) :**

| OAR | Modèle | DSC | HD95 (mm) | Surface-DSC 2 mm |
|---|---|---|---|---|
| Moelle | SegResNet | 0,881 → 0,882 | 2,64 → 2,50 | 0,922 → 0,926 |
| Moelle | U-Net | 0,794 → 0,775 | 7,69 → 4,72 | 0,758 → 0,741 |
| Aorte | SegResNet | 0,897 → 0,903 | 4,42 → 3,23 | 0,886 → 0,897 |
| Aorte | U-Net | 0,790 → 0,790 | 14,18 → 7,91 | 0,664 → 0,635 |

**Différence SegResNet − U-Net dans la région thoracique :**

| OAR | ΔDSC [IC 95 %] | ΔHD95 (mm) [IC 95 %] | Patients où SegResNet a le meilleur DSC | p_Holm |
|---|---|---|---|---|
| Moelle | +0,094 [0,075 ; 0,113] | −1,85 [−2,81 ; −1,47] | 99 % | < 0,001 |
| Aorte | +0,089 [0,074 ; 0,113] | −2,65 [−3,71 ; −2,03] | 97 % | < 0,001 |

Pour les 4 OAR historiques, le DSC change de moins de 0,006 entre les deux régions. Les écarts entre modèles sont inchangés en sens et en significativité (`stats_thorax.csv`).

## Interprétation possible

- Les valeurs de SegResNet pour la moelle et l'aorte (DSC 0,88 et 0,90) **tiennent** quand on se limite au thorax.
- L'avantage de SegResNet est même un peu plus grand pour le DSC (+0,094 au lieu de +0,077 pour la moelle).
- Les HD95 de l'U-Net baissent nettement dans la région thoracique (moelle : 7,7 → 4,7 mm ; aorte : 14,2 → 7,9 mm). Une partie de ses grandes erreurs de surface se trouvait donc hors du thorax.

## Limites

- **Les erreurs hors de la région sont ignorées.** Une fausse détection dans l'abdomen ne compte plus, ce qui explique que le HD95 des poumons de l'U-Net passe de 15,5 à 12,0 mm alors que la référence des poumons est entièrement thoracique. Pour les 4 OAR historiques, il faut garder les métriques sur l'étendue complète.
- **La région est définie à partir de la référence** (les coupes pulmonaires), comme on le ferait en clinique avec le volume de planification.
- **La limite tombe à une coupe près :** l'épaisseur native varie (jusqu'à 4,4 mm chez certains patients).
- **Les modèles ont été entraînés sur les labels complets.** Seule l'évaluation est restreinte.
