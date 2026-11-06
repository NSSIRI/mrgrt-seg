# Phase 4 : segmentation de 6 OAR thoraciques (+ moelle épinière, aorte), SegResNet et U-Net à 18,8 M

*Rapport du 06/11/2026. Les valeurs viennent de `phase4_6oar/analyse/p4_*.csv` (`phase4_stats.py`). Les figures sont dans `phase4_6oar/figures/`. Aucun fichier existant n'a été modifié.*

**Légende :**
- **[O]** résultat observé ;
- **[L]** résultat de la littérature ;
- **[H]** interprétation possible, non démontrée.

---

## 0. Synthèse

- **SegResNet surpasse le U-Net à capacité égale pour les 6 OAR [O].**
  - **DSC** : de +0,029 (poumons) à **+0,260 (œsophage)**, avec +0,077 pour la moelle et +0,082 pour l'aorte.
  - Toutes les métriques vont dans le même sens : p_Holm ≤ 6e-8 et r de rang bisérial ≥ 0,60.
- **Les deux nouveaux organes sont bien segmentés par SegResNet [O].**
  - Moelle : DSC médian **0,881**, HD95 **2,6 mm**, Surface-DSC 2 mm **0,92**.
  - Aorte : DSC médian **0,897**, HD95 **4,4 mm**, Surface-DSC 2 mm **0,89**.
- **Les fragments lointains du U-Net s'aggravent avec 6 classes [O].**
  - HDmax médian des poumons : 140 à 166 mm pour le U-Net, contre 20 mm pour SegResNet.
  - Moelle : 45 mm contre 7 mm ; aorte : 69 mm contre 16 mm.

---

## 1. Protocole

- **Cohorte D115** : patients de D187 dont la moelle couvre au moins 95 % de la hauteur pulmonaire et l'aorte au moins 95 % des 75 % inférieurs (`results/phase4_organes/`). Un contrôle visuel des cas exclus montre des annotations incomplètes ou des acquisitions atypiques.
- **Labels** :
  - classes 1 à 4 identiques à D187 (vérifié) ;
  - 5 = moelle et 6 = aorte, prises dans TotalSegmentator MRI v2.0 et ajoutées seulement sur le fond (`data_6oar/`).
- **Plis** : KFold(5, graine 42) sur les 115 patients, soit 23 patients par pli.
- **Entraînement** :
  - même protocole que les phases 2 et 3 (DiceCE, AdamW, 150 époques, mêmes augmentations), avec 7 sorties ;
  - code au commit `65a63af`, torch 2.10.0+cu128, MONAI 1.5.2, GPU T4 ;
  - SegResNet : 18 796 167 paramètres ; U-Net : 18 808 159.
- **Évaluation** : même script que les phases 2 et 3, sur la grille native, 115 patients pour chaque modèle et chaque OAR.
- **Garde-fou** : le premier job a validé la chaîne complète avant le lancement des 9 autres.

---

## 2. Résultats [O]

**Médiane [Q1–Q3] (`p4_descriptifs.csv`) :**

| OAR | DSC U-Net | DSC SegResNet | HD95 U-Net (mm) | HD95 SegResNet (mm) | Surface-DSC 2 mm U-Net / SegResNet |
|---|---|---|---|---|---|
| Poumon G | 0,908 [0,796–0,939] | **0,944 [0,895–0,961]** | 15,5 | **4,9** | 0,69 / **0,82** |
| Poumon D | 0,919 [0,839–0,952] | **0,951 [0,915–0,970]** | 9,8 | **4,3** | 0,76 / **0,87** |
| Cœur | 0,841 [0,720–0,888] | **0,896 [0,821–0,931]** | 13,2 | **10,0** | 0,48 / **0,65** |
| Œsophage | 0,404 [0,285–0,501] | **0,678 [0,572–0,764]** | 15,4 | **7,9** | 0,46 / **0,68** |
| **Moelle** | 0,794 [0,729–0,846] | **0,881 [0,836–0,905]** | 7,7 | **2,6** | 0,76 / **0,92** |
| **Aorte** | 0,790 [0,702–0,859] | **0,897 [0,847–0,926]** | 14,2 | **4,4** | 0,66 / **0,89** |

**Différence appariée SegResNet − U-Net (médiane [IC 95 % bootstrap], `p4_comparaisons.csv`) :**

| OAR | DSC | % patients meilleurs avec SegResNet | HD95 (mm) | r (DSC) |
|---|---|---|---|---|
| Poumon G | +0,030 [0,026 ; 0,040] | 92 % | −3,3 [−6,3 ; −2,6] | 0,87 |
| Poumon D | +0,029 [0,020 ; 0,033] | 93 % | −3,6 [−5,6 ; −2,3] | 0,93 |
| Cœur | +0,051 [0,043 ; 0,057] | 92 % | −3,1 [−4,2 ; −1,7] | 0,92 |
| Œsophage | **+0,260 [0,234 ; 0,284]** | 98 % | −5,8 [−8,6 ; −4,3] | 1,00 |
| Moelle | +0,077 [0,061 ; 0,090] | 99 % | −2,3 [−4,3 ; −1,6] | 0,97 |
| Aorte | +0,082 [0,070 ; 0,106] | 98 % | −4,6 [−7,9 ; −3,4] | 0,99 |

**Tests :** Wilcoxon apparié bilatéral, correction de Holm sur les 6 OAR pour chaque métrique. p_Holm maximal : DSC 6e-16, HD95 6e-8, ASSD 6e-12, Surface-DSC 2e-18, HDmax 3e-10.

**Stabilité entre plis (`p4_dsc_par_pli.csv`) :** SegResNet dépasse le U-Net dans les **30 couples pli × OAR sur 30**.

**Prédictions vides :** U-Net 2 pour le cœur et 1 pour l'œsophage ; SegResNet 1 pour l'œsophage.

Figures `p4_fig1_boxplots` et `p4_fig2_differences`.

---

## 3. Comparaison avec les phases 2 et 3

| DSC médian | D187, 4 OAR (phase 2) | D115, 6 OAR (phase 4) |
|---|---|---|
| SegResNet, œsophage | 0,679 | 0,678 |
| U-Net, œsophage | 0,502 | 0,404 |
| SegResNet, cœur | 0,886 | 0,896 |
| U-Net, cœur | 0,825 | 0,841 |

**Ces chiffres ne se comparent pas directement :** les patients sont différents, il y a environ 92 patients d'entraînement par pli au lieu de 149, et 2 classes de plus. **[O]** SegResNet garde le même niveau pour l'œsophage, alors que le U-Net perd environ 0,10.

---

## 4. Littérature [L]

- **Moelle et aorte en IRM thoracique :** il n'y a pas de valeur de référence directement comparable dans les sources déjà réunies. Ribeiro et al. 2023 (MRIdian 0,35 T) porte sur les poumons, le cœur et l'œsophage. Une recherche ciblée reste à faire avant toute comparaison chiffrée.

---

## 5. Interprétation possible [H]

1. **L'avantage de SegResNet se généralise à la moelle et à l'aorte**, ce qui va dans le sens de l'ablation : les blocs résiduels réduisent les fragments lointains, qui sont la principale faiblesse du U-Net plain.
2. **Le U-Net plain semble plus sensible à la réduction du nombre de patients d'entraînement et à l'ajout de classes** (œsophage −0,10). Une meilleure efficacité de SegResNet vis-à-vis des données est une hypothèse plausible. Elle n'est pas testée, car la cohorte et les classes changent en même temps.
3. **La moelle**, petite et tubulaire mais très contrastée en IRM, est bien segmentée par SegResNet. Le HD95 médian de 2,6 mm est à rapprocher des marges de planification, ce qui reste à discuter cliniquement.

---

## 6. Limites

1. **Une seule graine**, et pas d'ablation sur 6 OAR.
2. **Labels de moelle et d'aorte issus de TotalSegmentator** : en partie semi-automatiques, ce ne sont pas des contours de radiothérapie validés. Ils incluent toute la partie visible, parfois cervicale ou abdominale.
3. **Exclusion des patients dont l'annotation est incomplète** : cela donne une cohorte « propre », ce qui peut surestimer les performances en conditions réelles.
4. **Pas de jeu de test séparé** (le choix de l'époque se fait sur la validation), **pas de test externe**, et des **IRM diagnostiques** plutôt que des IRM de planification MRgRT.
5. **Pas d'analyse de fragments ni de latéralité au niveau des masques** pour la phase 4. Le HDmax en donne une idée indirecte.
