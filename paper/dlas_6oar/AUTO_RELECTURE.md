# Auto-relecture : manuscrit DLAS 6 OAR (06/11/2026)

**Fichiers :**
- `Manuscript_DLAS_6OAR.docx` et `.md` : le manuscrit ;
- `References_DLAS_6OAR.md` : les références ;
- `figures/Fig2–Fig4` : figures PNG 300 dpi ;
- `tables/*.csv` : les tableaux ;
- `analysis_dlas.py` et `build_manuscript.py` : les scripts.

## 1. Cohérence entre les chiffres du texte et les tableaux

**Vérifications automatiques.**
- Les tableaux T1, T2, T4, T5 (U-Net) et T6 sont générés directement à partir des CSV calculés depuis les fichiers bruts (`metrics_long.csv` par patient, `meta.csv`, labels, historiques, journaux).
- Les métriques sont recalculées depuis les CSV par patient (grille native). Elles sont identiques aux résultats de phase 4 et au recalcul thoracique, qui avait été contrôlé avec un écart de 6e-14.

**Chiffres écrits à la main dans le texte, recontrôlés un par un contre les CSV :**
- [x] **DSC et HD95 médians des 6 OAR** (résumé, sections 3.2 et 5) : correspondent à T4.
- [x] **Région thoracique** : DSC 0,882 / 0,903, HD95 2,5 / 3,2 mm. La part de la référence dans le thorax vaut 38 % et 57 % (`couverture_thorax.csv`).
- [x] **Entraînement** (`training_stats.csv`) :
  - meilleure époque 144 (131–149) ;
  - DSC de validation 0,838 ;
  - 97 s par époque, environ 4,0 h par pli ;
  - inférence 4,4 s / 0,9 s.
- [x] **Échecs** (`outliers_SegResNet.csv`, `outliers_association.csv`, plus le calcul complémentaire du 06/11) :
  - 31/115 patients ;
  - par organe : 9 / 10 / 11 / 16 / 5 / 8 ;
  - 8 patients en échec à la fois pour l'œsophage et un autre OAR ;
  - 1 prédiction vide ;
  - séquences hors écho de gradient : 13/25 contre 10/90 en écho de gradient (Fisher p = 3,6e-5) ;
  - volumes œsophagiens 4,8 contre 14,3 mL (Mann–Whitney p = 1,4e-4) ;
  - 3 examens du rachis sur 3 en échec.
- [x] **Écart de volume** : ±2,5 % pour tous les OAR, sauf l'œsophage (+7,9 %).
- [x] **Plages par pli** : correspondent à `perfold_mean_dsc_SegResNet.csv`.
- [x] **Écart avec Ribeiro** : 0,003–0,044. J'ai corrigé une première formulation (« 0,01–0,04 »).
- [x] **Données manquantes** : âge et sexe manquent chacun pour 33/115 patients (29 %). J'ai corrigé une première formulation erronée.
- [ ] **À refaire après toute modification des données** : relancer `analysis_dlas.py`, puis `build_manuscript.py`, puis relire les chiffres écrits à la main.

## 2. Références
- **39 références citées**, numérotées automatiquement dans l'ordre de première citation (style Vancouver).
- **37 sont vérifiées sur PubMed** (titre, auteurs, revue, volume, DOI). L'année retenue est celle de la publication PubMed, soit la date de mise en ligne : pour certains articles, elle diffère de l'année du volume. Exemples : Lustberg 2017 (vol. 126 de 2018), Isensee 2020 (vol. 18 de 2021). **Choisir une convention selon la revue visée.**
- **2 sont à vérifier** (hors PubMed) :
  - Myronenko 2018 (SegResNet) : titre, auteur et année confirmés par Consensus, mais DOI et pages non confirmés ;
  - Cardoso 2022 (MONAI, arXiv).
- Le prompt demandait 30 à 40 références : la contrainte est respectée.
- **Valeurs du tableau 7** : elles sont tirées des résumés seulement. Les cases [À COMPLÉTER] correspondent à des valeurs absentes du résumé (n de Yang, Francis, Summerfield). Il faut lire les articles complets.

## 3. Points faibles qu'un relecteur soulèvera
1. **Ce ne sont pas des images de MR-Linac.**
   - Le prompt évoquait « Elekta Unity 1,5 T », mais les données sont des IRM diagnostiques (TotalSegmentator MRI).
   - Le titre et le résumé n'affirment rien sur Unity. Il faut garder cette prudence partout, y compris dans la lettre à l'éditeur.
   - Une revue de radiothérapie demandera une validation sur MR-Linac.
2. **Vérité terrain.**
   - Les labels de référence viennent du jeu public. Leur mode de production (semi-automatique, avec corrections) doit être décrit précisément d'après l'article TotalSegmentator MRI.
   - Il y a un risque de biais si les labels ont été produits avec l'aide d'un réseau.
   - Un label suspect a été repéré : œsophage de 100,9 mL.
3. **Validation.**
   - Il n'y a pas de jeu de test séparé.
   - Le meilleur checkpoint est choisi sur le pli qui sert aussi à rapporter les résultats, ce qui donne un biais optimiste. Cette limite est déclarée.
   - Une réponse possible : utiliser la partition officielle « test » de TotalSegmentator (13 patients de D115) comme test indépendant, en réentraînant sans eux.
4. **Doublon avec l'article d'ablation (`paper/Article_Ablation_6OAR_v1`).** Mêmes données, mêmes modèles 6 OAR, mêmes résultats U-Net / SegResNet. **Risque de publication redondante.** Il faut :
   - soit fusionner les deux articles ;
   - soit n'en soumettre qu'un ;
   - soit, au minimum, citer l'autre et le déclarer à l'éditeur.
5. **Pas de grading clinique ni de dosimétrie.** Ces sections sont marquées [À COMPLÉTER]. Or le prompt et les revues visées (*Radiation Oncology*, *phiRO*) les attendent.
6. **Seuils d'échec.**
   - DSC < 0,7, et < 0,5 pour l'œsophage : choisis avant l'analyse des échecs, mais après avoir vu la médiane de l'œsophage.
   - Un relecteur peut juger le seuil de l'œsophage arbitraire.
   - Proposer une analyse de sensibilité, par exemple avec un seuil de 0,6.
7. **Causes d'échec.**
   - Elles sont déduites automatiquement (règles sur les composantes connexes, le volume, la séquence) et n'ont pas toutes été revues visuellement.
   - Le critère « coupes épaisses » utilise l'espacement du 3ᵉ axe du tableau d'image, qui n'est pas toujours l'axe de coupe (acquisitions sagittales ou coronales).
8. **Pas de comparaison avec nnU-Net** dans cette cohorte : c'est la référence attendue par les relecteurs.
9. **Une seule graine d'entraînement et aucun réglage d'hyperparamètres** (tableau 3 : « not tuned »).
10. **Rotations de 90° dans le plan** utilisées en augmentation : sans effet miroir, donc sans risque pour la latéralité, mais à justifier.
11. **Figure 1** (schéma de l'étude) : non produite.
12. **Environnement** : Kaggle (T4), et non le HPC MARWAN. La section « Network implementation » le dit correctement.
13. **Équation de la perte** : elle est écrite en texte linéaire. Pour la soumission, il faut la convertir en équation Word.

## 4. Reste à faire avant soumission
- [ ] Toutes les cases [À COMPLÉTER] : éthique, financement, contributions, grading, valeurs n du tableau 7, Figure 1.
- [ ] Vérifier les DOI de Myronenko et MONAI, et la convention d'année des références.
- [ ] Décider du rapport avec l'article d'ablation (point 3.4).
- [ ] Adapter le format à la revue : limite de mots (le manuscrit fait environ 6 100 mots avec les tableaux), nombre de tableaux et de figures, résumé de 300 mots maximum (environ 268 actuellement).
