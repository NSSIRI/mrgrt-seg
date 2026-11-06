# Code des figures : article et rapports

Tout le code est dans le dépôt (branche `ablation-18m`). Les scripts **lisent** les résultats et ne modifient aucun fichier de données ou de résultats. Ils tracent les valeurs brutes, sans lissage, interpolation ni sous-échantillonnage.

## 1. Figures de l'article, en anglais (`paper/figures_en/`)

**Script :** `paper/figures_en/make_figures_en.py`
**Commande :** `.venv\Scripts\python.exe paper/figures_en/make_figures_en.py`
**Sorties :** PNG 300 dpi + PDF, dans le même dossier.

| Figure | Fonction | Contenu | Données lues |
|---|---|---|---|
| `fig1_capacity_matched_boxplots` | `fig1()` | DSC et HD95 (échelle log), E1 / E3 / E2, 187 patients | `results/phase2_eval/analyse/combined_long.csv` (grille native) |
| `fig2_paired_differences_vs_E3` | `fig2()` | Différences appariées vs E3 (médiane, IC 95 %), bande de variabilité due à la graine | `phase3_ablation/analyse/stats_q2_variantes.csv`, `stats_q1_graines.csv` |
| `fig3_share_of_gap_closed` | `fig3()` | Part de l'écart E3 → E2 comblée par A1, A2, A3 (IC bootstrap) | `phase3_ablation/analyse/stats_part_ecart.csv` |
| `fig4_seed_reproducibility` | `fig4()` | DSC moyen par pli, graine 42 vs graine 123 (E2, E3) | `combined_long.csv` + `results/phase3_ablation/*/abl18m-*/ablation_eval/metrics_long.csv`, via `ablation_stats.load()` |
| `fig5_fragmentation_laterality` | `fig5()` | Fragments (> 1 % hors de la plus grande composante), confusion G/D > 5 % | `phase3_ablation/analyse/ablation_masks.csv` |
| `fig6_six_oar_boxplots` | `fig6()` | DSC et HD95, U-Net vs SegResNet, 6 OAR, 115 patients | `results/phase3_ablation/*/abl18m-*6-s42-f*/ablation_eval/metrics_long.csv` |
| `fig7_qualitative_overlays` | `fig7()` | Superpositions référence / E3 / E2 : patient au DSC œsophage E3 médian (s0404) et au pire (s0032) | `data/<patient>/{image,label}.nii.gz`, `results/phase2_eval/*/phase2_eval/predictions.zip` |

**Script des figures 8 et 9 :** `paper/figures_en/make_figures_xai_en.py`
**Commande :** `.venv\Scripts\python.exe paper/figures_en/make_figures_xai_en.py`

| Figure | Fonction | Contenu | Données lues |
|---|---|---|---|
| `fig8_thoracic_region_cord_aorta` | `fig8()` | Moelle et aorte : étendue complète vs coupes pulmonaires seules, DSC et HD95 (échelle log), U-Net vs SegResNet, 115 patients | `phase4_6oar/thorax/metrics_thorax_long.csv` |
| `fig9_xai_matched_layers` | `fig9()` | Seg-Grad-CAM, in-organ ratio à 4 niveaux appariés : D187 (E3 vs E2) et D115 (U-Net vs SegResNet) | `results/phase3_ablation/xai_matched/*/*/xai_matched/xai_matched_long.csv` |

## 2. Calculs en amont des figures

Les CSV ci-dessus sont produits par les scripts suivants, à lancer dans cet ordre si on veut tout refaire :

| Script | Produit | Remarque |
|---|---|---|
| `phase3_ablation/analyse/ablation_stats.py` | `stats_*.csv` (Wilcoxon, Holm, bootstrap 10 000 tirages, graine fixée 20261106) | Utilisé par les figures 2, 3 et 4 |
| `phase3_ablation/analyse/ablation_masks.py` | `ablation_masks.csv` (composantes connexes en connexité 26, confusion G/D) | Long : 187 patients × 7 modèles. Lancer avec 3 processus (`... ablation_masks.py 3`) pour limiter la mémoire |
| `phase4_6oar/analyse/phase4_stats.py` | `p4_*.csv` (+ figures de la phase 4 en français) | |
| `phase4_6oar/thorax/metrics_thorax.py` puis `stats_thorax.py` | `metrics_thorax_long.csv`, `stats_thorax.csv`, `couverture_thorax.csv` | Utilisé par la figure 8 et le tableau 7. Contrôle : sans restriction = `metrics_long.csv` |
| `phase3_ablation/xai_matched/analyse/analyse_xai_matched.py`, `analyse_xai6.py` | `xai_stats_apparies.csv`, `xai6_stats_apparies.csv` | Utilisé par les tableaux 8 et 9 |
| `results/analyse_complete/analyse_complete.py` | Tableaux et figures de la phase 2 (en français) | |
| `paper/build_article_ablation.py` | `paper/Article_Ablation_6OAR_v1.md` (tableaux générés depuis les CSV) | Le texte vient de `paper/Article_Ablation_6OAR_template.md` |
| `paper/md2docx.js` | `paper/Article_Ablation_6OAR_v1.docx` (figures en anglais intégrées) | Node.js + `npm install docx@9.5.1` |

## 3. Figures des rapports, en français

| Script | Dossier de sortie |
|---|---|
| `results/analyse_complete/analyse_complete.py` | `results/analyse_complete/figures/` (fig01 à fig07) |
| `phase3_ablation/analyse/ablation_figures.py` | `phase3_ablation/figures/` (abl_fig1 à abl_fig5) |
| `phase4_6oar/analyse/phase4_stats.py` | `phase4_6oar/figures/` (p4_fig1, p4_fig2) |
| `results/phase4_organes/controle_visuel.py` | `results/phase4_organes/controle_visuel_faible_couverture.png` |

## 4. Couleurs (palette du projet)

| Modèle | Couleur |
|---|---|
| E1 | bleu `#2a78d6` |
| E2 / SegResNet | orange `#eb6834` |
| E3 / U-Net | aqua `#1baf7a` |
| Variantes A1–A3 | violet `#6250d6` |

- Ce trio (aqua / violet / orange) passe toutes les vérifications de `validate_palette.js`, y compris pour le daltonisme, en comparant toutes les paires.
- Les variantes sont identifiées par leur étiquette, jamais par la couleur seule.
- Fond `#fcfcfb` ; texte `#0b0b0b` / `#52514e`.

## 5. Environnement utilisé

Python 3.11.9 ; matplotlib 3.10.9 ; numpy 2.4.3 ; pandas 3.0.3 ; scipy 1.17.1 ; nibabel 5.4.2. Environnement virtuel `.venv` du projet.
